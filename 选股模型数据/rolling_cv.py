"""滚动年交叉验证: 每年轮流做测试集, 三分类"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df["usable"]].copy()

# 三分类
def cls(ret):
    if ret < 0: return 0
    if ret <= 0.05: return 1
    return 2
u["label"] = u["forward_return"].apply(cls)

# 特征
feat = [c for c in u.columns if c not in
    ["code","year","quarter","forward_return","label","usable","industry"]]
ind = pd.get_dummies(u["industry"], prefix="ind").astype(float)
X_all = pd.concat([u[feat], ind], axis=1)
X_all = X_all.fillna(X_all.median())
y_all = u["label"].values
years = u["year"].values

# 权重
cw = {0: 2.0, 1: 1.0, 2: 1.0}
class MLP3(nn.Module):
    def __init__(self, n_in, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h//2, 3)
        )
    def forward(self, x): return self.net(x)

def train_mlp(Xtr_np, ytr_np, Xte_np, h=128, use_weights=True):
    sc = StandardScaler()
    Xtr = torch.FloatTensor(sc.fit_transform(Xtr_np))
    Xte = torch.FloatTensor(sc.transform(Xte_np))
    ytr = torch.LongTensor(ytr_np)
    w = torch.FloatTensor([2.0, 1.0, 1.0]) if use_weights else None
    m = MLP3(Xtr.shape[1], h)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    best_loss, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad()
        loss = loss_fn(m(Xtr), ytr)
        loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        return torch.argmax(m(Xte), dim=1).numpy()

test_years = [2020, 2021, 2022, 2023, 2024]

for label, use_weights in [("有惩罚(下降×2)", True), ("无惩罚(平等)", False)]:
    results = []
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'年份':<6s} {'准确率':>7s} {'降召回':>7s} {'微召回':>7s} {'大涨召回':>6s} {'降误判涨':>8s}")
    print("-" * 50)
    
    for ty in test_years:
        tr = (years != ty); te = (years == ty)
        Xtr_np, ytr_np = X_all[tr].values, y_all[tr]
        Xte_np, yte_np = X_all[te].values, y_all[te]
        
        pred = train_mlp(Xtr_np, ytr_np, Xte_np, use_weights=use_weights)
        cm = confusion_matrix(yte_np, pred)
        acc = np.mean(pred == yte_np)
        r0 = cm[0,0]/cm[0].sum()*100 if cm[0].sum()>0 else 0
        r1 = cm[1,1]/cm[1].sum()*100 if cm.shape[0]>1 and cm[1].sum()>0 else 0
        r2 = cm[2,2]/cm[2].sum()*100 if cm.shape[0]>2 and cm[2].sum()>0 else 0
        down_fp = (cm[0,1]+cm[0,2])/cm[0].sum()*100 if cm[0].sum()>0 else 0
        print(f"{ty:<6d} {acc*100:>6.1f}% {r0:>6.1f}% {r1:>6.1f}% {r2:>5.1f}% {down_fp:>7.1f}%")
        results.append({"acc":acc,"r0":r0,"r2":r2,"down_fp":down_fp,"cm":cm})
    
    avg_acc = np.mean([r["acc"] for r in results])
    avg_r0 = np.mean([r["r0"] for r in results])
    avg_r2 = np.mean([r["r2"] for r in results])
    avg_fp = np.mean([r["down_fp"] for r in results])
    cm_all = sum([r["cm"] for r in results])
    total = cm_all.sum()
    acc_all = (cm_all[0,0]+cm_all[1,1]+cm_all[2,2])/total
    
    print(f"平均: 准确率={avg_acc*100:.1f}% 降召回={avg_r0:.1f}% 大涨召回={avg_r2:.1f}% 降误判涨={avg_fp:.1f}%")
    print(f"合并混淆:")
    for i, lb in enumerate(["下降","微涨","大涨"]):
        print(f"  {lb}: 降={cm_all[i,0]} 微={cm_all[i,1]} 涨={cm_all[i,2]}")
