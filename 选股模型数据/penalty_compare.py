"""惩罚权重对比: ×2 vs ×3 vs ×5"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df["usable"]].copy()
u["label"] = u["forward_return"].apply(lambda r: 0 if r < 0 else (1 if r <= 0.05 else 2))

feat = [c for c in u.columns if c not in
    ["code","year","quarter","forward_return","label","usable","industry"]]
ind = pd.get_dummies(u["industry"], prefix="ind").astype(float)
X_all = pd.concat([u[feat], ind], axis=1)
X_all = X_all.fillna(X_all.median())
y_all = u["label"].values; years = u["year"].values

class MLP3(nn.Module):
    def __init__(self, n_in, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h // 2, 3)
        )
    def forward(self, x): return self.net(x)

def train_mlp(Xtr_np, ytr_np, Xte_np, w0):
    sc = StandardScaler()
    Xtr = torch.FloatTensor(sc.fit_transform(Xtr_np))
    Xte = torch.FloatTensor(sc.transform(Xte_np))
    ytr = torch.LongTensor(ytr_np)
    m = MLP3(Xtr.shape[1], 128)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([w0, 1.0, 1.0]))
    best_loss, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad()
        loss = loss_fn(m(Xtr), ytr); loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        return torch.argmax(m(Xte), dim=1).numpy()

test_years = [2020, 2021, 2022, 2023, 2024]
hist = {}

for w0, desc in [(2, "×2"), (3, "×3"), (5, "×5")]:
    print(f"\n{'='*50}")
    print(f"惩罚={desc}")
    print(f"{'年份':<6s} {'准确率':>6s} {'降召回':>6s} {'大涨召回':>6s} {'降误判涨':>8s}")
    data = []
    for ty in test_years:
        tr = (years != ty); te = (years == ty)
        pred = train_mlp(X_all[tr].values, y_all[tr], X_all[te].values, w0)
        cm = confusion_matrix(y_all[te], pred)
        acc = (cm[0,0]+cm[1,1]+cm[2,2])/cm.sum()*100
        r0 = cm[0,0]/cm[0].sum()*100 if cm[0].sum()>0 else 0
        r2 = cm[2,2]/cm[2].sum()*100 if cm.shape[0]>2 else 0
        dfp = (cm[0,1]+cm[0,2])/cm[0].sum()*100
        print(f"{ty:<6d} {acc:>5.1f}% {r0:>5.1f}% {r2:>5.1f}% {dfp:>7.1f}%")
        data.append({"acc": acc, "r0": r0, "r2": r2, "dfp": dfp, "cm": cm})
    hist[desc] = {"data": data, "avg_acc": np.mean([d["acc"] for d in data]),
                   "avg_r0": np.mean([d["r0"] for d in data]),
                   "avg_r2": np.mean([d["r2"] for d in data]),
                   "avg_dfp": np.mean([d["dfp"] for d in data])}

# 汇总
print(f"\n{'='*50}")
print(f"汇总对比")
print(f"{'惩罚':<8s} {'准确率':>7s} {'降召回':>7s} {'大涨召回':>7s} {'降误判涨':>8s}")
for desc in ["×2", "×3", "×5"]:
    h = hist[desc]
    print(f"{desc:<8s} {h['avg_acc']:>6.1f}% {h['avg_r0']:>6.1f}% {h['avg_r2']:>6.1f}% {h['avg_dfp']:>7.1f}%")

# 打印全量混淆矩阵
for desc in ["×2", "×3", "×5"]:
    cms = [d["cm"] for d in hist[desc]["data"]]
    cm_all = sum(cms)
    n = cm_all.sum()
    acc = (cm_all[0,0]+cm_all[1,1]+cm_all[2,2])/n*100
    dfp = (cm_all[0,1]+cm_all[0,2])/cm_all[0].sum()*100 if cm_all[0].sum()>0 else 0
    r0 = cm_all[0,0]/cm_all[0].sum()*100 if cm_all[0].sum()>0 else 0
    r2 = cm_all[2,2]/cm_all[2].sum()*100 if cm_all.shape[0]>2 else 0
    print(f"\n{'='*50}")
    print(f"惩罚{desc} 五年合并混淆矩阵 (总共{n}条)")
    print(f"准确率={acc:.1f}%  下降召回={r0:.1f}%  大涨召回={r2:.1f}%  下降误判涨={dfp:.1f}%")
    print(f"")
    print(f"          预测:降   微涨   大涨  |  合计")
    print(f"实际下降{cm_all[0,0]:>7d}{cm_all[0,1]:>6d}{cm_all[0,2]:>7d}  |{cm_all[0].sum():>6d}")
    print(f"实际微涨{cm_all[1,0]:>7d}{cm_all[1,1]:>6d}{cm_all[1,2]:>7d}  |{cm_all[1].sum():>6d}")
    print(f"实际大涨{cm_all[2,0]:>7d}{cm_all[2,1]:>6d}{cm_all[2,2]:>7d}  |{cm_all[2].sum():>6d}")
