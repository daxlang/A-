"""仅S2排除训练 5折MLP"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
S2 = ~((u.cfo_to_profit < 0) & (u.net_margin > 0))
print(f"仅S2排除: 排{(~S2).sum()}条 剩{S2.sum()}条")

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def train_save(Xtr, ytr, path):
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(Xtr)); ys = torch.LongTensor(ytr)
    m = M5(Xs.shape[1]); opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]))
    bl, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval(); torch.save({"model":bs,"scaler":sc}, path)

def predict(Xte, path):
    ckpt = torch.load(path, weights_only=False); m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

for ty in TY:
    tr = (yrs != ty) & S2; te = (yrs == ty) & S2
    train_save(X[tr].values, y[tr], os.path.join(MDIR, f"S2_only_{ty}.pt"))
    print(f"  fold {ty}: 训{tr.sum()} 测{te.sum()} done")

pr_s2 = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty) & S2; pr_s2[te] = predict(X[te].values, os.path.join(MDIR, f"S2_only_{ty}.pt"))

S1 = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5)
pr_old = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); pr_old[te] = predict(X[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))

for name, p, m in [("MLP0_5+S1预筛", pr_old, S1), ("S2_only", pr_s2, S2)]:
    cm = confusion_matrix(y[m], p[m]); n = cm.sum(); acc = sum(cm[i,i] for i in range(5)) / n * 100
    print(f"\n=== {name} ({n}条 acc={acc:.1f}%) ===")
    print(f"实际↓预测 |暴跌  跌 小涨 中涨 暴涨| 合计")
    for i, lb in enumerate(LAB):
        print(f"  {lb:6s}|{cm[i,0]:>4d} {cm[i,1]:>3d} {cm[i,2]:>4d} {cm[i,3]:>4d} {cm[i,4]:>4d}|{cm[i].sum():>5d}")
    buy = (p == 4) & m; nb = buy.sum(); br = rets[buy]
    yrly = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
    print(f"买入: {nb}只 均{br.mean()*100:+.1f}% 年{np.mean(yrly):+.1f}%")
