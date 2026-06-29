"""LR筛暴跌 → MLP/RF 两阶段"""
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1)

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def load_mlp(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

# LR全量预测
pr_lr = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty)
    lr = pickle.load(open(os.path.join(MDIR, f"LR_{ty}.pkl"), "rb"))
    pr_lr[te] = lr.predict(X[te].values)

n_screened = sum(MASK & (pr_lr == 0))
print(f"LR筛掉暴跌: {n_screened}/{sum(MASK)}只")

# LR筛后 → MLP
pr_mlp = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); keep = (pr_lr != 0) & MASK; te_k = te & keep
    if te_k.sum() > 0:
        pr_mlp[te_k] = load_mlp(X[te_k].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))

# LR筛后 → RF
pr_rf = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); keep = (pr_lr != 0) & MASK; te_k = te & keep
    if te_k.sum() > 0:
        rf = pickle.load(open(os.path.join(MDIR, f"RF_{ty}.pkl"), "rb"))
        pr_rf[te_k] = rf.predict(X[te_k].values)

EVAL = MASK & (pr_lr != 0)
for name, p in [("LR筛+MLP", pr_mlp), ("LR筛+RF", pr_rf)]:
    cm = confusion_matrix(y[EVAL], p[EVAL]); n = cm.sum()
    acc = sum(cm[i,i] for i in range(5)) / n * 100
    print(f"\n=== {name} ({n}条 acc={acc:.1f}%) ===")
    print(f"实际↓预测 |暴跌  跌 小涨 中涨 暴涨| 合计")
    for i, lb in enumerate(LAB):
        print(f"  {lb:6s}|{cm[i,0]:>4d} {cm[i,1]:>3d} {cm[i,2]:>4d} {cm[i,3]:>4d} {cm[i,4]:>4d}|{cm[i].sum():>5d}")
    buy = (p == 4) & EVAL; nb = buy.sum(); br = rets[buy]
    yr = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
    print(f"买入: {nb}只 均{br.mean()*100:+.1f}% 年{np.mean(yr):+.1f}%")

print(f"\n基准 MLP0_5→S1∩S3: 2101条 买231 年+21.8%")
