"""FULL共识: MLP∩RF共同买入"""
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_full.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
X = pd.concat([u[feat]], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8)

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def lp(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

pr_mlp = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); pr_mlp[te] = lp(X[te].values, os.path.join(MDIR, f"FULL_{ty}.pt"))

print("训练 RF...")
pr_rf = np.zeros(len(u), dtype=int)
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    rf = RandomForestClassifier(100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(X[tr].values, y[tr]); pickle.dump(rf, open(os.path.join(MDIR, f"FULL_RF_{ty}.pkl"), "wb"))
    pr_rf[te] = rf.predict(X[te].values)
    print(f"  fold {ty} done")

# 共同买入
both = (pr_mlp == 4) & (pr_rf == 4) & MASK
nb = both.sum(); br = rets[both]
print(f"\n=== MLP∩RF 共同买入: {nb}只 均{br.mean()*100:.1f}% ===")
yrly = []
for ty in TY:
    b = (yrs == ty) & both; n = b.sum(); a = rets[b]
    yl = a.mean()*100 if n > 0 else 0; yrly.append(yl)
    print(f"  {ty}: {n}只 均{a.mean()*100:.1f}%" if n > 0 else f"  {ty}: 未买")
print(f"年均: {np.mean(yrly):.1f}%")

# 行业
print(f"\n行业分布:")
for ind in sorted(u.loc[both, "industry"].unique()):
    n = sum(both & (u.industry == ind)); r = rets[both & (u.industry == ind)]
    print(f"  {ind}: {n}只 均{r.mean()*100:.1f}%")
