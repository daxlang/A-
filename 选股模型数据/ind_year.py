"""ALL_14逐年行业明细"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
df = pd.read_csv(os.path.join(BASE, "training_all.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry","interest_coverage"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
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

pr = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); pr[te] = lp(X[te].values, os.path.join(MDIR, f"ALL_14_{ty}.pt"))

INDS = sorted(u.industry.unique())
# header
hdr = f"{'行业':<8s}"
for ty in TY: hdr += f"  {ty:>5d}"
hdr += f"  {'合计':>5s}"
print(hdr)

for ind_name in INDS:
    row = f"{ind_name:<8s}"
    ttl = 0; ttl_ret = []
    for ty in TY:
        b = (yrs == ty) & (pr == 4) & MASK & (u.industry == ind_name)
        n = b.sum(); a = rets[b]
        ra = a.mean()*100 if n > 0 else 0
        row += f" {n:>3d}只{ra:>+4.0f}%"
        ttl += n; ttl_ret.extend(a)
    ra = np.mean(ttl_ret)*100 if ttl > 0 else 0
    row += f" {ttl:>3d}只{ra:>+4.0f}%"
    print(row)

# total
row = f"{'合计':<8s}"
all_ttl = 0; all_ret = []
for ty in TY:
    b = (yrs == ty) & (pr == 4) & MASK; n = b.sum(); a = rets[b]
    ra = a.mean()*100 if n > 0 else 0
    row += f" {n:>3d}只{ra:>+4.0f}%"
    all_ttl += n; all_ret.extend(a)
ra = np.mean(all_ret)*100 if all_ttl > 0 else 0
row += f" {all_ttl:>3d}只{ra:>+4.0f}%"
print(row)
