"""逐年×行业 对比"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
TY = [2020, 2021, 2022, 2023, 2024]

df = pd.read_csv(os.path.join(BASE, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
rets = u.forward_return.values; yrs = u.year.values
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1)

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def load_pred(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

pr_base = np.zeros(len(u), dtype=int); pr_cart = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); Xte = X[te].values
    pr_base[te] = load_pred(Xte, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
    pr_cart[te] = load_pred(Xte, os.path.join(MDIR, f"CART_{ty}.pt"))

INDS = ["钢铁","白酒","银行","游戏"]
TITLES = {"MLP0_5": pr_base, "CART": pr_cart}
for tag, pr in TITLES.items():
    print(f"\n逐年x行业: {tag}")
    hdr = "  {:<6s}".format("行业")
    for ty in sorted(set(yrs)):
        hdr += "  {:>5d}".format(ty)
    hdr += "  {:>5s}".format("合计")
    print(hdr)
    for ind_name in INDS:
        row = "  {:<6s}".format(ind_name)
        ttl_n = 0; ttl_r = []
        for ty in sorted(set(yrs)):
            b = (yrs == ty) & (pr == 4) & MASK & (u.industry == ind_name)
            n = b.sum(); a = rets[b]
            ra = a.mean()*100 if n > 0 else 0
            row += " {:>3d}只{:>+3.0f}%".format(n, ra)
            ttl_n += n; ttl_r.extend(a)
        re = np.mean(ttl_r)*100 if ttl_n > 0 else 0
        row += " {:>3d}只{:>+3.0f}%".format(ttl_n, re)
        print(row)
    row = "  {:<6s}".format("合计")
    tn = 0; tr = []
    for ty in sorted(set(yrs)):
        b = (yrs == ty) & (pr == 4) & MASK; n = b.sum(); a = rets[b]
        ra = a.mean()*100 if n > 0 else 0
        row += " {:>3d}只{:>+3.0f}%".format(n, ra)
        tn += n; tr.extend(a)
    row += " {:>3d}只{:>+3.0f}%".format(tn, np.mean(tr)*100)
    print(row)
