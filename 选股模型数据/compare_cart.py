"""CART vs MLP0_5 详细对比: 混淆矩阵 + 逐年×行业"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
TY = [2020, 2021, 2022, 2023, 2024]

df = pd.read_csv(os.path.join(BASE, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
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

# 两种模型的预测
pr_base = np.zeros(len(u), dtype=int); pr_cart = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); Xte = X[te].values
    pr_base[te] = load_pred(Xte, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
    pr_cart[te] = load_pred(Xte, os.path.join(MDIR, f"CART_{ty}.pt"))

# 混淆矩阵对比
for tag, pr in [("MLP0_5", pr_base), ("CART", pr_cart)]:
    cm = confusion_matrix(y[MASK], pr[MASK]); n = cm.sum()
    acc = sum(cm[i,i] for i in range(5)) / n * 100
    buy = (pr == 4) & MASK; nb = buy.sum(); br = rets[buy]
    yr = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
    print(f"\n=== {tag} → S1∩S3 ({n}条 acc={acc:.1f}%) ===")
    print(f"  实际→预测  {'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
    for i, lb in enumerate(LAB):
        print(f"  {lb:<8s}{cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
    print(f"  买入: {nb}只 均{br.mean()*100:.1f}% 年{np.mean(yr):.1f}% 暴涨召回{cm[4,4]}/{cm[4].sum()}")
    for ty in sorted(set(yrs)):
        b = (yrs == ty) & buy; n = b.sum(); a = rets[b]
        print(f"    {ty}: {n}只 {a.mean()*100:.1f}%" if n > 0 else f"    {ty}: 未买")

# 逐年×行业对比
INDS = ["钢铁","白酒","银行","游戏"]
for tag, pr in [("MLP0_5", pr_base), ("CART", pr_cart)]:
    print(f"\n{'='*70}")
    print(f"逐年×行业: {tag}")
    print(f"  {'行业':<6s}", end="")
    for ty in sorted(set(yrs)):
        print(f"{ty:>7s}", end="")
    print(f"  {'合计':>7s}")
    for ind_name in INDS:
        print(f"  {ind_name:<6s}", end="")
        ttl_ret = []; ttl_n = 0
        for ty in sorted(set(yrs)):
            b = (yrs == ty) & (pr == 4) & MASK & (u.industry == ind_name)
            n = b.sum(); a = rets[b]
            ra = a.mean()*100 if n > 0 else 0
            print(f" {n:>3d}只{ra:>+3.0f}%", end="")
            ttl_n += n; ttl_ret.extend(a)
        ra_e = np.mean(ttl_ret)*100 if ttl_n > 0 else 0
        print(f" {ttl_n:>3d}只{ra_e:>+3.0f}%")
    print(f"  {'合计':<6s}", end="")
    ttl_all_n = 0; ttl_all_r = []
    for ty in sorted(set(yrs)):
        b = (yrs == ty) & (pr == 4) & MASK; n = b.sum(); a = rets[b]
        ra = a.mean()*100 if n > 0 else 0
        print(f" {n:>3d}只{ra:>+3.0f}%", end="")
        ttl_all_n += n; ttl_all_r.extend(a)
    print(f" {ttl_all_n:>3d}只{np.mean(ttl_all_r)*100:>+3.0f}%")
