"""MLP vs RF 逐年对比"""
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
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

pr_mlp = np.zeros(len(u), dtype=int)
pr_rf = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty)
    pr_mlp[te] = load_mlp(X[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
    rf = pickle.load(open(os.path.join(MDIR, f"RF_{ty}.pkl"), "rb"))
    pr_rf[te] = rf.predict(X[te].values)

print("逐年买入 (S1∩S3宽松):")
print(f"{'年':>4s} {'MLP买':>5s} {'MLP均':>7s} {'RF买':>5s} {'RF均':>7s} {'全市场':>7s} {'MLP超额':>7s} {'RF超额':>7s}")
for ty in TY:
    te = (yrs == ty) & MASK
    bm, br = (pr_mlp == 4), (pr_rf == 4)
    nb_m, nb_r = sum(te & bm), sum(te & br)
    am, ar, aa = rets[te & bm], rets[te & br], rets[te]
    print(f"{ty:>4d} {nb_m:>5d} {am.mean()*100:>+6.1f}% {nb_r:>5d} {ar.mean()*100:>+6.1f}% {aa.mean()*100:>+6.1f}% {am.mean()*100-aa.mean()*100:>+6.1f}% {ar.mean()*100-aa.mean()*100:>+6.1f}%")

# 共同 vs 独有
both = (pr_mlp == 4) & (pr_rf == 4) & MASK
mlp_only = (pr_mlp == 4) & (pr_rf != 4) & MASK
rf_only = (pr_rf == 4) & (pr_mlp != 4) & MASK
print(f"\n共同买入: {both.sum()}只 均{rets[both].mean()*100:+.1f}%")
print(f"MLP独有: {mlp_only.sum()}只 均{rets[mlp_only].mean()*100:+.1f}%")
print(f"RF独有: {rf_only.sum()}只 均{rets[rf_only].mean()*100:+.1f}%")

# 共同买入逐年
print(f"\n共同买入(MLP∩RF)逐年:")
print(f"{'年':>4s} {'买入':>5s} {'均值':>7s} {'中位':>7s} {'全市场':>7s} {'超额':>7s}")
yrly = []
for ty in TY:
    te = (yrs == ty); b = both & te; n = b.sum()
    a = rets[b]; aa = rets[te & MASK]
    yl = a.mean()*100 if n > 0 else 0; yrly.append(yl)
    print(f"{ty:>4d} {n:>5d} {yl:>+6.1f}% {np.median(a)*100:>+6.1f}% {aa.mean()*100:>+6.1f}% {yl-aa.mean()*100:>+6.1f}%")
print(f"年均: {np.mean(yrly):+.1f}%")

# LR prediction
pr_lr = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty)
    lr = pickle.load(open(os.path.join(MDIR, f"LR_{ty}.pkl"), "rb"))
    pr_lr[te] = lr.predict(X[te].values)

# MLP∩LR
for tag, p1, p2 in [("MLP∩LR", pr_mlp, pr_lr), ("RF∩LR", pr_rf, pr_lr)]:
    both2 = (p1 == 4) & (p2 == 4) & MASK
    print(f"\n共同买入({tag})逐年:")
    print(f"{'年':>4s} {'买入':>5s} {'均值':>7s} {'中位':>7s} {'全市场':>7s} {'超额':>7s}")
    yrly = []
    for ty in TY:
        te = (yrs == ty); b = both2 & te; n = b.sum()
        a = rets[b]; aa = rets[te & MASK]
        yl = a.mean()*100 if n > 0 else 0; yrly.append(yl)
        print(f"{ty:>4d} {n:>5d} {yl:>+6.1f}% {np.median(a)*100:>+6.1f}% {aa.mean()*100:>+6.1f}% {yl-aa.mean()*100:>+6.1f}%")
    print(f"年均: {np.mean(yrly):+.1f}%  总{both2.sum()}只 均{rets[both2].mean()*100:+.1f}%")

# 共识筛暴跌: 两模型都判0→排除, 剩余→MLP
print(f"\n{'='*60}")
for tag, p_scr in [("MLP∩RF筛暴跌后MLP", pr_rf), ("MLP∩LR筛暴跌后MLP", pr_lr)]:
    consensus_drop = (pr_mlp == 0) & (p_scr == 0)
    keep = MASK & ~consensus_drop
    print(f"\n{tag}: 筛掉{sum(consensus_drop&MASK)}只 剩{sum(keep)}只")
    # 只看keep的MLP预测
    buy = (pr_mlp == 4) & keep; nb = buy.sum(); br = rets[buy]
    yrly = []
    for ty in TY:
        b = (yrs == ty) & buy; n = b.sum()
        a = rets[b]; yrly.append(a.mean()*100 if n > 0 else 0)
        aa = rets[(yrs == ty) & keep]
        print(f"  {ty}: 买{n}只 均{a.mean()*100:+.1f}%" if n > 0 else f"  {ty}: 未买")
    print(f"  总买{nb}只 均{br.mean()*100:+.1f}% 年{np.mean(yrly):+.1f}%")

print(f"\n基准 MLP→S1∩S3: 买231只 年+21.8%")
