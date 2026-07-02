"""固定延迟版 暴涨惩罚扫描 (AutoDL GPU)
用法: python w_sweep.py
输出: 各w值的5折CV买入数、年均收益、混淆矩阵
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEV}")
TY = [2020, 2021, 2022, 2023, 2024]
WEIGHTS = [0.5, 0.3, 0.25, 0.2, 0.15, 0.1]
LAB = ["暴跌", "跌", "小涨", "中涨", "暴涨"]

# ===== 加载 + 预处理 =====
df = pd.read_csv("training_extended.csv", dtype={"code": str})
u = df[(df.usable) & (df.year <= 2024)].copy()
print(f"样本: {len(u)}条 {u.code.nunique()}只")
u["gm_pct"] = u.groupby(["industry", "year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
def lbl(r):
    if pd.isna(r): return 0
    if r < -0.1: return 0
    if r < 0: return 1
    if r < 0.1: return 2
    if r < 0.3: return 3
    return 4
u["L"] = u.forward_return.apply(lbl)
feat = [c for c in u.columns if c not in ["code", "year", "quarter", "forward_return", "gm_pct", "L", "usable", "industry", "interest_coverage", "buy_price", "pe", "pb", "ps"]]
FCOLS = feat + ["buy_price", "pe", "pb", "ps"]
for yr in sorted(u.year.unique()):
    m = u.year == yr
    for c in FCOLS:
        try:
            lo, hi = u.loc[m, c].quantile(0.01), u.loc[m, c].quantile(0.99)
            u.loc[m, c] = u.loc[m, c].clip(lo, hi)
        except: pass
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
for c in ind.columns: u[c] = ind[c].values
DF = u[feat + list(ind.columns)].copy()
for c in DF.columns: DF[c] = DF[c].fillna(DF[c].median())

y = u.L.values.astype(int); rets = u.forward_return.values; yrs = u.year.values
MASK = np.array((u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8))
N = DF.shape[1]
print(f"特征: {N}  掩码内: {MASK.sum()}\n")

# ===== 扫描 =====
for w in WEIGHTS:
    pr_all = np.zeros(len(u), dtype=int); t0 = time.time()
    print(f"w={w}...", end=" ", flush=True)
    for ty in TY:
        tr = (yrs != ty); te = (yrs == ty)
        sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(DF.iloc[tr].values)).to(DEV)
        ys = torch.LongTensor(y[tr]).to(DEV)
        m = nn.Sequential(nn.Linear(N, 256), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(64, 5)).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
        lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1, 1, 1, w]).to(DEV))
        bl = 1e9; bs = None
        for ep in range(5001):
            m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
            if loss.item() < bl: bl = loss.item(); bs = {k: v.clone() for k, v in m.state_dict().items()}
        m.load_state_dict(bs); m = m.cpu().eval()
        Xte = torch.FloatTensor(sc.transform(DF.iloc[te].values))
        with torch.no_grad(): pr_all[te] = torch.argmax(m(Xte), dim=1).numpy()
        print(ty, end=" ", flush=True)

    dt = time.time() - t0; buy = (pr_all == 4) & MASK
    nb = int(buy.sum()); br = rets[buy]; cm = confusion_matrix(y[MASK], pr_all[MASK])
    yr_vals = [rets[(yrs == ty) & buy].mean() * 100 for ty in TY if sum((yrs == ty) & buy) > 0]
    ym = np.mean(yr_vals) if yr_vals else 0
    rec = cm[4, 4] / cm[4].sum() * 100 if cm[4].sum() > 0 else 0
    fb = int(cm[0, 4])
    print(f"({dt:.0f}s)")
    print(f"  买{nb}只  年{ym:+.1f}%  召回{rec:.1f}%  暴跌->暴涨{fb}")
    for ty_val in TY:
        bv = (yrs == ty_val) & buy; n1 = bv.sum(); a = rets[bv]
        if n1 > 0: print(f"    {ty_val}: {n1}只 {a.mean()*100:.1f}%")
    hdr = f"    {'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}"
    print(hdr)
    for i, lb in enumerate(LAB):
        print(f"    {cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
    print()

print("全部完成!")
