"""固定延迟版 REG512+RF多版本共识 (AutoDL GPU)
复用weights2/中已训练的512-256-128权重 + 4种RF
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, time, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEV}")
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌", "跌", "小涨", "中涨", "暴涨"]

df = pd.read_csv("training_extended.csv", dtype={"code": str})
u = df[(df.usable) & (df.year <= 2024)].copy()
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
        try: lo,hi=u.loc[m,c].quantile(0.01),u.loc[m,c].quantile(0.99);u.loc[m,c]=u.loc[m,c].clip(lo,hi)
        except: pass
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
for c in ind.columns: u[c] = ind[c].values
DF = u[feat + list(ind.columns)].copy()
for c in DF.columns: DF[c] = DF[c].fillna(DF[c].median())
y_all = u.L.values.astype(int); rets_all = u.forward_return.values; yrs_all = u.year.values
MASK_all = np.array((u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8))
N = DF.shape[1]
print(f"Feats={N} 样本={len(u)} 掩码={MASK_all.sum()}\n")

os.makedirs("weights3", exist_ok=True)

# ===== 1. 加载已训练的REG 512-256-128 =====
print("=== REG 512-256-128 (复用weights2) ===")
reg_pr = np.zeros(len(u), dtype=int)
for ty in TY:
    ckpt = torch.load(f"weights2/基线512_256_128_{ty}.pt", weights_only=False)
    dims = ckpt["dims"]
    m = nn.Sequential(
        nn.Linear(N, dims[0]), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(dims[1], dims[2]), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(dims[2], 5)
    )
    m.load_state_dict(ckpt["model"]); m.eval()
    te = (yrs_all == ty)
    Xte = torch.FloatTensor(ckpt["scaler"].transform(DF.iloc[te].values))
    with torch.no_grad(): reg_pr[te] = torch.argmax(m(Xte), dim=1).numpy()
buy = (reg_pr == 4) & MASK_all; nb = int(buy.sum()); br = rets_all[buy]
yr_vals = [rets_all[(yrs_all == ty) & buy].mean() * 100 for ty in TY if sum((yrs_all == ty) & buy) > 0]
print(f"REG: {nb}买 年{np.mean(yr_vals):.1f}%")

# ===== 2. 训练多种RF =====
RF_CONFIGS = [
    ("RF_100_d20", 100, 20),
    ("RF_200_d20", 200, 20),
    ("RF_100_d15", 100, 15),
    ("RF_200_d15", 200, 15),
]

print("\n=== RF训练 ===")
rf_preds = {}
for name, ntrees, mdepth in RF_CONFIGS:
    rf_all = np.zeros(len(u), dtype=int); t0 = time.time()
    print(f"{name}...", end=" ", flush=True)
    for ty in TY:
        tr = (yrs_all != ty); te = (yrs_all == ty)
        rf = RandomForestClassifier(n_estimators=ntrees, max_depth=mdepth, min_samples_leaf=50,
            class_weight={0:1,1:1,2:1,3:1,4:0.5}, n_jobs=-1, random_state=42)
        rf.fit(DF.iloc[tr].values, y_all[tr])
        rf_all[te] = rf.predict(DF.iloc[te].values)
        with open(f"weights3/{name}_{ty}.pkl", "wb") as f: pickle.dump(rf, f)
    rf_preds[name] = rf_all
    buy_rf = (rf_all == 4) & MASK_all
    nb_rf = int(buy_rf.sum()); br_rf = rets_all[buy_rf]
    yr_rf = [rets_all[(yrs_all == ty) & buy_rf].mean() * 100 for ty in TY if sum((yrs_all == ty) & buy_rf) > 0]
    print(f"({time.time()-t0:.0f}s) {nb_rf}买 年{np.mean(yr_rf):.1f}%")

# ===== 3. REG∩RF共识 =====
print("\n=== REG∩RF共识 ===")
print(f"{'RF版本':<15s}{'REG∩RF买':>10s}{'年均':>8s}{'>=0年':>8s}")
for name in RF_CONFIGS:
    rf_pr = rf_preds[name[0]]
    buy_c = (reg_pr == 4) & (rf_pr == 4) & MASK_all
    nb_c = int(buy_c.sum()); br_c = rets_all[buy_c]
    yr_c = [rets_all[(yrs_all == ty) & buy_c].mean() * 100 for ty in TY if sum((yrs_all == ty) & buy_c) > 0]
    ym_c = np.mean(yr_c) if yr_c else 0
    pos_yrs = sum(1 for v in yr_c if v > 0)
    print(f"{name[0]:<15s}{nb_c:>10d}{ym_c:>+7.1f}%  {pos_yrs}/{len(yr_c)}年")
    
    # 详细每年
    if nb_c > 0:
        print(f"  ", end="")
        for ty in TY:
            bv = (yrs_all == ty) & buy_c; n1 = bv.sum(); a = rets_all[bv]
            if n1 > 0: print(f"{ty}:{n1}只{a.mean()*100:.1f}%  ", end="")
        print()

# ===== 4. 最佳RF单独也报 =====
print(f"\n=== 等权基线: +12.3%  REG512: 4166买+23.4%(复跑) ===")
print("Done! 权重在 weights3/")
