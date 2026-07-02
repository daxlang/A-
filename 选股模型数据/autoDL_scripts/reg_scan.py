"""REG512保守化扫描: w值和wd值 (AutoDL GPU)
复用预处理+数据, 仅扫REG 512-256-128的不同惩罚
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEV}")
TY = [2020, 2021, 2022, 2023, 2024]

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
os.makedirs("weights_reg", exist_ok=True)

def build_net():
    return nn.Sequential(nn.Linear(N,512),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(128,5))

# 扫描: w值 + wd值
SWEEPS = [
    ("w0.50_wd0.02", 0.50, 0.02),
    ("w0.45_wd0.02", 0.45, 0.02),
    ("w0.40_wd0.02", 0.40, 0.02),
    ("w0.35_wd0.02", 0.35, 0.02),
    ("w0.50_wd0.05", 0.50, 0.05),
    ("w0.50_wd0.10", 0.50, 0.10),
]

for name, w, wd in SWEEPS:
    pr_all = np.zeros(len(u), dtype=int); t0 = time.time()
    print(f"{name}...", end=" ", flush=True)
    
    for ty in TY:
        tr = (yrs_all != ty); te = (yrs_all == ty)
        sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(DF.iloc[tr].values)).to(DEV)
        ys = torch.LongTensor(y_all[tr]).to(DEV)
        m = build_net().to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=wd)
        lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,w]).to(DEV))
        bl = 1e9; bs = None
        for ep in range(5001):
            m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
            if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
        m.load_state_dict(bs); m = m.cpu().eval()
        torch.save({"model":{k:v.cpu() for k,v in bs.items()},"scaler":sc},
                    f"weights_reg/{name}_{ty}.pt")
        Xte = torch.FloatTensor(sc.transform(DF.iloc[te].values))
        with torch.no_grad(): pr_all[te] = torch.argmax(m(Xte), dim=1).numpy()
        del m, sc; print(ty, end=" ", flush=True)
    
    dt = time.time() - t0; buy = (pr_all == 4) & MASK_all
    nb = int(buy.sum()); br = rets_all[buy]
    yr_vals = [rets_all[(yrs_all == ty) & buy].mean() * 100 for ty in TY if sum((yrs_all == ty) & buy) > 0]
    ym = np.mean(yr_vals) if yr_vals else 0
    cm = confusion_matrix(y_all[MASK_all], pr_all[MASK_all])
    rec = cm[4,4] / cm[4].sum() * 100 if cm[4].sum() > 0 else 0
    fb = int(cm[0,4])
    print(f"({dt:.0f}s)")
    print(f"  买{nb}只  年{ym:+.1f}%  召回{rec:.1f}%  暴跌->暴涨{fb}")
    for ty_val in TY:
        bv = (yrs_all == ty_val) & buy; n1 = bv.sum(); a = rets_all[bv]
        if n1 > 0: print(f"    {ty_val}: {n1}只 {a.mean()*100:.1f}%")
    print()

print("Done! weights_reg/")
