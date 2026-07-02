"""固定延迟版 w=0.5 架构扫描第二轮 (AutoDL GPU)
候选: 更宽/瓶颈/五层/高drop/lr半/无gm_pct
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

# gm_pct 变体: 不加gm_pct时用gross_margin代替
feat_no_gm = feat[:]  # feat doesn't include gm_pct but includes gross_margin
DF_base = u[feat + list(ind.columns)].copy()

# 无gm_pct版本: 把gm_pct从feat去掉（gross_margin还在）
feat_nogm = [c for c in feat if c != 'gm_pct']
DF_nogm = u[feat_nogm + list(ind.columns)].copy()

for dframe in [DF_base, DF_nogm]:
    for c in dframe.columns: dframe[c] = dframe[c].fillna(dframe[c].median())

y_all = u.L.values.astype(int); rets_all = u.forward_return.values; yrs_all = u.year.values
MASK_all = np.array((u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8))
N = DF_base.shape[1]; N_nogm = DF_nogm.shape[1]
print(f"Feats base={N} nogm={N_nogm} 样本={len(u)} 掩码={MASK_all.sum()}\n")

os.makedirs("weights2", exist_ok=True)

def build_net(dims, drop, N):
    layers = []; prev = N
    for d in dims:
        layers.append(nn.Linear(prev, d)); layers.append(nn.ReLU()); layers.append(nn.Dropout(drop))
        prev = d
    layers.append(nn.Linear(prev, 5))
    return nn.Sequential(*layers)

CONFIGS = [
    # 基线复跑
    ("基线512-256-128", DF_base, N, [512,256,128], 0.5, 0.02, 0.001),
    # 更宽
    ("超宽768-384-192", DF_base, N, [768,384,192], 0.5, 0.02, 0.001),
    # 瓶颈结构
    ("瓶颈512-128-256", DF_base, N, [512,128,256], 0.5, 0.02, 0.001),
    # 五层
    ("五层512-256-128-64-32", DF_base, N, [512,256,128,64,32], 0.5, 0.02, 0.001),
    # 高drop
    ("高drop0.6", DF_base, N, [512,256,128], 0.6, 0.02, 0.001),
    # 半lr
    ("半lr512-256-128", DF_base, N, [512,256,128], 0.5, 0.02, 0.0005),
    # 无gm_pct
    ("无gm_pct512-256", DF_nogm, N_nogm, [512,256,128], 0.5, 0.02, 0.001),
]

for name, DF, nf, dims, drop, wd, lr in CONFIGS:
    pr_all = np.zeros(len(u), dtype=int); t0 = time.time()
    print(f"{name}...", end=" ", flush=True)
    
    for ty in TY:
        tr = (yrs_all != ty); te = (yrs_all == ty)
        sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(DF.iloc[tr].values)).to(DEV)
        ys = torch.LongTensor(y_all[tr]).to(DEV)
        m = build_net(dims, drop, nf).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
        bl = 1e9; bs = None
        for ep in range(5001):
            m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
            if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
        m.load_state_dict(bs); m = m.cpu().eval()
        safe_name = name.replace(" ","_").replace("-","_")
        torch.save({"model":{k:v.cpu() for k,v in bs.items()},"scaler":sc,"dims":dims},
                    f"weights2/{safe_name}_{ty}.pt")
        Xte = torch.FloatTensor(sc.transform(DF.iloc[te].values))
        with torch.no_grad(): pr_all[te] = torch.argmax(m(Xte), dim=1).numpy()
        del m, sc; print(ty, end=" ", flush=True)
    
    dt = time.time() - t0; buy = (pr_all == 4) & MASK_all
    nb = int(buy.sum()); br = rets_all[buy]
    cm = confusion_matrix(y_all[MASK_all], pr_all[MASK_all])
    yr_vals = [rets_all[(yrs_all == ty) & buy].mean() * 100 for ty in TY if sum((yrs_all == ty) & buy) > 0]
    ym = np.mean(yr_vals) if yr_vals else 0
    rec = cm[4,4] / cm[4].sum() * 100 if cm[4].sum() > 0 else 0
    fb = int(cm[0,4])
    print(f"({dt:.0f}s)")
    print(f"  买{nb}只  年{ym:+.1f}%  召回{rec:.1f}%  FB{fb}")
    for ty_val in TY:
        bv = (yrs_all == ty_val) & buy; n1 = bv.sum(); a = rets_all[bv]
        if n1 > 0: print(f"    {ty_val}: {n1}只 {a.mean()*100:.1f}%")
    print()

print("Done! 权重在 weights2/")
