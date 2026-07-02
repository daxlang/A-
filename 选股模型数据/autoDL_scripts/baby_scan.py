"""BabyStep v2 暴涨惩罚扫描: w=0.4, 0.35, 0.3, 0.25 (AutoDL GPU)
REG 512-256-128, BabyStep 2000+8000ep, 5折CV, 保存权重
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.preprocessing import StandardScaler
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
os.makedirs("weights_baby", exist_ok=True)

def build_net(N):
    return nn.Sequential(nn.Linear(N,512),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(128,5))

def build_binary_net(N):
    return nn.Sequential(nn.Linear(N,512),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(128,2))

def train_baby(X_tr, y_tr, N, w):
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV)
    mask_bin = (ys == 0) | (ys == 4)
    X_bin = Xs[mask_bin]; y_bin = (ys[mask_bin] == 4).long()
    # 阶段1: 二分类 2000ep, 也用对应w
    m_bin = build_binary_net(N).to(DEV)
    opt_bin = torch.optim.AdamW(m_bin.parameters(), lr=0.001, weight_decay=0.02)
    lfn_bin = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, w]).to(DEV))
    for ep in range(2001):
        m_bin.train(); opt_bin.zero_grad(); lfn_bin(m_bin(X_bin), y_bin).backward(); opt_bin.step()
    bin_state = {k: v for k, v in m_bin.state_dict().items()}
    # 阶段2: 五分类 8000ep
    m = build_net(N).to(DEV)
    m_state = m.state_dict()
    for k in m_state:
        if k in bin_state and m_state[k].shape == bin_state[k].shape:
            m_state[k] = bin_state[k]
    m.load_state_dict(m_state)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,w]).to(DEV))
    bl = 1e9; bs = None
    for ep in range(8001):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.cpu().eval()
    return m, sc

print(f"{'w':>6s} {'买':>6s} {'年':>7s} {'召回':>6s} {'FB':>6s} {'2023':>8s}")
for w in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25]:
    pr_all = np.zeros(len(u), dtype=int); t0 = time.time()
    for ty in TY:
        tr = (yrs_all != ty); te = (yrs_all == ty)
        m, sc = train_baby(DF.iloc[tr].values, y_all[tr], N, w)
        torch.save({"model":{k:v.cpu() for k,v in m.state_dict().items()},"scaler":sc},
                    f"weights_baby/baby_w{w}_{ty}.pt")
        Xte = torch.FloatTensor(sc.transform(DF.iloc[te].values))
        with torch.no_grad(): pr_all[te] = torch.argmax(m.cpu()(Xte), dim=1).numpy()
        del m, sc
    
    dt = time.time() - t0; buy = (pr_all == 4) & MASK_all
    nb = int(buy.sum()); br = rets_all[buy]
    yr_vals = [rets_all[(yrs_all == ty) & buy].mean() * 100 for ty in TY if sum((yrs_all == ty) & buy) > 0]
    ym = np.mean(yr_vals) if yr_vals else 0
    cm = confusion_matrix(y_all[MASK_all], pr_all[MASK_all])
    rec = cm[4,4] / cm[4].sum() * 100 if cm[4].sum() > 0 else 0
    fb = int(cm[0,4])
    yr_2023 = [rets_all[(yrs_all == 2023) & buy].mean() * 100]
    yr23_str = f"{yr_2023[0]:.1f}%" if buy.sum() > 0 and sum((yrs_all==2023)&buy)>0 else "0"
    print(f"{w:>6.2f} {nb:>6d} {ym:>+6.1f}% {rec:>5.1f}% {fb:>6d}  {yr23_str:>7s} ({dt:.0f}s)")
    for ty_val in TY:
        bv = (yrs_all == ty_val) & buy; n1 = bv.sum(); a = rets_all[bv]
        if n1 > 0: print(f"    {ty_val}: {n1}只 {a.mean()*100:.1f}%")

print(f"\n对比: 基线1w 4085买+19.5%  Baby0.5 5591买+19.0%(召回8.1%)")
print(f"Done! 权重在 weights_baby/")
