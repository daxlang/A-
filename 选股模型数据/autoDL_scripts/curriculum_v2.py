"""课程学习v2: 慢节奏SPL/Anti + 长BabyStep (AutoDL GPU)
REG 512-256-128, w=0.5, 5折CV, 10000ep, 保存权重
SPL/Anti: 0.2→0.4→0.8→1.0 每1000ep
Baby: 二分类2000ep + 五分类8000ep
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
os.makedirs("weights_cl2", exist_ok=True)

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

def get_keep_ratio(ep, fast=False):
    """慢节奏: 0.2@0, 0.4@1000, 0.8@2000, 1.0@3000"""
    if fast: stage = ep // 500    # 快节奏对比
    else:    stage = ep // 1000
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    return ratios[min(stage, len(ratios)-1)]

# ===== 基线 10000ep =====
def standard_train(X_tr, y_tr, N):
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV)
    m = build_net(N).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
    bl = 1e9; bs = None
    for ep in range(10001):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.cpu().eval()
    return m, sc

# ===== SPL v2 慢节奏 =====
def train_spl(X_tr, y_tr, N, fast=False):
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV); n_samples = len(y_tr)
    m = build_net(N).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
    bl = 1e9; bs = None
    cw = torch.FloatTensor([1,1,1,1,0.5]).to(DEV)
    for ep in range(10001):
        m.train(); keep_ratio = get_keep_ratio(ep, fast)
        logits = m(Xs); loss_ps = nn.functional.cross_entropy(logits, ys, reduction='none')
        if keep_ratio < 1.0:
            th = torch.quantile(loss_ps, keep_ratio); mask = loss_ps <= th
            for lb in range(5):
                if mask[ys == lb].sum() == 0: mask[ys == lb] = True
        else:
            mask = torch.ones(n_samples, dtype=torch.bool, device=DEV)
        opt.zero_grad()
        loss = (loss_ps * cw[ys] * mask.float()).sum() / mask.float().sum()
        loss.backward(); opt.step()
        with torch.no_grad(): fl = lfn(m(Xs), ys)
        if fl.item() < bl: bl = fl.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.cpu().eval()
    return m, sc

# ===== Anti v2 慢节奏 =====
def train_anti(X_tr, y_tr, N):
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV); n_samples = len(y_tr)
    m = build_net(N).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
    bl = 1e9; bs = None
    cw = torch.FloatTensor([1,1,1,1,0.5]).to(DEV)
    for ep in range(10001):
        m.train(); keep_ratio = get_keep_ratio(ep)
        logits = m(Xs); loss_ps = nn.functional.cross_entropy(logits, ys, reduction='none')
        if keep_ratio < 1.0:
            th = torch.quantile(loss_ps, 1.0 - keep_ratio); mask = loss_ps >= th
            for lb in range(5):
                if mask[ys == lb].sum() == 0: mask[ys == lb] = True
        else:
            mask = torch.ones(n_samples, dtype=torch.bool, device=DEV)
        opt.zero_grad()
        loss = (loss_ps * cw[ys] * mask.float()).sum() / mask.float().sum()
        loss.backward(); opt.step()
        with torch.no_grad(): fl = lfn(m(Xs), ys)
        if fl.item() < bl: bl = fl.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.cpu().eval()
    return m, sc

# ===== BabyStep v2: 2000+8000 =====
def train_babystep(X_tr, y_tr, N):
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV)
    mask_bin = (ys == 0) | (ys == 4)
    X_bin = Xs[mask_bin]; y_bin = (ys[mask_bin] == 4).long()
    # 阶段1: 2000ep
    m_bin = build_binary_net(N).to(DEV)
    opt_bin = torch.optim.AdamW(m_bin.parameters(), lr=0.001, weight_decay=0.02)
    lfn_bin = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 0.5]).to(DEV))
    for ep in range(2001):
        m_bin.train(); opt_bin.zero_grad(); lfn_bin(m_bin(X_bin), y_bin).backward(); opt_bin.step()
    bin_state = {k: v for k, v in m_bin.state_dict().items()}
    # 阶段2: 8000ep
    m = build_net(N).to(DEV)
    m_state = m.state_dict()
    for k in m_state:
        if k in bin_state and m_state[k].shape == bin_state[k].shape:
            m_state[k] = bin_state[k]
    m.load_state_dict(m_state)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
    bl = 1e9; bs = None
    for ep in range(8001):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.cpu().eval()
    return m, sc

# ===== 运行 =====
METHODS = [
    ("基线1w", standard_train),
    ("SPLv2", train_spl),
    ("Antiv2", train_anti),
    ("Babyv2", train_babystep),
]

for meth_name, train_fn in METHODS:
    pr_all = np.zeros(len(u), dtype=int); t0 = time.time()
    print(f"\n=== {meth_name} ===")
    for ty in TY:
        tr = (yrs_all != ty); te = (yrs_all == ty)
        m, sc = train_fn(DF.iloc[tr].values, y_all[tr], N)
        torch.save({"model":{k:v.cpu() for k,v in m.state_dict().items()},"scaler":sc},
                    f"weights_cl2/{meth_name}_{ty}.pt")
        Xte = torch.FloatTensor(sc.transform(DF.iloc[te].values))
        with torch.no_grad(): pr_all[te] = torch.argmax(m.cpu()(Xte), dim=1).numpy()
        del m, sc; print(ty, end=" ", flush=True)
    dt = time.time() - t0; buy = (pr_all == 4) & MASK_all
    nb = int(buy.sum()); br = rets_all[buy]
    yr_vals = [rets_all[(yrs_all == ty) & buy].mean() * 100 for ty in TY if sum((yrs_all == ty) & buy) > 0]
    ym = np.mean(yr_vals) if yr_vals else 0
    cm = confusion_matrix(y_all[MASK_all], pr_all[MASK_all])
    rec = cm[4,4] / cm[4].sum() * 100 if cm[4].sum() > 0 else 0
    fb = int(cm[0,4])
    print(f"({dt:.0f}s)")
    print(f"  买{nb}只  年{ym:+.1f}%  召回{rec:.1f}%  FB{fb}")
    for ty_val in TY:
        bv = (yrs_all == ty_val) & buy; n1 = bv.sum(); a = rets_all[bv]
        if n1 > 0: print(f"    {ty_val}: {n1}只 {a.mean()*100:.1f}%")

print(f"\nDone! 权重在 weights_cl2/")
