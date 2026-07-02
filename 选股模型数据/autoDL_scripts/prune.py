"""固定延迟版 w=0.5 Cartography去Hard剪枝 (AutoDL GPU)
策略: 每折首轮全量训 → 每500ep记录置信度 → 按(1-conf)*var排名 → 删TopHard% → 重训
剪枝比: 0%, 12%, 24%, 36%, 48%, 60%
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEV}")
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌", "跌", "小涨", "中涨", "暴涨"]
RECORD_EVERY = 500  # 每500ep记录一次置信度

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
        try: lo, hi = u.loc[m, c].quantile(0.01), u.loc[m, c].quantile(0.99); u.loc[m, c] = u.loc[m, c].clip(lo, hi)
        except: pass
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
for c in ind.columns: u[c] = ind[c].values
DF_all = u[feat + list(ind.columns)].copy()
for c in DF_all.columns: DF_all[c] = DF_all[c].fillna(DF_all[c].median())

y_all = u.L.values.astype(int); rets_all = u.forward_return.values; yrs_all = u.year.values
MASK_all = np.array((u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8))
N = DF_all.shape[1]
print(f"特征: {N}  掩码内: {MASK_all.sum()}\n")

def cartography_first_pass(X_tr, y_tr, N, dev):
    """首轮训练 + 记录每RECORD_EVERY步的置信度"""
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(dev)
    ys = torch.LongTensor(y_tr).to(dev)
    m = nn.Sequential(nn.Linear(N, 256), nn.ReLU(), nn.Dropout(0.5),
                      nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
                      nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
                      nn.Linear(64, 5)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1, 1, 1, 0.5]).to(dev))
    bl = 1e9; bs = None
    n_samples = len(y_tr); n_records = 5000 // RECORD_EVERY
    confidence_snapshots = np.zeros((n_records, n_samples))
    
    for ep in range(1, 5001):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k: v.clone() for k, v in m.state_dict().items()}
        if ep % RECORD_EVERY == 0:
            m.eval()
            with torch.no_grad():
                probs = torch.softmax(m(Xs), dim=1)
                conf = probs[np.arange(n_samples), ys].cpu().numpy()
            confidence_snapshots[ep // RECORD_EVERY - 1] = conf
    m.load_state_dict(bs)
    # 每个样本的 mean confidence 和 variance
    mean_conf = confidence_snapshots.mean(axis=0)
    var_conf = confidence_snapshots.var(axis=0)
    # Hard score: 高var + 低conf → 难样本
    hard_score = var_conf * (1 - mean_conf)
    return m.cpu(), sc, hard_score

def train_fold(X_tr, y_tr, N, dev):
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(dev)
    ys = torch.LongTensor(y_tr).to(dev)
    m = nn.Sequential(nn.Linear(N, 256), nn.ReLU(), nn.Dropout(0.5),
                      nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
                      nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
                      nn.Linear(64, 5)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1, 1, 1, 0.5]).to(dev))
    bl = 1e9; bs = None
    for ep in range(5001):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    return m.cpu(), sc

def predict(m, sc, X_te):
    Xt = torch.FloatTensor(sc.transform(X_te))
    with torch.no_grad(): pred = torch.argmax(m.cpu()(Xt), dim=1)
    return pred.numpy()

# ===== 扫描 =====
for prune_pct in [0, 12, 24, 36, 48, 60]:
    pr_all = np.zeros(len(u), dtype=int); t0 = time.time()
    print(f"prune={prune_pct}%...", end=" ", flush=True)

    for ty in TY:
        tr = (yrs_all != ty); te = (yrs_all == ty)
        X_tr = DF_all.iloc[tr].values; y_tr = y_all[tr]

        if prune_pct > 0:
            m1, sc1, hard_score = cartography_first_pass(X_tr, y_tr, N, DEV)
            # 按hard score排序，删除top prune_pct%
            threshold = np.percentile(hard_score, 100 - prune_pct)
            keep = hard_score <= threshold
            kept_pct = keep.mean() * 100
            X_tr = X_tr[keep]; y_tr = y_tr[keep]
            del m1, sc1
        else:
            kept_pct = 100

        m, sc = train_fold(X_tr, y_tr, N, DEV)
        pred_te = predict(m, sc, DF_all.iloc[te].values)
        pr_all[te] = pred_te
        del m, sc
        print(ty, end=" ", flush=True)

    dt = time.time() - t0; buy = (pr_all == 4) & MASK_all
    nb = int(buy.sum()); br = rets_all[buy]
    cm = confusion_matrix(y_all[MASK_all], pr_all[MASK_all])
    yr_vals = [rets_all[(yrs_all == ty) & buy].mean() * 100 for ty in TY if sum((yrs_all == ty) & buy) > 0]
    ym = np.mean(yr_vals) if yr_vals else 0
    rec = cm[4, 4] / cm[4].sum() * 100 if cm[4].sum() > 0 else 0
    fb = int(cm[0, 4])

    print(f"({dt:.0f}s)")
    print(f"  保留{kept_pct:.0f}%  买{nb}只  年{ym:+.1f}%  召回{rec:.1f}%  FB{fb}")
    for ty_val in TY:
        bv = (yrs_all == ty_val) & buy; n1 = bv.sum(); a = rets_all[bv]
        if n1 > 0: print(f"    {ty_val}: {n1}只 {a.mean()*100:.1f}%")
    hdr = f"    {'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}"
    print(hdr)
    for i, lb in enumerate(LAB):
        print(f"    {cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
    print()

print("全部完成!")
