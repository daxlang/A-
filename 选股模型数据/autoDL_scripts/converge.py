"""收敛检查: 基线 vs SPL 在折2020上 (AutoDL GPU)"""
import numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEV}")

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
y_all = u.L.values.astype(int); yrs_all = u.year.values; N = DF.shape[1]

tr = (yrs_all != 2020)
X_tr = DF.iloc[tr].values; y_tr = y_all[tr]
sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
ys = torch.LongTensor(y_tr).to(DEV)

def build_net(N):
    return nn.Sequential(nn.Linear(N,512),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(128,5))

# ===== 基线 10000ep =====
print("\n=== 基线 (全量训练) 10000ep ===")
m = build_net(N).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
bl = 1e9
for ep in range(1, 10001):
    m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
    if loss.item() < bl: bl = loss.item()
    if ep % 1000 == 0:
        m.eval()
        with torch.no_grad():
            fl = lfn(m(Xs), ys); pred = torch.argmax(m(Xs), dim=1)
            acc = (pred == ys).float().mean()
        print(f"  ep{ep:5d}: loss={loss.item():.4f} best={bl:.4f} full={fl.item():.4f} acc={acc:.3f}")

# ===== SPL 10000ep =====
print("\n=== SPL 10000ep ===")
m = build_net(N).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
bl = 1e9
class_weights = torch.FloatTensor([1,1,1,1,0.5]).to(DEV)
for ep in range(1, 10001):
    m.train()
    if ep == 1: keep_ratio = 0.2
    elif ep % 500 == 1: keep_ratio = min(keep_ratio * 2, 1.0)
    logits = m(Xs); loss_ps = nn.functional.cross_entropy(logits, ys, reduction='none')
    if keep_ratio < 1.0:
        th = torch.quantile(loss_ps, keep_ratio); mask = loss_ps <= th
        for lb in range(5):
            if mask[ys == lb].sum() == 0: mask[ys == lb] = True
    else:
        mask = torch.ones(len(y_tr), dtype=torch.bool, device=DEV)
    opt.zero_grad()
    wps = class_weights[ys]
    loss = (loss_ps * wps * mask.float()).sum() / mask.float().sum()
    loss.backward(); opt.step()
    with torch.no_grad():
        fl = lfn(m(Xs), ys)
    if fl.item() < bl: bl = fl.item()
    if ep % 1000 == 0:
        m.eval()
        with torch.no_grad():
            pred = torch.argmax(m(Xs), dim=1); acc = (pred == ys).float().mean()
        print(f"  ep{ep:5d}: keep={keep_ratio:.1%} loss={loss.item():.4f} best={bl:.4f} full={fl.item():.4f} acc={acc:.3f} mask={mask.float().mean():.2%}")
