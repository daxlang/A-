"""Cartography ALL: 全量 80/20 tracking + clean重训"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models"); SEED = 42
LAB = ["暴跌","跌","小涨","中涨","暴涨"]

df = pd.read_csv(os.path.join(BASE, "training_all.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry","interest_coverage"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8)
np.random.seed(SEED); n = len(u); idx = np.random.permutation(n)
te_idx = idx[int(n*0.8):]; tr_idx = idx[:int(n*0.8)]
print(f"全量Tracking: 训{len(tr_idx)}条, 测{len(te_idx)}条")
N_EP = 5000; SAVE_EP = 250; n_tr = len(tr_idx)

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X.iloc[tr_idx].values)); ys = torch.LongTensor(y[tr_idx])
m = M5(Xs.shape[1]); opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]))
bl, bs = 1e9, None; ep_confs = {}

import time; t0 = time.time()
for ep in range(1, N_EP+1):
    m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
    if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    if ep % SAVE_EP == 0:
        m.eval()
        with torch.no_grad():
            probs = torch.softmax(m(Xs), dim=1).cpu().numpy()
            ep_confs[ep] = probs[np.arange(n_tr), ys.cpu().numpy()]
    if ep % 500 == 0:
        elapsed = time.time() - t0
        eta = elapsed / ep * (N_EP - ep)
        print(f"  ep {ep}/{N_EP}  elapsed {elapsed/60:.0f}min  eta {eta/60:.0f}min  loss={loss.item():.4f}")

m.load_state_dict(bs); m.eval()
torch.save({"model":bs,"scaler":sc}, os.path.join(MDIR, "RAND_ALL.pt"))
print(f"Tracking完成: {time.time()-t0:.0f}秒")

# Cartography
epochs = sorted(ep_confs.keys()); cmat = np.array([ep_confs[e] for e in epochs])
conf = cmat.mean(axis=0); var = cmat.std(axis=0)
c_thresh = np.percentile(conf, [33, 66]); v_thresh = np.percentile(var, [33, 66])
is_hard = (conf <= c_thresh[0]) & (var <= v_thresh[0])
print(f"\nCartography: conf阈值={c_thresh[0]:.3f}/{c_thresh[1]:.3f} var阈值={v_thresh[0]:.4f}/{v_thresh[1]:.4f}")
print(f"  Hard: {is_hard.sum()}({is_hard.sum()/n_tr*100:.0f}%)")
for lb in range(5):
    mask = (y[tr_idx] == lb) & is_hard; n_lb = mask.sum()
    avg = rets[tr_idx][mask].mean()*100 if n_lb > 0 else 0
    print(f"  Hard区{LAB[lb]}: {n_lb}条 均收益{avg:.1f}%")

# 重训去Hard
tr_clean = tr_idx[~is_hard]
print(f"\n重训: 去Hard{is_hard.sum()}条, 剩{len(tr_clean)}条...")
t1 = time.time()
sc_c = StandardScaler(); Xs_c = torch.FloatTensor(sc_c.fit_transform(X.iloc[tr_clean].values)); ys_c = torch.LongTensor(y[tr_clean])
m_c = M5(Xs_c.shape[1]); opt_c = optim.AdamW(m_c.parameters(), lr=0.001, weight_decay=5e-3)
bl_c, bs_c = 1e9, None
for ep in range(1, 5001):
    m_c.train(); opt_c.zero_grad(); loss_c = lfn(m_c(Xs_c), ys_c); loss_c.backward(); opt_c.step()
    if loss_c.item() < bl_c: bl_c = loss_c.item(); bs_c = {k:v.clone() for k,v in m_c.state_dict().items()}
    if ep % 1000 == 0: print(f"  clean ep {ep}")
m_c.load_state_dict(bs_c); m_c.eval()
torch.save({"model":bs_c,"scaler":sc_c}, os.path.join(MDIR, "CART_ALL.pt"))
print(f"Clean重训完成: {time.time()-t1:.0f}秒")

# 评估
print("\n评估...")
yrs_te = u.iloc[te_idx]["year"].values; msk_te = MASK.iloc[te_idx].values
X_te = X.iloc[te_idx].values; y_te = y[te_idx]; rets_te = rets[te_idx]

for tag, wtag in [("RAND_ALL(基准)", "RAND_ALL.pt"), ("CART_ALL(去Hard)", "CART_ALL.pt")]:
    ckpt = torch.load(os.path.join(MDIR, wtag), weights_only=False)
    mm = M5(X_te.shape[1]); mm.load_state_dict(ckpt["model"]); mm.eval()
    Xte = torch.FloatTensor(ckpt["scaler"].transform(X_te))
    with torch.no_grad(): pr = torch.argmax(mm(Xte), dim=1).numpy()
    cm = confusion_matrix(y_te[msk_te], pr[msk_te]); n_cm = cm.sum()
    acc = sum(cm[i,i] for i in range(5))/n_cm*100
    buy = (pr == 4) & msk_te; nb = buy.sum(); br = rets_te[buy]
    yrly = []
    for ty in sorted(set(yrs_te)):
        b = (yrs_te == ty) & buy; n_b = b.sum(); a = rets_te[b]
        yl = a.mean()*100 if n_b > 0 else 0; yrly.append(yl)
    yr = np.mean(yrly)
    print(f"\n=== {tag} ({n_cm}条 acc={acc:.1f}%) ===")
    print(f"  {'实际':<6s}{'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
    for i, lb in enumerate(LAB):
        print(f"  {lb:<6s}{cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
    print(f"  买入: {nb}只 均{br.mean()*100:.1f}% 年{yr:.1f}% 召回{cm[4,4]}/{cm[4].sum()}")
    for ty in sorted(set(yrs_te)):
        b = (yrs_te == ty) & buy; n_b = b.sum(); a = rets_te[b]
        print(f"    {ty}: {n_b}只 {a.mean()*100:.1f}%" if n_b > 0 else f"    {ty}: 未买")
