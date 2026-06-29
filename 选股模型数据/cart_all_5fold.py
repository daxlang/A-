"""CART ALL 5折: 逐年tracking + 去Hard重训 (约3-4小时)"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, pickle, time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
TY = [2020, 2021, 2022, 2023, 2024]
SAVE_EP = 250; N_EP = 5000

df = pd.read_csv(os.path.join(BASE, "training_all.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry","interest_coverage"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8)

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

pr = np.zeros(len(u), dtype=int)
for ty_idx, ty in enumerate(TY):
    t0 = time.time()
    tr = (yrs != ty); te = (yrs == ty); n_tr = tr.sum()
    print(f"\nFold {ty_idx+1}/5: year {ty} 训{n_tr}条 测{te.sum()}条")
    
    # Tracking
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X[tr].values)); ys = torch.LongTensor(y[tr])
    m = M5(Xs.shape[1]); opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]))
    bl, bs = 1e9, None; ep_confs = {}
    for ep in range(1, N_EP+1):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
        if ep % SAVE_EP == 0:
            m.eval()
            with torch.no_grad():
                probs = torch.softmax(m(Xs), dim=1).cpu().numpy()
                ep_confs[ep] = probs[np.arange(n_tr), ys.cpu().numpy()]
        if ep == 1000 or ep == 3000:
            print(f"  track ep {ep} loss={loss.item():.4f}")
    m.load_state_dict(bs); m.eval()
    print(f"  tracking完成: {(time.time()-t0):.0f}秒")
    
    # Cartography
    epochs = sorted(ep_confs.keys()); cmat = np.array([ep_confs[e] for e in epochs])
    conf = cmat.mean(axis=0); var = cmat.std(axis=0)
    is_hard = (conf <= np.percentile(conf, 33)) & (var <= np.percentile(var, 33))
    n_hard = is_hard.sum()
    print(f"  Hard: {n_hard}({n_hard/n_tr*100:.0f}%)")
    
    # 重训去Hard
    tr_clean = np.where(tr)[0][~is_hard]
    t1 = time.time()
    sc_c = StandardScaler(); Xs_c = torch.FloatTensor(sc_c.fit_transform(X[tr_clean].values)); ys_c = torch.LongTensor(y[tr_clean])
    m_c = M5(Xs_c.shape[1]); opt_c = optim.AdamW(m_c.parameters(), lr=0.001, weight_decay=5e-3)
    bl_c, bs_c = 1e9, None
    for ep in range(1, 5001):
        m_c.train(); opt_c.zero_grad(); loss_c = lfn(m_c(Xs_c), ys_c); loss_c.backward(); opt_c.step()
        if loss_c.item() < bl_c: bl_c = loss_c.item(); bs_c = {k:v.clone() for k,v in m_c.state_dict().items()}
    m_c.load_state_dict(bs_c); m_c.eval()
    torch.save({"model":bs_c,"scaler":sc_c}, os.path.join(MDIR, f"CART_ALL_{ty}.pt"))
    print(f"  clean重训完成: {(time.time()-t1):.0f}秒 总{(time.time()-t0):.0f}秒")

# 加载ALL_14基准预测
print("\n加载ALL_14基准...")
pr_base = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty)
    ckpt = torch.load(os.path.join(MDIR, f"ALL_14_{ty}.pt"), weights_only=False)
    m = M5(X[te].shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xte = torch.FloatTensor(ckpt["scaler"].transform(X[te].values))
    with torch.no_grad(): pr_base[te] = torch.argmax(m(Xte), dim=1).numpy()

# 加载CART预测
pr_cart = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty)
    ckpt = torch.load(os.path.join(MDIR, f"CART_ALL_{ty}.pt"), weights_only=False)
    m = M5(X[te].shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xte = torch.FloatTensor(ckpt["scaler"].transform(X[te].values))
    with torch.no_grad(): pr_cart[te] = torch.argmax(m(Xte), dim=1).numpy()

# 评估
for tag, p in [("ALL_14(基准)", pr_base), ("CART_ALL(去Hard)", pr_cart)]:
    cm = confusion_matrix(y[MASK], p[MASK]); n_cm = cm.sum()
    acc = sum(cm[i,i] for i in range(5))/n_cm*100
    buy = (p == 4) & MASK; nb = buy.sum(); br = rets[buy]
    yr = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
    print(f"\n=== {tag} ({n_cm}条 acc={acc:.1f}%) ===")
    for i, lb in enumerate(LAB):
        print(f"  {lb:<6s}{cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
    print(f"  买入: {nb}只 均{br.mean()*100:.1f}% 年{np.mean(yr):.1f}% 暴涨召回{cm[4,4]}/{cm[4].sum()}")
    for ty in TY:
        b = (yrs == ty) & buy; n = b.sum(); a = rets[b]
        print(f"    {ty}: {n}只 {a.mean()*100:.1f}%" if n > 0 else f"    {ty}: 未买")
