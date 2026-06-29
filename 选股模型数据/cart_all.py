"""Cartography ALL: RAND 80/20, 子采样30%训track, 全量clean重训"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models"); SEED = 42
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
SAVE_EP = 200; N_EP = 5000

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
te_idx = idx[int(n*0.8):]; tr_pool = idx[:int(n*0.8)]

# 子采样30%做tracking(全量太慢)
np.random.seed(999)
tr_small = np.random.choice(tr_pool, size=int(len(tr_pool)*0.3), replace=False)
print(f"Tracking: 训{len(tr_small)}条 (30%子采样)")

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

# Step1: Tracking
sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X.iloc[tr_small].values)); ys = torch.LongTensor(y[tr_small])
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
            ep_confs[ep] = probs[np.arange(len(tr_small)), ys.cpu().numpy()]
m.load_state_dict(bs); m.eval()
torch.save({"model":bs,"scaler":sc}, os.path.join(MDIR, "RAND_ALL.pt"))

# Step2: Cartography on training subset
epochs = sorted(ep_confs.keys()); cmat = np.array([ep_confs[e] for e in epochs])
conf = cmat.mean(axis=0); var = cmat.std(axis=0)
# 用tracking set的阈值 → 推广到全量training set需要全量预测
# 方案: 用训练好的m预测全训集, 取最终epoch的prob
m_cpu = M5(Xs.shape[1]); m_cpu.load_state_dict(bs); m_cpu.eval()
X_tr_full = torch.FloatTensor(sc.transform(X.iloc[tr_pool].values))
with torch.no_grad():
    probs_full = torch.softmax(m_cpu(X_tr_full), dim=1).cpu().numpy()
    conf_full = probs_full[np.arange(len(tr_pool)), y[tr_pool]]
var_full = np.zeros_like(conf_full)  # 全量没有variability, 只用confidence

c_thresh = np.percentile(conf, [33, 66])
is_hard = conf_full <= c_thresh[0]  # 只用conf分区
is_easy = conf_full >= c_thresh[1]
n_hard = is_hard.sum(); n_easy = is_easy.sum()
n_amb = len(tr_pool) - n_hard - n_easy
print(f"\nCartography(全量训集 {len(tr_pool)}条, conf阈值={c_thresh[0]:.3f}/{c_thresh[1]:.3f}):")
print(f"  Easy: {n_easy}({n_easy/len(tr_pool)*100:.0f}%)  Amb: {n_amb}({n_amb/len(tr_pool)*100:.0f}%)  Hard: {n_hard}({n_hard/len(tr_pool)*100:.0f}%)")

# Hard区分布
hard_subset = rets[tr_pool][is_hard]
hard_lab = y[tr_pool][is_hard]
for lb in range(5):
    n_lb = (hard_lab == lb).sum(); avg = hard_subset[hard_lab == lb].mean()*100 if n_lb > 0 else 0
    print(f"  Hard区 {LAB[lb]}: {n_lb}条 均收益{avg:.1f}%")

# Step3: 重训去Hard
tr_clean = tr_pool[~is_hard]
print(f"\nStep3: 重训(去Hard {n_hard}条, 剩{len(tr_clean)}条)...")
sc_c = StandardScaler(); Xs_c = torch.FloatTensor(sc_c.fit_transform(X.iloc[tr_clean].values)); ys_c = torch.LongTensor(y[tr_clean])
m_c = M5(Xs_c.shape[1]); opt_c = optim.AdamW(m_c.parameters(), lr=0.001, weight_decay=5e-3)
bl_c, bs_c = 1e9, None
for _ in range(5000):
    m_c.train(); opt_c.zero_grad(); loss_c = lfn(m_c(Xs_c), ys_c); loss_c.backward(); opt_c.step()
    if loss_c.item() < bl_c: bl_c = loss_c.item(); bs_c = {k:v.clone() for k,v in m_c.state_dict().items()}
m_c.load_state_dict(bs_c); m_c.eval()
torch.save({"model":bs_c,"scaler":sc_c}, os.path.join(MDIR, "CART_ALL.pt"))

# Step4: 评估对比
print("\nStep4: 评估...")
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
    print(f"  实际->预测  {'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
    for i, lb in enumerate(LAB):
        print(f"  {lb:<8s}{cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
    print(f"  买入: {nb}只 均{br.mean()*100:.1f}% 年{yr:.1f}% 暴涨召回{cm[4,4]}/{cm[4].sum()}")
    for ty in sorted(set(yrs_te)):
        b = (yrs_te == ty) & buy; n_b = b.sum(); a = rets_te[b]
        print(f"    {ty}: {n_b}只 {a.mean()*100:.1f}%" if n_b > 0 else f"    {ty}: 未买")
