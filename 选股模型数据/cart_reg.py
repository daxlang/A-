"""CART+REG: 去Hard训练集 + 高惩罚256x128x64"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, pickle, copy, time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
DEV = torch.device("cuda")
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
TY = [2020, 2021, 2022, 2023, 2024]

df = pd.read_csv(os.path.join(BASE, "training_all.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry","interest_coverage"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8)

TAG = "REG_CART_256x128x64_d05_wd02"
NET = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,5))
WD = 0.02

pr_all = np.zeros(len(u), dtype=int); t0 = time.time()
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    # 加载cartography去Hard
    cache_path = os.path.join(MDIR, f"cart_cache_{ty}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f: data = pickle.load(f)
        confs = data["confs"]; tr_idx_local = data["tr_idx"]
        epochs = sorted(confs.keys()); cmat = np.array([confs[ep] for ep in epochs])
        conf = cmat.mean(axis=0); var = cmat.std(axis=0)
        is_hard = (conf <= np.percentile(conf, 33)) & (var <= np.percentile(var, 33))
        tr_clean = tr_idx_local[~is_hard]
    else:
        tr_clean = np.where(tr)[0]
    n_hard = tr.sum() - len(tr_clean)
    print(f"fold {ty}: 去Hard{n_hard}条 剩训{len(tr_clean)}条")
    
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X.iloc[tr_clean].values)).to(DEV)
    ys = torch.LongTensor(y[tr_clean]).to(DEV)
    m = copy.deepcopy(NET).to(DEV)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=WD)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
    bl, bs = 1e9, None
    for ep in range(1, 5001):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m = m.cpu(); bs_cpu = {k:v.cpu() for k,v in bs.items()}
    torch.save({"model":bs_cpu,"scaler":sc}, os.path.join(MDIR, f"{TAG}_{ty}.pt"))
    
    m_eval = copy.deepcopy(NET); m_eval.load_state_dict(bs_cpu); m_eval.eval()
    Xte = torch.FloatTensor(sc.transform(X.iloc[te].values))
    with torch.no_grad(): pr_all[te] = torch.argmax(m_eval(Xte), dim=1).numpy()

cm = confusion_matrix(y[MASK], pr_all[MASK]); n_cm = cm.sum()
acc = sum(cm[i,i] for i in range(5))/n_cm*100
buy = (pr_all == 4) & MASK; nb = buy.sum(); br = rets[buy]
yr = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
rec = cm[4,4]/cm[4].sum()*100
n_params = sum(p.numel() for p in copy.deepcopy(NET).parameters())

print(f"\n{TAG} 参数{n_params} acc={acc:.1f}% 买{nb}只 年{np.mean(yr):.1f}% 召回{rec:.1f}%")
print(f"  {'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
for i, lb in enumerate(LAB):
    print(f"  {cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
for ty in TY:
    b = (yrs==ty)&buy; n=b.sum(); a=rets[b]
    print(f"    {ty}: {n}只 {a.mean()*100:.1f}%" if n>0 else f"    {ty}: 0只")

print(f"\n对比:")
print(f"  REG_256x128x64(全量):  593买  +46.9%")
print(f"  CART_MLP(128x64去Hard): 1284买 +39.7%")
print(f"  {TAG}: {nb}买 {np.mean(yr):.1f}%")
