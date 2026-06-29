"""架构扫描: 子采样30%训练, 5000epoch, 4变体"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
SEED = 42

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
te_idx = idx[int(n*0.8):]  # 20% test (same as RAND)
tr_pool = idx[:int(n*0.8)]  # 80% train pool
# 只用30%的train pool做架构扫描 (够区分好坏了)
np.random.seed(999)
tr_idx = np.random.choice(tr_pool, size=int(len(tr_pool)*0.3), replace=False)

yrs_te = u.iloc[te_idx]["year"].values; msk_te = MASK.iloc[te_idx].values
X_tr = X.iloc[tr_idx].values; y_tr = y[tr_idx]
X_te = X.iloc[te_idx].values; y_te = y[te_idx]; rets_te = rets[te_idx]
print(f"子采样训{len(tr_idx)}条 测{len(te_idx)}条 ({msk_te.sum()}条掩码内)")

archs = {
    "128x64_d02": nn.Sequential(nn.Linear(20,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5)),
    "256x128_d02": nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,5)),
    "128x128x64_d02": nn.Sequential(nn.Linear(20,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5)),
    "128x64_d04": nn.Sequential(nn.Linear(20,128),nn.ReLU(),nn.Dropout(0.4),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.4),nn.Linear(64,5)),
}

results = []
for name, net in archs.items():
    n_params = sum(p.numel() for p in net.parameters())
    Xs = torch.FloatTensor(StandardScaler().fit_transform(X_tr)); ys = torch.LongTensor(y_tr)
    opt = optim.AdamW(net.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]))
    bl, bs = 1e9, None
    for _ in range(5000):
        net.train(); opt.zero_grad(); loss = lfn(net(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in net.state_dict().items()}
    net.load_state_dict(bs); net.eval()
    
    Xte = torch.FloatTensor(StandardScaler().fit_transform(X_tr))
    Xte_t = torch.FloatTensor(StandardScaler().fit(X_tr).transform(X_te))
    with torch.no_grad(): pr = torch.argmax(net(Xte_t), dim=1).numpy()
    
    cm = confusion_matrix(y_te[msk_te], pr[msk_te])
    acc = sum(cm[i,i] for i in range(5))/cm.sum()*100
    buy = (pr == 4) & msk_te; nb = buy.sum(); br = rets_te[buy]
    yrly = []
    for ty in sorted(set(yrs_te)):
        b = (yrs_te == ty) & buy; n = b.sum(); a = rets_te[b]
        yl = a.mean()*100 if n > 0 else 0; yrly.append(yl)
    yr = np.mean(yrly)
    rec = cm[4,4]/cm[4].sum()*100
    print(f"{name:<20s} 参{n_params:>5d}  acc={acc:>5.1f}%  买{nb:>3d}只  年{yr:>+5.1f}%  召回{rec:>4.1f}%")
    results.append((name, n_params, acc, nb, yr, rec))
    torch.save({"model":bs,"scaler":StandardScaler().fit(X_tr)}, os.path.join(MDIR, f"ARCH_{name}.pt"))

print(f"\n{'架构':<20s} {'参数':>5s} {'acc':>6s} {'买入':>5s} {'年均':>6s} {'暴涨召回':>6s}")
for r in results:
    print(f"{r[0]:<20s} {r[1]:>5d} {r[2]:>5.1f}% {r[3]:>4d}只 {r[4]:>+5.1f}% {r[5]:>5.1f}%")
print(f"\n基准 ALL_14(5折): 966只 年+35.8%")
