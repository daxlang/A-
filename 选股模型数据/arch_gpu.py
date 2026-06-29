"""ALL架构扫描 GPU: 128x64 vs 256x128 vs 256x128x64"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, time
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

def make_arch(name):
    if name == "128x64":
        return nn.Sequential(nn.Linear(20,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    elif name == "256x128":
        return nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,5))
    elif name == "256x128x64":
        return nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))

arch_names = ["128x64","256x128","256x128x64"]
results = {}
for arch_name in arch_names:
    print(f"\nArch: {arch_name}")
    pr_all = np.zeros(len(u), dtype=int); t0 = time.time()
    
    for ty in TY:
        tr = (yrs != ty); te = (yrs == ty)
        sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X.iloc[tr].values)).to(DEV)
        ys = torch.LongTensor(y[tr]).to(DEV)
        m = make_arch(arch_name).to(DEV)
        opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
        lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
        bl, bs = 1e9, None
        for ep in range(1, 5001):
            m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
            if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
        m.load_state_dict(bs); m = m.cpu(); 
        bs_cpu = {k:v.cpu() for k,v in bs.items()}
        torch.save({"model":bs_cpu,"scaler":sc}, os.path.join(MDIR, f"ARCH_{arch_name}_{ty}.pt"))
        
        m_eval = make_arch(arch_name); m_eval.load_state_dict(bs_cpu); m_eval.eval()
        Xte = torch.FloatTensor(sc.transform(X.iloc[te].values))
        with torch.no_grad(): pr_all[te] = torch.argmax(m_eval(Xte), dim=1).numpy()
        print(f"  fold {ty} done")
    
    cm = confusion_matrix(y[MASK], pr_all[MASK]); n_cm = cm.sum()
    acc = sum(cm[i,i] for i in range(5))/n_cm*100
    buy = (pr_all == 4) & MASK; nb = buy.sum(); br = rets[buy]
    yr = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
    elapsed = time.time()-t0
    rec = cm[4,4]/cm[4].sum()*100
    print(f"  acc={acc:.1f}% 买{nb}只 年{np.mean(yr):.1f}% 召回{cm[4,4]}/{cm[4].sum()}")
    
    results[arch_name] = {"params":sum(p.numel() for p in make_arch(arch_name).parameters()),
                          "acc":acc,"buys":nb,"yr":np.mean(yr),"rec":rec,"time":elapsed}

print(f"\n{'='*60}")
print(f"{'架构':<15s} {'参数':>6s} {'acc':>6s} {'买入':>5s} {'年均':>6s} {'召回':>6s} {'时间':>6s}")
for name, r in results.items():
    print(f"{name:<15s} {r['params']:>6d} {r['acc']:>5.1f}% {r['buys']:>4d}只 {r['yr']:>+5.1f}% {r['rec']:>5.1f}% {r['time']:>5.0f}s")
print(f"\n基准 ALL_14: 966买 +35.8%")
print(f"基准 RF: 425买 +40.8%")
print(f"基准 CART_ALL: 1284买 +39.7%")
