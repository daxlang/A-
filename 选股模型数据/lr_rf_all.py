"""ALL: LR + RF 5折 + 共识买入 对比"""
import os, numpy as np, pandas as pd, pickle, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings, time; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])

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

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def ld(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

# 加载CART_ALL预测
print("加载CART_ALL权重...")
pr_cart = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); Xte = X.iloc[te].values
    pr_cart[te] = ld(Xte, os.path.join(MDIR, f"CART_ALL_{ty}.pt"))

# LR 训练 (5折)
print("\n训练Logistic Regression(5折)...")
pr_lr = np.zeros(len(u), dtype=int)
cw = {0:1, 1:1, 2:1, 3:1, 4:0.5}
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    sc = StandardScaler(); Xtr = sc.fit_transform(X.iloc[tr].values); Xte = sc.transform(X.iloc[te].values)
    lr = LogisticRegression(max_iter=5000, multi_class="multinomial", class_weight=cw, C=0.1)
    lr.fit(Xtr, y[tr])
    pr_lr[te] = lr.predict(Xte)
    with open(os.path.join(MDIR, f"LR_ALL_{ty}.pkl"), "wb") as f: pickle.dump(lr, f)
    print(f"  fold {ty} done")

# RF 训练 (5折)
print("\n训练Random Forest(5折)...")
pr_rf = np.zeros(len(u), dtype=int)
for ty in TY:
    t0 = time.time()
    tr = (yrs != ty); te = (yrs == ty)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=50,
                                 class_weight=cw, n_jobs=-1, random_state=42)
    rf.fit(X.iloc[tr].values, y[tr])
    pr_rf[te] = rf.predict(X.iloc[te].values)
    with open(os.path.join(MDIR, f"RF_ALL_{ty}.pkl"), "wb") as f: pickle.dump(rf, f)
    print(f"  fold {ty} done ({time.time()-t0:.0f}s)")

# 共识买入
mk = MASK
P = {"CART_MLP": pr_cart, "LR": pr_lr, "RF": pr_rf}
consensus = {
    "CART∩LR": (pr_cart == 4) & (pr_lr == 4),
    "CART∩RF": (pr_cart == 4) & (pr_rf == 4),
    "LR∩RF":   (pr_lr == 4) & (pr_rf == 4),
    "三模型共识": (pr_cart == 4) & (pr_lr == 4) & (pr_rf == 4),
}

print("\n" + "="*70)
print("单模型 + 共识对比 (S1∩S3∩S4):")
print("="*70)
for tag, p in [("CART_MLP", pr_cart), ("LR", pr_lr), ("RF", pr_rf)] + list(consensus.items()):
    buy = p if isinstance(p, np.ndarray) and p.dtype == bool else (p == 4)
    if isinstance(p, np.ndarray) and p.dtype == bool:
        buy_final = buy & mk
    else:
        buy_final = buy & mk
    n = buy_final.sum(); br = rets[buy_final]
    yrly = []
    for ty in TY:
        b = buy_final & (yrs == ty); nb = b.sum(); a = rets[b]
        yl = a.mean()*100 if nb > 0 else 0; yrly.append(yl)
    yr = np.mean(yrly)
    yearly_str = ""
    for ty in TY:
        b = buy_final & (yrs == ty); nb = b.sum(); a = rets[b]
        yearly_str += f" {ty}:{nb}只{a.mean()*100:.0f}%" if nb > 0 else f" {ty}:0只"
    print(f"{tag:<12s} 买{n:>4d}只 均{br.mean()*100:>5.1f}% 年{yr:>+5.1f}%{yearly_str}")

# 逐年共识详细
print(f"\n{'='*70}")
print("逐年共识买入详情:")
for pair_name, pair_mask in [("CART∩LR", consensus["CART∩LR"]),
                              ("CART∩RF", consensus["CART∩RF"]),
                              ("三模型共识", consensus["三模型共识"])]:
    print(f"\n{pair_name}:")
    for ty in TY:
        b = pair_mask & mk & (yrs == ty); n = b.sum(); a = rets[b]
        print(f"  {ty}: {n}只 均{a.mean()*100:.1f}%" if n > 0 else f"  {ty}: 0只")

# CART ALL逐年x行业 for reference
print(f"\n{'='*70}")
print("CART_ALL逐行业逐年 (参考):")
INDS = sorted(u.industry.unique())
for ind_name in INDS:
    row = f"  {ind_name:<6s}"; tn = 0; tr = []
    for ty in TY:
        b = (yrs == ty) & (pr_cart == 4) & mk & (u.industry == ind_name)
        n = b.sum(); a = rets[b]
        ra = a.mean()*100 if n > 0 else 0
        row += " {:>3d}只{:>+3.0f}%".format(n, ra); tn += n; tr.extend(a)
    re = np.mean(tr)*100 if tn > 0 else 0
    row += " {:>3d}只{:>+3.0f}%".format(tn, re); print(row)
