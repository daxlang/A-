"""CART ALL vs RAND_ALL 评估对比"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
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
te_idx = idx[int(n*0.8):]; yrs_te = u.iloc[te_idx]["year"].values; msk_te = MASK.iloc[te_idx].values
X_te = X.iloc[te_idx].values; y_te = y[te_idx]; rets_te = rets[te_idx]

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def eval_model(wtag):
    ckpt = torch.load(os.path.join(MDIR, wtag), weights_only=False)
    m = M5(X_te.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xte = torch.FloatTensor(ckpt["scaler"].transform(X_te))
    with torch.no_grad(): pr = torch.argmax(m(Xte), dim=1).numpy()
    return pr

pr_rand = eval_model("RAND_ALL.pt")
pr_cart = eval_model("CART_ALL.pt")

INDS = sorted(u.industry.unique())
for tag, pr in [("RAND_ALL(基准)", pr_rand), ("CART_ALL(去Hard)", pr_cart)]:
    cm = confusion_matrix(y_te[msk_te], pr[msk_te]); n_cm = cm.sum()
    acc = sum(cm[i,i] for i in range(5))/n_cm*100
    buy = (pr == 4) & msk_te; nb = buy.sum(); br = rets_te[buy]
    yrly = []
    for ty in sorted(set(yrs_te)):
        b = (yrs_te == ty) & buy; n_b = b.sum(); a = rets_te[b]
        yl = a.mean()*100 if n_b > 0 else 0; yrly.append(yl)
    yr = np.mean(yrly)
    
    print(f"\n{'='*60}")
    print(f"=== {tag} ({n_cm}条 acc={acc:.1f}%) ===")
    print(f"  {'实际':<6s}{'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
    for i, lb in enumerate(LAB):
        print(f"  {lb:<6s}{cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
    print(f"  买入: {nb}只 均{br.mean()*100:.1f}% 年{yr:.1f}% 暴涨召回{cm[4,4]}/{cm[4].sum()}")
    for ty in sorted(set(yrs_te)):
        b = (yrs_te == ty) & buy; n_b = b.sum(); a = rets_te[b]
        print(f"    {ty}: {n_b}只 {a.mean()*100:.1f}%" if n_b > 0 else f"    {ty}: 未买")
    
    # 逐年x行业
    print(f"\n  逐年x行业:")
    hdr = "  {:<6s}".format("行业")
    for ty in sorted(set(yrs_te)): hdr += "  {:>5d}".format(ty)
    hdr += "  {:>5s}".format("合计")
    print(hdr)
    for ind_name in INDS:
        row = "  {:<6s}".format(ind_name)
        tn = 0; tr = []
        for ty in sorted(set(yrs_te)):
            b = (yrs_te == ty) & (pr == 4) & msk_te & (u.industry.iloc[te_idx].values == ind_name)
            nb = b.sum(); a = rets_te[b]
            ra = a.mean()*100 if nb > 0 else 0
            row += " {:>3d}只{:>+3.0f}%".format(nb, ra)
            tn += nb; tr.extend(a)
        re = np.mean(tr)*100 if tn > 0 else 0
        row += " {:>3d}只{:>+3.0f}%".format(tn, re)
        print(row)
    row = "  {:<6s}".format("合计")
    tn = 0; tr = []
    for ty in sorted(set(yrs_te)):
        b = (yrs_te == ty) & (pr == 4) & msk_te; nb = b.sum(); a = rets_te[b]
        ra = a.mean()*100 if nb > 0 else 0
        row += " {:>3d}只{:>+3.0f}%".format(nb, ra); tn += nb; tr.extend(a)
    row += " {:>3d}只{:>+3.0f}%".format(tn, np.mean(tr)*100)
    print(row)
