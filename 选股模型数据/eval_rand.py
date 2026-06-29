"""RAND 3模型评估"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
SEED = 42

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def eval_one(name, csv_file, wtag, mask_func, use_ic=False):
    df = pd.read_csv(os.path.join(OUT, csv_file), dtype={"code": str})
    u = df[df.usable].copy()
    u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
    u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
    
    DROP = [] if use_ic else ["interest_coverage"]
    feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"] + DROP]
    ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
    X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
    y = u.L.values; rets = u.forward_return.values
    
    np.random.seed(SEED)
    n = len(u); idx = np.random.permutation(n)
    tr_idx = idx[:int(n*0.8)]; te_idx = idx[int(n*0.8):]
    
    ckpt = torch.load(os.path.join(MDIR, wtag), weights_only=False)
    m = M5(X.iloc[te_idx].shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xte = torch.FloatTensor(ckpt["scaler"].transform(X.iloc[te_idx].values))
    with torch.no_grad(): pr_te = torch.argmax(m(Xte), dim=1).numpy()
    
    y_te = y[te_idx]; rets_te = rets[te_idx]
    msk = mask_func(u.iloc[te_idx]).values
    yrs_te = u.iloc[te_idx]["year"].values
    
    cm = confusion_matrix(y_te[msk], pr_te[msk]); n = cm.sum()
    acc = sum(cm[i,i] for i in range(5)) / n * 100
    print(f"\n=== {name} ({n}条 acc={acc:.1f}%) ===")
    for i, lb in enumerate(LAB):
        print(f"  {lb:6s}|{cm[i,0]:>5d} {cm[i,1]:>4d} {cm[i,2]:>5d} {cm[i,3]:>5d} {cm[i,4]:>5d}|{cm[i].sum():>5d}")
    buy = (pr_te == 4) & msk; nb = buy.sum(); br = rets_te[buy]
    yr_returns = []
    for ty in sorted(set(yrs_te)):
        b = (yrs_te == ty) & buy; n_b = b.sum(); a = rets_te[b]
        yl = a.mean()*100 if n_b > 0 else 0; yr_returns.append(yl)
        print(f"  {ty}: {n_b}只 {a.mean()*100:.1f}%" if n_b > 0 else f"  {ty}: 未买")
    r = np.mean(yr_returns)
    print(f"买入: {nb}只 均{br.mean()*100:.1f}% 年均{r:.1f}% 暴涨召回{cm[4,4]}/{cm[4].sum()}")
    return r

def m4(u): return (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1)
def m8(u): return m4(u) & (u.liability_to_asset <= 0.8)

r4 = eval_one("RAND_4(去IC)", "training_final.csv", "RAND_4.pt", m4)
r8 = eval_one("RAND_8(去IC)", "training_full.csv", "RAND_8.pt", m8)
ra = eval_one("RAND_ALL(去IC)", "training_all.csv", "RAND_ALL.pt", m8)

print(f"\n汇总: RAND_4={r4:.1f}%  RAND_8={r8:.1f}%  RAND_ALL={ra:.1f}%")
print(f"原版(5折): MLP0_5=+21.8%  D14_8=+25.6%  ALL_14=+35.8%")
