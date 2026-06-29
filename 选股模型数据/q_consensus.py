"""季度MLP∩RF共同买入"""
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u = u.sort_values(["code", "year", "quarter"])
u["next_price"] = u.groupby("code")["buy_price"].shift(-1)
u["q_return"] = u.next_price / u.buy_price - 1
u = u[u.q_return.notna()].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.q_return.apply(lambda r: 0 if r < -0.05 else (1 if r < 0 else (2 if r < 0.05 else (3 if r < 0.15 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","L","usable","industry","q_return","next_price","gm_pct"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.q_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8)

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def load_mlp(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

pr_mlp = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); pr_mlp[te] = load_mlp(X[te].values, os.path.join(MDIR, f"Q_MLP0_5_{ty}.pt"))

print("训练 Q_RF...")
pr_rf = np.zeros(len(u), dtype=int)
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    rf = RandomForestClassifier(100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(X[tr].values, y[tr]); pickle.dump(rf, open(os.path.join(MDIR, f"Q_RF_{ty}.pkl"), "wb"))
    pr_rf[te] = rf.predict(X[te].values)
    print(f"  fold {ty} done")

for name, p in [("Q_MLP", pr_mlp), ("Q_RF", pr_rf)]:
    buy = (p == 4) & MASK; nb = buy.sum(); br = rets[buy]
    cm = confusion_matrix(y[MASK], p[MASK])
    acc = sum(cm[i,i] for i in range(5)) / cm.sum() * 100
    print(f"\n{name}: acc={acc:.1f}% 买{nb}只 季均{br.mean()*100:.2f}%  暴涨召回{cm[4,4]}/{cm[4].sum()}")

# 共同
both = (pr_mlp == 4) & (pr_rf == 4) & MASK
nb = both.sum(); br = rets[both]
print(f"\n=== Q_MLP∩RF 共同买入 ===")
print(f"共{nb}只 季均{br.mean()*100:.2f}%  年化≈{((1+br.mean())**4-1)*100:.1f}%")
for ty in TY:
    b = (yrs == ty) & both; n = b.sum()
    a = rets[b]
    print(f"  {ty}: {n}只 季均{a.mean()*100:.2f}%" if n > 0 else f"  {ty}: 未买")
