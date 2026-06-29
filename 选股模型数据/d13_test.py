"""消融实验: 去掉current_ratio和interest_coverage, 13维训练"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
# 去掉 current_ratio 和 interest_coverage
DROP = ["current_ratio", "interest_coverage"]
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"] + DROP]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1)
print(f"特征: {X.shape[1]}维 (原15→丢2+4行业={X.shape[1]})")

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def train_save(Xtr, ytr, path):
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(Xtr)); ys = torch.LongTensor(ytr)
    m = M5(Xs.shape[1]); opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]))
    bl, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval(); torch.save({"model":bs,"scaler":sc}, path)

def load_pred(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

print("训练 D13_MLP...")
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    train_save(X[tr].values, y[tr], os.path.join(MDIR, f"D13_{ty}.pt"))
    print(f"  fold {ty}: 训{tr.sum()} 测{te.sum()}")

pr = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); pr[te] = load_pred(X[te].values, os.path.join(MDIR, f"D13_{ty}.pt"))

cm = confusion_matrix(y[MASK], pr[MASK]); n = cm.sum()
acc = sum(cm[i,i] for i in range(5)) / n * 100
print(f"\n=== D13(13维)→S1∩S3宽松 ({n}条 acc={acc:.1f}%) ===")
for i, lb in enumerate(LAB):
    print(f"  {lb:6s}|{cm[i,0]:>4d} {cm[i,1]:>3d} {cm[i,2]:>4d} {cm[i,3]:>4d} {cm[i,4]:>4d}|{cm[i].sum():>5d}")
buy = (pr == 4) & MASK; nb = buy.sum(); br = rets[buy]
yrly = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
print(f"买入: {nb}只 均{br.mean()*100:.1f}% 年{np.mean(yrly):+.1f}%")

# 对比基准
print(f"\n=== 对比基准(15维) ===")
pr15 = np.zeros(len(u), dtype=int)
feat15 = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind15 = pd.get_dummies(u.industry, prefix="ind").astype(float)
X15 = pd.concat([u[feat15], ind15], axis=1); X15 = X15.fillna(X15.median())
for ty in TY:
    te = (yrs == ty); pr15[te] = load_pred(X15[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
cm15 = confusion_matrix(y[MASK], pr15[MASK])
acc15 = sum(cm15[i,i] for i in range(5)) / cm15.sum() * 100
buy15 = (pr15 == 4) & MASK; nb15 = buy15.sum(); br15 = rets[buy15]
yr15 = [rets[(yrs==ty)&buy15].mean()*100 for ty in TY if sum((yrs==ty)&buy15)>0]
print(f"基准: acc={acc15:.1f}% 买{nb15}只 均{br15.mean()*100:.1f}% 年{np.mean(yr15):+.1f}%")
print(f"差值: 年化{np.mean(yrly)-np.mean(yr15):+.1f}pp 买入{nb-nb15:+d}只")
