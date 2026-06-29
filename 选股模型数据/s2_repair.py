"""S2修复版: PE周期修复+排除, 训练MLP0_5, S1测试对比"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")

df = pd.read_csv(os.path.join(OUT, "training_s2_repaired.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X_rep = pd.concat([u[feat], ind], axis=1); X_rep = X_rep.fillna(X_rep.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]

# 原始数据集 for comparison
df_orig = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u_orig = df_orig[df_orig.usable].copy()
u_orig["L"] = u_orig.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
X_orig = pd.concat([u_orig[feat], pd.get_dummies(u_orig.industry, prefix="ind").astype(float)], axis=1)
X_orig = X_orig.fillna(X_orig.median())
y_orig = u_orig.L.values
yrs_orig = u_orig.year.values
S1 = (u_orig.cfo_to_revenue >= 0) & (u_orig.current_ratio >= 0.5)

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
    ckpt = torch.load(path, weights_only=False); m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

# 训练修复版
print("训练 S2_repair...")
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    train_save(X_rep[tr].values, y[tr], os.path.join(MDIR, f"S2_repair_{ty}.pt"))
    print(f"  fold {ty}: 训{tr.sum()} 测{te.sum()}")

yrs_orig = u_orig.year.values
# 预测(用原始X, 因为S1是基于原始数据的)
pr_rep = np.zeros(len(u_orig), dtype=int)
for ty in TY:
    te = (yrs_orig == ty)
    pr_rep[te] = load_pred(X_orig[te].values, os.path.join(MDIR, f"S2_repair_{ty}.pt"))

# 对比
pr_old = np.zeros(len(u_orig), dtype=int)
for ty in TY:
    te = (yrs_orig == ty)
    pr_old[te] = load_pred(X_orig[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))

for name, p in [("MLP0_5原版→S1", pr_old), ("S2_repair→S1", pr_rep)]:
    cm = confusion_matrix(y_orig[S1], p[S1]); n = cm.sum()
    acc = sum(cm[i,i] for i in range(5)) / n * 100
    print(f"\n=== {name} ({n}条 acc={acc:.1f}%) ===")
    print(f"实际↓预测 |暴跌  跌 小涨 中涨 暴涨| 合计")
    for i, lb in enumerate(LAB):
        print(f"  {lb:6s}|{cm[i,0]:>4d} {cm[i,1]:>3d} {cm[i,2]:>4d} {cm[i,3]:>4d} {cm[i,4]:>4d}|{cm[i].sum():>5d}")
    buy = (p == 4) & S1; nb = buy.sum(); br = u_orig.forward_return.values[buy]
    yrly = [u_orig.forward_return.values[(yrs_orig==ty)&buy].mean()*100 for ty in TY if sum((yrs_orig==ty)&buy)>0]
    print(f"买入: {nb}只 均{br.mean()*100:+.1f}% 年{np.mean(yrly):+.1f}%")
