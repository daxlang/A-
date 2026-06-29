"""纯MLP0_5 vs S1_PASS 五年混淆矩阵"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1)
X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
PASS = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5)
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 5)
        )
    def forward(self, x): return self.net(x)

def load_predict(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

pure = np.zeros(len(u), dtype=int)
sp = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty)
    pure[te] = load_predict(X[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
    te_p = te & PASS
    sp[te_p] = load_predict(X[te_p].values, os.path.join(MDIR, f"S1_PASS_{ty}.pt"))

for name, pr in [("纯MLP0_5在PASS子集上", pure), ("S1_PASS分组建模", sp)]:
    cm = confusion_matrix(y[PASS], pr[PASS])
    n = cm.sum(); acc = sum(cm[i,i] for i in range(5))/n*100
    print(f"\n=== {name} ({n}条 PASS组 准确率={acc:.1f}%) ===")
    print(f"实际↓预测 |暴跌  跌 小涨 中涨 暴涨| 合计")
    for i, lb in enumerate(LAB):
        print(f"  {lb:6s}|{cm[i,0]:>4d} {cm[i,1]:>3d} {cm[i,2]:>4d} {cm[i,3]:>4d} {cm[i,4]:>4d}|{cm[i].sum():>5d}")

    buy = (pr == 4) & PASS
    nb = buy.sum(); b_ret = rets[buy]
    print(f"买入暴涨: {nb}只 均{b_ret.mean()*100:+.1f}% 中{np.median(b_ret)*100:+.1f}%")
    yrly = []
    for ty in TY:
        m_ty = (yrs == ty) & buy
        r = rets[m_ty]
        yrly.append(r.mean()*100 if len(r)>0 else 0)
    print(f"年均: {np.mean(yrly):+.1f}%")
