"""纯MLP0_5 分行业混淆矩阵+回测"""
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
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
INDS = u.industry.values

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 5)
        )
    def forward(self, x): return self.net(x)

pr = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty)
    ckpt = torch.load(os.path.join(MDIR, f"MLP0_5_{ty}.pt"), weights_only=False)
    m = M5(X[te].shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(X[te].values))
    with torch.no_grad(): pr[te] = torch.argmax(m(Xs), dim=1).numpy()

for ind_name in ["钢铁", "白酒", "银行", "游戏"]:
    mask = (INDS == ind_name)
    cm = confusion_matrix(y[mask], pr[mask]); n = cm.sum()
    acc = sum(cm[i,i] for i in range(5)) / n * 100
    buy = (pr == 4) & mask; nb = buy.sum(); b_ret = rets[buy]
    yrly = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
    print(f"\n=== {ind_name}({n}条 准确率={acc:.1f}%) ===")
    print(f"实际↓预测 |暴跌 跌 小涨 中涨 暴涨| 合计")
    for i, lb in enumerate(LAB):
        print(f"  {lb:6s}|{cm[i,0]:>4d} {cm[i,1]:>3d} {cm[i,2]:>4d} {cm[i,3]:>4d} {cm[i,4]:>4d}|{cm[i].sum():>5d}")
    print(f"买入: {nb}只 均{b_ret.mean()*100:+.1f}% 中{np.median(b_ret)*100:+.1f}% 年均{np.mean(yrly):+.1f}%")
