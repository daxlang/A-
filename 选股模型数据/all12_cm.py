"""全部12个混淆矩阵: 4行业 × 3模型"""
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
INDS = u.industry.values
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

# 三种模型的全量预测
all_preds = {}
for name, prefix in [("MLP0_5", "MLP0_5"), ("S1_PASS", "S1_PASS"), ("S1_PASS_w0.5", "S1_PASS_w0.5")]:
    pr = np.zeros(len(u), dtype=int)
    for ty in TY:
        te = (yrs == ty)
        pr[te] = load_predict(X[te].values, os.path.join(MDIR, f"{prefix}_{ty}.pt"))
    all_preds[name] = pr

for ind_name in ["钢铁", "白酒", "银行", "游戏"]:
    mask = (INDS == ind_name); n = mask.sum()
    print(f"\n{'='*70}")
    print(f"{ind_name}({n}条) 三模型对比:")
    print(f"{'实际↓预测 |暴跌 跌 小涨 中涨 暴涨| 合计':>35s}  {'买入':>6s} {'均值':>7s} {'中位':>7s} {'年均':>7s}")
    print("-" * 70)
    for m_name, pr in all_preds.items():
        cm = confusion_matrix(y[mask], pr[mask])
        line0 = f"  暴跌 |{cm[0,0]:>4d} {cm[0,1]:>3d} {cm[0,2]:>4d} {cm[0,3]:>4d} {cm[0,4]:>4d}|{cm[0].sum():>5d}"
        line1 = f"  跌   |{cm[1,0]:>4d} {cm[1,1]:>3d} {cm[1,2]:>4d} {cm[1,3]:>4d} {cm[1,4]:>4d}|{cm[1].sum():>5d}"
        line2 = f"  小涨 |{cm[2,0]:>4d} {cm[2,1]:>3d} {cm[2,2]:>4d} {cm[2,3]:>4d} {cm[2,4]:>4d}|{cm[2].sum():>5d}"
        line3 = f"  中涨 |{cm[3,0]:>4d} {cm[3,1]:>3d} {cm[3,2]:>4d} {cm[3,3]:>4d} {cm[3,4]:>4d}|{cm[3].sum():>5d}"
        line4 = f"  暴涨 |{cm[4,0]:>4d} {cm[4,1]:>3d} {cm[4,2]:>4d} {cm[4,3]:>4d} {cm[4,4]:>4d}|{cm[4].sum():>5d}"
        buy = (pr == 4) & mask; nb = buy.sum(); br = rets[buy]
        yrly = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
        tag = f"{m_name:<12s}"
        print(f"{tag}{line0}  {nb:>5d} {br.mean()*100:>+6.1f}% {np.median(br)*100:>+6.1f}% {np.mean(yrly):>+6.1f}%")
        print(f"{' ':<12s}{line1}")
        print(f"{' ':<12s}{line2}")
        print(f"{' ':<12s}{line3}")
        print(f"{' ':<12s}{line4}")
    print("-" * 70)
