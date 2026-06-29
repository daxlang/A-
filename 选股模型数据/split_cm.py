"""分组建模: 读S1权重 → 混淆矩阵"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")

df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
def c5(r):
    if r < -0.1: return 0
    if r < 0: return 1
    if r < 0.1: return 2
    if r < 0.3: return 3
    return 4
u["label"] = u.forward_return.apply(c5)
feat = [c for c in u.columns if c not in
    ["code","year","quarter","forward_return","label","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1)
X = X.fillna(X.median())
y = u.label.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
PASS = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5)

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 5)
        )
    def forward(self, x): return self.net(x)

def predict(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

# 逐 fold 预测
preds = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty)
    for gname, gmask in [("PASS", PASS), ("FAIL", ~PASS)]:
        te_g = te & gmask
        if te_g.sum() == 0: continue
        path = os.path.join(MDIR, f"S1_{gname}_{ty}.pt")
        if not os.path.exists(path): continue
        preds[te_g] = predict(X[te_g].values, path)

buy = pd.DataFrame({"pred": preds, "ret": rets, "year": yrs})
LAB = ["暴跌","跌","小涨","中涨","暴涨"]

print("逐年 + 分组混淆矩阵:")
for ty in TY:
    te = (yrs == ty)
    for gname, gmask in [("PASS", PASS), ("FAIL", ~PASS)]:
        te_g = te & gmask
        if te_g.sum() == 0: continue
        cm = confusion_matrix(y[te_g], preds[te_g]); n_g = cm.sum()
        acc = sum(cm[i,i] for i in range(5))/n_g*100
        print(f"\n  {ty}年 {gname}({n_g}只) 准确率={acc:.1f}%")
        print(f"  实际↓预测 |暴跌 跌 小涨 中涨 暴涨|计")
        for i, lb in enumerate(LAB):
            print(f"    {lb:6s}|{cm[i,0]:>3d} {cm[i,1]:>2d} {cm[i,2]:>3d} {cm[i,3]:>3d} {cm[i,4]:>3d}|{cm[i].sum():>4d}")
        bought = buy[(buy.year==ty) & (buy.pred==4) & gmask]
        n_b = len(bought)
        print(f"  买入: {n_b}只 均{bought.ret.mean()*100:+.1f}%" if n_b>0 else f"  未买入")

    print(f"\nPASS+FAIL 合计: 测试{len(preds)}条 = 每年520只×5年")
