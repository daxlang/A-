"""阶段1预筛: 用已保存MLP0_5权重, 测试时先筛掉生存不合格的"""
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
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]

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
    with torch.no_grad():
        return torch.argmax(m(Xs), dim=1).numpy()

# 阶段1筛子
filters = {
    "无筛选": np.ones(len(u), dtype=bool),
    "cfo>0": u.cfo_to_revenue.values >= 0,
    "cfo>0+流动>0.5": (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5),
    "cfo>0+流动>0.5+利息>1": (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.interest_coverage >= 1),
}

print(f"{'筛子':<22s} {'剩余':>5s} {'买入':>5s} {'均值':>7s} {'中位':>7s} {'正收益':>7s} {'年均':>7s}")
print("-" * 75)

for fname, fmask in filters.items():
    preds = np.zeros(len(u), dtype=int)
    for ty in TY:
        te = (yrs == ty)
        keep = fmask[te]
        if keep.sum() == 0:
            continue
        idx_te = np.where(te)[0]
        idx_keep = idx_te[keep]
        p = predict(X.values[idx_keep], os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
        preds[idx_keep] = p

    buy = pd.DataFrame({"pred": preds, "ret": rets, "year": yrs})
    bought = buy[buy.pred == 4]
    n = len(bought); avg = bought.ret.mean(); med = bought.ret.median()
    win = (bought.ret > 0).mean()*100
    n_kept = fmask.sum()
    yrly = []
    for ty in TY:
        b = buy[(buy.year == ty) & (buy.pred == 4)]
        if len(b) > 0: yrly.append(b.ret.mean())
    ayr = np.mean(yrly) if yrly else 0

    print(f"{fname:<22s} {n_kept:>5d} {n:>5d} {avg*100:>+6.1f}% {med*100:>+6.1f}% {win:>6.1f}% {ayr*100:>+6.1f}%")

print(f"\n对比 纯MLP0_5:                  337  +20.7%   +8.7%   57.3%  +19.8%")
