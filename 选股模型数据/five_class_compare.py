"""5分类对比: 不同惩罚 + 线性 + RF, 只买暴涨组"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
RES = os.path.join(OUT, "five_class_results.txt")
def log(msg):
    print(msg, flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
with open(RES, "w", encoding="utf-8") as f: f.write("")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()

def c5(ret):
    if ret < -0.10: return 0
    if ret < 0: return 1
    if ret < 0.10: return 2
    if ret < 0.30: return 3
    return 4
u["label"] = u.forward_return.apply(c5)

feat = [c for c in u.columns if c not in
    ["code","year","quarter","forward_return","label","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.label.values; yrs = u.year.values

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 5)
        )
    def forward(self, x): return self.net(x)

def train_mlp(Xtr_np, ytr_np, Xte_np, w4):
    sc = StandardScaler()
    Xtr = torch.FloatTensor(sc.fit_transform(Xtr_np))
    Xte = torch.FloatTensor(sc.transform(Xte_np))
    ytr = torch.LongTensor(ytr_np)
    m = M5(Xtr.shape[1])
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    w = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, w4])
    lfn = nn.CrossEntropyLoss(weight=w)
    bl, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad()
        loss = lfn(m(Xtr), ytr)
        loss.backward(); opt.step()
        if loss.item() < bl:
            bl = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): return torch.argmax(m(Xte), dim=1).numpy()

def backtest(preds, rets, yrs):
    buy = pd.DataFrame({"pred": preds, "ret": rets, "year": yrs})
    all_avg = buy.ret.mean()
    bought = buy[buy.pred == 4]
    n = len(bought)
    avg_r = bought.ret.mean()
    med_r = bought.ret.median()
    win = (bought.ret > 0).mean() * 100
    exc = avg_r - all_avg
    total = 1.0
    for ty in [2020,2021,2022,2023,2024]:
        b = buy[(buy.year == ty) & (buy.pred == 4)]
        if len(b) > 0: total *= 1 + b.ret.mean()
    return n, avg_r, med_r, win, exc, total

log(f"{'方法':<18s} {'买入':>5s} {'均值':>7s} {'中位':>7s} {'正收益':>7s} {'超额':>7s} {'5年复利':>8s}")
log("-" * 75)

# 线性回归
all_preds, all_rets, all_years = [], [], []
for ty in [2020,2021,2022,2023,2024]:
    tr = (yrs != ty); te = (yrs == ty)
    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(X[tr].values, y[tr])
    p = lr.predict(X[te].values)
    all_preds.extend(p); all_rets.extend(u.loc[te,"forward_return"].values); all_years.extend(u.loc[te,"year"].values)
n, ar, mr, wr, ex, tot = backtest(all_preds, all_rets, all_years)
log(f"{'逻辑回归':<18s} {n:>5d} {ar*100:>+6.1f}% {mr*100:>+6.1f}% {wr:>6.1f}% {ex*100:>+6.1f}% {tot:>7.3f}x")

# 随机森林
all_preds, all_rets, all_years = [], [], []
for ty in [2020,2021,2022,2023,2024]:
    tr = (yrs != ty); te = (yrs == ty)
    rf = RandomForestClassifier(100, max_depth=5, min_samples_leaf=10, random_state=42)
    rf.fit(X[tr].values, y[tr])
    p = rf.predict(X[te].values)
    all_preds.extend(p); all_rets.extend(u.loc[te,"forward_return"].values); all_years.extend(u.loc[te,"year"].values)
n, ar, mr, wr, ex, tot = backtest(all_preds, all_rets, all_years)
log(f"{'随机森林':<18s} {n:>5d} {ar*100:>+6.1f}% {mr*100:>+6.1f}% {wr:>6.1f}% {ex*100:>+6.1f}% {tot:>7.3f}x")

# MLP 不同惩罚
for w4, desc in [(1,"MLP 无惩罚"), (2,"MLP 暴涨×2"), (3,"MLP 暴涨×3"), (4,"MLP 暴涨×4"), (5,"MLP 暴涨×5")]:
    all_preds, all_rets, all_years = [], [], []
    for ty in [2020,2021,2022,2023,2024]:
        tr = (yrs != ty); te = (yrs == ty)
        p = train_mlp(X[tr].values, y[tr], X[te].values, w4)
        all_preds.extend(p); all_rets.extend(u.loc[te,"forward_return"].values); all_years.extend(u.loc[te,"year"].values)
    n, ar, mr, wr, ex, tot = backtest(all_preds, all_rets, all_years)
    print(f"{desc:<18s} {n:>5d} {ar*100:>+6.1f}% {mr*100:>+6.1f}% {wr:>6.1f}% {ex*100:>+6.1f}% {tot:>7.3f}x")
