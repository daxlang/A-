"""5分类+正确惩罚方向: 压低暴涨类权重使模型更保守"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
RES = os.path.join(OUT, "five_correct_penalty.txt")
def log(msg):
    print(msg, flush=True)
    with open(RES, "a", encoding="utf-8") as f: f.write(msg + "\n")
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
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 1.0, 1.0, 1.0, w4]))
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

def eval_model(name, preds, rets, yrs):
    buy = pd.DataFrame({"pred": preds, "ret": rets, "year": yrs})
    all_avg = buy.ret.mean()
    cm = confusion_matrix(np.array([c5(r) for r in rets]), preds)
    acc = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4])/cm.sum()*100
    log(f"\n{'='*60}")
    log(f"【{name}】 (混淆矩阵 {cm.sum()}条, 准确率={acc:.1f}%)")
    log(f"实际->预测 |暴跌| 跌 |小涨|中涨|暴涨|合计")
    for i, nm in enumerate(["暴跌","跌","小涨","中涨","暴涨"]):
        log(f"  {nm:6s}|{cm[i,0]:>4d}|{cm[i,1]:>4d}|{cm[i,2]:>4d}|{cm[i,3]:>4d}|{cm[i,4]:>4d}|{cm[i].sum():>5d}")
    bought = buy[buy.pred == 4]
    n = len(bought); avg_r = bought.ret.mean(); med_r = bought.ret.median()
    win = (bought.ret > 0).mean()*100; exc = avg_r - all_avg
    log(f"买入暴涨组: {n}次  全市场均{all_avg*100:+.1f}%")
    log(f"均值{avg_r*100:+.1f}% 中位{med_r*100:+.1f}% 正收益{win:.1f}% 超额{exc*100:+.1f}%")
    log("逐年:")
    total = 1.0
    for ty in [2020,2021,2022,2023,2024]:
        b = buy[(buy.year == ty) & (buy.pred == 4)]
        a = buy[buy.year == ty]
        if len(b) > 0:
            ex = b.ret.mean() - a.ret.mean()
            total *= 1 + b.ret.mean()
            log(f"  {ty}: 全{a.ret.mean()*100:+.1f}% 买{len(b)}只 均{b.ret.mean()*100:+.1f}% 中{b.ret.median()*100:+.1f}% 正{(b.ret>0).mean()*100:.0f}% 超额{ex*100:+.1f}%")
        else:
            log(f"  {ty}: 未买入")
    log(f"  5年复利: {total:.3f}x = {(total-1)*100:.1f}%\n")

# 正确方向: w4 < 1 使模型更保守
for w4, desc in [(1.0, "MLP无惩罚"), (0.5, "MLP暴涨×0.5"), (0.25, "MLP暴涨×0.25"), (0.1, "MLP暴涨×0.1")]:
    all_p, all_r, all_y = [], [], []
    for ty in [2020,2021,2022,2023,2024]:
        tr = (yrs != ty); te = (yrs == ty)
        p = train_mlp(X[tr].values, y[tr], X[te].values, w4)
        all_p.extend(p); all_r.extend(u.loc[te,"forward_return"].values); all_y.extend(u.loc[te,"year"].values)
    eval_model(desc, all_p, all_r, all_y)

# 线性 + RF
for name, clf in [("线性回归", LogisticRegression(max_iter=5000, random_state=42)),
                    ("随机森林", RandomForestClassifier(100, max_depth=5, min_samples_leaf=10, random_state=42))]:
    all_p, all_r, all_y = [], [], []
    for ty in [2020,2021,2022,2023,2024]:
        tr = (yrs != ty); te = (yrs == ty)
        clf.fit(X[tr].values, y[tr])
        p = clf.predict(X[te].values)
        all_p.extend(p); all_r.extend(u.loc[te,"forward_return"].values); all_y.extend(u.loc[te,"year"].values)
    eval_model(name, all_p, all_r, all_y)

log("DONE")
