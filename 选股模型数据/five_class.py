"""5分类: <-10%/-10~0%/0~10%/10~30%/>30% → 策略回测"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()

def c5(ret):
    if ret < -0.10: return 0
    if ret < 0: return 1
    if ret < 0.10: return 2
    if ret < 0.30: return 3
    return 4

u["label"] = u.forward_return.apply(c5)
print("5类分布:")
for i, nm in enumerate([">-10%暴跌","-10~0%跌","0~10%小涨","10~30%中涨","30%+暴涨"]):
    print(f"  {i}: {nm} = {(u.label==i).sum()}")
print()

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

def train(Xtr_np, ytr_np, Xte_np):
    sc = StandardScaler()
    Xtr = torch.FloatTensor(sc.fit_transform(Xtr_np))
    Xte = torch.FloatTensor(sc.transform(Xte_np))
    ytr = torch.LongTensor(ytr_np)
    m = M5(Xtr.shape[1])
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss()
    bl, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad()
        loss = lfn(m(Xtr), ytr)
        loss.backward(); opt.step()
        if loss.item() < bl:
            bl = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        return torch.argmax(m(Xte), dim=1).numpy()

all_preds, all_rets, all_years = [], [], []
for ty in [2020, 2021, 2022, 2023, 2024]:
    tr = (yrs != ty); te = (yrs == ty)
    p = train(X[tr].values, y[tr], X[te].values)
    all_preds.extend(p)
    all_rets.extend(u.loc[te, "forward_return"].values)
    all_years.extend(u.loc[te, "year"].values)

# 混淆矩阵
cm = confusion_matrix(np.array([c5(r) for r in all_rets]), all_preds)
acc = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4])/cm.sum()*100
print(f"=== 混淆矩阵 (5年合并 {cm.sum()}条, 准确率={acc:.1f}%) ===")
print(f"实际↓预测 |暴跌| 跌 |小涨|中涨|暴涨| 合计")
for i, nm in enumerate(["暴跌","跌","小涨","中涨","暴涨"]):
    print(f"  {nm:6s}|{cm[i,0]:>4d}|{cm[i,1]:>4d}|{cm[i,2]:>4d}|{cm[i,3]:>4d}|{cm[i,4]:>4d}|{cm[i].sum():>5d}")

# 每类召回
print()
for i, nm in enumerate(["暴跌","跌","小涨","中涨","暴涨"]):
    r = cm[i,i]/cm[i].sum()*100 if cm[i].sum()>0 else 0
    print(f"  {nm} 召回: {r:.1f}%")

# 策略回测
buy = pd.DataFrame({"pred": all_preds, "ret": all_rets, "year": all_years})
all_avg = buy.ret.mean()
print(f"\n全市场平均: {all_avg*100:+.1f}%")

for rule, name in [([4], "只买暴涨(>30%)"), ([3,4], "买中涨+暴涨(>10%)")]:
    bought = buy[buy.pred.isin(rule)]
    n = len(bought)
    avg_r = bought.ret.mean()
    med_r = bought.ret.median()
    win = (bought.ret > 0).mean() * 100
    big = (bought.ret > 0.3).mean() * 100
    exc = avg_r - all_avg
    print(f"\n=== {name} ===")
    print(f"买入: {n}次 平均: {avg_r*100:+.1f}% 中位: {med_r*100:+.1f}% 超额: {exc*100:+.1f}%")
    print(f"正收益: {win:.1f}%  暴涨(>30%): {big:.1f}%")
    print("逐年:")
    total = 1.0
    for ty in [2020, 2021, 2022, 2023, 2024]:
        b = buy[(buy.year == ty) & (buy.pred.isin(rule))]
        a = buy[buy.year == ty]
        if len(b) > 0:
            ex = b.ret.mean() - a.ret.mean()
            total *= 1 + b.ret.mean()
            print(f"  {ty}: 买{len(b)}只 均{b.ret.mean()*100:+.1f}% 超额{ex*100:+.1f}%")
        else:
            print(f"  {ty}: 未买入")
    print(f"  5年复利: {total:.3f}x = {(total-1)*100:+.1f}%")
