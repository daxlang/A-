"""策略回测: 买所有预测为大涨的股票, 持有一年"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["label"] = u.forward_return.apply(lambda r: 0 if r < 0 else (1 if r <= 0.05 else 2))
feat = [c for c in u.columns if c not in
    ["code","year","quarter","forward_return","label","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.label.values; yrs = u.year.values

class M2(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)

def train(Xtr_np, ytr_np, Xte_np):
    sc = StandardScaler()
    Xtr = torch.FloatTensor(sc.fit_transform(Xtr_np))
    Xte = torch.FloatTensor(sc.transform(Xte_np))
    ytr = torch.LongTensor(ytr_np)
    m = M2(Xtr.shape[1])
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.0, 1.0, 1.0]))
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
    rets = u.loc[te, ["code", "year", "quarter", "forward_return", "industry"]].copy()
    all_preds.extend(p)
    all_rets.extend(rets.forward_return.values)
    all_years.extend(rets.year.values)

buy = pd.DataFrame({"pred": all_preds, "ret": all_rets, "year": all_years})
bought = buy[buy.pred == 2]
n_buy = len(bought)
avg_ret = bought.ret.mean()
med_ret = bought.ret.median()
win_rate = (bought.ret > 0).mean() * 100
big_win = (bought.ret > 0.3).mean() * 100
all_avg = buy.ret.mean()

print(f"策略: 买入模型预测为'大涨'的所有股票, 等权持有1年")
print(f"{'='*50}")
print(f"总买入次数: {n_buy} (共2600次机会)")
print(f"平均收益率: {avg_ret*100:+.1f}%")
print(f"中位收益率: {med_ret*100:+.1f}%")
print(f"正收益比例: {win_rate:.1f}%")
print(f"暴涨(>30%)比例: {big_win:.1f}%")
print(f"全市场平均: {all_avg*100:+.1f}%")
print(f"超额收益: {(avg_ret-all_avg)*100:+.1f}%")
print()
print("逐年:")
for ty in [2020, 2021, 2022, 2023, 2024]:
    b = buy[(buy.year == ty) & (buy.pred == 2)]
    a = buy[buy.year == ty]
    exc = b.ret.mean() - a.ret.mean() if len(b) > 0 else 0
    print(f"  {ty}: 买{len(b)}只 策略{b.ret.mean()*100:+.1f}% 全{a.ret.mean()*100:+.1f}% 超额{exc*100:+.1f}%")

total = 1.0
for ty in [2020, 2021, 2022, 2023, 2024]:
    b = buy[(buy.year == ty) & (buy.pred == 2)]
    if len(b) > 0:
        total *= 1 + b.ret.mean()
print(f"\n5年复利: {total:.3f}x = {(total-1)*100:+.1f}%")
