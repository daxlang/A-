"""附录D补充"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
LAB = ["暴跌","跌","小涨","中涨","暴涨"]

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def load_mlp(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

def print_cm_entry(title, cm, buys_detail, yrly):
    print(f"\n### {title}")
    print("```")
    print(f"{'实际->预测':<10s}{'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
    for i, lb in enumerate(LAB):
        print(f"{lb:<10s}{cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
    print(", ".join(buys_detail))
    print(f"年均: {yrly}")
    print("```")

# ===== D.1 PE修复 =====
print("\n## 附录D: 补充详细数据")
print("\n### D.1 PE周期修复 (S2_repair)")
df = pd.read_csv(os.path.join(BASE, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
rets = u.forward_return.values; yrs = u.year.values
M_s13 = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1)

pr_repair = np.zeros(len(u), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (u.year == ty)
    pr_repair[te] = load_mlp(X[te].values, os.path.join(MDIR, f"S2_repair_{ty}.pt"))

cm = confusion_matrix(u.L[M_s13], pr_repair[M_s13])
acc = sum(cm[i,i] for i in range(5))/cm.sum()*100
buy = (pr_repair == 4) & M_s13; nb = buy.sum(); br = rets[buy]
print(f"S2_repair->S1capS3 ({cm.sum()}tiao acc={acc:.1f}%) 买{nb}只 暴涨召回{cm[4,4]}/{cm[4].sum()}")
print(f"基准 MLP0_5->S1capS3: 231买 年+21.8%")

print("```")
print(f"{'实际->预测':<10s}{'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
for i, lb in enumerate(LAB):
    print(f"{lb:<10s}{cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
yrly = []
for ty in sorted(set(yrs)):
    b = (yrs == ty) & buy; n = b.sum(); a = rets[b]
    yl = a.mean()*100 if n > 0 else 0; yrly.append(yl)
    print(f"  {ty}: {n}只 {a.mean()*100:.1f}%" if n > 0 else f"  {ty}: 0只")
print(f"  年均: {np.mean(yrly):.1f}%")
print("```")

# ===== D.2 季度预测 =====
print("\n### D.2 季度预测 (Q_MLP0_5)")
u2 = df[df.usable].copy()
u2 = u2.sort_values(["code","year","quarter"])
u2["next_price"] = u2.groupby("code")["buy_price"].shift(-1)
u2["q_return"] = u2.next_price / u2.buy_price - 1; u2 = u2[u2.q_return.notna()].copy()
u2["gm_pct"] = u2.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u2["L"] = u2.q_return.apply(lambda r: 0 if r<-0.05 else (1 if r<0 else (2 if r<0.05 else (3 if r<0.15 else 4))))
feat2 = [c for c in u2.columns if c not in ["code","year","quarter","forward_return","q_return","L","usable","industry","gm_pct","next_price"]]
ind2 = pd.get_dummies(u2.industry, prefix="ind").astype(float)
X2 = pd.concat([u2[feat2], ind2], axis=1); X2 = X2.fillna(X2.median())
Mq = (u2.cfo_to_revenue >= 0) & (u2.current_ratio >= 0.5) & (u2.gm_pct >= 0.1) & (u2.liability_to_asset <= 0.8)
yq = u2.L.values; retsq = u2.q_return.values; yrsq = u2.year.values

prq = np.zeros(len(u2), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (u2.year == ty)
    prq[te] = load_mlp(X2[te].values, os.path.join(MDIR, f"Q_MLP0_5_{ty}.pt"))

cmq = confusion_matrix(yq[Mq], prq[Mq])
accq = sum(cmq[i,i] for i in range(5))/cmq.sum()*100
buyq = (prq == 4) & Mq; nbq = buyq.sum(); brq = retsq[buyq]
print(f"Q_MLP0_5->S1capS3capS4 ({cmq.sum()}tiao acc={accq:.1f}%) 买{nbq}只 暴涨召回{cmq[4,4]}/{cmq[4].sum()}")

print("```")
print(f"{'实际->预测':<10s}{'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
for i, lb in enumerate(LAB):
    print(f"{lb:<10s}{cmq[i,0]:>5d}{cmq[i,1]:>4d}{cmq[i,2]:>5d}{cmq[i,3]:>5d}{cmq[i,4]:>5d}|{cmq[i].sum():>5d}")
yrlyq = []
for ty in sorted(set(yrsq)):
    b = (yrsq == ty) & buyq; n = b.sum(); a = retsq[b]
    yl = a.mean()*100 if n > 0 else 0; yrlyq.append(yl)
    print(f"  {ty}: {n}只 季均{a.mean()*100:.1f}%" if n > 0 else f"  {ty}: 0只")
print(f"  季均: {brq.mean()*100:.1f}%  年化={(1+brq.mean())**4-1:.1%}")
print("```")

# ===== D.3 动态卖出 =====
print("\n### D.3 动态卖出策略逐年明细")
u3 = df[df.usable].copy()
u3["gm_pct"] = u3.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u3["L"] = u3.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
feat3 = [c for c in u3.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind3 = pd.get_dummies(u3.industry, prefix="ind").astype(float)
X3 = pd.concat([u3[feat3], ind3], axis=1); X3 = X3.fillna(X3.median())
M3 = (u3.cfo_to_revenue >= 0) & (u3.current_ratio >= 0.5) & (u3.gm_pct >= 0.1)
pr3 = np.zeros(len(u3), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (u3.year == ty)
    pr3[te] = load_mlp(X3[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))

u3["pred"] = pr3
nxt_p = []; nxt_r = []
for idx in u3.index:
    row = u3.loc[idx]
    nxt = u3[(u3.code == row.code) & (u3.quarter == row.quarter) & (u3.year == row.year + 1)]
    if len(nxt) > 0:
        nxt_p.append(nxt.iloc[0].pred); nxt_r.append(nxt.iloc[0].forward_return)
    else:
        nxt_p.append(-1); nxt_r.append(np.nan)
u3["next_pred"] = nxt_p; u3["next_ret"] = nxt_r

bought = u3[(u3.pred == 4) & M3 & (u3.year <= 2023)]
sold = bought[bought.next_pred.isin([0, 1])]
held = bought[bought.next_pred.isin([2, 3, 4])]

sold_ann = 1 + sold.forward_return.values
held_2yr = (1 + held.forward_return.values) * (1 + held.next_ret.values) - 1
held_ann = np.sqrt(1 + held_2yr)
dyn = (len(sold)*sold_ann.mean() + len(held)*held_ann.mean()) / (len(sold)+len(held)) - 1

print("买入153只(2019-2023, 有次年预测)")
print(f"下年预测: 暴跌{(bought.next_pred==0).sum()} 跌{(bought.next_pred==1).sum()} 小涨{(bought.next_pred==2).sum()} 中涨{(bought.next_pred==3).sum()} 暴涨{(bought.next_pred==4).sum()}")
print(f"卖出组({len(sold)}只): 年化{sold_ann.mean()-1:.1%} 次年{sold.next_ret.mean()*100:.1f}% 跌率{((sold.next_ret<0).sum()/len(sold))*100:.0f}%")
print(f"持有组({len(held)}只): 2年年化{held_ann.mean()-1:.1%} 次年{held.next_ret.mean()*100:.1f}% 跌率{((held.next_ret<0).sum()/len(held))*100:.0f}%")
print(f"动态年化: {dyn:.1%} vs 固定一年: {bought.forward_return.mean():.1%}")

# ===== D.4 三模型共识 =====
print("\n### D.4 三模型共识买入 (4行业)")
print("MLP0_5(4行业)->S1capS3: 231买 年+21.8%")
print("MLPcapRF: 54买 年+47.6% 2020:10+0% 2021:18+87% 2022:7+20% 2023:14-9% 2024:5+139%")
print("MLPcapLR: 47买 年+16.1% 2022:0")
print("RFcapLR: 53买 年+8.9% 2022/2024:0")
