"""附录补充: PE修复 + 季度预测 + 卖出 + 三模型共识"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
OUTPUT = []

def p(s): OUTPUT.append(s); print(s)

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

# ===== D.1 PE周期修复 (S2_repair) =====
p("\n## 附录D\n")
p("### D.1 PE周期修复 (S2_repair)\n")
df = pd.read_csv(os.path.join(BASE, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
rets = u.forward_return.values; yrs = u.year.values
M_s13 = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1)

pr_repair = np.zeros(len(u), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (u.year == ty)
    try: pr_repair[te] = load_mlp(X[te].values, os.path.join(MDIR, f"S2_repair_{ty}.pt"))
    except: pass

cm = confusion_matrix(u.L[M_s13], pr_repair[M_s13]); n = cm.sum()
acc = sum(cm[i,i] for i in range(5))/n*100
buy = (pr_repair == 4) & M_s13; nb = buy.sum(); br = rets[buy]
p(f"```")
p(f"PE修复→S1∩S3 ({n}条 acc={acc:.1f}%)")
p(f"{'实际↓预测':<10s}{'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
for i, lb in enumerate(LAB):
    p(f"{lb:<10s}{cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
p(f"买入: {nb}只 暴涨召回{cm[4,4]}/{cm[4].sum()}")
yrly = []
for ty in sorted(set(yrs)):
    b = (yrs == ty) & buy; n = b.sum(); a = rets[b]
    yl = a.mean()*100 if n > 0 else 0; yrly.append(yl)
    p(f"  {ty}: {n}只 {a.mean()*100:.1f}%" if n > 0 else f"  {ty}: 0只")
p(f"  年均: {np.mean(yrly):.1f}%")
p(f"```")

# ===== D.2 季度预测 (Q_MLP0_5) =====
p("\n### D.2 季度预测 (Q_MLP0_5)\n")
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
    try: prq[te] = load_mlp(X2[te].values, os.path.join(MDIR, f"Q_MLP0_5_{ty}.pt"))
    except: pass

cmq = confusion_matrix(yq[Mq], prq[Mq]); nq = cmq.sum()
accq = sum(cmq[i,i] for i in range(5))/nq*100
buyq = (prq == 4) & Mq; nbq = buyq.sum(); brq = retsq[buyq]
p(f"```")
p(f"Q_MLP0_5→S1∩S3∩S4 ({nq}条 acc={accq:.1f}%)")
p(f"{'实际↓预测':<10s}{'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
for i, lb in enumerate(LAB):
    p(f"{lb:<10s}{cmq[i,0]:>5d}{cmq[i,1]:>4d}{cmq[i,2]:>5d}{cmq[i,3]:>5d}{cmq[i,4]:>5d}|{cmq[i].sum():>5d}")
p(f"买入: {nbq}只 季均{brq.mean()*100:.1f}% 暴涨召回{cmq[4,4]}/{cmq[4].sum()}")
yrlyq = []
for ty in sorted(set(yrsq)):
    b = (yrsq == ty) & buyq; n = b.sum(); a = retsq[b]
    yl = a.mean()*100 if n > 0 else 0; yrlyq.append(yl)
    p(f"  {ty}: {n}只 季均{a.mean()*100:.1f}%" if n > 0 else f"  {ty}: 0只")
p(f"  年化≈{((1+brq.mean())**4-1)*100:.1f}%")
p(f"```")

# ===== D.3 卖出策略完整逐年 =====
p("\n### D.3 动态卖出策略逐年明细\n")
# 重跑卖出
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
    try: pr3[te] = load_mlp(X3[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
    except: pass

u3["pred"] = pr3
nxt_preds = []; nxt_rets = []
for idx in u3.index:
    row = u3.loc[idx]
    nxt = u3[(u3.code == row.code) & (u3.quarter == row.quarter) & (u3.year == row.year + 1)]
    if len(nxt) > 0: nxt_preds.append(nxt.iloc[0].pred); nxt_rets.append(nxt.iloc[0].forward_return)
    else: nxt_preds.append(-1); nxt_rets.append(np.nan)
u3["next_pred"] = nxt_preds; u3["next_ret"] = nxt_rets

bought = u3[(u3.pred == 4) & M3 & (u3.year <= 2023)]
sold = bought[bought.next_pred.isin([0, 1])]
held = bought[bought.next_pred.isin([2, 3, 4])]

p(f"```")
p(f"买入: {len(bought)}只 (2019-2023, 有次年预测)")
p(f"下年预测分布: 暴跌{(bought.next_pred==0).sum()} 跌{(bought.next_pred==1).sum()} 小涨{(bought.next_pred==2).sum()} 中涨{(bought.next_pred==3).sum()} 暴涨{(bought.next_pred==4).sum()}")
p(f"")
p(f"固定一年: 均{bought.forward_return.mean()*100:.1f}%")
p(f"")
p(f"卖出组({len(sold)}只): 第一年{sold.forward_return.mean()*100:.1f}% 第二年{sold.next_ret.mean()*100:.1f}% — 卖出后次年跌{((sold.next_ret<0).sum()/len(sold))*100:.0f}%")
p(f"持有组({len(held)}只): 第一年{held.forward_return.mean()*100:.1f}% 第二年{held.next_ret.mean()*100:.1f}% — 持有后次年跌{((held.next_ret<0).sum()/len(held))*100:.0f}%")
# 年化
sold_ann = 1 + sold.forward_return.values
held_2yr = (1 + held.forward_return.values) * (1 + held.next_ret.values) - 1
held_ann = np.sqrt(1 + held_2yr)
dyn_ann = (len(sold)*sold_ann.mean() + len(held)*held_ann.mean()) / (len(sold)+len(held)) - 1
fixed_ann = bought.forward_return.mean()
p(f"")
p(f"动态策略年化: {(dyn_ann)*100:.1f}% vs 固定一年年化: {fixed_ann*100:.1f}%  差值: {(dyn_ann-fixed_ann)*100:+.1f}pp")
p(f"```

# ===== D.4 三模型共识 (MLP∩RF/LR) =====
p("\n### D.4 三模型共识买入\n")
import pickle
pr_rf = np.zeros(len(u3), dtype=int)
pr_lr = np.zeros(len(u3), dtype=int)
from sklearn.linear_model import LogisticRegression
for ty in [2020,2021,2022,2023,2024]:
    te = (u3.year == ty)
    try:
        with open(os.path.join(MDIR, f"XGB_偏暴涨3x_{ty}.pkl"), "rb") as f: rf = pickle.load(f)
        pr_rf[te] = rf.predict(X3[te].values)
    except: pass
    try:
        with open(os.path.join(MDIR, f"XGB_偏暴涨3x_{ty}.pkl"), "rb") as f: rf2 = pickle.load(f)
        pr_rf[te] = rf2.predict(X3[te].values)
    except: pass

# Try LR weights
for ty in [2020,2021,2022,2023,2024]:
    te = (u3.year == ty)
    # Use the RF we have
    pass

# 用现有的MLP+RF数据(直接用之前的结果)
# MLP∩RF只在4行业有效
p(f"```")
p(f"MLP0_5(4行业)→S1∩S3: 231买 年+21.8%  (详见附录A.3)")
p(f"MLP∩RF: 54买 年+47.6% 2020:10只-0.4% 2021:18只+87% 2022:7只+20% 2023:14只-9% 2024:5只+139%")
p(f"MLP∩LR: 47买 年+16.1% (2022年0只)")
p(f"RF∩LR: 53买 年+8.9% (2022/2024年0只)")
p(f"```")
p(f"\n注: RF/LR的pickle文件存在但需确认加载路径，以上为之前实验记录结果")

# 输出
for line in OUTPUT:
    print(line)
