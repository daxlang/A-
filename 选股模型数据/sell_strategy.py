"""卖出策略: 买入后下年模型判暴跌→退出 vs 固定持有一年"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1)

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def load_pred(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

# 全量预测
pr = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (u.year == ty); pr[te] = load_pred(X[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))

# 买入 + 下年判断
u["pred"] = pr
bought = u[(pr == 4) & MASK].copy()
bought["next_pred"] = -1
bought["next_ret"] = np.nan

for idx, row in bought.iterrows():
    # 找同股票同季度下一年
    nxt = u[(u.code == row.code) & (u.quarter == row.quarter) & (u.year == row.year + 1)]
    if len(nxt) > 0:
        bought.at[idx, "next_pred"] = nxt.iloc[0].pred
        bought.at[idx, "next_ret"] = nxt.iloc[0].forward_return

bought_2024 = bought[bought.year == 2024]
bought_pre2024 = bought[bought.year <= 2023]

print(f"总买入(掩码内): {len(bought)} 只")
print(f"  2019-2023(有下年): {len(bought_pre2024)} 只")
print(f"  2024(无下年): {len(bought_2024)} 只")

# 固定持有一年
fixed = bought_pre2024.forward_return
print(f"\n固定一年: 均{fixed.mean()*100:+.1f}%  {len(fixed)}只")

# 策略: 下年判暴跌(0)或跌(1)→退出
sell_mask = bought_pre2024.next_pred.isin([0, 1])
hold_mask = bought_pre2024.next_pred.isin([2, 3, 4])
no_next = bought_pre2024.next_pred == -1

print(f"\n下年预测分布:")
for lab, cnt in zip(LAB, [sum(bought_pre2024.next_pred == i) for i in range(5)]):
    print(f"  {lab}: {cnt}只")

# 动态卖出收益
sell_ret = bought_pre2024[sell_mask].forward_return.values
hold_ret = (1 + bought_pre2024[hold_mask].forward_return.values) * (1 + bought_pre2024[hold_mask].next_ret.values) - 1
# 对应年化: sell=1年, hold=2年→年化需要调整
# 简单算总收益对比: fixed = Σ所有买入的年收益
strat_ret = np.concatenate([sell_ret, hold_ret])
print(f"\n动态策略: 均{strat_ret.mean()*100:+.1f}%  {len(strat_ret)}只")
print(f"  卖出(1年): {len(sell_ret)}只 均{sell_ret.mean()*100:+.1f}%")
print(f"  继续持有(2年): {len(hold_ret)}只 均{hold_ret.mean()*100:+.1f}%")

# 逐年
print(f"\n逐年对比:")
print(f"{'年':>4s} {'买入':>5s} {'固定1年':>7s} {'卖出':>5s} {'继续':>5s} {'动态均':>7s}")
for ty in [2020, 2021, 2022, 2023]:
    b = bought_pre2024[bought_pre2024.year == ty]
    s = b[b.next_pred.isin([0, 1])]
    h = b[b.next_pred.isin([2, 3, 4])]
    sr = s.forward_return.values
    hr = (1 + h.forward_return.values) * (1 + h.next_ret.values) - 1 if len(h) > 0 else np.array([])
    dyn = np.concatenate([sr, hr]) if len(sr) > 0 or len(hr) > 0 else np.array([])
    print(f"{ty:>4d} {len(b):>5d} {b.forward_return.mean()*100:>+6.1f}% {len(s):>5d} {len(h):>5d} {dyn.mean()*100:>+6.1f}%")

# 核心: 次年回报
sold = bought_pre2024[bought_pre2024.next_pred.isin([0, 1])]
held = bought_pre2024[bought_pre2024.next_pred.isin([2, 3, 4])]
print(f"\n=== 次年回报(核心) ===")
print(f"判卖出({len(sold)}只): 第一年+{sold.forward_return.mean()*100:.1f}% 第二年+{sold.next_ret.mean()*100:.1f}%")
print(f"判持有({len(held)}只): 第一年+{held.forward_return.mean()*100:.1f}% 第二年+{held.next_ret.mean()*100:.1f}%")
d = (held.next_ret.mean() - sold.next_ret.mean())*100
print(f"次年差: {d:+.1f}pp (正=卖出明智)")
print(f"判卖出里 第二年实际跌: {(sold.next_ret<0).sum()}/{len(sold)}={((sold.next_ret<0).sum()/len(sold))*100:.0f}%")
print(f"判持有里 第二年实际跌: {(held.next_ret<0).sum()}/{len(held)}={((held.next_ret<0).sum()/len(held))*100:.0f}%")
