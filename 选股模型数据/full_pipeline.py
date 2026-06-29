"""全量数据: 过滤 → 训练集 → 基线对比"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

# === 1. 过滤 ===
df_codes = pd.read_csv(os.path.join(OUT, "stocks_after_pe_filter.csv"), dtype=str)
valid_codes = set(df_codes["code"].tolist())
print(f"PE过滤后: {len(valid_codes)}只")

# === 2. 构建训练集 ===
df_f = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
df_f = df_f[df_f["code"].isin(valid_codes)]

df_p = pd.read_csv(os.path.join(OUT, "prices_daily.csv"), dtype={"code": str})
df_p["date"] = pd.to_datetime(df_p["date"])
df_p["close"] = pd.to_numeric(df_p["close"], errors="coerce")

# 价格索引
price_map = {}
for code, grp in df_p.groupby("code"):
    grp = grp.dropna(subset=["close"]).sort_values("date")
    price_map[code] = grp.set_index("date")["close"]

def nearest_price(code, target_date):
    if code not in price_map: return np.nan
    ts = pd.Timestamp(target_date)
    s = price_map[code]
    mask = s.index >= ts
    return float(s[mask].iloc[0]) if mask.any() else float(s.iloc[-1])

def get_buy_sell(year, quarter):
    if quarter == 1: return (f"{year}-04-30", f"{year+1}-04-30")
    elif quarter == 2: return (f"{year}-07-31", f"{year+1}-07-31")
    elif quarter == 3: return (f"{year}-10-31", f"{year+1}-10-31")
    else: return (f"{year+1}-04-30", f"{year+2}-04-30")

rows = []
code_ind = dict(zip(df_codes["code"], df_codes["industry"]))
for _, r in df_f.iterrows():
    code, year, qtr = r["code"], int(r["year"]), int(r["quarter"])
    buy_d, sell_d = get_buy_sell(year, qtr)
    p_buy = nearest_price(code, buy_d)
    p_sell = nearest_price(code, sell_d)
    ret = (p_sell - p_buy) / p_buy if (pd.notna(p_buy) and pd.notna(p_sell) and p_buy > 0) else np.nan
    rows.append({
        "code": code, "year": year, "quarter": qtr,
        "roe": r.get("roe"), "gross_margin": r.get("gross_margin"),
        "net_margin": r.get("net_margin"), "profit_yoy": r.get("profit_yoy"),
        "asset_turnover": r.get("asset_turnover"), "inventory_turnover": r.get("inventory_turnover"),
        "forward_return": ret, "industry": code_ind.get(code, ""),
    })

df_train = pd.DataFrame(rows)
df_train["usable"] = df_train["forward_return"].notna() & (df_train["year"] <= 2024)
usable = df_train[df_train["usable"]]
n_usable = len(usable)
n_stocks = usable["code"].nunique()
print(f"训练样本: {n_usable}条, {n_stocks}只股票 ({n_usable//n_stocks}季度/只)")
print(f"目标均值: {usable.forward_return.mean()*100:+.1f}% 中位: {usable.forward_return.median()*100:+.1f}% 标准差: {usable.forward_return.std():.3f}")

# === 3. 特征工程 ===
feat_cols = ['roe','gross_margin','net_margin','profit_yoy','asset_turnover','inventory_turnover']
ind = pd.get_dummies(usable['industry'], prefix='ind').astype(float)
X = pd.concat([usable[feat_cols], ind], axis=1)
X = X.fillna(X.median())
y = usable['forward_return'].values.ravel()

train_mask = (usable['year'] < 2024) | ((usable['year'] == 2024) & (usable['quarter'] <= 2))
test_mask = ~train_mask

sc_X = StandardScaler(); sc_y = StandardScaler()
Xtr = torch.FloatTensor(sc_X.fit_transform(X[train_mask]))
ytr = torch.FloatTensor(sc_y.fit_transform(y[train_mask].reshape(-1,1)))
Xte = torch.FloatTensor(sc_X.transform(X[test_mask]))
yte_raw = y[test_mask]
n_feat = Xtr.shape[1]

print(f"\n训练: {Xtr.shape[0]}条  测试: {Xte.shape[0]}条  特征: {n_feat}维")

# === 基线 ===
naive = np.full_like(yte_raw, y[train_mask].mean())
mae_naive = np.mean(np.abs(yte_raw - naive))
print(f"\n基线(猜均值) MAE={mae_naive:.4f}")

lr = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5)
lr.fit(Xtr.numpy(), y[train_mask])
pred_lr = lr.predict(Xte.numpy())
mae_lr = np.mean(np.abs(yte_raw - pred_lr))
ric_lr, _ = spearmanr(yte_raw, pred_lr)
print(f"线性回归 MAE={mae_lr:.4f} Rank_IC={ric_lr:+.4f} 改善={+(1-mae_lr/mae_naive)*100:.1f}%")

rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
rf.fit(Xtr.numpy(), y[train_mask])
pred_rf = rf.predict(Xte.numpy())
mae_rf = np.mean(np.abs(yte_raw - pred_rf))
ric_rf, _ = spearmanr(yte_raw, pred_rf)
print(f"随机森林 MAE={mae_rf:.4f} Rank_IC={ric_rf:+.4f} 改善={+(1-mae_rf/mae_naive)*100:.1f}%")

# === MLP ===
class TinyMLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h // 2, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(h=16, lr=0.005, epochs=2000):
    torch.manual_seed(42); m = TinyMLP(h)
    opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    best, bs = 1e9, None
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        loss = nn.MSELoss()(m(Xtr), ytr)
        loss.backward()
        opt.step()
        if loss.item() < best:
            best = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): return sc_y.inverse_transform(m(Xte).numpy()).ravel()

print(f"\n=== MLP ===")
best_mae, best_h = 1e9, 0
for h in [8,16,32,64,128]:
    n_param = h*n_feat + h + h*(h//2) + (h//2) + (h//2) + 1
    pred = train_mlp(h=h)
    mae_m = np.mean(np.abs(yte_raw - pred))
    ric_m, _ = spearmanr(yte_raw, pred)
    impr = (1 - mae_m / mae_naive) * 100
    tag = " < 最优MLP" if mae_m < best_mae else ""
    print(f"  MLP(h={h:3d}, {n_param:4d}参) MAE={mae_m:.4f} Rank_IC={ric_m:+.4f} 改善={impr:+.1f}%{tag}")
    if mae_m < best_mae:
        best_mae, best_h = mae_m, h

print(f"\n=== 汇总 ===")
print(f"线性回归 MAE={mae_lr:.4f} Rank_IC={ric_lr:+.3f}")
print(f"随机森林 MAE={mae_rf:.4f} Rank_IC={ric_rf:+.3f}")
print(f"最优MLP(h={best_h}) MAE={best_mae:.4f}")
winner = "线性回归" if mae_lr <= min(mae_rf, best_mae) else ("随机森林" if mae_rf <= best_mae else "MLP")
print(f"MAE最优: {winner}")

# 保存
# 保存
df_train.to_csv(os.path.join(OUT, "training_dataset.csv"), index=False, encoding="utf-8-sig")
print(f"\n训练集已保存: training_dataset.csv ({n_usable}条)")
