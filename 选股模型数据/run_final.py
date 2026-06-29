"""Step1: 补买入价 + 行业均值填充 + 建训练集 + 跑模型"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import warnings, os, time
warnings.filterwarnings("ignore")

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
LOG = os.path.join(OUT, "final_run.txt")

def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{t}] {msg}\n")

with open(LOG, "w", encoding="utf-8") as f: f.write("")

# === 加载 ===
dc = pd.read_csv(os.path.join(OUT, "all_stock_codes_full.csv"), dtype=str)
ci = dict(zip(dc.code, dc.industry))
df_f = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
df_p = pd.read_csv(os.path.join(OUT, "prices_daily.csv"), dtype={"code": str})
df_p["date"] = pd.to_datetime(df_p["date"])
df_p["close"] = pd.to_numeric(df_p["close"], errors="coerce")
log(f"财报: {len(df_f)}条 {df_f.code.nunique()}只")

# === Step1: 补买入价 ===
pm = {}
for c, g in df_p.groupby("code"):
    g = g.dropna(subset=["close"]).sort_values("date")
    pm[c] = g.set_index("date")["close"]

def buy_date(y, q):
    if q == 1: return f"{y}-04-30"
    if q == 2: return f"{y}-07-31"
    if q == 3: return f"{y}-10-31"
    return f"{y+1}-04-30"

def nearest_price(code, target):
    if code not in pm: return np.nan
    ts = pd.Timestamp(target); s = pm[code]
    m = s.index >= ts
    return float(s[m].iloc[0]) if m.any() else float(s.iloc[-1])

df_f["buy_price"] = df_f.apply(
    lambda r: nearest_price(r["code"], buy_date(int(r["year"]), int(r["quarter"]))), axis=1
)
log(f"买入价补完: {df_f.buy_price.notna().sum()}/{len(df_f)}")

# === Step2: 行业均值填充 ===
df_f["industry"] = df_f["code"].map(ci)
feat_cols = ["roe", "gross_margin", "net_margin", "profit_yoy", "asset_turnover",
             "inventory_turnover", "liability_to_asset", "current_ratio",
             "cfo_to_revenue", "cfo_to_profit", "interest_coverage",
             "buy_price", "pe", "pb", "ps"]

for col in feat_cols:
    if col in df_f.columns:
        missing = df_f[col].isna().sum()
        if missing > 0:
            # 按(行业, 年, 季度)取均值, 避免跨期混合和前视偏差
            ind_yr_q_means = df_f.groupby(["industry", "year", "quarter"])[col].transform("mean")
            df_f[col] = df_f[col].fillna(ind_yr_q_means)
            # 仍有NaN的降级到(行业, 年)均值
            ind_yr_means = df_f.groupby(["industry", "year"])[col].transform("mean")
            df_f[col] = df_f[col].fillna(ind_yr_means)
            # 再降: 行业全局均值
            ind_means = df_f.groupby("industry")[col].transform("mean")
            df_f[col] = df_f[col].fillna(ind_means)
            # 兜底: 全局均值
            df_f[col] = df_f[col].fillna(df_f[col].mean())
            log(f"  {col}: 填充{missing}个 (行业季>行业年>行业>全局) → 剩余{df_f[col].isna().sum()}个")

# === Step3: 建训练集 ===
def sell_date(y, q):
    if q == 1: return f"{y+1}-04-30"
    if q == 2: return f"{y+1}-07-31"
    if q == 3: return f"{y+1}-10-31"
    return f"{y+2}-04-30"

rows = []
for _, r in df_f.iterrows():
    c, y, q = r["code"], int(r["year"]), int(r["quarter"])
    bd = buy_date(y, q); sd = sell_date(y, q)
    pb_v = nearest_price(c, bd); ps_v = nearest_price(c, sd)
    ret = (ps_v - pb_v) / pb_v if (pd.notna(pb_v) and pd.notna(ps_v) and pb_v > 0) else np.nan
    
    row = {"code": c, "year": y, "quarter": q, "forward_return": ret, "industry": ci.get(c, "")}
    for col in feat_cols:
        row[col] = r.get(col, np.nan)
    rows.append(row)

df = pd.DataFrame(rows)
df["usable"] = df["forward_return"].notna() & (df["year"] <= 2024)
u = df[df["usable"]].copy()
log(f"训练集: {len(u)}条 {u.code.nunique()}只, 目标均值={u.forward_return.mean()*100:+.1f}% 中位={u.forward_return.median()*100:+.1f}%")

# === Step4: 特征 ===
stable_feat = [c for c in feat_cols if u[c].notna().sum() > len(u) * 0.3]
log(f"可用特征({len(stable_feat)}维): {stable_feat}")

ind = pd.get_dummies(u["industry"], prefix="ind").astype(float)
X = pd.concat([u[stable_feat], ind], axis=1)
X = X.fillna(X.median())
yt = u["forward_return"].values.ravel()

tm = (u["year"] < 2024) | ((u["year"] == 2024) & (u["quarter"] <= 2))
vm = ~tm

sc_X = StandardScaler(); sc_y = StandardScaler()
Xtr = torch.FloatTensor(sc_X.fit_transform(X[tm]))
ytr = torch.FloatTensor(sc_y.fit_transform(yt[tm].reshape(-1, 1)))
Xte = torch.FloatTensor(sc_X.transform(X[vm]))
yte = yt[vm]
naive = np.mean(np.abs(yte - np.mean(yt[tm])))
log(f"训练:{Xtr.shape[0]} 测试:{Xte.shape[0]} 特征:{Xtr.shape[1]}维 基线={naive:.4f}")

# === Step5: 线性基线 ===
lr = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
lr.fit(Xtr.numpy(), yt[tm])
pl = lr.predict(Xte.numpy())
ml, rl = np.mean(np.abs(yte - pl)), spearmanr(yte, pl)[0]
log(f"线性: MAE={ml:.4f} R_IC={rl:+.3f} 改善={(1-ml/naive)*100:+.1f}%")

# 线性权重
coefs = pd.DataFrame({"feat": X.columns, "coef": lr.coef_}).sort_values("coef", key=abs, ascending=False)
log(f"  权重top5: {coefs.head(5).to_dict('records')}")

# RF
rf = RandomForestRegressor(100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
rf.fit(Xtr.numpy(), yt[tm])
pr = rf.predict(Xte.numpy())
mr, rr = np.mean(np.abs(yte - pr)), spearmanr(yte, pr)[0]
log(f"RF:   MAE={mr:.4f} R_IC={rr:+.3f} 改善={(1-mr/naive)*100:+.1f}%")

# MLP
class TM(nn.Module):
    def __init__(self, n_in, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h // 2, 1)
        )
    def forward(self, x):
        return self.net(x)

log("=== MLP ===")
best_mae, best_h = 1e9, 0
best_ric = 0
for h in [8, 16, 32, 64, 128]:
    n_p = (Xtr.shape[1] * h + h) + (h * h // 2 + h // 2) + (h // 2 + 1)
    torch.manual_seed(42)
    m = TM(Xtr.shape[1], h)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    best_loss, bs = 1e9, None
    for ep in range(3000):
        m.train(); opt.zero_grad()
        loss = nn.MSELoss()(m(Xtr), ytr)
        loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        pred = sc_y.inverse_transform(m(Xte).numpy()).ravel()
    mm, rm = np.mean(np.abs(yte - pred)), spearmanr(yte, pred)[0]
    impr = (1 - mm / naive) * 100
    tag = " <" if mm < best_mae else ""
    if mm < best_mae:
        best_mae, best_h, best_ric = mm, h, rm
    log(f"  h={h:3d} {n_p:5d}参: MAE={mm:.4f} R_IC={rm:+.3f} 改善={impr:+.1f}%{tag}")

# === 最终 ===
log(f"\n=== 最终 ===")
log(f"特征: {Xtr.shape[1]}维, 样本: {Xtr.shape[0]}训/{Xte.shape[0]}测")
log(f"基线(猜均值): MAE={naive:.4f}")
log(f"线性: MAE={ml:.4f} R_IC={rl:+.3f} 改善={(1-ml/naive)*100:+.1f}%")
log(f"RF:   MAE={mr:.4f} R_IC={rr:+.3f} 改善={(1-mr/naive)*100:+.1f}%")
log(f"MLP(h={best_h}): MAE={best_mae:.4f} R_IC={best_ric:+.3f} 改善={(1-best_mae/naive)*100:+.1f}%")
winners = {"线性": (1-ml/naive)*100, "RF": (1-mr/naive)*100, "MLP": (1-best_mae/naive)*100}
winner = max(winners, key=winners.get)
log(f"最优模型: {winner} (+{winners[winner]:.1f}%)")

# 保存训练集
df.to_csv(os.path.join(OUT, "training_final.csv"), index=False, encoding="utf-8-sig")
log("训练集: training_final.csv")
log("DONE")
