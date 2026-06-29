"""一步到位: 补数据 → 建训练集 → 跑模型 → 写结果"""
import baostock as bs, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import warnings, os, time
warnings.filterwarnings("ignore")
if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda s,o,ig,**kw: pd.concat([s,o],ignore_index=ig)

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
RES = os.path.join(OUT, "final_results.txt")
os.makedirs(OUT, exist_ok=True)

def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(f"[{t}] {msg}\n")

with open(RES, "w", encoding="utf-8") as f:
    f.write("")

log("=== 步骤1: 补资产负债+现金流+估值 ===")

# 加载现有数据
dc = pd.read_csv(os.path.join(OUT, "all_stock_codes_full.csv"), dtype=str)
df_f = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
ci = dict(zip(dc.code, dc.industry))
log(f"现有数据: {len(df_f)}条, {df_f.code.nunique()}只")

bs.login()
t0 = time.time()

# 补列
new_bs = {}; new_cf = {}
for _, r in df_f.iterrows():
    c = r["code"]; cb = ("sh." if c.startswith("sh") else "sz.") + c[2:]
    y, q = int(r["year"]), int(r["quarter"])
    k = (c, y, q)
    try:
        rb = bs.query_balance_data(cb, year=y, quarter=q)
        if rb.error_code == "0":
            d = rb.get_data()
            if len(d) > 0:
                x = d.iloc[0]
                new_bs[k] = (x.get("liabilityToAsset"), x.get("currentRatio"))
    except: pass
    try:
        rc = bs.query_cash_flow_data(cb, year=y, quarter=q)
        if rc.error_code == "0":
            d = rc.get_data()
            if len(d) > 0:
                x = d.iloc[0]
                new_cf[k] = (x.get("CFOToOR"), x.get("CFOToNP"), x.get("ebitToInterest"))
    except: pass
    time.sleep(0.04)

df_f["liability_to_asset"] = df_f.apply(lambda r: new_bs.get((r.code,int(r.year),int(r.quarter)), (None,None))[0], axis=1)
df_f["current_ratio"] = df_f.apply(lambda r: new_bs.get((r.code,int(r.year),int(r.quarter)), (None,None))[1], axis=1)
df_f["cfo_to_revenue"] = df_f.apply(lambda r: new_cf.get((r.code,int(r.year),int(r.quarter)), (None,None,None))[0], axis=1)
df_f["cfo_to_profit"] = df_f.apply(lambda r: new_cf.get((r.code,int(r.year),int(r.quarter)), (None,None,None))[1], axis=1)
df_f["interest_coverage"] = df_f.apply(lambda r: new_cf.get((r.code,int(r.year),int(r.quarter)), (None,None,None))[2], axis=1)
log(f"补完: {time.time()-t0:.0f}s")

# 估值(买入日 PE/PB/PS/收盘价)
log("补估值...")
def bd(y, q):
    if q == 1: return f"{y}-04-30"
    if q == 2: return f"{y}-07-31"
    if q == 3: return f"{y}-10-31"
    return f"{y+1}-04-30"

val_map = {}
for i, (_, r) in enumerate(df_f.iterrows()):
    cb = ("sh." if r.code.startswith("sh") else "sz.") + r.code[2:]
    d = bd(int(r["year"]), int(r["quarter"]))
    try:
        rk = bs.query_history_k_data_plus(cb, "date,close,peTTM,pbMRQ,psTTM", start_date=d, end_date=d, frequency="d", adjustflag="2")
        if rk.error_code == "0":
            dk = rk.get_data()
            if len(dk) > 0:
                x = dk.iloc[-1]
                val_map[i] = (float(x.close), float(x.peTTM), float(x.pbMRQ), float(x.psTTM))
    except: pass
    time.sleep(0.04)
    if (i+1) % 500 == 0: log(f"  估值 {i+1}/{len(df_f)}")

df_f["buy_price"] = np.nan; df_f["pe"] = np.nan; df_f["pb"] = np.nan; df_f["ps"] = np.nan
for i, (bp, pe, pb, ps) in val_map.items():
    df_f.at[i, "buy_price"] = bp
    df_f.at[i, "pe"] = pe
    df_f.at[i, "pb"] = pb
    df_f.at[i, "ps"] = ps
df_f.to_csv(os.path.join(OUT, "financials_history.csv"), index=False, encoding="utf-8-sig")
bs.logout()
log(f"估值完成: {time.time()-t0:.0f}s")

# === 步骤2: 训练集 ===
log("=== 步骤2: 建训练集 ===")
df_p = pd.read_csv(os.path.join(OUT, "prices_daily.csv"), dtype={"code": str})
df_p["date"] = pd.to_datetime(df_p["date"])
df_p["close"] = pd.to_numeric(df_p["close"], errors="coerce")
pm = {}
for c, g in df_p.groupby("code"):
    g = g.dropna(subset=["close"]).sort_values("date")
    pm[c] = g.set_index("date")["close"]

def np_(c, t):
    if c not in pm: return np.nan
    ts = pd.Timestamp(t); s = pm[c]; m = s.index >= ts
    return float(s[m].iloc[0]) if m.any() else float(s.iloc[-1])

rows = []
for _, r in df_f.iterrows():
    c, y, q = str(r.code), int(r.year), int(r.quarter)
    bd_str, sd = bd(y, q), bd(y+1, q) if q < 4 else bd(y+2, q) if q == 4 else None
    if q == 1: sd = f"{y+1}-04-30"
    elif q == 2: sd = f"{y+1}-07-31"
    elif q == 3: sd = f"{y+1}-10-31"
    else: sd = f"{y+2}-04-30"
    pb_val = np_(c, bd_str); ps_val = np_(c, sd)
    ret = (ps_val - pb_val) / pb_val if (pd.notna(pb_val) and pd.notna(ps_val) and pb_val > 0) else np.nan
    rows.append({
        "code": c, "year": y, "quarter": q,
        "roe": r.get("roe"), "gross_margin": r.get("gross_margin"),
        "net_margin": r.get("net_margin"), "profit_yoy": r.get("profit_yoy"),
        "asset_turnover": r.get("asset_turnover"), "inventory_turnover": r.get("inventory_turnover"),
        "liability_to_asset": r.get("liability_to_asset"), "current_ratio": r.get("current_ratio"),
        "cfo_to_revenue": r.get("cfo_to_revenue"), "cfo_to_profit": r.get("cfo_to_profit"),
        "interest_coverage": r.get("interest_coverage"),
        "buy_price": r.get("buy_price"), "pe": r.get("pe"), "pb": r.get("pb"), "ps": r.get("ps"),
        "forward_return": ret, "industry": ci.get(c, ""),
    })

df = pd.DataFrame(rows)
df["usable"] = df["forward_return"].notna() & (df["year"] <= 2024)
u = df[df["usable"]].copy()
log(f"训练集: {len(u)}条, {u.code.nunique()}只, 目标均值={u.forward_return.mean()*100:+.1f}%")

# === 步骤3: 特征 ===
feat = ["roe", "gross_margin", "net_margin", "profit_yoy", "asset_turnover",
        "inventory_turnover", "liability_to_asset", "current_ratio",
        "cfo_to_revenue", "cfo_to_profit", "interest_coverage",
        "buy_price", "pe", "pb", "ps"]
ind = pd.get_dummies(u["industry"], prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1)
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
log(f"训练:{Xtr.shape[0]} 测试:{Xte.shape[0]} 特征:{Xtr.shape[1]}维 基线MAE={naive:.4f}")

# === 步骤4: 模型 ===
# 线性
lr = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
lr.fit(Xtr.numpy(), yt[tm])
pl = lr.predict(Xte.numpy())
ml, rl = np.mean(np.abs(yte - pl)), spearmanr(yte, pl)[0]
log(f"线性: MAE={ml:.4f} R_IC={rl:+.3f} 改善={(1-ml/naive)*100:+.1f}%")

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
for h in [8, 16, 32, 64, 128]:
    n_p = (Xtr.shape[1] * h + h) + (h * h // 2 + h // 2) + (h // 2 + 1)
    torch.manual_seed(42)
    m = TM(Xtr.shape[1], h)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    best_loss, bs = 1e9, None
    for _ in range(3000):
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
    log(f"  h={h:3d} {n_p:5d}参: MAE={mm:.4f} R_IC={rm:+.3f} 改善={impr:+.1f}%{tag}")
    if mm < best_mae: best_mae, best_h = mm, h

# === 汇总 ===
log("")
log(f"=== 最终 ===")
log(f"特征数: {Xtr.shape[1]}, 训练: {Xtr.shape[0]}条, 测试: {Xte.shape[0]}条")
log(f"基线(猜均值): MAE={naive:.4f}")
log(f"线性回归: MAE={ml:.4f} R_IC={rl:+.3f} 改善={(1-ml/naive)*100:+.1f}%")
log(f"随机森林: MAE={mr:.4f} R_IC={rr:+.3f} 改善={(1-mr/naive)*100:+.1f}%")
log(f"MLP(h={best_h}): MAE={best_mae:.4f}")
winner = "线性" if ml <= min(mr, best_mae) else ("RF" if mr <= best_mae else "MLP")
log(f"最优: {winner}")
log("DONE")
