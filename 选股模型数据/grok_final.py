"""大参数MLP 30万轮x4 完整对比"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

df_pe = pd.read_csv(os.path.join(OUT, "stocks_after_pe_filter.csv"), dtype=str)
pev = set(df_pe.code.tolist())
df_f = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
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

def gd(y, q):
    if q == 1: return (f"{y}-04-30", f"{y+1}-04-30")
    if q == 2: return (f"{y}-07-31", f"{y+1}-07-31")
    if q == 3: return (f"{y}-10-31", f"{y+1}-10-31")
    return (f"{y+1}-04-30", f"{y+2}-04-30")

dc = pd.read_csv(os.path.join(OUT, "all_stock_codes_full.csv"), dtype=str)
ci = dict(zip(dc.code, dc.industry))

rows = []
for _, r in df_f.iterrows():
    c = str(r["code"]); y = int(r["year"]); q = int(r["quarter"])
    bd, sd = gd(y, q)
    pb = np_(c, bd); ps = np_(c, sd)
    ret = (ps - pb) / pb if (pd.notna(pb) and pd.notna(ps) and pb > 0) else np.nan
    rows.append({"code": c, "year": y, "quarter": q, "roe": r.get("roe"),
        "gross_margin": r.get("gross_margin"), "net_margin": r.get("net_margin"),
        "profit_yoy": r.get("profit_yoy"), "asset_turnover": r.get("asset_turnover"),
        "inventory_turnover": r.get("inventory_turnover"),
        "forward_return": ret, "industry": ci.get(c, "")})

df = pd.DataFrame(rows)
df = df[df["code"].isin(pev) & (df["year"] >= 2022)]
df["usable"] = df["forward_return"].notna() & (df["year"] <= 2024)
u = df[df["usable"]]

feat = ["roe", "gross_margin", "net_margin", "profit_yoy", "asset_turnover", "inventory_turnover"]
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

log_path = os.path.join(OUT, "grok_results.txt")
with open(log_path, "w") as log:
    log.write(f"B组PE过滤 930训/186测 基线MAE={naive:.4f}\n\n")

    for h, lr, wd, desc in [(64, 1e-3, 5e-3, "h=64"), (128, 1e-3, 5e-3, "h=128"),
                               (256, 5e-4, 1e-2, "h=256"), (512, 3e-4, 2e-2, "h=512")]:
        n_in = Xtr.shape[1]
        n_p = (n_in * h + h) + (h * h // 2 + h // 2) + (h // 2 + 1)
        m = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h // 2, 1)
        )
        opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()
        best_te = 1e9; best_st = None; best_ep = 0

        for ep in range(1, 300001):
            m.train(); opt.zero_grad()
            loss = loss_fn(m(Xtr), ytr)
            loss.backward(); opt.step()
            if ep % 50000 == 0:
                m.eval()
                with torch.no_grad():
                    pte = sc_y.inverse_transform(m(Xte).numpy()).ravel()
                te_m = np.mean(np.abs(yte - pte)); te_r, _ = spearmanr(yte, pte)
                if te_m < best_te:
                    best_te = te_m; best_ep = ep
                    best_st = {k: v.clone() for k, v in m.state_dict().items()}

        m.load_state_dict(best_st)
        with torch.no_grad():
            pf = sc_y.inverse_transform(m(Xte).numpy()).ravel()
        fm = np.mean(np.abs(yte - pf)); fr, _ = spearmanr(yte, pf)
        impr = (1 - fm / naive) * 100
        line = f"{desc} {n_p:>6d}参: MAE={fm:.4f} R_IC={fr:+.3f} 改善={impr:+.1f}% 最佳@{best_ep//1000}k"
        print(line, flush=True)
        log.write(line + "\n")

    log.write(f"\n基线MAE={naive:.4f}")

print(f"\n结果已写入: {log_path}")
