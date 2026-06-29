"""补资产负债表+现金流+估值, 追加到现有financials_history.csv"""
import baostock as bs, pandas as pd, numpy as np, time, os
if not hasattr(pd.DataFrame,"append"):
    pd.DataFrame.append = lambda s,o,ig,**kw: pd.concat([s,o],ignore_index=ig)

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
PROG = os.path.join(OUT, "p2.txt")

def p(msg):
    print(msg, flush=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

with open(PROG, "w") as f: f.write("")

df_f = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
p(f"数据: {len(df_f)}条, {df_f.code.nunique()}只")

# 登录(重试)
for attempt in range(5):
    bs.logout()
    time.sleep(1)
    lg = bs.login()
    if lg.error_code == "0":
        p(f"登录成功(第{attempt+1}次)")
        break
    p(f"登录失败(第{attempt+1}次): {lg.error_msg}")
    time.sleep(3)
else:
    p("登录失败, 退出")
    exit(1)

# 1. 拉资产负债表+现金流量表
p("1/3 拉资产负债+现金流...")
bs_data = []; cf_data = []
logged_in = lg.error_code == "0"
for i, (_, r) in enumerate(df_f.iterrows()):
    c = r["code"]; cb = ("sh." if c.startswith("sh") else "sz.") + c[2:]
    y, q = int(r["year"]), int(r["quarter"])
    if not logged_in:
        p("登录失败, 中断")
        break
    try:
        rb = bs.query_balance_data(cb, year=y, quarter=q)
        if rb.error_code == "0": d = rb.get_data()
        else: d = pd.DataFrame()
        if len(d) > 0:
            x = d.iloc[0]
            bs_data.append({"code": c, "year": y, "quarter": q,
                "liability_to_asset": x.get("liabilityToAsset"),
                "current_ratio": x.get("currentRatio")})
    except: pass
    try:
        rc = bs.query_cash_flow_data(cb, year=y, quarter=q)
        if rc.error_code == "0": d = rc.get_data()
        else: d = pd.DataFrame()
        if len(d) > 0:
            x = d.iloc[0]
            cf_data.append({"code": c, "year": y, "quarter": q,
                "cfo_to_revenue": x.get("CFOToOR"),
                "cfo_to_profit": x.get("CFOToNP"),
                "interest_coverage": x.get("ebitToInterest")})
    except: pass
    time.sleep(0.04)
    if (i+1) % 500 == 0: p(f"  {i+1}/{len(df_f)}")

df_bs = pd.DataFrame(bs_data); df_cf = pd.DataFrame(cf_data)
p(f"BS:{len(df_bs)}条 CF:{len(df_cf)}条")

# 2. 拉估值 (PE/PB/PS/收盘价)
p("2/3 拉估值...")
def bd(y, q):
    if q == 1: return f"{y}-04-30"
    if q == 2: return f"{y}-07-31"
    if q == 3: return f"{y}-10-31"
    return f"{y+1}-04-30"

val_data = []
for i, (_, r) in enumerate(df_f.iterrows()):
    c = r["code"]; cb = ("sh." if c.startswith("sh") else "sz.") + c[2:]
    y, q = int(r["year"]), int(r["quarter"])
    try:
        rk = bs.query_history_k_data_plus(cb, "date,close,peTTM,pbMRQ,psTTM",
            start_date=bd(y,q), end_date=bd(y,q), frequency="d", adjustflag="2")
        if rk.error_code == "0": dk = rk.get_data()
        else: dk = pd.DataFrame()
        if len(dk) > 0:
            x = dk.iloc[-1]
            val_data.append({"code": c, "year": y, "quarter": q,
                "buy_price": float(x.close), "pe": float(x.peTTM),
                "pb": float(x.pbMRQ), "ps": float(x.psTTM)})
    except: pass
    time.sleep(0.04)
    if (i+1) % 500 == 0: p(f"  {i+1}/{len(df_f)}")

df_val = pd.DataFrame(val_data)
p(f"估值:{len(df_val)}条")

# 3. 合并
p("3/3 合并保存...")
key = ["code", "year", "quarter"]
for k in key: df_f[k] = df_f[k].astype(str)
for k in key: df_bs[k] = df_bs[k].astype(str)
for k in key: df_cf[k] = df_cf[k].astype(str)
for k in key: df_val[k] = df_val[k].astype(str)

df_f = df_f.merge(df_bs, on=key, how="left", suffixes=("", "_bs"))
df_f = df_f.merge(df_cf, on=key, how="left", suffixes=("", "_cf"))
df_f = df_f.merge(df_val, on=key, how="left", suffixes=("", "_val"))
df_f.to_csv(os.path.join(OUT, "financials_history.csv"), index=False, encoding="utf-8-sig")
bs.logout()

new_cols = [c for c in df_f.columns if c not in ["code","year","quarter","roe","gross_margin","net_margin","profit_yoy","asset_turnover","inventory_turnover"]]
p(f"完成: {len(df_f)}条, 新增 {len(new_cols)} 列: {new_cols}")
for c in new_cols:
    p(f"  {c}: {df_f[c].notna().sum()}/{len(df_f)}")
p("DONE")
