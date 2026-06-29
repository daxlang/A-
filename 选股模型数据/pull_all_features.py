"""全量重拉: 130只 x 2019-2025 x 5张表 + 估值, 写进度文件"""
import baostock as bs, pandas as pd, numpy as np, time, os, sys
if not hasattr(pd.DataFrame,"append"):
    pd.DataFrame.append = lambda s,o,ig,**kw: pd.concat([s,o],ignore_index=ig)

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
LOG = os.path.join(OUT, "pull_progress.txt")
os.makedirs(OUT, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

# 清空日志
with open(LOG, "w", encoding="utf-8") as f:
    f.write("")

log("开始: 130只 x 2019-2025 x 5张表 + 估值")

dc = pd.read_csv(os.path.join(OUT, "all_stock_codes_full.csv"), dtype=str)
codes = [(("sh." + c[2:] if c.startswith("sh") else "sz." + c[2:]), c) for c in dc.code]
n = len(codes)

bs.login()
t0 = time.time()

# === 1. 财报五表合并 (2019-2025) ===
log(f"=== 1. 财报五表 ===")
f_all = []
for i, (cb, cr) in enumerate(codes):
    for y in range(2019, 2026):
        for q in range(1, 5):
            try:
                rp = bs.query_profit_data(cb, year=y, quarter=q)
                rg = bs.query_growth_data(cb, year=y, quarter=q)
                ro = bs.query_operation_data(cb, year=y, quarter=q)
                rb = bs.query_balance_data(cb, year=y, quarter=q)
                rc = bs.query_cash_flow_data(cb, year=y, quarter=q)
            except:
                continue
            
            row = {"code": cr, "year": y, "quarter": q}
            
            # 利润表
            try:
                d = rp.get_data() if rp.error_code == "0" else pd.DataFrame()
                if len(d) > 0:
                    r = d.iloc[0]
                    row["roe"] = r.get("roeAvg")
                    row["gross_margin"] = r.get("gpMargin")
                    row["net_margin"] = r.get("npMargin")
            except: pass
            
            # 成长
            try:
                d = rg.get_data() if rg.error_code == "0" else pd.DataFrame()
                if len(d) > 0:
                    row["profit_yoy"] = d.iloc[0].get("YOYNI")
            except: pass
            
            # 营运
            try:
                d = ro.get_data() if ro.error_code == "0" else pd.DataFrame()
                if len(d) > 0:
                    r = d.iloc[0]
                    row["asset_turnover"] = r.get("AssetTurnRatio")
                    row["inventory_turnover"] = r.get("INVTurnRatio")
            except: pass
            
            # 资产负债
            try:
                d = rb.get_data() if rb.error_code == "0" else pd.DataFrame()
                if len(d) > 0:
                    r = d.iloc[0]
                    row["liability_to_asset"] = r.get("liabilityToAsset")
                    row["current_ratio"] = r.get("currentRatio")
            except: pass
            
            # 现金流
            try:
                d = rc.get_data() if rc.error_code == "0" else pd.DataFrame()
                if len(d) > 0:
                    r = d.iloc[0]
                    row["cfo_to_revenue"] = r.get("CFOToOR")
                    row["cfo_to_profit"] = r.get("CFOToNP")
                    row["interest_coverage"] = r.get("ebitToInterest")
            except: pass
            
            f_all.append(row)
            time.sleep(0.06)
    
    if (i + 1) % 20 == 0:
        log(f"  财报 {i+1}/{n} ({len(f_all)}条, {time.time()-t0:.0f}s)")

df_f = pd.DataFrame(f_all)
log(f"财报完成: {len(df_f)}条, {time.time()-t0:.0f}s")

# 保存中间结果(不含估值)
df_f.to_csv(os.path.join(OUT, "financials_tmp.csv"), index=False, encoding="utf-8-sig")

# === 2. 估值数据 (买入日PE/PB/PS/收盘价) ===
log(f"=== 2. 估值 ===")
def buy_date(y, q):
    if q == 1: return f"{y}-04-30"
    if q == 2: return f"{y}-07-31"
    if q == 3: return f"{y}-10-31"
    return f"{y+1}-04-30"

val_cols = {}
for i, row in df_f.iterrows():
    cr = row["code"]
    cb = "sh." + cr[2:] if cr.startswith("sh") else "sz." + cr[2:]
    y, q = int(row["year"]), int(row["quarter"])
    bd = buy_date(y, q)
    try:
        rk = bs.query_history_k_data_plus(cb, "date,close,peTTM,pbMRQ,psTTM",
                                          start_date=bd, end_date=bd, frequency="d", adjustflag="2")
        if rk.error_code == "0":
            dk = rk.get_data()
            if len(dk) > 0:
                x = dk.iloc[-1]
                val_cols[i] = {
                    "buy_price": float(x.get("close", 0)),
                    "pe": float(x.get("peTTM", 0)),
                    "pb": float(x.get("pbMRQ", 0)),
                    "ps": float(x.get("psTTM", 0)),
                }
    except: pass
    time.sleep(0.06)
    if (i + 1) % 500 == 0:
        log(f"  估值 {i+1}/{len(df_f)} ({time.time()-t0:.0f}s)")

# 合并估值
df_f["buy_price"] = np.nan; df_f["pe"] = np.nan; df_f["pb"] = np.nan; df_f["ps"] = np.nan
for i, vals in val_cols.items():
    for k, v in vals.items():
        df_f.at[i, k] = v

# === 保存 ===
df_f.to_csv(os.path.join(OUT, "financials_history.csv"), index=False, encoding="utf-8-sig")
bs.logout()

n_cols = len(df_f.columns)
n_rows = len(df_f)
cov = {c: df_f[c].notna().sum() for c in df_f.columns if c not in ["code","year","quarter"]}
log(f"完成: {n_rows}条, {n_cols}列, {time.time()-t0:.0f}s")
log(f"覆盖: {cov}")
log("DONE")
