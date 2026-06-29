"""一步到位: 130只 × 2019-2025 全量拉取"""
import baostock as bs, pandas as pd, time, os
if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda s, o, ignore_index=False, **kw: pd.concat([s, o], ignore_index=ignore_index)

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
df_codes = pd.read_csv(os.path.join(OUT, "all_stock_codes_full.csv"), dtype=str)
codes = [(("sh." if c.startswith("sh") else "sz.") + c[2:], c) for c in df_codes["code"]]
n = len(codes)
print(f"130只, 2019Q1~2025Q4", flush=True)
bs.login()
t0 = time.time()

# === 股价 ===
print("股价...", flush=True)
p_all = []
for i, (cb, cr) in enumerate(codes):
    try:
        rs = bs.query_history_k_data_plus(cb, "date,close", start_date="2019-01-01", end_date="2026-06-25", frequency="d", adjustflag="2")
        if rs.error_code == "0":
            d = rs.get_data(); d["code"] = cr
            p_all.append(d[["code", "date", "close"]])
    except: pass
    if (i+1) % 30 == 0: print(f"  股价 {i+1}/{n}", flush=True)
    time.sleep(0.15)

df_p = pd.concat(p_all, ignore_index=True)
df_p.to_csv(os.path.join(OUT, "prices_daily.csv"), index=False, encoding="utf-8-sig")
print(f"股价: {len(df_p)}条 {time.time()-t0:.0f}s", flush=True)

# === 财报 ===
print("财报...", flush=True)
f_all = []
for i, (cb, cr) in enumerate(codes):
    for y in range(2019, 2026):
        for q in range(1, 5):
            try:
                rp = bs.query_profit_data(cb, year=y, quarter=q)
                rg = bs.query_growth_data(cb, year=y, quarter=q)
                ro = bs.query_operation_data(cb, year=y, quarter=q)
                row = {"code": cr, "year": y, "quarter": q}
                rp_d = None; rg_d = None; ro_d = None
                if rp.error_code == "0":
                    rp_d = rp.get_data()
                if rg.error_code == "0":
                    rg_d = rg.get_data()
                if ro.error_code == "0":
                    ro_d = ro.get_data()
                if rp_d is not None and len(rp_d) > 0:
                    r = rp_d.iloc[0]
                    row["roe"] = r.get("roeAvg")
                    row["gross_margin"] = r.get("gpMargin")
                    row["net_margin"] = r.get("npMargin")
                if rg_d is not None and len(rg_d) > 0:
                    r = rg_d.iloc[0]
                    row["profit_yoy"] = r.get("YOYNI")
                if ro_d is not None and len(ro_d) > 0:
                    r = ro_d.iloc[0]
                    row["asset_turnover"] = r.get("AssetTurnRatio")
                    row["inventory_turnover"] = r.get("INVTurnRatio")
                f_all.append(row)
            except: pass
            time.sleep(0.08)
    if (i+1) % 10 == 0: print(f"  财报 {i+1}/{n}", flush=True)

df_f = pd.DataFrame(f_all)
df_f.to_csv(os.path.join(OUT, "financials_history.csv"), index=False, encoding="utf-8-sig")
print(f"财报: {len(df_f)}条 {time.time()-t0:.0f}s", flush=True)

bs.logout()
print(f"完成: {time.time()-t0:.0f}s", flush=True)
