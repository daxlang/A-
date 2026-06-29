"""扩充时间跨度: 补拉 2019Q1~2021Q4 的财报+股价"""
import baostock as bs, pandas as pd, numpy as np, time, os
if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda self, other, ignore_index=False, **kw: pd.concat([self, other], ignore_index=ignore_index)

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

# 93只通过PE过滤的股票
df_codes = pd.read_csv(os.path.join(OUT, "stocks_after_pe_filter.csv"), dtype=str)
codes_raw = df_codes["code"].tolist()
codes_bs = [("sh." + c[2:] if c.startswith("sh") else "sz." + c[2:], c) for c in codes_raw]
print(f"93只股票, 补拉 2019Q1~2021Q4")

bs.login()
t0 = time.time()

# === 股价: 2019-01-01 ~ 2021-12-31 ===
print("\n=== 拉股价 (2019-2021) ===", flush=True)
p_new = []
for i, (c_bs, c_raw) in enumerate(codes_bs):
    try:
        rs = bs.query_history_k_data_plus(c_bs, "date,close",
            start_date="2019-01-01", end_date="2021-12-31",
            frequency="d", adjustflag="2")
        if rs.error_code == "0":
            d = rs.get_data()
            d["code"] = c_raw
            p_new.append(d[["code", "date", "close"]])
    except: pass
    if (i+1) % 20 == 0:
        print(f"  股价 {i+1}/{len(codes_bs)} ({len(p_new)}成功)", flush=True)
    time.sleep(0.2)
print(f"  股价完成: {len(p_new)}只", flush=True)

# === 财报: 2019Q1 ~ 2021Q4 ===
print("\n=== 拉财报 (2019Q1~2021Q4) ===", flush=True)
f_new = []
for i, (c_bs, c_raw) in enumerate(codes_bs):
    for y in range(2019, 2022):
        for q in range(1, 5):
            try:
                rp = bs.query_profit_data(c_bs, year=y, quarter=q)
                rg = bs.query_growth_data(c_bs, year=y, quarter=q)
                ro = bs.query_operation_data(c_bs, year=y, quarter=q)
                row = {"code": c_raw, "year": y, "quarter": q}
                if rp.error_code == "0":
                    d = rp.get_data()
                    if len(d) > 0:
                        r = d.iloc[0]
                        row["roe"] = r.get("roeAvg")
                        row["gross_margin"] = r.get("gpMargin")
                        row["net_margin"] = r.get("npMargin")
                if rg.error_code == "0":
                    d = rg.get_data()
                    if len(d) > 0:
                        r = d.iloc[0]
                        row["profit_yoy"] = r.get("YOYNI")
                if ro.error_code == "0":
                    d = ro.get_data()
                    if len(d) > 0:
                        r = d.iloc[0]
                        row["asset_turnover"] = r.get("AssetTurnRatio")
                        row["inventory_turnover"] = r.get("INVTurnRatio")
                f_new.append(row)
            except: pass
            time.sleep(0.1)
    if (i+1) % 10 == 0:
        print(f"  财报 {i+1}/{len(codes_bs)} ({len(f_new)}条)", flush=True)

bs.logout()

# === 合并 ===
if p_new:
    df_p_exist = pd.read_csv(os.path.join(OUT, "prices_daily.csv"), dtype={"code": str})
    df_p_new = pd.concat(p_new, ignore_index=True)
    df_p_all = pd.concat([df_p_exist, df_p_new], ignore_index=True)
    df_p_all = df_p_all.drop_duplicates(subset=["code", "date"])
    df_p_all.to_csv(os.path.join(OUT, "prices_daily.csv"), index=False, encoding="utf-8-sig")
    print(f"\n股价: +{len(df_p_new)}条, 去重后 {len(df_p_all)}条")

if f_new:
    df_f_exist = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
    df_f_new = pd.DataFrame(f_new)
    df_f_all = pd.concat([df_f_exist, df_f_new], ignore_index=True)
    df_f_all = df_f_all.drop_duplicates(subset=["code", "year", "quarter"])
    df_f_all.to_csv(os.path.join(OUT, "financials_history.csv"), index=False, encoding="utf-8-sig")
    print(f"财报: +{len(df_f_new)}条, 去重后 {len(df_f_all)}条")

print(f"\n总耗时: {time.time()-t0:.0f}s")
