#!/usr/bin/env python3
"""批量拉取80家公司历史数据（baostock）"""
import baostock as bs, pandas as pd, numpy as np, time, os, sys

if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda self, other, ignore_index=False, **kw: pd.concat([self, other], ignore_index=ignore_index)

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
os.makedirs(OUT, exist_ok=True)

STOCKS = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\all_stocks_q1_2026_clean.csv"
df_stocks = pd.read_csv(STOCKS, encoding="utf-8-sig")

def to_bs(c):
    return ("sh." if c.startswith("sh") else "sz.") + c[2:]

codes_raw = [(row["code"], row["name"]) for _, row in df_stocks.iterrows()]
codes_bs = [(to_bs(c), n) for c, n in codes_raw]
n_total = len(codes_bs)
print(f"共 {n_total} 只股票", flush=True)

bs.login()
t0 = time.time()

# === 股价 ===
print("\n=== 拉股价 ===", flush=True)
p_all = []
for i, (c, name) in enumerate(codes_bs):
    try:
        rs = bs.query_history_k_data_plus(c, "date,close", start_date="2022-01-01", end_date="2026-06-25", frequency="d", adjustflag="2")
        if rs.error_code=="0":
            df = rs.get_data()
            df["code"] = c.replace(".","")
            p_all.append(df[["code","date","close"]])
    except Exception as e:
        pass
    if (i+1)%10==0 or i==n_total-1:
        print(f"  股价 {i+1}/{n_total} ({len(p_all)}成功)", flush=True)
    time.sleep(0.2)

df_p = pd.concat(p_all, ignore_index=True) if p_all else pd.DataFrame()
df_p.to_csv(os.path.join(OUT, "prices_daily.csv"), index=False, encoding="utf-8-sig")
print(f"股价: {len(df_p)} 条 ({time.time()-t0:.0f}s)", flush=True)

# === 财报 ===
print("\n=== 拉财报 (2022Q1~2025Q4) ===", flush=True)
f_all = []
for i, (c, name) in enumerate(codes_bs):
    for y in range(2022, 2026):
        for q in range(1, 5):
            try:
                rp = bs.query_profit_data(c, year=y, quarter=q)
                rg = bs.query_growth_data(c, year=y, quarter=q)
                ro = bs.query_operation_data(c, year=y, quarter=q)
                row = {"code": c.replace(".",""), "year": y, "quarter": q}
                if rp.error_code=="0":
                    d = rp.get_data()
                    if len(d)>0:
                        r = d.iloc[0]
                        row["roe"] = r.get("roeAvg")
                        row["gross_margin"] = r.get("gpMargin")
                        row["net_margin"] = r.get("npMargin")
                if rg.error_code=="0":
                    d = rg.get_data()
                    if len(d)>0:
                        r = d.iloc[0]
                        row["rev_yoy"] = r.get("YOYOperateIncome")
                        row["profit_yoy"] = r.get("YOYNI")
                if ro.error_code=="0":
                    d = ro.get_data()
                    if len(d)>0:
                        r = d.iloc[0]
                        row["asset_turnover"] = r.get("AssetTurnRatio")
                        row["inventory_turnover"] = r.get("INVTurnRatio")
                f_all.append(row)
            except:
                pass
            time.sleep(0.12)
    if (i+1)%5==0 or i==n_total-1:
        print(f"  财报 {i+1}/{n_total} ({len(f_all)}条)", flush=True)

df_f = pd.DataFrame(f_all)
df_f.to_csv(os.path.join(OUT, "financials_history.csv"), index=False, encoding="utf-8-sig")
print(f"财报: {len(df_f)} 条 ({time.time()-t0:.0f}s)", flush=True)

bs.logout()
print(f"\n完成: {time.time()-t0:.0f}秒", flush=True)
