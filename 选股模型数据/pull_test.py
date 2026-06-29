#!/usr/bin/env python3
"""拉取历史数据 - 先跑5只验证，再批量"""
import baostock as bs, pandas as pd, numpy as np, time, os, sys

# 兼容性修复
if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda self, other, ignore_index=False, **kw: pd.concat([self, other], ignore_index=ignore_index)

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
os.makedirs(OUT, exist_ok=True)

bs.login()
codes = ["sh.600019", "sh.600282", "sz.000568", "sh.601398", "sz.002555"]
print("测试5只股票", flush=True)

# === 股价 ===
print("拉股价...", flush=True)
rows = []
for i, c in enumerate(codes):
    try:
        rs = bs.query_history_k_data_plus(c, "date,close", start_date="2022-01-01", end_date="2026-06-25", frequency="d", adjustflag="2")
        if rs.error_code == "0":
            df = rs.get_data()
            df["code"] = c
            rows.append(df[["code","date","close"]])
            print(f"  {c}: {len(df)}条", flush=True)
        else:
            print(f"  {c}: error {rs.error_msg}", flush=True)
    except Exception as e:
        print(f"  {c}: {e}", flush=True)
    time.sleep(0.3)

df_p = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
df_p.to_csv(os.path.join(OUT, "prices_daily.csv"), index=False, encoding="utf-8-sig")
print(f"股价: {len(df_p)}条", flush=True)

# === 财报 ===
print("拉财报...", flush=True)
fin_rows = []
for i, c in enumerate(codes):
    print(f"  {c}", flush=True)
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
                        row["gpMargin"] = r.get("gpMargin")
                        row["npMargin"] = r.get("npMargin")
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
                fin_rows.append(row)
            except Exception as e:
                pass
            time.sleep(0.15)

df_f = pd.DataFrame(fin_rows)
df_f.to_csv(os.path.join(OUT, "financials_history.csv"), index=False, encoding="utf-8-sig")
print(f"财报: {len(df_f)}条", flush=True)

bs.logout()
print("完成", flush=True)
