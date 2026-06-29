"""扩展数据集: 补全同行业所有公司"""
import pandas as pd, baostock as bs, numpy as np, time, os

if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda self, other, ignore_index=False, **kw: pd.concat([self, other], ignore_index=ignore_index)

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
TMP = os.path.join(OUT, "tmp"); os.makedirs(TMP, exist_ok=True)

# === 1. 汇总全行业代码 ===
df_clean = pd.read_csv(r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\all_stocks_q1_2026_clean.csv")
existing = set(df_clean["code"])

# 钢铁全量(43只)
steel = pd.read_csv(r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股武器库\真实财报\钢铁行业财报数据\processed\利润表_combined.csv")
steel_codes = sorted(steel["symbol"].unique())

# 白酒全量(19只)
baijiu = pd.read_csv(r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股武器库\真实财报\Q1_2026_白酒\白酒行业2026Q1分析.csv")
baijiu_codes = sorted(baijiu["code"].unique())

# 银行全量(42只) - already all included
bank_codes = sorted(df_clean[df_clean["industry"] == "银行"]["code"].unique())

# 游戏全量(26只)
game = pd.read_csv(r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股武器库\真实财报\Q1_2026_游戏\游戏行业2026Q1分析.csv")
game_codes = sorted(game["code"].unique())

all_codes = []
for codes, ind in [(steel_codes, "钢铁"), (baijiu_codes, "白酒"), (bank_codes, "银行"), (game_codes, "游戏")]:
    for c in codes:
        all_codes.append((c, ind))

print(f"全行业: 钢铁{len(steel_codes)} 白酒{len(baijiu_codes)} 银行{len(bank_codes)} 游戏{len(game_codes)} = {len(all_codes)}只")
print(f"其中已有: {sum(1 for c,_ in all_codes if c in existing)}只, 需补: {sum(1 for c,_ in all_codes if c not in existing)}只")

# 只拉缺失的
to_pull = [(c, ind) for c, ind in all_codes if c not in existing]
codes_bs = [("sh." + c[2:] if c.startswith("sh") else "sz." + c[2:], c, ind) for c, ind in to_pull]
print(f"\n实际需拉取: {len(codes_bs)}只")

if len(codes_bs) == 0:
    print("无需补拉, 退出")
    exit()

# === 2. 拉取 ===
bs.login()
t0 = time.time()

# 股价
print(f"\n=== 拉股价 ===", flush=True)
p_new = []
for i, (c_bs, c_raw, ind) in enumerate(codes_bs):
    try:
        rs = bs.query_history_k_data_plus(c_bs, "date,close", start_date="2022-01-01", end_date="2026-06-25", frequency="d", adjustflag="2")
        if rs.error_code == "0":
            df = rs.get_data()
            df["code"] = c_raw
            p_new.append(df[["code", "date", "close"]])
    except: pass
    if (i+1)%10 == 0: print(f"  股价 {i+1}/{len(codes_bs)} ({len(p_new)}成功)", flush=True)
    time.sleep(0.2)
print(f"  股价 {len(codes_bs)}/{len(codes_bs)} ({len(p_new)}成功)", flush=True)

# 财报
print(f"\n=== 拉财报 (2022Q1~2025Q4) ===", flush=True)
f_new = []
for i, (c_bs, c_raw, ind) in enumerate(codes_bs):
    for y in range(2022, 2026):
        for q in range(1, 5):
            try:
                rp = bs.query_profit_data(c_bs, year=y, quarter=q)
                rg = bs.query_growth_data(c_bs, year=y, quarter=q)
                ro = bs.query_operation_data(c_bs, year=y, quarter=q)
                row = {"code": c_raw, "year": y, "quarter": q, "industry": ind}
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
    if (i+1) % 5 == 0:
        print(f"  财报 {i+1}/{len(codes_bs)} ({len(f_new)}条)", flush=True)

bs.logout()

# === 3. 合并到已有文件 ===
# 股价
if p_new:
    df_p_exist = pd.read_csv(os.path.join(OUT, "prices_daily.csv"), dtype={"code": str})
    df_p_new = pd.concat(p_new, ignore_index=True)
    df_p_all = pd.concat([df_p_exist, df_p_new], ignore_index=True)
    df_p_all.to_csv(os.path.join(OUT, "prices_daily.csv"), index=False, encoding="utf-8-sig")
    print(f"\n股价: +{len(df_p_new)}条 → {len(df_p_all)}条")

# 财报
if f_new:
    df_f_exist = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
    df_f_new = pd.DataFrame(f_new)
    df_f_all = pd.concat([df_f_exist, df_f_new], ignore_index=True)
    df_f_all.to_csv(os.path.join(OUT, "financials_history.csv"), index=False, encoding="utf-8-sig")
    print(f"财报: +{len(df_f_new)}条 → {len(df_f_all)}条")

# 全量代码
all_df = pd.DataFrame(all_codes, columns=["code", "industry"])
all_df.to_csv(os.path.join(OUT, "all_stock_codes_full.csv"), index=False, encoding="utf-8-sig")
print(f"\n全量代码清单: {len(all_df)}只 → all_stock_codes_full.csv")
print(f"完成: {time.time()-t0:.0f}s")
