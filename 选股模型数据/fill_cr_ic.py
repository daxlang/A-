"""补 current_ratio(从ths) + interest_coverage(用原数据行业填) → AkShare数据集"""
import pandas as pd, numpy as np, os, time, glob
import akshare as ak

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
PC = os.path.join(OUT, "price_cache")
CR_CACHE = os.path.join(OUT, "cr_cache")
os.makedirs(CR_CACHE, exist_ok=True)

# 加载现有数据
df = pd.read_csv(os.path.join(OUT, "training_full.csv"), dtype={"code": str})
print(f"现有数据: {len(df)}条 {df.code.nunique()}只")

# === current_ratio: 从stock_financial_abstract_ths拉 ===
print("\n补 current_ratio (ths接口, 约25分钟)...")
all_codes = sorted(df[df.usable].code.unique())
cr_map = {}
for i, code in enumerate(all_codes):
    fp = os.path.join(CR_CACHE, f"{code}.csv")
    if os.path.exists(fp):
        cr_map[code] = pd.read_csv(fp, dtype={"code": str})
        continue
    try:
        raw = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        # 提取日期+流动比率
        raw = raw[["报告期", "流动比率"]].rename(columns={"报告期": "report_date", "流动比率": "current_ratio_th"})
        raw["year"] = pd.to_datetime(raw["report_date"]).dt.year
        raw["quarter"] = pd.to_datetime(raw["report_date"]).dt.quarter
        raw["code"] = code
        raw.to_csv(fp, index=False)
        cr_map[code] = raw
        time.sleep(0.3)
    except Exception as e:
        pass
    if (i + 1) % 30 == 0:
        print(f"  [{i+1}/{len(all_codes)}] {len(cr_map)}只OK")

print(f"  流动比率获取: {len(cr_map)}/{len(all_codes)}只")

# 合并到数据集
cr_all = pd.concat(cr_map.values(), ignore_index=True)
cr_all["qdate_key"] = cr_all["year"].astype(str) + cr_all["quarter"].astype(str).str.zfill(2)
df["qdate_key"] = df["year"].astype(str) + df["quarter"].astype(str).str.zfill(2)

cr_lookup = cr_all.set_index(["code", "qdate_key"])["current_ratio_th"].to_dict()
df["current_ratio_ak"] = df.apply(lambda r: cr_lookup.get((r["code"], r["qdate_key"]), np.nan), axis=1)
# 缺失的用行业-年-季中位填补
df["current_ratio"] = df["current_ratio_ak"].fillna(
    df.groupby(["industry", "year", "quarter"])["current_ratio_ak"].transform("median")
)
# 仍有缺失的用全局中位
df["current_ratio"] = df["current_ratio"].fillna(df["current_ratio"].median())

crm = df["current_ratio"].isna().sum()
print(f"  current_ratio 最终缺失: {crm}/{len(df)}")

# === interest_coverage: 从原始数据行业-年-季填 ===
print("\n补 interest_coverage (原始4行业中位数)...")
orig = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
orig_u = orig[orig.usable]
ic_med = orig_u.groupby(["industry", "year", "quarter"])["interest_coverage"].median().reset_index()
ic_med = ic_med.rename(columns={"interest_coverage": "ic_fill"})

df = df.merge(ic_med, on=["industry", "year", "quarter"], how="left")
df["interest_coverage"] = df["ic_fill"]
# 新行业没有原始数据→用所有行业全局中位
df["interest_coverage"] = df["interest_coverage"].fillna(orig_u["interest_coverage"].median())
icm = df["interest_coverage"].isna().sum()
print(f"  interest_coverage 最终缺失: {icm}/{len(df)}")

# 保存
KEEP = ["code", "year", "quarter", "forward_return", "industry"]
FEAT15 = ["roe", "gross_margin", "net_margin", "profit_yoy", "asset_turnover", "inventory_turnover",
          "liability_to_asset", "current_ratio", "cfo_to_revenue", "cfo_to_profit", "interest_coverage",
          "buy_price", "pe", "pb", "ps"]
# 确保列存在
for f in FEAT15:
    if f not in df.columns:
        df[f] = np.nan

result = df[KEEP + FEAT15 + ["usable"]].copy()
result.to_csv(os.path.join(OUT, "training_full.csv"), index=False)

# 统计
print(f"\n保存: training_full.csv {len(result)}条")
for f in FEAT15:
    m = result[f].isna().sum()
    if m > 0:
        print(f"  {f}: 缺失{m}/{len(result)}")

print(f"\n可用: {result.usable.sum()}条 {result[result.usable].code.nunique()}只")
print("行业分布:")
for ind in sorted(result.industry.unique()):
    n_u = result[(result.industry == ind) & result.usable].code.nunique()
    n = result[result.industry == ind].code.nunique()
    print(f"  {ind}: {n_u}/{n}只有价格")
