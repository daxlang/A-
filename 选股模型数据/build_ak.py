"""AkShare → 15特征映射: 4行业全量数据集"""
import pandas as pd, numpy as np, os, glob

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
CACHE = os.path.join(OUT, "ak_cache")
QUARTERS = [f"{y}{q}" for y in range(2019, 2025) for q in ["0331", "0630", "0930", "1231"]]
IND_MAP = {"钢铁": ["普钢", "特钢Ⅱ", "冶钢原料"], "白酒": ["白酒Ⅱ"], "银行": ["银行Ⅱ"], "游戏": ["游戏Ⅱ"]}

# 加载全部yjbb+zcfz
all_rows = []
for dt in QUARTERS:
    yj = pd.read_parquet(os.path.join(CACHE, f"yjbb_{dt}.parquet"))
    zc = pd.read_parquet(os.path.join(CACHE, f"zcfz_{dt}.parquet"))
    # 合并: 股票代码
    yj["code"] = yj["股票代码"].astype(str).str.zfill(6)
    zc["code"] = zc["股票代码"].astype(str).str.zfill(6)
    merged = yj.merge(zc, on="code", suffixes=("_yi", "_zc"))
    merged["quarter_date"] = dt
    all_rows.append(merged)
    print(f"  {dt}: yj{len(yj)} + zc{len(zc)} → {len(merged)}")

full = pd.concat(all_rows, ignore_index=True)
print(f"\n全量: {len(full)}条 {full.code.nunique()}只")

# 筛选4行业
full["industry"] = "其他"
for ind_name, ak_names in IND_MAP.items():
    full.loc[full["所处行业"].isin(ak_names), "industry"] = ind_name
n_ind = full[full.industry != "其他"]
print(f"4行业过滤后: {len(n_ind)}条 {n_ind.code.nunique()}只")
print(f"钢铁:{n_ind[n_ind.industry=='钢铁'].code.nunique()} 白酒:{n_ind[n_ind.industry=='白酒'].code.nunique()} 银行:{n_ind[n_ind.industry=='银行'].code.nunique()} 游戏:{n_ind[n_ind.industry=='游戏'].code.nunique()}")

# 特征工程
data = n_ind.copy()
data["year"] = data["quarter_date"].str[:4].astype(int)
data["quarter"] = data["quarter_date"].str[4:6].astype(int)

# ROE(直接用)
data["roe"] = data["净资产收益率"] / 100

# 毛利率(直接用)
data["gross_margin"] = data["销售毛利率"] / 100

# 净利率
data["net_margin"] = data["净利润-净利润"] / data["营业总收入-营业总收入"].replace(0, np.nan)

# 利润同比(直接用, 除以100)
data["profit_yoy"] = data["净利润-同比增长"] / 100

# 资产周转率 = 营收*4 / 总资产
data["asset_turnover"] = data["营业总收入-营业总收入"] * 4 / data["资产-总资产"].replace(0, np.nan)

# 库存周转率 = 营收*4 / 存货
data["inventory_turnover"] = data["营业总收入-营业总收入"] * 4 / data["资产-存货"].replace(0, np.nan)

# 负债率(直接用, 除以100)
data["liability_to_asset"] = data["资产负债率"] / 100

# 流动比率 ≈ (货币+应收+存货) / (应付+预收)
data["cur_asset"] = data["资产-货币资金"] + data["资产-应收账款"] + data["资产-存货"]
data["cur_liab"] = data["负债-应付账款"] + data["负债-预收账款"]
data["current_ratio"] = data["cur_asset"] / data["cur_liab"].replace(0, np.nan)

# 经营CF/营收 = 每股经营CF * (净利/每股收益) / 营收
data["shares_est"] = data["净利润-净利润"] / data["每股收益"].replace(0, np.nan)
data["cfo_to_revenue"] = (data["每股经营现金流量"] * data["shares_est"]) / data["营业总收入-营业总收入"].replace(0, np.nan)

# cfo_to_profit
data["cfo_to_profit"] = (data["每股经营现金流量"] * data["shares_est"]) / data["净利润-净利润"].replace(0, np.nan)

# 利息保障, 价格, 估值 = 缺失(AkShare无)
data["interest_coverage"] = np.nan
data["buy_price"] = np.nan
data["pe"] = np.nan
data["pb"] = np.nan
data["ps"] = np.nan
data["forward_return"] = np.nan

# 清理
KEEP = ["code", "year", "quarter", "forward_return", "industry"]
FEAT15 = ["roe", "gross_margin", "net_margin", "profit_yoy", "asset_turnover", "inventory_turnover",
          "liability_to_asset", "current_ratio", "cfo_to_revenue", "cfo_to_profit", "interest_coverage",
          "buy_price", "pe", "pb", "ps"]
result = data[KEEP + FEAT15].copy()
result["usable"] = True  # 默认全可用

# 统计缺失
for f in FEAT15:
    missing = result[f].isna().sum()
    if missing > 0:
        print(f"  {f}: 缺失{missing}/{len(result)}")

result.to_csv(os.path.join(OUT, "training_akshare.csv"), index=False)
print(f"\n保存: training_akshare.csv {len(result)}条")
