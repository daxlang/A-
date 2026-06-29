"""方案B: 8行业全量数据集 — 财报+价格"""
import pandas as pd, numpy as np, os, time, pickle
import akshare as ak

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
CACHE = os.path.join(OUT, "ak_cache")
os.makedirs(os.path.join(OUT, "price_cache"), exist_ok=True)

QUARTERS = [f"{y}{q}" for y in range(2019, 2025) for q in ["0331", "0630", "0930", "1231"]]
NDU_CODES = {"钢铁": ["普钢", "特钢Ⅱ", "冶钢原料"], "白酒": ["白酒Ⅱ"], "银行": ["银行Ⅱ"], "游戏": ["游戏Ⅱ"],
             "半导体": ["半导体"], "电力": ["电力"], "化学制药": ["化学制药"], "汽车零部件": ["汽车零部件"]}
ALL_INDS = list(NDU_CODES.keys())

# ===== Step 1: 财报特征映射 =====
print("Step1: 财报特征映射(从缓存)...")
rows = []
for dt in QUARTERS:
    yj = pd.read_parquet(os.path.join(CACHE, f"yjbb_{dt}.parquet"))
    zc = pd.read_parquet(os.path.join(CACHE, f"zcfz_{dt}.parquet"))
    yj["code"] = yj["股票代码"].astype(str).str.zfill(6)
    zc["code"] = zc["股票代码"].astype(str).str.zfill(6)
    m = yj.merge(zc, on="code", suffixes=("_yi", "_zc"))
    m["qdate"] = dt
    rows.append(m)
full = pd.concat(rows, ignore_index=True)
full["industry"] = "其他"
for ind, names in NDU_CODES.items():
    full.loc[full["所处行业"].isin(names), "industry"] = ind

data = full[full.industry.isin(ALL_INDS)].copy()
data["year"] = data["qdate"].str[:4].astype(int)
data["quarter"] = data["qdate"].str[4:6].astype(int)

# 特征计算
data["roe"] = data["净资产收益率"] / 100
data["gross_margin"] = data["销售毛利率"] / 100
data["net_margin"] = data["净利润-净利润"] / data["营业总收入-营业总收入"].replace(0, np.nan)
data["profit_yoy"] = data["净利润-同比增长"] / 100
data["asset_turnover"] = data["营业总收入-营业总收入"] * 4 / data["资产-总资产"].replace(0, np.nan)
data["inventory_turnover"] = data["营业总收入-营业总收入"] * 4 / data["资产-存货"].replace(0, np.nan)
data["liability_to_asset"] = data["资产负债率"] / 100
cur_a = data["资产-货币资金"] + data["资产-应收账款"] + data["资产-存货"]
cur_l = data["负债-应付账款"] + data["负债-预收账款"]
data["current_ratio"] = cur_a / cur_l.replace(0, np.nan)
shares = data["净利润-净利润"] / data["每股收益"].replace(0, np.nan)
data["cfo_to_revenue"] = (data["每股经营现金流量"] * shares) / data["营业总收入-营业总收入"].replace(0, np.nan)
data["cfo_to_profit"] = (data["每股经营现金流量"] * shares) / data["净利润-净利润"].replace(0, np.nan)
data["interest_coverage"] = np.nan  # 无利息支出数据
print(f"  财报条数: {len(data)}  股票数: {data.code.nunique()}")
print(f"  行业分布: {dict(data.groupby('industry').size())}")

# ===== Step 2: 拉价格(缓存) =====
print("\nStep2: 拉价格(缓存已有跳过)...")
all_codes = sorted(data.code.unique())
prices = {}
PRICE_CACHE = os.path.join(OUT, "price_cache")
for i, code in enumerate(all_codes):
    fp = os.path.join(PRICE_CACHE, f"{code}.pkl")
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            prices[code] = pickle.load(f)
        continue
    try:
        symbol = code
        df_p = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20190101", end_date="20250601", adjust="qfq")
        df_p = df_p[["日期", "收盘", "股票代码"]].rename(columns={"股票代码": "code_p", "收盘": "close"})
        df_p["date"] = pd.to_datetime(df_p["日期"])
        prices[code] = df_p
        with open(fp, "wb") as f:
            pickle.dump(df_p, f)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(all_codes)}] 已完成")
        time.sleep(0.3)
    except Exception as e:
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}] {code} skip: {e}")
        time.sleep(0.5)

print(f"  价格缓存: {len(prices)}/{len(all_codes)}")

# ===== Step 3: 计算buy_price和forward_return =====
print("\nStep3: 计算估值和forward_return...")
buy_prices = []
fwd_rets = []
pes = []; pbs = []; pss = []

for _, row in data.iterrows():
    code = row["code"]
    qdate_str = row["qdate"]
    qend = pd.Timestamp(f"{qdate_str[:4]}-{qdate_str[4:6]}-{int(qdate_str[4:6])}")
    # quarter end date: last day of the month
    qend = qend + pd.offsets.MonthEnd(0)
    fy_end = qend + pd.DateOffset(years=1)

    p_df = prices.get(code)
    bp, fr, pe, pv, ps = np.nan, np.nan, np.nan, np.nan, np.nan

    if p_df is not None:
        # buy_price: quarter-end close or nearest
        before = p_df[p_df["date"] <= qend]
        if len(before) > 0:
            bp = before["close"].iloc[-1]
        # sell_price: 1 year later
        after = p_df[(p_df["date"] <= fy_end) & (p_df["date"] > qend + pd.Timedelta(days=340))]
        if len(after) > 0 and bp > 0:
            sp = after["close"].iloc[-1]
            fr = (sp / bp - 1)

        # PE/PB/PS
        eps = row.get("每股收益", np.nan)
        if pd.notna(eps) and eps > 0 and bp > 0:
            pe = bp / eps
        bps = row.get("每股净资产", np.nan)
        if pd.notna(bps) and bps > 0 and bp > 0:
            pv = bp / bps
        rev_per_share = row["营业总收入-营业总收入"] / shares.iloc[row.name] if pd.notna(shares.iloc[row.name]) else np.nan
        if pd.notna(rev_per_share) and rev_per_share > 0 and bp > 0:
            ps = bp / rev_per_share

    buy_prices.append(bp); fwd_rets.append(fr)
    pes.append(pe); pbs.append(pv); pss.append(ps)

data["buy_price"] = buy_prices
data["forward_return"] = fwd_rets
data["pe"] = pes
data["pb"] = pbs
data["ps"] = pss

# ===== Step 4: 保存 =====
KEEP = ["code", "year", "quarter", "forward_return", "industry"]
FEAT15 = ["roe", "gross_margin", "net_margin", "profit_yoy", "asset_turnover", "inventory_turnover",
          "liability_to_asset", "current_ratio", "cfo_to_revenue", "cfo_to_profit", "interest_coverage",
          "buy_price", "pe", "pb", "ps"]
result = data[KEEP + FEAT15].copy()
result["usable"] = result["forward_return"].notna()
missing_stats = {f: result[f].isna().sum() for f in FEAT15}
print(f"\n缺失统计:")
for f, v in missing_stats.items():
    if v > 0:
        print(f"  {f}: {v}/{len(result)} ({v/len(result)*100:.0f}%)")

result.to_csv(os.path.join(OUT, "training_full.csv"), index=False)
print(f"\n保存: training_full.csv {len(result)}条 usable={result.usable.sum()}")
print(f"行业分布: {dict(result[result.usable].groupby('industry').size())}")
