"""全行业数据集: 6大类 + 全A股票价格缓存"""
import pandas as pd, numpy as np, os, time, glob, pickle
import akshare as ak

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
CACHE = os.path.join(OUT, "ak_cache")
PC = os.path.join(OUT, "price_cache")
os.makedirs(PC, exist_ok=True)

# ===== 6大类映射 =====
IND_MAP = {
    "周期资源": ["普钢","特钢Ⅱ","冶钢原料","煤炭开采","油服工程","炼化及贸易","农化制品","化学原料","化学制品","化学纤维","塑料","橡胶","非金属材料Ⅱ","水泥","玻璃玻纤","装修建材","工业金属","金属新材料","小金属","能源金属","贵金属"],
    "制造": ["通用设备","专用设备","自动化设备","轨交设备Ⅱ","工程机械","其他通用设备","电机Ⅱ","纺织服饰","纺织制造","家居用品","造纸","包装印刷","汽车零部件","汽车服务","摩托车及其他","商用车","乘用车","家电零部件Ⅱ","黑色家电","白色家电","小家电","照明设备Ⅱ","厨卫电器"],
    "科技": ["半导体","电子化学品Ⅱ","元件","光学光电子","消费电子","其他电子Ⅱ","软件开发","IT服务Ⅱ","计算机设备","通信服务","通信设备","军工电子Ⅱ","航空装备Ⅱ","航天装备Ⅱ","地面兵装Ⅱ","航海装备Ⅱ","电池","光伏设备","风电设备","其他电源设备Ⅱ"],
    "消费": ["白酒Ⅱ","非白酒","饮料乳品","调味发酵品Ⅱ","休闲食品","食品加工","医药商业","中药Ⅱ","化学制药","生物制品","医疗器械","医疗服务","教育","影视院线","电视广播Ⅱ","出版","数字媒体","广告营销","游戏Ⅱ","旅游及景区","酒店餐饮","旅游零售Ⅱ","专业服务","化妆品","个护用品","服装家纺","一般零售","文娱用品","饰品","养殖业","种植业","农产品加工"],
    "金融地产": ["银行Ⅱ","证券Ⅱ","保险Ⅱ","多元金融","房地产开发","房地产服务"],
    "公用事业": ["电力","电网设备","燃气Ⅱ","水务","环保设备Ⅱ","环境治理","物流","铁路公路","航运港口","航空机场","基础建设","专业工程","房屋建设Ⅱ","建筑装饰","工程咨询服务Ⅱ","装修装饰Ⅱ","综合Ⅱ"],
}

# ===== Step 1: 财报特征(从已有缓存) =====
print("Step1: 构建全行业财报特征(从缓存)...")
QS = [f"{y}{q}" for y in range(2019, 2025) for q in ["0331", "0630", "0930", "1231"]]

rows = []
for dt in QS:
    yj = pd.read_parquet(os.path.join(CACHE, f"yjbb_{dt}.parquet"))
    zc = pd.read_parquet(os.path.join(CACHE, f"zcfz_{dt}.parquet"))
    yj["code"] = yj["股票代码"].astype(str).str.zfill(6)
    zc["code"] = zc["股票代码"].astype(str).str.zfill(6)
    m = yj.merge(zc, on="code", suffixes=("_yi", "_zc"))
    m["qdate"] = dt; rows.append(m)
full = pd.concat(rows, ignore_index=True)

# 行业映射
full["industry"] = "其他"
for macro, subs in IND_MAP.items():
    full.loc[full["所处行业"].isin(subs), "industry"] = macro
data = full[full.industry != "其他"].copy().reset_index(drop=True)
data["year"] = data["qdate"].str[:4].astype(int); data["quarter"] = data["qdate"].str[4:6].astype(int)

# 特征计算
data["roe"] = data["净资产收益率"] / 100
data["gross_margin"] = data["销售毛利率"] / 100
data["net_margin"] = data["净利润-净利润"] / data["营业总收入-营业总收入"].replace(0, np.nan)
data["profit_yoy"] = data["净利润-同比增长"] / 100
data["asset_turnover"] = data["营业总收入-营业总收入"] * 4 / data["资产-总资产"].replace(0, np.nan)
data["inventory_turnover"] = data["营业总收入-营业总收入"] * 4 / data["资产-存货"].replace(0, np.nan)
data["liability_to_asset"] = data["资产负债率"] / 100
ca = data["资产-货币资金"] + data["资产-应收账款"] + data["资产-存货"]
cl = data["负债-应付账款"] + data["负债-预收账款"]
data["current_ratio"] = (ca / cl.replace(0, np.nan)).values
sh = (data["净利润-净利润"] / data["每股收益"].replace(0, np.nan)).values
data["cfo_to_revenue"] = (data["每股经营现金流量"] * sh) / data["营业总收入-营业总收入"].replace(0, np.nan)
data["cfo_to_profit"] = (data["每股经营现金流量"] * sh) / data["净利润-净利润"].replace(0, np.nan)
data["interest_coverage"] = np.nan
data["buy_price"] = np.nan; data["forward_return"] = np.nan
data["pe"] = np.nan; data["pb"] = np.nan; data["ps"] = np.nan
print(f"  财报: {len(data)}条 {data.code.nunique()}只")
for ind in sorted(data.industry.unique()):
    print(f"    {ind}: {data[data.industry==ind].code.nunique()}只")

# ===== Step 2: 拉价格(增量) =====
print("\nStep2: 拉价格(增量, 约60分钟)...")
all_codes = sorted(data.code.unique())
done = set(f[:-4] for f in os.listdir(PC) if f.endswith(".pkl"))
todo = [c for c in all_codes if c not in done]
print(f"  需拉: {len(todo)}只 (已缓存{len(done)})")
for i, code in enumerate(todo):
    fp = os.path.join(PC, f"{code}.pkl")
    try:
        df_p = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20190101", end_date="20250601", adjust="qfq")
        df_p = df_p[["日期", "收盘"]].rename(columns={"收盘": "close"})
        df_p["date"] = pd.to_datetime(df_p["日期"])
        with open(fp, "wb") as f:
            pickle.dump(df_p, f)
        time.sleep(0.3)
    except Exception as e:
        time.sleep(0.5)
    if (i + 1) % 50 == 0:
        print(f"    [{i+1}/{len(todo)}]")
    elif (i + 1) % 10 == 0 and len(todo) < 50:
        print(f"    [{i+1}/{len(todo)}]")

final_done = set(f[:-4] for f in os.listdir(PC) if f.endswith(".pkl"))
print(f"  价格缓存: {len(final_done)}只")

# ===== Step 3: 计算估值和forward_return =====
print("\nStep3: 计算估值...")
from pandas.tseries.offsets import MonthEnd, DateOffset

prices = {}
for f in glob.glob(PC + "/*.pkl"):
    code = os.path.basename(f)[:-4]
    with open(f, "rb") as fh:
        p = pickle.load(fh)
    prices[code] = p

for i, (_, r) in enumerate(data.iterrows()):
    code = r["code"]
    qs = r["qdate"]
    qe = pd.Timestamp(f"{qs[:4]}-{qs[4:6]}-01") + MonthEnd(0)
    fe = qe + DateOffset(years=1)
    p = prices.get(code)
    bp, fr, pe, pv, psv = np.nan, np.nan, np.nan, np.nan, np.nan
    if p is not None:
        b = p[p["date"] <= qe]
        bp = b["close"].iloc[-1] if len(b) > 0 else np.nan
        a = p[(p["date"] <= fe) & (p["date"] > qe + DateOffset(days=340))]
        if len(a) > 0 and pd.notna(bp) and bp > 0:
            fr = a["close"].iloc[-1] / bp - 1
        eps = r["每股收益"]; bpsv = r["每股净资产"]
        if pd.notna(eps) and eps > 0 and pd.notna(bp) and bp > 0:
            pe = bp / eps
        if pd.notna(bpsv) and bpsv > 0 and pd.notna(bp) and bp > 0:
            pv = bp / bpsv
        rev = r["营业总收入-营业总收入"]
        profit = r["净利润-净利润"]
        shares_est = profit / eps if pd.notna(eps) and eps != 0 else np.nan
        if pd.notna(shares_est) and shares_est > 0:
            sps = rev / shares_est
            if sps > 0 and pd.notna(bp):
                psv = bp / sps
    data.at[i, "buy_price"] = bp; data.at[i, "forward_return"] = fr
    data.at[i, "pe"] = pe; data.at[i, "pb"] = pv; data.at[i, "ps"] = psv

# ===== Step 4: 填充缺失指标 =====
print("\nStep4: 填充current_ratio+interest_coverage...")
# current_ratio: 行业年季中位 → 全局中位
for f in ["current_ratio"]:
    data[f] = data[f].fillna(data.groupby(["industry", "year", "quarter"])[f].transform("median"))
    data[f] = data[f].fillna(data[f].median())

# interest_coverage: 用已有的任何值填
if data["interest_coverage"].isna().sum() > 0:
    data["interest_coverage"] = data["interest_coverage"].fillna(10.0)  # 保守默认

# ===== Step 5: 保存 =====
print("Step5: 保存...")
KEEP = ["code", "year", "quarter", "forward_return", "industry"]
FEAT15 = ["roe", "gross_margin", "net_margin", "profit_yoy", "asset_turnover", "inventory_turnover",
          "liability_to_asset", "current_ratio", "cfo_to_revenue", "cfo_to_profit", "interest_coverage",
          "buy_price", "pe", "pb", "ps"]
res = data[KEEP + FEAT15].copy()
res["usable"] = res["forward_return"].notna()
for f in FEAT15:
    missing = res[f].isna().sum()
    if missing > 0:
        res[f] = res[f].fillna(res[f].median())

res.to_csv(os.path.join(OUT, "training_all.csv"), index=False)
print(f"\n保存: training_all.csv {len(res)}条 usable={res.usable.sum()}")
for ind in sorted(res.industry.unique()):
    ns = res[(res.industry == ind) & res.usable]
    nall = res[res.industry == ind]
    print(f"  {ind}: {ns.code.nunique()}/{nall.code.nunique()}只有价格 {len(ns)}行")

# 缺失检查
for f in FEAT15:
    n = res[f].isna().sum()
    if n > 0:
        print(f"  警告: {f} 缺失{n}/{len(res)}")
