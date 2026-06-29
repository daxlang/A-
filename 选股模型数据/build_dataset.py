#!/usr/bin/env python3
"""构建训练集: 财报指标(X) + 12月未来收益(Y)"""
import pandas as pd, numpy as np, os

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

# 加载数据
df_f = pd.read_csv(os.path.join(OUT, "financials_history.csv"))
df_p = pd.read_csv(os.path.join(OUT, "prices_daily.csv"))
df_s = pd.read_csv(r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\all_stocks_q1_2026_clean.csv")

df_p["date"] = pd.to_datetime(df_p["date"])
df_p["close"] = pd.to_numeric(df_p["close"], errors="coerce")

# 为每只股票建一个价格索引，加速查找
price_map = {}
for code, grp in df_p.groupby("code"):
    grp = grp.dropna(subset=["close"]).sort_values("date")
    price_map[code] = grp.set_index("date")["close"]

# 确定买入/卖出日
def get_buy_sell_dates(year, quarter):
    """返回 (买入日字符串, 卖出日字符串)"""
    if quarter == 1:
        return (f"{year}-04-30", f"{year+1}-04-30")
    elif quarter == 2:
        return (f"{year}-07-31", f"{year+1}-07-31")
    elif quarter == 3:
        return (f"{year}-10-31", f"{year+1}-10-31")
    else:  # Q4
        return (f"{year+1}-04-30", f"{year+2}-04-30")

def nearest_price(code, target_date):
    """找到 target_date 或之后最近的交易日收盘价"""
    if code not in price_map:
        return np.nan
    ts = pd.Timestamp(target_date)
    s = price_map[code]
    # 找 >= target_date 的第一个交易日
    mask = s.index >= ts
    if mask.any():
        return float(s[mask].iloc[0])
    # 如果未来没有数据，取最近的
    return float(s.iloc[-1])

# 构建训练集
rows = []
industry_map = dict(zip(df_s["code"], df_s["industry"]))

for _, r in df_f.iterrows():
    code = r["code"]
    year = int(r["year"])
    quarter = int(r["quarter"])
    
    buy_str, sell_str = get_buy_sell_dates(year, quarter)
    
    p_buy = nearest_price(code, buy_str)
    p_sell = nearest_price(code, sell_str)
    
    if pd.notna(p_buy) and pd.notna(p_sell) and p_buy > 0:
        forward_return = (p_sell - p_buy) / p_buy
    else:
        forward_return = np.nan
    
    row = {
        "code": code,
        "year": year,
        "quarter": quarter,
        "buy_date": buy_str,
        "sell_date": sell_str,
        "buy_price": round(p_buy, 4) if pd.notna(p_buy) else np.nan,
        "sell_price": round(p_sell, 4) if pd.notna(p_sell) else np.nan,
        "forward_return": forward_return,
        "industry": industry_map.get(code, "未知"),
        # 特征
        "roe": r.get("roe"),
        "gross_margin": r.get("gross_margin"),
        "net_margin": r.get("net_margin"),
        "profit_yoy": r.get("profit_yoy"),
        "asset_turnover": r.get("asset_turnover"),
        "inventory_turnover": r.get("inventory_turnover"),
    }
    rows.append(row)

df_train = pd.DataFrame(rows)

# 标记可用
df_train["usable"] = df_train["forward_return"].notna() & (df_train["year"] <= 2024)

# 统计
n_total = len(df_train)
n_usable = df_train["usable"].sum()
n_future = (df_train["year"] == 2025).sum()
n_nan = df_train["forward_return"].isna().sum() - n_future

print(f"总样本: {n_total}")
print(f"可用训练: {n_usable} (2022Q1~2024Q4, 有完整12月收益)")
print(f"未来(2025年): {n_future} (收益未发生)")
print(f"缺价格: {n_nan}")
print(f"平均12月收益: {df_train.loc[df_train['usable'], 'forward_return'].mean():.4f} ({df_train.loc[df_train['usable'], 'forward_return'].mean()*100:.1f}%)")
print(f"收益标准差: {df_train.loc[df_train['usable'], 'forward_return'].std():.4f}")

# 保存
outpath = os.path.join(OUT, "training_dataset.csv")
df_train.to_csv(outpath, index=False, encoding="utf-8-sig")
print(f"\n保存: {outpath}")

# 展示样本
print("\n=== 前5条 ===")
print(df_train.head().to_string())
