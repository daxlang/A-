"""AkShare批量拉取全A财报数据: 2019Q1-2024Q4"""
import akshare as ak, pandas as pd, time, os

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data\ak_cache"
os.makedirs(OUT, exist_ok=True)

QUARTERS = []
for y in range(2019, 2025):
    for q in ["0331", "0630", "0930", "1231"]:
        QUARTERS.append(f"{y}{q}")

print(f"共{len(QUARTERS)}个季度: {QUARTERS[0]}~{QUARTERS[-1]}")

# 1. 拉业绩报表(利润表+CF核心)
all_yjbb = []
for i, dt in enumerate(QUARTERS):
    try:
        df = ak.stock_yjbb_em(date=dt)
        df["qdate"] = dt
        all_yjbb.append(df)
        print(f"  [{i+1}/24] 业绩报 {dt}: {len(df)}条")
        time.sleep(0.5)
    except Exception as e:
        print(f"  [{i+1}/24] 业绩报 {dt}: 失败 {e}")
yjbb = pd.concat(all_yjbb, ignore_index=True)
yjbb.to_parquet(os.path.join(OUT, "yjbb_all.parquet"), index=False)
print(f"业绩报合计: {len(yjbb)}条")

# 2. 拉资产负债表
all_zcfz = []
for i, dt in enumerate(QUARTERS):
    try:
        df = ak.stock_zcfz_em(date=dt)
        df["qdate"] = dt
        all_zcfz.append(df)
        print(f"  [{i+1}/24] 负债表 {dt}: {len(df)}条")
        time.sleep(0.5)
    except Exception as e:
        print(f"  [{i+1}/24] 负债表 {dt}: 失败 {e}")
zcfz = pd.concat(all_zcfz, ignore_index=True)
zcfz.to_parquet(os.path.join(OUT, "zcfz_all.parquet"), index=False)
print(f"负债表合计: {len(zcfz)}条")

# 3. 拉现金流量表(只有13个季度有数据? 试试看)
# ak stock_xjll_em 可能不可用, 但yjbb里已经有了每股经营CF
# 从yjbb可以近似: 经营CF ≈ 每股经营现金流量 × 总股本
# 暂时跳过, 用yjbb就够了

print(f"\n缓存完成 → {OUT}")
print(f"下一步: 按行业筛选+特征拼接")
