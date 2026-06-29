"""拉取缺失季度: 2024Q4 ~ 2026Q1 的yjbb+zcfz"""
import akshare as ak, os, time, pandas as pd

CACHE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data\ak_cache"

# 要拉取的季度
NEW_QS = [
    "20241231",  # 2024Q4
    "20250331",  # 2025Q1
    "20250630",  # 2025Q2
    "20250930",  # 2025Q3
    "20251231",  # 2025Q4
    "20260331",  # 2026Q1
]

for dt in NEW_QS:
    fyj = os.path.join(CACHE, f"yjbb_{dt}.parquet")
    fzc = os.path.join(CACHE, f"zcfz_{dt}.parquet")
    
    if os.path.exists(fyj) and os.path.exists(fzc):
        print(f"{dt}: 已存在, 跳过")
        continue
    
    print(f"拉取 {dt}...")
    try:
        yj = ak.stock_yjbb_em(date=dt)
        yj.to_parquet(fyj)
        print(f"  yjbb: {len(yj)}条")
        time.sleep(2)
    except Exception as e:
        print(f"  yjbb 失败: {e}")
    
    try:
        zc = ak.stock_zcfz_em(date=dt)
        zc.to_parquet(fzc)
        print(f"  zcfz: {len(zc)}条")
        time.sleep(2)
    except Exception as e:
        print(f"  zcfz 失败: {e}")

print("\n拉取完成")
