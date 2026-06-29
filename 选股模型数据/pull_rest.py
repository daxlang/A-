"""拉全剩余20季度业绩报 + 24季度负债表"""
import akshare as ak, time, os, pandas as pd
OUT = r"C:\Users\daxlang\Desktop\杂7杂8的笔记\选股模型数据\history_data\ak_cache"
os.makedirs(OUT, exist_ok=True)
QUARTERS = [f"{y}{q}" for y in range(2019, 2025) for q in ["0331","0630","0930","1231"]]

# 已有: 20240630, 20240930, 20241231, 20250331
# 需要补: 20190331 ~ 20240331
done = set(os.listdir(OUT))
todo_yjbb = [d for d in QUARTERS if f"yjbb_{d}.parquet" not in done]
todo_zcfz = [d for d in QUARTERS if f"zcfz_{d}.parquet" not in done]
print(f"业绩报待拉: {len(todo_yjbb)}个 {todo_yjbb[:3]}...")
print(f"负债表待拉: {len(todo_zcfz)}个")

# 业绩报
for i, dt in enumerate(todo_yjbb):
    try:
        df = ak.stock_yjbb_em(date=dt); df["qdate"] = dt
        df.to_parquet(os.path.join(OUT, f"yjbb_{dt}.parquet"))
        print(f"yjbb [{i+1}/{len(todo_yjbb)}] {dt}: {len(df)}条")
        time.sleep(1)
    except Exception as e:
        print(f"yjbb {dt}: FAIL {e}")

# 负债表
for i, dt in enumerate(todo_zcfz):
    try:
        df = ak.stock_zcfz_em(date=dt); df["qdate"] = dt
        df.to_parquet(os.path.join(OUT, f"zcfz_{dt}.parquet"))
        print(f"zcfz [{i+1}/{len(todo_zcfz)}] {dt}: {len(df)}条")
        time.sleep(1)
    except Exception as e:
        print(f"zcfz {dt}: FAIL {e}")

print("\n全部拉取完成")
