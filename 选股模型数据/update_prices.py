"""更新全部价格缓存到20260630 (增量, 只拉已有缓存的新日期段)"""
import akshare as ak, os, time, glob, pickle, pandas as pd

PC = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data\price_cache"
END = "20260630"

files = glob.glob(os.path.join(PC, "*.pkl"))
total = len(files)
print(f"总缓存: {total}只")

new_count = 0
for i, fp in enumerate(files):
    with open(fp, "rb") as f: p = pickle.load(f)
    last_date = p["date"].max()
    last_str = last_date.strftime("%Y%m%d") if hasattr(last_date, "strftime") else str(last_date)
    
    if last_str >= END:
        continue
    
    code = os.path.basename(fp)[:-4]
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
    try:
        new = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=END, adjust="qfq")
        if len(new) > 0:
            new = new[["日期", "收盘"]].rename(columns={"收盘": "close"})
            new["date"] = pd.to_datetime(new["日期"])
            combined = pd.concat([p, new], ignore_index=True).drop_duplicates(subset=["date"])
            with open(fp, "wb") as f: pickle.dump(combined, f)
            new_count += 1
    except: pass
    time.sleep(0.2)
    
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{total}] 已更新{new_count}只")
    elif (i + 1) % 20 == 0 and total < 100:
        print(f"  [{i+1}/{total}] 已更新{new_count}只")

print(f"\n完成: 更新{new_count}/{total}只")
