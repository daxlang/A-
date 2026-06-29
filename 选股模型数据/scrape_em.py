"""批量抓取东方财富财务数据 - 130只股票"""
import requests, re, json, pandas as pd, numpy as np, time, os

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
PROG = os.path.join(OUT, "scrape_progress.txt")

def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(f"[{t}] {msg}\n")

with open(PROG, "w", encoding="utf-8") as f: f.write("")

# 股票列表
dc = pd.read_csv(os.path.join(OUT, "all_stock_codes_full.csv"), dtype=str)
codes = dc["code"].tolist()
n = len(codes)
log(f"开始: {n}只")

# 指标名到英文名的映射
indicator_map = {
    "基本每股收益": "eps",
    "每股净资产": "bps",
    "每股经营现金流": "ocf_per_share",
    "营业总收入": "revenue",
    "毛利润": "gross_profit",
    "归属净利润": "net_profit",
    "扣非净利润": "deducted_profit",
    "营业总收入同比增长": "revenue_yoy",
    "归属净利润同比增长": "profit_yoy",
    "净资产收益率(加权)": "roe",
    "总资产收益率(加权)": "roa",
    "毛利率": "gross_margin",
    "净利率": "net_margin",
    "销售净现金流/营业总收入": "sales_cash_ratio",
    "经营净现金流/营业总收入": "cfo_to_revenue",
    "流动比率": "current_ratio",
    "速动比率": "quick_ratio",
    "资产负债率": "debt_ratio",
    "总资产周转率": "asset_turnover",
    "存货周转率": "inventory_turnover",
    "应收账款周转率": "receivable_turnover",
}

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

all_rows = []
success = 0; fail = 0
dates_cache = None

for idx, code in enumerate(codes):
    code_em = ("SH" if code.startswith("sh") else "SZ") + code[2:]
    url = f"https://emweb.securities.eastmoney.com/PC_HSF10/FinanceAnalysis/Index?type=web&code={code_em}"
    
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            fail += 1; continue
    except:
        fail += 1; continue
    
    html = r.text
    
    # 提取日期
    if dates_cache is None:
        dates = re.findall(r'>(\d{2}-\d{2}-\d{2})<', html)
        if dates:
            dates_cache = []
            seen = set()
            for d in dates:
                yr, mo, _ = d.split("-")
                qtr_map = {"03": 1, "06": 2, "09": 3, "12": 4}
                qtr = qtr_map.get(mo, 0)
                if isinstance(yr, str):
                    full_yr = 2000 + int(yr)
                    ds = f"{full_yr}Q{qtr}"
                else:
                    ds = f"20{yr}Q{qtr}"
                if ds not in seen:
                    seen.add(ds)
                    if len(ds) == 7:
                        dates_cache.append(ds)
            if dates_cache:
                dates_cache = dates_cache[:36]
                log(f"日期{len(dates_cache)}: {dates_cache[:12]}...")
    
    # 提取所有指标
    found = {}
    for cn_name, en_name in indicator_map.items():
        pattern = re.escape(cn_name) + r"[^\-]*?([\d\.\-]+[万亿%]?)"
        matches = re.findall(pattern, html)
        if matches:
            vals = []
            for m in matches:
                m = m.strip()
                if "亿" in m:
                    vals.append(float(m.replace("亿", "").replace("--", "").replace(",", "")) * 1e8 if m != "--" and m != "" else np.nan)
                elif "万" in m:
                    vals.append(float(m.replace("万", "")) * 1e4)
                elif "%" in m:
                    vals.append(float(m.replace("%", "")))
                elif m == "--" or m == "":
                    vals.append(np.nan)
                else:
                    try:
                        vals.append(float(m))
                    except:
                        vals.append(np.nan)
            found[en_name] = vals
    
    # 构建行
    if dates_cache:
        for i, date_str in enumerate(dates_cache):
            y, q = date_str.replace("20", "").split("Q")[0], date_str.split("Q")[1]
            y_full = 2000 + int(y)
            q_int = int(q)
            row = {"code": code, "year": y_full, "quarter": q_int}
            for en, vals in found.items():
                if i < len(vals) and not (isinstance(vals[i], float) and np.isnan(vals[i])):
                    row[en] = vals[i]
            all_rows.append(row)
    
    success += 1
    if (idx + 1) % 10 == 0:
        log(f"  进度 {idx+1}/{n} (成功{success} 失败{fail})")

# 保存
df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(OUT, "financials_em.csv"), index=False, encoding="utf-8-sig")
log(f"完成: {len(df)}条, {df.code.nunique()}只, 列: {df.columns.tolist()}")
for c in df.columns:
    if c not in ["code", "year", "quarter"]:
        log(f"  {c}: {df[c].notna().sum()}/{len(df)}")
log("DONE")
