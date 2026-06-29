#!/usr/bin/env python3
"""PE异常值过滤 + 负债率>100%过滤"""

import pandas as pd
import numpy as np

df = pd.read_csv(
    r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\all_stocks_q1_2026.csv",
    encoding="utf-8-sig"
)
original_count = len(df)
print(f"原始样本数: {original_count}")

removed = []

# 规则1: PE超出合理范围
# 负PE: 公司亏损，PE无意义
# PE>200: 微利导致PE爆炸，同样无意义
pe_mask = (df["pe"] > 0) & (df["pe"] <= 200)
pe_outliers = df[~pe_mask][["code", "name", "industry", "pe"]]
for _, row in pe_outliers.iterrows():
    removed.append({
        "code": row["code"],
        "name": row["name"],
        "industry": row["industry"],
        "reason": f"PE={row['pe']:.1f} (超出 0~200 范围)",
        "detail": "微利/亏损导致PE失效" if row["pe"] <= 0 else "微利导致PE爆炸"
    })
    print(f"  PE异常: {row['code']} {row['name']} PE={row['pe']:.1f}")

# 规则2: 负债率 > 100% (资不抵债)
debt_mask = (df["debt_ratio"].isna()) | (df["debt_ratio"] <= 100)
debt_outliers = df[~debt_mask][["code", "name", "industry", "debt_ratio"]]
for _, row in debt_outliers.iterrows():
    removed.append({
        "code": row["code"],
        "name": row["name"],
        "industry": row["industry"],
        "reason": f"负债率={row['debt_ratio']:.1f}% (资不抵债)",
        "detail": "已丧失正常经营基础"
    })
    print(f"  负债异常: {row['code']} {row['name']} 负债率={row['debt_ratio']:.1f}%")

# 应用过滤
combined_mask = pe_mask & debt_mask
df_clean = df[combined_mask].copy()

removed_count = original_count - len(df_clean)
print(f"\n移除: {removed_count} 家")
print(f"保留: {len(df_clean)} 家")

# 保存
outpath = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\all_stocks_q1_2026_clean.csv"
df_clean.to_csv(outpath, index=False, encoding="utf-8-sig")
print(f"\n清洗后数据: {outpath}")

# 输出移除清单
if removed:
    print("\n=== 移除清单 ===")
    for r in removed:
        print(f"  {r['code']} {r['name']} ({r['industry']}) — {r['reason']}")
