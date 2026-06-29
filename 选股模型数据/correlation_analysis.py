#!/usr/bin/env python3
"""特征相关性分析：找出冗余指标"""

import pandas as pd
import numpy as np

df = pd.read_csv(
    r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\all_stocks_q1_2026_clean.csv",
    encoding="utf-8-sig"
)

# 选择用于相关性分析的特征列（排除ID列和覆盖率太低的列）
feature_cols = [
    "rev", "gross_margin", "roa", "roe",
    "asset_turnover", "debt_ratio",
    "pe", "pb", "ps", "dividend_yield", "market_cap",
    "net_margin", "quality_score", "valuation_score",
]

# 只保留覆盖率>50%的列
available = [c for c in feature_cols if df[c].notna().sum() > len(df) * 0.5]
print(f"参与相关性分析的列 ({len(available)} 列): {available}\n")

# 计算 Spearman 秩相关（对偏态金融数据更稳健）
corr = df[available].corr(method="spearman")

# 找出绝对值>0.7的高相关对
print("=== 高相关对 (|Spearman| > 0.7) ===\n")
high_pairs = []
for i in range(len(available)):
    for j in range(i + 1, len(available)):
        val = corr.iloc[i, j]
        if abs(val) > 0.7:
            high_pairs.append((available[i], available[j], val))
            print(f"  {available[i]:<20s} vs {available[j]:<20s}  r={val:+.3f}")

if not high_pairs:
    print("  无")

# 分行业看关键相关对
print("\n=== 分行业: PE vs PB 相关 ===\n")
for ind in df["industry"].unique():
    sub = df[df["industry"] == ind]
    if len(sub) >= 5:
        r = sub[["pe", "pb"]].corr(method="spearman").iloc[0, 1]
        print(f"  {ind}: Spearman r={r:+.3f} (n={len(sub)})")

print("\n=== 分行业: PB vs PS 相关 ===\n")
for ind in df["industry"].unique():
    sub = df[df["industry"] == ind]
    if len(sub) >= 5:
        r = sub[["pb", "ps"]].corr(method="spearman").iloc[0, 1]
        print(f"  {ind}: Spearman r={r:+.3f} (n={len(sub)})")

print("\n=== 分行业: gross_margin vs roa 相关 ===\n")
for ind in df["industry"].unique():
    sub = df[df["industry"] == ind]
    sub2 = sub[["gross_margin", "roa"]].dropna()
    if len(sub2) >= 5:
        r = sub2.corr(method="spearman").iloc[0, 1]
        print(f"  {ind}: Spearman r={r:+.3f} (n={len(sub2)})")

# roa vs roe
print("\n=== roa vs roe (仅银行有ROE) ===\n")
sub = df[df["industry"] == "银行"][["roa", "roe"]].dropna()
if len(sub) >= 5:
    r = sub.corr(method="spearman").iloc[0, 1]
    print(f"  银行: Spearman r={r:+.3f} (n={len(sub)})")

# 完整矩阵
print("\n\n========== 完整 Spearman 相关矩阵 ==========\n")
print(corr.to_string())
