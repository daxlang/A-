"""清洗 training_all.csv + training_full.csv"""
import pandas as pd, numpy as np
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

for csv_name in ["training_all", "training_full"]:
    fpath = f"{BASE}/{csv_name}.csv"
    df = pd.read_csv(fpath, dtype={"code": str})
    n0 = len(df); u0 = df.usable.sum()
    
    # 修正季度编码
    df["quarter"] = df["quarter"].map({3:1, 6:2, 9:3, 12:4, 1:1, 2:2, 3:3, 4:4})
    
    # buy_price > 0
    if csv_name == "training_all":
        df.loc[df.buy_price <= 0, "usable"] = False
    
    # interest_coverage = 4.0
    df["interest_coverage"] = df["interest_coverage"].fillna(4.0)
    
    # forward_return 同年缩尾 1%/99%
    for yr in df.year.unique():
        mask = df.year == yr
        fr = df.loc[mask, "forward_return"]
        if fr.notna().sum() > 0:
            p1, p99 = np.percentile(fr.dropna(), [1, 99])
            df.loc[mask, "forward_return"] = fr.clip(p1, p99)
    
    # 其他特征缩尾
    for f in ["roe","gross_margin","net_margin","profit_yoy","asset_turnover",
              "inventory_turnover","liability_to_asset","current_ratio",
              "cfo_to_revenue","cfo_to_profit","pe","pb","ps"]:
        if f in df.columns:
            vals = df[f].dropna()
            if len(vals) > 0:
                p1, p99 = np.percentile(vals, [1, 99])
                df[f] = df[f].clip(p1, p99)
    
    n1 = len(df); u1 = df.usable.sum()
    print(f"{csv_name}: {n0}→{n1}条 可用{u0}→{u1}")
    
    # 最终 fillna
    feat = ["roe","gross_margin","net_margin","profit_yoy","asset_turnover","inventory_turnover",
            "liability_to_asset","current_ratio","cfo_to_revenue","cfo_to_profit","interest_coverage",
            "buy_price","pe","pb","ps"]
    for f in feat:
        if f in df.columns and df[f].isna().sum() > 0:
            df[f] = df[f].fillna(df[f].median())
    
    df.to_csv(fpath, index=False)
    print(f"  保存完成, NaN={df[feat].isna().sum().sum()}")

# 检查最终结果
print("\n=== 清洗后 training_all ===")
a = pd.read_csv(f"{BASE}/training_all.csv", dtype={"code": str})
u = a[a.usable]
print(f"总{len(a)}条 可用{len(u)}条 {u.code.nunique()}只")
print(f"fr: mean={u.forward_return.mean()*100:.1f}% P1={np.percentile(u.forward_return,1)*100:.1f}% P99={np.percentile(u.forward_return,99)*100:.1f}%")
for ind in sorted(u.industry.unique()):
    s = u[u.industry == ind]
    print(f"  {ind}: {s.code.nunique()}只 {len(s)}行 均{s.forward_return.mean()*100:.1f}%")
