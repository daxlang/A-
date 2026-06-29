#!/usr/bin/env python3
"""基线模型: 线性回归 + 随机森林 → 预测12月收益"""
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings, os
warnings.filterwarnings("ignore")

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
df = pd.read_csv(os.path.join(OUT, "training_dataset.csv"))
df = df[df["usable"]].copy()
print(f"加载: {len(df)} 条训练样本\n")

# === 特征工程 ===
feature_cols = ["roe", "gross_margin", "net_margin", "profit_yoy",
                "asset_turnover", "inventory_turnover"]
industry_dummies = pd.get_dummies(df["industry"], prefix="ind").astype(float)
X_raw = pd.concat([df[feature_cols], industry_dummies], axis=1)
X_raw = X_raw.fillna(X_raw.median())
y = df["forward_return"].values

# 时间切分
train_mask = (df["year"] < 2024) | ((df["year"] == 2024) & (df["quarter"] <= 2))
test_mask = ~train_mask
X_train, X_test = X_raw[train_mask], X_raw[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
print(f"训练集: {len(X_train)} 条 (2022Q1~2024Q2)")
print(f"测试集: {len(X_test)} 条 (2024Q3~2024Q4)")
print(f"特征: {X_raw.columns.tolist()}\n")

# 标准化
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# === 评估指标 ===
def evaluate(name, y_true, y_pred):
    r2 = 1 - ((y_true - y_pred)**2).sum() / ((y_true - y_true.mean())**2).sum()
    rank_ic, _ = spearmanr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"{name}: R2={r2:+.4f}  Rank_IC={rank_ic:+.4f}  MAE={mae:.4f}")
    return {"R2": r2, "Rank_IC": rank_ic, "MAE": mae}

# === 1. 线性回归 (Ridge) ===
print("=== 线性回归 (RidgeCV) ===")
lr = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
lr.fit(X_train_s, y_train)
pred_lr_train = lr.predict(X_train_s)
pred_lr_test = lr.predict(X_test_s)
evaluate("  训练集", y_train, pred_lr_train)
res_lr = evaluate("  测试集", y_test, pred_lr_test)

# 线性权重
coef_df = pd.DataFrame({"feature": X_raw.columns, "coef": lr.coef_}).sort_values("coef", key=abs, ascending=False)
print("  权重(top5):", coef_df.head(5).to_dict("records"))

# === 2. 随机森林 ===
print("\n=== 随机森林 (100树, max_depth=5) ===")
rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10,
                           random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
pred_rf_train = rf.predict(X_train_s)
pred_rf_test = rf.predict(X_test_s)
evaluate("  训练集", y_train, pred_rf_train)
res_rf = evaluate("  测试集", y_test, pred_rf_test)

# 特征重要性
imp_df = pd.DataFrame({"feature": X_raw.columns, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
print("  重要性(top5):", imp_df.head(5).to_dict("records"))

# === 3. 分层收益 ===
print("\n=== 分层收益 (测试集, RF预测) ===")
test_df = df[test_mask].copy()
test_df["pred"] = pred_rf_test
test_df["bucket"] = pd.qcut(test_df["pred"], 5, labels=["Q1最低", "Q2", "Q3", "Q4", "Q5最高"])
buckets = test_df.groupby("bucket", observed=False)["forward_return"].mean()
for b, v in buckets.items():
    bar = "+" * max(0, int(v * 100)) if v > 0 else "-" * max(0, int(-v * 100))
    print(f"  {b}: {v*100:+.1f}% {bar}")
print(f"  Q5-Q1 利差: {(buckets.iloc[-1]-buckets.iloc[0])*100:+.1f}%")

# === 汇总 ===
print("\n=== 汇总 ===")
print(f"  线性回归 Rank_IC: {res_lr['Rank_IC']:+.4f}")
print(f"  随机森林 Rank_IC: {res_rf['Rank_IC']:+.4f}")
if res_rf["Rank_IC"] > res_lr["Rank_IC"] + 0.02:
    print("  → 随机森林显著优于线性，存在非线性信号")
elif res_rf["Rank_IC"] < res_lr["Rank_IC"] - 0.02:
    print("  → 线性回归更优，信噪比不足以支撑非线性模型")
else:
    print("  → 二者接近，金融数据信噪比低是正常的")
