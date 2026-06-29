"""XGBoost多参数试跑: 5折CV, 五分类, 只买暴涨"""
import os, pickle, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except:
    print("请先 pip install xgboost")
    exit(1)

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")

df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
def c5(r):
    if r < -0.1: return 0
    if r < 0: return 1
    if r < 0.1: return 2
    if r < 0.3: return 3
    return 4
u["label"] = u.forward_return.apply(c5)
feat = [c for c in u.columns if c not in
    ["code","year","quarter","forward_return","label","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.label.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]

params_list = [
    ("XGB_默认_均衡", dict(max_depth=5, n_estimators=100, learning_rate=0.05), 2.0),
    ("XGB_深慢_均衡", dict(max_depth=4, n_estimators=300, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8), 2.0),
    ("XGB_宽浅_均衡", dict(max_depth=3, n_estimators=400, learning_rate=0.02, subsample=0.7), 2.0),
    ("XGB_默认_偏暴涨", dict(max_depth=5, n_estimators=100, learning_rate=0.05), 3.0),
    ("XGB_深快_均衡", dict(max_depth=7, n_estimators=200, learning_rate=0.05, min_child_weight=3), 2.0),
    ("XGB_偏暴涨3x", dict(max_depth=4, n_estimators=200, learning_rate=0.03, subsample=0.7), 3.0),
]

print("=== XGBoost 多参数 5折CV, 策略: 只买暴涨 ===\n")
print(f"{'配置':<16s} {'买入':>5s} {'均值':>7s} {'中位':>7s} {'正收益':>7s} {'超额':>7s} {'年均':>7s}")
print("-" * 70)

for name, params, w4 in params_list:
    preds = np.zeros(len(u), dtype=int)
    for ty in TY:
        tr = (yrs != ty); te = (yrs == ty)
        m = xgb.XGBClassifier(**params, objective="multi:softmax", num_class=5,
                               random_state=42, eval_metric="mlogloss")
        sw = np.ones_like(y[tr], dtype=float)
        sw[y[tr] == 4] = w4  # 加重暴涨类样本权重
        m.fit(X[tr].values, y[tr], sample_weight=sw)
        preds[te] = m.predict(X[te].values)
        pickle.dump(m, open(os.path.join(MDIR, f"{name}_{ty}.pkl"), "wb"))

    buy = pd.DataFrame({"pred": preds, "ret": rets, "year": yrs})
    bought = buy[buy.pred == 4]
    n = len(bought); avg = bought.ret.mean(); med = bought.ret.median()
    win = (bought.ret > 0).mean()*100; exc = avg - rets.mean()
    yrly = [buy[(buy.year==ty)&(buy.pred==4)].ret.mean() for ty in TY
            if len(buy[(buy.year==ty)&(buy.pred==4)]) > 0]
    ayr_val = np.mean(yrly) if yrly else 0

    print(f"{name:<16s} {n:>5d} {avg*100:>+6.1f}% {med*100:>+6.1f}% {win:>6.1f}% {exc*100:>+6.1f}% {ayr_val*100:>+6.1f}%")

# 对比
print(f"\n(对比: MLP w=0.5  337次 +20.7% +8.7% 57.3% +9.1% +19.8%)")
