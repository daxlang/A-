"""LR/RF 加权重: balanced + w=0.5, 5折训练+评估"""
import os, pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1)

configs = [
    ("LR_balanced", "lr", {"max_iter": 5000, "random_state": 42, "class_weight": "balanced"}),
    ("LR_w0.5", "lr", {"max_iter": 5000, "random_state": 42, "class_weight": {0:1,1:1,2:1,3:1,4:0.5}}),
    ("RF_balanced", "rf", {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 10, "random_state": 42, "class_weight": "balanced", "n_jobs": -1}),
    ("RF_w0.5", "rf", {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 10, "random_state": 42, "class_weight": {0:1,1:1,2:1,3:1,4:0.5}, "n_jobs": -1}),
]

for name, model_type, params in configs:
    print(f"\n训练 {name}...")
    pr = np.zeros(len(u), dtype=int)
    for ty in TY:
        tr = (yrs != ty); te = (yrs == ty)
        if model_type == "lr":
            clf = LogisticRegression(**params)
        else:
            clf = RandomForestClassifier(**params)
        clf.fit(X[tr].values, y[tr])
        pickle.dump(clf, open(os.path.join(MDIR, f"{name}_{ty}.pkl"), "wb"))
        pr[te] = clf.predict(X[te].values)
        print(f"  fold {ty} done")

    cm = confusion_matrix(y[MASK], pr[MASK]); n = cm.sum()
    acc = sum(cm[i,i] for i in range(5)) / n * 100
    print(f"=== {name}→S1∩S3 ({n}条 acc={acc:.1f}%) ===")
    print(f"实际↓预测 |暴跌  跌 小涨 中涨 暴涨| 合计")
    for i, lb in enumerate(LAB):
        print(f"  {lb:6s}|{cm[i,0]:>4d} {cm[i,1]:>3d} {cm[i,2]:>4d} {cm[i,3]:>4d} {cm[i,4]:>4d}|{cm[i].sum():>5d}")
    buy = (pr == 4) & MASK; nb = buy.sum(); br = rets[buy]
    yr = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
    print(f"买入: {nb}只 均{br.mean()*100:+.1f}% 年{np.mean(yr):+.1f}%")

print(f"\n基准 MLP0_5: 231只 年+21.8%")
