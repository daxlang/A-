import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data\training_dataset.csv')
df = df[df['usable']].copy()

feat = ['roe','gross_margin','net_margin','profit_yoy','asset_turnover','inventory_turnover']
ind = pd.get_dummies(df['industry'], prefix='ind').astype(float)
X = pd.concat([df[feat], ind], axis=1)
X = X.fillna(X.median())
y = df['forward_return'].values

train_mask = (df['year'] < 2024) | ((df['year'] == 2024) & (df['quarter'] <= 2))
test_mask = ~train_mask
sc = StandardScaler()
Xtr = sc.fit_transform(X[train_mask])
Xte = sc.transform(X[test_mask])
ytr, yte = y[train_mask], y[test_mask]

lr = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5)
lr.fit(Xtr, ytr)
pred = lr.predict(Xte)
true = yte

mae_model = np.mean(np.abs(true - pred))
mse_model = np.mean((true - pred)**2)
naive = np.full_like(true, ytr.mean())
mae_naive = np.mean(np.abs(true - naive))
mse_naive = np.mean((true - naive)**2)
mae_zero = np.mean(np.abs(true))
rank_ic, _ = spearmanr(true, pred)

print(f"=== 绝对误差对比 (160条测试集) ===")
print(f"{'':25s}  MAE       MSE")
print(f"{'模型(线性回归)':25s}  {mae_model:.4f}    {mse_model:.4f}")
print(f"{'永远猜均值':25s}  {mae_naive:.4f}    {mse_naive:.4f}")
print(f"{'永远猜0':25s}  {mae_zero:.4f}    -")
print(f"")
print(f"模型 vs 猜均值: MAE改善 {(1-mae_model/mae_naive)*100:.1f}%")
print(f"Rank_IC: {rank_ic:+.4f}")
print(f"")
print(f"原因: 模型预测的标准差={np.std(pred):.3f}, 真实收益标准差={np.std(true):.3f}")
print(f"→ 模型天然保守(回归向均值), 所以MSE难以大幅击败猜均值")
print(f"→ 但排序相关(Rank_IC={rank_ic:+.2f}), 高排名确实比低排名赚得多")
