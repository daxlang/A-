"""MLP vs 线性: 用MSE做损失，看绝对误差能不能打败基线"""
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

# === 数据 ===
df = pd.read_csv(r'C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data\training_dataset.csv')
df = df[df['usable']].copy()
feat_cols = ['roe','gross_margin','net_margin','profit_yoy','asset_turnover','inventory_turnover']
ind = pd.get_dummies(df['industry'], prefix='ind').astype(float)
X = pd.concat([df[feat_cols], ind], axis=1)
X = X.fillna(X.median())
y = df['forward_return'].values.reshape(-1, 1)
train = (df['year']<2024)|((df['year']==2024)&(df['quarter']<=2))
test = ~train

sc_X = StandardScaler(); sc_y = StandardScaler()
Xtr = torch.FloatTensor(sc_X.fit_transform(X[train])); ytr = torch.FloatTensor(sc_y.fit_transform(y[train]))
Xte = torch.FloatTensor(sc_X.transform(X[test])); yte_raw = y[test].ravel()
n_feat = Xtr.shape[1]  # 10

# === 微型 MLP ===
class TinyMLP(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(hidden=16, lr=0.005, epochs=2000):
    torch.manual_seed(42)
    model = TinyMLP(hidden)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    best_loss = 1e9; best_state = None
    for e in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item(); best_state = {k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = sc_y.inverse_transform(model(Xte).numpy()).ravel()
    return pred

# === 评估 ===
def evaluate(name, y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rank_ic, _ = spearmanr(y_true, y_pred)
    return mae, mse, rank_ic

# 基线
naive = np.full_like(yte_raw, y[train].mean())
mae_naive, mse_naive, _ = evaluate("猜均值", yte_raw, naive)
print(f"基线(猜均值): MAE={mae_naive:.4f} MSE={mse_naive:.4f}\n")

# 线性回归
lr = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5)
lr.fit(Xtr.numpy(), y[train].ravel())
pred_lr = lr.predict(Xte.numpy())
mae_lr, mse_lr, ric_lr = evaluate("线性", yte_raw, pred_lr)
print(f"线性回归: MAE={mae_lr:.4f} MSE={mse_lr:.4f} Rank_IC={ric_lr:+.4f}")
print(f"  MAE vs 基线: {(1-mae_lr/mae_naive)*100:+.1f}%\n")

# 随机森林
rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
rf.fit(Xtr.numpy(), y[train].ravel())
pred_rf = rf.predict(Xte.numpy())
mae_rf, mse_rf, ric_rf = evaluate("RF", yte_raw, pred_rf)
print(f"随机森林: MAE={mae_rf:.4f} MSE={mse_rf:.4f} Rank_IC={ric_rf:+.4f}")
print(f"  MAE vs 基线: {(1-mae_rf/mae_naive)*100:+.1f}%\n")

# MLP - 试多个大小
print("=== 微型MLP (MSE损失, 800条训练) ===")
configs = [(8, 0.003), (16, 0.005), (32, 0.005), (64, 0.003)]
for hidden, lr in configs:
    n_params = hidden * n_feat + hidden + hidden * (hidden//2) + (hidden//2) + (hidden//2) + 1
    pred_mlp = train_mlp(hidden=hidden, lr=lr, epochs=2000)
    mae_m, mse_m, ric_m = evaluate("MLP", yte_raw, pred_mlp)
    improvement = (1 - mae_m / mae_naive) * 100
    best = " < 最优" if mae_m < mae_lr and mae_m < mae_rf else ""
    print(f"  MLP(h={hidden:2d}, {n_params}参): MAE={mae_m:.4f} MSE={mse_m:.4f} Rank_IC={ric_m:+.4f} 改善={improvement:+.1f}%{best}")
