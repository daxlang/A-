"""Top5 暴涨推荐 - fold2024 MLP0_5"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
S1 = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5)

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

# fold 2024
te2024 = (u.year == 2024)
te_s1 = te2024 & S1
X_2024 = X[te_s1].values
ckpt = torch.load(os.path.join(MDIR, "MLP0_5_2024.pt"), weights_only=False)
m = M5(X_2024.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
Xs = torch.FloatTensor(ckpt["scaler"].transform(X_2024))
with torch.no_grad():
    logits = m(Xs)
    logit4 = logits[:, 4].numpy()

top5 = np.argsort(logit4)[-5:][::-1]
print("Top5 暴涨推荐 (MLP0_5 fold2024, S1过滤后307只):")
print(f"{'股票':<10s} {'行业':<4s} {'logit4':>8s} {'毛利率':>7s} {'PE':>7s} {'ROE':>7s} {'实际收益':>8s}")
print("-" * 60)
for pos in top5:
    sub = u[te_s1].iloc[pos]
    print(f"{sub.code:<10s} {sub.industry:<4s} {logit4[pos]:>+8.3f} {sub.gross_margin:>+6.1f}% {sub.pe:>+6.1f} {sub.roe:>+6.1f}% {sub.forward_return*100:>+7.1f}%")

actual = u[te_s1].iloc[top5].forward_return
print(f"\n5只平均实际收益: {actual.mean()*100:+.1f}%")
print(f"S1全307只均值: {u.loc[te_s1,'forward_return'].mean()*100:+.1f}%")
print(f"全520只均值: {u.loc[te2024,'forward_return'].mean()*100:+.1f}%")
