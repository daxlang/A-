"""S3特征: gm_rank + gm_cfo_cross 加入训练, 对比原版"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")

# 原版数据
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()

# 新特征
u["gm_rank"] = u.groupby(["industry", "year"])["gross_margin"].rank(pct=True)
u["gm_yoy"] = u.groupby("code")["gross_margin"].shift(4)  # 去年同期
u["gm_dir"] = np.where(u.gross_margin >= u.gm_yoy, 1, -1)  # 毛利率方向
u["cfo_dir"] = np.where(u.cfo_to_revenue >= u.groupby("code")["cfo_to_revenue"].shift(4), 1, -1)
u["gm_cfo_cross"] = u["gm_dir"] * u["cfo_dir"]  # +1=同向, -1=背离
u.drop(columns=["gm_yoy", "gm_dir", "cfo_dir"], inplace=True)

# 保存新数据集
u.to_csv(os.path.join(OUT, "training_s3_feat.csv"), index=False)
print(f"新特征: gm_rank, gm_cfo_cross")
print(f"保存: training_s3_feat.csv {len(u)}条")

# 训练+评估
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
S1 = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5)

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def train_save(Xtr, ytr, path):
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(Xtr)); ys = torch.LongTensor(ytr)
    m = M5(Xs.shape[1]); opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]))
    bl, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval(); torch.save({"model":bs,"scaler":sc}, path)

def load_pred(Xte, path):
    ckpt = torch.load(path, weights_only=False); m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

print("训练 S3_feat...")
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    train_save(X[tr].values, y[tr], os.path.join(MDIR, f"S3_feat_{ty}.pt"))
    print(f"  fold {ty}: 训{tr.sum()} 测{te.sum()}")

pr_new = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); pr_new[te] = load_pred(X[te].values, os.path.join(MDIR, f"S3_feat_{ty}.pt"))

# 原版对比
u_orig = df[df.usable].copy()
u_orig["L"] = u_orig.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
feat_orig = [c for c in u_orig.columns if c not in ["code","year","quarter","forward_return","L","usable","industry"]]
ind_orig = pd.get_dummies(u_orig.industry, prefix="ind").astype(float)
X_orig = pd.concat([u_orig[feat_orig], ind_orig], axis=1); X_orig = X_orig.fillna(X_orig.median())
y_orig = u_orig.L.values; rets_orig = u_orig.forward_return.values; yrs_orig = u_orig.year.values
S1_orig = (u_orig.cfo_to_revenue >= 0) & (u_orig.current_ratio >= 0.5)

pr_old = np.zeros(len(u_orig), dtype=int)
for ty in TY:
    te = (yrs_orig == ty); pr_old[te] = load_pred(X_orig[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))

for name, p, X_t, y_t, ret_t, yrs_t, m in [
    ("原版MLP0_5→S1", pr_old, X_orig.values, y_orig, rets_orig, yrs_orig, S1_orig),
    ("S3_feat→S1", pr_new, X.values, y, rets, yrs, S1),
]:
    cm = confusion_matrix(y_t[m], p[m]); n = cm.sum()
    acc = sum(cm[i,i] for i in range(5)) / n * 100
    print(f"\n=== {name} ({n}条 acc={acc:.1f}%) ===")
    print(f"实际↓预测 |暴跌  跌 小涨 中涨 暴涨| 合计")
    for i, lb in enumerate(LAB):
        print(f"  {lb:6s}|{cm[i,0]:>4d} {cm[i,1]:>3d} {cm[i,2]:>4d} {cm[i,3]:>4d} {cm[i,4]:>4d}|{cm[i].sum():>5d}")
    buy = (p == 4) & m; nb = buy.sum(); br = ret_t[buy]
    yr = [ret_t[(yrs_t==ty)&buy].mean()*100 for ty in TY if sum((yrs_t==ty)&buy)>0]
    print(f"买入: {nb}只 均{br.mean()*100:+.1f}% 年{np.mean(yr):+.1f}%")
