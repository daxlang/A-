"""季度收益: 训练5折MLP w=0.5, S1∩S3∩S4测试"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()

# 季度收益
u = u.sort_values(["code", "year", "quarter"])
u["next_price"] = u.groupby("code")["buy_price"].shift(-1)
u["q_return"] = (u.next_price / u.buy_price - 1)
u = u[u.q_return.notna()].copy()  # 去掉最后一行(无下季度)
print(f"季度收益: {len(u)}条 均值{u.q_return.mean()*100:.1f}%")

# 5分类(季度阈值: -5%, 0%, 5%, 15%)
u["L"] = u.q_return.apply(lambda r: 0 if r < -0.05 else (1 if r < 0 else (2 if r < 0.05 else (3 if r < 0.15 else 4))))
# 放弃 forward_return列, 用q_return代替
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","L","usable","industry","q_return","next_price"]]
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","L","usable","industry","q_return","next_price","gm_pct"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.q_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]

print(f"特征数: {len(feat)}+{ind.shape[1]}={X.shape[1]}")
print(f"标签分布: {[(LAB[i],(y==i).sum()) for i in range(5)]}")

MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8)
print(f"测试集(S1∩S3∩S4): {MASK.sum()}条")

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
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

# 训练5折
print("\n训练 Q_MLP0_5...")
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    train_save(X[tr].values, y[tr], os.path.join(MDIR, f"Q_MLP0_5_{ty}.pt"))
    print(f"  fold {ty}: 训{tr.sum()} 测{te.sum()}")

# 预测
pr = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); pr[te] = load_pred(X[te].values, os.path.join(MDIR, f"Q_MLP0_5_{ty}.pt"))

cm = confusion_matrix(y[MASK], pr[MASK]); n = cm.sum()
acc = sum(cm[i,i] for i in range(5)) / n * 100
print(f"\n=== Q_MLP0_5→S1∩S3∩S4 ({n}条 acc={acc:.1f}%) ===")
print(f"实际↓预测 |暴跌  跌 小涨 中涨 暴涨| 合计")
for i, lb in enumerate(LAB):
    print(f"  {lb:6s}|{cm[i,0]:>4d} {cm[i,1]:>3d} {cm[i,2]:>4d} {cm[i,3]:>4d} {cm[i,4]:>4d}|{cm[i].sum():>5d}")

buy = (pr == 4) & MASK; nb = buy.sum(); br = rets[buy]
yrly = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
print(f"买入暴涨: {nb}只 均{br.mean()*100:.1f}% 年化{br.mean()*100:.1f}%")
for ty in TY:
    b = (yrs == ty) & buy; n_b = b.sum()
    a = rets[b]
    print(f"  {ty}: 买{n_b}只 均{a.mean()*100:.1f}%" if n_b > 0 else f"  {ty}: 未买")

print(f"\n基准(年频): MLP0_5+S1∩S3∩S4 231买入 年+21.8%")
print(f"注意: 季度收益需年化比较: 季均{br.mean()*100:.1f}% × 4 ≈ {(1+br.mean())**4-1:.1%}")
