"""全行业模型: 6大类 14维(无IC) + 6行业one-hot w=0.5"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
df = pd.read_csv(os.path.join(OUT, "training_all.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
# 14特征(无interest_coverage) + 6行业one-hot
DROP = ["interest_coverage"]
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"] + DROP]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8)
print(f"特征: {X.shape[1]}维(14+{ind.shape[1]}行业)  全量{len(u)}条  测试{MASK.sum()}条")
print(f"标签: {dict(zip(LAB, [(y==i).sum() for i in range(5)]))}")

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def train_save(Xtr, ytr, path, n_epochs=5000):
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(Xtr)); ys = torch.LongTensor(ytr)
    m = M5(Xs.shape[1]); opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]))
    bl, bs = 1e9, None
    for _ in range(n_epochs):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval(); torch.save({"model":bs,"scaler":sc}, path)

def load_pred(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

print("\n训练 ALL_14 (87k条, 约15分钟)...")
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    n_ep = 3000 if ty in [2022,2023] else 5000  # 大fold少迭代
    train_save(X[tr].values, y[tr], os.path.join(MDIR, f"ALL_14_{ty}.pt"), n_ep)
    print(f"  fold {ty}: 训{tr.sum()}条 测{te.sum()}条")

pr = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty); pr[te] = load_pred(X[te].values, os.path.join(MDIR, f"ALL_14_{ty}.pt"))

cm = confusion_matrix(y[MASK], pr[MASK]); n = cm.sum()
acc = sum(cm[i,i] for i in range(5)) / n * 100
print(f"\n{'='*60}")
print(f"=== ALL_14 → S1∩S3∩S4 ({n}条 acc={acc:.1f}%) ===")
for i, lb in enumerate(LAB):
    print(f"  {lb:6s}|{cm[i,0]:>5d} {cm[i,1]:>4d} {cm[i,2]:>5d} {cm[i,3]:>5d} {cm[i,4]:>5d}|{cm[i].sum():>5d}")
buy = (pr == 4) & MASK; nb = buy.sum(); br = rets[buy]
yr = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
print(f"\n买入: {nb}只 均{br.mean()*100:.1f}% 年{np.mean(yr):.1f}% 暴涨召回{cm[4,4]}/{cm[4].sum()}")
for ty in TY:
    b = (yrs == ty) & buy; n = b.sum(); a = rets[b]
    print(f"  {ty}: {n}只 {a.mean()*100:.1f}%" if n > 0 else f"  {ty}: 未买")

# 行业分布
print(f"\n买入行业分布:")
for ind_name in sorted(u.loc[buy,"industry"].unique()):
    n = sum(buy & (u.industry == ind_name)); r = rets[buy & (u.industry == ind_name)]
    print(f"  {ind_name}: {n}只 均{r.mean()*100:.1f}%")

# 对比基准
print(f"\n基准对比:")
print(f"  FULL_IND(8行业,含IC): 325只 年+25.2%  召回4.6%")
print(f"  D14_8(8行业,去IC):   388只 年+25.6%  召回6.3%")
print(f"  ALL_14(全行业,去IC):  {nb}只 年{np.mean(yr):.1f}%  召回{cm[4,4]/cm[4].sum()*100:.1f}%")
