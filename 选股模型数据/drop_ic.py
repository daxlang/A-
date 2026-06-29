"""消融: 去掉interest_coverage (4行业+8行业, 共10个权重)"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")

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

def run_exp(name, csv_file, wtag, mask_func, label):
    df = pd.read_csv(os.path.join(OUT, csv_file), dtype={"code": str})
    u = df[df.usable].copy()
    u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
    u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
    DROP = ["interest_coverage"]
    feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"] + DROP]
    ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
    X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
    y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
    TY = [2020, 2021, 2022, 2023, 2024]
    LAB = ["暴跌","跌","小涨","中涨","暴涨"]
    MASK = mask_func(u)
    print(f"\n{'='*60}")
    print(f"{name}: {X.shape[1]}维 训{len(u)}条 测{MASK.sum()}条")
    
    # 训练
    for ty in TY:
        tr = (yrs != ty); te = (yrs == ty)
        train_save(X[tr].values, y[tr], os.path.join(MDIR, f"{wtag}_{ty}.pt"))
        print(f"  fold {ty}: 训{tr.sum()} 测{te.sum()}")
    
    # 预测
    pr = np.zeros(len(u), dtype=int)
    for ty in TY:
        te = (yrs == ty); pr[te] = load_pred(X[te].values, os.path.join(MDIR, f"{wtag}_{ty}.pt"))
    
    cm = confusion_matrix(y[MASK], pr[MASK]); n = cm.sum(); acc = sum(cm[i,i] for i in range(5)) / n * 100
    print(f"=== {label} ({n}条 acc={acc:.1f}%) ===")
    for i, lb in enumerate(LAB):
        print(f"  {lb:6s}|{cm[i,0]:>5d} {cm[i,1]:>4d} {cm[i,2]:>5d} {cm[i,3]:>5d} {cm[i,4]:>5d}|{cm[i].sum():>5d}")
    buy = (pr == 4) & MASK; nb = buy.sum(); br = rets[buy]
    yr = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
    print(f"买入: {nb}只 均{br.mean()*100:.1f}% 年{np.mean(yr):.1f}% 暴涨召回{cm[4,4]}/{cm[4].sum()}")
    return np.mean(yr)

# ===== 4行业 =====
print("--- 训练 4行业(无interest_coverage) ---")
r4 = run_exp("4行业去IC", "training_final.csv", "D14_4",
    lambda u: (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1),
    "D14_4→S1∩S3")

print(f"\n基准 MLP0_5(4行业): 231只 年+21.8%")
print(f"D14_4(去IC): 年+{r4:.1f}%  差{r4-21.8:+.1f}pp")

# ===== 8行业 =====
print("\n--- 训练 8行业(无interest_coverage) ---")
r8 = run_exp("8行业去IC", "training_full.csv", "D14_8",
    lambda u: (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8),
    "D14_8→S1∩S3∩S4")

print(f"\n基准 FULL_IND(8行业): 325只 年+25.2%")
print(f"D14_8(去IC): 年+{r8:.1f}%  差{r8-25.2:+.1f}pp")
