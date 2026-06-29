"""Data Cartography: 4行业 MLP0_5 每epoch记录confidence + 分区重训"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
N_EP = 5000
SAVE_EVERY = 100  # 每100epoch存一次

df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1)

# ===== Step 1: 每fold训练 + 记录conf =====
print("Step1: 训练+记录每epoch概率...")
all_conf = {}  # fold -> {epoch: np.array of probs}

for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    n_tr = tr.sum()
    print(f"  fold {ty}: 训{n_tr}条")
    
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X[tr].values)); ys = torch.LongTensor(y[tr])
    
    class M5(nn.Module):
        def __init__(self): super().__init__()
            self.net = nn.Sequential(nn.Linear(Xs.shape[1],128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
        def forward(self, x): return self.net(x)
    
    m = M5(); opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]))
    bl, bs = 1e9, None
    fold_confs = {}
    
    for ep in range(1, N_EP+1):
        m.train(); opt.zero_grad(); out = m(Xs); loss = lfn(out, ys)
        loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
        if ep % SAVE_EVERY == 0:
            m.eval()
            with torch.no_grad():
                probs = torch.softmax(m(Xs), dim=1).cpu().numpy()
                gold_probs = probs[np.arange(n_tr), ys.cpu().numpy()]
            fold_confs[ep] = gold_probs
    
    # 最终保存
    m.load_state_dict(bs); m.eval(); torch.save({"model":bs,"scaler":sc}, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
    all_conf[ty] = fold_confs

# ===== Step 2: 计算 cartography 指标 =====
print("\nStep2: 计算 Cartography 指标...")
cartography = {}  # fold -> DataFrame with confidence, variability

for ty in TY:
    tr = (yrs != ty);
    epochs = sorted(all_conf[ty].keys())
    conf_matrix = np.array([all_conf[ty][ep] for ep in epochs])  # [n_epochs, n_samples]
    
    confidence = conf_matrix.mean(axis=0)
    variability = conf_matrix.std(axis=0)
    correctness = conf_matrix[-1]  # final epoch prob
    
    n_tr = tr.sum()
    df_fold = pd.DataFrame({
        "confidence": confidence, "variability": variability,
        "correctness": correctness, "true_label": y[tr],
        "year": yrs[tr], "industry": u.iloc[tr]["industry"].values,
        "forward_return": rets[tr], "idx_orig": np.where(tr)[0]
    })
    cartography[ty] = df_fold
    n_easy = (df_fold.confidence > 0.8).sum()
    n_hard = ((df_fold.confidence < 0.3) & (df_fold.variability < 0.15)).sum()
    n_amb = len(df_fold) - n_easy - n_hard
    print(f"  fold {ty}: Easy={n_easy}({n_easy/len(df_fold)*100:.0f}%) Amb={n_amb}({n_amb/len(df_fold)*100:.0f}%) Hard={n_hard}({n_hard/len(df_fold)*100:.0f}%)")

# ===== Step 3: 全局汇总购物车图 =====
all_rows = []
for ty in TY: all_rows.append(cartography[ty])
cart_all = pd.concat(all_rows, ignore_index=True)
n_e = (cart_all.confidence > 0.8).sum()
n_h = ((cart_all.confidence < 0.3) & (cart_all.variability < 0.15)).sum()
n_a = len(cart_all) - n_e - n_h
print(f"\n全量(跨fold): Easy={n_e}({n_e/len(cart_all)*100:.0f}%) Amb={n_a}({n_a/len(cart_all)*100:.0f}%) Hard={n_h}({n_h/len(cart_all)*100:.0f}%)")

# 检查Hard区样本分布
hard = cart_all[(cart_all.confidence < 0.3) & (cart_all.variability < 0.15)]
print(f"\nHard区({len(hard)}条) 分布:")
for lb in range(5):
    n = (hard.true_label == lb).sum()
    print(f"  {LAB[lb]}: {n}条 ({n/len(hard)*100:.0f}%) 均收益{hard[hard.true_label==lb].forward_return.mean()*100:.1f}%")
for ind_name in sorted(hard.industry.unique()):
    n = (hard.industry == ind_name).sum()
    print(f"  {ind_name}: {n}条 均收益{hard[hard.industry==ind_name].forward_return.mean()*100:.1f}%")

# ===== Step 4: 重训(仅Easy+Ambiguous) =====
print("\nStep4: 仅Easy+Ambiguous子集重训...")

class M5(nn.Module):
    def __init__(self, n_in): super().__init__()
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

# 对每个fold: 找到Hard样本, 从训练集剔除, 重训
pr_all = np.zeros(len(u), dtype=int)
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    df_f = cartography[ty]
    hard_idx = df_f[(df_f.confidence < 0.3) & (df_f.variability < 0.15)].idx_orig.values
    tr_clean = tr.copy()
    for h in hard_idx: tr_clean[h] = False
    n_removed = tr.sum() - tr_clean.sum()
    train_save(X[tr_clean].values, y[tr_clean], os.path.join(MDIR, f"CART_{ty}.pt"))
    pr_all[te] = load_pred(X[te].values, os.path.join(MDIR, f"CART_{ty}.pt"))
    print(f"  fold {ty}: 剔除Hard{n_removed}条 剩训{tr_clean.sum()}条")

# ===== Step 5: 评估 =====
print("\nStep5: 评估...")
cm = confusion_matrix(y[MASK], pr_all[MASK]); n = cm.sum()
acc = sum(cm[i,i] for i in range(5)) / n * 100
print(f"=== CART(Easy+Ambiguous) → S1∩S3 ({n}条 acc={acc:.1f}%) ===")
for i, lb in enumerate(LAB):
    print(f"  {lb:6s}|{cm[i,0]:>5d} {cm[i,1]:>4d} {cm[i,2]:>5d} {cm[i,3]:>5d} {cm[i,4]:>5d}|{cm[i].sum():>5d}")
buy = (pr_all == 4) & MASK; nb = buy.sum(); br = rets[buy]
yr = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
print(f"买入: {nb}只 均{br.mean()*100:.1f}% 年{np.mean(yr):.1f}% 暴涨召回{cm[4,4]}/{cm[4].sum()}")
for ty in TY:
    b = (yrs == ty) & buy; n = b.sum(); a = rets[b]
    print(f"  {ty}: {n}只 {a.mean()*100:.1f}%" if n > 0 else f"  {ty}: 未买")

print(f"\n基准 MLP0_5: 231买 年+21.8%")
print(f"CART去Hard: {nb}买 年{np.mean(yr):.1f}%")
