"""Cartography v2: 轻量版, 每200epoch记录, 存中间结果"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
LAB = ["暴跌","跌","小涨","中涨","暴涨"]

df = pd.read_csv(os.path.join(BASE, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.L.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]
MASK = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1)
SAVE_EP = 200; N_EP = 5000

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

# Step1: 每个fold训练+记录, 存中间文件
print("Step1: 训练tracking...")
for ty in TY:
    cache_path = os.path.join(MDIR, f"cart_cache_{ty}.pkl")
    if os.path.exists(cache_path):
        print(f"  fold {ty}: 缓存命中, 跳过")
        continue
    
    tr = (yrs != ty); n_tr = tr.sum()
    print(f"  fold {ty}: 训{n_tr}条...")
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X[tr].values)); ys = torch.LongTensor(y[tr])
    m = M5(Xs.shape[1]); opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]))
    bl, bs = 1e9, None
    ep_confs = {}
    
    for ep in range(1, N_EP+1):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
        if ep % SAVE_EP == 0:
            m.eval()
            with torch.no_grad():
                probs = torch.softmax(m(Xs), dim=1).cpu().numpy()
                ep_confs[ep] = probs[np.arange(n_tr), ys.cpu().numpy()]
    
    m.load_state_dict(bs); m.eval()
    torch.save({"model":bs,"scaler":sc}, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
    with open(cache_path, "wb") as f: pickle.dump({"confs":ep_confs, "tr_idx":np.where(tr)[0]}, f)
    print(f"    已存 cache {ty}")

# Step2: 计算cartography
print("\nStep2: 计算指标...")
cart_all_rows = []
for ty in TY:
    with open(os.path.join(MDIR, f"cart_cache_{ty}.pkl"), "rb") as f:
        data = pickle.load(f)
    confs = data["confs"]; tr_idx = data["tr_idx"]
    epochs = sorted(confs.keys())
    cmat = np.array([confs[ep] for ep in epochs])
    confidence = cmat.mean(axis=0); variability = cmat.std(axis=0)
    
    cart_all_rows.append(pd.DataFrame({
        "fold": ty, "confidence": confidence, "variability": variability,
        "correctness": cmat[-1], "true_label": y[tr_idx],
        "year": yrs[tr_idx], "industry": u.iloc[tr_idx]["industry"].values,
        "forward_return": rets[tr_idx], "idx_orig": tr_idx
    }))

cart = pd.concat(cart_all_rows, ignore_index=True)
# 阈值: 论文使用top/bottom 33%
c_thresh = np.percentile(cart.confidence, [33, 66])
v_thresh = np.percentile(cart.variability, [33, 66])
# Easy: high conf low var; Ambiguous: low conf high var; Hard: low conf low var
is_easy = (cart.confidence >= c_thresh[1]) & (cart.variability <= v_thresh[0])
is_amb = (cart.confidence <= c_thresh[0]) & (cart.variability >= v_thresh[1])
is_hard = (cart.confidence <= c_thresh[0]) & (cart.variability <= v_thresh[0])

print(f"总训练样本: {len(cart)}")
print(f"  Easy(高conf低var): {is_easy.sum()}({is_easy.sum()/len(cart)*100:.0f}%)")
print(f"  Ambiguous(低conf高var): {is_amb.sum()}({is_amb.sum()/len(cart)*100:.0f}%)")
print(f"  Hard(低conf低var): {is_hard.sum()}({is_hard.sum()/len(cart)*100:.0f}%)")

# Hard区分布
hard = cart[is_hard]
print(f"\nHard区 {len(hard)}条:")
for lb in range(5):
    n = (hard.true_label == lb).sum()
    print(f"  {LAB[lb]}: {n}条 均收益{hard[hard.true_label==lb].forward_return.mean()*100:.1f}%")

# Step3: 重训(去Hard)
print("\nStep3: 去Hard重训...")

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

hard_idx_set = set(cart[is_hard].idx_orig.values)
pr = np.zeros(len(u), dtype=int)
for ty in TY:
    tr = (yrs != ty); te = (yrs == ty)
    tr_clean = tr.copy()
    for h in hard_idx_set: tr_clean[h] = False
    n_rem = tr.sum() - tr_clean.sum()
    train_save(X[tr_clean].values, y[tr_clean], os.path.join(MDIR, f"CART_{ty}.pt"))
    pr[te] = load_pred(X[te].values, os.path.join(MDIR, f"CART_{ty}.pt"))
    print(f"  fold {ty}: 去Hard{n_rem}条 剩训{tr_clean.sum()}条")

# Step4: 评估
print("\nStep4: 评估...")
cm = confusion_matrix(y[MASK], pr[MASK]); n = cm.sum()
acc = sum(cm[i,i] for i in range(5)) / n * 100
print(f"=== CART(去Hard) → S1∩S3 ({n}条 acc={acc:.1f}%) ===")
for i, lb in enumerate(LAB):
    print(f"  {lb:6s}|{cm[i,0]:>5d} {cm[i,1]:>4d} {cm[i,2]:>5d} {cm[i,3]:>5d} {cm[i,4]:>5d}|{cm[i].sum():>5d}")
buy = (pr == 4) & MASK; nb = buy.sum(); br = rets[buy]
yr = [rets[(yrs==ty)&buy].mean()*100 for ty in TY if sum((yrs==ty)&buy)>0]
print(f"买入: {nb}只 均{br.mean()*100:.1f}% 年{np.mean(yr):.1f}% 暴涨召回{cm[4,4]}/{cm[4].sum()}")
for ty in TY:
    b = (yrs == ty) & buy; n = b.sum(); a = rets[b]
    print(f"  {ty}: {n}只 {a.mean()*100:.1f}%" if n > 0 else f"  {ty}: 未买")
print(f"\n基准 MLP0_5: 231买 年+21.8%")
print(f"CART去Hard: {nb}买 年{np.mean(yr):.1f}%")
