"""部署训练: 2019-2025训REG+RF, 2026Q1预测"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, pickle, copy, time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
DEV = torch.device("cuda")
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
SEED = 42

# ===== 加载 + 预处理 =====
df = pd.read_csv(os.path.join(BASE, "training_extended.csv"), dtype={"code": str})
u = df[df.usable].copy()

# 2026Q1: 预测用, 放宽usable
u_2026q1 = df[(df.year == 2026) & (df.quarter == 1)].copy()
if len(u_2026q1) > 0:
    u_2026q1["usable"] = True
    u = pd.concat([u, u_2026q1], ignore_index=True)

print(f"数据: {len(u)}条 ({u.code.nunique()}只)")

# 特征
u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
u["L"] = u.forward_return.apply(lambda r: 0 if pd.isna(r) else (0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4)))))
feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry","interest_coverage","buy_price","pe","pb","ps"]]

# Winsorize: 特征列1%/99% 训测分别缩尾
FCOLS = feat + ["buy_price","pe","pb","ps"]
u_tr = u[(u.year < 2026) | ((u.year == 2026) & (u.quarter < 1))]
u_te = u[(u.year == 2026) & (u.quarter == 1)]

for c in FCOLS:
    lo, hi = u_tr[c].quantile(0.01), u_tr[c].quantile(0.99)
    u_tr[c] = u_tr[c].clip(lo, hi)
    u_te[c] = u_te[c].clip(lo, hi)

# Industry dummies
ind = pd.get_dummies(u_tr.industry, prefix="ind").astype(float)
for c in ind.columns:
    u_tr[c] = ind[c].values
ind_te = pd.get_dummies(u_te.industry, prefix="ind").astype(float)
for c in ind.columns:
    u_te[c] = ind_te[c].values if c in ind_te.columns else 0.0

FEAT_FINAL = feat + list(ind.columns)
DF = pd.concat([u_tr[FCOLS], u_tr[FEAT_FINAL]], axis=1)
DF_te = pd.concat([u_te[FCOLS], u_te[FEAT_FINAL]], axis=1)

# 填补
for c in DF.columns: DF[c] = DF[c].fillna(DF[c].median())
for c in DF_te.columns: DF_te[c] = DF_te[c].fillna(DF_te[c].median())

# 仅训练前年份的数据(可计算forward_return的)
u_tr_train = u_tr[u_tr.forward_return.notna()]
DF_train = DF.iloc[u_tr_train.index]
y_train = u_tr_train.L.values.astype(int)

# 掩码
MASK_te = (u_te.cfo_to_revenue >= 0) & (u_te.current_ratio >= 0.5) & (u_te.gm_pct >= 0.1) & (u_te.liability_to_asset <= 0.8)

print(f"训练: {len(DF_train)}条 测试: {len(DF_te)}条 掩码内: {MASK_te.sum()}条")

# ===== 训练 RF =====
print("\n训练 RF...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=50,
    class_weight={0:1,1:1,2:1,3:1,4:0.5}, n_jobs=-1, random_state=SEED)
rf.fit(DF_train.values, y_train)
rf_pred = rf.predict(DF_te.values)
with open(os.path.join(MDIR, "RF_DEPLOY.pkl"), "wb") as f: pickle.dump(rf, f)
print(f"  done ({time.time()-t0:.0f}s)")

# ===== 训练 REG =====
print("\n训练 REG...")
sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(DF_train.values)).to(DEV); ys = torch.LongTensor(y_train).to(DEV)
net = nn.Sequential(nn.Linear(Xs.shape[1],256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,5)).to(DEV)
opt = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.02)
lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
bl, bs = 1e9, None
t0 = time.time()
for ep in range(1, 5001):
    net.train(); opt.zero_grad(); loss = lfn(net(Xs), ys); loss.backward(); opt.step()
    if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in net.state_dict().items()}
net.load_state_dict(bs); net = net.cpu()
bs_cpu = {k:v.cpu() for k,v in bs.items()}
torch.save({"model":bs_cpu,"scaler":sc}, os.path.join(MDIR, "REG_DEPLOY.pt"))
print(f"  done ({time.time()-t0:.0f}s)")

# 预测
Xte = torch.FloatTensor(sc.transform(DF_te.values))
net.eval()
with torch.no_grad(): reg_pred = torch.argmax(net(Xte), dim=1).numpy()

# ===== 选股 =====
u_te = u_te.copy()
u_te["pred_rf"] = rf_pred
u_te["pred_reg"] = reg_pred

consensus = (reg_pred == 4) & (rf_pred == 4) & MASK_te.values
reg_only = (reg_pred == 4) & MASK_te.values

# REG∩RF
picks_c = u_te[consensus][["code","industry"]].drop_duplicates()
print(f"\n{'='*60}")
print(f"REG∩RF共识: {len(picks_c)}只")
for _, r in picks_c.iterrows():
    print(f"  {r['code']}  {r['industry']}")

# REG单独备
reg_no_consensus = reg_only & ~consensus
if picks_c.empty:
    picks_r = u_te[reg_only][["code","industry"]].drop_duplicates()
    print(f"\nREG单独(兜底): {len(picks_r)}只")
    for _, r in picks_r.iterrows():
        print(f"  {r['code']}  {r['industry']}")
elif reg_no_consensus.sum() > 0:
    print(f"\nREG兜底备选(无共识时): {u_te[reg_no_consensus].code.nunique()}只")

print(f"\n掩码内总数: {MASK_te.sum()} REG判涨:{reg_only.sum()} RF判涨:{(rf_pred==4).sum()}")

# 保存预测结果
u_te[["code","industry","pred_rf","pred_reg"]].to_csv(os.path.join(BASE, "deploy_pred_2026Q1.csv"), index=False)
print("结果已存: deploy_pred_2026Q1.csv")
