"""生成附录: 各实验详细数据(混淆矩阵/逐年/行业)"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
LAB = ["暴跌","跌","小涨","中涨","暴涨"]

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def load_pred(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

def prep_4ind(csv="training_final.csv", drop=[], add_ind=True):
    df = pd.read_csv(os.path.join(BASE, csv), dtype={"code": str})
    u = df[df.usable].copy()
    u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
    u["L"] = u.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
    feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]+drop]
    X = pd.concat([u[feat]], axis=1)
    if add_ind:
        ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
        X = pd.concat([X, ind], axis=1)
    X = X.fillna(X.median())
    return u, X, u.L.values, u.forward_return.values, u.year.values

def print_cm(y, pr, mask, title):
    cm = confusion_matrix(y[mask], pr[mask]); n = cm.sum()
    acc = sum(cm[i,i] for i in range(5))/n*100
    buy = (pr == 4) & mask; nb = buy.sum(); br = rets[buy]  # need rets from outer scope
    print(f"\n### {title} ({n}条 acc={acc:.1f}%)")
    print(f"```")
    print(f"{'实际↓预测':<10s}{'暴跌':>5s}{'跌':>4s}{'小涨':>5s}{'中涨':>5s}{'暴涨':>5s}|{'合计':>5s}")
    for i, lb in enumerate(LAB):
        print(f"{lb:<10s}{cm[i,0]:>5d}{cm[i,1]:>4d}{cm[i,2]:>5d}{cm[i,3]:>5d}{cm[i,4]:>5d}|{cm[i].sum():>5d}")
    print(f"买入: {nb}只 暴涨召回{cm[4,4]}/{cm[4].sum()}")
    yrly = []
    for ty in sorted(set(yrs)):
        b = (yrs == ty) & buy; n = b.sum(); a = rets[b]
        yl = a.mean()*100 if n > 0 else 0; yrly.append(yl)
        print(f"  {ty}: {n}只 {a.mean()*100:.1f}%" if n > 0 else f"  {ty}: 0只")
    print(f"  年均: {np.mean(yrly):.1f}%")
    print(f"```")

# ====== 附录 A: 4行业基准 MLP0_5 各掩码版本 ======
print("## 附录A: 4行业 MLP0_5 详细数据\n")
u4, X4, y4, rets_full, yrs_full = prep_4ind("training_final.csv", [], True)

pr4 = np.zeros(len(u4), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (u4.year == ty); pr4[te] = load_pred(X4[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))

rets = rets_full; yrs = yrs_full
# A.1 无掩码全量
print_cm(y4, pr4, np.ones(len(u4), dtype=bool), "A.1 MLP0_5 无掩码(全量3120条)")
# A.2 S1掩码
m_s1 = (u4.cfo_to_revenue >= 0) & (u4.current_ratio >= 0.5)
print_cm(y4, pr4, m_s1, "A.2 MLP0_5 → S1(cfo>=0 & flow>=0.5)")
# A.3 S1∩S3
m_s13 = m_s1 & (u4.gm_pct >= 0.1)
print_cm(y4, pr4, m_s13, "A.3 MLP0_5 → S1∩S3(宽松)")
# A.4 S1∩S3∩S4
m_s134 = m_s13 & (u4.liability_to_asset <= 0.8)
print_cm(y4, pr4, m_s134, "A.4 MLP0_5 → S1∩S3∩S4")

# ====== 附录 B: 单权重消融模型 ======
print("\n## 附录B: 消融实验 详细数据\n")

# B.1 NoInd (无行业编码)
u4n, X4n, y4n, rets_full2, yrs_full2 = prep_4ind("training_final.csv", [], False)
pr4n = np.zeros(len(u4n), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (u4n.year == ty); pr4n[te] = load_pred(X4n[te].values, os.path.join(MDIR, f"NoInd_{ty}.pt"))
rets = rets_full2; yrs = yrs_full2
print_cm(y4n, pr4n, m_s13, "B.1 NoInd(4行业,无行业编码)→S1∩S3")

# B.2 D13(去current_ratio+interest_coverage)
u4d, X4d, y4d, rets_d, yrs_d = prep_4ind("training_final.csv", ["current_ratio","interest_coverage"], True)
pr4d = np.zeros(len(u4d), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (u4d.year == ty); pr4d[te] = load_pred(X4d[te].values, os.path.join(MDIR, f"D13_{ty}.pt"))
rets = rets_d; yrs = yrs_d
print_cm(y4d, pr4d, m_s13, "B.2 D13(去cr+ic, 13维)→S1∩S3")

# B.3 D14_4(去interest_coverage only)
u4i, X4i, y4i, rets_i, yrs_i = prep_4ind("training_final.csv", ["interest_coverage"], True)
pr4i = np.zeros(len(u4i), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (u4i.year == ty); pr4i[te] = load_pred(X4i[te].values, os.path.join(MDIR, f"D14_4_{ty}.pt"))
rets = rets_i; yrs = yrs_i
print_cm(y4i, pr4i, m_s13, "B.3 D14_4(去IC, 14维)→S1∩S3")

# B.4 S3_feat(加gm特征)
for ty in [2020,2021,2022,2023,2024]:
    te = (u4.year == ty)
    try:
        pr4[te] = load_pred(X4[te].values, os.path.join(MDIR, f"S3_feat_{ty}.pt"))
    except: pass
rets = rets_full; yrs = yrs_full
print_cm(y4, pr4, m_s13, "B.4 S3_feat(加gm_rank+gm_cfo_cross)→S1∩S3")

# ====== 附录 C: 8行业/全行业关键数据 ======
print("\n## 附录C: 8行业和全行业模型\n")

# C.1 FULL_IND
def prep_generic(csv, drop=[], add_ind=True):
    df = pd.read_csv(os.path.join(BASE, csv), dtype={"code": str})
    u = df[df.usable].copy()
    u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
    u["L"] = u.forward_return.apply(lambda r: 0 if r<-0.1 else (1 if r<0 else (2 if r<0.1 else (3 if r<0.3 else 4))))
    feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"]+drop]
    X = pd.concat([u[feat]], axis=1)
    if add_ind:
        ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
        X = pd.concat([X, ind], axis=1)
    X = X.fillna(X.median())
    return u, X, u.L.values, u.forward_return.values, u.year.values

u8, X8, y8, r8, yrs8 = prep_generic("training_full.csv")
m8 = (u8.cfo_to_revenue>=0)&(u8.current_ratio>=0.5)&(u8.gm_pct>=0.1)&(u8.liability_to_asset<=0.8)
pr8 = np.zeros(len(u8), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (u8.year == ty); pr8[te] = load_pred(X8[te].values, os.path.join(MDIR, f"FULL_IND_{ty}.pt"))
rets = r8; yrs = yrs8
print_cm(y8, pr8, m8, "C.1 FULL_IND(8行业)→S1∩S3∩S4")

# C.2 D14_8
u8d, X8d, y8d, r8d, yrs8d = prep_generic("training_full.csv", ["interest_coverage"])
pr8d = np.zeros(len(u8d), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (u8d.year == ty); pr8d[te] = load_pred(X8d[te].values, os.path.join(MDIR, f"D14_8_{ty}.pt"))
rets = r8d; yrs = yrs8d
print_cm(y8d, pr8d, m8, "C.2 D14_8(8行业去IC)→S1∩S3∩S4")

# C.3 ALL_14
ua, Xa, ya, ra, yrsa = prep_generic("training_all.csv", ["interest_coverage"])
ma = (ua.cfo_to_revenue>=0)&(ua.current_ratio>=0.5)&(ua.gm_pct>=0.1)&(ua.liability_to_asset<=0.8)
pra = np.zeros(len(ua), dtype=int)
for ty in [2020,2021,2022,2023,2024]:
    te = (ua.year == ty); pra[te] = load_pred(Xa[te].values, os.path.join(MDIR, f"ALL_14_{ty}.pt"))
rets = ra; yrs = yrsa
print_cm(ya, pra, ma, "C.3 ALL_14(全行业)→S1∩S3∩S4")

# ALL_14 逐年行业表
print(f"\n### C.3b ALL_14 逐年×行业")
print(f"```")
INDS = sorted(ua.industry.unique())
hdr = f"{'行业':<8s}"
for ty in sorted(set(yrsa)): hdr += f"  {ty:>5d}"
hdr += f"  {'合计':>5s}"
print(hdr)
for ind_name in INDS:
    row = f"{ind_name:<8s}"; ttl = 0; ttl_ret = []
    for ty in sorted(set(yrsa)):
        b = (yrsa == ty) & (pra == 4) & ma & (ua.industry == ind_name)
        n = b.sum(); a = rets[b]
        ra_val = a.mean()*100 if n > 0 else 0
        row += f" {n:>3d}只{ra_val:>+4.0f}%"; ttl += n; ttl_ret.extend(a)
    ra_final = np.mean(ttl_ret)*100 if ttl > 0 else 0
    row += f" {ttl:>3d}只{ra_final:>+4.0f}%"; print(row)
row = f"{'合计':<8s}"; all_ttl = 0; all_ret = []
for ty in sorted(set(yrsa)):
    b = (yrsa == ty) & (pra == 4) & ma; n = b.sum(); a = rets[b]
    ra_val = a.mean()*100 if n > 0 else 0
    row += f" {n:>3d}只{ra_val:>+4.0f}%"; all_ttl += n; all_ret.extend(a)
ra_final = np.mean(all_ret)*100 if all_ttl > 0 else 0
row += f" {all_ttl:>3d}只{ra_final:>+4.0f}%"; print(row)
print(f"```")

print("\n---")
print("未补充数据(需手动回测):")
print("- PE周期修复实验(18.6): S2_repair权重存在, 可补充")
print("- S4单独效果(18.12): 使用MLP0_5+S1∩S3 vs S1∩S3∩S4见A.3/A.4")
print("- 卖出策略全表(18.13): 需用卖出脚本重跑生成完整逐年明细")
print("- 季度预测(18.14): Q_MLP0_5权重存在, 需补充混淆矩阵")
print("- 三模型共识(18.10): RF/LR pickle存在, 可补充")
print("- RAND三模型(18.25): 已在主记录中")
