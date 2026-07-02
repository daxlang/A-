"""课程学习3方案: SPL / Anti-Curriculum / BabyStep (AutoDL GPU)
REG 512-256-128, w=0.5, 5折CV, 保存权重
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEV}")
TY = [2020, 2021, 2022, 2023, 2024]
LAB = ["暴跌", "跌", "小涨", "中涨", "暴涨"]

# ===== 数据 =====
df = pd.read_csv("training_extended.csv", dtype={"code": str})
u = df[(df.usable) & (df.year <= 2024)].copy()
u["gm_pct"] = u.groupby(["industry", "year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
def lbl(r):
    if pd.isna(r): return 0
    if r < -0.1: return 0
    if r < 0: return 1
    if r < 0.1: return 2
    if r < 0.3: return 3
    return 4
u["L"] = u.forward_return.apply(lbl)
feat = [c for c in u.columns if c not in ["code", "year", "quarter", "forward_return", "gm_pct", "L", "usable", "industry", "interest_coverage", "buy_price", "pe", "pb", "ps"]]
FCOLS = feat + ["buy_price", "pe", "pb", "ps"]
for yr in sorted(u.year.unique()):
    m = u.year == yr
    for c in FCOLS:
        try: lo,hi=u.loc[m,c].quantile(0.01),u.loc[m,c].quantile(0.99);u.loc[m,c]=u.loc[m,c].clip(lo,hi)
        except: pass
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
for c in ind.columns: u[c] = ind[c].values
DF = u[feat + list(ind.columns)].copy()
for c in DF.columns: DF[c] = DF[c].fillna(DF[c].median())
y_all = u.L.values.astype(int); rets_all = u.forward_return.values; yrs_all = u.year.values
MASK_all = np.array((u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5) & (u.gm_pct >= 0.1) & (u.liability_to_asset <= 0.8))
N = DF.shape[1]
print(f"Feats={N} 样本={len(u)} 掩码={MASK_all.sum()}\n")
os.makedirs("weights_cl", exist_ok=True)

def build_net(N):
    return nn.Sequential(nn.Linear(N,512),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(128,5))

def build_binary_net(N):
    return nn.Sequential(nn.Linear(N,512),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(128,2))

def standard_train(X_tr, y_tr, N, w):
    """基线训练 (无课程学习)"""
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV)
    m = build_net(N).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,w]).to(DEV))
    bl = 1e9; bs = None
    for ep in range(5001):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m = m.cpu().eval()
    return m, sc

# ==================================
# 1. SPL (Self-Paced Learning)
# ==================================
def train_spl(X_tr, y_tr, N, w):
    """自步学习: loss小的先学, 每200ep放宽阈值"""
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV)
    n_samples = len(y_tr)
    m = build_net(N).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    bl = 1e9; bs = None

    # 阈值策略: λ(t) = λ0 * (rate)^t, λ0=初始保留比例, 逐epoch放宽
    # 初期只保留loss最小的20%样本, 每500ep翻倍
    for ep in range(1, 5001):
        m.train()
        # 每200ep更新阈值
        if ep == 1: keep_ratio = 0.2
        elif ep % 500 == 1: keep_ratio = min(keep_ratio * 2, 1.0)
        # 计算每个样本的loss
        logits = m(Xs); loss_per_sample = nn.functional.cross_entropy(logits, ys, reduction='none')
        if keep_ratio < 1.0:
            threshold = torch.quantile(loss_per_sample, keep_ratio)
            mask = loss_per_sample <= threshold
            # 确保各类别都有样本
            for lb in range(5):
                if mask[ys == lb].sum() == 0:
                    mask[ys == lb] = True
        else:
            mask = torch.ones(n_samples, dtype=torch.bool, device=DEV)
        
        opt.zero_grad()
        # 加权loss (用原始类别权重)
        class_weights = torch.FloatTensor([1,1,1,1,w]).to(DEV)
        w_per_sample = class_weights[ys]
        loss = (loss_per_sample * w_per_sample * mask.float()).sum() / mask.float().sum()
        loss.backward(); opt.step()
        
        # 全量前向检查best
        with torch.no_grad():
            full_loss = nn.functional.cross_entropy(m(Xs), ys, weight=torch.FloatTensor([1,1,1,1,w]).to(DEV))
        if full_loss.item() < bl:
            bl = full_loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    
    m.load_state_dict(bs); m = m.cpu().eval()
    return m, sc

# ==================================
# 2. Anti-Curriculum
# ==================================
def train_anti(X_tr, y_tr, N, w):
    """反课程: loss大的(难的)先学, 逐步加入简单的"""
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV)
    n_samples = len(y_tr)
    m = build_net(N).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    bl = 1e9; bs = None

    for ep in range(1, 5001):
        m.train()
        if ep == 1: keep_ratio = 0.2  # 初始只保留最难的20%
        elif ep % 500 == 1: keep_ratio = min(keep_ratio * 2, 1.0)
        
        logits = m(Xs); loss_per_sample = nn.functional.cross_entropy(logits, ys, reduction='none')
        if keep_ratio < 1.0:
            # 反着来: 保留loss最大的
            threshold = torch.quantile(loss_per_sample, 1.0 - keep_ratio)
            mask = loss_per_sample >= threshold
            for lb in range(5):
                if mask[ys == lb].sum() == 0:
                    mask[ys == lb] = True
        else:
            mask = torch.ones(n_samples, dtype=torch.bool, device=DEV)
        
        class_weights = torch.FloatTensor([1,1,1,1,w]).to(DEV)
        w_per_sample = class_weights[ys]
        opt.zero_grad()
        loss = (loss_per_sample * w_per_sample * mask.float()).sum() / mask.float().sum()
        loss.backward(); opt.step()
        
        with torch.no_grad():
            full_loss = nn.functional.cross_entropy(m(Xs), ys, weight=torch.FloatTensor([1,1,1,1,w]).to(DEV))
        if full_loss.item() < bl:
            bl = full_loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    
    m.load_state_dict(bs); m = m.cpu().eval()
    return m, sc

# ==================================
# 3. Baby Step
# ==================================
def train_babystep(X_tr, y_tr, N, w):
    """Baby Step: 先二分类(暴涨vs暴跌) 1000ep, 再五分类 4000ep"""
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV)
    
    # 构造二分类: 只保留暴跌(0)和暴涨(4)
    mask_binary = (ys == 0) | (ys == 4)
    X_bin = Xs[mask_binary]; y_bin = (ys[mask_binary] == 4).long()  # 0=暴跌, 1=暴涨
    
    # 第一阶段: 二分类 1000ep
    m_bin = build_binary_net(N).to(DEV)
    opt_bin = torch.optim.AdamW(m_bin.parameters(), lr=0.001, weight_decay=0.02)
    lfn_bin = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, w]).to(DEV))
    for ep in range(1001):
        m_bin.train(); opt_bin.zero_grad(); loss = lfn_bin(m_bin(X_bin), y_bin); loss.backward(); opt_bin.step()
    
    # 提取 backbone 权重 (前 5 层, 共享)
    bin_state = {k: v for k, v in m_bin.state_dict().items()}
    
    # 第二阶段: 五分类 4000ep, 从二分类权重复制 backbone
    m = build_net(N).to(DEV)
    # 复制共享层 (所有层名相同除了最后一层)
    m_state = m.state_dict()
    for k in m_state:
        if k in bin_state and m_state[k].shape == bin_state[k].shape:
            m_state[k] = bin_state[k]
    m.load_state_dict(m_state)
    
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,w]).to(DEV))
    bl = 1e9; bs = None
    for ep in range(4001):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    
    m.load_state_dict(bs); m = m.cpu().eval()
    return m, sc

# ==================================
# 运行扫描
# ==================================
METHODS = [
    ("基线", standard_train),
    ("SPL", train_spl),
    ("Anti", train_anti),
    ("Baby", train_babystep),
]

for meth_name, train_fn in METHODS:
    pr_all = np.zeros(len(u), dtype=int); t0 = time.time()
    print(f"\n=== {meth_name} ===")
    
    for ty in TY:
        tr = (yrs_all != ty); te = (yrs_all == ty)
        X_tr = DF.iloc[tr].values; y_tr = y_all[tr]
        m, sc = train_fn(X_tr, y_tr, N, 0.5)
        torch.save({"model":{k:v.cpu() for k,v in m.state_dict().items()},"scaler":sc},
                    f"weights_cl/{meth_name}_{ty}.pt")
        Xte = torch.FloatTensor(sc.transform(DF.iloc[te].values))
        with torch.no_grad(): pr_all[te] = torch.argmax(m.cpu()(Xte), dim=1).numpy()
        del m, sc; print(ty, end=" ", flush=True)
    
    dt = time.time() - t0; buy = (pr_all == 4) & MASK_all
    nb = int(buy.sum()); br = rets_all[buy]
    yr_vals = [rets_all[(yrs_all == ty) & buy].mean() * 100 for ty in TY if sum((yrs_all == ty) & buy) > 0]
    ym = np.mean(yr_vals) if yr_vals else 0
    cm = confusion_matrix(y_all[MASK_all], pr_all[MASK_all])
    rec = cm[4,4] / cm[4].sum() * 100 if cm[4].sum() > 0 else 0
    fb = int(cm[0,4])
    print(f"({dt:.0f}s)")
    print(f"  买{nb}只  年{ym:+.1f}%  召回{rec:.1f}%  FB{fb}")
    for ty_val in TY:
        bv = (yrs_all == ty_val) & buy; n1 = bv.sum(); a = rets_all[bv]
        if n1 > 0: print(f"    {ty_val}: {n1}只 {a.mean()*100:.1f}%")
    print()

# 对比
print(f"{'方法':<8s}{'买':>6s}{'年':>7s}{'FB':>6s}{'2023':>8s}")
print(f"{'基线':<8s}(回想: ~4200买 +19-23% 等权基线+12.3%)")
