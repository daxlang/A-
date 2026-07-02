"""双维度课程学习: erank(冻结基底) + loss (AutoDL GPU)
方法A: 全量预训→冻结backbone→每样本erank+loss→课程
方法B: 在线erank→每500ep重新排序
vs 基线 10000ep
REG 512-256-128, w=0.5, 5折CV, 保存权重
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
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
os.makedirs("weights_erank", exist_ok=True)

# Backbone extractor: 512→256→128 (不含classifier)
class Backbone(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N,512),nn.ReLU(),nn.Dropout(0.5),
                                 nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),
                                 nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5))
    def forward(self, x):
        return self.net(x)

def full_net(N):
    return nn.Sequential(Backbone(N), nn.Linear(128, 5))

def compute_erank(activations, k=50):
    """计算per-sample局部erank: 对每个样本找k近邻, 构成矩阵X, SVD后Shannon熵"""
    act = activations.cpu().numpy()
    # 用余弦距离找k近邻
    from sklearn.metrics.pairwise import cosine_distances
    n = len(act)
    eranks = np.zeros(n)
    for i in range(0, n, 500):  # 分批处理
        end = min(i + 500, n)
        batch = act[i:end]
        # 对batch中每个样本, 找全局中最近的k个
        dists = cosine_distances(batch, act)
        for j in range(len(batch)):
            knn_idx = np.argpartition(dists[j], k)[:k]
            knn_act = act[knn_idx] - act[knn_idx].mean(axis=0, keepdims=True)
            try:
                s = np.linalg.svd(knn_act, compute_uv=False)
                s = s[s > 1e-10]
                p = s / s.sum()
                eranks[i+j] = np.exp(-np.sum(p * np.log(p + 1e-10)))
            except:
                eranks[i+j] = 0.0
    return eranks

# ==================================
# 方法A: Frozen Backbone erank + loss
# ==================================
def train_erank_frozen(X_tr, y_tr, N):
    """预训一个模型→冻结backbone→计算erank→用erank+loss课程"""
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV); n_samples = len(y_tr)
    
    # Step1: 预训 5000ep
    t0 = time.time()
    m_pre = full_net(N).to(DEV)
    opt_pre = torch.optim.AdamW(m_pre.parameters(), lr=0.001, weight_decay=0.02)
    lfn_pre = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
    bl_pre = 1e9; bs_pre = None
    for ep in range(5001):
        m_pre.train(); opt_pre.zero_grad(); loss = lfn_pre(m_pre(Xs), ys); loss.backward(); opt_pre.step()
        if loss.item() < bl_pre: bl_pre = loss.item(); bs_pre = {k:v.clone() for k,v in m_pre.state_dict().items()}
    m_pre.load_state_dict(bs_pre); m_pre.eval()
    print(f"   预训 {(time.time()-t0):.0f}s", end="", flush=True)
    
    # Step2: 冻结backbone, 提取所有样本的128维激活
    backbone = m_pre[0]
    with torch.no_grad():
        acts = backbone(Xs)
    # 计算per-sample erank (选128维中语义最丰富的10维,投影到k-NN局部)
    # 简化: 直接用激活向量的全局归一化erank
    print(f" erank...", end="", flush=True)
    eranks = compute_erank(acts, k=50)
    # 归一化erank: 越高越好(表征丰富)→难度低
    erank_score = (eranks - eranks.min()) / (eranks.max() - eranks.min() + 1e-10)
    
    # Step3: 首轮全量评估loss
    with torch.no_grad():
        logits = m_pre(Xs)
        initial_loss = nn.functional.cross_entropy(logits, ys, reduction='none').cpu().numpy()
    loss_score = (initial_loss - initial_loss.min()) / (initial_loss.max() - initial_loss.min() + 1e-10)
    
    # 综合难度: erank低 + loss高 = 难; erank高 + loss低 = 易
    difficulty = loss_score * 0.5 + (1 - erank_score) * 0.5
    sorted_idx = np.argsort(difficulty)  # 从易到难
    
    del m_pre; torch.cuda.empty_cache()
    
    # Step4: 课程训练 10000ep
    m = full_net(N).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
    bl = 1e9; bs = None; cw = torch.FloatTensor([1,1,1,1,0.5]).to(DEV)
    
    for ep in range(10001):
        m.train()
        # 按ep决定保留比例
        stage = ep // 2000
        ratios = [0.2, 0.35, 0.55, 0.75, 1.0]
        keep_ratio = ratios[min(stage, len(ratios)-1)]
        keep_n = max(int(n_samples * keep_ratio), 500)
        cur_idx = sorted_idx[:keep_n]
        
        X_cur = Xs[cur_idx]; y_cur = ys[cur_idx]
        logits_c = m(X_cur); loss_ps = nn.functional.cross_entropy(logits_c, y_cur, reduction='none')
        opt.zero_grad()
        loss = (loss_ps * cw[y_cur]).mean()
        loss.backward(); opt.step()
        
        with torch.no_grad():
            fl = lfn(m(Xs), ys)
        if fl.item() < bl: bl = fl.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    
    m.load_state_dict(bs); m.cpu().eval()
    return m, sc

# ==================================
# 方法B: 在线erank + loss (每500ep)
# ==================================
def train_erank_online(X_tr, y_tr, N):
    """在线erank: 每500ep提取当前激活, 算erank, 更新难度排序"""
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
    ys = torch.LongTensor(y_tr).to(DEV); n_samples = len(y_tr)
    
    m = full_net(N).to(DEV)
    backbone = m[0]
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
    bl = 1e9; bs = None; cw = torch.FloatTensor([1,1,1,1,0.5]).to(DEV)
    
    # 初始难度: 只用loss (erank初始随机所以先不用)
    with torch.no_grad():
        logits = m(Xs)
        initial_loss = nn.functional.cross_entropy(logits, ys, reduction='none').cpu().numpy()
    loss_score = (initial_loss - initial_loss.min()) / (initial_loss.max() - initial_loss.min() + 1e-10)
    difficulty = loss_score
    sorted_idx = np.argsort(difficulty)
    erank_updated = False
    
    for ep in range(10001):
        m.train()
        stage = ep // 2000
        ratios = [0.2, 0.35, 0.55, 0.75, 1.0]
        keep_ratio = ratios[min(stage, len(ratios)-1)]
        keep_n = max(int(n_samples * keep_ratio), 500)
        
        # 每500ep在线更新难度
        if ep > 0 and ep % 500 == 0:
            m.eval()
            with torch.no_grad():
                acts = backbone(Xs)
                eranks = compute_erank(acts, k=50)
                erank_score = (eranks - eranks.min()) / (eranks.max() - eranks.min() + 1e-10)
                logits = m(Xs)
                cur_loss = nn.functional.cross_entropy(logits, ys, reduction='none').cpu().numpy()
                loss_score = (cur_loss - cur_loss.min()) / (cur_loss.max() - cur_loss.min() + 1e-10)
            # 动态混合: 初期loss主导, 后期erank主导
            alpha = min(ep / 5000, 1.0)
            difficulty = loss_score * (1 - alpha * 0.5) + (1 - erank_score) * (alpha * 0.5)
            sorted_idx = np.argsort(difficulty)
            erank_updated = True
        
        cur_idx = sorted_idx[:keep_n]
        X_cur = Xs[cur_idx]; y_cur = ys[cur_idx]
        logits_c = m(X_cur); loss_ps = nn.functional.cross_entropy(logits_c, y_cur, reduction='none')
        opt.zero_grad()
        loss = (loss_ps * cw[y_cur]).mean()
        loss.backward(); opt.step()
        
        with torch.no_grad():
            fl = lfn(m(Xs), ys)
        if fl.item() < bl: bl = fl.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    
    m.load_state_dict(bs); m.cpu().eval()
    return m, sc

# ===== 运行 =====
METHODS = [
    ("erankFrozen", train_erank_frozen),
    ("erankOnline", train_erank_online),
]

for meth_name, train_fn in METHODS:
    pr_all = np.zeros(len(u), dtype=int); t0 = time.time()
    print(f"\n=== {meth_name} ===")
    for ty in TY:
        tr = (yrs_all != ty); te = (yrs_all == ty)
        m, sc = train_fn(DF.iloc[tr].values, y_all[tr], N)
        torch.save({"model":{k:v.cpu() for k,v in m.state_dict().items()},"scaler":sc},
                    f"weights_erank/{meth_name}_{ty}.pt")
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

print(f"\n对比基线1w:  ~4200买 +19%")
print(f"Done! 权重在 weights_erank/")
