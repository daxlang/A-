"""erankOnline 全量部署: 2019-2025训练 → 2026Q1预测 (AutoDL GPU)
REG 512-256-128, w=0.5, erank在线课程学习, 保存权重+选股
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
import warnings; warnings.filterwarnings("ignore")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEV}")

df = pd.read_csv("training_extended.csv", dtype={"code": str})
u_all = df[(df.usable) | ((df.year == 2026) & (df.quarter == 1))].copy()
u_all["gm_pct"] = u_all.groupby(["industry", "year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
def lbl(r):
    if pd.isna(r): return 0
    if r < -0.1: return 0
    if r < 0: return 1
    if r < 0.1: return 2
    if r < 0.3: return 3
    return 4
u_all["L"] = u_all.forward_return.apply(lbl)
feat = [c for c in u_all.columns if c not in ["code", "year", "quarter", "forward_return", "gm_pct", "L", "usable", "industry", "interest_coverage", "buy_price", "pe", "pb", "ps"]]
FCOLS = feat + ["buy_price", "pe", "pb", "ps"]
for yr in sorted(u_all.year.unique()):
    m = u_all.year == yr
    for c in FCOLS:
        try: lo,hi=u_all.loc[m,c].quantile(0.01),u_all.loc[m,c].quantile(0.99);u_all.loc[m,c]=u_all.loc[m,c].clip(lo,hi)
        except: pass
ind = pd.get_dummies(u_all.industry, prefix="ind").astype(float)
for c in ind.columns: u_all[c] = ind[c].values
DF = u_all[feat + list(ind.columns)].copy()
for c in DF.columns: DF[c] = DF[c].fillna(DF[c].median())

tr = (u_all.year < 2026).values; te = (u_all.year == 2026).values
X_tr = DF.values[tr]; X_te = DF.values[te]
y_tr = u_all.loc[u_all.year < 2026, "L"].values.astype(int)
N = DF.shape[1]
print(f"训练: {tr.sum()}条  测试: {te.sum()}条  特征: {N}")

os.makedirs("weights_deploy_erank", exist_ok=True)

class Backbone(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N,512),nn.ReLU(),nn.Dropout(0.5),
                                 nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),
                                 nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5))
    def forward(self, x): return self.net(x)

def full_net(N):
    return nn.Sequential(Backbone(N), nn.Linear(128, 5))

def compute_erank(act, k=50):
    """局部有效秩: per-sample k近邻矩阵的Shannon熵"""
    act_np = act.cpu().numpy(); n = len(act_np)
    eranks = np.zeros(n)
    for i in range(0, n, 500):
        end = min(i + 500, n); batch = act_np[i:end]
        dists = cosine_distances(batch, act_np)
        for j in range(len(batch)):
            knn_idx = np.argpartition(dists[j], k)[:k]
            knn_act = act_np[knn_idx] - act_np[knn_idx].mean(axis=0, keepdims=True)
            try:
                s = np.linalg.svd(knn_act, compute_uv=False)
                s = s[s > 1e-10]; p = s / s.sum()
                eranks[i+j] = np.exp(-np.sum(p * np.log(p + 1e-10)))
            except: eranks[i+j] = 0.0
    return eranks

# ===== erankOnline 全量训练 =====
sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
ys = torch.LongTensor(y_tr).to(DEV); n_samples = len(y_tr)
print(f"全量训练... {n_samples}样本")

m = full_net(N).to(DEV); backbone = m[0]
opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
bl = 1e9; bs = None; cw = torch.FloatTensor([1,1,1,1,0.5]).to(DEV)

with torch.no_grad():
    logits = m(Xs)
    il = nn.functional.cross_entropy(logits, ys, reduction='none').cpu().numpy()
loss_score = (il - il.min()) / (il.max() - il.min() + 1e-10)
difficulty = loss_score; sorted_idx = np.argsort(difficulty)
t0 = time.time()

for ep in range(10001):
    m.train()
    stage = ep // 2000
    ratios = [0.2, 0.35, 0.55, 0.75, 1.0]
    keep_ratio = ratios[min(stage, len(ratios)-1)]
    keep_n = max(int(n_samples * keep_ratio), 500)

    if ep > 0 and ep % 500 == 0:
        m.eval()
        with torch.no_grad():
            acts = backbone(Xs)
            eranks = compute_erank(acts, k=50)
            erank_score = (eranks - eranks.min()) / (eranks.max() - eranks.min() + 1e-10)
            logits_f = m(Xs)
            cur_loss = nn.functional.cross_entropy(logits_f, ys, reduction='none').cpu().numpy()
            loss_score = (cur_loss - cur_loss.min()) / (cur_loss.max() - cur_loss.min() + 1e-10)
        alpha = min(ep / 5000, 1.0)
        difficulty = loss_score * (1 - alpha * 0.5) + (1 - erank_score) * (alpha * 0.5)
        sorted_idx = np.argsort(difficulty)
    elif ep == 1:
        print(f"  难度排序完成")

    cur_idx = sorted_idx[:keep_n]
    X_cur = Xs[cur_idx]; y_cur = ys[cur_idx]
    logits_c = m(X_cur); loss_ps = nn.functional.cross_entropy(logits_c, y_cur, reduction='none')
    opt.zero_grad()
    loss = (loss_ps * cw[y_cur]).mean()
    loss.backward(); opt.step()

    with torch.no_grad():
        fl = lfn(m(Xs), ys)
    if fl.item() < bl: bl = fl.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    if ep % 2000 == 0:
        print(f"  ep{ep}: loss={fl.item():.4f} best={bl:.4f} keep={keep_ratio:.0%}")

print(f"训练完成 ({time.time()-t0:.0f}s)")
m.load_state_dict(bs); m = m.cpu().eval()
torch.save({"model":{k:v.cpu() for k,v in bs.items()},"scaler":sc}, "weights_deploy_erank/ERANK_DEPLOY.pt")

# ===== 预测 2026Q1 =====
te_mask = np.array((u_all.loc[u_all.year==2026,"cfo_to_revenue"]>=0)&
                   (u_all.loc[u_all.year==2026,"current_ratio"]>=0.5)&
                   (u_all.loc[u_all.year==2026,"gm_pct"]>=0.1)&
                   (u_all.loc[u_all.year==2026,"liability_to_asset"]<=0.8))
Xte = torch.FloatTensor(sc.transform(X_te))
with torch.no_grad(): pred = torch.argmax(m(Xte), dim=1).numpy()
buy = (pred == 4) & te_mask
te_codes = u_all.loc[u_all.year==2026, "code"].values
te_ind = u_all.loc[u_all.year==2026, "industry"].values
codes_buy = te_codes[buy]; inds_buy = te_ind[buy]
print(f"\nS1234掩码内: {te_mask.sum()}只  erankOnline判涨: {len(codes_buy)}只")
for c,ind in zip(codes_buy, inds_buy):
    print(f"  {c} {ind}")

result = u_all.loc[u_all.year==2026, ["code","industry"]].copy()
result["pred"] = pred
result.to_csv("deploy_2026Q1_erank.csv", index=False)
print(f"\nDone! 权重: weights_deploy_erank/ERANK_DEPLOY.pt  选股: deploy_2026Q1_erank.csv")
