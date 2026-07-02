"""固定延迟版 部署: REG512 w=0.5 训2019-2025 测2026Q1 (AutoDL GPU)"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.preprocessing import StandardScaler
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

tr = u_all.year < 2026; te = u_all.year == 2026
print(f"训练: {tr.sum()}条 ({u_all[tr].code.nunique()}只)  测试: {te.sum()}条")

X_tr = DF.iloc[tr].values; y_tr = u_all.loc[tr, "L"].values.astype(int)
X_te = DF.iloc[te].values
MASK_te = np.array((u_all.loc[te, "cfo_to_revenue"] >= 0) & (u_all.loc[te, "current_ratio"] >= 0.5) &
                    (u_all.loc[te, "gm_pct"] >= 0.1) & (u_all.loc[te, "liability_to_asset"] <= 0.8))
N = DF.shape[1]

os.makedirs("weights_deploy", exist_ok=True)

# ===== REG 512-256-128 =====
print("\n训练 REG 512-256-128 w=0.5...")
sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X_tr)).to(DEV)
ys = torch.LongTensor(y_tr).to(DEV)
m = nn.Sequential(nn.Linear(N,512),nn.ReLU(),nn.Dropout(0.5),
                  nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.5),
                  nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5),
                  nn.Linear(128,5)).to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.02)
lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]).to(DEV))
bl = 1e9; bs = None; t0 = time.time()
for ep in range(5001):
    m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
    if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
m.load_state_dict(bs); m = m.cpu().eval()
torch.save({"model":{k:v.cpu() for k,v in bs.items()},"scaler":sc}, "weights_deploy/REG512_DEPLOY.pt")
print(f"  done ({time.time()-t0:.0f}s)")

Xte = torch.FloatTensor(sc.transform(X_te))
with torch.no_grad(): reg_pred = torch.argmax(m(Xte), dim=1).numpy()

buy = (reg_pred == 4) & MASK_te
te_codes = u_all.loc[te, "code"].values
te_ind = u_all.loc[te, "industry"].values
print(f"\nS1234掩码内: {MASK_te.sum()}只  REG判涨: {buy.sum()}只")

# 输出选股
codes_buy = te_codes[buy]; inds_buy = te_ind[buy]
print(f"\n选股 {len(codes_buy)}只:")
for c, ind in zip(codes_buy, inds_buy):
    print(f"  {c} {ind}")

# 保存
u_all.loc[te, ["code","industry"]].assign(pred=reg_pred).to_csv("deploy_2026Q1.csv", index=False)

# ===== 拉当前价+公告日买入价 =====
print("\n=== 选股当前表现 ===")
codes_buy = list(te_codes[buy])
inds_buy = list(te_ind[buy])

# 从数据集中取出每只的公告日买入价(已计算好的buy_price)
bp_map = dict(zip(u_all.loc[te, "code"], u_all.loc[te, "buy_price"]))

# 从价格缓存拉最新收盘价
import pickle
price_cache_dir = "price_cache"  # 需要用户把价格缓存也上传
# 如果没有价格缓存,尝试直接用akshare实时拉
try:
    import akshare as ak
    names = {}
    print("拉取股票名称...", end=" ", flush=True)
    all_stocks = ak.stock_info_a_code_name()
    for c in codes_buy:
        m = all_stocks[all_stocks['code'] == c]
        names[c] = m.iloc[0]['name'] if len(m) > 0 else c
    print("done")
    prices = {}
    print("拉取当前价...", end=" ", flush=True)
    for c in codes_buy:
        try:
            k = ak.stock_zh_a_hist(symbol=c, period='daily', start_date='20260601', end_date='20260630', adjust='qfq')
            if len(k) > 0: prices[c] = k['收盘'].iloc[-1]
        except: prices[c] = None
    print("done")
    
    from tabulate import tabulate
    print()
    rows = []
    for c, ind in zip(codes_buy, inds_buy):
        bp = bp_map.get(c, None)
        cur = prices.get(c, None)
        if bp and cur:
            ch = (cur - bp) / bp * 100
            rows.append([c, names.get(c, c), ind, f"{bp:.2f}", f"{cur:.2f}", f"{ch:+.1f}%"])
        elif bp:
            rows.append([c, names.get(c, c), ind, f"{bp:.2f}", "--", "--"])
    rows.sort(key=lambda x: float(x[-1].replace('%','').replace('+','')) if x[-1] != '--' else 0, reverse=True)
    print(f"{'代码':<8s}{'名称':<8s}{'行业':<8s}{'买入价':>8s}{'当前价':>8s}{'涨跌':>8s}")
    for r in rows:
        print(f"{r[0]:<8s}{r[1]:<8s}{r[2]:<8s}{r[3]:>8s}{r[4]:>8s}{r[5]:>8s}")
    
    gains = [float(r[5].replace('%','')) for r in rows if r[5] != '--']
    if gains:
        print(f"\n等权平均: {sum(gains)/len(gains):.1f}%  上涨: {sum(1 for g in gains if g>0)}/{len(gains)}只")
except Exception as e:
    print(f"价格拉取失败({e}), 选股代码已保存到 deploy_2026Q1.csv")

print(f"\nDone! 权重: weights_deploy/REG512_DEPLOY.pt")
