"""130只, 2019-2025, 全流程对比 (无PE过滤)"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

# === 构建训练集 ===
df_codes = pd.read_csv(os.path.join(OUT, "all_stock_codes_full.csv"), dtype=str)
code_ind = dict(zip(df_codes["code"], df_codes["industry"]))

df_f = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
df_p = pd.read_csv(os.path.join(OUT, "prices_daily.csv"), dtype={"code": str})
df_p["date"] = pd.to_datetime(df_p["date"])
df_p["close"] = pd.to_numeric(df_p["close"], errors="coerce")

price_map = {}
for code, grp in df_p.groupby("code"):
    grp = grp.dropna(subset=["close"]).sort_values("date")
    price_map[code] = grp.set_index("date")["close"]

def nearest_price(code, target):
    if code not in price_map: return np.nan
    ts = pd.Timestamp(target); s = price_map[code]
    mask = s.index >= ts
    return float(s[mask].iloc[0]) if mask.any() else float(s.iloc[-1])

def get_dates(y, q):
    if q==1: return (f"{y}-04-30", f"{y+1}-04-30")
    if q==2: return (f"{y}-07-31", f"{y+1}-07-31")
    if q==3: return (f"{y}-10-31", f"{y+1}-10-31")
    return (f"{y+1}-04-30", f"{y+2}-04-30")

rows = []
for _, r in df_f.iterrows():
    code = str(r["code"]); y = int(r["year"]); q = int(r["quarter"])
    bd, sd = get_dates(y, q)
    pb = nearest_price(code, bd); ps = nearest_price(code, sd)
    ret = (ps-pb)/pb if (pd.notna(pb) and pd.notna(ps) and pb>0) else np.nan
    rows.append({
        "code":code,"year":y,"quarter":q,
        "roe":r.get("roe"),"gross_margin":r.get("gross_margin"),
        "net_margin":r.get("net_margin"),"profit_yoy":r.get("profit_yoy"),
        "asset_turnover":r.get("asset_turnover"),"inventory_turnover":r.get("inventory_turnover"),
        "forward_return":ret,"industry":code_ind.get(code,"")
    })

df_train = pd.DataFrame(rows)
df_train["usable"] = df_train["forward_return"].notna() & (df_train["year"]<=2024)
usable = df_train[df_train["usable"]]
print(f"样本: {len(usable)}条, {usable.code.nunique()}只")
print(f"目标均值: {usable.forward_return.mean()*100:+.1f}% 中位: {usable.forward_return.median()*100:+.1f}%")

# === 特征 ===
feat = ['roe','gross_margin','net_margin','profit_yoy','asset_turnover','inventory_turnover']
ind = pd.get_dummies(usable['industry'],prefix='ind').astype(float)
X = pd.concat([usable[feat], ind], axis=1)
X = X.fillna(X.median())
y_true = usable['forward_return'].values.ravel()

train_m = (usable['year']<2024) | ((usable['year']==2024)&(usable['quarter']<=2))
test_m = ~train_m

sc_X = StandardScaler(); sc_y = StandardScaler()
Xtr = torch.FloatTensor(sc_X.fit_transform(X[train_m]))
ytr = torch.FloatTensor(sc_y.fit_transform(y_true[train_m].reshape(-1,1)))
Xte = torch.FloatTensor(sc_X.transform(X[test_m]))
yte = y_true[test_m]
print(f"训练:{Xtr.shape[0]} 测试:{Xte.shape[0]} 特征:{Xtr.shape[1]}维")

# === 基线 ===
naive = np.full_like(yte, y_true[train_m].mean())
mae0 = np.mean(np.abs(yte-naive))
print(f"\n基线(猜均值) MAE={mae0:.4f}")

lr = RidgeCV(alphas=[0.01,0.1,1,10,100],cv=5)
lr.fit(Xtr.numpy(),y_true[train_m])
pl = lr.predict(Xte.numpy())
ml= np.mean(np.abs(yte-pl)); rl,_=spearmanr(yte,pl)
print(f"线性: MAE={ml:.4f} Rank_IC={rl:+.3f} 改善={(1-ml/mae0)*100:+.1f}%")

rf = RandomForestRegressor(100,max_depth=5,min_samples_leaf=10,random_state=42)
rf.fit(Xtr.numpy(),y_true[train_m])
pr = rf.predict(Xte.numpy())
mr=np.mean(np.abs(yte-pr)); rr,_=spearmanr(yte,pr)
print(f"RF:   MAE={mr:.4f} Rank_IC={rr:+.3f} 改善={(1-mr/mae0)*100:+.1f}%")

class TM(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Xtr.shape[1], h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h // 2, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(h, lr=0.005, ep=2000):
    torch.manual_seed(42)
    m = TM(h)
    opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    best, bs = 1e9, None
    for _ in range(ep):
        m.train()
        opt.zero_grad()
        loss = nn.MSELoss()(m(Xtr), ytr)
        loss.backward()
        opt.step()
        if loss.item() < best:
            best = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): return sc_y.inverse_transform(m(Xte).numpy()).ravel()

print(f"\n=== MLP ===")
best_mae,best_h=1e9,0
for h in [8,16,32,64,128,256]:
    npv = h*Xtr.shape[1]+h + h*(h//2)+h//2 + h//2+1
    pred = train_mlp(h)
    mm = np.mean(np.abs(yte-pred)); rm,_=spearmanr(yte,pred)
    impr=(1-mm/mae0)*100
    tag=" <" if mm<best_mae else ""
    print(f"  h={h:3d} {npv:5d}参 MAE={mm:.4f} Rank_IC={rm:+.3f} 改善={impr:+.1f}%{tag}")
    if mm<best_mae: best_mae,best_h=mm,h

print(f"\n=== 汇总 ===")
print(f"线性 MAE={ml:.4f} Rank_IC={rl:+.3f} 改善={(1-ml/mae0)*100:+.1f}%")
print(f"RF   MAE={mr:.4f} Rank_IC={rr:+.3f} 改善={(1-mr/mae0)*100:+.1f}%")
print(f"MLP  MAE={best_mae:.4f} (h={best_h})")
winner = "线性" if ml<=min(mr,best_mae) else ("RF" if mr<=best_mae else "MLP")
print(f"最优: {winner}")
