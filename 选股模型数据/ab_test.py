"""A/B对照: PE过滤 vs 无过滤 (同套2019-2025数据)"""
import warnings; warnings.filterwarnings('ignore')
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

df_codes = pd.read_csv(os.path.join(OUT, "all_stock_codes_full.csv"), dtype=str)
code_ind = dict(zip(df_codes["code"], df_codes["industry"]))

df_f = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
df_p = pd.read_csv(os.path.join(OUT, "prices_daily.csv"), dtype={"code": str})
df_p["date"] = pd.to_datetime(df_p["date"])
df_p["close"] = pd.to_numeric(df_p["close"], errors="coerce")

# 价格索引
pm = {}
for code, grp in df_p.groupby("code"):
    grp = grp.dropna(subset=["close"]).sort_values("date")
    pm[code] = grp.set_index("date")["close"]

def np_(code, target):
    if code not in pm: return np.nan
    ts = pd.Timestamp(target); s = pm[code]
    m = s.index >= ts
    return float(s[m].iloc[0]) if m.any() else float(s.iloc[-1])

def gd(y, q):
    if q==1: return (f"{y}-04-30", f"{y+1}-04-30")
    if q==2: return (f"{y}-07-31", f"{y+1}-07-31")
    if q==3: return (f"{y}-10-31", f"{y+1}-10-31")
    return (f"{y+1}-04-30", f"{y+2}-04-30")

# 构建全量训练集
rows = []
for _, r in df_f.iterrows():
    code = str(r["code"]); y = int(r["year"]); q = int(r["quarter"])
    bd, sd = gd(y, q)
    pb = np_(code, bd); ps = np_(code, sd)
    ret = (ps-pb)/pb if (pd.notna(pb) and pd.notna(ps) and pb>0) else np.nan
    rows.append({
        "code":code,"year":y,"quarter":q,
        "roe":r.get("roe"),"gross_margin":r.get("gross_margin"),
        "net_margin":r.get("net_margin"),"profit_yoy":r.get("profit_yoy"),
        "asset_turnover":r.get("asset_turnover"),"inventory_turnover":r.get("inventory_turnover"),
        "forward_return":ret,"industry":code_ind.get(code,"")
    })

df_all = pd.DataFrame(rows)
df_all["usable"] = df_all["forward_return"].notna() & (df_all["year"]<=2024)

# PE标签
df_pe = pd.read_csv(os.path.join(OUT, "stocks_after_pe_filter.csv"), dtype=str)
pe_valid = set(df_pe["code"].tolist())

# === 模型函数 ===
feat = ['roe','gross_margin','net_margin','profit_yoy','asset_turnover','inventory_turnover']

class TM(nn.Module):
    def __init__(self, n_in, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h//2, 1)
        )
    def forward(self, x):
        return self.net(x)

def run_group(label, df_sub):
    usable = df_sub[df_sub["usable"]]
    ind = pd.get_dummies(usable['industry'], prefix='ind').astype(float)
    X = pd.concat([usable[feat], ind], axis=1)
    X = X.fillna(X.median())
    y_true = usable['forward_return'].values.ravel()
    
    train_m = (usable['year']<2024)|((usable['year']==2024)&(usable['quarter']<=2))
    test_m = ~train_m
    
    sc_X = StandardScaler(); sc_y = StandardScaler()
    Xtr = torch.FloatTensor(sc_X.fit_transform(X[train_m]))
    ytr = torch.FloatTensor(sc_y.fit_transform(y_true[train_m].reshape(-1,1)))
    Xte = torch.FloatTensor(sc_X.transform(X[test_m]))
    yte = y_true[test_m]
    
    naive = np.full_like(yte, y_true[train_m].mean())
    mae0 = np.mean(np.abs(yte-naive))
    
    # 线性
    lr = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5)
    lr.fit(Xtr.numpy(), y_true[train_m])
    pl = lr.predict(Xte.numpy())
    ml = np.mean(np.abs(yte-pl)); rl,_ = spearmanr(yte,pl)
    
    # RF
    rf = RandomForestRegressor(100, max_depth=5, min_samples_leaf=10, random_state=42)
    rf.fit(Xtr.numpy(), y_true[train_m])
    pr = rf.predict(Xte.numpy())
    mr = np.mean(np.abs(yte-pr)); rr,_ = spearmanr(yte,pr)
    
    # MLP - 试到最优
    best_m, best_h, best_r = 1e9, 0, 0
    for h in [8,16,32,64,128,256]:
        torch.manual_seed(42)
        m = TM(Xtr.shape[1], h)
        opt = optim.AdamW(m.parameters(), lr=0.005, weight_decay=1e-3)
        best_loss, bs = 1e9, None
        for _ in range(2000):
            m.train(); opt.zero_grad()
            loss = nn.MSELoss()(m(Xtr), ytr)
            loss.backward(); opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                bs = {k: v.clone() for k, v in m.state_dict().items()}
        m.load_state_dict(bs); m.eval()
        with torch.no_grad():
            pred = sc_y.inverse_transform(m(Xte).numpy()).ravel()
        mm = np.mean(np.abs(yte-pred)); rm,_ = spearmanr(yte,pred)
        if mm < best_m:
            best_m, best_h, best_r = mm, h, rm
    
    print(f"\n{label}: {Xtr.shape[0]}训/{Xte.shape[0]}测, {usable.code.nunique()}只")
    print(f"  基线MAE={mae0:.4f}")
    print(f"  线性:  MAE={ml:.4f}  R_IC={rl:+.3f}  改善={(1-ml/mae0)*100:+.1f}%")
    print(f"  RF:    MAE={mr:.4f}  R_IC={rr:+.3f}  改善={(1-mr/mae0)*100:+.1f}%")
    print(f"  MLP(h={best_h}): MAE={best_m:.4f}  R_IC={best_r:+.3f}  改善={(1-best_m/mae0)*100:+.1f}%")
    return {"lr_mae":ml,"lr_ric":rl,"rf_mae":mr,"rf_ric":rr,"mlp_mae":best_m,"mlp_ric":best_r}

# === 跑两组 ===
print(f"全量: {len(df_all)}条, 可用: {df_all['usable'].sum()}条")
print(f"PE有效: {len(pe_valid)}只\n")

rA = run_group("A组: 130只(不过滤)", df_all)
df_B = df_all[df_all["code"].isin(pe_valid)]
rB = run_group("B组: 93只(PE过滤)", df_B)

print(f"\n=== A/B对照 ===")
for k in ['lr_mae','lr_ric','rf_mae','rf_ric','mlp_mae','mlp_ric']:
    va, vb = rA[k], rB[k]
    better = "B更优" if (vb<va if 'mae' in k else vb>va) else "A更优"
    print(f"  {k}: A={va:.4f}  B={vb:.4f}  {better}")
