"""顿悟实验: MLP长训练, 看测试集是否突然提升"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

# 用B组: 93只PE过滤
df_pe = pd.read_csv(os.path.join(OUT, "stocks_after_pe_filter.csv"), dtype=str)
pe_valid = set(df_pe["code"].tolist())

df_f = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
df_p = pd.read_csv(os.path.join(OUT, "prices_daily.csv"), dtype={"code": str})
df_p["date"] = pd.to_datetime(df_p["date"])
df_p["close"] = pd.to_numeric(df_p["close"], errors="coerce")

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

df_codes = pd.read_csv(os.path.join(OUT, "all_stock_codes_full.csv"), dtype=str)
code_ind = dict(zip(df_codes["code"], df_codes["industry"]))

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
df_all = df_all[df_all["code"].isin(pe_valid)]
df_all["usable"] = df_all["forward_return"].notna() & (df_all["year"]<=2024)
usable = df_all[df_all["usable"]]

feat = ['roe','gross_margin','net_margin','profit_yoy','asset_turnover','inventory_turnover']
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

print(f"B组PE过滤: {Xtr.shape[0]}训/{Xte.shape[0]}测, {Xtr.shape[1]}维, {usable.code.nunique()}只")

class TM(nn.Module):
    def __init__(self, n_in, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(),
            nn.Linear(h, h//2), nn.ReLU(),
            nn.Linear(h//2, 1)
        )
    def forward(self, x):
        return self.net(x)

# 顿悟实验: 多组配置
configs = [
    # (hidden, lr, weight_decay, epochs, 说明)
    (32,  0.001, 1e-2, 50000, "h=32, 低lr, 强正则, 5万轮"),
    (64,  0.001, 1e-2, 50000, "h=64, 低lr, 强正则, 5万轮"),
    (16,  0.001, 5e-3, 30000, "h=16, 低lr, 中正则, 3万轮"),
    (64,  0.0005,1e-2, 80000, "h=64, 极低lr, 强正则, 8万轮"),
]

results = []
for h, lr, wd, epochs, desc in configs:
    torch.manual_seed(42)
    m = TM(Xtr.shape[1], h)
    opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    
    # 每N轮记录一次
    log_every = max(1, epochs // 20)
    log_epochs = []; log_train_mae = []; log_test_mae = []; log_test_ric = []
    best_test_mae = 1e9; best_state = None; best_epoch = 0
    
    for ep in range(1, epochs+1):
        m.train()
        opt.zero_grad()
        loss = loss_fn(m(Xtr), ytr)
        loss.backward()
        opt.step()
        
        if ep % log_every == 0 or ep == epochs:
            m.eval()
            with torch.no_grad():
                pred_tr = sc_y.inverse_transform(m(Xtr).numpy()).ravel()
                pred_te = sc_y.inverse_transform(m(Xte).numpy()).ravel()
            tr_mae = np.mean(np.abs(y_true[train_m] - pred_tr))
            te_mae = np.mean(np.abs(yte - pred_te))
            te_ric, _ = spearmanr(yte, pred_te)
            log_epochs.append(ep); log_train_mae.append(tr_mae); log_test_mae.append(te_mae); log_test_ric.append(te_ric)
            
            if te_mae < best_test_mae:
                best_test_mae = te_mae
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
                best_epoch = ep
    
    # 找顿悟点: 测试MAE连续5个记录点持续下降
    test_mae_series = np.array(log_test_mae)
    grok_epoch = None
    for i in range(5, len(test_mae_series)):
        if np.all(np.diff(test_mae_series[i-5:i+1]) < 0):
            grok_epoch = log_epochs[i]
            break
    
    # 最终评估
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        pred_final = sc_y.inverse_transform(m(Xte).numpy()).ravel()
    final_mae = np.mean(np.abs(yte - pred_final))
    final_ric, _ = spearmanr(yte, pred_final)
    
    print(f"\n{desc}")
    print(f"  最佳轮次: {best_epoch}, 测试MAE={best_test_mae:.4f}, R_IC={final_ric:+.3f}")
    if grok_epoch:
        print(f"  ⚡ 可能顿悟点: 第{grok_epoch}轮")
    else:
        print(f"  无顿悟迹象")
    
    results.append({
        "desc": desc, "best_epoch": best_epoch,
        "final_mae": final_mae, "final_ric": final_ric,
        "grok_epoch": grok_epoch,
        "log": (log_epochs, log_test_mae, log_test_ric)
    })

# 汇总
naive = np.mean(np.abs(yte - np.mean(y_true[train_m])))
print(f"\n=== 汇总 ===")
print(f"基线(猜均值): {naive:.4f}")
for r in results:
    impr = (1 - r['final_mae']/naive)*100
    grok = f" 顿悟@{r['grok_epoch']}" if r['grok_epoch'] else ""
    print(f"  {r['desc']}: MAE={r['final_mae']:.4f} R_IC={r['final_ric']:+.3f} 改善={impr:+.1f}%{grok}")
