"""2年期持有: 三分类 + 滚动交叉验证"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

df_f = pd.read_csv(os.path.join(OUT, "financials_history.csv"), dtype={"code": str})
dc = pd.read_csv(os.path.join(OUT, "all_stock_codes_full.csv"), dtype=str)
ci = dict(zip(dc.code, dc.industry))
df_p = pd.read_csv(os.path.join(OUT, "prices_daily.csv"), dtype={"code": str})
df_p["date"] = pd.to_datetime(df_p["date"]); df_p["close"] = pd.to_numeric(df_p["close"], errors="coerce")

pm = {}
for c, g in df_p.groupby("code"):
    g = g.dropna(subset=["close"]).sort_values("date"); pm[c] = g.set_index("date")["close"]

def nearest(code, target):
    if code not in pm: return np.nan
    ts = pd.Timestamp(target); s = pm[code]; m = s.index >= ts
    return float(s[m].iloc[0]) if m.any() else float(s.iloc[-1])

def buy2(y, q):
    if q == 1: return f"{y}-04-30"
    if q == 2: return f"{y}-07-31"
    if q == 3: return f"{y}-10-31"
    return f"{y+1}-04-30"

def sell2(y, q):
    if q == 1: return f"{y+2}-04-30"
    if q == 2: return f"{y+2}-07-31"
    if q == 3: return f"{y+2}-10-31"
    return f"{y+3}-04-30"

# 建2年期训练集
feat_cols = [c for c in df_f.columns if c not in ["code","year","quarter","industry"]]
feat_cols = [c for c in feat_cols if df_f[c].notna().sum() > len(df_f)*0.3]

rows = []
for _, r in df_f.iterrows():
    c, y, q = r["code"], int(r["year"]), int(r["quarter"])
    bd = buy2(y, q); sd = sell2(y, q)
    pb_v = nearest(c, bd); ps_v = nearest(c, sd)
    ret = (ps_v - pb_v) / pb_v if (pd.notna(pb_v) and pd.notna(ps_v) and pb_v > 0) else np.nan
    row = {"code": c, "year": y, "quarter": q, "forward_return": ret, "industry": ci.get(c, "")}
    for col in feat_cols: row[col] = r.get(col, np.nan)
    rows.append(row)

df = pd.DataFrame(rows)
df["usable"] = df["forward_return"].notna() & (df["year"] <= 2023)
u = df[df["usable"]].copy()

def cls(ret):
    if ret < 0: return 0
    if ret <= 0.05: return 1
    return 2
u["label"] = u["forward_return"].apply(cls)

# 特征填充
for col in feat_cols:
    if u[col].isna().sum() > 0:
        u[col] = u[col].fillna(u.groupby(["industry","year","quarter"])[col].transform("mean"))
        u[col] = u[col].fillna(u.groupby(["industry","year"])[col].transform("mean"))
        u[col] = u[col].fillna(u.groupby("industry")[col].transform("mean"))
        u[col] = u[col].fillna(u[col].mean())

# 特征矩阵
ind = pd.get_dummies(u["industry"], prefix="ind").astype(float)
feat_use = [c for c in feat_cols if c in u.columns]
X_all = pd.concat([u[feat_use], ind], axis=1)
X_all = X_all.fillna(X_all.median())
y_all = u["label"].values
years = u["year"].values

print(f"类别分布: 下降={(y_all==0).sum()} 微涨={(y_all==1).sum()} 大涨={(y_all==2).sum()}")
print(f"目标均值={u.forward_return.mean()*100:+.1f}% 中位={u.forward_return.median()*100:+.1f}%\n")

class MLP3(nn.Module):
    def __init__(self, n_in, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h//2, 3)
        )
    def forward(self, x): return self.net(x)

def train_mlp(Xtr_np, ytr_np, Xte_np):
    sc = StandardScaler()
    Xtr = torch.FloatTensor(sc.fit_transform(Xtr_np)); Xte = torch.FloatTensor(sc.transform(Xte_np))
    ytr = torch.LongTensor(ytr_np)
    m = MLP3(Xtr.shape[1], 128)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([2.0, 1.0, 1.0]))
    best_loss, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad()
        loss = loss_fn(m(Xtr), ytr); loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): return torch.argmax(m(Xte), dim=1).numpy()

print("=== 2年期持有 滚动交叉验证 ===\n")
test_years = [2020, 2021, 2022, 2023]

for ty in test_years:
    tr = (years != ty); te = (years == ty)
    pred = train_mlp(X_all[tr].values, y_all[tr], X_all[te].values)
    cm = confusion_matrix(y_all[te], pred)
    n = cm.sum()
    acc = (cm[0,0]+cm[1,1]+cm[2,2])/n*100
    r0 = cm[0,0]/cm[0].sum()*100 if cm[0].sum()>0 else 0
    r2 = cm[2,2]/cm[2].sum()*100 if cm.shape[0]>2 else 0
    dfp = (cm[0,1]+cm[0,2])/cm[0].sum()*100 if cm[0].sum()>0 else 0
    
    print(f"【{ty}年】 训练{X_all[tr].shape[0]}条 测试{n}条 准确率={acc:.1f}%")
    print(f"        预测:降  微涨  大涨")
    lbs = ["实际下降", "实际微涨", "实际大涨"]
    for i in range(3):
        print(f"  {lbs[i]:<8s} {cm[i,0]:>4d}  {cm[i,1]:>3d}  {cm[i,2]:>4d}")
    print(f"  下降误判为涨: {dfp:.1f}%  大涨召回: {r2:.1f}%\n")
