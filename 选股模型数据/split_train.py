"""分两组训练: 通过/未通过阶段1, 各自5折MLP"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")

df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df.usable].copy()
def c5(r):
    if r < -0.1: return 0
    if r < 0: return 1
    if r < 0.1: return 2
    if r < 0.3: return 3
    return 4
u["label"] = u.forward_return.apply(c5)
feat = [c for c in u.columns if c not in
    ["code","year","quarter","forward_return","label","usable","industry"]]
ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u.label.values; rets = u.forward_return.values; yrs = u.year.values
TY = [2020, 2021, 2022, 2023, 2024]

# 阶段1分组
PASS = (u.cfo_to_revenue >= 0) & (u.current_ratio >= 0.5)
FAIL = ~PASS
print(f"通过: {PASS.sum()}条 ({PASS.sum()/len(u)*100:.0f}%)  未通过: {FAIL.sum()}条 ({FAIL.sum()/len(u)*100:.0f}%)\n")

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 5)
        )
    def forward(self, x): return self.net(x)

def train_save(Xtr, ytr, path):
    sc = StandardScaler()
    Xs = torch.FloatTensor(sc.fit_transform(Xtr))
    ys = torch.LongTensor(ytr)
    m = M5(Xs.shape[1])
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1, 1, 1, 0.5]))
    bl, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad()
        loss = lfn(m(Xs), ys)
        loss.backward(); opt.step()
        if loss.item() < bl:
            bl = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    torch.save({"model": bs, "scaler": sc}, path)

def predict(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad():
        return torch.argmax(m(Xs), dim=1).numpy()

# 训练两个模型组
for gname, gmask in [("PASS", PASS), ("FAIL", FAIL)]:
    print(f"训练 {gname} 组...")
    for ty in TY:
        tr = (yrs != ty) & gmask
        te = (yrs == ty) & gmask
        if tr.sum() < 50 or te.sum() < 5:
            print(f"  fold {ty}: 训{tr.sum()}条 测{te.sum()}条 → 跳过")
            continue
        train_save(X[tr].values, y[tr], os.path.join(MDIR, f"S1_{gname}_{ty}.pt"))
        print(f"  fold {ty}: 训{tr.sum()} 测{te.sum()} done")

# 预测+回测
print("\n=== 回测 ===")
preds = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty)
    for gname, gmask in [("PASS", PASS), ("FAIL", FAIL)]:
        te_g = te & gmask
        if te_g.sum() == 0: continue
        path = os.path.join(MDIR, f"S1_{gname}_{ty}.pt")
        if not os.path.exists(path): continue
        preds[te_g] = predict(X[te_g].values, path)

buy = pd.DataFrame({"pred": preds, "ret": rets, "year": yrs})
bought = buy[buy.pred == 4]
n = len(bought); avg = bought.ret.mean(); med = bought.ret.median()
win = (bought.ret > 0).mean()*100
yrly = [buy[(buy.year==ty)&(buy.pred==4)].ret.mean() for ty in TY
        if len(buy[(buy.year==ty)&(buy.pred==4)]) > 0]
ayr = np.mean(yrly) if yrly else 0

print(f"分组建模: 买{n}次 均{avg*100:+.1f}% 中{med*100:+.1f}% 正{win:.1f}% 年均{ayr*100:+.1f}%")

# 逐年
for ty in TY:
    b = buy[(buy.year==ty)&(buy.pred==4)]
    n_p = PASS[(yrs==ty) & (preds==4)].sum() if False else 0
    # 算了, 直接用已有的
    if len(b) > 0:
        print(f"  {ty}: 买{len(b)}只 均{b.ret.mean()*100:+.1f}%")
    else:
        print(f"  {ty}: 未买")

print(f"\n(对比 纯MLP0_5: 337次 +20.7% +8.7% 57.3% +19.8%)")
print(f"权重: models/S1_PASS_*.pt + S1_FAIL_*.pt 共10个")
