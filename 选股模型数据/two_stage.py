"""两阶段: 激进模型筛掉跌/暴跌 → 保守模型挑暴涨"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
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

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 5)
        )
    def forward(self, x): return self.net(x)

def train_save(Xtr, ytr, weights, path):
    sc = StandardScaler()
    Xs = torch.FloatTensor(sc.fit_transform(Xtr))
    ys = torch.LongTensor(ytr)
    m = M5(Xs.shape[1])
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
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
    with torch.no_grad(): return torch.argmax(m(Xs), dim=1).numpy()

# 激进筛选用: 重罚漏掉跌/暴跌 → [3,3,1,1,0.5]
aggressive_configs = [
    ("A_[3,3,1,1,0.5]", [3.0, 3.0, 1.0, 1.0, 0.5]),
    ("A_[2,2,1,1,0.5]", [2.0, 2.0, 1.0, 1.0, 0.5]),
]

# 第二阶段用已保存的 w=0.5 和 w=0.35
stage2_models = ["MLP0_5", "MLP0_35"]

print("=== 训练激进筛选模型 ===")
for tag, wts in aggressive_configs:
    for ty in TY:
        tr = (yrs != ty)
        train_save(X[tr].values, y[tr], wts, os.path.join(MDIR, f"{tag}_{ty}.pt"))
        # 检查筛掉率
        predict(X[yrs == ty].values, os.path.join(MDIR, f"{tag}_{ty}.pt"))  # just to verify
    print(f"  {tag} done")

print("\n=== 两阶段回测 ===")
for a_tag, _ in aggressive_configs:
    for s2 in stage2_models:
        preds = np.zeros(len(u), dtype=int)
        for ty in TY:
            tr = (yrs != ty); te = (yrs == ty)
            # Stage1: 激进模型筛掉 暴跌(0) 和 跌(1)
            p1 = predict(X[te].values, os.path.join(MDIR, f"{a_tag}_{ty}.pt"))
            keep = (p1 != 0) & (p1 != 1)
            idx_keep = np.where(te)[0][keep]
            # Stage2: 保守模型从筛余里选暴涨
            p2 = predict(X[te].values[keep], os.path.join(MDIR, f"{s2}_{ty}.pt"))
            preds[idx_keep] = p2

        buy = pd.DataFrame({"pred": preds, "ret": rets, "year": yrs})
        bought = buy[buy.pred == 4]
        n_b = len(bought); avg = bought.ret.mean(); med = bought.ret.median()
        win = (bought.ret > 0).mean()*100; exc = avg - rets.mean()
        n_kept = (preds == 4).sum()
        yrly = []
        for ty in TY:
            b = buy[(buy.year == ty) & (buy.pred == 4)]
            yrly.append(b.ret.mean() if len(b) > 0 else 0)

        print(f"{a_tag} + {s2}: 买{n_b}次 均{avg*100:+.1f}% 中{med*100:+.1f}% 正{win:.1f}% 超额{exc*100:+.1f}% 年均{np.mean(yrly)*100:+.1f}%")

# 对比: 纯 w=0.5
print("\n=== 纯 w=0.5 对比 ===")
preds = np.zeros(len(u), dtype=int)
for ty in TY:
    te = (yrs == ty)
    preds[te] = predict(X[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
buy = pd.DataFrame({"pred": preds, "ret": rets, "year": yrs})
bought = buy[buy.pred == 4]
n_b = len(bought); avg = bought.ret.mean(); med = bought.ret.median()
win = (bought.ret > 0).mean()*100
yrly = [buy[(buy.year==ty)&(buy.pred==4)].ret.mean() for ty in TY]
print(f"纯MLP0_5: 买{n_b}次 均{avg*100:+.1f}% 中{med*100:+.1f}% 正{win:.1f}% 年均{np.mean(yrly)*100:+.1f}%")
