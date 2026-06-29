"""纯评估: 读已保存模型, 逐年混淆矩阵 + 策略回测"""
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
# torch 2.6 fix
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
RES = os.path.join(OUT, "final_report2.txt")

def log(msg):
    print(msg, flush=True)
    with open(RES, "a", encoding="utf-8") as f: f.write(msg + "\n")

with open(RES, "w", encoding="utf-8") as f: f.write("")

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
X = pd.concat([u[feat], ind], axis=1).fillna(X.median()) if "X" in dir() else pd.concat([u[feat], ind], axis=1).fillna(pd.concat([u[feat], ind], axis=1).median())
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

def predict_mlp(Xte, path):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1])
    m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad():
        return torch.argmax(m(Xs), dim=1).numpy()

def evaluate(name, preds):
    buy = pd.DataFrame({"pred": preds, "ret": rets, "year": yrs})
    cm_all = confusion_matrix(y, preds)
    acc = sum(cm_all[i,i] for i in range(5)) / cm_all.sum() * 100

    log(f"\n{'='*60}")
    log(f"【{name}】")
    log(f"\n逐年混淆矩阵:")
    for ty in TY:
        te = (yrs == ty)
        cm = confusion_matrix(y[te], preds[te])
        n = cm.sum()
        acc_y = sum(cm[i,i] for i in range(5)) / n * 100
        log(f"  {ty}年({n}只): 准确率={acc_y:.1f}%")
        for i, nm in enumerate(["暴跌(< -10%)","跌(-10~0)","小涨(0~10%)","中涨(10~30%)","暴涨(>30%)"]):
            tl = cm[i].sum()
            corr = cm[i,i]
            log(f"    {nm}: {corr}/{tl} 召回={corr/tl*100:.0f}%  (预暴跌{cm[i,0]} 预跌{cm[i,1]} 预小涨{cm[i,2]} 预中涨{cm[i,3]} 预暴涨{cm[i,4]})")

    log(f"\n  合并混淆矩阵 ({cm_all.sum()}条, 总准确率={acc:.1f}%):")
    log(f"  实际->预测 |暴跌 跌 小涨 中涨 暴涨| 合计")
    for i, nm in enumerate(["暴跌","跌","小涨","中涨","暴涨"]):
        log(f"    {nm:6s}|{cm_all[i,0]:>4d} {cm_all[i,1]:>3d} {cm_all[i,2]:>4d} {cm_all[i,3]:>4d} {cm_all[i,4]:>4d}| {cm_all[i].sum():>4d}")

    # 策略: 只买暴涨组
    bought = buy[buy.pred == 4]
    n = len(bought); avg = bought.ret.mean(); med = bought.ret.median()
    win = (bought.ret > 0).mean()*100; exc = avg - rets.mean()
    log(f"\n  策略(只买暴涨): {n}次 全市场均{rets.mean()*100:+.1f}%")
    log(f"  均值{avg*100:+.1f}% 中位{med*100:+.1f}% 正收益{win:.1f}% 超额{exc*100:+.1f}%")
    log(f"  逐年:")
    total = 1.0
    for ty in TY:
        b = buy[(buy.year == ty) & (buy.pred == 4)]
        a = buy[buy.year == ty]
        if len(b) > 0:
            ex = b.ret.mean() - a.ret.mean()
            total *= 1 + b.ret.mean()
            log(f"    {ty}: 全{a.ret.mean()*100:+.1f}% 买{len(b)}只 均{b.ret.mean()*100:+.1f}% 超额{ex*100:+.1f}%")
        else:
            log(f"    {ty}: 未买")
    log(f"  5年复利: {total:.3f}x")

# 先确保X正确
X = pd.concat([u[feat], ind], axis=1)
X = X.fillna(X.median())

# MLP 1.0
preds = np.zeros(len(u), dtype=int)
for ti, ty in enumerate(TY):
    te = (yrs == ty)
    p = predict_mlp(X[te].values, os.path.join(MDIR, f"MLP1_0_{ty}.pt"))
    preds[te] = p
evaluate("MLP 无惩罚(w=1.0)", preds)

# MLP 0.5
preds = np.zeros(len(u), dtype=int)
for ti, ty in enumerate(TY):
    te = (yrs == ty)
    p = predict_mlp(X[te].values, os.path.join(MDIR, f"MLP0_5_{ty}.pt"))
    preds[te] = p
evaluate("MLP 保守(w=0.5)", preds)

# LR
preds = np.zeros(len(u), dtype=int)
for ti, ty in enumerate(TY):
    te = (yrs == ty)
    lr = pickle.load(open(os.path.join(MDIR, f"LR_{ty}.pkl"), "rb"))
    preds[te] = lr.predict(X[te].values)
evaluate("逻辑回归", preds)

# RF
preds = np.zeros(len(u), dtype=int)
for ti, ty in enumerate(TY):
    te = (yrs == ty)
    rf = pickle.load(open(os.path.join(MDIR, f"RF_{ty}.pkl"), "rb"))
    preds[te] = rf.predict(X[te].values)
evaluate("随机森林", preds)

log("\nDONE")
