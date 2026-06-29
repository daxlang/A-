"""正式: 5折训练+保存权重+逐年混淆矩阵+策略回测"""
import os, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models"); os.makedirs(MDIR, exist_ok=True)
RES = os.path.join(OUT, "final_report.txt")

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
X = pd.concat([u[feat], ind], axis=1)
X = X.fillna(X.median())
y = u.label.values
rets = u.forward_return.values
yrs = u.year.values
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

def train_save_mlp(Xtr, ytr, w4, path):
    sc = StandardScaler()
    Xs = torch.FloatTensor(sc.fit_transform(Xtr))
    ys = torch.LongTensor(ytr)
    m = M5(Xs.shape[1])
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1, 1, 1, w4]))
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

def predict_mlp(Xte, path):
    ckpt = torch.load(path, weights_only=False)
    m = M5(Xte.shape[1])
    m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(Xte))
    with torch.no_grad():
        return torch.argmax(m(Xs), dim=1).numpy()

def evaluate(name, preds, cms):
    buy = pd.DataFrame({"pred": preds, "ret": rets, "year": yrs})
    cm_all = sum(cms)
    acc = sum(cm_all[i,i] for i in range(5)) / cm_all.sum() * 100
    log(f"\n{'='*60}")
    log(f"【{name}】")
    log(f"逐年混淆矩阵:")
    for ti, ty in enumerate(TY):
        cm = cms[ti]; n = cm.sum()
        log(f"  {ty}年({n}只)准确率={(cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4])/n*100:.1f}%")
        for i, nm in enumerate(["暴跌","跌","小涨","中涨","暴涨"]):
            tl = cm[i].sum()
            log(f"    {nm}({tl}):预跌{cm[i,0]}预跌{cm[i,1]}预小{cm[i,2]}预中{cm[i,3]}预暴{cm[i,4]}")
    log(f"  合并({cm_all.sum()}条,准确率{acc:.1f}%):")
    for i, nm in enumerate(["暴跌","跌","小涨","中涨","暴涨"]):
        log(f"    {nm}({cm_all[i].sum()}):{cm_all[i,0]}/{cm_all[i,1]}/{cm_all[i,2]}/{cm_all[i,3]}/{cm_all[i,4]}")

    bought = buy[buy.pred == 4]
    n = len(bought); avg = bought.ret.mean(); med = bought.ret.median()
    win = (bought.ret > 0).mean()*100; exc = avg - rets.mean()
    log(f"  回测: 买{n}次 均{avg*100:+.1f}% 中{med*100:+.1f}% 正{win:.1f}% 超额{exc*100:+.1f}%")
    total = 1.0
    for ty in TY:
        b = buy[buy.year == ty][bought.pred == 4] if False else buy[(buy.year == ty) & (buy.pred == 4)]
        a = buy[buy.year == ty]
        if len(b) > 0:
            total *= 1 + b.ret.mean()
            log(f"    {ty}: 买{len(b)}只 均{b.ret.mean()*100:+.1f}% 超额{(b.ret.mean()-a.ret.mean())*100:+.1f}%")
        else:
            log(f"    {ty}: 未买")
    log(f"  复利: {total:.3f}x")

# === 训练所有模型 ===
log("训练模型并保存...")
# MLP 两类
for w4, tag in [(1.0, "MLP1_0"), (0.5, "MLP0_5")]:
    for ti, ty in enumerate(TY):
        tr = (yrs != ty)
        train_save_mlp(X[tr].values, y[tr], w4, os.path.join(MDIR, f"{tag}_{ty}.pt"))
        log(f"  {tag} fold={ty} done")
# LR + RF
for ti, ty in enumerate(TY):
    tr = (yrs != ty); te = (yrs == ty)
    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr.fit(X[tr].values, y[tr])
    pickle.dump(lr, open(os.path.join(MDIR, f"LR_{ty}.pkl"), "wb"))
    rf = RandomForestClassifier(100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(X[tr].values, y[tr])
    pickle.dump(rf, open(os.path.join(MDIR, f"RF_{ty}.pkl"), "wb"))
    log(f"  LR+RF fold={ty} done")

# === 预测和评估 ===
log("\n评估...")
# MLP
for w4, tag in [(1.0, "MLP1_0"), (0.5, "MLP0_5")]:
    preds, cms = [], []
    for ti, ty in enumerate(TY):
        te = (yrs == ty)
        p = predict_mlp(X[te].values, os.path.join(MDIR, f"{tag}_{ty}.pt"))
        preds.extend(p)
        cms.append(confusion_matrix(y[te], p))
    evaluate(tag, preds, cms)

# LR
preds, cms = [], []
for ti, ty in enumerate(TY):
    te = (yrs == ty)
    lr = pickle.load(open(os.path.join(MDIR, f"LR_{ty}.pkl"), "rb"))
    p = lr.predict(X[te].values)
    preds.extend(p); cms.append(confusion_matrix(y[te], p))
evaluate("LR", preds, cms)

# RF
preds, cms = [], []
for ti, ty in enumerate(TY):
    te = (yrs == ty)
    rf = pickle.load(open(os.path.join(MDIR, f"RF_{ty}.pkl"), "rb"))
    p = rf.predict(X[te].values)
    preds.extend(p); cms.append(confusion_matrix(y[te], p))
evaluate("RF", preds, cms)

log(f"\n模型保存在: {MDIR}")
log("DONE")
