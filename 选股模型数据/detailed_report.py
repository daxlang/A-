"""详细评估报告: 全部指标"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import warnings; warnings.filterwarnings("ignore")

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df["usable"]].copy()

def cls(ret):
    if ret < 0: return 0
    if ret <= 0.05: return 1
    return 2
u["label"] = u["forward_return"].apply(cls)

feat = [c for c in u.columns if c not in
    ["code","year","quarter","forward_return","label","usable","industry"]]
ind = pd.get_dummies(u["industry"], prefix="ind").astype(float)
X_all = pd.concat([u[feat], ind], axis=1).fillna(X_all.median()) if False else pd.concat([u[feat], ind], axis=1).fillna(pd.concat([u[feat], ind], axis=1).median())
X_all = pd.concat([u[feat], ind], axis=1)
X_all = X_all.fillna(X_all.median())
y_all = u["label"].values
years = u["year"].values

class MLP3(nn.Module):
    def __init__(self, n_in, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h//2, 3)
        )
    def forward(self, x): return self.net(x)

def train_mlp(Xtr_np, ytr_np, Xte_np, use_weights=True):
    sc = StandardScaler()
    Xtr = torch.FloatTensor(sc.fit_transform(Xtr_np)); Xte = torch.FloatTensor(sc.transform(Xte_np))
    ytr = torch.LongTensor(ytr_np)
    w = torch.FloatTensor([2.0, 1.0, 1.0]) if use_weights else None
    m = MLP3(Xtr.shape[1], 128)
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    best_loss, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad()
        loss = loss_fn(m(Xtr), ytr); loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): return torch.argmax(m(Xte), dim=1).numpy()

def detailed_eval(label, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n = cm.sum()
    acc = (cm[0,0]+cm[1,1]+cm[2,2]) / n
    labels = ["下降", "微涨", "大涨"]
    
    print(f"\n## {label}")
    print(f"\n### 混淆矩阵 (行=实际, 列=预测)")
    print(f"| 实际↓预测 | 下降 | 微涨 | 大涨 | 合计 |")
    print(f"|---|---|---|---|---|")
    for i, lb in enumerate(labels):
        print(f"| {lb} | {cm[i,0]} | {cm[i,1]} | {cm[i,2]} | {cm[i].sum()} |")
    print(f"| 合计 | {cm[:,0].sum()} | {cm[:,1].sum()} | {cm[:,2].sum()} | {n} |")
    
    print(f"\n### 各类指标")
    print(f"| 类别 | 精确率 | 召回率 | F1 | 样本数 |")
    print(f"|---|---|---|---|---|")
    for i, lb in enumerate(labels):
        tp = cm[i,i]; gt = cm[i].sum(); pred_total = cm[:,i].sum()
        prec = tp/pred_total*100 if pred_total > 0 else 0
        rec = tp/gt*100 if gt > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
        print(f"| {lb} | {prec:.1f}% | {rec:.1f}% | {f1:.1f}% | {gt} |")
    
    # 特殊指标
    down_fp = (cm[0,1]+cm[0,2])/cm[0].sum()*100 if cm[0].sum()>0 else 0
    up_recall = (cm[1,1]+cm[2,2])/(cm[1].sum()+cm[2].sum())*100 if cm[1].sum()+cm[2].sum()>0 else 0
    
    print(f"\n### 实用指标")
    print(f"| 指标 | 值 |")
    print(f"|---|---|")
    print(f"| 总体准确率 | {acc*100:.1f}% |")
    print(f"| 下降误判为上涨率 | {down_fp:.1f}% |")
    print(f"| 上涨综合召回率 | {up_recall:.1f}% |")
    print(f"| 大涨精确率(预测大涨中实际的确大涨) | {cm[2,2]/cm[:,2].sum()*100:.1f}% |")
    
    return {"cm": cm, "acc": acc, "down_fp": down_fp, "up_recall": up_recall}

print("# 五年滚动交叉验证详细报告\n")
print(f"**训练集:** 每年4年数据(约2600条), **测试集:** 轮流1年(约520条)")
print(f"**模型:** MLP(h=128), 19维特征, 三分类(下降/微涨/大涨)")
print(f"**类别分布(全量3120条):** 下降1458(46.7%) 微涨286(9.2%) 大涨1376(44.1%)\n")

test_years = [2020, 2021, 2022, 2023, 2024]

for label, use_w in [("有惩罚(下降误判惩罚×2)", True), ("无惩罚(各类平等)", False)]:
    all_preds = []; all_trues = []
    print(f"{'='*60}")
    print(f"## {label}\n")
    
    print("### 逐年结果")
    print(f"| 年份 | 训练 | 测试 | 准确率 | 降召回 | 降精确 | 微召回 | 微精确 | 大涨召回 | 大涨精确 | 降误判涨 |")
    print(f"|------|------|------|--------|--------|--------|--------|--------|----------|----------|----------|")
    
    for ty in test_years:
        tr = (years != ty); te = (years == ty)
        pred = train_mlp(X_all[tr].values, y_all[tr], X_all[te].values, use_weights=use_w)
        cm = confusion_matrix(y_all[te], pred); n = cm.sum()
        acc = (cm[0,0]+cm[1,1]+cm[2,2])/n*100
        r0 = cm[0,0]/cm[0].sum()*100 if cm[0].sum()>0 else 0
        p0 = cm[0,0]/cm[:,0].sum()*100 if cm[:,0].sum()>0 else 0
        r1 = cm[1,1]/cm[1].sum()*100 if cm.shape[0]>1 and cm[1].sum()>0 else 0
        p1 = cm[1,1]/cm[:,1].sum()*100 if cm[:,1].sum()>0 else 0
        r2 = cm[2,2]/cm[2].sum()*100 if cm.shape[0]>2 and cm[2].sum()>0 else 0
        p2 = cm[2,2]/cm[:,2].sum()*100 if cm[:,2].sum()>0 else 0
        dfp = (cm[0,1]+cm[0,2])/cm[0].sum()*100 if cm[0].sum()>0 else 0
        print(f"| {ty} | {X_all[tr].shape[0]} | {n} | {acc:.1f}% | {r0:.1f}% | {p0:.1f}% | {r1:.1f}% | {p1:.1f}% | {r2:.1f}% | {p2:.1f}% | {dfp:.1f}% |")
        all_preds.extend(pred); all_trues.extend(y_all[te])
    
    detailed_eval(f"五年合并混淆矩阵 — {label}", np.array(all_trues), np.array(all_preds))
