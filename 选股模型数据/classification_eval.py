"""三分类: 下降 / 微涨(0-5%) / 大涨(>5%), 加权损失"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings, os
warnings.filterwarnings("ignore")

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df["usable"]].copy()

# 三分类标签
def classify(ret):
    if ret < 0: return 0       # 下降
    if ret <= 0.05: return 1    # 微涨(0-5%)
    return 2                     # 大涨(>5%)

u["label"] = u["forward_return"].apply(classify)
print(f"类别分布: 下降(0)={(u.label==0).sum()} 微涨(1)={(u.label==1).sum()} 大涨(2)={(u.label==2).sum()}")
print(f"目标均值={u.forward_return.mean()*100:+.1f}% 中位={u.forward_return.median()*100:+.1f}%\n")

# 特征
feat = [c for c in u.columns if c not in
    ["code","year","quarter","forward_return","label","usable","industry"]]
ind = pd.get_dummies(u["industry"], prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1)
X = X.fillna(X.median())
y = u["label"].values

tm = (u["year"] < 2024) | ((u["year"] == 2024) & (u["quarter"] <= 2))
vm = ~tm

sc = StandardScaler()
Xtr = torch.FloatTensor(sc.fit_transform(X[tm]))
Xte = torch.FloatTensor(sc.transform(X[vm]))
ytr = torch.LongTensor(y[tm]); yte = y[vm]

# 类别权重: 下降误判为上涨惩罚×2
class_weights = torch.FloatTensor([2.0, 1.0, 1.0])
print(f"训练:{Xtr.shape[0]} 测试:{Xte.shape[0]} 特征:{Xtr.shape[1]}维")
print(f"类别权重: 下降×2.0 微涨×1.0 大涨×1.0\n")

# === 逻辑回归 ===
lr = LogisticRegression(max_iter=5000, class_weight={0:2,1:1,2:1}, random_state=42)
lr.fit(Xtr.numpy(), y[tm])
pl = lr.predict(Xte.numpy())
cm_lr = confusion_matrix(yte, pl)
acc_lr = np.mean(pl == yte)

# === 随机森林 ===
rf = RandomForestClassifier(100, max_depth=5, min_samples_leaf=10,
    class_weight={0:2,1:1,2:1}, random_state=42)
rf.fit(Xtr.numpy(), y[tm])
pr = rf.predict(Xte.numpy())
cm_rf = confusion_matrix(yte, pr)
acc_rf = np.mean(pr == yte)

# === MLP 分类器 ===
class MLPClassifier(nn.Module):
    def __init__(self, n_in, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h//2, 3)
        )
    def forward(self, x): return self.net(x)

def train_mlp(h, lr=0.001, wd=5e-3, epochs=5000):
    torch.manual_seed(42)
    m = MLPClassifier(Xtr.shape[1], h)
    opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    best_loss, bs = 1e9, None
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = loss_fn(m(Xtr), ytr)
        loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        logits = m(Xte)
        pred = torch.argmax(logits, dim=1).numpy()
    return pred

print("=== MLP ===")
best_acc, best_cm, best_h = 0, None, 0
for h in [8, 16, 32, 64, 128]:
    pred = train_mlp(h)
    cm = confusion_matrix(yte, pred)
    acc = np.mean(pred == yte)
    # 关键指标: 上涨类的召回率
    if cm.shape[0] >= 3:
        up_recall = (cm[1,1] + cm[2,2]) / (cm[1].sum() + cm[2].sum()) if (cm[1].sum()+cm[2].sum())>0 else 0
    else:
        up_recall = 0
    # 下降误判为涨的惩罚: FP_涨 / 下降总数
    if cm.shape[0] >= 3:
        down_fp = (cm[0,1] + cm[0,2]) / cm[0].sum() if cm[0].sum() > 0 else 0
    else:
        down_fp = 0
    n_p = (Xtr.shape[1]*h+h)+(h*h//2+h//2)+(h//2+1)*3
    tag = " <" if acc > best_acc else ""
    print(f"  h={h:3d}: acc={acc:.3f} 涨回={(up_recall*100):.0f}% 降误判跌={(down_fp*100):.0f}%{tag}")
    if acc > best_acc:
        best_acc, best_cm, best_h = acc, cm, h

# === 详细评估 ===
def eval_cm(name, cm, acc):
    n = cm.sum()
    # 三分类详细
    print(f"\n{'='*50}")
    print(f"{name} (准确率={acc:.3f})")
    labels = ["下降", "微涨", "大涨"]
    for i in range(min(3, cm.shape[0])):
        total = cm[i].sum()
        correct = cm[i, i]
        print(f"  {labels[i]}: 真正={correct}/{total} (召回={correct/total*100:.1f}%)")
    if cm.shape[0] >= 3:
        down_wrong = cm[0,1] + cm[0,2]  # 下降被误判为上涨
        up_correct = cm[1,1] + cm[2,2]   # 上涨判对
        up_total = cm[1].sum() + cm[2].sum()
        print(f"  上涨综合召回: {up_correct}/{up_total} ({up_correct/up_total*100:.1f}%)")
        print(f"  下降被误判为涨: {down_wrong}/{cm[0].sum()} ({down_wrong/cm[0].sum()*100:.1f}%) ← 惩罚×2")
    print(f"\n  混淆矩阵:")
    print(f"    实际↓ 预降 预微涨 预大涨")
    for i in range(min(3, cm.shape[0])):
        print(f"    {labels[i]:<6s} {cm[i,0]:>4d} {cm[i,1]:>6d} {cm[i,2]:>6d}")

eval_cm("逻辑回归", cm_lr, acc_lr)
eval_cm("随机森林", cm_rf, acc_rf)
eval_cm("MLP(h={})".format(best_h), best_cm, best_acc)
