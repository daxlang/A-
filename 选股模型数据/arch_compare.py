"""架构对比: MLP vs XGBoost vs ResMLP vs Wide, 惩罚×3, 5折CV"""
import os, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"

df = pd.read_csv(os.path.join(OUT, "training_final.csv"), dtype={"code": str})
u = df[df["usable"]].copy()
u["label"] = u["forward_return"].apply(lambda r: 0 if r < 0 else (1 if r <= 0.05 else 2))

feat = [c for c in u.columns if c not in
    ["code","year","quarter","forward_return","label","usable","industry"]]
ind = pd.get_dummies(u["industry"], prefix="ind").astype(float)
X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
y = u["label"].values; yrs = u["year"].values

W = torch.FloatTensor([3.0, 1.0, 1.0])  # ×3惩罚
test_years = [2020, 2021, 2022, 2023, 2024]

# === 架构定义 ===
class MLP_Baseline(nn.Module):
    """当前架构: 128→64"""
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    def forward(self, x): return self.net(x)

class MLP_Residual(nn.Module):
    """残差MLP: 64→64→64 with skip"""
    def __init__(self, n_in):
        super().__init__()
        self.proj = nn.Linear(n_in, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.proj(x))))
        r1 = x
        x = self.dropout(torch.relu(self.bn2(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc2(x) + r1)))  # skip
        x = self.fc3(x)
        return self.out(x)

class MLP_Wide(nn.Module):
    """宽单层: 256"""
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 3)
        )
    def forward(self, x): return self.net(x)

class MLP_Deep(nn.Module):
    """深层: 64→48→32→24"""
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 48), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(48, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 24), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(24, 3)
        )
    def forward(self, x): return self.net(x)

def train_nn(model_class, Xtr_np, ytr_np, Xte_np):
    sc = StandardScaler()
    Xtr = torch.FloatTensor(sc.fit_transform(Xtr_np))
    Xte = torch.FloatTensor(sc.transform(Xte_np))
    ytr = torch.LongTensor(ytr_np)
    m = model_class(Xtr.shape[1])
    opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    loss_fn = nn.CrossEntropyLoss(weight=W)
    best_loss, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad()
        loss = loss_fn(m(Xtr), ytr); loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): return torch.argmax(m(Xte), dim=1).numpy()

def eval_cv(model_class_or_name, use_xgb=False):
    preds = []; trues = []
    for ty in test_years:
        tr = (yrs != ty); te = (yrs == ty)
        if use_xgb:
            m = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                objective='multi:softmax', num_class=3, random_state=42,
                scale_pos_weight=3.0)  # 给class 0更高权重
            m.fit(X[tr].values, y[tr])
            p = m.predict(X[te].values)
        else:
            p = train_nn(model_class_or_name, X[tr].values, y[tr], X[te].values)
        preds.extend(p); trues.extend(y[te])
    cm = confusion_matrix(trues, preds)
    n = cm.sum()
    acc = (cm[0,0]+cm[1,1]+cm[2,2])/n*100
    r0 = cm[0,0]/cm[0].sum()*100 if cm[0].sum()>0 else 0
    r2 = cm[2,2]/cm[2].sum()*100 if cm.shape[0]>2 else 0
    dfp = (cm[0,1]+cm[0,2])/cm[0].sum()*100
    n_p = sum(p.numel() for p in model_class_or_name(n_in=19).parameters()) if not use_xgb else 0
    return {"acc": acc, "r0": r0, "r2": r2, "dfp": dfp, "params": n_p}

# === 跑 ===
print(f"{'='*60}")
print(f"架构对比 (惩罚×3, 5折CV)")
print(f"{'架构':<12s} {'参数':>6s} {'准确率':>7s} {'降召回':>7s} {'大涨召回':>7s} {'踩雷率':>7s}")
print(f"{'-'*50}")

archs = [
    ("Baseline", MLP_Baseline, False),
    ("Residual", MLP_Residual, False),
    ("Wide", MLP_Wide, False),
    ("Deep", MLP_Deep, False),
]
if HAS_XGB:
    archs.append(("XGBoost", None, True))
else:
    print("(XGBoost未安装, 跳过)\n")

for name, arch, xgb_flag in archs:
    r = eval_cv(arch, use_xgb=xgb_flag)
    print(f"{name:<12s} {r['params']:>6d} {r['acc']:>6.1f}% {r['r0']:>6.1f}% {r['r2']:>6.1f}% {r['dfp']:>6.1f}%")
