"""随机80/20切分训练: 1权重/数据集, 3个数据集"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
OUT = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(OUT, "models")
LAB = ["暴跌","跌","小涨","中涨","暴涨"]
SEED = 42

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

def run_exp(name, csv_file, wtag, mask_func, use_ic=False):
    df = pd.read_csv(os.path.join(OUT, csv_file), dtype={"code": str})
    u = df[df.usable].copy()
    u["gm_pct"] = u.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
    u["L"] = u.forward_return.apply(lambda r: 0 if r < -0.1 else (1 if r < 0 else (2 if r < 0.1 else (3 if r < 0.3 else 4))))
    
    DROP = [] if use_ic else ["interest_coverage"]
    feat = [c for c in u.columns if c not in ["code","year","quarter","forward_return","gm_pct","L","usable","industry"] + DROP]
    ind = pd.get_dummies(u.industry, prefix="ind").astype(float)
    X = pd.concat([u[feat], ind], axis=1); X = X.fillna(X.median())
    y = u.L.values; rets = u.forward_return.values
    
    # 80/20随机切分
    np.random.seed(SEED)
    n = len(u); idx = np.random.permutation(n)
    tr_idx = idx[:int(n*0.8)]; te_idx = idx[int(n*0.8):]
    print(f"\n{'='*60}")
    print(f"{name}: {X.shape[1]}维(14+{ind.shape[1]}行业) 训{len(tr_idx)}条 测{len(te_idx)}条")
    
    # 训练
    sc = StandardScaler(); Xs = torch.FloatTensor(sc.fit_transform(X.iloc[tr_idx].values))
    ys = torch.LongTensor(y[tr_idx])
    m = M5(Xs.shape[1]); opt = optim.AdamW(m.parameters(), lr=0.001, weight_decay=5e-3)
    lfn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,0.5]))
    bl, bs = 1e9, None
    for _ in range(5000):
        m.train(); opt.zero_grad(); loss = lfn(m(Xs), ys); loss.backward(); opt.step()
        if loss.item() < bl: bl = loss.item(); bs = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval(); torch.save({"model":bs,"scaler":sc}, os.path.join(MDIR, f"{wtag}.pt"))
    print(f"  {wtag}.pt saved")
    
    # 预测
    Xte = torch.FloatTensor(sc.transform(X.iloc[te_idx].values))
    with torch.no_grad(): pr_te = torch.argmax(m(Xte), dim=1).numpy()
    
    # 掩码过滤
    y_te = y[te_idx]; rets_te = rets[te_idx]
    msk = mask_func(u.iloc[te_idx]).values if callable(mask_func) else mask_func.iloc[te_idx].values
    
    cm = confusion_matrix(y_te[msk], pr_te[msk]); n = cm.sum()
    acc = sum(cm[i,i] for i in range(5)) / n * 100
    print(f"=== {wtag} → 测试集掩码 ({n}条 acc={acc:.1f}%) ===")
    for i, lb in enumerate(LAB):
        print(f"  {lb:6s}|{cm[i,0]:>5d} {cm[i,1]:>4d} {cm[i,2]:>5d} {cm[i,3]:>5d} {cm[i,4]:>5d}|{cm[i].sum():>5d}")
    buy = (pr_te == 4) & msk; nb = buy.sum(); br = rets_te[buy]
    print(f"买入: {nb}只 均{br.mean()*100:.1f}% 暴涨召回{cm[4,4]}/{cm[4].sum()}")
    
    # 逐年(用year列分组)
    yrs_te = u.iloc[te_idx]["year"].values
    yr_returns = []
    for ty in sorted(set(yrs_te)):
        b = (yrs_te == ty) & buy; n_b = b.sum(); a = rets_te[b]
        yl = a.mean()*100 if n_b > 0 else 0; yr_returns.append(yl)
        print(f"  {ty}: {n_b}只 {a.mean()*100:.1f}%" if n_b > 0 else f"  {ty}: 未买")
    print(f"  年均: {np.mean(yr_returns):.1f}%")
    return np.mean(yr_returns)

# 定义掩码函数
def mask_4(u_part): return (u_part.cfo_to_revenue >= 0) & (u_part.current_ratio >= 0.5) & (u_part.gm_pct >= 0.1)
def mask_8(u_part): return mask_4(u_part) & (u_part.liability_to_asset <= 0.8)

# 三组实验: 4行业, 8行业, 全行业
# 4行业 → 去掉IC (D14结论: 4行业丢IC亏2.5pp, 但公平对比用相同特征)
r4 = run_exp("4行业(去IC)", "training_final.csv", "RAND_4", mask_4, use_ic=False)
r8 = run_exp("8行业(去IC)", "training_full.csv", "RAND_8", mask_8, use_ic=False)
ra = run_exp("全行业(去IC)", "training_all.csv", "RAND_ALL", mask_8, use_ic=False)

print(f"\n{'='*60}")
print(f"汇总: RAND_4={r4:.1f}%  RAND_8={r8:.1f}%  RAND_ALL={ra:.1f}%")
print(f"原版(5折): MLP0_5=+21.8%  D14_8=+25.6%  ALL_14=+35.8%")
