"""D14_8逐年明细"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
import warnings; warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals(["sklearn.preprocessing._data.StandardScaler"])
BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
MDIR = os.path.join(BASE, "models")
TY = [2020, 2021, 2022, 2023, 2024]

d8 = pd.read_csv(os.path.join(BASE, "training_full.csv"), dtype={"code": str})
u8 = d8[d8.usable].copy()
u8["gm_pct"] = u8.groupby(["industry","year"])["gross_margin"].transform(lambda x: x.rank(pct=True))
feat8 = [c for c in u8.columns if c not in ["code","year","quarter","forward_return","gm_pct","usable","industry","interest_coverage"]]
ind8 = pd.get_dummies(u8.industry, prefix="ind").astype(float)
X8 = pd.concat([u8[feat8], ind8], axis=1); X8 = X8.fillna(X8.median())
M8 = (u8.cfo_to_revenue >= 0) & (u8.current_ratio >= 0.5) & (u8.gm_pct >= 0.1) & (u8.liability_to_asset <= 0.8)
rets8 = u8.forward_return.values; yrs8 = u8.year.values

class M5(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,5))
    def forward(self, x): return self.net(x)

pr8 = np.zeros(len(u8), dtype=int)
for ty in TY:
    te = (yrs8 == ty)
    ckpt = torch.load(os.path.join(MDIR, f"D14_8_{ty}.pt"), weights_only=False)
    m = M5(X8[te].shape[1]); m.load_state_dict(ckpt["model"]); m.eval()
    Xs = torch.FloatTensor(ckpt["scaler"].transform(X8[te].values))
    with torch.no_grad(): pr8[te] = torch.argmax(m(Xs), dim=1).numpy()

yrly = []
for ty in TY:
    b = (yrs8 == ty) & (pr8 == 4) & M8; n = b.sum(); a = rets8[b]
    yl = a.mean()*100 if n > 0 else 0; yrly.append(yl)
    print(f"{ty}: 买{n}只 均{a.mean()*100:.1f}% 中{np.median(a)*100:.1f}% min{a.min()*100:.1f}% max{a.max()*100:.1f}%" if n > 0 else f"{ty}: 未买")
print(f"年均: {np.mean(yrly):.1f}%")
