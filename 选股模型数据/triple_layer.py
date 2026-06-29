"""三层MLP: 256→128→64 及变体"""
import os,pandas as pd,numpy as np,torch,torch.nn as nn,torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings; warnings.filterwarnings("ignore")
OUT=r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\history_data"
df=pd.read_csv(os.path.join(OUT,"training_final.csv"),dtype={"code":str})
u=df[df.usable].copy(); u["label"]=u.forward_return.apply(lambda r:0 if r<0 else (1 if r<=0.05 else 2))
feat=[c for c in u.columns if c not in ["code","year","quarter","forward_return","label","usable","industry"]]
ind=pd.get_dummies(u.industry,prefix="ind").astype(float)
X=pd.concat([u[feat],ind],axis=1); X=X.fillna(X.median())
y=u.label.values; yrs=u.year.values

class M3(nn.Module):
    def __init__(self,n_in,h):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_in,h),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(h//2,h//4),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(h//4,3))
    def forward(self,x): return self.net(x)

def train(Xtr_np,ytr_np,Xte_np,h):
    sc=StandardScaler()
    Xtr=torch.FloatTensor(sc.fit_transform(Xtr_np)); Xte=torch.FloatTensor(sc.transform(Xte_np))
    ytr=torch.LongTensor(ytr_np)
    m=M3(Xtr.shape[1],h); opt=optim.AdamW(m.parameters(),lr=0.001,weight_decay=5e-3)
    lfn=nn.CrossEntropyLoss(weight=torch.FloatTensor([3.0,1.0,1.0]))
    bl,bs=1e9,None
    for _ in range(5000):
        m.train(); opt.zero_grad()
        loss = lfn(m(Xtr), ytr)
        loss.backward(); opt.step()
        if loss.item() < bl:
            bl = loss.item()
            bs = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): return torch.argmax(m(Xte),dim=1).numpy()

for h,desc in [(128,"128→64→32"),(256,"256→128→64"),(384,"384→192→96")]:
    preds=[]; trues=[]
    for ty in [2020,2021,2022,2023,2024]:
        tr=(yrs!=ty); te=(yrs==ty)
        p=train(X[tr].values,y[tr],X[te].values,h)
        preds.extend(p); trues.extend(y[te])
    cm=confusion_matrix(trues,preds)
    n_p=(19*h+h)+(h*h//2+h//2)+(h//2*h//4+h//4)+(h//4*3)
    acc=(cm[0,0]+cm[1,1]+cm[2,2])/cm.sum()*100
    r0=cm[0,0]/cm[0].sum()*100; r2=cm[2,2]/cm[2].sum()*100
    dfp=(cm[0,1]+cm[0,2])/cm[0].sum()*100
    print(f"{desc} {n_p:>7d}参: acc={acc:.1f}% 降回={r0:.1f}% 大回={r2:.1f}% 踩雷={dfp:.1f}%")
    print(f"  混淆: 降{cm[0,0]}/{cm[0,1]}/{cm[0,2]}  微{cm[1,0]}/{cm[1,1]}/{cm[1,2]}  涨{cm[2,0]}/{cm[2,1]}/{cm[2,2]}")
print("\n基线(128→64两层): 47.2% 72.8% 28.7% 27.2%")
