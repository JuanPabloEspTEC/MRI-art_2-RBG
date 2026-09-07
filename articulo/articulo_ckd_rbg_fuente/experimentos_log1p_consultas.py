import numpy as np, pandas as pd
from scipy import stats, optimize
rng=np.random.default_rng(42)
d=pd.read_csv('notebook/data/processed/ckd_imputado.csv')
raw=pd.read_csv('data/raw/ckd.csv')
cls=(raw['classification'].astype(str).str.strip().str.lower()=='ckd').astype(int).values
V=list(d.columns); n=len(d); logn=np.log(n)
DAG2=[("age","bp"),("age","bgr"),("bp","al"),("bp","sg"),("bgr","al"),("bgr","sg"),("al","sc"),
      ("al","bu"),("sg","sc"),("sg","bu"),("sc","sod"),("sc","pot"),("sc","hemo"),("bu","sod"),
      ("bu","pot"),("bu","hemo"),("hemo","pcv"),("hemo","rc"),("sc","wc")]
PA={v:[a for a,b in DAG2 if b==v] for v in V}
ORDER=["age","bp","bgr","sg","al","bu","sc","sod","pot","hemo","wc","pcv","rc"]
def ols(y,Xp):
    M=np.column_stack([np.ones(len(y))]+list(Xp)) if Xp else np.ones((len(y),1))
    b,*_=np.linalg.lstsq(M,y,rcond=None); return b,y-M@b
T=lambda x: np.log1p(x); Ti=lambda z: np.expm1(z); J=lambda x: -np.log1p(x).sum()

def build(strat, transform):
    P={}
    for v in V:
        X=T(d[v].values.astype(float)) if transform else d[v].values.astype(float)
        Pp=[(T(d[p].values.astype(float)) if transform else d[p].values.astype(float)) for p in PA[v]]
        P[v]={}
        for g in ([None] if not strat else [0,1]):
            m=np.ones(n,bool) if g is None else cls==g
            b,r=ols(X[m],[c[m] for c in Pp]); P[v][g]=(b,np.sqrt((r**2).mean()))
    return P
def score(strat,transform):
    ll=0.0; npar=0
    for v in V:
        X=T(d[v].values.astype(float)) if transform else d[v].values.astype(float)
        Pp=[(T(d[p].values.astype(float)) if transform else d[p].values.astype(float)) for p in PA[v]]
        for g in ([None] if not strat else [0,1]):
            m=np.ones(n,bool) if g is None else cls==g
            b,r=ols(X[m],[c[m] for c in Pp]); s=np.sqrt((r**2).mean())
            ll+=stats.norm.logpdf(r,0,s).sum(); npar+=len(PA[v])+2
    if transform: ll+=sum(J(d[v].values.astype(float)) for v in V)   # lambda fija: 0 parametros extra
    return ll,npar,ll-0.5*logn*npar,ll-npar
def cv(strat,transform,K=5):
    idx=np.arange(n); np.random.default_rng(7).shuffle(idx); folds=np.array_split(idx,K); tot=0.0
    for k in range(K):
        te=folds[k]; tr=np.concatenate([folds[j] for j in range(K) if j!=k])
        dtr=d.iloc[tr]; dte=d.iloc[te]; gtr=cls[tr]; gte=cls[te]; ll=0.0
        f=(lambda a:T(a)) if transform else (lambda a:a)
        for v in V:
            for g in ([None] if not strat else [0,1]):
                mtr=np.ones(len(tr),bool) if g is None else gtr==g
                mte=np.ones(len(te),bool) if g is None else gte==g
                if mte.sum()==0: continue
                b,r=ols(f(dtr[v].values[mtr]),[f(dtr[p].values[mtr]) for p in PA[v]])
                M=np.column_stack([np.ones(mte.sum())]+[f(dte[p].values[mte]) for p in PA[v]])
                rte=f(dte[v].values[mte])-M@b; s=max(np.sqrt((r**2).mean()),1e-9)
                ll+=stats.norm.logpdf(rte,0,s).sum()
        if transform: ll+=sum(J(dte[v].values.astype(float)) for v in V)
        tot+=ll
    return tot
for nm,(st,tf) in {"L0 gaussiana":(False,False),"L1 log1p":(False,True),
                   "L2 estratificada":(True,False),"L3 log1p + estratificada":(True,True)}.items():
    ll,p,b,a=score(st,tf); print(f"{nm:26s} par={p:3d} loglik={ll:10.1f} BIC={b:10.1f} AIC={a:10.1f} CV={cv(st,tf):10.1f}")

def sample(P,strat,transform,N=400000):
    g=rng.random(N)<cls.mean() if strat else None; S={}
    for v in ORDER:
        z=np.empty(N)
        for grp in ([None] if not strat else [0,1]):
            m=np.ones(N,bool) if grp is None else (g==(grp==1))
            b,s=P[v][grp]; mu=b[0]+sum(b[i+1]*S[p][m] for i,p in enumerate(PA[v]))
            z[m]=mu+rng.normal(0,s,m.sum())
        S[v]=z
    return {v:Ti(S[v]) for v in V} if transform else S
obs={'C1':(d.sc[d.bp>=90]>3).mean(),'C2':d.pcv[d.hemo<10].mean(),'C3':d.hemo[d.sc>5].mean(),
     'C4':(d.pot[d.bu>100]>5.5).mean(),'C5a':d.al[d.bgr<140].mean(),'C5b':d.al[d.bgr>200].mean()}
def resp(S):
    return {'C1':(S['sc'][S['bp']>=90]>3).mean(),'C2':S['pcv'][S['hemo']<10].mean(),
            'C3':S['hemo'][S['sc']>5].mean(),'C4':(S['pot'][S['bu']>100]>5.5).mean(),
            'C5a':S['al'][S['bgr']<140].mean(),'C5b':S['al'][S['bgr']>200].mean()}
r0=resp(sample(build(False,False),False,False)); r3=resp(sample(build(True,True),True,True))
print(f"\n{'':5s} {'observado':>10s} {'M0':>9s} {'L3':>9s} {'errM0':>8s} {'errL3':>8s}")
e0=[];e3=[]
for k in ['C1','C2','C3','C4','C5a','C5b']:
    a=abs(r0[k]-obs[k])/abs(obs[k])*100; b_=abs(r3[k]-obs[k])/abs(obs[k])*100; e0.append(a); e3.append(b_)
    print(f"{k:5s} {obs[k]:10.3f} {r0[k]:9.3f} {r3[k]:9.3f} {a:7.1f}% {b_:7.1f}%")
print(f"\nerror relativo medio: M0={np.mean(e0):.1f}%  L3={np.mean(e3):.1f}%")
