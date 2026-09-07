import numpy as np, pandas as pd
from scipy import stats, optimize
d=pd.read_csv('notebook/data/processed/ckd_imputado.csv')
raw=pd.read_csv('data/raw/ckd.csv')
cls=(raw['classification'].astype(str).str.strip().str.lower()=='ckd').astype(int).values
V=list(d.columns); n=len(d); logn=np.log(n)
DAG2=[("age","bp"),("age","bgr"),("bp","al"),("bp","sg"),("bgr","al"),("bgr","sg"),("al","sc"),
      ("al","bu"),("sg","sc"),("sg","bu"),("sc","sod"),("sc","pot"),("sc","hemo"),("bu","sod"),
      ("bu","pot"),("bu","hemo"),("hemo","pcv"),("hemo","rc"),("sc","wc")]
PA={v:[a for a,b in DAG2 if b==v] for v in V}
def yj(x,l): return np.log1p(x) if abs(l)<1e-8 else ((x+1)**l-1)/l
def jac(x,l): return (l-1)*np.log1p(x).sum()
def fitlam(x):
    f=lambda l:(lambda z: 1e12 if z.std()<=0 or not np.isfinite(z.std()) else
                -(stats.norm.logpdf(z,z.mean(),z.std()).sum()+jac(x,l)))(yj(x,l))
    return float(optimize.minimize_scalar(f,bounds=(-8,8),method='bounded').x)
def ols(y,Xp):
    M=np.column_stack([np.ones(len(y))]+list(Xp)) if Xp else np.ones((len(y),1))
    b,*_=np.linalg.lstsq(M,y,rcond=None); return b,y-M@b
def predict(b,Xp,m):
    M=np.column_stack([np.ones(m)]+list(Xp)) if Xp else np.ones((m,1)); return M@b
def loo_kde(r,h):
    D=(r[:,None]-r[None,:])/h; K=np.exp(-0.5*D**2)/np.sqrt(2*np.pi); np.fill_diagonal(K,0)
    return np.log(np.maximum(K.sum(1)/((len(r)-1)*h),1e-300)).sum()
def best_h(r):
    s=max(r.std(),1e-9); grid=np.exp(np.linspace(np.log(0.02*s),np.log(2*s),40))
    return max(grid,key=lambda h:loo_kde(r,h))
def kde_at(re,rt,h):
    D=(re[:,None]-rt[None,:])/h
    return np.log(np.maximum((np.exp(-0.5*D**2)/np.sqrt(2*np.pi)).sum(1)/(len(rt)*h),1e-300)).sum()

def cv(transform,strat,resid,K=5,seed=7):
    idx=np.arange(n); np.random.default_rng(seed).shuffle(idx)
    folds=np.array_split(idx,K); tot=0.0
    for k in range(K):
        te=folds[k]; tr=np.concatenate([folds[j] for j in range(K) if j!=k])
        dtr=d.iloc[tr]; dte=d.iloc[te]; gtr=cls[tr]; gte=cls[te]
        Xtr={};Xte={};jt=0.0
        for v in V:
            a=dtr[v].values.astype(float); b_=dte[v].values.astype(float)
            if transform:
                l=fitlam(a); Xtr[v]=yj(a,l); Xte[v]=yj(b_,l); jt+=jac(b_,l)
            else: Xtr[v]=a; Xte[v]=b_
        ll=0.0
        for v in V:
            for g in ([None] if not strat else [0,1]):
                mtr=np.ones(len(tr),bool) if g is None else gtr==g
                mte=np.ones(len(te),bool) if g is None else gte==g
                if mte.sum()==0: continue
                b,r=ols(Xtr[v][mtr],[Xtr[p][mtr] for p in PA[v]])
                rte=Xte[v][mte]-predict(b,[Xte[p][mte] for p in PA[v]],mte.sum())
                if resid=='gauss':
                    s=max(np.sqrt((r**2).mean()),1e-9); ll+=stats.norm.logpdf(rte,0,s).sum()
                else:
                    ll+=kde_at(rte,r,best_h(r))
        tot+=ll+jt
    return tot

lams={v:fitlam(d[v].values.astype(float)) for v in V}
def insample(transform,strat,resid):
    ll=0.0; npar=0
    for v in V:
        X=yj(d[v].values.astype(float),lams[v]) if transform else d[v].values.astype(float)
        P=[(yj(d[p].values.astype(float),lams[p]) if transform else d[p].values.astype(float)) for p in PA[v]]
        for g in ([None] if not strat else [0,1]):
            m=np.ones(n,bool) if g is None else cls==g
            b,r=ols(X[m],[c[m] for c in P])
            if resid=='gauss':
                s=np.sqrt((r**2).mean()); ll+=stats.norm.logpdf(r,0,s).sum()
            else:
                ll+=loo_kde(r,best_h(r))
            npar+=len(PA[v])+2
    if transform: ll+=sum(jac(d[v].values.astype(float),lams[v]) for v in V); npar+=len(V)
    return ll,npar

modelos={
 "M0 GBN gaussiana (DAG 2)":            (False,False,'gauss'),
 "M1 + Yeo-Johnson":                    (True, False,'gauss'),
 "M2 + estratificada por diagnostico":  (False,True, 'gauss'),
 "M3 Yeo-Johnson + estratificada":      (True, True, 'gauss'),
 "M4 KDE en residuos":                  (False,False,'kde'),
 "M5 Yeo-Johnson + KDE":                (True, False,'kde'),
 "M6 YJ + estratificada + KDE":         (True, True, 'kde'),
}
print(f"{'modelo':38s} {'par':>4s} {'loglik':>11s} {'BIC':>11s} {'AIC':>11s} {'CV-5':>11s}")
res={}
for nm,(t,s,r) in modelos.items():
    ll,p=insample(t,s,r); b=ll-0.5*logn*p; a=ll-p; c=cv(t,s,r)
    res[nm]=(p,ll,b,a,c)
    print(f"{nm:38s} {p:4d} {ll:11.1f} {b:11.1f} {a:11.1f} {c:11.1f}")
b0=res["M0 GBN gaussiana (DAG 2)"]
print("\nmejora respecto de M0 (BIC / CV):")
for nm,(p,ll,b,a,c) in res.items():
    print(f"  {nm:38s} {b-b0[2]:+9.1f} {c-b0[4]:+9.1f}")
