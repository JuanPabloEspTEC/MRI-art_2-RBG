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
def ols(y,Xp):
    M=np.column_stack([np.ones(len(y))]+list(Xp)) if Xp else np.ones((len(y),1))
    b,*_=np.linalg.lstsq(M,y,rcond=None); return b,y-M@b

print("=== 1. lambdas Yeo-Johnson con limites amplios ===")
def yj(x,l): return np.log1p(x) if abs(l)<1e-8 else ((x+1)**l-1)/l
def jac(x,l): return (l-1)*np.log1p(x).sum()
lams={}
for v in V:
    x=d[v].values.astype(float)
    f=lambda l:(lambda z: 1e12 if z.std()<=0 or not np.isfinite(z.std()) else
                -(stats.norm.logpdf(z,z.mean(),z.std()).sum()+jac(x,l)))(yj(x,l))
    r=optimize.minimize_scalar(f,bounds=(-8,8),method='bounded'); lams[v]=float(r.x)
print({k:round(v,2) for k,v in lams.items()})

def gbn_score(strat, transform):
    ll=0.0; npar=0
    for v in V:
        x=d[v].values.astype(float)
        X=yj(x,lams[v]) if transform else x
        P=[(yj(d[p].values.astype(float),lams[p]) if transform else d[p].values.astype(float)) for p in PA[v]]
        for g in ([None] if not strat else [0,1]):
            m=np.ones(n,bool) if g is None else cls==g
            b,r=ols(X[m],[p[m] for p in P]); s=np.sqrt((r**2).mean())
            ll+=stats.norm.logpdf(r,0,s).sum(); npar+=len(PA[v])+2
    if transform:
        ll+=sum(jac(d[v].values.astype(float),lams[v]) for v in V); npar+=len(V)
    return ll,npar,ll-0.5*logn*npar,ll-npar
for nm,(st,tf) in {"M0":(False,False),"M1":(True,)[0:0] or (False,True),"M2":(True,False),"M3":(True,True)}.items():
    ll,p,b,a=gbn_score(st,tf); print(f"{nm}: par={p} loglik={ll:.2f} BIC={b:.2f} AIC={a:.2f}")

print("\n=== 2. KDE con ancho de banda optimizado por validacion cruzada (LOO) ===")
def loo_kde(r,h):
    D=(r[:,None]-r[None,:])/h; K=np.exp(-0.5*D**2)/np.sqrt(2*np.pi); np.fill_diagonal(K,0)
    return np.log(np.maximum(K.sum(1)/((len(r)-1)*h),1e-300)).sum()
for transform in [False,True]:
    tot_g=0.0; tot_k=0.0; peor=[]
    for v in V:
        x=d[v].values.astype(float); X=yj(x,lams[v]) if transform else x
        P=[(yj(d[p].values.astype(float),lams[p]) if transform else d[p].values.astype(float)) for p in PA[v]]
        b,r=ols(X,P); s=np.sqrt((r**2).mean())
        g=stats.norm.logpdf(r,0,s).sum()
        hs=np.exp(np.linspace(np.log(0.02*s),np.log(2*s),40))
        k=max(loo_kde(r,h) for h in hs)
        tot_g+=g; tot_k+=k; peor.append((v,round(k-g,1)))
    print(f"transform={transform}: loglik gauss={tot_g:.2f}  KDE(h optimo)={tot_k:.2f}  dif={tot_k-tot_g:+.2f}")
    print("   por nodo (KDE - gauss):", sorted(peor,key=lambda t:t[1])[:4], "...", sorted(peor,key=lambda t:-t[1])[:4])

print("\n=== 3. Donde colocar la diabetes (BIC condicional gaussiano, n=398) ===")
dm=raw['dm'].astype(str).str.strip().str.lower().replace({'':None,'?':None})
ok=dm.isin(['yes','no']).values
dmv=(dm[ok]=='yes').astype(int).values
dd=d[ok].reset_index(drop=True); m=len(dd)
print("n =",m," dm: no =",int((dmv==0).sum())," si =",int((dmv==1).sum()))
def cg_score(extra_parents):
    """extra_parents: nodos continuos que tienen a dm como padre discreto"""
    p1=dmv.mean(); ll=(dmv*np.log(p1)+(1-dmv)*np.log(1-p1)).sum(); npar=1
    for v in V:
        y=dd[v].values.astype(float); P=[dd[p].values.astype(float) for p in PA[v]]
        if v in extra_parents:
            for g in [0,1]:
                mm=dmv==g; b,r=ols(y[mm],[c[mm] for c in P]); s=np.sqrt((r**2).mean())
                ll+=stats.norm.logpdf(r,0,s).sum(); npar+=len(PA[v])+2
        else:
            b,r=ols(y,P); s=np.sqrt((r**2).mean())
            ll+=stats.norm.logpdf(r,0,s).sum(); npar+=len(PA[v])+2
    return ll,npar,ll-0.5*np.log(m)*npar
base=cg_score(set())
print(f"dm aislada (sin arcos):        BIC={base[2]:.2f}")
for cand in [{"bgr"},{"bp"},{"sc"},{"bgr","bp"},{"bgr","sc"},{"bgr","bp","sc"}]:
    s=cg_score(cand); print(f"dm -> {str(sorted(cand)):28s} BIC={s[2]:.2f}  (dif {s[2]-base[2]:+.2f})")
# busqueda voraz
sel=set(); cur=base[2]
while True:
    best=(cur,None)
    for v in V:
        if v in sel: continue
        s=cg_score(sel|{v})[2]
        if s>best[0]: best=(s,v)
    if best[1] is None: break
    sel.add(best[1]); cur=best[0]
print("voraz -> dm padre de:", sorted(sel), f" BIC={cur:.2f} (dif {cur-base[2]:+.2f})")

print("\n=== 4. Estratificacion selectiva por diagnostico (que nodos la piden) ===")
gan=[]
for v in V:
    x=d[v].values.astype(float); P=[d[p].values.astype(float) for p in PA[v]]
    b,r=ols(x,P); s=np.sqrt((r**2).mean()); g0=stats.norm.logpdf(r,0,s).sum(); n0=len(PA[v])+2
    g1=0.0; n1=0
    for g in [0,1]:
        mm=cls==g; b,r=ols(x[mm],[c[mm] for c in P]); s=np.sqrt((r**2).mean())
        g1+=stats.norm.logpdf(r,0,s).sum(); n1+=len(PA[v])+2
    gan.append((v, round((g1-0.5*logn*n1)-(g0-0.5*logn*n0),1)))
gan.sort(key=lambda t:-t[1]); print(gan)
