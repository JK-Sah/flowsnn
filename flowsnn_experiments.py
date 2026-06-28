"""
FlowSNN — NumPy Prototype (LEGACY)
===================================
"""
import numpy as np
from scipy.signal import bilinear, lfilter
import json, sys, time, os

CKPT='/sessions/ecstatic-happy-archimedes/mnt/outputs/flowsnn_results.json'

N_PIL=8; SP=0.01; FN=35.; ZETA=0.12; D=0.012
T=256; FS=500.; BETA=0.9; THRESH=1.0; SLOPE=25.
U_MIN,U_MAX=0.05,0.40; ST=0.20; CONV=0.69
DFmin,DFmax=10.,45.; DSmin,DSmax=0.01,0.05
TURB=0.06; AR1=0.85; NOISE=0.02
PT=0.5; TG=0.12; LPA=0.9
NCHAN=64; H=128; LR=2e-3; BATCH=128; LREG=0.5
NC=3; NR=2; E_MAC=4.6e-12; E_AC=0.9e-12
SEEDS=[0,1,2]
px=np.arange(N_PIL)*SP
wn=2*np.pi*FN; b_c=np.array([1.]); a_c=np.array([1/wn**2,2*ZETA/wn,1.])
b_d,a_d=bilinear(b_c,a_c,fs=FS)

# ── data gen ──────────────────────────────────────────────────────────────────
def gen_uniform(n,rng):
    U=rng.uniform(U_MIN,U_MAX,n); th=rng.uniform(-np.pi,np.pi,n)
    ux=U[:,None,None]*np.cos(th[:,None,None])*np.ones((n,T,N_PIL))
    uy=U[:,None,None]*np.sin(th[:,None,None])*np.ones((n,T,N_PIL))
    return ux,uy,np.zeros(n,int),np.stack([U*np.cos(th),U*np.sin(th)],-1)

def gen_wake(n,rng):
    U=rng.uniform(U_MIN,U_MAX,n); th=rng.uniform(-np.pi,np.pi,n)
    tv=np.arange(T)/FS; ux=np.zeros((n,T,N_PIL)); uy=np.zeros((n,T,N_PIL))
    for i in range(n):
        fs_s=ST*U[i]/D; ph=2*np.pi*fs_s*px/(CONV*U[i])
        ux[i]=U[i]*np.cos(th[i])
        uy[i]=0.3*U[i]*np.sin(2*np.pi*fs_s*tv[:,None]-ph)+U[i]*np.sin(th[i])
    return ux,uy,np.ones(n,int),np.stack([U*np.cos(th),U*np.sin(th)],-1)

def gen_dipole(n,rng):
    U=rng.uniform(U_MIN,U_MAX,n); th=rng.uniform(-np.pi,np.pi,n)
    tv=np.arange(T)/FS; ux=np.zeros((n,T,N_PIL)); uy=np.zeros((n,T,N_PIL))
    for i in range(n):
        fd=rng.uniform(DFmin,DFmax); xs=rng.uniform(px[0],px[-1]); ys=rng.uniform(DSmin,DSmax)
        r2=(px-xs)**2+ys**2+1e-6; amp=rng.uniform(.05,.15)*U[i]
        uy[i]=amp*np.sin(2*np.pi*fd*tv[:,None])/r2+U[i]*np.sin(th[i])
        ux[i]=U[i]*np.cos(th[i])
    return ux,uy,2*np.ones(n,int),np.stack([U*np.cos(th),U*np.sin(th)],-1)

def filt_defl(v):
    """v:(n,T,N_PIL) → LTI-filtered deflection"""
    n_,Ts_,N_=v.shape
    return lfilter(b_d,a_d,v.transpose(0,2,1).reshape(n_*N_,Ts_),axis=-1).reshape(n_,N_,Ts_).transpose(0,2,1)

def gen_dataset(npc,seed):
    rng=np.random.default_rng(seed); parts=[]
    for gen in [gen_uniform,gen_wake,gen_dipole]:
        ux,uy,lbl,tgt=gen(npc,rng)
        # Spatially-independent AR(1) turbulence per pillar (different RNG per component)
        s_t=TURB*U_MAX
        tx=np.zeros((npc,T,N_PIL)); tx[:,0,:]=rng.normal(0,s_t,(npc,N_PIL))
        ty=np.zeros((npc,T,N_PIL)); ty[:,0,:]=rng.normal(0,s_t,(npc,N_PIL))
        for t in range(1,T):
            tx[:,t,:]=AR1*tx[:,t-1,:]+np.sqrt(1-AR1**2)*rng.normal(0,s_t,(npc,N_PIL))
            ty[:,t,:]=AR1*ty[:,t-1,:]+np.sqrt(1-AR1**2)*rng.normal(0,s_t,(npc,N_PIL))
        ux+=tx; uy+=ty
        d=np.concatenate([filt_defl(ux),filt_defl(uy)],-1).astype(np.float32)
        sigma=d.std(axis=(0,1),keepdims=True)+1e-8
        d+=NOISE*sigma*rng.standard_normal(d.shape).astype(np.float32)
        parts.append((d,lbl,tgt))
    D=np.concatenate([p[0] for p in parts]); Y=np.concatenate([p[1] for p in parts])
    R=np.concatenate([p[2] for p in parts]).astype(np.float32)
    return D,Y,R

def standardize(D,stats=None):
    if stats is None:
        mu=D.mean((0,1),keepdims=True); std=D.std((0,1),keepdims=True)+1e-8
        return (D-mu)/std,(mu,std)
    return (D-stats[0])/stats[1]

# ── encoder ───────────────────────────────────────────────────────────────────
def encode_phasic(x):
    n,Ts,C=x.shape; on=np.zeros((n,Ts,C),np.float32); off=np.zeros_like(on)
    ref=x[:,0,:].copy()
    for t in range(1,Ts):
        d=x[:,t,:]-ref; on[:,t,:]=(d>=PT).astype(np.float32); off[:,t,:]=(d<=-PT).astype(np.float32)
        ref=x[:,t,:].copy()
    return on,off

def encode_tonic(x):
    n,Ts,C=x.shape; ft=1./(TG+1e-9)
    lp=np.zeros_like(x); lp[:,0,:]=x[:,0,:]
    for t in range(1,Ts): lp[:,t,:]=LPA*lp[:,t-1,:]+(1-LPA)*x[:,t,:]
    ap=np.zeros((n,C),np.float32); am=np.zeros_like(ap)
    pl=np.zeros((n,Ts,C),np.float32); mi=np.zeros_like(pl)
    for t in range(Ts):
        ap+=np.maximum(lp[:,t,:],0); am+=np.maximum(-lp[:,t,:],0)
        fp=ap>=ft; fm=am>=ft
        pl[:,t,:]=fp.astype(np.float32); mi[:,t,:]=fm.astype(np.float32)
        ap-=fp*ft; am-=fm*ft
    return pl,mi

def encode(D,mode='combined'):
    on,off=encode_phasic(D); pl,mi=encode_tonic(D); z=np.zeros_like(on)
    if mode=='phasic':  return np.concatenate([on,off,z,z],-1).astype(np.float32)
    elif mode=='tonic': return np.concatenate([z,z,pl,mi],-1).astype(np.float32)
    else:               return np.concatenate([on,off,pl,mi],-1).astype(np.float32)

# ── utilities ─────────────────────────────────────────────────────────────────
def softmax(z):
    e=np.exp(z-z.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)

def surr(U):
    """Fast-sigmoid surrogate gradient."""
    return 1./(1.+SLOPE*np.abs(U-THRESH))**2

def adam_step(p,g,m,v,step,lr=LR):
    m=0.9*m+(1-0.9)*g; v=0.999*v+(1-0.999)*g**2
    mh=m/(1-0.9**step); vh=v/(1-0.999**step)
    return p-lr*mh/(np.sqrt(vh)+1e-8),m,v

def clip_grads(grads,maxn=1.0):
    n=np.sqrt(sum((v**2).sum() for v in grads.values()))
    if n>maxn:
        for k in grads: grads[k]=grads[k]/n

def apply_adam(model,grads):
    model.step+=1
    for k in grads:
        p,m,v=adam_step(getattr(model,k),grads[k],model.ms[k],model.vs[k],model.step)
        setattr(model,k,p); model.ms[k]=m; model.vs[k]=v

def split_data(X,Y,R,seed):
    rng=np.random.default_rng(seed*31+11); n=len(X); idx=rng.permutation(n)
    ntr=int(n*.7); nva=int(n*.1); tr,va,te=idx[:ntr],idx[ntr:ntr+nva],idx[ntr+nva:]
    return X[tr],Y[tr],R[tr],X[va],Y[va],R[va],X[te],Y[te],R[te]

def calc_metrics(pc,prn,rstd,rmu,Cte,Rte):
    pr=prn*rstd+rmu; acc=(pc==Cte).mean()
    ph=np.arctan2(pr[:,1],pr[:,0]); th=np.arctan2(Rte[:,1],Rte[:,0])
    hmae=np.degrees(np.abs(np.arctan2(np.sin(ph-th),np.cos(ph-th))).mean())
    smae=np.abs(np.linalg.norm(pr,-1)-np.linalg.norm(Rte,-1)).mean()*1000
    return float(acc),float(hmae),float(smae)

# ── SNN with correct hard-reset LIF and BPTT ─────────────────────────────────
class SNN:
    def __init__(self,rng):
        s=lambda i,o:np.sqrt(2/(i+o))
        # W1 scaled for sparse-spike input (1.8% rate): target sigma=0.61 → 5% LIF firing rate
        # sigma_z1=W1_std*sqrt(f*(1-f)*NCHAN). Need sigma_z1=0.61 → W1_std=0.57
        self.W1=rng.normal(0,0.57,(H,NCHAN)).astype(np.float32)
        # W2: S1 fires ~5.8%; need sigma_z2=0.61 → W2_std=0.61/sqrt(0.058*128)=0.22
        self.W2=rng.normal(0,0.22,(H,H)).astype(np.float32)
        # Wc/Wr: cnt2≈15/neuron; small init to keep logits near 0 initially
        self.Wc=rng.normal(0,0.008,(NC,H)).astype(np.float32)
        self.Wr=rng.normal(0,0.008,(NR,H)).astype(np.float32)
        self.b1=np.zeros(H,np.float32); self.b2=np.zeros(H,np.float32)
        self.bc=np.zeros(NC,np.float32); self.br=np.zeros(NR,np.float32)
        ks=['W1','W2','Wc','Wr','b1','b2','bc','br']
        self.ms={k:np.zeros_like(getattr(self,k)) for k in ks}
        self.vs={k:np.zeros_like(getattr(self,k)) for k in ks}
        self.step=0

    def fwd(self,x):
        """x:(T,B,NCHAN)→logits,reg,cache. Hard-reset LIF."""
        Ts,B,_=x.shape
        # Pre-compute z1 for all timesteps (vectorised matmul)
        z1=x@self.W1.T+self.b1   # (T,B,H)

        U1=np.zeros((B,H),np.float32); S1=np.zeros((B,H),np.float32)
        all_U1=np.empty((Ts,B,H),np.float32); all_S1=np.empty((Ts,B,H),np.float32)
        U2=np.zeros((B,H),np.float32); S2=np.zeros((B,H),np.float32)
        all_U2=np.empty((Ts,B,H),np.float32); all_S2=np.empty((Ts,B,H),np.float32)
        cnt2=np.zeros((B,H),np.float32)

        for t in range(Ts):
            # Layer 1: hard reset
            U1=BETA*U1+z1[t]-THRESH*S1   # S1 is previous S (reset)
            S1=(U1>=THRESH).astype(np.float32)
            all_U1[t]=U1; all_S1[t]=S1
            # Layer 2: hard reset
            z2t=S1@self.W2.T+self.b2     # (B,H) — per-step matmul
            U2=BETA*U2+z2t-THRESH*S2
            S2=(U2>=THRESH).astype(np.float32)
            all_U2[t]=U2; all_S2[t]=S2; cnt2+=S2

        logits=cnt2@self.Wc.T+self.bc; reg=cnt2@self.Wr.T+self.br
        return logits,reg,(x,all_U1,all_U2,all_S1,all_S2,cnt2)

    def train(self,x,yc,yr):
        logits,reg,cache=self.fwd(x)
        x_,U1,U2,S1,S2,cnt2=cache; Ts,B,_=x_.shape

        # Output gradients
        pr=softmax(logits); oh=np.zeros_like(pr); oh[np.arange(B),yc]=1
        dl=(pr-oh)/B; dr=2*(reg-yr)/B*LREG
        dWc=dl.T@cnt2; dbc=dl.sum(0); dWr=dr.T@cnt2; dbr=dr.sum(0)
        d2=dl@self.Wc+dr@self.Wr   # (B,H)

        # BPTT layer 2 — correct formula: g=(d + (-thresh)*g)*surr + beta*g
        # d2 is the gradient of loss w.r.t. each S2[t] summed into cnt2
        sv2=surr(U2)   # (T,B,H) surrogate gradients
        g2=np.zeros((B,H),np.float32); g2t=np.empty((Ts,B,H),np.float32)
        for t in range(Ts-1,-1,-1):
            g2=(d2+(-THRESH)*g2)*sv2[t]+BETA*g2   # CORRECT BPTT
            g2t[t]=g2
        # Vectorised weight grad: dW2 = sum_t g2t[t].T @ S1[t]
        dW2=g2t.reshape(Ts*B,H).T@S1.reshape(Ts*B,H)
        db2=g2t.sum((0,1))

        # BPTT layer 1 — input grad from layer2: d1t[t] = g2t[t] @ W2
        d1t=g2t@self.W2   # (T,B,H) — vectorised
        sv1=surr(U1)
        g1=np.zeros((B,H),np.float32); g1t=np.empty((Ts,B,H),np.float32)
        for t in range(Ts-1,-1,-1):
            g1=(d1t[t]+(-THRESH)*g1)*sv1[t]+BETA*g1
            g1t[t]=g1
        dW1=g1t.reshape(Ts*B,H).T@x_.reshape(Ts*B,NCHAN)
        db1=g1t.sum((0,1))

        grads={'W1':dW1,'W2':dW2,'Wc':dWc,'Wr':dWr,'b1':db1,'b2':db2,'bc':dbc,'br':dbr}
        clip_grads(grads); apply_adam(self,grads)

    def predict(self,X,bs=256):
        logs=[]; regs=[]
        for i in range(0,len(X),bs):
            lg,rg,_=self.fwd(X[i:i+bs].transpose(1,0,2)); logs.append(lg); regs.append(rg)
        return np.concatenate(logs).argmax(-1),np.concatenate(regs)

def train_snn(Xtr,Ctr,Rtr,Xva,Cva,Rva,Xte,Cte,Rte,epochs,seed):
    rng=np.random.default_rng(seed*17+3); model=SNN(rng); n=len(Xtr)
    rmu=Rtr.mean(0); rstd=Rtr.std(0)+1e-8
    Rtr_n=(Rtr-rmu)/rstd; Rva_n=(Rva-rmu)/rstd
    best=np.inf; no_imp=0; bw={k:getattr(model,k).copy() for k in['W1','W2','Wc','Wr']}
    for ep in range(epochs):
        idx=rng.permutation(n)
        for s in range(0,n,BATCH):
            bi=idx[s:s+BATCH]; model.train(Xtr[bi].transpose(1,0,2),Ctr[bi],Rtr_n[bi])
        vl=0; nv=len(Xva)
        for i in range(0,nv,BATCH):
            xb=Xva[i:i+BATCH].transpose(1,0,2); lg,rg,_=model.fwd(xb)
            pr=softmax(lg); oh=np.zeros_like(pr); oh[np.arange(len(pr)),Cva[i:i+BATCH]]=1
            vl+=(-np.log(pr+1e-9)[oh==1].mean()+0.5*((rg-Rva_n[i:i+BATCH])**2).mean())
        if vl<best: best=vl; no_imp=0; bw={k:getattr(model,k).copy() for k in bw}
        else:
            no_imp+=1
            if no_imp>=5: break
    for k,v in bw.items(): setattr(model,k,v)
    pc,prn=model.predict(Xte)
    return calc_metrics(pc,prn,rstd,rmu,Cte,Rte)

# ── MLP baseline ──────────────────────────────────────────────────────────────
class MLP:
    def __init__(self,rng):
        I=2*N_PIL; s=lambda i:np.sqrt(2/i)
        self.W1=rng.normal(0,s(I),(H,I)).astype(np.float32); self.b1=np.zeros(H,np.float32)
        self.W2=rng.normal(0,s(H),(H,H)).astype(np.float32); self.b2=np.zeros(H,np.float32)
        self.Wc=rng.normal(0,s(H),(NC,H)).astype(np.float32); self.bc=np.zeros(NC,np.float32)
        self.Wr=rng.normal(0,s(H),(NR,H)).astype(np.float32); self.br=np.zeros(NR,np.float32)
        ks=['W1','b1','W2','b2','Wc','bc','Wr','br']
        self.ms={k:np.zeros_like(getattr(self,k)) for k in ks}
        self.vs={k:np.zeros_like(getattr(self,k)) for k in ks}
        self.step=0
    def fwd(self,x):
        Ts,B,_=x.shape; xf=x.reshape(Ts*B,-1)
        h1f=np.maximum(0,xf@self.W1.T+self.b1); h2f=np.maximum(0,h1f@self.W2.T+self.b2)
        h2=h2f.reshape(Ts,B,H).mean(0)
        return h2@self.Wc.T+self.bc,h2@self.Wr.T+self.br,(xf,h1f,h2f)
    def train(self,x,yc,yr):
        Ts,B,_=x.shape; lg,rg,cache=self.fwd(x); xf,h1f,h2f,=cache
        pr=softmax(lg); oh=np.zeros_like(pr); oh[np.arange(B),yc]=1
        h2=h2f.reshape(Ts,B,H).mean(0)
        dl=(pr-oh)/B; dr=2*(rg-yr)/B
        dWc=dl.T@h2; dbc=dl.sum(0); dWr=dr.T@h2; dbr=dr.sum(0)
        dh2=dl@self.Wc+dr@self.Wr
        dh2f=np.tile(dh2/Ts,(Ts,1))*(h2f>0)
        dW2=dh2f.T@h1f; db2=dh2f.sum(0)
        dh1f=dh2f@self.W2*(h1f>0); dW1=dh1f.T@xf; db1=dh1f.sum(0)
        grads={'W1':dW1,'b1':db1,'W2':dW2,'b2':db2,'Wc':dWc,'bc':dbc,'Wr':dWr,'br':dbr}
        clip_grads(grads); apply_adam(self,grads)
    def predict(self,X,bs=256):
        logs=[]; regs=[]
        for i in range(0,len(X),bs):
            lg,rg,_=self.fwd(X[i:i+bs].transpose(1,0,2)); logs.append(lg); regs.append(rg)
        return np.concatenate(logs).argmax(-1),np.concatenate(regs)

# ── CNN baseline ──────────────────────────────────────────────────────────────
def conv1d_fwd(x,W,b,k):
    """x:(B,T,Ci), W:(Co,Ci,k) → (B,T-k+1,Co) with relu"""
    B,Ts,Ci=x.shape; To=Ts-k+1; Co=W.shape[0]
    # sliding_window_view along axis=1 → (B, To, Ci, k)
    win=np.lib.stride_tricks.sliding_window_view(x,k,axis=1)   # (B,To,Ci,k)
    col=win.reshape(B,To,Ci*k)   # (B,To,Ci*k)
    Wf=W.reshape(Co,Ci*k)
    return np.maximum(0,col@Wf.T+b), col    # out:(B,To,Co), col for backward

def conv1d_bwd(dout,col,x,W,b,k,h_in):
    """Backward through conv1d+relu. Returns dW,db,dh_in."""
    B,To,Co=dout.shape; Ci=W.shape[1]; Wf=W.reshape(Co,Ci*k)
    dout_=dout.reshape(To*B,Co)
    dcol=(dout_@Wf).reshape(B,To,Ci*k)
    dW=(dout_.T@col.reshape(To*B,Ci*k)).reshape(Co,Ci,k); db=dout.sum((0,1))
    # scatter dcol back into dh_in
    dh=np.zeros_like(h_in)   # (B,T,Ci)
    for kk in range(k): dh[:,kk:kk+To,:]+=dcol.reshape(B,To,Ci,k)[:,:,:,kk]
    return dW,db,dh

class CNN1D:
    def __init__(self,rng):
        I=2*N_PIL; ch=32; k1=7; k2=5; self.k1=k1; self.k2=k2; self.ch=ch
        self.C1=rng.normal(0,np.sqrt(2/(I*k1)),(ch,I,k1)).astype(np.float32); self.cb1=np.zeros(ch,np.float32)
        self.C2=rng.normal(0,np.sqrt(2/(ch*k2)),(ch,ch,k2)).astype(np.float32); self.cb2=np.zeros(ch,np.float32)
        self.Wc=rng.normal(0,np.sqrt(2/(ch+NC)),(NC,ch)).astype(np.float32); self.bc=np.zeros(NC,np.float32)
        self.Wr=rng.normal(0,np.sqrt(2/(ch+NR)),(NR,ch)).astype(np.float32); self.br=np.zeros(NR,np.float32)
        ks=['C1','cb1','C2','cb2','Wc','bc','Wr','br']
        self.ms={k:np.zeros_like(getattr(self,k)) for k in ks}
        self.vs={k:np.zeros_like(getattr(self,k)) for k in ks}
        self.step=0
    def fwd(self,x):
        """x:(T,B,2N)"""
        xb=x.transpose(1,0,2)   # (B,T,2N)
        h1,col1=conv1d_fwd(xb,self.C1,self.cb1,self.k1)   # (B,T1,ch)
        h2,col2=conv1d_fwd(h1,self.C2,self.cb2,self.k2)   # (B,T2,ch)
        gap=h2.mean(1)   # (B,ch)
        return gap@self.Wc.T+self.bc,gap@self.Wr.T+self.br,(xb,h1,h2,gap,col1,col2)
    def train(self,x,yc,yr):
        lg,rg,cache=self.fwd(x); xb,h1,h2,gap,col1,col2=cache; B=xb.shape[0]
        pr=softmax(lg); oh=np.zeros_like(pr); oh[np.arange(B),yc]=1
        dl=(pr-oh)/B; dr=2*(rg-yr)/B
        dWc=dl.T@gap; dbc=dl.sum(0); dWr=dr.T@gap; dbr=dr.sum(0)
        dgap=dl@self.Wc+dr@self.Wr   # (B,ch)
        T2=h2.shape[1]; dh2_gap=np.tile(dgap[:,None,:],(1,T2,1))/T2*(h2>0)
        dC2,dcb2,dh1_=conv1d_bwd(dh2_gap,col2,h1,self.C2,self.cb2,self.k2,h1)
        dh1_relu=dh1_*(h1>0)
        dC1,dcb1,_=conv1d_bwd(dh1_relu,col1,xb,self.C1,self.cb1,self.k1,xb)
        grads={'C1':dC1,'cb1':dcb1,'C2':dC2,'cb2':dcb2,'Wc':dWc,'bc':dbc,'Wr':dWr,'br':dbr}
        clip_grads(grads); apply_adam(self,grads)
    def predict(self,X,bs=256):
        logs=[]; regs=[]
        for i in range(0,len(X),bs):
            lg,rg,_=self.fwd(X[i:i+bs].transpose(1,0,2)); logs.append(lg); regs.append(rg)
        return np.concatenate(logs).argmax(-1),np.concatenate(regs)

# ── GRU baseline ──────────────────────────────────────────────────────────────
class GRUNet:
    def __init__(self,rng):
        sz=H; I=2*N_PIL; s=lambda n:np.sqrt(1/n)
        for nm,sh in [('Wz',(sz,I)),('Wrg',(sz,I)),('Wh',(sz,I)),
                      ('Uz',(sz,sz)),('Ur',(sz,sz)),('Uh',(sz,sz)),
                      ('Wc',(NC,sz)),('Wr2',(NR,sz))]:
            setattr(self,nm,rng.normal(0,s(sh[1]),sh).astype(np.float32))
        for nm in ['bz','brg','bh']: setattr(self,nm,np.zeros(sz,np.float32))
        self.bc=np.zeros(NC,np.float32); self.br2=np.zeros(NR,np.float32)
        ks=['Wz','Wrg','Wh','Uz','Ur','Uh','bz','brg','bh','Wc','Wr2','bc','br2']
        self.ms={k:np.zeros_like(getattr(self,k)) for k in ks}
        self.vs={k:np.zeros_like(getattr(self,k)) for k in ks}
        self.step=0
    def _sig(self,x): return 1/(1+np.exp(-np.clip(x,-15,15)))
    def fwd(self,x):
        Ts,B,_=x.shape; h=np.zeros((B,H),np.float32)
        hs=np.zeros((Ts,B,H),np.float32); zs=np.zeros_like(hs); rs=np.zeros_like(hs); hns=np.zeros_like(hs)
        for t in range(Ts):
            z=self._sig(x[t]@self.Wz.T+h@self.Uz.T+self.bz)
            r=self._sig(x[t]@self.Wrg.T+h@self.Ur.T+self.brg)
            hn=np.tanh(x[t]@self.Wh.T+(r*h)@self.Uh.T+self.bh)
            h=(1-z)*h+z*hn; hs[t]=h; zs[t]=z; rs[t]=r; hns[t]=hn
        lg=h@self.Wc.T+self.bc; rg=h@self.Wr2.T+self.br2
        return lg,rg,(x,hs,zs,rs,hns)
    def train(self,x,yc,yr):
        Ts,B,_=x.shape; lg,rg,cache=self.fwd(x); _,hs,zs,rs,hns=cache
        pr=softmax(lg); oh=np.zeros_like(pr); oh[np.arange(B),yc]=1
        dl=(pr-oh)/B; dr=2*(rg-yr)/B
        dWc=dl.T@hs[-1]; dbc=dl.sum(0); dWr2=dr.T@hs[-1]; dbr2=dr.sum(0)
        dh=dl@self.Wc+dr@self.Wr2
        TRUNC=32
        dWz=np.zeros_like(self.Wz); dWrg=np.zeros_like(self.Wrg); dWh=np.zeros_like(self.Wh)
        dUz=np.zeros_like(self.Uz); dUr=np.zeros_like(self.Ur); dUh=np.zeros_like(self.Uh)
        dbz=np.zeros(H,np.float32); dbrg=np.zeros_like(dbz); dbh=np.zeros_like(dbz)
        for t in range(Ts-1,max(Ts-TRUNC-1,-1),-1):
            hp=hs[t-1] if t>0 else np.zeros_like(hs[0])
            z=zs[t]; r=rs[t]; hn=hns[t]; xt=x[t]
            dhn=dh*z; dzg=dh*(hn-hp); dhp=dh*(1-z)
            dt=dhn*(1-hn**2)
            dWh+=dt.T@xt; dbh+=dt.sum(0); dUh+=dt.T@(r*hp)
            drh=dt@self.Uh; dr_=drh*hp; dhp+=drh*r
            dsigr=dr_*r*(1-r)
            dWrg+=dsigr.T@xt; dbrg+=dsigr.sum(0); dUr+=dsigr.T@hp; dhp+=dsigr@self.Ur
            dsigz=dzg*z*(1-z)
            dWz+=dsigz.T@xt; dbz+=dsigz.sum(0); dUz+=dsigz.T@hp; dhp+=dsigz@self.Uz
            dh=dhp
        grads={'Wz':dWz,'Wrg':dWrg,'Wh':dWh,'Uz':dUz,'Ur':dUr,'Uh':dUh,
               'bz':dbz,'brg':dbrg,'bh':dbh,'Wc':dWc,'Wr2':dWr2,'bc':dbc,'br2':dbr2}
        clip_grads(grads); apply_adam(self,grads)
    def predict(self,X,bs=256):
        logs=[]; regs=[]
        for i in range(0,len(X),bs):
            lg,rg,_=self.fwd(X[i:i+bs].transpose(1,0,2)); logs.append(lg); regs.append(rg)
        return np.concatenate(logs).argmax(-1),np.concatenate(regs)

# ── training loop (shared for baselines) ──────────────────────────────────────
def train_baseline(model,Xtr,Ctr,Rtr,Xva,Cva,Rva,Xte,Cte,Rte,epochs,seed):
    rng=np.random.default_rng(seed*19+7); n=len(Xtr)
    rmu=Rtr.mean(0); rstd=Rtr.std(0)+1e-8; Rtr_n=(Rtr-rmu)/rstd; Rva_n=(Rva-rmu)/rstd
    best=np.inf; no_imp=0
    for ep in range(epochs):
        idx=rng.permutation(n)
        for s in range(0,n,BATCH):
            bi=idx[s:s+BATCH]; model.train(Xtr[bi].transpose(1,0,2),Ctr[bi],Rtr_n[bi])
        vl=0; nv=len(Xva)
        for i in range(0,nv,BATCH):
            xb=Xva[i:i+BATCH].transpose(1,0,2); lg,rg,_=model.fwd(xb)
            pr=softmax(lg); oh=np.zeros_like(pr); oh[np.arange(len(pr)),Cva[i:i+BATCH]]=1
            vl+=(-np.log(pr+1e-9)[oh==1].mean()+0.5*((rg-Rva_n[i:i+BATCH])**2).mean())
        if vl<best: best=vl; no_imp=0
        else:
            no_imp+=1
            if no_imp>=5: break
    pc,prn=model.predict(Xte)
    return calc_metrics(pc,prn,rstd,rmu,Cte,Rte)

def energy_uj(bname):
    I=2*N_PIL
    if bname=='mlp': return T*(I*H+H*H+H*(NC+NR))*E_MAC*1e6
    elif bname=='cnn':
        T1=T-7+1; T2=T1-5+1; return (T1*I*7*32+T2*32*5*32+32*(NC+NR))*E_MAC*1e6
    else: return T*(3*I*H+3*H*H+H*(NC+NR))*E_MAC*1e6

# ── checkpoint & runners ──────────────────────────────────────────────────────
def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f: return json.load(f)
    return {}
def save_ckpt(d):
    with open(CKPT,'w') as f: json.dump(d,f,indent=2)

def run_snn(name,npc,epochs,mode='combined'):
    res=load_ckpt()
    if name in res: print(f'[skip] {name}'); return
    print(f'\n=== {name} ===',flush=True); accs=[]; hmaes=[]; smaes=[]
    for seed in SEEDS:
        t0=time.time()
        D,Y,R=gen_dataset(npc,seed); D,_=standardize(D); S=encode(D,mode)
        Xtr,Ctr,Rtr,Xva,Cva,Rva,Xte,Cte,Rte=split_data(S,Y,R,seed)
        acc,hmae,smae=train_snn(Xtr,Ctr,Rtr,Xva,Cva,Rva,Xte,Cte,Rte,epochs,seed)
        accs.append(acc); hmaes.append(hmae); smaes.append(smae)
        print(f'  s{seed}: acc={acc:.3f} h={hmae:.1f}° sp={smae:.1f}mm/s [{time.time()-t0:.1f}s]',flush=True)
    res[name]={'acc_mean':float(np.mean(accs)),'acc_std':float(np.std(accs)),
               'hmae_mean':float(np.mean(hmaes)),'hmae_std':float(np.std(hmaes)),
               'smae_mean':float(np.mean(smaes)),'smae_std':float(np.std(smaes))}
    save_ckpt(res)
    print(f'  RESULT: acc={np.mean(accs):.3f}±{np.std(accs):.3f}  h={np.mean(hmaes):.1f}±{np.std(hmaes):.1f}°  sp={np.mean(smaes):.1f}±{np.std(smaes):.1f}mm/s',flush=True)

def run_baseline(name,bname,npc,epochs):
    res=load_ckpt()
    if name in res: print(f'[skip] {name}'); return
    print(f'\n=== {name} ===',flush=True); accs=[]; hmaes=[]; smaes=[]
    for seed in SEEDS:
        t0=time.time()
        D,Y,R=gen_dataset(npc,seed); D,_=standardize(D)
        Xtr,Ctr,Rtr,Xva,Cva,Rva,Xte,Cte,Rte=split_data(D,Y,R,seed)
        rng_m=np.random.default_rng(seed*23+5)
        model={'mlp':MLP,'cnn':CNN1D,'gru':GRUNet}[bname](rng_m)
        acc,hmae,smae=train_baseline(model,Xtr,Ctr,Rtr,Xva,Cva,Rva,Xte,Cte,Rte,epochs,seed)
        accs.append(acc); hmaes.append(hmae); smaes.append(smae)
        print(f'  s{seed}: acc={acc:.3f} h={hmae:.1f}° sp={smae:.1f}mm/s [{time.time()-t0:.1f}s]',flush=True)
    euj=energy_uj(bname)
    res[name]={'acc_mean':float(np.mean(accs)),'acc_std':float(np.std(accs)),
               'hmae_mean':float(np.mean(hmaes)),'hmae_std':float(np.std(hmaes)),
               'smae_mean':float(np.mean(smaes)),'smae_std':float(np.std(smaes)),'energy_uj':euj}
    save_ckpt(res)
    print(f'  RESULT: acc={np.mean(accs):.3f}±{np.std(accs):.3f}  h={np.mean(hmaes):.1f}±{np.std(hmaes):.1f}°  E={euj:.1f}µJ',flush=True)

sec=int(sys.argv[1]) if len(sys.argv)>1 else 0
if sec in (0,1): run_snn('main_snn',400,22,'combined')
if sec in (0,2): run_snn('abl_phasic',300,18,'phasic'); run_snn('abl_tonic',300,18,'tonic')
if sec in (0,3): run_snn('abl_combined',300,18,'combined'); run_baseline('baseline_mlp','mlp',300,18)
if sec in (0,4): run_baseline('baseline_cnn','cnn',300,18); run_baseline('baseline_gru','gru',300,18)

print('\n=== SUMMARY ===')
for k,v in load_ckpt().items():
    print(f'{k}: acc={v.get("acc_mean",0):.3f}±{v.get("acc_std",0):.3f}  h={v.get("hmae_mean",0):.1f}±{v.get("hmae_std",0):.1f}°  sp={v.get("smae_mean",0):.1f}mm/s')
