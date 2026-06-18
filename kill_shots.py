"""Reviewer #2 kill shots: vocab scale, D shrink, random encoder, permutation"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, random

DEV = torch.device('cuda')

def run_test(D, V, n_pairs=10, n_noise=4, n_samples=2000, steps=800, random_enc=False):
    class CoreM(nn.Module):
        def __init__(self):
            super().__init__()
            W=torch.randn(D,D)*0.3; m=(torch.rand(D,D)>0.5).float(); W=W*m
            s=torch.linalg.norm(W,2)
            self.W=nn.Parameter(W*(1.5/(s+1e-8)))
            self.b=nn.Parameter(torch.zeros(D))
            self.g=nn.Parameter(torch.tensor(0.1))
        def fwd(self,x):
            B,T,d=x.shape; h=torch.zeros(B,D,device=x.device)
            for t in range(T): h=h.detach(); h=0.9*torch.tanh(h@self.W.T+self.b+x[:,t,:])+0.1*self.g*h
            return h

    class Enc(nn.Module):
        def __init__(self):
            super().__init__(); self.emb=nn.Embedding(V,D)
            self.pos=nn.Parameter(torch.randn(1,2048,D)*0.02)
        def forward(self,x): B,T=x.shape; return self.emb(x)+self.pos[:,:T,:]

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(D,D*2),nn.GELU(),nn.Linear(D*2,D),nn.GELU(),nn.Linear(D,V))
        def forward(self,h): return self.net(h)

    X,Y=[],[]
    for _ in range(n_samples):
        ks=[random.randint(1,V-1) for _ in range(n_pairs)]
        vs=[random.randint(1,V-1) for _ in range(n_pairs)]
        qi=random.randint(0,n_pairs-1); s=[]
        for k,v in zip(ks,vs): s.extend([k,v])
        s.extend([random.randint(1,V-1) for _ in range(n_noise)]); s.append(ks[qi])
        X.append(s); Y.append(vs[qi])
    X,Y=torch.tensor(X).to(DEV),torch.tensor(Y).to(DEV)

    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    if random_enc:
        core,dec=CoreM().to(DEV),Dec().to(DEV)
        opt=torch.optim.Adam(list(core.parameters())+list(dec.parameters()),lr=0.002)
    else:
        enc,core,dec=Enc().to(DEV),CoreM().to(DEV),Dec().to(DEV)
        opt=torch.optim.Adam(list(enc.parameters())+list(core.parameters())+list(dec.parameters()),lr=0.002)

    for _ in range(steps):
        idx=torch.randint(0,len(X),(64,))
        if random_enc:
            e=torch.randn(len(idx),X.shape[1],D,device=DEV)*0.5
        else:
            e=enc(X[idx])
        loss=F.cross_entropy(dec(core.fwd(e)),Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pl=X.shape[1]-1
        if random_enc:
            e=torch.randn(200,X.shape[1],D,device=DEV)*0.5
        else:
            e=enc(X[:200])
        post=(dec(core.fwd(e)).argmax(-1)==Y[:200]).float().mean().item()
        pre=(dec(core.fwd(e[:,:pl,:])).argmax(-1)==Y[:200]).float().mean().item()
    return post,pre


print('=== KILL SHOTS ===')
print()

# K1: VOCAB SCALE
print('K1: VOCAB SCALE (D=128, 10 pairs)')
for V in [26, 100, 500, 1000, 5000]:
    p,pr = run_test(128, V, steps=800)
    print(f'  vocab={V:>5d} (random={1/V:.4f}): post={p:.4f} pre={pr:.4f}')

# K2: D SHRINK
print()
print('K2: D SHRINK (vocab=26, 10 pairs)')
for D_val in [64, 32]:
    p,pr = run_test(D_val, 26, steps=800)
    print(f'  D={D_val:>3d}: post={p:.4f} pre={pr:.4f}')

# K3: RANDOM ENCODER
print()
print('K3: RANDOM ENCODER (no real input)')
p,pr = run_test(128, 26, steps=800, random_enc=True)
print(f'  post={p:.4f} pre={pr:.4f} (should be ~random if encoder matters)')

# K4: PERMUTATION TEST
print()
print('K4: PERMUTATION TEST')
V=26;D=128
import torch.nn as nn
class CoreP(nn.Module):
    def __init__(self):
        super().__init__()
        W=torch.randn(D,D)*0.3; m=(torch.rand(D,D)>0.5).float(); W=W*m
        s=torch.linalg.norm(W,2)
        self.W=nn.Parameter(W*(1.5/(s+1e-8)))
        self.b=nn.Parameter(torch.zeros(D)); self.g=nn.Parameter(torch.tensor(0.1))
    def fwd(self,x):
        B,T,d=x.shape; h=torch.zeros(B,D,device=x.device)
        for t in range(T): h=h.detach(); h=0.9*torch.tanh(h@self.W.T+self.b+x[:,t,:])+0.1*self.g*h
        return h
class EncP(nn.Module):
    def __init__(self): super().__init__(); self.emb=nn.Embedding(V,D); self.pos=nn.Parameter(torch.randn(1,2048,D)*0.02)
    def forward(self,x): B,T=x.shape; return self.emb(x)+self.pos[:,:T,:]
class DecP(nn.Module):
    def __init__(self): super().__init__()
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(D,D*2),nn.GELU(),nn.Linear(D*2,D),nn.GELU(),nn.Linear(D,V))
    def forward(self,h): return self.net(h)

X_tr,Y_tr=[],[]
for _ in range(2000):
    ks=[random.randint(1,V-1) for _ in range(10)]; vs=[random.randint(1,V-1) for _ in range(10)]
    qi=random.randint(0,9); s=[]
    for k,v in zip(ks,vs): s.extend([k,v])
    s.extend([random.randint(1,V-1) for _ in range(4)]); s.append(ks[qi])
    X_tr.append(s); Y_tr.append(vs[qi])
X_tr,Y_tr=torch.tensor(X_tr).to(DEV),torch.tensor(Y_tr).to(DEV)

torch.manual_seed(42);np.random.seed(42);random.seed(42)
enc_p,core_p,dec_p=EncP().to(DEV),CoreP().to(DEV),DecP().to(DEV)
opt_p=torch.optim.Adam(list(enc_p.parameters())+list(core_p.parameters())+list(dec_p.parameters()),lr=0.002)
for _ in range(800):
    idx=torch.randint(0,len(X_tr),(64,))
    loss=F.cross_entropy(dec_p(core_p.fwd(enc_p(X_tr[idx]))),Y_tr[idx]); opt_p.zero_grad(); loss.backward(); opt_p.step()

# Test with REMAPPED values
X_te,Y_te=[],[]
for _ in range(500):
    ks=[random.randint(1,V-1) for _ in range(10)]
    remap_vs=[random.randint(1,V-1) for _ in range(10)]; qi=random.randint(0,9)
    s=[]
    for k,v in zip(ks,remap_vs): s.extend([k,v])
    s.extend([random.randint(1,V-1) for _ in range(4)]); s.append(ks[qi])
    X_te.append(s); Y_te.append(remap_vs[qi])
X_te,Y_te=torch.tensor(X_te).to(DEV),torch.tensor(Y_te).to(DEV)
with torch.no_grad():
    e=enc_p(X_te); post_remap=(dec_p(core_p.fwd(e)).argmax(-1)==Y_te).float().mean().item()
print(f'  Trained mapping:    post=1.0000')
print(f'  Remapped K-V test:  post={post_remap:.4f} (random={1/V:.4f})')
