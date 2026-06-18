"""GPT's knife suite: random isolation, OOV, contribution split"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, secrets, random

DEV = torch.device('cuda')
D = 128; V = 5000

class CM(nn.Module):
    def __init__(self):
        super().__init__()
        W=torch.randn(D,D)*0.3; m=(torch.rand(D,D)>0.5).float(); W=W*m
        s=torch.linalg.norm(W,2); self.W=nn.Parameter(W*(1.5/(s+1e-8)))
        self.b=nn.Parameter(torch.zeros(D)); self.g=nn.Parameter(torch.tensor(0.1))
    def fwd(self,x):
        B,T,d=x.shape; h=torch.zeros(B,D,device=x.device)
        for t in range(T): h=h.detach(); h=0.9*torch.tanh(h@self.W.T+self.b+x[:,t,:])+0.1*self.g*h
        return h

class EN(nn.Module):
    def __init__(self,v=V): super().__init__(); self.emb=nn.Embedding(v,D); self.pos=nn.Parameter(torch.randn(1,4096,D)*0.02)
    def forward(self,x): B,T=x.shape; return self.emb(x)+self.pos[:,:T,:]

class DE(nn.Module):
    def __init__(self,v=V): super().__init__(); self.net=nn.Sequential(nn.Linear(D,D*2),nn.GELU(),nn.Linear(D*2,D),nn.GELU(),nn.Linear(D,v))
    def forward(self,h): return self.net(h)

def gen_data(n_pairs,n_samples,seed_val,rng_fn):
    """rng_fn(n) -> random int in [1, n-1]"""
    rng = random.Random(seed_val)
    X,Y=[],[]
    for _ in range(n_samples):
        ks=[rng_fn(rng, V-1) for _ in range(n_pairs)]
        vs=[rng_fn(rng, V-1) for _ in range(n_pairs)]
        qi=rng.randint(0,n_pairs-1)
        s=[]
        for k,v in zip(ks,vs): s.extend([k,v])
        s.extend([rng_fn(rng, V-1) for _ in range(4)])
        s.append(ks[qi])
        X.append(s); Y.append(vs[qi])
    return torch.tensor(X),torch.tensor(Y)

def py_rand(rng,n): return rng.randint(1,n)
def sec_rand(rng,n): return secrets.randbelow(n-1)+1  # crypto-random, ignores rng

def train(X,Y,steps=800):
    X,Y=X.to(DEV),Y.to(DEV)
    enc,core,dec=EN().to(DEV),CM().to(DEV),DE().to(DEV)
    opt=torch.optim.Adam(list(enc.parameters())+list(core.parameters())+list(dec.parameters()),lr=0.002)
    for _ in range(steps):
        idx=torch.randint(0,len(X),(64,))
        loss=F.cross_entropy(dec(core.fwd(enc(X[idx]))),Y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return enc,core,dec

def test(enc,core,dec,X,Y):
    X,Y=X.to(DEV),Y.to(DEV)
    with torch.no_grad():
        e=enc(X); pl=X.shape[1]-1
        post=(dec(core.fwd(e)).argmax(-1)==Y).float().mean().item()
        pre=(dec(core.fwd(e[:,:pl,:])).argmax(-1)==Y).float().mean().item()
    return post,pre

print('=== HONESTY TESTS ===')
print()

# K1: ISOLATED RANDOM SOURCES (training seed ≠ test seed)
print('K1: ISOLATED RANDOM SOURCES')
X_tr,Y_tr = gen_data(10,2000,12345,py_rand)   # train seed=12345
X_te,Y_te = gen_data(10,500,987654321,py_rand) # test seed=987654321 (completely different)
enc,core,dec = train(X_tr,Y_tr)
post,pre = test(enc,core,dec,X_te,Y_te)
print(f'  py_rand(train≠test): post={post:.4f} pre={pre:.4f}')

# Crypto-random
X_tr_c,Y_tr_c = gen_data(10,2000,42,sec_rand)
X_te_c,Y_te_c = gen_data(10,500,999,sec_rand)
enc_c,core_c,dec_c = train(X_tr_c,Y_tr_c)
post_c,pre_c = test(enc_c,core_c,dec_c,X_te_c,Y_te_c)
print(f'  secrets (crypto):     post={post_c:.4f} pre={pre_c:.4f}')

# K2: OUT-OF-VOCAB (train on 1-2500, test on 2501-5000)
print()
print('K2: OUT-OF-VOCAB GENERALIZATION')
# Train: V=2500, Test: same structure but all chars shifted by 2500
V_train = 2500
class EN_oov(nn.Module):
    def __init__(self,v): super().__init__(); self.emb=nn.Embedding(v,D); self.pos=nn.Parameter(torch.randn(1,4096,D)*0.02)
    def forward(self,x): B,T=x.shape; return self.emb(x)+self.pos[:,:T,:]
class DE_oov(nn.Module):
    def __init__(self,v): super().__init__(); self.net=nn.Sequential(nn.Linear(D,D*2),nn.GELU(),nn.Linear(D*2,D),nn.GELU(),nn.Linear(D,v))
    def forward(self,h): return self.net(h)

rng=random.Random(42)
X_tr_oov,Y_tr_oov=[],[]
for _ in range(2000):
    ks=[rng.randint(1,V_train-1) for _ in range(10)]
    vs=[rng.randint(1,V_train-1) for _ in range(10)]
    qi=rng.randint(0,9); s=[]
    for k,v in zip(ks,vs): s.extend([k,v])
    s.extend([rng.randint(1,V_train-1) for _ in range(4)]); s.append(ks[qi])
    X_tr_oov.append(s); Y_tr_oov.append(vs[qi])
X_tr_oov,Y_tr_oov=torch.tensor(X_tr_oov),torch.tensor(Y_tr_oov)

enc_o,core_o,dec_o=EN(v=V_train).to(DEV),CM().to(DEV),DE(v=V_train).to(DEV)
Xo,Yo=X_tr_oov.to(DEV),Y_tr_oov.to(DEV)
opt_o=torch.optim.Adam(list(enc_o.parameters())+list(core_o.parameters())+list(dec_o.parameters()),lr=0.002)
for _ in range(800):
    idx=torch.randint(0,len(Xo),(64,))
    loss=F.cross_entropy(dec_o(core_o.fwd(enc_o(Xo[idx]))),Yo[idx]); opt_o.zero_grad(); loss.backward(); opt_o.step()

# Test: shift all token IDs by +2500
rng2=random.Random(999)
X_te_oov,Y_te_oov=[],[]
for _ in range(500):
    ks=[rng2.randint(1,V_train-1) for _ in range(10)]
    vs=[rng2.randint(1,V_train-1) for _ in range(10)]
    qi=rng2.randint(0,9); s=[]
    for k,v in zip(ks,vs): s.extend([k+2500,v+2500])  # SHIFTED!
    s.extend([rng2.randint(1,V_train-1)+2500 for _ in range(4)])
    s.append(ks[qi]+2500)
    X_te_oov.append(s); Y_te_oov.append(vs[qi]+2500)

# Build new Decoder for full vocab
enc_full = EN(v=V).to(DEV)
enc_full.emb.weight.data[:V_train] = enc_o.emb.weight.data  # copy trained embeddings
enc_full.emb.weight.data[V_train:] = enc_o.emb.weight.data[:V_train]  # repeat (won't be used)
enc_full = enc_full.to(DEV)
core_full = core_o  # same Core
dec_full = DE(v=V).to(DEV)
# Copy trained decoder weights for the first V_train outputs
dec_full.net[-1].weight.data[:V_train] = dec_o.net[-1].weight.data
dec_full.net[-1].weight.data[V_train:] = dec_o.net[-1].weight.data[:V_train]  # repeat
dec_full = dec_full.to(DEV)

X_te_oov,Y_te_oov=torch.tensor(X_te_oov).to(DEV),torch.tensor(Y_te_oov).to(DEV)
with torch.no_grad():
    e=enc_full(X_te_oov); pl=X_te_oov.shape[1]-1
    post_oov=(dec_full(core_full.fwd(e)).argmax(-1)==Y_te_oov).float().mean().item()
    pre_oov=(dec_full(core_full.fwd(e[:,:pl,:])).argmax(-1)==Y_te_oov).float().mean().item()
print(f'  Train vocab=2500, Test vocab=2501-5000: post={post_oov:.4f} pre={pre_oov:.4f}')
print(f'  (random baseline: 1/5000={1/5000:.4f})')

# K3: CONTRIBUTION SPLIT
print()
print('K3: ENCODER/CORE/DECODER CONTRIBUTION SPLIT')
rng3=random.Random(42)
X_s,Y_s=[],[]
for _ in range(2000):
    ks=[rng3.randint(1,V-1) for _ in range(10)]; vs=[rng3.randint(1,V-1) for _ in range(10)]
    qi=rng3.randint(0,9); s=[]
    for k,v in zip(ks,vs): s.extend([k,v])
    s.extend([rng3.randint(1,V-1) for _ in range(4)]); s.append(ks[qi])
    X_s.append(s); Y_s.append(vs[qi])
X_s,Y_s=torch.tensor(X_s).to(DEV),torch.tensor(Y_s).to(DEV)

# Full model
enc_s,core_s,dec_s=EN().to(DEV),CM().to(DEV),DE().to(DEV)
opt_s=torch.optim.Adam(list(enc_s.parameters())+list(core_s.parameters())+list(dec_s.parameters()),lr=0.002)
for _ in range(800):
    idx=torch.randint(0,len(X_s),(64,))
    loss=F.cross_entropy(dec_s(core_s.fwd(enc_s(X_s[idx]))),Y_s[idx]); opt_s.zero_grad(); loss.backward(); opt_s.step()
with torch.no_grad():
    e=enc_s(X_s[:200]); pl=X_s.shape[1]-1
    post_full=(dec_s(core_s.fwd(e)).argmax(-1)==Y_s[:200]).float().mean().item()
print(f'  Full model:              post={post_full:.4f}')

# No Encoder (random input)
core_n,dec_n=CM().to(DEV),DE().to(DEV)
opt_n=torch.optim.Adam(list(core_n.parameters())+list(dec_n.parameters()),lr=0.002)
for _ in range(800):
    idx=torch.randint(0,len(X_s),(64,))
    e_rand=torch.randn(len(idx),X_s.shape[1],D,device=DEV)*0.5
    loss=F.cross_entropy(dec_n(core_n.fwd(e_rand)),Y_s[idx]); opt_n.zero_grad(); loss.backward(); opt_n.step()
with torch.no_grad():
    e_rand=torch.randn(200,X_s.shape[1],D,device=DEV)*0.5
    post_noenc=(dec_n(core_n.fwd(e_rand)).argmax(-1)==Y_s[:200]).float().mean().item()
print(f'  No Encoder (random):     post={post_noenc:.4f}')

# No Core (direct encoder→decoder)
class NoCore(nn.Module):
    def fwd(self,x): return x.mean(dim=1)  # just average sequence
enc_nc,dec_nc=EN().to(DEV),DE().to(DEV); core_nc=NoCore()
opt_nc=torch.optim.Adam(list(enc_nc.parameters())+list(dec_nc.parameters()),lr=0.002)
for _ in range(800):
    idx=torch.randint(0,len(X_s),(64,))
    loss=F.cross_entropy(dec_nc(core_nc.fwd(enc_nc(X_s[idx]))),Y_s[idx]); opt_nc.zero_grad(); loss.backward(); opt_nc.step()
with torch.no_grad():
    e=enc_nc(X_s[:200]); pl=X_s.shape[1]-1
    post_nocore=(dec_nc(core_nc.fwd(e)).argmax(-1)==Y_s[:200]).float().mean().item()
print(f'  No Core (avg only):      post={post_nocore:.4f}')

# Frozen Encoder (random init, not trained)
enc_fr=EN().to(DEV); core_fr=CM().to(DEV); dec_fr=DE().to(DEV)
enc_fr.emb.weight.requires_grad=False; enc_fr.pos.requires_grad=False
opt_fr=torch.optim.Adam(list(core_fr.parameters())+list(dec_fr.parameters()),lr=0.002)
for _ in range(800):
    idx=torch.randint(0,len(X_s),(64,))
    loss=F.cross_entropy(dec_fr(core_fr.fwd(enc_fr(X_s[idx]))),Y_s[idx]); opt_fr.zero_grad(); loss.backward(); opt_fr.step()
with torch.no_grad():
    e=enc_fr(X_s[:200]); pl=X_s.shape[1]-1
    post_frenc=(dec_fr(core_fr.fwd(e)).argmax(-1)==Y_s[:200]).float().mean().item()
print(f'  Frozen Encoder:          post={post_frenc:.4f}')
print(f'  Random baseline:         {1/V:.4f}')
