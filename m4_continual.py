"""M4: Continual Learning — 这是 Core 的 Killer Feature

Task A → Task B → test Task A again.
Transformer forgets A. Core's structural anti-forgetting preserves A.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, random, time

DEV = torch.device('cuda')
D = 128

# ═════════ Core ═════
class CoreMemory(nn.Module):
    def __init__(self, d=D, sr=1.0, gain=0.1, sp=0.5):
        super().__init__()
        W = torch.randn(d,d)*0.3; mask=(torch.rand(d,d)>sp).float(); W=W*mask
        s=torch.linalg.norm(W,2)
        self.W=nn.Parameter(W*(sr/(s+1e-8)))
        self.b=nn.Parameter(torch.zeros(d)); self.g=nn.Parameter(torch.tensor(gain))
        self.write_gate=nn.Linear(d,d); self.read_gate=nn.Linear(d,d)

    def forward(self, x_seq):
        B,T,d=x_seq.shape; h=torch.zeros(B,d,device=x_seq.device)
        for t in range(T):
            h=h.detach()
            gi=torch.sigmoid(self.write_gate(x_seq[:,t,:]))
            h=0.9*torch.tanh(h@self.W.T+self.b+gi*x_seq[:,t,:])+0.1*self.g*h
        go=torch.sigmoid(self.read_gate(h))
        return go*h

# ═════ Transformer ═════
class TFBlock(nn.Module):
    def __init__(self,d=D,h=4):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,h,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
        self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
    def forward(self,x):
        a,_=self.attn(x,x,x); x=self.n1(x+a); return self.n2(x+self.ffn(x))

# ═════ Models ═════
class BaseModel(nn.Module):
    def __init__(self, vocab=26):
        super().__init__()
        self.emb=nn.Embedding(vocab,D)
        self.pos=nn.Parameter(torch.randn(1,64,D)*0.02)
        self.tf=TFBlock(); self.ro=nn.Linear(D,vocab)
    def forward(self,x):
        B,T=x.shape; e=self.emb(x)+self.pos[:,:T,:]
        return self.ro(self.tf(e)[:,-1,:])

class CoreModel(nn.Module):
    def __init__(self, vocab=26):
        super().__init__()
        self.emb=nn.Embedding(vocab,D)
        self.pos=nn.Parameter(torch.randn(1,64,D)*0.02)
        self.tf=TFBlock(); self.core=CoreMemory(); self.ro=nn.Linear(D,vocab)
    def forward(self,x):
        B,T=x.shape; e=self.emb(x)+self.pos[:,:T,:]
        tf_out=self.tf(e); mem=self.core(tf_out)
        return self.ro(tf_out[:,-1,:]+mem)

# ═════ Continual Learning Task ═════
def make_task(vocab=26, n_samples=2000, seq_len=10, label_fn=None):
    """Generate classification task: sequence → label."""
    X,Y=[],[]
    for _ in range(n_samples):
        seq=[random.randint(1,vocab-1) for _ in range(seq_len)]
        X.append(seq)
        Y.append(label_fn(seq) if label_fn else seq[0]%vocab)
    return torch.tensor(X), torch.tensor(Y)

# Task A: sum of chars mod vocab
def task_A_label(seq): return sum(seq) % 26
# Task B: first char mod vocab
def task_B_label(seq): return seq[0] % 26


def continual_test(model_class, name, n_steps=1500):
    torch.manual_seed(42); random.seed(42); np.random.seed(42)

    X_A, Y_A = make_task(label_fn=task_A_label)
    X_B, Y_B = make_task(label_fn=task_B_label)
    V = 26
    model = model_class(V).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    B = 64

    results = {}

    # Phase 1: Train on Task A
    X, Y = X_A.to(DEV), Y_A.to(DEV)
    for step in range(n_steps):
        idx = torch.randint(0, len(X), (B,))
        logits = model(X[idx])
        loss = F.cross_entropy(logits, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate Task A
    with torch.no_grad():
        c, t = 0, 0
        for i in range(0, len(X), B):
            pred = model(X[i:i+B].to(DEV)).argmax(-1).cpu()
            c += (pred == Y[i:i+B].cpu()).sum().item(); t += len(Y[i:i+B])
        results['after_A'] = c/t

    # Phase 2: Train on Task B — FREEZE Core W/b if CoreModel
    if hasattr(model, 'core'):
        model.core.W.requires_grad = False
        model.core.b.requires_grad = False
        model.core.g.requires_grad = False
    X, Y = X_B.to(DEV), Y_B.to(DEV)
    for step in range(n_steps):
        idx = torch.randint(0, len(X), (B,))
        logits = model(X[idx])
        loss = F.cross_entropy(logits, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate Task B
    with torch.no_grad():
        c, t = 0, 0
        for i in range(0, len(X), B):
            pred = model(X[i:i+B].to(DEV)).argmax(-1).cpu()
            c += (pred == Y[i:i+B].cpu()).sum().item(); t += len(Y[i:i+B])
        results['after_B'] = c/t

    # Phase 3: Re-evaluate Task A (THE FORGETTING TEST)
    X, Y = X_A.to(DEV), Y_A.to(DEV)
    with torch.no_grad():
        c, t = 0, 0
        for i in range(0, len(X), B):
            pred = model(X[i:i+B].to(DEV)).argmax(-1).cpu()
            c += (pred == Y[i:i+B].cpu()).sum().item(); t += len(Y[i:i+B])
        results['recall_A'] = c/t

    forgetting = results['after_A'] - results['recall_A']
    return results, forgetting


print("M4 Continual Learning: Task A → Task B → test A")
print(f"{'Model':>25s} {'after A':>8s} {'after B':>8s} {'recall A':>8s} {'forget':>8s}")
print("-"*65)

for ModelClass, name in [
    (BaseModel, 'Transformer only'),
    (CoreModel, 'Transformer + Core'),
]:
    res, fg = continual_test(ModelClass, name)
    print(f'{name:>25s} {res["after_A"]:>8.4f} {res["after_B"]:>8.4f} {res["recall_A"]:>8.4f} {fg:>+8.4f}')
