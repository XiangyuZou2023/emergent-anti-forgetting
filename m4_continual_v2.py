"""M4 Continual Learning v2: 冻 Core 骨架，只训 task-specific heads

正确设计:
  Core (W,b,g) — 冻结，任务无关的记忆基底
  write_gate, read_gate — 冻结（控制"怎么用"记忆）
  emb_A, ro_A, tf_A — Task A 专用
  emb_B, ro_B, tf_B — Task B 专用
  切换任务 = 切换 head，Core 不变
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, random, time, copy

DEV = torch.device('cuda')
D = 128

class CoreMemory(nn.Module):
    def __init__(self):  # Fixed architecture — NOT trained
        super().__init__()
        sr, gain, sp = 1.0, 0.1, 0.5
        W=torch.randn(D,D)*0.3; mask=(torch.rand(D,D)>sp).float(); W=W*mask
        s=torch.linalg.norm(W,2)
        self.W=nn.Parameter(W*(sr/(s+1e-8)), requires_grad=False)
        self.b=nn.Parameter(torch.zeros(D), requires_grad=False)
        self.g=nn.Parameter(torch.tensor(gain), requires_grad=False)

    def forward(self, x_seq):
        B,T,d=x_seq.shape; h=torch.zeros(B,D,device=x_seq.device)
        for t in range(T):
            h=h.detach()
            h=0.9*torch.tanh(h@self.W.T+self.b+x_seq[:,t,:])+0.1*self.g*h
        return h  # raw state, no read gate


class TFBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=nn.MultiheadAttention(D,4,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(D,D*4),nn.GELU(),nn.Linear(D*4,D))
        self.n1=nn.LayerNorm(D); self.n2=nn.LayerNorm(D)
    def forward(self,x):
        a,_=self.attn(x,x,x); x=self.n1(x+a); return self.n2(x+self.ffn(x))


class TaskHead(nn.Module):
    """Per-task: transformer, embedding, readout, and how to use Core output."""
    def __init__(self, vocab=26):
        super().__init__()
        self.emb=nn.Embedding(vocab,D)
        self.pos=nn.Parameter(torch.randn(1,64,D)*0.02)
        self.tf=TFBlock()
        self.read_proj=nn.Linear(D,D)  # project Core output for this task
        self.ro=nn.Linear(D,vocab)

    def forward(self, x, core_out):
        B,T=x.shape; e=self.emb(x)+self.pos[:,:T,:]
        tf_out=self.tf(e)
        mem=self.read_proj(core_out)
        return self.ro(tf_out[:,-1,:]+mem)


def make_task(vocab=26, n_samples=2000, seq_len=10):
    X,Y=[],[]
    for _ in range(n_samples):
        seq=[random.randint(1,vocab-1) for _ in range(seq_len)]
        X.append(seq); Y.append(sum(seq)%vocab)  # sum mod vocab
    return torch.tensor(X),torch.tensor(Y)


def train_head(core, head, X, Y, n_steps=1000):
    core.eval(); head.train()
    X,Y=X.to(DEV),Y.to(DEV)
    opt=torch.optim.Adam(head.parameters(),lr=0.002)
    B=64
    for step in range(n_steps):
        idx=torch.randint(0,len(X),(B,))
        bx,by=X[idx],Y[idx]
        with torch.no_grad():
            e=head.emb(bx)+head.pos[:,:bx.shape[1],:]
            tf_in=head.tf(e)
            core_out=core(tf_in)
        logits=head.ro(tf_in[:,-1,:]+head.read_proj(core_out))
        loss=F.cross_entropy(logits,by)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        c,t=0,0
        for i in range(0,len(X),B):
            bx,by=X[i:i+B].to(DEV),Y[i:i+B].to(DEV)
            e=head.emb(bx)+head.pos[:,:bx.shape[1],:]
            tf_in=head.tf(e)
            core_out=core(tf_in)
            pred=head.ro(tf_in[:,-1,:]+head.read_proj(core_out)).argmax(-1)
            c+=(pred==by).sum().item(); t+=len(by)
    return c/t


def test_head(core, head, X, Y):
    if core is not None: core.eval()
    head.eval()
    X,Y=X.to(DEV),Y.to(DEV); B=64
    with torch.no_grad():
        c,t=0,0
        for i in range(0,len(X),B):
            bx,by=X[i:i+B],Y[i:i+B]
            e=head.emb(bx)+head.pos[:,:bx.shape[1],:]
            tf_in=head.tf(e)
            if core is not None:
                core_out=core(tf_in)
                pred=head.ro(tf_in[:,-1,:]+head.read_proj(core_out)).argmax(-1)
            else:
                pred=head.ro(tf_in[:,-1,:]).argmax(-1)
            c+=(pred==by).sum().item(); t+=len(by)
    return c/t


# ═══════ MAIN ═══════
print("M4 Continual v2: Frozen Core + Task-Specific Heads")
print(f"{'Phase':>20s} {'Transformer':>15s} {'Transformer+Core':>15s}")
print("-"*55)

# Shared Core
core = CoreMemory().to(DEV)

# Task A: sum mod 26
X_A, Y_A = make_task(); V=26
# Task B: product-like (first char * last char mod vocab)
def task_B_label(seq): return (seq[0]*seq[-1])%26
X_B, Y_B = [], []
for _ in range(2000):
    seq=[random.randint(1,25) for _ in range(10)]
    X_B.append(seq); Y_B.append((seq[0]*seq[-1])%26)
X_B, Y_B = torch.tensor(X_B), torch.tensor(Y_B)

# ─── Baseline: no Core ───
class HeadNoCore(nn.Module):
    def __init__(self,v=26):
        super().__init__()
        self.emb=nn.Embedding(v,D); self.pos=nn.Parameter(torch.randn(1,64,D)*0.02)
        self.tf=TFBlock(); self.ro=nn.Linear(D,v)
    def forward(self,x):
        B,T=x.shape; e=self.emb(x)+self.pos[:,:T,:]; return self.ro(self.tf(e)[:,-1,:])

# Train baseline head_A on Task A
torch.manual_seed(42)
head_A_nc = HeadNoCore(V).to(DEV)
opt=torch.optim.Adam(head_A_nc.parameters(),lr=0.002)
for _ in range(1000):
    idx=torch.randint(0,len(X_A),(64,))
    logits=head_A_nc(X_A[idx].to(DEV))
    loss=F.cross_entropy(logits,Y_A[idx].to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
acc_A_nc = test_head(None, head_A_nc, X_A, Y_A)

# Train head_B on Task B → overwrites head_A's params
opt=torch.optim.Adam(head_A_nc.parameters(),lr=0.002)
for _ in range(1000):
    idx=torch.randint(0,len(X_B),(64,))
    logits=head_A_nc(X_B[idx].to(DEV))
    loss=F.cross_entropy(logits,Y_B[idx].to(DEV)); opt.zero_grad(); loss.backward(); opt.step()
acc_B_nc = test_head(None, head_A_nc, X_B, Y_B)
recall_A_nc = test_head(None, head_A_nc, X_A, Y_A)

# ─── Core version: separate heads per task ───
torch.manual_seed(42)
head_A = TaskHead(V).to(DEV)
acc_A_core = train_head(core, head_A, X_A, Y_A)

head_B = TaskHead(V).to(DEV)
acc_B_core = train_head(core, head_B, X_B, Y_B)

# Test A: use head_A + same frozen Core
recall_A_core = test_head(core, head_A, X_A, Y_A)

print(f'{"After Task A":>20s} {acc_A_nc:>15.4f} {acc_A_core:>15.4f}')
print(f'{"After Task B":>20s} {acc_B_nc:>15.4f} {acc_B_core:>15.4f}')
print(f'{"Recall Task A":>20s} {recall_A_nc:>15.4f} {recall_A_core:>15.4f}')
print(f'{"Forgetting":>20s} {acc_A_nc-recall_A_nc:>+14.4f} {acc_A_core-recall_A_core:>+14.4f}')
