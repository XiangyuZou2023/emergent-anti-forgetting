"""M4 v2: Fast Transformer + Core Memory — 全并行 Transformer，Core 只步进

任务: delayed recall — 前5个字符记住，20个干扰后预测
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, random, time

DEV = torch.device('cuda')
D = 128

class CoreMemory(nn.Module):
    def __init__(self, d=D, sr=1.5, gain=0.1, sp=0.5):
        super().__init__()
        W = torch.randn(d,d)*0.3; mask = (torch.rand(d,d)>sp).float(); W=W*mask
        s = torch.linalg.norm(W,2)
        self.W = nn.Parameter(W*(sr/(s+1e-8)))
        self.b = nn.Parameter(torch.zeros(d))
        self.g = nn.Parameter(torch.tensor(gain))
        self.write_gate = nn.Linear(d, d)
        self.read_gate = nn.Linear(d, d)

    def forward(self, x_seq):
        """x_seq: (B, T, d). Returns final h (B, d)."""
        B, T, d = x_seq.shape
        h = torch.zeros(B, d, device=x_seq.device)
        for t in range(T):
            h = h.detach()
            gate_in = torch.sigmoid(self.write_gate(x_seq[:, t, :]))
            ext = gate_in * x_seq[:, t, :]
            h = 0.9*torch.tanh(h@self.W.T+self.b+ext)+0.1*self.g*h
        gate_out = torch.sigmoid(self.read_gate(h))
        return gate_out * h


class TransformerBlock(nn.Module):
    def __init__(self, d=D, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d,d*4), nn.GELU(), nn.Linear(d*4,d))
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))


# ─── Models ───
class TransformerOnly(nn.Module):
    def __init__(self, vocab=26):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.pos = nn.Parameter(torch.randn(1, 64, D)*0.02)
        self.tf = TransformerBlock()
        self.ro = nn.Linear(D, vocab)

    def forward(self, x):
        B, T = x.shape
        e = self.emb(x) + self.pos[:, :T, :]
        h = self.tf(e)
        return self.ro(h[:, -1, :])


class TransformerCore(nn.Module):
    """Transformer on full seq (fast) → Core on transformer outputs (sequential) → combine."""
    def __init__(self, vocab=26):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.pos = nn.Parameter(torch.randn(1, 64, D)*0.02)
        self.tf = TransformerBlock()
        self.core = CoreMemory()
        self.ro = nn.Linear(D, vocab)

    def forward(self, x):
        B, T = x.shape
        e = self.emb(x) + self.pos[:, :T, :]
        tf_out = self.tf(e)  # (B, T, D) — parallel!
        mem_out = self.core(tf_out)  # (B, D) — sequential on T steps
        combined = tf_out[:, -1, :] + mem_out  # last position + memory
        return self.ro(combined)


class TransformerLSTM(nn.Module):
    def __init__(self, vocab=26):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.pos = nn.Parameter(torch.randn(1, 64, D)*0.02)
        self.tf = TransformerBlock()
        self.lstm = nn.LSTM(D, D, batch_first=True)
        self.ro = nn.Linear(D, vocab)

    def forward(self, x):
        B, T = x.shape
        e = self.emb(x) + self.pos[:, :T, :]
        tf_out = self.tf(e)
        lstm_out, _ = self.lstm(tf_out)
        combined = tf_out[:, -1, :] + lstm_out[:, -1, :]
        return self.ro(combined)


# ─── Hard task: replace first 5 chars with noise after showing them ───
def memory_task(vocab=26, n_targets=5, n_distractors=20, n_samples=3000):
    """Show targets, then distractors, then ask for the last target char.

    Targets are presented but REPLACED by random chars in the input,
    so the Transformer CANNOT attend to them at query time.
    Only a memory module (Core/LSTM) that saw the original sequence can answer.
    """
    X_vis, X_hid, Y = [], [], []
    for _ in range(n_samples):
        targets = [random.randint(1, vocab-1) for _ in range(n_targets)]
        distractors = [random.randint(1, vocab-1) for _ in range(n_distractors)]
        noise = [random.randint(1, vocab-1) for _ in range(n_targets)]

        # Visible input: noise replaces targets (transformer can't cheat)
        visible = noise + distractors
        # Hidden input: what actually went through the memory module (only for training)
        # For this test: Core sees visible, transformer sees visible
        # The memory must store info from early positions

        X_vis.append(visible)
        Y.append(targets[-1])  # recall the last target

    return torch.tensor(X_vis), torch.tensor(Y), vocab


def memory_task_harder(vocab=26, n_targets=5, n_distractors=20, n_samples=3000):
    """V2: targets presented normally at START, then distractors.
    Transformer CAN attend to targets, but must filter through noise.
    """
    X, Y = [], []
    for _ in range(n_samples):
        targets = [random.randint(1, vocab-1) for _ in range(n_targets)]
        distractors = [random.randint(1, vocab-1) for _ in range(n_distractors)]
        X.append(targets + distractors)
        Y.append(targets[0])  # recall first target after all distractors
    return torch.tensor(X), torch.tensor(Y), vocab


# ─── Train ───
def train_model(model, name, X, Y, V, n_steps=2000, lr=0.002):
    model = model.to(DEV); X, Y = X.to(DEV), Y.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    B = 64; accs = {}; t0 = time.time()
    for step in range(n_steps):
        idx = torch.randint(0, len(X), (B,))
        logits = model(X[idx])
        loss = F.cross_entropy(logits, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 500 == 0:
            with torch.no_grad():
                c, t = 0, 0
                for i in range(0, len(X), B):
                    bx, by = X[i:i+B], Y[i:i+B]
                    pred = model(bx.to(DEV)).argmax(-1)
                    c += (pred == by.to(DEV)).sum().item(); t += len(by)
                accs[step+1] = c/t
    dt = time.time()-t0
    return accs, dt


# ═══════════════════════
print("M4 v2: First-char recall through distractors (5 targets + 20 distractors)")
X, Y, V = memory_task_harder()
print(f"Task: {X.shape[1]} steps, vocab={V}, random baseline={1/V:.4f}")
print(f"{'Model':>25s} {'step500':>8s} {'step1000':>8s} {'step2000':>8s} {'time':>6s}")
print("-"*65)

for ModelClass, name, lr in [
    (TransformerOnly,   'A: Transformer only',     0.002),
    (TransformerCore,   'B: Transformer + Core',    0.002),
    (TransformerLSTM,   'C: Transformer + LSTM',    0.002),
]:
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    model = ModelClass(V)
    n = sum(p.numel() for p in model.parameters())
    accs, dt = train_model(model, name, X, Y, V, n_steps=2000, lr=lr)
    print(f'{name:>25s} {accs[500]:>8.4f} {accs[1000]:>8.4f} {accs[2000]:>8.4f} {dt:>5.0f}s ({n/1000:.0f}k)')
