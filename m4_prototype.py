"""
M4: 可训练 Core 作为 Transformer 记忆模块

对比:
  A: 纯 Transformer (baseline)
  B: Transformer + Core Memory (我们的)
  C: Transformer + LSTM (传统方案)

任务: delayed recall — 必须跨时间步记住信息
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, random, time

DEV = torch.device('cuda')

# ═══════════════════════════════════════
# Core Memory Module (trainable sr/gain)
# ═══════════════════════════════════════
class CoreMemory(nn.Module):
    def __init__(self, d=128, sr=1.5, gain=0.1, sp=0.5):
        super().__init__()
        W = torch.randn(d,d)*0.3; mask = (torch.rand(d,d)>sp).float(); W = W*mask
        s = torch.linalg.norm(W,2)
        self.W = nn.Parameter(W*(sr/(s+1e-8)))
        self.b = nn.Parameter(torch.zeros(d))
        self.g = nn.Parameter(torch.tensor(gain))
        # Input gate: learn what to write
        self.write_gate = nn.Linear(d, d)
        # Output gate: learn what to read
        self.read_gate = nn.Linear(d, d)

    def forward_step(self, h, x):
        """One step: write x into h, return memory-augmented output."""
        h = h.detach()  # truncated BPTT
        gate_in = torch.sigmoid(self.write_gate(x))
        ext = gate_in * x
        h_new = 0.9 * torch.tanh(h @ self.W.T + self.b + ext) + 0.1 * self.g * h
        gate_out = torch.sigmoid(self.read_gate(h_new))
        return h_new, gate_out * h_new

    def forward_sequence(self, h0, x_seq):
        """Batch process entire sequence: h0 (B,d), x_seq (B,T,d) → h_final (B,d)."""
        B, T, d = x_seq.shape
        h = h0
        for t in range(T):
            h, _ = self.forward_step(h, x_seq[:, t, :])
        return h


# ═══════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════
class TransformerBlock(nn.Module):
    def __init__(self, d=128, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, causal=False):
        # x: (B, T, d)
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device)*float('-inf'), diagonal=1) if causal else None
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ═══════════════════════════════════════
# Model variants
# ═══════════════════════════════════════
class BaseModel(nn.Module):
    def __init__(self, vocab=26, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, 32, d)*0.02)
        self.transformer = TransformerBlock(d)
        self.ro = nn.Linear(d, vocab)

    def forward(self, x):
        B, T = x.shape
        e = self.emb(x) + self.pos[:, :T, :]
        h = self.transformer(e, causal=True)
        return self.ro(h[:, -1, :])  # predict from last position


class CoreModel(nn.Module):
    """Transformer + Core Memory in residual stream."""
    def __init__(self, vocab=26, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, 32, d)*0.02)
        self.transformer = TransformerBlock(d)
        self.core = CoreMemory(d)
        self.ro = nn.Linear(d, vocab)

    def forward(self, x):
        B, T = x.shape
        d = self.emb.embedding_dim
        e = self.emb(x) + self.pos[:, :T, :]
        h_mem = torch.zeros(B, d, device=x.device)  # Core state
        outputs = []
        for t in range(T):
            # Transformer: process current + context (last 4 positions)
            ctx_start = max(0, t-3)
            ctx = e[:, ctx_start:t+1, :]
            trans_out = self.transformer(ctx, causal=True)[:, -1, :]  # (B, d)
            # Core: update memory, get readout
            h_mem, mem_out = self.core.forward_step(h_mem, trans_out)
            # Combine
            combined = trans_out + mem_out
            outputs.append(combined)
        out = torch.stack(outputs, dim=1)  # (B, T, d)
        return self.ro(out[:, -1, :])


class LSTMModel(nn.Module):
    """Transformer + LSTM baseline."""
    def __init__(self, vocab=26, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, 32, d)*0.02)
        self.transformer = TransformerBlock(d)
        self.lstm = nn.LSTMCell(d, d)
        self.ro = nn.Linear(d, vocab)

    def forward(self, x):
        B, T = x.shape
        d = self.emb.embedding_dim
        e = self.emb(x) + self.pos[:, :T, :]
        h_mem = torch.zeros(B, d, device=x.device)
        c_mem = torch.zeros(B, d, device=x.device)
        for t in range(T):
            ctx_start = max(0, t-3)
            ctx = e[:, ctx_start:t+1, :]
            trans_out = self.transformer(ctx, causal=True)[:, -1, :]
            h_mem, c_mem = self.lstm(trans_out, (h_mem, c_mem))
        return self.ro(h_mem)


# ═══════════════════════════════════════
# Data
# ═══════════════════════════════════════
def delayed_recall(vocab=26, seq_len=5, delay=20, n_samples=3000, causal_mask=False):
    Xd, Yd = [], []
    for _ in range(n_samples):
        chars = [random.randint(1, vocab-1) for _ in range(seq_len)]
        Xd.append(chars + [0]*delay)
        Yd.append(chars[0])
    return torch.tensor(Xd), torch.tensor(Yd), vocab


# ═══════════════════════════════════════
# Train
# ═══════════════════════════════════════
def train_model(model, name, n_steps=3000, lr=0.002):
    model = model.to(DEV)
    X, Y, V = delayed_recall()
    X, Y = X.to(DEV), Y.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    B = 64
    accs = {}
    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randint(0, len(X), (B,))
        bx, by = X[idx], Y[idx]
        logits = model(bx)
        loss = F.cross_entropy(logits, by)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 500 == 0:
            with torch.no_grad():
                c, t = 0, 0
                for i in range(0, len(X), B):
                    bx_b, by_b = X[i:i+B], Y[i:i+B]
                    pred = model(bx_b.to(DEV)).argmax(-1)
                    c += (pred == by_b.to(DEV)).sum().item(); t += len(by_b)
                accs[step+1] = c/t
    dt = time.time() - t0
    return accs, dt


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
print("M4 Prototype: Delayed Recall (5 chars + 5 blanks)")
print(f"{'Model':>25s} {'step500':>8s} {'step1500':>8s} {'step3000':>8s} {'time':>6s}")
print("-"*65)

for ModelClass, name, lr in [
    (BaseModel,  'A: Transformer only', 0.002),
    (CoreModel,  'B: Transformer+Core', 0.002),
    (LSTMModel,  'C: Transformer+LSTM', 0.002),
]:
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    model = ModelClass()
    n_params = sum(p.numel() for p in model.parameters())
    accs, dt = train_model(model, name, n_steps=3000, lr=lr)
    print(f'{name:>25s} {accs[500]:>8.4f} {accs[1500]:>8.4f} {accs[3000]:>8.4f} {dt:>5.0f}s ({n_params/1000:.0f}k)')
