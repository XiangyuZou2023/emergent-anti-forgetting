"""
Copy task: 逼 Core 的循环动力学真正干活

输入: [A, B, C, D, _, _, _, ?]
目标: 在 ? 位置预测 [A, B, C, D]

只有靠循环动力学记住前面的字符，emb+ro 无法靠局部统计解决。
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json, random, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from transplant_stress import Core
from neuron_clone import clone_W

DEV = torch.device('cuda')

def generate_copy_task(vocab=26, seq_len=4, delay=3, n_samples=2000):
    """Generate copy task data.

    Sequence: [c1, c2, c3, c4, 0, 0, 0, ?, ?, ?, ?]
    Target:   [c1, c2, c3, c4] at the last 4 positions
    """
    X, Y_mask, Y_val = [], [], []
    for _ in range(n_samples):
        chars = [random.randint(1, vocab-1) for _ in range(seq_len)]  # 1..vocab-1
        # Input: chars + blanks + query markers
        seq = chars + [0]*delay + [vocab-1]*seq_len  # vocab-1 = query token
        X.append(seq)
        Y_mask.append([0]*(seq_len+delay) + [1]*seq_len)  # only predict last seq_len
        Y_val.append([0]*(seq_len+delay) + chars)
    return (torch.tensor(X), torch.tensor(Y_mask), torch.tensor(Y_val), vocab)


def train_and_eval(core, n_steps=500, seq_len=4, delay=3):
    """Train emb+ro with frozen Core.W, test copy accuracy."""
    core = core.to(DEV); d = core.W.shape[0]
    core.W.requires_grad = False
    X, Y_mask, Y_val, V = generate_copy_task(seq_len=seq_len, delay=delay)
    X, Y_mask, Y_val = X.to(DEV), Y_mask.to(DEV), Y_val.to(DEV)

    emb = nn.Embedding(V, d//4).to(DEV)
    ro = nn.Linear(d, V).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters())+list(ro.parameters()), lr=0.003)
    B = 64

    accs = {}
    for step in range(n_steps):
        idx = torch.randint(0, len(X), (B,))
        bx = X[idx]       # (B, T) where T=seq_len+delay+seq_len
        bm = Y_mask[idx]  # (B, T)
        by = Y_val[idx]   # (B, T)

        T = bx.shape[1]
        # Process sequential input through Core
        e = emb(bx)  # (B, T, d//4)
        e_flat = e.reshape(B, T, d//4)
        h = torch.zeros(B, d, device=DEV)
        for t in range(T):
            # Feed one position at a time
            h = core.step(h, e_flat[:, t, :].repeat(1,4))

        # Readout: only predict at masked positions
        logits = ro(h)  # (B, V)
        # Only last position matters for copy — h should store all seq info
        # Actually we need per-position prediction
        # Simpler: h after full sequence should encode all chars
        # Use h to predict each target char independently via separate readouts
        loss = F.cross_entropy(logits, by[:, -1])  # just last target for now
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step+1 in [20, 50, 100, 200, 300, 400, 500]:
            with torch.no_grad():
                correct, total = 0, 0
                for i in range(0, len(X), B):
                    bx_b = X[i:i+B]; bm_b = Y_mask[i:i+B]; by_b = Y_val[i:i+B]
                    e_b = emb(bx_b).reshape(bx_b.shape[0], bx_b.shape[1], d//4)
                    h_b = torch.zeros(bx_b.shape[0], d, device=DEV)
                    for t in range(bx_b.shape[1]):
                        h_b = core.step(h_b, e_b[:, t, :].repeat(1,4))
                    logits_b = ro(h_b)
                    pred = logits_b.argmax(-1)
                    correct += (pred == by_b[:, -1]).sum().item()
                    total += by_b.shape[0]
                accs[step+1] = correct / total
    core.W.requires_grad = True
    return accs


def main():
    with open(os.path.join(SCRIPT_DIR, 'runs', 'transplant_stress', 'best_genome_D64.json')) as f:
        gd = json.load(f)
    sr, gain, sp = gd['sr'], gd['gain'], gd['sp']
    print(f"Genome: sr={sr:.4f} gain={gain:.4f} sp={sp:.4f}")

    # Train D=64 on copy task
    print("\n=== Training D=64 on copy task ===")
    core64 = Core(64, sr, gain, sp, w_seed=42).to(DEV)
    X, Y_mask, Y_val, V = generate_copy_task()
    X, Y_mask, Y_val = X.to(DEV), Y_mask.to(DEV), Y_val.to(DEV)
    emb64 = nn.Embedding(V, 16).to(DEV); ro64 = nn.Linear(64, V).to(DEV)
    opt64 = torch.optim.Adam(list(emb64.parameters())+list(ro64.parameters())+list(core64.parameters()), lr=0.003)
    B = 64
    d64_accs = {}
    for step in range(500):
        idx = torch.randint(0, len(X), (B,))
        bx = X[idx]; bm = Y_mask[idx]; by = Y_val[idx]
        e = emb64(bx).reshape(bx.shape[0], bx.shape[1], 16)
        h = torch.zeros(B, 64, device=DEV)
        for t in range(bx.shape[1]):
            h = core64.step(h, e[:, t, :].repeat(1,4))
        logits = ro64(h); loss = F.cross_entropy(logits, by[:, -1])
        opt64.zero_grad(); loss.backward(); opt64.step()
        if step+1 in [20, 50, 100, 200, 300, 400, 500]:
            with torch.no_grad():
                c,t=0,0
                for i in range(0, len(X), B):
                    bx_b=X[i:i+B]; by_b=Y_val[i:i+B]
                    e_b=emb64(bx_b).reshape(bx_b.shape[0],bx_b.shape[1],16)
                    h_b=torch.zeros(bx_b.shape[0],64,device=DEV)
                    for tt in range(bx_b.shape[1]): h_b=core64.step(h_b, e_b[:,tt,:].repeat(1,4))
                    pred=ro64(h_b).argmax(-1); c+=(pred==by_b[:,-1]).sum().item(); t+=by_b.shape[0]
                d64_accs[step+1]=c/t

    W_64_trained = core64.W.detach().cpu().numpy()
    core64 = core64.cpu()

    # Clone to D=256
    np.random.seed(42)
    # Use preserved version (no divide by 16)
    d_src, factor = 64, 4
    W_clone = np.zeros((256, 256), dtype=np.float32)
    for i in range(d_src):
        for j in range(d_src):
            w = W_64_trained[i, j]
            if abs(w) < 1e-8: continue
            for di in range(factor):
                for dj in range(factor):
                    noise = np.random.randn() * 0.03 * abs(w)
                    W_clone[i*factor+di, j*factor+dj] = w + noise
    # Normalize
    s = np.linalg.norm(W_clone, 2); W_clone = W_clone * (sr / (s+1e-8))

    core_clone = Core(256, sr, gain, sp, w_seed=42)
    core_clone.W.data = torch.tensor(W_clone, dtype=torch.float32)
    core_random = Core(256, sr, gain, sp, w_seed=999)

    print(f'\n=== D=256 FROZEN W: Copy Task ===')
    print(f'{"step":>6s} {"D64_train":>12s} {"B1_clone":>10s} {"Random":>10s}')
    acc_clone = train_and_eval(core_clone)  # with delay=3
    acc_rand = train_and_eval(core_random)
    for step in [20, 50, 100, 200, 300, 400, 500]:
        a64 = d64_accs.get(step, 0)
        ac = acc_clone[step]; ar = acc_rand[step]
        print(f'{step:>6d} {a64:>12.4f} {ac:>10.4f} {ar:>10.4f}  Δ={ac-ar:+.4f}')


if __name__ == '__main__':
    main()
