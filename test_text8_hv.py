"""
text8 English char prediction: standard evolution vs high-variance (ADHD simulation)
Tests whether high exploration closes the gap between single-model (0.36) and MoE (0.94)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random, time, os
from collections import Counter

DEV = torch.device('cuda')
D = 64

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'text8')
with open(data_path, 'r', encoding='utf-8') as f:
    raw = f.read(300000)
cnt = Counter(raw)
all_chars = sorted(cnt.keys())
print(f'text8 chars: {len(all_chars)} -> {all_chars}')

def make_taskAB(vs):
    chars = ['<PAD>', '<UNK>'] + all_chars[:vs]
    c2i = {c: i for i, c in enumerate(chars)}
    ids = [c2i.get(c, 1) for c in raw][:20000]
    Xs, Ya, Yb = [], [], []
    for i in range(len(ids) - 6):
        Xs.append(ids[i:i + 4]); Ya.append(ids[i + 4]); Yb.append(ids[i + 5])
    return torch.tensor(Xs), torch.tensor(Ya), torch.tensor(Yb), len(chars)

class Core(nn.Module):
    def __init__(self, sr=0.9, gain=0.1, sp=0.5):
        super().__init__()
        W = torch.randn(D, D) * 0.3
        mask = (torch.rand(D, D) > sp).float(); W = W * mask
        s = torch.linalg.norm(W, 2)
        self.W = nn.Parameter(W * (sr / (s + 1e-8)))
        self.b = nn.Parameter(torch.zeros(D)); self.g = nn.Parameter(torch.tensor(gain))
        self.inp = nn.Linear(D, D, bias=False)
    def step(self, h, x=None):
        h = h.detach()
        ext = self.inp(x) if x is not None else torch.zeros_like(h)
        return 0.9 * torch.tanh(h @ self.W.T + self.b + ext) + 0.1 * self.g * h + torch.randn_like(h) * 0.005

def fast_A(core, h, n=40):
    for _ in range(10): h = core.step(h)
    tr = []
    for _ in range(n):
        h = core.step(h); tr.append(h / (h.norm(dim=-1, keepdim=True) + 1e-8))
    ds = [.5 * (tr[i] - tr[i+1]).norm(dim=-1).mean().item() for i in range(len(tr)-1)]
    S = 1 - min(np.mean(ds), 1.)
    return 4 * (1 - S) * S / 0.3

class Genome:
    def __init__(self, sr=0.9, gain=0.1, sp=0.5):
        self.sr = np.clip(sr + random.uniform(-0.05, 0.05), 0.1, 2.0)
        self.gain = np.clip(gain + random.uniform(-0.05, 0.05), 0.0, 1.0)
        self.sp = np.clip(sp + random.uniform(-0.1, 0.1), 0.0, 0.9)
    def mutate(self, prob=0.4, scale=0.2):
        if random.random() < prob:
            self.sr = np.clip(self.sr + random.uniform(-scale, scale), 0.1, 2.0)
        if random.random() < prob:
            self.gain = np.clip(self.gain + random.uniform(-scale, scale), 0.0, 1.0)
        if random.random() < prob:
            self.sp = np.clip(self.sp + random.uniform(-scale, scale), 0.0, 0.9)
    def build(self): return Core(self.sr, self.gain, self.sp).to(DEV)
    @staticmethod
    def crossover(a, b):
        c = Genome()
        c.sr = a.sr if random.random() < 0.5 else b.sr
        c.gain = a.gain if random.random() < 0.5 else b.gain
        c.sp = a.sp if random.random() < 0.5 else b.sp
        return c

def measure(ind, X, YA, YB, vocab):
    core = ind.build()
    h = torch.randn(4, D, device=DEV) * 0.5
    A = fast_A(core, h, 40)
    if A < 1.0: return A, -1.0, 0, 0, 0, 0
    emb = nn.Embedding(vocab, D // 4).to(DEV); ro = nn.Linear(D, vocab).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ro.parameters()), lr=0.01)
    def te(tX, tY, n=200):
        perm = torch.randperm(len(tX))[:n]
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]; bx = tX[idx].to(DEV); by = tY[idx].to(DEV)
            e = emb(bx).flatten(1); hh = torch.zeros(len(bx), D, device=DEV)
            for _ in range(3): hh = core.step(hh, e)
            loss = F.cross_entropy(ro(hh), by); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tidx = torch.randint(0, len(tX), (200,)); bx = tX[tidx].to(DEV); by = tY[tidx].to(DEV)
            e = emb(bx).flatten(1); hh = torch.zeros(len(bx), D, device=DEV)
            for _ in range(3): hh = core.step(hh, e)
            return (ro(hh).argmax(-1) == by).float().mean().item()
    aA = te(X, YA, 200); aB = te(X, YB, 200); aR = te(X, YA, 50)
    w = aA * 0.2 + aB * 0.2 + aR * 0.6
    hc = ind.sr * D * (1 - ind.sp) + ind.gain * 10
    lp = 0
    if hc > 30: lp = (hc - 30) * 0.003
    elif hc < 8: lp = (8 - hc) * 0.01
    return A, w - lp, aA, aB, aR, hc

def run(label, mut_prob, mut_scale, fit_pct, random_inject, seed):
    POP, GEN = 40, 60
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    pop = [Genome() for _ in range(POP)]
    vocab = 5
    for gen in range(GEN):
        X, YA, YB, V = make_taskAB(vocab)
        scores, As, accAs, accBs, retAs, hs = [], [], [], [], [], []
        for ind in pop:
            A_vec, fit, aA, aB, aR, hc = measure(ind, X, YA, YB, V)
            scores.append(fit); As.append(A_vec); accAs.append(aA)
            accBs.append(aB); retAs.append(aR); hs.append(hc)

        alive_rate = sum(1 for a in As if a >= 1.0) / POP * 100
        med_ret = np.median(retAs)

        fit_cutoff = np.percentile([s for s in scores if s >= 0], fit_pct)
        lean_hi, lean_lo = np.percentile(hs, 70), np.percentile(hs, 10)
        keep = [i for i, s in enumerate(scores) if As[i] >= 1.0
                and s >= fit_cutoff and lean_lo <= hs[i] <= lean_hi]
        if len(keep) < POP * 0.2: keep = list(range(POP))

        if alive_rate > 40 and med_ret > 0.7 and 10 < np.median(hs) < 40:
            vocab = min(len(all_chars), vocab + 2)
        elif alive_rate < 15 or med_ret < 0.3:
            vocab = max(5, vocab - 2)

        survivors = [pop[i] for i in keep]
        new_pop = [pop[i] for i in keep[:max(1, int(len(keep) * 0.25))]]
        while len(new_pop) < POP:
            if len(survivors) < 2: break
            t1 = random.sample(range(len(survivors)), min(3, len(survivors)))
            t2 = random.sample(range(len(survivors)), min(3, len(survivors)))
            c = Genome.crossover(survivors[t1[0]], survivors[t2[0]])
            c.mutate(mut_prob, mut_scale); new_pop.append(c)
        for _ in range(random_inject):
            if len(new_pop) >= POP: break
            new_pop.append(Genome())
        while len(new_pop) < POP: new_pop.append(Genome())
        pop = new_pop[:POP]

    return {'label': label, 'vocab': vocab, 'alive': alive_rate,
            'aA': np.median(accAs), 'aB': np.median(accBs), 'rA': np.median(retAs),
            'fg': np.median(accAs) - np.median(retAs),
            'A': np.median([a for a in As if a >= 0] or [0])}

# ═══ RUN ═══
SEEDS = [42, 123]
variants = [
    ('standard',      0.40, 0.20, 30, 0),
    ('high_var+cut40', 0.70, 0.40, 40, 5),
]
print(f'text8 English char prediction (27 chars)')
print(f'{"label":<18s} {"vocab":>5s} {"accA":>7s} {"accB":>7s} {"retA":>7s} {"forget":>8s} {"A*":>6s} {"alive%":>6s}')
print('-' * 72)
for label, mp, ms, pc, rj in variants:
    rs = [run(label, mp, ms, pc, rj, s) for s in SEEDS]
    avg = lambda k: np.mean([r[k] for r in rs])
    print(f'{label:<18s} {avg("vocab"):>5.0f} {avg("aA"):>7.3f} {avg("aB"):>7.3f} '
          f'{avg("rA"):>7.3f} {avg("fg"):>+8.3f} {avg("A"):>6.3f} {avg("alive"):>5.0f}%')
