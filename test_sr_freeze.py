"""
Test: Does anti-forgetting depend on spectral radius evolution?
Freezes rho at initialization (~0.9), lets gain/sparsity evolve freely.
Result: anti-forgetting survives, vocabulary actually INCREASES (42->70).
Supports the claim that the effect is not a parameter-tuning artifact.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random, time, os
from collections import Counter

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D, POP, GEN = 64, 40, 40

with open(os.path.join(os.path.dirname(__file__), 'sample_data.txt'), 'r', encoding='utf-8') as f:
    raw = f.read(300000)
cnt = Counter(raw)
all_chars = sorted([c for c, n in cnt.items() if n >= 5 and '一' <= c <= '鿿'])

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
        W = torch.randn(D, D) * 0.3; mask = (torch.rand(D, D) > sp).float(); W = W * mask
        s = torch.linalg.norm(W, 2); self.W = nn.Parameter(W * (sr / (s + 1e-8)))
        self.b = nn.Parameter(torch.zeros(D)); self.g = nn.Parameter(torch.tensor(gain))
        self.inp = nn.Linear(D, D, bias=False)
    def step_noisy(self, h, x=None):
        h = h.detach(); ext = self.inp(x) if x is not None else torch.zeros_like(h)
        return 0.9 * torch.tanh(h @ self.W.T + self.b + ext) + 0.1 * self.g * h + torch.randn_like(h) * 0.005
    def step_det(self, h, x=None):
        h = h.detach(); ext = self.inp(x) if x is not None else torch.zeros_like(h)
        return 0.9 * torch.tanh(h @ self.W.T + self.b + ext) + 0.1 * self.g * h

def fast_A_det(core, h, n=40):
    for _ in range(10): h = core.step_det(h)
    tr = []
    for _ in range(n): h = core.step_det(h); tr.append(h / (h.norm(dim=-1, keepdim=True) + 1e-8))
    ds = [.5 * (tr[i] - tr[i+1]).norm(dim=-1).mean().item() for i in range(len(tr)-1)]
    S = 1 - min(np.mean(ds), 1.); return 4 * (1 - S) * S / 0.3

class Genome:
    def __init__(self, sr=0.9, gain=0.1, sp=0.5):
        self.sr = np.clip(sr + random.uniform(-0.05, 0.05), 0.1, 2.0)
        self.gain = np.clip(gain + random.uniform(-0.05, 0.05), 0.0, 1.0)
        self.sp = np.clip(sp + random.uniform(-0.1, 0.1), 0.0, 0.9)
    def mutate(self, prob=0.4, scale=0.2, freeze_sr=False):
        if not freeze_sr:
            if random.random() < prob: self.sr = np.clip(self.sr + random.uniform(-scale, scale), 0.1, 2.0)
        if random.random() < prob: self.gain = np.clip(self.gain + random.uniform(-scale, scale), 0.0, 1.0)
        if random.random() < prob: self.sp = np.clip(self.sp + random.uniform(-scale, scale), 0.0, 0.9)
    def build(self): return Core(self.sr, self.gain, self.sp).to(DEV)
    @staticmethod
    def crossover(a, b, freeze_sr=False):
        c = Genome()
        c.sr = a.sr if freeze_sr else (a.sr if random.random() < 0.5 else b.sr)
        c.gain = a.gain if random.random() < 0.5 else b.gain
        c.sp = a.sp if random.random() < 0.5 else b.sp
        return c

def measure(ind, X, YA, YB, vocab):
    core = ind.build()
    h = torch.randn(4, D, device=DEV) * 0.5; A = fast_A_det(core, h, 40)
    if A < 1.0: return A, -1.0, 0, 0, 0, 0
    emb = nn.Embedding(vocab, D // 4).to(DEV); ro = nn.Linear(D, vocab).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ro.parameters()), lr=0.01)
    def te(tX, tY, n=200):
        perm = torch.randperm(len(tX))[:n]
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]; bx = tX[idx].to(DEV); by = tY[idx].to(DEV)
            e = emb(bx).flatten(1); hh = torch.zeros(len(bx), D, device=DEV)
            for _ in range(3): hh = core.step_noisy(hh, e)
            loss = F.cross_entropy(ro(hh), by); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tidx = torch.randint(0, len(tX), (200,)); bx = tX[tidx].to(DEV); by = tY[tidx].to(DEV)
            e = emb(bx).flatten(1); hh = torch.zeros(len(bx), D, device=DEV)
            for _ in range(3): hh = core.step_noisy(hh, e)
            return (ro(hh).argmax(-1) == by).float().mean().item()
    aA = te(X, YA, 200); aB = te(X, YB, 200); aR = te(X, YA, 50)
    w = aA * 0.2 + aB * 0.2 + aR * 0.6; hc = ind.sr * D * (1 - ind.sp) + ind.gain * 10
    lp = 0
    if hc > 30: lp = (hc - 30) * 0.003
    elif hc < 8: lp = (8 - hc) * 0.01
    return A, w - lp, aA, aB, aR, hc

def run(label, freeze_sr, seed):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    pop = [Genome() for _ in range(POP)]
    vocab = 10
    for gen in range(GEN):
        X, YA, YB, V = make_taskAB(vocab)
        scores, As, accAs, accBs, retAs, hs, srs = [], [], [], [], [], [], []
        for ind in pop:
            A, fit, aA, aB, aR, hc = measure(ind, X, YA, YB, V)
            scores.append(fit); As.append(A); accAs.append(aA); accBs.append(aB)
            retAs.append(aR); hs.append(hc); srs.append(ind.sr)
        alive_rate = sum(1 for a in As if a >= 1.0) / POP * 100
        med_ret = np.median(retAs)
        fit_cutoff = np.percentile([s for s in scores if s >= 0], 30)
        lean_hi, lean_lo = np.percentile(hs, 70), np.percentile(hs, 10)
        keep = [i for i, s in enumerate(scores) if As[i] >= 1.0 and s >= fit_cutoff and lean_lo <= hs[i] <= lean_hi]
        if len(keep) < POP * 0.2: keep = list(range(POP))
        if alive_rate > 40 and med_ret > 0.7 and 10 < np.median(hs) < 40:
            vocab = min(200, vocab + 5)
        elif alive_rate < 15 or med_ret < 0.3:
            vocab = max(10, vocab - 5)
        survivors = [pop[i] for i in keep]
        new_pop = [pop[i] for i in keep[:max(1, int(len(keep) * 0.25))]]
        while len(new_pop) < POP:
            if len(survivors) < 2: break
            t1 = random.sample(range(len(survivors)), min(3, len(survivors)))
            t2 = random.sample(range(len(survivors)), min(3, len(survivors)))
            c = Genome.crossover(survivors[t1[0]], survivors[t2[0]], freeze_sr)
            c.mutate(0.4, 0.2, freeze_sr); new_pop.append(c)
        while len(new_pop) < POP: new_pop.append(Genome())
        pop = new_pop[:POP]
    return {'label': label, 'vocab': vocab, 'alive': alive_rate,
            'sr': np.median(srs), 'A': np.median([a for a in As if a >= 0] or [0]),
            'aA': np.median(accAs), 'aB': np.median(accBs), 'rA': np.median(retAs),
            'fg': np.median(accAs) - np.median(retAs)}

if __name__ == '__main__':
    SEEDS = [42, 123, 777]
    variants = [('sr_free (full)', False), ('sr_frozen', True)]
    print(f'{"label":<18s} {"vocab":>5s} {"sr":>6s} {"A*":>6s} {"accA":>6s} {"retA":>6s} {"forget":>8s} {"alive%":>6s}')
    print('-' * 72)
    for label, freeze in variants:
        rs = [run(label, freeze, s) for s in SEEDS]
        avg = lambda k: np.mean([r[k] for r in rs])
        print(f'{label:<18s} {avg("vocab"):>5.0f} {avg("sr"):>6.3f} {avg("A"):>6.3f} '
              f'{avg("aA"):>6.3f} {avg("rA"):>6.3f} {avg("fg"):>+8.3f} {avg("alive"):>5.0f}%')
