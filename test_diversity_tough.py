"""
Test: diversity + tougher selection. Can diversity raise the ceiling?
Compares: standard(no div,cut30%) vs div3/cut30% vs div5/cut40% vs div5/cut50%
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random, time, os
from collections import Counter

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D, POP, GEN = 64, 40, 40

with open(os.path.join(os.path.dirname(__file__), 'sample_data.txt'), 'r', encoding='utf-8') as f:
    raw = f.read(300_000)
cnt = Counter(raw)
all_chars = sorted([c for c, n in cnt.items() if n >= 5 and '一' <= c <= '鿿'])

def make_taskAB(vs):
    chars = ['<PAD>', '<UNK>'] + all_chars[:vs]
    c2i = {c: i for i, c in enumerate(chars)}
    ids = [c2i.get(c, 1) for c in raw if '一' <= c <= '鿿'][:20000]
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
    def step_noisy(self, h, x=None):
        h = h.detach()
        ext = self.inp(x) if x is not None else torch.zeros_like(h)
        return 0.9 * torch.tanh(h @ self.W.T + self.b + ext) + 0.1 * self.g * h + torch.randn_like(h) * 0.005
    def step_det(self, h, x=None):
        h = h.detach()
        ext = self.inp(x) if x is not None else torch.zeros_like(h)
        return 0.9 * torch.tanh(h @ self.W.T + self.b + ext) + 0.1 * self.g * h

def fast_A_det(core, h, n=40):
    for _ in range(10): h = core.step_det(h)
    tr = []
    for _ in range(n):
        h = core.step_det(h); tr.append(h / (h.norm(dim=-1, keepdim=True) + 1e-8))
    ds = [.5 * (tr[i] - tr[i+1]).norm(dim=-1).mean().item() for i in range(len(tr)-1)]
    S = 1 - min(np.mean(ds), 1.)
    return 4 * (1 - S) * S / 0.3

class Genome:
    def __init__(self, sr=0.9, gain=0.1, sp=0.5):
        self.sr = np.clip(sr + random.uniform(-0.05, 0.05), 0.1, 2.0)
        self.gain = np.clip(gain + random.uniform(-0.05, 0.05), 0.0, 1.0)
        self.sp = np.clip(sp + random.uniform(-0.1, 0.1), 0.0, 0.9)
    def mutate(self, r=0.3):
        if random.random() < r: self.sr = np.clip(self.sr + random.uniform(-0.2, 0.2), 0.1, 2.0)
        if random.random() < r: self.gain = np.clip(self.gain + random.uniform(-0.2, 0.2), 0.0, 1.0)
        if random.random() < r: self.sp = np.clip(self.sp + random.uniform(-0.2, 0.2), 0.0, 0.9)
    def build(self): return Core(self.sr, self.gain, self.sp).to(DEV)
    @staticmethod
    def crossover(a, b):
        c = Genome()
        c.sr = a.sr if random.random() < 0.5 else b.sr
        c.gain = a.gain if random.random() < 0.5 else b.gain
        c.sp = a.sp if random.random() < 0.5 else b.sp
        return c

def niche_grid(pop, bins=5):
    se = np.linspace(0.1, 2.0, bins + 1)
    ge = np.linspace(0.0, 1.0, bins + 1)
    pe = np.linspace(0.0, 0.9, bins + 1)
    niche = {}
    for i, ind in enumerate(pop):
        sb = min(bins-1, np.digitize(ind.sr, se) - 1)
        gb = min(bins-1, np.digitize(ind.gain, ge) - 1)
        pb = min(bins-1, np.digitize(ind.sp, pe) - 1)
        niche.setdefault(f'{sb}_{gb}_{pb}', []).append(i)
    return niche

def measure(ind, X, YA, YB, vocab):
    core = ind.build()
    h = torch.randn(4, D, device=DEV) * 0.5
    A_star = fast_A_det(core, h, 40)
    if A_star < 1.0: return A_star, -1.0, 0, 0, 0, 0
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
    w = aA * 0.2 + aB * 0.2 + aR * 0.6
    hc = ind.sr * D * (1 - ind.sp) + ind.gain * 10
    lp = 0
    if hc > 30: lp = (hc - 30) * 0.003
    elif hc < 8: lp = (8 - hc) * 0.01
    return A_star, w - lp, aA, aB, aR, hc

def run_exp(label, niche_grace, fit_pct, seed):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    pop = [Genome() for _ in range(POP)]
    vocab = 10; endangered = {}
    for gen in range(GEN):
        X, YA, YB, V = make_taskAB(vocab)
        scores, As, accAs, accBs, retAs, hs, srs = [], [], [], [], [], [], []
        for ind in pop:
            A, fit, aA, aB, aR, hc = measure(ind, X, YA, YB, V)
            scores.append(fit); As.append(A); accAs.append(aA); accBs.append(aB)
            retAs.append(aR); hs.append(hc); srs.append(ind.sr)

        alive_rate = sum(1 for a in As if a >= 1.0) / POP * 100
        med_ret = np.median(retAs)

        fit_cutoff = np.percentile([s for s in scores if s >= 0], fit_pct)
        lean_hi = np.percentile(hs, 70); lean_lo = np.percentile(hs, 10)
        standard_keep = [i for i, s in enumerate(scores)
                         if As[i] >= 1.0 and s >= fit_cutoff
                         and lean_lo <= hs[i] <= lean_hi]

        rescued = set()
        if niche_grace > 0:
            niche = niche_grid(pop)
            for nid, members in niche.items():
                if len(members) == 1:
                    i = members[0]
                    if i not in standard_keep:
                        endangered[id(pop[i])] = endangered.get(id(pop[i]), 0) + 1
                        if endangered[id(pop[i])] < niche_grace:
                            rescued.add(i)

        keep = sorted(set(standard_keep) | rescued)
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
            c = Genome.crossover(survivors[t1[0]], survivors[t2[0]]); c.mutate(0.4)
            new_pop.append(c)
        while len(new_pop) < POP: new_pop.append(Genome())
        pop = new_pop

    return {'label': label, 'vocab': vocab, 'alive': alive_rate,
            'sr': round(np.median(srs), 3),
            'A': round(np.median([a for a in As if a >= 0] or [0]), 3),
            'aA': round(np.median(accAs), 3), 'aB': round(np.median(accBs), 3),
            'rA': round(np.median(retAs), 3),
            'fg': round(np.median(accAs) - np.median(retAs), 3)}

# ========================================
variants = [
    ('standard(30%)',   0, 30),
    ('div3+cut30%',     3, 30),
    ('div5+cut40%',     5, 40),
    ('div5+cut50%',     5, 50),
]
SEEDS = [42, 123]

print(f'{"label":<18s} {"vocab":>5s} {"sr":>6s} {"A*":>6s} {"accA":>6s} {"retA":>6s} {"forget":>7s} {"alive%":>6s}')
print('-' * 70)
for label, grace, pct in variants:
    results = []
    for seed in SEEDS:
        r = run_exp(label, grace, pct, seed)
        results.append(r)
    avg = lambda k: np.mean([r[k] for r in results])
    print(f'{label:<18s} {avg("vocab"):>5.0f} {avg("sr"):>6.3f} {avg("A"):>6.3f} '
          f'{avg("aA"):>6.3f} {avg("rA"):>6.3f} {avg("fg"):>+7.3f} {avg("alive"):>5.0f}%')
