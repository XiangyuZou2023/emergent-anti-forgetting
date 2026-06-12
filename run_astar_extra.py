"""Run A* gating experiment for additional seeds (456, 789, 111).
Directly imports needed components, no text manipulation."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random, time
from collections import Counter

DEV = torch.device('cuda')
D = 64
POP, GEN = 40, 40
EXTRA_SEEDS = [456, 789, 111]

# ── Data ──
with open(os.path.join(os.path.dirname(__file__), 'sample_data.txt'), 'r', encoding='utf-8') as f:
    raw = f.read(300_000)
cnt = Counter(raw)
all_chars = sorted([c for c, n in cnt.items() if n >= 5 and '一' <= c <= '鿿'])

def make_taskAB(vocab_size):
    chars = ['<PAD>', '<UNK>'] + all_chars[:vocab_size]
    c2i = {c: i for i, c in enumerate(chars)}
    ids = [c2i.get(c, 1) for c in raw if '一' <= c <= '鿿'][:20000]
    Xs, Ya, Yb = [], [], []
    for i in range(len(ids) - 6):
        Xs.append(ids[i:i + 4]); Ya.append(ids[i + 4]); Yb.append(ids[i + 5])
    return torch.tensor(Xs), torch.tensor(Ya), torch.tensor(Yb), len(chars)

# ── Model ──
class Core(nn.Module):
    def __init__(self, sr=0.9, gain=0.1, sp=0.5):
        super().__init__()
        W = torch.randn(D, D) * 0.3
        mask = (torch.rand(D, D) > sp).float(); W = W * mask
        s = torch.linalg.norm(W, 2)
        self.W = nn.Parameter(W * (sr / (s + 1e-8)))
        self.b = nn.Parameter(torch.zeros(D))
        self.g = nn.Parameter(torch.tensor(gain))
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
        c = Genome(); c.sr = a.sr if random.random() < 0.5 else b.sr
        c.gain = a.gain if random.random() < 0.5 else b.gain
        c.sp = a.sp if random.random() < 0.5 else b.sp; return c

def measure(ind, X, YA, YB, vocab, use_astar):
    core = ind.build()
    h = torch.randn(4, D, device=DEV) * 0.5
    A_star = fast_A_det(core, h, 40)
    if use_astar and A_star < 1.0:
        return A_star, -1.0, 0, 0, 0, 0
    emb = nn.Embedding(vocab, D // 4).to(DEV); ro = nn.Linear(D, vocab).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ro.parameters()), lr=0.01)
    def train_eval(tX, tY, n_train=200):
        perm = torch.randperm(len(tX))[:n_train]
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]; bx = tX[idx].to(DEV); by = tY[idx].to(DEV)
            e = emb(bx).flatten(1); hh = torch.zeros(len(bx), D, device=DEV)
            for _ in range(3): hh = core.step_noisy(hh, e)
            loss = F.cross_entropy(ro(hh), by); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tidx = torch.randint(0, len(tX), (200,))
            bx = tX[tidx].to(DEV); by = tY[tidx].to(DEV)
            e = emb(bx).flatten(1); hh = torch.zeros(len(bx), D, device=DEV)
            for _ in range(3): hh = core.step_noisy(hh, e)
            return (ro(hh).argmax(-1) == by).float().mean().item()
    acc_A = train_eval(X, YA, 200); acc_B = train_eval(X, YB, 200); acc_A_ret = train_eval(X, YA, 50)
    w_acc = acc_A * 0.2 + acc_B * 0.2 + acc_A_ret * 0.6
    h_complexity = ind.sr * D * (1 - ind.sp) + ind.gain * 10
    lean_penalty = 0
    if h_complexity > 30: lean_penalty = (h_complexity - 30) * 0.003
    elif h_complexity < 8: lean_penalty = (8 - h_complexity) * 0.01
    return A_star, w_acc - lean_penalty, acc_A, acc_B, acc_A_ret, h_complexity

def run(cfg_name, use_astar, seed):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    pop = [Genome() for _ in range(POP)]
    vocab = 10

    for gen in range(GEN):
        X, YA, YB, V = make_taskAB(vocab)
        scores, As, accAs, accBs, retAs, hs, srs = [], [], [], [], [], [], []
        for ind in pop:
            A, fit, aA, aB, aR, h_val = measure(ind, X, YA, YB, V, use_astar)
            scores.append(fit); As.append(A); accAs.append(aA)
            accBs.append(aB); retAs.append(aR); hs.append(h_val); srs.append(ind.sr)

        alive_rate = sum(1 for a in As if a >= 1.0) / POP * 100
        med_h = np.median(hs); med_ret = np.median(retAs); med_sr = np.median(srs)

        fit_cutoff = np.percentile([s for s in scores if s >= 0], 30)
        lean_hi = np.percentile(hs, 70); lean_lo = np.percentile(hs, 10)
        keep = [i for i, s in enumerate(scores)
                if As[i] >= (1.0 if use_astar else -99)
                and s >= fit_cutoff and lean_lo <= hs[i] <= lean_hi]
        if len(keep) < POP * 0.2: keep = list(range(POP))

        if alive_rate > 40 and med_ret > 0.7 and 10 < med_h < 40:
            vocab = min(200, vocab + 5)
        elif alive_rate < 15 or med_ret < 0.3:
            vocab = max(10, vocab - 5)

        survivors = [pop[i] for i in keep]
        new_pop = [pop[i] for i in keep[:max(1, int(len(keep) * 0.25))]]
        while len(new_pop) < POP:
            if len(survivors) < 2: break
            t1 = random.sample(range(len(survivors)), min(3, len(survivors)))
            p1 = survivors[t1[0]]
            t2 = random.sample(range(len(survivors)), min(3, len(survivors)))
            p2 = survivors[t2[0]]
            c = Genome.crossover(p1, p2); c.mutate(0.4); new_pop.append(c)
        while len(new_pop) < POP: new_pop.append(Genome())
        pop = new_pop

    final_A = np.median(As); final_ret = np.median(retAs)
    final_accA = np.median(accAs); final_accB = np.median(accBs)
    final_forget = final_accA - final_ret
    t_elapsed = 0  # approximate
    return {
        'cfg': cfg_name, 'seed': seed,
        'vocab': vocab, 'alive_pct': alive_rate,
        'acc_A': round(final_accA, 4), 'acc_B': round(final_accB, 4),
        'ret_A': round(final_ret, 4), 'forget': round(final_forget, 4),
        'med_sr': round(med_sr, 3), 'med_h': round(med_h, 1),
        'med_A': round(final_A, 3), 'time': t_elapsed,
    }

# ── RUN ──
print("=" * 60)
print("A* Gating: Additional 3 Seeds (456, 789, 111)")
print("=" * 60)

all_results = []
for seed in EXTRA_SEEDS:
    print(f"\n--- seed={seed} ---")
    for cfg_name, use_astar in [('WITH_A*', True), ('NO_A*', False)]:
        t0 = time.time()
        r = run(cfg_name, use_astar, seed)
        r['time'] = round(time.time() - t0, 1)
        all_results.append(r)
        tag = "ALIVE" if r['med_A'] >= 1.0 else "DEAD"
        print(f"  {cfg_name}: sr={r['med_sr']:.3f} A*={r['med_A']:.3f}[{tag}] "
              f"acc_A={r['acc_A']:.3f} ret={r['ret_A']:.3f} "
              f"forget={r['forget']:+.3f} vocab={r['vocab']} alive%={r['alive_pct']:.0f} "
              f"t={r['time']}s")

print("\n" + "=" * 60)
print("ALL RESULTS (including previous 2 seeds)")
print("=" * 60)

# Combine with previous results
prev = [
    {'cfg': 'WITH_A*', 'seed': 42, 'med_sr': 1.259, 'med_A': 2.545, 'alive_pct': 92, 'vocab': 60, 'acc_A': 0.917, 'ret_A': 0.915, 'forget': 0.003, 'med_h': 8.7},
    {'cfg': 'NO_A*', 'seed': 42, 'med_sr': 0.372, 'med_A': 0.134, 'alive_pct': 0, 'vocab': 10, 'acc_A': 0.970, 'ret_A': 0.965, 'forget': 0.005, 'med_h': 30.6},
    {'cfg': 'WITH_A*', 'seed': 123, 'med_sr': 1.167, 'med_A': 2.629, 'alive_pct': 95, 'vocab': 45, 'acc_A': 0.935, 'ret_A': 0.935, 'forget': 0.000, 'med_h': 7.2},
    {'cfg': 'NO_A*', 'seed': 123, 'med_sr': 0.353, 'med_A': 0.099, 'alive_pct': 0, 'vocab': 10, 'acc_A': 0.970, 'ret_A': 0.970, 'forget': 0.000, 'med_h': 30.6},
]
all_data = prev + all_results

for cfg_name in ['WITH_A*', 'NO_A*']:
    rows = [r for r in all_data if r['cfg'] == cfg_name]
    avg = lambda k: np.mean([r[k] for r in rows])
    print(f"\n{cfg_name} (n={len(rows)} seeds):")
    print(f"  sr={avg('med_sr'):.3f} A*={avg('med_A'):.3f} alive%={avg('alive_pct'):.0f}")
    print(f"  vocab={avg('vocab'):.0f} acc_A={avg('acc_A'):.3f} forget={avg('forget'):+.4f}")

# Save
with open(os.path.join(os.path.dirname(__file__), '..', 'paper', 'astar_5seeds.json'), 'w') as f:
    json.dump(all_data, f, indent=2)
print("\nSaved to paper/astar_5seeds.json")
