"""
Meta-Cognitive Search vs A* Gate: Can diversity protection replace A*?

Tests 4 conditions × 2 seeds:
  WITH_A* + standard  = original (should be stable)
  WITH_A* + diversity = A* + niche protection
  NO_A*  + standard  = original (should collapse)
  NO_A*  + diversity = KEY TEST: can diversity alone prevent collapse?

Source: idea-14, 2026-06-10
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random, time, json, os
from collections import Counter

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = 64
POP, GEN = 40, 40
NICHE_GRACE = 3
BINS = 5

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
        Xs.append(ids[i:i + 4])
        Ya.append(ids[i + 4])
        Yb.append(ids[i + 5])
    return torch.tensor(Xs), torch.tensor(Ya), torch.tensor(Yb), len(chars)

class Core(nn.Module):
    def __init__(self, sr=0.9, gain=0.1, sp=0.5):
        super().__init__()
        W = torch.randn(D, D) * 0.3
        mask = (torch.rand(D, D) > sp).float()
        W = W * mask
        s = torch.linalg.norm(W, 2)
        self.W = nn.Parameter(W * (sr / (s + 1e-8)))
        self.b = nn.Parameter(torch.zeros(D))
        self.g = nn.Parameter(torch.tensor(gain))
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

def compute_niche_grid(population, bins=BINS):
    srs = [ind.sr for ind in population]
    gains = [ind.gain for ind in population]
    sps = [ind.sp for ind in population]
    sr_edges = np.linspace(0.1, 2.0, bins + 1)
    gain_edges = np.linspace(0.0, 1.0, bins + 1)
    sp_edges = np.linspace(0.0, 0.9, bins + 1)
    niche = {}
    for i, ind in enumerate(population):
        sr_bin = min(bins-1, np.digitize(ind.sr, sr_edges) - 1)
        gain_bin = min(bins-1, np.digitize(ind.gain, gain_edges) - 1)
        sp_bin = min(bins-1, np.digitize(ind.sp, sp_edges) - 1)
        nid = f"{sr_bin}_{gain_bin}_{sp_bin}"
        niche.setdefault(nid, []).append(i)
    return niche

def measure(ind, X, YA, YB, vocab, use_astar):
    core = ind.build()
    h = torch.randn(4, D, device=DEV) * 0.5
    A_star = fast_A_det(core, h, 40)
    if use_astar and A_star < 1.0:
        return A_star, -1.0, 0, 0, 0, 0
    emb = nn.Embedding(vocab, D // 4).to(DEV)
    ro = nn.Linear(D, vocab).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ro.parameters()), lr=0.01)
    def train_eval(tX, tY, n_train=200):
        perm = torch.randperm(len(tX))[:n_train]
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            bx = tX[idx].to(DEV); by = tY[idx].to(DEV)
            e = emb(bx).flatten(1)
            hh = torch.zeros(len(bx), D, device=DEV)
            for _ in range(3): hh = core.step_noisy(hh, e)
            loss = F.cross_entropy(ro(hh), by)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tidx = torch.randint(0, len(tX), (200,))
            bx = tX[tidx].to(DEV); by = tY[tidx].to(DEV)
            e = emb(bx).flatten(1)
            hh = torch.zeros(len(bx), D, device=DEV)
            for _ in range(3): hh = core.step_noisy(hh, e)
            return (ro(hh).argmax(-1) == by).float().mean().item()
    acc_A = train_eval(X, YA, 200)
    acc_B = train_eval(X, YB, 200)
    acc_A_ret = train_eval(X, YA, 50)
    w_acc = acc_A * 0.2 + acc_B * 0.2 + acc_A_ret * 0.6
    h_complexity = ind.sr * D * (1 - ind.sp) + ind.gain * 10
    lean_penalty = 0
    if h_complexity > 30: lean_penalty = (h_complexity - 30) * 0.003
    elif h_complexity < 8: lean_penalty = (8 - h_complexity) * 0.01
    return A_star, w_acc - lean_penalty, acc_A, acc_B, acc_A_ret, h_complexity

def run(cfg_name, use_astar, use_diversity, seed):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    pop = [Genome() for _ in range(POP)]
    vocab = 10
    endangered_counter = {}
    gen_log = []

    for gen in range(GEN):
        X, YA, YB, V = make_taskAB(vocab)
        scores, As, accAs, accBs, retAs, hs, srs = [], [], [], [], [], [], []
        for ind in pop:
            A, fit, aA, aB, aR, h_val = measure(ind, X, YA, YB, V, use_astar)
            scores.append(fit); As.append(A)
            accAs.append(aA); accBs.append(aB); retAs.append(aR)
            hs.append(h_val); srs.append(ind.sr)

        alive_rate = sum(1 for a in As if a >= 1.0) / POP * 100
        med_sr = np.median(srs); med_A = np.median([a for a in As if a >= 0] or [0])
        med_ret = np.median(retAs)

        fit_cutoff = np.percentile([s for s in scores if s >= 0], 30)
        lean_hi = np.percentile(hs, 70); lean_lo = np.percentile(hs, 10)
        standard_keep = [i for i, s in enumerate(scores)
                         if As[i] >= (1.0 if use_astar else -99)
                         and s >= fit_cutoff and lean_lo <= hs[i] <= lean_hi]

        # === Diversity protection (only when use_diversity=True) ===
        rescued = set()
        if use_diversity:
            niche = compute_niche_grid(pop)
            for i in range(POP):
                ind_id = id(pop[i])
                if i not in standard_keep:
                    endangered_counter[ind_id] = endangered_counter.get(ind_id, 0) + 1
                else:
                    endangered_counter[ind_id] = 0
            for nid, members in niche.items():
                if len(members) == 1:
                    i = members[0]
                    if i not in standard_keep:
                        if endangered_counter.get(id(pop[i]), 0) < NICHE_GRACE:
                            rescued.add(i)
            keep = sorted(set(standard_keep) | rescued)
        else:
            keep = standard_keep

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
            p1 = survivors[t1[0]]
            t2 = random.sample(range(len(survivors)), min(3, len(survivors)))
            p2 = survivors[t2[0]]
            c = Genome.crossover(p1, p2); c.mutate(0.4); new_pop.append(c)
        while len(new_pop) < POP: new_pop.append(Genome())
        pop = new_pop

        gen_log.append({'gen': gen, 'vocab': vocab, 'alive': alive_rate,
                         'med_sr': float(med_sr), 'med_A': float(med_A),
                         'rescued': len(rescued) - len([r for r in rescued if r in standard_keep])})

    fin = gen_log[-1]
    return {
        'cfg': cfg_name, 'seed': seed,
        'vocab': fin['vocab'], 'alive_pct': fin['alive'],
        'med_sr': fin['med_sr'], 'med_A': fin['med_A'],
        'acc_A': round(np.median(accAs), 4), 'acc_B': round(np.median(accBs), 4),
        'ret_A': round(np.median(retAs), 4),
        'forget': round(np.median(accAs) - np.median(retAs), 4),
        'gen_log': gen_log,
    }


# ═══ RUN ═══
SEEDS = [42, 123]
CONFIGS = [
    ('WITH_A*  (standard)',  True,  False),
    ('WITH_A*  +diversity',  True,  True),
    ('NO_A*   (standard)',   False, False),
    ('NO_A*   +diversity',   False, True),  # KEY TEST
]

print("=" * 78)
print("META-COGNITIVE SEARCH: Can diversity protection replace A* gate?")
print("=" * 78)

all_results = []
for seed in SEEDS:
    print(f"\n--- seed={seed} ---")
    for cfg_name, use_astar, use_div in CONFIGS:
        t0 = time.time()
        r = run(cfg_name, use_astar, use_div, seed)
        r['time'] = round(time.time() - t0, 1)
        all_results.append(r)
        tag = "ALIVE" if r['med_A'] >= 1.0 else "DEAD"
        arrow = "→"
        print(f"  {cfg_name:<25s}: sr={r['med_sr']:.3f} A*={r['med_A']:.3f}[{tag}] "
              f"forget={r['forget']:+.3f} vocab={r['vocab']} alive%={r['alive_pct']:.0f} "
              f"t={r['time']}s")

# Summary
print("\n" + "=" * 78)
print("SUMMARY (averaged over seeds)")
print("=" * 78)
for cfg_name, _, _ in CONFIGS:
    rows = [r for r in all_results if r['cfg'] == cfg_name]
    avg = lambda k: np.mean([r[k] for r in rows])
    print(f"  {cfg_name:<25s}: sr={avg('med_sr'):.3f} A*={avg('med_A'):.3f} "
          f"forget={avg('forget'):+.4f} alive%={avg('alive_pct'):.0f}")

print("\nKEY QUESTION ANSWERED:")
print("  Can diversity protection alone prevent population collapse?")
no_std = [r for r in all_results if r['cfg'] == 'NO_A*   (standard)']
no_div = [r for r in all_results if r['cfg'] == 'NO_A*   +diversity']
print(f"  NO_A* standard:  sr={np.mean([r['med_sr'] for r in no_std]):.3f} alive%={np.mean([r['alive_pct'] for r in no_std]):.0f}")
print(f"  NO_A* diversity: sr={np.mean([r['med_sr'] for r in no_div]):.3f} alive%={np.mean([r['alive_pct'] for r in no_div]):.0f}")
