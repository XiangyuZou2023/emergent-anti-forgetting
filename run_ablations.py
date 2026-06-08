"""
Ablation experiments for: "Emergent Anti-Forgetting through Autopoietic Evolution"

Tests 4 configurations × N seeds:
  full:      A* gate + A→B→A fitness + complexity tax (our method)
  no_astar:  No A* survival gate — all individuals survive
  no_ret:    No A→B→A — fitness = avg(acc_A, acc_B) only
  no_tax:    No complexity tax — architectures can grow unbounded

Also runs 2 fixed baselines (no evolution):
  fixed_best:   Fixed architecture with best evolved params
  fixed_ffwd:   Feedforward only (no recurrence)

Output: paper_experiments/results.json + console comparison table.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sys, io, numpy as np, random, time, json, os
from collections import Counter

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")

# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'cc_clean.txt')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw = f.read(300_000)
cnt = Counter(raw)
all_chars = sorted([c for c, n in cnt.items() if n >= 5 and '一' <= c <= '鿿'])
print(f"Available chars: {len(all_chars)}")


def make_taskAB(vocab_size):
    chars = ['<PAD>', '<UNK>'] + all_chars[:vocab_size]
    c2i = {c: i for i, c in enumerate(chars)}
    ids = [c2i.get(c, 1) for c in raw if '一' <= c <= '鿿'][:20000]
    Seq = 4
    Xs, Ya, Yb = [], [], []
    for i in range(len(ids) - Seq - 2):
        Xs.append(ids[i:i + Seq])
        Ya.append(ids[i + Seq])
        Yb.append(ids[i + Seq + 1])
    return torch.tensor(Xs), torch.tensor(Ya), torch.tensor(Yb), len(chars)


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════
class Core(nn.Module):
    def __init__(self, d=64, sr=0.9, gain=0.1, sp=0.5, in_dim=64):
        super().__init__()
        W = torch.randn(d, d) * 0.3
        mask = (torch.rand(d, d) > sp).float()
        W = W * mask
        s = torch.linalg.norm(W, 2)
        self.W = nn.Parameter(W * (sr / (s + 1e-8)))
        self.b = nn.Parameter(torch.zeros(d))
        self.g = nn.Parameter(torch.tensor(gain))
        self.inp = nn.Linear(in_dim, d, bias=False)

    def step(self, h, x=None):
        h = h.detach()
        ext = torch.zeros_like(h)
        if x is not None:
            ext = self.inp(x)
        return 0.9 * torch.tanh(h @ self.W.T + self.b + ext) + 0.1 * self.g * h + torch.randn_like(h) * 0.005


def fast_A(core, h, n=40):
    for _ in range(10):
        h = core.step(h)
    tr = []
    for _ in range(n):
        h = core.step(h)
        tr.append(h / (h.norm(dim=-1, keepdim=True) + 1e-8))
    ds = [.5 * (tr[i] - tr[i + 1]).norm(dim=-1).mean().item() for i in range(len(tr) - 1)]
    S = 1 - min(np.mean(ds), 1.)
    return 4 * (1 - S) * S / 0.3


# ═══════════════════════════════════════════════════════════════
# Genome
# ═══════════════════════════════════════════════════════════════
class Genome:
    d = 64

    def __init__(self, sr=0.9, gain=0.1, sp=0.5):
        self.sr = np.clip(sr + random.uniform(-0.05, 0.05), 0.5, 2.0)
        self.gain = np.clip(gain + random.uniform(-0.05, 0.05), 0.0, 1.0)
        self.sp = np.clip(sp + random.uniform(-0.1, 0.1), 0.0, 0.9)

    def mutate(self, r=0.3):
        if random.random() < r: self.sr = np.clip(self.sr + random.uniform(-0.2, 0.2), 0.5, 2.0)
        if random.random() < r: self.gain = np.clip(self.gain + random.uniform(-0.2, 0.2), 0.0, 1.0)
        if random.random() < r: self.sp = np.clip(self.sp + random.uniform(-0.2, 0.2), 0.0, 0.9)

    def build(self):
        return Core(self.d, self.sr, self.gain, self.sp, self.d).to(DEV)

    @staticmethod
    def crossover(a, b):
        c = Genome()
        c.sr = a.sr if random.random() < 0.5 else b.sr
        c.gain = a.gain if random.random() < 0.5 else b.gain
        c.sp = a.sp if random.random() < 0.5 else b.sp
        return c


# ═══════════════════════════════════════════════════════════════
# Measure (configurable)
# ═══════════════════════════════════════════════════════════════
def measure(ind, X, YA, YB, vocab, cfg, B=32):
    """
    cfg: dict with keys:
      use_astar: bool    — use A* survival gate
      use_ret:   bool    — include ret_A in fitness
      use_tax:   bool    — apply complexity tax
    """
    core = ind.build()
    d = ind.d

    # A* gate (only if enabled)
    A_star = 2.0  # default: alive
    if cfg['use_astar']:
        h = torch.randn(4, d, device=DEV) * 0.5
        A_star = fast_A(core, h, 40)
        if A_star < 1.0:
            return A_star, -1.0, 0, 0, 0, 0

    emb = nn.Embedding(vocab, d // 4).to(DEV)
    ro = nn.Linear(d, vocab).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ro.parameters()), lr=0.01)

    def train_eval(tX, tY, n_train=200):
        perm = torch.randperm(len(tX))[:n_train]
        for i in range(0, len(perm), B):
            idx = perm[i:i + B]
            bx = tX[idx].to(DEV)
            by = tY[idx].to(DEV)
            e = emb(bx).flatten(1)
            hh = torch.zeros(bx.shape[0], d, device=DEV)
            for _ in range(3):
                hh = core.step(hh, e)
            loss = F.cross_entropy(ro(hh), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            tidx = torch.randint(0, len(tX), (200,))
            bx = tX[tidx].to(DEV)
            by = tY[tidx].to(DEV)
            e = emb(bx).flatten(1)
            hh = torch.zeros(len(bx), d, device=DEV)
            for _ in range(3):
                hh = core.step(hh, e)
            acc = (ro(hh).argmax(-1) == by).float().mean().item()
        return acc

    acc_A = train_eval(X, YA, 200)
    acc_B = train_eval(X, YB, 200)
    acc_A_ret = train_eval(X, YA, 50)

    # Fitness composition
    if cfg['use_ret']:
        w_acc = acc_A * 0.2 + acc_B * 0.2 + acc_A_ret * 0.6
    else:
        w_acc = acc_A * 0.5 + acc_B * 0.5

    # Complexity tax
    h_complexity = ind.sr * ind.d * (1 - ind.sp) + ind.gain * 10
    lean_penalty = 0.0
    if cfg['use_tax']:
        if h_complexity > 30:
            lean_penalty = (h_complexity - 30) * 0.003
        elif h_complexity < 8:
            lean_penalty = (8 - h_complexity) * 0.01

    return A_star, w_acc - lean_penalty, acc_A, acc_B, acc_A_ret, h_complexity


# ═══════════════════════════════════════════════════════════════
# Evolution loop
# ═══════════════════════════════════════════════════════════════
def run_evolution(cfg, seed=42):
    """Run one evolutionary run, return summary stats."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    POP, GEN = 40, 60
    pop = [Genome() for _ in range(POP)]
    vocab = 10
    best_ind, best_fit = None, -99
    t0 = time.time()

    history = []  # per-generation stats

    for gen in range(GEN):
        X, YA, YB, V = make_taskAB(vocab)
        scores, As, accAs, accBs, retAs, hs = [], [], [], [], [], []
        for ind in pop:
            A, fit, aA, aB, aR, h_val = measure(ind, X, YA, YB, V, cfg)
            scores.append(fit)
            As.append(A)
            accAs.append(aA)
            accBs.append(aB)
            retAs.append(aR)
            hs.append(h_val)

        alive_rate = sum(1 for a in As if a >= 1.0) / POP * 100
        med_h = np.median(hs)
        med_ret = np.median(retAs)

        # Selection: fitness bottom 30% + fattest 30% + thinnest 10%
        fit_cutoff = np.percentile([s for s in scores if s >= 0], 30)
        lean_hi = np.percentile(hs, 70)
        lean_lo = np.percentile(hs, 10)
        keep = [i for i, s in enumerate(scores)
                if As[i] >= (1.0 if cfg['use_astar'] else 0.0)
                and s >= fit_cutoff
                and lean_lo <= hs[i] <= lean_hi]
        if len(keep) < POP * 0.2:
            keep = list(range(POP))

        idx = scores.index(max(scores))
        if scores[idx] > best_fit:
            best_fit = scores[idx]
            best_ind = pop[idx]

        # Adaptive vocab
        if cfg['use_ret']:
            if alive_rate > 40 and med_ret > 0.7 and 10 < med_h < 40:
                vocab = min(500, vocab + 5)
            elif alive_rate < 15 or med_ret < 0.3:
                vocab = max(10, vocab - 5)
        else:
            if alive_rate > 40 and 10 < med_h < 40:
                vocab = min(500, vocab + 5)
            elif alive_rate < 15:
                vocab = max(10, vocab - 5)

        survivors = [pop[i] for i in keep]
        new_pop = []
        for i in keep[:max(1, int(len(keep) * 0.25))]:
            new_pop.append(pop[i])
        while len(new_pop) < POP:
            if len(survivors) < 2: break
            t = random.sample(range(len(survivors)), min(3, len(survivors)))
            p1 = survivors[t[0]]
            t = random.sample(range(len(survivors)), min(3, len(survivors)))
            p2 = survivors[t[0]]
            c = Genome.crossover(p1, p2)
            c.mutate(0.4)
            new_pop.append(c)
        while len(new_pop) < POP:
            new_pop.append(Genome())
        pop = new_pop

        history.append({
            'gen': gen, 'vocab': vocab, 'alive_rate': alive_rate,
            'med_accA': float(np.median(accAs)), 'med_accB': float(np.median(accBs)),
            'med_retA': float(np.median(retAs)), 'best_fit': float(scores[idx]),
            'med_h': float(med_h),
        })

    elapsed = time.time() - t0
    final = history[-1]
    return {
        'config': cfg['name'],
        'seed': seed,
        'time_s': round(elapsed, 1),
        'final_vocab': vocab,
        'final_alive_pct': final['alive_rate'],
        'acc_A': final['med_accA'],
        'acc_B': final['med_accB'],
        'ret_A': final['med_retA'],
        'forgetting': round(final['med_accA'] - final['med_retA'], 4),  # >0 = forgetting
        'best_fit': final['best_fit'],
        'med_h': final['med_h'],
        'history': history,
    }


# ═══════════════════════════════════════════════════════════════
# Fixed baselines (no evolution)
# ═══════════════════════════════════════════════════════════════
def run_fixed(sr, gain, sp, label, vocab=30, seed=42):
    """Single fixed-architecture run, no evolution."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ind = Genome()
    ind.sr, ind.gain, ind.sp = sr, gain, sp
    core = ind.build()
    d = ind.d

    # A*
    h = torch.randn(4, d, device=DEV) * 0.5
    A_star = fast_A(core, h, 40)

    X, YA, YB, V = make_taskAB(vocab)
    cfg_fake = {'use_astar': False, 'use_ret': True, 'use_tax': False}
    A, fit, acc_A, acc_B, acc_A_ret, h_val = measure(ind, X, YA, YB, V, cfg_fake)

    # Now retest with fresh embed/readout
    emb = nn.Embedding(V, d // 4).to(DEV)
    ro = nn.Linear(d, V).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ro.parameters()), lr=0.01)

    def train_eval(tX, tY, n=200):
        perm = torch.randperm(len(tX))[:n]
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            bx = tX[idx].to(DEV)
            by = tY[idx].to(DEV)
            e = emb(bx).flatten(1)
            hh = torch.zeros(bx.shape[0], d, device=DEV)
            for _ in range(3): hh = core.step(hh, e)
            loss = F.cross_entropy(ro(hh), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            tidx = torch.randint(0, len(tX), (200,))
            bx = tX[tidx].to(DEV)
            by = tY[tidx].to(DEV)
            e = emb(bx).flatten(1)
            hh = torch.zeros(len(bx), d, device=DEV)
            for _ in range(3): hh = core.step(hh, e)
            return (ro(hh).argmax(-1) == by).float().mean().item()

    acc_A_final = train_eval(X, YA, 200)
    acc_B_final = train_eval(X, YB, 200)
    acc_A_ret_final = train_eval(X, YA, 50)

    return {
        'config': label,
        'seed': seed,
        'time_s': 0,
        'final_vocab': vocab,
        'final_alive_pct': 100 if A_star >= 1.0 else 0,
        'acc_A': round(acc_A_final, 4),
        'acc_B': round(acc_B_final, 4),
        'ret_A': round(acc_A_ret_final, 4),
        'forgetting': round(acc_A_final - acc_A_ret_final, 4),
        'best_fit': 0,
        'med_h': round(sr * d * (1 - sp) + gain * 10, 1),
        'history': [],
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        choices=['full', 'no_astar', 'no_ret', 'no_tax',
                                 'fixed_best', 'fixed_ffwd', 'all'])
    parser.add_argument('--seeds', type=str, default='42,123,777',
                        help='Comma-separated seed list')
    args = parser.parse_args()

    SEEDS = [int(s) for s in args.seeds.split(',')]
    all_results = []

    config_map = {
        'full':     {'name': 'full',     'use_astar': True,  'use_ret': True,  'use_tax': True},
        'no_astar': {'name': 'no_astar', 'use_astar': False, 'use_ret': True,  'use_tax': True},
        'no_ret':   {'name': 'no_ret',   'use_astar': True,  'use_ret': False, 'use_tax': True},
        'no_tax':   {'name': 'no_tax',   'use_astar': True,  'use_ret': True,  'use_tax': False},
    }

    if args.config == 'all':
        print("Run with --config <name> to pick one. Choices:", list(config_map.keys()) + ['fixed_best', 'fixed_ffwd'])
        import sys; sys.exit(0)

    if args.config in config_map:
        cfg = config_map[args.config]
        print(f"Config: {cfg['name']} | Seeds: {SEEDS}")
        for seed in SEEDS:
            r = run_evolution(cfg, seed)
            all_results.append(r)
            print(f"  seed={seed}: A={r['acc_A']:.3f} B={r['acc_B']:.3f} "
                  f"ret={r['ret_A']:.3f} forget={r['forgetting']:.3f} "
                  f"alive%={r['final_alive_pct']:.0f} vocab={r['final_vocab']} "
                  f"h={r['med_h']:.1f} t={r['time_s']}s", flush=True)
    elif args.config == 'fixed_best':
        print(f"Config: fixed_best | Seeds: {SEEDS}")
        for seed in SEEDS:
            r = run_fixed(0.87, 0.05, 0.46, 'fixed_best', seed=seed)
            all_results.append(r)
            print(f"  seed={seed}: A={r['acc_A']:.3f} B={r['acc_B']:.3f} "
                  f"ret={r['ret_A']:.3f} forget={r['forgetting']:.3f}", flush=True)
    elif args.config == 'fixed_ffwd':
        print(f"Config: fixed_ffwd | Seeds: {SEEDS}")
        for seed in SEEDS:
            r = run_fixed(0.0, 0.0, 0.9, 'fixed_ffwd', seed=seed)
            all_results.append(r)
            print(f"  seed={seed}: A={r['acc_A']:.3f} B={r['acc_B']:.3f} "
                  f"ret={r['ret_A']:.3f} forget={r['forgetting']:.3f}", flush=True)

    # Append to cumulative results file
    out_path = os.path.join(os.path.dirname(__file__), 'results.jsonl')
    with open(out_path, 'a', encoding='utf-8') as f:
        for r in all_results:
            slim = {k: v for k, v in r.items() if k != 'history'}
            f.write(json.dumps(slim, ensure_ascii=False) + '\n')
    print(f"Appended {len(all_results)} runs to {out_path}", flush=True)
