"""
Meta-Cognitive Diversity Search — 对论文消融实验的增强

核心改动：在淘汰逻辑上加"生态位保护"
  - 弱个体如果占据独特的架构生态位 → 多活K代（延迟淘汰）
  - 模拟 ADHD不关筛选 → 给够时间让弱连接证明自己

来源: idea-14, 2026-06-10 认知评估元认知发现
"""

import sys, os, random, time, json
import torch, numpy as np

# Import the original ablation module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ablations import (
    DEV, make_taskAB, Core, Genome, fast_A, measure, run_evolution,
    run_fixed
)

# ============================================================
# Diversity Niche Protection
# ============================================================

def compute_niche_grid(population, bins=5):
    """
    Discretize the 3D genome space (sr, gain, sp) into bins^3 cells.
    Return dict: niche_id -> list of individual indices in that niche.
    """
    srs = [ind.sr for ind in population]
    gains = [ind.gain for ind in population]
    sps = [ind.sp for ind in population]

    sr_edges = np.linspace(0.5, 2.0, bins + 1)
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


def run_diversity_evolution(cfg, seed=42, niche_grace=3, bins=5):
    """
    Like run_evolution, but with diversity niche protection.

    niche_grace: how many extra generations a unique-niche weakling gets
    bins: discretization granularity for niche grid
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    POP, GEN = 40, 60
    pop = [Genome() for _ in range(POP)]
    vocab = 10
    best_ind, best_fit = None, -99
    t0 = time.time()

    # Track: how many gens has each individual been "endangered"
    endangered_counter = {}  # id(ind) -> consecutive_gens_below_cutoff

    history = []

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

        # Standard cutoffs (same as original)
        valid_scores = [s for s in scores if s >= 0]
        fit_cutoff = np.percentile(valid_scores, 30) if valid_scores else -99
        lean_hi = np.percentile(hs, 70)
        lean_lo = np.percentile(hs, 10)

        # Standard keep logic
        astar_thresh = 1.0 if cfg['use_astar'] else 0.0
        standard_keep = [i for i, s in enumerate(scores)
                         if As[i] >= astar_thresh
                         and s >= fit_cutoff
                         and lean_lo <= hs[i] <= lean_hi]

        # === DIVERSITY NICHE PROTECTION ===
        niche = compute_niche_grid(pop, bins)

        # Update endangered counters
        for i in range(POP):
            ind_id = id(pop[i])
            if i not in standard_keep:
                endangered_counter[ind_id] = endangered_counter.get(ind_id, 0) + 1
            else:
                endangered_counter[ind_id] = 0

        # Rescue: individuals in unique niches get extra grace
        rescued = set()
        unique_count = 0
        for nid, members in niche.items():
            if len(members) == 1:
                i = members[0]
                if i not in standard_keep:
                    ind_id = id(pop[i])
                    grace_used = endangered_counter.get(ind_id, 0)
                    if grace_used < niche_grace:
                        rescued.add(i)
                        unique_count += 1

        keep = sorted(set(standard_keep) | rescued)

        if len(keep) < POP * 0.2:
            keep = list(range(POP))

        # Log diversity stats
        niche_sizes = [len(m) for m in niche.values()]
        unique_niches = sum(1 for s in niche_sizes if s == 1)
        crowded_niches = sum(1 for s in niche_sizes if s >= 5)

        idx = scores.index(max(scores))
        if scores[idx] > best_fit:
            best_fit = scores[idx]
            best_ind = pop[idx]

        # Adaptive vocab (same as original)
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

        # Reproduce (same as original)
        survivors = [pop[i] for i in keep]
        new_pop = []
        for i in keep[:max(1, int(len(keep) * 0.25))]:
            new_pop.append(pop[i])
        while len(new_pop) < POP:
            if len(survivors) < 2:
                break
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
            'unique_niches': unique_niches, 'crowded_niches': crowded_niches,
            'rescued': len(rescued) - len([r for r in rescued if r in standard_keep]),
        })

    elapsed = time.time() - t0
    final = history[-1]
    return {
        'config': f"{cfg['name']}_diversity",
        'seed': seed,
        'time_s': round(elapsed, 1),
        'final_vocab': vocab,
        'final_alive_pct': final['alive_rate'],
        'acc_A': final['med_accA'],
        'acc_B': final['med_accB'],
        'ret_A': final['med_retA'],
        'forgetting': round(final['med_accA'] - final['med_retA'], 4),
        'best_fit': final['best_fit'],
        'med_h': final['med_h'],
        'final_unique_niches': final['unique_niches'],
        'history': history,
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='full',
                        choices=['full', 'no_astar', 'no_ret', 'no_tax'])
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--niche-grace', type=int, default=3,
                        help='Extra generations for unique-niche weaklings')
    parser.add_argument('--bins', type=int, default=5,
                        help='Niche grid granularity')
    args = parser.parse_args()

    config_map = {
        'full':     {'name': 'full',     'use_astar': True,  'use_ret': True,  'use_tax': True},
        'no_astar': {'name': 'no_astar', 'use_astar': False, 'use_ret': True,  'use_tax': True},
        'no_ret':   {'name': 'no_ret',   'use_astar': True,  'use_ret': False, 'use_tax': True},
        'no_tax':   {'name': 'no_tax',   'use_astar': True,  'use_ret': True,  'use_tax': False},
    }

    SEED_LIST = [int(s) for s in args.seeds.split(',')]
    cfg = config_map[args.config]

    print(f"Config: {cfg['name']}_diversity | niche_grace={args.niche_grace} | "
          f"bins={args.bins} | Seeds: {SEED_LIST}")

    all_results = []
    for seed in SEED_LIST:
        r = run_diversity_evolution(cfg, seed, args.niche_grace, args.bins)
        all_results.append(r)
        print(f"  seed={seed}: A={r['acc_A']:.3f} B={r['acc_B']:.3f} "
              f"ret={r['ret_A']:.3f} forget={r['forgetting']:.3f} "
              f"alive%={r['final_alive_pct']:.0f} vocab={r['final_vocab']} "
              f"unicorns={r['final_unique_niches']} "
              f"h={r['med_h']:.1f} t={r['time_s']}s", flush=True)

    # Also run original full for comparison
    print("\n--- Baseline full (no diversity) for comparison ---", flush=True)
    orig_cfg = {'name': 'full', 'use_astar': True, 'use_ret': True, 'use_tax': True}
    for seed in SEED_LIST:
        r = run_evolution(orig_cfg, seed)
        print(f"  seed={seed}: A={r['acc_A']:.3f} B={r['acc_B']:.3f} "
              f"ret={r['ret_A']:.3f} forget={r['forgetting']:.3f} "
              f"alive%={r['final_alive_pct']:.0f} vocab={r['final_vocab']} "
              f"h={r['med_h']:.1f} t={r['time_s']}s", flush=True)
        all_results.append(r)

    out_path = os.path.join(os.path.dirname(__file__), 'diversity_results.jsonl')
    with open(out_path, 'a', encoding='utf-8') as f:
        for r in all_results:
            slim = {k: v for k, v in r.items() if k != 'history'}
            f.write(json.dumps(slim, ensure_ascii=False) + '\n')
    print(f"\nSaved to {out_path}", flush=True)
