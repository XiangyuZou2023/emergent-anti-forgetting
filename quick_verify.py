"""
Quick verification: 6 configs × 1 seed × 10 gens — validates experiment code.
Runs the IDENTICAL code path as run_ablations.py but with GEN=10.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Monkey-patch before importing the ablation module
import run_ablations

# Override the main config
run_ablations.POP = 20
run_ablations.GEN = 10

# Quick test configs with just 1 seed each
CONFIGS = ['full', 'no_astar', 'no_ret', 'no_tax', 'fixed_best', 'fixed_ffwd']

results = []
for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Running: {cfg} (GEN=10, POP=20, seed=42)")
    print(f"{'='*60}")

    try:
        stats = run_ablations.run_experiment(cfg, seed=42, pop=20, gen=10)
        results.append(stats)
        print(f"  {cfg}: forget={stats.get('forgetting', 'N/A'):.4f}"
              if isinstance(stats.get('forgetting'), float)
              else f"  {cfg}: done")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print(f"RESULTS: {len(results)}/{len(CONFIGS)} configs passed")
print(f"{'='*60}")
