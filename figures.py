"""
Figure generation for: "Emergent Anti-Forgetting through Autopoietic Evolution"
Reads paper_experiments/results.jsonl -> generates all paper figures.
"""
import json, os, sys
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ══════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════
OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
}
plt.rcParams.update(STYLE)

# ══════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════
DATA_PATH = os.path.join(os.path.dirname(__file__), 'results.jsonl')
records = []
with open(DATA_PATH) as f:
    for line in f:
        records.append(json.loads(line))

# Group by config
by_cfg = defaultdict(list)
for r in records:
    by_cfg[r['config']].append(r)

EVOLVING = ['full', 'no_astar', 'no_ret', 'no_tax']
FIXED = ['fixed_best', 'fixed_ffwd']

CONFIG_LABELS = {
    'full': 'Full\n(A*+Ret+Tax)',
    'no_astar': 'No A*\nGate',
    'no_ret': 'No A→B→A',
    'no_tax': 'No Tax',
    'fixed_best': 'Fixed\n(Best Params)',
    'fixed_ffwd': 'Fixed\n(Feedforward)',
}
CONFIG_COLORS = {
    'full': '#2E86AB',
    'no_astar': '#A23B72',
    'no_ret': '#F18F01',
    'no_tax': '#C73E1D',
    'fixed_best': '#D64933',
    'fixed_ffwd': '#E87461',
}

# ══════════════════════════════════════════════════════
# Figure 1: Forgetting comparison (main result)
# ══════════════════════════════════════════════════════
def fig1_forgetting():
    """Bar chart: forgetting rate per config (mean±std, 3 seeds)."""
    configs = EVOLVING + FIXED
    means, stds = [], []
    for cfg in configs:
        vals = [r['forgetting'] for r in by_cfg[cfg]]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))
    colors = [CONFIG_COLORS[c] for c in configs]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='white', linewidth=0.8, width=0.6)

    # Zero line
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    # Shade: evolving zone
    ax.axvspan(-0.5, 3.5, alpha=0.04, color='green', label='Evolving (≤0 forgetting)')
    ax.axvspan(3.5, 5.5, alpha=0.04, color='red', label='Fixed (>0 forgetting)')

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs])
    ax.set_ylabel('Forgetting Rate\n(negative = anti-forgetting)')
    ax.set_title('Figure 1: Evolution as Anti-Forgetting Engine\n(6 configs × 3 seeds, mean±std)')

    # Value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        color = 'green' if m <= 0 else 'red'
        ax.text(i, m + s + 0.003, f'{m:+.3f}±{s:.3f}',
                ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.03, 0.06)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_forgetting.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig1_forgetting.png'))
    plt.close(fig)
    print('[OK] fig1_forgetting')

# ══════════════════════════════════════════════════════
# Figure 2: Accuracy trade-off
# ══════════════════════════════════════════════════════
def fig2_accuracy_tradeoff():
    """Scatter: accuracy vs forgetting, color by config type."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for cfg in EVOLVING:
        pts = [(r['acc_A'], r['forgetting']) for r in by_cfg[cfg]]
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=CONFIG_COLORS[cfg], label=CONFIG_LABELS[cfg].replace('\n',' '),
                   s=80, edgecolors='white', linewidth=0.5, zorder=5, alpha=0.85)

    for cfg in FIXED:
        pts = [(r['acc_A'], r['forgetting']) for r in by_cfg[cfg]]
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=CONFIG_COLORS[cfg], label=CONFIG_LABELS[cfg].replace('\n',' '),
                   s=80, marker='s', edgecolors='white', linewidth=0.5, zorder=5, alpha=0.85)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Accuracy on Task A')
    ax.set_ylabel('Forgetting Rate')
    ax.set_title('Figure 2: Accuracy vs Forgetting Trade-off\n(Evolving: circles, Fixed: squares)')
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2_accuracy_tradeoff.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig2_accuracy_tradeoff.png'))
    plt.close(fig)
    print('[OK] fig2_accuracy_tradeoff')

# ══════════════════════════════════════════════════════
# Figure 3: Ablation panel (4 metrics × evolving configs)
# ══════════════════════════════════════════════════════
def fig3_ablation():
    """Multi-panel: accuracy, forgetting, vocab, A* per evolving config."""
    metrics = [
        ('acc_A', 'Task A Accuracy'),
        ('forgetting', 'Forgetting Rate'),
        ('final_vocab', 'Final Vocabulary Size'),
        ('med_h', 'Median A* (H-score)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    configs = EVOLVING

    for ax, (key, label) in zip(axes.flat, metrics):
        x = np.arange(len(configs))
        means = [np.mean([r[key] for r in by_cfg[c]]) for c in configs]
        stds = [np.std([r[key] for r in by_cfg[c]]) for c in configs]
        colors = [CONFIG_COLORS[c] for c in configs]

        ax.bar(x, means, yerr=stds, capsize=4, color=colors,
               edgecolor='white', linewidth=0.8, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label)

        # Individual seed dots
        for i, cfg in enumerate(configs):
            vals = [r[key] for r in by_cfg[cfg]]
            ax.scatter([i]*len(vals), vals, c='black', s=20, zorder=10, alpha=0.6)

    fig.suptitle('Figure 3: Ablation Analysis — Effect of Removing Components\n(4 evolving configs × 3 seeds)', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_ablation.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig3_ablation.png'))
    plt.close(fig)
    print('[OK] fig3_ablation')

# ══════════════════════════════════════════════════════
# Figure 4: Evolving vs Fixed summary
# ══════════════════════════════════════════════════════
def fig4_evolving_vs_fixed():
    """Side-by-side: evolving (pooled) vs fixed architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # Pool data
    evo_forget = []
    for c in EVOLVING:
        evo_forget.extend([r['forgetting'] for r in by_cfg[c]])
    fix_forget = []
    for c in FIXED:
        fix_forget.extend([r['forgetting'] for r in by_cfg[c]])

    evo_acc = []
    for c in EVOLVING:
        evo_acc.extend([r['acc_A'] for r in by_cfg[c]])
    fix_acc = []
    for c in FIXED:
        fix_acc.extend([r['acc_A'] for r in by_cfg[c]])

    # Forgetting
    ax = axes[0]
    positions = [1, 2]
    parts = ax.violinplot([evo_forget, fix_forget], positions=positions,
                          showmeans=True, showmedians=True, widths=0.4)
    for pc, color in zip(parts['bodies'], ['green', 'red']):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
    ax.scatter([1]*len(evo_forget), evo_forget, c='green', s=30, alpha=0.5, zorder=10)
    ax.scatter([2]*len(fix_forget), fix_forget, c='red', s=30, alpha=0.5, marker='s', zorder=10)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Evolving\n(n={len(evo_forget)})', f'Fixed\n(n={len(fix_forget)})'])
    ax.set_ylabel('Forgetting Rate')
    ax.set_title(f'Forgetting\nEvo μ={np.mean(evo_forget):.4f} vs Fix μ={np.mean(fix_forget):.4f}')

    # Accuracy
    ax = axes[1]
    parts2 = ax.violinplot([evo_acc, fix_acc], positions=positions,
                           showmeans=True, showmedians=True, widths=0.4)
    for pc, color in zip(parts2['bodies'], ['green', 'red']):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
    ax.scatter([1]*len(evo_acc), evo_acc, c='green', s=30, alpha=0.5, zorder=10)
    ax.scatter([2]*len(fix_acc), fix_acc, c='red', s=30, alpha=0.5, marker='s', zorder=10)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Evolving\n(n={len(evo_acc)})', f'Fixed\n(n={len(fix_acc)})'])
    ax.set_ylabel('Accuracy (Task A)')
    ax.set_title(f'Accuracy\nEvo μ={np.mean(evo_acc):.3f} vs Fix μ={np.mean(fix_acc):.3f}')

    fig.suptitle('Figure 4: Evolving vs Fixed — Zero Forgetting at ~10pp Accuracy Cost',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig4_evolving_vs_fixed.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig4_evolving_vs_fixed.png'))
    plt.close(fig)
    print('[OK] fig4_evolving_vs_fixed')

# ══════════════════════════════════════════════════════
# Table 1: Full results summary (LaTeX)
# ══════════════════════════════════════════════════════
def table1_latex():
    """Generate LaTeX table of all results."""
    configs = EVOLVING + FIXED

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{Full experimental results. 6 configurations × 3 seeds. Mean ± std.}')
    lines.append(r'\label{tab:results}')
    lines.append(r'\begin{tabular}{lcccccc}')
    lines.append(r'\toprule')
    lines.append(r'Config & Acc$_A$ & Acc$_B$ & Ret$_A$ & Forgetting & Vocab & A* (med H) \\')
    lines.append(r'\midrule')

    for cfg in configs:
        rs = by_cfg[cfg]
        acc_A = f"{np.mean([r['acc_A'] for r in rs]):.3f}±{np.std([r['acc_A'] for r in rs]):.3f}"
        acc_B = f"{np.mean([r['acc_B'] for r in rs]):.3f}±{np.std([r['acc_B'] for r in rs]):.3f}"
        ret_A = f"{np.mean([r['ret_A'] for r in rs]):.3f}±{np.std([r['ret_A'] for r in rs]):.3f}"
        forget = f"{np.mean([r['forgetting'] for r in rs]):+.3f}±{np.std([r['forgetting'] for r in rs]):.3f}"
        vocab = f"{np.mean([r['final_vocab'] for r in rs]):.0f}"
        medh = f"{np.mean([r['med_h'] for r in rs]):.1f}"
        label = CONFIG_LABELS[cfg].replace('\n',' ')
        lines.append(f'{label} & {acc_A} & {acc_B} & {ret_A} & {forget} & {vocab} & {medh} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines)
    with open(os.path.join(OUT_DIR, 'table1_results.tex'), 'w') as f:
        f.write(tex)
    print('[OK] table1_results.tex')

# ══════════════════════════════════════════════════════
def main():
    print(f"Generating figures from {len(records)} records...")
    fig1_forgetting()
    fig2_accuracy_tradeoff()
    fig3_ablation()
    fig4_evolving_vs_fixed()
    table1_latex()
    print(f"\nDone! Figures saved to {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")

if __name__ == '__main__':
    main()
