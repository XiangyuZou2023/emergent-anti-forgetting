"""Generate A* gating comparison figure (5 seeds)."""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Load data
data_path = os.path.join(os.path.dirname(__file__), '..', 'paper', 'astar_5seeds.json')
with open(data_path) as f:
    data = json.load(f)

with_astar = [r for r in data if r['cfg'] == 'WITH_A*']
no_astar = [r for r in data if r['cfg'] == 'NO_A*']

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Panel 1: A* score
ax = axes[0, 0]
x = [1, 2]
wa_vals = [r['med_A'] for r in with_astar]
na_vals = [r['med_A'] for r in no_astar]
bp = ax.boxplot([wa_vals, na_vals], positions=x, widths=0.3, patch_artist=True)
bp['boxes'][0].set_facecolor('#2E86AB'); bp['boxes'][1].set_facecolor('#C73E1D')
ax.scatter([1]*5, wa_vals, c='#2E86AB', s=50, zorder=10)
ax.scatter([2]*5, na_vals, c='#C73E1D', s=50, zorder=10)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Recurrence threshold')
ax.set_xticks([1, 2])
ax.set_xticklabels(['WITH A*\n(n=5)', 'NO A*\n(n=5)'])
ax.set_ylabel('A* Score')
ax.set_title(f'A: Deterministic A*\nμ={np.mean(wa_vals):.2f} vs μ={np.mean(na_vals):.2f}')
ax.legend(fontsize=8)

# Panel 2: Spectral Radius
ax = axes[0, 1]
wsr = [r['med_sr'] for r in with_astar]
nsr = [r['med_sr'] for r in no_astar]
bp2 = ax.boxplot([wsr, nsr], positions=x, widths=0.3, patch_artist=True)
bp2['boxes'][0].set_facecolor('#2E86AB'); bp2['boxes'][1].set_facecolor('#C73E1D')
ax.scatter([1]*5, wsr, c='#2E86AB', s=50, zorder=10)
ax.scatter([2]*5, nsr, c='#C73E1D', s=50, zorder=10)
ax.set_xticks([1, 2])
ax.set_xticklabels(['WITH A*\n(n=5)', 'NO A*\n(n=5)'])
ax.set_ylabel('Spectral Radius')
ax.set_title(f'B: Evolved Spectral Radius\nμ={np.mean(wsr):.2f} vs μ={np.mean(nsr):.2f}')

# Panel 3: Alive %
ax = axes[1, 0]
wal = [r['alive_pct'] for r in with_astar]
nal = [r['alive_pct'] for r in no_astar]
bp3 = ax.boxplot([wal, nal], positions=x, widths=0.3, patch_artist=True)
bp3['boxes'][0].set_facecolor('#2E86AB'); bp3['boxes'][1].set_facecolor('#C73E1D')
ax.scatter([1]*5, wal, c='#2E86AB', s=50, zorder=10)
ax.scatter([2]*5, nal, c='#C73E1D', s=50, zorder=10)
ax.set_xticks([1, 2])
ax.set_xticklabels(['WITH A*\n(n=5)', 'NO A*\n(n=5)'])
ax.set_ylabel('Alive Population (%)')
ax.set_title(f'C: Population Viability\nμ={np.mean(wal):.0f}% vs μ={np.mean(nal):.0f}%')

# Panel 4: Final Vocabulary
ax = axes[1, 1]
wv = [r['vocab'] for r in with_astar]
nv = [r['vocab'] for r in no_astar]
bp4 = ax.boxplot([wv, nv], positions=x, widths=0.3, patch_artist=True)
bp4['boxes'][0].set_facecolor('#2E86AB'); bp4['boxes'][1].set_facecolor('#C73E1D')
ax.scatter([1]*5, wv, c='#2E86AB', s=50, zorder=10)
ax.scatter([2]*5, nv, c='#C73E1D', s=50, zorder=10)
ax.set_xticks([1, 2])
ax.set_xticklabels(['WITH A*\n(n=5)', 'NO A*\n(n=5)'])
ax.set_ylabel('Final Vocabulary Size')
ax.set_title(f'D: Task Difficulty Progression\nμ={np.mean(wv):.0f} vs μ={np.mean(nv):.0f} chars')

fig.suptitle('Figure 5: A* Gating Prevents Population Collapse (5 seeds)',
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig5_astar_gating.pdf'))
fig.savefig(os.path.join(OUT_DIR, 'fig5_astar_gating.png'))
plt.close(fig)
print(f'[OK] fig5_astar_gating saved to {OUT_DIR}')
