"""D=256 3D地形图 — 标注输入字符分区，看信息怎么流动"""
import torch, torch.nn as nn, numpy as np, os, json, argparse

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class Core(nn.Module):
    def __init__(self, d=64, sr=0.9, gain=0.1, sp=0.5, w_seed=None):
        super().__init__(); self.d = d
        if w_seed is not None: torch.manual_seed(w_seed)
        W = torch.randn(d,d)*0.3; mask = (torch.rand(d,d)>sp).float(); W = W*mask
        s = torch.linalg.norm(W,2)
        self.W = nn.Parameter(W*(sr/(s+1e-8)))
        self.b = nn.Parameter(torch.zeros(d)); self.g = nn.Parameter(torch.tensor(gain))
        self.inp = nn.Linear(d, d, bias=False)

    def step_det(self, h, x=None):
        hh = h.detach(); ext = self.inp(x) if x is not None else torch.zeros_like(hh)
        return 0.9*torch.tanh(hh@self.W.T+self.b+ext)+0.1*self.g*hh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    D = args.D
    # Load best genome
    genome_path = os.path.join(SCRIPT_DIR, 'runs', 'transplant_stress', 'best_genome_D64.json')
    if os.path.exists(genome_path):
        with open(genome_path) as f: gd = json.load(f)
        sr, gain, sp = gd['sr'], gd['gain'], gd['sp']
    else:
        sr, gain, sp = 0.87, 0.12, 0.54
    print(f"Genome: sr={sr:.4f} gain={gain:.4f} sp={sp:.4f}")

    torch.manual_seed(args.seed)
    core = Core(D, sr, gain, sp, w_seed=args.seed).to(DEV)
    core.eval()

    # Grid: D → roughly square
    import math
    cols = int(math.ceil(math.sqrt(D)))
    rows = int(math.ceil(D / cols))
    print(f"Grid: {rows}x{cols} = {rows*cols} cells (D={D})")

    # Input: simulate 4-char embedding → flattened
    embed_dim = D // 4  # 64
    with torch.no_grad():
        # Create 4 distinct "character embeddings"
        char_embs = torch.randn(4, embed_dim, device=DEV) * 0.5
        x_flat = char_embs.flatten().unsqueeze(0)  # (1, D)
        inp_vec = core.inp(x_flat)

    # Run trajectory
    h = torch.zeros(1, D, device=DEV)
    trajectory = [h.detach().cpu().numpy().flatten()]
    for s in range(args.steps):
        h = core.step_det(h, inp_vec)
        trajectory.append(h.detach().cpu().numpy().flatten())

    # Stats
    print(f"\n{'Step':>5s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'active':>7s}")
    print("-"*50)
    for i, h_flat in enumerate(trajectory):
        print(f"{i:>5d} {h_flat.mean():>+8.4f} {h_flat.std():>8.4f} "
              f"{h_flat.min():>+8.4f} {h_flat.max():>+8.4f} "
              f"{(np.abs(h_flat)>0.1).sum():>5d}/{D}")

    # Per-char-region activity
    print(f"\nPer-character region activity (step {args.steps}):")
    h_final = trajectory[-1]
    for ci in range(4):
        start = ci * embed_dim
        end = start + embed_dim
        region = h_final[start:end]
        print(f"  Char {ci} (neurons {start:3d}-{end:3d}): "
              f"mean={region.mean():+.4f} std={region.std():.4f} "
              f"active={(np.abs(region)>0.1).sum():>2d}/{embed_dim}")

    # ─── PLOT ───
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.animation import FuncAnimation, PillowWriter

    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    Z_frames = []
    for h_flat in trajectory:
        padded = np.zeros(rows * cols)
        padded[:D] = h_flat
        Z_frames.append(padded.reshape(rows, cols))

    all_h = np.concatenate([z.flatten() for z in Z_frames])
    z_abs = max(abs(float(all_h.min())), abs(float(all_h.max())))
    z_lim = (-z_abs, z_abs) if z_abs > 0 else (-1, 1)

    # Color each grid cell by which character region it belongs to
    region_map = np.zeros((rows, cols), dtype=int) - 1
    for i in range(D):
        r, c = i // cols, i % cols
        region_map[r, c] = i // embed_dim  # 0,1,2,3 for 4 chars
    region_colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a', 3: '#984ea3'}
    region_names = {0: 'Char 0', 1: 'Char 1', 2: 'Char 2', 3: 'Char 3'}

    save_dir = os.path.join(SCRIPT_DIR, 'runs', 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    # Static: final frame with region markers
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    Z = Z_frames[-1]
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.85,
                           vmin=z_lim[0], vmax=z_lim[1])
    # Mark region boundaries
    for ci in range(4):
        start_r = (ci * embed_dim) // cols
        end_r = ((ci+1) * embed_dim - 1) // cols
        mid_c = cols // 2
        mid_r = (start_r + end_r) // 2
        ax.text(mid_c, mid_r, Z[mid_r, mid_c] + z_abs*0.3,
                f'Char {ci}', color=region_colors[ci], fontsize=12, fontweight='bold',
                ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.set_title(f'D={D} Final State (step {args.steps}) — Input character regions marked')
    ax.set_xlabel('col'); ax.set_ylabel('row'); ax.set_zlabel('h')
    ax.set_zlim(z_lim)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    path = os.path.join(save_dir, f'D256_final_step{args.steps}.png')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")

    # Animation
    fig = plt.figure(figsize=(12, 10))
    def update(frame_idx):
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')
        Z = Z_frames[frame_idx]
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.9,
                               vmin=z_lim[0], vmax=z_lim[1])
        ax.set_zlim(z_lim)
        h = trajectory[frame_idx]
        ax.set_title(f'D={D} Step {frame_idx} — active={(np.abs(h)>0.1).sum()}/{D}')
        ax.set_xlabel('col'); ax.set_ylabel('row'); ax.set_zlabel('h')
        # Region boundaries as horizontal lines
        for ci in range(1, 4):
            boundary_row = (ci * embed_dim) // cols
            ax.plot([0, cols-1], [boundary_row, boundary_row], [z_lim[0], z_lim[0]],
                    color=region_colors[ci], linewidth=2, linestyle='--', alpha=0.5)
        return [surf]

    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=400, blit=False, repeat=True)
    gif_path = os.path.join(save_dir, f'D256_dynamics.gif')
    anim.save(gif_path, writer=PillowWriter(fps=2.5), dpi=100)
    print(f"Saved: {gif_path}")

    # Also: 2D heatmap of W connectivity (sparsity pattern)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    W_np = core.W.detach().cpu().numpy()
    axes[0].imshow(W_np[:64, :64], cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    axes[0].set_title('W[:64,:64] char0→char0 block')
    axes[1].imshow(W_np, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    axes[1].set_title(f'W full {D}x{D}')
    # Block structure: average abs weight per region
    block_mag = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            ri, rj = i*embed_dim, j*embed_dim
            block_mag[i,j] = np.abs(W_np[ri:ri+embed_dim, rj:rj+embed_dim]).mean()
    im = axes[2].imshow(block_mag, cmap='YlOrRd', aspect='auto')
    axes[2].set_title('Inter-char region |W| mean')
    axes[2].set_xticklabels(['']+[f'C{i}' for i in range(4)])
    axes[2].set_yticklabels(['']+[f'C{i}' for i in range(4)])
    for i in range(4):
        for j in range(4):
            axes[2].text(j, i, f'{block_mag[i,j]:.3f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=axes[2])
    path = os.path.join(save_dir, f'D256_W_structure.png')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
