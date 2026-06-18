"""
3D 地形图：Core 隐藏状态 h 在输入驱动下的动力学演化

把 D 个神经元排成 2D 网格，h[i] 当高度 → 3D surface
每一步传播 = 一张地形图 → 看"山"怎么被输入搅动、怎么传播、怎么消散

用法：
  python visualize_dynamics.py                    # D=64, 交互式 3D
  python visualize_dynamics.py --D 256            # D=256 (16×16)
  python visualize_dynamics.py --save             # 保存图片不弹窗
  python visualize_dynamics.py --animate          # 生成动画 GIF
"""
import torch, torch.nn as nn, numpy as np, os, argparse, json

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

    def step(self, h, x=None):
        h = h.detach(); ext = self.inp(x) if x is not None else torch.zeros_like(h)
        return 0.9*torch.tanh(h@self.W.T+self.b+ext)+0.1*self.g*h+torch.randn_like(h)*0.005

    def step_det(self, h, x=None):
        h = h.detach(); ext = self.inp(x) if x is not None else torch.zeros_like(h)
        return 0.9*torch.tanh(h@self.W.T+self.b+ext)+0.1*self.g*h


def build_grid(d):
    """Map D neurons to a roughly square 2D grid."""
    import math
    cols = int(math.ceil(math.sqrt(d)))
    rows = int(math.ceil(d / cols))
    # Pad to fill grid
    return rows, cols


def run_trajectory(core, inp_vec, n_steps=8, noise=False):
    """Run core from zero init, return h at each step."""
    d = core.W.shape[0]
    h = torch.zeros(1, d, device=DEV)
    trajectory = [h.cpu().numpy().flatten()]
    for s in range(n_steps):
        if noise:
            h = core.step(h, inp_vec)
        else:
            h = core.step_det(h, inp_vec)
        trajectory.append(h.detach().cpu().numpy().flatten())
    return trajectory


def plot_3d_frames(trajectory, d, title_prefix="", save_dir=None):
    """Create a 3D surface plot for each step in trajectory."""
    import matplotlib
    matplotlib.use('TkAgg' if save_dir is None else 'Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    rows, cols = build_grid(d)
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    n_frames = len(trajectory)
    fig_rows = int(np.ceil(n_frames / 2))
    fig = plt.figure(figsize=(14, 4 * fig_rows))

    for i, h_flat in enumerate(trajectory):
        # Pad to fill grid
        padded = np.zeros(rows * cols)
        padded[:d] = h_flat
        Z = padded.reshape(rows, cols)

        ax = fig.add_subplot(fig_rows, min(2, n_frames), i+1, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                               linewidth=0, antialiased=True, alpha=0.9)
        ax.set_title(f'{title_prefix}Step {i}' + (' (init)' if i==0 else ''))
        ax.set_xlabel('col'); ax.set_ylabel('row'); ax.set_zlabel('h')
        ax.set_zlim(-1.0, 1.0)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'dynamics_{d}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()


def plot_single_frame_comparison(traj_with, traj_without, d, step_idx=3, save_dir=None):
    """Side-by-side: with input vs without input at a specific step."""
    import matplotlib
    matplotlib.use('TkAgg' if save_dir is None else 'Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    rows, cols = build_grid(d)
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    fig = plt.figure(figsize=(12, 5))

    for idx, (traj, label) in enumerate([(traj_with, 'With Input'), (traj_without, 'No Input (free run)')]):
        h_flat = traj[min(step_idx, len(traj)-1)]
        padded = np.zeros(rows * cols)
        padded[:d] = h_flat
        Z = padded.reshape(rows, cols)

        ax = fig.add_subplot(1, 2, idx+1, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm if idx==0 else cm.viridis,
                               linewidth=0, antialiased=True, alpha=0.9)
        ax.set_title(f'{label} @ Step {step_idx}')
        ax.set_xlabel('col'); ax.set_ylabel('row'); ax.set_zlabel('h')
        ax.set_zlim(-1.0, 1.0)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'compare_step{step_idx}_{d}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()


def animate_3d(trajectory, d, title="Dynamics", save_path=None, fps=2):
    """Generate animated GIF of 3D surface evolving over time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.animation import FuncAnimation, PillowWriter

    rows, cols = build_grid(d)
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Pre-compute all Z values
    Z_frames = []
    for h_flat in trajectory:
        padded = np.zeros(rows * cols)
        padded[:d] = h_flat
        Z_frames.append(padded.reshape(rows, cols))

    # Global z limits for consistent coloring
    all_h = np.concatenate([z.flatten() for z in Z_frames])
    z_min, z_max = float(all_h.min()), float(all_h.max())
    z_abs = max(abs(z_min), abs(z_max))
    z_lim = (-z_abs, z_abs)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        Z = Z_frames[frame_idx]
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=True, alpha=0.9,
                               vmin=z_lim[0], vmax=z_lim[1])
        ax.set_zlim(z_lim)
        ax.set_title(f'{title}  Step {frame_idx}')
        ax.set_xlabel('col'); ax.set_ylabel('row'); ax.set_zlabel('h')
        # Add stats
        h = trajectory[frame_idx]
        ax.text2D(0.02, 0.98, f'mean={h.mean():+.3f} std={h.std():.3f} active={int((np.abs(h)>0.1).sum())}/{d}',
                  transform=ax.transAxes, fontsize=9, verticalalignment='top')
        return [surf]

    anim = FuncAnimation(fig, update, frames=len(trajectory),
                         interval=int(1000/fps), blit=False, repeat=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        print(f"Animation saved: {save_path}")
        plt.close()
    return anim


def animate_dual(traj_with, traj_without, d, title="Dual View", save_path=None, fps=2):
    """Side-by-side animated 3D: with input vs without."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.animation import FuncAnimation, PillowWriter

    rows, cols = build_grid(d)
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    Z_with = []; Z_without = []
    for h_flat in traj_with:
        padded = np.zeros(rows * cols); padded[:d] = h_flat
        Z_with.append(padded.reshape(rows, cols))
    for h_flat in traj_without:
        padded = np.zeros(rows * cols); padded[:d] = h_flat
        Z_without.append(padded.reshape(rows, cols))

    all_h = np.concatenate([z.flatten() for z in Z_with] + [z.flatten() for z in Z_without])
    z_abs = max(abs(float(all_h.min())), abs(float(all_h.max())))
    z_lim = (-z_abs, z_abs)

    fig = plt.figure(figsize=(18, 8))

    def update(frame_idx):
        fig.clear()
        for idx, (Z_list, label, cmap_name) in enumerate([
            (Z_with, 'With Input', 'coolwarm'),
            (Z_without, 'No Input (free run)', 'viridis')
        ]):
            ax = fig.add_subplot(1, 2, idx+1, projection='3d')
            Z = Z_list[min(frame_idx, len(Z_list)-1)]
            surf = ax.plot_surface(X, Y, Z, cmap=getattr(cm, cmap_name),
                                   linewidth=0, antialiased=True, alpha=0.9,
                                   vmin=z_lim[0], vmax=z_lim[1])
            ax.set_zlim(z_lim)
            ax.set_title(f'{label}  Step {frame_idx}')
            ax.set_xlabel('col'); ax.set_ylabel('row')

    anim = FuncAnimation(fig, update, frames=max(len(traj_with), len(traj_without)),
                         interval=int(1000/fps), blit=False, repeat=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
        print(f"Dual animation saved: {save_path}")
        plt.close()
    return anim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--save", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--animate", action="store_true", help="Generate animated GIF")
    p.add_argument("--dual", action="store_true", help="Side-by-side: with input vs free run")
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    save_dir = os.path.join(SCRIPT_DIR, 'runs', 'visualizations') if args.save else None

    # Load best genome if available, else use defaults
    genome_path = os.path.join(SCRIPT_DIR, 'runs', 'transplant_stress', 'best_genome_D64.json')
    if os.path.exists(genome_path):
        with open(genome_path) as f:
            gd = json.load(f)
        sr, gain, sp = gd['sr'], gd['gain'], gd['sp']
        print(f"Using evolved genome: sr={sr:.4f} gain={gain:.4f} sp={sp:.4f}")
    else:
        sr, gain, sp = 0.87, 0.12, 0.54
        print(f"Using default genome: sr={sr:.4f} gain={gain:.4f} sp={sp:.4f}")

    # Build core
    torch.manual_seed(args.seed)
    core = Core(args.D, sr, gain, sp, w_seed=args.seed).to(DEV)
    core.eval()

    # Create input: simulate a specific character embedding
    # Use the actual inp weights to create a plausible input
    with torch.no_grad():
        inp_vec = core.inp(torch.randn(1, args.D, device=DEV) * 0.5)

    print(f"Core D={args.D} | Grid: {build_grid(args.D)}")

    # Trajectory WITH input
    print("Running with input...")
    traj_with = run_trajectory(core, inp_vec, n_steps=args.steps, noise=False)

    # Trajectory WITHOUT input (free evolution from same init)
    zero_inp = torch.zeros(1, args.D, device=DEV)
    print("Running without input...")
    traj_without = run_trajectory(core, zero_inp, n_steps=args.steps, noise=False)

    # Also show W connectivity as a heatmap
    if args.animate:
        print("Generating animation...")
        if args.dual:
            path = os.path.join(save_dir or os.path.join(SCRIPT_DIR, 'runs', 'visualizations'),
                                f'dual_dynamics_D{args.D}.gif')
            animate_dual(traj_with, traj_without, args.D, save_path=path, fps=args.fps)
        else:
            path = os.path.join(save_dir or os.path.join(SCRIPT_DIR, 'runs', 'visualizations'),
                                f'dynamics_D{args.D}.gif')
            animate_3d(traj_with, args.D, save_path=path, fps=args.fps)
    elif args.compare:
        plot_single_frame_comparison(traj_with, traj_without, args.D, step_idx=3, save_dir=save_dir)
        plot_single_frame_comparison(traj_with, traj_without, args.D, step_idx=args.steps, save_dir=save_dir)
    else:
        plot_3d_frames(traj_with, args.D,
                       title_prefix=f'D={args.D} Input-Driven ',
                       save_dir=save_dir)

    # Print summary
    print("\nTrajectory stats (with input):")
    for i, h in enumerate(traj_with):
        print(f"  Step {i}: mean={h.mean():+.4f} std={h.std():.4f} "
              f"min={h.min():+.4f} max={h.max():+.4f} "
              f"active={(np.abs(h)>0.1).sum():>3d}/{args.D}")

    # W connectivity
    if args.save:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        W_np = core.W.detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im = axes[0].imshow(W_np, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
        axes[0].set_title(f'W matrix ({args.D}x{args.D}) sr={np.max(np.abs(np.linalg.eigvals(W_np))):.3f}')
        plt.colorbar(im, ax=axes[0])
        # Also show eigenvector localization
        _, V = np.linalg.eig(W_np)
        iprs = [(np.abs(V[:,i])**4).sum()/((np.abs(V[:,i])**2).sum()**2+1e-8) for i in range(V.shape[1])]
        axes[1].bar(range(len(iprs)), sorted(iprs, reverse=True), width=1.0)
        axes[1].set_title('Eigenvector IPR (high=localized)')
        axes[1].set_xlabel('eig index (sorted)'); axes[1].set_ylabel('IPR')
        path = os.path.join(save_dir, f'W_analysis_{args.D}.png')
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"Saved: {path}")


if __name__ == '__main__':
    main()
