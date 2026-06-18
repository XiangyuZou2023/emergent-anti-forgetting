"""B1 神经元克隆：D=64 W → D=256 W，每个神经元分裂为4个

规则：
- 神经元 i → [i*4, i*4+1, i*4+2, i*4+3]
- 原始 w[i][j] → 16 条子连接均分 + 小噪声
- 保持 sr/sp 约束
"""
import torch, numpy as np, json, os, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device('cuda')


def clone_W(W_64, target_d=256, noise_std=0.02, sp_target=None, sr_target=None):
    """Neuron cloning: each neuron → factor neurons, weights distributed with noise."""
    d_src = W_64.shape[0]
    factor = target_d // d_src
    assert target_d == d_src * factor, f"target_d must be multiple of source d"

    W_new = np.zeros((target_d, target_d), dtype=np.float32)

    for i in range(d_src):
        for j in range(d_src):
            w_ij = W_64[i, j]
            if abs(w_ij) < 1e-8:
                continue  # zero stays zero

            # Base weight per sub-connection
            base = w_ij / (factor * factor)

            # Distribute across factor×factor sub-connections
            for di in range(factor):
                for dj in range(factor):
                    noise = np.random.randn() * noise_std * abs(base)
                    W_new[i*factor+di, j*factor+dj] = base + noise

    # Apply sparsity constraint (randomly zero out weakest weights)
    if sp_target is not None:
        nonzero = np.abs(W_new) > 1e-8
        n_nonzero = nonzero.sum()
        target_nonzero = int(target_d * target_d * (1 - sp_target))
        if n_nonzero > target_nonzero:
            # Zero out the weakest connections
            abs_flat = np.abs(W_new.flatten())
            threshold = np.sort(abs_flat[abs_flat > 1e-8])[n_nonzero - target_nonzero]
            W_new[np.abs(W_new) < threshold] = 0

    # Normalize spectral radius
    if sr_target is not None:
        s = np.linalg.norm(W_new, 2)
        if s > 1e-8:
            W_new = W_new * (sr_target / s)

    return W_new


def compare_blocks(W64, W256, n_chars=4):
    """Compare block-wise coupling between source and cloned W."""
    src_dim = W64.shape[0] // n_chars
    tgt_dim = W256.shape[0] // n_chars

    blocks_64 = np.zeros((n_chars, n_chars))
    blocks_256 = np.zeros((n_chars, n_chars))

    for i in range(n_chars):
        for j in range(n_chars):
            bi, bj = i*src_dim, j*src_dim
            blocks_64[i, j] = np.abs(W64[bi:bi+src_dim, bj:bj+src_dim]).mean()
            bi256, bj256 = i*tgt_dim, j*tgt_dim
            blocks_256[i, j] = np.abs(W256[bi256:bi256+tgt_dim, bj256:bj256+tgt_dim]).mean()

    block_diff = np.abs(blocks_256 - blocks_64).mean()
    return blocks_64, blocks_256, block_diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_d", type=int, default=256)
    p.add_argument("--noise", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load best genome
    genome_path = os.path.join(SCRIPT_DIR, 'runs', 'transplant_stress', 'best_genome_D64.json')
    with open(genome_path) as f:
        gd = json.load(f)
    sr, gain, sp = gd['sr'], gd['gain'], gd['sp']
    print(f"Genome: sr={sr:.4f} gain={gain:.4f} sp={sp:.4f}")

    # Build D=64 core, get W
    from transplant_stress import Core
    core64 = Core(64, sr, gain, sp, w_seed=args.seed)
    W_64 = core64.W.detach().cpu().numpy()
    print(f"W_64: sr={np.max(np.abs(np.linalg.eigvals(W_64))):.3f} "
          f"sparsity={(np.abs(W_64)<1e-8).mean():.3f}")

    # Clone to D=256
    W_256 = clone_W(W_64, target_d=args.target_d, noise_std=args.noise,
                    sp_target=sp, sr_target=sr)
    print(f"W_256: sr={np.max(np.abs(np.linalg.eigvals(W_256))):.3f} "
          f"sparsity={(np.abs(W_256)<1e-8).mean():.3f}")

    # Compare blocks
    b64, b256, diff = compare_blocks(W_64, W_256)
    print(f"\nBlock |W| comparison:")
    print(f"  Source (64):\n{b64}")
    print(f"  Clone  (256):\n{b256}")
    print(f"  Mean diff: {diff:.6f}")

    # Build cloned Core and upload
    core256 = Core(args.target_d, sr, gain, sp, w_seed=args.seed)
    core256.W.data = torch.tensor(W_256, dtype=torch.float32)

    # Upload both
    from idea.viz import upload_to_viewer
    upload_to_viewer(core64, label=f'B1_source_D64', n_steps=12, device='cpu')
    upload_to_viewer(core256, label=f'B1_clone_D{args.target_d}_noise{args.noise}', n_steps=12, device='cpu')

    print(f"\nCompare in viz: B1_source_D64 vs B1_clone_D{args.target_d}")
    print("Check: 3D Surface (terrain similarity), Block |W| (coupling match)")
    print("If terrain has visible correlation → B1 works")


if __name__ == '__main__':
    main()
