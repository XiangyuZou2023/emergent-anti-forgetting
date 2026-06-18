"""
做法A 移植研究：D=64 best genome → D=256 直接移植，诊断死因

关键约束（用户指定）：
1. 必须用 Mountain 1 泛化设置（train domain A, test domain B）
   不泛化 = 纯背诵 = W 内部无信息结构
2. 研究"怎么死的"，不是"能不能活"
3. 对比 D=64 原版 vs D=256 移植版的全部光谱/动力学数据
4. 所有数据存盘，为做法B（谱插值）提供设计依据

输出：
  runs/transplant_study/
  ├── config.json           ← 实验参数
  ├── best_genome_D64.json  ← D=64 最优 genome 参数
  ├── native_D64.json       ← D=64 原版全部诊断
  ├── transplant_D256.json  ← D=256 移植版全部诊断
  └── summary.txt           ← 人类可读总结
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sys, os, json, numpy as np, random, time, argparse
from collections import Counter

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'runs', 'transplant_study')
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════
# DATA — Mountain 1 generalization split
# ═══════════════════════════════════════════════════
DATA_PATH = os.path.join(SCRIPT_DIR, 'sample_data.txt')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw = f.read(300_000)
cnt = Counter(raw)
all_chars = sorted([c for c, n in cnt.items() if n >= 5 and '一' <= c <= '鿿'])
mid = len(all_chars) // 2
TRAIN_CHARS = all_chars[:mid]
TEST_CHARS  = all_chars[mid:]
print(f"Chars: {len(all_chars)} total → {len(TRAIN_CHARS)} train / {len(TEST_CHARS)} test")


def make_taskAB(vocab_size, char_set):
    """Build 4-gram next-char prediction from specified character set."""
    chars = ['<PAD>', '<UNK>'] + char_set[:vocab_size]
    c2i = {c: i for i, c in enumerate(chars)}
    ids = [c2i.get(c, 1) for c in raw if '一' <= c <= '鿿'][:20000]
    Xs, Ya, Yb = [], [], []
    for i in range(len(ids) - 6):
        Xs.append(ids[i:i+4])
        Ya.append(ids[i+4])
        Yb.append(ids[i+5])
    return (torch.tensor(Xs), torch.tensor(Ya), torch.tensor(Yb), len(chars))


# ═══════════════════════════════════════════════════
# MODEL — identical to run_ablations.py Core (with self.inp)
# ═══════════════════════════════════════════════════
class Core(nn.Module):
    def __init__(self, d=64, sr=0.9, gain=0.1, sp=0.5, seed=None):
        super().__init__()
        self.d = d
        if seed is not None:
            torch.manual_seed(seed)
        W = torch.randn(d, d) * 0.3
        mask = (torch.rand(d, d) > sp).float()
        W = W * mask
        s = torch.linalg.norm(W, 2)
        self.W = nn.Parameter(W * (sr / (s + 1e-8)))
        self.b = nn.Parameter(torch.zeros(d))
        self.g = nn.Parameter(torch.tensor(gain))
        self.inp = nn.Linear(d, d, bias=False)

    def step(self, h, x=None):
        h = h.detach()
        ext = torch.zeros_like(h)
        if x is not None:
            ext = self.inp(x)
        return 0.9 * torch.tanh(h @ self.W.T + self.b + ext) \
               + 0.1 * self.g * h \
               + torch.randn_like(h) * 0.005

    def step_det(self, h, x=None):
        """Deterministic step for spectrum/stability analysis."""
        h = h.detach()
        ext = torch.zeros_like(h)
        if x is not None:
            ext = self.inp(x)
        return 0.9 * torch.tanh(h @ self.W.T + self.b + ext) + 0.1 * self.g * h


# ═══════════════════════════════════════════════════
# A* GATE — dynamical stability measure
# ═══════════════════════════════════════════════════
def fast_A_det(core, h, n=40):
    """A* = trajectory stability (deterministic, no noise)."""
    for _ in range(10):
        h = core.step_det(h)
    tr = []
    for _ in range(n):
        h = core.step_det(h)
        tr.append(h / (h.norm(dim=-1, keepdim=True) + 1e-8))
    ds = [0.5 * (tr[i] - tr[i+1]).norm(dim=-1).mean().item()
          for i in range(len(tr)-1)]
    S = 1 - min(np.mean(ds), 1.)
    return 4 * (1 - S) * S / 0.3


# ═══════════════════════════════════════════════════
# SPECTRUM DIAGNOSTICS
# ═══════════════════════════════════════════════════
def analyze_spectrum(W_np, label=""):
    """Full spectrum diagnostics."""
    eigs = np.linalg.eigvals(W_np)
    U, S, Vh = np.linalg.svd(W_np, full_matrices=False)

    # Eigenvalue stats
    abs_eigs = np.abs(eigs)
    real_eigs = np.real(eigs)

    # Participation ratio (inverse Herfindahl of normalized singular values)
    s_norm = S / (S.sum() + 1e-8)
    participation = 1.0 / ((s_norm ** 2).sum() + 1e-8)
    effective_rank = int(np.sum(S > 0.01 * S[0]))

    # Spectral gap
    sorted_abs = np.sort(abs_eigs)[::-1]
    gap_01 = sorted_abs[0] - sorted_abs[1] if len(sorted_abs) > 1 else 0
    gap_ratio = gap_01 / (sorted_abs[0] + 1e-8)

    # Localization: mean inverse participation ratio of eigenvectors
    # High IPR = localized (few entries large), Low IPR = delocalized
    _, V = np.linalg.eig(W_np)
    iprs = []
    for i in range(V.shape[1]):
        v = V[:, i]
        ipr = (np.abs(v) ** 4).sum() / ((np.abs(v) ** 2).sum() ** 2 + 1e-8)
        iprs.append(float(ipr))
    mean_ipr = np.mean(iprs)

    # Sparsity
    sparsity = float((np.abs(W_np) < 1e-8).mean())

    return {
        "label": label,
        "D": W_np.shape[0],
        "spectral_radius": float(np.max(abs_eigs)),
        "mean_abs_eig": float(np.mean(abs_eigs)),
        "std_abs_eig": float(np.std(abs_eigs)),
        "max_real_eig": float(np.max(real_eigs)),
        "effective_rank": effective_rank,
        "participation_ratio": float(participation),
        "spectral_gap_01": float(gap_01),
        "spectral_gap_ratio": float(gap_ratio),
        "mean_ipr": float(mean_ipr),
        "sparsity": sparsity,
        "top10_sv": [float(x) for x in S[:10].tolist()],
        "sv_decay": [float(S[i] / (S[0] + 1e-8)) for i in [0, 1, 2, 4, 8, 16, 32, 63]],
    }


# ═══════════════════════════════════════════════════
# H DYNAMICS DIAGNOSTICS
# ═══════════════════════════════════════════════════
@torch.no_grad()
def analyze_dynamics(core, n_samples=100):
    """Analyze h dynamics: effective dim, norm flow, correlation decay."""
    d = core.W.shape[0]
    h = torch.randn(n_samples, d, device=DEV) * 0.5

    # Record trajectory over 20 deterministic steps
    norms, dims = [], []
    prev_h = h.clone()
    for step in range(20):
        h = core.step_det(h)
        norms.append(float(h.norm(dim=-1).mean()))
        # Effective dimension via participation ratio of covariance
        h_centered = h - h.mean(dim=0, keepdim=True)
        cov = h_centered.T @ h_centered / (n_samples - 1)
        eigs = torch.linalg.eigvalsh(cov)
        eigs_pos = eigs[eigs > 1e-8]
        if len(eigs_pos) > 0:
            p = eigs_pos / eigs_pos.sum()
            eff_dim = float(1.0 / (p**2).sum())
        else:
            eff_dim = 0.0
        dims.append(eff_dim)

    # Autocorrelation decay
    h_final = h
    # Re-run from same init, compare
    h2 = torch.randn(n_samples, d, device=DEV) * 0.5
    h2_initial = h2.clone()
    for _ in range(10):
        h2 = core.step_det(h2)
    corr_decay = float(F.cosine_similarity(
        h2_initial.flatten().unsqueeze(0),
        h2.flatten().unsqueeze(0)).item())

    return {
        "norm_trajectory": norms,
        "eff_dim_trajectory": dims,
        "final_norm": norms[-1] if norms else 0,
        "norm_divergence": norms[-1] - norms[0] if len(norms) > 1 else 0,
        "autocorr_10step": corr_decay,
        "eff_dim_mean": float(np.mean(dims[5:])) if len(dims) > 5 else float(np.mean(dims)),
    }


# ═══════════════════════════════════════════════════
# TRAINING + EVALUATION (Mountain 1: cross-domain generalization)
# ═══════════════════════════════════════════════════
def train_eval_cross_domain(core, vocab, n_train=200, n_test=200, B=32):
    """
    Train on TRAIN_CHARS domain, evaluate on TEST_CHARS domain.
    Returns: train_acc (on A), test_acc (on B), test_ret_acc, loss_curve
    """
    d = core.W.shape[0]
    X_tr, YA_tr, YB_tr, V = make_taskAB(vocab, TRAIN_CHARS)
    X_te, YA_te, YB_te, _ = make_taskAB(vocab, TEST_CHARS)

    emb = nn.Embedding(V, d // 4).to(DEV)
    ro = nn.Linear(d, V).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ro.parameters()), lr=0.01)

    loss_curve = []

    # Train on domain A (train chars)
    for X, Y, n in [(X_tr, YA_tr, n_train), (X_tr, YB_tr, n_train)]:
        perm = torch.randperm(len(X))[:n]
        for i in range(0, len(perm), B):
            idx = perm[i:i+B]
            bx, by = X[idx].to(DEV), Y[idx].to(DEV)
            e = emb(bx).flatten(1)
            hh = torch.zeros(len(bx), d, device=DEV)
            for _ in range(3):
                hh = core.step(hh, e)
            loss = F.cross_entropy(ro(hh), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_curve.append(float(loss.item()))

    # Test on domain B (test chars — unseen characters!)
    @torch.no_grad()
    def test_acc(X, Y):
        correct, total = 0, 0
        for i in range(0, len(X), B):
            bx, by = X[i:i+B].to(DEV), Y[i:i+B].to(DEV)
            e = emb(bx).flatten(1)
            hh = torch.zeros(len(bx), d, device=DEV)
            for _ in range(3):
                hh = core.step(hh, e)
            correct += (ro(hh).argmax(-1) == by).sum().item()
            total += len(by)
        return correct / total

    acc_A = test_acc(X_tr, YA_tr)
    acc_B = test_acc(X_te, YA_te)
    ret_acc = test_acc(X_te, YA_te)  # re-test on test domain

    # Retrain briefly on A, re-test on B
    opt_ret = torch.optim.Adam(list(emb.parameters()) + list(ro.parameters()), lr=0.01)
    perm = torch.randperm(len(X_tr))[:50]
    for i in range(0, len(perm), B):
        idx = perm[i:i+B]
        bx, by = X_tr[idx].to(DEV), YA_tr[idx].to(DEV)
        e = emb(bx).flatten(1)
        hh = torch.zeros(len(bx), d, device=DEV)
        for _ in range(3):
            hh = core.step(hh, e)
        loss = F.cross_entropy(ro(hh), by)
        opt_ret.zero_grad()
        loss.backward()
        opt_ret.step()
    retB_acc = test_acc(X_te, YA_te)

    return {
        "train_acc_A": acc_A,
        "test_acc_B": acc_B,
        "ret_acc_B": retB_acc,
        "forgetting": acc_B - retB_acc,
        "loss_curve": loss_curve,
    }


# ═══════════════════════════════════════════════════
# GA: find best D=64 genome (Mountain 1 generalization)
# ═══════════════════════════════════════════════════
class Genome:
    d = 64

    def __init__(self, sr=0.9, gain=0.1, sp=0.5):
        self.sr = np.clip(sr + random.uniform(-0.05, 0.05), 0.5, 2.0)
        self.gain = np.clip(gain + random.uniform(-0.05, 0.05), 0.0, 1.0)
        self.sp = np.clip(sp + random.uniform(-0.1, 0.1), 0.0, 0.9)

    def mutate(self, r=0.3):
        if random.random() < r:
            self.sr = np.clip(self.sr + random.uniform(-0.2, 0.2), 0.5, 2.0)
        if random.random() < r:
            self.gain = np.clip(self.gain + random.uniform(-0.2, 0.2), 0.0, 1.0)
        if random.random() < r:
            self.sp = np.clip(self.sp + random.uniform(-0.2, 0.2), 0.0, 0.9)

    def build(self, seed=None):
        return Core(self.d, self.sr, self.gain, self.sp, seed=seed).to(DEV)

    @staticmethod
    def crossover(a, b):
        c = Genome()
        c.sr = a.sr if random.random() < 0.5 else b.sr
        c.gain = a.gain if random.random() < 0.5 else b.gain
        c.sp = a.sp if random.random() < 0.5 else b.sp
        return c

    def to_dict(self):
        return {"sr": self.sr, "gain": self.gain, "sp": self.sp, "d": self.d}


def measure_generalize(ind, vocab, cfg):
    """Mountain 1: train on TRAIN_CHARS, evaluate generalization on TEST_CHARS."""
    core = ind.build()
    d = ind.d

    # A* gate
    A_star = 2.0
    if cfg.get('use_astar', True):
        h = torch.randn(4, d, device=DEV) * 0.5
        A_star = fast_A_det(core, h, 40)
        if A_star < 1.0:
            return A_star, -1.0, 0, 0, 0, 0

    X_tr, YA_tr, YB_tr, V = make_taskAB(vocab, TRAIN_CHARS)
    X_te, YA_te, YB_te, _ = make_taskAB(vocab, TEST_CHARS)

    emb = nn.Embedding(V, d // 4).to(DEV)
    ro = nn.Linear(d, V).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters()) + list(ro.parameters()), lr=0.01)

    def train_eval(tX, tY, n=200):
        perm = torch.randperm(len(tX))[:n]
        for i in range(0, len(perm), 32):
            idx = perm[i:i+32]
            bx, by = tX[idx].to(DEV), tY[idx].to(DEV)
            e = emb(bx).flatten(1)
            hh = torch.zeros(len(bx), d, device=DEV)
            for _ in range(3):
                hh = core.step(hh, e)
            loss = F.cross_entropy(ro(hh), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            tidx = torch.randint(0, len(tX), (200,))
            bx, by = tX[tidx].to(DEV), tY[tidx].to(DEV)
            e = emb(bx).flatten(1)
            hh = torch.zeros(200, d, device=DEV)
            for _ in range(3):
                hh = core.step(hh, e)
            return (ro(hh).argmax(-1) == by).float().mean().item()

    # Train on TRAIN, test on TEST (generalization!)
    train_eval(X_tr, YA_tr, 200)
    train_eval(X_tr, YB_tr, 200)
    acc_A = train_eval(X_te, YA_te, 200)  # test on unseen chars
    acc_B = train_eval(X_te, YB_te, 200)
    ret_A = train_eval(X_te, YA_te, 50)

    w_acc = acc_A * 0.2 + acc_B * 0.2 + ret_A * 0.6
    hc = ind.sr * d * (1 - ind.sp) + ind.gain * 10
    lp = 0.0
    if cfg.get('use_tax', True):
        if hc > 30:
            lp = (hc - 30) * 0.003
        elif hc < 8:
            lp = (8 - hc) * 0.01
    return A_star, w_acc - lp, acc_A, acc_B, ret_A, hc


def evolve_best_genome(seed=42, pop=20, gens=30, target_vocab=30):
    """Quick GA to find best D=64 genome with Mountain 1 generalization."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    pop_list = [Genome() for _ in range(pop)]
    vocab = 10
    best_ind, best_fit = None, -99
    cfg = {'use_astar': True, 'use_tax': True}

    for gen in range(gens):
        scores, As, accAs, retAs, hs = [], [], [], [], []
        for ind in pop_list:
            A, fit, aA, aB, aR, h_val = measure_generalize(ind, vocab, cfg)
            scores.append(fit)
            As.append(A)
            accAs.append((aA + aB) / 2)
            retAs.append(aR)
            hs.append(h_val)

        alive_rate = sum(1 for a in As if a >= 1.0) / pop * 100
        med_ret = np.median(retAs)
        med_h = np.median([h for h in hs if h < 900])

        bi = np.argmax(scores)
        if scores[bi] > best_fit:
            best_fit = scores[bi]
            best_ind = pop_list[bi]

        # Selection
        fit_cutoff = np.percentile([s for s in scores if s >= 0], 30)
        lean_hi = np.percentile(hs, 70)
        lean_lo = np.percentile(hs, 10)
        keep = [i for i, s in enumerate(scores)
                if As[i] >= 1.0 and s >= fit_cutoff
                and lean_lo <= hs[i] <= lean_hi]
        if len(keep) < pop * 0.2:
            keep = list(range(pop))

        # Adaptive vocab
        if alive_rate > 30 and med_ret > 0.5 and 10 < med_h < 40:
            vocab = min(target_vocab, vocab + 3)
        elif alive_rate < 10 or med_ret < 0.2:
            vocab = max(10, vocab - 3)

        survivors = [pop_list[i] for i in keep]
        new_pop = [pop_list[i] for i in keep[:max(1, len(keep)//4)]]
        while len(new_pop) < pop:
            if len(survivors) < 2:
                break
            p1, p2 = random.sample(survivors, 2)
            c = Genome.crossover(p1, p2)
            c.mutate(0.4)
            new_pop.append(c)
        while len(new_pop) < pop:
            new_pop.append(Genome())
        pop_list = new_pop

        if gen % 10 == 0:
            print(f"  GA g{gen}: vocab={vocab} alive={alive_rate:.0f}% "
                  f"best_fit={scores[bi]:.4f} med_h={med_h:.1f}")

    print(f"  GA done: vocab={vocab} best_fit={best_fit:.4f} "
          f"best=({best_ind.sr:.3f}, {best_ind.gain:.3f}, {best_ind.sp:.3f})")
    return best_ind, best_fit


# ═══════════════════════════════════════════════════
# TRANSPLANT: create D=256 version with same (sr,gain,sp)
# ═══════════════════════════════════════════════════
def transplant(best_genome, target_d=256, n_copies=3):
    """Create D=target_d Core with same (sr,gain,sp), random W (different seeds)."""
    cores = []
    for i in range(n_copies):
        core = Core(
            d=target_d,
            sr=best_genome.sr,
            gain=best_genome.gain,
            sp=best_genome.sp,
            seed=42 + i * 100
        ).to(DEV)
        cores.append(core)
    return cores


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_d", type=int, default=256)
    p.add_argument("--vocab", type=int, default=30)
    p.add_argument("--skip_ga", action="store_true",
                   help="Skip GA, load best genome from file")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"=== 做法A 移植研究: D=64 → D={args.target_d} ===")
    print(f"Device: {DEV}  Vocab: {args.vocab}")
    print()

    # ─── Step 1: Get best D=64 genome ───
    genome_path = os.path.join(OUT_DIR, 'best_genome_D64.json')
    if args.skip_ga and os.path.exists(genome_path):
        with open(genome_path) as f:
            gd = json.load(f)
        best = Genome()
        Genome.d = 64
        best.sr = gd['sr']
        best.gain = gd['gain']
        best.sp = gd['sp']
        best.d = 64
        print(f"Loaded best genome: sr={best.sr:.4f} gain={best.gain:.4f} sp={best.sp:.4f}")
    else:
        print("Evolving best D=64 genome (Mountain 1 generalization)...")
        best, best_fit = evolve_best_genome(args.seed, pop=20, gens=30, target_vocab=args.vocab)
        with open(genome_path, 'w') as f:
            json.dump(best.to_dict(), f, indent=2)
        print(f"Saved: {genome_path}")

    # ─── Step 2: Build native D=64 core ───
    print("\n─── D=64 Native Diagnostics ───")
    core64 = best.build(seed=args.seed)

    # Spectrum BEFORE training
    spec64_pre = analyze_spectrum(core64.W.detach().cpu().numpy(), "D64_pre")
    print(f"  W spectrum (pre): sr={spec64_pre['spectral_radius']:.3f} "
          f"eff_rank={spec64_pre['effective_rank']} "
          f"participation={spec64_pre['participation_ratio']:.1f}")

    # A* gate
    h_init = torch.randn(4, 64, device=DEV) * 0.5
    A64 = fast_A_det(core64, h_init, 40)
    print(f"  A* gate: {A64:.3f}")

    # Dynamics
    dyn64_pre = analyze_dynamics(core64)
    print(f"  Dynamics: norm_final={dyn64_pre['final_norm']:.3f} "
          f"eff_dim={dyn64_pre['eff_dim_mean']:.1f}")

    # Train (Mountain 1 generalization)
    print("  Training (cross-domain)...")
    t0 = time.time()
    train64 = train_eval_cross_domain(core64, args.vocab, n_train=200)
    dt64 = time.time() - t0
    print(f"  Train A acc: {train64['train_acc_A']:.4f}  "
          f"Test B acc: {train64['test_acc_B']:.4f}  "
          f"Ret B acc: {train64['ret_acc_B']:.4f}  "
          f"Forgetting: {train64['forgetting']:+.4f}  ({dt64:.1f}s)")

    # Spectrum AFTER training
    spec64_post = analyze_spectrum(core64.W.detach().cpu().numpy(), "D64_post")
    print(f"  W spectrum (post): sr={spec64_post['spectral_radius']:.3f} "
          f"eff_rank={spec64_post['effective_rank']} "
          f"participation={spec64_post['participation_ratio']:.1f}")

    # ─── Step 3: Transplant to D=256 ───
    print(f"\n─── D={args.target_d} Transplant Diagnostics ───")
    transplant_cores = transplant(best, target_d=args.target_d, n_copies=3)

    transplant_results = []
    for i, tcore in enumerate(transplant_cores):
        print(f"\n  Transplant copy {i+1}:")
        # Spectrum BEFORE
        spec_pre = analyze_spectrum(tcore.W.detach().cpu().numpy(), f"D{args.target_d}_pre_seed{i}")
        print(f"    W spectrum (pre): sr={spec_pre['spectral_radius']:.3f} "
              f"eff_rank={spec_pre['effective_rank']} "
              f"participation={spec_pre['participation_ratio']:.1f} "
              f"gap_ratio={spec_pre['spectral_gap_ratio']:.3f}")

        # A* gate
        h_init = torch.randn(4, args.target_d, device=DEV) * 0.5
        A_val = fast_A_det(tcore, h_init, 40)
        print(f"    A* gate: {A_val:.3f} " + ("OK" if A_val >= 1.0 else "DEAD"))

        # Dynamics
        dyn_pre = analyze_dynamics(tcore)
        print(f"    Dynamics: norm_final={dyn_pre['final_norm']:.3f} "
              f"eff_dim={dyn_pre['eff_dim_mean']:.1f} "
              f"norm_div={dyn_pre['norm_divergence']:+.3f}")

        # Train — even if A* dead, try training to see what happens
        print(f"    Training (cross-domain)...")
        t0 = time.time()
        try:
            train_res = train_eval_cross_domain(tcore, args.vocab, n_train=200)
            dt = time.time() - t0
            print(f"    Train A acc: {train_res['train_acc_A']:.4f}  "
                  f"Test B acc: {train_res['test_acc_B']:.4f}  "
                  f"Ret B acc: {train_res['ret_acc_B']:.4f}  "
                  f"Forgetting: {train_res['forgetting']:+.4f}  ({dt:.1f}s)")
        except Exception as e:
            train_res = {"error": str(e)}
            print(f"    Training failed: {e}")

        # Spectrum AFTER
        spec_post = analyze_spectrum(tcore.W.detach().cpu().numpy(), f"D{args.target_d}_post_seed{i}")
        print(f"    W spectrum (post): sr={spec_post['spectral_radius']:.3f} "
              f"eff_rank={spec_post['effective_rank']} "
              f"participation={spec_post['participation_ratio']:.1f}")

        transplant_results.append({
            "copy": i,
            "sr": best.sr, "gain": best.gain, "sp": best.sp,
            "A_star": float(A_val),
            "spectrum_pre": spec_pre,
            "spectrum_post": spec_post,
            "dynamics_pre": dyn_pre,
            "training": {k: v for k, v in train_res.items() if k != 'loss_curve'},
        })

    # ─── Step 4: Save everything ───
    print(f"\n─── Saving to {OUT_DIR} ───")

    native_result = {
        "sr": best.sr, "gain": best.gain, "sp": best.sp, "D": 64,
        "A_star": float(A64),
        "spectrum_pre": spec64_pre,
        "spectrum_post": spec64_post,
        "dynamics_pre": dyn64_pre,
        "training": {k: v for k, v in train64.items() if k != 'loss_curve'},
    }
    with open(os.path.join(OUT_DIR, 'native_D64.json'), 'w') as f:
        json.dump(native_result, f, indent=2, default=float)

    config = {
        "experiment": "transplant_study_method_A",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "target_D": args.target_d,
        "vocab": args.vocab,
        "seed": args.seed,
        "method": "Direct parameter transplant (sr,gain,sp) with random W",
        "mountain1_generalization": True,
    }
    with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    transplant_full = {
        "source_genome": best.to_dict(),
        "target_D": args.target_d,
        "copies": transplant_results,
    }
    with open(os.path.join(OUT_DIR, f'transplant_D{args.target_d}.json'), 'w') as f:
        json.dump(transplant_full, f, indent=2, default=float)

    # Summary
    lines = []
    lines.append(f"=== 做法A 移植研究总结 ===")
    lines.append(f"日期: {config['date']}")
    lines.append(f"")
    lines.append(f"D=64 最优 genome: sr={best.sr:.4f} gain={best.gain:.4f} sp={best.sp:.4f}")
    lines.append(f"  A*: {A64:.3f}")
    lines.append(f"  泛化训练: A acc={train64['train_acc_A']:.4f} B acc={train64['test_acc_B']:.4f}")
    lines.append(f"")
    lines.append(f"D={args.target_d} 移植 (3 copies):")
    for i, tr in enumerate(transplant_results):
        lines.append(f"  Copy {i+1}: A*={tr['A_star']:.3f}")
        if 'test_acc_B' in tr.get('training', {}):
            lines.append(f"    泛化训练: B acc={tr['training']['test_acc_B']:.4f}")
        else:
            lines.append(f"    训练: {tr.get('training', {}).get('error', '?')}")
        lines.append(f"    W pre: sr={tr['spectrum_pre']['spectral_radius']:.3f} "
                     f"eff_rank={tr['spectrum_pre']['effective_rank']} "
                     f"participation={tr['spectrum_pre']['participation_ratio']:.1f}")
    lines.append(f"")
    lines.append(f"关键诊断数据已保存至: {OUT_DIR}")
    lines.append(f"  - native_D64.json: D=64 原版全谱+动力学+训练")
    lines.append(f"  - transplant_D{args.target_d}.json: D={args.target_d} 移植版×3全谱+动力学+训练")
    lines.append(f"  - best_genome_D64.json: D=64 最优参数")
    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    for line in lines:
        print(line)
    print("\nDone.")


if __name__ == '__main__':
    main()
