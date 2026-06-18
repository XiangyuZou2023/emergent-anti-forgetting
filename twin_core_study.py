"""双胞胎实验：同(sr,gain,sp)，不同W种子，不同训练数据 → W谱结构有共同点吗？

这是山3的核心问题——genome参数编码的是"功能蓝图"还是仅仅"动力学稳定性"？
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, os, random, time
from collections import Counter

DEV = torch.device('cuda')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'runs', 'twin_study')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Data ───
DATA_PATH = os.path.join(SCRIPT_DIR, 'sample_data.txt')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw = f.read(300_000)
cnt = Counter(raw)
all_chars = sorted([c for c, n in cnt.items() if n >= 5 and '一' <= c <= '鿿'])
mid = len(all_chars) // 2

def make_taskAB(vocab, char_set):
    chars = ['<PAD>', '<UNK>'] + char_set[:vocab]
    c2i = {c: i for i, c in enumerate(chars)}
    ids = [c2i.get(c, 1) for c in raw if '一' <= c <= '鿿'][:20000]
    Xs, Ya, Yb = [], [], []
    for i in range(len(ids) - 6):
        Xs.append(ids[i:i+4]); Ya.append(ids[i+4]); Yb.append(ids[i+5])
    return (torch.tensor(Xs), torch.tensor(Ya), torch.tensor(Yb), len(chars))

# ─── Model ───
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

# ─── Spectrum ───
def analyze_W(W_np):
    eigs = np.linalg.eigvals(W_np); U,S,Vh = np.linalg.svd(W_np,full_matrices=False)
    abs_eigs = np.abs(eigs); real_eigs = np.real(eigs)
    s_norm = S/(S.sum()+1e-8); participation = 1.0/((s_norm**2).sum()+1e-8)
    sorted_abs = np.sort(abs_eigs)[::-1]
    gap_01 = sorted_abs[0]-sorted_abs[1] if len(sorted_abs)>1 else 0
    _, V = np.linalg.eig(W_np)
    iprs = [(np.abs(V[:,i])**4).sum()/((np.abs(V[:,i])**2).sum()**2+1e-8) for i in range(V.shape[1])]
    return {"spectral_radius":float(np.max(abs_eigs)),
            "mean_abs_eig":float(np.mean(abs_eigs)),
            "effective_rank":int(np.sum(S>0.01*S[0])),
            "participation":float(participation),
            "spectral_gap":float(gap_01/(sorted_abs[0]+1e-8)),
            "mean_ipr":float(np.mean(iprs)),
            "sv_decay":[float(S[i]/(S[0]+1e-8)) for i in [0,1,2,4,8,16,32,63]],
            "eig_real_hist": np.histogram(real_eigs, bins=50, range=(-1,1))[0].tolist(),
    }

# ─── Train with W ───
def train_core(core, vocab, char_set_train, char_set_test, n_steps=400):
    d = core.W.shape[0]
    X_tr, YA_tr, YB_tr, V = make_taskAB(vocab, char_set_train)
    X_te, YA_te, YB_te, _ = make_taskAB(vocab, char_set_test)

    emb = nn.Embedding(V, d//4).to(DEV); ro = nn.Linear(d, V).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters())+list(ro.parameters())+list(core.parameters()), lr=0.005)
    B=32

    for step in range(n_steps):
        idx = torch.randint(0, len(X_tr), (B,))
        bx, by = X_tr[idx].to(DEV), YA_tr[idx].to(DEV)
        e = emb(bx).flatten(1); hh = torch.zeros(B, d, device=DEV)
        for _ in range(3): hh = core.step(hh, e)
        loss = F.cross_entropy(ro(hh), by)
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def acc(X, Y):
        c,t=0,0
        for i in range(0, min(len(X),500), B):
            bx,by=X[i:i+B].to(DEV),Y[i:i+B].to(DEV)
            e=emb(bx).flatten(1); hh=torch.zeros(len(bx),d,device=DEV)
            for _ in range(3): hh=core.step(hh,e)
            c+=(ro(hh).argmax(-1)==by).sum().item(); t+=len(by)
        return c/t
    return acc(X_tr, YA_tr), acc(X_te, YA_te)

# ─── Compare two Cores ───
def compare_W_spectra(W1, W2):
    """Structural similarity between two W matrices."""
    U1,S1,_ = np.linalg.svd(W1, full_matrices=False)
    U2,S2,_ = np.linalg.svd(W2, full_matrices=False)
    # Singular value correlation
    sv_corr = np.corrcoef(S1[:30], S2[:30])[0,1]
    # Subspace overlap: principal angles between top-k subspaces
    k = min(10, len(S1))
    overlap = np.linalg.norm(U1[:,:k].T @ U2[:,:k], 2)  # 1=identical, 0=orthogonal
    return float(sv_corr), float(overlap)

# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
def main():
    # Load best genome
    genome_path = os.path.join(SCRIPT_DIR, 'runs', 'transplant_stress', 'best_genome_D64.json')
    with open(genome_path) as f: gd = json.load(f)
    sr, gain, sp = gd['sr'], gd['gain'], gd['sp']
    print(f"Genome: sr={sr:.4f} gain={gain:.4f} sp={sp:.4f}")

    # Split data into 3 disjoint sets
    third = len(all_chars) // 3
    char_sets = [
        all_chars[:third],
        all_chars[third:2*third],
        all_chars[2*third:]
    ]
    # Test set is always the last portion
    test_set = all_chars[2*third:mid]

    print(f"Data split: train1={len(char_sets[0])} train2={len(char_sets[1])} test={len(test_set)} chars")

    N_TWINS = 3  # 3 Cores with same genome, different W seeds, different data
    results = []

    for i in range(N_TWINS):
        w_seed = 42 + i * 100
        train_chars = char_sets[i % 2]  # Core 0 & 2 share data, Core 1 different

        print(f"\n--- Core {i}: w_seed={w_seed}, train_chars[{i%2}] ---")
        core = Core(64, sr, gain, sp, w_seed=w_seed).to(DEV)

        # Pre-training W spectrum
        W_pre = core.W.detach().cpu().numpy()
        spec_pre = analyze_W(W_pre)
        print(f"  W pre:  sr={spec_pre['spectral_radius']:.3f} eff_rank={spec_pre['effective_rank']} participation={spec_pre['participation']:.1f}")

        # Train
        vocab = 30
        t0 = time.time()
        train_acc, test_acc = train_core(core, vocab, train_chars, test_set, n_steps=400)
        dt = time.time() - t0
        print(f"  Train: acc_train={train_acc:.4f} acc_test={test_acc:.4f} ({dt:.0f}s)")

        # Post-training W spectrum
        W_post = core.W.detach().cpu().numpy()
        spec_post = analyze_W(W_post)
        print(f"  W post: sr={spec_post['spectral_radius']:.3f} eff_rank={spec_post['effective_rank']} participation={spec_post['participation']:.1f}")

        results.append({
            "core_id": i,
            "w_seed": w_seed,
            "train_chars_set": i % 2,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "W_pre": spec_pre,
            "W_post": spec_post,
        })

    # ─── Cross-comparison ───
    print("\n=== Cross-Core W Comparison ===")
    Ws_post = [np.array(r['W_post']) for r in results]
    # Actually we need the actual W matrices, not just spectra
    # Re-load from cores — but we lost them. Use stored spectra instead.

    print(f"\n{'Pair':>8s} {'sv_corr':>9s} {'subspace_overlap':>10s}")
    print("-"*35)
    # Rebuild Cores to get actual W
    cores_post = []
    for i in range(N_TWINS):
        w_seed = 42 + i * 100
        train_chars = char_sets[i % 2]
        core = Core(64, sr, gain, sp, w_seed=w_seed).to(DEV)
        train_core(core, 30, train_chars, test_set, n_steps=400)
        cores_post.append(core.W.detach().cpu().numpy())

    for i in range(N_TWINS):
        for j in range(i+1, N_TWINS):
            sv_corr, overlap = compare_W_spectra(cores_post[i], cores_post[j])
            shared_data = "same_data" if results[i]["train_chars_set"] == results[j]["train_chars_set"] else "diff_data"
            print(f"Core{i}-Core{j} ({shared_data:>9s}): {sv_corr:>+.4f}     {overlap:>.4f}")

    # ─── Save ───
    # Also save pre-training Ws for comparison
    cores_pre = []
    for i in range(N_TWINS):
        core = Core(64, sr, gain, sp, w_seed=42+i*100).to(DEV)
        cores_pre.append(core.W.detach().cpu().numpy())

    pre_comparisons = []
    post_comparisons = []
    for i in range(N_TWINS):
        for j in range(i+1, N_TWINS):
            sv_pre, ov_pre = compare_W_spectra(cores_pre[i], cores_pre[j])
            sv_post, ov_post = compare_W_spectra(cores_post[i], cores_post[j])
            pre_comparisons.append({"pair": f"{i}-{j}", "sv_corr": sv_pre, "overlap": ov_pre})
            post_comparisons.append({"pair": f"{i}-{j}", "sv_corr": sv_post, "overlap": ov_post})

    output = {
        "config": {"sr": sr, "gain": gain, "sp": sp, "N_TWINS": N_TWINS, "D": 64},
        "individual_results": results,
        "pre_training_pairwise": pre_comparisons,
        "post_training_pairwise": post_comparisons,
        "key_question": "Does training push W spectra toward a common structure (convergence) or keep them distinct?"
    }
    with open(os.path.join(OUT_DIR, 'twin_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=float)

    # Summary
    print("\n=== Summary ===")
    print("Pre-training: random W with same (sr,gain,sp) — spectra should be similar (same constraints)")
    print("Post-training: W adapted to different data — do spectra converge or diverge?")
    for pc in pre_comparisons:
        print(f"  Pre  {pc['pair']}: sv_corr={pc['sv_corr']:.4f} overlap={pc['overlap']:.4f}")
    for pc in post_comparisons:
        print(f"  Post {pc['pair']}: sv_corr={pc['sv_corr']:.4f} overlap={pc['overlap']:.4f}")

    print(f"\nSaved: {OUT_DIR}/twin_results.json")


if __name__ == "__main__":
    main()
