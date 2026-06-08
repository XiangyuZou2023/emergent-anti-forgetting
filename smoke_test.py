"""Smoke test: verifies environment works in ~30 seconds."""
import torch, numpy as np, sys

print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")

# Quick architecture test
D = 64
W = torch.randn(D, D, device=DEV) * 0.3
s = torch.linalg.norm(W, 2)
W = W * (0.9 / (s + 1e-8))

h = torch.randn(4, D, device=DEV)
for _ in range(40):
    h = 0.9 * torch.tanh(h @ W.T) + 0.1 * 0.1 * h

S = 1 - torch.mean(torch.norm(h[:-1] - h[1:], dim=-1)) / 2
A_star = 4 * (1 - S.item()) * S.item() / 0.3
print(f"A* = {A_star:.3f} (expect ~0.5-3.0)")

# Quick GA test
class Genome:
    def __init__(self):
        self.sr = np.clip(0.9 + np.random.uniform(-0.05, 0.05), 0.1, 2.0)
        self.gain = np.clip(0.1 + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
        self.sp = np.clip(0.5 + np.random.uniform(-0.1, 0.1), 0.0, 0.9)

pop = [Genome() for _ in range(10)]
for g in range(3):
    scores = [np.random.random() for _ in pop]
    survivors = [pop[i] for i in np.argsort(scores)[-5:]]
    pop = survivors + [Genome() for _ in range(5)]
    print(f"  gen {g}: best={max(scores):.3f}")

print("\n✅ Environment OK. Ready to run experiments.")
print("   Full ablation: docker compose --profile gpu up ablation  (~2h GPU / ~20h CPU)")
print("   Pre-computed:  results are in results.jsonl (from original paper runs)")
