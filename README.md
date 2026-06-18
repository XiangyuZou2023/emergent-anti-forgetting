# Emergent Anti-Forgetting through Autopoietic Evolution

Reproduction code and data for the paper.

## Files

| File | Purpose |
|------|---------|
| `run_ablations.py` | Main ablation experiment (6 configs × 3 seeds) |
| `test_astar_matters.py` | A* gating comparison (WITH vs WITHOUT, 2 seeds) |
| `run_astar_extra.py` | A* gating additional seeds (3 more) |
| `figures.py` | Generate paper figures 1-4 from results.jsonl |
| `fig_astar.py` | Generate A* gating figure 5 |
| `results.jsonl` | Ablation raw data (24 records) |
| `astar_5seeds.json` | A* gating combined results (5 seeds) |
| `gruau_result.json` | Gruau architecture learnability test |
| `moe_result.json` | MoE survival-driven specialization |
| `references.bib` | All citations in BibTeX format |
| `sample_data.txt` | Chinese text corpus sample (300K chars) |
| `requirements.txt` | Python dependencies |
| `survival_axioms.lean` | Lean 4 formalization (Supplementary Material B) |
| `survival_proof/` | Full Lean 4 project with lake dependencies |

## Quick Start

### Smoke test (verify environment, ~30s)

```bash
# GPU
docker compose --profile gpu up smoke

# CPU-only
docker compose up smoke-cpu
```

### Option A: Use pre-computed results (instant)

The file `results.jsonl` contains all 18 ablation runs from the paper. Figures can be generated directly:

```bash
docker compose up figures        # → figures_out/
```

### Option B: Reproduce from scratch

```bash
# GPU (~2 hours): requires nvidia-container-toolkit
docker compose --profile gpu up ablation

# CPU (~20 hours): no GPU needed
docker compose up ablation-cpu

# A* gating experiment (~20 min GPU)
docker compose --profile gpu up astar
```

> ⚠️ The ablation runs 6 configs × 3 seeds = 18 experiments. On CPU this is slow but works — the code auto-detects CUDA and falls back to CPU. All results match the paper's `results.jsonl` within statistical noise (random seed variation).

### Option C: Manual setup

```bash
pip install torch numpy matplotlib
python run_ablations.py           # ~1.2 GPU-hours
python test_astar_matters.py      # A* gating
python figures.py                 # Generate figures
```

## Key Results

### Reproduce Paper Tables & Figures

| Paper Element | Command | Output | Time |
|--------------|---------|--------|------|
| **Table 1** (Ablation, 6 configs×3 seeds) | `python run_ablations.py --config all` (run each config separately) | `results.jsonl` | ~1.2 GPU-hr |
| **Table 2** (A* gating, 5 seeds) | `python test_astar_matters.py && python run_astar_extra.py` | `astar_5seeds.json` | ~0.3 GPU-hr |
| **Figure 1-4** (Forgetting, accuracy, ablation, violin) | `python figures.py` | `figures/fig1-4_*.pdf` | <1 min |
| **Figure 5** (A* gating) | `python fig_astar.py` | `figures/fig5_astar_gating.pdf` | <1 min |
| **High-Variance Exploration** (Section 4.5) | `python test_high_variance.py` | console output + `hv_results.json` | ~13 min GPU |
| **Diversity vs A* Gate** (Section 4.2 supplement) | `python test_diversity_astar.py` | console output | ~5 min GPU |
| **Niche Protection Variants** | `python test_diversity_tough.py` | console output | ~8 min GPU |
| **text8 High-Variance** | `python test_text8_hv.py` | console output | ~10 min GPU |
| **Smoke test** (verify env) | `python smoke_test.py` | pass/fail | ~30s |

### Expected Results

#### Ablation (Table 1): Evolution = Zero Forgetting
All 4 evolving configurations → forgetting ≤ 0.
Fixed architectures → forgetting +0.02.

#### A* Gating (Table 2): Prevents Population Collapse
WITH gate: 94% alive, vocab reaches 58.
WITHOUT gate: 3% alive, vocab stuck at 12.

#### High-Variance Exploration (Section 4.5)
Standard search: vocab ~52. High-variance search: vocab ~175 (+237%).
Zero forgetting maintained. A* > 2.0. Confirms ceiling is search-limited.

#### text8 High-Variance
All search strategies stuck at vocab=7, acc~0.57.
Capacity-limited (D=64 single-layer), not search-limited.

#### MoE Survival-Driven Specialization
Single expert: 0.360. 10-expert ensemble: 0.950 (+0.590).
Expert specialization via capacity distribution (10×640 vs 64 dims).

## License

MIT
