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
| `results.jsonl` | Ablation raw data (18 records) |
| `astar_5seeds.json` | A* gating combined results (5 seeds) |
| `gruau_result.json` | Gruau architecture learnability test |
| `moe_result.json` | MoE survival-driven specialization |
| `references.bib` | All citations in BibTeX format |
| `sample_data.txt` | Chinese text corpus sample (300K chars) |
| `requirements.txt` | Python dependencies |

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

### Ablation: Evolution = Zero Forgetting
All 4 evolving configurations → forgetting ≤ 0.
Fixed architectures → forgetting +0.02.

### A* Gating: Prevents Population Collapse
WITH gate: 94% alive, vocab reaches 58.
WITHOUT gate: 3% alive, vocab stuck at 12.

### Gruau Architecture Learnability
Evolved architecture: 100% convergence (10/10 trials).
Random architecture: 39% convergence (0/20).

### MoE Survival-Driven Specialization
Single expert: 0.360. 10-expert ensemble: 0.950 (+0.590).

## License

MIT
