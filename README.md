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
| `requirements.txt` | Python dependencies |

## Quick Start

```bash
pip install -r requirements.txt

# Run ablation (GPU recommended, ~1.2 GPU-hours total)
python run_ablations.py

# Run A* gating comparison
python test_astar_matters.py
python run_astar_extra.py

# Generate figures
python figures.py
python fig_astar.py
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
