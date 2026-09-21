# Previous Experiments & New Experiments Setup

## Context

This document summarizes the conversation about configuring `experiments.py` for a long-run sweep of the DSM-EDA pipeline (Star Discrepancy Problem, dimension 2).

---

## Existing Results (experiments_dim2.csv)

Full grid: 2 learning methods × 2 warm_start × 2 svd_noise = **8 configs**, n ∈ {20, 50, 100, 180}, 60 generations, population_size=60, selection_size=25%, workers=9.

| n | learning_method | warm_start | svd_noise | best_fitness | wall_clock_seconds |
|---|---|---|---|---|---|
| 20 | birkhoff | False | False | 0.063959 | 445.5 |
| 20 | birkhoff | False | True  | 0.067708 | 651.5 |
| 20 | birkhoff | True  | False | 0.064531 | 557.8 |
| 20 | birkhoff | True  | True  | 0.068421 | 626.1 |
| 20 | pbil     | False | False | 0.063392 | 331.1 |
| 20 | pbil     | False | True  | 0.066973 | 662.9 |
| 20 | pbil     | True  | False | 0.061904 | 150.9 |
| 20 | pbil     | True  | True  | 0.067741 | 647.3 |
| 50 | birkhoff | False | False | 0.037101 | 4144.1 |
| 50 | birkhoff | False | True  | 0.038208 | 4087.0 |
| 50 | birkhoff | True  | False | 0.036335 | 4035.6 |
| 50 | birkhoff | True  | True  | 0.037048 | 4055.5 |
| 50 | pbil     | False | False | 0.033608 | 3692.3 |
| 50 | pbil     | False | True  | 0.038260 | 4045.7 |
| 50 | pbil     | True  | False | 0.027999 | 2459.6 |
| 50 | pbil     | True  | True  | 0.037542 | 4046.9 |
| 100 | birkhoff | False | False | 0.024568 | 8214.5 |
| 100 | birkhoff | False | True  | 0.024242 | 8029.8 |
| 100 | birkhoff | True  | False | 0.024806 | 8057.1 |
| 100 | birkhoff | True  | True  | 0.025191 | 8001.2 |
| 100 | pbil     | False | False | 0.022760 | 7705.2 |
| 100 | pbil     | False | True  | 0.025597 | 8093.2 |
| 100 | pbil     | True  | False | 0.019085 | 8441.0 |
| 100 | pbil     | True  | True  | 0.025052 | 7969.9 |
| 180 | birkhoff | False | False | 0.017790 | 16036.5 |
| 180 | birkhoff | False | True  | 0.017545 | 15784.2 |
| 180 | birkhoff | True  | False | 0.018312 | 15794.4 |
| 180 | birkhoff | True  | True  | 0.017055 | 15628.4 |
| 180 | pbil     | False | False | 0.016161 | 16357.5 |
| 180 | pbil     | False | True  | 0.017499 | 15816.0 |
| 180 | pbil     | True  | False | 0.014569 | 18389.4 |
| 180 | pbil     | True  | True  | 0.018184 | 15672.1 |

---

## New Long-Run Configuration (`--long` flag)

### Change made to `experiments.py`

Added a `--long` CLI flag that overrides defaults:

- **n values**: `{20, 25}` (dropped n=50, n=100, n=180)
- **generations**: 500 (up from 60)
- **output**: `results/experiments_long/experiments_long.csv`
- **log dir**: `logs/experiments_long`

All other parameters (population_size, selection_size, workers, etc.) remain as passed or defaulted.

### How to run

```bash
# Long-run sweep (n=20, n=25, 500 generations, all 8 configs):
python experiments.py --long

# With custom workers or repeats:
python experiments.py --long --workers 8 --repeats 3
```

### Runtime Estimate

Based on actual wall-clock times at 60 generations, scaled linearly to 500 (×8.33):

| n | Avg wall-clock @ 60 gen | Scaled to 500 gen | × 8 configs |
|---|---|---|---|
| 20 | ~478s | ~3,980s (~1.1h) | ~8.9h |
| 25 | ~1,000–1,500s (interpolated) | ~8,300–12,500s | ~18–28h |

**Total estimated runtime: ~27–37 hours**

> Note: Scaling is optimistic — later generations may be harder for Gurobi to prove global optimality. Consider using `--max-duration` to cap per-run wall-clock if needed.

---

## Key Observations from Previous Results

- **PBIL + warm start + no SVD** consistently achieves the best (lowest) fitness values across all n.
- **SVD noise injection** tends to increase wall-clock time (~1.5–2× for n=20) without clear fitness improvement.
- Wall-clock time scales roughly with n²–n³ between n=20 and n=180, dominated by Gurobi QCQP solve time.
- The pipeline is CPU-bound (Gurobi branch-and-bound); parallelism via `ProcessPoolExecutor` across workers is the main lever.
