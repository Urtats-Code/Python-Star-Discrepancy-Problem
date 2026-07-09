# Star Discrepancy Minimization via EDA

This project implements an **Estimation of Distribution Algorithm (EDA)** to minimize the **Star Discrepancy** of a point set in [0,1]². It combines evolutionary heuristics with an exact **Semidefinite/Quadratic Programming (SDP/MIQCP)** formulation solved via Gurobi.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Mathematical Formulation](#mathematical-formulation)
- [Pipeline Overview](#pipeline-overview)
- [Algorithm Details](#algorithm-details)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Visualization](#visualization)
- [Grid Search](#grid-search)
- [Dependencies](#dependencies)

---

## Problem Statement

Given a set of `n` points in the unit square [0,1]², the **star discrepancy** measures how far the empirical distribution of those points deviates from the uniform distribution. Formally, for a point set `P = {p_1, ..., p_n}`:

```
D*(P) = sup_{B ⊆ [0,1]²} | A(B, P)/n - Vol(B) |
```

where `A(B, P)` is the number of points in box `B` and `Vol(B)` is its volume. The supremum is taken over all axis-aligned boxes anchored at the origin (i.e., of the form [0, x] × [0, y]).

The goal is to **find a permutation** (representing the relative ordering of points) that **minimizes D***. This is NP-hard in general, so the project uses an EDA to guide a search over the combinatorial space of permutations.

---

## Mathematical Formulation

### From Permutation to Binary Matrix

Each candidate solution is a permutation `σ` of `{1, ..., n}`. It is converted to an `(n+1) × (n+1)` binary matrix `A` where:

- `A[σ(i), i] = 1` for each `i ∈ {0, ..., n-1}` (one point per row/column)
- Dummy points are added at boundaries: `A[0][n] = 1` and `A[n][0] = 1`

### SDP Optimization (per permutation)

Given the binary matrix `A`, the star discrepancy is computed by solving the following optimization problem (a Mixed-Integer Quadratically Constrained Program, MIQCP):

**Decision variables:**
- `f` — scalar, the star discrepancy value (objective)
- `x[0..n]` — x-coordinates of points in [0,1]
- `y[0..n]` — y-coordinates of points in [0,1]

**Objective:**
```
Minimize: f
```

**Constraints:**

**(2a) Upper discrepancy bound:**
```
(1/n) * Σ_{u≤i, v≤j} A[u,v]  -  x[i] * y[j]  ≤  f + w_ij
```
where `w_ij = 2 - col_sums[i,j] - row_sums[i,j]` accounts for the binary matrix structure.

**(2b) Lower discrepancy bound:**
```
-(1/n) * Σ_{u<i, v<j} A[u,v]  +  x[i] * y[j]  ≤  f + w'_ij
```

**(2c) Boundary conditions:**
```
x[n] = 1,   y[n] = 1
```

**(2d) Ordering in x:**
```
x[i+1] - x[i] ≥ ε   for all i ∈ {0, ..., n-2}
```

**(2e) Ordering in y:**
```
y[i+1] - y[i] ≥ ε   for all i ∈ {0, ..., n-2}
```

where `ε` (default `0.0001`) enforces a minimum separation between consecutive coordinates.

**Efficiency:** Prefix sum arrays `sum_auv`, `col_sums`, `row_sums` are precomputed in O(n²) so each constraint parameter lookup is O(1).

**Solver settings:** Gurobi with `NonConvex=2` (global optimization), `MIPGap=0.0`, `TimeLimit=600s`, `Threads=1`.

### Doubly Stochastic Matrix (DSM)

The EDA maintains an `n × n` **Doubly Stochastic Matrix** (DSM) `M` where:

- `M[i, j]` = probability that position `i` in the permutation is assigned to element `j`
- All entries are non-negative
- All row sums equal 1
- All column sums equal 1

The DSM starts uniform (`M[i,j] = 1/n`) and is iteratively updated based on elite permutations to concentrate probability mass on high-quality orderings.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    EDA Main Loop                            │
│                                                             │
│  DSM (uniform init)                                         │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────┐       │
│  │  1. Sample population of N permutations from DSM │       │
│  └──────────────────────────────────────────────────┘       │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────┐       │
│  │  2. Evaluate fitness in parallel (SDP per perm.) │       │
│  │     (cached: skip already-evaluated permutations)│       │
│  └──────────────────────────────────────────────────┘       │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────┐       │
│  │  3. Select top-k permutations (elite set)        │       │
│  └──────────────────────────────────────────────────┘       │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────┐       │
│  │  4. Update DSM (Birkhoff or PBIL)                │       │
│  └──────────────────────────────────────────────────┘       │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────┐       │
│  │  5. Log generation data to HDF5                  │       │
│  └──────────────────────────────────────────────────┘       │
│       │                                                     │
│       └──────────────── repeat for G generations ──────────┘
│                                                             │
│  After all generations: Visualize & export results          │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-step

1. **Initialization** — Load config, initialize DSM as `1/n` everywhere.
2. **Sampling** — For each row `i`, sample column `j` from `DSM[i, available_cols]` (without replacement) to produce a valid permutation.
3. **Parallel Evaluation** — Each permutation is evaluated by solving the MIQCP via Gurobi. Already-seen permutations are served from cache. Workers use a persistent Gurobi environment to reduce overhead.
4. **Selection** — Retain the top `selection_size` permutations (lowest discrepancy = best).
5. **DSM Update** — Learn a new DSM from the elite permutations using one of two strategies (see below).
6. **Logging** — Write population, fitness scores, best fitness, and DSM to an HDF5 file (background thread).
7. **Repeat** for `generations` iterations.

---

## Algorithm Details

### Learning Strategies

#### Birkhoff Learning

```
dsm = (alpha / n) * ones(n, n)          # small uniform base
freq = frequency_matrix(elite_perms)    # count elite co-occurrences
dsm += freq * (1 - alpha) / m          # add weighted elite signal
```

- `alpha` controls exploration vs. exploitation (default `0.1`)
- The result is guaranteed doubly stochastic by construction

#### PBIL (Population-Based Incremental Learning)

```
target = frequency_matrix(elite_perms) / m
dsm = (1 - lr) * dsm + lr * target     # exponential smoothing
dsm = (1 - mr) * dsm + mr * (1/n)     # mutation toward uniform
```

- `learning_rate` controls step size toward elite signal (default `0.1`)
- `mutation_rate` injects diversity to prevent premature convergence (default `0.01`)

### Caching

`PermutationHashMap` stores `(permutation tuple → fitness score)` to avoid redundant SDP calls across generations. This is critical since the same permutation can appear in multiple generations.

### Persistent Solver

`PersistentSDPSolver` maintains a single Gurobi model per worker process, dynamically updating constraints between evaluations rather than rebuilding the model from scratch, reducing license and initialization overhead.

---

## Project Structure

```
.
├── DSM_EDA_Pipeline.py       # Main EDA pipeline (entry point)
├── SDPProblem.py             # SDP/MIQCP problem definition (Gurobi)
├── PersistentSDPSolver.py    # Reusable Gurobi solver for parallel eval
├── PermutationHashMap.py     # Permutation evaluation cache
├── GenerationLogger.py       # Async HDF5 logging per generation
├── Visualize.py              # Plotting and animation utilities
├── GridSearch.py             # Hyperparameter grid search
├── QuasiMontecarlo.py        # Quasi-Monte Carlo area estimation demo
├── generation_jsonify.py     # Convert HDF5 logs to JSON
├── verify_visualize.py       # Visualization testing script
├── configs/
│   ├── config.yaml           # Main pipeline configuration
│   └── config-grid-search.yaml  # Grid search parameter space
├── logs/                     # HDF5 generation logs (output)
└── plots/                    # Visualization outputs (output)
```

---

## Configuration

Edit `configs/config.yaml` before running:

```yaml
pipeline:
  n: 8                        # Number of points in [0,1]²
  population_size: 200        # Permutations sampled per generation
  selection_size: 20          # % of population selected as elite
  learning_method: "birkhoff" # "birkhoff" or "pbil"

run:
  generations: 500            # Total EDA generations
  epsilon: 0.0001             # Min separation between coordinates
  num_workers: null           # Parallel workers (null = auto)

birkhoff:
  alpha: 0.1                  # Exploration parameter

pbil:
  learning_rate: 0.1          # DSM update step size
  mutation_rate: 0.01         # Mutation probability

logging:
  gen_log_file: "./logs/generation_log.h5"

visualization:
  enabled: true
  save_animation: true
  save_combined_animation: true
```

---

## Running the Pipeline

```bash
python DSM_EDA_Pipeline.py --config configs/config.yaml
```

Results are written to `logs/generation_log.h5`. If `visualization.enabled: true`, plots and animations are automatically generated in `plots/`.

---

## Visualization

Run manually on an existing log file:

```bash
python Visualize.py --file logs/generation_log.h5 --action <action>
```

| Action | Description |
|--------|-------------|
| `fitness` | Line plot of best fitness per generation |
| `dsm` | Heatmap of the final DSM |
| `history` | Animated DSM evolution across generations |
| `both` | Side-by-side static: DSM heatmap + fitness plot |
| `both-history` | Synchronized animation: DSM + fitness evolution |

**Log to JSON export:**
```bash
python generation_jsonify.py --file logs/generation_log.h5
```

**HDF5 log structure:**
```
metadata/           <- run parameters (n, population_size, method, ...)
generation_000000/
  ├── population    (population_size x n)
  ├── fitness       (population_size,)
  ├── best_fitness  (scalar)
  └── dsm           (n x n)
generation_000001/
  ...
```

---

## Grid Search

To sweep hyperparameters, configure `configs/config-grid-search.yaml` and run:

```bash
python GridSearch.py --config configs/config-grid-search.yaml
```

Example search space:
```yaml
pipeline:
  n: [8]
  population_size: [20, 50, 100, 200]
  selection_size: [10, 20, 30, 50]
  learning_method: ["birkhoff", "pbil"]
run:
  generations: [50, 100, 200]
```

Results are written to `grid_search_results.csv` with columns for each hyperparameter and the best fitness achieved.

---

## Dependencies

- **Python 3.9+**
- **Gurobi** (with valid license) — for MIQCP solving
- `numpy`
- `scipy` — quasi-Monte Carlo sequences
- `h5py` — HDF5 log I/O
- `matplotlib` — plotting and animation
- `pyyaml` — config parsing
- `ffmpeg` — MP4 animation export (system dependency)

Install Python dependencies:
```bash
pip install numpy scipy h5py matplotlib pyyaml gurobipy
```
