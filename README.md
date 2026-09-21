# Python Star Discrepancy Problem — DSM-EDA Pipeline

A Doubly Stochastic Matrix (DSM) Estimation of Distribution Algorithm (EDA)
for the permutation stage of the two-step Star Discrepancy Problem (SDP)
reformulation: an outer EDA searches over permutations, and each sampled
permutation is scored by fixing it in a Gurobi continuous NLP that optimizes
the coordinates (`PersistentSDPSolver.py`).

This README covers environment setup on both **Windows** and **macOS**, the
configuration system, and how to run every experiment in the project.

---

## 1. Prerequisites

- **Python 3.10** (the pinned dependency versions in `requirements.txt`,
  notably `numpy==2.2.6` and `gurobipy==13.0.2`, are validated against 3.10).
- **A Gurobi license.** `gurobipy` is installed from PyPI by
  `requirements.txt`, but solving anything requires an activated license
  (a free academic license works). See [Section 4](#4-gurobi-license).
- Git, to clone the repository.

> **Important — don't share a `venv/` across operating systems.** A
> virtualenv is not portable between Windows and macOS (it embeds
> OS-specific interpreter paths and compiled `.so`/`.pyd`/`.dylib` binaries
> for packages like `numpy` and `gurobipy`). `venv/` is already listed in
> `.gitignore`, so it won't be tracked by git — but if you copy this project
> folder directly from one machine to another (rather than cloning fresh),
> delete any existing `venv/` first and recreate it locally as shown below.
> A venv built on macOS will fail to import on Windows (and vice versa) with
> errors like `ImportError: DLL load failed` or missing `.so` files.

---

## 2. Setup on macOS

```bash
# from the project root
python3 --version        # confirm 3.10.x

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

To leave the virtualenv later: `deactivate`.

---

## 3. Setup on Windows

Using **PowerShell**:

```powershell
# from the project root
python --version          # confirm 3.10.x

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

If `Activate.ps1` is blocked by the execution policy, either run PowerShell
as the current user with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

or use the `cmd.exe` activator instead:

```cmd
venv\Scripts\activate.bat
```

To leave the virtualenv later: `deactivate`.

---

## 4. Gurobi license

`pip install gurobipy` installs the solver library, but every solve needs a
license. The most common options:

- **Academic named-user license** (recommended if you have a `.edu`-style
  institutional email): register at
  [gurobi.com/free-trial](https://www.gurobi.com/free-trial/), then run
  `grbgetkey <your-key>` (installed alongside `gurobipy` in the venv) to
  activate it — this works identically on Windows and macOS.
- **License file**: if you were given a `gurobi.lic` file directly, point
  the `GRB_LICENSE_FILE` environment variable at it:
  - macOS/bash: `export GRB_LICENSE_FILE=/path/to/gurobi.lic`
  - Windows/PowerShell: `$env:GRB_LICENSE_FILE = "C:\path\to\gurobi.lic"`

Verify the license works:

```bash
python -c "import gurobipy as gp; m = gp.Model(); print('Gurobi OK')"
```

If this raises a license error, fix that before running anything below —
every run in this repo calls Gurobi on every evaluated permutation.

---

## 5. Configuration files

All single-run experiments are driven by a YAML config under `configs/`.
Three pre-built configs are provided:

| File | Purpose |
|---|---|
| `configs/config.yaml` | Full run: `n=8`, 500 generations, no time limit. |
| `configs/config-smoke-test.yaml` | Quick validation: same as above but capped at 90 s wall-clock. |
| `configs/config-grid-search.yaml` | Reference showing the hyperparameter search space (lists per field). |

### 5.1 Configuration sections and keys

| Section | Key | Type | Meaning |
|---|---|---|---|
| `pipeline` | `n` | int | Number of points (SDP instance size). |
| | `population_size` | int | EDA population size λ. |
| | `selection_size` | int | Elite size as a **percentage** of the population (e.g. `25` = top 25 %). |
| | `learning_method` | str | `"birkhoff"` or `"pbil"`. |
| `run` | `generations` | int | Maximum number of generations to run. |
| | `epsilon` | float | Minimum coordinate spacing in the Step-2 NLP (default `0.0001`). |
| | `num_workers` | int \| null | CPU worker processes for parallel evaluation; `null` = `cpu_count() − 1`. |
| | `time_limit` | float \| null | Per-permutation Gurobi solve cap in seconds. `null` = solve to proven optimality (fine for small `n`, very slow for large `n`). |
| | `max_duration_seconds` | float \| null | Wall-clock budget for the **whole run**. When reached, the generation loop stops cleanly and keeps the current best DSM/solution and all completed generations in the log. `null` = run all `generations`. |
| `birkhoff` | `alpha` | float | Learning rate for the Birkhoff DSM update (default `0.1`). |
| `pbil` | `alpha` | float | Smoothing factor toward the best-so-far solution in PBIL (default `0.1`). |
| `warm_start` | `enabled` | bool | Bias the initial DSM toward a Fibonacci-pattern permutation instead of a uniform start. |
| | `alpha0` | float | Mixing weight in M⁰ = α₀ · Ū + (1−α₀) · P_σ₀ (default `0.3`). |
| | `shift` | int | Starting index offset for the Fibonacci sequence (default `1`). |
| `svd` | `enabled` | bool | Apply SVD noise-injection perturbation each generation to counter premature convergence. |
| | `theta` | float | Standard deviation of the noise added to singular values (default `0.3`). |
| | `apply_every` | int | Apply the perturbation every N generations (default `1`). |
| `sinkhorn` | `iterations` | int | Maximum Sinkhorn-Knopp iterations to restore doubly stochasticity (default `10000`). |
| `logging` | `gen_log_file` | str | Path where the per-generation HDF5 log is written. |
| `visualization` | `enabled` | bool | Whether to generate plots after the run. |
| | `output_dir` | str | Directory where plots are saved. |
| | `plot_fitness` | bool | Fitness evolution line chart. |
| | `plot_dsm` | bool | Final DSM heatmap. |
| | `plot_combined` | bool | Fitness curve + DSM side-by-side. |
| | `save_animation` | bool | Animated MP4 of DSM evolution. |
| | `save_combined_animation` | bool | Animated MP4 of fitness + DSM combined. |

---

## 6. Running the pipeline (single run)

```bash
python DSM_EDA_Pipeline.py --config configs/config.yaml
```

This prints per-generation progress (`Gen 000 | Best Fitness: ... | Diversity: ...`),
then a final best permutation/fitness summary. Results are logged to the HDF5
file named in `logging.gen_log_file`, and — if `visualization.enabled: true` —
plots are written to `visualization.output_dir`.

### 6.1 Smoke test

`configs/config-smoke-test.yaml` mirrors a full-size config (`n=8`,
`generations=500`, `population_size=200`) but caps the run at
`max_duration_seconds: 90` so it validates the whole pipeline quickly:

```bash
python DSM_EDA_Pipeline.py --config configs/config-smoke-test.yaml
```

The log is written to `logs/smoke_test_log.h5` and plots go to `plots/smoke_test/`
(static images only — MP4 animations are disabled).

Inspect the result to check whether all generations completed or the run hit
the time budget:

```bash
python generation_jsonify.py --file logs/smoke_test_log.h5
# look at the "summary" section: "stopped_early" and "generations_completed"
```

---

## 7. Running the experiment grid (`experiments.py`)

`experiments.py` runs a systematic **2×2×2 grid** of configurations:

- **Learning method**: Birkhoff vs PBIL
- **Initialisation**: cold start (uniform DSM) vs warm start (Fibonacci-biased DSM)
- **Exploration**: no SVD noise vs SVD noise injection

That gives 8 configurations, each run across a list of `n` values.

### 7.1 Quick wiring check

```bash
python experiments.py --dry-run
```

Runs one configuration at `n=20` for 3 generations to confirm the pipeline
wires up correctly before committing to a long sweep.

### 7.2 Full grid sweep

```bash
python experiments.py --output results/experiments_dim2.csv
```

Defaults: `n ∈ {20, 50, 100, 180}`, 8 configurations each — 32 runs total.
Results are **appended** to the CSV as each run finishes, so an interrupted
sweep keeps whatever it already completed.

### 7.3 Long-run preset

```bash
python experiments.py --long
```

Preset for 500-generation runs at `n ∈ {20, 25}`. Logs go to
`logs/experiments_long/` and results to `results/experiments_long/experiments_long.csv`.

### 7.4 Custom sweep

```bash
python experiments.py \
  --n-values 20,50 \
  --generations 60 \
  --population-size 60 \
  --selection-size 25 \
  --workers 4 \
  --time-limit 15 \
  --max-duration 3600 \
  --repeats 3 \
  --seed 42 \
  --log-dir logs/my_experiment \
  --output results/my_experiment.csv
```

### 7.5 Available flags

| Flag | Default | Meaning |
|---|---|---|
| `--n-values` | `20,50,100,180` | Comma-separated list of `n` values to sweep. |
| `--generations` | 60 | Generations per run. |
| `--population-size` | 60 | EDA population size λ. |
| `--selection-size` | 25 | Elite percentage (0–100). |
| `--workers` | `cpu_count() − 1` | Parallel CPU worker processes. |
| `--time-limit` | `15.0` | Per-evaluation Gurobi cap in seconds. |
| `--max-duration` | `None` | Wall-clock budget per run in seconds. |
| `--repeats` | 1 | Independent reruns per configuration. |
| `--seed` | 0 | Base random seed (incremented per run). |
| `--log-dir` | `logs/experiments` | Directory for per-run HDF5 logs. |
| `--output` | — | CSV file to append results to. |
| `--long` | — | Preset: 500 gens, `n ∈ {20, 25}`, long-run paths. |
| `--dry-run` | — | 3 gens, pop=10, one config — wiring check only. |

### 7.6 Output

- **CSV**: one row per run with `n`, `learning_method`, `warm_start`, `svd_noise`,
  `repeat`, `population_size`, `selection_size_pct`, `generations`, `time_limit`,
  `workers`, `best_fitness`, `wall_clock_seconds`, `log_file`.
- **HDF5 logs**: `logs/experiments/n{n}_{method}_{warm|cold}_{svd|nosvd}_r{repeat}.h5`

### 7.7 Plotting grid results

```bash
python plot_experiments.py
# or with explicit paths:
python plot_experiments.py \
  --csv results/experiments_dim2.csv \
  --log-dir logs/experiments \
  --out-dir plots/experiments
```

Produces four plot types: best-fitness bar charts by `n`, fitness convergence
curves per `n`, Hamming diversity curves per `n`, and a summary heatmap
(learning method × `n`, split by warm start / SVD noise).

For long-run results:

```bash
python plot_experiments_long.py \
  --csv results/experiments_long/experiments_long.csv \
  --log-dir logs/experiments_long \
  --out-dir plots/experiments_long
```

---

## 8. Running the cache experiment (`experiments_cache.py`)

Tracks permutation **cache hit rate** per generation to measure how much
caching reduces Gurobi calls. Only two configurations are compared:

- Birkhoff + warm start (no SVD)
- PBIL + warm start (no SVD)

### 8.1 Basic usage

```bash
python experiments_cache.py
```

Defaults: `n=50`, 500 generations, population size 50, 1 repeat,
`time_limit=15 s` per evaluation.

### 8.2 Quick wiring check

```bash
python experiments_cache.py --dry-run
```

Runs 3 generations with population 10 to verify wiring.

### 8.3 Custom run

```bash
python experiments_cache.py \
  --n 50 \
  --generations 500 \
  --population-size 50 \
  --selection-size 25 \
  --workers 4 \
  --time-limit 15 \
  --repeats 3 \
  --seed 0 \
  --log-dir logs/cache_experiment \
  --output results/cache_experiment/results.csv
```

### 8.4 Available flags

| Flag | Default | Meaning |
|---|---|---|
| `--n` | 50 | Problem dimension. |
| `--generations` | 500 | Generations per run. |
| `--population-size` | 50 | EDA population size λ. |
| `--selection-size` | 25 | Elite percentage. |
| `--workers` | `cpu_count() − 1` | Parallel CPU workers. |
| `--time-limit` | 15.0 | Per-evaluation Gurobi cap in seconds. |
| `--repeats` | 1 | Independent reruns per configuration. |
| `--max-duration` | `None` | Wall-clock budget per run in seconds. |
| `--seed` | 0 | Base random seed. |
| `--log-dir` | `logs/cache_experiment` | Directory for HDF5 logs. |
| `--output` | `results/cache_experiment/results.csv` | Summary CSV path. |
| `--dry-run` | — | 3 gens, pop=10 — wiring check only. |

### 8.5 Output

- **Summary CSV** (`results/cache_experiment/results.csv`): one row per run with
  `best_fitness`, `wall_clock_seconds`, `cache_hit_rate_mean`, `cache_hit_rate_std`,
  `total_evaluations` (Gurobi calls), and `total_cache_hits`.
- **Per-generation CSV** (`results/cache_experiment/cache_hit_rate_per_generation.csv`):
  one row per generation with `generation`, `learning_method`, `warm_start`,
  `repeat`, and `cache_hit_rate`.
- **HDF5 logs**: `logs/cache_experiment/n{n}_{method}_{warm|cold}_nosvd_r{repeat}.h5`

### 8.6 Plotting cache results

```bash
python plot_experiments_cache.py
# or with explicit paths:
python plot_experiments_cache.py \
  --csv results/cache_experiment/results.csv \
  --per-gen-csv results/cache_experiment/cache_hit_rate_per_generation.csv \
  --out-dir plots/cache_experiment
```

---

## 9. Inspecting HDF5 logs

Convert any HDF5 log to JSON for quick inspection:

```bash
python generation_jsonify.py --file logs/generation_log.h5
```

Each log contains:
- **`metadata/`** — run parameters (`n`, `population_size`, `learning_method`, etc.)
- **`generation_NNNNNN/`** — per-generation arrays: `population`, `fitness`,
  `best_fitness`, `dsm`, and `diversity/` (Hamming distance statistics).
- **`summary/`** (written after the run) — `stopped_early`, `generations_requested`,
  `generations_completed`, `wall_clock_seconds`, `final_best_fitness`.

---

## 10. Running the tests

```bash
pytest tests/
```

Works identically on both platforms once the venv is active.

---

## 11. Troubleshooting

- **`ImportError: DLL load failed` / missing `.so` on import** — you're
  likely using a `venv/` created on the other OS. Delete it and recreate per
  [Section 2](#2-setup-on-macos) or [Section 3](#3-setup-on-windows).
- **Gurobi license errors on `optimize()`** — see [Section 4](#4-gurobi-license).
- **A run seems stuck** — for larger `n`, each permutation evaluation is a
  nonconvex QCQP; without `run.time_limit` set, Gurobi tries to prove exact
  global optimality, which grows very slowly with `n`. Set `time_limit` (per
  evaluation) and/or `max_duration_seconds` (for the whole run) in your config,
  or use `--time-limit` / `--max-duration` flags in the experiment scripts.
- **GPU** — this pipeline does not use one. The bottleneck is Gurobi's
  per-permutation CPU solve; the parallelism that helps is
  `run.num_workers` / `--workers`, not a GPU.
