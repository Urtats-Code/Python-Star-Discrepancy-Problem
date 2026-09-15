# Python Star Discrepancy Problem — DSM-EDA Pipeline

A Doubly Stochastic Matrix (DSM) Estimation of Distribution Algorithm (EDA)
for the permutation stage of the two-step Star Discrepancy Problem (SDP)
reformulation: an outer EDA searches over permutations, and each sampled
permutation is scored by fixing it in a Gurobi continuous NLP that optimizes
the coordinates (`PersistentSDPSolver.py`).

This README covers environment setup on both **Windows** and **macOS**, the
configuration options, and how to run the pipeline, the smoke test, and the
experiment grid sweep.

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

All runs are driven by a YAML config under `configs/`. Key sections
(see `configs/config.yaml` for the full commented reference):

| Section | Key | Meaning |
|---|---|---|
| `pipeline` | `n` | Number of points (dimension-2 SDP instance size). |
| | `population_size` | EDA population size $\lambda$. |
| | `selection_size` | Elite size as a **percentage** of the population (e.g. `25` = top 25%). |
| | `learning_method` | `"birkhoff"` or `"pbil"`. |
| `run` | `generations` | Max number of generations to run. |
| | `epsilon` | Minimum coordinate spacing in the Step-2 NLP. |
| | `num_workers` | CPU worker processes for parallel evaluation; `null` = `cpu_count() - 1`. |
| | `time_limit` | Per-permutation Gurobi solve cap, in seconds. `null` = solve to proven optimality (fine for small `n`, can be very slow for large `n`). |
| | `max_duration_seconds` | Wall-clock budget for the **whole training run**. When reached, the generation loop stops cleanly (no kill) and keeps the current best DSM/solution and every completed generation already logged. `null` = run all `generations`. |
| `birkhoff` | `alpha` | Exploration weight in the Birkhoff update rule. |
| `pbil` | `alpha` | Smoothing factor toward the best-so-far solution. |
| `warm_start` | `enabled` | Bias $M^{(0)}$ toward a Fibonacci-pattern permutation instead of a uniform start. |
| | `alpha0`, `shift` | Mixing weight and Fibonacci-sequence shift. |
| `svd` | `enabled` | Apply SVD noise-injection perturbation each generation to counter premature convergence. |
| | `theta`, `apply_every` | Noise intensity and how often (in generations) to apply it. |
| `logging` | `gen_log_file` | Path to the per-generation HDF5 log. |
| `visualization` | `enabled`, ... | Whether/what plots to generate after the run. |

Both commands below (macOS and Windows) are otherwise identical once the
venv is active — only the activation step above differs.

---

## 6. Running the pipeline

```bash
python DSM_EDA_Pipeline.py --config configs/config.yaml
```

This prints per-generation progress (`Gen 000 | Best Fitness: ... |
Diversity: ...`), then a final best permutation/fitness summary. Results are
logged to the HDF5 file named in `logging.gen_log_file`, and (if
`visualization.enabled: true`) plots are written to `visualization.output_dir`.

### Smoke test

`configs/config-smoke-test.yaml` mirrors a full-size config (`n=8`,
`generations=500`, `population_size=200`) but caps the run at
`max_duration_seconds: 90`, so it validates the whole pipeline quickly:

```bash
python DSM_EDA_Pipeline.py --config configs/config-smoke-test.yaml
```

Check whether it completed all generations or stopped on the time budget:

```bash
python generation_jsonify.py --file logs/smoke_test_log.h5
# then inspect the "summary" section of the printed JSON, in particular
# "stopped_early" and "generations_completed"
```

---

## 7. Running the experiment grid (`experiments.py`)

Sweeps every implemented configuration axis (Birkhoff/PBIL × cold/warm start
× SVD-noise on/off) across a list of `n` values in dimension 2:

```bash
# quick wiring check first (n=20, 1 config, 3 generations):
python experiments.py --dry-run

# full grid (defaults to n = 20, 50, 100, 180 — 8 configs each):
python experiments.py --output results/experiments_dim2.csv
```

Useful flags: `--n-values 20,50`, `--generations`, `--population-size`,
`--selection-size`, `--workers`, `--time-limit` (per-evaluation Gurobi cap),
`--max-duration` (wall-clock cap per run), `--repeats` (independent reruns
per configuration). Results are appended to the output CSV as each run
finishes, so an interrupted sweep keeps whatever it already completed.

Per-run HDF5 logs land under `logs/experiments/` by default
(`--log-dir` to change it).

---

## 8. Inspecting results

- `python generation_jsonify.py --file <path>.h5` converts a log to JSON for
  quick inspection.
- `Visualize` (used internally by `pipeline.visualize(...)`) can plot best-
  fitness curves, the final DSM, and combined/animated views from a log
  file; see `Visualize.py` for the individual plotting methods.

---

## 9. Running the tests

```bash
pytest tests/
```

Works identically on both platforms once the venv is active.

---

## 10. Troubleshooting

- **`ImportError: DLL load failed` / missing `.so` on import** — you're
  likely using a `venv/` created on the other OS. Delete it and recreate per
  [Section 2](#2-setup-on-macos) or [Section 3](#3-setup-on-windows).
- **Gurobi license errors on `optimize()`** — see [Section 4](#4-gurobi-license).
- **A run seems stuck** — for larger `n`, each permutation evaluation is a
  nonconvex QCQP; without `run.time_limit` set, Gurobi tries to prove exact
  global optimality, which grows very slowly with `n`. Set `time_limit` (per
  evaluation) and/or `max_duration_seconds` (for the whole run) in your config.
- **GPU** — this pipeline does not use one. The bottleneck is Gurobi's
  per-permutation CPU solve; the parallelism that helps is
  `run.num_workers` / `experiments.py --workers`, not a GPU.
