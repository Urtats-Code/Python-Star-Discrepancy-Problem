"""
experiments.py - grid-sweep driver for the DSM-EDA pipeline in dimension 2,
covering every configuration axis described in proposed_algorithm.tex that
is currently implemented in DSM_EDA_Pipeline.py:

  - initialization:  cold start              vs. warm start (Fibonacci bias, eq:warm_start)
  - learning rule:    Birkhoff (eq:birkhoff_update)  vs. PBIL (eq:pbil_learning)
  - exploration:      no SVD perturbation     vs. SVD noise injection (eq:svd_noise)

That is a 2 x 2 x 2 = 8-configuration grid, run for each requested n. With
the default n values (20, 50, 100, 180) this is 32 runs.

Note on the other three SVD strategies (zeroing out, evenly reduce, randomly
reduce): only noise injection is implemented in svd/SVD.py today, so it is
the only exploration variant this script can switch on or off. Adding the
other three would extend the grid to 5 exploration settings (none + 4
strategies) once they exist.

Compute note - why this script does not use the GPU
----------------------------------------------------
The per-run cost is dominated by PersistentSDPProblem.evaluate(): a
nonconvex QCQP solved by Gurobi (NonConvex=2, branch-and-bound over a
spatial relaxation) once per distinct permutation. This is CPU-bound work;
Gurobi has no GPU solve path for QCQP in this installation (gurobipy
13.0.2), so a GPU on this machine would not speed up the actual bottleneck.
The parallelism that matters is CPU workers evaluating different
permutations of the same generation concurrently, which
DSM_EDA_Pipeline.run() already does via ProcessPoolExecutor. This script
therefore sizes --workers off os.cpu_count() rather than attempting any GPU
offload. (A GPU would only help if the Step-2 coordinate solve were
reformulated as a differentiable/GPU-based optimization instead of a Gurobi
MIQCP - that is a separate, much larger change to PersistentSDPSolver.py.)

Runtime warning
---------------
Evaluating a single permutation at n=100 or n=180 is far more expensive
than at n=20 (see the per-n time limits below). Without a Gurobi time limit,
PersistentSDPProblem tries to prove exact global optimality on every
evaluation, which the source papers themselves needed minutes to hours for
at much smaller n. --time-limit (or the DEFAULT_TIME_LIMITS table) caps this
per evaluation; tune it for your machine with --dry-run first.

Usage
-----
    # quick smoke test before committing to the full grid:
    python experiments.py --dry-run

    # full sweep with defaults (2x2x2 configs x n in {20,50,100,180}):
    python experiments.py --output results/experiments_dim2.csv

    # custom budget:
    python experiments.py --n-values 20,50 --generations 40 \\
        --population-size 50 --selection-size 25 --workers 8 \\
        --time-limit 20 --repeats 3
"""
import argparse
import csv
import itertools
import os
import time
from dataclasses import dataclass, asdict, fields

import numpy as np

from DSM_EDA_Pipeline import DSM_EDA_Pipeline, LearningStrategy

# Conservative starting points for the per-evaluation Gurobi time limit (seconds),
# increasing with n since the Step-2 QCQP grows harder to solve/prove optimal.
# These are engineering defaults chosen to keep the grid tractable, not values
# taken from the thesis or its source papers - tune them with --dry-run and
# --time-limit for your machine.
DEFAULT_TIME_LIMITS = {20: 5.0, 50: 15.0, 100: 30.0, 180: 60.0}
FALLBACK_TIME_LIMIT = 60.0


@dataclass
class RunResult:
    n: int
    learning_method: str
    warm_start: bool
    svd_noise: bool
    repeat: int
    population_size: int
    selection_size_pct: int
    generations: int
    time_limit: float
    workers: int
    best_fitness: float
    wall_clock_seconds: float
    log_file: str


def build_configurations():
    """The 2x2x2 grid of (learning_method, warm_start, svd_noise) currently implemented."""
    return list(itertools.product(
        [LearningStrategy.BIRKHOFF, LearningStrategy.PBIL],
        [False, True],   # warm start
        [False, True],   # svd noise injection
    ))


def run_single(n, learning_method, warm_start, svd_noise, *, population_size,
               selection_size_pct, generations, time_limit, workers, epsilon,
               log_dir, repeat=0, seed=None, max_duration_seconds=None):
    if seed is not None:
        np.random.seed(seed)

    tag = (f"n{n}_{learning_method.value}"
           f"_{'warm' if warm_start else 'cold'}"
           f"_{'svd' if svd_noise else 'nosvd'}_r{repeat}")
    log_file = os.path.join(log_dir, f"{tag}.h5")

    pipeline = DSM_EDA_Pipeline(
        n=n,
        population_size=population_size,
        selection_size=selection_size_pct,
        learning_method=learning_method,
        gen_log_file=log_file,
    )

    if learning_method == LearningStrategy.BIRKHOFF:
        pipeline.birkhoff_alpha = 0.1
    else:
        pipeline.pbil_alpha = 0.1

    pipeline.time_limit = time_limit
    pipeline.svd_enabled = svd_noise
    pipeline.svd_theta = 0.3
    pipeline.svd_every = 1

    if warm_start:
        pipeline.warm_start(alpha0=0.3, shift=1)

    start = time.perf_counter()
    _, best_fitness = pipeline.run(
        epsilon=epsilon, generations=generations, num_workers=workers,
        max_duration_seconds=max_duration_seconds,
    )
    elapsed = time.perf_counter() - start

    return RunResult(
        n=n,
        learning_method=learning_method.value,
        warm_start=warm_start,
        svd_noise=svd_noise,
        repeat=repeat,
        population_size=population_size,
        selection_size_pct=selection_size_pct,
        generations=generations,
        time_limit=time_limit if time_limit is not None else -1.0,
        workers=workers,
        best_fitness=best_fitness,
        wall_clock_seconds=elapsed,
        log_file=pipeline.gen_log_file,
    )


def append_result(csv_path, result: RunResult, write_header: bool):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(RunResult)])
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def main():
    parser = argparse.ArgumentParser(
        description="Grid sweep over all implemented DSM-EDA configurations (dimension 2).",
    )
    parser.add_argument("--n-values", default="20,50,100,180", help="Comma-separated list of n values.")
    parser.add_argument("--long", action="store_true",
                         help="Long-run preset: n in {20,50}, 500 generations, "
                              "output/logs go to results/experiments_long and logs/experiments_long.")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--selection-size", type=int, default=25, help="Percent of population kept as elite.")
    parser.add_argument("--workers", type=int, default=None, help="Defaults to os.cpu_count() - 1.")
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--time-limit", type=float, default=None,
                         help="Override the per-n Gurobi time limit (seconds) for every n. "
                              "Defaults to the DEFAULT_TIME_LIMITS table.")
    parser.add_argument("--repeats", type=int, default=1, help="Independent repeats per configuration.")
    parser.add_argument("--max-duration", type=float, default=None,
                         help="Wall-clock budget per run (seconds). When reached, that run's "
                              "generation loop stops cleanly with the current best DSM/solution "
                              "kept, instead of running the full --generations count.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/experiments_dim2.csv")
    parser.add_argument("--log-dir", default="logs/experiments")
    parser.add_argument("--dry-run", action="store_true",
                         help="Tiny smoke test (n=20, 1 config, 3 generations) to validate "
                              "wiring and estimate per-evaluation solve time before the full grid.")
    args = parser.parse_args()

    # --long overrides output/log paths and key hyperparameters
    if args.long:
        if args.output == "results/experiments_dim2.csv":
            args.output = "results/experiments_long/experiments_long.csv"
        if args.log_dir == "logs/experiments":
            args.log_dir = "logs/experiments_long"
        if args.n_values == "20,50,100,180":
            args.n_values = "20,25"
        if args.generations == 60:
            args.generations = 500

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(args.log_dir, exist_ok=True)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.dry_run:
        n_values = [20]
        configs = build_configurations()[:1]
        generations = 3
        population_size = 10
        repeats = 1
    else:
        n_values = [int(x) for x in args.n_values.split(",")]
        configs = build_configurations()
        generations = args.generations
        population_size = args.population_size
        repeats = args.repeats

    write_header = not os.path.exists(args.output)
    run_index = 0
    total_runs = len(n_values) * len(configs) * repeats
    for n in n_values:
        time_limit = args.time_limit if args.time_limit is not None else DEFAULT_TIME_LIMITS.get(n, FALLBACK_TIME_LIMIT)
        for learning_method, warm_start, svd_noise in configs:
            for repeat in range(repeats):
                run_index += 1
                print(f"\n=== Run {run_index}/{total_runs}: n={n} method={learning_method.value} "
                      f"warm_start={warm_start} svd_noise={svd_noise} repeat={repeat} "
                      f"(time_limit={time_limit}s, workers={workers}) ===")
                result = run_single(
                    n, learning_method, warm_start, svd_noise,
                    population_size=population_size,
                    selection_size_pct=args.selection_size,
                    generations=generations,
                    time_limit=time_limit,
                    workers=workers,
                    epsilon=args.epsilon,
                    log_dir=args.log_dir,
                    repeat=repeat,
                    seed=args.seed + run_index,
                    max_duration_seconds=args.max_duration,
                )
                append_result(args.output, result, write_header)
                write_header = False
                print(f"--- Done: best_fitness={result.best_fitness:.6f} "
                      f"wall_clock={result.wall_clock_seconds:.1f}s ---")

    print(f"\nAll runs complete. Results written to {args.output}")


if __name__ == "__main__":
    main()
