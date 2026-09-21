"""
experiments_cache.py - Focused comparison of Birkhoff warm-start vs PBIL warm-start
at n=50 for 500 generations, with cache hit rate tracking.

Only two configurations are run:
  - birkhoff  warm_start=True  svd_noise=False
  - pbil      warm_start=True  svd_noise=False

Cache hit rate is measured per generation (hits / population_size) and reported
as mean ± std over all generations, plus a per-generation CSV is written so the
evolution of cache efficiency can be plotted.

Usage:
    python experiments_cache.py

    # Override defaults:
    python experiments_cache.py --population-size 50 --generations 500 \
        --workers 4 --output results/cache_experiment/results.csv
"""
import argparse
import concurrent.futures
import csv
import os
import time
from dataclasses import dataclass, asdict, fields

import numpy as np

from DSM_EDA_Pipeline import DSM_EDA_Pipeline, LearningStrategy, init_worker, worker_evaluate

DEFAULT_N = 50
DEFAULT_GENERATIONS = 500
DEFAULT_POPULATION_SIZE = 50
DEFAULT_SELECTION_SIZE_PCT = 25
DEFAULT_TIME_LIMIT = 15.0  # seconds per Gurobi evaluation at n=50


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

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
    cache_hit_rate_mean: float   # mean across all generations
    cache_hit_rate_std: float    # std across all generations
    total_evaluations: int       # total distinct permutations evaluated by Gurobi
    total_cache_hits: int        # total cache hits across all generations


# ---------------------------------------------------------------------------
# Instrumented pipeline subclass that records cache hit rate per generation
# ---------------------------------------------------------------------------

class CacheTrackingPipeline(DSM_EDA_Pipeline):
    """Extends DSM_EDA_Pipeline to record cache hits/misses per generation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_hit_rates: list[float] = []
        self.total_cache_hits: int = 0
        self.total_evaluations: int = 0  # = total gurobi calls (misses)

    def evaluate_population_parallel(self, population, executor):
        uncached = [p for p in population if not self.cache.contains(p)]
        n_hits = len(population) - len(uncached)

        if uncached:
            new_scores = list(executor.map(worker_evaluate, uncached))
            for perm, score in zip(uncached, new_scores):
                self.cache.set(perm, score)

        fitness_scores = [self.cache.get(p) for p in population]
        sorted_indices = np.argsort(fitness_scores)

        hit_rate = n_hits / len(population)
        self.cache_hit_rates.append(hit_rate)
        self.total_cache_hits += n_hits
        self.total_evaluations += len(uncached)

        return [population[i] for i in sorted_indices], [fitness_scores[i] for i in sorted_indices]


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def run_single(n, learning_method, warm_start, *, population_size, selection_size_pct,
               generations, time_limit, workers, epsilon, log_dir, repeat=0, seed=None,
               max_duration_seconds=None):
    if seed is not None:
        np.random.seed(seed)

    tag = (f"n{n}_{learning_method.value}"
           f"_{'warm' if warm_start else 'cold'}"
           f"_nosvd_r{repeat}")
    log_file = os.path.join(log_dir, f"{tag}.h5")

    pipeline = CacheTrackingPipeline(
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
    pipeline.svd_enabled = False

    if warm_start:
        pipeline.warm_start(alpha0=0.3, shift=1)

    start = time.perf_counter()
    _, best_fitness = pipeline.run(
        epsilon=epsilon,
        generations=generations,
        num_workers=workers,
        max_duration_seconds=max_duration_seconds,
    )
    elapsed = time.perf_counter() - start

    hit_rates = pipeline.cache_hit_rates
    mean_hit_rate = float(np.mean(hit_rates)) if hit_rates else 0.0
    std_hit_rate = float(np.std(hit_rates))  if hit_rates else 0.0

    return RunResult(
        n=n,
        learning_method=learning_method.value,
        warm_start=warm_start,
        svd_noise=False,
        repeat=repeat,
        population_size=population_size,
        selection_size_pct=selection_size_pct,
        generations=generations,
        time_limit=time_limit,
        workers=workers,
        best_fitness=best_fitness,
        wall_clock_seconds=elapsed,
        log_file=pipeline.gen_log_file,
        cache_hit_rate_mean=mean_hit_rate,
        cache_hit_rate_std=std_hit_rate,
        total_evaluations=pipeline.total_evaluations,
        total_cache_hits=pipeline.total_cache_hits,
    ), hit_rates


def append_result(csv_path, result: RunResult, write_header: bool):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(RunResult)])
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def write_per_generation_cache(csv_path, learning_method: str, warm_start: bool,
                                repeat: int, hit_rates: list[float], write_header: bool):
    """Write per-generation cache hit rate to a separate CSV for plotting."""
    with open(csv_path, "a", newline="") as f:
        fieldnames = ["generation", "learning_method", "warm_start", "repeat", "cache_hit_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for gen, rate in enumerate(hit_rates):
            writer.writerow({
                "generation": gen,
                "learning_method": learning_method,
                "warm_start": warm_start,
                "repeat": repeat,
                "cache_hit_rate": rate,
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIGURATIONS = [
    (LearningStrategy.BIRKHOFF, True),
    (LearningStrategy.PBIL,     True),
]


def main():
    parser = argparse.ArgumentParser(
        description="Compare Birkhoff warm vs PBIL warm at n=50 for 500 generations, "
                    "with cache hit rate tracking.",
    )
    parser.add_argument("--n",               type=int,   default=DEFAULT_N)
    parser.add_argument("--generations",     type=int,   default=DEFAULT_GENERATIONS)
    parser.add_argument("--population-size", type=int,   default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--selection-size",  type=int,   default=DEFAULT_SELECTION_SIZE_PCT,
                        help="Percent of population kept as elite.")
    parser.add_argument("--workers",         type=int,   default=None,
                        help="Defaults to os.cpu_count() - 1.")
    parser.add_argument("--epsilon",         type=float, default=1e-4)
    parser.add_argument("--time-limit",      type=float, default=DEFAULT_TIME_LIMIT,
                        help="Per-evaluation Gurobi time limit (seconds).")
    parser.add_argument("--repeats",         type=int,   default=1)
    parser.add_argument("--max-duration",    type=float, default=None,
                        help="Wall-clock budget per run (seconds).")
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output",          default="results/cache_experiment/results.csv")
    parser.add_argument("--log-dir",         default="logs/cache_experiment")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Tiny smoke test (3 generations, pop=10) to validate wiring.")
    args = parser.parse_args()

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(args.log_dir, exist_ok=True)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    per_gen_csv = os.path.join(out_dir or ".", "cache_hit_rate_per_generation.csv")

    if args.dry_run:
        generations = 3
        population_size = 10
        repeats = 1
    else:
        generations = args.generations
        population_size = args.population_size
        repeats = args.repeats

    total_runs = len(CONFIGURATIONS) * repeats
    run_index = 0
    write_header = not os.path.exists(args.output)
    per_gen_header = not os.path.exists(per_gen_csv)

    print(f"=== Cache Experiment: n={args.n}, generations={generations}, "
          f"pop={population_size}, workers={workers} ===")
    print(f"Configs: birkhoff warm  |  pbil warm  (no SVD noise)\n")

    for learning_method, warm_start in CONFIGURATIONS:
        for repeat in range(repeats):
            run_index += 1
            print(f"\n--- Run {run_index}/{total_runs}: "
                  f"method={learning_method.value} warm_start={warm_start} repeat={repeat} ---")

            result, hit_rates = run_single(
                n=args.n,
                learning_method=learning_method,
                warm_start=warm_start,
                population_size=population_size,
                selection_size_pct=args.selection_size,
                generations=generations,
                time_limit=args.time_limit,
                workers=workers,
                epsilon=args.epsilon,
                log_dir=args.log_dir,
                repeat=repeat,
                seed=args.seed + run_index,
                max_duration_seconds=args.max_duration,
            )

            append_result(args.output, result, write_header)
            write_header = False

            write_per_generation_cache(
                per_gen_csv, learning_method.value, warm_start, repeat,
                hit_rates, per_gen_header,
            )
            per_gen_header = False

            print(f"    best_fitness        = {result.best_fitness:.6f}")
            print(f"    wall_clock          = {result.wall_clock_seconds:.1f}s")
            print(f"    cache hit rate      = {result.cache_hit_rate_mean:.3f} "
                  f"± {result.cache_hit_rate_std:.3f}")
            print(f"    total gurobi calls  = {result.total_evaluations}")
            print(f"    total cache hits    = {result.total_cache_hits}")

    print(f"\nAll runs complete.")
    print(f"  Summary CSV      : {args.output}")
    print(f"  Per-generation   : {per_gen_csv}")
    print(f"  Generation logs  : {args.log_dir}/")


if __name__ == "__main__":
    main()
