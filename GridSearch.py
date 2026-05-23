import argparse
import csv
import itertools
import multiprocessing
import concurrent.futures
import os

import yaml

from DSM_EDA_Pipeline import DSM_EDA_Pipeline, LearningStrategy


def _ensure_list(value):
    return value if isinstance(value, list) else [value]


def load_grid(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["pipeline"]
    r_cfg = cfg["run"]
    pbil_cfg = cfg.get("pbil", {})
    birkhoff_cfg = cfg.get("birkhoff", {})
    log_cfg = cfg.get("logging", {})

    axes = {
        "n":               _ensure_list(p_cfg["n"]),
        "population_size": _ensure_list(p_cfg["population_size"]),
        "selection_size":  _ensure_list(p_cfg["selection_size"]),
        "learning_method": _ensure_list(p_cfg["learning_method"]),
        "generations":     _ensure_list(r_cfg["generations"]),
        "epsilon":         _ensure_list(r_cfg["epsilon"]),
        "pbil_learning_rate": _ensure_list(pbil_cfg.get("learning_rate", 0.1)),
        "pbil_mutation_rate": _ensure_list(pbil_cfg.get("mutation_rate", 0.01)),
        "birkhoff_alpha":     _ensure_list(birkhoff_cfg.get("alpha", 0.1)),
    }

    base_log = log_cfg.get("gen_log_file", "generation_log.h5")
    base, ext = os.path.splitext(base_log)

    keys = list(axes.keys())
    combinations = []
    for i, values in enumerate(itertools.product(*axes.values())):
        combo = dict(zip(keys, values))
        combo["gen_log_file"] = f"{base}_run{i:04d}{ext}"
        combinations.append(combo)
    return combinations


def run_combination(params: dict) -> dict:
    pipeline = DSM_EDA_Pipeline(
        n=params["n"],
        population_size=params["population_size"],
        selection_size=params["selection_size"],
        learning_method=LearningStrategy(params["learning_method"]),
        gen_log_file=params["gen_log_file"],
    )

    if LearningStrategy(params["learning_method"]) == LearningStrategy.PBIL:
        pipeline.pbil_learning_rate = params["pbil_learning_rate"]
        pipeline.pbil_mutation_rate = params["pbil_mutation_rate"]
    if LearningStrategy(params["learning_method"]) == LearningStrategy.BIRKHOFF:
        pipeline.birkhoff_alpha = params["birkhoff_alpha"]

    best_solution, best_fitness = pipeline.run(
        epsilon=params["epsilon"],
        generations=params["generations"],
        num_workers=1,
    )

    return {
        **{k: v for k, v in params.items() if k != "gen_log_file"},
        "best_fitness": best_fitness,
        "best_solution": best_solution.tolist(),
    }


def main(config_path: str):
    combinations = load_grid(config_path)
    total = len(combinations)
    print(f"Grid search: {total} combinations")

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_combination, p): i for i, p in enumerate(combinations)}
        for future in concurrent.futures.as_completed(futures):
            run_index = futures[future]
            try:
                result = future.result()
                result["run_index"] = run_index
                results.append(result)
                print(f"[{len(results)}/{total}] run {run_index} done — best_fitness={result['best_fitness']:.8f}")
            except Exception as e:
                print(f"[run {run_index}] FAILED: {e}")

    if not results:
        return

    results.sort(key=lambda r: r["run_index"])

    output_csv = "grid_search_results.csv"
    fieldnames = [k for k in results[0].keys()]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    best = min(results, key=lambda r: r["best_fitness"])
    print(f"\nGrid search complete. Results saved to {output_csv}")
    print(f"Best run: index={best['run_index']} fitness={best['best_fitness']:.8f} params={best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search over DSM EDA Pipeline hyperparameters.")
    parser.add_argument("--config", default="configs/config-grid-search.yaml", help="Path to grid search config YAML")
    args = parser.parse_args()
    main(args.config)
