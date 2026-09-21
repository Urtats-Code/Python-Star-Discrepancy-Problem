"""
plot_experiments_cache.py - Visualisations for the cache-tracking experiment.

Produces four types of plots (each also exported as a .csv):
  1. Cache hit rate evolution per generation (line plot, one curve per config)
  2. Fitness convergence curves per config (best_fitness from .h5 logs)
  3. Diversity evolution per config (Hamming diversity from .h5 logs)
  4. Summary bar chart: best_fitness, wall_clock, cache hit rate mean ± std

All figures are saved to --out-dir (default: plots/experiments_cache/).
Companion CSVs are written alongside each figure under the same directory.

Usage:
    python plot_experiments_cache.py
    python plot_experiments_cache.py \
        --csv results/cache_experiment/results.csv \
        --per-gen-csv results/cache_experiment/cache_hit_rate_per_generation.csv \
        --log-dir logs/cache_experiment \
        --out-dir plots/experiments_cache
"""
import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ── colour / style helpers ──────────────────────────────────────────────────

METHOD_COLOUR = {
    "birkhoff": "#1f77b4",
    "pbil":     "#ff7f0e",
}

METHOD_LINE = {
    "birkhoff": "-",
    "pbil":     "--",
}

METHOD_MARKER = {
    "birkhoff": "o",
    "pbil":     "s",
}


def config_label(row):
    ws = "warm" if row["warm_start"] else "cold"
    return f"{row['learning_method']} {ws}"


# ── H5 loader (mirrors plot_experiments_long.py) ────────────────────────────

def load_h5(path):
    data = {"generations": [], "best_fitness": [], "diversity": []}
    with h5py.File(path, "r") as f:
        gen_keys = sorted(k for k in f.keys() if k.startswith("generation_"))
        for key in gen_keys:
            grp = f[key]
            data["generations"].append(int(key.split("_")[1]))
            data["best_fitness"].append(float(grp["best_fitness"][()]))
            if "diversity" in grp:
                div_item = grp["diversity"]
                if hasattr(div_item, "keys"):
                    div = float(div_item["mean_normalized"][()])
                else:
                    div = float(div_item[()])
            else:
                div = np.nan
            data["diversity"].append(div)
    return {k: np.array(v) for k, v in data.items()}


def _resolve_h5(h5_path):
    if not os.path.isabs(h5_path):
        h5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), h5_path)
    return h5_path


# ── CSV helper ───────────────────────────────────────────────────────────────

def save_csv(df, out_dir, filename):
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"  CSV  -> {path}")


# ── Plot 1: cache hit rate evolution ────────────────────────────────────────

def plot_cache_evolution(per_gen_df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    records = []
    for (method, ws, repeat), grp in per_gen_df.groupby(
            ["learning_method", "warm_start", "repeat"]):
        grp = grp.sort_values("generation")
        label = f"{method} {'warm' if ws else 'cold'} r{repeat}"
        colour = METHOD_COLOUR.get(method, "gray")
        ls     = METHOD_LINE.get(method, "-")
        mk     = METHOD_MARKER.get(method, "o")
        ax.plot(grp["generation"], grp["cache_hit_rate"],
                color=colour, linestyle=ls, marker=mk,
                markersize=3, linewidth=1.2, markevery=25, label=label)
        for _, row in grp.iterrows():
            records.append({
                "generation":       row["generation"],
                "learning_method":  method,
                "warm_start":       ws,
                "repeat":           repeat,
                "cache_hit_rate":   row["cache_hit_rate"],
            })

    ax.set_xlabel("Generation")
    ax.set_ylabel("Cache hit rate")
    ax.set_title("Cache hit rate evolution per generation", fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "cache_hit_rate_evolution.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Fig  -> {fig_path}")

    save_csv(pd.DataFrame(records), out_dir, "cache_hit_rate_evolution.csv")


# ── Plot 2: fitness convergence ──────────────────────────────────────────────

def plot_fitness_convergence(summary_df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    records = []
    for _, row in summary_df.iterrows():
        h5_path = _resolve_h5(row["log_file"])
        if not os.path.exists(h5_path):
            print(f"  [warn] missing H5: {h5_path}")
            continue
        h5      = load_h5(h5_path)
        label   = config_label(row)
        colour  = METHOD_COLOUR.get(row["learning_method"], "gray")
        ls      = METHOD_LINE.get(row["learning_method"], "-")
        mk      = METHOD_MARKER.get(row["learning_method"], "o")
        ax.plot(h5["generations"], h5["best_fitness"],
                color=colour, linestyle=ls, marker=mk,
                markersize=3, linewidth=1.2, markevery=25, label=label)
        for gen, fit in zip(h5["generations"], h5["best_fitness"]):
            records.append({
                "generation":      gen,
                "learning_method": row["learning_method"],
                "warm_start":      row["warm_start"],
                "repeat":          row["repeat"],
                "best_fitness":    fit,
            })

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (star discrepancy)")
    ax.set_title("Fitness convergence (lower is better)", fontsize=13)
    ax.grid(linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fitness_convergence.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Fig  -> {fig_path}")

    save_csv(pd.DataFrame(records), out_dir, "fitness_convergence.csv")


# ── Plot 3: diversity evolution ──────────────────────────────────────────────

def plot_diversity_evolution(summary_df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    records = []
    any_data = False
    for _, row in summary_df.iterrows():
        h5_path = _resolve_h5(row["log_file"])
        if not os.path.exists(h5_path):
            continue
        h5 = load_h5(h5_path)
        if np.all(np.isnan(h5["diversity"])):
            continue
        any_data = True
        label  = config_label(row)
        colour = METHOD_COLOUR.get(row["learning_method"], "gray")
        ls     = METHOD_LINE.get(row["learning_method"], "-")
        mk     = METHOD_MARKER.get(row["learning_method"], "o")
        ax.plot(h5["generations"], h5["diversity"],
                color=colour, linestyle=ls, marker=mk,
                markersize=3, linewidth=1.2, markevery=25, label=label)
        for gen, div in zip(h5["generations"], h5["diversity"]):
            records.append({
                "generation":      gen,
                "learning_method": row["learning_method"],
                "warm_start":      row["warm_start"],
                "repeat":          row["repeat"],
                "diversity":       div,
            })

    if not any_data:
        print("  [warn] no diversity data found in H5 logs — skipping diversity plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Generation")
    ax.set_ylabel("Population diversity (Hamming)")
    ax.set_title("Diversity evolution per generation", fontsize=13)
    ax.grid(linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "diversity_evolution.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Fig  -> {fig_path}")

    save_csv(pd.DataFrame(records), out_dir, "diversity_evolution.csv")


# ── Plot 4: summary bar chart ────────────────────────────────────────────────

def plot_summary(summary_df, out_dir):
    """Three-panel bar chart: best_fitness | wall_clock | cache hit rate mean ± std."""
    df = summary_df.copy()
    df["label"] = df.apply(config_label, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colours = [METHOD_COLOUR.get(m, "gray") for m in df["learning_method"]]
    x = range(len(df))

    # Panel A – best fitness
    ax = axes[0]
    bars = ax.bar(x, df["best_fitness"], color=colours, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, df["best_fitness"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                f"{val:.5f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Best fitness (star discrepancy)")
    ax.set_title("Best fitness (lower is better)")
    ax.grid(axis="y", linewidth=0.4, alpha=0.6)

    # Panel B – wall-clock time
    ax = axes[1]
    times_min = df["wall_clock_seconds"] / 60
    bars = ax.bar(x, times_min, color=colours, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, times_min):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Wall-clock time (min)")
    ax.set_title("Wall-clock time")
    ax.grid(axis="y", linewidth=0.4, alpha=0.6)

    # Panel C – cache hit rate mean ± std
    ax = axes[2]
    bars = ax.bar(x, df["cache_hit_rate_mean"], yerr=df["cache_hit_rate_std"],
                  color=colours, edgecolor="black", linewidth=0.5,
                  capsize=5, error_kw={"elinewidth": 1.2})
    for bar, val in zip(bars, df["cache_hit_rate_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Cache hit rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Cache hit rate (mean ± std)")
    ax.grid(axis="y", linewidth=0.4, alpha=0.6)

    handles = [Patch(facecolor=c, label=m) for m, c in METHOD_COLOUR.items()
               if m in df["learning_method"].values]
    fig.legend(handles=handles, title="Learning method", loc="upper right", fontsize=8)
    fig.suptitle("Experiment summary", fontsize=14, y=1.01)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "summary_bar.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Fig  -> {fig_path}")

    # CSV: the summary table itself (keep all columns, add label)
    save_csv(df, out_dir, "summary.csv")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot cache-tracking experiment results.",
    )
    parser.add_argument(
        "--csv",
        default="results/cache_experiment/results.csv",
        help="Summary CSV written by experiments_cache.py",
    )
    parser.add_argument(
        "--per-gen-csv",
        default="results/cache_experiment/cache_hit_rate_per_generation.csv",
        help="Per-generation cache hit rate CSV written by experiments_cache.py",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/cache_experiment",
        help="Directory containing the .h5 generation logs",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/experiments_cache",
        help="Output directory for figures and companion CSVs",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load summary CSV
    summary_df = pd.read_csv(args.csv)
    summary_df = summary_df.drop_duplicates(
        subset=["n", "learning_method", "warm_start", "svd_noise", "repeat"],
        keep="last",
    )
    print(f"Loaded {len(summary_df)} run(s) from {args.csv}")
    print(summary_df[["n", "learning_method", "warm_start",
                       "best_fitness", "wall_clock_seconds",
                       "cache_hit_rate_mean", "cache_hit_rate_std",
                       "total_evaluations", "total_cache_hits"]].to_string(index=False))
    print()

    # Load per-generation cache CSV
    per_gen_df = pd.read_csv(args.per_gen_csv)
    print(f"Loaded {len(per_gen_df)} per-generation rows from {args.per_gen_csv}\n")

    print("=== Plot 1: cache hit rate evolution ===")
    plot_cache_evolution(per_gen_df, args.out_dir)

    print("\n=== Plot 2: fitness convergence ===")
    plot_fitness_convergence(summary_df, args.out_dir)

    print("\n=== Plot 3: diversity evolution ===")
    plot_diversity_evolution(summary_df, args.out_dir)

    print("\n=== Plot 4: summary bar chart ===")
    plot_summary(summary_df, args.out_dir)

    print(f"\nAll outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
