"""
plot_experiments_long.py - Visualizations for the DSM-EDA long-run grid-sweep results.

Produces five types of plots:
  1. Best fitness by n (bar chart comparing all configs)
  2. Fitness convergence curves per n (line plots from .h5 logs)
  3. Diversity evolution per n (Hamming diversity over generations)
  4. Summary heatmap: best_fitness by (learning_method x n), split by warm_start / svd_noise
  5. Wall-clock time per config

Usage:
    python plot_experiments_long.py
    python plot_experiments_long.py --csv results/experiments_long/experiments_long.csv \
        --log-dir logs/experiments_long --out-dir plots/experiments_long
"""
import argparse
import math
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ── colour / style helpers ──────────────────────────────────────────────────

CONFIG_STYLE = {
    # (warm_start, svd_noise) -> (linestyle, marker)
    (False, False): ("-",  "o"),
    (False, True):  ("--", "s"),
    (True,  False): ("-.", "^"),
    (True,  True):  (":",  "D"),
}

METHOD_COLOUR = {
    "birkhoff": "#1f77b4",
    "pbil":     "#ff7f0e",
}


def config_label(row):
    ws  = "warm"  if row["warm_start"] else "cold"
    svd = "+svd"  if row["svd_noise"]  else ""
    return f"{row['learning_method']} {ws}{svd}"


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
                if hasattr(div_item, "keys"):          # it's a group
                    div = float(div_item["mean_normalized"][()])
                else:
                    div = float(div_item[()])
            else:
                div = np.nan
            data["diversity"].append(div)
    return {k: np.array(v) for k, v in data.items()}


def _make_grid(n_plots):
    """Return (nrows, ncols) for a roughly-square subplot grid."""
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)
    return nrows, ncols


# ── Plot 1: bar chart of best_fitness per config for each n ─────────────────

def plot_bar_by_n(df, out_dir):
    n_values = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, len(n_values), figsize=(5 * len(n_values), 5), sharey=False)
    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        sub = df[df["n"] == n].copy()
        sub["label"] = sub.apply(config_label, axis=1)
        colours = [METHOD_COLOUR[m] for m in sub["learning_method"]]
        bars = ax.bar(range(len(sub)), sub["best_fitness"],
                      color=colours, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"n = {n}", fontsize=11)
        ax.set_ylabel("Best fitness (star discrepancy)" if n == n_values[0] else "")
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)
        for bar, val in zip(bars, sub["best_fitness"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=6)

    handles = [Patch(facecolor=c, label=m) for m, c in METHOD_COLOUR.items()]
    fig.legend(handles=handles, title="Learning method", loc="upper right", fontsize=8)
    fig.suptitle("Best fitness per configuration (lower is better)", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "bar_best_fitness_by_n.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 2: fitness convergence curves per n ────────────────────────────────

def plot_convergence(df, out_dir):
    n_values = sorted(df["n"].unique())
    nrows, ncols = _make_grid(len(n_values))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, n in zip(axes, n_values):
        sub = df[df["n"] == n]
        for _, row in sub.iterrows():
            h5_path = row["log_file"]
            if not os.path.isabs(h5_path):
                h5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), h5_path)
            if not os.path.exists(h5_path):
                continue
            h5 = load_h5(h5_path)
            ls, mk = CONFIG_STYLE[(row["warm_start"], row["svd_noise"])]
            colour = METHOD_COLOUR[row["learning_method"]]
            ax.plot(h5["generations"], h5["best_fitness"],
                    linestyle=ls, marker=mk, markersize=3,
                    color=colour, label=config_label(row), linewidth=1.2, markevery=25)
        ax.set_title(f"n = {n}", fontsize=11)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best fitness")
        ax.grid(linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=6, ncol=2)

    for ax in axes[len(n_values):]:
        ax.set_visible(False)

    fig.suptitle("Fitness convergence per configuration (lower is better)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "convergence_curves.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 3: diversity evolution per n ───────────────────────────────────────

def plot_diversity(df, out_dir):
    n_values = sorted(df["n"].unique())
    nrows, ncols = _make_grid(len(n_values))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, n in zip(axes, n_values):
        sub = df[df["n"] == n]
        for _, row in sub.iterrows():
            h5_path = row["log_file"]
            if not os.path.isabs(h5_path):
                h5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), h5_path)
            if not os.path.exists(h5_path):
                continue
            h5 = load_h5(h5_path)
            if np.all(np.isnan(h5["diversity"])):
                continue
            ls, mk = CONFIG_STYLE[(row["warm_start"], row["svd_noise"])]
            colour = METHOD_COLOUR[row["learning_method"]]
            ax.plot(h5["generations"], h5["diversity"],
                    linestyle=ls, marker=mk, markersize=3,
                    color=colour, label=config_label(row), linewidth=1.2, markevery=25)
        ax.set_title(f"n = {n}", fontsize=11)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Population diversity (Hamming)")
        ax.grid(linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=6, ncol=2)

    for ax in axes[len(n_values):]:
        ax.set_visible(False)

    fig.suptitle("Population diversity evolution", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "diversity_evolution.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 4: heatmap — best_fitness[learning_method x n] for each (ws, svd) ─

def plot_heatmap(df, out_dir):
    combos = [(ws, svd, f"{'warm' if ws else 'cold'} {'+ svd' if svd else 'no svd'}")
              for ws in [False, True] for svd in [False, True]]
    methods  = sorted(df["learning_method"].unique())
    n_values = sorted(df["n"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (ws, svd, title) in zip(axes, combos):
        sub = df[(df["warm_start"] == ws) & (df["svd_noise"] == svd)]
        matrix = np.full((len(methods), len(n_values)), np.nan)
        for i, m in enumerate(methods):
            for j, n in enumerate(n_values):
                cell = sub[(sub["learning_method"] == m) & (sub["n"] == n)]["best_fitness"]
                if not cell.empty:
                    matrix[i, j] = cell.values[0]
        im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto")
        ax.set_xticks(range(len(n_values)))
        ax.set_xticklabels(n_values)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("n")
        ax.set_ylabel("Learning method")
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(methods)):
            for j in range(len(n_values)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                            fontsize=8, color="black")

    fig.suptitle("Best fitness heatmap (lower = better)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "heatmap_best_fitness.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 5: wall-clock time per config ──────────────────────────────────────

def plot_wall_clock(df, out_dir):
    n_values = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, len(n_values), figsize=(5 * len(n_values), 5), sharey=False)
    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        sub = df[df["n"] == n].copy()
        sub["label"] = sub.apply(config_label, axis=1)
        colours = [METHOD_COLOUR[m] for m in sub["learning_method"]]
        bars = ax.bar(range(len(sub)), sub["wall_clock_seconds"] / 60,
                      color=colours, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"n = {n}", fontsize=11)
        ax.set_ylabel("Wall-clock time (min)" if n == n_values[0] else "")
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)
        for bar, val in zip(bars, sub["wall_clock_seconds"] / 60):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6)

    handles = [Patch(facecolor=c, label=m) for m, c in METHOD_COLOUR.items()]
    fig.legend(handles=handles, title="Learning method", loc="upper right", fontsize=8)
    fig.suptitle("Wall-clock time per configuration", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "wall_clock_by_n.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="results/experiments_long/experiments_long.csv")
    parser.add_argument("--log-dir", default="logs/experiments_long")
    parser.add_argument("--out-dir", default="plots/experiments_long")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset=["n", "learning_method", "warm_start", "svd_noise", "repeat"],
                            keep="last")

    print(f"Loaded {len(df)} runs from {args.csv}")
    print(df[["n", "learning_method", "warm_start", "svd_noise", "best_fitness",
              "wall_clock_seconds"]].to_string(index=False))
    print()

    plot_bar_by_n(df, args.out_dir)
    plot_convergence(df, args.out_dir)
    plot_diversity(df, args.out_dir)
    plot_heatmap(df, args.out_dir)
    plot_wall_clock(df, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
