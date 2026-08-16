"""Publication-ready visualization for benchmark results.

Generates the four headline figures into ``benchmark/results/figures/``:

    1. mtc_vs_swarm_size.png   - MTC (s) vs swarm size, error bars, per baseline
    2. pdr_vs_packet_loss.png  - PDR vs nominal packet loss, bar chart
    3. convergence_curve.png   - empirical CDF of consensus frames (per baseline)
    4. heatmap.png             - MTC heatmap: swarm size x packet loss

Figures use the matplotlib Agg backend so they render headlessly and are
safe to generate on CI / terminal-only machines.

Usage:
    python benchmark/analyzers/visualizer.py [--files benchmark/results/benchmark_*.csv]
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJECT_ROOT, os.path.dirname(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import glob
from typing import List

import matplotlib
matplotlib.use("Agg")  # noqa: E305 - must be set before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.analyzers.stat_analyzer import load_results  # noqa: E402

FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
DPI = 300
PALETTE = {
    "ideal_omni": "#2ca02c",
    "tdma_clean": "#1f77b4",
    "no_tdma_packet_loss": "#ff7f0e",
    "tdma_packet_loss": "#d62728",
}


def _style(ax, xlabel, ylabel, title=None, legend=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    if legend:
        ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_mtc_vs_swarm_size(data: pd.DataFrame, out: str):
    """Line chart with error bars: MTC vs swarm size per baseline."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for baseline, g in data.groupby("baseline"):
        agg = g.groupby("swarm_size")["mtc"]
        means = agg.mean()
        stds = agg.std()
        ax.errorbar(means.index, means, yerr=stds, label=baseline,
                    marker="o", capsize=4, color=PALETTE.get(baseline, None))
    _style(ax, "Swarm Size", "Mean Time to Convergence (s)",
           "MTC vs Swarm Size")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[i] Saved {out}")


def plot_pdr_vs_packet_loss(data: pd.DataFrame, out: str):
    """Grouped bar chart with error bars: PDR vs nominal packet loss."""
    fig, ax = plt.subplots(figsize=(8, 6))
    pls = sorted(data["packet_loss"].dropna().unique())
    lossy = data[data["packet_loss"] > 0]
    lossy_baselines = sorted(lossy["baseline"].unique())
    width = 0.8 / max(len(lossy_baselines), 1)
    for i, baseline in enumerate(lossy_baselines):
        g = lossy[lossy["baseline"] == baseline]
        means = g.groupby("packet_loss")["pdr"].mean().reindex(pls)
        stds = g.groupby("packet_loss")["pdr"].std().reindex(pls)
        x = np.array([float(p) for p in pls]) + (i - (len(lossy_baselines) - 1) / 2) * width
        ax.bar(x, means, width=width, yerr=stds, capsize=3,
               label=baseline, color=PALETTE.get(baseline, None), alpha=0.85)
    ax.set_xticks(pls)
    ax.set_xticklabels([f"{p*100:.0f}%" for p in pls])
    _style(ax, "Nominal Packet Loss Rate", "Packet Delivery Ratio (PDR)",
           "PDR vs Packet Loss (lossy baselines)")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[i] Saved {out}")
def plot_convergence_curve(data: pd.DataFrame, out: str, horizon: int = 500):
    """Empirical CDF of consensus time per baseline (convergence curve)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = np.arange(0, horizon + 1, 5)
    for baseline, g in data.groupby("baseline"):
        conv = g["frames_to_consensus"].values.astype(float)
        conv = conv[(conv >= 0) & (conv <= horizon)]
        if len(conv) == 0:
            continue
        y = np.array([np.mean(conv <= x) for x in xs])
        ax.plot(xs, y, drawstyle="steps-post", label=baseline,
                color=PALETTE.get(baseline, None), linewidth=2)
    _style(ax, "Frame", "Fraction of Trials Converged",
           "Convergence Curve (empirical CDF)",
           legend=len(data["baseline"].unique()) > 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, horizon)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[i] Saved {out}")


def plot_heatmap(data: pd.DataFrame, out: str):
    """Heatmap: mean MTC across swarm_size x packet_loss (proposed baseline)."""
    metric_col = "mtc" if "mtc" in data.columns else "frames_to_consensus"
    candidate = data[data["baseline"] == "tdma_packet_loss"] if \
        "tdma_packet_loss" in set(data["baseline"]) else data
    pivot = candidate.pivot_table(index="swarm_size", columns="packet_loss",
                                  values=metric_col, aggfunc="mean")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c*100:.0f}%" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(i)}" for i in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="white")
    ax.set_xlabel("Packet Loss Rate")
    ax.set_ylabel("Swarm Size")
    ax.set_title("Mean Time to Convergence (s): swarm size x packet loss")
    fig.colorbar(im, ax=ax, shrink=0.8, label="MTC (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[i] Saved {out}")


def generate_all(data: pd.DataFrame, out_dir: str = FIG_DIR) -> List[str]:
    """Create every figure in the results/figures directory."""
    os.makedirs(out_dir, exist_ok=True)

    out = []
    out.append(plot_mtc_vs_swarm_size(
        data, os.path.join(out_dir, "mtc_vs_swarm_size.png")))
    out.append(plot_pdr_vs_packet_loss(
        data, os.path.join(out_dir, "pdr_vs_packet_loss.png")))
    out.append(plot_convergence_curve(
        data, os.path.join(out_dir, "convergence_curve.png")))
    out.append(plot_heatmap(
        data, os.path.join(out_dir, "heatmap.png")))
    return out


def _collect(paths: List[str]) -> pd.DataFrame:
    frames = [load_results(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures.")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Benchmark CSVs (glob ok). Default: all results/*.csv")
    parser.add_argument("--out-dir", default=FIG_DIR)
    args = parser.parse_args()

    if args.files:
        paths = [p for pat in args.files for p in sorted(glob.glob(pat))]
    else:
        results_dir = os.path.dirname(FIG_DIR)
        paths = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not paths:
        raise SystemExit("No CSVs found. Run the benchmark runner first.")

    data = _collect(paths)
    print(f"Loaded {len(data)} rows from {len(paths)} file(s).")
    generate_all(data, out_dir=args.out_dir)
    print("All figures written.")


if __name__ == "__main__":
    main()