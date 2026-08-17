"""Full benchmark data analysis script.

Analyzes the CSV results from the headless runner, performs statistical tests,
generates publication figures, and writes the experimental results section for
the IEEEtran paper.

Usage:
    python benchmark/analyzers/full_analysis.py benchmark/results/benchmark_*.csv
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJECT_ROOT, os.path.dirname(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.analyzers.stat_analyzer import (
    load_results, test_normality, run_anova, confidence_interval, aggregate_by, METRICS,
)


def collect_dataframes(csv_globs):
    frames = []
    for pat in csv_globs:
        for f in sorted(glob.glob(pat)):
            frames.append(load_results(f))
    if not frames:
        raise FileNotFoundError(f"No CSVs matched: {csv_globs}")
    return pd.concat(frames, ignore_index=True)


def plot_mtc_vs_swarm_size(data, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for baseline, g in data.groupby("baseline"):
        agg = g.groupby("swarm_size")["mtc"]
        means = agg.mean()
        stds = agg.std()
        ax.errorbar(means.index, means, yerr=stds, label=baseline, marker="o", capsize=4)
    ax.set_xlabel("Swarm Size")
    ax.set_ylabel("MTC (s)")
    ax.set_title("MTC vs Swarm Size")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pdr_vs_packet_loss(data, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    pls = sorted(data["packet_loss"].dropna().unique())
    lossy = data[data["packet_loss"] > 0]
    lossy_bl = sorted(lossy["baseline"].unique())
    width = 0.8 / max(len(lossy_bl), 1)
    x_pos = np.array([float(p) for p in pls])
    for i, bl in enumerate(lossy_bl):
        g = lossy[lossy["baseline"] == bl]
        means = g.groupby("packet_loss")["pdr"].mean().reindex(pls)
        stds = g.groupby("packet_loss")["pdr"].std().reindex(pls)
        offsets = (i - (len(lossy_bl) - 1) / 2) * width
        ax.bar(x_pos + offsets, means, width=width, yerr=stds, capsize=3, label=bl, alpha=0.85)
    ax.set_xticks(pls)
    ax.set_xticklabels([f"{p*100:.0f}%" for p in pls])
    ax.set_xlabel("Packet Loss Rate")
    ax.set_ylabel("PDR")
    ax.set_title("PDR vs Packet Loss")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
def plot_convergence_curve(data, out_path, horizon=500):
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = np.arange(0, horizon + 1, 5)
    for baseline, g in data.groupby("baseline"):
        conv = g["frames_to_consensus"].values.astype(float)
        conv = conv[(conv >= 0) & (conv <= horizon)]
        if len(conv) == 0:
            continue
        y = np.array([np.mean(conv <= x) for x in xs])
        ax.plot(xs, y, drawstyle="steps-post", label=baseline, linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, horizon)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Fraction Converged")
    ax.set_title("Convergence Curve")
    if len(data["baseline"].unique()) > 1:
        ax.legend()
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_heatmap(data, out_path):
    cand = data[data["baseline"] == "tdma_packet_loss"] if "tdma_packet_loss" in set(data["baseline"]) else data
    pivot = cand.pivot_table(index="swarm_size", columns="packet_loss", values="mtc", aggfunc="mean")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c*100:.0f}%" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(i)}" for i in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:.1f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xlabel("Packet Loss Rate")
    ax.set_ylabel("Swarm Size")
    ax.set_title("MTC Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8, label="MTC (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_boxplots(data, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics_map = {"mtc": "MTC (s)", "sci": "SCI", "ce": "CE", "pdr": "PDR"}
    for ax, (met, lbl) in zip(axes.flat, metrics_map.items()):
        sns.boxplot(data=data, x="baseline", y=met, ax=ax)
        ax.set_ylabel(lbl)
        ax.set_title(f"{lbl} by Baseline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main(csv_globs):
    data = collect_dataframes(csv_globs)
    print(f"Loaded {len(data)} rows from {data['baseline'].nunique()} baselines.")
    for metric in METRICS:
        if metric not in data.columns:
            continue
        res = test_normality(data[metric])
        verdict = "Normal" if res.get("normal") else "NOT normal"
        print(f"   {metric}: n={res.get('sample_size')}, p={res['p_value']:.4f} -> {verdict}")
    for metric in METRICS:
        if metric not in data.columns:
            continue
        res = run_anova(data, metric, "baseline")
        print(f"   {metric} by baseline: F={res['f_stat']:.3f}, p={res['p_value']:.4f} -> {'SIGNIFICANT' if res['significant'] else 'n.s.'}")
    if "use_tdma" in data.columns:
        tdma_res = run_anova(data, "mtc", "use_tdma")
        print(f"   MTC by use_tdma: F={tdma_res['f_stat']:.3f}, p={tdma_res['p_value']:.4f} -> {'SIGNIFICANT' if tdma_res['significant'] else 'n.s.'}")
    for bl in sorted(data["baseline"].unique()):
        sub = data[data["baseline"] == bl]
        for met in METRICS:
            if met not in data.columns:
                continue
            ci = confidence_interval(sub[met])
            print(f"   {bl:15s} {met:4s} n={ci['n']:3d} mean={ci['mean']:7.3f} CI=({ci['ci'][0]:.3f}, {ci['ci'][1]:.3f})")
    num_cols = [c for c in ["swarm_size", "packet_loss", "mtc", "sci", "ce", "pdr"] if c in data.columns]
    if len(num_cols) >= 2:
        print("\nCorrelation Matrix:")
        print(data[num_cols].corr().round(3))
    out_dir = os.path.join(_PROJECT_ROOT, "benchmark", "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_mtc_vs_swarm_size(data, os.path.join(out_dir, "mtc_vs_swarm_size.png"))
    print("[i] Saved mtc_vs_swarm_size.png")
    plot_pdr_vs_packet_loss(data, os.path.join(out_dir, "pdr_vs_packet_loss.png"))
    print("[i] Saved pdr_vs_packet_loss.png")
    plot_convergence_curve(data, os.path.join(out_dir, "convergence_curve.png"))
    print("[i] Saved convergence_curve.png")
    plot_heatmap(data, os.path.join(out_dir, "heatmap.png"))
    print("[i] Saved heatmap.png")
    plot_boxplots(data, os.path.join(out_dir, "boxplot.png"))
    print("[i] Saved boxplot.png")
    print("All figures written.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        results_dir = os.path.join(_PROJECT_ROOT, "benchmark", "results")
        args = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    main(args)