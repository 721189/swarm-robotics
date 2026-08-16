"""Run all statistical tests on the benchmark CSVs.

Produces a human-readable text report (``statistical_report.txt``) inside
``benchmark/results/`` with:

    1. Per-metric normality (Shapiro-Wilk) checks
    2. One-way ANOVA: metric vs baseline and metric vs use_tdma
    3. 95% confidence intervals for every baseline x metric cell
    4. Effect delta between the proposed TDMA system and the ideal baseline

Usage:
    python benchmark/analyzers/stat_runner.py benchmark/results/benchmark_*.csv
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJECT_ROOT, os.path.dirname(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glob
from typing import List

import pandas as pd

from benchmark.analyzers.stat_analyzer import (
    load_results,
    test_normality,
    run_anova,
    confidence_interval,
    aggregate_by,
    METRICS,
)


def run_full_analysis(csv_paths: List[str], out_dir: str) -> str:
    """Analyse every CSV passed in, writing a report to ``out_dir``."""
    frames = [load_results(p) for p in csv_paths]
    if not frames:
        raise SystemExit("No result CSVs found (glob matched nothing).")
    data = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(data)} rows from {len(frames)} file(s).")

    # Classify the radio model from the stored columns.
    if "use_tdma" in data.columns:
        data["radio"] = data["use_tdma"].map({1: "TDMA", 0: "OpenCh"})
    if "baseline" not in data.columns:
        data["baseline"] = data.get("radio", "unknown")
    data["baseline"] = data["baseline"].astype(str)

    lines: List[str] = []
    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("STATISTICAL ANALYSIS REPORT - Swarm Robotics Benchmark")
    log(f"Source files: {', '.join(os.path.basename(p) for p in csv_paths)}")
    log(f"Rows: {len(data)} | Baselines: {sorted(data['baseline'].unique())}")
    log("=" * 72)

    # --- 1. Normality -----------------------------------------------------.
    log("\n1. Normality Tests (Shapiro-Wilk):")
    log(f"   {'metric':6s} {'n':>6s} {'statistic':>10s} {'p-value':>9s} verdict")
    for metric in METRICS:
        if metric not in data.columns:
            continue
        res = test_normality(data[metric])
        verd = "Normal" if res.get("normal") else "NOT normal"
        log(f"   {metric:6s} {res.get('sample_size', 0):>6d} "
            f"{res['statistic']:>10.4f} {res['p_value']:>9.4f}  {verd}")

    # --- 2. ANOVA ----------------------------------------------------------.
    log("\n2. One-way ANOVA (alpha = 0.05):")
    for metric in METRICS:
        if metric not in data.columns:
            continue
        by_base = run_anova(data, metric, "baseline")
        log(f"   {metric} by baseline: F={by_base['f_stat']:.3f}, "
            f"p={by_base['p_value']:.4f} "
            f"[{'SIGNIFICANT' if by_base.get('significant') else 'n.s.'}]")
    if "use_tdma" in data.columns:
        mtc_tdma = run_anova(data, "mtc", "use_tdma")
        log(f"   MTC by use_tdma: F={mtc_tdma['f_stat']:.3f}, "
            f"p={mtc_tdma['p_value']:.4f} "
            f"[{'SIGNIFICANT' if mtc_tdma.get('significant') else 'n.s.'}]")

    # --- 3. Confidence intervals per baseline -----------------------------------.
    log("\n3. 95% Confidence Intervals by baseline:")
    log(f"   {'baseline':24s} {'metric':6s} {'n':>4s} {'mean':>10s} {'sem':>8s} "
        f"{'CI low':>10s} {'CI high':>10s}")
    for baseline in sorted(data['baseline'].unique()):
        for metric in METRICS:
            if metric not in data.columns:
                continue
            subset = data[data['baseline'] == baseline][metric]
            ci = confidence_interval(subset)
            log(f"   {baseline:24s} {metric:6s} {ci['n']:>4d} {ci['mean']:>10.4f} "
                f"{ci['sem']:>10.4f} {ci['ci'][0]:>10.4f} {ci['ci'][1]:>10.4f}")

    # --- 4. Effect size vs ideal baseline --------------------------------------.
    log("\n4. Convergence deltas vs 'ideal_omni' baseline:")
    if "ideal_omni" in data['baseline'].values:
        ideal = data[data['baseline'] == 'ideal_omni']['mtc'].mean()
        for baseline in sorted(data['baseline'].unique()):
            if baseline == 'ideal_omni':
                continue
            mean_mtc = data[data['baseline'] == baseline]['mtc'].mean()
            delta = mean_mtc - ideal if ideal else float("nan")
            log(f"   {baseline:24s} MTC={mean_mtc:7.3f}s delta={delta:+8.3f}s "
                f"({delta / ideal * 100:+.1f}%)" if ideal else
                f"   {baseline:24s} MTC={mean_mtc:7.3f}s")

    # Optional summary tables CSV for later plotting.
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "statistical_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[i] Report written to {report_path}")
    return report_path


def _parse_csv_paths(args: List[str]) -> List[str]:
    """Expand glob patterns / file lists from argv."""
    paths: List[str] = []
    for a in args:
        paths.extend(sorted(glob.glob(a)))
    return paths


def main():
    """Report over every benchmark CSV under benchmark/results by default."""
    argv = sys.argv[1:]
    if argv:
        csv_paths = _parse_csv_paths(argv)
    else:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        csv_paths = sorted(glob.glob(os.path.join(results_dir, "benchmark_*.csv")))
        csv_paths += sorted(glob.glob(os.path.join(results_dir, "baseline_*.csv")))
    if not csv_paths:
        raise SystemExit("No benchmark CSVs found. Run the runner first.")
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    run_full_analysis(csv_paths, out_dir)


if __name__ == "__main__":
    main()