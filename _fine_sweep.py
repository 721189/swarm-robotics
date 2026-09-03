"""
Paper-grade synchronization--staleness transition experiment.

Tests the predicted boundary

    rho = N * tau / T_val = 1

with tau = 0.1 s and T_val = 2.0 s, giving N* = 20.
The experiment samples both sides of the boundary using
30 independent seeds per condition.

Output:
    benchmark/results/phase_transition.csv
    benchmark/results/phase_transition_summary.csv
    benchmark/results/figures/phase_transition_curve.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from benchmark.runners.headless_runner import (
    run_single_trial,
    save_results,
)

TAU = 0.100
T_VAL = 2.0

SWARM_SIZES = list(range(12, 31, 2))
LOSSES = [0.0, 0.05, 0.15]

TRIALS = 30
BASE_SEED = 42

ROWS = []
total = len(SWARM_SIZES) * len(LOSSES) * TRIALS
done = 0

for n in SWARM_SIZES:
    for packet_loss in LOSSES:
        for trial in range(TRIALS):
            row = run_single_trial(
                swarm_size=n,
                packet_loss=packet_loss,
                num_objectives=5,
                num_threats=2,
                slot_duration=TAU,
                trial=trial,
                use_tdma=True,
                baseline="phase_transition",
                max_frames=500,
                seed=BASE_SEED,
            )
            row["T_val"] = T_VAL
            row["staleness_ratio"] = n * TAU / T_VAL
            ROWS.append(row)

            done += 1
            print("[{}/{}] N={} loss={} trial={}".format(
                done, total, n, packet_loss, trial), flush=True)

out_csv = os.path.join(ROOT, "benchmark", "results", "phase_transition.csv")
save_results(ROWS, out_csv)

df = pd.DataFrame(ROWS)

summary = (
    df.groupby(["swarm_size", "packet_loss"], as_index=False)
    .agg(
        convergence_rate=("converged", "mean"),
        mtc_mean=("mtc", "mean"),
        mtc_median=("mtc", "median"),
        n=("converged", "size"),
    )
)
summary["staleness_ratio"] = summary["swarm_size"] * TAU / T_VAL

summary_csv = os.path.join(
    ROOT, "benchmark", "results", "phase_transition_summary.csv")
summary.to_csv(summary_csv, index=False)

# ---------------------------------------------
# Plot convergence probability vs ratio
# ---------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

for loss in LOSSES:
    g = summary[summary["packet_loss"] == loss]
    ax.plot(
        g["staleness_ratio"],
        g["convergence_rate"],
        marker="o",
        label="p_tx={:.2f}".format(loss),
    )

ax.axvline(1.0, linestyle="--", label="Predicted threshold")
ax.set_xlabel(r"Staleness ratio $N\tau/T_{\mathrm{val}}$")
ax.set_ylabel("Convergence rate")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Synchronization--Staleness Phase Transition")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(
    os.path.join(ROOT, "benchmark", "results", "figures",
                 "phase_transition_curve.png"),
    dpi=300,
)
plt.close(fig)

print("")
print("PHASE-TRANSITION EXPERIMENT COMPLETE")
print("Raw:", out_csv)
print("Summary:", summary_csv)