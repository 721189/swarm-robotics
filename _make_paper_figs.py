"""Generate ablation + transition figures for the revised paper.

Outputs (into paper/figures/):
  ablation_mtc_vs_swarm.png   - MTC vs swarm size, 4 matched-topology baselines
  ablation_conv_rate.png      - Convergence rate vs packet loss, 4 baselines
  ablation_pdr_vs_loss.png    - PDR vs packet loss (radio-model invariance)
  transition_sweep.png        - Fine sweep around N*tau = T_val
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGD, exist_ok=True)

COLORS = {
    "ideal_omni": "#2ca02c",
    "no_tdma_packet_loss": "#ff7f0e",
    "tdma_clean": "#1f77b4",
    "tdma_packet_loss": "#d62728",
    "fine_sweep_p0.00": "#1f77b4",
    "fine_sweep_p0.15": "#d62728",
}
LABELS = {
    "ideal_omni": "Ideal CBBA",
    "no_tdma_packet_loss": "Loss-only (open ch.)",
    "tdma_clean": "TDMA-only",
    "tdma_packet_loss": "TDMA-CBBA (proposed)",
}

base = pd.read_csv(os.path.join(ROOT, "benchmark", "results", "baseline_sweep.csv"))
fine = pd.read_csv(os.path.join(ROOT, "benchmark", "results", "fine_sweep.csv"))

# ---- Fig 1: ablation MTC vs swarm size -------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for bl, g in base.groupby("baseline"):
    agg = g.groupby("swarm_size")["mtc"]
    ax.errorbar(agg.mean().index, agg.mean(), yerr=agg.std(),
                marker="o", capsize=3, label=LABELS[bl],
                color=COLORS[bl])
ax.set_yscale("log")
ax.set_xlabel("Swarm size $N$")
ax.set_ylabel("Mean time to consensus (s)")
ax.set_title("Radio-model ablation (matched topology:\n"
             "$M{=}5$, $k{=}2$, $\\tau{=}50$\\,ms, 5 trials/point)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "ablation_mtc_vs_swarm.png"), dpi=300)
plt.close(fig)

# ---- Fig 2: convergence rate vs loss ----------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for bl, g in base.groupby("baseline"):
    agg = g.groupby("packet_loss")["converged"].mean() * 100
    ax.plot(agg.index * 100, agg, marker="s", label=LABELS[bl],
            color=COLORS[bl])
ax.set_xlabel("Transmit-leg packet loss $p_{tx}$ (%)")
ax.set_ylabel("Convergence rate within horizon (%)")
ax.set_ylim(-3, 103)
ax.set_title("Convergence rate vs packet loss\n(matched topology)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "ablation_conv_rate.png"), dpi=300)
plt.close(fig)

# ---- Fig 3: PDR invariance across radio models ------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for bl, g in base.groupby("baseline"):
    gl = g[g.packet_loss > 0]
    if len(gl) == 0:
        continue
    agg = gl.groupby("packet_loss")["pdr"].mean()
    ax.plot(agg.index * 100, agg, marker="o", label=LABELS[bl],
            color=COLORS[bl])
ptx = np.linspace(0.04, 0.31, 100)
ax.plot(ptx * 100, 1 - ptx / 2, "k--", linewidth=1.2,
        label=r"analytic $1-p_{tx}/2$")
ax.set_xlabel("Transmit-leg packet loss $p_{tx}$ (%)")
ax.set_ylabel("PDR (conditional on transmission)")
ax.set_title("Per-link PDR is radio-model invariant:\nit measures the channel, not the scheduler")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGD, "ablation_pdr_vs_loss.png"), dpi=300)
plt.close(fig)

# ---- Fig 4: fine transition sweep -------------------------------------------
T_VAL = 2.0
fine["ratio"] = fine["swarm_size"] * 0.100 / T_VAL
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

for pl, g in fine.groupby("packet_loss"):
    key = f"fine_sweep_p{pl:.2f}"
    agg = g.groupby("ratio").agg(conv=("converged", "mean"),
                                 mtc_med=("mtc", "median"))
    axes[0].plot(agg.index, agg.conv * 100, marker="o",
                 label=f"$p_{{tx}}={pl}$", color=COLORS[key])
    axes[1].plot(agg.index, agg.mtc_med, marker="o",
                 label=f"$p_{{tx}}={pl}$", color=COLORS[key])

for ax in axes:
    ax.axvline(1.0, color="k", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$N\tau \,/\, T_{val}$")

axes[0].set_ylabel("First-passage convergence rate (%)")
axes[0].set_ylim(-3, 103)
axes[0].set_title("Convergence rate")
axes[1].set_ylabel("Median MTC (s)")
axes[1].set_title("Median time to consensus")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

# annotate immunity budget regions on left panel
axes[0].annotate(r"$\Lambda \geq 1$", xy=(0.72, 55), fontsize=9, ha="center")
axes[0].annotate(r"$\Lambda = 0$", xy=(0.95, 30), fontsize=9, ha="center")
axes[0].annotate(r"$\Lambda < 0$" + "\ncollapsed", xy=(1.35, 40),
                 fontsize=9, ha="center", color="#d62728")

fig.suptitle(r"Synchronization--staleness transition, $\tau = 100$\,ms, "
             r"$T_{val} = 2$\,s ($N^{\ast} = 20$)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIGD, "transition_sweep.png"), dpi=300)
plt.close(fig)

print("figures written:", sorted(os.listdir(FIGD)))