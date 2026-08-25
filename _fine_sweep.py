"""Fine-grained synchronization-staleness transition sweep.

Probes the neighborhood of the theoretical threshold N*tau = T_val by holding
tau = 100 ms, T_val = 2 s (threshold at N*=20) and sweeping N through
{12,14,16,18,20,22,24,26,30} at two loss levels.

Output: benchmark/results/fine_sweep.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.runners.headless_runner import run_single_trial, save_results

SWARM_SIZES = [12, 14, 16, 18, 20, 22, 24, 26, 30]
TAU = 0.100                 # threshold at N* = 20
LOSSES = [0.0, 0.15]
TRIALS = 5
BASE_SEED = 42

rows = []
total = len(SWARM_SIZES) * len(LOSSES) * TRIALS
done = 0
for n in SWARM_SIZES:
    for pl in LOSSES:
        for t in range(TRIALS):
            rows.append(run_single_trial(
                swarm_size=n,
                packet_loss=pl,
                num_objectives=5,
                num_threats=2,
                slot_duration=TAU,
                trial=t,
                use_tdma=True,
                baseline="fine_sweep",
                max_frames=500,
                seed=BASE_SEED,
            ))
            done += 1
            print(f"[{done}/{total}] N={n} pl={pl} trial={t}", flush=True)

save_results(rows, os.path.join("benchmark", "results", "fine_sweep.csv"))
print("FINE SWEEP DONE:", len(rows), "runs")