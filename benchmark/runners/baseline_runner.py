"""Baseline runner - compares the TDMA+packet-loss swarm against control baselines.

Four radio configurations are benchmarked so the *effect* of each design
decision can be isolated:

    ==========================  ============================================
    label                       model
    ==========================  ============================================
    ideal_omniscient            no TDMA, no packet loss (perfect channel)
    tdma_clean               TDMA slots, no packet loss (loss isolated)
    no_tdma_packet_loss         open channel, packet loss (TDMA isolated)
    tdma_packet_loss            TDMA slots + packet loss (the proposed)
    ==========================  ============================================

Usage
-----
    python benchmark/runners/baseline_runner.py [--trials 5] [--output ...]
"""

import os
import sys
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Only the repo root goes onto sys.path (never its parent), so the repo's own
# ``src/`` cannot be shadowed by a sibling checkout with an older codebase.

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse  # noqa: E402

from benchmark.runners.headless_runner import run_single_trial, save_results  # noqa: E402

# Fixed topology for the baseline comparison sweep (keeps it fast but focused).
OBJ = 5
THREATS = 2
SLOT = 0.050


def run_baseline(
    config_type: str,
    swarm_size: int,
    packet_loss: float,
    trial: int,
    slot_duration: float = SLOT,
    num_objectives: int = OBJ,
    num_threats: int = THREATS,
    max_frames: int = 500,
) -> Dict[str, Any]:
    """Run a single trial for a named baseline radio model."""
    specs = {
        "ideal_omni": {"use_tdma": False, "packet_loss": 0.0},
        "tdma_clean": {"use_tdma": True, "packet_loss": 0.0},
        "no_tdma_packet_loss": {"use_tdma": False, "packet_loss": packet_loss},
        "tdma_packet_loss": {"use_tdma": True, "packet_loss": packet_loss},
    }
    if config_type not in specs:
        raise ValueError(f"Unknown baseline: {config_type!r}. "
                         f"Choose from {sorted(specs)}")

    params = dict(specs[config_type])
    return run_single_trial(
        swarm_size=swarm_size,
        packet_loss=params["packet_loss"],
        num_objectives=num_objectives,
        num_threats=num_threats,
        slot_duration=slot_duration,
        trial=trial,
        use_tdma=params["use_tdma"],
        baseline=config_type,
        max_frames=max_frames,
    )


def run_baseline_matrix(
    trials: int = 5,
    swarm_sizes: Optional[List[int]] = None,
    packet_losses: Optional[List[float]] = None,
    output: Optional[str] = None,
) -> str:
    """Run the baseline sweep and persist a combined CSV.

    The sweep keeps the topology fixed (objectives, threats, slot) and varies
    swarm size and channel loss so the headline figures have clean error bars.
    """
    swarm_sizes = swarm_sizes or [10, 30, 50]
    packet_losses = packet_losses or [0.0, 0.05, 0.15, 0.30]
    baselines = ["ideal_omni", "tdma_clean", "no_tdma_packet_loss", "tdma_packet_loss"]

    # Lossy baselines sweep the PL axis; no-loss baselines are single point.
    lossy_baselines = {"no_tdma_packet_loss", "tdma_packet_loss"}
    pl_runs = sum(
        (len(packet_losses) if b in lossy_baselines else 1)
        for b in baselines
    )
    total = pl_runs * len(swarm_sizes) * trials
    print(f"Baseline matrix: {total} trials "
          f"({len(baselines)} baselines x {len(swarm_sizes)} swarms x "
          f"{pl_runs} PL combos x {trials} trials)")

    rows: List[Dict[str, Any]] = []
    run = 0
    for config_type in baselines:
        for swarm in swarm_sizes:
            pl_values = packet_losses if config_type in lossy_baselines else [0.0]
            for pl in pl_values:
                for trial in range(trials):
                    run += 1
                    if run % 20 == 0 or run == 1:
                        print(f"[{run}/{total}] {config_type:20s} swarm={swarm} "
                              f"pl={pl:.2f} trial={trial}")
                    rows.append(run_baseline(
                        config_type, swarm_size=swarm, packet_loss=pl, trial=trial
                    ))

    print(f"\nBaseline matrix finished ({len(rows)} rows).")
    return save_results(rows, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the baseline sweep.")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-frames", type=int, default=500)
    args = parser.parse_args()

    _save_path = run_baseline_matrix(trials=args.trials, output=args.output)
    print(f"[i] Baselines written to {_save_path}")