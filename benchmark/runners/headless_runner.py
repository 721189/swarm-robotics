"""Headless simulation runner for benchmark automation.

Runs the (CBBA + TDMA + packet-loss) swarm simulation with absolutely no UI
and no rendering so the benchmark can be executed back-to-back with analysis.
Each trial is fully scripted and deterministic:

    * CBBA assignment (distributed consensus auctions) over a TDMA radio
      discipline (or an open / omniscient radio channel for baselines)
    * Configurable packet loss on the broadcast and per-link reception legs
    * Frame-by-frame convergence tracking (consensus + flock stability)

Every trial calculates the "Big Four" metrics:

    1. SCI  - Swarm Cohesion Index    (normalised mean inter-agent distance)
    2. CE   - Coverage Efficiency     (fraction of objectives with a winner)
    3. PDR  - Packet Delivery Ratio   (received / sent)
    4. MTC  - Mean Time to Convergence (frames-to-consensus * dt)

Results are appended to a CSV in ``benchmark/results/``.

Usage
-----
    # Full 12 960-run paper-grade matrix (long) - default
    python benchmark/runners/headless_runner.py

    # Smaller matrix (good default for development)
    python benchmark/runners/headless_runner.py --config reduced

    # One-row smoke test
    python benchmark/runners/headless_runner.py --config quick

    # Tuned run
    python benchmark/runners/headless_runner.py --config reduced ^
        --trials 5 --workers 4 --max-frames 500
"""

import os
import sys
import csv
import time
import random
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap: make ``src`` and ``benchmark`` importable regardless of the
# working directory or invocation style (``python -m`` vs ``python file.py``).
# NOTE: only the repo root is added -- NEVER its parent -- so the repo's own
# ``src/`` cannot be shadowed by a sibling checkout with an older codebase.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.simulation_engine import SwarmSimulation  # noqa: E402
from src.config.config import (  # noqa: E402
    SwarmConfig,
    SimParamsConfig,
    AgentConfig,
    ObjectiveConfig,
    ThreatConfig,
)
from src.communication.tdma_scheduler import TDMAScheduler  # noqa: E402
from benchmark.metrics.metrics_calculator import (  # noqa: E402
    calculate_sci,
    calculate_ce,
    calculate_pdr,
    calculate_mtc,
)

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
DT = 0.1                         # Simulation timestep (s)
FIELD_BOUNDS = (0.0, 100.0)      # Square battlefield
FIELD_SIZE = 100.0               # For SCI normalisation
DEFAULT_MAX_FRAMES = 500         # Simulation horizon (frames)
TELEMETRY_MAX_AGE = 2.0          # Perceived-world timeout for ghost logic (s)
STABILITY_HOLD = 5               # Consecutive identical assignment-signature frames
                                 # required to declare the swarm "stable".
CONSENSUS_TIMEOUT = 500           # MTC cap when consensus never arrives
VERIFY_HOLD = int(round(TELEMETRY_MAX_AGE / DT))  # Frames (>= T_val) that ledger
                                 # agreement must PERSIST to count as *verified*
                                 # consensus, not merely first-passage agreement.

RESULTS_HEADERS = [
    "baseline", "use_tdma", "swarm_size", "packet_loss", "num_objectives",
    "num_threats", "slot_duration", "trial", "sci", "ce", "pdr", "mtc",
    "converged", "frames_to_consensus", "consensus_verified",
    "consensus_regressions", "frames_to_stability",
    "packets_sent", "packets_received", "packets_offered",
    "tx_opportunities", "p_tx_success", "p_rx_given_tx", "p_e2e",
    "total_frames",
    "timestamp",
]
def run_single_trial(
    swarm_size: int,
    packet_loss: float,
    num_objectives: int,
    num_threats: int,
    slot_duration: float,
    trial: int,
    use_tdma: bool = True,
    baseline: str = "tdma_packet_loss",
    max_frames: int = DEFAULT_MAX_FRAMES,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one simulation trial and return the "Big Four" metrics.

    Parameters
    ----------
    swarm_size, packet_loss, num_objectives, num_threats, slot_duration, trial :
        Benchmark matrix coordinates. The per-trial RNG is ``seed + trial`` so
        every config x trial combination is reproducible.
    use_tdma : if True a TDMAScheduler arbitrates who may speak every frame;
        if False every alive drone broadcasts each frame (open channel).
    baseline : label written into the results row (e.g. ``ideal_omniscient``).
    """
    trial_rng = random.Random(seed + trial)

    # --- Build Config --------------------------------------------------------
    sim_params = SimParamsConfig(
        seed=seed + trial,
        bounds=FIELD_BOUNDS,
        dt=DT,
        max_frames=max_frames,
        algorithm="cbba",
    )

    agents = [
        AgentConfig(
            name="combat_drone",
            count=swarm_size,
            params={
                "speed": 1.5,
                "max_force": 0.2,
                "sense_radius": 25.0,
                "battery": 9999.0,
                "drain": 0.0,
                "attr_weight": 1.0,
                "sam_repulsion_weight": 4.5,
                "boids_separation_weight": 1.5,
            },
        )
    ]

    positions = [
        (trial_rng.uniform(15.0, 85.0), trial_rng.uniform(15.0, 85.0))
        for _ in range(num_objectives)
    ]
    objectives = ObjectiveConfig(count=num_objectives, positions=positions)

    threats = [
        ThreatConfig(
            id=i,
            x=trial_rng.uniform(20.0, 80.0),
            y=trial_rng.uniform(20.0, 80.0),
            radius=15.0,
            strength=60.0,
        )
        for i in range(num_threats)
    ]

    config = SwarmConfig(
        simulation=sim_params, agents=agents, objectives=objectives, threats=threats
    )

    # --- Initialise simulation + radio discipline ----------------------------
    sim = SwarmSimulation(config)
    tdma: Optional[TDMAScheduler] = None
    if use_tdma:
        tdma = TDMAScheduler(num_drones=swarm_size, slot_duration_ms=slot_duration * 1000.0)

        # --- Metrics tracking -----------------------------------------------------
    frames_to_consensus: Optional[int] = None
    frames_to_stability: Optional[int] = None
    consensus_regressions = 0  # agreement broken after first passage (expiry events)
    agree_streak = 0
    packets_sent = 0
    packets_received = 0
    packets_offered = 0
    tx_opportunities = 0  # scheduled speaker slots with an alive speaker (TDMA)
    # Rolling window of the per-frame assignment signature, used to detect
    # task-assignment stability (agents stop flip-flopping between targets).
    signature_history: List[tuple] = []

    def _consensus_reached() -> bool:
        """Global CBBA consensus: all alive drones share one winner ledger.

        Radio gossip (TDMA slots / packet loss) is how each drone learns
        every other drone's bids, so the frame this becomes true is the
        convergence time the benchmark is after.
        """
        a_agents = [a for a in sim.agents if a.alive]
        if len(a_agents) < 2 or not sim.objectives:
            return False
        obj_ids = [o.obj_id for o in sim.objectives]
        sigs = set()
        for a in a_agents:
            sigs.add(tuple(
                (oid, a.local_winners.get(oid, -1)) for oid in obj_ids
            ))
        if len(sigs) != 1:
            return False
        first = next(iter(sigs))
        return any(w != -1 for (_, w) in first)

    def _assignment_signature() -> tuple:
        """Per-agent task assignment (sorted by agent id) for stability test."""
        return tuple(sorted(
            (a.agent_id, a.assigned_task_id if a.assigned_task_id is not None else -1)
            for a in sim.agents if a.alive
        ))

    # --- Run loop --------------------------------------------------------------
    for frame_idx in range(max_frames):
        # 1. Radio step: advance the TDMA clock first; if there is no scheduler
        #    (open-channel baselines), every alive drone is a transmitter.
        if tdma is not None:
            slot_events = tdma.advance(DT)

            for event in slot_events:
                speaker_id = event.speaker_id

                if speaker_id >= len(sim.agents):
                    continue

                speaker = sim.agents[speaker_id]

                if not speaker.alive:
                    continue

                # Every scheduled speaker -> receiver pair is an offered
                # end-to-end link opportunity, including TX-side failure.
                receivers = [
                    a for a in sim.agents
                    if a.alive and a.agent_id != speaker_id
                ]

                packets_offered += len(receivers)
                tx_opportunities += 1

                # TX-stage failure affects the entire broadcast.
                if trial_rng.random() <= packet_loss:
                    if verbose:
                        print(f"   [frame {frame_idx}] packet LOST at sender {speaker_id}")
                    continue

                packets_sent += 1

                packet = speaker.prepare_broadcast(event.start_time)

                for agent in receivers:
                    if trial_rng.random() <= packet_loss * 0.5:
                        continue

                    agent.receive_broadcast(speaker_id, packet)
                    packets_received += 1
        else:
            # Baseline / open channel: every alive drone broadcasts every frame.
            for agent in sim.agents:
                if not agent.alive:
                    continue
                receivers = [
                    a for a in sim.agents
                    if a.alive and a.agent_id != agent.agent_id
                ]
                packets_offered += len(receivers)
                tx_opportunities += 1
                if trial_rng.random() <= packet_loss:
                    continue
                packets_sent += 1
                packet = agent.prepare_broadcast(float(frame_idx) * DT)
                for r in receivers:
                    if trial_rng.random() <= packet_loss * 0.5:
                        continue
                    r.receive_broadcast(agent.agent_id, packet)
                    packets_received += 1

        # 2. (Reception handled per-link inside the radio phase above.)

        # 3. Agent decisions/state updates under their (lossy) perceived world.
        sim_time = tdma.current_time if tdma is not None else float(frame_idx) * DT
        for agent in sim.agents:
            if agent.alive:
                perceived = agent.get_perceived_world(sim_time, max_age=TELEMETRY_MAX_AGE)
                agent.step(dt=DT, perceived_swarm=perceived, simulation=sim)

        # 4. Radio clock was already advanced by advance(DT) in step 1.

        # --- Convergence / stability checks -------------------------------------
        alive = [a for a in sim.agents if a.alive]

        # Consensus tracking distinguishes TWO definitions:
        #   first-passage: the FIRST frame at which all alive ledgers agree
        #   verified     : agreement then PERSISTS for VERIFY_HOLD consecutive
        #                  frames (>= T_val), i.e. belief expiry does not break it.
        agree_now = _consensus_reached()
        if frames_to_consensus is None and agree_now:
            frames_to_consensus = frame_idx
        elif frames_to_consensus is not None and not agree_now:
            consensus_regressions += 1
        agree_streak = agree_streak + 1 if agree_now else 0

        # Stability: assignment unchanged for a window of STABILITY_HOLD frames.
        if frames_to_stability is None and len(alive) > 0:
            signature_history.append(_assignment_signature())
            if len(signature_history) >= STABILITY_HOLD and \
                    len(set(signature_history[-STABILITY_HOLD:])) == 1:
                frames_to_stability = frame_idx

        consensus_verified = (
            frames_to_consensus is not None
            and frames_to_stability is not None
            and agree_streak >= VERIFY_HOLD
        )
        if consensus_verified:
            break

    # --- Calculate the "Big Four" metrics ---------------------------------------
    sci = calculate_sci(sim.agents, field_size=FIELD_SIZE)
    ce = calculate_ce(sim.agents, sim.objectives)
    pdr = calculate_pdr(packets_sent, packets_received, packets_offered)
    mtc = calculate_mtc(frames_to_consensus, dt=DT, max_frames=max_frames)

    # Communication reliability decomposition (Sec. Radio model):
    #   p_tx_success : P(packet leaves the sender | scheduled opportunity)
    #   pdr          : end-to-end link success = #received / #offered. Offered
    #                  counts EVERY scheduled sender->receiver pair, including
    #                  TX-failed slots, so pdr == q = (1-p_tx)(1-p_rx).
    #   p_rx_given_tx: empirical decode rate conditional on a surviving TX.
    verified = (frames_to_consensus is not None and agree_streak >= VERIFY_HOLD)
    p_tx_success = packets_sent / tx_opportunities if tx_opportunities > 0 else 0.0
    p_rx_cond = pdr / p_tx_success if p_tx_success > 0 else 0.0
    p_e2e = pdr

    result = {
        "baseline": baseline,
        "use_tdma": int(bool(use_tdma)),
        "swarm_size": swarm_size,
        "packet_loss": round(packet_loss, 4),
        "num_objectives": num_objectives,
        "num_threats": num_threats,
        "slot_duration": slot_duration,
        "trial": trial,
        "sci": round(sci, 6),
        "ce": round(ce, 6),
        "pdr": round(pdr, 6),
        "mtc": round(mtc, 6),
        "converged": frames_to_consensus is not None,
        "frames_to_consensus": frames_to_consensus if frames_to_consensus is not None else max_frames,
        "consensus_verified": int(verified),
        "consensus_regressions": consensus_regressions,
        "frames_to_stability": frames_to_stability if frames_to_stability is not None else max_frames,
        "packets_sent": packets_sent,
        "packets_received": packets_received,
        "packets_offered": packets_offered,
        "tx_opportunities": tx_opportunities,
        "p_tx_success": round(p_tx_success, 6),
        "p_rx_given_tx": round(p_rx_cond, 6),
        "p_e2e": round(p_e2e, 6),
        "total_frames": frame_idx + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return result
# ---------------------------------------------------------------------------
# Benchmark matrix builders
# ---------------------------------------------------------------------------
def build_matrix(config: str = "reduced", trials: Optional[int] = None) -> tuple:
    """Return (list-of-config-dicts, trials_per_config, human description)."""
    if config == "full":
        # The paper-grade grid from the Phase-1 spec.
        runs_per = trials or 30
        matrix = dict(
            swarm_sizes=[10, 30, 50, 100],
            packet_losses=[0.0, 0.05, 0.15, 0.30],
            num_objectives_list=[3, 5, 10],
            num_threats_list=[0, 2, 5],
            slot_durations=[0.025, 0.050, 0.100],
        )
    elif config == "quick":
        runs_per = 1
        matrix = dict(
            swarm_sizes=[5],
            packet_losses=[0.0],
            num_objectives_list=[3],
            num_threats_list=[0],
            slot_durations=[0.050],
        )
    else:  # "reduced" - development default, still statistically usable.
        runs_per = trials or 3
        matrix = dict(
            swarm_sizes=[10, 30, 50],
            packet_losses=[0.0, 0.05, 0.15, 0.30],
            num_objectives_list=[3, 5, 10],
            num_threats_list=[0, 2, 5],
            slot_durations=[0.025, 0.050, 0.100],
        )

    tasks: List[Dict[str, Any]] = []
    for swarm_size in matrix["swarm_sizes"]:
        for packet_loss in matrix["packet_losses"]:
            for num_objectives in matrix["num_objectives_list"]:
                for num_threats in matrix["num_threats_list"]:
                    for slot_duration in matrix["slot_durations"]:
                        for trial in range(runs_per):
                            tasks.append(
                                {
                                    "swarm_size": swarm_size,
                                    "packet_loss": packet_loss,
                                    "num_objectives": num_objectives,
                                    "num_threats": num_threats,
                                    "slot_duration": slot_duration,
                                    "trial": trial,
                                    "use_tdma": True,
                                    "baseline": "tdma_packet_loss",
                                }
                            )
    desc = (
        f"{config} matrix: {len(tasks)} trials "
        f"({len(matrix['swarm_sizes'])} swarms x {len(matrix['packet_losses'])} PL x "
        f"{len(matrix['num_objectives_list'])} objectives x {len(matrix['num_threats_list'])} threats x "
        f"{len(matrix['slot_durations'])} slots x {runs_per} trials)"
    )
    return tasks, runs_per, desc


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def save_results(results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
    """Save results to CSV. Returns the filename written."""
    if not results:
        raise ValueError("No results to save.")

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
            f"benchmark_{timestamp}.csv",
        )

    filename = os.path.abspath(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fieldnames = RESULTS_HEADERS if all(h in results[0] for h in RESULTS_HEADERS) else list(results[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[i] Results saved to {filename} ({len(results)} rows)")
    return filename
# ---------------------------------------------------------------------------
# Benchmark drivers
# ---------------------------------------------------------------------------
def run_benchmark(
    config: str = "reduced",
    trials: Optional[int] = None,
    workers: int = 1,
    max_frames: int = DEFAULT_MAX_FRAMES,
    seed: int = 42,
    output: Optional[str] = None,
) -> str:
    """Run the benchmark matrix (sequential or multi-process) and save CSV."""
    tasks, runs_per, desc = build_matrix(config, trials)
    total_runs = len(tasks)

    print(f"Starting benchmark: {desc}")
    print(f"Max frames/trial: {max_frames}, workers: {workers}")
    print("=" * 70)

    results: List[Dict[str, Any]] = []
    start = time.perf_counter()

    def _prepare(task: Dict[str, Any]) -> Dict[str, Any]:
        t = dict(task)
        t["use_tdma"] = True
        t["baseline"] = "tdma_packet_loss"
        t.setdefault("max_frames", max_frames)
        t.setdefault("seed", seed)
        return t

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        prepared = [_prepare(t) for t in tasks]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            done = 0
            for res in executor.map(_run_task, prepared, chunksize=4):
                results.append(res)
                done += 1
                if done % 20 == 0 or done == total_runs:
                    elapsed = time.perf_counter() - start
                    print(f"[{done}/{total_runs}] done in {elapsed:.1f}s "
                          f"({elapsed / max(done, 1):.3f}s/run)")
    else:
        for i, task in enumerate(tasks, start=1):
            t = _prepare(task)
            if i % 20 == 0 or i in (1, total_runs):
                elapsed = time.perf_counter() - start
                print(f"[{i}/{total_runs}] swarm={t['swarm_size']} pl={t['packet_loss']:.2f} "
                      f"obj={t['num_objectives']} thr={t['num_threats']} "
                      f"slot={t['slot_duration']*1000:.0f}ms trial={t['trial']} "
                      f"({elapsed:.1f}s, {elapsed / i:.3f}s/run)")
            results.append(_run_task(t))

    elapsed = time.perf_counter() - start
    print(f"\nBenchmark finished in {elapsed:.1f}s "
          f"({elapsed / max(len(results), 1):.3f}s/run).")
    return save_results(results, output)


def _run_task(params: Dict[str, Any]) -> Dict[str, Any]:
    """Thin wrapper so concurrent.futures can serialize the worker cleanly."""
    try:
        return run_single_trial(**params)
    except Exception as exc:  # keep the run alive; row is still recorded
        return {
            "baseline": params.get("baseline", "error"),
            "use_tdma": params.get("use_tdma", True),
            "swarm_size": params.get("swarm_size", -1),
            "packet_loss": params.get("packet_loss", -1.0),
            "num_objectives": params.get("num_objectives", -1),
            "num_threats": params.get("num_threats", -1),
            "slot_duration": params.get("slot_duration", -1.0),
            "trial": params.get("trial", -1),
            "sci": -1.0, "ce": -1.0, "pdr": -1.0, "mtc": -1.0,
            "converged": False, "frames_to_consensus": -1,
            "frames_to_stability": -1, "packets_sent": -1,
            "packets_received": -1, "total_frames": -1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(description="Headless swarm benchmark runner.")
    parser.add_argument("--config", choices=["full", "reduced", "quick"],
                        default="reduced",
                        help="Matrix size (default: reduced). full = paper grid.")
    parser.add_argument("--trials", type=int, default=None,
                        help="Override trials-per-config.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (0/1 = sequential).")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help="Simulation horizon in frames.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed; real seed = base + trial.")
    parser.add_argument("--output", default=None,
                        help="Explicit output CSV path.")
    args = parser.parse_args()

    run_benchmark(
        config=args.config,
        trials=args.trials,
        workers=args.workers,
        max_frames=args.max_frames,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()