"""Core metrics calculator for swarm evaluation.

The four headline metrics computed by the benchmark harness:

    1. SCI - Swarm Cohesion Index: normalised mean pairwise distance between
            alive agents (0 = perfectly co-located, ~1 = spread across field).
    2. CE  - Coverage Efficiency:   fraction of objectives covered by an
            assigned (winning) agent, capped at 1.0.
    3. PDR - Packet Delivery Ratio: messages received / messages sent.
    4. MTC - Mean Time to Convergence: frames-to-consensus converted to time
            using the simulation timestep (default dt = 0.1 s).
"""

import os
import sys
import math
from typing import List, Optional

# Allow standalone use of this module (e.g. inside a Jupyter notebook).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.base_agent import BaseAgent  # noqa: E402


def calculate_sci(agents: List[BaseAgent], field_size: float = 100.0) -> float:
    """Swarm Cohesion Index.

    Average normalised pairwise distance between alive agents.
    Low SCI  -> drones are clustered (cohesive).
    High SCI -> drones are spread out.
    """
    active = [a for a in agents if a.alive]
    if len(active) < 2:
        # No pair distance to measure - behave neutrally.
        return 1.0

    total_dist = 0.0
    count = 0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            total_dist += math.hypot(
                active[i].x - active[j].x, active[i].y - active[j].y
            )
            count += 1

    avg_dist = total_dist / count
    return avg_dist / field_size


def calculate_ce(agents: List[BaseAgent], objectives: list) -> float:
    """Coverage Efficiency.

    Fraction of objectives that have an assigned (winning) alive agent.
    """
    if not objectives:
        return 0.0

    assigned = sum(1 for a in agents if a.alive and a.assigned_task_id is not None)
    return min(1.0, assigned / len(objectives))


def calculate_pdr(packets_sent: int, packets_received: int,
                  packets_offered: Optional[int] = None) -> float:
    """Packet Delivery Ratio.

    When ``packets_offered`` (total attempted sender->receiver links) is
    supplied the ratio is the conventional per-link reliability
    ``received / offered`` which is bounded in [0, 1].  Without it we fall
    back to the throughput-style ratio ``received / sent``.
    """
    if packets_sent <= 0:
        return 0.0
    if packets_offered is not None and packets_offered > 0:
        return packets_received / packets_offered
    return packets_received / packets_sent


def calculate_mtc(frames_to_consensus: Optional[int], dt: float = 0.1,
                  max_frames: int = 500) -> float:
    """Mean Time to Convergence (in seconds).

    Returns ``max_frames * dt`` (default 50.0 s) when consensus was never
    reached within the simulation horizon.
    """
    if frames_to_consensus is None:
        return max_frames * dt
    return frames_to_consensus * dt