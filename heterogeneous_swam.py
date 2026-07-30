"""Heterogeneous swarm: scouts explore and paint objectives; workers follow paint only.

Run:
    python "heterogeneous swam.py"
    python "heterogeneous swam.py" --out heterogeneous_swarm.gif

Local interaction rules work across robot species via shared environmental
markers (scout paint) rather than direct peer-to-peer messaging.
"""

import argparse
import math
import random
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def normalize(dx: float, dy: float) -> Tuple[float, float]:
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0, 0.0
    return dx / length, dy / length


class Objective:
    """Target location; workers ignore it until a scout paints it."""

    def __init__(self, obj_id: int, x: float, y: float):
        self.obj_id = obj_id
        self.x = x
        self.y = y
        self.painted = False
        self.paint_strength = 0.0
        self.painted_by: Optional[int] = None

    def apply_paint(self, scout_id: int, amount: float = 1.0) -> None:
        self.painted = True
        self.paint_strength = min(1.0, self.paint_strength + amount)
        self.painted_by = scout_id


class Agent:
    species: str = "base"
    base_speed: float = 1.0
    sense_radius: float = 5.0
    max_battery: float = 100.0
    battery_drain: float = 0.05
    idle_drain: float = 0.01

    def __init__(self, agent_id: int, x: float, y: float):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.battery = self.max_battery
        self.state = "active"
        self.task = "idle"

    @property
    def alive(self) -> bool:
        return self.battery > 0 and self.state != "depleted"

    def drain_battery(self, moving: bool) -> None:
        cost = self.battery_drain if moving else self.idle_drain
        self.battery = max(0.0, self.battery - cost)
        if self.battery <= 0:
            self.state = "depleted"
            self.task = "depleted"

    def step(self, bounds: Tuple[float, float]) -> None:
        if not self.alive:
            return
        dx, dy, moving = self.decide()
        speed = self.base_speed * (self.battery / self.max_battery * 0.4 + 0.6)
        self.x = clamp(self.x + dx * speed, bounds[0] + 1, bounds[1] - 1)
        self.y = clamp(self.y + dy * speed, bounds[0] + 1, bounds[1] - 1)
        self.drain_battery(moving)

    def decide(self) -> Tuple[float, float, bool]:
        raise NotImplementedError

    def sense_objectives(
        self, objectives: List[Objective], require_paint: bool
    ) -> List[Objective]:
        visible = []
        for obj in objectives:
            if require_paint and not obj.painted:
                continue
            if distance(self.x, self.y, obj.x, obj.y) <= self.sense_radius:
                visible.append(obj)
        return visible


class Scout(Agent):
    """Fast explorer with wide sensing; burns battery quickly; paints objectives."""

    species = "scout"
    base_speed = 2.0
    sense_radius = 18.0
    max_battery = 80.0
    battery_drain = 0.22
    idle_drain = 0.08

    def __init__(self, agent_id: int, x: float, y: float):
        super().__init__(agent_id, x, y)
        self.wander_angle = random.uniform(0, 2 * math.pi)

    def decide(self) -> Tuple[float, float, bool]:
        return 0.0, 0.0, False

    def step(
        self,
        bounds: Tuple[float, float],
        objectives: List[Objective],
        all_agents: List[Agent],
    ) -> None:
        if not self.alive:
            return

        visible = self.sense_objectives(objectives, require_paint=False)
        dx, dy, moving = self._explore(visible, all_agents)

        speed = self.base_speed * (self.battery / self.max_battery * 0.35 + 0.65)
        self.x = clamp(self.x + dx * speed, bounds[0] + 1, bounds[1] - 1)
        self.y = clamp(self.y + dy * speed, bounds[0] + 1, bounds[1] - 1)
        self.drain_battery(moving)

        for obj in visible:
            if not obj.painted:
                obj.apply_paint(self.agent_id)
                self.task = f"painting #{obj.obj_id}"

        if not visible:
            self.task = "exploring"

    def _explore(
        self, visible: List[Objective], all_agents: List[Agent]
    ) -> Tuple[float, float, bool]:
        if visible:
            target = min(visible, key=lambda o: distance(self.x, self.y, o.x, o.y))
            self.task = f"approaching #{target.obj_id}"
            return normalize(target.x - self.x, target.y - self.y) + (True,)

        self.wander_angle += random.uniform(-0.7, 0.7)
        dx = math.cos(self.wander_angle)
        dy = math.sin(self.wander_angle)

        for other in all_agents:
            if other.agent_id == self.agent_id or not other.alive:
                continue
            dist = distance(self.x, self.y, other.x, other.y)
            if dist < 4 and dist > 0:
                push = (4 - dist) / 4
                dx += (self.x - other.x) / dist * push
                dy += (self.y - other.y) / dist * push

        dx, dy = normalize(dx, dy)
        return dx, dy, True


class Worker(Agent):
    """Slow, long-range battery; moves only toward scout-painted objectives."""

    species = "worker"
    base_speed = 1.0
    sense_radius = 7.0
    max_battery = 220.0
    battery_drain = 0.04
    idle_drain = 0.008

    def decide(self) -> Tuple[float, float, bool]:
        return 0.0, 0.0, False

    def step(
        self,
        bounds: Tuple[float, float],
        objectives: List[Objective],
        all_agents: List[Agent],
    ) -> None:
        if not self.alive:
            return

        painted_visible = self.sense_objectives(objectives, require_paint=True)
        unpainted_visible = [
            o
            for o in objectives
            if not o.painted
            and distance(self.x, self.y, o.x, o.y) <= self.sense_radius
        ]

        if painted_visible:
            target = min(
                painted_visible,
                key=lambda o: distance(self.x, self.y, o.x, o.y),
            )
            dx, dy = normalize(target.x - self.x, target.y - self.y)
            self.task = f"following paint -> #{target.obj_id}"
            moving = True
        elif unpainted_visible:
            dx, dy = 0.0, 0.0
            self.task = "waiting for scout paint"
            moving = False
        else:
            dx, dy = self._local_drift(all_agents)
            self.task = "patrolling"
            moving = math.hypot(dx, dy) > 0.01

        speed = self.base_speed * (self.battery / self.max_battery * 0.5 + 0.5)
        self.x = clamp(self.x + dx * speed, bounds[0] + 1, bounds[1] - 1)
        self.y = clamp(self.y + dy * speed, bounds[0] + 1, bounds[1] - 1)
        self.drain_battery(moving)

    def _local_drift(self, all_agents: List[Agent]) -> Tuple[float, float]:
        scouts = [
            a
            for a in all_agents
            if isinstance(a, Scout) and a.alive and a.task.startswith("painting")
        ]
        if scouts:
            scout = min(scouts, key=lambda s: distance(self.x, self.y, s.x, s.y))
            if distance(self.x, self.y, scout.x, scout.y) <= self.sense_radius * 1.5:
                return normalize(scout.x - self.x, scout.y - self.y)

        angle = random.uniform(0, 2 * math.pi)
        return math.cos(angle) * 0.25, math.sin(angle) * 0.25


def build_swarm(
    num_agents: int,
    bounds: Tuple[float, float],
    scout_ratio: float = 1 / 3,
) -> Tuple[List[Agent], List[Objective]]:
    num_scouts = max(1, int(num_agents * scout_ratio))
    num_workers = num_agents - num_scouts

    agents: List[Agent] = []
    for i in range(num_scouts):
        agents.append(
            Scout(
                i,
                x=random.uniform(bounds[0] + 2, bounds[1] - 2),
                y=random.uniform(bounds[0] + 2, bounds[1] - 2),
            )
        )
    for i in range(num_scouts, num_agents):
        agents.append(
            Worker(
                i,
                x=random.uniform(bounds[0] + 2, bounds[1] - 2),
                y=random.uniform(bounds[0] + 2, bounds[1] - 2),
            )
        )

    objectives = [
        Objective(
            j,
            x=random.uniform(bounds[0] + 5, bounds[1] - 5),
            y=random.uniform(bounds[0] + 5, bounds[1] - 5),
        )
        for j in range(5)
    ]

    return agents, objectives


def run_simulation(
    num_agents: int = 30,
    bounds: Tuple[float, float] = (-40, 40),
    frames: int = 250,
    interval: int = 120,
    save_path: Optional[str] = None,
) -> None:
    agents, objectives = build_swarm(num_agents, bounds)
    scouts = [a for a in agents if isinstance(a, Scout)]
    workers = [a for a in agents if isinstance(a, Worker)]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    scout_dots = ax.plot([], [], "o", color="#ff7b00", markersize=9, label="Scout")[0]
    worker_dots = ax.plot([], [], "o", color="#58a6ff", markersize=8, label="Worker")[0]
    obj_unpainted = ax.plot([], [], "s", color="#6e7681", markersize=10, label="Unpainted objective")[0]
    obj_painted = ax.plot([], [], "s", color="#ffd700", markersize=11, label="Painted objective")[0]

    for agent in agents:
        circle = plt.Circle(
            (agent.x, agent.y),
            agent.sense_radius,
            fill=False,
            linestyle="--",
            linewidth=0.6,
            alpha=0.25,
            color="#ff7b00" if isinstance(agent, Scout) else "#58a6ff",
        )
        ax.add_patch(circle)
        agent._sense_ring = circle  # noqa: SLF001 — viz only

    title = ax.set_title("", color="white", fontsize=11)
    ax.legend(loc="upper right", facecolor="#161b22", labelcolor="white", fontsize=9)
    ax.set_xlabel("Scouts paint objectives; workers follow paint only", color="#8b949e")

    def update(frame: int):
        for agent in agents:
            if isinstance(agent, Scout):
                agent.step(bounds, objectives, agents)
            elif isinstance(agent, Worker):
                agent.step(bounds, objectives, agents)

        scout_x = [s.x for s in scouts if s.alive]
        scout_y = [s.y for s in scouts if s.alive]
        worker_x = [w.x for w in workers if w.alive]
        worker_y = [w.y for w in workers if w.alive]

        scout_dots.set_data(scout_x, scout_y)
        worker_dots.set_data(worker_x, worker_y)

        unpainted_x = [o.x for o in objectives if not o.painted]
        unpainted_y = [o.y for o in objectives if not o.painted]
        painted_x = [o.x for o in objectives if o.painted]
        painted_y = [o.y for o in objectives if o.painted]

        obj_unpainted.set_data(unpainted_x, unpainted_y)
        obj_painted.set_data(painted_x, painted_y)

        for agent in agents:
            agent._sense_ring.center = (agent.x, agent.y)
            agent._sense_ring.set_visible(agent.alive)

        painted_count = sum(1 for o in objectives if o.painted)
        active_scouts = sum(1 for s in scouts if s.alive)
        active_workers = sum(1 for w in workers if w.alive)
        avg_scout_batt = (
            sum(s.battery for s in scouts if s.alive) / active_scouts if active_scouts else 0
        )
        avg_worker_batt = (
            sum(w.battery for w in workers if w.alive) / active_workers if active_workers else 0
        )

        title.set_text(
            f"Heterogeneous Swarm | Frame {frame} | "
            f"Painted: {painted_count}/{len(objectives)} | "
            f"Scouts: {active_scouts} (avg batt {avg_scout_batt:.0f}) | "
            f"Workers: {active_workers} (avg batt {avg_worker_batt:.0f})"
        )

        return scout_dots, worker_dots, obj_unpainted, obj_painted, title

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)

    if save_path:
        ani.save(save_path, fps=max(1, 1000 // interval), dpi=120)
        plt.close(fig)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Heterogeneous scout/worker swarm simulation")
    parser.add_argument("--agents", type=int, default=30, help="Total agents (default 30)")
    parser.add_argument("--frames", type=int, default=250, help="Animation frames")
    parser.add_argument("--interval", type=int, default=120, help="Ms between frames")
    parser.add_argument("--out", type=str, default=None, help="Save animation path (gif/mp4)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        num_agents=args.agents,
        frames=args.frames,
        interval=args.interval,
        save_path=args.out,
    )
