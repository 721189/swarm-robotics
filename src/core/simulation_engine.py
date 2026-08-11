import random
import math
from typing import List, Dict, Any, Tuple
import numpy as np

from src.config.config import SwarmConfig
from src.core.base_agent import BaseAgent
from src.core.metrics import MetricsLogger

class Objective:
    """Target location that agents might visit, paint, or assign to."""
    def __init__(self, obj_id: int, x: float, y: float):
        self.obj_id = obj_id
        self.x = x
        self.y = y
        self.paint_strength = 0.0
        self.painted = False

    def apply_paint(self, amount: float = 1.0):
        self.painted = True
        self.paint_strength = min(1.0, self.paint_strength + amount)

    def decay(self, rate: float = 0.005):
        if self.paint_strength > 0:
            self.paint_strength -= rate
            if self.paint_strength <= 0:
                self.paint_strength = 0.0
                self.painted = False

class SAMThreat:
    """Represents a Surface-to-Air Missile / Radar engagement bubble."""
    def __init__(self, threat_id: int, x: float, y: float, radius: float, strength: float = 50.0):
        self.threat_id = threat_id
        self.x = x
        self.y = y
        self.radius = radius
        self.strength = strength

class SwarmSimulation:
    """Unified engine for all algorithms"""
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.agents: List[BaseAgent] = []
        self.objectives: List[Objective] = []
        self.threats: List[SAMThreat] = []
        self.metrics_logger = MetricsLogger()
        self.frame = 0
        self.reset()

    def reset(self):
        """Reproducible re-initialization"""
        # Seed both standard libraries
        random.seed(self.config.simulation.seed)
        np.random.seed(self.config.simulation.seed)
        self.frame = 0

        self.agents = []
        self.objectives = []
        self.threats = []

        bounds = self.config.simulation.bounds
        algo = self.config.simulation.algorithm.lower()

        # Instantiate agents based on types defined in config
        agent_id = 0
        for ac in self.config.agents:
            for _ in range(ac.count):
                x = random.uniform(bounds[0] + 2.0, bounds[1] - 2.0)
                y = random.uniform(bounds[0] + 2.0, bounds[1] - 2.0)
                agent = self._create_agent(algo, ac.name, agent_id, x, y, ac.params)
                self.agents.append(agent)
                agent_id += 1

        # Instantiate objectives
        obj_count = self.config.objectives.count
        positions = self.config.objectives.positions
        for i in range(obj_count):
            if i < len(positions):
                x, y = positions[i]
            else:
                x = random.uniform(bounds[0] + 5.0, bounds[1] - 5.0)
                y = random.uniform(bounds[0] + 5.0, bounds[1] - 5.0)
            self.objectives.append(Objective(i, x, y))

        # Instantiate threats
        for tc in self.config.threats:
            self.threats.append(SAMThreat(tc.id, tc.x, tc.y, tc.radius, tc.strength))

        self.metrics_logger.reset()
        # Log frame 0
        self.metrics_logger.log_frame(self.frame, self.agents, self.objectives)
        self.frame += 1

    def _create_agent(self, algo: str, type_name: str, agent_id: int, x: float, y: float, params: Dict[str, Any]) -> BaseAgent:
        if algo == "reynolds":
            from src.algorithms.reynolds import ReynoldsAgent
            return ReynoldsAgent(agent_id, x, y, params)
        elif algo == "stigmergy":
            from src.algorithms.stigmergy import ScoutAgent, WorkerAgent
            if type_name.lower() == "scout":
                return ScoutAgent(agent_id, x, y, params)
            else:
                return WorkerAgent(agent_id, x, y, params)
        elif algo == "cbba":
            from src.algorithms.cbba import CombatDroneAgent
            return CombatDroneAgent(agent_id, x, y, params)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    def step(self, dt: float):
        """Single simulation tick - decoupled from rendering"""
        bounds = self.config.simulation.bounds
        algo = self.config.simulation.algorithm.lower()

        # 1. Decay environment stigmergy markers (Pheromone/Paint)
        if algo == "stigmergy":
            # Let decay rate be config-driven if present
            decay_rate = self.config.simulation.__dict__.get("decay_rate", 0.005)
            for obj in self.objectives:
                obj.decay(decay_rate)

        # 2. Gossip consensus for CBBA before bidding phase
        if algo == "cbba":
            # Fetch communication range from first agent or config
            comm_range = 25.0
            if self.agents:
                comm_range = self.agents[0].sense_radius

            # Gossip Consensus
            for agent in self.agents:
                if not agent.alive:
                    continue
                # Get neighbors within radio range
                neighbors = []
                for other in self.agents:
                    if other.agent_id != agent.agent_id and other.alive:
                        dist = math.hypot(agent.x - other.x, agent.y - other.y)
                        if dist <= comm_range:
                            neighbors.append(other)
                agent.sync_ledgers(neighbors)

            # Local Re-Bidding Calculation
            for agent in self.agents:
                if not agent.alive:
                    continue
                agent.run_local_auction(self.objectives)

        # 3. Dynamic steering decision calculations
        agent_decisions = []
        for agent in self.agents:
            if agent.alive:
                dx, dy, moving = agent.decide(self)
                agent_decisions.append((agent, dx, dy, moving))
            else:
                agent_decisions.append((agent, 0.0, 0.0, False))

        # 4. Execute physical kinematics & battery consumption
        for agent, dx, dy, moving in agent_decisions:
            agent.update_state(dx, dy, moving, bounds, dt)

        # 5. Stigmergy actions post-step (Scouts painting Objectives)
        if algo == "stigmergy":
            for agent in self.agents:
                if agent.alive and getattr(agent, "species", "") == "scout":
                    for obj in self.objectives:
                        dist = math.hypot(agent.x - obj.x, agent.y - obj.y)
                        if dist <= 2.0:
                            obj.apply_paint(1.0)

        # Log metrics
        self.metrics_logger.log_frame(self.frame, self.agents, self.objectives)
        self.frame += 1

    def to_state_dict(self) -> Dict[str, Any]:
        """Checkpoint for analysis"""
        return {
            "frame": self.frame,
            "agents": [a.to_dict() for a in self.agents],
            "objectives": [(o.obj_id, o.x, o.y, o.paint_strength) for o in self.objectives],
            "threats": [(t.threat_id, t.x, t.y, t.radius, t.strength) for t in self.threats],
            "config": {
                "algorithm": self.config.simulation.algorithm,
                "bounds": self.config.simulation.bounds
            }
        }
