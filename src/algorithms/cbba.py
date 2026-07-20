import math
import random
from typing import Tuple, Dict, Any, List

from src.core.base_agent import BaseAgent

def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)

def normalize(dx: float, dy: float) -> Tuple[float, float]:
    mag = math.hypot(dx, dy)
    if mag == 0:
        return 0.0, 0.0
    return dx / mag, dy / mag

class CombatDroneAgent(BaseAgent):
    def __init__(self, agent_id: int, x: float, y: float, params: Dict[str, Any]):
        # Default CombatDrone parameters: speed=1.5, max_force=0.2, sense_radius=25.0 (comm range), battery=1000.0, drain=0.0
        merged_params = {
            "speed": 1.5,
            "max_force": 0.2,
            "sense_radius": 25.0,
            "battery": 1000.0,
            "drain": 0.0
        }
        merged_params.update(params)
        super().__init__(agent_id, x, y, merged_params)
        self.species = "combat_drone"
        self.max_force = merged_params["max_force"]
        
        # Init random velocity
        self.vx = random.uniform(-1.0, 1.0)
        self.vy = random.uniform(-1.0, 1.0)
        self.vx, self.vy = normalize(self.vx, self.vy)

        # Local auction ledgers (CBBA Style)
        self.assigned_task_id = None
        self.local_winners = {}  # Maps Obj ID -> Winning Drone ID
        self.local_bids = {}     # Maps Obj ID -> Highest Known Bid

    def calculate_local_bid(self, objective) -> float:
        dist = distance(self.x, self.y, objective.x, objective.y)
        return max(0.1, 100.0 - dist)

    def sync_ledgers(self, neighbors: List['CombatDroneAgent']):
        for neighbor in neighbors:
            for obj_id, bid in neighbor.local_bids.items():
                # Ensure we have this objective in our local ledger
                if obj_id not in self.local_bids:
                    self.local_bids[obj_id] = 0.0
                    self.local_winners[obj_id] = -1

                # If neighbor knows a higher bid, update our local ledger
                if bid > self.local_bids[obj_id]:
                    self.local_bids[obj_id] = bid
                    self.local_winners[obj_id] = neighbor.local_winners[obj_id]
                    
                    # Outbid gating: if I got outbid on my active task, drop it
                    if self.assigned_task_id == obj_id and self.local_winners[obj_id] != self.agent_id:
                        self.assigned_task_id = None

    def run_local_auction(self, objectives):
        # Init ledgers for any new objectives if not already there
        for obj in objectives:
            if obj.obj_id not in self.local_bids:
                self.local_bids[obj.obj_id] = 0.0
                self.local_winners[obj.obj_id] = -1

        best_target_id = None
        highest_net_gain = -1.0

        for obj in objectives:
            my_bid = self.calculate_local_bid(obj)
            if my_bid > self.local_bids[obj.obj_id]:
                gain = my_bid - self.local_bids[obj.obj_id]
                if gain > highest_net_gain:
                    highest_net_gain = gain
                    best_target_id = obj.obj_id

        # Claim the target locally if it provides gain
        if best_target_id is not None:
            # Release previous target ownership locally
            if self.assigned_task_id is not None and self.assigned_task_id != best_target_id:
                self.local_bids[self.assigned_task_id] = 0.0
                self.local_winners[self.assigned_task_id] = -1
            
            self.assigned_task_id = best_target_id
            target_obj = next(o for o in objectives if o.obj_id == best_target_id)
            self.local_bids[best_target_id] = self.calculate_local_bid(target_obj)
            self.local_winners[best_target_id] = self.agent_id

    def decide(self, simulation: Any) -> Tuple[float, float, bool]:
        # Force weights
        attr_w = self.params.get("attr_weight", 1.0)
        rep_w = self.params.get("sam_repulsion_weight", 4.5)
        sep_w = self.params.get("boids_separation_weight", 1.5)

        fx, fy = 0.0, 0.0

        # 1. Attractive Potential Field towards assigned objective
        if self.assigned_task_id is not None:
            target = next(o for o in simulation.objectives if o.obj_id == self.assigned_task_id)
            tax, tay = normalize(target.x - self.x, target.y - self.y)
            fx += tax * attr_w
            fy += tay * attr_w
            self.task = f"assigned #{self.assigned_task_id}"
        else:
            # Brownian idle drift
            fx += self.vx * 0.2
            fy += self.vy * 0.2
            self.task = "idle search"

        # 2. Repulsive Potential Field from SAM threats
        for threat in simulation.threats:
            dist_to_threat = distance(self.x, self.y, threat.x, threat.y)
            if dist_to_threat < threat.radius:
                dist_to_threat = max(dist_to_threat, 1.0)
                repulsion_mag = threat.strength * ((1.0 / dist_to_threat) - (1.0 / threat.radius))**2
                trx, try_ = normalize(self.x - threat.x, self.y - threat.y)
                fx += trx * repulsion_mag * rep_w
                fy += try_ * repulsion_mag * rep_w

        # 3. Reynolds separation to avoid drone collisions
        sep_x, sep_y = 0.0, 0.0
        for other in simulation.agents:
            if other.agent_id != self.agent_id and other.alive:
                d = distance(self.x, self.y, other.x, other.y)
                if d < 4.0:
                    d = max(d, 0.5)
                    dx, dy = normalize(self.x - other.x, self.y - other.y)
                    sep_x += dx / d
                    sep_y += dy / d
        
        fx += sep_x * sep_w
        fy += sep_y * sep_w

        # Steering limits
        fx, fy = normalize(fx, fy)
        self.vx += fx * self.max_force
        self.vy += fy * self.max_force
        self.vx, self.vy = normalize(self.vx, self.vy)

        # Color/state logic: Orange if task assigned, Blue if searching
        if self.assigned_task_id is not None:
            self.state = "comfortable" # maps to gold/orange in viz
        else:
            self.state = "lonely" # maps to blue in viz

        # Drone is always active/moving
        return self.vx, self.vy, True
