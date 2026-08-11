import math
import random
from typing import Tuple, Dict, Any, List

from src.core.base_agent import BaseAgent

def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)

def normalize(dx: float, dy: float) -> Tuple[float, float]:
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0, 0.0
    return dx / length, dy / length

class ScoutAgent(BaseAgent):
    def __init__(self, agent_id: int, x: float, y: float, params: Dict[str, Any]):
        # Default Scout parameters if not supplied: speed=2.0, sense_radius=18.0, battery=80.0, drain=0.22
        merged_params = {
            "speed": 2.0,
            "sense_radius": 18.0,
            "battery": 80.0,
            "drain": 0.22
        }
        merged_params.update(params)
        super().__init__(agent_id, x, y, merged_params)
        self.species = "scout"
        self.wander_angle = random.uniform(0, 2 * math.pi)

    def decide(self, simulation: Any) -> Tuple[float, float, bool]:
        closest_obj = None
        min_dist = float('inf')
        
        # Scans all objectives
        for obj in simulation.objectives:
            dist = distance(self.x, self.y, obj.x, obj.y)
            if dist <= self.sense_radius and dist < min_dist:
                closest_obj = obj
                min_dist = dist
                
        dx, dy = 0.0, 0.0
        moving = False

        if closest_obj:
            if min_dist < 2.0:
                closest_obj.apply_paint(1.0)
                self.task = f"painting #{closest_obj.obj_id}"
                # Scatter after painting to explore new sectors
                self.wander_angle = random.uniform(0, 2 * math.pi)
                dx = math.cos(self.wander_angle)
                dy = math.sin(self.wander_angle)
                moving = True
            else:
                dx, dy = normalize(closest_obj.x - self.x, closest_obj.y - self.y)
                self.task = f"approaching #{closest_obj.obj_id}"
                moving = True
        else:
            self.task = "exploring"
            # Wander randomly
            self.wander_angle += random.uniform(-0.7, 0.7)
            dx = math.cos(self.wander_angle)
            dy = math.sin(self.wander_angle)
            moving = True

            # Push away from other alive agents to scatter
            for other in simulation.agents:
                if other.agent_id == self.agent_id or not other.alive:
                    continue
                dist = distance(self.x, self.y, other.x, other.y)
                if dist < 4.0 and dist > 0:
                    push = (4.0 - dist) / 4.0
                    dx += (self.x - other.x) / dist * push
                    dy += (self.y - other.y) / dist * push

            dx, dy = normalize(dx, dy)

        self.state = "active"
        return dx, dy, moving

class WorkerAgent(BaseAgent):
    def __init__(self, agent_id: int, x: float, y: float, params: Dict[str, Any]):
        # Default Worker parameters: speed=1.0, sense_radius=7.0, battery=220.0, drain=0.04
        merged_params = {
            "speed": 1.0,
            "sense_radius": 7.0,
            "battery": 220.0,
            "drain": 0.04
        }
        merged_params.update(params)
        super().__init__(agent_id, x, y, merged_params)
        self.species = "worker"
        # Trajectory
        angle = random.uniform(0, 2 * math.pi)
        self.dx = math.cos(angle)
        self.dy = math.sin(angle)

    def decide(self, simulation: Any) -> Tuple[float, float, bool]:
        threshold = simulation.config.simulation.__dict__.get("min_pheromone_threshold", 0.1)

        # Filters painted objectives only
        valid_targets = [
            obj for obj in simulation.objectives 
            if distance(self.x, self.y, obj.x, obj.y) <= self.sense_radius 
            and obj.paint_strength >= threshold
        ]
        
        dx, dy = 0.0, 0.0
        moving = False

        if valid_targets:
            # Exploitation: Prioritize the highest intensity pheromone
            target = max(valid_targets, key=lambda o: o.paint_strength)
            dist = distance(self.x, self.y, target.x, target.y)
            
            if dist > 1.0:
                dx, dy = normalize(target.x - self.x, target.y - self.y)
                self.dx, self.dy = dx, dy
                self.task = f"following paint -> #{target.obj_id}"
                moving = True
            else:
                # Idle state: Preserves energy upon reaching objective
                dx, dy = 0.0, 0.0
                self.dx, self.dy = dx, dy
                self.task = "at objective"
                moving = False
        else:
            # Brownian motion / Foraging drift
            if random.random() < 0.05:
                angle = random.uniform(0, 2 * math.pi)
                dx = math.cos(angle)
                dy = math.sin(angle)
                self.dx, self.dy = dx, dy
            else:
                dx, dy = self.dx, self.dy
            
            # Slow local drift drift force
            dx, dy = normalize(dx, dy)
            dx *= 0.25
            dy *= 0.25
            self.task = "patrolling"
            moving = math.hypot(dx, dy) > 0.01

        self.state = "active"
        return dx, dy, moving
