import math
import random
from typing import Tuple, Dict, Any

from src.core.base_agent import BaseAgent

class ReynoldsAgent(BaseAgent):
    def __init__(self, agent_id: int, x: float, y: float, params: Dict[str, Any]):
        super().__init__(agent_id, x, y, params)
        self.species = "reynolds"

    def decide(self, simulation: Any) -> Tuple[float, float, bool]:
        nearby = []
        for other in simulation.agents:
            if other.agent_id == self.agent_id or not other.alive:
                continue
            distance = math.hypot(self.x - other.x, self.y - other.y)
            if distance <= self.sense_radius:
                nearby.append(other)

        dx = dy = 0.0
        moving = False
        
        # Sense criteria matching voronoi_demo.py
        crowded_threshold = self.params.get("crowded_threshold", 4)

        if len(nearby) == 0:
            dx = -self.x
            dy = -self.y
            self.state = "lonely"
            moving = True
        elif len(nearby) >= crowded_threshold:
            avg_x = sum(r.x for r in nearby) / len(nearby)
            avg_y = sum(r.y for r in nearby) / len(nearby)
            dx = self.x - avg_x
            dy = self.y - avg_y
            self.state = "crowded"
            moving = True
        else:
            self.state = "comfortable"
            moving = False

        # Normalize steering vector
        length = math.hypot(dx, dy)
        if length > 0:
            dx /= length
            dy /= length

        return dx, dy, moving
