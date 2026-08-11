from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseAgent(ABC):
    def __init__(self, agent_id: int, x: float, y: float, params: Dict[str, Any]):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.params = params
        self.state = "comfortable"
        self.task = "idle"
        self.battery = params.get("battery", 100.0)
        self.max_battery = params.get("battery", 100.0)
        self.drain_rate = params.get("drain", 0.05)
        self.speed = params.get("speed", 1.5)
        self.sense_radius = params.get("sense_radius", 5.0)
        self.alive = True

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @abstractmethod
    def decide(self, simulation: Any) -> Tuple[float, float, bool]:
        """
        Determine movement vector and state.
        Returns:
            dx (float): Target direction X (normalized/steering)
            dy (float): Target direction Y (normalized/steering)
            moving (bool): True if the agent is actively moving, False otherwise
        """
        pass

    def update_state(self, dx: float, dy: float, moving: bool, bounds: Tuple[float, float], dt: float):
        if not self.alive:
            return
            
        # Update positions
        self.x += dx * self.speed
        self.y += dy * self.speed
        
        # Bounding limits clamp
        self.x = max(bounds[0] + 1.0, min(bounds[1] - 1.0, self.x))
        self.y = max(bounds[0] + 1.0, min(bounds[1] - 1.0, self.y))
        
        # Battery drain logic
        cost = self.drain_rate if moving else (self.drain_rate / 4.0) # idle drain
        self.battery = max(0.0, self.battery - cost)
        if self.battery <= 0.0:
            self.alive = False
            self.state = "depleted"
            self.task = "depleted"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "x": self.x,
            "y": self.y,
            "state": self.state,
            "task": self.task,
            "battery": self.battery,
            "alive": self.alive,
            "species": getattr(self, "species", "base")
        }
