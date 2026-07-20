from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import yaml
import json
import os

@dataclass
class AgentConfig:
    name: str
    count: int
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThreatConfig:
    id: int
    x: float
    y: float
    radius: float
    strength: float

@dataclass
class ObjectiveConfig:
    count: int
    positions: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class SimParamsConfig:
    seed: int = 42
    bounds: Tuple[float, float] = (-50.0, 50.0)
    dt: float = 0.1
    max_frames: int = 300
    algorithm: str = "reynolds"

@dataclass
class SwarmConfig:
    simulation: SimParamsConfig
    agents: List[AgentConfig]
    objectives: ObjectiveConfig
    threats: List[ThreatConfig] = field(default_factory=list)

    @classmethod
    def from_file(cls, filepath: str) -> 'SwarmConfig':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            if filepath.endswith(('.yaml', '.yml')):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        sim_data = data.get('simulation', {})
        sim = SimParamsConfig(
            seed=sim_data.get('seed', 42),
            bounds=tuple(sim_data.get('bounds', [-50.0, 50.0])),
            dt=sim_data.get('dt', 0.1),
            max_frames=sim_data.get('max_frames', 300),
            algorithm=sim_data.get('algorithm', 'reynolds')
        )
        
        agents = []
        agents_section = data.get('agents', {})
        types = agents_section.get('types', [])
        for t in types:
            agents.append(AgentConfig(
                name=t.get('name', 'base'),
                count=t.get('count', 0),
                params=t.get('params', {})
            ))
            
        obj_section = data.get('objectives', {})
        objs = ObjectiveConfig(
            count=obj_section.get('count', 0),
            positions=[tuple(p) for p in obj_section.get('positions', [])]
        )
        
        threats = []
        threats_section = data.get('threats', [])
        for t in threats_section:
            threats.append(ThreatConfig(
                id=t.get('id'),
                x=t.get('x'),
                y=t.get('y'),
                radius=t.get('radius'),
                strength=t.get('strength')
            ))
            
        return cls(simulation=sim, agents=agents, objectives=objs, threats=threats)
