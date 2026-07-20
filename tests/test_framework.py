import unittest
import os
import shutil
import tempfile

from src.config.config import SwarmConfig
from src.core.simulation_engine import SwarmSimulation

class TestSwarmFramework(unittest.TestCase):
    def setUp(self):
        # Create temp folder for files
        self.test_dir = tempfile.mkdtemp()
        self.config_content = """
simulation:
  seed: 100
  bounds: [-20.0, 20.0]
  dt: 0.1
  max_frames: 10
  algorithm: "reynolds"

agents:
  types:
    - name: "base"
      count: 5
      params:
        speed: 1.0
        sense_radius: 5.0
        battery: 100.0
        drain: 0.1

objectives:
  count: 2
  positions: [[5.0, 5.0], [-5.0, -5.0]]

threats: []
"""
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        with open(self.config_path, "w") as f:
            f.write(self.config_content)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_config_parsing(self):
        config = SwarmConfig.from_file(self.config_path)
        self.assertEqual(config.simulation.seed, 100)
        self.assertEqual(config.simulation.bounds, (-20.0, 20.0))
        self.assertEqual(config.simulation.algorithm, "reynolds")
        self.assertEqual(len(config.agents), 1)
        self.assertEqual(config.agents[0].count, 5)
        self.assertEqual(config.objectives.count, 2)
        self.assertEqual(len(config.objectives.positions), 2)

    def test_simulation_reynolds(self):
        config = SwarmConfig.from_file(self.config_path)
        sim = SwarmSimulation(config)
        self.assertEqual(len(sim.agents), 5)
        self.assertEqual(len(sim.objectives), 2)
        
        # Step simulation
        sim.step(0.1)
        self.assertEqual(sim.frame, 2) # frame 0 is logged at reset, frame 1 after step, wait frame counter starts at 0, after reset logged frame=0 and frame set to 1. After step, logs frame 1 and sets frame to 2. Correct.
        self.assertEqual(len(sim.metrics_logger.history), 2)

    def test_simulation_stigmergy(self):
        config = SwarmConfig.from_file(self.config_path)
        config.simulation.algorithm = "stigmergy"
        config.agents[0].name = "scout"
        config.agents[0].count = 2
        # Let's add workers too
        from src.config.config import AgentConfig
        config.agents.append(AgentConfig(name="worker", count=3, params={"speed": 0.5}))
        
        sim = SwarmSimulation(config)
        self.assertEqual(len(sim.agents), 5)
        sim.step(0.1)
        self.assertEqual(sim.frame, 2)

    def test_simulation_cbba(self):
        config = SwarmConfig.from_file(self.config_path)
        config.simulation.algorithm = "cbba"
        config.agents[0].name = "combat_drone"
        
        sim = SwarmSimulation(config)
        self.assertEqual(len(sim.agents), 5)
        sim.step(0.1)
        self.assertEqual(sim.frame, 2)

if __name__ == "__main__":
    unittest.main()
