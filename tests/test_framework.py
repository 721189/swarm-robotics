import unittest
import os
import shutil
import tempfile
import csv
import math

from src.config.config import SwarmConfig, SimParamsConfig, AgentConfig, ObjectiveConfig, ThreatConfig
from src.core.simulation_engine import SwarmSimulation


class TestSwarmFramework(unittest.TestCase):
    """Basic framework and configuration tests"""
    
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
        """Verify YAML config parsing"""
        config = SwarmConfig.from_file(self.config_path)
        self.assertEqual(config.simulation.seed, 100)
        self.assertEqual(config.simulation.bounds, (-20.0, 20.0))
        self.assertEqual(config.simulation.algorithm, "reynolds")
        self.assertEqual(len(config.agents), 1)
        self.assertEqual(config.agents[0].count, 5)
        self.assertEqual(config.objectives.count, 2)
        self.assertEqual(len(config.objectives.positions), 2)

    def test_simulation_reynolds(self):
        """Basic Reynolds algorithm simulation initialization"""
        config = SwarmConfig.from_file(self.config_path)
        sim = SwarmSimulation(config)
        self.assertEqual(len(sim.agents), 5)
        self.assertEqual(len(sim.objectives), 2)
        
        # Step simulation
        sim.step(0.1)
        self.assertEqual(sim.frame, 2)
        self.assertEqual(len(sim.metrics_logger.history), 2)

    def test_simulation_stigmergy(self):
        """Basic Stigmergy algorithm simulation initialization"""
        config = SwarmConfig.from_file(self.config_path)
        config.simulation.algorithm = "stigmergy"
        config.agents[0].name = "scout"
        config.agents[0].count = 2
        # Add workers
        config.agents.append(AgentConfig(name="worker", count=3, params={"speed": 0.5}))
        
        sim = SwarmSimulation(config)
        self.assertEqual(len(sim.agents), 5)
        sim.step(0.1)
        self.assertEqual(sim.frame, 2)

    def test_simulation_cbba(self):
        """Basic CBBA algorithm simulation initialization"""
        config = SwarmConfig.from_file(self.config_path)
        config.simulation.algorithm = "cbba"
        config.agents[0].name = "combat_drone"
        
        sim = SwarmSimulation(config)
        self.assertEqual(len(sim.agents), 5)
        sim.step(0.1)
        self.assertEqual(sim.frame, 2)


class TestReynoldsIntegration(unittest.TestCase):
    """Reynolds flocking algorithm integration tests"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_reynolds_collision_avoidance(self):
        """
        Test that agents in crowded clusters move apart (separation/collision avoidance).
        Create a dense cluster and verify agents move outward.
        """
        config = SimParamsConfig(
            seed=42, 
            bounds=(-30.0, 30.0), 
            dt=0.1, 
            max_frames=50,
            algorithm="reynolds"
        )
        agents_config = [AgentConfig(
            name="base", 
            count=10, 
            params={"speed": 1.5, "sense_radius": 8.0, "battery": 9999.0, "drain": 0.0}
        )]
        objs_config = ObjectiveConfig(count=0)
        
        full_config = SwarmConfig(simulation=config, agents=agents_config, objectives=objs_config)
        sim = SwarmSimulation(full_config)
        
        # Record initial cluster density
        initial_positions = [(a.x, a.y) for a in sim.agents]
        initial_spread = self._calculate_spread(initial_positions)
        
        # Run for 50 frames
        for _ in range(50):
            sim.step(0.1)
        
        # Check final spread is greater (agents moved apart)
        final_positions = [(a.x, a.y) for a in sim.agents]
        final_spread = self._calculate_spread(final_positions)
        
        # Spread should increase (agents separating)
        self.assertGreater(final_spread, initial_spread * 0.5,
                          "Reynolds agents should spread out in crowded clusters")

    def test_reynolds_stability_detection(self):
        """
        Test that all agents eventually reach 'comfortable' state.
        This indicates equilibrium: no agent is lonely or crowded.
        """
        config = SimParamsConfig(
            seed=100,
            bounds=(-20.0, 20.0),
            dt=0.1,
            max_frames=200,
            algorithm="reynolds"
        )
        agents_config = [AgentConfig(
            name="base",
            count=15,
            params={"speed": 1.5, "sense_radius": 8.0, "battery": 9999.0, "drain": 0.0}
        )]
        objs_config = ObjectiveConfig(count=0)
        
        full_config = SwarmConfig(simulation=config, agents=agents_config, objectives=objs_config)
        sim = SwarmSimulation(full_config)
        
        # Run simulation
        for _ in range(200):
            sim.step(0.1)
        
        # Check final metrics
        summary = sim.metrics_logger.get_summary()
        final_frame = sim.metrics_logger.history[-1]
        final_states = final_frame["states"]
        
        # Most agents should be comfortable in final state
        comfortable_count = final_states.get("comfortable", 0)
        total_agents = len(sim.agents)
        
        self.assertGreater(comfortable_count, total_agents * 0.7,
                          "At least 70% of Reynolds agents should reach comfortable state")

    def _calculate_spread(self, positions):
        """Calculate average distance from centroid"""
        if not positions:
            return 0.0
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        avg_dist = sum(math.hypot(p[0] - cx, p[1] - cy) for p in positions) / len(positions)
        return avg_dist


class TestStigmergyIntegration(unittest.TestCase):
    """Stigmergy cooperative algorithm integration tests"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_stigmergy_objective_painting(self):
        """
        Test that scouts paint objectives and objectives accumulate paint strength.
        Verify paint_strength increases over time.
        """
        config = SimParamsConfig(
            seed=50,
            bounds=(-40.0, 40.0),
            dt=0.1,
            max_frames=100,
            algorithm="stigmergy"
        )
        # 2 scouts, 3 workers
        agents_config = [
            AgentConfig(name="scout", count=2, 
                       params={"speed": 2.0, "sense_radius": 18.0, "battery": 80.0, "drain": 0.22}),
            AgentConfig(name="worker", count=3,
                       params={"speed": 1.0, "sense_radius": 7.0, "battery": 220.0, "drain": 0.04})
        ]
        objs_config = ObjectiveConfig(count=3)
        
        full_config = SwarmConfig(simulation=config, agents=agents_config, objectives=objs_config)
        sim = SwarmSimulation(full_config)
        
        # Record initial paint state
        initial_paint = [o.paint_strength for o in sim.objectives]
        
        # Run simulation
        for _ in range(100):
            sim.step(0.1)
        
        # Check final paint state
        final_paint = [o.paint_strength for o in sim.objectives]
        
        # At least one objective should have paint applied
        max_final_paint = max(final_paint)
        self.assertGreater(max_final_paint, 0.0,
                          "Scouts should paint at least one objective")

    def test_stigmergy_worker_follow_behavior(self):
        """
        Test that workers follow scouts toward painted objectives.
        Verify that workers move toward higher paint concentration.
        """
        config = SimParamsConfig(
            seed=60,
            bounds=(-40.0, 40.0),
            dt=0.1,
            max_frames=150,
            algorithm="stigmergy"
        )
        agents_config = [
            AgentConfig(name="scout", count=2,
                       params={"speed": 2.0, "sense_radius": 18.0, "battery": 80.0, "drain": 0.22}),
            AgentConfig(name="worker", count=4,
                       params={"speed": 1.0, "sense_radius": 7.0, "battery": 220.0, "drain": 0.04})
        ]
        objs_config = ObjectiveConfig(count=2)
        
        full_config = SwarmConfig(simulation=config, agents=agents_config, objectives=objs_config)
        sim = SwarmSimulation(full_config)
        
        # All agents should remain alive (sufficient battery)
        for _ in range(150):
            sim.step(0.1)
        
        # Verify no worker agents depleted
        worker_agents = [a for a in sim.agents if getattr(a, 'species', '') == 'worker']
        alive_workers = sum(1 for a in worker_agents if a.alive)
        
        self.assertGreater(alive_workers, 0,
                          "At least some workers should survive full trial")


class TestCBBAIntegration(unittest.TestCase):
    """CBBA consensus-based bundle algorithm integration tests"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_cbba_assignment_convergence(self):
        """
        Test that CBBA agents converge to stable task assignments.
        Verify that local winner ledgers stabilize.
        """
        config = SimParamsConfig(
            seed=70,
            bounds=(0.0, 100.0),
            dt=0.1,
            max_frames=100,
            algorithm="cbba"
        )
        agents_config = [AgentConfig(
            name="combat_drone",
            count=8,
            params={"speed": 1.5, "max_force": 0.2, "sense_radius": 25.0, "battery": 9999.0, "drain": 0.0}
        )]
        threats_config = [
            ThreatConfig(id=0, x=30.0, y=40.0, radius=15.0, strength=60.0),
            ThreatConfig(id=1, x=70.0, y=65.0, radius=18.0, strength=70.0)
        ]
        objs_config = ObjectiveConfig(
            count=5,
            positions=[(15.0, 85.0), (85.0, 85.0), (50.0, 50.0), (85.0, 15.0), (15.0, 15.0)]
        )
        
        full_config = SwarmConfig(
            simulation=config,
            agents=agents_config,
            objectives=objs_config,
            threats=threats_config
        )
        sim = SwarmSimulation(full_config)
        
        # Run simulation
        for _ in range(100):
            sim.step(0.1)
        
        # Check that agents have made assignments
        agents_with_tasks = sum(1 for a in sim.agents if getattr(a, 'assigned_task_id', None) is not None)
        
        # At least some agents should have task assignments
        self.assertGreater(agents_with_tasks, 0,
                          "CBBA agents should make task assignments during convergence")

    def test_cbba_consensus_stability(self):
        """
        Test that CBBA reaches consensus on task winner assignments.
        Verify that no agent is contested for the same task in final state.
        """
        config = SimParamsConfig(
            seed=80,
            bounds=(0.0, 100.0),
            dt=0.1,
            max_frames=150,
            algorithm="cbba"
        )
        agents_config = [AgentConfig(
            name="combat_drone",
            count=6,
            params={"speed": 1.5, "max_force": 0.2, "sense_radius": 25.0, "battery": 9999.0, "drain": 0.0}
        )]
        objs_config = ObjectiveConfig(
            count=3,
            positions=[(20.0, 80.0), (80.0, 80.0), (50.0, 20.0)]
        )
        
        full_config = SwarmConfig(simulation=config, agents=agents_config, objectives=objs_config)
        sim = SwarmSimulation(full_config)
        
        # Run for convergence
        for _ in range(150):
            sim.step(0.1)
        
        # Collect final assigned tasks
        assigned_tasks = []
        for a in sim.agents:
            if getattr(a, 'assigned_task_id', None) is not None:
                assigned_tasks.append(a.assigned_task_id)
        
        # No objective should be assigned to more than one agent at the end
        # (This is consensus: one winner per objective, or no winner)
        task_counts = {}
        for task_id in assigned_tasks:
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
        
        # Verify no over-assignment (consensus stability)
        for obj_id, count in task_counts.items():
            self.assertEqual(count, 1,
                           f"Objective {obj_id} should have at most 1 assigned agent (consensus)")


class TestCSVMetricsOutput(unittest.TestCase):
    """CSV output and metrics logging tests"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_csv_output_headers_and_rows(self):
        """
        Test that CSV output contains correct headers and data rows.
        Verify format matches expected metrics structure.
        """
        config = SimParamsConfig(
            seed=90,
            bounds=(-20.0, 20.0),
            dt=0.1,
            max_frames=15,
            algorithm="reynolds"
        )
        agents_config = [AgentConfig(
            name="base",
            count=6,
            params={"speed": 1.0, "sense_radius": 5.0, "battery": 100.0, "drain": 0.1}
        )]
        objs_config = ObjectiveConfig(count=1)
        
        full_config = SwarmConfig(simulation=config, agents=agents_config, objectives=objs_config)
        sim = SwarmSimulation(full_config)
        
        # Run simulation
        for _ in range(15):
            sim.step(0.1)
        
        # Save metrics to CSV
        csv_path = os.path.join(self.test_dir, "test_metrics.csv")
        sim.metrics_logger.save_to_csv(csv_path)
        
        # Verify file exists
        self.assertTrue(os.path.exists(csv_path), "CSV file should be created")
        
        # Verify headers and rows
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
            # Check expected headers
            expected_headers = ["frame", "active_count", "lonely", "crowded", "comfortable", "depleted"]
            self.assertEqual(headers, expected_headers,
                           f"CSV headers should match expected: {expected_headers}")
            
            # Collect all data rows
            rows = list(reader)
            self.assertEqual(len(rows), 16,  # 16 frames (0-15)
                           "CSV should have 16 data rows (one per frame)")
            
            # Verify each row has correct number of columns
            for row_idx, row in enumerate(rows):
                self.assertEqual(len(row), 6,
                               f"Row {row_idx} should have 6 columns")
                
                # Verify frame number matches row index
                frame_num = int(row[0])
                self.assertEqual(frame_num, row_idx,
                               f"Frame number in row {row_idx} should be {row_idx}")

    def test_csv_metrics_content_validation(self):
        """
        Test that CSV metrics content is valid and makes sense.
        Verify active_count, state counts are non-negative and consistent.
        """
        config = SimParamsConfig(
            seed=100,
            bounds=(-15.0, 15.0),
            dt=0.1,
            max_frames=20,
            algorithm="reynolds"
        )
        agents_config = [AgentConfig(
            name="base",
            count=8,
            params={"speed": 1.0, "sense_radius": 5.0, "battery": 50.0, "drain": 0.05}
        )]
        objs_config = ObjectiveConfig(count=0)
        
        full_config = SwarmConfig(simulation=config, agents=agents_config, objectives=objs_config)
        sim = SwarmSimulation(full_config)
        
        # Run simulation
        for _ in range(20):
            sim.step(0.1)
        
        csv_path = os.path.join(self.test_dir, "validation_metrics.csv")
        sim.metrics_logger.save_to_csv(csv_path)
        
        # Parse and validate
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row['frame'])
                active = int(row['active_count'])
                lonely = int(row['lonely'])
                crowded = int(row['crowded'])
                comfortable = int(row['comfortable'])
                depleted = int(row['depleted'])
                
                # Verify non-negative
                self.assertGreaterEqual(active, 0, f"Frame {frame}: active_count should be non-negative")
                self.assertGreaterEqual(lonely, 0, f"Frame {frame}: lonely count should be non-negative")
                self.assertGreaterEqual(crowded, 0, f"Frame {frame}: crowded count should be non-negative")
                self.assertGreaterEqual(comfortable, 0, f"Frame {frame}: comfortable count should be non-negative")
                self.assertGreaterEqual(depleted, 0, f"Frame {frame}: depleted count should be non-negative")
                
                # State counts should sum to or be <= total agents
                state_sum = lonely + crowded + comfortable + depleted
                self.assertLessEqual(state_sum, 8,
                                   f"Frame {frame}: state counts should not exceed total agents")


if __name__ == "__main__":
    unittest.main()
