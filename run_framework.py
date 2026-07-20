import argparse
import sys
import os

from src.config.config import SwarmConfig, SimParamsConfig, AgentConfig, ObjectiveConfig, ThreatConfig
from src.core.simulation_engine import SwarmSimulation
from src.visualization.renderer import AsyncRenderer

def build_default_config(algo: str, num_robots: int, frames: int, seed: int) -> SwarmConfig:
    algo = algo.lower()
    if algo == "reynolds":
        sim = SimParamsConfig(seed=seed, bounds=(-30.0, 30.0), dt=0.1, max_frames=frames, algorithm="reynolds")
        agents = [AgentConfig(name="base", count=num_robots, params={"speed": 1.5, "sense_radius": 8.0, "battery": 9999.0, "drain": 0.0})]
        objs = ObjectiveConfig(count=0)
        return SwarmConfig(simulation=sim, agents=agents, objectives=objs)
        
    elif algo == "stigmergy":
        sim = SimParamsConfig(seed=seed, bounds=(-40.0, 40.0), dt=0.1, max_frames=frames, algorithm="stigmergy")
        scouts = max(1, int(num_robots * (1/3)))
        workers = num_robots - scouts
        agents = [
            AgentConfig(name="scout", count=scouts, params={"speed": 2.0, "sense_radius": 18.0, "battery": 80.0, "drain": 0.22}),
            AgentConfig(name="worker", count=workers, params={"speed": 1.0, "sense_radius": 7.0, "battery": 220.0, "drain": 0.04})
        ]
        objs = ObjectiveConfig(count=5)
        return SwarmConfig(simulation=sim, agents=agents, objectives=objs)
        
    elif algo == "cbba":
        sim = SimParamsConfig(seed=seed, bounds=(0.0, 100.0), dt=0.1, max_frames=frames, algorithm="cbba")
        agents = [AgentConfig(name="combat_drone", count=num_robots, params={"speed": 1.5, "max_force": 0.2, "sense_radius": 25.0, "battery": 9999.0, "drain": 0.0})]
        
        # SAM Threats from original simulation
        threats = [
            ThreatConfig(id=0, x=30.0, y=40.0, radius=15.0, strength=60.0),
            ThreatConfig(id=1, x=70.0, y=65.0, radius=18.0, strength=70.0),
            ThreatConfig(id=2, x=50.0, y=20.0, radius=12.0, strength=50.0)
        ]
        # Specific target positions
        objs = ObjectiveConfig(count=5, positions=[
            (15.0, 85.0),
            (85.0, 85.0),
            (50.0, 50.0),
            (85.0, 15.0),
            (15.0, 15.0)
        ])
        return SwarmConfig(simulation=sim, agents=agents, objectives=objs, threats=threats)
    else:
        raise ValueError(f"Unknown default setup for algorithm: {algo}")

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Swarm Robotics Simulation CLI Framework")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (yaml/json)")
    parser.add_argument("--algo", type=str, default="reynolds", choices=["reynolds", "stigmergy", "cbba"], help="Swarm algorithm model")
    parser.add_argument("--robots", type=int, default=30, help="Number of robots (overrides default/config)")
    parser.add_argument("--frames", type=int, default=200, help="Number of animation frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replication")
    parser.add_argument("--headless", action="store_true", help="Run simulation headlessly (no visual UI, stats logging only)")
    parser.add_argument("--out", type=str, default=None, help="Path to save animation file (e.g. simulation.gif)")
    parser.add_argument("--csv", type=str, default=None, help="Path to save metrics log data to CSV")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = SwarmConfig.from_file(args.config)
        # Apply CLI overrides if specifically requested
        if args.algo:
            config.simulation.algorithm = args.algo
    else:
        print(f"Using default configurations for algorithm: {args.algo}...")
        config = build_default_config(args.algo, args.robots, args.frames, args.seed)

    print("Initialising Swarm Simulation Engine...")
    sim = SwarmSimulation(config)

    # Logging setup if path requested
    csv_path = args.csv

    if args.headless:
        print(f"Running simulation headlessly for {config.simulation.max_frames} ticks...")
        dt = config.simulation.dt
        for _ in range(config.simulation.max_frames):
            sim.step(dt)
        print("Headless run completed successfully.")
    else:
        print("Launching rendering module...")
        renderer = AsyncRenderer(sim, interval=100)
        renderer.start(save_path=args.out)

    # Save metrics to CSV if requested
    if csv_path:
        print(f"Saving frame metrics logging data to {csv_path}...")
        sim.metrics_logger.save_to_csv(csv_path)
        
    summary = sim.metrics_logger.get_summary()
    print("--- Simulation Executed Summary ---")
    print(f"Algorithm Run: {config.simulation.algorithm}")
    print(f"Total Ticks: {summary.get('total_frames')}")
    print(f"Stability Frame: {summary.get('stability_frame')}")
    print(f"Final Active Agents: {summary.get('final_active_count')}")

if __name__ == "__main__":
    main()
