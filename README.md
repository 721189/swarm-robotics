<img width="1350" height="1350" alt="voronoi_50" src="https://github.com/user-attachments/assets/2857de28-b0f8-4a74-b6a1-d63c68437f14" />
<img width="1536" height="754" alt="heterogeneous" src="https://github.com/user-attachments/assets/23b3699a-1121-4fb4-8b6d-ad9814390c11" />
<img width="1536" height="754" alt="robustness stress test" src="https://github.com/user-attachments/assets/393d8ffe-6495-4d02-a424-84352d6c1276" />

# Swarm Robotics Simulation

A Python-based multi-agent simulation exploring how complex collective 
intelligence emerges from simple, local interactions — inspired by 
biological systems like ant colonies, bird flocking, and drone swarms.

Built from scratch by Shivam Singh, starting from zero Python knowledge.


## What This Project Does

Simulates a swarm of 30 autonomous robots that:
- Move independently with battery tracking
- Sense nearby neighbors using Euclidean distance
- Make decisions based only on local information (no central controller)
- Self-organize using Reynolds Rules: Cohesion, Separation, Alignment
- Partition space using Voronoi tessellation for zero-overlap coverage


## Key Concepts Implemented

| Concept | Description |
|---|---|
| Reynolds Rules | Cohesion, Separation, Alignment |
| Local Sensing | Each robot senses neighbors within radius=8 |
| Voronoi Partitioning | Dynamic territory assignment per robot |
| Emergent Behavior | No central controller — order from local rules |
| Multi-Agent System | 30 independent agents with shared environment |


## Files

| File | Description |
|---|---|
| `swarm_visualisation.py` | Core Robot class (sensing, decision logic) + real-time animation |
| `voronoi_demo.py` | Voronoi-style territory visualization |
| `swarm_neighbour.py` | Small utility: neighbor bearing/direction classification |
| `heterogeneous_swam.py`, `dynamicfeatures.py` | Standalone stigmergy prototypes (superseded by `src/algorithms/stigmergy.py`) |
| `CBBAandSAM_feature.py` | Standalone CBBA + SAM-threat prototype (superseded by `src/algorithms/cbba.py`) |
| `run_framework.py` | Unified CLI — run reynolds/stigmergy/cbba via config or flags |
| `src/` | Framework: config schema, base agent, algorithms, async renderer, metrics |
| `tests/test_framework.py` | Unit tests for config parsing + all 3 algorithm modes |


## Experimental Results (30-Agent Trials)

### Run 1

| Metric | Value |
|---|---|
| Avg frames to stability | 33.37 |
| Avg consensus frame | 26.33 |
| Avg boundary crossings | 7.53 |

### Run 2

| Metric | Value |
|---|---|
| Avg frames to stability | 30.4 |
| Avg consensus frame | 23.6 |
| Avg boundary crossings | 9.9 |

### Key Findings

1. Consensus-Stability Gap
Swarm consistently reaches consensus ~7-9 frames before full 
positional stability. Suggests neighbor agreement precedes 
positional lock-in across all trials.

2. Outlier Behavior
Trial 27 (Run 2): 190 boundary crossings vs avg 9.9
Trial 19 (Run 1): 55 boundary crossings vs avg 7.53
Hypothesis: High-density random initialization creates chaotic 
cluster dynamics that take significantly longer to resolve.

**3. Equilibrium Confirmed**
All 30 robots consistently reach comfortable state 
(Lonely: 0, Crowded: 0, Comfortable: 30) by frame ~128.

## How To Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the visualization
python swarm_visualisation.py

# Run Voronoi simulation
python voronoi_demo.py

# Run experiments / frameworks (headless example)
python run_framework.py --algo cbba --headless --frames 200 --csv results.csv
python run_framework.py --algo reynolds
```

## Roadmap

- Basic Robot class with movement and battery
- Local sensing via Euclidean distance
- Reynolds Rules implementation
- Real-time color-coded visualization
- Voronoi territory partitioning
- 30-agent experimental trials with metrics
- 3 neighbor-selection variant comparison
- Unknown map exploration
- Obstacle avoidance
- Inter-robot communication protocol
- Hardware prototype (Raspberry Pi)

## Background

This project is part of a long-term journey toward building 
AI + hardware systems for autonomous swarm applications in 
defence and exploration contexts.

Inspired by research in distributed robotics, stigmergy, 
and multi-agent systems.

Started: March 2026 | Author: Shivam Singh | India.
