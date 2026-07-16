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

 Concept  Description 

 Reynolds Rules  Cohesion, Separation, Alignment 
 Local Sensing  Each robot senses neighbors within radius=8 
 Voronoi Partitioning  Dynamic territory assignment per robot 
 Emergent Behavior  No central controller  order from local rules |
| Multi-Agent System | 30 independent agents with shared environment


## Files

 File Description 
 `swarm_neighbour.py`  Core Robot class with sensing and decision logic
 `swarm_visualisation.py`  Real-time color-coded animation (matplotlib) 
 `voronoi_demo.py` | Live Voronoi territory visualization 
 `trial_runner.py`  Automated 30-trial data collection 
 `swarm_trial_data_.csv  Raw experimental data from trials 

## Experimental Results (30-Agent Trials)

### Run 1
 Metric  Value 

 Avg frames to stability  31.5 
 Avg consensus frame  22.2  Avg boundary crossings  5.8 

### Run 2
 Metric  Value 

 Avg frames to stability  30.4 
 Avg consensus frame  23.6 
Avg boundary crossings  9.9 

### Key Findings

1. Consensus-Stability Gap
Swarm consistently reaches consensus ~7-9 frames before full 
positional stability. Suggests neighbor agreement precedes 
positional lock-in across all trials.

2. Outlier Behavior
Trial 27 (Run 2): 190 boundary crossings vs avg 9.9
Trial 10 (Run 1): 56 boundary crossings vs avg 5.8
Hypothesis: High-density random initialization creates chaotic 
cluster dynamics that take significantly longer to resolve.

**3. Equilibrium Confirmed**
All 30 robots consistently reach comfortable state 
(Lonely: 0, Crowded: 0, Comfortable: 30) by frame ~128.

## How To Run

```bash
pip install matplotlib numpy

# Run the visualization
python swarm_visualisation.py

# Run Voronoi simulation  
python voronoi_demo.py

# Run 30-trial data collection
python trial_runner.py


## Roadmap

- Basic Robot class with movement and battery
-  Local sensing via Euclidean distance
-  Reynolds Rules implementation
- Real-time color-coded visualization
- Voronoi territory partitioning
-  30-agent experimental trials with metrics
-  3 neighbor-selection variant comparison
-  Unknown map exploration
-  Obstacle avoidance
-  Inter-robot communication protocol
- Hardware prototype (Raspberry Pi)

## Background

This project is part of a long-term journey toward building 
AI + hardware systems for autonomous swarm applications in 
defence and exploration contexts.

Inspired by research in distributed robotics, stigmergy, 
and multi-agent systems.

Started: March 2026 | Author: Shivam Singh | India.
```
