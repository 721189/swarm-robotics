# Swarm Robotics - Critical Maintenance Roadmap

**Status**: In Progress  
**Branch**: `critical-refactor/architecture-v2`  
**Lead**: Senior DeepMind Engineer  
**Last Updated**: 2026-08-04  

## Executive Summary

This project has grown organically and now requires systematic refactoring to meet production standards equivalent to DeepMind/Google quality gates. This roadmap addresses **9 critical phases** affecting reliability, correctness, and extensibility.

**Total Scope**: ~63 engineering hours  
**Target Completion**: Production-ready within 2-3 weeks  

---

## PHASE 1: Architecture Cleanup 🔴 CRITICAL

### Objective
Eliminate duplicate code, archive legacy prototypes, establish single source of truth.

### Legacy Files to Archive
```
Current State (MESSY):
- heterogeneous_swam.py           ❌ Duplicate of stigmergy
- dynamicfeatures.py              ❌ Unused proof-of-concept
- CBBAandSAM_feature.py           ❌ Duplicate of cbba
- swarm_trial_data_*.csv          ❌ Test artifacts

Target State (CLEAN):
archive/
├── prototypes/
│   ├── heterogeneous_swam.py     (with deprecation notice)
│   ├── dynamicfeatures.py        (with deprecation notice)
│   └── CBBAandSAM_feature.py     (with deprecation notice)
└── README.md                      (explains history)

data/
└── archive/                       (old test results)
```

### Target Project Structure
```
swarm-robotics/
├── src/                           # Official framework only
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Single abstract base
│   │   ├── simulation_engine.py  # Unified engine (NO DUPES)
│   │   ├── metrics.py            # Centralized logging
│   │   └── physics.py            # NEW: Shared physics lib
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── algorithms/               # Official implementations ONLY
│   │   ├── __init__.py
│   │   ├── reynolds.py           # Single implementation
│   │   ├── stigmergy.py          # Single implementation
│   │   ├── cbba.py               # Single implementation
│   │   └── physics_utils.py      # Shared helpers
│   └── visualization/
│       └── renderer.py
├── archive/                       # Historical/deprecated
│   └── prototypes/
├── tests/
│   ├── test_framework.py         # Core tests
│   ├── test_determinism.py       # NEW: Reproducibility
│   ├── test_scalability.py       # NEW: Performance
│   └── test_resilience.py        # NEW: Fault tolerance
├── configs/                       # Example YAML files
├── benchmark/                     # NEW: Experiment suite
│   ├── run_matrix.py
│   ├── analyze_results.py
│   └── visualize.py
├── data/
│   ├── results/                  # Benchmark outputs
│   └── archive/                  # Old CSV files
├── run_framework.py              # Single CLI entry point
├── requirements.txt
├── MAINTENANCE_ROADMAP.md        # This file
└── README.md
```

### Acceptance Criteria
- [ ] All duplicate files moved to archive/
- [ ] No import conflicts
- [ ] All tests pass
- [ ] No `.py` files in root except `run_framework.py`
- [ ] `.gitignore` updated for archive/

---

## PHASE 2: Algorithm Correctness 🔴 CRITICAL

### 2.1 Time-Step Integration (dt Handling)

**Problem**: Movement and battery drain calculations are **frame-dependent**, not time-dependent.

```python
# BROKEN (current code):
class BaseAgent:
    def update_state(self, dx, dy, moving, bounds, dt):
        self.x += dx * self.speed           # ❌ Ignores dt!
        self.y += dy * self.speed           # ❌ Ignores dt!
        
        cost = self.drain_rate if moving else self.drain_rate / 4.0
        self.battery -= cost                # ❌ Per-frame, not per-second!

# CORRECT (required):
class BaseAgent:
    def update_state(self, dx, dy, moving, bounds, dt):
        self.x += dx * self.speed * dt      # ✅ Scales with time
        self.y += dy * self.speed * dt      # ✅ Scales with time
        
        cost = self.drain_rate * dt if moving else (self.drain_rate * dt / 4.0)
        self.battery = max(0.0, self.battery - cost)  # ✅ Time-dependent
```

**Files to Fix**:
1. `src/core/base_agent.py:update_state()` - Add dt scaling
2. `src/core/simulation_engine.py:Objective.decay()` - Add dt to decay rate
3. `src/algorithms/reynolds.py:decide()` - Verify output scaling
4. `src/algorithms/stigmergy.py:decide()` - Verify output scaling
5. `src/algorithms/cbba.py:decide()` - Verify output scaling

**Validation Test**:
```python
def test_dt_scaling():
    """
    Run same scenario with dt=0.1 (10x per second) vs dt=1.0 (1x per second)
    Expected: Energy consumption should be identical per unit time
    """
    config1 = create_config(max_frames=100, dt=0.1, seed=42)  # 10 seconds
    config2 = create_config(max_frames=10, dt=1.0, seed=42)   # 10 seconds
    
    sim1 = SwarmSimulation(config1)
    sim2 = SwarmSimulation(config2)
    
    for _ in range(100): sim1.step(0.1)
    for _ in range(10):  sim2.step(1.0)
    
    assert abs(sim1.agents[0].battery - sim2.agents[0].battery) < 0.01
    assert sim1.agents[0].x == approx(sim2.agents[0].x)
```

### 2.2 Deterministic Seeding

**Problem**: Non-reproducible results across runs.

**Fix Strategy**:
```python
class SwarmSimulation:
    def reset(self):
        """Reproducible re-initialization"""
        # MUST seed before ANY random operations
        random.seed(self.config.simulation.seed)
        np.random.seed(self.config.simulation.seed)
        
        # ALL position initialization AFTER seeding
        for ac in self.config.agents:
            for _ in range(ac.count):
                x = random.uniform(bounds[0] + 2.0, bounds[1] - 2.0)  # ✅ Seeded
                y = random.uniform(bounds[0] + 2.0, bounds[1] - 2.0)  # ✅ Seeded
                # ...
```

**Validation Test**:
```python
def test_determinism():
    """5 runs with same seed must produce IDENTICAL trajectories"""
    trajectories = []
    for trial in range(5):
        sim = SwarmSimulation(config_seed_42)
        for _ in range(200):
            sim.step(0.1)
        trajectories.append([(a.x, a.y) for a in sim.agents])
    
    # All trajectories must be identical to float precision
    for i in range(1, 5):
        for j, pos in enumerate(trajectories[i]):
            assert_almost_equal(pos, trajectories[0][j], decimal=10)
```

### 2.3 Algorithm Consistency Validation

| Algorithm | Validation | Acceptance |
|-----------|------------|-----------|
| **Reynolds** | Neighborhood detection via `euclidean(a, b) ≤ sense_radius` | All agents use same sense radius consistently |
| **Stigmergy** | Paint accumulation: `strength(t+1) = min(1.0, strength(t) + 1.0)` | Exponential decay: `strength(t+1) = strength(t) * (1 - decay_rate * dt)` |
| **CBBA** | Bid calculation deterministic given positions/objectives | Consensus achieved within 200 frames for 10 agents, 5 tasks |

---

## PHASE 3: CBBA Enhancements 🟠 HIGH

### 3.1 Multi-Task Bundle Assignment

**Current Implementation**:
```python
class CombatDroneAgent(BaseAgent):
    assigned_task_id: Optional[int] = None  # ❌ Only one task
```

**Target Implementation**:
```python
class CombatDroneAgent(BaseAgent):
    assigned_tasks: Set[int] = set()        # ✅ Bundle of tasks
    task_precedence: List[int] = []         # ✅ Execution order
    
    def run_local_auction(self, objectives):
        """Find best task bundle for this agent"""
        best_bundle = None
        best_value = -float('inf')
        
        # Enumerate all possible bundles (2^M combinations)
        for bundle in powerset(objectives):
            bundle_value = self._evaluate_bundle(bundle)
            if bundle_value > best_value:
                best_value = bundle_value
                best_bundle = bundle
        
        self.assigned_tasks = best_bundle
        self.task_precedence = self._compute_precedence(best_bundle)
```

### 3.2 Advanced Bidding Strategy

**Current**:
```python
def calculate_local_bid(self, objective) -> float:
    dist = distance(self.x, self.y, objective.x, objective.y)
    return max(0.1, 100.0 - dist)  # ❌ Only Euclidean distance
```

**Target** (Multi-Factor):
```python
def calculate_bid(self, objective, environment) -> float:
    """
    Bid composition:
    - Distance cost (40%)
    - Fuel efficiency (30%)
    - Threat avoidance (20%)
    - Time urgency (10%)
    """
    dist = distance(self.x, self.y, objective.x, objective.y)
    distance_score = max(0.1, 100.0 - dist)
    
    fuel_efficiency = (self.battery / self.max_battery) * 50
    
    threat_penalty = 0
    for threat in environment.threats:
        threat_dist = distance(objective.x, objective.y, threat.x, threat.y)
        if threat_dist < threat.radius * 1.5:  # Risk zone
            threat_penalty += threat.strength / threat_dist
    
    time_urgency = (objective.time_remaining / objective.max_time) * 20
    
    bid = (0.4 * distance_score
         + 0.3 * fuel_efficiency
         - 0.2 * threat_penalty
         + 0.1 * time_urgency)
    
    return max(0.1, bid)
```

### 3.3 Communication Delay Simulation

```python
class MessageQueue:
    """Models communication latency"""
    
    def __init__(self, default_delay_frames: int = 5):
        self.queue: List[Message] = []
        self.default_delay = default_delay_frames
    
    def send(self, from_agent: int, to_agent: int, 
             payload: Dict, delivery_frame: int = None):
        if delivery_frame is None:
            delivery_frame = current_frame + self.default_delay
        
        msg = Message(from_agent, to_agent, payload, delivery_frame)
        self.queue.append(msg)
    
    def step(self, current_frame: int) -> List[Message]:
        """Return messages ready for delivery"""
        delivered = []
        remaining = []
        
        for msg in self.queue:
            if msg.delivery_frame <= current_frame:
                delivered.append(msg)
            else:
                remaining.append(msg)
        
        self.queue = remaining
        return delivered
```

### 3.4 Conflict Resolution

```python
def resolve_assignment_conflict(self, agent_bids: Dict[int, float]) -> int:
    """
    Break ties in CBBA consensus deterministically
    
    Args:
        agent_bids: {agent_id: bid_value}
    
    Returns:
        agent_id of winner
    
    Tie-breaker: Lowest agent_id (deterministic)
    """
    max_bid = max(agent_bids.values())
    tied_agents = [aid for aid, bid in agent_bids.items() if bid == max_bid]
    
    # Deterministic: choose lowest ID
    winner = min(tied_agents)
    return winner
```

---

## PHASE 4: Threat Avoidance & SAM 🟠 HIGH

### 4.1 Obstacle Avoidance Algorithm

Implement **RVO2** (Reciprocal Collision Avoidance):

```python
class RVO2Avoidance:
    """Velocity obstacle-based collision avoidance"""
    
    def compute_avoidance_velocity(self, agent, neighbors, obstacles, dt):
        """
        Compute collision-free velocity using RVO2 principles
        """
        preferred_velocity = agent.desired_velocity
        
        # Build velocity obstacles for all neighbors
        vo_set = []
        for neighbor in neighbors:
            vo = self._compute_velocity_obstacle(agent, neighbor)
            vo_set.append(vo)
        
        # Find velocity outside all obstacles
        new_velocity = self._find_best_velocity(preferred_velocity, vo_set)
        return new_velocity
```

### 4.2 Dynamic Threat Prediction

```python
class ThreatPredictor:
    def predict_threat_trajectory(self, threat: SAMThreat, frames_ahead: int):
        """Predict threat position N frames in future"""
        # Model: Threat searches in expanding spiral
        future_pos = []
        for t in range(frames_ahead):
            angle = threat.search_angle + (2 * np.pi * t) / threat.search_period
            radius = threat.search_radius
            x = threat.x + radius * np.cos(angle)
            y = threat.y + radius * np.sin(angle)
            future_pos.append((x, y))
        return future_pos
    
    def will_be_in_threat_zone(self, agent_trajectory, threat, frames_ahead=10):
        """Check if agent will enter threat engagement zone"""
        threat_predictions = self.predict_threat_trajectory(threat, frames_ahead)
        
        for frame, (tx, ty) in enumerate(threat_predictions):
            if frame < len(agent_trajectory):
                ax, ay = agent_trajectory[frame]
                dist = distance(ax, ay, tx, ty)
                if dist < threat.radius:
                    return True, frame  # Will collide at frame
        
        return False, None
```

### 4.3 Moving Threat Collision Avoidance

```python
def decide(self, simulation: Any) -> Tuple[float, float, bool]:
    """CBBA agent decision with threat avoidance"""
    
    # Standard task-based steering
    target = self._select_target(simulation.objectives)
    if target:
        desired_dx = target.x - self.x
        desired_dy = target.y - self.y
    else:
        desired_dx, desired_dy = 0, 0
    
    # Add threat avoidance layer
    for threat in simulation.threats:
        threat_dist = distance(self.x, self.y, threat.x, threat.y)
        
        if threat_dist < threat.radius * 2:  # Warning zone
            # Compute repulsion from threat
            away_dx = self.x - threat.x
            away_dy = self.y - threat.y
            away_dist = distance(away_dx, away_dy, 0, 0)
            
            if away_dist > 0:
                away_dx /= away_dist
                away_dy /= away_dist
                
                # Blend: 50% desired, 50% evasion
                desired_dx = 0.5 * desired_dx + 0.5 * away_dx
                desired_dy = 0.5 * desired_dy + 0.5 * away_dy
    
    # Normalize and return
    length = distance(desired_dx, desired_dy, 0, 0)
    if length > 0:
        desired_dx /= length
        desired_dy /= length
    
    return desired_dx, desired_dy, True
```

### 4.4 Post-Escape Recovery

```python
class AgentDamageModel:
    def apply_damage(self, agent, damage_amount: float):
        """Apply damage and degradation"""
        agent.state = "damaged"
        agent.health = max(0.0, agent.health - damage_amount)
        
        # Capability degradation
        agent.speed *= 0.7              # 30% speed loss
        agent.battery *= 0.8            # 20% battery loss
        agent.sense_radius *= 0.9       # 10% sensing loss
        
        # Mark for recovery
        agent.repair_frame = current_frame + 50  # 50-frame repair time
    
    def attempt_repair(self, agent, current_frame: int):
        """Try to repair damaged agent"""
        if current_frame >= agent.repair_frame:
            agent.state = "active"
            agent.health = agent.max_health
            agent.speed = agent.original_speed
            agent.battery = agent.max_battery * 0.9  # 90% capacity recovered
            agent.sense_radius = agent.original_sense_radius
```

---

## PHASE 5: Reynolds Flocking Improvements 🟡 MEDIUM

### 5.1 Configurable Flocking Parameters

```yaml
# configs/reynolds_advanced.yaml
simulation:
  algorithm: "reynolds"
  seed: 42
  bounds: [-40.0, 40.0]
  dt: 0.05
  max_frames: 500

reynolds:
  cohesion_weight: 1.0
  separation_weight: 1.5
  alignment_weight: 0.8
  
  max_velocity: 2.0
  max_acceleration: 0.3
  
  separation_distance: 3.0
  comfortable_neighbor_count: 3

agents:
  types:
    - name: "base"
      count: 30
      params:
        speed: 1.5
        sense_radius: 8.0
        battery: 9999.0
        drain: 0.0
```

### 5.2 Smooth Steering (Damped Steering)

```python
class ReynoldsAgent(BaseAgent):
    def __init__(self, agent_id, x, y, params):
        super().__init__(agent_id, x, y, params)
        self.velocity = np.array([0.0, 0.0])
        self.steering_damping = params.get("steering_damping", 0.15)
    
    def decide(self, simulation) -> Tuple[float, float, bool]:
        """Smooth steering with acceleration limits"""
        
        # Compute desired direction (no instant changes)
        desired_direction = self._compute_desired_direction(simulation)
        
        # Apply velocity damping (smooth acceleration)
        max_accel = self.params.get("max_acceleration", 0.3)
        
        # LERP: smooth interpolation instead of bang-bang
        desired_vel = desired_direction * self.speed
        
        # Limit acceleration
        delta = desired_vel - self.velocity
        delta_mag = np.linalg.norm(delta)
        
        if delta_mag > max_accel:
            delta = (delta / delta_mag) * max_accel
        
        self.velocity = self.velocity + delta
        
        # Limit max velocity
        vel_mag = np.linalg.norm(self.velocity)
        max_vel = self.params.get("max_velocity", 2.0)
        if vel_mag > max_vel:
            self.velocity = (self.velocity / vel_mag) * max_vel
        
        # Return normalized steering
        dx, dy = self.velocity[0], self.velocity[1]
        moving = vel_mag > 0.01
        
        return dx, dy, moving
```

### 5.3 Spatial Partitioning for Neighbor Search

Replace O(n²) all-pairs with O(n) hash grid:

```python
class SpatialGrid:
    """Hash-based spatial indexing for efficient neighbor queries"""
    
    def __init__(self, bounds, cell_size):
        self.bounds = bounds
        self.cell_size = cell_size
        self.grid = defaultdict(list)
    
    def insert(self, agent):
        """Insert agent into grid"""
        cell = self._get_cell(agent.x, agent.y)
        self.grid[cell].append(agent)
    
    def get_neighbors(self, x, y, search_radius):
        """Get all agents within search_radius in O(1) avg time"""
        cell_x = int((x - self.bounds[0]) / self.cell_size)
        cell_y = int((y - self.bounds[0]) / self.cell_size)
        
        neighbors = []
        # Check 3x3 cell neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (cell_x + dx, cell_y + dy)
                for agent in self.grid[cell]:
                    dist = distance(x, y, agent.x, agent.y)
                    if dist <= search_radius:
                        neighbors.append(agent)
        
        return neighbors
```

**Performance Impact**: 
- Old: O(n²) = 100 agents → 10,000 comparisons
- New: O(n) = 100 agents → ~300 comparisons

---

## PHASE 6: Heterogeneous Agents 🟡 MEDIUM

### 6.1 Parameter Distribution Support

```yaml
# configs/heterogeneous_swarm.yaml
agents:
  types:
    - name: "scout"
      count: 5
      params:
        speed: {distribution: "normal", mean: 2.0, std: 0.2}
        sense_radius: {distribution: "normal", mean: 20.0, std: 2.0}
        battery: {distribution: "normal", mean: 80.0, std: 5.0}
        drain_rate: {distribution: "normal", mean: 0.22, std: 0.02}
        max_force: {distribution: "uniform", min: 0.15, max: 0.25}
    
    - name: "worker"
      count: 15
      params:
        speed: {distribution: "normal", mean: 1.0, std: 0.1}
        sense_radius: {distribution: "normal", mean: 8.0, std: 1.0}
        battery: {distribution: "normal", mean: 200.0, std: 20.0}
        drain_rate: {distribution: "normal", mean: 0.05, std: 0.01}
```

### 6.2 Implementation

```python
class AgentConfig:
    def sample_params(self, seed: int) -> Dict[str, float]:
        """Sample parameters from distributions"""
        random.seed(seed)
        np.random.seed(seed)
        
        sampled = {}
        for key, spec in self.params.items():
            if isinstance(spec, dict) and 'distribution' in spec:
                dist_type = spec['distribution']
                
                if dist_type == 'normal':
                    sampled[key] = np.random.normal(spec['mean'], spec['std'])
                elif dist_type == 'uniform':
                    sampled[key] = np.random.uniform(spec['min'], spec['max'])
                else:
                    raise ValueError(f"Unknown distribution: {dist_type}")
            else:
                sampled[key] = spec  # Use constant value
        
        return sampled
```

---

## PHASE 7: Network Resilience 🟢 LOW

### 7.1 Fault Injection Framework

```python
class FaultModel:
    """Inject controlled faults for resilience testing"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.faults = []
    
    def schedule_agent_failure(self, agent_id: int, frame: int):
        """Mark agent for failure at specific frame"""
        self.faults.append(('agent_failure', agent_id, frame))
    
    def schedule_link_failure(self, agent_a: int, agent_b: int, 
                            start_frame: int, duration: int):
        """Block communication between two agents"""
        self.faults.append(('link_failure', (agent_a, agent_b), start_frame, duration))
    
    def schedule_sensor_degradation(self, agent_id: int, frame: int, 
                                   reduction_factor: float):
        """Reduce sensor range"""
        self.faults.append(('sensor_degrade', agent_id, frame, reduction_factor))
    
    def apply(self, simulation, current_frame: int):
        """Apply scheduled faults"""
        for fault in self.faults:
            if fault[2] == current_frame:
                if fault[0] == 'agent_failure':
                    simulation.agents[fault[1]].alive = False
                    simulation.agents[fault[1]].state = "failed"
```

### 7.2 Connectivity Monitoring

```python
class ConnectivityGraph:
    """Track network topology for resilience metrics"""
    
    def update(self, agents, comm_range):
        """Update connectivity based on positions"""
        self.edges = []
        
        for i, a1 in enumerate(agents):
            if not a1.alive: continue
            for j, a2 in enumerate(agents):
                if i >= j or not a2.alive: continue
                
                dist = distance(a1.x, a1.y, a2.x, a2.y)
                if dist <= comm_range:
                    self.edges.append((i, j))
        
        self.graph = self._build_graph(agents, self.edges)
    
    def is_connected(self):
        """Check if network forms single connected component"""
        return self._num_connected_components() == 1
    
    def find_partitions(self):
        """Find disconnected subswarms"""
        return self._get_connected_components()
```

---

## PHASE 8: Advanced Metrics 🟠 HIGH

### 8.1 Extended Metrics Logger

```python
class AdvancedMetricsLogger(MetricsLogger):
    """Track detailed performance metrics"""
    
    def __init__(self):
        super().__init__()
        self.collision_count = 0
        self.messages_sent = 0
        self.completed_tasks = set()
    
    def log_frame(self, frame_id, agents, objectives, threats=None):
        super().log_frame(frame_id, agents, objectives)
        
        frame_data = self.history[-1]
        
        # Task metrics
        frame_data['task_completion_rate'] = len(self.completed_tasks) / len(objectives)
        
        # Collision metrics
        frame_data['collision_count'] = self._detect_collisions(agents)
        
        # Coverage metrics
        frame_data['coverage_percent'] = self._compute_coverage(agents)
        
        # Energy metrics
        frame_data['avg_battery'] = np.mean([a.battery for a in agents if a.alive])
        frame_data['total_energy_consumed'] = sum(
            a.max_battery - a.battery for a in agents
        )
        
        # Efficiency metrics
        frame_data['idle_robot_percent'] = sum(
            1 for a in agents if a.alive and not getattr(a, 'moving', False)
        ) / len(agents)
        
        # Connectivity
        frame_data['connectivity_score'] = self._compute_connectivity(agents)
```

### 8.2 Comprehensive Metrics Snapshot

```python
METRICS_SCHEMA = {
    # Frame basics
    'frame': int,
    'timestamp': float,
    
    # Agent states
    'active_count': int,
    'states': Dict[str, int],  # {lonely, crowded, comfortable, depleted, failed}
    
    # Task completion
    'task_completion_rate': float,          # 0.0 - 1.0
    'tasks_completed_total': int,
    'avg_task_wait_time': float,
    
    # Collisions
    'agent_collisions': int,
    'agent_obstacle_collisions': int,
    'total_collisions': int,
    
    # Coverage & exploration
    'coverage_percent': float,              # 0.0 - 100.0
    'explored_area': float,
    'revisit_ratio': float,
    
    # Communication
    'connectivity_score': float,            # 0.0 - 1.0 (% connected)
    'messages_sent': int,
    'messages_delivered': int,
    'communication_overhead': float,
    
    # Energy & efficiency
    'avg_battery': float,
    'min_battery': float,
    'max_battery': float,
    'total_energy_consumed': float,
    'battery_utilization': float,          # Energy used / available
    
    # Timing
    'consensus_frame': int,                 # When algorithm converged
    'stability_frame': int,                 # When swarm settled
    
    # Swarm efficiency
    'avg_path_length': float,              # Total distance / agents
    'swarm_efficiency': float,             # Tasks / Energy / Time
    'load_balance_score': float,           # Std dev of task assignments
    
    # Movement
    'idle_robot_percent': float,           # % not moving
    'avg_velocity': float,
    'formation_cohesion': float,
}
```

---

## PHASE 9: Experiment Validation 🔴 CRITICAL

### 9.1 Benchmark Experiment Matrix

```python
# benchmark/run_matrix.py

EXPERIMENT_MATRIX = {
    'agent_counts': [10, 20, 50, 100],
    'map_sizes': [40, 100, 200],
    'obstacle_densities': [0.0, 0.1, 0.25, 0.5],
    'threat_counts': [0, 2, 5, 10],
    'seeds': list(range(1, 11)),
}

# Total combinations: 4 × 3 × 4 × 4 × 10 = 1920 runs
```

### 9.2 Baseline Validation Criteria

| Component | Metric | Target | Tolerance |
|-----------|--------|--------|-----------|
| **Reynolds** | Stability time (30 agents) | < 100 frames | ±10% |
| **Reynolds** | Final comfortable % | ≥ 90% | ±5% |
| **Reynolds** | Avg velocity smoothness | High | Visual inspection |
| **Stigmergy** | Time to paint all objectives | < 200 frames | ±20% |
| **Stigmergy** | Worker survival rate | ≥ 95% | ±3% |
| **Stigmergy** | Objective coverage % | 100% | Exact |
| **CBBA** | Task assignment rate | 100% | Exact |
| **CBBA** | Consensus time | < 150 frames | ±15% |
| **CBBA** | No over-assignment | 0 conflicts | Exact |
| **All** | Determinism (5 runs) | 100% identical | Exact |
| **All** | Crash rate | 0% | Exact |
| **All** | NaN values | 0 | Exact |
| **Scaling** | 10→100 agents | Linear time | ±10% |

### 9.3 Analysis & Reporting

```python
# benchmark/analyze_results.py

def generate_report(results_dir):
    results = load_all_runs(results_dir)
    
    report = {
        'summary': {
            'total_runs': len(results),
            'successful_runs': sum(1 for r in results if r.success),
            'failed_runs': sum(1 for r in results if not r.success),
            'crash_rate': f"{100 * sum(1 for r in results if r.crashed) / len(results):.2f}%",
        },
        
        'by_algorithm': {
            'reynolds': analyze_algorithm(results, 'reynolds'),
            'stigmergy': analyze_algorithm(results, 'stigmergy'),
            'cbba': analyze_algorithm(results, 'cbba'),
        },
        
        'scalability': {
            'time_10_agents': results.filter(agents=10).avg_runtime,
            'time_100_agents': results.filter(agents=100).avg_runtime,
            'scaling_factor': scaling_factor,  # Should be ~10x
        },
        
        'reliability': {
            'determinism_violations': count_nondeterministic_runs(results),
            'crash_incidents': list_crashes(results),
            'nan_violations': list_nan_incidents(results),
        },
        
        'performance_baselines': generate_baseline_table(results),
    }
    
    return report
```

---

## Sign-Off & Quality Gate Checklist

Before merging to `main`, ALL must pass:

### Code Quality
- [ ] No duplicate implementations
- [ ] All files properly organized
- [ ] Type hints on all functions
- [ ] Docstrings on all classes/methods
- [ ] Code review approved by 2+ engineers

### Testing
- [ ] Unit tests: 100% pass rate
- [ ] Integration tests: 100% pass rate
- [ ] Determinism tests: 5/5 identical runs
- [ ] Scalability tests: 1920 runs completed
- [ ] Resilience tests: Fault injection validated

### Physics & Algorithms
- [ ] dt-aware calculations verified
- [ ] Reynolds: Stability frame < 100
- [ ] Stigmergy: All objectives painted < 200 frames
- [ ] CBBA: 100% task assignment < 150 frames
- [ ] No NaN values in any metric
- [ ] No crashes in 1920-run benchmark

### Metrics & Monitoring
- [ ] All 20+ metrics logged
- [ ] CSV/JSON/Parquet export working
- [ ] Performance baselines established
- [ ] Anomalies identified & documented

### Documentation
- [ ] README updated with new structure
- [ ] MAINTENANCE_ROADMAP.md complete
- [ ] Algorithm references added (papers)
- [ ] Example configs in `configs/`
- [ ] Benchmark instructions in `benchmark/README.md`

### Git Workflow
- [ ] All commits atomic & meaningful
- [ ] Branch name follows convention
- [ ] PR description detailed
- [ ] No merge conflicts
- [ ] CI/CD passes

---

## Engineering Standards (DeepMind Level)

✅ **Reproducibility**
- Deterministic with fixed seed
- dt-aware physics
- Comprehensive logging

✅ **Performance**
- O(n) neighbor search, not O(n²)
- Vectorized operations where possible
- Benchmarks on standard hardware

✅ **Correctness**
- Unit tests for all modules
- Integration tests for workflows
- Validation against known results

✅ **Maintainability**
- Single source of truth
- Clear separation of concerns
- Extensive documentation

---

## Timeline & Allocation

**Total Scope**: 63 engineering hours over 2-3 weeks

| Phase | Hours | Target Completion |
|-------|-------|------------------|
| 1. Architecture | 3 | Week 1 Mon |
| 2. Physics & Determinism | 8 | Week 1 Wed |
| 3. CBBA Improvements | 10 | Week 1 Fri |
| 4. Threat Avoidance | 8 | Week 2 Wed |
| 5. Reynolds Improvements | 6 | Week 2 Thu |
| 6. Heterogeneous Agents | 8 | Week 2 Fri |
| 7. Resilience Testing | 4 | Week 3 Mon |
| 8. Advanced Metrics | 6 | Week 3 Wed |
| 9. Validation & Benchmarks | 10 | Week 3 Fri |
| **TOTAL** | **63** | **Week 3** |

---

## Contact & Escalation

- **Lead**: Senior DeepMind Engineer
- **Branch**: `critical-refactor/architecture-v2`
- **PR Requirement**: All phases must pass before merge to main
- **Code Review**: Minimum 2 approvals required

