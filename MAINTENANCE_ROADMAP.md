# Swarm Robotics - Critical Maintenance Roadmap

**Status**: In Progress  
**Branch**: `critical-refactor/architecture-v2`  
**Lead**: Senior DeepMind Engineer  
**Last Updated**: 2026-08-04  

## Executive Summary

This project requires systematic refactoring to meet production standards. This roadmap addresses **9 critical phases** affecting reliability, correctness, and extensibility.

**Total Scope**: ~63 engineering hours | **Timeline**: 2-3 weeks

---

## PHASE 1: Architecture Cleanup 🔴 CRITICAL (3 hrs)

**Remove duplicates & archive legacy code**

Legacy files to move to `archive/prototypes/`:
- `heterogeneous_swam.py` (duplicate of stigmergy)
- `dynamicfeatures.py` (unused proof-of-concept)
- `CBBAandSAM_feature.py` (duplicate of cbba)
- CSV test artifacts → `data/archive/`

Target structure:
```
src/core/               # Single implementations only
src/algorithms/         # Official Reynolds, Stigmergy, CBBA
tests/                  # Expanded test suite
archive/prototypes/     # Historical versions
```

**Acceptance**: No duplicate code, no root .py files except run_framework.py

---

## PHASE 2: Algorithm Correctness 🔴 CRITICAL (8 hrs)

### 2.1 Fix dt (Time-Step) Integration
**Problem**: Movement/battery drain ignore dt parameter

```python
# BROKEN:
self.x += dx * self.speed           # ❌ Per-frame
self.battery -= drain_rate          # ❌ Per-frame

# FIXED:
self.x += dx * self.speed * dt      # ✅ Time-scaled
self.battery -= drain_rate * dt     # ✅ Time-scaled
```

Files: base_agent.py, simulation_engine.py, all algorithms

### 2.2 Deterministic Seeding
**Problem**: Non-reproducible runs

**Fix**: Seed before ALL randomness, validate 5 runs identical

### 2.3 Algorithm Consistency
Validate Reynolds, Stigmergy, CBBA produce consistent results

---

## PHASE 3: CBBA Enhancements 🟠 HIGH (10 hrs)

- Multi-task bundle assignment (not single task)
- Advanced bidding: distance (40%) + fuel (30%) + threat (20%) + urgency (10%)
- Communication delay simulation
- Deterministic tie-breaking

---

## PHASE 4: Threat Avoidance & SAM 🟠 HIGH (8 hrs)

- RVO2 collision avoidance
- Dynamic threat prediction
- Moving threat collision detection
- Post-escape recovery & damage model

---

## PHASE 5: Reynolds Improvements 🟡 MEDIUM (6 hrs)

- Configurable cohesion/separation/alignment weights
- Smooth steering (damped, not bang-bang)
- O(n) spatial grid neighbor search (not O(n²))

---

## PHASE 6: Heterogeneous Agents 🟡 MEDIUM (8 hrs)

Support parameter distributions:
```yaml
speed: {distribution: "normal", mean: 2.0, std: 0.2}
sense_radius: {distribution: "normal", mean: 20.0, std: 2.0}
```

Different agent types with variance

---

## PHASE 7: Network Resilience 🟢 LOW (4 hrs)

- Fault injection (agent failure, link failure, sensor degradation)
- Connectivity monitoring
- Recovery from network partitions

---

## PHASE 8: Advanced Metrics 🟠 HIGH (6 hrs)

Track 20+ metrics:
- Task completion rate
- Collision count
- Coverage %
- Connectivity score
- Communication overhead
- Consensus time
- Energy efficiency
- Swarm efficiency
- Load balancing

Export: CSV, JSON, Parquet

---

## PHASE 9: Experiment Validation 🔴 CRITICAL (10 hrs)

**1920 benchmark runs**:
- Agents: [10, 20, 50, 100]
- Map size: [40×40, 100×100, 200×200]
- Obstacle density: [0%, 10%, 25%, 50%]
- Threat count: [0, 2, 5, 10]
- Seeds: [1..10]

**Validation Criteria**:
- Reynolds stability < 100 frames (30 agents)
- Stigmergy: all objectives painted < 200 frames
- CBBA: 100% task assignment < 150 frames
- Determinism: 100% identical runs (5 trials)
- Crash rate: 0%

---

## Sign-Off Checklist (Before Merge to Main)

- [ ] Phase 1: No duplicates, archive created
- [ ] Phase 2: dt-aware physics, deterministic seeding
- [ ] Phase 3: CBBA consensus tested
- [ ] Phase 4: Threat avoidance validated
- [ ] Phase 5: Reynolds smooth & O(n)
- [ ] Phase 6: Heterogeneous agents working
- [ ] Phase 7: Resilience tests passed
- [ ] Phase 8: All metrics logged
- [ ] Phase 9: 1920 runs, baselines established
- [ ] 100% test pass rate
- [ ] Code review approved (2+ engineers)
- [ ] Zero crashes in extended runs

---

## Timeline

| Phase | Hours | Target |
|-------|-------|--------|
| 1. Architecture | 3 | Week 1 Mon |
| 2. Physics & Determinism | 8 | Week 1 Wed |
| 3. CBBA | 10 | Week 1 Fri |
| 4. Threats | 8 | Week 2 Wed |
| 5. Reynolds | 6 | Week 2 Thu |
| 6. Heterogeneous | 8 | Week 2 Fri |
| 7. Resilience | 4 | Week 3 Mon |
| 8. Metrics | 6 | Week 3 Wed |
| 9. Validation | 10 | Week 3 Fri |
| **TOTAL** | **63** | **Week 3** |

---

## Professional Standards (DeepMind Level)

✅ **Reproducibility**: Deterministic + dt-aware + comprehensive logging  
✅ **Performance**: O(n) search, vectorized ops, benchmarks  
✅ **Correctness**: Unit + integration + validation tests  
✅ **Maintainability**: Single source of truth, clear separation, docs

