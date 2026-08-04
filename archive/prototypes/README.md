# Archive: Legacy Prototypes & Deprecated Code

This directory contains historical implementations that have been superseded by official framework versions in `src/`.

**DO NOT USE THESE FILES IN PRODUCTION** — they are for reference/history only.

---

## Migration Guide

| Legacy File | Deprecated | Superseded By | Migration Status |
|------------|-----------|--------------|-----------------|
| `heterogeneous_swam.py` | Proof-of-concept stigmergy with scout/worker agents | `src/algorithms/stigmergy.py` | ✅ Fully replaced |
| `dynamicfeatures.py` | Early mediated stigmergy with pheromone decay | `src/algorithms/stigmergy.py` | ✅ Fully replaced |
| `CBBAandSAM_feature.py` | CBBA + SAM threat avoidance prototype | `src/algorithms/cbba.py` + threat models | ✅ Fully replaced |

---

## Why These Files Were Archived

### Code Quality Issues
1. **Duplicate Logic**: Each file reimplemented algorithms independently without code reuse
2. **No Separation of Concerns**: Physics, algorithms, and visualization tightly coupled
3. **No Time-Step Scaling**: Movement/battery drain were frame-dependent, not time-dependent
4. **No Determinism**: Seeding was inconsistent; results non-reproducible

### Architectural Issues
1. **No Configuration Management**: Hard-coded parameters scattered throughout files
2. **No Metrics Logging**: No standardized performance tracking
3. **No Test Infrastructure**: No unit or integration tests
4. **Visualization Tightly Coupled**: Each prototype had its own plotting logic

---

## What Changed in Official Framework

### `src/algorithms/stigmergy.py`
- ✅ Extends `BaseAgent` with proper inheritance
- ✅ Integrates with `SwarmSimulation` engine
- ✅ dt-aware physics (time-step scaling)
- ✅ Centralized configuration via YAML
- ✅ Proper metrics logging
- ✅ Unit tested

### `src/algorithms/cbba.py`
- ✅ Consensus-based bundle algorithm
- ✅ Gossip-based ledger synchronization
- ✅ SAM threat avoidance integrated
- ✅ Multi-agent coordination framework
- ✅ Full test coverage

---

## How to Reference Old Code

If you need to understand the original implementations:

1. **For Stigmergy Concepts**: See `heterogeneous_swam.py` or `dynamicfeatures.py`
2. **For CBBA Algorithm**: See `CBBAandSAM_feature.py`
3. **For How it Evolved**: Check git history: `git log --follow -- src/algorithms/`

---

## Restoration Policy

**These files cannot be restored to production.** If you need functionality from archived code:

1. Create a GitHub Issue describing what you need
2. Implement feature in `src/` following DeepMind standards
3. Add tests in `tests/`
4. Create PR with review gate

---

## Questions?

See `MAINTENANCE_ROADMAP.md` for architecture decisions and migration timeline.
