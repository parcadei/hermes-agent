# CMA-ES Optimizer Spec -- hermes_memory/optimizer.py

## Overview

Thin optimization layer atop engine.py and sensitivity.py. Maps raw CMA-ES
vectors to constrained ParameterSets, evaluates them against expanded benchmark
scenarios, and runs CMA-ES to find optimal parameters.

New module: `hermes_memory/optimizer.py`
New test file: `tests/test_optimizer.py`
Expanded: `sensitivity.py::build_standard_scenarios` → `optimizer.py::build_benchmark_scenarios`

---

## 1. Constants

```python
# 6 free parameters for CMA-ES (from sensitivity analysis)
PARAM_ORDER: list[str] = ["w1", "w2", "w4", "alpha", "beta", "temperature"]

# Bounds per parameter
BOUNDS: dict[str, tuple[float, float]] = {
    "w1":          (0.05, 0.60),
    "w2":          (0.05, 0.39),
    "w4":          (0.05, 0.50),
    "alpha":       (0.01, 0.50),
    "beta":        (0.01, 1.00),
    "temperature": (0.10, 10.0),
}

# Pinned defaults for insensitive parameters
PINNED: dict[str, float] = {
    "delta_t": 1.0,
    "s_max": 10.0,
    "s0": 1.0,
    "novelty_start": 0.3,
    "novelty_decay": 0.1,
    "survival_threshold": 0.05,
    "feedback_sensitivity": 0.1,
}

# Objective weights
STATIC_WEIGHT: float = 0.60
STABILITY_WEIGHT: float = 0.25
MARGIN_WEIGHT: float = 0.15

# Penalty for infeasible candidates
INFEASIBLE_PENALTY: float = 1000.0
```

---

## 2. decode(x: np.ndarray) -> ParameterSet | None

### Contract
- Input: raw numpy array of shape (6,)
- Output: valid ParameterSet or None if infeasible

### Algorithm
1. Clip each element to its BOUNDS range
2. Derive w3 = 1.0 - w1 - w2 - w4
3. If w3 < 0: return None (infeasible)
4. If w3 >= 1.0: return None (degenerate)
5. Construct ParameterSet with clipped values + PINNED defaults + derived w3
6. If ParameterSet.__post_init__ raises ValueError: return None
7. Otherwise return the ParameterSet

### Edge Cases
- x exactly at bounds → valid (boundary is included for w1/w2/w4, open for alpha/beta/temp but clip handles it)
- w1 + w2 + w4 > 1.0 → w3 < 0 → None
- w1 + w2 + w4 = 1.0 → w3 = 0.0 → valid (importance weight is zero)
- alpha at 0.01 with beta at 0.01 and temperature at 0.1 → may violate contraction → None
- NaN or Inf in x → clip produces boundary value, then ParameterSet validates

### Invariants
- If returns non-None, returned ParameterSet satisfies all constraints including contraction
- Weights sum to 1.0 (within 1e-10)
- w2 < 0.4

---

## 3. build_benchmark_scenarios(params: ParameterSet) -> list[Scenario]

### Contract
- Input: a valid ParameterSet
- Output: list of 120+ Scenario instances covering 4 competencies
- The existing 10 scenarios from build_standard_scenarios are included verbatim
- Additional scenarios are procedurally generated with difficulty variations

### Competency Breakdown (target: ~30 per competency)

#### Accurate Retrieval (AR) — ~30 scenarios
Base pattern: One memory has high relevance, competitor has some advantage elsewhere.
Variations:
- Relevance gap: 0.6 vs 0.3 (easy), 0.55 vs 0.45 (hard)
- Competitor advantages: recency, importance, activation, strength
- Pool sizes: 2, 3, 5, 10 memories
- Distractor memories with random moderate stats

#### Test-Time Learning (TTL) — ~30 scenarios
Base pattern: New memory (creation_time=0) enters established pool.
Variations:
- Pool size: 2, 5, 10, 20 established memories
- New memory relevance: 0.9 (easy), 0.6 (medium), 0.3 (hard — novelty must save it)
- Established memory strength: s_max*0.5, s_max*0.8, s_max*0.95
- Established memory recency: recent (lat=5), moderate (lat=20), stale (lat=100)

#### Selective Forgetting (SF) — ~30 scenarios
Base pattern: Stale high-strength memory vs fresh relevant memory.
Variations:
- Staleness levels: lat=50, 100, 200, 500
- Fresh memory relevance: 0.6 (minimum), 0.8 (typical), 0.95 (easy)
- Stale memory access count: 10, 50, 100, 500
- Stale memory importance: 0.3, 0.6, 0.9

#### Stability (STAB) — ~30 scenarios
Base pattern: Multiple similar-score memories, run simulation, check no monopolization.
Variations:
- Pool size: 2, 3, 4, 5, 8 memories
- Score similarity: identical (same relevance), near-identical (0.01 gap), moderate (0.1 gap)
- Simulation length: 50, 100, 200 steps
- All stability scenarios use expected_winner=-1

### Scenario Generation

Each non-stability scenario is verified at construction: `rank_memories()` must
agree with `expected_winner`. If it doesn't, the scenario is adjusted or dropped.

Stability scenarios are NOT verified at construction — they are evaluated
dynamically in the objective function.

### held_out parameter

`build_benchmark_scenarios(params, held_out=False)`
- held_out=False (default): returns training scenarios (used during optimization)
- held_out=True: returns validation scenarios (used after optimization)
- Training and validation scenarios cover the same competency space but use
  different numerical values (e.g., training uses relevance=0.55, validation
  uses relevance=0.57)

---

## 4. objective(x: np.ndarray, scenarios: list[Scenario] | None = None) -> float

### Contract
- Input: raw numpy array of shape (6,), optional pre-built scenarios
- Output: float to MINIMIZE (lower is better)
- Returns INFEASIBLE_PENALTY (1000.0) for infeasible parameter vectors

### Algorithm
1. Call decode(x) → params. If None: return INFEASIBLE_PENALTY
2. If scenarios is None: build_benchmark_scenarios(params)
3. Static scoring:
   - For each non-stability scenario: rank_memories(params, ...) and check if
     ranked[0][0] == expected_winner
   - static_accuracy = correct / total
4. Stability scoring (Finding 4 — explicit algorithm):
   - For each stability scenario:
     ```python
     sim = simulate(params, scenario.memories, 100, [-1]*100, rng=Random(42))
     # Count which memory index ranks first at each step
     win_counts = {}
     for ranking in sim.rankings_per_step:
         winner = ranking[0]
         win_counts[winner] = win_counts.get(winner, 0) + 1
     max_share = max(win_counts.values()) / 100
     passes = (max_share <= 0.80)
     ```
   - stability_accuracy = sum(passes_per_scenario) / max(stability_count, 1)
5. Contraction margin:
   - margin = params.contraction_margin()
   - margin_bonus = min(margin, 0.3) / 0.3  (normalize to [0,1], cap at 0.3)
6. Composite: -(STATIC_WEIGHT * static + STABILITY_WEIGHT * stability + MARGIN_WEIGHT * margin_bonus)

### Sign Convention (Finding 3 — explicit documentation)
CMA-ES minimizes the objective function. Since we want to maximize score:
- objective() returns negative values in [-1, 0] for feasible candidates
- Lower objective → better parameters
- OptimizationResult.best_score = -objective(best_x) (positive for reporting)

### Invariants
- Return value for feasible candidates: in [-1.0, 0.0]
- Return value for infeasible: 1000.0
- Deterministic for same x (rng seeded)
- Static scoring uses rank_memories (deterministic)
- Stability scoring uses simulate with fixed seed (Random(42))

### Edge Cases
- All scenarios correct → returns -(0.60 + 0.25 * stability + 0.15 * margin_bonus)
- Zero scenarios correct → returns -(0.25 * stability + 0.15 * margin_bonus)
- No stability scenarios → stability_accuracy = 0 (division by max(count, 1))
- Infeasible params → 1000.0 (never evaluated)

---

## 5. run_optimization(n_generations: int = 300, seed: int = 42) -> OptimizationResult

### OptimizationResult dataclass
```python
@dataclass(frozen=True)
class OptimizationResult:
    best_x: np.ndarray           # Raw CMA-ES vector (6,)
    best_score: float            # Composite score (positive, higher is better)
    best_params: ParameterSet    # Decoded winner
    history: list[float]         # Best score per generation
    generations: int             # Total generations run
    per_competency: dict[str, float]  # Accuracy per competency category
```

### Contract
- Input: generation count, random seed
- Output: OptimizationResult with the best-found parameters
- Uses CMA-ES from cmaes library
- Initial mean: center of BOUNDS for each parameter
- Initial sigma: mean of (hi - lo) / 4 for each parameter

### Algorithm
1. Compute x0 (center of bounds) and sigma0
2. Create CMA(mean=x0, sigma=sigma0, seed=seed)
3. Build scenarios once with center-of-bounds params (or rebuild per candidate if needed)
4. For each generation up to n_generations:
   a. For each population member: ask() → evaluate objective()
   b. tell(solutions)
   c. Track best_score and best_x
   d. Check should_stop() for early convergence
5. Decode best_x → best_params
6. Compute per-competency accuracy breakdown
7. Return OptimizationResult

### Invariants
- best_params.satisfies_contraction() is True
- best_score >= 0 (it's the positive version of the objective)
- history is monotonically non-decreasing
- Reproducible given same seed

### Scenario Rebuilding Issue
`build_benchmark_scenarios(params)` depends on params (for s_max, s0 in memory
construction). Two approaches:
- A: Rebuild scenarios per candidate (correct but slower)
- B: Build once with PINNED defaults and reuse (faster, scenarios don't change)

**Decision: Approach B.** Scenarios are constructed from PINNED s_max and s0,
which don't change. The 6 free params (w1, w2, w4, alpha, beta, temperature)
only affect scoring/ranking, not memory construction. So scenarios are stable
across candidates.

BUT: scenario verification (expected_winner matches rank) must use a reference
parameter set, not each candidate. Scenarios are designed to have a clear
expected winner under the reference params. Different candidates may produce
different rankings — that's what the objective function measures.

**Implementation:** Build scenarios once at optimization start using a reference
ParameterSet (center of bounds). Remove the verification assertion from
build_benchmark_scenarios for the optimizer case — instead, add a
`verify=True` parameter that defaults to True but can be set to False.

---

## 6. validate(params: ParameterSet) -> ValidationResult

### ValidationResult dataclass
```python
@dataclass(frozen=True)
class ValidationResult:
    contraction_margin: float
    static_accuracy: float
    stability_accuracy: float
    held_out_accuracy: float
    steady_state_strength: float
    steady_state_ratio: float     # s_star / s_max
    exploration_window: float
    weight_summary: dict[str, float]
    all_passed: bool
    failures: list[str]
```

### Contract
- Input: a valid ParameterSet (the optimization winner)
- Output: ValidationResult with all checks
- Does NOT raise — collects failures into a list

### Checks
1. Contraction margin > 0.01 (strict) and > 0.05 (preferred)
2. All training scenarios pass (100% static accuracy)
3. All stability scenarios pass
4. Held-out scenarios: >= 90% accuracy (unseen variations)
5. Analytical cross-check: S* < s_max
6. Analytical cross-check: exploration window W > 0
7. w2 < 0.4

### Analytical Formulas
- Steady-state strength: `S* = alpha * s_max / (1 - (1-alpha) * exp(-beta * delta_t))`
- Exploration window: `W = ln(novelty_start / survival_threshold) / novelty_decay`

---

## 7. Error Handling

- decode() returns None for infeasible — never raises
- objective() returns penalty for infeasible — never raises
- run_optimization() may raise if CMA-ES library errors (shouldn't happen)
- validate() never raises — collects failures
- build_benchmark_scenarios() raises RuntimeError if expected_winner doesn't match
  (when verify=True)

---

## 8. Dependencies

- `cmaes>=0.11` (CMA class)
- `numpy>=1.24`
- Everything else from engine.py and sensitivity.py

---

## 9. Test Categories

### Unit Tests (~30)
- decode: valid vector, boundary vectors, infeasible (w3<0), contraction violation
- objective: feasible evaluation, infeasible penalty, determinism, score range
- build_benchmark_scenarios: count >= 120, competency distribution, expected winner verification
- validate: all-pass case, margin failure, accuracy failure

### Integration Tests (~15)
- Full optimization loop with small generation count (n=10) produces valid result
- Result improves over random search baseline
- Reproducibility: same seed → same result
- Held-out validation passes for optimized params

### Property Tests (~10, via Hypothesis)
- Any valid ParameterSet decoded from in-bounds vector satisfies contraction
- objective() returns value in [-1, 0] for feasible, 1000.0 for infeasible
- Scenarios always have expected_winner in range or -1

### Slow Tests (~5, marked @pytest.mark.slow)
- Full 300-generation optimization
- Validation with held-out scenarios
- Convergence: history[-1] > history[0]
