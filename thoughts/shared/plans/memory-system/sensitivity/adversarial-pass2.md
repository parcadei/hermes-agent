# Adversarial Pass 2: Scale-Check and Materialize

Created: 2026-02-27
Author: architect-agent (Brenner operators: ⊞ Scale-Check, ⌂ Materialize)
Status: COMPLETE — findings require spec corrections before implementation

---

## Executive Summary

Pass 2 found **2 scenario failures**, **3 structural blindspots** in the
sensitivity framework, and **1 misleading metric**. The core math is sound.
The problems are in scenario design and framework coverage.

| Severity | Count | Summary |
|----------|-------|---------|
| CRITICAL | 2 | Scenarios AR-1 and SF-2 produce the WRONG expected winner |
| HIGH | 1 | Parameters `temperature`, `feedback_sensitivity`, `survival_threshold` are invisible to static ranking analysis |
| MEDIUM | 1 | Sensitivity report's tau/score metrics are misleading for contraction-only parameters |
| LOW | 1 | Exploration window W is underspecified (task prompt vs REFERENCE params differ) |

---

## Part 1: ⊞ Scale-Check Findings

### 1.1 Steady-State Strength S*

**Calculation:**
```
alpha=0.1, beta=0.1, delta_t=1.0, s_max=10.0
gamma = (1 - 0.1) * exp(-0.1 * 1.0) = 0.814354
S* = 0.1 * 10.0 / (1 - 0.814354) = 5.3866
```

**Assessment:** S0=1.0 is 18.6% of S* and 10% of Smax. This is reasonable
as an initial strength — new memories start weak and must earn their strength
through repeated access. Convergence to S* is fast: 90% reached by step 10,
99% by step 20. No issue here.

### 1.2 Exploration Window W

**Task prompt params (N0=0.5, gamma=0.2):**
```
W = ln(0.5 / 0.05) / 0.2 = ln(10) / 0.2 = 11.51 steps
```

**REFERENCE_PARAMS (N0=0.3, gamma=0.1):**
```
W = ln(0.3 / 0.05) / 0.1 = ln(6) / 0.1 = 17.92 steps
```

**Assessment:** Both windows are adequate. A new memory has 12-18 steps to
prove itself before the novelty bonus decays below the survival threshold.
Given that convergence to steady-state takes ~20 steps, this is well-matched:
the novelty window covers the "cold start" period almost exactly.

Note: The task prompt uses different novelty params (N0=0.5, gamma=0.2) than
REFERENCE_PARAMS (N0=0.3, gamma=0.1). The test file `test_sensitivity.py`
defines BASELINE_PARAMS with the task prompt values as a cross-check. No
inconsistency, but the two parameter sets exercise slightly different regimes.

### 1.3 soft_select Temperature Analysis

**Calculation:** `soft_select(3.0, 3.5, 5.0) = sigmoid(-0.5/5.0) = sigmoid(-0.1) = 0.4750`

**Is T=5.0 too high?** At T=5.0, the score-to-probability mapping is extremely
flat:

| Score gap | P(winner) | Interpretation |
|-----------|-----------|----------------|
| 0.1 | 0.505 | Coin flip |
| 0.5 | 0.525 | Barely detectable preference |
| 1.0 | 0.550 | Slight preference |
| 2.0 | 0.599 | Moderate preference |
| 5.0 | 0.731 | Clear preference |

**But note:** Base scores live in [0, 1], so the maximum possible gap is 1.0
(+ novelty). At T=5.0, a maximal score gap of 1.0 only gives P=0.55. This
means soft selection at T=5.0 is near-uniform for realistic score differences.
This is **by design** — high temperature prevents thrashing and allows
exploration. However, it means `select_memory` is only weakly correlated with
score rankings. The `test_select_large_score_gap_favors_top` test requires 80%
selection rate over 100 trials, which may fail at T=5.0 if the score gap is
less than ~5.0.

**Finding:** T=5.0 is intentionally high. The sequential tournament compounds
pairwise selections, which partially compensates — the highest-scored memory
starts as incumbent and faces each challenger with >=50% win probability. But
the test at line 498 (`high_count >= 80`) should be verified: with 2 memories
and a score gap of ~0.8 (relevance 1.0 vs 0.0), `P(winner) = soft_select(score_high, score_low, 5.0)`. Let me compute: max score ~1.0, min score ~0.1,
gap ~0.9, P = sigmoid(0.9/5.0) = sigmoid(0.18) = 0.5449. Over 100 trials of
a single pairwise comparison, E[high wins] = 54.5 with SD ~5. Getting 80+ is
a ~5-sigma event. **This test will fail.**

**CORRECTION NEEDED:** Either lower the temperature in the test scenario,
increase the score gap, or lower the threshold from 80 to something achievable
(e.g., 55).

### 1.4 Sensitivity Analysis Computation Count

```
14 params × 6 perturbation levels = 84 perturbation evaluations
Each evaluation: rank 8 non-stability scenarios × ~3 memories each
Total score_memory calls: 84 × 8 × 3 = ~2016
Plus 8 baseline evaluations × 3 = 24
Grand total: ~2040 score_memory calls
```

Each `score_memory` is 5-6 `math.exp` calls plus arithmetic. At ~100ns per
call, total wall time is ~0.2ms. Even with Python overhead (100x), this is
well under 1 second. **No scale concern.**

The 2 stability scenarios (STAB-1: 100 steps, STAB-2: 200 steps) run
`step_dynamics` and `rank_memories` per step but are evaluated only at
baseline, not per perturbation level (since `analyze_parameter` skips
`expected_winner == -1` scenarios). So they add ~300 step evaluations only
once. Still trivial.

### 1.5 rank_memories Complexity

`rank_memories` computes N scores (O(N)), then sorts (O(N log N)). It does
NOT perform pairwise soft_select — that is only in `select_memory`.

`select_memory` does a sequential tournament: O(N-1) pairwise soft_select
calls. This is linear, not quadratic. For 100 memories: 99 sigmoid
evaluations. Negligible.

**No scale concern for either function.**

---

## Part 2: ⌂ Materialize Findings

### 2.1 CRITICAL: AR-1 Produces Wrong Winner

**Scenario:** "High relevance beats high recency"
- Memory 0: relevance=0.95, last_access_time=50, strength=s0=1.0, access_count=2
- Memory 1: relevance=0.40, last_access_time=1, strength=8.0, access_count=10

**Actual scores:**
```
Memory 0:
  w1*rel = 0.35 × 0.95 = 0.3325
  w2*rec = 0.20 × retention(50, 1.0) = 0.20 × 1.93e-22 ≈ 0
  w3*imp = 0.25 × 0.3 = 0.0750
  w4*sig(2) = 0.20 × 0.8808 = 0.1762
  TOTAL = 0.5837

Memory 1:
  w1*rel = 0.35 × 0.4 = 0.1400
  w2*rec = 0.20 × retention(1, 8.0) = 0.20 × 0.8825 = 0.1765
  w3*imp = 0.25 × 0.3 = 0.0750
  w4*sig(10) = 0.20 × 0.9999 = 0.2000
  TOTAL = 0.5915

WINNER: Memory 1 (gap = 0.0078)
```

**Root cause:** The scenario description claims "relevance dominates when
w1 > w2, which is enforced by w2 < 0.4 constraint." This reasoning is
incomplete. While w1=0.35 > w2=0.20, Memory 0's retention is effectively
zero (exp(-50/1) = 1.9e-22) while Memory 1 has near-perfect retention
(exp(-1/8) = 0.88). The recency gap (+0.1765) plus the activation gap
(sigmoid(10) vs sigmoid(2) gives +0.0238) exceeds the relevance advantage
(+0.1925). Net to Memory 1: +0.0078.

**Fix options:**
1. Increase Memory 0's strength so its recency is not catastrophically bad
   (e.g., strength = s_max * 0.5 = 5.0 gives retention(50,5)=exp(-10)≈4.5e-5,
   still ~0 though)
2. Reduce Memory 1's activation advantage (lower access_count from 10 to 2)
3. Reduce Memory 1's recency advantage (increase last_access_time from 1 to 10)
4. Widen the relevance gap (Memory 1 relevance from 0.4 to 0.2)

**Recommended fix:** Change Memory 1's last_access_time from 1.0 to 20.0 AND
access_count from 10 to 2. This preserves the scenario intent (recency should
not override relevance) while making the numbers actually work:
```
Memory 1 revised: retention(20, 8.0) = exp(-2.5) = 0.0821
  w2*rec = 0.20 × 0.0821 = 0.0164
  w4*sig(2) = 0.20 × 0.8808 = 0.1762
  TOTAL ≈ 0.35×0.4 + 0.0164 + 0.075 + 0.1762 = 0.4076
  Gap: 0.5837 - 0.4076 = 0.176 (Memory 0 wins comfortably)
```

### 2.2 CRITICAL: SF-2 Produces Wrong Winner

**Scenario:** "Anti-lock-in: established memory does not dominate forever"
- Memory 0: relevance=0.7, last_access=2, importance=0.5, count=5, strength=s0=1.0
- Memory 1: relevance=0.65, last_access=1, importance=0.9, count=1000, strength=9.5

**Actual scores:**
```
Memory 0:
  base = 0.35×0.7 + 0.20×retention(2,1.0) + 0.25×0.5 + 0.20×sigmoid(5)
       = 0.245 + 0.027 + 0.125 + 0.199 = 0.596
  novelty = 0.3×exp(-1.0) = 0.110
  TOTAL = 0.706

Memory 1:
  base = 0.35×0.65 + 0.20×retention(1,9.5) + 0.25×0.9 + 0.20×sigmoid(1000)
       = 0.228 + 0.180 + 0.225 + 0.200 = 0.833
  novelty ≈ 0
  TOTAL = 0.833

WINNER: Memory 1 (gap = 0.126)
```

**Root cause:** The scenario intends to test anti-lock-in, but gives Memory 1
overwhelming advantages in THREE dimensions:
- **Recency:** retention(1, 9.5) = 0.900 vs retention(2, 1.0) = 0.135
  (Memory 1's high strength + recent access = near-perfect recency)
- **Importance:** 0.9 vs 0.5
- **Activation:** sigmoid(1000) = 1.000 vs sigmoid(5) = 0.993 (marginal)

The anti-lock-in mechanism works through **strength decay** reducing recency
over time. But in this scenario, Memory 1 was accessed at t=1 (very recently)
AND has strength=9.5, so retention is excellent. Anti-lock-in would manifest
if Memory 1 had NOT been accessed recently (e.g., last_access_time=100).

**Recommended fix:** The scenario should test what it claims: an established
memory that has NOT been accessed recently. Change Memory 1's
last_access_time from 1.0 to 50.0, which gives retention(50, 9.5) =
exp(-5.26) = 0.0052, demolishing its recency advantage:
```
Memory 1 revised:
  base = 0.228 + 0.20×0.0052 + 0.225 + 0.200 = 0.654
  Memory 0 total = 0.706 > 0.654 → Memory 0 wins
```

### 2.3 Cold Start Scenario (TTL-2): PASSES Correctly

**Actual scores:**
```
Memory A (new, zero relevance):
  base = 0.0 + 0.20×1.0 + 0.0 + 0.20×0.5 = 0.300
  novelty = 0.300 (full N0)
  TOTAL = 0.600

Memory B (old, moderate relevance):
  base = 0.105 + 0.0 + 0.075 + 0.199 = 0.379
  novelty ≈ 0
  TOTAL = 0.379
```

Memory A wins (gap = 0.221). The novelty bonus (0.300) exceeds Memory B's
base score advantage (0.079). Cold-start survival theorem holds at these
parameters. The margin is healthy: novelty would need to drop below 0.079
(i.e., N0 < 0.079, or roughly N0 reduced by 74%) before TTL-2 fails.

### 2.4 Selective Forgetting (Retention Decay): Works Correctly

```
retention(500, 8.0) = exp(-62.5) = 7.19e-28 ≈ 0
retention(200, 3.0) = exp(-66.7) = 1.11e-29 ≈ 0
```

At these timescales, retention is astronomically small. The exponential decay
is doing its job. Strength only delays the decay (S appears in the denominator
of the exponent), but even strength=8.0 cannot save a 500-step-old memory.
**No issue.**

### 2.5 Sensitivity Report Appearance: Structural Blindspot

**For an insensitive parameter (feedback_sensitivity):**

The sensitivity report for `feedback_sensitivity` will show:
```
| feedback_sensitivity | insensitive | 1.00 | 8/8 | 0.045 |
```

ALL metrics will be perfect (tau=1.0, all scenarios correct, unchanged
contraction margin) because `feedback_sensitivity` does not appear in
`score_memory()` at all. It only affects `importance_update` inside
`step_dynamics`. Since `analyze_parameter` evaluates static single-point
rankings (not multi-step simulations), changes to feedback_sensitivity are
completely invisible.

**For a critical parameter (temperature):**

The report for `temperature` will show:
```
| temperature | critical | 1.00 | 8/8 | -0.005 |
```

Temperature is correctly flagged as critical because the contraction margin
goes negative at factor=-0.5 (T=2.5, K=1.005 > 1). However, the Kendall tau
and score change columns are **misleading** — they show 1.00 and 0.0 because
temperature does not appear in `score_memory()`. It only affects
`soft_select()` and the contraction condition.

**Structural blindspot:** Three parameters are invisible to the ranking
analysis:
1. `temperature` — only in soft_select and contraction
2. `feedback_sensitivity` — only in importance_update (dynamics)
3. `survival_threshold` — only in the Lean coldStart_survival theorem
   (not used in any computation)

The sensitivity framework correctly catches `temperature` via contraction
margin, but the tau/score metrics give a false sense of stability. For
`feedback_sensitivity` and `survival_threshold`, the framework will always
report "insensitive" regardless of perturbation magnitude, which is technically
correct for static rankings but misses their role in multi-step behavior.

**Recommendation:** Add a note in the report for parameters where ALL ranking
metrics are perfect (tau=1.0, all correct), flagging them as "not testable via
static ranking analysis." Alternatively, add a separate dynamics-based
sensitivity check using the `simulate()` function for these parameters.

---

## Part 3: Scenario Verification Summary

| Scenario | Expected | Actual | Status | Gap |
|----------|----------|--------|--------|-----|
| AR-1: High relevance vs high recency | M0 | M1 | **FAIL** | -0.008 |
| AR-2: Relevance tiebreaker via importance | M0 | M0 | PASS | +0.175 |
| AR-3: High relevance vs high activation | M0 | M0 | PASS | +0.041 |
| TTL-1: New high-relevance enters pool | M0 | M0 | PASS | +0.392 |
| TTL-2: Novelty saves zero-base memory | M0 | M0 | PASS | +0.221 |
| SF-1: Stale heavily-accessed loses | M0 | M0 | PASS | +0.160 |
| SF-2: Anti-lock-in | M0 | M1 | **FAIL** | -0.126 |
| SF-3: Low-importance/relevance loses | M0 | M0 | PASS | +0.263 |

AR-3 passes but with a thin margin (0.041). It is robust to moderate
perturbations but could flip under large w2 or w4 perturbations.

---

## Part 4: Test File Concerns

### 4.1 test_select_large_score_gap_favors_top (line 487-500)

At T=5.0, a 2-memory tournament with score gap ~0.8 gives P(winner) ≈ 0.545.
Over 100 trials, E[wins] = 54.5 with SD ≈ 5.0. The test requires
`high_count >= 80`, which is ~5 sigma above the mean. **This test will almost
certainly fail.**

**Fix:** Either lower the threshold to 55, or use T=0.5 for this specific
test, or run 1000 trials and require >= 600.

### 4.2 test_contraction_violating_params (line 201-220)

The test creates params with T=0.1, which gives L = 0.25/0.1 = 2.5. Combined
with alpha=0.9 and s_max=100, K = 0.905 + 225 = 225.9. This is valid — the
test correctly expects `satisfies_contraction() is False`. No issue.

### 4.3 test_scenario_expected_winners_correct_for_baseline (line 757-782)

This meta-test will catch the AR-1 and SF-2 failures during implementation.
This is good — it validates scenario design. But it means the test suite
CANNOT pass until the scenarios are fixed. The test correctly exercises this.

---

## Part 5: Contraction Boundary Map

The contraction condition is K = exp(-beta*delta_t) + (0.25/T)*alpha*s_max < 1.

At baseline: K = 0.9048 + 0.05 = 0.9548, margin = 0.0452.

**Parameters that can break contraction under standard perturbation [-0.5, +0.5]:**

| Parameter | Direction | New value | New K | Margin | Breaks? |
|-----------|-----------|-----------|-------|--------|---------|
| temperature | -50% | T=2.5 | 1.005 | -0.005 | YES |
| alpha | +50% | alpha=0.15 | 0.980 | +0.020 | No |
| s_max | +50% | s_max=15 | 0.980 | +0.020 | No |
| beta | -50% | beta=0.05 | 0.976 | +0.024 | No |
| delta_t | -50% | dt=0.5 | 0.976 | +0.024 | No |

Only `temperature` breaks contraction within the standard perturbation range.
This is because the margin (0.0452) is thin, and temperature appears as 1/T
in the Lipschitz constant, making it a nonlinear amplifier when decreased.

The critical temperature boundary is T_crit = 2.627. The -0.5 perturbation
gives T=2.5, which is just past the boundary. The -0.25 perturbation gives
T=3.75, which is safe (margin = +0.021).

---

## Part 6: Required Spec Corrections

### Correction 1: Fix AR-1 scenario memory values

**File:** spec.md, Section 2.3 (AR-1)

**Current:**
```
Memory 1: relevance=0.4, last_access_time=1.0, importance=0.3,
          access_count=10, strength=s_max*0.8, creation_time=100.0
```

**Corrected:**
```
Memory 1: relevance=0.4, last_access_time=20.0, importance=0.3,
          access_count=2, strength=s_max*0.8, creation_time=100.0
```

**Rationale:** The scenario tests "high relevance beats high recency." But
with last_access_time=1 and strength=8.0, Memory 1's recency + activation
advantage (0.200) exceeds Memory 0's relevance advantage (0.193). Increasing
Memory 1's last_access_time to 20 and reducing access_count to 2 removes the
compound advantage that was masking the relevance signal.

### Correction 2: Fix SF-2 scenario memory values

**File:** spec.md, Section 2.3 (SF-2)

**Current:**
```
Memory 1: relevance=0.65, last_access_time=1.0, importance=0.9,
          access_count=1000, strength=s_max*0.95, creation_time=1000.0
```

**Corrected:**
```
Memory 1: relevance=0.65, last_access_time=50.0, importance=0.9,
          access_count=1000, strength=s_max*0.95, creation_time=1000.0
```

**Rationale:** Anti-lock-in is about established memories that are no longer
being accessed losing to newer, more relevant ones. Setting last_access_time=1
gives Memory 1 near-perfect recency, which contradicts the anti-lock-in
premise. With last_access_time=50, retention(50, 9.5)=exp(-5.26)=0.005,
and Memory 0 wins as intended.

### Correction 3: Document parameter visibility matrix

**File:** spec.md, new subsection in Part 2 (after Section 2.5)

Add a "Parameter Visibility" section documenting which parameters are
exercised by static ranking analysis vs. multi-step dynamics:

| Parameter | In score_memory? | In step_dynamics? | In contraction? | Static analysis detects? |
|-----------|-------------------|-------------------|-----------------|--------------------------|
| alpha | No | Yes (strength_update) | Yes | Contraction only |
| beta | No | Yes (strength_decay) | Yes | Contraction only |
| delta_t | No | Yes (decay, time advance) | Yes | Contraction only |
| s_max | Indirectly (strength bound) | Yes (clamp) | Yes | Partially |
| s0 | No | No (initial condition) | No | Through scenario construction |
| temperature | No | No (soft_select only) | Yes | Contraction only |
| novelty_start | Yes (novelty_bonus) | No | No | Yes |
| novelty_decay | Yes (novelty_bonus) | No | No | Yes |
| survival_threshold | No | No | No | NEVER |
| feedback_sensitivity | No | Yes (importance_update) | No | NEVER |
| w1-w4 | Yes (score) | No | No | Yes |

### Correction 4: Note on test_select_large_score_gap_favors_top

**File:** test_engine.py, line 498

The test threshold of `high_count >= 80` is incompatible with T=5.0. At
T=5.0, the probability of the higher-scored memory winning a single pairwise
comparison with score gap ~0.8 is ~0.545. Over 100 trials, P(>=80) is
effectively zero.

**Options:**
- Lower temperature in this specific test (e.g., use T=0.5)
- Lower threshold to 55 (or 60 with 1000 trials)
- Use a larger score gap (relevance=1.0 vs 0.0 with all other factors
  matching, giving gap ~0.35*1.0 = 0.35, P = sigmoid(0.07) = 0.517, still
  not enough)

---

## Part 7: Open Questions from Pass 2

1. **Should `analyze_parameter` also run the stability simulation scenarios
   under perturbation?** Currently it skips `expected_winner == -1` scenarios
   entirely. This means no sensitivity data is collected for the stability
   competency under parameter perturbation. Consider adding a separate
   stability evaluation path that runs `simulate()` under each perturbation
   and checks the monopolization/convergence criteria.

2. **The spec says STAB-2 checks "final strengths within 20% of
   steady_state_strength." But `steady_state_strength` is for exclusive
   access (q=1). With 2 equal memories sharing via soft_select, the effective
   steady state is S* = q*alpha*Smax / (1 - (1-q*alpha)*exp(-beta*dt)) where
   q≈0.5, giving S*_shared = 3.56 vs S*_exclusive = 5.39.** Which steady
   state should STAB-2 compare against? The shared one is physically correct
   but requires knowing q, which depends on the soft_select dynamics.

3. **The `score` function uses `sigmoid(access_count)` where access_count is
   an integer.** For access_count=0, sigmoid(0)=0.5. For access_count=1,
   sigmoid(1)=0.731. This means even unaccessed memories get a w4*0.5
   activation contribution. Is this intended? It creates a baseline "floor"
   of 0.20*0.5=0.10 from activation alone with REFERENCE weights.
