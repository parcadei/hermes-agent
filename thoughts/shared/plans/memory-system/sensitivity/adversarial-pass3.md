# Adversarial Analysis — Pass 3/3 (Brenner Kernel)

Created: 2026-02-27
Author: architect-agent (Opus 4.6)
Operators: Exclusion-Test, Object-Transpose

---

## Executive Summary

Pass 3 reveals **three structural defects** in the spec that, left unaddressed,
would make the sensitivity analysis produce misleading results. Two are
show-stoppers (incorrect scenario winners), one is a design-level information
loss (most parameters are invisible to the metric suite).

| Finding | Severity | Operator |
|---------|----------|----------|
| F1. Two scenarios have wrong expected winners | CRITICAL | Exclusion-Test |
| F2. 8/14 parameters have zero effect on rankings | CRITICAL | Exclusion-Test |
| F3. Remaining scenarios are mostly non-discriminative | HIGH | Exclusion-Test |
| F4. Kendall tau is degenerate for 2-memory scenarios | MEDIUM | Object-Transpose |
| F5. alpha/s_max and beta/delta_t are perfectly correlated | MEDIUM | Exclusion-Test |
| F6. ParameterSet conflates two independent layers | LOW | Object-Transpose |
| F7. Tournament vs sort question is architecturally moot | INFO | Object-Transpose |

---

## F1. CRITICAL: Two Scenarios Have Incorrect Expected Winners

### AR-1: "High relevance beats high recency" -- ACTUALLY WRONG

The spec states Memory 0 (relevance=0.95, strength=s0=1.0, last_access_time=50)
should beat Memory 1 (relevance=0.4, strength=s_max*0.8=8.0, last_access_time=1).

**Numerical verification:**
```
M0: w1*0.95 = 0.3325,  w2*retention(50, 1.0) = w2*exp(-50) ~ 0.0000
    w3*0.3 = 0.0750,    w4*sigmoid(2) = 0.1762
    Total: 0.5837

M1: w1*0.4 = 0.1400,   w2*retention(1, 8.0) = w2*0.8825 = 0.1765
    w3*0.3 = 0.0750,    w4*sigmoid(10) = 0.2000
    Total: 0.5915

Winner: M1 (not M0). Gap: 0.0078.
```

**Root cause:** Memory 0 has strength=s0=1.0 and last_access_time=50. The
retention function retention(50, 1) = exp(-50) is effectively zero. The spec
assumed relevance would dominate, but the recency component of M0 is completely
dead. Meanwhile M1's retention(1, 8) = exp(-0.125) = 0.88 is excellent.

**The spec's narrative is correct** ("high relevance should beat high recency")
**but the memory parameters don't instantiate that narrative.** Memory 0 needs
higher strength to have non-negligible recency, or a much shorter
last_access_time.

### SF-2: "Anti-lock-in: established memory does not dominate" -- ACTUALLY WRONG

The spec states Memory 0 (relevance=0.7, strength=s0=1.0, last_access_time=2)
should beat Memory 1 (relevance=0.65, strength=s_max*0.95=9.5,
last_access_time=1, importance=0.9, access_count=1000).

**Numerical verification:**
```
M0: total = 0.7061
M1: total = 0.8325
Winner: M1 (not M0). Gap: 0.1264.
```

**Root cause:** The spec comment says "strength decay has reduced memory 1's
recency to near-zero" but this is wrong. Memory 1 has last_access_time=1 and
strength=9.5, giving retention(1, 9.5) = exp(-1/9.5) = 0.900. The spec
confused the effect of creation_time (old memory) with last_access_time
(recent access). Memory 1 was accessed very recently (lat=1) and has high
strength, so its recency is excellent. Combined with importance=0.9 and
sigmoid(1000) ~ 1.0, Memory 1 dominates.

### Exclusion-Test (F1)

This is a **deletion-level** finding. If the test
`test_scenario_expected_winners_correct_for_baseline` (test_sensitivity.py:757)
runs against the implementation, it will FAIL for AR-1 and SF-2, blocking all
downstream analysis. The entire sensitivity analysis rests on scenario
correctness.

### Mitigation (F1)

Fix the scenario memory parameters to match their narrative intent:

**AR-1 fix:** Give Memory 0 sufficient strength for non-negligible recency, or
reduce its last_access_time:
```python
# Option A: increase M0 strength
Memory 0: strength=s_max*0.5  (retention(50, 5.0) = exp(-10) ~ 0.000045 -- still low)
# Option B: reduce M0 last_access_time
Memory 0: last_access_time=5.0  (retention(5, 1.0) = exp(-5) = 0.0067 -- marginal)
# Option C: do both
Memory 0: last_access_time=5.0, strength=s0*3  (retention(5, 3) = exp(-1.67) = 0.189)
```

**SF-2 fix:** The scenario should demonstrate that high access_count + high
strength does NOT guarantee winning. But the current M1 also has high recency
(lat=1) and high importance (0.9). Fix by increasing M1's last_access_time:
```python
Memory 1: last_access_time=50.0  (retention(50, 9.5) = exp(-5.26) = 0.005 -- now actually stale)
```

---

## F2. CRITICAL: 8/14 Parameters Have Zero Effect on Rankings

For non-stability scenarios, `analyze_parameter` computes Kendall tau and
score_change by calling `rank_memories` and `score_memory` at a single time
point. The scoring formula is:

```
score = w1*rel + w2*retention(lat, strength) + w3*imp + w4*sigmoid(ac) + novelty_bonus(N0, gamma, ct)
```

The following 8 parameters **do not appear anywhere in this formula**:

| Parameter | Where it matters (NOT in scoring) |
|-----------|-----------------------------------|
| alpha | strength_update (dynamics), contraction condition |
| beta | strength_decay (dynamics), contraction condition |
| delta_t | dynamics step size, contraction condition |
| s_max | domain bound, contraction condition |
| s0 | initial strength (used in memory construction only) |
| temperature | soft_select probability, contraction condition |
| feedback_sensitivity | importance_update (dynamics) |
| survival_threshold | exploration_window (theoretical) |

For ALL of these, the OAT analysis will report:
- `kendall_tau_per_level = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- `score_change_per_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- `scenarios_correct_per_level = [8, 8, 8, 8, 8, 8]`

The ONLY differentiating signal will be `contraction_margin_per_level`, which
changes for alpha, beta, delta_t, s_max, and temperature, but NOT for s0,
feedback_sensitivity, or survival_threshold.

### Exclusion-Test (F2)

**Degenerate-case test:** If ALL parameters are classified as insensitive, the
scoring function is essentially a constant (does not discriminate). This is
exactly the degenerate case the prompt warned about -- and it applies to 8/14
parameters.

**How to detect this:** Add a diagnostic check in `run_sensitivity_analysis`:
if more than half of parameters have `max(score_change_per_level) < epsilon`,
emit a WARNING that the scenario suite is not exercising those parameters.

### Exclusion-Test (F2, parameter correlation)

alpha and s_max appear as the product `L * alpha * s_max` in the contraction
factor K. A 10% increase in alpha has EXACTLY the same effect on K as a 10%
increase in s_max (verified numerically: both shift K by L * 0.01 * s_max =
0.005). This means their `contraction_margin_per_level` curves will be
identical, and the OAT analysis cannot distinguish their contributions.

Similarly, beta and delta_t appear as `exp(-beta * delta_t)`, so their marginal
effects on K are:
- dK/d(beta) * delta_beta = delta_t * exp(-beta*dt) * delta_beta
- dK/d(delta_t) * delta_dt = beta * exp(-beta*dt) * delta_dt

When both are perturbed by the same multiplicative factor, the additive changes
differ only by the ratio beta/delta_t (or delta_t/beta). They are not identical
but are strongly correlated when beta ~ delta_t.

### Mitigation (F2)

**Option A (recommended):** Add multi-step dynamics scenarios for dynamics
parameters. Instead of single-point ranking, run `simulate()` for 20 steps and
check whether the ranking at step 20 matches the expected winner. This makes
alpha, beta, delta_t, s_max, temperature, and feedback_sensitivity visible to
the analysis.

**Option B (minimum):** Document clearly that 8 parameters are
contraction-only, and split the report into "Ranking Sensitivity" and "Stability
Sensitivity" sections. Do not conflate the two in the classification.

**Option C (correlation):** Add a post-hoc correlation check: after computing
all SensitivityResults, compute pairwise correlation of
`contraction_margin_per_level` curves. If two parameters have Pearson r > 0.99,
flag them as "correlated -- cannot be independently assessed by OAT."

---

## F3. HIGH: Remaining Scenarios Are Mostly Non-Discriminative

Of the 6 scenarios with correct expected winners (after fixing F1), I tested all
36 (param, level) combinations per scenario:

| Scenario | Flips / 36 | Score Gap | Assessment |
|----------|-----------|-----------|------------|
| AR-2 | 0/36 | 26.1% | Too easy -- never flips |
| AR-3 | 3/36 | 7.1% | Marginally discriminative |
| SF-1 | 0/36 | 21.7% | Too easy -- never flips |
| SF-3 | 0/36 | 48.7% | Way too easy |
| TTL-1 | (3 memories, unchecked individually, but TTL-1 has 40% gap) | ~40% | Too easy |
| TTL-2 | 0/36 | 58% | Way too easy |

Only AR-3 is flippable, and only at extreme perturbation levels (-50% on w1,
-25% on w1, +50% on w2).

### Exclusion-Test (F3)

This is the "scenarios too easy" degenerate case. When every perturbation still
gets the right winner, `scenarios_correct_per_level` is always 6/6, and the
classification is always "insensitive" regardless of the parameter. The
sensitivity analysis would report everything as insensitive and we learn nothing.

### What makes a scenario discriminative?

A scenario is discriminative when the score gap between the expected winner and
the runner-up is **small relative to the score change caused by perturbation**.
Specifically, if the gap is G and the typical score change from a 50%
perturbation is delta_S, the scenario is discriminative only when
G < delta_S.

For weight perturbations at 50%, the maximum score change is roughly
`0.5 * w_i * max_component`. With w1=0.35 and max relevance=0.95, that's about
0.35 * 0.5 * 0.95 = 0.166. A scenario with gap > 0.17 cannot be flipped by any
single weight perturbation at 50%. AR-2 (gap=0.175), SF-1 (gap=0.16), SF-3
(gap=0.26), and TTL-2 (gap=0.22) all exceed this threshold.

### Mitigation (F3)

Add "margin-calibrated" scenarios where the score gap is deliberately set to be
small (< 0.05). These scenarios are designed to be flippable:

```python
# "Razor-thin margin" scenario
# M0 barely beats M1 at baseline -- any perturbation could flip
Memory 0: relevance=0.60, importance=0.50, ...
Memory 1: relevance=0.58, importance=0.52, ...
# Score gap target: < 0.02
```

Add at least 3-4 razor-thin scenarios across different competencies. These
should be designed so that different parameters flip different scenarios,
providing discriminative power.

---

## F4. MEDIUM: Kendall Tau Is Degenerate for 2-Memory Scenarios

7 out of 8 non-stability scenarios have exactly 2 memories. For a 2-element
ranking, Kendall tau can only be +1 (same order) or -1 (reversed). There are no
intermediate values.

This means the average tau across scenarios is essentially a weighted vote:
`mean_tau = (n_same - n_flipped) / n_total`. It provides exactly the same
information as `scenarios_correct`, just normalized differently.

### Object-Transpose (F4)

The prompt asked: "Would a simpler metric (did the winner change?) be more
informative?"

**Answer: YES, for this scenario suite.** With 7/8 scenarios having 2 memories,
Kendall tau degenerates to exactly the winner-change metric. The more complex
Kendall tau computation adds no information. Only TTL-1 (3 memories) can produce
intermediate tau values.

**Recommendation:** Either:
(a) Replace Kendall tau with the simpler "scenarios_correct" count (since they
    carry the same information for 2-memory scenarios), OR
(b) Add more scenarios with 3+ memories where tau provides genuinely graded
    ranking similarity information.

---

## F5. MEDIUM: Perfect Parameter Correlations

### alpha / s_max correlation

Both appear as a product in the contraction factor:
`K = exp(-beta*dt) + (0.25/T) * alpha * s_max`

A multiplicative perturbation of alpha by factor (1+f) shifts the term by:
`delta = (0.25/T) * alpha * f * s_max`

A multiplicative perturbation of s_max by factor (1+f) shifts the term by:
`delta = (0.25/T) * alpha * s_max * f`

These are **identical**. Numerically verified: for all 6 perturbation levels,
K(alpha perturbed) == K(s_max perturbed) to machine precision.

### beta / delta_t correlation

Both appear in `exp(-beta * delta_t)`. A multiplicative perturbation of beta by
(1+f) gives `exp(-beta*(1+f)*dt) = exp(-beta*dt) * exp(-beta*f*dt)`. The same
perturbation of delta_t gives `exp(-beta*dt*(1+f)) = exp(-beta*dt) * exp(-beta*dt*f)`.

Since beta*f*dt == beta*dt*f, these are also identical.

### Exclusion-Test (F5)

The OAT analysis will produce identical `contraction_margin_per_level` curves
for (alpha, s_max) and for (beta, delta_t). It will classify them identically.
This means OAT is blind to the question "which of these two is more important?"
-- it cannot answer it because they are perfectly confounded in the contraction
condition.

### Mitigation (F5)

Add a correlation-detection step in `run_sensitivity_analysis` or
`generate_report`. After computing all results, check pairwise Pearson
correlation of the contraction_margin curves. If r > 0.99, report:
"Parameters X and Y are perfectly correlated in their stability effect; OAT
cannot distinguish their individual contributions. Consider interaction analysis
(Morris screening or Sobol indices) if individual attribution is needed."

---

## F6. LOW: ParameterSet Conflates Two Independent Layers

### Object-Transpose (F6)

The prompt asked: "Would it be better to have DynamicsParams + ScoringWeights
as separate objects?"

**Analysis:** The spec correctly notes that dynamics parameters (alpha, beta,
delta_t, s_max, s0, temperature, feedback_sensitivity) and scoring weights
(w1-w4) serve different purposes. In fact, Finding F2 shows they don't even
interact in the single-point analysis.

However, some parameters bridge both: temperature affects contraction (dynamics)
AND soft_select (selection). Novelty parameters affect scoring directly.

**Recommendation:** Keep ParameterSet as a single object for now (it's the
correct abstraction for the contraction condition, which involves both layers).
But add a `scoring_params()` method that returns only the parameters relevant
to `score_memory`, and a `dynamics_params()` method for the dynamics-only
parameters. This helps the report clearly separate the two sensitivity
dimensions.

---

## F7. INFO: Tournament vs Sort Question Is Architecturally Moot

The prompt asked whether the pairwise tournament for N>2 memories produces the
same ranking as sorting by score.

**Answer:** This question is not applicable to the current design because:

1. `rank_memories` uses **sort** (deterministic) and returns a full ranking.
   It is used for Kendall tau computation.
2. `select_memory` uses **tournament** (stochastic) and returns a single winner.
   It is only used in stability scenarios (access_pattern = -1).

These are different functions for different purposes. The question of
tournament-vs-sort equivalence only matters if both were used to produce
rankings, which they are not.

The tournament in `select_memory` is intentionally different from sort: it
introduces stochasticity via `soft_select`, which is the whole point (preventing
deterministic lock-in). Whether it agrees with sort is irrelevant; it is
supposed to disagree probabilistically.

---

## Object-Transpose: OAT vs Morris Screening vs Sobol

### Assessment

| Method | Interactions | Cost | Information |
|--------|-------------|------|-------------|
| OAT | None | 14 * 6 = 84 evals | First-order only |
| Morris screening | First-order approx | ~140-280 evals | Elementary effects, interaction hints |
| Sobol (first order) | Full | ~1000+ evals | Variance decomposition |

For 14 parameters with the current scenario suite, OAT is sufficient **IF the
scenario suite is fixed** (Findings F1-F3). OAT misses parameter interactions,
but the most important interaction (alpha * s_max in contraction) can be
detected by the correlation check in F5.

Morris screening would provide elementary effects mu* and sigma, where sigma
indicates interaction strength. This would catch the alpha/s_max correlation
automatically. Cost is ~2x OAT.

**Recommendation:** Keep OAT for the initial implementation. Add a
`detect_correlated_parameters` post-processing step (cost: zero additional
simulations, just compare existing results). If the report reveals too many
"insensitive" classifications, upgrade to Morris screening in a follow-up.

---

## Object-Transpose: Perturbation in Contraction-Margin Coordinates

### Assessment

The prompt asked: would it be better to perturb in terms of contraction
condition margin rather than raw percentage?

For the 8 parameters that only affect contraction (not scoring), this is the
**only meaningful perturbation coordinate**. Perturbing alpha by 50% vs
perturbing temperature by 50% produces wildly different margin changes:

```
alpha +50%:  K = 0.9048 + 0.075 = 0.9798,  margin = 0.0202
temperature -50%: K = 0.9048 + 0.1 = 1.0048,  margin = -0.0048  (VIOLATED)
```

Temperature at -50% already breaks contraction, while alpha at +50% doesn't.
But this says more about the baseline operating point than the parameter's
intrinsic sensitivity.

**Recommendation:** Add a `margin_perturbation_levels` option: instead of
"perturb parameter by X%", compute "perturb parameter until margin decreases by
X%". This normalizes the perturbation scale across parameters and makes the
contraction sensitivity analysis more comparable.

---

## Object-Transpose: Skip Soft Selection for Sensitivity Analysis

### Assessment

The prompt asked whether we should skip soft_select entirely for sensitivity
analysis (just rank by score).

For non-stability scenarios, `analyze_parameter` already uses `rank_memories`
(sort-based), not `select_memory` (tournament-based). Soft selection is only
involved in stability scenarios (STAB-1, STAB-2) which use expected_winner=-1
and are excluded from the tau/winner computations.

**So the answer is: the spec already does this.** Soft selection does not
participate in the ranking sensitivity metrics. It only appears in stability
scenario evaluation, which is a separate code path.

---

## Summary of Required Spec Changes

### Must Fix (blocks implementation correctness)

1. **AR-1 scenario parameters:** Adjust M0 strength or last_access_time so that
   M0 actually wins. Suggested: M0 last_access_time=2.0, strength=s_max*0.5.
   Verify numerically.

2. **SF-2 scenario parameters:** Adjust M1 last_access_time so that M1's
   recency is actually degraded. Suggested: M1 last_access_time=50.0.
   Verify numerically.

### Should Fix (avoids misleading results)

3. **Add razor-thin scenarios:** At least 3 scenarios with score gap < 0.05
   across different competencies. Without these, the sensitivity analysis
   reports "everything is insensitive" for the 6 scoring-relevant parameters.

4. **Add multi-step dynamics scenarios:** At least 2 scenarios that use
   `simulate()` to make dynamics parameters (alpha, beta, delta_t, s_max,
   feedback_sensitivity) visible to the ranking analysis.

5. **Add degenerate-case detection:** In `run_sensitivity_analysis`, check if
   all parameters show tau=1.0 and score_change=0. If so, warn that the
   scenario suite is non-discriminative.

6. **Add correlation detection:** Check pairwise correlation of
   contraction_margin curves. Report perfectly correlated parameter pairs.

### Nice to Have (design improvements)

7. Replace Kendall tau with scenarios_correct for 2-memory scenarios, or add
   more 3+ memory scenarios.

8. Add margin-relative perturbation mode for contraction-sensitive parameters.

9. Add DynamicsParams/ScoringWeights accessor methods to ParameterSet.
