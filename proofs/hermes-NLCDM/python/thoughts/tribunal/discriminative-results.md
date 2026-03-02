# Discriminative Validation Results — Phases 8-10

Date: 2026-03-02
Tribunal: 3 parallel agents (test designer, adversarial critic, hypothesis generator)

## Verdicts

| Hypothesis | Mathematical | Operational | Verdict |
|-----------|-------------|-------------|---------|
| H1: Bridge Formation | VALIDATED | VALIDATED | **VALIDATED** |
| H2: Conditional Monotonicity | VALIDATED | VACUOUS | **VALIDATED (vacuous)** |
| H3: Transfer Dynamics | VALIDATED | QUESTIONABLE at scale | **INCONCLUSIVE** |

## H1: Bridge Formation (Phase 8) — VALIDATED

**All three agents agree.** The mathematical claim holds:

- Raw dot product of centroid with target = mean of constituent dot products (linearity)
- Lean bound: `dot(mean(v_k), target) >= sigma` when all `dot(v_k, target) >= sigma` ✓
- Python normalization **strengthens** the bound: `||raw_centroid|| <= 1` for unit vectors, so `cosine(norm_centroid, target) = dot/||c|| >= dot >= sigma` ✓
- Measured: raw dot = 0.650000, normalized cosine = 0.694250 (normalization helps)

**Caveats (non-killing):**
- Bridges are average-strength, not amplified (H1-A confirmed)
- For orthogonal source/target, max equal cosine sim is 1/sqrt(2) = 0.707 (geometric constraint)
- Centroid bridges have smaller alignment gaps than peaked individual patterns (H1-B)

**Decision: Proceed.** No kill signal. The normalization gap is resolved (normalization preserves bounds).

## H2: Conditional Bridge Monotonicity (Phase 9b) — VALIDATED (operationally vacuous)

**Mathematical correctness confirmed.** When P ⊆ T (all protected survive), bridges within P are exactly preserved (not just non-decreasing — the proof gives equality).

**Three operational concerns (consensus across agents):**

1. **Tautology problem** (architect + critic agree): The theorem proves "if you don't change the patterns, their dot products don't change." All difficulty is pushed into the precondition P ⊆ T, which is *assumed*, not proved.

2. **Precondition violation** (test designer confirmed): `nrem_prune_xb` DOES remove high-importance near-duplicates. Test: two patterns with importance 0.95 and 0.90, sim=0.993 — the 0.90 pattern was removed. P ⊆ T fails when protected patterns are near-duplicates.

3. **Bridge count within P likely 0** (architect + critic agree): Important memories are semantically diverse → low pairwise similarity → typically 0 bridges within the protected set. The guarantee is vacuously satisfied.

**Decision: Proceed.** The theorem is correct and its conditional nature is explicitly stated. The vacuousness is a design insight, not a bug: the real protection comes from importance-based survival, not bridge counting. Flag for documentation.

## H3: Transfer Dynamics (Phase 10) — INCONCLUSIVE

**Mathematical bound validated in all tests.** 18/18 pass across N ∈ {5..1000}. But three agents raised serious concerns:

### What holds:
- Softmax weight bound `w_mu >= 1/(1+(N-1)*exp(-beta*delta))` ✓ (all N tested)
- Dot product decomposition `dot(T(xi), target) = sum w_mu * dot(x_mu, target)` ✓
- High-beta limit: at β=100, transfer approaches sigma exactly (difference: 1.1e-16) ✓
- Dot-to-cosine relationship: `cosine >= dot` when `||T(xi)|| <= 1` ✓

### What's concerning:

**1. Bound collapse at scale (critic RED ALERT, architect H3-A):**
```
N=10:   bound = 0.40     (useful)
N=100:  bound = 0.077    (weak)
N=1000: bound = 0.024    (dot) but cosine = 0.758 (direction preserved!)
```
The dot product bound degrades as O(sigma/N), but the test designer discovered that cosine similarity stays strong because ||T(xi)|| shrinks proportionally. **The theorem bounds the wrong quantity** — direction (cosine) matters more than magnitude (dot product) for retrieval.

**2. Alignment gap precondition fails for cross-domain queries (architect H3-A, CRITICAL):**
The theorem requires delta > 0 (bridge pattern has highest alignment with query). For cross-domain queries from domain A, domain-A-specific patterns will have *higher* alignment with the query than the bridge centroid. Delta < 0 precisely when you need the theorem most.

This was NOT directly tested (would require realistic domain-specific patterns). The test designer's tests use synthetic queries constructed to satisfy delta > 0 — a false positive path the critic warned about.

**3. Non-negativity assumption unrealistic (architect H3-D):**
The theorem requires `hothers_nn: forall nu, dot(x_nu, target) >= 0`. Random unit vectors have ~50% negative dot products. Test designer confirmed: when precondition is violated, the bound can still hold empirically but is no longer guaranteed.

**4. Implementation divergence (architect H3-C, HIGH):**
The Python pipeline discovers cross-domain associations via `rem_explore_cross_domain_xb` (perturbation-response correlation), NOT via Hopfield dynamics. The formal proofs analyze mechanism A; the code uses mechanism B. These may or may not correlate.

### Tribunal disagreement:
- Test designer: VALIDATED (bound holds in all tests)
- Critic: RED (bound useless at scale, tests give false confidence)
- Architect: CRITICAL (precondition fails for real cross-domain queries, implementation divergence)

**Decision: INCONCLUSIVE.** The mathematical bound is correct but:
1. It bounds dot product when cosine matters (pessimistic by factor 1/||T(xi)||)
2. The precondition delta > 0 likely fails for genuine cross-domain queries
3. The Python code uses a different mechanism than what's proven

**Recommended follow-up:**
- Test with realistic domain embeddings (not synthetic) to measure actual delta values
- Compare perturbation-correlation associations with Hopfield transfer predictions
- Consider proving a cosine-based bound (would be much tighter)
- The theorem is not wrong — it's a valid but potentially vacuous guarantee at scale

## Key Findings for Architecture

1. **Normalization is safe.** The gap between Lean (unnormalized) and Python (normalized) closes in favor of Python — normalization preserves or strengthens bounds.

2. **Conditional monotonicity is a documentation concern, not a bug.** The theorem correctly states its conditional nature. The operational insight: importance-based survival is what matters, not bridge counting within the protected set.

3. **Transfer dynamics needs a tighter theorem.** The current bound is provably correct but O(1/N)-pessimistic. Two paths forward:
   - Prove a cosine-based bound (eliminates the ||T(xi)|| penalty)
   - Prove a multi-bridge bound (multiple bridges compound the signal)
   - Accept the bound as a worst-case guarantee and rely on empirical β → ∞ behavior

4. **Implementation divergence is the most actionable finding.** The formal proofs validate Hopfield transfer, but the code uses perturbation-response correlation. Either:
   - Prove that the two mechanisms correlate
   - Formally verify the actual mechanism
   - Accept the proofs as "theoretical foundations" rather than "implementation guarantees"

## Files Produced

- `test_bridge_integration.py` — 18 discriminative tests (all passing)
- `thoughts/tribunal/hypotheses.md` — competing hypotheses from architect
- `thoughts/tribunal/critique.md` — adversarial critique (delivered in agent output, not saved to file)
- `thoughts/tribunal/discriminative-results.md` — this synthesis
