# Adversarial Review: CMA-ES Optimizer (Pass 1)

Date: 2026-02-27

## Summary
- 4 CRITICAL, 8 HIGH, 7 MEDIUM, 4 LOW findings
- All 4 CRITICAL findings addressed in spec and tests

## Critical Findings (FIXED)
1. Lower bounds produce infeasible ParameterSet — DOCUMENTED (expected, penalty approach)
2. Center-of-bounds test was vacuous — FIXED (added value assertions)
3. Sign convention ambiguous — FIXED (documented in spec)
4. Stability check logic undefined — FIXED (explicit algorithm in spec, stronger test)

## High Findings (ADDRESSED)
5. Upper bounds guarantee w3 < 0 — DOCUMENTED (expected, simplex constraint)
6. Contraction margin test has loose bounds — FIXED (tightened to 0.004-0.005)
7. Stability scenario count not enforced — DEFERRED (will enforce in build_benchmark_scenarios)
8. Held-out scenarios not verified different — DEFERRED (will verify in implementation)
9. Margin bonus cap creates flat region — ACCEPTED (0.3 cap is deliberate; most candidates have margin < 0.3)
10. Reproducibility test only checks final — FIXED (added history check)
11. Scenario count unverified — DEFERRED (will be constrained by implementation)
12. Per-competency accuracy not cross-checked — DEFERRED (test already verifies structure)

## Test Additions
- PARAM_ORDER invariant test (Finding 20)
- Stability comparison test (monopoly vs balanced)
- Empty scenarios range assertion (Finding 15)
- History reproducibility check (Finding 10)
- Center-of-bounds value assertions (Finding 2)
- Tighter margin bounds (Finding 6)
