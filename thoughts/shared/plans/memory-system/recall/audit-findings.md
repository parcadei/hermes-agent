# Audit Findings: hermes_memory/recall.py

**Date:** 2026-02-27
**Verdict:** PASS (after fixes)
**Implementation:** 879 lines
**Tests:** 201 (560 total suite)

## Issues Found and Resolved

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 1 | CRITICAL | Default high_max_chars=500/300/100 deviated from spec 400/200/80 | FIXED |
| 2 | HIGH | Undocumented multi-word trivial gating bypass | FIXED (removed) |
| 3 | MEDIUM | 1e-9 tolerance for tier thresholds not documented in spec | Acceptable |
| 4 | MEDIUM | should_recall check order differs from spec (equivalent results) | Acceptable |
| 5 | MEDIUM | Epsilon calibration warning not in code comments | Noted |
| 6 | LOW | Positional fallback comment lacks rationale | Noted |
| 7 | LOW | _truncate_to_limit doesn't handle max_chars < 3 | Noted |

## Adversarial Mitigations Verified (28/28)

- AP1-F1: budget_overflow field — VERIFIED
- AP1-F3: Positional fallback for identical scores — VERIFIED
- AP1-F4: contents=None normalization — VERIFIED
- AP1-F5: Empty-memories fast-path before gate — VERIFIED
- AP1-F7: Threshold boundary strictly above — VERIFIED
- AP2-F1: chars_per_token configurable — VERIFIED
- AP2-F3: Top-slice truncation in adaptive_k — VERIFIED
- AP2-F4: MIN_DEMOTION_SAVINGS_TOKENS — VERIFIED
- AP3-E1: No temperature dependency — VERIFIED
- AP3-T3: Demotion skip optimization — VERIFIED

## Final Test Results

```
560 passed, 19 deselected (slow) in 78.82s
```

No regressions across encoding.py (227), optimizer.py (132), or recall.py (201) tests.
