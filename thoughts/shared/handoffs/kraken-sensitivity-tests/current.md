---
root_span_id: 7afefa71-de5f-4cd7-82ba-be5760689687
turn_span_id:
session_id: 7afefa71-de5f-4cd7-82ba-be5760689687
---

# Kraken: Sensitivity Analysis Tests

## Checkpoints
<!-- Resumable state for kraken agent -->
**Task:** Build engine.py and fix tests per adversarial review findings
**Started:** 2026-02-27T00:00:00Z
**Last Updated:** 2026-02-27T12:00:00Z

### Phase Status
- Phase 1 (Tests Written): VALIDATED (46 sensitivity tests, all fail with ImportError as expected)
- Phase 1b (Test Fixes): VALIDATED (test_engine.py and test_sensitivity.py fixed per adversarial findings)
- Phase 2 (Engine Implementation): VALIDATED (52 engine tests passing, 270 total tests passing)
- Phase 3 (Sensitivity Implementation): PENDING

### Validation State
```json
{
  "test_count_engine": 52,
  "tests_passing_engine": 52,
  "test_count_total": 270,
  "tests_passing_total": 270,
  "files_created": ["proofs/hermes-memory/python/hermes_memory/engine.py"],
  "files_modified": [
    "proofs/hermes-memory/python/tests/test_engine.py",
    "proofs/hermes-memory/python/tests/test_sensitivity.py"
  ],
  "last_test_command": "cd proofs/hermes-memory/python && .venv/bin/python -m pytest tests/ -q --tb=short -m 'not slow' --ignore=tests/test_sensitivity.py",
  "last_test_exit_code": 0,
  "existing_tests_pass": true
}
```

### Resume Context
- Current focus: Engine implementation complete and validated
- Next action: Implement hermes_memory/sensitivity.py (Phase 3)
- Blockers: None

### Test Fixes Applied (Adversarial Findings)
1. test_select_large_score_gap_favors_top: T=0.5 with contraction-safe params, 1000 trials
2. test_contraction_violating_params_raise: Expects ValueError on construction
3. Reference params standardized: N0=0.5, gamma=0.2, w2=0.25, w3=0.20
4. test_epsilon_exceeds_novelty_start_raises: survival_threshold=0.5
5. AR-1 scenario: M0 lat=5.0, strength=3.0; M1 lat=20.0, ac=2
6. REFERENCE_PARAMS in test_sensitivity.py: Standardized to match
