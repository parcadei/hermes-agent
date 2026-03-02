# Adversarial Review Pass 1: Consolidation Spec

**Reviewer:** architect-agent (adversarial)
**Date:** 2026-02-27
**Spec:** `/thoughts/shared/plans/memory-system/consolidation/spec.md`
**Operators:** Level-Split, Paradox-Hunt

---

## Finding 1: Paradox-Hunt -- ConsolidationResult Invariant Self-Contradiction

**Spec text (Section 3.7, lines 493-510):**

First block:
```
consolidated_count == len(archived)
consolidated_count == len(extractions)   # each archived memory produces exactly one extraction
consolidated_count >= 0
```

Then immediately corrected:
```
consolidated_count == len(archived)
len(extractions) <= len(archived)        # clusters: N sources -> 1 extraction
len(extractions) >= 1 if consolidated_count >= 1
consolidated_count >= 0
```

**Problem:** The spec contains TWO contradictory invariant blocks for the same dataclass. The first says `consolidated_count == len(extractions)` (1:1 ratio). The second says `len(extractions) <= len(archived)` (N:1 for clusters). Both cannot be true simultaneously when clustering produces N>1 archived memories per extraction. The "Note on the 1:1 invariant" paragraph acknowledges the contradiction but leaves both versions in the spec, creating an ambiguous contract.

This is a textbook Paradox-Hunt: the spec contradicts itself within the same section. The first invariant block is dead text that was superseded but never removed.

**Fix applied:** Remove the first (incorrect) invariant block and keep only the corrected version. Add explicit documentation that the original was superseded.

---

## Finding 2: Level-Split -- Retention Threshold vs Level-Specific Threshold Confusion

**Spec text (Section 2.1 vs Section 2.3 vs Section 3.3):**

Section 2.1 defines level-specific thresholds:
```
Level 1: consolidate when retention < 0.4
Level 2: consolidate when retention < 0.3
```

Section 2.3 defines a generic trigger:
```
retention < config.retention_threshold (default 0.3)
```

Section 3.3 defines level-specific config fields:
```
l1_to_l2_retention: float = 0.4
l2_to_l3_retention: float = 0.3
```

But Section 5.1 (`should_consolidate()`) uses only the GENERIC threshold:
```python
current_retention < config.retention_threshold   # 0.3
```

It NEVER uses `l1_to_l2_retention` (0.4) or `l2_to_l3_retention` (0.3).

**Problem:** This is a Level-Split confusion between INTERFACE (config fields exist for level-specific thresholds) and IMPLEMENTATION (the algorithm ignores them and uses only the generic threshold). The config defines `l1_to_l2_retention = 0.4` and `l2_to_l3_retention = 0.3`, but `should_consolidate()` never branches on `candidate.level` to select the appropriate threshold. An L1 memory will be evaluated against 0.3 (the generic), not 0.4 (the level-specific).

This means:
- L1 memories need to decay MORE than the diagram suggests (0.3 vs 0.4) before consolidation triggers.
- The `l1_to_l2_retention` and `l2_to_l3_retention` config fields are defined, validated, and documented but NEVER USED by any function in the spec.
- The validation rule `l2_to_l3_retention <= l1_to_l2_retention` is enforced but meaningless since neither field is referenced in any algorithm.

**Fix applied:** Modify `should_consolidate()` to use level-specific thresholds. Map `candidate.level` to the appropriate threshold from config. Fall back to `config.retention_threshold` for levels not covered (L3->L4 uses cluster similarity, not retention).

---

## Finding 3: Paradox-Hunt -- Intra-Session Temporal Threshold Unit Collision

**Spec text (Section 2.2 vs Section 3.3):**

Section 2.2:
```
temporal proximity < 1 hour as measured by creation_time difference
```

Section 3.3:
```
intra_session_temporal: float = 1.0
    Domain: > 0.0. Default: 1.0.
```

Meanwhile, Section 2.3:
```
recency_guard: float = 1.0  (default 1.0 time unit, representing ~24 hours)
```

**Problem:** The spec says `1.0 time unit = ~24 hours` (in the recency_guard description), but `intra_session_temporal = 1.0` is described as `< 1 hour`. These cannot both be true. If 1.0 time unit represents ~24 hours, then `intra_session_temporal = 1.0` means "within 24 hours", not "within 1 hour." If it means 1 hour, then the recency_guard of 1.0 protects memories younger than 1 hour, not 24 hours.

The time unit system is ambiguous: `creation_time` is described as "Time steps since memory was created" (Section 3.4) but the mapping from time steps to wall-clock time is inconsistent. The spec uses "~24 hours" in one place and "1 hour" in another for the same unit value.

**Fix applied:** Remove the wall-clock annotations entirely. Time units are abstract and caller-defined. The spec should not embed a specific wall-clock mapping that it then contradicts. Document that the relationship between time units and wall-clock time is the caller's responsibility.

---

## Finding 4: Level-Split -- Config as Interface vs Config as Behavior

**Spec text (Section 3.3):**

`ConsolidationConfig` contains 21 fields covering:
1. Trigger thresholds (5 fields)
2. Inhibitor thresholds (2 fields)
3. Level transition thresholds (3 fields)
4. Process parameters (3 fields)
5. Clustering parameters (3 fields)
6. Intra-session parameters (2 fields)
7. General parameters (1 field)
8. Plus the unused `retention_threshold` which overlaps with `l1_to_l2_retention`/`l2_to_l3_retention`

**Problem:** The config conflates TWO distinct configuration concerns:
- **Per-function config**: thresholds used by specific functions (e.g., `semantic_weight_beta` is ONLY used by `compute_affinity`, `intra_session_similarity` is ONLY used by integration code).
- **Cross-cutting config**: thresholds used across multiple functions (e.g., `recency_guard` is used by both `should_consolidate` and `select_consolidation_candidates`).

Moreover, `intra_session_similarity` and `intra_session_temporal` are config fields used ONLY by the integration code (Section 7.3), NOT by any function defined in the consolidation module. They are dead config from the module's perspective -- the module defines them, validates them, but never reads them. The integration code reads them.

This is a Level-Split between "config fields the module owns" and "config fields the module carries for others."

**Fix applied:** Add documentation in the ConsolidationConfig docstring clarifying which fields are used by which functions, and explicitly mark `intra_session_similarity` and `intra_session_temporal` as integration-facing config fields. This is less disruptive than splitting the config class while still resolving the confusion.

---

## Finding 5: Paradox-Hunt -- Correction Inhibitor vs Correction in CATEGORY_COMPRESSION

**Spec text (Section 2.3, Section 4.2, Section 4.6):**

Section 2.3, Inhibitor 4:
```
Category is "correction": corrections preserve full context unconditionally (compression ratio 1:1)
```

Section 4.2:
```
"correction": 1.0,    # DO NOT compress -- preserve full context
```

Section 4.6:
```
NON_CONSOLIDATABLE_CATEGORIES: frozenset[str] = frozenset({"correction"})
```

The spec says corrections are NEVER consolidated (inhibitor blocks them). Then it also defines a compression ratio for corrections (1.0) and says this is "defense-in-depth."

Section 4.2 explanation:
```
This is enforced by the inhibitor check (Section 2.3, inhibitor 4) which
prevents corrections from entering the consolidation pipeline at all. The
ratio 1.0 is a defense-in-depth measure.
```

And Section 5.3 algorithm explicitly handles corrections:
```python
if category == "correction":
    formatted = content  # Corrections: preserve full content
```

**Problem:** This is not a true paradox -- the spec explicitly calls it defense-in-depth. However, there IS a real paradox hiding here: `extract_semantic()` accepts `category="correction"` without raising an error, even though `should_consolidate()` would never allow a correction to reach it. The function-level contract says "category in VALID_CATEGORIES" (which includes correction), but the pipeline-level contract says "corrections never enter the pipeline." If a caller invokes `extract_semantic(content, "correction", ...)` directly (bypassing `should_consolidate`), it succeeds silently. Is this intentional? The spec is internally consistent on this point (defense-in-depth), but the behavioral contract for `extract_semantic` should state this explicitly.

**Fix applied:** Add a behavioral contract note to `extract_semantic()` documenting that `"correction"` is accepted but should not reach this function under normal pipeline flow.

---

## Finding 6: Level-Split -- `creation_time` Semantics: Age vs Timestamp

**Spec text (Section 3.4 vs Section 5.1):**

Section 3.4 (ConsolidationCandidate):
```
creation_time: Time steps since memory was created (from MemoryState).
```

Section 5.1, Inhibitor I2:
```python
if candidate.creation_time < config.recency_guard:
    return False
```

This reads as: "if time steps since creation < 1.0, block." So `creation_time` is an AGE (duration since birth, monotonically increasing).

But in `engine.py` (MemoryState, line 211):
```
creation_time: Time steps since creation. Domain: [0, +inf).
```

And `last_access_time` (MemoryState, line 207):
```
last_access_time: Time steps since last accessed. Domain: [0, +inf).
```

**Problem:** `creation_time` as "time steps SINCE creation" is an age that increases over time. But `last_access_time` as "time steps SINCE last accessed" is also an age that increases. In the spec, `retention(last_access_time, strength)` is called where `retention(t, S) = exp(-t/S)`. This is correct: higher `last_access_time` = more time passed = lower retention.

The inhibitor `creation_time < recency_guard` means "younger than guard" = "recently created." This is semantically correct.

But `SemanticExtraction.first_observed` is computed as `min(source_creation_times)` and `last_updated` as `max(source_creation_times)`. If `creation_time` is an AGE (increases over time), then `min(creation_times)` = the YOUNGEST source, and `max(creation_times)` = the OLDEST source. This is BACKWARDS from the field names: `first_observed` should be the OLDEST (earliest), not the youngest.

This is a Level-Split between two interpretations of `creation_time`:
- **Interpretation A (age):** creation_time = time elapsed since birth. Older memories have LARGER values. `first_observed = max(ages)` = oldest.
- **Interpretation B (timestamp):** creation_time = absolute timestamp of birth. Older memories have SMALLER values. `first_observed = min(timestamps)` = oldest.

The inhibitor check `creation_time < recency_guard` is consistent with Interpretation A (age). But `first_observed = min(creation_times)` is consistent with Interpretation B (timestamp). The spec uses BOTH interpretations simultaneously.

**Fix applied:** Resolve in favor of Interpretation A (age), consistent with `engine.py`'s MemoryState which uses "time steps since" for both fields. Swap `first_observed = max(source_creation_times)` and `last_updated = min(source_creation_times)` in the `extract_semantic()` algorithm.

---

## Finding 7: Paradox-Hunt -- `consolidate_memory()` Mode is Always RETRIEVAL_TRIGGERED

**Spec text (Section 5.4):**

```python
mode=ConsolidationMode.RETRIEVAL_TRIGGERED,  # default for single
```

And in the behavioral contracts:
```
The result's mode defaults to RETRIEVAL_TRIGGERED for single-memory consolidation.
Batch callers should override this.
```

**Problem:** `ConsolidationResult` is a frozen dataclass. The `mode` field is set at construction time. There is no way for "batch callers to override" the mode after construction because the dataclass is frozen. The only way to change mode would be to construct a NEW `ConsolidationResult`, but `consolidate_memory()` returns the result with `RETRIEVAL_TRIGGERED` already baked in.

The spec says callers "should override this" but provides no mechanism to do so. The `consolidate_memory()` function does not accept a `mode` parameter. The result is immutable.

This is a Paradox-Hunt: the spec promises a capability (mode override) that is structurally impossible given the type system (frozen dataclass) and the function signature (no mode parameter).

**Fix applied:** Add `mode: ConsolidationMode = ConsolidationMode.RETRIEVAL_TRIGGERED` parameter to `consolidate_memory()`, allowing callers to specify the mode. This resolves the paradox by providing the mechanism the spec promises.

---

## Finding 8: Paradox-Hunt -- `l3_to_l4_cluster_sim` Has No Function

**Spec text (Section 2.1 vs Section 3.3 vs all function algorithms):**

Section 2.1:
```
Level 3 (SEMANTIC_FACT): consolidate when cluster similarity > 0.8
```

Section 3.3:
```
l3_to_l4_cluster_sim: float = 0.8
    Domain: (0.0, 1.0]. Default: 0.8.
```

**Problem:** No function in Section 5 (the complete function listing) uses `l3_to_l4_cluster_sim`. The `should_consolidate()` function checks retention and semantic value but never checks cluster similarity. The `compute_affinity()` function uses `cluster_threshold` (0.75) for general clustering. The `select_consolidation_candidates()` function never references `l3_to_l4_cluster_sim`.

Like Finding 2, this is a config field that is defined, validated, and documented but NEVER REFERENCED by any algorithm in the spec. The L3->L4 transition is described in the overview as requiring "cluster similarity > 0.8" but no function implements this check.

**Fix applied:** Document `l3_to_l4_cluster_sim` as integration-facing (the batch clustering code in Section 7.2 would use it). Add a note to the `select_consolidation_candidates()` behavioral contracts explaining that L3->L4 promotion is handled by the clustering integration code, not by `should_consolidate()`.

---

## Finding 9: Level-Split -- `CATEGORY_COMPRESSION` vs `CATEGORY_IMPORTANCE` Coverage Mismatch

**Spec text (Section 4.2 vs encoding.py):**

Spec Section 4.2 `CATEGORY_COMPRESSION`:
```python
{
    "preference": 10.0,
    "fact": 5.0,
    "reasoning": 3.0,
    "instruction": 5.0,
    "correction": 1.0,
    "greeting": 10.0,
    "transactional": 10.0,
}
```

encoding.py `CATEGORY_IMPORTANCE`:
```python
{
    "correction": 0.9,
    "instruction": 0.85,
    "preference": 0.8,
    "fact": 0.6,
    "reasoning": 0.4,
    "greeting": 0.0,
    "transactional": 0.0,
}
```

encoding.py `VALID_CATEGORIES`:
```python
frozenset({"preference", "fact", "correction", "reasoning", "instruction", "greeting", "transactional"})
```

**Problem:** The coverage is actually aligned (all 7 categories in both dicts). No paradox here. However, `_compute_semantic_value()` uses `CATEGORY_IMPORTANCE[candidate.category]` with weight 0.6. For `greeting` and `transactional`, the category importance is 0.0, yielding:

```
semantic_value = clamp01(0.0 * 0.6 + importance * 0.3 + access_bonus * 0.1)
              = clamp01(importance * 0.3 + access_bonus * 0.1)
              <= 0.3 + 0.1 = 0.4
```

This means `semantic_value` for greeting/transactional can NEVER reach 0.6 (the `semantic_value_threshold`). Therefore these categories can NEVER pass the primary trigger in `should_consolidate()`, and also cannot pass the secondary trigger (`semantic_value >= config.semantic_value_threshold`).

This means greeting and transactional memories are de facto non-consolidatable through the semantic value gate, even though they are NOT in `NON_CONSOLIDATABLE_CATEGORIES`. Section 4.6 says: "If they somehow reach consolidation, they ARE eligible for consolidation." But they mathematically CANNOT reach consolidation through `should_consolidate()`. The only path is INTRA_SESSION mode in `select_consolidation_candidates()`, which skips `should_consolidate()`.

This is not exactly a paradox (the math is consistent) but it IS a Level-Split between the stated POLICY ("greeting/transactional are eligible") and the actual BEHAVIOR ("they can never pass the semantic value gate"). The spec should acknowledge this explicitly rather than implying they are merely "rare."

**Fix applied:** Add a note to Section 4.6 documenting that greeting/transactional are de facto blocked by the semantic value gate in ASYNC_BATCH and RETRIEVAL_TRIGGERED modes, and only eligible via INTRA_SESSION mode. This makes the behavioral reality match the documentation.

---

## Finding 10: Paradox-Hunt -- `consolidation_decay` Applied to Strength Without Clamping

**Spec text (Section 5.7):**

```python
decayed_importance = clamp01(candidate.importance * config.consolidation_decay)
decayed_strength = candidate.strength * config.consolidation_decay
```

Invariant from Section 3.6:
```
decayed_strength >= 0.0
decayed_importance <= original_importance   # decay never increases
decayed_strength <= original_strength       # decay never increases
```

**Problem:** `importance` is clamped via `clamp01()` but `strength` is NOT clamped. Given `consolidation_decay` is in `(0.0, 1.0)` and `strength >= 0.0`, the multiplication `strength * consolidation_decay` always produces a non-negative result smaller than the original. So the invariants hold mathematically. No paradox here.

However, there is a subtle issue: `clamp01` on importance is unnecessary. If `importance` is in `[0, 1]` (guaranteed by ConsolidationCandidate validation) and `consolidation_decay` is in `(0, 1)`, then `importance * consolidation_decay` is already in `[0, 1)`. The `clamp01` is a no-op. This is defense-in-depth, so acceptable -- but worth noting that the asymmetric treatment (clamp importance, don't clamp strength) could confuse implementers. No fix needed, but documented.

---

## Summary of Fixes Applied to spec.md

| # | Operator | Finding | Fix |
|---|----------|---------|-----|
| 1 | Paradox-Hunt | ConsolidationResult has two contradictory invariant blocks | Remove dead first block |
| 2 | Level-Split | Level-specific retention thresholds defined but unused | Wire `should_consolidate()` to use level-specific thresholds |
| 3 | Paradox-Hunt | Time unit 1.0 = "~24 hours" AND "1 hour" | Remove wall-clock annotations; time units are abstract |
| 4 | Level-Split | Config carries fields for other modules | Document which fields are module-internal vs integration-facing |
| 5 | Paradox-Hunt | `extract_semantic` silently accepts corrections | Add behavioral contract note |
| 6 | Level-Split | `creation_time` used as both age and timestamp | Swap min/max in extract_semantic for `first_observed`/`last_updated` |
| 7 | Paradox-Hunt | Mode override promised on frozen dataclass | Add `mode` parameter to `consolidate_memory()` |
| 8 | Paradox-Hunt | `l3_to_l4_cluster_sim` defined but unused | Document as integration-facing; note in behavioral contracts |
| 9 | Level-Split | Greeting/transactional "eligible" but mathematically blocked | Document the semantic value gate behavior |
| 10 | (observation) | Asymmetric clamping in archive | Documented, no fix needed |

