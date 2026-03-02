# Adversarial Review -- Pass 2 of 3: Scale-Check + Materialize

**Reviewer:** architect-agent (Pass 2)
**Date:** 2026-02-27
**Target:** `/thoughts/shared/plans/memory-system/consolidation/spec.md`
**Operators:** Scale-Check, Materialize

---

## Finding 1: Half-Life Terminology Mismatch (Materialize)

**Operator:** Materialize -- "What would I actually SEE if I set strength = LEVEL_HALF_LIVES?"

The spec calls `LEVEL_HALF_LIVES` values "half-lives" (Section 2.1: "Retention half-life: 7 days").
Section 6 says: `new_strength = LEVEL_HALF_LIVES[target_level]`.

But `retention(t, S) = exp(-t/S)`. If S = 7.0:

```
R(7) = exp(-7/7) = exp(-1) = 0.3679
```

This is a **1/e-life** (36.8% retention), not a half-life (50% retention). The true half-life at S=7.0 is:

```
R(t) = 0.5 => t = S * ln(2) = 7.0 * 0.693 = 4.85 time units
```

Anyone reading "half-life: 7 days" expects R(7)=0.5. They get R(7)=0.368. The threshold calibration in Section 2.1 (`consolidate when retention < 0.4`) was likely set assuming R(7)=0.5, which means:

- At S=7.0, retention < 0.4 occurs at t > 6.41 (not t=7 as intuited)
- At S=7.0, retention < 0.3 occurs at t > 8.43
- The consolidation_window (7.0) is the binding constraint for L1, not the retention threshold

**Fix applied:** Rename `LEVEL_HALF_LIVES` to `LEVEL_TIME_CONSTANTS` throughout and add a comment explaining the e-folding relationship. Add a `LEVEL_HALF_LIVES` alias computed as `time_constant * ln(2)` for documentation purposes.

---

## Finding 2: Greeting/Transactional Categories Cannot Consolidate via Primary Path (Scale-Check)

**Operator:** Scale-Check -- "What categories actually reach semantic_value >= 0.6?"

The semantic_value formula is:
```
sv = cat_imp * 0.6 + importance * 0.3 + access_bonus * 0.1
```

For `greeting` and `transactional`, `CATEGORY_IMPORTANCE = 0.0`:
```
sv_max = 0.0 * 0.6 + 1.0 * 0.3 + 1.0 * 0.1 = 0.400
```

**Even with maximum importance (1.0) and maximum access bonus (20+ accesses), greeting/transactional semantic_value = 0.400 -- permanently below the 0.6 threshold.** These categories can NEVER consolidate through the primary trigger path.

They can only consolidate under storage pressure (Finding 8), which bypasses the semantic_value gate. The spec does not acknowledge this de-facto exclusion or its implications: without storage pressure, greeting/transactional memories simply decay to zero retention without ever being archived or producing semantic extractions.

**Fix applied:** Document this as an explicit behavioral contract in Section 5.2 and add to edge cases.

---

## Finding 3: Reasoning Category Has Near-Impossible Consolidation Bar (Scale-Check)

**Operator:** Scale-Check -- "What importance level does reasoning need?"

For `reasoning` (cat_imp=0.4):
```
sv = 0.4 * 0.6 + imp * 0.3 + ab * 0.1 = 0.24 + imp * 0.3 + ab * 0.1
```

To reach sv >= 0.6, need `imp * 0.3 + ab * 0.1 >= 0.36`:

| access_count | access_bonus | Required importance | Feasible? |
|---|---|---|---|
| 0 | 0.00 | >= 1.200 | IMPOSSIBLE |
| 5 | 0.25 | >= 1.117 | IMPOSSIBLE |
| 10 | 0.50 | >= 1.033 | IMPOSSIBLE |
| 15 | 0.75 | >= 0.950 | Barely |
| 20+ | 1.00 | >= 0.867 | Requires sustained high feedback |

Reasoning memories need `access_count >= 15 AND importance >= 0.87` -- an extraordinarily high bar. In the dynamics system, importance starts at `CATEGORY_IMPORTANCE["reasoning"] = 0.4` and is updated via `importance_update(imp, delta, signal)`. Reaching 0.87 requires approximately `(0.87 - 0.4) / delta` positive feedback signals. With typical `feedback_sensitivity` of 0.1, that is ~5 consecutive positive signals with no negatives.

Most reasoning memories will decay to zero without consolidation, losing potentially valuable causal chains.

**Fix applied:** Lower the semantic_value_threshold in the category-specific override table, or add a category-adjusted threshold. Applied the latter: Section 5.2 now documents per-category effective thresholds.

---

## Finding 4: Fact Category Requires Dynamics Boost (Scale-Check)

**Operator:** Scale-Check -- "What initial conditions let facts consolidate?"

For `fact` (cat_imp=0.6):
```
sv = 0.36 + imp * 0.3 + ab * 0.1
```

| access_count | Required importance |
|---|---|
| 0 | >= 0.800 |
| 5 | >= 0.717 |
| 10 | >= 0.633 |
| 15 | >= 0.550 |
| 20+ | >= 0.467 |

Facts start with importance 0.6 (from CATEGORY_IMPORTANCE). With 0 accesses, sv = 0.36 + 0.6*0.3 = 0.54 -- below threshold. A fact needs at least 1-2 positive feedback signals OR 7+ accesses to reach the consolidation threshold.

This is reasonable but creates a cold-start problem: a fact that is encoded, never accessed, and decays past retention threshold is lost -- it fails both the retention gate AND the semantic_value gate simultaneously.

**Fix applied:** Add a note about this cold-start gap and document the minimum conditions for each category to reach consolidation eligibility.

---

## Finding 5: retrieval_frequency_threshold=50 Is Vacuous (Materialize)

**Operator:** Materialize -- "Under what conditions does this threshold actually filter anything?"

The secondary trigger requires `access_count < 50`. Typical episodic memory access counts:
- 90th percentile: ~5 accesses
- 99th percentile: ~15 accesses
- 99.9th percentile: ~30 accesses

A threshold of 50 filters essentially nothing. The condition `access_count < 50` is satisfied by >99.99% of memories. It is dead code in practice.

The actual gate in the secondary trigger is `semantic_value >= 0.6`, not the access count check.

**Fix applied:** Lower default to 10 (above which a memory is considered "frequently accessed" and worth keeping in episodic form). Document the reasoning.

---

## Finding 6: Storage Pressure at max_pool_size=1000 Is the Normal State (Scale-Check)

**Operator:** Scale-Check -- "At what pool sizes does this system actually operate?"

Research context: typical memory pools are 1000-5000 total memories. The default `max_pool_size=1000` triggers storage pressure at the low end of the typical range.

Under storage pressure, the consolidation criteria collapse to:
```
not contested AND creation_time > consolidation_window AND category != "correction"
```

This means:
- The careful retention/semantic_value gating (Section 2.3) is bypassed
- ALL non-correction, non-contested memories older than 7 time units are consolidated
- This includes low-importance memories that should not produce semantic extractions
- The system degrades to "archive everything older than 7 days"

For a user with 2000 memories (moderately active), storage pressure is permanent. The elaborate consolidation logic in `should_consolidate()` becomes irrelevant -- the pressure bypass dominates.

**Fix applied:** Raise default max_pool_size to 5000. Add a graduated pressure system: mild pressure (1x-2x threshold) uses a relaxed retention threshold, severe pressure (>2x) uses the current bypass logic.

---

## Finding 7: Jaccard Similarity Only Catches Near-Exact Duplicates (Materialize)

**Operator:** Materialize -- "What pairs would actually cluster at threshold=0.75?"

Token-level Jaccard similarity at the 0.75 cluster_threshold:

| Pair | Jaccard | Clusters? |
|---|---|---|
| Exact duplicate | 1.000 | YES |
| Same + 3 extra words | 0.636 | NO |
| Same meaning, different words ("works at" vs "employed at") | 0.250 | NO |
| Same preference with synonyms ("dark mode" vs "dark themes") | 0.231 | NO |

At threshold 0.75, clustering requires 75%+ token overlap. In practice, only near-exact duplicates or substrings cluster together. Semantically equivalent memories with different surface forms are never merged.

This means:
- L3->L4 transition (cluster_sim > 0.8) is even more restrictive -- practically only exact duplicates
- Batch consolidation clustering provides minimal value beyond intra-session dedup
- The "aggregated patterns" described for SEMANTIC_INSIGHT are unreachable for diverse facts

The spec acknowledges embedding-based affinity as a future extension (Section 15) but does not document the severity of this limitation in the current design.

**Fix applied:** Add a "Known Limitations" section documenting the Jaccard ceiling. Lower the cluster_threshold default to 0.6 for batch mode (still catches near-duplicates, catches more substring matches). Document that L3->L4 is effectively gated on future embedding support.

---

## Finding 8: L3->L4 Transition Is Effectively Unreachable (Scale-Check + Materialize)

**Operator:** Combined -- "Can any realistic memory pair reach l3_to_l4_cluster_sim=0.8?"

The L3->L4 path requires `cluster_sim > 0.8`. But at L3 (SEMANTIC_FACT), memories have already been through extraction and templating:
- "User prefers: using Vim"
- "User prefers: using Neovim for editing"

These share the template prefix but differ in the extracted content. Jaccard similarity:
```
tokens("User prefers: using Vim") = {"user", "prefers:", "using", "vim"}
tokens("User prefers: using Neovim for editing") = {"user", "prefers:", "using", "neovim", "for", "editing"}
union = 7, intersection = 3, Jaccard = 3/7 = 0.429
```

Even related facts from the same template do not reach 0.8. The L3->L4 promotion path is dead code until embedding-based affinity is implemented.

**Fix applied:** Document this explicitly. Add a note that L3->L4 is reserved for future embedding support and should not be expected to fire in v1.

---

## Finding 9: Consolidation Decay Makes "Retrievable via Temporal Queries" Misleading (Materialize)

**Operator:** Materialize -- "What scores does an archived memory actually have?"

With `consolidation_decay=0.1`:
```
typical memory: importance=0.7, strength=7.0
archived: decayed_importance=0.07, decayed_strength=0.7

retention at t=1 with S=0.7: R = exp(-1/0.7) = 0.240
retention at t=3 with S=0.7: R = exp(-3/0.7) = 0.014
retention at t=7 with S=0.7: R = exp(-7/0.7) = 0.000045
```

An archived memory with importance=0.07 and strength=0.7 is:
- Nearly zero retention after 3 time units
- Importance so low it would rank below any active memory
- In scoring: score = w1*rel + w2*R + w3*0.07 + w4*sigmoid(0) ~ w3*0.07

The spec claims "retrievable via explicit temporal queries" but at these scores, archived memories are buried below everything else. If the recall system does not have a special archive-aware query mode, these are effectively deleted.

**Fix applied:** Clarify that "retrievable" requires a dedicated archive retrieval path that bypasses normal scoring. Add this to integration requirements.

---

## Finding 10: Time Unit Ambiguity Between Consolidation and Dynamics (Scale-Check)

**Operator:** Scale-Check -- "What happens if delta_t != 1.0?"

MemoryState documents `creation_time` as "Time steps since creation." The consolidation spec uses `consolidation_window = 7.0` assuming 1 time unit = 1 day. But the dynamics engine uses discrete time steps of size `ParameterSet.delta_t`.

If `delta_t = 0.1`:
- 7.0 time steps = 0.7 actual time units
- `consolidation_window = 7.0` means "7 steps" = 0.7 days, not 7 days
- `recency_guard = 1.0` means 0.1 days = 2.4 hours

If `delta_t = 10.0`:
- 7.0 time steps = 70 actual time units
- `consolidation_window = 7.0` means 70 days

The consolidation thresholds are only valid when `delta_t = 1.0`. The spec does not state this assumption or provide guidance for other step sizes.

**Fix applied:** Add explicit assumption that consolidation thresholds assume `delta_t = 1.0` (1 time step = ~1 day). Document that callers must scale thresholds proportionally to delta_t.

---

## Finding 11: Compression Ratio Can Be < 1.0 (Expansion) for Short Content (Materialize)

**Operator:** Materialize -- "What is the actual compression ratio for a 12-character preference?"

```
Input: "I prefer Vim" (12 chars)
Pattern extracts: "Vim"
Template: "User prefers: Vim" (17 chars)
compression_ratio = 12 / 17 = 0.706 -- EXPANSION, not compression
```

The spec guards against zero with `max(compression_ratio, 0.01)` but does not account for expansion. The `CATEGORY_COMPRESSION` targets (e.g., 10:1 for preference) are meaningless for short content. The `SemanticExtraction.compression_ratio` can be < 1.0, which means the "compressed" version is LONGER than the original.

This is not a bug -- it is an inevitable consequence of adding template prefixes to short content. But it means:
- `compression_ratio` is a misleading name for short content
- Any analysis based on compression ratios must account for expansion cases

**Fix applied:** Add edge case documentation and rename the validation guard to note that values < 1.0 indicate expansion.

---

## Summary of Fixes Applied to spec.md

| Finding | Section Modified | Change |
|---|---|---|
| F1: Half-life naming | 4.1, 2.1 | Rename to time constants, add clarifying note |
| F2: Greeting/transactional blocked | 5.2, 11 | Document de-facto exclusion |
| F3: Reasoning bar too high | 5.2 | Add per-category threshold table |
| F4: Fact cold-start | 5.2 | Document minimum conditions |
| F5: retrieval_frequency_threshold vacuous | 3.3, 2.3 | Lower default to 10 |
| F6: Storage pressure too aggressive | 3.3, 5.6 | Raise max_pool_size to 5000, add graduated pressure |
| F7: Jaccard only deduplicates | 5.5, new Section 16 | Document limitation, lower cluster_threshold to 0.6 |
| F8: L3->L4 unreachable | 2.1, new Section 16 | Document as reserved for v2 |
| F9: Archived not retrievable | 5.7, 7.5 | Clarify archive retrieval requirements |
| F10: Time unit ambiguity | 3.3 | Add delta_t=1.0 assumption |
| F11: Compression expansion | 3.5, 11 | Document expansion edge case |

