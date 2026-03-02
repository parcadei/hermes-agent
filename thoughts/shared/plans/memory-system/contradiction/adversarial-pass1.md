# Adversarial Pass 1: Contradiction/Supersession Spec

**Date:** 2026-02-27
**Analyst:** architect-agent (Brenner-style adversarial)
**Scope:** Specification gaps, contradictions, missing edge cases, integration mismatches
**Input files:**
- Spec: `thoughts/shared/plans/memory-system/contradiction/spec.md`
- Tests: `tests/test_contradiction.py`
- Encoding: `hermes_memory/encoding.py`
- Engine: `hermes_memory/engine.py`
- Recall: `hermes_memory/recall.py`

**Total findings:** 14

---

## AP1-F1: MemoryState Has No `active` Field -- Supersession Cannot Work

**Severity:** CRITICAL
**Section reference:** Spec 7.2 (Supersession Semantics), Spec 13 (Relation to Recall Pipeline)

**Problem:**
The spec states (Section 7.2, item 1): "The old memory's `active` flag is set to False (soft delete)." Section 13 states: "recall observes through the memory pool... the caller sets those memories' `active` flag to False." Section 7.2, item 4 says: "Recall pipeline filters out inactive memories (already handled by existing MemoryState.active field convention)."

The actual `MemoryState` in `engine.py` (lines 195-246) has exactly six fields: `relevance`, `last_access_time`, `importance`, `access_count`, `strength`, `creation_time`. **There is no `active` field.** Furthermore, `MemoryState` is a frozen dataclass -- you cannot set any field on it after construction.

The `recall()` function in `recall.py` (lines 719-867) takes `list[MemoryState]` and performs no filtering based on an `active` flag. It passes the entire list directly to `rank_memories()`.

**Why it matters:**
The entire supersession lifecycle is built on the assumption that there exists an `active` field that can be toggled and that `recall()` reads. Neither exists. The spec claims an integration that is physically impossible with the current codebase. An implementor would either: (a) modify `MemoryState` (breaking the frozen contract and all existing tests), or (b) add a separate data structure to track active/inactive, which the spec does not describe.

**Proposed fix:**
Option A: Add `active: bool = True` to `MemoryState` and ensure `recall()` filters on it before ranking. This requires changes to engine.py, recall.py, and all their tests.
Option B: Specify an explicit `MemoryStore` abstraction (even if just a list wrapper) that holds `(MemoryState, active: bool)` tuples, and have `contradiction.py` return deactivation indices that the store applies before passing memories to `recall()`. The spec must describe this intermediate layer.
Option C: Specify that the caller is responsible for removing superseded memories from the list before passing to `recall()`. This is simpler but loses the audit trail benefit of soft delete.

---

## AP1-F2: `inherited_access_bonus` References Nonexistent Memory Metadata

**Severity:** HIGH
**Section reference:** Spec 7.2 (Supersession Semantics), item 3

**Problem:**
The spec states: "The new memory inherits the old memory's access_count as a bonus: `inherited_access_bonus = min(old.access_count // 2, 5)`."

However, `detect_contradictions()` and `resolve_contradictions()` operate on `str` texts and `str` categories -- not on `MemoryState` objects. The contradiction pipeline never receives `MemoryState` objects and therefore has no access to `access_count`. The `resolve_contradictions()` signature (Section 7.1) takes `(result, candidate_text, existing_texts, timestamp)` -- all strings and a float.

**Why it matters:**
There is no mechanism in the specified API for the contradiction module to compute or return `inherited_access_bonus`. An implementor has two contradictory directives: the API signature says "strings only," but the inheritance semantics require `MemoryState` access. The tests do not test this behavior at all.

**Proposed fix:**
Either: (a) add `existing_memories: list[MemoryState]` to `resolve_contradictions()` and include `inherited_access_bonus` in `SupersessionRecord`, or (b) move the inheritance logic to the caller (memory store/coordinator) and remove it from the contradiction spec entirely, documenting it as a coordinator responsibility.

---

## AP1-F3: Category-Specific Confidence Thresholds Contradict Global Threshold

**Severity:** HIGH
**Section reference:** Spec 7.3 vs Spec 6.2 (step 7)

**Problem:**
Section 6.2 (Pipeline Steps, step 7) defines the action resolution logic as:
- `confidence >= config.confidence_threshold AND config.enable_auto_supersede -> AUTO_SUPERSEDE`
- `confidence >= config.confidence_threshold AND NOT config.enable_auto_supersede -> FLAG_CONFLICT`
- `confidence > 0 but < config.confidence_threshold -> FLAG_CONFLICT`

The default `confidence_threshold` is 0.7.

But Section 7.3 defines **different** per-category thresholds:
- preference: supersedes at confidence >= **0.6** (below the global 0.7 threshold!)
- fact: supersedes at >= 0.7
- instruction: supersedes at >= 0.7
- correction: "always supersedes" (no threshold)

These two sections are contradictory. The pipeline in Section 6.2 uses a single global `confidence_threshold`, but Section 7.3 requires per-category thresholds. If the implementor follows Section 6.2, a preference reversal at confidence 0.65 would be FLAG_CONFLICT (below 0.7). If they follow Section 7.3, it should be AUTO_SUPERSEDE (at or above 0.6).

The "correction always supersedes" rule from 7.3 is also not represented in the pipeline logic of 6.2 at all -- there is no special case for corrections.

**Why it matters:**
Two implementors following different sections of the same spec would produce different behavior for identical inputs. The tests are ambiguous -- `test_correction_always_supersedes` checks for AUTO_SUPERSEDE but the pipeline logic has no mechanism to force this.

**Proposed fix:**
Either: (a) make `ContradictionConfig` carry per-category confidence thresholds (e.g., `category_confidence_thresholds: dict[str, float]`), update the pipeline logic in Section 6.2 step 7 to use them, and remove Section 7.3 as a separate rule; or (b) add explicit per-category override logic to Section 6.2 step 7 (e.g., "if candidate_category == 'correction', always AUTO_SUPERSEDE regardless of confidence_threshold") and reconcile the numbers.

---

## AP1-F4: `category_weights` Domain Mismatch -- 0.0 Allowed in Config But Used as Denominator Guard

**Severity:** MEDIUM
**Section reference:** Spec 3.2 (ContradictionConfig validation), Spec 6.2 (Pipeline Step 2, 5a)

**Problem:**
The validation rules state: "All values in category_weights are in **[0.0, 2.0]**" -- inclusive of 0.0.
The pipeline (Section 6.2) uses weight 0.0 as a skip sentinel: "Early exit: if candidate_category has weight 0.0" and "Skip if existing category has weight 0.0."
But the confidence calculations (Section 5.1, 5.2, etc.) multiply by `category_weight`: `confidence = min(1.0, subject_overlap * category_weight * correction_signal_strength)`.

If a user provides custom `category_weights` with `{"fact": 0.0}`, facts would be silently dropped from the pipeline. This is a behavioral footgun -- the spec never warns that setting a weight to 0.0 disables contradiction detection for that category.

More critically: the spec allows a user to set `category_weights = {"correction": 0.0, ...}` which would make corrections -- which "always supersede" per Section 7.3 -- silently skip. This directly contradicts Section 7.3.

**Why it matters:**
Config validation passes, but the behavior is counter-intuitive and contradicts other spec sections. No test covers the case of user-supplied 0.0 weights for normally-active categories.

**Proposed fix:**
Either: (a) document that weight 0.0 disables contradiction detection for that category and add a warning log, or (b) change the domain to (0.0, 2.0] (exclusive of zero) for categories that have special supersession rules (correction, fact, instruction, preference), or (c) separate the skip-sentinel logic from the confidence multiplier.

---

## AP1-F5: `SKIP` Action Is Specified But Never Produced by the Pipeline

**Severity:** MEDIUM
**Section reference:** Spec 3.5 (SupersessionAction), Spec 6.2 (Pipeline Step 7)

**Problem:**
The `SupersessionAction` enum has three members: `AUTO_SUPERSEDE`, `FLAG_CONFLICT`, `SKIP`. The pipeline logic in Section 6.2 step 7 defines only two action paths:
1. confidence >= threshold and auto_supersede enabled -> AUTO_SUPERSEDE
2. confidence >= threshold and auto_supersede disabled -> FLAG_CONFLICT
3. confidence > 0 but < threshold -> FLAG_CONFLICT

There is no path that produces `SKIP`. Every detection with confidence > 0 gets either AUTO_SUPERSEDE or FLAG_CONFLICT. Step 5g says "If confidence > 0: create ContradictionDetection" -- so only confidence > 0 ever reaches step 7, and every such detection gets a non-SKIP action.

Yet `test_skip_actions_ignored` (line 1468) manually constructs a `ContradictionResult` with a SKIP action and tests that `resolve_contradictions` ignores it. This is testing behavior that the pipeline can never produce organically.

**Why it matters:**
Dead code in the enum. If SKIP cannot be produced, it should not exist, or the pipeline should have a path to produce it (e.g., confidence > 0 but below some minimum floor). An implementor might add a SKIP path that the tests do not anticipate.

**Proposed fix:**
Either: (a) add a minimum confidence floor (e.g., `confidence < 0.1 -> SKIP`) to the pipeline, or (b) remove SKIP from the enum and document it as not needed, or (c) define SKIP as the action for detections where `confidence > 0` but `confidence < some_min_threshold` (distinct from `confidence_threshold`).

---

## AP1-F6: Spec Claims "Reuse CORRECTION_PATTERNS from encoding.py" But Contradiction Module Has No Import

**Severity:** MEDIUM
**Section reference:** Spec 1 (Dependencies), Spec 5.1 (DIRECT_NEGATION)

**Problem:**
Section 1 (Dependencies) lists:
- `encoding.py`: `EncodingDecision`, `VALID_CATEGORIES`, `CATEGORY_IMPORTANCE`

Section 5.1 says: "Candidate text contains correction markers... (reuse CORRECTION_PATTERNS from encoding.py)."

But `CORRECTION_PATTERNS` is NOT listed in the dependency table. The spec tells the implementor to reuse it but does not list it as a dependency. Furthermore, the contradiction module's public API (Section 10) does not export it or reference it.

This also raises a deeper question: should the contradiction module import and reuse encoding.py's pattern lists, or define its own? If it imports them, any change to encoding.py's patterns silently changes contradiction detection behavior. If it duplicates them, they can drift.

**Why it matters:**
Ambiguous dependency. An implementor might duplicate the patterns (fragile) or import them (coupling not declared). The test file does not import or verify alignment with encoding.py's CORRECTION_PATTERNS.

**Proposed fix:**
Either: (a) add `CORRECTION_PATTERNS` to the dependency table in Section 1 and import it in the module, or (b) define a separate `CONTRADICTION_CORRECTION_MARKERS` list in contradiction.py with an explicit note about the relationship, or (c) extract shared patterns to a `constants.py` that both modules import.

---

## AP1-F7: `extract_subject` Signature Allows `category: str | None` But Pipeline Always Passes a Category

**Severity:** LOW
**Section reference:** Spec 4.2 (Extraction Algorithm), Spec 6.2 (Pipeline Step 3)

**Problem:**
The `extract_subject` signature is `(text: str, category: str | None = None) -> SubjectExtraction`. The category parameter is optional.

But the spec never describes how `category` influences extraction behavior. Section 4.2 says "Try field-specific extractors in FIELD_EXTRACTORS order" -- no mention of using category to prioritize or filter extractors. The pipeline (Section 6.2 step 3) always passes the candidate category.

The test `test_category_hint_does_not_crash` (line 691) passes `category="preference"` and checks that the result is still a preference field_type, but this is tautological -- the text "I prefer dark mode" would match the preference extractor regardless of the category hint.

**Why it matters:**
The `category` parameter is specified but its semantics are undefined. Two implementors could interpret it differently: one might use it to short-circuit extraction (only try extractors matching the category), another might ignore it entirely. Both would satisfy the spec.

**Proposed fix:**
Either: (a) define explicit semantics for the category hint (e.g., "when category is provided, try extractors matching that category first, then fall back to all extractors"), or (b) remove the parameter from the spec since it has no defined behavior.

---

## AP1-F8: `subject_overlap` Symmetry Broken by `field_type` Bonus When Types Differ

**Severity:** LOW
**Section reference:** Spec 4.4 (Subject Overlap)

**Problem:**
The spec says subject_overlap computes Jaccard + 0.2 bonus when field_types match. The Hypothesis test `test_subject_overlap_symmetric` (line 1744) tests symmetry but always uses `field_type="unknown"` for both inputs -- so it never tests the case where field_types differ asymmetrically.

The Jaccard index is inherently symmetric, and the field_type bonus is also symmetric (it only fires when both types are equal). So symmetry holds in this spec. However, the spec says "Exact field_type match gets a 0.2 bonus" but does not specify what happens when one extraction has `field_type="unknown"` and the other has a specific type. Should "unknown" ever match anything? The spec is silent.

This is a minor concern now, but if future extensions add weighted field_type bonuses (e.g., "location" matching "name" gets 0.1 instead of 0.0), the symmetry property would break without test coverage.

**Why it matters:**
The property test has a blind spot. More importantly, the behavior of "unknown" field_type in overlap calculations is undefined.

**Proposed fix:**
(a) Add a statement: "`unknown` field_type never receives the bonus (it only applies when both field_types are identical and neither is `unknown`)." (b) Expand the Hypothesis test to generate mixed field_types.

---

## AP1-F9: Duplicate Detection Logic Is Underspecified for Near-Duplicates

**Severity:** HIGH
**Section reference:** Spec 11.2 (Self-contradiction)

**Problem:**
Section 11.2 says: "If candidate_text appears in existing_texts (exact duplicate), it is NOT a contradiction -- it's a duplicate. Skip it. Subject overlap will be 1.0 but value will be identical, so VALUE_UPDATE won't trigger."

This only handles **exact** duplicates. It does not address:
1. Case-normalized duplicates: "I live in New York" vs "i live in new york" (different as strings, identical after normalization)
2. Whitespace-normalized duplicates: "I live in  New York" vs "I live in New York"
3. Semantic duplicates with trivial rephrasing: "I live in NYC" vs "I live in New York City"
4. Substring containment: "I live in New York" vs "The user lives in New York and works remotely"

For case 1: After subject extraction and normalization, both produce identical SubjectExtractions. The VALUE_UPDATE strategy checks "values differ (after normalization)" -- they won't differ, so it won't trigger. But the DIRECT_NEGATION strategy might still trigger if the candidate happens to be categorized as "correction" (since it only checks subject overlap and correction markers, not value identity).

For case 2: Same issue.

The test `test_self_contradiction_duplicate_not_detected` only tests exact string equality.

**Why it matters:**
In production, memories are often stored after normalization or rephrasing. The spec only guards against exact-string duplicates. Near-duplicates could trigger false-positive contradictions, leading to erroneous supersession.

**Proposed fix:**
Add a pre-check: after subject extraction, if `candidate_subject.value == existing_subject.value` (after normalization) AND `candidate_subject.subject == existing_subject.subject`, skip that pair regardless of which detection strategy is being evaluated. Document this explicitly. Add tests for case-normalized and whitespace-normalized near-duplicates.

---

## AP1-F10: `resolve_contradictions` Does Not Validate Index Bounds

**Severity:** MEDIUM
**Section reference:** Spec 7.1 (resolve_contradictions)

**Problem:**
The `resolve_contradictions` signature takes `existing_texts: list[str]` and a `ContradictionResult` whose `superseded_indices` can contain arbitrary integers. The spec only specifies one validation: `TypeError` if `result` is not a `ContradictionResult`.

It does not specify what happens when:
1. `superseded_indices` contains an index >= len(existing_texts)
2. `superseded_indices` contains a negative index (other than -1 for new_index)
3. `existing_texts` is empty but `superseded_indices` is non-empty

Since `resolve_contradictions` receives pre-computed results from `detect_contradictions`, in normal flow the indices should be valid. But if someone constructs a `ContradictionResult` manually (as several tests do) or if there's a bug in detection, out-of-bounds indices would cause silent corruption or IndexError at the caller.

**Why it matters:**
Defensive programming. The test `test_multiple_supersessions` passes `existing_texts=["a", "b", "c"]` but constructs detections for indices 0 and 1 -- never testing boundary violations. A malformed `ContradictionResult` could cause runtime crashes downstream.

**Proposed fix:**
Add validation: "If any index in `superseded_indices` or `flagged_indices` is >= len(existing_texts) or < 0, raise ValueError." Add a test for this.

---

## AP1-F11: `FIELD_EXTRACTORS` Type Signature Mismatch Between Spec and Usage

**Severity:** MEDIUM
**Section reference:** Spec 4.1 (Field-Specific Extractors), Spec 10.3 (Constants)

**Problem:**
The spec defines FIELD_EXTRACTORS as:
```python
FIELD_EXTRACTORS: dict[str, list[tuple[re.Pattern, str, str]]]
```
where each tuple is `(pattern, subject_name, field_type)`.

But the patterns shown in the spec are raw regex strings (`r"(?:i (?:live|reside)..."`), not compiled `re.Pattern` objects. This is a type inconsistency -- if the type annotation says `re.Pattern` but the implementation stores raw strings and compiles on-the-fly, the type is wrong. If it stores compiled patterns, the spec examples are misleading.

Furthermore, the test `test_field_extractors_is_dict` (line 1804) only checks `isinstance(FIELD_EXTRACTORS, dict)` and `test_field_extractors_has_expected_keys` checks keys. Neither test validates the structure of the values (that they are lists of tuples with the correct types).

**Why it matters:**
An implementor choosing to store raw strings would satisfy the tests but violate the type annotation. An implementor storing compiled patterns would match the type but differ from the spec examples. The tests are insufficient to disambiguate.

**Proposed fix:**
(a) Clarify the type: either `dict[str, list[tuple[str, str, str]]]` for raw patterns or `dict[str, list[tuple[re.Pattern, str, str]]]` for compiled. (b) Add a test that validates the value structure (each entry is a 3-tuple with the correct member types).

---

## AP1-F12: Spec Section 6.2 Step 5e Strategy Selection Is Incomplete

**Severity:** HIGH
**Section reference:** Spec 6.2 (Pipeline Step 5e)

**Problem:**
Step 5e defines which detection strategies are "applicable":
- DIRECT_NEGATION: if `candidate_category == "correction"`
- VALUE_UPDATE: if both have non-None values and same field_type
- PREFERENCE_REVERSAL: if either is "preference"
- INSTRUCTION_CONFLICT: if both are "instruction"

This leaves gaps:
1. **Cross-category fact updates are undetected.** If the candidate is category "fact" (not "correction") and the existing memory is also "fact", but with different values for the same field, only VALUE_UPDATE applies. That works. But what if the candidate is category "instruction" and contradicts an existing "fact"? ("Always deploy to production" existing vs "Never deploy to production" candidate -- the candidate is "instruction" and the existing is "instruction", so INSTRUCTION_CONFLICT applies. But "My deploy target is staging" (fact) vs "My deploy target is production" (fact) -- VALUE_UPDATE. OK, that works.)

2. **Correction of an instruction.** Candidate is "correction", existing is "instruction". Only DIRECT_NEGATION is tried (because candidate_category == "correction"). But INSTRUCTION_CONFLICT is NOT tried (because it requires both to be "instruction"). So a correction that says "No, actually never use tabs" correcting "Always use tabs" only gets DIRECT_NEGATION scoring, not the more precise INSTRUCTION_CONFLICT scoring.

3. **Preference that is also a correction.** Candidate is "correction" (e.g., "Actually, I prefer light mode"), existing is "preference". Only DIRECT_NEGATION applies. PREFERENCE_REVERSAL would be more appropriate but requires "either is preference" -- the candidate is "correction", not "preference". The test `test_preference_correction_triggers_reversal` (line 1031) expects this to work but per the strategy selection rules, only DIRECT_NEGATION would be tried.

**Why it matters:**
The strategy selection predicates are too narrow. A correction of a preference should trigger PREFERENCE_REVERSAL (or at least both strategies should be tried). The tests expect behavior that the pipeline logic does not support.

**Proposed fix:**
Expand the applicability predicates:
- PREFERENCE_REVERSAL: if either is "preference" **OR** candidate is "correction" and existing is "preference"
- INSTRUCTION_CONFLICT: if both are "instruction" **OR** candidate is "correction" and existing is "instruction"
OR: simply try all four strategies for every pair and let the highest confidence win. The strategies that don't apply will produce confidence 0.

---

## AP1-F13: No Specification of Thread Safety or Reentrancy

**Severity:** LOW
**Section reference:** Spec overall, Spec 12 (Performance Constraints)

**Problem:**
The spec says "No external calls. No embeddings. No LLM. Pure text processing." and encoding.py documents itself as "Thread-safe: all methods are pure functions." But the contradiction spec does not state whether `detect_contradictions` or `resolve_contradictions` are thread-safe.

Since `ContradictionConfig` is frozen and all inputs are immutable (strings, frozen dataclasses), thread safety is likely a natural consequence. But this is not stated. If an implementor uses module-level mutable state (e.g., a compiled regex cache using `functools.lru_cache` or a module-level dict), thread safety could be violated.

**Why it matters:**
The memory system will be called from potentially concurrent contexts (multi-turn conversations, background encoding). Without an explicit thread-safety guarantee, an implementor might introduce mutable state.

**Proposed fix:**
Add a statement to the spec: "All public functions are thread-safe. No mutable module-level state. Regex patterns, if cached, must use thread-safe caching (e.g., module-level compiled constants, not runtime-populated dicts)."

---

## AP1-F14: Test Suite Does Not Test `value_pattern_min_tokens` Behavior

**Severity:** MEDIUM
**Section reference:** Spec 3.2 (ContradictionConfig), Spec 4 (Subject Extraction)

**Problem:**
`ContradictionConfig` has a field `value_pattern_min_tokens` with domain `>= 1` and default `2`. The spec says this is the "Minimum token count for value extraction."

But the spec never describes HOW this parameter is used. Section 4 (Subject Extraction) does not mention it. The extraction algorithm in Section 4.2 does not reference it. The confidence calculations in Section 5 do not reference it.

The test suite validates the config field exists and rejects invalid values (line 195: `("value_pattern_min_tokens", 0, "value_pattern_min_tokens")`), and tests that the default is 2 (line 228). But no behavioral test verifies that changing this parameter affects any output.

**Why it matters:**
A config parameter with no specified behavior is a dead field. An implementor either ignores it (making the validation pointless) or invents behavior for it (creating an unspec'd feature). The tests would pass either way.

**Proposed fix:**
Either: (a) define the behavior (e.g., "During subject extraction, if the extracted value has fewer than `value_pattern_min_tokens` tokens, set `value` to None"), add it to Section 4.2, and add behavioral tests; or (b) remove the parameter from the spec if it is not needed in v1.

---

## Summary Table

| ID | Severity | Section | Finding |
|----|----------|---------|---------|
| AP1-F1 | CRITICAL | 7.2, 13 | `MemoryState` has no `active` field; supersession lifecycle is impossible |
| AP1-F2 | HIGH | 7.2 | `inherited_access_bonus` requires `MemoryState` data unavailable in API |
| AP1-F3 | HIGH | 7.3 vs 6.2 | Per-category thresholds contradict global `confidence_threshold` logic |
| AP1-F4 | MEDIUM | 3.2, 6.2 | Weight 0.0 is valid config but silently disables categories, contradicting 7.3 |
| AP1-F5 | MEDIUM | 3.5, 6.2 | `SKIP` action can never be produced by the pipeline |
| AP1-F6 | MEDIUM | 1, 5.1 | Spec says reuse `CORRECTION_PATTERNS` but does not list it as dependency |
| AP1-F7 | LOW | 4.2, 6.2 | `category` parameter in `extract_subject` has undefined semantics |
| AP1-F8 | LOW | 4.4 | `unknown` field_type behavior in overlap bonus is unspecified |
| AP1-F9 | HIGH | 11.2 | Near-duplicate detection only handles exact strings, not normalized matches |
| AP1-F10 | MEDIUM | 7.1 | `resolve_contradictions` does not validate index bounds |
| AP1-F11 | MEDIUM | 4.1, 10.3 | `FIELD_EXTRACTORS` type annotation contradicts spec examples |
| AP1-F12 | HIGH | 6.2 | Strategy selection predicates too narrow; corrections of preferences/instructions under-detected |
| AP1-F13 | LOW | 12 | Thread safety not specified despite concurrent usage expectation |
| AP1-F14 | MEDIUM | 3.2, 4 | `value_pattern_min_tokens` has no specified behavior; dead config parameter |

**Critical:** 1 | **High:** 4 | **Medium:** 6 | **Low:** 3

---

## Recommended Priority for Spec Revision

1. **AP1-F1** (CRITICAL): Must resolve before implementation begins. The entire supersession lifecycle depends on a mechanism that does not exist.
2. **AP1-F3** (HIGH): Must reconcile per-category thresholds with global pipeline logic. This is a spec contradiction.
3. **AP1-F12** (HIGH): Strategy selection predicates must be expanded or tests must be revised. Spec and tests disagree.
4. **AP1-F9** (HIGH): Near-duplicate handling must be specified beyond exact-string matching.
5. **AP1-F2** (HIGH): Inherited access bonus must either be removed from spec or API must be expanded.
6. Remaining MEDIUM/LOW findings can be addressed during implementation or in Pass 2.
