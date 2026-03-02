# Phased Implementation Plan: Encoding Layer

Created: 2026-02-27
Author: architect-agent
Status: READY FOR IMPLEMENTATION

---

## Overview

This plan decomposes the encoding layer (spec: `encoding/spec.md`) into 6
dependency-ordered phases. Each phase is one testable unit with clear inputs,
outputs, and a list of tests it should make pass.

**Total tests:** 68 (in `tests/test_encoding.py`)
**Existing tests (other modules):** 127 across 10 test files
**Target file:** `hermes_memory/encoding.py` (single module, stdlib-only: `dataclasses`, `re`)
**Test runner:** `/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py -q --tb=short`

### Dependency Chain

```
Phase A: Data Types
    |
    v
Phase B: Pattern Engine
    |
    v
Phase C: Write Policy
    |
    v
Phase D: evaluate() Integration
    |
    v
Phase E: Property Invariants
    |
    v
Phase F: Test Alignment & Full Suite
```

### Prerequisites

Before any phase, ensure the venv has dependencies installed:

```bash
cd /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python
/Users/cosimo/.hermes/hermes-agent/venv/bin/pip install hypothesis pytest numpy scipy
```

---

## Phase A: Data Types

**Goal:** Create the module with frozen config dataclass, decision dataclass,
constants, and class stub. This gives the test file something to import.

**Files to create:**
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py`

**Files to modify:**
- None (do NOT add to `__init__.py` yet -- tests import directly from `hermes_memory.encoding`)

### Implementation Details

1. **`EncodingConfig`** -- frozen dataclass with these fields and defaults:
   - `min_reasoning_length: int = 100`
   - `min_reasoning_connectives: int = 2`
   - `max_greeting_length: int = 50`
   - `max_transactional_length: int = 80`
   - `episode_length_offset: int = 150`
   - `confidence_threshold: float = 0.5`
   - `use_word_boundaries: bool = True`

2. **`VALID_CATEGORIES`** -- `frozenset` of 7 strings:
   `{"preference", "fact", "correction", "reasoning", "instruction", "greeting", "transactional"}`

3. **`CATEGORY_IMPORTANCE`** -- `dict[str, float]`:
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

4. **`EncodingDecision`** -- non-frozen dataclass with fields:
   - `should_store: bool`
   - `category: str`
   - `confidence: float`
   - `reason: str`
   - `initial_importance: float`

5. **`KNOWLEDGE_TYPE_TO_CATEGORY`** -- `dict[str, str]`:
   ```python
   {
       "personal_fact": "fact",
       "preference": "preference",
       "instruction": "instruction",
       "correction": "correction",
       "opinion": "preference",
       "skill": "fact",
       "relationship": "fact",
       "habit": "preference",
       "goal": "instruction",
       "context": "fact",
   }
   ```

6. **`SINGLE_PATTERN_CONFIDENCE_FLOOR`** -- constant `float = 0.35`

7. **`PRIORITY_ORDER`** -- list defining classification priority:
   ```python
   ["correction", "instruction", "preference", "fact", "reasoning", "transactional", "greeting"]
   ```

8. **`EncodingPolicy`** -- class stub with `__init__(self, config=None)` that
   stores `self.config = config or EncodingConfig()`. Add stub methods that
   raise `NotImplementedError` for `evaluate()`, `classify()`,
   `_apply_write_policy()`.

### Tests Expected to Pass (13 tests)

**TestCategoryImportanceMapping (3 tests):**
- `test_all_categories_have_importance` -- iterates VALID_CATEGORIES, checks each in CATEGORY_IMPORTANCE
- `test_importance_values_match_spec` -- checks exact values per spec Section 2.3
- `test_importance_ordering` -- checks relative ordering of importance values

**TestConfiguration (2 tests):**
- `test_config_is_frozen` -- attempts mutation, expects `FrozenInstanceError`
- `test_default_config_works` -- requires `evaluate()` to work, so this one defers to Phase D

**TestIntegration (1 test):**
- `test_encoding_config_is_frozen_dataclass` -- checks `is_dataclass`, checks frozen, checks default values

**Net count for Phase A: 4 tests pass** (the 3 importance-mapping tests + the frozen dataclass config test). The rest require `evaluate()`.

**Note:** `test_default_config_works` and `test_encoding_decision_is_dataclass` need `evaluate()` to produce an `EncodingDecision`, so they pass in Phase D.

### Key Implementation Notes

- `EncodingConfig` MUST be `@dataclass(frozen=True)` -- the test literally catches `FrozenInstanceError`.
- `EncodingDecision` is NOT frozen (spec Section 2.2: "not shared state").
- All constants are module-level, not class-level, because tests import them directly:
  `from hermes_memory.encoding import CATEGORY_IMPORTANCE, VALID_CATEGORIES`.
- The `EncodingPolicy` stub must accept `EncodingConfig | None` in `__init__`.

### Verification

```bash
cd /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py::TestCategoryImportanceMapping -q --tb=short
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py::TestIntegration::test_encoding_config_is_frozen_dataclass -q --tb=short
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py::TestConfiguration::test_config_is_frozen -q --tb=short
```

### Lint Check

```bash
ruff check /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py
```

---

## Phase B: Pattern Engine

**Goal:** Implement all pattern sets (first-person and third-person), the
`_count_matches()` helper, and the `classify()` method. This is the core
classification logic without write policy or evaluate() wiring.

**Files to modify:**
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py`

### Implementation Details

1. **First-person pattern sets** (module-level constants, all lowercase):
   - `PREFERENCE_PATTERNS` -- 14 patterns from spec Section 3.1
   - `FACT_PATTERNS` -- 21 patterns
   - `CORRECTION_PATTERNS` -- 19 patterns
   - `INSTRUCTION_PATTERNS` -- 13 patterns
   - `REASONING_CONNECTIVES` -- 14 patterns
   - `GREETING_PATTERNS` -- 15 patterns
   - `TRANSACTIONAL_PATTERNS` -- 17 patterns

2. **Third-person pattern sets** (module-level constants):
   - `THIRD_PERSON_PREFERENCE_PATTERNS` -- 14 patterns from spec Section 3.4
   - `THIRD_PERSON_FACT_PATTERNS` -- 14 patterns
   - `THIRD_PERSON_CORRECTION_PATTERNS` -- 13 patterns
   - `THIRD_PERSON_INSTRUCTION_PATTERNS` -- 12 patterns
   - `THIRD_PERSON_REASONING_CONNECTIVES` -- 12 patterns
   - `THIRD_PERSON_GREETING_PATTERNS` -- 9 patterns
   - `THIRD_PERSON_TRANSACTIONAL_PATTERNS` -- 13 patterns

3. **Aggregate dicts** for iteration:
   ```python
   FIRST_PERSON_PATTERNS: dict[str, list[str]] = {
       "preference": PREFERENCE_PATTERNS,
       "fact": FACT_PATTERNS,
       "correction": CORRECTION_PATTERNS,
       "instruction": INSTRUCTION_PATTERNS,
       "reasoning": REASONING_CONNECTIVES,
       "greeting": GREETING_PATTERNS,
       "transactional": TRANSACTIONAL_PATTERNS,
   }
   THIRD_PERSON_PATTERNS: dict[str, list[str]] = {
       "preference": THIRD_PERSON_PREFERENCE_PATTERNS,
       # ... same structure
   }
   ```

4. **`_count_matches(self, text_lower: str, pattern_list: list[str]) -> int`**
   - When `self.config.use_word_boundaries` is True:
     `sum(1 for p in pattern_list if re.search(r'\b' + re.escape(p) + r'\b', text_lower))`
   - When False:
     `sum(1 for p in pattern_list if p in text_lower)`
   - Import `re` at module top.

5. **`classify(self, text: str, metadata: dict | None = None) -> tuple[str, float]`**

   Algorithm (from spec Section 5.3):
   ```
   text_lower = text.lower()
   matches: dict[str, int] = {}

   # Always check first-person patterns
   for category, pattern_list in FIRST_PERSON_PATTERNS.items():
       count = self._count_matches(text_lower, pattern_list)
       if count > 0:
           matches[category] = matches.get(category, 0) + count

   # Also check third-person patterns for episode/semantic content
   if metadata and metadata.get("source_type") in ("episode", "semantic"):
       for category, pattern_list in THIRD_PERSON_PATTERNS.items():
           count = self._count_matches(text_lower, pattern_list)
           if count > 0:
               matches[category] = matches.get(category, 0) + count

   if not matches:
       # No patterns matched
       effective_greeting_threshold = self.config.max_greeting_length
       if metadata and metadata.get("source_type") == "episode":
           effective_greeting_threshold += self.config.episode_length_offset
       if len(text) < effective_greeting_threshold:
           return ("greeting", 0.3)
       else:
           return ("fact", 0.2)

   # Pick highest-priority category among matches
   for category in PRIORITY_ORDER:
       if category in matches:
           matched_count = matches[category]
           total_patterns = len(FIRST_PERSON_PATTERNS[category])
           if metadata and metadata.get("source_type") in ("episode", "semantic"):
               total_patterns += len(THIRD_PERSON_PATTERNS[category])

           # Calibrated confidence (spec Section 3.3)
           raw_density = matched_count / total_patterns
           if matched_count >= 1:
               confidence = max(SINGLE_PATTERN_CONFIDENCE_FLOOR, raw_density)
               confidence = min(1.0, confidence + 0.08 * (matched_count - 1))
           else:
               confidence = raw_density

           return (category, confidence)

   return ("greeting", 0.1)  # unreachable if PRIORITY_ORDER covers all
   ```

### Tests Expected to Pass After Phase B (0 additional)

The `classify()` method is not called directly by any test -- tests call
`evaluate()` which calls `classify()` internally. So Phase B itself does not
directly make new tests pass, but it is a prerequisite for Phases C and D.

However, we can verify `classify()` works via manual/unit testing:
```bash
cd /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -c "
from hermes_memory.encoding import EncodingPolicy
p = EncodingPolicy()
print(p.classify('I prefer dark mode'))       # -> ('preference', ~0.35)
print(p.classify('Hello!'))                   # -> ('greeting', ~0.35)
print(p.classify('No, actually that is wrong')) # -> ('correction', ~0.43)
"
```

### Key Implementation Notes

- **ADVERSARIAL FINDING: Word boundaries.** The default `use_word_boundaries=True`
  means patterns like `"no "` become `\bno \b` -- but `re.escape("no ")` produces
  `"no\\ "` and `\b` around it becomes `\bno \b`. The trailing space matters.
  For patterns ending with a space (like `"no "`, `"so "`, `"hi "`, `"run "`),
  word-boundary matching needs care. The `\b` after a space is a word boundary
  at the space/word transition, which may not behave as expected. **Solution:**
  strip trailing spaces from the pattern before applying `\b`, and require
  the pattern to be followed by a word boundary OR whitespace. Alternatively,
  for patterns with trailing space like `"no "`, match as `\bno\b` followed by
  `\s`. Test this carefully.

  Simpler approach: use `\b` around the whole pattern as-is. For "no ", the
  regex `\bno \b` will match "no " as a whole-word token followed by a space
  and word boundary. This works for most cases. The spec itself acknowledges
  residual false positives.

  **Recommended:** For patterns ending in space/comma (like `"no,"`, `"no "`,
  `"hi "`, `"so "`), strip the trailing punctuation/space and use pure `\b`
  word boundary matching: `\bno\b`, `\bhi\b`, `\bso\b`. This is cleaner and
  what the spec intends with "standalone 'no'". For multi-word patterns like
  `"i like"`, `\bi like\b` works correctly.

- **ADVERSARIAL FINDING: Calibrated confidence.** `SINGLE_PATTERN_CONFIDENCE_FLOOR = 0.35`.
  With 1 match: confidence = max(0.35, raw_density) = 0.35.
  With 2 matches: 0.35 + 0.08*(2-1) = 0.43.
  With 3 matches: 0.35 + 0.08*(3-1) = 0.51 (above threshold).
  This calibration is critical for the write policy to function correctly.

- **Third-person patterns** are checked ONLY when metadata["source_type"] is
  "episode" or "semantic". Without metadata, only first-person patterns run.

### Verification

```bash
cd /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -c "
from hermes_memory.encoding import EncodingPolicy
p = EncodingPolicy()
cat, conf = p.classify('I prefer dark mode')
assert cat == 'preference', f'got {cat}'
assert 0.3 <= conf <= 0.5, f'got {conf}'
print('classify() smoke test passed')
"
```

### Lint Check

```bash
ruff check /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py
```

---

## Phase C: Write Policy

**Goal:** Implement `_apply_write_policy()` and the reclassification helper
`_reclassify_without()`. These are the filtering logic that decides
`should_store` based on category.

**Files to modify:**
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py`

### Implementation Details

1. **`_apply_write_policy(self, category: str, text: str, confidence: float) -> bool`**

   From spec Section 5.4:
   ```python
   if category in {"preference", "fact", "correction", "instruction"}:
       return True

   if category == "reasoning":
       text_lower = text.lower()
       connective_count = self._count_matches(text_lower, REASONING_CONNECTIVES)
       return (
           len(text) >= self.config.min_reasoning_length
           and connective_count >= self.config.min_reasoning_connectives
       )

   # greeting, transactional
   return False
   ```

   **Note on reasoning connective counting:** The write policy counts
   FIRST-PERSON reasoning connectives for the length/connective check. This
   is because the check is about textual density of causal language, not about
   source type. Both "because" and "therefore" appear in both first-person and
   third-person text.

2. **`_reclassify_without(self, text: str, exclude_category: str, metadata: dict | None = None) -> tuple[str, float]`**

   Re-runs classification with one category's patterns excluded:
   ```python
   text_lower = text.lower()
   matches: dict[str, int] = {}

   for category, pattern_list in FIRST_PERSON_PATTERNS.items():
       if category == exclude_category:
           continue
       count = self._count_matches(text_lower, pattern_list)
       if count > 0:
           matches[category] = matches.get(category, 0) + count

   if metadata and metadata.get("source_type") in ("episode", "semantic"):
       for category, pattern_list in THIRD_PERSON_PATTERNS.items():
           if category == exclude_category:
               continue
           count = self._count_matches(text_lower, pattern_list)
           if count > 0:
               matches[category] = matches.get(category, 0) + count

   if not matches:
       # After excluding the category, nothing else matches
       return (exclude_category, 0.3)  # keep original category, low confidence

   for category in PRIORITY_ORDER:
       if category == exclude_category:
           continue
       if category in matches:
           matched_count = matches[category]
           total_patterns = len(FIRST_PERSON_PATTERNS[category])
           if metadata and metadata.get("source_type") in ("episode", "semantic"):
               total_patterns += len(THIRD_PERSON_PATTERNS[category])
           raw_density = matched_count / total_patterns
           if matched_count >= 1:
               confidence = max(SINGLE_PATTERN_CONFIDENCE_FLOOR, raw_density)
               confidence = min(1.0, confidence + 0.08 * (matched_count - 1))
           else:
               confidence = raw_density
           return (category, confidence)

   return (exclude_category, 0.1)
   ```

### Tests Expected to Pass After Phase C (0 additional directly)

Like Phase B, the write policy is not tested in isolation by the test file.
All tests go through `evaluate()`. Phase C is a prerequisite for Phase D.

### Key Implementation Notes

- **Reasoning write policy** uses BOTH conditions: `len(text) >= min_reasoning_length`
  AND `connective_count >= min_reasoning_connectives`. Both must hold.
- **Reclassification** runs exactly once (no recursion). If reclassification
  produces another NEVER-store category, the fail-open gate in `evaluate()`
  handles it.
- The `_reclassify_without()` method shares the confidence calibration logic
  with `classify()`. Consider extracting the calibration into a helper
  `_calibrate_confidence(matched_count, total_patterns)` to avoid duplication.
  This is a DRY improvement, not a spec requirement.

### Verification

```bash
cd /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -c "
from hermes_memory.encoding import EncodingPolicy
p = EncodingPolicy()
# Test write policy directly
assert p._apply_write_policy('preference', 'test', 0.5) is True
assert p._apply_write_policy('greeting', 'hi', 0.5) is False
assert p._apply_write_policy('reasoning', 'short because text', 0.5) is False
print('_apply_write_policy() smoke test passed')
"
```

### Lint Check

```bash
ruff check /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py
```

---

## Phase D: evaluate() Integration

**Goal:** Wire `classify()`, metadata boosts, length-based reclassification,
`_apply_write_policy()`, fail-open, and importance lookup into the full
`evaluate()` method. This is where the bulk of tests start passing.

**Files to modify:**
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py`

### Implementation Details

Implement `evaluate()` per spec Section 5.2 algorithm:

```python
def evaluate(
    self,
    episode_content: str,
    metadata: dict | None = None,
) -> EncodingDecision:
    # Step 1: Handle empty/None input
    if episode_content is None or episode_content.strip() == "":
        return EncodingDecision(
            should_store=False,
            category="greeting",
            confidence=1.0,
            reason="Empty input",
            initial_importance=0.0,
        )

    # Step 2: Semantic memory shortcut
    if metadata and metadata.get("source_type") == "semantic":
        knowledge_type = metadata.get("knowledge_type")
        if knowledge_type and knowledge_type in KNOWLEDGE_TYPE_TO_CATEGORY:
            category = KNOWLEDGE_TYPE_TO_CATEGORY[knowledge_type]
            confidence = 0.85
            should_store = self._apply_write_policy(category, episode_content, confidence)
            reason = f"Semantic memory shortcut: knowledge_type '{knowledge_type}' -> '{category}'"
            if not should_store and confidence < self.config.confidence_threshold:
                should_store = True
                reason += f" [fail-open: confidence {confidence:.2f} < {self.config.confidence_threshold}]"
            return EncodingDecision(
                should_store=should_store,
                category=category,
                confidence=confidence,
                reason=reason,
                initial_importance=CATEGORY_IMPORTANCE[category],
            )

    # Step 3: Classify
    text = episode_content.strip()
    category, confidence = self.classify(text, metadata)

    # Step 4: Metadata boost
    if metadata and metadata.get("message_count", 0) > 5:
        confidence = min(1.0, confidence + 0.1)
    if len(text) > 500:
        confidence = min(1.0, confidence + 0.05)

    # Step 4b: Length-based reclassification
    effective_greeting_len = self.config.max_greeting_length
    effective_transactional_len = self.config.max_transactional_length
    if metadata and metadata.get("source_type") == "episode":
        effective_greeting_len += self.config.episode_length_offset
        effective_transactional_len += self.config.episode_length_offset

    if category == "greeting" and len(text) > effective_greeting_len:
        category, confidence = self._reclassify_without(text, "greeting", metadata)
        # Re-apply metadata boost on new confidence
        if metadata and metadata.get("message_count", 0) > 5:
            confidence = min(1.0, confidence + 0.1)
        if len(text) > 500:
            confidence = min(1.0, confidence + 0.05)

    if category == "transactional" and len(text) > effective_transactional_len:
        category, confidence = self._reclassify_without(text, "transactional", metadata)
        if metadata and metadata.get("message_count", 0) > 5:
            confidence = min(1.0, confidence + 0.1)
        if len(text) > 500:
            confidence = min(1.0, confidence + 0.05)

    # Step 5: Apply write policy
    should_store = self._apply_write_policy(category, text, confidence)

    # Step 6: Fail-open
    reason = self._build_reason(category, confidence, text)
    if not should_store and confidence < self.config.confidence_threshold:
        should_store = True
        reason += f" [fail-open: confidence {confidence:.2f} < {self.config.confidence_threshold}]"

    # Step 7: Importance
    initial_importance = CATEGORY_IMPORTANCE[category]

    # Step 8: Return
    return EncodingDecision(
        should_store=should_store,
        category=category,
        confidence=confidence,
        reason=reason,
        initial_importance=initial_importance,
    )
```

Also implement `_build_reason()`:
```python
def _build_reason(self, category: str, confidence: float, text: str) -> str:
    """Build human-readable reason string."""
    return f"Classified as '{category}' with confidence {confidence:.2f}"
```

### Tests Expected to Pass After Phase D (approximately 55 tests)

**TestCategoryClassification (15 tests):**
- `test_classifies_preference_i_like`
- `test_classifies_preference_i_prefer`
- `test_classifies_preference_negative` -- expects "instruction" (priority: "always" instruction > "I don't like" preference)
- `test_classifies_fact_personal`
- `test_classifies_fact_location`
- `test_classifies_correction_no_actually`
- `test_classifies_correction_thats_wrong`
- `test_classifies_instruction_always`
- `test_classifies_instruction_remember`
- `test_classifies_reasoning_with_because`
- `test_classifies_greeting_hello`
- `test_classifies_greeting_thanks`
- `test_classifies_transactional_run`
- `test_classifies_transactional_open`
- `test_classifies_transactional_show`

**TestWritePolicy (10 tests):**
- `test_preference_always_stored`
- `test_fact_always_stored`
- `test_correction_always_stored`
- `test_instruction_always_stored`
- `test_greeting_never_stored`
- `test_transactional_never_stored`
- `test_reasoning_stored_when_long_enough`
- `test_reasoning_rejected_when_short`
- `test_reasoning_rejected_insufficient_connectives`
- `test_fail_open_low_confidence`

**TestInitialImportance (7 tests):**
- All 7 importance tests

**TestConfidence (5 tests):**
- `test_confidence_clear_preference_high`
- `test_confidence_ambiguous_text_lower`
- `test_confidence_always_in_range`
- `test_confidence_empty_string_high`
- `test_confidence_multiple_indicators_boosted`

**TestEdgeCases (8 tests):**
- `test_empty_string_is_greeting`
- `test_none_content_raises_or_handles`
- `test_very_long_unclassifiable_stored`
- `test_mixed_preference_and_fact`
- `test_non_english_text_stored`
- `test_greeting_long_reclassified`
- `test_transactional_long_reclassified`
- `test_whitespace_only_is_greeting`

**TestConfiguration (3 more tests, 5 total):**
- `test_custom_reasoning_length_threshold`
- `test_custom_greeting_length_threshold`
- `test_custom_confidence_threshold`
- `test_default_config_works`
- `test_config_is_frozen` (already passing from Phase A)

**TestPriorityOrdering (4 tests):**
- `test_correction_beats_preference`
- `test_correction_beats_instruction`
- `test_instruction_beats_fact`
- `test_preference_beats_transactional`

**TestMetadataInfluence (3 tests):**
- `test_message_count_boosts_confidence`
- `test_long_text_boosts_confidence`
- `test_metadata_does_not_change_category`

**TestIntegration (2 more tests, 3 total):**
- `test_importance_feeds_dynamics`
- `test_encoding_decision_is_dataclass`
- `test_encoding_config_is_frozen_dataclass` (already passing from Phase A)

**Total after Phase D: ~59 tests passing** (4 from Phase A + ~55 new)

### Key Implementation Notes

- **Empty/None handling:** Test `test_whitespace_only_is_greeting` passes `"   \n\t  "`
  which strips to `""`. The empty check must use `.strip() == ""`, not `== ""`.
- **Fail-open and greeting interaction:** `test_greeting_never_stored` uses a
  `confidence_threshold=0.0` policy to disable fail-open. This means ANY
  confidence >= 0.0 skips fail-open. The greeting's `should_store` comes purely
  from the write policy (False).
- **Reasoning rejection tests:** `test_reasoning_rejected_when_short` and
  `test_reasoning_rejected_insufficient_connectives` both use
  `confidence_threshold=0.0` to disable fail-open. Without this, the low
  confidence from a single connective match (0.35) would trigger fail-open
  and store the reasoning.
- **`test_classifies_preference_negative`:** The input `"I don't like tabs, always use spaces"`
  matches both preference ("I don't like") and instruction ("always"). The test
  expects `"instruction"` because instruction has higher priority than preference.
  This confirms that priority ordering works.
- **`test_greeting_long_reclassified`:** The input is 119 chars, well above
  `max_greeting_length=50`. After reclassification without greeting, it should
  match fact ("I'm") and be reclassified to "fact". The test asserts
  `category != "greeting"` and `should_store is True`.
- **`test_transactional_long_reclassified`:** Input is >80 chars. After
  reclassification without transactional, "I prefer" and "remember"/"going forward"
  match. Instruction ("going forward") has higher priority than preference
  ("I prefer"), so the test asserts `category != "transactional"` and
  `should_store is True`.
- **Metadata boost on reclassification:** When reclassification happens, the
  metadata boosts need to be re-applied to the new confidence value. The
  evaluate() algorithm above handles this.
- **`test_very_long_unclassifiable_stored`:** Uses "Lorem ipsum" repeated 200
  times. With word-boundary matching, check that no patterns accidentally match
  Latin text. The word "sit" does not match any pattern. "ipsum" does not match
  any pattern. This should default to "fact" (long unclassifiable) with
  confidence 0.2, +0.05 length bonus = 0.25, which is < 0.5, triggering fail-open.

### Verification

```bash
cd /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py -q --tb=short -x 2>&1 | tail -20
```

Run specific test classes to isolate failures:
```bash
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py::TestCategoryClassification -q --tb=short
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py::TestWritePolicy -q --tb=short
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py::TestEdgeCases -q --tb=short
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py::TestPriorityOrdering -q --tb=short
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py::TestMetadataInfluence -q --tb=short
```

### Lint Check

```bash
ruff check /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py
```

---

## Phase E: Property Invariants

**Goal:** Fix any property-based test failures found by Hypothesis. The 5
property-based tests exercise random inputs and may uncover edge cases that
the deterministic tests in Phase D missed.

**Files to modify:**
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py`

### Tests Expected to Pass (5 tests)

**TestPropertyBased (5 tests):**
- `test_confidence_always_bounded` -- confidence in [0.0, 1.0] for ANY text
- `test_importance_always_bounded` -- importance in [0.0, 1.0] for ANY text
- `test_category_always_valid` -- category in VALID_CATEGORIES for ANY text
- `test_deterministic_output` -- same input produces same output
- `test_should_store_consistent_with_category` -- ALWAYS-store categories are always stored; NEVER-store categories with confidence >= threshold are never stored

### Key Implementation Notes

**Likely failure modes from Hypothesis fuzzing:**

1. **Unicode edge cases in regex.** Hypothesis generates arbitrary Unicode strings.
   `re.search(r'\b' + re.escape(pattern) + r'\b', text_lower)` may behave
   unexpectedly with Unicode word boundaries. `\b` in Python regex matches at
   Unicode word boundaries, which includes transitions between word characters
   (`\w` = `[a-zA-Z0-9_]` plus Unicode letters/digits) and non-word characters.
   This should be fine for our English patterns, but test with:
   - Empty string (handled in Step 1)
   - String of only Unicode symbols
   - String with embedded null bytes
   - Very long strings

2. **Division by zero in confidence.** `raw_density = matched_count / total_patterns`.
   If `total_patterns == 0` (impossible because all pattern lists are non-empty
   constants), this would crash. Verify all pattern lists are non-empty.

3. **`test_should_store_consistent_with_category`** checks the compound invariant:
   - If category in {"preference", "fact", "correction", "instruction"} then `should_store is True`
   - If category in {"greeting", "transactional"} AND confidence >= 0.5 then `should_store is False`

   The second condition is tricky. After reclassification, a greeting might
   get re-classified to another greeting (if no other patterns match) but with
   different confidence. Ensure that:
   - Reclassification that fails to find another category returns the original
     category with LOW confidence (0.3), which triggers fail-open (0.3 < 0.5),
     making should_store=True. So the property `confidence >= 0.5 -> should_store=False`
     is not violated because reclassification yields confidence 0.3.
   - Direct greeting classification with 3+ patterns yields confidence >= 0.51,
     which should correctly reject.

4. **Confidence capping at 1.0.** The metadata boost adds 0.1 + 0.05 = 0.15.
   The pattern calibration can produce up to 1.0. Ensure `min(1.0, ...)` is
   applied everywhere. The test `test_confidence_always_bounded` will catch this.

5. **Determinism with regex.** `re.search()` is deterministic. `re.escape()` is
   deterministic. No randomness in the pipeline. This should pass.

### Debugging Strategy

If a Hypothesis test fails, it will print the minimal failing example. Use that
example to trace through `evaluate()` manually:

```bash
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -c "
from hermes_memory.encoding import EncodingPolicy
p = EncodingPolicy()
# Paste the failing input from Hypothesis
d = p.evaluate('<failing input here>')
print(f'category={d.category}, confidence={d.confidence}, should_store={d.should_store}, importance={d.initial_importance}')
"
```

### Verification

```bash
cd /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py::TestPropertyBased -q --tb=short -x
```

### Lint Check

```bash
ruff check /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py
```

---

## Phase F: Test Alignment & Full Suite

**Goal:** Ensure all 68 encoding tests pass. Reconcile any remaining failures
that arise from test expectations not matching the post-adversarial spec.
Then run the full 195-test suite (127 existing + 68 new).

**Files to potentially modify:**
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py` (implementation fixes)
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/tests/test_encoding.py` (test alignment, ONLY if a test's expectation contradicts the updated spec)

### Tests Expected to Pass (remaining ~4 tests + full suite)

All 68 tests in `test_encoding.py` should pass. Remaining tests from Phase D
that may need attention:

1. **`test_confidence_clear_preference_high`** -- checks that matching 3 preference
   patterns gives confidence > 0.07. With calibration: 0.35 + 0.08*2 = 0.51.
   Input: `"I prefer, like, and love using Vim"`. Need to verify these 3 patterns
   actually match with word-boundary matching:
   - `"i prefer"` -- matches `\bi prefer\b` in "i prefer, like, and love using vim"? The comma after "prefer" is a non-word character, so `\b` matches. YES.
   - `"i like"` -- does NOT appear in the text. The text has "like," not "i like". So only 1 match? Check carefully.
   - `"i love"` -- same issue. "love" appears but not "i love". The text says "I prefer, like, and love using Vim" -> lowered: "i prefer, like, and love using vim". Does `\bi love\b` match? "love" appears at position after "and " -- "and love using vim". There is no "i " before "love". So `"i love"` does NOT match.

   **CRITICAL:** This test expects confidence > 0.07, which is satisfied by a
   single calibrated match (0.35 > 0.07). But the test comment says "Matches
   'I prefer' and 'I like' and 'I love' = 3/14". With word boundaries, only
   `"I prefer"` matches. Confidence = 0.35, which IS > 0.07. Test passes.

2. **`test_message_count_boosts_confidence`** -- asserts exactly +0.1 boost from
   message_count metadata. Input: `"ok thanks"`. This matches greeting pattern
   `"thanks"`. Confidence without metadata: 0.35 (1 pattern, calibrated floor).
   With metadata (message_count=12 > 5): 0.35 + 0.1 = 0.45.
   Expected diff: 0.1. Actual diff: 0.1. Test passes.

3. **`test_long_text_boosts_confidence`** -- short text "I prefer dark mode"
   gets confidence 0.35 (single pattern). Long text "I prefer dark mode. " + padding
   gets confidence 0.35 + 0.05 (length > 500) = 0.40. The test asserts
   `long_decision.confidence >= short_decision.confidence`. 0.40 >= 0.35. Test passes.

4. **`test_metadata_does_not_change_category`** -- Input: `"Thanks for your help"`.
   Matches greeting pattern `"thanks"`. Category = "greeting".
   With metadata (message_count=20): still matches "thanks", same category "greeting".
   Confidence changes (0.35 -> 0.45) but category stays. Test passes.

### Potential Test-vs-Spec Conflicts

After thorough analysis, I identified these areas where tests might need
adjustment to align with the post-adversarial spec:

1. **`test_classifies_transactional_run`** -- Input: `"Run the test suite"`.
   With word boundaries, `\brun \b` needs to match "run the test suite" (lowered).
   The pattern is `"run "` -- `\brun \b` regex. "run " starts at position 0.
   `\b` before "r" is a word boundary (start of string). After "run ", the space
   is followed by "t" (word char), so `\b` matches at the space/word boundary.
   Also, `"test "` matches as a transactional pattern: `\btest \b` matches
   "test suite". So we get 2 transactional pattern matches. Category:
   transactional (no higher-priority category matches).

   BUT WAIT: "test " is also a transactional pattern. Does it match? "run the
   test suite" -> "test " appears as substring. With word boundaries:
   `\btest \b` -- "test " has `\b` before "t" (word boundary after space) and
   `\b` after the trailing space (boundary between space and "s" of "suite").
   Yes, matches. So 2 pattern matches -> confidence = 0.35 + 0.08 = 0.43.

   Test expects `category == "transactional"`. This works.

2. **`test_reasoning_rejected_insufficient_connectives`** -- The test text:
   `"The deployment pipeline is currently configured to run on every push to the main branch, which triggers a full rebuild of the Docker images and runs the entire integration test suite, all because the CI config was written that way initially."`

   This text is >100 chars, has 1 connective ("because"). Uses
   `confidence_threshold=0.0`. With word boundaries, "because" matches
   `\bbecause\b`. Connective count = 1 < 2 (min_reasoning_connectives).
   `_apply_write_policy` returns False. Confidence_threshold=0.0, so
   confidence (0.35) >= 0.0, fail-open does NOT trigger.
   `should_store = False`. Test passes.

   BUT: does "run " transactional pattern also match? "run on" -> `\brun \b`
   matches "run on" (word boundary before "run", space, then "on" starts with
   word char -> `\b` matches). Also "runs" -> `\brun \b` does NOT match "runs "
   because "runs" != "run ". Also "test " -> `\btest \b` matches "test suite".
   Also "build " -> `\bbuild \b` matches "rebuild " -- wait, "rebuild" contains
   "build" but `\bbuild \b` requires a word boundary before "build". In "rebuild",
   there is no word boundary before "build" (it's part of "rebuild"). So `\bbuild \b`
   does NOT match. Good.

   So transactional patterns match: "run " (in "run on"), "test " (in "test suite").
   That is 2 transactional matches.
   Reasoning matches: "because" = 1 match.

   Priority: reasoning (5) > transactional (6). So category = "reasoning".
   Write policy: len >= 100? YES. Connectives >= 2? NO (1 connective).
   `should_store = False`. With `confidence_threshold=0.0`, fail-open does not
   trigger. Test passes.

3. **`test_non_english_text_stored`** -- Input: `"Je prefere le mode sombre. Mon nom est Pierre."`.
   With word boundaries, check for accidental matches:
   - `"no "` -> `\bno\b` -- "nom" starts with "no" but `\bno\b` requires word boundary after "no". In "nom", "o" is followed by "m" (word char), so `\b` does NOT match. No match.
   - None of the English patterns should match French text with word boundaries.
   - Text length: 49 chars < max_greeting_length (50). So default: ("greeting", 0.3).
   - Fail-open: 0.3 < 0.5 -> should_store=True. Test passes.

   **Wait:** 49 chars -- is the text exactly 49 chars? Let me count:
   `"Je prefere le mode sombre. Mon nom est Pierre."` = 48 chars.
   48 < 50 -> "greeting" with confidence 0.3. Should_store=False (greeting policy).
   Fail-open: 0.3 < 0.5 -> override to True. Test asserts `should_store is True`
   and `confidence < 0.5`. Both pass.

### Reconciliation Rules

When a test expectation contradicts the updated spec:
- If the spec was updated by the adversarial review and the test was written
  before the update, the test expectation is stale. Update the test.
- If the test and spec agree but the implementation does not match, fix the
  implementation.
- Document every test change with a comment referencing the adversarial finding.

### Verification

```bash
cd /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python

# Run ALL encoding tests
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_encoding.py -q --tb=short

# Run the FULL test suite (all test files)
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/ -q --tb=short

# Verify no regressions in existing tests
/Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/ --ignore=tests/test_encoding.py -q --tb=short
```

### Lint Check

```bash
ruff check /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/hermes_memory/encoding.py
ruff check /Users/cosimo/.hermes/hermes-agent/proofs/hermes-memory/python/tests/test_encoding.py
```

---

## Summary: Test Distribution by Phase

| Phase | Description | New Tests Passing | Cumulative | Files Modified |
|-------|-------------|-------------------|------------|----------------|
| A | Data Types | 4 | 4 | encoding.py (create) |
| B | Pattern Engine | 0 (internal) | 4 | encoding.py |
| C | Write Policy | 0 (internal) | 4 | encoding.py |
| D | evaluate() Integration | ~55 | ~59 | encoding.py |
| E | Property Invariants | ~5 | ~64 | encoding.py |
| F | Test Alignment | ~4 | 68 | encoding.py, possibly test_encoding.py |

### Critical Path

Phases A-D are the critical path. Phase D is the largest single phase (~55 tests)
because `evaluate()` is the only public API that tests exercise. Phases B and C
are internal building blocks that enable Phase D.

If Phase D produces too many failures, consider splitting it:
- D1: evaluate() with empty input, None, whitespace (3 edge-case tests)
- D2: evaluate() for ALWAYS-store categories (preference, fact, correction, instruction) (~20 tests)
- D3: evaluate() for NEVER-store categories (greeting, transactional) with fail-open (~10 tests)
- D4: evaluate() for reasoning with conditional policy (~5 tests)
- D5: evaluate() with metadata, reclassification, priority (~17 tests)

### Word-Boundary Pattern Strategy

The single most likely source of test failures is the word-boundary regex
behavior on patterns that end with spaces or punctuation. Here is the
recommended approach:

For each pattern, preprocess before matching:
```python
def _make_pattern_regex(self, pattern: str) -> str:
    """Convert a pattern string to a word-boundary-aware regex."""
    stripped = pattern.strip()
    return r'\b' + re.escape(stripped) + r'\b'
```

This strips trailing spaces/commas from patterns like `"no,"`, `"no "`, `"hi "`,
`"so "` before applying `\b`. The word boundary `\b` after the stripped pattern
ensures the match is at a word edge, which is what the spec intends.

**Exception:** Patterns like `"im "` (informal "I'm") -- `\bim\b` would match
the word "im" standalone, which is correct. But it would NOT match "imagine"
(because `\b` does not trigger mid-word).

**Risk:** Some multi-word patterns with trailing space like `"i work at"` become
`\bi work at\b` which requires a word boundary after "at". In "I work at Google",
`\b` matches between "at" and space, then between space and "G". The `\b` after
"at" is at the "at"/space boundary. This should match correctly.

### Full Implementation Checklist

- [ ] Phase A: EncodingConfig, EncodingDecision, VALID_CATEGORIES, CATEGORY_IMPORTANCE, KNOWLEDGE_TYPE_TO_CATEGORY, PRIORITY_ORDER, SINGLE_PATTERN_CONFIDENCE_FLOOR, EncodingPolicy stub
- [ ] Phase B: All 14 pattern constants (7 first-person + 7 third-person), aggregate dicts, _count_matches(), classify()
- [ ] Phase C: _apply_write_policy(), _reclassify_without()
- [ ] Phase D: evaluate(), _build_reason()
- [ ] Phase E: Fix property-based test failures
- [ ] Phase F: Fix remaining test failures, run full suite
- [ ] Final: `ruff check` passes, all 68 encoding tests pass, all 127 existing tests still pass

---

**End of phased implementation plan.**
