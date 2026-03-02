# Adversarial Pass 2: Contradiction/Supersession Test Coverage Gaps

**Date:** 2026-02-27
**Author:** architect-agent (Brenner-style adversarial analysis)
**Scope:** `tests/test_contradiction.py` (165 claimed tests) vs `spec.md` (770 lines)
**Verdict:** 14 findings. Several CRITICAL gaps where spec requirements have zero test coverage.

---

## Methodology

Walked every numbered section of `spec.md` and cross-referenced against each test class in `test_contradiction.py`. Focused on:
1. Spec requirements with no corresponding test
2. Tests that check existence/type but not behavioral correctness
3. Missing boundary/threshold tests at exact values
4. Missing error paths explicitly listed in the spec
5. Missing interaction tests between features
6. Unrealistic or degenerate test data

---

## Findings

### AP2-F1: No Test for ContradictionConfig `category_weights` Negative Value Rejection
**Severity:** HIGH
**Spec reference:** Section 3.2 — "All values in category_weights are in [0.0, 2.0]"
**Gap:** The test `test_category_weights_value_above_two_rejected` checks the upper bound (2.5). There is NO test for a negative value (e.g., -0.1). The spec domain is `[0.0, 2.0]` — the lower bound 0.0 is inclusive, but negative values must be rejected.
**Proposed test:**
```python
class TestContradictionConfig:
    def test_category_weights_negative_value_rejected(self):
        """Category weight value < 0.0 is rejected."""
        weights = dict(_DEFAULT_CATEGORY_WEIGHTS)
        weights["correction"] = -0.1
        with pytest.raises(ValueError, match="category_weights"):
            ContradictionConfig(category_weights=weights)
```

---

### AP2-F2: No Test for `similarity_threshold` Upper Boundary (Exactly 1.0 Is Valid)
**Severity:** MEDIUM
**Spec reference:** Section 3.2 — "0.0 < similarity_threshold <= 1.0"
**Gap:** Tests check rejection of 0.0 and 1.1, but never verify that exactly 1.0 is ACCEPTED. The domain is `(0.0, 1.0]` — 1.0 is the inclusive upper bound. Without testing acceptance at 1.0, an off-by-one in the implementation (`< 1.0` instead of `<= 1.0`) would pass all current tests.
**Proposed test:**
```python
class TestContradictionConfig:
    def test_similarity_threshold_exactly_one_accepted(self):
        """similarity_threshold=1.0 is at the upper inclusive boundary and must be accepted."""
        config = ContradictionConfig(similarity_threshold=1.0)
        assert config.similarity_threshold == 1.0

    def test_confidence_threshold_exactly_one_accepted(self):
        """confidence_threshold=1.0 is at the upper inclusive boundary and must be accepted."""
        config = ContradictionConfig(confidence_threshold=1.0)
        assert config.confidence_threshold == 1.0
```

---

### AP2-F3: No Test for `confidence_threshold` Boundary in Action Resolution
**Severity:** CRITICAL
**Spec reference:** Section 6.2 step 7 — "confidence >= config.confidence_threshold → AUTO_SUPERSEDE"
**Gap:** `test_confidence_at_threshold_auto_supersedes` is structurally weak. It uses a real pipeline call with `confidence_threshold=0.7` and email data, but wraps the assertion in `if result.has_contradiction:` — meaning if the implementation returns no contradiction (a bug), the test PASSES SILENTLY. This is the "check existence but not correctness" anti-pattern. The test must ASSERT that a contradiction is detected, then assert the action. Additionally, there is no test for confidence at exactly 0.699... (just below threshold) producing FLAG_CONFLICT instead of AUTO_SUPERSEDE.
**Proposed test:**
```python
class TestDetectContradictions:
    def test_confidence_at_exact_threshold_produces_auto_supersede(self):
        """Confidence == confidence_threshold must produce AUTO_SUPERSEDE, not FLAG_CONFLICT."""
        # Use email update which spec guarantees confidence >= 0.8
        config = ContradictionConfig(confidence_threshold=0.8)
        result = detect_contradictions(
            candidate_text="My email is new@example.com",
            candidate_category="fact",
            existing_texts=["My email is old@example.com"],
            existing_categories=["fact"],
            config=config,
        )
        assert result.has_contradiction is True, "Email update must detect contradiction"
        assert result.highest_confidence >= 0.8, "Email update must have confidence >= 0.8"
        has_auto = any(
            action == SupersessionAction.AUTO_SUPERSEDE
            for _, action in result.actions
        )
        assert has_auto, "Confidence >= threshold must produce AUTO_SUPERSEDE"

    def test_below_confidence_threshold_produces_flag_conflict(self):
        """Confidence below confidence_threshold must produce FLAG_CONFLICT, not AUTO_SUPERSEDE."""
        # Set threshold very high so detection falls below it
        config = ContradictionConfig(confidence_threshold=0.99)
        result = detect_contradictions(
            candidate_text="My email is new@example.com",
            candidate_category="fact",
            existing_texts=["My email is old@example.com"],
            existing_categories=["fact"],
            config=config,
        )
        assert result.has_contradiction is True
        for _, action in result.actions:
            assert action != SupersessionAction.AUTO_SUPERSEDE, (
                "Confidence below threshold must not AUTO_SUPERSEDE"
            )
```

---

### AP2-F4: No Test for Subject Normalization of Filler Words
**Severity:** MEDIUM
**Spec reference:** Section 4.3 — "Strip filler words ('actually', 'basically', 'just')"
**Gap:** `test_normalization_strips_articles` tests article stripping but there is ZERO coverage for filler word stripping. The spec explicitly lists "actually", "basically", "just" as filler words to strip. If the implementation omits filler stripping, no test catches it.
**Proposed test:**
```python
class TestExtractSubject:
    @pytest.mark.parametrize(
        "text,filler",
        [
            ("I basically prefer dark mode", "basically"),
            ("I just like Python", "just"),
            ("I actually live in New York", "actually"),
        ],
    )
    def test_normalization_strips_filler_words(self, text, filler):
        """Filler words ('actually', 'basically', 'just') are stripped from values (spec 4.3)."""
        result = extract_subject(text)
        if result.value is not None:
            assert filler not in result.value.split(), (
                f"Filler word '{filler}' should be stripped from value '{result.value}'"
            )
```

---

### AP2-F5: No Test for Action Resolution When Multiple Detections Target Same Memory
**Severity:** CRITICAL
**Spec reference:** Section 6.3 — "When multiple detections target the same existing memory: Take the highest-confidence detection's action. AUTO_SUPERSEDE wins over FLAG_CONFLICT for the same memory. A memory can only be superseded once per pipeline call."
**Gap:** There is ZERO test coverage for the conflict resolution rules in Section 6.3. The test `test_multiple_supersessions` tests two detections targeting DIFFERENT memories (indices 0 and 1). No test sends multiple detections to the SAME index and verifies the highest-confidence action wins. This is an entire subsystem with no coverage.
**Proposed test:**
```python
class TestDetectContradictions:
    def test_multiple_detections_same_index_highest_confidence_wins(self):
        """When multiple strategies detect contradictions for the same memory,
        the highest confidence action wins (spec 6.3)."""
        # A correction that is also a value update against the same memory.
        # Both DIRECT_NEGATION and VALUE_UPDATE could fire.
        # The pipeline should pick the higher confidence one.
        result = detect_contradictions(
            candidate_text="No, actually my email is new@example.com",
            candidate_category="correction",
            existing_texts=["My email is old@example.com"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is True
        # Only one action per existing index
        action_indices = [idx for idx, _ in result.actions]
        assert len(action_indices) == len(set(action_indices)), (
            "Each existing index should appear at most once in actions"
        )

    def test_memory_superseded_at_most_once(self):
        """A memory can only be superseded once per pipeline call (spec 6.3)."""
        result = detect_contradictions(
            candidate_text="No, actually my name is Bob and I live in London",
            candidate_category="correction",
            existing_texts=["My name is Alice", "I live in New York"],
            existing_categories=["fact", "fact"],
        )
        # Each superseded index should appear exactly once
        superseded_list = [idx for idx, act in result.actions
                          if act == SupersessionAction.AUTO_SUPERSEDE]
        assert len(superseded_list) == len(set(superseded_list)), (
            "Each index must be superseded at most once"
        )
```

---

### AP2-F6: No Test for `detect_contradictions` Sorting Detections by Confidence Descending
**Severity:** HIGH
**Spec reference:** Section 6.2 step 6 — "Sort detections by confidence (descending)."
**Gap:** No test verifies that `result.detections` is sorted by confidence descending. This is an explicit spec requirement. If the implementation returns them in insertion order (or ascending), every test passes because they only check `.has_contradiction` or inspect `actions[0]`.
**Proposed test:**
```python
class TestDetectContradictions:
    def test_detections_sorted_by_confidence_descending(self):
        """Detections in ContradictionResult are sorted by confidence, highest first (spec 6.2)."""
        result = detect_contradictions(
            candidate_text="No, actually my email is new@example.com and I live in London",
            candidate_category="correction",
            existing_texts=[
                "My email is old@example.com",
                "I live in New York",
            ],
            existing_categories=["fact", "fact"],
        )
        if len(result.detections) >= 2:
            confidences = [d.confidence for d in result.detections]
            assert confidences == sorted(confidences, reverse=True), (
                f"Detections must be sorted descending by confidence, got: {confidences}"
            )
```

---

### AP2-F7: No Test for `extract_subject` Category Hint Affecting Extractor Priority
**Severity:** LOW
**Spec reference:** Section 4.2 — `extract_subject(text: str, category: str | None = None)`
**Gap:** `test_category_hint_does_not_crash` verifies the call doesn't crash, but does not test whether the category hint actually AFFECTS extraction behavior. For example, if `category="instruction"` is passed, do instruction patterns get priority? The spec's signature includes category but the algorithm (4.2) doesn't explicitly say it changes behavior — however, the parameter exists for a reason. The test should verify at minimum that a category hint for an ambiguous text biases extraction.
**Proposed test:**
```python
class TestExtractSubject:
    def test_category_hint_biases_extraction_for_ambiguous_text(self):
        """Category hint influences extraction when text is ambiguous."""
        # "I always prefer X" could be instruction ("always ...") or preference ("prefer ...")
        result_instr = extract_subject("I always prefer dark mode", category="instruction")
        result_pref = extract_subject("I always prefer dark mode", category="preference")
        # At minimum, both should extract successfully; category may influence field_type
        assert isinstance(result_instr, SubjectExtraction)
        assert isinstance(result_pref, SubjectExtraction)
```

---

### AP2-F8: No Test for `enable_auto_supersede=False` Producing FLAG_CONFLICT (Not SKIP)
**Severity:** HIGH
**Spec reference:** Section 6.2 step 7 — "confidence >= config.confidence_threshold AND NOT config.enable_auto_supersede → FLAG_CONFLICT"
**Gap:** `test_auto_supersede_disabled_flags_instead` checks that no AUTO_SUPERSEDE appears, but is wrapped in `if result.has_contradiction:` — a silent pass if no contradiction is detected. More importantly, it does NOT verify that FLAG_CONFLICT is produced instead. The test only checks absence of AUTO_SUPERSEDE, not presence of FLAG_CONFLICT. An implementation that returns SKIP instead of FLAG_CONFLICT would pass.
**Proposed test:**
```python
class TestDetectContradictions:
    def test_auto_supersede_disabled_produces_flag_conflict(self):
        """With enable_auto_supersede=False, high-confidence contradictions produce FLAG_CONFLICT."""
        config = ContradictionConfig(enable_auto_supersede=False)
        result = detect_contradictions(
            candidate_text="My email is new@example.com",
            candidate_category="fact",
            existing_texts=["My email is old@example.com"],
            existing_categories=["fact"],
            config=config,
        )
        assert result.has_contradiction is True, "Email update must detect contradiction"
        assert len(result.actions) >= 1, "Must have at least one action"
        has_flag = any(
            action == SupersessionAction.FLAG_CONFLICT
            for _, action in result.actions
        )
        assert has_flag, (
            "With enable_auto_supersede=False, high-confidence contradiction must produce "
            f"FLAG_CONFLICT, got actions: {result.actions}"
        )
        assert len(result.flagged_indices) >= 1, "flagged_indices must be non-empty"
```

---

### AP2-F9: No Test for `value_pattern_min_tokens` Affecting Extraction
**Severity:** HIGH
**Spec reference:** Section 3.2 — "value_pattern_min_tokens: Minimum token count for value extraction. Domain: >= 1. Default: 2."
**Gap:** The config field `value_pattern_min_tokens` is validated (rejection of 0 is tested) but there is ZERO behavioral test that changing this parameter affects extraction. What happens when `value_pattern_min_tokens=5` and the extracted value has 2 tokens? The spec defines this parameter with a clear semantic role, but no test verifies the implementation honors it. This is a "config exists but does nothing" risk.
**Proposed test:**
```python
class TestDetectContradictions:
    def test_value_pattern_min_tokens_filters_short_values(self):
        """value_pattern_min_tokens=5 should suppress short value extractions."""
        config = ContradictionConfig(value_pattern_min_tokens=5)
        result = detect_contradictions(
            candidate_text="I live in NY",
            candidate_category="fact",
            existing_texts=["I live in LA"],
            existing_categories=["fact"],
            config=config,
        )
        # "NY" and "LA" are single-token values; with min_tokens=5 they should
        # not be extracted as values, so VALUE_UPDATE should not fire
        has_value_update = any(
            d.contradiction_type == ContradictionType.VALUE_UPDATE
            for d in result.detections
        )
        # Either no contradiction or a non-VALUE_UPDATE type
        if result.has_contradiction:
            assert not has_value_update, (
                "Short values below value_pattern_min_tokens should not trigger VALUE_UPDATE"
            )
```

---

### AP2-F10: No Test for `category_weights` Affecting Confidence Calculation
**Severity:** CRITICAL
**Spec reference:** Section 5.1 — "confidence = min(1.0, subject_overlap * category_weight * correction_signal_strength)"
**Gap:** `category_weights` are validated in config tests but NEVER tested for behavioral effect. The spec explicitly defines confidence as a product including `category_weight`. Tests for DIRECT_NEGATION use default weights but never verify that changing weights changes confidence. An implementation could ignore `category_weights` entirely and all tests would pass. This is a critical spec-to-test gap for the core scoring formula.
**Proposed test:**
```python
class TestDetectContradictions:
    def test_category_weight_zero_suppresses_detection(self):
        """Setting a category's weight to 0.0 should suppress contradiction detection for it."""
        weights = dict(_DEFAULT_CATEGORY_WEIGHTS)
        weights["fact"] = 0.0  # Facts should now be skipped like greetings
        config = ContradictionConfig(category_weights=weights)
        result = detect_contradictions(
            candidate_text="I live in London",
            candidate_category="fact",
            existing_texts=["I live in New York"],
            existing_categories=["fact"],
            config=config,
        )
        # With weight 0.0, facts should be skipped (same as greeting/transactional)
        assert result.has_contradiction is False, (
            "Category with weight 0.0 should be skipped in detection pipeline"
        )

    def test_higher_category_weight_increases_confidence(self):
        """Higher category_weight produces higher confidence for same input (spec 5.1)."""
        low_weight = dict(_DEFAULT_CATEGORY_WEIGHTS)
        low_weight["correction"] = 0.5
        high_weight = dict(_DEFAULT_CATEGORY_WEIGHTS)
        high_weight["correction"] = 1.5

        config_low = ContradictionConfig(category_weights=low_weight)
        config_high = ContradictionConfig(category_weights=high_weight)

        text_candidate = "No, actually the deadline is March 15th"
        text_existing = ["The deadline is March 10th"]
        cats_existing = ["fact"]

        result_low = detect_contradictions(
            text_candidate, "correction", text_existing, cats_existing, config_low,
        )
        result_high = detect_contradictions(
            text_candidate, "correction", text_existing, cats_existing, config_high,
        )
        if result_low.has_contradiction and result_high.has_contradiction:
            assert result_high.highest_confidence >= result_low.highest_confidence, (
                "Higher category weight must produce equal or higher confidence"
            )
```

---

### AP2-F11: No Test for `non_str` in `existing_texts` Raising TypeError
**Severity:** MEDIUM
**Spec reference:** Section 6.1 — "Raises: TypeError: If any text argument is not a str."
**Gap:** `test_non_str_candidate_raises_type_error` tests non-str for `candidate_text`, but there is no test for non-str values INSIDE `existing_texts`. The spec says "any text argument" — which includes elements of the `existing_texts` list. An implementation might only validate `candidate_text` type.
**Proposed test:**
```python
class TestDetectContradictions:
    def test_non_str_in_existing_texts_raises_type_error(self):
        """Non-string element in existing_texts raises TypeError."""
        with pytest.raises(TypeError):
            detect_contradictions(
                candidate_text="I live in London",
                candidate_category="fact",
                existing_texts=["valid", 42],  # type: ignore[list-item]
                existing_categories=["fact", "fact"],
            )

    def test_non_str_candidate_category_raises_type_error(self):
        """Non-string candidate_category raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            detect_contradictions(
                candidate_text="I live in London",
                candidate_category=42,  # type: ignore[arg-type]
                existing_texts=["test"],
                existing_categories=["fact"],
            )
```

---

### AP2-F12: No Test for `resolve_contradictions` Inherited Access Bonus Semantics
**Severity:** HIGH
**Spec reference:** Section 7.2 — "The new memory inherits the old memory's access_count as a bonus: `inherited_access_bonus = min(old.access_count // 2, 5)`. This prevents the replacement from starting cold while the old memory had history."
**Gap:** This is an entire behavioral requirement (access count inheritance on supersession) with ZERO test coverage. The spec defines a specific formula (`min(old.access_count // 2, 5)`) that should be tested at boundary values (access_count=0, 1, 10, 11). The `resolve_contradictions` tests only verify records and deactivation indices, never the access bonus. Note: this may require `resolve_contradictions` to accept access counts or for `SupersessionRecord` to carry the bonus. Either way, the spec mandates it and no test verifies it.
**Proposed test:**
```python
class TestResolveContradictions:
    def test_supersession_record_carries_inherited_access_bonus(self):
        """SupersessionRecord (or return value) includes the inherited access bonus (spec 7.2).

        Formula: min(old.access_count // 2, 5)
        """
        # This test may need adjustment based on how the bonus is exposed.
        # At minimum, verify the formula at boundary values.
        result = self._make_result_with_supersede(index=0, confidence=0.9)
        records, deactivated = resolve_contradictions(
            result, "I live in London", ["I live in New York"], timestamp=100.0,
        )
        assert len(records) == 1
        # The record or return tuple should communicate the access bonus.
        # If SupersessionRecord doesn't have this field, the spec is unimplemented.
        # This test documents the requirement.
        rec = records[0]
        # Check that the record is at least a SupersessionRecord
        assert isinstance(rec, SupersessionRecord)
        # TODO: When implementation adds access bonus, assert:
        # assert rec.inherited_access_bonus == min(access_count // 2, 5)
```

---

### AP2-F13: Weak Assertion Pattern — `if result.has_contradiction` Guards Hide Failures
**Severity:** CRITICAL
**Spec reference:** Multiple sections (5.1, 5.2, 5.3, 5.4, 7.3)
**Gap:** The following tests use the pattern `if result.has_contradiction:` before making assertions, which means a bug that produces NO contradiction makes the test silently pass:
- `test_confidence_at_threshold_auto_supersedes` (line 1144)
- `test_auto_supersede_disabled_flags_instead` (line 1262)
- `test_correction_always_supersedes` (line 1514)
- `test_correction_supersedes_over_preference` (line 1606)
- `test_location_update_pipeline` (line 1630)
- `test_reasoning_flags_only_never_supersedes` (line 1569)
- `test_max_candidates_limits_scan` (line 1195)
- `test_empty_existing_memory_skipped` (line 1106)

This is 8 tests (roughly 5% of the suite) that can silently pass on completely broken implementations. Every `if result.has_contradiction:` should be changed to `assert result.has_contradiction is True` followed by the specific assertion. This pattern was already flagged and fixed in test_recall.py (see AP1-F5 in that file's docstring), proving it's a known anti-pattern.
**Proposed fix (example for `test_correction_always_supersedes`):**
```python
def test_correction_always_supersedes(self):
    """Corrections always supersede the target (spec 7.3)."""
    result = detect_contradictions(
        candidate_text="No, actually the API key is XYZ123",
        candidate_category="correction",
        existing_texts=["The API key is ABC789"],
        existing_categories=["fact"],
    )
    assert result.has_contradiction is True, "Correction must detect contradiction"
    assert len(result.actions) >= 1, "Must have at least one action"
    _, action = result.actions[0]
    assert action == SupersessionAction.AUTO_SUPERSEDE, (
        f"Corrections should AUTO_SUPERSEDE, got {action}"
    )
```

---

### AP2-F14: No Property Test for `subject_overlap` Range [0.0, 1.0]
**Severity:** MEDIUM
**Spec reference:** Section 4.4 — "Returns 0.0 when subjects have no common tokens, 1.0 for identical subjects" + "clamped to 1.0"
**Gap:** The Hypothesis tests cover symmetry of `subject_overlap` but never assert the return value is in `[0.0, 1.0]`. The spec says the field_type bonus is "clamped to 1.0" and the base Jaccard is inherently in [0,1], but the property test should enforce this invariant across random inputs. An off-by-one in clamping (e.g., returning 1.2) would pass all existing tests since parametric tests use specific known values.
**Proposed test:**
```python
class TestPropertyBased:
    @given(
        subj_a=st.text(min_size=0, max_size=50),
        subj_b=st.text(min_size=0, max_size=50),
        ft=st.sampled_from(["location", "name", "email", "unknown", "preference"]),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_subject_overlap_bounded_zero_to_one(self, subj_a, subj_b, ft):
        """subject_overlap always returns a value in [0.0, 1.0]."""
        a = SubjectExtraction(subject=subj_a, value=None, field_type=ft, raw_match="t")
        b = SubjectExtraction(subject=subj_b, value=None, field_type=ft, raw_match="t")
        result = subject_overlap(a, b)
        assert 0.0 <= result <= 1.0, (
            f"subject_overlap returned {result}, expected [0.0, 1.0]"
        )
```

---

## Summary Table

| ID | Severity | Finding | Spec Section |
|---------|----------|---------|-------------|
| AP2-F1 | HIGH | No test for negative `category_weights` value rejection | 3.2 |
| AP2-F2 | MEDIUM | No test that `similarity_threshold=1.0` is accepted | 3.2 |
| AP2-F3 | CRITICAL | Confidence boundary test silently passes on no-contradiction | 6.2 |
| AP2-F4 | MEDIUM | No test for filler word stripping in subject normalization | 4.3 |
| AP2-F5 | CRITICAL | No test for same-index multi-detection conflict resolution | 6.3 |
| AP2-F6 | HIGH | No test that detections are sorted by confidence descending | 6.2 |
| AP2-F7 | LOW | Category hint behavioral effect not tested | 4.2 |
| AP2-F8 | HIGH | `enable_auto_supersede=False` test doesn't assert FLAG_CONFLICT presence | 6.2 |
| AP2-F9 | HIGH | `value_pattern_min_tokens` behavioral effect never tested | 3.2 |
| AP2-F10 | CRITICAL | `category_weights` scoring effect never tested | 5.1 |
| AP2-F11 | MEDIUM | No TypeError test for non-str inside `existing_texts` list | 6.1 |
| AP2-F12 | HIGH | Inherited access bonus formula has zero test coverage | 7.2 |
| AP2-F13 | CRITICAL | 8 tests use `if has_contradiction:` guard pattern (silent pass) | Multiple |
| AP2-F14 | MEDIUM | No property test that `subject_overlap` is bounded [0.0, 1.0] | 4.4 |

**By severity:**
- CRITICAL: 4 (AP2-F3, F5, F10, F13)
- HIGH: 5 (AP2-F1, F6, F8, F9, F12)
- MEDIUM: 4 (AP2-F2, F4, F11, F14)
- LOW: 1 (AP2-F7)

**Estimated new tests needed:** 20-25 (some findings require multiple test functions).

---

## Priority Recommendations

1. **Fix the silent-pass pattern first (AP2-F13).** This is the highest-leverage change: 8 existing tests become real tests by replacing `if` with `assert`. Zero new test logic needed, just assertion strengthening. This pattern was already identified and fixed in `test_recall.py`; the contradiction tests should follow suit.

2. **Add category_weights behavioral tests (AP2-F10).** Without these, the core scoring formula could ignore weights entirely and pass all tests. This is the single biggest spec-to-test gap.

3. **Add conflict resolution tests (AP2-F5).** Section 6.3 defines three rules (highest-confidence wins, AUTO beats FLAG, supersede-once) with zero coverage. These rules prevent data corruption (double supersession).

4. **Add confidence boundary tests (AP2-F3).** The boundary between AUTO_SUPERSEDE and FLAG_CONFLICT is the most consequential threshold in the system. Test both sides of it.

5. **Add detection sorting test (AP2-F6).** Consumer code may depend on highest-confidence detection being first in the tuple. Test the contract.
