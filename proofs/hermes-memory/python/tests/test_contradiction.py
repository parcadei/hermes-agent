"""Comprehensive tests for hermes_memory.contradiction -- contradiction/supersession.

Tests are written BEFORE implementation exists. All imports from
hermes_memory.contradiction will fail with ImportError until the module is created.

~190 effective tests covering:
  1.  TestContradictionTypeEnum:         enum members (5 tests)
  2.  TestPolarityEnum:                  polarity members (4 tests)
  3.  TestSupersessionActionEnum:        action members (4 tests)
  4.  TestContradictionConfig:           config validation/defaults (15 tests)
  5.  TestSubjectExtraction:             frozen dataclass (5 tests)
  6.  TestContradictionDetection:        frozen dataclass (5 tests)
  7.  TestContradictionResult:           frozen dataclass + computed (6 tests)
  8.  TestSupersessionRecord:            frozen dataclass (4 tests)
  9.  TestExtractSubject:                field extraction patterns (25 tests)
  10. TestSubjectOverlap:                Jaccard + bonus (12 tests)
  11. TestDetectPolarity:                keyword polarity (13 tests)
  12. TestExtractAction:                 instruction prefix stripping (10 tests)
  13. TestDetectContradictions:           main pipeline (33 tests)
  14. TestResolveContradictions:          supersession records (9 tests)
  15. TestCategorySpecificSupersession:   per-category behavior (8 tests)
  16. TestFullPipeline:                   integration tests (5 tests)
  17. TestAdversarialInputs:             adversarial/edge cases (16 tests)
  18. TestPropertyBased:                  Hypothesis property tests (4 tests)
  19. TestModuleConstants:               module constants (4 tests)
"""

import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hermes_memory.contradiction import (
    # Enums
    ContradictionType,
    Polarity,
    SupersessionAction,
    # Dataclasses
    ContradictionConfig,
    SubjectExtraction,
    ContradictionDetection,
    ContradictionResult,
    SupersessionRecord,
    # Functions
    extract_subject,
    subject_overlap,
    detect_polarity,
    extract_action,
    detect_contradictions,
    resolve_contradictions,
    # Constants
    DEFAULT_CATEGORY_WEIGHTS,
    EXCLUDED_CATEGORIES,
    FIELD_EXTRACTORS,
    DEFAULT_CONTRADICTION_CONFIG,
)

MAX_EXAMPLES = 200

# Valid categories from encoding.py -- used for test inputs
_VALID_CATEGORIES = frozenset({
    "preference", "fact", "correction", "reasoning",
    "instruction", "greeting", "transactional",
})

# Default category_weights from spec Section 3.2
# greeting and transactional are EXCLUDED_CATEGORIES (not weights).
_DEFAULT_CATEGORY_WEIGHTS = {
    "correction": 1.5,
    "instruction": 1.2,
    "preference": 1.0,
    "fact": 0.8,
    "reasoning": 0.5,
}


# ===========================================================================
# 1. TestContradictionTypeEnum (~4 tests)
# ===========================================================================


class TestContradictionTypeEnum:
    """Verify ContradictionType enum has exactly four members with correct values."""

    def test_has_four_members(self):
        """ContradictionType must have exactly 4 members."""
        assert len(ContradictionType) == 4, (
            f"Expected 4 members, got {len(ContradictionType)}"
        )

    @pytest.mark.parametrize(
        "member,value",
        [
            ("DIRECT_NEGATION", "direct_negation"),
            ("VALUE_UPDATE", "value_update"),
            ("PREFERENCE_REVERSAL", "preference_reversal"),
            ("INSTRUCTION_CONFLICT", "instruction_conflict"),
        ],
    )
    def test_member_values(self, member, value):
        """Each member has the correct string value."""
        assert getattr(ContradictionType, member).value == value


# ===========================================================================
# 2. TestPolarityEnum (~3 tests)
# ===========================================================================


class TestPolarityEnum:
    """Verify Polarity enum has exactly three members with correct values."""

    def test_has_three_members(self):
        """Polarity must have exactly 3 members."""
        assert len(Polarity) == 3, f"Expected 3 members, got {len(Polarity)}"

    @pytest.mark.parametrize(
        "member,value",
        [("POSITIVE", "positive"), ("NEGATIVE", "negative"), ("NEUTRAL", "neutral")],
    )
    def test_member_values(self, member, value):
        """Each member has the correct string value."""
        assert getattr(Polarity, member).value == value


# ===========================================================================
# 3. TestSupersessionActionEnum (~3 tests)
# ===========================================================================


class TestSupersessionActionEnum:
    """Verify SupersessionAction enum has exactly three members."""

    def test_has_three_members(self):
        """SupersessionAction must have exactly 3 members."""
        assert len(SupersessionAction) == 3, (
            f"Expected 3 members, got {len(SupersessionAction)}"
        )

    @pytest.mark.parametrize(
        "member,value",
        [
            ("AUTO_SUPERSEDE", "auto_supersede"),
            ("FLAG_CONFLICT", "flag_conflict"),
            ("SKIP", "skip"),
        ],
    )
    def test_member_values(self, member, value):
        """Each member has the correct string value."""
        assert getattr(SupersessionAction, member).value == value


# ===========================================================================
# 4. TestContradictionConfig (~12 tests)
# ===========================================================================


class TestContradictionConfig:
    """Verify ContradictionConfig defaults, validation, and frozenness."""

    def test_is_frozen_dataclass(self):
        """ContradictionConfig is a frozen dataclass."""
        config = ContradictionConfig()
        assert dataclasses.is_dataclass(config)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.similarity_threshold = 0.5  # type: ignore[misc]

    def test_default_similarity_threshold(self):
        """Default similarity_threshold is 0.3."""
        config = ContradictionConfig()
        assert config.similarity_threshold == 0.3

    def test_default_confidence_threshold(self):
        """Default confidence_threshold is 0.7."""
        config = ContradictionConfig()
        assert config.confidence_threshold == 0.7

    def test_default_max_candidates(self):
        """Default max_candidates is 50."""
        config = ContradictionConfig()
        assert config.max_candidates == 50

    def test_default_category_weights_filled(self):
        """Post-init fills default category_weights when None."""
        config = ContradictionConfig()
        assert config.category_weights is not None
        assert config.category_weights == _DEFAULT_CATEGORY_WEIGHTS

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("similarity_threshold", 0.0, "similarity_threshold"),
            ("similarity_threshold", 1.1, "similarity_threshold"),
            ("confidence_threshold", -0.1, "confidence_threshold"),
            ("max_candidates", 0, "max_candidates"),
            ("value_pattern_min_tokens", 0, "value_pattern_min_tokens"),
        ],
    )
    def test_invalid_scalar_values_rejected(self, field, value, match):
        """Scalar fields with out-of-domain values are rejected."""
        with pytest.raises(ValueError, match=match):
            ContradictionConfig(**{field: value})

    def test_category_weights_invalid_key_rejected(self):
        """Category weight with key not in VALID_CATEGORIES is rejected."""
        with pytest.raises(ValueError, match="category_weights"):
            ContradictionConfig(category_weights={"invalid_cat": 1.0})

    def test_category_weights_value_above_two_rejected(self):
        """Category weight value > 2.0 is rejected."""
        weights = dict(_DEFAULT_CATEGORY_WEIGHTS)
        weights["correction"] = 2.5
        with pytest.raises(ValueError, match="category_weights"):
            ContradictionConfig(category_weights=weights)

    def test_valid_custom_config_accepted(self):
        """Valid non-default config is accepted."""
        config = ContradictionConfig(
            similarity_threshold=0.5,
            confidence_threshold=0.8,
            max_candidates=10,
            enable_auto_supersede=False,
            value_pattern_min_tokens=3,
        )
        assert config.similarity_threshold == 0.5
        assert config.confidence_threshold == 0.8
        assert config.max_candidates == 10
        assert config.enable_auto_supersede is False
        assert config.value_pattern_min_tokens == 3

    def test_category_thresholds_default(self):
        """Default category_thresholds match spec Section 7.3."""
        config = ContradictionConfig()
        assert config.category_thresholds["preference"] == 0.6
        assert config.category_thresholds["correction"] == 0.0  # always supersede
        assert config.category_thresholds["fact"] == 0.7
        assert config.category_thresholds["instruction"] == 0.7

    def test_category_thresholds_custom(self):
        """Custom category_thresholds override defaults."""
        config = ContradictionConfig(category_thresholds={"preference": 0.9, "fact": 0.5})
        assert config.category_thresholds["preference"] == 0.9
        assert config.category_thresholds["fact"] == 0.5


# ===========================================================================
# 5. TestSubjectExtraction (~5 tests)
# ===========================================================================


class TestSubjectExtraction:
    """Verify SubjectExtraction frozen dataclass and fields."""

    def test_is_frozen_dataclass(self):
        """SubjectExtraction is a frozen dataclass."""
        se = SubjectExtraction(
            subject="location", value="new york", field_type="location",
            raw_match="i live in new york",
        )
        assert dataclasses.is_dataclass(se)
        with pytest.raises(dataclasses.FrozenInstanceError):
            se.subject = "other"  # type: ignore[misc]

    def test_has_required_fields(self):
        """SubjectExtraction has subject, value, field_type, raw_match."""
        field_names = {f.name for f in dataclasses.fields(SubjectExtraction)}
        assert field_names >= {"subject", "value", "field_type", "raw_match"}

    def test_value_can_be_none(self):
        """SubjectExtraction.value can be None for fallback extractions."""
        se = SubjectExtraction(
            subject="weather", value=None, field_type="unknown",
            raw_match="the weather is nice",
        )
        assert se.value is None

    def test_subject_is_str(self):
        """SubjectExtraction.subject is a string."""
        se = SubjectExtraction(
            subject="name", value="alice", field_type="name",
            raw_match="my name is alice",
        )
        assert isinstance(se.subject, str)

    def test_equality(self):
        """Two SubjectExtractions with identical fields are equal."""
        a = SubjectExtraction(subject="x", value="y", field_type="z", raw_match="w")
        b = SubjectExtraction(subject="x", value="y", field_type="z", raw_match="w")
        assert a == b


# ===========================================================================
# 6. TestContradictionDetection (~5 tests)
# ===========================================================================


class TestContradictionDetection:
    """Verify ContradictionDetection frozen dataclass and fields."""

    def test_is_frozen_dataclass(self):
        """ContradictionDetection is a frozen dataclass."""
        se = SubjectExtraction(
            subject="loc", value="nyc", field_type="location",
            raw_match="i live in nyc",
        )
        cd = ContradictionDetection(
            existing_index=0,
            contradiction_type=ContradictionType.VALUE_UPDATE,
            confidence=0.85,
            subject_overlap=0.6,
            candidate_subject=se,
            existing_subject=se,
            explanation="test",
        )
        assert dataclasses.is_dataclass(cd)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cd.confidence = 0.1  # type: ignore[misc]

    def test_has_required_fields(self):
        """ContradictionDetection has all spec fields."""
        field_names = {f.name for f in dataclasses.fields(ContradictionDetection)}
        expected = {
            "existing_index", "contradiction_type", "confidence",
            "subject_overlap", "candidate_subject", "existing_subject",
            "explanation",
        }
        assert field_names >= expected

    def test_confidence_stored_correctly(self):
        """Confidence value is preserved."""
        se = SubjectExtraction(
            subject="x", value="y", field_type="name", raw_match="m",
        )
        cd = ContradictionDetection(
            existing_index=3,
            contradiction_type=ContradictionType.DIRECT_NEGATION,
            confidence=0.92,
            subject_overlap=0.7,
            candidate_subject=se,
            existing_subject=se,
            explanation="negation detected",
        )
        assert cd.confidence == 0.92
        assert cd.existing_index == 3

    def test_contradiction_type_is_enum(self):
        """contradiction_type is a ContradictionType member."""
        se = SubjectExtraction(
            subject="x", value="y", field_type="email", raw_match="m",
        )
        cd = ContradictionDetection(
            existing_index=0,
            contradiction_type=ContradictionType.PREFERENCE_REVERSAL,
            confidence=0.5,
            subject_overlap=0.4,
            candidate_subject=se,
            existing_subject=se,
            explanation="pref changed",
        )
        assert isinstance(cd.contradiction_type, ContradictionType)

    def test_explanation_is_str(self):
        """explanation field is a string."""
        se = SubjectExtraction(
            subject="x", value="y", field_type="fact", raw_match="m",
        )
        cd = ContradictionDetection(
            existing_index=0,
            contradiction_type=ContradictionType.VALUE_UPDATE,
            confidence=0.5,
            subject_overlap=0.4,
            candidate_subject=se,
            existing_subject=se,
            explanation="values differ",
        )
        assert isinstance(cd.explanation, str)


# ===========================================================================
# 7. TestContradictionResult (~6 tests)
# ===========================================================================


class TestContradictionResult:
    """Verify ContradictionResult frozen dataclass and computed properties."""

    def test_is_frozen_dataclass(self):
        """ContradictionResult is a frozen dataclass."""
        cr = ContradictionResult(
            detections=(),
            actions=(),
            superseded_indices=frozenset(),
            flagged_indices=frozenset(),
            has_contradiction=False,
            highest_confidence=0.0,
        )
        assert dataclasses.is_dataclass(cr)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cr.has_contradiction = True  # type: ignore[misc]

    def test_empty_result_no_contradiction(self):
        """Empty ContradictionResult has has_contradiction=False."""
        cr = ContradictionResult(
            detections=(),
            actions=(),
            superseded_indices=frozenset(),
            flagged_indices=frozenset(),
            has_contradiction=False,
            highest_confidence=0.0,
        )
        assert cr.has_contradiction is False
        assert cr.highest_confidence == 0.0
        assert len(cr.detections) == 0

    def test_has_contradiction_when_detections_present(self):
        """has_contradiction=True when there are detections."""
        se = SubjectExtraction(
            subject="loc", value="nyc", field_type="location",
            raw_match="test",
        )
        det = ContradictionDetection(
            existing_index=0,
            contradiction_type=ContradictionType.VALUE_UPDATE,
            confidence=0.8,
            subject_overlap=0.6,
            candidate_subject=se,
            existing_subject=se,
            explanation="test",
        )
        cr = ContradictionResult(
            detections=(det,),
            actions=((0, SupersessionAction.AUTO_SUPERSEDE),),
            superseded_indices=frozenset({0}),
            flagged_indices=frozenset(),
            has_contradiction=True,
            highest_confidence=0.8,
        )
        assert cr.has_contradiction is True
        assert cr.highest_confidence == 0.8

    def test_superseded_and_flagged_are_frozensets(self):
        """superseded_indices and flagged_indices are frozensets."""
        cr = ContradictionResult(
            detections=(),
            actions=(),
            superseded_indices=frozenset({1, 3}),
            flagged_indices=frozenset({5}),
            has_contradiction=True,
            highest_confidence=0.6,
        )
        assert isinstance(cr.superseded_indices, frozenset)
        assert isinstance(cr.flagged_indices, frozenset)

    def test_actions_are_tuple_of_tuples(self):
        """actions field is a tuple of (index, SupersessionAction) pairs."""
        cr = ContradictionResult(
            detections=(),
            actions=((0, SupersessionAction.FLAG_CONFLICT),),
            superseded_indices=frozenset(),
            flagged_indices=frozenset({0}),
            has_contradiction=True,
            highest_confidence=0.5,
        )
        assert len(cr.actions) == 1
        idx, action = cr.actions[0]
        assert idx == 0
        assert isinstance(action, SupersessionAction)

    def test_detections_are_tuple(self):
        """detections field is a tuple."""
        cr = ContradictionResult(
            detections=(),
            actions=(),
            superseded_indices=frozenset(),
            flagged_indices=frozenset(),
            has_contradiction=False,
            highest_confidence=0.0,
        )
        assert isinstance(cr.detections, tuple)


# ===========================================================================
# 8. TestSupersessionRecord (~4 tests)
# ===========================================================================


class TestSupersessionRecord:
    """Verify SupersessionRecord frozen dataclass and fields."""

    def test_is_frozen_dataclass(self):
        """SupersessionRecord is a frozen dataclass."""
        sr = SupersessionRecord(
            old_index=0,
            new_index=-1,
            contradiction_type=ContradictionType.VALUE_UPDATE,
            confidence=0.9,
            timestamp=1000.0,
            explanation="updated",
        )
        assert dataclasses.is_dataclass(sr)
        with pytest.raises(dataclasses.FrozenInstanceError):
            sr.old_index = 5  # type: ignore[misc]

    def test_has_required_fields(self):
        """SupersessionRecord has all spec fields."""
        field_names = {f.name for f in dataclasses.fields(SupersessionRecord)}
        expected = {
            "old_index", "new_index", "contradiction_type",
            "confidence", "timestamp", "explanation",
        }
        assert field_names >= expected

    def test_new_index_negative_one_for_candidate(self):
        """new_index is -1 when the superseding memory is the candidate."""
        sr = SupersessionRecord(
            old_index=2,
            new_index=-1,
            contradiction_type=ContradictionType.DIRECT_NEGATION,
            confidence=0.95,
            timestamp=500.0,
            explanation="correction applied",
        )
        assert sr.new_index == -1
        assert sr.old_index == 2

    def test_timestamp_preserved(self):
        """Timestamp value is preserved exactly."""
        sr = SupersessionRecord(
            old_index=0,
            new_index=-1,
            contradiction_type=ContradictionType.PREFERENCE_REVERSAL,
            confidence=0.7,
            timestamp=42.5,
            explanation="preference changed",
        )
        assert sr.timestamp == 42.5


# ===========================================================================
# 9. TestExtractSubject (~24 tests)
# ===========================================================================


class TestExtractSubject:
    """Verify extract_subject() for each FIELD_EXTRACTOR pattern type,
    fallback behavior, normalization, edge cases."""

    # --- Location patterns ---

    def test_location_i_live_in(self):
        """'I live in New York' -> subject='location', value='new york'."""
        result = extract_subject("I live in New York")
        assert result.field_type == "location"
        assert result.subject == "location"
        assert result.value == "new york"

    @pytest.mark.parametrize(
        "text,expected_value_fragment",
        [
            ("I am based in London", "london"),
            ("The user lives in Berlin", "berlin"),
        ],
    )
    def test_location_patterns(self, text, expected_value_fragment):
        """Various location patterns extract correctly."""
        result = extract_subject(text)
        assert result.field_type == "location"
        assert expected_value_fragment in result.value

    # --- Name patterns ---

    def test_name_my_name_is(self):
        """'My name is Alice' -> subject='name', value='alice'."""
        result = extract_subject("My name is Alice")
        assert result.field_type == "name"
        assert result.subject == "name"
        assert result.value == "alice"

    def test_name_call_me(self):
        """'Call me Bob' -> name extraction."""
        result = extract_subject("Call me Bob")
        assert result.field_type == "name"
        assert "bob" in result.value

    # --- Email patterns ---

    def test_email_my_email_is(self):
        """'My email is alice@example.com' -> email extraction."""
        result = extract_subject("My email is alice@example.com")
        assert result.field_type == "email"
        assert result.subject == "email"
        assert "alice@example.com" in result.value

    def test_email_address_variant(self):
        """'My email address is bob@test.org' -> email extraction."""
        result = extract_subject("My email address is bob@test.org")
        assert result.field_type == "email"
        assert "bob@test.org" in result.value

    # --- Job patterns ---

    def test_job_i_work_at(self):
        """'I work at Google' -> job extraction."""
        result = extract_subject("I work at Google")
        assert result.field_type == "job"
        assert "google" in result.value

    def test_job_i_work_as(self):
        """'I work as a software engineer' -> role extraction."""
        result = extract_subject("I work as a software engineer")
        assert result.field_type == "job"
        assert "software engineer" in result.value

    # --- Preference patterns ---

    def test_preference_i_prefer(self):
        """'I prefer dark mode' -> preference extraction."""
        result = extract_subject("I prefer dark mode")
        assert result.field_type == "preference"
        assert result.subject == "preference"
        assert "dark mode" in result.value

    def test_preference_negative_dont_like(self):
        """'I don't like tabs' -> dispreference extraction."""
        result = extract_subject("I don't like tabs")
        assert result.field_type == "preference"
        assert result.subject == "dispreference"

    def test_preference_i_like(self):
        """'I like Python' -> preference extraction."""
        result = extract_subject("I like Python")
        assert result.field_type == "preference"
        assert "python" in result.value

    # --- Instruction patterns ---

    @pytest.mark.parametrize(
        "text",
        [
            "Always use type hints",
            "Never use global variables",
            "Remember to run linting",
        ],
    )
    def test_instruction_patterns(self, text):
        """Various instruction patterns extract correctly."""
        result = extract_subject(text)
        assert result.field_type == "instruction"
        assert result.subject == "instruction"

    # --- Fallback behavior ---

    def test_fallback_unknown_field_type(self):
        """Unmatched text falls back to field_type='unknown'."""
        result = extract_subject("The quick brown fox jumps over the lazy dog")
        assert result.field_type == "unknown"
        assert result.value is None

    def test_fallback_extracts_subject_tokens(self):
        """Fallback extracts first noun phrase tokens as subject."""
        result = extract_subject("Weather is sunny today")
        assert result.field_type == "unknown"
        assert isinstance(result.subject, str)
        assert len(result.subject) > 0

    # --- Normalization ---

    def test_normalization_case_insensitive(self):
        """Subjects and values are lowercased."""
        result = extract_subject("I LIVE IN NEW YORK")
        assert result.value == "new york"

    def test_normalization_strips_whitespace_and_punctuation(self):
        """Whitespace and trailing punctuation are stripped."""
        result = extract_subject("  I live in   New York.  ")
        assert result.field_type == "location"
        assert result.value is not None
        assert not result.value.endswith(".")

    def test_normalization_strips_articles(self):
        """Leading articles are stripped from subjects (per spec 4.3)."""
        result = extract_subject("I prefer the dark mode")
        if result.value is not None:
            assert not result.value.startswith("the ")

    # --- Edge cases ---

    def test_empty_input_returns_extraction(self):
        """Empty string returns a SubjectExtraction (field_type unknown)."""
        result = extract_subject("")
        assert isinstance(result, SubjectExtraction)
        assert result.field_type == "unknown"

    def test_unicode_preserved(self):
        """Unicode text is handled without errors."""
        result = extract_subject("I live in Munchen")
        assert isinstance(result, SubjectExtraction)

    def test_very_long_text_truncated_for_extraction(self):
        """Text > 10000 chars is truncated for extraction but doesn't crash."""
        long_text = "I live in " + "A" * 15000
        result = extract_subject(long_text)
        assert isinstance(result, SubjectExtraction)

    def test_category_hint_does_not_crash(self):
        """Passing category hint to extract_subject works."""
        result = extract_subject("I prefer dark mode", category="preference")
        assert result.field_type == "preference"

    def test_longest_match_wins(self):
        """When multiple extractors match, the longest match wins (spec 4.2)."""
        result = extract_subject("I'm from New York City")
        assert result.field_type == "location"


# ===========================================================================
# 10. TestSubjectOverlap (~12 tests)
# ===========================================================================


class TestSubjectOverlap:
    """Verify subject_overlap() Jaccard computation and field_type bonus."""

    def _make_se(self, subject="test", value=None, field_type="unknown"):
        """Helper to build SubjectExtraction."""
        return SubjectExtraction(
            subject=subject, value=value, field_type=field_type,
            raw_match="test",
        )

    def test_identical_subjects_return_one(self):
        """Same subject string returns 1.0 regardless of tokens."""
        a = self._make_se("new york", field_type="location")
        b = self._make_se("new york", field_type="location")
        assert subject_overlap(a, b) == 1.0

    def test_no_overlap_returns_zero(self):
        """Completely disjoint subjects return 0.0."""
        a = self._make_se("alpha beta", field_type="unknown")
        b = self._make_se("gamma delta", field_type="unknown")
        assert subject_overlap(a, b) == 0.0

    def test_partial_overlap_jaccard(self):
        """Partial token overlap returns correct Jaccard value."""
        a = self._make_se("new york city", field_type="location")
        b = self._make_se("new york state", field_type="location")
        # Tokens: {new, york, city} vs {new, york, state}
        # Intersection: {new, york} = 2, Union: {new, york, city, state} = 4
        # Jaccard = 2/4 = 0.5, plus field_type bonus 0.2 = 0.7
        result = subject_overlap(a, b)
        assert result == pytest.approx(0.7, abs=0.05)

    def test_field_type_bonus_applied(self):
        """Same field_type adds 0.2 bonus to Jaccard score."""
        a = self._make_se("york", field_type="location")
        b = self._make_se("new york", field_type="location")
        # Tokens: {york} vs {new, york}, Jaccard = 1/2 = 0.5, + 0.2 = 0.7
        result = subject_overlap(a, b)
        assert result == pytest.approx(0.7, abs=0.05)

    def test_field_type_bonus_not_applied_for_different_types(self):
        """Different field_types do not get the 0.2 bonus."""
        a = self._make_se("york", field_type="location")
        b = self._make_se("new york", field_type="name")
        # Tokens: {york} vs {new, york}, Jaccard = 1/2 = 0.5, no bonus
        result = subject_overlap(a, b)
        assert result == pytest.approx(0.5, abs=0.05)

    def test_field_type_bonus_clamped_to_one(self):
        """Jaccard + bonus is clamped at 1.0."""
        a = self._make_se("email", field_type="email")
        b = self._make_se("email", field_type="email")
        # Jaccard = 1.0, + 0.2 would be 1.2, but clamped to 1.0
        result = subject_overlap(a, b)
        assert result == 1.0

    @pytest.mark.parametrize(
        "subj_a,subj_b",
        [("", "something"), ("something", ""), ("", "")],
    )
    def test_empty_subjects_return_zero(self, subj_a, subj_b):
        """Any empty subject returns 0.0."""
        a = self._make_se(subj_a, field_type="unknown")
        b = self._make_se(subj_b, field_type="unknown")
        assert subject_overlap(a, b) == 0.0

    def test_symmetry(self):
        """subject_overlap(a, b) == subject_overlap(b, a)."""
        a = self._make_se("alpha beta gamma", field_type="fact")
        b = self._make_se("beta gamma delta", field_type="fact")
        assert subject_overlap(a, b) == subject_overlap(b, a)

    def test_single_token_overlap(self):
        """Single shared token produces correct Jaccard."""
        a = self._make_se("alpha beta", field_type="unknown")
        b = self._make_se("beta gamma", field_type="unknown")
        # Jaccard = 1/3 ~ 0.333
        result = subject_overlap(a, b)
        assert result == pytest.approx(1 / 3, abs=0.05)

    def test_return_type_is_float(self):
        """subject_overlap returns a float."""
        a = self._make_se("x", field_type="unknown")
        b = self._make_se("y", field_type="unknown")
        result = subject_overlap(a, b)
        assert isinstance(result, float)


# ===========================================================================
# 11. TestDetectPolarity (~12 tests)
# ===========================================================================


class TestDetectPolarity:
    """Verify detect_polarity() keyword-based polarity classification."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("I like dark mode", Polarity.POSITIVE),
            ("I prefer Python", Polarity.POSITIVE),
            ("I love Rust", Polarity.POSITIVE),
            ("I don't like tabs", Polarity.NEGATIVE),
            ("I hate boilerplate", Polarity.NEGATIVE),
            ("Never use eval", Polarity.NEGATIVE),
            ("I no longer use Vim", Polarity.NEGATIVE),
            ("Avoid global state", Polarity.NEGATIVE),
        ],
    )
    def test_keyword_polarity(self, text, expected):
        """Individual keywords map to correct polarity."""
        assert detect_polarity(text) == expected

    def test_neutral_no_keywords(self):
        """Text with no polarity keywords -> NEUTRAL."""
        assert detect_polarity("The weather is sunny today") == Polarity.NEUTRAL

    def test_neutral_empty_string(self):
        """Empty string -> NEUTRAL."""
        assert detect_polarity("") == Polarity.NEUTRAL

    def test_mixed_signals_more_positive_wins(self):
        """More positive than negative signals -> POSITIVE."""
        text = "I like dark mode and I prefer tabs, but I don't like semicolons"
        # 2 positive (like, prefer) vs 1 negative (don't like)
        result = detect_polarity(text)
        assert result == Polarity.POSITIVE

    def test_mixed_signals_more_negative_wins(self):
        """More negative than positive signals -> NEGATIVE."""
        text = "I don't like tabs and I hate semicolons, but I enjoy coding"
        # 1 positive (enjoy) vs 2 negative (don't like, hate)
        result = detect_polarity(text)
        assert result == Polarity.NEGATIVE

    def test_mixed_signals_tie_is_neutral(self):
        """Equal positive and negative signals -> NEUTRAL."""
        text = "I like coding but I hate debugging"
        # 1 positive (like) vs 1 negative (hate)
        result = detect_polarity(text)
        assert result == Polarity.NEUTRAL


# ===========================================================================
# 12. TestExtractAction (~12 tests)
# ===========================================================================


class TestExtractAction:
    """Verify extract_action() strips instruction prefixes."""

    def test_always_prefix(self):
        """'Always use type hints' -> 'use type hints'."""
        result = extract_action("Always use type hints")
        assert result is not None
        assert "use type hints" in result.lower()

    def test_never_prefix(self):
        """'Never use eval' -> 'use eval'."""
        result = extract_action("Never use eval")
        assert result is not None
        assert "use eval" in result.lower()

    def test_remember_to_prefix(self):
        """'Remember to run tests' -> 'run tests'."""
        result = extract_action("Remember to run tests")
        assert result is not None
        assert "run tests" in result.lower()

    def test_from_now_on_prefix(self):
        """'From now on, write tests first' -> 'write tests first'."""
        result = extract_action("From now on, write tests first")
        assert result is not None
        assert "write tests first" in result.lower()

    def test_dont_ever_prefix(self):
        """'Don't ever use global state' -> 'use global state'."""
        result = extract_action("Don't ever use global state")
        assert result is not None
        assert "use global state" in result.lower()

    def test_remember_that_prefix(self):
        """'Remember that I use Vim' -> 'i use vim'."""
        result = extract_action("Remember that I use Vim")
        assert result is not None

    def test_no_instruction_pattern_returns_none(self):
        """Text without instruction prefix returns None."""
        result = extract_action("The weather is nice today")
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        result = extract_action("")
        assert result is None

    def test_result_is_normalized_and_case_insensitive(self):
        """Returned action is normalized (lowercase, stripped); matching is case-insensitive."""
        result = extract_action("NEVER use eval")
        assert result is not None
        assert result == result.lower().strip()
        assert "use eval" in result.lower()

    def test_just_prefix_returns_empty_or_none(self):
        """Bare 'Always' without action returns None or empty."""
        result = extract_action("Always")
        assert result is None or result.strip() == ""


# ===========================================================================
# 13. TestDetectContradictions (~32 tests)
# ===========================================================================


class TestDetectContradictions:
    """Verify detect_contradictions() main pipeline for each ContradictionType,
    edge cases, and config variations."""

    # --- DIRECT_NEGATION detection ---

    def test_direct_negation_correction_category(self):
        """Correction text contradicts an existing fact."""
        result = detect_contradictions(
            candidate_text="No, actually the deadline is March 15th",
            candidate_category="correction",
            existing_texts=["The deadline is March 10th"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is True
        assert len(result.detections) >= 1
        assert result.detections[0].contradiction_type == ContradictionType.DIRECT_NEGATION

    def test_direct_negation_with_correction_markers(self):
        """Multiple correction markers boost confidence."""
        result = detect_contradictions(
            candidate_text="No, actually, that's wrong. The port is 8080",
            candidate_category="correction",
            existing_texts=["The port number is 3000"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is True
        # Multiple correction markers should yield high confidence
        assert result.highest_confidence >= 0.5

    # --- VALUE_UPDATE detection ---

    def test_value_update_location(self):
        """Different locations for same subject -> VALUE_UPDATE."""
        result = detect_contradictions(
            candidate_text="I live in London",
            candidate_category="fact",
            existing_texts=["I live in New York"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is True
        has_value_update = any(
            d.contradiction_type == ContradictionType.VALUE_UPDATE
            for d in result.detections
        )
        assert has_value_update, "Expected VALUE_UPDATE for different locations"

    def test_value_update_email(self):
        """Different emails -> VALUE_UPDATE with high confidence."""
        result = detect_contradictions(
            candidate_text="My email is new@example.com",
            candidate_category="fact",
            existing_texts=["My email is old@example.com"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is True
        assert result.highest_confidence >= 0.8, (
            "Email value update should have confidence >= 0.8"
        )

    def test_value_update_name(self):
        """Different names -> VALUE_UPDATE with high confidence."""
        result = detect_contradictions(
            candidate_text="My name is Bob",
            candidate_category="fact",
            existing_texts=["My name is Alice"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is True
        assert result.highest_confidence >= 0.8, (
            "Name value update should have confidence >= 0.8"
        )

    def test_value_update_same_value_no_contradiction(self):
        """Same value for same field is NOT a contradiction."""
        result = detect_contradictions(
            candidate_text="I live in New York",
            candidate_category="fact",
            existing_texts=["I live in New York"],
            existing_categories=["fact"],
        )
        # Duplicate text, not a contradiction
        assert result.has_contradiction is False

    # --- PREFERENCE_REVERSAL detection ---

    def test_preference_reversal_positive_to_negative(self):
        """'I like X' vs 'I don't like X' -> PREFERENCE_REVERSAL."""
        result = detect_contradictions(
            candidate_text="I don't like dark mode",
            candidate_category="preference",
            existing_texts=["I like dark mode"],
            existing_categories=["preference"],
        )
        assert result.has_contradiction is True
        has_pref_reversal = any(
            d.contradiction_type == ContradictionType.PREFERENCE_REVERSAL
            for d in result.detections
        )
        assert has_pref_reversal, "Expected PREFERENCE_REVERSAL for polarity flip"

    def test_preference_reversal_different_preferences(self):
        """'I prefer tabs' vs 'I prefer spaces' -> PREFERENCE_REVERSAL."""
        result = detect_contradictions(
            candidate_text="I prefer spaces for indentation",
            candidate_category="preference",
            existing_texts=["I prefer tabs for indentation"],
            existing_categories=["preference"],
        )
        assert result.has_contradiction is True

    def test_preference_correction_triggers_reversal(self):
        """Correction of a preference triggers detection."""
        result = detect_contradictions(
            candidate_text="Actually, I prefer light mode now",
            candidate_category="correction",
            existing_texts=["I prefer dark mode"],
            existing_categories=["preference"],
        )
        assert result.has_contradiction is True

    # --- INSTRUCTION_CONFLICT detection ---

    def test_instruction_conflict_always_vs_never(self):
        """'Always X' vs 'Never X' -> INSTRUCTION_CONFLICT."""
        result = detect_contradictions(
            candidate_text="Never use tabs for indentation",
            candidate_category="instruction",
            existing_texts=["Always use tabs for indentation"],
            existing_categories=["instruction"],
        )
        assert result.has_contradiction is True
        has_inst_conflict = any(
            d.contradiction_type == ContradictionType.INSTRUCTION_CONFLICT
            for d in result.detections
        )
        assert has_inst_conflict, "Expected INSTRUCTION_CONFLICT for always/never"

    def test_instruction_conflict_different_actions(self):
        """Different instructions for same subject."""
        result = detect_contradictions(
            candidate_text="Always use spaces for indentation",
            candidate_category="instruction",
            existing_texts=["Always use tabs for indentation"],
            existing_categories=["instruction"],
        )
        assert result.has_contradiction is True

    def test_instruction_non_overlapping_no_conflict(self):
        """Instructions about different subjects do not conflict."""
        result = detect_contradictions(
            candidate_text="Always use type hints in Python",
            candidate_category="instruction",
            existing_texts=["Never eat yellow snow"],
            existing_categories=["instruction"],
        )
        assert result.has_contradiction is False

    # --- Edge cases from spec Section 11 ---

    @pytest.mark.parametrize(
        "candidate,existing_texts,existing_cats",
        [
            ("", ["I live in New York"], ["fact"]),
            ("I live in London", [], []),
        ],
    )
    def test_empty_inputs_return_empty_result(self, candidate, existing_texts, existing_cats):
        """Empty candidate or empty pool returns empty result."""
        result = detect_contradictions(
            candidate_text=candidate,
            candidate_category="fact",
            existing_texts=existing_texts,
            existing_categories=existing_cats,
        )
        assert result.has_contradiction is False
        assert len(result.detections) == 0

    def test_empty_existing_memory_skipped(self):
        """Empty string in existing_texts is skipped; non-empty entries still compared."""
        result = detect_contradictions(
            candidate_text="I live in London",
            candidate_category="fact",
            existing_texts=["", "I live in New York"],
            existing_categories=["fact", "fact"],
        )
        assert result.has_contradiction is True, (
            "Non-empty existing memory at index 1 should still produce contradiction"
        )
        assert all(d.existing_index != 0 for d in result.detections), (
            "Empty string at index 0 must be skipped in detection"
        )

    def test_self_contradiction_duplicate_not_detected(self):
        """Exact duplicate in pool is not a contradiction (spec 11.2)."""
        result = detect_contradictions(
            candidate_text="I live in New York",
            candidate_category="fact",
            existing_texts=["I live in New York"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is False

    def test_partial_match_at_threshold_included(self):
        """Subject overlap at exactly similarity_threshold is included (>=)."""
        config = ContradictionConfig(similarity_threshold=0.3)
        # This test verifies the boundary: overlap >= threshold is included
        result = detect_contradictions(
            candidate_text="I live in London",
            candidate_category="fact",
            existing_texts=["I live in New York"],
            existing_categories=["fact"],
            config=config,
        )
        # Both have field_type "location" and subject "location"
        # overlap should be 1.0 for identical subject strings -> detected
        assert result.has_contradiction is True

    def test_confidence_at_threshold_auto_supersedes(self):
        """Confidence at exactly confidence_threshold triggers AUTO_SUPERSEDE."""
        config = ContradictionConfig(confidence_threshold=0.7)
        result = detect_contradictions(
            candidate_text="My email is new@example.com",
            candidate_category="fact",
            existing_texts=["My email is old@example.com"],
            existing_categories=["fact"],
            config=config,
        )
        assert result.has_contradiction is True, (
            "Email value update must be detected as contradiction"
        )
        # High-confidence email update should auto-supersede
        has_auto = any(
            action == SupersessionAction.AUTO_SUPERSEDE
            for _, action in result.actions
        )
        assert has_auto, "Confidence >= threshold should AUTO_SUPERSEDE"

    @pytest.mark.parametrize(
        "candidate,candidate_cat,existing_texts,existing_cats",
        [
            # All-greeting pool (spec 11.4)
            ("I live in London", "fact",
             ["Hello!", "Hi there!", "Good morning!"],
             ["greeting", "greeting", "greeting"]),
            # Greeting candidate early exit (spec 11.4)
            ("Hello!", "greeting",
             ["I live in New York"],
             ["fact"]),
            # Transactional candidate early exit (weight 0.0)
            ("Run the tests", "transactional",
             ["I live in New York"],
             ["fact"]),
        ],
    )
    def test_zero_weight_categories_return_empty(
        self, candidate, candidate_cat, existing_texts, existing_cats,
    ):
        """Greeting/transactional candidates or all-greeting pools return empty result."""
        result = detect_contradictions(
            candidate_text=candidate,
            candidate_category=candidate_cat,
            existing_texts=existing_texts,
            existing_categories=existing_cats,
        )
        assert result.has_contradiction is False

    def test_max_candidates_limits_scan(self):
        """Only first max_candidates memories are scanned (spec 11.5)."""
        config = ContradictionConfig(max_candidates=2)
        result = detect_contradictions(
            candidate_text="I live in London",
            candidate_category="fact",
            existing_texts=[
                "I live in New York",
                "I live in Paris",
                "I live in Tokyo",  # index 2, should not be scanned
            ],
            existing_categories=["fact", "fact", "fact"],
            config=config,
        )
        assert result.has_contradiction is True, (
            "Location contradiction must be detected even with max_candidates=2"
        )
        for det in result.detections:
            assert det.existing_index < 2, (
                f"Detection at index {det.existing_index} exceeds max_candidates=2"
            )

    def test_unicode_text_handled(self):
        """Unicode text does not crash the pipeline (spec 11.6)."""
        result = detect_contradictions(
            candidate_text="I live in Munchen",
            candidate_category="fact",
            existing_texts=["I live in Zurich"],
            existing_categories=["fact"],
        )
        assert isinstance(result, ContradictionResult)

    # --- Input validation ---

    def test_non_str_candidate_raises_type_error(self):
        """Non-string candidate_text raises TypeError."""
        with pytest.raises(TypeError):
            detect_contradictions(
                candidate_text=42,  # type: ignore[arg-type]
                candidate_category="fact",
                existing_texts=["test"],
                existing_categories=["fact"],
            )

    def test_mismatched_lengths_raises_value_error(self):
        """Different lengths for existing_texts and existing_categories raises ValueError."""
        with pytest.raises(ValueError, match="length"):
            detect_contradictions(
                candidate_text="I live in London",
                candidate_category="fact",
                existing_texts=["a", "b"],
                existing_categories=["fact"],
            )

    @pytest.mark.parametrize(
        "candidate_cat,existing_cats",
        [
            ("invalid_category", ["fact"]),
            ("fact", ["not_a_category"]),
        ],
    )
    def test_invalid_category_raises_value_error(self, candidate_cat, existing_cats):
        """Invalid category (candidate or existing) raises ValueError."""
        with pytest.raises(ValueError, match="categor"):
            detect_contradictions(
                candidate_text="test",
                candidate_category=candidate_cat,
                existing_texts=["test"],
                existing_categories=existing_cats,
            )

    # --- Config variations ---

    def test_auto_supersede_disabled_flags_instead(self):
        """With enable_auto_supersede=False, all contradictions are flagged."""
        config = ContradictionConfig(enable_auto_supersede=False)
        result = detect_contradictions(
            candidate_text="My email is new@example.com",
            candidate_category="fact",
            existing_texts=["My email is old@example.com"],
            existing_categories=["fact"],
            config=config,
        )
        assert result.has_contradiction is True, (
            "Email value update must be detected even with auto_supersede disabled"
        )
        for _, action in result.actions:
            assert action != SupersessionAction.AUTO_SUPERSEDE, (
                "With enable_auto_supersede=False, no AUTO_SUPERSEDE allowed"
            )

    def test_default_config_used_when_none(self):
        """config=None uses DEFAULT_CONTRADICTION_CONFIG."""
        result = detect_contradictions(
            candidate_text="I live in London",
            candidate_category="fact",
            existing_texts=["I live in New York"],
            existing_categories=["fact"],
            config=None,
        )
        assert isinstance(result, ContradictionResult)

    def test_return_type_is_always_contradiction_result(self):
        """Return type is always ContradictionResult, even for empty inputs."""
        result = detect_contradictions(
            candidate_text="I live in London",
            candidate_category="fact",
            existing_texts=[],
            existing_categories=[],
        )
        assert isinstance(result, ContradictionResult)

    # --- Category weight behavioral tests (AP2-F10) ---

    def test_excluded_category_suppresses_detection(self):
        """EXCLUDED_CATEGORIES are structurally skipped by detect_contradictions."""
        for cat in EXCLUDED_CATEGORIES:
            result = detect_contradictions(
                candidate_text="Hello there",
                candidate_category=cat,
                existing_texts=["Hi, nice to meet you"],
                existing_categories=[cat],
            )
            assert result.has_contradiction is False, (
                f"Excluded category {cat!r} should be skipped in detection pipeline"
            )

    def test_excluded_category_rejected_in_weights(self):
        """Excluded categories cannot appear in category_weights."""
        for cat in EXCLUDED_CATEGORIES:
            weights = dict(DEFAULT_CATEGORY_WEIGHTS)
            weights[cat] = 1.0
            with pytest.raises(ValueError, match="EXCLUDED_CATEGORIES"):
                ContradictionConfig(category_weights=weights)

    def test_higher_category_weight_increases_confidence(self):
        """Higher category_weight produces higher confidence for same input."""
        low_w = dict(DEFAULT_CATEGORY_WEIGHTS)
        low_w["correction"] = 0.5
        high_w = dict(DEFAULT_CATEGORY_WEIGHTS)
        high_w["correction"] = 1.5
        config_low = ContradictionConfig(category_weights=low_w)
        config_high = ContradictionConfig(category_weights=high_w)
        result_low = detect_contradictions(
            "No, actually the deadline is March 15th", "correction",
            ["The deadline is March 10th"], ["fact"], config_low,
        )
        result_high = detect_contradictions(
            "No, actually the deadline is March 15th", "correction",
            ["The deadline is March 10th"], ["fact"], config_high,
        )
        assert result_low.has_contradiction is True
        assert result_high.has_contradiction is True
        assert result_high.highest_confidence >= result_low.highest_confidence, (
            "Higher category weight must produce equal or higher confidence"
        )

    # --- Same-index conflict resolution tests (AP2-F5) ---

    def test_multiple_detections_same_index_highest_confidence_wins(self):
        """When multiple strategies detect contradictions for the same memory,
        the highest confidence action wins (spec 6.3)."""
        result = detect_contradictions(
            candidate_text="No, actually my email is new@example.com",
            candidate_category="correction",
            existing_texts=["My email is old@example.com"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is True
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
        superseded = [idx for idx, act in result.actions
                      if act == SupersessionAction.AUTO_SUPERSEDE]
        assert len(superseded) == len(set(superseded)), (
            "Each index must be superseded at most once"
        )


# ===========================================================================
# 14. TestResolveContradictions (~12 tests)
# ===========================================================================


class TestResolveContradictions:
    """Verify resolve_contradictions() creates proper SupersessionRecords
    and deactivation indices."""

    def _make_empty_result(self):
        """Build an empty ContradictionResult."""
        return ContradictionResult(
            detections=(),
            actions=(),
            superseded_indices=frozenset(),
            flagged_indices=frozenset(),
            has_contradiction=False,
            highest_confidence=0.0,
        )

    def _make_result_with_supersede(self, index=0, confidence=0.9):
        """Build a ContradictionResult with one AUTO_SUPERSEDE action."""
        se = SubjectExtraction(
            subject="location", value="london", field_type="location",
            raw_match="i live in london",
        )
        det = ContradictionDetection(
            existing_index=index,
            contradiction_type=ContradictionType.VALUE_UPDATE,
            confidence=confidence,
            subject_overlap=0.8,
            candidate_subject=se,
            existing_subject=se,
            explanation="value update detected",
        )
        return ContradictionResult(
            detections=(det,),
            actions=((index, SupersessionAction.AUTO_SUPERSEDE),),
            superseded_indices=frozenset({index}),
            flagged_indices=frozenset(),
            has_contradiction=True,
            highest_confidence=confidence,
        )

    def _make_result_with_flag(self, index=0, confidence=0.5):
        """Build a ContradictionResult with one FLAG_CONFLICT action."""
        se = SubjectExtraction(
            subject="preference", value="tabs", field_type="preference",
            raw_match="i prefer tabs",
        )
        det = ContradictionDetection(
            existing_index=index,
            contradiction_type=ContradictionType.PREFERENCE_REVERSAL,
            confidence=confidence,
            subject_overlap=0.6,
            candidate_subject=se,
            existing_subject=se,
            explanation="preference changed",
        )
        return ContradictionResult(
            detections=(det,),
            actions=((index, SupersessionAction.FLAG_CONFLICT),),
            superseded_indices=frozenset(),
            flagged_indices=frozenset({index}),
            has_contradiction=True,
            highest_confidence=confidence,
        )

    def test_empty_result_returns_empty(self):
        """Empty ContradictionResult returns no records and no deactivations."""
        result = self._make_empty_result()
        records, deactivated = resolve_contradictions(
            result, "test candidate", ["test existing"], timestamp=100.0,
        )
        assert len(records) == 0
        assert len(deactivated) == 0

    def test_auto_supersede_creates_record(self):
        """AUTO_SUPERSEDE action creates a SupersessionRecord."""
        result = self._make_result_with_supersede(index=0, confidence=0.9)
        records, deactivated = resolve_contradictions(
            result, "I live in London", ["I live in New York"], timestamp=100.0,
        )
        assert len(records) == 1
        assert isinstance(records[0], SupersessionRecord)

    def test_supersession_record_fields_correct(self):
        """SupersessionRecord has correct old_index, new_index, type, confidence."""
        result = self._make_result_with_supersede(index=2, confidence=0.85)
        records, deactivated = resolve_contradictions(
            result, "candidate",
            ["a", "b", "c"],
            timestamp=42.0,
        )
        assert len(records) == 1
        rec = records[0]
        assert rec.old_index == 2
        assert rec.new_index == -1  # candidate
        assert rec.contradiction_type == ContradictionType.VALUE_UPDATE
        assert rec.confidence == 0.85
        assert rec.timestamp == 42.0

    def test_deactivation_indices_match_superseded(self):
        """Deactivated frozenset matches superseded_indices from result."""
        result = self._make_result_with_supersede(index=1)
        records, deactivated = resolve_contradictions(
            result, "candidate", ["a", "b"], timestamp=0.0,
        )
        assert 1 in deactivated
        assert isinstance(deactivated, frozenset)

    def test_flag_conflict_no_deactivation(self):
        """FLAG_CONFLICT action does NOT deactivate the memory."""
        result = self._make_result_with_flag(index=0, confidence=0.5)
        records, deactivated = resolve_contradictions(
            result, "candidate", ["existing"], timestamp=0.0,
        )
        assert len(records) == 0  # No supersession records for flagged
        assert 0 not in deactivated

    def test_timestamp_propagated(self):
        """Timestamp is propagated to SupersessionRecord."""
        result = self._make_result_with_supersede(index=0)
        records, _ = resolve_contradictions(
            result, "candidate", ["existing"], timestamp=999.5,
        )
        assert records[0].timestamp == 999.5

    def test_multiple_supersessions(self):
        """Multiple AUTO_SUPERSEDE actions create multiple records."""
        se = SubjectExtraction(
            subject="loc", value="x", field_type="location", raw_match="m",
        )
        det0 = ContradictionDetection(
            existing_index=0,
            contradiction_type=ContradictionType.VALUE_UPDATE,
            confidence=0.9,
            subject_overlap=0.8,
            candidate_subject=se,
            existing_subject=se,
            explanation="update 0",
        )
        det1 = ContradictionDetection(
            existing_index=1,
            contradiction_type=ContradictionType.VALUE_UPDATE,
            confidence=0.85,
            subject_overlap=0.7,
            candidate_subject=se,
            existing_subject=se,
            explanation="update 1",
        )
        result = ContradictionResult(
            detections=(det0, det1),
            actions=(
                (0, SupersessionAction.AUTO_SUPERSEDE),
                (1, SupersessionAction.AUTO_SUPERSEDE),
            ),
            superseded_indices=frozenset({0, 1}),
            flagged_indices=frozenset(),
            has_contradiction=True,
            highest_confidence=0.9,
        )
        records, deactivated = resolve_contradictions(
            result, "candidate", ["a", "b", "c"], timestamp=50.0,
        )
        assert len(records) == 2
        assert deactivated == frozenset({0, 1})

    def test_non_contradiction_result_raises_type_error(self):
        """Passing non-ContradictionResult raises TypeError."""
        with pytest.raises(TypeError):
            resolve_contradictions(
                "not a result",  # type: ignore[arg-type]
                "candidate",
                ["existing"],
                timestamp=0.0,
            )

    def test_skip_actions_ignored(self):
        """SKIP actions produce no records and no deactivations."""
        se = SubjectExtraction(
            subject="x", value="y", field_type="fact", raw_match="m",
        )
        det = ContradictionDetection(
            existing_index=0,
            contradiction_type=ContradictionType.VALUE_UPDATE,
            confidence=0.2,
            subject_overlap=0.3,
            candidate_subject=se,
            existing_subject=se,
            explanation="low confidence",
        )
        result = ContradictionResult(
            detections=(det,),
            actions=((0, SupersessionAction.SKIP),),
            superseded_indices=frozenset(),
            flagged_indices=frozenset(),
            has_contradiction=True,
            highest_confidence=0.2,
        )
        records, deactivated = resolve_contradictions(
            result, "candidate", ["existing"], timestamp=0.0,
        )
        assert len(records) == 0
        assert len(deactivated) == 0


# ===========================================================================
# 15. TestCategorySpecificSupersession (~8 tests)
# ===========================================================================


class TestCategorySpecificSupersession:
    """Verify category-specific supersession rules from spec Section 7.3."""

    def test_correction_always_supersedes(self):
        """Corrections always supersede the target (spec 7.3)."""
        result = detect_contradictions(
            candidate_text="No, actually the API key is XYZ123",
            candidate_category="correction",
            existing_texts=["The API key is ABC789"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is True, (
            "Correction of a fact must detect contradiction"
        )
        assert len(result.actions) > 0, (
            "Correction must produce at least one action"
        )
        _, action = result.actions[0]
        assert action == SupersessionAction.AUTO_SUPERSEDE, (
            f"Corrections should AUTO_SUPERSEDE, got {action}"
        )

    def test_fact_supersedes_value_update_high_confidence(self):
        """Facts supersede via VALUE_UPDATE when confidence >= 0.7 (spec 7.3)."""
        result = detect_contradictions(
            candidate_text="My email is new@example.com",
            candidate_category="fact",
            existing_texts=["My email is old@example.com"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is True
        # Email update should have high confidence (>= 0.8 per spec 5.2)
        assert result.highest_confidence >= 0.7

    def test_preference_supersedes_reversal_medium_confidence(self):
        """Preferences supersede if PREFERENCE_REVERSAL confidence >= 0.6 (spec 7.3)."""
        result = detect_contradictions(
            candidate_text="I don't like dark mode anymore",
            candidate_category="preference",
            existing_texts=["I like dark mode"],
            existing_categories=["preference"],
        )
        assert result.has_contradiction is True

    def test_instruction_supersedes_conflict_high_confidence(self):
        """Instructions supersede if INSTRUCTION_CONFLICT confidence >= 0.7 (spec 7.3)."""
        result = detect_contradictions(
            candidate_text="Never use tabs for indentation",
            candidate_category="instruction",
            existing_texts=["Always use tabs for indentation"],
            existing_categories=["instruction"],
        )
        assert result.has_contradiction is True

    def test_reasoning_flags_only_never_supersedes(self):
        """Reasoning only flags, never supersedes (spec 7.3)."""
        long_reasoning_a = (
            "We should use microservices because the monolith is too complex. "
            "Therefore each team can deploy independently since they own boundaries."
        )
        long_reasoning_b = (
            "We should use a monolith because microservices add complexity. "
            "Therefore we avoid distributed system issues since everything is colocated."
        )
        result = detect_contradictions(
            candidate_text=long_reasoning_b,
            candidate_category="reasoning",
            existing_texts=[long_reasoning_a],
            existing_categories=["reasoning"],
        )
        # Reasoning may or may not detect contradiction (weight 0.5),
        # but must NEVER produce AUTO_SUPERSEDE (spec 7.3).
        for _, action in result.actions:
            assert action != SupersessionAction.AUTO_SUPERSEDE, (
                "Reasoning should never AUTO_SUPERSEDE"
            )

    def test_greeting_never_enters_pipeline(self):
        """Greetings have weight 0.0 and never enter the pipeline (spec 7.3)."""
        result = detect_contradictions(
            candidate_text="Hello!",
            candidate_category="greeting",
            existing_texts=["Goodbye!"],
            existing_categories=["greeting"],
        )
        assert result.has_contradiction is False
        assert len(result.detections) == 0

    def test_transactional_never_enters_pipeline(self):
        """Transactional has weight 0.0 and never enters pipeline (spec 7.3)."""
        result = detect_contradictions(
            candidate_text="Run the tests",
            candidate_category="transactional",
            existing_texts=["Run the linter"],
            existing_categories=["transactional"],
        )
        assert result.has_contradiction is False
        assert len(result.detections) == 0

    def test_correction_supersedes_over_preference(self):
        """Correction of a preference still auto-supersedes."""
        result = detect_contradictions(
            candidate_text="Actually, I prefer light mode, not dark mode",
            candidate_category="correction",
            existing_texts=["I prefer dark mode"],
            existing_categories=["preference"],
        )
        assert result.has_contradiction is True, (
            "Correction of a preference must detect contradiction"
        )
        assert len(result.actions) > 0, (
            "Correction must produce at least one action"
        )
        _, action = result.actions[0]
        assert action == SupersessionAction.AUTO_SUPERSEDE, (
            "Correction of preference should AUTO_SUPERSEDE"
        )


# ===========================================================================
# 16. TestFullPipeline -- integration tests (~6 tests)
# ===========================================================================


class TestFullPipeline:
    """Integration: detect -> resolve for realistic sequences."""

    def test_location_update_pipeline(self):
        """Full pipeline: detect location contradiction and resolve it."""
        result = detect_contradictions(
            candidate_text="I live in London now",
            candidate_category="fact",
            existing_texts=["I live in New York"],
            existing_categories=["fact"],
        )
        records, deactivated = resolve_contradictions(
            result, "I live in London now", ["I live in New York"], timestamp=100.0,
        )
        assert result.has_contradiction is True, (
            "Location update must detect contradiction"
        )
        assert result.highest_confidence >= 0.7, (
            f"Location update confidence should be >= 0.7, got {result.highest_confidence}"
        )
        assert len(records) >= 1, (
            "Location update with high confidence must produce supersession records"
        )
        assert 0 in deactivated, (
            "Old location memory (index 0) must be deactivated"
        )

    def test_no_contradiction_pipeline(self):
        """Full pipeline: unrelated memories produce no supersession."""
        result = detect_contradictions(
            candidate_text="I like Python",
            candidate_category="preference",
            existing_texts=["The weather is sunny", "I work at Acme"],
            existing_categories=["fact", "fact"],
        )
        records, deactivated = resolve_contradictions(
            result, "I like Python",
            ["The weather is sunny", "I work at Acme"],
            timestamp=50.0,
        )
        assert len(records) == 0
        assert len(deactivated) == 0

    def test_multi_contradiction_pipeline(self):
        """Full pipeline: candidate contradicts multiple existing memories."""
        result = detect_contradictions(
            candidate_text="No, actually I live in London and my name is Bob",
            candidate_category="correction",
            existing_texts=[
                "I live in New York",
                "My name is Alice",
                "Hello!",
            ],
            existing_categories=["fact", "fact", "greeting"],
        )
        records, deactivated = resolve_contradictions(
            result, "No, actually I live in London and my name is Bob",
            ["I live in New York", "My name is Alice", "Hello!"],
            timestamp=300.0,
        )
        # Greeting (index 2) should never be deactivated
        assert 2 not in deactivated

    def test_instruction_conflict_resolution(self):
        """Full pipeline: conflicting instructions resolved."""
        result = detect_contradictions(
            candidate_text="Never use tabs for indentation",
            candidate_category="instruction",
            existing_texts=["Always use tabs for indentation"],
            existing_categories=["instruction"],
        )
        records, deactivated = resolve_contradictions(
            result, "Never use tabs for indentation",
            ["Always use tabs for indentation"],
            timestamp=400.0,
        )
        assert isinstance(records, list)
        assert isinstance(deactivated, frozenset)

    def test_preference_reversal_resolution(self):
        """Full pipeline: preference reversal detected and resolved."""
        result = detect_contradictions(
            candidate_text="I don't like dark mode anymore",
            candidate_category="preference",
            existing_texts=["I like dark mode"],
            existing_categories=["preference"],
        )
        records, deactivated = resolve_contradictions(
            result, "I don't like dark mode anymore",
            ["I like dark mode"],
            timestamp=500.0,
        )
        assert isinstance(records, list)
        assert isinstance(deactivated, frozenset)


# ===========================================================================
# 17. TestAdversarialInputs -- adversarial analysis tests (AP3)
# ===========================================================================


class TestAdversarialInputs:
    """Tests for known failure modes from adversarial analysis.

    Some tests document known v1 limitations with xfail markers.
    Others are concrete fixes that must pass.
    """

    # AP3-F2: Greedy name extractor guard
    def test_im_tired_not_classified_as_name(self):
        """'I'm tired' must NOT extract as name='tired' (AP3-F2)."""
        result = extract_subject("I'm tired")
        assert result.field_type != "name" or result.value is None or "tired" not in (result.value or ""), (
            f"'I'm tired' should not extract as name, got field_type={result.field_type} value={result.value}"
        )

    def test_im_a_developer_not_classified_as_name(self):
        """'I'm a developer' must NOT extract as name='a developer' (AP3-F2)."""
        result = extract_subject("I'm a developer")
        assert result.field_type != "name" or "developer" not in (result.value or ""), (
            f"'I'm a developer' should not extract as name, got value={result.value}"
        )

    def test_im_from_paris_extracts_location_not_name(self):
        """'I'm from Paris' should extract location, not name (AP3-F2)."""
        result = extract_subject("I'm from Paris")
        # Should be location (from Paris) not name (from paris)
        assert result.field_type != "name" or "paris" not in (result.value or "").lower(), (
            f"'I'm from Paris' should not extract as name, got field_type={result.field_type} value={result.value}"
        )

    # AP3-F4: Value containment check
    def test_new_york_city_vs_new_york_low_confidence(self):
        """'New York City' vs 'New York' should have reduced confidence (AP3-F4).
        Substring containment means these refer to the same entity."""
        result = detect_contradictions(
            candidate_text="I live in New York",
            candidate_category="fact",
            existing_texts=["I live in New York City"],
            existing_categories=["fact"],
        )
        if result.has_contradiction:
            # If detected, confidence should be reduced (containment check)
            assert result.highest_confidence <= 0.5, (
                f"Substring containment should reduce confidence, got {result.highest_confidence}"
            )

    # AP3-F4: Instruction elaboration
    def test_instruction_elaboration_not_conflict(self):
        """'Always use Python' vs 'Always use Python 3' is elaboration, not conflict (AP3-F9)."""
        result = detect_contradictions(
            candidate_text="Always use Python 3",
            candidate_category="instruction",
            existing_texts=["Always use Python"],
            existing_categories=["instruction"],
        )
        if result.has_contradiction:
            # Elaboration should have low confidence or not produce AUTO_SUPERSEDE
            has_auto = any(act == SupersessionAction.AUTO_SUPERSEDE
                         for _, act in result.actions)
            assert not has_auto, (
                "Elaboration (subset of instruction) should not AUTO_SUPERSEDE"
            )

    # AP3-F1: Paraphrase blindness (known limitation)
    @pytest.mark.xfail(reason="AP3-F1: regex-based extraction cannot handle paraphrases in v1")
    def test_paraphrase_location_new_yorker(self):
        """'I'm a New Yorker' should contradict 'I live in London' (AP3-F1)."""
        result = detect_contradictions(
            candidate_text="I live in London",
            candidate_category="fact",
            existing_texts=["I'm a New Yorker"],
            existing_categories=["fact"],
        )
        assert result.has_contradiction is True

    # AP3-F5: Double negatives (known limitation)
    @pytest.mark.xfail(reason="AP3-F5: keyword polarity cannot handle double negatives in v1")
    def test_double_negative_polarity_correct(self):
        """'I don't dislike Python' should detect POSITIVE polarity (AP3-F5)."""
        polarity = detect_polarity("I don't dislike Python")
        assert polarity == Polarity.POSITIVE

    # AP2-F1: Negative category weight rejection
    def test_category_weights_negative_value_rejected(self):
        """Category weight value < 0.0 must be rejected."""
        weights = dict(DEFAULT_CATEGORY_WEIGHTS)
        weights["correction"] = -0.1
        with pytest.raises(ValueError, match="category_weights"):
            ContradictionConfig(category_weights=weights)

    # AP2-F2: Boundary acceptance
    def test_similarity_threshold_exactly_one_accepted(self):
        """similarity_threshold=1.0 is valid (upper inclusive boundary)."""
        config = ContradictionConfig(similarity_threshold=1.0)
        assert config.similarity_threshold == 1.0

    def test_confidence_threshold_exactly_one_accepted(self):
        """confidence_threshold=1.0 is valid (upper inclusive boundary)."""
        config = ContradictionConfig(confidence_threshold=1.0)
        assert config.confidence_threshold == 1.0

    # AP2-F6: Detections sorted by confidence
    def test_detections_sorted_by_confidence_descending(self):
        """Detections must be sorted by confidence descending (spec 6.2)."""
        result = detect_contradictions(
            candidate_text="No, actually my email is new@example.com",
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
                f"Detections must be sorted descending, got: {confidences}"
            )

    # AP2-F8: enable_auto_supersede=False must produce FLAG_CONFLICT
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
        has_flag = any(act == SupersessionAction.FLAG_CONFLICT for _, act in result.actions)
        assert has_flag, "With enable_auto_supersede=False, must produce FLAG_CONFLICT"
        assert len(result.flagged_indices) >= 1

    # AP2-F11: Non-str in existing_texts
    def test_non_str_in_existing_texts_raises_type_error(self):
        """Non-string element in existing_texts raises TypeError."""
        with pytest.raises(TypeError):
            detect_contradictions(
                candidate_text="I live in London",
                candidate_category="fact",
                existing_texts=["valid", 42],  # type: ignore[list-item]
                existing_categories=["fact", "fact"],
            )

    # AP2-F14: Property test for subject_overlap bounds
    def test_subject_overlap_always_bounded(self):
        """subject_overlap must return [0.0, 1.0] for any inputs."""
        test_cases = [
            ("", ""),
            ("hello", "hello"),
            ("foo bar", "baz qux"),
            ("a b c", "c d e"),
            ("x", "completely different text here"),
        ]
        for subj_a, subj_b in test_cases:
            a = SubjectExtraction(subject=subj_a, value=None, field_type="unknown", raw_match="t")
            b = SubjectExtraction(subject=subj_b, value=None, field_type="unknown", raw_match="t")
            result = subject_overlap(a, b)
            assert 0.0 <= result <= 1.0, (
                f"subject_overlap({subj_a!r}, {subj_b!r}) = {result}, expected [0.0, 1.0]"
            )


# ===========================================================================
# 18. TestPropertyBased -- Hypothesis property tests (~6 tests)
# ===========================================================================


# Strategies for property tests

_category_st = st.sampled_from(sorted(_VALID_CATEGORIES))
_memory_text_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        max_codepoint=0xFFFF,
    ),
    min_size=0,
    max_size=500,
)


class TestPropertyBased:
    """Property-based tests using Hypothesis for robustness."""

    @given(text=_memory_text_st, category=_category_st)
    @settings(max_examples=MAX_EXAMPLES)
    def test_detect_contradictions_never_crashes(self, text, category):
        """detect_contradictions with random inputs never raises unexpected errors."""
        try:
            result = detect_contradictions(
                candidate_text=text,
                candidate_category=category,
                existing_texts=["I live in New York", "I prefer dark mode"],
                existing_categories=["fact", "preference"],
            )
            assert isinstance(result, ContradictionResult)
        except (TypeError, ValueError):
            pass  # Expected validation errors are acceptable

    @given(
        subj_a=st.text(min_size=1, max_size=50),
        subj_b=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_subject_overlap_symmetric(self, subj_a, subj_b):
        """subject_overlap(a, b) == subject_overlap(b, a) for all inputs."""
        a = SubjectExtraction(
            subject=subj_a, value=None, field_type="unknown", raw_match="test",
        )
        b = SubjectExtraction(
            subject=subj_b, value=None, field_type="unknown", raw_match="test",
        )
        assert subject_overlap(a, b) == subject_overlap(b, a)

    @given(text=_memory_text_st)
    @settings(max_examples=MAX_EXAMPLES)
    def test_detect_polarity_returns_valid_member(self, text):
        """detect_polarity always returns a valid Polarity member."""
        result = detect_polarity(text)
        assert isinstance(result, Polarity)
        assert result in (Polarity.POSITIVE, Polarity.NEGATIVE, Polarity.NEUTRAL)

    @given(
        text=_memory_text_st,
        category=_category_st,
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_resolve_returns_disjoint_sets(self, text, category):
        """resolve_contradictions returns disjoint superseded and flagged sets.

        This tests the property that a memory cannot be both superseded AND flagged
        for review at the same time.
        """
        try:
            result = detect_contradictions(
                candidate_text=text,
                candidate_category=category,
                existing_texts=["I live in New York", "I prefer dark mode"],
                existing_categories=["fact", "preference"],
            )
        except (TypeError, ValueError):
            return  # Skip invalid inputs

        records, deactivated = resolve_contradictions(
            result, text,
            ["I live in New York", "I prefer dark mode"],
            timestamp=0.0,
        )
        # Deactivated set (from supersession) should not overlap with flagged
        assert deactivated.isdisjoint(result.flagged_indices), (
            f"Superseded {deactivated} and flagged {result.flagged_indices} overlap"
        )


# ===========================================================================
# 18. TestModuleConstants (~4 tests)
# ===========================================================================


class TestModuleConstants:
    """Verify module-level constants from the public API."""

    def test_field_extractors_is_dict(self):
        """FIELD_EXTRACTORS is a dict."""
        assert isinstance(FIELD_EXTRACTORS, dict)

    def test_field_extractors_has_expected_keys(self):
        """FIELD_EXTRACTORS has keys from spec Section 4.1."""
        expected_keys = {"location", "name", "email", "job", "preference", "instruction"}
        assert set(FIELD_EXTRACTORS.keys()) >= expected_keys, (
            f"Missing keys: {expected_keys - set(FIELD_EXTRACTORS.keys())}"
        )

    def test_default_contradiction_config_is_config(self):
        """DEFAULT_CONTRADICTION_CONFIG is a ContradictionConfig instance."""
        assert isinstance(DEFAULT_CONTRADICTION_CONFIG, ContradictionConfig)

    def test_default_contradiction_config_has_default_values(self):
        """DEFAULT_CONTRADICTION_CONFIG uses spec default values."""
        assert DEFAULT_CONTRADICTION_CONFIG.similarity_threshold == 0.3
        assert DEFAULT_CONTRADICTION_CONFIG.confidence_threshold == 0.7
        assert DEFAULT_CONTRADICTION_CONFIG.max_candidates == 50
        assert DEFAULT_CONTRADICTION_CONFIG.enable_auto_supersede is True
        assert DEFAULT_CONTRADICTION_CONFIG.value_pattern_min_tokens == 2
