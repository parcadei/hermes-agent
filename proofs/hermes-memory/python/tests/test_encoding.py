"""Tests for the encoding gate: classification, write policy, importance mapping.

Tests are written BEFORE implementation exists. All imports from
hermes_memory.encoding will fail with ImportError until the module is created.

Test categories:
  1. Category Classification (~15 tests)
  2. Write Policy (~10 tests)
  3. Initial Importance (~7 tests)
  4. Confidence (~5 tests)
  5. Edge Cases (~8 tests)
  6. Property-Based / Hypothesis (~5 tests)
  7. Configuration (~5 tests)
  8. Integration (~3 tests)
"""

import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hermes_memory.encoding import (
    EncodingConfig,
    EncodingPolicy,
    CATEGORY_IMPORTANCE,
    VALID_CATEGORIES,
    _detect_non_english,
)

# Re-use project strategies where applicable

MAX_EXAMPLES = 200


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def policy():
    """Default encoding policy with standard config."""
    return EncodingPolicy()


@pytest.fixture()
def config():
    """Default encoding config."""
    return EncodingConfig()


# ============================================================
# 1. Category Classification Tests (~15 tests)
# ============================================================


class TestCategoryClassification:
    """Verify that clear-cut examples are classified into the correct category."""

    def test_classifies_preference_i_like(self, policy):
        """'I like dark themes for my IDE' -> preference."""
        decision = policy.evaluate("I like dark themes for my IDE")
        assert decision.category == "preference", (
            f"Expected 'preference', got '{decision.category}'"
        )

    def test_classifies_preference_i_prefer(self, policy):
        """'I prefer Python over JavaScript' -> preference."""
        decision = policy.evaluate("I prefer Python over JavaScript")
        assert decision.category == "preference", (
            f"Expected 'preference', got '{decision.category}'"
        )

    def test_classifies_preference_negative(self, policy):
        """'I don't like tabs, always use spaces' -> instruction wins over preference
        because 'always' is an instruction pattern with higher priority."""
        decision = policy.evaluate("I don't like tabs, always use spaces")
        # 'always' matches instruction (priority 2), 'I don't like' matches preference (priority 3)
        # instruction > preference in priority
        assert decision.category == "instruction", (
            f"Expected 'instruction' (higher priority than preference), got '{decision.category}'"
        )

    def test_classifies_fact_personal(self, policy):
        """'I am a software engineer at Google' -> fact."""
        decision = policy.evaluate("I am a software engineer at Google")
        assert decision.category == "fact", (
            f"Expected 'fact', got '{decision.category}'"
        )

    def test_classifies_fact_location(self, policy):
        """'I live in San Francisco' -> fact."""
        decision = policy.evaluate("I live in San Francisco")
        assert decision.category == "fact", (
            f"Expected 'fact', got '{decision.category}'"
        )

    def test_classifies_correction_no_actually(self, policy):
        """'No, actually the deadline is March 15th, not March 10th' -> correction."""
        decision = policy.evaluate(
            "No, actually the deadline is March 15th, not March 10th"
        )
        assert decision.category == "correction", (
            f"Expected 'correction', got '{decision.category}'"
        )

    def test_classifies_correction_thats_wrong(self, policy):
        """'That's wrong, the API endpoint is /v2/users' -> correction."""
        decision = policy.evaluate("That's wrong, the API endpoint is /v2/users")
        assert decision.category == "correction", (
            f"Expected 'correction', got '{decision.category}'"
        )

    # -- Issue 2: "no" false-positive regression tests --
    # Bare "no" with word boundaries matches any "no" in word-boundary context.
    # After replacing "no," and "no " with compound correction-intent patterns,
    # these casual uses of "no" must NOT be classified as corrections.

    def test_correction_no_problem_not_correction(self, policy):
        """'no problem, happy to help' must NOT be classified as correction."""
        decision = policy.evaluate("no problem, happy to help")
        assert decision.category != "correction", (
            f"Expected NOT 'correction', got '{decision.category}' — "
            f"bare 'no' should not trigger correction"
        )

    def test_correction_no_thanks_not_correction(self, policy):
        """'no thanks, I am good' must NOT be classified as correction."""
        decision = policy.evaluate("no thanks, I am good")
        assert decision.category != "correction", (
            f"Expected NOT 'correction', got '{decision.category}' — "
            f"bare 'no' should not trigger correction"
        )

    def test_correction_no_idea_not_correction(self, policy):
        """'I have no idea what you mean' must NOT be classified as correction."""
        decision = policy.evaluate("I have no idea what you mean")
        assert decision.category != "correction", (
            f"Expected NOT 'correction', got '{decision.category}' — "
            f"bare 'no' should not trigger correction"
        )

    def test_correction_no_way_not_correction(self, policy):
        """'no way that is possible' must NOT be classified as correction."""
        decision = policy.evaluate("no way that is possible")
        assert decision.category != "correction", (
            f"Expected NOT 'correction', got '{decision.category}' — "
            f"bare 'no' should not trigger correction"
        )

    # -- Issue 2: compound correction-intent positive tests --
    # These DO contain genuine correction intent (no + comma/period + context).

    def test_correction_no_comma_i(self, policy):
        """'No, I prefer dark mode over light mode' -> correction."""
        decision = policy.evaluate("No, I prefer dark mode over light mode")
        assert decision.category == "correction", (
            f"Expected 'correction', got '{decision.category}'"
        )

    def test_correction_no_actually(self, policy):
        """'No, actually the deadline is Friday not Thursday' -> correction."""
        decision = policy.evaluate(
            "No, actually the deadline is Friday not Thursday"
        )
        assert decision.category == "correction", (
            f"Expected 'correction', got '{decision.category}'"
        )

    def test_correction_no_that(self, policy):
        """'No, that is completely wrong' -> correction."""
        decision = policy.evaluate("No, that is completely wrong")
        assert decision.category == "correction", (
            f"Expected 'correction', got '{decision.category}'"
        )

    def test_correction_no_it(self, policy):
        """'No, it should be version 3, not version 2' -> correction."""
        decision = policy.evaluate("No, it should be version 3, not version 2")
        assert decision.category == "correction", (
            f"Expected 'correction', got '{decision.category}'"
        )

    def test_correction_no_what_i(self, policy):
        """'No, what I meant was the blue button' -> correction."""
        decision = policy.evaluate("No, what I meant was the blue button")
        assert decision.category == "correction", (
            f"Expected 'correction', got '{decision.category}'"
        )

    def test_correction_no_period_that(self, policy):
        """'No. That is incorrect, the answer is 42' -> correction."""
        decision = policy.evaluate("No. That is incorrect, the answer is 42")
        assert decision.category == "correction", (
            f"Expected 'correction', got '{decision.category}'"
        )

    def test_correction_no_period_the(self, policy):
        """'No. The correct answer is 42, not 43' -> correction."""
        decision = policy.evaluate("No. The correct answer is 42, not 43")
        assert decision.category == "correction", (
            f"Expected 'correction', got '{decision.category}'"
        )

    def test_correction_no_period_it(self, policy):
        """'No. It was supposed to be blue, not red' -> correction."""
        decision = policy.evaluate("No. It was supposed to be blue, not red")
        assert decision.category == "correction", (
            f"Expected 'correction', got '{decision.category}'"
        )

    def test_classifies_instruction_always(self, policy):
        """'Always use type hints in Python code' -> instruction."""
        decision = policy.evaluate("Always use type hints in Python code")
        assert decision.category == "instruction", (
            f"Expected 'instruction', got '{decision.category}'"
        )

    def test_classifies_instruction_remember(self, policy):
        """'Remember to run tests before committing' -> instruction."""
        decision = policy.evaluate("Remember to run tests before committing")
        assert decision.category == "instruction", (
            f"Expected 'instruction', got '{decision.category}'"
        )

    def test_classifies_reasoning_with_because(self, policy):
        """Long paragraph with causal connectives -> reasoning.

        Text must be >100 chars and contain >=2 connectives to be classified
        as reasoning that passes the write policy, but the *category* is
        assigned purely by pattern match.
        """
        text = (
            "The system should use a write-ahead log because it provides durability "
            "guarantees. Therefore, even if the process crashes, we can recover the "
            "state from the log since the entries are flushed to disk before "
            "acknowledging the write."
        )
        assert len(text) > 100, "Test text must exceed reasoning length threshold"
        decision = policy.evaluate(text)
        assert decision.category == "reasoning", (
            f"Expected 'reasoning', got '{decision.category}'"
        )

    def test_classifies_greeting_hello(self, policy):
        """'Hello!' -> greeting."""
        decision = policy.evaluate("Hello!")
        assert decision.category == "greeting", (
            f"Expected 'greeting', got '{decision.category}'"
        )

    def test_classifies_greeting_thanks(self, policy):
        """'Thanks, bye!' -> greeting."""
        decision = policy.evaluate("Thanks, bye!")
        assert decision.category == "greeting", (
            f"Expected 'greeting', got '{decision.category}'"
        )

    def test_classifies_transactional_run(self, policy):
        """'Run the test suite' -> transactional."""
        decision = policy.evaluate("Run the test suite")
        assert decision.category == "transactional", (
            f"Expected 'transactional', got '{decision.category}'"
        )

    def test_classifies_transactional_open(self, policy):
        """'Open the config file' -> transactional."""
        decision = policy.evaluate("Open the config file")
        assert decision.category == "transactional", (
            f"Expected 'transactional', got '{decision.category}'"
        )

    def test_classifies_transactional_show(self, policy):
        """'Show me the logs' -> transactional."""
        decision = policy.evaluate("Show me the logs")
        assert decision.category == "transactional", (
            f"Expected 'transactional', got '{decision.category}'"
        )


# ============================================================
# 2. Write Policy Tests (~10 tests)
# ============================================================


class TestWritePolicy:
    """Verify that should_store matches the decision table from spec Section 4.1."""

    def test_preference_always_stored(self, policy):
        """Preferences have should_store=True."""
        decision = policy.evaluate("I prefer dark mode for my editor")
        assert decision.category == "preference"
        assert decision.should_store is True, "Preferences must always be stored"

    def test_fact_always_stored(self, policy):
        """Facts have should_store=True."""
        decision = policy.evaluate("I am a backend developer")
        assert decision.category == "fact"
        assert decision.should_store is True, "Facts must always be stored"

    def test_correction_always_stored(self, policy):
        """Corrections have should_store=True."""
        decision = policy.evaluate("No, actually the port number is 8080, not 3000")
        assert decision.category == "correction"
        assert decision.should_store is True, "Corrections must always be stored"

    def test_instruction_always_stored(self, policy):
        """Instructions have should_store=True."""
        decision = policy.evaluate("Always run lint before pushing code")
        assert decision.category == "instruction"
        assert decision.should_store is True, "Instructions must always be stored"

    def test_greeting_never_stored(self, policy):
        """Greetings have should_store=False (when confidence is above threshold)."""
        # "Hello!" is short, matches greeting clearly.
        # Confidence = 1/15 = 0.067, which is < 0.5, so fail-open would trigger.
        # But per spec, empty/very-short greetings with high match confidence...
        # We test a greeting that has higher confidence by matching multiple patterns.
        # Actually, per the spec the fail-open check happens AFTER the write policy.
        # A greeting with confidence >= threshold should NOT be stored.
        # We use a custom config with a very low threshold to avoid fail-open.
        low_threshold_policy = EncodingPolicy(EncodingConfig(confidence_threshold=0.0))
        decision = low_threshold_policy.evaluate("Hello!")
        assert decision.category == "greeting"
        assert decision.should_store is False, (
            "Greetings must not be stored when confidence >= threshold"
        )

    def test_transactional_never_stored(self, policy):
        """Transactional commands have should_store=False (when confidence >= threshold)."""
        low_threshold_policy = EncodingPolicy(EncodingConfig(confidence_threshold=0.0))
        decision = low_threshold_policy.evaluate("Run the test suite")
        assert decision.category == "transactional"
        assert decision.should_store is False, (
            "Transactional commands must not be stored when confidence >= threshold"
        )

    def test_reasoning_stored_when_long_enough(self, policy):
        """Reasoning above length and connective thresholds is stored."""
        text = (
            "We should adopt a microservices architecture because the monolith has "
            "become too difficult to deploy independently. Therefore each team can "
            "ship at their own cadence since they own their service boundaries. "
            "This implies we also need an API gateway to manage cross-service calls."
        )
        assert len(text) >= 100, "Text must meet min_reasoning_length"
        decision = policy.evaluate(text)
        assert decision.category == "reasoning"
        assert decision.should_store is True, (
            "Reasoning above thresholds must be stored"
        )

    def test_reasoning_rejected_when_short(self, policy):
        """Reasoning below length threshold is not stored (before fail-open).

        Note: with default config, low confidence may trigger fail-open.
        We use confidence_threshold=0.0 to isolate the write policy check.
        """
        strict_policy = EncodingPolicy(EncodingConfig(confidence_threshold=0.0))
        decision = strict_policy.evaluate("Do X because Y")
        # This is short (<100 chars) and has only 1 connective (<2)
        assert decision.category == "reasoning"
        assert decision.should_store is False, (
            "Short reasoning below thresholds must not be stored"
        )

    def test_reasoning_rejected_insufficient_connectives(self, policy):
        """Reasoning with only 1 connective is not stored (before fail-open).

        Text is long enough (>100 chars) but has only 1 causal connective.
        """
        strict_policy = EncodingPolicy(EncodingConfig(confidence_threshold=0.0))
        # Long text but only one connective ("because")
        text = (
            "The deployment pipeline is currently configured to run on every push "
            "to the main branch, which triggers a full rebuild of the Docker images "
            "and runs the entire integration test suite, all because the CI config "
            "was written that way initially."
        )
        assert len(text) >= 100
        # Count connectives: only "because" appears once
        decision = strict_policy.evaluate(text)
        assert decision.category == "reasoning"
        assert decision.should_store is False, (
            "Reasoning with insufficient connectives must not be stored"
        )

    def test_fail_open_low_confidence(self, policy):
        """Ambiguous text with low confidence gets stored via fail-open."""
        # Non-English text: no patterns match, default confidence is low
        decision = policy.evaluate("Bonjour, je suis un developpeur logiciel")
        assert decision.confidence < 0.5, (
            f"Expected low confidence for non-pattern text, got {decision.confidence}"
        )
        assert decision.should_store is True, (
            "Low-confidence classifications must trigger fail-open (should_store=True)"
        )


# ============================================================
# 3. Initial Importance Tests (~7 tests)
# ============================================================


class TestInitialImportance:
    """Verify initial_importance maps correctly from category."""

    def test_importance_correction_highest(self, policy):
        """Corrections get importance 0.9."""
        decision = policy.evaluate("No, actually it should be version 3.0")
        assert decision.category == "correction"
        assert decision.initial_importance == 0.9

    def test_importance_instruction_high(self, policy):
        """Instructions get importance 0.85."""
        decision = policy.evaluate("Always format code with black before committing")
        assert decision.category == "instruction"
        assert decision.initial_importance == 0.85

    def test_importance_preference_high(self, policy):
        """Preferences get importance 0.8."""
        decision = policy.evaluate("I prefer tabs over spaces")
        assert decision.category == "preference"
        assert decision.initial_importance == 0.8

    def test_importance_fact_medium(self, policy):
        """Facts get importance 0.6."""
        decision = policy.evaluate("I am a data scientist")
        assert decision.category == "fact"
        assert decision.initial_importance == 0.6

    def test_importance_reasoning_low(self, policy):
        """Reasoning gets importance 0.4."""
        text = (
            "We need to refactor the authentication module because the current "
            "implementation has grown too complex. Therefore, splitting it into "
            "smaller components will make it more maintainable since each piece "
            "can be tested independently."
        )
        decision = policy.evaluate(text)
        assert decision.category == "reasoning"
        assert decision.initial_importance == 0.4

    def test_importance_greeting_zero(self, policy):
        """Greetings get importance 0.0."""
        decision = policy.evaluate("Hello!")
        assert decision.category == "greeting"
        assert decision.initial_importance == 0.0

    def test_importance_transactional_zero(self, policy):
        """Transactional commands get importance 0.0."""
        decision = policy.evaluate("Run the tests")
        assert decision.category == "transactional"
        assert decision.initial_importance == 0.0


# ============================================================
# 4. Confidence Tests (~5 tests)
# ============================================================


class TestConfidence:
    """Verify confidence scoring behavior."""

    def test_confidence_clear_preference_high(self, policy):
        """Clear 'I prefer X' gets confidence > 1/total_patterns.

        With 14 preference patterns, matching 1 gives confidence = 1/14 ~ 0.07.
        Matching 'I prefer' is a single pattern match so confidence ~ 0.07.
        But if we match multiple preference patterns, confidence should be higher.
        """
        # Matches "I prefer" and "I like" and "I love" = 3/14 ~ 0.21
        decision = policy.evaluate("I prefer, like, and love using Vim")
        assert decision.confidence > 0.07, (
            f"Multiple-pattern match should yield confidence > single-pattern, "
            f"got {decision.confidence}"
        )

    def test_confidence_ambiguous_text_lower(self, policy):
        """Text with no clear indicators gets confidence < 0.5."""
        decision = policy.evaluate("The weather is nice today")
        assert decision.confidence < 0.5, (
            f"Ambiguous text should have low confidence, got {decision.confidence}"
        )

    def test_confidence_always_in_range(self, policy):
        """Confidence is always in [0.0, 1.0]."""
        test_cases = [
            "Hello!",
            "I prefer Python",
            "I am an engineer",
            "No, that's wrong",
            "Always use type hints",
            "Run the tests",
            "",
            "a" * 10000,
            "Because therefore since thus hence consequently as a result",
        ]
        for text in test_cases:
            decision = policy.evaluate(text)
            assert 0.0 <= decision.confidence <= 1.0, (
                f"Confidence {decision.confidence} out of range for: {text[:50]!r}"
            )

    def test_confidence_empty_string_high(self, policy):
        """Empty string is classified as greeting with confidence 1.0.

        Per spec Section 6.1, empty input gets confidence=1.0 because
        the classifier is certain this is vacuous content.
        """
        decision = policy.evaluate("")
        assert decision.category == "greeting"
        assert decision.confidence == 1.0, (
            f"Empty string should have confidence 1.0, got {decision.confidence}"
        )

    def test_confidence_multiple_indicators_boosted(self, policy):
        """Text matching multiple patterns of same category gets higher confidence."""
        # Single preference pattern
        single = policy.evaluate("I prefer dark mode")
        # Multiple preference patterns
        multi = policy.evaluate("I prefer dark mode. I like it. I enjoy the contrast.")
        assert multi.confidence > single.confidence, (
            f"Multiple indicators ({multi.confidence}) should yield higher confidence "
            f"than single ({single.confidence})"
        )


# ============================================================
# 5. Edge Case Tests (~8 tests)
# ============================================================


class TestEdgeCases:
    """Edge cases from spec Section 6 and additional boundary conditions."""

    def test_empty_string_is_greeting(self, policy):
        """'' -> greeting, should_store=False, confidence=1.0."""
        decision = policy.evaluate("")
        assert decision.should_store is False
        assert decision.category == "greeting"
        assert decision.confidence == 1.0
        assert decision.initial_importance == 0.0

    def test_none_content_raises_or_handles(self, policy):
        """None input is handled gracefully (treated as empty string per spec 6.2)."""
        decision = policy.evaluate(None)
        assert decision.should_store is False
        assert decision.category == "greeting"
        assert decision.confidence == 1.0
        assert decision.initial_importance == 0.0

    def test_very_long_unclassifiable_stored(self, policy):
        """10000+ char text with no indicators -> fail-open, stored.

        Per spec Section 6.3, long unclassifiable text defaults to 'fact'
        with very low confidence, triggering fail-open.
        """
        # Generate text that matches no patterns
        text = "Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 200
        assert len(text) > 10000
        decision = policy.evaluate(text)
        assert decision.should_store is True, (
            "Very long unclassifiable text must be stored via fail-open"
        )
        # Long unclassifiable text defaults to 'fact' per spec Section 5.3
        assert decision.category == "fact", (
            f"Long unclassifiable text should default to 'fact', got '{decision.category}'"
        )

    def test_mixed_preference_and_fact(self, policy):
        """'I'm a developer and I prefer Rust' -> preference wins over fact.

        Both fact ('I'm') and preference ('I prefer') match.
        Priority: preference (3) > fact (4), so preference wins.
        """
        decision = policy.evaluate("I'm a developer and I prefer Rust")
        assert decision.category == "preference", (
            f"Preference has higher priority than fact, got '{decision.category}'"
        )

    def test_non_english_text_stored(self, policy):
        """Non-English text -> low confidence -> fail-open -> stored."""
        decision = policy.evaluate("Je prefere le mode sombre. Mon nom est Pierre.")
        assert decision.should_store is True, (
            "Non-English text must be stored via fail-open"
        )
        assert decision.confidence < 0.5, (
            f"Non-English text should have low confidence, got {decision.confidence}"
        )

    def test_greeting_long_reclassified(self, policy):
        """Long text starting with 'Hello' gets reclassified.

        Per spec Section 6.7: 'Hello! I wanted to tell you that I just moved
        to a new apartment...' exceeds max_greeting_length (50) and gets
        reclassified to 'fact' because it contains 'I'm'/'I just moved' patterns.
        """
        text = (
            "Hello! I wanted to tell you that I just moved to a new apartment "
            "in downtown Portland and I'm really excited about it."
        )
        assert len(text) > 50, "Text must exceed max_greeting_length"
        decision = policy.evaluate(text)
        assert decision.category != "greeting", (
            f"Long greeting-like text should be reclassified, got '{decision.category}'"
        )
        assert decision.should_store is True, "Reclassified greeting should be stored"

    def test_transactional_long_reclassified(self, policy):
        """Long transactional text with embedded preference gets reclassified.

        'Run this analysis but also remember I prefer...' exceeds
        max_transactional_length (80) and should be reclassified.
        """
        text = (
            "Run this analysis but also remember that I prefer using pytest "
            "over unittest for all my testing needs going forward"
        )
        assert len(text) > 80, "Text must exceed max_transactional_length"
        decision = policy.evaluate(text)
        # After reclassification without transactional, instruction or preference
        # should win due to priority
        assert decision.category != "transactional", (
            f"Long transactional text should be reclassified, got '{decision.category}'"
        )
        assert decision.should_store is True, (
            "Reclassified transactional text should be stored"
        )

    def test_whitespace_only_is_greeting(self, policy):
        """'   \\n\\t  ' -> greeting, should_store=False (treated as empty)."""
        decision = policy.evaluate("   \n\t  ")
        assert decision.should_store is False
        assert decision.category == "greeting"
        assert decision.confidence == 1.0
        assert decision.initial_importance == 0.0


# ============================================================
# 6. Property-Based Tests (Hypothesis) (~5 tests)
# ============================================================


class TestPropertyBased:
    """Property-based tests using Hypothesis, following the project's
    existing pattern with @given decorators."""

    @given(text=st.text(min_size=0, max_size=5000))
    @settings(max_examples=MAX_EXAMPLES)
    def test_confidence_always_bounded(self, text):
        """For any text, confidence is in [0.0, 1.0]."""
        policy = EncodingPolicy()
        decision = policy.evaluate(text)
        assert 0.0 <= decision.confidence <= 1.0, (
            f"Confidence {decision.confidence} out of bounds for {text[:80]!r}"
        )

    @given(text=st.text(min_size=0, max_size=5000))
    @settings(max_examples=MAX_EXAMPLES)
    def test_importance_always_bounded(self, text):
        """For any text, initial_importance is in [0.0, 1.0]."""
        policy = EncodingPolicy()
        decision = policy.evaluate(text)
        assert 0.0 <= decision.initial_importance <= 1.0, (
            f"Importance {decision.initial_importance} out of bounds for {text[:80]!r}"
        )

    @given(text=st.text(min_size=0, max_size=5000))
    @settings(max_examples=MAX_EXAMPLES)
    def test_category_always_valid(self, text):
        """Category is always one of 8 valid values."""
        policy = EncodingPolicy()
        decision = policy.evaluate(text)
        assert decision.category in VALID_CATEGORIES, (
            f"Category '{decision.category}' not in {VALID_CATEGORIES}"
        )

    @given(text=st.text(min_size=0, max_size=5000))
    @settings(max_examples=MAX_EXAMPLES)
    def test_deterministic_output(self, text):
        """Same input always produces same output."""
        policy = EncodingPolicy()
        d1 = policy.evaluate(text)
        d2 = policy.evaluate(text)
        assert d1.should_store == d2.should_store, (
            f"Determinism violated: should_store {d1.should_store} != {d2.should_store}"
        )
        assert d1.category == d2.category, (
            f"Determinism violated: category '{d1.category}' != '{d2.category}'"
        )
        assert d1.confidence == d2.confidence, (
            f"Determinism violated: confidence {d1.confidence} != {d2.confidence}"
        )
        assert d1.initial_importance == d2.initial_importance, (
            f"Determinism violated: importance {d1.initial_importance} != {d2.initial_importance}"
        )

    @given(text=st.text(min_size=0, max_size=5000))
    @settings(max_examples=MAX_EXAMPLES)
    def test_should_store_consistent_with_category(self, text):
        """If category is ALWAYS-store, should_store must be True.

        Property 8.3 and 8.7 from the spec:
        - preference, fact, correction, instruction -> always stored
        - greeting, transactional with confidence >= threshold -> never stored
        """
        policy = EncodingPolicy()
        config = EncodingConfig()
        decision = policy.evaluate(text)

        always_store = {"preference", "fact", "correction", "instruction", "unclassified"}
        never_store = {"greeting", "transactional"}

        if decision.category in always_store:
            assert decision.should_store is True, (
                f"ALWAYS-store category '{decision.category}' has should_store=False "
                f"for text: {text[:80]!r}"
            )
        if (
            decision.category in never_store
            and decision.confidence >= config.confidence_threshold
        ):
            assert decision.should_store is False, (
                f"NEVER-store category '{decision.category}' with confidence "
                f"{decision.confidence} >= threshold {config.confidence_threshold} "
                f"has should_store=True for text: {text[:80]!r}"
            )


# ============================================================
# 7. Configuration Tests (~5 tests)
# ============================================================


class TestConfiguration:
    """Verify EncodingConfig behavior and customization."""

    def test_custom_reasoning_length_threshold(self):
        """Custom config with min_reasoning_length=200 rejects shorter reasoning."""
        config = EncodingConfig(
            min_reasoning_length=200,
            confidence_threshold=0.0,  # disable fail-open to isolate write policy
        )
        policy = EncodingPolicy(config)
        # Text has connectives and is >100 but <200 chars
        text = (
            "We should use caching because the database queries are slow. "
            "Therefore adding Redis will reduce latency significantly."
        )
        assert 100 < len(text) < 200, f"Text length {len(text)} not in (100, 200)"
        decision = policy.evaluate(text)
        assert decision.category == "reasoning"
        assert decision.should_store is False, (
            "Reasoning below custom min_reasoning_length=200 should not be stored"
        )

    def test_custom_greeting_length_threshold(self):
        """Custom config with max_greeting_length=100 keeps more greetings."""
        config = EncodingConfig(
            max_greeting_length=100,
            confidence_threshold=0.0,  # disable fail-open
        )
        policy = EncodingPolicy(config)
        # 60 chars, would be reclassified at default max_greeting_length=50
        text = "Hello there! I hope you are having a great day today."
        assert 50 < len(text) < 100, f"Text length {len(text)} not in (50, 100)"
        decision = policy.evaluate(text)
        assert decision.category == "greeting", (
            f"Text under custom max_greeting_length=100 should remain 'greeting', "
            f"got '{decision.category}'"
        )

    def test_custom_confidence_threshold(self):
        """Custom config with confidence_threshold=0.4 changes fail-open boundary.

        Post-adversarial review: calibrated confidence floors single-pattern
        matches at 0.35 (SINGLE_PATTERN_CONFIDENCE_FLOOR). A threshold of 0.4
        is above 0.35, so a single greeting match triggers fail-open.
        """
        config = EncodingConfig(confidence_threshold=0.4)
        policy = EncodingPolicy(config)
        # "Hello" matches 1/15 greeting patterns -> calibrated confidence 0.35
        # 0.35 < 0.4 threshold -> fail-open should trigger
        decision = policy.evaluate("Hello!")
        assert decision.should_store is True, (
            "Low-confidence greeting should be stored via fail-open at threshold=0.4"
        )

    def test_default_config_works(self):
        """EncodingPolicy() with no args uses sensible defaults."""
        policy = EncodingPolicy()
        decision = policy.evaluate("I prefer using Vim")
        assert decision.should_store is True
        assert decision.category == "preference"

    def test_config_is_frozen(self):
        """EncodingConfig is frozen, attributes cannot be changed after creation."""
        config = EncodingConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.min_reasoning_length = 500


# ============================================================
# 8. Integration Tests (~3 tests)
# ============================================================


class TestIntegration:
    """Integration with existing hermes_memory types."""

    def test_importance_feeds_dynamics(self, policy):
        """initial_importance can be passed to importance_update() from core.py.

        The encoding gate's initial_importance seeds the dynamics system.
        Verify the value is compatible with importance_update(imp, delta, signal).
        """
        from hermes_memory.core import importance_update

        decision = policy.evaluate("No, that's wrong, the correct value is 42")
        assert decision.category == "correction"
        imp = decision.initial_importance
        assert imp == 0.9

        # Feed into dynamics: positive signal should increase, negative decrease
        imp_up = importance_update(imp, delta=0.1, signal=1.0)
        assert 0.0 <= imp_up <= 1.0
        assert imp_up >= imp, (
            f"Positive signal should increase importance: {imp} -> {imp_up}"
        )

        imp_down = importance_update(imp, delta=0.1, signal=-1.0)
        assert 0.0 <= imp_down <= 1.0
        assert imp_down <= imp, (
            f"Negative signal should decrease importance: {imp} -> {imp_down}"
        )

    def test_encoding_decision_is_dataclass(self, policy):
        """EncodingDecision has all required fields as a dataclass."""
        decision = policy.evaluate("I like Python")
        assert dataclasses.is_dataclass(decision), (
            "EncodingDecision must be a dataclass"
        )
        field_names = {f.name for f in dataclasses.fields(decision)}
        required_fields = {
            "should_store",
            "category",
            "confidence",
            "reason",
            "initial_importance",
        }
        assert required_fields <= field_names, (
            f"EncodingDecision missing fields: {required_fields - field_names}"
        )

    def test_encoding_config_is_frozen_dataclass(self):
        """EncodingConfig is a frozen dataclass with documented defaults."""
        config = EncodingConfig()
        assert dataclasses.is_dataclass(config), "EncodingConfig must be a dataclass"
        # Verify it is frozen
        assert dataclasses.fields(config)  # has fields
        # Check default values match spec Section 2.1
        assert config.min_reasoning_length == 100
        assert config.min_reasoning_connectives == 2
        assert config.max_greeting_length == 50
        assert config.max_transactional_length == 80
        assert config.confidence_threshold == 0.5


# ============================================================
# Priority Ordering Tests (supplementary)
# ============================================================


class TestPriorityOrdering:
    """Verify classification priority from spec Section 3.2.

    Priority: correction > instruction > preference > fact > reasoning > transactional > greeting
    """

    def test_correction_beats_preference(self, policy):
        """Text matching both correction and preference -> correction wins."""
        decision = policy.evaluate("No, actually I prefer the other approach")
        assert decision.category == "correction", (
            f"Correction should beat preference, got '{decision.category}'"
        )

    def test_correction_beats_instruction(self, policy):
        """Text matching both correction and instruction -> correction wins."""
        decision = policy.evaluate("No, actually you should always use the new API")
        assert decision.category == "correction", (
            f"Correction should beat instruction, got '{decision.category}'"
        )

    def test_instruction_beats_fact(self, policy):
        """Text matching both instruction and fact -> instruction wins."""
        decision = policy.evaluate(
            "I am the team lead, so always follow the coding standards"
        )
        assert decision.category == "instruction", (
            f"Instruction should beat fact, got '{decision.category}'"
        )

    def test_preference_beats_transactional(self, policy):
        """Text matching both preference and transactional -> preference wins."""
        decision = policy.evaluate(
            "Run the tests but I prefer using pytest over unittest"
        )
        assert decision.category == "preference", (
            f"Preference should beat transactional, got '{decision.category}'"
        )


# ============================================================
# Metadata Influence Tests (supplementary)
# ============================================================


class TestMetadataInfluence:
    """Verify metadata boost effects on confidence scoring."""

    def test_message_count_boosts_confidence(self, policy):
        """Metadata with message_count > 5 adds 0.1 to confidence."""
        without = policy.evaluate("ok thanks")
        with_meta = policy.evaluate("ok thanks", metadata={"message_count": 12})
        # Spec says +0.1 boost when message_count > 5
        assert with_meta.confidence >= without.confidence, (
            f"Metadata boost should not decrease confidence: "
            f"{without.confidence} -> {with_meta.confidence}"
        )
        expected_diff = 0.1
        actual_diff = with_meta.confidence - without.confidence
        assert actual_diff == pytest.approx(expected_diff, abs=0.01), (
            f"Expected ~{expected_diff} confidence boost from message_count, "
            f"got {actual_diff}"
        )

    def test_long_text_boosts_confidence(self, policy):
        """Text > 500 chars gets +0.05 confidence boost."""
        short_text = "I prefer dark mode"
        long_text = "I prefer dark mode. " + ("Extra context. " * 40)
        assert len(long_text) > 500

        short_decision = policy.evaluate(short_text)
        long_decision = policy.evaluate(long_text)
        # Both should be preference, but long one gets +0.05
        assert long_decision.confidence >= short_decision.confidence, (
            f"Long text should have higher confidence: "
            f"{short_decision.confidence} -> {long_decision.confidence}"
        )

    def test_metadata_does_not_change_category(self, policy):
        """Metadata boosts confidence but does not change category assignment."""
        text = "Thanks for your help"
        without = policy.evaluate(text)
        with_meta = policy.evaluate(text, metadata={"message_count": 20})
        assert without.category == with_meta.category, (
            f"Metadata should not change category: "
            f"'{without.category}' -> '{with_meta.category}'"
        )


# ============================================================
# CATEGORY_IMPORTANCE Constant Tests
# ============================================================


class TestCategoryImportanceMapping:
    """Verify the CATEGORY_IMPORTANCE constant matches the spec."""

    def test_all_categories_have_importance(self):
        """Every valid category has an importance mapping."""
        for cat in VALID_CATEGORIES:
            assert cat in CATEGORY_IMPORTANCE, (
                f"Category '{cat}' missing from CATEGORY_IMPORTANCE"
            )

    def test_importance_values_match_spec(self):
        """Importance values match spec Section 2.3."""
        expected = {
            "correction": 0.9,
            "instruction": 0.85,
            "preference": 0.8,
            "fact": 0.6,
            "unclassified": 0.5,
            "reasoning": 0.4,
            "greeting": 0.0,
            "transactional": 0.0,
        }
        for cat, imp in expected.items():
            assert CATEGORY_IMPORTANCE[cat] == imp, (
                f"CATEGORY_IMPORTANCE['{cat}'] = {CATEGORY_IMPORTANCE[cat]}, expected {imp}"
            )

    def test_importance_ordering(self):
        """Importance follows: correction > instruction > preference > fact > unclassified > reasoning > greeting = transactional."""
        assert CATEGORY_IMPORTANCE["correction"] > CATEGORY_IMPORTANCE["instruction"]
        assert CATEGORY_IMPORTANCE["instruction"] > CATEGORY_IMPORTANCE["preference"]
        assert CATEGORY_IMPORTANCE["preference"] > CATEGORY_IMPORTANCE["fact"]
        assert CATEGORY_IMPORTANCE["fact"] > CATEGORY_IMPORTANCE["unclassified"]
        assert CATEGORY_IMPORTANCE["unclassified"] > CATEGORY_IMPORTANCE["reasoning"]
        assert CATEGORY_IMPORTANCE["reasoning"] > CATEGORY_IMPORTANCE["greeting"]
        assert CATEGORY_IMPORTANCE["greeting"] == CATEGORY_IMPORTANCE["transactional"]


# ============================================================
# Audit Finding Tests (Findings 7-19)
# ============================================================


class TestWordBoundaryPatterns:
    """Tests for Findings 7, 8, 9: word boundary correctness."""

    def test_hello_does_not_match_othello(self, policy):
        """Finding 7: Pattern 'hello' should NOT match 'othello'.

        With use_word_boundaries=True, 'hello' becomes r'\\bhello\\b'
        which should not match inside 'othello'. Verify at the pattern
        matching level that greeting patterns don't fire on 'othello'.
        """
        text_lower = "i saw othello at the theater last night"
        compiled_greeting = policy._compiled_patterns.get("first_greeting", [])
        greeting_matches = policy._count_matches(
            text_lower, [], compiled_greeting
        )
        assert greeting_matches == 0, (
            f"'othello' should not trigger greeting pattern 'hello', "
            f"got {greeting_matches} greeting matches"
        )

    def test_hello_in_othello_no_false_positive_category(self, policy):
        """Finding 7 (extended): text containing 'othello' and greeting content.

        If text contains 'othello' AND starts with a real greeting pattern,
        only the real greeting should match, not the substring in 'othello'.
        """
        # Without word boundaries, "othello" would double-count "hello"
        no_boundary_policy = EncodingPolicy(EncodingConfig(use_word_boundaries=False))
        text = "othello is a great play"
        # With word boundaries: no greeting match from "othello"
        compiled_wb = policy._compiled_patterns.get("first_greeting", [])
        matches_wb = policy._count_matches(text.lower(), [], compiled_wb)
        # Without word boundaries: "hello" substring matches in "othello"
        from hermes_memory.encoding import GREETING_PATTERNS
        matches_no_wb = no_boundary_policy._count_matches(
            text.lower(), GREETING_PATTERNS, None
        )
        assert matches_wb == 0, (
            f"Word boundaries should prevent 'hello' matching in 'othello', "
            f"got {matches_wb} matches"
        )
        assert matches_no_wb > 0, (
            f"Without word boundaries, 'hello' should match in 'othello', "
            f"got {matches_no_wb} matches"
        )

    def test_so_does_not_match_also(self, policy):
        """Finding 8: Pattern 'so' should NOT match inside 'also'.

        With word boundaries, 'so' becomes r'\\bso\\b' which should not
        match as a substring of 'also'.
        """
        # "also" should not trigger the "so " reasoning connective
        text = "I also think this approach works well for our use case"
        decision = policy.evaluate(text)
        # The text should NOT match reasoning from "so" inside "also"
        # It should match preference/fact from "I" patterns instead
        category, confidence = policy.classify(text)
        # Verify "reasoning" was not assigned from "also" matching "so"
        text_lower = text.lower()
        compiled_reasoning = policy._compiled_patterns.get("first_reasoning", [])
        reasoning_matches = policy._count_matches(
            text_lower, [], compiled_reasoning
        )
        # "also" should not count as a match for the "so" pattern
        assert reasoning_matches == 0, (
            f"'also' should not match reasoning connective 'so', "
            f"got {reasoning_matches} reasoning matches"
        )

    def test_no_does_not_match_know(self, policy):
        """Finding 9: Pattern 'no' should NOT match inside 'know'.

        With word boundaries, 'no' becomes r'\\bno\\b' which should not
        match as a substring of 'know'.
        """
        text = "I know the answer to that question"
        decision = policy.evaluate(text)
        # "know" should not trigger the "no " correction pattern
        assert decision.category != "correction", (
            f"'know' should not trigger correction pattern 'no', "
            f"got category '{decision.category}'"
        )

    def test_standalone_no_still_matches(self, policy):
        """Word boundaries should still match standalone 'no'."""
        decision = policy.evaluate("No, that is incorrect")
        assert decision.category == "correction", (
            f"Standalone 'no' should still match correction, "
            f"got '{decision.category}'"
        )

    def test_standalone_hello_still_matches(self, policy):
        """Word boundaries should still match standalone 'hello'."""
        decision = policy.evaluate("Hello!")
        assert decision.category == "greeting", (
            f"Standalone 'hello' should still match greeting, "
            f"got '{decision.category}'"
        )


class TestConfidenceDenominator:
    """Tests for Finding 10: third-person patterns should not inflate denominator."""

    def test_first_person_confidence_not_diluted_by_third_person(self, policy):
        """Finding 10: Confidence should use per-set denominators, not combined.

        When text matches first-person patterns only (no metadata for
        third-person), confidence denominator should be first-person
        pattern count only, not first + third combined.
        """
        # "I prefer" matches 1 first-person preference pattern
        # With only first-person patterns (14 total), calibrated floor is 0.35
        decision = policy.evaluate("I prefer dark mode")
        assert decision.confidence >= 0.35, (
            f"Single first-person pattern match should yield >= 0.35 confidence, "
            f"got {decision.confidence}"
        )

    def test_episode_source_confidence_uses_combined_denominator(self, policy):
        """When source_type=episode, third-person patterns are checked too.

        The denominator includes both sets, which is correct for episode content.
        """
        decision = policy.evaluate(
            "The user expressed a preference for dark mode",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "preference"
        # With combined patterns, denominator is larger but match count is also
        # potentially larger. Confidence should still be calibrated properly.
        assert decision.confidence >= 0.35, (
            f"Episode preference should have calibrated confidence >= 0.35, "
            f"got {decision.confidence}"
        )


class TestConnectiveCountCaching:
    """Tests for Finding 11: connective count recomputed in write policy."""

    def test_reasoning_write_policy_consistent_with_classify(self, policy):
        """Finding 11: _apply_write_policy should use same connective counting.

        This is a behavioral test: verify that the reasoning write policy
        produces correct results. The fix is caching connective count
        to avoid recomputation, but the observable behavior should be identical.
        """
        text = (
            "The system should use a write-ahead log because it provides durability "
            "guarantees. Therefore, even if the process crashes, we can recover the "
            "state from the log since the entries are flushed to disk before "
            "acknowledging the write."
        )
        decision = policy.evaluate(text)
        assert decision.category == "reasoning"
        assert decision.should_store is True, (
            "Long reasoning with multiple connectives should be stored"
        )


class TestCategoryValidation:
    """Tests for Finding 12: no category validation."""

    def test_classify_returns_valid_category(self, policy):
        """Finding 12: classify() must always return a valid category."""
        test_cases = [
            "I like Python",
            "Run the tests",
            "Hello!",
            "I am a developer",
            "No, that's wrong",
            "Always use type hints",
            "Because therefore since thus hence",
            "Random text with no patterns whatsoever xyz abc",
            "",
        ]
        for text in test_cases:
            if text == "":
                # Empty text handled by evaluate() before classify()
                continue
            category, confidence = policy.classify(text)
            assert category in VALID_CATEGORIES, (
                f"classify() returned invalid category '{category}' for: {text[:50]!r}"
            )

    def test_evaluate_returns_valid_category_always(self, policy):
        """Finding 12: evaluate() must always return a valid category."""
        test_cases = [
            "I like Python",
            "Run the tests",
            "Hello!",
            "",
            None,
            "a" * 10000,
            "12345 67890 !@#$%",
        ]
        for text in test_cases:
            decision = policy.evaluate(text)
            assert decision.category in VALID_CATEGORIES, (
                f"evaluate() returned invalid category '{decision.category}'"
            )


class TestBuildReasonInformative:
    """Tests for Finding 13: _build_reason minimally informative."""

    def test_reason_includes_category(self, policy):
        """Finding 13: reason string should include the category."""
        decision = policy.evaluate("I prefer dark mode")
        assert "preference" in decision.reason.lower(), (
            f"Reason should mention category 'preference', got: {decision.reason}"
        )

    def test_reason_includes_confidence(self, policy):
        """Finding 13: reason string should include confidence score."""
        decision = policy.evaluate("I prefer dark mode")
        # Reason should contain a confidence number
        assert any(c.isdigit() for c in decision.reason), (
            f"Reason should include confidence value, got: {decision.reason}"
        )

    def test_reason_includes_matched_patterns(self, policy):
        """Finding 13: reason string should include which patterns matched."""
        decision = policy.evaluate("I prefer dark mode")
        # After the fix, reason should mention the matched patterns
        assert "prefer" in decision.reason.lower() or "pattern" in decision.reason.lower(), (
            f"Reason should mention matched patterns, got: {decision.reason}"
        )

    def test_reason_includes_policy(self, policy):
        """Finding 13: reason string should mention write policy."""
        decision = policy.evaluate("I prefer dark mode")
        assert "always" in decision.reason.lower() or "policy" in decision.reason.lower() or "store" in decision.reason.lower(), (
            f"Reason should mention write policy, got: {decision.reason}"
        )


class TestLogging:
    """Tests for Finding 14: no logging/telemetry."""

    def test_encoding_module_has_logger(self):
        """Finding 14: encoding module should have a logger."""
        import hermes_memory.encoding as enc
        import logging

        assert hasattr(enc, "logger"), "encoding module should define a logger"
        assert isinstance(enc.logger, logging.Logger), (
            f"encoding.logger should be a logging.Logger, got {type(enc.logger)}"
        )

    def test_logger_name_matches_module(self):
        """Finding 14: logger name should be the module name."""
        import hermes_memory.encoding as enc

        assert enc.logger.name == "hermes_memory.encoding", (
            f"Logger name should be 'hermes_memory.encoding', got '{enc.logger.name}'"
        )

    def test_evaluate_logs_at_debug_level(self, policy):
        """Finding 14: evaluate() should produce debug-level log messages."""
        import logging

        logger = logging.getLogger("hermes_memory.encoding")
        with LogCapture(logger) as captured:
            policy.evaluate("I prefer dark mode")
        assert len(captured.records) > 0, (
            "evaluate() should produce at least one debug log message"
        )

    def test_evaluate_logs_classification_result(self, policy):
        """Finding 14: debug logs should include classification result."""
        import logging

        logger = logging.getLogger("hermes_memory.encoding")
        with LogCapture(logger) as captured:
            policy.evaluate("I prefer dark mode")
        log_text = " ".join(r.getMessage() for r in captured.records)
        assert "preference" in log_text.lower(), (
            f"Debug logs should include category, got: {log_text}"
        )


class LogCapture:
    """Context manager to capture log records from a logger."""

    def __init__(self, logger):
        self.logger = logger
        self.records = []
        self._handler = None
        self._old_level = None

    def __enter__(self):
        import logging

        self._old_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)

        class _Handler(logging.Handler):
            def __init__(self, records):
                super().__init__(level=logging.DEBUG)
                self.records = records

            def emit(self, record):
                self.records.append(record)

        self._handler = _Handler(self.records)
        self.logger.addHandler(self._handler)
        return self

    def __exit__(self, *args):
        self.logger.removeHandler(self._handler)
        self.logger.setLevel(self._old_level)


class TestPriorityOrderDerived:
    """Tests for Finding 15: PRIORITY_ORDER hardcoded."""

    def test_priority_order_covers_all_categories(self):
        """Finding 15: PRIORITY_ORDER must cover all valid categories."""
        from hermes_memory.encoding import PRIORITY_ORDER

        assert set(PRIORITY_ORDER) == set(VALID_CATEGORIES), (
            f"PRIORITY_ORDER {set(PRIORITY_ORDER)} != VALID_CATEGORIES {set(VALID_CATEGORIES)}"
        )

    def test_priority_order_has_no_duplicates(self):
        """Finding 15: PRIORITY_ORDER should have no duplicate entries."""
        from hermes_memory.encoding import PRIORITY_ORDER

        assert len(PRIORITY_ORDER) == len(set(PRIORITY_ORDER)), (
            f"PRIORITY_ORDER has duplicates: {PRIORITY_ORDER}"
        )


class TestEncodingConfigValidation:
    """Tests for Finding 16: EncodingConfig lacks validation."""

    def test_confidence_threshold_must_be_in_range(self):
        """Finding 16: confidence_threshold must be in [0.0, 1.0]."""
        with pytest.raises((ValueError, TypeError)):
            EncodingPolicy(EncodingConfig(confidence_threshold=1.5))

    def test_confidence_threshold_negative_rejected(self):
        """Finding 16: negative confidence_threshold must be rejected."""
        with pytest.raises((ValueError, TypeError)):
            EncodingPolicy(EncodingConfig(confidence_threshold=-0.1))

    def test_min_reasoning_length_must_be_positive(self):
        """Finding 16: min_reasoning_length must be > 0."""
        with pytest.raises((ValueError, TypeError)):
            EncodingPolicy(EncodingConfig(min_reasoning_length=0))

    def test_min_reasoning_connectives_must_be_nonnegative(self):
        """Finding 16: min_reasoning_connectives must be >= 0."""
        with pytest.raises((ValueError, TypeError)):
            EncodingPolicy(EncodingConfig(min_reasoning_connectives=-1))

    def test_max_greeting_length_must_be_positive(self):
        """Finding 16: max_greeting_length must be > 0."""
        with pytest.raises((ValueError, TypeError)):
            EncodingPolicy(EncodingConfig(max_greeting_length=0))

    def test_max_transactional_length_must_be_positive(self):
        """Finding 16: max_transactional_length must be > 0."""
        with pytest.raises((ValueError, TypeError)):
            EncodingPolicy(EncodingConfig(max_transactional_length=0))

    def test_valid_config_accepted(self):
        """Valid configuration values should be accepted."""
        config = EncodingConfig(
            confidence_threshold=0.5,
            min_reasoning_length=100,
            min_reasoning_connectives=2,
            max_greeting_length=50,
            max_transactional_length=80,
        )
        policy = EncodingPolicy(config)
        assert policy.config == config


class TestDocumentation:
    """Tests for Findings 17, 18: missing documentation."""

    def test_single_pattern_confidence_floor_documented(self):
        """Finding 17: SINGLE_PATTERN_CONFIDENCE_FLOOR should have documentation."""
        import hermes_memory.encoding as enc
        import inspect

        source = inspect.getsource(enc)
        # There should be a comment or docstring near SINGLE_PATTERN_CONFIDENCE_FLOOR
        lines = source.split("\n")
        floor_line = None
        for i, line in enumerate(lines):
            if "SINGLE_PATTERN_CONFIDENCE_FLOOR" in line and "=" in line:
                floor_line = i
                break
        assert floor_line is not None, (
            "SINGLE_PATTERN_CONFIDENCE_FLOOR definition not found"
        )
        # Check that there's a comment nearby (within 3 lines before or after)
        nearby = "\n".join(lines[max(0, floor_line - 3) : floor_line + 4])
        has_doc = "#" in nearby or '"""' in nearby
        assert has_doc, (
            f"SINGLE_PATTERN_CONFIDENCE_FLOOR should have nearby documentation. "
            f"Context: {nearby}"
        )

    def test_episode_length_offset_documented(self):
        """Finding 18: episode_length_offset field should have documentation."""
        import hermes_memory.encoding as enc
        import inspect

        source = inspect.getsource(enc)
        lines = source.split("\n")
        offset_line = None
        for i, line in enumerate(lines):
            if "episode_length_offset" in line and ":" in line:
                offset_line = i
                break
        assert offset_line is not None, (
            "episode_length_offset field definition not found"
        )
        # Check for a comment nearby
        nearby = "\n".join(lines[max(0, offset_line - 3) : offset_line + 6])
        has_doc = "#" in nearby or '"""' in nearby
        assert has_doc, (
            f"episode_length_offset should have nearby documentation. "
            f"Context: {nearby}"
        )


class TestReclassifyRefactored:
    """Tests for Finding 19: _reclassify_without duplicates classify logic."""

    def test_reclassify_without_greeting_gives_valid_result(self, policy):
        """Finding 19: _reclassify_without should produce valid categories."""
        text = (
            "Hello! I wanted to tell you that I just moved to a new apartment "
            "in downtown Portland and I'm really excited about it."
        )
        category, confidence = policy._reclassify_without(text, "greeting")
        assert category in VALID_CATEGORIES, (
            f"_reclassify_without returned invalid category '{category}'"
        )
        assert category != "greeting", (
            f"_reclassify_without should not return the excluded category 'greeting'"
        )
        assert 0.0 <= confidence <= 1.0

    def test_reclassify_consistent_with_classify(self, policy):
        """Finding 19: reclassification should be consistent with classify.

        If we classify text that matches preference + greeting, and then
        reclassify without greeting, the result should still be preference
        (since preference has higher priority anyway).
        """
        text = "Hello, I prefer using dark mode for all my editors"
        # Regular classify should give preference (higher priority than greeting)
        cat1, conf1 = policy.classify(text)
        # Reclassify without greeting should also give preference
        cat2, conf2 = policy._reclassify_without(text, "greeting")
        assert cat2 == cat1 or cat2 == "preference", (
            f"Reclassify without greeting should still find preference, "
            f"got '{cat2}' (original was '{cat1}')"
        )


# ============================================================
# Reasoning Knowledge Type Shortcut Tests
# ============================================================


class TestReasoningKnowledgeTypeShortcut:
    """Verify that knowledge_type='reasoning' maps to category='reasoning'
    via the semantic memory shortcut (KNOWLEDGE_TYPE_TO_CATEGORY).

    Gap identified in integration review: KNOWLEDGE_TYPE_TO_CATEGORY maps 10
    knowledge types but has no 'reasoning' entry. Smart agents storing
    architecture decisions via metadata path can't get reasoning (0.40
    importance), must use 'context' -> fact (0.60) which over-weights
    transient reasoning.
    """

    def test_reasoning_knowledge_type_in_mapping(self):
        """'reasoning' must be a key in KNOWLEDGE_TYPE_TO_CATEGORY."""
        from hermes_memory.encoding import KNOWLEDGE_TYPE_TO_CATEGORY

        assert "reasoning" in KNOWLEDGE_TYPE_TO_CATEGORY, (
            "'reasoning' missing from KNOWLEDGE_TYPE_TO_CATEGORY — "
            "smart agents can't classify reasoning via metadata path"
        )
        assert KNOWLEDGE_TYPE_TO_CATEGORY["reasoning"] == "reasoning", (
            f"Expected 'reasoning' -> 'reasoning', "
            f"got 'reasoning' -> '{KNOWLEDGE_TYPE_TO_CATEGORY.get('reasoning')}'"
        )

    def test_semantic_shortcut_reasoning_category(self, policy):
        """Semantic memory with knowledge_type='reasoning' -> category='reasoning'."""
        decision = policy.evaluate(
            "The user reasoned that microservices would improve deploy velocity",
            metadata={"source_type": "semantic", "knowledge_type": "reasoning"},
        )
        assert decision.category == "reasoning", (
            f"Expected category 'reasoning' via semantic shortcut, "
            f"got '{decision.category}'"
        )

    def test_semantic_shortcut_reasoning_confidence(self, policy):
        """Semantic shortcut assigns confidence=0.85 for reasoning."""
        decision = policy.evaluate(
            "The user explained that caching reduces latency due to fewer DB round trips",
            metadata={"source_type": "semantic", "knowledge_type": "reasoning"},
        )
        assert decision.confidence == 0.85, (
            f"Expected confidence 0.85 from semantic shortcut, "
            f"got {decision.confidence}"
        )

    def test_semantic_shortcut_reasoning_importance(self, policy):
        """Reasoning via semantic shortcut gets importance=0.4."""
        decision = policy.evaluate(
            "The user's reasoning was that eventual consistency suits this use case",
            metadata={"source_type": "semantic", "knowledge_type": "reasoning"},
        )
        assert decision.initial_importance == 0.4, (
            f"Expected importance 0.4 for reasoning, "
            f"got {decision.initial_importance}"
        )

    def test_semantic_shortcut_reasoning_stored(self, policy):
        """Reasoning via semantic shortcut should_store=True (LLM validated)."""
        decision = policy.evaluate(
            "Architecture decision: use event sourcing because it provides auditability",
            metadata={"source_type": "semantic", "knowledge_type": "reasoning"},
        )
        assert decision.should_store is True, (
            "Semantic reasoning should be stored (LLM already validated content)"
        )

    def test_semantic_shortcut_reasoning_short_text_still_stored(self, policy):
        """Short reasoning via semantic shortcut bypasses length/connective checks.

        The write policy has a special case: when source_type='semantic' and
        knowledge_type is in the mapping, skip the min_reasoning_length and
        min_reasoning_connectives checks. The LLM has already validated this
        is genuine reasoning.
        """
        decision = policy.evaluate(
            "Use Redis for caching",
            metadata={"source_type": "semantic", "knowledge_type": "reasoning"},
        )
        assert decision.should_store is True, (
            "Short semantic reasoning should still be stored — "
            "LLM validated, bypass length/connective checks"
        )
        assert decision.category == "reasoning"
        assert decision.initial_importance == 0.4

    def test_reasoning_not_confused_with_context(self, policy):
        """knowledge_type='reasoning' should NOT fall through to 'context' -> fact.

        This is the specific bug: without 'reasoning' in the mapping, agents
        must use knowledge_type='context' which maps to 'fact' (importance 0.6)
        instead of 'reasoning' (importance 0.4), over-weighting transient reasoning.
        """
        reasoning_decision = policy.evaluate(
            "The team decided to use PostgreSQL because of ACID guarantees",
            metadata={"source_type": "semantic", "knowledge_type": "reasoning"},
        )
        context_decision = policy.evaluate(
            "The team decided to use PostgreSQL because of ACID guarantees",
            metadata={"source_type": "semantic", "knowledge_type": "context"},
        )
        assert reasoning_decision.category == "reasoning", (
            f"knowledge_type='reasoning' should map to 'reasoning', "
            f"got '{reasoning_decision.category}'"
        )
        assert context_decision.category == "fact", (
            f"knowledge_type='context' should map to 'fact', "
            f"got '{context_decision.category}'"
        )
        assert reasoning_decision.initial_importance < context_decision.initial_importance, (
            f"Reasoning importance ({reasoning_decision.initial_importance}) should be "
            f"less than fact importance ({context_decision.initial_importance})"
        )


# ============================================================
# 9. Emotional Valence Tests (~8 tests)
# ============================================================


class TestEmotionalValence:
    """Verify emotional valence as a dimensional modifier on EncodingDecision.

    Per NLP literature (EMNLP 2021, NRC-VAD), emotion is NOT a separate category
    but a dimensional modifier. "I'm frustrated with TypeScript" is a preference
    with negative valence, not a new "emotion" category.

    Emotional valence:
      - None = no emotion detected
      - positive float (0.0 to 1.0) = positive emotion
      - negative float (-1.0 to 0.0) = negative emotion
    """

    def test_emotion_field_exists(self, policy):
        """EncodingDecision must have an emotional_valence field."""
        decision = policy.evaluate("I like Python")
        assert hasattr(decision, "emotional_valence"), (
            "EncodingDecision must have an 'emotional_valence' field"
        )

    def test_emotion_detection_positive(self, policy):
        """'I love this approach' -> emotional_valence > 0."""
        decision = policy.evaluate("I love this approach")
        assert decision.emotional_valence is not None, (
            "Positive emotion keyword 'love' should be detected"
        )
        assert decision.emotional_valence > 0, (
            f"Expected positive valence for 'love', got {decision.emotional_valence}"
        )

    def test_emotion_detection_negative(self, policy):
        """'I'm frustrated with TypeScript' -> emotional_valence < 0."""
        decision = policy.evaluate("I'm frustrated with TypeScript")
        assert decision.emotional_valence is not None, (
            "Negative emotion keyword 'frustrated' should be detected"
        )
        assert decision.emotional_valence < 0, (
            f"Expected negative valence for 'frustrated', got {decision.emotional_valence}"
        )

    def test_emotion_no_detection(self, policy):
        """'The sky is blue' -> emotional_valence is None."""
        decision = policy.evaluate("The sky is blue")
        assert decision.emotional_valence is None, (
            f"No emotion keywords in 'The sky is blue', expected None, "
            f"got {decision.emotional_valence}"
        )

    def test_emotion_importance_boost(self, policy):
        """Emotional content gets +0.05 importance boost."""
        # "I love dark mode" has emotion ("love") and is a preference (base 0.8)
        emotional = policy.evaluate("I love dark mode")
        # "I prefer dark mode" has no strong emotion keyword, same category
        neutral = policy.evaluate("I prefer dark mode")

        assert emotional.category == "preference", (
            f"Expected 'preference', got '{emotional.category}'"
        )
        assert neutral.category == "preference", (
            f"Expected 'preference', got '{neutral.category}'"
        )
        assert emotional.initial_importance == neutral.initial_importance + 0.05, (
            f"Emotional importance ({emotional.initial_importance}) should be "
            f"neutral importance ({neutral.initial_importance}) + 0.05"
        )

    def test_emotion_preserves_category(self, policy):
        """'I love dark mode' still classified as 'preference', not a new category."""
        decision = policy.evaluate("I love dark mode")
        assert decision.category == "preference", (
            f"Emotional text should keep its base category 'preference', "
            f"got '{decision.category}'"
        )
        assert decision.category in VALID_CATEGORIES, (
            f"Category must remain in VALID_CATEGORIES, got '{decision.category}'"
        )

    def test_emotion_mixed_valence(self, policy):
        """Text with both positive and negative keywords gets net valence."""
        decision = policy.evaluate(
            "I love the flexibility but I'm frustrated with the complexity"
        )
        assert decision.emotional_valence is not None, (
            "Mixed emotional text should still have a valence (not None)"
        )
        # With 1 positive ("love") and 1 negative ("frustrated"), net is ~0
        # The exact value depends on the implementation but should reflect
        # the mixed nature (could be slightly positive, negative, or near zero)

    def test_emotion_strong_positive(self, policy):
        """Strong positive emotion keywords detected."""
        decision = policy.evaluate("I'm ecstatic about this new framework")
        assert decision.emotional_valence is not None, (
            "Strong emotion keyword 'ecstatic' should be detected"
        )
        assert decision.emotional_valence > 0, (
            f"Expected positive valence for 'ecstatic', got {decision.emotional_valence}"
        )

    def test_emotion_strong_negative(self, policy):
        """Strong negative emotion keywords detected."""
        decision = policy.evaluate("I'm furious about the breaking changes")
        assert decision.emotional_valence is not None, (
            "Strong emotion keyword 'furious' should be detected"
        )
        assert decision.emotional_valence < 0, (
            f"Expected negative valence for 'furious', got {decision.emotional_valence}"
        )

    def test_emotion_importance_capped_at_one(self, policy):
        """Emotion boost should not push importance above 1.0."""
        # Correction has importance 0.9; with +0.05 boost = 0.95, within bounds
        decision = policy.evaluate("No, actually I'm furious that you got this wrong")
        assert decision.initial_importance <= 1.0, (
            f"Importance must not exceed 1.0, got {decision.initial_importance}"
        )

    def test_emotion_valence_range(self, policy):
        """emotional_valence must be None or in [-1.0, 1.0]."""
        test_cases = [
            "I love Python",
            "I hate bugs",
            "The sky is blue",
            "I'm frustrated with JavaScript",
            "I'm ecstatic about the results",
            "Hello!",
            "",
        ]
        for text in test_cases:
            decision = policy.evaluate(text)
            if decision.emotional_valence is not None:
                assert -1.0 <= decision.emotional_valence <= 1.0, (
                    f"emotional_valence {decision.emotional_valence} out of range "
                    f"for: {text!r}"
                )

    def test_emotion_empty_input_no_valence(self, policy):
        """Empty input should have emotional_valence=None."""
        decision = policy.evaluate("")
        assert decision.emotional_valence is None, (
            f"Empty input should have emotional_valence=None, "
            f"got {decision.emotional_valence}"
        )

    def test_emotion_does_not_change_category_set(self):
        """VALID_CATEGORIES must NOT include 'emotion' as a category."""
        assert "emotion" not in VALID_CATEGORIES, (
            "Emotion must NOT be a category -- it is a dimensional modifier"
        )

    def test_emotion_valence_deterministic(self, policy):
        """Same input always produces same emotional_valence."""
        text = "I love this feature but hate the bugs"
        d1 = policy.evaluate(text)
        d2 = policy.evaluate(text)
        assert d1.emotional_valence == d2.emotional_valence, (
            f"Determinism violated: emotional_valence "
            f"{d1.emotional_valence} != {d2.emotional_valence}"
        )


class TestEmotionalValencePropertyBased:
    """Property-based tests for emotional valence."""

    @given(text=st.text(min_size=0, max_size=5000))
    @settings(max_examples=MAX_EXAMPLES)
    def test_emotional_valence_always_valid(self, text):
        """For any text, emotional_valence is None or in [-1.0, 1.0]."""
        policy = EncodingPolicy()
        decision = policy.evaluate(text)
        if decision.emotional_valence is not None:
            assert -1.0 <= decision.emotional_valence <= 1.0, (
                f"emotional_valence {decision.emotional_valence} out of bounds "
                f"for {text[:80]!r}"
            )


# ============================================================
# 10. Third-Person Pattern Coverage Parity Tests (Issue 3)
# ============================================================


class TestThirdPersonCoverageParity:
    """Verify third-person pattern counts are >= first-person counts.

    Issue 3: Third-person pattern sets were 14-38% smaller than first-person
    equivalents, causing episodes using "they"/"their"/proper names to fall
    through to default classification.
    """

    def test_preference_pattern_parity(self):
        """Third-person preference patterns >= first-person count."""
        from hermes_memory.encoding import (
            PREFERENCE_PATTERNS,
            THIRD_PERSON_PREFERENCE_PATTERNS,
        )
        assert len(THIRD_PERSON_PREFERENCE_PATTERNS) >= len(PREFERENCE_PATTERNS), (
            f"Third-person preference patterns ({len(THIRD_PERSON_PREFERENCE_PATTERNS)}) "
            f"< first-person ({len(PREFERENCE_PATTERNS)})"
        )

    def test_fact_pattern_parity(self):
        """Third-person fact patterns >= first-person count."""
        from hermes_memory.encoding import (
            FACT_PATTERNS,
            THIRD_PERSON_FACT_PATTERNS,
        )
        assert len(THIRD_PERSON_FACT_PATTERNS) >= len(FACT_PATTERNS), (
            f"Third-person fact patterns ({len(THIRD_PERSON_FACT_PATTERNS)}) "
            f"< first-person ({len(FACT_PATTERNS)})"
        )

    def test_correction_pattern_parity(self):
        """Third-person correction patterns >= first-person count."""
        from hermes_memory.encoding import (
            CORRECTION_PATTERNS,
            THIRD_PERSON_CORRECTION_PATTERNS,
        )
        assert len(THIRD_PERSON_CORRECTION_PATTERNS) >= len(CORRECTION_PATTERNS), (
            f"Third-person correction patterns ({len(THIRD_PERSON_CORRECTION_PATTERNS)}) "
            f"< first-person ({len(CORRECTION_PATTERNS)})"
        )

    def test_instruction_pattern_parity(self):
        """Third-person instruction patterns >= first-person count."""
        from hermes_memory.encoding import (
            INSTRUCTION_PATTERNS,
            THIRD_PERSON_INSTRUCTION_PATTERNS,
        )
        assert len(THIRD_PERSON_INSTRUCTION_PATTERNS) >= len(INSTRUCTION_PATTERNS), (
            f"Third-person instruction patterns ({len(THIRD_PERSON_INSTRUCTION_PATTERNS)}) "
            f"< first-person ({len(INSTRUCTION_PATTERNS)})"
        )

    def test_reasoning_pattern_parity(self):
        """Third-person reasoning patterns >= first-person count."""
        from hermes_memory.encoding import (
            REASONING_CONNECTIVES,
            THIRD_PERSON_REASONING_CONNECTIVES,
        )
        assert len(THIRD_PERSON_REASONING_CONNECTIVES) >= len(REASONING_CONNECTIVES), (
            f"Third-person reasoning patterns ({len(THIRD_PERSON_REASONING_CONNECTIVES)}) "
            f"< first-person ({len(REASONING_CONNECTIVES)})"
        )

    def test_greeting_pattern_parity(self):
        """Third-person greeting patterns >= first-person count."""
        from hermes_memory.encoding import (
            GREETING_PATTERNS,
            THIRD_PERSON_GREETING_PATTERNS,
        )
        assert len(THIRD_PERSON_GREETING_PATTERNS) >= len(GREETING_PATTERNS), (
            f"Third-person greeting patterns ({len(THIRD_PERSON_GREETING_PATTERNS)}) "
            f"< first-person ({len(GREETING_PATTERNS)})"
        )

    def test_transactional_pattern_parity(self):
        """Third-person transactional patterns >= first-person count."""
        from hermes_memory.encoding import (
            TRANSACTIONAL_PATTERNS,
            THIRD_PERSON_TRANSACTIONAL_PATTERNS,
        )
        assert len(THIRD_PERSON_TRANSACTIONAL_PATTERNS) >= len(TRANSACTIONAL_PATTERNS), (
            f"Third-person transactional patterns ({len(THIRD_PERSON_TRANSACTIONAL_PATTERNS)}) "
            f"< first-person ({len(TRANSACTIONAL_PATTERNS)})"
        )


class TestThirdPersonTheyPronouns:
    """Verify 'they/their' pronoun variants classify correctly in episode context."""

    def test_they_prefer_preference(self, policy):
        """'they prefer dark mode' -> preference (episode context)."""
        decision = policy.evaluate(
            "they prefer dark mode",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "preference", (
            f"Expected 'preference' for 'they prefer', got '{decision.category}'"
        )

    def test_they_like_preference(self, policy):
        """'they like using vim' -> preference (episode context)."""
        decision = policy.evaluate(
            "they like using vim",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "preference", (
            f"Expected 'preference' for 'they like', got '{decision.category}'"
        )

    def test_they_enjoy_preference(self, policy):
        """'they enjoy pair programming' -> preference (episode context)."""
        decision = policy.evaluate(
            "they enjoy pair programming",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "preference", (
            f"Expected 'preference' for 'they enjoy', got '{decision.category}'"
        )

    def test_their_favorite_preference(self, policy):
        """'their favorite language is Rust' -> preference (episode context)."""
        decision = policy.evaluate(
            "their favorite language is Rust",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "preference", (
            f"Expected 'preference' for 'their favorite', got '{decision.category}'"
        )

    def test_they_live_in_fact(self, policy):
        """'they live in San Francisco' -> fact (episode context)."""
        decision = policy.evaluate(
            "they live in San Francisco",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "fact", (
            f"Expected 'fact' for 'they live in', got '{decision.category}'"
        )

    def test_they_work_at_fact(self, policy):
        """'they work at Google' -> fact (episode context)."""
        decision = policy.evaluate(
            "they work at Google",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "fact", (
            f"Expected 'fact' for 'they work at', got '{decision.category}'"
        )

    def test_their_name_fact(self, policy):
        """'their name is Alice' -> fact (episode context)."""
        decision = policy.evaluate(
            "their name is Alice",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "fact", (
            f"Expected 'fact' for 'their name', got '{decision.category}'"
        )

    def test_their_email_fact(self, policy):
        """'their email is alice@example.com' -> fact (episode context)."""
        decision = policy.evaluate(
            "their email is alice@example.com",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "fact", (
            f"Expected 'fact' for 'their email', got '{decision.category}'"
        )

    def test_their_job_fact(self, policy):
        """'their job is software engineer' -> fact (episode context)."""
        decision = policy.evaluate(
            "their job is software engineer",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "fact", (
            f"Expected 'fact' for 'their job', got '{decision.category}'"
        )

    def test_they_corrected_correction(self, policy):
        """'they corrected the date to March 15' -> correction (episode context)."""
        decision = policy.evaluate(
            "they corrected the date to March 15",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "correction", (
            f"Expected 'correction' for 'they corrected', got '{decision.category}'"
        )

    def test_they_clarified_correction(self, policy):
        """'they clarified the requirements' -> correction (episode context)."""
        decision = policy.evaluate(
            "they clarified the requirements",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "correction", (
            f"Expected 'correction' for 'they clarified', got '{decision.category}'"
        )

    def test_they_pointed_out_correction(self, policy):
        """'they pointed out the mistake' -> correction (episode context)."""
        decision = policy.evaluate(
            "they pointed out the mistake",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "correction", (
            f"Expected 'correction' for 'they pointed out', got '{decision.category}'"
        )

    def test_they_instructed_instruction(self, policy):
        """'they instructed to always use type hints' -> instruction (episode context)."""
        decision = policy.evaluate(
            "they instructed to always use type hints",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "instruction", (
            f"Expected 'instruction' for 'they instructed', got '{decision.category}'"
        )

    def test_they_requested_instruction(self, policy):
        """'they requested that tests be run first' -> instruction (episode context)."""
        decision = policy.evaluate(
            "they requested that tests be run first",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "instruction", (
            f"Expected 'instruction' for 'they requested', got '{decision.category}'"
        )

    def test_they_emphasized_instruction(self, policy):
        """'they emphasized the importance of code review' -> instruction (episode context)."""
        decision = policy.evaluate(
            "they emphasized the importance of code review",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "instruction", (
            f"Expected 'instruction' for 'they emphasized', got '{decision.category}'"
        )

    def test_they_reasoned_reasoning(self, policy):
        """'they reasoned that caching would help' -> reasoning (episode context)."""
        decision = policy.evaluate(
            "they reasoned that caching would help",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "reasoning", (
            f"Expected 'reasoning' for 'they reasoned', got '{decision.category}'"
        )

    def test_they_explained_reasoning(self, policy):
        """'they explained that the design was more scalable' -> reasoning (episode context)."""
        decision = policy.evaluate(
            "they explained that the design was more scalable",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "reasoning", (
            f"Expected 'reasoning' for 'they explained', got '{decision.category}'"
        )

    def test_their_reasoning_reasoning(self, policy):
        """'their reasoning was based on performance data' -> reasoning (episode context)."""
        decision = policy.evaluate(
            "their reasoning was based on performance data",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "reasoning", (
            f"Expected 'reasoning' for 'their reasoning', got '{decision.category}'"
        )

    def test_they_greeted_greeting(self, policy):
        """'they greeted the assistant' -> greeting (episode context)."""
        decision = policy.evaluate(
            "they greeted the assistant",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "greeting", (
            f"Expected 'greeting' for 'they greeted', got '{decision.category}'"
        )

    def test_they_thanked_greeting(self, policy):
        """'they thanked the assistant for help' -> greeting (episode context)."""
        decision = policy.evaluate(
            "they thanked the assistant for help",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "greeting", (
            f"Expected 'greeting' for 'they thanked', got '{decision.category}'"
        )

    def test_they_said_goodbye_greeting(self, policy):
        """'they said goodbye and ended the session' -> greeting (episode context)."""
        decision = policy.evaluate(
            "they said goodbye and ended the session",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "greeting", (
            f"Expected 'greeting' for 'they said goodbye', got '{decision.category}'"
        )

    def test_they_asked_to_run_transactional(self, policy):
        """'they asked to run the tests' -> transactional (episode context)."""
        decision = policy.evaluate(
            "they asked to run the tests",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "transactional", (
            f"Expected 'transactional' for 'they asked to run', got '{decision.category}'"
        )

    def test_they_asked_to_deploy_transactional(self, policy):
        """'they asked to deploy the application' -> transactional (episode context)."""
        decision = policy.evaluate(
            "they asked to deploy the application",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "transactional", (
            f"Expected 'transactional' for 'they asked to deploy', got '{decision.category}'"
        )

    def test_they_asked_to_start_transactional(self, policy):
        """'they asked to start the server' -> transactional (episode context)."""
        decision = policy.evaluate(
            "the user asked to start the server",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "transactional", (
            f"Expected 'transactional' for 'the user asked to start', got '{decision.category}'"
        )


class TestThirdPersonPossessiveVariants:
    """Verify 'their' possessive variants in third-person patterns."""

    def test_their_preference_for(self, policy):
        """'their preference is for dark themes' -> preference (episode context)."""
        decision = policy.evaluate(
            "their preference is for dark themes",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "preference", (
            f"Expected 'preference' for 'their preference', got '{decision.category}'"
        )

    def test_their_address_fact(self, policy):
        """'their address is 123 Main St' -> fact (episode context)."""
        decision = policy.evaluate(
            "their address is 123 Main St",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "fact", (
            f"Expected 'fact' for 'their address', got '{decision.category}'"
        )

    def test_their_birthday_fact(self, policy):
        """'their birthday is in March' -> fact (episode context)."""
        decision = policy.evaluate(
            "their birthday is in March",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "fact", (
            f"Expected 'fact' for 'their birthday', got '{decision.category}'"
        )

    def test_their_role_fact(self, policy):
        """'their role is senior engineer' -> fact (episode context)."""
        decision = policy.evaluate(
            "their role is senior engineer",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "fact", (
            f"Expected 'fact' for 'their role', got '{decision.category}'"
        )

    def test_they_corrected_that_correction(self, policy):
        """'they corrected that the port is 8080' -> correction (episode context)."""
        decision = policy.evaluate(
            "they corrected that the port is 8080",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "correction", (
            f"Expected 'correction' for 'they corrected that', got '{decision.category}'"
        )

    def test_they_directed_instruction(self, policy):
        """'they directed the assistant to use black' -> instruction (episode context)."""
        decision = policy.evaluate(
            "they directed the assistant to use black",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "instruction", (
            f"Expected 'instruction' for 'they directed', got '{decision.category}'"
        )

    def test_they_specified_instruction(self, policy):
        """'they specified that all files must use UTF-8' -> instruction (episode context)."""
        decision = policy.evaluate(
            "they specified that all files must use UTF-8",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "instruction", (
            f"Expected 'instruction' for 'they specified', got '{decision.category}'"
        )

    def test_their_logic_reasoning(self, policy):
        """'their logic was that caching reduces latency' -> reasoning (episode context)."""
        decision = policy.evaluate(
            "their logic was that caching reduces latency",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "reasoning", (
            f"Expected 'reasoning' for 'their logic', got '{decision.category}'"
        )

    def test_they_asked_to_build_transactional(self, policy):
        """'they asked to build the Docker image' -> transactional (episode context)."""
        decision = policy.evaluate(
            "they asked to build the Docker image",
            metadata={"source_type": "episode"},
        )
        assert decision.category == "transactional", (
            f"Expected 'transactional' for 'they asked to build', got '{decision.category}'"
        )


# ============================================================
# Issue 5: Language Detection and Unclassified Category
# ============================================================


class TestDetectNonEnglish:
    """Tests for _detect_non_english() heuristic function.

    The heuristic detects non-English text by measuring the ratio of non-ASCII
    alpha characters. This reliably catches non-Latin-script languages (CJK,
    Cyrillic, Arabic, Devanagari, etc.) but may not catch European languages
    written mostly in ASCII-compatible Latin characters (e.g., French without
    accents, German). This is an acceptable trade-off for a zero-dependency
    heuristic in a proof system.
    """

    def test_cyrillic_detected_as_non_english(self):
        """Russian text (Cyrillic script) is detected as non-English."""
        # "Privet, menya zovut Pyotr" in Russian
        assert _detect_non_english(
            "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0435\u043d\u044f "
            "\u0437\u043e\u0432\u0443\u0442 \u041f\u0451\u0442\u0440"
        ) is True

    def test_english_not_detected(self):
        """English text 'Hello, my name is Pierre' is NOT detected as non-English."""
        assert _detect_non_english("Hello, my name is Pierre") is False

    def test_mixed_language_mostly_english(self):
        """'I like cafe au lait' (mostly English/ASCII) is NOT detected as non-English."""
        assert _detect_non_english("I like cafe au lait") is False

    def test_empty_text_not_non_english(self):
        """Empty string is not detected as non-English."""
        assert _detect_non_english("") is False

    def test_ascii_ratio_threshold(self):
        """Text with exactly 30% non-ASCII alpha chars is NOT detected (threshold is >30%)."""
        # Build text: 7 ASCII alpha + 3 non-ASCII alpha = 30% non-ASCII exactly
        # 30% is not > 30%, so should return False
        text = "abcdefg\u00e9\u00e8\u00ea"  # 7 ASCII + 3 non-ASCII = 30%
        assert _detect_non_english(text) is False

    def test_above_threshold_detected(self):
        """Text with >30% non-ASCII alpha chars is detected as non-English."""
        # 6 ASCII + 4 non-ASCII = 40% > 30%
        text = "abcdef\u00e9\u00e8\u00ea\u00eb"
        assert _detect_non_english(text) is True

    def test_numbers_and_punctuation_ignored(self):
        """Numbers and punctuation are not counted in the ratio."""
        # Only alpha chars matter. "123!@# abc" has 3 alpha, all ASCII
        assert _detect_non_english("123!@# abc") is False

    def test_japanese_detected(self):
        """Japanese text is detected as non-English."""
        assert _detect_non_english("\u3053\u3093\u306b\u3061\u306f\u4e16\u754c") is True

    def test_arabic_detected(self):
        """Arabic text is detected as non-English."""
        assert _detect_non_english(
            "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"
        ) is True

    def test_cyrillic_detected(self):
        """Cyrillic/Russian text is detected as non-English."""
        assert _detect_non_english(
            "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"
        ) is True

    def test_pure_numbers_not_non_english(self):
        """String with only numbers and no alpha is not non-English."""
        assert _detect_non_english("12345 67890") is False

    def test_whitespace_only_not_non_english(self):
        """Whitespace-only is not non-English."""
        assert _detect_non_english("   \t\n  ") is False

    def test_french_with_accents_below_threshold(self):
        """French text with accents but still under 30% non-ASCII is not detected.

        This is a known limitation of the ASCII-ratio heuristic: European
        languages in Latin script often have too few non-ASCII characters.
        """
        # "Je suis developpeur francais" -> ~6% non-ASCII
        text = "Bonjour, je suis d\u00e9veloppeur fran\u00e7ais"
        assert _detect_non_english(text) is False

    def test_hindi_devanagari_detected(self):
        """Hindi text (Devanagari script) is detected as non-English."""
        # "Namaste duniya" in Hindi
        assert _detect_non_english(
            "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u0927"
        ) is True

    def test_korean_detected(self):
        """Korean text (Hangul script) is detected as non-English."""
        # "Annyeonghaseyo" in Korean
        assert _detect_non_english(
            "\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4"
        ) is True


class TestUnclassifiedCategory:
    """Tests for the 'unclassified' category for non-English content.

    Uses non-Latin-script languages (Russian, Japanese, Arabic) because the
    ASCII-ratio heuristic reliably detects these. European languages in Latin
    script (French, German) may not trigger detection due to low non-ASCII ratios.
    """

    def test_unclassified_in_valid_categories(self):
        """'unclassified' must be in VALID_CATEGORIES."""
        assert "unclassified" in VALID_CATEGORIES

    def test_unclassified_importance_is_neutral(self):
        """'unclassified' importance must be 0.5 (neutral)."""
        assert CATEGORY_IMPORTANCE["unclassified"] == 0.5

    def test_non_english_gets_unclassified(self, policy):
        """Non-English text with no pattern matches -> category 'unclassified'."""
        # Russian: "I am a software developer and I work at a company"
        decision = policy.evaluate(
            "\u042f \u0440\u0430\u0437\u0440\u0430\u0431\u043e\u0442\u0447\u0438\u043a "
            "\u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u043d\u043e\u0433\u043e "
            "\u043e\u0431\u0435\u0441\u043f\u0435\u0447\u0435\u043d\u0438\u044f"
        )
        assert decision.category == "unclassified", (
            f"Non-English text should be 'unclassified', got '{decision.category}'"
        )

    def test_non_english_importance_0_5(self, policy):
        """Non-English text gets importance 0.5 (neutral)."""
        # Japanese: "Programming is a craft that requires patience"
        decision = policy.evaluate(
            "\u30d7\u30ed\u30b0\u30e9\u30df\u30f3\u30b0\u306f\u5fcd\u8010\u3092"
            "\u5fc5\u8981\u3068\u3059\u308b\u6280\u8853\u3067\u3059"
        )
        assert decision.initial_importance == 0.5, (
            f"Non-English unclassified text should have importance 0.5, "
            f"got {decision.initial_importance}"
        )

    def test_unclassified_always_stored(self, policy):
        """Unclassified content should_store=True (ALWAYS store policy)."""
        # Arabic text
        decision = policy.evaluate(
            "\u0623\u0646\u0627 \u0645\u0637\u0648\u0631 \u0628\u0631\u0645\u062c\u064a\u0627\u062a"
        )
        assert decision.should_store is True, (
            "Unclassified non-English content must be stored"
        )

    def test_english_no_patterns_not_unclassified(self, policy):
        """English text with no pattern matches should NOT be 'unclassified'.

        English text that matches no patterns should still use the normal
        length-based fallback (greeting for short, fact for long).
        """
        # "Lorem ipsum" is ASCII / Latin, under 30% non-ASCII
        decision = policy.evaluate("Lorem ipsum dolor sit amet")
        assert decision.category != "unclassified", (
            f"English/ASCII text should not be 'unclassified', "
            f"got '{decision.category}'"
        )

    def test_non_english_with_matching_pattern_not_unclassified(self, policy):
        """Non-English text that matches a pattern gets that category, not unclassified.

        Patterns take precedence over language detection. If text containing
        non-ASCII characters also contains 'I prefer', the preference pattern
        should win because patterns are checked before the fallback path.
        """
        # Mix of English pattern with Cyrillic context
        decision = policy.evaluate(
            "I prefer \u044d\u0442\u043e\u0442 \u043f\u043e\u0434\u0445\u043e\u0434"
        )
        assert decision.category == "preference", (
            f"Pattern match should take precedence over language detection, "
            f"got '{decision.category}'"
        )

    def test_detected_language_field_english(self, policy):
        """English text gets detected_language='en'."""
        decision = policy.evaluate("I prefer dark mode")
        assert decision.detected_language == "en", (
            f"Expected detected_language='en' for English text, "
            f"got '{decision.detected_language}'"
        )

    def test_detected_language_field_non_english(self, policy):
        """Non-English text gets detected_language='non-en'."""
        # Russian text
        decision = policy.evaluate(
            "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0435\u043d\u044f "
            "\u0437\u043e\u0432\u0443\u0442 \u041f\u0451\u0442\u0440 "
            "\u0438 \u044f \u0440\u0430\u0437\u0440\u0430\u0431\u043e\u0442\u0447\u0438\u043a"
        )
        assert decision.detected_language == "non-en", (
            f"Expected detected_language='non-en' for Russian text, "
            f"got '{decision.detected_language}'"
        )

    def test_detected_language_field_uncertain(self, policy):
        """Text where language is uncertain gets detected_language=None."""
        # Pure numbers/symbols - uncertain
        decision = policy.evaluate("12345 67890 !@#$%")
        assert decision.detected_language is None, (
            f"Expected detected_language=None for uncertain text, "
            f"got '{decision.detected_language}'"
        )

    def test_short_non_english_unclassified(self, policy):
        """Short non-English text (would be 'greeting' for English) -> 'unclassified'."""
        # Short Korean text -- would normally fall to "greeting" via length heuristic
        decision = policy.evaluate("\uc548\ub155\ud558\uc138\uc694")
        assert decision.category == "unclassified", (
            f"Short non-English text should be 'unclassified', "
            f"got '{decision.category}'"
        )
        assert decision.initial_importance == 0.5

    def test_long_non_english_unclassified(self, policy):
        """Long non-English text (would be 'fact' for English) -> 'unclassified'."""
        # Long Russian text about programming
        text = (
            "\u041f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 "
            "\u044d\u0442\u043e \u0438\u0441\u043a\u0443\u0441\u0441\u0442\u0432\u043e, "
            "\u043a\u043e\u0442\u043e\u0440\u043e\u0435 \u0442\u0440\u0435\u0431\u0443\u0435\u0442 "
            "\u043c\u043d\u043e\u0433\u043e \u0442\u0435\u0440\u043f\u0435\u043d\u0438\u044f "
            "\u0438 \u043f\u0440\u0430\u043a\u0442\u0438\u043a\u0438 \u0434\u043b\u044f "
            "\u043e\u0441\u0432\u043e\u0435\u043d\u0438\u044f \u043e\u0441\u043d\u043e\u0432"
        )
        decision = policy.evaluate(text)
        assert decision.category == "unclassified", (
            f"Long non-English text should be 'unclassified', "
            f"got '{decision.category}'"
        )
        assert decision.initial_importance == 0.5

    def test_unclassified_in_priority_order(self):
        """'unclassified' must be in PRIORITY_ORDER (last position)."""
        from hermes_memory.encoding import PRIORITY_ORDER

        assert "unclassified" in PRIORITY_ORDER
        assert PRIORITY_ORDER[-1] == "unclassified", (
            f"'unclassified' should be last in PRIORITY_ORDER, "
            f"order is: {PRIORITY_ORDER}"
        )

    def test_unclassified_in_consolidation_compression(self):
        """'unclassified' must have a compression ratio in consolidation."""
        from hermes_memory.consolidation import CATEGORY_COMPRESSION_RATIOS

        assert "unclassified" in CATEGORY_COMPRESSION_RATIOS, (
            "'unclassified' missing from CATEGORY_COMPRESSION_RATIOS"
        )
        assert CATEGORY_COMPRESSION_RATIOS["unclassified"] == 5

    def test_unclassified_excluded_from_contradiction(self):
        """'unclassified' must be excluded from contradiction detection."""
        from hermes_memory.contradiction import EXCLUDED_CATEGORIES

        assert "unclassified" in EXCLUDED_CATEGORIES, (
            "'unclassified' should be excluded from contradiction detection"
        )
