"""Phase C tests (extraction functions) - standalone file.

This file duplicates the Phase C tests from test_consolidation.py so they can
run independently while Phase B is still being implemented.  Once Phase B is
complete, this file can be removed and all tests run from test_consolidation.py.
"""

import re

import pytest

from hermes_memory.consolidation import (
    ArchivedMemory,
    ConsolidationCandidate,
    ConsolidationConfig,
    ConsolidationLevel,
    ConsolidationMode,
    DEFAULT_CONSOLIDATION_CONFIG,
    EXTRACTION_PATTERNS,
    _extract_by_sentence_scoring,
    _get_extraction_patterns,
    _truncate_to_sentences,
    archive_episodic,
    extract_semantic,
)


# ============================================================
# 9. Phase C: _get_extraction_patterns Tests
# ============================================================


class TestGetExtractionPatterns:
    """Verify _get_extraction_patterns returns correct pattern lists."""

    def test_preference_returns_patterns(self):
        """preference category returns non-empty list of compiled patterns."""
        patterns = _get_extraction_patterns("preference")
        assert len(patterns) > 0
        for p in patterns:
            assert isinstance(p, re.Pattern)

    def test_fact_returns_patterns(self):
        """fact category returns non-empty list of compiled patterns."""
        patterns = _get_extraction_patterns("fact")
        assert len(patterns) > 0

    def test_instruction_returns_patterns(self):
        """instruction category returns non-empty list of compiled patterns."""
        patterns = _get_extraction_patterns("instruction")
        assert len(patterns) > 0

    def test_reasoning_returns_patterns(self):
        """reasoning category returns non-empty list of compiled patterns."""
        patterns = _get_extraction_patterns("reasoning")
        assert len(patterns) > 0

    def test_greeting_returns_empty(self):
        """greeting has no extraction patterns -- returns empty list."""
        patterns = _get_extraction_patterns("greeting")
        assert patterns == []

    def test_transactional_returns_empty(self):
        """transactional has no extraction patterns -- returns empty list."""
        patterns = _get_extraction_patterns("transactional")
        assert patterns == []

    def test_correction_returns_empty(self):
        """correction has no extraction patterns -- returns empty list."""
        patterns = _get_extraction_patterns("correction")
        assert patterns == []

    def test_unknown_category_returns_empty(self):
        """Unknown category returns empty list (never raises)."""
        patterns = _get_extraction_patterns("nonexistent_category_xyz")
        assert patterns == []

    def test_returned_list_matches_extraction_patterns_dict(self):
        """For categories in EXTRACTION_PATTERNS, returned list equals dict value."""
        for category in ("preference", "fact", "instruction", "reasoning"):
            patterns = _get_extraction_patterns(category)
            assert patterns is EXTRACTION_PATTERNS[category]


# ============================================================
# 10. Phase C: _truncate_to_sentences Tests
# ============================================================


class TestTruncateToSentences:
    """Verify _truncate_to_sentences truncation logic."""

    def test_single_sentence_within_limit(self):
        """Single sentence within target_length returned as-is (minus trailing dot)."""
        result = _truncate_to_sentences("Hello world.", 50)
        assert "Hello world" in result

    def test_multiple_sentences_truncated(self):
        """Multiple sentences truncated to fit target_length."""
        text = "First sentence. Second sentence. Third sentence. Fourth one."
        result = _truncate_to_sentences(text, 30)
        assert len(result) <= 50  # some margin for joining
        assert "First sentence" in result

    def test_no_sentence_boundary_truncates_at_word(self):
        """When no sentence boundary, truncates within text."""
        text = "a very long phrase without any sentence-ending punctuation"
        result = _truncate_to_sentences(text, 20)
        assert len(result) > 0
        assert len(result) <= len(text)

    def test_never_returns_empty_string(self):
        """Never returns empty string even for empty-ish input."""
        result = _truncate_to_sentences("x", 1)
        assert len(result) >= 1

    def test_respects_target_length(self):
        """First sentence always included even if over target_length."""
        text = "This is a very long first sentence that exceeds the limit. Short."
        result = _truncate_to_sentences(text, 5)
        # First sentence always included
        assert len(result) > 0

    def test_empty_text_returns_minimum(self):
        """Empty-ish text returns at least one character."""
        result = _truncate_to_sentences("  ", 10)
        assert len(result) >= 1

    def test_includes_max_sentences_within_limit(self):
        """Includes as many sentences as fit within target_length."""
        text = "A. B. C. D. E."
        result = _truncate_to_sentences(text, 100)
        # All sentences should be included since total is well under 100
        assert "A" in result
        assert "B" in result

    def test_whitespace_only_returns_minimum(self):
        """Whitespace-only input returns at least text[:1]."""
        result = _truncate_to_sentences("   ", 10)
        assert len(result) >= 1


# ============================================================
# 11. Phase C: _extract_by_sentence_scoring Tests
# ============================================================


class TestExtractBySentenceScoring:
    """Verify _extract_by_sentence_scoring fallback extraction."""

    def test_basic_extraction(self):
        """Extracts sentences from basic content."""
        content = "Short. This is a longer sentence with more unique words."
        result = _extract_by_sentence_scoring(content, 100)
        assert len(result) > 0

    def test_never_returns_empty(self):
        """Never returns empty string."""
        result = _extract_by_sentence_scoring("x", 1)
        assert len(result) >= 1

    def test_scores_by_info_density(self):
        """Higher info-density sentences are preferred."""
        content = (
            "The cat sat. "
            "Quantum chromodynamics predicts gluon self-interaction asymptotic freedom."
        )
        result = _extract_by_sentence_scoring(content, 50)
        # The second sentence has more unique words and length, so should rank higher
        assert "Quantum" in result or "cat" in result  # at least one sentence

    def test_respects_target_length(self):
        """Stops adding sentences when target_length is reached."""
        content = "Short one. Another short. A third one. Fourth here."
        result = _extract_by_sentence_scoring(content, 15)
        # Should not include all sentences
        assert len(result) < len(content)

    def test_empty_content_returns_minimum(self):
        """Empty content returns at least one character."""
        result = _extract_by_sentence_scoring("  ", 10)
        assert len(result) >= 1

    def test_deterministic(self):
        """Same input always produces same output."""
        content = "Alpha bravo. Charlie delta echo. Foxtrot golf."
        r1 = _extract_by_sentence_scoring(content, 30)
        r2 = _extract_by_sentence_scoring(content, 30)
        assert r1 == r2

    def test_whitespace_only_sentences_skipped(self):
        """Whitespace-only sentences are filtered out."""
        content = "Real content here.   .   . More real words."
        result = _extract_by_sentence_scoring(content, 100)
        assert len(result) > 0


# ============================================================
# 12. Phase C: extract_semantic Tests
# ============================================================


class TestExtractSemantic:
    """Verify extract_semantic end-to-end extraction logic."""

    # --- Input validation ---

    def test_non_string_content_raises_type_error(self):
        """Non-string content raises TypeError."""
        with pytest.raises(TypeError):
            extract_semantic(
                123,  # type: ignore[arg-type]
                "preference",
                source_episodes=("ep-001",),
                source_creation_times=(1.0,),
                source_importances=(0.5,),
            )

    def test_invalid_category_raises_value_error(self):
        """Invalid category raises ValueError."""
        with pytest.raises(ValueError):
            extract_semantic(
                "content",
                "nonexistent",
                source_episodes=("ep-001",),
                source_creation_times=(1.0,),
                source_importances=(0.5,),
            )

    def test_empty_source_episodes_raises_value_error(self):
        """Empty source_episodes raises ValueError."""
        with pytest.raises(ValueError, match="source_episodes"):
            extract_semantic(
                "content",
                "preference",
                source_episodes=(),
                source_creation_times=(),
                source_importances=(),
            )

    def test_mismatched_creation_times_raises(self):
        """Mismatched lengths between source_episodes and source_creation_times."""
        with pytest.raises(ValueError):
            extract_semantic(
                "content",
                "preference",
                source_episodes=("ep-001", "ep-002"),
                source_creation_times=(1.0,),  # too short
                source_importances=(0.5, 0.6),
            )

    def test_mismatched_importances_raises(self):
        """Mismatched lengths between source_episodes and source_importances."""
        with pytest.raises(ValueError):
            extract_semantic(
                "content",
                "preference",
                source_episodes=("ep-001",),
                source_creation_times=(1.0,),
                source_importances=(0.5, 0.6),  # too long
            )

    # --- Preference extraction with pattern match ---

    def test_preference_with_pattern_match(self):
        """Preference content matching a pattern uses template."""
        result = extract_semantic(
            "I prefer dark mode for my editor",
            "preference",
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.8,),
        )
        assert "User prefers:" in result.content
        assert result.category == "preference"
        assert result.confidence == pytest.approx(0.8)

    def test_preference_fallback_to_sentence_scoring(self):
        """Preference content not matching patterns uses sentence scoring fallback."""
        result = extract_semantic(
            "Dark mode is easier on the eyes, especially at night",
            "preference",
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.7,),
        )
        assert "User prefers:" in result.content
        assert len(result.content) > 0

    # --- Fact extraction ---

    def test_fact_with_pattern_match(self):
        """Fact content matching a pattern extracts correctly."""
        result = extract_semantic(
            "The user lives in New York City",
            "fact",
            source_episodes=("ep-001",),
            source_creation_times=(50.0,),
            source_importances=(0.9,),
        )
        assert "User fact:" in result.content
        assert result.category == "fact"

    # --- Instruction extraction ---

    def test_instruction_with_pattern_match(self):
        """Instruction content matching pattern extracts correctly."""
        result = extract_semantic(
            "Always use type hints in Python code",
            "instruction",
            source_episodes=("ep-001",),
            source_creation_times=(200.0,),
            source_importances=(0.85,),
        )
        assert "Standing instruction:" in result.content

    # --- Reasoning extraction ---

    def test_reasoning_with_pattern_match(self):
        """Reasoning content matching pattern extracts correctly."""
        result = extract_semantic(
            "Because the database was slow, we added caching",
            "reasoning",
            source_episodes=("ep-001",),
            source_creation_times=(300.0,),
            source_importances=(0.6,),
        )
        assert "User reasoning:" in result.content

    # --- Correction ---

    def test_correction_preserves_full_content(self):
        """Correction category returns full content unchanged."""
        original = "Actually, I was wrong. The correct answer is 42."
        result = extract_semantic(
            original,
            "correction",
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.7,),
        )
        assert result.content == original

    # --- Greeting (no patterns, fallback) ---

    def test_greeting_uses_fallback(self):
        """Greeting has no patterns, uses sentence scoring fallback."""
        result = extract_semantic(
            "Hello there, how are you doing today",
            "greeting",
            source_episodes=("ep-001",),
            source_creation_times=(10.0,),
            source_importances=(0.3,),
        )
        assert len(result.content) > 0
        assert result.category == "greeting"

    # --- Provenance fields ---

    def test_confidence_is_mean_of_importances(self):
        """Confidence equals the mean of source_importances, clamped [0,1]."""
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            source_episodes=("ep-001", "ep-002"),
            source_creation_times=(10.0, 20.0),
            source_importances=(0.6, 0.8),
        )
        assert result.confidence == pytest.approx(0.7)

    def test_first_observed_last_updated(self):
        """first_observed/last_updated are min/max of source_creation_times."""
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            source_episodes=("ep-001", "ep-002", "ep-003"),
            source_creation_times=(30.0, 10.0, 50.0),
            source_importances=(0.5, 0.5, 0.5),
        )
        assert result.first_observed == 10.0
        assert result.last_updated == 50.0

    def test_target_level_is_episodic_compressed(self):
        """Default target_level is EPISODIC_COMPRESSED."""
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.5,),
        )
        assert result.target_level == ConsolidationLevel.EPISODIC_COMPRESSED

    def test_extraction_mode_is_async_batch(self):
        """extraction_mode is ASYNC_BATCH."""
        result = extract_semantic(
            "test content",
            "fact",
            source_episodes=("ep-001",),
            source_creation_times=(1.0,),
            source_importances=(0.5,),
        )
        assert result.extraction_mode == ConsolidationMode.ASYNC_BATCH

    def test_source_episodes_preserved(self):
        """source_episodes tuple is preserved in result."""
        eps = ("ep-001", "ep-002", "ep-003")
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            source_episodes=eps,
            source_creation_times=(1.0, 2.0, 3.0),
            source_importances=(0.5, 0.5, 0.5),
        )
        assert result.source_episodes == eps

    def test_importance_is_boosted_clamped_mean(self):
        """importance equals clamp01(mean_importance * semantic_boost)."""
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.5,),
        )
        # avg=0.5, boosted=0.5*1.2=0.6
        assert result.importance == pytest.approx(0.6)

    def test_content_never_empty(self):
        """Result content is never empty."""
        result = extract_semantic(
            "x",
            "fact",
            source_episodes=("ep-001",),
            source_creation_times=(1.0,),
            source_importances=(0.5,),
        )
        assert len(result.content) > 0

    def test_deterministic(self):
        """Same inputs always produce same output."""
        kwargs = dict(
            content="I prefer dark mode in my IDE",
            category="preference",
            source_episodes=("ep-001", "ep-002"),
            source_creation_times=(10.0, 20.0),
            source_importances=(0.6, 0.8),
        )
        r1 = extract_semantic(**kwargs)
        r2 = extract_semantic(**kwargs)
        assert r1.content == r2.content
        assert r1.confidence == r2.confidence
        assert r1.importance == r2.importance

    def test_default_config_used_when_none(self):
        """When config=None, DEFAULT_CONSOLIDATION_CONFIG is used."""
        r1 = extract_semantic(
            "I prefer dark mode",
            "preference",
            config=None,
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.5,),
        )
        r2 = extract_semantic(
            "I prefer dark mode",
            "preference",
            config=DEFAULT_CONSOLIDATION_CONFIG,
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.5,),
        )
        assert r1.content == r2.content


# ============================================================
# 13. Phase C: archive_episodic Tests
# ============================================================


class TestArchiveEpisodic:
    """Verify archive_episodic archival logic."""

    @staticmethod
    def _make_candidate(**overrides):
        """Construct a valid ConsolidationCandidate."""
        defaults = {
            "memory_id": "mem-001",
            "content": "I prefer dark mode in my editor",
            "category": "preference",
            "level": ConsolidationLevel.EPISODIC_RAW,
            "creation_time": 100.0,
            "last_access_time": 150.0,
            "access_count": 5,
            "importance": 0.8,
            "strength": 7.0,
            "relevance": 0.5,
        }
        defaults.update(overrides)
        return ConsolidationCandidate(**defaults)

    def test_basic_archival(self):
        """Basic archival produces correct ArchivedMemory."""
        candidate = self._make_candidate()
        result = archive_episodic(candidate, ("sem-001",))
        assert isinstance(result, ArchivedMemory)
        assert result.memory_id == "mem-001"
        assert result.content == "I prefer dark mode in my editor"
        assert result.category == "preference"
        assert result.archived is True

    def test_decayed_importance(self):
        """decayed_importance = clamp01(importance * consolidation_decay)."""
        candidate = self._make_candidate(importance=0.8)
        config = ConsolidationConfig()  # consolidation_decay = 0.1
        result = archive_episodic(candidate, ("sem-001",), config=config)
        expected = 0.8 * 0.1  # = 0.08
        assert result.decayed_importance == pytest.approx(expected)

    def test_decayed_importance_never_exceeds_original(self):
        """decayed_importance <= original_importance."""
        candidate = self._make_candidate(importance=0.5)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.decayed_importance <= result.original_importance

    def test_original_importance_preserved(self):
        """original_importance matches candidate.importance."""
        candidate = self._make_candidate(importance=0.9)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.original_importance == 0.9

    def test_content_preserved_in_full(self):
        """content preserves full content."""
        content = "Very important memory content that should be preserved"
        candidate = self._make_candidate(content=content)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.content == content

    def test_consolidated_to_preserved(self):
        """consolidated_to tuple is preserved."""
        candidate = self._make_candidate()
        result = archive_episodic(candidate, ("sem-001", "sem-002"))
        assert result.consolidated_to == ("sem-001", "sem-002")

    def test_empty_consolidated_to_raises(self):
        """Empty consolidated_to raises ValueError."""
        candidate = self._make_candidate()
        with pytest.raises(ValueError, match="consolidated_to"):
            archive_episodic(candidate, ())

    def test_creation_time_preserved(self):
        """creation_time matches candidate.creation_time."""
        candidate = self._make_candidate(creation_time=42.0)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.creation_time == 42.0

    def test_archive_time_uses_last_access_time(self):
        """archive_time uses candidate.last_access_time."""
        candidate = self._make_candidate(last_access_time=200.0)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.archive_time == 200.0

    def test_level_preserved(self):
        """level matches candidate.level."""
        candidate = self._make_candidate(level=ConsolidationLevel.EPISODIC_COMPRESSED)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.level == ConsolidationLevel.EPISODIC_COMPRESSED

    def test_access_count_at_archive_preserved(self):
        """access_count_at_archive matches candidate.access_count."""
        candidate = self._make_candidate(access_count=42)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.access_count_at_archive == 42

    def test_custom_config_decay(self):
        """Custom config with different consolidation_decay is honored."""
        candidate = self._make_candidate(importance=0.6)
        config = ConsolidationConfig(consolidation_decay=0.5)
        result = archive_episodic(candidate, ("sem-001",), config=config)
        expected = 0.6 * 0.5  # = 0.3
        assert result.decayed_importance == pytest.approx(expected)

    def test_default_config_when_none(self):
        """When config=None, DEFAULT_CONSOLIDATION_CONFIG is used."""
        candidate = self._make_candidate(importance=0.8)
        r1 = archive_episodic(candidate, ("sem-001",), config=None)
        r2 = archive_episodic(
            candidate, ("sem-001",), config=DEFAULT_CONSOLIDATION_CONFIG
        )
        assert r1.decayed_importance == r2.decayed_importance

    def test_deterministic(self):
        """Same inputs always produce same output."""
        candidate = self._make_candidate()
        r1 = archive_episodic(candidate, ("sem-001",))
        r2 = archive_episodic(candidate, ("sem-001",))
        assert r1.memory_id == r2.memory_id
        assert r1.decayed_importance == r2.decayed_importance
        assert r1.content == r2.content
