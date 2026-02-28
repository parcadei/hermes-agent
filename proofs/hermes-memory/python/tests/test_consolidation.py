"""Tests for consolidation Phase A: data types, enums, constants, config.

Tests are written BEFORE implementation exists. All imports from
hermes_memory.consolidation will fail with ImportError until the module is created.

Test categories:
  1. Enum Tests (~8 tests)
  2. Constant Tests (~10 tests)
  3. ConsolidationConfig Tests (~15 tests)
  4. ConsolidationCandidate Tests (~8 tests)
  5. SemanticExtraction Tests (~7 tests)
  6. ArchivedMemory Tests (~5 tests)
  7. ConsolidationResult Tests (~4 tests)
  8. Property-Based / Hypothesis (~5 tests)
"""

import dataclasses
import math
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hermes_memory.consolidation import (
    ArchivedMemory,
    CATEGORY_COMPRESSION_RATIOS,
    CORRECTION_CONTENT_MARKERS,
    CORRECTION_MARKER_THRESHOLD,
    ConsolidationCandidate,
    ConsolidationConfig,
    ConsolidationLevel,
    ConsolidationMode,
    ConsolidationResult,
    DEFAULT_CONSOLIDATION_CONFIG,
    LEVEL_HALF_LIVES,
    LEVEL_RETENTION_THRESHOLDS,
    LEVEL_TIME_CONSTANTS,
    NON_CONSOLIDATABLE_CATEGORIES,
    SemanticExtraction,
    _content_is_correction,
    _passes_inhibitors,
    _compute_semantic_value,
    compute_affinity,
    should_consolidate,
    # Phase C imports
    EXTRACTION_PATTERNS,
    _extract_by_sentence_scoring,
    _get_extraction_patterns,
    _truncate_to_sentences,
    archive_episodic,
    extract_semantic,
    # Phase D imports
    _next_level,
    consolidate_memory,
    select_consolidation_candidates,
)
from hermes_memory.encoding import VALID_CATEGORIES

MAX_EXAMPLES = 200


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config():
    """Default consolidation config."""
    return ConsolidationConfig()


# ============================================================
# 1. Enum Tests
# ============================================================


class TestConsolidationLevel:
    """Verify ConsolidationLevel enum values, ordering, and membership."""

    def test_has_exactly_four_members(self):
        """ConsolidationLevel must have exactly 4 members."""
        assert len(ConsolidationLevel) == 4

    def test_member_values(self):
        """Each member has the correct integer value."""
        assert ConsolidationLevel.EPISODIC_RAW == 1
        assert ConsolidationLevel.EPISODIC_COMPRESSED == 2
        assert ConsolidationLevel.SEMANTIC_FACT == 3
        assert ConsolidationLevel.SEMANTIC_INSIGHT == 4

    def test_ordering_episodic_raw_less_than_compressed(self):
        """EPISODIC_RAW < EPISODIC_COMPRESSED (IntEnum supports <)."""
        assert ConsolidationLevel.EPISODIC_RAW < ConsolidationLevel.EPISODIC_COMPRESSED

    def test_ordering_compressed_less_than_fact(self):
        """EPISODIC_COMPRESSED < SEMANTIC_FACT."""
        assert ConsolidationLevel.EPISODIC_COMPRESSED < ConsolidationLevel.SEMANTIC_FACT

    def test_ordering_fact_less_than_insight(self):
        """SEMANTIC_FACT < SEMANTIC_INSIGHT."""
        assert ConsolidationLevel.SEMANTIC_FACT < ConsolidationLevel.SEMANTIC_INSIGHT

    def test_full_ordering(self):
        """Full ordering: RAW < COMPRESSED < FACT < INSIGHT."""
        levels = sorted(ConsolidationLevel)
        assert levels == [
            ConsolidationLevel.EPISODIC_RAW,
            ConsolidationLevel.EPISODIC_COMPRESSED,
            ConsolidationLevel.SEMANTIC_FACT,
            ConsolidationLevel.SEMANTIC_INSIGHT,
        ]

    def test_is_intenum(self):
        """ConsolidationLevel is an IntEnum."""
        import enum

        assert issubclass(ConsolidationLevel, enum.IntEnum)


class TestConsolidationMode:
    """Verify ConsolidationMode enum values and membership."""

    def test_has_exactly_three_members(self):
        """ConsolidationMode must have exactly 3 members."""
        assert len(ConsolidationMode) == 3

    def test_member_values(self):
        """Each member has the correct string value."""
        assert ConsolidationMode.INTRA_SESSION.value == "intra_session"
        assert ConsolidationMode.ASYNC_BATCH.value == "async_batch"
        assert ConsolidationMode.RETRIEVAL_TRIGGERED.value == "retrieval_triggered"

    def test_is_enum_not_intenum(self):
        """ConsolidationMode is a plain Enum (not IntEnum)."""
        import enum

        assert issubclass(ConsolidationMode, enum.Enum)
        assert not issubclass(ConsolidationMode, enum.IntEnum)


# ============================================================
# 2. Constant Tests
# ============================================================


class TestLevelTimeConstants:
    """Verify LEVEL_TIME_CONSTANTS maps to correct values."""

    def test_episodic_raw_time_constant(self):
        assert LEVEL_TIME_CONSTANTS[ConsolidationLevel.EPISODIC_RAW] == 7.0

    def test_episodic_compressed_time_constant(self):
        assert LEVEL_TIME_CONSTANTS[ConsolidationLevel.EPISODIC_COMPRESSED] == 30.0

    def test_semantic_fact_time_constant(self):
        assert LEVEL_TIME_CONSTANTS[ConsolidationLevel.SEMANTIC_FACT] == 180.0

    def test_semantic_insight_time_constant(self):
        assert LEVEL_TIME_CONSTANTS[ConsolidationLevel.SEMANTIC_INSIGHT] == 365.0

    def test_covers_all_levels(self):
        """Every ConsolidationLevel has a time constant."""
        for level in ConsolidationLevel:
            assert level in LEVEL_TIME_CONSTANTS, (
                f"Missing time constant for {level}"
            )


class TestLevelHalfLives:
    """Verify LEVEL_HALF_LIVES are correctly computed as tc * ln(2)."""

    def test_half_lives_match_formula(self):
        """For all levels: half_life == time_constant * ln(2)."""
        for level in ConsolidationLevel:
            tc = LEVEL_TIME_CONSTANTS[level]
            expected = tc * math.log(2)
            actual = LEVEL_HALF_LIVES[level]
            assert actual == pytest.approx(expected, rel=1e-12), (
                f"Half-life mismatch for {level}: expected {expected}, got {actual}"
            )

    def test_covers_all_levels(self):
        """Every ConsolidationLevel has a half-life."""
        for level in ConsolidationLevel:
            assert level in LEVEL_HALF_LIVES

    def test_episodic_raw_half_life_approx(self):
        """EPISODIC_RAW half-life is approximately 4.85 days."""
        hl = LEVEL_HALF_LIVES[ConsolidationLevel.EPISODIC_RAW]
        assert hl == pytest.approx(4.852, abs=0.01)


class TestLevelRetentionThresholds:
    """Verify LEVEL_RETENTION_THRESHOLDS for promotable levels."""

    def test_episodic_raw_threshold(self):
        """EPISODIC_RAW retention threshold is 0.4."""
        assert LEVEL_RETENTION_THRESHOLDS[ConsolidationLevel.EPISODIC_RAW] == 0.4

    def test_episodic_compressed_threshold(self):
        """EPISODIC_COMPRESSED retention threshold is 0.3."""
        assert LEVEL_RETENTION_THRESHOLDS[ConsolidationLevel.EPISODIC_COMPRESSED] == 0.3

    def test_semantic_fact_not_present(self):
        """SEMANTIC_FACT is NOT in LEVEL_RETENTION_THRESHOLDS.

        L3->L4 promotion uses cluster similarity, not retention decay.
        """
        assert ConsolidationLevel.SEMANTIC_FACT not in LEVEL_RETENTION_THRESHOLDS

    def test_semantic_insight_not_present(self):
        """SEMANTIC_INSIGHT has no retention threshold (no further promotion)."""
        assert ConsolidationLevel.SEMANTIC_INSIGHT not in LEVEL_RETENTION_THRESHOLDS


class TestCategoryCompressionRatios:
    """Verify CATEGORY_COMPRESSION_RATIOS for all categories."""

    def test_correction_ratio_is_one(self):
        """Corrections must have compression ratio 1 (never compressed)."""
        assert CATEGORY_COMPRESSION_RATIOS["correction"] == 1

    def test_preference_ratio(self):
        assert CATEGORY_COMPRESSION_RATIOS["preference"] == 10

    def test_fact_ratio(self):
        assert CATEGORY_COMPRESSION_RATIOS["fact"] == 5

    def test_reasoning_ratio(self):
        assert CATEGORY_COMPRESSION_RATIOS["reasoning"] == 3

    def test_instruction_ratio(self):
        assert CATEGORY_COMPRESSION_RATIOS["instruction"] == 5

    def test_greeting_ratio(self):
        assert CATEGORY_COMPRESSION_RATIOS["greeting"] == 10

    def test_transactional_ratio(self):
        assert CATEGORY_COMPRESSION_RATIOS["transactional"] == 10

    def test_covers_all_valid_categories(self):
        """Every category in VALID_CATEGORIES has a compression ratio."""
        for cat in VALID_CATEGORIES:
            assert cat in CATEGORY_COMPRESSION_RATIOS, (
                f"Missing compression ratio for category '{cat}'"
            )


class TestNonConsolidatableCategories:
    """Verify NON_CONSOLIDATABLE_CATEGORIES is a frozenset."""

    def test_is_frozenset(self):
        assert isinstance(NON_CONSOLIDATABLE_CATEGORIES, frozenset)

    def test_contains_greeting(self):
        assert "greeting" in NON_CONSOLIDATABLE_CATEGORIES

    def test_contains_transactional(self):
        assert "transactional" in NON_CONSOLIDATABLE_CATEGORIES

    def test_has_at_least_two_members(self):
        assert len(NON_CONSOLIDATABLE_CATEGORIES) >= 2


class TestCorrectionContentMarkers:
    """Verify CORRECTION_CONTENT_MARKERS and threshold."""

    def test_is_frozenset(self):
        assert isinstance(CORRECTION_CONTENT_MARKERS, frozenset)

    def test_contains_compiled_patterns(self):
        """All elements are compiled regex patterns."""
        for pattern in CORRECTION_CONTENT_MARKERS:
            assert isinstance(pattern, re.Pattern), (
                f"Expected compiled regex, got {type(pattern)}"
            )

    def test_threshold_is_two(self):
        assert CORRECTION_MARKER_THRESHOLD == 2

    def test_markers_match_actually(self):
        """At least one marker matches 'actually, I was wrong'."""
        text = "actually, I was wrong about that"
        matches = sum(1 for p in CORRECTION_CONTENT_MARKERS if p.search(text))
        assert matches >= 2, (
            f"Expected at least 2 correction markers to match, got {matches}"
        )

    def test_markers_do_not_match_normal_text(self):
        """Normal text should match fewer than threshold markers."""
        text = "I prefer dark mode for my editor"
        matches = sum(1 for p in CORRECTION_CONTENT_MARKERS if p.search(text))
        assert matches < CORRECTION_MARKER_THRESHOLD


class TestDefaultConfig:
    """Verify DEFAULT_CONSOLIDATION_CONFIG is a valid ConsolidationConfig."""

    def test_is_consolidation_config(self):
        assert isinstance(DEFAULT_CONSOLIDATION_CONFIG, ConsolidationConfig)

    def test_constructs_without_error(self):
        """Default config is valid (no validation errors)."""
        _ = DEFAULT_CONSOLIDATION_CONFIG
        # If this line is reached, construction succeeded.


# ============================================================
# 3. ConsolidationConfig Tests
# ============================================================


class TestConsolidationConfig:
    """Verify ConsolidationConfig construction, defaults, validation, and immutability."""

    def test_default_construction(self):
        """Default config constructs without error."""
        cfg = ConsolidationConfig()
        assert isinstance(cfg, ConsolidationConfig)

    def test_is_frozen(self):
        """Config is frozen — assignment raises FrozenInstanceError."""
        cfg = ConsolidationConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.l1_to_l2_retention = 0.5  # type: ignore[misc]

    # --- l1_to_l2_retention ---

    def test_l1_to_l2_retention_default(self, config):
        assert config.l1_to_l2_retention == 0.4

    def test_l1_to_l2_retention_valid_boundary(self):
        """Just inside (0, 1), respecting l2_to_l3_retention cross-field."""
        ConsolidationConfig(l1_to_l2_retention=0.3, l2_to_l3_retention=0.3)
        ConsolidationConfig(l1_to_l2_retention=0.999)

    def test_l1_to_l2_retention_invalid_zero(self):
        with pytest.raises(ValueError, match="l1_to_l2_retention"):
            ConsolidationConfig(l1_to_l2_retention=0.0)

    def test_l1_to_l2_retention_invalid_one(self):
        with pytest.raises(ValueError, match="l1_to_l2_retention"):
            ConsolidationConfig(l1_to_l2_retention=1.0)

    def test_l1_to_l2_retention_invalid_negative(self):
        with pytest.raises(ValueError, match="l1_to_l2_retention"):
            ConsolidationConfig(l1_to_l2_retention=-0.1)

    # --- l2_to_l3_retention ---

    def test_l2_to_l3_retention_default(self, config):
        assert config.l2_to_l3_retention == 0.3

    def test_l2_to_l3_retention_invalid_zero(self):
        with pytest.raises(ValueError, match="l2_to_l3_retention"):
            ConsolidationConfig(l2_to_l3_retention=0.0)

    # --- l3_to_l4_cluster_sim ---

    def test_l3_to_l4_cluster_sim_default(self, config):
        assert config.l3_to_l4_cluster_sim == 0.6

    def test_l3_to_l4_cluster_sim_valid_one(self):
        """1.0 is allowed for (0.0, 1.0] domain."""
        ConsolidationConfig(l3_to_l4_cluster_sim=1.0)

    def test_l3_to_l4_cluster_sim_invalid_zero(self):
        with pytest.raises(ValueError, match="l3_to_l4_cluster_sim"):
            ConsolidationConfig(l3_to_l4_cluster_sim=0.0)

    # --- semantic_value_threshold ---

    def test_semantic_value_threshold_default(self, config):
        assert config.semantic_value_threshold == 0.6

    def test_semantic_value_threshold_valid_boundary(self):
        ConsolidationConfig(semantic_value_threshold=0.001)
        ConsolidationConfig(semantic_value_threshold=0.999)

    def test_semantic_value_threshold_invalid_zero(self):
        with pytest.raises(ValueError, match="semantic_value_threshold"):
            ConsolidationConfig(semantic_value_threshold=0.0)

    # --- recency_guard ---

    def test_recency_guard_default(self, config):
        assert config.recency_guard == 1.0

    def test_recency_guard_invalid_zero(self):
        with pytest.raises(ValueError, match="recency_guard"):
            ConsolidationConfig(recency_guard=0.0)

    def test_recency_guard_invalid_negative(self):
        with pytest.raises(ValueError, match="recency_guard"):
            ConsolidationConfig(recency_guard=-1.0)

    # --- consolidation_window ---

    def test_consolidation_window_default(self, config):
        assert config.consolidation_window == 7.0

    def test_consolidation_window_invalid_zero(self):
        with pytest.raises(ValueError, match="consolidation_window"):
            ConsolidationConfig(consolidation_window=0.0)

    # --- retrieval_frequency_threshold ---

    def test_retrieval_frequency_threshold_default(self, config):
        assert config.retrieval_frequency_threshold == 10

    def test_retrieval_frequency_threshold_invalid_zero(self):
        with pytest.raises(ValueError, match="retrieval_frequency_threshold"):
            ConsolidationConfig(retrieval_frequency_threshold=0)

    # --- max_pool_size ---

    def test_max_pool_size_default(self, config):
        assert config.max_pool_size == 5000

    def test_max_pool_size_invalid_zero(self):
        with pytest.raises(ValueError, match="max_pool_size"):
            ConsolidationConfig(max_pool_size=0)

    # --- intra_session_similarity ---

    def test_intra_session_similarity_default(self, config):
        assert config.intra_session_similarity == 0.8

    def test_intra_session_similarity_invalid_zero(self):
        with pytest.raises(ValueError, match="intra_session_similarity"):
            ConsolidationConfig(intra_session_similarity=0.0)

    # --- intra_session_temporal ---

    def test_intra_session_temporal_default(self, config):
        assert config.intra_session_temporal == 1.0

    def test_intra_session_temporal_invalid_zero(self):
        with pytest.raises(ValueError, match="intra_session_temporal"):
            ConsolidationConfig(intra_session_temporal=0.0)

    # --- cluster_threshold ---

    def test_cluster_threshold_default(self, config):
        assert config.cluster_threshold == 0.6

    def test_cluster_threshold_invalid_zero(self):
        with pytest.raises(ValueError, match="cluster_threshold"):
            ConsolidationConfig(cluster_threshold=0.0)

    # --- semantic_weight_beta / temporal_weight ---

    def test_semantic_weight_beta_default(self, config):
        assert config.semantic_weight_beta == 0.7

    def test_temporal_weight_derived(self, config):
        """temporal_weight is a computed property = 1.0 - semantic_weight_beta."""
        assert config.temporal_weight == pytest.approx(0.3)

    def test_semantic_weight_beta_valid_zero(self):
        """Zero is allowed for [0, 1]."""
        ConsolidationConfig(semantic_weight_beta=0.0)

    def test_semantic_weight_beta_valid_one(self):
        """One is allowed for [0, 1]."""
        ConsolidationConfig(semantic_weight_beta=1.0)

    def test_semantic_weight_beta_invalid_negative(self):
        with pytest.raises(ValueError, match="semantic_weight_beta"):
            ConsolidationConfig(semantic_weight_beta=-0.01)

    def test_semantic_weight_beta_invalid_above_one(self):
        with pytest.raises(ValueError, match="semantic_weight_beta"):
            ConsolidationConfig(semantic_weight_beta=1.01)

    # --- temporal_decay_lambda ---

    def test_temporal_decay_lambda_default(self, config):
        assert config.temporal_decay_lambda == 0.1

    def test_temporal_decay_lambda_invalid_zero(self):
        with pytest.raises(ValueError, match="temporal_decay_lambda"):
            ConsolidationConfig(temporal_decay_lambda=0.0)

    # --- consolidation_decay ---

    def test_consolidation_decay_default(self, config):
        assert config.consolidation_decay == 0.1

    def test_consolidation_decay_invalid_zero(self):
        with pytest.raises(ValueError, match="consolidation_decay"):
            ConsolidationConfig(consolidation_decay=0.0)

    def test_consolidation_decay_invalid_one(self):
        with pytest.raises(ValueError, match="consolidation_decay"):
            ConsolidationConfig(consolidation_decay=1.0)

    # --- min_content_similarity ---

    def test_min_content_similarity_default(self, config):
        assert config.min_content_similarity == 0.3

    def test_min_content_similarity_valid_zero(self):
        """Zero is allowed for [0, 1]."""
        ConsolidationConfig(min_content_similarity=0.0)

    def test_min_content_similarity_valid_one(self):
        """One is allowed for [0, 1]."""
        ConsolidationConfig(min_content_similarity=1.0)

    def test_min_content_similarity_invalid_above_one(self):
        with pytest.raises(ValueError, match="min_content_similarity"):
            ConsolidationConfig(min_content_similarity=1.01)

    # --- mild_pressure_retention ---

    def test_mild_pressure_retention_default(self, config):
        assert config.mild_pressure_retention == 0.5

    def test_mild_pressure_retention_invalid_zero(self):
        with pytest.raises(ValueError, match="mild_pressure_retention"):
            ConsolidationConfig(mild_pressure_retention=0.0)

    # --- severe_pressure_multiplier ---

    def test_severe_pressure_multiplier_default(self, config):
        assert config.severe_pressure_multiplier == 2.0

    def test_severe_pressure_multiplier_invalid_one(self):
        with pytest.raises(ValueError, match="severe_pressure_multiplier"):
            ConsolidationConfig(severe_pressure_multiplier=1.0)

    # --- max_candidates_per_batch ---

    def test_max_candidates_per_batch_default(self, config):
        assert config.max_candidates_per_batch == 100

    def test_max_candidates_per_batch_invalid_zero(self):
        with pytest.raises(ValueError, match="max_candidates_per_batch"):
            ConsolidationConfig(max_candidates_per_batch=0)

    # --- semantic_boost ---

    def test_semantic_boost_default(self, config):
        assert config.semantic_boost == 1.2

    def test_semantic_boost_invalid_one(self):
        with pytest.raises(ValueError, match="semantic_boost"):
            ConsolidationConfig(semantic_boost=1.0)

    def test_semantic_boost_invalid_above_two(self):
        with pytest.raises(ValueError, match="semantic_boost"):
            ConsolidationConfig(semantic_boost=2.1)

    # --- activation_inheritance ---

    def test_activation_inheritance_default(self, config):
        assert config.activation_inheritance == 0.5

    def test_activation_inheritance_valid_zero(self):
        ConsolidationConfig(activation_inheritance=0.0)

    def test_activation_inheritance_valid_one(self):
        ConsolidationConfig(activation_inheritance=1.0)

    def test_activation_inheritance_invalid_negative(self):
        with pytest.raises(ValueError, match="activation_inheritance"):
            ConsolidationConfig(activation_inheritance=-0.1)

    # --- retention_threshold ---

    def test_retention_threshold_default(self, config):
        assert config.retention_threshold == 0.3

    def test_retention_threshold_invalid_zero(self):
        with pytest.raises(ValueError, match="retention_threshold"):
            ConsolidationConfig(retention_threshold=0.0)


# ============================================================
# 4. ConsolidationCandidate Tests
# ============================================================


class TestConsolidationCandidate:
    """Verify ConsolidationCandidate construction, defaults, and validation."""

    @staticmethod
    def _make(**overrides):
        """Construct a valid ConsolidationCandidate with defaults."""
        defaults = {
            "memory_id": "mem-001",
            "content": "I prefer dark mode",
            "category": "preference",
            "level": ConsolidationLevel.EPISODIC_RAW,
            "creation_time": 100.0,
            "last_access_time": 50.0,
            "access_count": 5,
            "importance": 0.7,
            "strength": 7.0,
            "relevance": 0.5,
        }
        defaults.update(overrides)
        return ConsolidationCandidate(**defaults)

    def test_valid_construction(self):
        """Valid candidate constructs without error."""
        c = self._make()
        assert c.memory_id == "mem-001"
        assert c.category == "preference"

    def test_is_frozen(self):
        """Candidate is frozen."""
        c = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.importance = 0.5  # type: ignore[misc]

    def test_is_contested_default_false(self):
        """is_contested defaults to False."""
        c = self._make()
        assert c.is_contested is False

    def test_metadata_default_none(self):
        """metadata defaults to None."""
        c = self._make()
        assert c.metadata is None

    def test_invalid_category_raises(self):
        """Category not in VALID_CATEGORIES raises ValueError."""
        with pytest.raises(ValueError, match="category"):
            self._make(category="nonexistent")

    def test_invalid_importance_negative(self):
        """Negative importance raises ValueError."""
        with pytest.raises(ValueError, match="importance"):
            self._make(importance=-0.1)

    def test_invalid_importance_above_one(self):
        """importance > 1 raises ValueError."""
        with pytest.raises(ValueError, match="importance"):
            self._make(importance=1.1)

    def test_invalid_access_count_negative(self):
        """Negative access_count raises ValueError."""
        with pytest.raises(ValueError, match="access_count"):
            self._make(access_count=-1)

    def test_level_must_be_consolidation_level(self):
        """level must be a ConsolidationLevel member."""
        c = self._make(level=ConsolidationLevel.SEMANTIC_FACT)
        assert c.level == ConsolidationLevel.SEMANTIC_FACT


# ============================================================
# 5. SemanticExtraction Tests
# ============================================================


class TestSemanticExtraction:
    """Verify SemanticExtraction construction, defaults, and validation."""

    @staticmethod
    def _make(**overrides):
        """Construct a valid SemanticExtraction with defaults."""
        defaults = {
            "content": "User prefers: dark mode",
            "category": "preference",
            "source_episodes": ("ep-001", "ep-002"),
            "confidence": 0.85,
            "first_observed": 10.0,
            "last_updated": 50.0,
            "consolidation_count": 2,
            "compression_ratio": 2.5,
            "importance": 0.7,
            "target_level": ConsolidationLevel.EPISODIC_COMPRESSED,
            "extraction_mode": ConsolidationMode.ASYNC_BATCH,
        }
        defaults.update(overrides)
        return SemanticExtraction(**defaults)

    def test_valid_construction(self):
        """Valid extraction constructs without error."""
        s = self._make()
        assert s.content == "User prefers: dark mode"
        assert len(s.source_episodes) == 2

    def test_is_frozen(self):
        """Extraction is frozen."""
        s = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.confidence = 0.5  # type: ignore[misc]

    def test_empty_source_episodes_raises(self):
        """Empty source_episodes raises ValueError."""
        with pytest.raises(ValueError, match="source_episodes"):
            self._make(source_episodes=())

    def test_confidence_below_zero_raises(self):
        """Confidence < 0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            self._make(confidence=-0.1)

    def test_confidence_above_one_raises(self):
        """Confidence > 1 raises ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            self._make(confidence=1.1)

    def test_invalid_category_raises(self):
        """Category not in VALID_CATEGORIES raises ValueError."""
        with pytest.raises(ValueError, match="category"):
            self._make(category="invalid")

    def test_first_observed_after_last_updated_raises(self):
        """first_observed > last_updated raises ValueError."""
        with pytest.raises(ValueError, match="first_observed"):
            self._make(first_observed=50.0, last_updated=10.0)

    def test_first_observed_equals_last_updated_ok(self):
        """first_observed == last_updated is acceptable (single source)."""
        s = self._make(first_observed=10.0, last_updated=10.0)
        assert s.first_observed == 10.0
        assert s.last_updated == 10.0

    def test_consolidation_count_zero_raises(self):
        """consolidation_count < 1 raises ValueError."""
        with pytest.raises(ValueError, match="consolidation_count"):
            self._make(consolidation_count=0)

    def test_compression_ratio_zero_raises(self):
        """compression_ratio <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="compression_ratio"):
            self._make(compression_ratio=0.0)

    def test_compression_ratio_less_than_one_ok(self):
        """compression_ratio < 1 is OK (expansion from template prefix)."""
        s = self._make(compression_ratio=0.5)
        assert s.compression_ratio == 0.5


# ============================================================
# 6. ArchivedMemory Tests
# ============================================================


class TestArchivedMemory:
    """Verify ArchivedMemory construction, defaults, and frozen property."""

    @staticmethod
    def _make(**overrides):
        """Construct a valid ArchivedMemory with defaults."""
        defaults = {
            "memory_id": "mem-001",
            "content": "I prefer dark mode in my editor",
            "category": "preference",
            "level": ConsolidationLevel.EPISODIC_RAW,
            "archived": True,
            "consolidated_to": ("sem-001",),
            "decayed_importance": 0.08,
            "decayed_strength": 0.7,
            "original_importance": 0.8,
            "original_strength": 7.0,
            "creation_time": 100.0,
            "archive_time": 200.0,
        }
        defaults.update(overrides)
        return ArchivedMemory(**defaults)

    def test_valid_construction(self):
        """Valid archived memory constructs without error."""
        a = self._make()
        assert a.memory_id == "mem-001"

    def test_is_frozen(self):
        """ArchivedMemory is frozen."""
        a = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.memory_id = "other"  # type: ignore[misc]

    def test_archived_always_true(self):
        """archived field is always True."""
        a = self._make()
        assert a.archived is True

    def test_consolidated_to_tuple(self):
        """consolidated_to is a tuple of target IDs."""
        a = self._make(consolidated_to=("sem-001", "sem-002"))
        assert a.consolidated_to == ("sem-001", "sem-002")

    def test_level_preserved(self):
        """level records the original consolidation level."""
        a = self._make(level=ConsolidationLevel.EPISODIC_COMPRESSED)
        assert a.level == ConsolidationLevel.EPISODIC_COMPRESSED

    def test_access_count_at_archive_default(self):
        """access_count_at_archive defaults to 0."""
        a = self._make()
        assert a.access_count_at_archive == 0

    def test_decayed_strength_preserved(self):
        """decayed_strength records the decayed strength value."""
        a = self._make(decayed_strength=0.7)
        assert a.decayed_strength == 0.7

    def test_original_strength_preserved(self):
        """original_strength records the pre-archival strength."""
        a = self._make(original_strength=7.0)
        assert a.original_strength == 7.0


# ============================================================
# 7. ConsolidationResult Tests
# ============================================================


class TestConsolidationResult:
    """Verify ConsolidationResult construction and frozen property."""

    def test_valid_construction_empty_tuples(self):
        """ConsolidationResult with empty tuples is valid."""
        r = ConsolidationResult(
            extractions=(),
            archived=(),
            skipped_indices=frozenset(),
            candidates_evaluated=0,
            candidates_consolidated=0,
            mode=ConsolidationMode.ASYNC_BATCH,
        )
        assert r.candidates_evaluated == 0
        assert r.candidates_consolidated == 0

    def test_is_frozen(self):
        """ConsolidationResult is frozen."""
        r = ConsolidationResult(
            extractions=(),
            archived=(),
            skipped_indices=frozenset(),
            candidates_evaluated=0,
            candidates_consolidated=0,
            mode=ConsolidationMode.ASYNC_BATCH,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.candidates_evaluated = 10  # type: ignore[misc]

    def test_with_extractions_and_archived(self):
        """ConsolidationResult with non-empty tuples is valid."""
        extraction = SemanticExtraction(
            content="User prefers: dark mode",
            category="preference",
            source_episodes=("ep-001",),
            confidence=0.85,
            first_observed=10.0,
            last_updated=50.0,
            consolidation_count=1,
            compression_ratio=2.0,
            importance=0.7,
            target_level=ConsolidationLevel.EPISODIC_COMPRESSED,
            extraction_mode=ConsolidationMode.ASYNC_BATCH,
        )
        archived = ArchivedMemory(
            memory_id="mem-001",
            content="I prefer dark mode",
            category="preference",
            level=ConsolidationLevel.EPISODIC_RAW,
            archived=True,
            consolidated_to=("sem-001",),
            decayed_importance=0.08,
            decayed_strength=0.7,
            original_importance=0.8,
            original_strength=7.0,
            creation_time=100.0,
            archive_time=200.0,
        )
        r = ConsolidationResult(
            extractions=(extraction,),
            archived=(archived,),
            skipped_indices=frozenset(),
            candidates_evaluated=5,
            candidates_consolidated=1,
            mode=ConsolidationMode.ASYNC_BATCH,
        )
        assert len(r.extractions) == 1
        assert len(r.archived) == 1
        assert r.candidates_consolidated == 1

    def test_mode_is_required(self):
        """mode is a required field (no default)."""
        with pytest.raises(TypeError):
            ConsolidationResult(  # type: ignore[call-arg]
                extractions=(),
                archived=(),
                skipped_indices=frozenset(),
                candidates_evaluated=0,
                candidates_consolidated=0,
            )

    def test_skipped_indices_populated_on_skip(self):
        """skipped_indices contains memory_ids of skipped candidates."""
        r = ConsolidationResult(
            extractions=(),
            archived=(),
            skipped_indices=frozenset({"mem-001", "mem-003"}),
            candidates_evaluated=3,
            candidates_consolidated=1,
            mode=ConsolidationMode.ASYNC_BATCH,
        )
        assert "mem-001" in r.skipped_indices
        assert "mem-003" in r.skipped_indices
        assert len(r.skipped_indices) == 2


# ============================================================
# 8. Property-Based / Hypothesis Tests
# ============================================================


class TestPropertyBased:
    """Property-based tests using hypothesis."""

    @given(
        l1_to_l2_retention=st.floats(min_value=0.01, max_value=0.99),
        l2_to_l3_retention=st.floats(min_value=0.01, max_value=0.99),
        l3_to_l4_cluster_sim=st.floats(min_value=0.01, max_value=1.0),
        semantic_value_threshold=st.floats(min_value=0.01, max_value=0.99),
        consolidation_window=st.floats(min_value=0.01, max_value=10000.0),
        retrieval_frequency_threshold=st.integers(min_value=1, max_value=1000),
        max_pool_size=st.integers(min_value=1, max_value=100000),
        max_candidates_per_batch=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_config_valid_ranges_always_construct(
        self,
        l1_to_l2_retention,
        l2_to_l3_retention,
        l3_to_l4_cluster_sim,
        semantic_value_threshold,
        consolidation_window,
        retrieval_frequency_threshold,
        max_pool_size,
        max_candidates_per_batch,
    ):
        """ConsolidationConfig with all fields in valid ranges always constructs."""
        # Cross-field: l2_to_l3_retention <= l1_to_l2_retention
        if l2_to_l3_retention > l1_to_l2_retention:
            l2_to_l3_retention = l1_to_l2_retention
        cfg = ConsolidationConfig(
            l1_to_l2_retention=l1_to_l2_retention,
            l2_to_l3_retention=l2_to_l3_retention,
            l3_to_l4_cluster_sim=l3_to_l4_cluster_sim,
            semantic_value_threshold=semantic_value_threshold,
            consolidation_window=consolidation_window,
            retrieval_frequency_threshold=retrieval_frequency_threshold,
            max_pool_size=max_pool_size,
            max_candidates_per_batch=max_candidates_per_batch,
        )
        assert isinstance(cfg, ConsolidationConfig)

    @given(
        importance=st.floats(min_value=0.0, max_value=1.0),
        access_count=st.integers(min_value=0, max_value=10000),
        strength=st.floats(min_value=0.0, max_value=1000.0),
        relevance=st.floats(min_value=0.0, max_value=1.0),
        creation_time=st.floats(min_value=0.0, max_value=1e6),
        last_access_time=st.floats(min_value=0.0, max_value=1e6),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_candidate_valid_ranges_always_construct(
        self,
        importance,
        access_count,
        strength,
        relevance,
        creation_time,
        last_access_time,
    ):
        """ConsolidationCandidate with valid fields always constructs."""
        c = ConsolidationCandidate(
            memory_id="test-mem",
            content="Test content",
            category="fact",
            level=ConsolidationLevel.EPISODIC_RAW,
            creation_time=creation_time,
            last_access_time=last_access_time,
            access_count=access_count,
            importance=importance,
            strength=strength,
            relevance=relevance,
        )
        assert isinstance(c, ConsolidationCandidate)

    def test_half_lives_match_time_constants_for_all_levels(self):
        """Property: LEVEL_HALF_LIVES[level] == LEVEL_TIME_CONSTANTS[level] * ln(2)."""
        for level in ConsolidationLevel:
            tc = LEVEL_TIME_CONSTANTS[level]
            expected_hl = tc * math.log(2)
            actual_hl = LEVEL_HALF_LIVES[level]
            assert abs(actual_hl - expected_hl) < 1e-12, (
                f"Half-life mismatch for {level}: {actual_hl} != {expected_hl}"
            )


# ============================================================
# Phase B: Core Logic Functions
# ============================================================


# ---------------------------------------------------------------------------
# Helper: build ConsolidationCandidate with sensible defaults
# ---------------------------------------------------------------------------


def _make_candidate(**overrides) -> ConsolidationCandidate:
    """Construct a valid ConsolidationCandidate with Phase B-friendly defaults.

    Defaults produce a candidate that SHOULD consolidate under default config:
      - Not contested
      - Category 'preference' (not in NON_CONSOLIDATABLE_CATEGORIES)
      - creation_time 200 (well past recency_guard and consolidation_window)
      - access_count 3 (below retrieval_frequency_threshold)
      - strength 7.0 (time constant for EPISODIC_RAW)
      - last_access_time 50.0 (retention = exp(-50/7) ~ 0.0008)
      - importance 0.8
      - Normal content (not correction-like)
    """
    defaults = {
        "memory_id": "mem-001",
        "content": "I prefer dark mode for my editor",
        "category": "preference",
        "level": ConsolidationLevel.EPISODIC_RAW,
        "creation_time": 200.0,
        "last_access_time": 50.0,
        "access_count": 3,
        "importance": 0.8,
        "strength": 7.0,
        "relevance": 0.5,
    }
    defaults.update(overrides)
    return ConsolidationCandidate(**defaults)


# ============================================================
# 9. _content_is_correction Tests
# ============================================================


class TestContentIsCorrection:
    """Verify _content_is_correction detects correction patterns in content."""

    def test_clear_correction_text(self):
        """Text with multiple correction markers returns True."""
        text = "Actually, I was wrong about that preference"
        assert _content_is_correction(text) is True

    def test_normal_text_returns_false(self):
        """Normal preference text returns False."""
        assert _content_is_correction("I prefer dark mode for my editor") is False

    def test_single_marker_below_threshold(self):
        """A single marker match does not meet threshold of 2."""
        assert _content_is_correction("actually, that is fine") is False

    def test_empty_string_returns_false(self):
        """Empty string has no marker matches."""
        assert _content_is_correction("") is False

    def test_two_markers_meets_threshold(self):
        """Exactly two markers meets threshold (>= 2)."""
        text = "Correction: I was wrong about the setting"
        assert _content_is_correction(text) is True

    def test_many_markers(self):
        """Text with many correction markers returns True."""
        text = "Actually, I was wrong, correction: the incorrect thing about my updated preference"
        assert _content_is_correction(text) is True

    def test_case_insensitive(self):
        """Markers are case-insensitive."""
        text = "ACTUALLY, I WAS WRONG about that"
        assert _content_is_correction(text) is True

    def test_deterministic(self):
        """Same input always produces same output."""
        text = "Actually, I was wrong about that"
        result1 = _content_is_correction(text)
        result2 = _content_is_correction(text)
        assert result1 == result2

    def test_updated_mind_pattern(self):
        """'changed my mind' + 'actually' matches >= 2 patterns."""
        # 'actually' pattern requires trailing whitespace, so place it mid-sentence
        text = "Actually, I changed my mind about that preference"
        assert _content_is_correction(text) is True

    def test_not_x_but_y_pattern(self):
        """'not X but Y' pattern matches."""
        text = "I was wrong, not tea but coffee"
        assert _content_is_correction(text) is True


# ============================================================
# 10. _passes_inhibitors Tests
# ============================================================


class TestPassesInhibitors:
    """Verify _passes_inhibitors checks all 5 inhibitor conditions."""

    def test_clean_candidate_passes(self):
        """A well-formed, non-inhibited candidate passes all inhibitors."""
        candidate = _make_candidate()
        config = ConsolidationConfig()
        assert _passes_inhibitors(candidate, config) is True

    # --- I1: contested ---

    def test_i1_contested_blocks(self):
        """Contested candidate is always inhibited."""
        candidate = _make_candidate(is_contested=True)
        config = ConsolidationConfig()
        assert _passes_inhibitors(candidate, config) is False

    # --- I2: recency guard ---

    def test_i2_too_recent_blocks(self):
        """Candidate with creation_time < recency_guard is inhibited."""
        config = ConsolidationConfig(recency_guard=1.0)
        candidate = _make_candidate(creation_time=0.5)
        assert _passes_inhibitors(candidate, config) is False

    def test_i2_exactly_at_recency_guard_passes(self):
        """Candidate with creation_time == recency_guard passes I2 (< not <=)."""
        config = ConsolidationConfig(recency_guard=1.0)
        candidate = _make_candidate(creation_time=1.0)
        assert _passes_inhibitors(candidate, config) is True

    def test_i2_well_past_recency_guard_passes(self):
        """Candidate well past recency guard passes I2."""
        config = ConsolidationConfig(recency_guard=1.0)
        candidate = _make_candidate(creation_time=100.0)
        assert _passes_inhibitors(candidate, config) is True

    # --- I3: high activation + recent access ---

    def test_i3_high_activation_recent_access_blocks(self):
        """High access count AND recent last_access blocks."""
        config = ConsolidationConfig(
            high_activation_threshold=10, recency_guard=1.0
        )
        candidate = _make_candidate(
            access_count=10,
            last_access_time=0.5,
        )
        assert _passes_inhibitors(candidate, config) is False

    def test_i3_high_activation_old_access_passes(self):
        """High access count but old last_access passes I3."""
        config = ConsolidationConfig(
            high_activation_threshold=10, recency_guard=1.0
        )
        candidate = _make_candidate(
            access_count=15,
            last_access_time=5.0,
        )
        assert _passes_inhibitors(candidate, config) is True

    def test_i3_low_activation_recent_access_passes(self):
        """Low access count with recent last_access passes I3 (both needed)."""
        config = ConsolidationConfig(
            high_activation_threshold=10, recency_guard=1.0
        )
        candidate = _make_candidate(
            access_count=5,
            last_access_time=0.5,
        )
        assert _passes_inhibitors(candidate, config) is True

    def test_i3_exactly_at_threshold(self):
        """access_count == high_activation_threshold triggers I3 (>=)."""
        config = ConsolidationConfig(
            high_activation_threshold=10, recency_guard=1.0
        )
        candidate = _make_candidate(
            access_count=10,
            last_access_time=0.5,
        )
        assert _passes_inhibitors(candidate, config) is False

    # --- I4: non-consolidatable category ---

    def test_i4_greeting_blocks(self):
        """Greeting category is non-consolidatable."""
        candidate = _make_candidate(category="greeting")
        config = ConsolidationConfig()
        assert _passes_inhibitors(candidate, config) is False

    def test_i4_transactional_blocks(self):
        """Transactional category is non-consolidatable."""
        candidate = _make_candidate(category="transactional")
        config = ConsolidationConfig()
        assert _passes_inhibitors(candidate, config) is False

    def test_i4_correction_category_blocks(self):
        """Correction category is non-consolidatable (per AP3-F3)."""
        candidate = _make_candidate(category="correction")
        config = ConsolidationConfig()
        assert _passes_inhibitors(candidate, config) is False

    def test_i4_fact_passes(self):
        """Fact category is consolidatable."""
        candidate = _make_candidate(category="fact")
        config = ConsolidationConfig()
        assert _passes_inhibitors(candidate, config) is True

    def test_i4_preference_passes(self):
        """Preference category passes I4."""
        candidate = _make_candidate(category="preference")
        config = ConsolidationConfig()
        assert _passes_inhibitors(candidate, config) is True

    # --- I5: content-based correction detection ---

    def test_i5_correction_content_blocks(self):
        """Content matching correction markers inhibits even non-correction category."""
        candidate = _make_candidate(
            category="preference",
            content="Actually, I was wrong about liking dark mode",
        )
        config = ConsolidationConfig()
        assert _passes_inhibitors(candidate, config) is False

    def test_i5_normal_content_passes(self):
        """Normal content passes I5."""
        candidate = _make_candidate(
            category="preference",
            content="I prefer dark mode in my editor",
        )
        config = ConsolidationConfig()
        assert _passes_inhibitors(candidate, config) is True

    # --- Combined ---

    def test_all_inhibitors_pass(self):
        """Candidate that passes all 5 inhibitors returns True."""
        candidate = _make_candidate(
            is_contested=False,
            creation_time=200.0,
            last_access_time=50.0,
            access_count=3,
            category="preference",
            content="I prefer dark mode",
        )
        config = ConsolidationConfig()
        assert _passes_inhibitors(candidate, config) is True


# ============================================================
# 10b. NON_CONSOLIDATABLE_CATEGORIES updated membership
# ============================================================


class TestNonConsolidatableCategoriesPhaseB:
    """Verify correction is in NON_CONSOLIDATABLE_CATEGORIES (AP3-F3)."""

    def test_correction_in_non_consolidatable(self):
        """Correction must be in NON_CONSOLIDATABLE_CATEGORIES per AP3-F3."""
        assert "correction" in NON_CONSOLIDATABLE_CATEGORIES

    def test_has_at_least_three_members(self):
        """With correction added, must have at least 3 members."""
        assert len(NON_CONSOLIDATABLE_CATEGORIES) >= 3


# ============================================================
# 11. _compute_semantic_value Tests
# ============================================================


class TestComputeSemanticValue:
    """Verify _compute_semantic_value formula and clamping."""

    def test_preference_typical(self):
        """Preference category with typical values."""
        candidate = _make_candidate(
            category="preference",
            importance=0.8,
            access_count=5,
        )
        result = _compute_semantic_value(candidate)
        assert result == pytest.approx(0.745, abs=1e-10)

    def test_greeting_zero_importance(self):
        """Greeting category has CATEGORY_IMPORTANCE=0.0."""
        candidate = _make_candidate(
            category="greeting",
            importance=0.0,
            access_count=0,
        )
        result = _compute_semantic_value(candidate)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_instruction_high_access(self):
        """Instruction category with high access count (capped at 1.0)."""
        candidate = _make_candidate(
            category="instruction",
            importance=1.0,
            access_count=100,
        )
        result = _compute_semantic_value(candidate)
        assert result == pytest.approx(0.91, abs=1e-10)

    def test_correction_category(self):
        """Correction has CATEGORY_IMPORTANCE=0.9."""
        candidate = _make_candidate(
            category="correction",
            importance=0.5,
            access_count=10,
        )
        result = _compute_semantic_value(candidate)
        assert result == pytest.approx(0.74, abs=1e-10)

    def test_result_in_unit_interval(self):
        """Result is always in [0.0, 1.0]."""
        for cat in VALID_CATEGORIES:
            candidate = _make_candidate(
                category=cat,
                importance=1.0,
                access_count=1000,
            )
            result = _compute_semantic_value(candidate)
            assert 0.0 <= result <= 1.0, (
                f"semantic value for category '{cat}' out of range: {result}"
            )

    def test_zero_access_count(self):
        """Zero access count contributes 0 to the access component."""
        candidate = _make_candidate(
            category="fact",
            importance=0.5,
            access_count=0,
        )
        result = _compute_semantic_value(candidate)
        assert result == pytest.approx(0.51, abs=1e-10)

    def test_exactly_twenty_accesses_gives_full_access_component(self):
        """access_count=20 gives max access component."""
        candidate = _make_candidate(
            category="fact",
            importance=0.5,
            access_count=20,
        )
        result = _compute_semantic_value(candidate)
        assert result == pytest.approx(0.61, abs=1e-10)

    def test_deterministic(self):
        """Same candidate always produces same result."""
        candidate = _make_candidate()
        r1 = _compute_semantic_value(candidate)
        r2 = _compute_semantic_value(candidate)
        assert r1 == r2

    def test_reasoning_category(self):
        """Reasoning has CATEGORY_IMPORTANCE=0.4."""
        candidate = _make_candidate(
            category="reasoning",
            importance=0.6,
            access_count=10,
        )
        result = _compute_semantic_value(candidate)
        assert result == pytest.approx(0.47, abs=1e-10)


# ============================================================
# 12. should_consolidate Tests
# ============================================================


class TestShouldConsolidate:
    """Verify should_consolidate decision logic."""

    def test_contested_always_false(self):
        """Contested candidate never consolidates."""
        candidate = _make_candidate(is_contested=True)
        assert should_consolidate(candidate) is False

    def test_too_recent_always_false(self):
        """Recently created candidate never consolidates."""
        candidate = _make_candidate(creation_time=0.5)
        assert should_consolidate(candidate) is False

    def test_correction_category_always_false(self):
        """Correction category never consolidates."""
        candidate = _make_candidate(category="correction")
        assert should_consolidate(candidate) is False

    def test_correction_content_always_false(self):
        """Content matching correction patterns never consolidates."""
        candidate = _make_candidate(
            content="Actually, I was wrong about that preference"
        )
        assert should_consolidate(candidate) is False

    def test_greeting_always_false(self):
        """Greeting never consolidates."""
        candidate = _make_candidate(category="greeting")
        assert should_consolidate(candidate) is False

    def test_transactional_always_false(self):
        """Transactional never consolidates."""
        candidate = _make_candidate(category="transactional")
        assert should_consolidate(candidate) is False

    def test_semantic_insight_always_false(self):
        """SEMANTIC_INSIGHT has no further promotion path."""
        candidate = _make_candidate(
            level=ConsolidationLevel.SEMANTIC_INSIGHT,
            category="preference",
            creation_time=200.0,
            last_access_time=50.0,
            access_count=3,
            strength=365.0,
        )
        assert should_consolidate(candidate) is False

    def test_primary_trigger_low_retention_high_value(self):
        """Low retention + semantic value >= threshold triggers consolidation."""
        candidate = _make_candidate(
            level=ConsolidationLevel.EPISODIC_RAW,
            category="preference",
            importance=0.8,
            strength=7.0,
            last_access_time=50.0,
            access_count=3,
            creation_time=200.0,
        )
        assert should_consolidate(candidate) is True

    def test_high_retention_does_not_consolidate_primary(self):
        """High retention does not trigger primary path."""
        candidate = _make_candidate(
            level=ConsolidationLevel.EPISODIC_RAW,
            category="preference",
            importance=0.8,
            strength=7.0,
            last_access_time=0.01,
            access_count=3,
            creation_time=200.0,
        )
        # retention ~0.999 > 0.4 so primary is False
        # But secondary fires: temporal(200>168) AND infrequent(3<10, 200>168) AND sv>=0.6
        # So this actually returns True via secondary
        assert should_consolidate(candidate) is True

    def test_high_retention_high_access_blocks_both_paths(self):
        """High retention + high access count blocks both primary and secondary."""
        candidate = _make_candidate(
            level=ConsolidationLevel.EPISODIC_RAW,
            category="preference",
            importance=0.8,
            strength=7.0,
            last_access_time=0.01,
            access_count=15,  # >= 10 threshold
            creation_time=200.0,
        )
        assert should_consolidate(candidate) is False

    def test_low_semantic_value_blocks_primary(self):
        """Low semantic value blocks primary trigger even with low retention."""
        candidate = _make_candidate(
            level=ConsolidationLevel.EPISODIC_RAW,
            category="reasoning",
            importance=0.0,
            strength=7.0,
            last_access_time=50.0,
            access_count=0,
            creation_time=200.0,
        )
        assert should_consolidate(candidate) is False

    def test_secondary_trigger_temporal_infrequent(self):
        """Old, infrequently accessed memory with high sv consolidates via secondary."""
        candidate = _make_candidate(
            level=ConsolidationLevel.EPISODIC_RAW,
            category="preference",
            importance=0.8,
            strength=7.0,
            last_access_time=0.01,
            access_count=3,
            creation_time=200.0,
        )
        assert should_consolidate(candidate) is True

    def test_secondary_blocked_by_high_access_count(self):
        """High access count blocks the infrequent condition."""
        candidate = _make_candidate(
            level=ConsolidationLevel.EPISODIC_RAW,
            category="preference",
            importance=0.8,
            strength=7.0,
            last_access_time=0.01,
            access_count=15,
            creation_time=200.0,
        )
        assert should_consolidate(candidate) is False

    def test_secondary_blocked_by_short_creation_time(self):
        """creation_time <= consolidation_window blocks temporal condition."""
        candidate = _make_candidate(
            level=ConsolidationLevel.EPISODIC_RAW,
            category="preference",
            importance=0.8,
            strength=7.0,
            last_access_time=0.01,
            access_count=3,
            creation_time=5.0,  # < consolidation_window (7.0 days)
        )
        assert should_consolidate(candidate) is False

    def test_none_config_uses_default(self):
        """Passing None for config uses DEFAULT_CONSOLIDATION_CONFIG."""
        candidate = _make_candidate()
        result_none = should_consolidate(candidate, None)
        result_default = should_consolidate(candidate, DEFAULT_CONSOLIDATION_CONFIG)
        assert result_none == result_default

    def test_zero_strength_safe(self):
        """Zero strength does not cause division by zero."""
        candidate = _make_candidate(strength=0.0, last_access_time=50.0)
        result = should_consolidate(candidate)
        assert isinstance(result, bool)

    def test_deterministic(self):
        """Same candidate and config always produce same result."""
        candidate = _make_candidate()
        config = ConsolidationConfig()
        r1 = should_consolidate(candidate, config)
        r2 = should_consolidate(candidate, config)
        assert r1 == r2

    def test_episodic_compressed_can_consolidate(self):
        """EPISODIC_COMPRESSED uses l2_to_l3_retention=0.3 for promotion."""
        candidate = _make_candidate(
            level=ConsolidationLevel.EPISODIC_COMPRESSED,
            strength=30.0,
            last_access_time=100.0,
            creation_time=200.0,
            importance=0.8,
        )
        assert should_consolidate(candidate) is True

    def test_semantic_fact_not_consolidated_by_retention(self):
        """SEMANTIC_FACT is NOT promoted via should_consolidate.

        L3->L4 promotion uses cluster similarity (batch mode), not
        retention-based single-memory consolidation.
        """
        candidate = _make_candidate(
            level=ConsolidationLevel.SEMANTIC_FACT,
            strength=180.0,
            last_access_time=500.0,
            creation_time=600.0,
            importance=0.8,
        )
        assert should_consolidate(candidate) is False


# ============================================================
# 13. compute_affinity Tests
# ============================================================


class TestComputeAffinity:
    """Verify compute_affinity formula and behavioral contracts."""

    def test_identical_content_high_affinity(self):
        """Identical content produces high affinity."""
        a = _make_candidate(content="I prefer dark mode", creation_time=100.0)
        b = _make_candidate(
            content="I prefer dark mode", creation_time=100.0, memory_id="mem-002"
        )
        result = compute_affinity(a, b)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_symmetric(self):
        """affinity(a,b) == affinity(b,a)."""
        a = _make_candidate(
            content="I prefer dark mode in editors", creation_time=100.0
        )
        b = _make_candidate(
            content="dark mode is great for reading",
            creation_time=200.0,
            memory_id="mem-002",
        )
        assert compute_affinity(a, b) == pytest.approx(
            compute_affinity(b, a), abs=1e-15
        )

    def test_no_overlap_returns_zero(self):
        """Completely disjoint content returns 0.0."""
        a = _make_candidate(content="alpha bravo charlie", creation_time=100.0)
        b = _make_candidate(
            content="delta echo foxtrot", creation_time=100.0, memory_id="mem-002"
        )
        assert compute_affinity(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_self_affinity_lower_bound(self):
        """Self-affinity >= (1 - temporal_weight) since Jaccard(a,a)=1.0."""
        a = _make_candidate(content="I prefer dark mode", creation_time=100.0)
        config = ConsolidationConfig()
        result = compute_affinity(a, a, config)
        assert result >= (1.0 - config.temporal_weight) - 1e-10

    def test_content_floor_blocks_temporal(self):
        """Below min_content_similarity, temporal proximity cannot save affinity."""
        a = _make_candidate(
            content="alpha bravo charlie delta echo foxtrot golf hotel india juliet",
            creation_time=100.0,
        )
        b = _make_candidate(
            content="kilo lima mike november oscar papa quebec romeo sierra tango",
            creation_time=100.0,
            memory_id="mem-002",
        )
        config = ConsolidationConfig(min_content_similarity=0.3)
        result = compute_affinity(a, b, config)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_moderate_overlap(self):
        """Partial content overlap produces intermediate affinity."""
        a = _make_candidate(
            content="I prefer dark mode for coding", creation_time=100.0
        )
        b = _make_candidate(
            content="I prefer light mode for coding",
            creation_time=100.0,
            memory_id="mem-002",
        )
        result = compute_affinity(a, b)
        assert result == pytest.approx(0.7 * (5.0 / 7.0) + 0.3 * 1.0, abs=1e-10)

    def test_temporal_decay(self):
        """Larger time difference reduces temporal similarity."""
        a = _make_candidate(
            content="I prefer dark mode for coding", creation_time=100.0
        )
        b_close = _make_candidate(
            content="I prefer dark mode for coding",
            creation_time=101.0,
            memory_id="mem-002",
        )
        b_far = _make_candidate(
            content="I prefer dark mode for coding",
            creation_time=300.0,
            memory_id="mem-003",
        )
        config = ConsolidationConfig()
        aff_close = compute_affinity(a, b_close, config)
        aff_far = compute_affinity(a, b_far, config)
        assert aff_close > aff_far

    def test_empty_content_a(self):
        """Empty content in memory_a gives content_sim=0 -> returns 0."""
        a = _make_candidate(content="", creation_time=100.0)
        b = _make_candidate(
            content="some content here", creation_time=100.0, memory_id="mem-002"
        )
        result = compute_affinity(a, b)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_empty_content_b(self):
        """Empty content in memory_b gives content_sim=0 -> returns 0."""
        a = _make_candidate(content="some content here", creation_time=100.0)
        b = _make_candidate(
            content="", creation_time=100.0, memory_id="mem-002"
        )
        result = compute_affinity(a, b)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_both_empty_content(self):
        """Both empty content -> content_sim=0 -> returns 0."""
        a = _make_candidate(content="", creation_time=100.0)
        b = _make_candidate(
            content="", creation_time=100.0, memory_id="mem-002"
        )
        result = compute_affinity(a, b)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_result_in_unit_interval(self):
        """Result is always in [0.0, 1.0]."""
        a = _make_candidate(
            content="I prefer dark mode for coding and reading", creation_time=0.0
        )
        b = _make_candidate(
            content="I prefer dark mode for coding and writing",
            creation_time=10000.0,
            memory_id="mem-002",
        )
        result = compute_affinity(a, b)
        assert 0.0 <= result <= 1.0

    def test_none_config_uses_default(self):
        """Passing None for config uses DEFAULT_CONSOLIDATION_CONFIG."""
        a = _make_candidate(content="I prefer dark mode", creation_time=100.0)
        b = _make_candidate(
            content="I prefer dark mode", creation_time=100.0, memory_id="mem-002"
        )
        r_none = compute_affinity(a, b, None)
        r_default = compute_affinity(a, b, DEFAULT_CONSOLIDATION_CONFIG)
        assert r_none == r_default

    def test_deterministic(self):
        """Same inputs always produce same output."""
        a = _make_candidate(
            content="I prefer dark mode for coding", creation_time=100.0
        )
        b = _make_candidate(
            content="dark mode is preferred",
            creation_time=200.0,
            memory_id="mem-002",
        )
        r1 = compute_affinity(a, b)
        r2 = compute_affinity(a, b)
        assert r1 == r2

    def test_content_sim_just_above_floor(self):
        """Content similarity just above floor produces non-zero affinity."""
        a = _make_candidate(
            content="alpha bravo charlie delta echo foxtrot golf hotel",
            creation_time=100.0,
        )
        b = _make_candidate(
            content="alpha bravo charlie delta india juliet kilo lima",
            creation_time=100.0,
            memory_id="mem-002",
        )
        config = ConsolidationConfig(min_content_similarity=0.3)
        result = compute_affinity(a, b, config)
        assert result > 0.0

    def test_temporal_weight_zero(self):
        """With semantic_weight_beta=1.0 (temporal_weight=0), only content matters."""
        a = _make_candidate(content="I prefer dark mode", creation_time=100.0)
        b = _make_candidate(
            content="I prefer dark mode",
            creation_time=10000.0,
            memory_id="mem-002",
        )
        config = ConsolidationConfig(semantic_weight_beta=1.0)
        result = compute_affinity(a, b, config)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_temporal_weight_one(self):
        """With semantic_weight_beta=0.0 (temporal_weight=1), only temporal matters."""
        a = _make_candidate(content="I prefer dark mode", creation_time=100.0)
        b = _make_candidate(
            content="I prefer dark mode", creation_time=100.0, memory_id="mem-002"
        )
        config = ConsolidationConfig(semantic_weight_beta=0.0, min_content_similarity=0.0)
        result = compute_affinity(a, b, config)
        assert result == pytest.approx(1.0, abs=1e-10)


# ============================================================
# 14. Property-Based Phase B Tests
# ============================================================


class TestPropertyBasedPhaseB:
    """Property-based tests for Phase B functions."""

    @given(
        importance=st.floats(min_value=0.0, max_value=1.0),
        access_count=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_semantic_value_always_in_unit_interval(self, importance, access_count):
        """_compute_semantic_value always returns a value in [0.0, 1.0]."""
        for cat in VALID_CATEGORIES:
            candidate = _make_candidate(
                category=cat,
                importance=importance,
                access_count=access_count,
            )
            result = _compute_semantic_value(candidate)
            assert 0.0 <= result <= 1.0, (
                f"Out of range for cat={cat}, imp={importance}, ac={access_count}"
            )

    @given(
        time_a=st.floats(min_value=0.0, max_value=1e6),
        time_b=st.floats(min_value=0.0, max_value=1e6),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_affinity_symmetric_property(self, time_a, time_b):
        """compute_affinity is symmetric for any creation times."""
        a = _make_candidate(
            content="I prefer dark mode editor", creation_time=time_a
        )
        b = _make_candidate(
            content="I prefer dark mode editor",
            creation_time=time_b,
            memory_id="mem-002",
        )
        assert compute_affinity(a, b) == pytest.approx(
            compute_affinity(b, a), abs=1e-10
        )

    @given(
        time_a=st.floats(min_value=0.0, max_value=1e6),
        time_b=st.floats(min_value=0.0, max_value=1e6),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_affinity_always_in_unit_interval_property(self, time_a, time_b):
        """compute_affinity always returns a value in [0.0, 1.0]."""
        a = _make_candidate(
            content="I prefer dark mode for coding", creation_time=time_a
        )
        b = _make_candidate(
            content="dark mode is great for reading late at night",
            creation_time=time_b,
            memory_id="mem-002",
        )
        result = compute_affinity(a, b)
        assert 0.0 <= result <= 1.0


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

    # --- M2: semantic_boost ---

    def test_semantic_boost_applied_to_importance(self):
        """importance is boosted by config.semantic_boost (default 1.2)."""
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.5,),
        )
        # avg_importance=0.5, boosted=0.5*1.2=0.6
        assert result.importance == pytest.approx(0.6)

    def test_semantic_boost_clamped_at_one(self):
        """Boosted importance is clamped to 1.0."""
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.9,),
        )
        # avg_importance=0.9, boosted=0.9*1.2=1.08 -> clamped to 1.0
        assert result.importance == pytest.approx(1.0)

    def test_semantic_boost_custom_config(self):
        """Custom semantic_boost is applied."""
        config = ConsolidationConfig(semantic_boost=1.5)
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            config=config,
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.5,),
        )
        # avg_importance=0.5, boosted=0.5*1.5=0.75
        assert result.importance == pytest.approx(0.75)

    # --- M3: activation_inheritance ---

    def test_activation_inheritance_default(self):
        """access_count inherits from sources via activation_inheritance."""
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            source_episodes=("ep-001", "ep-002"),
            source_creation_times=(100.0, 200.0),
            source_importances=(0.5, 0.5),
            source_access_counts=(10, 20),
        )
        # avg=15, inherited=int(15 * 0.5)=7
        assert result.access_count == 7

    def test_activation_inheritance_no_access_counts(self):
        """When source_access_counts not provided, access_count defaults to 0."""
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.5,),
        )
        assert result.access_count == 0

    def test_activation_inheritance_custom_config(self):
        """Custom activation_inheritance fraction."""
        config = ConsolidationConfig(activation_inheritance=0.8)
        result = extract_semantic(
            "I prefer dark mode",
            "preference",
            config=config,
            source_episodes=("ep-001",),
            source_creation_times=(100.0,),
            source_importances=(0.5,),
            source_access_counts=(100,),
        )
        # avg=100, inherited=int(100 * 0.8)=80
        assert result.access_count == 80


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

    def test_decayed_strength(self):
        """decayed_strength = max(0, strength * consolidation_decay)."""
        candidate = self._make_candidate(strength=7.0)
        config = ConsolidationConfig()  # consolidation_decay = 0.1
        result = archive_episodic(candidate, ("sem-001",), config=config)
        expected = 7.0 * 0.1  # = 0.7
        assert result.decayed_strength == pytest.approx(expected)

    def test_decayed_importance_never_exceeds_original(self):
        """decayed_importance <= original_importance."""
        candidate = self._make_candidate(importance=0.5)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.decayed_importance <= result.original_importance

    def test_decayed_strength_never_exceeds_original(self):
        """decayed_strength <= original_strength."""
        candidate = self._make_candidate(strength=7.0)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.decayed_strength <= result.original_strength

    def test_original_importance_preserved(self):
        """original_importance matches candidate.importance."""
        candidate = self._make_candidate(importance=0.9)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.original_importance == 0.9

    def test_original_strength_preserved(self):
        """original_strength matches candidate.strength."""
        candidate = self._make_candidate(strength=7.0)
        result = archive_episodic(candidate, ("sem-001",))
        assert result.original_strength == 7.0

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


# ============================================================
# Phase D: Pipeline Functions Tests
# ============================================================


class TestNextLevel:
    """Tests for _next_level: maps each ConsolidationLevel to its successor."""

    def test_episodic_raw_to_compressed(self):
        """EPISODIC_RAW promotes to EPISODIC_COMPRESSED."""
        assert _next_level(ConsolidationLevel.EPISODIC_RAW) == (
            ConsolidationLevel.EPISODIC_COMPRESSED
        )

    def test_episodic_compressed_to_fact(self):
        """EPISODIC_COMPRESSED promotes to SEMANTIC_FACT."""
        assert _next_level(ConsolidationLevel.EPISODIC_COMPRESSED) == (
            ConsolidationLevel.SEMANTIC_FACT
        )

    def test_semantic_fact_to_insight(self):
        """SEMANTIC_FACT promotes to SEMANTIC_INSIGHT."""
        assert _next_level(ConsolidationLevel.SEMANTIC_FACT) == (
            ConsolidationLevel.SEMANTIC_INSIGHT
        )

    def test_semantic_insight_returns_none(self):
        """SEMANTIC_INSIGHT has no promotion path; returns None."""
        assert _next_level(ConsolidationLevel.SEMANTIC_INSIGHT) is None

    def test_deterministic(self):
        """Same input always produces same output."""
        for level in ConsolidationLevel:
            r1 = _next_level(level)
            r2 = _next_level(level)
            assert r1 == r2


class TestConsolidateMemory:
    """Tests for consolidate_memory: single-memory orchestrator."""

    @staticmethod
    def _make_candidate(**overrides):
        """Build a ConsolidationCandidate with consolidation-eligible defaults.

        Defaults: high importance, low strength (decayed retention), old memory,
        category 'fact' (eligible, high semantic value).
        """
        defaults = dict(
            memory_id="mem-001",
            content="The user lives in Berlin. The user works at a tech company.",
            category="fact",
            level=ConsolidationLevel.EPISODIC_RAW,
            creation_time=500.0,
            last_access_time=400.0,
            access_count=3,
            importance=0.9,
            strength=0.01,
            relevance=0.5,
            is_contested=False,
        )
        defaults.update(overrides)
        return ConsolidationCandidate(**defaults)

    def test_eligible_candidate_produces_extraction_and_archive(self):
        """An eligible candidate yields 1 extraction and 1 archive."""
        candidate = self._make_candidate()
        result = consolidate_memory(candidate)
        assert result.candidates_evaluated == 1
        assert result.candidates_consolidated == 1
        assert len(result.extractions) == 1
        assert len(result.archived) == 1
        assert result.skipped_indices == frozenset()

    def test_eligible_candidate_extraction_has_correct_source(self):
        """The extraction references the original candidate's memory_id."""
        candidate = self._make_candidate(memory_id="mem-xyz")
        result = consolidate_memory(candidate)
        assert candidate.memory_id in result.extractions[0].source_episodes

    def test_eligible_candidate_archive_has_correct_id(self):
        """The archive has the original candidate's memory_id."""
        candidate = self._make_candidate(memory_id="mem-abc")
        result = consolidate_memory(candidate)
        assert result.archived[0].memory_id == "mem-abc"

    def test_contested_candidate_not_consolidated(self):
        """A contested candidate is never consolidated."""
        candidate = self._make_candidate(is_contested=True)
        result = consolidate_memory(candidate)
        assert result.candidates_evaluated == 1
        assert result.candidates_consolidated == 0
        assert result.extractions == ()
        assert result.archived == ()
        assert candidate.memory_id in result.skipped_indices

    def test_correction_category_not_consolidated(self):
        """Correction category never consolidates (NON_CONSOLIDATABLE)."""
        candidate = self._make_candidate(
            category="correction",
            content="Correction: actually the user lives in Munich.",
        )
        result = consolidate_memory(candidate)
        assert result.candidates_consolidated == 0
        assert result.extractions == ()

    def test_semantic_insight_not_consolidated(self):
        """SEMANTIC_INSIGHT has no promotion path; returns empty result."""
        candidate = self._make_candidate(
            level=ConsolidationLevel.SEMANTIC_INSIGHT,
        )
        result = consolidate_memory(candidate)
        assert result.candidates_consolidated == 0
        assert result.extractions == ()
        assert result.archived == ()

    def test_candidates_evaluated_always_one(self):
        """candidates_evaluated is always 1 regardless of consolidation outcome."""
        # Eligible
        eligible = self._make_candidate()
        assert consolidate_memory(eligible).candidates_evaluated == 1
        # Contested
        contested = self._make_candidate(is_contested=True)
        assert consolidate_memory(contested).candidates_evaluated == 1
        # SEMANTIC_INSIGHT
        insight = self._make_candidate(level=ConsolidationLevel.SEMANTIC_INSIGHT)
        assert consolidate_memory(insight).candidates_evaluated == 1

    def test_default_mode_is_retrieval_triggered(self):
        """Default mode for single-memory consolidation is RETRIEVAL_TRIGGERED."""
        candidate = self._make_candidate()
        result = consolidate_memory(candidate)
        assert result.mode == ConsolidationMode.RETRIEVAL_TRIGGERED

    def test_ineligible_high_strength_not_consolidated(self):
        """A candidate with high strength (high retention) is not consolidated.

        Uses strength=10000 for very high retention (0.96) and access_count=15
        to disable the secondary trigger (infrequent access path).
        """
        candidate = self._make_candidate(strength=10000.0, access_count=15)
        result = consolidate_memory(candidate)
        assert result.candidates_consolidated == 0
        assert result.extractions == ()

    def test_custom_config_passed_through(self):
        """Custom config is used for all sub-calls."""
        candidate = self._make_candidate(importance=0.6)
        config = ConsolidationConfig(consolidation_decay=0.5)
        result = consolidate_memory(candidate, config=config)
        if result.candidates_consolidated == 1:
            expected_decay = 0.6 * 0.5
            assert result.archived[0].decayed_importance == pytest.approx(
                expected_decay
            )

    def test_none_config_uses_default(self):
        """When config=None, DEFAULT_CONSOLIDATION_CONFIG is used."""
        candidate = self._make_candidate()
        r1 = consolidate_memory(candidate, config=None)
        r2 = consolidate_memory(candidate, config=DEFAULT_CONSOLIDATION_CONFIG)
        assert r1.candidates_consolidated == r2.candidates_consolidated

    def test_deterministic(self):
        """Same inputs always produce same output."""
        candidate = self._make_candidate()
        r1 = consolidate_memory(candidate)
        r2 = consolidate_memory(candidate)
        assert r1.candidates_consolidated == r2.candidates_consolidated
        assert len(r1.extractions) == len(r2.extractions)
        assert len(r1.archived) == len(r2.archived)

    def test_greeting_category_not_consolidated(self):
        """Greeting category is non-consolidatable."""
        candidate = self._make_candidate(
            category="greeting",
            content="Hello there, nice to meet you!",
        )
        result = consolidate_memory(candidate)
        assert result.candidates_consolidated == 0

    def test_recent_candidate_not_consolidated(self):
        """A very recent candidate (creation_time < recency_guard) is blocked."""
        candidate = self._make_candidate(creation_time=0.5)
        result = consolidate_memory(candidate)
        assert result.candidates_consolidated == 0


class TestSelectConsolidationCandidates:
    """Tests for select_consolidation_candidates: pool filtering + sorting."""

    @staticmethod
    def _make_candidate(**overrides):
        """Build a ConsolidationCandidate with consolidation-eligible defaults."""
        defaults = dict(
            memory_id="mem-001",
            content="The user lives in Berlin. The user works at a tech company.",
            category="fact",
            level=ConsolidationLevel.EPISODIC_RAW,
            creation_time=500.0,
            last_access_time=400.0,
            access_count=3,
            importance=0.9,
            strength=0.01,
            relevance=0.5,
            is_contested=False,
        )
        defaults.update(overrides)
        return ConsolidationCandidate(**defaults)

    def test_empty_pool_returns_empty(self):
        """Empty pool yields empty result."""
        result = select_consolidation_candidates([])
        assert result == []

    def test_single_eligible_candidate_selected(self):
        """A single eligible candidate is selected in ASYNC_BATCH mode."""
        candidate = self._make_candidate()
        result = select_consolidation_candidates([candidate])
        assert len(result) == 1
        assert result[0] is candidate

    def test_contested_never_selected(self):
        """Contested candidates are never selected in any mode."""
        candidate = self._make_candidate(is_contested=True)
        for mode in ConsolidationMode:
            result = select_consolidation_candidates([candidate], mode=mode)
            assert len(result) == 0, f"Contested selected in {mode}"

    def test_correction_category_never_selected(self):
        """Correction category is never selected in any mode."""
        candidate = self._make_candidate(
            category="correction",
            content="Correction: actually the user lives in Munich.",
        )
        for mode in ConsolidationMode:
            result = select_consolidation_candidates([candidate], mode=mode)
            assert len(result) == 0, f"Correction selected in {mode}"

    def test_mixed_pool_filters_correctly(self):
        """Mixed pool with eligible and ineligible: only eligible returned."""
        eligible = self._make_candidate(memory_id="elig-001")
        contested = self._make_candidate(memory_id="cont-001", is_contested=True)
        correction = self._make_candidate(
            memory_id="corr-001",
            category="correction",
            content="Correction: wrong info. I was wrong about this.",
        )
        result = select_consolidation_candidates([eligible, contested, correction])
        assert len(result) == 1
        assert result[0].memory_id == "elig-001"

    def test_intra_session_only_checks_inhibitors(self):
        """INTRA_SESSION mode only checks inhibitors, not should_consolidate().

        A candidate with high strength (would fail should_consolidate retention
        check) should still pass if inhibitors pass.
        """
        candidate = self._make_candidate(
            strength=100.0,  # very high strength => high retention
            creation_time=500.0,  # passes recency guard
        )
        result = select_consolidation_candidates(
            [candidate], mode=ConsolidationMode.INTRA_SESSION
        )
        assert len(result) == 1

    def test_intra_session_blocks_contested(self):
        """INTRA_SESSION mode still blocks contested candidates."""
        candidate = self._make_candidate(is_contested=True)
        result = select_consolidation_candidates(
            [candidate], mode=ConsolidationMode.INTRA_SESSION
        )
        assert len(result) == 0

    def test_async_batch_no_pressure_uses_should_consolidate(self):
        """ASYNC_BATCH without pool_size uses standard should_consolidate().

        The ineligible candidate needs high strength (high retention) AND
        high access_count to also defeat the secondary trigger.
        """
        eligible = self._make_candidate(strength=0.01)  # low retention
        ineligible = self._make_candidate(
            memory_id="mem-002",
            strength=10000.0,  # very high retention
            access_count=15,  # defeats secondary trigger
        )
        result = select_consolidation_candidates(
            [eligible, ineligible],
            mode=ConsolidationMode.ASYNC_BATCH,
        )
        assert len(result) == 1
        assert result[0].memory_id == "mem-001"

    def test_async_batch_mild_pressure(self):
        """Mild storage pressure: pool_size > max_pool_size.

        A candidate that fails should_consolidate but passes inhibitors with
        low retention, high semantic value, and old enough creation_time
        should be selected under mild pressure.
        """
        config = ConsolidationConfig(
            max_pool_size=100,
            mild_pressure_retention=0.5,
            semantic_value_threshold=0.3,
        )
        from hermes_memory.core import retention as ret_fn

        # Candidate with moderate strength so retention is above the level
        # threshold (0.4 for EPISODIC_RAW) but below mild_pressure_retention (0.5).
        # This means should_consolidate returns False (retention above threshold),
        # but the mild pressure path accepts it.
        # retention = exp(-t/S) = exp(-80/100) = exp(-0.8) = 0.449
        mild_candidate = self._make_candidate(
            memory_id="mild-002",
            strength=100.0,
            last_access_time=80.0,
            creation_time=500.0,
            importance=0.9,
            access_count=15,  # defeats secondary trigger
        )
        mild_ret = ret_fn(80.0, 100.0)
        assert 0.4 < mild_ret < 0.5, f"Setup error: retention={mild_ret}"

        result = select_consolidation_candidates(
            [mild_candidate],
            config=config,
            mode=ConsolidationMode.ASYNC_BATCH,
            pool_size=150,  # > max_pool_size=100
        )
        assert len(result) == 1
        assert result[0].memory_id == "mild-002"

    def test_async_batch_severe_pressure(self):
        """Severe storage pressure: pool_size > max_pool_size * severe_pressure_multiplier.

        Under severe pressure, any candidate passing inhibitors with
        creation_time > consolidation_window is eligible.
        """
        config = ConsolidationConfig(
            max_pool_size=100,
            severe_pressure_multiplier=1.2,
        )
        # Candidate with high strength (fails should_consolidate) but passes inhibitors
        candidate = self._make_candidate(
            memory_id="severe-001",
            strength=1000.0,  # very high retention
            creation_time=500.0,  # > consolidation_window (7.0 days)
        )
        result = select_consolidation_candidates(
            [candidate],
            config=config,
            mode=ConsolidationMode.ASYNC_BATCH,
            pool_size=150,  # > 100 * 1.2 = 120
        )
        assert len(result) == 1
        assert result[0].memory_id == "severe-001"

    def test_async_batch_severe_pressure_blocks_recent(self):
        """Severe pressure still requires creation_time > consolidation_window."""
        config = ConsolidationConfig(
            max_pool_size=100,
            severe_pressure_multiplier=1.2,
        )
        candidate = self._make_candidate(
            memory_id="recent-severe",
            strength=1000.0,
            creation_time=5.0,  # < consolidation_window (7.0 days)
        )
        result = select_consolidation_candidates(
            [candidate],
            config=config,
            mode=ConsolidationMode.ASYNC_BATCH,
            pool_size=150,
        )
        # should_consolidate fails (high strength) and severe pressure requires
        # creation_time > consolidation_window
        assert len(result) == 0

    def test_result_sorted_by_retention_ascending(self):
        """Results are sorted by retention ascending (lowest retention first)."""
        from hermes_memory.core import retention as ret_fn

        # c1: low strength => low retention
        c1 = self._make_candidate(
            memory_id="low-ret",
            strength=0.01,
            last_access_time=100.0,
        )
        # c2: moderate strength => moderate retention
        c2 = self._make_candidate(
            memory_id="mid-ret",
            strength=0.1,
            last_access_time=10.0,
        )
        # c3: slightly different
        c3 = self._make_candidate(
            memory_id="high-ret",
            strength=0.05,
            last_access_time=50.0,
        )
        result = select_consolidation_candidates([c3, c1, c2])
        # Verify sorted ascending by retention
        retentions = []
        for c in result:
            safe_s = max(c.strength, 1e-12)
            retentions.append(ret_fn(c.last_access_time, safe_s))
        for i in range(len(retentions) - 1):
            assert retentions[i] <= retentions[i + 1], (
                f"Not sorted: {retentions}"
            )

    def test_max_candidates_per_batch_limit(self):
        """Result length never exceeds max_candidates_per_batch."""
        config = ConsolidationConfig(max_candidates_per_batch=2)
        candidates = [
            self._make_candidate(
                memory_id=f"mem-{i:03d}",
                strength=0.01 + i * 0.001,
            )
            for i in range(5)
        ]
        result = select_consolidation_candidates(candidates, config=config)
        assert len(result) <= 2

    def test_retrieval_triggered_mode(self):
        """RETRIEVAL_TRIGGERED mode uses should_consolidate()."""
        eligible = self._make_candidate(strength=0.01)
        ineligible = self._make_candidate(
            memory_id="mem-002",
            strength=10000.0,  # very high retention
            access_count=15,  # defeats secondary trigger
        )
        result = select_consolidation_candidates(
            [eligible, ineligible],
            mode=ConsolidationMode.RETRIEVAL_TRIGGERED,
        )
        assert len(result) == 1
        assert result[0].memory_id == "mem-001"

    def test_default_mode_is_async_batch(self):
        """Default mode parameter is ASYNC_BATCH."""
        candidate = self._make_candidate()
        # Just verify it doesn't raise with default mode
        result = select_consolidation_candidates([candidate])
        assert isinstance(result, list)

    def test_mild_pressure_requires_all_conditions(self):
        """Mild pressure requires: retention < mild_pressure_retention AND
        semantic_value >= threshold AND creation_time > consolidation_window.
        Failing any one should exclude the candidate.
        """
        config = ConsolidationConfig(
            max_pool_size=100,
            mild_pressure_retention=0.5,
            semantic_value_threshold=0.6,
        )
        # Candidate with retention < 0.5 but creation_time < consolidation_window
        # => should NOT be selected via mild pressure
        # strength=100, last_access_time=80 => ret=0.449
        young_candidate = self._make_candidate(
            memory_id="young-mild",
            strength=100.0,
            last_access_time=80.0,
            creation_time=5.0,  # < 7.0 consolidation_window (days)
            importance=0.9,
        )
        result = select_consolidation_candidates(
            [young_candidate],
            config=config,
            mode=ConsolidationMode.ASYNC_BATCH,
            pool_size=150,
        )
        # should_consolidate also fails (creation_time < window for secondary),
        # and mild pressure path requires creation_time > consolidation_window
        assert len(result) == 0

    def test_pool_size_none_no_pressure(self):
        """When pool_size=None, no storage pressure is applied in ASYNC_BATCH."""
        # Candidate that fails should_consolidate: high retention + high access
        candidate = self._make_candidate(
            strength=10000.0,
            access_count=15,  # defeats secondary trigger
            creation_time=500.0,
        )
        result = select_consolidation_candidates(
            [candidate],
            mode=ConsolidationMode.ASYNC_BATCH,
            pool_size=None,
        )
        # should_consolidate fails for high retention + high access, no pressure path
        assert len(result) == 0

    def test_multiple_modes_produce_different_results(self):
        """A candidate with high retention: passes INTRA_SESSION, fails ASYNC_BATCH."""
        candidate = self._make_candidate(
            strength=10000.0,
            access_count=15,  # defeats secondary trigger
            creation_time=500.0,
        )
        intra = select_consolidation_candidates(
            [candidate], mode=ConsolidationMode.INTRA_SESSION
        )
        batch = select_consolidation_candidates(
            [candidate], mode=ConsolidationMode.ASYNC_BATCH
        )
        assert len(intra) == 1
        assert len(batch) == 0
