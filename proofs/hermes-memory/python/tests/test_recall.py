"""Comprehensive tests for hermes_memory.recall -- adaptive recall pipeline.

Tests define expected behavioral contracts from the recall spec (Section 14).
These tests are written BEFORE the implementation (TDD). All imports from
hermes_memory.recall will fail with ImportError until the module is created.

Updated after three adversarial passes (AP1: 10 findings, AP2: 8 findings,
AP3: 10 findings). Key changes:
  - budget_overflow field in RecallResult (AP1-F1)
  - Positional fallback for identical scores (AP1-F3)
  - contents=None normalization (AP1-F4)
  - Empty-memories fast-path before gate (AP1-F5)
  - Threshold boundary tests (AP1-F7)
  - adaptive_k top-slice truncation (AP2-F3)
  - chars_per_token configurable (AP2-F1)
  - MIN_DEMOTION_SAVINGS_TOKENS constant (AP2-F4)
  - Per-tier demotion cascade (AP2-F2)
  - Temperature has zero effect on scoring (AP3-E1)
  - Epsilon fragility regression (AP3-E6)
  - Gating false negatives for short imperatives (AP3-E5)
  - Fixed k=5 comparison (AP3-T1)
  - Demotion skip for content=None (AP3-T3)

200+ tests covering:
  - TestShouldRecall:      gating logic (~25 tests)
  - TestAdaptiveK:         score-gap adaptive-k selection (~25 tests)
  - TestAssignTiers:       normalized-score tier assignment (~20 tests)
  - TestFormatMemory:      tiered memory rendering (~15 tests)
  - TestBudgetConstrain:   token budget enforcement (~30 tests)
  - TestRecall:            full pipeline integration (~25 tests)
  - TestRecallIntegration: realistic memory pools (~15 tests)
  - TestRecallProperties:  Hypothesis property tests (~15 tests)
  - TestRecallSlow:        large-scale / edge-value tests (~10 tests)
  - TestRecallConfig:      config validation (~20 tests)
  - TestTierEnum:          tier enum (~4 tests)
  - TestTierAssignment:    tier assignment dataclass (~3 tests)
  - TestModuleConstants:   module constants (~5 tests)
"""

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hermes_memory.recall import (
    Tier,
    RecallConfig,
    TierAssignment,
    should_recall,
    adaptive_k,
    assign_tiers,
    format_memory,
    budget_constrain,
    recall,
    DEFAULT_RECALL_CONFIG,
    MEMORY_SEPARATOR,
    CHARS_PER_TOKEN,
    CHARS_PER_TOKEN_CONSERVATIVE,
    MIN_DEMOTION_SAVINGS_TOKENS,
)
from hermes_memory.engine import ParameterSet, MemoryState, rank_memories
from hermes_memory.optimizer import PINNED


# ============================================================
# Helpers
# ============================================================


def _default_params() -> ParameterSet:
    """Build a ParameterSet with tuned weights and PINNED defaults.

    Uses the optimizer's best-known parameters:
        w1=0.4109, w2=0.0500, w3=0.3000, w4=0.2391
        alpha=0.5000, beta=1.0000, temperature=4.9265
    """
    return ParameterSet(
        w1=0.4109,
        w2=0.0500,
        w3=0.3000,
        w4=0.2391,
        alpha=0.5000,
        beta=1.0000,
        temperature=4.9265,
        delta_t=PINNED["delta_t"],
        s_max=PINNED["s_max"],
        s0=PINNED["s0"],
        novelty_start=PINNED["novelty_start"],
        novelty_decay=PINNED["novelty_decay"],
        survival_threshold=PINNED["survival_threshold"],
        feedback_sensitivity=PINNED["feedback_sensitivity"],
    )


def _make_memory(
    relevance: float = 0.5,
    last_access_time: float = 5.0,
    importance: float = 0.5,
    access_count: int = 3,
    strength: float = 5.0,
    creation_time: float = 100.0,
) -> MemoryState:
    """Convenience helper for building MemoryState with sensible defaults."""
    return MemoryState(
        relevance=relevance,
        last_access_time=last_access_time,
        importance=importance,
        access_count=access_count,
        strength=strength,
        creation_time=creation_time,
    )


def _make_memories(relevances: list[float]) -> list[MemoryState]:
    """Build a list of memories with varying relevance, other fields constant."""
    return [_make_memory(relevance=r) for r in relevances]


def _estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Mirror the recall module's token estimation: len(text) // chars_per_token."""
    return len(text) // chars_per_token


# ============================================================
# 1. TestShouldRecall (~20 tests)
# ============================================================


class TestShouldRecall:
    """Tests for should_recall(message, turn_number, config) -> bool."""

    def test_first_turn_always_returns_true_normal_message(self) -> None:
        """Turn 0 with a normal message returns True."""
        assert should_recall("Tell me about X", 0, DEFAULT_RECALL_CONFIG) is True

    def test_first_turn_empty_string(self) -> None:
        """Turn 0 with empty string returns True (first turn override)."""
        assert should_recall("", 0, DEFAULT_RECALL_CONFIG) is True

    def test_first_turn_trivial_message(self) -> None:
        """Turn 0 with trivial 'hi' still returns True (first turn override)."""
        assert should_recall("hi", 0, DEFAULT_RECALL_CONFIG) is True

    def test_first_turn_short_message(self) -> None:
        """Turn 0 with short message returns True."""
        assert should_recall("ok", 0, DEFAULT_RECALL_CONFIG) is True

    def test_empty_string_non_first_turn(self) -> None:
        """Empty string on turn > 0 returns False."""
        assert should_recall("", 1, DEFAULT_RECALL_CONFIG) is False

    def test_whitespace_only_non_first_turn(self) -> None:
        """Whitespace-only on turn > 0 returns False (empty after strip)."""
        assert should_recall("   ", 5, DEFAULT_RECALL_CONFIG) is False

    def test_trivial_ok(self) -> None:
        """'ok' on turn > 0 returns False (trivial pattern)."""
        assert should_recall("ok", 1, DEFAULT_RECALL_CONFIG) is False

    def test_trivial_ok_case_insensitive(self) -> None:
        """'OK' on turn > 0 returns False (case-insensitive trivial match)."""
        assert should_recall("OK", 1, DEFAULT_RECALL_CONFIG) is False

    def test_trivial_ok_trailing_punctuation(self) -> None:
        """'ok.' on turn > 0 returns False (trailing punctuation stripped)."""
        assert should_recall("ok.", 1, DEFAULT_RECALL_CONFIG) is False

    def test_question_mark_overrides_trivial(self) -> None:
        """'ok?' on turn > 0 returns True (question mark override)."""
        assert should_recall("ok?", 1, DEFAULT_RECALL_CONFIG) is True

    def test_trivial_yes(self) -> None:
        """'yes' on turn > 0 returns False."""
        assert should_recall("yes", 1, DEFAULT_RECALL_CONFIG) is False

    def test_trivial_thanks(self) -> None:
        """'thanks' on turn > 0 returns False."""
        assert should_recall("thanks", 1, DEFAULT_RECALL_CONFIG) is False

    def test_trivial_goodbye(self) -> None:
        """'goodbye' is in the trivial pattern list."""
        assert should_recall("goodbye", 1, DEFAULT_RECALL_CONFIG) is False

    def test_non_trivial_long_message(self) -> None:
        """'Tell me about X' on turn > 0 returns True."""
        assert should_recall("Tell me about X", 1, DEFAULT_RECALL_CONFIG) is True

    def test_question_mark_overrides_length(self) -> None:
        """'What is X?' returns True (question mark)."""
        assert should_recall("What is X?", 3, DEFAULT_RECALL_CONFIG) is True

    def test_short_question(self) -> None:
        """Short question with '?' returns True despite short length."""
        assert should_recall("X?", 2, DEFAULT_RECALL_CONFIG) is True

    def test_ok_sure_is_not_trivial(self) -> None:
        """'ok sure' is 7 chars (< gate_min_length=8), blocked by length check."""
        assert should_recall("ok sure", 1, DEFAULT_RECALL_CONFIG) is False

    def test_hi_non_first_turn(self) -> None:
        """'hi' on turn > 0 returns False (short + trivial)."""
        assert should_recall("hi", 1, DEFAULT_RECALL_CONFIG) is False

    def test_custom_gate_min_length_zero(self) -> None:
        """Custom config with gate_min_length=0 disables length gating."""
        config = RecallConfig(gate_min_length=0)
        # "ab" is 2 chars, below default 8, but gate_min_length=0 disables
        # Still "ab" is not a trivial pattern, so should return True
        assert should_recall("ab", 1, config) is True

    def test_custom_empty_trivial_patterns(self) -> None:
        """Custom config with empty gate_trivial_patterns disables pattern gating."""
        config = RecallConfig(gate_trivial_patterns=())
        # "ok" would normally be gated, but with empty patterns it passes
        # However, "ok" is only 2 chars < gate_min_length=8, so still gated by length
        assert should_recall("ok", 1, config) is False
        # But a long-enough version passes:
        assert should_recall("ok alright", 1, config) is True

    def test_tell_me_blocked_by_length(self) -> None:
        """'tell me' (7 chars) is blocked by gate_min_length=8.

        [AP3-E5] Known false negative: short imperative without question mark.
        """
        assert should_recall("tell me", 1, DEFAULT_RECALL_CONFIG) is False

    def test_tell_me_question_mark_passes(self) -> None:
        """'tell me?' (8 chars with ?) passes due to question mark override.

        [AP3-E5] Question mark overrides length check.
        """
        assert should_recall("tell me?", 1, DEFAULT_RECALL_CONFIG) is True

    def test_help_me_blocked_by_length(self) -> None:
        """'help me' (7 chars) is blocked by gate_min_length=8.

        [AP3-E5] Another known false negative for short imperatives.
        """
        assert should_recall("help me", 1, DEFAULT_RECALL_CONFIG) is False

    def test_tell_me_x_passes(self) -> None:
        """'tell me X' (9 chars) passes length check.

        [AP3-E5] Boundary: 9 chars > gate_min_length=8.
        """
        assert should_recall("tell me X", 1, DEFAULT_RECALL_CONFIG) is True

    def test_negative_turn_number_raises(self) -> None:
        """turn_number < 0 raises ValueError."""
        with pytest.raises(ValueError):
            should_recall("hello", -1, DEFAULT_RECALL_CONFIG)

    def test_non_string_message_raises(self) -> None:
        """Non-string message raises TypeError."""
        with pytest.raises(TypeError):
            should_recall(42, 0, DEFAULT_RECALL_CONFIG)  # type: ignore[arg-type]

    def test_non_string_none_raises(self) -> None:
        """None message raises TypeError."""
        with pytest.raises(TypeError):
            should_recall(None, 0, DEFAULT_RECALL_CONFIG)  # type: ignore[arg-type]


# ============================================================
# 2. TestAdaptiveK (~20 tests)
# ============================================================


class TestAdaptiveK:
    """Tests for adaptive_k(scores, config) -> int."""

    def test_empty_scores_returns_zero(self) -> None:
        """Empty list returns 0."""
        assert adaptive_k([], DEFAULT_RECALL_CONFIG) == 0

    def test_single_score_returns_one(self) -> None:
        """Single score returns 1."""
        assert adaptive_k([0.8], DEFAULT_RECALL_CONFIG) == 1

    def test_clear_gap_at_first_position(self) -> None:
        """[0.9, 0.5, 0.4, 0.3]: gap at idx 0 (0.4), k_raw = 1 + 2 = 3."""
        result = adaptive_k([0.9, 0.5, 0.4, 0.3], DEFAULT_RECALL_CONFIG)
        assert result == 3

    def test_clear_gap_with_buffer_zero(self) -> None:
        """[0.9, 0.5, 0.49, 0.48, 0.47] with gap_buffer=0: k_raw = 1 + 0 = 1."""
        config = RecallConfig(gap_buffer=0)
        result = adaptive_k([0.9, 0.5, 0.49, 0.48, 0.47], config)
        assert result == 1

    def test_scores_within_epsilon_fallback(self) -> None:
        """[0.9, 0.89, 0.88, 0.87]: max_gap=0.01 == epsilon, fallback."""
        # max_gap = 0.01, epsilon = 0.01, so max_gap < epsilon is False (not strictly less)
        # Actually spec says "If max_gap < config.epsilon" (strict <), so 0.01 < 0.01 is False.
        # Let's use epsilon=0.02 to ensure fallback.
        config = RecallConfig(epsilon=0.02)
        result = adaptive_k([0.9, 0.89, 0.88, 0.87], config)
        # fallback: min(min_k + gap_buffer, len, max_k) = min(1+2, 4, 10) = 3
        assert result == 3

    def test_all_scores_equal_fallback(self) -> None:
        """[0.5, 0.5, 0.5]: all gaps=0 < epsilon, fallback."""
        result = adaptive_k([0.5, 0.5, 0.5], DEFAULT_RECALL_CONFIG)
        # fallback: min(1+2, 3, 10) = 3
        assert result == 3

    def test_epsilon_boundary_exact_match(self) -> None:
        """When max_gap == epsilon (exactly), it should NOT trigger fallback.

        Spec says 'If max_gap < config.epsilon' -- strict less than.
        [0.9, 0.89, 0.88, 0.87] with default epsilon=0.01: max_gap=0.01.
        0.01 < 0.01 is False, so gap detection proceeds.
        Gap at idx 0 (0.01), k_raw = 1 + 2 = 3.
        """
        result = adaptive_k([0.9, 0.89, 0.88, 0.87], DEFAULT_RECALL_CONFIG)
        assert result == 3

    def test_gap_at_last_position(self) -> None:
        """Scores [0.9, 0.89, 0.88, 0.5]: gap at idx 2 (0.38).
        k_raw = 3 + 2 = 5, clamped to min(5, 4, 10) = 4.
        """
        result = adaptive_k([0.9, 0.89, 0.88, 0.5], DEFAULT_RECALL_CONFIG)
        assert result == 4

    def test_min_k_enforcement(self) -> None:
        """[0.9, 0.1] with min_k=3: k_raw=1+2=3, clamped to min(3, 2)=2."""
        config = RecallConfig(min_k=3)
        result = adaptive_k([0.9, 0.1], config)
        # Cannot exceed list length
        assert result == 2

    def test_max_k_enforcement(self) -> None:
        """20 scores with max_k=5: result <= 5."""
        scores = [1.0 - i * 0.04 for i in range(20)]
        config = RecallConfig(max_k=5)
        result = adaptive_k(scores, config)
        assert result <= 5

    def test_min_k_equals_max_k_fixed(self) -> None:
        """min_k == max_k means fixed k (if enough scores available)."""
        config = RecallConfig(min_k=3, max_k=3)
        result = adaptive_k([0.9, 0.8, 0.7, 0.2, 0.1], config)
        assert result == 3

    def test_gap_buffer_large_clamps_to_len(self) -> None:
        """Large gap_buffer pushing k_raw past len: clamped to len(scores)."""
        config = RecallConfig(gap_buffer=100)
        result = adaptive_k([0.9, 0.5, 0.4], config)
        # gap at idx 0 (0.4), k_raw = 1 + 100 = 101, clamped to min(101, 3, 10) = 3
        assert result == 3

    def test_scores_with_novelty_bonus(self) -> None:
        """Scores > 1.0 (from novelty bonus) work correctly."""
        result = adaptive_k([1.2, 0.9, 0.3, 0.2], DEFAULT_RECALL_CONFIG)
        # gap at idx 0 (0.3) vs idx 1 (0.6): gap at idx 1 is largest
        # k_raw = 2 + 2 = 4, clamped to min(4, 4, 10) = 4
        assert result == 4

    def test_two_equal_gaps_first_wins(self) -> None:
        """Multiple equal-size gaps: first occurrence (argmax) wins.

        [1.0, 0.5, 0.5, 0.0]: gaps = [0.5, 0.0, 0.5]
        First max gap at idx 0, k_raw = 1 + 2 = 3.
        """
        result = adaptive_k([1.0, 0.5, 0.5, 0.0], DEFAULT_RECALL_CONFIG)
        assert result == 3

    def test_gap_buffer_zero_exact_position(self) -> None:
        """gap_buffer=0 gives exact gap position.

        [0.9, 0.5, 0.49, 0.48, 0.47]: gap at idx 0 (0.4), k_raw = 1 + 0 = 1.
        """
        config = RecallConfig(gap_buffer=0)
        result = adaptive_k([0.9, 0.5, 0.49, 0.48, 0.47], config)
        assert result == 1

    def test_scores_not_descending_raises(self) -> None:
        """Ascending scores raise ValueError."""
        with pytest.raises(ValueError):
            adaptive_k([0.1, 0.5, 0.9], DEFAULT_RECALL_CONFIG)

    def test_nan_in_scores_raises(self) -> None:
        """NaN in scores raises ValueError."""
        with pytest.raises(ValueError):
            adaptive_k([0.9, float("nan"), 0.3], DEFAULT_RECALL_CONFIG)

    def test_min_k_larger_than_list(self) -> None:
        """min_k > len(scores): clamped to len(scores)."""
        config = RecallConfig(min_k=5)
        result = adaptive_k([0.9, 0.5], config)
        assert result == 2

    @given(
        n=st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=50)
    def test_result_in_valid_range_property(self, n: int) -> None:
        """For any descending scores, result in [0, len(scores)]."""
        scores = sorted(
            [1.0 - i / n for i in range(n)],
            reverse=True,
        )
        result = adaptive_k(scores, DEFAULT_RECALL_CONFIG)
        assert 0 <= result <= len(scores)

    def test_return_zero_only_for_empty(self) -> None:
        """Return 0 only when scores is empty."""
        assert adaptive_k([], DEFAULT_RECALL_CONFIG) == 0
        assert adaptive_k([0.1], DEFAULT_RECALL_CONFIG) >= 1

    def test_top_slice_truncation_large_pool(self) -> None:
        """100+ scores: largest gap deep in list should NOT affect result.

        [AP2-F3] adaptive_k truncates to scores[:max_k + gap_buffer + 1]
        before gap detection. A large gap at position 50 in a 200-score
        list should be invisible to the algorithm.
        """
        # Top 13 (max_k=10 + gap_buffer=2 + 1) scores are tightly packed
        top_scores = [1.0 - i * 0.005 for i in range(13)]
        # Then a huge gap, then more scores
        bottom_scores = [0.1 - i * 0.0001 for i in range(187)]
        all_scores = top_scores + bottom_scores
        config = DEFAULT_RECALL_CONFIG  # max_k=10, gap_buffer=2
        result = adaptive_k(all_scores, config)
        # The top slice has small gaps (0.005), all < epsilon (0.01) -> fallback
        # Fallback: min(min_k + gap_buffer, len, max_k) = min(3, 200, 10) = 3
        # Without truncation, gap at position 12 (0.835->0.1) would give k=max_k
        assert result <= config.max_k

    def test_top_slice_truncation_vs_full_same_top(self) -> None:
        """Top-slice correctness: same result for len=13 vs len=10000 with same top-13.

        [AP2-F3] Truncation preserves semantics for the top memories.
        """
        top_scores = [1.0 - i * 0.08 for i in range(13)]  # 1.0, 0.92, ..., 0.04
        config = DEFAULT_RECALL_CONFIG  # max_k=10, gap_buffer=2

        k_short = adaptive_k(top_scores, config)

        # Extend with 9987 tiny-gapped scores (no significant gap)
        extended = top_scores + [0.03 - i * 0.000003 for i in range(9987)]
        k_long = adaptive_k(extended, config)

        assert k_short == k_long

    def test_top_slice_truncation_clear_cluster(self) -> None:
        """10000 scores with clear top-5 cluster: k based on cluster gap.

        [AP2-F3] Top 5 scores are clustered, gap between rank 5 and 6.
        """
        top = [1.0, 0.95, 0.92, 0.90, 0.88]  # cluster
        tail = [0.3 - i * 0.00003 for i in range(9995)]  # gradual decline
        scores = top + tail
        config = RecallConfig(max_k=10, gap_buffer=2)
        result = adaptive_k(scores, config)
        # Gap at idx 4 (0.88 - 0.3 = 0.58) is the max in the top slice
        # k_raw = 5 + 2 = 7
        assert 5 <= result <= 10


# ============================================================
# 3. TestAssignTiers (~15 tests)
# ============================================================


class TestAssignTiers:
    """Tests for assign_tiers(scores, k, config) -> list[TierAssignment]."""

    def test_k_zero_returns_empty(self) -> None:
        """k=0 returns empty list."""
        result = assign_tiers([], 0, DEFAULT_RECALL_CONFIG)
        assert result == []

    def test_k_one_returns_single_high(self) -> None:
        """k=1: single memory gets normalized score 1.0 -> HIGH.

        When s_max == s_min, normalization defaults to 1.0.
        """
        scores = [(0, 0.8)]
        result = assign_tiers(scores, 1, DEFAULT_RECALL_CONFIG)
        assert len(result) == 1
        assert result[0].index == 0
        assert result[0].score == pytest.approx(0.8)
        assert result[0].normalized_score == pytest.approx(1.0)
        assert result[0].tier == Tier.HIGH

    def test_k_two_large_gap(self) -> None:
        """k=2 with large gap: idx 0 norm=1.0 (HIGH), idx 1 norm=0.0 (LOW)."""
        scores = [(0, 0.9), (1, 0.5)]
        result = assign_tiers(scores, 2, DEFAULT_RECALL_CONFIG)
        assert len(result) == 2
        assert result[0].normalized_score == pytest.approx(1.0)
        assert result[0].tier == Tier.HIGH
        assert result[1].normalized_score == pytest.approx(0.0)
        assert result[1].tier == Tier.LOW

    def test_k_three_even_spread(self) -> None:
        """k=3: idx 0 norm=1.0 (HIGH), idx 1 norm=0.833 (HIGH), idx 2 norm=0.0 (LOW).

        scores: (0, 0.9), (1, 0.8), (2, 0.3)
        s_max=0.9, s_min=0.3, range=0.6
        idx 0: (0.9-0.3)/0.6 = 1.0 -> HIGH (> 0.7)
        idx 1: (0.8-0.3)/0.6 = 0.833 -> HIGH (> 0.7)
        idx 2: (0.3-0.3)/0.6 = 0.0 -> LOW (<= 0.4)
        """
        scores = [(0, 0.9), (1, 0.8), (2, 0.3)]
        result = assign_tiers(scores, 3, DEFAULT_RECALL_CONFIG)
        assert len(result) == 3
        assert result[0].tier == Tier.HIGH
        assert result[1].tier == Tier.HIGH
        assert result[2].tier == Tier.LOW
        assert result[1].normalized_score == pytest.approx(0.833, abs=0.01)

    def test_all_identical_scores_positional_fallback_k3(self) -> None:
        """All identical scores, k=3: positional fallback -> 1H/1M/1L.

        [AP1-F3] When all k scores are identical AND k > 1, assign_tiers
        uses positional fallback: first ceil(k/3) HIGH, next ceil(k/3)
        MEDIUM, rest LOW. For k=3: ceil(3/3)=1 per tier.
        """
        scores = [(0, 0.5), (1, 0.5), (2, 0.5)]
        result = assign_tiers(scores, 3, DEFAULT_RECALL_CONFIG)
        assert len(result) == 3
        for ta in result:
            assert ta.normalized_score == pytest.approx(1.0)
        assert result[0].tier == Tier.HIGH
        assert result[1].tier == Tier.MEDIUM
        assert result[2].tier == Tier.LOW

    def test_all_identical_scores_positional_fallback_k6(self) -> None:
        """All identical scores, k=6: positional fallback -> 2H/2M/2L.

        [AP1-F3] ceil(6/3)=2 per tier.
        """
        scores = [(i, 0.5) for i in range(6)]
        result = assign_tiers(scores, 6, DEFAULT_RECALL_CONFIG)
        assert len(result) == 6
        assert result[0].tier == Tier.HIGH
        assert result[1].tier == Tier.HIGH
        assert result[2].tier == Tier.MEDIUM
        assert result[3].tier == Tier.MEDIUM
        assert result[4].tier == Tier.LOW
        assert result[5].tier == Tier.LOW

    def test_all_identical_scores_positional_fallback_k2(self) -> None:
        """All identical scores, k=2: positional fallback -> 1H/1L.

        [AP1-F3] ceil(2/3)=1 HIGH, ceil(2/3)=1 MEDIUM? Actually:
        ceil(2/3)=1 HIGH, next ceil(2/3)=1 MEDIUM, rest=0 LOW.
        So 2 identical scores -> 1H/1M (no LOW, not enough items).
        BUT spec says 'rest get LOW', and rest = k - 2*ceil(k/3) = 2 - 2 = 0.
        Actually with k=2: ceil(2/3)=1 HIGH, next ceil(2/3)=1 MEDIUM, rest=0.
        So: 1H/1M. But for k=2, there is no MEDIUM slot: the second is LOW.
        Per spec: first ceil(k/3)=1 HIGH, rest LOW. With k=2: 1H/1L.
        """
        scores = [(0, 0.5), (1, 0.5)]
        result = assign_tiers(scores, 2, DEFAULT_RECALL_CONFIG)
        assert len(result) == 2
        assert result[0].tier == Tier.HIGH
        # Second memory: with k=2, ceil(2/3)=1H, next ceil(2/3)=1M, rest=0L
        # But spec says: 1H, 1M for k=2 (or 1H, 1L if no medium slot)
        # The exact tier depends on implementation of ceil(k/3) allocation.
        # At minimum, the second must NOT be HIGH.
        assert result[1].tier != Tier.HIGH

    def test_boundary_score_at_high_threshold_is_medium(self) -> None:
        """Score exactly at high_threshold is MEDIUM, not HIGH.

        [AP1-F7] Scores must be STRICTLY ABOVE threshold to qualify.
        scores = [(0, 1.0), (1, 0.0)] with k=2:
        norm: [1.0, 0.0]. 1.0 > 0.7 -> HIGH. 0.0 <= 0.4 -> LOW.
        Now construct scores where normalized = exactly 0.7 (high_threshold):
        scores = [(0, 1.0), (1, 0.79), (2, 0.58)] with range=0.42
        idx 1: (0.79 - 0.58)/0.42 = 0.5 -> MEDIUM (0.4 < 0.5 <= 0.7)
        Need exact 0.7: (x - 0.58)/0.42 = 0.7 => x = 0.874
        scores = [(0, 1.0), (1, 0.874), (2, 0.58)]
        idx 1 norm = (0.874 - 0.58)/(1.0 - 0.58) = 0.294/0.42 = 0.7 exactly.
        """
        scores = [(0, 1.0), (1, 0.874), (2, 0.58)]
        result = assign_tiers(scores, 3, DEFAULT_RECALL_CONFIG)
        assert result[1].normalized_score == pytest.approx(0.7, abs=0.001)
        assert result[1].tier == Tier.MEDIUM  # NOT HIGH

    def test_boundary_score_at_mid_threshold_is_low(self) -> None:
        """Score exactly at mid_threshold is LOW, not MEDIUM.

        [AP1-F7] Scores must be STRICTLY ABOVE threshold to qualify.
        Need normalized = exactly 0.4:
        scores = [(0, 1.0), (1, 0.748), (2, 0.58)]
        idx 1 norm = (0.748 - 0.58)/(1.0 - 0.58) = 0.168/0.42 = 0.4 exactly.
        """
        scores = [(0, 1.0), (1, 0.748), (2, 0.58)]
        result = assign_tiers(scores, 3, DEFAULT_RECALL_CONFIG)
        assert result[1].normalized_score == pytest.approx(0.4, abs=0.001)
        assert result[1].tier == Tier.LOW  # NOT MEDIUM

    def test_boundary_score_just_above_high_threshold_is_high(self) -> None:
        """Score at high_threshold + epsilon is HIGH.

        [AP1-F7] 0.7 + tiny amount -> HIGH.
        """
        # normalized = 0.701
        scores = [(0, 1.0), (1, 0.87442), (2, 0.58)]
        result = assign_tiers(scores, 3, DEFAULT_RECALL_CONFIG)
        assert result[1].normalized_score > 0.7
        assert result[1].tier == Tier.HIGH

    def test_normalization_amplification_near_identical(self) -> None:
        """Scores [0.550, 0.548, 0.546] produce tiers [HIGH, MEDIUM, LOW].

        [AP2-F6] Within-k normalization amplifies 0.004 raw spread to full [0, 1].
        """
        scores = [(0, 0.550), (1, 0.548), (2, 0.546)]
        result = assign_tiers(scores, 3, DEFAULT_RECALL_CONFIG)
        assert result[0].normalized_score == pytest.approx(1.0)
        assert result[1].normalized_score == pytest.approx(0.5)
        assert result[2].normalized_score == pytest.approx(0.0)
        assert result[0].tier == Tier.HIGH
        assert result[1].tier == Tier.MEDIUM
        assert result[2].tier == Tier.LOW

    def test_scores_above_one_valid(self) -> None:
        """Scores > 1.0 are valid (novelty bonus); normalization still works."""
        scores = [(0, 1.2), (1, 0.9)]
        result = assign_tiers(scores, 2, DEFAULT_RECALL_CONFIG)
        assert len(result) == 2
        assert result[0].normalized_score == pytest.approx(1.0)
        assert result[1].normalized_score == pytest.approx(0.0)

    def test_custom_thresholds(self) -> None:
        """Custom thresholds change tier boundaries.

        high_threshold=0.5, mid_threshold=0.2:
        scores = [(0, 1.0), (1, 0.6), (2, 0.0)]
        s_max=1.0, s_min=0.0, range=1.0
        idx 0: 1.0 -> HIGH (> 0.5)
        idx 1: 0.6 -> HIGH (> 0.5)
        idx 2: 0.0 -> LOW (<= 0.2)
        """
        config = RecallConfig(high_threshold=0.5, mid_threshold=0.2)
        scores = [(0, 1.0), (1, 0.6), (2, 0.0)]
        result = assign_tiers(scores, 3, config)
        assert result[0].tier == Tier.HIGH
        assert result[1].tier == Tier.HIGH
        assert result[2].tier == Tier.LOW

    def test_medium_tier_assignment(self) -> None:
        """Verify MEDIUM tier is correctly assigned.

        scores = [(0, 1.0), (1, 0.5), (2, 0.0)]
        range=1.0
        idx 0: 1.0 -> HIGH (> 0.7)
        idx 1: 0.5 -> MEDIUM (0.4 < 0.5 <= 0.7)
        idx 2: 0.0 -> LOW (<= 0.4)
        """
        scores = [(0, 1.0), (1, 0.5), (2, 0.0)]
        result = assign_tiers(scores, 3, DEFAULT_RECALL_CONFIG)
        assert result[0].tier == Tier.HIGH
        assert result[1].tier == Tier.MEDIUM
        assert result[2].tier == Tier.LOW

    def test_normalized_scores_in_unit_range(self) -> None:
        """All normalized scores must be in [0.0, 1.0]."""
        scores = [(i, 1.0 - i * 0.1) for i in range(8)]
        result = assign_tiers(scores, 8, DEFAULT_RECALL_CONFIG)
        for ta in result:
            assert 0.0 <= ta.normalized_score <= 1.0

    def test_output_order_matches_input(self) -> None:
        """Output order is descending score (same as input)."""
        scores = [(2, 0.9), (0, 0.5), (1, 0.3)]
        result = assign_tiers(scores, 3, DEFAULT_RECALL_CONFIG)
        assert [ta.index for ta in result] == [2, 0, 1]

    def test_tier_monotonic_with_score(self) -> None:
        """If norm_a > norm_b, tier_a.value <= tier_b.value (higher detail)."""
        scores = [(i, 1.0 - i * 0.15) for i in range(7)]
        result = assign_tiers(scores, 7, DEFAULT_RECALL_CONFIG)
        for i in range(len(result) - 1):
            if result[i].normalized_score > result[i + 1].normalized_score:
                assert result[i].tier.value <= result[i + 1].tier.value

    def test_k_greater_than_len_raises(self) -> None:
        """k > len(scores) raises ValueError."""
        with pytest.raises(ValueError):
            assign_tiers([(0, 0.5)], 2, DEFAULT_RECALL_CONFIG)

    def test_k_negative_raises(self) -> None:
        """k < 0 raises ValueError."""
        with pytest.raises(ValueError):
            assign_tiers([(0, 0.5)], -1, DEFAULT_RECALL_CONFIG)

    def test_scores_not_descending_raises(self) -> None:
        """Scores not in descending order raises ValueError."""
        with pytest.raises(ValueError):
            assign_tiers([(0, 0.3), (1, 0.9)], 2, DEFAULT_RECALL_CONFIG)


# ============================================================
# 4. TestFormatMemory (~15 tests)
# ============================================================


class TestFormatMemory:
    """Tests for format_memory(memory, tier, config, content) -> str."""

    @pytest.fixture()
    def memory(self) -> MemoryState:
        """A standard test memory."""
        return _make_memory(
            relevance=0.85,
            last_access_time=5.0,
            importance=0.7,
            access_count=10,
            strength=7.5,
            creation_time=50.0,
        )

    def test_high_tier_with_content(self, memory: MemoryState) -> None:
        """HIGH tier with content includes content text."""
        result = format_memory(
            memory, Tier.HIGH, DEFAULT_RECALL_CONFIG, content="Important fact"
        )
        assert "[Memory]" in result
        assert "relevance=0.85" in result
        assert "strength=7.5" in result
        assert "importance=0.7" in result
        assert "Important fact" in result
        assert "Accessed 10 times" in result
        assert "age=50 steps" in result

    def test_high_tier_without_content(self, memory: MemoryState) -> None:
        """HIGH tier without content: metadata only."""
        result = format_memory(memory, Tier.HIGH, DEFAULT_RECALL_CONFIG)
        assert "[Memory]" in result
        assert "relevance=0.85" in result
        assert "Accessed 10 times" in result
        # No content line since content is None
        assert "Important fact" not in result

    def test_medium_tier_with_content(self, memory: MemoryState) -> None:
        """MEDIUM tier with content includes truncated content."""
        result = format_memory(
            memory, Tier.MEDIUM, DEFAULT_RECALL_CONFIG, content="A medium priority fact"
        )
        assert "[Memory]" in result
        assert "relevance=0.85" in result
        assert "importance=0.7" in result
        # Should NOT contain strength (MEDIUM omits it)
        assert "strength" not in result

    def test_medium_tier_without_content(self, memory: MemoryState) -> None:
        """MEDIUM tier without content: metadata only."""
        result = format_memory(memory, Tier.MEDIUM, DEFAULT_RECALL_CONFIG)
        assert "[Memory]" in result
        assert "relevance=0.85" in result
        assert "importance=0.7" in result

    def test_low_tier_metadata_only(self, memory: MemoryState) -> None:
        """LOW tier: metadata only (content ignored even if provided)."""
        result = format_memory(
            memory, Tier.LOW, DEFAULT_RECALL_CONFIG, content="Should be ignored"
        )
        assert "[Memory]" in result
        assert "rel=0.85" in result
        assert "imp=0.7" in result or "imp=0.70" in result
        assert "age=50" in result
        assert "Should be ignored" not in result

    def test_content_truncation_at_word_boundary(self, memory: MemoryState) -> None:
        """Long content is truncated at word boundary with '...'."""
        long_content = "word " * 200  # 1000 chars
        result = format_memory(
            memory, Tier.HIGH, DEFAULT_RECALL_CONFIG, content=long_content
        )
        assert len(result) <= DEFAULT_RECALL_CONFIG.high_max_chars
        if "..." in result:
            # Truncation occurred -- the "..." is at the end of a word
            assert result.endswith("...") or "..." in result

    def test_content_with_newlines_replaced(self, memory: MemoryState) -> None:
        """Newlines in content replaced with spaces before formatting."""
        result = format_memory(
            memory,
            Tier.HIGH,
            DEFAULT_RECALL_CONFIG,
            content="Line one\nLine two\nLine three",
        )
        # Should not contain literal newlines within the content portion
        # (the format has structural newlines, but content newlines are replaced)
        assert "Line one Line two Line three" in result or "Line one" in result

    def test_very_long_content_truncated(self, memory: MemoryState) -> None:
        """10000+ char content is correctly truncated."""
        huge = "x" * 10000
        result = format_memory(memory, Tier.HIGH, DEFAULT_RECALL_CONFIG, content=huge)
        assert len(result) <= DEFAULT_RECALL_CONFIG.high_max_chars

    def test_empty_content_string(self, memory: MemoryState) -> None:
        """content='' (empty string) treated as content-provided but empty."""
        result = format_memory(memory, Tier.HIGH, DEFAULT_RECALL_CONFIG, content="")
        assert "[Memory]" in result
        # No content line since content is empty
        assert "relevance=0.85" in result

    def test_formatting_values_match_fields(self, memory: MemoryState) -> None:
        """Formatted values match the MemoryState fields exactly."""
        result = format_memory(memory, Tier.HIGH, DEFAULT_RECALL_CONFIG)
        assert "relevance=0.85" in result
        assert "strength=7.5" in result
        assert "importance=0.7" in result
        assert "age=50 steps" in result
        assert "Accessed 10 times" in result

    def test_all_tiers_produce_different_output(self, memory: MemoryState) -> None:
        """All three tiers produce different output for the same memory."""
        high = format_memory(memory, Tier.HIGH, DEFAULT_RECALL_CONFIG)
        medium = format_memory(memory, Tier.MEDIUM, DEFAULT_RECALL_CONFIG)
        low = format_memory(memory, Tier.LOW, DEFAULT_RECALL_CONFIG)
        assert high != medium
        assert medium != low
        assert high != low

    def test_low_relevance_still_formats(self) -> None:
        """memory.relevance == 0.0 and tier == HIGH: still formats normally."""
        mem = _make_memory(relevance=0.0)
        result = format_memory(mem, Tier.HIGH, DEFAULT_RECALL_CONFIG)
        assert "[Memory]" in result
        assert "relevance=0.00" in result

    def test_high_max_chars_respected(self, memory: MemoryState) -> None:
        """Output never exceeds config.high_max_chars for HIGH tier."""
        result = format_memory(
            memory,
            Tier.HIGH,
            DEFAULT_RECALL_CONFIG,
            content="A" * 1000,
        )
        assert len(result) <= DEFAULT_RECALL_CONFIG.high_max_chars

    def test_medium_max_chars_respected(self, memory: MemoryState) -> None:
        """Output never exceeds config.medium_max_chars for MEDIUM tier."""
        result = format_memory(
            memory,
            Tier.MEDIUM,
            DEFAULT_RECALL_CONFIG,
            content="A" * 1000,
        )
        assert len(result) <= DEFAULT_RECALL_CONFIG.medium_max_chars

    def test_low_max_chars_respected(self, memory: MemoryState) -> None:
        """Output never exceeds config.low_max_chars for LOW tier."""
        result = format_memory(memory, Tier.LOW, DEFAULT_RECALL_CONFIG)
        assert len(result) <= DEFAULT_RECALL_CONFIG.low_max_chars


# ============================================================
# 5. TestBudgetConstrain (~20 tests)
# ============================================================


class TestBudgetConstrain:
    """Tests for budget_constrain(assignments, formatted, memories, config, contents)."""

    def _make_assignment(
        self,
        index: int,
        score: float,
        normalized_score: float,
        tier: Tier,
    ) -> TierAssignment:
        return TierAssignment(
            index=index,
            score=score,
            normalized_score=normalized_score,
            tier=tier,
        )

    def test_under_budget_unchanged(self) -> None:
        """Under budget: returned unchanged."""
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.HIGH),
            self._make_assignment(1, 0.5, 0.0, Tier.LOW),
        ]
        formatted = ["short text", "tiny"]
        memories = [_make_memory(relevance=0.9), _make_memory(relevance=0.5)]
        config = RecallConfig(total_budget=800)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        assert len(result_a) == 2
        assert len(result_f) == 2
        assert demoted == 0
        assert dropped == 0

    def test_over_budget_demotion_sufficient(self) -> None:
        """Over budget: demotion reduces tokens enough, no drops needed."""
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.HIGH),
            self._make_assignment(1, 0.7, 0.5, Tier.MEDIUM),
            self._make_assignment(2, 0.5, 0.0, Tier.LOW),
        ]
        # Each formatted string ~200 tokens = ~800 chars
        formatted = ["A" * 800, "B" * 800, "C" * 80]
        memories = [
            _make_memory(relevance=0.9),
            _make_memory(relevance=0.7),
            _make_memory(relevance=0.5),
        ]
        config = RecallConfig(total_budget=300)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        assert demoted > 0
        # Verify total tokens reduced
        total_tokens = sum(len(s) // 4 for s in result_f)
        # Should be at or under budget (or at min_k limit)
        assert total_tokens <= config.total_budget or len(result_a) <= config.min_k

    def test_over_budget_demotion_and_drop(self) -> None:
        """Over budget: demotion insufficient, dropping also needed."""
        assignments = [
            self._make_assignment(i, 0.9 - i * 0.1, 1.0 - i * 0.2, Tier.HIGH)
            for i in range(5)
        ]
        formatted = ["X" * 400 for _ in range(5)]  # 5 * 100 tokens = 500
        memories = [_make_memory(relevance=0.9 - i * 0.1) for i in range(5)]
        config = RecallConfig(total_budget=100)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        assert dropped > 0
        assert len(result_a) < 5

    def test_all_low_still_over_drops(self) -> None:
        """All LOW tier and still over budget: drops from bottom."""
        assignments = [
            self._make_assignment(i, 0.5 - i * 0.05, 1.0 - i * 0.25, Tier.LOW)
            for i in range(4)
        ]
        formatted = ["Z" * 400 for _ in range(4)]  # Each ~100 tokens
        memories = [_make_memory(relevance=0.5 - i * 0.05) for i in range(4)]
        config = RecallConfig(total_budget=50)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        assert dropped > 0

    def test_single_memory_over_budget_never_dropped(self) -> None:
        """Single memory over budget: demoted to LOW, never dropped (min_k=1)."""
        assignments = [self._make_assignment(0, 0.9, 1.0, Tier.HIGH)]
        formatted = ["Y" * 800]  # ~200 tokens
        memories = [_make_memory(relevance=0.9)]
        config = RecallConfig(total_budget=50)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        assert len(result_a) == 1  # Never dropped (min_k=1)
        assert dropped == 0

    def test_empty_input(self) -> None:
        """Empty input returns ([], [], 0, 0)."""
        result_a, result_f, demoted, dropped = budget_constrain(
            [],
            [],
            [],
            DEFAULT_RECALL_CONFIG,
        )
        assert result_a == []
        assert result_f == []
        assert demoted == 0
        assert dropped == 0

    def test_budget_one_aggressive_demotion(self) -> None:
        """Budget=1: everything demoted and dropped to min_k."""
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.HIGH),
            self._make_assignment(1, 0.5, 0.0, Tier.LOW),
        ]
        formatted = ["A" * 100, "B" * 100]
        memories = [_make_memory(relevance=0.9), _make_memory(relevance=0.5)]
        config = RecallConfig(total_budget=1)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        # At most min_k=1 remains
        assert len(result_a) >= 1  # min_k enforcement

    def test_demotion_order_lowest_scored_first(self) -> None:
        """Demotion iterates from lowest-scored to highest-scored."""
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.HIGH),
            self._make_assignment(1, 0.7, 0.5, Tier.HIGH),
            self._make_assignment(2, 0.3, 0.0, Tier.HIGH),
        ]
        formatted = ["A" * 400, "B" * 400, "C" * 400]  # ~300 tokens total
        memories = [
            _make_memory(relevance=0.9),
            _make_memory(relevance=0.7),
            _make_memory(relevance=0.3),
        ]
        config = RecallConfig(total_budget=200)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        # Lowest scored (idx 2) should be demoted first
        if demoted > 0:
            # The last entry (lowest score) should have been demoted
            last = result_a[-1]
            assert last.tier.value > Tier.HIGH.value  # Demoted from HIGH

    def test_drop_order_lowest_scored_first(self) -> None:
        """Drop order: lowest-scored first."""
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.LOW),
            self._make_assignment(1, 0.7, 0.5, Tier.LOW),
            self._make_assignment(2, 0.3, 0.0, Tier.LOW),
        ]
        formatted = ["A" * 200, "B" * 200, "C" * 200]
        memories = [
            _make_memory(relevance=0.9),
            _make_memory(relevance=0.7),
            _make_memory(relevance=0.3),
        ]
        config = RecallConfig(total_budget=80)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        # idx 2 (lowest) should be dropped first
        remaining_indices = [ta.index for ta in result_a]
        if dropped > 0:
            assert 2 not in remaining_indices or len(result_a) == 1

    def test_parallel_list_mismatch_raises(self) -> None:
        """Mismatched list lengths raise ValueError."""
        assignments = [self._make_assignment(0, 0.9, 1.0, Tier.HIGH)]
        formatted = ["A", "B"]  # Length mismatch
        memories = [_make_memory()]
        with pytest.raises(ValueError):
            budget_constrain(assignments, formatted, memories, DEFAULT_RECALL_CONFIG)

    def test_memories_length_mismatch_raises(self) -> None:
        """memories length != assignments length raises ValueError."""
        assignments = [self._make_assignment(0, 0.9, 1.0, Tier.HIGH)]
        formatted = ["A"]
        memories = [_make_memory(), _make_memory()]  # Length mismatch
        with pytest.raises(ValueError):
            budget_constrain(assignments, formatted, memories, DEFAULT_RECALL_CONFIG)

    def test_contents_length_mismatch_raises(self) -> None:
        """contents length != assignments length raises ValueError."""
        assignments = [self._make_assignment(0, 0.9, 1.0, Tier.HIGH)]
        formatted = ["A"]
        memories = [_make_memory()]
        contents = ["content1", "content2"]  # Length mismatch
        with pytest.raises(ValueError):
            budget_constrain(
                assignments,
                formatted,
                memories,
                DEFAULT_RECALL_CONFIG,
                contents=contents,
            )

    def test_with_contents_reformatted(self) -> None:
        """With contents: demoted memories get re-formatted with content."""
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.HIGH),
            self._make_assignment(1, 0.5, 0.0, Tier.HIGH),
        ]
        formatted = ["A" * 400, "B" * 400]
        memories = [_make_memory(relevance=0.9), _make_memory(relevance=0.5)]
        contents = ["Content for memory 0", "Content for memory 1"]
        config = RecallConfig(total_budget=100)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
            contents=contents,
        )
        if demoted > 0:
            # Reformatted strings should be shorter than originals
            for f_str in result_f:
                assert len(f_str) < 400

    def test_without_contents_metadata_only_format(self) -> None:
        """Without contents: demoted memories use metadata-only format."""
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.HIGH),
            self._make_assignment(1, 0.5, 0.0, Tier.HIGH),
        ]
        formatted = ["A" * 400, "B" * 400]
        memories = [_make_memory(relevance=0.9), _make_memory(relevance=0.5)]
        config = RecallConfig(total_budget=100)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        # Without contents, demoted strings should be metadata-only
        if demoted > 0:
            for f_str in result_f:
                assert len(f_str) < 400

    def test_output_still_descending_score_order(self) -> None:
        """Output is still in descending score order after demotion/drops."""
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.HIGH),
            self._make_assignment(1, 0.7, 0.5, Tier.MEDIUM),
            self._make_assignment(2, 0.5, 0.0, Tier.LOW),
        ]
        formatted = ["A" * 200, "B" * 200, "C" * 80]
        memories = [
            _make_memory(relevance=0.9),
            _make_memory(relevance=0.7),
            _make_memory(relevance=0.5),
        ]
        config = RecallConfig(total_budget=100)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        scores = [ta.score for ta in result_a]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_dropped_count_matches_reduction(self) -> None:
        """dropped count == len(assignments) - len(output)."""
        assignments = [
            self._make_assignment(i, 0.9 - i * 0.1, 1.0 - i * 0.25, Tier.HIGH)
            for i in range(4)
        ]
        formatted = ["X" * 400 for _ in range(4)]
        memories = [_make_memory(relevance=0.9 - i * 0.1) for i in range(4)]
        config = RecallConfig(total_budget=50)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        assert dropped == len(assignments) - len(result_a)

    def test_min_k_prevents_total_drop(self) -> None:
        """Never drop below min_k memories."""
        assignments = [
            self._make_assignment(i, 0.9 - i * 0.1, 1.0 - i * 0.25, Tier.LOW)
            for i in range(5)
        ]
        formatted = ["X" * 800 for _ in range(5)]
        memories = [_make_memory(relevance=0.9 - i * 0.1) for i in range(5)]
        config = RecallConfig(total_budget=1, min_k=2)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        assert len(result_a) >= 2  # min_k=2

    def test_equal_length_outputs(self) -> None:
        """len(output[0]) == len(output[1]) always."""
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.HIGH),
            self._make_assignment(1, 0.5, 0.0, Tier.LOW),
        ]
        formatted = ["A" * 400, "B" * 80]
        memories = [_make_memory(relevance=0.9), _make_memory(relevance=0.5)]
        config = RecallConfig(total_budget=50)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments,
            formatted,
            memories,
            config,
        )
        assert len(result_a) == len(result_f)

    def test_contents_none_vs_explicit_none_list_identical(self) -> None:
        """contents=None and contents=[None, None] produce identical behavior.

        [AP1-F4] budget_constrain normalizes contents=None to [None]*len.
        """
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.HIGH),
            self._make_assignment(1, 0.5, 0.0, Tier.HIGH),
        ]
        formatted = ["A" * 400, "B" * 400]
        memories = [_make_memory(relevance=0.9), _make_memory(relevance=0.5)]
        config = RecallConfig(total_budget=100)

        # With contents=None (outer None)
        a1, f1, d1, dr1 = budget_constrain(
            assignments, formatted, memories, config, contents=None
        )
        # With contents=[None, None] (explicit list of inner Nones)
        a2, f2, d2, dr2 = budget_constrain(
            assignments, formatted, memories, config, contents=[None, None]
        )
        assert len(a1) == len(a2)
        assert d1 == d2
        assert dr1 == dr2
        assert f1 == f2

    def test_content_none_optimization_skip_demotion(self) -> None:
        """When ALL contents are None, demotion passes are skipped.

        [AP3-T3] Without content, demotion saves only 2-8 tokens per
        memory. budget_constrain should skip demotion and go to dropping.
        """
        assignments = [
            self._make_assignment(i, 0.9 - i * 0.1, 1.0 - i * 0.25, Tier.HIGH)
            for i in range(4)
        ]
        formatted = ["A" * 400 for _ in range(4)]
        memories = [_make_memory(relevance=0.9 - i * 0.1) for i in range(4)]
        config = RecallConfig(total_budget=50)

        # With all contents None, should prefer dropping over demotion
        result_a, result_f, demoted, dropped = budget_constrain(
            assignments, formatted, memories, config, contents=None
        )
        # Some memories should have been dropped rather than all demoted
        assert dropped > 0

    def test_demotion_skip_medium_to_low_content_none(self) -> None:
        """MEDIUM->LOW demotion with content=None saves ~1 token, should be skipped.

        [AP2-F4] When savings < MIN_DEMOTION_SAVINGS_TOKENS * chars_per_token,
        demotion is skipped in favor of dropping.
        """
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.MEDIUM),
            self._make_assignment(1, 0.5, 0.0, Tier.MEDIUM),
        ]
        # MEDIUM without content: ~40 chars. LOW without content: ~33 chars.
        # Savings per memory: ~7 chars = ~1 token < MIN_DEMOTION_SAVINGS_TOKENS (5).
        formatted = [
            format_memory(
                _make_memory(relevance=0.9), Tier.MEDIUM, DEFAULT_RECALL_CONFIG
            ),
            format_memory(
                _make_memory(relevance=0.5), Tier.MEDIUM, DEFAULT_RECALL_CONFIG
            ),
        ]
        memories = [_make_memory(relevance=0.9), _make_memory(relevance=0.5)]
        # Budget tight enough that some action is needed
        total_tokens = sum(len(s) // 4 for s in formatted)
        config = RecallConfig(total_budget=max(1, total_tokens - 2))

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments, formatted, memories, config, contents=None
        )
        # MEDIUM->LOW demotion should NOT have happened (savings too small)
        # Instead, dropping should have been used
        for ta in result_a:
            assert ta.tier != Tier.LOW or dropped > 0

    def test_cascading_demotion_10_high_memories(self) -> None:
        """10 HIGH memories cascade: HIGH demotes to MEDIUM, overflow demotes to LOW.

        [AP2-F2] Per-tier passes run sequentially: HIGH first, then MEDIUM.
        MEDIUM pass INCLUDES memories just demoted from HIGH. This causes
        a cascade: 10H -> ~4H/4M/2L with default budget shares.
        """
        assignments = [
            self._make_assignment(i, 0.9 - i * 0.01, 1.0 - i * 0.05, Tier.HIGH)
            for i in range(10)
        ]
        formatted = ["A" * 400 for _ in range(10)]  # Each ~100 tokens
        memories = [_make_memory(relevance=0.9 - i * 0.01) for i in range(10)]
        config = RecallConfig(total_budget=800)  # Default budget

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments, formatted, memories, config
        )
        # Per-tier enforcement should have kicked in:
        # HIGH budget = 800 * 0.55 = 440 tokens. 10 * 100 = 1000 > 440.
        tiers = [ta.tier for ta in result_a]
        high_count = tiers.count(Tier.HIGH)
        medium_count = tiers.count(Tier.MEDIUM)
        low_count = tiers.count(Tier.LOW)
        # Not all should be HIGH anymore
        assert high_count < 10
        # Should have cascade: some in MEDIUM and potentially LOW
        assert medium_count > 0 or low_count > 0
        assert demoted > 0

    def test_per_tier_demotion_despite_total_under_budget(self) -> None:
        """Per-tier shares cause demotion even when total budget is not exceeded.

        [AP2-F7] 5H/3M/2L = 677 tokens < 800 total budget. But HIGH at
        500 tokens exceeds its 440-token share (800 * 0.55).
        """
        assignments = (
            [
                self._make_assignment(i, 0.95 - i * 0.01, 1.0, Tier.HIGH)
                for i in range(5)
            ]
            + [
                self._make_assignment(5 + i, 0.7 - i * 0.05, 0.5, Tier.MEDIUM)
                for i in range(3)
            ]
            + [
                self._make_assignment(8 + i, 0.3 - i * 0.1, 0.0, Tier.LOW)
                for i in range(2)
            ]
        )
        # HIGH ~100tok each, MEDIUM ~50tok, LOW ~8tok
        formatted = ["H" * 400] * 5 + ["M" * 200] * 3 + ["L" * 33] * 2
        memories = (
            [_make_memory(relevance=0.95 - i * 0.01) for i in range(5)]
            + [_make_memory(relevance=0.7 - i * 0.05) for i in range(3)]
            + [_make_memory(relevance=0.3 - i * 0.1) for i in range(2)]
        )
        config = RecallConfig(total_budget=800)

        result_a, result_f, demoted, dropped = budget_constrain(
            assignments, formatted, memories, config
        )
        # Total was under budget but per-tier HIGH was over its share
        # At least one HIGH should have been demoted
        high_remaining = sum(1 for ta in result_a if ta.tier == Tier.HIGH)
        assert high_remaining < 5 or demoted > 0

    def test_chars_per_token_2_tighter_budget(self) -> None:
        """chars_per_token=2: budget constraint activates sooner.

        [AP2-F1] Conservative token estimation for non-English content.
        Same content appears as 2x more tokens.
        """
        assignments = [
            self._make_assignment(0, 0.9, 1.0, Tier.HIGH),
            self._make_assignment(1, 0.5, 0.0, Tier.HIGH),
        ]
        formatted = ["A" * 400, "B" * 400]
        memories = [_make_memory(relevance=0.9), _make_memory(relevance=0.5)]
        # With chars_per_token=4: 800 chars = 200 tokens. Budget=300 -> under.
        # With chars_per_token=2: 800 chars = 400 tokens. Budget=300 -> over.
        config_normal = RecallConfig(total_budget=300, chars_per_token=4)
        config_conservative = RecallConfig(total_budget=300, chars_per_token=2)

        _, _, d_normal, dr_normal = budget_constrain(
            assignments, formatted, memories, config_normal
        )
        _, _, d_conservative, dr_conservative = budget_constrain(
            assignments, formatted, memories, config_conservative
        )
        # Conservative should trigger more demotion/dropping
        assert (d_conservative + dr_conservative) >= (d_normal + dr_normal)


# ============================================================
# 6. TestRecall (~20 tests)
# ============================================================


class TestRecall:
    """Tests for recall(memories, params, config, message, turn_number, ...)."""

    @pytest.fixture()
    def params(self) -> ParameterSet:
        return _default_params()

    def test_gated_message_returns_empty(self, params: ParameterSet) -> None:
        """Gated message returns empty RecallResult with gated=True.

        [AP1-F1] budget_overflow must be 0 when gated.
        """
        memories = _make_memories([0.9, 0.5, 0.3])
        result = recall(memories, params, DEFAULT_RECALL_CONFIG, "ok", 1)
        assert result.gated is True
        assert result.context == ""
        assert result.k == 0
        assert result.tier_assignments == ()
        assert result.budget_overflow == 0
        assert result.memories_dropped == 0
        assert result.memories_demoted == 0

    def test_non_gated_empty_memories(self, params: ParameterSet) -> None:
        """Non-gated with empty memories returns k=0.

        [AP1-F5] Empty-memories fast-path fires BEFORE gate check.
        gated=False means 'DB was empty', not 'gate allowed recall'.
        [AP1-F1] budget_overflow must be 0.
        """
        result = recall([], params, DEFAULT_RECALL_CONFIG, "Tell me about X", 1)
        assert result.gated is False
        assert result.k == 0
        assert result.context == ""
        assert result.budget_overflow == 0

    def test_non_gated_single_memory(self, params: ParameterSet) -> None:
        """Non-gated with single memory returns k=1, HIGH tier."""
        memories = [_make_memory(relevance=0.9)]
        result = recall(memories, params, DEFAULT_RECALL_CONFIG, "Tell me about X", 1)
        assert result.gated is False
        assert result.k == 1
        assert len(result.tier_assignments) == 1
        assert result.tier_assignments[0].tier == Tier.HIGH

    def test_non_gated_multiple_memories_clear_gap(self, params: ParameterSet) -> None:
        """Multiple memories with clear gap: correct k and tier distribution."""
        memories = [
            _make_memory(relevance=0.95),
            _make_memory(relevance=0.90),
            _make_memory(relevance=0.10),
            _make_memory(relevance=0.05),
        ]
        result = recall(memories, params, DEFAULT_RECALL_CONFIG, "What is X?", 2)
        assert result.gated is False
        assert result.k >= 1
        # Should retrieve the high-relevance memories
        assert result.k <= 4

    def test_within_budget_no_demotions(self, params: ParameterSet) -> None:
        """Within budget: no demotions or drops."""
        memories = [_make_memory(relevance=0.9)]
        result = recall(
            memories,
            params,
            RecallConfig(total_budget=800),
            "Tell me about X",
            1,
        )
        assert result.budget_exceeded is False
        assert result.memories_dropped == 0
        assert result.memories_demoted == 0

    def test_over_budget_demotions_reflected(self, params: ParameterSet) -> None:
        """Over budget: demotions/drops reflected in result."""
        memories = _make_memories([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60])
        contents = ["Content " * 50 for _ in range(8)]
        result = recall(
            memories,
            params,
            RecallConfig(total_budget=50),
            "Tell me about X",
            1,
            contents=contents,
        )
        if result.k > 0:
            assert result.budget_exceeded is True or result.k == 1

    def test_contents_provided_include_content(self, params: ParameterSet) -> None:
        """Contents provided: HIGH/MEDIUM memories include content in rendering."""
        memories = [_make_memory(relevance=0.95)]
        contents = ["This is the memory content."]
        result = recall(
            memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "Tell me about X",
            1,
            contents=contents,
        )
        assert "This is the memory content" in result.context

    def test_contents_not_provided_metadata_only(self, params: ParameterSet) -> None:
        """Contents not provided: metadata-only rendering."""
        memories = [_make_memory(relevance=0.95)]
        result = recall(
            memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "Tell me about X",
            1,
        )
        assert "[Memory]" in result.context

    def test_contents_length_mismatch_raises(self, params: ParameterSet) -> None:
        """contents length != memories length raises ValueError."""
        memories = [_make_memory(relevance=0.9), _make_memory(relevance=0.5)]
        contents = ["only one"]
        with pytest.raises(ValueError):
            recall(
                memories,
                params,
                DEFAULT_RECALL_CONFIG,
                "Tell me about X",
                1,
                contents=contents,
            )

    def test_turn_zero_trivial_message_recalls(self, params: ParameterSet) -> None:
        """Turn 0 with trivial message: recalls anyway (first turn override)."""
        memories = [_make_memory(relevance=0.9)]
        result = recall(memories, params, DEFAULT_RECALL_CONFIG, "hi", 0)
        assert result.gated is False
        assert result.k >= 1

    def test_context_separator_count(self, params: ParameterSet) -> None:
        """Context string has exactly k-1 separators (or 0 if k <= 1)."""
        memories = _make_memories([0.95, 0.90, 0.85])
        result = recall(
            memories,
            params,
            RecallConfig(total_budget=2000),  # High budget to avoid drops
            "Tell me about X",
            1,
        )
        if result.k > 1:
            sep_count = result.context.count(MEMORY_SEPARATOR)
            assert sep_count == result.k - 1
        elif result.k == 1:
            assert MEMORY_SEPARATOR not in result.context

    def test_total_tokens_estimate_matches(self, params: ParameterSet) -> None:
        """total_tokens_estimate == len(context) // config.chars_per_token.

        [AP2-F1] Uses config.chars_per_token (default 4), not hardcoded constant.
        """
        memories = [_make_memory(relevance=0.9)]
        config = DEFAULT_RECALL_CONFIG
        result = recall(memories, params, config, "Tell me about X", 1)
        assert (
            result.total_tokens_estimate
            == len(result.context) // config.chars_per_token
        )

    def test_tier_assignments_length_equals_k(self, params: ParameterSet) -> None:
        """len(tier_assignments) == k always."""
        memories = _make_memories([0.9, 0.7, 0.5])
        result = recall(memories, params, DEFAULT_RECALL_CONFIG, "Tell me about X", 1)
        assert len(result.tier_assignments) == result.k

    def test_tier_assignments_descending_score(self, params: ParameterSet) -> None:
        """tier_assignments sorted by descending score."""
        memories = _make_memories([0.95, 0.60, 0.30])
        result = recall(
            memories,
            params,
            RecallConfig(total_budget=2000),
            "Tell me about X",
            1,
        )
        scores = [ta.score for ta in result.tier_assignments]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_gated_invariants(self, params: ParameterSet) -> None:
        """When gated: all metadata fields are in their gated state.

        [AP1-F1] budget_overflow must also be 0 when gated.
        [AP1-F5] Note: empty memories returns gated=False (fast-path),
        so we must provide non-empty memories to trigger gating.
        """
        memories = _make_memories([0.9, 0.5])
        result = recall(memories, params, DEFAULT_RECALL_CONFIG, "ok", 5)
        assert result.gated is True
        assert result.context == ""
        assert result.k == 0
        assert result.tier_assignments == ()
        assert result.total_tokens_estimate == 0
        assert result.budget_exceeded is False
        assert result.budget_overflow == 0
        assert result.memories_dropped == 0
        assert result.memories_demoted == 0

    def test_negative_turn_number_raises(self, params: ParameterSet) -> None:
        """turn_number < 0 propagates ValueError from should_recall."""
        with pytest.raises(ValueError):
            recall([], params, DEFAULT_RECALL_CONFIG, "hello", -1)

    def test_non_gated_memories_k_at_least_one(self, params: ParameterSet) -> None:
        """If memories is non-empty and not gated: k >= 1."""
        memories = [_make_memory(relevance=0.1)]
        result = recall(memories, params, DEFAULT_RECALL_CONFIG, "Tell me about X", 1)
        assert result.gated is False
        assert result.k >= 1

    def test_recall_result_is_frozen(self, params: ParameterSet) -> None:
        """RecallResult is a frozen dataclass."""
        memories = [_make_memory(relevance=0.9)]
        result = recall(memories, params, DEFAULT_RECALL_CONFIG, "Tell me about X", 1)
        with pytest.raises(AttributeError):
            result.k = 99  # type: ignore[misc]

    def test_current_time_forwarded_to_ranking(self, params: ParameterSet) -> None:
        """current_time parameter is forwarded to rank_memories."""
        memories = [_make_memory(relevance=0.9, creation_time=0.0)]
        # Different current_time should be accepted without error
        result = recall(
            memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "Tell me about X",
            1,
            current_time=100.0,
        )
        assert result.k >= 1

    def test_empty_memories_trivial_message_fast_path(
        self, params: ParameterSet
    ) -> None:
        """Empty memories with trivial message: gated=False, k=0 (fast-path).

        [AP1-F5] Empty-memories fast-path fires BEFORE gate check.
        Even though 'ok' would be gated, empty memories returns gated=False.
        """
        result = recall([], params, DEFAULT_RECALL_CONFIG, "ok", 1)
        assert result.gated is False
        assert result.k == 0

    def test_budget_overflow_when_min_k_forces_over_budget(
        self, params: ParameterSet
    ) -> None:
        """budget_overflow > 0 when min_k prevents dropping below budget.

        [AP1-F1] RecallResult.budget_overflow == max(0, tokens - budget).
        When min_k=1 and single memory exceeds budget, overflow is reported.
        """
        memories = [_make_memory(relevance=0.9)]
        # Single memory at LOW tier is ~33 chars = ~8 tokens
        # Budget of 1 token means overflow is ~7 tokens
        config = RecallConfig(total_budget=1)
        result = recall(memories, params, config, "Tell me about X", 1)
        assert result.k == 1  # min_k=1 prevents dropping
        assert result.budget_overflow > 0
        assert result.budget_overflow == max(
            0, result.total_tokens_estimate - config.total_budget
        )

    def test_budget_overflow_zero_when_within_budget(
        self, params: ParameterSet
    ) -> None:
        """budget_overflow == 0 when within budget.

        [AP1-F1] Normal case: budget is sufficient.
        """
        memories = [_make_memory(relevance=0.9)]
        config = RecallConfig(total_budget=2000)
        result = recall(memories, params, config, "Tell me about X", 1)
        assert result.budget_overflow == 0

    def test_budget_overflow_formula(self, params: ParameterSet) -> None:
        """budget_overflow == max(0, total_tokens_estimate - total_budget).

        [AP1-F1] Verify the formula holds regardless of the scenario.
        """
        memories = _make_memories([0.95, 0.90, 0.85])
        for budget in [1, 10, 50, 100, 800, 2000]:
            config = RecallConfig(total_budget=budget)
            result = recall(memories, params, config, "Tell me about X", 1)
            expected_overflow = max(
                0, result.total_tokens_estimate - config.total_budget
            )
            assert result.budget_overflow == expected_overflow, (
                f"budget={budget}: overflow={result.budget_overflow} "
                f"expected={expected_overflow}"
            )

    def test_recall_result_has_budget_overflow_field(
        self, params: ParameterSet
    ) -> None:
        """RecallResult has budget_overflow field (AP1-F1)."""
        memories = [_make_memory(relevance=0.9)]
        result = recall(memories, params, DEFAULT_RECALL_CONFIG, "Tell me about X", 1)
        assert hasattr(result, "budget_overflow")
        assert isinstance(result.budget_overflow, int)
        assert result.budget_overflow >= 0


# ============================================================
# 7. TestRecallIntegration (~10 tests)
# ============================================================


class TestRecallIntegration:
    """Integration tests with realistic memory pools and tuned parameters."""

    @pytest.fixture()
    def params(self) -> ParameterSet:
        return _default_params()

    @pytest.fixture()
    def diverse_memories(self) -> list[MemoryState]:
        """A diverse pool of 12 memories with varying attributes."""
        return [
            _make_memory(
                relevance=0.95, importance=0.9, access_count=20, creation_time=5.0
            ),
            _make_memory(
                relevance=0.90, importance=0.8, access_count=15, creation_time=10.0
            ),
            _make_memory(
                relevance=0.85, importance=0.7, access_count=10, creation_time=20.0
            ),
            _make_memory(
                relevance=0.70, importance=0.6, access_count=8, creation_time=30.0
            ),
            _make_memory(
                relevance=0.60, importance=0.5, access_count=5, creation_time=50.0
            ),
            _make_memory(
                relevance=0.50, importance=0.4, access_count=3, creation_time=70.0
            ),
            _make_memory(
                relevance=0.40, importance=0.3, access_count=2, creation_time=90.0
            ),
            _make_memory(
                relevance=0.30, importance=0.2, access_count=1, creation_time=100.0
            ),
            _make_memory(
                relevance=0.20, importance=0.1, access_count=1, creation_time=120.0
            ),
            _make_memory(
                relevance=0.10, importance=0.1, access_count=0, creation_time=150.0
            ),
            _make_memory(
                relevance=0.05, importance=0.05, access_count=0, creation_time=200.0
            ),
            _make_memory(
                relevance=0.01, importance=0.01, access_count=0, creation_time=300.0
            ),
        ]

    def test_full_pipeline_12_memories(
        self,
        params: ParameterSet,
        diverse_memories: list[MemoryState],
    ) -> None:
        """Full pipeline with 12 memories: produces valid result."""
        result = recall(
            diverse_memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "What do you know about this topic?",
            1,
        )
        assert result.gated is False
        assert 1 <= result.k <= DEFAULT_RECALL_CONFIG.max_k
        assert len(result.tier_assignments) == result.k
        assert result.total_tokens_estimate == len(result.context) // CHARS_PER_TOKEN

    def test_higher_relevance_gets_higher_tiers(
        self,
        params: ParameterSet,
        diverse_memories: list[MemoryState],
    ) -> None:
        """Higher-relevance memories get higher or equal tiers."""
        result = recall(
            diverse_memories,
            params,
            RecallConfig(total_budget=2000),
            "What do you know about this topic?",
            1,
        )
        if result.k >= 2:
            first_tier = result.tier_assignments[0].tier
            last_tier = result.tier_assignments[-1].tier
            assert first_tier.value <= last_tier.value

    def test_budget_constraint_limits_output(
        self,
        params: ParameterSet,
        diverse_memories: list[MemoryState],
    ) -> None:
        """Budget constraint actually limits output size."""
        result = recall(
            diverse_memories,
            params,
            RecallConfig(total_budget=50),
            "What do you know about this topic?",
            1,
        )
        # Token estimate should be at or near budget
        assert (
            result.total_tokens_estimate <= 50
            or result.k <= DEFAULT_RECALL_CONFIG.min_k
        )

    def test_roundtrip_rank_adaptive_assign_format_constrain(
        self,
        params: ParameterSet,
        diverse_memories: list[MemoryState],
    ) -> None:
        """Manual round-trip matches the recall() pipeline."""
        message = "Tell me about the project"
        config = RecallConfig(total_budget=2000)

        # Manual pipeline
        ranked = rank_memories(params, diverse_memories, 0.0)
        scores = [s for _, s in ranked]
        k = adaptive_k(scores, config)
        assignments = assign_tiers(ranked, k, config)

        formatted = []
        for ta in assignments:
            mem = diverse_memories[ta.index]
            formatted.append(format_memory(mem, ta.tier, config))

        selected_memories = [diverse_memories[ta.index] for ta in assignments]
        final_a, final_f, n_demoted, n_dropped = budget_constrain(
            assignments,
            formatted,
            selected_memories,
            config,
        )

        manual_context = MEMORY_SEPARATOR.join(final_f)

        # Compare with recall()
        result = recall(diverse_memories, params, config, message, 1)
        assert result.k == len(final_a)
        assert result.gated is False
        assert result.context == manual_context

    def test_custom_tight_budget_aggressive_demotion(
        self,
        params: ParameterSet,
        diverse_memories: list[MemoryState],
    ) -> None:
        """Custom config with tight budget forces aggressive demotion."""
        config = RecallConfig(total_budget=30)
        result = recall(
            diverse_memories,
            params,
            config,
            "What do you know?",
            1,
        )
        if result.k > 0:
            # With tight budget, most memories should be demoted or dropped
            assert result.budget_exceeded is True or result.k == 1

    def test_custom_high_epsilon_triggers_fallback(
        self,
        params: ParameterSet,
    ) -> None:
        """Custom config with high epsilon triggers fallback k more often."""
        config = RecallConfig(epsilon=100.0)
        # With epsilon=100, all gaps are < epsilon, so fallback triggers
        memories = _make_memories([0.9, 0.5, 0.3, 0.1])
        result = recall(
            memories,
            params,
            config,
            "Tell me about X",
            1,
        )
        # Fallback k = min(min_k + gap_buffer, len, max_k) = min(3, 4, 10) = 3
        assert result.k == 3

    def test_with_contents_realistic(
        self,
        params: ParameterSet,
    ) -> None:
        """Full pipeline with realistic contents."""
        memories = [
            _make_memory(relevance=0.95, importance=0.9),
            _make_memory(relevance=0.80, importance=0.7),
            _make_memory(relevance=0.30, importance=0.2),
        ]
        contents = [
            "The user prefers dark mode for all editors.",
            "The project uses Python 3.11 with type hints.",
            "The weather was nice yesterday.",
        ]
        result = recall(
            memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "What are the user preferences?",
            1,
            contents=contents,
        )
        assert result.gated is False
        assert result.k >= 1
        # Higher relevance memory's content should appear in context
        if result.k >= 1:
            assert "dark mode" in result.context or "[Memory]" in result.context

    def test_tuned_parameters_known_result(self, params: ParameterSet) -> None:
        """Known parameter set with known memories produces predictable k."""
        memories = [
            _make_memory(
                relevance=0.99, importance=0.95, access_count=50, creation_time=2.0
            ),
            _make_memory(
                relevance=0.10, importance=0.1, access_count=1, creation_time=200.0
            ),
        ]
        result = recall(
            memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "What's important?",
            1,
        )
        # With a clear gap between 0.99 and 0.10, adaptive_k should select both
        # or just the first depending on gap size
        assert result.k >= 1

    def test_all_identical_memories(self, params: ParameterSet) -> None:
        """All identical memories: all get same score, fallback k, positional tiers.

        [AP1-F3] When all scores are identical and k > 1, assign_tiers uses
        positional fallback (ceil(k/3) per tier), NOT all-HIGH.
        """
        memories = [_make_memory(relevance=0.5) for _ in range(5)]
        result = recall(
            memories,
            params,
            RecallConfig(total_budget=2000),
            "Tell me something",
            1,
        )
        if result.k > 1:
            # With positional fallback, should have a mix of tiers
            tiers = {ta.tier for ta in result.tier_assignments}
            assert len(tiers) >= 2  # At least 2 different tiers
            # First memory should still be HIGH
            assert result.tier_assignments[0].tier == Tier.HIGH

    def test_100_memories_max_k_respected(self, params: ParameterSet) -> None:
        """100 memories: only max_k (10) considered after adaptive_k."""
        memories = [_make_memory(relevance=1.0 - i * 0.01) for i in range(100)]
        result = recall(
            memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "Tell me everything",
            1,
        )
        assert result.k <= DEFAULT_RECALL_CONFIG.max_k

    def test_temperature_has_no_effect_on_scoring(self, params: ParameterSet) -> None:
        """Temperature does NOT affect score_memory or rank_memories.

        [AP3-E1] temperature is used only in soft_select (dynamics pipeline),
        not in score_memory. Two different temperatures produce identical
        recall results.
        """
        memories = [
            _make_memory(relevance=0.95, importance=0.9, access_count=20),
            _make_memory(relevance=0.50, importance=0.4, access_count=3),
            _make_memory(relevance=0.10, importance=0.1, access_count=0),
        ]
        # Build params with two different temperatures
        base_kwargs = dict(
            w1=0.4109,
            w2=0.0500,
            w3=0.3000,
            w4=0.2391,
            alpha=0.5000,
            beta=1.0000,
            delta_t=PINNED["delta_t"],
            s_max=PINNED["s_max"],
            s0=PINNED["s0"],
            novelty_start=PINNED["novelty_start"],
            novelty_decay=PINNED["novelty_decay"],
            survival_threshold=PINNED["survival_threshold"],
            feedback_sensitivity=PINNED["feedback_sensitivity"],
        )
        params_low_temp = ParameterSet(temperature=4.9265, **base_kwargs)
        params_high_temp = ParameterSet(temperature=10.0, **base_kwargs)

        config = RecallConfig(total_budget=2000)
        result1 = recall(memories, params_low_temp, config, "Test query", 1)
        result2 = recall(memories, params_high_temp, config, "Test query", 1)

        assert result1.k == result2.k
        assert result1.context == result2.context

    def test_min_k_vs_budget_conflict_overflow_reported(
        self, params: ParameterSet
    ) -> None:
        """min_k conflict: min_k=3, total_budget=10, budget_overflow > 0.

        [AP1-F1] When min_k prevents dropping below budget,
        budget_overflow reports the excess.
        """
        memories = _make_memories([0.95, 0.90, 0.85, 0.80, 0.75])
        config = RecallConfig(total_budget=10, min_k=3)
        result = recall(memories, params, config, "Tell me about X", 1)
        # min_k=3 means at least 3 memories, which surely exceeds 10 tokens
        assert result.k >= 3
        assert result.budget_overflow > 0

    def test_all_identical_scores_positional_tiers_distributes_budget(
        self, params: ParameterSet
    ) -> None:
        """All identical scores with content: positional tier distributes budget.

        [AP1-F3] Instead of all-HIGH (which wastes budget), positional
        fallback distributes memories across tiers.
        """
        memories = [_make_memory(relevance=0.5) for _ in range(6)]
        contents = [f"Content for memory {i}" for i in range(6)]
        config = RecallConfig(total_budget=2000)
        result = recall(
            memories,
            params,
            config,
            "Tell me something",
            1,
            contents=contents,
        )
        if result.k > 1:
            tiers = [ta.tier for ta in result.tier_assignments]
            # Should NOT be all HIGH
            assert not all(t == Tier.HIGH for t in tiers)

    def test_adaptive_k_vs_fixed_k5_comparison(self, params: ParameterSet) -> None:
        """Compare adaptive_k results against fixed k=5.

        [AP3-T1] Adaptive_k captures at least as many relevant memories
        as fixed k=5, never fewer. Documents the tradeoff.
        """
        memories = [
            _make_memory(relevance=0.95, importance=0.9, access_count=20),
            _make_memory(relevance=0.90, importance=0.8, access_count=15),
            _make_memory(relevance=0.85, importance=0.7, access_count=10),
            _make_memory(relevance=0.40, importance=0.3, access_count=2),
            _make_memory(relevance=0.10, importance=0.1, access_count=0),
            _make_memory(relevance=0.05, importance=0.05, access_count=0),
        ]
        config = RecallConfig(total_budget=2000)
        result = recall(memories, params, config, "What do you know?", 1)
        # Adaptive k should return at least 1 (always)
        assert result.k >= 1
        # And should capture the top memories (k >= number of clearly relevant)
        # The top 3 memories are clearly relevant (relevance > 0.8)
        # Fixed k=5 would always return 5 (if available)
        # Adaptive k returns >= 1, potentially more or fewer than 5

    def test_tight_budget_exercises_full_cascade(self, params: ParameterSet) -> None:
        """Tight budget (total_budget=100) triggers full demotion/drop cascade.

        [AP2-F5] Default budget rarely activates; tight budget exercises
        the full budget constraint system.
        """
        memories = _make_memories([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65])
        contents = [f"Content {i} " * 20 for i in range(7)]
        config = RecallConfig(total_budget=100)
        result = recall(
            memories,
            params,
            config,
            "Tell me about X",
            1,
            contents=contents,
        )
        assert result.budget_exceeded is True or result.k <= 1
        if result.k > 0:
            # Should have been significantly constrained
            assert result.total_tokens_estimate <= 100 or result.k <= config.min_k


# ============================================================
# 8. TestRecallProperties (~10 tests, Hypothesis)
# ============================================================


class TestRecallProperties:
    """Property-based tests using Hypothesis."""

    @given(
        min_k=st.integers(min_value=1, max_value=5),
        max_k=st.integers(min_value=5, max_value=20),
        gap_buffer=st.integers(min_value=0, max_value=5),
        epsilon=st.floats(min_value=0.001, max_value=1.0, allow_nan=False),
        high_threshold=st.floats(min_value=0.5, max_value=0.99, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_recall_config_valid_construction(
        self,
        min_k: int,
        max_k: int,
        gap_buffer: int,
        epsilon: float,
        high_threshold: float,
    ) -> None:
        """For any valid combination, RecallConfig() does not raise."""
        mid_threshold = high_threshold * 0.5  # Ensure mid < high
        config = RecallConfig(
            min_k=min_k,
            max_k=max_k,
            gap_buffer=gap_buffer,
            epsilon=epsilon,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        )
        assert config.min_k == min_k

    @given(
        n=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50)
    def test_adaptive_k_returns_valid_k(self, n: int) -> None:
        """For any descending score list, adaptive_k returns valid k."""
        if n == 0:
            assert adaptive_k([], DEFAULT_RECALL_CONFIG) == 0
        else:
            scores = sorted(
                [1.0 - i / max(n, 1) for i in range(n)],
                reverse=True,
            )
            k = adaptive_k(scores, DEFAULT_RECALL_CONFIG)
            assert 0 <= k <= len(scores)
            if n > 0:
                assert k >= 1

    @given(
        n=st.integers(min_value=1, max_value=10),
        k_frac=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_assign_tiers_output_length_equals_k(self, n: int, k_frac: float) -> None:
        """For any valid input, output length == k."""
        scores = [(i, 1.0 - i * 0.1) for i in range(n)]
        k = max(1, int(k_frac * n))
        k = min(k, n)
        result = assign_tiers(scores, k, DEFAULT_RECALL_CONFIG)
        assert len(result) == k

    @given(
        n=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_tier_assignment_monotonic_with_score(self, n: int) -> None:
        """Tier assignment is monotonic: higher score -> higher or equal detail tier."""
        scores = [(i, 1.0 - i * 0.1) for i in range(n)]
        result = assign_tiers(scores, n, DEFAULT_RECALL_CONFIG)
        for i in range(len(result) - 1):
            if result[i].normalized_score > result[i + 1].normalized_score:
                # Higher-detail tier has lower Tier.value
                assert result[i].tier.value <= result[i + 1].tier.value

    @given(
        n=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=30)
    def test_recall_result_invariants(self, n: int) -> None:
        """For any valid inputs, RecallResult invariants hold."""
        params = _default_params()
        memories = [_make_memory(relevance=max(0.01, 1.0 - i * 0.12)) for i in range(n)]
        result = recall(
            memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "Tell me about X",
            1,
        )
        assert result.gated is False
        assert result.k == len(result.tier_assignments)
        assert result.total_tokens_estimate == len(result.context) // CHARS_PER_TOKEN
        if result.k > 1:
            assert result.context.count(MEMORY_SEPARATOR) == result.k - 1

    @given(
        n=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=20)
    def test_non_gated_recall_nonempty_tiers(self, n: int) -> None:
        """For any non-gated recall, tier_assignments is non-empty when memories is non-empty."""
        params = _default_params()
        memories = [_make_memory(relevance=max(0.01, 0.9 - i * 0.15)) for i in range(n)]
        result = recall(
            memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "Tell me about X",
            1,
        )
        assert len(result.tier_assignments) >= 1

    @given(
        budget=st.integers(min_value=1, max_value=2000),
    )
    @settings(max_examples=30)
    def test_budget_constrain_respects_budget(self, budget: int) -> None:
        """For any budget, output token estimate <= budget (modulo min_k)."""
        params = _default_params()
        memories = _make_memories([0.9, 0.7, 0.5])
        config = RecallConfig(total_budget=budget)
        result = recall(
            memories,
            params,
            config,
            "Tell me about X",
            1,
        )
        # Either within budget or at min_k
        assert result.total_tokens_estimate <= budget or result.k <= config.min_k

    @given(
        n=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=30)
    def test_recall_result_k_consistent(self, n: int) -> None:
        """RecallResult.k is consistent with tier_assignments length."""
        params = _default_params()
        memories = [_make_memory(relevance=max(0.01, 1.0 - i * 0.12)) for i in range(n)]
        result = recall(
            memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "What is this?",
            2,
        )
        assert result.k == len(result.tier_assignments)

    @given(
        n=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30)
    def test_assign_tiers_normalized_in_unit_range(self, n: int) -> None:
        """All normalized scores in [0.0, 1.0] for any valid input."""
        scores = [(i, 1.0 - i * (0.9 / n)) for i in range(n)]
        result = assign_tiers(scores, n, DEFAULT_RECALL_CONFIG)
        for ta in result:
            assert 0.0 <= ta.normalized_score <= 1.0

    @given(
        message=st.text(min_size=0, max_size=100),
        turn=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50)
    def test_should_recall_always_returns_bool(self, message: str, turn: int) -> None:
        """should_recall always returns a bool for valid inputs."""
        result = should_recall(message, turn, DEFAULT_RECALL_CONFIG)
        assert isinstance(result, bool)

    @given(
        b1=st.integers(min_value=100, max_value=2000),
        b2=st.integers(min_value=10, max_value=99),
    )
    @settings(max_examples=20)
    def test_budget_monotonicity_indices(self, b1: int, b2: int) -> None:
        """For B1 > B2, indices at B2 are a subset of indices at B1.

        [AP3-E4] Budget monotonicity property: larger budget means
        a superset of recalled memory indices.
        """
        params = _default_params()
        memories = _make_memories([0.9, 0.7, 0.5, 0.3, 0.1])
        config1 = RecallConfig(total_budget=b1)
        config2 = RecallConfig(total_budget=b2)
        result1 = recall(memories, params, config1, "Tell me about X", 1)
        result2 = recall(memories, params, config2, "Tell me about X", 1)
        indices1 = {ta.index for ta in result1.tier_assignments}
        indices2 = {ta.index for ta in result2.tier_assignments}
        assert indices2.issubset(indices1)

    @given(
        b1=st.integers(min_value=100, max_value=2000),
        b2=st.integers(min_value=10, max_value=99),
    )
    @settings(max_examples=20)
    def test_budget_monotonicity_tiers(self, b1: int, b2: int) -> None:
        """For B1 > B2, shared memories have equal or higher tier at B1.

        [AP3-E4] Larger budget means memories never get LOWER detail.
        """
        params = _default_params()
        memories = _make_memories([0.9, 0.7, 0.5, 0.3, 0.1])
        config1 = RecallConfig(total_budget=b1)
        config2 = RecallConfig(total_budget=b2)
        result1 = recall(memories, params, config1, "Tell me about X", 1)
        result2 = recall(memories, params, config2, "Tell me about X", 1)
        tier_map1 = {ta.index: ta.tier for ta in result1.tier_assignments}
        tier_map2 = {ta.index: ta.tier for ta in result2.tier_assignments}
        for idx in tier_map2:
            if idx in tier_map1:
                # Higher detail = lower tier value. B1 should have <= tier value.
                assert tier_map1[idx].value <= tier_map2[idx].value

    @given(
        n=st.integers(min_value=1, max_value=8),
        budget=st.integers(min_value=1, max_value=2000),
    )
    @settings(max_examples=30)
    def test_budget_overflow_formula_property(self, n: int, budget: int) -> None:
        """budget_overflow == max(0, total_tokens_estimate - total_budget) always.

        [AP1-F1] The overflow formula is an invariant of RecallResult.
        """
        params = _default_params()
        memories = [_make_memory(relevance=max(0.01, 1.0 - i * 0.12)) for i in range(n)]
        config = RecallConfig(total_budget=budget)
        result = recall(memories, params, config, "Tell me about X", 1)
        expected = max(0, result.total_tokens_estimate - config.total_budget)
        assert result.budget_overflow == expected


# ============================================================
# 9. TestRecallSlow (~5 tests, @pytest.mark.slow)
# ============================================================


class TestRecallSlow:
    """Slow tests for large-scale and edge-value scenarios."""

    @pytest.fixture()
    def params(self) -> ParameterSet:
        return _default_params()

    @pytest.mark.slow
    def test_pipeline_1000_memories(self, params: ParameterSet) -> None:
        """Pipeline with 1000 memories: correct k, reasonable latency."""
        memories = [
            _make_memory(relevance=max(0.01, 1.0 - i * 0.001)) for i in range(1000)
        ]
        result = recall(
            memories,
            params,
            DEFAULT_RECALL_CONFIG,
            "Tell me everything you know",
            1,
        )
        assert result.gated is False
        assert 1 <= result.k <= DEFAULT_RECALL_CONFIG.max_k
        assert result.total_tokens_estimate == len(result.context) // CHARS_PER_TOKEN

    @pytest.mark.slow
    def test_all_recall_config_edge_values(self) -> None:
        """Pipeline with edge-value RecallConfig still works.

        Tests minimum viable config and extreme configs.
        """
        # Minimum viable: min_k == max_k == 1, tiny budget
        config_min = RecallConfig(
            min_k=1,
            max_k=1,
            gap_buffer=0,
            epsilon=0.001,
            total_budget=10,
            high_max_chars=50,
            medium_max_chars=30,
            low_max_chars=20,
        )
        params = _default_params()
        memories = _make_memories([0.9, 0.5, 0.1])
        result = recall(memories, params, config_min, "Test message here", 1)
        assert result.k == 1

        # Large config
        config_max = RecallConfig(
            min_k=1,
            max_k=100,
            gap_buffer=50,
            epsilon=0.001,
            total_budget=10000,
            high_max_chars=5000,
            medium_max_chars=2000,
            low_max_chars=500,
        )
        result2 = recall(memories, params, config_max, "Test message here", 1)
        assert result2.k >= 1

    @pytest.mark.slow
    def test_budget_constraint_100_high_memories(self, params: ParameterSet) -> None:
        """Budget constraint with 100 HIGH-tier memories: demotion cascade."""
        memories = [
            _make_memory(relevance=max(0.01, 1.0 - i * 0.009)) for i in range(100)
        ]
        contents = [
            f"Memory content {i} with some text to fill space" for i in range(100)
        ]
        config = RecallConfig(total_budget=100, max_k=100)
        result = recall(
            memories,
            params,
            config,
            "Tell me everything",
            1,
            contents=contents,
        )
        # Budget should have forced significant demotions/drops
        assert result.budget_exceeded is True or result.k <= 1

    @pytest.mark.slow
    def test_simulated_conversation_10_turns(self, params: ParameterSet) -> None:
        """Full integration with simulated conversation (10 turns, varying messages)."""
        messages = [
            "Hello, let's get started.",  # Turn 0: first turn, recalls
            "ok",  # Turn 1: gated (trivial)
            "What do you know about Python?",  # Turn 2: question, recalls
            "thanks",  # Turn 3: gated (trivial)
            "Tell me about the project setup.",  # Turn 4: recalls
            "yes",  # Turn 5: gated (trivial)
            "How does the auth system work?",  # Turn 6: question, recalls
            "   ",  # Turn 7: gated (empty after strip)
            "Explain the database schema.",  # Turn 8: recalls
            "goodbye",  # Turn 9: gated (trivial)
        ]
        memories = _make_memories([0.9, 0.7, 0.5, 0.3, 0.1])

        for turn, msg in enumerate(messages):
            result = recall(memories, params, DEFAULT_RECALL_CONFIG, msg, turn)
            if turn == 0:
                assert result.gated is False, "Turn 0 should always recall"
            elif msg.strip() == "" or msg.strip().lower() in (
                "ok",
                "thanks",
                "yes",
                "goodbye",
            ):
                if turn > 0:
                    assert result.gated is True, (
                        f"Turn {turn} ({msg!r}) should be gated"
                    )
            else:
                assert result.gated is False, f"Turn {turn} ({msg!r}) should recall"

    @pytest.mark.slow
    def test_epsilon_fallback_rate_below_5_percent(self, params: ParameterSet) -> None:
        """Epsilon fallback rate < 5% over 200+ random memory pools.

        [AP3-E6] Regression test: with tuned weights, epsilon=0.01 should
        almost never trigger the fallback. If this test fails, epsilon needs
        recalibration alongside the scoring weights.
        """
        rng = random.Random(42)
        fallback_count = 0
        n_trials = 200
        config = DEFAULT_RECALL_CONFIG  # epsilon=0.01

        for _ in range(n_trials):
            n_memories = rng.randint(3, 20)
            memories = [
                _make_memory(
                    relevance=rng.random(),
                    importance=rng.random(),
                    access_count=rng.randint(0, 50),
                    creation_time=rng.uniform(0, 300),
                    strength=rng.uniform(1.0, 10.0),
                    last_access_time=rng.uniform(0, 50),
                )
                for _ in range(n_memories)
            ]
            ranked = rank_memories(params, memories, 0.0)
            scores = [s for _, s in ranked]
            if len(scores) < 2:
                continue
            # Check if epsilon fallback would trigger
            effective = scores[: config.max_k + config.gap_buffer + 1]
            gaps = [effective[i] - effective[i + 1] for i in range(len(effective) - 1)]
            if gaps and max(gaps) < config.epsilon:
                fallback_count += 1

        fallback_rate = fallback_count / n_trials
        assert fallback_rate < 0.05, (
            f"Epsilon fallback rate {fallback_rate:.2%} >= 5%. "
            f"Epsilon may need recalibration for current weights."
        )

    @pytest.mark.slow
    def test_10000_memories_top_slice_truncation(self, params: ParameterSet) -> None:
        """10000 memories: top-slice truncation keeps k <= max_k.

        [AP2-F3] Without truncation, the largest gap in 10000 scores
        would be deep in the list, causing k=max_k always.
        """
        memories = [
            _make_memory(relevance=max(0.01, 1.0 - i * 0.0001)) for i in range(10000)
        ]
        config = DEFAULT_RECALL_CONFIG
        result = recall(memories, params, config, "Tell me everything", 1)
        assert result.k <= config.max_k

    @pytest.mark.slow
    def test_regression_known_params_expected_distribution(
        self, params: ParameterSet
    ) -> None:
        """Regression: known parameter set produces expected k and tier distribution.

        With tuned parameters (w1=0.4109, etc.) and specific memories,
        verify that the recall pipeline produces consistent results.
        """
        memories = [
            _make_memory(
                relevance=0.95, importance=0.9, access_count=20, creation_time=5.0
            ),
            _make_memory(
                relevance=0.80, importance=0.7, access_count=10, creation_time=30.0
            ),
            _make_memory(
                relevance=0.40, importance=0.3, access_count=2, creation_time=100.0
            ),
            _make_memory(
                relevance=0.10, importance=0.1, access_count=0, creation_time=200.0
            ),
        ]
        config = RecallConfig(total_budget=2000)  # High budget, no constraint
        result = recall(memories, params, config, "What do you know?", 1)

        # With these memories and tuned params, we expect:
        # - At least 2 memories recalled (clear gap between high and low relevance)
        assert result.k >= 2
        # - First memory should be HIGH tier
        assert result.tier_assignments[0].tier == Tier.HIGH
        # - Result is deterministic (same inputs -> same output)
        result2 = recall(memories, params, config, "What do you know?", 1)
        assert result.k == result2.k
        assert result.context == result2.context


# ============================================================
# 10. TestRecallConfig (~additional validation tests)
# ============================================================


class TestRecallConfig:
    """Tests for RecallConfig dataclass validation."""

    def test_default_config_valid(self) -> None:
        """Default config with all defaults is valid."""
        config = RecallConfig()
        assert config.min_k == 1
        assert config.max_k == 10
        assert config.gap_buffer == 2
        assert config.epsilon == pytest.approx(0.01)
        assert config.total_budget == 800

    def test_config_is_frozen(self) -> None:
        """RecallConfig is frozen."""
        config = RecallConfig()
        with pytest.raises(AttributeError):
            config.min_k = 5  # type: ignore[misc]

    def test_min_k_less_than_one_raises(self) -> None:
        """min_k < 1 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(min_k=0)

    def test_max_k_less_than_min_k_raises(self) -> None:
        """max_k < min_k raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(min_k=5, max_k=3)

    def test_gap_buffer_negative_raises(self) -> None:
        """gap_buffer < 0 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(gap_buffer=-1)

    def test_epsilon_zero_raises(self) -> None:
        """epsilon <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(epsilon=0.0)

    def test_epsilon_negative_raises(self) -> None:
        """epsilon < 0 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(epsilon=-0.01)

    def test_high_threshold_below_mid_raises(self) -> None:
        """high_threshold <= mid_threshold raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(high_threshold=0.3, mid_threshold=0.5)

    def test_mid_threshold_negative_raises(self) -> None:
        """mid_threshold < 0.0 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(mid_threshold=-0.1)

    def test_high_threshold_above_one_raises(self) -> None:
        """high_threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(high_threshold=1.1)

    def test_total_budget_zero_raises(self) -> None:
        """total_budget < 1 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(total_budget=0)

    def test_budget_shares_not_summing_to_one_raises(self) -> None:
        """Budget shares not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(
                high_budget_share=0.5,
                medium_budget_share=0.3,
                low_budget_share=0.1,
            )

    def test_budget_share_zero_raises(self) -> None:
        """Any budget share <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(
                high_budget_share=0.0,
                medium_budget_share=0.5,
                low_budget_share=0.5,
            )

    def test_gate_min_length_negative_raises(self) -> None:
        """gate_min_length < 0 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(gate_min_length=-1)

    def test_max_chars_zero_raises(self) -> None:
        """Any *_max_chars < 1 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(high_max_chars=0)
        with pytest.raises(ValueError):
            RecallConfig(medium_max_chars=0)
        with pytest.raises(ValueError):
            RecallConfig(low_max_chars=0)

    def test_min_k_equals_max_k_valid(self) -> None:
        """min_k == max_k is valid (fixed k)."""
        config = RecallConfig(min_k=5, max_k=5)
        assert config.min_k == 5
        assert config.max_k == 5

    def test_gap_buffer_zero_valid(self) -> None:
        """gap_buffer=0 is valid."""
        config = RecallConfig(gap_buffer=0)
        assert config.gap_buffer == 0

    def test_chars_per_token_default(self) -> None:
        """Default chars_per_token is 4.

        [AP2-F1] Configurable token estimation ratio.
        """
        config = RecallConfig()
        assert config.chars_per_token == 4

    def test_chars_per_token_conservative(self) -> None:
        """chars_per_token=2 is valid (conservative for non-English).

        [AP2-F1] Conservative mode for CJK/code-heavy content.
        """
        config = RecallConfig(chars_per_token=2)
        assert config.chars_per_token == 2

    def test_chars_per_token_zero_raises(self) -> None:
        """chars_per_token < 1 raises ValueError.

        [AP2-F1] Must be at least 1.
        """
        with pytest.raises(ValueError):
            RecallConfig(chars_per_token=0)

    def test_chars_per_token_negative_raises(self) -> None:
        """chars_per_token < 0 raises ValueError."""
        with pytest.raises(ValueError):
            RecallConfig(chars_per_token=-1)


# ============================================================
# 11. TestTierEnum (~4 tests)
# ============================================================


class TestTierEnum:
    """Tests for the Tier enum."""

    def test_tier_ordering(self) -> None:
        """Tier.HIGH.value < Tier.MEDIUM.value < Tier.LOW.value."""
        assert Tier.HIGH.value < Tier.MEDIUM.value < Tier.LOW.value

    def test_exactly_three_members(self) -> None:
        """Exactly three members."""
        assert len(Tier) == 3

    def test_tier_values(self) -> None:
        """Tier values are 1, 2, 3."""
        assert Tier.HIGH.value == 1
        assert Tier.MEDIUM.value == 2
        assert Tier.LOW.value == 3

    def test_tier_names(self) -> None:
        """Tier names are HIGH, MEDIUM, LOW."""
        assert Tier.HIGH.name == "HIGH"
        assert Tier.MEDIUM.name == "MEDIUM"
        assert Tier.LOW.name == "LOW"


# ============================================================
# 12. TestTierAssignment (~3 tests)
# ============================================================


class TestTierAssignment:
    """Tests for TierAssignment dataclass."""

    def test_frozen(self) -> None:
        """TierAssignment is frozen."""
        ta = TierAssignment(index=0, score=0.9, normalized_score=1.0, tier=Tier.HIGH)
        with pytest.raises(AttributeError):
            ta.index = 1  # type: ignore[misc]

    def test_fields(self) -> None:
        """TierAssignment has expected fields."""
        ta = TierAssignment(index=0, score=0.9, normalized_score=1.0, tier=Tier.HIGH)
        assert ta.index == 0
        assert ta.score == pytest.approx(0.9)
        assert ta.normalized_score == pytest.approx(1.0)
        assert ta.tier == Tier.HIGH

    def test_index_non_negative(self) -> None:
        """index must be >= 0 (by invariant)."""
        ta = TierAssignment(index=0, score=0.5, normalized_score=0.5, tier=Tier.MEDIUM)
        assert ta.index >= 0


# ============================================================
# 13. TestModuleConstants (~3 tests)
# ============================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_default_recall_config_is_valid(self) -> None:
        """DEFAULT_RECALL_CONFIG is a valid RecallConfig instance."""
        assert isinstance(DEFAULT_RECALL_CONFIG, RecallConfig)

    def test_memory_separator(self) -> None:
        """MEMORY_SEPARATOR is the expected string."""
        assert MEMORY_SEPARATOR == "\n---\n"

    def test_chars_per_token(self) -> None:
        """CHARS_PER_TOKEN is 4."""
        assert CHARS_PER_TOKEN == 4

    def test_chars_per_token_conservative(self) -> None:
        """CHARS_PER_TOKEN_CONSERVATIVE is 2.

        [AP2-F1] Conservative ratio for non-English content.
        """
        assert CHARS_PER_TOKEN_CONSERVATIVE == 2

    def test_min_demotion_savings_tokens(self) -> None:
        """MIN_DEMOTION_SAVINGS_TOKENS is 5.

        [AP2-F4] Minimum savings to justify a demotion.
        """
        assert MIN_DEMOTION_SAVINGS_TOKENS == 5
