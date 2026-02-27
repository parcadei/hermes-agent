"""Comprehensive tests for hermes_memory.engine — the N-memory retrieval engine.

Tests define expected behavioral contracts from the sensitivity analysis spec.
These tests are written BEFORE the implementation (TDD). They should all fail
with ImportError until engine.py is created.

52 tests covering:
  - ParameterSet validation (10 tests)
  - MemoryState validation (5 tests)
  - score_memory (8 tests)
  - rank_memories (6 tests)
  - select_memory (5 tests)
  - step_dynamics (8 tests)
  - simulate (5 tests)
  - Property-based tests with Hypothesis (5 tests)
"""

import math
import random

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hermes_memory.engine import (
    ParameterSet,
    MemoryState,
    SimulationResult,
    score_memory,
    rank_memories,
    select_memory,
    step_dynamics,
    simulate,
)

from hermes_memory.core import (
    retention,
    sigmoid,
    ScoringWeights,
    score as core_score,
    novelty_bonus as core_novelty_bonus,
    strength_decay as core_strength_decay,
)

from tests.strategies import (
    pos_real,
    unit_interval,
    open_unit,
    alpha_st,
    pos_time,
    strength_st,
    smax_st,
)


MAX_EXAMPLES = 200


# ============================================================
# Reference parameter set (standardized across both test files)
# K = exp(-0.1) + (0.25/5)*0.1*10 = 0.9048 + 0.05 = 0.9548 < 1
# Contraction margin: (1 - exp(-0.1)) - (0.25/5)*0.1*10 = 0.0452
# ============================================================

def make_baseline_params() -> ParameterSet:
    """Construct the reference parameter set for tests.

    Standardized across both test_engine.py and test_sensitivity.py:
    alpha=0.1, beta=0.1, delta_t=1.0, Smax=10.0, S0=1.0, T=5.0,
    N0=0.5, gamma=0.2, epsilon=0.05, delta=0.1,
    w1=0.35, w2=0.25, w3=0.20, w4=0.20
    """
    return ParameterSet(
        alpha=0.1,
        beta=0.1,
        delta_t=1.0,
        s_max=10.0,
        s0=1.0,
        temperature=5.0,
        novelty_start=0.5,
        novelty_decay=0.2,
        survival_threshold=0.05,
        feedback_sensitivity=0.1,
        w1=0.35,
        w2=0.25,
        w3=0.20,
        w4=0.20,
    )


def make_simple_memory(**overrides) -> MemoryState:
    """Construct a simple MemoryState with sensible defaults."""
    defaults = dict(
        relevance=0.5,
        last_access_time=5.0,
        importance=0.5,
        access_count=3,
        strength=1.0,
        creation_time=50.0,
    )
    defaults.update(overrides)
    return MemoryState(**defaults)


# ============================================================
# 1. ParameterSet Validation (~10 tests)
# ============================================================


class TestParameterSetValidation:
    """Validate ParameterSet construction and constraint enforcement."""

    def test_baseline_params_construct(self):
        """Known-good baseline params construct without error."""
        params = make_baseline_params()
        assert params.alpha == 0.1
        assert params.beta == 0.1
        assert params.delta_t == 1.0
        assert params.s_max == 10.0
        assert params.s0 == 1.0
        assert params.temperature == 5.0
        assert params.novelty_start == 0.5
        assert params.novelty_decay == 0.2
        assert params.survival_threshold == 0.05
        assert params.feedback_sensitivity == 0.1
        assert params.w1 == 0.35
        assert params.w2 == 0.25
        assert params.w3 == 0.20
        assert params.w4 == 0.20

    def test_satisfies_contraction_baseline(self):
        """Baseline params satisfy contraction: K < 1."""
        params = make_baseline_params()
        assert params.satisfies_contraction() is True

    def test_contraction_margin_positive_baseline(self):
        """Baseline params have positive contraction margin."""
        params = make_baseline_params()
        margin = params.contraction_margin()
        assert margin > 0
        # Manual computation: K = exp(-0.1) + (0.25/5)*0.1*10
        expected_K = math.exp(-0.1) + (0.25 / 5.0) * 0.1 * 10.0
        expected_margin = 1.0 - expected_K
        assert margin == pytest.approx(expected_margin, rel=1e-10)

    def test_alpha_zero_raises(self):
        """alpha=0 violates 0 < alpha < 1."""
        with pytest.raises(ValueError):
            ParameterSet(
                alpha=0.0, beta=0.1, delta_t=1.0, s_max=10.0, s0=1.0,
                temperature=5.0, novelty_start=0.5, novelty_decay=0.2,
                survival_threshold=0.05, feedback_sensitivity=0.1,
                w1=0.35, w2=0.20, w3=0.25, w4=0.20,
            )

    def test_alpha_one_raises(self):
        """alpha=1 violates 0 < alpha < 1."""
        with pytest.raises(ValueError):
            ParameterSet(
                alpha=1.0, beta=0.1, delta_t=1.0, s_max=10.0, s0=1.0,
                temperature=5.0, novelty_start=0.5, novelty_decay=0.2,
                survival_threshold=0.05, feedback_sensitivity=0.1,
                w1=0.35, w2=0.20, w3=0.25, w4=0.20,
            )

    def test_w2_too_large_raises(self):
        """w2=0.5 violates w2 < 0.4 recency cap."""
        with pytest.raises(ValueError):
            ParameterSet(
                alpha=0.1, beta=0.1, delta_t=1.0, s_max=10.0, s0=1.0,
                temperature=5.0, novelty_start=0.5, novelty_decay=0.2,
                survival_threshold=0.05, feedback_sensitivity=0.1,
                w1=0.20, w2=0.50, w3=0.15, w4=0.15,
            )

    def test_weights_not_sum_one_raises(self):
        """Weights not summing to 1 raises ValueError."""
        with pytest.raises(ValueError):
            ParameterSet(
                alpha=0.1, beta=0.1, delta_t=1.0, s_max=10.0, s0=1.0,
                temperature=5.0, novelty_start=0.5, novelty_decay=0.2,
                survival_threshold=0.05, feedback_sensitivity=0.1,
                w1=0.30, w2=0.20, w3=0.25, w4=0.20,  # sum = 0.95
            )

    def test_s0_exceeds_smax_raises(self):
        """s0 >= s_max raises ValueError."""
        with pytest.raises(ValueError):
            ParameterSet(
                alpha=0.1, beta=0.1, delta_t=1.0, s_max=10.0, s0=10.0,
                temperature=5.0, novelty_start=0.5, novelty_decay=0.2,
                survival_threshold=0.05, feedback_sensitivity=0.1,
                w1=0.35, w2=0.20, w3=0.25, w4=0.20,
            )

    def test_epsilon_exceeds_novelty_start_raises(self):
        """survival_threshold >= novelty_start raises ValueError."""
        with pytest.raises(ValueError):
            ParameterSet(
                alpha=0.1, beta=0.1, delta_t=1.0, s_max=10.0, s0=1.0,
                temperature=5.0, novelty_start=0.5, novelty_decay=0.2,
                survival_threshold=0.5,  # equal to novelty_start
                feedback_sensitivity=0.1,
                w1=0.35, w2=0.25, w3=0.20, w4=0.20,
            )

    def test_contraction_violating_params_raise(self):
        """Params that violate contraction condition raise ValueError.

        With alpha=0.9, s_max=100, T=0.1:
        L = 0.25 / 0.1 = 2.5
        K = exp(-0.1*1.0) + 2.5 * 0.9 * 100 = 0.9048 + 225 >> 1
        Contraction is checked in __post_init__ and raises ValueError.
        """
        with pytest.raises(ValueError, match="contraction"):
            ParameterSet(
                alpha=0.9, beta=0.1, delta_t=1.0, s_max=100.0, s0=1.0,
                temperature=0.1, novelty_start=0.3, novelty_decay=0.1,
                survival_threshold=0.05, feedback_sensitivity=0.1,
                w1=0.35, w2=0.20, w3=0.25, w4=0.20,
            )


# ============================================================
# 2. MemoryState Validation (~5 tests)
# ============================================================


class TestMemoryStateValidation:
    """Validate MemoryState construction and field constraints."""

    def test_valid_memory_construct(self):
        """Known-good MemoryState constructs without error."""
        m = MemoryState(
            relevance=0.8,
            last_access_time=5.0,
            importance=0.6,
            access_count=10,
            strength=2.0,
            creation_time=100.0,
        )
        assert m.relevance == 0.8
        assert m.last_access_time == 5.0
        assert m.importance == 0.6
        assert m.access_count == 10
        assert m.strength == 2.0
        assert m.creation_time == 100.0

    def test_relevance_above_one_raises(self):
        """Relevance > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            MemoryState(
                relevance=1.1,
                last_access_time=5.0,
                importance=0.5,
                access_count=3,
                strength=1.0,
                creation_time=50.0,
            )

    def test_relevance_below_zero_raises(self):
        """Relevance < 0.0 raises ValueError."""
        with pytest.raises(ValueError):
            MemoryState(
                relevance=-0.1,
                last_access_time=5.0,
                importance=0.5,
                access_count=3,
                strength=1.0,
                creation_time=50.0,
            )

    def test_negative_access_count_raises(self):
        """Negative access_count raises ValueError."""
        with pytest.raises(ValueError):
            MemoryState(
                relevance=0.5,
                last_access_time=5.0,
                importance=0.5,
                access_count=-1,
                strength=1.0,
                creation_time=50.0,
            )

    def test_negative_strength_raises(self):
        """Negative strength raises ValueError."""
        with pytest.raises(ValueError):
            MemoryState(
                relevance=0.5,
                last_access_time=5.0,
                importance=0.5,
                access_count=3,
                strength=-0.5,
                creation_time=50.0,
            )


# ============================================================
# 3. score_memory (~8 tests)
# ============================================================


class TestScoreMemory:
    """Tests for score_memory(params, memory, current_time)."""

    def test_score_in_valid_range(self):
        """Score with baseline params is in [0, 1 + novelty_start]."""
        params = make_baseline_params()
        memory = make_simple_memory()
        s = score_memory(params, memory, 0.0)
        assert 0.0 <= s <= 1.0 + params.novelty_start

    def test_score_high_relevance_recent_important(self):
        """Memory with max relevance, recent access, high importance scores near 1."""
        params = make_baseline_params()
        memory = make_simple_memory(
            relevance=1.0,
            last_access_time=0.0,  # just accessed
            importance=1.0,
            access_count=100,
            strength=10.0,  # max strength
            creation_time=1000.0,  # old, no novelty bonus
        )
        s = score_memory(params, memory, 0.0)
        # Base score should be close to 1.0: w1*1 + w2*1 + w3*1 + w4*sigmoid(100)
        # sigmoid(100) ~ 1.0
        # base ~ 0.35 + 0.25 + 0.20 + 0.20*1.0 = 1.0
        # novelty ~ 0 (old creation_time)
        assert s >= 0.9

    def test_score_zero_relevance_old_access_zero_importance(self):
        """Memory with zero relevance, old access, zero importance gets low score."""
        params = make_baseline_params()
        memory = make_simple_memory(
            relevance=0.0,
            last_access_time=1000.0,  # very old
            importance=0.0,
            access_count=0,
            strength=0.5,
            creation_time=1000.0,  # old
        )
        s = score_memory(params, memory, 0.0)
        # Base: w1*0 + w2*retention(1000, 0.5) + w3*0 + w4*sigmoid(0)
        # retention(1000, 0.5) ~ 0, sigmoid(0) = 0.5
        # base ~ 0 + 0 + 0 + 0.20*0.5 = 0.10 (w4=0.20 unchanged)
        # novelty ~ 0
        assert s < 0.2
        assert s >= 0.0

    def test_novelty_bonus_applies_for_new_memory(self):
        """Brand-new memory (creation_time=0) gets full novelty bonus."""
        params = make_baseline_params()
        memory_new = make_simple_memory(creation_time=0.0)
        memory_old = make_simple_memory(creation_time=1000.0)
        s_new = score_memory(params, memory_new, 0.0)
        s_old = score_memory(params, memory_old, 0.0)
        # New memory should have novelty_bonus(0.5, 0.2, 0.0) = 0.5 added
        assert s_new > s_old
        assert s_new - s_old == pytest.approx(
            core_novelty_bonus(params.novelty_start, params.novelty_decay, 0.0)
            - core_novelty_bonus(params.novelty_start, params.novelty_decay, 1000.0),
            abs=1e-10,
        )

    def test_novelty_bonus_decays_for_old_memory(self):
        """Novelty bonus decays to approximately zero for old memories."""
        params = make_baseline_params()
        memory = make_simple_memory(creation_time=500.0)
        # novelty_bonus(0.5, 0.2, 500) = 0.5 * exp(-100) ~ 0
        bonus = core_novelty_bonus(
            params.novelty_start, params.novelty_decay, memory.creation_time
        )
        assert bonus < 1e-10

    def test_score_bounded_even_with_novelty(self):
        """Score is bounded by [0, 1 + novelty_start] even with full novelty."""
        params = make_baseline_params()
        # Best possible memory with maximum novelty
        memory = make_simple_memory(
            relevance=1.0,
            last_access_time=0.0,
            importance=1.0,
            access_count=100,
            strength=10.0,
            creation_time=0.0,  # brand new, full novelty
        )
        s = score_memory(params, memory, 0.0)
        assert s <= 1.0 + params.novelty_start + 1e-10

    def test_score_uses_core_retention(self):
        """Score changes when strength changes, proving it uses core.retention."""
        params = make_baseline_params()
        memory_low_s = make_simple_memory(strength=0.1, last_access_time=10.0)
        memory_high_s = make_simple_memory(strength=5.0, last_access_time=10.0)
        s_low = score_memory(params, memory_low_s, 0.0)
        s_high = score_memory(params, memory_high_s, 0.0)
        # Higher strength -> slower retention decay -> higher recency -> higher score
        assert s_high > s_low

    def test_score_uses_core_sigmoid_for_activation(self):
        """Score uses sigmoid(access_count) for activation component."""
        params = make_baseline_params()
        memory_low_ac = make_simple_memory(access_count=0, creation_time=1000.0)
        memory_high_ac = make_simple_memory(access_count=100, creation_time=1000.0)
        s_low = score_memory(params, memory_low_ac, 0.0)
        s_high = score_memory(params, memory_high_ac, 0.0)
        # Higher access_count -> higher sigmoid(access_count) -> higher activation score
        assert s_high > s_low


# ============================================================
# 4. rank_memories (~6 tests)
# ============================================================


class TestRankMemories:
    """Tests for rank_memories(params, memories, current_time)."""

    def test_rank_three_memories_sorted_descending(self):
        """Rank 3 memories with different relevance -> sorted highest first."""
        params = make_baseline_params()
        m_high = make_simple_memory(relevance=0.9, creation_time=1000.0)
        m_mid = make_simple_memory(relevance=0.5, creation_time=1000.0)
        m_low = make_simple_memory(relevance=0.2, creation_time=1000.0)
        ranked = rank_memories(params, [m_low, m_high, m_mid], 0.0)
        # Highest relevance should rank first
        assert ranked[0][0] == 1  # m_high is at index 1
        assert ranked[1][0] == 2  # m_mid is at index 2
        assert ranked[2][0] == 0  # m_low is at index 0
        # Scores should be descending
        assert ranked[0][1] >= ranked[1][1]
        assert ranked[1][1] >= ranked[2][1]

    def test_rank_tiebreak_lower_index_wins(self):
        """Equal scores -> lower index ranks first (tie-breaking policy)."""
        params = make_baseline_params()
        # Identical memories
        m = make_simple_memory(creation_time=1000.0)
        ranked = rank_memories(params, [m, m, m], 0.0)
        # All equal scores, tie-breaking by ascending index
        assert ranked[0][0] == 0
        assert ranked[1][0] == 1
        assert ranked[2][0] == 2

    def test_rank_single_memory(self):
        """Single memory returns [(0, score)]."""
        params = make_baseline_params()
        m = make_simple_memory()
        ranked = rank_memories(params, [m], 0.0)
        assert len(ranked) == 1
        assert ranked[0][0] == 0
        assert isinstance(ranked[0][1], float)

    def test_rank_empty_list(self):
        """Empty list returns []."""
        params = make_baseline_params()
        ranked = rank_memories(params, [], 0.0)
        assert ranked == []

    def test_rank_all_identical_ascending_order(self):
        """All identical memories -> indices in ascending order [0, 1, 2, ...]."""
        params = make_baseline_params()
        m = make_simple_memory(creation_time=1000.0)
        memories = [m, m, m, m, m]
        ranked = rank_memories(params, memories, 0.0)
        indices = [idx for idx, _ in ranked]
        assert indices == [0, 1, 2, 3, 4]

    def test_rank_respects_relevance(self):
        """Highest relevance memory ranks first when other factors equal."""
        params = make_baseline_params()
        m_a = make_simple_memory(relevance=0.3, creation_time=1000.0)
        m_b = make_simple_memory(relevance=0.9, creation_time=1000.0)
        m_c = make_simple_memory(relevance=0.6, creation_time=1000.0)
        ranked = rank_memories(params, [m_a, m_b, m_c], 0.0)
        # m_b (index 1) has highest relevance
        assert ranked[0][0] == 1


# ============================================================
# 5. select_memory (~5 tests)
# ============================================================


class TestSelectMemory:
    """Tests for select_memory(params, memories, current_time)."""

    def test_select_large_score_gap_favors_top(self):
        """With large score gap and low temperature, almost always selects top.

        At T=5.0, the win probability for a score gap of ~0.8 is only ~54.5%
        (adversarial finding pass2 section 1.3). Use T=0.5 for this test so
        that the score gap translates into a strong selection preference.
        """
        # Use low temperature for this specific test (T=0.5 gives strong preference)
        # Use small alpha*s_max to satisfy contraction at T=0.5:
        # K = exp(-0.1) + (0.25/0.5)*0.01*1.0 = 0.9048 + 0.005 = 0.9098 < 1
        params_low_t = ParameterSet(
            alpha=0.01, beta=0.1, delta_t=1.0, s_max=1.0, s0=0.5,
            temperature=0.5,  # low temperature for deterministic selection
            novelty_start=0.5, novelty_decay=0.2,
            survival_threshold=0.05, feedback_sensitivity=0.1,
            w1=0.35, w2=0.25, w3=0.20, w4=0.20,
        )
        m_high = make_simple_memory(relevance=1.0, creation_time=1000.0)
        m_low = make_simple_memory(relevance=0.0, creation_time=1000.0)
        # Score gap is w1*(1.0-0.0) = 0.35
        # At T=0.5: P(high wins) = sigmoid(0.35/0.5) = sigmoid(0.7) ~ 0.668
        # Run 1000 trials for statistical power, expect >= 600 wins
        rng = random.Random(42)
        selections = [
            select_memory(params_low_t, [m_high, m_low], 0.0, rng=rng)
            for _ in range(1000)
        ]
        # Top-scored memory (index 0) should win the majority
        high_count = selections.count(0)
        assert high_count >= 600, (
            f"Expected >= 600 wins out of 1000, got {high_count}"
        )

    def test_select_equal_scores_roughly_uniform(self):
        """With equal scores, selection is roughly 50/50 (statistical test)."""
        params = make_baseline_params()
        m = make_simple_memory(creation_time=1000.0)
        rng = random.Random(123)
        selections = [
            select_memory(params, [m, m], 0.0, rng=rng)
            for _ in range(1000)
        ]
        count_0 = selections.count(0)
        # Should be roughly 500 with standard deviation ~15.8
        # Allow 4 sigma range: 500 +/- 64
        assert 350 < count_0 < 650, (
            f"Expected ~500 selections of index 0, got {count_0}"
        )

    def test_select_high_temperature_more_uniform(self):
        """Higher temperature -> more uniform selection (less deterministic)."""
        # Low temperature params (use small alpha*s_max to satisfy contraction at T=0.5)
        # K = exp(-0.1) + (0.25/0.5)*0.01*1.0 = 0.9048 + 0.005 = 0.9098 < 1
        params_low_t = ParameterSet(
            alpha=0.01, beta=0.1, delta_t=1.0, s_max=1.0, s0=0.5,
            temperature=0.5,  # low temperature
            novelty_start=0.5, novelty_decay=0.2,
            survival_threshold=0.05, feedback_sensitivity=0.1,
            w1=0.35, w2=0.25, w3=0.20, w4=0.20,
        )
        # High temperature params
        params_high_t = ParameterSet(
            alpha=0.01, beta=0.1, delta_t=1.0, s_max=1.0, s0=0.5,
            temperature=50.0,  # high temperature
            novelty_start=0.5, novelty_decay=0.2,
            survival_threshold=0.05, feedback_sensitivity=0.1,
            w1=0.35, w2=0.25, w3=0.20, w4=0.20,
        )
        m_high = make_simple_memory(relevance=0.8, creation_time=1000.0)
        m_low = make_simple_memory(relevance=0.3, creation_time=1000.0)
        memories = [m_high, m_low]

        rng_low = random.Random(42)
        rng_high = random.Random(42)

        sel_low_t = [
            select_memory(params_low_t, memories, 0.0, rng=rng_low)
            for _ in range(500)
        ]
        sel_high_t = [
            select_memory(params_high_t, memories, 0.0, rng=rng_high)
            for _ in range(500)
        ]
        # High temperature should have more selections of the lower-scoring memory
        low_t_underdog = sel_low_t.count(1)
        high_t_underdog = sel_high_t.count(1)
        assert high_t_underdog > low_t_underdog

    def test_select_single_memory_deterministic(self):
        """Single memory always returns index 0."""
        params = make_baseline_params()
        m = make_simple_memory()
        rng = random.Random(99)
        for _ in range(10):
            assert select_memory(params, [m], 0.0, rng=rng) == 0

    def test_select_deterministic_with_seed(self):
        """Same seed produces same selection sequence."""
        params = make_baseline_params()
        m_a = make_simple_memory(relevance=0.7, creation_time=1000.0)
        m_b = make_simple_memory(relevance=0.5, creation_time=1000.0)
        memories = [m_a, m_b]

        results_1 = []
        rng1 = random.Random(777)
        for _ in range(20):
            results_1.append(select_memory(params, memories, 0.0, rng=rng1))

        results_2 = []
        rng2 = random.Random(777)
        for _ in range(20):
            results_2.append(select_memory(params, memories, 0.0, rng=rng2))

        assert results_1 == results_2


# ============================================================
# 6. step_dynamics (~8 tests)
# ============================================================


class TestStepDynamics:
    """Tests for step_dynamics(params, memories, accessed_idx, feedback_signal, current_time)."""

    def test_all_strengths_decay(self):
        """After a step, all memory strengths reflect decay."""
        params = make_baseline_params()
        m = make_simple_memory(strength=5.0)
        memories = [m, m, m]
        result = step_dynamics(params, memories, 0, 1.0, 0.0)
        # Non-accessed memories should have pure decay
        for i in [1, 2]:
            expected_decayed = core_strength_decay(params.beta, 5.0, params.delta_t)
            assert result[i].strength == pytest.approx(expected_decayed, rel=1e-10)

    def test_accessed_memory_gets_reinforcement(self):
        """Accessed memory gets strength reinforcement after decay."""
        params = make_baseline_params()
        m = make_simple_memory(strength=2.0)
        result = step_dynamics(params, [m], 0, 1.0, 0.0)
        # Accessed: decay first, then reinforce
        decayed = core_strength_decay(params.beta, 2.0, params.delta_t)
        # After reinforcement, strength should be > decayed value
        assert result[0].strength >= decayed

    def test_accessed_strength_bounded_by_smax(self):
        """Accessed memory strength stays <= s_max after reinforcement."""
        params = make_baseline_params()
        # Start near s_max
        m = make_simple_memory(strength=params.s_max * 0.99)
        result = step_dynamics(params, [m], 0, 1.0, 0.0)
        assert result[0].strength <= params.s_max

    def test_unaccessed_strength_only_decays(self):
        """Non-accessed memories only experience strength decay."""
        params = make_baseline_params()
        initial_strength = 5.0
        m0 = make_simple_memory(strength=initial_strength)
        m1 = make_simple_memory(strength=initial_strength)
        result = step_dynamics(params, [m0, m1], 0, 1.0, 0.0)
        # Memory 1 was not accessed; its strength should be decayed
        assert result[1].strength < initial_strength
        expected = core_strength_decay(params.beta, initial_strength, params.delta_t)
        assert result[1].strength == pytest.approx(expected, rel=1e-10)

    def test_importance_updates_on_positive_feedback(self):
        """Importance updates for accessed memory with positive feedback."""
        params = make_baseline_params()
        m = make_simple_memory(importance=0.5)
        result = step_dynamics(params, [m], 0, 1.0, 0.0)  # signal=1.0
        # importance_update(0.5, 0.1, 1.0) = clamp01(0.5 + 0.1*1.0) = 0.6
        assert result[0].importance == pytest.approx(0.6, abs=1e-10)

    def test_importance_stays_in_unit_interval(self):
        """Importance stays in [0, 1] even with extreme feedback."""
        params = make_baseline_params()
        m = make_simple_memory(importance=0.99)
        result = step_dynamics(params, [m], 0, 100.0, 0.0)  # large positive signal
        assert 0.0 <= result[0].importance <= 1.0

    def test_access_count_increments(self):
        """Access count increments for accessed memory."""
        params = make_baseline_params()
        m = make_simple_memory(access_count=5)
        result = step_dynamics(params, [m], 0, 1.0, 0.0)
        assert result[0].access_count == 6

    def test_last_access_time_resets_for_accessed(self):
        """Accessed memory has last_access_time reset to 0."""
        params = make_baseline_params()
        m = make_simple_memory(last_access_time=20.0)
        result = step_dynamics(params, [m], 0, 1.0, 0.0)
        assert result[0].last_access_time == 0.0


# ============================================================
# 7. simulate (~5 tests)
# ============================================================


class TestSimulate:
    """Tests for simulate(params, memories, n_steps, access_pattern, ...)."""

    def test_simulate_constant_access_strengths_converge(self):
        """Run 10 steps with constant access pattern -> strengths change monotonically."""
        params = make_baseline_params()
        m0 = make_simple_memory(strength=params.s0, relevance=0.8, creation_time=1000.0)
        m1 = make_simple_memory(strength=params.s0, relevance=0.3, creation_time=1000.0)
        # Always access memory 0
        access_pattern = [0] * 10
        result = simulate(params, [m0, m1], 10, access_pattern)
        # Memory 0 should gain strength over time
        initial_s0 = params.s0
        final_s0 = result.final_memories[0].strength
        assert final_s0 > initial_s0
        # Memory 1 should only decay
        final_s1 = result.final_memories[1].strength
        assert final_s1 < initial_s0

    def test_simulate_result_correct_step_count(self):
        """SimulationResult has correct number of steps."""
        params = make_baseline_params()
        m = make_simple_memory()
        n_steps = 5
        result = simulate(params, [m], n_steps, [0] * n_steps)
        assert len(result.rankings_per_step) == n_steps
        assert len(result.scores_per_step) == n_steps
        assert len(result.strengths_per_step) == n_steps

    def test_simulate_rankings_valid_permutations(self):
        """Rankings per step are valid permutations of input indices."""
        params = make_baseline_params()
        m0 = make_simple_memory(relevance=0.8, creation_time=1000.0)
        m1 = make_simple_memory(relevance=0.5, creation_time=1000.0)
        m2 = make_simple_memory(relevance=0.3, creation_time=1000.0)
        result = simulate(
            params, [m0, m1, m2], 5, [0, 1, 2, 0, 1]
        )
        for ranking in result.rankings_per_step:
            assert sorted(ranking) == [0, 1, 2]

    def test_simulate_scores_in_range(self):
        """All scores per step are in [0, 1 + novelty_start]."""
        params = make_baseline_params()
        m = make_simple_memory()
        result = simulate(params, [m], 5, [0] * 5)
        for step_scores in result.scores_per_step:
            for s in step_scores:
                assert 0.0 <= s <= 1.0 + params.novelty_start + 1e-10

    def test_simulate_final_memories_updated(self):
        """Final memories have updated strengths and importance."""
        params = make_baseline_params()
        m = make_simple_memory(strength=params.s0, importance=0.5, access_count=0)
        result = simulate(params, [m], 3, [0, 0, 0])
        final = result.final_memories[0]
        # After 3 accesses, access_count should be 3
        assert final.access_count == 3
        # Importance should have been updated 3 times with default signal 1.0
        assert final.importance > 0.5


# ============================================================
# 8. Property-Based Tests with Hypothesis (~5 tests)
# ============================================================


def parameter_set_st():
    """Hypothesis strategy that generates valid ParameterSet instances."""
    return st.builds(
        _build_valid_params,
        alpha=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        s_max=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        temperature=st.floats(min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        novelty_start=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
        novelty_decay=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        feedback_sensitivity=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        w1_raw=st.floats(min_value=0.2, max_value=0.5, allow_nan=False, allow_infinity=False),
        w2_raw=st.floats(min_value=0.05, max_value=0.35, allow_nan=False, allow_infinity=False),
        w3_raw=st.floats(min_value=0.1, max_value=0.4, allow_nan=False, allow_infinity=False),
    )


def _build_valid_params(
    alpha, beta, delta_t, s_max, temperature,
    novelty_start, novelty_decay, feedback_sensitivity,
    w1_raw, w2_raw, w3_raw,
):
    """Build a valid ParameterSet from raw Hypothesis values."""
    s0 = s_max * 0.1  # always valid: s0 < s_max
    survival_threshold = novelty_start * 0.1  # always valid: eps < N0

    # Normalize weights to sum to 1 with w2 < 0.4
    w2 = min(w2_raw, 0.39)
    remaining = 1.0 - w2
    w_total = w1_raw + w3_raw
    if w_total < 1e-10:
        w1 = remaining / 3
        w3 = remaining / 3
        w4 = remaining / 3
    else:
        w1 = w1_raw / w_total * remaining * 0.5
        w3 = w3_raw / w_total * remaining * 0.3
        w4 = remaining - w1 - w3

    # Ensure all weights are non-negative
    if w4 < 0:
        w4 = 0.01
        w1 = (remaining - w3 - w4)
        if w1 < 0:
            w1 = 0.01
            w3 = remaining - w1 - w4

    total = w1 + w2 + w3 + w4
    w1, w2, w3, w4 = w1 / total, w2 / total, w3 / total, w4 / total

    # Re-cap w2
    if w2 >= 0.4:
        w2 = 0.39
        rest = 1.0 - w2
        w1 = rest * 0.4
        w3 = rest * 0.35
        w4 = rest * 0.25

    return ParameterSet(
        alpha=alpha,
        beta=beta,
        delta_t=delta_t,
        s_max=s_max,
        s0=s0,
        temperature=temperature,
        novelty_start=novelty_start,
        novelty_decay=novelty_decay,
        survival_threshold=survival_threshold,
        feedback_sensitivity=feedback_sensitivity,
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
    )


def memory_state_st(s_max=10.0):
    """Hypothesis strategy that generates valid MemoryState instances."""
    return st.builds(
        MemoryState,
        relevance=unit_interval(),
        last_access_time=st.floats(
            min_value=0.0, max_value=100.0,
            allow_nan=False, allow_infinity=False,
        ),
        importance=unit_interval(),
        access_count=st.integers(min_value=0, max_value=100),
        strength=st.floats(
            min_value=0.0, max_value=s_max,
            allow_nan=False, allow_infinity=False,
        ),
        creation_time=st.floats(
            min_value=0.0, max_value=500.0,
            allow_nan=False, allow_infinity=False,
        ),
    )


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(memory=memory_state_st())
    @settings(max_examples=MAX_EXAMPLES)
    def test_score_always_nonneg(self, memory):
        """For any valid ParameterSet and MemoryState, score >= 0."""
        params = make_baseline_params()
        s = score_memory(params, memory, 0.0)
        assert s >= -1e-15

    @given(memory=memory_state_st())
    @settings(max_examples=MAX_EXAMPLES)
    def test_score_bounded_above(self, memory):
        """For any valid params and memory, score <= 1 + novelty_start."""
        params = make_baseline_params()
        s = score_memory(params, memory, 0.0)
        assert s <= 1.0 + params.novelty_start + 1e-10

    @given(
        memories=st.lists(memory_state_st(), min_size=1, max_size=10),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_step_preserves_invariants(self, memories):
        """step_dynamics preserves strength >= 0 and importance in [0, 1]."""
        params = make_baseline_params()
        result = step_dynamics(params, memories, 0, 1.0, 0.0)
        for m in result:
            assert m.strength >= 0.0
            assert m.strength <= params.s_max + 1e-10
            assert 0.0 <= m.importance <= 1.0
            assert m.access_count >= 0
            assert m.last_access_time >= 0.0
            assert m.creation_time >= 0.0

    @given(
        memories=st.lists(memory_state_st(), min_size=1, max_size=10),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_rank_returns_permutation(self, memories):
        """rank_memories always returns a permutation of input indices."""
        params = make_baseline_params()
        ranked = rank_memories(params, memories, 0.0)
        indices = sorted([idx for idx, _ in ranked])
        assert indices == list(range(len(memories)))

    @given(
        rel_low=st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False),
        rel_high=st.floats(min_value=0.51, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_score_monotone_in_relevance(self, rel_low, rel_high):
        """Higher relevance -> higher score, all other things equal."""
        params = make_baseline_params()
        m_low = make_simple_memory(relevance=rel_low, creation_time=1000.0)
        m_high = make_simple_memory(relevance=rel_high, creation_time=1000.0)
        s_low = score_memory(params, m_low, 0.0)
        s_high = score_memory(params, m_high, 0.0)
        assert s_high >= s_low - 1e-12
