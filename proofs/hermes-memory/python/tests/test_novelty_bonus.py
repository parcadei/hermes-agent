"""Property tests mapping Lean theorems from Section 9 to Python.

10 theorems covering:
  novelty_bonus: positivity, initial value, upper bound, monotonicity, limit
  exploration_window: positivity, threshold guarantee
  boosted_score: lower bounds
  Anti-cold-start guarantee
"""

import math

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hermes_memory.core import (
    novelty_bonus,
    exploration_window,
    boosted_score,
)

from tests.strategies import pos_real, pos_time

MAX_EXAMPLES = 200

# Strategies for novelty-specific params
n0_st = lambda: st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
gamma_st = lambda: st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
epsilon_st = lambda: st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False)


# ============================================================
# novelty_bonus (5 theorems)
# ============================================================


class TestNoveltyBonus:
    """Properties of novelty(t) = N0 * exp(-gamma*t)."""

    @given(N0=n0_st(), gamma=gamma_st(), t=pos_time())
    @settings(max_examples=MAX_EXAMPLES)
    def test_noveltyBonus_pos(self, N0, gamma, t):
        """noveltyBonus_pos: novelty_bonus(N0, gamma, t) > 0 for N0 > 0.

        In exact math, N0*exp(-gamma*t) > 0. In IEEE754, underflows when gamma*t > ~708.
        """
        assume(gamma * t < 700)
        assert novelty_bonus(N0, gamma, t) > 0

    @given(N0=n0_st(), gamma=gamma_st())
    @settings(max_examples=MAX_EXAMPLES)
    def test_noveltyBonus_at_zero(self, N0, gamma):
        """noveltyBonus_at_zero: novelty_bonus(N0, gamma, 0) == N0."""
        assert novelty_bonus(N0, gamma, 0.0) == pytest.approx(N0)

    @given(N0=n0_st(), gamma=gamma_st(), t=pos_time())
    @settings(max_examples=MAX_EXAMPLES)
    def test_noveltyBonus_le_init(self, N0, gamma, t):
        """noveltyBonus_le_init: novelty_bonus <= N0 for gamma > 0, t >= 0."""
        assert novelty_bonus(N0, gamma, t) <= N0 + 1e-12

    @given(N0=n0_st(), gamma=gamma_st(), t1=pos_time(), t2=pos_time())
    @settings(max_examples=MAX_EXAMPLES)
    def test_noveltyBonus_antitone(self, N0, gamma, t1, t2):
        """noveltyBonus_antitone: t1 <= t2 ==> novelty_bonus(t1) >= novelty_bonus(t2)."""
        assume(t1 <= t2)
        assert novelty_bonus(N0, gamma, t1) >= novelty_bonus(N0, gamma, t2) - 1e-12

    @given(N0=n0_st(), gamma=gamma_st())
    @settings(max_examples=MAX_EXAMPLES)
    def test_noveltyBonus_tendsto_zero(self, N0, gamma):
        """noveltyBonus_tendsto_zero: novelty_bonus -> 0 as t -> inf."""
        val = novelty_bonus(N0, gamma, 1000.0)
        assert val < 1e-4 * N0


# ============================================================
# exploration_window (2 theorems)
# ============================================================


class TestExplorationWindow:
    """Properties of W = ln(N0/epsilon) / gamma."""

    @given(N0=n0_st(), gamma=gamma_st(), epsilon=epsilon_st())
    @settings(max_examples=MAX_EXAMPLES)
    def test_explorationWindow_pos(self, N0, gamma, epsilon):
        """explorationWindow_pos: exploration_window > 0 when epsilon < N0."""
        assume(epsilon < N0)
        assert exploration_window(N0, gamma, epsilon) > 0

    @given(
        N0=n0_st(),
        gamma=gamma_st(),
        epsilon=epsilon_st(),
        t=pos_time(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_noveltyBonus_above_threshold(self, N0, gamma, epsilon, t):
        """noveltyBonus_above_threshold: novelty_bonus >= epsilon for t <= exploration_window.

        This is the ANTI-COLD-START guarantee.
        """
        assume(epsilon < N0)
        W = exploration_window(N0, gamma, epsilon)
        assume(t <= W)
        assert novelty_bonus(N0, gamma, t) >= epsilon - 1e-12


# ============================================================
# boosted_score (3 theorems)
# ============================================================


class TestBoostedScore:
    """Properties of boosted = base_score + novelty_bonus."""

    @given(
        base=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        N0=n0_st(),
        gamma=gamma_st(),
        t=pos_time(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_boostedScore_ge_novelty(self, base, N0, gamma, t):
        """boostedScore_ge_novelty: boosted_score >= novelty_bonus when base >= 0."""
        nb = novelty_bonus(N0, gamma, t)
        bs = boosted_score(base, N0, gamma, t)
        assert bs >= nb - 1e-12

    @given(
        base=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        N0=n0_st(),
        gamma=gamma_st(),
        t=pos_time(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_boostedScore_ge_base(self, base, N0, gamma, t):
        """boostedScore_ge_base: boosted_score >= base_score when N0 > 0."""
        bs = boosted_score(base, N0, gamma, t)
        assert bs >= base - 1e-12

    @given(
        N0=n0_st(),
        gamma=gamma_st(),
        epsilon=epsilon_st(),
        t=pos_time(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_coldStart_survival(self, N0, gamma, epsilon, t):
        """coldStart_survival: boosted_score(0, N0, gamma, t) >= epsilon for t <= exploration_window.

        A new memory (base_score=0) stays above threshold during the exploration window.
        """
        assume(epsilon < N0)
        W = exploration_window(N0, gamma, epsilon)
        assume(t <= W)
        bs = boosted_score(0.0, N0, gamma, t)
        assert bs >= epsilon - 1e-12
