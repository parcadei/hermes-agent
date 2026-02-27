"""Property tests mapping Lean theorems from Section 7 to Python.

11 theorems covering:
  strength_decay, combined_factor, steady_state_strength
  Anti-lock-in guarantee: S* < Smax
"""

import math

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hermes_memory.core import (
    strength_decay,
    combined_factor,
    steady_state_strength,
    strength_update,
)

from tests.strategies import (
    pos_real,
    alpha_st,
    pos_time,
    strength_st,
    smax_st,
)

MAX_EXAMPLES = 200


# ============================================================
# strength_decay (5 theorems)
# ============================================================


class TestStrengthDecay:
    """Properties of S(t) = S0 * exp(-beta*t)."""

    @given(
        beta=pos_real(),
        S0=pos_real(),
        t=pos_time(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_strengthDecay_pos(self, beta, S0, t):
        """strengthDecay_pos: strength_decay(beta, S0, t) > 0 for S0 > 0.

        In exact math, exp(-beta*t) > 0. In IEEE754, underflows to 0.0 when
        beta*t > ~708. Constrain to avoid underflow.
        """
        assume(beta * t < 700)
        assert strength_decay(beta, S0, t) > 0

    @given(
        beta=pos_real(),
        S0=pos_real(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_strengthDecay_at_zero(self, beta, S0):
        """strengthDecay_at_zero: strength_decay(beta, S0, 0) == S0."""
        assert strength_decay(beta, S0, 0.0) == pytest.approx(S0)

    @given(
        beta=pos_real(),
        S0=pos_real(),
        t=pos_time(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_strengthDecay_le_init(self, beta, S0, t):
        """strengthDecay_le_init: strength_decay(beta, S0, t) <= S0 for beta > 0, t >= 0."""
        assert strength_decay(beta, S0, t) <= S0 + 1e-12

    @given(
        beta=pos_real(),
        S0=pos_real(),
        t1=pos_time(),
        t2=pos_time(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_strengthDecay_antitone(self, beta, S0, t1, t2):
        """strengthDecay_antitone: t1 <= t2 ==> strength_decay(t1) >= strength_decay(t2)."""
        assume(t1 <= t2)
        assert strength_decay(beta, S0, t1) >= strength_decay(beta, S0, t2) - 1e-12

    @given(
        beta=pos_real(),
        S0=pos_real(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_strengthDecay_tendsto_zero(self, beta, S0):
        """strengthDecay_tendsto_zero: strength_decay -> 0 as t -> inf."""
        val = strength_decay(beta, S0, 1000.0)
        assert val < 1e-4 * S0


# ============================================================
# combined_factor (2 theorems + 1 helper)
# ============================================================


class TestCombinedFactor:
    """Properties of gamma = (1-alpha) * exp(-beta*delta_t)."""

    @given(
        alpha=alpha_st(),
        beta=pos_real(),
        delta_t=pos_time(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_combinedFactor_pos(self, alpha, beta, delta_t):
        """combinedFactor_pos: combined_factor > 0 for alpha < 1.

        In exact math, (1-alpha)*exp(-beta*delta_t) > 0 for alpha < 1.
        In IEEE754, exp(-beta*delta_t) underflows to 0.0 when beta*delta_t > ~708.
        """
        assume(beta * delta_t < 700)
        assert combined_factor(alpha, beta, delta_t) > 0

    @given(
        alpha=alpha_st(),
        beta=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_combinedFactor_lt_one(self, alpha, beta, delta_t):
        """combinedFactor_lt_one: combined_factor < 1 for 0 < alpha < 1, beta > 0, delta_t > 0."""
        assert combined_factor(alpha, beta, delta_t) < 1.0

    @given(
        alpha=alpha_st(),
        beta=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_steadyState_denom_pos(self, alpha, beta, delta_t):
        """steadyState_denom_pos: 1 - combined_factor > 0."""
        gamma = combined_factor(alpha, beta, delta_t)
        assert 1.0 - gamma > 0


# ============================================================
# steady_state_strength (3 theorems)
# ============================================================


class TestSteadyState:
    """Properties of S* = alpha*Smax / (1 - combined_factor)."""

    @given(
        alpha=alpha_st(),
        beta=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        Smax=smax_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_steadyState_pos(self, alpha, beta, delta_t, Smax):
        """steadyState_pos: steady_state_strength > 0."""
        s_star = steady_state_strength(alpha, beta, delta_t, Smax)
        assert s_star > 0

    @given(
        alpha=alpha_st(),
        beta=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        Smax=smax_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_steadyState_lt_Smax(self, alpha, beta, delta_t, Smax):
        """steadyState_lt_Smax: S* < Smax (ANTI-LOCK-IN)."""
        s_star = steady_state_strength(alpha, beta, delta_t, Smax)
        assert s_star < Smax

    @given(
        alpha=alpha_st(),
        beta=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        Smax=smax_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_steadyState_is_fixpoint(self, alpha, beta, delta_t, Smax):
        """steadyState_is_fixpoint: S* is a fixed point of decay + update.

        That is: strength_update(alpha, strength_decay(beta, S*, delta_t), Smax) == S*.
        """
        s_star = steady_state_strength(alpha, beta, delta_t, Smax)
        # Apply decay then update
        decayed = strength_decay(beta, s_star, delta_t)
        updated = strength_update(alpha, decayed, Smax)
        assert updated == pytest.approx(s_star, rel=1e-9)
