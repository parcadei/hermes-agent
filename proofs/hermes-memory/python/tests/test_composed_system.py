"""Property tests mapping Lean theorems from Sections 10-12 to Python.

12 theorems covering:
  expected_strength_update: non-negativity, upper bound
  Fixed-point properties: existence, uniqueness, convergence
  Lipschitz continuity
  Composed system safety guarantees
"""

import math

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hermes_memory.core import (
    expected_strength_update,
    composed_expected_map,
    composed_contraction_factor,
    soft_select,
    novelty_bonus,
    boosted_score,
    sigmoid,
)

from tests.strategies import alpha_st, pos_real, smax_st, unit_interval, temperature_st

MAX_EXAMPLES = 200

# Constrained strategies for composed system (need beta, delta_t that don't cause overflow)
beta_st = lambda: st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False)
delta_t_st = lambda: st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
# Smaller Smax to keep contraction conditions feasible
small_smax = lambda: st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)


def select_prob_factory(T):
    """Create a selection probability function with temperature T."""
    def select_prob(s1, s2):
        return soft_select(s1, s2, T)
    return select_prob


# ============================================================
# expected_strength_update (2 theorems)
# ============================================================


class TestExpectedStrengthUpdate:
    """Properties of E[S'] = (1 - q*alpha)*exp(-beta*delta_t)*S + q*alpha*Smax."""

    @given(
        alpha=alpha_st(),
        beta=beta_st(),
        delta_t=delta_t_st(),
        Smax=small_smax(),
        q=unit_interval(),
        S=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_expectedStrengthUpdate_nonneg(self, alpha, beta, delta_t, Smax, q, S):
        """expectedStrengthUpdate_nonneg: E[S'] >= 0 for valid params."""
        assume(S <= Smax)
        assume(q * alpha <= 1.0)
        result = expected_strength_update(alpha, beta, delta_t, Smax, q, S)
        assert result >= -1e-12

    @given(
        alpha=alpha_st(),
        beta=beta_st(),
        delta_t=delta_t_st(),
        Smax=small_smax(),
        q=unit_interval(),
        S=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_expectedStrengthUpdate_le_Smax(self, alpha, beta, delta_t, Smax, q, S):
        """expectedStrengthUpdate_le_Smax: E[S'] <= Smax for S in [0, Smax].

        E[S'] = (1-q*alpha)*e^(-beta*delta_t)*S + q*alpha*Smax
              <= (1-q*alpha)*S + q*alpha*Smax  (since e^(-x) <= 1)
              <= (1-q*alpha)*Smax + q*alpha*Smax = Smax
        """
        assume(S <= Smax)
        assume(q * alpha <= 1.0)
        result = expected_strength_update(alpha, beta, delta_t, Smax, q, S)
        assert result <= Smax + 1e-9


# ============================================================
# Fixed-point properties (3 theorems)
# ============================================================


class TestFixedPoint:
    """Fixed-point existence for composed_expected_map."""

    @given(
        alpha=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        Smax=st.floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_fixedPoint_lt_Smax(self, alpha, beta, delta_t, Smax, T):
        """fixedPoint_lt_Smax: If S = T(q,S) and q > 0, then S < Smax.

        Iterate the expected map until convergence, verify S* < Smax.
        """
        select_fn = select_prob_factory(T)
        S = (Smax / 2.0, Smax / 2.0)
        for _ in range(5000):
            S_new = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, S)
            S = S_new
        assert S[0] < Smax
        assert S[1] < Smax

    @given(
        alpha=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        Smax=st.floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_composedFixedPoint_fst_lt_Smax(self, alpha, beta, delta_t, Smax, T):
        """composedFixedPoint_fst_lt_Smax: first component of composed fixed point < Smax."""
        select_fn = select_prob_factory(T)
        S = (Smax * 0.3, Smax * 0.7)
        for _ in range(5000):
            S = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, S)
        assert S[0] < Smax

    @given(
        alpha=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        Smax=st.floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_composedFixedPoint_snd_lt_Smax(self, alpha, beta, delta_t, Smax, T):
        """composedFixedPoint_snd_lt_Smax: second component < Smax."""
        select_fn = select_prob_factory(T)
        S = (Smax * 0.7, Smax * 0.3)
        for _ in range(5000):
            S = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, S)
        assert S[1] < Smax


# ============================================================
# Contraction factor (1 theorem)
# ============================================================


class TestContractionFactor:
    """Properties of K = exp(-beta*delta_t) + L*alpha*Smax."""

    @given(
        alpha=st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        Smax=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        L=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_composedContractionFactor_lt_one(self, alpha, beta, delta_t, Smax, L):
        """composedContractionFactor_lt_one: K < 1 when L*alpha*Smax < 1 - exp(-beta*delta_t)."""
        e_decay = math.exp(-beta * delta_t)
        assume(L * alpha * Smax < 1.0 - e_decay)
        K = composed_contraction_factor(beta, delta_t, L, alpha, Smax)
        assert K < 1.0


# ============================================================
# Lipschitz continuity (3 theorems)
# ============================================================


class TestLipschitz:
    """Lipschitz continuity of expected_strength_update."""

    @given(
        alpha=alpha_st(),
        beta=beta_st(),
        delta_t=delta_t_st(),
        Smax=small_smax(),
        q=unit_interval(),
        S=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        S_prime=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_expectedUpdate_lipschitz_in_S(self, alpha, beta, delta_t, Smax, q, S, S_prime):
        """expectedUpdate_lipschitz_in_S: |T(q,S)-T(q,S')| <= e^(-beta*delta_t)|S-S'|.

        Holding q fixed, the Lipschitz constant in S is (1-q*alpha)*e^(-beta*delta_t) <= e^(-beta*delta_t).
        """
        assume(S <= Smax)
        assume(S_prime <= Smax)
        assume(q * alpha <= 1.0)
        T_S = expected_strength_update(alpha, beta, delta_t, Smax, q, S)
        T_S_prime = expected_strength_update(alpha, beta, delta_t, Smax, q, S_prime)
        e_decay = math.exp(-beta * delta_t)
        assert abs(T_S - T_S_prime) <= e_decay * abs(S - S_prime) + 1e-12

    @given(
        alpha=alpha_st(),
        beta=beta_st(),
        delta_t=delta_t_st(),
        Smax=small_smax(),
        q=unit_interval(),
        q_prime=unit_interval(),
        S=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_expectedUpdate_lipschitz_in_q(self, alpha, beta, delta_t, Smax, q, q_prime, S):
        """expectedUpdate_lipschitz_in_q: |T(q,S)-T(q',S)| <= alpha*Smax*|q-q'|.

        T(q,S) = (1-q*alpha)*e^(-beta*dt)*S + q*alpha*Smax
        dT/dq = -alpha*e^(-beta*dt)*S + alpha*Smax = alpha*(Smax - e^(-beta*dt)*S)
        |dT/dq| <= alpha*Smax  (since e^(-beta*dt)*S >= 0)
        """
        assume(S <= Smax)
        assume(q * alpha <= 1.0)
        assume(q_prime * alpha <= 1.0)
        T_q = expected_strength_update(alpha, beta, delta_t, Smax, q, S)
        T_q_prime = expected_strength_update(alpha, beta, delta_t, Smax, q_prime, S)
        assert abs(T_q - T_q_prime) <= alpha * Smax * abs(q - q_prime) + 1e-12

    @given(
        alpha=alpha_st(),
        beta=beta_st(),
        delta_t=delta_t_st(),
        Smax=small_smax(),
        q=unit_interval(),
        q_prime=unit_interval(),
        S=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        S_prime=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_expectedUpdate_lipschitz_full(self, alpha, beta, delta_t, Smax, q, q_prime, S, S_prime):
        """expectedUpdate_lipschitz_full: |T(q,S)-T(q',S')| <= e^(-beta*dt)|S-S'| + alpha*Smax|q-q'|.

        Triangle inequality applied to the two Lipschitz bounds.
        """
        assume(S <= Smax)
        assume(S_prime <= Smax)
        assume(q * alpha <= 1.0)
        assume(q_prime * alpha <= 1.0)
        T1 = expected_strength_update(alpha, beta, delta_t, Smax, q, S)
        T2 = expected_strength_update(alpha, beta, delta_t, Smax, q_prime, S_prime)
        e_decay = math.exp(-beta * delta_t)
        bound = e_decay * abs(S - S_prime) + alpha * Smax * abs(q - q_prime)
        assert abs(T1 - T2) <= bound + 1e-10


# ============================================================
# Composed system properties (3 theorems)
# ============================================================


class TestComposedSystemProperties:
    """Properties of the composed 2-memory system at fixed point."""

    @given(
        alpha=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        Smax=st.floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_composedFixedPoint_selection_pos(self, alpha, beta, delta_t, Smax, T):
        """composedFixedPoint_selection_pos: soft_select at fixed point is in (0,1)."""
        select_fn = select_prob_factory(T)
        S = (Smax / 2.0, Smax / 2.0)
        for _ in range(5000):
            S = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, S)
        p = soft_select(S[0], S[1], T)
        assert 0.0 < p < 1.0

    @given(
        N0=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        gamma=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
        t=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        base=unit_interval(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_composedNoveltyBonus_bounded(self, N0, gamma, t, base):
        """composedNoveltyBonus_bounded: boosted_score <= 1 + N0."""
        bs = boosted_score(base, N0, gamma, t)
        assert bs <= 1.0 + N0 + 1e-12

    @given(
        alpha=st.floats(min_value=0.05, max_value=0.5, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        Smax=st.floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_composedSystem_safe(self, alpha, beta, delta_t, Smax, T):
        """composedSystem_safe: all 3 safety guarantees hold at fixed point.

        1. Anti-lock-in: S* < Smax (both components)
        2. Anti-thrashing: soft_select in (0,1)
        3. Both components are non-negative
        """
        select_fn = select_prob_factory(T)
        S = (Smax / 2.0, Smax / 2.0)
        for _ in range(5000):
            S = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, S)

        # 1. Anti-lock-in
        assert S[0] < Smax, f"S[0]={S[0]} not < Smax={Smax}"
        assert S[1] < Smax, f"S[1]={S[1]} not < Smax={Smax}"

        # 2. Anti-thrashing
        p = soft_select(S[0], S[1], T)
        assert 0.0 < p < 1.0, f"selection prob {p} not in (0,1)"

        # 3. Non-negativity
        assert S[0] >= 0, f"S[0]={S[0]} is negative"
        assert S[1] >= 0, f"S[1]={S[1]} is negative"
