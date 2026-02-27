"""Property tests mapping Lean theorems from Section 8 to Python.

9 theorems covering:
  sigmoid monotonicity
  soft_select: positivity, bounds, complementarity, monotonicity, equal-score case
  Anti-thrashing guarantee
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hermes_memory.core import sigmoid, soft_select

from tests.strategies import pos_real, temperature_st

MAX_EXAMPLES = 200


# ============================================================
# Sigmoid monotonicity (1 theorem)
# ============================================================


class TestSigmoidMonotone:
    """Properties of sigmoid used by soft_select."""

    @given(
        x1=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        x2=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_sigmoid_monotone(self, x1, x2):
        """sigmoid_monotone: x1 <= x2 ==> sigmoid(x1) <= sigmoid(x2)."""
        assume(x1 <= x2)
        assert sigmoid(x1) <= sigmoid(x2) + 1e-15


# ============================================================
# Soft selection (8 theorems)
# ============================================================


class TestSoftSelect:
    """Properties of P(select 1) = sigmoid((s1 - s2) / T)."""

    @given(
        s1=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        s2=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=temperature_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_softSelect_pos(self, s1, s2, T):
        """softSelect_pos: soft_select > 0.

        soft_select = sigmoid((s1-s2)/T). In exact math, sigmoid > 0 always.
        In IEEE754, sigmoid underflows for very negative args. Constrain |s1-s2|/T < 36.
        """
        assume(abs(s1 - s2) / T < 36)
        assert soft_select(s1, s2, T) > 0

    @given(
        s1=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        s2=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=temperature_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_softSelect_lt_one(self, s1, s2, T):
        """softSelect_lt_one: soft_select < 1.

        sigmoid((s1-s2)/T) < 1 in exact math. In IEEE754, overflows to 1.0
        when (s1-s2)/T > ~36. Constrain to avoid.
        """
        assume(abs(s1 - s2) / T < 36)
        assert soft_select(s1, s2, T) < 1.0

    @given(
        s1=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        s2=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=temperature_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_softSelect_mem_Ioo(self, s1, s2, T):
        """softSelect_mem_Ioo: 0 < soft_select < 1.

        Constrained to avoid IEEE754 underflow/overflow in sigmoid.
        """
        assume(abs(s1 - s2) / T < 36)
        p = soft_select(s1, s2, T)
        assert 0.0 < p < 1.0

    @given(
        s1=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        s2=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=temperature_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_softSelect_complementary(self, s1, s2, T):
        """softSelect_complementary: soft_select(s1,s2,T) + soft_select(s2,s1,T) == 1."""
        p1 = soft_select(s1, s2, T)
        p2 = soft_select(s2, s1, T)
        assert p1 + p2 == pytest.approx(1.0, abs=1e-12)

    @given(
        s1=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        s1_prime=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        s2=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=temperature_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_softSelect_monotone_score(self, s1, s1_prime, s2, T):
        """softSelect_monotone_score: s1 <= s1' ==> soft_select(s1,s2,T) <= soft_select(s1',s2,T)."""
        assume(s1 <= s1_prime)
        assert soft_select(s1, s2, T) <= soft_select(s1_prime, s2, T) + 1e-15

    @given(
        s=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=temperature_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_softSelect_equal_scores(self, s, T):
        """softSelect_equal_scores: soft_select(s, s, T) == sigmoid(0)."""
        assert soft_select(s, s, T) == pytest.approx(sigmoid(0.0), abs=1e-15)

    def test_sigmoid_at_zero(self):
        """sigmoid_at_zero: sigmoid(0) == 0.5."""
        assert sigmoid(0.0) == pytest.approx(0.5, abs=1e-15)

    @given(
        s=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=temperature_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_softSelect_equal_is_half(self, s, T):
        """softSelect_equal_is_half: soft_select(s, s, T) == 0.5."""
        assert soft_select(s, s, T) == pytest.approx(0.5, abs=1e-15)
