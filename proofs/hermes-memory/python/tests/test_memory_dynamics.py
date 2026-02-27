"""Property tests mapping Lean theorems from Sections 1-6 to Python.

25 theorems covering:
  Section 1 - Retention (forgetting curve)
  Section 2 - Strength update (discrete dynamics)
  Section 3 - Sigmoid function
  Section 4 - Scoring function
  Section 5 - Feedback loop (clamp01, importance_update)
  Section 6 - System-level score bounds
"""

import math

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hermes_memory.core import (
    retention,
    strength_update,
    strength_iter,
    sigmoid,
    ScoringWeights,
    score,
    clamp01,
    importance_update,
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
# Section 1: Retention (7 theorems)
# ============================================================


class TestRetention:
    """Properties of R(t) = exp(-t/S)."""

    @given(t=pos_time(), S=pos_real())
    @settings(max_examples=MAX_EXAMPLES)
    def test_retention_pos(self, t, S):
        """retention_pos: retention(t, S) > 0 for all t, S > 0.

        In exact math this always holds. In IEEE754, exp(-t/S) underflows to 0.0
        when t/S > ~708. We constrain to avoid underflow while still testing the
        property meaningfully.
        """
        assume(t / S < 700)  # avoid IEEE754 underflow
        assert retention(t, S) > 0

    @given(S=pos_real())
    @settings(max_examples=MAX_EXAMPLES)
    def test_retention_at_zero(self, S):
        """retention_at_zero: retention(0, S) == 1 for S != 0."""
        assert retention(0.0, S) == pytest.approx(1.0)

    @given(t=pos_time(), S=pos_real())
    @settings(max_examples=MAX_EXAMPLES)
    def test_retention_le_one(self, t, S):
        """retention_le_one: retention(t, S) <= 1 for t >= 0, S > 0."""
        assert retention(t, S) <= 1.0 + 1e-15

    @given(t=pos_time(), S=pos_real())
    @settings(max_examples=MAX_EXAMPLES)
    def test_retention_mem_Icc(self, t, S):
        """retention_mem_Icc: 0 <= retention(t, S) <= 1 for t >= 0, S > 0."""
        r = retention(t, S)
        assert 0.0 <= r <= 1.0 + 1e-15

    @given(
        t1=pos_time(),
        t2=pos_time(),
        S=pos_real(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_retention_antitone(self, t1, t2, S):
        """retention_antitone: t1 <= t2 ==> retention(t1, S) >= retention(t2, S)."""
        assume(t1 <= t2)
        assert retention(t1, S) >= retention(t2, S) - 1e-15

    @given(
        t=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        S1=pos_real(),
        S2=pos_real(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_retention_mono_strength(self, t, S1, S2):
        """retention_mono_strength: S1 <= S2 ==> retention(t, S1) <= retention(t, S2) for t > 0."""
        assume(S1 <= S2)
        assert retention(t, S1) <= retention(t, S2) + 1e-15

    @given(S=pos_real())
    @settings(max_examples=MAX_EXAMPLES)
    def test_retention_tendsto_zero(self, S):
        """retention_tendsto_zero: retention(t, S) -> 0 as t -> inf."""
        # Test at large t
        r = retention(1000.0, S)
        assert r < 1e-4


# ============================================================
# Section 2: Strength Update (6 theorems)
# ============================================================


class TestStrengthUpdate:
    """Properties of S' = S + alpha * (Smax - S)."""

    @given(alpha=alpha_st(), S=strength_st(), Smax=smax_st())
    @settings(max_examples=MAX_EXAMPLES)
    def test_strength_increases(self, alpha, S, Smax):
        """strength_increases: S < Smax, alpha > 0 ==> S < strength_update(alpha, S, Smax)."""
        assume(S < Smax)
        assert S < strength_update(alpha, S, Smax)

    @given(
        alpha=unit_interval(),
        S=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        Smax=smax_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_strength_le_max(self, alpha, S, Smax):
        """strength_le_max: alpha <= 1, S <= Smax ==> strength_update(alpha, S, Smax) <= Smax."""
        assume(S <= Smax)
        assert strength_update(alpha, S, Smax) <= Smax + 1e-12

    @given(
        alpha=unit_interval(),
        S=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        Smax=smax_st(),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_strength_nondecreasing(self, alpha, S, Smax):
        """strength_nondecreasing: alpha >= 0, S <= Smax ==> S <= strength_update(alpha, S, Smax)."""
        assume(S <= Smax)
        assert S <= strength_update(alpha, S, Smax) + 1e-12

    @given(alpha=alpha_st(), S=strength_st(), Smax=smax_st())
    @settings(max_examples=MAX_EXAMPLES)
    def test_strengthUpdate_alt(self, alpha, S, Smax):
        """strengthUpdate_alt: strength_update(alpha, S, Smax) == (1-alpha)*S + alpha*Smax."""
        result = strength_update(alpha, S, Smax)
        alt = (1.0 - alpha) * S + alpha * Smax
        assert result == pytest.approx(alt, rel=1e-12)

    @given(alpha=alpha_st(), S0=strength_st(), Smax=smax_st())
    @settings(max_examples=MAX_EXAMPLES)
    def test_strengthIter_closed(self, alpha, S0, Smax):
        """strengthIter_closed: strength_iter matches closed form Smax - (Smax-S0)*(1-alpha)^n."""
        assume(S0 <= Smax)
        n = 10
        # Iterate manually
        S = S0
        for _ in range(n):
            S = strength_update(alpha, S, Smax)
        closed_form = strength_iter(alpha, S0, Smax, n)
        assert S == pytest.approx(closed_form, rel=1e-9)

    @given(alpha=alpha_st(), S0=strength_st(), Smax=smax_st())
    @settings(max_examples=MAX_EXAMPLES)
    def test_strengthIter_tendsto(self, alpha, S0, Smax):
        """strengthIter_tendsto: strength_iter(alpha, S0, Smax, n) -> Smax as n -> inf.

        For small alpha, (1-alpha)^1000 can still be non-negligible.
        Use n=10000 and a relative tolerance that accounts for (1-alpha)^n residual.
        """
        assume(S0 <= Smax)
        val = strength_iter(alpha, S0, Smax, 10000)
        assert val == pytest.approx(Smax, rel=1e-4)


# ============================================================
# Section 3: Sigmoid (4 theorems)
# ============================================================


class TestSigmoid:
    """Properties of sigma(x) = 1 / (1 + exp(-x))."""

    @given(x=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=MAX_EXAMPLES)
    def test_sigmoid_pos(self, x):
        """sigmoid_pos: sigmoid(x) > 0."""
        assert sigmoid(x) > 0

    @given(x=st.floats(min_value=-36.0, max_value=36.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=MAX_EXAMPLES)
    def test_sigmoid_lt_one(self, x):
        """sigmoid_lt_one: sigmoid(x) < 1.

        In exact math this holds for all finite x. In IEEE754, exp(-x) underflows
        to 0.0 for x > ~36, making 1/(1+0) = 1.0 exactly. Constrain to |x| < 36.
        """
        assert sigmoid(x) < 1.0

    @given(x=st.floats(min_value=-36.0, max_value=36.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=MAX_EXAMPLES)
    def test_sigmoid_mem_Ioo(self, x):
        """sigmoid_mem_Ioo: 0 < sigmoid(x) < 1.

        Constrained to avoid IEEE754 underflow/overflow in exp.
        """
        s = sigmoid(x)
        assert 0.0 < s < 1.0

    @given(x=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=MAX_EXAMPLES)
    def test_sigmoid_mem_Icc(self, x):
        """sigmoid_mem_Icc: 0 <= sigmoid(x) <= 1."""
        s = sigmoid(x)
        assert 0.0 <= s <= 1.0

    def test_sigmoid_at_zero(self):
        """sigmoid_at_zero: sigmoid(0) == 0.5 (midpoint property)."""
        assert sigmoid(0.0) == pytest.approx(0.5, abs=1e-15)


# ============================================================
# Section 4-6: Scoring, Feedback (8 theorems)
# ============================================================


def scoring_weights_st():
    """Strategy that generates valid ScoringWeights (non-negative, sum to 1)."""
    return st.tuples(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ).filter(
        lambda t: sum(t) > 0.01  # avoid near-zero sum
    ).map(
        lambda t: _normalize_weights(t)
    )


def _normalize_weights(t):
    """Normalize a 4-tuple to sum to 1.0."""
    total = sum(t)
    return ScoringWeights(t[0] / total, t[1] / total, t[2] / total, t[3] / total)


class TestScoring:
    """Properties of score and feedback functions."""

    @given(
        w=scoring_weights_st(),
        rel=unit_interval(),
        rec=unit_interval(),
        imp=unit_interval(),
        act=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_score_nonneg(self, w, rel, rec, imp, act):
        """score_nonneg: score >= 0 when components >= 0."""
        assert score(w, rel, rec, imp, act) >= -1e-15

    @given(
        w=scoring_weights_st(),
        rel=unit_interval(),
        rec=unit_interval(),
        imp=unit_interval(),
        act=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_score_le_one(self, w, rel, rec, imp, act):
        """score_le_one: score <= 1 when components in [0,1]."""
        # score = w1*rel + w2*rec + w3*imp + w4*sigmoid(act)
        # Since rel,rec,imp in [0,1] and sigmoid in (0,1), and weights sum to 1:
        # score <= w1*1 + w2*1 + w3*1 + w4*1 = 1
        assert score(w, rel, rec, imp, act) <= 1.0 + 1e-12

    @given(
        w=scoring_weights_st(),
        rel=unit_interval(),
        rec=unit_interval(),
        imp=unit_interval(),
        act=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_score_mem_Icc(self, w, rel, rec, imp, act):
        """score_mem_Icc: score in [0,1] when components in [0,1]."""
        s = score(w, rel, rec, imp, act)
        assert -1e-15 <= s <= 1.0 + 1e-12


class TestFeedback:
    """Properties of clamp01 and importance_update."""

    @given(x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=MAX_EXAMPLES)
    def test_clamp01_mem_Icc(self, x):
        """clamp01_mem_Icc: 0 <= clamp01(x) <= 1."""
        c = clamp01(x)
        assert 0.0 <= c <= 1.0

    @given(
        imp=unit_interval(),
        delta=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        signal=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_importanceUpdate_mem_Icc(self, imp, delta, signal):
        """importanceUpdate_mem_Icc: 0 <= importance_update(imp, delta, signal) <= 1."""
        result = importance_update(imp, delta, signal)
        assert 0.0 <= result <= 1.0


class TestSystemScore:
    """System-level score bounds with retention and feedback."""

    @given(
        w=scoring_weights_st(),
        t=pos_time(),
        S=pos_real(),
        imp=unit_interval(),
        act=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_system_score_bounded(self, w, t, S, imp, act):
        """system_score_bounded: score with retention in [0,1] when all inputs valid."""
        rel = retention(t, S)  # in [0,1]
        rec = retention(t, S)  # use retention as recency proxy
        s = score(w, rel, rec, imp, act)
        assert -1e-15 <= s <= 1.0 + 1e-12

    @given(
        w=scoring_weights_st(),
        t=pos_time(),
        S=pos_real(),
        imp=unit_interval(),
        delta=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        signal=unit_interval(),
        act=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_system_score_bounded_after_feedback(self, w, t, S, imp, delta, signal, act):
        """system_score_bounded_after_feedback: score with importance_update still in [0,1]."""
        rel = retention(t, S)
        rec = retention(t, S)
        imp_new = importance_update(imp, delta, signal)
        s = score(w, rel, rec, imp_new, act)
        assert -1e-15 <= s <= 1.0 + 1e-12
