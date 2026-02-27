"""Property tests mapping Lean theorems from Sections 13-18 (ContractionWiring) to Python.

8 theorems covering:
  composedDomain: containment, non-emptiness
  composedExpectedMap: maps-to, Lipschitz, contraction
  stationaryState: existence, uniqueness, convergence
  stableStationaryState: safety guarantees
"""

import math

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hermes_memory.core import (
    composed_domain_contains,
    composed_expected_map,
    composed_contraction_factor,
    expected_strength_update,
    soft_select,
)

from tests.strategies import alpha_st, smax_st, temperature_st

MAX_EXAMPLES = 200

# Tighter strategies for contraction tests
c_alpha = lambda: st.floats(min_value=0.01, max_value=0.3, allow_nan=False, allow_infinity=False)
c_beta = lambda: st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False)
c_delta_t = lambda: st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False)
c_smax = lambda: st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)
c_temp = lambda: st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)


def select_prob_factory(T):
    """Create a selection probability function with temperature T."""
    def select_prob(s1, s2):
        return soft_select(s1, s2, T)
    return select_prob


def iterate_to_fixed_point(select_fn, alpha, beta, delta_t, Smax, S0, n_iter=10000):
    """Iterate composed_expected_map to convergence."""
    S = S0
    for _ in range(n_iter):
        S = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, S)
    return S


# ============================================================
# composedDomain (2 theorems)
# ============================================================


class TestComposedDomain:
    """Properties of the domain [0, Smax]^2."""

    @given(Smax=c_smax())
    @settings(max_examples=MAX_EXAMPLES)
    def test_composedDomain_contains(self, Smax):
        """composedDomain_contains: (0,0) and (Smax,Smax) are in [0,Smax]^2."""
        assert composed_domain_contains(Smax, (0.0, 0.0))
        assert composed_domain_contains(Smax, (Smax, Smax))

    @given(Smax=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=MAX_EXAMPLES)
    def test_composedDomain_nonempty(self, Smax):
        """composedDomain_nonempty: domain is nonempty for Smax > 0."""
        # The domain [0, Smax]^2 is nonempty iff Smax >= 0.
        # We can witness it with (0, 0).
        assert composed_domain_contains(Smax, (0.0, 0.0))


# ============================================================
# composedExpectedMap (3 theorems)
# ============================================================


class TestComposedExpectedMap:
    """Properties of T: [0,Smax]^2 -> [0,Smax]^2."""

    @given(
        alpha=c_alpha(),
        beta=c_beta(),
        delta_t=c_delta_t(),
        Smax=c_smax(),
        T=c_temp(),
        s1=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        s2=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_composedExpectedMap_mapsTo(self, alpha, beta, delta_t, Smax, T, s1, s2):
        """composedExpectedMap_mapsTo: T maps [0,Smax]^2 -> [0,Smax]^2."""
        assume(s1 <= Smax)
        assume(s2 <= Smax)
        select_fn = select_prob_factory(T)
        S_in = (s1, s2)
        S_out = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, S_in)
        assert composed_domain_contains(Smax, S_in)
        # Output must also be in domain
        assert S_out[0] >= -1e-12, f"S_out[0]={S_out[0]} < 0"
        assert S_out[1] >= -1e-12, f"S_out[1]={S_out[1]} < 0"
        assert S_out[0] <= Smax + 1e-9, f"S_out[0]={S_out[0]} > Smax={Smax}"
        assert S_out[1] <= Smax + 1e-9, f"S_out[1]={S_out[1]} > Smax={Smax}"

    @given(
        alpha=c_alpha(),
        beta=c_beta(),
        delta_t=c_delta_t(),
        Smax=c_smax(),
        T=c_temp(),
        s1a=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        s2a=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        s1b=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        s2b=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_composedExpectedMap_lipschitz_on_domain(self, alpha, beta, delta_t, Smax, T,
                                                      s1a, s2a, s1b, s2b):
        """composedExpectedMap_lipschitz_on_domain: |T(S)-T(S')| <= K*dist(S,S') on domain.

        For the composed map T(S) = (T_1(p(S),S_1), T_2(1-p(S),S_2)):
          - T_i has Lip e^(-beta*dt) in S_i (holding q fixed)
          - T_i has Lip alpha*Smax in q
          - soft_select has Lip <= 1/(4T) per coordinate (since dsigmoid/dx <= 1/4
            and the argument is (s1-s2)/T, giving 2/(4T) = 1/(2T) total)
          - So the full Lip constant is e^(-beta*dt) + alpha*Smax/(2T)
        We use the L1 distance on (S1,S2) to get a correct bound.
        """
        assume(s1a <= Smax and s2a <= Smax)
        assume(s1b <= Smax and s2b <= Smax)
        select_fn = select_prob_factory(T)
        Sa = (s1a, s2a)
        Sb = (s1b, s2b)
        Ta = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, Sa)
        Tb = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, Sb)

        # Use L1 distance for both input and output
        dist_in = abs(s1a - s1b) + abs(s2a - s2b)
        dist_out = abs(Ta[0] - Tb[0]) + abs(Ta[1] - Tb[1])

        # Generous Lipschitz constant: the q-variation contributes alpha*Smax*Lip_select
        # and the S-variation contributes e^(-beta*dt).
        # soft_select Lip w.r.t. (s1,s2) in L1 is at most 1/(2T) (derivative is 1/4 per unit)
        e_decay = math.exp(-beta * delta_t)
        K = e_decay + alpha * Smax / (2.0 * T)
        # Only check when K is meaningful (not trivially large)
        if K < 100.0:
            assert dist_out <= K * dist_in + 1e-9

    @given(
        alpha=st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        Smax=st.floats(min_value=1.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        T=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_composedExpectedMap_contractingWith(self, alpha, beta, delta_t, Smax, T):
        """composedExpectedMap_contractingWith: K < 1 under contraction condition.

        With small alpha, large beta*delta_t, and large T, the map is a contraction.
        """
        L_sigmoid = 0.25
        L = L_sigmoid / T
        e_decay = math.exp(-beta * delta_t)
        # Verify the contraction condition holds for these parameters
        assume(L * alpha * Smax < 1.0 - e_decay)
        K = composed_contraction_factor(beta, delta_t, L, alpha, Smax)
        assert K < 1.0


# ============================================================
# stationaryState (3 theorems)
# ============================================================


class TestStationaryState:
    """Existence, uniqueness, and convergence of the stationary state."""

    @given(
        alpha=st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        Smax=st.floats(min_value=1.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        T=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_stationaryState_exists_unique(self, alpha, beta, delta_t, Smax, T):
        """stationaryState_exists_unique: T has a unique fixed point.

        Iterate from multiple starting points, all converge to the same S*.
        Uniqueness requires the contraction condition K < 1, which holds when
        the temperature T is high enough relative to alpha*Smax and the decay
        rate is sufficient. We constrain parameters to ensure this.
        """
        # Verify contraction condition holds
        e_decay = math.exp(-beta * delta_t)
        L = 0.25 / T  # Lipschitz constant of soft_select
        K = e_decay + L * alpha * Smax
        assume(K < 1.0)

        select_fn = select_prob_factory(T)
        starts = [
            (0.0, 0.0),
            (Smax, Smax),
            (0.0, Smax),
            (Smax, 0.0),
            (Smax / 2.0, Smax / 2.0),
        ]
        fixed_points = []
        for s0 in starts:
            fp = iterate_to_fixed_point(select_fn, alpha, beta, delta_t, Smax, s0)
            fixed_points.append(fp)

        # All fixed points should be the same within tolerance
        ref = fixed_points[0]
        for fp in fixed_points[1:]:
            assert fp[0] == pytest.approx(ref[0], abs=1e-4), \
                f"Fixed points differ: {ref} vs {fp}"
            assert fp[1] == pytest.approx(ref[1], abs=1e-4), \
                f"Fixed points differ: {ref} vs {fp}"

    @given(
        alpha=st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        Smax=st.floats(min_value=1.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        T=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_stationaryState_convergent(self, alpha, beta, delta_t, Smax, T):
        """stationaryState_convergent: iterates converge to S*.

        Check that successive iterates get closer to the fixed point.
        Requires the contraction condition K < 1.
        """
        e_decay = math.exp(-beta * delta_t)
        L = 0.25 / T
        K = e_decay + L * alpha * Smax
        assume(K < 1.0)

        select_fn = select_prob_factory(T)
        # First find the fixed point with many iterations
        S_star = iterate_to_fixed_point(select_fn, alpha, beta, delta_t, Smax,
                                         (Smax / 3.0, Smax * 2.0 / 3.0))

        # Now iterate from a different start and check convergence
        S = (0.1, Smax * 0.9)
        prev_dist = max(abs(S[0] - S_star[0]), abs(S[1] - S_star[1]))

        # After 100 iterations, distance should be much smaller
        for _ in range(100):
            S = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, S)
        curr_dist = max(abs(S[0] - S_star[0]), abs(S[1] - S_star[1]))
        assert curr_dist < prev_dist, f"Not converging: {prev_dist} -> {curr_dist}"


# ============================================================
# stableStationaryState (1 theorem)
# ============================================================


class TestStableStationaryState:
    """The fixed point satisfies all 3 safety guarantees."""

    @given(
        alpha=st.floats(min_value=0.05, max_value=0.3, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        delta_t=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        Smax=st.floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        T=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_stableStationaryState_safe(self, alpha, beta, delta_t, Smax, T):
        """stableStationaryState_safe: fixed point satisfies all 3 safety guarantees.

        1. Anti-lock-in: S*_1, S*_2 < Smax
        2. Anti-thrashing: soft_select(S*_1, S*_2, T) in (0,1)
        3. Non-negativity: S*_1, S*_2 >= 0
        """
        select_fn = select_prob_factory(T)
        S_star = iterate_to_fixed_point(select_fn, alpha, beta, delta_t, Smax,
                                         (Smax / 2.0, Smax / 2.0))

        # Verify it is actually a fixed point (self-consistency)
        S_next = composed_expected_map(select_fn, alpha, beta, delta_t, Smax, S_star)
        assert S_next[0] == pytest.approx(S_star[0], abs=1e-6)
        assert S_next[1] == pytest.approx(S_star[1], abs=1e-6)

        # Safety guarantee 1: Anti-lock-in
        assert S_star[0] < Smax, f"Anti-lock-in violated: S*[0]={S_star[0]} >= Smax={Smax}"
        assert S_star[1] < Smax, f"Anti-lock-in violated: S*[1]={S_star[1]} >= Smax={Smax}"

        # Safety guarantee 2: Anti-thrashing
        p = soft_select(S_star[0], S_star[1], T)
        assert 0.0 < p < 1.0, f"Anti-thrashing violated: p={p}"

        # Safety guarantee 3: Non-negativity
        assert S_star[0] >= 0, f"Non-negativity violated: S*[0]={S_star[0]}"
        assert S_star[1] >= 0, f"Non-negativity violated: S*[1]={S_star[1]}"
