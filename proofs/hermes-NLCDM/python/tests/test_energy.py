"""Tests for NLCDM energy functions — mirrors Lean Phase 1 properties."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nlcdm_core import (
    WeightParams,
    sigmoid,
    smooth_weight,
    log_sum_exp,
    cosine_sim,
    local_energy,
    coupling_energy,
    total_energy,
    softmax,
)


# ---------------------------------------------------------------------------
# Sigmoid properties (mirror Lean sigmoid_pos, sigmoid_lt_one, sigmoid_strictMono)
# ---------------------------------------------------------------------------

class TestSigmoid:
    @given(st.floats(min_value=-50, max_value=50))
    def test_positive(self, x):
        """sigmoid(x) > 0 for all x (mirrors sigmoid_pos)"""
        assert sigmoid(x) > 0

    @given(st.floats(min_value=-30, max_value=30))
    def test_less_than_one(self, x):
        """sigmoid(x) < 1 for all x (mirrors sigmoid_lt_one)
        NOTE: float64 rounds to 1.0 for x > ~36, so range restricted."""
        assert sigmoid(x) < 1

    @given(
        st.floats(min_value=-30, max_value=30),
        st.floats(min_value=-30, max_value=30),
    )
    def test_strict_mono(self, a, b):
        """a < b → sigmoid(a) < sigmoid(b) (mirrors sigmoid_strictMono)
        NOTE: float64 can't distinguish values closer than ~1e-16."""
        assume(b - a > 1e-10)  # minimum separation for float64
        assert sigmoid(a) < sigmoid(b)


# ---------------------------------------------------------------------------
# Smooth weight properties (mirror Lean smoothWeight_zero, smoothWeight_continuous)
# ---------------------------------------------------------------------------

class TestSmoothWeight:
    def test_zero_similarity_gives_zero_weight(self):
        """W(0) = 0 (mirrors smoothWeight_zero)"""
        wp = WeightParams()
        assert smooth_weight(wp, 0.0) == pytest.approx(0.0, abs=1e-15)

    @given(st.floats(min_value=0.8, max_value=1.0))
    def test_attractive_regime(self, s):
        """s >> τ_high → W(s) ≈ s (positive coupling)"""
        wp = WeightParams(tau_high=0.65, k=20.0)
        w = smooth_weight(wp, s)
        assert w > 0, f"W({s}) = {w} should be positive in attractive regime"

    @given(st.floats(min_value=-1.0, max_value=-0.3))
    def test_repulsive_regime(self, s):
        """s << τ_low → W(s) ≈ -s > 0 (amplifies negative similarity).
        The repulsion comes from E_coupling = -½ W · ‖x_i‖ · ‖x_j‖:
        since W > 0 but the coupling energy formula has a negative sign,
        and the memories had negative cosine sim to begin with, the
        coupling energy becomes positive (repulsive in the energy landscape)."""
        wp = WeightParams(tau_high=0.65, tau_low=-0.1, k=20.0)
        w = smooth_weight(wp, s)
        # W(s) ≈ -s for s << τ_low, so W is positive (≈ |s|)
        assert w > 0, f"W({s}) = {w} should be positive (amplifying negative sim)"
        # And should be close to -s = |s|
        assert abs(w - (-s)) < 0.1, f"W({s}) = {w}, expected ≈ {-s}"

    @given(st.floats(min_value=0.1, max_value=0.4))
    def test_neutral_regime(self, s):
        """τ_low << s << τ_high → W(s) ≈ 0 (neutral)"""
        wp = WeightParams(tau_high=0.65, tau_low=-0.1, k=20.0)
        w = smooth_weight(wp, s)
        assert abs(w) < 0.1, f"W({s}) = {w} should be near zero in neutral regime"

    def test_continuous_no_jumps(self):
        """Sweep s and check no discontinuous jumps (mirrors smoothWeight_continuous)"""
        wp = WeightParams()
        s_vals = np.linspace(-1, 1, 10000)
        w_vals = smooth_weight(wp, s_vals)
        diffs = np.abs(np.diff(w_vals))
        max_jump = np.max(diffs)
        step = s_vals[1] - s_vals[0]
        # Max derivative bounded by k * max(|s|) * 0.25 ≈ 5.0
        # So max jump ≈ 5.0 * step ≈ 0.001
        assert max_jump < 0.01, f"Max jump {max_jump} too large for step {step}"


# ---------------------------------------------------------------------------
# Log-sum-exp properties (mirror Lean lse_ge_max, lse_le_max_plus)
# ---------------------------------------------------------------------------

class TestLogSumExp:
    @given(
        st.floats(min_value=0.01, max_value=10.0),
        arrays(np.float64, (10,), elements=st.floats(-5, 5)),
    )
    def test_ge_max(self, beta, z):
        """lse(β, z) ≥ max(z) (mirrors lse_ge_max)"""
        assume(np.all(np.isfinite(z)))
        lse = log_sum_exp(beta, z)
        assert lse >= np.max(z) - 1e-10

    @given(
        st.floats(min_value=0.01, max_value=10.0),
        arrays(np.float64, (10,), elements=st.floats(-5, 5)),
    )
    def test_le_max_plus_log_n(self, beta, z):
        """lse(β, z) ≤ max(z) + β⁻¹ log N (mirrors lse_le_max_plus)"""
        assume(np.all(np.isfinite(z)))
        N = len(z)
        lse = log_sum_exp(beta, z)
        assert lse <= np.max(z) + (1 / beta) * np.log(N) + 1e-10


# ---------------------------------------------------------------------------
# Energy bounded below (mirrors Lean totalEnergy_bounded_below)
# ---------------------------------------------------------------------------

class TestEnergyBounds:
    @given(st.integers(min_value=2, max_value=20))
    @settings(max_examples=50)
    def test_local_energy_bounded_below(self, N):
        """E_local(ξ) has a finite lower bound (mirrors localEnergy_bounded_below)"""
        d = 64
        beta = 1.0
        patterns = np.random.randn(N, d)
        patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)

        # Sample many random ξ, check energy is bounded
        energies = []
        for _ in range(100):
            xi = np.random.randn(d) * 3  # large norm
            e = local_energy(beta, patterns, xi)
            energies.append(e)

        min_e = min(energies)
        # Energy should be bounded below (not -∞)
        assert np.isfinite(min_e), f"Energy went to -inf: {min_e}"

    def test_coupling_energy_bounded(self):
        """|E_coupling| ≤ ½‖x_i‖·‖x_j‖ (mirrors couplingEnergy_abs_le)"""
        wp = WeightParams()
        d = 64
        for _ in range(100):
            x_i = np.random.randn(d)
            x_j = np.random.randn(d)
            e = coupling_energy(wp, x_i, x_j)
            bound = 0.5 * np.linalg.norm(x_i) * np.linalg.norm(x_j)
            assert abs(e) <= bound + 1e-10


# ---------------------------------------------------------------------------
# Softmax properties (mirrors Lean softmax_nonneg, softmax_sum_one)
# ---------------------------------------------------------------------------

class TestSoftmax:
    @given(
        st.floats(min_value=0.01, max_value=10.0),
        arrays(np.float64, (10,), elements=st.floats(-5, 5)),
    )
    def test_nonneg(self, beta, z):
        """All softmax entries ≥ 0 (mirrors softmax_nonneg)"""
        assume(np.all(np.isfinite(z)))
        p = softmax(beta, z)
        assert np.all(p >= -1e-15)

    @given(
        st.floats(min_value=0.01, max_value=10.0),
        arrays(np.float64, (10,), elements=st.floats(-5, 5)),
    )
    def test_sum_one(self, beta, z):
        """Softmax sums to 1 (mirrors softmax_sum_one)"""
        assume(np.all(np.isfinite(z)))
        p = softmax(beta, z)
        assert np.sum(p) == pytest.approx(1.0, abs=1e-10)

    def test_concentration(self):
        """Softmax concentrates on max entry as β grows
        (mirrors softmax_concentration)"""
        z = np.array([0.0, 0.5, 1.0, 0.3, 0.7])
        for beta in [1.0, 5.0, 20.0, 100.0]:
            p = softmax(beta, z)
            assert p[2] > p[0], f"β={beta}: max entry should dominate"
        # At high β, should be near 1
        p_high = softmax(100.0, z)
        assert p_high[2] > 0.99
