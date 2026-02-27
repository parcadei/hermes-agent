"""Spectral gap verification tests for the Hermes memory Markov chain.

The spectral gap of the transition matrix P is the key quantity governing
mixing. If the spectral gap gamma = 1 - |lambda_2| > 0, then:
  - The chain is geometrically ergodic
  - Total variation distance to stationarity decays as |lambda_2|^n
  - Mixing time ~ 1/gamma

These tests discretize the continuous chain onto an n_grid x n_grid lattice
and verify spectral properties of the resulting n_grid^2 x n_grid^2 matrix.

Parameters:
  alpha=0.1, beta=0.1, delta_t=1.0, S_max=10.0, T_temp=5.0
  K = 0.9548 < 1 (contraction condition satisfied)
  n_grid=30 (900 states -- large enough for accuracy, small enough to compute)
"""

import math

import numpy as np
import pytest

from hermes_memory.markov_chain import (
    build_transition_matrix,
    simulate_coupling,
    spectral_analysis,
)


# ── Parameters ────────────────────────────────────────────────────
ALPHA = 0.1
BETA = 0.1
DELTA_T = 1.0
S_MAX = 10.0
T_TEMP = 5.0
N_GRID = 30

# Additional parameter sets for cross-parameter tests
PARAM_SETS = [
    # (alpha, beta, delta_t, S_max, T_temp, name)
    (0.1, 0.1, 1.0, 10.0, 5.0, "primary"),
    (0.05, 0.2, 1.0, 5.0, 3.0, "fast-decay"),
    (0.08, 0.15, 2.0, 8.0, 4.0, "long-timestep"),
]

# Verify all satisfy contraction
for _a, _b, _dt, _sm, _T, _name in PARAM_SETS:
    _K = math.exp(-_b * _dt) + 0.25 * _a * _sm / _T
    assert _K < 1.0, f"Contraction violated for {_name}: K={_K}"


@pytest.fixture(scope="module")
def transition_matrix():
    """Build the transition matrix P and grid for primary parameters."""
    return build_transition_matrix(
        ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_grid=N_GRID,
    )


@pytest.fixture(scope="module")
def spectral_result(transition_matrix):
    """Compute spectral analysis of the primary transition matrix."""
    P, _ = transition_matrix
    return spectral_analysis(P)


class TestTransitionMatrixStochastic:
    """Verify the transition matrix is a valid stochastic matrix."""

    def test_transition_matrix_is_stochastic(self, transition_matrix):
        """Build transition matrix P. Verify all rows sum to 1 (within 1e-10).
        Verify all entries >= 0.

        A row-stochastic matrix has:
          1. P[i,j] >= 0 for all i,j (non-negativity)
          2. sum_j P[i,j] = 1 for all i (row normalization)
        """
        P, _ = transition_matrix

        # All entries non-negative
        assert np.all(P >= -1e-15), (
            f"Negative entry found: min={P.min():.2e}"
        )

        # All rows sum to 1
        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-10,
            err_msg="Not all rows sum to 1",
        )


class TestSpectralGap:
    """Verify the spectral gap is positive -- the core ergodicity proof."""

    def test_spectral_gap_positive(self, spectral_result):
        """Compute spectral_analysis. Assert spectral_gap > 0 and lambda_2 < 1.

        This IS the geometric ergodicity proof: if lambda_2 < 1, then
        the chain converges to its unique stationary distribution at
        geometric rate |lambda_2|^n. The spectral gap gamma = 1 - lambda_2
        controls the rate.
        """
        assert spectral_result["spectral_gap"] > 0, (
            f"Spectral gap not positive: {spectral_result['spectral_gap']}"
        )
        assert spectral_result["lambda_2"] < 1.0, (
            f"lambda_2 not < 1: {spectral_result['lambda_2']}"
        )

    @pytest.mark.slow
    def test_spectral_gap_across_parameters(self):
        """Test 3 parameter sets. All should have spectral_gap > 0.01.

        The spectral gap should be robustly positive across different
        parameter regimes that satisfy the contraction condition.
        """
        for alpha, beta, dt, s_max, t_temp, name in PARAM_SETS:
            P, _ = build_transition_matrix(
                alpha, beta, dt, s_max, t_temp, n_grid=N_GRID,
            )
            result = spectral_analysis(P)

            print(f"\n{name}: spectral_gap={result['spectral_gap']:.4f}, "
                  f"lambda_2={result['lambda_2']:.4f}, "
                  f"mixing_time={result['mixing_time']:.2f}")

            assert result["spectral_gap"] > 0.01, (
                f"{name}: spectral gap {result['spectral_gap']:.6f} <= 0.01"
            )


class TestStationaryDistribution:
    """Verify properties of the stationary distribution pi."""

    def test_stationary_distribution_is_probability(self, spectral_result):
        """Assert all entries >= 0 (within tolerance). Assert sum ~= 1.

        The stationary distribution pi must be a valid probability vector:
        non-negative entries that sum to 1.
        """
        pi = spectral_result["stationary"]

        # Non-negative (allow small numerical noise)
        assert np.all(pi >= -1e-10), (
            f"Negative stationary entry: min={pi.min():.2e}"
        )

        # Sums to 1
        assert abs(pi.sum() - 1.0) < 1e-6, (
            f"Stationary distribution sums to {pi.sum():.10f}, not 1"
        )

    def test_stationary_distribution_concentrated(self, spectral_result):
        """Verify the stationary distribution has most mass concentrated
        in a small region.

        We check that the entropy of pi is significantly less than the
        maximum entropy log(n_states). A uniform distribution has entropy
        = log(900) = 6.80. The stationary distribution should have much
        lower entropy, indicating concentration.

        Additionally verify that the mode captures significant mass:
        the 7x7 neighborhood around the mode (49 states out of 900)
        should contain at least 20% of the total mass.
        """
        pi = spectral_result["stationary"]
        n_states = N_GRID * N_GRID

        # Entropy check
        # Filter out zero/negative entries for log
        pi_pos = pi[pi > 1e-30]
        entropy = -np.sum(pi_pos * np.log(pi_pos))
        max_entropy = math.log(n_states)

        # Entropy should be significantly less than maximum
        # (ratio < 0.7 means the distribution is far from uniform)
        entropy_ratio = entropy / max_entropy
        assert entropy_ratio < 0.7, (
            f"Distribution too spread: entropy ratio = {entropy_ratio:.4f} "
            f"(entropy={entropy:.4f}, max={max_entropy:.4f})"
        )

        # Mode concentration check
        mode_idx = np.argmax(pi)
        mode_i1 = mode_idx // N_GRID
        mode_i2 = mode_idx % N_GRID

        # Mass in 7x7 neighborhood of mode
        mass_near_mode = 0.0
        for i1 in range(max(0, mode_i1 - 3), min(N_GRID, mode_i1 + 4)):
            for i2 in range(max(0, mode_i2 - 3), min(N_GRID, mode_i2 + 4)):
                mass_near_mode += pi[i1 * N_GRID + i2]

        assert mass_near_mode > 0.20, (
            f"Mode neighborhood mass = {mass_near_mode:.4f} < 0.20. "
            f"Distribution not concentrated around mode at "
            f"grid[{mode_i1}], grid[{mode_i2}]"
        )


class TestSpectralCouplingConsistency:
    """Cross-validate spectral and coupling analyses."""

    @pytest.mark.slow
    def test_lambda2_matches_coupling_rate(self, spectral_result):
        """The spectral gap should predict the coupling rate.

        Verify that mixing_time (from spectral) is within order of magnitude
        of mean coupling_time (from coupling tests).

        The spectral mixing time is 1/gamma where gamma is the spectral gap.
        The coupling time is an upper bound on mixing time (by the coupling
        inequality). So we expect:

            coupling_time ~ O(mixing_time)

        We use a coarse check:
            0.1 * mixing_time < mean_coupling_time < 10 * mixing_time
        """
        mixing_time = spectral_result["mixing_time"]

        # Run coupling experiments
        coupling_times = []
        for seed in range(50):
            _, _, tau = simulate_coupling(
                ALPHA, BETA, DELTA_T, S_MAX, T_TEMP,
                n_steps=5000, seed=seed,
            )
            assert tau is not None, f"Coupling failed with seed={seed}"
            coupling_times.append(tau)

        mean_coupling = np.mean(coupling_times)

        print(f"\nSpectral mixing time: {mixing_time:.2f}")
        print(f"Mean coupling time:   {mean_coupling:.2f}")
        print(f"Ratio (coupling/spectral): {mean_coupling / mixing_time:.2f}")

        # Order-of-magnitude consistency
        assert 0.1 * mixing_time < mean_coupling < 10 * mixing_time, (
            f"Spectral and coupling disagree: "
            f"mixing_time={mixing_time:.2f}, "
            f"mean_coupling={mean_coupling:.2f}, "
            f"ratio={mean_coupling / mixing_time:.2f}"
        )
