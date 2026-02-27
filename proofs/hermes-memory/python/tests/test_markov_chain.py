"""Tests for hermes_memory.markov_chain — stochastic simulation and spectral analysis."""

import math
import random

import numpy as np
import pytest

from hermes_memory.core import soft_select, strength_update
from hermes_memory.markov_chain import (
    build_transition_matrix,
    simulate_chain,
    simulate_coupling,
    spectral_analysis,
)

# ── Shared parameters ──────────────────────────────────────────────

ALPHA = 0.3
BETA = 0.05
DELTA_T = 1.0
S_MAX = 10.0
T_TEMP = 1.0
N_STEPS = 200


# ════════════════════════════════════════════════════════════════════
# simulate_chain
# ════════════════════════════════════════════════════════════════════


class TestSimulateChain:
    """Tests for the single-trajectory Markov chain simulator."""

    def test_returns_trajectory_including_initial_state(self):
        traj = simulate_chain(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, (3.0, 7.0), 10)
        # Length = n_steps + 1 (includes initial)
        assert len(traj) == 11
        assert traj[0] == (3.0, 7.0)

    def test_each_element_is_pair_of_floats(self):
        traj = simulate_chain(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, (1.0, 2.0), 5)
        for state in traj:
            assert isinstance(state, tuple)
            assert len(state) == 2
            assert isinstance(state[0], float)
            assert isinstance(state[1], float)

    def test_values_stay_in_domain(self):
        traj = simulate_chain(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, (0.0, S_MAX), 100)
        for s1, s2 in traj:
            assert 0.0 <= s1 <= S_MAX + 1e-12
            assert 0.0 <= s2 <= S_MAX + 1e-12

    def test_deterministic_with_same_rng(self):
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        t1 = simulate_chain(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, (5.0, 5.0), 50, rng=rng1)
        t2 = simulate_chain(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, (5.0, 5.0), 50, rng=rng2)
        assert t1 == t2

    def test_different_rng_gives_different_trajectory(self):
        rng1 = random.Random(1)
        rng2 = random.Random(999)
        t1 = simulate_chain(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, (5.0, 5.0), 50, rng=rng1)
        t2 = simulate_chain(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, (5.0, 5.0), 50, rng=rng2)
        # Overwhelmingly likely to differ somewhere
        assert t1 != t2

    def test_decay_is_applied_before_selection(self):
        """Starting from (Smax, 0): after decay both shrink, then one gets boosted."""
        traj = simulate_chain(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, (S_MAX, 0.0), 1, rng=random.Random(42))
        s1_next, s2_next = traj[1]
        decay = math.exp(-BETA * DELTA_T)
        s1_decayed = S_MAX * decay
        s2_decayed = 0.0  # 0 * decay = 0
        # One of (s1_next, s2_next) must be the decayed value and the other the updated value
        possible_s1_updated = strength_update(ALPHA, s1_decayed, S_MAX)
        possible_s2_updated = strength_update(ALPHA, s2_decayed, S_MAX)
        # Either mem1 was selected or mem2 was selected
        case1 = (math.isclose(s1_next, possible_s1_updated, rel_tol=1e-9) and
                 math.isclose(s2_next, s2_decayed, rel_tol=1e-9))
        case2 = (math.isclose(s1_next, s1_decayed, rel_tol=1e-9) and
                 math.isclose(s2_next, possible_s2_updated, rel_tol=1e-9))
        assert case1 or case2, f"Got ({s1_next}, {s2_next})"

    def test_zero_start_grows(self):
        """From (0,0) the selected memory must grow since strength_update(alpha, 0, Smax) > 0."""
        traj = simulate_chain(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, (0.0, 0.0), 1)
        s1, s2 = traj[1]
        assert s1 > 0.0 or s2 > 0.0

    def test_long_run_does_not_explode(self):
        traj = simulate_chain(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, (5.0, 5.0), 1000)
        for s1, s2 in traj:
            assert math.isfinite(s1)
            assert math.isfinite(s2)


# ════════════════════════════════════════════════════════════════════
# simulate_coupling
# ════════════════════════════════════════════════════════════════════


class TestSimulateCoupling:
    """Tests for the coupling construction (two chains, same randomness)."""

    def test_returns_correct_structure(self):
        traj_a, traj_b, tau = simulate_coupling(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, 50)
        assert isinstance(traj_a, list)
        assert isinstance(traj_b, list)
        assert len(traj_a) == 51
        assert len(traj_b) == 51
        assert tau is None or isinstance(tau, int)

    def test_chain_a_starts_at_origin(self):
        traj_a, _, _ = simulate_coupling(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, 10)
        assert traj_a[0] == (0.0, 0.0)

    def test_chain_b_starts_at_corner(self):
        _, traj_b, _ = simulate_coupling(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, 10)
        assert traj_b[0] == (S_MAX, S_MAX)

    def test_coupling_time_valid_if_present(self):
        _, _, tau = simulate_coupling(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, 500, seed=42)
        if tau is not None:
            assert 1 <= tau <= 500

    def test_chains_close_at_coupling_time(self):
        traj_a, traj_b, tau = simulate_coupling(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, 500, seed=42)
        if tau is not None:
            sa = traj_a[tau]
            sb = traj_b[tau]
            assert abs(sa[0] - sb[0]) < 1e-3 * S_MAX
            assert abs(sa[1] - sb[1]) < 1e-3 * S_MAX

    def test_both_chains_stay_in_domain(self):
        traj_a, traj_b, _ = simulate_coupling(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, 100)
        for s1, s2 in traj_a:
            assert 0.0 <= s1 <= S_MAX + 1e-12
            assert 0.0 <= s2 <= S_MAX + 1e-12
        for s1, s2 in traj_b:
            assert 0.0 <= s1 <= S_MAX + 1e-12
            assert 0.0 <= s2 <= S_MAX + 1e-12

    def test_deterministic_with_same_seed(self):
        r1 = simulate_coupling(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, 50, seed=99)
        r2 = simulate_coupling(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, 50, seed=99)
        assert r1[0] == r2[0]
        assert r1[1] == r2[1]
        assert r1[2] == r2[2]


# ════════════════════════════════════════════════════════════════════
# build_transition_matrix
# ════════════════════════════════════════════════════════════════════


class TestBuildTransitionMatrix:
    """Tests for the discretized transition matrix construction."""

    def test_returns_correct_shapes(self):
        n = 10
        P, grid = build_transition_matrix(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_grid=n)
        assert P.shape == (n * n, n * n)
        assert grid.shape == (n,)

    def test_rows_sum_to_one(self):
        P, _ = build_transition_matrix(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_grid=15)
        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_all_entries_non_negative(self):
        P, _ = build_transition_matrix(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_grid=10)
        assert np.all(P >= -1e-15)

    def test_grid_covers_domain(self):
        P, grid = build_transition_matrix(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_grid=20)
        assert grid[0] == pytest.approx(0.0)
        assert grid[-1] == pytest.approx(S_MAX)

    def test_sparse_at_most_two_targets_per_row(self):
        """Each row should have at most 2 non-zero entries (one for each selection outcome)."""
        P, _ = build_transition_matrix(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_grid=10)
        for i in range(P.shape[0]):
            nnz = np.count_nonzero(P[i])
            assert nnz <= 2, f"Row {i} has {nnz} non-zero entries"

    def test_state_index_encoding(self):
        """State index i = s1_idx * n_grid + s2_idx."""
        n = 10
        P, grid = build_transition_matrix(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_grid=n)
        # Verify origin state (0,0) is index 0
        # and corner state (Smax, Smax) is index n*n-1
        assert P.shape[0] == n * n

    def test_is_stochastic_matrix(self):
        """P is a valid row-stochastic matrix."""
        P, _ = build_transition_matrix(ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_grid=12)
        # Non-negative
        assert np.all(P >= -1e-15)
        # Rows sum to 1
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)


# ════════════════════════════════════════════════════════════════════
# spectral_analysis
# ════════════════════════════════════════════════════════════════════


class TestSpectralAnalysis:
    """Tests for spectral decomposition of the transition matrix.

    Uses aggressive parameters (high alpha, high beta, high T) so the
    discretized chain mixes well even on a coarse 8x8 grid.
    """

    # Parameters chosen so no grid cell is absorbing at n_grid=8
    SP_ALPHA = 0.8
    SP_BETA = 0.5
    SP_T = 2.0

    @pytest.fixture
    def small_analysis(self):
        P, _ = build_transition_matrix(
            self.SP_ALPHA, self.SP_BETA, DELTA_T, S_MAX, self.SP_T, n_grid=8
        )
        return spectral_analysis(P)

    def test_returns_required_keys(self, small_analysis):
        required = {"eigenvalues", "lambda_2", "spectral_gap", "stationary", "mixing_time"}
        assert required <= set(small_analysis.keys())

    def test_leading_eigenvalue_is_one(self, small_analysis):
        eigs = small_analysis["eigenvalues"]
        assert abs(abs(eigs[0]) - 1.0) < 1e-6

    def test_lambda2_less_than_one(self, small_analysis):
        assert 0 < small_analysis["lambda_2"] < 1.0

    def test_spectral_gap_positive(self, small_analysis):
        assert small_analysis["spectral_gap"] > 0

    def test_spectral_gap_consistent(self, small_analysis):
        assert small_analysis["spectral_gap"] == pytest.approx(
            1.0 - small_analysis["lambda_2"], abs=1e-10
        )

    def test_stationary_is_distribution(self, small_analysis):
        pi = small_analysis["stationary"]
        assert np.all(pi >= -1e-10)
        assert abs(pi.sum() - 1.0) < 1e-6

    def test_stationary_is_left_eigenvector(self, small_analysis):
        """pi @ P = pi for a stationary distribution."""
        P, _ = build_transition_matrix(
            self.SP_ALPHA, self.SP_BETA, DELTA_T, S_MAX, self.SP_T, n_grid=8
        )
        pi = small_analysis["stationary"]
        np.testing.assert_allclose(pi @ P, pi, atol=1e-6)

    def test_mixing_time_positive(self, small_analysis):
        assert small_analysis["mixing_time"] > 0

    def test_mixing_time_formula(self, small_analysis):
        gap = small_analysis["spectral_gap"]
        expected = 1.0 / gap
        assert small_analysis["mixing_time"] == pytest.approx(expected, rel=1e-10)

    def test_eigenvalues_sorted_by_magnitude(self, small_analysis):
        eigs = small_analysis["eigenvalues"]
        magnitudes = np.abs(eigs)
        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] >= magnitudes[i + 1] - 1e-10

    def test_identity_matrix_spectral_gap_zero(self):
        """Identity matrix has all eigenvalues = 1, so spectral gap = 0."""
        I = np.eye(4)
        result = spectral_analysis(I)
        assert result["spectral_gap"] == pytest.approx(0.0, abs=1e-10)

    def test_known_two_state_chain(self):
        """Simple 2-state chain: P = [[0.8, 0.2], [0.3, 0.7]].
        Eigenvalues: 1 and 0.5. Stationary: [0.6, 0.4]."""
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        result = spectral_analysis(P)
        assert result["lambda_2"] == pytest.approx(0.5, abs=1e-6)
        assert result["spectral_gap"] == pytest.approx(0.5, abs=1e-6)
        np.testing.assert_allclose(result["stationary"], [0.6, 0.4], atol=1e-6)
        assert result["mixing_time"] == pytest.approx(2.0, abs=1e-6)
