"""Coupling tests for the Hermes memory Markov chain.

Coupling is a probabilistic technique: run two copies of the chain with
SHARED randomness from opposite corners. If they merge (couple), this
proves the chain forgets its initial state and mixes. The coupling time
gives an upper bound on the mixing time.

These tests verify that coupling occurs reliably and quickly under
parameters satisfying the contraction condition K < 1.

Primary parameters:
  alpha=0.1, beta=0.1, delta_t=1.0, S_max=10.0, T_temp=5.0
  K = 0.9548 < 1
"""

import math

import numpy as np
import pytest

from hermes_memory.markov_chain import simulate_coupling


# ── Parameters satisfying contraction condition K < 1 ─────────────
ALPHA = 0.1
BETA = 0.1
DELTA_T = 1.0
S_MAX = 10.0
T_TEMP = 5.0

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


class TestCouplingOccurs:
    """Verify that coupling happens: chains from opposite corners merge."""

    def test_coupling_occurs(self):
        """Run simulate_coupling for 5000 steps. Assert coupling_time is
        not None. The chains MUST merge.

        Under the contraction condition, the expected distance between chains
        shrinks geometrically by factor K < 1 per step. With K=0.9548,
        after 5000 steps the expected distance ratio is K^5000 ~ 0, so
        coupling is essentially certain.
        """
        _, _, coupling_time = simulate_coupling(
            ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_steps=5000, seed=42,
        )
        assert coupling_time is not None, (
            "Coupling did not occur within 5000 steps"
        )

    def test_coupling_time_finite(self):
        """Run 20 independent coupling experiments with different seeds.
        Assert ALL coupling times are finite and < 3000 steps.

        The chain mixes in finite time. Under contraction, the coupling
        time has an exponential tail bound, so P(tau > 3000) is negligible.
        """
        coupling_times = []
        for seed in range(20):
            _, _, tau = simulate_coupling(
                ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_steps=5000, seed=seed,
            )
            assert tau is not None, f"Coupling failed with seed={seed}"
            assert tau < 3000, (
                f"Coupling too slow with seed={seed}: tau={tau} >= 3000"
            )
            coupling_times.append(tau)

        # All 20 must have coupled
        assert len(coupling_times) == 20

    @pytest.mark.slow
    def test_coupling_time_distribution(self):
        """Run 100 coupling experiments. Compute mean and std of coupling times.
        Assert mean < 500 steps (fast mixing). Assert std < mean (not too variable).

        The coupling time distribution should be concentrated: most experiments
        couple in a similar number of steps, with light tails.
        """
        coupling_times = []
        for seed in range(100):
            _, _, tau = simulate_coupling(
                ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_steps=5000, seed=seed,
            )
            assert tau is not None, f"Coupling failed with seed={seed}"
            coupling_times.append(tau)

        ct = np.array(coupling_times, dtype=float)
        mean_tau = ct.mean()
        std_tau = ct.std()

        # Print distribution stats for diagnostics
        print(f"\nCoupling time distribution (n=100):")
        print(f"  Mean:   {mean_tau:.1f}")
        print(f"  Std:    {std_tau:.1f}")
        print(f"  Min:    {ct.min():.0f}")
        print(f"  Max:    {ct.max():.0f}")
        print(f"  Median: {np.median(ct):.0f}")

        assert mean_tau < 500, f"Mean coupling time {mean_tau:.1f} >= 500"
        assert std_tau < mean_tau, (
            f"Coupling time too variable: std={std_tau:.1f} >= mean={mean_tau:.1f}"
        )


class TestCouplingStability:
    """Verify that after coupling, chains remain close."""

    @pytest.mark.slow
    def test_chains_agree_after_coupling(self):
        """After coupling, both chains should stay close forever.

        Run coupling for 5000 steps. After coupling_time, verify max
        distance between chains stays < 0.1*Smax for all remaining steps.

        Once the chains have coupled (entered the same state region with
        shared randomness), the shared randomness keeps them synchronized.
        Residual divergence is bounded by the coupling threshold (1e-3*Smax).
        """
        traj_a, traj_b, coupling_time = simulate_coupling(
            ALPHA, BETA, DELTA_T, S_MAX, T_TEMP, n_steps=5000, seed=42,
        )
        assert coupling_time is not None, "Coupling must occur for this test"

        threshold = 0.1 * S_MAX  # 1.0
        max_post_coupling_dist = 0.0

        for t in range(coupling_time, len(traj_a)):
            a = traj_a[t]
            b = traj_b[t]
            dist = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
            max_post_coupling_dist = max(max_post_coupling_dist, dist)

        assert max_post_coupling_dist < threshold, (
            f"Chains diverged after coupling: max dist={max_post_coupling_dist:.4f} "
            f">= threshold={threshold:.4f}"
        )


class TestCouplingAcrossParameters:
    """Verify coupling is not parameter-specific."""

    @pytest.mark.slow
    def test_coupling_across_parameters(self):
        """Try 3 different parameter sets that all satisfy the contraction
        condition. All must couple. Tests that the result is not parameter-specific.

        Each parameter set represents a different regime:
        - primary: moderate decay, wide domain
        - fast-decay: strong decay, narrow domain
        - long-timestep: slower updates, larger time intervals
        """
        for alpha, beta, dt, s_max, t_temp, name in PARAM_SETS:
            coupling_times = []
            for seed in range(10):
                _, _, tau = simulate_coupling(
                    alpha, beta, dt, s_max, t_temp, n_steps=5000, seed=seed,
                )
                assert tau is not None, (
                    f"Coupling failed for {name} with seed={seed}"
                )
                coupling_times.append(tau)

            mean_tau = np.mean(coupling_times)
            print(f"\n{name}: mean coupling time = {mean_tau:.1f} "
                  f"(range: {min(coupling_times)}-{max(coupling_times)})")

            # All parameter sets should couple reasonably fast
            assert mean_tau < 500, (
                f"{name}: mean coupling time {mean_tau:.1f} >= 500"
            )
