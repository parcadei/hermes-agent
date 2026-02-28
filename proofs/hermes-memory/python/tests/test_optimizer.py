"""Comprehensive tests for hermes_memory.optimizer -- CMA-ES parameter optimization.

Tests define expected behavioral contracts from the optimizer spec.
These tests are written BEFORE the implementation (TDD). They should all fail
with ImportError until optimizer.py is created.

48 tests covering:
  - decode: vector-to-ParameterSet mapping (12 tests)
  - objective: fitness evaluation (10 tests)
  - build_benchmark_scenarios: scenario generation (8 tests)
  - run_optimization: CMA-ES loop (6 tests + 2 slow)
  - validate: post-optimization checks (8 tests)
  - OptimizationResult: dataclass invariants (4 tests)
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hermes_memory.optimizer import (
    PARAM_ORDER,
    BOUNDS,
    PINNED,
    STATIC_WEIGHT,
    STABILITY_WEIGHT,
    MARGIN_WEIGHT,
    INFEASIBLE_PENALTY,
    STABILITY_SIM_STEPS,
    decode,
    objective,
    build_benchmark_scenarios,
    run_optimization,
    validate,
    OptimizationResult,
    ValidationResult,
)

from hermes_memory.engine import (
    ParameterSet,
    MemoryState,
    rank_memories,
    simulate,
)

from hermes_memory.sensitivity import Scenario


# ============================================================
# Helpers
# ============================================================


def _center_vector() -> np.ndarray:
    """Return the center of BOUNDS for all 6 parameters."""
    return np.array([
        (lo + hi) / 2.0 for lo, hi in [
            BOUNDS[p] for p in PARAM_ORDER
        ]
    ])


def _lower_vector() -> np.ndarray:
    """Return the lower bounds for all 6 parameters."""
    return np.array([
        BOUNDS[p][0] for p in PARAM_ORDER
    ])


def _upper_vector() -> np.ndarray:
    """Return the upper bounds for all 6 parameters."""
    return np.array([
        BOUNDS[p][1] for p in PARAM_ORDER
    ])


def _make_vector(w1: float, w2: float, w4: float,
                 alpha: float, beta: float, temperature: float) -> np.ndarray:
    """Build a raw 6-element vector in PARAM_ORDER."""
    return np.array([w1, w2, w4, alpha, beta, temperature])


# ============================================================
# 1. TestDecode (~12 tests)
# ============================================================


class TestDecode:
    """Tests for decode(x) -> ParameterSet | None."""

    def test_center_of_bounds_produces_valid_params(self) -> None:
        """Center of all 6 bounds decodes to a valid ParameterSet.

        Center: w1=0.325, w2=0.22, w4=0.275, alpha=0.255, beta=0.505, temp=5.05
        w3 = 1.0 - 0.325 - 0.22 - 0.275 = 0.18
        All positive, w2 < 0.4, should satisfy contraction with high temp.
        """
        x = _center_vector()
        params = decode(x)
        assert params is not None
        assert isinstance(params, ParameterSet)
        # Finding 2 fix: verify computed values, not just non-None
        assert params.w1 == pytest.approx(0.325, abs=1e-10)
        assert params.w2 == pytest.approx(0.22, abs=1e-10)
        assert params.w4 == pytest.approx(0.275, abs=1e-10)
        assert params.w3 == pytest.approx(0.18, abs=1e-10)
        assert params.alpha == pytest.approx(0.255, abs=1e-10)
        assert params.beta == pytest.approx(0.505, abs=1e-10)
        assert params.temperature == pytest.approx(5.05, abs=1e-10)
        assert params.satisfies_contraction()

    def test_all_lower_bounds(self) -> None:
        """All params at lower bound: w1=0.05, w2=0.05, w4=0.05.

        w3 = 1.0 - 0.05 - 0.05 - 0.05 = 0.85
        alpha=0.01, beta=0.01, temperature=0.10
        K = exp(-0.01*1.0) + (0.25/0.10)*0.01*10 = 0.9900 + 0.25 = 1.24 > 1
        Contraction violated => None.
        """
        x = _lower_vector()
        params = decode(x)
        # Low temperature (0.10) with s_max=10 makes L*alpha*s_max = 2.5*0.01*10 = 0.25
        # exp(-0.01) ~ 0.9900, total K ~ 1.24 > 1 => contraction fails
        assert params is None

    def test_all_upper_bounds(self) -> None:
        """All params at upper bound: w1=0.60, w2=0.39, w4=0.50.

        w3 = 1.0 - 0.60 - 0.39 - 0.50 = -0.49 < 0 => None.
        """
        x = _upper_vector()
        params = decode(x)
        assert params is None

    def test_w3_negative_returns_none(self) -> None:
        """When w1 + w2 + w4 > 1.0, w3 would be negative => None.

        w1=0.50, w2=0.30, w4=0.30 => w3 = 1.0 - 1.10 = -0.10 < 0.
        """
        x = _make_vector(w1=0.50, w2=0.30, w4=0.30,
                         alpha=0.10, beta=0.50, temperature=5.0)
        params = decode(x)
        assert params is None

    def test_w3_zero_is_valid(self) -> None:
        """When w1 + w2 + w4 = 1.0, w3 = 0.0 which is valid.

        w1=0.35, w2=0.35, w4=0.30 => w3 = 0.0.
        alpha=0.10, beta=0.50, temperature=5.0
        K = exp(-0.50) + (0.25/5.0)*0.10*10 = 0.6065 + 0.05 = 0.6565 < 1
        """
        x = _make_vector(w1=0.35, w2=0.35, w4=0.30,
                         alpha=0.10, beta=0.50, temperature=5.0)
        params = decode(x)
        assert params is not None
        assert params.w3 == pytest.approx(0.0, abs=1e-10)
        assert params.w1 + params.w2 + params.w3 + params.w4 == pytest.approx(1.0, abs=1e-10)

    def test_contraction_violation_returns_none(self) -> None:
        """alpha=0.50, beta=0.01, temperature=0.10 violates contraction.

        L = 0.25/0.10 = 2.5
        K = exp(-0.01*1.0) + 2.5 * 0.50 * 10 = 0.9900 + 12.5 >> 1
        """
        x = _make_vector(w1=0.20, w2=0.20, w4=0.20,
                         alpha=0.50, beta=0.01, temperature=0.10)
        params = decode(x)
        assert params is None

    def test_decode_preserves_pinned_defaults(self) -> None:
        """Decoded params carry correct PINNED values for insensitive parameters."""
        x = _center_vector()
        params = decode(x)
        assert params is not None
        assert params.delta_t == 1.0
        assert params.s_max == 10.0
        assert params.s0 == 1.0
        assert params.novelty_start == 0.3
        assert params.novelty_decay == 0.1
        assert params.survival_threshold == 0.05
        assert params.feedback_sensitivity == 0.1

    def test_decode_clips_to_bounds(self) -> None:
        """Out-of-bounds values are clipped before decoding.

        Pass w1=2.0 (above 0.60 upper bound) => clips to 0.60.
        Pass alpha=-1.0 (below 0.01 lower bound) => clips to 0.01.
        The result should either be a valid ParameterSet or None
        (if clipped values still violate constraints), but never raise.
        """
        x = np.array([2.0, 0.10, 0.10, -1.0, 0.50, 5.0])
        result = decode(x)
        # w1 clips to 0.60, w2=0.10, w4=0.10 => w3 = 1.0 - 0.60 - 0.10 - 0.10 = 0.20
        # alpha clips to 0.01, beta=0.50, temperature=5.0
        # K = exp(-0.50) + (0.25/5.0)*0.01*10 = 0.6065 + 0.005 = 0.6115 < 1
        # Should be valid
        assert result is not None
        assert result.w1 == pytest.approx(0.60, abs=1e-10)
        assert result.alpha == pytest.approx(0.01, abs=1e-10)

    def test_decode_weights_sum_to_one(self) -> None:
        """Any valid decode has w1 + w2 + w3 + w4 == 1.0 within tolerance."""
        x = _center_vector()
        params = decode(x)
        assert params is not None
        total = params.w1 + params.w2 + params.w3 + params.w4
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_decode_w2_under_cap(self) -> None:
        """Any valid decode has w2 < 0.4 (recency cap from ParameterSet).

        Even if raw w2 is at upper bound (0.39), it must be < 0.4.
        """
        x = _make_vector(w1=0.20, w2=0.39, w4=0.20,
                         alpha=0.10, beta=0.50, temperature=5.0)
        params = decode(x)
        assert params is not None
        assert params.w2 < 0.4

    def test_decode_deterministic(self) -> None:
        """Same input vector always produces the same output."""
        x = _center_vector()
        result1 = decode(x)
        result2 = decode(x.copy())
        assert result1 is not None
        assert result2 is not None
        assert result1 == result2

    def test_param_order_invariant(self) -> None:
        """PARAM_ORDER is exactly the expected 6 names in order (Finding 20)."""
        assert PARAM_ORDER == ["w1", "w2", "w4", "alpha", "beta", "temperature"]

    def test_decode_shape_mismatch(self) -> None:
        """Wrong-shape input raises an appropriate error."""
        x_short = np.array([0.3, 0.2, 0.2])
        with pytest.raises((ValueError, IndexError)):
            decode(x_short)

        x_long = np.zeros(10)
        with pytest.raises((ValueError, IndexError)):
            decode(x_long)


# ============================================================
# 2. TestObjective (~10 tests)
# ============================================================


class TestObjective:
    """Tests for objective(x, scenarios) -> float."""

    def test_feasible_returns_negative(self) -> None:
        """Valid vector produces a return value in [-1, 0].

        The composite score is negated: -(weighted_sum) where weighted_sum in [0, 1].
        """
        x = _center_vector()
        result = objective(x)
        assert -1.0 <= result <= 0.0

    def test_infeasible_returns_penalty(self) -> None:
        """Infeasible vector (w3 < 0) returns INFEASIBLE_PENALTY = 1000.0."""
        x = _make_vector(w1=0.50, w2=0.30, w4=0.30,
                         alpha=0.10, beta=0.50, temperature=5.0)
        result = objective(x)
        assert result == 1000.0

    def test_deterministic(self) -> None:
        """Same vector evaluated twice produces the same result.

        Stability scoring uses simulate with fixed rng seed internally.
        """
        x = _center_vector()
        r1 = objective(x)
        r2 = objective(x.copy())
        assert r1 == r2

    def test_perfect_score_range(self) -> None:
        """Best possible objective is close to -1.0.

        If static_accuracy=1, stability_accuracy=1, margin_bonus=1:
        objective = -(0.60*1 + 0.25*1 + 0.15*1) = -1.0
        In practice, perfect margin_bonus requires margin >= 0.3.
        """
        x = _center_vector()
        result = objective(x)
        # Even the best center-of-bounds might not be perfect, but must be negative
        assert result < 0.0

    def test_better_params_get_lower_objective(self) -> None:
        """Compare two known vectors; the one with better characteristics
        should produce a lower (more negative) objective.

        Vector A: balanced weights, moderate dynamics (center-ish)
        Vector B: extreme weights that produce low accuracy
        """
        # A: center of bounds (should be reasonable)
        x_good = _center_vector()

        # B: w1 very low (relevance barely matters), should hurt accuracy
        x_bad = _make_vector(w1=0.05, w2=0.05, w4=0.05,
                             alpha=0.10, beta=0.50, temperature=5.0)
        # w3 = 1.0 - 0.05 - 0.05 - 0.05 = 0.85 (importance dominates)

        result_good = objective(x_good)
        result_bad = objective(x_bad)

        # If bad is infeasible, penalty is 1000 which is worse (higher)
        # If both feasible, good should have lower (more negative) objective
        assert result_good <= result_bad

    def test_contraction_margin_contributes(self) -> None:
        """Two params with same static accuracy but different contraction margins
        should produce different objectives.

        Higher margin => higher margin_bonus => more negative objective (better).
        """
        # High temperature => large margin (L = 0.25/T is small)
        x_high_margin = _make_vector(w1=0.30, w2=0.20, w4=0.20,
                                     alpha=0.10, beta=0.50, temperature=8.0)
        # Low temperature => small margin (L = 0.25/T is larger)
        x_low_margin = _make_vector(w1=0.30, w2=0.20, w4=0.20,
                                    alpha=0.10, beta=0.50, temperature=1.0)

        # Verify both are feasible
        p_high = decode(x_high_margin)
        p_low = decode(x_low_margin)
        assert p_high is not None, "High-margin vector should be feasible"
        assert p_low is not None, "Low-margin vector should be feasible"

        # Both have same weights so same static accuracy; margin should differ
        margin_high = p_high.contraction_margin()
        margin_low = p_low.contraction_margin()
        assert margin_high > margin_low

        # Higher margin => lower (better) objective
        obj_high = objective(x_high_margin)
        obj_low = objective(x_low_margin)
        assert obj_high <= obj_low

    def test_custom_scenarios_accepted(self) -> None:
        """Passing explicit scenario list overrides internal scenario building."""
        x = _center_vector()
        params = decode(x)
        assert params is not None

        # Build a small custom scenario list
        scenario = Scenario(
            name="custom-AR",
            memories=[
                MemoryState(relevance=0.9, last_access_time=5.0,
                            importance=0.5, access_count=3,
                            strength=5.0, creation_time=100.0),
                MemoryState(relevance=0.3, last_access_time=10.0,
                            importance=0.5, access_count=3,
                            strength=5.0, creation_time=100.0),
            ],
            current_time=0.0,
            expected_winner=0,
            competency="accurate_retrieval",
        )
        result = objective(x, scenarios=[scenario])
        # Should be a valid negative number (feasible with 1 scenario)
        assert -1.0 <= result <= 0.0

    def test_empty_scenarios_returns_penalty(self) -> None:
        """Passing an empty scenario list is an edge case.

        With 0 static scenarios and 0 stability scenarios, the accuracy
        terms are 0/max(0,1) = 0. The objective should still be valid
        (not crash), returning -(0*0.60 + 0*0.25 + margin_bonus*0.15).
        """
        x = _center_vector()
        result = objective(x, scenarios=[])
        # With 0 scenarios: static=0, stability=0
        # Result = -(0*0.60 + 0*0.25 + margin_bonus*0.15)
        # Finding 15: verify specific behavior (not just isinstance)
        assert isinstance(result, float)
        assert -1.0 <= result <= 0.0, \
            "Empty scenarios with feasible params should return valid objective"

    def test_stability_scoring(self) -> None:
        """Stability check detects monopolization.

        Create a stability scenario with 2 memories where one has
        much higher relevance. Objective should reflect stability failure
        (lower stability_accuracy).
        """
        x = _center_vector()
        params = decode(x)
        assert params is not None

        # Stability scenario: one memory dominates
        stab_scenario = Scenario(
            name="STAB-monopoly",
            memories=[
                MemoryState(relevance=0.99, last_access_time=1.0,
                            importance=0.9, access_count=50,
                            strength=9.0, creation_time=200.0),
                MemoryState(relevance=0.01, last_access_time=50.0,
                            importance=0.1, access_count=1,
                            strength=0.5, creation_time=200.0),
            ],
            current_time=0.0,
            expected_winner=-1,
            competency="stability",
        )
        result = objective(x, scenarios=[stab_scenario])
        # With only a stability scenario (no static ones), static_accuracy = 0.
        # The dominant memory (relevance=0.99 vs 0.01) likely monopolizes
        # (wins >80% of 100 steps) => stability_accuracy = 0.
        # Result = -(0*0.60 + 0*0.25 + margin_bonus*0.15) = -(margin_bonus*0.15)
        assert isinstance(result, float)
        assert result <= 0.0  # Still feasible
        # With center-of-bounds params, margin ~ 0.27, so margin_bonus ~ 0.9
        # Expected result ~ -(0 + 0 + 0.15 * 0.9) ~ -0.135
        # The monopoly scenario should NOT give stability credit
        # Compare: a balanced scenario where stability passes:
        balanced_stab = Scenario(
            name="STAB-balanced",
            memories=[
                MemoryState(relevance=0.50, last_access_time=10.0,
                            importance=0.5, access_count=5,
                            strength=5.0, creation_time=200.0),
                MemoryState(relevance=0.50, last_access_time=10.0,
                            importance=0.5, access_count=5,
                            strength=5.0, creation_time=200.0),
            ],
            current_time=0.0,
            expected_winner=-1,
            competency="stability",
        )
        result_balanced = objective(x, scenarios=[balanced_stab])
        # Balanced scenario should get stability credit => lower (better) objective
        assert result_balanced < result, \
            f"Balanced stab ({result_balanced}) should beat monopoly stab ({result})"

    @given(
        w1=st.floats(min_value=0.05, max_value=0.60, allow_nan=False),
        w2=st.floats(min_value=0.05, max_value=0.39, allow_nan=False),
        w4=st.floats(min_value=0.05, max_value=0.50, allow_nan=False),
        alpha=st.floats(min_value=0.01, max_value=0.50, allow_nan=False),
        beta=st.floats(min_value=0.01, max_value=1.00, allow_nan=False),
        temperature=st.floats(min_value=0.10, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_objective_range_property(self, w1: float, w2: float, w4: float,
                                      alpha: float, beta: float,
                                      temperature: float) -> None:
        """Property: objective returns value in [-1, 0] or exactly 1000.0."""
        x = np.array([w1, w2, w4, alpha, beta, temperature])
        result = objective(x)
        assert result == 1000.0 or (-1.0 <= result <= 0.0), \
            f"objective returned {result}, expected [-1, 0] or 1000.0"


# ============================================================
# 3. TestBuildBenchmarkScenarios (~8 tests)
# ============================================================


class TestBuildBenchmarkScenarios:
    """Tests for build_benchmark_scenarios(params) -> list[Scenario]."""

    @pytest.fixture
    def ref_params(self) -> ParameterSet:
        """Reference ParameterSet at center of bounds for scenario building."""
        x = _center_vector()
        params = decode(x)
        assert params is not None
        return params

    def test_count_at_least_120(self, ref_params: ParameterSet) -> None:
        """Total scenario count is at least 120 (30 per competency target)."""
        scenarios = build_benchmark_scenarios(ref_params)
        assert len(scenarios) >= 120

    def test_competency_distribution(self, ref_params: ParameterSet) -> None:
        """Each of the 4 competencies has at least 25 scenarios."""
        scenarios = build_benchmark_scenarios(ref_params)
        competency_counts: dict[str, int] = {}
        for s in scenarios:
            competency_counts[s.competency] = competency_counts.get(s.competency, 0) + 1

        expected_competencies = {
            "accurate_retrieval",
            "test_time_learning",
            "selective_forgetting",
            "stability",
        }
        assert set(competency_counts.keys()) == expected_competencies
        for comp, count in competency_counts.items():
            assert count >= 25, f"{comp} has only {count} scenarios (need >= 25)"

    def test_all_expected_winners_valid(self, ref_params: ParameterSet) -> None:
        """Every scenario's expected_winner is a valid index or -1 (stability sentinel)."""
        scenarios = build_benchmark_scenarios(ref_params)
        for s in scenarios:
            if s.expected_winner == -1:
                assert s.competency == "stability", \
                    f"Scenario {s.name!r}: expected_winner=-1 only valid for stability"
            else:
                assert 0 <= s.expected_winner < len(s.memories), \
                    f"Scenario {s.name!r}: expected_winner={s.expected_winner} " \
                    f"out of range for {len(s.memories)} memories"

    def test_stability_scenarios_have_sentinel(self, ref_params: ParameterSet) -> None:
        """All stability scenarios use expected_winner=-1."""
        scenarios = build_benchmark_scenarios(ref_params)
        stability_scenarios = [s for s in scenarios if s.competency == "stability"]
        assert len(stability_scenarios) > 0
        for s in stability_scenarios:
            assert s.expected_winner == -1, \
                f"Stability scenario {s.name!r} has expected_winner={s.expected_winner}, want -1"

    def test_non_stability_verified(self, ref_params: ParameterSet) -> None:
        """All non-stability scenarios have expected_winner matching rank_memories result.

        The scenario construction verifies that the expected winner actually
        ranks first under the reference params.
        """
        scenarios = build_benchmark_scenarios(ref_params)
        non_stability = [s for s in scenarios if s.competency != "stability"]
        for s in non_stability:
            ranked = rank_memories(ref_params, s.memories, s.current_time)
            actual_winner = ranked[0][0]  # index of highest-scoring memory
            assert actual_winner == s.expected_winner, \
                f"Scenario {s.name!r}: rank says {actual_winner}, " \
                f"expected {s.expected_winner}"

    def test_held_out_flag(self, ref_params: ParameterSet) -> None:
        """held_out=True returns different scenarios than held_out=False.

        Training and validation use different numerical values.
        """
        train = build_benchmark_scenarios(ref_params, held_out=False)
        held = build_benchmark_scenarios(ref_params, held_out=True)

        train_names = {s.name for s in train}
        held_names = {s.name for s in held}
        # They should not be identical sets
        assert train_names != held_names, \
            "Training and held-out scenarios should differ"
        # Both should have substantial count
        assert len(held) >= 120

    def test_held_out_same_competencies(self, ref_params: ParameterSet) -> None:
        """Held-out scenarios cover all 4 competencies."""
        held = build_benchmark_scenarios(ref_params, held_out=True)
        competencies = {s.competency for s in held}
        assert competencies == {
            "accurate_retrieval",
            "test_time_learning",
            "selective_forgetting",
            "stability",
        }

    def test_scenarios_use_pinned_smax(self, ref_params: ParameterSet) -> None:
        """Memory strengths in scenarios reference PINNED s_max=10.0.

        No memory should have strength > s_max.
        """
        scenarios = build_benchmark_scenarios(ref_params)
        s_max = PINNED["s_max"]
        for s in scenarios:
            for i, m in enumerate(s.memories):
                assert m.strength <= s_max + 1e-10, \
                    f"Scenario {s.name!r}, memory {i}: " \
                    f"strength={m.strength} > s_max={s_max}"


# ============================================================
# 4. TestRunOptimization (~6 tests + 2 slow)
# ============================================================


class TestRunOptimization:
    """Tests for run_optimization(n_generations, seed) -> OptimizationResult."""

    def test_returns_optimization_result(self) -> None:
        """Basic type check: run_optimization returns an OptimizationResult."""
        result = run_optimization(n_generations=5, seed=42)
        assert isinstance(result, OptimizationResult)

    def test_result_has_valid_params(self) -> None:
        """best_params in the result satisfies contraction condition."""
        result = run_optimization(n_generations=5, seed=42)
        assert result.best_params.satisfies_contraction()

    def test_result_improves_over_baseline(self) -> None:
        """best_score should be better than evaluating the center-of-bounds vector.

        Even a short optimization (5 generations) with CMA-ES population
        should find something at least as good as the starting point.
        """
        result = run_optimization(n_generations=5, seed=42)
        # Evaluate center of bounds for comparison
        x_center = _center_vector()
        center_obj = objective(x_center)
        center_score = -center_obj if center_obj < INFEASIBLE_PENALTY else 0.0
        # best_score is the positive version (higher is better)
        assert result.best_score >= center_score

    def test_history_monotonic(self) -> None:
        """History (best score per generation) is monotonically non-decreasing."""
        result = run_optimization(n_generations=10, seed=42)
        for i in range(1, len(result.history)):
            assert result.history[i] >= result.history[i - 1], \
                f"History not monotonic at gen {i}: " \
                f"{result.history[i]} < {result.history[i - 1]}"

    def test_reproducible(self) -> None:
        """Same seed produces the same result."""
        r1 = run_optimization(n_generations=5, seed=123)
        r2 = run_optimization(n_generations=5, seed=123)
        assert r1.best_score == r2.best_score
        np.testing.assert_array_equal(r1.best_x, r2.best_x)
        # Finding 10: also check full history for exact reproducibility
        assert r1.history == r2.history, "Full generation history must be identical"

    def test_small_run_completes(self) -> None:
        """n_generations=5 completes without error and has expected structure."""
        result = run_optimization(n_generations=5, seed=42)
        assert result.generations >= 5
        assert len(result.history) >= 5
        assert result.best_score >= 0.0
        assert result.best_x.shape == (6,)
        assert isinstance(result.per_competency, dict)

    @pytest.mark.slow
    def test_full_optimization_300_gen(self) -> None:
        """Full 300-generation optimization converges to a good score.

        The composite score (positive version) should exceed 0.70, meaning
        the weighted sum of static accuracy, stability, and margin is > 70%.
        """
        result = run_optimization(n_generations=300, seed=42)
        assert result.best_score > 0.70, \
            f"300-gen optimization only reached {result.best_score:.4f}"
        # History should show convergence
        assert result.history[-1] >= result.history[0]
        assert result.best_params.satisfies_contraction()

    @pytest.mark.slow
    def test_optimization_beats_random(self) -> None:
        """CMA-ES result beats 100 random valid vectors.

        Generate 100 random in-bounds vectors, keep the best feasible one,
        and verify CMA-ES outperforms it.
        """
        result = run_optimization(n_generations=300, seed=42)
        rng = np.random.default_rng(seed=99)

        best_random_score = 0.0
        for _ in range(100):
            x = np.array([
                rng.uniform(BOUNDS[p][0], BOUNDS[p][1])
                for p in PARAM_ORDER
            ])
            obj = objective(x)
            if obj < INFEASIBLE_PENALTY:
                score = -obj
                best_random_score = max(best_random_score, score)

        assert result.best_score >= best_random_score, \
            f"CMA-ES ({result.best_score:.4f}) lost to random ({best_random_score:.4f})"


# ============================================================
# 5. TestValidate (~8 tests)
# ============================================================


class TestValidate:
    """Tests for validate(params) -> ValidationResult."""

    @pytest.fixture
    def optimized_params(self) -> ParameterSet:
        """Run a short optimization and return best_params for validation."""
        result = run_optimization(n_generations=10, seed=42)
        return result.best_params

    def test_valid_params_all_pass(self) -> None:
        """Params from a decent optimization should pass all checks.

        A 50-gen run should be good enough to pass basic validation.
        """
        result = run_optimization(n_generations=50, seed=42)
        vr = validate(result.best_params)
        assert isinstance(vr, ValidationResult)
        # At minimum, contraction and w2 checks should pass
        assert vr.contraction_margin > 0.01

    def test_contraction_margin_check(self, optimized_params: ParameterSet) -> None:
        """Contraction margin is correctly computed and reported."""
        vr = validate(optimized_params)
        expected_margin = optimized_params.contraction_margin()
        assert vr.contraction_margin == pytest.approx(expected_margin, rel=1e-10)
        # If margin < 0.01, it should appear in failures
        if vr.contraction_margin < 0.01:
            assert any("contraction" in f.lower() for f in vr.failures)

    def test_held_out_accuracy(self, optimized_params: ParameterSet) -> None:
        """Held-out accuracy is reported and >= 0.0 (at minimum a valid float)."""
        vr = validate(optimized_params)
        assert 0.0 <= vr.held_out_accuracy <= 1.0

    def test_steady_state_below_smax(self, optimized_params: ParameterSet) -> None:
        """Analytical steady-state strength S* < s_max.

        S* = alpha * s_max / (1 - (1-alpha) * exp(-beta * delta_t))
        For valid contraction, this should be < s_max.
        """
        vr = validate(optimized_params)
        s_max = optimized_params.s_max
        assert vr.steady_state_strength < s_max, \
            f"S* = {vr.steady_state_strength} >= s_max = {s_max}"
        assert vr.steady_state_ratio < 1.0

    def test_exploration_window_positive(self, optimized_params: ParameterSet) -> None:
        """Exploration window W > 0.

        W = ln(novelty_start / survival_threshold) / novelty_decay
        With PINNED defaults: ln(0.3 / 0.05) / 0.1 = ln(6) / 0.1 ~ 17.9
        """
        vr = validate(optimized_params)
        assert vr.exploration_window > 0.0

    def test_w2_cap_enforced(self) -> None:
        """If w2 >= 0.4, validation should report a failure.

        We cannot construct such a ParameterSet directly (it raises ValueError),
        but validation should also check it independently. Testing that the
        validate function includes this check in its logic.
        """
        # Build valid params with w2 close to cap
        x = _make_vector(w1=0.20, w2=0.39, w4=0.20,
                         alpha=0.10, beta=0.50, temperature=5.0)
        params = decode(x)
        assert params is not None
        vr = validate(params)
        # w2=0.39 is valid (< 0.4), should NOT appear in failures
        assert not any("w2" in f.lower() and "cap" in f.lower() for f in vr.failures)

    def test_failures_list_populated(self) -> None:
        """Params with marginal contraction should have non-empty failures.

        Construct params barely satisfying contraction (margin ~ 0.005).
        The strict check (margin > 0.01) should fail.
        """
        # alpha=0.25, beta=0.05, temperature=2.0, s_max=10.0, delta_t=1.0
        # L = 0.25/2.0 = 0.125
        # K = exp(-0.05) + 0.125 * 0.25 * 10 = 0.9512 + 0.3125 = 1.2637 > 1
        # That's infeasible. Try temperature=3.0:
        # L = 0.25/3.0 = 0.0833
        # K = exp(-0.05) + 0.0833 * 0.25 * 10 = 0.9512 + 0.2083 = 1.1595 > 1
        # Still infeasible. Try beta=0.50:
        # K = exp(-0.50) + 0.0833 * 0.25 * 10 = 0.6065 + 0.2083 = 0.8148 < 1
        # margin = 1 - 0.8148 = 0.1852 -- too large.
        # We need a scenario where margin is between 0 and 0.01.
        # alpha=0.48, beta=0.50, temperature=1.3
        # L = 0.25/1.3 = 0.1923
        # K = exp(-0.50) + 0.1923 * 0.48 * 10 = 0.6065 + 0.9231 = 1.5296 > 1 (infeasible)
        # alpha=0.05, beta=0.01, temperature=0.40
        # L = 0.25/0.40 = 0.625
        # K = exp(-0.01) + 0.625 * 0.05 * 10 = 0.9900 + 0.3125 = 1.3025 > 1 (infeasible)
        # alpha=0.05, beta=1.0, temperature=0.40
        # K = exp(-1.0) + 0.625 * 0.05 * 10 = 0.3679 + 0.3125 = 0.6804 (margin=0.32)
        # alpha=0.05, beta=0.05, temperature=5.0
        # L = 0.25/5.0 = 0.05
        # K = exp(-0.05) + 0.05 * 0.05 * 10 = 0.9512 + 0.025 = 0.9762, margin=0.0238
        # alpha=0.05, beta=0.02, temperature=5.0
        # K = exp(-0.02) + 0.05 * 0.05 * 10 = 0.9802 + 0.025 = 1.0052 > 1 (infeasible)
        # alpha=0.05, beta=0.03, temperature=5.0
        # K = exp(-0.03) + 0.05 * 0.05 * 10 = 0.9704 + 0.025 = 0.9954, margin=0.0046
        # That's < 0.01! But feasible. Let's use this.
        x = _make_vector(w1=0.30, w2=0.20, w4=0.20,
                         alpha=0.05, beta=0.03, temperature=5.0)
        params = decode(x)
        assert params is not None
        margin = params.contraction_margin()
        assert margin > 0.0, f"Should be feasible, margin={margin}"
        assert 0.004 < margin < 0.005, \
            f"Expected margin ~ 0.0046 (Finding 6: tighten bounds), got {margin}"

        vr = validate(params)
        assert len(vr.failures) > 0, \
            "Marginal contraction (margin < 0.01) should produce validation failures"
        assert any("contraction" in f.lower() or "margin" in f.lower()
                    for f in vr.failures)

    def test_analytical_formulas_correct(self, optimized_params: ParameterSet) -> None:
        """Cross-check S* and W formulas against manual computation.

        S* = alpha * s_max / (1 - (1-alpha) * exp(-beta * delta_t))
        W = ln(novelty_start / survival_threshold) / novelty_decay
        """
        vr = validate(optimized_params)
        p = optimized_params

        # Manual S* computation
        expected_s_star = (p.alpha * p.s_max) / (
            1.0 - (1.0 - p.alpha) * math.exp(-p.beta * p.delta_t)
        )
        assert vr.steady_state_strength == pytest.approx(expected_s_star, rel=1e-8)

        # Manual W computation
        expected_W = math.log(p.novelty_start / p.survival_threshold) / p.novelty_decay
        assert vr.exploration_window == pytest.approx(expected_W, rel=1e-8)

        # Ratio cross-check
        assert vr.steady_state_ratio == pytest.approx(
            expected_s_star / p.s_max, rel=1e-8
        )


# ============================================================
# 6. TestOptimizationResult (~4 tests)
# ============================================================


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass properties."""

    @pytest.fixture
    def result(self) -> OptimizationResult:
        """Short optimization for dataclass tests."""
        return run_optimization(n_generations=5, seed=42)

    def test_dataclass_frozen(self, result: OptimizationResult) -> None:
        """OptimizationResult is frozen: cannot mutate fields after construction."""
        with pytest.raises(AttributeError):
            result.best_score = 999.0  # type: ignore[misc]

    def test_per_competency_keys(self, result: OptimizationResult) -> None:
        """per_competency dict contains all 4 competency names."""
        expected_keys = {
            "accurate_retrieval",
            "test_time_learning",
            "selective_forgetting",
            "stability",
        }
        assert set(result.per_competency.keys()) == expected_keys
        # Each value should be a float in [0, 1]
        for key, val in result.per_competency.items():
            assert 0.0 <= val <= 1.0, \
                f"per_competency[{key!r}] = {val} not in [0, 1]"

    def test_best_score_matches_params(self, result: OptimizationResult) -> None:
        """Re-evaluating best_params via objective confirms best_score.

        best_score = -objective(best_x), so the values should match.
        """
        obj_val = objective(result.best_x)
        expected_score = -obj_val
        assert result.best_score == pytest.approx(expected_score, rel=1e-8)

    def test_best_x_shape(self, result: OptimizationResult) -> None:
        """best_x has shape (6,) matching PARAM_ORDER."""
        assert result.best_x.shape == (6,)
        assert len(PARAM_ORDER) == 6


# ============================================================
# 7. TestValidationResult (~additional structural tests)
# ============================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass properties."""

    def test_validation_result_frozen(self) -> None:
        """ValidationResult is frozen: cannot mutate fields."""
        result = run_optimization(n_generations=5, seed=42)
        vr = validate(result.best_params)
        with pytest.raises(AttributeError):
            vr.all_passed = False  # type: ignore[misc]

    def test_weight_summary_keys(self) -> None:
        """weight_summary contains w1, w2, w3, w4."""
        result = run_optimization(n_generations=5, seed=42)
        vr = validate(result.best_params)
        assert "w1" in vr.weight_summary
        assert "w2" in vr.weight_summary
        assert "w3" in vr.weight_summary
        assert "w4" in vr.weight_summary
        total = sum(vr.weight_summary.values())
        assert total == pytest.approx(1.0, abs=1e-10)


# ============================================================
# 8. TestStabilitySimSteps -- constant extraction
# ============================================================


class TestStabilitySimSteps:
    """Tests for the STABILITY_SIM_STEPS module-level constant."""

    def test_stability_sim_steps_value(self) -> None:
        """STABILITY_SIM_STEPS is 100 (the canonical number of simulation steps)."""
        assert STABILITY_SIM_STEPS == 100

    def test_stability_sim_steps_is_int(self) -> None:
        """STABILITY_SIM_STEPS is an integer, not a float."""
        assert isinstance(STABILITY_SIM_STEPS, int)
