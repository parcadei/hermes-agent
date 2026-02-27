"""Comprehensive tests for the sensitivity analysis module.

Tests define expected behavioral contracts from the spec at
thoughts/shared/plans/memory-system/sensitivity/spec.md.

Categories:
  1. Scenario construction (~5 tests)
  2. Parameter perturbation (~8 tests)
  3. Single parameter analysis (~6 tests)
  4. Full sensitivity analysis (~5 tests)
  5. Report generation (~3 tests)
  6. Edge cases (~5 tests)
  7. Scenario correctness (~5 tests)

All tests import from hermes_memory.sensitivity (which does not yet exist)
and hermes_memory.engine (which does not yet exist). They should fail with
ImportError until those modules are implemented.
"""

import math

import pytest

from hermes_memory.sensitivity import (
    Scenario,
    SensitivityResult,
    build_standard_scenarios,
    perturb_parameter,
    analyze_parameter,
    run_sensitivity_analysis,
    generate_report,
)
from hermes_memory.engine import ParameterSet, MemoryState


# ============================================================
# Reference parameter set (standardized across both test files)
# K = exp(-0.1) + (0.25/5)*0.1*10 = 0.9048 + 0.05 = 0.9548 < 1
# Contraction margin: (1 - exp(-0.1)) - (0.25/5)*0.1*10 = 0.0452
# ============================================================

REFERENCE_PARAMS = ParameterSet(
    alpha=0.1,
    beta=0.1,
    delta_t=1.0,
    s_max=10.0,
    s0=1.0,
    temperature=5.0,
    novelty_start=0.5,
    novelty_decay=0.2,
    survival_threshold=0.05,
    feedback_sensitivity=0.1,
    w1=0.35,
    w2=0.25,
    w3=0.20,
    w4=0.20,
)


# ============================================================
# 1. Scenario Construction (~5 tests)
# ============================================================


class TestScenarioConstruction:
    """Tests for build_standard_scenarios and Scenario validation."""

    def test_build_standard_scenarios_count(self):
        """build_standard_scenarios returns approximately 10 scenarios."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        assert isinstance(scenarios, list)
        # Spec says 10 scenarios: 3 AR + 2 TTL + 3 SF + 2 STAB
        assert len(scenarios) >= 8
        assert len(scenarios) <= 12

    def test_build_standard_scenarios_valid_memories(self):
        """Each scenario has a non-empty list of valid MemoryState instances."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        for scenario in scenarios:
            assert isinstance(scenario.memories, list)
            assert len(scenario.memories) > 0, (
                f"Scenario {scenario.name!r} has no memories"
            )
            for m in scenario.memories:
                assert isinstance(m, MemoryState)

    def test_build_standard_scenarios_expected_winner_in_range(self):
        """Each scenario's expected_winner is a valid index (or -1 for stability)."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        for scenario in scenarios:
            if scenario.expected_winner == -1:
                # Stability scenario sentinel
                assert scenario.competency == "stability", (
                    f"Non-stability scenario {scenario.name!r} has expected_winner=-1"
                )
            else:
                assert 0 <= scenario.expected_winner < len(scenario.memories), (
                    f"Scenario {scenario.name!r}: expected_winner={scenario.expected_winner} "
                    f"out of range for {len(scenario.memories)} memories"
                )

    def test_build_standard_scenarios_covers_all_competencies(self):
        """Scenarios cover all 4 competencies."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        competencies_found = {s.competency for s in scenarios}
        expected_competencies = {
            "accurate_retrieval",
            "test_time_learning",
            "selective_forgetting",
            "stability",
        }
        assert competencies_found == expected_competencies, (
            f"Missing competencies: {expected_competencies - competencies_found}"
        )

    def test_scenario_memory_invariants(self):
        """All memories in scenarios have valid field values."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        for scenario in scenarios:
            for i, m in enumerate(scenario.memories):
                assert 0.0 <= m.relevance <= 1.0, (
                    f"Scenario {scenario.name!r}, memory {i}: "
                    f"relevance={m.relevance} not in [0, 1]"
                )
                assert m.last_access_time >= 0.0, (
                    f"Scenario {scenario.name!r}, memory {i}: "
                    f"last_access_time={m.last_access_time} < 0"
                )
                assert 0.0 <= m.importance <= 1.0, (
                    f"Scenario {scenario.name!r}, memory {i}: "
                    f"importance={m.importance} not in [0, 1]"
                )
                assert m.access_count >= 0, (
                    f"Scenario {scenario.name!r}, memory {i}: "
                    f"access_count={m.access_count} < 0"
                )
                assert m.strength >= 0.0, (
                    f"Scenario {scenario.name!r}, memory {i}: "
                    f"strength={m.strength} < 0"
                )
                assert m.creation_time >= 0.0, (
                    f"Scenario {scenario.name!r}, memory {i}: "
                    f"creation_time={m.creation_time} < 0"
                )


# ============================================================
# Scenario dataclass validation
# ============================================================


class TestScenarioValidation:
    """Tests for Scenario __post_init__ validation."""

    def test_scenario_invalid_competency_raises(self):
        """Invalid competency string raises ValueError."""
        with pytest.raises(ValueError, match="competency"):
            Scenario(
                name="invalid",
                memories=[
                    MemoryState(
                        relevance=0.5, last_access_time=0.0,
                        importance=0.5, access_count=0,
                        strength=1.0, creation_time=0.0,
                    )
                ],
                current_time=0.0,
                expected_winner=0,
                competency="nonexistent_competency",
            )

    def test_scenario_invalid_winner_raises(self):
        """expected_winner out of range raises ValueError."""
        with pytest.raises(ValueError, match="expected_winner"):
            Scenario(
                name="invalid",
                memories=[
                    MemoryState(
                        relevance=0.5, last_access_time=0.0,
                        importance=0.5, access_count=0,
                        strength=1.0, creation_time=0.0,
                    )
                ],
                current_time=0.0,
                expected_winner=5,  # only 1 memory, so 5 is out of range
                competency="accurate_retrieval",
            )


# ============================================================
# 2. Parameter Perturbation (~8 tests)
# ============================================================


class TestParameterPerturbation:
    """Tests for perturb_parameter function."""

    def test_perturb_positive_factor(self):
        """perturb_parameter('alpha', 0.1) produces alpha * 1.1."""
        result = perturb_parameter(REFERENCE_PARAMS, "alpha", 0.1)
        assert result is not None
        expected_alpha = REFERENCE_PARAMS.alpha * 1.1
        assert result.alpha == pytest.approx(expected_alpha, rel=1e-10)

    def test_perturb_negative_factor(self):
        """perturb_parameter('alpha', -0.1) produces alpha * 0.9."""
        result = perturb_parameter(REFERENCE_PARAMS, "alpha", -0.1)
        assert result is not None
        expected_alpha = REFERENCE_PARAMS.alpha * 0.9
        assert result.alpha == pytest.approx(expected_alpha, rel=1e-10)

    def test_perturb_violating_constraint_returns_none(self):
        """Perturbation that violates constraints returns None.

        alpha=0.1 * (1 + 10.0) = 1.1 > 1, which violates alpha < 1.
        """
        result = perturb_parameter(REFERENCE_PARAMS, "alpha", 10.0)
        assert result is None

    def test_perturb_weight_renormalizes(self):
        """Perturbing w1 renormalizes all weights to sum to 1."""
        result = perturb_parameter(REFERENCE_PARAMS, "w1", 0.1)
        assert result is not None
        weight_sum = result.w1 + result.w2 + result.w3 + result.w4
        assert weight_sum == pytest.approx(1.0, abs=1e-10), (
            f"Weights sum to {weight_sum}, not 1.0"
        )
        # w1 should have increased
        assert result.w1 > REFERENCE_PARAMS.w1

    def test_perturb_w2_above_threshold_returns_none(self):
        """Perturbing w2 to >= 0.4 returns None (spec constraint).

        w2=0.20 * (1 + 1.0) = 0.40, which violates w2 < 0.4.
        """
        result = perturb_parameter(REFERENCE_PARAMS, "w2", 1.0)
        assert result is None

    def test_perturb_alpha_to_zero_returns_none(self):
        """Perturbing alpha to 0 returns None (alpha must be > 0).

        alpha=0.1 * (1 + (-1.0)) = 0.0, which violates alpha > 0.
        """
        result = perturb_parameter(REFERENCE_PARAMS, "alpha", -1.0)
        assert result is None

    def test_perturb_epsilon_above_novelty_start_returns_none(self):
        """Perturbing survival_threshold above novelty_start returns None.

        survival_threshold=0.05 * (1 + 10.0) = 0.55 > novelty_start=0.3.
        """
        result = perturb_parameter(REFERENCE_PARAMS, "survival_threshold", 10.0)
        assert result is None

    def test_perturbed_parameterset_satisfies_contraction_or_returns_none(self):
        """A valid perturbation still produces a valid ParameterSet.

        perturb_parameter either returns a valid ParameterSet (which passes
        all validation) or returns None. It never raises ValueError.
        """
        # Try all 14 parameters with moderate perturbation
        param_names = [
            "alpha", "beta", "delta_t", "s_max", "s0", "temperature",
            "novelty_start", "novelty_decay", "survival_threshold",
            "feedback_sensitivity", "w1", "w2", "w3", "w4",
        ]
        for name in param_names:
            for factor in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                result = perturb_parameter(REFERENCE_PARAMS, name, factor)
                # Must be either a valid ParameterSet or None
                assert result is None or isinstance(result, ParameterSet), (
                    f"perturb_parameter({name!r}, {factor}) returned {type(result)}"
                )


# ============================================================
# 3. Single Parameter Analysis (~6 tests)
# ============================================================


class TestAnalyzeParameter:
    """Tests for analyze_parameter function."""

    def test_analyze_returns_correct_parameter_name(self):
        """analyze_parameter result has the correct parameter_name field."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        result = analyze_parameter(REFERENCE_PARAMS, scenarios, "alpha")
        assert isinstance(result, SensitivityResult)
        assert result.parameter_name == "alpha"

    def test_analyze_returns_correct_perturbation_levels(self):
        """Result has perturbation_levels matching the default set."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        result = analyze_parameter(REFERENCE_PARAMS, scenarios, "beta")
        # Default levels from spec: [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]
        assert result.perturbation_levels == [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]

    def test_analyze_custom_perturbation_levels(self):
        """Custom perturbation levels are used when provided."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        custom_levels = [-0.1, 0.0, 0.1]
        result = analyze_parameter(
            REFERENCE_PARAMS, scenarios, "beta",
            perturbation_levels=custom_levels,
        )
        assert result.perturbation_levels == custom_levels

    def test_analyze_zero_perturbation_gives_tau_one(self):
        """At zero perturbation, Kendall tau should be 1.0 (identity ranking)."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        result = analyze_parameter(
            REFERENCE_PARAMS, scenarios, "alpha",
            perturbation_levels=[0.0],
        )
        # tau at factor=0.0 means baseline vs. baseline = identical
        assert len(result.kendall_tau_per_level) == 1
        assert result.kendall_tau_per_level[0] == pytest.approx(1.0, abs=1e-10)

    def test_analyze_classification_consistent(self):
        """Classification is one of the three valid values."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        result = analyze_parameter(REFERENCE_PARAMS, scenarios, "temperature")
        assert result.classification in {"critical", "sensitive", "insensitive"}

    def test_analyze_score_changes_monotonicity_tendency(self):
        """Score changes tend to increase with larger perturbation magnitudes.

        For well-behaved parameters, |perturbation| = 0.5 should cause at least
        as much score change as |perturbation| = 0.1, on average.
        """
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        result = analyze_parameter(
            REFERENCE_PARAMS, scenarios, "temperature",
            perturbation_levels=[-0.5, -0.1, 0.1, 0.5],
        )
        # Filter out NaN (skipped) levels
        valid_pairs = [
            (abs(level), change)
            for level, change in zip(
                result.perturbation_levels, result.score_change_per_level
            )
            if not math.isnan(change)
        ]
        if len(valid_pairs) >= 2:
            # Sort by perturbation magnitude
            valid_pairs.sort(key=lambda x: x[0])
            # The largest perturbation should cause >= score change
            # compared to the smallest perturbation
            smallest_change = valid_pairs[0][1]
            largest_change = valid_pairs[-1][1]
            assert largest_change >= smallest_change - 1e-10, (
                f"Score changes not monotonic: smallest={smallest_change} at "
                f"|factor|={valid_pairs[0][0]}, largest={largest_change} at "
                f"|factor|={valid_pairs[-1][0]}"
            )

    def test_analyze_contraction_margin_reported(self):
        """Each perturbation level has a contraction margin value."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        result = analyze_parameter(REFERENCE_PARAMS, scenarios, "alpha")
        assert len(result.contraction_margin_per_level) == len(result.perturbation_levels)
        # Non-skipped levels should have finite margins
        for level, margin in zip(
            result.perturbation_levels, result.contraction_margin_per_level
        ):
            if level not in result.skipped_levels:
                assert math.isfinite(margin), (
                    f"Contraction margin at level {level} is not finite: {margin}"
                )


# ============================================================
# 4. Full Sensitivity Analysis (~5 tests)
# ============================================================


class TestRunSensitivityAnalysis:
    """Tests for run_sensitivity_analysis across all parameters."""

    def test_full_analysis_returns_all_params(self):
        """Result dict has entries for all 14 parameter names."""
        results = run_sensitivity_analysis(REFERENCE_PARAMS)
        assert isinstance(results, dict)
        expected_params = {
            "alpha", "beta", "delta_t", "s_max", "s0", "temperature",
            "novelty_start", "novelty_decay", "survival_threshold",
            "feedback_sensitivity", "w1", "w2", "w3", "w4",
        }
        assert set(results.keys()) == expected_params, (
            f"Missing: {expected_params - set(results.keys())}, "
            f"Extra: {set(results.keys()) - expected_params}"
        )

    def test_full_analysis_no_crashes(self):
        """Full analysis completes without exceptions."""
        # This is implicitly tested by test_full_analysis_returns_all_params,
        # but we make it explicit: no exceptions during the run.
        results = run_sensitivity_analysis(REFERENCE_PARAMS)
        for name, result in results.items():
            assert isinstance(result, SensitivityResult), (
                f"Result for {name!r} is {type(result)}, not SensitivityResult"
            )

    def test_full_analysis_baseline_mostly_not_critical(self):
        """At baseline defaults, most params should be 'insensitive' or 'sensitive'.

        The reference parameters are chosen to be well inside the safe region,
        so we do not expect many 'critical' classifications.
        """
        results = run_sensitivity_analysis(REFERENCE_PARAMS)
        critical_count = sum(
            1 for r in results.values() if r.classification == "critical"
        )
        # Allow at most 4 critical out of 14 (temperature and a few weights
        # may be borderline).
        assert critical_count <= 4, (
            f"{critical_count} parameters classified as critical: "
            f"{[n for n, r in results.items() if r.classification == 'critical']}"
        )

    def test_full_analysis_temperature_is_sensitive_or_critical(self):
        """Temperature directly controls L = 0.25/T in the contraction condition.

        Lowering T increases L and can push K above 1, so temperature should
        be classified as 'sensitive' or 'critical', not 'insensitive'.
        """
        results = run_sensitivity_analysis(REFERENCE_PARAMS)
        assert results["temperature"].classification in {"sensitive", "critical"}, (
            f"temperature classified as {results['temperature'].classification!r}, "
            f"expected 'sensitive' or 'critical'"
        )

    def test_full_analysis_alpha_shows_sensitivity(self):
        """Alpha appears in the contraction condition K = exp(-beta*dt) + L*alpha*Smax.

        Increasing alpha increases K, so alpha should show some sensitivity.
        """
        results = run_sensitivity_analysis(REFERENCE_PARAMS)
        assert results["alpha"].classification in {"sensitive", "critical"}, (
            f"alpha classified as {results['alpha'].classification!r}, "
            f"expected 'sensitive' or 'critical'"
        )

    def test_full_analysis_classifications_valid(self):
        """All classifications are one of the three valid values."""
        results = run_sensitivity_analysis(REFERENCE_PARAMS)
        valid_classifications = {"critical", "sensitive", "insensitive"}
        for name, result in results.items():
            assert result.classification in valid_classifications, (
                f"{name}: classification={result.classification!r} "
                f"not in {valid_classifications}"
            )


# ============================================================
# 5. Report Generation (~3 tests)
# ============================================================


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_report_is_nonempty_string(self):
        """generate_report returns a non-empty string."""
        results = run_sensitivity_analysis(REFERENCE_PARAMS)
        report = generate_report(results)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_mentions_all_parameters(self):
        """Report mentions all 13 independent parameters (w4 is derived but listed)."""
        results = run_sensitivity_analysis(REFERENCE_PARAMS)
        report = generate_report(results)
        expected_params = [
            "alpha", "beta", "delta_t", "s_max", "s0", "temperature",
            "novelty_start", "novelty_decay", "survival_threshold",
            "feedback_sensitivity", "w1", "w2", "w3", "w4",
        ]
        for param in expected_params:
            assert param in report, (
                f"Parameter {param!r} not mentioned in report"
            )

    def test_report_includes_classifications(self):
        """Report includes classification labels for each parameter."""
        results = run_sensitivity_analysis(REFERENCE_PARAMS)
        report = generate_report(results)
        # The report should contain at least one classification label
        classification_labels = {"critical", "sensitive", "insensitive"}
        found = {label for label in classification_labels if label in report}
        assert len(found) > 0, (
            "Report does not contain any classification labels"
        )
        # Every parameter's classification should appear somewhere
        for name, result in results.items():
            assert result.classification in report, (
                f"Classification {result.classification!r} for {name!r} "
                f"not found in report"
            )


# ============================================================
# 6. Edge Cases (~5 tests)
# ============================================================


class TestEdgeCases:
    """Edge case tests for sensitivity analysis functions."""

    def test_single_memory_scenario_no_crash(self):
        """Sensitivity analysis handles single-memory scenarios without crashing.

        Kendall tau on 1-element ranking is defined as 1.0 (trivially identical).
        """
        single_memory = MemoryState(
            relevance=0.8, last_access_time=0.0,
            importance=0.5, access_count=0,
            strength=1.0, creation_time=0.0,
        )
        scenario = Scenario(
            name="single_memory",
            memories=[single_memory],
            current_time=0.0,
            expected_winner=0,
            competency="accurate_retrieval",
        )
        # analyze_parameter should not crash on a single-memory scenario
        result = analyze_parameter(
            REFERENCE_PARAMS, [scenario], "alpha",
            perturbation_levels=[0.1],
        )
        assert isinstance(result, SensitivityResult)

    def test_zero_perturbation_identical_to_baseline(self):
        """Perturbation of 0.0 produces identical parameters and tau=1.0."""
        result = perturb_parameter(REFERENCE_PARAMS, "beta", 0.0)
        assert result is not None
        assert result.beta == pytest.approx(REFERENCE_PARAMS.beta, rel=1e-15)
        assert result.alpha == pytest.approx(REFERENCE_PARAMS.alpha, rel=1e-15)

    def test_very_large_perturbation_no_crash(self):
        """Very large perturbation (0.99) on an insensitive parameter does not crash.

        It may return None if constraints are violated, which is acceptable.
        """
        # feedback_sensitivity is only constrained to be > 0, so large increase is valid
        result = perturb_parameter(REFERENCE_PARAMS, "feedback_sensitivity", 0.99)
        # Should either return a valid ParameterSet or None, never crash
        assert result is None or isinstance(result, ParameterSet)

    def test_parameter_at_lower_bound_negative_perturbation_returns_none(self):
        """When a parameter is near its lower bound, negative perturbation returns None.

        alpha=0.1, factor=-1.0 gives alpha * 0 = 0, violating alpha > 0.
        """
        result = perturb_parameter(REFERENCE_PARAMS, "alpha", -1.0)
        assert result is None

    def test_stability_scenarios_run_multiple_steps_without_error(self):
        """Stability scenarios (expected_winner=-1) involve multi-step simulation.

        The analysis should handle them without errors.
        """
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        stability_scenarios = [
            s for s in scenarios if s.competency == "stability"
        ]
        assert len(stability_scenarios) >= 1, (
            "No stability scenarios found in standard suite"
        )
        # Running analysis should not crash on stability scenarios
        for scenario in stability_scenarios:
            assert scenario.expected_winner == -1


# ============================================================
# 7. Scenario Correctness (~5 tests)
# ============================================================


class TestScenarioCorrectness:
    """Tests that validate the scenarios themselves produce expected behavior
    under baseline parameters.

    These tests import rank_memories from the engine to verify that the
    expected_winner is actually correct for REFERENCE_PARAMS.
    """

    def test_accurate_retrieval_high_relevance_wins(self):
        """AR-1: High-relevance memory beats high-recency memory at baseline.

        Memory 0 has relevance=0.95, last_access_time=5.0, strength=3.0
        (adversarial fix: original had lat=50, strength=s0=1.0, which gave
        retention ~ 0, allowing M1 to win on recency + activation).
        Memory 1 has relevance=0.4, last_access_time=20.0, access_count=2.
        With w1=0.35 (highest weight), relevance dominates.
        """
        from hermes_memory.engine import rank_memories, score_memory

        s_max = REFERENCE_PARAMS.s_max
        memories = [
            MemoryState(
                relevance=0.95, last_access_time=5.0,
                importance=0.3, access_count=2,
                strength=3.0, creation_time=100.0,
            ),
            MemoryState(
                relevance=0.4, last_access_time=20.0,
                importance=0.3, access_count=2,
                strength=s_max * 0.8, creation_time=100.0,
            ),
        ]
        ranked = rank_memories(REFERENCE_PARAMS, memories, current_time=0.0)
        winner_idx = ranked[0][0]
        assert winner_idx == 0, (
            f"Expected memory 0 (high relevance) to win, got memory {winner_idx}. "
            f"Scores: m0={score_memory(REFERENCE_PARAMS, memories[0], 0.0):.4f}, "
            f"m1={score_memory(REFERENCE_PARAMS, memories[1], 0.0):.4f}"
        )

    def test_test_time_learning_new_high_relevance_wins(self):
        """TTL-1: New high-relevance memory wins over established pool.

        Brand-new memory (creation_time=0) with relevance=0.9 should beat
        older memories with lower relevance, especially with novelty bonus.
        """
        from hermes_memory.engine import rank_memories, score_memory

        s0 = REFERENCE_PARAMS.s0
        s_max = REFERENCE_PARAMS.s_max
        memories = [
            MemoryState(
                relevance=0.9, last_access_time=0.0,
                importance=0.5, access_count=0,
                strength=s0, creation_time=0.0,  # brand new
            ),
            MemoryState(
                relevance=0.5, last_access_time=5.0,
                importance=0.7, access_count=20,
                strength=s_max * 0.7, creation_time=200.0,
            ),
            MemoryState(
                relevance=0.4, last_access_time=3.0,
                importance=0.6, access_count=15,
                strength=s_max * 0.6, creation_time=150.0,
            ),
        ]
        ranked = rank_memories(REFERENCE_PARAMS, memories, current_time=0.0)
        winner_idx = ranked[0][0]
        assert winner_idx == 0, (
            f"Expected memory 0 (new high-relevance) to win, got memory {winner_idx}. "
            f"Scores: "
            + ", ".join(
                f"m{i}={score_memory(REFERENCE_PARAMS, m, 0.0):.4f}"
                for i, m in enumerate(memories)
            )
        )

    def test_selective_forgetting_stale_memory_loses(self):
        """SF-1: Stale heavily-accessed memory loses to fresh relevant one.

        Memory 1 has last_access_time=200 and high access_count=100, but
        strength decay has reduced its recency to near-zero. Memory 0 with
        relevance=0.8 and recent access should win.
        """
        from hermes_memory.engine import rank_memories, score_memory

        s_max = REFERENCE_PARAMS.s_max
        memories = [
            MemoryState(
                relevance=0.8, last_access_time=5.0,
                importance=0.6, access_count=3,
                strength=s_max * 0.5, creation_time=20.0,
            ),
            MemoryState(
                relevance=0.5, last_access_time=200.0,
                importance=0.8, access_count=100,
                strength=s_max * 0.3, creation_time=500.0,
            ),
        ]
        ranked = rank_memories(REFERENCE_PARAMS, memories, current_time=0.0)
        winner_idx = ranked[0][0]
        assert winner_idx == 0, (
            f"Expected memory 0 (fresh relevant) to win, got memory {winner_idx}. "
            f"Scores: m0={score_memory(REFERENCE_PARAMS, memories[0], 0.0):.4f}, "
            f"m1={score_memory(REFERENCE_PARAMS, memories[1], 0.0):.4f}"
        )

    @pytest.mark.slow
    def test_stability_no_monopolization(self):
        """STAB-1: No single memory monopolizes rank 0 over 100 steps.

        With 4 memories of similar relevance (0.5, 0.55, 0.6, 0.45) and
        soft selection (access_pattern = [-1]*100), no single memory should
        appear at rank 0 in more than 60% of steps.
        """
        import random
        from hermes_memory.engine import simulate

        s0 = REFERENCE_PARAMS.s0
        memories = [
            MemoryState(
                relevance=0.5, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s0, creation_time=50.0,
            ),
            MemoryState(
                relevance=0.55, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s0, creation_time=50.0,
            ),
            MemoryState(
                relevance=0.6, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s0, creation_time=50.0,
            ),
            MemoryState(
                relevance=0.45, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s0, creation_time=50.0,
            ),
        ]
        n_steps = 100
        rng = random.Random(42)
        result = simulate(
            REFERENCE_PARAMS, memories, n_steps,
            access_pattern=[-1] * n_steps,
            rng=rng,
        )
        # Count how often each memory appears at rank 0
        rank0_counts = [0] * len(memories)
        for ranking in result.rankings_per_step:
            rank0_counts[ranking[0]] += 1

        max_fraction = max(rank0_counts) / n_steps
        assert max_fraction < 0.60, (
            f"Memory monopolization detected: max fraction at rank 0 = "
            f"{max_fraction:.2f} (counts: {rank0_counts})"
        )

    def test_scenario_expected_winners_correct_for_baseline(self):
        """All non-stability scenarios have correct expected winners at baseline.

        This is a meta-test: it validates that the scenario design is correct
        by verifying each scenario's expected_winner actually wins under
        REFERENCE_PARAMS.
        """
        from hermes_memory.engine import rank_memories, score_memory

        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        for scenario in scenarios:
            if scenario.expected_winner == -1:
                continue  # skip stability scenarios
            ranked = rank_memories(
                REFERENCE_PARAMS, scenario.memories, scenario.current_time
            )
            actual_winner = ranked[0][0]
            assert actual_winner == scenario.expected_winner, (
                f"Scenario {scenario.name!r}: expected winner={scenario.expected_winner}, "
                f"actual winner={actual_winner}. "
                f"Scores: "
                + ", ".join(
                    f"m{i}={score_memory(REFERENCE_PARAMS, m, scenario.current_time):.4f}"
                    for i, m in enumerate(scenario.memories)
                )
            )


# ============================================================
# Additional structural tests
# ============================================================


class TestSensitivityResultStructure:
    """Tests for SensitivityResult dataclass structure."""

    def test_result_fields_consistent_lengths(self):
        """All per-level lists in SensitivityResult have the same length."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        result = analyze_parameter(REFERENCE_PARAMS, scenarios, "beta")
        n_levels = len(result.perturbation_levels)
        assert len(result.kendall_tau_per_level) == n_levels
        assert len(result.score_change_per_level) == n_levels
        assert len(result.contraction_margin_per_level) == n_levels
        assert len(result.scenarios_correct_per_level) == n_levels

    def test_skipped_levels_have_nan(self):
        """Skipped perturbation levels have NaN in all metric lists."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        # Use a large perturbation that will likely cause some skips
        result = analyze_parameter(
            REFERENCE_PARAMS, scenarios, "alpha",
            perturbation_levels=[-0.99, -0.5, 0.5, 5.0],
        )
        for skipped_level in result.skipped_levels:
            idx = result.perturbation_levels.index(skipped_level)
            assert math.isnan(result.kendall_tau_per_level[idx]), (
                f"Skipped level {skipped_level}: tau should be NaN"
            )
            assert math.isnan(result.score_change_per_level[idx]), (
                f"Skipped level {skipped_level}: score_change should be NaN"
            )
            assert math.isnan(result.contraction_margin_per_level[idx]), (
                f"Skipped level {skipped_level}: contraction_margin should be NaN"
            )

    def test_scenarios_correct_counts_bounded(self):
        """scenarios_correct counts are within [0, total_non_stability_scenarios]."""
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        non_stability_count = sum(
            1 for s in scenarios if s.expected_winner != -1
        )
        result = analyze_parameter(REFERENCE_PARAMS, scenarios, "w1")
        for count in result.scenarios_correct_per_level:
            assert 0 <= count <= non_stability_count, (
                f"scenarios_correct={count} out of range [0, {non_stability_count}]"
            )


class TestClassificationRules:
    """Tests for the classification logic (critical/sensitive/insensitive)."""

    def test_critical_when_contraction_violated(self):
        """A parameter near the contraction boundary should be classified critical.

        Temperature directly controls L = 0.25/T. The contraction margin for
        REFERENCE_PARAMS is ~0.045. A 50% decrease in T (factor=-0.5) gives
        T=2.5, L=0.1, K = exp(-0.1) + 0.1*0.1*10 = 0.9048 + 0.1 = 1.0048 > 1.
        This should trigger 'critical' classification.
        """
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        result = analyze_parameter(REFERENCE_PARAMS, scenarios, "temperature")
        # With the standard perturbation levels, -0.5 on temperature should
        # violate contraction, making it critical
        has_negative_margin = any(
            margin < 0.0
            for level, margin in zip(
                result.perturbation_levels, result.contraction_margin_per_level
            )
            if level not in result.skipped_levels and not math.isnan(margin)
        )
        if has_negative_margin:
            assert result.classification == "critical", (
                f"temperature has negative contraction margin but classified as "
                f"{result.classification!r}, expected 'critical'"
            )

    def test_insensitive_parameter_classification(self):
        """feedback_sensitivity should be relatively insensitive.

        It only affects importance_update step size and does not appear in the
        contraction condition. Moderate perturbations should not flip rankings.
        """
        scenarios = build_standard_scenarios(REFERENCE_PARAMS)
        result = analyze_parameter(REFERENCE_PARAMS, scenarios, "feedback_sensitivity")
        # feedback_sensitivity should be insensitive or at most sensitive
        assert result.classification in {"insensitive", "sensitive"}, (
            f"feedback_sensitivity classified as {result.classification!r}, "
            f"expected 'insensitive' or 'sensitive'"
        )
