"""Tests for DreamParams parameterization and full-stack wiring.

Validates:
  1. DreamParams new fields + Lean-proven safe bound validation
  2. dream_cycle_xb uses DreamParams instead of hardcoded values
  3. CoupledEngine.dream() passes DreamParams through
  4. HermesMemoryAgent accepts and propagates DreamParams
  5. run_hermes.py CLI constructs DreamParams from args
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Setup paths
_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_NLCDM_PYTHON))
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))


from dream_ops import DreamParams, DreamReport, dream_cycle_xb


# ==========================================================================
# Section 1: DreamParams field existence and defaults
# ==========================================================================


class TestDreamParamsFields:
    """Verify new fields exist on DreamParams with correct defaults."""

    def test_min_sep_default(self):
        p = DreamParams()
        assert p.min_sep == 0.3

    def test_prune_threshold_default(self):
        p = DreamParams()
        assert p.prune_threshold == 0.95

    def test_merge_threshold_default(self):
        p = DreamParams()
        assert p.merge_threshold == 0.90

    def test_merge_min_group_default(self):
        p = DreamParams()
        assert p.merge_min_group == 3

    def test_n_probes_default(self):
        p = DreamParams()
        assert p.n_probes == 200

    def test_separation_rate_default(self):
        p = DreamParams()
        assert p.separation_rate == 0.02

    def test_custom_values(self):
        p = DreamParams(min_sep=0.5, prune_threshold=0.8, merge_threshold=0.7,
                        merge_min_group=5, n_probes=100, separation_rate=0.05)
        assert p.min_sep == 0.5
        assert p.prune_threshold == 0.8
        assert p.merge_threshold == 0.7
        assert p.merge_min_group == 5
        assert p.n_probes == 100
        assert p.separation_rate == 0.05

    def test_frozen(self):
        """DreamParams is frozen — fields cannot be assigned after creation."""
        from dataclasses import FrozenInstanceError
        p = DreamParams()
        with pytest.raises(FrozenInstanceError):
            p.min_sep = 0.5


# ==========================================================================
# Section 2: DreamParams.validate() — Lean-proven safe bounds
# ==========================================================================


class TestDreamParamsValidation:
    """Validate the Lean-proven safe bounds from DreamConvergence.lean."""

    def test_default_params_are_safe(self):
        """Default params must pass validation (mirrors default_params_safe theorem)."""
        p = DreamParams()
        p.validate()  # Should not raise

    def test_eta_must_be_positive(self):
        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(eta=0.0).validate()

    def test_eta_must_be_less_than_min_sep_half(self):
        # eta=0.2, min_sep=0.3 => min_sep/2=0.15 => eta > min_sep/2 => fail
        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(eta=0.2, min_sep=0.3).validate()

    def test_eta_at_boundary_fails(self):
        # eta = min_sep/2 exactly => not strictly less => fail
        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(eta=0.15, min_sep=0.3).validate()

    def test_min_sep_must_be_positive(self):
        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(eta=0.001, min_sep=0.0).validate()

    def test_min_sep_can_equal_one(self):
        """min_sep <= 1 means equality is allowed."""
        p = DreamParams(eta=0.01, min_sep=1.0)
        p.validate()  # Should not raise

    def test_min_sep_above_one_fails(self):
        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(eta=0.01, min_sep=1.1).validate()

    def test_merge_must_be_less_than_prune(self):
        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(merge_threshold=0.95, prune_threshold=0.95).validate()

    def test_merge_threshold_must_be_positive(self):
        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(merge_threshold=0.0, prune_threshold=0.5).validate()

    def test_prune_threshold_can_equal_one(self):
        """prune_threshold <= 1 means equality is allowed."""
        p = DreamParams(merge_threshold=0.5, prune_threshold=1.0)
        p.validate()  # Should not raise

    def test_prune_threshold_above_one_fails(self):
        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(merge_threshold=0.5, prune_threshold=1.1).validate()

    def test_auto_validation_on_construction(self):
        """__post_init__ should call validate(), so bad params raise immediately."""
        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(eta=0.0)


# ==========================================================================
# Section 3: dream_cycle_xb accepts and uses DreamParams
# ==========================================================================


class TestDreamCycleXbParams:
    """Verify dream_cycle_xb accepts params: DreamParams and uses the values."""

    @pytest.fixture
    def small_patterns(self):
        """10 random unit-norm patterns in 8 dimensions."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 8))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def test_accepts_params_kwarg(self, small_patterns):
        """dream_cycle_xb should accept params= keyword argument."""
        result = dream_cycle_xb(small_patterns, beta=5.0, params=DreamParams(), seed=42)
        assert isinstance(result, DreamReport)

    def test_none_params_uses_defaults(self, small_patterns):
        """params=None should behave identically to no params (backward compat)."""
        r1 = dream_cycle_xb(small_patterns, beta=5.0, seed=42)
        r2 = dream_cycle_xb(small_patterns, beta=5.0, params=None, seed=42)
        np.testing.assert_array_equal(r1.patterns, r2.patterns)

    def test_custom_params_affect_output(self, small_patterns):
        """Non-default params should produce different results."""
        default_result = dream_cycle_xb(small_patterns, beta=5.0, params=DreamParams(), seed=42)
        custom = DreamParams(eta=0.05, min_sep=0.5, prune_threshold=0.8,
                             merge_threshold=0.6, n_probes=50, separation_rate=0.1)
        custom_result = dream_cycle_xb(small_patterns, beta=5.0, params=custom, seed=42)
        # At least one of patterns shape or values should differ
        differs = (
            default_result.patterns.shape != custom_result.patterns.shape
            or not np.allclose(default_result.patterns, custom_result.patterns)
        )
        assert differs, "Custom params should produce different output"

    def test_params_seed_fallback(self, small_patterns):
        """If seed arg is None, fall back to params.seed."""
        p = DreamParams(seed=123)
        r1 = dream_cycle_xb(small_patterns, beta=5.0, params=p)
        r2 = dream_cycle_xb(small_patterns, beta=5.0, params=p)
        np.testing.assert_array_equal(r1.patterns, r2.patterns)

    def test_explicit_seed_overrides_params_seed(self, small_patterns):
        """Explicit seed= arg should override params.seed."""
        p = DreamParams(seed=123)
        r1 = dream_cycle_xb(small_patterns, beta=5.0, params=p, seed=456)
        r2 = dream_cycle_xb(small_patterns, beta=5.0, seed=456)
        np.testing.assert_array_equal(r1.patterns, r2.patterns)


# ==========================================================================
# Section 4: CoupledEngine.dream() passes DreamParams through
# ==========================================================================


class TestCoupledEngineDreamParams:
    """Verify CoupledEngine stores and passes DreamParams to dream_cycle_xb."""

    def test_init_accepts_dream_params(self):
        from coupled_engine import CoupledEngine
        p = DreamParams(n_probes=50)
        engine = CoupledEngine(dim=8, dream_params=p)
        assert engine.dream_params is p

    def test_init_default_dream_params_is_none(self):
        from coupled_engine import CoupledEngine
        engine = CoupledEngine(dim=8)
        assert engine.dream_params is None

    def test_dream_passes_params(self):
        """dream() should forward dream_params to dream_cycle_xb.

        Capacity gating (Phase 14) may tighten prune/merge thresholds
        when below capacity, so we check that user-provided fields are
        preserved and thresholds are >= the original (only tightened).
        """
        from coupled_engine import CoupledEngine
        p = DreamParams(n_probes=50)
        engine = CoupledEngine(dim=8, dream_params=p)

        # Store a few patterns so dream has data to work with
        rng = np.random.default_rng(42)
        for i in range(5):
            emb = rng.standard_normal(8)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"fact {i}", embedding=emb)

        with patch("coupled_engine.dream_cycle_xb", wraps=dream_cycle_xb) as mock_dcxb:
            engine.dream()
            mock_dcxb.assert_called_once()
            _, kwargs = mock_dcxb.call_args
            forwarded = kwargs.get("params")
            # User-set field must be preserved
            assert forwarded.n_probes == p.n_probes
            # Capacity gating may tighten thresholds but never loosen
            assert forwarded.prune_threshold >= p.prune_threshold
            assert forwarded.merge_threshold >= p.merge_threshold

    def test_dream_kwarg_overrides_stored(self):
        """dream(dream_params=X) should override self.dream_params.

        Capacity gating (Phase 14) may tighten prune/merge thresholds,
        so we check that the override's non-threshold fields win.
        """
        from coupled_engine import CoupledEngine
        stored = DreamParams(n_probes=50)
        override = DreamParams(n_probes=100)
        engine = CoupledEngine(dim=8, dream_params=stored)

        rng = np.random.default_rng(42)
        for i in range(5):
            emb = rng.standard_normal(8)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"fact {i}", embedding=emb)

        with patch("coupled_engine.dream_cycle_xb", wraps=dream_cycle_xb) as mock_dcxb:
            engine.dream(dream_params=override)
            _, kwargs = mock_dcxb.call_args
            forwarded = kwargs.get("params")
            # Override's n_probes must win over stored
            assert forwarded.n_probes == override.n_probes
            assert forwarded.n_probes != stored.n_probes
            # Thresholds may be tightened by capacity gating
            assert forwarded.prune_threshold >= override.prune_threshold
            assert forwarded.merge_threshold >= override.merge_threshold


# ==========================================================================
# Section 5: HermesMemoryAgent propagates DreamParams
# ==========================================================================


class TestHermesAgentDreamParams:
    """Verify HermesMemoryAgent.__init__ accepts dream_params and wires it."""

    def test_init_accepts_dream_params(self):
        from mabench.hermes_agent import HermesMemoryAgent
        p = DreamParams(n_probes=50)
        agent = HermesMemoryAgent(dream_params=p)
        assert agent.dream_params is p

    def test_default_dream_params_is_none(self):
        from mabench.hermes_agent import HermesMemoryAgent
        agent = HermesMemoryAgent()
        assert agent.dream_params is None

    def test_dream_params_passed_to_coupled_engine(self):
        from mabench.hermes_agent import HermesMemoryAgent
        p = DreamParams(n_probes=50)
        agent = HermesMemoryAgent(dream_params=p)
        assert agent.coupled_engine.dream_params is p

    def test_reset_preserves_dream_params(self):
        from mabench.hermes_agent import HermesMemoryAgent
        p = DreamParams(n_probes=50)
        agent = HermesMemoryAgent(dream_params=p)
        agent.reset()
        assert agent.coupled_engine.dream_params is p


# ==========================================================================
# Section 6: CLI arg parsing (run_hermes.py)
# ==========================================================================


class TestRunHermesCLI:
    """Verify run_hermes.py parse_args includes dream param flags."""

    def test_parse_dream_eta(self):
        sys.path.insert(0, str(_NLCDM_PYTHON / "mabench"))
        from run_hermes import parse_args
        with patch("sys.argv", ["run_hermes.py",
                                "--dataset_config", "dummy.yaml",
                                "--dream_eta", "0.05"]):
            args = parse_args()
            assert args.dream_eta == 0.05

    def test_parse_dream_min_sep(self):
        sys.path.insert(0, str(_NLCDM_PYTHON / "mabench"))
        from run_hermes import parse_args
        with patch("sys.argv", ["run_hermes.py",
                                "--dataset_config", "dummy.yaml",
                                "--dream_min_sep", "0.4"]):
            args = parse_args()
            assert args.dream_min_sep == 0.4

    def test_parse_dream_prune_threshold(self):
        sys.path.insert(0, str(_NLCDM_PYTHON / "mabench"))
        from run_hermes import parse_args
        with patch("sys.argv", ["run_hermes.py",
                                "--dataset_config", "dummy.yaml",
                                "--dream_prune_threshold", "0.85"]):
            args = parse_args()
            assert args.dream_prune_threshold == 0.85

    def test_parse_dream_merge_threshold(self):
        sys.path.insert(0, str(_NLCDM_PYTHON / "mabench"))
        from run_hermes import parse_args
        with patch("sys.argv", ["run_hermes.py",
                                "--dataset_config", "dummy.yaml",
                                "--dream_merge_threshold", "0.80"]):
            args = parse_args()
            assert args.dream_merge_threshold == 0.80

    def test_defaults_match_dream_params_defaults(self):
        sys.path.insert(0, str(_NLCDM_PYTHON / "mabench"))
        from run_hermes import parse_args
        with patch("sys.argv", ["run_hermes.py",
                                "--dataset_config", "dummy.yaml"]):
            args = parse_args()
            assert args.dream_eta == 0.01
            assert args.dream_min_sep == 0.3
            assert args.dream_prune_threshold == 0.95
            assert args.dream_merge_threshold == 0.90
