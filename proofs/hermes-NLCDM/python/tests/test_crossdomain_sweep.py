"""Tests for cross-domain sweep script and CLI arg wiring.

Tests:
  1. run_hermes.py accepts --cross_domain_probes, --probe_frequency, --probes_per_session CLI args
  2. run_crossdomain_sweep.py configs list has required entries
  3. run_crossdomain_sweep.py is importable and has make_agent_factory, main
  4. make_agent_factory produces valid agents with coretrieval settings
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

# Setup paths
_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_NLCDM_PYTHON))
sys.path.insert(0, str(_NLCDM_PYTHON / "mabench"))


# ---------------------------------------------------------------------------
# Test: run_hermes.py CLI arg parsing accepts new cross-domain flags
# ---------------------------------------------------------------------------


class TestRunHermesCLIArgs:
    """run_hermes.py parse_args() should accept the new cross-domain probe flags."""

    def test_parse_args_accepts_cross_domain_probes(self):
        """--cross_domain_probes should be a valid store_true arg."""
        from mabench.run_hermes import parse_args
        # Monkey-patch sys.argv to simulate CLI args
        import sys as _sys
        original_argv = _sys.argv
        try:
            _sys.argv = [
                "run_hermes.py",
                "--dataset_config", "dummy.yaml",
                "--cross_domain_probes",
            ]
            args = parse_args()
            assert args.cross_domain_probes is True
        finally:
            _sys.argv = original_argv

    def test_parse_args_cross_domain_probes_default_false(self):
        """--cross_domain_probes should default to False."""
        from mabench.run_hermes import parse_args
        import sys as _sys
        original_argv = _sys.argv
        try:
            _sys.argv = [
                "run_hermes.py",
                "--dataset_config", "dummy.yaml",
            ]
            args = parse_args()
            assert args.cross_domain_probes is False
        finally:
            _sys.argv = original_argv

    def test_parse_args_accepts_probe_frequency(self):
        """--probe_frequency should be an int arg with default 10."""
        from mabench.run_hermes import parse_args
        import sys as _sys
        original_argv = _sys.argv
        try:
            _sys.argv = [
                "run_hermes.py",
                "--dataset_config", "dummy.yaml",
                "--probe_frequency", "5",
            ]
            args = parse_args()
            assert args.probe_frequency == 5
        finally:
            _sys.argv = original_argv

    def test_parse_args_probe_frequency_default(self):
        """--probe_frequency should default to 10."""
        from mabench.run_hermes import parse_args
        import sys as _sys
        original_argv = _sys.argv
        try:
            _sys.argv = [
                "run_hermes.py",
                "--dataset_config", "dummy.yaml",
            ]
            args = parse_args()
            assert args.probe_frequency == 10
        finally:
            _sys.argv = original_argv

    def test_parse_args_accepts_probes_per_session(self):
        """--probes_per_session should be an int arg with default 3."""
        from mabench.run_hermes import parse_args
        import sys as _sys
        original_argv = _sys.argv
        try:
            _sys.argv = [
                "run_hermes.py",
                "--dataset_config", "dummy.yaml",
                "--probes_per_session", "7",
            ]
            args = parse_args()
            assert args.probes_per_session == 7
        finally:
            _sys.argv = original_argv

    def test_parse_args_probes_per_session_default(self):
        """--probes_per_session should default to 3."""
        from mabench.run_hermes import parse_args
        import sys as _sys
        original_argv = _sys.argv
        try:
            _sys.argv = [
                "run_hermes.py",
                "--dataset_config", "dummy.yaml",
            ]
            args = parse_args()
            assert args.probes_per_session == 3
        finally:
            _sys.argv = original_argv


# ---------------------------------------------------------------------------
# Test: run_crossdomain_sweep.py structure
# ---------------------------------------------------------------------------


class TestCrossdomainSweepScript:
    """run_crossdomain_sweep.py should be importable with correct structure."""

    def test_syntax_valid(self):
        """Script should be valid Python syntax."""
        script = _NLCDM_PYTHON / "run_crossdomain_sweep.py"
        source = script.read_text()
        # Will raise SyntaxError if invalid
        ast.parse(source)

    def test_importable(self):
        """Script should be importable without errors."""
        import run_crossdomain_sweep
        assert hasattr(run_crossdomain_sweep, "configs")
        assert hasattr(run_crossdomain_sweep, "make_agent_factory")
        assert hasattr(run_crossdomain_sweep, "main")

    def test_configs_has_six_entries(self):
        """configs list should have 6 configurations."""
        from run_crossdomain_sweep import configs
        assert len(configs) == 6

    def test_configs_has_cosine_baseline(self):
        """First config should be the cosine baseline."""
        from run_crossdomain_sweep import configs
        baseline = configs[0]
        assert baseline["name"] == "cosine_baseline"
        assert baseline["coretrieval"] is False
        assert baseline["cross_domain_probes"] is False

    def test_configs_has_probes_only_control(self):
        """Second config: probes but no co-retrieval (control)."""
        from run_crossdomain_sweep import configs
        ctrl = configs[1]
        assert ctrl["name"] == "probes_only_no_coretrieval"
        assert ctrl["coretrieval"] is False
        assert ctrl["cross_domain_probes"] is True
        assert ctrl["probe_frequency"] == 10
        assert ctrl["probes_per_session"] == 3

    def test_configs_has_coretrieval_b005(self):
        """Third config: probes + co-retrieval bonus=0.05."""
        from run_crossdomain_sweep import configs
        cfg = configs[2]
        assert cfg["name"] == "probes_coretrieval_b005"
        assert cfg["coretrieval"] is True
        assert cfg["cross_domain_probes"] is True
        assert cfg["coretrieval_bonus"] == 0.05

    def test_configs_has_coretrieval_b010(self):
        """Fourth config: probes + co-retrieval bonus=0.10."""
        from run_crossdomain_sweep import configs
        cfg = configs[3]
        assert cfg["name"] == "probes_coretrieval_b010"
        assert cfg["coretrieval"] is True
        assert cfg["coretrieval_bonus"] == 0.10

    def test_configs_has_coretrieval_b015(self):
        """Fifth config: probes + co-retrieval bonus=0.15."""
        from run_crossdomain_sweep import configs
        cfg = configs[4]
        assert cfg["name"] == "probes_coretrieval_b015"
        assert cfg["coretrieval"] is True
        assert cfg["coretrieval_bonus"] == 0.15

    def test_configs_has_freq5_variant(self):
        """Sixth config: higher probe frequency (every 5 sessions)."""
        from run_crossdomain_sweep import configs
        cfg = configs[5]
        assert cfg["name"] == "probes_freq5_b010"
        assert cfg["probe_frequency"] == 5
        assert cfg["coretrieval_bonus"] == 0.10

    def test_make_agent_factory_returns_callable(self):
        """make_agent_factory should return a callable."""
        from run_crossdomain_sweep import make_agent_factory
        factory = make_agent_factory(coretrieval=False)
        assert callable(factory)

    def test_make_agent_factory_with_coretrieval(self):
        """make_agent_factory with coretrieval=True should produce agent with flag set."""
        from run_crossdomain_sweep import make_agent_factory
        factory = make_agent_factory(
            coretrieval=True, coretrieval_bonus=0.1, coretrieval_min_count=1.0,
        )
        agent = factory()
        assert agent.coretrieval_retrieval is True
        assert agent.coretrieval_bonus == 0.1
        assert agent.coretrieval_min_count == 1.0

    def test_all_configs_have_required_keys(self):
        """Every config must have at least 'name', 'coretrieval', 'cross_domain_probes'."""
        from run_crossdomain_sweep import configs
        required = {"name", "coretrieval", "cross_domain_probes"}
        for i, cfg in enumerate(configs):
            missing = required - set(cfg.keys())
            assert not missing, (
                f"Config {i} ({cfg.get('name', '?')}) missing keys: {missing}"
            )
