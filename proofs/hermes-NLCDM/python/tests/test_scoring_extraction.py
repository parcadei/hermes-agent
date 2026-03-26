"""Tests for EmbeddingRelevanceScorer extraction to mabench.scoring.

Verifies:
1. The class can be imported from its new canonical location (mabench.scoring)
2. The class is still importable from mabench.hermes_agent (re-export)
3. The class inherits from RelevanceScorer
4. The class has all expected methods (embed, embed_batch, score, save_disk_cache)
5. Both import paths resolve to the same class object
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure sys.path includes the necessary directories
_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))


def test_import_from_scoring_module():
    """EmbeddingRelevanceScorer can be imported from mabench.scoring."""
    from mabench.scoring import EmbeddingRelevanceScorer
    assert EmbeddingRelevanceScorer is not None


def test_import_from_hermes_agent_reexport():
    """EmbeddingRelevanceScorer is still importable from mabench.hermes_agent."""
    from mabench.hermes_agent import EmbeddingRelevanceScorer
    assert EmbeddingRelevanceScorer is not None


def test_both_imports_are_same_class():
    """Both import paths resolve to the exact same class object."""
    from mabench.scoring import EmbeddingRelevanceScorer as FromScoring
    from mabench.hermes_agent import EmbeddingRelevanceScorer as FromAgent
    assert FromScoring is FromAgent


def test_inherits_from_relevance_scorer():
    """EmbeddingRelevanceScorer inherits from RelevanceScorer."""
    from mabench.scoring import EmbeddingRelevanceScorer
    from hermes_memory.orchestrator import RelevanceScorer
    assert issubclass(EmbeddingRelevanceScorer, RelevanceScorer)


def test_has_expected_methods():
    """EmbeddingRelevanceScorer has all expected public methods."""
    from mabench.scoring import EmbeddingRelevanceScorer
    expected_methods = ["embed", "embed_batch", "score", "save_disk_cache"]
    for method_name in expected_methods:
        assert hasattr(EmbeddingRelevanceScorer, method_name), (
            f"Missing method: {method_name}"
        )


def test_class_has_disk_cache_dir():
    """EmbeddingRelevanceScorer has _DISK_CACHE_DIR class attribute."""
    from mabench.scoring import EmbeddingRelevanceScorer
    assert hasattr(EmbeddingRelevanceScorer, "_DISK_CACHE_DIR")
    assert isinstance(EmbeddingRelevanceScorer._DISK_CACHE_DIR, Path)


def test_scoring_module_does_not_contain_hermes_agent_classes():
    """mabench.scoring should NOT contain HermesMemoryAgent."""
    import mabench.scoring as scoring_mod
    assert not hasattr(scoring_mod, "HermesMemoryAgent")


def test_hermes_agent_still_has_its_own_classes():
    """hermes_agent.py should still contain HermesMemoryAgent."""
    from mabench.hermes_agent import HermesMemoryAgent
    assert HermesMemoryAgent is not None
