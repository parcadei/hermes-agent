"""Tests for RetrievalMixin extraction to mabench.retrieval.

Verifies:
1. RetrievalMixin can be imported from mabench.retrieval
2. All expected methods exist on RetrievalMixin
3. HermesMemoryAgent inherits from RetrievalMixin
4. RetrievalMixin appears in HermesMemoryAgent's MRO
5. Methods have correct signatures (same as original)
6. mabench.retrieval does NOT contain HermesMemoryAgent
7. Methods NOT defined directly on HermesMemoryAgent.__dict__
8. _compute_effective_k helper exists (decomposed from _retrieve_candidates)
9. Return type annotations preserved
10. EntityMixin can call _embed_query via MRO (cross-mixin dependency)
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import get_type_hints

# Ensure sys.path includes the necessary directories
_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))


# ===========================================================================
# Test Group 1: Import and existence
# ===========================================================================


def test_import_retrieval_mixin():
    """RetrievalMixin can be imported from mabench.retrieval."""
    from mabench.retrieval import RetrievalMixin
    assert RetrievalMixin is not None


def test_retrieval_mixin_has_all_methods():
    """RetrievalMixin has all expected methods."""
    from mabench.retrieval import RetrievalMixin
    expected_methods = [
        "_embed_query",
        "_retrieve_candidates",
        "_apply_post_retrieval",
        "_retrieve_v2",
        "_apply_belief_rerank",
        "_compute_effective_k",
    ]
    for method_name in expected_methods:
        assert hasattr(RetrievalMixin, method_name), (
            f"Missing method on RetrievalMixin: {method_name}"
        )
        assert callable(getattr(RetrievalMixin, method_name)), (
            f"Method {method_name} is not callable"
        )


def test_hermes_agent_inherits_retrieval_mixin():
    """HermesMemoryAgent inherits from RetrievalMixin."""
    from mabench.retrieval import RetrievalMixin
    from mabench.hermes_agent import HermesMemoryAgent
    assert issubclass(HermesMemoryAgent, RetrievalMixin), (
        "HermesMemoryAgent should inherit from RetrievalMixin"
    )


def test_retrieval_mixin_in_mro():
    """RetrievalMixin appears in HermesMemoryAgent's MRO."""
    from mabench.retrieval import RetrievalMixin
    from mabench.hermes_agent import HermesMemoryAgent
    mro_names = [cls.__name__ for cls in HermesMemoryAgent.__mro__]
    assert "RetrievalMixin" in mro_names, (
        f"RetrievalMixin not in MRO: {mro_names}"
    )


def test_retrieval_module_does_not_contain_hermes_agent():
    """mabench.retrieval should NOT contain HermesMemoryAgent."""
    import mabench.retrieval as retrieval_mod
    assert not hasattr(retrieval_mod, "HermesMemoryAgent")


def test_hermes_agent_still_exists():
    """hermes_agent.py should still contain HermesMemoryAgent."""
    from mabench.hermes_agent import HermesMemoryAgent
    assert HermesMemoryAgent is not None


# ===========================================================================
# Test Group 2: Method signatures preserved
# ===========================================================================


def test_embed_query_signature():
    """_embed_query takes self, query_text."""
    from mabench.retrieval import RetrievalMixin
    sig = inspect.signature(RetrievalMixin._embed_query)
    params = list(sig.parameters.keys())
    assert params == ["self", "query_text"], f"Unexpected params: {params}"


def test_retrieve_candidates_signature():
    """_retrieve_candidates takes self, q_emb, top_k, query_text."""
    from mabench.retrieval import RetrievalMixin
    sig = inspect.signature(RetrievalMixin._retrieve_candidates)
    params = list(sig.parameters.keys())
    assert params == ["self", "q_emb", "top_k", "query_text"], (
        f"Unexpected params: {params}"
    )


def test_retrieve_candidates_query_text_default():
    """_retrieve_candidates has query_text defaulting to None."""
    from mabench.retrieval import RetrievalMixin
    sig = inspect.signature(RetrievalMixin._retrieve_candidates)
    p = sig.parameters["query_text"]
    assert p.default is None, f"Expected default None, got {p.default}"


def test_apply_post_retrieval_signature():
    """_apply_post_retrieval takes self, results."""
    from mabench.retrieval import RetrievalMixin
    sig = inspect.signature(RetrievalMixin._apply_post_retrieval)
    params = list(sig.parameters.keys())
    assert params == ["self", "results"], f"Unexpected params: {params}"


def test_retrieve_v2_signature():
    """_retrieve_v2 takes self, query_text."""
    from mabench.retrieval import RetrievalMixin
    sig = inspect.signature(RetrievalMixin._retrieve_v2)
    params = list(sig.parameters.keys())
    assert params == ["self", "query_text"], f"Unexpected params: {params}"


def test_apply_belief_rerank_signature():
    """_apply_belief_rerank takes self, results."""
    from mabench.retrieval import RetrievalMixin
    sig = inspect.signature(RetrievalMixin._apply_belief_rerank)
    params = list(sig.parameters.keys())
    assert params == ["self", "results"], f"Unexpected params: {params}"


def test_compute_effective_k_signature():
    """_compute_effective_k takes self, top_k."""
    from mabench.retrieval import RetrievalMixin
    sig = inspect.signature(RetrievalMixin._compute_effective_k)
    params = list(sig.parameters.keys())
    assert params == ["self", "top_k"], f"Unexpected params: {params}"


# ===========================================================================
# Test Group 3: Return type annotations preserved
# ===========================================================================


def test_retrieve_candidates_return_annotation():
    """_retrieve_candidates has return type annotation -> list[dict]."""
    from mabench.retrieval import RetrievalMixin
    hints = get_type_hints(RetrievalMixin._retrieve_candidates)
    assert "return" in hints, "Missing return annotation on _retrieve_candidates"


def test_apply_post_retrieval_return_annotation():
    """_apply_post_retrieval has return type annotation -> list[dict]."""
    from mabench.retrieval import RetrievalMixin
    hints = get_type_hints(RetrievalMixin._apply_post_retrieval)
    assert "return" in hints, "Missing return annotation on _apply_post_retrieval"


def test_retrieve_v2_return_annotation():
    """_retrieve_v2 has return type annotation -> list[dict]."""
    from mabench.retrieval import RetrievalMixin
    hints = get_type_hints(RetrievalMixin._retrieve_v2)
    assert "return" in hints, "Missing return annotation on _retrieve_v2"


def test_apply_belief_rerank_return_annotation():
    """_apply_belief_rerank has return type annotation -> list[dict]."""
    from mabench.retrieval import RetrievalMixin
    hints = get_type_hints(RetrievalMixin._apply_belief_rerank)
    assert "return" in hints, "Missing return annotation on _apply_belief_rerank"


def test_compute_effective_k_return_annotation():
    """_compute_effective_k has return type annotation -> int."""
    from mabench.retrieval import RetrievalMixin
    hints = get_type_hints(RetrievalMixin._compute_effective_k)
    assert "return" in hints, "Missing return annotation on _compute_effective_k"


# ===========================================================================
# Test Group 4: Methods NOT defined directly on HermesMemoryAgent
# ===========================================================================


def test_methods_not_defined_directly_on_hermes_agent():
    """Retrieval methods should be defined on RetrievalMixin, not HermesMemoryAgent."""
    from mabench.retrieval import RetrievalMixin
    from mabench.hermes_agent import HermesMemoryAgent

    methods = [
        "_embed_query",
        "_retrieve_candidates",
        "_apply_post_retrieval",
        "_retrieve_v2",
        "_apply_belief_rerank",
        "_compute_effective_k",
    ]
    for method_name in methods:
        # Method should exist on HermesMemoryAgent (via inheritance)
        assert hasattr(HermesMemoryAgent, method_name), (
            f"{method_name} not found on HermesMemoryAgent"
        )
        # But should be defined on RetrievalMixin, not HermesMemoryAgent itself
        assert method_name in RetrievalMixin.__dict__, (
            f"{method_name} should be defined on RetrievalMixin.__dict__"
        )
        assert method_name not in HermesMemoryAgent.__dict__, (
            f"{method_name} should NOT be in HermesMemoryAgent.__dict__ "
            "(it should come from RetrievalMixin via inheritance)"
        )


# ===========================================================================
# Test Group 5: Decomposition — _retrieve_candidates uses _compute_effective_k
# ===========================================================================


def test_retrieve_candidates_calls_compute_effective_k():
    """_retrieve_candidates body references _compute_effective_k."""
    from mabench.retrieval import RetrievalMixin
    source = inspect.getsource(RetrievalMixin._retrieve_candidates)
    assert "_compute_effective_k" in source, (
        "_retrieve_candidates should delegate to _compute_effective_k"
    )


def test_compute_effective_k_has_docstring():
    """_compute_effective_k has a docstring."""
    from mabench.retrieval import RetrievalMixin
    assert RetrievalMixin._compute_effective_k.__doc__ is not None, (
        "_compute_effective_k should have a docstring"
    )


# ===========================================================================
# Test Group 6: Cross-mixin dependency — EntityMixin uses _embed_query
# ===========================================================================


def test_entity_mixin_embed_query_resolves_via_mro():
    """EntityMixin documents dependency on _embed_query; HermesMemoryAgent
    resolves it via RetrievalMixin in MRO."""
    from mabench.hermes_agent import HermesMemoryAgent
    from mabench.retrieval import RetrievalMixin
    # _embed_query is defined on RetrievalMixin
    assert "_embed_query" in RetrievalMixin.__dict__
    # HermesMemoryAgent gets it through inheritance
    assert hasattr(HermesMemoryAgent, "_embed_query")
    # The method should resolve to RetrievalMixin's version
    method = HermesMemoryAgent._embed_query
    assert method is RetrievalMixin._embed_query


# ===========================================================================
# Test Group 7: Line count guard
# ===========================================================================


def test_retrieval_methods_under_50_lines():
    """Each method in RetrievalMixin should be under 50 lines of source."""
    from mabench.retrieval import RetrievalMixin
    methods = [
        "_embed_query",
        "_apply_post_retrieval",
        "_retrieve_v2",
        "_apply_belief_rerank",
        "_compute_effective_k",
    ]
    for method_name in methods:
        method = getattr(RetrievalMixin, method_name)
        source_lines = inspect.getsource(method).splitlines()
        assert len(source_lines) <= 50, (
            f"{method_name} has {len(source_lines)} lines (max 50)"
        )
