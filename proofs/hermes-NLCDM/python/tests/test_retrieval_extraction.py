"""Tests for Retriever (formerly RetrievalMixin) in mabench.retrieval.

Verifies:
1. RetrievalMixin can be imported from mabench.retrieval (backward compat alias)
2. Retriever can be imported from mabench.retrieval (new name)
3. All expected methods exist on Retriever / RetrievalMixin
4. HermesMemoryAgent inherits from RetrievalMixin (i.e. Retriever)
5. Retriever or RetrievalMixin appears in HermesMemoryAgent's MRO
6. Methods have correct signatures (same as original)
7. mabench.retrieval does NOT contain HermesMemoryAgent
8. Methods NOT defined directly on HermesMemoryAgent.__dict__
9. _compute_effective_k helper exists (decomposed from _retrieve_candidates)
10. Return type annotations preserved
11. EntityMixin can call _embed_query via MRO (cross-mixin dependency)
12. Standalone Retriever can be instantiated with explicit deps + config
13. _resolve_* properties use config in standalone mode, fallback in mixin mode
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
    """Retriever (or RetrievalMixin) appears in HermesMemoryAgent's MRO."""
    from mabench.retrieval import RetrievalMixin
    from mabench.hermes_agent import HermesMemoryAgent
    mro_names = [cls.__name__ for cls in HermesMemoryAgent.__mro__]
    assert "RetrievalMixin" in mro_names or "Retriever" in mro_names, (
        f"Neither RetrievalMixin nor Retriever in MRO: {mro_names}"
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


# ===========================================================================
# Test Group 8: Retriever class (new standalone name)
# ===========================================================================


def test_import_retriever():
    """Retriever can be imported from mabench.retrieval."""
    from mabench.retrieval import Retriever
    assert Retriever is not None


def test_retrieval_mixin_is_retriever_alias():
    """RetrievalMixin is an alias for Retriever."""
    from mabench.retrieval import Retriever, RetrievalMixin
    assert RetrievalMixin is Retriever


def test_retriever_has_init():
    """Retriever has an __init__ method defined on itself."""
    from mabench.retrieval import Retriever
    assert "__init__" in Retriever.__dict__, (
        "Retriever should define its own __init__"
    )


def test_retriever_init_signature():
    """Retriever.__init__ accepts engine, scorer, belief_index, and config objects."""
    from mabench.retrieval import Retriever
    sig = inspect.signature(Retriever.__init__)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "engine" in params
    assert "scorer" in params
    assert "belief_index" in params
    assert "config" in params
    assert "cr_config" in params
    assert "dedup_config" in params
    assert "belief_config" in params


def test_retriever_init_defaults_to_none():
    """All config params in __init__ default to None."""
    from mabench.retrieval import Retriever
    sig = inspect.signature(Retriever.__init__)
    for name in ("config", "cr_config", "dedup_config", "belief_config",
                 "belief_index"):
        p = sig.parameters[name]
        assert p.default is None, (
            f"{name} should default to None, got {p.default}"
        )


# ===========================================================================
# Test Group 9: Standalone instantiation with config
# ===========================================================================


class _FakeScorer:
    """Minimal scorer stub for standalone Retriever tests."""
    def embed(self, text: str):
        import numpy as np
        return np.zeros(8)


class _FakeEngine:
    """Minimal engine stub for standalone Retriever tests."""
    memory_store = []

    def query(self, embedding, top_k):
        return [{"text": "fact1", "score": 0.9}]


def test_retriever_standalone_instantiation():
    """Retriever can be instantiated with explicit deps and default configs."""
    from mabench.retrieval import Retriever
    r = Retriever(engine=_FakeEngine(), scorer=_FakeScorer())
    assert r._engine is not None
    assert r._scorer is not None
    assert r._belief_index is None


def test_retriever_standalone_with_explicit_config():
    """Retriever uses the provided RetrievalConfig."""
    from mabench.retrieval import Retriever
    from mabench.configs import RetrievalConfig
    cfg = RetrievalConfig(retrieve_num=20, cooc_boost=True)
    r = Retriever(engine=_FakeEngine(), scorer=_FakeScorer(), config=cfg)
    assert r._cfg.retrieve_num == 20
    assert r._cfg.cooc_boost is True


def test_retriever_standalone_embed_query():
    """Standalone Retriever._embed_query delegates to scorer.embed()."""
    from mabench.retrieval import Retriever
    r = Retriever(engine=_FakeEngine(), scorer=_FakeScorer())
    emb = r._embed_query("test query")
    assert emb is not None
    assert len(emb) == 8


def test_retriever_standalone_compute_effective_k_no_cr():
    """Standalone Retriever._compute_effective_k returns base k when no CR."""
    from mabench.retrieval import Retriever
    from mabench.configs import RetrievalConfig, CRConfig
    r = Retriever(
        engine=_FakeEngine(),
        scorer=_FakeScorer(),
        config=RetrievalConfig(retrieve_num=10),
        cr_config=CRConfig(enabled=False),
    )
    assert r._compute_effective_k(10) == 10


def test_retriever_standalone_compute_effective_k_with_cr():
    """Standalone Retriever._compute_effective_k applies CR overfetch."""
    from mabench.retrieval import Retriever
    from mabench.configs import RetrievalConfig, CRConfig
    r = Retriever(
        engine=_FakeEngine(),
        scorer=_FakeScorer(),
        config=RetrievalConfig(retrieve_num=10),
        cr_config=CRConfig(enabled=True, overfetch=3),
    )
    assert r._compute_effective_k(10) == 30


def test_retriever_standalone_retrieve_candidates():
    """Standalone Retriever._retrieve_candidates calls engine.query."""
    import numpy as np
    from mabench.retrieval import Retriever
    r = Retriever(engine=_FakeEngine(), scorer=_FakeScorer())
    results = r._retrieve_candidates(np.zeros(8), top_k=10)
    assert len(results) == 1
    assert results[0]["text"] == "fact1"


def test_retriever_standalone_retrieve_v2():
    """Standalone Retriever._retrieve_v2 runs full pipeline."""
    from mabench.retrieval import Retriever
    r = Retriever(engine=_FakeEngine(), scorer=_FakeScorer())
    results = r._retrieve_v2("test query")
    assert len(results) == 1


# ===========================================================================
# Test Group 10: _resolve_* properties (config vs mixin fallback)
# ===========================================================================


def test_resolve_properties_exist():
    """Retriever has _resolve_* properties for all config-derived attributes."""
    from mabench.retrieval import Retriever
    expected_resolvers = [
        "_resolve_retrieve_num",
        "_resolve_conflict_resolution",
        "_resolve_cosine_dedup",
        "_resolve_cr_overfetch",
        "_resolve_belief_overfetch",
        "_resolve_engine",
        "_resolve_belief_index",
        "_resolve_scorer",
        "_resolve_transfer",
        "_resolve_coretrieval",
        "_resolve_cooc_boost",
        "_resolve_ppr",
        "_resolve_hybrid",
        "_resolve_associative",
        "_resolve_sparse",
        "_resolve_transfer_k",
        "_resolve_coretrieval_bonus",
        "_resolve_coretrieval_min_count",
        "_resolve_cooc_weight",
        "_resolve_cooc_gate_threshold",
        "_resolve_ppr_weight",
        "_resolve_ppr_damping",
        "_resolve_bm25_weight",
        "_resolve_dream_weight",
    ]
    for prop_name in expected_resolvers:
        assert hasattr(Retriever, prop_name), (
            f"Missing resolver property: {prop_name}"
        )


def test_resolve_retrieve_num_standalone():
    """_resolve_retrieve_num reads from config in standalone mode."""
    from mabench.retrieval import Retriever
    from mabench.configs import RetrievalConfig
    r = Retriever(
        engine=_FakeEngine(),
        scorer=_FakeScorer(),
        config=RetrievalConfig(retrieve_num=42),
    )
    assert r._resolve_retrieve_num == 42


def test_resolve_conflict_resolution_standalone():
    """_resolve_conflict_resolution reads from CRConfig in standalone mode."""
    from mabench.retrieval import Retriever
    from mabench.configs import CRConfig
    r = Retriever(
        engine=_FakeEngine(),
        scorer=_FakeScorer(),
        cr_config=CRConfig(enabled=True),
    )
    assert r._resolve_conflict_resolution is True


def test_resolve_engine_standalone():
    """_resolve_engine returns the injected engine in standalone mode."""
    from mabench.retrieval import Retriever
    eng = _FakeEngine()
    r = Retriever(engine=eng, scorer=_FakeScorer())
    assert r._resolve_engine is eng
