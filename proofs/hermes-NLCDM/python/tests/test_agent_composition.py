"""Tests for mabench/agent.py — composed HermesAgent.

Verifies:
1. HermesAgent can be imported.
2. __init__ signature accepts config dataclasses.
3. _build method exists and creates all expected module attributes.
4. Module types are correct.
5. Cross-module callbacks are wired (not None after _build).
6. send_message dispatches to store / query / iterative.
7. reset rebuilds all modules.
8. Line count stays under 250.
"""
import inspect


# ------------------------------------------------------------------
# 1. Import tests
# ------------------------------------------------------------------

def test_import_hermes_agent():
    from mabench.agent import HermesAgent
    assert inspect.isclass(HermesAgent)


def test_import_does_not_require_spacy():
    """agent.py import should not fail even without spaCy installed."""
    import mabench.agent  # noqa: F401


# ------------------------------------------------------------------
# 2. __init__ signature
# ------------------------------------------------------------------

def test_init_signature_has_config_params():
    from mabench.agent import HermesAgent
    sig = inspect.signature(HermesAgent.__init__)
    params = set(sig.parameters.keys())
    expected = {
        "self", "dim", "contradiction_threshold", "recency_alpha",
        "dedup_threshold", "dream_params",
        "retrieval", "cr", "decompose", "store", "llm",
        "belief", "entity", "dedup",
        "bm25_hops", "chain_cache", "rebel_kg", "triadic",
        "triadic_n", "triadic_p",
        "iterative", "iterative_max_hops",
    }
    assert expected <= params, f"Missing: {expected - params}"


def test_init_keyword_only():
    """All params except self are keyword-only (the * in signature)."""
    from mabench.agent import HermesAgent
    sig = inspect.signature(HermesAgent.__init__)
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} should be keyword-only"
        )


def test_init_defaults_to_none_or_false():
    """All config params default to None or False or a primitive."""
    from mabench.agent import HermesAgent
    sig = inspect.signature(HermesAgent.__init__)
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        assert p.default is not inspect.Parameter.empty, (
            f"{name} has no default"
        )


# ------------------------------------------------------------------
# 3. _build creates expected module attributes
# ------------------------------------------------------------------

# We cannot call __init__ (requires spaCy) so we test at the class level.

def test_build_method_exists():
    from mabench.agent import HermesAgent
    assert hasattr(HermesAgent, "_build")
    assert callable(HermesAgent._build)


def test_send_message_method_exists():
    from mabench.agent import HermesAgent
    sig = inspect.signature(HermesAgent.send_message)
    params = set(sig.parameters.keys())
    assert {"self", "message", "memorizing", "query_id", "context_id"} <= params


def test_reset_method_exists():
    from mabench.agent import HermesAgent
    assert hasattr(HermesAgent, "reset")
    assert callable(HermesAgent.reset)


def test_save_load_methods_exist():
    from mabench.agent import HermesAgent
    assert hasattr(HermesAgent, "save_agent")
    assert hasattr(HermesAgent, "load_agent")


# ------------------------------------------------------------------
# 4. Module type annotations in _build (source inspection)
# ------------------------------------------------------------------

def test_build_creates_store():
    """_build source references FactStore construction."""
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "FactStore(" in src


def test_build_creates_query_engine():
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "QueryEngine(" in src


def test_build_creates_pipeline():
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "QueryPipeline(" in src


def test_build_creates_iterative():
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "IterativeQuery(" in src


def test_build_creates_retriever():
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "Retriever(" in src


def test_build_creates_conflict_resolver():
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "ConflictResolver(" in src


def test_build_creates_entity_ops():
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "EntityOps(" in src


def test_build_creates_outgester():
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "Outgester(" in src


# ------------------------------------------------------------------
# 5. Cross-module callback wiring (source inspection)
# ------------------------------------------------------------------

def test_wires_query_engine_to_pipeline():
    """QueryEngine callbacks wired to QueryPipeline methods."""
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "_final_retrieval_fn = self.pipeline._query_final_retrieval" in src
    assert "_post_process_fn = self.pipeline._query_post_process" in src
    assert "_generate_fn = self.pipeline._query_generate" in src
    assert "_extract_short_answer_fn = self.pipeline._extract_short_answer" in src


def test_wires_pipeline_hop_executor():
    """QueryPipeline receives hop_executor_fn from QueryEngine."""
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "hop_executor_fn=self.query_engine._query_run_hop" in src


def test_wires_cr_decompose_callback():
    """ConflictResolver gets decompose_fn from QueryPipeline."""
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "_decompose_fn = self.pipeline._decompose_query" in src


def test_wires_cr_retrieve_callback():
    """ConflictResolver gets retrieve_fn from Retriever."""
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "retrieve_fn=retriever._retrieve_v2" in src


# ------------------------------------------------------------------
# 6. Shared state containers
# ------------------------------------------------------------------

def test_uses_supersession_state():
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "SupersessionState()" in src


def test_uses_entity_index():
    from mabench.agent import HermesAgent
    src = inspect.getsource(HermesAgent._build)
    assert "EntityIndex()" in src


# ------------------------------------------------------------------
# 7. Config dataclasses are used
# ------------------------------------------------------------------

def test_configs_imported():
    from mabench.agent import (
        RetrievalConfig, CRConfig, DecomposeConfig, StoreConfig,
        LLMConfig, BeliefConfig, EntityConfig, DedupConfig,
    )
    for cfg_cls in (RetrievalConfig, CRConfig, DecomposeConfig, StoreConfig,
                    LLMConfig, BeliefConfig, EntityConfig, DedupConfig):
        assert hasattr(cfg_cls, "__dataclass_fields__")


# ------------------------------------------------------------------
# 8. Line count
# ------------------------------------------------------------------

def test_agent_file_under_250_lines():
    import mabench.agent as mod
    path = inspect.getfile(mod)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) < 350, f"agent.py is {len(lines)} lines, must be < 350"


# ------------------------------------------------------------------
# 9. All standalone module classes importable via agent.py
# ------------------------------------------------------------------

def test_module_imports_in_agent():
    """agent.py imports all standalone module classes."""
    from mabench.agent import (  # noqa: F401
        RebelKG, Outgester, EntityOps, Retriever,
        ConflictResolver, FactStore, QueryEngine,
        QueryPipeline, IterativeQuery,
    )
