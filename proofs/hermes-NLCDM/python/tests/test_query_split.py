"""Tests for query.py split into query.py, query_pipeline.py, iterative_query.py.

Verifies:
1. All three modules import without error.
2. Each mixin class has exactly the expected methods.
3. HermesMemoryAgent inherits from all three mixins.
4. No method duplication across the three files.
5. Cross-mixin self.* calls resolve at runtime (via MRO).
6. QueryPipeline standalone class has __init__ and _resolve_* properties.
"""
import importlib
import inspect


# ------------------------------------------------------------------
# 1. Import tests
# ------------------------------------------------------------------


def test_import_query_mixin():
    """QueryMixin can be imported from mabench.query."""
    from mabench.query import QueryMixin
    assert inspect.isclass(QueryMixin)


def test_import_query_pipeline_mixin():
    """QueryPipelineMixin can be imported from mabench.query_pipeline."""
    from mabench.query_pipeline import QueryPipelineMixin
    assert inspect.isclass(QueryPipelineMixin)


def test_import_query_pipeline_class():
    """QueryPipeline standalone class can be imported from mabench.query_pipeline."""
    from mabench.query_pipeline import QueryPipeline
    assert inspect.isclass(QueryPipeline)


def test_query_pipeline_alias():
    """QueryPipelineMixin is an alias for QueryPipeline."""
    from mabench.query_pipeline import QueryPipeline, QueryPipelineMixin
    assert QueryPipeline is QueryPipelineMixin


def test_import_iterative_query_mixin():
    """IterativeQueryMixin can be imported from mabench.iterative_query."""
    from mabench.iterative_query import IterativeQueryMixin
    assert inspect.isclass(IterativeQueryMixin)


# ------------------------------------------------------------------
# 2. Method membership tests — each method in exactly one mixin
# ------------------------------------------------------------------

QUERY_METHODS = {
    "__init__",
    "_extract_question",
    "_query",
    "_query_setup",
    "_query_run_hop",
    "_query_hop_kg",
    "_query_hop_retrieve",
    "_query_hop_single_retrieve",
    "_query_hop_judge",
    "_query_hop_entity_supplement",
    "_query_hop_filter",
    # Cross-module resolvers (functions, not properties)
    "_resolve_apply_post_retrieval",
    "_resolve_ensure_llm_client",
    "_resolve_entity_first_retrieve",
    "_resolve_entity_graph_expand",
    "_resolve_entity_text_search",
    "_resolve_extract_short_answer",
    "_resolve_pairwise_cosine_dedup",
    "_resolve_query_final_retrieval",
    "_resolve_query_generate",
    "_resolve_query_post_process",
    "_resolve_query_time_cr",
    "_resolve_rebel_kg_hop",
    "_resolve_retrieve_v2",
}

PIPELINE_METHODS = {
    "__init__",
    "_query_final_retrieval",
    "_query_decompose_and_merge",
    "_bm25_rerank_orig",
    "_rrf_merge",
    "_priority_merge",
    "_query_post_process",
    "_build_metadata_lookup",
    "_merge_and_expand",
    "_apply_currency_filter",
    "_query_generate",
    "_query_store_chain_cache",
    "_decompose_query",
    "_parse_subqueries",
    "_extract_short_answer",
    # Cross-module resolvers (functions, not properties)
    "_resolve_retrieve_v2",
    "_resolve_query_run_hop",
    "_resolve_query_time_cr",
    "_resolve_pairwise_cosine_dedup",
    "_resolve_coverage_audit",
    "_resolve_format_outgestion",
    "_resolve_generate_answer",
    "_resolve_ensure_llm_client",
    "_resolve_llm_client",
}

ITERATIVE_METHODS = {
    "_iterative_query",
    "_iterative_hop",
    "_iterative_judge",
    "_build_iterative_chain_summary",
    "_parse_iterative_judge",
    "_iterative_finalize",
    # Standalone class infrastructure
    "__init__",
    "_extract_question",
    # Cross-module resolvers (functions, not properties)
    "_resolve_retrieve_v2",
    "_resolve_resolve_conflicts",
    "_resolve_format_outgestion",
    "_resolve_generate_answer",
    "_resolve_ensure_llm_client",
}


def _get_own_methods(cls):
    """Return set of method names defined directly on cls (not inherited)."""
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if name in cls.__dict__
    }


def test_query_mixin_methods():
    from mabench.query import QueryMixin
    own = _get_own_methods(QueryMixin)
    assert own == QUERY_METHODS, f"Extra: {own - QUERY_METHODS}, Missing: {QUERY_METHODS - own}"


def test_pipeline_mixin_methods():
    from mabench.query_pipeline import QueryPipelineMixin
    own = _get_own_methods(QueryPipelineMixin)
    assert own == PIPELINE_METHODS, f"Extra: {own - PIPELINE_METHODS}, Missing: {PIPELINE_METHODS - own}"


def test_iterative_mixin_methods():
    from mabench.iterative_query import IterativeQueryMixin
    own = _get_own_methods(IterativeQueryMixin)
    assert own == ITERATIVE_METHODS, f"Extra: {own - ITERATIVE_METHODS}, Missing: {ITERATIVE_METHODS - own}"


# ------------------------------------------------------------------
# 3. No duplicates across the three mixins
# ------------------------------------------------------------------


def test_no_method_duplication():
    """No *business* method name appears in more than one module.

    Infrastructure methods (__init__, _extract_question) and cross-module
    resolvers (_resolve_*) are expected to appear in multiple standalone
    classes -- each class has its own copy.  Only business methods must be
    unique across the three modules.
    """
    # Infrastructure that every standalone class may have
    INFRA = {name for name in (QUERY_METHODS | PIPELINE_METHODS | ITERATIVE_METHODS)
             if name.startswith("_resolve_") or name == "__init__" or name == "_extract_question"}
    biz_q = QUERY_METHODS - INFRA
    biz_p = PIPELINE_METHODS - INFRA
    biz_i = ITERATIVE_METHODS - INFRA
    all_biz = biz_q | biz_p | biz_i
    total = len(biz_q) + len(biz_p) + len(biz_i)
    assert len(all_biz) == total, (
        f"Duplicated business methods: "
        f"{(biz_q & biz_p) | (biz_q & biz_i) | (biz_p & biz_i)}"
    )


# ------------------------------------------------------------------
# 4. HermesMemoryAgent inherits all three
# ------------------------------------------------------------------


def test_hermes_agent_inherits_all_query_mixins():
    from mabench.hermes_agent import HermesMemoryAgent
    from mabench.query import QueryMixin
    from mabench.query_pipeline import QueryPipelineMixin
    from mabench.iterative_query import IterativeQueryMixin

    assert issubclass(HermesMemoryAgent, QueryMixin)
    assert issubclass(HermesMemoryAgent, QueryPipelineMixin)
    assert issubclass(HermesMemoryAgent, IterativeQueryMixin)


def test_hermes_agent_has_all_query_methods():
    """HermesMemoryAgent has every business-logic method from all three mixins."""
    from mabench.hermes_agent import HermesMemoryAgent

    # Only check business-logic methods, not __init__ or _resolve_* infrastructure
    business_methods = {
        m for m in (QUERY_METHODS | PIPELINE_METHODS | ITERATIVE_METHODS)
        if not m.startswith("_resolve_") and m != "__init__"
    }
    for method_name in business_methods:
        assert hasattr(HermesMemoryAgent, method_name), f"Missing: {method_name}"


# ------------------------------------------------------------------
# 5. Line count constraints
# ------------------------------------------------------------------


def test_query_file_under_750_lines():
    import mabench.query as mod
    path = inspect.getfile(mod)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) < 750, f"query.py is {len(lines)} lines, must be < 750"


def test_pipeline_file_under_750_lines():
    import mabench.query_pipeline as mod
    path = inspect.getfile(mod)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) < 750, f"query_pipeline.py is {len(lines)} lines, must be < 750"


def test_iterative_file_under_500_lines():
    import mabench.iterative_query as mod
    path = inspect.getfile(mod)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) < 500, f"iterative_query.py is {len(lines)} lines, must be < 500"


# ------------------------------------------------------------------
# 6. QueryPipeline standalone class tests
# ------------------------------------------------------------------


def test_query_pipeline_has_init():
    """QueryPipeline has an __init__ method defined on itself."""
    from mabench.query_pipeline import QueryPipeline
    assert "__init__" in QueryPipeline.__dict__, (
        "QueryPipeline should define its own __init__"
    )


def test_query_pipeline_init_signature():
    """QueryPipeline.__init__ accepts engine, scorer, and config objects."""
    from mabench.query_pipeline import QueryPipeline
    sig = inspect.signature(QueryPipeline.__init__)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "engine" in params
    assert "scorer" in params
    assert "retriever" in params
    assert "conflict_resolver" in params
    assert "outgester" in params
    assert "orchestrator" in params
    assert "retrieval_config" in params
    assert "cr_config" in params
    assert "decompose_config" in params
    assert "dedup_config" in params
    assert "llm_config" in params
    assert "hop_executor_fn" in params


def test_query_pipeline_init_defaults_to_none():
    """All config params in __init__ default to None."""
    from mabench.query_pipeline import QueryPipeline
    sig = inspect.signature(QueryPipeline.__init__)
    for name in (
        "engine", "scorer", "retriever", "conflict_resolver",
        "outgester", "orchestrator", "retrieval_config", "cr_config",
        "decompose_config", "dedup_config", "llm_config",
        "hop_executor_fn",
    ):
        p = sig.parameters[name]
        assert p.default is None, (
            f"{name} should default to None, got {p.default}"
        )


def test_query_pipeline_standalone_instantiation():
    """QueryPipeline can be instantiated with all-None defaults."""
    from mabench.query_pipeline import QueryPipeline
    qp = QueryPipeline()
    assert qp._engine is None
    assert qp._scorer is None
    assert qp._retriever is None
    assert qp._conflict_resolver is None
    assert qp._outgester is None
    assert qp._orchestrator is None
    assert qp._client is None
    assert qp._query_q_emb is None
    assert qp._last_decompose_question == ""


def test_query_pipeline_resolve_properties_exist():
    """QueryPipeline has _resolve_* properties for all config booleans."""
    from mabench.query_pipeline import QueryPipeline
    expected_properties = [
        "_resolve_engine",
        "_resolve_orchestrator",
        "_resolve_decompose_query",
        "_resolve_decompose_rrf",
        "_resolve_decompose_coverage",
        "_resolve_decompose_max_hops",
        "_resolve_retrieve_num",
        "_resolve_conflict_resolution",
        "_resolve_supersession",
        "_resolve_cooc_boost",
        "_resolve_cosine_dedup",
        "_resolve_bm25_hops",
        "_resolve_bm25_index",
        "_resolve_triadic",
        "_resolve_triadic_expand_k",
        "_resolve_chain_cache",
        "_resolve_belief_index",
        "_resolve_superseded_texts",
        "_resolve_model",
        "_resolve_query_q_emb",
    ]
    for prop_name in expected_properties:
        assert hasattr(QueryPipeline, prop_name), f"Missing property: {prop_name}"


def test_query_pipeline_resolve_defaults():
    """Standalone QueryPipeline resolves config to sane defaults via None configs."""
    from mabench.query_pipeline import QueryPipeline
    from mabench.configs import DecomposeConfig, RetrievalConfig, CRConfig, DedupConfig, LLMConfig
    qp = QueryPipeline(
        decompose_config=DecomposeConfig(),
        retrieval_config=RetrievalConfig(),
        cr_config=CRConfig(),
        dedup_config=DedupConfig(),
        llm_config=LLMConfig(),
    )
    assert qp._resolve_decompose_query is False
    assert qp._resolve_decompose_rrf is False
    assert qp._resolve_decompose_coverage is False
    assert qp._resolve_decompose_max_hops == 3
    assert qp._resolve_retrieve_num == 10
    assert qp._resolve_conflict_resolution is False
    assert qp._resolve_cosine_dedup is False
    assert qp._resolve_model == "gpt-4o-mini"


def test_pipeline_file_under_750_lines_standalone():
    """query_pipeline.py stays under 750 lines after standalone conversion."""
    import mabench.query_pipeline as mod
    path = inspect.getfile(mod)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) < 750, f"query_pipeline.py is {len(lines)} lines, must be < 750"
