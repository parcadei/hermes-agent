"""Tests for query.py split into query.py, query_pipeline.py, iterative_query.py.

Verifies:
1. All three modules import without error.
2. Each mixin class has exactly the expected methods.
3. HermesMemoryAgent inherits from all three mixins.
4. No method duplication across the three files.
5. Cross-mixin self.* calls resolve at runtime (via MRO).
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


def test_import_iterative_query_mixin():
    """IterativeQueryMixin can be imported from mabench.iterative_query."""
    from mabench.iterative_query import IterativeQueryMixin
    assert inspect.isclass(IterativeQueryMixin)


# ------------------------------------------------------------------
# 2. Method membership tests — each method in exactly one mixin
# ------------------------------------------------------------------

QUERY_METHODS = {
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
}

PIPELINE_METHODS = {
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
}

ITERATIVE_METHODS = {
    "_iterative_query",
    "_iterative_hop",
    "_iterative_judge",
    "_build_iterative_chain_summary",
    "_parse_iterative_judge",
    "_iterative_finalize",
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
    """No method name appears in more than one mixin."""
    all_methods = QUERY_METHODS | PIPELINE_METHODS | ITERATIVE_METHODS
    total = len(QUERY_METHODS) + len(PIPELINE_METHODS) + len(ITERATIVE_METHODS)
    assert len(all_methods) == total, (
        f"Duplicated methods: "
        f"{(QUERY_METHODS & PIPELINE_METHODS) | (QUERY_METHODS & ITERATIVE_METHODS) | (PIPELINE_METHODS & ITERATIVE_METHODS)}"
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
    """HermesMemoryAgent has every method from all three mixins."""
    from mabench.hermes_agent import HermesMemoryAgent

    all_methods = QUERY_METHODS | PIPELINE_METHODS | ITERATIVE_METHODS
    for method_name in all_methods:
        assert hasattr(HermesMemoryAgent, method_name), f"Missing: {method_name}"


# ------------------------------------------------------------------
# 5. Line count constraints
# ------------------------------------------------------------------


def test_query_file_under_500_lines():
    import mabench.query as mod
    path = inspect.getfile(mod)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) < 500, f"query.py is {len(lines)} lines, must be < 500"


def test_pipeline_file_under_500_lines():
    import mabench.query_pipeline as mod
    path = inspect.getfile(mod)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) < 500, f"query_pipeline.py is {len(lines)} lines, must be < 500"


def test_iterative_file_under_500_lines():
    import mabench.iterative_query as mod
    path = inspect.getfile(mod)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) < 500, f"iterative_query.py is {len(lines)} lines, must be < 500"
