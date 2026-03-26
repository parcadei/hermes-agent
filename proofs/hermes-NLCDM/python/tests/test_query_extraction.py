"""Discriminative tests for QueryEngine extraction from HermesMemoryAgent.

Validates that:
1. query.py exports QueryEngine and QueryMixin (backward-compat alias)
2. QueryEngine has all 11 methods defined in query.py
3. QueryEngine has __init__ for standalone usage
4. HermesMemoryAgent inherits from QueryMixin
5. Methods are defined on the class, not directly on the agent class
6. Method signatures are preserved exactly
7. _extract_question is a @staticmethod
8. _query decomposition helpers exist and are <50 lines each
9. MRO is correct (no diamond conflicts)
10. _resolve_* properties route to config or mixin fallback
11. Methods belonging to OTHER classes are NOT checked here

Methods on QueryEngine (query.py):
- _extract_question (static)
- _query
- _query_setup
- _query_run_hop
- _query_hop_kg
- _query_hop_retrieve
- _query_hop_single_retrieve
- _query_hop_judge
- _query_hop_entity_supplement
- _query_hop_filter

Methods NOT on QueryEngine (they live elsewhere):
- _decompose_query -> QueryPipelineMixin (query_pipeline.py)
- _extract_short_answer -> QueryPipelineMixin (query_pipeline.py)
- _query_final_retrieval -> QueryPipelineMixin (query_pipeline.py)
- _query_post_process -> QueryPipelineMixin (query_pipeline.py)
- _query_generate -> QueryPipelineMixin (query_pipeline.py)
- _iterative_query -> IterativeQueryMixin (iterative_query.py)
- _iterative_hop -> IterativeQueryMixin (iterative_query.py)
- _iterative_finalize -> IterativeQueryMixin (iterative_query.py)

All tests run on CPU. No GPU, no external models, no LLM calls.
"""

import ast
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MABENCH_DIR = Path(__file__).resolve().parent.parent / "mabench"


# --- Module existence ---


class TestQueryModuleExists:
    """Verify query.py exists and exports expected symbols."""

    def test_module_importable(self):
        """query module must be importable."""
        import mabench.query  # noqa: F401

    def test_exports_query_engine_class(self):
        """QueryEngine must be a class."""
        from mabench.query import QueryEngine

        assert isinstance(QueryEngine, type)

    def test_exports_mixin_alias(self):
        """QueryMixin must be a backward-compat alias for QueryEngine."""
        from mabench.query import QueryEngine, QueryMixin

        assert QueryMixin is QueryEngine

    def test_module_has_docstring(self):
        """query module must have a docstring."""
        import mabench.query as mod

        assert mod.__doc__ is not None
        assert len(mod.__doc__.strip()) > 0


# --- __init__ for standalone usage ---


class TestQueryEngineInit:
    """QueryEngine must have __init__ for standalone construction."""

    def test_has_init(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "__init__")
        assert QueryEngine.__init__ is not object.__init__

    def test_init_accepts_engine_and_scorer(self):
        """__init__ must accept at least engine and scorer positional args."""
        from mabench.query import QueryEngine

        sig = inspect.signature(QueryEngine.__init__)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "engine" in params
        assert "scorer" in params

    def test_init_accepts_config_objects(self):
        """__init__ must accept config objects for standalone mode."""
        from mabench.query import QueryEngine

        sig = inspect.signature(QueryEngine.__init__)
        params = list(sig.parameters.keys())
        # Must accept config objects for various subsystems
        assert "retrieval_config" in params or "retriever" in params
        assert "cr_config" in params or "conflict_resolver" in params


# --- Method existence on QueryEngine ---


class TestQueryEngineMethods:
    """All 10 methods from query.py must live on QueryEngine."""

    def test_has_extract_question(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_extract_question")
        assert callable(QueryEngine._extract_question)

    def test_has_query(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query")
        assert callable(QueryEngine._query)

    def test_has_query_setup(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_setup")
        assert callable(QueryEngine._query_setup)

    def test_has_query_run_hop(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_run_hop")
        assert callable(QueryEngine._query_run_hop)

    def test_has_query_hop_kg(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_hop_kg")
        assert callable(QueryEngine._query_hop_kg)

    def test_has_query_hop_retrieve(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_hop_retrieve")
        assert callable(QueryEngine._query_hop_retrieve)

    def test_has_query_hop_single_retrieve(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_hop_single_retrieve")
        assert callable(QueryEngine._query_hop_single_retrieve)

    def test_has_query_hop_judge(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_hop_judge")
        assert callable(QueryEngine._query_hop_judge)

    def test_has_query_hop_entity_supplement(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_hop_entity_supplement")
        assert callable(QueryEngine._query_hop_entity_supplement)

    def test_has_query_hop_filter(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_hop_filter")
        assert callable(QueryEngine._query_hop_filter)


# --- Methods NOT on QueryEngine (belong to other classes) ---


class TestMethodsNotOnQueryEngine:
    """Methods from other classes must NOT be on QueryEngine."""

    def test_decompose_query_not_on_query_engine(self):
        """_decompose_query belongs to QueryPipelineMixin."""
        from mabench.query import QueryEngine

        assert "_decompose_query" not in QueryEngine.__dict__

    def test_extract_short_answer_not_on_query_engine(self):
        """_extract_short_answer belongs to QueryPipelineMixin."""
        from mabench.query import QueryEngine

        assert "_extract_short_answer" not in QueryEngine.__dict__

    def test_iterative_query_not_on_query_engine(self):
        """_iterative_query belongs to IterativeQueryMixin."""
        from mabench.query import QueryEngine

        assert "_iterative_query" not in QueryEngine.__dict__

    def test_query_final_retrieval_not_on_query_engine(self):
        """_query_final_retrieval belongs to QueryPipelineMixin."""
        from mabench.query import QueryEngine

        assert "_query_final_retrieval" not in QueryEngine.__dict__

    def test_query_post_process_not_on_query_engine(self):
        """_query_post_process belongs to QueryPipelineMixin."""
        from mabench.query import QueryEngine

        assert "_query_post_process" not in QueryEngine.__dict__

    def test_query_generate_not_on_query_engine(self):
        """_query_generate belongs to QueryPipelineMixin."""
        from mabench.query import QueryEngine

        assert "_query_generate" not in QueryEngine.__dict__


# --- @staticmethod check ---


class TestStaticMethodPreserved:
    """_extract_question must remain a @staticmethod."""

    def test_extract_question_is_staticmethod(self):
        from mabench.query import QueryEngine

        # In the class dict, staticmethods are stored as staticmethod objects
        raw = QueryEngine.__dict__["_extract_question"]
        assert isinstance(raw, staticmethod), (
            f"_extract_question should be a staticmethod, got {type(raw)}"
        )

    def test_extract_question_callable_on_class(self):
        """Can be called directly on the class (no self)."""
        from mabench.query import QueryEngine

        # Note: _extract_question looks for "Question:" ... "\nAnswer:"
        prompt = (
            "Pretend you are... Question: What color?\nAnswer:"
        )
        result = QueryEngine._extract_question(prompt)
        assert result == "What color?"


# --- Method signatures ---


class TestMethodSignatures:
    """Method signatures must be preserved exactly."""

    def test_extract_question_signature(self):
        from mabench.query import QueryEngine

        sig = inspect.signature(QueryEngine._extract_question)
        params = list(sig.parameters.keys())
        assert params == ["prompt"], f"Expected ['prompt'], got {params}"

    def test_query_signature(self):
        from mabench.query import QueryEngine

        sig = inspect.signature(QueryEngine._query)
        params = list(sig.parameters.keys())
        assert params == ["self", "question"], f"Expected ['self', 'question'], got {params}"

    def test_query_setup_signature(self):
        from mabench.query import QueryEngine

        sig = inspect.signature(QueryEngine._query_setup)
        params = list(sig.parameters.keys())
        assert params == ["self", "bare_question", "query_start"]

    def test_query_run_hop_signature(self):
        from mabench.query import QueryEngine

        sig = inspect.signature(QueryEngine._query_run_hop)
        params = list(sig.parameters.keys())
        expected = [
            "self", "sq_template", "hop_idx", "num_subqueries",
            "answers", "evidence_chain", "sq_chunks", "sq_chunks_raw",
        ]
        assert params == expected, f"Expected {expected}, got {params}"


# --- Decomposition helpers for _query ---


class TestQueryDecompositionHelpers:
    """_query must call pipeline helpers and be decomposed into sub-methods."""

    def test_has_query_setup(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_setup")
        assert callable(QueryEngine._query_setup)

    def test_has_query_run_hop(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_run_hop")
        assert callable(QueryEngine._query_run_hop)

    def test_has_query_hop_kg(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_hop_kg")
        assert callable(QueryEngine._query_hop_kg)

    def test_has_query_hop_retrieve(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_query_hop_retrieve")
        assert callable(QueryEngine._query_hop_retrieve)


# --- Line count checks ---


class TestLineCounts:
    """Every function on QueryEngine must be <50 lines of code."""

    def _count_code_lines(self, method):
        """Count non-blank, non-comment, non-docstring lines."""
        source = inspect.getsource(method)
        lines = source.split("\n")
        code_lines = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    continue
                elif stripped.count('"""') == 2 or stripped.count("'''") == 2:
                    continue  # single-line docstring
                else:
                    in_docstring = True
                    continue
            if in_docstring:
                continue
            if stripped and not stripped.startswith("#"):
                code_lines.append(line)
        return code_lines

    def test_all_methods_under_50_lines(self):
        from mabench.query import QueryEngine

        for name, method in inspect.getmembers(QueryEngine, predicate=inspect.isfunction):
            if name.startswith("_"):
                code_lines = self._count_code_lines(method)
                assert len(code_lines) <= 50, (
                    f"QueryEngine.{name} has {len(code_lines)} code lines (max 50):\n"
                    + "\n".join(code_lines[:55])
                )

    def test_query_method_under_50_lines(self):
        """Specifically check _query is decomposed (not a monolith)."""
        from mabench.query import QueryEngine

        code_lines = self._count_code_lines(QueryEngine._query)
        assert len(code_lines) <= 50, (
            f"_query has {len(code_lines)} code lines. Must be decomposed into helpers."
        )


# --- Inheritance and MRO ---


class TestHermesAgentInheritance:
    """HermesMemoryAgent must inherit from QueryMixin."""

    def test_mro_includes_query_mixin(self):
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.query import QueryMixin

        assert QueryMixin in HermesMemoryAgent.__mro__

    def test_agent_has_query(self):
        from mabench.hermes_agent import HermesMemoryAgent

        assert hasattr(HermesMemoryAgent, "_query")

    def test_agent_has_extract_question(self):
        from mabench.hermes_agent import HermesMemoryAgent

        assert hasattr(HermesMemoryAgent, "_extract_question")

    def test_agent_has_query_run_hop(self):
        from mabench.hermes_agent import HermesMemoryAgent

        assert hasattr(HermesMemoryAgent, "_query_run_hop")

    def test_agent_has_query_hop_filter(self):
        from mabench.hermes_agent import HermesMemoryAgent

        assert hasattr(HermesMemoryAgent, "_query_hop_filter")


class TestNoMethodsLeftOnAgent:
    """Query methods must NOT be defined directly on HermesMemoryAgent."""

    def _get_agent_method_names(self):
        source = (MABENCH_DIR / "hermes_agent.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "HermesMemoryAgent":
                return [
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
        return []

    def test_query_not_on_agent(self):
        method_names = self._get_agent_method_names()
        assert "_query" not in method_names, "_query should be on QueryEngine, not HermesMemoryAgent"

    def test_extract_question_not_on_agent(self):
        method_names = self._get_agent_method_names()
        assert "_extract_question" not in method_names

    def test_query_hop_filter_not_on_agent(self):
        method_names = self._get_agent_method_names()
        assert "_query_hop_filter" not in method_names


class TestImportInHermesAgent:
    """hermes_agent.py must import QueryMixin from mabench.query."""

    def test_import_statement(self):
        source = (MABENCH_DIR / "hermes_agent.py").read_text()
        assert "from mabench.query import QueryMixin" in source

    def test_query_mixin_in_bases(self):
        source = (MABENCH_DIR / "hermes_agent.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "HermesMemoryAgent":
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                assert "QueryMixin" in base_names, (
                    f"HermesMemoryAgent bases: {base_names}"
                )


# --- _extract_question behavioral test ---


class TestExtractQuestionBehavior:
    """Test _extract_question behavior directly (no agent init needed)."""

    def test_extracts_from_mabench_prompt(self):
        from mabench.query import QueryEngine

        prompt = (
            "Pretend you are a knowledge management system. "
            "You can remember things and answer questions. "
            "Question: What is the capital of France?\nAnswer:"
        )
        assert QueryEngine._extract_question(prompt) == "What is the capital of France?"

    def test_fallback_on_no_pattern(self):
        from mabench.query import QueryEngine

        prompt = "Just a random string with no question format"
        assert QueryEngine._extract_question(prompt) == prompt

    def test_extracts_last_question(self):
        from mabench.query import QueryEngine

        prompt = (
            "Question: first?\nAnswer: yes\n"
            "Question: second?\nAnswer:"
        )
        assert QueryEngine._extract_question(prompt) == "second?"


# --- _resolve_* properties ---


class TestResolveProperties:
    """QueryEngine must have _resolve_* properties for dual-mode operation."""

    def test_has_resolve_engine(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_resolve_engine")

    def test_has_resolve_decompose_query(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_resolve_decompose_query")

    def test_has_resolve_retrieve_num(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_resolve_retrieve_num")

    def test_has_resolve_conflict_resolution(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_resolve_conflict_resolution")

    def test_has_resolve_cr_per_hop(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_resolve_cr_per_hop")

    def test_has_resolve_cosine_dedup(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_resolve_cosine_dedup")

    def test_has_resolve_entity_retrieval(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_resolve_entity_retrieval")

    def test_has_resolve_model(self):
        from mabench.query import QueryEngine

        assert hasattr(QueryEngine, "_resolve_model")


# --- MRO ordering ---


class TestMROOrdering:
    """QueryMixin must appear in MRO at the right position."""

    def test_query_mixin_once_in_mro(self):
        from mabench.hermes_agent import HermesMemoryAgent

        mro_names = [c.__name__ for c in HermesMemoryAgent.__mro__]
        # QueryMixin is an alias for QueryEngine, so check both possible names
        # The class name in MRO will be QueryEngine (the actual class name)
        query_count = sum(
            1 for name in mro_names if name in ("QueryMixin", "QueryEngine")
        )
        assert query_count == 1, (
            f"Expected exactly 1 QueryEngine/QueryMixin in MRO, got {query_count}: {mro_names}"
        )

    def test_query_engine_before_object(self):
        from mabench.hermes_agent import HermesMemoryAgent

        mro_names = [c.__name__ for c in HermesMemoryAgent.__mro__]
        # Find the QueryEngine entry (might be listed as QueryEngine since that's the real name)
        query_idx = None
        for i, name in enumerate(mro_names):
            if name in ("QueryMixin", "QueryEngine"):
                query_idx = i
                break
        assert query_idx is not None, f"QueryEngine not in MRO: {mro_names}"
        assert query_idx < mro_names.index("object")


class TestNoCircularImport:
    """Importing query does not cause circular imports."""

    def test_import_query_module(self):
        import importlib
        import mabench.query

        importlib.reload(mabench.query)

    def test_import_hermes_agent_after_query(self):
        import importlib
        import mabench.query  # noqa: F401
        import mabench.hermes_agent

        importlib.reload(mabench.hermes_agent)
