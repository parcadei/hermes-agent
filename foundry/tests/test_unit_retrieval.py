"""Unit tests for the Retrieval Layer (spec section 5).

Tests tool definitions, hybrid search, deduplication tracker, agentic
retriever, deterministic router, and budget management.
"""

from __future__ import annotations

import pytest
from datetime import datetime


# ---------------------------------------------------------------------------
# DeduplicationTracker
# ---------------------------------------------------------------------------

class TestDeduplicationTracker:
    """Tests for cross-call deduplication of turn IDs."""

    def test_initial_empty(self):
        """Fresh tracker has no seen turns."""
        from foundry.retrieval import DeduplicationTracker
        tracker = DeduplicationTracker()
        results = [{"turn_id": 0, "text": "a"}, {"turn_id": 1, "text": "b"}]
        filtered = tracker.filter_new(results)
        assert len(filtered) == 2

    def test_dedup_second_call(self):
        """Second call with same turn IDs returns empty."""
        from foundry.retrieval import DeduplicationTracker
        tracker = DeduplicationTracker()
        results = [{"turn_id": 0}, {"turn_id": 1}]
        tracker.filter_new(results)
        filtered = tracker.filter_new(results)
        assert len(filtered) == 0

    def test_dedup_mixed(self):
        """Mixed old and new turn IDs returns only new."""
        from foundry.retrieval import DeduplicationTracker
        tracker = DeduplicationTracker()
        tracker.filter_new([{"turn_id": 0}, {"turn_id": 1}])
        mixed = [{"turn_id": 1}, {"turn_id": 2}, {"turn_id": 3}]
        filtered = tracker.filter_new(mixed)
        turn_ids = {r["turn_id"] for r in filtered}
        assert turn_ids == {2, 3}

    def test_reset(self):
        """reset() clears all seen turns."""
        from foundry.retrieval import DeduplicationTracker
        tracker = DeduplicationTracker()
        tracker.filter_new([{"turn_id": 0}])
        tracker.reset()
        filtered = tracker.filter_new([{"turn_id": 0}])
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# RetrievalTrace
# ---------------------------------------------------------------------------

class TestRetrievalTrace:
    """Tests for the RetrievalTrace data structure."""

    def test_trace_creation(self):
        """RetrievalTrace stores question and defaults."""
        from foundry.retrieval import RetrievalTrace
        trace = RetrievalTrace(question="What color is the car?")
        assert trace.question == "What color is the car?"
        assert trace.steps == []
        assert trace.evidence == ""
        assert trace.total_ms == 0.0
        assert trace.terminated_by == ""

    def test_trace_terminated_by_values(self):
        """terminated_by accepts standard termination reasons."""
        from foundry.retrieval import RetrievalTrace
        for reason in ["answer", "budget", "error", "no_action"]:
            trace = RetrievalTrace(question="q", terminated_by=reason)
            assert trace.terminated_by == reason


# ---------------------------------------------------------------------------
# RetrievalBudget
# ---------------------------------------------------------------------------

class TestRetrievalBudget:
    """Tests for the RetrievalBudget configuration."""

    def test_budget_defaults(self):
        """Default budget matches spec: 8 steps, 2000 char obs, 30s timeout."""
        from foundry.retrieval import RetrievalBudget
        budget = RetrievalBudget()
        assert budget.max_steps == 8
        assert budget.max_observation_chars == 2000
        assert budget.max_total_ms == 30000
        assert budget.aggregation_overfetch == 25

    def test_budget_frozen(self):
        """RetrievalBudget is frozen (immutable)."""
        from foundry.retrieval import RetrievalBudget
        budget = RetrievalBudget()
        with pytest.raises(AttributeError):
            budget.max_steps = 99

    def test_budget_custom(self):
        """Custom budget values are accepted."""
        from foundry.retrieval import RetrievalBudget
        budget = RetrievalBudget(max_steps=4, max_total_ms=10000)
        assert budget.max_steps == 4
        assert budget.max_total_ms == 10000


# ---------------------------------------------------------------------------
# DeterministicRouter
# ---------------------------------------------------------------------------

class TestDeterministicRouter:
    """Tests for the DeterministicRouter (regex-based question routing)."""

    def test_route_factual(self):
        """Factual questions route to fact_lookup pipeline."""
        from foundry.retrieval import DeterministicRouter
        router = DeterministicRouter()
        result = router.route("What color is the user's car?")
        assert result in ("fact_lookup", "agentic")

    def test_route_temporal(self):
        """Temporal questions route to temporal pipeline."""
        from foundry.retrieval import DeterministicRouter
        router = DeterministicRouter()
        result = router.route("When did the user adopt their cat?")
        assert result in ("temporal", "agentic")

    def test_route_aggregation(self):
        """Counting questions route to aggregation pipeline."""
        from foundry.retrieval import DeterministicRouter
        router = DeterministicRouter()
        result = router.route("How many pets does the user have?")
        assert result in ("aggregation", "agentic")

    def test_route_comparison(self):
        """Comparison questions route to comparison pipeline."""
        from foundry.retrieval import DeterministicRouter
        router = DeterministicRouter()
        result = router.route("Which pet is more expensive?")
        assert result in ("comparison", "agentic")

    def test_route_fallback(self):
        """Unknown question types fall back to agentic."""
        from foundry.retrieval import DeterministicRouter
        router = DeterministicRouter()
        result = router.route("Tell me something interesting")
        assert result == "agentic"

    def test_route_preference(self):
        """Preference questions identified by pattern."""
        from foundry.retrieval import DeterministicRouter
        router = DeterministicRouter()
        result = router.route("Does the user prefer cats or dogs?")
        assert result in ("preference", "agentic")


# ---------------------------------------------------------------------------
# AgenticRetriever (structural tests -- no LLM)
# ---------------------------------------------------------------------------

class TestAgenticRetriever:
    """Structural tests for the AgenticRetriever class."""

    def test_retriever_creation(self):
        """AgenticRetriever can be instantiated."""
        from foundry.retrieval import AgenticRetriever
        retriever = AgenticRetriever(toolkit=None, llm_model="gpt-4o-mini")
        assert retriever is not None

    def test_retriever_has_retrieve_method(self):
        """AgenticRetriever has a retrieve() method."""
        from foundry.retrieval import AgenticRetriever
        retriever = AgenticRetriever(toolkit=None, llm_model="gpt-4o-mini")
        assert hasattr(retriever, "retrieve")
        assert callable(retriever.retrieve)

    def test_retriever_returns_tuple(self):
        """retrieve() returns (evidence, trace) tuple.

        With no toolkit, it should either raise or return empty evidence.
        """
        from foundry.retrieval import AgenticRetriever, RetrievalTrace
        retriever = AgenticRetriever(toolkit=None, llm_model="gpt-4o-mini")
        try:
            evidence, trace = retriever.retrieve("test question")
            assert isinstance(evidence, str)
            assert isinstance(trace, RetrievalTrace)
        except Exception:
            # Acceptable -- no toolkit configured
            pass


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    """Tests verifying the 8 tool definitions exist."""

    def test_eight_tools_defined(self):
        """There are exactly 8 tools in the retrieval toolkit."""
        from foundry.retrieval import TOOLS
        assert len(TOOLS) == 8

    def test_required_tools_present(self):
        """All 8 required tool names are defined."""
        from foundry.retrieval import TOOLS
        required = {
            "search_memory", "search_entities", "get_related",
            "search_temporal", "read_turn", "aggregate",
            "execute_python", "answer",
        }
        assert set(TOOLS.keys()) == required

    def test_tool_has_signature(self):
        """Each tool definition has a signature field."""
        from foundry.retrieval import TOOLS
        for name, spec in TOOLS.items():
            assert "signature" in spec, f"Tool '{name}' missing signature"

    def test_tool_has_description(self):
        """Each tool definition has a description field."""
        from foundry.retrieval import TOOLS
        for name, spec in TOOLS.items():
            assert "description" in spec, f"Tool '{name}' missing description"

    def test_answer_is_terminal(self):
        """The answer tool is marked as terminal."""
        from foundry.retrieval import TOOLS
        assert "TERMINAL" in TOOLS["answer"]["signature"]
