"""Discriminative tests for agentic retrieval module."""

from __future__ import annotations

import pytest

from mabench.agentic_retrieval import (
    AgenticRetriever,
    RetrievalTrace,
    ToolCall,
    ToolKit,
    parse_react_output,
)


# ---------------------------------------------------------------------------
# parse_react_output
# ---------------------------------------------------------------------------

class TestParseReactOutput:
    def test_thought_and_action(self):
        text = 'Thought: I need to find the airline.\nAction: search_memory("airline")'
        thought, tool, args = parse_react_output(text)
        assert thought == "I need to find the airline."
        assert tool == "search_memory"
        assert args == "airline"

    def test_single_quoted_args(self):
        text = "Thought: Looking for Delta.\nAction: search_entities('Delta')"
        thought, tool, args = parse_react_output(text)
        assert tool == "search_entities"
        assert args == "Delta"

    def test_no_action(self):
        text = "I think the answer is 42."
        thought, tool, args = parse_react_output(text)
        assert tool == ""
        assert args == ""

    def test_answer_action(self):
        text = 'Thought: I have enough info.\nAction: answer("The user flies Delta.")'
        thought, tool, args = parse_react_output(text)
        assert tool == "answer"
        assert args == "The user flies Delta."

    def test_numeric_args(self):
        text = "Thought: Read turn 42.\nAction: read_turn(42)"
        thought, tool, args = parse_react_output(text)
        assert tool == "read_turn"
        assert args == "42"

    def test_multiword_thought(self):
        text = (
            "Thought: The question asks about a pet named Max. "
            "I should search for Max in entities.\n"
            'Action: search_entities("Max pet")'
        )
        thought, tool, args = parse_react_output(text)
        assert "Max" in thought
        assert tool == "search_entities"
        assert args == "Max pet"


# ---------------------------------------------------------------------------
# ToolKit — search_memory
# ---------------------------------------------------------------------------

class FakeEntry:
    def __init__(self, text, role="user"):
        self.text = text
        self.role = role


class TestToolKitSearchMemory:
    def _make_toolkit(self, entries):
        class FakeBM25:
            def __init__(self, entries):
                self._entries = entries
            def query(self, q, top_k=10):
                # Simple substring match scoring
                results = []
                for i, e in enumerate(self._entries):
                    if q.lower() in e.text.lower():
                        results.append((i, 1.0))
                return results[:top_k]
        return ToolKit(
            bm25_index=FakeBM25(entries),
            memory_store=entries,
        )

    def test_finds_matching_turn(self):
        entries = [
            FakeEntry("I love flying Delta airlines"),
            FakeEntry("My dog Max is a golden retriever"),
        ]
        tk = self._make_toolkit(entries)
        results = tk.execute("search_memory", "Delta")
        assert len(results) == 1
        assert results[0]["turn_id"] == 0
        assert "Delta" in results[0]["text"]

    def test_no_results(self):
        entries = [FakeEntry("Hello world")]
        tk = self._make_toolkit(entries)
        results = tk.execute("search_memory", "nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# ToolKit — search_entities
# ---------------------------------------------------------------------------

class TestToolKitSearchEntities:
    def _make_toolkit(self):
        class FakeEntityBM25:
            _n = 2
            def query(self, q, top_k=10):
                data = {
                    "delta": ("delta", {0, 5}, 2.5),
                    "max": ("max", {1, 3}, 1.8),
                }
                results = []
                for key, (canon, turns, score) in data.items():
                    if q.lower() in key:
                        results.append((canon, turns, score))
                return results[:top_k]

        class FakeEntityGraph:
            class FakeNode:
                def __init__(self, facets, turns):
                    self.facets = facets
                    self.mention_turns = turns
            entities = {
                "delta": FakeNode({"what": "airline", "where": ""}, {0, 5}),
                "max": FakeNode({"what": "dog", "status": "golden retriever"}, {1, 3}),
            }

        return ToolKit(
            entity_bm25=FakeEntityBM25(),
            entity_graph=FakeEntityGraph(),
        )

    def test_finds_entity_by_name(self):
        tk = self._make_toolkit()
        results = tk.execute("search_entities", "delta")
        assert len(results) == 1
        assert results[0]["entity"] == "delta"
        assert results[0]["facets"]["what"] == "airline"

    def test_returns_turns(self):
        tk = self._make_toolkit()
        results = tk.execute("search_entities", "max")
        assert 1 in results[0]["turns"]
        assert 3 in results[0]["turns"]


# ---------------------------------------------------------------------------
# ToolKit — get_related
# ---------------------------------------------------------------------------

class TestToolKitGetRelated:
    def _make_toolkit(self):
        class FakeNode:
            def __init__(self, facets, turns):
                self.facets = facets
                self.mention_turns = turns

        class FakeEntityGraph:
            entities = {
                "delta": FakeNode({"what": "airline"}, {0, 5, 8}),
                "skymiles": FakeNode({"what": "loyalty program"}, {5, 8}),
                "max": FakeNode({"what": "dog"}, {1, 3}),
            }

        return ToolKit(entity_graph=FakeEntityGraph())

    def test_finds_co_mentions(self):
        tk = self._make_toolkit()
        result = tk.execute("get_related", "delta")
        assert result["entity"] == "delta"
        # skymiles shares turns 5 and 8 with delta
        co = [c for c in result["co_mentions"] if c["entity"] == "skymiles"]
        assert len(co) == 1
        assert 5 in co[0]["shared_turns"]

    def test_entity_not_found(self):
        tk = self._make_toolkit()
        result = tk.execute("get_related", "nonexistent")
        assert "error" in result

    def test_fuzzy_match(self):
        tk = self._make_toolkit()
        result = tk.execute("get_related", "Delta")  # uppercase
        assert result["entity"] == "delta"


# ---------------------------------------------------------------------------
# ToolKit — read_turn
# ---------------------------------------------------------------------------

class TestToolKitReadTurn:
    def test_reads_turn(self):
        entries = [FakeEntry("Turn zero"), FakeEntry("Turn one", "assistant")]
        tk = ToolKit(memory_store=entries)
        result = tk.execute("read_turn", "1")
        assert result["text"] == "Turn one"
        assert result["role"] == "assistant"

    def test_out_of_range(self):
        tk = ToolKit(memory_store=[FakeEntry("only one")])
        result = tk.execute("read_turn", "5")
        assert "error" in result

    def test_invalid_id(self):
        tk = ToolKit(memory_store=[FakeEntry("x")])
        result = tk.execute("read_turn", "abc")
        assert "error" in result


# ---------------------------------------------------------------------------
# ToolKit — answer (terminal)
# ---------------------------------------------------------------------------

class TestToolKitAnswer:
    def test_returns_terminal(self):
        tk = ToolKit()
        result = tk.execute("answer", "The user flies Delta.")
        assert result["terminal"] is True
        assert result["evidence"] == "The user flies Delta."


# ---------------------------------------------------------------------------
# ToolKit — unknown tool
# ---------------------------------------------------------------------------

class TestToolKitUnknown:
    def test_unknown_tool(self):
        tk = ToolKit()
        result = tk.execute("unknown_tool", "args")
        assert "error" in result


# ---------------------------------------------------------------------------
# RetrievalTrace serialization
# ---------------------------------------------------------------------------

class TestRetrievalTrace:
    def test_to_dict(self):
        trace = RetrievalTrace(question="test?")
        trace.steps.append(ToolCall(
            tool="search_memory", args="test", result=[{"text": "hi"}],
            thought="testing", elapsed_ms=5.0,
        ))
        trace.evidence = "found it"
        trace.terminated_by = "answer"
        trace.total_ms = 100.0

        d = trace.to_dict()
        assert d["question"] == "test?"
        assert d["n_steps"] == 1
        assert d["terminated_by"] == "answer"
        assert d["steps"][0]["tool"] == "search_memory"
