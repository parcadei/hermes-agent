"""Unit tests for ChronosAgent tools (no LLM calls).

Tests the deterministic tool implementations:
  * _tool_search_turns (vector search via engine)
  * _tool_search_events (vector search via event index)
  * _tool_grep_turns (substring match on memory store)
  * _tool_grep_events (substring match on event index)
  * _tool_read_turns (fetch text by index)
  * _execute_tool dispatch
  * _fallback_answer helper
  * _summarize_result helper
  * _safe_json_loads helper
  * answer() ReAct loop with FakeLLMClient
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mabench.chronos_agent import (
    CHRONOS_SYSTEM_PROMPT,
    TOOL_SPECS,
    ChronosAgent,
    _fallback_answer,
    _safe_json_loads,
    _summarize_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeScorer:
    """Minimal scorer stub with .embed()."""

    def embed(self, text: str):
        import numpy as np
        return np.ones(4, dtype=float)


class _FakeEngine:
    """Minimal CoupledEngine stub with .query()."""

    def __init__(self, hits: list[dict]):
        self._hits = hits

    def query(self, emb, top_k: int = 10):
        return self._hits[:top_k]


class _FakeEventIndex:
    """Minimal ChronosEventIndex stub with .search() and .grep()."""

    def __init__(
        self,
        search_hits: list[tuple[int, float]] | None = None,
        grep_hits: list[tuple[int, str]] | None = None,
    ):
        self._search_hits = search_hits or []
        self._grep_hits = grep_hits or []

    def search(self, query: str, top_k: int = 10):
        return self._search_hits[:top_k]

    def grep(self, pattern: str):
        return self._grep_hits

    def search_by_date(self, start, end, top_k: int = 50):
        return []


class _FakeEventDB:
    """Minimal EventDB stub for query_events testing."""

    def __init__(
        self,
        lookup_results: list[tuple[int, float]] | None = None,
        type_results: list[tuple[int, float]] | None = None,
        predicate_results: list[tuple[int, float]] | None = None,
        facts: dict[int, list[str]] | None = None,
    ):
        self._lookup_results = lookup_results or []
        self._type_results = type_results or []
        self._predicate_results = predicate_results or []
        self._facts = facts or {}

    def lookup(self, entities, top_k=10, **kwargs):
        return self._lookup_results[:top_k]

    def lookup_by_type(self, type_name, top_k=10):
        return self._type_results[:top_k]

    def lookup_by_predicate(self, predicate, top_k=10):
        return self._predicate_results[:top_k]

    def get_facts_for_chunks(self, chunk_ids):
        return {cid: self._facts[cid] for cid in chunk_ids if cid in self._facts}


def _make_agent(
    memory_store=None,
    event_index=None,
    event_db=None,
    engine=None,
    scorer=None,
    max_hops: int = 5,
    max_results: int = 15,
    max_read_chars: int = 6000,
    pre_context: str = "",
    retrieval_guidance: str = "",
) -> ChronosAgent:
    return ChronosAgent(
        memory_store=[] if memory_store is None else memory_store,
        event_index=event_index,
        event_db=event_db,
        engine=engine,
        scorer=scorer,
        max_hops=max_hops,
        max_results=max_results,
        max_read_chars=max_read_chars,
        pre_context=pre_context,
        retrieval_guidance=retrieval_guidance,
    )


# ---------------------------------------------------------------------------
# TOOL_SPECS structure
# ---------------------------------------------------------------------------

def test_tool_specs_has_four_tools():
    assert len(TOOL_SPECS) == 4


def test_tool_specs_names():
    names = [t["function"]["name"] for t in TOOL_SPECS]
    assert names == [
        "search_turns",
        "query_events",
        "search_events_by_date",
        "finalize_answer",
    ]


def test_tool_specs_are_function_type():
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        assert "function" in spec
        assert "name" in spec["function"]
        assert "description" in spec["function"]
        assert "parameters" in spec["function"]


def test_finalize_answer_is_last_tool():
    assert TOOL_SPECS[-1]["function"]["name"] == "finalize_answer"


# ---------------------------------------------------------------------------
# CHRONOS_SYSTEM_PROMPT
# ---------------------------------------------------------------------------

def test_system_prompt_has_max_hops_placeholder():
    assert "{max_hops}" in CHRONOS_SYSTEM_PROMPT


def test_system_prompt_formats_correctly():
    formatted = CHRONOS_SYSTEM_PROMPT.format(max_hops=5, question_date_line="")
    assert "5" in formatted
    assert "{max_hops}" not in formatted


def test_system_prompt_references_new_tools():
    """System prompt mentions the 4-tool architecture, not dropped tools."""
    assert "query_events" in CHRONOS_SYSTEM_PROMPT
    assert "search_turns" in CHRONOS_SYSTEM_PROMPT
    assert "grep_turns" not in CHRONOS_SYSTEM_PROMPT
    assert "grep_events" not in CHRONOS_SYSTEM_PROMPT
    assert "read_turns" not in CHRONOS_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# search_turns
# ---------------------------------------------------------------------------

def test_search_turns_returns_indices():
    engine = _FakeEngine([
        {"index": 4, "score": 0.9},
        {"index": 1, "score": 0.8},
    ])
    agent = _make_agent(engine=engine, scorer=_FakeScorer())
    result = agent._tool_search_turns("hello", top_k=2)
    assert result["indices"] == [4, 1]
    assert result["query"] == "hello"


def test_search_turns_caps_top_k():
    engine = _FakeEngine([{"index": i, "score": 1.0} for i in range(50)])
    agent = _make_agent(engine=engine, scorer=_FakeScorer(), max_results=15)
    result = agent._tool_search_turns("q", top_k=100)
    assert len(result["indices"]) == 15


def test_search_turns_clamps_top_k_min():
    engine = _FakeEngine([{"index": 1, "score": 1.0}])
    agent = _make_agent(engine=engine, scorer=_FakeScorer())
    result = agent._tool_search_turns("q", top_k=-5)
    # top_k clamped to at least 1
    assert isinstance(result["indices"], list)


def test_search_turns_unavailable_without_engine():
    agent = _make_agent(engine=None, scorer=_FakeScorer())
    result = agent._tool_search_turns("query")
    assert result["indices"] == []
    assert "unavailable" in result.get("message", "")


def test_search_turns_unavailable_without_scorer():
    engine = _FakeEngine([{"index": 1, "score": 1.0}])
    agent = _make_agent(engine=engine, scorer=None)
    result = agent._tool_search_turns("query")
    assert result["indices"] == []


def test_search_turns_empty_query():
    agent = _make_agent(engine=_FakeEngine([]), scorer=_FakeScorer())
    result = agent._tool_search_turns("")
    assert result["indices"] == []


# ---------------------------------------------------------------------------
# query_events
# ---------------------------------------------------------------------------

def test_query_events_direct_entity_lookup():
    """Direct entity lookup returns matching chunks with facts."""
    event_db = _FakeEventDB(
        lookup_results=[(0, 1.0), (2, 0.5)],
        facts={0: ["user visited MoMA"], 2: ["user visited Rome"]},
    )
    agent = _make_agent(event_db=event_db)
    result = agent._tool_query_events(["MoMA"])
    assert len(result["results"]) == 2
    assert result["results"][0]["chunk_id"] == 0
    assert result["results"][0]["score"] == 1.0
    assert "user visited MoMA" in result["results"][0]["facts"]


def test_query_events_type_resolution():
    """When direct lookup misses, fall back to type-based resolution."""
    event_db = _FakeEventDB(
        lookup_results=[],  # direct lookup misses
        type_results=[(0, 1.0), (2, 0.8)],  # type resolution finds these
        facts={0: ["MoMA type museum"], 2: ["Rijksmuseum type museum"]},
    )
    agent = _make_agent(event_db=event_db)
    result = agent._tool_query_events(["museum"])
    assert len(result["results"]) == 2
    assert result["results"][0]["chunk_id"] == 0


def test_query_events_predicate_filter():
    """Predicate filter narrows results to matching predicates only."""
    event_db = _FakeEventDB(
        lookup_results=[(0, 1.0), (1, 0.8), (2, 0.6)],
        predicate_results=[(0, 1.0), (2, 0.5)],  # only chunks 0,2 match predicate
        facts={0: ["user visited MoMA"], 2: ["user visited Rome"]},
    )
    agent = _make_agent(event_db=event_db)
    result = agent._tool_query_events(["user"], predicates=["visited"])
    chunk_ids = [r["chunk_id"] for r in result["results"]]
    assert 0 in chunk_ids
    assert 2 in chunk_ids
    assert 1 not in chunk_ids  # filtered out by predicate


def test_query_events_empty_entities():
    event_db = _FakeEventDB()
    agent = _make_agent(event_db=event_db)
    result = agent._tool_query_events([])
    assert result["results"] == []
    assert "no entities" in result.get("message", "")


def test_query_events_no_event_db():
    agent = _make_agent(event_db=None)
    result = agent._tool_query_events(["MoMA"])
    assert result["results"] == []
    assert "unavailable" in result.get("message", "")


def test_query_events_caps_top_k():
    hits = [(i, 1.0 - i * 0.01) for i in range(30)]
    facts = {i: [f"fact about {i}"] for i in range(30)}
    event_db = _FakeEventDB(lookup_results=hits, facts=facts)
    agent = _make_agent(event_db=event_db, max_results=10)
    result = agent._tool_query_events(["entity"], top_k=100)
    assert len(result["results"]) == 10


# ---------------------------------------------------------------------------
# _execute_tool dispatch
# ---------------------------------------------------------------------------

def test_execute_tool_search_turns():
    engine = _FakeEngine([{"index": 7, "score": 0.5}])
    agent = _make_agent(engine=engine, scorer=_FakeScorer())
    result = agent._execute_tool("search_turns", {"query": "foo", "top_k": 5})
    assert result["indices"] == [7]


def test_execute_tool_query_events():
    event_db = _FakeEventDB(
        lookup_results=[(3, 0.9)],
        facts={3: ["user visited Rome"]},
    )
    agent = _make_agent(event_db=event_db)
    result = agent._execute_tool("query_events", {"entities": ["Rome"]})
    assert len(result["results"]) == 1
    assert result["results"][0]["chunk_id"] == 3


def test_execute_tool_search_events_by_date():
    """search_events_by_date dispatch still works."""
    event_idx = _FakeEventIndex()
    agent = _make_agent(event_index=event_idx)
    result = agent._execute_tool("search_events_by_date", {
        "start_date": "2023-01-01", "end_date": "2023-12-31",
    })
    assert "results" in result


def test_execute_tool_finalize_answer():
    agent = _make_agent()
    result = agent._execute_tool("finalize_answer", {"answer": "42"})
    assert result["answer"] == "42"


def test_execute_tool_unknown():
    agent = _make_agent()
    result = agent._execute_tool("nonexistent_tool", {})
    assert "error" in result


def test_execute_tool_dropped_tools_are_unknown():
    """Verify dropped tools return errors, not results."""
    agent = _make_agent()
    for name in ["grep_turns", "grep_events", "read_turns", "search_events"]:
        result = agent._execute_tool(name, {})
        assert "error" in result, f"{name} should be unknown"


# ---------------------------------------------------------------------------
# _safe_json_loads
# ---------------------------------------------------------------------------

def test_safe_json_loads_valid():
    assert _safe_json_loads('{"a": 1}') == {"a": 1}


def test_safe_json_loads_invalid():
    assert _safe_json_loads("not json") == {}
    assert _safe_json_loads("") == {}


# ---------------------------------------------------------------------------
# _fallback_answer
# ---------------------------------------------------------------------------

def test_fallback_answer_with_evidence():
    result = _fallback_answer(["first snippet", "second snippet"], "question?")
    assert result == "first snippet"


def test_fallback_answer_skips_empty():
    result = _fallback_answer(["", "  ", "real content"], "question?")
    assert result == "real content"


def test_fallback_answer_no_evidence():
    assert _fallback_answer([], "question?") == "I don't know."
    assert _fallback_answer(["", "  "], "question?") == "I don't know."


def test_fallback_answer_truncates():
    long = "x" * 500
    result = _fallback_answer([long], "q?")
    assert len(result) <= 200


# ---------------------------------------------------------------------------
# _summarize_result
# ---------------------------------------------------------------------------

def test_summarize_result_indices():
    assert _summarize_result({"indices": [1, 2, 3]}) == "indices=3"


def test_summarize_result_turns():
    assert _summarize_result({"turns": [{"idx": 1}, {"idx": 2}]}) == "turns=2"


def test_summarize_result_results():
    assert _summarize_result({"results": [{"chunk_id": 1}]}) == "results=1"


def test_summarize_result_other():
    assert "error" in _summarize_result({"error": "bad input"})


# ---------------------------------------------------------------------------
# _get_text
# ---------------------------------------------------------------------------

def test_get_text_from_entry_with_text_attr():
    store = [SimpleNamespace(text="hello")]
    agent = _make_agent(memory_store=store)
    assert agent._get_text(0) == "hello"


def test_get_text_from_plain_string():
    store = ["plain string"]
    agent = _make_agent(memory_store=store)
    assert agent._get_text(0) == "plain string"


def test_get_text_out_of_range():
    agent = _make_agent(memory_store=["a"])
    assert agent._get_text(5) is None
    assert agent._get_text(-1) is None


# ---------------------------------------------------------------------------
# answer() ReAct loop with FakeLLMClient
# ---------------------------------------------------------------------------

class _FakeToolCall:
    """Fake OpenAI tool_call object."""

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _FakeChoice:
    """Fake OpenAI choice object."""

    def __init__(self, content: str | None, tool_calls: list | None):
        self.message = SimpleNamespace(content=content, tool_calls=tool_calls)


class _FakeResponse:
    """Fake OpenAI completion response."""

    def __init__(self, content: str | None, tool_calls: list | None,
                 prompt_tokens: int = 100, completion_tokens: int = 50):
        self.choices = [_FakeChoice(content, tool_calls)]
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class _FakeLLMClient:
    """Fake OpenAI client that returns scripted responses."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self._call_count = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


def test_answer_immediate_finalize():
    """Agent calls finalize_answer on first hop."""
    responses = [
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc1", "finalize_answer", '{"answer": "42"}'),
            ],
        ),
    ]
    agent = _make_agent()
    agent._client = _FakeLLMClient(responses)
    result = agent.answer("What is the answer?")
    assert result["output"] == "42"
    assert result["hops"] == 1
    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 50


def test_answer_search_then_finalize():
    """Agent searches events, then finalizes."""
    event_db = _FakeEventDB(
        lookup_results=[(0, 1.0)],
        facts={0: ["Alice likes cats"]},
    )
    responses = [
        # Hop 1: query_events for "Alice"
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc1", "query_events", '{"entities": ["Alice"]}'),
            ],
        ),
        # Hop 2: finalize
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc2", "finalize_answer", '{"answer": "cats"}'),
            ],
        ),
    ]
    agent = _make_agent(event_db=event_db)
    agent._client = _FakeLLMClient(responses)
    result = agent.answer("What does Alice like?")
    assert result["output"] == "cats"
    assert result["hops"] == 2
    assert len(result["tool_log"]) == 2


def test_answer_no_tool_calls_uses_content():
    """LLM responds with content and no tool calls."""
    responses = [
        _FakeResponse(content="I know the answer: 42", tool_calls=None),
    ]
    agent = _make_agent()
    agent._client = _FakeLLMClient(responses)
    result = agent.answer("What is the answer?")
    assert result["output"] == "I know the answer: 42"
    assert result["hops"] == 0


def test_answer_exhausts_hops_uses_fallback():
    """Agent runs out of hops without finalizing."""
    event_db = _FakeEventDB(
        lookup_results=[(0, 1.0)],
        facts={0: ["evidence text"]},
    )
    responses = [
        # Hop 1: query_events
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc1", "query_events", '{"entities": ["evidence"]}'),
            ],
        ),
        # Hop 2: search_turns
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc2", "search_turns", '{"query": "evidence"}'),
            ],
        ),
        # max_hops+1 call with forced finalize
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc3", "finalize_answer", '{"answer": "forced"}'),
            ],
        ),
    ]
    engine = _FakeEngine([{"index": 0, "score": 0.9}])
    agent = _make_agent(event_db=event_db, engine=engine, scorer=_FakeScorer(), max_hops=2)
    agent._client = _FakeLLMClient(responses)
    result = agent.answer("Question?")
    assert result["output"] == "forced"


def test_answer_tracks_token_counts():
    """Token counts are accumulated across hops."""
    event_db = _FakeEventDB(
        lookup_results=[(0, 1.0)],
        facts={0: ["x marks the spot"]},
    )
    responses = [
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc1", "query_events", '{"entities": ["x"]}'),
            ],
            prompt_tokens=200,
            completion_tokens=30,
        ),
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc2", "finalize_answer", '{"answer": "done"}'),
            ],
            prompt_tokens=300,
            completion_tokens=20,
        ),
    ]
    agent = _make_agent(event_db=event_db)
    agent._client = _FakeLLMClient(responses)
    result = agent.answer("Q?")
    assert result["prompt_tokens"] == 500
    assert result["completion_tokens"] == 50


def test_answer_gen_time_is_positive():
    responses = [
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc1", "finalize_answer", '{"answer": "ok"}'),
            ],
        ),
    ]
    agent = _make_agent()
    agent._client = _FakeLLMClient(responses)
    result = agent.answer("Q?")
    assert result["gen_time"] >= 0


def test_answer_includes_pre_context_in_system():
    """Pre-context is appended to system prompt."""
    responses = [
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc1", "finalize_answer", '{"answer": "ok"}'),
            ],
        ),
    ]
    agent = _make_agent(pre_context="Here is some pre-retrieved context.")
    agent._client = _FakeLLMClient(responses)

    # Capture the messages sent to LLM
    calls = []
    original_create = agent._client.chat.completions.create

    def capture_create(**kwargs):
        calls.append(kwargs)
        return original_create(**kwargs)

    agent._client.chat.completions.create = capture_create
    agent.answer("Q?")

    # System message should contain pre-context
    system_msg = calls[0]["messages"][0]["content"]
    assert "Pre-Retrieved Context" in system_msg
    assert "Here is some pre-retrieved context." in system_msg


def test_answer_includes_retrieval_guidance_in_system():
    """Retrieval guidance is appended to system prompt."""
    responses = [
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc1", "finalize_answer", '{"answer": "ok"}'),
            ],
        ),
    ]
    agent = _make_agent(retrieval_guidance="Focus on entity names.")
    agent._client = _FakeLLMClient(responses)

    calls = []
    original_create = agent._client.chat.completions.create

    def capture_create(**kwargs):
        calls.append(kwargs)
        return original_create(**kwargs)

    agent._client.chat.completions.create = capture_create
    agent.answer("Q?")

    system_msg = calls[0]["messages"][0]["content"]
    assert "Retrieval Guidance" in system_msg
    assert "Focus on entity names." in system_msg


def test_answer_fallback_on_empty_finalize():
    """Empty finalize_answer triggers fallback from evidence."""
    event_db = _FakeEventDB(
        lookup_results=[(0, 1.0)],
        facts={0: ["the real answer is here"]},
    )
    responses = [
        # Query some evidence
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc1", "query_events", '{"entities": ["answer"]}'),
            ],
        ),
        # Empty finalize
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc2", "finalize_answer", '{"answer": ""}'),
            ],
        ),
    ]
    agent = _make_agent(event_db=event_db)
    agent._client = _FakeLLMClient(responses)
    result = agent.answer("Q?")
    # Fallback should use pre-context or evidence
    assert result["output"] != ""


def test_answer_tool_log_records_all_calls():
    """Tool log captures every tool call made."""
    event_db = _FakeEventDB(
        lookup_results=[(0, 1.0)],
        facts={0: ["content fact"]},
    )
    engine = _FakeEngine([{"index": 0, "score": 0.9}])
    responses = [
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc1", "query_events", '{"entities": ["content"]}'),
            ],
        ),
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc2", "search_turns", '{"query": "content"}'),
            ],
        ),
        _FakeResponse(
            content=None,
            tool_calls=[
                _FakeToolCall("tc3", "finalize_answer", '{"answer": "done"}'),
            ],
        ),
    ]
    agent = _make_agent(event_db=event_db, engine=engine, scorer=_FakeScorer())
    agent._client = _FakeLLMClient(responses)
    result = agent.answer("Q?")
    assert len(result["tool_log"]) == 3
    assert result["tool_log"][0]["tool"] == "query_events"
    assert result["tool_log"][1]["tool"] == "search_turns"
    assert result["tool_log"][2]["tool"] == "finalize_answer"
    assert result["tool_log"][2]["result"] == "FINAL"


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------

def test_constructor_defaults():
    agent = ChronosAgent(memory_store=[])
    assert agent._max_hops == 5
    assert agent._max_results == 15
    assert agent._max_read_chars == 6000
    assert agent._pre_context == ""
    assert agent._retrieval_guidance == ""
    assert agent._client is None
    assert agent._event_db is None


def test_constructor_custom_params():
    agent = ChronosAgent(
        memory_store=["a"],
        max_hops=3,
        max_results=10,
        max_read_chars=2000,
        pre_context="ctx",
        retrieval_guidance="guide",
    )
    assert agent._max_hops == 3
    assert agent._max_results == 10
    assert agent._max_read_chars == 2000
    assert agent._pre_context == "ctx"
    assert agent._retrieval_guidance == "guide"
