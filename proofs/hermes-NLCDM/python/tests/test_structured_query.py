"""Unit tests for StructuredQueryAgent tools (no LLM calls).

Tests the deterministic tool implementations:
  * _tool_find_by_entity
  * _tool_find_by_date_range
  * _tool_search_by_keyword
  * _tool_read_turns
  * _execute_tool dispatch
  * _fallback_answer helper
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from mabench.structured_query import (
    StructuredQueryAgent,
    _fallback_answer,
    _parse_date,
    _safe_json_loads,
    _summarize_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeBM25:
    """Minimal BM25 stub returning fixed hits."""

    def __init__(self, hits: list[tuple[int, float]]):
        self._hits = hits

    def query(self, text: str, top_k: int = 10) -> list[tuple[int, float]]:
        return self._hits[:top_k]


def _make_agent(
    memory_store=None,
    entity_inv_idx=None,
    event_calendar=None,
    bm25=None,
) -> StructuredQueryAgent:
    # Explicit None checks: empty list is falsy but valid (test expects
    # mutations on the caller-owned list to be visible to the agent).
    return StructuredQueryAgent(
        memory_store=[] if memory_store is None else memory_store,
        entity_inv_idx={} if entity_inv_idx is None else entity_inv_idx,
        event_calendar=[] if event_calendar is None else event_calendar,
        bm25_index=bm25,
    )


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------

def test_parse_date_valid():
    assert _parse_date("2025-03-14") == date(2025, 3, 14)
    assert _parse_date("  2025-03-14  ") == date(2025, 3, 14)


def test_parse_date_single_digit_month_day():
    assert _parse_date("2025-3-7") == date(2025, 3, 7)


def test_parse_date_invalid():
    assert _parse_date("") is None
    assert _parse_date("not-a-date") is None
    assert _parse_date("2025-13-01") is None  # invalid month
    assert _parse_date(None) is None  # type: ignore[arg-type]


def test_safe_json_loads():
    assert _safe_json_loads('{"a": 1}') == {"a": 1}
    assert _safe_json_loads("not json") == {}
    assert _safe_json_loads("") == {}


# ---------------------------------------------------------------------------
# find_turns_by_entity
# ---------------------------------------------------------------------------

def test_find_by_entity_exact_match():
    agent = _make_agent(
        entity_inv_idx={"john": {1, 5, 9}, "paris": {2, 3}},
    )
    result = agent._tool_find_by_entity("john")
    assert set(result["indices"]) == {1, 5, 9}
    assert result["count"] == 3


def test_find_by_entity_case_insensitive():
    agent = _make_agent(entity_inv_idx={"john": {1, 5}})
    result = agent._tool_find_by_entity("John")
    assert set(result["indices"]) == {1, 5}


def test_find_by_entity_substring_fallback():
    agent = _make_agent(
        entity_inv_idx={"john smith": {1, 2}, "john doe": {3}},
    )
    result = agent._tool_find_by_entity("john")
    assert set(result["indices"]) == {1, 2, 3}


def test_find_by_entity_empty():
    agent = _make_agent(entity_inv_idx={"john": {1}})
    result = agent._tool_find_by_entity("")
    assert result["indices"] == []


def test_find_by_entity_no_match():
    agent = _make_agent(entity_inv_idx={"paris": {1}})
    result = agent._tool_find_by_entity("london")
    assert result["indices"] == []


def test_find_by_entity_returns_recency_sorted():
    agent = _make_agent(entity_inv_idx={"john": {1, 10, 5}})
    result = agent._tool_find_by_entity("john")
    # Most recent (highest idx) first
    assert result["indices"] == [10, 5, 1]


# ---------------------------------------------------------------------------
# find_turns_by_date_range
# ---------------------------------------------------------------------------

def test_find_by_date_range_basic():
    calendar = [
        {"turn_idx": 1, "date": datetime(2025, 3, 10, 10, 0)},
        {"turn_idx": 2, "date": datetime(2025, 3, 15, 10, 0)},
        {"turn_idx": 3, "date": datetime(2025, 3, 20, 10, 0)},
        {"turn_idx": 4, "date": datetime(2025, 4, 1, 10, 0)},
    ]
    agent = _make_agent(event_calendar=calendar)
    result = agent._tool_find_by_date_range("2025-03-10", "2025-03-20")
    assert set(result["indices"]) == {1, 2, 3}
    assert result["start"] == "2025-03-10"


def test_find_by_date_range_date_objects():
    calendar = [
        {"turn_idx": 1, "date": date(2025, 3, 10)},
        {"turn_idx": 2, "date": date(2025, 3, 15)},
    ]
    agent = _make_agent(event_calendar=calendar)
    result = agent._tool_find_by_date_range("2025-03-01", "2025-03-31")
    assert set(result["indices"]) == {1, 2}


def test_find_by_date_range_swaps_reversed():
    calendar = [
        {"turn_idx": 1, "date": datetime(2025, 3, 10)},
    ]
    agent = _make_agent(event_calendar=calendar)
    # User passes end < start — agent should swap
    result = agent._tool_find_by_date_range("2025-03-31", "2025-03-01")
    assert 1 in result["indices"]


def test_find_by_date_range_invalid_format():
    calendar = [{"turn_idx": 1, "date": datetime(2025, 3, 10)}]
    agent = _make_agent(event_calendar=calendar)
    result = agent._tool_find_by_date_range("garbage", "2025-03-10")
    assert result["indices"] == []
    assert "error" in result


def test_find_by_date_range_no_matches():
    calendar = [{"turn_idx": 1, "date": datetime(2025, 3, 10)}]
    agent = _make_agent(event_calendar=calendar)
    result = agent._tool_find_by_date_range("2024-01-01", "2024-12-31")
    assert result["indices"] == []


# ---------------------------------------------------------------------------
# search_turns_by_keyword
# ---------------------------------------------------------------------------

def test_search_by_keyword_bm25():
    bm25 = _FakeBM25([(3, 1.5), (1, 0.8), (7, 0.4)])
    agent = _make_agent(bm25=bm25)
    result = agent._tool_search_by_keyword("hello world", top_k=3)
    assert result["indices"] == [3, 1, 7]


def test_search_by_keyword_caps_top_k():
    bm25 = _FakeBM25([(i, 1.0) for i in range(50)])
    agent = _make_agent(bm25=bm25)
    result = agent._tool_search_by_keyword("query", top_k=100)
    # Capped at max_turns_per_tool (15)
    assert len(result["indices"]) == 15


def test_search_by_keyword_empty():
    agent = _make_agent(bm25=_FakeBM25([]))
    result = agent._tool_search_by_keyword("")
    assert result["indices"] == []


def test_search_by_keyword_no_bm25():
    agent = _make_agent(bm25=None)
    result = agent._tool_search_by_keyword("query")
    assert result["indices"] == []


# ---------------------------------------------------------------------------
# semantic_search
# ---------------------------------------------------------------------------

class _FakeScorer:
    def embed(self, text: str):
        import numpy as np
        return np.ones(4, dtype=float)


class _FakeEngine:
    def __init__(self, hits):
        self._hits = hits

    def query(self, emb, top_k: int = 10):
        return self._hits[:top_k]


def test_semantic_search_returns_indices():
    agent = _make_agent()
    agent._engine = _FakeEngine([{"index": 4, "score": 0.9}, {"index": 1, "score": 0.8}])
    agent._scorer = _FakeScorer()
    result = agent._tool_semantic_search("hello", top_k=2)
    assert result["indices"] == [4, 1]


def test_semantic_search_caps_top_k():
    agent = _make_agent()
    agent._engine = _FakeEngine([{"index": i, "score": 1.0} for i in range(50)])
    agent._scorer = _FakeScorer()
    result = agent._tool_semantic_search("q", top_k=100)
    assert len(result["indices"]) == 15  # max_turns_per_tool cap


def test_semantic_search_unavailable_without_engine():
    agent = _make_agent()
    result = agent._tool_semantic_search("q")
    assert result["indices"] == []
    assert "unavailable" in result.get("message", "")


def test_semantic_search_empty_query():
    agent = _make_agent()
    agent._engine = _FakeEngine([{"index": 1, "score": 1.0}])
    agent._scorer = _FakeScorer()
    result = agent._tool_semantic_search("")
    assert result["indices"] == []


def test_compute_date_diff_basic():
    agent = _make_agent()
    result = agent._tool_compute_date_diff("2025-03-20", "2025-03-10")
    assert result["days_between"] == 10
    assert result["weeks_between"] == 1
    assert result["earlier_date"] == "2025-03-10"
    assert result["later_date"] == "2025-03-20"
    assert result["same_date"] is False


def test_compute_date_diff_reversed():
    agent = _make_agent()
    result = agent._tool_compute_date_diff("2025-03-10", "2025-03-20")
    assert result["days_between"] == 10
    assert result["days_signed"] == -10
    assert result["earlier_date"] == "2025-03-10"
    assert result["later_date"] == "2025-03-20"


def test_compute_date_diff_months():
    agent = _make_agent()
    result = agent._tool_compute_date_diff("2025-09-15", "2025-03-15")
    assert result["days_between"] == 184
    assert result["months_between"] == 6


def test_compute_date_diff_invalid():
    agent = _make_agent()
    result = agent._tool_compute_date_diff("garbage", "2025-03-10")
    assert "error" in result


def test_execute_tool_date_diff():
    agent = _make_agent()
    result = agent._execute_tool(
        "compute_date_diff", {"date_a": "2025-03-15", "date_b": "2025-03-01"}
    )
    assert result["days_between"] == 14
    assert result["weeks_between"] == 2


def test_execute_tool_semantic_search():
    agent = _make_agent()
    agent._engine = _FakeEngine([{"index": 7, "score": 0.5}])
    agent._scorer = _FakeScorer()
    result = agent._execute_tool(
        "semantic_search", {"query": "foo", "top_k": 5}
    )
    assert result["indices"] == [7]


# ---------------------------------------------------------------------------
# read_turns
# ---------------------------------------------------------------------------

def test_read_turns_with_text_attr():
    store = [
        SimpleNamespace(text="turn zero"),
        SimpleNamespace(text="turn one"),
        SimpleNamespace(text="turn two"),
    ]
    agent = _make_agent(memory_store=store)
    result = agent._tool_read_turns([0, 2])
    assert len(result["turns"]) == 2
    assert result["turns"][0] == {"idx": 0, "text": "turn zero"}
    assert result["turns"][1] == {"idx": 2, "text": "turn two"}


def test_read_turns_plain_strings():
    store = ["zero", "one", "two"]
    agent = _make_agent(memory_store=store)
    result = agent._tool_read_turns([1])
    assert result["turns"][0]["text"] == "one"


def test_read_turns_out_of_range():
    store = [SimpleNamespace(text="a")]
    agent = _make_agent(memory_store=store)
    result = agent._tool_read_turns([0, 5, -1])
    assert len(result["turns"]) == 1


def test_read_turns_empty():
    agent = _make_agent(memory_store=[SimpleNamespace(text="a")])
    result = agent._tool_read_turns([])
    assert result["turns"] == []


def test_read_turns_caps_at_15():
    store = [SimpleNamespace(text=f"turn{i}") for i in range(50)]
    agent = _make_agent(memory_store=store)
    result = agent._tool_read_turns(list(range(50)))
    assert len(result["turns"]) == 15


def test_read_turns_char_budget():
    store = [SimpleNamespace(text="x" * 5000) for _ in range(5)]
    agent = _make_agent(memory_store=store)
    result = agent._tool_read_turns([0, 1, 2, 3, 4])
    # With max_read_chars=6000, expect truncation after ~2 turns
    total = sum(len(t["text"]) for t in result["turns"])
    assert total <= 6100  # allow ellipsis overhead


# ---------------------------------------------------------------------------
# _execute_tool dispatch
# ---------------------------------------------------------------------------

def test_execute_tool_entity():
    agent = _make_agent(entity_inv_idx={"john": {1}})
    result = agent._execute_tool("find_turns_by_entity", {"entity": "john"})
    assert 1 in result["indices"]


def test_execute_tool_date_range():
    calendar = [{"turn_idx": 1, "date": datetime(2025, 3, 10)}]
    agent = _make_agent(event_calendar=calendar)
    result = agent._execute_tool(
        "find_turns_by_date_range",
        {"start_date": "2025-03-01", "end_date": "2025-03-31"},
    )
    assert 1 in result["indices"]


def test_execute_tool_keyword():
    agent = _make_agent(bm25=_FakeBM25([(5, 0.9)]))
    result = agent._execute_tool(
        "search_turns_by_keyword", {"query": "foo", "top_k": 5}
    )
    assert result["indices"] == [5]


def test_execute_tool_read():
    agent = _make_agent(memory_store=[SimpleNamespace(text="hello")])
    result = agent._execute_tool("read_turns", {"indices": [0]})
    assert result["turns"][0]["text"] == "hello"


def test_execute_tool_unknown():
    agent = _make_agent()
    result = agent._execute_tool("nonexistent_tool", {})
    assert "error" in result


# ---------------------------------------------------------------------------
# Fallback helpers
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


def test_summarize_result_indices():
    assert _summarize_result({"indices": [1, 2, 3]}) == "indices=3"


def test_summarize_result_turns():
    assert _summarize_result({"turns": [{"idx": 1}, {"idx": 2}]}) == "turns=2"


def test_summarize_result_other():
    assert "error" in _summarize_result({"error": "bad input"})


# ---------------------------------------------------------------------------
# Date index construction
# ---------------------------------------------------------------------------

def test_date_index_built_from_calendar():
    calendar = [
        {"turn_idx": 1, "date": datetime(2025, 3, 10, 10, 0)},
        {"turn_idx": 2, "date": datetime(2025, 3, 10, 14, 0)},  # same day
        {"turn_idx": 3, "date": date(2025, 3, 15)},
    ]
    agent = _make_agent(event_calendar=calendar)
    agent._ensure_date_index()
    assert date(2025, 3, 10) in agent._date_index
    assert set(agent._date_index[date(2025, 3, 10)]) == {1, 2}
    assert agent._date_index[date(2025, 3, 15)] == [3]


def test_date_index_handles_missing_date():
    calendar = [
        {"turn_idx": 1, "date": datetime(2025, 3, 10)},
        {"turn_idx": 2},  # no date
        {"turn_idx": 3, "date": None},
    ]
    agent = _make_agent(event_calendar=calendar)
    agent._ensure_date_index()
    assert len(agent._date_index) == 1
    assert 1 in agent._date_index[date(2025, 3, 10)]


def test_date_index_rebuilds_lazily():
    calendar: list[dict] = []
    agent = _make_agent(event_calendar=calendar)
    # Calendar empty at construction
    assert agent._date_index == {}
    # Populate calendar AFTER construction (simulates ingestion)
    calendar.append({"turn_idx": 5, "date": datetime(2025, 3, 10)})
    result = agent._tool_find_by_date_range("2025-03-01", "2025-03-31")
    assert 5 in result["indices"]
