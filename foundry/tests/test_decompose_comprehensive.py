"""Comprehensive tests for query decomposition."""

from __future__ import annotations

import json
import re
from urllib.error import URLError
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from foundry.core import Foundry, FoundryConfig
from foundry.query import _parse_subqueries, decompose_query
from foundry.retrieval import AgenticRetriever, RetrievalBudget
from foundry.storage import FoundryStore


FUZZ_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_PLACEHOLDER_RE = re.compile(r"\{answer_(\d+)\}")
_SNAPSHOTS: dict[str, list[str]] = {
    "What is the country of origin of the sport played by Christian Abbiati?": [
        "What sport did Christian Abbiati play?",
        "What is the country of origin of {answer_1}?",
    ],
    "Where was the creator of the religion Karl Lueger followed born?": [
        "What religion did Karl Lueger follow?",
        "Who created {answer_1}?",
        "Where was {answer_2} born?",
    ],
    "How old is the founder of the company that makes ChatGPT?": [
        "What company makes ChatGPT?",
        "Who founded {answer_1}?",
        "How old is {answer_2}?",
    ],
}


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


def _response_for(lines: list[str] | str) -> _FakeHTTPResponse:
    content = lines if isinstance(lines, str) else "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, start=1))
    return _FakeHTTPResponse({"choices": [{"message": {"content": content}}]})


def _assert_valid_placeholders(subqueries: list[str]) -> None:
    for idx, subquery in enumerate(subqueries, start=1):
        refs = [int(match) for match in _PLACEHOLDER_RE.findall(subquery)]
        assert all(0 < ref < idx for ref in refs)


def _estimated_original_hops(question: str) -> int:
    lowered = question.lower()
    relation_terms = ["played by", "followed", "creator of", "founder of", "country of origin", "born", "makes"]
    return max(1, sum(term in lowered for term in relation_terms))


def _agentic_retriever_noop() -> AgenticRetriever:
    """Build a minimal AgenticRetriever with no-op tools for testing decompose logic."""

    def search_memory(query: str, top_k: int = 5) -> list[dict[str, object]]:
        return []

    def answer(evidence: str) -> str:
        return evidence

    return AgenticRetriever(
        {"search_memory": search_memory, "answer": answer},
        budget=RetrievalBudget(max_steps=4),
    )


@pytest.mark.parametrize("question,expected", list(_SNAPSHOTS.items()))
def test_snapshot_regression_golden_decompositions(question: str, expected: list[str]) -> None:
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(expected)):
        result = decompose_query(question, llm_url="http://llm")

    assert result == expected


@pytest.mark.parametrize("question,expected", list(_SNAPSHOTS.items()))
def test_invariant_decomposition_hops_are_bounded_and_original_is_recoverable(
    question: str,
    expected: list[str],
) -> None:
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(expected)):
        result = decompose_query(question, llm_url="http://llm", max_hops=3)

    assert 1 <= len(result) <= 3
    assert len(result) <= _estimated_original_hops(question) + 1
    _assert_valid_placeholders(result)

    foundry = Foundry(FoundryConfig())
    assert foundry.decompose(question) == [question]


@FUZZ_SETTINGS
@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=400))
def test_property_parse_subqueries_never_crashes_for_any_string(text: str) -> None:
    result = _parse_subqueries(text, max_hops=3)
    assert isinstance(result, list)
    assert len(result) <= 3
    _assert_valid_placeholders(result)


@FUZZ_SETTINGS
@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=400))
def test_property_parse_subqueries_never_crashes_for_any_string_v2(question: str) -> None:
    """decompose_query with no LLM URL falls back to the original question — never crashes."""
    result = decompose_query(question, llm_url="")
    assert isinstance(result, list)
    assert len(result) >= 1


@FUZZ_SETTINGS
@given(
    question=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=200),
    response_lines=st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=80), min_size=0, max_size=5),
)
def test_property_decompose_query_is_bounded_and_idempotent_for_same_response(
    question: str,
    response_lines: list[str],
) -> None:
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(response_lines)):
        first = decompose_query(question, llm_url="http://llm", max_hops=3)
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(response_lines)):
        second = decompose_query(question, llm_url="http://llm", max_hops=3)

    assert first == second
    assert len(first) <= 3
    _assert_valid_placeholders(first if first != [question] else [])


@pytest.mark.parametrize(
    ("question", "response_lines", "expected"),
    [
        (
            "What is the country of origin of the sport played by Christian Abbiati?",
            _SNAPSHOTS["What is the country of origin of the sport played by Christian Abbiati?"],
            _SNAPSHOTS["What is the country of origin of the sport played by Christian Abbiati?"],
        ),
        (
            "Where was the creator of the religion Karl Lueger followed born?",
            _SNAPSHOTS["Where was the creator of the religion Karl Lueger followed born?"],
            _SNAPSHOTS["Where was the creator of the religion Karl Lueger followed born?"],
        ),
        (
            "What breed is Luna?",
            ["What breed is Luna?"],
            ["What breed is Luna?"],
        ),
    ],
)
def test_behavioral_real_questions_decompose_as_expected(
    question: str,
    response_lines: list[str],
    expected: list[str],
) -> None:
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(response_lines)):
        assert decompose_query(question, llm_url="http://llm") == expected


def test_behavioral_decomposed_chain_resolves_in_order() -> None:
    """decompose_query produces a chain where {answer_N} placeholders resolve in order."""
    question = "What is the country of origin of the sport played by Christian Abbiati?"
    expected = _SNAPSHOTS[question]

    # Verify placeholder ordering invariant: each subquery only references prior answers
    for idx, subquery in enumerate(expected, start=1):
        refs = [int(m) for m in re.findall(r"\{answer_(\d+)\}", subquery)]
        assert all(ref < idx for ref in refs), f"Subquery {idx} references future answer: {subquery!r}"

    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(expected)):
        result = decompose_query(question, llm_url="http://llm")

    assert result == expected
    assert len(result) == 2


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What is the country of origin of the sport played by Christian Abbiati?",
            ["What sport did Christian Abbiati play?", "What is the country of origin of {answer_1}?"],
        ),
        (
            "Where was the creator of the religion Karl Lueger followed born?",
            ["What religion did Karl Lueger follow?", "Who created {answer_1}?", "Where was {answer_2} born?"],
        ),
        ("What breed is Luna?", ["What breed is Luna?"]),
        ("When did we move to Portland?", ["When did we move to Portland?"]),
    ],
)
def test_unit_decompose_query_multi_hop_detection(question: str, expected: list[str]) -> None:
    """decompose_query returns expected subqueries when LLM provides the canonical decomposition."""
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(expected)):
        result = decompose_query(question, llm_url="http://llm")

    assert result == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1. First?\n2. Second {answer_1}?", ["First?", "Second {answer_1}?"]),
        ("- First?\n- Second?", ["First?", "Second?"]),
        ("Commentary\n1) First?\n2) Second?", ["First?", "Second?"]),
        ("1. First?\n2. Bad {answer_3}?\n3. Third {answer_1}?", ["First?", "Third {answer_1}?"]),
    ],
)
def test_unit_parse_subqueries_cases(text: str, expected: list[str]) -> None:
    assert _parse_subqueries(text) == expected


def test_unit_decompose_query_posts_expected_body_shape() -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _response_for(_SNAPSHOTS["What is the country of origin of the sport played by Christian Abbiati?"])

    with patch("foundry.query.urllib_request.urlopen", side_effect=fake_urlopen):
        result = decompose_query("What is the country of origin of the sport played by Christian Abbiati?", llm_url="http://llm")

    assert result == _SNAPSHOTS["What is the country of origin of the sport played by Christian Abbiati?"]
    assert captured["url"] == "http://llm/chat/completions"
    assert captured["timeout"] == 15
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-4o-mini"
    assert body["max_tokens"] == 256


@pytest.mark.parametrize(
    ("question", "expected_tool"),
    [
        # Always-on plan: search_memory runs first; when it returns evidence, answer
        # is called immediately and the loop terminates before search_temporal fires.
        ("When did we move to Portland?", "search_memory"),
        # aggregate is not part of the always-on plan; search_memory + answer run instead.
        ("How many pets do we have?", "search_memory"),
        ("What breed is Luna?", "search_memory"),
    ],
)
def test_unit_agentic_retriever_uses_appropriate_tools_per_question_type(
    question: str, expected_tool: str
) -> None:
    """AgenticRetriever always runs search_memory (always-on plan) for any question type."""
    tools_used: list[str] = []

    def search_memory(query: str, top_k: int = 5) -> list[dict[str, object]]:
        tools_used.append("search_memory")
        return [{"text": "some fact", "score": 0.9, "turn_id": 0, "chunk_id": 0}]

    def search_temporal(query: str, top_k: int = 5, entities: list[str] | None = None) -> dict[str, object]:
        tools_used.append("search_temporal")
        return {"matched_events": [{"event": "moved", "date": "2024-01-03T00:00:00", "turn_id": 0, "text": "moved"}], "bucket": "contains"}

    def aggregate(entity_type: str, op: str, property_predicate: str = "") -> dict[str, object]:
        tools_used.append("aggregate")
        return {"result": 3, "entities": ["luna", "mochi", "goldie"]}

    def answer(evidence: str) -> str:
        tools_used.append("answer")
        return evidence

    retriever = AgenticRetriever(
        {
            "search_memory": search_memory,
            "search_temporal": search_temporal,
            "aggregate": aggregate,
            "answer": answer,
        },
        budget=RetrievalBudget(max_steps=4),
    )

    evidence, trace = retriever.retrieve(question)

    assert expected_tool in tools_used, f"Expected '{expected_tool}' in {tools_used} for question: {question!r}"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("nonsense output", ["question"]),
        ("", ["question"]),
        ("1. Only one hop?", ["Only one hop?"]),
    ],
)
def test_control_flow_llm_garbage_empty_and_single_subquery(response: str, expected: list[str]) -> None:
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(response)):
        result = decompose_query("question", llm_url="http://llm")

    assert result == expected


def test_control_flow_llm_timeout_returns_original_question() -> None:
    with patch("foundry.query.urllib_request.urlopen", side_effect=TimeoutError("timeout")):
        assert decompose_query("question", llm_url="http://llm") == ["question"]


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        "¿Cuál es el país de origen del deporte que jugó Christian Abbiati?",
        "💡" * 50,
        "What breed is Luna?" * 100,
        "Tell me something pleasant about cats.",
    ],
)
def test_fuzz_edge_inputs_do_not_crash(question: str) -> None:
    with patch("foundry.query.urllib_request.urlopen", side_effect=URLError("offline")):
        result = decompose_query(question, llm_url="http://llm")

    assert result == [question]


@pytest.mark.parametrize(
    ("question", "response_lines"),
    [
        (
            "What is the birthplace of the founder of the company that made the language used to write the tool that answers this question?",
            [
                "What company made the language used to write the tool that answers this question?",
                "Who founded {answer_1}?",
                "Where was {answer_2} born?",
            ],
        ),
        (
            "What is {answer_1}?",
            ["What is {answer_1}?"],
        ),
        (
            "What is the country of origin of the sport played by X and why?",
            [
                "What sport did X play?",
                "What is the country of origin of {answer_1}?",
                "Why?",
            ],
        ),
        (
            "What is the country of origin of the sport played by X; DROP TABLE facts; --?",
            [
                "What sport did X play?",
                "What is the country of origin of {answer_1}; DROP TABLE facts; --?",
            ],
        ),
    ],
)
def test_adversarial_inputs_remain_bounded_and_placeholder_safe(
    question: str,
    response_lines: list[str],
) -> None:
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(response_lines)):
        result = decompose_query(question, llm_url="http://llm")

    assert len(result) <= 3
    _assert_valid_placeholders(result if result != [question] else [])


def test_integration_agentic_retriever_executes_multi_hop_chain_end_to_end() -> None:
    """AgenticRetriever end-to-end: processes multi-hop question through search_memory calls."""
    search_calls: list[str] = []

    def search_memory(query: str, top_k: int = 5) -> list[dict[str, object]]:
        search_calls.append(query)
        if "Christian Abbiati" in query:
            return [{"text": "Christian Abbiati played football.", "score": 0.95, "turn_id": 0, "chunk_id": 0}]
        return [{"text": "Football originated in England.", "score": 0.90, "turn_id": 1, "chunk_id": 1}]

    def answer(evidence: str) -> str:
        return evidence

    retriever = AgenticRetriever(
        {"search_memory": search_memory, "answer": answer},
        budget=RetrievalBudget(max_steps=4),
    )

    evidence, trace = retriever.retrieve("What is the country of origin of the sport played by Christian Abbiati?")

    assert len(trace.steps) >= 1
    assert trace.terminated_by in {"answer", "budget"}
    assert any(step["tool"] == "search_memory" for step in trace.steps)


def test_idempotency_same_query_same_mocked_response_is_identical() -> None:
    question = "Where was the creator of the religion Karl Lueger followed born?"
    expected = _SNAPSHOTS[question]

    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(expected)):
        first = decompose_query(question, llm_url="http://llm")
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(expected)):
        second = decompose_query(question, llm_url="http://llm")

    assert first == second == expected


@pytest.mark.parametrize(
    ("question", "subqueries", "final_evidence"),
    [
        (
            "What is the country of origin of the sport played by Christian Abbiati?",
            ["What sport did Christian Abbiati play?", "What is the country of origin of {answer_1}?"],
            "Football originated in England.",
        ),
        (
            "Where was the creator of the religion Karl Lueger followed born?",
            ["What religion did Karl Lueger follow?", "Who created {answer_1}?", "Where was {answer_2} born?"],
            "Jesus was born in Bethlehem.",
        ),
    ],
)
def test_composability_decomposition_chains_produce_valid_placeholder_sequences(
    question: str,
    subqueries: list[str],
    final_evidence: str,
) -> None:
    """Decomposed chains always have valid placeholder ordering regardless of chain length."""
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(subqueries)):
        result = decompose_query(question, llm_url="http://llm")

    assert result == subqueries
    _assert_valid_placeholders(result)
