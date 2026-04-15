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
from foundry.retrieval import DeterministicRouter, _looks_multi_hop
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


def _router() -> DeterministicRouter:
    return DeterministicRouter(FoundryStore(), llm_url="http://llm")


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
def test_property_router_route_never_crashes_for_any_string(question: str) -> None:
    route = DeterministicRouter().route(question)
    assert route in {
        "agentic",
        "aggregation",
        "comparison",
        "fact_lookup",
        "meta_cognitive",
        "multi_hop",
        "preference",
        "social_relational",
        "spatial",
        "temporal",
    }


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
    router = _router()
    asked: list[str] = []

    def fake_fact_lookup(question: str, trace) -> str:  # type: ignore[no-untyped-def]
        del trace
        asked.append(question)
        if len(asked) == 1:
            return "turn 1: Christian Abbiati played football."
        return "turn 2: Football originated in England."

    with patch("foundry.retrieval.decompose_query", return_value=_SNAPSHOTS["What is the country of origin of the sport played by Christian Abbiati?"]):
        router._execute_fact_lookup = fake_fact_lookup  # type: ignore[method-assign]
        evidence, meta = router.execute("What is the country of origin of the sport played by Christian Abbiati?", "multi_hop")

    assert asked == [
        "What sport did Christian Abbiati play?",
        "What is the country of origin of football?",
    ]
    assert "England" in evidence
    assert meta["fallback"] is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the country of origin of the sport played by Christian Abbiati?", True),
        ("Where was the creator of the religion Karl Lueger followed born?", True),
        ("What breed is Luna?", False),
        ("When did we move to Portland?", False),
        ("Who is connected to Luna?", False),
    ],
)
def test_unit_looks_multi_hop(question: str, expected: bool) -> None:
    assert _looks_multi_hop(question) is expected


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
    ("question", "expected"),
    [
        ("When did we move to Portland?", "temporal"),
        ("How many pets do we have?", "aggregation"),
        ("What is your favorite restaurant?", "preference"),
        ("Compare Luna versus Mochi", "comparison"),
        ("Where does Luna live?", "spatial"),
        ("Who is connected to Luna?", "social_relational"),
        ("Do you remember if I have a cat?", "meta_cognitive"),
        ("What breed is Luna?", "fact_lookup"),
        ("Write a haiku about cats", "agentic"),
        ("What is the country of origin of the sport played by Christian Abbiati?", "multi_hop"),
    ],
)
def test_unit_every_router_classification_path(question: str, expected: str) -> None:
    assert DeterministicRouter().route(question) == expected


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


def test_integration_deterministic_router_executes_multi_hop_chain_end_to_end() -> None:
    router = _router()
    calls: list[str] = []

    def fact_lookup(question: str, trace) -> str:  # type: ignore[no-untyped-def]
        del trace
        calls.append(f"fact:{question}")
        if question == "What sport did Christian Abbiati play?":
            return "turn 1: Christian Abbiati played football."
        return "turn 2: Football originated in England."

    with patch("foundry.retrieval.decompose_query", return_value=_SNAPSHOTS["What is the country of origin of the sport played by Christian Abbiati?"]):
        router._execute_fact_lookup = fact_lookup  # type: ignore[method-assign]
        evidence, meta = router.execute("What is the country of origin of the sport played by Christian Abbiati?", "multi_hop")

    assert calls == [
        "fact:What sport did Christian Abbiati play?",
        "fact:What is the country of origin of football?",
    ]
    assert evidence == "turn 2: Football originated in England."
    assert meta["pipeline"] == "multi_hop"
    assert meta["fallback"] is None


def test_idempotency_same_query_same_mocked_response_is_identical() -> None:
    question = "Where was the creator of the religion Karl Lueger followed born?"
    expected = _SNAPSHOTS[question]

    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(expected)):
        first = decompose_query(question, llm_url="http://llm")
    with patch("foundry.query.urllib_request.urlopen", return_value=_response_for(expected)):
        second = decompose_query(question, llm_url="http://llm")

    assert first == second == expected


@pytest.mark.parametrize(
    ("subqueries", "expected_tools", "responses"),
    [
        (
            ["What sport did Christian Abbiati play?", "What is the country of origin of {answer_1}?"],
            ["fact_lookup", "fact_lookup"],
            {
                "What sport did Christian Abbiati play?": "turn 1: Christian Abbiati played football.",
                "What is the country of origin of football?": "turn 2: Football originated in England.",
            },
        ),
        (
            ["What religion did Karl Lueger follow?", "Who created {answer_1}?", "Where was {answer_2} born?"],
            ["fact_lookup", "fact_lookup", "spatial"],
            {
                "What religion did Karl Lueger follow?": "turn 1: Karl Lueger followed christianity.",
                "Who created christianity?": "turn 2: Christianity was created by Jesus.",
                "Where was Jesus born?": "jesus: born Bethlehem",
            },
        ),
        (
            ["What project did we launch?", "When did we launch {answer_1}?"],
            ["fact_lookup", "temporal"],
            {
                "What project did we launch?": "turn 1: We launched Apollo.",
                "When did we launch Apollo?": "Apollo @ 2024-01-03T00:00:00",
            },
        ),
    ],
)
def test_composability_decomposition_works_with_multiple_router_types(
    subqueries: list[str],
    expected_tools: list[str],
    responses: dict[str, str],
) -> None:
    router = _router()
    seen_tools: list[str] = []

    def fact_lookup(question: str, trace) -> str:  # type: ignore[no-untyped-def]
        del trace
        seen_tools.append("fact_lookup")
        return responses[question]

    def temporal(question: str, trace) -> str:  # type: ignore[no-untyped-def]
        del trace
        seen_tools.append("temporal")
        return responses[question]

    def spatial(question: str, trace) -> str:  # type: ignore[no-untyped-def]
        del trace
        seen_tools.append("spatial")
        return responses[question]

    router._execute_fact_lookup = fact_lookup  # type: ignore[method-assign]
    router._execute_temporal = temporal  # type: ignore[method-assign]
    router._execute_spatial = spatial  # type: ignore[method-assign]

    with patch("foundry.retrieval.decompose_query", return_value=subqueries):
        evidence, meta = router.execute("seed question", "multi_hop")

    assert seen_tools == expected_tools
    assert evidence == responses[list(responses)[-1]]
    assert meta["fallback"] is None
