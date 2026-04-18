"""Tests for query decomposition support."""

from __future__ import annotations

import json
from urllib.error import URLError
from unittest.mock import patch

from foundry.core import Foundry, FoundryConfig
from foundry.query import _parse_subqueries, decompose_query
from foundry.retrieval import AgenticRetriever, RetrievalBudget
from foundry.storage import FoundryStore


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


def test_parse_subqueries_supports_numbered_dashed_and_mixed_prefixes() -> None:
    text = "\n".join(
        [
            "1. What sport did Christian Abbiati play?",
            "2) What is the country of origin of {answer_1}?",
            "- Where was {answer_2} born?",
        ]
    )

    assert _parse_subqueries(text) == [
        "What sport did Christian Abbiati play?",
        "What is the country of origin of {answer_1}?",
        "Where was {answer_2} born?",
    ]


def test_parse_subqueries_validates_placeholders_and_skips_future_references() -> None:
    text = "\n".join(
        [
            "1. What religion did Karl Lueger follow?",
            "2. Who created {answer_1}?",
            "3. Where was {answer_4} born?",
            "4. Where was {answer_2} born?",
        ]
    )

    assert _parse_subqueries(text, max_hops=4) == [
        "What religion did Karl Lueger follow?",
        "Who created {answer_1}?",
        "Where was {answer_2} born?",
    ]


def test_parse_subqueries_ignores_non_numbered_commentary_lines() -> None:
    text = "\n".join(
        [
            "Here are the steps:",
            "1. What sport did Christian Abbiati play?",
            "2. What is the country of origin of {answer_1}?",
        ]
    )

    assert _parse_subqueries(text) == [
        "What sport did Christian Abbiati play?",
        "What is the country of origin of {answer_1}?",
    ]


def test_decompose_query_calls_openai_compatible_endpoint_and_parses_response() -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "1. What sport did Christian Abbiati play?\n"
                                "2. What is the country of origin of {answer_1}?"
                            )
                        }
                    }
                ]
            }
        )

    question = "What is the country of origin of the sport played by Christian Abbiati?"
    with patch("foundry.query.urllib_request.urlopen", side_effect=fake_urlopen):
        result = decompose_query(question, llm_url="http://llm", llm_model="gpt-4o-mini")

    assert result == [
        "What sport did Christian Abbiati play?",
        "What is the country of origin of {answer_1}?",
    ]
    assert captured["url"] == "http://llm/chat/completions"
    assert captured["timeout"] == 15
    assert captured["body"] == {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You decompose multi-hop questions into chained sub-questions. Use {answer_N} placeholders. Be concise.",
            },
            {
                "role": "user",
                "content": (
                    "Break this question into simple sub-questions that each require looking up a single fact. "
                    "Start from the innermost entity and work outward. Use {answer_1}, {answer_2}, etc. as "
                    "placeholders for prior answers. Use the actual entity names from the question, not pronouns.\n\n"
                    "Example:\n"
                    "Q: What is the country of origin of the sport played by Christian Abbiati?\n"
                    "1. What sport did Christian Abbiati play?\n"
                    "2. What is the country of origin of {answer_1}?\n\n"
                    "Example:\n"
                    "Q: Where was the creator of the religion Karl Lueger followed born?\n"
                    "1. What religion did Karl Lueger follow?\n"
                    "2. Who created {answer_1}?\n"
                    "3. Where was {answer_2} born?\n\n"
                    f"Q: {question}\n"
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }


def test_decompose_query_falls_back_to_original_question_when_llm_unavailable() -> None:
    question = "What is the country of origin of the sport played by Christian Abbiati?"

    with patch("foundry.query.urllib_request.urlopen", side_effect=URLError("offline")):
        result = decompose_query(question, llm_url="http://llm")

    assert result == [question]


def test_decompose_query_caps_subqueries_by_max_hops() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "\n".join(
                        [
                            "1. First hop?",
                            "2. Second hop with {answer_1}?",
                            "3. Third hop with {answer_2}?",
                            "4. Fourth hop with {answer_3}?",
                        ]
                    )
                }
            }
        ]
    }

    with patch("foundry.query.urllib_request.urlopen", return_value=_FakeHTTPResponse(payload)):
        result = decompose_query("complex question", llm_url="http://llm", max_hops=2)

    assert result == ["First hop?", "Second hop with {answer_1}?"]


def test_foundry_decompose_returns_original_question_without_llm_endpoint() -> None:
    foundry = Foundry(FoundryConfig())
    question = "Where was the creator of the religion Karl Lueger followed born?"

    assert foundry.decompose(question) == [question]


def test_decompose_query_handles_empty_and_single_hop_questions() -> None:
    with patch("foundry.query.urllib_request.urlopen") as mocked_urlopen:
        assert decompose_query("", llm_url="http://llm") == [""]
        mocked_urlopen.assert_not_called()

    payload = {"choices": [{"message": {"content": "1. What breed is Luna?"}}]}
    with patch("foundry.query.urllib_request.urlopen", return_value=_FakeHTTPResponse(payload)):
        assert decompose_query("What breed is Luna?", llm_url="http://llm") == ["What breed is Luna?"]


def test_decompose_query_handles_very_long_questions() -> None:
    question = "What is the country of origin of the sport played by " + ("Christian Abbiati " * 50).strip() + "?"
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "1. What sport did Christian Abbiati play?\n"
                        "2. What is the country of origin of {answer_1}?"
                    )
                }
            }
        ]
    }

    with patch("foundry.query.urllib_request.urlopen", return_value=_FakeHTTPResponse(payload)):
        result = decompose_query(question, llm_url="http://llm")

    assert result == [
        "What sport did Christian Abbiati play?",
        "What is the country of origin of {answer_1}?",
    ]


def test_agentic_retriever_executes_multi_hop_subqueries_via_search_memory() -> None:
    """AgenticRetriever processes multi-hop questions through search_memory tool."""
    search_calls: list[str] = []

    def search_memory(query: str, top_k: int = 5) -> list[dict[str, object]]:
        search_calls.append(query)
        if "Christian Abbiati" in query:
            return [{"text": "Christian Abbiati played football.", "score": 0.9, "turn_id": 0, "chunk_id": 0}]
        return [{"text": "Football originated in England.", "score": 0.8, "turn_id": 1, "chunk_id": 1}]

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


def test_agentic_retriever_always_starts_with_search_memory_tool() -> None:
    """AgenticRetriever always begins retrieval with a search_memory call."""
    search_calls: list[str] = []

    def search_memory(query: str, top_k: int = 5) -> list[dict[str, object]]:
        search_calls.append(query)
        return [{"text": "Luna is a tabby cat.", "score": 0.9, "turn_id": 0, "chunk_id": 0}]

    def answer(evidence: str) -> str:
        return evidence

    retriever = AgenticRetriever(
        {"search_memory": search_memory, "answer": answer},
        budget=RetrievalBudget(max_steps=3),
    )

    retriever.retrieve("What breed is Luna?")

    assert len(search_calls) >= 1


def test_agentic_retriever_returns_evidence_string_and_trace() -> None:
    """retrieve() returns a non-empty evidence string and a populated trace."""

    def search_memory(query: str, top_k: int = 5) -> list[dict[str, object]]:
        return [{"text": "Football originated in England.", "score": 0.9, "turn_id": 0, "chunk_id": 0}]

    def answer(evidence: str) -> str:
        return evidence

    retriever = AgenticRetriever(
        {"search_memory": search_memory, "answer": answer},
        budget=RetrievalBudget(max_steps=3),
    )

    evidence, trace = retriever.retrieve("Where was the sport played by Christian Abbiati from?")

    assert isinstance(evidence, str)
    assert len(trace.steps) >= 1


def test_agentic_retriever_handles_empty_search_results_without_crashing() -> None:
    """AgenticRetriever does not crash when search returns no results."""

    def search_memory(query: str, top_k: int = 5) -> list[dict[str, object]]:
        return []

    def answer(evidence: str) -> str:
        return evidence

    retriever = AgenticRetriever(
        {"search_memory": search_memory, "answer": answer},
        budget=RetrievalBudget(max_steps=2),
    )

    evidence, trace = retriever.retrieve("What is the country of origin of the sport played by Christian Abbiati?")

    assert isinstance(evidence, str)
    assert trace.terminated_by in {"answer", "budget", "no_action", "error"}


def test_agentic_retriever_respects_budget_max_steps() -> None:
    """AgenticRetriever terminates once max_steps is reached."""
    call_count = 0

    def search_memory(query: str, top_k: int = 5) -> list[dict[str, object]]:
        nonlocal call_count
        call_count += 1
        return []

    def answer(evidence: str) -> str:
        return evidence

    budget = RetrievalBudget(max_steps=2)
    retriever = AgenticRetriever(
        {"search_memory": search_memory, "answer": answer},
        budget=budget,
    )

    evidence, trace = retriever.retrieve("complex multi-hop question about sports origins")

    assert len(trace.steps) <= budget.max_steps + 1  # +1 for terminal answer step
