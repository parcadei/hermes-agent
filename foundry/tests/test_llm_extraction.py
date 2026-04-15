"""Tests for Foundry's LLM-backed extraction path."""

from __future__ import annotations

import json
from urllib import error as urllib_error

import pytest

from foundry.core import Foundry, FoundryConfig
from foundry.ingestion import (
    V9_JSON_SCHEMA,
    V9_SYSTEM_PROMPT,
    ExtractedTriple,
    FacetEntity,
    ConversationTurn,
    chunk_turns,
    extract_entities_and_triples,
    extract_via_llm,
    fact_from_triple,
)
from foundry.ontology import PropertyType
from foundry.storage import RawChunk


class _MockHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_MockHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _completion_response(structured: dict) -> _MockHTTPResponse:
    payload = {"choices": [{"message": {"content": json.dumps(structured)}}]}
    return _MockHTTPResponse(json.dumps(payload).encode("utf-8"))


def test_v9_prompt_constants_match_expected_shape() -> None:
    assert "STEP 1 — ENTITY SCAN" in V9_SYSTEM_PROMPT
    assert "STEP 6 — PROVENANCE" in V9_SYSTEM_PROMPT
    assert "Process User turns ONLY" in V9_SYSTEM_PROMPT
    assert V9_JSON_SCHEMA["name"] == "extraction"
    assert V9_JSON_SCHEMA["strict"] is True
    required = V9_JSON_SCHEMA["schema"]["properties"]["relations"]["items"]["required"]
    assert required == [
        "subject",
        "predicate",
        "object",
        "role",
        "when",
        "when_type",
        "quantity",
        "unit",
        "confidence",
        "sentiment",
        "source_turn",
    ]


def test_extract_via_llm_posts_structured_request_and_maps_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    long_suffix = " and ".join(["bikes"] * 256)

    def _fake_urlopen(request, timeout: int = 0):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _completion_response(
            {
                "entities": [{"name": "Bamboo Sushi", "entity_type": "restaurant"}],
                "relations": [
                    {
                        "subject": "user",
                        "predicate": "visits",
                        "object": "Bamboo Sushi",
                        "role": "event",
                        "when": "every Friday",
                        "when_type": "recurring",
                        "quantity": "2",
                        "unit": "times",
                        "confidence": "definite",
                        "sentiment": "positive",
                        "source_turn": 7,
                    }
                ],
            }
        )

    monkeypatch.setattr("foundry.ingestion.urllib_request.urlopen", _fake_urlopen)

    entities, triples = extract_via_llm(
        f"[Turn 7] User: Bamboo Sushi is my favorite {long_suffix}",
        "http://localhost:1234/v1",
        llm_model="qwen/qwen3.5-9b",
    )

    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    assert captured["timeout"] == 60
    body = captured["body"]
    assert body["model"] == "qwen/qwen3.5-9b"
    assert body["messages"][0]["content"] == V9_SYSTEM_PROMPT
    assert body["response_format"] == {"type": "json_schema", "json_schema": V9_JSON_SCHEMA}
    assert entities == [
        FacetEntity(
            name="Bamboo Sushi",
            facets={"what": "restaurant"},
            turn_idx=7,
            source_text=f"Bamboo Sushi is my favorite {long_suffix}",
        )
    ]
    assert len(triples) == 1
    triple = triples[0]
    assert triple.subject == "user"
    assert triple.predicate == "visits"
    assert triple.object == "Bamboo Sushi"
    assert triple.role == "event"
    assert triple.when == "every Friday"
    assert triple.when_type == "recurring"
    assert triple.quantity == "2"
    assert triple.unit == "times"
    assert triple.confidence == "definite"
    assert triple.sentiment == "positive"
    assert triple.source_turn == 7
    assert triple.turn_idx == 7


def test_extract_entities_and_triples_dispatches_to_llm_with_wrapped_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}
    expected_entities = [FacetEntity(name="Luna", facets={"what": "animal"}, turn_idx=-1, source_text="wrapped")]
    expected_triples = [ExtractedTriple(subject="user", predicate="has", object="Luna", source_text="wrapped")]

    def _fake_extract(chunk_text: str, llm_url: str, llm_model: str = "", max_tokens: int = 16000, temperature: float = 0.0):
        captured["chunk_text"] = chunk_text
        captured["llm_url"] = llm_url
        captured["llm_model"] = llm_model
        return expected_entities, expected_triples

    monkeypatch.setattr("foundry.ingestion.extract_via_llm", _fake_extract)

    chunk = RawChunk(chunk_idx=3, text="I have a cat named Luna.", turn_idx=11, role="user")
    entities, triples = extract_entities_and_triples(chunk, "http://localhost:1234/v1", "qwen/qwen3.5-9b")

    assert captured == {
        "chunk_text": "[Turn 11] User: I have a cat named Luna.",
        "llm_url": "http://localhost:1234/v1",
        "llm_model": "qwen/qwen3.5-9b",
    }
    assert entities[0].turn_idx == 11
    assert entities[0].source_text == chunk.text
    assert triples[0].chunk_idx == 3
    assert triples[0].turn_idx == 11
    assert triples[0].source_turn == 11
    assert triples[0].source_text == chunk.text


def test_extract_entities_and_triples_uses_regex_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"entities": 0, "triples": 0}
    expected_entities = [FacetEntity(name="Luna", facets={"what": "cat"}, turn_idx=0, source_text="src")]
    expected_triples = [ExtractedTriple(subject="user", predicate="has", object="Luna", turn_idx=0, source_text="src")]

    def _fake_entities(chunk, llm_url: str, model: str):
        called["entities"] += 1
        return expected_entities

    def _fake_triples(chunk, llm_url: str, model: str):
        called["triples"] += 1
        return expected_triples

    monkeypatch.setattr("foundry.ingestion.extract_faceted_entities", _fake_entities)
    monkeypatch.setattr("foundry.ingestion.extract_triples", _fake_triples)

    chunk = RawChunk(chunk_idx=0, text="I have a cat named Luna.", turn_idx=0, role="user")
    entities, triples = extract_entities_and_triples(chunk, "", "unused")

    assert called == {"entities": 1, "triples": 1}
    assert entities == expected_entities
    assert triples == expected_triples


def test_extract_via_llm_rejects_invalid_content_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(request, timeout: int = 0):
        payload = {"choices": [{"message": {"content": "not-json"}}]}
        return _MockHTTPResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("foundry.ingestion.urllib_request.urlopen", _fake_urlopen)

    with pytest.raises(ValueError, match="content was not valid JSON"):
        extract_via_llm("[Turn 0] User: I have a cat named Luna.", "http://localhost:1234/v1")


def test_extract_via_llm_handles_endpoint_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(request, timeout: int = 0):
        raise urllib_error.URLError("connection refused")

    monkeypatch.setattr("foundry.ingestion.urllib_request.urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError, match="request failed"):
        extract_via_llm("[Turn 0] User: I have a cat named Luna.", "http://localhost:1234/v1")


def test_extract_via_llm_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(request, timeout: int = 0):
        raise TimeoutError("timed out")

    monkeypatch.setattr("foundry.ingestion.urllib_request.urlopen", _fake_urlopen)

    with pytest.raises(TimeoutError, match="timed out"):
        extract_via_llm("[Turn 0] User: I have a cat named Luna.", "http://localhost:1234/v1")


def test_extract_via_llm_returns_empty_for_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError("urlopen should not be called for empty text")

    monkeypatch.setattr("foundry.ingestion.urllib_request.urlopen", _unexpected)

    assert extract_via_llm("   ", "http://localhost:1234/v1") == ([], [])


def test_extract_entities_and_triples_skips_assistant_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError("extract_via_llm should not run for assistant chunks")

    monkeypatch.setattr("foundry.ingestion.extract_via_llm", _unexpected)

    chunk = RawChunk(chunk_idx=1, text="Luna is a tabby cat.", turn_idx=1, role="assistant")
    assert extract_entities_and_triples(chunk, "http://localhost:1234/v1", "qwen/qwen3.5-9b") == ([], [])


def test_chunk_turns_preserves_original_role_for_split_chunks() -> None:
    turns = [ConversationTurn(turn_idx=4, role="assistant", text=" ".join(["word"] * 10))]

    chunks = chunk_turns(turns, chunk_size=3)

    assert len(chunks) > 1
    assert {chunk.role for chunk in chunks} == {"assistant"}


def test_fact_from_triple_defaults_source_turn_to_turn_index() -> None:
    fact = fact_from_triple(
        ExtractedTriple(
            subject="user",
            predicate="salary",
            object="$100k",
            turn_idx=8,
            source_turn=-1,
            source_text="I make $100k",
        ),
        canonical_subject="user",
    )

    assert fact.object_type == PropertyType.NUMERIC
    assert fact.source_turn == 8


def test_foundry_ingest_uses_llm_extraction_and_preserves_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(request, timeout: int = 0):
        return _completion_response(
            {
                "entities": [
                    {"name": "Bamboo Sushi", "entity_type": "restaurant"},
                    {"name": "Luna", "entity_type": "animal"},
                ],
                "relations": [
                    {
                        "subject": "user",
                        "predicate": "favorite restaurant",
                        "object": "Bamboo Sushi",
                        "role": "opinion",
                        "when": "",
                        "when_type": "",
                        "quantity": "0",
                        "unit": "",
                        "confidence": "definite",
                        "sentiment": "positive",
                        "source_turn": 0,
                    },
                    {
                        "subject": "user",
                        "predicate": "has",
                        "object": "Luna",
                        "role": "state",
                        "when": "",
                        "when_type": "",
                        "quantity": "2",
                        "unit": "",
                        "confidence": "definite",
                        "sentiment": "neutral",
                        "source_turn": 0,
                    },
                ],
            }
        )

    monkeypatch.setattr("foundry.ingestion.urllib_request.urlopen", _fake_urlopen)

    foundry = Foundry(
        FoundryConfig(
            extraction_url="http://localhost:1234/v1",
            extraction_model="qwen/qwen3.5-9b",
        )
    )
    report = foundry.ingest(
        "User: Bamboo Sushi is my favorite restaurant. I have two cats named Luna.\n"
        "Assistant: Nice choice."
    )

    assert report.chunks_processed == 2
    assert foundry.entity("bamboo_sushi") is not None
    user_facts = foundry.facts("user", current_only=False)
    favorite = next(fact for fact in user_facts if fact.predicate == "favorite_restaurant")
    has_luna = next(fact for fact in user_facts if fact.predicate == "has")
    assert favorite.object == "Bamboo Sushi"
    assert favorite.sentiment == "positive"
    assert favorite.confidence == "definite"
    assert favorite.source_turn == 0
    assert has_luna.object == "Luna"
    assert has_luna.quantity == "2"
    assert has_luna.role == "state"
