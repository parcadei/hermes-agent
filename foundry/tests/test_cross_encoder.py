"""Cross-encoder reranking tests for Hermes Foundry."""

from __future__ import annotations

import sys
import types

import pytest

from foundry.core import Foundry, FoundryConfig
from foundry.retrieval import CrossEncoderReranker, search_memory_hybrid
from foundry.storage import BM25Index


def _install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch, scores: list[float]) -> dict[str, object]:
    calls = {"init": 0, "predict": 0, "pairs": []}
    score_values = list(scores)

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            calls["init"] += 1
            calls["model_name"] = model_name

        def predict(self, pairs: list[list[str]]) -> list[float]:
            calls["predict"] += 1
            calls["pairs"] = [list(pair) for pair in pairs]
            return list(score_values)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return calls


def test_cross_encoder_reranker_sorts_scores_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_sentence_transformers(monkeypatch, [0.2, 0.9, 0.4])
    reranker = CrossEncoderReranker("fake-model", top_k=2)
    results = [
        {"text": "alpha", "score": 0.1},
        {"text": "beta", "score": 0.2},
        {"text": "gamma", "score": 0.3},
    ]

    ranked = reranker.rerank("pets", results)

    assert [row["text"] for row in ranked] == ["beta", "gamma"]
    assert [row["cross_encoder_score"] for row in ranked] == [0.9, 0.4]
    assert calls["model_name"] == "fake-model"
    assert calls["pairs"] == [["pets", "alpha"], ["pets", "beta"], ["pets", "gamma"]]


def test_cross_encoder_reranker_lazy_loads_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_sentence_transformers(monkeypatch, [0.5, 0.4])
    reranker = CrossEncoderReranker("lazy-model")

    first = reranker.rerank("query", [{"text": "one", "score": 1.0}])
    second = reranker.rerank("query", [{"text": "two", "score": 2.0}])

    assert first[0]["cross_encoder_score"] == 0.5
    assert second[0]["cross_encoder_score"] == 0.5
    assert calls["init"] == 1
    assert calls["predict"] == 2


def test_cross_encoder_reranker_empty_results_short_circuit() -> None:
    reranker = CrossEncoderReranker()

    assert reranker.rerank("query", []) == []


def test_cross_encoder_reranker_keeps_stable_order_for_tied_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sentence_transformers(monkeypatch, [0.7, 0.7, 0.7])
    reranker = CrossEncoderReranker()
    results = [
        {"text": "first", "score": 0.3},
        {"text": "second", "score": 0.2},
        {"text": "third", "score": 0.1},
    ]

    ranked = reranker.rerank("query", results, top_k=3)

    assert [row["text"] for row in ranked] == ["first", "second", "third"]


def test_cross_encoder_reranker_handles_single_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sentence_transformers(monkeypatch, [0.8])
    reranker = CrossEncoderReranker()

    ranked = reranker.rerank("query", [{"text": "solo", "score": 0.1}])

    assert ranked == [{"text": "solo", "score": 0.1, "cross_encoder_score": 0.8}]


def test_cross_encoder_reranker_requires_text_field(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sentence_transformers(monkeypatch, [0.1])
    reranker = CrossEncoderReranker()

    with pytest.raises(KeyError):
        reranker.rerank("query", [{"score": 0.1}])


def test_search_memory_hybrid_accepts_reranker_callable() -> None:
    bm25 = BM25Index()
    bm25.add(0, "alpha cat cat")
    bm25.add(1, "beta cat")

    observed: dict[str, object] = {}

    def reranker(query: str, results: list[dict[str, object]]) -> list[dict[str, object]]:
        observed["query"] = query
        observed["texts"] = [row["text"] for row in results]
        ranked = list(results)
        ranked.reverse()
        for idx, row in enumerate(ranked, start=1):
            row["cross_encoder_score"] = float(idx)
        return ranked

    hits = search_memory_hybrid("cat", bm25, reranker=reranker, top_k=2)

    assert observed == {"query": "cat", "texts": ["alpha cat cat", "beta cat"]}
    assert [row["text"] for row in hits] == ["beta cat", "alpha cat cat"]
    assert [row["cross_encoder_score"] for row in hits] == [1.0, 2.0]


def test_foundry_reranker_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCrossEncoderReranker:
        def __init__(self, model_name: str, top_k: int = 10) -> None:
            self.model_name = model_name
            self.top_k = top_k

        def rerank(self, query: str, results: list[dict[str, object]], top_k: int | None = None) -> list[dict[str, object]]:
            return results

    monkeypatch.setattr("foundry.core.CrossEncoderReranker", FakeCrossEncoderReranker)

    disabled = Foundry(FoundryConfig(reranker_model=None))
    enabled = Foundry(FoundryConfig(reranker_model="stub-reranker"))

    assert disabled._reranker is None
    assert isinstance(enabled._reranker, FakeCrossEncoderReranker)
    assert enabled._reranker.model_name == "stub-reranker"


def test_foundry_passes_reranker_into_retrieval_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCrossEncoderReranker:
        def __init__(self, model_name: str, top_k: int = 10) -> None:
            self.model_name = model_name
            self.top_k = top_k

        def rerank(self, query: str, results: list[dict[str, object]], top_k: int | None = None) -> list[dict[str, object]]:
            for row in results:
                row["cross_encoder_score"] = 1.0
            return results

    captured: dict[str, object] = {}

    class FakeAgenticRetriever:
        def __init__(self, store: object, **kwargs: object) -> None:
            captured["agentic_reranker"] = kwargs.get("reranker")

        def retrieve(self, question: str) -> tuple[str, object]:
            return "agentic evidence", types.SimpleNamespace(terminated_by="answer")

    class FakeDeterministicRouter:
        def __init__(self, store: object, **kwargs: object) -> None:
            captured["deterministic_reranker"] = kwargs.get("reranker")

        def route(self, question: str) -> str:
            return "fact_lookup"

        def execute(self, question: str, pipeline: str) -> tuple[str, dict[str, object]]:
            return "deterministic evidence", {"pipeline": pipeline}

    monkeypatch.setattr("foundry.core.CrossEncoderReranker", FakeCrossEncoderReranker)
    monkeypatch.setattr("foundry.core.AgenticRetriever", FakeAgenticRetriever)
    monkeypatch.setattr("foundry.core.DeterministicRouter", FakeDeterministicRouter)

    foundry = Foundry(FoundryConfig(reranker_model="stub-reranker"))

    assert foundry.retrieve("what do you remember?")[0] == "agentic evidence"
    assert foundry.retrieve_deterministic("what do you remember?")[0] == "deterministic evidence"
    assert getattr(captured["agentic_reranker"], "__self__", None) is foundry._reranker
    assert getattr(captured["deterministic_reranker"], "__self__", None) is foundry._reranker
