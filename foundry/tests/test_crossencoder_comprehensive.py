"""Comprehensive cross-encoder reranking coverage for Hermes Foundry."""

from __future__ import annotations

import copy
import math
import sys
import types

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from foundry.core import Foundry, FoundryConfig
from foundry.retrieval import CrossEncoderReranker, search_memory_hybrid
from foundry.storage import BM25Index


class FakeModel:
    def __init__(self, scores: object) -> None:
        self._scores = scores
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]]) -> object:
        self.calls.append([list(pair) for pair in pairs])
        if callable(self._scores):
            return self._scores(pairs)
        return self._scores


def _make_results(texts: list[str], *, scores: list[object] | None = None) -> list[dict[str, object]]:
    base_scores = scores if scores is not None else [float(idx) for idx in range(len(texts))]
    return [
        {"turn_id": idx, "chunk_id": idx, "text": text, "score": base_scores[idx]}
        for idx, text in enumerate(texts)
    ]


def _install_sentence_transformers_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    predict_output: object | None = None,
    init_error: Exception | None = None,
) -> dict[str, object]:
    calls = {"init": 0, "predict": 0, "pairs": []}

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            calls["init"] += 1
            calls["model_name"] = model_name
            if init_error is not None:
                raise init_error

        def predict(self, pairs: list[list[str]]) -> object:
            calls["predict"] += 1
            calls["pairs"] = [list(pair) for pair in pairs]
            if callable(predict_output):
                return predict_output(pairs)
            return predict_output

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return calls


def _length_scores(pairs: list[list[str]]) -> list[float]:
    return [float(len(pair[1])) for pair in pairs]


def _keyword_scores(keyword: str) -> callable:
    def score_fn(pairs: list[list[str]]) -> list[float]:
        return [10.0 if keyword in pair[1].lower() else 0.5 for pair in pairs]

    return score_fn


def test_invariant_reranking_preserves_result_set_when_top_k_covers_all() -> None:
    reranker = CrossEncoderReranker(top_k=4)
    reranker._model = FakeModel([0.1, 0.9, 0.4, 0.2])
    original = _make_results(["alpha", "beta", "gamma", "delta"])
    results = copy.deepcopy(original)

    ranked = reranker.rerank("query", results, top_k=4)

    assert {row["chunk_id"] for row in ranked} == {row["chunk_id"] for row in original}
    assert len(ranked) == 4
    assert all(isinstance(row["cross_encoder_score"], float) for row in ranked)
    assert all(math.isfinite(row["cross_encoder_score"]) for row in ranked)


def test_invariant_top_k_is_honored_and_scores_are_descending() -> None:
    reranker = CrossEncoderReranker(top_k=2)
    reranker._model = FakeModel([0.3, 0.8, 0.1])

    ranked = reranker.rerank("query", _make_results(["one", "two", "three"]))

    assert len(ranked) == 2
    assert [row["cross_encoder_score"] for row in ranked] == [0.8, 0.3]


def test_fuzz_empty_candidate_list_returns_empty() -> None:
    reranker = CrossEncoderReranker()

    assert reranker.rerank("query", []) == []


def test_fuzz_single_candidate_with_none_score_is_supported() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = FakeModel([0.75])

    ranked = reranker.rerank("query", [{"text": "solo", "score": None}])

    assert ranked == [{"text": "solo", "score": None, "cross_encoder_score": 0.75}]


def test_fuzz_candidates_with_empty_text_are_rerankable() -> None:
    reranker = CrossEncoderReranker()
    model = FakeModel([0.2, 0.1])
    reranker._model = model

    ranked = reranker.rerank("query", _make_results(["", "non-empty"]))

    assert len(ranked) == 2
    assert model.calls[0] == [["query", ""], ["query", "non-empty"]]


def test_fuzz_unicode_query_is_passed_through_to_model() -> None:
    reranker = CrossEncoderReranker()
    model = FakeModel([0.4])
    reranker._model = model

    reranker.rerank("猫 と café", [{"text": "gato", "score": 0.1}])

    assert model.calls[0] == [["猫 と café", "gato"]]


def test_behavioral_relevant_results_rank_higher_after_reranking() -> None:
    reranker = CrossEncoderReranker(top_k=3)
    reranker._model = FakeModel(_keyword_scores("luna"))
    results = _make_results(
        [
            "generic memory",
            "Luna is a tabby cat",
            "another unrelated note",
        ]
    )

    ranked = reranker.rerank("What about Luna?", results)

    assert ranked[0]["text"] == "Luna is a tabby cat"


def test_behavioral_irrelevant_results_are_demoted() -> None:
    reranker = CrossEncoderReranker(top_k=3)
    reranker._model = FakeModel([0.1, 0.9, 0.2])
    results = _make_results(
        [
            "unrelated bike maintenance",
            "Luna prefers salmon treats",
            "weather tomorrow",
        ]
    )

    ranked = reranker.rerank("What treats does Luna like?", results)

    assert ranked[-1]["text"] == "unrelated bike maintenance"


def test_behavioral_reranking_improves_hybrid_retrieval_quality() -> None:
    bm25 = BM25Index()
    bm25.add(0, "Luna like like like memory memory memory")
    bm25.add(1, "Luna likes salmon treats")
    bm25.add(2, "another generic memory")

    baseline = search_memory_hybrid("Luna like", bm25, top_k=3)
    reranked = search_memory_hybrid(
        "Luna like",
        bm25,
        reranker=lambda query, results: sorted(
            list(results),
            key=lambda row: ("salmon" not in str(row["text"]).lower(), "treat" not in str(row["text"]).lower()),
        ),
        top_k=3,
    )

    assert baseline[0]["text"] != "Luna likes salmon treats"
    assert reranked[0]["text"] == "Luna likes salmon treats"


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    texts=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=20),
        min_size=0,
        max_size=10,
    ),
    top_k=st.integers(min_value=0, max_value=12),
)
def test_property_output_length_respects_input_and_top_k(texts: list[str], top_k: int) -> None:
    reranker = CrossEncoderReranker(top_k=max(top_k, 1))
    reranker._model = FakeModel(_length_scores)
    results = _make_results(texts)

    ranked = reranker.rerank("query", results, top_k=top_k)

    assert len(ranked) <= len(results)
    assert len(ranked) <= max(top_k, 0)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    texts=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=20),
        min_size=1,
        max_size=10,
    ),
    top_k=st.integers(min_value=1, max_value=10),
)
def test_property_scores_are_sorted_descending(texts: list[str], top_k: int) -> None:
    reranker = CrossEncoderReranker(top_k=top_k)
    reranker._model = FakeModel(_length_scores)

    ranked = reranker.rerank("query", _make_results(texts), top_k=top_k)

    scores = [row["cross_encoder_score"] for row in ranked]
    assert scores == sorted(scores, reverse=True)


def test_unit_rerank_sorts_and_truncates() -> None:
    reranker = CrossEncoderReranker(top_k=2)
    reranker._model = FakeModel(np.array([0.2, 0.7, 0.4], dtype=np.float32))

    ranked = reranker.rerank("query", _make_results(["a", "b", "c"]))

    assert [row["text"] for row in ranked] == ["b", "c"]


def test_unit_lazy_loading_initializes_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_sentence_transformers_module(monkeypatch, predict_output=[0.4])
    reranker = CrossEncoderReranker("lazy-model")

    reranker.rerank("query", [{"text": "once", "score": 0.1}])
    reranker.rerank("query", [{"text": "twice", "score": 0.1}])

    assert calls["init"] == 1
    assert calls["model_name"] == "lazy-model"


def test_unit_model_initialization_sets_internal_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_sentence_transformers_module(monkeypatch, predict_output=[0.5])
    reranker = CrossEncoderReranker("init-model")

    reranker._ensure_model()

    assert reranker._model is not None


def test_unit_score_normalization_accepts_numpy_array() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = FakeModel(np.array([0.2, 0.8], dtype=np.float32))

    ranked = reranker.rerank("query", _make_results(["short", "longer"]))

    assert [type(row["cross_encoder_score"]) for row in ranked] == [float, float]


def test_control_flow_reranker_disabled_when_config_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCrossEncoderReranker:
        def __init__(self, model_name: str, top_k: int = 10) -> None:
            self.model_name = model_name

    monkeypatch.setattr("foundry.core.CrossEncoderReranker", FakeCrossEncoderReranker)

    foundry = Foundry(FoundryConfig(reranker_model=None))

    assert foundry._reranker is None


def test_control_flow_model_load_failure_propagates_on_direct_use(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_sentence_transformers_module(monkeypatch, init_error=RuntimeError("load failed"))
    reranker = CrossEncoderReranker("broken-model")

    with pytest.raises(RuntimeError, match="load failed"):
        reranker.rerank("query", [{"text": "alpha", "score": 0.1}])


def test_control_flow_predict_scalar_and_array_outputs_both_work(monkeypatch: pytest.MonkeyPatch) -> None:
    scalar_calls = _install_sentence_transformers_module(monkeypatch, predict_output=0.6)
    scalar_reranker = CrossEncoderReranker("scalar-model")
    scalar_ranked = scalar_reranker.rerank("query", [{"text": "solo", "score": 0.1}])

    array_model = FakeModel(np.array([0.1, 0.9], dtype=np.float32))
    array_reranker = CrossEncoderReranker()
    array_reranker._model = array_model
    array_ranked = array_reranker.rerank("query", _make_results(["first", "second"]))

    assert scalar_ranked[0]["cross_encoder_score"] == 0.6
    assert scalar_calls["pairs"] == [["query", "solo"]]
    assert [row["text"] for row in array_ranked] == ["second", "first"]


def test_control_flow_predict_numpy_zero_dim_scalar_output_works() -> None:
    reranker = CrossEncoderReranker("numpy-scalar-model")
    reranker._model = FakeModel(np.array(0.6, dtype=np.float32))

    ranked = reranker.rerank("query", [{"text": "solo", "score": 0.1}])

    assert ranked[0]["text"] == "solo"
    assert ranked[0]["score"] == 0.1
    assert ranked[0]["cross_encoder_score"] == pytest.approx(0.6)


def test_control_flow_whitespace_only_model_name_disables_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCrossEncoderReranker:
        def __init__(self, model_name: str, top_k: int = 10) -> None:
            self.model_name = model_name

    monkeypatch.setattr("foundry.core.CrossEncoderReranker", FakeCrossEncoderReranker)

    foundry = Foundry(FoundryConfig(reranker_model="   "))

    assert foundry._reranker is None


def test_adversarial_extremely_long_texts_do_not_break_reranking() -> None:
    reranker = CrossEncoderReranker(top_k=1)
    reranker._model = FakeModel([0.9, 0.1])
    long_text = "Luna " * 10000

    ranked = reranker.rerank("Luna", _make_results([long_text, "short note"]))

    assert ranked[0]["text"] == long_text


def test_adversarial_mismatched_query_and_document_languages_are_supported() -> None:
    reranker = CrossEncoderReranker(top_k=2)
    reranker._model = FakeModel([0.2, 0.8])

    ranked = reranker.rerank("猫", _make_results(["dog", "猫 は 元気"]))

    assert ranked[0]["text"] == "猫 は 元気"


@pytest.mark.parametrize("top_k", [0, -5])
def test_adversarial_non_positive_top_k_returns_empty(top_k: int) -> None:
    reranker = CrossEncoderReranker(top_k=max(top_k, 0))
    reranker._model = FakeModel([0.8])

    assert reranker.rerank("query", _make_results(["alpha"]), top_k=top_k) == []


def test_adversarial_huge_candidate_pool_is_overfetched_then_truncated() -> None:
    bm25 = BM25Index()
    for idx in range(500):
        bm25.add(idx, f"luna memory {idx}")

    observed: dict[str, object] = {}

    def reranker(query: str, results: list[dict[str, object]]) -> list[dict[str, object]]:
        observed["count"] = len(results)
        return list(reversed(results))

    hits = search_memory_hybrid("luna", bm25, reranker=reranker, top_k=5)

    assert observed["count"] == 20
    assert len(hits) == 5


def test_integration_search_memory_hybrid_overfetches_reranks_and_truncates() -> None:
    bm25 = BM25Index()
    for idx in range(30):
        token = "relevant" if idx == 19 else "background"
        bm25.add(idx, f"{token} document {idx}")

    observed: dict[str, object] = {}

    def reranker(query: str, results: list[dict[str, object]]) -> list[dict[str, object]]:
        observed["count"] = len(results)
        ranked = sorted(results, key=lambda row: ("relevant" not in row["text"], -row["chunk_id"]))
        for idx, row in enumerate(ranked):
            row["cross_encoder_score"] = float(len(ranked) - idx)
        return ranked

    hits = search_memory_hybrid("document", bm25, reranker=reranker, top_k=5)

    assert observed["count"] == 20
    assert len(hits) == 5
    assert hits[0]["text"] == "relevant document 19"


def test_snapshot_regression_fixed_query_candidate_order() -> None:
    reranker = CrossEncoderReranker(top_k=4)
    reranker._model = FakeModel([0.2, 0.95, 0.4, 0.7])
    results = _make_results(
        [
            "bike repair notes",
            "Luna likes salmon treats",
            "weather summary",
            "Mochi sleeps by the window",
        ]
    )

    ranked = reranker.rerank("What does Luna like?", results, top_k=4)

    assert [row["text"] for row in ranked] == [
        "Luna likes salmon treats",
        "Mochi sleeps by the window",
        "weather summary",
        "bike repair notes",
    ]


def test_idempotency_same_inputs_produce_same_order() -> None:
    reranker = CrossEncoderReranker(top_k=3)
    reranker._model = FakeModel([0.3, 0.8, 0.4])
    source = _make_results(["a", "b", "c"])

    first = reranker.rerank("query", copy.deepcopy(source), top_k=3)
    second = reranker.rerank("query", copy.deepcopy(source), top_k=3)

    assert [row["text"] for row in first] == [row["text"] for row in second]


def test_fallback_reranker_exception_degrades_to_unreranked_top_k() -> None:
    bm25 = BM25Index()
    for idx in range(25):
        bm25.add(idx, f"fallback document {idx}")

    baseline = search_memory_hybrid("fallback", bm25, top_k=4)
    degraded = search_memory_hybrid(
        "fallback",
        bm25,
        reranker=lambda query, results: (_ for _ in ()).throw(RuntimeError("boom")),
        top_k=4,
    )

    assert degraded == baseline
