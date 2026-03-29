"""Tests for CrossEncoderReranker — ms-marco-MiniLM-L-6-v2 reranking module.

Cross-encoder reranking takes [query][SEP][document] pairs through a
cross-encoder and reorders candidates by relevance score.  This is the
single biggest contributor in agentmemory's ablation (-30.7 pp).

All fast tests use mocks (no model loading). The slow test (@pytest.mark.slow)
loads the real 22M-param cross-encoder.

Test patterns follow EntityReranker test conventions:
  - No conftest.py — self-contained
  - DIM=64, np.random.RandomState(42), L2-normalized fake embeddings
  - list[dict] with {"text": str, "score": float} interface
"""

import numpy as np
import pytest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 64
RNG = np.random.RandomState(42)


def _make_results(n: int, base_score: float = 0.9) -> list[dict]:
    """Create n fake retrieval results with descending scores."""
    return [
        {
            "text": f"Fact number {i} about some topic",
            "score": base_score - i * 0.01,
        }
        for i in range(n)
    ]


def _make_results_with_known_texts(texts: list[str]) -> list[dict]:
    """Create results from specific text strings, with uniform scores."""
    return [
        {"text": t, "score": 0.5}
        for t in texts
    ]


# ===========================================================================
# Test Group 1: Core reranking behaviour (mocked model)
# ===========================================================================


class TestCrossEncoderRerankerCore:
    """Verify reranking logic with mocked cross-encoder model."""

    def test_rerank_empty_returns_empty(self):
        """Empty input list passes through unchanged."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        result = reranker.rerank("some query", [])
        assert result == []

    def test_rerank_reorders_by_cross_encoder_score(self):
        """Results are reordered by cross-encoder score, not original score."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        # Three results where original order is 0, 1, 2
        results = [
            {"text": "least relevant", "score": 0.99},
            {"text": "most relevant", "score": 0.80},
            {"text": "middle relevant", "score": 0.85},
        ]

        # Mock model: assign scores that REVERSE the original order
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
        reranker._model = mock_model

        reranked = reranker.rerank("test query", results, top_k=3)

        assert reranked[0]["text"] == "most relevant"
        assert reranked[1]["text"] == "middle relevant"
        assert reranked[2]["text"] == "least relevant"

    def test_rerank_truncates_to_top_k(self):
        """When given 30 candidates with top_k=10, returns exactly 10."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(top_k=10)

        results = _make_results(30)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.arange(30, 0, -1, dtype=float)
        reranker._model = mock_model

        reranked = reranker.rerank("test query", results)
        assert len(reranked) == 10

    def test_rerank_preserves_dict_fields(self):
        """Original fields (text, score) are preserved in output dicts."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        results = [
            {"text": "Paris is the capital of France", "score": 0.95,
             "source": "cosine", "extra_field": 42},
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])
        reranker._model = mock_model

        reranked = reranker.rerank("What is the capital of France?", results, top_k=5)

        assert len(reranked) == 1
        r = reranked[0]
        assert r["text"] == "Paris is the capital of France"
        assert r["score"] == 0.95  # original score preserved
        assert r["source"] == "cosine"
        assert r["extra_field"] == 42

    def test_rerank_adds_cross_encoder_score_field(self):
        """Each result dict gets a 'cross_encoder_score' field."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        results = _make_results(3)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.3, 0.7, 0.5])
        reranker._model = mock_model

        reranked = reranker.rerank("test query", results, top_k=3)

        for r in reranked:
            assert "cross_encoder_score" in r, (
                f"Missing 'cross_encoder_score' field in result: {r}"
            )
            assert isinstance(r["cross_encoder_score"], float)

    def test_cross_encoder_scores_descending(self):
        """Output is sorted by cross_encoder_score in descending order."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        results = _make_results(5)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        reranker._model = mock_model

        reranked = reranker.rerank("test query", results, top_k=5)

        ce_scores = [r["cross_encoder_score"] for r in reranked]
        assert ce_scores == sorted(ce_scores, reverse=True), (
            f"Cross-encoder scores not descending: {ce_scores}"
        )

    def test_rerank_passes_correct_pairs_to_model(self):
        """Model.predict receives [[query, text1], [query, text2], ...] pairs."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        results = [
            {"text": "fact A", "score": 0.9},
            {"text": "fact B", "score": 0.8},
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.6])
        reranker._model = mock_model

        reranker.rerank("my query", results, top_k=2)

        call_args = mock_model.predict.call_args[0][0]
        assert call_args == [["my query", "fact A"], ["my query", "fact B"]]

    def test_top_k_override(self):
        """Explicit top_k argument overrides the instance default."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(top_k=5)

        results = _make_results(20)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.arange(20, 0, -1, dtype=float)
        reranker._model = mock_model

        # Override to top_k=3
        reranked = reranker.rerank("test query", results, top_k=3)
        assert len(reranked) == 3

    def test_fewer_results_than_top_k(self):
        """When fewer results than top_k, returns all of them."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(top_k=10)

        results = _make_results(3)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.3, 0.7])
        reranker._model = mock_model

        reranked = reranker.rerank("test query", results)
        assert len(reranked) == 3


# ===========================================================================
# Test Group 2: Lazy loading and defaults
# ===========================================================================


class TestCrossEncoderLazyLoading:
    """Verify lazy model loading and default configuration."""

    def test_lazy_model_loading(self):
        """Model is NOT loaded at construction time, only at first rerank()."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        assert reranker._model is None, (
            "Model should be None until first rerank() call"
        )

    def test_default_model_name(self):
        """Default model is cross-encoder/ms-marco-MiniLM-L-6-v2."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_custom_model_name(self):
        """Custom model name is stored correctly."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="custom/model")
        assert reranker._model_name == "custom/model"

    def test_default_top_k(self):
        """Default top_k is 10."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        assert reranker._top_k == 10

    def test_ensure_model_loads_once(self):
        """_ensure_model only loads the model once (idempotent)."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        mock_instance = MagicMock()
        mock_ce_class = MagicMock(return_value=mock_instance)

        with patch.dict(
            "sys.modules",
            {"sentence_transformers": MagicMock(CrossEncoder=mock_ce_class)},
        ):
            reranker._ensure_model()
            reranker._ensure_model()

        # Should only be constructed once (second call is a no-op)
        mock_ce_class.assert_called_once()


# ===========================================================================
# Test Group 3: Integration with retrieval pipeline (config + wiring)
# ===========================================================================


class TestCrossEncoderConfig:
    """Verify RetrievalConfig has cross_encoder fields."""

    def test_config_has_cross_encoder_field(self):
        """RetrievalConfig has cross_encoder bool field, default False."""
        from mabench.configs import RetrievalConfig

        cfg = RetrievalConfig()
        assert hasattr(cfg, "cross_encoder"), (
            "RetrievalConfig missing 'cross_encoder' field"
        )
        assert cfg.cross_encoder is False

    def test_config_has_cross_encoder_top_k_field(self):
        """RetrievalConfig has cross_encoder_top_k int field, default 10."""
        from mabench.configs import RetrievalConfig

        cfg = RetrievalConfig()
        assert hasattr(cfg, "cross_encoder_top_k"), (
            "RetrievalConfig missing 'cross_encoder_top_k' field"
        )
        assert cfg.cross_encoder_top_k == 10

    def test_config_cross_encoder_enabled(self):
        """RetrievalConfig can be created with cross_encoder=True."""
        from mabench.configs import RetrievalConfig

        cfg = RetrievalConfig(cross_encoder=True, cross_encoder_top_k=5)
        assert cfg.cross_encoder is True
        assert cfg.cross_encoder_top_k == 5


class TestCrossEncoderRetrievalWiring:
    """Verify cross-encoder is wired into Retriever._retrieve_v2."""

    def test_retriever_accepts_cross_encoder_reranker(self):
        """Retriever.__init__ accepts a cross_encoder_reranker parameter."""
        from mabench.retrieval import Retriever

        mock_reranker = MagicMock()
        retriever = Retriever(cross_encoder_reranker=mock_reranker)
        assert retriever._cross_encoder_reranker is mock_reranker

    def test_retriever_default_no_cross_encoder(self):
        """By default, Retriever has no cross-encoder reranker."""
        from mabench.retrieval import Retriever

        retriever = Retriever()
        assert retriever._cross_encoder_reranker is None

    def test_retrieve_v2_calls_cross_encoder_when_present(self):
        """_retrieve_v2 calls cross_encoder_reranker.rerank when it exists."""
        from mabench.retrieval import Retriever
        from mabench.configs import RetrievalConfig

        mock_engine = MagicMock()
        mock_engine.query_cooc_boost.return_value = _make_results(5)
        mock_scorer = MagicMock()
        mock_scorer.embed.return_value = RNG.randn(DIM).astype(np.float32)

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = _make_results(3)

        retriever = Retriever(
            engine=mock_engine,
            scorer=mock_scorer,
            config=RetrievalConfig(cooc_boost=True),
            cross_encoder_reranker=mock_reranker,
        )

        results = retriever._retrieve_v2("test query")

        mock_reranker.rerank.assert_called_once()
        # First arg should be the query text
        call_args = mock_reranker.rerank.call_args
        assert call_args[0][0] == "test query"

    def test_retrieve_v2_skips_cross_encoder_when_absent(self):
        """_retrieve_v2 works without cross-encoder (no crash, no reranking)."""
        from mabench.retrieval import Retriever
        from mabench.configs import RetrievalConfig

        mock_engine = MagicMock()
        mock_engine.query_cooc_boost.return_value = _make_results(5)
        mock_scorer = MagicMock()
        mock_scorer.embed.return_value = RNG.randn(DIM).astype(np.float32)

        retriever = Retriever(
            engine=mock_engine,
            scorer=mock_scorer,
            config=RetrievalConfig(cooc_boost=True),
        )

        results = retriever._retrieve_v2("test query")
        assert len(results) > 0  # should still work


# ===========================================================================
# Test Group 4: Real model (slow, requires model download)
# ===========================================================================


@pytest.mark.slow
class TestCrossEncoderRealModel:
    """Integration test with actual cross-encoder model.

    Requires: pip install sentence-transformers
    Downloads: ~22MB ms-marco-MiniLM-L-6-v2 model on first run.
    """

    def test_cross_encoder_real_relevance_ordering(self):
        """Real model ranks relevant document above irrelevant ones."""
        from mabench.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        results = _make_results_with_known_texts([
            "Berlin is the capital of Germany and has many museums",
            "The weather today is sunny with clear skies",
            "Paris is the capital of France and a major European city",
        ])

        reranked = reranker.rerank(
            "What is the capital of France?",
            results,
            top_k=3,
        )

        # Paris fact should be ranked first
        assert "Paris" in reranked[0]["text"], (
            f"Expected Paris fact first, got: {reranked[0]['text']}"
        )
        # Each result should have a cross_encoder_score
        for r in reranked:
            assert "cross_encoder_score" in r
            assert isinstance(r["cross_encoder_score"], float)
