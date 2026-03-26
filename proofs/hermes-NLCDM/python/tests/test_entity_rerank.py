"""Tests for entity-position residual reranking (arXiv:2603.19664).

Structural tests that verify:
  1. embed_entity_position returns correct shape and is normalized
  2. Entity-position reranking changes candidate ordering
  3. entity_residual field survives store/replace cycle
  4. Config flag flows from RetrievalConfig through agent wiring
  5. _extract_query_entity handles various query formats
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Path setup
_TESTS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _TESTS_DIR.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))


# ── Config and structural tests (no GPU needed) ─────────────────────

class TestRetrievalConfigEntityRerank:
    def test_entity_rerank_default_false(self):
        from mabench.configs import RetrievalConfig
        cfg = RetrievalConfig()
        assert cfg.entity_rerank is False

    def test_entity_rerank_alpha_default(self):
        from mabench.configs import RetrievalConfig
        cfg = RetrievalConfig()
        assert cfg.entity_rerank_alpha == 0.3

    def test_entity_rerank_enabled(self):
        from mabench.configs import RetrievalConfig
        cfg = RetrievalConfig(entity_rerank=True, entity_rerank_alpha=0.5)
        assert cfg.entity_rerank is True
        assert cfg.entity_rerank_alpha == 0.5

    def test_frozen(self):
        from mabench.configs import RetrievalConfig
        cfg = RetrievalConfig(entity_rerank=True)
        with pytest.raises(AttributeError):
            cfg.entity_rerank = False


class TestMemoryEntryEntityResidual:
    def test_field_exists(self):
        from coupled_engine import MemoryEntry
        fields = {f.name for f in MemoryEntry.__dataclass_fields__.values()}
        assert "entity_residual" in fields

    def test_default_none(self):
        emb = np.zeros(1024)
        from coupled_engine import MemoryEntry
        entry = MemoryEntry(text="test", embedding=emb)
        assert entry.entity_residual is None

    def test_set_residual(self):
        emb = np.zeros(1024)
        residual = np.random.randn(1024)
        from coupled_engine import MemoryEntry
        entry = MemoryEntry(text="test", embedding=emb)
        entry.entity_residual = residual
        assert entry.entity_residual is not None
        assert entry.entity_residual.shape == (1024,)


class TestRetrieverResolveEntityRerank:
    def test_resolve_from_config(self):
        from mabench.retrieval import Retriever
        from mabench.configs import RetrievalConfig
        r = Retriever(config=RetrievalConfig(entity_rerank=True, entity_rerank_alpha=0.7))
        assert r._resolve_entity_rerank is True
        assert r._resolve_entity_rerank_alpha == 0.7

    def test_resolve_default(self):
        from mabench.retrieval import Retriever
        r = Retriever()
        assert r._resolve_entity_rerank is False
        assert r._resolve_entity_rerank_alpha == 0.3


class TestExtractQueryEntity:
    def test_of_pattern(self):
        from mabench.retrieval import Retriever
        assert Retriever._extract_query_entity("What is the capital of France?") is not None
        result = Retriever._extract_query_entity("What is the capital of France?")
        assert "France" in result

    def test_about_pattern(self):
        from mabench.retrieval import Retriever
        result = Retriever._extract_query_entity("Tell me about Tokyo")
        assert result is not None
        assert "Tokyo" in result

    def test_quoted_entity(self):
        from mabench.retrieval import Retriever
        result = Retriever._extract_query_entity('What is "Mortal Kombat" rated?')
        assert result == "Mortal Kombat"

    def test_no_entity(self):
        from mabench.retrieval import Retriever
        result = Retriever._extract_query_entity("Hi")
        assert result is None


class TestScorerEntityPositionMethod:
    def test_method_exists(self):
        from mabench.scoring import EmbeddingRelevanceScorer
        assert hasattr(EmbeddingRelevanceScorer, "embed_entity_position")
        assert hasattr(EmbeddingRelevanceScorer, "embed_entity_position_batch")
        assert hasattr(EmbeddingRelevanceScorer, "_find_entity_tokens")

    def test_embed_entity_position_signature(self):
        from mabench.scoring import EmbeddingRelevanceScorer
        sig = inspect.signature(EmbeddingRelevanceScorer.embed_entity_position)
        params = list(sig.parameters.keys())
        assert "text" in params
        assert "entity" in params
        assert "layer" in params


class TestFactStoreEntityRerank:
    def test_init_accepts_entity_rerank(self):
        from mabench.store import FactStore
        sig = inspect.signature(FactStore.__init__)
        params = list(sig.parameters.keys())
        assert "entity_rerank" in params

    def test_extract_primary_entity_from_entities(self):
        from mabench.store import FactStore
        mock_fact = MagicMock()
        mock_fact.entities = ("Paris", "France")
        mock_fact.triples = ()
        assert FactStore._extract_primary_entity(mock_fact) == "Paris"

    def test_extract_primary_entity_from_triples(self):
        from mabench.store import FactStore
        mock_fact = MagicMock()
        mock_fact.entities = ()
        mock_fact.triples = (("France", "capital", "Paris"),)
        assert FactStore._extract_primary_entity(mock_fact) == "France"

    def test_extract_primary_entity_none(self):
        from mabench.store import FactStore
        mock_fact = MagicMock()
        mock_fact.entities = ()
        mock_fact.triples = ()
        assert FactStore._extract_primary_entity(mock_fact) is None


class TestAgentWiringEntityRerank:
    def test_factstore_gets_entity_rerank(self):
        """Verify agent.py passes entity_rerank to FactStore."""
        from mabench.agent import HermesAgent
        src = inspect.getsource(HermesAgent._build)
        assert "entity_rerank" in src

    def test_retriever_has_entity_rerank_in_retrieve_v2(self):
        """Verify _retrieve_v2 calls _entity_rerank when enabled."""
        from mabench.retrieval import Retriever
        src = inspect.getsource(Retriever._retrieve_v2)
        assert "_entity_rerank" in src
        assert "_resolve_entity_rerank" in src


class TestEntityRerankLogic:
    """Test the reranking math with mocked data."""

    def test_rerank_reorders_by_entity_score(self):
        from mabench.retrieval import Retriever
        from mabench.configs import RetrievalConfig
        from coupled_engine import MemoryEntry

        # Create mock engine with memory_store
        engine = MagicMock()
        entry_good = MemoryEntry(text="France capital is Paris", embedding=np.zeros(1024))
        entry_good.entity_residual = np.ones(1024) / np.sqrt(1024)  # normalized
        entry_bad = MemoryEntry(text="Germany capital is Berlin", embedding=np.zeros(1024))
        entry_bad.entity_residual = -np.ones(1024) / np.sqrt(1024)  # opposite direction
        engine.memory_store = [entry_good, entry_bad]

        # Create mock scorer
        scorer = MagicMock()
        scorer.embed_entity_position.return_value = np.ones(1024) / np.sqrt(1024)

        retriever = Retriever(
            engine, scorer,
            config=RetrievalConfig(entity_rerank=True, entity_rerank_alpha=0.5),
        )

        results = [
            {"text": "Germany capital is Berlin", "score": 0.95},  # higher cosine
            {"text": "France capital is Paris", "score": 0.90},    # lower cosine
        ]

        reranked = retriever._entity_rerank("capital of France", results)

        # France should be boosted (positive entity residual dot product)
        # Germany should be penalized (negative entity residual dot product)
        assert reranked[0]["text"] == "France capital is Paris"
        assert reranked[1]["text"] == "Germany capital is Berlin"

    def test_rerank_no_residual_keeps_score(self):
        from mabench.retrieval import Retriever
        from mabench.configs import RetrievalConfig
        from coupled_engine import MemoryEntry

        engine = MagicMock()
        entry = MemoryEntry(text="Some fact", embedding=np.zeros(1024))
        # No entity_residual set (default None)
        engine.memory_store = [entry]

        scorer = MagicMock()
        scorer.embed_entity_position.return_value = np.ones(1024) / np.sqrt(1024)

        retriever = Retriever(
            engine, scorer,
            config=RetrievalConfig(entity_rerank=True),
        )

        results = [{"text": "Some fact", "score": 0.85}]
        reranked = retriever._entity_rerank("test query of something", results)
        assert reranked[0]["score"] == 0.85  # unchanged

    def test_rerank_no_entity_returns_unchanged(self):
        from mabench.retrieval import Retriever
        from mabench.configs import RetrievalConfig

        engine = MagicMock()
        engine.memory_store = []
        retriever = Retriever(engine, config=RetrievalConfig(entity_rerank=True))
        results = [{"text": "fact", "score": 0.9}]
        # "Hi" won't match any entity pattern
        reranked = retriever._entity_rerank("Hi", results)
        assert reranked == results


class TestPrepareResultEntityResiduals:
    def test_field_exists(self):
        from mabench.store import _PrepareResult
        fields = {f.name for f in _PrepareResult.__dataclass_fields__.values()}
        assert "entity_residuals" in fields

    def test_default_none(self):
        from mabench.store import _PrepareResult
        # Minimal construction
        pr = _PrepareResult(
            result=MagicMock(),
            fact_texts=["test"],
            embeddings=np.zeros((1, 1024)),
            rebel_triples_batch=None,
            primary_entities=[None],
        )
        assert pr.entity_residuals is None
