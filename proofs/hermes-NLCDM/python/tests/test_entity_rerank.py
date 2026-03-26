"""Tests for EntityReranker module (arXiv:2603.19664).

Tests the standalone EntityReranker module:
  1. Config: entity_rerank flag on RetrievalConfig
  2. MemoryEntry: entity_residual field
  3. EntityReranker: entity extraction, residual computation, gating, rerank
  4. Agent wiring: EntityReranker constructed and passed to store + retriever
  5. Integration: annotate_entry + rerank pipeline
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _TESTS_DIR.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))


# ── Config tests ─────────────────────────────────────────────────

class TestRetrievalConfigEntityRerank:
    def test_entity_rerank_default_false(self):
        from mabench.configs import RetrievalConfig
        assert RetrievalConfig().entity_rerank is False

    def test_entity_rerank_enabled(self):
        from mabench.configs import RetrievalConfig
        cfg = RetrievalConfig(entity_rerank=True)
        assert cfg.entity_rerank is True

    def test_frozen(self):
        from mabench.configs import RetrievalConfig
        cfg = RetrievalConfig(entity_rerank=True)
        with pytest.raises(AttributeError):
            cfg.entity_rerank = False


# ── MemoryEntry tests ────────────────────────────────────────────

class TestMemoryEntryEntityResidual:
    def test_field_exists(self):
        from coupled_engine import MemoryEntry
        assert "entity_residual" in MemoryEntry.__dataclass_fields__

    def test_default_none(self):
        from coupled_engine import MemoryEntry
        entry = MemoryEntry(text="test", embedding=np.zeros(1024))
        assert entry.entity_residual is None

    def test_set_residual(self):
        from coupled_engine import MemoryEntry
        entry = MemoryEntry(text="test", embedding=np.zeros(1024))
        residual = np.random.randn(1024)
        entry.entity_residual = residual
        assert entry.entity_residual is not None
        assert entry.entity_residual.shape == (1024,)


# ── Module structure tests ───────────────────────────────────────

class TestEntityRerankerStructure:
    def test_module_exists(self):
        from mabench.entity_reranker import EntityReranker
        assert EntityReranker is not None

    def test_standalone_class(self):
        from mabench.entity_reranker import EntityReranker
        assert not any(
            "FactStore" in str(b) or "Retriever" in str(b)
            for b in EntityReranker.__mro__
        )

    def test_init_signature(self):
        from mabench.entity_reranker import EntityReranker
        sig = inspect.signature(EntityReranker.__init__)
        params = list(sig.parameters.keys())
        assert "scorer" in params
        assert "nlp" in params

    def test_has_annotate_entry(self):
        from mabench.entity_reranker import EntityReranker
        assert hasattr(EntityReranker, "annotate_entry")

    def test_has_rerank(self):
        from mabench.entity_reranker import EntityReranker
        assert hasattr(EntityReranker, "rerank")

    def test_gate_is_classmethod(self):
        from mabench.entity_reranker import EntityReranker
        assert isinstance(
            inspect.getattr_static(EntityReranker, "_gate"), classmethod
        )


class TestNoInlineEntityCode:
    """Verify entity reranking logic lives in EntityReranker, not inline."""

    def test_store_accepts_entity_reranker(self):
        from mabench.store import FactStore
        sig = inspect.signature(FactStore.__init__)
        assert "entity_reranker" in sig.parameters

    def test_store_no_extract_primary_entity(self):
        from mabench.store import FactStore
        assert not hasattr(FactStore, "_extract_primary_entity")

    def test_store_no_spacy_extract_entity(self):
        from mabench.store import FactStore
        assert not hasattr(FactStore, "_spacy_extract_entity")

    def test_retriever_accepts_entity_reranker(self):
        from mabench.retrieval import Retriever
        sig = inspect.signature(Retriever.__init__)
        assert "entity_reranker" in sig.parameters

    def test_retriever_no_entity_rerank_method(self):
        from mabench.retrieval import Retriever
        assert not hasattr(Retriever, "_entity_rerank")

    def test_retriever_no_extract_query_entity(self):
        from mabench.retrieval import Retriever
        assert not hasattr(Retriever, "_extract_query_entity")


# ── Entity extraction tests ──────────────────────────────────────

class TestEntityExtraction:
    def _make_nlp(self, entity_text="France", label="GPE"):
        nlp = MagicMock()
        ent = MagicMock()
        ent.text = entity_text
        ent.label_ = label
        doc = MagicMock()
        doc.ents = [ent]
        nlp.return_value = doc
        return nlp

    def _make_fact(self, text="", entities=(), triples=()):
        fact = MagicMock()
        fact.text = text
        fact.entities = entities
        fact.triples = triples
        return fact

    def test_extract_from_fact_entities(self):
        from mabench.entity_reranker import EntityReranker
        er = EntityReranker(scorer=MagicMock())
        fact = self._make_fact(entities=("Paris", "France"))
        assert er._extract_entity_from_fact(fact) == "Paris"

    def test_extract_from_fact_triples(self):
        from mabench.entity_reranker import EntityReranker
        er = EntityReranker(scorer=MagicMock())
        fact = self._make_fact(triples=(("France", "capital", "Paris"),))
        assert er._extract_entity_from_fact(fact) == "France"

    def test_extract_from_fact_spacy_fallback(self):
        from mabench.entity_reranker import EntityReranker
        nlp = self._make_nlp("France")
        er = EntityReranker(scorer=MagicMock(), nlp=nlp)
        fact = self._make_fact(text="The capital of France is Paris")
        result = er._extract_entity_from_fact(fact)
        nlp.assert_called_once_with("The capital of France is Paris")
        assert result == "France"

    def test_extract_from_fact_none_without_nlp(self):
        from mabench.entity_reranker import EntityReranker
        er = EntityReranker(scorer=MagicMock())
        fact = self._make_fact(text="The capital of France is Paris")
        assert er._extract_entity_from_fact(fact) is None

    def test_extract_query_entity(self):
        from mabench.entity_reranker import EntityReranker
        nlp = self._make_nlp("Christian Abbiati", "PERSON")
        er = EntityReranker(scorer=MagicMock(), nlp=nlp)
        assert er._extract_query_entity("What sport did Christian Abbiati play?") == "Christian Abbiati"

    def test_extract_query_entity_no_nlp(self):
        from mabench.entity_reranker import EntityReranker
        er = EntityReranker(scorer=MagicMock())
        assert er._extract_query_entity("What is the capital?") is None

    def test_extract_prefers_person(self):
        from mabench.entity_reranker import EntityReranker
        nlp = MagicMock()
        gpe_ent = MagicMock(); gpe_ent.text = "Milan"; gpe_ent.label_ = "GPE"
        per_ent = MagicMock(); per_ent.text = "Abbiati"; per_ent.label_ = "PERSON"
        doc = MagicMock()
        doc.ents = [gpe_ent, per_ent]
        nlp.return_value = doc
        er = EntityReranker(scorer=MagicMock(), nlp=nlp)
        assert er._extract_query_entity("What sport did Abbiati play in Milan?") == "Abbiati"


# ── Multiplicative gating tests ──────────────────────────────────

class TestMultiplicativeGating:
    def test_gate_match(self):
        from mabench.entity_reranker import EntityReranker
        assert EntityReranker._gate(0.99) == 1.0
        assert EntityReranker._gate(0.96) == 1.0

    def test_gate_ambiguous(self):
        from mabench.entity_reranker import EntityReranker
        assert EntityReranker._gate(0.90) == 0.5
        assert EntityReranker._gate(0.85) == 0.5
        assert EntityReranker._gate(0.81) == 0.5

    def test_gate_mismatch(self):
        from mabench.entity_reranker import EntityReranker
        assert EntityReranker._gate(0.79) == 0.2
        assert EntityReranker._gate(0.50) == 0.2
        assert EntityReranker._gate(0.0) == 0.2

    def test_gate_boundaries(self):
        from mabench.entity_reranker import EntityReranker
        # >0.95 is match, =0.95 is ambiguous
        assert EntityReranker._gate(0.95) == 0.5
        assert EntityReranker._gate(0.9501) == 1.0
        # >0.80 is ambiguous, =0.80 is mismatch
        assert EntityReranker._gate(0.80) == 0.2
        assert EntityReranker._gate(0.8001) == 0.5

    def test_gate_constants(self):
        from mabench.entity_reranker import EntityReranker
        assert EntityReranker.GATE_HIGH == 0.95
        assert EntityReranker.GATE_LOW == 0.80
        assert EntityReranker.GATE_MATCH == 1.0
        assert EntityReranker.GATE_AMBIGUOUS == 0.5
        assert EntityReranker.GATE_MISMATCH == 0.2


# ── Annotation tests ────────────────────────────────────────────

class TestAnnotation:
    def test_annotate_sets_residual(self):
        from mabench.entity_reranker import EntityReranker
        from coupled_engine import MemoryEntry

        residual = np.ones(1024) / np.sqrt(1024)
        scorer = MagicMock()
        scorer.embed_entity_isolated.return_value = residual

        er = EntityReranker(scorer)
        entry = MemoryEntry(text="France capital is Paris", embedding=np.zeros(1024))
        fact = MagicMock()
        fact.entities = ("France",)
        fact.triples = ()

        er.annotate_entry(entry, fact)
        assert entry.entity_residual is not None
        np.testing.assert_array_equal(entry.entity_residual, residual)
        scorer.embed_entity_isolated.assert_called_once_with("France")

    def test_annotate_no_entity_leaves_none(self):
        from mabench.entity_reranker import EntityReranker
        from coupled_engine import MemoryEntry

        er = EntityReranker(scorer=MagicMock())
        entry = MemoryEntry(text="test", embedding=np.zeros(1024))
        fact = MagicMock()
        fact.entities = ()
        fact.triples = ()
        fact.text = ""

        er.annotate_entry(entry, fact)
        assert entry.entity_residual is None

    def test_annotate_caches_entity_residual(self):
        from mabench.entity_reranker import EntityReranker
        from coupled_engine import MemoryEntry

        scorer = MagicMock()
        scorer.embed_entity_isolated.return_value = np.ones(1024) / np.sqrt(1024)

        er = EntityReranker(scorer)
        for i in range(3):
            entry = MemoryEntry(text=f"fact about France #{i}", embedding=np.zeros(1024))
            fact = MagicMock()
            fact.entities = ("France",)
            fact.triples = ()
            er.annotate_entry(entry, fact)

        # embed_entity_isolated called only ONCE (cached by entity string)
        scorer.embed_entity_isolated.assert_called_once_with("France")

    def test_annotate_populates_text_to_residual(self):
        from mabench.entity_reranker import EntityReranker
        from coupled_engine import MemoryEntry

        scorer = MagicMock()
        scorer.embed_entity_isolated.return_value = np.ones(1024) / np.sqrt(1024)

        er = EntityReranker(scorer)
        entry = MemoryEntry(text="The capital of France is Paris", embedding=np.zeros(1024))
        fact = MagicMock()
        fact.entities = ("France",)
        fact.triples = ()

        er.annotate_entry(entry, fact)
        assert "The capital of France is Paris" in er._text_to_residual


# ── Reranking logic tests ────────────────────────────────────────

class TestRerankLogic:
    def _make_reranker(self, query_entity="France"):
        from mabench.entity_reranker import EntityReranker

        scorer = MagicMock()
        nlp = MagicMock()

        ent = MagicMock()
        ent.text = query_entity
        ent.label_ = "GPE"
        doc = MagicMock()
        doc.ents = [ent]
        nlp.return_value = doc

        france_residual = np.ones(1024) / np.sqrt(1024)
        germany_residual = np.zeros(1024)
        germany_residual[0] = 1.0  # orthogonal-ish to France

        def mock_embed(entity_text):
            if entity_text == "France":
                return france_residual
            return germany_residual

        scorer.embed_entity_isolated.side_effect = mock_embed

        reranker = EntityReranker(scorer, nlp=nlp)
        # Pre-populate text_to_residual (simulates ingestion)
        reranker._text_to_residual["France capital is Paris"] = france_residual
        reranker._text_to_residual["Germany capital is Berlin"] = germany_residual

        return reranker

    def test_rerank_reorders_by_gate(self):
        reranker = self._make_reranker("France")
        engine = MagicMock()
        engine.memory_store = []

        results = [
            {"text": "Germany capital is Berlin", "score": 0.95},
            {"text": "France capital is Paris", "score": 0.90},
        ]
        reranked = reranker.rerank("What is the capital of France?", results, engine)

        # France (gate=1.0): 0.90 * 1.0 = 0.90
        # Germany (gate=0.2): 0.95 * 0.2 = 0.19
        assert reranked[0]["text"] == "France capital is Paris"
        assert reranked[1]["text"] == "Germany capital is Berlin"

    def test_rerank_adds_entity_fields(self):
        reranker = self._make_reranker("France")
        engine = MagicMock()
        engine.memory_store = []

        results = [{"text": "France capital is Paris", "score": 0.90}]
        reranked = reranker.rerank("capital of France", results, engine)
        assert "entity_gate" in reranked[0]
        assert "entity_cos" in reranked[0]

    def test_rerank_no_residual_keeps_score(self):
        from mabench.entity_reranker import EntityReranker

        scorer = MagicMock()
        scorer.embed_entity_isolated.return_value = np.ones(1024) / np.sqrt(1024)
        nlp = MagicMock()
        ent = MagicMock()
        ent.text = "France"
        ent.label_ = "GPE"
        doc = MagicMock()
        doc.ents = [ent]
        nlp.return_value = doc

        reranker = EntityReranker(scorer, nlp=nlp)
        engine = MagicMock()
        engine.memory_store = []

        results = [{"text": "Unknown fact", "score": 0.85}]
        reranked = reranker.rerank("capital of France", results, engine)
        assert reranked[0]["score"] == 0.85
        assert reranked[0]["entity_gate"] == 1.0

    def test_rerank_no_entity_returns_unchanged(self):
        from mabench.entity_reranker import EntityReranker

        nlp = MagicMock()
        doc = MagicMock()
        doc.ents = []
        nlp.return_value = doc

        reranker = EntityReranker(scorer=MagicMock(), nlp=nlp)
        results = [{"text": "fact", "score": 0.9}]
        reranked = reranker.rerank("Hello", results, MagicMock())
        assert reranked == results

    def test_rerank_no_nlp_returns_unchanged(self):
        from mabench.entity_reranker import EntityReranker

        reranker = EntityReranker(scorer=MagicMock())
        results = [{"text": "fact", "score": 0.9}]
        reranked = reranker.rerank("What is the capital?", results, MagicMock())
        assert reranked == results

    def test_rerank_fallback_to_engine(self):
        """When text_to_residual misses, falls back to engine.memory_store."""
        from mabench.entity_reranker import EntityReranker
        from coupled_engine import MemoryEntry

        france_residual = np.ones(1024) / np.sqrt(1024)
        scorer = MagicMock()
        scorer.embed_entity_isolated.return_value = france_residual

        nlp = MagicMock()
        ent = MagicMock()
        ent.text = "France"
        ent.label_ = "GPE"
        doc = MagicMock()
        doc.ents = [ent]
        nlp.return_value = doc

        reranker = EntityReranker(scorer, nlp=nlp)

        entry = MemoryEntry(text="France capital is Paris", embedding=np.zeros(1024))
        entry.entity_residual = france_residual

        engine = MagicMock()
        engine.memory_store = [entry]

        results = [{"text": "France capital is Paris", "score": 0.90}]
        reranked = reranker.rerank("capital of France", results, engine)

        assert reranked[0]["entity_cos"] is not None
        assert reranked[0]["entity_gate"] == 1.0
        assert reranked[0]["score"] == pytest.approx(0.90)


# ── Scorer method tests ──────────────────────────────────────────

class TestScorerEntityMethods:
    def test_embed_entity_isolated_exists(self):
        from mabench.scoring import EmbeddingRelevanceScorer
        assert hasattr(EmbeddingRelevanceScorer, "embed_entity_isolated")

    def test_embed_entity_isolated_signature(self):
        from mabench.scoring import EmbeddingRelevanceScorer
        sig = inspect.signature(EmbeddingRelevanceScorer.embed_entity_isolated)
        params = list(sig.parameters.keys())
        assert "entity_text" in params
        assert "layer" in params

    def test_embed_entity_position_exists(self):
        from mabench.scoring import EmbeddingRelevanceScorer
        assert hasattr(EmbeddingRelevanceScorer, "embed_entity_position")

    def test_find_entity_tokens_exists(self):
        from mabench.scoring import EmbeddingRelevanceScorer
        assert hasattr(EmbeddingRelevanceScorer, "_find_entity_tokens")


# ── Agent wiring tests ──────────────────────────────────────────

class TestAgentWiring:
    def test_agent_imports_entity_reranker(self):
        from mabench import agent
        src = inspect.getsource(agent)
        assert "EntityReranker" in src

    def test_build_creates_entity_reranker(self):
        from mabench.agent import HermesAgent
        src = inspect.getsource(HermesAgent._build)
        assert "EntityReranker(" in src

    def test_build_passes_to_retriever(self):
        from mabench.agent import HermesAgent
        src = inspect.getsource(HermesAgent._build)
        assert "entity_reranker=ent_reranker" in src

    def test_build_passes_to_store(self):
        from mabench.agent import HermesAgent
        src = inspect.getsource(HermesAgent._build)
        assert "entity_reranker=ent_reranker" in src

    def test_store_single_fact_calls_annotate(self):
        from mabench.store import FactStore
        src = inspect.getsource(FactStore._store_single_fact)
        assert "_entity_reranker" in src
        assert ".annotate_entry(" in src

    def test_retrieve_v2_calls_rerank(self):
        from mabench.retrieval import Retriever
        src = inspect.getsource(Retriever._retrieve_v2)
        assert "_entity_reranker" in src
        assert ".rerank(" in src
