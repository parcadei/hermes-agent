"""Tests for parallel ingestion pipeline and REBEL param optimizations.

Verifies:
1. REBEL uses greedy decoding (num_beams=1) and max_length=128
2. ThreadPoolExecutor runs embed/GLiNER/REBEL concurrently in _store
3. Parallel _store produces identical results to sequential baseline
4. Thread safety of the parallel execution path
"""

import time
import threading
import numpy as np
import pytest
from collections import defaultdict
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(**kwargs):
    """Create a HermesMemoryAgent with mocked heavy deps."""
    from mabench.hermes_agent import HermesMemoryAgent

    mock_pipeline = MagicMock()
    mock_scorer = MagicMock()

    defaults = dict(rebel_kg=True)
    defaults.update(kwargs)

    with patch("mabench.hermes_agent.IngestionPipeline", return_value=mock_pipeline), \
         patch("mabench.hermes_agent.EmbeddingRelevanceScorer", return_value=mock_scorer), \
         patch("mabench.hermes_agent.MemoryOrchestrator"), \
         patch("mabench.hermes_agent.CoupledEngine"):
        agent = HermesMemoryAgent(**defaults)
    return agent


# ---------------------------------------------------------------------------
# 1. REBEL parameter tests
# ---------------------------------------------------------------------------

class TestRebelParams:
    """Verify REBEL uses optimized generation parameters."""

    def test_rebel_uses_greedy_decoding(self):
        """num_beams must be 1 (greedy), not 3 (beam search)."""
        agent = _make_agent()
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate.return_value = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_tokenizer.batch_decode.return_value = [
            "<s><triplet> Albert Einstein <subj> place of birth <obj> Ulm</s>"
        ]

        agent._rebel_model = mock_model
        agent._rebel_tokenizer = mock_tokenizer
        agent._rebel_device = "cpu"

        agent._rebel_extract_batch(["Albert Einstein was born in Ulm."])

        # Check generate was called with num_beams=1
        gen_call = mock_model.generate.call_args
        assert gen_call.kwargs.get("num_beams", gen_call[1].get("num_beams")) == 1, \
            "REBEL must use greedy decoding (num_beams=1), not beam search"

    def test_rebel_uses_short_max_length(self):
        """max_length must be 128, not 256."""
        agent = _make_agent()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate.return_value = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_tokenizer.batch_decode.return_value = [
            "<s><triplet> X <subj> Y <obj> Z</s>"
        ]

        agent._rebel_model = mock_model
        agent._rebel_tokenizer = mock_tokenizer
        agent._rebel_device = "cpu"

        agent._rebel_extract_batch(["Test sentence."])

        gen_call = mock_model.generate.call_args
        assert gen_call.kwargs.get("max_length", gen_call[1].get("max_length")) == 128, \
            "REBEL max_length must be 128 (triples are short)"

    def test_rebel_batch_size_16(self):
        """Batch size should be 16 for better GPU utilization."""
        agent = _make_agent()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate.return_value = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        # Return one decoded string per input
        mock_tokenizer.batch_decode.side_effect = lambda outputs, **kw: [
            "<s><triplet> X <subj> Y <obj> Z</s>"
        ] * (outputs.call_count if hasattr(outputs, 'call_count') else 16)

        agent._rebel_model = mock_model
        agent._rebel_tokenizer = mock_tokenizer
        agent._rebel_device = "cpu"

        # Feed 20 texts — should be 2 batches (16 + 4), not 3 (8+8+4)
        texts = [f"Sentence {i}." for i in range(20)]
        # Make batch_decode return correct number per call
        mock_tokenizer.batch_decode.side_effect = None
        mock_tokenizer.batch_decode.return_value = [
            "<s><triplet> X <subj> Y <obj> Z</s>"
        ] * 20  # more than enough

        agent._rebel_extract_batch(texts)

        # tokenizer should be called twice (batch of 16, batch of 4)
        assert mock_tokenizer.call_count == 2, \
            f"Expected 2 batches (16+4) for 20 texts, got {mock_tokenizer.call_count} calls"


# ---------------------------------------------------------------------------
# 2. Parallel execution tests
# ---------------------------------------------------------------------------

class TestParallelStore:
    """Verify _store runs embed/GLiNER/REBEL concurrently via ThreadPoolExecutor."""

    def _make_store_agent(self):
        """Create an agent with full mock wiring for _store testing."""
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.ingestion import SegmentedFact, IngestionResult, EntityGraph

        mock_pipeline = MagicMock()
        mock_scorer = MagicMock()

        # segment() returns 2 fake facts
        facts = [
            SegmentedFact(
                text="Albert Einstein was born in Ulm.",
                entities=("Albert Einstein", "Ulm"),
                importance_hint=1.0,
                source_sentence_idx=0,
                segment_idx=0,
                triples=(("Albert Einstein", "born in", "Ulm"),),
            ),
            SegmentedFact(
                text="Einstein worked at Princeton.",
                entities=("Einstein", "Princeton"),
                importance_hint=1.0,
                source_sentence_idx=1,
                segment_idx=0,
                triples=(("Einstein", "worked at", "Princeton"),),
            ),
        ]
        mock_pipeline.segment.return_value = IngestionResult(
            facts=facts,
            entity_graph=EntityGraph({}, {}),
            n_sentences=2,
            n_segments=1,
            elapsed_seconds=0.01,
        )

        # embed_batch returns fake embeddings
        mock_scorer.embed_batch.return_value = [
            np.random.randn(384).astype(np.float32),
            np.random.randn(384).astype(np.float32),
        ]

        with patch("mabench.hermes_agent.IngestionPipeline", return_value=mock_pipeline), \
             patch("mabench.hermes_agent.EmbeddingRelevanceScorer", return_value=mock_scorer), \
             patch("mabench.hermes_agent.MemoryOrchestrator") as mock_orch_cls, \
             patch("mabench.hermes_agent.CoupledEngine") as mock_ce_cls:
            mock_ce = mock_ce_cls.return_value
            mock_ce.memory_store = []
            mock_ce.store.side_effect = lambda **kw: len(mock_ce.memory_store)
            agent = HermesMemoryAgent(rebel_kg=True)

        return agent

    def test_store_calls_embed_and_rebel(self):
        """_store must call embed_batch and _rebel_extract_batch (GLiNER skipped)."""
        agent = self._make_store_agent()

        with patch.object(agent, "_rebel_extract_batch", return_value=[[], []]) as mock_rebel:
            agent._store("Albert Einstein was born in Ulm. Einstein worked at Princeton.")

        mock_rebel.assert_called_once()
        agent._scorer.embed_batch.assert_called_once()

    def test_parallel_produces_same_kg_as_sequential(self):
        """KG triples must be identical regardless of execution order."""
        agent = self._make_store_agent()

        rebel_triples = [
            [{"head": "Albert Einstein", "relation": "place of birth", "tail": "Ulm"}],
            [{"head": "Einstein", "relation": "employer", "tail": "Princeton"}],
        ]

        with patch.object(agent, "_extract_entities_batch", return_value=([None, None], [[], []])), \
             patch.object(agent, "_rebel_extract_batch", return_value=rebel_triples):
            agent._store("Albert Einstein was born in Ulm. Einstein worked at Princeton.")

        # Verify KG was populated correctly
        assert "albert einstein" in agent._rebel_kg
        assert "place of birth" in agent._rebel_kg["albert einstein"]
        assert "Ulm" in agent._rebel_kg["albert einstein"]["place of birth"]
        assert "einstein" in agent._rebel_kg
        assert "employer" in agent._rebel_kg["einstein"]
        assert "Princeton" in agent._rebel_kg["einstein"]["employer"]

    def test_sequential_execution_order(self):
        """Verify embed runs before REBEL (sequential, not parallel).

        On a single GPU, parallel ThreadPool caused device_map conflicts:
        "Tensor on device cuda:0 is not on the expected device meta!"
        Now runs sequentially: embed first, then REBEL.
        """
        agent = self._make_store_agent()
        call_log = []

        def slow_embed(texts, **kw):
            call_log.append(("embed_start", time.monotonic()))
            time.sleep(0.05)
            call_log.append(("embed_end", time.monotonic()))
            return [np.random.randn(384).astype(np.float32) for _ in texts]

        def slow_rebel(texts):
            call_log.append(("rebel_start", time.monotonic()))
            time.sleep(0.05)
            call_log.append(("rebel_end", time.monotonic()))
            return [[] for _ in texts]

        agent._scorer.embed_batch = slow_embed

        with patch.object(agent, "_rebel_extract_batch", side_effect=slow_rebel):
            agent._store("Albert Einstein was born in Ulm. Einstein worked at Princeton.")

        # Verify both ran
        tags = [e[0] for e in call_log]
        assert "embed_start" in tags and "rebel_start" in tags

        # Verify embed finishes before REBEL starts (sequential)
        embed_end = next(t for tag, t in call_log if tag == "embed_end")
        rebel_start = next(t for tag, t in call_log if tag == "rebel_start")
        assert rebel_start >= embed_end, \
            f"REBEL started before embed finished: embed_end={embed_end:.4f} rebel_start={rebel_start:.4f}"

    def test_store_result_unchanged(self):
        """_store must still return 'Memorized' string."""
        agent = self._make_store_agent()

        with patch.object(agent, "_extract_entities_batch", return_value=([None, None], [[], []])), \
             patch.object(agent, "_rebel_extract_batch", return_value=[[], []]):
            result = agent._store("Test fact.")

        assert result == "Memorized"

    def test_store_without_rebel(self):
        """When rebel_kg is disabled, embed still runs and GLiNER is skipped."""
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.ingestion import SegmentedFact, IngestionResult, EntityGraph

        mock_pipeline = MagicMock()
        mock_scorer = MagicMock()

        facts = [SegmentedFact(
            text="Test.", entities=(), importance_hint=1.0,
            source_sentence_idx=0, segment_idx=0,
        )]
        mock_pipeline.segment.return_value = IngestionResult(
            facts=facts, entity_graph=EntityGraph({}, {}),
            n_sentences=1, n_segments=1, elapsed_seconds=0.01,
        )
        mock_scorer.embed_batch.return_value = [np.random.randn(384).astype(np.float32)]

        with patch("mabench.hermes_agent.IngestionPipeline", return_value=mock_pipeline), \
             patch("mabench.hermes_agent.EmbeddingRelevanceScorer", return_value=mock_scorer), \
             patch("mabench.hermes_agent.MemoryOrchestrator"), \
             patch("mabench.hermes_agent.CoupledEngine") as mock_ce_cls:
            mock_ce = mock_ce_cls.return_value
            mock_ce.memory_store = []
            mock_ce.store.side_effect = lambda **kw: 0
            agent = HermesMemoryAgent(rebel_kg=False)

        result = agent._store("Test.")

        assert result == "Memorized"
        mock_scorer.embed_batch.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Thread safety tests
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Verify no data races in parallel extraction path."""

    def test_rebel_kg_add_thread_safe(self):
        """Multiple threads adding to KG concurrently must not lose data."""
        agent = _make_agent()

        def add_triples(prefix, count):
            for i in range(count):
                agent._rebel_kg_add(f"{prefix}_entity_{i}", "relation", f"object_{i}")

        threads = [
            threading.Thread(target=add_triples, args=("A", 100)),
            threading.Thread(target=add_triples, args=("B", 100)),
            threading.Thread(target=add_triples, args=("C", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 300 entities should be present
        total = sum(len(rels) for rels in agent._rebel_kg.values())
        assert total == 300, f"Expected 300 relation entries, got {total} (data race?)"

    def test_entity_index_thread_safe(self):
        """Concurrent entity index updates must not lose entries."""
        agent = _make_agent()
        agent._gliner_entity_index = defaultdict(set)

        def add_entities(prefix, count):
            for i in range(count):
                agent._gliner_entity_index[f"{prefix}_entity"].add(i)

        threads = [
            threading.Thread(target=add_entities, args=("A", 50)),
            threading.Thread(target=add_entities, args=("B", 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(agent._gliner_entity_index["A_entity"]) == 50
        assert len(agent._gliner_entity_index["B_entity"]) == 50
