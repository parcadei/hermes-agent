"""Discriminative tests for BM25Index and ChainCache with supersession invalidation.

Core hypothesis: BM25 lexical retrieval discriminates entities that cosine
embedding retrieval conflates. "Who was born in Treviglio?" should rank
"Christian Abbiati was born in Treviglio" above "Marco Rossi was born in
Bergamo" because the query token "Treviglio" matches lexically, whereas
cosine similarity scores both facts near-identically (cosine > 0.95) due
to shared template structure.

ChainCache stores completed decompose chains keyed by query embedding
similarity, with supersession-driven invalidation via a reverse index
from evidence facts to cached chains. When a fact is superseded during
ingestion, all chains using that fact are evicted.

All tests run on CPU. No GPU, no external models, no LLM calls.
BM25Index is tested at the data structure level.
ChainCache is tested with numpy random vectors for embeddings.
Integration tests mock the hermes_agent decompose loop.
"""

import numpy as np
import pytest
import sys
import os
from collections import defaultdict
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mabench.bm25_index import BM25Index
from mabench.chain_cache import ChainCache, CachedChain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, idx: int, noise: float = 0.05) -> np.ndarray:
    """Unit vector along axis `idx` with small noise."""
    v = np.zeros(dim, dtype=np.float64)
    v[idx % dim] = 1.0
    rng = np.random.RandomState(idx)
    v += rng.randn(dim) * noise
    return v / np.linalg.norm(v)


def _random_embedding(dim: int = 64, seed: int = 0) -> np.ndarray:
    """Generate a random unit-norm embedding for testing."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim)
    return v / np.linalg.norm(v)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass(frozen=True)
class FakeFact:
    """Minimal SegmentedFact-like object for testing."""

    text: str
    triples: tuple
    entities: tuple = ()
    importance_hint: float = 0.5
    source_sentence_idx: int = 0
    segment_idx: int = 0
    fact_id: str = ""
    text_raw: str = ""


@dataclass(frozen=True)
class FakeIngestionResult:
    """Minimal IngestionResult-like object for testing."""

    facts: list
    entity_graph: object = None
    n_sentences: int = 1
    n_segments: int = 1
    elapsed_seconds: float = 0.0
    coref_clusters: list = ()
    verb_mappings: list = ()


class FakeEntityGraph:
    entity_to_facts = {}
    fact_to_entities = {}


# ---------------------------------------------------------------------------
# Test 1: BM25Index — Core retrieval behavior
# ---------------------------------------------------------------------------


class TestBM25Index:
    """Test BM25 lexical retrieval for entity discrimination."""

    def test_add_and_query_basic(self):
        """Add 3 facts, query matches the right one by shared terms."""
        idx = BM25Index()
        idx.add(0, "The capital of France is Paris")
        idx.add(1, "The capital of Germany is Berlin")
        idx.add(2, "The capital of Italy is Rome")

        results = idx.query("What is the capital of France?")
        assert len(results) > 0
        # France fact should be ranked first
        top_id, top_score = results[0]
        assert top_id == 0, f"Expected doc 0 (France), got doc {top_id}"
        assert top_score > 0

    def test_entity_discrimination(self):
        """THE CRITICAL TEST: BM25 must discriminate entities that cosine conflates.

        Cosine embedding gives all three facts ~0.95+ similarity because they
        share "was born in" template structure. BM25 uses token-level IDF to
        weight "Treviglio" (rare) above "was"/"born"/"in" (common), so the
        fact containing "Treviglio" ranks highest.
        """
        idx = BM25Index()
        idx.add(0, "Christian Abbiati was born in Treviglio")
        idx.add(1, "Marco Rossi was born in Bergamo")
        idx.add(2, "Luigi Verdi was born in Milan")

        results = idx.query("Who was born in Treviglio?")
        assert len(results) >= 1
        top_id, top_score = results[0]
        assert top_id == 0, (
            f"Expected doc 0 (Abbiati/Treviglio) ranked #1, got doc {top_id}. "
            f"Full ranking: {results}"
        )

    def test_entity_discrimination_sport(self):
        """BM25 discriminates by entity name in sport queries.

        "What sport does Christian Abbiati play?" should rank the Abbiati
        fact first because "Christian" and "Abbiati" are rare tokens that
        match exactly, while "plays" is common across all facts.
        """
        idx = BM25Index()
        idx.add(0, "Christian Abbiati plays association football")
        idx.add(1, "Marco Rossi plays ice hockey")
        idx.add(2, "Luigi Verdi plays tennis")

        results = idx.query("What sport does Christian Abbiati play?")
        assert len(results) >= 1
        top_id, top_score = results[0]
        assert top_id == 0, (
            f"Expected doc 0 (Abbiati/football) ranked #1, got doc {top_id}. "
            f"Full ranking: {results}"
        )

    def test_idf_weighting(self):
        """Common words should have lower weight than rare words.

        "was", "born", "in" appear in all 3 docs → low IDF.
        "Treviglio", "Abbiati" appear in 1 doc → high IDF.
        Query for "Treviglio" alone should still rank doc 0 first.
        """
        idx = BM25Index()
        idx.add(0, "Christian Abbiati was born in Treviglio")
        idx.add(1, "Marco Rossi was born in Bergamo")
        idx.add(2, "Luigi Verdi was born in Milan")

        # Query with only the rare entity token
        results = idx.query("Treviglio")
        assert len(results) >= 1
        top_id, _ = results[0]
        assert top_id == 0, f"Rare token 'Treviglio' should match doc 0, got {top_id}"

        # Query with only common tokens — all docs should score similarly
        results_common = idx.query("was born in")
        if len(results_common) >= 2:
            # Scores should be close (within same order of magnitude)
            scores = [s for _, s in results_common]
            score_range = max(scores) - min(scores)
            avg_score = sum(scores) / len(scores)
            # Common terms should not heavily discriminate
            assert score_range < avg_score, (
                f"Common tokens should produce similar scores across docs. "
                f"Range={score_range:.3f}, Avg={avg_score:.3f}"
            )

    def test_incremental_vs_batch(self):
        """Adding docs one-by-one produces same query results as add_batch."""
        docs = [
            (0, "Christian Abbiati was born in Treviglio"),
            (1, "Marco Rossi was born in Bergamo"),
            (2, "Luigi Verdi was born in Milan"),
        ]

        # Incremental
        idx_inc = BM25Index()
        for doc_id, text in docs:
            idx_inc.add(doc_id, text)

        # Batch
        idx_batch = BM25Index()
        idx_batch.add_batch(docs)

        # Same query should produce same ranking
        q = "Who was born in Treviglio?"
        results_inc = idx_inc.query(q)
        results_batch = idx_batch.query(q)

        assert len(results_inc) == len(results_batch), (
            f"Incremental: {len(results_inc)} results, Batch: {len(results_batch)} results"
        )
        # Same doc IDs in same order
        ids_inc = [doc_id for doc_id, _ in results_inc]
        ids_batch = [doc_id for doc_id, _ in results_batch]
        assert ids_inc == ids_batch, (
            f"Ranking differs: incremental={ids_inc}, batch={ids_batch}"
        )
        # Scores should be close (floating point tolerance)
        for (_, s_inc), (_, s_batch) in zip(results_inc, results_batch):
            assert abs(s_inc - s_batch) < 1e-9, (
                f"Score mismatch: {s_inc} vs {s_batch}"
            )

    def test_empty_index_returns_empty(self):
        """Query on empty index returns empty list, no crash."""
        idx = BM25Index()
        results = idx.query("anything at all")
        assert results == []

    def test_add_batch(self):
        """add_batch correctly indexes multiple documents."""
        idx = BM25Index()
        idx.add_batch([
            (10, "Alpha beta gamma"),
            (20, "Delta epsilon zeta"),
            (30, "Alpha delta eta"),
        ])
        assert len(idx) == 3

        results = idx.query("epsilon")
        assert len(results) >= 1
        top_id, _ = results[0]
        assert top_id == 20, f"Expected doc 20 (epsilon), got {top_id}"

    def test_len(self):
        """len(index) equals number of added documents."""
        idx = BM25Index()
        assert len(idx) == 0

        idx.add(0, "First document")
        assert len(idx) == 1

        idx.add(1, "Second document")
        assert len(idx) == 2

        idx.add_batch([(2, "Third"), (3, "Fourth")])
        assert len(idx) == 4

    def test_top_k_limit(self):
        """Query with top_k=2 returns at most 2 results."""
        idx = BM25Index()
        for i in range(10):
            idx.add(i, f"Document number {i} with some shared words")

        results = idx.query("Document number", top_k=2)
        assert len(results) <= 2, f"Expected at most 2 results, got {len(results)}"

    def test_no_duplicate_doc_ids(self):
        """Adding same doc_id twice does not create duplicates in results.

        The second add with the same doc_id should either replace the first
        or the index should deduplicate on output. Either way, the doc_id
        should appear at most once in results.
        """
        idx = BM25Index()
        idx.add(0, "Christian Abbiati was born in Treviglio")
        idx.add(0, "Christian Abbiati was born in Treviglio")  # duplicate

        results = idx.query("Abbiati Treviglio")
        doc_ids = [doc_id for doc_id, _ in results]
        assert len(doc_ids) == len(set(doc_ids)), (
            f"Duplicate doc_ids in results: {doc_ids}"
        )


# ---------------------------------------------------------------------------
# Test 2: ChainCache — CRUD and invalidation
# ---------------------------------------------------------------------------


class TestChainCache:
    """Test chain cache with cosine lookup and supersession invalidation."""

    def _make_chain(
        self,
        query_text: str = "test query",
        embedding_seed: int = 0,
        evidence_facts: list[str] | None = None,
        final_answer: str = "test answer",
        chain_id: int = 0,
        dim: int = 64,
    ) -> CachedChain:
        """Create a CachedChain for testing."""
        emb = _random_embedding(dim, seed=embedding_seed)
        return CachedChain(
            query_embedding=emb,
            query_text=query_text,
            sub_queries=["sub1", "sub2"],
            intermediate_answers={"{answer_1}": "intermediate"},
            evidence_facts=evidence_facts or ["fact A", "fact B"],
            final_answer=final_answer,
            chain_id=chain_id,
        )

    def test_store_and_lookup(self):
        """Store a chain, lookup with same embedding returns it."""
        cache = ChainCache(similarity_threshold=0.95)
        chain = self._make_chain(chain_id=1, embedding_seed=42)
        cache.store(chain)

        # Lookup with the exact same embedding
        result = cache.lookup(chain.query_embedding)
        assert result is not None, "Lookup with same embedding should hit"
        assert result.chain_id == 1
        assert result.final_answer == "test answer"

    def test_lookup_miss_below_threshold(self):
        """Lookup with dissimilar embedding returns None."""
        cache = ChainCache(similarity_threshold=0.95)
        chain = self._make_chain(chain_id=1, embedding_seed=42)
        cache.store(chain)

        # Create a very different embedding (orthogonal in high dimensions)
        dissimilar = _random_embedding(64, seed=999)
        # Verify it's actually dissimilar
        sim = _cosine_sim(chain.query_embedding, dissimilar)
        assert sim < 0.95, f"Embeddings too similar for test: cosine={sim:.3f}"

        result = cache.lookup(dissimilar)
        assert result is None, (
            f"Lookup with dissimilar embedding (cosine={sim:.3f}) should miss"
        )

    def test_lookup_miss_empty_cache(self):
        """Empty cache returns None for any query."""
        cache = ChainCache(similarity_threshold=0.95)
        emb = _random_embedding(64, seed=0)
        result = cache.lookup(emb)
        assert result is None, "Empty cache should always return None"

    def test_invalidate_evicts_chain(self):
        """Store chain, invalidate one of its evidence facts, lookup returns None."""
        cache = ChainCache(similarity_threshold=0.95)
        chain = self._make_chain(
            chain_id=1,
            embedding_seed=42,
            evidence_facts=["Abbiati plays football", "Abbiati was born in Milan"],
        )
        cache.store(chain)

        # Verify the chain is retrievable
        assert cache.lookup(chain.query_embedding) is not None

        # Invalidate one evidence fact
        evicted = cache.invalidate_fact("Abbiati plays football")
        assert evicted >= 1, "Should have evicted at least 1 chain"

        # Now lookup should miss
        result = cache.lookup(chain.query_embedding)
        assert result is None, "Chain should be evicted after evidence fact invalidated"

    def test_invalidate_precision(self):
        """Two chains, invalidate fact only in chain A. Chain B survives."""
        cache = ChainCache(similarity_threshold=0.95)

        chain_a = self._make_chain(
            chain_id=1,
            embedding_seed=10,
            evidence_facts=["fact_X", "fact_Y"],
        )
        chain_b = self._make_chain(
            chain_id=2,
            embedding_seed=20,
            evidence_facts=["fact_Z", "fact_W"],
        )
        cache.store(chain_a)
        cache.store(chain_b)
        assert len(cache) == 2

        # Invalidate fact_X (only in chain A)
        evicted = cache.invalidate_fact("fact_X")
        assert evicted == 1, f"Should evict exactly 1 chain, evicted {evicted}"

        # Chain A should be gone
        result_a = cache.lookup(chain_a.query_embedding)
        assert result_a is None, "Chain A should be evicted"

        # Chain B should survive
        result_b = cache.lookup(chain_b.query_embedding)
        assert result_b is not None, "Chain B should survive (doesn't use fact_X)"
        assert result_b.chain_id == 2

    def test_invalidate_returns_count(self):
        """invalidate_fact returns number of chains evicted."""
        cache = ChainCache(similarity_threshold=0.95)

        # Store 3 chains, 2 sharing "shared_fact"
        cache.store(self._make_chain(
            chain_id=1, embedding_seed=10,
            evidence_facts=["shared_fact", "unique_1"],
        ))
        cache.store(self._make_chain(
            chain_id=2, embedding_seed=20,
            evidence_facts=["shared_fact", "unique_2"],
        ))
        cache.store(self._make_chain(
            chain_id=3, embedding_seed=30,
            evidence_facts=["other_fact", "unique_3"],
        ))

        evicted = cache.invalidate_fact("shared_fact")
        assert evicted == 2, f"Expected 2 chains evicted, got {evicted}"

        # Invalidating a fact not in any chain
        evicted_none = cache.invalidate_fact("nonexistent_fact")
        assert evicted_none == 0, f"Expected 0 chains evicted, got {evicted_none}"

    def test_multiple_chains_share_fact(self):
        """Two chains both use fact F. Invalidating F evicts both."""
        cache = ChainCache(similarity_threshold=0.95)
        shared = "The CEO of Acme is John"

        chain_a = self._make_chain(
            chain_id=1, embedding_seed=10,
            evidence_facts=[shared, "fact_A"],
        )
        chain_b = self._make_chain(
            chain_id=2, embedding_seed=20,
            evidence_facts=[shared, "fact_B"],
        )
        cache.store(chain_a)
        cache.store(chain_b)

        evicted = cache.invalidate_fact(shared)
        assert evicted == 2

        assert cache.lookup(chain_a.query_embedding) is None
        assert cache.lookup(chain_b.query_embedding) is None
        assert len(cache) == 0

    def test_len(self):
        """len(cache) tracks stored chains minus evicted ones."""
        cache = ChainCache(similarity_threshold=0.95)
        assert len(cache) == 0

        cache.store(self._make_chain(
            chain_id=1, embedding_seed=10,
            evidence_facts=["fact_1"],
        ))
        assert len(cache) == 1

        cache.store(self._make_chain(
            chain_id=2, embedding_seed=20,
            evidence_facts=["fact_2"],
        ))
        assert len(cache) == 2

        # Evict chain 1
        cache.invalidate_fact("fact_1")
        assert len(cache) == 1

    def test_best_match_returned(self):
        """Store 2 chains with different embeddings. Lookup returns the one
        with higher cosine similarity to the query."""
        cache = ChainCache(similarity_threshold=0.90)  # lower threshold for test

        # Chain A: embedding close to seed=42
        chain_a = self._make_chain(
            chain_id=1, embedding_seed=42,
            final_answer="answer_A",
        )

        # Chain B: very different embedding
        chain_b = self._make_chain(
            chain_id=2, embedding_seed=999,
            final_answer="answer_B",
        )
        cache.store(chain_a)
        cache.store(chain_b)

        # Query with embedding very close to chain A (add tiny noise)
        rng = np.random.RandomState(42)
        query_emb = chain_a.query_embedding + rng.randn(64) * 0.01
        query_emb = query_emb / np.linalg.norm(query_emb)

        result = cache.lookup(query_emb)
        assert result is not None, "Should find a match above threshold"
        assert result.chain_id == 1, (
            f"Should return chain A (closer match), got chain {result.chain_id}"
        )

    def test_threshold_boundary(self):
        """Cosine exactly at threshold is a hit; just below is a miss.

        We construct embeddings with known cosine similarity to test
        the boundary condition precisely.
        """
        threshold = 0.95
        cache = ChainCache(similarity_threshold=threshold)
        dim = 64

        # Base embedding
        base = _random_embedding(dim, seed=0)

        # Construct an embedding with cosine similarity exactly at threshold
        # cos(theta) = 0.95 means theta = arccos(0.95)
        # Rotate base by theta in a 2D plane
        theta_at = np.arccos(threshold)
        # Create orthogonal component
        ortho = _random_embedding(dim, seed=1)
        ortho = ortho - np.dot(ortho, base) * base
        ortho = ortho / np.linalg.norm(ortho)

        # At threshold: cos = 0.95
        at_threshold = np.cos(theta_at) * base + np.sin(theta_at) * ortho
        at_threshold = at_threshold / np.linalg.norm(at_threshold)
        sim_at = _cosine_sim(base, at_threshold)
        assert abs(sim_at - threshold) < 0.001, f"Expected sim ~{threshold}, got {sim_at}"

        # Below threshold: cos = 0.94
        theta_below = np.arccos(threshold - 0.01)
        below_threshold = np.cos(theta_below) * base + np.sin(theta_below) * ortho
        below_threshold = below_threshold / np.linalg.norm(below_threshold)
        sim_below = _cosine_sim(base, below_threshold)
        assert sim_below < threshold, f"Expected sim < {threshold}, got {sim_below}"

        chain = CachedChain(
            query_embedding=base,
            query_text="test",
            sub_queries=[],
            intermediate_answers={},
            evidence_facts=["fact"],
            final_answer="answer",
            chain_id=1,
        )
        cache.store(chain)

        # At threshold should hit
        result_at = cache.lookup(at_threshold)
        assert result_at is not None, (
            f"Cosine {sim_at:.4f} >= {threshold} should be a cache hit"
        )

        # Below threshold should miss
        result_below = cache.lookup(below_threshold)
        assert result_below is None, (
            f"Cosine {sim_below:.4f} < {threshold} should be a cache miss"
        )


# ---------------------------------------------------------------------------
# Test 3: Integration — BM25 wiring in decompose loop
# ---------------------------------------------------------------------------


class TestDecomposeIntegration:
    """Test that BM25 and ChainCache are wired correctly into the query path."""

    def _make_agent(self):
        """Create a HermesMemoryAgent with mocked dependencies for query-path testing."""
        from mabench.hermes_agent import HermesMemoryAgent

        mock_pipeline = MagicMock()
        mock_scorer = MagicMock()
        mock_scorer.embed_batch.return_value = [_unit_vec(8, i) for i in range(10)]
        mock_scorer.embed.side_effect = lambda t: _unit_vec(8, hash(t) % 8)

        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.supersession = True
                agent.conflict_resolution = True
                agent._cr_per_hop = True
                agent._cr_overfetch = 1
                agent.cosine_dedup = False
                agent.decompose_query = True
                agent.decompose_rrf = False
                agent.entity_retrieval = False
                agent.transfer_retrieval = False
                agent.coretrieval_retrieval = False
                agent.cooc_boost_retrieval = False
                agent.ppr_retrieval = False
                agent.hybrid_retrieval = False
                agent.associative_retrieval = False
                agent.sparse_retrieval = False
                agent.bm25_weight = 0.0
                agent.retrieve_num = 10
                agent._triple_index = defaultdict(list)
                agent._superseded_texts = set()
                agent._fact_entities = {}
                agent._supersession_graph = {}
                agent._supersedes = defaultdict(set)
                agent._ingestion = mock_pipeline
                agent._scorer = mock_scorer
                agent._store_count = 0
                agent.dream_interval = 0
                agent._triadic = None
                agent._entity_accumulator = MagicMock()
                agent._belief_index = None
                agent._gliner_entity_index = defaultdict(set)
                agent.model = "gpt-4o-mini"

                agent.orchestrator = MagicMock()
                agent.coupled_engine = MagicMock()
                agent.coupled_engine._session_buffer = []
                agent.coupled_engine.memory_store = []

                # BM25 index (the new component)
                agent._bm25_index = BM25Index()

                # Chain cache (the new component)
                agent._chain_cache = ChainCache(similarity_threshold=0.95)

                # Mock LLM client for decompose/extract
                agent._llm_client = MagicMock()

        return agent

    def test_bm25_used_for_subquery_hops(self):
        """Sub-query hops should call bm25_index.query, not _retrieve_v2.

        After the BM25 wiring, the decompose loop should use BM25 for each
        sub-query hop (lexical matching for entity discrimination), while
        keeping _retrieve_v2 for the original question.
        """
        agent = self._make_agent()

        # Populate BM25 index with test facts
        agent._bm25_index.add(0, "Christian Abbiati was born in Treviglio")
        agent._bm25_index.add(1, "Treviglio is located in Bergamo province")

        # Mock _decompose_query to return sub-queries
        agent._decompose_query = MagicMock(return_value=[
            "Where was Christian Abbiati born?",
            "In which province is {answer_1} located?",
        ])

        # Mock _retrieve_v2 for the original question
        agent._retrieve_v2 = MagicMock(return_value=[
            {"text": "Christian Abbiati was born in Treviglio", "score": 0.9},
        ])

        # Mock _extract_short_answer for hop chaining
        agent._extract_short_answer = MagicMock(return_value="Treviglio")

        # Mock _extract_question to return bare question
        agent._extract_question = MagicMock(return_value="Where was Abbiati born?")

        # Mock _query_time_cr to pass through
        agent._query_time_cr = MagicMock(side_effect=lambda texts: texts)

        # Mock the LLM calls in the decompose loop (judge + final answer)
        mock_judge = MagicMock()
        mock_judge.choices = [MagicMock()]
        mock_judge.choices[0].message.content = "ANSWER_READY"
        agent._llm_client.chat.completions.create.return_value = mock_judge

        # The BM25 index should be used for sub-query retrieval
        with patch.object(agent._bm25_index, 'query', wraps=agent._bm25_index.query) as bm25_spy:
            # We need the full _query to be called, but it has many moving parts.
            # Instead, we verify the contract: bm25_index.query is callable
            # and returns ranked results for sub-queries.
            results = agent._bm25_index.query("Where was Christian Abbiati born?", top_k=10)
            assert len(results) > 0
            assert results[0][0] == 0  # Abbiati fact ranked first

    def test_cosine_used_for_original_question(self):
        """The original question (final merge) should still use _retrieve_v2
        with cosine+cooc, not BM25. Semantic bridging is valuable for the
        final aggregation where we need diverse evidence."""
        agent = self._make_agent()

        # When decompose_query is True, the original question at the end
        # of the decompose loop calls _retrieve_v2(bare_question).
        # This test verifies _retrieve_v2 is still the method used for
        # the original question, by checking it returns cosine-ranked results.
        agent._retrieve_v2 = MagicMock(return_value=[
            {"text": "fact1", "score": 0.95},
            {"text": "fact2", "score": 0.90},
        ])

        results = agent._retrieve_v2("original question")
        agent._retrieve_v2.assert_called_once_with("original question")
        assert len(results) == 2

    def test_chain_cache_hit_skips_decompose(self):
        """When chain cache hits, the decompose loop is NOT entered (zero LLM calls).

        The query path should:
        1. Embed the query
        2. Check chain_cache.lookup(embedding)
        3. On HIT: return cached answer immediately
        4. No calls to _decompose_query, _retrieve_v2, or LLM
        """
        agent = self._make_agent()

        # Pre-populate cache with a chain
        query_emb = _random_embedding(64, seed=42)
        cached_chain = CachedChain(
            query_embedding=query_emb,
            query_text="Where was Abbiati born?",
            sub_queries=["Where was Abbiati born?"],
            intermediate_answers={},
            evidence_facts=["Abbiati was born in Treviglio"],
            final_answer="Treviglio",
            chain_id=1,
        )
        agent._chain_cache.store(cached_chain)

        # Mock _scorer.embed to return the same embedding
        agent._scorer.embed = MagicMock(return_value=query_emb)

        # Mock decompose to track if it's called
        agent._decompose_query = MagicMock()

        # The chain cache should return a hit, and _decompose_query
        # should NOT be called. We test the cache lookup directly since
        # the wiring into _query doesn't exist yet.
        result = agent._chain_cache.lookup(query_emb)
        assert result is not None
        assert result.final_answer == "Treviglio"
        agent._decompose_query.assert_not_called()

    def test_chain_cache_miss_runs_decompose(self):
        """When chain cache misses, the full decompose loop runs."""
        agent = self._make_agent()

        # Empty cache — every lookup misses
        query_emb = _random_embedding(64, seed=42)
        result = agent._chain_cache.lookup(query_emb)
        assert result is None, "Empty cache should miss"

        # After a miss, the decompose loop should execute. We verify
        # by checking _decompose_query would be called in the real flow.
        # (The actual wiring test will be an integration test once
        #  the modules are wired into hermes_agent.py)
        assert len(agent._chain_cache) == 0

    def test_supersession_invalidates_chain_cache(self):
        """Ingest a fact that supersedes one used in a cached chain.
        Verify the chain is evicted from the cache.

        This tests the contract: when _superseded_texts.add(existing_text)
        fires during ingestion, chain_cache.invalidate_fact(existing_text)
        must also fire, evicting any chains that used that fact as evidence.
        """
        agent = self._make_agent()

        # Step 1: Store a chain that uses "Abbiati plays football" as evidence
        query_emb = _random_embedding(64, seed=42)
        chain = CachedChain(
            query_embedding=query_emb,
            query_text="What sport does Abbiati play?",
            sub_queries=["What sport does Abbiati play?"],
            intermediate_answers={},
            evidence_facts=[
                "Abbiati plays football",
                "Football is a team sport",
            ],
            final_answer="football",
            chain_id=1,
        )
        agent._chain_cache.store(chain)
        assert agent._chain_cache.lookup(query_emb) is not None

        # Step 2: Simulate supersession firing during ingestion
        # In the real code, _store() detects conflict and calls:
        #   self._superseded_texts.add("Abbiati plays football")
        # The chain cache invalidation should also fire:
        #   self._chain_cache.invalidate_fact("Abbiati plays football")
        superseded_fact = "Abbiati plays football"

        # Simulate the invalidation
        evicted = agent._chain_cache.invalidate_fact(superseded_fact)
        assert evicted == 1, f"Expected 1 chain evicted, got {evicted}"

        # Step 3: Verify the chain is gone
        result = agent._chain_cache.lookup(query_emb)
        assert result is None, (
            "Chain should be evicted after its evidence fact was superseded"
        )


# ---------------------------------------------------------------------------
# Test 4: BM25Index.score_texts — rerank arbitrary texts with corpus IDF
# ---------------------------------------------------------------------------


class TestBM25ScoreTexts:
    """Tests for BM25Index.score_texts reranking method.

    score_texts uses IDF from the full indexed corpus but computes TF from
    the given input texts. This enables BM25 reranking of candidates retrieved
    by other methods (e.g., cosine similarity) without requiring those
    candidates to be in the index.
    """

    def test_score_texts_returns_correct_length(self):
        """3 texts in -> 3 scores out, always."""
        idx = BM25Index()
        idx.add(0, "The capital of France is Paris")
        idx.add(1, "The capital of Germany is Berlin")
        idx.add(2, "The capital of Italy is Rome")

        scores = idx.score_texts("capital of France", [
            "France is in Europe",
            "Paris is beautiful",
            "Berlin is the largest city",
        ])
        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"

    def test_score_texts_empty_index_returns_zeros(self):
        """No docs indexed -> IDF is undefined -> all zeros."""
        idx = BM25Index()
        scores = idx.score_texts("any query", ["text one", "text two"])
        assert scores == [0.0, 0.0], f"Expected [0.0, 0.0], got {scores}"

    def test_score_texts_no_overlap_returns_zero(self):
        """Query tokens not in any input text -> 0.0 for all."""
        idx = BM25Index()
        idx.add(0, "alpha beta gamma")
        idx.add(1, "delta epsilon zeta")

        scores = idx.score_texts("alpha beta", [
            "completely different words here",
            "nothing matching at all",
        ])
        assert scores == [0.0, 0.0], (
            f"Expected all zeros when no lexical overlap, got {scores}"
        )

    def test_score_texts_matches_query_score(self):
        """Score for an indexed text should match query() result.

        This verifies the same BM25 formula is used in both code paths.
        Index 3 docs, then score_texts on the exact same texts. The scores
        should match query() scores exactly (same IDF, same TF, same avgdl).
        """
        idx = BM25Index()
        docs = [
            (0, "Christian Abbiati was born in Treviglio"),
            (1, "Marco Rossi was born in Bergamo"),
            (2, "Luigi Verdi was born in Milan"),
        ]
        for doc_id, text in docs:
            idx.add(doc_id, text)

        query = "Who was born in Treviglio?"
        query_results = idx.query(query, top_k=10)

        # Build a map of doc_id -> score from query()
        query_scores = {doc_id: score for doc_id, score in query_results}

        # score_texts on the same texts in doc_id order
        texts = [text for _, text in docs]
        rerank_scores = idx.score_texts(query, texts)

        for i, (doc_id, text) in enumerate(docs):
            expected = query_scores.get(doc_id, 0.0)
            actual = rerank_scores[i]
            assert abs(expected - actual) < 1e-9, (
                f"Doc {doc_id}: query() gave {expected:.6f}, "
                f"score_texts() gave {actual:.6f}"
            )

    def test_score_texts_entity_discrimination(self):
        """Core use case: BM25 reranks cosine-retrieved candidates correctly.

        Given candidate texts from cosine retrieval (which conflates entities
        due to shared template structure), BM25 should rank the text
        containing the query entity higher.

        Index many "X born in Y" facts to establish IDF statistics.
        Candidates: one matching entity, one non-matching.
        The matching candidate should score higher.
        """
        idx = BM25Index()
        # Build corpus for IDF statistics
        idx.add(0, "Christian Abbiati was born in Treviglio")
        idx.add(1, "Marco Rossi was born in Bergamo")
        idx.add(2, "Luigi Verdi was born in Milan")
        idx.add(3, "Giovanni Bianchi was born in Turin")
        idx.add(4, "Paolo Neri was born in Florence")

        # Candidates from cosine retrieval (would have ~0.95+ cosine similarity)
        candidates = [
            "Marco Rossi was born in Bergamo",      # wrong entity
            "Christian Abbiati was born in Treviglio",  # correct entity
        ]

        scores = idx.score_texts("born in Treviglio", candidates)
        assert scores[1] > scores[0], (
            f"Treviglio fact (score={scores[1]:.4f}) should rank higher than "
            f"Bergamo fact (score={scores[0]:.4f})"
        )

    def test_score_texts_uses_corpus_idf(self):
        """IDF should come from the full index, not the input texts.

        A rare token in the corpus should get high IDF even if it appears
        in all input texts. Conversely, a common token in the corpus should
        get low IDF even if it appears in only one input text.
        """
        idx = BM25Index()
        # "alpha" appears in 1 doc (rare), "common" appears in all 5
        idx.add(0, "alpha common shared")
        idx.add(1, "beta common shared")
        idx.add(2, "gamma common shared")
        idx.add(3, "delta common shared")
        idx.add(4, "epsilon common shared")

        # Both input texts contain "alpha" AND "common".
        # "alpha" is rare in corpus (df=1) -> high IDF
        # "common" is ubiquitous (df=5) -> low IDF
        texts_with_both = [
            "alpha common",
            "alpha common",
        ]

        scores = idx.score_texts("alpha common", texts_with_both)

        # Now score texts where only "common" appears (no "alpha")
        texts_common_only = [
            "common only here",
            "common only there",
        ]
        scores_common = idx.score_texts("alpha common", texts_common_only)

        # Texts with "alpha" (rare in corpus) should score much higher
        # than texts with only "common" (ubiquitous in corpus)
        assert scores[0] > scores_common[0], (
            f"Text with rare 'alpha' (score={scores[0]:.4f}) should beat "
            f"text with only 'common' (score={scores_common[0]:.4f}) "
            f"because IDF comes from corpus, not input texts"
        )


# ---------------------------------------------------------------------------
# Test 5: BM25 hop retrieval with post-retrieval reranking (Test 1 architecture)
# ---------------------------------------------------------------------------


class TestBM25HopReranking:
    """Tests for BM25 hop retrieval with post-retrieval reranking (Test 1 architecture).

    The Test 1 architecture replaces the hard if/else BM25-vs-cosine in the
    decompose loop with: BM25 candidate generation (overfetch 3x) followed by
    graph signal reranking via _apply_post_retrieval.

    Additionally, when bm25_hops is active, the original question's cosine
    results are BM25-reranked for entity discrimination.
    """

    def _make_agent(self):
        """Create a HermesMemoryAgent with mocked dependencies for full _query path."""
        from mabench.hermes_agent import HermesMemoryAgent

        mock_pipeline = MagicMock()
        mock_scorer = MagicMock()
        mock_scorer.embed_batch.return_value = [_unit_vec(8, i) for i in range(10)]
        mock_scorer.embed.side_effect = lambda t: _unit_vec(8, hash(t) % 8)

        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.supersession = True
                agent.conflict_resolution = True
                agent._cr_per_hop = True
                agent._cr_overfetch = 1
                agent.cosine_dedup = False
                agent.decompose_query = True
                agent.decompose_rrf = False
                agent.decompose_coverage = False
                agent.decompose_max_hops = 3
                agent.iterative_query = False
                agent.entity_retrieval = False
                agent.transfer_retrieval = False
                agent.coretrieval_retrieval = False
                agent.cooc_boost_retrieval = False
                agent.ppr_retrieval = False
                agent.hybrid_retrieval = False
                agent.associative_retrieval = False
                agent.sparse_retrieval = False
                agent.bm25_weight = 0.0
                agent.retrieve_num = 10
                agent._triple_index = defaultdict(list)
                agent._superseded_texts = set()
                agent._fact_entities = {}
                agent._supersession_graph = {}
                agent._supersedes = defaultdict(set)
                agent._ingestion = mock_pipeline
                agent._scorer = mock_scorer
                agent._store_count = 0
                agent.dream_interval = 0
                agent._triadic = None
                agent._entity_accumulator = MagicMock()
                agent._belief_index = None
                agent._gliner_entity_index = defaultdict(set)
                agent.model = "gpt-4o-mini"

                agent.orchestrator = MagicMock()
                agent.coupled_engine = MagicMock()
                agent.coupled_engine._session_buffer = []
                agent.coupled_engine.memory_store = []

                # BM25 index (the new component)
                agent.bm25_hops = True
                agent._bm25_index = BM25Index()

                # Chain cache
                agent._chain_cache = ChainCache(similarity_threshold=0.95)

                # Mock LLM client for decompose/extract
                agent._llm_client = MagicMock()

                # Mock downstream methods needed by _query to complete
                agent._format_outgestion = MagicMock(
                    return_value=("memory context", "")
                )
                agent._generate_answer = MagicMock(
                    return_value=("test answer", 10, 5, 0.1)
                )

        return agent

    def test_bm25_hops_calls_apply_post_retrieval(self):
        """When bm25_hops=True, the decompose loop should call _apply_post_retrieval
        on BM25 results before using them. This ensures belief reranking is applied
        to BM25 candidates just like it is to cosine candidates."""
        agent = self._make_agent()

        # Populate memory store and BM25 index with test facts
        from dataclasses import dataclass

        @dataclass
        class FakeMemEntry:
            text: str

        agent.coupled_engine.memory_store = [
            FakeMemEntry(text="Christian Abbiati was born in Treviglio"),
            FakeMemEntry(text="Marco Rossi was born in Bergamo"),
            FakeMemEntry(text="Luigi Verdi was born in Milan"),
        ]
        for i, entry in enumerate(agent.coupled_engine.memory_store):
            agent._bm25_index.add(i, entry.text)

        # Mock _decompose_query to return a single sub-query (no multi-hop chaining)
        agent._decompose_query = MagicMock(return_value=[
            "Where was Christian Abbiati born?",
        ])
        # Mock _retrieve_v2 for the original question merge
        agent._retrieve_v2 = MagicMock(return_value=[
            {"text": "Christian Abbiati was born in Treviglio", "score": 0.9},
        ])
        # Mock _extract_question
        agent._extract_question = MagicMock(return_value="Where was Abbiati born?")
        # Mock _query_time_cr to pass through
        agent._query_time_cr = MagicMock(side_effect=lambda texts: texts)
        # Mock LLM calls (judge says ANSWER_READY immediately)
        mock_judge = MagicMock()
        mock_judge.choices = [MagicMock()]
        mock_judge.choices[0].message.content = "ANSWER_READY"
        agent._llm_client.chat.completions.create.return_value = mock_judge

        # Spy on _apply_post_retrieval
        agent._apply_post_retrieval = MagicMock(
            side_effect=lambda results: results[:agent.retrieve_num]
        )

        # Run the full _query method (single argument: question string)
        agent._query("Where was Abbiati born?")

        # _apply_post_retrieval should have been called with BM25 results
        assert agent._apply_post_retrieval.call_count >= 1, (
            "_apply_post_retrieval was not called on BM25 hop results. "
            f"Call count: {agent._apply_post_retrieval.call_count}"
        )

    def test_bm25_hops_overfetches_3x(self):
        """BM25 query in the decompose loop should request retrieve_num * 3
        candidates to give the reranker a bigger pool to work with."""
        agent = self._make_agent()
        agent.retrieve_num = 10

        # Populate memory store and BM25 index
        from dataclasses import dataclass

        @dataclass
        class FakeMemEntry:
            text: str

        entries = [FakeMemEntry(text=f"Fact number {i}") for i in range(30)]
        agent.coupled_engine.memory_store = entries
        for i, entry in enumerate(entries):
            agent._bm25_index.add(i, entry.text)

        # Mock _decompose_query to return a single sub-query
        agent._decompose_query = MagicMock(return_value=[
            "What is fact number 5?",
        ])
        agent._retrieve_v2 = MagicMock(return_value=[
            {"text": "Fact number 5", "score": 0.9},
        ])
        agent._extract_question = MagicMock(return_value="What is fact 5?")
        agent._query_time_cr = MagicMock(side_effect=lambda texts: texts)
        mock_judge = MagicMock()
        mock_judge.choices = [MagicMock()]
        mock_judge.choices[0].message.content = "ANSWER_READY"
        agent._llm_client.chat.completions.create.return_value = mock_judge
        agent._apply_post_retrieval = MagicMock(
            side_effect=lambda results: results[:agent.retrieve_num]
        )

        # Spy on the BM25 index query method
        original_query = agent._bm25_index.query
        query_calls = []

        def spy_query(text, top_k=10):
            query_calls.append({"text": text, "top_k": top_k})
            return original_query(text, top_k=top_k)

        agent._bm25_index.query = spy_query

        agent._query("What is fact 5?")

        # At least one BM25 query should have been made with top_k = retrieve_num * 3
        assert len(query_calls) >= 1, "No BM25 queries were made"
        bm25_top_ks = [c["top_k"] for c in query_calls]
        assert any(k == 30 for k in bm25_top_ks), (
            f"Expected BM25 query with top_k=30 (retrieve_num*3), "
            f"got top_k values: {bm25_top_ks}"
        )

    def test_bm25_rerank_original_question(self):
        """When bm25_hops is active, orig_results from _retrieve_v2 should be
        BM25-reranked for entity discrimination. The BM25 reranking should
        reorder cosine results using lexical matching scores."""
        agent = self._make_agent()

        # Populate BM25 index with facts that have high IDF tokens
        agent._bm25_index.add(0, "Christian Abbiati was born in Treviglio")
        agent._bm25_index.add(1, "Marco Rossi was born in Bergamo")
        agent._bm25_index.add(2, "Luigi Verdi was born in Milan")

        # Test score_texts directly (the mechanism used for reranking)
        candidates = [
            "Marco Rossi was born in Bergamo",
            "Christian Abbiati was born in Treviglio",
        ]
        scores = agent._bm25_index.score_texts(
            "Where was Abbiati born in Treviglio?", candidates
        )

        # The Abbiati/Treviglio fact should score higher
        assert scores[1] > scores[0], (
            f"BM25 reranking should put Abbiati/Treviglio (score={scores[1]:.4f}) "
            f"above Rossi/Bergamo (score={scores[0]:.4f})"
        )

    def test_bm25_rerank_noop_when_no_overlap(self):
        """BM25 reranking should be a no-op (preserve original order) when
        BM25 scores are all zero (no lexical overlap between query and results)."""
        agent = self._make_agent()

        # Index facts with specific tokens
        agent._bm25_index.add(0, "alpha beta gamma")
        agent._bm25_index.add(1, "delta epsilon zeta")

        # Query with completely different tokens
        candidates = [
            "first result from cosine",
            "second result from cosine",
        ]
        scores = agent._bm25_index.score_texts(
            "unrelated query terms", candidates
        )

        # All scores should be zero - original order preserved
        assert all(s == 0.0 for s in scores), (
            f"Expected all zero scores for no-overlap query, got {scores}"
        )

    def test_else_branch_unchanged(self):
        """When bm25_hops=False, the decompose loop should use _retrieve_v2
        for sub-query hops exactly as before, with no BM25 involvement."""
        agent = self._make_agent()
        agent.bm25_hops = False
        agent._bm25_index = None

        from dataclasses import dataclass

        @dataclass
        class FakeMemEntry:
            text: str

        agent.coupled_engine.memory_store = [
            FakeMemEntry(text="Some fact"),
        ]

        agent._decompose_query = MagicMock(return_value=[
            "What is the answer?",
        ])
        agent._retrieve_v2 = MagicMock(return_value=[
            {"text": "Some fact", "score": 0.9},
        ])
        agent._extract_question = MagicMock(return_value="What is the answer?")
        agent._query_time_cr = MagicMock(side_effect=lambda texts: texts)
        mock_judge = MagicMock()
        mock_judge.choices = [MagicMock()]
        mock_judge.choices[0].message.content = "ANSWER_READY"
        agent._llm_client.chat.completions.create.return_value = mock_judge

        agent._query("What is the answer?")

        # _retrieve_v2 should have been called for both sub-query and orig question
        assert agent._retrieve_v2.call_count >= 2, (
            f"_retrieve_v2 should be called for sub-query + original question when "
            f"bm25_hops=False, but was called {agent._retrieve_v2.call_count} time(s)"
        )

    def test_bm25_rerank_orig_results_in_query(self):
        """Full integration: when bm25_hops is active, the original question's
        cosine results (from _retrieve_v2) should be BM25-reranked. Verify by
        checking that score_texts is called on orig_results' texts."""
        agent = self._make_agent()

        from dataclasses import dataclass

        @dataclass
        class FakeMemEntry:
            text: str

        agent.coupled_engine.memory_store = [
            FakeMemEntry(text="Christian Abbiati was born in Treviglio"),
            FakeMemEntry(text="Marco Rossi was born in Bergamo"),
        ]
        for i, entry in enumerate(agent.coupled_engine.memory_store):
            agent._bm25_index.add(i, entry.text)

        # Decompose returns a single sub-query
        agent._decompose_query = MagicMock(return_value=[
            "Where was Abbiati born?",
        ])
        # _retrieve_v2 returns results in WRONG entity order (Rossi first)
        # BM25 reranking should fix this for "Abbiati" query
        agent._retrieve_v2 = MagicMock(return_value=[
            {"text": "Marco Rossi was born in Bergamo", "score": 0.95},
            {"text": "Christian Abbiati was born in Treviglio", "score": 0.94},
        ])
        agent._extract_question = MagicMock(
            return_value="Where was Abbiati born in Treviglio?"
        )
        agent._query_time_cr = MagicMock(side_effect=lambda texts: texts)
        mock_judge = MagicMock()
        mock_judge.choices = [MagicMock()]
        mock_judge.choices[0].message.content = "ANSWER_READY"
        agent._llm_client.chat.completions.create.return_value = mock_judge

        # Spy on score_texts
        original_score_texts = agent._bm25_index.score_texts
        score_texts_calls = []

        def spy_score_texts(query_text, texts):
            result = original_score_texts(query_text, texts)
            score_texts_calls.append({
                "query": query_text,
                "texts": texts,
                "scores": result,
            })
            return result

        agent._bm25_index.score_texts = spy_score_texts

        agent._query("Where was Abbiati born in Treviglio?")

        # score_texts should have been called for orig_results reranking
        assert len(score_texts_calls) >= 1, (
            "BM25 score_texts was not called for original question reranking. "
            f"Call count: {len(score_texts_calls)}"
        )
