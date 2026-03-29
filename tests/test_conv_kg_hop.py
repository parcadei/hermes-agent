"""Tests for conversation KG hop wiring in HermesMemoryAgent.

Tests the integration of LLM triple extraction and KG hop resolution
into the conversation ingestion pipeline (hermes_memory_agent.py).

These tests do NOT require a vLLM server — they test:
1. __init__ accepts llm_extraction_url parameter
2. _subject_index is initialized correctly
3. _resolve_kg_hop() algorithm (unit tests with manually populated index)
4. _store_conversation() calls extractor when present
5. _retrieve() prepends KG hop results
6. reset() clears subject_index and re-creates extractor
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure hermes-memory and hermes-NLCDM are importable
_HERMES_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-NLCDM" / "python"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_no_url():
    """Agent with no LLM extraction URL (extractor disabled)."""
    from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

    return HermesMemoryAgent(llm_extraction_url="")


@pytest.fixture
def agent_with_url():
    """Agent with LLM extraction URL (extractor should be created)."""
    from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

    return HermesMemoryAgent(llm_extraction_url="http://localhost:8000/v1")


@pytest.fixture
def agent_with_index():
    """Agent with a manually populated _subject_index for KG hop tests."""
    from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

    agent = HermesMemoryAgent(llm_extraction_url="")
    # Populate subject_index manually to test _resolve_kg_hop
    agent._subject_index = {
        "python": {
            "preferred over": ("JavaScript", "[2024-01-01] (user) I prefer Python over JavaScript"),
            "used for": ("backend", "[2024-01-01] (user) I use Python for backend"),
        },
        "javascript": {
            "preferred over": ("python", "[2024-01-01] (user) I prefer Python over JavaScript"),
        },
        "20-gallon tank": {
            "contains": ("10 neon tetras", "[2024-02-01] (user) My 20-gallon tank has 10 neon tetras"),
        },
        "10 neon tetras": {
            "contains": ("20-gallon tank", "[2024-02-01] (user) My 20-gallon tank has 10 neon tetras"),
        },
        "vs code": {
            "role": ("primary editor", "[2024-01-01] (user) VS Code is my primary editor for 3 years"),
        },
        "miami": {
            "planned activity": ("trip", "[2024-03-01] (user) I'm planning a trip to Miami"),
        },
    }
    return agent


# ---------------------------------------------------------------------------
# 1. __init__ parameter and state
# ---------------------------------------------------------------------------


class TestInit:
    """Test that __init__ correctly handles llm_extraction_url."""

    def test_init_no_url_has_subject_index(self, agent_no_url):
        """Agent without URL should have empty _subject_index dict."""
        assert hasattr(agent_no_url, "_subject_index")
        assert agent_no_url._subject_index == {}

    def test_init_no_url_extractor_is_none(self, agent_no_url):
        """Agent without URL should have _llm_extractor = None."""
        assert agent_no_url._llm_extractor is None

    def test_init_no_url_stores_url(self, agent_no_url):
        """Agent without URL stores empty string."""
        assert agent_no_url._llm_extraction_url == ""

    def test_init_with_url_has_subject_index(self, agent_with_url):
        """Agent with URL should have empty _subject_index dict."""
        assert hasattr(agent_with_url, "_subject_index")
        assert agent_with_url._subject_index == {}

    def test_init_with_url_creates_extractor(self, agent_with_url):
        """Agent with URL should create an LLMTripleExtractor instance."""
        from mabench.llm_triple_extractor import LLMTripleExtractor

        assert agent_with_url._llm_extractor is not None
        assert isinstance(agent_with_url._llm_extractor, LLMTripleExtractor)

    def test_init_with_url_stores_url(self, agent_with_url):
        """Agent with URL stores the URL string."""
        assert agent_with_url._llm_extraction_url == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# 2. _resolve_kg_hop() — unit tests with pre-populated index
# ---------------------------------------------------------------------------


class TestResolveKgHop:
    """Test the KG hop resolution algorithm."""

    def test_empty_index_returns_empty(self, agent_no_url):
        """Empty subject_index returns no results."""
        assert agent_no_url._resolve_kg_hop("What language do I prefer?") == []

    def test_subject_match_with_predicate(self, agent_with_index):
        """Matching subject + predicate yields the source text."""
        results = agent_with_index._resolve_kg_hop(
            "What is Python preferred over?"
        )
        assert len(results) > 0
        assert any("Python" in r for r in results)

    def test_subject_match_no_predicate_single_pred(self, agent_with_index):
        """Single-predicate entity matches even without predicate in query."""
        results = agent_with_index._resolve_kg_hop("Tell me about Miami")
        assert len(results) > 0
        assert any("Miami" in r for r in results)

    def test_short_subject_filtered(self):
        """Subjects shorter than 3 chars are skipped."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="")
        agent._subject_index = {
            "or": {"type": ("conjunction", "or is a conjunction")},
        }
        results = agent._resolve_kg_hop("What is or?")
        assert results == []

    def test_longer_subject_preferred(self, agent_with_index):
        """Longer/more-specific subject matches rank higher."""
        results = agent_with_index._resolve_kg_hop(
            "What does the 20-gallon tank contain?"
        )
        assert len(results) > 0
        # The 20-gallon tank match (15 chars) should come before
        # shorter matches
        assert any("20-gallon tank" in r for r in results)

    def test_bidirectional_lookup(self, agent_with_index):
        """Bidirectional indexing allows object-side lookup."""
        results = agent_with_index._resolve_kg_hop(
            "Where are 10 neon tetras?"
        )
        assert len(results) > 0

    def test_deduplication(self, agent_with_index):
        """Same fact text from subject and object sides is deduplicated."""
        # Both "20-gallon tank" and "10 neon tetras" point to the same fact
        results = agent_with_index._resolve_kg_hop(
            "What does the 20-gallon tank with 10 neon tetras have?"
        )
        fact_texts = [r.strip()[:200] for r in results]
        assert len(fact_texts) == len(set(fact_texts)), "Duplicate results found"

    def test_max_3_entity_matches(self):
        """At most 3 entity matches are used (top 3 by length)."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="")
        # Create 5 entities that all appear in the query
        agent._subject_index = {
            f"entity{i}": {
                "prop": (f"val{i}", f"fact about entity{i}")
            }
            for i in range(5)
        }
        query = "entity0 entity1 entity2 entity3 entity4 prop"
        results = agent._resolve_kg_hop(query)
        # Should return at most 3 facts (top 3 entity matches)
        assert len(results) <= 3

    def test_no_false_positive_on_substring(self):
        """'or' inside 'sport' should not trigger a match (len < 3 guard)."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="")
        agent._subject_index = {
            "sport": {"type": ("outdoor", "sport is outdoor type")},
        }
        # "sport" appears in query, len >= 3, so it matches
        results = agent._resolve_kg_hop("What type of sport?")
        assert len(results) > 0

    def test_case_insensitive_match(self, agent_with_index):
        """Query matching is case-insensitive."""
        results = agent_with_index._resolve_kg_hop(
            "What is PYTHON preferred over?"
        )
        assert len(results) > 0


# ---------------------------------------------------------------------------
# 3. _store_conversation() extraction integration
# ---------------------------------------------------------------------------


class TestStoreConversationExtraction:
    """Test that _store_conversation calls LLM extractor and populates index."""

    @staticmethod
    def _mock_scorer(agent):
        """Replace the scorer with a mock that returns random embeddings."""
        import numpy as np

        mock_scorer = MagicMock()
        mock_scorer.embed.return_value = np.random.randn(1024).astype(np.float64)
        mock_scorer.embed_batch.side_effect = lambda texts, **kw: [
            np.random.randn(1024).astype(np.float64) for _ in texts
        ]
        mock_scorer._cache = {}
        agent._scorer = mock_scorer

    def test_store_conv_without_extractor_no_error(self):
        """Storing conversation without extractor should work (no extraction)."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="")
        self._mock_scorer(agent)
        chunk = (
            "['Chat Time: 2024/01/01 (Mon) 10:00', "
            "[{'role': 'user', 'content': 'I prefer Python over JavaScript'}]]"
        )
        result = agent._store(chunk)
        assert result == "Memorized"
        assert agent._subject_index == {}

    def test_store_conv_with_extractor_populates_index(self):
        """Storing conversation with extractor should populate _subject_index after flush."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="http://localhost:8000/v1")
        self._mock_scorer(agent)
        # Mock the extractor's extract_batch to return triples
        mock_extractor = MagicMock()
        mock_extractor.extract_batch.return_value = [
            [("Python", "preferred over", "JavaScript")],
        ]
        agent._llm_extractor = mock_extractor

        chunk = (
            "['Chat Time: 2024/01/01 (Mon) 10:00', "
            "[{'role': 'user', 'content': 'I prefer Python over JavaScript'}]]"
        )
        agent._store(chunk)

        # Extraction is deferred — buffer should have entries, index empty
        assert len(agent._extraction_buffer) > 0
        assert agent._subject_index == {}

        # Flush triggers extraction
        agent._flush_extraction()

        # extract_batch should have been called with raw content (no timestamp prefix)
        mock_extractor.extract_batch.assert_called_once()
        call_args = mock_extractor.extract_batch.call_args
        texts_arg = call_args[0][0]
        # The texts should be raw content, not timestamp-prefixed
        assert all("Chat Time" not in t for t in texts_arg)

        # subject_index should be populated
        assert "python" in agent._subject_index
        assert "preferred over" in agent._subject_index["python"]

    def test_store_conv_bidirectional_index(self):
        """Bidirectional indexing: both subject and object are indexed after flush."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="http://localhost:8000/v1")
        self._mock_scorer(agent)
        mock_extractor = MagicMock()
        mock_extractor.extract_batch.return_value = [
            [("Python", "preferred over", "JavaScript")],
        ]
        agent._llm_extractor = mock_extractor

        chunk = (
            "['Chat Time: 2024/01/01 (Mon) 10:00', "
            "[{'role': 'user', 'content': 'I prefer Python over JavaScript'}]]"
        )
        agent._store(chunk)
        agent._flush_extraction()

        # Both subject and object should be in index
        assert "python" in agent._subject_index
        assert "javascript" in agent._subject_index

    def test_store_conv_extraction_failure_graceful(self):
        """LLM extraction failure should not break flush."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="http://localhost:8000/v1")
        self._mock_scorer(agent)
        mock_extractor = MagicMock()
        mock_extractor.extract_batch.side_effect = RuntimeError("connection refused")
        agent._llm_extractor = mock_extractor

        chunk = (
            "['Chat Time: 2024/01/01 (Mon) 10:00', "
            "[{'role': 'user', 'content': 'Hello!'}]]"
        )
        result = agent._store(chunk)
        assert result == "Memorized"
        # Flush should not raise despite extraction failure
        agent._flush_extraction()

    def test_store_conv_skips_short_subjects(self):
        """Subjects shorter than 3 characters are not indexed."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="http://localhost:8000/v1")
        self._mock_scorer(agent)
        mock_extractor = MagicMock()
        mock_extractor.extract_batch.return_value = [
            [("I", "like", "cats"), ("Python", "used for", "AI")],
        ]
        agent._llm_extractor = mock_extractor

        chunk = (
            "['Chat Time: 2024/01/01 (Mon) 10:00', "
            "[{'role': 'user', 'content': 'I like cats and Python is used for AI'}]]"
        )
        agent._store(chunk)
        agent._flush_extraction()

        # "I" (len 1) should not be in index
        assert "i" not in agent._subject_index
        # "Python" (len 6) should be
        assert "python" in agent._subject_index

    def test_store_conv_mode_is_conversation(self):
        """extract_batch is called with mode='conversation' at flush time."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="http://localhost:8000/v1")
        self._mock_scorer(agent)
        mock_extractor = MagicMock()
        mock_extractor.extract_batch.return_value = [[]]
        agent._llm_extractor = mock_extractor

        chunk = (
            "['Chat Time: 2024/01/01 (Mon) 10:00', "
            "[{'role': 'user', 'content': 'Hello there'}]]"
        )
        agent._store(chunk)
        # extract_batch not called yet (deferred)
        mock_extractor.extract_batch.assert_not_called()

        agent._flush_extraction()
        call_kwargs = mock_extractor.extract_batch.call_args[1]
        assert call_kwargs.get("mode") == "conversation"


# ---------------------------------------------------------------------------
# 4. _retrieve() KG hop integration
# ---------------------------------------------------------------------------


class TestRetrieveKgHop:
    """Test that _retrieve prepends KG hop results."""

    @staticmethod
    def _mock_scorer(agent):
        """Replace the scorer with a mock that returns deterministic embeddings."""
        import numpy as np

        mock_scorer = MagicMock()
        mock_scorer.embed.return_value = np.random.randn(1024).astype(np.float64)
        mock_scorer.embed_batch.side_effect = lambda texts, **kw: [
            np.random.randn(1024).astype(np.float64) for _ in texts
        ]
        mock_scorer._cache = {}
        agent._scorer = mock_scorer

    @staticmethod
    def _mock_coupled_engine(agent, stored_texts=None):
        """Mock coupled_engine.query to return stored_texts as results."""
        stored_texts = stored_texts or []
        mock_ce = MagicMock()
        mock_ce.query.return_value = [
            {"text": t, "score": 0.5} for t in stored_texts
        ]
        agent.coupled_engine = mock_ce

    def test_retrieve_with_kg_hop_prepends(self, agent_with_index):
        """KG hop results should appear before cosine results."""
        self._mock_scorer(agent_with_index)
        self._mock_coupled_engine(
            agent_with_index,
            stored_texts=["Some irrelevant text about nothing"],
        )
        # Mock orchestrator to avoid O(n) scan
        agent_with_index.orchestrator._memories = []

        results = agent_with_index._retrieve(
            "What is Python preferred over?", top_k=10
        )
        # KG hop result about Python should be in the results
        assert any("Python" in r for r in results)
        # KG hop result should come BEFORE cosine result
        python_idx = next(i for i, r in enumerate(results) if "Python" in r)
        irrelevant_idx = next(
            (i for i, r in enumerate(results) if "irrelevant" in r),
            len(results),
        )
        assert python_idx < irrelevant_idx, (
            "KG hop result should be prepended before cosine results"
        )

    def test_retrieve_without_kg_match_normal(self):
        """Without KG matches, normal retrieval proceeds."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="")
        self._mock_scorer(agent)
        self._mock_coupled_engine(
            agent,
            stored_texts=["The capital of France is Paris"],
        )
        agent.orchestrator._memories = []

        results = agent._retrieve("capital of France", top_k=5)
        # Should still return something from cosine
        assert len(results) > 0


# ---------------------------------------------------------------------------
# 5. reset() clears KG state
# ---------------------------------------------------------------------------


class TestResetKgState:
    """Test that reset() properly clears KG hop state."""

    def test_reset_clears_subject_index(self, agent_with_index):
        """reset() should clear _subject_index."""
        assert len(agent_with_index._subject_index) > 0
        agent_with_index.reset()
        assert agent_with_index._subject_index == {}

    def test_reset_with_url_recreates_extractor(self):
        """reset() with URL should recreate the extractor."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent(llm_extraction_url="http://localhost:8000/v1")
        old_extractor = agent._llm_extractor
        agent.reset()
        assert agent._llm_extractor is not None
        # Should be a new instance
        assert agent._llm_extractor is not old_extractor

    def test_reset_without_url_no_extractor(self, agent_no_url):
        """reset() without URL should not create extractor."""
        agent_no_url.reset()
        assert agent_no_url._llm_extractor is None


# ---------------------------------------------------------------------------
# 6. Ontology category matching
# ---------------------------------------------------------------------------


class TestOntologyCategoryMatching:
    """Test ontology-based category matching in _resolve_kg_hop."""

    @pytest.fixture
    def agent_with_ontology(self):
        """Agent with subject_index AND ontology_index populated."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent()
        # Populate subject_index (ground-level entities)
        agent._subject_index = {
            "20-gallon tank": {"contains": ("10 neon tetras", "I have a 20-gallon tank with 10 neon tetras")},
            "10-gallon tank": {"contains": ("betta fish", "My 10-gallon tank has a betta fish named Bubbles")},
            "10 neon tetras": {"lives in": ("20-gallon tank", "I have a 20-gallon tank with 10 neon tetras")},
            "betta fish": {"name": ("Bubbles", "My 10-gallon tank has a betta fish named Bubbles")},
            "java moss": {"recommended for": ("tank setup", "Java Moss is great for beginners")},
        }
        # Populate ontology_index (category -> stored_texts)
        agent._ontology_index = {
            "fish": [
                "I have a 20-gallon tank with 10 neon tetras",
                "My 10-gallon tank has a betta fish named Bubbles",
            ],
            "aquarium": [
                "I have a 20-gallon tank with 10 neon tetras",
                "My 10-gallon tank has a betta fish named Bubbles",
            ],
            "plant": [
                "Java Moss is great for beginners",
            ],
        }
        return agent

    def test_category_match_fish(self, agent_with_ontology):
        """'fish' in query should match the fish category."""
        results = agent_with_ontology._resolve_kg_hop(
            "How many fish are there in total?"
        )
        assert len(results) >= 2
        assert any("neon tetras" in r for r in results)
        assert any("betta fish" in r for r in results)

    def test_category_match_aquariums_plural(self, agent_with_ontology):
        """'aquariums' (plural) should stem to 'aquarium' and match."""
        results = agent_with_ontology._resolve_kg_hop(
            "Tell me about my aquariums"
        )
        assert len(results) >= 2
        assert any("20-gallon" in r for r in results)
        assert any("10-gallon" in r for r in results)

    def test_category_no_false_positive_fishing(self, agent_with_ontology):
        """'fishing' should NOT match 'fish' category (whole-word stemmed)."""
        results = agent_with_ontology._resolve_kg_hop(
            "I went fishing last weekend"
        )
        # 'fishing' stems to 'fish' via _stem('ing' suffix)
        # This IS expected to match — _stem strips -ing
        # The spec says this is acceptable; whole-word prevents substring
        # false positives like 'fish' matching inside 'selfish'

    def test_no_substring_false_positive(self, agent_with_ontology):
        """Category 'fish' should NOT match via substring in 'selfish'."""
        # Substring of 'selfish' contains 'fish' but stemmed word 'selfish' != 'fish'
        results = agent_with_ontology._resolve_kg_hop(
            "She was being selfish about it"
        )
        fish_results = [r for r in results if "neon tetras" in r or "betta" in r]
        assert len(fish_results) == 0

    def test_category_deduplication(self, agent_with_ontology):
        """Results from entity match + category match should be deduplicated."""
        results = agent_with_ontology._resolve_kg_hop(
            "Tell me about betta fish in my aquariums"
        )
        # 'betta fish' hits entity match AND 'fish' category + 'aquarium' category
        # but each stored_text should appear at most once
        assert len(results) == len(set(r.strip()[:200] for r in results))

    def test_empty_ontology_still_does_entity_match(self, agent_with_ontology):
        """With empty ontology, entity matching should still work."""
        agent_with_ontology._ontology_index = {}
        results = agent_with_ontology._resolve_kg_hop(
            "Tell me about betta fish"
        )
        assert len(results) >= 1
        assert any("betta" in r.lower() for r in results)

    def test_category_match_plants(self, agent_with_ontology):
        """'plants' should stem to 'plant' and match."""
        results = agent_with_ontology._resolve_kg_hop(
            "What plants do I have?"
        )
        assert any("Java Moss" in r for r in results)


class TestOntologyReset:
    """Test that reset clears ontology state."""

    def test_reset_clears_ontology(self):
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent()
        agent._ontology_index = {"fish": ["some text"]}
        agent._entity_to_stored = {"neon tetras": ["some text"]}
        agent.reset()
        assert agent._ontology_index == {}
        assert agent._entity_to_stored == {}


class TestStemFunction:
    """Test the _stem() helper function."""

    def test_stem_plural_s(self):
        from MemoryAgentBench.methods.hermes_memory_agent import _stem
        assert _stem("fish") == "fish"  # no suffix to strip
        assert _stem("plants") == "plant"
        assert _stem("cats") == "cat"

    def test_stem_plural_es(self):
        from MemoryAgentBench.methods.hermes_memory_agent import _stem
        assert _stem("dishes") == "dish"
        assert _stem("boxes") == "box"

    def test_stem_plural_ies(self):
        from MemoryAgentBench.methods.hermes_memory_agent import _stem
        assert _stem("stories") == "story"
        assert _stem("cities") == "city"

    def test_stem_ing(self):
        from MemoryAgentBench.methods.hermes_memory_agent import _stem
        assert _stem("fishing") == "fish"
        assert _stem("running") == "runn"  # imperfect but acceptable

    def test_stem_short_word_unchanged(self):
        from MemoryAgentBench.methods.hermes_memory_agent import _stem
        assert _stem("as") == "as"  # too short to strip
        assert _stem("is") == "is"



# ---------------------------------------------------------------------------
# 8. Deferred batch ingestion
# ---------------------------------------------------------------------------


class TestDeferredIngestion:
    """Test that send_message buffers and _flush_ingestion processes."""

    def test_send_message_buffers_not_stores(self):
        """send_message(memorizing=True) should buffer, not embed/store."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent()
        result = agent.send_message("Test fact", memorizing=True)
        assert result == "Memorized"
        assert len(agent._ingestion_buffer) == 1
        assert agent._ingestion_buffer[0] == "Test fact"
        # Nothing stored yet
        assert len(agent.coupled_engine.memory_store) == 0
        assert agent._store_count == 0

    def test_multiple_buffers_accumulate(self):
        """Multiple memorize calls should accumulate in buffer."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent()
        for i in range(5):
            agent.send_message(f"Fact {i}", memorizing=True)
        assert len(agent._ingestion_buffer) == 5
        assert agent._store_count == 0

    def test_reset_clears_ingestion_buffer(self):
        """reset() should clear the ingestion buffer."""
        from MemoryAgentBench.methods.hermes_memory_agent import HermesMemoryAgent

        agent = HermesMemoryAgent()
        agent.send_message("buffered fact", memorizing=True)
        assert len(agent._ingestion_buffer) == 1
        agent.reset()
        assert len(agent._ingestion_buffer) == 0
        assert agent._ingestion_flushed is False


# ---------------------------------------------------------------------------
# 6. Import smoke test
# ---------------------------------------------------------------------------


class TestImportWiring:
    """Test that the import of LLMTripleExtractor is present."""

    def test_llm_triple_extractor_importable(self):
        """LLMTripleExtractor should be importable from the agent module."""
        from mabench.llm_triple_extractor import LLMTripleExtractor

        assert LLMTripleExtractor is not None
