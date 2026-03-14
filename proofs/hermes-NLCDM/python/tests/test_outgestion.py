"""Discriminative tests for outgestion: does enriching LLM context with
metadata signals improve answer quality?

Core hypothesis: the LLM context is a lossy compression bottleneck.
CoupledEngine stores rich metadata (recency, importance, access_count,
layer, fact_type) but retrieval results only surface text. The LLM
can't resolve conflicts, weigh confidence, or understand temporal
ordering because all that signal is lost at the context boundary.

Test 1: TEMPORAL ORDERING
  Setup: Store facts with known recency values, some contradicting.
  Baseline: flat text, no ordering, no annotations.
  Treatment: sorted oldest→newest, [older]/[newer] tags, system prompt
             instructs "prefer newest when facts conflict."
  Discriminative prediction: treatment resolves conflicts correctly;
  baseline picks whichever fact appears first (or matches training data).

All tests run on CPU. No GPU, no external models, no LLM calls.
Embeddings are synthetic unit vectors with controlled recency values.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coupled_engine import CoupledEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(dim: int = 8) -> CoupledEngine:
    return CoupledEngine(dim=dim)


def _unit_vec(dim: int, idx: int, noise: float = 0.05) -> np.ndarray:
    """Unit vector along axis `idx` with small noise."""
    v = np.zeros(dim, dtype=np.float64)
    v[idx % dim] = 1.0
    rng = np.random.RandomState(idx)
    v += rng.randn(dim) * noise
    return v / np.linalg.norm(v)


def _similar_vec(base: np.ndarray, noise: float = 0.1, seed: int = 0) -> np.ndarray:
    """Vector similar to `base` with controlled noise."""
    rng = np.random.RandomState(seed)
    v = base + rng.randn(len(base)) * noise
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Test 1: Metadata envelope is populated
# ---------------------------------------------------------------------------

class TestMetadataEnvelope:
    """Verify that retrieval results include the full metadata envelope."""

    def test_query_cooc_boost_returns_metadata(self):
        """query_cooc_boost should return recency, importance, etc."""
        e = _make_engine()
        e.store("Born in Lima", _unit_vec(8, 0), recency=100.0)
        e.store("Born in Madrid", _similar_vec(_unit_vec(8, 0), seed=1), recency=500.0)

        results = e.query_cooc_boost(_unit_vec(8, 0), top_k=2)
        assert len(results) == 2

        for r in results:
            assert "recency" in r, "Missing recency in result"
            assert "importance" in r, "Missing importance in result"
            assert "access_count" in r, "Missing access_count in result"
            assert "layer" in r, "Missing layer in result"
            assert "fact_type" in r, "Missing fact_type in result"
            assert isinstance(r["recency"], float)
            assert isinstance(r["importance"], float)

    def test_recency_values_preserved(self):
        """Stored recency values should round-trip through query."""
        e = _make_engine()
        e.store("Fact A", _unit_vec(8, 0), recency=42.0)
        e.store("Fact B", _similar_vec(_unit_vec(8, 0), seed=2), recency=999.0)

        results = e.query_cooc_boost(_unit_vec(8, 0), top_k=2)
        recencies = {r["text"]: r["recency"] for r in results}
        assert recencies["Fact A"] == 42.0
        assert recencies["Fact B"] == 999.0

    def test_metadata_across_query_methods(self):
        """All query methods should return the metadata envelope."""
        e = _make_engine()
        emb = _unit_vec(8, 0)
        e.store("Test fact", emb, recency=10.0)

        # Test basic query path
        results = e.query_cooc_boost(emb, top_k=1)
        assert "recency" in results[0]

        # Test with different cooc_weight
        results = e.query_cooc_boost(emb, top_k=1, cooc_weight=0.5)
        assert "recency" in results[0]


# ---------------------------------------------------------------------------
# Test 2: Temporal ordering formatter
# ---------------------------------------------------------------------------

class TestTemporalOrdering:
    """Verify the outgestion formatter sorts by recency and annotates."""

    def _make_agent_minimal(self, temporal_context: bool = False):
        """Create a minimal HermesMemoryAgent-like object for testing
        the _format_outgestion method without LLM dependencies."""
        from mabench.hermes_agent import HermesMemoryAgent

        class MinimalAgent:
            """Stub that only has _format_outgestion."""
            def __init__(self, tc):
                self.temporal_context = tc
                # Bind the real method
                self._format_outgestion = HermesMemoryAgent._format_outgestion.__get__(self)

        return MinimalAgent(temporal_context)

    def test_baseline_no_ordering(self):
        """Without temporal_context, output is flat text in original order."""
        agent = self._make_agent_minimal(temporal_context=False)
        chunks = ["Fact C", "Fact A", "Fact B"]
        metadata = {
            "Fact C": {"recency": 300.0},
            "Fact A": {"recency": 100.0},
            "Fact B": {"recency": 200.0},
        }
        context, instructions = agent._format_outgestion(chunks, metadata)
        assert context == "Fact C\n\nFact A\n\nFact B"
        assert instructions == ""

    def test_temporal_sorts_by_recency(self):
        """With temporal_context, facts should be sorted oldest→newest."""
        agent = self._make_agent_minimal(temporal_context=True)
        chunks = ["Fact C", "Fact A", "Fact B"]
        metadata = {
            "Fact C": {"recency": 300.0},
            "Fact A": {"recency": 100.0},
            "Fact B": {"recency": 200.0},
        }
        context, instructions = agent._format_outgestion(chunks, metadata)

        # Should be sorted: A (100) → B (200) → C (300)
        lines = context.split("\n\n")
        assert "Fact A" in lines[0], f"Expected oldest first, got: {lines}"
        assert "Fact B" in lines[1]
        assert "Fact C" in lines[2]

    def test_temporal_adds_annotations(self):
        """Temporal context should add [fact N, older]/[fact N, newer] tags."""
        agent = self._make_agent_minimal(temporal_context=True)
        # Need enough facts for the 1/3 split to produce tags
        chunks = [f"Fact {i}" for i in range(6)]
        metadata = {
            f"Fact {i}": {"recency": float(i * 100)}
            for i in range(6)
        }
        context, instructions = agent._format_outgestion(chunks, metadata)
        lines = context.split("\n\n")

        # First third should have [fact N, older]
        assert "older]" in lines[0], f"Expected older tag: {lines[0]}"
        assert "older]" in lines[1], f"Expected older tag: {lines[1]}"
        # Last third should have [fact N, newer]
        assert "newer]" in lines[-1], f"Expected newer tag: {lines[-1]}"
        assert "newer]" in lines[-2], f"Expected newer tag: {lines[-2]}"
        # All facts should have fact numbering
        for i, line in enumerate(lines):
            assert f"[fact {i + 1}" in line, f"Missing fact number: {line}"

    def test_temporal_system_instructions(self):
        """Temporal context should produce system prompt instructions."""
        agent = self._make_agent_minimal(temporal_context=True)
        chunks = ["Fact A"]
        metadata = {"Fact A": {"recency": 1.0}}
        _, instructions = agent._format_outgestion(chunks, metadata)
        assert "oldest to newest" in instructions or "oldest" in instructions
        assert "newer" in instructions.lower() or "recent" in instructions.lower()

    def test_missing_metadata_gets_zero_recency(self):
        """Chunks without metadata (e.g. from triadic expansion) get recency=0."""
        agent = self._make_agent_minimal(temporal_context=True)
        chunks = ["Known fact", "Triadic expansion"]
        metadata = {"Known fact": {"recency": 500.0}}
        # "Triadic expansion" has no metadata entry
        context, _ = agent._format_outgestion(chunks, metadata)
        lines = context.split("\n\n")
        # Triadic (recency=0) should sort before Known (recency=500)
        assert "Triadic" in lines[0], f"Zero-recency should sort first: {lines}"


# ---------------------------------------------------------------------------
# Test 3: Discriminative — temporal ordering resolves conflicts
# ---------------------------------------------------------------------------

class TestTemporalConflictResolution:
    """Discriminative test: does temporal ordering let a downstream consumer
    resolve fact conflicts that flat text cannot?

    We don't call an LLM — instead we verify the STRUCTURE of the context
    that would be sent to one. The discriminative signal is:
      - In flat text, conflicting facts appear in retrieval-score order
        (which is arbitrary w.r.t. truth).
      - In temporal text, the NEWEST fact appears last with [newer] tag,
        making the correct answer unambiguous.
    """

    def test_conflicting_facts_resolved_by_position(self):
        """Newer fact should appear AFTER older fact in temporal context."""
        e = _make_engine()
        base_emb = _unit_vec(8, 0)

        # Older fact (recency=100): "Born in Lima"
        e.store("Born in Lima", _similar_vec(base_emb, noise=0.02, seed=10),
                recency=100.0)
        # Newer fact (recency=500): "Born in Madrid" (updated info)
        e.store("Born in Madrid", _similar_vec(base_emb, noise=0.02, seed=11),
                recency=500.0)
        # Unrelated fact
        e.store("Studied at MIT", _unit_vec(8, 3), recency=300.0)

        results = e.query_cooc_boost(base_emb, top_k=3)

        # Build metadata lookup (same as hermes_agent._query does)
        metadata_by_key = {}
        for r in results:
            k = r["text"].strip()[:200]
            metadata_by_key[k] = r

        texts = [r["text"] for r in results]

        # Temporal formatter
        from mabench.hermes_agent import HermesMemoryAgent

        class StubAgent:
            temporal_context = True
            _format_outgestion = HermesMemoryAgent._format_outgestion.__get__(None, HermesMemoryAgent)

        stub = StubAgent()
        stub._format_outgestion = HermesMemoryAgent._format_outgestion.__get__(stub)

        temporal_context, instructions = stub._format_outgestion(
            texts, metadata_by_key
        )

        # The newer fact ("Born in Madrid", recency=500) should appear
        # AFTER the older fact ("Born in Lima", recency=100)
        lima_pos = temporal_context.find("Born in Lima")
        madrid_pos = temporal_context.find("Born in Madrid")
        assert lima_pos < madrid_pos, (
            f"Newer fact should appear after older fact in temporal context. "
            f"Lima@{lima_pos}, Madrid@{madrid_pos}"
        )

        # Instructions should tell LLM to prefer newer
        assert "newer" in instructions.lower() or "recent" in instructions.lower()

    def test_baseline_does_not_guarantee_order(self):
        """Without temporal context, order follows retrieval score, not recency."""
        e = _make_engine()
        base_emb = _unit_vec(8, 0)

        # Store facts with recency, but different similarity to query
        e.store("Born in Lima", _similar_vec(base_emb, noise=0.01, seed=20),
                recency=100.0)  # older, slightly more similar
        e.store("Born in Madrid", _similar_vec(base_emb, noise=0.08, seed=21),
                recency=500.0)  # newer, less similar

        results = e.query_cooc_boost(base_emb, top_k=2)
        texts = [r["text"] for r in results]
        metadata_by_key = {r["text"].strip()[:200]: r for r in results}

        from mabench.hermes_agent import HermesMemoryAgent

        class StubAgent:
            temporal_context = False
            _format_outgestion = None

        stub = StubAgent()
        stub._format_outgestion = HermesMemoryAgent._format_outgestion.__get__(stub)

        baseline_context, instructions = stub._format_outgestion(
            texts, metadata_by_key
        )

        # Baseline: no instructions, original order preserved
        assert instructions == ""
        # Order follows score, not recency — Lima (more similar) comes first
        assert baseline_context.index("Born in Lima") < baseline_context.index("Born in Madrid")


# ---------------------------------------------------------------------------
# Test 4: End-to-end metadata flow
# ---------------------------------------------------------------------------

class TestMetadataFlow:
    """Verify metadata survives from store → query → outgestion formatter."""

    def test_full_pipeline(self):
        """Store facts → query → format with temporal → verify annotations."""
        e = _make_engine()
        dim = 8

        # Store 5 facts with known recency values
        facts = [
            ("Capital is Paris", _unit_vec(dim, 0), 100.0),
            ("Capital is Berlin", _similar_vec(_unit_vec(dim, 0), seed=30), 200.0),
            ("Population is 67M", _similar_vec(_unit_vec(dim, 0), noise=0.15, seed=31), 300.0),
            ("Capital changed to Lyon", _similar_vec(_unit_vec(dim, 0), seed=32), 400.0),
            ("Official language French", _similar_vec(_unit_vec(dim, 0), noise=0.12, seed=33), 500.0),
        ]
        for text, emb, recency in facts:
            e.store(text, emb, recency=recency)

        # Query
        results = e.query_cooc_boost(_unit_vec(dim, 0), top_k=5)
        assert len(results) >= 4  # at least 4 should be similar enough

        # Build metadata lookup
        metadata_by_key = {r["text"].strip()[:200]: r for r in results}
        texts = [r["text"] for r in results]

        # Format with temporal
        from mabench.hermes_agent import HermesMemoryAgent

        class StubAgent:
            temporal_context = True

        stub = StubAgent()
        stub._format_outgestion = HermesMemoryAgent._format_outgestion.__get__(stub)

        context, instructions = stub._format_outgestion(texts, metadata_by_key)

        # Verify sorted by recency — strip fact number + temporal tags
        import re
        lines = [re.sub(r'^\[fact \d+.*?\]\s*', '', l)
                 for l in context.split("\n\n")]

        recency_order = []
        for line in lines:
            clean = line.strip()
            for text, _, rec in facts:
                if text in clean:
                    recency_order.append(rec)
                    break

        # Recency values should be monotonically increasing
        for i in range(len(recency_order) - 1):
            assert recency_order[i] <= recency_order[i + 1], (
                f"Recency not sorted: {recency_order}"
            )

        # Verify instructions present
        assert len(instructions) > 0


# ---------------------------------------------------------------------------
# Test 5: Contradiction detection
# ---------------------------------------------------------------------------

class TestSoftContradictionSurfacing:
    """Verify that high-similarity chunks with different recency values
    are cross-referenced with [CONFLICTS WITH] tags on BOTH facts.

    The key insight: surface conflicts, never remove content. Both facts
    remain visible, cross-referenced by fact number, with temporal ordering
    as the primary resolution signal.
    """

    def _make_agent_minimal(self, temporal_context=True,
                            contradiction_context=True,
                            contradiction_sim_threshold=0.85):
        from mabench.hermes_agent import HermesMemoryAgent

        class MinimalAgent:
            def __init__(self, tc, cc, cst):
                self.temporal_context = tc
                self.contradiction_context = cc
                self.contradiction_sim_threshold = cst
                self._format_outgestion = HermesMemoryAgent._format_outgestion.__get__(self)

        return MinimalAgent(temporal_context, contradiction_context,
                            contradiction_sim_threshold)

    def test_conflicting_facts_cross_referenced(self):
        """Two similar embeddings with different recency → both get CONFLICTS WITH."""
        agent = self._make_agent_minimal()
        dim = 8
        emb_base = _unit_vec(dim, 0)
        emb_similar = _similar_vec(emb_base, noise=0.02, seed=50)
        emb_unrelated = _unit_vec(dim, 3)

        chunks = ["Born in Lima", "Born in Madrid", "Studied at MIT"]
        metadata = {
            "Born in Lima": {"recency": 100.0, "embedding": emb_base},
            "Born in Madrid": {"recency": 500.0, "embedding": emb_similar},
            "Studied at MIT": {"recency": 300.0, "embedding": emb_unrelated},
        }

        context, instructions = agent._format_outgestion(chunks, metadata)

        # Both Lima AND Madrid should have CONFLICTS WITH tags
        lines = context.split("\n\n")
        lima_line = [l for l in lines if "Born in Lima" in l][0]
        madrid_line = [l for l in lines if "Born in Madrid" in l][0]
        assert "CONFLICTS WITH" in lima_line, f"Lima should be cross-referenced: {lima_line}"
        assert "CONFLICTS WITH" in madrid_line, f"Madrid should be cross-referenced: {madrid_line}"

        # Both facts should still be present (not removed)
        assert "Born in Lima" in context
        assert "Born in Madrid" in context

        # MIT (unrelated) should NOT have CONFLICTS WITH tag
        mit_line = [l for l in lines if "Studied at MIT" in l][0]
        assert "CONFLICTS WITH" not in mit_line

    def test_cross_reference_fact_numbers(self):
        """Conflicting facts should reference each other by fact number."""
        agent = self._make_agent_minimal()
        dim = 8
        emb = _unit_vec(dim, 0)

        chunks = ["Born in Lima", "Born in Madrid"]
        metadata = {
            "Born in Lima": {"recency": 100.0, "embedding": emb},
            "Born in Madrid": {"recency": 500.0, "embedding": _similar_vec(emb, noise=0.02, seed=54)},
        }
        context, _ = agent._format_outgestion(chunks, metadata)
        lines = context.split("\n\n")

        # Lima is fact 1 (older, sorted first), Madrid is fact 2
        assert "[fact 1, CONFLICTS WITH fact 2]" in lines[0], f"Wrong cross-ref: {lines[0]}"
        assert "[fact 2, CONFLICTS WITH fact 1]" in lines[1], f"Wrong cross-ref: {lines[1]}"

    def test_no_contradiction_without_flag(self):
        """With contradiction_context=False, no CONFLICTS WITH tags."""
        agent = self._make_agent_minimal(contradiction_context=False)
        dim = 8
        emb = _unit_vec(dim, 0)

        chunks = ["Born in Lima", "Born in Madrid"]
        metadata = {
            "Born in Lima": {"recency": 100.0, "embedding": emb},
            "Born in Madrid": {"recency": 500.0, "embedding": _similar_vec(emb, noise=0.02, seed=51)},
        }
        context, _ = agent._format_outgestion(chunks, metadata)
        assert "CONFLICTS WITH" not in context

    def test_low_similarity_not_flagged(self):
        """Dissimilar chunks should not be flagged even if recency differs."""
        agent = self._make_agent_minimal(contradiction_sim_threshold=0.85)
        dim = 8

        chunks = ["Born in Lima", "Studied at MIT"]
        metadata = {
            "Born in Lima": {"recency": 100.0, "embedding": _unit_vec(dim, 0)},
            "Studied at MIT": {"recency": 500.0, "embedding": _unit_vec(dim, 3)},
        }
        context, _ = agent._format_outgestion(chunks, metadata)
        assert "CONFLICTS WITH" not in context

    def test_same_recency_not_flagged(self):
        """Similar chunks with same recency should not be flagged."""
        agent = self._make_agent_minimal()
        dim = 8
        emb = _unit_vec(dim, 0)

        chunks = ["Born in Lima", "Born in Madrid"]
        metadata = {
            "Born in Lima": {"recency": 100.0, "embedding": emb},
            "Born in Madrid": {"recency": 100.0, "embedding": _similar_vec(emb, noise=0.02, seed=52)},
        }
        context, _ = agent._format_outgestion(chunks, metadata)
        assert "CONFLICTS WITH" not in context

    def test_instructions_mention_conflicts(self):
        """When contradictions found, instructions should mention CONFLICTS WITH."""
        agent = self._make_agent_minimal()
        dim = 8
        emb = _unit_vec(dim, 0)

        chunks = ["Born in Lima", "Born in Madrid"]
        metadata = {
            "Born in Lima": {"recency": 100.0, "embedding": emb},
            "Born in Madrid": {"recency": 500.0, "embedding": _similar_vec(emb, noise=0.02, seed=53)},
        }
        _, instructions = agent._format_outgestion(chunks, metadata)
        assert "CONFLICTS WITH" in instructions
        assert "prefer" in instructions.lower()
        # Should NOT mention "ignore" — soft signals don't remove content
        assert "ignore" not in instructions.lower()

    def test_contradiction_with_engine_embeddings(self):
        """End-to-end: store in CoupledEngine, query, check CONFLICTS WITH tags."""
        e = _make_engine()
        dim = 8
        base_emb = _unit_vec(dim, 0)

        # Two conflicting facts about the same subject
        e.store("Capital is Paris", _similar_vec(base_emb, noise=0.02, seed=60),
                recency=100.0)
        e.store("Capital is Berlin", _similar_vec(base_emb, noise=0.02, seed=61),
                recency=500.0)
        # Unrelated
        e.store("Population is 67M", _unit_vec(dim, 4), recency=300.0)

        results = e.query_cooc_boost(base_emb, top_k=3)
        metadata_by_key = {r["text"].strip()[:200]: r for r in results}
        texts = [r["text"] for r in results]

        from mabench.hermes_agent import HermesMemoryAgent

        class StubAgent:
            temporal_context = True
            contradiction_context = True
            contradiction_sim_threshold = 0.85

        stub = StubAgent()
        stub._format_outgestion = HermesMemoryAgent._format_outgestion.__get__(stub)

        context, instructions = stub._format_outgestion(texts, metadata_by_key)

        # Both Paris and Berlin should have CONFLICTS WITH tags
        lines = context.split("\n\n")
        paris_line = [l for l in lines if "Paris" in l][0]
        berlin_line = [l for l in lines if "Berlin" in l][0]
        assert "CONFLICTS WITH" in paris_line, f"Paris should be cross-referenced: {paris_line}"
        assert "CONFLICTS WITH" in berlin_line, f"Berlin should be cross-referenced: {berlin_line}"

        # Population line should not be tagged as contradiction
        pop_lines = [l for l in lines if "Population" in l]
        if pop_lines:
            assert "CONFLICTS WITH" not in pop_lines[0]

    def test_missing_embedding_skips_contradiction(self):
        """Chunks without embeddings in metadata should not crash."""
        agent = self._make_agent_minimal()
        chunks = ["Fact A", "Fact B"]
        metadata = {
            "Fact A": {"recency": 100.0},  # no embedding
            "Fact B": {"recency": 500.0},  # no embedding
        }
        context, _ = agent._format_outgestion(chunks, metadata)
        # Should not crash, no contradiction tags
        assert "CONFLICTS WITH" not in context

    def test_all_facts_numbered(self):
        """All facts should have [fact N] numbering when temporal is enabled."""
        agent = self._make_agent_minimal(contradiction_context=False)
        chunks = [f"Fact {i}" for i in range(4)]
        metadata = {
            f"Fact {i}": {"recency": float(i * 100)}
            for i in range(4)
        }
        context, _ = agent._format_outgestion(chunks, metadata)
        lines = context.split("\n\n")
        for i, line in enumerate(lines):
            assert f"[fact {i + 1}" in line, f"Missing fact number in: {line}"
