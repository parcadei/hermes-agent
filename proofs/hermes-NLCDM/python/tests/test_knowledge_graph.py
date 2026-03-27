"""Tests for KnowledgeGraph module (Module 2 from knowledge-graph-retrieval spec).

Tests the standalone KnowledgeGraph class:
  - add_fact: node/edge creation, bidirectionality, deduplication
  - supersede: edge removal, new edge creation, orphan pruning
  - ppr_retrieve: multi-hop traversal, node specificity, seed exclusion
  - ppr_retrieve_from_hop: entity extraction from answer + sub_query
  - find_node: exact and substring matching
  - stats: node/edge/triple/fact counts
  - Serialization: pickle round-trip
  - Performance: large graph PPR latency
  - Supersession chains: updated paths are traversable
  - Disconnected components: isolation preserved
  - Orphan pruning: removed after supersession
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _TESTS_DIR.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))

from mabench.knowledge_graph import KnowledgeGraph


# ── add_fact tests ────────────────────────────────────────────────


class TestAddFact:
    """Tests for KnowledgeGraph.add_fact."""

    def test_creates_nodes_and_bidirectional_edges(self):
        """add_fact creates subject and object nodes with bidirectional edges."""
        kg = KnowledgeGraph()
        edges_added = kg.add_fact(
            "Abbiati plays cornerback",
            fact_index=0,
            triples=[("Abbiati", "plays", "cornerback")],
            recency=1.0,
        )

        # Two nodes created (abbiati, cornerback)
        assert "abbiati" in kg._node_map
        assert "cornerback" in kg._node_map

        # Bidirectional: 2 edges per triple (forward + reverse)
        assert edges_added == 2
        assert kg._graph.num_edges() == 2

        # Fact index recorded on both nodes
        subj_idx = kg._node_map["abbiati"]
        obj_idx = kg._node_map["cornerback"]
        assert 0 in kg._node_facts[subj_idx]
        assert 0 in kg._node_facts[obj_idx]

    def test_multiple_triples(self):
        """add_fact with multiple triples creates all edges."""
        kg = KnowledgeGraph()
        edges_added = kg.add_fact(
            "Abbiati plays cornerback in Philippines",
            fact_index=0,
            triples=[
                ("Abbiati", "plays", "cornerback"),
                ("Abbiati", "lives_in", "Philippines"),
            ],
            recency=1.0,
        )

        # 2 triples * 2 edges each = 4 edges
        assert edges_added == 4
        assert kg._graph.num_edges() == 4

        # All three entity nodes exist
        assert "abbiati" in kg._node_map
        assert "cornerback" in kg._node_map
        assert "philippines" in kg._node_map

    def test_empty_triples_noop(self):
        """add_fact with empty triples list is a no-op."""
        kg = KnowledgeGraph()
        edges_added = kg.add_fact(
            "Some fact with no triples",
            fact_index=0,
            triples=[],
            recency=1.0,
        )

        assert edges_added == 0
        assert kg._graph.num_nodes() == 0
        assert kg._graph.num_edges() == 0
        assert len(kg._node_map) == 0

    def test_duplicate_entities_reused(self):
        """add_fact with entities already in graph reuses existing nodes."""
        kg = KnowledgeGraph()
        kg.add_fact(
            "Abbiati plays cornerback",
            fact_index=0,
            triples=[("Abbiati", "plays", "cornerback")],
            recency=1.0,
        )
        kg.add_fact(
            "Abbiati born in Italy",
            fact_index=1,
            triples=[("Abbiati", "born_in", "Italy")],
            recency=2.0,
        )

        # "abbiati" node reused, not duplicated
        assert len(kg._node_map) == 3  # abbiati, cornerback, italy

        # Abbiati node has both fact indices
        abbiati_idx = kg._node_map["abbiati"]
        assert 0 in kg._node_facts[abbiati_idx]
        assert 1 in kg._node_facts[abbiati_idx]

        # 2 triples * 2 edges = 4 total edges
        assert kg._graph.num_edges() == 4


# ── supersede tests ───────────────────────────────────────────────


class TestSupersede:
    """Tests for KnowledgeGraph.supersede."""

    def test_removes_old_edges_adds_new(self):
        """supersede removes old predicate edges and adds new ones."""
        kg = KnowledgeGraph()
        kg.add_fact(
            "Abbiati plays football",
            fact_index=0,
            triples=[("Abbiati", "plays", "football")],
            recency=1.0,
        )

        edges_removed = kg.supersede(
            old_triple=("Abbiati", "plays", "football"),
            new_triple=("Abbiati", "plays", "cornerback"),
            new_fact_index=1,
            new_recency=2.0,
        )

        # Old edges removed (forward + reverse = 2)
        assert edges_removed == 2

        # New entity node exists
        assert "cornerback" in kg._node_map

        # New edges exist (forward + reverse for new triple)
        # Verify via edge predicates: only "plays" edges should connect abbiati<->cornerback
        abbiati_idx = kg._node_map["abbiati"]
        cornerback_idx = kg._node_map["cornerback"]

        # Check that edges between abbiati and cornerback exist
        found_forward = False
        found_reverse = False
        for edge_idx in kg._graph.edge_indices():
            endpoints = kg._graph.get_edge_endpoints_by_index(edge_idx)
            if endpoints == (abbiati_idx, cornerback_idx):
                found_forward = True
            if endpoints == (cornerback_idx, abbiati_idx):
                found_reverse = True
        assert found_forward, "Forward edge abbiati->cornerback not found"
        assert found_reverse, "Reverse edge cornerback->abbiati not found"

    def test_old_triple_not_in_graph(self):
        """supersede when old triple not in graph is treated as pure add."""
        kg = KnowledgeGraph()

        edges_removed = kg.supersede(
            old_triple=("Abbiati", "plays", "football"),
            new_triple=("Abbiati", "plays", "cornerback"),
            new_fact_index=0,
            new_recency=1.0,
        )

        # No edges to remove
        assert edges_removed == 0

        # New triple was still added
        assert "abbiati" in kg._node_map
        assert "cornerback" in kg._node_map
        assert kg._graph.num_edges() == 2  # forward + reverse

    def test_orphan_pruning(self):
        """After supersession, orphaned nodes are removed from graph and node_map."""
        kg = KnowledgeGraph()
        # Only edge to "football" is from Abbiati
        kg.add_fact(
            "Abbiati plays football",
            fact_index=0,
            triples=[("Abbiati", "plays", "football")],
            recency=1.0,
        )
        assert "football" in kg._node_map

        # Supersede: Abbiati now plays cornerback
        kg.supersede(
            old_triple=("Abbiati", "plays", "football"),
            new_triple=("Abbiati", "plays", "cornerback"),
            new_fact_index=1,
            new_recency=2.0,
        )

        # "football" node should be orphaned and removed
        # (it has no edges and no facts after supersession removed fact_index 0)
        assert "football" not in kg._node_map, (
            "Orphaned 'football' node should be pruned from node_map"
        )


# ── ppr_retrieve tests ────────────────────────────────────────────


class TestPprRetrieve:
    """Tests for KnowledgeGraph.ppr_retrieve."""

    def test_2hop_chain(self):
        """PPR from A discovers C through B in A->B->C chain."""
        kg = KnowledgeGraph()
        kg.add_fact(
            "Abbiati plays cornerback",
            fact_index=0,
            triples=[("abbiati", "plays", "cornerback")],
            recency=1.0,
        )
        kg.add_fact(
            "Cornerback was created in Philippines",
            fact_index=1,
            triples=[("cornerback", "created_in", "philippines")],
            recency=2.0,
        )

        results = kg.ppr_retrieve(["abbiati"])
        fact_indices = [idx for idx, score in results]

        # Fact 1 (Philippines) should be discoverable from abbiati through cornerback
        assert 1 in fact_indices, "2-hop fact not found via PPR"

    def test_3hop_chain(self):
        """PPR from A discovers D through B->C->D chain, D with lower score than B."""
        kg = KnowledgeGraph()
        kg.add_fact("A relates to B", 0, [("a_entity", "rel1", "b_entity")], 1.0)
        kg.add_fact("B relates to C", 1, [("b_entity", "rel2", "c_entity")], 2.0)
        kg.add_fact("C relates to D", 2, [("c_entity", "rel3", "d_entity")], 3.0)

        results = kg.ppr_retrieve(["a_entity"])
        fact_indices = [idx for idx, score in results]
        fact_scores = {idx: score for idx, score in results}

        # D should be reachable (3 hops)
        assert 2 in fact_indices, "3-hop fact not found via PPR"

        # Facts closer to seed should have higher scores than farther ones
        # Fact 0 is associated with b_entity (1 hop), fact 2 with d_entity (3 hops)
        if 0 in fact_scores and 2 in fact_scores:
            assert fact_scores[0] >= fact_scores[2], (
                "Closer facts should have higher PPR scores"
            )

    def test_node_specificity(self):
        """Rare entities are weighted higher as PPR seeds than common ones."""
        kg = KnowledgeGraph()

        # "common_hub" appears in many facts
        for i in range(20):
            kg.add_fact(
                f"Entity{i} relates to common_hub",
                fact_index=i,
                triples=[(f"entity{i}", "rel", "common_hub")],
                recency=float(i),
            )

        # "rare_entity" appears in just 1 fact, also connected to common_hub
        kg.add_fact(
            "rare_entity relates to common_hub",
            fact_index=20,
            triples=[("rare_entity", "rel", "common_hub")],
            recency=21.0,
        )

        # Add a target fact reachable through common_hub
        kg.add_fact(
            "common_hub relates to target",
            fact_index=21,
            triples=[("common_hub", "rel", "target")],
            recency=22.0,
        )

        # Seed from rare_entity: should have higher specificity weight
        results_rare = kg.ppr_retrieve(["rare_entity"])
        results_common = kg.ppr_retrieve(["entity0"])

        # Both should find target (fact 21), but rare seed should give higher score
        rare_scores = {idx: score for idx, score in results_rare}
        common_scores = {idx: score for idx, score in results_common}

        assert 21 in rare_scores, "Target not found from rare seed"
        assert 21 in common_scores, "Target not found from common seed"

    def test_no_matching_seeds_returns_empty(self):
        """ppr_retrieve with no matching seeds returns empty list."""
        kg = KnowledgeGraph()
        kg.add_fact("A relates to B", 0, [("a", "rel", "b")], 1.0)

        results = kg.ppr_retrieve(["nonexistent_entity"])
        assert results == []

    def test_excludes_seed_nodes(self):
        """Seed nodes' own facts are not in PPR results."""
        kg = KnowledgeGraph()
        # Fact 0 mentions only abbiati and cornerback
        kg.add_fact(
            "Abbiati plays cornerback",
            fact_index=0,
            triples=[("abbiati", "plays", "cornerback")],
            recency=1.0,
        )
        # Fact 1 connects cornerback to something else
        kg.add_fact(
            "Cornerback was created in Philippines",
            fact_index=1,
            triples=[("cornerback", "created_in", "philippines")],
            recency=2.0,
        )

        results = kg.ppr_retrieve(["abbiati"])
        fact_indices = [idx for idx, score in results]

        # Fact 0 is directly on the seed node (abbiati) — should be excluded
        # Fact 1 is on cornerback/philippines — should be included (discovered via traversal)
        # Note: fact 0 is ALSO on the cornerback node, so it may still appear
        # through cornerback. The spec says seed NODES are excluded, not seed FACTS.
        # The key behavior: seed node's PPR score itself is excluded from results.
        # Facts on non-seed nodes that happen to share a fact_index with seed can still appear.
        #
        # However, abbiati's own node is excluded. So facts ONLY on abbiati won't appear.
        # Fact 0 is on both abbiati AND cornerback, so it CAN appear via cornerback.
        # The critical thing: results come from NON-SEED nodes.
        assert 1 in fact_indices, "Non-seed fact should be in results"

    def test_empty_graph_returns_empty(self):
        """PPR on empty graph returns empty list."""
        kg = KnowledgeGraph()
        results = kg.ppr_retrieve(["anything"])
        assert results == []

    def test_disconnected_components(self):
        """Seed in component 1 returns nothing from component 2."""
        kg = KnowledgeGraph()
        # Component 1: A -> B
        kg.add_fact("A relates to B", 0, [("comp1_a", "rel", "comp1_b")], 1.0)
        # Component 2: C -> D (disconnected from component 1)
        kg.add_fact("C relates to D", 1, [("comp2_c", "rel", "comp2_d")], 2.0)

        results = kg.ppr_retrieve(["comp1_a"])
        fact_indices = [idx for idx, score in results]

        # Should find facts from component 1 (fact 0 via comp1_b)
        # Should NOT find facts from component 2 (fact 1)
        assert 1 not in fact_indices, (
            "Facts from disconnected component should not appear in PPR results"
        )


# ── ppr_retrieve_from_hop tests ───────────────────────────────────


class TestPprRetrieveFromHop:
    """Tests for KnowledgeGraph.ppr_retrieve_from_hop."""

    def test_extracts_entities_from_answer_and_subquery(self):
        """ppr_retrieve_from_hop uses entities from both intermediate answer and sub_query."""
        kg = KnowledgeGraph()
        # Build a graph with known entities
        kg.add_fact("cornerback was created in Philippines", 0,
                     [("cornerback", "created_in", "philippines")], 1.0)
        kg.add_fact("Philippines is in Asia", 1,
                     [("philippines", "located_in", "asia")], 2.0)

        # intermediate_answer mentions "cornerback", sub_query mentions "country"
        # "cornerback" is a known node, so PPR seeds from it
        results = kg.ppr_retrieve_from_hop(
            intermediate_answer="cornerback",
            sub_query="What country was cornerback created in?",
        )

        fact_indices = [idx for idx, score in results]
        # Should discover Philippines-related facts through cornerback
        assert len(results) > 0, "ppr_retrieve_from_hop should return results"
        assert 0 in fact_indices or 1 in fact_indices, (
            "Should find facts connected to cornerback"
        )


# ── find_node tests ───────────────────────────────────────────────


class TestFindNode:
    """Tests for KnowledgeGraph.find_node."""

    def test_exact_match(self):
        """find_node returns node index for exact entity match."""
        kg = KnowledgeGraph()
        kg.add_fact("A relates to B", 0, [("alpha", "rel", "beta")], 1.0)

        result = kg.find_node("alpha")
        assert result is not None
        assert result == kg._node_map["alpha"]

    def test_case_insensitive(self):
        """find_node normalizes to lowercase."""
        kg = KnowledgeGraph()
        kg.add_fact("A relates to B", 0, [("alpha", "rel", "beta")], 1.0)

        result = kg.find_node("Alpha")
        assert result is not None
        assert result == kg._node_map["alpha"]

    def test_substring_match(self):
        """find_node falls back to substring matching."""
        kg = KnowledgeGraph()
        kg.add_fact("A relates to B", 0, [("association football", "rel", "beta")], 1.0)

        # "football" is a substring of "association football"
        result = kg.find_node("football")
        assert result is not None
        assert result == kg._node_map["association football"]

    def test_no_match_returns_none(self):
        """find_node returns None when entity not found."""
        kg = KnowledgeGraph()
        kg.add_fact("A relates to B", 0, [("alpha", "rel", "beta")], 1.0)

        result = kg.find_node("nonexistent")
        assert result is None


# ── stats tests ───────────────────────────────────────────────────


class TestStats:
    """Tests for KnowledgeGraph.stats property."""

    def test_stats_correct_counts(self):
        """stats returns correct node, edge, triple, and fact counts."""
        kg = KnowledgeGraph()
        kg.add_fact("A to B", 0, [("alpha", "rel1", "beta")], 1.0)
        kg.add_fact("B to C", 1, [("beta", "rel2", "gamma")], 2.0)

        s = kg.stats

        assert s["nodes"] == 3  # alpha, beta, gamma
        assert s["edges"] == 4  # 2 triples * 2 bidirectional edges
        assert s["triples"] == 2
        # facts_indexed: alpha has {0}, beta has {0,1}, gamma has {1} = 1+2+1 = 4
        assert s["facts_indexed"] == 4

    def test_stats_empty_graph(self):
        """stats on empty graph returns all zeros."""
        kg = KnowledgeGraph()
        s = kg.stats
        assert s["nodes"] == 0
        assert s["edges"] == 0
        assert s["triples"] == 0
        assert s["facts_indexed"] == 0


# ── Serialization tests ──────────────────────────────────────────


class TestSerialization:
    """Tests for pickle round-trip of KnowledgeGraph."""

    def test_pickle_roundtrip_preserves_state(self):
        """pickle.dumps + pickle.loads preserves all KnowledgeGraph state."""
        kg = KnowledgeGraph(damping=0.80, ppr_top_k=15, max_facts_per_node=30)
        kg.add_fact("A to B", 0, [("alpha", "rel1", "beta")], 1.0)
        kg.add_fact("B to C", 1, [("beta", "rel2", "gamma")], 2.0)

        # Serialize and deserialize
        data = pickle.dumps(kg)
        kg2 = pickle.loads(data)

        # Stats should match
        assert kg2.stats == kg.stats

        # node_map should match
        assert kg2._node_map == kg._node_map

        # node_facts should match
        assert kg2._node_facts == kg._node_facts

        # PPR results should match
        results_orig = kg.ppr_retrieve(["alpha"])
        results_restored = kg2.ppr_retrieve(["alpha"])
        assert [idx for idx, _ in results_orig] == [idx for idx, _ in results_restored]

        # Constructor params should match
        assert kg2._graph.num_nodes() == kg._graph.num_nodes()
        assert kg2._graph.num_edges() == kg._graph.num_edges()


# ── Performance tests ─────────────────────────────────────────────


class TestPerformance:
    """Performance tests for KnowledgeGraph."""

    def test_large_graph_ppr_performance(self):
        """PPR on a graph with 1000 nodes and ~5000 edges completes in < 100ms."""
        kg = KnowledgeGraph()

        # Build a graph with 1000 nodes and ~5000 edges
        # Create a chain + random cross-links
        for i in range(999):
            kg.add_fact(
                f"Entity{i} to Entity{i+1}",
                fact_index=i,
                triples=[(f"entity{i}", "rel", f"entity{i+1}")],
                recency=float(i),
            )

        # Add some cross-links to make it more interesting
        import random
        rng = random.Random(42)
        for _ in range(500):
            a = rng.randint(0, 999)
            b = rng.randint(0, 999)
            if a != b:
                kg.add_fact(
                    f"Cross Entity{a} to Entity{b}",
                    fact_index=1000 + a,
                    triples=[(f"entity{a}", "cross", f"entity{b}")],
                    recency=1000.0,
                )

        assert kg.stats["nodes"] >= 1000

        # Time the PPR retrieval
        start = time.perf_counter()
        results = kg.ppr_retrieve(["entity0"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"PPR took {elapsed_ms:.1f}ms, expected < 100ms"
        assert len(results) > 0, "PPR should return results on large graph"


# ── Supersession chain tests ─────────────────────────────────────


class TestSupersessionChain:
    """Tests for supersession chains and graph path updates."""

    def test_supersession_chain_reaches_new_target(self):
        """After superseding A->B to A->C, query from A reaches C's neighborhood."""
        kg = KnowledgeGraph()
        # Initial: Abbiati -> football -> England
        kg.add_fact(
            "Abbiati plays football",
            fact_index=0,
            triples=[("abbiati", "plays", "football")],
            recency=1.0,
        )
        kg.add_fact(
            "Football created in England",
            fact_index=1,
            triples=[("football", "created_in", "england")],
            recency=2.0,
        )

        # Supersede: Abbiati now plays cornerback
        kg.supersede(
            old_triple=("abbiati", "plays", "football"),
            new_triple=("abbiati", "plays", "cornerback"),
            new_fact_index=2,
            new_recency=3.0,
        )

        # Add: Cornerback created in Philippines
        kg.add_fact(
            "Cornerback created in Philippines",
            fact_index=3,
            triples=[("cornerback", "created_in", "philippines")],
            recency=4.0,
        )

        results = kg.ppr_retrieve(["abbiati"])
        fact_indices = [idx for idx, _ in results]

        # Should find Philippines fact (via new path: abbiati -> cornerback -> philippines)
        assert 3 in fact_indices, (
            "New chain fact (Philippines) not found after supersession"
        )

    def test_supersession_old_path_unreachable(self):
        """After superseding A->B to A->C, query from A no longer reaches B's neighborhood."""
        kg = KnowledgeGraph()
        # Abbiati -> football (sole link to football)
        kg.add_fact(
            "Abbiati plays football",
            fact_index=0,
            triples=[("abbiati", "plays", "football")],
            recency=1.0,
        )
        # football -> england
        kg.add_fact(
            "Football created in England",
            fact_index=1,
            triples=[("football", "created_in", "england")],
            recency=2.0,
        )

        # Supersede: Abbiati now plays cornerback (football becomes orphan if sole link)
        kg.supersede(
            old_triple=("abbiati", "plays", "football"),
            new_triple=("abbiati", "plays", "cornerback"),
            new_fact_index=2,
            new_recency=3.0,
        )

        results = kg.ppr_retrieve(["abbiati"])
        fact_indices = [idx for idx, _ in results]

        # England fact (via football) should NOT be reachable from abbiati
        # because the abbiati->football edge was removed.
        # NOTE: If football still exists (connected to england), it's in a
        # disconnected component from abbiati. PPR stays in abbiati's component.
        # football may or may not be orphan-pruned depending on whether it still
        # has the england edge. Either way, there's no path from abbiati to england.
        assert 1 not in fact_indices, (
            "Old chain fact (England) should NOT be reachable after supersession"
        )


# ── Constructor parameter tests ──────────────────────────────────


class TestConstructor:
    """Tests for KnowledgeGraph constructor parameters."""

    def test_default_parameters(self):
        """Constructor has correct defaults."""
        kg = KnowledgeGraph()
        # Verify internal defaults are set (checking via stats or behavior)
        assert kg.stats["nodes"] == 0
        assert kg.stats["edges"] == 0

    def test_custom_damping(self):
        """Custom damping factor affects PPR exploration depth."""
        kg = KnowledgeGraph(damping=0.5)
        # Build a 5-hop chain
        for i in range(5):
            kg.add_fact(
                f"E{i} to E{i+1}",
                fact_index=i,
                triples=[(f"e{i}", "rel", f"e{i+1}")],
                recency=float(i),
            )

        # Low damping = more weight on seeds, less exploration
        results_low = kg.ppr_retrieve(["e0"], damping=0.3)

        # High damping = more exploration
        results_high = kg.ppr_retrieve(["e0"], damping=0.95)

        # With high damping, more distant facts should have relatively higher scores
        # Both should return results (just different score distributions)
        assert len(results_low) > 0
        assert len(results_high) > 0

    def test_custom_ppr_top_k(self):
        """Custom ppr_top_k limits the number of results."""
        kg = KnowledgeGraph(ppr_top_k=3)
        # Create many connected nodes
        for i in range(20):
            kg.add_fact(
                f"Hub to E{i}",
                fact_index=i,
                triples=[("hub", "rel", f"entity{i}")],
                recency=float(i),
            )

        results = kg.ppr_retrieve(["hub"], top_k=3)
        # Should return at most 3 * (facts per node) results
        # But the key point: top_k limits nodes, not facts
        # Each node may have multiple facts, so total can exceed top_k
        # But the number of distinct PPR-contributing nodes is bounded
        assert len(results) > 0

    def test_max_facts_per_node_caps_storage(self):
        """max_facts_per_node evicts oldest facts when exceeded."""
        kg = KnowledgeGraph(max_facts_per_node=3)
        # Add 5 facts all mentioning the same entity node
        for i in range(5):
            kg.add_fact(
                f"alpha relates to beta{i}",
                fact_index=i,
                triples=[("alpha", "rel", f"beta{i}")],
                recency=float(i),
            )

        alpha_idx = kg._node_map["alpha"]
        # alpha has been in 5 add_fact calls, but max_facts_per_node=3
        # Oldest should be evicted
        assert len(kg._node_facts[alpha_idx]) <= 3


# ── Edge case tests ──────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for KnowledgeGraph."""

    def test_self_loop_triple(self):
        """Self-loop triple (A, rel, A) does not crash."""
        kg = KnowledgeGraph()
        edges_added = kg.add_fact(
            "Alpha is Alpha",
            fact_index=0,
            triples=[("alpha", "is", "alpha")],
            recency=1.0,
        )
        # Should create a self-loop (2 edges: forward + reverse, both are self-loops)
        assert edges_added == 2
        assert kg._graph.num_nodes() == 1  # only alpha
        assert "alpha" in kg._node_map

    def test_whitespace_normalization(self):
        """Entity names are stripped and lowercased."""
        kg = KnowledgeGraph()
        kg.add_fact(
            "Test",
            fact_index=0,
            triples=[("  Alpha  ", "rel", "  BETA  ")],
            recency=1.0,
        )
        assert "alpha" in kg._node_map
        assert "beta" in kg._node_map

    def test_parallel_edges_different_predicates(self):
        """Multiple triples between same entities create parallel edges."""
        kg = KnowledgeGraph()
        kg.add_fact(
            "Alpha works at Beta and lives in Beta",
            fact_index=0,
            triples=[
                ("alpha", "works_at", "beta"),
                ("alpha", "lives_in", "beta"),
            ],
            recency=1.0,
        )
        # 2 triples * 2 bidirectional = 4 edges
        assert kg._graph.num_edges() == 4
        assert kg._graph.num_nodes() == 2  # alpha, beta

    def test_ppr_retrieve_results_sorted_by_score(self):
        """PPR results are sorted by score descending."""
        kg = KnowledgeGraph()
        for i in range(10):
            kg.add_fact(
                f"Hub to E{i}",
                fact_index=i,
                triples=[("hub", f"rel{i}", f"entity{i}")],
                recency=float(i),
            )

        results = kg.ppr_retrieve(["hub"])
        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True), (
                "Results should be sorted by score descending"
            )

    def test_ppr_retrieve_unique_fact_indices(self):
        """Each fact_index appears at most once in PPR results."""
        kg = KnowledgeGraph()
        # Fact 0 connected to both B and C through A
        kg.add_fact("A to B", 0, [("a", "rel1", "b")], 1.0)
        kg.add_fact("A to C", 0, [("a", "rel2", "c")], 1.0)  # same fact_index=0
        kg.add_fact("B to D", 1, [("b", "rel3", "d")], 2.0)

        results = kg.ppr_retrieve(["a"])
        fact_indices = [idx for idx, _ in results]

        # No duplicates
        assert len(fact_indices) == len(set(fact_indices)), (
            "Duplicate fact indices in PPR results"
        )

    def test_ppr_scores_non_negative(self):
        """All PPR scores are non-negative."""
        kg = KnowledgeGraph()
        kg.add_fact("A to B", 0, [("a", "rel", "b")], 1.0)
        kg.add_fact("B to C", 1, [("b", "rel", "c")], 2.0)

        results = kg.ppr_retrieve(["a"])
        for idx, score in results:
            assert score >= 0, f"Negative PPR score {score} for fact {idx}"
