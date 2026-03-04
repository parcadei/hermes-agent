"""Discriminative tests: dream() should log associations into _co_retrieval.

Currently (as of 2026-03-04), CoupledEngine.dream() returns associations
in its output dict but NEVER writes them to self._co_retrieval. These tests
define the expected behavior: dream-discovered cross-domain edges should
feed into the co-retrieval graph so that query_coretrieval() can expand
through dream-discovered associations, not just query co-occurrence.

All 4 tests are expected to FAIL today. They will PASS once dream()
is updated to log report.associations into self._co_retrieval.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup (consistent with existing test files)
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
sys.path.insert(0, str(_NLCDM_PYTHON))

from coupled_engine import CoupledEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clustered_patterns(
    n_per_cluster: int = 10,
    n_clusters: int = 3,
    dim: int = 64,
    noise: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Create clusters with enough spread for dream cross-domain discovery.

    Each cluster is built around a random unit centroid with Gaussian noise.
    noise=0.3 provides inter-cluster overlap that REM-explore needs to
    discover perturbation-response correlations across domains. The clusters
    are still well-separated (random unit vectors in 64-d have expected
    cosine ~ 0), but individual members have enough variance for dream
    to identify cross-domain associations.
    """
    rng = np.random.default_rng(seed)
    patterns = []
    for _ in range(n_clusters):
        centroid = rng.standard_normal(dim)
        centroid /= np.linalg.norm(centroid)
        for _ in range(n_per_cluster):
            p = centroid + noise * rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            patterns.append(p)
    return np.array(patterns)


def _store_patterns(
    engine: CoupledEngine,
    patterns: np.ndarray,
    importance: float = 0.8,
) -> None:
    """Store all patterns into the engine with distinct text labels.

    importance=0.8 ensures patterns are tagged for dream processing
    (the auto-tagging threshold is 0.7 effective importance).
    """
    for i, p in enumerate(patterns):
        engine.store(text=f"pattern-{i}", embedding=p, importance=importance)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestDreamCoretrieval:
    """Tests that dream() logs discovered associations into _co_retrieval.

    These are discriminative: they pass IFF dream() writes associations
    into the co-retrieval graph. Today dream() does NOT do this, so all
    tests should fail with AssertionError.
    """

    def test_dream_associations_logged_in_co_retrieval(self):
        """After dream(), associations from REM-explore should appear
        as edges in self._co_retrieval.

        Steps:
        1. Create engine with multi-cluster data (3 clusters, 10 each)
        2. Verify _co_retrieval is empty before dream
        3. Run dream(seed=42)
        4. Assert that dream returned non-empty associations
        5. Assert that those pairs now exist in _co_retrieval
        """
        engine = CoupledEngine(dim=64)
        patterns = _make_clustered_patterns(
            n_per_cluster=10, n_clusters=3, dim=64, noise=0.3, seed=42,
        )
        _store_patterns(engine, patterns, importance=0.8)

        # Pre-condition: no co-retrieval edges yet (no queries have run)
        assert len(engine._co_retrieval) == 0, (
            "co_retrieval should be empty before any queries or dreams"
        )

        result = engine.dream(seed=42)
        associations = result["associations"]

        # Dream with 3 clusters at noise=0.3 reliably discovers cross-domain
        # associations. If it finds none, the test setup is wrong.
        assert len(associations) > 0, (
            f"Dream should discover cross-domain associations with 3 clusters "
            f"of 10 patterns each at noise=0.3. Got 0 associations. This may "
            f"indicate a dream_ops issue, not the feature under test."
        )

        # THE DISCRIMINATIVE ASSERTION: dream associations must be logged
        # into _co_retrieval. Today this fails because dream() returns
        # associations but never writes them to self._co_retrieval.
        assert len(engine._co_retrieval) > 0, (
            f"dream() returned {len(associations)} associations but "
            f"_co_retrieval is still empty. dream() must log associations "
            f"into _co_retrieval so query_coretrieval() can expand through "
            f"dream-discovered edges."
        )

        # Verify that actual association pairs exist in the graph
        found_in_graph = 0
        for i, j, _sim in associations:
            if i in engine._co_retrieval and j in engine._co_retrieval[i]:
                found_in_graph += 1
        assert found_in_graph > 0, (
            f"None of the {len(associations)} dream associations were found "
            f"as edges in _co_retrieval. Expected at least some to be logged."
        )

    def test_dream_edges_weighted_by_similarity(self):
        """Dream edges in _co_retrieval should carry the perturbation-response
        correlation strength (similarity score), not binary 1.0.

        This ensures that query_coretrieval can distinguish strong dream
        associations from weak ones when filtering by min_coretrieval_count.
        """
        engine = CoupledEngine(dim=64)
        patterns = _make_clustered_patterns(
            n_per_cluster=10, n_clusters=3, dim=64, noise=0.3, seed=99,
        )
        _store_patterns(engine, patterns, importance=0.8)

        result = engine.dream(seed=99)
        associations = result["associations"]
        assert len(associations) > 0, (
            "Need at least one association to test weight propagation"
        )

        # Check that each logged edge carries the similarity score
        for i, j, sim in associations:
            # Skip if indices are out of range (handled by test 3)
            if i >= engine.n_memories or j >= engine.n_memories:
                continue

            # The edge (i -> j) must exist in _co_retrieval
            assert i in engine._co_retrieval, (
                f"Association ({i}, {j}, {sim:.4f}) not logged: "
                f"index {i} not in _co_retrieval keys"
            )
            assert j in engine._co_retrieval[i], (
                f"Association ({i}, {j}, {sim:.4f}) not logged: "
                f"index {j} not in _co_retrieval[{i}]"
            )

            logged_weight = engine._co_retrieval[i][j]
            assert abs(logged_weight - sim) < 1e-6, (
                f"Edge ({i}, {j}) has weight {logged_weight:.6f} but "
                f"dream association similarity was {sim:.6f}. "
                f"Dream edges should carry the correlation strength, "
                f"not binary 1.0."
            )

            # Verify symmetry: (j -> i) should also exist with same weight
            assert j in engine._co_retrieval, (
                f"Symmetric edge missing: {j} not in _co_retrieval keys"
            )
            assert i in engine._co_retrieval[j], (
                f"Symmetric edge missing: {i} not in _co_retrieval[{j}]"
            )
            assert abs(engine._co_retrieval[j][i] - sim) < 1e-6, (
                f"Symmetric edge ({j}, {i}) has weight "
                f"{engine._co_retrieval[j][i]:.6f} but expected {sim:.6f}"
            )

    def test_dream_edges_survive_index_remapping(self):
        """After dream(), all _co_retrieval keys and neighbor indices must
        reference valid indices in the post-dream memory_store.

        Dream can prune/merge patterns, changing the total count. If
        associations used pre-dream indices, they would reference stale
        positions. DreamReport.associations already uses OUTPUT indices
        (post-prune/merge), so this should hold if logged correctly.
        """
        engine = CoupledEngine(dim=64)
        rng = np.random.default_rng(77)

        # Create base patterns with enough noise for associations
        base_patterns = _make_clustered_patterns(
            n_per_cluster=5, n_clusters=3, dim=64, noise=0.3, seed=77,
        )
        # Add near-duplicates (will be pruned by dream)
        duplicate_patterns = []
        for p in base_patterns[:3]:
            dup = p + 0.001 * rng.standard_normal(64)
            dup /= np.linalg.norm(dup)
            duplicate_patterns.append(dup)

        all_patterns = np.vstack([base_patterns, np.array(duplicate_patterns)])
        _store_patterns(engine, all_patterns, importance=0.8)

        n_before = engine.n_memories
        result = engine.dream(seed=77)
        n_after = engine.n_memories

        # Verify dream did some structural work (prune or merge)
        # This is a sanity check on the test setup, not the feature
        assert result["pruned"] > 0 or result["merged"] > 0 or n_after <= n_before, (
            f"Test setup issue: dream did not prune or merge. "
            f"n_before={n_before}, n_after={n_after}, "
            f"pruned={result['pruned']}, merged={result['merged']}"
        )

        # THE DISCRIMINATIVE ASSERTION: dream associations must populate
        # _co_retrieval. Without the feature, _co_retrieval stays empty.
        if len(result["associations"]) > 0:
            assert len(engine._co_retrieval) > 0, (
                f"dream() returned {len(result['associations'])} associations "
                f"but _co_retrieval is empty"
            )

        # All co-retrieval indices must be valid in post-dream memory_store
        for src_idx, neighbors in engine._co_retrieval.items():
            assert src_idx < engine.n_memories, (
                f"Stale index in _co_retrieval: key {src_idx} >= "
                f"n_memories {engine.n_memories}. Dream associations must "
                f"use post-dream OUTPUT indices, not pre-dream INPUT indices."
            )
            for neighbor_idx in neighbors:
                assert neighbor_idx < engine.n_memories, (
                    f"Stale index in _co_retrieval: _co_retrieval[{src_idx}] "
                    f"contains neighbor {neighbor_idx} >= n_memories "
                    f"{engine.n_memories}. Dream associations must use "
                    f"post-dream OUTPUT indices."
                )

    def test_query_coretrieval_uses_dream_edges(self):
        """End-to-end: query_coretrieval() should expand through dream edges
        to pull in patterns that would not appear without dream integration.

        Strategy: run query_coretrieval twice on the same engine state --
        once BEFORE dream (no dream edges in _co_retrieval) and once AFTER
        dream. The after-dream query should return additional patterns that
        were pulled in via dream association edges.

        This avoids depending on cross-cluster associations (dream may find
        intra-cluster associations too) and directly tests whether dream
        edges cause expansion in query_coretrieval.
        """
        dim = 64
        engine = CoupledEngine(dim=dim)

        patterns = _make_clustered_patterns(
            n_per_cluster=10, n_clusters=3, dim=dim, noise=0.3, seed=42,
        )
        _store_patterns(engine, patterns, importance=0.8)

        # Pick a query embedding (use one of the stored patterns)
        query_emb = engine.memory_store[0].embedding.copy()

        # BEFORE dream: query_coretrieval with narrow first_hop_k
        # No dream edges exist yet, so expansion is limited
        results_before = engine.query_coretrieval(
            embedding=query_emb,
            top_k=20,
            first_hop_k=3,
            min_coretrieval_count=0.0,
            coretrieval_bonus=0.5,
        )
        indices_before = {r["index"] for r in results_before}

        # Note: query_coretrieval logs its own co-retrieval edges from
        # the results. We need to clear them to isolate the dream effect.
        engine._co_retrieval.clear()
        engine._co_retrieval_query_count = 0

        # Run dream to discover associations
        result = engine.dream(seed=42)
        associations = result["associations"]
        assert len(associations) > 0, (
            "Need dream associations for end-to-end test"
        )

        # THE DISCRIMINATIVE ASSERTION: after dream(), the _co_retrieval
        # graph should contain dream edges. Without the feature, it stays
        # empty and the second query produces the same result as the first.
        assert len(engine._co_retrieval) > 0, (
            f"dream() returned {len(associations)} associations but "
            f"_co_retrieval is still empty after dream. Dream edges must "
            f"be logged for query_coretrieval to use them."
        )

        # AFTER dream: query_coretrieval should expand through dream edges
        results_after = engine.query_coretrieval(
            embedding=query_emb,
            top_k=20,
            first_hop_k=3,
            min_coretrieval_count=0.0,
            coretrieval_bonus=0.5,
        )
        indices_after = {r["index"] for r in results_after}

        # Dream edges should cause at least one additional pattern to be
        # pulled in via two-hop expansion that was not in the first query
        new_indices = indices_after - indices_before
        assert len(new_indices) > 0, (
            f"query_coretrieval returned the same {len(indices_before)} "
            f"patterns before and after dream. Dream edges should enable "
            f"two-hop expansion to pull in additional patterns. "
            f"Before: {sorted(indices_before)}, After: {sorted(indices_after)}"
        )
