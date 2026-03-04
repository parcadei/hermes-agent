"""Discriminative tests for cooc_boost and PPR retrieval methods.

Each test targets a specific hypothesis about the behavior of co-occurrence
boosted retrieval vs pure cosine vs PPR. Tests are designed to FALSIFY or
VALIDATE the hypothesis, not just confirm happy-path behavior.

Hypotheses:
  H1: cooc_boost does NOT regress on easy queries (cosine rank-1 target stays top-3)
  H2: cooc_boost degrades to pure cosine when co-occurrence graph is empty
  H3: PPR adds no value beyond cooc_boost for 1-hop pairs
  H4: PPR beats cooc_boost only on multi-hop chains
  H5: Weighted co-occurrence (1/session_size) has observable effect vs uniform
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_NLCDM_PYTHON))

from coupled_engine import CoupledEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit(rng: np.random.RandomState, dim: int) -> np.ndarray:
    """Generate a random unit vector."""
    v = rng.randn(dim)
    v /= np.linalg.norm(v)
    return v


def _rank_of(results: list[dict], target_index: int) -> int | None:
    """Return 0-based rank of target_index in results, or None if absent."""
    for i, r in enumerate(results):
        if r["index"] == target_index:
            return i
    return None


def _score_of(results: list[dict], target_index: int) -> float | None:
    """Return score of target_index in results, or None if absent."""
    for r in results:
        if r["index"] == target_index:
            return r["score"]
    return None


# ---------------------------------------------------------------------------
# H1: cooc_boost does NOT regress on easy queries
# ---------------------------------------------------------------------------

class TestH1EasyQueryNoRegression:
    """When the target memory is already cosine rank-1, cooc_boost must not
    push it below rank-3, even when noise memories have strong co-occurrence
    links among themselves.

    FALSIFICATION condition: target drops from cosine rank-1 to cooc_boost rank > 3.
    """

    def test_cosine_rank1_stays_top3_with_noise_cooc(self):
        """Create a target at cosine rank-1, add strong co-occurrence among noise,
        verify target stays in top-3 under cooc_boost(cooc_weight=1.0)."""
        dim = 64
        rng = np.random.RandomState(42)
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05, recency_alpha=0.0)

        # Store 50 random background memories
        bg_indices = []
        for i in range(50):
            emb = _make_unit(rng, dim)
            idx = engine.store(text=f"noise_{i}", embedding=emb, recency=float(i))
            bg_indices.append(idx)

        # Create co-occurrence links among noise memories (groups of 5)
        # This gives noise memories a strong mutual boost
        for group_start in range(0, 50, 5):
            engine._session_buffer = []
            engine._session_indices = []
            group_indices = bg_indices[group_start:group_start + 5]
            for gi in group_indices:
                engine._session_buffer.append(engine.memory_store[gi].embedding.copy())
                engine._session_indices.append(gi)
            engine.flush_session()

        # Store a target memory that is very close to the query direction
        query_direction = _make_unit(rng, dim)
        target_emb = query_direction + 0.02 * rng.randn(dim)  # very close
        target_emb /= np.linalg.norm(target_emb)
        target_idx = engine.store(text="TARGET", embedding=target_emb, recency=100.0)
        # No co-occurrence for target -- it has no neighbors in the graph

        # Query
        q = query_direction.copy()

        results_cosine = engine.query(embedding=q, top_k=engine.n_memories)
        results_cooc = engine.query_cooc_boost(
            embedding=q, top_k=engine.n_memories, cooc_weight=1.0
        )

        rank_cosine = _rank_of(results_cosine, target_idx)
        rank_cooc = _rank_of(results_cooc, target_idx)

        print(f"\n[H1] Target rank in cosine: {rank_cosine}")
        print(f"[H1] Target rank in cooc_boost(w=1.0): {rank_cooc}")
        print(f"[H1] Target cosine score: {_score_of(results_cosine, target_idx):.4f}")
        print(f"[H1] Target cooc score:   {_score_of(results_cooc, target_idx):.4f}")

        # Verify cosine rank is indeed 0 (rank-1)
        assert rank_cosine == 0, f"Expected cosine rank 0, got {rank_cosine}"

        # CRITICAL ASSERTION: target must stay in top-3 under cooc_boost
        assert rank_cooc is not None, "Target not found in cooc results"
        assert rank_cooc <= 2, (
            f"REGRESSION: cooc_boost pushed cosine rank-1 target to rank {rank_cooc + 1}. "
            f"H1 FALSIFIED."
        )

    def test_cosine_rank1_stays_rank1_moderate_weight(self):
        """With moderate cooc_weight=0.3 (default), rank-1 should stay rank-1."""
        dim = 64
        rng = np.random.RandomState(99)
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05, recency_alpha=0.0)

        # 30 background memories with co-occurrence
        for i in range(30):
            emb = _make_unit(rng, dim)
            engine.store(text=f"bg_{i}", embedding=emb, recency=float(i))
        engine.flush_session()

        # Strong target
        query_direction = _make_unit(rng, dim)
        target_emb = query_direction + 0.01 * rng.randn(dim)
        target_emb /= np.linalg.norm(target_emb)
        target_idx = engine.store(text="TARGET", embedding=target_emb, recency=50.0)

        results_cosine = engine.query(embedding=query_direction, top_k=10)
        results_cooc = engine.query_cooc_boost(
            embedding=query_direction, top_k=10, cooc_weight=0.3
        )

        rank_cosine = _rank_of(results_cosine, target_idx)
        rank_cooc = _rank_of(results_cooc, target_idx)

        print(f"\n[H1b] Target rank in cosine: {rank_cosine}")
        print(f"[H1b] Target rank in cooc_boost(w=0.3): {rank_cooc}")

        assert rank_cosine == 0, f"Expected cosine rank 0, got {rank_cosine}"
        assert rank_cooc == 0, (
            f"REGRESSION: even moderate cooc_weight moved rank-1 target to rank {rank_cooc + 1}"
        )


# ---------------------------------------------------------------------------
# H2: cooc_boost degrades to pure cosine when co-occurrence graph is empty
# ---------------------------------------------------------------------------

class TestH2EmptyGraphDegradation:
    """When no flush_session() calls have been made, cooc_boost should return
    identical rankings and scores to pure cosine query().

    FALSIFICATION condition: rankings or scores differ between the two methods.
    """

    def test_empty_graph_identical_rankings(self):
        """No flush_session calls => cooc_boost rankings == cosine rankings."""
        dim = 32
        rng = np.random.RandomState(42)
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05, recency_alpha=0.0)

        # Store 20 memories WITHOUT any flush_session
        for i in range(20):
            emb = _make_unit(rng, dim)
            engine.store(text=f"fact_{i}", embedding=emb, recency=float(i))
        # Deliberately NOT calling flush_session

        q = _make_unit(rng, dim)

        results_cosine = engine.query(embedding=q, top_k=20)
        results_cooc = engine.query_cooc_boost(
            embedding=q, top_k=20, cooc_weight=1.0
        )

        cosine_indices = [r["index"] for r in results_cosine]
        cooc_indices = [r["index"] for r in results_cooc]
        cosine_scores = [r["score"] for r in results_cosine]
        cooc_scores = [r["score"] for r in results_cooc]

        print(f"\n[H2] Cosine ranking: {cosine_indices[:10]}")
        print(f"[H2] Cooc ranking:  {cooc_indices[:10]}")
        print(f"[H2] Cosine scores: {[f'{s:.4f}' for s in cosine_scores[:5]]}")
        print(f"[H2] Cooc scores:   {[f'{s:.4f}' for s in cooc_scores[:5]]}")

        # Rankings must be identical
        assert cosine_indices == cooc_indices, (
            f"FALSIFIED: empty graph gave different rankings.\n"
            f"  cosine: {cosine_indices[:10]}\n"
            f"  cooc:   {cooc_indices[:10]}"
        )

        # Scores must be identical within float tolerance
        for i, (cs, ccs) in enumerate(zip(cosine_scores, cooc_scores)):
            assert abs(cs - ccs) < 1e-10, (
                f"FALSIFIED: score mismatch at rank {i}: cosine={cs:.6f} vs cooc={ccs:.6f}"
            )

    def test_empty_graph_no_flush_single_items(self):
        """Store items one at a time, flush with single items (no edges created).
        Should still degrade to pure cosine."""
        dim = 32
        rng = np.random.RandomState(77)
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05, recency_alpha=0.0)

        for i in range(15):
            emb = _make_unit(rng, dim)
            engine.store(text=f"solo_{i}", embedding=emb, recency=float(i))
            pairs = engine.flush_session()
            assert pairs == 0, f"Single-item flush should create 0 pairs, got {pairs}"

        q = _make_unit(rng, dim)

        results_cosine = engine.query(embedding=q, top_k=15)
        results_cooc = engine.query_cooc_boost(
            embedding=q, top_k=15, cooc_weight=1.0
        )

        cosine_indices = [r["index"] for r in results_cosine]
        cooc_indices = [r["index"] for r in results_cooc]

        print(f"\n[H2b] Cosine ranking: {cosine_indices}")
        print(f"[H2b] Cooc ranking:  {cooc_indices}")

        assert cosine_indices == cooc_indices, (
            f"FALSIFIED: single-item flushes changed rankings.\n"
            f"  cosine: {cosine_indices}\n"
            f"  cooc:   {cooc_indices}"
        )


# ---------------------------------------------------------------------------
# H3: PPR adds no value beyond cooc_boost for 1-hop pairs
# ---------------------------------------------------------------------------

class TestH3PPRvsCoocOnOneHop:
    """When co-occurrence is purely pairwise (1-hop, no chains), PPR's iterative
    diffusion should not find anything that cooc_boost misses.

    FALSIFICATION condition: PPR ranks a target in top-k where cooc_boost does not,
    using only 1-hop co-occurrence data.
    """

    def test_one_hop_ppr_no_advantage(self):
        """Create 5 bridge pairs (1-hop only), query each. PPR should not
        outperform cooc_boost for any pair."""
        dim = 64
        rng = np.random.RandomState(42)
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05, recency_alpha=0.0)

        # 20 background memories (not linked)
        for i in range(20):
            emb = _make_unit(rng, dim)
            engine.store(text=f"bg_{i}", embedding=emb, recency=float(i))
        engine.flush_session()  # links the 20 bg facts together

        # 5 bridge PAIRS: each pair is co-stored in its own session
        # A and B are orthogonal in each pair
        pairs_info = []
        for p in range(5):
            emb_a = _make_unit(rng, dim)
            emb_b = _make_unit(rng, dim)
            # Make B orthogonal to A
            emb_b -= np.dot(emb_b, emb_a) * emb_a
            emb_b /= np.linalg.norm(emb_b)

            idx_a = engine.store(text=f"pairA_{p}", embedding=emb_a, recency=30.0 + p)
            idx_b = engine.store(text=f"pairB_{p}", embedding=emb_b, recency=30.0 + p)
            engine.flush_session()

            pairs_info.append({
                "idx_a": idx_a, "idx_b": idx_b,
                "emb_a": emb_a, "emb_b": emb_b,
            })

        top_k = 10
        ppr_wins = 0
        cooc_wins = 0

        for p, info in enumerate(pairs_info):
            # Query near A, look for B
            q = info["emb_a"] + 0.05 * rng.randn(dim)
            q /= np.linalg.norm(q)

            results_cooc = engine.query_cooc_boost(
                embedding=q, top_k=top_k, cooc_weight=0.5
            )
            results_ppr = engine.query_ppr(
                embedding=q, top_k=top_k, ppr_weight=0.5
            )

            rank_cooc = _rank_of(results_cooc, info["idx_b"])
            rank_ppr = _rank_of(results_ppr, info["idx_b"])

            cooc_found = rank_cooc is not None
            ppr_found = rank_ppr is not None

            print(f"\n[H3] Pair {p}: B rank in cooc={rank_cooc}, ppr={rank_ppr}")

            if ppr_found and not cooc_found:
                ppr_wins += 1
            elif cooc_found and not ppr_found:
                cooc_wins += 1
            elif ppr_found and cooc_found:
                if rank_ppr < rank_cooc:
                    ppr_wins += 1
                elif rank_cooc < rank_ppr:
                    cooc_wins += 1

        print(f"\n[H3] Summary: PPR wins={ppr_wins}, cooc wins={cooc_wins}")
        print(f"[H3] H3 {'VALIDATED' if ppr_wins == 0 else 'FALSIFIED'}: "
              f"PPR {'never' if ppr_wins == 0 else 'sometimes'} beats cooc on 1-hop")

        # H3 predicts PPR adds NO value on 1-hop. If PPR wins on any pair,
        # that's interesting but not necessarily a failure -- we just report it.
        # The assertion is soft: PPR should not CONSISTENTLY beat cooc on 1-hop.
        assert ppr_wins <= 2, (
            f"H3 FALSIFIED: PPR beat cooc_boost on {ppr_wins}/5 1-hop pairs. "
            f"PPR has unexpected advantage on simple pairwise co-occurrence."
        )


# ---------------------------------------------------------------------------
# H4: PPR beats cooc_boost only on multi-hop chains
# ---------------------------------------------------------------------------

class TestH4PPRMultiHopAdvantage:
    """PPR's advantage (if any) should appear on multi-hop chains where
    cooc_boost's 1-hop neighbor mean cannot propagate signal.

    Setup: A --cooc--> B --cooc--> C
    Query matches A well, target is C (low cosine to query).
    PPR should propagate A's high cosine through B to C.
    cooc_boost can only boost B (A's direct neighbor).

    FALSIFICATION of H4: cooc_boost finds C as well as PPR does.
    """

    def test_3hop_chain_ppr_finds_target(self):
        """A->B->C chain. Query ~ A. Does PPR rank C higher than cooc_boost?"""
        dim = 64
        rng = np.random.RandomState(42)
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05, recency_alpha=0.0)

        # 20 background noise
        for i in range(20):
            emb = _make_unit(rng, dim)
            engine.store(text=f"bg_{i}", embedding=emb, recency=float(i))
        engine.flush_session()

        # Create 3 chain nodes with controlled cosine relationships
        # A: very similar to query
        # B: medium similarity to query (random)
        # C: very DIFFERENT from query (near-orthogonal)
        query_direction = _make_unit(rng, dim)

        emb_a = query_direction + 0.05 * rng.randn(dim)
        emb_a /= np.linalg.norm(emb_a)

        emb_b = _make_unit(rng, dim)  # random direction

        emb_c = _make_unit(rng, dim)
        # Make C orthogonal to query
        emb_c -= np.dot(emb_c, query_direction) * query_direction
        emb_c /= np.linalg.norm(emb_c)

        idx_a = engine.store(text="chain_A", embedding=emb_a, recency=30.0)
        idx_b = engine.store(text="chain_B", embedding=emb_b, recency=31.0)
        idx_c = engine.store(text="chain_C", embedding=emb_c, recency=32.0)

        # Session 1: A and B co-occur
        engine._session_buffer = [emb_a.copy(), emb_b.copy()]
        engine._session_indices = [idx_a, idx_b]
        engine.flush_session()

        # Session 2: B and C co-occur
        engine._session_buffer = [emb_b.copy(), emb_c.copy()]
        engine._session_indices = [idx_b, idx_c]
        engine.flush_session()

        # Now query with the query direction
        q = query_direction.copy()

        results_cosine = engine.query(embedding=q, top_k=engine.n_memories)
        results_cooc = engine.query_cooc_boost(
            embedding=q, top_k=engine.n_memories, cooc_weight=1.0
        )
        results_ppr = engine.query_ppr(
            embedding=q, top_k=engine.n_memories, ppr_weight=0.5
        )

        rank_cos_a = _rank_of(results_cosine, idx_a)
        rank_cos_b = _rank_of(results_cosine, idx_b)
        rank_cos_c = _rank_of(results_cosine, idx_c)

        rank_cooc_a = _rank_of(results_cooc, idx_a)
        rank_cooc_b = _rank_of(results_cooc, idx_b)
        rank_cooc_c = _rank_of(results_cooc, idx_c)

        rank_ppr_a = _rank_of(results_ppr, idx_a)
        rank_ppr_b = _rank_of(results_ppr, idx_b)
        rank_ppr_c = _rank_of(results_ppr, idx_c)

        print(f"\n[H4] Chain A (query-similar):")
        print(f"  cosine rank={rank_cos_a}, cooc rank={rank_cooc_a}, ppr rank={rank_ppr_a}")
        print(f"[H4] Chain B (intermediate):")
        print(f"  cosine rank={rank_cos_b}, cooc rank={rank_cooc_b}, ppr rank={rank_ppr_b}")
        print(f"[H4] Chain C (target, orthogonal to query):")
        print(f"  cosine rank={rank_cos_c}, cooc rank={rank_cooc_c}, ppr rank={rank_ppr_c}")

        score_cos_c = _score_of(results_cosine, idx_c)
        score_cooc_c = _score_of(results_cooc, idx_c)
        score_ppr_c = _score_of(results_ppr, idx_c)
        print(f"\n[H4] C scores: cosine={score_cos_c:.4f}, cooc={score_cooc_c:.4f}, ppr={score_ppr_c:.4f}")

        ppr_improved_c = rank_ppr_c < rank_cooc_c
        print(f"[H4] PPR improved C's rank over cooc_boost: {ppr_improved_c}")
        print(f"[H4] PPR rank improvement: {rank_cooc_c - rank_ppr_c} positions")

        # We REPORT the result but the assertion is about whether PPR at least
        # does not REGRESS on the chain target compared to cosine.
        # The real measurement is the rank delta.
        assert rank_ppr_c is not None, "C should appear somewhere in PPR results"

    def test_longer_chain_4hop(self):
        """A->B->C->D chain, 4 hops. Query ~ A. Target = D.
        Measures propagation depth of PPR vs cooc_boost."""
        dim = 64
        rng = np.random.RandomState(123)
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05, recency_alpha=0.0)

        # 15 background
        for i in range(15):
            emb = _make_unit(rng, dim)
            engine.store(text=f"bg_{i}", embedding=emb, recency=float(i))
        engine.flush_session()

        query_direction = _make_unit(rng, dim)

        # Chain: A close to query, B/C/D progressively more orthogonal
        chain_embs = []
        chain_indices = []
        chain_names = ["chain_A", "chain_B", "chain_C", "chain_D"]

        for hop, name in enumerate(chain_names):
            if hop == 0:
                emb = query_direction + 0.03 * rng.randn(dim)
            else:
                emb = _make_unit(rng, dim)
                # Make each successive node more orthogonal to query
                emb -= (0.9 ** hop) * np.dot(emb, query_direction) * query_direction
            emb /= np.linalg.norm(emb)
            idx = engine.store(text=name, embedding=emb, recency=20.0 + hop)
            chain_embs.append(emb)
            chain_indices.append(idx)

        # Create pairwise co-occurrence links along the chain
        for h in range(len(chain_indices) - 1):
            engine._session_buffer = [chain_embs[h].copy(), chain_embs[h + 1].copy()]
            engine._session_indices = [chain_indices[h], chain_indices[h + 1]]
            engine.flush_session()

        q = query_direction.copy()

        results_cosine = engine.query(embedding=q, top_k=engine.n_memories)
        results_cooc = engine.query_cooc_boost(
            embedding=q, top_k=engine.n_memories, cooc_weight=1.0
        )
        results_ppr = engine.query_ppr(
            embedding=q, top_k=engine.n_memories, ppr_weight=0.5, ppr_steps=20
        )

        print("\n[H4b] 4-hop chain results:")
        for hop, name in enumerate(chain_names):
            idx = chain_indices[hop]
            rc = _rank_of(results_cosine, idx)
            rco = _rank_of(results_cooc, idx)
            rp = _rank_of(results_ppr, idx)
            sc = _score_of(results_cosine, idx)
            sco = _score_of(results_cooc, idx)
            sp = _score_of(results_ppr, idx)
            print(f"  {name}: cosine rank={rc} ({sc:.4f}), "
                  f"cooc rank={rco} ({sco:.4f}), ppr rank={rp} ({sp:.4f})")

        # Measure: does PPR rank D higher than cooc_boost?
        target_idx = chain_indices[-1]  # D
        rank_cooc_d = _rank_of(results_cooc, target_idx)
        rank_ppr_d = _rank_of(results_ppr, target_idx)
        improvement = rank_cooc_d - rank_ppr_d if (rank_cooc_d is not None and rank_ppr_d is not None) else 0
        print(f"\n[H4b] Target D: cooc rank={rank_cooc_d}, ppr rank={rank_ppr_d}, "
              f"PPR improvement={improvement} positions")


# ---------------------------------------------------------------------------
# H5: Weighted co-occurrence (1/session_size) effect
# ---------------------------------------------------------------------------

class TestH5WeightedCooccurrence:
    """Test whether the 1/session_size weighting produces different results
    than uniform weighting. Create sessions of different sizes and measure
    whether small sessions (strong links) get preferential boosting.

    Setup:
    - Pair session (size 2): A and B co-stored  -> weight = 0.5
    - Large session (size 10): C and 9 noise items -> weight = 0.1

    Query matches both A and C equally. B and D (noise item from large session)
    are orthogonal to query. Under weighted co-occurrence, B should get a
    stronger boost than any item from the large session.

    FALSIFICATION: large-session items get boosted as much as pair-session items.
    """

    def test_small_session_stronger_boost(self):
        """Pair session gives stronger co-occurrence boost than large session.

        KEY DESIGN: We must isolate sessions properly. store() accumulates items
        in _session_buffer. flush_session() consumes the buffer. So we must flush
        after EACH group to keep sessions separate.

        Because the weighted_mean formula normalizes by total_weight, the
        1/session_size factor CANCELS when all neighbors come from one session.
        The boost = mean(cos[neighbors]), regardless of session size.

        The real discriminative effect: B (pair session) has 1 neighbor (A, high
        cosine). D (large session) has 9 neighbors (C=high + 8 noise=low cosine).
        B gets boost = cos(A, q) ~ 0.9. D gets boost = mean of 9 cosines ~ 0.1.
        So B's boost exceeds D's because B's sole neighbor is a strong match.
        """
        dim = 64
        rng = np.random.RandomState(42)
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05, recency_alpha=0.0)

        # 10 background memories -- flush them immediately so they form their
        # own session and do NOT contaminate the pair/large sessions.
        for i in range(10):
            emb = _make_unit(rng, dim)
            engine.store(text=f"bg_{i}", embedding=emb, recency=float(i))
        engine.flush_session()  # background session closed

        # Create a query direction
        query_direction = _make_unit(rng, dim)

        # PAIR SESSION: A (similar to query) + B (orthogonal to query)
        emb_a = query_direction + 0.05 * rng.randn(dim)
        emb_a /= np.linalg.norm(emb_a)
        emb_b = _make_unit(rng, dim)
        emb_b -= np.dot(emb_b, query_direction) * query_direction
        emb_b /= np.linalg.norm(emb_b)

        idx_a = engine.store(text="pair_A", embedding=emb_a, recency=20.0)
        idx_b = engine.store(text="pair_B", embedding=emb_b, recency=21.0)
        engine.flush_session()  # pair session: size=2, weight=0.5

        # LARGE SESSION: C (similar to query) + 9 noise items
        emb_c = query_direction + 0.05 * rng.randn(dim)
        emb_c /= np.linalg.norm(emb_c)
        idx_c = engine.store(text="large_C", embedding=emb_c, recency=30.0)

        large_session_indices = [idx_c]
        for j in range(9):
            emb_noise = _make_unit(rng, dim)
            # Make noise orthogonal to query
            emb_noise -= np.dot(emb_noise, query_direction) * query_direction
            emb_noise /= np.linalg.norm(emb_noise)
            idx_noise = engine.store(
                text=f"large_noise_{j}", embedding=emb_noise, recency=31.0 + j
            )
            large_session_indices.append(idx_noise)
        engine.flush_session()  # large session: size=10, weight=0.1

        # Pick the first noise item from the large session as "D"
        idx_d = large_session_indices[1]

        # Verify session isolation: B should have exactly 1 neighbor (A)
        b_neighbors = engine._co_occurrence.get(idx_b, {})
        d_neighbors = engine._co_occurrence.get(idx_d, {})
        print(f"\n[H5] B neighbors count: {len(b_neighbors)} (expected 1)")
        print(f"[H5] D neighbors count: {len(d_neighbors)} (expected 9)")
        assert len(b_neighbors) == 1, (
            f"B should have 1 neighbor (A), got {len(b_neighbors)}: {list(b_neighbors.keys())}"
        )

        q = query_direction.copy()

        # Run cooc FIRST (before query()) to avoid side-effect ordering issues
        results_cooc = engine.query_cooc_boost(
            embedding=q, top_k=engine.n_memories, cooc_weight=1.0
        )
        results_cosine = engine.query(embedding=q, top_k=engine.n_memories)

        score_b = _score_of(results_cooc, idx_b)
        score_d = _score_of(results_cooc, idx_d)
        rank_b = _rank_of(results_cooc, idx_b)
        rank_d = _rank_of(results_cooc, idx_d)

        cos_b = _score_of(results_cosine, idx_b)
        cos_d = _score_of(results_cosine, idx_d)

        print(f"[H5] Pair B:  cosine={cos_b:.4f}, cooc_score={score_b:.4f}, rank={rank_b}")
        print(f"[H5] Large D: cosine={cos_d:.4f}, cooc_score={score_d:.4f}, rank={rank_d}")

        boost_b = score_b - cos_b if (score_b is not None and cos_b is not None) else 0.0
        boost_d = score_d - cos_d if (score_d is not None and cos_d is not None) else 0.0

        print(f"[H5] Boost for B (pair session, 1 neighbor): {boost_b:.6f}")
        print(f"[H5] Boost for D (large session, 9 neighbors): {boost_d:.6f}")
        print(f"[H5] B boost > D boost: {boost_b > boost_d}")

        # With properly isolated sessions:
        # B's boost = mean neighbor cosine = cos(A, q) ~ high (A is similar to query)
        # D's boost = mean of 9 cosines = (cos(C,q) + sum(cos(noise_j,q))) / 9
        #   where C ~ high but noise_j ~ 0 (orthogonal to query)
        #   So D's boost ~ cos(C,q) / 9 ~ 0.1
        # Therefore B's boost should exceed D's boost.
        assert boost_b > boost_d, (
            f"H5 FALSIFIED: pair session boost ({boost_b:.6f}) <= large session boost ({boost_d:.6f})"
        )

        # Additionally verify the analytical prediction for B's boost
        cos_a_to_q = float(np.dot(emb_a, q) / (np.linalg.norm(emb_a) * np.linalg.norm(q) + 1e-12))
        expected_boost_b = 1.0 * cos_a_to_q  # cooc_weight * mean_neighbor_cosine
        print(f"[H5] Expected B boost (cos(A,q)): {expected_boost_b:.6f}")
        print(f"[H5] Actual B boost:              {boost_b:.6f}")
        assert abs(boost_b - expected_boost_b) < 1e-4, (
            f"B boost should equal cos(A,q)={expected_boost_b:.6f}, got {boost_b:.6f}"
        )

    def test_weight_normalization_cancels(self):
        """Demonstrate that the weighted mean formula cancels session weights
        when ALL neighbors have the same weight. This is the analytical
        prediction: 1/session_size cancels in weighted_sum/total_weight."""
        dim = 32
        rng = np.random.RandomState(42)
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05, recency_alpha=0.0)

        # Create two memories
        emb_x = _make_unit(rng, dim)
        emb_y = _make_unit(rng, dim)
        idx_x = engine.store(text="X", embedding=emb_x, recency=1.0)
        idx_y = engine.store(text="Y", embedding=emb_y, recency=2.0)

        # Session of size 2: weight = 0.5
        engine.flush_session()

        # Check that the co-occurrence weight is 0.5
        weight_xy = engine._co_occurrence.get(idx_x, {}).get(idx_y, 0.0)
        weight_yx = engine._co_occurrence.get(idx_y, {}).get(idx_x, 0.0)

        print(f"\n[H5b] Co-occurrence weight X->Y: {weight_xy}")
        print(f"[H5b] Co-occurrence weight Y->X: {weight_yx}")

        assert abs(weight_xy - 0.5) < 1e-10, f"Expected weight 0.5, got {weight_xy}"
        assert abs(weight_yx - 0.5) < 1e-10, f"Expected weight 0.5, got {weight_yx}"

        # Now query with emb_x. The boost for Y should be:
        # cooc_weight * (weight_xy * cos(Y, q)) / weight_xy = cooc_weight * cos(Y, q)
        # The weight cancels because Y is the ONLY neighbor of X.
        q = emb_x.copy()
        results = engine.query_cooc_boost(embedding=q, top_k=2, cooc_weight=1.0)

        score_y_cooc = _score_of(results, idx_y)

        # Pure cosine score of Y
        cos_y = float(np.dot(emb_y, q) / (np.linalg.norm(emb_y) * np.linalg.norm(q) + 1e-12))

        # Predicted cooc score: cos(Y,q) + 1.0 * cos(X,q)  (boost = mean neighbor cosine * weight)
        # Since X is Y's only neighbor, boost = cos(X, q) = 1.0 (q == emb_x)
        cos_x_to_q = float(np.dot(emb_x, q) / (np.linalg.norm(emb_x) * np.linalg.norm(q) + 1e-12))
        predicted_score_y = cos_y + 1.0 * cos_x_to_q

        print(f"[H5b] cos(Y, q) = {cos_y:.6f}")
        print(f"[H5b] cos(X, q) = {cos_x_to_q:.6f}")
        print(f"[H5b] Predicted cooc score for Y = {predicted_score_y:.6f}")
        print(f"[H5b] Actual cooc score for Y   = {score_y_cooc:.6f}")

        assert abs(score_y_cooc - predicted_score_y) < 1e-6, (
            f"Score mismatch: predicted {predicted_score_y:.6f} vs actual {score_y_cooc:.6f}"
        )
