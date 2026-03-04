"""Tests for co_occurrence_bonus in query_twohop()."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from coupled_engine import CoupledEngine


def make_orthogonal_patterns(n: int, dim: int, seed: int = 42) -> list[np.ndarray]:
    """Create n orthogonal unit vectors in R^dim."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, dim))
    Q, _ = np.linalg.qr(M.T)
    return [Q[:, i] for i in range(n)]


class TestCoOccurrenceBonus:
    """Verify that co_occurrence_bonus parameter works correctly."""

    def test_bonus_parameter_exists(self):
        """query_twohop accepts co_occurrence_bonus parameter."""
        dim = 16
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)
        a = np.random.default_rng(0).standard_normal(dim)
        a /= np.linalg.norm(a)
        engine.store("a", a)
        engine.flush_session()
        # Should not raise
        engine.query_twohop(embedding=a, top_k=1, co_occurrence_bonus=0.5)

    def test_bonus_boosts_co_occurrence_facts(self):
        """Co-occurrence-expanded facts get boosted scores, not first-hop facts."""
        dim = 32
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)

        # Need enough memories so first_hop_k=1 selects only the closest match
        # and b enters only via co-occurrence expansion.
        patterns = make_orthogonal_patterns(6, dim, seed=42)

        # Session 1: store a, b together (co-occurring)
        engine.store("fact_a", patterns[0])
        engine.store("fact_b", patterns[1])
        engine.flush_session()

        # Fill in more standalone facts so the first-hop can be restricted
        for i in range(2, 6):
            engine.store(f"fact_{i}", patterns[i])
            engine.flush_session()

        # Query close to a
        query = patterns[0] + 0.05 * np.random.default_rng(42).standard_normal(dim)
        query /= np.linalg.norm(query)

        # Use first_hop_k=1 so only 'a' is in first hop; 'b' enters via expansion
        results_no_bonus = engine.query_twohop(
            embedding=query, top_k=6, first_hop_k=1, co_occurrence_bonus=0.0
        )

        results_with_bonus = engine.query_twohop(
            embedding=query, top_k=6, first_hop_k=1, co_occurrence_bonus=0.3
        )

        # Find b's score in both
        b_score_no_bonus = None
        b_score_with_bonus = None
        for r in results_no_bonus:
            if r["index"] == 1:  # b is index 1
                b_score_no_bonus = r["score"]
        for r in results_with_bonus:
            if r["index"] == 1:  # b is index 1
                b_score_with_bonus = r["score"]

        assert b_score_no_bonus is not None, "b should appear in results"
        assert b_score_with_bonus is not None, "b should appear in results"
        assert b_score_with_bonus > b_score_no_bonus, (
            f"Bonus should increase b's score: {b_score_with_bonus} > {b_score_no_bonus}"
        )

    def test_bonus_does_not_apply_to_first_hop(self):
        """First-hop facts (found by cosine) should NOT get the bonus."""
        dim = 32
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)

        patterns = make_orthogonal_patterns(4, dim, seed=42)

        # Session 1: store a, b together
        engine.store("fact_a", patterns[0])
        engine.store("fact_b", patterns[1])
        engine.flush_session()

        # More standalone facts so first_hop_k=1 is meaningful
        engine.store("fact_c", patterns[2])
        engine.store("fact_d", patterns[3])
        engine.flush_session()

        # Query very close to a
        query = patterns[0].copy()
        query /= np.linalg.norm(query)

        # first_hop_k=1: only a is in first hop (highest cosine)
        # b enters via co-occurrence, so b GETS the bonus but a does NOT
        results_no_bonus = engine.query_twohop(
            embedding=query, top_k=4, first_hop_k=1, co_occurrence_bonus=0.0
        )
        results_with_bonus = engine.query_twohop(
            embedding=query, top_k=4, first_hop_k=1, co_occurrence_bonus=0.5
        )

        a_score_no_bonus = None
        a_score_with_bonus = None
        for r in results_no_bonus:
            if r["index"] == 0:  # a is index 0
                a_score_no_bonus = r["score"]
        for r in results_with_bonus:
            if r["index"] == 0:
                a_score_with_bonus = r["score"]

        assert a_score_no_bonus is not None
        assert a_score_with_bonus is not None
        # First-hop score should be identical (no bonus applied)
        assert abs(a_score_no_bonus - a_score_with_bonus) < 1e-10, (
            f"First-hop fact should not get bonus: {a_score_no_bonus} vs {a_score_with_bonus}"
        )

    def test_default_bonus_is_0_3(self):
        """Default co_occurrence_bonus should be 0.3."""
        import inspect
        sig = inspect.signature(CoupledEngine.query_twohop)
        default = sig.parameters["co_occurrence_bonus"].default
        assert default == 0.3, f"Expected default 0.3, got {default}"
