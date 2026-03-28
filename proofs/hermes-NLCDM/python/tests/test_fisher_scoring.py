"""Tests for Fisher-Rao weighted retrieval scoring (arXiv 2603.14588v1, Eq. 14-16).

Tests the mathematical primitives (_compute_sigma_sq, _fisher_score) and
their integration into CoupledEngine.query_cooc_boost via the fisher_scoring flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure imports resolve
_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_NLCDM_PYTHON))
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))


from coupled_engine import CoupledEngine, MemoryEntry, _compute_sigma_sq, _fisher_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIM = 64  # small for fast tests
SIGMA_MAX = 1.0
SIGMA_MIN = 0.01
EPS = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_unit(dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a random L2-normalized vector."""
    if rng is None:
        rng = np.random.default_rng(42)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def _build_engine(dim: int = DIM) -> CoupledEngine:
    """Create a minimal CoupledEngine with cooc disabled."""
    return CoupledEngine(
        dim=dim,
        contradiction_aware=False,
        emotional_tagging=False,
        reconsolidation=False,
    )


# ---------------------------------------------------------------------------
# Test: _compute_sigma_sq
# ---------------------------------------------------------------------------

class TestComputeSigmaSq:

    def test_shape(self):
        """sigma_sq has same shape as input embedding."""
        emb = _random_unit(DIM)
        sigma_sq = _compute_sigma_sq(emb)
        assert sigma_sq.shape == (DIM,), f"Expected shape ({DIM},), got {sigma_sq.shape}"

    def test_range(self):
        """All sigma_sq values lie within [sigma_min + eps, sigma_max + eps]."""
        rng = np.random.default_rng(123)
        emb = _random_unit(DIM, rng)
        sigma_sq = _compute_sigma_sq(emb, sigma_max=SIGMA_MAX, sigma_min=SIGMA_MIN, eps=EPS)
        assert np.all(sigma_sq >= SIGMA_MIN), (
            f"Found sigma_sq < sigma_min: {sigma_sq.min()}"
        )
        assert np.all(sigma_sq <= SIGMA_MAX + EPS), (
            f"Found sigma_sq > sigma_max + eps: {sigma_sq.max()}"
        )

    def test_large_activation_low_variance(self):
        """Dimension with max |e_hat_k| gets sigma_min (+ eps)."""
        # Create embedding where dimension 0 has the largest absolute activation
        emb = np.zeros(DIM)
        emb[0] = 1.0  # This will be max after L2 normalization
        emb[1] = 0.5
        emb_normed = emb / np.linalg.norm(emb)

        sigma_sq = _compute_sigma_sq(emb_normed, sigma_max=SIGMA_MAX, sigma_min=SIGMA_MIN, eps=EPS)

        # Dimension 0 has the largest activation -> should get lowest variance
        max_dim = np.argmax(np.abs(emb_normed))
        assert max_dim == 0
        # The formula: sigma_max - (sigma_max - sigma_min) * |e_hat_k| / max_j(|e_hat_j|) + eps
        # When |e_hat_k| == max_j(|e_hat_j|), the ratio is 1.0, so:
        # sigma_sq = sigma_max - (sigma_max - sigma_min) * 1.0 + eps = sigma_min + eps
        expected = SIGMA_MIN + EPS
        assert abs(sigma_sq[0] - expected) < 1e-10, (
            f"Expected sigma_sq[0] = {expected}, got {sigma_sq[0]}"
        )

    def test_zero_activation_high_variance(self):
        """Dimension with 0 activation gets sigma_max (+ eps)."""
        # Create embedding with zero in dimension 2
        emb = np.zeros(DIM)
        emb[0] = 1.0
        emb[1] = 0.5
        # emb[2] remains 0
        emb_normed = emb / np.linalg.norm(emb)

        sigma_sq = _compute_sigma_sq(emb_normed, sigma_max=SIGMA_MAX, sigma_min=SIGMA_MIN, eps=EPS)

        # Dimension 2 has zero activation -> highest variance
        # sigma_sq = sigma_max - (sigma_max - sigma_min) * 0 / max + eps = sigma_max + eps
        expected = SIGMA_MAX + EPS
        assert abs(sigma_sq[2] - expected) < 1e-10, (
            f"Expected sigma_sq[2] = {expected}, got {sigma_sq[2]}"
        )

    def test_monotonicity(self):
        """Larger absolute activation -> lower sigma_sq (more confident)."""
        rng = np.random.default_rng(99)
        emb = _random_unit(DIM, rng)
        sigma_sq = _compute_sigma_sq(emb)

        abs_emb = np.abs(emb / np.linalg.norm(emb))
        # Sort: higher activation should correspond to lower sigma_sq
        sorted_act_indices = np.argsort(abs_emb)
        sorted_sigmas = sigma_sq[sorted_act_indices]
        # sigma_sq should be non-increasing as activation increases
        diffs = np.diff(sorted_sigmas)
        assert np.all(diffs <= 1e-10), (
            "sigma_sq should be non-increasing with increasing activation magnitude"
        )


# ---------------------------------------------------------------------------
# Test: _fisher_score
# ---------------------------------------------------------------------------

class TestFisherScore:

    def test_shape(self):
        """Fisher score returns (N,) for (N, d) embeddings."""
        rng = np.random.default_rng(42)
        query = _random_unit(DIM, rng)
        N = 5
        memories = np.array([_random_unit(DIM, np.random.default_rng(i)) for i in range(N)])
        sigma_sq = np.stack([_compute_sigma_sq(m) for m in memories])

        scores = _fisher_score(query, memories, sigma_sq, temperature=1.0)
        assert scores.shape == (N,), f"Expected shape ({N},), got {scores.shape}"

    def test_identical_high_score(self):
        """Identical query and memory produce a score close to 1.0."""
        emb = _random_unit(DIM)
        memories = emb.reshape(1, -1)
        sigma_sq = _compute_sigma_sq(emb).reshape(1, -1)

        scores = _fisher_score(emb, memories, sigma_sq, temperature=1.0)
        assert scores[0] > 0.99, (
            f"Identical vectors should score near 1.0, got {scores[0]}"
        )

    def test_orthogonal_low_score(self):
        """Orthogonal query and memory produce a low score."""
        # Create two orthogonal unit vectors
        emb_q = np.zeros(DIM)
        emb_q[0] = 1.0
        emb_m = np.zeros(DIM)
        emb_m[1] = 1.0

        memories = emb_m.reshape(1, -1)
        sigma_sq = _compute_sigma_sq(emb_m).reshape(1, -1)

        scores = _fisher_score(emb_q, memories, sigma_sq, temperature=1.0)
        assert scores[0] < 0.5, (
            f"Orthogonal vectors should score low, got {scores[0]}"
        )

    def test_temperature_effect(self):
        """Higher temperature makes scores closer together (less discriminative)."""
        rng = np.random.default_rng(42)
        query = _random_unit(DIM, rng)
        # One close, one far
        close = query + 0.05 * _random_unit(DIM, np.random.default_rng(1))
        close /= np.linalg.norm(close)
        far = _random_unit(DIM, np.random.default_rng(99))

        memories = np.stack([close, far])
        sigma_sq = np.stack([_compute_sigma_sq(m) for m in memories])

        scores_low_t = _fisher_score(query, memories, sigma_sq, temperature=0.1)
        scores_high_t = _fisher_score(query, memories, sigma_sq, temperature=10.0)

        gap_low_t = scores_low_t[0] - scores_low_t[1]
        gap_high_t = scores_high_t[0] - scores_high_t[1]

        assert gap_low_t > gap_high_t, (
            f"Low T gap ({gap_low_t:.4f}) should exceed high T gap ({gap_high_t:.4f})"
        )

    def test_scores_nonnegative(self):
        """All Fisher scores are in [0, 1] (exp of negative value)."""
        rng = np.random.default_rng(42)
        query = _random_unit(DIM, rng)
        N = 10
        memories = np.array([_random_unit(DIM, np.random.default_rng(i)) for i in range(N)])
        sigma_sq = np.stack([_compute_sigma_sq(m) for m in memories])

        scores = _fisher_score(query, memories, sigma_sq, temperature=1.0)
        assert np.all(scores >= 0.0), f"Found negative score: {scores.min()}"
        assert np.all(scores <= 1.0 + 1e-10), f"Found score > 1: {scores.max()}"


# ---------------------------------------------------------------------------
# Test: Fisher discriminates template siblings
# ---------------------------------------------------------------------------

class TestFisherDiscrimination:

    def test_discriminates_template_siblings(self):
        """Fisher score ranks the correct memory higher than cosine for template-similar embeddings.

        This simulates the real problem: "capital of France" and "capital of Germany"
        produce embeddings with cosine similarity ~0.99 (only a few dimensions differ).
        Fisher weighting amplifies the confident dimensions where they differ.
        """
        rng = np.random.default_rng(42)
        dim = 256

        # Create a base "template" embedding (shared structure, large enough
        # that small perturbations in a few dims preserve high cosine)
        template = rng.standard_normal(dim)
        template /= np.linalg.norm(template)

        # Create two "sibling" memories with SMALL entity-specific perturbations.
        # The perturbation magnitude is small relative to the full vector norm
        # so cosine similarity stays high (> 0.95), mirroring the real noise floor.
        perturbation_scale = 0.08
        sibling_a = template.copy()
        sibling_a[0:4] += perturbation_scale * np.array([1.0, 0.7, -0.5, 0.8])
        sibling_a /= np.linalg.norm(sibling_a)

        sibling_b = template.copy()
        sibling_b[0:4] += perturbation_scale * np.array([-1.0, -0.7, 0.5, -0.8])
        sibling_b /= np.linalg.norm(sibling_b)

        # Verify siblings are cosine-similar (simulating the noise floor)
        cos_ab = float(sibling_a @ sibling_b)
        assert cos_ab > 0.95, (
            f"Siblings should be cosine-similar (>0.95), got cos={cos_ab:.4f}"
        )

        # Query that matches sibling_a (with tiny noise)
        query = sibling_a + 0.01 * rng.standard_normal(dim)
        query /= np.linalg.norm(query)

        # Cosine scores
        cos_score_a = float(query @ sibling_a)
        cos_score_b = float(query @ sibling_b)
        cos_gap = cos_score_a - cos_score_b

        # Fisher scores -- use a temperature that enhances discrimination
        memories = np.stack([sibling_a, sibling_b])
        sigma_sq = np.stack([_compute_sigma_sq(m) for m in memories])
        fisher_scores = _fisher_score(query, memories, sigma_sq, temperature=1.0)
        fisher_gap = fisher_scores[0] - fisher_scores[1]

        # Fisher should provide LARGER separation than cosine
        assert fisher_gap > cos_gap, (
            f"Fisher gap ({fisher_gap:.6f}) should exceed cosine gap ({cos_gap:.6f})"
        )
        # And the correct answer should be ranked first by both
        assert fisher_scores[0] > fisher_scores[1], (
            "Fisher should rank sibling_a above sibling_b for query matching A"
        )


# ---------------------------------------------------------------------------
# Test: Integration with query_cooc_boost
# ---------------------------------------------------------------------------

class TestQueryCoocBoostFisher:

    def _build_engine_with_memories(self, dim: int = DIM) -> CoupledEngine:
        """Build a CoupledEngine with some stored memories + co-occurrence."""
        engine = _build_engine(dim)
        rng = np.random.default_rng(42)
        facts = [
            "The capital of France is Paris",
            "The capital of Germany is Berlin",
            "The capital of Italy is Rome",
            "The sport played by Abbiati is football",
            "The tallest mountain in the world is Everest",
        ]
        for i, text in enumerate(facts):
            emb = _random_unit(dim, np.random.default_rng(i + 100))
            engine.store(text, emb, recency=float(i))
        return engine

    def test_fisher_flag_returns_results(self):
        """query_cooc_boost with fisher_scoring=True returns results."""
        engine = self._build_engine_with_memories()
        rng = np.random.default_rng(999)
        q_emb = _random_unit(DIM, rng)

        results = engine.query_cooc_boost(
            embedding=q_emb, top_k=3,
            fisher_scoring=True, fisher_temperature=1.0,
        )
        assert len(results) == 3
        for r in results:
            assert "text" in r
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_fisher_vs_cosine_different_rankings(self):
        """Fisher and cosine may produce different rankings for template-similar memories."""
        dim = 128
        engine = _build_engine(dim)
        rng = np.random.default_rng(42)

        # Create template-similar memories (simulating entity confusion)
        template = rng.standard_normal(dim)
        template /= np.linalg.norm(template)

        siblings = []
        for i in range(5):
            s = template.copy()
            # Perturb a few dimensions differently for each
            s[i * 2: i * 2 + 2] += (i + 1) * 0.3
            s /= np.linalg.norm(s)
            siblings.append(s)

        texts = [
            "The capital of France is Paris",
            "The capital of Germany is Berlin",
            "The capital of Italy is Rome",
            "The capital of Spain is Madrid",
            "The capital of Portugal is Lisbon",
        ]
        for i, (text, emb) in enumerate(zip(texts, siblings)):
            engine.store(text, emb, recency=float(i))

        # Query targeting the first sibling
        query = siblings[0] + 0.01 * rng.standard_normal(dim)
        query /= np.linalg.norm(query)

        cosine_results = engine.query_cooc_boost(
            embedding=query, top_k=5, fisher_scoring=False,
        )
        fisher_results = engine.query_cooc_boost(
            embedding=query, top_k=5, fisher_scoring=True, fisher_temperature=1.0,
        )

        cosine_order = [r["text"] for r in cosine_results]
        fisher_order = [r["text"] for r in fisher_results]

        # Fisher should rank the matching memory (France/Paris) first
        assert fisher_results[0]["text"] == texts[0], (
            f"Fisher should rank target first, got: {fisher_results[0]['text']}"
        )

    def test_fisher_sigma_sq_cached_on_store(self):
        """After store(), MemoryEntry has sigma_sq populated."""
        engine = _build_engine()
        emb = _random_unit(DIM)
        idx = engine.store("test fact", emb)

        entry = engine.memory_store[idx]
        assert entry.sigma_sq is not None, "sigma_sq should be populated after store()"
        assert entry.sigma_sq.shape == (DIM,)

    def test_fisher_sigma_sq_cached_on_replace(self):
        """After replace_entry(), the new entry has sigma_sq recomputed."""
        engine = _build_engine()
        emb1 = _random_unit(DIM, np.random.default_rng(1))
        idx = engine.store("original fact", emb1)

        emb2 = _random_unit(DIM, np.random.default_rng(2))
        engine.replace_entry(idx, "replaced fact", emb2)

        entry = engine.memory_store[idx]
        assert entry.sigma_sq is not None, "sigma_sq should be recomputed after replace"
        # Verify it corresponds to emb2, not emb1
        expected = _compute_sigma_sq(emb2 / np.linalg.norm(emb2))
        np.testing.assert_allclose(entry.sigma_sq, expected, atol=1e-10)

    def test_fisher_temperature_zero_guard(self):
        """Temperature must be positive; raise on T <= 0."""
        rng = np.random.default_rng(42)
        query = _random_unit(DIM, rng)
        memories = _random_unit(DIM, np.random.default_rng(1)).reshape(1, -1)
        sigma_sq = _compute_sigma_sq(memories[0]).reshape(1, -1)

        with pytest.raises(ValueError, match="[Tt]emperature"):
            _fisher_score(query, memories, sigma_sq, temperature=0.0)
        with pytest.raises(ValueError, match="[Tt]emperature"):
            _fisher_score(query, memories, sigma_sq, temperature=-1.0)
