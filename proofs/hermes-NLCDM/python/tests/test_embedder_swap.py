"""Tests for the embedder swap: Qwen3-0.6B (1024-dim) -> all-mpnet-base-v2 (768-dim).

Validates:
  1. Output dimensions (768)
  2. L2 normalization
  3. Semantic similarity ordering
  4. Entity layer bounds for the new model
  5. Default model name
  6. Dimension defaults in agent constructors

Tests 1-5 load the real model and are marked @pytest.mark.slow.
Test 6+ are fast structural checks.
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


# ---------------------------------------------------------------------------
# Slow tests (load real model)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestEmbedDimensions:
    """Verify embedding output shape and dtype after swap to all-mpnet-base-v2."""

    @pytest.fixture(scope="class")
    def scorer(self):
        from mabench.scoring import EmbeddingRelevanceScorer
        # Use default model (should be all-mpnet-base-v2 after swap)
        return EmbeddingRelevanceScorer()

    def test_embed_returns_correct_dim(self, scorer):
        """embed('hello') returns ndarray of shape (768,) with float64."""
        emb = scorer.embed("hello")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (768,), f"Expected (768,), got {emb.shape}"
        assert emb.dtype == np.float64, f"Expected float64, got {emb.dtype}"

    def test_embed_batch_returns_correct_dims(self, scorer):
        """embed_batch(['a','b','c']) returns list of 3 ndarrays each (768,)."""
        texts = ["alpha", "beta", "gamma"]
        results = scorer.embed_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 3
        for i, emb in enumerate(results):
            assert isinstance(emb, np.ndarray), f"Item {i} is not ndarray"
            assert emb.shape == (768,), f"Item {i}: expected (768,), got {emb.shape}"
            assert emb.dtype == np.float64, f"Item {i}: expected float64, got {emb.dtype}"

    def test_embeddings_are_l2_normalized(self, scorer):
        """All embeddings should have L2 norm approximately 1.0."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Quantum computing leverages superposition for parallel computation",
            "Paris is the capital of France",
        ]
        for text in texts:
            emb = scorer.embed(text)
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-5, (
                f"Norm of embed('{text[:30]}...') = {norm}, expected ~1.0"
            )

    def test_similar_texts_higher_cosine(self, scorer):
        """Semantically similar texts should have higher cosine similarity."""
        text_a = "The cat sat on the mat"
        text_b = "The cat was on the mat"
        text_c = "Quantum physics is hard"

        emb_a = scorer.embed(text_a)
        emb_b = scorer.embed(text_b)
        emb_c = scorer.embed(text_c)

        sim_ab = float(np.dot(emb_a, emb_b))
        sim_ac = float(np.dot(emb_a, emb_c))

        assert sim_ab > sim_ac, (
            f"sim(cat_mat, cat_mat_variant)={sim_ab:.4f} should be > "
            f"sim(cat_mat, quantum)={sim_ac:.4f}"
        )
        # The gap should be substantial for a purpose-built sentence embedder
        assert sim_ab - sim_ac > 0.1, (
            f"Gap {sim_ab - sim_ac:.4f} too small for a purpose-built embedder"
        )


@pytest.mark.slow
class TestEntityLayerBounds:
    """Verify _ENTITY_LAYER is within bounds for all-mpnet-base-v2 (12 layers)."""

    @pytest.fixture(scope="class")
    def scorer(self):
        from mabench.scoring import EmbeddingRelevanceScorer
        return EmbeddingRelevanceScorer()

    def test_entity_layer_within_bounds(self, scorer):
        """_ENTITY_LAYER must be < num_hidden_layers of the loaded model."""
        scorer._ensure_model()
        # SentenceTransformer wraps the model; access the underlying config
        if hasattr(scorer._model, '__getitem__'):
            # SentenceTransformer: model[0] is the Transformer module
            num_layers = scorer._model[0].auto_model.config.num_hidden_layers
        else:
            # Raw HuggingFace model
            num_layers = scorer._model.config.num_hidden_layers

        entity_layer = scorer._ENTITY_LAYER
        assert entity_layer < num_layers, (
            f"_ENTITY_LAYER={entity_layer} is out of bounds for a model with "
            f"{num_layers} hidden layers (max index = {num_layers - 1})"
        )
        # Also verify it's non-negative
        assert entity_layer >= 0, f"_ENTITY_LAYER={entity_layer} must be >= 0"


# ---------------------------------------------------------------------------
# Fast tests (no model loading)
# ---------------------------------------------------------------------------

class TestDefaultModelName:
    """Verify the default model name is all-mpnet-base-v2 (no model load needed)."""

    def test_default_model_is_mpnet(self):
        from mabench.scoring import EmbeddingRelevanceScorer
        scorer = EmbeddingRelevanceScorer()
        assert scorer._model_name == "sentence-transformers/all-mpnet-base-v2", (
            f"Expected 'sentence-transformers/all-mpnet-base-v2', "
            f"got '{scorer._model_name}'"
        )


class TestDimensionDefaults:
    """Verify dim=768 defaults in agent constructors."""

    def test_hermes_agent_default_dim(self):
        """HermesAgent.__init__ should default dim=768."""
        import inspect
        from mabench.agent import HermesAgent
        sig = inspect.signature(HermesAgent.__init__)
        dim_param = sig.parameters.get("dim")
        assert dim_param is not None, "HermesAgent.__init__ has no 'dim' parameter"
        assert dim_param.default == 768, (
            f"HermesAgent dim default is {dim_param.default}, expected 768"
        )

    def test_hermes_monolithic_default_dim(self):
        """HermesMemoryAgent.__init__ should default dim=768."""
        import inspect
        from mabench.hermes_monolithic import HermesMemoryAgent
        sig = inspect.signature(HermesMemoryAgent.__init__)
        dim_param = sig.parameters.get("dim")
        assert dim_param is not None, "HermesMemoryAgent.__init__ has no 'dim' parameter"
        assert dim_param.default == 768, (
            f"HermesMemoryAgent dim default is {dim_param.default}, expected 768"
        )

    def test_monolithic_scorer_default_model(self):
        """hermes_monolithic.EmbeddingRelevanceScorer should also default to mpnet."""
        from mabench.hermes_monolithic import EmbeddingRelevanceScorer
        scorer = EmbeddingRelevanceScorer()
        assert scorer._model_name == "sentence-transformers/all-mpnet-base-v2", (
            f"Monolithic scorer default is '{scorer._model_name}', "
            f"expected 'sentence-transformers/all-mpnet-base-v2'"
        )


class TestCacheInvalidationComment:
    """Verify that scoring.py documents cache invalidation requirement."""

    def test_cache_invalidation_documented(self):
        """scoring.py should contain a comment about cache invalidation on model switch."""
        scoring_path = (
            Path(__file__).resolve().parent.parent / "mabench" / "scoring.py"
        )
        content = scoring_path.read_text()
        assert "cache" in content.lower() and "invalidat" in content.lower(), (
            "scoring.py should document that embed cache must be invalidated "
            "when switching models"
        )
