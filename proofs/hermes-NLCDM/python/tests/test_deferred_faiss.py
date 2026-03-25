"""TDD tests for deferred FAISS rebuild optimization.

These tests define the behavioral contract for the optimization:
1. Contradiction replacements set _faiss_dirty instead of rebuilding
2. _norm_embs_f32 is maintained incrementally alongside memory_store
3. rebuild_faiss_if_dirty() rebuilds only when dirty
4. New (non-contradiction) entries do incremental faiss.add + append to norm matrix
5. Multiple contradictions accumulate without rebuilding; one rebuild at the end

All tests run on CPU. dim=64 for speed.
"""

import numpy as np
import pytest
import sys
import os

# Prevent OpenMP conflict between torch and FAISS on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coupled_engine import CoupledEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 64
RNG = np.random.RandomState(42)


def _random_emb(dim: int = DIM) -> np.ndarray:
    """Generate a random L2-normalized embedding."""
    v = RNG.randn(dim).astype(np.float64)
    return v / np.linalg.norm(v)


def _near_duplicate(base: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
    """Create a near-duplicate of `base` with small noise, then re-normalize.

    At noise_scale=0.01, cosine similarity to `base` is ~0.9999.
    This will always exceed any reasonable contradiction_threshold (e.g., 0.5).
    """
    v = base + RNG.randn(len(base)) * noise_scale
    return v / np.linalg.norm(v)


def _make_engine(
    dim: int = DIM,
    contradiction_threshold: float = 0.5,
) -> CoupledEngine:
    """Create a CoupledEngine with contradiction_aware=True for testing."""
    return CoupledEngine(
        dim=dim,
        contradiction_aware=True,
        contradiction_threshold=contradiction_threshold,
        emotional_tagging=False,
    )


def _store_n(engine: CoupledEngine, n: int, dim: int = DIM) -> list[np.ndarray]:
    """Store n random entries and return the embeddings used."""
    embs = []
    for i in range(n):
        emb = _random_emb(dim)
        engine.store(text=f"fact_{i}", embedding=emb)
        embs.append(emb)
    return embs


# ---------------------------------------------------------------------------
# Test 1: store() contradiction does NOT rebuild FAISS
# ---------------------------------------------------------------------------

class TestStoreContradictionDoesNotRebuildFaiss:
    """When a contradiction is detected, the FAISS index should NOT be rebuilt.
    Instead, _faiss_dirty should be set to True."""

    def test_store_contradiction_does_not_rebuild_faiss(self):
        engine = _make_engine()
        embs = _store_n(engine, 100)

        # Verify FAISS has 100 entries
        assert engine._faiss_index is not None
        assert engine._faiss_index.ntotal == 100

        # Store a near-duplicate of entry 50 (will trigger contradiction)
        dup_emb = _near_duplicate(embs[50])
        engine.store(text="updated_fact_50", embedding=dup_emb)

        # The FAISS index should NOT have been rebuilt — ntotal should remain
        # at the old count (100), not reflect the current memory_store length
        # (still 100 because contradiction replaces in-place, not appends).
        # The key assertion: _faiss_dirty is True (deferred rebuild).
        assert engine._faiss_dirty is True, (
            "_faiss_dirty should be True after contradiction replacement"
        )
        # ntotal should still be 100 (no rebuild happened, but also no new entry)
        assert engine._faiss_index.ntotal == 100

        # The memory_store entry should have been updated
        assert engine.memory_store[50].text == "updated_fact_50"


# ---------------------------------------------------------------------------
# Test 2: _norm_embs_f32 maintained incrementally
# ---------------------------------------------------------------------------

class TestNormEmbsF32Maintained:
    """The _norm_embs_f32 matrix should be maintained incrementally:
    - Grows on new entries (append)
    - Updates in-place on contradiction replacements
    - All rows are L2-normalized
    """

    def test_shape_after_initial_stores(self):
        engine = _make_engine()
        _store_n(engine, 10)

        assert hasattr(engine, "_norm_embs_f32"), (
            "Engine should have _norm_embs_f32 attribute"
        )
        assert engine._norm_embs_f32.shape == (10, DIM)

    def test_all_rows_l2_normalized(self):
        engine = _make_engine()
        _store_n(engine, 10)

        norms = np.linalg.norm(engine._norm_embs_f32, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_contradiction_replaces_in_place(self):
        engine = _make_engine()
        embs = _store_n(engine, 10)

        # Store a near-duplicate of entry 5
        dup_emb = _near_duplicate(embs[5])
        engine.store(text="replaced_5", embedding=dup_emb)

        # Shape should still be (10, dim) — no new row
        assert engine._norm_embs_f32.shape == (10, DIM)

        # The replaced row should match the NEW embedding (normalized)
        dup_normed = dup_emb / np.linalg.norm(dup_emb)
        replaced_row = engine._norm_embs_f32[5]
        cos_sim = float(np.dot(replaced_row, dup_normed.astype(np.float32)))
        assert cos_sim > 0.99, (
            f"Replaced row should match new embedding, got cosine={cos_sim}"
        )


# ---------------------------------------------------------------------------
# Test 3: rebuild_faiss_if_dirty
# ---------------------------------------------------------------------------

class TestRebuildFaissIfDirty:
    """rebuild_faiss_if_dirty() should rebuild FAISS only when _faiss_dirty is True."""

    def test_rebuild_clears_dirty_flag(self):
        engine = _make_engine()
        embs = _store_n(engine, 20)

        # Trigger a contradiction to set _faiss_dirty
        dup_emb = _near_duplicate(embs[10])
        engine.store(text="replaced_10", embedding=dup_emb)
        assert engine._faiss_dirty is True

        # Rebuild
        engine.rebuild_faiss_if_dirty()

        # Dirty flag should be cleared
        assert engine._faiss_dirty is False

    def test_rebuild_updates_faiss_ntotal(self):
        engine = _make_engine()
        embs = _store_n(engine, 20)

        # Trigger contradiction
        dup_emb = _near_duplicate(embs[10])
        engine.store(text="replaced_10", embedding=dup_emb)

        engine.rebuild_faiss_if_dirty()

        # ntotal should match memory_store length
        assert engine._faiss_index.ntotal == len(engine.memory_store)

    def test_rebuild_search_correctness(self):
        engine = _make_engine()
        embs = _store_n(engine, 20)

        # Trigger contradiction at index 10
        dup_emb = _near_duplicate(embs[10])
        engine.store(text="replaced_10", embedding=dup_emb)

        engine.rebuild_faiss_if_dirty()

        # Search for the replacement embedding — should find index 10
        query = dup_emb / np.linalg.norm(dup_emb)
        query_f32 = query.astype(np.float32).reshape(1, -1)
        D, I = engine._faiss_index.search(query_f32, 1)
        assert int(I[0, 0]) == 10, (
            f"FAISS search after rebuild should find replaced entry at index 10, got {I[0,0]}"
        )


# ---------------------------------------------------------------------------
# Test 4: rebuild_faiss_if_dirty is no-op when clean
# ---------------------------------------------------------------------------

class TestRebuildFaissNoopWhenClean:
    """When _faiss_dirty is False, rebuild_faiss_if_dirty() should be a no-op."""

    def test_noop_when_clean(self):
        engine = _make_engine()
        _store_n(engine, 15)

        # No contradictions — dirty should be False
        assert engine._faiss_dirty is False

        old_ntotal = engine._faiss_index.ntotal
        engine.rebuild_faiss_if_dirty()

        # Nothing should have changed
        assert engine._faiss_index.ntotal == old_ntotal
        assert engine._faiss_dirty is False


# ---------------------------------------------------------------------------
# Test 5: contradiction detection uses norm matrix (not stale FAISS)
# ---------------------------------------------------------------------------

class TestContradictionDetectionUsesNormMatrix:
    """After a contradiction sets _faiss_dirty=True, subsequent contradictions
    should still correctly detect the most similar entry using _norm_embs_f32
    (numpy matmul), not the stale FAISS index."""

    def test_detection_correct_with_stale_faiss(self):
        engine = _make_engine()
        embs = _store_n(engine, 30)

        # First contradiction: replace entry 10
        dup1 = _near_duplicate(embs[10])
        idx1 = engine.store(text="replaced_10_v1", embedding=dup1)
        assert idx1 == 10
        assert engine._faiss_dirty is True  # FAISS is now stale

        # Second contradiction: replace entry 10 AGAIN with a near-duplicate
        # of the UPDATED embedding. Even though FAISS is stale, the engine
        # should detect the similarity via _norm_embs_f32.
        dup2 = _near_duplicate(dup1)
        idx2 = engine.store(text="replaced_10_v2", embedding=dup2)
        assert idx2 == 10, (
            f"Second contradiction should also target index 10, got {idx2}"
        )
        assert engine.memory_store[10].text == "replaced_10_v2"


# ---------------------------------------------------------------------------
# Test 6: new entry appends to norm matrix
# ---------------------------------------------------------------------------

class TestNewEntryAppendsToNormMatrix:
    """Genuinely new entries (low cosine with all existing) should:
    - Append a row to _norm_embs_f32
    - Do incremental faiss.add (no dirty flag)
    """

    def test_append_on_new_entry(self):
        engine = _make_engine()
        _store_n(engine, 5)

        assert engine._norm_embs_f32.shape[0] == 5

        # Store a genuinely new entry (random, unlikely to be near any existing)
        new_emb = _random_emb()
        engine.store(text="genuinely_new", embedding=new_emb)

        # Norm matrix should have grown by 1
        assert engine._norm_embs_f32.shape[0] == 6

        # FAISS should NOT be dirty (incremental add, not a contradiction)
        assert engine._faiss_dirty is False

        # FAISS ntotal should also be 6
        assert engine._faiss_index.ntotal == 6


# ---------------------------------------------------------------------------
# Test 7: multiple contradictions, single rebuild
# ---------------------------------------------------------------------------

class TestMultipleContradictionsSingleRebuild:
    """Multiple contradictions should accumulate _faiss_dirty without triggering
    individual rebuilds. A single rebuild_faiss_if_dirty() at the end suffices."""

    def test_multiple_contradictions_single_rebuild(self):
        engine = _make_engine()
        embs = _store_n(engine, 50)

        initial_ntotal = engine._faiss_index.ntotal
        assert initial_ntotal == 50

        # Store 10 near-duplicates targeting different entries
        targets = [5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
        for i, target_idx in enumerate(targets):
            dup = _near_duplicate(embs[target_idx])
            idx = engine.store(text=f"updated_{target_idx}", embedding=dup)
            assert idx == target_idx, (
                f"Contradiction #{i} should target index {target_idx}, got {idx}"
            )

        # After 10 contradictions:
        # - _faiss_dirty should be True
        # - FAISS should NOT have been rebuilt 10 times
        #   (ntotal should still be 50 — contradictions replace in-place)
        assert engine._faiss_dirty is True
        assert engine._faiss_index.ntotal == 50

        # All replaced entries should have updated text
        for target_idx in targets:
            assert engine.memory_store[target_idx].text == f"updated_{target_idx}"

        # Single rebuild
        engine.rebuild_faiss_if_dirty()

        # FAISS is now clean and correct
        assert engine._faiss_dirty is False
        assert engine._faiss_index.ntotal == 50

        # Verify search correctness: search for one of the replaced embeddings
        target_emb = engine.memory_store[25].embedding
        target_normed = target_emb / (np.linalg.norm(target_emb) + 1e-12)
        query_f32 = target_normed.astype(np.float32).reshape(1, -1)
        D, I = engine._faiss_index.search(query_f32, 1)
        assert int(I[0, 0]) == 25, (
            f"After rebuild, search for entry 25 should return 25, got {I[0,0]}"
        )
