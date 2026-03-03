"""Tests for FAISS ANN pre-filter optimization.

Written BEFORE implementation (TDD). These tests define the behavioral
contracts for the FAISS ANN pre-filter on MemoryOrchestrator.store() and
CoupledEngine.store(). All tests should FAIL until the implementation is
complete.

Spec: thoughts/shared/plans/faiss-ann/spec.md
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup: need both hermes-NLCDM/python and hermes-memory/python
# ---------------------------------------------------------------------------

_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
_HERMES_MEMORY = _NLCDM_PYTHON.parent.parent / "hermes-memory" / "python"
sys.path.insert(0, str(_NLCDM_PYTHON))
sys.path.insert(0, str(_HERMES_MEMORY))

import faiss  # noqa: E402 (must come after path setup)
from coupled_engine import CoupledEngine  # noqa: E402
from hermes_memory.contradiction import (  # noqa: E402
    ContradictionConfig,
    ContradictionDetection,
    ContradictionResult,
    ContradictionType,
    SubjectExtraction,
    SupersessionAction,
)
from hermes_memory.engine import ParameterSet  # noqa: E402
from hermes_memory.orchestrator import MemoryOrchestrator, RelevanceScorer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockScorer:
    """Simple mock scorer that satisfies the RelevanceScorer protocol."""

    def score(self, query: str, content: str) -> float:
        return 0.5


def _default_params() -> ParameterSet:
    """Construct a valid ParameterSet with typical test values."""
    return ParameterSet(
        alpha=0.1,
        beta=0.05,
        delta_t=1.0,
        s_max=1.0,
        s0=0.5,
        temperature=1.0,
        novelty_start=0.2,
        novelty_decay=0.1,
        survival_threshold=0.1,
        feedback_sensitivity=0.1,
        w1=0.25,
        w2=0.25,
        w3=0.25,
        w4=0.25,
    )


def _make_orchestrator(
    contradiction_config: ContradictionConfig | None = None,
) -> MemoryOrchestrator:
    """Build a MemoryOrchestrator with sensible test defaults."""
    return MemoryOrchestrator(
        params=_default_params(),
        contradiction_config=contradiction_config or ContradictionConfig(),
        relevance_scorer=MockScorer(),
    )


def random_embedding(dim: int = 1024, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random L2-normalized embedding vector."""
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(dim).astype(np.float64)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        v[0] = 1.0
        norm = 1.0
    return v / norm


def similar_embedding(
    base: np.ndarray,
    noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an embedding similar to base (high cosine similarity).

    The noise parameter controls deviation -- smaller noise = higher cosine.
    With noise=0.1 on 1024-dim vectors, cosine similarity is typically > 0.99.
    """
    if rng is None:
        rng = np.random.default_rng()
    v = base + rng.standard_normal(len(base)) * noise
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        v[0] = 1.0
        norm = 1.0
    return v / norm


def distant_embedding(
    base: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an embedding far from base (low cosine similarity).

    Creates a random vector and subtracts the base component to push
    it orthogonal, then adds random noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(len(base)).astype(np.float64)
    # Remove the component along base to make approximately orthogonal
    v = v - np.dot(v, base) * base
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        v[0] = 1.0
        norm = 1.0
    return v / norm


# ---------------------------------------------------------------------------
# 1. test_orchestrator_store_with_embedding_creates_faiss_index
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestOrchestratorStoreWithEmbedding:
    """Verify that store() with an embedding creates and populates a FAISS index."""

    def test_store_with_embedding_creates_faiss_index(self):
        """Create a MemoryOrchestrator, call store() with a random 1024-dim
        L2-normalized embedding. Verify that after the store, the orchestrator
        has a _faiss_index attribute and _faiss_index.ntotal == 1.
        """
        orch = _make_orchestrator()
        emb = random_embedding(1024)

        orch.store(content="Alice's favorite color is blue", embedding=emb)

        assert hasattr(orch, "_faiss_index"), (
            "MemoryOrchestrator should have _faiss_index after store with embedding"
        )
        assert orch._faiss_index is not None, (
            "_faiss_index should not be None after store with embedding"
        )
        assert orch._faiss_index.ntotal == 1, (
            f"Expected 1 vector in FAISS index, got {orch._faiss_index.ntotal}"
        )

    def test_faiss_id_map_populated(self):
        """After store with embedding, _faiss_id_map should have one entry
        mapping to the memory_id of the stored memory."""
        orch = _make_orchestrator()
        emb = random_embedding(1024)

        result = orch.store(content="Alice's favorite color is blue", embedding=emb)

        assert hasattr(orch, "_faiss_id_map")
        assert len(orch._faiss_id_map) == 1
        assert orch._faiss_id_map[0] == result.memory_id

    def test_faiss_dim_set(self):
        """After store with embedding, _faiss_dim should be set to embedding
        dimension."""
        orch = _make_orchestrator()
        emb = random_embedding(1024)

        orch.store(content="Alice's favorite color is blue", embedding=emb)

        assert hasattr(orch, "_faiss_dim")
        assert orch._faiss_dim == 1024


# ---------------------------------------------------------------------------
# 2. test_orchestrator_store_without_embedding_no_faiss
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestOrchestratorStoreWithoutEmbedding:
    """Verify that store() without embedding preserves backwards compatibility."""

    def test_store_without_embedding_still_works(self):
        """Call store() WITHOUT embedding. Verify it still works (backward
        compat) and _faiss_index remains None.

        This test validates backward compatibility: store() without the new
        embedding= parameter should behave identically to pre-FAISS behavior.
        The _faiss_index field must exist (initialized in __init__ to None)
        after the FAISS implementation is added.
        """
        orch = _make_orchestrator()

        result = orch.store(content="Alice's favorite color is blue")

        assert result.stored is True
        # After FAISS implementation, _faiss_index should exist as a field
        # (initialized to None in __init__) even when no embedding is provided.
        assert hasattr(orch, "_faiss_index"), (
            "After FAISS implementation, _faiss_index should be an attribute "
            "(initialized to None in __init__)"
        )
        assert orch._faiss_index is None, (
            "_faiss_index should be None when no embedding provided"
        )

    @pytest.mark.xfail(reason="FAISS ANN not yet implemented")
    def test_mixed_store_only_embedding_stores_in_faiss(self):
        """Store once without embedding, once with. Only the second should
        appear in the FAISS index."""
        orch = _make_orchestrator()

        # First store: no embedding
        orch.store(content="Some unrelated fact")

        # Second store: with embedding
        emb = random_embedding(1024)
        orch.store(content="Another fact", embedding=emb)

        assert orch._faiss_index is not None
        assert orch._faiss_index.ntotal == 1, (
            "Only embedding-backed stores should appear in FAISS index"
        )


# ---------------------------------------------------------------------------
# 3. test_contradiction_detected_via_ann
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestContradictionDetectedViaANN:
    """Verify contradictions are still detected when using ANN pre-filter."""

    def test_contradiction_detected_with_similar_embeddings(self):
        """Store 'Alice's favorite color is blue' with an embedding, then
        store 'Alice's favorite color is red' with a similar embedding.
        Verify contradiction is detected."""
        orch = _make_orchestrator()
        rng = np.random.default_rng(42)

        base_emb = random_embedding(1024, rng=rng)

        # Store first fact
        result1 = orch.store(
            content="Alice's favorite color is blue",
            embedding=base_emb,
        )
        assert result1.stored is True

        # Store contradicting fact with similar embedding
        emb2 = similar_embedding(base_emb, noise=0.05, rng=rng)
        result2 = orch.store(
            content="Alice's favorite color is red",
            embedding=emb2,
        )

        # Contradiction should be detected
        assert result2.contradiction_result is not None, (
            "Contradiction should be detected between 'blue' and 'red' color preferences"
        )
        assert result2.contradiction_result.has_contradiction is True


# ---------------------------------------------------------------------------
# 4. test_ann_filters_distant_memories
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestANNFiltersDistantMemories:
    """Verify that ANN pre-filter reduces the candidate set for
    detect_contradictions, passing only nearby memories."""

    def test_distant_memories_not_passed_to_detection(self):
        """Store many unrelated memories with diverse embeddings, then store
        a contradicting memory. Verify that only nearby memories are passed
        to detect_contradictions (not all N)."""
        orch = _make_orchestrator()
        rng = np.random.default_rng(42)

        # Store 60 unrelated memories with diverse embeddings
        base_embs = []
        for i in range(60):
            emb = random_embedding(1024, rng=rng)
            orch.store(
                content=f"Unrelated fact number {i} about topic {i}",
                embedding=emb,
            )
            base_embs.append(emb)

        # Now store a memory that should only contradict the first one
        # Use a similar embedding to base_embs[0]
        near_emb = similar_embedding(base_embs[0], noise=0.05, rng=rng)

        # Patch detect_contradictions to observe what it receives
        with patch(
            "hermes_memory.orchestrator.detect_contradictions",
            wraps=__import__(
                "hermes_memory.contradiction", fromlist=["detect_contradictions"]
            ).detect_contradictions,
        ) as mock_detect:
            orch.store(
                content="Contradicting fact about topic 0",
                embedding=near_emb,
            )

            if mock_detect.called:
                call_args = mock_detect.call_args
                existing_texts = call_args.kwargs.get(
                    "existing_texts",
                    call_args.args[2] if len(call_args.args) > 2 else None,
                )
                if existing_texts is not None:
                    # ANN filter should reduce the candidate set significantly
                    # (50 max_candidates or fewer, not all 60)
                    assert len(existing_texts) <= 50, (
                        f"ANN should filter candidates to <= 50, got {len(existing_texts)}"
                    )


# ---------------------------------------------------------------------------
# 5. test_dimension_mismatch_raises
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestDimensionMismatchRaises:
    """Verify that mismatched embedding dimensions raise ValueError."""

    def test_dimension_mismatch_raises_value_error(self):
        """Store with a 1024-dim embedding, then try to store with a 512-dim
        embedding. Expect ValueError."""
        orch = _make_orchestrator()

        emb_1024 = random_embedding(1024)
        orch.store(content="First fact", embedding=emb_1024)

        emb_512 = random_embedding(512)
        with pytest.raises(ValueError, match=r"[Dd]imension.*512.*1024|[Dd]imension.*1024.*512"):
            orch.store(content="Second fact", embedding=emb_512)

    def test_consistent_dimensions_no_error(self):
        """Multiple stores with same dimension should not raise."""
        orch = _make_orchestrator()
        rng = np.random.default_rng(42)

        for i in range(5):
            emb = random_embedding(1024, rng=rng)
            result = orch.store(content=f"Fact {i}", embedding=emb)
            assert result.stored is True

        assert orch._faiss_index.ntotal == 5


# ---------------------------------------------------------------------------
# 6. test_first_store_empty_index
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestFirstStoreEmptyIndex:
    """Verify first store with embedding does not crash (no ANN search
    on empty index)."""

    def test_first_store_no_crash(self):
        """First store with embedding should work without errors.
        FAISS index starts empty, so no search should be attempted."""
        orch = _make_orchestrator()
        emb = random_embedding(1024)

        # This must not raise
        result = orch.store(content="Very first memory ever", embedding=emb)

        assert result.stored is True
        assert orch._faiss_index is not None
        assert orch._faiss_index.ntotal == 1

    def test_second_store_searches_first(self):
        """Second store should search the FAISS index (which has 1 entry)
        without errors."""
        orch = _make_orchestrator()
        rng = np.random.default_rng(42)
        emb1 = random_embedding(1024, rng=rng)
        emb2 = random_embedding(1024, rng=rng)

        orch.store(content="First memory", embedding=emb1)
        result2 = orch.store(content="Second memory", embedding=emb2)

        assert result2.stored is True
        assert orch._faiss_index.ntotal == 2


# ---------------------------------------------------------------------------
# 7. test_incremental_rebuild_index
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestIncrementalRebuildIndex:
    """Verify _id_to_idx is correctly maintained with incremental updates."""

    def test_100_memories_correct_id_mapping(self):
        """Store 100 memories. Verify _id_to_idx has 100 entries and each
        memory's id maps to the correct index in _memories."""
        orch = _make_orchestrator()
        rng = np.random.default_rng(42)

        stored_ids = []
        for i in range(100):
            emb = random_embedding(1024, rng=rng)
            result = orch.store(content=f"Memory number {i}", embedding=emb)
            if result.stored and result.memory_id is not None:
                stored_ids.append(result.memory_id)

        # Every stored memory_id should be in _id_to_idx
        for mid in stored_ids:
            assert mid in orch._id_to_idx, (
                f"Memory ID {mid} not found in _id_to_idx"
            )
            # The index should point to a memory with the same ID
            idx = orch._id_to_idx[mid]
            assert orch._memories[idx].memory_id == mid, (
                f"_id_to_idx[{mid}] = {idx} but _memories[{idx}].memory_id = "
                f"{orch._memories[idx].memory_id}"
            )

    def test_id_to_idx_count_matches_memories(self):
        """_id_to_idx should have exactly as many entries as _memories."""
        orch = _make_orchestrator()
        rng = np.random.default_rng(42)

        for i in range(50):
            emb = random_embedding(1024, rng=rng)
            orch.store(content=f"Memory {i}", embedding=emb)

        assert len(orch._id_to_idx) == len(orch._memories), (
            f"_id_to_idx has {len(orch._id_to_idx)} entries but "
            f"_memories has {len(orch._memories)}"
        )


# ---------------------------------------------------------------------------
# 8. test_remap_contradiction_result
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestRemapContradictionResult:
    """Test _remap_contradiction_result() with known mappings."""

    def _make_detection(self, existing_index: int) -> ContradictionDetection:
        """Create a minimal ContradictionDetection for testing."""
        return ContradictionDetection(
            existing_index=existing_index,
            contradiction_type=ContradictionType.VALUE_UPDATE,
            confidence=0.9,
            subject_overlap=0.8,
            candidate_subject=SubjectExtraction(
                subject="alice color",
                value="red",
                field_type="preference",
                raw_match="Alice's favorite color is red",
            ),
            existing_subject=SubjectExtraction(
                subject="alice color",
                value="blue",
                field_type="preference",
                raw_match="Alice's favorite color is blue",
            ),
            explanation="Value update: color changed from blue to red",
        )

    def test_remap_indices_correctly(self):
        """Create a mock ContradictionResult with subset indices.
        Call _remap_contradiction_result() with a known mapping.
        Verify indices are correctly remapped."""
        # Subset-space indices: detection at position 2 in the subset
        detection = self._make_detection(existing_index=2)
        result = ContradictionResult(
            detections=(detection,),
            actions=((2, SupersessionAction.AUTO_SUPERSEDE),),
            superseded_indices=frozenset({2}),
            flagged_indices=frozenset(),
            has_contradiction=True,
            highest_confidence=0.9,
        )

        # Mapping: subset index -> active-list index
        # subset[0] -> active[10], subset[1] -> active[25], subset[2] -> active[42]
        subset_to_active = [10, 25, 42]

        # Import and call the remap function
        from hermes_memory.orchestrator import MemoryOrchestrator

        remapped = MemoryOrchestrator._remap_contradiction_result(
            result, subset_to_active
        )

        # Verify remapped indices
        assert remapped.detections[0].existing_index == 42, (
            f"Expected remapped index 42, got {remapped.detections[0].existing_index}"
        )
        assert remapped.actions[0][0] == 42, (
            f"Expected remapped action index 42, got {remapped.actions[0][0]}"
        )
        assert 42 in remapped.superseded_indices, (
            f"Expected 42 in superseded_indices, got {remapped.superseded_indices}"
        )
        assert remapped.has_contradiction is True
        assert remapped.highest_confidence == 0.9

    def test_remap_multiple_detections(self):
        """Remap with multiple detections at different subset positions."""
        det0 = self._make_detection(existing_index=0)
        det1 = self._make_detection(existing_index=3)

        result = ContradictionResult(
            detections=(det0, det1),
            actions=(
                (0, SupersessionAction.AUTO_SUPERSEDE),
                (3, SupersessionAction.FLAG_CONFLICT),
            ),
            superseded_indices=frozenset({0}),
            flagged_indices=frozenset({3}),
            has_contradiction=True,
            highest_confidence=0.9,
        )

        subset_to_active = [5, 15, 25, 35, 45]

        from hermes_memory.orchestrator import MemoryOrchestrator

        remapped = MemoryOrchestrator._remap_contradiction_result(
            result, subset_to_active
        )

        assert remapped.detections[0].existing_index == 5
        assert remapped.detections[1].existing_index == 35
        assert remapped.actions[0][0] == 5
        assert remapped.actions[1][0] == 35
        assert 5 in remapped.superseded_indices
        assert 35 in remapped.flagged_indices


# ---------------------------------------------------------------------------
# 9. test_coupled_engine_faiss_store
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestCoupledEngineFAISSStore:
    """Verify CoupledEngine gets a FAISS index that grows with stores."""

    def test_faiss_index_exists_after_store(self):
        """Create a CoupledEngine, store entries with embeddings.
        Verify it has a _faiss_index attribute that grows."""
        ce = CoupledEngine(dim=1024, contradiction_aware=True)
        rng = np.random.default_rng(42)

        for i in range(5):
            emb = random_embedding(1024, rng=rng)
            ce.store(text=f"Memory {i}", embedding=emb)

        assert hasattr(ce, "_faiss_index"), (
            "CoupledEngine should have _faiss_index after stores"
        )
        assert ce._faiss_index is not None
        assert ce._faiss_index.ntotal == 5, (
            f"Expected 5 vectors in FAISS index, got {ce._faiss_index.ntotal}"
        )

    def test_faiss_index_grows_incrementally(self):
        """Each store should add exactly one vector to the FAISS index."""
        ce = CoupledEngine(dim=1024, contradiction_aware=True)
        rng = np.random.default_rng(42)

        for i in range(10):
            emb = random_embedding(1024, rng=rng)
            ce.store(text=f"Memory {i}", embedding=emb)
            assert ce._faiss_index.ntotal == i + 1, (
                f"After {i+1} stores, expected {i+1} vectors, "
                f"got {ce._faiss_index.ntotal}"
            )

    def test_faiss_index_used_for_contradiction(self):
        """When contradiction_aware=True, FAISS should find the near-duplicate
        and the engine should replace it. After replacement, the FAISS index
        must reflect the update (stale row removed or updated)."""
        ce = CoupledEngine(
            dim=1024,
            contradiction_aware=True,
            contradiction_threshold=0.85,
        )
        rng = np.random.default_rng(42)

        base_emb = random_embedding(1024, rng=rng)
        ce.store(text="Alice likes blue", embedding=base_emb)
        assert ce.n_memories == 1

        # Store near-duplicate -- should replace, not add
        near_emb = similar_embedding(base_emb, noise=0.01, rng=rng)
        ce.store(text="Alice likes red", embedding=near_emb)

        # With contradiction_aware, the old entry should be replaced
        assert ce.n_memories == 1, (
            f"Expected 1 memory after contradiction replacement, got {ce.n_memories}"
        )

        # FAISS-specific: the index must exist and be consistent
        assert hasattr(ce, "_faiss_index"), (
            "CoupledEngine should have _faiss_index after stores"
        )
        assert ce._faiss_index is not None, (
            "_faiss_index should not be None after stores"
        )
        # After contradiction replacement, FAISS index should have exactly 1
        # entry (the replacement), not 2 (stale + new)
        assert ce._faiss_index.ntotal == 1, (
            f"After contradiction replacement, FAISS index should have 1 entry, "
            f"got {ce._faiss_index.ntotal}"
        )


# ---------------------------------------------------------------------------
# 10. test_coupled_engine_prediction_error_uses_faiss
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestCoupledEnginePredictionErrorFAISS:
    """Verify _prediction_error uses FAISS for O(1) lookup.

    The behavioral contract of _prediction_error is unchanged (same numeric
    result). The FAISS optimization is an internal implementation detail.
    These tests verify both correctness AND that the FAISS index is being
    used (by checking _faiss_index exists and is populated).
    """

    def test_prediction_error_similar_embedding_near_zero(self):
        """Store a known embedding, then compute prediction error for a
        very similar embedding. Should be near 0 (low surprise).
        Also verify FAISS index is used."""
        ce = CoupledEngine(dim=1024, emotional_tagging=True)
        rng = np.random.default_rng(42)

        base_emb = random_embedding(1024, rng=rng)
        ce.store(text="Base memory", embedding=base_emb)

        # FAISS index must exist after store
        assert hasattr(ce, "_faiss_index") and ce._faiss_index is not None, (
            "CoupledEngine._faiss_index should exist and be non-None after store"
        )
        assert ce._faiss_index.ntotal == 1

        # Very similar embedding -> low prediction error
        near_emb = similar_embedding(base_emb, noise=0.01, rng=rng)
        error = ce._prediction_error(near_emb)

        assert error < 0.1, (
            f"Prediction error for similar embedding should be near 0, got {error}"
        )

    def test_prediction_error_distant_embedding_near_one(self):
        """Compute prediction error for a very different embedding.
        Should be near 1 (high surprise).
        Also verify FAISS index is used."""
        ce = CoupledEngine(dim=1024, emotional_tagging=True)
        rng = np.random.default_rng(42)

        base_emb = random_embedding(1024, rng=rng)
        ce.store(text="Base memory", embedding=base_emb)

        # FAISS index must exist
        assert hasattr(ce, "_faiss_index") and ce._faiss_index is not None

        # Orthogonal embedding -> high prediction error
        far_emb = distant_embedding(base_emb, rng=rng)
        error = ce._prediction_error(far_emb)

        assert error > 0.8, (
            f"Prediction error for distant embedding should be near 1, got {error}"
        )

    def test_prediction_error_empty_returns_one(self):
        """Empty engine should return prediction error of 1.0.
        FAISS index should be None or have ntotal == 0."""
        ce = CoupledEngine(dim=1024, emotional_tagging=True)

        # Before any store, _faiss_index should be initialized to None
        assert hasattr(ce, "_faiss_index"), (
            "CoupledEngine should have _faiss_index attribute after __init__"
        )
        faiss_idx = ce._faiss_index
        assert faiss_idx is None or faiss_idx.ntotal == 0

        emb = random_embedding(1024)
        error = ce._prediction_error(emb)
        assert error == 1.0, (
            f"Prediction error with no memories should be 1.0, got {error}"
        )

    def test_prediction_error_consistent_with_direct_computation(self):
        """Prediction error computed via FAISS should match the value
        computed via direct cosine similarity (within tolerance).
        Verify FAISS index is populated correctly."""
        ce = CoupledEngine(dim=1024, emotional_tagging=True)
        rng = np.random.default_rng(42)

        # Store several memories
        embeddings = []
        for i in range(20):
            emb = random_embedding(1024, rng=rng)
            ce.store(text=f"Memory {i}", embedding=emb)
            embeddings.append(emb)

        # FAISS index must have all 20 entries
        assert hasattr(ce, "_faiss_index") and ce._faiss_index is not None
        assert ce._faiss_index.ntotal == 20, (
            f"FAISS index should have 20 entries, got {ce._faiss_index.ntotal}"
        )

        # Query with a new embedding
        query_emb = random_embedding(1024, rng=rng)

        # Compute expected prediction error via direct cosine similarity
        X = np.array(embeddings, dtype=np.float64)
        q_norm = np.linalg.norm(query_emb)
        x_norms = np.linalg.norm(X, axis=1)
        sims = X @ query_emb / (x_norms * q_norm + 1e-12)
        max_sim = float(np.max(sims))
        expected_error = max(0.0, min(1.0 - max_sim, 1.0))

        # Compute via CoupledEngine (which should use FAISS after implementation)
        actual_error = ce._prediction_error(query_emb)

        np.testing.assert_allclose(
            actual_error,
            expected_error,
            atol=1e-6,
            err_msg=(
                f"FAISS-based prediction error {actual_error} differs from "
                f"direct computation {expected_error}"
            ),
        )


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="FAISS ANN not yet implemented")
class TestFAISSIndexConsistency:
    """Verify FAISS index consistency contracts from spec section 5.3."""

    def test_ntotal_equals_id_map_length(self):
        """After every store, _faiss_index.ntotal == len(_faiss_id_map)."""
        orch = _make_orchestrator()
        rng = np.random.default_rng(42)

        for i in range(20):
            emb = random_embedding(1024, rng=rng)
            orch.store(content=f"Fact {i}", embedding=emb)

            assert orch._faiss_index.ntotal == len(orch._faiss_id_map), (
                f"After store {i+1}: ntotal={orch._faiss_index.ntotal} != "
                f"len(_faiss_id_map)={len(orch._faiss_id_map)}"
            )

    def test_all_faiss_ids_exist_in_id_to_idx(self):
        """Every memory_id in _faiss_id_map exists in _id_to_idx."""
        orch = _make_orchestrator()
        rng = np.random.default_rng(42)

        for i in range(20):
            emb = random_embedding(1024, rng=rng)
            orch.store(content=f"Fact {i}", embedding=emb)

        for mid in orch._faiss_id_map:
            assert mid in orch._id_to_idx, (
                f"Memory ID {mid} from _faiss_id_map not in _id_to_idx"
            )
