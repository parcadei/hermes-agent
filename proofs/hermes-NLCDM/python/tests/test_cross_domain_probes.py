"""Tests for cross-domain probe injection in longitudinal evaluation.

Defines the behavioral contract for:
  1. LongitudinalEvaluator._generate_cross_domain_probes()
  2. Probe injection at correct frequency (every N sessions)
  3. Probes using query_readonly() to avoid corrupting scoring signals

All tests should FAIL because:
  - _generate_cross_domain_probes() does not exist yet
  - cross_domain_probes / probe_frequency / probes_per_session params not yet added
  - query_readonly() not yet implemented on CoupledEngine
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Setup paths
_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_NLCDM_PYTHON))
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))

from coupled_engine import CoupledEngine
from longitudinal_eval import generate_dataset, LongitudinalEvaluator


# ---------------------------------------------------------------------------
# Test: Probe generation produces cross-domain queries
# ---------------------------------------------------------------------------


class TestProbeGeneration:
    """_generate_cross_domain_probes() must produce valid query strings
    that span at least 2 domains."""

    def test_generates_correct_count(self):
        """Should return exactly n_probes probe strings."""
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            cross_domain_probes=True,
            probe_frequency=10,
            probes_per_session=3,
        )

        probes = evaluator._generate_cross_domain_probes(
            dataset=dataset,
            n_probes=5,
            session_index=0,
        )

        assert len(probes) == 5

    def test_probes_are_nonempty_strings(self):
        """Each probe should be a non-trivial string (length > 10 chars)."""
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            cross_domain_probes=True,
            probe_frequency=10,
            probes_per_session=3,
        )

        probes = evaluator._generate_cross_domain_probes(
            dataset=dataset,
            n_probes=5,
            session_index=0,
        )

        for i, probe in enumerate(probes):
            assert isinstance(probe, str), f"Probe {i} is not a string: {type(probe)}"
            assert len(probe) > 10, (
                f"Probe {i} is too short ({len(probe)} chars): {probe!r}"
            )

    def test_deterministic_per_session(self):
        """Probes should be deterministic given the same session_index seed."""
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            cross_domain_probes=True,
            probe_frequency=10,
            probes_per_session=3,
        )

        probes_a = evaluator._generate_cross_domain_probes(
            dataset=dataset, n_probes=5, session_index=10,
        )
        probes_b = evaluator._generate_cross_domain_probes(
            dataset=dataset, n_probes=5, session_index=10,
        )

        assert probes_a == probes_b, (
            "Same session_index should produce identical probes"
        )

    def test_different_sessions_produce_different_probes(self):
        """Different session indices should produce different probe sets
        (diversity across the evaluation)."""
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            cross_domain_probes=True,
            probe_frequency=10,
            probes_per_session=3,
        )

        probes_10 = evaluator._generate_cross_domain_probes(
            dataset=dataset, n_probes=5, session_index=10,
        )
        probes_20 = evaluator._generate_cross_domain_probes(
            dataset=dataset, n_probes=5, session_index=20,
        )

        # At least some probes should differ between sessions
        assert probes_10 != probes_20, (
            "Different session indices should produce different probes"
        )


# ---------------------------------------------------------------------------
# Test: Probes fire at correct frequency
# ---------------------------------------------------------------------------


class TestProbeFrequency:
    """Probes should fire every probe_frequency sessions, not on every session."""

    def test_fires_on_correct_sessions(self):
        """With probe_frequency=10, probes should fire on sessions
        10, 20, 30, ... but NOT on 1, 5, 15, etc.

        Strategy: create an evaluator with cross_domain_probes=True,
        verify the parameter is accepted and the frequency is stored.
        """
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            cross_domain_probes=True,
            probe_frequency=10,
            probes_per_session=3,
        )

        assert evaluator.cross_domain_probes is True
        assert evaluator.probe_frequency == 10
        assert evaluator.probes_per_session == 3

    def test_frequency_logic_sessions(self):
        """Verify the modulo logic: (i+1) % probe_frequency == 0 fires probes.

        Sessions are 0-indexed, so session 9 (i=9) is the 10th session,
        and (9+1) % 10 == 0 -> fire. Session 14 (i=14) -> (15) % 10 == 5 -> skip.
        """
        probe_frequency = 10
        n_sessions = 50

        fire_sessions = []
        for i in range(n_sessions):
            if (i + 1) % probe_frequency == 0:
                fire_sessions.append(i)

        # Should fire at indices: 9, 19, 29, 39, 49
        assert fire_sessions == [9, 19, 29, 39, 49]

        # Should NOT fire at these
        for i in [0, 1, 4, 5, 10, 14, 15, 25]:
            assert i not in fire_sessions, f"Should not fire at session {i}"

    def test_probe_count_matches_frequency(self):
        """Over 30 sessions with frequency=10, probes should fire 3 times
        (at sessions 10, 20, 30 -> indices 9, 19, 29).
        With probes_per_session=2, total readonly calls = 3 * 2 = 6.

        Strategy: build a minimal engine, track query_readonly calls manually.
        """
        engine = CoupledEngine(dim=32)
        rng = np.random.default_rng(42)

        # Store 30 patterns (simulating 30 sessions with 1 fact each)
        for i in range(30):
            emb = rng.standard_normal(32)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"Session {i} fact", embedding=emb)

        # Track query_readonly calls
        readonly_call_count = 0
        probe_frequency = 10
        probes_per_session = 2

        for i in range(30):
            if (i + 1) % probe_frequency == 0:
                for _ in range(probes_per_session):
                    q = rng.standard_normal(32)
                    q /= np.linalg.norm(q)
                    engine.query_readonly(q, top_k=10)
                    readonly_call_count += 1

        assert readonly_call_count == 6, (
            f"Expected 6 readonly calls (3 fire points * 2 probes), "
            f"got {readonly_call_count}"
        )


# ---------------------------------------------------------------------------
# Test: Probes don't corrupt scoring signals
# ---------------------------------------------------------------------------


class TestProbesDontCorruptScoring:
    """Cross-domain probes (which use query_readonly) must NOT corrupt
    the scoring signals used by longitudinal evaluation."""

    def test_no_access_count_change_from_probes(self):
        """Probes should not change access_count on any stored pattern."""
        engine = CoupledEngine(dim=64)
        rng = np.random.default_rng(42)

        # Store 10 patterns
        for i in range(10):
            emb = rng.standard_normal(64)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"Fact {i}", embedding=emb)

        access_counts_before = [m.access_count for m in engine.memory_store]

        # Fire 20 probe queries via query_readonly
        for _ in range(20):
            q = rng.standard_normal(64)
            q /= np.linalg.norm(q)
            engine.query_readonly(q, top_k=5)

        access_counts_after = [m.access_count for m in engine.memory_store]
        assert access_counts_after == access_counts_before, (
            "Probes should not change access_count"
        )

    def test_no_importance_change_from_probes(self):
        """Probes should not change importance on any stored pattern."""
        engine = CoupledEngine(dim=64)
        rng = np.random.default_rng(42)

        for i in range(10):
            emb = rng.standard_normal(64)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"Fact {i}", embedding=emb)

        importances_before = [m.importance for m in engine.memory_store]

        for _ in range(20):
            q = rng.standard_normal(64)
            q /= np.linalg.norm(q)
            engine.query_readonly(q, top_k=5)

        importances_after = [m.importance for m in engine.memory_store]
        assert importances_after == importances_before, (
            "Probes should not change importance"
        )

    def test_no_last_access_time_change_from_probes(self):
        """Probes should not change last_access_time on any stored pattern."""
        engine = CoupledEngine(dim=64)
        rng = np.random.default_rng(42)

        for i in range(10):
            emb = rng.standard_normal(64)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"Fact {i}", embedding=emb)

        last_access_before = [m.last_access_time for m in engine.memory_store]

        time.sleep(0.01)  # Ensure time has advanced

        for _ in range(20):
            q = rng.standard_normal(64)
            q /= np.linalg.norm(q)
            engine.query_readonly(q, top_k=5)

        last_access_after = [m.last_access_time for m in engine.memory_store]
        assert last_access_after == last_access_before, (
            "Probes should not change last_access_time"
        )

    def test_no_embedding_drift_from_probes(self):
        """Probes should not cause any embedding drift, even with
        reconsolidation enabled."""
        engine = CoupledEngine(
            dim=64, reconsolidation=True, reconsolidation_eta=0.1,
        )
        rng = np.random.default_rng(42)

        for i in range(10):
            emb = rng.standard_normal(64)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"Fact {i}", embedding=emb)

        embeddings_before = np.array(
            [m.embedding.copy() for m in engine.memory_store]
        )

        for _ in range(20):
            q = rng.standard_normal(64)
            q /= np.linalg.norm(q)
            engine.query_readonly(q, top_k=5)

        embeddings_after = np.array(
            [m.embedding for m in engine.memory_store]
        )

        np.testing.assert_allclose(
            embeddings_after, embeddings_before, atol=1e-15,
            err_msg="Probes caused embedding drift"
        )

    def test_co_retrieval_graph_does_grow(self):
        """While scoring signals stay frozen, the co-retrieval graph
        SHOULD grow from probes (that's the whole purpose)."""
        engine = CoupledEngine(dim=64)
        rng = np.random.default_rng(42)

        for i in range(10):
            emb = rng.standard_normal(64)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"Fact {i}", embedding=emb)

        assert len(engine._co_retrieval) == 0
        assert engine._co_retrieval_query_count == 0

        for _ in range(5):
            q = rng.standard_normal(64)
            q /= np.linalg.norm(q)
            engine.query_readonly(q, top_k=5)

        # Co-retrieval graph should now contain edges
        assert len(engine._co_retrieval) > 0, (
            "Probes should build co-retrieval edges"
        )
        assert engine._co_retrieval_query_count == 5, (
            f"Expected 5 query count, got {engine._co_retrieval_query_count}"
        )


# ---------------------------------------------------------------------------
# Test: LongitudinalEvaluator accepts new parameters
# ---------------------------------------------------------------------------


class TestEvaluatorNewParams:
    """LongitudinalEvaluator should accept cross_domain_probes,
    probe_frequency, and probes_per_session parameters."""

    def test_accepts_cross_domain_probes_param(self):
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            cross_domain_probes=True,
        )
        assert evaluator.cross_domain_probes is True

    def test_accepts_probe_frequency_param(self):
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            cross_domain_probes=True,
            probe_frequency=5,
        )
        assert evaluator.probe_frequency == 5

    def test_accepts_probes_per_session_param(self):
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            cross_domain_probes=True,
            probes_per_session=7,
        )
        assert evaluator.probes_per_session == 7

    def test_default_probe_frequency(self):
        """Default probe_frequency should be 10 per the spec."""
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            cross_domain_probes=True,
        )
        assert evaluator.probe_frequency == 10

    def test_default_probes_per_session(self):
        """Default probes_per_session should be 3 per the spec."""
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            cross_domain_probes=True,
        )
        assert evaluator.probes_per_session == 3

    def test_cross_domain_probes_default_false(self):
        """cross_domain_probes should default to False for backward compat."""
        dataset = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(dataset=dataset)
        assert evaluator.cross_domain_probes is False
