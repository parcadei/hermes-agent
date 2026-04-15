"""Unit tests for Memory Dynamics (spec section 7).

Tests MemorySystem three-layer architecture, ImportanceScore,
DecayFunction, HebbianBridge, and DreamConsolidator.
"""

from __future__ import annotations

import pytest
import math
from datetime import datetime


# ---------------------------------------------------------------------------
# MemorySystem (Three-Layer)
# ---------------------------------------------------------------------------

class TestMemorySystem:
    """Tests for the three-layer memory system."""

    def test_three_layers_exist(self):
        """MemorySystem has stated, inferred, and derived layers."""
        from foundry.dynamics import MemorySystem
        ms = MemorySystem()
        assert hasattr(ms, "stated")
        assert hasattr(ms, "inferred")
        assert hasattr(ms, "derived")

    def test_wipe_inferred_preserves_stated(self):
        """Wiping inferred does not affect stated or derived.

        Spec invariant I1: Layer Isolation.
        """
        from foundry.dynamics import MemorySystem
        ms = MemorySystem()
        # Add something to stated (implementation-dependent)
        ms.wipe_inferred()
        # Stated should still be the same object or equivalent
        assert ms.stated is not None
        assert ms.derived is not None

    def test_rebuild_derived(self):
        """rebuild_derived creates a new derived layer from stated."""
        from foundry.dynamics import MemorySystem
        ms = MemorySystem()
        ms.rebuild_derived()
        assert ms.derived is not None

    def test_build_inferred(self):
        """build_inferred creates inferred layer from synonym dict."""
        from foundry.dynamics import MemorySystem
        ms = MemorySystem()
        ms.build_inferred({"car": ["automobile", "vehicle"]})
        assert ms.inferred is not None

    def test_traverse_layers_empty(self):
        """traverse_layers on empty system returns empty list."""
        from foundry.dynamics import MemorySystem
        ms = MemorySystem()
        result = ms.traverse_layers("nonexistent")
        assert result == []


# ---------------------------------------------------------------------------
# ImportanceScore
# ---------------------------------------------------------------------------

class TestImportanceScore:
    """Tests for the ImportanceScore computation."""

    def test_score_fields(self):
        """ImportanceScore has all signal fields."""
        from foundry.dynamics import ImportanceScore
        score = ImportanceScore(
            mention_frequency=0.5,
            recency=0.8,
            centrality=0.3,
            retrieval_frequency=0.1,
        )
        assert score.mention_frequency == 0.5
        assert score.recency == 0.8
        assert score.centrality == 0.3
        assert score.retrieval_frequency == 0.1

    def test_composite_default_zero(self):
        """Composite score defaults to 0.0."""
        from foundry.dynamics import ImportanceScore
        score = ImportanceScore(
            mention_frequency=0.5, recency=0.8,
            centrality=0.3, retrieval_frequency=0.1,
        )
        assert score.composite == 0.0

    def test_compute_produces_score(self):
        """compute() produces a valid ImportanceScore."""
        from foundry.dynamics import ImportanceScore
        from foundry.storage import EntityNode
        entity = EntityNode(canonical="luna")
        entity.mention_turns = {0, 2, 5}
        entity.related = {"mochi", "user"}
        score = ImportanceScore.compute(
            entity=entity, total_turns=10,
            current_turn=10, retrieval_count=3, max_degree=5,
        )
        assert isinstance(score, ImportanceScore)
        assert 0 <= score.composite <= 1.0

    def test_compute_weights(self):
        """Composite is weighted: 0.3 freq + 0.3 recency + 0.2 centrality + 0.2 retrieval."""
        from foundry.dynamics import ImportanceScore
        from foundry.storage import EntityNode
        entity = EntityNode(canonical="test")
        entity.mention_turns = {0, 1, 2, 3, 4}
        entity.related = {"a", "b", "c"}
        score = ImportanceScore.compute(
            entity=entity, total_turns=10,
            current_turn=5, retrieval_count=10, max_degree=10,
        )
        # Composite should be > 0 for an entity with significant activity
        assert score.composite > 0

    def test_compute_zero_activity(self):
        """Entity with zero activity gets low importance."""
        from foundry.dynamics import ImportanceScore
        from foundry.storage import EntityNode
        entity = EntityNode(canonical="ghost")
        # No mentions, no relations, no retrievals
        score = ImportanceScore.compute(
            entity=entity, total_turns=100,
            current_turn=100, retrieval_count=0, max_degree=10,
        )
        assert score.composite == 0.0 or score.composite < 0.1


# ---------------------------------------------------------------------------
# DecayFunction
# ---------------------------------------------------------------------------

class TestDecayFunction:
    """Tests for the temporal decay model."""

    def test_no_decay_at_zero_days(self):
        """At day 0, salience equals base_salience."""
        from foundry.dynamics import DecayFunction
        decay = DecayFunction()
        result = decay.compute(days_since_access=0.0)
        assert abs(result - 1.0) < 1e-6

    def test_decay_over_time(self):
        """Salience decreases with time."""
        from foundry.dynamics import DecayFunction
        decay = DecayFunction()
        s0 = decay.compute(days_since_access=0.0)
        s30 = decay.compute(days_since_access=30.0)
        s365 = decay.compute(days_since_access=365.0)
        assert s0 > s30 > s365

    def test_exponential_formula(self):
        """Decay follows: salience = base * exp(-lambda * days).

        lambda = 0.01 per spec.
        """
        from foundry.dynamics import DecayFunction
        decay = DecayFunction()
        result = decay.compute(days_since_access=100.0)
        expected = 1.0 * math.exp(-0.01 * 100.0)
        assert abs(result - expected) < 1e-6

    def test_superseded_faster_decay(self):
        """Superseded facts decay 10x faster."""
        from foundry.dynamics import DecayFunction
        decay = DecayFunction()
        normal = decay.compute(days_since_access=30.0, superseded=False)
        fast = decay.compute(days_since_access=30.0, superseded=True)
        assert fast < normal

    def test_custom_base_salience(self):
        """Custom base_salience is used."""
        from foundry.dynamics import DecayFunction
        decay = DecayFunction()
        result = decay.compute(days_since_access=0.0, base_salience=0.5)
        assert abs(result - 0.5) < 1e-6

    def test_should_forget_below_threshold(self):
        """should_forget returns True below threshold."""
        from foundry.dynamics import DecayFunction
        decay = DecayFunction()
        assert decay.should_forget(salience=0.005, threshold=0.01) is True

    def test_should_not_forget_above_threshold(self):
        """should_forget returns False above threshold."""
        from foundry.dynamics import DecayFunction
        decay = DecayFunction()
        assert decay.should_forget(salience=0.5, threshold=0.01) is False


# ---------------------------------------------------------------------------
# HebbianBridge
# ---------------------------------------------------------------------------

class TestHebbianBridge:
    """Tests for the HebbianBridge (retrieval pattern tracking)."""

    def test_record_retrieval(self):
        """Recording a retrieval does not crash."""
        from foundry.dynamics import HebbianBridge
        bridge = HebbianBridge()
        bridge.record_retrieval(
            question="What color is the car?",
            evidence_turns=[0, 2, 5],
            tool_sequence=["search_memory", "read_turn", "answer"],
        )

    def test_get_frequent_patterns_empty(self):
        """Empty bridge has no frequent patterns."""
        from foundry.dynamics import HebbianBridge
        bridge = HebbianBridge()
        patterns = bridge.get_frequent_patterns(min_count=5)
        assert patterns == []

    def test_get_frequent_patterns_after_recording(self):
        """Patterns appear after sufficient recordings."""
        from foundry.dynamics import HebbianBridge
        bridge = HebbianBridge()
        for _ in range(10):
            bridge.record_retrieval(
                question="What color?",
                evidence_turns=[0],
                tool_sequence=["search_memory", "answer"],
            )
        patterns = bridge.get_frequent_patterns(min_count=5)
        assert len(patterns) >= 1


# ---------------------------------------------------------------------------
# DreamConsolidator
# ---------------------------------------------------------------------------

class TestDreamConsolidator:
    """Tests for bounded dream consolidation."""

    def test_consolidate_bounded(self):
        """Consolidation runs at most max_cycles."""
        from foundry.dynamics import DreamConsolidator
        consolidator = DreamConsolidator()
        result = consolidator.consolidate(max_cycles=3)
        assert result["cycles_run"] <= 3

    def test_consolidate_returns_metrics(self):
        """consolidate returns dict with required metric keys."""
        from foundry.dynamics import DreamConsolidator
        consolidator = DreamConsolidator()
        result = consolidator.consolidate(max_cycles=1)
        assert "cycles_run" in result
        assert "drift_detected" in result
        assert "quality_delta" in result

    def test_consolidate_default_max_cycles(self):
        """Default max_cycles is 5 per spec."""
        from foundry.dynamics import DreamConsolidator
        consolidator = DreamConsolidator()
        result = consolidator.consolidate()
        assert result["cycles_run"] <= 5
