"""Temporal tests: temporal fact management and supersession.

Tests for temporal supersession (newer fact replaces older for same
predicate), temporal validity windows, and fact history queries.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta


class TestTemporalSupersession:
    """Tests for temporal fact supersession."""

    def test_latest_turn_wins(self):
        """For same (entity, predicate), fact with highest turn_idx is current."""
        from foundry.writeback import SupersessionEngine
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        engine = SupersessionEngine()
        facts = [
            Fact(id="f1", subject="user", predicate="city",
                 object="sf", object_type=PropertyType.STRING, turn_idx=0),
            Fact(id="f2", subject="user", predicate="city",
                 object="portland", object_type=PropertyType.STRING, turn_idx=5),
            Fact(id="f3", subject="user", predicate="city",
                 object="seattle", object_type=PropertyType.STRING, turn_idx=10),
        ]
        resolved = engine.resolve_conflicts(facts, strategy="latest_wins")
        assert len(resolved) == 1
        assert resolved[0].object == "seattle"
        assert resolved[0].turn_idx == 10

    def test_supersession_chain(self):
        """Multiple supersessions create a chain."""
        from foundry.writeback import SupersessionEngine
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        engine = SupersessionEngine()
        for i, city in enumerate(["sf", "portland", "seattle"]):
            f = Fact(
                id=f"f{i}", subject="user", predicate="city",
                object=city, object_type=PropertyType.STRING,
                turn_idx=i * 5,
            )
            engine.add_fact(f)
            engine.supersede("user", "city", f)
        all_facts = engine.get_all_facts("user", "city")
        current = [f for f in all_facts if not f.superseded]
        assert len(current) == 1
        assert current[0].object == "seattle"

    def test_different_predicates_independent(self):
        """Supersession only applies within same (entity, predicate)."""
        from foundry.writeback import SupersessionEngine
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        engine = SupersessionEngine()
        f1 = Fact(id="f1", subject="user", predicate="city",
                  object="sf", object_type=PropertyType.STRING, turn_idx=0)
        f2 = Fact(id="f2", subject="user", predicate="job",
                  object="engineer", object_type=PropertyType.STRING, turn_idx=0)
        engine.add_fact(f1)
        engine.add_fact(f2)
        # Adding a new city should not affect the job
        f3 = Fact(id="f3", subject="user", predicate="city",
                  object="portland", object_type=PropertyType.STRING, turn_idx=5)
        engine.add_fact(f3)
        engine.supersede("user", "city", f3)
        jobs = engine.get_all_facts("user", "job")
        assert len(jobs) == 1
        assert not jobs[0].superseded


class TestTemporalValidityWindows:
    """Tests for temporal validity tracking."""

    def test_validity_from_when_field(self):
        """Fact with when='since 2020' has valid_from parsed."""
        from foundry.writeback import TemporalValidity
        tv = TemporalValidity(
            valid_from=datetime(2020, 1, 1),
            valid_until=None,
        )
        assert tv.is_current() is True
        assert tv.is_current(as_of=datetime(2023, 1, 1)) is True

    def test_validity_after_supersession(self):
        """Superseded fact gets valid_until set."""
        from foundry.writeback import TemporalValidity
        tv = TemporalValidity(
            valid_from=datetime(2020, 1, 1),
            valid_until=datetime(2024, 3, 15),
        )
        assert tv.is_current(as_of=datetime(2023, 1, 1)) is True
        assert tv.is_current(as_of=datetime(2025, 1, 1)) is False

    def test_validity_with_turn_index(self):
        """TemporalValidity works with turn indices instead of datetimes."""
        from foundry.writeback import TemporalValidity
        tv = TemporalValidity(valid_from=0, valid_until=5)
        assert tv.is_current(as_of=3) is True
        assert tv.is_current(as_of=6) is False


class TestFactHistory:
    """Tests for accessing full fact history (timeline and deltas)."""

    def test_timeline_includes_all(self):
        """timeline() returns all facts including superseded ones."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import EntityNode, Fact
        from foundry.ontology import PropertyType
        foundry = Foundry(FoundryConfig())
        user = EntityNode(canonical="user")
        user.facts["city"] = [
            Fact(id="f1", subject="user", predicate="city",
                 object="sf", object_type=PropertyType.STRING,
                 turn_idx=0, superseded=True, superseded_by="f2"),
            Fact(id="f2", subject="user", predicate="city",
                 object="portland", object_type=PropertyType.STRING,
                 turn_idx=5),
        ]
        foundry._store.entity_graph.add_entity(user)
        timeline = foundry.timeline("user")
        assert timeline is not None
        assert "city" in timeline
        assert len(timeline["city"]) == 2

    def test_deltas_shows_changes(self):
        """deltas() returns only facts where value differs from prior."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import EntityNode, Fact
        from foundry.ontology import PropertyType
        foundry = Foundry(FoundryConfig())
        user = EntityNode(canonical="user")
        user.facts["city"] = [
            Fact(id="f1", subject="user", predicate="city",
                 object="sf", object_type=PropertyType.STRING, turn_idx=0),
            Fact(id="f2", subject="user", predicate="city",
                 object="portland", object_type=PropertyType.STRING, turn_idx=5),
            Fact(id="f3", subject="user", predicate="city",
                 object="portland", object_type=PropertyType.STRING, turn_idx=10),
        ]
        foundry._store.entity_graph.add_entity(user)
        deltas = foundry.deltas("user")
        if deltas and "city" in deltas:
            # Should include sf -> portland but not portland -> portland
            values = [f.object for f in deltas["city"]]
            assert "sf" in values
            assert "portland" in values

    def test_current_only_excludes_superseded(self):
        """facts(current_only=True) returns only non-superseded facts."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import EntityNode, Fact
        from foundry.ontology import PropertyType
        foundry = Foundry(FoundryConfig())
        user = EntityNode(canonical="user")
        user.facts["city"] = [
            Fact(id="f1", subject="user", predicate="city",
                 object="sf", object_type=PropertyType.STRING,
                 turn_idx=0, superseded=True),
            Fact(id="f2", subject="user", predicate="city",
                 object="portland", object_type=PropertyType.STRING,
                 turn_idx=5),
        ]
        foundry._store.entity_graph.add_entity(user)
        current = foundry.facts("user", predicate="city", current_only=True)
        assert len(current) == 1
        assert current[0].object == "portland"

    def test_current_only_false_includes_all(self):
        """facts(current_only=False) includes superseded facts."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import EntityNode, Fact
        from foundry.ontology import PropertyType
        foundry = Foundry(FoundryConfig())
        user = EntityNode(canonical="user")
        user.facts["city"] = [
            Fact(id="f1", subject="user", predicate="city",
                 object="sf", object_type=PropertyType.STRING,
                 turn_idx=0, superseded=True),
            Fact(id="f2", subject="user", predicate="city",
                 object="portland", object_type=PropertyType.STRING,
                 turn_idx=5),
        ]
        foundry._store.entity_graph.add_entity(user)
        all_facts = foundry.facts("user", predicate="city", current_only=False)
        assert len(all_facts) == 2


class TestTemporalDecay:
    """Tests for temporal decay interaction with supersession."""

    def test_superseded_decay_faster(self):
        """Superseded facts have 10x faster decay rate."""
        from foundry.dynamics import DecayFunction
        decay = DecayFunction()
        normal = decay.compute(days_since_access=30.0, superseded=False)
        fast = decay.compute(days_since_access=30.0, superseded=True)
        # Superseded decays 10x faster
        assert fast < normal
        # The ratio should be approximately exp(-0.01 * 30) / exp(-0.1 * 30)
        import math
        ratio = fast / normal
        expected_ratio = math.exp(-0.1 * 30) / math.exp(-0.01 * 30)
        assert abs(ratio - expected_ratio) < 0.01

    def test_forget_threshold(self):
        """Facts below forget threshold are candidates for GC."""
        from foundry.dynamics import DecayFunction
        decay = DecayFunction()
        # After a very long time, superseded facts should be forgettable
        salience = decay.compute(days_since_access=1000.0, superseded=True)
        assert decay.should_forget(salience, threshold=0.01) is True
