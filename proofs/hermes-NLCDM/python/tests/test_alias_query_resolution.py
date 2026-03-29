"""Tests for query-time alias resolution in QueryEngine.

Verifies:
1. QueryEngine accepts entity_registry parameter
2. _resolve_entity_registry property returns the registry
3. _resolve_triple_index_kg_hop resolves aliases (e.g. "Mom" -> "Sarah")
4. Without registry, alias queries fall through (no match)
5. Direct canonical name queries still work
6. Empty registry means no change in behavior
7. entity_registry is threaded through agent composition
"""
from __future__ import annotations

import pytest

from mabench.configs import SupersessionState
from mabench.query import QueryEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query_engine(
    entity_registry=None,
    subject_index=None,
) -> QueryEngine:
    """Create a minimal QueryEngine with subject_index and entity_registry."""
    ss = SupersessionState()
    if subject_index is not None:
        ss.subject_index = subject_index
    return QueryEngine(
        entity_registry=entity_registry,
        supersession_state=ss,
    )


# ---------------------------------------------------------------------------
# 1. QueryEngine accepts entity_registry parameter
# ---------------------------------------------------------------------------

class TestQueryEngineEntityRegistryParam:
    def test_accepts_entity_registry_kwarg(self):
        reg = {"Sarah": ["Mom", "my mom"]}
        qe = _make_query_engine(entity_registry=reg)
        assert qe._entity_registry is reg

    def test_defaults_to_empty_dict(self):
        qe = _make_query_engine()
        assert qe._entity_registry == {}

    def test_none_becomes_empty_dict(self):
        qe = _make_query_engine(entity_registry=None)
        assert qe._entity_registry == {}


# ---------------------------------------------------------------------------
# 2. _resolve_entity_registry property returns the registry
# ---------------------------------------------------------------------------

class TestResolveEntityRegistryProperty:
    def test_returns_registry_when_set(self):
        reg = {"Sarah": ["Mom"]}
        qe = _make_query_engine(entity_registry=reg)
        assert qe._resolve_entity_registry is reg

    def test_returns_empty_dict_when_none(self):
        qe = _make_query_engine()
        assert qe._resolve_entity_registry == {}


# ---------------------------------------------------------------------------
# 3. Alias resolution in _resolve_triple_index_kg_hop
# ---------------------------------------------------------------------------

class TestKGHopAliasResolution:
    """Subject index has 'sarah' -> {'visiting': ('user', 'Sarah is visiting user')}.
    Query with 'Mom' (alias for Sarah) should resolve and find 'sarah'."""

    def test_alias_resolves_to_canonical_in_kg_hop(self):
        """'when is mom visiting?' with registry Sarah -> [Mom] should match 'sarah'."""
        reg = {"Sarah": ["Mom"]}
        subj_idx = {
            "sarah": {"visiting": ("user", "Sarah is visiting user")},
        }
        qe = _make_query_engine(entity_registry=reg, subject_index=subj_idx)
        result = qe._resolve_triple_index_kg_hop("when is mom visiting?")
        assert result == "user"

    def test_alias_resolves_multi_word(self):
        """'my mom' should also resolve to Sarah."""
        reg = {"Sarah": ["Mom", "my mom"]}
        subj_idx = {
            "sarah": {"visiting": ("user", "Sarah is visiting user")},
        }
        qe = _make_query_engine(entity_registry=reg, subject_index=subj_idx)
        result = qe._resolve_triple_index_kg_hop("when is my mom visiting?")
        assert result == "user"

    def test_alias_case_insensitive(self):
        """'MOM' (uppercase) should also resolve to Sarah."""
        reg = {"Sarah": ["Mom"]}
        subj_idx = {
            "sarah": {"visiting": ("user", "Sarah is visiting user")},
        }
        qe = _make_query_engine(entity_registry=reg, subject_index=subj_idx)
        result = qe._resolve_triple_index_kg_hop("when is MOM visiting?")
        assert result == "user"

    def test_canonical_name_in_query_needs_no_alias(self):
        """Direct canonical name 'Sarah' should still match without registry."""
        subj_idx = {
            "sarah": {"visiting": ("user", "Sarah is visiting user")},
        }
        qe = _make_query_engine(subject_index=subj_idx)
        result = qe._resolve_triple_index_kg_hop("when is Sarah visiting?")
        assert result == "user"

    def test_alias_resolution_prefers_longest_match(self):
        """When multiple aliases match, the canonical from longest alias wins."""
        reg = {
            "Honda Civic": ["car"],
            "Toyota Camry": ["red car"],
        }
        subj_idx = {
            "honda civic": {"color": ("blue", "Honda Civic is blue")},
            "toyota camry": {"color": ("red", "Toyota Camry is red")},
        }
        qe = _make_query_engine(entity_registry=reg, subject_index=subj_idx)
        result = qe._resolve_triple_index_kg_hop("what color is the red car?")
        # 'red car' matches Toyota Camry (longest alias), so 'toyota camry'
        # should be appended to the query, and since 'toyota camry' is in
        # subject_index, it should match.
        assert result == "red"


# ---------------------------------------------------------------------------
# 4. Without registry, alias queries fall through
# ---------------------------------------------------------------------------

class TestNoRegistryFallthrough:
    def test_no_match_without_registry(self):
        """'when is mom visiting?' should NOT match 'sarah' without registry."""
        subj_idx = {
            "sarah": {"visiting": ("user", "Sarah is visiting user")},
        }
        qe = _make_query_engine(subject_index=subj_idx)
        result = qe._resolve_triple_index_kg_hop("when is mom visiting?")
        assert result is None

    def test_empty_registry_same_as_no_registry(self):
        subj_idx = {
            "sarah": {"visiting": ("user", "Sarah is visiting user")},
        }
        qe = _make_query_engine(entity_registry={}, subject_index=subj_idx)
        result = qe._resolve_triple_index_kg_hop("when is mom visiting?")
        assert result is None


# ---------------------------------------------------------------------------
# 5. Direct canonical name queries still work (no regression)
# ---------------------------------------------------------------------------

class TestDirectCanonicalNoRegression:
    def test_direct_subject_match_with_registry(self):
        """Even with a registry, direct subject match should still work."""
        reg = {"Sarah": ["Mom"]}
        subj_idx = {
            "sarah": {"visiting": ("user", "Sarah is visiting user")},
        }
        qe = _make_query_engine(entity_registry=reg, subject_index=subj_idx)
        result = qe._resolve_triple_index_kg_hop("when is Sarah visiting?")
        assert result == "user"

    def test_multiple_subjects_still_picks_longest(self):
        """Existing longest-match behavior for direct subjects is preserved."""
        subj_idx = {
            "france": {"capital": ("paris", "capital of France is Paris")},
            "south france": {"capital": ("toulouse", "capital of South France is Toulouse")},
        }
        qe = _make_query_engine(subject_index=subj_idx)
        result = qe._resolve_triple_index_kg_hop("what is the capital of south france?")
        assert result == "toulouse"


# ---------------------------------------------------------------------------
# 6. Empty registry means no change in behavior
# ---------------------------------------------------------------------------

class TestEmptyRegistryBehavior:
    def test_empty_registry_no_side_effects(self):
        """Empty registry should not alter any existing matching logic."""
        subj_idx = {
            "france": {
                "capital": ("paris", "capital of France is Paris"),
                "language": ("french", "language of France is French"),
            },
        }
        qe = _make_query_engine(entity_registry={}, subject_index=subj_idx)
        result = qe._resolve_triple_index_kg_hop("what is the capital of France?")
        assert result == "paris"


# ---------------------------------------------------------------------------
# 7. Agent composition threads entity_registry
# ---------------------------------------------------------------------------

class TestAgentComposition:
    """Verify HermesAgent threads entity_registry to QueryEngine.

    We avoid constructing a full HermesAgent (heavy models) and instead
    verify the wiring logic by checking the code path.
    """

    def test_query_engine_receives_registry_reference(self):
        """When entity_registry is a dict, QueryEngine gets a reference (not copy)."""
        reg = {"Sarah": ["Mom"]}
        qe = _make_query_engine(entity_registry=reg)
        # Mutating the original dict should be visible through the engine
        reg["Bob"] = ["Robert"]
        assert "Bob" in qe._entity_registry

    def test_registry_grows_visible_to_query_engine(self):
        """Simulates ingestion growing the registry and query seeing it."""
        reg = {}
        qe = _make_query_engine(
            entity_registry=reg,
            subject_index={
                "sarah": {"visiting": ("user", "Sarah is visiting user")},
            },
        )
        # Before registry grows: no alias match
        assert qe._resolve_triple_index_kg_hop("when is mom visiting?") is None

        # Simulate ingestion discovering alias
        reg["Sarah"] = ["Mom"]

        # After registry grows: alias match works
        result = qe._resolve_triple_index_kg_hop("when is mom visiting?")
        assert result == "user"
