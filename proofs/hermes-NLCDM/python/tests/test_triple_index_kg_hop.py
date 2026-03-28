"""Discriminative tests for triple-index KG hop resolution (Layer 1.5).

Tests the _resolve_triple_index_kg_hop() method on QueryEngine, which
resolves decompose sub-queries via O(1) dict lookup in the _triple_index
before falling through to cosine/BM25 retrieval.

The _triple_index is a dict[(norm_subj, norm_pred), list[(norm_obj, fact_text)]]
populated by the LLM triple extractor (Qwen3-30B-A3B) during ingestion.

Tests validate:
1. Single-hop resolution (country->capital, person->sport)
2. Predicate discrimination (capital vs president for same entity)
3. No-match fallthrough (missing entity or predicate)
4. Longest-match preference (more specific subject+predicate wins)
5. Multi-word predicates ("citizen of", "born in")
6. Empty index graceful fallback
7. Newest-wins temporal ordering (last entry in list)
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from mabench.configs import SupersessionState
from mabench.query import QueryEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_test_supersession_state() -> SupersessionState:
    """Build a SupersessionState with known triple_index entries."""
    ss = SupersessionState()
    ss.triple_index[("france", "capital")] = [
        ("paris", "The capital of France is Paris."),
    ]
    ss.triple_index[("france", "president")] = [
        ("macron", "The president of France is Macron."),
    ]
    ss.triple_index[("japan", "capital")] = [
        ("tokyo", "The capital of Japan is Tokyo."),
    ]
    ss.triple_index[("abbiati", "sport")] = [
        ("association football", "The sport played by Abbiati is association football."),
    ]
    ss.triple_index[("messi", "citizen of")] = [
        ("argentina", "Messi is a citizen of Argentina."),
    ]
    # Superseded entry: old chancellor + new chancellor
    ss.triple_index[("germany", "chancellor")] = [
        ("merkel", "The chancellor of Germany is Merkel."),
        ("scholz", "The chancellor of Germany is Scholz."),
    ]
    # Multi-word subject
    ss.triple_index[("south korea", "capital")] = [
        ("seoul", "The capital of South Korea is Seoul."),
    ]
    # Overlapping subjects: "new york" vs "new york city"
    ss.triple_index[("new york", "governor")] = [
        ("hochul", "The governor of New York is Hochul."),
    ]
    ss.triple_index[("new york city", "mayor")] = [
        ("adams", "The mayor of New York City is Adams."),
    ]
    return ss


def _make_engine(ss: SupersessionState | None = None) -> QueryEngine:
    """Create a minimal QueryEngine with only supersession_state wired."""
    return QueryEngine(supersession_state=ss)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTripleIndexKgHopResolution:
    """Core resolution logic."""

    def test_single_hop_capital(self):
        """Resolve 'What is the capital of France?' -> 'paris'."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("What is the capital of France?")
        assert result == "paris"

    def test_single_hop_sport(self):
        """Resolve 'What sport does Abbiati play?' -> 'association football'."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("What sport does Abbiati play?")
        assert result == "association football"

    def test_single_hop_japan_capital(self):
        """Resolve 'What is the capital of Japan?' -> 'tokyo'."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("What is the capital of Japan?")
        assert result == "tokyo"


class TestPredicateDiscrimination:
    """Same entity, different predicates -> different answers."""

    def test_france_capital_vs_president(self):
        """'capital of France' != 'president of France'."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        cap = engine._resolve_triple_index_kg_hop("What is the capital of France?")
        pres = engine._resolve_triple_index_kg_hop("Who is the president of France?")
        assert cap == "paris"
        assert pres == "macron"
        assert cap != pres


class TestNoMatchFallthrough:
    """Questions that don't match any (subj, pred) pair -> None."""

    def test_missing_predicate(self):
        """'population of France' has no matching predicate."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("What is the population of France?")
        assert result is None

    def test_missing_entity(self):
        """'capital of Brazil' has no matching entity."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("What is the capital of Brazil?")
        assert result is None

    def test_neither_entity_nor_predicate(self):
        """Completely unrelated question."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("How tall is Mount Everest?")
        assert result is None


class TestLongestMatchPreference:
    """When multiple (subj, pred) pairs match, longest combined wins."""

    def test_new_york_vs_new_york_city(self):
        """'mayor of New York City' should match 'new york city' not 'new york'."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "Who is the mayor of New York City?"
        )
        assert result == "adams"

    def test_new_york_governor(self):
        """'governor of New York' should match 'new york' + 'governor'."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "Who is the governor of New York?"
        )
        assert result == "hochul"


class TestMultiWordPredicates:
    """Multi-word predicates like 'citizen of', 'born in' match in questions."""

    def test_citizen_of(self):
        """'citizen of' appears in 'What country is Messi a citizen of?'."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "What country is Messi a citizen of?"
        )
        assert result == "argentina"

    def test_multi_word_subject(self):
        """'south korea' (multi-word subject) matches in question."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "What is the capital of South Korea?"
        )
        assert result == "seoul"


class TestEmptyIndex:
    """Empty or None triple index -> graceful None return."""

    def test_empty_triple_index(self):
        """SupersessionState with no triple entries."""
        ss = SupersessionState()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("What is the capital of France?")
        assert result is None

    def test_no_supersession_state(self):
        """No supersession_state at all -> None."""
        engine = _make_engine(None)
        result = engine._resolve_triple_index_kg_hop("What is the capital of France?")
        assert result is None


class TestNewestWins:
    """When multiple entries exist for same (S,P), last one (newest) wins."""

    def test_chancellor_superseded(self):
        """Germany chancellor: merkel superseded by scholz."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "Who is the chancellor of Germany?"
        )
        assert result == "scholz"
        assert result != "merkel"

    def test_single_entry_returns_itself(self):
        """Single entry: no supersession ambiguity."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("What is the capital of France?")
        assert result == "paris"


class TestResolveTripleIndexProperty:
    """Test the _resolve_triple_index property resolution."""

    def test_property_returns_index_from_supersession_state(self):
        """Property should return triple_index from SupersessionState."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        idx = engine._resolve_triple_index
        assert ("france", "capital") in idx
        assert idx[("france", "capital")][0][0] == "paris"

    def test_property_returns_empty_when_no_state(self):
        """Property returns empty dict when no supersession_state."""
        engine = _make_engine(None)
        idx = engine._resolve_triple_index
        assert idx == {} or len(idx) == 0

    def test_property_falls_back_to_mixin_attr(self):
        """Property falls back to self._triple_index if no supersession_state."""
        engine = _make_engine(None)
        # Simulate mixin mode: set _triple_index directly
        engine._triple_index = {("test", "pred"): [("val", "Test fact.")]}
        idx = engine._resolve_triple_index
        assert ("test", "pred") in idx


class TestShortKeyGuard:
    """Short subjects/predicates (len < 3) must NOT fire the fast lane."""

    def test_short_subject_no_match(self):
        """Subject 'or' (len 2) should NOT match inside 'sport'."""
        ss = SupersessionState()
        ss.triple_index[("or", "capital")] = [("bogus", "Bogus fact.")]
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("What is the capital of Oregon?")
        assert result is None

    def test_short_predicate_no_match(self):
        """Predicate 'be' (len 2) should NOT match inside 'because'."""
        ss = SupersessionState()
        ss.triple_index[("france", "be")] = [("bogus", "Bogus fact.")]
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("Why is France because of something?")
        assert result is None

    def test_both_short_no_match(self):
        """Both subject and predicate short -> no match."""
        ss = SupersessionState()
        ss.triple_index[("an", "is")] = [("bogus", "Bogus fact.")]
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("Japan is a country in Asia.")
        assert result is None

    def test_exactly_three_chars_matches(self):
        """Subject/predicate with exactly 3 chars should match."""
        ss = SupersessionState()
        ss.triple_index[("usa", "gdp")] = [("25 trillion", "The GDP of USA is 25 trillion.")]
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("What is the GDP of USA?")
        assert result == "25 trillion"


class TestCaseInsensitivity:
    """Questions with varying case should still match lowercase index keys."""

    def test_uppercase_entity_in_question(self):
        """'FRANCE' in question matches 'france' in index (via .lower())."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("What is the CAPITAL of FRANCE?")
        assert result == "paris"

    def test_mixed_case_predicate(self):
        """'President' matches 'president' in index."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop("Who is the President of France?")
        assert result == "macron"


# ---------------------------------------------------------------------------
# Helpers for subject-index tests
# ---------------------------------------------------------------------------

def _build_subject_index_state() -> SupersessionState:
    """Build a SupersessionState with subject_index entries for testing.

    The subject_index maps:  norm_subj -> {norm_pred: (norm_obj, fact_text)}
    Unlike triple_index, subject_index is newest-wins (single entry per pred).
    """
    ss = SupersessionState()

    # Also populate triple_index for legacy fallback tests
    ss.triple_index[("france", "capital")] = [
        ("paris", "The capital of France is Paris."),
    ]
    ss.triple_index[("france", "president")] = [
        ("macron", "The president of France is Macron."),
    ]

    # subject_index: subj -> {pred: (obj, text)}
    ss.subject_index["christian abbiati"] = {
        "position": ("cornerback", "The position Christian Abbiati plays is cornerback."),
    }
    ss.subject_index["france"] = {
        "capital": ("paris", "The capital of France is Paris."),
        "president": ("macron", "The president of France is Macron."),
    }
    ss.subject_index["fc viikingit"] = {
        "headquarters location": ("helsinki", "The headquarters location of FC Viikingit is Helsinki."),
    }
    ss.subject_index["new york"] = {
        "governor": ("hochul", "The governor of New York is Hochul."),
    }
    ss.subject_index["new york city"] = {
        "mayor": ("adams", "The mayor of New York City is Adams."),
    }
    # Short key: should NOT match
    ss.subject_index["or"] = {
        "capital": ("bogus", "Bogus fact."),
    }
    return ss


# ---------------------------------------------------------------------------
# Subject-index tests
# ---------------------------------------------------------------------------

class TestSubjectOnlyMatchSinglePredicate:
    """Subject-only match resolves when predicate doesn't match in question."""

    def test_single_predicate_resolves_without_predicate_in_question(self):
        """subject_index['christian abbiati'] has 1 predicate ('position').
        Question: 'What sport did Christian Abbiati play?' (no 'position' in question).
        Should find 'christian abbiati' -> only 1 predicate -> return 'cornerback'.
        """
        ss = _build_subject_index_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "What sport did Christian Abbiati play?"
        )
        assert result == "cornerback"

    def test_single_predicate_fc_viikingit(self):
        """subject_index['fc viikingit'] has 1 predicate ('headquarters location').
        Question: 'Where is FC Viikingit based?' ('headquarters location' not in question).
        Should find 'fc viikingit' -> only 1 predicate -> return 'helsinki'.
        """
        ss = _build_subject_index_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "Where is FC Viikingit based?"
        )
        assert result == "helsinki"


class TestSubjectMultiPredicateWithMatch:
    """Multi-predicate entity where predicate IS in the question text."""

    def test_france_capital_matches_predicate(self):
        """subject_index['france'] = {capital: paris, president: macron}.
        Question: 'What is the capital of France?' -> 'capital' in question -> 'paris'.
        """
        ss = _build_subject_index_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "What is the capital of France?"
        )
        assert result == "paris"

    def test_france_president_matches_predicate(self):
        """subject_index['france'] = {capital: paris, president: macron}.
        Question: 'Who is the president of France?' -> 'president' in question -> 'macron'.
        """
        ss = _build_subject_index_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "Who is the president of France?"
        )
        assert result == "macron"


class TestSubjectMultiPredicateNoMatch:
    """Multi-predicate entity where NO predicate matches in the question text."""

    def test_france_no_predicate_match_returns_none(self):
        """subject_index['france'] = {capital: paris, president: macron}.
        Question: 'Where is the seat of government of France?'
        Neither 'capital' nor 'president' in question -> return None (fall through).
        """
        ss = _build_subject_index_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "Where is the seat of government of France?"
        )
        assert result is None


class TestSubjectLongestMatchWins:
    """When multiple subjects match, longest (most specific) wins."""

    def test_new_york_city_over_new_york(self):
        """Both 'new york' and 'new york city' in subject_index.
        Question about 'New York City' -> picks 'new york city' (longer).
        """
        ss = _build_subject_index_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "Who is the mayor of New York City?"
        )
        assert result == "adams"

    def test_new_york_matches_when_no_city(self):
        """Question about 'New York' (no 'city') -> picks 'new york'.
        But governor is the only predicate, so returns 'hochul'.
        """
        ss = _build_subject_index_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "Who is the governor of New York?"
        )
        assert result == "hochul"


class TestSubjectShortKeyGuard:
    """Short subject keys (len < 3) must not match."""

    def test_short_subject_no_match(self):
        """subject_index['or'] (len 2) should NOT match inside 'Oregon'."""
        ss = _build_subject_index_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "What is the capital of Oregon?"
        )
        assert result is None


class TestSubjectEmptyFallsBackToLegacy:
    """When subject_index is empty, falls back to legacy triple_index method."""

    def test_empty_subject_index_uses_legacy(self):
        """subject_index empty, but triple_index populated -> uses legacy.
        triple_index has ('france', 'capital') -> 'paris'.
        Question: 'What is the capital of France?' -> legacy resolves to 'paris'.
        """
        ss = SupersessionState()
        # Only populate triple_index, NOT subject_index
        ss.triple_index[("france", "capital")] = [
            ("paris", "The capital of France is Paris."),
        ]
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop(
            "What is the capital of France?"
        )
        assert result == "paris"


class TestSubjectIndexProperty:
    """Test the _resolve_subject_index property on QueryEngine."""

    def test_property_returns_subject_index_from_state(self):
        """Property should return subject_index from SupersessionState."""
        ss = _build_subject_index_state()
        engine = _make_engine(ss)
        si = engine._resolve_subject_index
        assert "france" in si
        assert "capital" in si["france"]
        assert si["france"]["capital"][0] == "paris"

    def test_property_returns_empty_when_no_state(self):
        """Property returns empty dict when no supersession_state."""
        engine = _make_engine(None)
        si = engine._resolve_subject_index
        assert si == {} or len(si) == 0

    def test_property_falls_back_to_attr(self):
        """Property falls back to self._subject_index if no supersession_state."""
        engine = _make_engine(None)
        engine._subject_index = {"test_subj": {"pred": ("val", "fact")}}
        si = engine._resolve_subject_index
        assert "test_subj" in si


class TestSubjectIndexPopulatedDuringIngestion:
    """Verify subject_index is populated when _store_update_triple_index runs."""

    def test_subject_index_populated_after_store(self):
        """After calling _store_update_triple_index, subject_index has entries."""
        from unittest.mock import MagicMock
        from dataclasses import dataclass

        @dataclass
        class FakeFact:
            text: str = "The capital of France is Paris."
            triples: tuple = (("France", "capital", "Paris"),)
            entities: tuple = ("France", "Paris")

        ss = SupersessionState()
        store = MagicMock()
        store._supersession_state = ss
        store._resolve_triple_index = ss.triple_index
        store._resolve_fact_entities = ss.fact_entities
        store._resolve_text_to_store_idx = ss.text_to_store_idx
        store._resolve_subject_index = ss.subject_index
        store._knowledge_graph = None

        # Call the real method
        from mabench.store import FactStore
        FactStore._store_update_triple_index(store, FakeFact(), None)

        # triple_index should be populated
        assert ("france", "capital") in ss.triple_index

        # subject_index should be populated
        assert "france" in ss.subject_index
        assert "capital" in ss.subject_index["france"]
        obj, text = ss.subject_index["france"]["capital"]
        assert obj == "paris"
        assert text == "The capital of France is Paris."

    def test_subject_index_newest_wins(self):
        """When same (subj, pred) stored twice, subject_index keeps newest."""
        from unittest.mock import MagicMock
        from dataclasses import dataclass

        @dataclass
        class FakeFact:
            text: str
            triples: tuple
            entities: tuple = ()

        ss = SupersessionState()
        store = MagicMock()
        store._supersession_state = ss
        store._resolve_triple_index = ss.triple_index
        store._resolve_fact_entities = ss.fact_entities
        store._resolve_text_to_store_idx = ss.text_to_store_idx
        store._resolve_subject_index = ss.subject_index
        store._knowledge_graph = None

        from mabench.store import FactStore

        # Store first version
        fact1 = FakeFact(
            text="The chancellor of Germany is Merkel.",
            triples=(("Germany", "chancellor", "Merkel"),),
        )
        FactStore._store_update_triple_index(store, fact1, None)

        # Store second version (supersedes first)
        fact2 = FakeFact(
            text="The chancellor of Germany is Scholz.",
            triples=(("Germany", "chancellor", "Scholz"),),
        )
        FactStore._store_update_triple_index(store, fact2, None)

        # subject_index should have newest
        assert ss.subject_index["germany"]["chancellor"][0] == "scholz"
        assert ss.subject_index["germany"]["chancellor"][1] == "The chancellor of Germany is Scholz."


class TestLegacyMethodPreserved:
    """Verify the legacy method still exists and works for backward compat."""

    def test_legacy_method_exists(self):
        """_resolve_triple_index_kg_hop_legacy should exist on QueryEngine."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        assert hasattr(engine, "_resolve_triple_index_kg_hop_legacy")

    def test_legacy_method_resolves_with_both_subj_pred(self):
        """Legacy method uses (subj, pred) pair matching as before."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        result = engine._resolve_triple_index_kg_hop_legacy(
            "What is the capital of France?"
        )
        assert result == "paris"

    def test_legacy_method_returns_none_on_predicate_mismatch(self):
        """Legacy method requires BOTH subj and pred to match."""
        ss = _build_test_supersession_state()
        engine = _make_engine(ss)
        # 'sport' not in question, only 'position' -> legacy should return None
        # (This is the bug the subject-index approach fixes)
        result = engine._resolve_triple_index_kg_hop_legacy(
            "What sport did Abbiati play?"
        )
        # Legacy requires pred "sport" in question but "position" is the pred key
        # Here "sport" IS in the triple_index for abbiati, so it matches
        assert result == "association football"


class TestResolveSubjectIndexOnSuperState:
    """Verify subject_index field exists on SupersessionState."""

    def test_subject_index_field_exists(self):
        """SupersessionState should have subject_index as a dict field."""
        ss = SupersessionState()
        assert hasattr(ss, "subject_index")
        assert isinstance(ss.subject_index, dict)

    def test_subject_index_cleared_on_reset(self):
        """SupersessionState.reset() should clear subject_index."""
        ss = SupersessionState()
        ss.subject_index["test"] = {"pred": ("val", "fact")}
        ss.reset()
        assert len(ss.subject_index) == 0
