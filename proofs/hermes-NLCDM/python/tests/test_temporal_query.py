"""Adversarial tests for query-time temporal resolution in QueryEngine.

Verifies:
1. _detect_temporal_intent: regex classifies when/duration/latest/None
2. _resolve_temporal_query: deterministic lookup via subject_index + event_time
3. _query short-circuits on temporal queries (no LLM call)
4. Edge cases: no entity match, no event_time, empty store
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import math

import numpy as np
import pytest

from mabench.query import QueryEngine
from mabench.configs import SupersessionState


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@dataclass
class MockMemoryEntry:
    """Minimal MemoryEntry stub with text and event_time."""
    text: str
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(4))
    recency: float = 0.0
    event_time: object = None  # datetime | None


@dataclass
class MockEngine:
    """Minimal CoupledEngine stub with memory_store list."""
    memory_store: list = field(default_factory=list)


def _make_query_engine(
    subject_index: dict,
    text_to_store_idx: dict,
    memory_store: list,
) -> QueryEngine:
    """Build a QueryEngine with mocked supersession state + engine."""
    ss = SupersessionState()
    ss.subject_index = subject_index
    ss.text_to_store_idx = text_to_store_idx

    engine = MockEngine(memory_store=memory_store)

    qe = QueryEngine(
        engine=engine,
        supersession_state=ss,
    )
    return qe


# ---------------------------------------------------------------------------
# _detect_temporal_intent
# ---------------------------------------------------------------------------

class TestDetectTemporalIntent:
    """Regex-based intent classification: when / duration / latest / None."""

    def test_when_did(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("When did Sarah graduate?") == "when"

    def test_when_was(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("When was the last time Mom called?") == "when"

    def test_when_is(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("When is the next meeting?") == "when"

    def test_what_date(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("What date did the project start?") == "when"

    def test_on_what_day(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("On what day did we meet?") == "when"

    def test_duration_how_long_since(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("How long since we adopted Bobo?") == "duration"

    def test_duration_how_many_days_since(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("How many days since the party?") == "duration"

    def test_duration_how_long_ago(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("How long ago did we move?") == "duration"

    def test_duration_how_many_years(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("How many years has Sarah been working?") == "duration"

    def test_duration_how_many_months(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("How many months since we started?") == "duration"

    def test_latest_current(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("What is Sarah's current job?") == "latest"

    def test_latest_most_recent(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("What is the most recent update?") == "latest"

    def test_latest_now(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("Where does Sarah live now?") == "latest"

    def test_latest_right_now(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("What is Sarah doing right now?") == "latest"

    def test_latest_these_days(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("What is Sarah reading these days?") == "latest"

    def test_none_factual_question(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("What sport does Abbiati play?") is None

    def test_none_location_question(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("Where did Sarah go to school?") is None

    def test_none_who_question(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("Who is Sarah's best friend?") is None

    def test_case_insensitive(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("WHEN DID SARAH GRADUATE?") == "when"

    def test_empty_string(self):
        qe = QueryEngine()
        assert qe._detect_temporal_intent("") is None


# ---------------------------------------------------------------------------
# _resolve_temporal_query — "when" intent
# ---------------------------------------------------------------------------

class TestResolveTemporalWhen:
    """Tests for 'when' intent: returns formatted event_time."""

    def test_when_returns_iso_date(self):
        """Known entity + predicate with event_time -> ISO date string."""
        store = [
            MockMemoryEntry(
                text="Sarah graduated from MIT",
                event_time=datetime(2019, 6, 1),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"graduated": ("MIT", "Sarah graduated from MIT")},
            },
            text_to_store_idx={"Sarah graduated from MIT": 0},
            memory_store=store,
        )
        result = qe._resolve_temporal_query("When did Sarah graduate?", "when")
        assert result is not None
        assert "2019-06-01" in result["output"]

    def test_when_no_event_time_returns_none(self):
        """Entity found but event_time is None -> fall through."""
        store = [
            MockMemoryEntry(
                text="Sarah graduated from MIT",
                event_time=None,
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"graduated": ("MIT", "Sarah graduated from MIT")},
            },
            text_to_store_idx={"Sarah graduated from MIT": 0},
            memory_store=store,
        )
        result = qe._resolve_temporal_query("When did Sarah graduate?", "when")
        assert result is None

    def test_when_no_entity_match_returns_none(self):
        """No entity in question matches subject_index -> None."""
        store = [
            MockMemoryEntry(
                text="Bob works at Google",
                event_time=datetime(2020, 1, 15),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "bob": {"works at": ("Google", "Bob works at Google")},
            },
            text_to_store_idx={"Bob works at Google": 0},
            memory_store=store,
        )
        # "Sarah" not in subject_index
        result = qe._resolve_temporal_query("When did Sarah graduate?", "when")
        assert result is None

    def test_when_predicate_matching(self):
        """Multiple predicates for same entity; matches best by substring."""
        store = [
            MockMemoryEntry(
                text="Sarah graduated from MIT",
                event_time=datetime(2019, 6, 1),
            ),
            MockMemoryEntry(
                text="Sarah started at Google",
                event_time=datetime(2020, 9, 15),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {
                    "graduated": ("MIT", "Sarah graduated from MIT"),
                    "started at": ("Google", "Sarah started at Google"),
                },
            },
            text_to_store_idx={
                "Sarah graduated from MIT": 0,
                "Sarah started at Google": 1,
            },
            memory_store=store,
        )
        result = qe._resolve_temporal_query("When did Sarah graduate?", "when")
        assert result is not None
        assert "2019-06-01" in result["output"]

    def test_when_short_entity_skipped(self):
        """Entity name shorter than 3 chars should not match (guard)."""
        store = [
            MockMemoryEntry(
                text="AI advanced rapidly",
                event_time=datetime(2023, 1, 1),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "ai": {"advanced": ("rapidly", "AI advanced rapidly")},
            },
            text_to_store_idx={"AI advanced rapidly": 0},
            memory_store=store,
        )
        result = qe._resolve_temporal_query("When did AI advance?", "when")
        assert result is None


# ---------------------------------------------------------------------------
# _resolve_temporal_query — "duration" intent
# ---------------------------------------------------------------------------

class TestResolveTemporalDuration:
    """Tests for 'duration' intent: returns human-readable timedelta."""

    def test_duration_days(self):
        """Recent event -> duration in days."""
        ref = datetime(2026, 3, 29)
        event = datetime(2026, 3, 8)  # 21 days ago
        store = [
            MockMemoryEntry(
                text="We adopted Bobo",
                event_time=event,
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "bobo": {"adopted": ("us", "We adopted Bobo")},
            },
            text_to_store_idx={"We adopted Bobo": 0},
            memory_store=store,
        )
        result = qe._resolve_temporal_query(
            "How long since we adopted Bobo?", "duration",
            reference_date=ref,
        )
        assert result is not None
        assert "21" in result["output"]
        assert "day" in result["output"].lower()

    def test_duration_months(self):
        """Event ~3 months ago -> mentions months."""
        ref = datetime(2026, 3, 29)
        event = datetime(2025, 12, 29)  # ~91 days = ~3 months
        store = [
            MockMemoryEntry(
                text="Sarah moved to Boston",
                event_time=event,
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"moved to": ("Boston", "Sarah moved to Boston")},
            },
            text_to_store_idx={"Sarah moved to Boston": 0},
            memory_store=store,
        )
        result = qe._resolve_temporal_query(
            "How long since Sarah moved?", "duration",
            reference_date=ref,
        )
        assert result is not None
        # Should mention months (approx 3)
        assert "3" in result["output"]
        assert "month" in result["output"].lower()

    def test_duration_years(self):
        """Event years ago -> mentions years."""
        ref = datetime(2026, 3, 29)
        event = datetime(2019, 6, 1)  # ~7 years
        store = [
            MockMemoryEntry(
                text="Sarah graduated from MIT",
                event_time=event,
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"graduated": ("MIT", "Sarah graduated from MIT")},
            },
            text_to_store_idx={"Sarah graduated from MIT": 0},
            memory_store=store,
        )
        result = qe._resolve_temporal_query(
            "How long since Sarah graduated?", "duration",
            reference_date=ref,
        )
        assert result is not None
        # ~6 years, 9 months -> should mention years
        assert "year" in result["output"].lower()

    def test_duration_no_event_time_returns_none(self):
        """Entity found but no event_time -> None."""
        store = [
            MockMemoryEntry(
                text="We adopted Bobo",
                event_time=None,
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "bobo": {"adopted": ("us", "We adopted Bobo")},
            },
            text_to_store_idx={"We adopted Bobo": 0},
            memory_store=store,
        )
        result = qe._resolve_temporal_query(
            "How long since we adopted Bobo?", "duration",
        )
        assert result is None


# ---------------------------------------------------------------------------
# _resolve_temporal_query — "latest" intent
# ---------------------------------------------------------------------------

class TestResolveTemporalLatest:
    """Tests for 'latest' intent: returns the most recent fact by event_time."""

    def test_latest_returns_most_recent(self):
        """Multiple facts with event_times -> returns the one with latest event_time."""
        store = [
            MockMemoryEntry(
                text="Sarah worked at Google",
                event_time=datetime(2020, 1, 1),
            ),
            MockMemoryEntry(
                text="Sarah worked at Meta",
                event_time=datetime(2023, 6, 15),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {
                    "worked at_old": ("Google", "Sarah worked at Google"),
                    "worked at": ("Meta", "Sarah worked at Meta"),
                },
            },
            text_to_store_idx={
                "Sarah worked at Google": 0,
                "Sarah worked at Meta": 1,
            },
            memory_store=store,
        )
        result = qe._resolve_temporal_query(
            "What is Sarah's current job?", "latest",
        )
        assert result is not None
        assert "Meta" in result["output"]

    def test_latest_single_fact(self):
        """Single fact with event_time -> returns it."""
        store = [
            MockMemoryEntry(
                text="Sarah lives in Boston",
                event_time=datetime(2024, 3, 1),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"lives in": ("Boston", "Sarah lives in Boston")},
            },
            text_to_store_idx={"Sarah lives in Boston": 0},
            memory_store=store,
        )
        result = qe._resolve_temporal_query(
            "Where does Sarah live now?", "latest",
        )
        assert result is not None
        assert "Boston" in result["output"]

    def test_latest_no_event_times_returns_none(self):
        """All facts for entity have event_time=None -> None (fall through)."""
        store = [
            MockMemoryEntry(
                text="Sarah likes coffee",
                event_time=None,
            ),
            MockMemoryEntry(
                text="Sarah likes tea",
                event_time=None,
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {
                    "likes_1": ("coffee", "Sarah likes coffee"),
                    "likes_2": ("tea", "Sarah likes tea"),
                },
            },
            text_to_store_idx={
                "Sarah likes coffee": 0,
                "Sarah likes tea": 1,
            },
            memory_store=store,
        )
        result = qe._resolve_temporal_query(
            "What does Sarah like these days?", "latest",
        )
        assert result is None

    def test_latest_mixed_event_times(self):
        """Some facts with event_time, some without -> only considers those with event_time."""
        store = [
            MockMemoryEntry(
                text="Sarah likes coffee",
                event_time=None,
            ),
            MockMemoryEntry(
                text="Sarah started running",
                event_time=datetime(2025, 1, 10),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {
                    "likes": ("coffee", "Sarah likes coffee"),
                    "started": ("running", "Sarah started running"),
                },
            },
            text_to_store_idx={
                "Sarah likes coffee": 0,
                "Sarah started running": 1,
            },
            memory_store=store,
        )
        result = qe._resolve_temporal_query(
            "What is Sarah doing these days?", "latest",
        )
        assert result is not None
        assert "running" in result["output"]


# ---------------------------------------------------------------------------
# _resolve_temporal_query — empty / edge cases
# ---------------------------------------------------------------------------

class TestResolveTemporalEdgeCases:
    """Edge cases: empty store, missing text_to_store_idx, etc."""

    def test_empty_subject_index(self):
        """No subjects at all -> None."""
        qe = _make_query_engine(
            subject_index={},
            text_to_store_idx={},
            memory_store=[],
        )
        result = qe._resolve_temporal_query("When did Sarah graduate?", "when")
        assert result is None

    def test_text_not_in_store_idx(self):
        """Subject found but fact text missing from text_to_store_idx -> skip gracefully."""
        store = [
            MockMemoryEntry(
                text="Sarah graduated from MIT",
                event_time=datetime(2019, 6, 1),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"graduated": ("MIT", "Sarah graduated from MIT")},
            },
            text_to_store_idx={},  # Deliberately empty
            memory_store=store,
        )
        result = qe._resolve_temporal_query("When did Sarah graduate?", "when")
        assert result is None

    def test_result_dict_format(self):
        """Verify the returned dict has the expected keys."""
        store = [
            MockMemoryEntry(
                text="Sarah graduated from MIT",
                event_time=datetime(2019, 6, 1),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"graduated": ("MIT", "Sarah graduated from MIT")},
            },
            text_to_store_idx={"Sarah graduated from MIT": 0},
            memory_store=store,
        )
        result = qe._resolve_temporal_query("When did Sarah graduate?", "when")
        assert result is not None
        assert "output" in result
        assert "input_len" in result
        assert "output_len" in result

    def test_longest_entity_match(self):
        """When multiple entities match, pick the longest."""
        store = [
            MockMemoryEntry(
                text="Sarah Smith graduated from MIT",
                event_time=datetime(2019, 6, 1),
            ),
            MockMemoryEntry(
                text="Sarah joined the team",
                event_time=datetime(2022, 3, 1),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"joined": ("team", "Sarah joined the team")},
                "sarah smith": {"graduated": ("MIT", "Sarah Smith graduated from MIT")},
            },
            text_to_store_idx={
                "Sarah Smith graduated from MIT": 0,
                "Sarah joined the team": 1,
            },
            memory_store=store,
        )
        result = qe._resolve_temporal_query(
            "When did Sarah Smith graduate?", "when",
        )
        assert result is not None
        assert "2019-06-01" in result["output"]


# ---------------------------------------------------------------------------
# Integration: _query short-circuits on temporal query
# ---------------------------------------------------------------------------

class TestQueryTemporalShortCircuit:
    """Verify that _query returns early for temporal queries without LLM calls."""

    def test_query_returns_temporal_result(self):
        """_query on a 'when' question should short-circuit via Phase 0."""
        store = [
            MockMemoryEntry(
                text="Sarah graduated from MIT",
                event_time=datetime(2019, 6, 1),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"graduated": ("MIT", "Sarah graduated from MIT")},
            },
            text_to_store_idx={"Sarah graduated from MIT": 0},
            memory_store=store,
        )
        # _query expects the full MABench prompt format
        prompt = (
            "Pretend you are a knowledge management system. "
            "Question: When did Sarah graduate?\nAnswer:"
        )
        result = qe._query(prompt)
        assert result is not None
        assert "2019-06-01" in result["output"]
        # Verify timing fields
        assert "memory_construction_time" in result
        assert "query_time_len" in result

    def test_query_falls_through_when_no_temporal_data(self):
        """Temporal intent detected but no event_time -> falls to normal pipeline.

        We mock the downstream pipeline to avoid needing full LLM setup.
        """
        store = [
            MockMemoryEntry(
                text="Sarah graduated from MIT",
                event_time=None,  # No temporal data
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"graduated": ("MIT", "Sarah graduated from MIT")},
            },
            text_to_store_idx={"Sarah graduated from MIT": 0},
            memory_store=store,
        )
        # Mock Phase 1+ so we don't need LLM
        qe._query_setup = MagicMock(return_value={
            "output": "fallback answer",
            "input_len": 0,
            "output_len": 0,
            "memory_construction_time": 0.0,
            "query_time_len": 0.0,
        })
        prompt = (
            "Pretend you are a knowledge management system. "
            "Question: When did Sarah graduate?\nAnswer:"
        )
        result = qe._query(prompt)
        # Should have fallen through to _query_setup (Phase 1)
        assert result["output"] == "fallback answer"
        qe._query_setup.assert_called_once()

    def test_query_non_temporal_skips_phase0(self):
        """Non-temporal question should not trigger temporal resolution."""
        store = [
            MockMemoryEntry(
                text="Sarah graduated from MIT",
                event_time=datetime(2019, 6, 1),
            ),
        ]
        qe = _make_query_engine(
            subject_index={
                "sarah": {"graduated": ("MIT", "Sarah graduated from MIT")},
            },
            text_to_store_idx={"Sarah graduated from MIT": 0},
            memory_store=store,
        )
        # Mock Phase 1 to return a known answer
        qe._query_setup = MagicMock(return_value={
            "output": "MIT",
            "input_len": 0,
            "output_len": 0,
            "memory_construction_time": 0.0,
            "query_time_len": 0.0,
        })
        prompt = (
            "Pretend you are a knowledge management system. "
            "Question: What sport does Abbiati play?\nAnswer:"
        )
        result = qe._query(prompt)
        # Non-temporal -> should have gone to Phase 1 (mocked _query_setup)
        assert result["output"] == "MIT"
        qe._query_setup.assert_called_once()
