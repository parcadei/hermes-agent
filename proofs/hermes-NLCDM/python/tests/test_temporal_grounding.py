"""Adversarial tests for temporal grounding in LLM triple extractor.

Verifies:
1. ISO 8601 dates parsed correctly from LLM "when" field
2. Empty/missing "when" → None (timeless facts preserved)
3. Mixed temporal + timeless facts in same chunk
4. Relative dates resolved by LLM given reference_date
5. Edge cases: bad dates, partial dates, future dates
6. Sequential chunks with temporal accumulation
7. Factual mode unaffected (no "when" field)
8. reference_date flows into prompt
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from mabench.llm_triple_extractor import (
    LLMTripleExtractor,
    _parse_event_time,
    CONVERSATION_EXTRACTION_PROMPT,
)


# ---------------------------------------------------------------------------
# _parse_event_time (deterministic ISO 8601 parsing)
# ---------------------------------------------------------------------------

class TestParseEventTime:
    def test_date_only(self):
        assert _parse_event_time("2026-03-15") == datetime(2026, 3, 15)

    def test_datetime_full(self):
        assert _parse_event_time("2026-03-15T14:30:00") == datetime(2026, 3, 15, 14, 30)

    def test_empty_string(self):
        assert _parse_event_time("") is None

    def test_none_like(self):
        assert _parse_event_time("") is None

    def test_garbage(self):
        assert _parse_event_time("next week") is None

    def test_partial_date(self):
        # "2026-03" is not valid ISO 8601 date → None
        assert _parse_event_time("2026-03") is None

    def test_natural_language_date(self):
        # LLM should have resolved this, but if it didn't → None
        assert _parse_event_time("March 15, 2026") is None

    def test_future_date(self):
        assert _parse_event_time("2030-01-01") == datetime(2030, 1, 1)

    def test_past_date(self):
        assert _parse_event_time("1999-12-31") == datetime(1999, 12, 31)

    def test_datetime_with_seconds(self):
        assert _parse_event_time("2026-03-15T14:30:45") == datetime(2026, 3, 15, 14, 30, 45)

    def test_whitespace_only(self):
        # The caller strips, but just in case
        assert _parse_event_time("   ") is None

    def test_iso_with_timezone_suffix(self):
        # fromisoformat handles this in Python 3.11+
        result = _parse_event_time("2026-03-15T14:30:00+00:00")
        assert result is not None
        assert result.year == 2026


# ---------------------------------------------------------------------------
# Conversation extraction with temporal grounding (mocked LLM)
# ---------------------------------------------------------------------------

def _make_extractor(**kwargs) -> LLMTripleExtractor:
    ext = LLMTripleExtractor(enable_grounding=False, **kwargs)
    ext._client = MagicMock()
    return ext


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestTemporalConversationExtraction:
    """Tests with mocked LLM responses simulating conversation mode."""

    def test_mixed_temporal_and_timeless(self):
        """Chunk with both dated events and timeless preferences."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "Sarah", "predicate": "visiting", "object": "user", "when": "2026-04-05"},'
            '{"subject": "Sarah", "predicate": "bringing", "object": "lasagna", "when": "2026-04-05"},'
            '{"subject": "Jake", "predicate": "relationship", "object": "brother", "when": ""}'
            '], "aliases": [{"canonical": "Sarah", "alias": "Mom"}]}'
        )

        triples = ext.extract(
            "Mom is visiting next week with lasagna. My brother Jake might come.",
            mode="conversation", reference_date="2026-03-29",
        )

        assert len(triples) == 3
        assert triples[0] == ("Sarah", "visiting", "user")
        assert triples[2] == ("Jake", "relationship", "brother")

        # Event times: first two temporal, third timeless
        et = ext.last_event_times[0]
        assert et[0] == datetime(2026, 4, 5)
        assert et[1] == datetime(2026, 4, 5)
        assert et[2] is None

    def test_relative_date_resolved_by_llm(self):
        """LLM resolves '3 weeks ago' to absolute date given reference."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "Bobo", "predicate": "adopted from", "object": "shelter", "when": "2026-03-08"},'
            '{"subject": "Bobo", "predicate": "breed", "object": "golden retriever", "when": ""}'
            '], "aliases": []}'
        )

        triples = ext.extract(
            "We adopted Bobo from the shelter about 3 weeks ago. He's a golden retriever.",
            mode="conversation", reference_date="2026-03-29",
        )

        assert len(triples) == 2
        et = ext.last_event_times[0]
        assert et[0] == datetime(2026, 3, 8)  # 3 weeks before March 29
        assert et[1] is None  # breed is timeless

    def test_all_timeless_facts(self):
        """Chunk with no temporal content → all event_times are None."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "user", "predicate": "prefers", "object": "Python", "when": ""},'
            '{"subject": "VS Code", "predicate": "role", "object": "primary editor", "when": ""}'
            '], "aliases": []}'
        )

        triples = ext.extract(
            "I prefer Python and use VS Code.",
            mode="conversation", reference_date="2026-03-29",
        )

        assert len(triples) == 2
        assert all(et is None for et in ext.last_event_times[0])

    def test_future_event(self):
        """LLM extracts a future date ('next summer')."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "user", "predicate": "planning trip to", "object": "Japan", "when": "2026-07-01"}'
            '], "aliases": []}'
        )

        triples = ext.extract(
            "I'm planning a trip to Japan next summer.",
            mode="conversation", reference_date="2026-03-29",
        )

        assert len(triples) == 1
        et = ext.last_event_times[0]
        assert et[0] == datetime(2026, 7, 1)

    def test_multiple_dates_in_one_chunk(self):
        """Chunk mentions several different dates."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "user", "predicate": "graduated from", "object": "MIT", "when": "2019-06-01"},'
            '{"subject": "user", "predicate": "started at", "object": "Google", "when": "2025-03-01"},'
            '{"subject": "user", "predicate": "moving to", "object": "Austin", "when": "2026-06-15"}'
            '], "aliases": []}'
        )

        triples = ext.extract(
            "I graduated from MIT in 2019, started at Google last March, "
            "and I'm moving to Austin in June.",
            mode="conversation", reference_date="2026-03-29",
        )

        et = ext.last_event_times[0]
        assert et[0] == datetime(2019, 6, 1)
        assert et[1] == datetime(2025, 3, 1)
        assert et[2] == datetime(2026, 6, 15)

    def test_llm_returns_unparseable_when(self):
        """LLM outputs natural language instead of ISO 8601 → graceful None."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "user", "predicate": "visited", "object": "Paris", "when": "around Christmas"}'
            '], "aliases": []}'
        )

        triples = ext.extract(
            "I visited Paris around Christmas.",
            mode="conversation", reference_date="2026-03-29",
        )

        assert len(triples) == 1
        # Graceful degradation — unparseable → None, not crash
        assert ext.last_event_times[0][0] is None

    def test_knowledge_update_supersession(self):
        """Two chunks about the same fact with different dates —
        temporal ordering enables 'newest wins'."""
        ext = _make_extractor()

        # Chunk 1: old fact
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "Sarah", "predicate": "works at", "object": "Google", "when": "2024-01-15"}'
            '], "aliases": []}'
        )
        ext.extract("Sarah started working at Google in January 2024.",
                     mode="conversation", reference_date="2026-03-29")
        et1 = ext.last_event_times[0][0]

        # Chunk 2: updated fact
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "Sarah", "predicate": "works at", "object": "Meta", "when": "2026-02-01"}'
            '], "aliases": []}'
        )
        ext.extract("Sarah just moved to Meta last month.",
                     mode="conversation", reference_date="2026-03-29")
        et2 = ext.last_event_times[0][0]

        # Temporal ordering: Meta is newer than Google
        assert et1 == datetime(2024, 1, 15)
        assert et2 == datetime(2026, 2, 1)
        assert et2 > et1  # newest wins


class TestTemporalEdgeCases:
    """Edge cases and adversarial inputs."""

    def test_empty_triples(self):
        """Greeting with no facts → empty event_times."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": [], "aliases": []}'
        )

        triples = ext.extract("Hi there!",
                              mode="conversation", reference_date="2026-03-29")
        assert triples == []
        assert ext.last_event_times[0] == []

    def test_factual_mode_no_when_field(self):
        """Factual mode schema has no 'when' → event_times all None."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": [{"subject": "France", "predicate": "capital", "object": "Paris"}]}'
        )

        triples = ext.extract("The capital of France is Paris.")
        assert triples == [("France", "capital", "Paris")]
        # Factual mode: no "when" key → all None
        assert ext.last_event_times[0] == [None]

    def test_reference_date_in_prompt(self):
        """Verify reference_date appears in the LLM prompt."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": [], "aliases": []}'
        )

        ext.extract("test", mode="conversation", reference_date="2026-03-29")

        call_args = ext._client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "2026-03-29" in prompt

    def test_reference_date_defaults_to_today(self):
        """When no reference_date given, today's date is used."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": [], "aliases": []}'
        )

        ext.extract("test", mode="conversation")

        call_args = ext._client.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        from datetime import date
        assert date.today().isoformat() in prompt

    def test_stats_track_temporal(self):
        """Stats count how many triples have temporal grounding."""
        ext = _make_extractor()
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "A", "predicate": "did", "object": "B", "when": "2026-01-01"},'
            '{"subject": "C", "predicate": "is", "object": "D", "when": ""},'
            '{"subject": "E", "predicate": "did", "object": "F", "when": "2026-06-15"}'
            '], "aliases": []}'
        )

        ext.extract("...", mode="conversation", reference_date="2026-03-29")
        assert ext.stats["temporal"] == 2
        assert ext.stats["triples"] == 3

    def test_batch_event_times(self):
        """Batch extraction produces parallel event_times for each text."""
        ext = _make_extractor()

        responses = [
            _mock_response(
                '{"triples": [{"subject": "A", "predicate": "x", "object": "B", "when": "2026-01-01"}],'
                '"aliases": []}'
            ),
            _mock_response(
                '{"triples": [{"subject": "C", "predicate": "y", "object": "D", "when": ""}],'
                '"aliases": []}'
            ),
        ]
        ext._client.chat.completions.create.side_effect = responses

        results = ext.extract_batch(
            ["Text about A", "Text about C"],
            mode="conversation", reference_date="2026-03-29",
        )

        assert len(results) == 2
        assert len(ext.last_event_times) == 2
        assert ext.last_event_times[0][0] == datetime(2026, 1, 1)
        assert ext.last_event_times[1][0] is None

    def test_llm_failure_returns_empty(self):
        """LLM error → empty triples and empty event_times."""
        ext = _make_extractor()
        ext._client.chat.completions.create.side_effect = Exception("API down")

        triples = ext.extract("some text", mode="conversation",
                              reference_date="2026-03-29")
        assert triples == []
        assert ext.last_event_times[0] == []

    def test_pickle_preserves_last_event_times(self):
        """Pickle round-trip preserves _last_event_times."""
        import pickle
        ext = LLMTripleExtractor()
        ext._last_event_times = [[datetime(2026, 3, 15), None]]
        restored = pickle.loads(pickle.dumps(ext))
        assert restored.last_event_times == [[datetime(2026, 3, 15), None]]


class TestTemporalConversationSequential:
    """Multi-chunk sequential extraction with temporal + alias accumulation."""

    def test_three_chunk_temporal_flow(self):
        """Simulates a real conversation with temporal progression:
        Chunk 1: introduces entity + past event with date
        Chunk 2: future event for same entity
        Chunk 3: knowledge update (job change)
        """
        ext = _make_extractor()

        # Chunk 1: "I met Sarah at a conference in January 2025"
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "user", "predicate": "met", "object": "Sarah", "when": "2025-01-15"},'
            '{"subject": "Sarah", "predicate": "met at", "object": "conference", "when": "2025-01-15"}'
            '], "aliases": []}'
        )
        t1 = ext.extract(
            "I met Sarah at a conference in January 2025.",
            mode="conversation", reference_date="2026-03-29",
        )
        assert len(t1) == 2
        assert ext.last_event_times[0][0] == datetime(2025, 1, 15)
        assert ext.last_event_times[0][1] == datetime(2025, 1, 15)

        # Chunk 2: "Sarah is coming to visit next Friday"
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "Sarah", "predicate": "visiting", "object": "user", "when": "2026-04-04"}'
            '], "aliases": []}'
        )
        t2 = ext.extract(
            "Sarah is coming to visit next Friday.",
            mode="conversation", reference_date="2026-03-29",
        )
        assert len(t2) == 1
        assert ext.last_event_times[0][0] == datetime(2026, 4, 4)

        # Chunk 3: "She just switched jobs — she's at OpenAI now"
        ext._client.chat.completions.create.return_value = _mock_response(
            '{"triples": ['
            '{"subject": "Sarah", "predicate": "works at", "object": "OpenAI", "when": "2026-03-29"}'
            '], "aliases": [{"canonical": "Sarah", "alias": "she"}]}'
        )
        t3 = ext.extract(
            "She just switched jobs — she's at OpenAI now.",
            mode="conversation", reference_date="2026-03-29",
        )
        assert len(t3) == 1
        assert t3[0] == ("Sarah", "works at", "OpenAI")
        assert ext.last_event_times[0][0] == datetime(2026, 3, 29)
        assert ext.entity_registry == {"Sarah": ["she"]}

        # Stats accumulated across all 3 chunks
        assert ext.stats["temporal"] == 4  # 2 + 1 + 1
        assert ext.stats["triples"] == 4
        assert ext.stats["aliases"] == 1


# ---------------------------------------------------------------------------
# Few-shot prompt: knowledge-update temporal examples
# ---------------------------------------------------------------------------

class TestKnowledgeUpdateFewShotExamples:
    """Verify the CONVERSATION_EXTRACTION_PROMPT contains few-shot examples
    demonstrating knowledge-update patterns with relative temporal expressions.
    These examples teach the LLM to resolve 'last month', 'last year',
    'two weeks ago' etc. in knowledge-update sentences."""

    def test_job_change_example_present(self):
        """Prompt contains the 'used to work at Google / started at OpenAI
        last month' knowledge-update example."""
        assert "used to work at" in CONVERSATION_EXTRACTION_PROMPT
        assert "started working at" in CONVERSATION_EXTRACTION_PROMPT
        assert "OpenAI" in CONVERSATION_EXTRACTION_PROMPT
        # The 'last month' relative date should be resolved to 2026-02-01
        assert "2026-02-01" in CONVERSATION_EXTRACTION_PROMPT

    def test_job_change_example_timeless_past(self):
        """The 'used to work at Google' fact has when="" because we only know
        it ended, not when it started."""
        # Find the specific pattern: used_to_work_at with empty when
        prompt = CONVERSATION_EXTRACTION_PROMPT
        # The example must show when="" for the past-tense "used to work at"
        assert '"used to work at", "object": "Google", "when": ""' in prompt

    def test_job_change_example_location_change(self):
        """The job change example also includes location moves
        (Mountain View -> San Francisco), resolved to same date."""
        prompt = CONVERSATION_EXTRACTION_PROMPT
        assert "Mountain View" in prompt
        assert "San Francisco" in prompt
        assert "moved from" in prompt
        assert "moved to" in prompt

    def test_education_timeline_example_present(self):
        """Prompt contains the PhD/startup/Series B education timeline example
        with mixed absolute ('in 2022') and relative ('last year', 'two weeks ago')
        dates."""
        prompt = CONVERSATION_EXTRACTION_PROMPT
        assert "finished PhD at" in prompt
        assert "Stanford" in prompt
        # "in 2022" -> 2022-06-01
        assert "2022-06-01" in prompt
        # "last year" -> 2025-01-01
        assert "2025-01-01" in prompt
        # "two weeks ago" -> 2026-03-15
        assert "2026-03-15" in prompt

    def test_education_timeline_example_subjects(self):
        """The education example uses 'user' for personal facts and 'startup'
        for the company fact."""
        prompt = CONVERSATION_EXTRACTION_PROMPT
        assert '"subject": "user", "predicate": "finished PhD at"' in prompt
        assert '"subject": "user", "predicate": "joined"' in prompt
        assert '"subject": "startup", "predicate": "raised"' in prompt

    def test_examples_before_greeting(self):
        """The two knowledge-update examples appear BEFORE the empty greeting
        example ('Hi! Can you help me with something?')."""
        prompt = CONVERSATION_EXTRACTION_PROMPT
        idx_job_change = prompt.index("used to work at")
        idx_education = prompt.index("finished PhD at")
        idx_greeting = prompt.index("Hi! Can you help me with something?")
        assert idx_job_change < idx_greeting
        assert idx_education < idx_greeting

    def test_examples_after_mom_called(self):
        """The two knowledge-update examples appear AFTER the 'Mom called again'
        alias resolution example."""
        prompt = CONVERSATION_EXTRACTION_PROMPT
        idx_mom_called = prompt.index("Mom called again")
        idx_job_change = prompt.index("used to work at")
        idx_education = prompt.index("finished PhD at")
        assert idx_mom_called < idx_job_change
        assert idx_mom_called < idx_education

    def test_temporal_instruction_mentions_changed(self):
        """The temporal grounding instruction includes 'changed' in the list
        of temporal triggers (started, ended, is scheduled, or changed)."""
        prompt = CONVERSATION_EXTRACTION_PROMPT
        assert "or changed)" in prompt

    def test_knowledge_update_instruction_present(self):
        """The prompt explicitly instructs LLM to resolve relative dates in
        knowledge-update sentences."""
        prompt = CONVERSATION_EXTRACTION_PROMPT
        assert "knowledge-update" in prompt.lower() or "knowledge-update" in prompt
        assert "ALWAYS resolve the relative date" in prompt
