"""Tests for deterministic date-arithmetic post-processor.

Covers classification, arithmetic computation, candidate-date extraction,
and the full question->answer pipeline. All tests are synthetic and
deterministic. No LLM calls, no fixture dependency.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from mabench.date_arithmetic import (
    AnsweredQuery,
    answer_date_arithmetic,
    classify_date_arithmetic_question,
    compute_days_between,
    compute_units_ago,
    extract_event_date_from_turns,
    extract_query_date,
    format_answer,
)

# ---------------------------------------------------------------------------
# classify_date_arithmetic_question
# ---------------------------------------------------------------------------

class TestClassify:
    def test_weeks_ago(self):
        assert (
            classify_date_arithmetic_question(
                "How many weeks ago did I attend the festival?"
            )
            == "weeks_ago"
        )

    def test_days_ago(self):
        assert (
            classify_date_arithmetic_question("How many days ago did I start?")
            == "days_ago"
        )

    def test_months_ago(self):
        assert (
            classify_date_arithmetic_question(
                "How many months ago did I attend the photography workshop?"
            )
            == "months_ago"
        )

    def test_years_ago(self):
        assert (
            classify_date_arithmetic_question("How many years ago did we meet?")
            == "years_ago"
        )

    def test_days_passed(self):
        assert (
            classify_date_arithmetic_question(
                "How many days had passed since I finished reading the book?"
            )
            == "days_passed"
        )

    def test_days_between(self):
        assert (
            classify_date_arithmetic_question(
                "How many days had passed between the day I bought a gift "
                "and the day I bought a birthday gift?"
            )
            == "days_between"
        )

    def test_date_of(self):
        assert (
            classify_date_arithmetic_question(
                "What was the date on which I attended the first BBQ event in June?"
            )
            == "date_of"
        )

    def test_how_long_ago(self):
        assert (
            classify_date_arithmetic_question("How long ago did I buy the car?")
            == "how_long_ago"
        )

    def test_negative_favorite_color(self):
        assert (
            classify_date_arithmetic_question("What is my favorite color?")
            == "none"
        )

    def test_negative_who(self):
        assert (
            classify_date_arithmetic_question("Who is Alex?") == "none"

        )

    def test_negative_aggregation(self):
        assert (
            classify_date_arithmetic_question(
                "How many hours do I work in a typical week?"
            )
            == "none"
        )


# ---------------------------------------------------------------------------
# extract_query_date
# ---------------------------------------------------------------------------

class TestExtractQueryDate:
    def test_slash_format(self):
        q = "Current Date: 2023/08/05 (Sat) 23:49,\n\n Now Answer the Question: ..."
        assert extract_query_date(q) == date(2023, 8, 5)

    def test_dash_format(self):
        q = "Current Date: 2023-08-05 ..."
        assert extract_query_date(q) == date(2023, 8, 5)

    def test_missing(self):
        assert extract_query_date("No date here") is None


# ---------------------------------------------------------------------------
# compute_units_ago / compute_days_between
# ---------------------------------------------------------------------------

class TestComputeUnits:
    def test_weeks_ago_exact(self):
        # 2023-08-05 - 2023-07-15 = 21 days = 3 weeks
        assert (
            compute_units_ago(date(2023, 8, 5), date(2023, 7, 15), "weeks")
            == 3
        )

    def test_days_ago(self):
        # 2023-07-01 - 2023-06-28 = 3 days
        assert (
            compute_units_ago(date(2023, 7, 1), date(2023, 6, 28), "days") == 3
        )

    def test_months_ago(self):
        # 2023-08-05 - 2023-05-05 = 92 days ~= 3 months
        assert (
            compute_units_ago(date(2023, 8, 5), date(2023, 5, 5), "months") == 3
        )

    def test_months_ago_feb_leap(self):
        # 2024-02-01 - 2023-11-01 = 92 days ~= 3 months
        assert (
            compute_units_ago(date(2024, 2, 1), date(2023, 11, 1), "months") == 3
        )

    def test_years_ago(self):
        assert (
            compute_units_ago(date(2024, 6, 1), date(2022, 6, 1), "years") == 2
        )

    def test_days_between_absolute(self):
        assert (
            compute_days_between(date(2023, 3, 29), date(2023, 3, 22)) == 7
        )

    def test_days_between_reversed_absolute(self):
        # Should be absolute value
        assert (
            compute_days_between(date(2023, 3, 22), date(2023, 3, 29)) == 7
        )


# ---------------------------------------------------------------------------
# extract_event_date_from_turns
# ---------------------------------------------------------------------------

class TestExtractEventDateFromTurns:
    def test_picks_dominant_date(self):
        """3 turns with one clear date mention -> returns that date."""
        turns = [
            ("[2023/07/16 (Sun) 14:00] (user) I attended the Summer Nights "
             "festival at Universal Studios Hollywood on July 15th.")
        ]
        anchor = datetime(2023, 8, 5)
        result = extract_event_date_from_turns(turns, anchor)
        assert result == date(2023, 7, 15)

    def test_multiple_turns_same_date(self):
        turns = [
            "[2023/07/16 (Sun) 14:00] (user) The festival was July 15th.",
            "[2023/07/16 (Sun) 14:05] (assistant) Great, July 15th it was.",
        ]
        anchor = datetime(2023, 8, 5)
        result = extract_event_date_from_turns(turns, anchor)
        assert result == date(2023, 7, 15)

    def test_no_explicit_dates_falls_back_to_turn_timestamp(self):
        """When text has no date mentions, the turn's own timestamp is used."""
        turns = ["[2023/07/16 (Sun) 14:00] (user) Hello world."]
        anchor = datetime(2023, 8, 5)
        # With turn-timestamp fallback enabled (default), returns 2023-07-16.
        assert extract_event_date_from_turns(turns, anchor) == date(2023, 7, 16)

    def test_no_explicit_dates_disabled_fallback_returns_none(self):
        turns = ["[2023/07/16 (Sun) 14:00] (user) Hello world."]
        anchor = datetime(2023, 8, 5)
        assert (
            extract_event_date_from_turns(
                turns, anchor, include_turn_timestamps=False
            )
            is None
        )

    def test_no_turn_timestamp_and_no_mentions_returns_none(self):
        turns = ["(user) Hello world, no timestamp."]
        anchor = datetime(2023, 8, 5)
        assert extract_event_date_from_turns(turns, anchor) is None

    def test_empty_turns(self):
        assert extract_event_date_from_turns([], datetime(2023, 8, 5)) is None

    def test_explicit_mention_outranks_turn_timestamp(self):
        """Explicit date mention should beat turn-timestamp fallback."""
        turns = [
            "[2023/07/16 (Sun) 14:00] (user) I attended the event on July 15th.",
        ]
        anchor = datetime(2023, 8, 5)
        # Explicit "July 15th" (weight 2) beats turn ts (weight 1)
        assert extract_event_date_from_turns(turns, anchor) == date(2023, 7, 15)


# ---------------------------------------------------------------------------
# format_answer
# ---------------------------------------------------------------------------

class TestFormatAnswer:
    def test_weeks_ago(self):
        assert format_answer("weeks_ago", 3) == "3 weeks ago"

    def test_days_ago(self):
        assert format_answer("days_ago", 3) == "3 days ago"

    def test_months_ago(self):
        assert format_answer("months_ago", 3) == "3 months ago"

    def test_days_between(self):
        assert format_answer("days_between", 7) == "7 days"

    def test_days_passed(self):
        assert format_answer("days_passed", 18) == "18 days"

    def test_date_of(self):
        assert format_answer("date_of", date(2023, 6, 3)) == "June 3rd"

    def test_date_of_first(self):
        assert format_answer("date_of", date(2023, 6, 1)) == "June 1st"

    def test_date_of_second(self):
        assert format_answer("date_of", date(2023, 6, 22)) == "June 22nd"

    def test_date_of_teen(self):
        # 11th, 12th, 13th all use "th"
        assert format_answer("date_of", date(2023, 6, 11)) == "June 11th"
        assert format_answer("date_of", date(2023, 6, 13)) == "June 13th"


# ---------------------------------------------------------------------------
# answer_date_arithmetic (full pipeline integration)
# ---------------------------------------------------------------------------

class TestAnswerDateArithmetic:
    def test_weeks_ago_pipeline(self):
        question = (
            "Current Date: 2023/08/05 (Sat) 23:49,\n\n Now Answer the Question: "
            "How many weeks ago did I attend the 'Summer Nights' festival?"
        )
        turns = [
            ("[2023/07/16 (Sun) 14:00] (user) I attended the Summer Nights "
             "festival on July 15th at Universal Studios Hollywood.")
        ]
        result = answer_date_arithmetic(question, turns)
        assert isinstance(result, AnsweredQuery)
        assert result.classification == "weeks_ago"
        assert result.event_date == date(2023, 7, 15)
        assert result.computed_answer == "3 weeks ago"
        assert result.confidence > 0.0

    def test_date_of_pipeline(self):
        question = (
            "Current Date: 2023/07/01 (Sat) 23:13,\n\n Now Answer the Question: "
            "What was the date on which I attended the first BBQ event in June?"
        )
        turns = [
            ("[2023/06/04 (Sun) 10:00] (user) I attended the BBQ event on "
             "June 3rd and it was fun.")
        ]
        result = answer_date_arithmetic(question, turns)
        assert result.classification == "date_of"
        assert result.event_date == date(2023, 6, 3)
        assert result.computed_answer == "June 3rd"

    def test_months_ago_pipeline(self):
        question = (
            "Current Date: 2024/02/01 (Thu) 22:41,\n\n Now Answer the Question: "
            "How many months ago did I attend the photography workshop?"
        )
        turns = [
            ("[2023/11/02 (Thu) 09:00] (user) I attended the photography "
             "workshop on November 1st, it was excellent."),
        ]
        result = answer_date_arithmetic(question, turns)
        assert result.classification == "months_ago"
        assert result.event_date == date(2023, 11, 1)
        assert result.computed_answer == "3 months ago"

    def test_non_arithmetic_returns_none_classification(self):
        question = "Current Date: 2023/05/27 (Sat) 22:54,\n\n Who is Alex?"
        result = answer_date_arithmetic(question, [])
        assert result.classification == "none"
        assert result.computed_answer is None

    def test_no_turns_low_confidence(self):
        question = (
            "Current Date: 2023/08/05 (Sat) 23:49,\n\n Now Answer the Question: "
            "How many weeks ago did I attend the festival?"
        )
        result = answer_date_arithmetic(question, [])
        assert result.classification == "weeks_ago"
        assert result.computed_answer is None
        assert result.confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
