"""Tests for LME-specific multi-hop query decomposer.

Rule-based (regex) classifier + session-aware aggregator for LongMemEval
multi-session aggregation questions ("how many X total", "count all Y",
"list all Z across conversations"). No LLM calls — memory must stay cheap.

Library only — not wired into the agent in this PR.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from mabench.lme_multihop import (
    AggregationPlan,
    aggregate_hop_results,
    classify_multihop_query,
    decompose_aggregation_query,
)

# ---------------------------------------------------------------------------
# classify_multihop_query: kind detection
# ---------------------------------------------------------------------------

class TestClassifyMultihopQuery:
    """Regex-based classifier. Five positives, three negatives."""

    # ---- positives ----

    def test_count_how_many(self):
        # Plain count: "how many X"
        assert classify_multihop_query(
            "How many fish are there in total in both of my aquariums?"
        ) == "count"

    def test_count_distinct_different(self):
        # "how many DIFFERENT X" / "distinct" / "unique" -> count_distinct
        assert classify_multihop_query(
            "How many different art-related events did I attend in the past month?"
        ) == "count_distinct"

    def test_total_sum_in_total(self):
        # "total cost / in total / altogether / sum"
        assert classify_multihop_query(
            "What is the total cost of the new food bowl, measuring cup, "
            "dental chews, and flea collar?"
        ) == "total_sum"

    def test_list_all(self):
        # "list all X" or "all the X"
        assert classify_multihop_query(
            "List all the restaurants I visited last month."
        ) == "list_all"

    def test_count_how_often(self):
        # "how often" is also a count pattern
        assert classify_multihop_query(
            "How often do I go to the gym?"
        ) == "count"

    # ---- negatives ----

    def test_none_temporal_date_diff(self):
        # "weeks ago" is temporal date-diff, NOT aggregation
        assert classify_multihop_query(
            "How many weeks ago did I go to Hawaii?"
        ) == "none"

    def test_none_single_fact_point_query(self):
        # Plain point query — no aggregation
        assert classify_multihop_query(
            "What is my favorite restaurant in Miami?"
        ) == "none"

    def test_none_days_passed_since(self):
        # "days passed since" is temporal, not aggregation
        assert classify_multihop_query(
            "How many days have passed since my last dentist visit?"
        ) == "none"


# ---------------------------------------------------------------------------
# decompose_aggregation_query: plan generation
# ---------------------------------------------------------------------------

class TestDecomposeAggregationQuery:
    """Given a multi-session query, produce AggregationPlan."""

    def test_returns_plan_for_count_query(self):
        plan = decompose_aggregation_query(
            "How many fish are there in total in both of my aquariums?",
            query_date=datetime(2023, 5, 27, 17, 21),
        )
        assert isinstance(plan, AggregationPlan)
        assert plan.kind == "count"
        assert plan.base_query  # non-empty
        assert isinstance(plan.sub_queries, list)
        # Should include base + at least one variant
        assert len(plan.sub_queries) >= 2
        assert plan.per_session_top_k >= 1
        assert plan.max_sessions >= 2
        assert plan.merge_strategy == "union_dedup"

    def test_returns_none_kind_for_point_query(self):
        plan = decompose_aggregation_query(
            "What is my favorite restaurant in Miami?",
            query_date=None,
        )
        assert plan.kind == "none"
        # For point queries, sub_queries is just [base] — no fan-out
        assert plan.sub_queries == [plan.base_query]

    def test_strips_aggregation_prefix_in_variant(self):
        # "how many fish in both aquariums" should produce a subject-stripped
        # variant like "fish in both aquariums" (no "how many").
        plan = decompose_aggregation_query(
            "How many fish are there in total in both of my aquariums?",
            query_date=None,
        )
        # At least one variant must NOT contain "how many"
        has_stripped = any(
            "how many" not in sq.lower()
            for sq in plan.sub_queries
        )
        assert has_stripped, (
            f"Expected at least one variant without 'how many'. "
            f"Got: {plan.sub_queries}"
        )

    def test_list_all_kind_propagated(self):
        plan = decompose_aggregation_query(
            "List all the concerts I went to this year.",
            query_date=None,
        )
        assert plan.kind == "list_all"
        assert len(plan.sub_queries) >= 2


# ---------------------------------------------------------------------------
# aggregate_hop_results: session-aware merge
# ---------------------------------------------------------------------------

class TestAggregateHopResults:
    """Round-robin interleave across sessions + dedup."""

    @staticmethod
    def _make_result(text: str, session_date: str | None, score: float):
        st = (
            datetime.fromisoformat(session_date + "T12:00:00")
            if session_date is not None
            else None
        )
        return {"text": text, "session_time": st, "score": score}

    def _build_plan(self, kind="count", per_session_top_k=2, max_sessions=3):
        return AggregationPlan(
            kind=kind,
            base_query="how many fish",
            sub_queries=["how many fish", "fish aquarium"],
            per_session_top_k=per_session_top_k,
            max_sessions=max_sessions,
            merge_strategy="union_dedup",
        )

    def test_round_robin_interleaves_sessions(self):
        # 3 sessions, 2 results each, all distinct
        results = [
            self._make_result("A1", "2023-01-01", 0.9),
            self._make_result("A2", "2023-01-01", 0.8),
            self._make_result("B1", "2023-02-15", 0.85),
            self._make_result("B2", "2023-02-15", 0.7),
            self._make_result("C1", "2023-03-20", 0.75),
            self._make_result("C2", "2023-03-20", 0.65),
        ]
        plan = self._build_plan(per_session_top_k=2, max_sessions=3)
        merged = aggregate_hop_results(results, plan)
        texts = [r["text"] for r in merged]
        # Round-robin: first pass picks top-1 from each session by score,
        # then second pass picks top-2 from each session. Exact ordering
        # within a round depends on session score; we just verify that
        # all three sessions are represented in the first 3 picks.
        first_three = set(texts[:3])
        assert "A1" in first_three  # A top-1
        assert "B1" in first_three  # B top-1
        assert "C1" in first_three  # C top-1
        # All 6 present
        assert len(merged) == 6
        assert set(texts) == {"A1", "A2", "B1", "B2", "C1", "C2"}

    def test_per_session_cap_enforced(self):
        # One session has 5 results; cap at 2
        results = [
            self._make_result(f"A{i}", "2023-01-01", 1.0 - i * 0.1)
            for i in range(5)
        ]
        plan = self._build_plan(per_session_top_k=2, max_sessions=3)
        merged = aggregate_hop_results(results, plan)
        texts = [r["text"] for r in merged]
        # Only top-2 by score kept: A0, A1
        assert set(texts) == {"A0", "A1"}
        # None of A2/A3/A4
        assert "A2" not in texts
        assert "A3" not in texts
        assert "A4" not in texts

    def test_max_sessions_cap_enforced(self):
        # 5 sessions but max_sessions=3
        results = []
        for i in range(5):
            date = f"2023-0{i + 1}-01"
            results.append(self._make_result(f"S{i}", date, 0.9 - i * 0.1))
        plan = self._build_plan(per_session_top_k=1, max_sessions=3)
        merged = aggregate_hop_results(results, plan)
        # Only top-3 sessions by score
        texts = [r["text"] for r in merged]
        assert len(merged) == 3
        assert "S0" in texts
        assert "S1" in texts
        assert "S2" in texts

    def test_dedup_by_text_prefix(self):
        # Duplicate text across sessions → dedup keeps first occurrence
        results = [
            self._make_result("my fish count is 10", "2023-01-01", 0.9),
            self._make_result("my fish count is 10", "2023-02-01", 0.8),
            self._make_result("fish in tank B", "2023-02-01", 0.85),
        ]
        plan = self._build_plan(per_session_top_k=2, max_sessions=3)
        merged = aggregate_hop_results(results, plan)
        # Only one "my fish count is 10" should remain
        fish_count = sum(
            1 for r in merged if r["text"] == "my fish count is 10"
        )
        assert fish_count == 1

    def test_none_session_time_bucket(self):
        # Results without session_time go into a None bucket — still returned
        results = [
            self._make_result("X1", None, 0.9),
            self._make_result("X2", "2023-01-01", 0.8),
        ]
        plan = self._build_plan(per_session_top_k=2, max_sessions=3)
        merged = aggregate_hop_results(results, plan)
        texts = [r["text"] for r in merged]
        # Both should be present (None and "2023-01-01" are 2 distinct buckets)
        assert "X1" in texts
        assert "X2" in texts


# ---------------------------------------------------------------------------
# End-to-end: synthetic 3-session fixture → aggregated answer scaffold
# ---------------------------------------------------------------------------

class TestEndToEndMultiSession:
    """Synthetic turns across 3 sessions; verify plan + aggregation surfaces
    evidence from all sessions for a multi-session aggregation query."""

    def test_three_session_fixture_produces_multi_session_context(self):
        # Synthetic "turns" mimicking retrieval output (text + session_time + score)
        turns = [
            # Session 1 (2023-01-15): aquarium A
            {"text": "I have 10 fish in my aquarium in the living room",
             "session_time": datetime(2023, 1, 15, 10, 0), "score": 0.91},
            {"text": "My living room tank has mostly guppies and tetras",
             "session_time": datetime(2023, 1, 15, 10, 5), "score": 0.72},
            # Session 2 (2023-02-20): aquarium B
            {"text": "I set up a second aquarium in my office with 7 fish",
             "session_time": datetime(2023, 2, 20, 14, 30), "score": 0.88},
            {"text": "The office tank is smaller, just a 10-gallon",
             "session_time": datetime(2023, 2, 20, 14, 35), "score": 0.65},
            # Session 3 (2023-03-10): unrelated pet talk (noise)
            {"text": "My dog Max is doing great on the new food",
             "session_time": datetime(2023, 3, 10, 9, 0), "score": 0.45},
        ]

        question = "How many fish are there in total in both of my aquariums?"
        plan = decompose_aggregation_query(
            question,
            query_date=datetime(2023, 5, 27, 17, 21),
        )

        # Plan reflects aggregation intent
        assert plan.kind == "count"
        assert len(plan.sub_queries) >= 2

        # Aggregate with session diversity
        merged = aggregate_hop_results(turns, plan)

        # Evidence from session 1 AND session 2 must both be in merged output
        session_dates = {
            r["session_time"].date().isoformat()
            for r in merged
            if r.get("session_time") is not None
        }
        assert "2023-01-15" in session_dates, (
            f"Session 1 (living room aquarium) missing. Got: {session_dates}"
        )
        assert "2023-02-20" in session_dates, (
            f"Session 2 (office aquarium) missing. Got: {session_dates}"
        )
        # At least one relevant chunk from each relevant session
        texts = " ".join(r["text"] for r in merged)
        assert "10 fish" in texts
        assert "7 fish" in texts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
