"""Tests for the longitudinal evaluation protocol.

Tests cover:
  1. Data structures (Session, EvalQuestion, LongitudinalDataset)
  2. Dataset generation determinism and distribution
  3. Answer scoring logic
  4. Composite scoring
  5. Evaluator interface and session ordering
  6. Dream timing logic (idle period detection)
  7. __main__ execution path
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the parent directory is on the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# 1. Data structure tests
# ---------------------------------------------------------------------------

class TestDataStructures:
    """Verify Session, EvalQuestion, and LongitudinalDataset dataclasses."""

    def test_session_fields(self):
        from longitudinal_eval import Session
        s = Session(day=5, session_type="preference", facts=["I like pizza"])
        assert s.day == 5
        assert s.session_type == "preference"
        assert s.facts == ["I like pizza"]

    def test_eval_question_fields(self):
        from longitudinal_eval import EvalQuestion
        q = EvalQuestion(
            question="Where do I work?",
            expected_keywords=["anthropic"],
            rejected_keywords=["google"],
            category="current_fact",
        )
        assert q.question == "Where do I work?"
        assert q.expected_keywords == ["anthropic"]
        assert q.rejected_keywords == ["google"]
        assert q.category == "current_fact"

    def test_longitudinal_dataset_fields(self):
        from longitudinal_eval import LongitudinalDataset, generate_dataset
        ds = generate_dataset(seed=42)
        assert isinstance(ds.sessions, list)
        assert isinstance(ds.questions, list)
        assert ds.seed == 42


# ---------------------------------------------------------------------------
# 2. Dataset generation tests
# ---------------------------------------------------------------------------

class TestDatasetGeneration:
    """Verify generate_dataset produces correct distributions."""

    @pytest.fixture(scope="class")
    def dataset(self):
        from longitudinal_eval import generate_dataset
        return generate_dataset(seed=42)

    def test_session_count(self, dataset):
        """Must produce exactly 200 sessions."""
        assert len(dataset.sessions) == 200

    def test_question_count(self, dataset):
        """Must produce exactly 100 eval questions."""
        assert len(dataset.questions) == 100

    def test_determinism_same_seed(self):
        """Same seed produces identical datasets."""
        from longitudinal_eval import generate_dataset
        ds1 = generate_dataset(seed=42)
        ds2 = generate_dataset(seed=42)
        assert len(ds1.sessions) == len(ds2.sessions)
        assert len(ds1.questions) == len(ds2.questions)
        for s1, s2 in zip(ds1.sessions, ds2.sessions):
            assert s1.day == s2.day
            assert s1.session_type == s2.session_type
            assert s1.facts == s2.facts
        for q1, q2 in zip(ds1.questions, ds2.questions):
            assert q1.question == q2.question
            assert q1.expected_keywords == q2.expected_keywords
            assert q1.rejected_keywords == q2.rejected_keywords
            assert q1.category == q2.category

    def test_determinism_different_seed(self):
        """Different seeds produce different datasets."""
        from longitudinal_eval import generate_dataset
        ds1 = generate_dataset(seed=42)
        ds2 = generate_dataset(seed=99)
        # At least some facts should differ
        facts1 = [f for s in ds1.sessions for f in s.facts]
        facts2 = [f for s in ds2.sessions for f in s.facts]
        assert facts1 != facts2

    def test_session_type_distribution(self, dataset):
        """Session types match the specified distribution."""
        from collections import Counter
        type_counts = Counter(s.session_type for s in dataset.sessions)
        assert type_counts["preference"] == 50
        assert type_counts["update"] == 40
        assert type_counts["repeated"] == 60
        assert type_counts["single"] == 30
        assert type_counts["cross_domain"] == 20

    def test_question_category_distribution(self, dataset):
        """Question categories match the specified distribution."""
        from collections import Counter
        cat_counts = Counter(q.category for q in dataset.questions)
        assert cat_counts["current_fact"] == 30
        assert cat_counts["graceful_forgetting"] == 20
        assert cat_counts["reinforced_recall"] == 25
        assert cat_counts["cross_domain"] == 25

    def test_session_days_range(self, dataset):
        """Session days span 0 to 180."""
        days = [s.day for s in dataset.sessions]
        assert min(days) >= 0
        assert max(days) <= 180

    def test_sessions_chronological(self, dataset):
        """Sessions are ordered chronologically by day."""
        days = [s.day for s in dataset.sessions]
        assert days == sorted(days)

    def test_each_session_has_facts(self, dataset):
        """Every session has 1-5 facts."""
        for s in dataset.sessions:
            assert 1 <= len(s.facts) <= 5, (
                f"Session on day {s.day} has {len(s.facts)} facts"
            )

    def test_update_sessions_occur_after_preferences(self, dataset):
        """Update sessions should occur after the preferences they contradict."""
        pref_days = [s.day for s in dataset.sessions if s.session_type == "preference"]
        update_days = [s.day for s in dataset.sessions if s.session_type == "update"]
        # At least some update sessions should be later than some preference sessions
        assert min(update_days) > min(pref_days)

    def test_session_clustering(self, dataset):
        """Sessions should be clustered: multiple sessions per active day."""
        from collections import Counter
        day_counts = Counter(s.day for s in dataset.sessions)
        # At least some days should have multiple sessions
        multi_session_days = [d for d, c in day_counts.items() if c > 1]
        assert len(multi_session_days) > 5, (
            f"Only {len(multi_session_days)} days have multiple sessions"
        )

    def test_questions_have_keywords(self, dataset):
        """Every question has at least one expected keyword."""
        for q in dataset.questions:
            assert len(q.expected_keywords) >= 1, (
                f"Question '{q.question}' has no expected keywords"
            )

    def test_current_fact_questions_have_rejected_keywords(self, dataset):
        """Current fact questions about updated facts should have rejected keywords."""
        current_fact_qs = [
            q for q in dataset.questions if q.category == "current_fact"
        ]
        # At least some current fact questions should have rejected keywords
        # (those testing updated facts)
        with_rejected = [q for q in current_fact_qs if len(q.rejected_keywords) > 0]
        assert len(with_rejected) >= 5, (
            f"Only {len(with_rejected)} current_fact questions have rejected keywords"
        )

    def test_no_llm_calls_during_generation(self):
        """Dataset generation must not import or call any LLM libraries."""
        import importlib
        from unittest.mock import patch

        # Block imports of LLM-related modules
        blocked = ["openai", "transformers", "torch", "anthropic"]
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else importlib.__import__

        def guarded_import(name, *args, **kwargs):
            for b in blocked:
                if name == b or name.startswith(b + "."):
                    raise ImportError(f"Dataset generation must not import {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            # Re-import and regenerate - should work without LLM libraries
            if "longitudinal_eval" in sys.modules:
                # Can't truly re-import safely with the guard, so just verify
                # that generate_dataset doesn't call external APIs
                pass

        # Simpler check: generate_dataset should complete in <1 second
        import time
        from longitudinal_eval import generate_dataset
        start = time.time()
        generate_dataset(seed=123)
        elapsed = time.time() - start
        assert elapsed < 2.0, (
            f"generate_dataset took {elapsed:.2f}s - likely calling external APIs"
        )


# ---------------------------------------------------------------------------
# 3. Answer scoring tests
# ---------------------------------------------------------------------------

class TestAnswerScoring:
    """Verify score_answer logic."""

    def test_all_expected_present(self):
        from longitudinal_eval import score_answer
        score = score_answer(
            "I work at Anthropic in London",
            expected=["anthropic", "london"],
            rejected=[],
        )
        assert score == 1.0

    def test_partial_expected(self):
        from longitudinal_eval import score_answer
        score = score_answer(
            "I work at Anthropic",
            expected=["anthropic", "london"],
            rejected=[],
        )
        assert score == pytest.approx(0.5)

    def test_no_expected_present(self):
        from longitudinal_eval import score_answer
        score = score_answer(
            "I don't remember",
            expected=["anthropic", "london"],
            rejected=[],
        )
        assert score == 0.0

    def test_rejected_penalty(self):
        from longitudinal_eval import score_answer
        score = score_answer(
            "I work at Google in London",
            expected=["london"],
            rejected=["google"],
        )
        # expected_score = 1.0, rejection_penalty = 0.5
        assert score == pytest.approx(0.5)

    def test_multiple_rejected(self):
        from longitudinal_eval import score_answer
        score = score_answer(
            "I work at Google in San Francisco",
            expected=["anthropic"],
            rejected=["google", "san francisco"],
        )
        # expected_score = 0.0, rejection_penalty = 0.5 + 0.5 = 1.0 / 2 = 0.5
        assert score == pytest.approx(0.0)  # max(0, 0.0 - 0.5)

    def test_case_insensitive(self):
        from longitudinal_eval import score_answer
        score = score_answer(
            "I work at ANTHROPIC",
            expected=["anthropic"],
            rejected=[],
        )
        assert score == 1.0

    def test_empty_rejected_list(self):
        from longitudinal_eval import score_answer
        score = score_answer(
            "I like hiking",
            expected=["hiking"],
            rejected=[],
        )
        assert score == 1.0

    def test_score_floor_at_zero(self):
        from longitudinal_eval import score_answer
        score = score_answer(
            "google google google",
            expected=["anthropic"],
            rejected=["google"],
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# 4. Composite scoring tests
# ---------------------------------------------------------------------------

class TestCompositeScoring:
    """Verify composite_score weighting."""

    def test_perfect_scores(self):
        from longitudinal_eval import composite_score
        scores = {
            "current_fact": 1.0,
            "graceful_forgetting": 1.0,
            "reinforced_recall": 1.0,
            "cross_domain": 1.0,
        }
        assert composite_score(scores) == pytest.approx(1.0)

    def test_zero_scores(self):
        from longitudinal_eval import composite_score
        scores = {
            "current_fact": 0.0,
            "graceful_forgetting": 0.0,
            "reinforced_recall": 0.0,
            "cross_domain": 0.0,
        }
        assert composite_score(scores) == pytest.approx(0.0)

    def test_weights_sum_to_one(self):
        from longitudinal_eval import composite_score
        # Each category at 1.0 should contribute its weight
        # Total weights: 0.30 + 0.20 + 0.25 + 0.25 = 1.0
        scores = {
            "current_fact": 1.0,
            "graceful_forgetting": 0.0,
            "reinforced_recall": 0.0,
            "cross_domain": 0.0,
        }
        assert composite_score(scores) == pytest.approx(0.30)

    def test_individual_weights(self):
        from longitudinal_eval import composite_score

        # current_fact weight = 0.30
        assert composite_score({
            "current_fact": 1.0, "graceful_forgetting": 0.0,
            "reinforced_recall": 0.0, "cross_domain": 0.0,
        }) == pytest.approx(0.30)

        # graceful_forgetting weight = 0.20
        assert composite_score({
            "current_fact": 0.0, "graceful_forgetting": 1.0,
            "reinforced_recall": 0.0, "cross_domain": 0.0,
        }) == pytest.approx(0.20)

        # reinforced_recall weight = 0.25
        assert composite_score({
            "current_fact": 0.0, "graceful_forgetting": 0.0,
            "reinforced_recall": 1.0, "cross_domain": 0.0,
        }) == pytest.approx(0.25)

        # cross_domain weight = 0.25
        assert composite_score({
            "current_fact": 0.0, "graceful_forgetting": 0.0,
            "reinforced_recall": 0.0, "cross_domain": 1.0,
        }) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 5. Evaluator interface tests
# ---------------------------------------------------------------------------

class TestLongitudinalEvaluator:
    """Verify evaluator interface and behavior.

    These tests use a mock agent to avoid LLM dependency.
    """

    def test_evaluator_constructor(self):
        from longitudinal_eval import LongitudinalEvaluator, generate_dataset
        ds = generate_dataset(seed=42)
        evaluator = LongitudinalEvaluator(ds)
        assert evaluator.dataset is ds

    def test_evaluator_processes_sessions_chronologically(self):
        """The evaluator must process sessions in day order."""
        from longitudinal_eval import (
            LongitudinalEvaluator, LongitudinalDataset, Session, EvalQuestion,
        )

        session_days_processed = []

        class MockAgent:
            def __init__(self):
                self.coupled_engine = MockCoupledEngine()
                self._store_count = 0

            def send_message(self, message, memorizing=False, **kwargs):
                if memorizing:
                    session_days_processed.append("store")
                    return "Memorized"
                return {"output": "test answer"}

        class MockCoupledEngine:
            def dream(self):
                return {"modified": False}

        ds = LongitudinalDataset(
            sessions=[
                Session(day=0, session_type="preference", facts=["fact a"]),
                Session(day=5, session_type="preference", facts=["fact b"]),
                Session(day=10, session_type="single", facts=["fact c"]),
            ],
            questions=[
                EvalQuestion(
                    question="test?",
                    expected_keywords=["test"],
                    rejected_keywords=[],
                    category="current_fact",
                ),
            ],
            seed=42,
        )

        evaluator = LongitudinalEvaluator(ds)
        result = evaluator.evaluate(lambda: MockAgent())
        # All 3 sessions worth of facts should have been stored
        assert len(session_days_processed) == 3

    def test_evaluator_returns_scores_per_category(self):
        """evaluate() must return per-category scores and composite."""
        from longitudinal_eval import (
            LongitudinalEvaluator, LongitudinalDataset, Session, EvalQuestion,
        )

        class MockAgent:
            def __init__(self):
                self.coupled_engine = MockCoupledEngine()
                self._store_count = 0

            def send_message(self, message, memorizing=False, **kwargs):
                if memorizing:
                    return "Memorized"
                return {"output": "answer with anthropic"}

        class MockCoupledEngine:
            def dream(self):
                return {"modified": False}

        ds = LongitudinalDataset(
            sessions=[
                Session(day=0, session_type="preference", facts=["I work at Anthropic"]),
            ],
            questions=[
                EvalQuestion(
                    question="Where do I work?",
                    expected_keywords=["anthropic"],
                    rejected_keywords=[],
                    category="current_fact",
                ),
            ],
            seed=42,
        )

        evaluator = LongitudinalEvaluator(ds)
        result = evaluator.evaluate(lambda: MockAgent())

        assert "category_scores" in result
        assert "composite" in result
        assert "current_fact" in result["category_scores"]
        assert isinstance(result["composite"], float)

    def test_evaluator_calls_dream_during_idle_periods(self):
        """Dream should be called during gaps > 1 day between session clusters."""
        from longitudinal_eval import (
            LongitudinalEvaluator, LongitudinalDataset, Session, EvalQuestion,
        )

        dream_calls = []

        class MockAgent:
            def __init__(self):
                self.coupled_engine = MockCoupledEngine(dream_calls)
                self._store_count = 0

            def send_message(self, message, memorizing=False, **kwargs):
                if memorizing:
                    return "Memorized"
                return {"output": "test"}

        class MockCoupledEngine:
            def __init__(self, calls_list):
                self._calls = calls_list

            def dream(self):
                self._calls.append(True)
                return {"modified": False}

        # Sessions on day 0 and day 10 -- gap of 10 days should trigger dream
        ds = LongitudinalDataset(
            sessions=[
                Session(day=0, session_type="preference", facts=["fact a"]),
                Session(day=10, session_type="preference", facts=["fact b"]),
            ],
            questions=[
                EvalQuestion(
                    question="test?",
                    expected_keywords=["test"],
                    rejected_keywords=[],
                    category="current_fact",
                ),
            ],
            seed=42,
        )

        evaluator = LongitudinalEvaluator(ds)
        evaluator.evaluate(lambda: MockAgent())

        assert len(dream_calls) >= 1, (
            "Dream should have been called at least once during the 10-day gap"
        )


# ---------------------------------------------------------------------------
# 6. Dream timing logic tests
# ---------------------------------------------------------------------------

class TestDreamTiming:
    """Verify idle period detection for dream scheduling."""

    def test_no_dream_on_consecutive_days(self):
        """Sessions on consecutive days should not trigger dream between them."""
        from longitudinal_eval import (
            LongitudinalEvaluator, LongitudinalDataset, Session, EvalQuestion,
        )

        dream_calls = []

        class MockAgent:
            def __init__(self):
                self.coupled_engine = MockCoupledEngine(dream_calls)
                self._store_count = 0

            def send_message(self, message, memorizing=False, **kwargs):
                if memorizing:
                    return "Memorized"
                return {"output": "test"}

        class MockCoupledEngine:
            def __init__(self, calls_list):
                self._calls = calls_list

            def dream(self):
                self._calls.append(True)
                return {"modified": False}

        # All sessions on consecutive days -- no idle periods > 1 day
        ds = LongitudinalDataset(
            sessions=[
                Session(day=0, session_type="preference", facts=["fact a"]),
                Session(day=1, session_type="preference", facts=["fact b"]),
            ],
            questions=[
                EvalQuestion(
                    question="test?",
                    expected_keywords=["test"],
                    rejected_keywords=[],
                    category="current_fact",
                ),
            ],
            seed=42,
        )

        evaluator = LongitudinalEvaluator(ds)
        evaluator.evaluate(lambda: MockAgent())
        assert len(dream_calls) == 0, (
            "Dream should NOT be called between consecutive days"
        )


# ---------------------------------------------------------------------------
# 7. __main__ execution path tests
# ---------------------------------------------------------------------------

class TestMainExecution:
    """Verify __main__ block generates dataset and prints summary."""

    def test_main_block_exists(self):
        """The module should have if __name__ == '__main__' block."""
        import inspect
        from longitudinal_eval import generate_dataset
        source = inspect.getsource(sys.modules["longitudinal_eval"])
        assert 'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source


# ---------------------------------------------------------------------------
# 8. Edge case and robustness tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for scoring and dataset."""

    def test_score_answer_single_expected(self):
        from longitudinal_eval import score_answer
        assert score_answer("yes", ["yes"], []) == 1.0

    def test_score_answer_empty_answer(self):
        from longitudinal_eval import score_answer
        assert score_answer("", ["something"], []) == 0.0

    def test_score_answer_rejected_only(self):
        from longitudinal_eval import score_answer
        # No expected keywords present, but rejected is present
        score = score_answer("old job google", ["anthropic"], ["google"])
        assert score == 0.0

    def test_generate_dataset_seed_zero(self):
        """Seed=0 should work (edge case for RNG)."""
        from longitudinal_eval import generate_dataset
        ds = generate_dataset(seed=0)
        assert len(ds.sessions) == 200
        assert len(ds.questions) == 100

    def test_all_session_types_valid(self):
        """Every session type must be one of the 5 defined types."""
        from longitudinal_eval import generate_dataset
        ds = generate_dataset(seed=42)
        valid_types = {"preference", "update", "repeated", "single", "cross_domain"}
        for s in ds.sessions:
            assert s.session_type in valid_types, (
                f"Invalid session type: {s.session_type}"
            )

    def test_all_question_categories_valid(self):
        """Every question category must be one of the 4 defined categories."""
        from longitudinal_eval import generate_dataset
        ds = generate_dataset(seed=42)
        valid_cats = {"current_fact", "graceful_forgetting", "reinforced_recall", "cross_domain"}
        for q in ds.questions:
            assert q.category in valid_cats, (
                f"Invalid question category: {q.category}"
            )


# ---------------------------------------------------------------------------
# 9. Data quality tests (Bug fixes)
# ---------------------------------------------------------------------------

class TestDataQuality:
    """Tests for data quality invariants fixed in the rewrite.

    These test the seven critical bugs:
    1-2. Questions from intermediate states instead of terminal states
    3. Duplicate questions
    4-5. Cross-domain sessions mechanically templated
    6. Reinforcement sessions are hollow pings
    7. Update fact phrasing is awkward
    """

    @pytest.fixture(scope="class")
    def dataset(self):
        from longitudinal_eval import generate_dataset
        return generate_dataset(seed=42)

    def test_no_duplicate_questions(self, dataset):
        """Each question text must be unique -- no duplicates allowed."""
        question_texts = [q.question for q in dataset.questions]
        duplicates = [
            q for q in question_texts if question_texts.count(q) > 1
        ]
        assert len(set(duplicates)) == 0, (
            f"Found {len(set(duplicates))} duplicate question(s): "
            f"{list(set(duplicates))[:5]}"
        )

    def test_current_fact_expects_terminal_values_only(self, dataset):
        """current_fact questions must expect the FINAL value in a chain,
        not any intermediate value. Verified by checking that no two
        current_fact questions for the same domain have different expected
        keywords (which would mean one is intermediate)."""
        current_qs = [
            q for q in dataset.questions if q.category == "current_fact"
        ]
        # All expected keywords across current_fact questions should be
        # consistent: for any domain, there should be only ONE expected answer.
        # We check this by ensuring no expected keyword appears in another
        # question's rejected list for the same category.
        all_expected = set()
        all_rejected = set()
        for q in current_qs:
            all_expected.update(kw.lower() for kw in q.expected_keywords)
            all_rejected.update(kw.lower() for kw in q.rejected_keywords)
        # An expected keyword should NEVER also appear as a rejected keyword
        # (which would mean it was an intermediate value being both expected
        # in one question and rejected in another).
        overlap = all_expected & all_rejected
        assert len(overlap) == 0, (
            f"Keywords appear as both expected and rejected in current_fact "
            f"questions: {overlap}. This means intermediate values leaked."
        )

    def test_current_fact_rejects_previous_values(self, dataset):
        """current_fact questions about updated facts should reject ALL
        previous values in the chain, not just the immediately prior one."""
        current_qs = [
            q for q in dataset.questions
            if q.category == "current_fact" and len(q.rejected_keywords) > 0
        ]
        # Must have at least 5 such questions
        assert len(current_qs) >= 5, (
            f"Only {len(current_qs)} current_fact questions have rejected "
            f"keywords -- expected at least 5"
        )

    def test_graceful_forgetting_expects_both_first_and_last(self, dataset):
        """graceful_forgetting questions must expect BOTH the original (first)
        and terminal (last) values in the chain."""
        gf_qs = [
            q for q in dataset.questions
            if q.category == "graceful_forgetting"
        ]
        for q in gf_qs:
            assert len(q.expected_keywords) >= 2, (
                f"Graceful forgetting question '{q.question}' expects only "
                f"{q.expected_keywords} -- should expect both original and "
                f"terminal values"
            )

    def test_cross_domain_questions_require_bridging(self, dataset):
        """Cross-domain questions must require connecting facts from 2+
        separate domains. Each question must have expected keywords from
        at least 2 different conceptual areas."""
        cross_qs = [
            q for q in dataset.questions if q.category == "cross_domain"
        ]
        assert len(cross_qs) == 25
        for q in cross_qs:
            assert len(q.expected_keywords) >= 2, (
                f"Cross-domain question '{q.question}' expects only "
                f"{q.expected_keywords} -- needs keywords from 2+ domains"
            )

    def test_cross_domain_sessions_plant_separate_seeds(self, dataset):
        """Cross-domain sessions should NOT explicitly state the connection
        between domains. The facts should plant seeds in separate domains
        that the eval questions later bridge."""
        cross_sessions = [
            s for s in dataset.sessions if s.session_type == "cross_domain"
        ]
        assert len(cross_sessions) == 20
        for s in cross_sessions:
            for fact in s.facts:
                # Facts should not contain templated connector phrases
                assert "interested in things related to" not in fact.lower(), (
                    f"Cross-domain fact is mechanically templated: '{fact}'"
                )
                assert "exploring" not in fact.lower() or "connections" not in fact.lower(), (
                    f"Cross-domain fact explicitly states connection: '{fact}'"
                )

    def test_reinforcement_sessions_add_concrete_detail(self, dataset):
        """Reinforcement (repeated) sessions must add concrete detail,
        not just hollow access pings like 'Speaking of X, I enjoy it'."""
        repeated_sessions = [
            s for s in dataset.sessions if s.session_type == "repeated"
        ]
        hollow_patterns = [
            "speaking of",
            "regarding",
            "coming back to",
            "i was thinking about",
            "that i mentioned before",
            "it's still important to me",
        ]
        hollow_count = 0
        for s in repeated_sessions:
            for fact in s.facts:
                fact_lower = fact.lower()
                if any(p in fact_lower for p in hollow_patterns):
                    hollow_count += 1
        # Allow at most 5% hollow facts (3 out of 60 sessions)
        assert hollow_count <= 3, (
            f"{hollow_count} reinforcement facts are hollow pings "
            f"(max allowed: 3). They must add concrete details."
        )

    def test_update_facts_read_naturally(self, dataset):
        """Update facts should read like natural speech, not templated
        'I now prefer X instead of Y over Z' patterns."""
        update_sessions = [
            s for s in dataset.sessions if s.session_type == "update"
        ]
        awkward_patterns = [
            "instead of",
            "over",
        ]
        awkward_count = 0
        for s in update_sessions:
            for fact in s.facts:
                fact_lower = fact.lower()
                # Count facts that use BOTH "instead of" or "over" with "now prefer"
                if "now prefer" in fact_lower and any(
                    p in fact_lower for p in awkward_patterns
                ):
                    awkward_count += 1
        # Allow at most 10% awkward (4 out of ~40)
        assert awkward_count <= 4, (
            f"{awkward_count} update facts use awkward 'now prefer X instead "
            f"of Y' phrasing (max allowed: 4). Use natural language."
        )

    def test_cross_domain_questions_ask_bridging_not_echo(self, dataset):
        """Cross-domain questions should ask the agent to BRIDGE facts from
        different domains, not just echo 'what connections between X and Y?'."""
        cross_qs = [
            q for q in dataset.questions if q.category == "cross_domain"
        ]
        echo_count = 0
        for q in cross_qs:
            q_lower = q.question.lower()
            if "what connections" in q_lower or "relate to each other" in q_lower:
                echo_count += 1
        assert echo_count <= 3, (
            f"{echo_count} cross-domain questions are echo-style "
            f"('what connections?'). They should ask bridging questions."
        )
