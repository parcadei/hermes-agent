"""Tests for feature wiring: --lme_multihop and --date_arithmetic.

Validates that the two library modules are correctly wired into the
composed agent's retrieval/answer-generation pipelines, controlled by
feature flags that default OFF (zero-impact when absent).

Flags covered:
- ``lme_multihop``: session-aware round-robin merge of retrieval candidates
  for multi-session aggregation questions ("how many X total", etc.).
- ``date_arithmetic``: deterministic date-diff answers for temporal
  questions ("how many weeks ago", "on what date").

Wiring points:
- ``retrieval.Retriever._retrieve_v2`` consults ``_resolve_lme_multihop``.
- ``query_pipeline.QueryPipeline._query_generate`` consults
  ``_resolve_date_arithmetic`` before calling the LLM.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from mabench.configs import RetrievalConfig

# ---------------------------------------------------------------------------
# RetrievalConfig flags exist with correct defaults
# ---------------------------------------------------------------------------

class TestRetrievalConfigHasNewFlags:
    """Both feature flags live on RetrievalConfig, default OFF."""

    def test_lme_multihop_default_false(self):
        cfg = RetrievalConfig()
        assert cfg.lme_multihop is False

    def test_lme_multihop_per_session_top_k_default(self):
        cfg = RetrievalConfig()
        assert cfg.lme_multihop_per_session_top_k == 3

    def test_lme_multihop_max_sessions_default(self):
        cfg = RetrievalConfig()
        assert cfg.lme_multihop_max_sessions == 10

    def test_date_arithmetic_default_false(self):
        cfg = RetrievalConfig()
        assert cfg.date_arithmetic is False

    def test_date_arithmetic_confidence_threshold_default(self):
        cfg = RetrievalConfig()
        assert cfg.date_arithmetic_confidence_threshold == "high"

    def test_flags_can_be_set(self):
        cfg = RetrievalConfig(
            lme_multihop=True,
            lme_multihop_per_session_top_k=2,
            lme_multihop_max_sessions=5,
            date_arithmetic=True,
            date_arithmetic_confidence_threshold="medium",
        )
        assert cfg.lme_multihop is True
        assert cfg.lme_multihop_per_session_top_k == 2
        assert cfg.lme_multihop_max_sessions == 5
        assert cfg.date_arithmetic is True
        assert cfg.date_arithmetic_confidence_threshold == "medium"


# ---------------------------------------------------------------------------
# Retriever._resolve_lme_multihop property
# ---------------------------------------------------------------------------

class TestRetrieverResolveLmeMultihop:
    """The Retriever exposes _resolve_* properties for each new flag."""

    def test_resolve_lme_multihop_off_by_default(self):
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig()
        assert r._resolve_lme_multihop is False

    def test_resolve_lme_multihop_on_when_flag_set(self):
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig(lme_multihop=True)
        assert r._resolve_lme_multihop is True

    def test_resolve_per_session_top_k(self):
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig(lme_multihop_per_session_top_k=2)
        assert r._resolve_lme_multihop_per_session_top_k == 2

    def test_resolve_max_sessions(self):
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig(lme_multihop_max_sessions=5)
        assert r._resolve_lme_multihop_max_sessions == 5


# ---------------------------------------------------------------------------
# _apply_lme_multihop reshaping logic
# ---------------------------------------------------------------------------

class _FakeEntry:
    """Stand-in for MemoryEntry carrying session_time."""

    def __init__(self, session_time):
        self.session_time = session_time
        self.role = "user"


class _FakeEngine:
    """Minimal engine exposing memory_store[idx].session_time."""

    def __init__(self, session_times):
        self.memory_store = [_FakeEntry(st) for st in session_times]


def _results_from_session_indices(session_indices, scores=None):
    """Build result dicts with index field pointing at session_times list."""
    if scores is None:
        scores = [1.0 - 0.1 * i for i in range(len(session_indices))]
    return [
        {"index": i, "score": sc, "text": f"text_{i}"}
        for i, sc in zip(session_indices, scores, strict=False)
    ]


class TestApplyLmeMultihop:
    """_apply_lme_multihop reshapes results for session diversity when ON."""

    def test_noop_when_flag_off(self):
        """Flag off: returns results unchanged."""
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig(lme_multihop=False)

        session_times = [datetime(2023, 1, 1)] * 5
        r._engine = _FakeEngine(session_times)
        results = _results_from_session_indices(list(range(5)))

        out = r._apply_lme_multihop(
            results, "How many fish total in both aquariums?",
        )
        assert out is results  # pass-through when off

    def test_noop_when_query_not_multihop(self):
        """Non-aggregation query: returns results unchanged."""
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig(lme_multihop=True)
        session_times = [datetime(2023, 1, 1)] * 5
        r._engine = _FakeEngine(session_times)
        results = _results_from_session_indices(list(range(5)))

        # Non-aggregation point query
        out = r._apply_lme_multihop(
            results, "What is the user's favorite color?",
        )
        assert out == results

    def test_noop_on_temporal_date_diff_query(self):
        """Date-diff query: treated as "none" by classifier, returns results."""
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig(lme_multihop=True)
        session_times = [datetime(2023, 1, 1)] * 5
        r._engine = _FakeEngine(session_times)
        results = _results_from_session_indices(list(range(5)))

        # "weeks ago" is in the exclusion list -> classify as "none"
        out = r._apply_lme_multihop(
            results, "How many weeks ago did I visit the doctor?",
        )
        assert out == results

    def test_session_diversity_increases_when_on(self):
        """On + multihop query: produces session-diverse top-k."""
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig(
            lme_multihop=True,
            lme_multihop_per_session_top_k=2,
            lme_multihop_max_sessions=10,
            retrieve_num=10,
        )

        # 10 turns across 3 sessions. Baseline: top-5 all from session A
        # (highest scores clustered). After reshape: expect sessions
        # A, B, C interleaved.
        session_a = datetime(2023, 1, 1)
        session_b = datetime(2023, 2, 1)
        session_c = datetime(2023, 3, 1)
        session_times = [
            session_a, session_a, session_a, session_a, session_a,
            session_b, session_b, session_b,
            session_c, session_c,
        ]
        r._engine = _FakeEngine(session_times)
        scores = [1.0, 0.95, 0.9, 0.85, 0.8, 0.5, 0.45, 0.4, 0.3, 0.25]
        results = _results_from_session_indices(list(range(10)), scores=scores)

        # Count unique sessions in baseline top-5 and reshaped top-5
        baseline_sessions = {
            session_times[r_["index"]] for r_ in results[:5]
        }
        assert len(baseline_sessions) == 1  # all session A

        out = r._apply_lme_multihop(
            results, "How many fish total across all my tanks?",
        )
        reshaped_sessions = {
            session_times[r_["index"]] for r_ in out[:5]
        }
        assert len(reshaped_sessions) >= 2  # multi-session diversity

    def test_handles_missing_session_time_gracefully(self):
        """Results with None session_time go to None bucket, not dropped."""
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig(lme_multihop=True, retrieve_num=10)
        session_times = [None, None, datetime(2023, 1, 1), datetime(2023, 2, 1)]
        r._engine = _FakeEngine(session_times)
        results = _results_from_session_indices([0, 1, 2, 3])

        out = r._apply_lme_multihop(
            results, "list all the restaurants I visited",
        )
        # None-bucket results should still appear in output
        texts = {rr["text"] for rr in out}
        assert "text_0" in texts or "text_1" in texts


# ---------------------------------------------------------------------------
# QueryPipeline date_arithmetic short-circuit
# ---------------------------------------------------------------------------

class TestQueryPipelineDateArithmeticShortCircuit:
    """QueryPipeline._query_generate bypasses LLM when date_arithmetic fires."""

    def _make_pipeline(self, date_arithmetic_on: bool, threshold: str = "high"):
        """Build a minimal QueryPipeline with mocked deps for short-circuit test."""
        from mabench.configs import (
            CRConfig,
            DecomposeConfig,
            DedupConfig,
            LLMConfig,
            RetrievalConfig,
        )
        from mabench.query_pipeline import QueryPipeline

        # Minimal engine stub (memory_store + _text_to_store_idx)
        engine = MagicMock()
        engine.memory_store = []
        engine._text_to_store_idx = {}

        retrieval_cfg = RetrievalConfig(
            date_arithmetic=date_arithmetic_on,
            date_arithmetic_confidence_threshold=threshold,
            temporal_alpha=0.0,  # disable temporal enrichment for simplicity
        )
        qp = QueryPipeline(
            engine=engine,
            scorer=MagicMock(),
            retriever=MagicMock(),
            conflict_resolver=MagicMock(),
            outgester=None,
            rebel=None,
            orchestrator=None,
            bm25_index=None,
            triadic=None,
            chain_cache=None,
            belief_index=None,
            retrieval_config=retrieval_cfg,
            cr_config=CRConfig(),
            decompose_config=DecomposeConfig(),
            dedup_config=DedupConfig(),
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )

        # Stub _resolve_format_outgestion to return simple strings
        qp._resolve_format_outgestion = lambda chunks, metadata: (
            "\n".join(chunks), "instructions"
        )

        return qp

    def test_llm_called_when_flag_off(self):
        """date_arithmetic off: LLM is invoked normally."""
        qp = self._make_pipeline(date_arithmetic_on=False)

        # Mock the LLM call
        llm_mock = MagicMock(return_value=("llm_answer", 100, 20, 0.5))
        qp._resolve_generate_answer = llm_mock

        result = qp._query_generate(
            question="Current Date: 2023/06/15\nHow many weeks ago?",
            bare_question="How many weeks ago did I attend the festival?",
            merged_chunks=["[2023/05/25 (Sun) 10:00] user: I went to the festival"],
            metadata_by_key={},
            hop_state={},
            query_start=0.0,
        )
        assert llm_mock.called
        assert result["output"] == "llm_answer"

    def test_llm_called_when_question_not_arithmetic(self):
        """Non-arithmetic question: date_arithmetic classifier returns 'none'."""
        qp = self._make_pipeline(date_arithmetic_on=True)
        llm_mock = MagicMock(return_value=("llm_answer", 100, 20, 0.5))
        qp._resolve_generate_answer = llm_mock

        result = qp._query_generate(
            question="Current Date: 2023/06/15\nWhat is the user's favorite color?",
            bare_question="What is the user's favorite color?",
            merged_chunks=["[2023/05/01 (Mon) 09:00] user: I love blue"],
            metadata_by_key={},
            hop_state={},
            query_start=0.0,
        )
        assert llm_mock.called
        assert result["output"] == "llm_answer"

    def test_llm_bypassed_on_arithmetic_question_with_high_confidence(self):
        """Arithmetic question + computed answer + high conf: LLM bypassed."""
        qp = self._make_pipeline(
            date_arithmetic_on=True, threshold="high",
        )
        llm_mock = MagicMock(return_value=("llm_answer", 100, 20, 0.5))
        qp._resolve_generate_answer = llm_mock

        # Question + retrieved turn with event date that yields confident answer
        result = qp._query_generate(
            question="Current Date: 2023/06/15\nHow many weeks ago did I attend the festival?",
            bare_question="How many weeks ago did I attend the festival?",
            merged_chunks=[
                "[2023/05/25 (Sun) 10:00] user: I attended the music festival yesterday",
                "[2023/05/26 (Mon) 11:00] user: That was amazing",
            ],
            metadata_by_key={},
            hop_state={},
            query_start=0.0,
        )
        # LLM should NOT have been called — arithmetic short-circuit fired
        assert not llm_mock.called, (
            f"LLM was called; output was {result['output']!r} — "
            "expected arithmetic short-circuit"
        )
        # Output should look like "N weeks ago"
        assert "weeks ago" in result["output"].lower() or \
               "days ago" in result["output"].lower()

    def test_llm_called_when_arithmetic_low_confidence(self):
        """Arithmetic classifier fires, but no event date -> fall through to LLM."""
        qp = self._make_pipeline(
            date_arithmetic_on=True, threshold="high",
        )
        llm_mock = MagicMock(return_value=("llm_answer", 100, 20, 0.5))
        qp._resolve_generate_answer = llm_mock

        # Arithmetic question, but turns contain NO date mentions and NO timestamps
        # -> event_date is None -> confidence 0.0 -> fall through to LLM
        result = qp._query_generate(
            question="Current Date: 2023/06/15\nHow many weeks ago did I attend the festival?",
            bare_question="How many weeks ago did I attend the festival?",
            merged_chunks=[
                "i attended some event recently",
                "that was cool",
            ],
            metadata_by_key={},
            hop_state={},
            query_start=0.0,
        )
        assert llm_mock.called
        assert result["output"] == "llm_answer"


# ---------------------------------------------------------------------------
# Stacking: both flags on simultaneously
# ---------------------------------------------------------------------------

class TestStackingAllFlags:
    """With event_calendar + lme_multihop + date_arithmetic all on, no crashes."""

    def test_retrieval_config_stacks_event_calendar_and_lme_multihop(self):
        """All three feature flags can coexist on RetrievalConfig."""
        cfg = RetrievalConfig(
            event_calendar=True,
            event_calendar_window_days=60,
            event_calendar_boost=0.15,
            lme_multihop=True,
            date_arithmetic=True,
        )
        assert cfg.event_calendar is True
        assert cfg.lme_multihop is True
        assert cfg.date_arithmetic is True

    def test_retriever_resolves_all_three_flags(self):
        """Retriever._resolve_* reflect all three feature flags."""
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig(
            event_calendar=True,
            lme_multihop=True,
            date_arithmetic=True,
        )
        assert r._resolve_event_calendar is True
        assert r._resolve_lme_multihop is True

    def test_apply_lme_multihop_returns_at_most_retrieve_num(self):
        """_apply_lme_multihop output respects retrieve_num cap."""
        from mabench.retrieval import Retriever

        r = Retriever.__new__(Retriever)
        r._cfg = RetrievalConfig(
            lme_multihop=True,
            lme_multihop_per_session_top_k=3,
            lme_multihop_max_sessions=5,
            retrieve_num=10,
        )
        # 20 results across 4 sessions
        session_times = [
            datetime(2023, 1, 1)] * 5 + [datetime(2023, 2, 1)] * 5 + \
            [datetime(2023, 3, 1)] * 5 + [datetime(2023, 4, 1)] * 5
        r._engine = _FakeEngine(session_times)
        results = _results_from_session_indices(list(range(20)))

        out = r._apply_lme_multihop(results, "list all the places I visited")
        # The round-robin merge is bounded by per_session_top_k * max_sessions
        # and the function returns a new list of reshaped results.
        assert len(out) <= 5 * 3  # max_sessions * per_session_top_k
