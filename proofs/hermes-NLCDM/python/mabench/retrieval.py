"""Core retrieval pipeline: embedding, candidate scoring, post-retrieval processing.

Converted from RetrievalMixin to standalone Retriever class.
All self.* attribute access replaced with explicit config + instance fields.

Migration notes for callers:
  - RetrievalMixin is a backward-compat alias for Retriever.
  - Mixin callers (hermes_agent.py) still work via MRO -- the alias
    keeps the import path stable.  HermesMemoryAgent.__init__ sets
    self.coupled_engine / self._scorer / self._belief_index / etc. directly;
    the _resolve_* properties fall back to those when _cfg is absent.
  - Standalone usage:
      from mabench.retrieval import Retriever
      from mabench.configs import RetrievalConfig
      retriever = Retriever(engine, scorer, config=RetrievalConfig(cooc_boost=True))
      results = retriever.retrieve("What sport does Abbiati play?")
"""

from __future__ import annotations

import re

import numpy as np

from mabench.configs import BeliefConfig, CRConfig, DedupConfig, RetrievalConfig

import json as _json
import logging as _logging
import os as _os

_log = _logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM query-term extraction (replaces regex extract_entities for EventDB)
# ---------------------------------------------------------------------------

_LLM_QUERY_PROMPT = """\
Given this question about a user's personal history, extract search terms \
that would match entities stored in a conversation knowledge graph. \
The graph stores (subject, predicate, object) triples extracted from chat messages.

Return a JSON object: {{"search_terms": ["term1", "term2", ...]}}

Rules:
- Include the key nouns, named entities, and concepts from the question.
- Add plausible synonyms or related terms (e.g., "cat" -> also "pet", "feline").
- Include compound terms when relevant (e.g., "bedroom walls", "BBQ sauce").
- Do NOT include generic words like "the", "a", "what", "is", "my", "your".
- Return 3-8 terms. Fewer is fine if the question is simple.

Question: {question}"""

# Module-level singleton client (lazy-initialized)
_query_llm_client = None


def _get_query_llm_client():
    global _query_llm_client
    if _query_llm_client is None:
        from openai import OpenAI
        _query_llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=_os.environ["OPENROUTER_API_KEY"],
        )
    return _query_llm_client


def llm_extract_query_terms(question: str) -> list[str]:
    """Use gpt-4o-mini to extract search terms from a question for EventDB lookup.

    Returns a list of lowercased search terms, or an empty list on failure.
    Cost: ~$0.001/call, ~0.5s latency.
    """
    if not question or not question.strip():
        return []
    try:
        client = _get_query_llm_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _LLM_QUERY_PROMPT.format(question=question.strip())}],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        # Parse JSON — handle markdown fenced blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = _json.loads(text)
        terms = data.get("search_terms", [])
        return [t.lower().strip() for t in terms if isinstance(t, str) and len(t.strip()) >= 2]
    except Exception as exc:
        _log.warning("llm_extract_query_terms failed: %s", exc)
        return []


# Query-time Current Date marker. Matches the LME evaluator's prompt format
# ("Current Date: 2023/06/15" or with trailing "(Day) HH:MM"). We only need
# year/month/day — intra-day resolution is not required for the calendar
# lookup window.
_CURRENT_DATE_RE = re.compile(
    r"Current\s*Date\s*:\s*(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})",
    re.IGNORECASE,
)


def _parse_current_date_from_query(query_text: str):
    """Extract (year, month, day) as a ``date`` from a ``Current Date:`` marker.

    Returns None when no marker is present or the fields don't form a valid
    date. The caller uses this to decide whether to consult the calendar.
    """
    if not query_text:
        return None
    m = _CURRENT_DATE_RE.search(query_text)
    if not m:
        return None
    try:
        from datetime import date as _date
        return _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except (ValueError, TypeError):
        return None


def _apply_event_calendar_boost(
    scored_docs: list[dict],
    query_text: str,
    calendar,
    window_days: int,
    boost: float,
) -> list[dict]:
    """Additively boost scored candidates whose turn_idx is in the calendar window.

    Neutral (pass-through, no copy) when any of the following hold:
      - ``calendar`` is None or empty,
      - ``boost`` is 0.0,
      - the query has no ``Current Date: YYYY/MM/DD`` marker,
      - ``calendar.lookup_window(q_date, window_days)`` returns no events.

    When active, each doc whose ``index`` is in the window's event turn_idx
    set receives exactly one additive boost on its current score. The result
    list is resorted descending by score.

    Args:
        scored_docs: list of retrieval hits, each with ``index`` and ``score``.
        query_text: raw query, expected to carry a ``Current Date:`` marker.
        calendar: EventCalendarLME instance (or None).
        window_days: half-window (±days) passed to ``calendar.lookup_window``.
        boost: additive delta applied to each in-window doc's score.

    Returns:
        A new list of dicts (sorted by adjusted score), or the original list
        unchanged when the pass-through conditions trigger.
    """
    # Pass-through guards (keep neutral when data is missing)
    if not scored_docs:
        return scored_docs
    if calendar is None or boost == 0.0:
        return scored_docs
    if len(calendar) == 0:
        return scored_docs

    q_date = _parse_current_date_from_query(query_text)
    if q_date is None:
        return scored_docs

    events = calendar.lookup_window(q_date, window_days)
    if not events:
        return scored_docs

    # Collect in-window turn idxs; dedupe so a turn is boosted at most once
    # regardless of how many event mentions it contributed.
    window_turn_idxs = {ev.turn_idx for ev in events}
    if not window_turn_idxs:
        return scored_docs

    boosted: list[dict] = []
    for doc in scored_docs:
        idx = doc.get("index")
        if idx is None:
            idx = doc.get("turn_idx")
        adjusted = dict(doc)
        if idx is not None and idx in window_turn_idxs:
            adjusted["score"] = doc.get("score", 0.0) + boost
        boosted.append(adjusted)
    boosted.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return boosted


# Aggregation query heuristic: queries asking to count/sum/list across sessions
# need more retrieval context. Tested on LME: 80% of multi-session questions
# match these patterns (see analyze_agg_questions.py for derivation).
_AGG_PATTERNS = [
    r"\bhow many\b", r"\btotal\b", r"\bnumber of\b", r"\bcount\b",
    r"\blist all\b", r"\ball the\b", r"\bhow often\b", r"\bhow much\b",
    r"\bfrequently\b", r"\bsum\b", r"\baltogether\b", r"\bin total\b",
]
# Exclusions: temporal date-diff questions ("how many weeks ago", "between X and Y")
# These ask for date arithmetic, not counting — more chunks hurts by adding date noise.
_AGG_EXCL_PATTERNS = [
    r"\b(days?|weeks?|months?|years?|hours?|minutes?)\s+ago\b",
    r"\bdays?\s+(had\s+)?passed\b",
    r"\bbetween\s+.{1,60}?\s+and\b",
    r"\bsince\s+(when|that|the|I|my|we|you)\b",
    r"\bhow\s+long\s+ago\b",
    r"\b(have|has|had)\s+passed\s+since\b",
]
_AGG_RE = re.compile("|".join(_AGG_PATTERNS), re.I)
_AGG_EXCL_RE = re.compile("|".join(_AGG_EXCL_PATTERNS), re.I)


def is_aggregation_query(query_text: str) -> bool:
    """Return True if the query requires aggregating facts across turns.

    Aggregation queries (how many, total, count, list all) benefit from
    more retrieval context because the LLM needs to see ALL relevant
    mentions, not just the top-k most similar. Excludes temporal date-diff
    patterns ("how many weeks ago") which ask for date arithmetic, not counts.
    """
    if not query_text:
        return False
    if not _AGG_RE.search(query_text):
        return False
    return not _AGG_EXCL_RE.search(query_text)


class Retriever:
    """Standalone retrieval pipeline: embed, retrieve candidates, post-process.

    Can be used independently or composed into HermesMemoryAgent.

    Usage::

        retriever = Retriever(engine, scorer, config=RetrievalConfig(cooc_boost=True))
        results = retriever.retrieve("What sport does Abbiati play?")
    """

    def __init__(
        self,
        engine=None,
        scorer=None,
        belief_index=None,
        config: RetrievalConfig | None = None,
        cr_config: CRConfig | None = None,
        dedup_config: DedupConfig | None = None,
        belief_config: BeliefConfig | None = None,
        nlp=None,
        entity_reranker=None,
        cross_encoder_reranker=None,
        bm25_index=None,
        structured_store=None,
    ) -> None:
        # _engine: CoupledEngine with .memory_store, .query(), .query_cooc_boost(), etc.
        self._engine = engine
        # _scorer.embed(text) -> np.ndarray of shape (dim,)
        self._scorer = scorer
        # _belief_index: BeliefIndex with .is_excluded(text)/.score(text), or None
        self._belief_index = belief_index
        # _nlp: spaCy Language model for entity extraction
        self._nlp = nlp
        # _entity_reranker: EntityReranker instance, or None
        self._entity_reranker = entity_reranker
        # _cross_encoder_reranker: CrossEncoderReranker instance, or None
        self._cross_encoder_reranker = cross_encoder_reranker
        # _bm25_index: BM25Index for lexical reranking
        self._bm25_index = bm25_index
        # _structured_store: StructuredFactStore for O(1) entity+predicate lookup, or None
        self._structured_store = structured_store

        # _cfg: RetrievalConfig(retrieve_num=10, cooc_boost=False, ...)
        self._cfg = config or RetrievalConfig()
        # _cr_cfg: CRConfig(enabled=False, overfetch=3)
        self._cr_cfg = cr_config or CRConfig()
        # _dedup_cfg: DedupConfig(enabled=False, threshold=0.95)
        self._dedup_cfg = dedup_config or DedupConfig()
        # _belief_cfg: BeliefConfig(enabled=False, overfetch=3)
        self._belief_cfg = belief_config or BeliefConfig()

    # ------------------------------------------------------------------
    # Attribute resolution: new config or old self.* fallback
    # ------------------------------------------------------------------
    # Pattern: hasattr(self, '_cfg') distinguishes standalone (has _cfg)
    # from mixin mode (no _cfg, uses raw self.X set by host class).

    @property
    def _resolve_engine(self):
        """CoupledEngine -- from __init__ or host class self.coupled_engine."""
        if hasattr(self, "_engine") and self._engine is not None:
            return self._engine
        return self.coupled_engine  # type: ignore[attr-defined]

    @property
    def _resolve_scorer(self):
        """EmbeddingRelevanceScorer -- from __init__ or host class self._scorer."""
        if hasattr(self, "_cfg") and hasattr(self, "_scorer") and self._scorer is not None:
            return self._scorer
        # In mixin mode, _scorer is set directly by host __init__
        return self._scorer  # type: ignore[has-type]

    @property
    def _resolve_belief_index(self):
        """BeliefIndex or None -- from __init__ or host class self._belief_index."""
        if hasattr(self, "_cfg"):
            return self._belief_index
        return self._belief_index  # type: ignore[has-type]

    @property
    def _resolve_retrieve_num(self) -> int:
        """Base number of results to retrieve."""
        if hasattr(self, "_cfg"):
            return self._cfg.retrieve_num
        return self.retrieve_num  # type: ignore[attr-defined]

    @property
    def _resolve_conflict_resolution(self) -> bool:
        """Whether conflict resolution is active."""
        if hasattr(self, "_cr_cfg"):
            return self._cr_cfg.enabled
        return getattr(self, "conflict_resolution", False)

    @property
    def _resolve_cosine_dedup(self) -> bool:
        """Whether cosine deduplication is active."""
        if hasattr(self, "_dedup_cfg"):
            return self._dedup_cfg.enabled
        return getattr(self, "cosine_dedup", False)

    @property
    def _resolve_cr_overfetch(self) -> int:
        """CR overfetch multiplier."""
        if hasattr(self, "_cr_cfg"):
            return self._cr_cfg.overfetch
        return getattr(self, "_cr_overfetch", 3)

    @property
    def _resolve_belief_overfetch(self) -> int:
        """Belief overfetch multiplier."""
        if hasattr(self, "_belief_cfg"):
            return self._belief_cfg.overfetch
        return getattr(self, "_belief_overfetch", 3)

    @property
    def _resolve_transfer(self) -> bool:
        """Whether transfer retrieval is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.transfer
        return getattr(self, "transfer_retrieval", False)

    @property
    def _resolve_coretrieval(self) -> bool:
        """Whether coretrieval is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.coretrieval
        return getattr(self, "coretrieval_retrieval", False)

    @property
    def _resolve_cooc_boost(self) -> bool:
        """Whether co-occurrence boosted cosine is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.cooc_boost
        return getattr(self, "cooc_boost_retrieval", False)

    @property
    def _resolve_ppr(self) -> bool:
        """Whether personalized PageRank is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.ppr
        return getattr(self, "ppr_retrieval", False)

    @property
    def _resolve_hybrid(self) -> bool:
        """Whether hybrid retrieval is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.hybrid
        return getattr(self, "hybrid_retrieval", False)

    @property
    def _resolve_associative(self) -> bool:
        """Whether associative retrieval is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.associative
        return getattr(self, "associative_retrieval", False)

    @property
    def _resolve_sparse(self) -> bool:
        """Whether sparse retrieval is active."""
        if hasattr(self, "_cfg"):
            return self._cfg.sparse
        return getattr(self, "sparse_retrieval", False)

    @property
    def _resolve_transfer_k(self) -> int | None:
        """Transfer k parameter."""
        if hasattr(self, "_cfg"):
            return self._cfg.transfer_k
        return getattr(self, "transfer_k", None)

    @property
    def _resolve_coretrieval_bonus(self) -> float:
        """Coretrieval bonus parameter."""
        if hasattr(self, "_cfg"):
            return self._cfg.coretrieval_bonus
        return getattr(self, "coretrieval_bonus", 0.3)

    @property
    def _resolve_coretrieval_min_count(self) -> float:
        """Coretrieval minimum count parameter."""
        if hasattr(self, "_cfg"):
            return self._cfg.coretrieval_min_count
        return getattr(self, "coretrieval_min_count", 2.0)

    @property
    def _resolve_cooc_weight(self) -> float:
        """Co-occurrence weight parameter."""
        if hasattr(self, "_cfg"):
            return self._cfg.cooc_weight
        return getattr(self, "cooc_weight", 0.3)

    @property
    def _resolve_cooc_gate_threshold(self) -> float:
        """Co-occurrence gate threshold."""
        if hasattr(self, "_cfg"):
            return self._cfg.cooc_gate_threshold
        return getattr(self, "cooc_gate_threshold", 0.0)

    @property
    def _resolve_ppr_weight(self) -> float:
        """Personalized PageRank weight."""
        if hasattr(self, "_cfg"):
            return self._cfg.ppr_weight
        return getattr(self, "ppr_weight", 0.5)

    @property
    def _resolve_ppr_damping(self) -> float:
        """Personalized PageRank damping factor."""
        if hasattr(self, "_cfg"):
            return self._cfg.ppr_damping
        return getattr(self, "ppr_damping", 0.85)

    @property
    def _resolve_bm25_weight(self) -> float:
        """BM25 hybrid weight (0 = no BM25)."""
        if hasattr(self, "_cfg"):
            return self._cfg.bm25_weight
        return getattr(self, "bm25_weight", 0.0)

    @property
    def _resolve_dream_weight(self) -> float:
        """Dream consolidation weight."""
        if hasattr(self, "_cfg"):
            return self._cfg.dream_weight
        return getattr(self, "dream_weight", 0.0)

    @property
    def _resolve_fisher_scoring(self) -> bool:
        """Whether Fisher-Rao scoring replaces cosine base similarity."""
        if hasattr(self, "_cfg"):
            return self._cfg.fisher_scoring
        return getattr(self, "fisher_scoring", False)

    @property
    def _resolve_fisher_temperature(self) -> float:
        """Fisher-Rao temperature parameter (lower = more discriminative)."""
        if hasattr(self, "_cfg"):
            return self._cfg.fisher_temperature
        return getattr(self, "fisher_temperature", 1.0)

    @property
    def _resolve_bm25_primary(self) -> bool:
        """Whether BM25 is the primary retrieval channel (not cosine)."""
        if hasattr(self, "_cfg"):
            return self._cfg.bm25_primary
        return getattr(self, "bm25_primary", False)

    @property
    def _resolve_bm25_rerank(self) -> bool:
        """Whether to apply BM25 score blending on cosine candidates."""
        if hasattr(self, "_cfg"):
            return self._cfg.bm25_rerank
        return getattr(self, "bm25_rerank", False)

    @property
    def _resolve_bm25_rerank_weight(self) -> float:
        """BM25 contribution in rerank blend (1-w = cosine weight)."""
        if hasattr(self, "_cfg"):
            return self._cfg.bm25_rerank_weight
        return getattr(self, "bm25_rerank_weight", 0.5)

    def _resolve_entity_primary(self) -> bool:
        return self._cfg.entity_primary

    @property
    def _resolve_entity_prefilter(self) -> bool:
        """Whether to fuse entity_scored + BM25 via RRF."""
        if hasattr(self, "_cfg"):
            return self._cfg.entity_prefilter
        return getattr(self, "entity_prefilter", False)

    @property
    def _resolve_event_calendar_retrieval(self) -> bool:
        """Whether to add event calendar as 3rd RRF channel."""
        if hasattr(self, "_cfg"):
            return self._cfg.event_calendar_retrieval
        return getattr(self, "event_calendar_retrieval", False)

    @property
    def _resolve_event_calendar(self) -> bool:
        """Whether to apply the LME date-mention calendar boost post-retrieval."""
        if hasattr(self, "_cfg"):
            return self._cfg.event_calendar
        return getattr(self, "event_calendar", False)

    @property
    def _resolve_event_calendar_window_days(self) -> int:
        """Half-window (±days) for the LME event calendar lookup."""
        if hasattr(self, "_cfg"):
            return self._cfg.event_calendar_window_days
        return getattr(self, "event_calendar_window_days", 60)

    @property
    def _resolve_event_calendar_boost(self) -> float:
        """Additive score boost for in-window candidates."""
        if hasattr(self, "_cfg"):
            return self._cfg.event_calendar_boost
        return getattr(self, "event_calendar_boost", 0.15)

    @property
    def _resolve_aggregation_overfetch(self) -> int:
        """Target top_k for aggregation queries (0 = disabled)."""
        if hasattr(self, "_cfg"):
            return self._cfg.aggregation_overfetch
        return getattr(self, "aggregation_overfetch", 0)

    @property
    def _resolve_lme_multihop(self) -> bool:
        """Whether to apply session-aware round-robin merge for multi-session aggregation."""
        if hasattr(self, "_cfg"):
            return self._cfg.lme_multihop
        return getattr(self, "lme_multihop", False)

    @property
    def _resolve_lme_multihop_per_session_top_k(self) -> int:
        """Max candidates to keep from each distinct session bucket."""
        if hasattr(self, "_cfg"):
            return self._cfg.lme_multihop_per_session_top_k
        return getattr(self, "lme_multihop_per_session_top_k", 3)

    @property
    def _resolve_lme_multihop_max_sessions(self) -> int:
        """Max distinct sessions to sample from during round-robin merge."""
        if hasattr(self, "_cfg"):
            return self._cfg.lme_multihop_max_sessions
        return getattr(self, "lme_multihop_max_sessions", 10)

    @property
    def _resolve_structured_primary(self) -> bool:
        """Whether structured EventDB is the primary retrieval path."""
        if hasattr(self, "_cfg"):
            return self._cfg.structured_primary
        return getattr(self, "structured_primary", False)

    @property
    def _resolve_structured_min_results(self) -> int:
        """Min results from EventDB before falling through to BM25."""
        if hasattr(self, "_cfg"):
            return self._cfg.structured_min_results
        return getattr(self, "structured_min_results", 3)

    @property
    def _resolve_structured_rrf(self) -> bool:
        """Whether to add EventDB as RRF channel in entity_prefilter."""
        if hasattr(self, "_cfg"):
            return self._cfg.structured_rrf
        return getattr(self, "structured_rrf", False)

    def _resolve_set_union_merge(self) -> bool:
        """Whether to use set-union instead of RRF for entity+BM25 fusion."""
        if hasattr(self, "_cfg"):
            return self._cfg.set_union_merge
        return getattr(self, "set_union_merge", False)

    def _resolve_noun_chunk_retrieval(self) -> bool:
        """Whether to add noun-chunk inverted index as 3rd RRF channel."""
        if hasattr(self, "_cfg"):
            return self._cfg.noun_chunk_retrieval
        return getattr(self, "noun_chunk_retrieval", False)

    @property
    def _resolve_entity_graph(self) -> bool:
        """Whether to add entity graph as RRF channel."""
        if hasattr(self, "_cfg"):
            return self._cfg.entity_graph
        return getattr(self, "entity_graph", False)

    @property
    def _resolve_layered_graph(self) -> bool:
        """Whether to add LayeredMemory as RRF channel."""
        if self._cfg is not None:
            return self._cfg.layered_graph
        return getattr(self, "layered_graph", False)

    def _resolve_exact_words_prompt(self) -> bool:
        """Whether to add 'use exact words' instruction to answer prompts."""
        if hasattr(self, "_cfg"):
            return self._cfg.exact_words_prompt
        return getattr(self, "exact_words_prompt", False)

    # ------------------------------------------------------------------
    # Core retrieval methods
    # ------------------------------------------------------------------

    def _embed_query(self, query_text: str):
        """Embed a query string.

        Returns:
            numpy array of shape (dim,) -- the query embedding.
        """
        return self._resolve_scorer.embed(query_text)

    def _compute_effective_k(self, top_k: int) -> int:
        """Compute the effective retrieval count after overfetch multipliers.

        Over-fetches when CR, cosine dedup, belief, or cross-encoder is active
        to give filtering/reranking more candidates.

        Args:
            top_k: Base number of results before overfetch.

        Returns:
            Effective k after applying all overfetch multipliers.
        """
        effective_k = top_k
        if self._resolve_conflict_resolution and self._resolve_cr_overfetch > 1:
            effective_k = top_k * self._resolve_cr_overfetch
        if self._resolve_cosine_dedup:
            effective_k = max(effective_k, top_k * self._resolve_cr_overfetch)
        if self._resolve_belief_index is not None:
            effective_k = max(effective_k, top_k * self._resolve_belief_overfetch)
        # Cross-encoder needs 3x candidates to rerank effectively.
        # Discriminative test: BM25 top-30 → xenc top-10 = 60.0% vs 58.3% pure BM25.
        if getattr(self, "_cross_encoder_reranker", None) is not None:
            effective_k = max(effective_k, top_k * 3)
        return effective_k

    def _retrieve_candidates(
        self,
        q_emb,
        top_k: int,
        query_text: str | None = None,
    ) -> list[dict]:
        """Run the appropriate retrieval backend based on config flags.

        Computes effective_k via ``_compute_effective_k`` (CR overfetch,
        cosine dedup, belief overfetch).  The *top_k* parameter is the BASE
        retrieve_num before overfetch multipliers.

        Args:
            q_emb: Query embedding (numpy array).
            top_k: Base number of results to retrieve (before overfetch).
            query_text: Original query string, needed by the cooc+bm25
                backend.  May be ``None`` for backends that don't use it.

        Returns:
            List of result dicts with at minimum {"text": str, "score": float}.
        """
        effective_k = self._compute_effective_k(top_k)
        # Aggregation query overfetch: bump effective_k when the query asks for
        # counting/summing/listing across turns. Does not affect point queries.
        agg_k = self._resolve_aggregation_overfetch
        if agg_k > effective_k and is_aggregation_query(query_text or ""):
            effective_k = agg_k
        engine = self._resolve_engine

        # Structured primary (Chronos-style): EventDB-only lookup.
        # LLM extracts query terms → EventDB dict lookup → return directly.
        # Falls through to BM25 only when EventDB returns zero hits.
        _eventdb_scored: list[tuple[int, float]] | None = None
        if self._resolve_structured_primary and query_text:
            event_db = getattr(self, "_event_db", None)
            if event_db is not None and event_db.n_events() > 0:
                q_core = query_text
                if "Now Answer the Question:" in q_core:
                    q_core = q_core.split("Now Answer the Question:", 1)[1]
                    if "\n" in q_core:
                        q_core = q_core.split("\n")[0]
                q_entities = llm_extract_query_terms(q_core.strip())
                if q_entities:
                    q_date = _parse_current_date_from_query(query_text)
                    scored = event_db.lookup(
                        q_entities,
                        top_k=effective_k,
                        date_center=q_date,
                        date_window_days=180,
                    )
                    if scored:
                        results = []
                        for chunk_id, score in scored:
                            if chunk_id < len(engine.memory_store):
                                entry = engine.memory_store[chunk_id]
                                results.append({
                                    "text": entry.text,
                                    "score": float(score),
                                    "index": chunk_id,
                                })
                        if results:
                            return results
            # Fall through to BM25 when EventDB has no hits

        # Entity primary: regex-based entity/bigram/unigram inverted index.
        # Discriminative test (2026-04-02): 72.2% oracle match on LME,
        # +3 over BM25_top10, 0 lost.
        if self._resolve_entity_primary and query_text:
            entity_inv = getattr(self, "_entity_inv_idx", None)
            bigram_inv = getattr(self, "_bigram_inv_idx", None)
            unigram_inv = getattr(self, "_unigram_inv_idx", None)
            if entity_inv is not None:
                from mabench.entity_index import entity_retrieve_scored
                n_turns = len(engine.memory_store)
                scored = entity_retrieve_scored(
                    query_text,
                    entity_inv, bigram_inv or {}, unigram_inv or {},
                    n_turns, top_k=effective_k,
                )
                results = []
                for doc_id, score in scored:
                    if doc_id < len(engine.memory_store):
                        entry = engine.memory_store[doc_id]
                        results.append({
                            "text": entry.text,
                            "score": float(score),
                            "index": doc_id,
                        })
                return results

        # Structured store + entity prefilter: 3-way RRF of BM25 + entity + structured.
        # The structured channel adds predicate-awareness on top of entity matching.
        # When structured_store is set AND entity_prefilter + bm25_primary are active,
        # this block runs a 3-channel RRF and returns.  Falls through to 2-way
        # entity_prefilter if structured_store is None.
        if (self._structured_store is not None
                and self._resolve_entity_prefilter
                and self._resolve_bm25_primary
                and query_text):
            entity_inv = getattr(self, "_entity_inv_idx", None)
            bm25_idx = getattr(self, "_bm25_index", None)
            if entity_inv is not None and bm25_idx is not None:
                from collections import defaultdict
                from mabench.entity_index import entity_retrieve_scored
                from mabench.query_parser import parse_query as _parse_query
                bigram_inv = getattr(self, "_bigram_inv_idx", None)
                unigram_inv = getattr(self, "_unigram_inv_idx", None)
                n_turns = len(engine.memory_store)
                rrf_k = effective_k * 3

                # Channel 1: BM25
                bm25_hits = bm25_idx.query(query_text, top_k=rrf_k)
                # Channel 2: entity_scored
                entity_scored = entity_retrieve_scored(
                    query_text, entity_inv, bigram_inv or {}, unigram_inv or {},
                    n_turns, top_k=rrf_k,
                )
                # Channel 3: structured store via query_parser
                pq = _parse_query(query_text)
                struct_hits = self._structured_store.query(
                    entity_hints=pq.entity_hints,
                    predicate_hints=pq.predicate_hints,
                    object_hints=pq.object_hints,
                    top_k=rrf_k,
                )

                # 3-way RRF fusion (k=60 is standard)
                rrf_scores: dict[int, float] = defaultdict(float)
                for rank, (doc_id, _) in enumerate(bm25_hits):
                    rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
                for rank, (doc_id, _) in enumerate(entity_scored):
                    rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
                for rank, (turn_idx, _turn_text, _score) in enumerate(struct_hits):
                    rrf_scores[turn_idx] += 1.0 / (60 + rank + 1)

                fused = sorted(rrf_scores.items(), key=lambda x: -x[1])[:effective_k]
                results = []
                for doc_id, score in fused:
                    if doc_id < len(engine.memory_store):
                        entry = engine.memory_store[doc_id]
                        results.append({
                            "text": entry.text,
                            "score": float(score),
                            "index": doc_id,
                        })
                return results

        # Entity prefilter: RRF fusion of BM25 + entity_scored.
        # Disc test (2026-04-03): 60.0% hit@10 = ties xenc_bm25_rerank, 15x faster.
        # Requires bm25_primary. Falls back to pure BM25 if no entity inv idx.
        if self._resolve_entity_prefilter and self._resolve_bm25_primary and query_text:
            entity_inv = getattr(self, "_entity_inv_idx", None)
            bm25_idx = getattr(self, "_bm25_index", None)
            if entity_inv is not None and bm25_idx is not None:
                from collections import defaultdict
                from mabench.entity_index import entity_retrieve_scored
                bigram_inv = getattr(self, "_bigram_inv_idx", None)
                unigram_inv = getattr(self, "_unigram_inv_idx", None)
                n_turns = len(engine.memory_store)
                # Get both channels at 3x effective_k for RRF input
                rrf_k = effective_k * 3
                entity_scored = entity_retrieve_scored(
                    query_text, entity_inv, bigram_inv or {}, unigram_inv or {},
                    n_turns, top_k=rrf_k,
                )
                # Reduce BM25 query to key terms only (entities + content words)
                # to avoid generic chunks outranking specific ones.
                # STEP 1: Strip dynamic prompt prefix so we extract from the
                # ACTUAL question, not the 312-char wrapper.
                q_core = query_text
                if "Now Answer the Question:" in q_core:
                    q_core = q_core.split("Now Answer the Question:", 1)[1]
                    # Drop the trailing "\n\n Answer:" scaffold
                    if "\n" in q_core:
                        q_core = q_core.split("\n")[0]
                    q_core = q_core.strip()
                # STEP 2: spaCy noun-chunk extraction on clean question.
                # Falls back to regex + raw content words if spaCy unavailable.
                try:
                    from mabench.noun_chunk_index import extract_noun_chunks
                    nc_terms = extract_noun_chunks(q_core)
                    reduced_terms = set(nc_terms)
                except Exception:
                    reduced_terms = set()
                # Always supplement with regex entities (proper nouns, numbers+units)
                # and content words to catch what spaCy misses.
                from mabench.entity_index import extract_entities, _STOP
                import re as _re
                reduced_terms |= extract_entities(q_core)
                reduced_terms |= {
                    w for w in _re.findall(r"\w+", q_core.lower())
                    if len(w) >= 4 and w not in _STOP
                }
                bm25_query = " ".join(reduced_terms) if reduced_terms else query_text
                bm25_hits = bm25_idx.query(bm25_query, top_k=rrf_k)

                # --- Set-union merge (Omni-SimpleMem discovery) ---
                # BM25 keeps its ranking; entity-only results appended.
                # R(q) = BM25(q) ∪ (Entity(q) \ BM25(q))
                if self._resolve_set_union_merge:
                    bm25_ids = [doc_id for doc_id, _ in bm25_hits]
                    bm25_set = set(bm25_ids)
                    # Entity-only: docs found by entity but not BM25
                    entity_only = [
                        doc_id for doc_id, _ in entity_scored
                        if doc_id not in bm25_set
                    ]
                    # BM25 results first (in BM25 rank order), then entity-only
                    fused_ids = bm25_ids[:effective_k]
                    remaining = effective_k - len(fused_ids)
                    if remaining > 0:
                        fused_ids.extend(entity_only[:remaining])
                    fused = [(doc_id, 1.0) for doc_id in fused_ids]
                else:
                    # RRF fusion (k=60 is standard)
                    rrf_scores: dict[int, float] = defaultdict(float)
                    for rank, (doc_id, _) in enumerate(bm25_hits):
                        rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
                    for rank, (doc_id, _) in enumerate(entity_scored):
                        rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
                    # 3rd RRF channel: event calendar keyword retrieval
                    if self._resolve_event_calendar_retrieval:
                        cal = getattr(self, "_event_calendar", None)
                        if cal:
                            from mabench.event_calendar import event_calendar_retrieve
                            cal_hits = event_calendar_retrieve(
                                query_text, cal, top_k=rrf_k,
                            )
                            for rank, (doc_id, _) in enumerate(cal_hits):
                                rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
                    # Noun-chunk RRF channel: spaCy noun phrases + content lemmas.
                    # Finds chunks BM25 misses due to common-word dilution.
                    if self._resolve_noun_chunk_retrieval:
                        nc_inv = getattr(self, "_noun_chunk_inv_idx", None)
                        if nc_inv:
                            from mabench.noun_chunk_index import noun_chunk_retrieve_scored
                            nc_hits = noun_chunk_retrieve_scored(
                                query_text, nc_inv, n_turns, top_k=rrf_k,
                            )
                            for rank, (doc_id, _) in enumerate(nc_hits):
                                rrf_scores[doc_id] += 1.0 / (60 + rank + 1)
                    # EventDB RRF channel: LLM-extracted query terms → entity-keyed lookup.
                    # Adds structured triple-store hits as additional RRF signal.
                    if self._resolve_structured_rrf:
                        event_db = getattr(self, "_event_db", None)
                        if event_db is not None and event_db.n_events() > 0:
                            q_core = query_text
                            if "Now Answer the Question:" in q_core:
                                q_core = q_core.split("Now Answer the Question:", 1)[1]
                                if "\n" in q_core:
                                    q_core = q_core.split("\n")[0]
                            q_terms = llm_extract_query_terms(q_core.strip())
                            if q_terms:
                                q_date = _parse_current_date_from_query(query_text)
                                eventdb_hits = event_db.lookup(
                                    q_terms, top_k=rrf_k,
                                    date_center=q_date, date_window_days=180,
                                )
                                for rank, (chunk_id, _) in enumerate(eventdb_hits):
                                    rrf_scores[chunk_id] += 1.0 / (60 + rank + 1)
                    # EntityGraph RRF channel: per-entity knowledge graph retrieval.
                    # Resolves entity mentions in query via alias/substring matching,
                    # then returns evidence turn indices from non-superseded facts.
                    if self._resolve_entity_graph:
                        eg = getattr(self, "_entity_graph", None)
                        if eg is not None and eg.entities:
                            eg_turns = eg.retrieve_for_question(
                                query_text, max_turns=rrf_k,
                            )
                            for rank, turn_idx in enumerate(eg_turns):
                                rrf_scores[turn_idx] += 1.0 / (60 + rank + 1)
                    # LayeredMemory: post-retrieval context expansion.
                    # Instead of RRF (which dilutes), use entity graph AFTER
                    # retrieval to find connected turns via entity links.
                    # Applied below after fused results are built.
                    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])[:effective_k]
                results = []
                for doc_id, score in fused:
                    if doc_id < len(engine.memory_store):
                        entry = engine.memory_store[doc_id]
                        results.append({
                            "text": entry.text,
                            "score": float(score),
                            "index": doc_id,
                        })

                # Post-retrieval entity expansion: find entities in retrieved
                # chunks, then pull in connected turns from the entity graph.
                # Replaces retrieved-set items only if expansion finds turns
                # that RRF missed entirely.
                if self._resolve_layered_graph and results:
                    lm = getattr(self, "_layered_memory", None)
                    if lm is not None and lm._entity_bm25._n > 0:
                        lm_hits = lm._entity_bm25.query(q_core, top_k=10)
                        if lm_hits:
                            retrieved_ids = {r["index"] for r in results}
                            connected_turns: list[tuple[int, float]] = []
                            for canon, turns, score in lm_hits:
                                for t in turns:
                                    if t not in retrieved_ids and t < len(engine.memory_store):
                                        connected_turns.append((t, score))
                            # Deduplicate, sort by entity score
                            seen: set[int] = set()
                            unique_connected = []
                            for t, s in sorted(connected_turns, key=lambda x: -x[1]):
                                if t not in seen:
                                    unique_connected.append((t, s))
                                    seen.add(t)
                            # Replace lowest-scored results with connected turns
                            n_expand = min(len(unique_connected), max(1, effective_k // 5))
                            if n_expand > 0 and len(results) >= n_expand:
                                for i in range(n_expand):
                                    t_idx, t_score = unique_connected[i]
                                    entry = engine.memory_store[t_idx]
                                    results[-(i + 1)] = {
                                        "text": entry.text,
                                        "score": float(t_score),
                                        "index": t_idx,
                                    }

                return results

        # BM25 primary: use lexical retrieval as the main channel.
        # Discriminative test (2026-04-02): 58.3% hit@10 vs 36.7% cosine
        # on full LME fixture (1282 turns, 60 questions).
        if self._resolve_bm25_primary:
            bm25_idx = getattr(self, "_bm25_index", None)
            if bm25_idx is not None:
                bm25_hits = bm25_idx.query(query_text or "", top_k=effective_k)
                results = []
                for doc_id, score in bm25_hits:
                    if doc_id < len(engine.memory_store):
                        entry = engine.memory_store[doc_id]
                        results.append({
                            "text": entry.text,
                            "score": float(score),
                            "index": doc_id,
                        })
                return results

        if self._resolve_transfer:
            results = engine.query_transfer(
                embedding=q_emb,
                top_k=effective_k,
                transfer_k=self._resolve_transfer_k,
            )
        elif self._resolve_coretrieval:
            results = engine.query_coretrieval(
                embedding=q_emb,
                top_k=effective_k,
                coretrieval_bonus=self._resolve_coretrieval_bonus,
                min_coretrieval_count=self._resolve_coretrieval_min_count,
            )
        elif self._resolve_cooc_boost and self._resolve_ppr:
            results = engine.query_cooc_ppr(
                embedding=q_emb,
                top_k=effective_k,
                cooc_weight=self._resolve_cooc_weight,
                gate_threshold=self._resolve_cooc_gate_threshold,
                ppr_weight=self._resolve_ppr_weight,
                damping=self._resolve_ppr_damping,
            )
        elif self._resolve_cooc_boost and self._resolve_bm25_weight > 0:
            results = engine.query_cooc_bm25(
                embedding=q_emb,
                query_text=query_text,
                top_k=effective_k,
                cooc_weight=self._resolve_cooc_weight,
                bm25_weight=self._resolve_bm25_weight,
                gate_threshold=self._resolve_cooc_gate_threshold,
            )
        elif self._resolve_cooc_boost:
            results = engine.query_cooc_boost(
                embedding=q_emb,
                top_k=effective_k,
                cooc_weight=self._resolve_cooc_weight,
                dream_weight=self._resolve_dream_weight,
                gate_threshold=self._resolve_cooc_gate_threshold,
                fisher_scoring=self._resolve_fisher_scoring,
                fisher_temperature=self._resolve_fisher_temperature,
            )
        elif self._resolve_ppr:
            results = engine.query_ppr(
                embedding=q_emb,
                top_k=effective_k,
                ppr_weight=self._resolve_ppr_weight,
                damping=self._resolve_ppr_damping,
            )
        elif self._resolve_hybrid:
            results = engine.query_hybrid(
                embedding=q_emb,
                top_k=effective_k,
            )
        elif self._resolve_associative or self._resolve_sparse:
            results = engine.query_associative(
                embedding=q_emb,
                top_k=effective_k,
                sparse=self._resolve_sparse,
            )
        else:
            results = engine.query(
                embedding=q_emb,
                top_k=effective_k,
            )

        return results

    def _apply_post_retrieval(self, results: list[dict]) -> list[dict]:
        """Apply post-retrieval reranking and filtering.

        Currently: belief reranking when active, truncation to retrieve_num.

        Args:
            results: Raw retrieval results.

        Returns:
            Reranked and truncated results.
        """
        if self._resolve_belief_index is not None:
            results = self._apply_belief_rerank(results)[
                : self._resolve_retrieve_num
            ]
        return results

    def _retrieve_v2(self, query_text: str) -> list[dict]:
        """Run V2 embedding retrieval for a single query string.

        Composed from independent stages:
          1. _embed_query      -- vectorize the query text
          2. _retrieve_candidates -- run the backend retrieval switch
          3. _apply_post_retrieval -- belief rerank + truncation
          3b. BM25 rerank (optional) -- blend cosine + BM25 scores
          4. entity reranker (optional) -- via external EntityReranker callback
          5. cross-encoder reranker (optional)
          6. role weighting -- discount assistant/continuation turns
          7. temporal proximity -- boost recency-relevant facts (no-op when no date in query)
        """
        q_emb = self._embed_query(query_text)
        results = self._retrieve_candidates(
            q_emb, self._resolve_retrieve_num, query_text=query_text,
        )
        results = self._apply_post_retrieval(results)
        # Event calendar boost (feature-flagged). Applied after fusion/rerank
        # so it can nudge calendar-relevant candidates above peers without
        # overriding them. Neutral when no calendar or no Current Date.
        if self._resolve_event_calendar:
            calendar = getattr(self, "_event_calendar_lme", None)
            results = _apply_event_calendar_boost(
                results,
                query_text,
                calendar,
                self._resolve_event_calendar_window_days,
                self._resolve_event_calendar_boost,
            )
        results = self._apply_bm25_rerank(results, query_text)
        if self._entity_reranker is not None:
            results = self._entity_reranker.rerank(
                query_text, results, self._resolve_engine,
            )
        ce_reranker = getattr(self, "_cross_encoder_reranker", None)
        if ce_reranker is not None:
            results = ce_reranker.rerank(query_text, results)
        results = self._apply_role_weighting(results)
        results = self._apply_temporal_proximity(results, query_text)
        # Session-aware round-robin merge (feature-flagged). Applied last
        # so it can reshape the final candidate order for multi-session
        # aggregation queries. Neutral when flag is off or classifier
        # returns "none" for this query.
        results = self._apply_lme_multihop(results, query_text)
        return results

    def _apply_bm25_rerank(self, results: list[dict], query_text: str) -> list[dict]:
        """Blend cosine scores with BM25 lexical scores and re-sort.

        score = (1 - w) * cosine + w * bm25_normalized

        Discriminative test (2026-04-02): 100% hit@10, avg rank 1.20.
        Uses BM25 IDF statistics from the full index but TF from each
        candidate's text, so it works on any cosine-retrieved candidate.
        """
        if not self._resolve_bm25_rerank:
            return results
        bm25_idx = getattr(self, "_bm25_index", None)
        if bm25_idx is None:
            return results
        texts = [r["text"] for r in results]
        bm25_scores = bm25_idx.score_texts(query_text, texts)
        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0
        w = self._resolve_bm25_rerank_weight
        for r, bs in zip(results, bm25_scores):
            r["score"] = (1 - w) * r["score"] + w * (bs / max_bm25)
        results.sort(key=lambda r: -r["score"])
        return results

    def _apply_route_role_boost(
        self, results: list[dict], route: str | None,
    ) -> list[dict]:
        """Boost assistant-role entry scores when route is ``assistant_recall``.

        When ``route="assistant_recall"``, entries whose corresponding
        ``MemoryEntry.role`` is ``"assistant"`` have their score multiplied
        by ``config.assistant_recall_boost`` (default 1.5). All other routes
        (including ``None``) pass results through unchanged.

        After boosting, results are re-sorted descending by score.
        """
        if route != "assistant_recall" or not results:
            return results

        boost = self._cfg.assistant_recall_boost if hasattr(self, "_cfg") else 1.5
        engine = self._resolve_engine
        ms = engine.memory_store

        boosted = []
        for r in results:
            adjusted = dict(r)
            idx = r.get("index")
            role = "unknown"
            if idx is not None and idx < len(ms):
                entry = ms[idx]
                role = getattr(entry, "role", "unknown")
            if role == "assistant":
                adjusted["score"] = r.get("score", 0.0) * boost
            boosted.append(adjusted)

        boosted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return boosted

    def _apply_role_weighting(self, results: list[dict]) -> list[dict]:
        """Multiply scores by role-based weights and re-sort.

        Weights are configured in RetrievalConfig (role_weight_user,
        role_weight_assistant, role_weight_continuation). When all
        weights are 1.0, this is a no-op.
        """
        cfg = self._cfg if hasattr(self, "_cfg") else None
        w_user = cfg.role_weight_user if cfg else 1.0
        w_asst = cfg.role_weight_assistant if cfg else 1.0
        w_cont = cfg.role_weight_continuation if cfg else 1.0
        # Fast path: skip if all weights are 1.0
        if w_user == 1.0 and w_asst == 1.0 and w_cont == 1.0:
            return results

        weight_map = {
            "user": w_user,
            "assistant": w_asst,
            "continuation": w_cont,
            "unknown": 1.0,
        }
        engine = self._resolve_engine
        reweighted = []
        for r in results:
            idx = r.get("index")
            role = "unknown"
            if idx is not None and idx < len(engine.memory_store):
                role = getattr(engine.memory_store[idx], "role", "unknown")
            w = weight_map.get(role, 1.0)
            adjusted = dict(r)
            adjusted["score"] = r.get("score", 0.0) * w
            reweighted.append(adjusted)
        reweighted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reweighted

    def _apply_temporal_proximity(
        self, results: list[dict], query_text: str
    ) -> list[dict]:
        """Multiply scores by temporal proximity to the query date and re-sort.

        Parses a date from *query_text* using ``parse_query_date`` from
        ``mabench.temporal``.  When no date is found (non-LME queries) or a
        MemoryEntry has no ``session_time``, ``temporal_proximity`` returns
        1.0 so this method is a no-op — neutral by design (like cooc_boost).

        Args:
            results: Retrieval results with at minimum ``{"index": int, "score": float}``.
            query_text: Raw query string, potentially containing a date cue.

        Returns:
            Results reweighted by temporal proximity, sorted descending by score.
        """
        from mabench.temporal import parse_query_date, temporal_proximity

        query_date = parse_query_date(query_text)
        if query_date is None:
            return results

        alpha = (
            self._cfg.temporal_alpha
            if hasattr(self, "_cfg") and self._cfg is not None
            else 0.01
        )
        engine = self._resolve_engine
        reweighted = []
        for r in results:
            idx = r.get("index")
            session_time = None
            if idx is not None and idx < len(engine.memory_store):
                entry = engine.memory_store[idx]
                session_time = getattr(entry, "session_time", None)
            w = temporal_proximity(session_time, query_date, alpha)
            adjusted = dict(r)
            adjusted["score"] = r.get("score", 0.0) * w
            reweighted.append(adjusted)
        reweighted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reweighted

    def _apply_lme_multihop(
        self, results: list[dict], query_text: str,
    ) -> list[dict]:
        """Session-aware round-robin merge of retrieval candidates.

        When ``--lme_multihop`` is on AND the query classifies as a
        multi-session aggregation ("how many X total", "list all Y"),
        groups candidates by ``session_time.date()`` and interleaves
        them round-robin so the LLM sees evidence from MULTIPLE sessions
        even when one session dominates similarity scores.

        Pass-through (returns *results* unchanged) when any of:
          - ``--lme_multihop`` is off,
          - the query is not a multi-session aggregation (classifier
            returns ``"none"`` for temporal date-diff and point queries),
          - ``results`` is empty.

        Args:
            results: retrieval output with ``{"index", "score", "text"}``.
            query_text: raw query string (may include "Current Date:" prefix).

        Returns:
            Session-diversified list of result dicts, or the original
            list unchanged when neutral conditions hold. Each kept result
            is annotated in-place with ``session_time`` for downstream
            consumers (not stored on the original dicts — a shallow copy
            is made when reshaping fires).
        """
        if not self._resolve_lme_multihop or not results or not query_text:
            return results

        from mabench.lme_multihop import (
            aggregate_hop_results,
            classify_multihop_query,
            decompose_aggregation_query,
        )

        # Strict gate: only fire on genuine multi-session aggregation queries.
        # Temporal date-diff, point lookups, etc. short-circuit to pass-through.
        if classify_multihop_query(query_text) == "none":
            return results

        # Annotate results with session_time from the memory store so the
        # library's session-bucketing logic can group by session.date().
        engine = self._resolve_engine
        mem_store = getattr(engine, "memory_store", None)
        if mem_store is None:
            return results

        annotated: list[dict] = []
        for r in results:
            idx = r.get("index")
            session_time = None
            if idx is not None and 0 <= idx < len(mem_store):
                session_time = getattr(mem_store[idx], "session_time", None)
            entry = dict(r)
            entry["session_time"] = session_time
            annotated.append(entry)

        # Build a plan with the configured per-session / max-session caps.
        plan = decompose_aggregation_query(query_text, None)
        # Override defaults from RetrievalConfig so operators can tune without
        # editing the library.
        plan = type(plan)(
            kind=plan.kind,
            base_query=plan.base_query,
            sub_queries=plan.sub_queries,
            per_session_top_k=self._resolve_lme_multihop_per_session_top_k,
            max_sessions=self._resolve_lme_multihop_max_sessions,
            merge_strategy=plan.merge_strategy,
        )
        reshaped = aggregate_hop_results(annotated, plan)
        # aggregate_hop_results already dedupes + interleaves. Return new list.
        return reshaped

    def _apply_belief_rerank(self, results: list[dict]) -> list[dict]:
        """Rerank retrieval results using Bayesian belief scores.

        1. Exclude facts below hard_floor (P(current) too low).
        2. Multiply each result's score by P(current).
        3. Re-sort by adjusted score descending.
        """
        belief = self._resolve_belief_index
        reranked = []
        for r in results:
            text = r.get("text", "")
            if belief.is_excluded(text):
                continue
            p = belief.score(text)
            adjusted = dict(r)
            adjusted["score"] = r.get("score", 0.0) * p
            reranked.append(adjusted)
        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reranked

    # ------------------------------------------------------------------
    # Public API aliases (for standalone callers who prefer clean names)
    # ------------------------------------------------------------------
    embed_query = _embed_query
    retrieve = _retrieve_v2
    retrieve_candidates = _retrieve_candidates
    compute_effective_k = _compute_effective_k
    apply_post_retrieval = _apply_post_retrieval
    apply_belief_rerank = _apply_belief_rerank


# Backward-compat alias -- hermes_agent.py imports RetrievalMixin
RetrievalMixin = Retriever
