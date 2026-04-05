"""LME-specific multi-hop query decomposer (library only, no LLM calls).

LongMemEval multi-session questions ask the LLM to AGGREGATE evidence
across multiple conversation sessions ("how many fish total", "list all
restaurants", "total cost of …"). Standard top-K retrieval is
session-blind: it picks the K highest similarity scores globally, which
tends to cluster around ONE session and starve evidence from the others.

This module provides three pure functions:

- ``classify_multihop_query(query_text)``: regex classifier that returns
  one of ``{"count", "count_distinct", "total_sum", "list_all", "none"}``.
  Extends the existing ``is_aggregation_query`` logic with a *kind*
  distinction and the same LME-tuned exclusion patterns for temporal
  date-diff questions ("weeks ago", "days passed since").

- ``decompose_aggregation_query(query_text, query_date)``: returns an
  ``AggregationPlan`` that bundles the classification kind, the base
  query, a list of retrieval seed variants (base + subject-stripped),
  and session-diversity parameters (per-session top-k, max sessions).
  **Not a chained ``{answer_N}`` decomposition** — sub-queries here are
  parallel retrieval seeds, not LLM sub-questions.

- ``aggregate_hop_results(results, plan)``: session-aware merge.
  Groups candidates by ``session_time.date()``, keeps top-k per session,
  round-robin interleaves across sessions, dedupes by text prefix.
  Guarantees the LLM sees evidence from MULTIPLE sessions even when one
  session dominates similarity scores.

This is **library-only**. No agent wiring. Feature flag
``--lme_multihop`` will be added in a follow-up PR.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns (LME-tuned; derived from `is_aggregation_query` in retrieval.py)
# ---------------------------------------------------------------------------

# count_distinct: "how many DIFFERENT X", "distinct", "unique". Checked FIRST
# so "how many different" does not fall into plain "count".
_COUNT_DISTINCT_RE = re.compile(
    r"\b(how\s+many\s+(different|distinct|unique))\b"
    r"|\b(different|distinct|unique)\s+(\w+\s+){0,3}(\w+s)\b",
    re.I,
)

# total_sum: totals and sums — "total cost", "in total", "altogether", "sum".
_TOTAL_SUM_RE = re.compile(
    r"\btotal\s+(cost|amount|price|number|sum)\b"
    r"|\bin\s+total\b"
    r"|\baltogether\b"
    r"|\b(total|sum)\s+of\b"
    r"|\bhow\s+much\s+(did|have|do)\s+I\s+spend\b",
    re.I,
)

# count: plain counts — "how many X", "number of", "count", "how often".
_COUNT_RE = re.compile(
    r"\bhow\s+many\b"
    r"|\bnumber\s+of\b"
    r"|\bcount\b"
    r"|\bhow\s+often\b"
    r"|\bfrequently\b",
    re.I,
)

# list_all: enumeration — "list all X", "all the X".
_LIST_ALL_RE = re.compile(
    r"\blist\s+all\b"
    r"|\ball\s+the\s+(\w+s)\b"
    r"|\bwhat\s+are\s+all\s+(the|my)\b",
    re.I,
)

# Exclusions: temporal date-diff, not aggregation.
# Copied from ``_AGG_EXCL_PATTERNS`` in retrieval.py to stay in sync.
_TEMPORAL_EXCL_RE = re.compile(
    r"\b(days?|weeks?|months?|years?|hours?|minutes?)\s+ago\b"
    r"|\bdays?\s+(had\s+)?passed\b"
    r"|\bbetween\s+.{1,60}?\s+and\b"
    r"|\bsince\s+(when|that|the|I|my|we|you)\b"
    r"|\bhow\s+long\s+ago\b"
    r"|\b(have|has|had)\s+passed\s+since\b",
    re.I,
)

# Prefix words stripped when generating a subject-variant sub-query
_STRIP_PREFIX_RE = re.compile(
    r"^\s*"
    r"(how\s+many\s+(different|distinct|unique)\s*"
    r"|how\s+many\s*"
    r"|how\s+often\s*"
    r"|how\s+much\s*"
    r"|list\s+all\s+(the|my)?\s*"
    r"|what\s+is\s+the\s+total\s+(cost|amount|price|number|sum)\s+of\s*"
    r"|what\s+is\s+the\s+total\s*"
    r"|what\s+are\s+all\s+(the|my)?\s*"
    r"|number\s+of\s*"
    r"|count\s+(of|the)?\s*"
    r"|all\s+the\s*"
    r")",
    re.I,
)

# Phrase words scrubbed anywhere in the query for the subject-stripped variant
_SCRUB_RE = re.compile(
    r"\bin\s+total\b"
    r"|\baltogether\b"
    r"|\bdid\s+I\b"
    r"|\bdo\s+I\b"
    r"|\bhave\s+I\b"
    r"|\bare\s+there\b"
    r"|\bare\s+in\b",
    re.I,
)


# ---------------------------------------------------------------------------
# classify_multihop_query
# ---------------------------------------------------------------------------

def classify_multihop_query(query_text: str) -> str:
    """Classify the aggregation style of an LME question.

    Returns one of:
      - ``"count"``           — plain counting ("how many X")
      - ``"count_distinct"``  — distinct counting ("how many different X")
      - ``"total_sum"``       — summing values ("total cost", "in total")
      - ``"list_all"``        — enumeration ("list all X")
      - ``"none"``            — not a multi-session aggregation query

    Temporal date-diff patterns ("weeks ago", "days passed since") are
    explicitly excluded even when they start with "how many" — those are
    date-arithmetic questions, not aggregation.
    """
    if not query_text:
        return "none"

    # Exclude temporal date-diff questions first.
    if _TEMPORAL_EXCL_RE.search(query_text):
        return "none"

    # Order matters: more specific patterns before generic "count".
    if _COUNT_DISTINCT_RE.search(query_text):
        return "count_distinct"
    # A "how many" question is a COUNT even if it uses "in total" as
    # emphasis ("how many fish in total" = 1 integer answer, not a sum).
    # Only route to total_sum when "how many" is NOT present.
    has_how_many = _COUNT_RE.search(query_text) is not None
    if _TOTAL_SUM_RE.search(query_text) and not has_how_many:
        return "total_sum"
    if _LIST_ALL_RE.search(query_text):
        return "list_all"
    if has_how_many:
        return "count"
    return "none"


# ---------------------------------------------------------------------------
# AggregationPlan dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregationPlan:
    """Plan describing how to retrieve + merge evidence for a multi-session
    aggregation question.

    Attributes:
        kind: classify_multihop_query result.
        base_query: cleaned original query string.
        sub_queries: list of retrieval seed variants (base + subject-stripped).
            Used as PARALLEL retrieval seeds — NOT chained like FC's
            ``{answer_N}`` decomposition.
        per_session_top_k: max candidates to keep from each distinct session.
        max_sessions: max distinct sessions to sample from.
        merge_strategy: currently only ``"union_dedup"``.
    """

    kind: str
    base_query: str
    sub_queries: list[str]
    per_session_top_k: int
    max_sessions: int
    merge_strategy: str


# ---------------------------------------------------------------------------
# decompose_aggregation_query
# ---------------------------------------------------------------------------

def _strip_aggregation_prefix(query_text: str) -> str:
    """Return the query with aggregation prefix words removed.

    "How many fish are there in total in both of my aquariums?"
    -> "fish in both of my aquariums?"

    "What is the total cost of the food bowl and collar?"
    -> "the food bowl and collar?"
    """
    stripped = _STRIP_PREFIX_RE.sub("", query_text).strip()
    stripped = _SCRUB_RE.sub(" ", stripped)
    # Collapse multiple spaces
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped


def decompose_aggregation_query(
    query_text: str,
    query_date: datetime | None = None,  # noqa: ARG001 - reserved for future session-window filtering
) -> AggregationPlan:
    """Produce a session-diversity retrieval plan for *query_text*.

    For aggregation queries, generates multiple retrieval seed variants
    that broaden the candidate pool away from the single dominant phrase.
    For non-aggregation queries, returns a ``kind="none"`` plan where
    ``sub_queries == [base_query]`` (no fan-out).

    Args:
        query_text: raw query string (may include LME "Current Date: …" boilerplate).
        query_date: parsed query datetime if available. Reserved for a future
            session-window narrowing step — not used in this version.

    Returns:
        AggregationPlan. ``sub_queries`` always contains the base query
        first, followed by variants when kind != "none".
    """
    base = (query_text or "").strip()
    kind = classify_multihop_query(base)

    if kind == "none":
        return AggregationPlan(
            kind="none",
            base_query=base,
            sub_queries=[base],
            per_session_top_k=1,
            max_sessions=1,
            merge_strategy="union_dedup",
        )

    # Generate variants. Always include the base query first.
    variants: list[str] = [base]
    stripped = _strip_aggregation_prefix(base)
    if stripped and stripped != base and len(stripped) >= 3:
        variants.append(stripped)

    # Defaults chosen to cover typical LME multi-session questions where
    # evidence is distributed across 2-10 distinct session dates.
    return AggregationPlan(
        kind=kind,
        base_query=base,
        sub_queries=variants,
        per_session_top_k=2,
        max_sessions=10,
        merge_strategy="union_dedup",
    )


# ---------------------------------------------------------------------------
# aggregate_hop_results
# ---------------------------------------------------------------------------

def _session_key(result: dict) -> str | None:
    """Extract session-date bucket key from a result dict.

    Returns ``"YYYY-MM-DD"`` string when ``session_time`` is a datetime,
    or ``None`` for results lacking session_time. The ``None`` bucket is
    treated as its own session so those results are not silently dropped.
    """
    st = result.get("session_time")
    if isinstance(st, datetime):
        return st.date().isoformat()
    return None


def aggregate_hop_results(
    results: list[dict],
    plan: AggregationPlan,
) -> list[dict]:
    """Session-aware round-robin merge with dedup.

    Algorithm:
      1. Group ``results`` by ``session_time.date()`` (or ``None`` bucket).
      2. Within each session bucket, sort by ``score`` descending and keep
         top ``plan.per_session_top_k``.
      3. Pick the top ``plan.max_sessions`` buckets by their best score.
      4. Round-robin interleave: session A #1, B #1, C #1, A #2, B #2, …
      5. Dedup by ``text.strip()[:200]`` — first occurrence wins.

    Guarantees evidence from multiple distinct sessions reaches the LLM
    even when one session dominates the similarity ranking.

    Args:
        results: retrieval output with at minimum ``text``, ``score``,
            and (optionally) ``session_time`` keys.
        plan: AggregationPlan controlling per-session / max-session caps.

    Returns:
        New list of result dicts, round-robin interleaved and deduped.
        Order preserves session-diversity in the top slots.
    """
    if not results:
        return []

    # Step 1: group by session key.
    buckets: dict[Any, list[dict]] = {}
    for r in results:
        key = _session_key(r)
        buckets.setdefault(key, []).append(r)

    # Step 2: sort each bucket by score desc; truncate to per_session_top_k.
    trimmed: dict[Any, list[dict]] = {}
    for key, items in buckets.items():
        items_sorted = sorted(
            items, key=lambda r: r.get("score", 0.0), reverse=True,
        )
        trimmed[key] = items_sorted[: plan.per_session_top_k]

    # Step 3: rank session buckets by their best (top) score; keep max_sessions.
    def _bucket_top_score(items: list[dict]) -> float:
        return items[0].get("score", 0.0) if items else 0.0

    ranked_keys = sorted(
        trimmed.keys(),
        key=lambda k: _bucket_top_score(trimmed[k]),
        reverse=True,
    )[: plan.max_sessions]

    # Step 4: round-robin interleave.
    merged: list[dict] = []
    seen: set[str] = set()
    max_depth = plan.per_session_top_k
    for depth in range(max_depth):
        for key in ranked_keys:
            bucket = trimmed.get(key, [])
            if depth >= len(bucket):
                continue
            r = bucket[depth]
            # Step 5: dedup by text prefix.
            text_key = (r.get("text") or "").strip()[:200]
            if not text_key or text_key in seen:
                continue
            seen.add(text_key)
            merged.append(r)
    return merged
