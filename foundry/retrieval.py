"""Retrieval primitives for the Hermes Foundry memory system.

This module implements the Phase 6 retrieval contract from ``foundry/spec.md``.
It provides:

- thin tool wrappers over the storage and query layers
- within-query deduplication of turn observations
- an agentic retriever with always-on tool planning

The implementation stays stdlib-first. When no runtime LLM/tool orchestration
is available, ``AgenticRetriever`` degrades gracefully to deterministic tool
execution and trace capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import ast
import math
import re
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from foundry.ontology import GroundingLayer
from foundry.query import QueryPlan, decompose_query
from foundry.storage import (
    BM25Index,
    EntityNode,
    Fact,
    FoundryStore,
    RawChunk,
    TemporalEvent,
    TriadicMemory,
    VectorIndex,
    _normalize_name,
    _tokenize,
)
from foundry.writeback import ConflictDetector


_RRF_K = 60.0
_DEFAULT_TOP_K = 10
_OBSERVATION_SEPARATOR = "\n\n"
_VALID_LAYERS = {layer.value for layer in GroundingLayer}
RerankerFn = Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
TOOLS: dict[str, dict[str, str]] = {
    "search_memory": {
        "signature": "(query: str) -> list[{turn_id, text, score}]",
        "description": (
            "BM25+cosine RRF over raw chunks with optional rerank. Best for names, "
            "phrases, and lexical evidence. Returns top-ranked turn snippets."
        ),
    },
    "search_entities": {
        "signature": "(query: str, layer: Optional[str] = None) -> list[{entity, facts, turns, score}]",
        "description": (
            "BM25 over entity documents (name + facets). Best for WHO/WHAT/WHERE "
            "lookups. Optional layer filter: stated, inferred, derived."
        ),
    },
    "get_related": {
        "signature": "(entity: str, layer: Optional[str] = None) -> {properties, co_mentions, neighbors}",
        "description": (
            "Retrieve properties, co-mentions, and graph neighbors for one entity "
            "across available grounding layers."
        ),
    },
    "search_temporal": {
        "signature": "(query: str) -> list[{turn_id, text, date, score}]",
        "description": (
            "Temporal/date-based retrieval using the temporal index. Use for WHEN, "
            "ordering, duration, and date-range questions."
        ),
    },
    "read_turn": {
        "signature": "(turn_id: int) -> {text, role, turn_id}",
        "description": "Read the full text of a specific conversation turn.",
    },
    "aggregate": {
        "signature": "(entity_type: str, op: str) -> {result, entities}",
        "description": (
            "Count or sum entities of a given type. entity_type matches the 'what' "
            "facet (e.g. 'bike', 'pet_fish', 'doctor', 'luxury_item'). "
            "op is one of: count, sum, list. "
            "count: sums quantity values across entities of that type (not just a count of names). "
            "sum: total of their numeric values (prices, amounts). "
            "list: returns names and quantities of matching entities. "
            "Use for 'how many X' or 'total cost of X' questions. "
            "Sub-types are resolved via hierarchy (e.g. 'pet_fish' excludes caught fish)."
        ),
    },
    "execute_python": {
        "signature": "(code: str) -> {result: str}",
        "description": (
            "Sandboxed Python for deterministic arithmetic. Only datetime and math "
            "are allowed imports."
        ),
    },
    "answer": {
        "signature": "(evidence: str) -> TERMINAL",
        "description": "Return gathered evidence and terminate the retrieval loop.",
    },
}


# ---------------------------------------------------------------------------
# ReAct loop infrastructure
# ---------------------------------------------------------------------------

_REACT_SYSTEM_PROMPT = """\
You are a memory search agent. Your job is to find information in a user's
conversation history to answer their question.

You have these tools:

{tool_descriptions}

RULES:
- Think step-by-step about what information you need.
- Start with the most specific search (entities or keywords from the question).
- If initial search doesn't find enough, try DIFFERENT search terms or tools.
- Use search_entities for WHO/WHAT/WHERE, then get_related to explore connections.
- For "how many X" questions, try aggregate(type, "count") or aggregate(type, "list") FIRST.
  Use a short noun as type (e.g. "fish", "doctor", "bike").
- Use read_turn to read full text when a snippet looks promising.
- Call answer() once you have sufficient evidence. Don't over-search.
- Maximum {max_steps} tool calls allowed. Budget wisely.
- IMPORTANT: Your answer() evidence should contain the specific facts/numbers
  from the conversation, not guesses. Only state what the data shows.
- Each Thought/Action must be on its own line.

FORMAT (strict):
Thought: <your reasoning about what to search for next>
Action: <tool_name>(<arguments>)

After each Action, you'll see an Observation with the results.
When ready, use: Action: answer(<your collected evidence>)"""


_ACTION_RE = re.compile(
    r"Action:\s*(\w+)\((.+?)\)\s*$",
    re.MULTILINE | re.DOTALL,
)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)

# Maps tool name → ordered parameter names for positional arg parsing.
_TOOL_PARAMS: dict[str, list[str]] = {
    "search_memory": ["query"],
    "search_entities": ["query"],
    "get_related": ["entity"],
    "search_temporal": ["query"],
    "read_turn": ["turn_id"],
    "aggregate": ["entity_type", "op", "property_predicate"],
    "execute_python": ["code"],
    "answer": ["evidence"],
}


def _build_tool_descriptions() -> str:
    """Build tool description text from the TOOLS registry."""
    lines: list[str] = []
    for name, info in TOOLS.items():
        lines.append(f"{name}{info['signature']}")
        lines.append(f"  {info['description']}")
        lines.append("")
    return "\n".join(lines)


def _split_react_args(s: str) -> list[str]:
    """Split comma-separated arguments respecting quoted strings."""
    parts: list[str] = []
    current: list[str] = []
    in_quote: str | None = None
    for ch in s:
        if ch in ('"', "'") and in_quote is None:
            in_quote = ch
        elif ch == in_quote:
            in_quote = None
        elif ch == "," and in_quote is None:
            parts.append("".join(current).strip().strip('"').strip("'"))
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip().strip('"').strip("'"))
    return parts


def parse_react_output(text: str) -> tuple[str, str, str]:
    """Parse LLM output into (thought, tool_name, raw_args).

    Returns ("", "", "") if parsing fails.
    """
    thought = ""
    tm = _THOUGHT_RE.search(text)
    if tm:
        thought = tm.group(1).strip()

    am = _ACTION_RE.search(text)
    if not am:
        return thought, "", ""

    tool_name = am.group(1).strip()
    raw_args = am.group(2).strip()
    if (raw_args.startswith('"') and raw_args.endswith('"')) or \
       (raw_args.startswith("'") and raw_args.endswith("'")):
        raw_args = raw_args[1:-1]
    return thought, tool_name, raw_args


def _parse_react_args(tool_name: str, raw_args: str) -> dict[str, Any]:
    """Convert positional ReAct arguments into keyword arguments."""
    params = _TOOL_PARAMS.get(tool_name, [])
    if not params:
        return {}
    if len(params) == 1:
        return {params[0]: raw_args}
    parts = _split_react_args(raw_args)
    return {param: parts[i] for i, param in enumerate(params) if i < len(parts)}


_OBS_BUDGET = 2000


def _format_observation(result: Any) -> str:
    """Format tool result to fit within budget without losing key information."""
    import json as _json

    if isinstance(result, list):
        if not result:
            return "No results found."
        for n in range(len(result), 0, -1):
            out = _json.dumps(result[:n], indent=None, default=str)
            if len(out) <= _OBS_BUDGET:
                return out
        item = dict(result[0]) if isinstance(result[0], dict) else result[0]
        if isinstance(item, dict) and "text" in item:
            item["text"] = item["text"][:300] + "..."
        return _json.dumps([item], indent=None, default=str)[:_OBS_BUDGET]

    if isinstance(result, dict):
        if "error" in result:
            return f"Error: {result['error']}"
        if "co_mentions" in result:
            out = dict(result)
            for n in range(len(out["co_mentions"]), -1, -1):
                out["co_mentions"] = result["co_mentions"][:n]
                s = _json.dumps(out, indent=None, default=str)
                if len(s) <= _OBS_BUDGET:
                    return s
        return _json.dumps(result, indent=None, default=str)[:_OBS_BUDGET]
    return str(result)[:_OBS_BUDGET]


@dataclass(frozen=True)
class RetrievalBudget:
    """Resource limits for a single retrieval operation."""

    max_steps: int = 8
    max_observation_chars: int = 2000
    max_total_ms: int = 30000
    aggregation_overfetch: int = 25
    max_context_tokens: int = 8192
    sliding_window_after: int = 4


@dataclass
class RetrievalTrace:
    """Full trace of one retrieval session."""

    question: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    evidence: str = ""
    total_ms: float = 0.0
    terminated_by: str = ""


class DeduplicationTracker:
    """Tracks seen turn IDs across tool calls within one question."""

    def __init__(self) -> None:
        self._seen: set[int] = set()

    @property
    def seen_turns(self) -> set[int]:
        return set(self._seen)

    def filter_new(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fresh: list[dict[str, Any]] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            turn_id = result.get("turn_id")
            if turn_id is None:
                fresh.append(result)
                continue
            try:
                normalized_turn = int(turn_id)
            except (TypeError, ValueError):
                fresh.append(result)
                continue
            if normalized_turn in self._seen:
                continue
            self._seen.add(normalized_turn)
            fresh.append(result)
        return fresh

    def reset(self) -> None:
        self._seen.clear()


def _chunk_text_for_preview(chunk: Optional[RawChunk], fallback_text: str = "") -> str:
    text = chunk.text if chunk is not None else fallback_text
    compact = " ".join(str(text).split())
    return compact[:300]


def _rrf_merge(
    rankings: Sequence[list[tuple[int, float]]],
    *,
    k: float = _RRF_K,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    merged = list(scores.items())
    merged.sort(key=lambda item: (-item[1], item[0]))
    return merged


def _result_turn_id(chunk_store: Optional[Any], doc_id: int) -> int:
    if chunk_store is None:
        return int(doc_id)
    try:
        chunk = chunk_store.get(doc_id)
    except Exception:
        return int(doc_id)
    return int(getattr(chunk, "turn_idx", doc_id))


def _get_chunk(chunk_store: Optional[Any], doc_id: int) -> Optional[RawChunk]:
    if chunk_store is None:
        return None
    try:
        return chunk_store.get(doc_id)
    except Exception:
        return None


def _full_chunk_text(chunk_store: Optional[Any], doc_id: int, fallback_text: str = "") -> str:
    chunk = _get_chunk(chunk_store, doc_id)
    text = getattr(chunk, "text", None) if chunk is not None else None
    return str(text or fallback_text or "")


def _chunk_lookup_by_text(chunk_store: Optional[Any]) -> dict[str, list[RawChunk]]:
    chunks = getattr(chunk_store, "_chunks", None)
    if not isinstance(chunks, Mapping):
        return {}
    lookup: dict[str, list[RawChunk]] = {}
    for chunk_id in sorted(chunks):
        chunk = chunks[chunk_id]
        text = str(getattr(chunk, "text", "") or "")
        if text:
            lookup.setdefault(text, []).append(chunk)
    return lookup


def _safe_layer(layer: Optional[str]) -> Optional[str]:
    if layer is None:
        return None
    normalized = str(layer).strip().lower()
    if not normalized:
        return None
    if normalized not in _VALID_LAYERS:
        raise ValueError(f"Unsupported layer: {layer}")
    return normalized


def _iter_graphs(store: FoundryStore, layer: Optional[str]) -> list[tuple[str, Any]]:
    requested = _safe_layer(layer)
    if requested == GroundingLayer.STATED.value:
        return [(GroundingLayer.STATED.value, store.entity_graph)]
    if requested == GroundingLayer.INFERRED.value:
        return [(GroundingLayer.INFERRED.value, store.inferred_graph)]
    if requested == GroundingLayer.DERIVED.value:
        return [(GroundingLayer.DERIVED.value, store.derived)]
    return [
        (GroundingLayer.STATED.value, store.entity_graph),
        (GroundingLayer.INFERRED.value, store.inferred_graph),
        (GroundingLayer.DERIVED.value, store.derived),
    ]


def _candidate_entity_for_doc(store: FoundryStore, doc_id: int, text: str = "") -> Optional[EntityNode]:
    entity = store.entity_graph.get_entity(str(doc_id))
    if entity is not None:
        return entity

    entities = store.entity_graph.all_entities()
    if 0 <= int(doc_id) < len(entities):
        return entities[int(doc_id)]

    text_tokens = set(_tokenize(text))
    best: tuple[float, Optional[EntityNode]] = (0.0, None)
    for candidate in entities:
        alias_tokens = set(_tokenize(candidate.canonical.replace("_", " ")))
        for alias in candidate.aliases:
            alias_tokens.update(_tokenize(alias))
        if not alias_tokens:
            continue
        overlap = len(alias_tokens & text_tokens) / max(len(alias_tokens), 1)
        if overlap > best[0]:
            best = (overlap, candidate)
    return best[1]


def _annotated_fact_payload(fact: Fact, source_layer: str) -> dict[str, Any]:
    return {
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "turn_id": fact.turn_idx,
        "chunk_id": fact.chunk_idx,
        "source_layer": source_layer,
        "superseded": fact.superseded,
    }


def _facts_for_entity(entity: EntityNode, source_layer: str, *, only_current: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for facts in entity.facts.values():
        for fact in facts:
            if only_current and fact.superseded:
                continue
            rows.append(_annotated_fact_payload(fact, source_layer))
    rows.sort(key=lambda item: (-item["turn_id"], item["predicate"], item["object"]))
    return rows


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder for reranking retrieval candidates."""

    def __init__(self, model_name: str = "mixedbread-ai/mxbai-rerank-large-v1", top_k: int = 10) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, results: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        """Rerank results by cross-encoder relevance score."""
        if not results:
            return results

        k = self._top_k if top_k is None else top_k
        if k <= 0:
            return []

        self._ensure_model()
        pairs = [[query, row["text"]] for row in results]
        raw_scores = self._model.predict(pairs)
        score_array = np.asarray(raw_scores)
        if score_array.ndim == 0:
            scores = [float(score_array.item())]
        else:
            scores = [float(score) for score in score_array]
        if len(scores) != len(results):
            raise ValueError("Cross-encoder returned a score count that does not match the candidate count")
        for row, score in zip(results, scores):
            row["cross_encoder_score"] = score
        results.sort(key=lambda row: row["cross_encoder_score"], reverse=True)
        return results[:k]

    def __call__(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.rerank(query, results)


def search_memory_hybrid(
    query: str,
    bm25: BM25Index,
    vectors: Optional[VectorIndex] = None,
    *,
    reranker: Optional[RerankerFn] = None,
    seen_turns: Optional[set[int]] = None,
    top_k: int = _DEFAULT_TOP_K,
    chunk_store: Optional[Any] = None,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    triadic: Optional[TriadicMemory] = None,
) -> list[dict[str, Any]]:
    """Hybrid BM25 + cosine retrieval with reciprocal-rank fusion."""

    if not query or top_k <= 0:
        return []

    candidate_limit = max(20, top_k * 2) if reranker is not None else top_k
    bm25_hits = bm25.query(query, top_k=max(20, top_k * 2))
    vector_hits: list[tuple[int, float]] = []
    if vectors is not None and embed_fn is not None:
        try:
            vector_hits = vectors.query(embed_fn(query), top_k=max(20, top_k * 2))
        except Exception:
            vector_hits = []

    merged = _rrf_merge([bm25_hits, vector_hits] if vector_hits else [bm25_hits])
    filtered: list[dict[str, Any]] = []
    seen = set(seen_turns or set())
    retrieved_texts: list[str] = []
    seen_texts: set[str] = set()

    for doc_id, score in merged:
        chunk = _get_chunk(chunk_store, doc_id)
        turn_id = _result_turn_id(chunk_store, doc_id)
        if turn_id in seen:
            continue
        full_text = _full_chunk_text(chunk_store, doc_id, getattr(bm25, "_texts", {}).get(doc_id, ""))
        text = _chunk_text_for_preview(chunk, full_text)
        filtered.append(
            {
                "turn_id": turn_id,
                "chunk_id": int(doc_id),
                "text": text,
                "score": float(score),
            }
        )
        if full_text:
            retrieved_texts.append(full_text)
            seen_texts.add(full_text)
        seen.add(turn_id)
        if len(filtered) >= candidate_limit:
            break

    if triadic is not None and triadic.tensor_entries > 0 and retrieved_texts:
        base_score = min((item["score"] for item in filtered), default=0.0) - 1e-6
        chunk_lookup = _chunk_lookup_by_text(chunk_store)
        expanded = triadic.expand(retrieved_texts, top_k=top_k)
        for rank, expanded_text in enumerate(expanded, start=1):
            if not expanded_text or expanded_text in seen_texts:
                continue
            chunk = None
            for candidate in chunk_lookup.get(expanded_text, []):
                candidate_turn_id = int(getattr(candidate, "turn_idx", getattr(candidate, "chunk_idx", -1)))
                if candidate_turn_id in seen:
                    continue
                chunk = candidate
                break
            if chunk is None:
                continue
            turn_id = int(getattr(chunk, "turn_idx", getattr(chunk, "chunk_idx", -1)))
            chunk_id = int(getattr(chunk, "chunk_idx", -1))
            filtered.append(
                {
                    "turn_id": turn_id,
                    "chunk_id": chunk_id,
                    "text": _chunk_text_for_preview(chunk, expanded_text),
                    "score": base_score - (rank * 1e-6),
                }
            )
            seen_texts.add(expanded_text)
            if turn_id >= 0:
                seen.add(turn_id)

    if reranker is not None and filtered:
        try:
            reranked = reranker(query, list(filtered))
            if isinstance(reranked, list) and reranked:
                filtered = reranked[:top_k]
        except Exception:
            pass

    return filtered[:top_k]


def search_memory(
    store: FoundryStore,
    query: str,
    *,
    dedup: Optional[DeduplicationTracker] = None,
    top_k: int = _DEFAULT_TOP_K,
    reranker: Optional[RerankerFn] = None,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    conflict_resolution: bool = False,
) -> list[dict[str, Any]]:
    expanded_top_k = max(top_k, top_k * 2)
    results = search_memory_hybrid(
        query,
        store.bm25_chunks,
        store.vectors,
        reranker=reranker,
        seen_turns=dedup.seen_turns if dedup is not None else None,
        top_k=expanded_top_k,
        chunk_store=store.chunks,
        embed_fn=embed_fn,
        triadic=store.triadic,
    )
    if conflict_resolution and results and len(store.entity_graph) > 0:
        results = ConflictDetector(store).filter_conflicts(results)
    if dedup is not None:
        results = dedup.filter_new(results)
    return results[:top_k]


def search_entities(
    store: FoundryStore,
    query: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    layer: Optional[str] = None,
) -> list[dict[str, Any]]:
    if not query or top_k <= 0:
        return []

    requested_layer = _safe_layer(layer)
    hits = store.bm25_entities.query(query, top_k=max(top_k * 2, top_k))
    results: list[dict[str, Any]] = []

    for doc_id, score in hits:
        text = getattr(store.bm25_entities, "_texts", {}).get(doc_id, "")
        entity = _candidate_entity_for_doc(store, doc_id, text)
        if entity is None:
            continue

        facts: list[dict[str, Any]] = []
        if requested_layer in (None, GroundingLayer.STATED.value):
            facts.extend(_facts_for_entity(entity, GroundingLayer.STATED.value))

        if requested_layer in (None, GroundingLayer.INFERRED.value):
            inferred = store.inferred_graph.get_entity(entity.canonical)
            if inferred is not None:
                facts.extend(_facts_for_entity(inferred, GroundingLayer.INFERRED.value))

        turns = sorted(
            {
                int(turn_idx)
                for turn_idx in entity.mention_turns
            }
            | {
                int(fact["turn_id"])
                for fact in facts
            }
        )
        results.append(
            {
                "entity": entity.canonical,
                "facts": facts,
                "turns": turns,
                "score": float(score),
            }
        )
        if len(results) >= top_k:
            break

    return results


def get_related(
    store: FoundryStore,
    entity: str,
    *,
    layer: Optional[str] = None,
    max_hops: int = 1,
) -> dict[str, Any]:
    requested_layer = _safe_layer(layer)
    canonical = _normalize_name(entity)
    stated = store.entity_graph.get_entity(canonical)
    inferred = store.inferred_graph.get_entity(canonical)
    node = stated or inferred

    if node is None:
        return {"entity": canonical, "properties": [], "co_mentions": {}, "neighbors": []}

    properties: list[dict[str, Any]] = []
    neighbors: set[str] = set()

    if requested_layer in (None, GroundingLayer.STATED.value) and stated is not None:
        properties.extend(_facts_for_entity(stated, GroundingLayer.STATED.value))
        neighbors.update(stated.related)
        for targets in stated.links.values():
            neighbors.update(targets)

    if requested_layer in (None, GroundingLayer.INFERRED.value) and inferred is not None:
        properties.extend(_facts_for_entity(inferred, GroundingLayer.INFERRED.value))
        neighbors.update(inferred.related)
        for targets in inferred.links.values():
            neighbors.update(targets)

    if requested_layer in (None, GroundingLayer.DERIVED.value):
        derived_neighbors = set(store.derived.interface_memberships.get(canonical, []))
        neighbors.update(derived_neighbors)

    if max_hops > 1 and stated is not None:
        traversal = QueryPlan(store, seed={stated.canonical}).related(max_hops=max_hops).execute()
        neighbors.update(entity_node.canonical for entity_node in traversal if entity_node is not None)

    co_mentions = store.cooccurrence.get_cooccurrences(node.canonical)
    return {
        "entity": node.canonical,
        "properties": properties,
        "co_mentions": co_mentions,
        "neighbors": sorted(neighbors - {node.canonical}),
    }


def search_temporal(
    store: FoundryStore,
    query: str,
    *,
    top_k: int = 5,
    entities: Optional[list[str]] = None,
) -> dict[str, Any]:
    if not query:
        return {"matched_events": [], "bucket": "unknown", "confidence": 0.0}

    matched = store.temporal.match_events(query, top_k=top_k, entities=entities)
    raw_answer = store.temporal.answer_temporal_query(query)
    payloads: list[dict[str, Any]] = []

    for event in matched:
        chunk = None
        if event.turn_id is not None:
            turn_chunks = store.chunks.get_by_turn(int(event.turn_id))
            chunk = turn_chunks[0] if turn_chunks else None
        payloads.append(
            {
                "turn_id": event.turn_id,
                "text": _chunk_text_for_preview(chunk, event.source),
                "date": event.start_dt.isoformat(),
                "score": float(event.confidence),
                "event": event.event,
                "subject": event.subject,
                "object": event.object,
            }
        )

    confidence = max((row["score"] for row in payloads), default=0.0)
    return {
        "matched_events": payloads,
        "bucket": raw_answer.get("bucket", "unknown"),
        "confidence": float(confidence),
    }


def read_turn(store: FoundryStore, turn_id: int) -> dict[str, Any]:
    chunks = store.chunks.get_by_turn(int(turn_id))
    if not chunks:
        return {"text": "", "role": "", "turn_id": int(turn_id)}
    role = chunks[0].role if chunks else ""
    text = "\n".join(chunk.text for chunk in chunks)
    return {"text": text, "role": role, "turn_id": int(turn_id)}


def _latest_numeric_fact(entity: EntityNode, predicate: str) -> Optional[float]:
    """Get the numeric value from the most recent (highest chunk_idx) fact.

    Handles temporal supersession: when the same entity has quantity facts
    across multiple chunks (e.g. bikes: qty=3 at chunk 61, qty=four at chunk 661),
    the latest chunk wins.
    """
    from foundry.storage import _parse_numeric

    facts = entity.facts.get(predicate, [])
    if not facts:
        return None
    # Sort by chunk_idx descending, take the first parseable numeric
    for fact in sorted(facts, key=lambda f: -f.chunk_idx):
        if fact.superseded:
            continue
        n = _parse_numeric(fact.object)
        if n is not None:
            return n
    return None


_GARBAGE_NAMES = {
    "i", "me", "my", "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "do", "does", "did", "have", "has", "had", "will",
    "would", "could", "should", "can", "may", "might", "shall",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "no", "so", "if", "then", "than",
    "that", "this", "these", "those", "it", "its", "there", "here",
    "how", "what", "when", "where", "which", "who", "whom", "why",
    "they", "them", "he", "she", "we", "you", "your", "his", "her",
    "ask", "arrange", "check", "consider", "create", "design",
    "don", "experiment", "follow", "however", "incorporate", "leave",
    "look", "mimic", "nature", "provide", "sat", "sleek", "also",
    "about", "answer", "question", "many", "much", "some", "any",
    "all", "both", "each", "every", "total", "category",
}


def _is_garbage_entity(entity: EntityNode) -> bool:
    """Filter extraction artifacts: stopwords, verbs, chunk-0-only mentions."""
    name = entity.canonical
    if len(name) <= 2:
        return True
    if name.lower() in _GARBAGE_NAMES:
        return True
    mt = getattr(entity, "mention_turns", set())
    if mt and mt == {0}:
        return True
    return False


def _dedup_quantity_entities(
    entity_qty_pairs: list[tuple["EntityNode", float]],
) -> list[tuple["EntityNode", float]]:
    """Deduplicate entities with the same quantity and overlapping mention chunks.

    When the extractor creates two entities for the same real-world group
    (e.g. 'schooling_fish' qty=10 and 'neon_tetras' qty=10 in the same chunk),
    only count the quantity once to avoid double-counting.
    """
    if len(entity_qty_pairs) <= 1:
        return entity_qty_pairs

    used = [False] * len(entity_qty_pairs)
    result: list[tuple["EntityNode", float]] = []

    for i, (e_i, q_i) in enumerate(entity_qty_pairs):
        if used[i]:
            continue
        result.append((e_i, q_i))
        used[i] = True
        for j in range(i + 1, len(entity_qty_pairs)):
            if used[j]:
                continue
            e_j, q_j = entity_qty_pairs[j]
            if q_i != q_j:
                continue
            chunks_i = e_i.mention_chunks
            chunks_j = e_j.mention_chunks
            if not chunks_i or not chunks_j:
                continue
            overlap = len(chunks_i & chunks_j)
            smaller = min(len(chunks_i), len(chunks_j))
            if smaller > 0 and overlap / smaller >= 0.5:
                used[j] = True

    return result


def _type_confidence(
    entity_canonical: str,
    queried_type: str,
    type_index: Any,
) -> float:
    """Fraction of entity's type assignments belonging to the queried type family.

    Uses the TypeSubsumptionIndex's internal assignment counts.  An entity
    typed as 'fish' in 1 out of 5 chunks (with 'website' in the other 4)
    gets 20% confidence for 'fish' — likely a false assignment.
    """
    assignments = type_index._type_assignments.get(entity_canonical)
    if not assignments:
        return 1.0
    total = sum(assignments.values())
    if total == 0:
        return 1.0
    relevant = type_index._descendants_with_depth(queried_type)
    relevant.add(_normalize_name(queried_type))
    matching = sum(assignments.get(t, 0) for t in relevant)
    return matching / total


def aggregate(
    store: FoundryStore,
    entity_type: str,
    op: str,
    *,
    property_predicate: str = "",
    limit: Optional[int] = None,
) -> dict[str, Any]:
    normalized_op = str(op).strip().lower()
    overfetch = limit
    if overfetch is None:
        overfetch = None
    depth = 2 if store.types._subsumptions else 0
    plan = QueryPlan(store).type(entity_type, depth=depth)
    entities = [e for e in plan.execute() if not _is_garbage_entity(e)]

    # Type confidence filter: remove entities whose type assignment is
    # low-confidence (e.g. 'facebook' indexed as 'fish' in 1 out of 5 chunks).
    if len(entities) > 2:
        confident = [
            e for e in entities
            if _type_confidence(e.canonical, entity_type, store.types) >= 0.5
        ]
        if confident:
            entities = confident

    if normalized_op == "count":
        # L1/L2: if entities have quantity facts, sum their latest values
        # instead of just counting entity objects. This handles:
        #   - "how many fish?" → sum qty facets (10+5+1+1=17)
        #   - "how many bikes?" → take latest qty per entity (3→4)
        entity_qty_pairs: list[tuple[EntityNode, float]] = []
        for entity in entities:
            n = _latest_numeric_fact(entity, "quantity")
            if n is not None:
                entity_qty_pairs.append((entity, n))
        # Deduplicate same-quantity entities with overlapping chunks
        entity_qty_pairs = _dedup_quantity_entities(entity_qty_pairs)
        if entity_qty_pairs:
            qty_total = sum(q for _, q in entity_qty_pairs)
            result: Any = int(qty_total) if qty_total == int(qty_total) else qty_total
        else:
            # No quantity facets — fall back to entity count
            result = len(entities)
    elif normalized_op == "list":
        result = [entity.canonical for entity in entities]
    elif normalized_op == "distinct":
        result = plan.aggregate(property_predicate, "distinct")
    elif normalized_op in {"sum", "avg", "min", "max"}:
        result = plan.aggregate(property_predicate, normalized_op)
    else:
        raise ValueError(f"Unsupported aggregate op: {op}")

    entity_names = [entity.canonical for entity in entities]
    if overfetch is not None:
        entity_names = entity_names[: int(overfetch)]
    return {"result": result, "entities": entity_names}


def execute_python(code: str) -> dict[str, str]:
    """Execute a tiny, deterministic Python expression/function body."""

    source = str(code or "").strip()
    if not source:
        return {"result": ""}

    tree = ast.parse(source, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {alias.name.split(".")[0] for alias in getattr(node, "names", [])}
            if names - {"datetime", "math"}:
                raise ValueError("Only datetime and math imports are allowed")
        if isinstance(node, (ast.With, ast.AsyncWith, ast.ClassDef, ast.Lambda, ast.Try, ast.Raise)):
            raise ValueError("Unsupported construct in execute_python")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "open", "__import__"}:
            raise ValueError("Unsafe call in execute_python")

    namespace = {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "sorted": sorted,
        },
        "datetime": datetime,
        "math": math,
        "result": "",
    }
    local_ns: dict[str, Any] = {}
    exec(compile(tree, "<execute_python>", "exec"), namespace, local_ns)
    result = local_ns.get("result", namespace.get("result", ""))
    return {"result": str(result)}


def answer(evidence: str) -> str:
    return str(evidence or "")


# ---------------------------------------------------------------------------
# recall() — single-function retrieval combining all indices
# ---------------------------------------------------------------------------

def recall(
    store: "FoundryStore",
    query: str,
    *,
    entity: str = "",
    predicate: str = "",
    time: str = "",
    top_k: int = 20,
    char_budget: int = 12000,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    reranker: Optional["CrossEncoderReranker"] = None,
) -> str:
    """Single retrieval function — the LLM fills entity/predicate/time,
    the system auto-configures which engines to use.

    The caller (an LLM) decomposes the user's question into structured
    parameters.  ``recall()`` dispatches to the optimal retrieval path
    based on what's populated:

      predicate=count + entity → type hierarchy count (aggregate)
      predicate=sum   + entity → type hierarchy sum of quantity facets
      predicate=list  + entity → type hierarchy entity names
      predicate=when  or time  → Allen temporal engine + BM25 evidence
      default (factual)        → BM25 + cosine RRF + cross-encoder rerank

    All available infrastructure is used automatically — no manual depth
    knob.  If ``embed_fn`` is provided, cosine search is active.  If
    ``reranker`` is provided, cross-encoder reranking is active.

    Args:
        store: The FoundryStore with BM25, entity graph, vectors, etc.
        query: Natural language question (always passed for BM25 fallback).
        entity: Entity type for hierarchy lookup (e.g. "instrument", "fish").
        predicate: What to compute: count, sum, list, when, or "" (factual).
        time: Temporal constraint (e.g. "last month", "2023/05").
        top_k: Max results per search pass.
        char_budget: Max total characters in returned evidence.
        embed_fn: Embedding function for cosine search.
        reranker: CrossEncoderReranker for cross-encoder reranking.

    Returns:
        Evidence string (concatenated turn texts, separated by newlines).
    """
    norm_predicate = predicate.strip().lower()
    norm_entity = entity.strip().lower().replace(" ", "_")

    # ── Route: count / sum / list → skip aggregate, use BM25-only ──
    # Aggregate entity counts are often wrong (extraction quality).
    # BM25 evidence gives the LLM raw conversation text to reason from.
    if norm_predicate in {"count", "sum", "list"} and norm_entity:
        return _recall_retrieve(
            store, query, top_k=top_k, char_budget=char_budget,
            embed_fn=embed_fn, reranker=reranker,
        )

    # ── Route: when / temporal → Allen engine + BM25 evidence ─────
    if norm_predicate == "when" or time:
        parts: list[str] = []
        remaining = char_budget
        # Allen temporal search
        if hasattr(store, "temporal") and store.temporal is not None:
            temporal_q = f"{query} {time}".strip() if time else query
            temporal_entities = [norm_entity] if norm_entity else None
            try:
                temporal_result = search_temporal(
                    store, temporal_q, top_k=top_k,
                    entities=temporal_entities,
                )
                for event in temporal_result.get("matched_events", []):
                    text = event.get("text", "")
                    date = event.get("date", "")
                    if date and text:
                        line = f"[{date}] {text}"
                    else:
                        line = text
                    if line.strip() and len(line) >= 20:
                        parts.append(line)
                        remaining -= len(line)
                        if remaining <= 0:
                            break
            except Exception:
                pass
        # Supplement with BM25+cosine+rerank evidence
        if remaining > 200:
            evidence = _recall_retrieve(
                store, query, top_k=top_k, char_budget=remaining,
                embed_fn=embed_fn, reranker=reranker,
            )
            if evidence:
                parts.append(evidence)
        return "\n".join(parts)

    # ── Default route: factual → BM25 + cosine RRF + rerank ──────
    return _recall_retrieve(
        store, query, top_k=top_k, char_budget=char_budget,
        embed_fn=embed_fn, reranker=reranker,
    )


_BM25_NOISE = {
    "how", "many", "much", "long", "what", "which", "where", "when", "who",
    "whom", "why", "did", "do", "does", "is", "are", "was", "were", "have",
    "has", "had", "can", "could", "would", "should", "will", "shall", "may",
    "might", "the", "a", "an", "i", "me", "my", "we", "our", "you", "your",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "and", "or",
    "but", "not", "that", "this", "it", "its", "be", "been", "being",
    "there", "here", "if", "so", "than", "then", "about", "between",
    "typically", "currently", "usually", "total", "different", "any",
    "answer", "question", "based", "history", "chats", "concisely",
}


def _clean_bm25_query(query: str) -> str:
    """Strip question boilerplate so BM25 matches on content words only."""
    tokens = re.findall(r"[a-zA-Z0-9']+", query.lower())
    content = [t for t in tokens if t not in _BM25_NOISE and len(t) > 1]
    return " ".join(content) if content else query


def _recall_retrieve(
    store: "FoundryStore",
    query: str,
    *,
    top_k: int = 20,
    char_budget: int = 12000,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    reranker: Optional["CrossEncoderReranker"] = None,
) -> str:
    """Core retrieval: BM25 + cosine RRF + cross-encoder rerank.

    Uses all available infrastructure automatically — no depth knob.
    """
    evidence_parts: list[str] = []
    seen_ids: set[int] = set()
    total_chars = 0

    def _add(text: str, doc_id: int = -1) -> bool:
        nonlocal total_chars
        if doc_id in seen_ids and doc_id >= 0:
            return True
        if not text.strip() or len(text) < 20:
            return True
        if total_chars + len(text) > char_budget:
            return False
        evidence_parts.append(text)
        total_chars += len(text)
        if doc_id >= 0:
            seen_ids.add(doc_id)
        return True

    bm25_texts = getattr(store.bm25_chunks, "_texts", {})

    def _full_text(doc_id: int) -> str:
        return bm25_texts.get(doc_id, "")

    # BM25 candidates — strip question noise for better lexical matching
    bm25_query = _clean_bm25_query(query)
    use_cosine = embed_fn is not None and store.vectors._doc_ids
    candidate_k = top_k * 2 if use_cosine else top_k
    bm25_hits = store.bm25_chunks.query(bm25_query, top_k=candidate_k)

    # Cosine candidates (if embeddings available)
    cosine_hits: list[tuple[int, float]] = []
    if use_cosine:
        try:
            query_emb = np.asarray(embed_fn(query), dtype=np.float32)
            cosine_hits = store.vectors.query(query_emb, top_k=candidate_k)
        except Exception:
            pass

    # RRF merge
    merged = _rrf_merge([bm25_hits, cosine_hits]) if cosine_hits else bm25_hits

    # Cross-encoder rerank (if reranker available)
    if reranker is not None and merged:
        candidates = []
        for doc_id, score in merged[:top_k * 2]:
            text = _full_text(int(doc_id))
            if text.strip():
                candidates.append({"text": text, "doc_id": int(doc_id), "score": score})
        if candidates:
            reranked = reranker.rerank(query, candidates, top_k=top_k)
            for item in reranked:
                if not _add(item["text"], item["doc_id"]):
                    break
            return "\n".join(evidence_parts)

    # No reranker — use merged ranking directly
    for doc_id, _score in merged[:top_k]:
        text = _full_text(int(doc_id))
        if not text.strip():
            continue
        if not _add(text, int(doc_id)):
            break

    return "\n".join(evidence_parts)


def _summarize_memory_hits(hits: list[dict[str, Any]], *, limit: int = 5) -> str:
    lines = [
        f"turn {hit['turn_id']}: {hit['text']}"
        for hit in hits[:limit]
        if str(hit.get("text", "")).strip()
    ]
    return "\n".join(lines)


class AgenticRetriever:
    """Bounded tool-driven retriever with trace capture.

    The class accepts either:
    - ``toolkit=None``: no-op / graceful empty retrieval
    - ``toolkit`` as a mapping of tool names to callables
    - ``toolkit`` as a ``FoundryStore`` (wrapped into the built-in tools)
    """

    def __init__(
        self,
        toolkit: Any,
        *,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        budget: RetrievalBudget = RetrievalBudget(),
        temperature: float = 0.0,
        max_tokens: int = 512,
        reranker: Optional[RerankerFn] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        import os
        self.llm_base_url = llm_base_url or os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.llm_api_key = llm_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.llm_model = llm_model
        self.budget = budget
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.reranker = reranker
        self.embed_fn = embed_fn
        self.dedup = DeduplicationTracker()
        self.toolkit = self._build_toolkit(toolkit)
        self.store = toolkit if isinstance(toolkit, FoundryStore) else None
        self._client: Any = None

    def _build_toolkit(self, toolkit: Any) -> Optional[dict[str, Callable[..., Any]]]:
        if toolkit is None:
            return None
        if isinstance(toolkit, Mapping):
            return {str(name): handler for name, handler in toolkit.items() if callable(handler)}
        if isinstance(toolkit, FoundryStore):
            return {
                "search_memory": lambda query, top_k=_DEFAULT_TOP_K: search_memory(
                    toolkit,
                    query,
                    dedup=self.dedup,
                    top_k=top_k,
                    reranker=self.reranker,
                    embed_fn=self.embed_fn,
                ),
                "search_entities": lambda query, top_k=_DEFAULT_TOP_K, layer=None: search_entities(
                    toolkit,
                    query,
                    top_k=top_k,
                    layer=layer,
                ),
                "get_related": lambda entity, layer=None, max_hops=1: get_related(
                    toolkit,
                    entity,
                    layer=layer,
                    max_hops=max_hops,
                ),
                "search_temporal": lambda query, top_k=5, entities=None: search_temporal(
                    toolkit,
                    query,
                    top_k=top_k,
                    entities=entities,
                ),
                "read_turn": lambda turn_id: read_turn(toolkit, int(turn_id)),
                "aggregate": lambda entity_type, op, property_predicate="": aggregate(
                    toolkit,
                    entity_type,
                    op,
                    property_predicate=property_predicate,
                    limit=self.budget.aggregation_overfetch,
                ),
                "execute_python": execute_python,
                "answer": answer,
            }
        return None

    def _ensure_client(self) -> Any:
        """Lazily create an OpenAI-compatible client for the ReAct loop."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.llm_base_url,
                api_key=self.llm_api_key,
            )
        return self._client

    def _record_step(
        self,
        trace: RetrievalTrace,
        tool_name: str,
        arguments: dict[str, Any],
        observation: Any,
        started_at: float,
    ) -> None:
        rendered = str(observation)
        if len(rendered) > self.budget.max_observation_chars:
            rendered = rendered[: self.budget.max_observation_chars]
        trace.steps.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "observation": rendered,
                "result": observation,
                "duration_ms": (time.perf_counter() - started_at) * 1000.0,
            }
        )

    def _execute_tool(self, name: str, *, trace: RetrievalTrace, **kwargs: Any) -> Any:
        if self.toolkit is None or name not in self.toolkit:
            raise KeyError(f"Unknown tool: {name}")
        started_at = time.perf_counter()
        observation = self.toolkit[name](**kwargs)
        self._record_step(trace, name, kwargs, observation, started_at)
        return observation

    def _compress_observations(self, trace: RetrievalTrace) -> None:
        if len(trace.steps) <= self.budget.sliding_window_after:
            return
        prefix = trace.steps[:-self.budget.sliding_window_after]
        suffix = trace.steps[-self.budget.sliding_window_after :]
        # Compress observation strings for logging but preserve raw results
        # so budget-exhaustion evidence can extract actual text content.
        all_results: list[Any] = [s.get("result") for s in prefix if s.get("result") is not None]
        summary = "; ".join(f"{step['tool']}({step['arguments']})" for step in prefix)
        trace.steps = [
            {
                "tool": "summary",
                "arguments": {},
                "observation": summary[: self.budget.max_observation_chars],
                "result": all_results,
            }
        ] + suffix

    def _deterministic_retrieve(
        self, trace: RetrievalTrace, started: float
    ) -> tuple[str, RetrievalTrace]:
        """Deterministic fallback when no LLM is available.

        Runs search_memory → search_entities → search_temporal in order,
        collects evidence, then calls answer.
        """
        evidence_parts: list[str] = []
        planned: list[tuple[str, dict[str, Any]]] = [
            ("search_memory", {"query": trace.question, "top_k": 5}),
        ]
        if "search_entities" in self.toolkit:
            planned.append(("search_entities", {"query": trace.question, "top_k": 5}))
        if "search_temporal" in self.toolkit:
            planned.append(("search_temporal", {"query": trace.question, "top_k": 5}))

        for tool_name, arguments in planned[: self.budget.max_steps]:
            if (time.perf_counter() - started) * 1000.0 >= self.budget.max_total_ms:
                trace.terminated_by = "budget"
                break
            observation = self._execute_tool(tool_name, trace=trace, **arguments)
            if tool_name == "search_memory":
                ev = _summarize_memory_hits(observation, limit=5) if isinstance(observation, list) else ""
                if ev:
                    evidence_parts.append(ev)
            elif tool_name == "search_temporal":
                lines = [
                    f"{row['event']} @ {row['date']}"
                    for row in (observation.get("matched_events", []) if isinstance(observation, dict) else [])
                ]
                if lines:
                    evidence_parts.append("\n".join(lines))
            elif tool_name == "search_entities":
                entity_lines: list[str] = []
                for hit in (observation if isinstance(observation, list) else []):
                    entity_lines.append(hit["entity"])
                    for fact in hit.get("facts", [])[:2]:
                        entity_lines.append(f"{hit['entity']}: {fact['predicate']} {fact['object']}")
                if entity_lines:
                    evidence_parts.append("\n".join(entity_lines))

            if evidence_parts and "answer" in self.toolkit:
                joined = _OBSERVATION_SEPARATOR.join(p for p in evidence_parts if p.strip())
                final = self._execute_tool("answer", trace=trace, evidence=joined)
                trace.evidence = str(final)
                trace.terminated_by = "answer"
                trace.total_ms = (time.perf_counter() - started) * 1000.0
                return trace.evidence, trace

        if not trace.terminated_by:
            trace.evidence = _OBSERVATION_SEPARATOR.join(p for p in evidence_parts if p.strip())
            trace.terminated_by = "budget" if trace.evidence else "no_action"

        trace.total_ms = (time.perf_counter() - started) * 1000.0
        return trace.evidence, trace

    def retrieve(self, question: str) -> tuple[str, RetrievalTrace]:
        """Run a ReAct loop: LLM decides which tools to call.

        Falls back to a single search_memory call when no LLM is
        available (missing API key or import error).
        """
        import json as _json
        import logging

        _log = logging.getLogger(__name__)

        trace = RetrievalTrace(question=str(question or ""))
        started = time.perf_counter()
        self.dedup.reset()

        if not trace.question.strip() or self.toolkit is None:
            trace.terminated_by = "no_action"
            return "", trace

        # --- Fallback: no LLM available → deterministic tool plan -----------
        try:
            client = self._ensure_client()
        except Exception:
            client = None

        if client is None or not self.llm_api_key:
            return self._deterministic_retrieve(trace, started)

        # --- ReAct loop: LLM picks tools -----------------------------------
        system = _REACT_SYSTEM_PROMPT.format(
            tool_descriptions=_build_tool_descriptions(),
            max_steps=self.budget.max_steps,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": trace.question},
        ]

        for step_i in range(self.budget.max_steps):
            if (time.perf_counter() - started) * 1000.0 >= self.budget.max_total_ms:
                trace.terminated_by = "budget"
                break

            try:
                resp = client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                llm_text = resp.choices[0].message.content or ""
            except Exception as exc:
                _log.warning("LLM call failed at step %d: %s", step_i, exc)
                trace.terminated_by = "error"
                break

            thought, tool_name, raw_args = parse_react_output(llm_text)

            if not tool_name:
                _log.info("No action parsed at step %d, using text as evidence", step_i)
                trace.evidence = llm_text
                trace.terminated_by = "no_action"
                break

            # Terminal tool: answer
            if tool_name == "answer":
                trace.steps.append({
                    "tool": "answer",
                    "arguments": {"evidence": raw_args},
                    "observation": raw_args,
                    "duration_ms": 0.0,
                })
                trace.evidence = raw_args
                trace.terminated_by = "answer"
                break

            # Parse positional args → kwargs and execute
            kwargs = _parse_react_args(tool_name, raw_args)
            if tool_name not in self.toolkit:
                obs_text = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    observation = self._execute_tool(tool_name, trace=trace, **kwargs)
                    obs_text = _format_observation(observation)
                except Exception as exc:
                    obs_text = f"Error: {exc}"
                    self._record_step(trace, tool_name, kwargs, obs_text, time.perf_counter())

            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": f"Observation: {obs_text}"})

            self._compress_observations(trace)

        else:
            # Budget exhausted — extract actual content from raw results.
            # Summary steps pack multiple results into a list; flatten them.
            trace.terminated_by = "budget"
            all_texts: list[str] = []

            def _extract_texts(obj: Any) -> None:
                if isinstance(obj, list):
                    for item in obj:
                        _extract_texts(item)
                elif isinstance(obj, dict) and "text" in obj:
                    t = str(obj["text"]).strip()
                    if t:
                        all_texts.append(t)

            for s in trace.steps:
                result = s.get("result")
                if result is not None:
                    _extract_texts(result)
            trace.evidence = "\n---\n".join(all_texts) if all_texts else ""

        trace.total_ms = (time.perf_counter() - started) * 1000.0
        return trace.evidence, trace


__all__ = [
    "TOOLS",
    "DeduplicationTracker",
    "AgenticRetriever",
    "RetrievalTrace",
    "RetrievalBudget",
]
