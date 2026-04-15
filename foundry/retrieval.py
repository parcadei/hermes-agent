"""Retrieval primitives for the Hermes Foundry memory system.

This module implements the Phase 6 retrieval contract from ``foundry/spec.md``.
It provides:

- thin tool wrappers over the storage and query layers
- within-query deduplication of turn observations
- deterministic regex-based routing for common question classes
- a bounded, tool-driven retriever scaffold that does not require an LLM

The implementation deliberately stays deterministic and stdlib-first. When no
runtime LLM/tool orchestration is available, ``AgenticRetriever`` degrades
gracefully to deterministic tool execution and trace capture.
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


_RRF_K = 60.0
_DEFAULT_TOP_K = 10
_OBSERVATION_SEPARATOR = "\n\n"
_VALID_LAYERS = {layer.value for layer in GroundingLayer}
RerankerFn = Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]
_TEMPORAL_PATTERNS = (
    re.compile(r"\bwhen\b", re.IGNORECASE),
    re.compile(r"\bbefore\b|\bafter\b|\bduring\b|\bbetween\b", re.IGNORECASE),
    re.compile(r"\bhow long\b|\bhow far\b|\bago\b", re.IGNORECASE),
    re.compile(r"\bdate\b|\btime\b|\blast\b|\bfirst\b", re.IGNORECASE),
)
_AGGREGATION_PATTERNS = (
    re.compile(r"\bhow many\b", re.IGNORECASE),
    re.compile(r"\bcount\b|\bnumber of\b|\btotal\b", re.IGNORECASE),
    re.compile(r"\blist all\b|\bwhich .* are\b", re.IGNORECASE),
)
_PREFERENCE_PATTERNS = (
    re.compile(r"\bprefer\b", re.IGNORECASE),
    re.compile(r"\bfavorite\b|\bfavourite\b|\blikes? better\b", re.IGNORECASE),
)
_COMPARISON_PATTERNS = (
    re.compile(r"\bcompare\b|\bversus\b|\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bmore\b|\bless\b|\bbetter\b|\bworse\b|\bcheaper\b|\bolder\b", re.IGNORECASE),
)
_SPATIAL_PATTERNS = (
    re.compile(r"\bwhere\b", re.IGNORECASE),
    re.compile(r"\blocated\b|\blives?\b|\bstays?\b|\bvisited\b", re.IGNORECASE),
)
_SOCIAL_PATTERNS = (
    re.compile(r"\bfriend\b|\bfamily\b|\bcolleague\b|\bknows?\b", re.IGNORECASE),
    re.compile(r"\bconnected\b|\brelationship\b|\brelated to\b", re.IGNORECASE),
)
_META_PATTERNS = (
    re.compile(r"\bdo(es)? (the )?user know\b", re.IGNORECASE),
    re.compile(r"\bhas\b|\bhave\b", re.IGNORECASE),
)
_MULTI_HOP_START_RE = re.compile(r"^(what|which|who|where|when|how)\b", re.IGNORECASE)
_MULTI_HOP_PHRASES = (
    "country of origin",
    "creator of",
    "founder of",
    "parent of",
    "spouse of",
    "capital of",
    "birthplace of",
    "played by",
    "followed by",
    "created by",
    "founded by",
    "written by",
    "directed by",
)
_MULTI_HOP_RELATION_TERMS = (
    "born",
    "followed",
    "played",
    "created",
    "founded",
    "written",
    "directed",
)
_TURN_PREFIX_RE = re.compile(r"^turn\s+\d+\s*:\s*", re.IGNORECASE)
_ANSWER_FROM_EVIDENCE_PATTERNS = (
    re.compile(r"\bplayed\s+(?P<answer>[^.?!,;]+)", re.IGNORECASE),
    re.compile(r"\bfollow(?:ed|s)?\s+(?P<answer>[^.?!,;]+)", re.IGNORECASE),
    re.compile(r"\blaunch(?:ed|es)?\s+(?P<answer>[^.?!,;]+)", re.IGNORECASE),
    re.compile(r"\bcreated by\s+(?P<answer>[^.?!,;]+)", re.IGNORECASE),
    re.compile(r"\bfounded by\s+(?P<answer>[^.?!,;]+)", re.IGNORECASE),
    re.compile(r"\bwritten by\s+(?P<answer>[^.?!,;]+)", re.IGNORECASE),
    re.compile(r"\bdirected by\s+(?P<answer>[^.?!,;]+)", re.IGNORECASE),
    re.compile(r"\bborn in\s+(?P<answer>[^.?!,;]+)", re.IGNORECASE),
    re.compile(r"\boriginated in\s+(?P<answer>[^.?!,;]+)", re.IGNORECASE),
    re.compile(r"\bfrom\s+(?P<answer>[^.?!,;]+)", re.IGNORECASE),
    re.compile(r"\b(?:is|are|was|were)\s+(?P<answer>[^.?!]+)", re.IGNORECASE),
)
_SPATIAL_FACT_PREDICATES = {
    "at",
    "birth_place",
    "birthplace",
    "born",
    "born_in",
    "from",
    "lives_in",
    "location",
    "place_of_birth",
    "visited",
}


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
            "Count, sum, average, distinct-list, or enumerate entities of a given "
            "type. Best for how-many and simple aggregate questions."
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


def _build_alias_index(store: FoundryStore) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for entity in store.entity_graph.all_entities():
        alias_set = {_normalize_name(alias) for alias in entity.aliases}
        alias_set.add(entity.canonical)
        aliases[entity.canonical] = alias_set
    return aliases


def augment_query(query: str, store: Optional[FoundryStore]) -> str:
    """Expand the query with known aliases when possible."""

    if store is None:
        return query
    alias_index = _build_alias_index(store)
    expanded: list[str] = []
    for token in query.split():
        normalized = _normalize_name(token)
        entity = store.entity_graph.get_entity(normalized)
        if entity is None:
            expanded.append(token)
            continue
        aliases = sorted(alias_index.get(entity.canonical, {entity.canonical}))
        expanded.append(" ".join([token, *aliases]))
    return " ".join(expanded)


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
) -> list[dict[str, Any]]:
    results = search_memory_hybrid(
        query,
        store.bm25_chunks,
        store.vectors,
        reranker=reranker,
        seen_turns=dedup.seen_turns if dedup is not None else None,
        top_k=top_k,
        chunk_store=store.chunks,
        embed_fn=embed_fn,
        triadic=store.triadic,
    )
    if dedup is not None:
        return dedup.filter_new(results)
    return results


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
    plan = QueryPlan(store).type(entity_type)
    entities = plan.execute()

    if normalized_op == "count":
        result: Any = len(entities)
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


def _contains_pattern(question: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(question) for pattern in patterns)


def _extract_type_hint(question: str) -> str:
    lowered = question.lower()
    match = re.search(r"\bhow many ([a-z_ ]+?)\b(?: does| do| are| were| has| have|\?|$)", lowered)
    if match:
        return match.group(1).strip()
    match = re.search(r"\bnumber of ([a-z_ ]+?)\b(?: does| do| are| were|\?|$)", lowered)
    if match:
        return match.group(1).strip()
    return ""


def _extract_entity_candidates(question: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    normalized = []
    for candidate in candidates:
        canonical = _normalize_name(candidate)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _pipeline_confidence(evidence: str, trace: RetrievalTrace) -> float:
    if not evidence.strip():
        return 0.0
    base = 0.5 if trace.terminated_by == "answer" else 0.1 if trace.terminated_by == "budget" else 0.0
    efficiency = max(0.0, 0.3 - (len(trace.steps) / max(RetrievalBudget().max_steps, 1)) * 0.3)
    evidence_bonus = 0.2 if len(evidence.strip()) > 20 else 0.0
    return min(1.0, base + efficiency + evidence_bonus)


def _summarize_memory_hits(hits: list[dict[str, Any]], *, limit: int = 5) -> str:
    lines = [
        f"turn {hit['turn_id']}: {hit['text']}"
        for hit in hits[:limit]
        if str(hit.get("text", "")).strip()
    ]
    return "\n".join(lines)


def _looks_multi_hop(question: str) -> bool:
    normalized = str(question or "").strip().lower()
    if not normalized or _MULTI_HOP_START_RE.search(normalized) is None:
        return False
    phrase_hits = sum(1 for phrase in _MULTI_HOP_PHRASES if phrase in normalized)
    if phrase_hits >= 2:
        return True
    if phrase_hits >= 1 and any(term in normalized for term in _MULTI_HOP_RELATION_TERMS):
        return True
    if phrase_hits >= 1 and normalized.count(" of ") >= 2:
        return True
    return bool(re.search(r"\bof the\b.*\b(?:by|followed by|played by|created by|founded by)\b", normalized))


def _substitute_answers(template: str, answers: Sequence[str]) -> str:
    resolved = str(template or "")
    for index, answer in enumerate(answers, start=1):
        resolved = resolved.replace(f"{{answer_{index}}}", answer)
    return resolved


def _extract_short_answer(question: str, evidence: str) -> str:
    del question
    lines = [line.strip() for line in str(evidence or "").splitlines() if line.strip()]
    for line in lines:
        normalized = _TURN_PREFIX_RE.sub("", line)
        candidate_segments = [normalized]
        if ":" in normalized:
            candidate_segments.append(normalized.split(":", 1)[1].strip())
        for segment in candidate_segments:
            for pattern in _ANSWER_FROM_EVIDENCE_PATTERNS:
                match = pattern.search(segment)
                if match:
                    answer = match.group("answer").strip(" .,:;!?\"'")
                    if answer:
                        return answer
    return ""


class DeterministicRouter:
    """Routes questions to hardcoded retrieval pipelines."""

    def __init__(
        self,
        store: Optional[FoundryStore] = None,
        *,
        budget: RetrievalBudget = RetrievalBudget(),
        reranker: Optional[RerankerFn] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        llm_url: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.store = store
        self.budget = budget
        self.reranker = reranker
        self.embed_fn = embed_fn
        self.llm_url = llm_url
        self.llm_model = llm_model

    def route(self, question: str) -> str:
        normalized = str(question or "").strip()
        if not normalized:
            return "agentic"

        lowered = normalized.lower()
        if _looks_multi_hop(lowered):
            return "multi_hop"
        return self._route_without_multi_hop(normalized)

    def _route_without_multi_hop(self, question: str) -> str:
        normalized = str(question or "").strip()
        if not normalized:
            return "agentic"

        lowered = normalized.lower()
        if _contains_pattern(lowered, _TEMPORAL_PATTERNS):
            return "temporal"
        if _contains_pattern(lowered, _AGGREGATION_PATTERNS):
            return "aggregation"
        if _contains_pattern(lowered, _PREFERENCE_PATTERNS):
            return "preference"
        if _contains_pattern(lowered, _COMPARISON_PATTERNS):
            return "comparison"
        if _contains_pattern(lowered, _SPATIAL_PATTERNS):
            return "spatial" if "where" in lowered else "fact_lookup"
        if _contains_pattern(lowered, _SOCIAL_PATTERNS):
            return "social_relational"
        if _contains_pattern(lowered, _META_PATTERNS) and ("remember" in lowered or "know" in lowered or "have" in lowered):
            return "meta_cognitive"
        if re.search(r"^(what|which|who|where|does|is|are|did|do)\b", lowered):
            return "fact_lookup"
        return "agentic"

    def execute(self, question: str, pipeline: str) -> tuple[str, Any]:
        trace = RetrievalTrace(question=question)
        if self.store is None:
            trace.terminated_by = "error"
            return "", {"pipeline": pipeline, "confidence": 0.0, "trace": trace, "fallback": "agentic"}

        handlers: dict[str, Callable[[str, RetrievalTrace], str]] = {
            "fact_lookup": self._execute_fact_lookup,
            "temporal": self._execute_temporal,
            "aggregation": self._execute_aggregation,
            "preference": self._execute_preference,
            "comparison": self._execute_comparison,
            "spatial": self._execute_spatial,
            "social_relational": self._execute_social_relational,
            "meta_cognitive": self._execute_meta_cognitive,
            "multi_hop": self._execute_multi_hop,
        }
        handler = handlers.get(pipeline)
        if handler is None:
            trace.terminated_by = "no_action"
            return "", {"pipeline": pipeline, "confidence": 0.0, "trace": trace, "fallback": "agentic"}

        started = time.perf_counter()
        evidence = handler(question, trace)
        trace.total_ms = (time.perf_counter() - started) * 1000.0

        if not self._validate(evidence, trace):
            expanded = augment_query(question, self.store)
            if expanded != question:
                retry_trace = RetrievalTrace(question=expanded)
                evidence = handler(expanded, retry_trace)
                retry_trace.total_ms = (time.perf_counter() - started) * 1000.0
                if self._validate(evidence, retry_trace):
                    return evidence, {
                        "pipeline": pipeline,
                        "confidence": _pipeline_confidence(evidence, retry_trace),
                        "trace": retry_trace,
                        "fallback": None,
                    }
            return "", {"pipeline": pipeline, "confidence": 0.0, "trace": trace, "fallback": "agentic"}

        return evidence, {
            "pipeline": pipeline,
            "confidence": _pipeline_confidence(evidence, trace),
            "trace": trace,
            "fallback": None,
        }

    def _validate(self, evidence: str, trace: RetrievalTrace) -> bool:
        return bool(evidence.strip()) and _pipeline_confidence(evidence, trace) > 0.3

    def _record(self, trace: RetrievalTrace, tool_name: str, arguments: dict[str, Any], observation: Any) -> None:
        rendered = str(observation)
        if len(rendered) > self.budget.max_observation_chars:
            rendered = rendered[: self.budget.max_observation_chars]
        trace.steps.append({"tool": tool_name, "arguments": arguments, "observation": rendered})

    def _execute_fact_lookup(self, question: str, trace: RetrievalTrace) -> str:
        hits = search_memory(
            self.store,
            question,
            top_k=5,
            reranker=self.reranker,
            embed_fn=self.embed_fn,
        )
        self._record(trace, "search_memory", {"query": question}, hits)
        evidence = _summarize_memory_hits(hits)
        trace.evidence = evidence
        trace.terminated_by = "answer" if evidence else "budget"
        return evidence

    def _execute_temporal(self, question: str, trace: RetrievalTrace) -> str:
        temporal_hits = search_temporal(self.store, question, top_k=5)
        memory_hits = search_memory(
            self.store,
            question,
            top_k=3,
            reranker=self.reranker,
            embed_fn=self.embed_fn,
        )
        self._record(trace, "search_temporal", {"query": question}, temporal_hits)
        self._record(trace, "search_memory", {"query": question}, memory_hits)
        lines = [
            f"{row['event']} @ {row['date']}"
            for row in temporal_hits.get("matched_events", [])
        ]
        if memory_hits:
            lines.append(_summarize_memory_hits(memory_hits, limit=2))
        evidence = "\n".join(line for line in lines if line)
        trace.evidence = evidence
        trace.terminated_by = "answer" if evidence else "budget"
        return evidence

    def _execute_aggregation(self, question: str, trace: RetrievalTrace) -> str:
        type_hint = _extract_type_hint(question)
        if not type_hint:
            trace.terminated_by = "no_action"
            return ""
        payload = aggregate(
            self.store,
            type_hint,
            "count" if "how many" in question.lower() or "count" in question.lower() else "list",
            limit=self.budget.aggregation_overfetch,
        )
        self._record(trace, "aggregate", {"entity_type": type_hint}, payload)
        if isinstance(payload["result"], list):
            evidence = ", ".join(payload["result"])
        else:
            evidence = str(payload["result"])
        trace.evidence = evidence
        trace.terminated_by = "answer" if evidence else "budget"
        return evidence

    def _execute_preference(self, question: str, trace: RetrievalTrace) -> str:
        hits = search_entities(self.store, question, top_k=5)
        self._record(trace, "search_entities", {"query": question}, hits)
        evidence_parts = []
        for hit in hits:
            sentiment_facts = [
                fact for fact in hit["facts"] if fact["predicate"] in {"likes", "dislikes", "prefers", "favorite"}
            ]
            if sentiment_facts:
                evidence_parts.append(f"{hit['entity']}: {sentiment_facts[0]['predicate']} {sentiment_facts[0]['object']}")
        evidence = "\n".join(evidence_parts)
        trace.evidence = evidence
        trace.terminated_by = "answer" if evidence else "budget"
        return evidence

    def _execute_comparison(self, question: str, trace: RetrievalTrace) -> str:
        candidates = _extract_entity_candidates(question)
        if len(candidates) < 2:
            trace.terminated_by = "no_action"
            return ""
        rows = []
        for candidate in candidates[:2]:
            related = get_related(self.store, candidate)
            self._record(trace, "get_related", {"entity": candidate}, related)
            summary = ", ".join(
                f"{fact['predicate']}={fact['object']}"
                for fact in related.get("properties", [])[:3]
            )
            rows.append(f"{candidate}: {summary}")
        evidence = "\n".join(rows)
        trace.evidence = evidence
        trace.terminated_by = "answer" if evidence else "budget"
        return evidence

    def _execute_spatial(self, question: str, trace: RetrievalTrace) -> str:
        hits = search_entities(self.store, question, top_k=5)
        self._record(trace, "search_entities", {"query": question}, hits)
        lines = []
        for hit in hits:
            for fact in hit["facts"]:
                if fact["predicate"] in _SPATIAL_FACT_PREDICATES:
                    lines.append(f"{hit['entity']}: {fact['predicate']} {fact['object']}")
        evidence = "\n".join(lines)
        trace.evidence = evidence
        trace.terminated_by = "answer" if evidence else "budget"
        return evidence

    def _execute_social_relational(self, question: str, trace: RetrievalTrace) -> str:
        candidates = _extract_entity_candidates(question)
        entity_name = candidates[0] if candidates else question
        related = get_related(self.store, entity_name, max_hops=2)
        self._record(trace, "get_related", {"entity": entity_name, "max_hops": 2}, related)
        neighbors = related.get("neighbors", [])
        evidence = ", ".join(neighbors[:10])
        trace.evidence = evidence
        trace.terminated_by = "answer" if evidence else "budget"
        return evidence

    def _execute_meta_cognitive(self, question: str, trace: RetrievalTrace) -> str:
        hits = search_memory(self.store, question, top_k=3, reranker=self.reranker, embed_fn=self.embed_fn)
        self._record(trace, "search_memory", {"query": question}, hits)
        evidence = _summarize_memory_hits(hits, limit=3)
        trace.evidence = evidence
        trace.terminated_by = "answer" if evidence else "budget"
        return evidence

    def _execute_multi_hop(self, question: str, trace: RetrievalTrace) -> str:
        subqueries = [question]
        if self.llm_url:
            subqueries = decompose_query(
                question,
                llm_url=self.llm_url,
                llm_model=self.llm_model,
                max_hops=3,
            )
        self._record(trace, "decompose_query", {"question": question}, subqueries)
        if not subqueries:
            trace.terminated_by = "no_action"
            return ""
        if len(subqueries) == 1 and str(subqueries[0]).strip() == str(question).strip():
            trace.terminated_by = "no_action"
            return ""
        handlers: dict[str, Callable[[str, RetrievalTrace], str]] = {
            "fact_lookup": self._execute_fact_lookup,
            "temporal": self._execute_temporal,
            "aggregation": self._execute_aggregation,
            "preference": self._execute_preference,
            "comparison": self._execute_comparison,
            "spatial": self._execute_spatial,
            "social_relational": self._execute_social_relational,
            "meta_cognitive": self._execute_meta_cognitive,
        }
        answers: list[str] = []
        final_evidence = ""
        for index, template in enumerate(subqueries, start=1):
            resolved_question = _substitute_answers(template, answers)
            if "{answer_" in resolved_question:
                trace.terminated_by = "budget"
                break
            pipeline = self._route_without_multi_hop(resolved_question)
            if pipeline == "agentic":
                trace.terminated_by = "no_action"
                final_evidence = ""
                break
            handler = handlers.get(pipeline)
            if handler is None:
                trace.terminated_by = "no_action"
                final_evidence = ""
                break
            final_evidence = handler(resolved_question, trace)
            if not final_evidence.strip():
                trace.terminated_by = "budget"
                break
            if index < len(subqueries):
                answer = _extract_short_answer(resolved_question, final_evidence)
                self._record(
                    trace,
                    "resolve_subquery",
                    {"query": resolved_question, "pipeline": pipeline},
                    {"answer": answer},
                )
                if not answer:
                    trace.terminated_by = "budget"
                    final_evidence = ""
                    break
                answers.append(answer)
        trace.evidence = final_evidence
        trace.terminated_by = "answer" if final_evidence else (trace.terminated_by or "budget")
        return final_evidence


class AgenticRetriever:
    """Bounded tool-driven retriever with trace capture.

    The class accepts either:
    - ``toolkit=None``: no-op / graceful empty retrieval
    - ``toolkit`` as a mapping of tool names to callables
    - ``toolkit`` as a ``FoundryStore`` (wrapped into the built-in tools)
    """

    early_termination_after: int = 4

    def __init__(
        self,
        toolkit: Any,
        *,
        llm_model: str = "gpt-4o-mini",
        budget: RetrievalBudget = RetrievalBudget(),
        temperature: float = 0.0,
        max_tokens: int = 512,
        reranker: Optional[RerankerFn] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        self.llm_model = llm_model
        self.budget = budget
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.reranker = reranker
        self.embed_fn = embed_fn
        self.dedup = DeduplicationTracker()
        self.toolkit = self._build_toolkit(toolkit)
        self.store = toolkit if isinstance(toolkit, FoundryStore) else None
        self.router = DeterministicRouter(
            self.store,
            budget=budget,
            reranker=reranker,
            embed_fn=embed_fn,
        )

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
        summary = "; ".join(f"{step['tool']}({step['arguments']})" for step in prefix)
        trace.steps = [
            {"tool": "summary", "arguments": {}, "observation": summary[: self.budget.max_observation_chars]}
        ] + suffix

    def retrieve(self, question: str) -> tuple[str, RetrievalTrace]:
        trace = RetrievalTrace(question=str(question or ""))
        started = time.perf_counter()
        self.dedup.reset()

        if not trace.question.strip():
            trace.terminated_by = "no_action"
            return "", trace

        if self.toolkit is None:
            trace.terminated_by = "no_action"
            return "", trace

        evidence_parts: list[str] = []
        route = self.router.route(trace.question)
        planned_tools: list[tuple[str, dict[str, Any]]] = [("search_memory", {"query": trace.question, "top_k": 5})]

        if route == "temporal" and "search_temporal" in self.toolkit:
            planned_tools.append(("search_temporal", {"query": trace.question, "top_k": 5}))
        elif route == "aggregation" and "aggregate" in self.toolkit:
            type_hint = _extract_type_hint(trace.question)
            if type_hint:
                planned_tools.append(("aggregate", {"entity_type": type_hint, "op": "count"}))
        elif route in {"preference", "comparison", "spatial", "social_relational", "multi_hop"}:
            planned_tools.append(("search_entities", {"query": trace.question, "top_k": 5}))

        for step_idx, (tool_name, arguments) in enumerate(planned_tools[: self.budget.max_steps], start=1):
            if (time.perf_counter() - started) * 1000.0 >= self.budget.max_total_ms:
                trace.terminated_by = "budget"
                break
            observation = self._execute_tool(tool_name, trace=trace, **arguments)
            if tool_name == "search_memory":
                evidence = _summarize_memory_hits(observation, limit=5)
                if evidence:
                    evidence_parts.append(evidence)
            elif tool_name == "search_temporal":
                lines = [
                    f"{row['event']} @ {row['date']}"
                    for row in observation.get("matched_events", [])
                ]
                if lines:
                    evidence_parts.append("\n".join(lines))
            elif tool_name == "aggregate":
                evidence_parts.append(str(observation.get("result", "")))
            elif tool_name == "search_entities":
                entity_lines = []
                for hit in observation:
                    entity_lines.append(hit["entity"])
                    for fact in hit.get("facts", [])[:2]:
                        entity_lines.append(f"{hit['entity']}: {fact['predicate']} {fact['object']}")
                if entity_lines:
                    evidence_parts.append("\n".join(entity_lines))

            if evidence_parts and "answer" in self.toolkit:
                joined = _OBSERVATION_SEPARATOR.join(part for part in evidence_parts if part.strip())
                final = self._execute_tool("answer", trace=trace, evidence=joined)
                trace.evidence = str(final)
                trace.terminated_by = "answer"
                break

            if step_idx >= self.early_termination_after:
                break

            self._compress_observations(trace)

        if not trace.terminated_by:
            trace.evidence = _OBSERVATION_SEPARATOR.join(part for part in evidence_parts if part.strip())
            trace.terminated_by = "budget" if trace.evidence else "no_action"

        trace.total_ms = (time.perf_counter() - started) * 1000.0
        return trace.evidence, trace


__all__ = [
    "TOOLS",
    "DeduplicationTracker",
    "AgenticRetriever",
    "RetrievalTrace",
    "DeterministicRouter",
    "RetrievalBudget",
]
