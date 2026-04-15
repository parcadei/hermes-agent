"""Storage primitives and indexes for the Hermes Foundry memory system.

This module implements the Phase 2 storage contract from ``foundry/spec.md``.
It keeps the implementation deterministic and stdlib-first while exposing the
data structures and indexes that later layers build on.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, ClassVar, Iterable, Mapping, Optional
import uuid

import numpy as np

from foundry.ontology import EntityType, GroundingLayer, PropertyType


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_NUMERIC_EXTRACT_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_ISO_Z_SUFFIX = "Z"
_TRIADIC_DERIVED_METADATA_KEY = "__foundry_internal_triadic__"
_DEFAULT_CONFIG_SNAPSHOT = {
    "chunk_size": 512,
    "extraction_model": "Qwen/Qwen2.5-7B-Instruct",
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
}


def _normalize_name(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    return "_".join(value.strip().lower().split())


def _normalize_predicate(value: str) -> str:
    return _normalize_name(value)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(_ISO_Z_SUFFIX):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def _parse_numeric(value: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMERIC_EXTRACT_RE.search(str(value))
    if not match:
        return None
    number = match.group(0).replace(",", "")
    try:
        return float(number)
    except ValueError:
        return None


class SchemaVersionError(RuntimeError):
    """Raised when a serialized store was written with a different schema."""


class PropertySearchability(str, Enum):
    """Controls how a property participates in search."""

    FULL = "full"
    STRUCTURED = "structured"
    NONE = "none"


@dataclass
class Fact:
    """A single predicate-object fact about an entity."""

    id: str
    subject: str
    predicate: str
    object: str
    object_type: PropertyType
    turn_idx: int
    chunk_idx: int = -1
    timestamp: Optional[datetime] = None
    session_time: Optional[datetime] = None
    source_text: str = ""
    layer: GroundingLayer = GroundingLayer.STATED
    superseded: bool = False
    superseded_by: Optional[str] = None
    when: str = ""
    role: str = ""
    sub_role: str = ""

    def __post_init__(self) -> None:
        self.id = str(self.id or uuid.uuid4())
        self.subject = _normalize_name(self.subject)
        self.predicate = _normalize_predicate(self.predicate)
        if not isinstance(self.object_type, PropertyType):
            self.object_type = PropertyType(self.object_type)
        if not isinstance(self.layer, GroundingLayer):
            self.layer = GroundingLayer(self.layer)
        self.timestamp = _parse_datetime(self.timestamp)
        self.session_time = _parse_datetime(self.session_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "object_type": self.object_type.value,
            "turn_idx": self.turn_idx,
            "chunk_idx": self.chunk_idx,
            "timestamp": _serialize_datetime(self.timestamp),
            "session_time": _serialize_datetime(self.session_time),
            "source_text": self.source_text,
            "layer": self.layer.value,
            "superseded": self.superseded,
            "superseded_by": self.superseded_by,
            "when": self.when,
            "role": self.role,
            "sub_role": self.sub_role,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Fact":
        return cls(
            id=payload["id"],
            subject=payload["subject"],
            predicate=payload["predicate"],
            object=payload["object"],
            object_type=PropertyType(payload["object_type"]),
            turn_idx=payload["turn_idx"],
            chunk_idx=payload.get("chunk_idx", -1),
            timestamp=_parse_datetime(payload.get("timestamp")),
            session_time=_parse_datetime(payload.get("session_time")),
            source_text=payload.get("source_text", ""),
            layer=GroundingLayer(payload.get("layer", GroundingLayer.STATED.value)),
            superseded=payload.get("superseded", False),
            superseded_by=payload.get("superseded_by"),
            when=payload.get("when", ""),
            role=payload.get("role", ""),
            sub_role=payload.get("sub_role", ""),
        )


@dataclass
class EntityNode:
    """A node in the entity graph representing one real-world entity."""

    canonical: str
    aliases: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)
    facts: dict[str, list[Fact]] = field(default_factory=dict)
    links: dict[str, set[str]] = field(default_factory=dict)
    related: set[str] = field(default_factory=set)
    mention_turns: set[int] = field(default_factory=set)
    mention_chunks: set[int] = field(default_factory=set)
    _type_counts: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.canonical = _normalize_name(self.canonical)
        self.types = {EntityType.normalize(t) for t in self.types if EntityType.normalize(t)}
        if not self._type_counts:
            self._type_counts = {type_label: 1 for type_label in self.types}
        self.links = {
            _normalize_predicate(predicate): {_normalize_name(target) for target in targets}
            for predicate, targets in self.links.items()
        }
        self.related = {_normalize_name(target) for target in self.related}
        self.facts = {
            _normalize_predicate(predicate): [
                fact if isinstance(fact, Fact) else Fact.from_dict(fact) for fact in facts
            ]
            for predicate, facts in self.facts.items()
        }

    def register_type(self, type_label: str) -> None:
        normalized = EntityType.normalize(type_label)
        if not normalized:
            return
        self.types.add(normalized)
        self._type_counts[normalized] = self._type_counts.get(normalized, 0) + 1

    def add_fact(self, fact: Fact) -> None:
        self.facts.setdefault(fact.predicate, []).append(fact)

    def add_link(self, predicate: str, target: str) -> None:
        normalized_predicate = _normalize_predicate(predicate)
        normalized_target = _normalize_name(target)
        self.links.setdefault(normalized_predicate, set()).add(normalized_target)
        if normalized_target != self.canonical:
            self.related.add(normalized_target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "aliases": sorted(self.aliases),
            "types": sorted(self.types),
            "facts": {
                predicate: [fact.to_dict() for fact in facts]
                for predicate, facts in sorted(self.facts.items())
            },
            "links": {
                predicate: sorted(targets)
                for predicate, targets in sorted(self.links.items())
            },
            "related": sorted(self.related),
            "mention_turns": sorted(self.mention_turns),
            "mention_chunks": sorted(self.mention_chunks),
            "_type_counts": dict(sorted(self._type_counts.items())),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityNode":
        return cls(
            canonical=payload["canonical"],
            aliases=set(payload.get("aliases", [])),
            types=set(payload.get("types", [])),
            facts={
                predicate: [Fact.from_dict(fact) for fact in facts]
                for predicate, facts in payload.get("facts", {}).items()
            },
            links={
                predicate: set(targets)
                for predicate, targets in payload.get("links", {}).items()
            },
            related=set(payload.get("related", [])),
            mention_turns=set(payload.get("mention_turns", [])),
            mention_chunks=set(payload.get("mention_chunks", [])),
            _type_counts=dict(payload.get("_type_counts", {})),
        )


@dataclass
class RawChunk:
    """A raw conversation chunk stored for retrieval."""

    chunk_idx: int
    text: str
    turn_idx: int
    role: str
    session_time: Optional[datetime] = None
    entities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.session_time = _parse_datetime(self.session_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_idx": self.chunk_idx,
            "text": self.text,
            "turn_idx": self.turn_idx,
            "role": self.role,
            "session_time": _serialize_datetime(self.session_time),
            "entities": list(self.entities),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RawChunk":
        return cls(
            chunk_idx=payload["chunk_idx"],
            text=payload["text"],
            turn_idx=payload["turn_idx"],
            role=payload["role"],
            session_time=_parse_datetime(payload.get("session_time")),
            entities=list(payload.get("entities", [])),
        )


@dataclass
class TemporalEvent:
    """A temporally-grounded event with an Allen interval."""

    event: str
    start_dt: datetime
    end_dt: datetime
    source: str
    turn_id: Optional[int] = None
    confidence: float = 1.0
    subject: Optional[str] = None
    verb: Optional[str] = None
    object: Optional[str] = None

    def __post_init__(self) -> None:
        self.start_dt = _parse_datetime(self.start_dt)  # type: ignore[assignment]
        self.end_dt = _parse_datetime(self.end_dt)  # type: ignore[assignment]
        if self.start_dt is None or self.end_dt is None:
            raise ValueError("TemporalEvent requires start_dt and end_dt")
        if self.start_dt > self.end_dt:
            raise ValueError(f"start_dt ({self.start_dt}) must be <= end_dt ({self.end_dt})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "start_dt": self.start_dt.isoformat(),
            "end_dt": self.end_dt.isoformat(),
            "source": self.source,
            "turn_id": self.turn_id,
            "confidence": self.confidence,
            "subject": self.subject,
            "verb": self.verb,
            "object": self.object,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalEvent":
        return cls(
            event=payload["event"],
            start_dt=_parse_datetime(payload["start_dt"]),
            end_dt=_parse_datetime(payload["end_dt"]),
            source=payload["source"],
            turn_id=payload.get("turn_id"),
            confidence=payload.get("confidence", 1.0),
            subject=payload.get("subject"),
            verb=payload.get("verb"),
            object=payload.get("object"),
        )


class BM25Index:
    """Okapi BM25 over tokenized documents."""

    k1: float = 1.5
    b: float = 0.75

    def __init__(self) -> None:
        self._texts: dict[int, str] = {}
        self._doc_lengths: dict[int, int] = {}
        self._term_freqs: dict[int, Counter[str]] = {}
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._doc_freqs: dict[str, int] = defaultdict(int)
        self._total_doc_length = 0

    def add(self, doc_id: int, text: str) -> None:
        tokens = _tokenize(text)
        old_length = self._doc_lengths.pop(doc_id, None)
        if old_length is not None:
            self._total_doc_length -= old_length
            old_freqs = self._term_freqs.pop(doc_id, Counter())
            for term in old_freqs:
                postings = self._postings.get(term)
                if postings is None:
                    continue
                postings.pop(doc_id, None)
                if not postings:
                    self._postings.pop(term, None)
                    self._doc_freqs.pop(term, None)
                else:
                    self._doc_freqs[term] = len(postings)

        self._texts[doc_id] = text
        freqs = Counter(tokens)
        self._term_freqs[doc_id] = freqs
        self._doc_lengths[doc_id] = len(tokens)
        self._total_doc_length += len(tokens)
        for term, freq in freqs.items():
            self._postings[term][doc_id] = freq
            self._doc_freqs[term] = len(self._postings[term])

    def query(self, q: str, top_k: int = 10) -> list[tuple[int, float]]:
        if not self._texts or top_k <= 0:
            return []
        query_terms = _tokenize(q)
        if not query_terms:
            return []

        avgdl = self._total_doc_length / len(self._texts) if self._texts else 0.0
        scores: dict[int, float] = {}
        n_docs = len(self._texts)
        score_get = scores.get
        doc_lengths = self._doc_lengths
        doc_freqs = self._doc_freqs
        postings_index = self._postings
        k1 = self.k1
        b = self.b

        for term in query_terms:
            doc_freq = doc_freqs.get(term, 0)
            if doc_freq == 0:
                continue
            idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            postings = postings_index[term]
            for doc_id, freq in postings.items():
                doc_len = doc_lengths.get(doc_id, 0)
                denom = freq + k1 * (1.0 - b + b * doc_len / (avgdl or 1.0))
                if denom <= 0:
                    continue
                scores[doc_id] = score_get(doc_id, 0.0) + (idf * ((freq * (k1 + 1.0)) / denom))

        if not scores:
            return []

        if len(scores) <= top_k:
            ranked = [(doc_id, score) for doc_id, score in scores.items() if score >= 0.0]
            ranked.sort(key=lambda item: (-item[1], item[0]))
            return ranked

        heap: list[tuple[float, int, int]] = []
        for doc_id, score in scores.items():
            if score < 0.0:
                continue
            item = (score, -doc_id, doc_id)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
                continue
            if item > heap[0]:
                heapq.heapreplace(heap, item)

        heap.sort(key=lambda item: (-item[0], item[2]))
        return [(doc_id, score) for score, _neg_doc_id, doc_id in heap]

    def __len__(self) -> int:
        return len(self._texts)

    def to_dict(self) -> dict[str, Any]:
        return {"documents": [{"doc_id": doc_id, "text": self._texts[doc_id]} for doc_id in sorted(self._texts)]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BM25Index":
        index = cls()
        for item in payload.get("documents", []):
            index.add(int(item["doc_id"]), item["text"])
        return index


class VectorIndex:
    """Dense vector similarity search using cosine similarity."""

    def __init__(self) -> None:
        self._doc_ids: list[int] = []
        self._id_to_pos: dict[int, int] = {}
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def add(self, doc_id: int, embedding: np.ndarray) -> None:
        vector = self._normalize_embedding(embedding)
        if self._matrix.size == 0:
            self._matrix = vector.reshape(1, -1)
            self._doc_ids = [doc_id]
            self._id_to_pos = {doc_id: 0}
            return

        if vector.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._matrix.shape[1]}, got {vector.shape[0]}"
            )

        if doc_id in self._id_to_pos:
            self._matrix[self._id_to_pos[doc_id]] = vector
            return

        self._matrix = np.vstack([self._matrix, vector])
        self._id_to_pos[doc_id] = len(self._doc_ids)
        self._doc_ids.append(doc_id)

    def add_batch(self, doc_ids: list[int], embeddings: np.ndarray) -> None:
        batch = np.asarray(embeddings, dtype=np.float32)
        if batch.ndim != 2:
            raise ValueError("Embeddings batch must be 2D")
        if len(doc_ids) != batch.shape[0]:
            raise ValueError("doc_ids and embeddings must have matching lengths")
        for doc_id, embedding in zip(doc_ids, batch):
            self.add(int(doc_id), embedding)

    def query(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        if top_k <= 0 or not self._doc_ids:
            return []
        query = self._normalize_embedding(query_embedding)
        if self._matrix.size == 0 or query.shape[0] != self._matrix.shape[1]:
            return []
        scores = self._matrix @ query
        if scores.size == 0:
            return []
        top_k = min(top_k, len(self._doc_ids))
        indices = np.argsort(-scores, kind="mergesort")[:top_k]
        return [(self._doc_ids[int(idx)], float(scores[int(idx)])) for idx in indices]

    def to_metadata_dict(self) -> dict[str, Any]:
        return {"doc_ids": list(self._doc_ids), "shape": list(self._matrix.shape)}

    @classmethod
    def from_parts(cls, doc_ids: list[int], matrix: np.ndarray) -> "VectorIndex":
        index = cls()
        if not doc_ids:
            return index
        normalized = np.asarray(matrix, dtype=np.float32)
        if normalized.ndim != 2:
            raise ValueError("Vector matrix must be 2D")
        index._doc_ids = [int(doc_id) for doc_id in doc_ids]
        index._id_to_pos = {doc_id: pos for pos, doc_id in enumerate(index._doc_ids)}
        index._matrix = normalized
        return index


class TypeIndex:
    """Inverted index from type label to entity canonicals."""

    def __init__(self) -> None:
        self._index: dict[str, set[str]] = defaultdict(set)

    def index(self, entity_canonical: str, type_label: str) -> None:
        normalized_type = EntityType.normalize(type_label)
        if not normalized_type:
            return
        self._index[normalized_type].add(_normalize_name(entity_canonical))

    def lookup(self, type_label: str) -> list[str]:
        normalized_type = EntityType.normalize(type_label)
        return sorted(self._index.get(normalized_type, set()))

    def count(self, type_label: str) -> int:
        return len(self.lookup(type_label))

    def to_dict(self) -> dict[str, Any]:
        return {type_label: sorted(entities) for type_label, entities in sorted(self._index.items())}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TypeIndex":
        index = cls()
        for type_label, entities in payload.items():
            index._index[type_label] = {_normalize_name(entity) for entity in entities}
        return index


class PropertyIndex:
    """Inverted index from property predicate to (entity, value, type) tuples."""

    def __init__(self) -> None:
        self._index: dict[str, list[tuple[str, str, PropertyType]]] = defaultdict(list)

    def index(self, entity: str, predicate: str, value: str, ptype: PropertyType) -> None:
        normalized_predicate = _normalize_predicate(predicate)
        canonical = _normalize_name(entity)
        property_type = ptype if isinstance(ptype, PropertyType) else PropertyType(ptype)
        self._index[normalized_predicate].append((canonical, value, property_type))

    def lookup(self, predicate: str) -> list[tuple[str, str, PropertyType]]:
        return list(self._index.get(_normalize_predicate(predicate), []))

    def lookup_numeric(self, predicate: str, op: str, threshold: float) -> list[tuple[str, float]]:
        operator = {
            "$eq": lambda value: value == threshold,
            "$ne": lambda value: value != threshold,
            "$lt": lambda value: value < threshold,
            "$lte": lambda value: value <= threshold,
            "$gt": lambda value: value > threshold,
            "$gte": lambda value: value >= threshold,
        }.get(op)
        if operator is None:
            raise ValueError(f"Unsupported numeric operator: {op}")

        results: list[tuple[str, float]] = []
        for entity, value, ptype in self.lookup(predicate):
            if ptype != PropertyType.NUMERIC:
                continue
            numeric = _parse_numeric(value)
            if numeric is None or not operator(numeric):
                continue
            results.append((entity, numeric))
        return results

    def entities_with_property(self, predicate: str) -> list[str]:
        return sorted({entity for entity, _, _ in self.lookup(predicate)})

    def to_dict(self) -> dict[str, Any]:
        return {
            predicate: [
                {"entity": entity, "value": value, "ptype": ptype.value}
                for entity, value, ptype in entries
            ]
            for predicate, entries in sorted(self._index.items())
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PropertyIndex":
        index = cls()
        for predicate, entries in payload.items():
            for entry in entries:
                index.index(entry["entity"], predicate, entry["value"], PropertyType(entry["ptype"]))
        return index


class TemporalIndex:
    """Allen interval algebra engine for temporal reasoning."""

    def __init__(self) -> None:
        self._events: list[TemporalEvent] = []
        self._bm25 = BM25Index()

    def ingest(self, event: TemporalEvent) -> None:
        temporal_event = event if isinstance(event, TemporalEvent) else TemporalEvent.from_dict(event)
        event_id = len(self._events)
        self._events.append(temporal_event)
        bm25_text = " ".join(
            part for part in [temporal_event.event, temporal_event.subject, temporal_event.verb, temporal_event.object] if part
        )
        self._bm25.add(event_id, bm25_text)

    def ingest_batch(self, events: list[TemporalEvent]) -> None:
        for event in events:
            self.ingest(event)

    def match_events(
        self,
        query: str,
        top_k: int = 5,
        entities: list[str] | None = None,
    ) -> list[TemporalEvent]:
        if not self._events or not query.strip():
            return []
        entity_filter = {_normalize_name(entity) for entity in entities or []}
        ranked = self._bm25.query(query, top_k=max(top_k * 4, top_k))
        results: list[TemporalEvent] = []
        for event_id, _score in ranked:
            event = self._events[event_id]
            if entity_filter:
                subject = _normalize_name(event.subject or "")
                obj = _normalize_name(event.object or "")
                if subject not in entity_filter and obj not in entity_filter:
                    continue
            results.append(event)
            if len(results) >= top_k:
                break
        return results

    def events_on_date(self, date: datetime) -> list[TemporalEvent]:
        point = _parse_datetime(date)
        if point is None:
            return []
        return [event for event in self._events if self.contains(event, point)]

    def events_between(self, start: datetime, end: datetime) -> list[TemporalEvent]:
        start_dt = _parse_datetime(start)
        end_dt = _parse_datetime(end)
        if start_dt is None or end_dt is None:
            return []
        matches = [
            event
            for event in self._events
            if event.start_dt <= end_dt and event.end_dt >= start_dt
        ]
        matches.sort(key=lambda event: (
            event.start_dt,
            event.end_dt,
            -1 if event.turn_id is None else int(event.turn_id),
            event.event,
        ))
        return matches

    def allen_before(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.end_dt < b.start_dt

    def allen_after(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.start_dt > b.end_dt

    def allen_meets(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.end_dt == b.start_dt

    def allen_met_by(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.start_dt == b.end_dt

    def allen_overlaps(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.start_dt < b.start_dt < a.end_dt < b.end_dt

    def allen_overlapped_by(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return b.start_dt < a.start_dt < b.end_dt < a.end_dt

    def allen_starts(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.start_dt == b.start_dt and a.end_dt < b.end_dt

    def allen_started_by(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.start_dt == b.start_dt and a.end_dt > b.end_dt

    def allen_during(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return b.start_dt < a.start_dt and a.end_dt < b.end_dt

    def allen_contains_interval(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.start_dt < b.start_dt and b.end_dt < a.end_dt

    def allen_finishes(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.start_dt > b.start_dt and a.end_dt == b.end_dt

    def allen_finished_by(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.start_dt < b.start_dt and a.end_dt == b.end_dt

    def allen_equals(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return a.start_dt == b.start_dt and a.end_dt == b.end_dt

    def allen_relation(self, a: TemporalEvent, b: TemporalEvent) -> str:
        checks = (
            ("before", self.allen_before),
            ("after", self.allen_after),
            ("meets", self.allen_meets),
            ("met_by", self.allen_met_by),
            ("overlaps", self.allen_overlaps),
            ("overlapped_by", self.allen_overlapped_by),
            ("starts", self.allen_starts),
            ("started_by", self.allen_started_by),
            ("during", self.allen_during),
            ("contains", self.allen_contains_interval),
            ("finishes", self.allen_finishes),
            ("finished_by", self.allen_finished_by),
            ("equals", self.allen_equals),
        )
        for name, predicate in checks:
            if predicate(a, b):
                return name
        raise ValueError(f"No Allen relation matched for {a!r} vs {b!r}")

    def before(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return self.allen_before(a, b)

    def after(self, a: TemporalEvent, b: TemporalEvent) -> bool:
        return self.allen_after(a, b)

    def distance(self, event: TemporalEvent, reference: datetime) -> timedelta:
        point = _parse_datetime(reference)
        if point is None:
            return timedelta(0)
        if point < event.start_dt:
            return event.start_dt - point
        if point > event.end_dt:
            return event.end_dt - point
        return timedelta(0)

    def duration_between(self, a: TemporalEvent, b: TemporalEvent) -> timedelta:
        if self.allen_before(a, b):
            return b.start_dt - a.end_dt
        if self.allen_after(a, b):
            return a.start_dt - b.end_dt
        return timedelta(0)

    def duration_of(self, event: TemporalEvent) -> timedelta:
        return event.end_dt - event.start_dt

    def contains(self, event: TemporalEvent, point: datetime) -> bool:
        return event.start_dt <= point <= event.end_dt

    def answer_temporal_query(
        self,
        question: str,
        reference_date: datetime | None = None,
    ) -> dict[str, Any]:
        query = (question or "").strip()
        if not query:
            return {"bucket": "unknown", "results": []}
        lowered = query.lower()
        if "how long" in lowered:
            bucket = "duration_of"
        elif "between" in lowered:
            bucket = "duration_between"
        elif "ago" in lowered:
            bucket = "distance"
        elif "first" in lowered or "before" in lowered or "after" in lowered:
            bucket = "ordering"
        elif "on " in lowered or "what happened" in lowered:
            bucket = "contains"
        else:
            bucket = "unknown"
        return {
            "bucket": bucket,
            "operation": bucket,
            "results": [event.to_dict() for event in self.match_events(query, top_k=5)],
            "matched_events": [event.to_dict() for event in self.match_events(query, top_k=5)],
            "reference_date": _serialize_datetime(reference_date),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"events": [event.to_dict() for event in self._events]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalIndex":
        index = cls()
        index.ingest_batch([TemporalEvent.from_dict(item) for item in payload.get("events", [])])
        return index


class CooccurrenceIndex:
    """Entity co-mention tracking within chunks with a bounded adjacency list."""

    MAX_NEIGHBORS_PER_ENTITY = 100

    def __init__(self, max_neighbors: int | None = None) -> None:
        self.max_neighbors = max_neighbors or self.MAX_NEIGHBORS_PER_ENTITY
        self._adjacency: dict[str, dict[str, set[int]]] = defaultdict(dict)

    def _remove_edge(self, entity_a: str, entity_b: str) -> None:
        neighbors = self._adjacency.get(entity_a)
        if neighbors is not None:
            neighbors.pop(entity_b, None)
            if not neighbors:
                self._adjacency.pop(entity_a, None)

    def _enforce_budget(self, entity: str) -> None:
        neighbors = self._adjacency.get(entity)
        if not neighbors or len(neighbors) <= self.max_neighbors:
            return
        victim, _ = min(neighbors.items(), key=lambda item: (len(item[1]), item[0]))
        self._remove_edge(entity, victim)
        self._remove_edge(victim, entity)

    def record(self, entity_a: str, entity_b: str, chunk_idx: int) -> None:
        left = _normalize_name(entity_a)
        right = _normalize_name(entity_b)
        if not left or not right or left == right:
            return
        self._adjacency[left].setdefault(right, set()).add(chunk_idx)
        self._adjacency[right].setdefault(left, set()).add(chunk_idx)
        self._enforce_budget(left)
        self._enforce_budget(right)

    def get_cooccurrences(self, entity: str) -> dict[str, int]:
        normalized = _normalize_name(entity)
        return {
            neighbor: len(chunks)
            for neighbor, chunks in sorted(self._adjacency.get(normalized, {}).items())
        }

    def get_shared_chunks(self, entity_a: str, entity_b: str) -> set[int]:
        left = _normalize_name(entity_a)
        right = _normalize_name(entity_b)
        return set(self._adjacency.get(left, {}).get(right, set()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_neighbors": self.max_neighbors,
            "adjacency": {
                entity: {neighbor: sorted(chunks) for neighbor, chunks in sorted(neighbors.items())}
                for entity, neighbors in sorted(self._adjacency.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CooccurrenceIndex":
        index = cls(max_neighbors=payload.get("max_neighbors"))
        for entity, neighbors in payload.get("adjacency", {}).items():
            index._adjacency[entity] = {
                neighbor: set(chunks) for neighbor, chunks in neighbors.items()
            }
        return index


class SynonymDict:
    """Exact alias resolution for triadic-memory query matching."""

    _PRONOUNS = frozenset(
        {
            "he",
            "she",
            "it",
            "they",
            "him",
            "her",
            "them",
            "his",
            "i",
            "my",
            "me",
            "we",
            "us",
            "our",
            "you",
            "your",
            "hers",
            "its",
            "their",
            "theirs",
        }
    )

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def add(self, alias: str, canonical: str) -> None:
        key = self._normalize(alias)
        value = self._normalize(canonical)
        if key and value and key != value:
            self._aliases[key] = value

    def resolve(self, word: str) -> str | None:
        key = self._normalize(word)
        if not key:
            return None
        return self._aliases.get(key)

    def add_from_coref_clusters(self, clusters: list[list[str]]) -> None:
        for cluster in clusters:
            canonical = None
            for mention in cluster:
                normalized = self._normalize(mention)
                if normalized and normalized not in self._PRONOUNS:
                    canonical = normalized
                    break
            if canonical is None:
                continue
            for mention in cluster:
                alias = self._normalize(mention)
                if alias and alias != canonical and alias not in self._PRONOUNS:
                    self.add(alias, canonical)

    def add_from_verb_mappings(self, mappings: list[tuple[str, str]]) -> None:
        for verb_form, predicate in mappings:
            self.add(verb_form, predicate)

    @property
    def size(self) -> int:
        return len(self._aliases)

    def to_dict(self) -> dict[str, Any]:
        return {"aliases": dict(sorted(self._aliases.items()))}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SynonymDict":
        synonym_dict = cls()
        if not payload:
            return synonym_dict
        for alias, canonical in dict(payload.get("aliases", {})).items():
            synonym_dict.add(alias, canonical)
        return synonym_dict


class TriadicMemory:
    """Sparse 3D tensor for associative recall of S-P-O triples."""

    def __init__(self, n: int = 1000, p: int = 10, seed: int = 42):
        if n <= 0:
            raise ValueError("n must be positive")
        if p <= 0 or p > n:
            raise ValueError("p must satisfy 0 < p <= n")
        self.n = int(n)
        self.p = int(p)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self._entries: set[tuple[int, int, int]] = set()
        self._xy_to_z: dict[tuple[int, int], set[int]] = defaultdict(set)
        self._yz_to_x: dict[tuple[int, int], set[int]] = defaultdict(set)
        self._codebook: dict[str, set[int]] = {}
        self._pending_I: list[np.ndarray] = []
        self._pending_J: list[np.ndarray] = []
        self._pending_K: list[np.ndarray] = []
        self._synonym_dict = SynonymDict()
        self._text_to_triples: dict[str, list[tuple[str, str, str]]] = {}
        self._concept_to_texts: dict[str, set[str]] = {}
        self._n_facts_with_triples = 0
        self._n_total_facts = 0

    @staticmethod
    def _normalize_concept(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _get_or_create_sdr(self, concept: str) -> set[int]:
        key = self._normalize_concept(concept)
        if key not in self._codebook:
            indices = set(int(index) for index in self.rng.choice(self.n, size=self.p, replace=False))
            self._codebook[key] = indices
        return self._codebook[key]

    def _apply_entry(self, i: int, j: int, k: int) -> None:
        entry = (int(i), int(j), int(k))
        if entry in self._entries:
            return
        self._entries.add(entry)
        self._xy_to_z[(entry[0], entry[1])].add(entry[2])
        self._yz_to_x[(entry[1], entry[2])].add(entry[0])

    def _restore_rng_state(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        try:
            self.rng.set_state(
                (
                    str(payload["algorithm"]),
                    np.array(payload["keys"], dtype=np.uint32),
                    int(payload["pos"]),
                    int(payload["has_gauss"]),
                    float(payload["cached_gaussian"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            return

    def store(self, x_name: str, y_name: str, z_name: str) -> None:
        x = self._get_or_create_sdr(x_name)
        y = self._get_or_create_sdr(y_name)
        z = self._get_or_create_sdr(z_name)
        ax = np.fromiter(x, dtype=np.intp)
        ay = np.fromiter(y, dtype=np.intp)
        az = np.fromiter(z, dtype=np.intp)
        ix, iy, iz = np.meshgrid(ax, ay, az, indexing="ij")
        self._pending_I.append(ix.ravel())
        self._pending_J.append(iy.ravel())
        self._pending_K.append(iz.ravel())

    def flush(self) -> None:
        if not self._pending_I:
            return
        all_i = np.concatenate(self._pending_I)
        all_j = np.concatenate(self._pending_J)
        all_k = np.concatenate(self._pending_K)
        for i, j, k in zip(all_i.tolist(), all_j.tolist(), all_k.tolist()):
            self._apply_entry(i, j, k)
        self._pending_I.clear()
        self._pending_J.clear()
        self._pending_K.clear()

    def store_fact(self, text: str, triples: list[tuple[str, str, str]]) -> None:
        fact_text = str(text or "")
        self._n_total_facts += 1
        normalized_triples: list[tuple[str, str, str]] = []
        seen_triples: set[tuple[str, str, str]] = set()
        for s, p, o in triples:
            triple = (str(s), str(p), str(o))
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
            normalized_triples.append(triple)
        if not normalized_triples:
            self._text_to_triples.setdefault(fact_text, [])
            return

        self._n_facts_with_triples += 1
        existing = self._text_to_triples.setdefault(fact_text, [])
        existing_set = set(existing)
        for s, p, o in normalized_triples:
            if (s, p, o) in existing_set:
                continue
            existing.append((s, p, o))
            self.store(s, p, o)
            for concept in (s, p, o):
                key = self._normalize_concept(concept)
                self._concept_to_texts.setdefault(key, set()).add(fact_text)

    def recall_z(self, x_name: str, y_name: str) -> set[int]:
        self.flush()
        scores = np.zeros(self.n, dtype=np.float64)
        x = self._get_or_create_sdr(x_name)
        y = self._get_or_create_sdr(y_name)
        for i in x:
            for j in y:
                for k in self._xy_to_z.get((i, j), ()):
                    scores[k] += 1.0
        return self._threshold(scores)

    def recall_x(self, y_name: str, z_name: str) -> set[int]:
        self.flush()
        scores = np.zeros(self.n, dtype=np.float64)
        y = self._get_or_create_sdr(y_name)
        z = self._get_or_create_sdr(z_name)
        for j in y:
            for k in z:
                for i in self._yz_to_x.get((j, k), ()):
                    scores[i] += 1.0
        return self._threshold(scores)

    def _threshold(self, scores: np.ndarray) -> set[int]:
        nonzero_count = int(np.count_nonzero(scores))
        if nonzero_count == 0:
            return set()
        if nonzero_count <= self.p:
            return set(int(index) for index in np.where(scores > 0)[0].tolist())
        threshold = float(np.partition(scores, -self.p)[-self.p])
        return set(int(index) for index in np.where(scores >= max(1.0, threshold))[0].tolist())

    def overlap(self, sdr_a: set[int], sdr_b: set[int]) -> float:
        if not sdr_a or not sdr_b:
            return 0.0
        return len(sdr_a & sdr_b) / self.p

    def expand(
        self,
        retrieved_texts: list[str],
        top_k: int = 5,
        overlap_threshold: float = 0.3,
        max_hops: int = 1,
        decay: float = 1.0,
    ) -> list[str]:
        if not self._text_to_triples:
            return []

        seed_set = set(retrieved_texts)
        all_known_concepts: set[str] = set()
        for text in retrieved_texts:
            for s, p, o in self._text_to_triples.get(text, []):
                all_known_concepts.update(
                    [
                        self._normalize_concept(s),
                        self._normalize_concept(p),
                        self._normalize_concept(o),
                    ]
                )

        frontier_texts = list(retrieved_texts)
        text_activation: dict[str, float] = {}
        all_retrieved = set(retrieved_texts)

        for hop in range(max(0, max_hops)):
            hop_decay = decay ** hop
            sp_pairs: list[tuple[str, str]] = []
            po_pairs: list[tuple[str, str]] = []

            for text in frontier_texts:
                for s, p, o in self._text_to_triples.get(text, []):
                    sp_pairs.append((s, p))
                    po_pairs.append((p, o))

            if not sp_pairs and not po_pairs:
                break

            new_concepts: dict[str, float] = {}

            for s, p in sp_pairs:
                recalled = self.recall_z(s, p)
                if not recalled:
                    continue
                for concept, sdr in self._codebook.items():
                    if concept in all_known_concepts:
                        continue
                    overlap = self.overlap(recalled, sdr)
                    if overlap > overlap_threshold:
                        new_concepts[concept] = new_concepts.get(concept, 0.0) + (overlap * hop_decay)

            for p, o in po_pairs:
                recalled = self.recall_x(p, o)
                if not recalled:
                    continue
                for concept, sdr in self._codebook.items():
                    if concept in all_known_concepts:
                        continue
                    overlap = self.overlap(recalled, sdr)
                    if overlap > overlap_threshold:
                        new_concepts[concept] = new_concepts.get(concept, 0.0) + (overlap * hop_decay)

            if not new_concepts:
                break

            next_frontier: list[str] = []
            for concept, activation in new_concepts.items():
                for text in self._concept_to_texts.get(concept, set()):
                    if text in seed_set:
                        continue
                    text_activation[text] = text_activation.get(text, 0.0) + activation
                    if text not in all_retrieved:
                        all_retrieved.add(text)
                        next_frontier.append(text)

            all_known_concepts.update(new_concepts.keys())
            frontier_texts = next_frontier
            if not frontier_texts:
                break

        ranked = sorted(text_activation.items(), key=lambda item: (-item[1], item[0]))
        return [text for text, _score in ranked[:top_k]]

    def query_direct(
        self,
        query_text: str,
        top_k: int = 5,
        overlap_threshold: float = 0.3,
    ) -> list[str]:
        if not self._concept_to_texts or not self._codebook:
            return []

        query_lower = " ".join(str(query_text or "").strip().lower().split())
        query_words = query_lower.split()
        query_word_set = set(query_words)
        query_concepts: set[str] = set()

        if self._synonym_dict.size > 0:
            for ngram_len in range(1, min(5, len(query_words) + 1)):
                for start in range(len(query_words) - ngram_len + 1):
                    phrase = " ".join(query_words[start : start + ngram_len])
                    resolved = self._synonym_dict.resolve(phrase)
                    if resolved:
                        normalized = self._normalize_concept(resolved)
                        if normalized in self._codebook:
                            query_concepts.add(normalized)

        for concept in self._codebook:
            surface = concept.replace("_", " ")
            if len(surface) < 3:
                continue
            if " " in surface:
                if surface in query_lower:
                    query_concepts.add(concept)
            elif surface in query_word_set:
                query_concepts.add(concept)

        if not query_concepts:
            return []

        sp_pairs: list[tuple[str, str]] = []
        po_pairs: list[tuple[str, str]] = []
        direct_texts: dict[str, float] = {}

        for concept in query_concepts:
            for text in self._concept_to_texts.get(concept, set()):
                direct_texts[text] = direct_texts.get(text, 0.0) + 1.0
            for text, triples in self._text_to_triples.items():
                for s, p, o in triples:
                    triple_concepts = {
                        self._normalize_concept(s),
                        self._normalize_concept(p),
                        self._normalize_concept(o),
                    }
                    if concept in triple_concepts:
                        sp_pairs.append((s, p))
                        po_pairs.append((p, o))

        expansion_concepts: dict[str, float] = {}
        seen_sp: set[tuple[str, str]] = set()
        seen_po: set[tuple[str, str]] = set()

        for s, p in sp_pairs:
            key = (self._normalize_concept(s), self._normalize_concept(p))
            if key in seen_sp:
                continue
            seen_sp.add(key)
            recalled = self.recall_z(s, p)
            if not recalled:
                continue
            for concept, sdr in self._codebook.items():
                overlap = self.overlap(recalled, sdr)
                if overlap > overlap_threshold:
                    expansion_concepts[concept] = max(expansion_concepts.get(concept, 0.0), overlap)

        for p, o in po_pairs:
            key = (self._normalize_concept(p), self._normalize_concept(o))
            if key in seen_po:
                continue
            seen_po.add(key)
            recalled = self.recall_x(p, o)
            if not recalled:
                continue
            for concept, sdr in self._codebook.items():
                overlap = self.overlap(recalled, sdr)
                if overlap > overlap_threshold:
                    expansion_concepts[concept] = max(expansion_concepts.get(concept, 0.0), overlap)

        tensor_texts: dict[str, float] = {}
        for concept, score in expansion_concepts.items():
            for text in self._concept_to_texts.get(concept, set()):
                tensor_texts[text] = max(tensor_texts.get(text, 0.0), score)

        merged: dict[str, float] = {}
        for text, score in direct_texts.items():
            merged[text] = score * 2.0
        for text, score in tensor_texts.items():
            merged[text] = max(merged.get(text, 0.0), score)

        ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
        return [text for text, _score in ranked[:top_k]]

    def add_coref_clusters(self, clusters: list[list[str]]) -> None:
        self._synonym_dict.add_from_coref_clusters(clusters)

    def add_verb_mappings(self, mappings: list[tuple[str, str]]) -> None:
        self._synonym_dict.add_from_verb_mappings(mappings)

    def coverage(self) -> tuple[int, int]:
        return self._n_facts_with_triples, self._n_total_facts

    @property
    def tensor_entries(self) -> int:
        self.flush()
        return len(self._entries)

    def to_dict(self) -> dict[str, Any]:
        self.flush()
        rng_state = self.rng.get_state()
        return {
            "n": self.n,
            "p": self.p,
            "seed": self.seed,
            "entries": [list(entry) for entry in sorted(self._entries)],
            "codebook": {concept: sorted(indices) for concept, indices in sorted(self._codebook.items())},
            "text_to_triples": {
                text: [list(triple) for triple in triples]
                for text, triples in sorted(self._text_to_triples.items())
            },
            "concept_to_texts": {
                concept: sorted(texts) for concept, texts in sorted(self._concept_to_texts.items())
            },
            "n_facts_with_triples": self._n_facts_with_triples,
            "n_total_facts": self._n_total_facts,
            "synonyms": self._synonym_dict.to_dict(),
            "rng_state": {
                "algorithm": str(rng_state[0]),
                "keys": rng_state[1].tolist(),
                "pos": int(rng_state[2]),
                "has_gauss": int(rng_state[3]),
                "cached_gaussian": float(rng_state[4]),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "TriadicMemory":
        if not payload:
            return cls()

        memory = cls(
            n=int(payload.get("n", 1000)),
            p=int(payload.get("p", 10)),
            seed=int(payload.get("seed", 42)),
        )
        memory._codebook = {
            cls._normalize_concept(concept): {int(index) for index in indices}
            for concept, indices in dict(payload.get("codebook", {})).items()
        }
        for entry in payload.get("entries", []):
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                continue
            memory._apply_entry(int(entry[0]), int(entry[1]), int(entry[2]))

        text_to_triples = dict(payload.get("text_to_triples", {}))
        for text, triples in text_to_triples.items():
            memory._text_to_triples[str(text)] = [
                (str(triple[0]), str(triple[1]), str(triple[2]))
                for triple in triples
                if isinstance(triple, (list, tuple)) and len(triple) == 3
            ]

        concept_to_texts = dict(payload.get("concept_to_texts", {}))
        if concept_to_texts:
            memory._concept_to_texts = {
                cls._normalize_concept(concept): {str(text) for text in texts}
                for concept, texts in concept_to_texts.items()
            }
        else:
            rebuilt: dict[str, set[str]] = {}
            for text, triples in memory._text_to_triples.items():
                for s, p, o in triples:
                    for concept in (s, p, o):
                        rebuilt.setdefault(cls._normalize_concept(concept), set()).add(text)
            memory._concept_to_texts = rebuilt

        memory._n_facts_with_triples = int(
            payload.get(
                "n_facts_with_triples",
                sum(1 for triples in memory._text_to_triples.values() if triples),
            )
        )
        memory._n_total_facts = int(
            payload.get(
                "n_total_facts",
                len(memory._text_to_triples),
            )
        )
        memory._synonym_dict = SynonymDict.from_dict(payload.get("synonyms"))
        memory._restore_rng_state(payload.get("rng_state"))
        return memory


class RawChunkStore:
    """Stores and retrieves raw conversation chunks."""

    def __init__(self) -> None:
        self._chunks: dict[int, RawChunk] = {}
        self._turn_index: dict[int, list[int]] = defaultdict(list)

    def store(self, chunk: RawChunk) -> None:
        raw_chunk = chunk if isinstance(chunk, RawChunk) else RawChunk.from_dict(chunk)
        existing = self._chunks.get(raw_chunk.chunk_idx)
        if existing is not None:
            turn_list = self._turn_index.get(existing.turn_idx, [])
            self._turn_index[existing.turn_idx] = [idx for idx in turn_list if idx != existing.chunk_idx]
        self._chunks[raw_chunk.chunk_idx] = raw_chunk
        self._turn_index[raw_chunk.turn_idx].append(raw_chunk.chunk_idx)
        self._turn_index[raw_chunk.turn_idx] = sorted(set(self._turn_index[raw_chunk.turn_idx]))

    def get(self, chunk_idx: int) -> RawChunk:
        return self._chunks[chunk_idx]

    def get_by_turn(self, turn_idx: int) -> list[RawChunk]:
        return [self._chunks[chunk_idx] for chunk_idx in self._turn_index.get(turn_idx, [])]

    def __len__(self) -> int:
        return len(self._chunks)

    def to_dict(self) -> dict[str, Any]:
        return {"chunks": [self._chunks[chunk_idx].to_dict() for chunk_idx in sorted(self._chunks)]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RawChunkStore":
        store = cls()
        for chunk in payload.get("chunks", []):
            store.store(RawChunk.from_dict(chunk))
        return store


class EntityGraph:
    """Mutable graph of entities keyed by canonical name."""

    def __init__(self) -> None:
        self._entities: dict[str, EntityNode] = {}
        self._alias_to_canonical: dict[str, str] = {}

    def add_entity(self, entity: EntityNode) -> EntityNode:
        node = entity if isinstance(entity, EntityNode) else EntityNode.from_dict(entity)
        existing = self._entities.get(node.canonical)
        if existing is None:
            self._entities[node.canonical] = node
            merged = node
        else:
            existing.aliases.update(node.aliases)
            for type_label in node.types:
                existing.register_type(type_label)
            existing.related.update(node.related)
            existing.mention_turns.update(node.mention_turns)
            existing.mention_chunks.update(node.mention_chunks)
            for predicate, facts in node.facts.items():
                existing.facts.setdefault(predicate, []).extend(facts)
            for predicate, targets in node.links.items():
                existing.links.setdefault(predicate, set()).update(targets)
            merged = existing

        self._alias_to_canonical[merged.canonical] = merged.canonical
        for alias in merged.aliases:
            self._alias_to_canonical[_normalize_name(alias)] = merged.canonical
        return merged

    def ensure_entity(self, canonical: str) -> EntityNode:
        normalized = _normalize_name(canonical)
        entity = self._entities.get(normalized)
        if entity is None:
            entity = EntityNode(canonical=normalized)
            self.add_entity(entity)
        return entity

    def get_entity(self, name_or_alias: str) -> Optional[EntityNode]:
        normalized = _normalize_name(name_or_alias)
        canonical = self._alias_to_canonical.get(normalized, normalized)
        return self._entities.get(canonical)

    def add_fact(self, fact: Fact) -> None:
        entity = self.ensure_entity(fact.subject)
        entity.add_fact(fact)

    def add_link(self, source: str, predicate: str, target: str) -> None:
        source_entity = self.ensure_entity(source)
        target_entity = self.ensure_entity(target)
        source_entity.add_link(predicate, target_entity.canonical)
        if source_entity.canonical != target_entity.canonical:
            source_entity.related.add(target_entity.canonical)
            target_entity.related.add(source_entity.canonical)

    def all_entities(self) -> list[EntityNode]:
        return [self._entities[key] for key in sorted(self._entities)]

    def __len__(self) -> int:
        return len(self._entities)

    def to_dict(self) -> dict[str, Any]:
        return {"entities": [entity.to_dict() for entity in self.all_entities()]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityGraph":
        graph = cls()
        for entity_payload in payload.get("entities", []):
            graph.add_entity(EntityNode.from_dict(entity_payload))
        return graph


class DerivedGraph:
    """Deterministic derived storage for aggregates and computed hints."""

    def __init__(self) -> None:
        self.type_counts: dict[str, int] = {}
        self.interface_memberships: dict[str, list[str]] = {}
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_counts": dict(sorted(self.type_counts.items())),
            "interface_memberships": {
                name: list(values) for name, values in sorted(self.interface_memberships.items())
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DerivedGraph":
        graph = cls()
        graph.type_counts = dict(payload.get("type_counts", {}))
        graph.interface_memberships = {
            key: list(values) for key, values in payload.get("interface_memberships", {}).items()
        }
        graph.metadata = dict(payload.get("metadata", {}))
        return graph


@dataclass
class FoundryStore:
    """Unified persistence interface for all Phase 2 indexes."""

    SCHEMA_VERSION: ClassVar[int] = 1
    FOUNDRY_VERSION: ClassVar[str] = "1.0.0"

    bm25_chunks: BM25Index = field(default_factory=BM25Index)
    bm25_entities: BM25Index = field(default_factory=BM25Index)
    vectors: VectorIndex = field(default_factory=VectorIndex)
    types: TypeIndex = field(default_factory=TypeIndex)
    properties: PropertyIndex = field(default_factory=PropertyIndex)
    temporal: TemporalIndex = field(default_factory=TemporalIndex)
    cooccurrence: CooccurrenceIndex = field(default_factory=CooccurrenceIndex)
    triadic: TriadicMemory = field(default_factory=TriadicMemory)
    chunks: RawChunkStore = field(default_factory=RawChunkStore)
    entity_graph: EntityGraph = field(default_factory=EntityGraph)
    inferred_graph: EntityGraph = field(default_factory=EntityGraph)
    derived: DerivedGraph = field(default_factory=DerivedGraph)

    def _build_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": datetime.utcnow().isoformat() + _ISO_Z_SUFFIX,
            "foundry_version": self.FOUNDRY_VERSION,
            "config_snapshot": dict(_DEFAULT_CONFIG_SNAPSHOT),
            "index_files": [
                "bm25_chunks.json",
                "bm25_entities.json",
                "vector_meta.json",
                "vectors.npy",
                "types.json",
                "properties.json",
                "temporal.json",
                "cooccurrence.json",
                "chunks.json",
                "entity_graph.json",
                "inferred_graph.json",
                "derived.json",
            ],
        }

    def save(self, path: str) -> None:
        target = Path(path)
        tmp = target.with_name(target.name + ".tmp")
        backup = target.with_name(target.name + ".bak")

        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)

        manifest = self._build_manifest()
        try:
            derived_payload = self.derived.to_dict()
            derived_metadata = dict(derived_payload.get("metadata", {}))
            derived_metadata[_TRIADIC_DERIVED_METADATA_KEY] = self.triadic.to_dict()
            derived_payload["metadata"] = derived_metadata
            self._write_json(tmp / "bm25_chunks.json", self.bm25_chunks.to_dict())
            self._write_json(tmp / "bm25_entities.json", self.bm25_entities.to_dict())
            self._write_json(tmp / "vector_meta.json", self.vectors.to_metadata_dict())
            np.save(tmp / "vectors.npy", self.vectors._matrix)
            self._write_json(tmp / "types.json", self.types.to_dict())
            self._write_json(tmp / "properties.json", self.properties.to_dict())
            self._write_json(tmp / "temporal.json", self.temporal.to_dict())
            self._write_json(tmp / "cooccurrence.json", self.cooccurrence.to_dict())
            self._write_json(tmp / "chunks.json", self.chunks.to_dict())
            self._write_json(tmp / "entity_graph.json", self.entity_graph.to_dict())
            self._write_json(tmp / "inferred_graph.json", self.inferred_graph.to_dict())
            self._write_json(tmp / "derived.json", derived_payload)
            self._write_json(tmp / "manifest.json", manifest)

            if target.exists():
                shutil.rmtree(backup, ignore_errors=True)
                os.replace(target, backup)
            os.replace(tmp, target)
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise

    @classmethod
    def load(cls, path: str) -> "FoundryStore":
        root = Path(path)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        manifest = cls._read_json(manifest_path)
        if manifest.get("schema_version") != cls.SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Unsupported schema_version={manifest.get('schema_version')}; "
                f"expected {cls.SCHEMA_VERSION}. Re-ingest or migrate the store."
            )

        required_files = ["manifest.json", *manifest.get("index_files", [])]
        missing = [name for name in required_files if not (root / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing store files: {', '.join(sorted(missing))}")

        store = cls()
        store.bm25_chunks = BM25Index.from_dict(cls._read_json(root / "bm25_chunks.json"))
        store.bm25_entities = BM25Index.from_dict(cls._read_json(root / "bm25_entities.json"))
        vector_meta = cls._read_json(root / "vector_meta.json")
        matrix = np.load(root / "vectors.npy", allow_pickle=False)
        store.vectors = VectorIndex.from_parts(vector_meta.get("doc_ids", []), matrix)
        store.types = TypeIndex.from_dict(cls._read_json(root / "types.json"))
        store.properties = PropertyIndex.from_dict(cls._read_json(root / "properties.json"))
        store.temporal = TemporalIndex.from_dict(cls._read_json(root / "temporal.json"))
        store.cooccurrence = CooccurrenceIndex.from_dict(cls._read_json(root / "cooccurrence.json"))
        derived_payload = cls._read_json(root / "derived.json")
        derived_metadata = dict(derived_payload.get("metadata", {}))
        triadic_payload = derived_metadata.pop(_TRIADIC_DERIVED_METADATA_KEY, None)
        triadic_path = root / "triadic.json"
        if triadic_payload is not None:
            store.triadic = TriadicMemory.from_dict(triadic_payload)
        elif triadic_path.exists():
            store.triadic = TriadicMemory.from_dict(cls._read_json(triadic_path))
        store.chunks = RawChunkStore.from_dict(cls._read_json(root / "chunks.json"))
        store.entity_graph = EntityGraph.from_dict(cls._read_json(root / "entity_graph.json"))
        store.inferred_graph = EntityGraph.from_dict(cls._read_json(root / "inferred_graph.json"))
        sanitized_derived = dict(derived_payload)
        sanitized_derived["metadata"] = derived_metadata
        store.derived = DerivedGraph.from_dict(sanitized_derived)
        return store

    def to_dict(self) -> dict[str, Any]:
        return {
            "bm25_chunks": self.bm25_chunks.to_dict(),
            "bm25_entities": self.bm25_entities.to_dict(),
            "vector_meta": self.vectors.to_metadata_dict(),
            "vectors": self.vectors._matrix.tolist(),
            "types": self.types.to_dict(),
            "properties": self.properties.to_dict(),
            "temporal": self.temporal.to_dict(),
            "cooccurrence": self.cooccurrence.to_dict(),
            "triadic": self.triadic.to_dict(),
            "chunks": self.chunks.to_dict(),
            "entity_graph": self.entity_graph.to_dict(),
            "inferred_graph": self.inferred_graph.to_dict(),
            "derived": self.derived.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FoundryStore":
        store = cls()
        store.bm25_chunks = BM25Index.from_dict(payload.get("bm25_chunks", {}))
        store.bm25_entities = BM25Index.from_dict(payload.get("bm25_entities", {}))
        vector_meta = dict(payload.get("vector_meta", {}))
        vectors_payload = payload.get("vectors")
        matrix = np.array(vectors_payload, dtype=np.float32) if vectors_payload is not None else np.zeros((0, 0), dtype=np.float32)
        store.vectors = VectorIndex.from_parts(vector_meta.get("doc_ids", []), matrix)
        store.types = TypeIndex.from_dict(payload.get("types", {}))
        store.properties = PropertyIndex.from_dict(payload.get("properties", {}))
        store.temporal = TemporalIndex.from_dict(payload.get("temporal", {}))
        store.cooccurrence = CooccurrenceIndex.from_dict(payload.get("cooccurrence", {}))
        store.triadic = TriadicMemory.from_dict(payload.get("triadic", {}))
        store.chunks = RawChunkStore.from_dict(payload.get("chunks", {}))
        store.entity_graph = EntityGraph.from_dict(payload.get("entity_graph", {}))
        store.inferred_graph = EntityGraph.from_dict(payload.get("inferred_graph", {}))
        store.derived = DerivedGraph.from_dict(payload.get("derived", {}))
        return store

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "BM25Index",
    "CooccurrenceIndex",
    "EntityNode",
    "Fact",
    "FoundryStore",
    "PropertyIndex",
    "SynonymDict",
    "TriadicMemory",
    "RawChunk",
    "RawChunkStore",
    "TemporalEvent",
    "TemporalIndex",
    "TypeIndex",
    "VectorIndex",
]
