"""Top-level Foundry API for the Engram memory system.

This module wires together the storage, ingestion, query, writeback,
and retrieval layers into one user-facing surface.  Entity and triple
extraction is delegated to the v9 LLM extraction prompt (via
``FoundryConfig.extraction_url``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
from typing import Any, Optional

from foundry.ingestion import (
    ConversationTurn,
    DerivedLayerBuilder,
    EntityResolver,
    ExtractedTriple,
    FacetEntity,
    IngestionReport,
    _extract_entities_and_triples_with_mode,
    chunk_turns,
    extract_entities_and_triples,
    extract_faceted_entities,
    extract_temporal,
    fact_from_triple,
    infer_type,
    parse_conversation,
    track_cooccurrences,
    verify_grounding,
)
from foundry.ontology import GroundingLayer, PropertyType
from foundry.query import QueryPlan
from foundry.retrieval import (
    AgenticRetriever,
    CrossEncoderReranker,
    RetrievalTrace,
    search_temporal as retrieval_search_temporal,
)
from foundry.storage import EntityGraph, EntityNode, Fact, FoundryStore, RawChunk, TemporalEvent, _normalize_name, _normalize_predicate
from foundry.writeback import SupersessionEngine

import re as _re

_MONTH_NAMES = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7,
    'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}
_MONTH_YEAR_RE = _re.compile(
    r'^(' + '|'.join(_MONTH_NAMES) + r')\.?\s+(\d{4})$', _re.IGNORECASE)
_MONTH_DAY_YEAR_RE = _re.compile(
    r'^(' + '|'.join(_MONTH_NAMES) + r')\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$', _re.IGNORECASE)
_MONTH_DAY_RE = _re.compile(
    r'^(' + '|'.join(_MONTH_NAMES) + r')\.?\s+(\d{1,2})(?:st|nd|rd|th)?$', _re.IGNORECASE)
_MONTH_ONLY_RE = _re.compile(
    r'^(' + '|'.join(_MONTH_NAMES) + r')\.?$', _re.IGNORECASE)
_DAY_OF_MONTH_RE = _re.compile(
    r'^(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(' + '|'.join(_MONTH_NAMES) + r')$', _re.IGNORECASE)
_YEAR_ONLY_RE = _re.compile(r'^(\d{4})$')
_DECADE_RE = _re.compile(r'^(\d{4})s$')
_YEAR_RANGE_RE = _re.compile(r'^(\d{4})\s*[-–]\s*(\d{4})$')
_MONTH_DAY_RANGE_YEAR_RE = _re.compile(
    r'^(' + '|'.join(_MONTH_NAMES) + r')\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*[-–]\s*(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$',
    _re.IGNORECASE)
_MONTH_DAY_RANGE_RE = _re.compile(
    r'^(' + '|'.join(_MONTH_NAMES) + r')\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*[-–]\s*(\d{1,2})(?:st|nd|rd|th)?$',
    _re.IGNORECASE)
_MID_MONTH_RE = _re.compile(
    r'^(?:mid|early|late)\s*[-–]?\s*(' + '|'.join(_MONTH_NAMES) + r')$', _re.IGNORECASE)
_CHAT_DATE_RE = _re.compile(r'Chat Time:\s*(\d{4})/(\d{2})/(\d{2})')
_WHEN_PREFIX_RE = _re.compile(
    r'^(?:on|born on|in|during|since|from)\s+', _re.IGNORECASE)


def _extract_reference_date(chunk_text: str) -> datetime | None:
    """Extract the most recent Chat Time date from chunk text."""
    matches = _CHAT_DATE_RE.findall(chunk_text)
    if matches:
        y, m, d = matches[-1]
        try:
            return datetime(int(y), int(m), int(d))
        except ValueError:
            pass
    return None


def _parse_v9_when(when_text: str, reference_date: datetime | None = None) -> tuple[datetime | None, datetime | None]:
    """Parse a v9 'when' expression into (start_dt, end_dt). Returns (None, None) on failure."""
    text = when_text.strip()
    if not text:
        return None, None
    # Strip common prefixes: "on February 10th" → "February 10th"
    text = _WHEN_PREFIX_RE.sub('', text).strip()
    if not text:
        return None, None
    # ISO
    try:
        dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
        return dt, dt
    except ValueError:
        pass
    # Relative dates (need reference_date)
    lower = text.lower()
    if reference_date:
        if lower == 'today':
            return reference_date, reference_date
        if lower == 'yesterday':
            dt = reference_date - timedelta(days=1)
            return dt, dt
    # Month Year: "March 2024"
    m = _MONTH_YEAR_RE.match(text)
    if m:
        month = _MONTH_NAMES[m.group(1).lower().rstrip('.')]
        return datetime(int(m.group(2)), month, 1), datetime(int(m.group(2)), month, 1)
    # Month Day, Year: "March 15, 2024"
    m = _MONTH_DAY_YEAR_RE.match(text)
    if m:
        month = _MONTH_NAMES[m.group(1).lower().rstrip('.')]
        try:
            dt = datetime(int(m.group(3)), month, int(m.group(2)))
            return dt, dt
        except ValueError:
            pass
    # Month Day without year: "March 3rd", "February 10th" (use reference_date's year)
    m = _MONTH_DAY_RE.match(text)
    if m:
        month = _MONTH_NAMES[m.group(1).lower().rstrip('.')]
        day = int(m.group(2))
        year = reference_date.year if reference_date else None
        if year:
            try:
                dt = datetime(year, month, day)
                return dt, dt
            except ValueError:
                pass
    # Day of Month: "15th of March", "3rd March" (use reference_date's year)
    m = _DAY_OF_MONTH_RE.match(text)
    if m:
        day = int(m.group(1))
        month = _MONTH_NAMES[m.group(2).lower().rstrip('.')]
        year = reference_date.year if reference_date else None
        if year:
            try:
                dt = datetime(year, month, day)
                return dt, dt
            except ValueError:
                pass
    # Month only: "May", "February" (use reference_date's year)
    m = _MONTH_ONLY_RE.match(text)
    if m:
        month = _MONTH_NAMES[m.group(1).lower().rstrip('.')]
        year = reference_date.year if reference_date else None
        if year:
            return datetime(year, month, 1), datetime(year, month, 1)
    # mid/early/late Month: "mid-January", "late March" (use reference_date's year)
    m = _MID_MONTH_RE.match(text)
    if m:
        month = _MONTH_NAMES[m.group(1).lower().rstrip('.')]
        year = reference_date.year if reference_date else None
        if year:
            return datetime(year, month, 1), datetime(year, month, 1)
    # Month Day-Day, Year: "April 17-18, 2023", "May 15-16, 2023"
    m = _MONTH_DAY_RANGE_YEAR_RE.match(text)
    if m:
        month = _MONTH_NAMES[m.group(1).lower().rstrip('.')]
        try:
            s = datetime(int(m.group(4)), month, int(m.group(2)))
            e = datetime(int(m.group(4)), month, int(m.group(3)))
            return s, e
        except ValueError:
            pass
    # Month Day-Day (no year): "July 10-12", "July 26-28"
    m = _MONTH_DAY_RANGE_RE.match(text)
    if m:
        month = _MONTH_NAMES[m.group(1).lower().rstrip('.')]
        year = reference_date.year if reference_date else None
        if year:
            try:
                s = datetime(year, month, int(m.group(2)))
                e = datetime(year, month, int(m.group(3)))
                return s, e
            except ValueError:
                pass
    # Plain year
    m = _YEAR_ONLY_RE.match(text)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return datetime(y, 1, 1), datetime(y, 12, 31)
    # Decade: "1970s" → 1970-1979
    m = _DECADE_RE.match(text)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return datetime(y, 1, 1), datetime(y + 9, 12, 31)
    # Year range: "1944-1948"
    m = _YEAR_RANGE_RE.match(text)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if 1900 <= y1 <= 2100 and 1900 <= y2 <= 2100:
            return datetime(y1, 1, 1), datetime(y2, 12, 31)
    # Decade range: "1970s-1980s"
    dm = _re.match(r'^(\d{4})s\s*[-–]\s*(\d{4})s$', text)
    if dm:
        y1, y2 = int(dm.group(1)), int(dm.group(2))
        if 1900 <= y1 <= 2100 and 1900 <= y2 <= 2100:
            return datetime(y1, 1, 1), datetime(y2 + 9, 12, 31)
    # Interval: "from X to Y", "X and Y" (for time spans)
    for sep in (' to ', ' through ', ' until ', ' and '):
        if sep in text.lower():
            parts = text.lower().split(sep, 1)
            s, _ = _parse_v9_when(parts[0].removeprefix('from ').strip(), reference_date)
            e, _ = _parse_v9_when(parts[1].strip(), reference_date)
            if s and e:
                return s, e
    return None, None


_SUPERSeding_PREDICATES = {
    "city",
    "favorite_restaurant",
    "job_title",
    "lives_in",
    "moved_to",
    "salary",
    "works_at",
}
_TYPE_SUPERTYPES = {
    "bike": {"item"},
    "bird": {"pet"},
    "cat": {"pet"},
    "dog": {"pet"},
    "fish": {"pet"},
    "golden_retriever": {"dog", "pet"},
    "gravel_bike": {"bike", "item"},
    "hamster": {"pet"},
    "kitten": {"cat", "pet"},
    "labrador_mix": {"dog", "pet"},
    "mountain_bike": {"bike", "item"},
    "parrot": {"bird", "pet"},
    "rabbit": {"pet"},
    "rescue_dog": {"dog", "pet"},
    "retriever": {"dog", "pet"},
    "road_bike": {"bike", "item"},
}


@dataclass
class FoundryConfig:
    """Complete configuration for a Foundry instance."""

    chunk_size: int = 512
    extraction_model: str = "Qwen/Qwen2.5-7B-Instruct"
    extraction_url: Optional[str] = None
    temporal_model: str = "THU-KEG/ADELIE-SFT-3B"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    reranker_model: Optional[str] = "mixedbread-ai/mxbai-rerank-large-v1"
    agent_model: str = "gpt-4o-mini"
    agent_max_steps: int = 8
    agent_temperature: float = 0.0
    max_entities: int = 100_000
    max_facts_per_entity: int = 1000
    entity_registry_cap: int = 50
    synonym_dict: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ConversationNode:
    turn_idx: int
    role: str
    text: str
    session_time: Optional[datetime] = None


@dataclass
class StorageManifest:
    entity_count: int = 0
    chunk_count: int = 0
    temporal_event_count: int = 0
    saved_at: Optional[datetime] = None


@dataclass
class SessionContext:
    turns: list[ConversationNode] = field(default_factory=list)


@dataclass
class PipelinePerformanceTracker:
    started_at: float = field(default_factory=time.perf_counter)
    spans_ms: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str, started_at: float) -> None:
        self.spans_ms[name] = self.spans_ms.get(name, 0.0) + ((time.perf_counter() - started_at) * 1000.0)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0


def _canonical_type_set(type_hint: Optional[str]) -> set[str]:
    if not type_hint:
        return set()
    normalized = _normalize_name(type_hint)
    related = {normalized}
    related.update(_TYPE_SUPERTYPES.get(normalized, set()))
    if normalized in {"cat", "dog", "bird", "fish", "hamster", "parrot", "rabbit"}:
        related.add("pet")
    return {value for value in related if value}


def _entity_doc_text(entity: EntityNode) -> str:
    parts = [entity.canonical, *sorted(entity.aliases), *sorted(entity.types)]
    for facts in entity.facts.values():
        parts.extend(fact.object for fact in facts)
    return " ".join(str(part) for part in parts if str(part).strip())


class Foundry:
    """The top-level deterministic API for Engram Foundry."""

    def __init__(self, config: FoundryConfig) -> None:
        self.config = config
        self._store = FoundryStore()
        self._session = SessionContext()
        self._manifest = StorageManifest()
        self._retrieval_counts: dict[str, int] = defaultdict(int)
        self._grounding_failures: list[str] = []
        self._supersession = SupersessionEngine(max_facts_per_key=config.max_facts_per_entity)
        self._reranker: Optional[CrossEncoderReranker] = None
        reranker_model = str(self.config.reranker_model or "").strip()
        if reranker_model:
            self._reranker = CrossEncoderReranker(reranker_model)
        # Lazy-loaded embedding model for cosine search
        self._embedder: Any = None
        if self.config.synonym_dict:
            self.rebuild_inferred(self.config.synonym_dict)

    def _ensure_embedder(self) -> Any:
        """Lazily load the SentenceTransformer embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    def _embed(self, text: str) -> "np.ndarray":
        """Embed a single text using the configured embedding model."""
        import numpy as np
        embedder = self._ensure_embedder()
        return np.asarray(embedder.encode(text), dtype=np.float32)

    @classmethod
    def load(cls, path: str, config: FoundryConfig) -> "Foundry":
        instance = cls(config)
        instance._store = FoundryStore.load(path)
        instance._grounding_failures = []
        instance._prime_supersession()
        instance._manifest = StorageManifest(
            entity_count=len(instance._store.entity_graph),
            chunk_count=len(instance._store.chunks),
            temporal_event_count=len(instance._store.temporal._events),
        )
        return instance

    def ingest(self, raw_text: str) -> IngestionReport:
        if raw_text is None:
            raise TypeError("raw_text cannot be None")
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string")
        if not raw_text.strip():
            return IngestionReport(elapsed_ms=0.0)
        turns = parse_conversation(raw_text)
        return self.ingest_turns(turns)

    def ingest_turns(self, turns: list[ConversationTurn]) -> IngestionReport:
        tracker = PipelinePerformanceTracker()
        if turns is None:
            raise TypeError("turns cannot be None")
        if not turns:
            return IngestionReport(elapsed_ms=0.0)

        self._session.turns.extend(
            ConversationNode(turn_idx=turn.turn_idx, role=turn.role, text=turn.text, session_time=turn.session_time)
            for turn in turns
        )

        chunks_started = time.perf_counter()
        chunks = chunk_turns(turns, chunk_size=self.config.chunk_size)
        tracker.mark("chunking", chunks_started)

        chunk_offset = len(self._store.chunks)
        adjusted_chunks: list[RawChunk] = []
        for offset, chunk in enumerate(chunks):
            adjusted_chunks.append(
                RawChunk(
                    chunk_idx=chunk_offset + offset,
                    text=chunk.text,
                    turn_idx=chunk.turn_idx,
                    role=chunk.role,
                    session_time=chunk.session_time,
                )
            )

        report = IngestionReport()
        resolver = EntityResolver(self._store)
        existing_entities = {entity.canonical for entity in self._store.entity_graph.all_entities()}

        for chunk in adjusted_chunks:
            chunk_report = self._ingest_chunk_impl(
                chunk,
                triples=None,
                entities=None,
                resolver=resolver,
                existing_entities=existing_entities,
            )
            report.chunks_processed += chunk_report.chunks_processed
            report.entities_created += chunk_report.entities_created
            report.entities_resolved += chunk_report.entities_resolved
            report.facts_added += chunk_report.facts_added
            report.facts_superseded += chunk_report.facts_superseded
            report.facts_rejected += chunk_report.facts_rejected
            report.temporal_events += chunk_report.temporal_events
            report.cooccurrences += chunk_report.cooccurrences
            report.grounding_failures.extend(chunk_report.grounding_failures)

        self.rebuild_derived()
        self.rebuild_inferred(self.config.synonym_dict)
        grounded, total, failures = self.grounding_report()
        report.grounding_rate = (grounded / total) if total else 0.0
        report.grounding_failures = failures
        report.elapsed_ms = max(tracker.elapsed_ms, 0.001)
        self._grounding_failures = failures
        self._manifest = StorageManifest(
            entity_count=len(self._store.entity_graph),
            chunk_count=len(self._store.chunks),
            temporal_event_count=len(self._store.temporal._events),
        )
        return report

    def ingest_chunk(
        self,
        chunk: RawChunk,
        triples: list[ExtractedTriple] | None = None,
        entities: list[FacetEntity] | None = None,
    ) -> IngestionReport:
        if chunk is None:
            raise TypeError("chunk cannot be None")
        report = self._ingest_chunk_impl(
            chunk,
            triples=triples,
            entities=entities,
            resolver=EntityResolver(self._store),
            existing_entities={entity.canonical for entity in self._store.entity_graph.all_entities()},
        )
        self.rebuild_derived()
        self.rebuild_inferred(self.config.synonym_dict)
        grounded, total, failures = self.grounding_report()
        report.grounding_rate = (grounded / total) if total else 0.0
        report.grounding_failures = failures
        return report

    def build_type_hierarchy(self, threshold: float = 0.7) -> int:
        """Build type subsumption hierarchy from extraction schema.

        Uses the structured extraction data (entity types from ``what``
        facet) to build parent→child links.  An entity whose canonical
        name is also a type label creates a chain:
        ``parent_type → canonical → child_entities``.

        For example, entity "guitar" with types {guitar, instrument}
        registers ``instrument → guitar`` because "guitar" is itself a
        type label (other entities have type=guitar).

        Must be called AFTER all chunks are ingested.

        Returns the number of subsumption links added.
        """
        type_labels = set(self._store.types._index.keys())
        added = 0
        seen_pairs: set[frozenset[str]] = set()

        for entity in self._store.entity_graph.all_entities():
            canonical = entity.canonical
            if canonical not in type_labels:
                continue
            for parent_type in entity.types:
                if parent_type == canonical:
                    continue
                if parent_type not in type_labels:
                    continue
                pair = frozenset([parent_type, canonical])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                self._store.types.add_subsumption(parent_type, canonical, 1.0)
                added += 1

        return added

    def rebuild_derived(self) -> None:
        builder = DerivedLayerBuilder(self._store)
        builder.build()
        self._store.derived.metadata["gists"] = builder.extract_gists()

    def rebuild_inferred(self, synonym_dict: dict[str, list[str]] | None = None) -> None:
        graph = EntityGraph()
        active_synonyms = synonym_dict if synonym_dict is not None else self.config.synonym_dict

        for stated in self._store.entity_graph.all_entities():
            inferred = graph.ensure_entity(stated.canonical)
            inferred.aliases.update(stated.aliases)
            inferred.types.update(stated.types)
            for type_label in stated.types:
                type_node = graph.ensure_entity(type_label)
                inferred.add_link("type", type_label)
                type_node.add_link("instance", stated.canonical)
                type_node.related.add(stated.canonical)
                for synonym in active_synonyms.get(type_label, []):
                    synonym_key = _normalize_name(synonym)
                    synonym_node = graph.ensure_entity(synonym_key)
                    synonym_node.add_link("synonym_of", type_label)
                    synonym_node.related.add(type_label)
                    type_node.add_link("synonym", synonym_key)
                    inferred.add_link("synonym", synonym_key)
                    inferred.related.add(synonym_key)
        self._store.inferred_graph = graph

    def entities(self) -> QueryPlan:
        return QueryPlan(self._store)

    def entity(self, name: str) -> Optional[EntityNode]:
        if name is None:
            return None
        if not str(name).strip():
            return None
        return self._store.entity_graph.get_entity(str(name))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if query is None:
            raise TypeError("query cannot be None")
        return self._store.bm25_chunks.query(str(query), top_k=top_k)

    def search_entities(self, query: str, top_k: int = 10) -> list[tuple[str, set[int], float]]:
        rows = []
        from foundry.retrieval import search_entities as retrieval_search_entities

        for hit in retrieval_search_entities(self._store, str(query), top_k=top_k):
            rows.append((hit["entity"], set(hit.get("turns", [])), float(hit.get("score", 0.0))))
        return rows

    def resolve_conflicts(self, results: list[dict]) -> list[dict]:
        """Apply conflict resolution to retrieved results."""

        from foundry.writeback import ConflictDetector

        detector = ConflictDetector(self._store)
        return detector.filter_conflicts(results)

    def search_temporal(self, query: str, reference_date: datetime | None = None) -> dict[str, Any]:
        del reference_date
        return retrieval_search_temporal(self._store, str(query or ""))

    def facts(self, entity: str, predicate: str = "", current_only: bool = True) -> list[Fact]:
        node = self.entity(entity)
        if node is None:
            return []
        predicates = [_normalize_name(predicate)] if predicate else sorted(node.facts)
        rows: list[Fact] = []
        for pred in predicates:
            for fact in node.facts.get(pred, []):
                if current_only and fact.superseded:
                    continue
                rows.append(fact)
        rows.sort(key=lambda fact: (fact.turn_idx, fact.chunk_idx, fact.id))
        return rows

    def timeline(self, entity: str) -> Optional[dict[str, list[Fact]]]:
        node = self.entity(entity)
        if node is None:
            return None
        timeline = {
            predicate: sorted(list(facts), key=lambda fact: (fact.turn_idx, fact.chunk_idx, fact.id))
            for predicate, facts in sorted(node.facts.items())
        }
        return timeline

    def deltas(self, entity: str) -> Optional[dict[str, list[Fact]]]:
        timeline = self.timeline(entity)
        if timeline is None:
            return None
        deltas: dict[str, list[Fact]] = {}
        for predicate, facts in timeline.items():
            previous: Optional[str] = None
            changed: list[Fact] = []
            for fact in facts:
                if previous != fact.object:
                    changed.append(fact)
                    previous = fact.object
            if changed:
                deltas[predicate] = changed
        return deltas

    def retrieve(self, question: str) -> tuple[str, RetrievalTrace]:
        embed_fn = self._embed if self._embedder is not None else None
        retriever = AgenticRetriever(
            self._store,
            llm_base_url=self.config.extraction_url,
            llm_model=self.config.agent_model,
            budget=self._retrieval_budget(),
            temperature=self.config.agent_temperature,
            reranker=self._reranker.rerank if self._reranker is not None else None,
            embed_fn=embed_fn,
        )
        evidence, trace = retriever.retrieve(question)
        if evidence.strip():
            self._retrieval_counts[_normalize_name(question)] += 1
        return evidence, trace

    def decompose(self, question: str) -> list[str]:
        """Decompose a multi-hop question into chained sub-queries."""

        if not self.config.extraction_url:
            return [question]
        from foundry.query import decompose_query

        return decompose_query(
            question,
            llm_url=self.config.extraction_url,
            llm_model=self.config.agent_model,
        )

    def save(self, path: str) -> None:
        self._store.save(path)
        self._manifest.saved_at = datetime.utcnow()

    def export_graph(self) -> dict[str, Any]:
        return {
            "entity_graph": self._store.entity_graph.to_dict(),
            "inferred_graph": self._store.inferred_graph.to_dict(),
            "derived": self._store.derived.to_dict(),
        }

    @classmethod
    def import_graph(cls, data: dict) -> EntityGraph:
        if "entity_graph" in data:
            return EntityGraph.from_dict(data["entity_graph"])
        if "entities" in data:
            return EntityGraph.from_dict({"entities": data["entities"]})
        return EntityGraph()

    def stats(self) -> dict[str, Any]:
        entities = self._store.entity_graph.all_entities()
        facts = [fact for entity in entities for values in entity.facts.values() for fact in values]
        current_facts = [fact for fact in facts if not fact.superseded]
        grounded, total, _failures = self.grounding_report()
        type_distribution = dict(sorted(self._store.derived.type_counts.items()))
        return {
            "entity_count": len(entities),
            "fact_count": len(facts),
            "current_fact_count": len(current_facts),
            "alias_count": sum(len(entity.aliases) for entity in entities),
            "chunk_count": len(self._store.chunks),
            "temporal_event_count": len(self._store.temporal._events),
            "type_distribution": type_distribution,
            "avg_facts_per_entity": (len(facts) / len(entities)) if entities else 0.0,
            "avg_aliases_per_entity": (sum(len(entity.aliases) for entity in entities) / len(entities)) if entities else 0.0,
            "grounding_rate": (grounded / total) if total else 0.0,
            "layer_sizes": {
                "stated": len(facts),
                "inferred": sum(len(values) for entity in self._store.inferred_graph.all_entities() for values in entity.facts.values()),
                "derived": len(self._store.derived.type_counts) + len(self._store.derived.interface_memberships),
            },
        }

    def grounding_report(self) -> tuple[int, int, list[str]]:
        grounded = 0
        total = 0
        failures: list[str] = []
        for entity in self._store.entity_graph.all_entities():
            for predicate, facts in entity.facts.items():
                for fact in facts:
                    if fact.layer != GroundingLayer.STATED:
                        continue
                    total += 1
                    if verify_grounding(fact):
                        grounded += 1
                    else:
                        failures.append(f"{entity.canonical}.{predicate} -> {fact.object}")
        return grounded, total, failures

    def _retrieval_budget(self):
        from foundry.retrieval import RetrievalBudget

        return RetrievalBudget(max_steps=self.config.agent_max_steps)

    def _prime_supersession(self) -> None:
        self._supersession = SupersessionEngine(max_facts_per_key=self.config.max_facts_per_entity)
        for entity in self._store.entity_graph.all_entities():
            for facts in entity.facts.values():
                for fact in facts:
                    self._supersession.add_fact(fact)

    def _ingest_chunk_impl(
        self,
        chunk: RawChunk,
        *,
        triples: list[ExtractedTriple] | None,
        entities: list[FacetEntity] | None,
        resolver: EntityResolver,
        existing_entities: set[str],
    ) -> IngestionReport:
        tracker = PipelinePerformanceTracker()
        report = IngestionReport(chunks_processed=1)

        store_started = time.perf_counter()
        self._store.chunks.store(chunk)
        self._store.bm25_chunks.add(chunk.chunk_idx, chunk.text)
        # Embed chunk for cosine search (lazy — only if model configured)
        if self.config.embedding_model:
            try:
                emb = self._embed(chunk.text)
                self._store.vectors.add(chunk.chunk_idx, emb)
            except Exception:
                pass  # embedding is optional; BM25 still works
        tracker.mark("store_chunk", store_started)

        auto_entities: list[FacetEntity] = []
        auto_triples: list[ExtractedTriple] = []
        if entities is None or triples is None:
            auto_entities, auto_triples, _ = _extract_entities_and_triples_with_mode(
                chunk,
                self.config.extraction_url or "",
                self.config.extraction_model,
            )

        entity_started = time.perf_counter()
        extracted_entities = list(entities if entities is not None else auto_entities)

        chunk_entities: list[str] = []
        for extracted in extracted_entities:
            entity_types = _canonical_type_set(infer_type(extracted))
            # Unified pool: index who facet as type too (spec §2.1)
            who_val = extracted.facets.get("who", "").strip()
            if who_val and who_val != "?":
                from foundry.ontology import EntityType
                who_type = EntityType.normalize(who_val)
                if who_type:
                    entity_types.add(who_type)
            canonical = resolver.resolve_or_create(extracted.name, hint_type=next(iter(entity_types), None))
            if canonical in existing_entities:
                report.entities_resolved += 1
            else:
                report.entities_created += 1
                existing_entities.add(canonical)
            node = self._store.entity_graph.ensure_entity(canonical)
            node.aliases.add(resolver.normalize(extracted.name))
            node.mention_turns.add(chunk.turn_idx)
            node.mention_chunks.add(chunk.chunk_idx)
            bridge_facets = {v for k, v in extracted.facets.items() if k in ("who", "what")}
            self._store.mention_bridges.add(extracted.name, bridge_facets)
            for type_label in entity_types:
                node.register_type(type_label)
                self._store.types.index(canonical, type_label)
            # Store all facets as facts on the entity
            for facet_key, facet_val in extracted.facets.items():
                if not facet_val or facet_val == "?":
                    continue
                node.add_fact(Fact(
                    id=f"{canonical}:{facet_key}:{chunk.chunk_idx}",
                    subject=canonical,
                    predicate=_normalize_predicate(facet_key),
                    object=str(facet_val),
                    object_type=PropertyType.STRING,
                    turn_idx=chunk.turn_idx,
                    chunk_idx=chunk.chunk_idx,
                    source_text=extracted.source_text[:200] if extracted.source_text else "",
                ))
            if canonical not in chunk_entities:
                chunk_entities.append(canonical)
        tracker.mark("entities", entity_started)

        temporal_started = time.perf_counter()
        events = extract_temporal(chunk, session_time=chunk.session_time)
        for event in events:
            self._store.temporal.ingest(event)
        report.temporal_events += len(events)
        tracker.mark("temporal", temporal_started)

        triple_started = time.perf_counter()
        extracted_triples = list(triples if triples is not None else auto_triples)

        facts_added = 0
        facts_superseded = 0

        for triple in extracted_triples:
            hint_type = None
            if triple.predicate == "type":
                hint_type = triple.object
            canonical_subject = resolver.resolve_or_create(triple.subject, hint_type=hint_type)
            node = self._store.entity_graph.ensure_entity(canonical_subject)
            node.mention_turns.add(chunk.turn_idx)
            node.mention_chunks.add(chunk.chunk_idx)
            if canonical_subject not in chunk_entities:
                chunk_entities.append(canonical_subject)

            fact = fact_from_triple(
                triple,
                canonical_subject=canonical_subject,
                layer=GroundingLayer.STATED,
                session_time=chunk.session_time,
            )

            if triple.predicate == "type":
                for type_label in _canonical_type_set(triple.object):
                    node.register_type(type_label)
                    self._store.types.index(canonical_subject, type_label)
                continue

            grounding_passed = verify_grounding(fact)
            if not grounding_passed:
                report.facts_rejected += 1
                report.grounding_failures.append(f"{triple.subject}.{triple.predicate} -> {triple.object}")
                continue

            stored = self._supersession.add_fact(fact)
            bucket = node.facts.setdefault(stored.predicate, [])
            if all(existing.id != stored.id for existing in bucket):
                bucket.append(stored)
                bucket.sort(key=lambda item: (item.turn_idx, item.chunk_idx, item.id))
                self._store.properties.index(canonical_subject, stored.predicate, stored.object, stored.object_type)
                facts_added += 1

            if stored.predicate in _SUPERSeding_PREDICATES:
                previous = self._supersession.supersede(canonical_subject, stored.predicate, stored)
                if previous is not None:
                    facts_superseded += 1

            if stored.predicate in {"has", "bought", "got", "ordered", "owns", "favorite_restaurant"}:
                target_name = resolver.resolve_or_create(stored.object)
                node.add_link(stored.predicate, target_name)
                target = self._store.entity_graph.ensure_entity(target_name)
                target.mention_turns.add(chunk.turn_idx)
                target.mention_chunks.add(chunk.chunk_idx)
                if stored.predicate in {"has", "bought", "got", "ordered", "owns"}:
                    target.related.add(canonical_subject)
                    node.related.add(target_name)
                    if target_name not in chunk_entities:
                        chunk_entities.append(target_name)
                # Propagate v9 quantity to target entity as a "quantity" fact
                # so aggregate engine can sum counts (e.g. "how many fish?")
                qty_str = getattr(triple, 'quantity', '0')
                if qty_str and qty_str != '0':
                    qty_fact = Fact(
                        id=f"{target_name}:quantity:{chunk.chunk_idx}",
                        subject=target_name,
                        predicate="quantity",
                        object=qty_str,
                        object_type=PropertyType.STRING,
                        turn_idx=chunk.turn_idx,
                        chunk_idx=chunk.chunk_idx,
                        source_text=triple.source_text[:200] if triple.source_text else "",
                    )
                    qty_bucket = target.facts.setdefault("quantity", [])
                    if all(existing.id != qty_fact.id for existing in qty_bucket):
                        qty_bucket.append(qty_fact)
                        qty_bucket.sort(key=lambda item: (item.turn_idx, item.chunk_idx, item.id))

        # Create TemporalEvents from triples with v9 'when' data
        ref_date = _extract_reference_date(chunk.text)
        for triple in extracted_triples:
            if not triple.when:
                continue
            start_dt, end_dt = _parse_v9_when(triple.when, ref_date)
            if start_dt and end_dt:
                try:
                    te = TemporalEvent(
                        event=f"{triple.subject} {triple.predicate} {triple.object}",
                        start_dt=start_dt,
                        end_dt=end_dt,
                        source=chunk.text,
                        turn_id=chunk.turn_idx,
                        subject=triple.subject,
                        verb=triple.predicate,
                        object=triple.object,
                    )
                    self._store.temporal.ingest(te)
                    report.temporal_events += 1
                except (ValueError, TypeError):
                    pass

        tracker.mark("triples", triple_started)

        self._store.triadic.store_fact(
            chunk.text,
            [(triple.subject, triple.predicate, triple.object) for triple in extracted_triples],
        )
        self._store.triadic.flush()

        for canonical in chunk_entities:
            entity = self._store.entity_graph.ensure_entity(canonical)
            entity.mention_turns.add(chunk.turn_idx)
            entity.mention_chunks.add(chunk.chunk_idx)
        chunk.entities = list(dict.fromkeys(chunk_entities))

        cooc_started = time.perf_counter()
        track_cooccurrences(chunk.entities, chunk.chunk_idx, self._store.cooccurrence)
        report.cooccurrences += len(chunk.entities)
        tracker.mark("cooccurrence", cooc_started)

        self._rebuild_entity_search_index()
        report.facts_added = facts_added
        report.facts_superseded = facts_superseded
        report.elapsed_ms = max(tracker.elapsed_ms, 0.001)
        return report

    def _rebuild_entity_search_index(self) -> None:
        from foundry.storage import BM25Index, PropertyIndex, TypeIndex

        self._store.bm25_entities = BM25Index()
        self._store.types = TypeIndex()
        self._store.properties = PropertyIndex()
        for doc_id, entity in enumerate(self._store.entity_graph.all_entities()):
            for type_label in entity.types:
                self._store.types.index(entity.canonical, type_label)
            for predicate, facts in entity.facts.items():
                for fact in facts:
                    self._store.properties.index(entity.canonical, predicate, fact.object, fact.object_type)
            self._store.bm25_entities.add(doc_id, _entity_doc_text(entity))

__all__ = ["Foundry", "FoundryConfig"]
