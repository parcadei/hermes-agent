"""Top-level Foundry API for the Engram memory system.

This module wires together the deterministic storage, ingestion, query,
writeback, and retrieval layers into one user-facing surface.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import re
import time
from typing import Any, Optional

from foundry.ingestion import (
    ConversationTurn,
    DerivedLayerBuilder,
    EntityResolver,
    ExtractedTriple,
    FacetEntity,
    IngestionReport,
    chunk_turns,
    extract_faceted_entities,
    extract_temporal,
    extract_triples,
    fact_from_triple,
    infer_type,
    parse_conversation,
    track_cooccurrences,
    verify_grounding,
)
from foundry.ontology import GroundingLayer
from foundry.query import QueryPlan
from foundry.retrieval import AgenticRetriever, DeterministicRouter, RetrievalTrace, search_temporal as retrieval_search_temporal
from foundry.storage import EntityGraph, EntityNode, Fact, FoundryStore, RawChunk, _normalize_name
from foundry.writeback import SupersessionEngine


_AGE_RE = re.compile(
    r"\b(?P<subject>[A-Z][a-z]+)\s+is(?:\s+just)?(?:\s+about)?\s+(?P<age>\d+(?:\s*-\s*\d+)?)\s+(?P<unit>years?|months?)\s+old\b",
    re.IGNORECASE,
)
_PRONOUN_AGE_RE = re.compile(
    r"\b(?:he|she|they)\s*(?:is|'s)\s+(?:just\s+)?(?:about\s+)?(?P<age>\d+(?:\s*-\s*\d+)?)\s+(?P<unit>years?|months?)\s+old\b",
    re.IGNORECASE,
)
_ADOPTION_YEAR_RE = re.compile(r"\b(?:adopted|got|bought)\s+(?:her|him|them|it)?\s*in\s*(?P<year>\d{4})\b", re.IGNORECASE)
_NAMED_ANIMAL_RE = re.compile(
    r"\b(?:a|an|my|our|the)?\s*(?P<descriptor>[a-z]+(?:\s+[a-z]+){0,2})\s+named\s+(?P<name>[A-Z][\w-]+)\b",
    re.IGNORECASE,
)
_LOCATION_FACT_RE = re.compile(
    r"\b(?:i|we)\s+(?P<verb>live in|lives in|moved to|relocated to)\s+(?P<place>[A-Z][A-Za-z\s-]+?)\b(?:[.,]|$)",
    re.IGNORECASE,
)
_FAVORITE_RESTAURANT_RE = re.compile(
    r"\b(?:favorite restaurant(?: used to be)?|new favorite(?: restaurant)?(?: is| called)?)\s+(?P<restaurant>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b",
    re.IGNORECASE,
)
_BIKE_LIST_RE = re.compile(
    r"\b(?:i|we)\s+have\s+\w+\s+bikes?\s*-\s*a\s+(?P<first>[a-z]+)\s+bike,\s*a\s+(?P<second>[a-z]+)\s+bike,\s+and\s+a\s+(?P<third>[a-z]+)\s+bike\b",
    re.IGNORECASE,
)
_BIKE_COST_RE = re.compile(
    r"\bmy\s+(?P<name>[a-z]+)\s+bike\s+is\s+(?P<model>[A-Z][\w-]*(?:\s+[A-Z0-9][\w-]*)*),\s*cost(?: me)?\s+\$?(?P<cost>[\d,]+)\b",
    re.IGNORECASE,
)
_BREED_SIMPLE_RE = re.compile(r"\b(?P<subject>[A-Z][a-z]+)\s+is\s+a\s+(?P<breed>[a-z-]+)\b", re.IGNORECASE)
_BREED_PRONOUN_RE = re.compile(r"\b(?:he|she|they)\s+is\s+a\s+(?P<breed>[a-z-]+)\b", re.IGNORECASE)

_ANIMAL_TO_TYPE = {
    "betta fish": ("fish", "betta"),
    "bird": ("bird", ""),
    "cat": ("cat", ""),
    "dog": ("dog", ""),
    "fish": ("fish", ""),
    "golden retriever": ("dog", "golden_retriever"),
    "hamster": ("hamster", ""),
    "kitten": ("cat", ""),
    "labrador mix": ("dog", "labrador_mix"),
    "parrot": ("parrot", ""),
    "rabbit": ("rabbit", ""),
    "rescue dog": ("dog", ""),
    "retriever": ("dog", "retriever"),
}
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
    reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v1"
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


def _infer_descriptor_types(descriptor: str) -> tuple[Optional[str], Optional[str]]:
    normalized = " ".join(str(descriptor or "").strip().lower().split())
    if not normalized:
        return None, None
    if normalized in _ANIMAL_TO_TYPE:
        entity_type, breed = _ANIMAL_TO_TYPE[normalized]
        return entity_type, breed or None
    parts = normalized.split()
    if parts and parts[-1] in _ANIMAL_TO_TYPE:
        entity_type, _breed = _ANIMAL_TO_TYPE[parts[-1]]
        if parts[0] in {"adopted", "found", "got", "has", "have", "keep", "owned", "owns"}:
            breed = None
        else:
            breed = _normalize_name(normalized) if len(parts) > 1 else None
        return entity_type, breed
    return None, None


def _first_named_subject(chunk: RawChunk) -> Optional[str]:
    for triple in extract_triples(chunk, llm_url=""):
        if triple.subject and triple.subject != "user":
            return triple.subject
    match = re.search(r"\b([A-Z][a-z]+)\b", chunk.text)
    if match:
        return match.group(1)
    return None


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
        if self.config.synonym_dict:
            self.rebuild_inferred(self.config.synonym_dict)

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
        retriever = AgenticRetriever(
            self._store,
            llm_model=self.config.agent_model,
            budget=self._retrieval_budget(),
            temperature=self.config.agent_temperature,
        )
        evidence, trace = retriever.retrieve(question)
        if evidence.strip():
            self._retrieval_counts[_normalize_name(question)] += 1
        return evidence, trace

    def retrieve_deterministic(self, question: str) -> tuple[str, Any]:
        router = DeterministicRouter(
            self._store,
            budget=self._retrieval_budget(),
            llm_url=self.config.extraction_url,
            llm_model=self.config.agent_model,
        )
        pipeline = router.route(question)
        evidence, meta = router.execute(question, pipeline)
        if meta.get("fallback") == "agentic":
            fallback_evidence, trace = self.retrieve(question)
            return fallback_evidence, {"pipeline": pipeline, "fallback": "agentic", "trace": trace}
        if evidence.strip():
            self._retrieval_counts[_normalize_name(question)] += 1
        return evidence, meta

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
        tracker.mark("store_chunk", store_started)

        entity_started = time.perf_counter()
        extracted_entities = list(entities or extract_faceted_entities(chunk, self.config.extraction_url or "", self.config.extraction_model))
        extracted_entities.extend(self._supplemental_entities(chunk))

        chunk_entities: list[str] = []
        for extracted in extracted_entities:
            entity_types = _canonical_type_set(infer_type(extracted))
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
            for type_label in entity_types:
                node.register_type(type_label)
                self._store.types.index(canonical, type_label)
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
        extracted_triples = list(triples or extract_triples(chunk, self.config.extraction_url or "", self.config.extraction_model))
        extracted_triples.extend(self._supplemental_triples(chunk))

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

            if stored.predicate in {"has", "favorite_restaurant"}:
                target_name = resolver.resolve_or_create(stored.object)
                node.add_link(stored.predicate, target_name)
                target = self._store.entity_graph.ensure_entity(target_name)
                target.mention_turns.add(chunk.turn_idx)
                target.mention_chunks.add(chunk.chunk_idx)
                if stored.predicate == "has":
                    target.related.add(canonical_subject)
                    node.related.add(target_name)
                    if target_name not in chunk_entities:
                        chunk_entities.append(target_name)

        tracker.mark("triples", triple_started)

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

    def _supplemental_entities(self, chunk: RawChunk) -> list[FacetEntity]:
        rows: list[FacetEntity] = []

        for match in _NAMED_ANIMAL_RE.finditer(chunk.text):
            entity_type, _breed = _infer_descriptor_types(match.group("descriptor"))
            if entity_type is None:
                continue
            rows.append(
                FacetEntity(
                    name=match.group("name"),
                    facets={"what": entity_type},
                    turn_idx=chunk.turn_idx,
                    source_text=chunk.text,
                )
            )

        bike_list = _BIKE_LIST_RE.search(chunk.text)
        if bike_list:
            for name in (bike_list.group("first"), bike_list.group("second"), bike_list.group("third")):
                rows.append(
                    FacetEntity(
                        name=f"{name} bike",
                        facets={"what": "bike"},
                        turn_idx=chunk.turn_idx,
                        source_text=chunk.text,
                    )
                )

        return rows

    def _supplemental_triples(self, chunk: RawChunk) -> list[ExtractedTriple]:
        rows: list[ExtractedTriple] = []
        last_named_subject = _first_named_subject(chunk)

        for match in _NAMED_ANIMAL_RE.finditer(chunk.text):
            entity_type, breed = _infer_descriptor_types(match.group("descriptor"))
            if entity_type:
                rows.append(
                    ExtractedTriple(
                        subject=match.group("name"),
                        predicate="type",
                        object=entity_type,
                        source_text=chunk.text,
                        turn_idx=chunk.turn_idx,
                        chunk_idx=chunk.chunk_idx,
                    )
                )
            if breed:
                rows.append(
                    ExtractedTriple(
                        subject=match.group("name"),
                        predicate="breed",
                        object=breed.replace("_", " "),
                        source_text=chunk.text,
                        turn_idx=chunk.turn_idx,
                        chunk_idx=chunk.chunk_idx,
                    )
                )

        for match in _AGE_RE.finditer(chunk.text):
            rows.append(
                ExtractedTriple(
                    subject=match.group("subject"),
                    predicate="age",
                    object=f"{match.group('age')} {match.group('unit')}".strip(),
                    source_text=chunk.text,
                    turn_idx=chunk.turn_idx,
                    chunk_idx=chunk.chunk_idx,
                )
            )

        for match in _BREED_SIMPLE_RE.finditer(chunk.text):
            if match.group("breed").lower() in {"a", "an", "the"}:
                continue
            rows.append(
                ExtractedTriple(
                    subject=match.group("subject"),
                    predicate="breed",
                    object=match.group("breed"),
                    source_text=chunk.text,
                    turn_idx=chunk.turn_idx,
                    chunk_idx=chunk.chunk_idx,
                )
            )

        if last_named_subject is not None:
            for match in _BREED_PRONOUN_RE.finditer(chunk.text):
                rows.append(
                    ExtractedTriple(
                        subject=last_named_subject,
                        predicate="breed",
                        object=match.group("breed"),
                        source_text=chunk.text,
                        turn_idx=chunk.turn_idx,
                        chunk_idx=chunk.chunk_idx,
                    )
                )
            for match in _PRONOUN_AGE_RE.finditer(chunk.text):
                rows.append(
                    ExtractedTriple(
                        subject=last_named_subject,
                        predicate="age",
                        object=f"{match.group('age')} {match.group('unit')}".strip(),
                        source_text=chunk.text,
                        turn_idx=chunk.turn_idx,
                        chunk_idx=chunk.chunk_idx,
                    )
                )
            adoption = _ADOPTION_YEAR_RE.search(chunk.text)
            if adoption is not None:
                rows.append(
                    ExtractedTriple(
                        subject=last_named_subject,
                        predicate="adoption_year",
                        object=adoption.group("year"),
                        source_text=chunk.text,
                        turn_idx=chunk.turn_idx,
                        chunk_idx=chunk.chunk_idx,
                    )
                )

        for match in _LOCATION_FACT_RE.finditer(chunk.text):
            predicate = "lives_in"
            rows.append(
                ExtractedTriple(
                    subject="user",
                    predicate=predicate,
                    object=match.group("place").strip(),
                    source_text=chunk.text,
                    turn_idx=chunk.turn_idx,
                    chunk_idx=chunk.chunk_idx,
                )
            )

        for match in _FAVORITE_RESTAURANT_RE.finditer(chunk.text):
            rows.append(
                ExtractedTriple(
                    subject="user",
                    predicate="favorite_restaurant",
                    object=match.group("restaurant").strip(),
                    source_text=chunk.text,
                    turn_idx=chunk.turn_idx,
                    chunk_idx=chunk.chunk_idx,
                )
            )

        bike_list = _BIKE_LIST_RE.search(chunk.text)
        if bike_list:
            for name in (bike_list.group("first"), bike_list.group("second"), bike_list.group("third")):
                rows.append(
                    ExtractedTriple(
                        subject=f"{name} bike",
                        predicate="type",
                        object="bike",
                        source_text=chunk.text,
                        turn_idx=chunk.turn_idx,
                        chunk_idx=chunk.chunk_idx,
                    )
                )

        bike_cost = _BIKE_COST_RE.search(chunk.text)
        if bike_cost:
            bike_name = f"{bike_cost.group('name')} bike"
            rows.append(
                ExtractedTriple(
                    subject=bike_name,
                    predicate="type",
                    object="bike",
                    source_text=chunk.text,
                    turn_idx=chunk.turn_idx,
                    chunk_idx=chunk.chunk_idx,
                )
            )
            rows.append(
                ExtractedTriple(
                    subject=bike_name,
                    predicate="cost",
                    object=bike_cost.group("cost"),
                    source_text=chunk.text,
                    turn_idx=chunk.turn_idx,
                    chunk_idx=chunk.chunk_idx,
                )
            )

        return rows


__all__ = ["Foundry", "FoundryConfig"]
