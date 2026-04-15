"""Writeback and mutation primitives for the Hermes Foundry memory system."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from foundry.storage import (
    BM25Index,
    CooccurrenceIndex,
    EntityNode,
    Fact,
    FoundryStore,
    PropertyIndex,
    TypeIndex,
)


def _normalize_name(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    return "_".join(value.strip().lower().split())


def _normalize_predicate(value: str) -> str:
    return _normalize_name(value)


@dataclass
class TemporalValidity:
    """Temporal validity window for a fact."""

    valid_from: Optional[int | datetime] = None
    valid_until: Optional[int | datetime] = None
    axis: str = "turn"

    def __post_init__(self) -> None:
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and type(self.valid_from) is not type(self.valid_until)
        ):
            raise TypeError(
                f"TemporalValidity requires a single comparison axis, got "
                f"{type(self.valid_from).__name__} and {type(self.valid_until).__name__}"
            )
        if self.valid_from is not None:
            self.axis = "event" if isinstance(self.valid_from, datetime) else "turn"
        elif self.valid_until is not None:
            self.axis = "event" if isinstance(self.valid_until, datetime) else "turn"

    def is_current(self, as_of: Optional[int | datetime] = None) -> bool:
        if as_of is None:
            return self.valid_until is None
        if self.valid_from is not None and as_of < self.valid_from:
            return False
        if self.valid_until is None:
            return True
        return as_of <= self.valid_until


class DeletionPolicy(str, Enum):
    """How facts are removed from memory."""

    SOFT_DELETE = "soft_delete"
    HARD_DELETE = "hard_delete"


@dataclass
class Tombstone:
    """Record of a deleted fact."""

    fact_id: str
    deleted_at: datetime
    reason: str

    def __post_init__(self) -> None:
        allowed = {"superseded", "user_request", "decay"}
        if self.reason not in allowed:
            raise ValueError(f"Unsupported tombstone reason: {self.reason}")


@dataclass
class MemoryAction:
    """Named atomic mutation operation on memory."""

    name: str
    entity_edits: list[dict[str, Any]] = field(default_factory=list)
    link_edits: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "ingestion"
    cascade: bool = False
    validate: bool = True


ACTION_TYPES = {
    "ingest_chunk": "Add a new chunk with extracted entities and facts",
    "supersede_fact": "Replace an existing fact with a newer value",
    "merge_entities": "Merge two entities discovered to be the same",
    "add_alias": "Register a new alias for an existing entity",
    "gc_decay": "Remove facts below salience threshold",
    "rebuild_derived": "Recompute derived layer from stated",
    "retract": "Soft-delete facts for a user-requested retraction",
}


@dataclass
class EditVersion:
    """Version tracking for optimistic concurrency control."""

    version: int
    entity_canonical: str
    predicate: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class VersionTracker:
    """Global version counter for edit ordering."""

    def __init__(self) -> None:
        self._version = 0

    def next_version(self) -> int:
        self._version += 1
        return self._version

    @property
    def current(self) -> int:
        return self._version


class SupersessionEngine:
    """Manage fact supersession while preserving history."""

    SUPERSESSION_GROUPS: dict[str, set[str]] = {
        "allergy": {
            "allergic_to",
            "can_eat",
            "tolerates",
            "outgrew_allergy",
            "allergy_status",
        },
        "location": {"lives_in", "moved_to", "relocated_to", "based_in", "residence"},
        "employment": {"works_at", "employed_by", "job", "quit", "fired_from", "retired_from"},
    }

    def __init__(
        self,
        *,
        max_facts_per_key: Optional[int] = None,
        supersession_groups: Optional[dict[str, set[str]]] = None,
    ) -> None:
        self.max_facts_per_key = max_facts_per_key
        self._facts: dict[tuple[str, str], list[Fact]] = defaultdict(list)
        self._fact_ids: dict[str, Fact] = {}
        self._validity: dict[str, TemporalValidity] = {}
        self._versions = VersionTracker()
        self._supersession_groups = supersession_groups or self.SUPERSESSION_GROUPS

    def add_fact(self, fact: Fact) -> Fact:
        key = self._bucket_key(fact.subject, fact.predicate)
        existing = self._find_duplicate(key, fact)
        if existing is not None:
            return existing
        self._facts[key].append(fact)
        self._facts[key].sort(key=lambda item: (item.turn_idx, item.id))
        self._fact_ids[fact.id] = fact
        self._validity.setdefault(
            fact.id,
            TemporalValidity(valid_from=self._default_valid_from(fact), valid_until=None),
        )
        return fact

    def get_all_facts(self, entity: str, predicate: str) -> list[Fact]:
        return list(self._facts.get(self._bucket_key(entity, predicate), []))

    def supersede(self, entity: str, predicate: str, new_fact: Fact) -> Optional[str]:
        subject = _normalize_name(entity)
        normalized_predicate = _normalize_predicate(predicate)
        stored = self._fact_ids.get(new_fact.id)
        if stored is None:
            stored = self.add_fact(new_fact)
        stored.subject = subject
        stored.predicate = normalized_predicate
        stored.superseded = False
        stored.superseded_by = None

        candidates = self._current_candidates(subject, normalized_predicate, exclude_fact_id=stored.id)
        if not candidates:
            self._maybe_evict(subject, normalized_predicate)
            return None

        candidates.sort(key=lambda fact: (fact.turn_idx, fact.id))
        for fact in candidates:
            fact.superseded = True
            fact.superseded_by = stored.id
            self._validity[fact.id] = TemporalValidity(
                valid_from=self._validity.get(fact.id, TemporalValidity()).valid_from,
                valid_until=self._supersession_point(fact, stored),
            )

        self._validity[stored.id] = TemporalValidity(
            valid_from=self._default_valid_from(stored),
            valid_until=None,
        )
        self._versions.next_version()
        self._maybe_evict(subject, normalized_predicate)
        return candidates[-1].id

    def resolve_conflicts(self, facts: list[Fact], strategy: str = "latest_wins") -> list[Fact]:
        if not facts:
            return []
        if strategy not in {"latest_wins", "user_wins", "per_hop"}:
            raise ValueError(f"Unsupported supersession strategy: {strategy}")

        winner = facts[0]
        for fact in sorted(facts[1:], key=lambda item: (item.turn_idx, item.id)):
            winner = self._pick_winner(winner, fact, strategy)
        return [winner]

    def get_validity(self, fact_id: str) -> Optional[TemporalValidity]:
        return self._validity.get(fact_id)

    def _bucket_key(self, entity: str, predicate: str) -> tuple[str, str]:
        return (_normalize_name(entity), _normalize_predicate(predicate))

    def _find_duplicate(self, key: tuple[str, str], fact: Fact) -> Optional[Fact]:
        identity = (
            _normalize_name(fact.subject),
            _normalize_predicate(fact.predicate),
            fact.object,
            fact.turn_idx,
            fact.chunk_idx,
        )
        for existing in self._facts.get(key, []):
            existing_identity = (
                existing.subject,
                existing.predicate,
                existing.object,
                existing.turn_idx,
                existing.chunk_idx,
            )
            if existing_identity == identity:
                return existing
        return None

    def _default_valid_from(self, fact: Fact) -> int | datetime:
        return fact.timestamp if fact.timestamp is not None else fact.turn_idx

    def _supersession_point(self, old_fact: Fact, new_fact: Fact) -> int | datetime:
        if old_fact.timestamp is not None and new_fact.timestamp is not None:
            return new_fact.timestamp
        return new_fact.turn_idx

    def _current_candidates(
        self,
        entity: str,
        predicate: str,
        *,
        exclude_fact_id: str,
    ) -> list[Fact]:
        keys = {self._bucket_key(entity, predicate)}
        for group in self._supersession_groups.values():
            if predicate in group:
                keys.update((entity, _normalize_predicate(other)) for other in group)

        candidates: list[Fact] = []
        for key in keys:
            for fact in self._facts.get(key, []):
                if fact.id == exclude_fact_id or fact.superseded:
                    continue
                candidates.append(fact)
        return candidates

    def _maybe_evict(self, entity: str, predicate: str) -> None:
        if self.max_facts_per_key is None:
            return
        bucket = self._facts.get(self._bucket_key(entity, predicate), [])
        while len(bucket) > self.max_facts_per_key:
            superseded = [fact for fact in bucket if fact.superseded]
            if not superseded:
                break
            oldest = min(superseded, key=lambda fact: (fact.turn_idx, fact.id))
            bucket.remove(oldest)
            self._fact_ids.pop(oldest.id, None)
            self._validity.pop(oldest.id, None)

    def _pick_winner(self, current: Fact, candidate: Fact, strategy: str) -> Fact:
        if strategy == "user_wins":
            current_is_user = self._is_user_fact(current)
            candidate_is_user = self._is_user_fact(candidate)
            if candidate_is_user and not current_is_user:
                return candidate
            if current_is_user and not candidate_is_user:
                return current

        if candidate.turn_idx < current.turn_idx:
            return current
        if candidate.turn_idx == current.turn_idx and candidate.id < current.id:
            return current

        current_rel = self._compute_reliability(current)
        candidate_rel = self._compute_reliability(candidate)
        if candidate_rel < current_rel * 0.8:
            return current
        return candidate

    def _is_user_fact(self, fact: Fact) -> bool:
        role = (fact.role or "").strip().lower()
        if role:
            return role == "user"
        source = (fact.source_text or "").lower()
        if "assistant" in source and "user" not in source:
            return False
        return "user" in source

    def _compute_reliability(self, fact: Fact) -> float:
        score = 0.5
        if self._is_user_fact(fact):
            score += 0.3
        source = (fact.source_text or "").lower()
        if any(marker in source for marker in ("actually", "i meant", "correction", "wait, ")):
            score += 0.1
        if any(marker in source for marker in ("i think", "maybe", "probably", "not sure")):
            score -= 0.1
        if (fact.sub_role or "").lower() == "hypothetical":
            score -= 0.3
        return max(0.0, min(1.0, score))


class EntityMerger:
    """Merge entities while preserving referential integrity."""

    def __init__(self, store: Optional[FoundryStore] = None) -> None:
        self.store = store

    def merge(self, source_canonical: str, target_canonical: str) -> None:
        source_name = _normalize_name(source_canonical)
        target_name = _normalize_name(target_canonical)
        if source_name == target_name:
            raise ValueError("Cannot merge an entity into itself")
        if self.store is None:
            raise ValueError("EntityMerger.merge() requires a FoundryStore")

        graph = self.store.entity_graph
        source = graph.get_entity(source_name)
        target = graph.get_entity(target_name)
        if source is None or target is None:
            raise ValueError("Both source and target entities must exist")

        self.merge_nodes(source, target)
        graph._entities.pop(source.canonical, None)
        graph._entities[target.canonical] = target
        self._rewrite_entity_references(source.canonical, target.canonical)
        self._rebuild_alias_index()
        self._rewrite_cooccurrence(source.canonical, target.canonical)
        self._rewrite_temporal(source.canonical, target.canonical)
        self._rewrite_chunks(source.canonical, target.canonical)
        self._rebuild_indexes()

    def merge_nodes(self, source: EntityNode, target: EntityNode) -> EntityNode:
        source_name = _normalize_name(source.canonical)
        target_name = _normalize_name(target.canonical)
        if source_name == target_name:
            raise ValueError("Cannot merge an entity into itself")

        target.aliases.update(source.aliases)
        target.aliases.add(source.canonical)
        target.aliases.discard(target.canonical)
        target.types.update(source.types)
        target.related.update(
            related
            for related in source.related
            if _normalize_name(related) not in {source_name, target_name}
        )
        target.related.discard(target.canonical)
        target.mention_turns.update(source.mention_turns)
        target.mention_chunks.update(source.mention_chunks)

        for type_label, count in source._type_counts.items():
            target._type_counts[type_label] = target._type_counts.get(type_label, 0) + count

        for predicate, facts in source.facts.items():
            target_bucket = target.facts.setdefault(predicate, [])
            for fact in facts:
                fact.subject = target_name
                target_bucket.append(fact)
            target_bucket.sort(key=lambda item: (item.turn_idx, item.id))

        for predicate, targets in source.links.items():
            rewritten = {
                target_name if _normalize_name(linked) == source_name else _normalize_name(linked)
                for linked in targets
            }
            rewritten.discard(target_name)
            if not rewritten:
                continue
            target.links.setdefault(predicate, set()).update(rewritten)

        for predicate, targets in target.links.items():
            normalized = {
                target_name if _normalize_name(linked) == source_name else _normalize_name(linked)
                for linked in targets
            }
            normalized.discard(target_name)
            target.links[predicate] = normalized

        target.related.update(
            linked for links in target.links.values() for linked in links if linked != target_name
        )
        target.related.discard(target_name)
        return target

    def _rewrite_entity_references(self, source: str, target: str) -> None:
        assert self.store is not None
        for entity in self.store.entity_graph._entities.values():
            entity.related = {
                target if _normalize_name(related) == source else _normalize_name(related)
                for related in entity.related
                if _normalize_name(related) != entity.canonical
            }
            for predicate, targets in list(entity.links.items()):
                rewritten = {
                    target if _normalize_name(linked) == source else _normalize_name(linked)
                    for linked in targets
                }
                rewritten.discard(entity.canonical)
                entity.links[predicate] = rewritten

    def _rebuild_alias_index(self) -> None:
        assert self.store is not None
        graph = self.store.entity_graph
        graph._alias_to_canonical = {}
        for entity in graph._entities.values():
            graph._alias_to_canonical[entity.canonical] = entity.canonical
            for alias in entity.aliases:
                graph._alias_to_canonical[_normalize_name(alias)] = entity.canonical

    def _rewrite_cooccurrence(self, source: str, target: str) -> None:
        assert self.store is not None
        old_index = self.store.cooccurrence
        new_index = CooccurrenceIndex(max_neighbors=old_index.max_neighbors)
        seen: set[tuple[str, str]] = set()
        for left, neighbors in old_index._adjacency.items():
            for right, chunks in neighbors.items():
                a = target if _normalize_name(left) == source else _normalize_name(left)
                b = target if _normalize_name(right) == source else _normalize_name(right)
                if a == b:
                    continue
                pair = tuple(sorted((a, b)))
                if pair in seen:
                    continue
                for chunk_idx in sorted(chunks):
                    new_index.record(a, b, chunk_idx)
                seen.add(pair)
        self.store.cooccurrence = new_index

    def _rewrite_temporal(self, source: str, target: str) -> None:
        assert self.store is not None
        for event in self.store.temporal._events:
            if event.subject and _normalize_name(event.subject) == source:
                event.subject = target
            if event.object and _normalize_name(event.object) == source:
                event.object = target

    def _rewrite_chunks(self, source: str, target: str) -> None:
        assert self.store is not None
        for chunk in self.store.chunks._chunks.values():
            updated: list[str] = []
            for entity in chunk.entities:
                normalized = _normalize_name(entity)
                replacement = target if normalized == source else normalized
                if replacement not in updated:
                    updated.append(replacement)
            chunk.entities = updated

    def _rebuild_indexes(self) -> None:
        assert self.store is not None
        self.store.types = TypeIndex()
        self.store.properties = PropertyIndex()
        self.store.bm25_entities = BM25Index()

        for doc_id, entity in enumerate(self.store.entity_graph.all_entities()):
            for type_label in entity.types:
                self.store.types.index(entity.canonical, type_label)
            for predicate, facts in entity.facts.items():
                for fact in facts:
                    self.store.properties.index(
                        entity.canonical,
                        predicate,
                        fact.object,
                        fact.object_type,
                    )
            text_parts = [entity.canonical, *sorted(entity.aliases), *sorted(entity.types)]
            for facts in entity.facts.values():
                text_parts.extend(fact.object for fact in facts)
            self.store.bm25_entities.add(doc_id, " ".join(part for part in text_parts if part))


__all__ = [
    "ACTION_TYPES",
    "DeletionPolicy",
    "EditVersion",
    "EntityMerger",
    "MemoryAction",
    "SupersessionEngine",
    "TemporalValidity",
    "Tombstone",
    "VersionTracker",
]
