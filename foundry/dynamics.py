"""Memory dynamics helpers for Engram Foundry.

The v1 implementation stays deterministic and conservative. It exposes the
specified surface without introducing speculative mutation of the core memory
store.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Optional

from foundry.storage import DerivedGraph, EntityGraph, EntityNode, _normalize_name


@dataclass
class ImportanceScore:
    """Multi-signal importance score for an entity or fact."""

    mention_frequency: float
    recency: float
    centrality: float
    retrieval_frequency: float
    composite: float = 0.0

    @staticmethod
    def compute(
        entity: EntityNode,
        total_turns: int,
        current_turn: int,
        retrieval_count: int,
        max_degree: int,
    ) -> "ImportanceScore":
        mention_count = len(getattr(entity, "mention_turns", set()))
        mention_frequency = 0.0
        if total_turns > 0:
            mention_frequency = min(1.0, mention_count / float(total_turns))

        if mention_count == 0:
            recency = 0.0
        else:
            last_turn = max(entity.mention_turns)
            turns_since = max(0, current_turn - last_turn)
            horizon = max(total_turns, 1)
            recency = math.exp(-(turns_since / float(horizon)))

        degree = len(getattr(entity, "related", set()))
        centrality = min(1.0, degree / float(max(max_degree, 1)))
        retrieval_frequency = min(1.0, max(0, retrieval_count) / float(max(total_turns, 1)))

        composite = (
            (mention_frequency * 0.3)
            + (recency * 0.3)
            + (centrality * 0.2)
            + (retrieval_frequency * 0.2)
        )
        return ImportanceScore(
            mention_frequency=mention_frequency,
            recency=recency,
            centrality=centrality,
            retrieval_frequency=retrieval_frequency,
            composite=max(0.0, min(1.0, composite)),
        )


class DecayFunction:
    """Temporal decay of fact salience."""

    def __init__(self, decay_lambda: float = 0.01) -> None:
        self.decay_lambda = float(decay_lambda)

    def compute(
        self,
        days_since_access: float,
        base_salience: float = 1.0,
        superseded: bool = False,
    ) -> float:
        days = max(0.0, float(days_since_access))
        decay_lambda = self.decay_lambda * (10.0 if superseded else 1.0)
        return float(base_salience) * math.exp(-(decay_lambda * days))

    def should_forget(self, salience: float, threshold: float = 0.01) -> bool:
        return float(salience) < float(threshold)


class HebbianBridge:
    """Tracks successful retrieval patterns for future hard-coding."""

    def __init__(self) -> None:
        self._patterns: dict[tuple[str, tuple[int, ...], tuple[str, ...]], dict[str, Any]] = {}

    def record_retrieval(
        self,
        question: str,
        evidence_turns: list[int],
        tool_sequence: list[str],
    ) -> None:
        key = (
            _normalize_name(question or ""),
            tuple(int(turn) for turn in evidence_turns),
            tuple(str(tool) for tool in tool_sequence),
        )
        record = self._patterns.setdefault(
            key,
            {
                "question": str(question or ""),
                "evidence_turns": list(int(turn) for turn in evidence_turns),
                "tool_sequence": list(str(tool) for tool in tool_sequence),
                "count": 0,
            },
        )
        record["count"] += 1

    def get_frequent_patterns(self, min_count: int = 5) -> list[dict]:
        rows = [
            dict(record)
            for record in self._patterns.values()
            if int(record.get("count", 0)) >= int(min_count)
        ]
        rows.sort(key=lambda row: (-int(row["count"]), row["question"]))
        return rows


class DreamConsolidator:
    """Bounded dream consolidation stub with explicit safeguards."""

    def __init__(self, drift_threshold: float = 0.25) -> None:
        self.drift_threshold = float(drift_threshold)

    def consolidate(self, max_cycles: int = 5) -> dict:
        requested = max(0, int(max_cycles))
        cycles_run = min(requested, 5)
        quality_delta = 0.0
        drift_detected = False

        for cycle in range(cycles_run):
            synthetic_drift = 0.0
            if synthetic_drift > self.drift_threshold:
                drift_detected = True
                cycles_run = cycle
                break

        return {
            "cycles_run": cycles_run,
            "drift_detected": drift_detected,
            "quality_delta": quality_delta,
            "rolled_back": False,
        }


class MemorySystem:
    """The complete three-layer memory system."""

    def __init__(self) -> None:
        self.stated: EntityGraph = EntityGraph()
        self.inferred: EntityGraph = EntityGraph()
        self.derived: DerivedGraph = DerivedGraph()
        self._bridge_lookup: dict[str, set[str]] = defaultdict(set)

    def traverse_layers(self, query_term: str) -> list[str]:
        normalized = _normalize_name(query_term or "")
        if not normalized:
            return []

        direct = self.stated.get_entity(normalized)
        if direct is not None:
            return [direct.canonical]

        candidates = set(self._bridge_lookup.get(normalized, set()))
        inferred_node = self.inferred.get_entity(normalized)
        if inferred_node is not None:
            for neighbor in inferred_node.related:
                candidates.update(self._bridge_lookup.get(neighbor, set()))
            for targets in inferred_node.links.values():
                for target in targets:
                    candidates.update(self._bridge_lookup.get(target, set()))
                    if self.stated.get_entity(target) is not None:
                        candidates.add(target)

        stated_only = sorted(
            canonical
            for canonical in candidates
            if self.stated.get_entity(canonical) is not None
        )
        return stated_only

    def wipe_inferred(self) -> None:
        self.inferred = EntityGraph()
        self._bridge_lookup = defaultdict(set)

    def rebuild_derived(self) -> None:
        type_counts: dict[str, int] = {}
        for entity in self.stated.all_entities():
            for type_label in entity.types:
                type_counts[type_label] = type_counts.get(type_label, 0) + 1
        derived = DerivedGraph()
        derived.type_counts = dict(sorted(type_counts.items()))
        self.derived = derived

    def build_inferred(self, synonym_dict: dict[str, list[str]]) -> None:
        self.wipe_inferred()

        for entity in self.stated.all_entities():
            self._bridge_lookup[entity.canonical].add(entity.canonical)
            inferred_entity = self.inferred.ensure_entity(entity.canonical)
            inferred_entity.aliases.update(entity.aliases)
            inferred_entity.types.update(entity.types)

            for alias in entity.aliases:
                alias_key = _normalize_name(alias)
                self._bridge_lookup[alias_key].add(entity.canonical)
                alias_node = self.inferred.ensure_entity(alias_key)
                alias_node.add_link("alias_of", entity.canonical)
                alias_node.related.add(entity.canonical)

            for type_label in entity.types:
                self._bridge_lookup[type_label].add(entity.canonical)
                type_node = self.inferred.ensure_entity(type_label)
                type_node.add_link("instance", entity.canonical)
                type_node.related.add(entity.canonical)
                inferred_entity.add_link("type", type_label)
                inferred_entity.related.add(type_label)

                for synonym in synonym_dict.get(type_label, []):
                    synonym_key = _normalize_name(synonym)
                    self._bridge_lookup[synonym_key].add(entity.canonical)
                    synonym_node = self.inferred.ensure_entity(synonym_key)
                    synonym_node.add_link("synonym_of", type_label)
                    synonym_node.related.add(type_label)
                    type_node.add_link("synonym", synonym_key)
                    type_node.related.add(synonym_key)


__all__ = [
    "MemorySystem",
    "ImportanceScore",
    "DecayFunction",
    "HebbianBridge",
    "DreamConsolidator",
]
