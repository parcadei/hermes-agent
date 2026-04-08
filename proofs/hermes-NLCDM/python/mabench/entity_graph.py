"""Entity-centric knowledge graph for conversational memory.

Builds a per-entity graph during ingestion. At query time, entity mentions
are resolved through alias matching and the graph returns evidence turn indices.
"""
from __future__ import annotations
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

_DETERMINERS = re.compile(
    r"^(the|a|an|my|our|that|this|his|her|their|its)\s+", re.IGNORECASE)
_POSSESSIVE = re.compile(r"'s\b", re.IGNORECASE)
_STOP_WORDS = {
    "what", "when", "where", "who", "how", "which", "does", "did", "was",
    "were", "is", "are", "has", "have", "had", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "with", "from", "by", "about", "it",
    "and", "or", "but", "not", "do", "can", "will", "would", "should",
    "could", "much", "many", "some", "any", "most", "very", "just",
    "than", "then", "also", "been", "being", "its", "that", "this",
    "these", "those", "they", "them", "their", "there", "here",
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "him", "her", "his", "us", "up", "out", "if", "so", "no", "yes",
}


@dataclass
class Fact:
    """A single predicate-object fact about an entity."""
    predicate: str
    object: str
    turn_idx: int
    chunk_idx: int = -1
    timestamp: Optional[datetime] = None
    superseded: bool = False
    source_text: str = ""


@dataclass
class EntityNode:
    """A node in the entity graph representing one real-world entity."""
    canonical: str
    aliases: set[str] = field(default_factory=set)
    facts: dict[str, list[Fact]] = field(default_factory=dict)
    related: set[str] = field(default_factory=set)
    mention_turns: set[int] = field(default_factory=set)
    _explicit_mentions: set[int] = field(default_factory=set, repr=False)


class EntityGraph:
    """Per-entity knowledge graph built incrementally during ingestion."""

    def __init__(self) -> None:
        self.entities: dict[str, EntityNode] = {}
        self.alias_index: dict[str, str] = {}

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, strip, collapse whitespace."""
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _strip_determiners(text: str) -> str:
        """Remove leading determiners."""
        return _DETERMINERS.sub("", text)

    @staticmethod
    def _strip_possessive(text: str) -> str:
        """Remove possessive 's from text."""
        return _POSSESSIVE.sub("", text)

    def resolve(self, mention: str) -> Optional[str]:
        """Resolve a mention to its canonical entity name.

        Order: exact norm -> determiner-stripped -> substring containment.
        """
        norm = self._normalize(mention)
        if norm in self.alias_index:
            return self.alias_index[norm]
        stripped = self._normalize(self._strip_determiners(norm))
        if stripped and stripped in self.alias_index:
            return self.alias_index[stripped]
        for canon in self.entities:
            if len(norm) >= 2 and len(canon) >= 2:
                if canon in norm or norm in canon:
                    return canon
        return None

    def _resolve_or_create(self, mention: str) -> str:
        """Resolve mention to canonical name, creating a new entity if needed."""
        canonical = self.resolve(mention)
        if canonical is not None:
            norm = self._normalize(mention)
            if norm != canonical:
                self.alias_index[norm] = canonical
                self.entities[canonical].aliases.add(norm)
            return canonical
        norm = self._normalize(mention)
        stripped = self._normalize(self._strip_determiners(norm))
        canon_key = stripped if stripped else norm
        node = EntityNode(canonical=canon_key)
        node.aliases.add(norm)
        if norm != canon_key:
            node.aliases.add(canon_key)
        self.entities[canon_key] = node
        self.alias_index[norm] = canon_key
        self.alias_index[canon_key] = canon_key
        return canon_key

    def ingest_triple(self, subj: str, pred: str, obj: str, turn_idx: int,
                      chunk_idx: int = -1, timestamp: Optional[datetime] = None,
                      source_text: str = "") -> None:
        """Ingest one S-P-O triple. Supersedes prior facts with same (entity, pred)."""
        if not subj or not subj.strip():
            return
        if not pred or not pred.strip():
            return
        if not obj or not obj.strip():
            return
        subj_canon = self._resolve_or_create(subj)
        obj_canon = self._resolve_or_create(obj)
        pred_norm = self._normalize(pred)
        subj_node = self.entities[subj_canon]
        obj_node = self.entities[obj_canon]
        if pred_norm in subj_node.facts:
            # Check for exact duplicate (same pred, object, turn_idx)
            for existing in subj_node.facts[pred_norm]:
                if (existing.object == obj and existing.turn_idx == turn_idx
                        and not existing.superseded):
                    return  # exact duplicate, skip
            for existing in subj_node.facts[pred_norm]:
                existing.superseded = True
        fact = Fact(predicate=pred_norm, object=obj, turn_idx=turn_idx,
                    chunk_idx=chunk_idx, timestamp=timestamp,
                    superseded=False, source_text=source_text)
        subj_node.facts.setdefault(pred_norm, []).append(fact)
        subj_node.related.add(obj_canon)
        obj_node.related.add(subj_canon)
        subj_node.mention_turns.add(turn_idx)
        obj_node.mention_turns.add(turn_idx)

    def ingest_entity_mention(self, entity: str, turn_idx: int) -> None:
        """Record that an entity was mentioned in a turn."""
        node = self.entities[self._resolve_or_create(entity)]
        node.mention_turns.add(turn_idx)
        node._explicit_mentions.add(turn_idx)

    def register_alias(self, alias: str, canonical: str) -> None:
        """Register an alias for a canonical entity, merging if alias was separate."""
        canon_norm = self._normalize(canonical)
        alias_norm = self._normalize(alias)
        if canon_norm not in self.entities:
            node = EntityNode(canonical=canon_norm)
            node.aliases.add(canon_norm)
            self.entities[canon_norm] = node
            self.alias_index[canon_norm] = canon_norm
        canon_node = self.entities[canon_norm]
        if alias_norm in self.entities and alias_norm != canon_norm:
            old = self.entities.pop(alias_norm)
            for pred, fl in old.facts.items():
                canon_node.facts.setdefault(pred, []).extend(fl)
            canon_node.related |= old.related
            canon_node.related.discard(alias_norm)
            canon_node.related.discard(canon_norm)
            canon_node.mention_turns |= old.mention_turns
            canon_node._explicit_mentions |= old._explicit_mentions
            canon_node.aliases |= old.aliases
            for a, c in list(self.alias_index.items()):
                if c == alias_norm:
                    self.alias_index[a] = canon_norm
            for oc in list(self.entities):
                on = self.entities[oc]
                if alias_norm in on.related:
                    on.related.discard(alias_norm)
                    if oc != canon_norm:
                        on.related.add(canon_norm)
        canon_node.aliases.add(alias_norm)
        self.alias_index[alias_norm] = canon_norm

    def query_entity(self, entity: str) -> Optional[EntityNode]:
        """Look up an entity node by mention (resolves aliases)."""
        canon = self.resolve(entity)
        return self.entities.get(canon) if canon else None

    def query_facts(self, entity: str, predicate: Optional[str] = None,
                    current_only: bool = True) -> list[Fact]:
        """Get facts about an entity, optionally filtered by predicate. Newest first."""
        canon = self.resolve(entity)
        if canon is None:
            return []
        node = self.entities.get(canon)
        if node is None:
            return []
        result: list[Fact] = []
        preds = [self._normalize(predicate)] if predicate else list(node.facts)
        for p in preds:
            for f in node.facts.get(p, []):
                if not (current_only and f.superseded):
                    result.append(f)
        result.sort(key=lambda f: f.turn_idx, reverse=True)
        return result

    def get_evidence_turns(self, entity: str,
                           predicate: Optional[str] = None) -> set[int]:
        """Get turn indices: explicit mentions + non-superseded fact turns."""
        canon = self.resolve(entity)
        if canon is None:
            return set()
        node = self.entities.get(canon)
        if node is None:
            return set()
        turns = set(node._explicit_mentions)
        preds = [self._normalize(predicate)] if predicate else list(node.facts)
        for p in preds:
            for f in node.facts.get(p, []):
                if not f.superseded:
                    turns.add(f.turn_idx)
        return turns

    def get_related_entities(self, entity: str, max_hops: int = 1) -> set[str]:
        """BFS through 'related' edges up to max_hops. Excludes seed entity."""
        canon = self.resolve(entity)
        if canon is None or canon not in self.entities:
            return set()
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(canon, 0)])
        while queue:
            cur, depth = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            if depth >= max_hops:
                continue
            node = self.entities.get(cur)
            if node:
                for nb in node.related:
                    if nb not in visited:
                        queue.append((nb, depth + 1))
        visited.discard(canon)
        return visited

    def retrieve_for_question(self, question_text: str,
                               max_turns: int = 20) -> list[int]:
        """Extract entities from question, return evidence turn indices ranked by frequency."""
        if not question_text or not question_text.strip():
            return []
        candidates = self._extract_question_entities(question_text)
        resolved: set[str] = set()
        for cand in candidates:
            canon = self.resolve(cand)
            if canon is not None:
                resolved.add(canon)
        if not resolved:
            return []
        all_ents = set(resolved)
        for ent in resolved:
            all_ents |= self.get_related_entities(ent, max_hops=1)
        tc: dict[int, int] = defaultdict(int)
        for ent in all_ents:
            for t in self.get_evidence_turns(ent):
                tc[t] += 1
        ranked = sorted(tc, key=lambda t: (-tc[t], t))
        return ranked[:max_turns]

    def answer_from_graph(
        self, entity: str, predicate: str, question: str,
        max_hops: int = 3,
    ) -> Optional[str]:
        """Try to answer a question by walking the graph with predicate ranking.

        Given a pre-extracted (entity, predicate) pair from the question,
        resolve the entity, rank ALL candidate predicates (direct + 1-hop
        neighbors) by word overlap with the full question text, and follow
        the chain up to max_hops.

        Returns the answer string or None if nothing found.
        """
        if not entity or not predicate or not question:
            return None
        q_words = set(
            re.sub(r"[?.!,'\"]", "", question.lower()).split()
        ) - _STOP_WORDS
        # Merge the LLM-extracted predicate words so "age" matches even when
        # the question says "how old" (no word overlap with "age").
        pred_words = set(predicate.lower().split()) - _STOP_WORDS
        match_words = q_words | pred_words

        current_entity = entity
        visited: set[str] = set()
        pred_satisfied = False

        for _ in range(max_hops):
            canon = self.resolve(current_entity)
            if canon is None or canon in visited:
                return None
            visited.add(canon)

            node = self.entities.get(canon)
            if node is None:
                return None

            best = self._rank_predicates(canon, node, match_words, visited)
            if best is None:
                return None

            _score, best_canon, best_pred, value = best
            facts = self.query_facts(best_canon, best_pred, current_only=True)
            if not facts:
                return None
            value = facts[0].object

            # Track whether the extracted predicate has been matched.
            bp_words = set(best_pred.split()) - _STOP_WORDS
            if bp_words & pred_words:
                pred_satisfied = True

            # If value is a leaf (not an entity, or entity with no facts), done.
            next_canon = self.resolve(value)
            if next_canon is None or next_canon not in self.entities:
                return value
            next_node = self.entities[next_canon]
            if not next_node.facts:
                return value

            # Termination: check if the next entity's DIRECT predicates
            # are relevant (excluding the predicate we just matched).
            has_forward_pred = False
            for p in next_node.facts:
                if p != best_pred and self._pred_score(p, match_words) > 0:
                    has_forward_pred = True
                    break

            if has_forward_pred:
                current_entity = value
                continue

            # If the extracted predicate was already satisfied, stop here.
            if pred_satisfied:
                return value

            # Predicate not yet satisfied — the answer might be further.
            # Continue walking if the next entity is reachable.
            if next_canon not in visited:
                current_entity = value
                continue

            return value

        return None

    def _rank_predicates(
        self, canon: str, node: EntityNode, q_words: set[str],
        visited: Optional[set[str]] = None,
    ) -> Optional[tuple[float, str, str, str]]:
        """Score all predicates on entity + 1-hop neighbors by question-word overlap.

        Returns (score, entity_canon, predicate, first_fact_value) or None.
        Neighbors in ``visited`` are skipped to prevent backward walks.
        """
        candidates: list[tuple[float, str, str, str]] = []

        # Direct predicates
        for pred in node.facts:
            score = self._pred_score(pred, q_words)
            facts = [f for f in node.facts[pred] if not f.superseded]
            if facts:
                candidates.append((score, canon, pred, facts[-1].object))

        # 1-hop neighbor predicates (skip visited to avoid backward walks)
        skip = visited or set()
        for nb in node.related:
            if nb in skip:
                continue
            nb_node = self.entities.get(nb)
            if nb_node is None:
                continue
            for pred in nb_node.facts:
                score = self._pred_score(pred, q_words)
                facts = [f for f in nb_node.facts[pred] if not f.superseded]
                if facts:
                    candidates.append((score, nb, pred, facts[-1].object))

        if not candidates:
            return None

        # Pick highest score; break ties by preferring direct entity
        candidates.sort(key=lambda c: (-c[0], 0 if c[1] == canon else 1))
        best = candidates[0]
        return best if best[0] > 0 else None

    @staticmethod
    def _pred_score(predicate: str, q_words: set[str]) -> float:
        """Score a predicate by word overlap with question words."""
        p_words = set(predicate.split()) - _STOP_WORDS
        if not p_words:
            return 0.0
        overlap = p_words & q_words
        return len(overlap) / len(p_words)

    @staticmethod
    def _extract_question_entities(text: str) -> list[str]:
        """Extract candidate entity mentions via bigram/trigram heuristic."""
        text = re.sub(r"\?", "", text)
        text = re.sub(r"^(what|when|where|who|how|which)\s+"
                       r"(is|are|was|were|did|does|do)\s+",
                       "", text, flags=re.IGNORECASE)
        words = text.lower().split()
        sig = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        cands: list[str] = []
        for i in range(len(sig)):
            cands.append(sig[i])
            if i + 1 < len(sig):
                cands.append(f"{sig[i]} {sig[i + 1]}")
            if i + 2 < len(sig):
                cands.append(f"{sig[i]} {sig[i + 1]} {sig[i + 2]}")
        return cands

    def to_dict(self) -> dict:
        """Serialize the graph to a JSON-compatible dict."""
        entities = {}
        for canon, node in self.entities.items():
            facts_d: dict[str, list[dict]] = {}
            for pred, fl in node.facts.items():
                facts_d[pred] = [{
                    "predicate": f.predicate, "object": f.object,
                    "turn_idx": f.turn_idx, "chunk_idx": f.chunk_idx,
                    "timestamp": f.timestamp.isoformat() if f.timestamp else None,
                    "superseded": f.superseded, "source_text": f.source_text,
                } for f in fl]
            entities[canon] = {
                "canonical": node.canonical,
                "aliases": sorted(node.aliases),
                "facts": facts_d,
                "related": sorted(node.related),
                "mention_turns": sorted(node.mention_turns),
                "explicit_mentions": sorted(node._explicit_mentions),
            }
        return {"entities": entities, "alias_index": dict(self.alias_index)}

    @classmethod
    def from_dict(cls, data: dict) -> EntityGraph:
        """Reconstruct an EntityGraph from a dict (inverse of to_dict)."""
        g = cls()
        g.alias_index = dict(data.get("alias_index", {}))
        for canon, nd in data.get("entities", {}).items():
            facts: dict[str, list[Fact]] = {}
            for pred, fl in nd.get("facts", {}).items():
                facts[pred] = [Fact(
                    predicate=fd["predicate"], object=fd["object"],
                    turn_idx=fd["turn_idx"], chunk_idx=fd.get("chunk_idx", -1),
                    timestamp=(datetime.fromisoformat(fd["timestamp"])
                               if fd.get("timestamp") else None),
                    superseded=fd.get("superseded", False),
                    source_text=fd.get("source_text", ""),
                ) for fd in fl]
            g.entities[canon] = EntityNode(
                canonical=nd["canonical"],
                aliases=set(nd.get("aliases", [])),
                facts=facts,
                related=set(nd.get("related", [])),
                mention_turns=set(nd.get("mention_turns", [])),
                _explicit_mentions=set(nd.get("explicit_mentions", [])),
            )
        return g

    def stats(self) -> dict:
        """Return summary statistics about the graph."""
        ec = len(self.entities)
        fc = cfc = ta = 0
        for node in self.entities.values():
            ta += len(node.aliases)
            for fl in node.facts.values():
                for f in fl:
                    fc += 1
                    if not f.superseded:
                        cfc += 1
        return {
            "entity_count": ec, "fact_count": fc,
            "current_fact_count": cfc,
            "alias_count": len(self.alias_index),
            "avg_facts_per_entity": fc / ec if ec else 0,
            "avg_aliases_per_entity": ta / ec if ec else 0,
        }
