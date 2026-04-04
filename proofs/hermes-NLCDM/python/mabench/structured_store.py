"""Structured fact store — O(1) entity and predicate lookup over SVO triples.

Indexes extracted (Subject, Verb, Object) triples into inverted indexes
for fast structured retrieval.  Each fact is anchored to a source turn
(turn_idx + turn_text) so results can be fused with BM25 and entity
channels via RRF.

Pure Python dicts, no external dependencies.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from mabench.event_calendar import _simple_stem


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StructuredFact:
    """A single extracted triple anchored to its source turn."""

    subject: str       # normalized (lowercased, stripped)
    predicate: str     # normalized (stemmed)
    object: str        # normalized (lowercased, stripped)
    turn_idx: int      # index into CoupledEngine.memory_store
    turn_text: str     # full text of the source turn
    fact_id: int       # auto-incremented unique ID within this store
    timestamp: datetime | None = None


# ---------------------------------------------------------------------------
# Scoring weights (module-level constants, overridable via config later)
# ---------------------------------------------------------------------------

WEIGHT_SUBJECT_EXACT: float = 5.0
WEIGHT_OBJECT_EXACT: float = 4.0
WEIGHT_ENTITY_PARTIAL: float = 2.0
WEIGHT_PREDICATE: float = 3.0
WEIGHT_OBJECT_WORD_OVERLAP: float = 1.0

# Minimum entity length for substring matching
_MIN_SUBSTRING_LEN = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize_entity(raw: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return _WS_RE.sub(" ", raw.strip().lower())


def _stem_predicate(raw: str) -> str:
    """Stem a predicate using _simple_stem from event_calendar.py.

    Multi-word predicates: stem each word independently and rejoin.
    Reuses the existing naive suffix stripper which handles:
    - tomatoes -> tomato, classes -> class
    - running -> run, attended -> attend
    """
    words = raw.strip().lower().split()
    stemmed = [_simple_stem(w) for w in words if w]
    return " ".join(stemmed)


# ---------------------------------------------------------------------------
# StructuredFactStore
# ---------------------------------------------------------------------------

class StructuredFactStore:
    """Dict-indexed fact store for O(1) entity and predicate lookup.

    Population: called during ingestion from SVO cache (pre-extracted)
    or live LFM/LLM extraction.  Each fact gets indexed in three
    inverted indexes:

    - entity_index:    entity_lower -> list[(fact_id, position)]
    - predicate_index: predicate_stem -> list[fact_id]
    - turn_index:      turn_idx -> list[fact_id]

    Scoring model:
    - Exact entity match in subject position:  5.0
    - Exact entity match in object position:   4.0
    - Partial entity match (substring >= 4ch): 2.0
    - Predicate stem match:                    3.0
    - Object text overlap with hints:          1.0 per word

    Pure Python dicts, no external dependencies.
    """

    def __init__(self) -> None:
        """Initialize empty indexes."""
        self._facts: list[StructuredFact] = []
        # entity_lower -> list[(fact_id, "subject"|"object")]
        self._entity_index: dict[str, list[tuple[int, str]]] = defaultdict(list)
        # pred_stem -> list[fact_id]
        self._predicate_index: dict[str, list[int]] = defaultdict(list)
        # turn_idx -> list[fact_id]
        self._turn_index: dict[int, list[int]] = defaultdict(list)
        # turn_idx -> turn_text (for returning turn_text in query results)
        self._turn_texts: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API: ingestion
    # ------------------------------------------------------------------

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        turn_idx: int,
        turn_text: str,
        timestamp: datetime | None = None,
    ) -> int:
        """Add a single fact and update all indexes.

        Parameters
        ----------
        subject : str
            Raw subject string (will be lowercased + stripped internally).
        predicate : str
            Raw predicate/verb string (will be stemmed internally).
        obj : str
            Raw object string (will be lowercased + stripped internally).
        turn_idx : int
            Index into CoupledEngine.memory_store for the source turn.
        turn_text : str
            Full text of the source conversation turn.
        timestamp : datetime | None
            Parsed timestamp, or None if not available.

        Returns
        -------
        int
            The assigned fact_id for this fact.
        """
        fact_id = len(self._facts)
        norm_subj = _normalize_entity(subject)
        norm_obj = _normalize_entity(obj)
        pred_stem = _stem_predicate(predicate)

        fact = StructuredFact(
            subject=norm_subj,
            predicate=pred_stem,
            object=norm_obj,
            turn_idx=turn_idx,
            turn_text=turn_text,
            fact_id=fact_id,
            timestamp=timestamp,
        )
        self._facts.append(fact)

        # Index entities with position tags
        if norm_subj:
            self._entity_index[norm_subj].append((fact_id, "subject"))
        if norm_obj:
            self._entity_index[norm_obj].append((fact_id, "object"))

        # Index predicate (each word stem is indexed separately for partial match)
        if pred_stem:
            self._predicate_index[pred_stem].append(fact_id)
            # Also index individual stemmed words for multi-word predicates
            stem_words = pred_stem.split()
            if len(stem_words) > 1:
                for sw in stem_words:
                    if sw:
                        self._predicate_index[sw].append(fact_id)

        # Index turn
        self._turn_index[turn_idx].append(fact_id)

        # Store turn text mapping
        self._turn_texts[turn_idx] = turn_text

        return fact_id

    def add_batch(
        self,
        triples: list[tuple[str, str, str]],
        turn_idx: int,
        turn_text: str,
        timestamp: datetime | None = None,
    ) -> list[int]:
        """Add multiple triples from the same turn.

        Parameters
        ----------
        triples : list[tuple[str, str, str]]
            List of (subject, predicate, object) tuples.
        turn_idx : int
            Source turn index.
        turn_text : str
            Source turn full text.
        timestamp : datetime | None
            Parsed timestamp, or None.

        Returns
        -------
        list[int]
            List of assigned fact_ids.
        """
        fact_ids = []
        for subj, pred, obj in triples:
            fid = self.add_fact(
                subject=subj, predicate=pred, obj=obj,
                turn_idx=turn_idx, turn_text=turn_text,
                timestamp=timestamp,
            )
            fact_ids.append(fid)
        return fact_ids

    # ------------------------------------------------------------------
    # Public API: retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        entity_hints: list[str] | None = None,
        predicate_hints: list[str] | None = None,
        object_hints: list[str] | None = None,
        top_k: int = 30,
    ) -> list[tuple[int, str, float]]:
        """Query the store, return (turn_idx, turn_text, score) ranked by score.

        Scoring:
        1. For each entity_hint, find matching fact_ids via entity_index.
           - Exact match on subject: +5.0
           - Exact match on object:  +4.0
           - Substring match (len >= 4): +2.0 per matching entity
        2. For each predicate_hint, stem it and find matching fact_ids.
           - Exact stem match: +3.0
        3. Object hints: +1.0 per overlapping word with fact object text.
        4. Aggregate scores per fact_id, deduplicate by turn_idx (max score
           per turn), sort descending, return top_k.

        Parameters
        ----------
        entity_hints : list[str] | None
            Entity strings extracted from the query (will be lowercased).
        predicate_hints : list[str] | None
            Predicate/verb strings extracted from the query.
        object_hints : list[str] | None
            Object hint words for text overlap scoring.
        top_k : int
            Max results to return.

        Returns
        -------
        list[tuple[int, str, float]]
            List of (turn_idx, turn_text, score) sorted by descending score.
            Turn indices are deduplicated -- multiple facts from the same
            turn contribute their max score, not sum.
        """
        entity_hints = entity_hints or []
        predicate_hints = predicate_hints or []
        object_hints = object_hints or []

        if not entity_hints and not predicate_hints and not object_hints:
            return []

        # fact_id -> best score for that fact
        fact_scores: dict[int, float] = defaultdict(float)

        # --- Entity scoring ---
        for hint in entity_hints:
            norm_hint = _normalize_entity(hint)
            if not norm_hint:
                continue

            # Exact match: look up the entity in the index directly
            if norm_hint in self._entity_index:
                for fid, position in self._entity_index[norm_hint]:
                    if position == "subject":
                        fact_scores[fid] = max(
                            fact_scores[fid], WEIGHT_SUBJECT_EXACT
                        )
                    else:
                        fact_scores[fid] = max(
                            fact_scores[fid], WEIGHT_OBJECT_EXACT
                        )

            # Substring matching: check all indexed entities
            if len(norm_hint) >= _MIN_SUBSTRING_LEN:
                for entity_key, entries in self._entity_index.items():
                    if entity_key == norm_hint:
                        continue  # already handled by exact match
                    # Check if hint is a substring of the indexed entity
                    # or the indexed entity is a substring of the hint
                    if norm_hint in entity_key or entity_key in norm_hint:
                        for fid, _position in entries:
                            fact_scores[fid] = max(
                                fact_scores[fid], WEIGHT_ENTITY_PARTIAL
                            )

        # --- Predicate scoring (additive on top of entity scores) ---
        for hint in predicate_hints:
            stem = _stem_predicate(hint)
            if not stem:
                continue
            # Check for exact stem match in predicate index
            if stem in self._predicate_index:
                for fid in self._predicate_index[stem]:
                    base = fact_scores.get(fid, 0.0)
                    fact_scores[fid] = max(
                        fact_scores[fid], base + WEIGHT_PREDICATE
                    )

        # --- Object hint word overlap scoring ---
        if object_hints:
            obj_words = set()
            for oh in object_hints:
                for w in oh.lower().split():
                    if len(w) >= 2:
                        obj_words.add(w)
            if obj_words:
                for fact in self._facts:
                    fact_obj_words = set(fact.object.split())
                    overlap = obj_words & fact_obj_words
                    if overlap:
                        bonus = len(overlap) * WEIGHT_OBJECT_WORD_OVERLAP
                        fact_scores[fact.fact_id] = (
                            fact_scores.get(fact.fact_id, 0.0) + bonus
                        )

        if not fact_scores:
            return []

        # --- Deduplicate by turn_idx: keep MAX score per turn ---
        turn_best: dict[int, float] = {}
        for fid, score in fact_scores.items():
            fact = self._facts[fid]
            tidx = fact.turn_idx
            if tidx not in turn_best or score > turn_best[tidx]:
                turn_best[tidx] = score

        # Sort by score descending, then by turn_idx ascending for stability
        sorted_turns = sorted(
            turn_best.items(),
            key=lambda x: (-x[1], x[0]),
        )

        # Apply top_k and build result tuples with turn_text
        results: list[tuple[int, str, float]] = []
        for tidx, score in sorted_turns[:top_k]:
            ttext = self._turn_texts.get(tidx, "")
            results.append((tidx, ttext, score))

        return results

    def query_by_turn(self, turn_idx: int) -> list[StructuredFact]:
        """Return all facts from a specific turn.

        Parameters
        ----------
        turn_idx : int
            Turn index to look up.

        Returns
        -------
        list[StructuredFact]
            All facts stored for this turn, in insertion order.
        """
        if turn_idx not in self._turn_index:
            return []
        return [self._facts[fid] for fid in self._turn_index[turn_idx]]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_facts(self) -> int:
        """Total number of indexed facts."""
        return len(self._facts)

    @property
    def n_entities(self) -> int:
        """Number of unique entity keys in the entity index."""
        return len(self._entity_index)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all facts and indexes.  Called on agent.reset()."""
        self._facts.clear()
        self._entity_index.clear()
        self._predicate_index.clear()
        self._turn_index.clear()
        self._turn_texts.clear()

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_svo_cache(
        cls,
        cache: dict[int, list[tuple[str, str, str]]],
        turns: list[str],
    ) -> StructuredFactStore:
        """Build from SVO extraction cache.

        Parameters
        ----------
        cache : dict[int, list[tuple[str, str, str]]]
            Mapping from turn_idx to list of (subject, predicate, object).
        turns : list[str]
            Ordered list of turn texts (indexed by turn_idx).

        Returns
        -------
        StructuredFactStore
            Populated store ready for querying.
        """
        store = cls()
        for turn_idx in sorted(cache.keys()):
            triples = cache[turn_idx]
            turn_text = turns[turn_idx] if turn_idx < len(turns) else ""
            store.add_batch(
                triples=triples,
                turn_idx=turn_idx,
                turn_text=turn_text,
            )
        return store
