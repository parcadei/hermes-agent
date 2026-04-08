"""Structured event database — entity-keyed + date-keyed triple storage.

The EventDB is the PRIMARY retrieval path when --structured_primary is on.
BM25 + entity_prefilter becomes the FALLBACK.

Architecture:
  - Entity index: lowercased entity name -> list of Events
  - Date index: date -> list of Events
  - Query-time: extract entities from question -> EventDB lookup -> chunk_ids

User's spec (session 91d782c0):
  "entity -> [(verb, object, chunk_id, timestamp)] dict, and query-time
   entity extraction to look up matching triples"
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(slots=True)
class Event:
    """A single structured event extracted from text."""

    subject: str
    predicate: str
    object: str
    chunk_id: int
    date: date | None = None
    session_id: str | None = None


class EventDB:
    """Entity-keyed structured event database.

    Primary index: entity_name (lowered) -> list[Event]
    Secondary index: date -> list[Event]
    """

    def __init__(self) -> None:
        self._entity_index: dict[str, list[Event]] = {}
        self._date_index: dict[date, list[Event]] = {}
        self._type_index: dict[str, set[str]] = {}
        self._predicate_index: dict[str, list[Event]] = {}
        self._chunk_index: dict[int, list[Event]] = {}

    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        chunk_id: int,
        event_date: date | None = None,
        session_id: str | None = None,
    ) -> None:
        event = Event(subject, predicate, obj, chunk_id, event_date, session_id)

        subj_key = subject.lower().strip()
        if subj_key and len(subj_key) >= 2:
            self._entity_index.setdefault(subj_key, []).append(event)

        obj_key = obj.lower().strip()
        if obj_key and len(obj_key) >= 2:
            self._entity_index.setdefault(obj_key, []).append(event)

        if event_date is not None:
            self._date_index.setdefault(event_date, []).append(event)

        # Type index: (X, "type", Y) -> type_index[Y_lower] = {X_lower, ...}
        pred_key = predicate.lower().strip()
        if pred_key == "type" and obj_key:
            self._type_index.setdefault(obj_key, set()).add(subj_key)

        # Predicate index
        if pred_key:
            self._predicate_index.setdefault(pred_key, []).append(event)

        # Chunk index
        self._chunk_index.setdefault(chunk_id, []).append(event)

    def add_batch(
        self,
        triples: list[tuple[str, str, str]],
        chunk_id: int,
        event_date: date | None = None,
        session_id: str | None = None,
        event_times: list[datetime | None] | None = None,
    ) -> None:
        for i, (s, p, o) in enumerate(triples):
            d = event_date
            if event_times and i < len(event_times) and event_times[i] is not None:
                et = event_times[i]
                d = et.date() if isinstance(et, datetime) else et
            self.add(s, p, o, chunk_id, d, session_id)

    def lookup(
        self,
        entities: list[str] | set[str],
        top_k: int = 10,
        date_center: date | None = None,
        date_window_days: int = 180,
    ) -> list[tuple[int, float]]:
        """Look up chunk_ids by entity names, ranked by hit count.

        Args:
            entities: Entity names extracted from the query.
            top_k: Max chunk_ids to return.
            date_center: When set, filter events to +/-date_window_days.
            date_window_days: Half-window in days for date filtering.

        Returns:
            List of (chunk_id, score) tuples ordered by descending hit count.
            Score is normalized hit count (hits / max_hits).
        """
        chunk_hits: Counter[int] = Counter()

        for ent in entities:
            key = ent.lower().strip()
            events = self._entity_index.get(key, [])
            for event in events:
                if date_center is not None and event.date is not None:
                    delta = abs((event.date - date_center).days)
                    if delta > date_window_days:
                        continue
                chunk_hits[event.chunk_id] += 1

        if not chunk_hits:
            return []

        max_hits = chunk_hits.most_common(1)[0][1]
        return [
            (cid, hits / max_hits)
            for cid, hits in chunk_hits.most_common(top_k)
        ]

    def lookup_by_type(
        self,
        type_name: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Resolve a type name to entities, then look up those entities.

        E.g. lookup_by_type("museum") finds entities typed as "museum"
        (MoMA, Rijksmuseum, ...) and returns chunk_ids where they appear.
        """
        key = type_name.lower().strip()
        entity_names = self._type_index.get(key, set())
        if not entity_names:
            return []
        return self.lookup(entity_names, top_k=top_k)

    def lookup_by_predicate(
        self,
        predicate: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Look up chunk_ids by predicate name."""
        key = predicate.lower().strip()
        events = self._predicate_index.get(key, [])
        if not events:
            return []
        chunk_hits: Counter[int] = Counter()
        for event in events:
            chunk_hits[event.chunk_id] += 1
        max_hits = chunk_hits.most_common(1)[0][1]
        return [
            (cid, hits / max_hits)
            for cid, hits in chunk_hits.most_common(top_k)
        ]

    def get_facts_for_chunks(
        self,
        chunk_ids: list[int],
    ) -> dict[int, list[str]]:
        """Get human-readable facts for given chunk_ids."""
        result: dict[int, list[str]] = {}
        for cid in chunk_ids:
            events = self._chunk_index.get(cid, [])
            if events:
                result[cid] = [
                    f"{e.subject} {e.predicate} {e.object}" for e in events
                ]
        return result

    def n_events(self) -> int:
        """Count unique events (each event is counted once, not per index entry)."""
        seen: set[int] = set()
        count = 0
        for events in self._entity_index.values():
            for e in events:
                eid = id(e)
                if eid not in seen:
                    seen.add(eid)
                    count += 1
        return count

    def n_entities(self) -> int:
        return len(self._entity_index)

    def __len__(self) -> int:
        return self.n_events()
