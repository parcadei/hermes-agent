"""Lazy query planning primitives for the Hermes Foundry memory system."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import base64
import json
import re
from typing import Any, Iterator, Optional, Union
from urllib import error as urllib_error
from urllib import request as urllib_request
import warnings

from foundry.ontology import EntityType, PropertyType
from foundry.storage import (
    EntityNode,
    Fact,
    FoundryStore,
    TemporalEvent,
    _normalize_name,
    _normalize_predicate,
    _parse_datetime,
    _parse_numeric,
    _tokenize,
)


MAX_HOPS_HARD_LIMIT = 3
DEFAULT_EXECUTE_LIMIT = 100_000
MAX_PAGE_SIZE = 10_000
_MIN_DATETIME = datetime.min
_MAX_DATETIME = datetime.max


@dataclass(frozen=True)
class FilterOp:
    """A single filter predicate in a query chain."""

    field: str
    operator: str
    value: Any
    negate: bool = False


@dataclass(frozen=True)
class LogicalOp:
    """Boolean composition of filters."""

    op: str
    children: tuple[Union[FilterOp, "LogicalOp"], ...]


@dataclass(frozen=True)
class TraversalOp:
    """Link traversal specification."""

    link_predicate: str
    direction: str
    max_hops: int = 1
    filter_target: Optional[FilterOp] = None


@dataclass
class PageResult:
    """A single page of query results with continuation token."""

    data: list[EntityNode]
    next_page_token: Optional[str]
    total_count: Optional[int] = None


@dataclass(frozen=True)
class _TemporalFilter:
    kind: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None


@dataclass(frozen=True)
class _TraversalStep:
    mode: str
    op: TraversalOp


@dataclass(frozen=True)
class _WhereRelatedStep:
    predicate: str
    target_filter: Union[FilterOp, LogicalOp]


@dataclass(frozen=True)
class _SetOp:
    kind: str
    left: "QueryPlan"
    right: "QueryPlan"


def _encode_page_token(offset: int) -> str:
    payload = json.dumps({"offset": offset}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


def _decode_page_token(token: Optional[str]) -> int:
    if not token:
        return 0
    payload = base64.urlsafe_b64decode(token.encode("ascii"))
    data = json.loads(payload.decode("utf-8"))
    return max(0, int(data.get("offset", 0)))


def _normalize_direction(direction: str) -> str:
    normalized = str(direction or "both").strip().lower()
    if normalized not in {"forward", "backward", "both"}:
        raise ValueError(f"Unsupported traversal direction: {direction}")
    return normalized


def _normalize_filter_count(filters: tuple[Union[FilterOp, LogicalOp], ...]) -> None:
    if len(filters) > 10:
        raise ValueError("QueryPlan refuses more than 10 chained filters")
    if len(filters) > 5:
        warnings.warn("QueryPlan has more than 5 filters; execution may be slow", RuntimeWarning, stacklevel=3)


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _auto_fuzziness(token: str) -> int:
    length = len(token)
    if length <= 2:
        return 0
    if length <= 5:
        return 1
    return 2


class GroupedQueryPlan:
    """A query plan with one or more grouping dimensions."""

    def __init__(self, plan: "QueryPlan", dimensions: tuple[str, ...]) -> None:
        self._plan = plan
        self._dimensions = tuple(_normalize_predicate(dimension) for dimension in dimensions)

    def group_by(self, property_predicate: str) -> "GroupedQueryPlan":
        return GroupedQueryPlan(self._plan, self._dimensions + (_normalize_predicate(property_predicate),))

    def count(self) -> dict[str, Any]:
        return self._aggregate("count", "")

    def sum(self, property_predicate: str) -> dict[str, Any]:
        return self._aggregate("sum", property_predicate)

    def avg(self, property_predicate: str) -> dict[str, Any]:
        return self._aggregate("avg", property_predicate)

    def list(self, property_predicate: str) -> dict[str, Any]:
        return self._aggregate("list", property_predicate)

    def _aggregate(self, op: str, property_predicate: str) -> dict[str, Any]:
        entities = self._plan.execute()
        groups: dict[tuple[str, ...], list[EntityNode]] = {}
        for entity in entities:
            key_paths = self._entity_group_keys(entity)
            for key in key_paths:
                groups.setdefault(key, []).append(entity)

        result: dict[str, Any] = {}
        for key, members in groups.items():
            current = result
            for part in key[:-1]:
                current = current.setdefault(part, {})
            current[key[-1]] = self._apply_group_aggregate(members, op, property_predicate)
        return result

    def _entity_group_keys(self, entity: EntityNode) -> list[tuple[str, ...]]:
        key_parts: list[list[str]] = []
        for dimension in self._dimensions:
            values = _entity_values_for_field(entity, dimension, current_only=True)
            key_parts.append(values or ["__missing__"])

        keys: list[tuple[str, ...]] = [tuple()]
        for dimension_values in key_parts:
            next_keys: list[tuple[str, ...]] = []
            for prefix in keys:
                for value in dimension_values:
                    next_keys.append(prefix + (value,))
            keys = next_keys
        return keys

    def _apply_group_aggregate(self, members: list[EntityNode], op: str, property_predicate: str) -> Any:
        if op == "count":
            return len(members)
        values = []
        for entity in members:
            values.extend(_entity_values_for_field(entity, property_predicate, current_only=True))
        if op == "list":
            return values
        numerics = [number for number in (_parse_numeric(value) for value in values) if number is not None]
        if op == "sum":
            return float(sum(numerics))
        if op == "avg":
            return float(sum(numerics) / len(numerics)) if numerics else 0.0
        raise ValueError(f"Unsupported grouped aggregate: {op}")


class TemporalQueryPlan:
    """Lazy, immutable, composable temporal query."""

    def __init__(
        self,
        store: FoundryStore,
        query: str = "",
        filters: tuple[tuple[str, Any], ...] = (),
    ) -> None:
        self._store = store
        self._query = query
        self._filters = filters
        self._result_cache: Optional[list[TemporalEvent]] = None

    def _clone(self, *, query: Optional[str] = None, extra_filter: Optional[tuple[str, Any]] = None) -> "TemporalQueryPlan":
        filters = self._filters
        if extra_filter is not None:
            filters = filters + (extra_filter,)
        return TemporalQueryPlan(self._store, self._query if query is None else query, filters)

    def match(self, query: str) -> "TemporalQueryPlan":
        return self._clone(query=query)

    def before(self, ref: TemporalEvent | datetime) -> "TemporalQueryPlan":
        return self._clone(extra_filter=("before", ref))

    def after(self, ref: TemporalEvent | datetime) -> "TemporalQueryPlan":
        return self._clone(extra_filter=("after", ref))

    def overlapping(self, interval: TemporalEvent) -> "TemporalQueryPlan":
        return self._clone(extra_filter=("overlapping", interval))

    def during(self, interval: TemporalEvent) -> "TemporalQueryPlan":
        return self._clone(extra_filter=("during", interval))

    def execute(self) -> list[TemporalEvent]:
        if self._result_cache is not None:
            return list(self._result_cache)
        if self._query:
            events = list(self._store.temporal.match_events(self._query, top_k=max(5, len(self._store.temporal._events))))
        else:
            events = list(self._store.temporal._events)
        for kind, reference in self._filters:
            if kind == "before":
                events = [event for event in events if event.end_dt < _event_point(reference, "start")]
            elif kind == "after":
                events = [event for event in events if event.start_dt > _event_point(reference, "end")]
            elif kind == "overlapping":
                interval = reference
                events = [
                    event
                    for event in events
                    if event.start_dt <= interval.end_dt and event.end_dt >= interval.start_dt
                ]
            elif kind == "during":
                interval = reference
                events = [
                    event
                    for event in events
                    if interval.start_dt <= event.start_dt and event.end_dt <= interval.end_dt
                ]
        self._result_cache = sorted(events, key=lambda event: (event.start_dt, event.end_dt, event.event))
        return list(self._result_cache)

    def to_entity_plan(self) -> "QueryPlan":
        seeds: set[str] = set()
        for event in self.execute():
            for name in (event.subject, event.object):
                if name:
                    seeds.add(_normalize_name(name))
        return QueryPlan(self._store, seed=seeds)


class QueryPlan:
    """Lazy, immutable, chainable query plan over entities."""

    def __init__(
        self,
        store: FoundryStore,
        seed: Optional[set[str]] = None,
        *,
        filters: tuple[Union[FilterOp, LogicalOp], ...] = (),
        traversals: tuple[_TraversalStep, ...] = (),
        temporal_filters: tuple[_TemporalFilter, ...] = (),
        where_related_steps: tuple[_WhereRelatedStep, ...] = (),
        order_spec: Optional[tuple[str, str]] = None,
        limit_value: Optional[int] = None,
        set_op: Optional[_SetOp] = None,
    ) -> None:
        self._store = store
        self._seed: Optional[set[str]] = None if seed is None else {_normalize_name(value) for value in seed}
        self._filters = filters
        self._traversals = traversals
        self._temporal_filters = temporal_filters
        self._where_related_steps = where_related_steps
        self._order_by = order_spec
        self._limit = limit_value
        self._set_op = set_op
        self._result_cache: Optional[list[EntityNode]] = None

    def _clone(
        self,
        *,
        seed: Optional[set[str]] | object = None,
        append_filter: Optional[Union[FilterOp, LogicalOp]] = None,
        append_traversal: Optional[_TraversalStep] = None,
        append_temporal: Optional[_TemporalFilter] = None,
        append_where_related: Optional[_WhereRelatedStep] = None,
        order_spec: Optional[tuple[str, str]] | object = None,
        limit_value: Optional[int] | object = None,
        set_op: Optional[_SetOp] | object = None,
    ) -> "QueryPlan":
        filters = self._filters if append_filter is None else self._filters + (append_filter,)
        _normalize_filter_count(filters)
        return QueryPlan(
            self._store,
            self._seed if seed is None else seed,  # type: ignore[arg-type]
            filters=filters,
            traversals=self._traversals if append_traversal is None else self._traversals + (append_traversal,),
            temporal_filters=(
                self._temporal_filters if append_temporal is None else self._temporal_filters + (append_temporal,)
            ),
            where_related_steps=(
                self._where_related_steps
                if append_where_related is None
                else self._where_related_steps + (append_where_related,)
            ),
            order_spec=self._order_by if order_spec is None else order_spec,  # type: ignore[arg-type]
            limit_value=self._limit if limit_value is None else limit_value,  # type: ignore[arg-type]
            set_op=self._set_op if set_op is None else set_op,  # type: ignore[arg-type]
        )

    def type(self, type_label: str) -> "QueryPlan":
        return self.filter(FilterOp(field="type", operator="$eq", value=EntityType.normalize(type_label)))

    def types(self, type_labels: list[str]) -> "QueryPlan":
        ops = tuple(
            FilterOp(field="type", operator="$eq", value=EntityType.normalize(type_label))
            for type_label in type_labels
        )
        return self.or_(*ops)

    def where(self, predicate: str, operator: str, value: Any) -> "QueryPlan":
        return self.filter(FilterOp(field=predicate, operator=operator, value=value))

    def filter(self, filter_op: Union[FilterOp, LogicalOp]) -> "QueryPlan":
        return self._clone(append_filter=filter_op)

    def and_(self, *ops: Union[FilterOp, LogicalOp]) -> "QueryPlan":
        return self.filter(LogicalOp(op="$and", children=tuple(ops)))

    def or_(self, *ops: Union[FilterOp, LogicalOp]) -> "QueryPlan":
        return self.filter(LogicalOp(op="$or", children=tuple(ops)))

    def not_(self, op: Union[FilterOp, LogicalOp]) -> "QueryPlan":
        return self.filter(LogicalOp(op="$not", children=(op,)))

    def related(self, link_predicate: str = "", direction: str = "both", max_hops: int = 1) -> "QueryPlan":
        traversal = TraversalOp(
            link_predicate=_normalize_predicate(link_predicate) if link_predicate else "",
            direction=_normalize_direction(direction),
            max_hops=max(1, min(int(max_hops), MAX_HOPS_HARD_LIMIT)),
        )
        return self._clone(append_traversal=_TraversalStep("expand", traversal))

    def pivot_to(self, link_predicate: str, direction: str = "forward", max_hops: int = 1) -> "QueryPlan":
        traversal = TraversalOp(
            link_predicate=_normalize_predicate(link_predicate) if link_predicate else "",
            direction=_normalize_direction(direction),
            max_hops=max(1, min(int(max_hops), MAX_HOPS_HARD_LIMIT)),
        )
        return self._clone(append_traversal=_TraversalStep("pivot", traversal))

    def search_around(self, predicate: str = "", max_hops: int = 1, target_type: str = "") -> "QueryPlan":
        plan = self.pivot_to(predicate, direction="both", max_hops=max_hops)
        if target_type:
            plan = plan.type(target_type)
        return plan

    def where_related(self, predicate: str, target_filter: Union[FilterOp, LogicalOp]) -> "QueryPlan":
        step = _WhereRelatedStep(predicate=_normalize_predicate(predicate), target_filter=target_filter)
        return self._clone(append_where_related=step)

    def before(self, date: datetime) -> "QueryPlan":
        return self._clone(append_temporal=_TemporalFilter("before", end=_parse_datetime(date)))

    def after(self, date: datetime) -> "QueryPlan":
        return self._clone(append_temporal=_TemporalFilter("after", start=_parse_datetime(date)))

    def during(self, start: datetime, end: datetime) -> "QueryPlan":
        return self._clone(
            append_temporal=_TemporalFilter("during", start=_parse_datetime(start), end=_parse_datetime(end))
        )

    def overlaps(self, start: datetime, end: datetime) -> "QueryPlan":
        return self._clone(
            append_temporal=_TemporalFilter("overlaps", start=_parse_datetime(start), end=_parse_datetime(end))
        )

    def order_by(self, field: str, direction: str = "asc") -> "QueryPlan":
        normalized_direction = str(direction).lower()
        if normalized_direction not in {"asc", "desc"}:
            raise ValueError(f"Unsupported order direction: {direction}")
        return self._clone(order_spec=(field, normalized_direction))

    def limit(self, n: int) -> "QueryPlan":
        return self._clone(limit_value=max(0, int(n)))

    def union(self, other: "QueryPlan") -> "QueryPlan":
        return QueryPlan(self._store, set_op=_SetOp("union", self, other))

    def intersection(self, other: "QueryPlan") -> "QueryPlan":
        return QueryPlan(self._store, set_op=_SetOp("intersection", self, other))

    def difference(self, other: "QueryPlan") -> "QueryPlan":
        return QueryPlan(self._store, set_op=_SetOp("difference", self, other))

    def group_by(self, property_predicate: str) -> GroupedQueryPlan:
        return GroupedQueryPlan(self, (_normalize_predicate(property_predicate),))

    def page(self, page_size: int = 100, page_token: Optional[str] = None) -> PageResult:
        size = max(1, min(int(page_size), MAX_PAGE_SIZE))
        offset = _decode_page_token(page_token)
        rows = self.execute()
        data = rows[offset : offset + size]
        next_offset = offset + len(data)
        next_page_token = _encode_page_token(next_offset) if next_offset < len(rows) else None
        return PageResult(data=data, next_page_token=next_page_token, total_count=len(rows))

    def take(self, n: int) -> list[EntityNode]:
        return self.limit(n).execute()

    def iter(self) -> Iterator[EntityNode]:
        page_token: Optional[str] = None
        while True:
            page = self.page(page_size=100, page_token=page_token)
            for entity in page.data:
                yield entity
            if page.next_page_token is None:
                break
            page_token = page.next_page_token

    def execute(self) -> list[EntityNode]:
        if self._result_cache is not None:
            return list(self._result_cache)
        canonicals = self._evaluate_canonicals()
        nodes = [self._store.entity_graph.get_entity(canonical) for canonical in canonicals]
        entities = [entity for entity in nodes if entity is not None]
        entities = self._apply_ordering(entities)
        effective_limit = DEFAULT_EXECUTE_LIMIT if self._limit is None else min(self._limit, DEFAULT_EXECUTE_LIMIT)
        self._result_cache = entities[:effective_limit]
        return list(self._result_cache)

    def count(self) -> int:
        return len(self.execute())

    def first(self) -> Optional[EntityNode]:
        rows = self.limit(1).execute()
        return rows[0] if rows else None

    def aggregate(self, property_predicate: str, op: str) -> Union[float, int, list]:
        entities = self.execute()
        normalized_op = str(op).lower()
        if normalized_op == "count":
            return len(entities)
        values: list[str] = []
        for entity in entities:
            values.extend(_entity_values_for_field(entity, property_predicate, current_only=True))
        if normalized_op == "list":
            return values
        if normalized_op == "distinct":
            return sorted(set(values))
        numerics = [number for number in (_parse_numeric(value) for value in values) if number is not None]
        if normalized_op == "sum":
            return float(sum(numerics))
        if normalized_op == "avg":
            return float(sum(numerics) / len(numerics)) if numerics else 0.0
        if normalized_op == "min":
            return min(numerics) if numerics else 0.0
        if normalized_op == "max":
            return max(numerics) if numerics else 0.0
        raise ValueError(f"Unsupported aggregate op: {op}")

    def facts(self, predicate: str = "", current_only: bool = True) -> list[Fact]:
        normalized_predicate = _normalize_predicate(predicate) if predicate else ""
        rows: list[Fact] = []
        for entity in self.execute():
            predicates = [normalized_predicate] if normalized_predicate else list(entity.facts)
            for pred in predicates:
                for fact in entity.facts.get(pred, []):
                    if current_only and fact.superseded:
                        continue
                    rows.append(fact)
        rows.sort(key=lambda fact: (-fact.turn_idx, -fact.chunk_idx, fact.id))
        return rows

    def evidence_turns(self) -> list[int]:
        counter: Counter[int] = Counter()
        for entity in self.execute():
            for turn_idx in entity.mention_turns:
                counter[turn_idx] += 1
        return [turn_idx for turn_idx, _count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]

    def _evaluate_canonicals(self) -> list[str]:
        if self._set_op is None:
            canonicals = self._initial_canonicals()
        else:
            canonicals = self._evaluate_set_op()

        for temporal_filter in self._temporal_filters:
            canonicals = self._apply_temporal_filter(canonicals, temporal_filter)

        for filter_op in self._filters:
            canonicals = self._apply_filter(canonicals, filter_op)

        for where_related_step in self._where_related_steps:
            canonicals = self._apply_where_related(canonicals, where_related_step)

        for traversal in self._traversals:
            canonicals = self._apply_traversal(canonicals, traversal)

        return canonicals

    def _initial_canonicals(self) -> list[str]:
        if self._seed is not None:
            return sorted(self._seed)
        return [entity.canonical for entity in self._store.entity_graph.all_entities()]

    def _evaluate_set_op(self) -> list[str]:
        assert self._set_op is not None
        left = set(self._set_op.left._evaluate_canonicals())
        right = set(self._set_op.right._evaluate_canonicals())
        if self._set_op.kind == "union":
            return sorted(left | right)
        if self._set_op.kind == "intersection":
            return sorted(left & right)
        if self._set_op.kind == "difference":
            return sorted(left - right)
        raise ValueError(f"Unsupported set op: {self._set_op.kind}")

    def _apply_filter(self, canonicals: list[str], op: Union[FilterOp, LogicalOp]) -> list[str]:
        return [canonical for canonical in canonicals if self._entity_matches_op(canonical, op)]

    def _entity_matches_op(self, canonical: str, op: Union[FilterOp, LogicalOp]) -> bool:
        entity = self._store.entity_graph.get_entity(canonical)
        if entity is None:
            return False
        if isinstance(op, LogicalOp):
            if op.op == "$and":
                return all(self._entity_matches_op(canonical, child) for child in op.children)
            if op.op == "$or":
                return any(self._entity_matches_op(canonical, child) for child in op.children)
            if op.op == "$not":
                return not any(self._entity_matches_op(canonical, child) for child in op.children)
            raise ValueError(f"Unsupported logical op: {op.op}")
        result = _filter_matches_entity(entity, op)
        return not result if op.negate else result

    def _apply_temporal_filter(self, canonicals: list[str], temporal_filter: _TemporalFilter) -> list[str]:
        if temporal_filter.kind in {"during", "overlaps"}:
            events = self._store.temporal.events_between(
                temporal_filter.start or _MIN_DATETIME,
                temporal_filter.end or _MAX_DATETIME,
            )
        elif temporal_filter.kind == "before":
            events = self._store.temporal.events_between(_MIN_DATETIME, temporal_filter.end or _MAX_DATETIME)
        elif temporal_filter.kind == "after":
            events = self._store.temporal.events_between(temporal_filter.start or _MIN_DATETIME, _MAX_DATETIME)
        else:
            events = []

        event_entities: set[str] = set()
        for event in events:
            if temporal_filter.kind == "before" and temporal_filter.end is not None and not event.end_dt < temporal_filter.end:
                continue
            if temporal_filter.kind == "after" and temporal_filter.start is not None and not event.start_dt > temporal_filter.start:
                continue
            for value in (event.subject, event.object):
                if value:
                    event_entities.add(_normalize_name(value))

        matched: list[str] = []
        for canonical in canonicals:
            entity = self._store.entity_graph.get_entity(canonical)
            if entity is None:
                continue
            if canonical in event_entities or _entity_matches_temporal_fact(entity, temporal_filter):
                matched.append(canonical)
        return matched

    def _apply_where_related(self, canonicals: list[str], step: _WhereRelatedStep) -> list[str]:
        matched: list[str] = []
        for canonical in canonicals:
            entity = self._store.entity_graph.get_entity(canonical)
            if entity is None:
                continue
            targets = entity.links.get(step.predicate, set())
            if any(self._entity_matches_op(target, step.target_filter) for target in targets):
                matched.append(canonical)
        return matched

    def _apply_traversal(self, canonicals: list[str], traversal: _TraversalStep) -> list[str]:
        current = {_normalize_name(canonical) for canonical in canonicals}
        frontier = set(current)
        discovered = set(current if traversal.mode == "expand" else [])

        for _ in range(traversal.op.max_hops):
            next_frontier: set[str] = set()
            for canonical in frontier:
                next_frontier.update(self._neighbor_canonicals(canonical, traversal.op))
            next_frontier -= discovered
            if traversal.op.filter_target is not None:
                next_frontier = {
                    canonical for canonical in next_frontier if self._entity_matches_op(canonical, traversal.op.filter_target)
                }
            if not next_frontier:
                break
            discovered.update(next_frontier)
            frontier = next_frontier

        if traversal.mode == "pivot":
            result = frontier if traversal.op.max_hops == 1 else discovered
        else:
            result = discovered
        return sorted(result)

    def _neighbor_canonicals(self, canonical: str, op: TraversalOp) -> set[str]:
        entity = self._store.entity_graph.get_entity(canonical)
        if entity is None:
            return set()
        neighbors: set[str] = set()
        predicate = op.link_predicate
        if op.direction in {"forward", "both"}:
            if predicate:
                neighbors.update(entity.links.get(predicate, set()))
            else:
                for targets in entity.links.values():
                    neighbors.update(targets)
        if op.direction in {"backward", "both"}:
            for other in self._store.entity_graph.all_entities():
                if other.canonical == canonical:
                    continue
                if predicate:
                    if canonical in other.links.get(predicate, set()):
                        neighbors.add(other.canonical)
                else:
                    if any(canonical in targets for targets in other.links.values()):
                        neighbors.add(other.canonical)
        return neighbors

    def _apply_ordering(self, entities: list[EntityNode]) -> list[EntityNode]:
        if self._order_by is None:
            return sorted(entities, key=lambda entity: entity.canonical)
        field, direction = self._order_by
        reverse = direction == "desc"
        return sorted(entities, key=lambda entity: _sortable_field_value(entity, field), reverse=reverse)


def _entity_matches_temporal_fact(entity: EntityNode, temporal_filter: _TemporalFilter) -> bool:
    for facts in entity.facts.values():
        for fact in facts:
            point = fact.timestamp or fact.session_time
            if point is None:
                continue
            if temporal_filter.kind == "before" and temporal_filter.end is not None and point < temporal_filter.end:
                return True
            if temporal_filter.kind == "after" and temporal_filter.start is not None and point > temporal_filter.start:
                return True
            if temporal_filter.kind in {"during", "overlaps"}:
                start = temporal_filter.start or _MIN_DATETIME
                end = temporal_filter.end or _MAX_DATETIME
                if start <= point <= end:
                    return True
    return False


def _entity_values_for_field(entity: EntityNode, field: str, current_only: bool = True) -> list[str]:
    normalized_field = _normalize_predicate(field)
    if normalized_field in {"name", "canonical"}:
        return [entity.canonical]
    if normalized_field == "type":
        return sorted(entity.types)
    facts = entity.facts.get(normalized_field, [])
    values = [fact.object for fact in facts if not current_only or not fact.superseded]
    return values


def _filter_matches_entity(entity: EntityNode, filter_op: FilterOp) -> bool:
    field = _normalize_predicate(filter_op.field)
    operator = filter_op.operator
    value = filter_op.value

    if field == "type":
        target_type = EntityType.normalize(str(value))
        has_type = target_type in entity.types
        if operator == "$eq":
            return has_type
        if operator == "$ne":
            return not has_type
        if operator in {"$contains", "$startsWith", "$regex", "$phrase", "$matchAnyToken", "$matchAllTokens", "$fuzzy"}:
            return any(_match_string(candidate, operator, value) for candidate in entity.types)
        if operator == "$isNull":
            return not entity.types
        if operator == "$hasProperty":
            return bool(entity.types)
        return False

    values = _entity_values_for_field(entity, field, current_only=True)
    if operator == "$isNull":
        return not values
    if operator == "$hasProperty":
        return bool(values)
    if not values:
        return False

    return any(_match_value(candidate, operator, value) for candidate in values)


def _match_value(candidate: str, operator: str, expected: Any) -> bool:
    if operator in {"$lt", "$lte", "$gt", "$gte", "$range"}:
        return _match_numeric(candidate, operator, expected)
    if operator in {"$before", "$after", "$between"}:
        return _match_date(candidate, operator, expected)
    if operator in {"$isTrue", "$isFalse"}:
        return _match_boolean(candidate, operator)
    return _match_string(candidate, operator, expected)


def _match_numeric(candidate: str, operator: str, expected: Any) -> bool:
    numeric = _parse_numeric(candidate)
    if numeric is None:
        return False
    if operator == "$range":
        if not isinstance(expected, (tuple, list)) or len(expected) != 2:
            return False
        lower = _parse_numeric(expected[0])
        upper = _parse_numeric(expected[1])
        if lower is None or upper is None:
            return False
        return lower <= numeric <= upper
    target = _parse_numeric(expected)
    if target is None:
        return False
    if operator == "$lt":
        return numeric < target
    if operator == "$lte":
        return numeric <= target
    if operator == "$gt":
        return numeric > target
    if operator == "$gte":
        return numeric >= target
    if operator == "$eq":
        return numeric == target
    if operator == "$ne":
        return numeric != target
    return False


def _match_date(candidate: str, operator: str, expected: Any) -> bool:
    point = _parse_datetime(candidate)
    if point is None:
        return False
    if operator == "$between":
        if not isinstance(expected, (tuple, list)) or len(expected) != 2:
            return False
        start = _parse_datetime(expected[0])
        end = _parse_datetime(expected[1])
        if start is None or end is None:
            return False
        return start <= point <= end
    target = _parse_datetime(expected)
    if target is None:
        return False
    if operator == "$before":
        return point < target
    if operator == "$after":
        return point > target
    return False


def _match_boolean(candidate: str, operator: str) -> bool:
    normalized = str(candidate).strip().lower()
    truthy = normalized in {"true", "yes", "1"}
    falsy = normalized in {"false", "no", "0"}
    if operator == "$isTrue":
        return truthy
    if operator == "$isFalse":
        return falsy
    return False


def _match_string(candidate: str, operator: str, expected: Any) -> bool:
    candidate_text = str(candidate)
    expected_text = "" if expected is None else str(expected)
    left = candidate_text.casefold()
    right = expected_text.casefold()

    if operator == "$eq":
        return left == right
    if operator == "$ne":
        return left != right
    if operator == "$contains":
        return right in left
    if operator == "$startsWith":
        return left.startswith(right)
    if operator == "$regex":
        try:
            return re.search(expected_text, candidate_text, flags=re.IGNORECASE) is not None
        except re.error:
            return False
    if operator == "$phrase":
        return " ".join(_tokenize(expected_text)) in " ".join(_tokenize(candidate_text))
    if operator == "$matchAnyToken":
        candidate_tokens = set(_tokenize(candidate_text))
        return any(token in candidate_tokens for token in _tokenize(expected_text))
    if operator == "$matchAllTokens":
        candidate_tokens = set(_tokenize(candidate_text))
        wanted = set(_tokenize(expected_text))
        return bool(wanted) and wanted.issubset(candidate_tokens)
    if operator == "$fuzzy":
        wanted_tokens = _tokenize(expected_text) or [right]
        candidate_tokens = _tokenize(candidate_text) or [left]
        for wanted in wanted_tokens:
            limit = _auto_fuzziness(wanted)
            if any(_levenshtein_distance(wanted, candidate_token) <= limit for candidate_token in candidate_tokens):
                return True
        return False
    return False


def _sortable_field_value(entity: EntityNode, field: str) -> tuple[int, Any]:
    values = _entity_values_for_field(entity, field, current_only=True)
    if not values:
        return (1, "")
    numeric = _parse_numeric(values[0])
    if numeric is not None:
        return (0, numeric)
    point = _parse_datetime(values[0])
    if point is not None:
        return (0, point)
    return (0, values[0].casefold())


def _event_point(reference: TemporalEvent | datetime, boundary: str) -> datetime:
    if isinstance(reference, TemporalEvent):
        return reference.start_dt if boundary == "start" else reference.end_dt
    parsed = _parse_datetime(reference)
    if parsed is None:
        raise ValueError("Temporal reference must be a datetime or TemporalEvent")
    return parsed


_SUBQUERY_PREFIX_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s*(?P<body>.+?)\s*$")
_ANSWER_PLACEHOLDER_RE = re.compile(r"\{answer_(\d+)\}")


def _parse_subqueries(text: str, max_hops: int = 3) -> list[str]:
    """Parse numbered sub-queries from raw LLM text."""

    subqueries: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _SUBQUERY_PREFIX_RE.match(line)
        if match is None:
            continue
        line = match.group("body").strip()
        placeholder_refs = [int(ref) for ref in _ANSWER_PLACEHOLDER_RE.findall(line)]
        current_position = len(subqueries) + 1
        if any(ref <= 0 or ref >= current_position for ref in placeholder_refs):
            continue
        if line:
            subqueries.append(line)
        if len(subqueries) >= max(1, int(max_hops)):
            break
    return subqueries


def decompose_query(
    question: str,
    llm_url: str,
    llm_model: str = "gpt-4o-mini",
    max_hops: int = 3,
    temperature: float = 0.0,
) -> list[str]:
    """Decompose a question into chained sub-queries via an OpenAI-compatible API."""

    original_question = str(question or "")
    if not original_question.strip() or not str(llm_url or "").strip():
        return [original_question]

    prompt = (
        "Break this question into simple sub-questions that each require looking up a single fact. "
        "Start from the innermost entity and work outward. Use {answer_1}, {answer_2}, etc. as "
        "placeholders for prior answers. Use the actual entity names from the question, not pronouns.\n\n"
        "Example:\n"
        "Q: What is the country of origin of the sport played by Christian Abbiati?\n"
        "1. What sport did Christian Abbiati play?\n"
        "2. What is the country of origin of {answer_1}?\n\n"
        "Example:\n"
        "Q: Where was the creator of the religion Karl Lueger followed born?\n"
        "1. What religion did Karl Lueger follow?\n"
        "2. Who created {answer_1}?\n"
        "3. Where was {answer_2} born?\n\n"
        f"Q: {original_question}\n"
    )

    payload = json.dumps(
        {
            "model": llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You decompose multi-hop questions into chained sub-questions. "
                        "Use {answer_N} placeholders. Be concise."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": 256,
        }
    ).encode("utf-8")
    request = urllib_request.Request(
        f"{str(llm_url).rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        subqueries = _parse_subqueries(str(content or ""), max_hops=max_hops)
        return subqueries if subqueries else [original_question]
    except (OSError, ValueError, KeyError, IndexError, TypeError, urllib_error.URLError):
        return [original_question]


__all__ = [
    "FilterOp",
    "GroupedQueryPlan",
    "LogicalOp",
    "PageResult",
    "QueryPlan",
    "TemporalQueryPlan",
    "TraversalOp",
    "_parse_subqueries",
    "decompose_query",
]
