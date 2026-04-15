"""Property-based and algebraic tests for Hermes Foundry."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from hypothesis import HealthCheck, given, settings, strategies as st

from foundry.ontology import EntityType, PropertyType
from foundry.query import FilterOp, QueryPlan
from foundry.storage import EntityNode, Fact, FoundryStore, TemporalEvent, TemporalIndex
from foundry.writeback import SupersessionEngine


PROPERTY_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _build_store() -> FoundryStore:
    store = FoundryStore()
    rows = [
        ("luna", "cat", "tabby", "brown", 3),
        ("mochi", "cat", "calico", "white", 4),
        ("roadster", "bike", "", "blue", 1),
    ]
    for idx, (canonical, type_label, breed, color, age) in enumerate(rows):
        entity = store.entity_graph.ensure_entity(canonical)
        entity.aliases.add(canonical)
        entity.register_type(type_label)
        store.types.index(canonical, type_label)
        for predicate, value, value_type in [
            ("color", color, PropertyType.STRING),
            ("age", str(age), PropertyType.NUMERIC),
        ]:
            fact = Fact(
                id=f"{canonical}-{predicate}",
                subject=canonical,
                predicate=predicate,
                object=value,
                object_type=value_type,
                turn_idx=idx,
                chunk_idx=idx,
                source_text=f"{canonical} {predicate} {value}",
            )
            store.entity_graph.add_fact(fact)
            store.properties.index(canonical, predicate, value, value_type)
        if breed:
            breed_fact = Fact(
                id=f"{canonical}-breed",
                subject=canonical,
                predicate="breed",
                object=breed,
                object_type=PropertyType.STRING,
                turn_idx=idx,
                chunk_idx=idx,
                source_text=f"{canonical} breed {breed}",
            )
            store.entity_graph.add_fact(breed_fact)
            store.properties.index(canonical, "breed", breed, PropertyType.STRING)
    return store


def _serialize_plan(plan: QueryPlan) -> dict[str, object]:
    return {
        "seed": None if plan._seed is None else set(plan._seed),
        "filters": plan._filters,
        "traversals": plan._traversals,
        "temporal_filters": plan._temporal_filters,
        "where_related_steps": plan._where_related_steps,
        "order_spec": plan._order_by,
        "limit_value": plan._limit,
        "set_op": plan._set_op,
    }


def _deserialize_plan(store: FoundryStore, state: dict[str, object]) -> QueryPlan:
    return QueryPlan(
        store,
        seed=state["seed"],  # type: ignore[arg-type]
        filters=state["filters"],  # type: ignore[arg-type]
        traversals=state["traversals"],  # type: ignore[arg-type]
        temporal_filters=state["temporal_filters"],  # type: ignore[arg-type]
        where_related_steps=state["where_related_steps"],  # type: ignore[arg-type]
        order_spec=state["order_spec"],  # type: ignore[arg-type]
        limit_value=state["limit_value"],  # type: ignore[arg-type]
        set_op=state["set_op"],  # type: ignore[arg-type]
    )


@PROPERTY_SETTINGS
@given(
    field=st.text(min_size=1, max_size=10),
    operator=st.sampled_from(["$eq", "$contains", "$regex", "$fuzzy"]),
    value=st.text(min_size=0, max_size=20),
    negate=st.booleans(),
)
def test_filterop_double_negation_returns_original(field: str, operator: str, value: str, negate: bool) -> None:
    original = FilterOp(field=field, operator=operator, value=value, negate=negate)
    negated = replace(original, negate=not original.negate)
    round_tripped = replace(negated, negate=not negated.negate)

    assert round_tripped == original


@PROPERTY_SETTINGS
@given(
    type_label=st.text(min_size=0, max_size=40),
)
def test_entitytype_normalize_is_idempotent(type_label: str) -> None:
    once = EntityType.normalize(type_label)
    twice = EntityType.normalize(once)
    assert once == twice


@PROPERTY_SETTINGS
@given(word=st.text(min_size=0, max_size=40))
def test_entitytype_depluralize_is_idempotent(word: str) -> None:
    once = EntityType.depluralize(word)
    twice = EntityType.depluralize(once)
    assert once == twice


def test_queryplan_state_roundtrip_preserves_results() -> None:
    store = _build_store()
    original = (
        QueryPlan(store)
        .type("cat")
        .where("color", "$contains", "w")
        .order_by("age", "desc")
        .limit(1)
    )

    restored = _deserialize_plan(store, _serialize_plan(original))

    assert [entity.canonical for entity in restored.execute()] == [
        entity.canonical for entity in original.execute()
    ]
    assert restored.count() == original.count()


def test_fact_equality_is_reflexive_symmetric_and_transitive() -> None:
    first = Fact(
        id="fact",
        subject="luna",
        predicate="color",
        object="brown",
        object_type=PropertyType.STRING,
        turn_idx=1,
        source_text="Luna is brown",
    )
    second = Fact.from_dict(first.to_dict())
    third = Fact.from_dict(second.to_dict())

    assert first == first
    assert first == second and second == first
    assert first == second and second == third and first == third


def test_duplicate_fact_addition_is_idempotent() -> None:
    engine = SupersessionEngine()
    fact = Fact(
        id="duplicate",
        subject="user",
        predicate="city",
        object="Portland",
        object_type=PropertyType.STRING,
        turn_idx=0,
        source_text="User lives in Portland",
    )

    first = engine.add_fact(fact)
    second = engine.add_fact(Fact.from_dict(fact.to_dict()))

    assert first.id == second.id == "duplicate"
    assert len(engine.get_all_facts("user", "city")) == 1


@PROPERTY_SETTINGS
@given(
    offset=st.integers(min_value=0, max_value=10),
    width=st.integers(min_value=0, max_value=10),
    point_offset=st.integers(min_value=0, max_value=10),
)
def test_temporal_contains_matches_events_between_for_point_queries(
    offset: int,
    width: int,
    point_offset: int,
) -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    event = TemporalEvent(
        event="window",
        start_dt=base + timedelta(days=offset),
        end_dt=base + timedelta(days=offset + width),
        source="source",
    )
    point = base + timedelta(days=point_offset)

    index = TemporalIndex()
    index.ingest(event)

    contains = index.contains(event, point)
    between = event in index.events_between(point, point)

    assert contains == between


def test_plan_roundtrip_with_seed_preserves_execution() -> None:
    store = _build_store()
    seeded = QueryPlan(store, seed={"luna", "mochi"}).where("breed", "$matchAnyToken", "tabby calico")
    restored = _deserialize_plan(store, _serialize_plan(seeded))

    assert {entity.canonical for entity in seeded.execute()} == {
        entity.canonical for entity in restored.execute()
    }


def test_manual_entitygraph_add_then_remove_restores_empty_store() -> None:
    store = FoundryStore()
    baseline = store.entity_graph.to_dict()
    entity = EntityNode(canonical="temp", aliases={"temporary"}, types={"cat"})
    store.entity_graph.add_entity(entity)

    store.entity_graph._entities.pop("temp", None)
    store.entity_graph._alias_to_canonical = {}

    assert store.entity_graph.to_dict() == baseline
