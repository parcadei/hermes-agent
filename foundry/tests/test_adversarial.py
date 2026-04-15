"""Adversarial input tests for Hermes Foundry."""

from __future__ import annotations

from datetime import datetime

from foundry.ingestion import EntityResolver, parse_conversation
from foundry.ontology import PropertyType, infer_property_type
from foundry.query import FilterOp, LogicalOp, QueryPlan
from foundry.storage import BM25Index, EntityNode, Fact, FoundryStore, TemporalEvent
from foundry.writeback import SupersessionEngine


def _minimal_store() -> FoundryStore:
    store = FoundryStore()
    for idx, name in enumerate(["alpha", "beta"]):
        entity = store.entity_graph.ensure_entity(name)
        entity.aliases.add(name)
        entity.register_type("thing")
        store.types.index(name, "thing")
        fact = Fact(
            id=f"{name}-fact",
            subject=name,
            predicate="label",
            object=name,
            object_type=PropertyType.STRING,
            turn_idx=idx,
            chunk_idx=idx,
            source_text=name,
        )
        store.entity_graph.add_fact(fact)
        store.properties.index(name, "label", name, PropertyType.STRING)
    return store


def test_entity_names_with_sql_injection_patterns_are_handled_deterministically() -> None:
    resolver = EntityResolver()
    canonical = resolver.resolve_or_create("'; DROP TABLE users; --")

    assert canonical
    assert resolver.resolve("'; DROP TABLE users; --") == canonical


def test_entity_names_with_path_traversal_patterns_are_handled_deterministically() -> None:
    resolver = EntityResolver()
    canonical = resolver.resolve_or_create("../../etc/passwd")

    assert canonical
    assert "/" not in canonical


def test_extremely_long_fact_values_are_storable_and_retrievable() -> None:
    huge_value = "x" * 100_000
    fact = Fact(
        id="huge",
        subject="entity",
        predicate="payload",
        object=huge_value,
        object_type=PropertyType.STRING,
        turn_idx=0,
        source_text=huge_value,
    )
    store = FoundryStore()
    store.entity_graph.add_fact(fact)
    store.properties.index("entity", "payload", huge_value, PropertyType.STRING)

    facts = store.entity_graph.ensure_entity("entity").facts["payload"]
    assert facts[0].object == huge_value


def test_large_nested_logical_query_does_not_overflow() -> None:
    store = _minimal_store()
    op: FilterOp | LogicalOp = FilterOp("label", "$contains", "alpha")
    for _ in range(200):
        op = LogicalOp("$or", (op, FilterOp("label", "$contains", "beta")))

    result = QueryPlan(store).filter(op).execute()
    assert {entity.canonical for entity in result} == {"alpha", "beta"}


def test_homoglyphs_are_handled_without_false_aliasing() -> None:
    resolver = EntityResolver()
    latin = resolver.resolve_or_create("a")
    cyrillic = resolver.resolve_or_create("а")  # Cyrillic small a

    assert latin != cyrillic
    assert resolver.resolve("a") == latin
    assert resolver.resolve("а") == cyrillic


def test_zero_width_and_null_bytes_are_normalized_safely() -> None:
    resolver = EntityResolver()
    zero_width = resolver.resolve_or_create("Lu\u200bna")
    null_byte = resolver.resolve_or_create("Lu\0na")

    assert zero_width
    assert null_byte
    assert "\x00" not in null_byte


def test_conversation_with_ten_thousand_turns_parses_without_blowing_up() -> None:
    raw = "\n".join(f"User: turn {idx}" for idx in range(10_000))
    turns = parse_conversation(raw)

    assert len(turns) == 10_000
    assert turns[-1].turn_idx == 9_999


def test_temporal_events_handle_far_future_and_epoch_dates() -> None:
    future = TemporalEvent("future", datetime(9999, 1, 1), datetime(9999, 1, 1), "source")
    epoch = TemporalEvent("epoch", datetime(1970, 1, 1), datetime(1970, 1, 1), "source")

    assert future.start_dt.year == 9999
    assert epoch.start_dt.year == 1970


def test_negative_quantities_are_treated_as_numeric() -> None:
    assert infer_property_type("-42") == PropertyType.NUMERIC


def test_empty_subject_predicate_and_object_are_handled() -> None:
    fact = Fact(
        id="empty",
        subject="",
        predicate="",
        object="",
        object_type=PropertyType.STRING,
        turn_idx=0,
        source_text="",
    )

    assert fact.subject == ""
    assert fact.predicate == ""
    assert fact.object == ""


def test_duplicate_facts_are_idempotent_in_supersession_storage() -> None:
    engine = SupersessionEngine()
    fact = Fact(
        id="dup",
        subject="entity",
        predicate="status",
        object="active",
        object_type=PropertyType.STRING,
        turn_idx=0,
        chunk_idx=0,
        source_text="entity is active",
    )

    engine.add_fact(fact)
    engine.add_fact(Fact.from_dict(fact.to_dict()))

    assert len(engine.get_all_facts("entity", "status")) == 1


def test_conflicting_entity_types_for_same_name_are_retained_gracefully() -> None:
    store = FoundryStore()
    resolver = EntityResolver(store)
    canonical = resolver.resolve_or_create("Phoenix", hint_type="city")
    second = resolver.resolve_or_create("Phoenix", hint_type="bird")
    entity = store.entity_graph.get_entity(canonical)

    assert entity is not None
    assert second == canonical
    assert "city" in entity.types
