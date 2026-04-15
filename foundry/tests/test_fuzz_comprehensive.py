"""Comprehensive fuzz tests for Hermes Foundry."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from foundry.ingestion import EntityResolver, chunk_turns, infer_type, parse_conversation
from foundry.ontology import PropertyType, infer_property_type
from foundry.query import FilterOp, LogicalOp, QueryPlan
from foundry.storage import BM25Index, EntityNode, Fact, FoundryStore, TemporalEvent
from foundry.writeback import SupersessionEngine


FUZZ_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _fact_strategy() -> st.SearchStrategy[Fact]:
    safe_text = st.text(min_size=0, max_size=32)
    return st.builds(
        Fact,
        id=st.text(min_size=1, max_size=16),
        subject=safe_text,
        predicate=safe_text,
        object=safe_text,
        object_type=st.sampled_from(list(PropertyType)),
        turn_idx=st.integers(min_value=0, max_value=30),
        chunk_idx=st.integers(min_value=-1, max_value=30),
        source_text=safe_text,
    )


def _filter_tree_strategy() -> st.SearchStrategy[FilterOp | LogicalOp]:
    field = st.sampled_from(["type", "breed", "color", "age", "status", "name"])
    operator = st.sampled_from(
        [
            "$eq",
            "$ne",
            "$contains",
            "$startsWith",
            "$regex",
            "$phrase",
            "$matchAnyToken",
            "$matchAllTokens",
            "$fuzzy",
            "$isNull",
            "$hasProperty",
        ]
    )
    leaf = st.builds(
        FilterOp,
        field=field,
        operator=operator,
        value=st.one_of(st.none(), st.text(max_size=20), st.integers(min_value=-5, max_value=20)),
        negate=st.booleans(),
    )
    return st.recursive(
        leaf,
        lambda children: st.builds(
            LogicalOp,
            op=st.sampled_from(["$and", "$or", "$not"]),
            children=st.lists(children, min_size=1, max_size=3).map(tuple),
        ),
        max_leaves=8,
    )


def _build_query_store() -> FoundryStore:
    store = FoundryStore()
    items = [
        ("luna", "cat", "tabby", "brown"),
        ("mochi", "cat", "calico", "white"),
        ("roadster", "bike", "", "blue"),
    ]
    for idx, (canonical, type_label, breed, color) in enumerate(items):
        entity = store.entity_graph.ensure_entity(canonical)
        entity.aliases.add(canonical)
        entity.register_type(type_label)
        entity.mention_turns.add(idx)
        store.types.index(canonical, type_label)
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
        color_fact = Fact(
            id=f"{canonical}-color",
            subject=canonical,
            predicate="color",
            object=color,
            object_type=PropertyType.STRING,
            turn_idx=idx,
            chunk_idx=idx,
            source_text=f"{canonical} color {color}",
        )
        store.entity_graph.add_fact(color_fact)
        store.properties.index(canonical, "color", color, PropertyType.STRING)
    return store


@FUZZ_SETTINGS
@given(
    subject=st.text(min_size=0, max_size=64),
    predicate=st.text(min_size=0, max_size=64),
    obj=st.text(min_size=0, max_size=64),
)
def test_random_unicode_fact_components_round_trip(subject: str, predicate: str, obj: str) -> None:
    fact = Fact(
        id="fuzz",
        subject=subject,
        predicate=predicate,
        object=obj,
        object_type=PropertyType.STRING,
        turn_idx=0,
        source_text=f"{subject} {predicate} {obj}",
    )

    payload = fact.to_dict()
    restored = Fact.from_dict(payload)

    assert restored.to_dict() == payload


@FUZZ_SETTINGS
@given(op_tree=_filter_tree_strategy())
def test_random_nested_query_trees_execute_without_crashing(op_tree: FilterOp | LogicalOp) -> None:
    store = _build_query_store()
    result = QueryPlan(store).filter(op_tree).execute()
    assert isinstance(result, list)


@FUZZ_SETTINGS
@given(facts=st.lists(_fact_strategy(), min_size=1, max_size=12))
def test_random_fact_streams_with_contradictions_leave_at_most_one_current_fact(facts: list[Fact]) -> None:
    engine = SupersessionEngine()
    for idx, fact in enumerate(facts):
        fact.id = f"fact-{idx}"
        fact.subject = "entity"
        fact.predicate = "status"
        fact.turn_idx = idx
        fact.chunk_idx = idx
        engine.add_fact(fact)
        engine.supersede("entity", "status", fact)

    current = [fact for fact in engine.get_all_facts("entity", "status") if not fact.superseded]
    assert len(current) <= 1


@FUZZ_SETTINGS
@given(
    start_offset=st.integers(min_value=-30, max_value=30),
    end_offset=st.integers(min_value=-30, max_value=30),
)
def test_random_temporal_intervals_accept_valid_and_reject_negative_ranges(
    start_offset: int,
    end_offset: int,
) -> None:
    base = datetime(2024, 1, 1)
    start = base + timedelta(days=start_offset)
    end = base + timedelta(days=end_offset)

    if start <= end:
        event = TemporalEvent(event="fuzz", start_dt=start, end_dt=end, source="source")
        assert event.start_dt <= event.end_dt
    else:
        with pytest.raises(ValueError):
            TemporalEvent(event="fuzz", start_dt=start, end_dt=end, source="source")


@FUZZ_SETTINGS
@given(
    lines=st.lists(
        st.tuples(
            st.sampled_from(["User", "Assistant", "System", "", "user", "assistant"]),
            st.one_of(st.none(), st.text(min_size=0, max_size=80)),
        ),
        min_size=0,
        max_size=20,
    )
)
def test_random_conversation_turns_with_missing_fields_parse_and_chunk(
    lines: list[tuple[str, str | None]],
) -> None:
    raw = "\n".join(
        f"{role}: {text or ''}" if role else (text or "")
        for role, text in lines
    )
    turns = parse_conversation(raw)
    chunks = chunk_turns(turns, chunk_size=8)

    assert all(turn.turn_idx >= 0 for turn in turns)
    assert all(chunk.chunk_idx >= 0 for chunk in chunks)


@FUZZ_SETTINGS
@given(
    canonical=st.text(min_size=1, max_size=20),
    aliases=st.sets(st.text(min_size=0, max_size=20), min_size=1, max_size=8),
    mention=st.text(min_size=0, max_size=20),
)
def test_entity_resolver_is_deterministic_for_adversarial_alias_sets(
    canonical: str,
    aliases: set[str],
    mention: str,
) -> None:
    resolver = EntityResolver()
    resolver.register(canonical, aliases=aliases, type_hint="pet")

    first = resolver.resolve(mention, hint_type="pet")
    second = resolver.resolve(mention, hint_type="pet")

    assert first == second


@FUZZ_SETTINGS
@given(value=st.text(min_size=0, max_size=40))
def test_infer_property_type_handles_edge_case_strings(value: str) -> None:
    result = infer_property_type(value)
    assert result in set(PropertyType)


@FUZZ_SETTINGS
@given(query=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=40))
def test_bm25_index_handles_queries_with_special_characters(query: str) -> None:
    index = BM25Index()
    index.add(0, "luna tabby cat")
    index.add(1, "mochi calico cat")
    index.add(2, "road bike blue")

    hits = index.query(query + r".*?[](){}^$\\", top_k=3)
    assert isinstance(hits, list)
