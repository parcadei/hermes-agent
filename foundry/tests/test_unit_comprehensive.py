"""Deep unit coverage for Hermes Foundry public surfaces."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from foundry.dynamics import DecayFunction, ImportanceScore
from foundry.ingestion import ConversationTurn, EntityResolver, ExtractedTriple, fact_from_triple, parse_conversation
from foundry.ontology import Cardinality, EntityType, Facet, GroundingLayer, LinkType, PropertyType, infer_property_type
from foundry.storage import BM25Index, CooccurrenceIndex, EntityNode, Fact, PropertyIndex, TemporalEvent, TemporalIndex, TypeIndex, VectorIndex
from foundry.writeback import DeletionPolicy, SupersessionEngine


@pytest.mark.parametrize("facet", list(Facet))
def test_every_facet_has_unique_string_representation(facet: Facet) -> None:
    values = [item.value for item in Facet]
    assert values.count(facet.value) == 1
    assert isinstance(facet, str)
    assert facet.value == facet


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("42", PropertyType.NUMERIC),
        ("2024-04-15", PropertyType.DATE),
        ("3 weeks", PropertyType.DURATION),
        ("true", PropertyType.BOOLEAN),
        ("cats, dogs, and fish", PropertyType.LIST),
        ("soft spoken", PropertyType.STRING),
    ],
)
def test_infer_property_type_for_canonical_inputs(value: str, expected: PropertyType) -> None:
    assert infer_property_type(value) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("   ", ""),
        (" Café Racer ", "café_racer"),
        ("123 BIKES", "123_bike"),
        ("MiXeD Case Pet", "mixed_case_pet"),
    ],
)
def test_entitytype_normalize_handles_edge_cases(raw: str, expected: str) -> None:
    assert EntityType.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("bikes", "bike"),
        ("hobbies", "hobby"),
        ("buses", "bus"),
        ("boxes", "box"),
        ("quizzes", "quizz"),
        ("fish", "fish"),
        ("glass", "glass"),
    ],
)
def test_entitytype_depluralize_handles_regular_and_irregular_cases(raw: str, expected: str) -> None:
    assert EntityType.depluralize(raw) == expected


def test_linktype_inverse_name_differs_from_name() -> None:
    link = LinkType(
        name="visited",
        inverse_name="visited_by",
        cardinality=Cardinality.MANY_TO_MANY,
        source_type="user",
        target_type="place",
    )
    assert link.name != link.inverse_name


def test_grounding_layer_ordering_and_uniqueness() -> None:
    assert [layer.value for layer in GroundingLayer] == ["stated", "inferred", "derived"]
    assert len({layer.value for layer in GroundingLayer}) == 3


def test_fact_from_triple_produces_non_negative_turn_index() -> None:
    triple = ExtractedTriple(
        subject="Luna",
        predicate="breed",
        object="tabby",
        turn_idx=4,
        chunk_idx=1,
        source_text="Luna is a tabby cat.",
    )
    fact = fact_from_triple(triple, canonical_subject="luna")
    assert fact.turn_idx >= 0
    assert fact.chunk_idx >= 0


def test_bm25_query_results_are_sorted_by_descending_score() -> None:
    index = BM25Index()
    index.add(0, "luna tabby cat")
    index.add(1, "luna cat")
    index.add(2, "bike commute")

    scores = index.query("luna cat", top_k=3)

    assert scores == sorted(scores, key=lambda item: (-item[1], item[0]))


def test_vector_query_results_are_sorted_by_descending_similarity() -> None:
    index = VectorIndex()
    index.add(0, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    index.add(1, np.array([0.8, 0.2, 0.0], dtype=np.float32))
    index.add(2, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    hits = index.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)

    assert hits == sorted(hits, key=lambda item: (-item[1], item[0]))


def test_typeindex_indexes_primary_and_secondary_types() -> None:
    index = TypeIndex()
    index.index("luna", "cat")
    index.index("luna", "pet")

    assert index.lookup("cat") == ["luna"]
    assert index.lookup("pet") == ["luna"]


def test_propertyindex_supports_all_propertytype_variants() -> None:
    index = PropertyIndex()
    for idx, ptype in enumerate(PropertyType):
        index.index("entity", f"prop_{idx}", ptype.value, ptype)

    numeric_key = next(idx for idx, ptype in enumerate(PropertyType) if ptype == PropertyType.NUMERIC)
    index.index("entity", "price", "42", PropertyType.NUMERIC)

    assert len(index.lookup("price")) == 1
    assert index.lookup_numeric("price", "$gte", 40.0) == [("entity", 42.0)]
    assert len(index.lookup(f"prop_{numeric_key}")) == 1


def test_temporalindex_handles_all_allen_relations() -> None:
    base = datetime(2024, 1, 1)
    index = TemporalIndex()
    before_a = TemporalEvent("a", base, base + timedelta(days=1), "src")
    before_b = TemporalEvent("b", base + timedelta(days=3), base + timedelta(days=4), "src")
    meets_a = TemporalEvent("c", base, base + timedelta(days=1), "src")
    meets_b = TemporalEvent("d", base + timedelta(days=1), base + timedelta(days=2), "src")
    overlaps_a = TemporalEvent("e", base, base + timedelta(days=3), "src")
    overlaps_b = TemporalEvent("f", base + timedelta(days=1), base + timedelta(days=4), "src")
    starts_a = TemporalEvent("g", base, base + timedelta(days=1), "src")
    starts_b = TemporalEvent("h", base, base + timedelta(days=3), "src")
    during_a = TemporalEvent("i", base + timedelta(days=1), base + timedelta(days=2), "src")
    during_b = TemporalEvent("j", base, base + timedelta(days=4), "src")
    finishes_a = TemporalEvent("k", base + timedelta(days=1), base + timedelta(days=4), "src")
    finishes_b = TemporalEvent("l", base, base + timedelta(days=4), "src")
    equals_a = TemporalEvent("m", base, base + timedelta(days=2), "src")
    equals_b = TemporalEvent("n", base, base + timedelta(days=2), "src")

    for event in [
        before_a,
        before_b,
        meets_a,
        meets_b,
        overlaps_a,
        overlaps_b,
        starts_a,
        starts_b,
        during_a,
        during_b,
        finishes_a,
        finishes_b,
        equals_a,
        equals_b,
    ]:
        index.ingest(event)

    assert index.allen_relation(before_a, before_b) == "before"
    assert index.allen_relation(before_b, before_a) == "after"
    assert index.allen_relation(meets_a, meets_b) == "meets"
    assert index.allen_relation(meets_b, meets_a) == "met_by"
    assert index.allen_relation(overlaps_a, overlaps_b) == "overlaps"
    assert index.allen_relation(overlaps_b, overlaps_a) == "overlapped_by"
    assert index.allen_relation(starts_a, starts_b) == "starts"
    assert index.allen_relation(starts_b, starts_a) == "started_by"
    assert index.allen_relation(during_a, during_b) == "during"
    assert index.allen_relation(during_b, during_a) == "contains"
    assert index.allen_relation(finishes_a, finishes_b) == "finishes"
    assert index.allen_relation(finishes_b, finishes_a) == "finished_by"
    assert index.allen_relation(equals_a, equals_b) == "equals"


def test_cooccurrence_index_weights_are_non_negative() -> None:
    index = CooccurrenceIndex()
    index.record("luna", "mochi", 0)
    index.record("luna", "mochi", 1)

    assert all(weight >= 0 for weight in index.get_cooccurrences("luna").values())


def test_conversation_turn_parsing_handles_user_assistant_and_manual_system_role() -> None:
    parsed = parse_conversation("User: hello\nAssistant: hi there")
    system_turn = ConversationTurn(turn_idx=2, role="system", text="policy")

    assert [turn.role for turn in parsed] == ["user", "assistant"]
    assert system_turn.role == "system"


def test_entityresolver_resolve_is_deterministic_for_same_input() -> None:
    resolver = EntityResolver()
    resolver.register("luna", aliases={"Luna", "my cat"}, type_hint="cat")

    first = resolver.resolve("my cat", hint_type="cat")
    second = resolver.resolve("my cat", hint_type="cat")

    assert first == second == "luna"


def test_supersessionengine_supersede_returns_replaced_fact_id_and_marks_old_fact() -> None:
    engine = SupersessionEngine()
    old_fact = Fact(
        id="old",
        subject="user",
        predicate="city",
        object="Seattle",
        object_type=PropertyType.STRING,
        turn_idx=0,
        source_text="I live in Seattle",
    )
    new_fact = Fact(
        id="new",
        subject="user",
        predicate="city",
        object="Portland",
        object_type=PropertyType.STRING,
        turn_idx=1,
        source_text="I moved to Portland",
    )

    engine.add_fact(old_fact)
    replaced = engine.supersede("user", "city", new_fact)

    assert replaced == "old"
    assert engine.get_all_facts("user", "city")[0].superseded is True


def test_deletionpolicy_enum_values_are_stable() -> None:
    assert {policy.value for policy in DeletionPolicy} == {"soft_delete", "hard_delete"}


def test_decayfunction_is_monotonically_decreasing() -> None:
    decay = DecayFunction(decay_lambda=0.02)
    scores = [decay.compute(days_since_access=days, base_salience=1.0) for days in [0, 5, 20, 100]]

    assert scores == sorted(scores, reverse=True)


def test_importancescore_combines_access_count_and_recency() -> None:
    stale = EntityNode(canonical="stale")
    stale.mention_turns = {0}
    stale.related = {"a"}
    fresh = EntityNode(canonical="fresh")
    fresh.mention_turns = {8, 9}
    fresh.related = {"a", "b", "c"}

    stale_score = ImportanceScore.compute(stale, total_turns=10, current_turn=10, retrieval_count=0, max_degree=3)
    fresh_score = ImportanceScore.compute(fresh, total_turns=10, current_turn=10, retrieval_count=3, max_degree=3)

    assert fresh_score.recency >= stale_score.recency
    assert fresh_score.retrieval_frequency >= stale_score.retrieval_frequency
    assert fresh_score.composite >= stale_score.composite
