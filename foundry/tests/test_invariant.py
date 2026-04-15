"""Invariant tests for Hermes Foundry.

These tests target properties that should continue to hold independent of
specific input examples.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from foundry.dynamics import DecayFunction, ImportanceScore
from foundry.ingestion import fact_from_triple
from foundry.ontology import EntityType, GroundingLayer, PropertyType
from foundry.query import FilterOp, QueryPlan
from foundry.storage import (
    BM25Index,
    CooccurrenceIndex,
    EntityNode,
    Fact,
    FoundryStore,
    RawChunk,
    TemporalEvent,
    VectorIndex,
)
from foundry.writeback import EntityMerger, SupersessionEngine


def _make_fact(
    fact_id: str,
    subject: str = "user",
    predicate: str = "city",
    obj: str = "Portland",
    turn_idx: int = 0,
    *,
    timestamp: datetime | None = None,
) -> Fact:
    return Fact(
        id=fact_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        object_type=PropertyType.STRING,
        turn_idx=turn_idx,
        chunk_idx=turn_idx,
        timestamp=timestamp,
        source_text=f"{subject} {predicate} {obj}",
    )


def _entity_signature(entity: EntityNode, *, aliases_to_ignore: set[str] | None = None) -> dict[str, object]:
    ignored = aliases_to_ignore or set()
    return {
        "aliases": sorted(alias for alias in entity.aliases if alias not in ignored),
        "types": sorted(entity.types),
        "related": sorted(entity.related),
        "mention_turns": sorted(entity.mention_turns),
        "mention_chunks": sorted(entity.mention_chunks),
        "facts": {
            predicate: sorted(
                (
                    "__merged__",
                    fact.predicate,
                    fact.object,
                    fact.turn_idx,
                    fact.chunk_idx,
                )
                for fact in facts
            )
            for predicate, facts in sorted(entity.facts.items())
        },
        "links": {
            predicate: sorted(targets)
            for predicate, targets in sorted(entity.links.items())
        },
    }


def _merge_result(left: EntityNode, right: EntityNode) -> dict[str, object]:
    merger = EntityMerger()
    merged = merger.merge_nodes(
        EntityNode.from_dict(left.to_dict()),
        EntityNode.from_dict(right.to_dict()),
    )
    return _entity_signature(merged, aliases_to_ignore={left.canonical, right.canonical})


def _build_rank_consistent_indexes() -> tuple[BM25Index, VectorIndex]:
    docs = {
        1: "luna tabby cat naps",
        2: "mochi tabby cat plays",
        3: "portland bike commute",
    }
    vocab = sorted({token for text in docs.values() for token in text.split()} | {"luna", "tabby", "cat"})
    positions = {token: idx for idx, token in enumerate(vocab)}

    bm25 = BM25Index()
    vectors = VectorIndex()
    for doc_id, text in docs.items():
        bm25.add(doc_id, text)
        embedding = np.zeros(len(vocab), dtype=np.float32)
        for token in text.split():
            embedding[positions[token]] += 1.0
        vectors.add(doc_id, embedding)
    return bm25, vectors


def _query_embedding(text: str, bm25: BM25Index) -> np.ndarray:
    vocab = sorted({token for doc in bm25.to_dict()["documents"] for token in doc["text"].split()} | set(text.split()))
    positions = {token: idx for idx, token in enumerate(vocab)}
    embedding = np.zeros(len(vocab), dtype=np.float32)
    for token in text.split():
        embedding[positions[token]] += 1.0
    for doc in bm25.to_dict()["documents"]:
        if len(vocab) == len(set(vocab)):
            continue
    return embedding


def _build_store_with_entities() -> FoundryStore:
    store = FoundryStore()

    chunk0 = RawChunk(chunk_idx=0, text="Luna is a tabby cat.", turn_idx=0, role="user", entities=["luna"])
    chunk1 = RawChunk(chunk_idx=1, text="Mochi is a cat friend.", turn_idx=1, role="assistant", entities=["mochi"])
    store.chunks.store(chunk0)
    store.chunks.store(chunk1)
    store.bm25_chunks.add(chunk0.chunk_idx, chunk0.text)
    store.bm25_chunks.add(chunk1.chunk_idx, chunk1.text)

    luna = EntityNode(canonical="luna", aliases={"Luna"}, types={"cat"})
    mochi = EntityNode(canonical="mochi", aliases={"Mochi"}, types={"cat"})

    for fact in [
        _make_fact("f-luna-breed", subject="luna", predicate="breed", obj="tabby", turn_idx=0),
        _make_fact("f-luna-color", subject="luna", predicate="color", obj="brown", turn_idx=0),
        _make_fact("f-mochi-breed", subject="mochi", predicate="breed", obj="calico", turn_idx=1),
    ]:
        store.entity_graph.add_fact(fact)
        store.properties.index(fact.subject, fact.predicate, fact.object, fact.object_type)

    luna = store.entity_graph.ensure_entity("luna")
    mochi = store.entity_graph.ensure_entity("mochi")
    luna.aliases.add("luna")
    mochi.aliases.add("mochi")
    luna.register_type("cat")
    mochi.register_type("cat")
    luna.add_link("friend", "mochi")
    mochi.add_link("friend", "luna")
    luna.related.add("mochi")
    mochi.related.add("luna")
    luna.mention_turns.add(0)
    mochi.mention_turns.add(1)
    luna.mention_chunks.add(0)
    mochi.mention_chunks.add(1)
    store.types.index("luna", "cat")
    store.types.index("mochi", "cat")
    store.bm25_entities.add(0, "luna cat tabby brown")
    store.bm25_entities.add(1, "mochi cat calico")
    return store


def test_every_stored_fact_can_be_retrieved() -> None:
    store = _build_store_with_entities()
    all_ids = {
        fact.id
        for entity in store.entity_graph.all_entities()
        for facts in entity.facts.values()
        for fact in facts
    }

    plan_ids = {fact.id for fact in QueryPlan(store).facts(current_only=False)}
    direct_ids = {fact.id for fact in store.entity_graph.get_entity("luna").facts["breed"]} | {
        fact.id for fact in store.entity_graph.get_entity("mochi").facts["breed"]
    }

    assert all_ids
    assert plan_ids == all_ids
    assert direct_ids <= all_ids


def test_entity_merge_is_commutative_up_to_target_name() -> None:
    left = EntityNode(
        canonical="entity_a",
        aliases={"A", "shared"},
        types={"cat"},
        related={"friend"},
        facts={"color": [_make_fact("a1", subject="entity_a", predicate="color", obj="brown")]},
        links={"friend": {"friend"}},
        mention_turns={1},
        mention_chunks={1},
    )
    right = EntityNode(
        canonical="entity_b",
        aliases={"B", "shared"},
        types={"pet"},
        related={"friend"},
        facts={"breed": [_make_fact("b1", subject="entity_b", predicate="breed", obj="tabby")]},
        links={"friend": {"friend"}},
        mention_turns={2},
        mention_chunks={2},
    )

    left_into_right = _merge_result(left, right)
    right_into_left = _merge_result(right, left)

    assert left_into_right == right_into_left


def test_supersession_never_increases_current_fact_count() -> None:
    engine = SupersessionEngine()
    facts = [
        _make_fact("f0", obj="san francisco", turn_idx=0),
        _make_fact("f1", obj="seattle", turn_idx=1),
        _make_fact("f2", obj="portland", turn_idx=2),
    ]

    current_counts: list[int] = []
    for fact in facts:
        engine.add_fact(fact)
        engine.supersede("user", "city", fact)
        current = [row for row in engine.get_all_facts("user", "city") if not row.superseded]
        current_counts.append(len(current))

    assert current_counts == sorted(current_counts, reverse=True)
    assert current_counts[-1] <= 1


def test_foundry_store_serialization_is_lossless(tmp_path) -> None:
    store = _build_store_with_entities()
    first_event = TemporalEvent(
        event="adopted luna",
        start_dt=datetime(2024, 1, 2),
        end_dt=datetime(2024, 1, 2),
        source="User adopted Luna on 2024-01-02.",
        turn_id=2,
        subject="luna",
    )
    store.temporal.ingest(first_event)
    store.cooccurrence.record("luna", "mochi", 0)

    path = tmp_path / "store"
    store.save(str(path))
    loaded = FoundryStore.load(str(path))

    assert loaded.entity_graph.to_dict() == store.entity_graph.to_dict()
    assert loaded.inferred_graph.to_dict() == store.inferred_graph.to_dict()
    assert loaded.derived.to_dict() == store.derived.to_dict()
    assert loaded.bm25_chunks.to_dict() == store.bm25_chunks.to_dict()
    assert loaded.bm25_entities.to_dict() == store.bm25_entities.to_dict()
    assert loaded.types.to_dict() == store.types.to_dict()
    assert loaded.properties.to_dict() == store.properties.to_dict()
    assert loaded.temporal.to_dict() == store.temporal.to_dict()
    assert loaded.cooccurrence.to_dict() == store.cooccurrence.to_dict()
    assert loaded.chunks.to_dict() == store.chunks.to_dict()


def test_bm25_and_vector_indexes_agree_for_identical_query_signal() -> None:
    bm25 = BM25Index()
    vectors = VectorIndex()
    docs = {
        1: "luna tabby cat naps",
        2: "mochi tabby cat plays",
        3: "portland bike commute",
    }
    vocab = sorted({token for text in docs.values() for token in text.split()} | {"luna", "tabby", "cat"})
    positions = {token: idx for idx, token in enumerate(vocab)}

    for doc_id, text in docs.items():
        bm25.add(doc_id, text)
        vec = np.zeros(len(vocab), dtype=np.float32)
        for token in text.split():
            vec[positions[token]] += 1.0
        vectors.add(doc_id, vec)

    query_vec = np.zeros(len(vocab), dtype=np.float32)
    for token in "luna tabby cat".split():
        query_vec[positions[token]] += 1.0

    bm25_hits = [doc_id for doc_id, _score in bm25.query("luna tabby cat", top_k=3)]
    vector_hits = [doc_id for doc_id, _score in vectors.query(query_vec, top_k=3)]

    assert bm25_hits[0] == vector_hits[0] == 1
    assert set(bm25_hits[:2]) == set(vector_hits[:2]) == {1, 2}


def test_cooccurrence_index_is_symmetric() -> None:
    index = CooccurrenceIndex()
    index.record("Luna", "Mochi", 2)
    index.record("Mochi", "Luna", 3)

    assert index.get_cooccurrences("luna") == {"mochi": 2}
    assert index.get_cooccurrences("mochi") == {"luna": 2}
    assert index.get_shared_chunks("luna", "mochi") == {2, 3}
    assert index.get_shared_chunks("mochi", "luna") == {2, 3}


def test_query_plan_union_and_intersection_are_associative() -> None:
    store = _build_store_with_entities()
    extra = EntityNode(canonical="bike_1", types={"bike"})
    extra_facts = [_make_fact("bike-type", subject="bike_1", predicate="kind", obj="road", turn_idx=3)]
    store.entity_graph.add_entity(extra)
    for fact in extra_facts:
        store.entity_graph.add_fact(fact)
        store.properties.index(fact.subject, fact.predicate, fact.object, fact.object_type)
    store.entity_graph.ensure_entity("bike_1").register_type("bike")
    store.types.index("bike_1", "bike")

    cats = QueryPlan(store).type("cat")
    pets = QueryPlan(store).where("breed", "$contains", "a")
    bikes = QueryPlan(store).type("bike")

    left_union = cats.union(pets).union(bikes)
    right_union = cats.union(pets.union(bikes))
    left_intersection = cats.intersection(pets).intersection(QueryPlan(store))
    right_intersection = cats.intersection(pets.intersection(QueryPlan(store)))

    assert {entity.canonical for entity in left_union.execute()} == {
        entity.canonical for entity in right_union.execute()
    }
    assert {entity.canonical for entity in left_intersection.execute()} == {
        entity.canonical for entity in right_intersection.execute()
    }


def test_grounding_layer_ordering_is_preserved() -> None:
    assert [layer.value for layer in GroundingLayer] == ["stated", "inferred", "derived"]


def test_temporal_validity_intervals_do_not_overlap_after_supersession() -> None:
    engine = SupersessionEngine()
    base = datetime(2024, 1, 1, 12, 0, 0)
    facts = [
        _make_fact("t0", predicate="lives_in", obj="sf", turn_idx=0, timestamp=base),
        _make_fact("t1", predicate="lives_in", obj="seattle", turn_idx=1, timestamp=base + timedelta(days=1)),
        _make_fact("t2", predicate="lives_in", obj="portland", turn_idx=2, timestamp=base + timedelta(days=2)),
    ]

    for fact in facts:
        engine.add_fact(fact)
        engine.supersede("user", "lives_in", fact)

    windows = [engine.get_validity(fact.id) for fact in facts]
    assert all(window is not None for window in windows)
    assert windows[0].valid_until <= windows[1].valid_from
    assert windows[1].valid_until <= windows[2].valid_from
    assert windows[2].valid_until is None


def test_importance_is_monotonically_decreasing_under_decay_without_access() -> None:
    entity = EntityNode(canonical="luna")
    entity.mention_turns = {0, 2, 4}
    entity.related = {"mochi", "user"}
    score = ImportanceScore.compute(
        entity=entity,
        total_turns=10,
        current_turn=10,
        retrieval_count=2,
        max_degree=5,
    )
    decay = DecayFunction(decay_lambda=0.05)

    day_0 = decay.compute(0, base_salience=score.composite)
    day_5 = decay.compute(5, base_salience=score.composite)
    day_20 = decay.compute(20, base_salience=score.composite)

    assert day_0 >= day_5 >= day_20 >= 0.0
