"""Tests for query-time conflict resolution in Engram retrieval."""

from __future__ import annotations

from foundry.core import Foundry, FoundryConfig
from foundry.ontology import PropertyType
from foundry.retrieval import search_memory, search_memory_hybrid
from foundry.storage import Fact, FoundryStore, RawChunk
from foundry.writeback import ConflictDetector


def _add_chunk_fact(
    store: FoundryStore,
    *,
    chunk_idx: int,
    turn_idx: int,
    text: str,
    entity: str,
    predicate: str,
    obj: str,
) -> Fact:
    chunk = RawChunk(
        chunk_idx=chunk_idx,
        text=text,
        turn_idx=turn_idx,
        role="user",
        entities=[entity],
    )
    store.chunks.store(chunk)
    store.bm25_chunks.add(chunk_idx, text)

    node = store.entity_graph.ensure_entity(entity)
    node.mention_turns.add(turn_idx)
    node.mention_chunks.add(chunk_idx)
    node.aliases.add(entity)
    fact = Fact(
        id=f"{entity}-{predicate}-{chunk_idx}",
        subject=entity,
        predicate=predicate,
        object=obj,
        object_type=PropertyType.STRING,
        turn_idx=turn_idx,
        chunk_idx=chunk_idx,
        source_text=text,
    )
    node.add_fact(fact)
    return fact


def test_conflictdetector_filters_older_conflicting_chunk() -> None:
    store = FoundryStore()
    _add_chunk_fact(
        store,
        chunk_idx=0,
        turn_idx=0,
        text="User: My favorite color is blue.",
        entity="user",
        predicate="favorite_color",
        obj="blue",
    )
    _add_chunk_fact(
        store,
        chunk_idx=1,
        turn_idx=1,
        text="User: My favorite color is green.",
        entity="user",
        predicate="favorite_color",
        obj="green",
    )

    detector = ConflictDetector(store)
    results = [
        {"turn_id": 0, "chunk_id": 0, "text": "My favorite color is blue.", "score": 0.9},
        {"turn_id": 1, "chunk_id": 1, "text": "My favorite color is green.", "score": 0.8},
    ]

    filtered = detector.filter_conflicts(results)

    assert [result["chunk_id"] for result in filtered] == [1]


def test_conflictdetector_keeps_results_when_predicates_differ() -> None:
    store = FoundryStore()
    _add_chunk_fact(
        store,
        chunk_idx=0,
        turn_idx=0,
        text="User: I live in Seattle.",
        entity="user",
        predicate="lives_in",
        obj="Seattle",
    )
    _add_chunk_fact(
        store,
        chunk_idx=1,
        turn_idx=1,
        text="User: I work at Acme.",
        entity="user",
        predicate="works_at",
        obj="Acme",
    )

    detector = ConflictDetector(store)
    results = [
        {"turn_id": 0, "chunk_id": 0, "text": "I live in Seattle.", "score": 0.9},
        {"turn_id": 1, "chunk_id": 1, "text": "I work at Acme.", "score": 0.8},
    ]

    assert detector.filter_conflicts(results) == results


def test_conflictdetector_keeps_repeated_same_object_facts() -> None:
    store = FoundryStore()
    _add_chunk_fact(
        store,
        chunk_idx=0,
        turn_idx=0,
        text="User: My dog is named Milo.",
        entity="dog",
        predicate="name",
        obj="Milo",
    )
    _add_chunk_fact(
        store,
        chunk_idx=1,
        turn_idx=1,
        text="User: My dog is still named Milo.",
        entity="dog",
        predicate="name",
        obj="Milo",
    )

    detector = ConflictDetector(store)
    results = [
        {"turn_id": 0, "chunk_id": 0, "text": "My dog is named Milo.", "score": 0.9},
        {"turn_id": 1, "chunk_id": 1, "text": "My dog is still named Milo.", "score": 0.8},
    ]

    assert detector.filter_conflicts(results) == results


def test_conflictdetector_detects_old_and_new_object_pairs() -> None:
    store = FoundryStore()
    _add_chunk_fact(
        store,
        chunk_idx=0,
        turn_idx=0,
        text="User: My favorite color is blue.",
        entity="user",
        predicate="favorite_color",
        obj="blue",
    )
    _add_chunk_fact(
        store,
        chunk_idx=1,
        turn_idx=1,
        text="User: My favorite color is green.",
        entity="user",
        predicate="favorite_color",
        obj="green",
    )

    detector = ConflictDetector(store)
    results = [
        {"turn_id": 0, "chunk_id": 0, "text": "My favorite color is blue.", "score": 0.9},
        {"turn_id": 1, "chunk_id": 1, "text": "My favorite color is green.", "score": 0.8},
    ]

    assert detector.detect_conflicts(results) == [("user", "favorite_color", "blue", "green")]


def test_conflictdetector_cosine_dedup_keeps_most_recent_duplicate() -> None:
    detector = ConflictDetector(FoundryStore())
    results = [
        {"turn_id": 0, "chunk_id": 0, "text": "I work at Acme in Seattle", "score": 0.95},
        {"turn_id": 2, "chunk_id": 2, "text": "I work at Acme in Seattle.", "score": 0.80},
        {"turn_id": 3, "chunk_id": 3, "text": "I bike to work on Fridays.", "score": 0.70},
    ]

    filtered = detector.cosine_dedup(results)

    assert [result["chunk_id"] for result in filtered] == [2, 3]


def test_search_memory_applies_conflict_resolution_after_ingest() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest(
        "\n".join(
            [
                "User: I live in Seattle.",
                "User: I live in Portland.",
            ]
        )
    )

    raw_results = search_memory_hybrid(
        "Seattle Portland live",
        foundry._store.bm25_chunks,
        foundry._store.vectors,
        top_k=5,
        chunk_store=foundry._store.chunks,
    )
    filtered = search_memory(foundry._store, "Seattle Portland live", top_k=5)

    assert any("Seattle" in result["text"] for result in raw_results)
    assert any("Portland" in result["text"] for result in raw_results)
    assert not any("Seattle" in result["text"] for result in filtered)
    assert any("Portland" in result["text"] for result in filtered)
    assert foundry.resolve_conflicts(raw_results) == filtered


def test_search_memory_overfetches_before_conflict_resolution() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest(
        "\n".join(
            [
                "User: I live in Seattle.",
                "User: I live in Portland.",
            ]
        )
    )

    raw_top_one = search_memory_hybrid(
        "live",
        foundry._store.bm25_chunks,
        foundry._store.vectors,
        top_k=1,
        chunk_store=foundry._store.chunks,
    )
    filtered_top_one = search_memory(foundry._store, "live", top_k=1)

    assert len(raw_top_one) == 1
    assert "Seattle" in raw_top_one[0]["text"]
    assert len(filtered_top_one) == 1
    assert "Portland" in filtered_top_one[0]["text"]


def test_conflictdetector_handles_empty_single_and_entityless_results() -> None:
    store = FoundryStore()
    store.chunks.store(RawChunk(chunk_idx=7, text="No entities here.", turn_idx=7, role="user", entities=[]))
    detector = ConflictDetector(store)

    empty: list[dict] = []
    single = [{"turn_id": 7, "chunk_id": 7, "text": "No entities here.", "score": 0.5}]

    assert detector.filter_conflicts(empty) == []
    assert detector.detect_conflicts(empty) == []
    assert detector.filter_conflicts(single) == single
