"""Snapshot-style regression tests for Hermes Foundry."""

from __future__ import annotations

import json

from foundry.core import Foundry, FoundryConfig


def test_snapshot_simple_conversation_counts_and_relations() -> None:
    raw = "\n".join(
        [
            "User: I have a cat named Luna.",
            "Assistant: Luna is a tabby cat.",
            "User: I live in Portland.",
        ]
    )
    foundry = Foundry(FoundryConfig())
    report = foundry.ingest(raw)
    stats = foundry.stats()

    assert {
        "chunks_processed": report.chunks_processed,
        "entities_created": report.entities_created,
        "entities_resolved": report.entities_resolved,
        "facts_added": report.facts_added,
        "facts_superseded": report.facts_superseded,
        "facts_rejected": report.facts_rejected,
        "temporal_events": report.temporal_events,
        "cooccurrences": report.cooccurrences,
    } == {
        "chunks_processed": 3,
        "entities_created": 2,
        "entities_resolved": 2,
        "facts_added": 3,
        "facts_superseded": 0,
        "facts_rejected": 0,
        "temporal_events": 0,
        "cooccurrences": 5,
    }

    assert {
        "entity_count": stats["entity_count"],
        "fact_count": stats["fact_count"],
        "current_fact_count": stats["current_fact_count"],
        "chunk_count": stats["chunk_count"],
        "type_distribution": stats["type_distribution"],
    } == {
        "entity_count": 3,
        "fact_count": 3,
        "current_fact_count": 3,
        "chunk_count": 3,
        "type_distribution": {"cat": 1, "pet": 1},
    }

    assert foundry.entity("luna").related == {"user"}


def test_snapshot_entity_query_result_ordering() -> None:
    raw = "\n".join(
        [
            "User: I have a cat named Luna.",
            "Assistant: Luna is a tabby cat.",
            "User: I live in Portland.",
        ]
    )
    foundry = Foundry(FoundryConfig())
    foundry.ingest(raw)

    results = foundry.search_entities("Luna", top_k=2)

    assert [(entity, sorted(turns)) for entity, turns, _score in results] == [
        ("luna", [0, 1]),
        ("user", [0, 2]),
    ]


def test_snapshot_supersession_edge_case_current_fact_set() -> None:
    raw = "\n".join(
        [
            "User: I moved to Seattle.",
            "Assistant: Noted.",
            "User: I moved to Portland.",
        ]
    )
    foundry = Foundry(FoundryConfig())
    foundry.ingest(raw)

    all_user_facts = [
        (fact.predicate, fact.object, fact.superseded, fact.superseded_by)
        for fact in foundry.facts("user", current_only=False)
    ]
    current_user_facts = [(fact.predicate, fact.object) for fact in foundry.facts("user", current_only=True)]

    assert current_user_facts == [("lives_in", "Portland")]
    assert all_user_facts == [
        ("lives_in", "Seattle", True, "user:moved_to:4a10004812d8"),
        ("moved_to", "Seattle", True, "user:lives_in:319b7c43381f"),
        ("lives_in", "Portland", False, None),
        ("moved_to", "Portland", True, "user:lives_in:4a10004812d8"),
    ]


def test_snapshot_serialization_schema_stability(tmp_path) -> None:
    raw = "\n".join(
        [
            "User: I have a cat named Luna.",
            "Assistant: Luna is a tabby cat.",
        ]
    )
    foundry = Foundry(FoundryConfig())
    foundry.ingest(raw)
    path = tmp_path / "store"
    foundry.save(str(path))

    manifest = json.loads((path / "manifest.json").read_text())
    entity_graph = json.loads((path / "entity_graph.json").read_text())

    assert sorted(manifest.keys()) == [
        "config_snapshot",
        "created_at",
        "foundry_version",
        "index_files",
        "schema_version",
    ]
    assert manifest["index_files"] == [
        "bm25_chunks.json",
        "bm25_entities.json",
        "vector_meta.json",
        "vectors.npy",
        "types.json",
        "properties.json",
        "temporal.json",
        "cooccurrence.json",
        "chunks.json",
        "entity_graph.json",
        "inferred_graph.json",
        "derived.json",
    ]
    assert sorted(entity_graph.keys()) == ["entities"]
    assert sorted(entity_graph["entities"][0].keys()) == [
        "_type_counts",
        "aliases",
        "canonical",
        "facts",
        "links",
        "mention_chunks",
        "mention_turns",
        "related",
        "types",
    ]
