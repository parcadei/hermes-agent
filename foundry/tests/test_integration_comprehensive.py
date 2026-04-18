"""Cross-module integration tests for Hermes Foundry."""

from __future__ import annotations

from datetime import datetime

from foundry.core import Foundry, FoundryConfig
from foundry.dynamics import MemorySystem
from foundry.ingestion import EntityResolver, chunk_turns, parse_conversation
from foundry.ontology import EntityType, PropertyType
from foundry.query import QueryPlan
from foundry.storage import EntityNode, Fact, FoundryStore
from foundry.writeback import SupersessionEngine


def test_ontology_to_storage_entitytype_roundtrips_through_typeindex() -> None:
    store = FoundryStore()
    type_label = EntityType.normalize("Pet Cats")
    node = EntityNode(canonical="luna", types={type_label})
    store.entity_graph.add_entity(node)
    store.types.index("luna", type_label)

    loaded = store.entity_graph.get_entity("luna")

    assert loaded is not None
    assert type_label in loaded.types
    assert store.types.lookup("pet cat") == ["luna"]


def test_storage_to_query_filters_stored_facts_correctly() -> None:
    store = FoundryStore()
    luna = store.entity_graph.ensure_entity("luna")
    luna.register_type("cat")
    store.types.index("luna", "cat")
    fact = Fact(
        id="luna-color",
        subject="luna",
        predicate="color",
        object="brown",
        object_type=PropertyType.STRING,
        turn_idx=0,
        chunk_idx=0,
        source_text="Luna is brown",
    )
    store.entity_graph.add_fact(fact)
    store.properties.index("luna", "color", "brown", PropertyType.STRING)

    result = QueryPlan(store).type("cat").where("color", "$eq", "brown").execute()

    assert [entity.canonical for entity in result] == ["luna"]


def test_ingestion_to_storage_parse_chunk_store_preserves_entities_and_facts() -> None:
    raw = "User: I have a cat named Luna.\nAssistant: Luna is a tabby cat."
    turns = parse_conversation(raw)
    chunks = chunk_turns(turns, chunk_size=32)
    foundry = Foundry(FoundryConfig())
    report = foundry.ingest_turns(turns)

    assert len(chunks) == 2
    assert report.chunks_processed == 2
    assert foundry.entity("luna") is not None
    assert any(fact.object == "tabby" for fact in foundry.facts("luna", current_only=False))


def test_ingestion_to_writeback_marks_old_facts_superseded() -> None:
    engine = SupersessionEngine()
    old = Fact(
        id="salary-old",
        subject="user",
        predicate="salary",
        object="$100k",
        object_type=PropertyType.NUMERIC,
        turn_idx=0,
        chunk_idx=0,
        source_text="I make $100k",
    )
    new = Fact(
        id="salary-new",
        subject="user",
        predicate="salary",
        object="$120k",
        object_type=PropertyType.NUMERIC,
        turn_idx=1,
        chunk_idx=1,
        source_text="Now I make $120k",
    )
    engine.add_fact(old)
    engine.add_fact(new)
    replaced = engine.supersede("user", "salary", new)

    assert replaced == "salary-old"
    assert engine.get_all_facts("user", "salary")[0].superseded is True


def test_query_to_retrieval_agentic_retriever_finds_fact_lookup_evidence() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest("User: I have a cat named Luna.\nAssistant: Luna is a tabby cat.")

    evidence, trace = foundry.retrieve("What breed is Luna?")

    assert trace.steps
    assert trace.terminated_by in {"answer", "budget"}
    assert "Luna" in evidence or "tabby" in evidence.lower()


def test_retrieval_to_core_foundry_retrieve_returns_trace_and_evidence() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest("User: I have a cat named Luna.\nAssistant: Luna is a tabby cat.")
    evidence, trace = foundry.retrieve("Luna tabby")

    assert evidence
    assert trace.steps
    assert trace.terminated_by in {"answer", "budget"}


def test_core_to_dynamics_memorysystem_rebuilds_from_foundry_graph() -> None:
    foundry = Foundry(FoundryConfig(synonym_dict={"cat": ["feline"]}))
    foundry.ingest("User: I have a cat named Luna.")

    memory = MemorySystem()
    memory.stated = foundry._store.entity_graph
    memory.build_inferred({"cat": ["feline"]})
    memory.rebuild_derived()

    assert "cat" in memory.derived.type_counts
    assert memory.traverse_layers("feline") == ["luna"]


def test_full_pipeline_raw_text_to_ingest_query_retrieve_answer() -> None:
    raw = "\n".join(
        [
            "Chat Time: 2024-06-20",
            "User: I have a cat named Luna.",
            "Assistant: Luna is a tabby cat.",
            "User: We moved to Portland on 2024-01-03.",
        ]
    )
    foundry = Foundry(FoundryConfig())
    report = foundry.ingest(raw)
    query_result = foundry.entities().type("cat").first()
    evidence, trace = foundry.retrieve("What breed is Luna?")

    assert report.chunks_processed == 3
    assert query_result is not None and query_result.canonical == "luna"
    assert evidence
    assert trace.terminated_by in {"answer", "budget"}
