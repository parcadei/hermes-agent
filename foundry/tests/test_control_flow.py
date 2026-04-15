"""Control-flow and edge-case tests for Hermes Foundry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from foundry.core import Foundry, FoundryConfig
from foundry.dynamics import DreamConsolidator, HebbianBridge
from foundry.ontology import PropertyType
from foundry.query import FilterOp, QueryPlan
from foundry.retrieval import AgenticRetriever, RetrievalBudget
from foundry.storage import EntityNode, Fact, FoundryStore


def _linked_store() -> FoundryStore:
    store = FoundryStore()
    for idx, (canonical, type_label, color) in enumerate(
        [("luna", "cat", "brown"), ("mochi", "cat", "white"), ("roadster", "bike", "blue")]
    ):
        entity = store.entity_graph.ensure_entity(canonical)
        entity.aliases.add(canonical)
        entity.register_type(type_label)
        entity.mention_turns.add(idx)
        entity.mention_chunks.add(idx)
        store.types.index(canonical, type_label)
        fact = Fact(
            id=f"{canonical}-color",
            subject=canonical,
            predicate="color",
            object=color,
            object_type=PropertyType.STRING,
            turn_idx=idx,
            chunk_idx=idx,
            source_text=f"{canonical} color {color}",
        )
        store.entity_graph.add_fact(fact)
        store.properties.index(canonical, "color", color, PropertyType.STRING)
    store.entity_graph.add_link("luna", "friend", "mochi")
    store.entity_graph.add_link("mochi", "friend", "roadster")
    store.entity_graph.add_link("roadster", "friend", "luna")
    return store


def test_empty_conversation_ingest_returns_empty_report() -> None:
    foundry = Foundry(FoundryConfig())
    report = foundry.ingest("")

    assert report.chunks_processed == 0
    assert report.facts_added == 0


def test_single_turn_conversation_creates_no_cooccurrences() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest("User: I have a cat named Luna.")

    co_mentions = foundry._store.cooccurrence.get_cooccurrences("luna")
    assert set(co_mentions) <= {"user"}


def test_query_with_no_matches_returns_empty_result_set() -> None:
    store = _linked_store()
    result = QueryPlan(store).where("color", "$eq", "green").execute()

    assert result == []


def test_query_with_all_matches_filtered_out_returns_empty_result_set() -> None:
    store = _linked_store()
    result = QueryPlan(store).type("cat").where("color", "$eq", "blue").execute()

    assert result == []


def test_nested_logical_ops_evaluate_correctly() -> None:
    store = _linked_store()
    logic = (
        QueryPlan(store)
        .filter(
            FilterOp("type", "$eq", "cat")
        )
        .and_(
            QueryPlan(store).or_(
                FilterOp("color", "$eq", "brown"),
                FilterOp("color", "$eq", "white"),
            )._filters[-1],
            QueryPlan(store).not_(FilterOp("name", "$eq", "roadster"))._filters[-1],
        )
    )

    assert {entity.canonical for entity in logic.execute()} == {"luna", "mochi"}


def test_seed_only_query_returns_only_source_node_without_traversal() -> None:
    store = _linked_store()
    result = QueryPlan(store, seed={"luna"}).execute()

    assert [entity.canonical for entity in result] == ["luna"]


def test_traversal_with_cycle_terminates_and_does_not_loop_forever() -> None:
    store = _linked_store()
    result = QueryPlan(store, seed={"luna"}).related("friend", max_hops=3).execute()

    assert {entity.canonical for entity in result} == {"luna", "mochi", "roadster"}


def test_agentic_retriever_with_zero_budget_returns_no_action() -> None:
    retriever = AgenticRetriever(
        {
            "search_memory": lambda query, top_k=5: [{"turn_id": 0, "text": "hit"}],
            "answer": lambda evidence: evidence,
        },
        budget=RetrievalBudget(max_steps=0),
    )

    evidence, trace = retriever.retrieve("What do you remember?")

    assert evidence == ""
    assert trace.terminated_by == "no_action"


def test_dream_consolidator_with_empty_store_is_safe_noop() -> None:
    result = DreamConsolidator().consolidate(max_cycles=0)

    assert result["cycles_run"] == 0
    assert result["drift_detected"] is False


def test_hebbian_bridge_with_no_data_creates_no_patterns() -> None:
    bridge = HebbianBridge()
    assert bridge.get_frequent_patterns(min_count=1) == []


def test_foundrystore_save_to_read_only_path_raises_os_error(tmp_path: Path) -> None:
    store = _linked_store()
    read_only_dir = tmp_path / "readonly"
    read_only_dir.mkdir()
    read_only_dir.chmod(0o500)

    try:
        with pytest.raises(OSError):
            store.save(str(read_only_dir / "store"))
    finally:
        read_only_dir.chmod(0o700)


def test_concurrent_read_access_patterns_return_consistent_results() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest("User: I have a cat named Luna.\nAssistant: Luna is a tabby cat.")

    def run_search() -> tuple[int, int]:
        return len(foundry.search("Luna", top_k=5)), foundry.entities().type("cat").count()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _idx: run_search(), range(8)))

    assert len(set(results)) == 1
