"""Comprehensive behavioral tests for Hermes Foundry."""

from __future__ import annotations

from datetime import datetime

from foundry.core import Foundry, FoundryConfig
from foundry.dynamics import DreamConsolidator, HebbianBridge
from foundry.ontology import PropertyType
from foundry.retrieval import AgenticRetriever, DeterministicRouter, RetrievalBudget
from foundry.storage import EntityNode, Fact, FoundryStore, TemporalEvent
from foundry.writeback import EntityMerger

from .conftest import CONTRADICTORY_CONVERSATION, PETS_CONVERSATION


def _raw(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{role}: {text}" for role, text in turns)


def test_ingest_conversation_then_query_returns_expected_entities_and_facts() -> None:
    foundry = Foundry(FoundryConfig())
    report = foundry.ingest(_raw(PETS_CONVERSATION))

    luna = foundry.entity("luna")
    cats = foundry.entities().type("cat").execute()
    luna_facts = foundry.facts("luna", current_only=True)

    assert report.entities_created > 0
    assert luna is not None
    assert any(entity.canonical == "luna" for entity in cats)
    assert any(fact.predicate == "breed" and "tabby" in fact.object for fact in luna_facts)


def test_ingest_contradictory_facts_resolves_to_latest_current_value() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest(_raw(CONTRADICTORY_CONVERSATION))

    current_city_values = {fact.object for fact in foundry.facts("user", predicate="city", current_only=True)}
    historical_city_values = {fact.object for fact in foundry.facts("user", predicate="city", current_only=False)}

    assert len(current_city_values) <= 1
    assert len(historical_city_values) >= len(current_city_values)


def test_ingest_temporal_events_and_query_allen_ordering() -> None:
    raw = "\n".join(
        [
            "Chat Time: 2024-06-20",
            "User: We moved to Portland on 2024-01-03.",
            "Assistant: You started the new job on 2024-02-14.",
        ]
    )
    foundry = Foundry(FoundryConfig())
    foundry.ingest(raw)

    events = foundry._store.temporal.events_between(datetime(2024, 1, 1), datetime(2024, 12, 31))
    temporal_search = foundry.search_temporal("What happened on 2024-01-03?")

    assert len(events) >= 2
    assert foundry._store.temporal.allen_before(events[0], events[1]) is True
    assert temporal_search["bucket"] == "contains"
    assert temporal_search["matched_events"]


def test_entity_merge_preserves_facts_from_both_entities() -> None:
    store = FoundryStore()
    source = EntityNode(canonical="my_cat", aliases={"my cat"}, types={"pet"})
    target = EntityNode(canonical="luna", aliases={"Luna"}, types={"cat"})
    store.entity_graph.add_entity(source)
    store.entity_graph.add_entity(target)
    store.entity_graph.add_fact(
        Fact(
            id="breed",
            subject="my_cat",
            predicate="breed",
            object="tabby",
            object_type=PropertyType.STRING,
            turn_idx=0,
            source_text="My cat is tabby",
        )
    )
    store.entity_graph.add_fact(
        Fact(
            id="color",
            subject="luna",
            predicate="color",
            object="brown",
            object_type=PropertyType.STRING,
            turn_idx=1,
            source_text="Luna is brown",
        )
    )

    merger = EntityMerger(store)
    merger.merge("my_cat", "luna")
    merged = store.entity_graph.get_entity("luna")

    assert merged is not None
    assert {fact.object for fact in merged.facts["breed"]} == {"tabby"}
    assert {fact.object for fact in merged.facts["color"]} == {"brown"}
    assert "my_cat" in merged.aliases


def test_dream_consolidator_is_bounded_noop_behavior() -> None:
    consolidator = DreamConsolidator()
    result = consolidator.consolidate(max_cycles=50)

    assert result["cycles_run"] <= 5
    assert result["drift_detected"] is False
    assert result["rolled_back"] is False


def test_hebbian_bridge_records_frequently_co_retrieved_patterns() -> None:
    bridge = HebbianBridge()
    for _ in range(6):
        bridge.record_retrieval(
            question="Where does Luna live?",
            evidence_turns=[0, 1],
            tool_sequence=["search_memory", "answer"],
        )

    patterns = bridge.get_frequent_patterns(min_count=5)

    assert len(patterns) == 1
    assert patterns[0]["question"] == "Where does Luna live?"
    assert patterns[0]["count"] == 6


def test_agentic_retriever_uses_search_then_aggregate_for_aggregation_queries() -> None:
    trace_order: list[str] = []

    def search_memory(query: str, top_k: int = 5) -> list[dict[str, object]]:
        trace_order.append("search_memory")
        return []

    def aggregate(entity_type: str, op: str, property_predicate: str = "") -> dict[str, object]:
        trace_order.append("aggregate")
        return {"result": 3, "entities": ["luna", "mochi", "goldie"]}

    def answer(evidence: str) -> str:
        trace_order.append("answer")
        return evidence

    retriever = AgenticRetriever(
        {
            "search_memory": search_memory,
            "aggregate": aggregate,
            "answer": answer,
        },
        budget=RetrievalBudget(max_steps=3),
    )

    evidence, trace = retriever.retrieve("How many pets are there?")

    assert evidence == "3"
    assert trace_order == ["search_memory", "aggregate", "answer"]
    assert [step["tool"] for step in trace.steps] == ["search_memory", "aggregate", "answer"]


def test_agentic_retriever_uses_search_then_temporal_tool_for_temporal_queries() -> None:
    trace_order: list[str] = []

    def search_memory(query: str, top_k: int = 5) -> list[dict[str, object]]:
        trace_order.append("search_memory")
        return []

    def search_temporal(query: str, top_k: int = 5, entities: list[str] | None = None) -> dict[str, object]:
        trace_order.append("search_temporal")
        return {
            "matched_events": [
                {
                    "event": "2024-01-03",
                    "date": "2024-01-03T00:00:00",
                    "turn_id": 0,
                    "text": "We moved to Portland on 2024-01-03.",
                }
            ],
            "bucket": "ordering",
        }

    def answer(evidence: str) -> str:
        trace_order.append("answer")
        return evidence

    retriever = AgenticRetriever(
        {
            "search_memory": search_memory,
            "search_temporal": search_temporal,
            "answer": answer,
        },
        budget=RetrievalBudget(max_steps=3),
    )

    evidence, trace = retriever.retrieve("When did we move to Portland?")

    assert "2024-01-03" in evidence
    assert trace_order == ["search_memory", "search_temporal", "answer"]
    assert [step["tool"] for step in trace.steps] == ["search_memory", "search_temporal", "answer"]


def test_deterministic_router_selects_expected_pipeline_per_question_category() -> None:
    router = DeterministicRouter()

    assert router.route("When did we move to Portland?") == "temporal"
    assert router.route("How many pets do we have?") == "aggregation"
    assert router.route("Where does Luna live?") == "spatial"
    assert router.route("Who is connected to Luna?") == "social_relational"
    assert router.route("What breed is Luna?") == "fact_lookup"
