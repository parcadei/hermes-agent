"""Comprehensive tests for Engram query-time conflict resolution."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import random

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

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
    timestamp: datetime | None = None,
    superseded: bool = False,
    superseded_by: str | None = None,
) -> Fact:
    chunk = store.chunks._chunks.get(chunk_idx)
    if chunk is None:
        chunk = RawChunk(
            chunk_idx=chunk_idx,
            text=text,
            turn_idx=turn_idx,
            role="user",
            entities=[entity],
        )
        store.chunks.store(chunk)
        store.bm25_chunks.add(chunk_idx, text)
    elif entity not in chunk.entities:
        chunk.entities.append(entity)

    node = store.entity_graph.ensure_entity(entity)
    node.mention_turns.add(turn_idx)
    node.mention_chunks.add(chunk_idx)
    node.aliases.add(entity)
    fact = Fact(
        id=f"{entity}-{predicate}-{chunk_idx}-{len(node.facts.get(predicate, []))}",
        subject=entity,
        predicate=predicate,
        object=obj,
        object_type=PropertyType.STRING,
        turn_idx=turn_idx,
        chunk_idx=chunk_idx,
        timestamp=timestamp,
        source_text=text,
        superseded=superseded,
        superseded_by=superseded_by,
    )
    node.add_fact(fact)
    return fact


def _result(chunk_idx: int, turn_idx: int, text: str, score: float = 1.0) -> dict[str, object]:
    return {"turn_id": turn_idx, "chunk_id": chunk_idx, "text": text, "score": score}


def _baseline_detector() -> tuple[FoundryStore, ConflictDetector, list[dict[str, object]]]:
    store = FoundryStore()
    seattle = _add_chunk_fact(
        store,
        chunk_idx=0,
        turn_idx=0,
        text="User: I live in Seattle.",
        entity="user",
        predicate="lives_in",
        obj="Seattle",
    )
    _ = seattle
    _add_chunk_fact(
        store,
        chunk_idx=1,
        turn_idx=1,
        text="User: I live in Portland.",
        entity="user",
        predicate="lives_in",
        obj="Portland",
    )
    results = [
        _result(0, 0, "I live in Seattle.", 0.9),
        _result(1, 1, "I live in Portland.", 0.8),
    ]
    return store, ConflictDetector(store), results


def _property_store_and_results(rows: list[tuple[str, str, str]]) -> tuple[FoundryStore, list[dict[str, object]]]:
    store = FoundryStore()
    results: list[dict[str, object]] = []
    for chunk_idx, (subject, predicate, obj) in enumerate(rows):
        text = f"{subject} {predicate} {obj}"
        _add_chunk_fact(
            store,
            chunk_idx=chunk_idx,
            turn_idx=chunk_idx,
            text=text,
            entity=subject,
            predicate=predicate,
            obj=obj,
        )
        results.append(_result(chunk_idx, chunk_idx, text, score=1.0 / (chunk_idx + 1)))
    return store, results


# 1. Invariant tests


def test_invariant_cr_never_adds_chunks_and_returns_subset() -> None:
    store, detector, results = _baseline_detector()
    _ = store

    filtered = detector.filter_conflicts(results)

    input_ids = {result["chunk_id"] for result in results}
    output_ids = {result["chunk_id"] for result in filtered}
    assert output_ids <= input_ids
    assert len(filtered) <= len(results)


def test_invariant_non_conflicting_chunks_always_preserved() -> None:
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
    results = [_result(0, 0, "I live in Seattle."), _result(1, 1, "I work at Acme.")]

    assert ConflictDetector(store).filter_conflicts(results) == results


def test_invariant_newest_fact_always_wins() -> None:
    store = FoundryStore()
    for turn_idx, city in enumerate(["Seattle", "Denver", "Portland"]):
        _add_chunk_fact(
            store,
            chunk_idx=turn_idx,
            turn_idx=turn_idx,
            text=f"User: I live in {city}.",
            entity="user",
            predicate="lives_in",
            obj=city,
        )
    results = [_result(0, 0, "I live in Seattle."), _result(1, 1, "I live in Denver."), _result(2, 2, "I live in Portland.")]

    filtered = ConflictDetector(store).filter_conflicts(results)

    assert filtered == [_result(2, 2, "I live in Portland.")]


# 2. Fuzz tests


def test_fuzz_empty_result_set_is_passthrough() -> None:
    detector = ConflictDetector(FoundryStore())
    assert detector.filter_conflicts([]) == []
    assert detector.detect([]) == []
    assert detector.cosine_dedup([]) == []


def test_fuzz_single_chunk_is_passthrough() -> None:
    store = FoundryStore()
    detector = ConflictDetector(store)
    single = [_result(7, 7, "Only one chunk.")]
    assert detector.filter_conflicts(single) == single


def test_fuzz_chunks_with_no_entities_or_facts_are_preserved() -> None:
    store = FoundryStore()
    store.chunks.store(RawChunk(chunk_idx=2, text="No entities here.", turn_idx=2, role="user", entities=[]))
    detector = ConflictDetector(store)
    single = [_result(2, 2, "No entities here.")]
    assert detector.filter_conflicts(single) == single


def test_fuzz_random_text_chunks_without_facts_do_not_crash() -> None:
    store = FoundryStore()
    rng = random.Random(7)
    results: list[dict[str, object]] = []
    for idx in range(8):
        text = "".join(rng.choice("abcXYZ123!? ") for _ in range(24))
        store.chunks.store(RawChunk(chunk_idx=idx, text=text, turn_idx=idx, role="user", entities=[]))
        store.bm25_chunks.add(idx, text)
        results.append(_result(idx, idx, text, score=1.0 / (idx + 1)))

    filtered = ConflictDetector(store).filter_conflicts(results)

    assert filtered == results


def test_fuzz_unicode_predicates_conflict_correctly() -> None:
    store = FoundryStore()
    _add_chunk_fact(
        store,
        chunk_idx=0,
        turn_idx=0,
        text="User: 住所 は 東京.",
        entity="user",
        predicate="住所",
        obj="東京",
    )
    _add_chunk_fact(
        store,
        chunk_idx=1,
        turn_idx=1,
        text="User: 住所 は 大阪.",
        entity="user",
        predicate="住所",
        obj="大阪",
    )
    results = [_result(0, 0, "住所 は 東京."), _result(1, 1, "住所 は 大阪.")]

    assert ConflictDetector(store).filter_conflicts(results) == [_result(1, 1, "住所 は 大阪.")]


# 3. Behavioral tests


def test_behavioral_contradictory_facts_detected_and_older_removed() -> None:
    _, detector, results = _baseline_detector()
    assert detector.detect(results) == [("user", "lives_in", "Seattle", "Portland")]
    assert detector.filter_conflicts(results) == [_result(1, 1, "I live in Portland.", 0.8)]


def test_behavioral_non_contradictory_facts_can_coexist() -> None:
    store = FoundryStore()
    _add_chunk_fact(store, chunk_idx=0, turn_idx=0, text="User: I own a bike.", entity="user", predicate="owns", obj="bike")
    _add_chunk_fact(store, chunk_idx=1, turn_idx=1, text="User: I wear a helmet.", entity="user", predicate="wears", obj="helmet")
    results = [_result(0, 0, "I own a bike."), _result(1, 1, "I wear a helmet.")]

    assert ConflictDetector(store).filter_conflicts(results) == results


def test_behavioral_temporal_ordering_respects_turn_idx_not_timestamp() -> None:
    store = FoundryStore()
    later_time = datetime(2024, 3, 1)
    earlier_time = datetime(2024, 1, 1)
    _add_chunk_fact(
        store,
        chunk_idx=0,
        turn_idx=0,
        text="User: I live in Seattle.",
        entity="user",
        predicate="lives_in",
        obj="Seattle",
        timestamp=later_time,
    )
    _add_chunk_fact(
        store,
        chunk_idx=1,
        turn_idx=1,
        text="User: I live in Portland.",
        entity="user",
        predicate="lives_in",
        obj="Portland",
        timestamp=earlier_time,
    )
    results = [_result(0, 0, "I live in Seattle."), _result(1, 1, "I live in Portland.")]

    assert ConflictDetector(store).filter_conflicts(results) == [_result(1, 1, "I live in Portland.")]


# 4. Property-based tests


@st.composite
def _rows_and_duplicates(draw: st.DrawFn) -> tuple[list[tuple[str, str, str]], list[int]]:
    row_count = draw(st.integers(min_value=0, max_value=8))
    rows = draw(
        st.lists(
            st.tuples(
                st.sampled_from(["user", "pet", "bike"]),
                st.sampled_from(["lives_in", "favorite_color", "owns", "住所"]),
                st.sampled_from(["Seattle", "Portland", "blue", "green", "bike", "東京"]),
            ),
            min_size=row_count,
            max_size=row_count,
        )
    )
    picks = draw(st.lists(st.integers(min_value=0, max_value=max(row_count - 1, 0)), min_size=0, max_size=max(row_count * 2, 1)))
    return rows, picks


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(data=_rows_and_duplicates())
def test_property_output_is_subset_of_input(data: tuple[list[tuple[str, str, str]], list[int]]) -> None:
    rows, picks = data
    store, canonical_results = _property_store_and_results(rows)
    input_results = [canonical_results[index] for index in picks if 0 <= index < len(canonical_results)]
    filtered = ConflictDetector(store).filter_conflicts(input_results)

    assert all(result in input_results for result in filtered)


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(data=_rows_and_duplicates())
def test_property_output_length_never_exceeds_input(data: tuple[list[tuple[str, str, str]], list[int]]) -> None:
    rows, picks = data
    store, canonical_results = _property_store_and_results(rows)
    input_results = [canonical_results[index] for index in picks if 0 <= index < len(canonical_results)]
    filtered = ConflictDetector(store).filter_conflicts(input_results)

    assert len(filtered) <= len(input_results)


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(data=_rows_and_duplicates())
def test_property_output_has_no_duplicate_chunks(data: tuple[list[tuple[str, str, str]], list[int]]) -> None:
    rows, picks = data
    store, canonical_results = _property_store_and_results(rows)
    input_results = [canonical_results[index] for index in picks if 0 <= index < len(canonical_results)]
    filtered = ConflictDetector(store).filter_conflicts(input_results)
    chunk_ids = [int(result["chunk_id"]) for result in filtered]

    assert len(chunk_ids) == len(set(chunk_ids))


# 5. Unit tests


def test_unit_detect_alias_matches_detect_conflicts() -> None:
    _, detector, results = _baseline_detector()
    assert detector.detect(results) == detector.detect_conflicts(results)


def test_unit_group_by_subject_predicate_groups_correctly() -> None:
    store = FoundryStore()
    fact_a = _add_chunk_fact(store, chunk_idx=0, turn_idx=0, text="User: I live in Seattle.", entity="user", predicate="lives_in", obj="Seattle")
    fact_b = _add_chunk_fact(store, chunk_idx=1, turn_idx=1, text="User: I live in Portland.", entity="user", predicate="lives_in", obj="Portland")
    _add_chunk_fact(store, chunk_idx=2, turn_idx=2, text="User: I work at Acme.", entity="user", predicate="works_at", obj="Acme")
    detector = ConflictDetector(store)
    grouped = detector.group_by_subject_predicate([
        _result(0, 0, "I live in Seattle."),
        _result(1, 1, "I live in Portland."),
        _result(2, 2, "I work at Acme."),
    ])

    assert Counter(grouped.keys()) == Counter({("user", "lives_in"): 1, ("user", "works_at"): 1})
    assert grouped[("user", "lives_in")] == [fact_a, fact_b]


def test_unit_is_conflicting_detects_object_variation() -> None:
    store = FoundryStore()
    detector = ConflictDetector(store)
    same = [
        Fact(id="a", subject="user", predicate="lives_in", object="Seattle", object_type=PropertyType.STRING, turn_idx=0),
        Fact(id="b", subject="user", predicate="lives_in", object="Seattle", object_type=PropertyType.STRING, turn_idx=1),
    ]
    different = [
        Fact(id="c", subject="user", predicate="lives_in", object="Seattle", object_type=PropertyType.STRING, turn_idx=0),
        Fact(id="d", subject="user", predicate="lives_in", object="Portland", object_type=PropertyType.STRING, turn_idx=1),
    ]

    assert detector._is_conflicting(same) is False
    assert detector._is_conflicting(different) is True


def test_unit_every_public_method_is_callable() -> None:
    store, detector, results = _baseline_detector()
    _ = store
    assert detector.filter_conflicts(results)
    assert detector.detect(results)
    assert detector.detect_conflicts(results)
    assert detector.group_by_subject_predicate(results)
    assert detector.cosine_dedup(results)


# 6. Control flow tests


def test_control_flow_no_conflicts_found_is_passthrough() -> None:
    store = FoundryStore()
    _add_chunk_fact(store, chunk_idx=0, turn_idx=0, text="User: I own a bike.", entity="user", predicate="owns", obj="bike")
    detector = ConflictDetector(store)
    results = [_result(0, 0, "I own a bike.")]

    assert detector.filter_conflicts(results) == results


def test_control_flow_all_conflicting_versions_reduce_to_latest() -> None:
    store = FoundryStore()
    for turn_idx, city in enumerate(["Seattle", "Denver", "Portland", "London"]):
        _add_chunk_fact(store, chunk_idx=turn_idx, turn_idx=turn_idx, text=f"User: I live in {city}.", entity="user", predicate="lives_in", obj=city)
    results = [_result(idx, idx, f"I live in {city}.") for idx, city in enumerate(["Seattle", "Denver", "Portland", "London"])]

    assert ConflictDetector(store).filter_conflicts(results) == [_result(3, 3, "I live in London.")]


def test_control_flow_mixed_chunks_remove_only_stale_conflict_containers() -> None:
    store = FoundryStore()
    _add_chunk_fact(store, chunk_idx=0, turn_idx=0, text="User: I live in Seattle and own a bike.", entity="user", predicate="lives_in", obj="Seattle")
    _add_chunk_fact(store, chunk_idx=0, turn_idx=0, text="User: I live in Seattle and own a bike.", entity="user", predicate="owns", obj="bike")
    _add_chunk_fact(store, chunk_idx=1, turn_idx=1, text="User: I live in Portland.", entity="user", predicate="lives_in", obj="Portland")
    _add_chunk_fact(store, chunk_idx=2, turn_idx=2, text="User: I own a helmet.", entity="user", predicate="owns", obj="helmet")
    results = [_result(0, 0, "I live in Seattle and own a bike."), _result(1, 1, "I live in Portland."), _result(2, 2, "I own a helmet.")]

    assert ConflictDetector(store).filter_conflicts(results) == [_result(1, 1, "I live in Portland."), _result(2, 2, "I own a helmet.")]


# 7. Adversarial tests


def test_adversarial_identical_timestamps_use_turn_idx_tiebreak() -> None:
    store = FoundryStore()
    stamp = datetime(2024, 1, 1)
    _add_chunk_fact(store, chunk_idx=0, turn_idx=0, text="User: I live in Seattle.", entity="user", predicate="lives_in", obj="Seattle", timestamp=stamp)
    _add_chunk_fact(store, chunk_idx=1, turn_idx=1, text="User: I live in Portland.", entity="user", predicate="lives_in", obj="Portland", timestamp=stamp)
    results = [_result(0, 0, "I live in Seattle."), _result(1, 1, "I live in Portland.")]

    assert ConflictDetector(store).filter_conflicts(results) == [_result(1, 1, "I live in Portland.")]


def test_adversarial_circular_supersession_chain_does_not_break_result_scoped_cr() -> None:
    store = FoundryStore()
    fact_a = _add_chunk_fact(store, chunk_idx=0, turn_idx=0, text="User: I live in Seattle.", entity="user", predicate="lives_in", obj="Seattle", superseded=True, superseded_by="b")
    fact_b = _add_chunk_fact(store, chunk_idx=1, turn_idx=1, text="User: I live in Portland.", entity="user", predicate="lives_in", obj="Portland", superseded=True, superseded_by="a")
    fact_a.superseded_by = fact_b.id
    fact_b.superseded_by = fact_a.id
    results = [_result(0, 0, "I live in Seattle."), _result(1, 1, "I live in Portland.")]

    assert ConflictDetector(store).filter_conflicts(results) == [_result(1, 1, "I live in Portland.")]


def test_adversarial_predicates_that_are_substrings_do_not_conflict() -> None:
    store = FoundryStore()
    _add_chunk_fact(store, chunk_idx=0, turn_idx=0, text="User: I live.", entity="user", predicate="live", obj="yes")
    _add_chunk_fact(store, chunk_idx=1, turn_idx=1, text="User: I live in Portland.", entity="user", predicate="lives_in", obj="Portland")
    results = [_result(0, 0, "I live."), _result(1, 1, "I live in Portland.")]

    assert ConflictDetector(store).filter_conflicts(results) == results


# 8. Integration tests


def test_integration_search_memory_invokes_conflict_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest("User: I live in Seattle.\nUser: I live in Portland.")
    called: list[int] = []
    original = ConflictDetector.filter_conflicts

    def wrapped(self: ConflictDetector, results: list[dict]) -> list[dict]:
        called.append(len(results))
        return original(self, results)

    monkeypatch.setattr(ConflictDetector, "filter_conflicts", wrapped)
    filtered = search_memory(foundry._store, "Seattle Portland live", top_k=5)

    assert called
    assert all("Seattle" not in row["text"] for row in filtered)


def test_integration_search_memory_returns_clean_results_after_cr() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest("User: I live in Seattle.\nUser: I live in Portland.")

    filtered = search_memory(foundry._store, "Seattle Portland live", top_k=5)

    assert len(filtered) == 1
    assert filtered[0]["turn_id"] == 1
    assert filtered[0]["chunk_id"] == 1
    assert filtered[0]["text"] == "I live in Portland."


# 9. Snapshot regression


def test_snapshot_regression_known_location_conflict() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest("User: I live in Seattle.\nUser: I live in Portland.")

    snapshot = search_memory(foundry._store, "Seattle Portland live", top_k=5)

    assert snapshot == [
        {
            "turn_id": 1,
            "chunk_id": 1,
            "text": "I live in Portland.",
            "score": 0.016129032258064516,
        }
    ]


def test_snapshot_regression_known_mixed_chunk_conflict() -> None:
    store = FoundryStore()
    _add_chunk_fact(store, chunk_idx=0, turn_idx=0, text="User: I live in Seattle and my favorite color is blue.", entity="user", predicate="lives_in", obj="Seattle")
    _add_chunk_fact(store, chunk_idx=0, turn_idx=0, text="User: I live in Seattle and my favorite color is blue.", entity="user", predicate="favorite_color", obj="blue")
    _add_chunk_fact(store, chunk_idx=1, turn_idx=1, text="User: I live in Portland.", entity="user", predicate="lives_in", obj="Portland")
    results = [_result(0, 0, "I live in Seattle and my favorite color is blue.", 0.9), _result(1, 1, "I live in Portland.", 0.8)]

    assert ConflictDetector(store).filter_conflicts(results) == [
        {"turn_id": 1, "chunk_id": 1, "text": "I live in Portland.", "score": 0.8}
    ]


# 10. Idempotency tests


def test_idempotency_filter_conflicts_is_idempotent() -> None:
    _, detector, results = _baseline_detector()
    once = detector.filter_conflicts(results)
    twice = detector.filter_conflicts(once)

    assert once == twice


def test_idempotency_detect_conflicts_is_idempotent() -> None:
    _, detector, results = _baseline_detector()

    assert detector.detect(results) == detector.detect(results)


# 11. Scope tests


def test_scope_cr_only_considers_retrieved_results_not_store_wide_state() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest("User: I live in Seattle.\nUser: I live in Portland.")
    detector = ConflictDetector(foundry._store)
    seattle_only = [_result(0, 0, "I live in Seattle.", 0.9)]

    assert detector.filter_conflicts(seattle_only) == seattle_only


def test_scope_seattle_query_does_not_disappear_because_portland_exists_elsewhere() -> None:
    foundry = Foundry(FoundryConfig())
    foundry.ingest("User: I live in Seattle.\nUser: I live in Portland.")

    seattle_results = search_memory(foundry._store, "Seattle", top_k=3)

    assert len(seattle_results) == 1
    assert seattle_results[0]["chunk_id"] == 0
    assert "Seattle" in seattle_results[0]["text"]
