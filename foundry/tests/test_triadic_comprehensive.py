"""Comprehensive tests for triadic memory in Hermes Foundry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from foundry.core import Foundry, FoundryConfig
from foundry.retrieval import search_memory, search_memory_hybrid
from foundry.storage import BM25Index, FoundryStore, RawChunk, RawChunkStore, TriadicMemory


TRIADIC_PROPERTY_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _safe_text_strategy(min_size: int = 0, max_size: int = 24) -> st.SearchStrategy[str]:
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=min_size,
        max_size=max_size,
    )


def _triple_strategy() -> st.SearchStrategy[tuple[str, str, str]]:
    atom = _safe_text_strategy(min_size=0, max_size=16)
    return st.tuples(atom, atom, atom)


def _build_chain_memory(length: int, *, seed: int = 42, decay: float = 0.5) -> tuple[TriadicMemory, list[str], float]:
    memory = TriadicMemory(n=2048, p=8, seed=seed)
    texts: list[str] = []
    texts.append("T0")
    memory.store_fact("T0", [("user", "has", "luna")])
    for idx in range(1, length + 1):
        text = f"T{idx}"
        texts.append(text)
        if idx == 1:
            triple = ("user", "has", "bridge_1")
        else:
            triple = (f"bridge_{idx - 1}", f"rel_{idx}", f"bridge_{idx}")
        memory.store_fact(text, [triple])
    memory.flush()
    return memory, texts, decay


def _build_foundry_for_bridge() -> Foundry:
    raw = "\n".join(
        [
            "User: I have a cat named Luna.",
            "User: I have a dog named Mochi.",
            "Assistant: Mochi is a playful dog.",
        ]
    )
    foundry = Foundry(FoundryConfig())
    foundry.ingest(raw)
    return foundry


# Invariant tests


def test_invariant_tensor_projection_symmetry_holds_for_every_entry() -> None:
    memory = TriadicMemory(n=128, p=6, seed=7)
    memory.store_fact("A", [("user", "has", "luna")])
    memory.store_fact("B", [("luna", "likes", "tuna")])
    memory.flush()

    for i, j, k in memory._entries:
        assert k in memory._xy_to_z[(i, j)]
        assert i in memory._yz_to_x[(j, k)]


def test_invariant_concept_index_consistency_matches_text_triples() -> None:
    memory = TriadicMemory(n=128, p=6, seed=11)
    rows = {
        "T1": [("user", "has", "luna")],
        "T2": [("luna", "type", "cat"), ("user", "has", "mochi")],
    }
    for text, triples in rows.items():
        memory.store_fact(text, triples)
    memory.flush()

    expected: dict[str, set[str]] = {}
    for text, triples in rows.items():
        for s, p, o in triples:
            for concept in (s, p, o):
                expected.setdefault(memory._normalize_concept(concept), set()).add(text)
    assert memory._concept_to_texts == expected


def test_invariant_store_load_roundtrip_identity_for_triadic_and_store(tmp_path: Path) -> None:
    store = FoundryStore()
    store.derived.metadata["gists"] = ["stable"]
    store.triadic.store_fact("T1", [("user", "has", "luna")])
    store.triadic.store_fact("T2", [("luna", "type", "cat")])
    store.triadic.flush()

    triadic_payload = store.triadic.to_dict()
    restored_triadic = TriadicMemory.from_dict(triadic_payload)
    assert restored_triadic.to_dict() == triadic_payload

    path = tmp_path / "store"
    store.save(str(path))
    loaded = FoundryStore.load(str(path))
    assert loaded.triadic.to_dict() == store.triadic.to_dict()
    assert loaded.derived.metadata == {"gists": ["stable"]}


# Fuzz tests


@pytest.mark.parametrize(
    ("text", "triples"),
    [
        ("", []),
        ("", [("", "", "")]),
        ("emoji \U0001f600 text", [("user", "has", "\U0001f431")]),
        ("symbols !@#$%^&*()", [("!@#", "$%^", "&*(")]),
        ("long " + ("x" * 2048), [("subject", "predicate", "object")]),
        (None, [(None, None, None)]),
        ("mixed unicode café 東京", [("café", "located_in", "東京")]),
    ],
)
def test_fuzz_store_fact_accepts_diverse_text_and_value_shapes(text: object, triples: list[tuple[object, object, object]]) -> None:
    memory = TriadicMemory(n=256, p=6, seed=19)
    memory.store_fact(text, triples)  # type: ignore[arg-type]
    memory.flush()

    assert memory.coverage()[1] == 1
    assert isinstance(memory.tensor_entries, int)
    assert memory.query_direct(str(text or "")) == memory.query_direct(str(text or ""))


def test_fuzz_store_fact_rejects_none_triple_collection() -> None:
    memory = TriadicMemory()
    with pytest.raises(TypeError):
        memory.store_fact("broken", None)  # type: ignore[arg-type]


@TRIADIC_PROPERTY_SETTINGS
@given(
    text=_safe_text_strategy(min_size=0, max_size=128),
    subject=_safe_text_strategy(min_size=0, max_size=64),
    predicate=_safe_text_strategy(min_size=0, max_size=64),
    obj=_safe_text_strategy(min_size=0, max_size=64),
)
def test_fuzz_random_unicode_and_special_character_inputs_do_not_break_roundtrip(
    text: str,
    subject: str,
    predicate: str,
    obj: str,
) -> None:
    memory = TriadicMemory(n=256, p=6, seed=23)
    memory.store_fact(text, [(subject, predicate, obj)])
    restored = TriadicMemory.from_dict(memory.to_dict())

    assert restored.coverage() == memory.coverage()
    assert restored.tensor_entries == memory.tensor_entries


# Behavioral tests


def test_behavior_ingest_then_triadic_search_finds_bridging_entities() -> None:
    foundry = _build_foundry_for_bridge()

    results = search_memory(foundry._store, "Luna", top_k=5)
    texts = [row["text"] for row in results]

    assert "I have a cat named Luna." in texts
    assert "I have a dog named Mochi." in texts
    assert "Mochi is a playful dog." in texts


def test_behavior_multi_hop_association_chain_reaches_distant_fact() -> None:
    memory, texts, decay = _build_chain_memory(4, seed=29, decay=0.5)

    expanded = memory.expand([texts[0]], top_k=10, max_hops=4, decay=decay)

    assert "T1" in expanded
    assert "T4" in expanded
    assert expanded.index("T1") < expanded.index("T4")


def test_behavior_reranker_can_promote_triadic_expansion_candidates() -> None:
    foundry = _build_foundry_for_bridge()

    def reranker(_query: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return sorted(rows, key=lambda row: "Mochi" not in str(row["text"]))

    results = search_memory(foundry._store, "Luna", top_k=3, reranker=reranker)

    assert results[0]["text"] in {"I have a dog named Mochi.", "Mochi is a playful dog."}


# Property-based tests


@TRIADIC_PROPERTY_SETTINGS
@given(triple=_triple_strategy())
def test_property_arbitrary_single_triple_is_retrievable(triple: tuple[str, str, str]) -> None:
    subject, predicate, obj = triple
    memory = TriadicMemory(n=512, p=6, seed=31)
    memory.store_fact("fact", [triple])
    memory.flush()

    assert memory.overlap(memory.recall_z(subject, predicate), memory._get_or_create_sdr(obj)) == 1.0
    assert memory.overlap(memory.recall_x(predicate, obj), memory._get_or_create_sdr(subject)) == 1.0


@TRIADIC_PROPERTY_SETTINGS
@given(length=st.integers(min_value=3, max_value=5))
def test_property_recall_strength_monotonically_decreases_with_distance(length: int) -> None:
    memory, texts, decay = _build_chain_memory(length, seed=37, decay=0.5)

    expanded = memory.expand([texts[0]], top_k=length + 2, max_hops=length + 1, decay=decay)
    assert expanded[-1] == texts[-1]


@TRIADIC_PROPERTY_SETTINGS
@given(triples=st.lists(_triple_strategy(), min_size=1, max_size=12))
def test_property_tensor_sparsity_is_bounded_by_unique_triples(triples: list[tuple[str, str, str]]) -> None:
    memory = TriadicMemory(n=512, p=4, seed=41)
    for idx, triple in enumerate(triples):
        memory.store_fact(f"T{idx}", [triple])
    memory.flush()

    unique_triples = len(set(triples))
    assert memory.tensor_entries <= unique_triples * (memory.p ** 3)


# Unit tests


def test_unit_store_and_flush_are_idempotent_on_second_flush() -> None:
    memory = TriadicMemory(n=64, p=4, seed=43)
    memory.store("user", "has", "luna")
    first_entries = memory.tensor_entries
    memory.flush()
    second_entries = memory.tensor_entries

    assert first_entries == second_entries == 64


def test_unit_recall_methods_return_empty_sets_for_unseen_axes() -> None:
    memory = TriadicMemory(n=64, p=4, seed=47)
    assert memory.recall_z("missing", "pair") == set()
    assert memory.recall_x("missing", "pair") == set()


def test_unit_add_coref_clusters_and_verb_mappings_affect_query_direct() -> None:
    memory = TriadicMemory(n=128, p=6, seed=53)
    memory.store_fact("User has Luna.", [("user", "has", "luna")])
    memory.add_coref_clusters([["Luna", "the cat", "she"]])
    memory.add_verb_mappings([("have", "has")])
    memory.flush()

    assert memory.query_direct("the cat") == ["User has Luna."]
    assert memory.query_direct("have") == ["User has Luna."]


def test_unit_expand_query_direct_coverage_and_tensor_entries_public_surface() -> None:
    memory = TriadicMemory(n=128, p=6, seed=59)
    memory.store_fact("T1", [("user", "has", "luna")])
    memory.store_fact("T2", [("user", "has", "mochi")])
    memory.flush()

    assert memory.expand(["T1"], top_k=5, max_hops=1) == ["T2"]
    assert memory.query_direct("luna") == ["T1", "T2"]
    assert memory.coverage() == (2, 2)
    assert memory.tensor_entries > 0


def test_unit_persist_and_load_through_foundry_store(tmp_path: Path) -> None:
    store = FoundryStore()
    store.triadic.store_fact("T1", [("user", "has", "luna")])
    store.triadic.flush()

    path = tmp_path / "store"
    store.save(str(path))
    loaded = FoundryStore.load(str(path))

    assert loaded.triadic.to_dict() == store.triadic.to_dict()


# Control flow tests


def test_control_flow_empty_tensor_single_entry_duplicate_triples() -> None:
    memory = TriadicMemory(n=64, p=4, seed=61)
    assert memory.expand(["none"]) == []

    memory.store_fact("T1", [("user", "has", "luna"), ("user", "has", "luna")])
    memory.flush()
    first_entries = memory.tensor_entries
    memory.store_fact("T1", [("user", "has", "luna")])
    memory.flush()

    assert memory.tensor_entries == first_entries
    assert memory.recall_z("user", "has") == memory._get_or_create_sdr("luna")


def test_control_flow_concurrent_store_and_recall_does_not_raise() -> None:
    memory = TriadicMemory(n=256, p=6, seed=67)
    errors: list[BaseException] = []

    def writer(idx: int) -> None:
        try:
            memory.store_fact(f"T{idx}", [(f"user_{idx}", "has", f"pet_{idx}")])
            memory.flush()
        except BaseException as exc:  # pragma: no cover - captured for assertion
            errors.append(exc)

    def reader(idx: int) -> None:
        try:
            memory.recall_z(f"user_{idx}", "has")
            memory.query_direct(f"pet_{idx}")
        except BaseException as exc:  # pragma: no cover - captured for assertion
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for idx in range(8):
            executor.submit(writer, idx)
            executor.submit(reader, idx)

    assert errors == []


# Adversarial tests


def test_adversarial_collision_prone_small_tensor_does_not_break() -> None:
    memory = TriadicMemory(n=8, p=1, seed=71)
    for idx in range(12):
        memory.store_fact(f"T{idx}", [(f"subject_{idx}", "edge", f"object_{idx}")])
    memory.flush()

    assert memory.tensor_entries <= 12
    assert isinstance(memory.query_direct("subject_0"), list)
    assert memory.recall_z("subject_0", "edge") <= set(range(memory.n))


def test_adversarial_large_dimension_tensor_still_persists_and_recalls() -> None:
    memory = TriadicMemory(n=4096, p=1, seed=73)
    memory.store_fact("T1", [("user", "has", "luna")])
    memory.flush()

    restored = TriadicMemory.from_dict(memory.to_dict())
    assert restored.tensor_entries == 1
    assert restored.recall_z("user", "has") == restored._get_or_create_sdr("luna")


def test_adversarial_overlapping_subject_object_names_do_not_confuse_indexes() -> None:
    memory = TriadicMemory(n=256, p=6, seed=79)
    memory.store_fact("T1", [("alpha", "rel", "alpha")])
    memory.store_fact("T2", [("rel", "alpha", "rel")])
    memory.flush()

    assert "T1" in memory._concept_to_texts["alpha"]
    assert "T2" in memory._concept_to_texts["rel"]
    assert memory.query_direct("alpha")


# Integration tests


def test_integration_core_ingest_to_retrieval_expansion_path() -> None:
    foundry = _build_foundry_for_bridge()
    bm25 = foundry._store.bm25_chunks

    results = search_memory_hybrid(
        "Luna",
        bm25,
        foundry._store.vectors,
        chunk_store=foundry._store.chunks,
        triadic=foundry._store.triadic,
        top_k=5,
    )

    texts = [row["text"] for row in results]
    assert "I have a dog named Mochi." in texts
    assert "Mochi is a playful dog." in texts


def test_integration_loaded_store_preserves_search_memory_hybrid_behavior(tmp_path: Path) -> None:
    foundry = _build_foundry_for_bridge()
    path = tmp_path / "store"
    foundry.save(str(path))
    loaded = FoundryStore.load(str(path))

    results = search_memory(loaded, "Luna", top_k=5)

    assert any(row["text"] == "I have a dog named Mochi." for row in results)


# Snapshot regression tests


def test_snapshot_regression_known_triadic_outputs_are_stable() -> None:
    memory = TriadicMemory(n=128, p=6, seed=83)
    memory.store_fact("T1", [("user", "has", "luna")])
    memory.store_fact("T2", [("user", "has", "mochi")])
    memory.store_fact("T3", [("mochi", "likes", "tuna")])
    memory.add_coref_clusters([["Luna", "the cat", "she"]])
    memory.add_verb_mappings([("have", "has")])
    memory.flush()

    snapshot = {
        "coverage": memory.coverage(),
        "tensor_entries": memory.tensor_entries,
        "expand": memory.expand(["T1"], top_k=5, max_hops=2, decay=0.5),
        "query_direct": memory.query_direct("the cat", top_k=5),
    }

    assert snapshot == {
        "coverage": (3, 3),
        "tensor_entries": 648,
        "expand": ["T3", "T2"],
        "query_direct": ["T1", "T2", "T3"],
    }


# Idempotency tests


def test_idempotency_storing_same_triple_twice_does_not_corrupt_tensor_state() -> None:
    memory = TriadicMemory(n=128, p=6, seed=89)
    memory.store_fact("T1", [("user", "has", "luna")])
    memory.flush()
    first_tensor_entries = memory.tensor_entries
    first_recall = memory.recall_z("user", "has")

    memory.store_fact("T1", [("user", "has", "luna")])
    memory.flush()

    assert memory.tensor_entries == first_tensor_entries
    assert memory.recall_z("user", "has") == first_recall
    assert memory.query_direct("luna") == ["T1"]


# Serialization tests


def test_serialization_to_dict_from_dict_preserves_exact_triadic_payload() -> None:
    memory = TriadicMemory(n=128, p=6, seed=97)
    memory.store_fact("T1", [("user", "has", "luna")])
    memory.store_fact("T2", [("luna", "type", "cat")])
    memory.add_coref_clusters([["Luna", "the cat", "she"]])
    memory.flush()

    payload = memory.to_dict()
    restored_payload = TriadicMemory.from_dict(payload).to_dict()

    assert restored_payload == payload


def test_serialization_file_format_preserves_exact_triadic_state(tmp_path: Path) -> None:
    store = FoundryStore()
    store.triadic.store_fact("T1", [("user", "has", "luna")])
    store.triadic.store_fact("T2", [("luna", "type", "cat")])
    store.triadic.flush()

    path = tmp_path / "store"
    store.save(str(path))
    loaded = FoundryStore.load(str(path))
    manifest = json.loads((path / "manifest.json").read_text())

    assert loaded.triadic.to_dict() == store.triadic.to_dict()
    assert manifest["schema_version"] == FoundryStore.SCHEMA_VERSION
