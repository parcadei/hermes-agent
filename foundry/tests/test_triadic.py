"""Triadic memory tests for Hermes Foundry."""

from __future__ import annotations

import json

from foundry.core import Foundry, FoundryConfig
from foundry.retrieval import search_memory, search_memory_hybrid
from foundry.storage import BM25Index, FoundryStore, RawChunk, RawChunkStore, SynonymDict, TriadicMemory


def test_synonym_dict_add_resolve_and_coref_pronoun_filtering() -> None:
    synonyms = SynonymDict()
    synonyms.add("kitty", "Luna")
    synonyms.add_from_coref_clusters([["Luna", "the cat", "she"], ["they", "them"]])
    synonyms.add_from_verb_mappings([("have", "has")])

    assert synonyms.resolve("kitty") == "luna"
    assert synonyms.resolve("the cat") == "luna"
    assert synonyms.resolve("have") == "has"
    assert synonyms.resolve("she") is None
    assert synonyms.resolve("they") is None


def test_triadic_memory_store_flush_recall_and_overlap() -> None:
    memory = TriadicMemory(n=64, p=4, seed=7)

    assert memory.tensor_entries == 0

    memory.store("user", "has", "luna")
    assert memory.tensor_entries == 64

    recalled_object = memory.recall_z("user", "has")
    recalled_subject = memory.recall_x("has", "luna")

    assert recalled_object == memory._codebook["luna"]
    assert recalled_subject == memory._codebook["user"]
    assert memory.overlap(recalled_object, memory._codebook["luna"]) == 1.0
    assert memory.overlap(set(), memory._codebook["luna"]) == 0.0


def test_triadic_memory_empty_single_triple_and_coverage_edges() -> None:
    memory = TriadicMemory(n=32, p=3, seed=11)

    assert memory.coverage() == (0, 0)
    assert memory.query_direct("anything") == []
    assert memory.expand(["missing"], top_k=3) == []
    assert memory._threshold(memory.rng.rand(memory.n) * 0.0) == set()

    memory.store_fact("empty text", [])
    assert memory.coverage() == (0, 1)
    assert memory.tensor_entries == 0

    memory.store_fact("User has Luna.", [("user", "has", "luna")])
    memory.flush()

    assert memory.coverage() == (1, 2)
    assert memory.tensor_entries == 27
    assert memory.recall_z("user", "has") == memory._codebook["luna"]


def test_triadic_memory_store_fact_builds_text_and_concept_indexes() -> None:
    memory = TriadicMemory(n=64, p=4, seed=3)
    triples = [("user", "has", "luna"), ("luna", "type", "cat")]

    memory.store_fact("User has Luna.", triples)
    memory.flush()

    assert memory._text_to_triples["User has Luna."] == triples
    assert memory._concept_to_texts["user"] == {"User has Luna."}
    assert memory._concept_to_texts["luna"] == {"User has Luna."}
    assert memory._concept_to_texts["cat"] == {"User has Luna."}
    assert memory.coverage() == (1, 1)


def test_triadic_memory_store_fact_merges_duplicate_raw_text_without_losing_coverage() -> None:
    memory = TriadicMemory(n=64, p=4, seed=17)
    memory.store_fact("Repeated text.", [("user", "has", "luna"), ("user", "has", "luna")])
    memory.store_fact("Repeated text.", [("user", "has", "mochi")])
    memory.flush()

    assert memory.coverage() == (2, 2)
    assert memory._text_to_triples["Repeated text."] == [
        ("user", "has", "luna"),
        ("user", "has", "mochi"),
    ]
    assert memory.query_direct("user has") == ["Repeated text."]


def test_triadic_memory_expand_supports_single_and_multi_hop_with_decay() -> None:
    memory = TriadicMemory(n=128, p=6, seed=5)
    memory.store_fact("T1", [("user", "has", "luna")])
    memory.store_fact("T2", [("user", "has", "mochi")])
    memory.store_fact("T3", [("mochi", "likes", "tuna")])
    memory.store_fact("T4", [("tuna", "type", "fish")])
    memory.flush()

    hop_one = memory.expand(["T1"], top_k=10, max_hops=1)
    hop_two = memory.expand(["T1"], top_k=10, max_hops=2)
    hop_two_decay = memory.expand(["T1"], top_k=10, max_hops=2, decay=0.5)

    assert hop_one == ["T2", "T3"]
    assert "T4" not in hop_one
    assert hop_two[:2] == ["T3", "T2"]
    assert "T4" in hop_two
    assert hop_two_decay[-1] == "T4"


def test_triadic_memory_query_direct_uses_synonyms_and_tensor_matching() -> None:
    memory = TriadicMemory(n=64, p=4, seed=9)
    memory.store_fact("User has Luna.", [("user", "has", "luna")])
    memory.store_fact("User has Mochi.", [("user", "has", "mochi")])
    memory.add_coref_clusters([["Luna", "the cat", "she"]])
    memory.add_verb_mappings([("have", "has")])
    memory.flush()

    assert memory.query_direct("the cat") == ["User has Luna.", "User has Mochi."]
    assert memory.query_direct("what do we have") == ["User has Luna.", "User has Mochi."]


def test_triadic_memory_to_dict_from_dict_round_trip_preserves_state() -> None:
    memory = TriadicMemory(n=64, p=4, seed=13)
    memory.store_fact("User has Luna.", [("user", "has", "luna")])
    memory.store_fact("User has Mochi.", [("user", "has", "mochi")])
    memory.add_coref_clusters([["Luna", "the cat", "she"]])
    memory.flush()

    restored = TriadicMemory.from_dict(memory.to_dict())

    assert restored.coverage() == memory.coverage()
    assert restored.tensor_entries == memory.tensor_entries
    assert restored._codebook == memory._codebook
    assert restored.expand(["User has Luna."], top_k=5) == memory.expand(["User has Luna."], top_k=5)
    assert restored.query_direct("the cat") == memory.query_direct("the cat")


def test_foundry_store_to_dict_from_dict_includes_triadic() -> None:
    store = FoundryStore()
    store.triadic.store_fact("User has Luna.", [("user", "has", "luna")])
    store.triadic.flush()

    restored = FoundryStore.from_dict(store.to_dict())

    assert restored.triadic.coverage() == (1, 1)
    assert restored.triadic.tensor_entries == store.triadic.tensor_entries
    assert restored.triadic.query_direct("user has") == ["User has Luna."]


def test_foundry_store_save_load_preserves_triadic_without_leaking_internal_metadata(tmp_path) -> None:
    store = FoundryStore()
    store.derived.metadata["gists"] = ["summary"]
    store.triadic.store_fact("User has Luna.", [("user", "has", "luna")])
    store.triadic.flush()

    path = tmp_path / "store"
    store.save(str(path))
    loaded = FoundryStore.load(str(path))
    derived_payload = json.loads((path / "derived.json").read_text())

    assert loaded.triadic.coverage() == (1, 1)
    assert loaded.triadic.tensor_entries == store.triadic.tensor_entries
    assert loaded.derived.metadata == {"gists": ["summary"]}
    assert "__foundry_internal_triadic__" in derived_payload["metadata"]


def test_foundry_ingest_populates_triadic_and_search_memory_expands_results() -> None:
    raw = "\n".join(
        [
            "User: I have a cat named Luna.",
            "User: I have a dog named Mochi.",
            "Assistant: Mochi is a playful dog.",
        ]
    )
    foundry = Foundry(FoundryConfig())
    report = foundry.ingest(raw)

    assert report.chunks_processed == 3
    assert foundry._store.triadic.coverage() == (3, 3)
    assert foundry._store.triadic.tensor_entries > 0

    results = search_memory(foundry._store, "Luna", top_k=5)
    texts = [row["text"] for row in results]

    assert texts[0] == "I have a cat named Luna."
    assert "I have a dog named Mochi." in texts
    assert "Mochi is a playful dog." in texts


def test_foundry_ingest_duplicate_chunk_texts_preserves_total_triadic_fact_count() -> None:
    raw = "\n".join(
        [
            "User: I have a cat named Luna.",
            "User: I have a cat named Luna.",
            "User: I have a dog named Mochi.",
            "Assistant: Mochi is a playful dog.",
        ]
    )
    foundry = Foundry(FoundryConfig())
    foundry.ingest(raw)

    assert foundry._store.triadic.coverage() == (4, 4)
    assert ("user", "has", "Luna") in foundry._store.triadic._text_to_triples["I have a cat named Luna."]


def test_search_memory_hybrid_uses_chunk_store_text_when_available() -> None:
    bm25 = BM25Index()
    bm25.add(0, "lookup-token")
    chunk_store = RawChunkStore()
    chunk_store.store(RawChunk(chunk_idx=0, turn_idx=7, role="user", text="Actual chunk text"))

    results = search_memory_hybrid("lookup-token", bm25, chunk_store=chunk_store, top_k=1)

    assert len(results) == 1
    assert results[0]["turn_id"] == 7
    assert results[0]["chunk_id"] == 0
    assert results[0]["text"] == "Actual chunk text"


def test_search_memory_hybrid_skips_expansions_without_backing_chunks() -> None:
    bm25 = BM25Index()
    bm25.add(0, "seed")
    chunk_store = RawChunkStore()
    chunk_store.store(RawChunk(chunk_idx=0, turn_idx=1, role="user", text="seed"))
    triadic = TriadicMemory(n=64, p=4, seed=21)
    triadic.store_fact("seed", [("user", "has", "luna")])
    triadic.store_fact("expanded", [("user", "has", "mochi")])
    triadic.flush()

    results = search_memory_hybrid("seed", bm25, chunk_store=chunk_store, triadic=triadic, top_k=5)

    assert len(results) == 1
    assert results[0]["turn_id"] == 1
    assert all(result["turn_id"] >= 0 for result in results)
