"""Live test: triadic expansion with entity alias bridging.

Ingests conversation chunks with entity aliases, then queries using
alias mentions to verify triadic expands across alias boundaries.

Usage:
    # From RunPod (vLLM running locally):
    python3 -u test_triadic_alias_live.py

    # With custom URL:
    python3 -u test_triadic_alias_live.py --url http://localhost:8000/v1
"""
import argparse
import sys
import time

sys.path.insert(0, ".")

import numpy as np
from mabench.triadic_memory import TriadicMemory
from mabench.llm_triple_extractor import LLMTripleExtractor


REFERENCE_DATE = "2026-03-29"

# Conversation chunks that build a connected entity graph with aliases.
CHUNKS = [
    "My mom Sarah called me yesterday. She wants to plan a trip to Italy.",
    "Sarah is a retired teacher and she loves cooking pasta from scratch.",
    "My brother Jake works at Google. He's thinking about switching to a startup.",
    "Mom mentioned that Jake's girlfriend Lisa just got promoted at her law firm.",
]

# Queries that use aliases — triadic should bridge to canonical names.
ALIAS_QUERIES = [
    ("Mom", "Sarah"),     # Mom → Sarah (alias)
    ("she", "Sarah"),     # she → Sarah (pronoun alias)
    ("He", "Jake"),       # He → Jake (pronoun alias)
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1")
    args = parser.parse_args()

    print(f"Connecting to vLLM at {args.url}")
    print(f"Reference date: {REFERENCE_DATE}")
    print("=" * 70)

    ext = LLMTripleExtractor(base_url=args.url, enable_grounding=False)
    tri = TriadicMemory(n=1000, p=10)

    # --- Phase 1: Ingest chunks sequentially ---
    print("\n--- INGESTION ---")
    all_triples = []
    for i, chunk in enumerate(CHUNKS):
        triples = ext.extract(chunk, mode="conversation", reference_date=REFERENCE_DATE)
        print(f"\nChunk {i+1}: {chunk[:60]}...")
        print(f"  Triples: {len(triples)}")
        for s, p, o in triples:
            print(f"    ({s}, {p}, {o})")

        # Store in triadic
        tri.store_fact(chunk, triples)
        all_triples.extend(triples)

    tri.flush()

    # Feed entity registry aliases as synonym clusters
    registry = ext.entity_registry
    print(f"\n--- ENTITY REGISTRY ---")
    for canonical, aliases in registry.items():
        print(f"  {canonical} → {aliases}")

    if registry:
        clusters = [
            [canonical] + aliases
            for canonical, aliases in registry.items()
            if aliases
        ]
        tri.add_coref_clusters(clusters)
        print(f"\nFed {len(clusters)} alias clusters into triadic synonym dict")

    # --- Phase 2: Check triadic state ---
    n_with, n_total = tri.coverage()
    print(f"\n--- TRIADIC STATE ---")
    print(f"  Facts with triples: {n_with}/{n_total}")
    print(f"  Codebook size: {len(tri._codebook)} concepts")
    print(f"  Tensor entries: {tri.tensor_entries}")
    print(f"  Synonym dict entries: {len(tri._synonym_dict._aliases)}")

    # --- Phase 3: Query with aliases ---
    print(f"\n--- ALIAS EXPANSION QUERIES ---")
    for alias, expected_canonical in ALIAS_QUERIES:
        print(f"\n  Query concept: '{alias}' (should bridge to '{expected_canonical}')")

        # Check if synonym dict resolves alias
        resolved = tri._synonym_dict.resolve(alias.lower())
        print(f"  Synonym resolve: {alias.lower()} → {resolved}")

        # Try query_direct with the alias
        results = tri.query_direct(alias, top_k=5)
        if results:
            print(f"  Triadic results ({len(results)}):")
            for r in results:
                print(f"    - {r[:80]}...")
        else:
            print(f"  Triadic results: EMPTY")

    # --- Phase 4: Expansion test (simulate retrieval + expansion) ---
    print(f"\n--- EXPANSION TEST ---")
    # Simulate: cosine retrieval found chunk about Sarah, can triadic expand to Jake?
    if len(CHUNKS) >= 2:
        retrieved = [CHUNKS[0]]  # "My mom Sarah called me..."
        expanded = tri.expand(retrieved, top_k=5)
        print(f"  Retrieved: {retrieved[0][:60]}...")
        print(f"  Expanded ({len(expanded)}):")
        for e in expanded:
            print(f"    + {e[:80]}...")

        # Now try with an alias-only chunk
        print(f"\n  Testing alias-bridged expansion:")
        retrieved2 = [CHUNKS[1]]  # "Sarah is a retired teacher..."
        expanded2 = tri.expand(retrieved2, top_k=5)
        print(f"  Retrieved: {retrieved2[0][:60]}...")
        print(f"  Expanded ({len(expanded2)}):")
        for e in expanded2:
            print(f"    + {e[:80]}...")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {len(all_triples)} triples from {len(CHUNKS)} chunks")
    print(f"Registry: {len(registry)} entities, "
          f"{sum(len(a) for a in registry.values())} total aliases")
    print(f"Stats: {ext.stats}")


if __name__ == "__main__":
    main()
