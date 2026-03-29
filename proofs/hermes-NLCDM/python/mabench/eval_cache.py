"""Persistent ingestion cache for MABench evaluation.

Separates the slow ingestion phase (~2h for 537 chunks) from fast query
evaluation (~5min for 100 questions). Cache key is derived from dataset
identity + chunk_size only, so query-side changes (CR, decompose, retrieval
strategy, model) reuse the cached ingestion state.

Usage in eval scripts:

    from mabench.eval_cache import cache_dir, save_context, load_context

    path = cache_dir(dataset_config["sub_dataset"], chunk_size)
    if not load_context(agent, path, context_idx):
        # ... do full ingestion ...
        save_context(agent, path, context_idx)

Cache layout:
    output/ingest_cache/
        {sub_dataset}_cs{chunk_size}/   (or custom --cache_label)
            context_0/
                state.pkl       # pickled agent state (coupled_engine, triadic, etc.)
                meta.json       # human-readable metadata
            context_1/
                ...
"""

from __future__ import annotations

import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mabench.hermes_agent import HermesMemoryAgent

_CACHE_ROOT = Path(__file__).resolve().parent.parent / "output" / "ingest_cache"


def cache_dir(
    sub_dataset: str,
    chunk_size: int,
    label: str | None = None,
    root: Path | None = None,
) -> Path:
    """Compute cache directory path for a given dataset + chunk_size.

    Parameters
    ----------
    sub_dataset : str
        Dataset sub-name (e.g. "factconsolidation_mh_262k").
    chunk_size : int
        Chunk size used for ingestion.
    label : str | None
        Override the auto-generated directory name.
    root : Path | None
        Override the cache root directory.
    """
    root = root or _CACHE_ROOT
    if label:
        return root / label
    return root / f"{sub_dataset}_cs{chunk_size}"


def save_context(
    agent: HermesMemoryAgent,
    path: Path,
    context_idx: int,
) -> None:
    """Save post-ingestion agent state for one context.

    Pickles all state that the query path needs: coupled_engine (memory_store,
    co_occurrence, co_retrieval, W_temporal), triadic memory, supersession
    state, belief index, and entity accumulator.

    The embedding scorer and LLM client are NOT saved (they're stateless
    singletons that reinitialize on demand).
    """
    ctx_dir = path / f"context_{context_idx}"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "coupled_engine": agent.coupled_engine,
        "store_count": agent._store_count,
        "triple_index": dict(agent._triple_index),
        "superseded_texts": agent._superseded_texts,
        "fact_entities": agent._fact_entities,
        "supersession_graph": agent._supersession_graph,
        "supersedes": {k: set(v) for k, v in agent._supersedes.items()},
        "gliner_entity_index": {k: set(v) for k, v in agent._gliner_entity_index.items()},
        "subject_index": dict(getattr(agent, "_subject_index", {})),
        "ontology_index": dict(getattr(agent, "_ontology_index", {})),
        "entity_to_stored": dict(getattr(agent, "_entity_to_stored", {})),
    }

    if getattr(agent, "_triadic", None) is not None:
        state["triadic"] = agent._triadic
    if getattr(agent, "_belief_index", None) is not None:
        state["belief_index"] = agent._belief_index
    if hasattr(agent, "_entity_accumulator"):
        state["entity_accumulator"] = agent._entity_accumulator
    # BM25 index can be rebuilt from memory_store (fast), but save for consistency
    if getattr(agent, "_bm25_index", None) is not None:
        state["bm25_index"] = agent._bm25_index
    # REBEL KG: dict-of-dicts knowledge graph
    if getattr(agent, "_rebel_kg", None) is not None:
        state["rebel_kg"] = agent._rebel_kg
        state["rebel_kg_reverse"] = agent._rebel_kg_reverse
    # Chain cache must be pickled (contains embeddings + reverse index)
    if getattr(agent, "_chain_cache", None) is not None:
        state["chain_cache"] = agent._chain_cache

    state_file = ctx_dir / "state.pkl"
    with open(state_file, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_memories = len(agent.coupled_engine.memory_store)
    meta = {
        "n_memories": n_memories,
        "store_count": agent._store_count,
        "has_triadic": "triadic" in state,
        "has_belief": "belief_index" in state,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(ctx_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = state_file.stat().st_size / (1024 * 1024)
    print(f"Saved ingestion cache: {state_file} ({size_mb:.1f} MB, {n_memories} memories)")


def load_context(
    agent: HermesMemoryAgent,
    path: Path,
    context_idx: int,
) -> bool:
    """Restore agent state from ingestion cache.

    Replaces coupled_engine, triadic, supersession state, etc. with the
    cached post-ingestion state. Also reloads the embedding disk cache
    so that previously-seen query texts are served from cache.

    Returns True if cache was found and loaded, False otherwise.
    """
    state_file = path / f"context_{context_idx}" / "state.pkl"
    if not state_file.exists():
        return False

    t0 = time.time()
    with open(state_file, "rb") as f:
        state = pickle.load(f)

    # Core memory state
    agent.coupled_engine = state["coupled_engine"]
    agent._store_count = state["store_count"]

    # Supersession state
    agent._triple_index = defaultdict(list, state.get("triple_index", {}))
    agent._superseded_texts = state.get("superseded_texts", set())
    agent._fact_entities = state.get("fact_entities", {})
    agent._supersession_graph = state.get("supersession_graph", {})
    raw = state.get("supersedes", {})
    agent._supersedes = defaultdict(set, {k: set(v) for k, v in raw.items()})

    # GLiNER entity index
    raw_idx = state.get("gliner_entity_index", {})
    agent._gliner_entity_index = defaultdict(set, {k: set(v) for k, v in raw_idx.items()})

    # Subject index (subject-based KG hop resolution)
    agent._subject_index = state.get("subject_index", {})

    # Ontology layer (category-based KG hop resolution)
    agent._ontology_index = state.get("ontology_index", {})
    agent._entity_to_stored = state.get("entity_to_stored", {})

    # Optional components
    if "triadic" in state and getattr(agent, "_triadic", None) is not None:
        agent._triadic = state["triadic"]
    if "belief_index" in state and getattr(agent, "_belief_index", None) is not None:
        agent._belief_index = state["belief_index"]
    if "entity_accumulator" in state and hasattr(agent, "_entity_accumulator"):
        agent._entity_accumulator = state["entity_accumulator"]

    # REBEL KG: restore knowledge graph
    if getattr(agent, "_rebel_kg", None) is not None and "rebel_kg" in state:
        agent._rebel_kg = state["rebel_kg"]
        agent._rebel_kg_reverse = state.get("rebel_kg_reverse", {})

    # BM25 index: always rebuild from memory_store to ensure doc_id == index.
    # Cached BM25 may have stale doc_ids from _store_count (which diverges
    # from memory_store indices when coupled_engine replaces contradictions).
    if getattr(agent, "_bm25_index", None) is not None:
        from mabench.bm25_index import BM25Index
        agent._bm25_index = BM25Index()
        agent._bm25_index.add_batch([
            (i, m.text) for i, m in enumerate(agent.coupled_engine.memory_store)
        ])
    # Chain cache: restore if available (not rebuilt — requires embeddings)
    if getattr(agent, "_chain_cache", None) is not None and "chain_cache" in state:
        agent._chain_cache = state["chain_cache"]

    # Reload embedding disk cache for query efficiency — previously-seen
    # query texts (the 100 MABench questions) are served without model load
    agent._scorer._load_disk_cache()

    n = len(agent.coupled_engine.memory_store)
    elapsed = time.time() - t0
    print(f"Loaded ingestion cache: {n} memories in {elapsed:.1f}s from {state_file}")
    return True


def list_caches(root: Path | None = None) -> list[dict]:
    """List all available ingestion caches with metadata."""
    root = root or _CACHE_ROOT
    if not root.exists():
        return []

    caches = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        contexts = sorted(d.glob("context_*/meta.json"))
        if not contexts:
            continue

        info = {"name": d.name, "path": str(d), "contexts": []}
        for mp in contexts:
            with open(mp) as f:
                meta = json.load(f)
            sp = mp.parent / "state.pkl"
            size_mb = sp.stat().st_size / (1024 * 1024) if sp.exists() else 0
            info["contexts"].append({
                "context": mp.parent.name,
                "n_memories": meta.get("n_memories", "?"),
                "saved_at": meta.get("saved_at", "?"),
                "size_mb": f"{size_mb:.1f}",
            })
        caches.append(info)

    return caches
