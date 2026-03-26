#!/usr/bin/env python3
"""Run MemoryAgentBench evaluation with the composed HermesAgent.

Uses the modular agent (mabench.agent.HermesAgent) with frozen config
dataclasses instead of the monolithic HermesMemoryAgent.

Usage (from hermes-agent root):
    python3 -u -m mabench.run_hermes_composed \
        --dataset_config MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_mh_262k.yaml \
        --cooc_boost --triadic --decompose_query --conflict_resolution \
        --cr_per_hop --bm25_hops --chunk_size 512
"""

import json
import os
import pickle
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import sys
import dotenv
import numpy as np
import yaml
from tqdm import tqdm

# Add MABench and hermes paths
_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
_MABENCH = _HERMES_ROOT / "MemoryAgentBench"

sys.path.insert(0, str(_MABENCH))
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))

from conversation_creator import ConversationCreator
from dream_ops import DreamParams
from mabench.agent import HermesAgent
from mabench.configs import (
    RetrievalConfig, CRConfig, DecomposeConfig, StoreConfig,
    LLMConfig, BeliefConfig, EntityConfig, DedupConfig,
)
from mabench.eval_cache import cache_dir, list_caches
from utils.eval_other_utils import metrics_summarization

dotenv.load_dotenv(_NLCDM_PYTHON / ".env")


# ---------------------------------------------------------------------------
# Ingestion cache adapted for HermesAgent attribute layout
# ---------------------------------------------------------------------------

def _save_context(agent: HermesAgent, path: Path, context_idx: int) -> None:
    """Save post-ingestion state for one context."""
    ctx_dir = path / f"context_{context_idx}"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    ss = agent._supersession_state
    ei = agent._entity_index

    state = {
        "coupled_engine": agent.engine,
        "store_count": agent.store._store_count,
        "triple_index": dict(ss.triple_index),
        "superseded_texts": ss.superseded_texts,
        "fact_entities": ss.fact_entities,
        "supersession_graph": ss.supersession_graph,
        "supersedes": {k: set(v) for k, v in ss.supersedes.items()},
        "gliner_entity_index": {k: set(v) for k, v in ei.index.items()},
    }

    tri = agent.store._triadic
    if tri is not None:
        state["triadic"] = tri
    bel = agent.store._belief_index
    if bel is not None:
        state["belief_index"] = bel
    ea = agent.store._entity_accumulator
    if ea is not None:
        state["entity_accumulator"] = ea
    bm25 = agent.store._bm25_index
    if bm25 is not None:
        state["bm25_index"] = bm25
    rebel = agent.rebel
    if rebel is not None:
        state["rebel_kg"] = getattr(rebel, "_kg", None) or getattr(rebel, "_rebel_kg", None)
        state["rebel_kg_reverse"] = getattr(rebel, "_kg_reverse", None) or getattr(rebel, "_rebel_kg_reverse", None)
    cc = agent.store._chain_cache
    if cc is not None:
        state["chain_cache"] = cc

    state_file = ctx_dir / "state.pkl"
    with open(state_file, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_memories = len(agent.engine.memory_store)
    meta = {
        "n_memories": n_memories,
        "store_count": agent.store._store_count,
        "has_triadic": "triadic" in state,
        "has_belief": "belief_index" in state,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(ctx_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = state_file.stat().st_size / (1024 * 1024)
    print(f"Saved ingestion cache: {state_file} ({size_mb:.1f} MB, {n_memories} memories)")


def _load_context(agent: HermesAgent, path: Path, context_idx: int) -> bool:
    """Restore agent state from ingestion cache. Returns True if loaded."""
    state_file = path / f"context_{context_idx}" / "state.pkl"
    if not state_file.exists():
        return False

    t0 = time.time()
    with open(state_file, "rb") as f:
        state = pickle.load(f)

    # Core memory state
    agent.engine = state["coupled_engine"]
    agent.store._store_count = state["store_count"]

    # Supersession state
    ss = agent._supersession_state
    ss.triple_index = defaultdict(list, state.get("triple_index", {}))
    ss.superseded_texts = state.get("superseded_texts", set())
    ss.fact_entities = state.get("fact_entities", {})
    ss.supersession_graph = state.get("supersession_graph", {})
    raw = state.get("supersedes", {})
    ss.supersedes = defaultdict(set, {k: set(v) for k, v in raw.items()})

    # Entity index
    raw_idx = state.get("gliner_entity_index", {})
    agent._entity_index.index = defaultdict(set, {k: set(v) for k, v in raw_idx.items()})

    # Optional infrastructure
    if "triadic" in state and agent.store._triadic is not None:
        agent.store._triadic = state["triadic"]
        # Also update query_engine's reference
        agent.query_engine._triadic = state["triadic"]
    if "belief_index" in state and agent.store._belief_index is not None:
        agent.store._belief_index = state["belief_index"]
    if "entity_accumulator" in state and agent.store._entity_accumulator is not None:
        agent.store._entity_accumulator = state["entity_accumulator"]

    # BM25: always rebuild from memory_store (doc_id consistency)
    if agent.store._bm25_index is not None:
        from mabench.bm25_index import BM25Index
        fresh_bm25 = BM25Index()
        fresh_bm25.add_batch([
            (i, m.text) for i, m in enumerate(agent.engine.memory_store)
        ])
        agent.store._bm25_index = fresh_bm25
        agent.query_engine._bm25_index = fresh_bm25

    # Chain cache
    if agent.store._chain_cache is not None and "chain_cache" in state:
        agent.store._chain_cache = state["chain_cache"]
        agent.query_engine._chain_cache = state["chain_cache"]

    # Reload embedding disk cache
    agent._scorer._load_disk_cache()

    # Update engine references on all modules that hold one
    for mod in (agent.store, agent.query_engine, agent.pipeline,
                agent.retriever, agent.conflict_resolver, agent.entity_ops,
                agent.iterative):
        if hasattr(mod, "_engine"):
            mod._engine = agent.engine

    n = len(agent.engine.memory_store)
    elapsed = time.time() - t0
    print(f"Loaded ingestion cache: {n} memories in {elapsed:.1f}s from {state_file}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(description="MABench eval with composed HermesAgent")
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--retrieve_num", type=int, default=10)
    parser.add_argument("--contradiction_threshold", type=float, default=0.95)
    parser.add_argument("--dream_interval", type=int, default=0)
    parser.add_argument("--max_test_queries", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--recency_alpha", type=float, default=0.1)

    # Dream params
    parser.add_argument("--dream_eta", type=float, default=0.01)
    parser.add_argument("--dream_min_sep", type=float, default=0.3)
    parser.add_argument("--dream_prune_threshold", type=float, default=0.95)
    parser.add_argument("--dream_merge_threshold", type=float, default=0.90)
    parser.add_argument("--dream_explore_only", action="store_true")

    # Retrieval
    parser.add_argument("--cooc_boost", action="store_true")
    parser.add_argument("--cooc_weight", type=float, default=1.0)
    parser.add_argument("--cooc_gate", type=float, default=0.0)
    parser.add_argument("--ppr", action="store_true")
    parser.add_argument("--ppr_weight", type=float, default=0.3)
    parser.add_argument("--ppr_damping", type=float, default=0.85)
    parser.add_argument("--triadic", action="store_true")
    parser.add_argument("--bm25_hops", action="store_true")
    parser.add_argument("--chain_cache", action="store_true")
    parser.add_argument("--rebel_kg", action="store_true")

    # Query decomposition
    parser.add_argument("--decompose_query", action="store_true")
    parser.add_argument("--decompose_max_hops", type=int, default=3)

    # Conflict resolution
    parser.add_argument("--conflict_resolution", action="store_true")
    parser.add_argument("--cr_per_hop", action="store_true")
    parser.add_argument("--cr_entity_gate", action="store_true")
    parser.add_argument("--cr_overfetch", type=int, default=3)

    # Dedup
    parser.add_argument("--cosine_dedup", action="store_true")
    parser.add_argument("--cosine_dedup_threshold", type=float, default=0.95)

    # Entity
    parser.add_argument("--entity_retrieval", action="store_true")
    parser.add_argument("--entity_first", action="store_true")
    parser.add_argument("--entity_first_aggressive", action="store_true")
    parser.add_argument("--entity_rerank", action="store_true",
                        help="Entity-position residual reranking (arXiv:2603.19664)")
    parser.add_argument("--entity_rerank_alpha", type=float, default=0.3,
                        help="Blend weight: (1-α)*cosine + α*entity_residual")

    # Cache
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--cache_label", type=str, default=None)
    parser.add_argument("--list_caches", action="store_true")

    return parser.parse_args()


def _build_agent(args) -> HermesAgent:
    """Map CLI flags to frozen config dataclasses and construct HermesAgent."""
    dream_params = DreamParams(
        eta=args.dream_eta,
        min_sep=args.dream_min_sep,
        prune_threshold=args.dream_prune_threshold,
        merge_threshold=args.dream_merge_threshold,
        explore_only=args.dream_explore_only,
    )

    retrieval = RetrievalConfig(
        retrieve_num=args.retrieve_num,
        cooc_boost=args.cooc_boost,
        cooc_weight=args.cooc_weight,
        cooc_gate_threshold=args.cooc_gate,
        ppr=args.ppr,
        ppr_weight=args.ppr_weight,
        ppr_damping=args.ppr_damping,
        entity_rerank=args.entity_rerank,
        entity_rerank_alpha=args.entity_rerank_alpha,
    )
    cr = CRConfig(
        enabled=args.conflict_resolution,
        per_hop=args.cr_per_hop,
        entity_gate=args.cr_entity_gate,
        overfetch=args.cr_overfetch,
    )
    decompose = DecomposeConfig(
        enabled=args.decompose_query,
        max_hops=args.decompose_max_hops,
    )
    store = StoreConfig(
        cooc_boost=args.cooc_boost,
        dream_interval=args.dream_interval,
    )
    llm_cfg = LLMConfig(
        model=args.model,
        temperature=0.0,
    )
    entity = EntityConfig(
        entity_retrieval=args.entity_retrieval,
        entity_first=args.entity_first,
        entity_first_aggressive=args.entity_first_aggressive,
    )
    dedup = DedupConfig(
        enabled=args.cosine_dedup,
        threshold=args.cosine_dedup_threshold,
    )

    return HermesAgent(
        dim=1024,
        contradiction_threshold=args.contradiction_threshold,
        recency_alpha=args.recency_alpha,
        dream_params=dream_params,
        retrieval=retrieval,
        cr=cr,
        decompose=decompose,
        store=store,
        llm=llm_cfg,
        entity=entity,
        dedup=dedup,
        bm25_hops=args.bm25_hops,
        chain_cache=args.chain_cache,
        rebel_kg=args.rebel_kg,
        triadic=args.triadic,
    )


def main():
    args = parse_args()

    if args.list_caches:
        for c in list_caches():
            print(f"  {c['name']}:")
            for ctx in c["contexts"]:
                print(f"    {ctx['context']}: {ctx['n_memories']} memories, "
                      f"{ctx['size_mb']} MB, saved {ctx['saved_at']}")
        return

    # Load dataset config
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)

    if args.chunk_size > 0:
        dataset_config["chunk_size"] = args.chunk_size

    agent_config = {
        "agent_name": "hermes_rag",
        "agent_chunk_size": None,
        "input_length_limit": 128000,
        "buffer_length": 1000,
        "model": args.model,
        "temperature": 0.0,
        "output_dir": "output/hermes_composed",
        "retrieve_num": args.retrieve_num,
    }

    output_dir = Path(agent_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    prefix = "composed"
    if args.cooc_boost:
        prefix += "_cooc"
    output_file = output_dir / f"{prefix}_{dataset_config['sub_dataset']}_{safe_model}.json"
    trace_file = output_dir / f"{prefix}_{dataset_config['sub_dataset']}_{safe_model}_traces.jsonl"

    # Truncate trace file at start of run (avoid stale appended data)
    if trace_file.exists():
        trace_file.unlink()

    # Print config summary
    print(f"Agent: HermesAgent (composed)")
    print(f"Dataset: {dataset_config['dataset']} / {dataset_config['sub_dataset']}")
    print(f"Model: {args.model}")
    print(f"Output: {output_file}")
    print(f"Retrieval: cooc_boost={args.cooc_boost}, bm25_hops={args.bm25_hops}, triadic={args.triadic}")
    print(f"Query: decompose={args.decompose_query}, cr={args.conflict_resolution}, cr_per_hop={args.cr_per_hop}")
    print()

    # Load data
    conversation_creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    print(f"Contexts: {len(all_chunks)}")
    total_queries = sum(len(qa) for qa in all_qa_pairs)
    print(f"Total queries: {total_queries}")

    # Ingestion cache setup
    use_cache = not args.no_cache
    cache_path = None
    if use_cache:
        effective_cs = dataset_config.get("chunk_size", 0)
        cache_path = cache_dir(dataset_config["sub_dataset"], effective_cs, args.cache_label)
        print(f"Ingestion cache: {cache_path}")
        if args.refresh_cache:
            print("  (refresh mode: will re-ingest and overwrite)")
    else:
        print("Ingestion cache: disabled")
    print()

    # Build agent
    agent = _build_agent(args)

    # Run evaluation
    metrics: dict[str, list] = defaultdict(list)
    results: list[dict] = []
    query_index = 0
    start_time = time.time()

    for context_idx, (chunks, qa_pairs) in enumerate(
        tqdm(
            zip(all_chunks, all_qa_pairs),
            total=len(all_chunks),
            desc="Contexts",
        )
    ):
        agent.reset()

        # Try loading from ingestion cache
        cached = False
        if use_cache and not args.refresh_cache:
            cached = _load_context(agent, cache_path, context_idx)

        if not cached:
            print(f"\n--- Context {context_idx}: memorizing {len(chunks)} chunks ---")
            for chunk in tqdm(chunks, desc="Memorizing", leave=False):
                agent.send_message(chunk, memorizing=True)
            agent._scorer.save_disk_cache()

            if use_cache:
                _save_context(agent, cache_path, context_idx)

        # Query
        print(f"    Querying {len(qa_pairs)} questions...")
        for query_data in tqdm(qa_pairs, desc="Querying", leave=False):
            if len(query_data) == 3:
                query, answer, qa_pair_id = query_data
            else:
                query, answer = query_data
                qa_pair_id = None

            if args.max_test_queries > 0 and query_index >= args.max_test_queries:
                break

            agent_output = agent.send_message(query, memorizing=False)

            # Write per-query trace to JSONL for easy review
            trace = agent_output.pop("trace", None)
            if trace is not None:
                trace_record = {
                    "query_index": query_index,
                    "question": query,
                    "expected_answer": answer if isinstance(answer, str) else answer[0] if answer else None,
                    "predicted_answer": agent_output.get("output", ""),
                    **trace,
                }
                with open(trace_file, "a") as tf:
                    tf.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

            metrics, results = metrics_summarization(
                agent_output,
                query,
                answer,
                dataset_config,
                metrics,
                results,
                query_index,
                qa_pair_id,
            )
            query_index += 1

        # Save intermediate results
        averaged_metrics = {
            key: np.mean(values) * (1 if ("_len" in key) or ("_time" in key) else 100)
            for key, values in metrics.items()
        }
        output_data = {
            "agent_config": agent_config,
            "dataset_config": dataset_config,
            "hermes_config": {
                "architecture": "composed",
                "retrieval": {
                    "cooc_boost": args.cooc_boost,
                    "cooc_weight": args.cooc_weight if args.cooc_boost else None,
                    "bm25_hops": args.bm25_hops,
                    "triadic": args.triadic,
                },
                "cr": {
                    "enabled": args.conflict_resolution,
                    "per_hop": args.cr_per_hop,
                },
                "decompose": {
                    "enabled": args.decompose_query,
                    "max_hops": args.decompose_max_hops,
                },
            },
            "data": results,
            "metrics": {k: v for k, v in metrics.items()},
            "averaged_metrics": averaged_metrics,
            "time_cost": time.time() - start_time,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Evaluation complete: {query_index} queries in {elapsed:.1f}s")
    print(f"Results saved to: {output_file}")
    print(f"\nMetrics:")
    for key, values in metrics.items():
        scale = 1 if ("_len" in key) or ("_time" in key) else 100
        print(f"  {key}: {np.mean(values) * scale:.2f}")


if __name__ == "__main__":
    main()
