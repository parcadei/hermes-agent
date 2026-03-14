#!/usr/bin/env python3
"""Run MemoryAgentBench evaluation with Hermes Memory Agent.

Usage (from hermes-agent root):
    proofs/hermes-NLCDM/python/.venv/bin/python proofs/hermes-NLCDM/python/mabench/run_hermes.py \
        --dataset_config MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml

Reuses MABench data loading and metrics, substitutes HermesMemoryAgent for AgentWrapper.
"""

import hashlib
import json
import logging
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
from dream_ops import DreamParams, GraphDreamParams
from mabench.hermes_agent import HermesMemoryAgent
from utils.eval_other_utils import metrics_summarization
from utils.templates import get_template

dotenv.load_dotenv(_NLCDM_PYTHON / ".env")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to dataset configuration YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for answer generation",
    )
    parser.add_argument(
        "--retrieve_num",
        type=int,
        default=10,
        help="Number of memories to retrieve",
    )
    parser.add_argument(
        "--contradiction_threshold",
        type=float,
        default=0.95,
        help="V2 cosine threshold for contradiction detection",
    )
    parser.add_argument(
        "--dream_interval",
        type=int,
        default=0,
        help="Dream cycle every N stores (0=disabled)",
    )
    parser.add_argument(
        "--max_test_queries",
        type=int,
        default=0,
        help="Limit max test queries (0=no limit)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Override dataset chunk_size (0=use config default)",
    )
    parser.add_argument(
        "--recency_alpha",
        type=float,
        default=0.1,
        help="Recency weighting alpha for retrieval (0=disabled)",
    )
    parser.add_argument("--fullpass", action="store_true",
                        help="Baseline: skip memory system, send full context to LLM directly")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    parser.add_argument(
        "--dream_eta", type=float, default=0.01,
        help="Dream cycle learning rate (Lean bound: 0 < eta < min_sep/2)",
    )
    parser.add_argument(
        "--dream_min_sep", type=float, default=0.3,
        help="NREM repulsion minimum separation (Lean bound: 0 < min_sep <= 1)",
    )
    parser.add_argument(
        "--dream_prune_threshold", type=float, default=0.95,
        help="NREM prune near-duplicate threshold (Lean bound: merge < prune <= 1)",
    )
    parser.add_argument(
        "--dream_merge_threshold", type=float, default=0.90,
        help="NREM merge group threshold (Lean bound: 0 < merge < prune)",
    )
    parser.add_argument("--cooc_boost", action="store_true",
                        help="Use co-occurrence boost retrieval (full-store neighbor signal)")
    parser.add_argument(
        "--cooc_weight", type=float, default=1.0,
        help="Co-occurrence boost weight (0=pure cosine, 1.0=optimal from grid sweep)",
    )
    parser.add_argument(
        "--cooc_gate", type=float, default=0.0,
        help="Confidence gate threshold: skip cooc boost when top cosine score >= this value (0=disabled)",
    )
    parser.add_argument("--ppr", action="store_true",
                        help="Use Personalized PageRank retrieval on co-occurrence graph")
    parser.add_argument(
        "--ppr_weight", type=float, default=0.3,
        help="PPR weight for score blending (0=pure cosine)",
    )
    parser.add_argument(
        "--ppr_damping", type=float, default=0.85,
        help="PPR damping factor",
    )
    parser.add_argument("--coretrieval", action="store_true",
                        help="Use co-retrieval graph for two-hop expansion")
    parser.add_argument(
        "--coretrieval_bonus", type=float, default=0.3,
        help="Score bonus for facts discovered via co-retrieval expansion",
    )
    parser.add_argument(
        "--coretrieval_min_count", type=float, default=2.0,
        help="Minimum co-retrieval count to qualify as an edge",
    )
    parser.add_argument("--transfer", action="store_true",
                        help="Use Hopfield transfer retrieval (max(cosine, transfer) scoring)")
    parser.add_argument(
        "--transfer_k", type=int, default=0,
        help="Prefilter to top-K patterns for Hopfield store (0=use all)",
    )
    parser.add_argument("--cross_domain_probes", action="store_true",
                        help="Inject cross-domain probe queries via query_readonly() to build co-retrieval edges")
    parser.add_argument(
        "--probe_frequency", type=int, default=10,
        help="Fire cross-domain probes every N sessions (default: 10)",
    )
    parser.add_argument(
        "--probes_per_session", type=int, default=3,
        help="Number of cross-domain probes per firing session (default: 3)",
    )
    parser.add_argument("--triadic", action="store_true",
                        help="Enable triadic memory for structural recall via Overmann 3D tensor")
    parser.add_argument(
        "--triadic_n", type=int, default=1000,
        help="Triadic memory dimension n (default: 1000)",
    )
    parser.add_argument(
        "--triadic_p", type=int, default=10,
        help="Triadic memory sparsity p (default: 10)",
    )
    parser.add_argument(
        "--triadic_expand_k", type=int, default=5,
        help="Triadic expansion top-k (default: 5)",
    )
    parser.add_argument(
        "--decompose_query", action="store_true",
        help="Enable multi-hop query decomposition via LLM",
    )
    parser.add_argument(
        "--decompose_max_hops", type=int, default=3,
        help="Maximum sub-queries for decomposition (default: 3)",
    )
    parser.add_argument("--decompose_coverage", action="store_true",
                        help="Use decomposition as coverage audit (gap-fill, not hop-chain)")
    parser.add_argument("--decompose_rrf", action="store_true",
                        help="Use RRF merge for decompose (instead of priority merge)")
    parser.add_argument(
        "--bm25_weight", type=float, default=0.0,
        help="BM25 lexical matching weight (0=disabled). Blends with cosine+cooc for entity discrimination.",
    )
    parser.add_argument(
        "--dedup_threshold", type=float, default=0.0,
        help="Ingestion-time dedup: skip storing when cosine > threshold (0=disabled, 0.98 recommended)",
    )
    parser.add_argument("--dream_after_ingest", action="store_true",
                        help="Run ONE dream cycle after ingestion completes (before queries)")
    parser.add_argument("--bridge_aware_dream", action="store_true",
                        help="Protect bridge facts from dream drift via structural importance (Tononi protection rule)")
    parser.add_argument("--dream_batch_mode", action="store_true",
                        help="Batch dream: skip repulsion+unlearn, only run prune+merge (safe dedup, no drift)")
    parser.add_argument("--graph_dream", action="store_true",
                        help="Run graph-level dream after ingest (replay + transitive closure + edge prune)")
    parser.add_argument(
        "--dream_weight", type=float, default=0.0,
        help="Dream boost weight for dual-graph retrieval (0=disabled, 0.3=recommended)",
    )
    parser.add_argument(
        "--dream_top_k", type=int, default=5,
        help="Top-k sparsification for dream edge discovery (default: 5)",
    )
    parser.add_argument("--temporal_context", action="store_true",
                        help="Enable temporal ordering in outgestion context "
                             "(sort by recency, annotate older/newer)")
    parser.add_argument("--contradiction_context", action="store_true",
                        help="Enable contradiction detection in outgestion "
                             "(flag high-similarity chunks as SUPERSEDED/CURRENT)")
    parser.add_argument("--contradiction_sim_threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for contradiction detection (default: 0.85)")
    parser.add_argument("--iterative_query", action="store_true",
                        help="Enable iterative multi-hop retrieval with LLM-in-the-loop "
                             "termination (retrieve → judge → re-query loop)")
    parser.add_argument("--iterative_max_hops", type=int, default=3,
                        help="Maximum hops for iterative retrieval (default: 3)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable ingest caching (always ingest fresh)")
    return parser.parse_args()


def _cache_key(dataset_config: dict, chunk_size: int) -> str:
    """Deterministic cache key from dataset config + chunk size."""
    key_data = json.dumps({
        "dataset": dataset_config.get("dataset"),
        "sub_dataset": dataset_config.get("sub_dataset"),
        "chunk_size": chunk_size,
        "data_path": dataset_config.get("data_path"),
    }, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _cache_dir(dataset_config: dict, chunk_size: int) -> Path:
    """Return the cache directory for a given dataset config."""
    key = _cache_key(dataset_config, chunk_size)
    sub = dataset_config.get("sub_dataset", "unknown")
    return _NLCDM_PYTHON / "output" / "ingest_cache" / f"{sub}_{key}"


def _save_ingest_cache(
    cache_path: Path,
    context_idx: int,
    agent,
) -> None:
    """Save agent state after ingestion for a single context."""
    ctx_dir = cache_path / f"context_{context_idx}"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    # Coupled engine: memory_store + graph structures
    ce = agent.coupled_engine
    with open(ctx_dir / "coupled_engine.pkl", "wb") as f:
        pickle.dump({
            "memory_store": ce.memory_store,
            "co_occurrence": getattr(ce, "_co_occurrence", None),
            "co_retrieval": getattr(ce, "_co_retrieval", None),
            "W_temporal": getattr(ce, "_W_temporal", None),
            "session_buffer": getattr(ce, "_session_buffer", []),
            "session_indices": getattr(ce, "_session_indices", []),
        }, f)

    # Triadic: save text_to_triples for reconstruction (NOT the tensor)
    triadic = getattr(agent, "_triadic", None)
    if triadic is not None:
        with open(ctx_dir / "triadic.pkl", "wb") as f:
            pickle.dump({
                "text_to_triples": triadic._text_to_triples,
                "n": triadic.n,
                "p": triadic.p,
            }, f)

    # V1 orchestrator memories
    with open(ctx_dir / "orchestrator.pkl", "wb") as f:
        pickle.dump(agent.orchestrator._memories, f)

    # Store count
    with open(ctx_dir / "meta.pkl", "wb") as f:
        pickle.dump({"store_count": agent._store_count}, f)


def _load_ingest_cache(
    cache_path: Path,
    context_idx: int,
    agent,
    enable_triadic: bool,
) -> bool:
    """Load cached ingest state into agent. Returns True if successful."""
    ctx_dir = cache_path / f"context_{context_idx}"
    ce_path = ctx_dir / "coupled_engine.pkl"
    if not ce_path.exists():
        return False

    # Coupled engine
    with open(ce_path, "rb") as f:
        ce_data = pickle.load(f)
    agent.coupled_engine.memory_store = ce_data["memory_store"]
    if ce_data.get("co_occurrence") is not None:
        agent.coupled_engine._co_occurrence = ce_data["co_occurrence"]
    if ce_data.get("co_retrieval") is not None:
        agent.coupled_engine._co_retrieval = ce_data["co_retrieval"]
    if ce_data.get("W_temporal") is not None:
        agent.coupled_engine._W_temporal = ce_data["W_temporal"]
    if ce_data.get("session_buffer"):
        agent.coupled_engine._session_buffer = ce_data["session_buffer"]
    if ce_data.get("session_indices"):
        agent.coupled_engine._session_indices = ce_data["session_indices"]
    if hasattr(agent.coupled_engine, "_invalidate_cache"):
        agent.coupled_engine._invalidate_cache()

    # Rebuild FAISS index from loaded embeddings
    entries = agent.coupled_engine.memory_store
    if entries:
        import faiss
        dim = len(entries[0].embedding)
        index = faiss.IndexFlatIP(dim)
        embs = np.array([e.embedding for e in entries], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        index.add(embs)
        agent.coupled_engine._faiss_index = index

    # Triadic: rebuild from text_to_triples (not raw tensor)
    tri_path = ctx_dir / "triadic.pkl"
    if enable_triadic and tri_path.exists():
        with open(tri_path, "rb") as f:
            tri_data = pickle.load(f)
        from mabench.triadic_memory import TriadicMemory
        triadic = TriadicMemory(
            n=tri_data.get("n", 1000), p=tri_data.get("p", 10),
        )
        for text, triples in tri_data["text_to_triples"].items():
            triadic.store_fact(text, [tuple(t) for t in triples])
        agent._triadic = triadic

    # V1 orchestrator
    orch_path = ctx_dir / "orchestrator.pkl"
    if orch_path.exists():
        with open(orch_path, "rb") as f:
            agent.orchestrator._memories = pickle.load(f)
        if agent.orchestrator._memories:
            max_ct = max(m.creation_time for m in agent.orchestrator._memories)
            agent.orchestrator._next_time = max_ct + 1.0
            # Rebuild memory_id -> index mapping (required for query lookups)
            agent.orchestrator._full_rebuild_index()

    # Meta
    meta_path = ctx_dir / "meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        agent._store_count = meta["store_count"]

    return True


def main():
    args = parse_args()

    # Configure logging so dream diagnostics are visible
    # force=True to override any prior basicConfig from imported libraries
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )

    # Load dataset config
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)

    # Override chunk_size if specified
    if args.chunk_size > 0:
        dataset_config["chunk_size"] = args.chunk_size

    # Build a minimal agent_config that ConversationCreator expects
    agent_config = {
        "agent_name": "hermes_rag",
        "agent_chunk_size": None,
        "input_length_limit": 128000,
        "buffer_length": 1000,
        "model": args.model,
        "temperature": 0.0,
        "output_dir": "output/hermes",
        "retrieve_num": args.retrieve_num,
    }

    # Create output directory
    output_dir = Path(agent_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    if args.fullpass:
        prefix = "fullpass"
    elif args.transfer:
        tk_str = f"_k{args.transfer_k}" if args.transfer_k > 0 else ""
        prefix = f"hermes_transfer{tk_str}"
    elif args.coretrieval:
        prefix = f"hermes_coretrieval_mc{args.coretrieval_min_count}"
    elif args.cooc_boost and args.ppr:
        prefix = "hermes_cooc_ppr"
    elif args.cooc_boost and args.cooc_gate > 0:
        prefix = f"hermes_cooc_g{args.cooc_gate}"
    elif args.cooc_boost:
        prefix = "hermes_cooc"
    elif args.ppr:
        prefix = "hermes_ppr"
    else:
        prefix = "hermes"
    # Append triadic/decompose suffixes so runs don't overwrite each other
    if args.triadic:
        prefix += "_triadic"
    if args.decompose_query:
        prefix += "_decompose"
    if args.decompose_rrf:
        prefix += "_rrf"
    if args.iterative_query:
        prefix += "_iterative"
    if args.decompose_coverage:
        prefix += "_coverage"
    if args.bm25_weight > 0:
        prefix += "_bm25"
    if args.dedup_threshold > 0:
        prefix += f"_dedup{args.dedup_threshold}"
    if getattr(args, 'dream_after_ingest', False):
        if getattr(args, 'dream_batch_mode', False):
            prefix += "_dream_batch"
        elif getattr(args, 'bridge_aware_dream', False):
            prefix += "_dream_bridge"
        else:
            prefix += "_dream"
    if getattr(args, 'graph_dream', False):
        prefix += "_graphdream"
    if args.dream_weight > 0:
        prefix += f"_dw{args.dream_weight}"
    if args.dream_top_k != 5:
        prefix += f"_dk{args.dream_top_k}"
    if args.temporal_context and args.contradiction_context:
        prefix += "_temporal_contradiction"
    elif args.temporal_context:
        prefix += "_temporal"
    elif args.contradiction_context:
        prefix += "_contradiction"
    output_file = output_dir / f"{prefix}_{dataset_config['sub_dataset']}_{safe_model}.json"

    print(f"Dataset: {dataset_config['dataset']} / {dataset_config['sub_dataset']}")
    print(f"Model: {args.model}")
    print(f"Mode: {'FULLPASS (no memory system)' if args.fullpass else 'Hermes memory agent'}")
    print(f"Output: {output_file}")
    if not args.fullpass:
        print(f"Contradiction threshold: {args.contradiction_threshold}")
        print(f"Dream interval: {args.dream_interval}")
        if args.transfer:
            tk_str = f", transfer_k={args.transfer_k}" if args.transfer_k > 0 else ", transfer_k=all"
            print(f"Retrieval: Hopfield transfer (max(cosine, transfer){tk_str})")
        elif args.coretrieval:
            print(f"Retrieval: co-retrieval (bonus={args.coretrieval_bonus}, min_count={args.coretrieval_min_count})")
        elif args.cooc_boost and args.ppr:
            print(f"Retrieval: cooc+PPR (cooc_weight={args.cooc_weight}, ppr_weight={args.ppr_weight}, damping={args.ppr_damping})")
        elif args.cooc_boost:
            gate_str = f", gate={args.cooc_gate}" if args.cooc_gate > 0 else ""
            print(f"Retrieval: cooc_boost (weight={args.cooc_weight}{gate_str})")
        elif args.ppr:
            print(f"Retrieval: PPR (weight={args.ppr_weight}, damping={args.ppr_damping})")
        else:
            print(f"Retrieval: cosine (baseline)")
        if args.triadic:
            print(f"Triadic retrieval: True (n={args.triadic_n}, p={args.triadic_p}, expand_k={args.triadic_expand_k})")
        if args.decompose_query:
            print(f"Query decomposition: True (max_hops={args.decompose_max_hops})")
        if args.iterative_query:
            print(f"Iterative retrieval: True (max_hops={args.iterative_max_hops}, LLM-in-the-loop)")
        if args.decompose_coverage:
            print(f"Coverage audit: True (decompose as gap-fill, not hop-chain)")
        if getattr(args, 'graph_dream', False):
            print(f"Graph dream: enabled (replay + transitive closure + edge prune, top_k={args.dream_top_k})")
        if args.dream_weight > 0:
            print(f"Dream boost: weight={args.dream_weight}")
    print()

    # Load data
    conversation_creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    print(f"Contexts: {len(all_chunks)}")
    total_queries = sum(len(qa) for qa in all_qa_pairs)
    print(f"Total queries: {total_queries}")
    print()

    # Initialize agent or LLM client
    if args.fullpass:
        from openai import OpenAI
        llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        agent = None
    else:
        llm_client = None
        dream_params = DreamParams(
            eta=args.dream_eta,
            min_sep=args.dream_min_sep,
            prune_threshold=args.dream_prune_threshold,
            merge_threshold=args.dream_merge_threshold,
        )
        agent = HermesMemoryAgent(
            model=args.model,
            contradiction_threshold=args.contradiction_threshold,
            retrieve_num=args.retrieve_num,
            dream_interval=args.dream_interval,
            temperature=0.0,
            recency_alpha=args.recency_alpha,
            dream_params=dream_params,
            cooc_boost_retrieval=args.cooc_boost,
            cooc_weight=args.cooc_weight,
            cooc_gate_threshold=args.cooc_gate,
            ppr_retrieval=args.ppr,
            ppr_weight=args.ppr_weight,
            ppr_damping=args.ppr_damping,
            coretrieval_retrieval=getattr(args, 'coretrieval', False),
            coretrieval_bonus=getattr(args, 'coretrieval_bonus', 0.3),
            coretrieval_min_count=getattr(args, 'coretrieval_min_count', 2.0),
            transfer_retrieval=args.transfer,
            transfer_k=args.transfer_k if args.transfer_k > 0 else None,
            triadic_retrieval=args.triadic,
            triadic_n=args.triadic_n,
            triadic_p=args.triadic_p,
            triadic_expand_k=args.triadic_expand_k,
            decompose_query=args.decompose_query,
            decompose_max_hops=args.decompose_max_hops,
            decompose_coverage=args.decompose_coverage,
            decompose_rrf=args.decompose_rrf,
            iterative_query=args.iterative_query,
            iterative_max_hops=args.iterative_max_hops,
            bm25_weight=args.bm25_weight,
            dedup_threshold=args.dedup_threshold,
            dream_weight=args.dream_weight,
            temporal_context=args.temporal_context,
            contradiction_context=args.contradiction_context,
            contradiction_sim_threshold=args.contradiction_sim_threshold,
        )

    # Ingest cache setup
    chunk_size = dataset_config.get("chunk_size", 0)
    cache_path = _cache_dir(dataset_config, chunk_size)
    use_cache = not args.fullpass and not args.no_cache

    if use_cache:
        print(f"Ingest cache: {cache_path}")

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
        if args.fullpass:
            # Full pass: concatenate all chunks as the knowledge pool
            full_context = "\n\n".join(chunks)
            print(f"\n--- Context {context_idx}: {len(chunks)} chunks, fullpass mode ---")
        else:
            # Reset agent for each new context
            agent.reset()

            # Try loading from cache first
            if use_cache and _load_ingest_cache(
                cache_path, context_idx, agent, args.triadic,
            ):
                n_mem = len(agent.coupled_engine.memory_store)
                tri_info = ""
                if getattr(agent, "_triadic", None) is not None:
                    tri_info = f", triadic={agent._triadic._n_facts_with_triples} facts"
                print(f"\n--- Context {context_idx}: loaded from cache ({n_mem} memories{tri_info}) ---")
            else:
                # Fresh ingest
                print(f"\n--- Context {context_idx}: memorizing {len(chunks)} chunks ---")
                for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Memorizing", leave=False)):
                    agent.send_message(chunk, memorizing=True)
                # Persist embedding cache after memorization
                agent._scorer.save_disk_cache()
                # Log dedup stats
                if args.dedup_threshold > 0:
                    skipped = agent.coupled_engine._dedup_skipped
                    n_mem = len(agent.coupled_engine.memory_store)
                    print(f"    Dedup: {skipped} skipped, {n_mem} stored (threshold={args.dedup_threshold})")
                # Save ingest cache for future runs
                if use_cache:
                    _save_ingest_cache(cache_path, context_idx, agent)
                    print(f"    Cached ingest to {cache_path / f'context_{context_idx}'}")

        # Dream once after ingest (if requested)
        if not args.fullpass and getattr(args, 'dream_after_ingest', False):
            n_before = len(agent.coupled_engine.memory_store)
            bridge_flag = getattr(args, 'bridge_aware_dream', False)
            batch_flag = getattr(args, 'dream_batch_mode', False)
            mode_parts = []
            if bridge_flag:
                mode_parts.append("bridge-aware")
            if batch_flag:
                mode_parts.append("batch(prune+merge only)")
            mode = ", ".join(mode_parts) if mode_parts else "standard"
            print(f"    Running post-ingest dream cycle ({n_before} memories, {mode})...")
            agent.coupled_engine.dream(bridge_aware=bridge_flag, batch_mode=batch_flag)
            n_after = len(agent.coupled_engine.memory_store)
            print(f"    Dream complete: {n_before} -> {n_after} memories")

        # Graph dream after ingest (if requested)
        if not args.fullpass and getattr(args, 'graph_dream', False):
            n_mem = len(agent.coupled_engine.memory_store)
            edges_before = sum(len(n) for n in agent.coupled_engine._co_occurrence.values())
            print(f"    Running graph dream ({n_mem} memories, {edges_before} edges, "
                  f"top_k={args.dream_top_k})...")
            gdp = GraphDreamParams(replay_top_k=args.dream_top_k)
            result = agent.coupled_engine.dream(graph_mode=True, graph_dream_params=gdp)
            edges_after = result.get("edges_after", 0)
            dream_edge_count = sum(len(n) for n in agent.coupled_engine._dream_edges.values())
            print(f"    Graph dream complete: {edges_before} cooc -> {dream_edge_count} dream edges "
                  f"(replay={result.get('edges_discovered', 0)}, "
                  f"tc={result.get('edges_from_tc', 0)}, "
                  f"pruned={result.get('edges_pruned', 0)})")

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

            if args.fullpass:
                # Direct LLM call with full context + question
                q_start = time.time()
                system_msg = (
                    "You are a helpful assistant that can read the context and "
                    "memorize it for future retrieval."
                )
                user_msg = (
                    f"{full_context}\n\n{query}"
                )
                response = llm_client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=max(dataset_config["generation_max_length"], 16),
                )
                output = response.choices[0].message.content or ""
                usage = response.usage
                agent_output = {
                    "output": output,
                    "input_len": usage.prompt_tokens,
                    "output_len": usage.completion_tokens,
                    "memory_construction_time": 0.0,
                    "query_time_len": time.time() - q_start,
                }
            else:
                # Get agent response
                agent_output = agent.send_message(query, memorizing=False)

            # Compute metrics using MABench utilities
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
                "contradiction_threshold": args.contradiction_threshold,
                "dream_interval": args.dream_interval,
                "retrieve_num": args.retrieve_num,
                "recency_alpha": args.recency_alpha if not args.fullpass else None,
                "fact_level_parsing": not args.fullpass,
                "cooc_boost": args.cooc_boost if not args.fullpass else False,
                "cooc_weight": args.cooc_weight if args.cooc_boost else None,
                "cooc_gate": args.cooc_gate if args.cooc_boost else None,
                "ppr": args.ppr if not args.fullpass else False,
                "ppr_weight": args.ppr_weight if args.ppr else None,
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
