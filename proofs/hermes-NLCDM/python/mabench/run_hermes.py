#!/usr/bin/env python3
"""Run MemoryAgentBench evaluation with Hermes Memory Agent.

Usage (from hermes-agent root):
    proofs/hermes-NLCDM/python/.venv/bin/python proofs/hermes-NLCDM/python/mabench/run_hermes.py \
        --dataset_config MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml

Reuses MABench data loading and metrics, substitutes HermesMemoryAgent for AgentWrapper.
"""

import json
import os
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
from mabench.hermes_agent import HermesMemoryAgent
from mabench.eval_cache import cache_dir, save_context, load_context, list_caches
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
                        help="Use triadic memory (Overmann 3D tensor) for structural expansion")
    parser.add_argument("--decompose_query", action="store_true",
                        help="Decompose multi-hop questions into chained sub-queries")
    parser.add_argument("--conflict_resolution", action="store_true",
                        help="Query-time conflict resolution (acl-aware copular extraction)")
    parser.add_argument("--cr_per_hop", action="store_true",
                        help="Apply CR before each intermediate answer in decompose loop")
    parser.add_argument("--cr_entity_gate", action="store_true",
                        help="Require NER entity overlap for conflict resolution")
    parser.add_argument("--cr_overfetch", type=int, default=3,
                        help="Over-fetch multiplier for CR (retrieve N*k, CR filters to k)")
    parser.add_argument("--cosine_dedup", action="store_true",
                        help="Pairwise cosine dedup: near-duplicate facts → keep newest")
    parser.add_argument("--cosine_dedup_threshold", type=float, default=0.95,
                        help="Cosine similarity threshold for pairwise dedup (default: 0.95)")
    parser.add_argument("--entity_retrieval", action="store_true",
                        help="Add entity name text search to decompose hops")
    parser.add_argument("--entity_first", action="store_true",
                        help="Entity-first retrieval: use entity index as PRIMARY path, cosine/BM25 as fallback")
    parser.add_argument("--entity_first_aggressive", action="store_true",
                        help="When entity-first hits, use ONLY entity facts + triadic. Skip BM25/cosine/supplement.")
    parser.add_argument("--bm25_hops", action="store_true",
                        help="Use BM25 lexical retrieval for sub-query hops (entity discrimination)")
    parser.add_argument("--chain_cache", action="store_true",
                        help="Cache completed decompose chains for identical queries")
    parser.add_argument("--rebel_kg", action="store_true",
                        help="Build REBEL knowledge graph at ingest, use graph hops in decompose")
    parser.add_argument("--dream_explore_only", action="store_true",
                        help="Dream with only cross-domain exploration (skip repulsion/prune/merge/unlearn)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable ingestion cache (always re-ingest)")
    parser.add_argument("--refresh_cache", action="store_true",
                        help="Re-ingest and overwrite cached state")
    parser.add_argument("--cache_label", type=str, default=None,
                        help="Custom label for ingestion cache directory")
    parser.add_argument("--list_caches", action="store_true",
                        help="List available ingestion caches and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle --list_caches
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
    elif args.cooc_boost and args.cooc_gate > 0:
        prefix = f"hermes_cooc_g{args.cooc_gate}"
    elif args.cooc_boost:
        prefix = "hermes_cooc"
    elif args.ppr:
        prefix = "hermes_ppr"
    else:
        prefix = "hermes"
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
        elif args.cooc_boost:
            gate_str = f", gate={args.cooc_gate}" if args.cooc_gate > 0 else ""
            print(f"Retrieval: cooc_boost (weight={args.cooc_weight}{gate_str})")
        elif args.ppr:
            print(f"Retrieval: PPR (weight={args.ppr_weight}, damping={args.ppr_damping})")
        else:
            print(f"Retrieval: cosine (baseline)")
    print()

    # Load data
    conversation_creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    print(f"Contexts: {len(all_chunks)}")
    total_queries = sum(len(qa) for qa in all_qa_pairs)
    print(f"Total queries: {total_queries}")

    # Ingestion cache setup
    use_cache = not args.no_cache and not args.fullpass
    cache_path = None
    if use_cache:
        effective_cs = dataset_config.get("chunk_size", 0)
        cache_path = cache_dir(dataset_config["sub_dataset"], effective_cs, args.cache_label)
        print(f"Ingestion cache: {cache_path}")
        if args.refresh_cache:
            print("  (refresh mode: will re-ingest and overwrite)")
    elif not args.fullpass:
        print("Ingestion cache: disabled")
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
            explore_only=args.dream_explore_only,
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
            decompose_query=args.decompose_query,
            conflict_resolution=args.conflict_resolution,
            cr_per_hop=args.cr_per_hop,
            cr_entity_gate=args.cr_entity_gate,
            cr_overfetch=args.cr_overfetch,
            cosine_dedup=args.cosine_dedup,
            cosine_dedup_threshold=args.cosine_dedup_threshold,
            entity_retrieval=args.entity_retrieval,
            entity_first=args.entity_first,
            entity_first_aggressive=args.entity_first_aggressive,
            bm25_hops=args.bm25_hops,
            chain_cache=args.chain_cache,
            rebel_kg=args.rebel_kg,
        )

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

            # Try loading from ingestion cache
            cached = False
            if use_cache and not args.refresh_cache:
                cached = load_context(agent, cache_path, context_idx)

            if not cached:
                # Full ingestion
                print(f"\n--- Context {context_idx}: memorizing {len(chunks)} chunks ---")
                for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Memorizing", leave=False)):
                    agent.send_message(chunk, memorizing=True)
                # Persist embedding cache after memorization
                agent._scorer.save_disk_cache()

                if use_cache:
                    save_context(agent, cache_path, context_idx)

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
