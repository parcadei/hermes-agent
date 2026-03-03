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
    return parser.parse_args()


def main():
    args = parse_args()

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
    prefix = "fullpass" if args.fullpass else "hermes"
    output_file = output_dir / f"{prefix}_{dataset_config['sub_dataset']}_{safe_model}.json"

    print(f"Dataset: {dataset_config['dataset']} / {dataset_config['sub_dataset']}")
    print(f"Model: {args.model}")
    print(f"Mode: {'FULLPASS (no memory system)' if args.fullpass else 'Hermes memory agent'}")
    print(f"Output: {output_file}")
    if not args.fullpass:
        print(f"Contradiction threshold: {args.contradiction_threshold}")
        print(f"Dream interval: {args.dream_interval}")
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
        agent = HermesMemoryAgent(
            model=args.model,
            contradiction_threshold=args.contradiction_threshold,
            retrieve_num=args.retrieve_num,
            dream_interval=args.dream_interval,
            temperature=0.0,
            recency_alpha=args.recency_alpha,
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
            # Memorize all chunks for this context
            print(f"\n--- Context {context_idx}: memorizing {len(chunks)} chunks ---")
            for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Memorizing", leave=False)):
                agent.send_message(chunk, memorizing=True)
            # Persist embedding cache after memorization
            agent._scorer.save_disk_cache()

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
