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

dotenv.load_dotenv()


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
    parser.add_argument("--force", action="store_true", help="Force re-run")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset config
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)

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
    output_file = output_dir / (
        f"hermes_{dataset_config['sub_dataset']}_{args.model}.json"
    )

    print(f"Dataset: {dataset_config['dataset']} / {dataset_config['sub_dataset']}")
    print(f"Model: {args.model}")
    print(f"Output: {output_file}")
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

    # Initialize Hermes agent
    agent = HermesMemoryAgent(
        model=args.model,
        contradiction_threshold=args.contradiction_threshold,
        retrieve_num=args.retrieve_num,
        dream_interval=args.dream_interval,
        temperature=0.0,
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
        # Reset agent for each new context
        agent.reset()

        # Memorize all chunks for this context
        print(f"\n--- Context {context_idx}: memorizing {len(chunks)} chunks ---")
        for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Memorizing", leave=False)):
            agent.send_message(chunk, memorizing=True)

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
