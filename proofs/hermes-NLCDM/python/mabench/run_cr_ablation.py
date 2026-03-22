#!/usr/bin/env python3
"""Query-time conflict resolution ablation: 4 configs from 1 ingestion.

All configs share the same ingested state (cooc+triadic+decompose+chunk512).
Only the query-time CR variant differs. Ingestion happens ONCE per context,
then each config runs queries against a deepcopy of that state.

Usage (on RunPod):
    python3 -u mabench/run_cr_ablation.py \
        --dataset_config MemoryAgentBench/configs/data_conf/Conflict_Resolution/factconsolidation_mh_262k.yaml \
        --chunk_size 512

Configs:
    1. baseline:         No CR (expect SubEM=22)
    2. cr_acl:           Query-time CR with acl-aware copular extraction
    3. cr_acl_perhop:    CR + per-hop dedup (before each intermediate answer)
    4. cr_entity_gate:   CR + per-hop + NER entity overlap required
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import dotenv
import numpy as np
import yaml
from tqdm import tqdm

# Path setup
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
from utils.eval_other_utils import metrics_summarization

dotenv.load_dotenv(_NLCDM_PYTHON / ".env")


CONFIGS = [
    {
        "name": "baseline",
        "conflict_resolution": False,
        "cr_per_hop": False,
        "cr_entity_gate": False,
    },
    {
        "name": "cr_acl",
        "conflict_resolution": True,
        "cr_per_hop": False,
        "cr_entity_gate": False,
    },
    {
        "name": "cr_acl_perhop",
        "conflict_resolution": True,
        "cr_per_hop": True,
        "cr_entity_gate": False,
    },
    {
        "name": "cr_entity_gate",
        "conflict_resolution": True,
        "cr_per_hop": True,
        "cr_entity_gate": True,
    },
]


def make_agent(model: str) -> HermesMemoryAgent:
    """Create agent with the winner config (CR flags don't matter for ingestion)."""
    return HermesMemoryAgent(
        model=model,
        retrieve_num=10,
        temperature=0.0,
        recency_alpha=0.1,
        dream_interval=0,
        cooc_boost_retrieval=True,
        cooc_weight=1.0,
        triadic_retrieval=True,
        decompose_query=True,
        decompose_max_hops=3,
        conflict_resolution=False,
        cr_per_hop=False,
        cr_entity_gate=False,
    )


def _snapshot_agent_state(agent: HermesMemoryAgent) -> dict:
    """Deepcopy mutable state that querying modifies.

    Querying updates: access_count, last_access_time, importance (on
    retrieved memories), co_retrieval graph, and co_retrieval_query_count.
    These live inside coupled_engine. Triadic and belief_index are read-only
    during queries but snapshot them for safety.
    """
    snap = {
        "coupled_engine": copy.deepcopy(agent.coupled_engine),
    }
    if getattr(agent, "_triadic", None) is not None:
        snap["triadic"] = copy.deepcopy(agent._triadic)
    if getattr(agent, "_belief_index", None) is not None:
        snap["belief_index"] = copy.deepcopy(agent._belief_index)
    return snap


def _restore_agent_state(agent: HermesMemoryAgent, snap: dict) -> None:
    """Restore agent to post-ingestion state from snapshot."""
    agent.coupled_engine = copy.deepcopy(snap["coupled_engine"])
    if "triadic" in snap:
        agent._triadic = copy.deepcopy(snap["triadic"])
    if "belief_index" in snap:
        agent._belief_index = copy.deepcopy(snap["belief_index"])


def _apply_cr_flags(agent: HermesMemoryAgent, cfg: dict) -> None:
    """Toggle query-time CR flags on an existing agent."""
    agent.conflict_resolution = cfg["conflict_resolution"]
    agent._cr_per_hop = cfg["cr_per_hop"]
    agent._cr_entity_gate = cfg["cr_entity_gate"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--configs", type=str, default="all",
                        help="Comma-separated config names or 'all'")
    args = parser.parse_args()

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
        "output_dir": "output/cr_ablation",
        "retrieve_num": 10,
    }

    output_dir = Path(agent_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    conversation_creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    # Select configs
    if args.configs == "all":
        selected = CONFIGS
    else:
        names = [n.strip() for n in args.configs.split(",")]
        selected = [c for c in CONFIGS if c["name"] in names]

    n_contexts = len(all_chunks)
    n_queries = sum(len(qa) for qa in all_qa_pairs)
    print(f"Dataset: {dataset_config['dataset']} / {dataset_config['sub_dataset']}")
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Contexts: {n_contexts}")
    print(f"Total queries: {n_queries}")
    print(f"Configs: {[c['name'] for c in selected]}")
    print(f"Strategy: ingest once per context, query {len(selected)}x")
    print()

    # Create ONE agent — CR flags are irrelevant during ingestion
    agent = make_agent(model=args.model)

    # Per-config accumulators
    config_metrics: dict[str, dict[str, list]] = {
        cfg["name"]: defaultdict(list) for cfg in selected
    }
    config_results: dict[str, list[dict]] = {
        cfg["name"]: [] for cfg in selected
    }
    config_query_idx: dict[str, int] = {
        cfg["name"]: 0 for cfg in selected
    }

    start_time = time.time()

    for context_idx, (chunks, qa_pairs) in enumerate(
        tqdm(
            zip(all_chunks, all_qa_pairs),
            total=n_contexts,
            desc="Contexts",
        )
    ):
        # ── INGEST ONCE ──────────────────────────────────────────
        agent.reset()
        print(f"\n--- Context {context_idx}: memorizing {len(chunks)} chunks ---")
        for chunk in tqdm(chunks, desc="Memorizing", leave=False):
            agent.send_message(chunk, memorizing=True)
        agent._scorer.save_disk_cache()

        # Snapshot post-ingestion state (coupled_engine + triadic + belief)
        snap = _snapshot_agent_state(agent)

        # ── QUERY EACH CONFIG ────────────────────────────────────
        for cfg in selected:
            cfg_name = cfg["name"]

            # Restore clean post-ingestion state
            _restore_agent_state(agent, snap)
            _apply_cr_flags(agent, cfg)

            print(f"    [{cfg_name}] Querying {len(qa_pairs)} questions...")
            for query_data in tqdm(qa_pairs, desc=f"[{cfg_name}]", leave=False):
                if len(query_data) == 3:
                    query, answer, qa_pair_id = query_data
                else:
                    query, answer = query_data
                    qa_pair_id = None

                agent_output = agent.send_message(query, memorizing=False)

                config_metrics[cfg_name], config_results[cfg_name] = (
                    metrics_summarization(
                        agent_output,
                        query,
                        answer,
                        dataset_config,
                        config_metrics[cfg_name],
                        config_results[cfg_name],
                        config_query_idx[cfg_name],
                        qa_pair_id,
                    )
                )
                config_query_idx[cfg_name] += 1

    elapsed = time.time() - start_time

    # Save per-config results
    summary = {}
    for cfg in selected:
        cfg_name = cfg["name"]
        averaged = {
            key: np.mean(values) * (1 if ("_len" in key) or ("_time" in key) else 100)
            for key, values in config_metrics[cfg_name].items()
        }
        summary[cfg_name] = averaged

        output_file = output_dir / f"cr_ablation_{cfg_name}.json"
        output_data = {
            "config_name": cfg_name,
            "conflict_resolution": cfg["conflict_resolution"],
            "cr_per_hop": cfg["cr_per_hop"],
            "cr_entity_gate": cfg["cr_entity_gate"],
            "dataset_config": dataset_config,
            "data": config_results[cfg_name],
            "metrics": dict(config_metrics[cfg_name]),
            "averaged_metrics": averaged,
            "time_cost": elapsed,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[{cfg_name}] Metrics: {averaged}")

    # Print summary table
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'SubEM':>8} {'EM':>8} {'F1':>8}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for name, m in summary.items():
        subem = m.get("substring_exact_match", 0)
        em = m.get("exact_match", 0)
        f1 = m.get("f1", 0)
        print(f"{name:<20} {subem:>8.1f} {em:>8.1f} {f1:>8.1f}")
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
