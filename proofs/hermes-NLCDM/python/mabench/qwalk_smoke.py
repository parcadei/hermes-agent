"""Smoke test: does quantum walk produce different retrieval than cosine at 262k?

Sweeps walk_time parameter for a few queries and reports:
  - Number of top-10 facts that differ from cosine baseline at each t
  - The t value with maximum divergence (if any)

If divergence is 0 across all t values → quantum walk is structurally
incapable of reranking on this graph (falsified).
If divergence > 0 → candidate for full 100-query Brenner discriminative test.

Usage:
  python mabench/qwalk_smoke.py \\
    --dataset_config MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_sh_262k.yaml \\
    --n_queries 5 --n_t_values 50
"""

import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import dotenv
import numpy as np

_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
_MABENCH = _HERMES_ROOT / "MemoryAgentBench"

sys.path.insert(0, str(_MABENCH))
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))

import yaml
from conversation_creator import ConversationCreator
from mabench.hermes_agent import HermesMemoryAgent

dotenv.load_dotenv(_NLCDM_PYTHON / ".env")


def main():
    parser = ArgumentParser(description="Quantum walk smoke test")
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--retrieve_num", type=int, default=10)
    parser.add_argument("--n_queries", type=int, default=5,
                        help="Number of queries to test (0 = all)")
    parser.add_argument("--n_t_values", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--recency_alpha", type=float, default=0.1)
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
        "output_dir": "output/hermes",
        "retrieve_num": args.retrieve_num,
    }

    agent = HermesMemoryAgent(
        model=args.model,
        retrieve_num=args.retrieve_num,
        dream_interval=0,
        temperature=0.0,
        recency_alpha=args.recency_alpha,
        cooc_boost_retrieval=True,
        cooc_weight=1.0,
        cooc_gate_threshold=0.0,
    )

    conversation_creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    print(f"Dataset: {dataset_config['sub_dataset']}")
    print(f"Contexts: {len(all_chunks)}, Queries: {sum(len(qa) for qa in all_qa_pairs)}")

    # --- Memorize first context ---
    chunks = all_chunks[0]
    qa_pairs = all_qa_pairs[0]

    agent.reset()
    print(f"\nMemorizing {len(chunks)} chunks...")
    from tqdm import tqdm
    for chunk in tqdm(chunks, desc="Memorizing"):
        agent.send_message(chunk, memorizing=True)
        agent.coupled_engine.flush_session()  # Build per-chunk co-occurrence edges
    agent._scorer.save_disk_cache()

    engine = agent.coupled_engine
    n_facts = engine.n_memories
    cooc = engine._co_occurrence
    n_edges = sum(len(v) for v in cooc.values()) // 2
    avg_degree = sum(len(v) for v in cooc.values()) / max(n_facts, 1)
    print(f"  {n_facts} facts, {n_edges} cooc edges, avg degree {avg_degree:.1f}")

    # --- Build t sweep values ---
    t_linear = np.linspace(0.01, 1.0, args.n_t_values // 2)
    t_log = np.logspace(0, 2, args.n_t_values // 2)  # 1.0 to 100.0
    t_values = np.unique(np.concatenate([t_linear, t_log]))
    print(f"\nSweeping {len(t_values)} t values from {t_values.min():.3f} to {t_values.max():.1f}")

    # --- Query loop ---
    n_queries = args.n_queries if args.n_queries > 0 else len(qa_pairs)
    n_queries = min(n_queries, len(qa_pairs))

    global_best_div = 0
    global_best_t = None
    global_best_query = None

    for qi in range(n_queries):
        query_data = qa_pairs[qi]
        if len(query_data) == 3:
            query, answer, _ = query_data
        else:
            query, answer = query_data

        bare_question = HermesMemoryAgent._extract_question(query)
        q_emb = agent._scorer.embed(bare_question)

        # Cosine baseline
        cos_results = engine.query(q_emb, top_k=args.retrieve_num)
        cos_indices = set(r["index"] for r in cos_results)

        best_div = 0
        best_t = None
        divs_at_t = []

        for t in t_values:
            # Blended (0.5)
            qw_res = engine.query_quantum_walk(
                q_emb, top_k=args.retrieve_num,
                walk_time=float(t), qw_weight=0.5,
            )
            qw_indices = set(r["index"] for r in qw_res)
            div = len(cos_indices - qw_indices)

            # Pure QW (1.0)
            qw_pure = engine.query_quantum_walk(
                q_emb, top_k=args.retrieve_num,
                walk_time=float(t), qw_weight=1.0,
            )
            qw_pure_indices = set(r["index"] for r in qw_pure)
            div_pure = len(cos_indices - qw_pure_indices)

            max_div = max(div, div_pure)
            divs_at_t.append((float(t), div, div_pure))

            if max_div > best_div:
                best_div = max_div
                best_t = float(t)

        nonzero = sum(1 for _, d, dp in divs_at_t if d > 0 or dp > 0)
        print(f"\n  Q{qi}: {bare_question[:70]}...")
        print(f"    Max divergence: {best_div}/10 (blended or pure)")
        print(f"    t values with any divergence: {nonzero}/{len(t_values)}")

        if best_div > 0:
            print(f"    DIVERGENT at t={best_t:.4f}")
            # Show profile
            for t_val, d, dp in divs_at_t:
                if d > 0 or dp > 0:
                    print(f"      t={t_val:8.4f}  blend_div={d}  pure_div={dp}")

        if best_div > global_best_div:
            global_best_div = best_div
            global_best_t = best_t
            global_best_query = qi

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"Quantum Walk Smoke Test Summary")
    print("=" * 70)
    print(f"  Dataset: {dataset_config['sub_dataset']}")
    print(f"  Facts: {n_facts}, Edges: {n_edges}, Avg degree: {avg_degree:.1f}")
    print(f"  Queries tested: {n_queries}")
    print(f"  t values per query: {len(t_values)}")
    print(f"  Global max divergence: {global_best_div}/10")

    if global_best_div > 0:
        print(f"\n  VERDICT: DIVERGENCE FOUND")
        print(f"  Best t={global_best_t:.4f} on query {global_best_query}")
        print(f"  → Quantum walk produces different retrieval than cosine.")
        print(f"  → Run full Brenner discriminative test with t={global_best_t:.4f}")
    else:
        print(f"\n  VERDICT: ALL IDENTICAL")
        print(f"  Quantum walk produces identical retrieval to cosine across")
        print(f"  {n_queries} queries × {len(t_values)} t values.")
        print(f"  → Graph eigenstructure does not support discriminative interference.")
        print(f"  → Theory-Kill quantum walk on this graph topology.")


if __name__ == "__main__":
    main()
