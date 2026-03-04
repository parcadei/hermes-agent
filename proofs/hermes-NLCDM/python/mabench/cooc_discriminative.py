#!/usr/bin/env python3
"""Discriminative test: 3 co-occurrence boost variants vs cosine baseline.

Memorize once, then run 4 retrieval paths per query:
  1. Pure cosine (baseline)
  2. Dense cooc_boost (current, uniform mean over all neighbors)
  3. Sparse top-k cooc (only top-k strongest co-occurrence edges)
  4. Sparsemax cooc (sparsemax attention over neighbor cosine scores)

For each query, records whether retrieval differs from baseline and
which variant changes the LLM answer. Only makes LLM calls for
variants that produce different retrieval results.

Brenner ✂ Exclusion-Test: if all 3 variants produce identical top-k
to cosine baseline at 262k, the co-occurrence graph is structurally
incapable of reranking at this density. Kill the entire approach.

Usage:
    .venv/bin/python mabench/cooc_discriminative.py \
        --dataset_config MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_sh_262k.yaml
"""

from __future__ import annotations

import json
import os
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dotenv
import numpy as np
import yaml
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
_MABENCH = _HERMES_ROOT / "MemoryAgentBench"

sys.path.insert(0, str(_MABENCH))
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))

from conversation_creator import ConversationCreator
from dream_ops import DreamParams
from mabench.hermes_agent import HermesMemoryAgent, EmbeddingRelevanceScorer
from utils.eval_other_utils import metrics_summarization

dotenv.load_dotenv(_NLCDM_PYTHON / ".env")

VARIANTS = ["cosine", "dense_cooc", "sparse_topk", "sparsemax_cooc"]


def parse_args():
    parser = ArgumentParser(description="Discriminative test: 3 cooc variants vs cosine")
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--retrieve_num", type=int, default=10)
    parser.add_argument("--contradiction_threshold", type=float, default=0.95)
    parser.add_argument("--max_test_queries", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--recency_alpha", type=float, default=0.1)
    parser.add_argument("--cooc_weight", type=float, default=1.0)
    parser.add_argument(
        "--neighbor_k", type=int, default=5,
        help="Top-k neighbors for sparse_topk variant",
    )
    return parser.parse_args()


def _generate_answer(llm_client, model, question, context):
    system_msg = (
        "You are a helpful assistant that answers questions from a "
        "knowledge pool. Give a very concise answer."
    )
    user_msg = f"[Knowledge Pool]\n{context}\n\n{question}\nAnswer:"
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    output = response.choices[0].message.content or ""
    usage = response.usage
    return output, usage.prompt_tokens, usage.completion_tokens


def main():
    args = parse_args()

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

    # Initialize agent (cooc structures built during memorization)
    agent = HermesMemoryAgent(
        model=args.model,
        contradiction_threshold=args.contradiction_threshold,
        retrieve_num=args.retrieve_num,
        dream_interval=0,
        temperature=0.0,
        recency_alpha=args.recency_alpha,
        cooc_boost_retrieval=True,
        cooc_weight=args.cooc_weight,
        cooc_gate_threshold=0.0,
    )

    from openai import OpenAI
    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    conversation_creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    print(f"Dataset: {dataset_config['sub_dataset']}")
    print(f"Variants: {VARIANTS}")
    print(f"cooc_weight={args.cooc_weight}, neighbor_k={args.neighbor_k}")
    print(f"Contexts: {len(all_chunks)}, Queries: {sum(len(qa) for qa in all_qa_pairs)}")
    print()

    output_dir = Path(agent_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    output_file = output_dir / f"cooc_discrim_{dataset_config['sub_dataset']}_{safe_model}.json"

    start_time = time.time()

    per_variant_metrics = {v: defaultdict(list) for v in VARIANTS}
    per_variant_results = {v: [] for v in VARIANTS}
    query_log = []
    query_index = 0

    for context_idx, (chunks, qa_pairs) in enumerate(
        tqdm(zip(all_chunks, all_qa_pairs), total=len(all_chunks), desc="Contexts")
    ):
        agent.reset()
        print(f"\n--- Context {context_idx}: memorizing {len(chunks)} chunks ---")
        mem_start = time.time()
        for chunk in tqdm(chunks, desc="Memorizing", leave=False):
            agent.send_message(chunk, memorizing=True)
        agent._scorer.save_disk_cache()
        mem_time = time.time() - mem_start

        # Diagnostic: co-occurrence graph density
        n_facts = agent.coupled_engine.n_memories
        cooc = agent.coupled_engine._co_occurrence
        n_edges = sum(len(v) for v in cooc.values()) // 2
        avg_degree = sum(len(v) for v in cooc.values()) / max(n_facts, 1)
        print(f"    {n_facts} facts, {n_edges} cooc edges, avg degree {avg_degree:.1f}")
        print(f"    Memorized in {mem_time:.1f}s")

        print(f"    Querying {len(qa_pairs)} questions (up to 4 LLM calls each)...")

        for query_data in tqdm(qa_pairs, desc="Querying", leave=False):
            if len(query_data) == 3:
                query, answer, qa_pair_id = query_data
            else:
                query, answer = query_data
                qa_pair_id = None

            if args.max_test_queries > 0 and query_index >= args.max_test_queries:
                break

            bare_question = HermesMemoryAgent._extract_question(query)
            q_emb = agent._scorer.embed(bare_question)

            # V1 retrieval (shared)
            v1_result = agent.orchestrator.query(message=query)
            v1_context = v1_result.context if v1_result.context else ""

            def _build_context(v2_texts):
                seen = set()
                merged = []
                for text in v2_texts:
                    key = text.strip()[:200]
                    if key not in seen:
                        seen.add(key)
                        merged.append(text)
                if v1_context:
                    for line in v1_context.split("\n---\n"):
                        line = line.strip()
                        key = line[:200]
                        if line and key not in seen:
                            seen.add(key)
                            merged.append(line)
                return "\n\n".join(merged[:args.retrieve_num])

            # ── Run all 4 retrieval paths ──────────────────────────

            # 1. Pure cosine
            cosine_res = agent.coupled_engine.query(
                embedding=q_emb, top_k=args.retrieve_num,
            )
            cosine_texts = [r["text"] for r in cosine_res]

            # 2. Dense cooc_boost (current implementation)
            dense_res = agent.coupled_engine.query_cooc_boost(
                embedding=q_emb, top_k=args.retrieve_num,
                cooc_weight=args.cooc_weight, gate_threshold=0.0,
            )
            dense_texts = [r["text"] for r in dense_res]

            # 3. Sparse top-k neighbors
            sparse_topk_res = agent.coupled_engine.query_cooc_sparse_topk(
                embedding=q_emb, top_k=args.retrieve_num,
                cooc_weight=args.cooc_weight, neighbor_k=args.neighbor_k,
            )
            sparse_topk_texts = [r["text"] for r in sparse_topk_res]

            # 4. Sparsemax cooc
            sparsemax_res = agent.coupled_engine.query_cooc_sparsemax(
                embedding=q_emb, top_k=args.retrieve_num,
                cooc_weight=args.cooc_weight,
            )
            sparsemax_texts = [r["text"] for r in sparsemax_res]

            all_texts = {
                "cosine": cosine_texts,
                "dense_cooc": dense_texts,
                "sparse_topk": sparse_topk_texts,
                "sparsemax_cooc": sparsemax_texts,
            }

            # ── Deduplicate LLM calls ─────────────────────────────
            # Group variants by identical retrieval results
            context_groups: dict[str, list[str]] = {}
            variant_to_context_key: dict[str, str] = {}
            for variant, texts in all_texts.items():
                ctx = _build_context(texts)
                # Use hash of context as key
                ctx_key = str(hash(ctx))
                variant_to_context_key[variant] = ctx_key
                if ctx_key not in context_groups:
                    context_groups[ctx_key] = []
                context_groups[ctx_key].append(variant)

            n_unique = len(context_groups)
            differs = {v: all_texts[v] != cosine_texts for v in VARIANTS}

            # Generate answers for each unique context (concurrent)
            ctx_to_answer: dict[str, tuple[str, int, int]] = {}
            unique_contexts = {
                k: _build_context(all_texts[members[0]])
                for k, members in context_groups.items()
            }

            if n_unique == 1:
                # All identical — 1 LLM call
                key = list(unique_contexts.keys())[0]
                ctx_to_answer[key] = _generate_answer(
                    llm_client, args.model, query, unique_contexts[key]
                )
            else:
                # Fan out concurrent LLM calls
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(
                            _generate_answer, llm_client, args.model,
                            query, ctx
                        ): key
                        for key, ctx in unique_contexts.items()
                    }
                    for fut in futures:
                        key = futures[fut]
                        ctx_to_answer[key] = fut.result()

            # ── Record results per variant ────────────────────────
            entry = {
                "query_index": query_index,
                "bare_question": bare_question,
                "gold_answer": answer,
                "n_unique_contexts": n_unique,
                "differs_from_cosine": differs,
                "answers": {},
            }

            for variant in VARIANTS:
                ctx_key = variant_to_context_key[variant]
                ans, in_tok, out_tok = ctx_to_answer[ctx_key]
                entry["answers"][variant] = ans

                agent_output = {
                    "output": ans,
                    "input_len": in_tok,
                    "output_len": out_tok,
                    "memory_construction_time": 0.0,
                    "query_time_len": 0.0,
                }
                per_variant_metrics[variant], per_variant_results[variant] = \
                    metrics_summarization(
                        agent_output, query, answer, dataset_config,
                        per_variant_metrics[variant], per_variant_results[variant],
                        query_index, qa_pair_id,
                    )

            query_log.append(entry)
            query_index += 1

    elapsed = time.time() - start_time

    # ── Results ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Discriminative Test Complete: {query_index} queries in {elapsed:.1f}s")
    print(f"{'='*70}")

    # Retrieval divergence summary
    for v in VARIANTS:
        if v == "cosine":
            continue
        n_diff = sum(1 for q in query_log if q["differs_from_cosine"].get(v, False))
        print(f"  {v}: {n_diff}/{query_index} queries differ from cosine baseline")

    # SubEM comparison
    print(f"\n{'Variant':>20}  {'SubEM':>8}")
    print(f"{'-'*20}  {'-'*8}")
    for v in VARIANTS:
        m = per_variant_metrics[v]
        subem = np.mean(m.get("substring_exact_match", [0])) * 100
        print(f"{v:>20}  {subem:>7.1f}%")

    # Win/Loss/Tie per variant vs cosine
    print(f"\n{'Variant':>20}  {'Win':>5} {'Loss':>5} {'Tie':>5}")
    print(f"{'-'*20}  {'-'*5} {'-'*5} {'-'*5}")
    for v in VARIANTS:
        if v == "cosine":
            continue
        wins, losses, ties = 0, 0, 0
        cos_m = per_variant_metrics["cosine"].get("substring_exact_match", [])
        var_m = per_variant_metrics[v].get("substring_exact_match", [])
        for c, x in zip(cos_m, var_m):
            if x > c:
                wins += 1
            elif x < c:
                losses += 1
            else:
                ties += 1
        print(f"{v:>20}  {wins:>5} {losses:>5} {ties:>5}")

    # Brenner ✂ Exclusion-Test verdict
    any_differ = any(
        q["n_unique_contexts"] > 1 for q in query_log
    )
    print(f"\n--- Brenner ✂ Exclusion-Test ---")
    if not any_differ:
        print("VERDICT: ALL variants produce identical retrieval to cosine baseline.")
        print("The co-occurrence graph is structurally incapable of reranking at this scale.")
        print("† Theory-Kill: co-occurrence boost (all variants) should be removed from 262k pipeline.")
    else:
        n_any_differ = sum(1 for q in query_log if q["n_unique_contexts"] > 1)
        print(f"VERDICT: {n_any_differ}/{query_index} queries show retrieval divergence.")
        print("Co-occurrence structure has discriminative potential. Analyze win/loss above.")

    # Save
    output_data = {
        "config": {
            "dataset": dataset_config["sub_dataset"],
            "model": args.model,
            "cooc_weight": args.cooc_weight,
            "neighbor_k": args.neighbor_k,
        },
        "variants": VARIANTS,
        "subem": {
            v: float(np.mean(per_variant_metrics[v].get("substring_exact_match", [0])) * 100)
            for v in VARIANTS
        },
        "per_query": query_log,
        "time_cost": elapsed,
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
