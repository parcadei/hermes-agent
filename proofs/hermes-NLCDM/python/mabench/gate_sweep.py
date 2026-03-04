#!/usr/bin/env python3
"""Single-memorize multi-gate sweep for cooc_boost gate_threshold optimization.

Key insight: the gate threshold is a post-retrieval filter on the max cosine
score (coupled_engine.py:901). For each query there are exactly two possible
retrieval outcomes:
  - GATED (pure cosine): when max_cosine >= gate_threshold
  - UNGATED (cooc boost): when max_cosine < gate_threshold

So we only need 2 LLM calls per query (one per retrieval path), then replay
the gate decision offline for any number of thresholds.

Total cost: memorize once (~10 min for 262k) + 200 LLM calls (100 queries × 2).
Sweep 50+ thresholds with zero additional LLM calls.

Usage:
    .venv/bin/python mabench/gate_sweep.py \
        --dataset_config MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_sh_262k.yaml \
        --n_thresholds 50
"""

from __future__ import annotations

import json
import os
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv
import numpy as np
import yaml
from tqdm import tqdm

# Path setup (same as run_hermes.py)
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
from utils.templates import get_template

dotenv.load_dotenv(_NLCDM_PYTHON / ".env")


def parse_args():
    parser = ArgumentParser(description="Gate threshold sweep with single memorization")
    parser.add_argument(
        "--dataset_config", type=str, required=True,
        help="Path to dataset configuration YAML",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--retrieve_num", type=int, default=10)
    parser.add_argument("--contradiction_threshold", type=float, default=0.95)
    parser.add_argument("--max_test_queries", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--recency_alpha", type=float, default=0.1)
    parser.add_argument("--cooc_weight", type=float, default=1.0)
    parser.add_argument(
        "--n_thresholds", type=int, default=50,
        help="Number of gate thresholds to sweep (linearly spaced 0.3-0.9)",
    )
    parser.add_argument(
        "--gate_min", type=float, default=0.30,
        help="Minimum gate threshold to sweep",
    )
    parser.add_argument(
        "--gate_max", type=float, default=0.90,
        help="Maximum gate threshold to sweep",
    )
    parser.add_argument(
        "--llm_workers", type=int, default=8,
        help="Concurrent LLM API calls (per query, 2 calls fan out)",
    )
    return parser.parse_args()


def _generate_answer(
    llm_client, model: str, question: str, context: str, max_gen_tokens: int = 64,
) -> tuple[str, int, int]:
    """Call LLM to generate an answer from retrieved context."""
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
        max_tokens=max(max_gen_tokens, 16),
    )
    output = response.choices[0].message.content or ""
    usage = response.usage
    return output, usage.prompt_tokens, usage.completion_tokens


def main():
    args = parse_args()

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
        "output_dir": "output/hermes",
        "retrieve_num": args.retrieve_num,
    }

    # Generate threshold grid
    thresholds = np.linspace(args.gate_min, args.gate_max, args.n_thresholds).tolist()
    print(f"Sweeping {len(thresholds)} gate thresholds: "
          f"[{thresholds[0]:.3f} ... {thresholds[-1]:.3f}]")

    # Initialize agent (cooc_boost ON, gate OFF — we'll simulate gating)
    agent = HermesMemoryAgent(
        model=args.model,
        contradiction_threshold=args.contradiction_threshold,
        retrieve_num=args.retrieve_num,
        dream_interval=0,
        temperature=0.0,
        recency_alpha=args.recency_alpha,
        cooc_boost_retrieval=True,
        cooc_weight=args.cooc_weight,
        cooc_gate_threshold=0.0,  # no gating — we do it ourselves
    )

    # LLM client for answer generation
    from openai import OpenAI
    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    # Load data
    conversation_creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    total_queries = sum(len(qa) for qa in all_qa_pairs)
    print(f"Contexts: {len(all_chunks)}")
    print(f"Total queries: {total_queries}")

    # Output
    output_dir = Path(agent_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    output_file = output_dir / f"gate_sweep_{dataset_config['sub_dataset']}_{safe_model}.json"
    print(f"Output: {output_file}\n")

    # ── Phase 1: Memorize (done once) ─────────────────────────────────
    start_time = time.time()

    # Per-threshold metrics accumulators
    # Also track: pure cosine baseline (= gate fires for everything → threshold=0)
    all_thresholds = [0.0] + thresholds  # 0.0 = cosine baseline (always gated)
    per_threshold_metrics: dict[float, dict[str, list]] = {
        t: defaultdict(list) for t in all_thresholds
    }
    per_threshold_results: dict[float, list[dict]] = {
        t: [] for t in all_thresholds
    }
    # Also track ungated cooc_boost (gate never fires → threshold=1.0)
    all_thresholds.append(1.0)
    per_threshold_metrics[1.0] = defaultdict(list)
    per_threshold_results[1.0] = []

    query_index = 0
    query_data_log: list[dict] = []  # per-query diagnostics

    for context_idx, (chunks, qa_pairs) in enumerate(
        tqdm(zip(all_chunks, all_qa_pairs), total=len(all_chunks), desc="Contexts")
    ):
        # Reset and memorize
        agent.reset()
        print(f"\n--- Context {context_idx}: memorizing {len(chunks)} chunks ---")
        mem_start = time.time()
        for chunk in tqdm(chunks, desc="Memorizing", leave=False):
            agent.send_message(chunk, memorizing=True)
        agent._scorer.save_disk_cache()
        mem_time = time.time() - mem_start
        print(f"    Memorized in {mem_time:.1f}s "
              f"({agent.coupled_engine.n_memories} facts in store)")

        # ── Phase 2: Query with dual retrieval ────────────────────────
        print(f"    Querying {len(qa_pairs)} questions (2 LLM calls each)...")

        for query_data in tqdm(qa_pairs, desc="Querying", leave=False):
            if len(query_data) == 3:
                query, answer, qa_pair_id = query_data
            else:
                query, answer = query_data
                qa_pair_id = None

            if args.max_test_queries > 0 and query_index >= args.max_test_queries:
                break

            q_start = time.time()

            # Extract bare question and embed
            bare_question = HermesMemoryAgent._extract_question(query)
            q_emb = agent._scorer.embed(bare_question)

            # ── Get BOTH retrieval paths ──────────────────────────────

            # Path A: Pure cosine (what happens when gate fires)
            cosine_results = agent.coupled_engine.query(
                embedding=q_emb, top_k=args.retrieve_num,
            )
            cosine_texts = [r["text"] for r in cosine_results]

            # Path B: Ungated cooc_boost (what happens when gate doesn't fire)
            boost_results = agent.coupled_engine.query_cooc_boost(
                embedding=q_emb, top_k=args.retrieve_num,
                cooc_weight=args.cooc_weight,
                gate_threshold=0.0,  # force ungated
            )
            boost_texts = [r["text"] for r in boost_results]

            # Record the max cosine score for gate decisions
            # Recompute from the engine (same as coupled_engine.py:894-898)
            emb = np.asarray(q_emb, dtype=np.float64).ravel()
            embeddings = agent.coupled_engine._embeddings_matrix()
            emb_norm = np.linalg.norm(emb)
            norms = np.linalg.norm(embeddings, axis=1) * emb_norm + 1e-12
            cos_scores = embeddings @ emb / norms
            max_cosine = float(cos_scores.max())

            # Check if retrieval paths actually differ
            paths_differ = cosine_texts != boost_texts

            # ── Build contexts and generate answers ───────────────────

            # V1 retrieval (shared across both paths)
            v1_result = agent.orchestrator.query(message=query)
            v1_context = v1_result.context if v1_result.context else ""

            def _build_context(v2_texts: list[str]) -> str:
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

            cosine_context = _build_context(cosine_texts)
            boost_context = _build_context(boost_texts)

            # Generate answers — concurrent if paths differ, single call if same
            if paths_differ:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_cosine = executor.submit(
                        _generate_answer, llm_client, args.model,
                        query, cosine_context,
                    )
                    fut_boost = executor.submit(
                        _generate_answer, llm_client, args.model,
                        query, boost_context,
                    )
                    cosine_answer, cos_in, cos_out = fut_cosine.result()
                    boost_answer, boost_in, boost_out = fut_boost.result()
            else:
                # Same retrieval → same answer, only 1 LLM call needed
                cosine_answer, cos_in, cos_out = _generate_answer(
                    llm_client, args.model, query, cosine_context,
                )
                boost_answer, boost_in, boost_out = cosine_answer, cos_in, cos_out

            query_time = time.time() - q_start

            # ── Log per-query diagnostics ─────────────────────────────
            query_data_log.append({
                "query_index": query_index,
                "max_cosine": max_cosine,
                "paths_differ": paths_differ,
                "cosine_answer": cosine_answer,
                "boost_answer": boost_answer,
                "gold_answer": answer,
                "bare_question": bare_question,
            })

            # ── Replay gate decision for every threshold ──────────────
            for t in all_thresholds:
                # Gate fires (use cosine) when max_cosine >= threshold
                # Special cases: t=0.0 → always gated (pure cosine baseline)
                #                t=1.0 → never gated (ungated cooc_boost)
                if t <= 0.0:
                    use_answer = cosine_answer
                    use_in, use_out = cos_in, cos_out
                elif t >= 1.0:
                    use_answer = boost_answer
                    use_in, use_out = boost_in, boost_out
                elif max_cosine >= t:
                    use_answer = cosine_answer
                    use_in, use_out = cos_in, cos_out
                else:
                    use_answer = boost_answer
                    use_in, use_out = boost_in, boost_out

                agent_output = {
                    "output": use_answer,
                    "input_len": use_in,
                    "output_len": use_out,
                    "memory_construction_time": 0.0,
                    "query_time_len": query_time,
                }

                per_threshold_metrics[t], per_threshold_results[t] = \
                    metrics_summarization(
                        agent_output, query, answer, dataset_config,
                        per_threshold_metrics[t], per_threshold_results[t],
                        query_index, qa_pair_id,
                    )

            query_index += 1

    elapsed = time.time() - start_time

    # ── Results matrix ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Gate Sweep Complete: {query_index} queries in {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"\n{'Threshold':>10}  {'SubEM':>8}  {'Gated%':>8}  Label")
    print(f"{'-'*10}  {'-'*8}  {'-'*8}  {'-'*20}")

    summary_rows = []
    for t in sorted(all_thresholds):
        m = per_threshold_metrics[t]
        subem = np.mean(m.get("substring_exact_match", [0])) * 100
        # Count how many queries had gate fire (used cosine)
        n_gated = sum(1 for q in query_data_log if q["max_cosine"] >= t) if t > 0 else query_index
        gated_pct = n_gated / max(query_index, 1) * 100

        if t <= 0.0:
            label = "cosine baseline"
        elif t >= 1.0:
            label = "ungated cooc_boost"
        else:
            label = ""

        print(f"{t:>10.3f}  {subem:>7.1f}%  {gated_pct:>7.1f}%  {label}")
        summary_rows.append({
            "gate_threshold": t,
            "subem": subem,
            "gated_pct": gated_pct,
            "label": label,
            "metrics": {
                k: float(np.mean(v) * (1 if ("_len" in k) or ("_time" in k) else 100))
                for k, v in m.items()
            },
        })

    # Find optimal threshold
    best = max(summary_rows, key=lambda r: r["subem"])
    print(f"\nBest: gate={best['gate_threshold']:.3f} → SubEM {best['subem']:.1f}%")

    # ── Save full results ─────────────────────────────────────────────
    output_data = {
        "config": {
            "dataset": dataset_config["sub_dataset"],
            "model": args.model,
            "retrieve_num": args.retrieve_num,
            "cooc_weight": args.cooc_weight,
            "n_thresholds": args.n_thresholds,
            "gate_range": [args.gate_min, args.gate_max],
        },
        "summary": summary_rows,
        "best": {
            "gate_threshold": best["gate_threshold"],
            "subem": best["subem"],
        },
        "per_query": query_data_log,
        "time_cost": elapsed,
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
