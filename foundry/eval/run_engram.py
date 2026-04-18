#!/usr/bin/env python3
"""Run ANY MABench evaluation with Engram.

Isolated runner — does NOT use the composed agent (mabench.agent).
Uses Engram's Foundry API directly: ingest chunks → retrieve answers.

Supports all MABench benchmark categories:
  - Conflict_Resolution: FC-SH, FC-MH (fact supersession)
  - Accurate_Retrieval: EventQA, LongMemEval, Ruler QA
  - Long_Range_Understanding: Detective_QA, InfBench_sum
  - Test_Time_Learning: ICL, Recsys

Usage:
    # FC single-hop
    python3 -u -m foundry.eval.run_engram \
        --dataset_config path/to/Factconsolidation_sh_262k.yaml --chunk_size 512

    # EventQA
    python3 -u -m foundry.eval.run_engram \
        --dataset_config path/to/Eventqa_full.yaml --chunk_size 4096

    # Detective QA
    python3 -u -m foundry.eval.run_engram \
        --dataset_config path/to/Detective_QA.yaml --chunk_size 4096
"""

from __future__ import annotations

import json
import re
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dotenv
import numpy as np
import openai
import yaml

# ---------------------------------------------------------------------------
# Path setup — add MABench and proofs to sys.path
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_FOUNDRY_ROOT = _THIS_DIR.parent
_WORKTREE_ROOT = _FOUNDRY_ROOT.parent
_HERMES_ROOT = _WORKTREE_ROOT.parent.parent.parent  # engram-build -> worktrees -> .claude -> hermes-agent
_MABENCH = _HERMES_ROOT / "MemoryAgentBench"
_NLCDM_PYTHON = _HERMES_ROOT / "proofs" / "hermes-NLCDM" / "python"

sys.path.insert(0, str(_WORKTREE_ROOT))

import importlib.util

def _import_from(module_name: str, file_path: Path):
    """Import a module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Pre-load MABench's utils package to avoid collision with worktree utils.py
if (_MABENCH / "utils" / "__init__.py").exists():
    _import_from("utils", _MABENCH / "utils" / "__init__.py")
_import_from("utils.eval_other_utils", _MABENCH / "utils" / "eval_other_utils.py")
_import_from("utils.eval_data_utils", _MABENCH / "utils" / "eval_data_utils.py")
_import_from("utils.templates", _MABENCH / "utils" / "templates.py")

sys.path.insert(0, str(_MABENCH))
sys.path.insert(0, str(_NLCDM_PYTHON))

from conversation_creator import ConversationCreator
from utils.eval_other_utils import metrics_summarization

dotenv.load_dotenv(_NLCDM_PYTHON / ".env")

from foundry.core import Foundry, FoundryConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(description="Engram MABench evaluation runner")

    # Dataset
    parser.add_argument("--dataset_config", type=str, required=True,
                        help="Path to MABench dataset config YAML")
    parser.add_argument("--chunk_size", type=int, default=512)

    # Engram config
    parser.add_argument("--extraction_url", type=str, default=None,
                        help="vLLM endpoint for LLM extraction (None = regex fallback)")
    parser.add_argument("--agent_model", type=str, default="gpt-4o-mini",
                        help="LLM model for answer generation")
    parser.add_argument("--conflict_resolution", action="store_true",
                        help="Enable conflict resolution on retrieval")
    parser.add_argument("--retrieve_top_k", type=int, default=20,
                        help="Number of BM25 chunks to retrieve as evidence")
    parser.add_argument("--use_agentic", action="store_true",
                        help="Use AgenticRetriever instead of direct BM25")

    # Ground truth correction (FC-MH only)
    parser.add_argument("--corrected_answers", type=str, default=None,
                        help="JSON file with corrected ground truth")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default=str(_NLCDM_PYTHON / "output" / "engram"))
    parser.add_argument("--tag", type=str, default=None,
                        help="Extra tag for output filename")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Query extraction — strip benchmark prompt wrappers to get the actual question
# ---------------------------------------------------------------------------

def extract_retrieval_query(raw_query: str, dataset: str) -> str:
    """Extract the actual question from a benchmark prompt for BM25 retrieval.

    Different benchmarks wrap the real question in different prompt templates.
    This strips the wrapper to get a clean query for BM25/entity search.
    """
    # FC and Ruler: "... Now Answer the Question: <actual Q>\nAnswer:"
    marker = "Now Answer the Question:"
    if marker in raw_query:
        q = raw_query.split(marker)[-1].strip()
        if q.endswith("Answer:"):
            q = q[:-len("Answer:")].strip()
        # Strip FC's "Based on the provided Knowledge Pool, "
        pool_prefix = "Based on the provided Knowledge Pool, "
        if q.startswith(pool_prefix):
            q = q[len(pool_prefix):]
        return q

    # Detective_QA: "... Question: <Q>\nA. ...\nB. ..."
    # Extract the question stem (before the options)
    if "detective" in dataset:
        m = re.search(r"Question:\s*(.+?)(?:\s*\n\s*A\.|$)", raw_query, re.DOTALL)
        if m:
            return m.group(1).strip()

    # EventQA: "These are the events that have already occurred: ... which event happens next"
    # The query IS the event list + options — use the event descriptions for retrieval
    if "eventqa" in dataset:
        m = re.search(r"These are the events that have already occurred:\s*(.+?)Below is a list",
                       raw_query, re.DOTALL)
        if m:
            return m.group(1).strip()

    # ICL: "Question:<text>\nlabel:"
    if "icl_" in dataset:
        m = re.search(r"Question:\s*(.+?)(?:\s*\n\s*label:|$)", raw_query, re.DOTALL)
        if m:
            return m.group(1).strip()

    # InfBench_sum, Recsys, LME, and anything else: use as-is
    return raw_query


# ---------------------------------------------------------------------------
# Answer generation — LLM produces answer from retrieved evidence
# ---------------------------------------------------------------------------

_llm_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _llm_client
    if _llm_client is None:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = None
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if api_key:
                base_url = "https://openrouter.ai/api/v1"
        if not api_key:
            raise RuntimeError("No OPENAI_API_KEY or OPENROUTER_API_KEY found in environment")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _llm_client = openai.OpenAI(**kwargs)
    return _llm_client


def _system_prompt_for_dataset(dataset: str) -> str:
    """Return an appropriate system prompt based on dataset type."""
    if "factconsolidation" in dataset:
        return (
            "You are a knowledge management system. Answer the question concisely "
            "using ONLY the provided evidence. Give a very short answer (a few words). "
            "If the evidence contains conflicting information, prefer the fact with "
            "the HIGHEST serial number (newest). If the evidence doesn't contain the "
            "answer, say 'unknown'."
        )
    if "detective" in dataset:
        return (
            "You are answering a multiple-choice question about a detective story. "
            "Use the provided evidence to determine the correct answer. "
            "Output ONLY the answer choice (e.g. 'C. The Brandt couple')."
        )
    if "eventqa" in dataset:
        return (
            "You are answering a question about what event happens next in a story. "
            "Use the provided evidence to determine the correct answer. "
            "Output ONLY the event that happens next, exactly as it appears in the options."
        )
    if "icl_" in dataset:
        return (
            "You are a text classifier. Based on the provided labeled examples as evidence, "
            "classify the query text. Output ONLY the numerical label."
        )
    if "infbench_sum" in dataset:
        return (
            "Summarize the provided evidence concisely. Include all key points, "
            "characters, events, and conclusions."
        )
    if "ruler" in dataset:
        return (
            "Answer the question concisely using ONLY the provided evidence. "
            "Give a very short answer (a few words). "
            "If the evidence doesn't contain the answer, say 'unknown'."
        )
    if "recsys" in dataset:
        return (
            "You are a recommendation system. Based on the provided evidence about "
            "user preferences, recommend items. Be concise."
        )
    # Generic fallback
    return (
        "Answer the question concisely using ONLY the provided evidence. "
        "If the evidence doesn't contain the answer, say 'unknown'."
    )


def generate_answer(query: str, evidence: str, model: str, dataset: str,
                    max_tokens: int = 64) -> str:
    """Use LLM to produce an answer from retrieved evidence."""
    if not evidence:
        return ""
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _system_prompt_for_dataset(dataset)},
            {"role": "user", "content": (
                f"Here is relevant context from memory:\n{evidence}\n\n"
                f"Now complete the following task:\n{query}"
            )},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    start_time = time.time()

    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)

    dataset_config["chunk_size"] = args.chunk_size
    sub_ds = dataset_config["sub_dataset"]
    gen_max_len = dataset_config.get("generation_max_length", 50)

    agent_config = {"agent_name": "rag", "agent_chunk_size": None}

    # Load corrected answers (FC-MH only)
    corrected_map: dict[int, str] = {}
    if args.corrected_answers:
        with open(args.corrected_answers) as f:
            corrections = json.load(f)
        for c in corrections:
            corrected_map[c["qi"]] = c["corrected_answer"]
        print(f"Loaded {len(corrected_map)} corrected answers from {args.corrected_answers}")

    # Build Foundry
    config = FoundryConfig(
        chunk_size=args.chunk_size,
        extraction_url=args.extraction_url,
        agent_model=args.agent_model,
    )

    # Load data
    conversation_creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    total_queries = sum(len(qa) for qa in all_qa_pairs)
    print(f"Dataset: {sub_ds}")
    print(f"Contexts: {len(all_chunks)}, Total queries: {total_queries}")
    print(f"Engram config: chunk_size={args.chunk_size}, extraction_url={args.extraction_url}")
    print(f"  agent_model={args.agent_model}, conflict_resolution={args.conflict_resolution}")
    print(f"  retrieve_top_k={args.retrieve_top_k}, use_agentic={args.use_agentic}")
    print(f"  generation_max_length={gen_max_len}")
    print()

    # Output setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_parts = [f"cs{args.chunk_size}"]
    if args.conflict_resolution:
        tag_parts.append("cr")
    if args.extraction_url:
        tag_parts.append("llmext")
    if args.corrected_answers:
        tag_parts.append("corrgt")
    if args.tag:
        tag_parts.append(args.tag)
    tag_str = "_".join(tag_parts)

    output_file = output_dir / f"{timestamp}_engram_{tag_str}_{sub_ds}_{args.agent_model}.json"
    print(f"Output: {output_file}")
    print()

    # Evaluation loop
    metrics: dict[str, list] = defaultdict(list)
    results: list[dict] = []
    query_index = 0

    for context_idx in range(len(all_chunks)):
        chunks = all_chunks[context_idx]
        qa_pairs = all_qa_pairs[context_idx]

        # Reset Foundry for each context
        foundry = Foundry(config)

        # Ingest
        print(f"\n--- Context {context_idx}: ingesting {len(chunks)} chunks ---")
        ingest_start = time.time()
        for chunk in chunks:
            foundry.ingest(chunk)
        ingest_time = time.time() - ingest_start
        print(f"  Ingested in {ingest_time:.1f}s")

        # Query
        print(f"  Querying {len(qa_pairs)} questions...")
        for query_data in qa_pairs:
            if len(query_data) == 3:
                query, answer, qa_pair_id = query_data
            else:
                query, answer = query_data
                qa_pair_id = None

            # Override broken ground truth
            if query_index in corrected_map:
                answer = corrected_map[query_index]

            # Extract actual question for retrieval
            retrieval_query = extract_retrieval_query(query, sub_ds)

            # Retrieve evidence
            query_start = time.time()
            if args.use_agentic:
                evidence, _trace = foundry.retrieve(retrieval_query)
            else:
                # Direct BM25 with high top_k — bypasses AgenticRetriever
                bm25_hits = foundry._store.bm25_chunks.query(
                    retrieval_query, top_k=args.retrieve_top_k
                )
                evidence_parts = []
                char_budget = 12000  # ~3K tokens, fits gpt-4o-mini context
                total_chars = 0
                for doc_id, score in bm25_hits:
                    text = foundry._store.bm25_chunks._texts.get(doc_id, "")
                    if not text.strip():
                        continue
                    # Always include at least one result; truncate if needed
                    if total_chars + len(text) > char_budget:
                        if not evidence_parts:
                            evidence_parts.append(text[:char_budget])
                            total_chars += char_budget
                        break
                    evidence_parts.append(text)
                    total_chars += len(text)
                evidence = "\n".join(evidence_parts)

            # Generate answer from evidence
            # Use full query for LLM (includes options/instructions),
            # retrieval_query was only for BM25 search
            answer_text = generate_answer(
                query, evidence or "", args.agent_model,
                sub_ds, max_tokens=gen_max_len,
            )
            query_time = time.time() - query_start

            if query_index < 3:
                print(f"    [{query_index}] Q: {retrieval_query[:80]}")
                print(f"    [{query_index}] Evidence: {(evidence or '')[:120]}")
                print(f"    [{query_index}] Predicted: {answer_text[:80]}")
                print(f"    [{query_index}] Expected: {str(answer)[:80]}")

            output = {
                "output": answer_text,
                "parsed_output": answer_text,
                "input_len": len(query),
                "output_len": len(answer_text),
                "memory_construction_time": ingest_time / max(len(chunks), 1),
                "query_time_len": query_time,
            }

            metrics, results = metrics_summarization(
                output, query, answer, dataset_config,
                metrics, results, query_index, qa_pair_id,
            )
            query_index += 1

    # Save results
    elapsed = time.time() - start_time
    averaged_metrics = {
        key: np.mean(values) * (1 if ("_len" in key) or ("_time" in key) else 100)
        for key, values in metrics.items()
    }

    output_data = {
        "agent_config": {"agent_name": "engram", "version": "0.1"},
        "dataset_config": dataset_config,
        "engram_config": {
            "chunk_size": args.chunk_size,
            "extraction_url": args.extraction_url,
            "agent_model": args.agent_model,
            "conflict_resolution": args.conflict_resolution,
            "corrected_answers": bool(args.corrected_answers),
        },
        "data": results,
        "metrics": {k: v for k, v in metrics.items()},
        "averaged_metrics": averaged_metrics,
        "time_cost": elapsed,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Evaluation complete: {query_index} queries in {elapsed:.1f}s")
    print(f"Results saved to: {output_file}")
    print(f"\nMetrics:")
    for key, values in metrics.items():
        scale = 1 if ("_len" in key) or ("_time" in key) else 100
        print(f"  {key}: {np.mean(values) * scale:.2f}")


if __name__ == "__main__":
    main()
