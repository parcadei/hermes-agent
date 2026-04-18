#!/usr/bin/env python3
"""Run FactConsolidation (FC-SH / FC-MH) evaluation with Engram.

Isolated runner — does NOT use the composed agent (mabench.agent).
Uses Engram's Foundry API directly: ingest chunks → retrieve answers.

Usage (from hermes-agent root):
    python3 -u -m foundry.eval.run_engram_fc \
        --dataset_config MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_mh_262k.yaml \
        --chunk_size 512

    # With corrected ground truth:
    python3 -u -m foundry.eval.run_engram_fc \
        --dataset_config MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_mh_262k.yaml \
        --chunk_size 512 \
        --corrected_answers output/factconsolidation_mh_corrected_answers.json
"""

from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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

# Foundry worktree root for foundry.* imports
sys.path.insert(0, str(_WORKTREE_ROOT))

# MABench has its own utils/ package that conflicts with worktree utils.py.
# Import MABench modules via importlib to avoid the collision.
import importlib.util

def _import_from(module_name: str, file_path: Path):
    """Import a module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Pre-load MABench's utils package so conversation_creator can find it
_mabench_utils = _import_from("utils", _MABENCH / "utils" / "__init__.py") if (_MABENCH / "utils" / "__init__.py").exists() else None
_import_from("utils.eval_other_utils", _MABENCH / "utils" / "eval_other_utils.py")
_import_from("utils.eval_data_utils", _MABENCH / "utils" / "eval_data_utils.py")
_import_from("utils.templates", _MABENCH / "utils" / "templates.py")

sys.path.insert(0, str(_MABENCH))
sys.path.insert(0, str(_NLCDM_PYTHON))

from conversation_creator import ConversationCreator
from utils.eval_other_utils import metrics_summarization

# Engram imports
from foundry.core import Foundry, FoundryConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(description="Engram FC evaluation runner")

    # Dataset
    parser.add_argument("--dataset_config", type=str, required=True,
                        help="Path to MABench dataset config YAML")
    parser.add_argument("--chunk_size", type=int, default=512)

    # Engram config
    parser.add_argument("--extraction_url", type=str, default=None,
                        help="vLLM endpoint for LLM extraction (None = regex fallback)")
    parser.add_argument("--agent_model", type=str, default="gpt-4o-mini",
                        help="LLM model for agentic retrieval")
    parser.add_argument("--conflict_resolution", action="store_true",
                        help="Enable conflict resolution on retrieval")
    parser.add_argument("--retrieve_top_k", type=int, default=10)

    # Ground truth correction
    parser.add_argument("--corrected_answers", type=str, default=None,
                        help="JSON file with corrected ground truth (overrides dataset answers)")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default=str(_NLCDM_PYTHON / "output" / "engram"))
    parser.add_argument("--tag", type=str, default=None,
                        help="Extra tag for output filename")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Answer generation — LLM produces concise answer from retrieved evidence
# ---------------------------------------------------------------------------

_llm_client: openai.OpenAI | None = None


def _get_client(base_url: str | None = None) -> openai.OpenAI:
    global _llm_client
    if _llm_client is None:
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        _llm_client = openai.OpenAI(**kwargs)
    return _llm_client


def generate_answer(query: str, evidence: str, model: str, max_tokens: int = 64) -> str:
    """Use LLM to produce a concise answer from retrieved evidence."""
    if not evidence:
        return ""
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": (
                "You are a knowledge management system. Answer the question concisely "
                "using ONLY the provided evidence. Give a very short answer (a few words). "
                "If the evidence doesn't contain the answer, say 'unknown'."
            )},
            {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {query}\nAnswer:"},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    start_time = time.time()

    # Load dataset config
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)

    # Override chunk_size from CLI
    dataset_config["chunk_size"] = args.chunk_size

    # Minimal agent_config for ConversationCreator
    # Use "rag" to match MABench's template system (basic chunk+retrieve pattern)
    agent_config = {
        "agent_name": "rag",
        "agent_chunk_size": None,
    }

    # Load corrected answers
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
    foundry = Foundry(config)

    # Load data
    conversation_creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    print(f"Contexts: {len(all_chunks)}")
    print(f"Engram config: chunk_size={args.chunk_size}, extraction_url={args.extraction_url}")
    print(f"  agent_model={args.agent_model}, conflict_resolution={args.conflict_resolution}")
    print()

    # Output setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_ds = dataset_config["sub_dataset"]
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

            # Retrieve via Engram
            query_start = time.time()
            evidence, trace = foundry.retrieve(query)

            # Generate concise answer from evidence
            answer_text = generate_answer(query, evidence or "", args.agent_model)
            query_time = time.time() - query_start

            # Build output dict matching metrics_summarization expectations
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
