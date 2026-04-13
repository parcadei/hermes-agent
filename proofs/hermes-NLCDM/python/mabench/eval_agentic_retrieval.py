"""Evaluate agentic retrieval on LME context-0 questions.

Ingests conversation data, builds all indexes, then runs the agentic
retriever on each question. Compares against baseline single-pass BM25.

Usage:
    python3 -u -m mabench.eval_agentic_retrieval \
        --fixture output/disc_fixture/lme_ctx0_full.json \
        --faceted_cache output/faceted_cache_qwen25_cs512.json \
        --llm_model gpt-4o-mini \
        --max_steps 6 \
        --sample 10

    Full 60 questions:
    python3 -u -m mabench.eval_agentic_retrieval \
        --fixture output/disc_fixture/lme_ctx0_full.json \
        --faceted_cache output/faceted_cache_qwen25_cs512.json \
        --llm_model gpt-4o-mini \
        --max_steps 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import string
import sys
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SubEM scoring (same as eval_graph_direct)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def subem_match(prediction: str, ground_truths: list[str]) -> bool:
    norm_pred = normalize_answer(prediction)
    return any(normalize_answer(gt) in norm_pred for gt in ground_truths)


# ---------------------------------------------------------------------------
# Strip LME template to get raw question
# ---------------------------------------------------------------------------

_LME_QUESTION_RE = re.compile(
    r"Now Answer the Question:\s*(.+?)(?:\s*\n\s*Answer:|$)", re.DOTALL
)


def strip_lme_template(question: str) -> str:
    m = _LME_QUESTION_RE.search(question)
    return m.group(1).strip() if m else question.strip()


# ---------------------------------------------------------------------------
# Build infrastructure from fixture
# ---------------------------------------------------------------------------

def build_infrastructure(fixture: dict, faceted_cache: dict | None = None):
    """Build BM25 index, entity graph, entity BM25 from fixture data."""
    from mabench.bm25_index import BM25Index
    from mabench.entity_graph import EntityGraph
    from mabench.layered_memory import LayeredMemory
    from mabench.ontology import FacetEntity

    class SimpleEntry:
        def __init__(self, text, role="user"):
            self.text = text
            self.role = role

    # Extract all conversation turns from the fixture
    memory_store = []
    chunks_by_hash = {}

    # Fixture format: {"turns": [...], "questions": [...]}
    raw_turns = fixture.get("turns", [])
    if raw_turns:
        for turn_text in raw_turns:
            if isinstance(turn_text, str) and turn_text.strip():
                memory_store.append(SimpleEntry(turn_text))
                h = hashlib.md5(turn_text.encode()).hexdigest()
                chunks_by_hash[h] = turn_text
    else:
        # Fallback: sessions-based format
        sessions = fixture.get("sessions", [])
        for session in sessions:
            turns = session.get("turns", session.get("chunks", []))
            for turn in turns:
                text = turn if isinstance(turn, str) else turn.get("text", "")
                role = "user" if not isinstance(turn, dict) else turn.get("role", "user")
                if text:
                    memory_store.append(SimpleEntry(text, role))
                    h = hashlib.md5(text.encode()).hexdigest()
                    chunks_by_hash[h] = text

    logger.info("Built memory store: %d turns", len(memory_store))

    # Build BM25
    bm25 = BM25Index()
    for i, entry in enumerate(memory_store):
        bm25.add(i, entry.text)
    logger.info("BM25 index: %d documents", len(bm25._docs))

    # Build entity graph + entity BM25
    lm = LayeredMemory()

    if faceted_cache:
        n_entities = 0
        for chunk_hash, cache_entry in faceted_cache.items():
            entities = cache_entry.get("entities", [])
            # Find the turn index for this chunk
            chunk_text = chunks_by_hash.get(chunk_hash, "")
            turn_idx = -1
            for i, entry in enumerate(memory_store):
                if entry.text == chunk_text:
                    turn_idx = i
                    break

            for ent_data in entities:
                name = ent_data.get("name", "")
                facets = ent_data.get("facets", {})
                if not name:
                    continue
                fe = FacetEntity(
                    name=name,
                    facets=facets,
                    turn_idx=turn_idx,
                    source_text=chunk_text[:200] if chunk_text else "",
                )
                lm.ingest_entity(fe)
                n_entities += 1
        logger.info("Ingested %d entities from faceted cache", n_entities)
    else:
        logger.warning("No faceted cache — entity search will be empty")

    return memory_store, bm25, lm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eval agentic retrieval on LME")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--faceted_cache", default=None)
    parser.add_argument("--llm_model", default="gpt-4o-mini")
    parser.add_argument("--llm_base_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--sample", type=int, default=0,
                        help="Run on first N questions only (0=all)")
    parser.add_argument("--output", default=None,
                        help="Save traces to JSON file")
    args = parser.parse_args()

    # Load fixture
    with open(args.fixture) as f:
        fixture = json.load(f)

    questions = fixture.get("questions", fixture) if isinstance(fixture, dict) else fixture
    if isinstance(questions, dict):
        questions = list(questions.values())

    logger.info("Loaded %d questions", len(questions))

    # Load faceted cache
    faceted_cache = None
    if args.faceted_cache:
        with open(args.faceted_cache) as f:
            faceted_cache = json.load(f)
        logger.info("Loaded faceted cache: %d chunks", len(faceted_cache))

    # Build infrastructure
    memory_store, bm25, lm = build_infrastructure(fixture, faceted_cache)

    # Build toolkit
    from mabench.agentic_retrieval import AgenticRetriever, ToolKit

    toolkit = ToolKit(
        bm25_index=bm25,
        entity_bm25=lm._entity_bm25,
        entity_graph=lm.stated,
        memory_store=memory_store,
    )

    retriever = AgenticRetriever(
        toolkit=toolkit,
        llm_base_url=args.llm_base_url,
        llm_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        llm_model=args.llm_model,
        max_steps=args.max_steps,
    )

    # Run eval
    if args.sample > 0:
        questions = questions[:args.sample]

    results = []
    correct = 0
    total = 0

    for i, q_data in enumerate(questions):
        question = q_data.get("question", q_data.get("input", ""))
        # LME fixture uses "answer" (singular string)
        raw_gt = q_data.get("answer", q_data.get("ground_truth", q_data.get("answers", [])))
        if isinstance(raw_gt, str):
            ground_truths = [raw_gt]
        elif isinstance(raw_gt, list):
            ground_truths = raw_gt
        else:
            ground_truths = [str(raw_gt)] if raw_gt else []

        q_text = strip_lme_template(question)

        logger.info("[%d/%d] Q: %s", i + 1, len(questions), q_text[:80])

        evidence, trace = retriever.retrieve(q_text)

        # Score: does evidence contain the answer?
        match = subem_match(evidence, ground_truths) if ground_truths else False
        if match:
            correct += 1
        total += 1

        result = {
            "question": q_text,
            "ground_truth": ground_truths,
            "evidence": evidence[:500],
            "match": match,
            "trace": trace.to_dict(),
        }
        results.append(result)

        status = "HIT" if match else "MISS"
        logger.info(
            "  %s | %d steps | %.0fms | terminated: %s",
            status, len(trace.steps), trace.total_ms, trace.terminated_by,
        )

    # Summary
    subem = correct / total * 100 if total > 0 else 0
    avg_steps = sum(len(r["trace"]["steps"]) for r in results) / len(results) if results else 0
    avg_ms = sum(r["trace"]["total_ms"] for r in results) / len(results) if results else 0

    print(f"\n{'='*60}")
    print(f"Agentic Retrieval Eval — {args.llm_model}")
    print(f"{'='*60}")
    print(f"Questions: {total}")
    print(f"Correct:   {correct}/{total} ({subem:.1f}%)")
    print(f"Avg steps: {avg_steps:.1f}")
    print(f"Avg time:  {avg_ms:.0f}ms")
    print(f"Max steps: {args.max_steps}")

    # Breakdown by termination type
    term_counts = defaultdict(int)
    for r in results:
        term_counts[r["trace"]["terminated_by"]] += 1
    print(f"\nTermination: {dict(term_counts)}")

    # Save traces
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "config": {
                    "model": args.llm_model,
                    "max_steps": args.max_steps,
                    "n_questions": total,
                },
                "summary": {
                    "subem": subem,
                    "correct": correct,
                    "total": total,
                    "avg_steps": avg_steps,
                    "avg_ms": avg_ms,
                },
                "results": results,
            }, f, indent=2)
        logger.info("Saved traces to %s", args.output)


if __name__ == "__main__":
    main()
