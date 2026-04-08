"""Evaluate graph-direct answering on LME context-0 (60 questions).

Pipeline:
  1. Load property triples → EntityGraph
  2. For each question: extract (entity, predicate) via LLM → answer_from_graph()
  3. Score with SubEM against ground truth

Usage:
    python3 -u -m mabench.eval_graph_direct \
        --triples output/property_triples_cs512.json \
        --fixture output/disc_fixture/lme_ctx0_full.json \
        --base_url http://127.0.0.1:1234/v1 \
        --model liquid/lfm2.5-1.2b
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import defaultdict

from mabench.entity_graph import EntityGraph

# ---------------------------------------------------------------------------
# Question → (entity, predicate) extraction prompt
# Validated at 89% accuracy with lfm2.5-1.2b (~130ms/question)
# ---------------------------------------------------------------------------

QUESTION_EXTRACT_SYSTEM = """Extract the entity and property being asked about. Return JSON: {"entity": "...", "property": "..."}

Examples:
"How many SkyMiles do I have?" → {"entity": "user", "property": "skymiles_balance"}
"What is Luna's age?" → {"entity": "Luna", "property": "age"}
"Where does my brother Jake work?" → {"entity": "Jake", "property": "employer"}
JSON:"""


# ---------------------------------------------------------------------------
# SubEM scoring (reproduces MABench logic)
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
# LLM call for question extraction
# ---------------------------------------------------------------------------

def extract_entity_predicate(
    question: str, base_url: str, model: str
) -> tuple[str, str]:
    """Extract (entity, predicate) from a question via LLM."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="unused")

    # Strip date prefix from question
    q_text = question
    m = re.match(r"Current Date:.*?\n+\s*Now Answer the Question:\s*", q_text, re.S)
    if m:
        q_text = q_text[m.end():]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QUESTION_EXTRACT_SYSTEM},
            {"role": "user", "content": q_text},
        ],
        temperature=0.0,
        max_tokens=128,
    )
    raw = resp.choices[0].message.content or ""

    # Parse JSON
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return "", ""
        else:
            return "", ""

    entity = data.get("entity", "")
    predicate = data.get("property", data.get("predicate", ""))

    # Normalize self-references to "user"
    if entity.lower() in (
        "i", "me", "my", "the speaker", "the user", "you", "speaker",
    ):
        entity = "user"

    return entity, predicate


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Graph-direct answering eval")
    parser.add_argument("--triples", required=True, help="Property triples JSON")
    parser.add_argument("--fixture", required=True,
                        help="LME fixture JSON (turns + questions)")
    parser.add_argument("--base_url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="liquid/lfm2.5-1.2b",
                        help="Model for question entity/predicate extraction")
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # 1. Load property triples and build EntityGraph
    print(f"Loading triples from {args.triples}...")
    with open(args.triples) as f:
        triples_data = json.load(f)

    graph = EntityGraph()
    total_triples = 0
    for turn_idx, (hash_key, entry) in enumerate(triples_data.items()):
        for s, p, v in entry.get("triples", []):
            graph.ingest_triple(s, p, v, turn_idx=turn_idx)
            total_triples += 1

    print(f"  {total_triples} triples → {len(graph.entities)} entities")

    # Show top entities by fact count
    entity_facts = sorted(
        [(c, sum(len(fl) for fl in n.facts.values())) for c, n in graph.entities.items()],
        key=lambda x: -x[1]
    )[:10]
    print("  Top entities:")
    for name, count in entity_facts:
        print(f"    {name}: {count} facts")

    # 2. Load fixture
    print(f"\nLoading fixture from {args.fixture}...")
    with open(args.fixture) as f:
        fixture = json.load(f)
    questions = fixture["questions"]
    print(f"  {len(questions)} questions")

    # 3. Eval loop
    print(f"\nRunning eval (model={args.model}, max_hops={args.max_hops})...\n")
    results = []
    correct = 0
    answered = 0
    type_stats = defaultdict(lambda: {"correct": 0, "answered": 0, "total": 0})

    for i, q in enumerate(questions):
        question = q["question"]
        gt_answers = q["answer"] if isinstance(q["answer"], list) else [q["answer"]]
        qtype = q.get("question_type", "unknown")

        # Extract entity + predicate from question
        t0 = time.time()
        entity, predicate = extract_entity_predicate(question, args.base_url, args.model)
        t_extract = time.time() - t0

        # Query graph
        t1 = time.time()
        answer = graph.answer_from_graph(entity, predicate, question, max_hops=args.max_hops)
        t_graph = time.time() - t1

        # Score
        is_correct = subem_match(answer, gt_answers) if answer else False
        if answer:
            answered += 1
        if is_correct:
            correct += 1

        type_stats[qtype]["total"] += 1
        if answer:
            type_stats[qtype]["answered"] += 1
        if is_correct:
            type_stats[qtype]["correct"] += 1

        # Strip date prefix for display
        q_short = question
        m = re.match(r"Current Date:.*?\n+\s*Now Answer the Question:\s*", q_short, re.S)
        if m:
            q_short = q_short[m.end():]
        q_short = q_short[:80]

        mark = "✓" if is_correct else ("✗" if answer else "—")
        print(
            f"  Q{i:>2d} [{mark}] ({entity}, {predicate}) → "
            f"{str(answer)[:50] if answer else 'None':<50s} "
            f"[{t_extract*1000:.0f}+{t_graph*1000:.0f}ms] "
            f"gt={str(gt_answers[0])[:40]}"
        )
        if args.verbose:
            print(f"       q: {q_short}")

        results.append({
            "index": i,
            "question_type": qtype,
            "entity": entity,
            "predicate": predicate,
            "answer": answer,
            "ground_truth": gt_answers,
            "correct": is_correct,
            "answered": answer is not None,
            "extract_ms": t_extract * 1000,
            "graph_ms": t_graph * 1000,
        })

    # 4. Summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {correct}/{len(questions)} SubEM = {correct/len(questions)*100:.1f}%")
    print(f"Answered: {answered}/{len(questions)} ({answered/len(questions)*100:.1f}%)")
    print(f"Precision: {correct/answered*100:.1f}%" if answered > 0 else "Precision: N/A")
    print()

    print(f"{'Question Type':<30s} {'Total':>5s} {'Ans':>5s} {'Corr':>5s} {'SubEM':>7s}")
    print("-" * 55)
    for qtype in sorted(type_stats):
        s = type_stats[qtype]
        pct = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {qtype:<28s} {s['total']:>5d} {s['answered']:>5d} {s['correct']:>5d} {pct:>6.1f}%")

    # Timing
    avg_extract = sum(r["extract_ms"] for r in results) / len(results)
    avg_graph = sum(r["graph_ms"] for r in results) / len(results)
    print(f"\nAvg latency: extract={avg_extract:.0f}ms, graph={avg_graph:.1f}ms")

    # Save detailed results
    out_path = args.triples.replace(".json", "_eval_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "triples": args.triples,
                "model": args.model,
                "max_hops": args.max_hops,
            },
            "summary": {
                "total": len(questions),
                "answered": answered,
                "correct": correct,
                "subem_pct": correct / len(questions) * 100,
                "precision_pct": correct / answered * 100 if answered > 0 else 0,
            },
            "by_type": {
                qtype: dict(s) for qtype, s in type_stats.items()
            },
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
