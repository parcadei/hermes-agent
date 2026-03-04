"""Discriminative probe: isolate retrieval vs generation for cross-domain.

Answers three questions:
1. RETRIEVAL: Are both relevant seed facts in top-k for cross-domain queries?
2. ASSOCIATION: Does dream_cycle_xb discover the cross-domain bridges?
3. EMBEDDING GEOMETRY: What's the actual cosine similarity between bridge pairs?

No LLM calls. Pure embedding + retrieval + dream analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_NLCDM_PYTHON = Path(__file__).resolve().parent
sys.path.insert(0, str(_NLCDM_PYTHON))

from coupled_engine import CoupledEngine
from dream_ops import DreamParams, dream_cycle_xb
from multidomain_eval import generate_multidomain_dataset
from mabench.hermes_agent import EmbeddingRelevanceScorer


def run_probe():
    scorer = EmbeddingRelevanceScorer()

    # Generate dataset
    ds = generate_multidomain_dataset(seed=42)

    # Separate sessions by type
    cross_domain_sessions = [s for s in ds.sessions if s.session_type == "cross_domain"]
    all_sessions = sorted(ds.sessions, key=lambda s: s.day)

    # Build engine with all facts (simulates full store pipeline)
    engine = CoupledEngine(
        dim=1024,
        contradiction_aware=True,
        contradiction_threshold=0.95,
        recency_alpha=0.1,
    )

    # Store all facts in day order, track which texts go in
    stored_texts: list[str] = []
    for session in all_sessions:
        for fact in session.facts:
            emb = scorer.embed(fact)
            engine.store(text=fact, embedding=emb, recency=float(len(stored_texts)))
            stored_texts.append(fact)

    print(f"Stored {len(stored_texts)} facts total")
    print(f"Engine has {engine.n_memories} memories (after contradiction dedup)")
    print()

    # ================================================================
    # PROBE 1: Retrieval quality for cross-domain questions
    # ================================================================
    print("=" * 70)
    print("PROBE 1: RETRIEVAL — Are both bridge facts in top-k?")
    print("=" * 70)

    cross_domain_questions = [q for q in ds.questions if q.category == "cross_domain"]

    retrieval_hits = 0
    retrieval_total = 0
    per_question_results = []

    for q in cross_domain_questions:
        q_emb = scorer.embed(q.question)
        results = engine.query(embedding=q_emb, top_k=10)

        # Check if expected keywords appear in retrieved texts
        retrieved_texts = [r["text"] for r in results]
        retrieved_concat = " ".join(retrieved_texts).lower()

        keywords_found = []
        keywords_missing = []
        for kw in q.expected_keywords:
            if kw.lower() in retrieved_concat:
                keywords_found.append(kw)
            else:
                keywords_missing.append(kw)

        all_found = len(keywords_missing) == 0
        if all_found:
            retrieval_hits += 1
        retrieval_total += 1

        per_question_results.append({
            "question": q.question[:80],
            "found": keywords_found,
            "missing": keywords_missing,
            "top3_scores": [round(r["score"], 4) for r in results[:3]],
            "top3_texts": [r["text"][:60] for r in results[:3]],
        })

    print(f"\nRetrieval hit rate: {retrieval_hits}/{retrieval_total} "
          f"({retrieval_hits/retrieval_total*100:.1f}%)")
    print(f"  (hit = ALL expected keywords found in top-10 retrieved texts)")
    print()

    for r in per_question_results:
        status = "OK" if not r["missing"] else "MISS"
        print(f"  [{status}] {r['question']}")
        if r["missing"]:
            print(f"       MISSING: {r['missing']}")
            print(f"       FOUND:   {r['found']}")
            print(f"       top3 scores: {r['top3_scores']}")
            for i, t in enumerate(r["top3_texts"]):
                print(f"         #{i+1}: {t}")
        print()

    # ================================================================
    # PROBE 2: Embedding geometry — cosine between bridge pairs
    # ================================================================
    print("=" * 70)
    print("PROBE 2: EMBEDDING GEOMETRY — cosine(seed_a, seed_b) for bridges")
    print("=" * 70)

    from multidomain_eval import _BRIDGE_DEFS

    bridge_sims = []
    for bdef in _BRIDGE_DEFS:
        seeds = bdef["seed_facts"]
        if len(seeds) >= 2:
            emb_a = scorer.embed(seeds[0][1])
            emb_b = scorer.embed(seeds[1][1])
            sim = float(np.dot(emb_a, emb_b))
            bridge_sims.append((
                f"{bdef['domain_a']}→{bdef['domain_b']}",
                bdef["keyword_a"], bdef["keyword_b"], sim,
            ))
            print(f"  {bdef['domain_a']:>10}→{bdef['domain_b']:<10}: "
                  f"cos={sim:.4f}  ({bdef['keyword_a']} ↔ {bdef['keyword_b']})")

    sims_array = [s[3] for s in bridge_sims]
    print(f"\n  Mean bridge similarity: {np.mean(sims_array):.4f}")
    print(f"  Min:  {np.min(sims_array):.4f}")
    print(f"  Max:  {np.max(sims_array):.4f}")
    print(f"  Genuinely orthogonal (cos<0.3): "
          f"{sum(1 for s in sims_array if s < 0.3)}/{len(sims_array)}")

    # Also: cosine between each question and its seed facts
    print()
    print("  Question→SeedFact cosine (how well does query find the bridge):")
    for bdef in _BRIDGE_DEFS[:4]:  # first 4 for brevity
        seeds = bdef["seed_facts"]
        for q_text, expected_kws in bdef["questions"][:1]:
            q_emb = scorer.embed(q_text)
            for day, seed_text in seeds:
                seed_emb = scorer.embed(seed_text)
                sim = float(np.dot(q_emb, seed_emb))
                seed_short = seed_text[:50]
                print(f"    Q: {q_text[:50]}")
                print(f"      → {seed_short}...  cos={sim:.4f}")
            print()

    # ================================================================
    # PROBE 3: Dream associations — does dream find the bridges?
    # ================================================================
    print("=" * 70)
    print("PROBE 3: DREAM ASSOCIATIONS — does dream_cycle_xb find bridges?")
    print("=" * 70)

    X = engine._embeddings_matrix()
    N = X.shape[0]
    print(f"\n  Patterns: {N}, dim: {X.shape[1]}")

    # Run dream with default params
    report = dream_cycle_xb(
        X, beta=engine.beta,
        tagged_indices=list(range(N)),
        seed=42,
    )

    print(f"  Post-dream patterns: {report.patterns.shape[0]}")
    print(f"  Pruned: {len(report.pruned_indices)}")
    print(f"  Merged: {sum(len(v) for v in report.merge_map.values())} into "
          f"{len(report.merge_map)} centroids")
    print(f"  Associations found: {len(report.associations)}")

    if report.associations:
        print(f"\n  Top 10 associations (by similarity):")
        sorted_assocs = sorted(report.associations, key=lambda x: -x[2])
        for idx_i, idx_j, sim in sorted_assocs[:10]:
            # Map back to text
            text_i = stored_texts[idx_i] if idx_i < len(stored_texts) else "?"
            text_j = stored_texts[idx_j] if idx_j < len(stored_texts) else "?"
            print(f"    [{idx_i}↔{idx_j}] sim={sim:.4f}")
            print(f"      A: {text_i[:70]}")
            print(f"      B: {text_j[:70]}")
    else:
        print("  NO associations discovered — REM explore found nothing.")
        print("  This means cross-domain bridge discovery is not working.")

    # ================================================================
    # PROBE 4: Per-keyword retrieval rank analysis
    # ================================================================
    print()
    print("=" * 70)
    print("PROBE 4: WHERE does each keyword rank in retrieval?")
    print("=" * 70)

    miss_count = 0
    for q in cross_domain_questions[:10]:
        q_emb = scorer.embed(q.question)
        results = engine.query(embedding=q_emb, top_k=50)
        retrieved_texts = [(i, r["text"], r["score"]) for i, r in enumerate(results)]

        print(f"\n  Q: {q.question[:70]}")
        for kw in q.expected_keywords:
            found_at = None
            for rank, (_, text, score) in enumerate(retrieved_texts):
                if kw.lower() in text.lower():
                    found_at = (rank, score, text[:50])
                    break
            if found_at:
                rank, score, text = found_at
                print(f"    '{kw}' → rank {rank+1}, score={score:.4f}: {text}")
            else:
                print(f"    '{kw}' → NOT IN TOP 50")
                miss_count += 1

    print(f"\n  Keywords missing from top-50: {miss_count}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Retrieval hit rate (all keywords in top-10): "
          f"{retrieval_hits}/{retrieval_total} ({retrieval_hits/retrieval_total*100:.1f}%)")
    print(f"  Mean bridge pair cosine: {np.mean(sims_array):.4f}")
    print(f"  Dream associations found: {len(report.associations)}")
    print(f"  Keywords missing from top-50: {miss_count}")
    print()

    if retrieval_hits / retrieval_total > 0.8:
        print("  DIAGNOSIS: Retrieval is GOOD. The gap is in GENERATION.")
        print("  → Store-time dynamics won't help. Focus on prompt/LLM.")
    elif retrieval_hits / retrieval_total > 0.5:
        print("  DIAGNOSIS: Retrieval is PARTIAL. Some bridges found, some missed.")
        print("  → Store-time dynamics (association injection) could help.")
    else:
        print("  DIAGNOSIS: Retrieval is POOR. Bridge facts not in top-k.")
        print("  → Embedding geometry is the bottleneck. Need bridge embeddings.")

    if len(report.associations) == 0:
        print("  → Dream REM-explore is NOT discovering bridges.")
        print("    This confirms associations are found but never used.")


if __name__ == "__main__":
    run_probe()
