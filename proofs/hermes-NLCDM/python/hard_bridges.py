"""Generate genuinely orthogonal cross-domain bridge pairs and probe retrieval.

Goal: create fact pairs where cos(A, B) < 0.3 but they're biographically
linked through the user's life. Then test whether cosine retrieval can
find both from a bridging question.

Uses OpenRouter LLM to generate realistic, register-pure content.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_NLCDM_PYTHON = Path(__file__).resolve().parent
sys.path.insert(0, str(_NLCDM_PYTHON))

from coupled_engine import CoupledEngine
from dream_ops import DreamParams, dream_cycle_xb
from mabench.hermes_agent import EmbeddingRelevanceScorer


def generate_hard_bridges(scorer: EmbeddingRelevanceScorer) -> list[dict]:
    """Generate bridge pairs, verify cosine < 0.3, keep only hard ones."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    prompt = """Generate 15 pairs of personal memory facts for a knowledge management system.

CRITICAL RULES:
- Each pair has fact_a and fact_b that are from COMPLETELY DIFFERENT life domains
- The facts must share ZERO overlapping vocabulary or semantic content
- They are linked ONLY through the person's biography (same person, different life areas)
- Each fact should be 1-2 sentences, written naturally as if journaling
- Include a bridging question that requires knowing BOTH facts to answer

DOMAIN PAIRS TO USE (pick different combinations):
- Pure technical code (algorithms, bugs, architecture) ↔ Physical fitness (gym, running, sports)
- Academic math/physics ↔ Cooking/recipes
- Legal/contracts ↔ Music/instruments
- Gardening/plants ↔ Financial investing
- Home renovation ↔ Foreign language learning
- Photography ↔ Database administration
- Woodworking ↔ Astronomy
- Knitting/textiles ↔ Cryptocurrency
- Beekeeping ↔ Game development
- Pottery/ceramics ↔ Networking/sysadmin
- Birdwatching ↔ Tax preparation
- Scuba diving ↔ Machine learning research
- Calligraphy ↔ Car mechanics
- Origami ↔ Real estate
- Fermentation/brewing ↔ Circuit design

FORMAT (JSON array):
[
  {
    "domain_a": "technical code",
    "domain_b": "physical fitness",
    "fact_a": "Finally fixed the O(n³) bottleneck in the graph traversal by switching to an adjacency list with lazy evaluation of edge weights",
    "fact_b": "Hit a new deadlift PR of 315 pounds today. The progressive overload program is really paying off after twelve weeks",
    "keyword_a": "adjacency list",
    "keyword_b": "deadlift",
    "bridge_question": "What personal achievements am I most proud of this month?",
    "bridge_keywords": ["adjacency", "deadlift"]
  }
]

The bridge question should be answerable ONLY by someone who knows both facts.
Make facts sound natural — like real journal entries, not contrived test data.
Each fact must use vocabulary EXCLUSIVE to its domain. No shared words like "progress", "working on", "challenge" that create semantic bridges."""

    resp = client.chat.completions.create(
        model="minimax/minimax-m2.5",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=4000,
    )

    raw = resp.choices[0].message.content
    # Extract JSON from response
    start = raw.find("[")
    end = raw.rfind("]") + 1
    bridges = json.loads(raw[start:end])

    # Verify cosine distances
    hard_bridges = []
    print("Generated bridges — checking cosine distances:\n")
    for b in bridges:
        emb_a = scorer.embed(b["fact_a"])
        emb_b = scorer.embed(b["fact_b"])
        sim = float(np.dot(emb_a, emb_b))
        status = "HARD" if sim < 0.3 else ("MEDIUM" if sim < 0.5 else "EASY")
        print(f"  [{status}] cos={sim:.4f}  {b['domain_a']} ↔ {b['domain_b']}")
        print(f"    A: {b['fact_a'][:70]}")
        print(f"    B: {b['fact_b'][:70]}")
        b["cosine"] = sim
        b["difficulty"] = status
        hard_bridges.append(b)

    hard_count = sum(1 for b in hard_bridges if b["cosine"] < 0.3)
    medium_count = sum(1 for b in hard_bridges if 0.3 <= b["cosine"] < 0.5)
    easy_count = sum(1 for b in hard_bridges if b["cosine"] >= 0.5)
    print(f"\n  HARD (cos<0.3): {hard_count}")
    print(f"  MEDIUM (0.3-0.5): {medium_count}")
    print(f"  EASY (cos≥0.5): {easy_count}")

    return hard_bridges


def run_hard_probe(bridges: list[dict], scorer: EmbeddingRelevanceScorer):
    """Run retrieval probe on hard bridges."""

    # Also load the standard multidomain dataset as background noise
    from multidomain_eval import generate_multidomain_dataset
    ds = generate_multidomain_dataset(seed=42)
    all_sessions = sorted(ds.sessions, key=lambda s: s.day)

    engine = CoupledEngine(
        dim=1024,
        contradiction_aware=True,
        contradiction_threshold=0.95,
        recency_alpha=0.1,
        hebbian_epsilon=0.05,
    )

    # Store all standard facts first (background noise)
    # Group by session so flush_session() creates temporal bindings within each session
    stored_texts: list[str] = []
    for session in all_sessions:
        for fact in session.facts:
            emb = scorer.embed(fact)
            engine.store(text=fact, embedding=emb, recency=float(len(stored_texts)))
            stored_texts.append(fact)
        engine.flush_session()

    # Now store the hard bridge facts
    # Each bridge pair is stored together in one session, then flushed
    # so that flush_session() creates co-occurrence links between fact_a and fact_b
    bridge_indices: list[tuple[int, int]] = []  # (idx_a, idx_b) in engine
    for b in bridges:
        emb_a = scorer.embed(b["fact_a"])
        idx_a = engine.store(
            text=b["fact_a"], embedding=emb_a,
            recency=float(len(stored_texts)),
        )
        stored_texts.append(b["fact_a"])

        emb_b = scorer.embed(b["fact_b"])
        idx_b = engine.store(
            text=b["fact_b"], embedding=emb_b,
            recency=float(len(stored_texts)),
        )
        stored_texts.append(b["fact_b"])
        bridge_indices.append((idx_a, idx_b))
        engine.flush_session()  # binds fact_a <-> fact_b via temporal Hebbian

    print(f"\nTotal memories: {engine.n_memories} "
          f"({len(stored_texts) - 2*len(bridges)} background + "
          f"{2*len(bridges)} bridge facts)")

    # ================================================================
    # MULTI-STRATEGY RETRIEVAL PROBE on hard bridges
    # ================================================================
    print("\n" + "=" * 70)
    print("HARD BRIDGE RETRIEVAL PROBE (multi-strategy)")
    print("=" * 70)

    strategies = {
        "cosine": lambda q: engine.query(embedding=q, top_k=50),
        "twohop": lambda q: engine.query_twohop(
            embedding=q, top_k=50, first_hop_k=10,
        ),
        "cooc_boost": lambda q: engine.query_cooc_boost(
            embedding=q, top_k=50, cooc_weight=0.3,
        ),
        "ppr": lambda q: engine.query_ppr(
            embedding=q, top_k=50, damping=0.85, ppr_steps=10, ppr_weight=0.5,
        ),
        "associative": lambda q: engine.query_associative(q, top_k=50, sparse=True),
        "hybrid": lambda q: engine.query_hybrid(q, top_k=50),
    }

    strategy_hits: dict[str, int] = {name: 0 for name in strategies}
    results_by_difficulty = {"HARD": [], "MEDIUM": [], "EASY": []}

    for b, (idx_a, idx_b) in zip(bridges, bridge_indices):
        q_text = b["bridge_question"]
        q_emb = scorer.embed(q_text)
        kw_a = b["keyword_a"].lower()
        kw_b = b["keyword_b"].lower()

        per_strategy: dict[str, dict] = {}
        for strat_name, strat_fn in strategies.items():
            results = strat_fn(q_emb)

            # Find keyword ranks in this strategy's results
            rank_a = None
            rank_b = None
            for rank, r in enumerate(results):
                if rank_a is None and kw_a in r["text"].lower():
                    rank_a = rank + 1
                if rank_b is None and kw_b in r["text"].lower():
                    rank_b = rank + 1

            both_in_10 = (rank_a is not None and rank_a <= 10 and
                          rank_b is not None and rank_b <= 10)
            if both_in_10:
                strategy_hits[strat_name] += 1

            per_strategy[strat_name] = {
                "rank_a": rank_a,
                "rank_b": rank_b,
                "both_in_10": both_in_10,
                "top3_scores": [round(r["score"], 4) for r in results[:3]],
            }

        # Record result using cosine as primary for difficulty breakdown
        cosine_info = per_strategy["cosine"]
        result = {
            "difficulty": b["difficulty"],
            "cosine": b["cosine"],
            "domains": f"{b['domain_a']}↔{b['domain_b']}",
            "question": q_text,
            "rank_a": cosine_info["rank_a"],
            "rank_b": cosine_info["rank_b"],
            "both_in_10": cosine_info["both_in_10"],
            "top3_scores": cosine_info["top3_scores"],
            "per_strategy": per_strategy,
        }
        results_by_difficulty[b["difficulty"]].append(result)

        # Print per-bridge detail
        cosine_status = "OK" if cosine_info["both_in_10"] else "MISS"
        print(f"\n  [{cosine_status}] cos={b['cosine']:.3f} "
              f"{b['domain_a']}↔{b['domain_b']}")
        print(f"    Q: {q_text[:70]}")
        for sn, si in per_strategy.items():
            tag = "+" if si["both_in_10"] else "-"
            print(f"    [{tag}] {sn:13s}  "
                  f"kw_a→{si['rank_a'] or '>50':>4}  "
                  f"kw_b→{si['rank_b'] or '>50':>4}")

    # ================================================================
    # STRATEGY COMPARISON TABLE
    # ================================================================
    n_bridges = len(bridges)
    print("\n" + "-" * 40)
    print(f"{'Strategy':<18s} Both-in-10")
    print("-" * 40)
    for sn in strategies:
        h = strategy_hits[sn]
        pct = h / n_bridges * 100 if n_bridges else 0
        print(f"{sn:<18s} {h}/{n_bridges} ({pct:.0f}%)")
    print("-" * 40)

    # ================================================================
    # DIAGNOSTIC: WHY DO 12/15 FAIL?
    # ================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: cos(question, fact) + global rank for each bridge")
    print("=" * 70)

    # Compute global cosine scores for all memories against each bridge question
    all_embeddings = engine._embeddings_matrix()
    all_norms = np.linalg.norm(all_embeddings, axis=1)

    failure_modes = {"reach": 0, "bonus": 0, "both_far": 0, "ok": 0}
    for b, (idx_a, idx_b) in zip(bridges, bridge_indices):
        q_emb = scorer.embed(b["bridge_question"])
        q_norm = np.linalg.norm(q_emb)
        global_scores = all_embeddings @ q_emb / (all_norms * q_norm + 1e-12)

        cos_qa = float(global_scores[idx_a])
        cos_qb = float(global_scores[idx_b])

        # Global rank (1-indexed) of each fact among ALL memories
        sorted_indices = np.argsort(global_scores)[::-1]
        rank_a = int(np.where(sorted_indices == idx_a)[0][0]) + 1
        rank_b = int(np.where(sorted_indices == idx_b)[0][0]) + 1

        # Is fact in first_hop_k=10 cosine window?
        in_hop_a = rank_a <= 10
        in_hop_b = rank_b <= 10

        # Classify failure mode
        twohop_info = per_strategy_all.get(
            f"{b['domain_a']}↔{b['domain_b']}", {}
        ) if 'per_strategy_all' in dir() else {}

        if rank_a <= 10 and rank_b <= 10:
            mode = "BOTH_IN_10 (cosine alone works)"
        elif rank_a <= 10 or rank_b <= 10:
            near = "A" if rank_a <= 10 else "B"
            far = "B" if rank_a <= 10 else "A"
            far_rank = rank_b if rank_a <= 10 else rank_a
            if far_rank <= 40:
                mode = f"REACH (fact_{far} at rank {far_rank}, reachable with higher first_hop_k)"
                failure_modes["reach"] += 1
            else:
                mode = f"FAR (fact_{far} at rank {far_rank}, needs co-occurrence expansion)"
                failure_modes["bonus"] += 1
        else:
            mode = f"BOTH_FAR (ranks {rank_a}, {rank_b} — neither in top-10)"
            failure_modes["both_far"] += 1

        # Check if co-occurrence link exists
        has_cooc = idx_b in engine._co_occurrence.get(idx_a, set())

        print(f"\n  {b['domain_a']}↔{b['domain_b']} (bridge cos={b['cosine']:.3f})")
        print(f"    Q: {b['bridge_question'][:70]}")
        print(f"    cos(Q,A)={cos_qa:.4f}  rank_A={rank_a:>4d}  "
              f"cos(Q,B)={cos_qb:.4f}  rank_B={rank_b:>4d}")
        print(f"    co-occurrence link: {'YES' if has_cooc else 'NO'}")
        print(f"    → {mode}")

    print(f"\n  Failure mode summary:")
    print(f"    REACH (bump first_hop_k):      {failure_modes['reach']}")
    print(f"    FAR (needs expansion/bonus):    {failure_modes['bonus']}")
    print(f"    BOTH_FAR (question too generic):{failure_modes['both_far']}")

    # Also test: what if first_hop_k were 20, 40, 50?
    print("\n" + "-" * 40)
    print("SENSITIVITY: first_hop_k sweep")
    print("-" * 40)
    for fhk in [10, 20, 30, 40, 50]:
        hits = 0
        for b, (idx_a, idx_b) in zip(bridges, bridge_indices):
            q_emb = scorer.embed(b["bridge_question"])
            results = engine.query_twohop(
                embedding=q_emb, top_k=50, first_hop_k=fhk,
            )
            kw_a = b["keyword_a"].lower()
            kw_b = b["keyword_b"].lower()
            ra = rb = None
            for rank, r in enumerate(results):
                if ra is None and kw_a in r["text"].lower():
                    ra = rank + 1
                if rb is None and kw_b in r["text"].lower():
                    rb = rank + 1
            if ra and ra <= 10 and rb and rb <= 10:
                hits += 1
        print(f"  first_hop_k={fhk:>3d}  → {hits}/{n_bridges} "
              f"({hits/n_bridges*100:.0f}%) both-in-10")

    # And co_occurrence_bonus sweep
    print("\n" + "-" * 40)
    print("SENSITIVITY: co_occurrence_bonus sweep (first_hop_k=10)")
    print("-" * 40)
    for bonus in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
        hits = 0
        for b, (idx_a, idx_b) in zip(bridges, bridge_indices):
            q_emb = scorer.embed(b["bridge_question"])
            results = engine.query_twohop(
                embedding=q_emb, top_k=50, first_hop_k=10,
                co_occurrence_bonus=bonus,
            )
            kw_a = b["keyword_a"].lower()
            kw_b = b["keyword_b"].lower()
            ra = rb = None
            for rank, r in enumerate(results):
                if ra is None and kw_a in r["text"].lower():
                    ra = rank + 1
                if rb is None and kw_b in r["text"].lower():
                    rb = rank + 1
            if ra and ra <= 10 and rb and rb <= 10:
                hits += 1
        print(f"  bonus={bonus:<4.1f}  → {hits}/{n_bridges} "
              f"({hits/n_bridges*100:.0f}%) both-in-10")

    # ================================================================
    # SUMMARY BY DIFFICULTY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY BY DIFFICULTY")
    print("=" * 70)

    for diff in ["HARD", "MEDIUM", "EASY"]:
        results = results_by_difficulty[diff]
        if not results:
            continue
        n = len(results)
        hits = sum(1 for r in results if r["both_in_10"])
        avg_cos = np.mean([r["cosine"] for r in results])

        # Average rank of keyword_a and keyword_b
        ranks_a = [r["rank_a"] for r in results if r["rank_a"] is not None]
        ranks_b = [r["rank_b"] for r in results if r["rank_b"] is not None]
        miss_a = sum(1 for r in results if r["rank_a"] is None)
        miss_b = sum(1 for r in results if r["rank_b"] is None)

        print(f"\n  {diff} (n={n}, avg cos={avg_cos:.3f}):")
        print(f"    Both in top-10: {hits}/{n} ({hits/n*100:.0f}%)")
        if ranks_a:
            print(f"    keyword_a: mean rank {np.mean(ranks_a):.1f}, "
                  f"missed top-50: {miss_a}")
        if ranks_b:
            print(f"    keyword_b: mean rank {np.mean(ranks_b):.1f}, "
                  f"missed top-50: {miss_b}")

    # ================================================================
    # DREAM PROBE on hard bridges
    # ================================================================
    print("\n" + "=" * 70)
    print("DREAM ASSOCIATION DISCOVERY ON HARD BRIDGES")
    print("=" * 70)

    X = engine._embeddings_matrix()
    N = X.shape[0]
    report = dream_cycle_xb(
        X, beta=engine.beta,
        tagged_indices=list(range(N)),
        seed=42,
    )

    print(f"  Patterns: {N} → {report.patterns.shape[0]} post-dream")
    print(f"  Associations found: {len(report.associations)}")

    # Check if any associations link bridge facts
    if report.associations:
        bridge_fact_indices = set()
        for idx_a, idx_b in bridge_indices:
            bridge_fact_indices.add(idx_a)
            bridge_fact_indices.add(idx_b)

        bridge_assocs = [
            (i, j, s) for i, j, s in report.associations
            if i in bridge_fact_indices or j in bridge_fact_indices
        ]
        print(f"  Associations involving bridge facts: {len(bridge_assocs)}")
        for i, j, s in bridge_assocs[:5]:
            print(f"    [{i}↔{j}] sim={s:.4f}")
    else:
        print("  Zero associations. REM-explore is inert on this data too.")

    # ================================================================
    # THE KEY QUESTION
    # ================================================================
    print("\n" + "=" * 70)
    print("VERDICT: WHERE DOES COSINE BREAK?")
    print("=" * 70)

    all_results = []
    for diff in ["HARD", "MEDIUM", "EASY"]:
        all_results.extend(results_by_difficulty[diff])

    if all_results:
        # Correlation between cosine and retrieval success
        cosines = [r["cosine"] for r in all_results]
        successes = [1.0 if r["both_in_10"] else 0.0 for r in all_results]

        # Find the threshold where retrieval starts failing
        sorted_by_cos = sorted(zip(cosines, successes))
        print(f"\n  Retrieval success vs bridge cosine:")
        for cos, suc in sorted_by_cos:
            bar = "###" if suc else "..."
            print(f"    cos={cos:.3f} {bar}")

        # Summary stats
        hard_success = np.mean([s for c, s in zip(cosines, successes) if c < 0.3]) if any(c < 0.3 for c in cosines) else float('nan')
        med_success = np.mean([s for c, s in zip(cosines, successes) if 0.3 <= c < 0.5]) if any(0.3 <= c < 0.5 for c in cosines) else float('nan')
        easy_success = np.mean([s for c, s in zip(cosines, successes) if c >= 0.5]) if any(c >= 0.5 for c in cosines) else float('nan')

        print(f"\n  cos < 0.3:  {hard_success*100:.0f}% retrieval success")
        print(f"  0.3-0.5:   {med_success*100:.0f}% retrieval success")
        print(f"  cos >= 0.5: {easy_success*100:.0f}% retrieval success")

        print(f"\n  This shows the cosine threshold below which store-time")
        print(f"  dynamics (association injection, temporal binding) become")
        print(f"  necessary for cross-domain retrieval.")


if __name__ == "__main__":
    scorer = EmbeddingRelevanceScorer()
    bridges = generate_hard_bridges(scorer)

    # Save for reproducibility
    output = []
    for b in bridges:
        out = {k: v for k, v in b.items()}
        output.append(out)
    with open("output/hard_bridges.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(bridges)} bridges to output/hard_bridges.json")

    run_hard_probe(bridges, scorer)
