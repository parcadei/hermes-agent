"""Query-time entity retrieval ablation.

For each LME question, extract entities via Qwen ontology prompt,
then search the faceted entity store with multiple strategies.
Measures hit@k: does a gold-answer chunk appear in the top-k results?

Strategies:
  1. exact   — case-insensitive entity name match
  2. bm25    — BM25 over chunk entity representations
  3. cosine  — embedding similarity (sentence-transformers)
  4. hybrid  — BM25 + cosine RRF fusion
  5. rerank  — hybrid + cross-encoder reranking

Usage:
  python -m mabench.ablate_entity_retrieval \
    --faceted_cache output/faceted_cache_qwen25_cs512.json \
    --multipass_cache output/multipass_cache_cs512.json \
    --ground_truth output/lme_300q_ground_truth_audit.json \
    --llm_url http://localhost:8000/v1 \
    --llm_model Qwen/Qwen2.5-7B-Instruct \
    [--top_k 10] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger(__name__)


# ── Entity index ──────────────────────────────────────────────────

def build_entity_index(
    faceted_cache: dict, multipass_cache: dict,
) -> tuple[list[str], list[str], list[set[str]], dict[str, str]]:
    """Build per-chunk entity representations.

    Returns:
        chunk_hashes: ordered list of chunk hashes
        chunk_docs:   entity "document" per chunk (names + facet values, space-joined)
        chunk_name_sets: set of lowercased entity names per chunk
        chunk_texts:  hash -> full chunk text (for gold matching)
    """
    chunk_hashes = []
    chunk_docs = []
    chunk_name_sets = []
    chunk_texts = {}

    for ch, data in faceted_cache.items():
        entities = data.get("entities", [])
        if not entities:
            continue
        names = []
        doc_parts = []
        for e in entities:
            name = e.get("name", "").strip()
            if not name:
                continue
            names.append(name.lower())
            doc_parts.append(name)
            for v in e.get("facets", {}).values():
                if v and v.strip():
                    doc_parts.append(v.strip())

        if not names:
            continue

        chunk_hashes.append(ch)
        chunk_docs.append(" ".join(doc_parts))
        chunk_name_sets.append(set(names))

        if ch in multipass_cache:
            chunk_texts[ch] = multipass_cache[ch].get("text", "")

    return chunk_hashes, chunk_docs, chunk_name_sets, chunk_texts


# ── Gold chunk identification ─────────────────────────────────────

def find_gold_chunks(
    question: dict, chunk_hashes: list[str], chunk_texts: dict[str, str],
) -> set[str]:
    """Find chunk hashes that contain gold answer evidence.

    Matches answer_msgs snippets against chunk texts via substring.
    """
    gold = set()
    for msg in question.get("answer_msgs", []):
        snippet = msg.get("snippet", "")
        if not snippet or len(snippet) < 20:
            continue
        # Use first 80 chars of snippet for matching (avoid truncation artifacts)
        needle = snippet[:80].strip()
        for ch in chunk_hashes:
            text = chunk_texts.get(ch, "")
            if needle in text:
                gold.add(ch)
                break
    return gold


# ── Strategy 1: Exact match ──────────────────────────────────────

def search_exact(
    query_names: list[str], chunk_hashes: list[str],
    chunk_name_sets: list[set[str]], top_k: int,
) -> list[str]:
    """Score chunks by number of exact entity name matches."""
    scores = []
    q_set = {n.lower() for n in query_names}
    for i, names in enumerate(chunk_name_sets):
        overlap = len(q_set & names)
        if overlap > 0:
            scores.append((chunk_hashes[i], overlap))
    scores.sort(key=lambda x: -x[1])
    return [h for h, _ in scores[:top_k]]


# ── Strategy 2: BM25 ─────────────────────────────────────────────

class SimpleBM25:
    """Minimal BM25 implementation over string documents."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.avgdl = 0.0
        self.doc_lens: list[int] = []
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = defaultdict(int)

        for doc in docs:
            tokens = doc.lower().split()
            self.doc_lens.append(len(tokens))
            self.avgdl += len(tokens)
            tf: dict[str, int] = defaultdict(int)
            seen: set[str] = set()
            for t in tokens:
                tf[t] += 1
                if t not in seen:
                    self.df[t] += 1
                    seen.add(t)
            self.tf.append(dict(tf))
        self.avgdl /= max(self.n, 1)

    def query(self, q: str, top_k: int = 10) -> list[tuple[int, float]]:
        tokens = q.lower().split()
        scores = np.zeros(self.n)
        for t in tokens:
            if t not in self.df:
                continue
            idf = math.log((self.n - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
            for i in range(self.n):
                tf_val = self.tf[i].get(t, 0)
                if tf_val == 0:
                    continue
                dl = self.doc_lens[i]
                num = tf_val * (self.k1 + 1)
                den = tf_val + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * num / den
        top_idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


def search_bm25(
    query_doc: str, bm25: SimpleBM25, chunk_hashes: list[str], top_k: int,
) -> list[str]:
    hits = bm25.query(query_doc, top_k=top_k)
    return [chunk_hashes[i] for i, _ in hits]


# ── Strategy 3: Cosine ────────────────────────────────────────────

def build_embeddings(docs: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """Embed documents using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(docs, show_progress_bar=True, batch_size=128,
                        normalize_embeddings=True)
    return model, np.array(embs)


def search_cosine(
    query_doc: str, model, doc_embs: np.ndarray,
    chunk_hashes: list[str], top_k: int,
) -> list[str]:
    q_emb = model.encode([query_doc], normalize_embeddings=True)
    sims = (doc_embs @ q_emb.T).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [chunk_hashes[int(i)] for i in top_idx if sims[i] > 0]


# ── Strategy 4: Hybrid (BM25 + cosine RRF) ───────────────────────

def search_hybrid(
    query_doc: str, bm25: SimpleBM25, model, doc_embs: np.ndarray,
    chunk_hashes: list[str], top_k: int, rrf_k: int = 60,
) -> list[str]:
    bm25_hits = bm25.query(query_doc, top_k=top_k * 3)
    q_emb = model.encode([query_doc], normalize_embeddings=True)
    sims = (doc_embs @ q_emb.T).flatten()
    cos_top = np.argsort(-sims)[:top_k * 3]

    rrf: dict[int, float] = defaultdict(float)
    for rank, (idx, _) in enumerate(bm25_hits):
        rrf[idx] += 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(cos_top):
        if sims[idx] > 0:
            rrf[int(idx)] += 1.0 / (rrf_k + rank + 1)

    sorted_idx = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
    return [chunk_hashes[i] for i, _ in sorted_idx]


# ── Strategy 5: Hybrid + reranker ─────────────────────────────────

def search_rerank(
    query_doc: str, candidates: list[str], candidate_docs: list[str],
    top_k: int,
) -> list[str]:
    """Rerank candidate chunks using a local cross-encoder."""
    if not candidates:
        return []
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query_doc, doc) for doc in candidate_docs]
        scores = ce.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [h for h, _ in ranked[:top_k]]
    except ImportError:
        logger.warning("CrossEncoder not available, skipping rerank")
        return candidates[:top_k]


# ── Question entity extraction ────────────────────────────────────

def extract_question_entities(
    question_text: str, llm_url: str, llm_model: str, timeout: float = 30.0,
) -> list[dict]:
    """Extract entities from a question using the same ontology prompt."""
    import requests
    from mabench.ontology import EXTRACTION_SYSTEM_PROMPT

    resp = requests.post(
        f"{llm_url.rstrip('/')}/chat/completions",
        json={
            "model": llm_model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT
                 + "\n\nOutput ONLY valid JSON. No markdown, no extra text."},
                {"role": "user", "content": question_text},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return []
    raw = json.loads(m.group())
    return raw.get("entities", [])


# ── Main ablation loop ────────────────────────────────────────────

def run_ablation(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load data
    logger.info("Loading caches...")
    with open(args.faceted_cache) as f:
        faceted = json.load(f)
    with open(args.multipass_cache) as f:
        multipass = json.load(f)
    with open(args.ground_truth) as f:
        gt_questions = json.load(f)

    logger.info("Building entity index...")
    chunk_hashes, chunk_docs, chunk_name_sets, chunk_texts = build_entity_index(
        faceted, multipass,
    )
    logger.info("  %d chunks with entities", len(chunk_hashes))

    # Build BM25 index
    logger.info("Building BM25 index over entity docs...")
    bm25 = SimpleBM25(chunk_docs)

    # Build embeddings
    logger.info("Building embeddings...")
    emb_model, doc_embs = build_embeddings(chunk_docs)

    # Build hash->index lookup for reranker
    hash_to_idx = {h: i for i, h in enumerate(chunk_hashes)}

    # Extract question entities via LLM
    logger.info("Extracting entities from %d questions via %s...",
                len(gt_questions), args.llm_model)

    question_entities: list[list[dict]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                extract_question_entities,
                q["question"], args.llm_url, args.llm_model,
            ): i
            for i, q in enumerate(gt_questions)
        }
        results_map: dict[int, list[dict]] = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results_map[idx] = future.result()
            except Exception as exc:
                logger.warning("  Error extracting Q%d: %s", idx, exc)
                results_map[idx] = []

    question_entities = [results_map[i] for i in range(len(gt_questions))]

    # Run ablation
    top_k = args.top_k
    strategies = ["exact", "bm25", "cosine", "hybrid", "rerank"]
    hits: dict[str, int] = {s: 0 for s in strategies}
    total = 0
    details: list[dict] = []

    for qi, q in enumerate(gt_questions):
        q_ents = question_entities[qi]
        if not q_ents:
            logger.info("  Q%d: no entities extracted, skipping", qi)
            continue

        # Build query representations
        q_names = [e.get("name", "") for e in q_ents if e.get("name")]
        q_doc_parts = []
        for e in q_ents:
            if e.get("name"):
                q_doc_parts.append(e["name"])
            for v in e.get("facets", {}).values():
                if v and v.strip():
                    q_doc_parts.append(v.strip())
        q_doc = " ".join(q_doc_parts)

        if not q_names:
            continue

        # Find gold chunks
        gold_chunks = find_gold_chunks(q, chunk_hashes, chunk_texts)

        detail = {
            "qi": qi,
            "question": q["question"][:120],
            "qtype": q.get("qtype", ""),
            "query_entities": [e.get("name") for e in q_ents],
            "gold_chunks": len(gold_chunks),
            "hits": {},
        }

        if not gold_chunks:
            detail["note"] = "no gold chunks found via snippet matching"
            details.append(detail)
            continue

        total += 1

        # Strategy 1: exact
        exact_results = search_exact(q_names, chunk_hashes, chunk_name_sets, top_k)
        hit_exact = bool(set(exact_results) & gold_chunks)
        hits["exact"] += int(hit_exact)
        detail["hits"]["exact"] = hit_exact

        # Strategy 2: bm25
        bm25_results = search_bm25(q_doc, bm25, chunk_hashes, top_k)
        hit_bm25 = bool(set(bm25_results) & gold_chunks)
        hits["bm25"] += int(hit_bm25)
        detail["hits"]["bm25"] = hit_bm25

        # Strategy 3: cosine
        cos_results = search_cosine(q_doc, emb_model, doc_embs, chunk_hashes, top_k)
        hit_cos = bool(set(cos_results) & gold_chunks)
        hits["cosine"] += int(hit_cos)
        detail["hits"]["cosine"] = hit_cos

        # Strategy 4: hybrid
        hyb_results = search_hybrid(q_doc, bm25, emb_model, doc_embs, chunk_hashes, top_k)
        hit_hyb = bool(set(hyb_results) & gold_chunks)
        hits["hybrid"] += int(hit_hyb)
        detail["hits"]["hybrid"] = hit_hyb

        # Strategy 5: rerank (rerank hybrid candidates)
        hyb_wide = search_hybrid(
            q_doc, bm25, emb_model, doc_embs, chunk_hashes, top_k * 5,
        )
        hyb_wide_docs = [chunk_docs[hash_to_idx[h]] for h in hyb_wide]
        rerank_results = search_rerank(q_doc, hyb_wide, hyb_wide_docs, top_k)
        hit_rerank = bool(set(rerank_results) & gold_chunks)
        hits["rerank"] += int(hit_rerank)
        detail["hits"]["rerank"] = hit_rerank

        details.append(detail)

        if (qi + 1) % 20 == 0:
            logger.info("  Processed %d/%d questions (%d with gold chunks)",
                        qi + 1, len(gt_questions), total)

    # Report
    logger.info("\n" + "=" * 60)
    logger.info("QUERY-TIME ENTITY RETRIEVAL ABLATION")
    logger.info("=" * 60)
    logger.info("Questions: %d total, %d with gold chunks", len(gt_questions), total)
    logger.info("Top-k: %d", top_k)
    logger.info("")
    logger.info("%-12s  hit@%d  %%", "Strategy", top_k)
    logger.info("-" * 30)
    for s in strategies:
        pct = 100 * hits[s] / total if total > 0 else 0
        logger.info("%-12s  %3d    %.1f%%", s, hits[s], pct)

    results = {
        "top_k": top_k,
        "total_questions": len(gt_questions),
        "questions_with_gold": total,
        "hits": hits,
        "percentages": {s: round(100 * hits[s] / max(total, 1), 1) for s in strategies},
        "details": details,
    }

    out_path = args.output or "output/entity_retrieval_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nDetails saved to %s", out_path)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faceted_cache", required=True)
    parser.add_argument("--multipass_cache", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--llm_url", required=True,
                        help="vLLM endpoint for question entity extraction")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
