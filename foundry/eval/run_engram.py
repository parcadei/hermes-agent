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

import hashlib

from foundry.core import Foundry, FoundryConfig
from foundry.ingestion import FacetEntity
from foundry.storage import RawChunk, VectorIndex


# ---------------------------------------------------------------------------
# Turn-level BM25 rebuild
# ---------------------------------------------------------------------------

def _rebuild_bm25_with_turns(foundry: "Foundry", raw_context) -> None:
    """Replace chunk-level BM25 with turn-level BM25 for finer granularity.

    Parses the raw context (stringified Python list of conversation sessions)
    into individual conversation turns using mabench's conversation_parser.
    Each turn is ~100-200 tokens, giving much better BM25 precision than
    512-token chunks (mabench uses turn-level and gets 43% SubEM vs 8% chunk-level).

    Entity graph stays indexed by original 512-token chunks (built during
    ingestion with faceted cache). Only BM25 + chunk store are rebuilt.
    """
    import ast
    from foundry.storage import BM25Index

    # Parse raw context into structured chunks
    try:
        context_chunks = ast.literal_eval(raw_context) if isinstance(raw_context, str) else raw_context
    except (ValueError, SyntaxError):
        print("  Skipping turn-level BM25: context not a parseable Python literal")
        return

    if not isinstance(context_chunks, list):
        print("  Skipping turn-level BM25: context is not a list")
        return

    # Use mabench's parser to extract individual turns (same as eval_agentic_retrieval)
    from mabench.conversation_parser import parse_chunk

    all_turns: list[tuple[str, str]] = []  # (text, role)
    counter = 0
    last_ts = ""
    for chunk_data in context_chunks:
        turns, counter, last_ts = parse_chunk(str(chunk_data), counter, last_ts)
        for t in turns:
            if not t.content or not t.content.strip():
                continue
            content = t.content.strip()
            if len(content) < 20:
                continue
            # Skip bare timestamp headers
            if content.startswith("Chat Time:") and len(content) < 60:
                continue
            prefix = f"[{t.timestamp}] " if t.timestamp else ""
            role_tag = f"({t.role}) " if t.role != "continuation" else ""
            turn_text = f"{prefix}{role_tag}{t.content}"
            all_turns.append((turn_text, t.role))

    if not all_turns:
        print("  Skipping turn-level BM25: no turns extracted from context")
        return

    # Rebuild BM25 and chunk store with individual turns
    store = foundry._store
    new_bm25 = BM25Index()

    for idx, (turn_text, role) in enumerate(all_turns):
        new_bm25.add(idx, turn_text)
        turn_chunk = RawChunk(
            chunk_idx=idx,
            turn_idx=idx,
            text=turn_text,
            role=role,
        )
        store.chunks.store(turn_chunk)

    store.bm25_chunks = new_bm25
    print(f"  Rebuilt BM25: {len(context_chunks)} session chunks → {len(all_turns)} individual turns")


def _embed_turns(foundry: "Foundry", embed_model: str) -> "Any":
    """Embed all turns in the store for cosine similarity search.

    Returns the SentenceTransformer model for query-time embedding.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embed_model)
    store = foundry._store
    texts = store.bm25_chunks._texts

    if not texts:
        return model

    doc_ids = sorted(texts.keys())
    turn_texts = [texts[doc_id] for doc_id in doc_ids]

    # Batch encode all turns
    embeddings = model.encode(turn_texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Populate vector index
    store.vectors = VectorIndex.from_parts(doc_ids, embeddings)
    print(f"  Embedded {len(doc_ids)} turns ({embed_model}, dim={embeddings.shape[1]})")
    return model


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
    parser.add_argument("--use_recall", action="store_true",
                        help="Use recall() — auto-routes based on query (BM25+cosine+rerank, aggregate, temporal)")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer model for cosine search")
    parser.add_argument("--rerank_model", type=str, default="mixedbread-ai/mxbai-rerank-base-v1",
                        help="Cross-encoder model for reranking")

    # Faceted cache (pre-computed Qwen2.5-7B extractions)
    parser.add_argument("--faceted_cache", type=str, default=None,
                        help="Path to faceted_cache JSON (md5-keyed). "
                             "Bypasses regex extraction with pre-computed entities.")

    # Store persistence (skip re-ingestion on reruns)
    parser.add_argument("--save_store", type=str, default=None,
                        help="Save ingested stores to this directory after ingestion "
                             "(one JSON per context: ctx_0.json, ctx_1.json, ...)")
    parser.add_argument("--load_store", type=str, default=None,
                        help="Load pre-ingested stores from this directory. "
                             "Skips ingestion entirely — queries only.")

    # Ground truth correction (FC-MH only)
    parser.add_argument("--corrected_answers", type=str, default=None,
                        help="JSON file with corrected ground truth")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default=str(_NLCDM_PYTHON / "output" / "engram"))
    parser.add_argument("--tag", type=str, default=None,
                        help="Extra tag for output filename")
    parser.add_argument("--max_contexts", type=int, default=None,
                        help="Limit number of contexts to evaluate (for debugging)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Query decomposition — extract entity/predicate/time from question text
# ---------------------------------------------------------------------------

import re as _re

_COUNT_RE = _re.compile(
    r"how many\b(.+?)(?:do|did|have|has|are|were|does|is|was|can)\b",
    _re.IGNORECASE,
)
_HOW_MUCH_RE = _re.compile(
    r"how much\b(.+?)(?:do|did|have|has|is|was|does)\b",
    _re.IGNORECASE,
)
_TOTAL_AMOUNT_RE = _re.compile(
    r"(?:total amount|total)\b.*?(?:spent on|of)\s+(.+?)(?:\s+in\b|\s+over\b|\?|$)",
    _re.IGNORECASE,
)
_PERCENTAGE_RE = _re.compile(
    r"what percentage.*?(?:on|for|of)\s+(.+?)(?:\?|$)",
    _re.IGNORECASE,
)
_WHEN_RE = _re.compile(
    r"(?:when\s+did\b)"
    r"|(?:which\b.*?\bhappened\s+first\b)"
    r"|(?:what\s+(?:is|was)\s+the\s+(?:order|sequence|chronological)\b)",
    _re.IGNORECASE,
)
_TIME_PHRASE_RE = _re.compile(
    r"\b((?:in the )?(?:past|last|previous|recent)\s+(?:few\s+)?(?:days?|weeks?|months?|years?)|"
    r"(?:ago)|"
    r"(?:in\s+(?:january|february|march|april|may|june|july|august|september|october|november|december))|"
    r"(?:\d{4}/\d{2}(?:/\d{2})?))",
    _re.IGNORECASE,
)

_STOP_WORDS = {
    "i", "me", "my", "the", "a", "an", "is", "are", "was", "were", "do",
    "did", "does", "have", "has", "had", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "and", "or", "but", "not", "that", "this",
    "there", "all", "both", "each", "every", "total", "different", "currently",
    "typically", "typical",
}


def decompose_recall_params(question: str) -> dict:
    """Extract entity/predicate/time from a natural language question.

    Returns dict with keys: entity, predicate, time (all strings, empty if absent).
    No LLM call — pure pattern matching.
    """
    q = question.strip()
    entity = ""
    predicate = ""
    time = ""

    q_lower = q.lower()

    # Temporal: "how many days/weeks/months", "how long", "which happened first", "what is the order"
    if _WHEN_RE.search(q_lower):
        predicate = "when"
        # Still try to extract entity for temporal queries
        m = _COUNT_RE.search(q_lower)
        if m:
            entity = _clean_entity(m.group(1))

    # Counting: "how many X"
    elif "how many" in q_lower:
        predicate = "count"
        m = _COUNT_RE.search(q_lower)
        if m:
            entity = _clean_entity(m.group(1))

    # Spending: "how much did I spend", "total amount spent on"
    elif "how much" in q_lower or "total amount" in q_lower:
        predicate = "sum"
        m = _HOW_MUCH_RE.search(q_lower) or _TOTAL_AMOUNT_RE.search(q_lower)
        if m:
            entity = _clean_entity(m.group(1))

    # Percentage: "what percentage discount"
    elif "what percentage" in q_lower:
        predicate = "count"
        m = _PERCENTAGE_RE.search(q_lower)
        if m:
            entity = _clean_entity(m.group(1))

    # Time phrase extraction (for any question type)
    tm = _TIME_PHRASE_RE.search(q)
    if tm:
        time = tm.group(0).strip()

    return {"entity": entity, "predicate": predicate, "time": time}


_NOISE_WORDS = {
    "different", "various", "pieces", "types", "kind", "kinds",
    "ago", "older", "currently", "typically", "total", "spent",
}


def _clean_entity(raw: str) -> str:
    """Clean extracted entity phrase into a normalized type label."""
    words = raw.strip().lower().split()
    # Remove stop words, noise words, and short words
    words = [w for w in words if w not in _STOP_WORDS and w not in _NOISE_WORDS and len(w) > 1]
    # Remove words that are just time units (they're not entity types)
    words = [w for w in words if w not in {"days", "weeks", "months", "years", "hours"}]
    # Take up to 3 content words, join with underscore
    entity = "_".join(words[:3])
    # Remove trailing/leading punctuation and hyphens
    entity = _re.sub(r"[?.!,;:]+$", "", entity)
    entity = entity.replace("-", "_")
    return entity


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
    # Generic fallback (LME, longmemeval)
    return (
        "Answer the question using ONLY the provided evidence. "
        "Quote relevant phrases from the evidence verbatim when possible. "
        "Keep your answer short (1-2 sentences) but include enough context "
        "to be self-contained. If the evidence doesn't contain the answer, say 'unknown'."
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

    # Load faceted cache (pre-computed Qwen2.5-7B extractions)
    faceted_cache: dict[str, list[dict]] = {}
    if args.faceted_cache:
        with open(args.faceted_cache) as f:
            raw_cache = json.load(f)
        for h, entry in raw_cache.items():
            faceted_cache[h] = entry.get("entities", [])
        print(f"Loaded faceted cache: {len(faceted_cache)} chunks from {args.faceted_cache}")

    # Build Foundry
    config = FoundryConfig(
        chunk_size=args.chunk_size,
        extraction_url=args.extraction_url,
        agent_model=args.agent_model,
        embedding_model="",  # Disable per-chunk embedding; we embed turns after BM25 rebuild
        reranker_model=None,  # Reranker loaded separately if depth >= 4
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
    print(f"  faceted_cache={'YES ('+str(len(faceted_cache))+' chunks)' if faceted_cache else 'NO (regex fallback)'}")
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
    traces: list[dict] = []
    query_index = 0

    n_contexts = len(all_chunks) if args.max_contexts is None else min(args.max_contexts, len(all_chunks))
    for context_idx in range(n_contexts):
        chunks = all_chunks[context_idx]
        qa_pairs = all_qa_pairs[context_idx]

        # Reset Foundry for each context
        foundry = Foundry(config)

        # Load or ingest
        store_path = None
        if args.load_store:
            store_path = Path(args.load_store) / f"ctx_{context_idx}.json"

        if store_path and store_path.exists():
            print(f"\n--- Context {context_idx}: loading store from {store_path} ---")
            ingest_start = time.time()
            foundry = Foundry.load(str(store_path), config)
            ingest_time = time.time() - ingest_start
            print(f"  Loaded in {ingest_time:.1f}s, {len(foundry._store.chunks)} chunks, {len(foundry._store.entity_graph)} entities")
        else:
            print(f"\n--- Context {context_idx}: ingesting {len(chunks)} chunks ---")
            ingest_start = time.time()
            cache_hits = 0
            for chunk_text in chunks:
                if faceted_cache:
                    h = hashlib.md5(chunk_text.encode()).hexdigest()
                    cached_ents = faceted_cache.get(h)
                    if cached_ents:
                        cache_hits += 1
                        next_idx = len(foundry._store.chunks)
                        raw_chunk = RawChunk(
                            text=chunk_text,
                            chunk_idx=next_idx,
                            turn_idx=next_idx,
                            role="user",
                        )
                        entities = [
                            FacetEntity(
                                name=e["name"],
                                facets=e.get("facets", {}),
                                turn_idx=raw_chunk.turn_idx,
                                source_text=chunk_text[:200],
                            )
                            for e in cached_ents
                            if e.get("name", "").strip()
                        ]
                        foundry.ingest_chunk(raw_chunk, entities=entities)
                        continue
                foundry.ingest(chunk_text)
            ingest_time = time.time() - ingest_start
            n_links = foundry.build_type_hierarchy()
            if faceted_cache:
                print(f"  Ingested in {ingest_time:.1f}s, {cache_hits}/{len(chunks)} cache hits, {n_links} type hierarchy links")
            else:
                print(f"  Ingested in {ingest_time:.1f}s, {n_links} type hierarchy links")

            # Rebuild BM25 with turn-level granularity.
            raw_context = conversation_creator.contexts[context_idx]
            _rebuild_bm25_with_turns(foundry, raw_context)

            # Embed turns for cosine search in recall()
            if args.use_recall:
                embed_start = time.time()
                _embed_model = _embed_turns(foundry, args.embed_model)
                print(f"  Embedded in {time.time() - embed_start:.1f}s")

            # Save store for future reruns
            if args.save_store:
                save_dir = Path(args.save_store)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"ctx_{context_idx}.json"
                foundry.save(str(save_path))
                print(f"  Saved store to {save_path}")

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
                traces.append({
                    "query_index": query_index,
                    "context_idx": context_idx,
                    "question": retrieval_query[:200],
                    "terminated_by": _trace.terminated_by,
                    "n_steps": len(_trace.steps),
                    "total_ms": _trace.total_ms,
                    "steps": [
                        {"tool": s.get("tool", ""), "args": s.get("arguments", {})}
                        for s in _trace.steps
                    ],
                })
            elif args.use_recall:
                from foundry.retrieval import recall
                embed_fn = None
                if '_embed_model' in dir():
                    embed_fn = lambda text: _embed_model.encode(text, normalize_embeddings=True)
                reranker = None
                if args.rerank_model:
                    from foundry.retrieval import CrossEncoderReranker
                    if not hasattr(parse_args, '_reranker'):
                        parse_args._reranker = CrossEncoderReranker(args.rerank_model, top_k=10)
                    reranker = parse_args._reranker
                params = decompose_recall_params(retrieval_query)
                evidence = recall(
                    foundry._store,
                    retrieval_query,
                    entity=params["entity"],
                    predicate=params["predicate"],
                    time=params["time"],
                    top_k=args.retrieve_top_k,
                    embed_fn=embed_fn,
                    reranker=reranker,
                )
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
        "traces": traces,
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
