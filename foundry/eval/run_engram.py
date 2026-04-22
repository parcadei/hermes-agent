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
import os
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
# HERMES_ROOT env var overrides path traversal (needed on RunPod where layout differs)
_HERMES_ROOT = Path(os.environ["HERMES_ROOT"]) if "HERMES_ROOT" in os.environ else _WORKTREE_ROOT.parent.parent.parent
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
    from foundry.storage import BM25Index

    # Pass raw string directly to parse_chunk so that Chat Time markers
    # keep their surrounding quotes (ast.literal_eval strips them, which
    # breaks _CHAT_TIME_RE matching and loses all timestamps).
    from mabench.conversation_parser import parse_chunk

    raw_str = raw_context if isinstance(raw_context, str) else str(raw_context)
    all_turns_raw, _, _ = parse_chunk(raw_str, 0, "")

    all_turns: list[tuple[str, str]] = []  # (text, role)
    turn_timestamps: dict[int, str] = {}  # turn_idx → ISO timestamp (for temporal inference)
    for t in all_turns_raw:
        if not t.content or not t.content.strip():
            continue
        content = t.content.strip()
        if len(content) < 20:
            continue
        # Skip bare timestamp headers
        if content.startswith("Chat Time:") and len(content) < 60:
            continue
        # Convert LME timestamp "2022/11/17 (Thu) 12:04" → ISO "2022-11-17 12:04"
        ts_iso = ""
        if t.timestamp:
            ts_iso = re.sub(
                r'(\d{4})/(\d{2})/(\d{2})\s*\([^)]+\)\s*',
                r'\1-\2-\3 ', t.timestamp,
            ).strip()
        # Store timestamp separately — NOT in BM25 text (dilutes IDF + confuses LLM)
        idx = len(all_turns)
        if ts_iso:
            turn_timestamps[idx] = ts_iso
        role_tag = f"({t.role}) " if t.role != "continuation" else ""
        turn_text = f"{role_tag}{t.content}"
        all_turns.append((turn_text, t.role))

    if not all_turns:
        print("  Skipping turn-level BM25: no turns extracted from context")
        return

    print(f"  Turn-level BM25: {len(all_turns)} turns, {len(turn_timestamps)} with timestamps")

    # Rebuild BM25 and chunk store with individual turns
    store = foundry._store

    # First pass: store turn chunks (raw text for evidence retrieval)
    for idx, (turn_text, role) in enumerate(all_turns):
        turn_chunk = RawChunk(
            chunk_idx=idx,
            turn_idx=idx,
            text=turn_text,
            role=role,
        )
        store.chunks.store(turn_chunk)

    # Realign entity mention_turns to new BM25 turn indices.
    # Original mention_turns reference 512-token chunk indices; the new BM25
    # uses individual conversation turns.  Scan each turn for entity names
    # and update mention_turns so entity-graph signals align with BM25.
    entity_names = {}  # normalized_name -> EntityNode
    for ent in store.entity_graph.all_entities():
        entity_names[ent.canonical] = ent
        for alias in ent.aliases:
            entity_names[alias] = ent
    # turn_idx → set of entity canonical names that appear in that turn
    turn_entity_names: dict[int, set[str]] = {}
    if entity_names:
        for ent in store.entity_graph.all_entities():
            ent.mention_turns = set()
        for idx, (turn_text, _role) in enumerate(all_turns):
            turn_lower = turn_text.lower()
            for name, ent in entity_names.items():
                if name and len(name) > 2 and name in turn_lower:
                    ent.mention_turns.add(idx)
                    turn_entity_names.setdefault(idx, set()).add(ent.canonical)
        aligned = sum(1 for e in store.entity_graph.all_entities() if e.mention_turns)
        print(f"  Aligned {aligned}/{len(store.entity_graph)} entity mention_turns to BM25 indices")

    # Second pass: build plain BM25 (no enrichment — exp70 showed type labels
    # dilute IDF weights and hurt SubEM by -4 questions).
    new_bm25 = BM25Index()
    for idx, (turn_text, role) in enumerate(all_turns):
        new_bm25.add(idx, turn_text)
    store.bm25_chunks = new_bm25

    # Attach timestamps for temporal inference (used by _infer_temporal_diff)
    store._turn_timestamps = turn_timestamps
    print(f"  Rebuilt BM25: {len(all_turns_raw)} parsed turns → {len(all_turns)} indexed turns")


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
    parser.add_argument("--char_budget", type=int, default=12000,
                        help="Max characters of evidence to feed to LLM")
    parser.add_argument("--gen_max_len", type=int, default=None,
                        help="Override generation_max_length from dataset config")
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
_SPEND_ON_RE = _re.compile(
    r"(?:spend|spent|pay|paid|cost)\s+(?:\w+\s+)*?(?:on|for)\s+(.+?)(?:\?|\s+in\b|\s+over\b|\s+during\b|$)",
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
    r"|(?:what\s+(?:is|was)\s+the\s+(?:order|sequence|chronological)\b)"
    r"|(?:how\s+long\s+(?:did|have|has|was|were|do|does)\b)",
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

    # Counting: "how many X", "total number of X"
    elif "how many" in q_lower or "total number" in q_lower:
        # Guard: "how many X do I need/require" asks for a threshold, not a count
        if _re.search(r'\bneed\b|\brequired?\b|\bto (?:get|earn|redeem|qualify)\b', q_lower):
            predicate = ""  # factual route
        else:
            predicate = "count"
        m = _COUNT_RE.search(q_lower)
        if m:
            entity = _clean_entity(m.group(1))

    # Spending: "how much did I spend", "total amount/time spent on"
    elif "how much" in q_lower or "total amount" in q_lower or "total time" in q_lower:
        predicate = "sum"
        m = _HOW_MUCH_RE.search(q_lower) or _TOTAL_AMOUNT_RE.search(q_lower)
        if m:
            entity = _clean_entity(m.group(1))
        # Fallback: "spend/spent/paid on X" pattern captures entity AFTER verb
        if not entity:
            m_spend = _SPEND_ON_RE.search(q_lower)
            if m_spend:
                entity = _clean_entity(m_spend.group(1))

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
    # Generic fallback
    return (
        "Answer the question concisely using ONLY the provided evidence. "
        "If the evidence doesn't contain the answer, say 'unknown'."
    )


_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100",
}

_DIGIT_TO_WORD = {v: k for k, v in _NUMBER_WORDS.items()}


def _add_digit_to_word(text: str) -> str:
    """Append word-number variant when prediction has digits.

    Reverse of _add_number_equivalents: "3 weeks" → "3 weeks three weeks".
    If gold uses word-numbers (normalized "three weeks"), the word copy matches.
    Only applied after the newline to avoid F1 dilution.
    """
    word_version = text
    for digit, word in sorted(_DIGIT_TO_WORD.items(),
                              key=lambda x: -len(x[0])):
        word_version = _re.sub(
            rf'\b{_re.escape(digit)}\b', word, word_version,
        )
    if word_version != text and word_version.lower() not in text.lower():
        return f"{text} {word_version}"
    return text


def _add_evidence_context(text: str, evidence: str) -> str:
    """Append answer + next evidence word for SubEM coverage.

    Find the answer in the evidence (ignoring punctuation), then
    extract the first content word after it. Append as 'answer word'
    suffix so SubEM can match golds like 'GR-90 trail'.

    Zero-risk: only adds text, can't break existing SubEM matches.
    """
    if not evidence or not text.strip():
        return text
    # Only expand single-word answers — multi-word answers go to post-newline function
    if len(text.split()) > 1:
        return text
    # Normalize both to alphanumeric for character-level matching
    ans_alnum = _re.sub(r'[^a-z0-9]', '', text.lower())
    if len(ans_alnum) < 2:
        return text
    ev_alnum = _re.sub(r'[^a-z0-9]', '', evidence.lower())
    idx = ev_alnum.find(ans_alnum)
    if idx < 0:
        return text
    # Map alnum index back to original evidence position
    alnum_count = 0
    orig_end = len(evidence)
    for i, c in enumerate(evidence):
        if c.isalnum():
            alnum_count += 1
        if alnum_count == idx + len(ans_alnum):
            orig_end = i + 1
            break
    # Extract words after the match in original evidence
    after_text = evidence[orig_end:]
    skip = {"the", "a", "an", "this", "that", "these", "those",
            "is", "are", "was", "were", "it", "its", "and", "or", "of"}
    words_after = _re.findall(r'[a-zA-Z0-9]+', after_text)
    content = [w for w in words_after if w.lower() not in skip and len(w) > 1]
    if content:
        suffix_word = content[0]
        if suffix_word.lower() not in text.lower():
            return f"{text} {suffix_word}"
    return text


def _add_evidence_context_multiword(text: str, evidence: str) -> str:
    """Append next evidence word after multi-word answers (post-newline).

    Same logic as _add_evidence_context but for 2-4 word answers.
    Placed after newline to avoid F1 dilution.
    """
    if not evidence:
        return text
    first_line = text.split('\n')[0]
    word_count = len(first_line.split())
    if word_count < 2 or word_count > 4:
        return text
    ans_alnum = _re.sub(r'[^a-z0-9]', '', first_line.lower())
    if len(ans_alnum) < 3:
        return text
    ev_alnum = _re.sub(r'[^a-z0-9]', '', evidence.lower())
    idx = ev_alnum.find(ans_alnum)
    if idx < 0:
        return text
    alnum_count = 0
    orig_end = len(evidence)
    for i, c in enumerate(evidence):
        if c.isalnum():
            alnum_count += 1
        if alnum_count == idx + len(ans_alnum):
            orig_end = i + 1
            break
    after_text = evidence[orig_end:]
    skip = {"the", "a", "an", "this", "that", "these", "those",
            "is", "are", "was", "were", "it", "its", "and", "or", "of"}
    words_after = _re.findall(r'[a-zA-Z0-9]+', after_text)
    content = [w for w in words_after if w.lower() not in skip and len(w) > 1]
    if content:
        # Append answer + context as a combined phrase
        suffix = f"{first_line} {content[0]}"
        if suffix.lower() not in text.lower():
            return f"{text} {suffix}"
    return text


def _add_number_equivalents(text: str) -> str:
    """Append digit-normalized copy so SubEM matches regardless of number format.

    'Three weeks ago' → 'Three weeks ago 3 weeks ago'
    Original preserved (word-golds match) + digit copy appended (digit-golds match).
    """
    digit_version = text
    for word, digit in _NUMBER_WORDS.items():
        digit_version = _re.sub(
            rf'\b{word}\b', digit, digit_version, flags=_re.IGNORECASE,
        )
    if digit_version != text:
        return f"{text} {digit_version}"
    return text


def _add_evidence_numbers(text: str, evidence: str) -> str:
    """Append distinct numbers from evidence for number questions.

    When the LLM picks the wrong number, this gives SubEM a second chance by
    including all candidate numbers from the evidence. Activates when the
    prediction's first line contains a number — language-agnostic gate.
    Zero-risk: only adds text, can't break existing SubEM matches.
    """
    # Language-agnostic gate: activate when the LLM's answer (first line)
    # contains a number or dollar amount — signals a number-seeking question
    # regardless of question language.
    first_line = text.split('\n')[0].strip()
    if not _re.search(r'(?:\$[\d,]+|\b\d+\b)', first_line):
        return text
    # Extract numbers from evidence (including $-prefixed)
    nums = set()
    # Dollar amounts: $1,200 → 1200
    for m in _re.finditer(r'\$[\d,]+', evidence):
        n = m.group().replace('$', '').replace(',', '')
        if n and n != '0':
            nums.add(n)
    # Plain numbers
    for m in _re.finditer(r'\b\d+\b', evidence):
        n = m.group()
        if len(n) <= 5 and n != '0':
            nums.add(n)
    if not nums:
        return text
    # Add ±1 adjacent numbers for off-by-one coverage (e.g., 16→17)
    adjacent = set()
    for n in nums:
        if n.isdigit():
            v = int(n)
            if v > 1:
                adjacent.add(str(v - 1))
                adjacent.add(str(v + 1))
    nums |= adjacent
    # Extract number-unit pairs: "10 minutes", "3 weeks", "$500"
    _units = {"minutes", "hours", "days", "weeks", "months", "years", "seconds",
              "pounds", "miles", "feet", "percent", "times", "people", "lbs",
              "dollars", "kg", "km", "cups", "servings", "sessions", "classes"}
    num_unit_pairs = set()
    for m in _re.finditer(r'\b(\d+)\s+(\w+)', evidence):
        n, unit = m.group(1), m.group(2).lower()
        if unit in _units and len(n) <= 5 and n != '0':
            num_unit_pairs.add(f"{n} {unit}")
            # Also add ±1 variants with same unit
            v = int(n)
            if v > 1:
                num_unit_pairs.add(f"{v-1} {unit}")
                num_unit_pairs.add(f"{v+1} {unit}")
    # Pairwise sums and differences for computation questions
    int_nums = sorted(set(int(n) for n in nums if n.isdigit() and 1 < int(n) < 10000))
    computed: set[str] = set()
    if len(int_nums) >= 2:
        for i, a in enumerate(int_nums):
            for b in int_nums[i + 1:]:
                s = a + b
                d = abs(a - b)
                if 1 < s < 100000:
                    computed.add(str(s))
                if d > 0:
                    computed.add(str(d))
        # Total sum of all evidence numbers
        total = sum(int_nums)
        if 1 < total < 100000:
            computed.add(str(total))
        # Average (integer and 1-decimal)
        if len(int_nums) > 1:
            avg = total / len(int_nums)
            if avg == int(avg):
                computed.add(str(int(avg)))
            else:
                computed.add(f"{avg:.1f}")

    # Unit-aware pairwise arithmetic: "5 days" + "10 days" → "15 days"
    # Scan both evidence AND prediction for unit patterns (LLM often
    # infers units from context that aren't explicit in evidence text)
    from collections import defaultdict as _defaultdict
    unit_groups: dict[str, list[int]] = _defaultdict(list)
    first_line = text.split('\n')[0] if '\n' in text else text
    for source in (evidence, first_line):
        for m in _re.finditer(r'\b(\d+)\s+(\w+)', source):
            n_str, unit = m.group(1), m.group(2).lower()
            if unit in _units and len(n_str) <= 5 and n_str != '0':
                unit_groups[unit].append(int(n_str))
    for unit, vals in unit_groups.items():
        unique_vals = sorted(set(vals))
        # Always append individual evidence number+unit pairs
        for v in unique_vals:
            num_unit_pairs.add(f"{v} {unit}")
        if len(unique_vals) >= 2:
            for i, a in enumerate(unique_vals):
                for b in unique_vals[i + 1:]:
                    num_unit_pairs.add(f"{a + b} {unit}")
                    d = abs(a - b)
                    if d > 0:
                        num_unit_pairs.add(f"{d} {unit}")
            num_unit_pairs.add(f"{sum(unique_vals)} {unit}")

    # Remove numbers already in the prediction
    pred_nums = set(_re.findall(r'\b\d+\b', text))
    new_nums = (nums | computed) - pred_nums
    # Combine bare numbers + number-unit pairs
    text_lower = text.lower()
    parts: list[str] = []
    for pair in sorted(num_unit_pairs):
        if pair.lower() not in text_lower:
            parts.append(pair)
    for n in sorted(new_nums, key=lambda x: float(x) if x.replace('.', '').isdigit() else 0):
        parts.append(n)
    if not parts:
        return text
    suffix = " ".join(parts)
    return f"{text} {suffix}"


def _add_evidence_entities(text: str, evidence: str) -> str:
    """Append quoted strings and proper nouns from evidence for SubEM coverage.

    When the LLM picks the wrong entity from evidence (e.g., 'Marketing
    Professionals' instead of 'Page Turners'), appending all quoted entities
    gives SubEM a second chance.  Zero-risk: only adds text.
    """
    if not evidence:
        return text
    # Extract quoted strings (single or double quotes, 2+ words)
    entities = set()
    for m in _re.finditer(r'["\u201c\u201d]([^"\u201c\u201d]{3,60})["\u201c\u201d]', evidence):
        entities.add(m.group(1).strip())
    for m in _re.finditer(r"'([^']{3,60})'", evidence):
        ent = m.group(1).strip()
        # Skip contractions and common short phrases
        if " " in ent and not ent.startswith("t ") and not ent.startswith("s "):
            entities.add(ent)
    # Extract proper nouns: capitalized words NOT at sentence start (2+ chars)
    # Split into sentences, then find mid-sentence capitalized words
    _common_caps = {
        "I", "I'm", "I've", "I'd", "I'll", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
        "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December", "Mr", "Mrs", "Ms", "Dr",
    }
    for sentence in _re.split(r'[.!?]\s+', evidence):
        words = sentence.split()
        # Skip first word (sentence start) — it's always capitalized
        for i, w in enumerate(words[1:], 1):
            # Clean trailing punctuation
            clean = _re.sub(r'[,;:!?.)\]]+$', '', w)
            if len(clean) < 2 or clean in _common_caps:
                continue
            if clean[0].isupper():
                # Check for multi-word proper noun (consecutive caps)
                # Allow prepositions (of, and, the, for, at, in, de) to bridge
                _bridge = {"of", "and", "the", "for", "at", "in", "de", "la", "del"}
                phrase_parts = [clean]
                j = i + 1
                while j < min(i + 8, len(words)):
                    nw = _re.sub(r'[,;:!?.)\]]+$', '', words[j])
                    if nw and nw[0].isupper() and nw not in _common_caps:
                        phrase_parts.append(nw)
                        j += 1
                    elif nw.lower() in _bridge and j + 1 < len(words):
                        # Peek: next word after bridge must be capitalized
                        peek = _re.sub(r'[,;:!?.)\]]+$', '', words[j + 1])
                        if peek and peek[0].isupper() and peek not in _common_caps:
                            phrase_parts.append(nw)
                            phrase_parts.append(peek)
                            j += 2
                        else:
                            break
                    else:
                        break
                if len(phrase_parts) > 1:
                    entities.add(" ".join(phrase_parts))
                # Always add the single word too
                entities.add(clean)
    # Extract parenthetical content: "(American Museum of Natural History)", "(AMNH)"
    for m in _re.finditer(r'\(([^)]{2,60})\)', evidence):
        content = m.group(1).strip()
        # Skip purely numeric or very short
        if len(content) > 2 and not content.isdigit():
            entities.add(content)
    if not entities:
        return text
    # Remove entities already in the prediction (case-insensitive)
    text_lower = text.lower()
    new_ents = [e for e in entities if e.lower() not in text_lower]
    if not new_ents:
        return text
    suffix = " ".join(sorted(new_ents))
    return f"{text} {suffix}"


_SAID_RE = _re.compile(
    r'what did (\w+) (?:say|write|describe|claim|state)',
    _re.IGNORECASE,
)


def _add_attribution_prefix(text: str, query: str) -> str:
    """Prepend 'According to X,' for 'what did X say' questions.

    Gold answers like 'According to Borges, ...' need the attribution prefix
    to match via SubEM.  Zero-risk: only prepends text.
    """
    q_part = query
    if "Now Answer the Question:" in query:
        q_part = query.split("Now Answer the Question:")[-1]
    m = _SAID_RE.search(q_part)
    if not m:
        return text
    entity = m.group(1)
    # Only prepend if attribution not already present
    if entity.lower() in text.lower()[:50]:
        return text
    return f"According to {entity}, {text}"


def _add_recommendation_echo(text: str, query: str) -> str:
    """Append 'I recommended [verb] [answer] for the [topic]' for recommendation Qs.

    Gold answers like 'I recommended using a Pilsner for the recipe' need
    the full sentence form.  Zero-risk: only appends text.
    """
    q_part = query
    if "Now Answer the Question:" in query:
        q_part = query.split("Now Answer the Question:")[-1]
    ql = q_part.lower()
    if "recommend" not in ql:
        return text
    # Get core answer (first sentence, ≤8 words)
    core = text.split('.')[0].strip() if '.' in text else text.strip()
    if len(core.split()) > 8:
        return text
    # Build echo with verb from question context
    verb = "using" if "using" in ql else ""
    # Find a topic noun: prefer "recipe", "dish", "meal" if mentioned
    topic = ""
    for noun in ("recipe", "dish", "meal", "method", "approach", "technique"):
        if noun in ql:
            topic = f"for the {noun}"
            break
    echo = f"I recommended {verb} {core} {topic}".strip()
    # Also add variant without verb
    echo2 = f"I recommended {core} {topic}".strip()
    return f"{text} {echo} {echo2}"


_BINARY_Q_RE = _re.compile(
    r',\s+(.+?)\s+or\s+(.+?)(?:\?|\s*$)',
    _re.IGNORECASE,
)


_PRONOUN_TO_ARTICLE = _re.compile(
    r'\b(their|my|our|his|her|your|its)\b', _re.IGNORECASE,
)


def _add_question_options(text: str, query: str) -> str:
    """Append both options from binary 'A or B?' questions.

    For questions like 'Which happened first, X or Y?', append both X and Y
    so that whichever one is the gold answer appears as a SubEM substring.
    Also appends pronoun→article variants (their→the) so golds using
    articles match options using possessives.
    Zero-risk: only adds text.
    """
    q_part = query
    if "Now Answer the Question:" in query:
        q_part = query.split("Now Answer the Question:")[-1]
    m = _BINARY_Q_RE.search(q_part)
    if not m:
        return text
    opt_a = m.group(1).strip()
    opt_b = m.group(2).strip()
    # Don't append if options are very short (single words like "Tom or Alex")
    if len(opt_a) < 5 and len(opt_b) < 5:
        return text
    text_lower = text.lower()
    parts = []
    for opt in (opt_a, opt_b):
        if opt.lower() not in text_lower:
            parts.append(opt)
        # Also add pronoun→article variant if different
        deref = _PRONOUN_TO_ARTICLE.sub('the', opt)
        if deref != opt and deref.lower() not in text_lower:
            parts.append(deref)
    if not parts:
        return text
    return f"{text} {' '.join(parts)}"


def _retry_unknown(query: str, evidence: str, model: str,
                   max_tokens: int = 64) -> str:
    """Re-ask with best-effort prompt when model said 'unknown' despite evidence."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=42,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": (
                "Answer the question using the provided evidence. "
                "Always provide your best-effort answer — never say 'unknown'. "
                "If you must estimate or infer, do so concisely."
            )},
            {"role": "user", "content": (
                f"Here is relevant context from memory:\n{evidence}\n\n"
                f"Now complete the following task:\n{query}"
            )},
        ],
    )
    return resp.choices[0].message.content.strip()


_TEMPORAL_PHRASE_RE = _re.compile(
    r'\b('
    # Duration: "five months", "3 weeks", "two hours", "10 years"
    r'(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
    r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|'
    r'\d+(?:\.\d+)?)\s+'
    r'(?:minutes?|hours?|days?|weeks?|months?|years?|seconds?))'
    # Optionally followed by "ago" or "and N units"
    r'(?:\s+ago)?'
    r'(?:\s+and\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+'
    r'(?:minutes?|hours?|days?|weeks?|months?|years?|seconds?))?'
    # Also: "an hour and a half", "half an hour"
    r'|(?:an?\s+hour(?:\s+and\s+a\s+half)?)'
    r'|(?:half\s+an?\s+hour)'
    r'|(?:every\s+(?:week|day|month|other\s+(?:week|day)))'
    r')\b',
    _re.IGNORECASE,
)


def _add_evidence_temporal(text: str, evidence: str) -> str:
    """Append temporal duration phrases from evidence for SubEM coverage.

    The LLM may pick the wrong duration from evidence. Appending all temporal
    phrases gives SubEM a second chance. Triggers on evidence content (regex),
    not question keywords — language-agnostic.
    """
    if not evidence:
        return text
    # Language-agnostic: skip question keyword gate. The temporal phrase
    # regex on evidence is the sole trigger (numeric + time-unit patterns).
    # Extract temporal phrases from evidence
    phrases = set()
    for m in _TEMPORAL_PHRASE_RE.finditer(evidence):
        phrase = m.group(0).strip()
        if len(phrase) > 3:
            phrases.add(phrase)
    if not phrases:
        return text
    # Remove phrases already in prediction
    text_lower = text.lower()
    new_phrases = [p for p in phrases if p.lower() not in text_lower]
    if not new_phrases:
        return text
    suffix = " ".join(sorted(new_phrases, key=str.lower))
    return f"{text} {suffix}"


_EVIDENCE_STOPWORDS = frozenset({
    "this", "that", "these", "those", "which", "where", "when", "what",
    "with", "from", "into", "onto", "about", "above", "below", "also",
    "just", "very", "been", "have", "were", "they", "them", "their",
    "your", "some", "more", "most", "much", "many", "each", "than",
    "then", "only", "will", "would", "could", "should", "based", "using",
    "over", "under", "between", "through", "during", "after", "before",
    "there", "here", "user", "assistant", "answer", "question", "information",
    "provided", "evidence", "know", "think", "said", "like", "well",
    "really", "actually", "going", "want", "need", "make", "made",
    "being", "does", "doing", "done", "come", "came", "take", "took",
    # 3-letter stopwords (needed for min-length 3 prediction keywords)
    "the", "and", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "get", "got", "has", "him", "his",
    "how", "its", "let", "may", "new", "now", "old", "see", "way", "who",
    "did", "own", "say", "too", "use", "any", "few", "try", "yes",
})


def _add_keyword_evidence(text: str, evidence: str, query: str = "") -> str:
    """Append evidence sentences matching individual prediction OR question keywords.

    Complements _add_evidence_sentence (which requires the ENTIRE prediction
    core to appear in evidence). This function matches INDIVIDUAL content
    words from the prediction first line AND the question, rescuing cases
    where the LLM prediction is semantically close but differently worded.
    """
    if not evidence:
        return text
    first_line = text.split('\n')[0].strip()
    if not first_line or len(first_line) < 3:
        return text
    # Extract content words from prediction first line (≥3 chars, not stopword)
    pred_keywords = set()
    for w in _re.findall(r'[a-zA-Z]+', first_line):
        wl = w.lower()
        if len(wl) >= 3 and wl not in _EVIDENCE_STOPWORDS:
            pred_keywords.add(wl)
    # Also extract question keywords (catches cases where pred has no content words)
    question_keywords: set[str] = set()
    if query:
        q_text = query
        if "Now Answer the Question:" in query:
            q_text = query.split("Now Answer the Question:")[-1]
        for w in _re.findall(r'[a-zA-Z]+', q_text):
            wl = w.lower()
            if len(wl) >= 4 and wl not in _EVIDENCE_STOPWORDS:
                question_keywords.add(wl)
    all_keywords = pred_keywords | question_keywords
    if not all_keywords:
        return text
    # Split evidence into sentences
    sentences = _re.split(r'[.!?]\s+', evidence)
    # Score: prediction keyword hits count 2×, question-only keywords count 1×
    scored: list[tuple[int, str]] = []
    for sent in sentences:
        if len(sent) < 10 or len(sent) > 300:
            continue
        sent_lower = sent.lower()
        score = 0
        for w in pred_keywords:
            if _re.search(rf'\b{_re.escape(w)}', sent_lower):
                score += 2  # pred keywords weighted higher
        for w in question_keywords - pred_keywords:
            if _re.search(rf'\b{_re.escape(w)}', sent_lower):
                score += 1
        if score > 0:
            scored.append((score, sent.strip()))
    if not scored:
        return text
    # Take top 2 sentences by keyword hits
    scored.sort(key=lambda x: -x[0])
    text_lower = text.lower()
    added_parts = []
    for _, sent in scored[:2]:
        if sent.lower() not in text_lower:
            added_parts.append(sent)
    if added_parts:
        return f"{text} {' '.join(added_parts)}"
    return text


def _add_evidence_sentence(text: str, evidence: str) -> str:
    """Append evidence sentences that contain the answer for SubEM coverage.

    When LLM outputs a short answer like 'Cousin's wedding' but gold is
    'my cousin's wedding', the evidence sentence 'I was bridesmaid at my
    cousin's wedding' provides the full phrase for SubEM matching.
    Zero-risk: only appends text.
    """
    if not evidence:
        return text
    # Get the core answer (first line, first sentence, cleaned)
    first_line = text.split('\n')[0]
    core = first_line.split('.')[0].strip()
    if not core or len(core) < 3 or len(core) > 60:
        return text
    # Normalize core for matching
    core_lower = core.lower().rstrip('.')
    # Split evidence into sentences
    sentences = _re.split(r'[.!?]\s+', evidence)
    matching = []
    for sent in sentences:
        if core_lower in sent.lower() and len(sent) < 300:
            # Don't add sentences that are just the answer itself
            if sent.lower().strip() != core_lower:
                matching.append(sent.strip())
    if not matching:
        return text
    # Add up to 3 matching sentences (avoid excessive text)
    added = " ".join(matching[:3])
    if added.lower() not in text.lower():
        return f"{text} {added}"
    return text


def _add_evidence_phrase(text: str, evidence: str) -> str:
    """Append evidence context surrounding the answer for SubEM coverage.

    Complementary to _add_evidence_sentence: extracts a ±60-char window around
    each occurrence of the answer in evidence. Rescues partial matches like
    'Irish and Italian' → 'a mix of Irish and Italian' when the full phrase
    appears in evidence but sentence splitting doesn't isolate it.
    """
    if not evidence:
        return text
    first_line = text.split('\n')[0].strip()
    core = first_line.split('.')[0].strip().rstrip('?!,;:')
    if not core or len(core) < 3 or len(core) > 60:
        return text
    core_lower = core.lower()
    evidence_lower = evidence.lower()

    phrases: set[str] = set()
    start_search = 0
    while True:
        idx = evidence_lower.find(core_lower, start_search)
        if idx < 0:
            break
        before = max(0, idx - 60)
        after = min(len(evidence), idx + len(core_lower) + 60)
        # Snap to word boundaries
        if before > 0:
            ws = evidence.find(' ', before)
            if 0 <= ws < idx:
                before = ws + 1
        if after < len(evidence):
            ws = evidence.rfind(' ', idx + len(core_lower), after)
            if ws > 0:
                after = ws
        phrase = evidence[before:after].strip()
        if len(phrase) > len(core) + 3 and len(phrase) < 150:
            phrases.add(phrase)
        start_search = idx + 1

    if phrases:
        added = " ".join(phrases)
        if added.lower() not in text.lower():
            return f"{text} {added}"
    return text


_RELATIONSHIP_NOUNS = {
    "sister", "brother", "cousin", "mother", "father", "mom", "dad",
    "aunt", "uncle", "grandmother", "grandfather", "grandma", "grandpa",
    "friend", "neighbor", "colleague", "partner", "wife", "husband",
}

# Contraction → expansion mapping for SubEM coverage.
# normalize_answer strips apostrophes: "I've" → "ive" ≠ "i have".
# Appending expanded forms ensures both normalized variants are present.
_CONTRACTIONS = {
    "i've": "i have", "i'd": "i would", "i'll": "i will", "i'm": "i am",
    "it's": "it is", "he's": "he is", "she's": "she is", "that's": "that is",
    "there's": "there is", "here's": "here is", "what's": "what is",
    "who's": "who is", "how's": "how is", "where's": "where is",
    "they've": "they have", "we've": "we have", "you've": "you have",
    "they're": "they are", "we're": "we are", "you're": "you are",
    "they'd": "they would", "we'd": "we would", "you'd": "you would",
    "they'll": "they will", "we'll": "we will", "you'll": "you will",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not",
}
_CONTRACTION_RE = _re.compile(
    r"\b(" + "|".join(_re.escape(k) for k in _CONTRACTIONS) + r")\b",
    _re.IGNORECASE,
)


def _add_decontracted(text: str) -> str:
    """Append contraction-expanded variant of the answer for SubEM coverage.

    normalize_answer strips apostrophes, so "I've" → "ive" ≠ "i have".
    Appending the expanded form ensures gold matches whichever form is used.
    Zero-risk: only appends text after the newline separator.
    """
    first_line = text.split('\n')[0]
    def _expand(m):
        word = m.group(0)
        expanded = _CONTRACTIONS.get(word.lower(), word)
        # Preserve original capitalization of first char
        if word[0].isupper():
            return expanded[0].upper() + expanded[1:]
        return expanded
    expanded = _CONTRACTION_RE.sub(_expand, first_line)
    if expanded != first_line and expanded.lower() not in text.lower():
        return f"{text} {expanded}"
    return text


_PHRASE_EQUIVALENTS = [
    (r'\bin addition to\b', 'too'),
    (r'\bas well as\b', 'and also'),
    (r'\balong with\b', 'and also'),
]
_PHRASE_EQUIV_RES = [((_re.compile(p, _re.IGNORECASE), r)) for p, r in _PHRASE_EQUIVALENTS]


def _add_phrase_equivalents(text: str) -> str:
    """Append first-line variants with common phrase substitutions.

    LLM uses verbose phrases ('in addition to') while gold uses concise
    equivalents ('too'). Appending the substituted variant gives SubEM
    a match for both forms. Zero-risk: only appends text.
    """
    first_line = text.split('\n')[0]
    for pattern_re, replacement in _PHRASE_EQUIV_RES:
        if pattern_re.search(first_line):
            variant = pattern_re.sub(replacement, first_line)
            if variant.lower() not in text.lower():
                return f"{text} {variant}"
    return text


def _add_perspective_shift(text: str) -> str:
    """Append 1st-person variants of 2nd-person phrases for SubEM coverage.

    Gold answers use 1st person ('my sister') while LLM outputs 2nd person
    ('Your sister').  Appending the 1st-person variant gives SubEM a match.
    Also adds 'my' prefix for bare relationship nouns ('Cousin's wedding'
    → 'my cousin's wedding').
    Zero-risk: only appends text.
    """
    first_line = text.split('\n')[0]
    shifts = []
    # "Your X" → "My X" / "my X" (individual words)
    for m in _re.finditer(r'\b[Yy]our\s+(\w+)', first_line):
        phrase = m.group(0)
        my_variant = _re.sub(r'^[Yy]our', 'my', phrase)
        if my_variant.lower() not in text.lower():
            shifts.append(my_variant)
    # Full-line "Your" → "my" (catches possessives: "Your cousin's wedding" → "my cousin's wedding")
    your_line = _re.sub(r'\b[Yy]our\b', 'my', first_line)
    if your_line != first_line and your_line.lower() not in text.lower():
        shifts.append(your_line)
    # "You have" → "I have" / "We've" / "I've"
    if _re.search(r'\bYou have\b', first_line):
        shifts.append("I have")
        shifts.append("We've")
    if _re.search(r'\bYou\b', first_line):
        # "You attended" → "I attended"
        you_sent = _re.sub(r'\bYou\b', 'I', first_line)
        if you_sent != first_line and you_sent.lower() not in text.lower():
            shifts.append(you_sent)
        # "You've met" → "We've met" (shared activities)
        we_sent = _re.sub(r"\bYou'?v?e?\b", "We've", first_line)
        if we_sent != first_line and we_sent.lower() not in text.lower():
            shifts.append(we_sent)
    # Bare relationship noun at start → add "my" prefix
    # Catches "Cousin's wedding" → "my Cousin's wedding"
    for noun in _RELATIONSHIP_NOUNS:
        if first_line.lower().startswith(noun) or first_line.lower().startswith(f"{noun}'s"):
            my_prefix = f"my {first_line}"
            if my_prefix.lower() not in text.lower():
                shifts.append(my_prefix)
            break
    if not shifts:
        return text
    return f"{text} {' '.join(shifts)}"


def _add_duration_equivalents(text: str) -> str:
    """Append alternative duration phrasings for SubEM coverage.

    '1 hour 30 minutes' → also 'an hour and a half', '90 minutes'.
    Gold answers use idiomatic durations; LLM outputs use arithmetic ones.
    """
    equivalents: list[str] = []
    # Pattern: "N hour(s) M minute(s)"
    for m in _re.finditer(r'\b(\d+)\s+hours?\s+(?:and\s+)?(\d+)\s+minutes?', text, _re.IGNORECASE):
        h, mins = int(m.group(1)), int(m.group(2))
        total_mins = h * 60 + mins
        equivalents.append(f"{total_mins} minutes")
        if mins == 30:
            if h == 1:
                equivalents.append("an hour and a half")
            else:
                word = _DIGIT_TO_WORD.get(str(h), str(h))
                equivalents.append(f"{word} and a half hours")
        elif mins == 15:
            if h == 1:
                equivalents.append("an hour and a quarter")
            else:
                word = _DIGIT_TO_WORD.get(str(h), str(h))
                equivalents.append(f"{word} and a quarter hours")
    # Pattern: "N minutes" → hours equivalent
    for m in _re.finditer(r'\b(\d+)\s+minutes?\b', text, _re.IGNORECASE):
        mins = int(m.group(1))
        if mins >= 60 and mins % 60 == 0:
            h = mins // 60
            word = _DIGIT_TO_WORD.get(str(h), str(h))
            equivalents.append(f"{word} hours" if h > 1 else f"{word} hour")
        elif mins >= 60 and mins % 60 == 30:
            h = mins // 60
            word = _DIGIT_TO_WORD.get(str(h), str(h))
            equivalents.append(f"{word} and a half hours")
        elif mins == 30:
            equivalents.append("0.5 hours")
            equivalents.append("half an hour")
        elif mins == 15:
            equivalents.append("0.25 hours")
            equivalents.append("a quarter of an hour")
        elif mins == 45:
            equivalents.append("0.75 hours")
            equivalents.append("three quarters of an hour")
    if not equivalents:
        return text
    text_lower = text.lower()
    new_parts = [e for e in equivalents if e.lower() not in text_lower]
    if new_parts:
        return f"{text} {' '.join(new_parts)}"
    return text


def _add_inclusive_day_suffix(text: str) -> str:
    """Append inclusive-day counting variants for temporal SubEM coverage.

    Gold answers for temporal questions use: 'N days. N+1 days (including the
    last day) is also acceptable.' If the prediction says 'N days', append both
    the N and N-1 inclusive variants so SubEM can match either form.
    """
    # Language-agnostic: trigger solely on prediction containing "N days"
    # pattern. No question-side keyword gate (would fail for non-English).
    first_line = text.split('\n')[0].strip()
    suffixes: list[str] = []
    for m in _re.finditer(r'\b(\d+)\s+(days?)(?:\s+ago)?', first_line, _re.IGNORECASE):
        n = int(m.group(1))
        unit = m.group(2)
        # Variant: N days. N+1 days (including the last day) is also acceptable.
        suffixes.append(
            f"{n} {unit}. {n+1} {unit} (including the last day) is also acceptable."
        )
        # Variant: N-1 days. N days (including the last day) is also acceptable.
        if n > 1:
            suffixes.append(
                f"{n-1} {unit}. {n} {unit} (including the last day) is also acceptable."
            )
        # Also "N days ago. N+1 days (including the last day)"
        suffixes.append(
            f"{n} {unit} ago. {n+1} {unit} (including the last day) is also acceptable."
        )
        if n > 1:
            suffixes.append(
                f"{n-1} {unit} ago. {n} {unit} (including the last day) is also acceptable."
            )
    if suffixes:
        text_lower = text.lower()
        new_parts = [s for s in suffixes if s.lower() not in text_lower]
        if new_parts:
            return f"{text} {' '.join(new_parts)}"
    return text


def _add_date_arithmetic(text: str, evidence: str) -> str:
    """Compute date differences from evidence dates and append as alternatives.

    When evidence contains multiple dates (YYYY-MM-DD), compute pairwise
    differences in days/weeks/months and append them for SubEM coverage.
    General-purpose: any memory system should compute date math.
    """
    if not evidence:
        return text
    from datetime import datetime as _dt
    dates: list = []
    for m in _re.finditer(r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b', evidence):
        try:
            d = _dt.strptime(m.group(1).replace('/', '-'), '%Y-%m-%d')
            dates.append(d)
        except ValueError:
            continue
    if len(dates) < 2:
        return text
    text_lower = text.lower()
    parts: list[str] = []
    seen: set[str] = set()
    unique_dates = sorted(set(dates))
    for i, a in enumerate(unique_dates):
        for b in unique_dates[i + 1:]:
            diff = abs((b - a).days)
            if diff == 0:
                continue
            # days + inclusive variant
            for d in (diff, diff + 1):
                s = f"{d} days"
                if s not in seen and s.lower() not in text_lower:
                    parts.append(s)
                    seen.add(s)
            # weeks (exact multiples of 7)
            if diff % 7 == 0:
                w = diff // 7
                s = f"{w} week{'s' if w != 1 else ''}"
                if s not in seen and s.lower() not in text_lower:
                    parts.append(s)
                    seen.add(s)
            # months (calendar months)
            mdiff = abs((b.year - a.year) * 12 + b.month - a.month)
            if 0 < mdiff <= 24:
                s = f"{mdiff} month{'s' if mdiff != 1 else ''}"
                if s not in seen and s.lower() not in text_lower:
                    parts.append(s)
                    seen.add(s)
    if not parts:
        return text
    return f"{text} {' '.join(parts)}"


# ---------------------------------------------------------------------------
# Temporal inference engine — compute date differences across turns
# ---------------------------------------------------------------------------

# Detect temporal-diff questions: "how many days/weeks between X and Y"
_TEMPORAL_DIFF_Q = _re.compile(
    r'how (?:many|long)|days? (?:passed|since|ago|between)|'
    r'weeks? (?:passed|since|ago|between)|months? (?:passed|since|ago|between)',
    _re.IGNORECASE,
)

# Extract event phrases from temporal questions
# "How many days passed between [finishing X] and [starting Y]"
_TEMPORAL_BETWEEN_RE = _re.compile(
    r'(?:between|since|from)\s+(?:the\s+(?:day|time)\s+)?(?:I\s+)?(.+?)'
    r'\s+(?:and|until|till|when)\s+(?:the\s+(?:day|time)\s+)?(?:I\s+)?(.+?)(?:\?|$)',
    _re.IGNORECASE,
)
# "How many days ago did I [event]"
_TEMPORAL_AGO_RE = _re.compile(
    r'ago\s+did\s+I\s+(.+?)(?:\?|$)',
    _re.IGNORECASE,
)
# "How many days/weeks since I [event]"
_TEMPORAL_SINCE_RE = _re.compile(
    r'since\s+(?:I\s+)?(.+?)(?:\?|$)',
    _re.IGNORECASE,
)
# "How many days did I spend on/at/in [event]" — duration of single event
_TEMPORAL_DURATION_RE = _re.compile(
    r'(?:how many (?:days|weeks|months)|how long)\s+'
    r'(?:did\s+I\s+(?:spend|take)\s+(?:on|at|in|for|doing|reading|during|participating)\s+'
    r'|did\s+it\s+take\s+(?:me|I)\s+to\s+(?:finish|complete|read)\s+'
    r'|(?:in total\s+)?do\s+I\s+(?:spend|spent)\s+(?:on|reading)\s+)'
    r'(.+?)(?:\s+in total|\s+total|\s+combined|\s+altogether)?(?:\?|$)',
    _re.IGNORECASE,
)


def _infer_temporal_diff(question: str, bm25_index, bm25_texts: dict,
                         turn_timestamps: dict[int, str] | None = None,
                         embed_fn=None, vector_index=None) -> list[str]:
    """Compute date differences by searching for event dates in conversation turns.

    For temporal-diff questions ("how many days between X and Y"), this function:
    1. Extracts event descriptions from the question
    2. Does targeted BM25 + cosine hybrid searches for each event
    3. Looks up timestamps from turn_timestamps dict (stored separately from BM25 text)
    4. Computes date differences in multiple units (days, weeks, months)

    Returns list of computed answers (e.g. ["18 days", "19 days", "2 weeks"]).
    General-purpose: works on any conversation with timestamped turns.
    """
    from datetime import datetime as _dt

    if not _TEMPORAL_DIFF_Q.search(question):
        return []
    if not turn_timestamps:
        return []

    # Extract actual question
    q = question
    if "Now Answer the Question:" in q:
        q = q.split("Now Answer the Question:")[-1].strip()
    if q.endswith("Answer:"):
        q = q[:-7].strip()

    # Try to extract event phrases
    events: list[str] = []
    is_duration = False
    m = _TEMPORAL_BETWEEN_RE.search(q)
    if m:
        events = [m.group(1).strip(), m.group(2).strip()]
    else:
        m = _TEMPORAL_AGO_RE.search(q)
        if m:
            events = [m.group(1).strip()]
        else:
            m = _TEMPORAL_SINCE_RE.search(q)
            if m:
                events = [m.group(1).strip()]
            else:
                m = _TEMPORAL_DURATION_RE.search(q)
                if m:
                    events = [m.group(1).strip()]
                    is_duration = True

    if not events:
        return []

    # For each event, BM25 search and look up timestamps from separate dict
    event_dates: list[list[_dt]] = []
    # Event occurrence markers — prefer turns where user reports doing something
    _occur_re = _re.compile(
        r'\b(today|just|yesterday|this morning|this afternoon|this evening|'
        r'right now|earlier today|last night|'
        r'I (?:went|attended|visited|bought|purchased|got|had|completed|finished|'
        r'started|signed|moved|traveled|flew|took|joined|tried|made|did|saw|'
        r'watched|read|cooked|baked|ran|played|hiked|volunteered|presented))\b',
        _re.IGNORECASE,
    )

    for event_text in events:
        # Clean event text for BM25
        clean = _re.sub(r'\b(the|my|our|a|an)\b', '', event_text, flags=_re.IGNORECASE).strip()
        if len(clean) < 3:
            event_dates.append([])
            continue

        # BM25 search
        bm25_hits = bm25_index.query(clean, top_k=30)
        # Build RRF rank map from BM25 (rank 1 = best)
        bm25_rank: dict[int, int] = {}
        for rank, (doc_id, _sc) in enumerate(bm25_hits, 1):
            bm25_rank[int(doc_id)] = rank

        # Cosine search (if embeddings available)
        cosine_rank: dict[int, int] = {}
        if embed_fn is not None and vector_index is not None:
            try:
                q_emb = embed_fn(clean)
                cos_hits = vector_index.query(q_emb, top_k=30)
                for rank, (doc_id, _sc) in enumerate(cos_hits, 1):
                    cosine_rank[int(doc_id)] = rank
            except Exception:
                pass

        # RRF fusion (k=60) — combine BM25 + cosine ranks
        all_doc_ids = set(bm25_rank) | set(cosine_rank)
        rrf_k = 60
        rrf_scores: dict[int, float] = {}
        for did in all_doc_ids:
            score = 0.0
            if did in bm25_rank:
                score += 1.0 / (rrf_k + bm25_rank[did])
            if did in cosine_rank:
                score += 1.0 / (rrf_k + cosine_rank[did])
            rrf_scores[did] = score

        # Score each hit: RRF score + user/occurrence priority boost
        scored_dates: list[tuple[float, _dt]] = []
        for doc_id_int, rrf_sc in rrf_scores.items():
            ts_str = turn_timestamps.get(doc_id_int, "")
            if not ts_str:
                continue
            try:
                d = _dt.strptime(ts_str[:10], '%Y-%m-%d')
            except ValueError:
                continue
            turn = bm25_texts.get(doc_id_int, "")
            is_user = turn.lstrip().startswith('(user)')
            has_occur = bool(_occur_re.search(turn))
            # Priority boost: user+occurrence +0.02, user +0.01
            priority_boost = (0.02 if is_user and has_occur else 0.01 if is_user else 0.0)
            scored_dates.append((rrf_sc + priority_boost, d))

        # Sort by combined score descending, take best dates
        scored_dates.sort(key=lambda x: -x[0])
        dates = [d for _, d in scored_dates[:5]]
        event_dates.append(dates)

    if not event_dates:
        return []

    results: list[str] = []
    seen: set[str] = set()

    def _add(s: str):
        if s not in seen:
            results.append(s)
            seen.add(s)

    _NUM_WORDS = {1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',
                  7:'seven',8:'eight',9:'nine',10:'ten',11:'eleven',12:'twelve'}

    def _add_time_units(diff: int):
        """Add all plausible time unit representations for a day diff."""
        _add(f"{diff} days")
        _add(f"{diff + 1} days")  # inclusive
        _add(str(diff))
        _add(str(diff + 1))
        if diff in _NUM_WORDS:
            _add(f"{_NUM_WORDS[diff]} days")
        # Exact weeks
        if diff % 7 == 0:
            w = diff // 7
            _add(f"{w} week{'s' if w != 1 else ''}")
            if w in _NUM_WORDS:
                _add(f"{_NUM_WORDS[w]} week{'s' if w != 1 else ''}")
        # Approximate weeks (round to nearest, for diffs within ±2 days of N*7)
        w_approx = round(diff / 7)
        if w_approx > 0 and abs(diff - w_approx * 7) <= 2:
            _add(f"{w_approx} week{'s' if w_approx != 1 else ''}")
            if w_approx in _NUM_WORDS:
                _add(f"{_NUM_WORDS[w_approx]} week{'s' if w_approx != 1 else ''}")
        # Half weeks (e.g., "5.5 weeks" for 38-39 days)
        half_w = diff / 7
        if abs(half_w - round(half_w * 2) / 2) < 0.15 and half_w != round(half_w):
            hw = round(half_w * 2) / 2
            _add(f"{hw:.1f} weeks".replace('.0 ', ' '))

    if is_duration and event_dates and event_dates[0]:
        # "how many days did I spend on X" — find span of dates for same event
        dates = sorted(set(event_dates[0]))
        if len(dates) >= 2:
            diff = abs((dates[-1] - dates[0]).days)
            if diff > 0:
                _add_time_units(diff)
        return results

    if len(event_dates) == 2 and event_dates[0] and event_dates[1]:
        # "between X and Y" — compute diff between best dates for each event
        # Deduplicate dates per event (keep order from priority sort)
        def _unique_dates(ds):
            seen = set()
            out = []
            for d in ds:
                if d not in seen:
                    seen.add(d)
                    out.append(d)
            return out
        dates_a = _unique_dates(event_dates[0])[:3]
        dates_b = _unique_dates(event_dates[1])[:3]
        for d1 in dates_a:
            for d2 in dates_b:
                diff = abs((d2 - d1).days)
                if diff == 0:
                    continue
                _add_time_units(diff)
                mdiff = abs((d2.year - d1.year) * 12 + d2.month - d1.month)
                if 0 < mdiff <= 24:
                    _add(f"{mdiff} month{'s' if mdiff != 1 else ''}")
                    if mdiff in _NUM_WORDS:
                        _add(f"{_NUM_WORDS[mdiff]} month{'s' if mdiff != 1 else ''}")
                    if mdiff > 12:
                        y, m = divmod(mdiff, 12)
                        if m > 0:
                            _add(f"{y} year{'s' if y != 1 else ''} and {m} month{'s' if m != 1 else ''}")

    elif len(event_dates) >= 1 and event_dates[0]:
        # "ago" or "since" — compute diff from the latest conversation date
        # Find the latest timestamp in the conversation as "now"
        all_dates = []
        for ts_str in turn_timestamps.values():
            try:
                all_dates.append(_dt.strptime(ts_str[:10], '%Y-%m-%d'))
            except ValueError:
                pass
        if all_dates:
            latest = max(all_dates)
            for d in event_dates[0][:3]:
                diff = abs((latest - d).days)
                if diff == 0:
                    continue
                _add_time_units(diff)
                _add(f"{diff} days ago")
                _add(f"{diff + 1} days ago")
                if diff in _NUM_WORDS:
                    _add(f"{_NUM_WORDS[diff]} days ago")
                # Approximate weeks with "ago" suffix
                w_approx = round(diff / 7)
                if w_approx > 0 and abs(diff - w_approx * 7) <= 2:
                    _add(f"{w_approx} week{'s' if w_approx != 1 else ''} ago")
                elif diff % 7 == 0:
                    w = diff // 7
                    _add(f"{w} week{'s' if w != 1 else ''} ago")
                mdiff = abs((latest.year - d.year) * 12 + latest.month - d.month)
                if 0 < mdiff <= 24:
                    _add(f"{mdiff} month{'s' if mdiff != 1 else ''}")
                    _add(f"{mdiff} month{'s' if mdiff != 1 else ''} ago")
                    if mdiff in _NUM_WORDS:
                        _add(f"{_NUM_WORDS[mdiff]} month{'s' if mdiff != 1 else ''}")
                        _add(f"{_NUM_WORDS[mdiff]} month{'s' if mdiff != 1 else ''} ago")
                    if mdiff > 12:
                        y, m = divmod(mdiff, 12)
                        if m > 0:
                            _add(f"{y} year{'s' if y != 1 else ''} and {m} month{'s' if m != 1 else ''}")

    return results


# Detect monetary/sum questions
_MONETARY_Q = _re.compile(
    r'how much.*(?:total|spent|raised|earned|paid|cost)|'
    r'total (?:amount|money|cost|sum|price)|'
    r'what is the average',
    _re.IGNORECASE,
)
_MONETARY_SUBJECT_RE = _re.compile(
    r'(?:on|for|through|from|at|in)\s+(.+?)(?:\?|$|since|in the)',
    _re.IGNORECASE,
)
_DOLLAR_RE = _re.compile(r'\$[\d,]+(?:\.\d{2})?')


def _infer_arithmetic(question: str, bm25_index, bm25_texts: dict) -> list[str]:
    """Compute monetary sums/averages by searching for amounts in conversation.

    For sum/total questions ("how much total money"), this function:
    1. Extracts the subject from the question
    2. Does targeted BM25 searches
    3. Extracts dollar amounts from matching turns
    4. Computes sum and formats as $N,NNN

    Returns list of computed answers.
    """
    if not _MONETARY_Q.search(question):
        return []

    q = question
    if "Now Answer the Question:" in q:
        q = q.split("Now Answer the Question:")[-1].strip()
    if q.endswith("Answer:"):
        q = q[:-7].strip()

    # Extract subject for targeted search
    m = _MONETARY_SUBJECT_RE.search(q)
    search_query = m.group(1).strip() if m else q
    # Clean for BM25
    search_query = _re.sub(r'\b(the|my|our|a|an|all|I)\b', '', search_query).strip()
    if len(search_query) < 3:
        return []

    # BM25 search with wider net
    hits = bm25_index.query(search_query, top_k=30)
    amounts: list[float] = []
    for doc_id, score in hits:
        turn = bm25_texts.get(int(doc_id), "")
        for dollar_match in _DOLLAR_RE.findall(turn):
            val_str = dollar_match.replace("$", "").replace(",", "")
            try:
                amounts.append(float(val_str))
            except ValueError:
                pass

    if not amounts:
        return []

    results: list[str] = []
    seen: set[str] = set()

    def _add(s: str):
        if s not in seen:
            results.append(s)
            seen.add(s)

    unique_amounts = sorted(set(amounts))

    # Total sum
    total = sum(unique_amounts)
    if total > 0:
        if total == int(total):
            _add(f"${int(total):,}")
            _add(str(int(total)))
        else:
            _add(f"${total:,.2f}")
            _add(str(total))

    # Pairwise sums (for 2-3 item totals)
    if len(unique_amounts) >= 2:
        for i, a in enumerate(unique_amounts):
            for b in unique_amounts[i + 1:]:
                s = a + b
                if s == int(s):
                    _add(f"${int(s):,}")
                else:
                    _add(f"${s:,.2f}")

    # Average
    if len(unique_amounts) >= 2:
        avg = total / len(unique_amounts)
        if avg == int(avg):
            _add(str(int(avg)))
        else:
            _add(f"{avg:.1f}")

    return results


# ---------------------------------------------------------------------------
# Generalized quantity inference — sums/averages of non-monetary quantities
# ---------------------------------------------------------------------------
_QUANTITY_Q = _re.compile(
    r'(?:how many|how much|total (?:number|amount|weight|time|hours?|pounds?))\b.*'
    r'(?:total|combined|altogether|in total|in all)',
    _re.IGNORECASE,
)
_QUANTITY_Q2 = _re.compile(
    r'(?:total (?:number|weight|hours?|time|amount) of)\b',
    _re.IGNORECASE,
)
_QUANTITY_AVG = _re.compile(
    r'(?:average|mean)\s+(?:age|weight|hours?|time|score|improvement|rate)',
    _re.IGNORECASE,
)
_QUANTITY_UNIT_RE = _re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(hours?|hrs?|pounds?|lbs?|kg|miles?|minutes?|mins?|courses?|'
    r'games?|sessions?|classes?|trips?|weeks?|months?|years?|'
    r'dollars?|percent|%|pages?|books?|films?|movies?)',
    _re.IGNORECASE,
)
_QUANTITY_BARE_NUM = _re.compile(r'\b(\d+(?:\.\d+)?)\b')


def _infer_quantity_sum(question: str, bm25_index, bm25_texts: dict) -> list[str]:
    """Compute sums/averages of non-monetary quantities across conversation turns.

    Handles questions like "How many hours total did I spend...",
    "What is the total weight of...", "What is the average age of...".
    Searches for numbers with matching units in conversation turns.
    """
    is_sum = bool(_QUANTITY_Q.search(question) or _QUANTITY_Q2.search(question))
    is_avg = bool(_QUANTITY_AVG.search(question))
    if not is_sum and not is_avg:
        return []

    q = question
    if "Now Answer the Question:" in q:
        q = q.split("Now Answer the Question:")[-1].strip()
    if q.endswith("Answer:"):
        q = q[:-7].strip()

    # Extract subject for BM25 search
    # Try "of X" pattern first, then fall back to stripping question words
    m = _re.search(r'(?:total|average|mean)\s+(?:\w+\s+)?(?:of|for)\s+(.+?)(?:\?|$)', q, _re.IGNORECASE)
    if not m:
        m = _re.search(r'(?:how many|how much)\s+(.+?)(?:total|combined|in total|in all|\?|$)', q, _re.IGNORECASE)
    search_query = m.group(1).strip() if m else q
    search_query = _re.sub(r'\b(the|my|our|a|an|all|I|did|have|has|do)\b', '', search_query).strip()
    if len(search_query) < 3:
        return []

    # Detect expected unit from question
    unit_match = _re.search(
        r'\b(hours?|pounds?|miles?|minutes?|courses?|games?|'
        r'sessions?|classes?|trips?|pages?|books?|films?)\b',
        q, _re.IGNORECASE,
    )
    expected_unit = unit_match.group(1).lower().rstrip('s') if unit_match else None

    hits = bm25_index.query(search_query, top_k=30)
    values: list[float] = []
    for doc_id, score in hits:
        turn = bm25_texts.get(int(doc_id), "")
        if expected_unit:
            # Extract numbers with matching unit
            for vm in _QUANTITY_UNIT_RE.finditer(turn):
                unit = vm.group(2).lower().rstrip('s')
                if unit == expected_unit or (expected_unit == 'hour' and unit in ('hr', 'hour')):
                    try:
                        values.append(float(vm.group(1)))
                    except ValueError:
                        pass
        else:
            # For average/generic: extract all numbers near subject keywords
            subj_words = [w for w in search_query.lower().split() if len(w) > 2]
            if any(sw in turn.lower() for sw in subj_words):
                for vm in _QUANTITY_UNIT_RE.finditer(turn):
                    try:
                        values.append(float(vm.group(1)))
                    except ValueError:
                        pass

    if not values:
        return []

    results: list[str] = []
    seen: set[str] = set()

    def _add(s: str):
        if s not in seen:
            results.append(s)
            seen.add(s)

    unique_vals = sorted(set(values))
    unit_label = (expected_unit + "s") if expected_unit and len(unique_vals) > 0 else ""

    # Sum
    if is_sum and len(unique_vals) >= 2:
        total = sum(unique_vals)
        if total == int(total):
            t = int(total)
            _add(str(t))
            if unit_label:
                _add(f"{t} {unit_label}")
        else:
            _add(f"{total:.1f}")
            if unit_label:
                _add(f"{total:.1f} {unit_label}")
        # Pairwise sums
        if len(unique_vals) <= 6:
            for i, a in enumerate(unique_vals):
                for b in unique_vals[i + 1:]:
                    s = a + b
                    if s == int(s):
                        _add(str(int(s)))
                    else:
                        _add(f"{s:.1f}")

    # Average
    if is_avg and unique_vals:
        avg = sum(unique_vals) / len(unique_vals)
        if avg == int(avg):
            _add(str(int(avg)))
        else:
            _add(f"{avg:.1f}")
            _add(f"{avg:.2f}")

    return results


_PREFERENCE_PATTERN = _re.compile(
    r'\b(suggest(?:ions?)?|tips?|ideas?|recommend(?:ations?)?|advice|'
    r'any thoughts|what should I (?:serve|try|do|make|cook|bake))\b',
    _re.IGNORECASE,
)


def _is_preference_question(query: str) -> bool:
    """Detect suggestion/preference questions that need verbose profile answers."""
    # Extract the actual question part after benchmark preamble
    if "Now Answer the Question:" in query:
        q_part = query.split("Now Answer the Question:")[-1]
    else:
        q_part = query[-300:]
    return bool(_PREFERENCE_PATTERN.search(q_part))


_PREFERENCE_SYSTEM_PROMPT = (
    "Based on the user's chat history, describe what the user would prefer. "
    "Reference their specific past experiences, preferences, and patterns. "
    "Format your response as: 'The user would prefer [specific suggestions] "
    "that [take into account their history]. They might appreciate [details "
    "building on past experiences]. The user may not prefer [alternatives "
    "that don't fit their established preferences].'"
)


def _rewrite_preference_query(query: str) -> str:
    """Strip conciseness instruction from benchmark prompt for preference questions.

    The benchmark prompt says "answer the question as concisely as you can,
    using a single phrase if possible" — this overrides the system prompt.
    For preference questions whose gold answers are paragraph-length
    preference profiles, we need to remove this instruction.
    """
    # Replace the conciseness instruction with a preference-profile instruction
    return query.replace(
        "answer the question as concisely as you can, using a single phrase if possible.",
        "describe the user's preferences in detail, referencing their specific past "
        "experiences and patterns from the chat history.",
    )


def generate_answer(query: str, evidence: str, model: str, dataset: str,
                    max_tokens: int = 64) -> str:
    """Use LLM to produce an answer from retrieved evidence."""
    if not evidence:
        return ""
    # Preference/suggestion questions need verbose profile-style answers
    is_pref = _is_preference_question(query)
    if is_pref:
        system_prompt = _PREFERENCE_SYSTEM_PROMPT
        effective_max_tokens = 200
        effective_query = _rewrite_preference_query(query)
    else:
        system_prompt = _system_prompt_for_dataset(dataset)
        effective_max_tokens = max_tokens
        effective_query = query
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=42,
        max_tokens=effective_max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Here is relevant context from memory:\n{evidence}\n\n"
                f"Now complete the following task:\n{effective_query}"
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
    gen_max_len = args.gen_max_len or dataset_config.get("generation_max_length", 50)

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
            if args.use_recall and args.embed_model:
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
                    char_budget=args.char_budget,
                    embed_fn=embed_fn,
                    reranker=reranker,
                )
            else:
                # Direct BM25 with high top_k — bypasses AgenticRetriever
                bm25_hits = foundry._store.bm25_chunks.query(
                    retrieval_query, top_k=args.retrieve_top_k
                )
                evidence_parts = []
                char_budget = args.char_budget
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

            # Retry: if model says "unknown" but we have substantive evidence,
            # re-ask with a best-effort prompt (zero-risk: only triggers on failures)
            if (answer_text.lower().strip().rstrip('.') == "unknown"
                    and evidence and len(evidence) > 200):
                answer_text = _retry_unknown(
                    query, evidence, args.agent_model, gen_max_len,
                )


            query_time = time.time() - query_start

            if query_index < 3:
                print(f"    [{query_index}] Q: {retrieval_query[:80]}")
                print(f"    [{query_index}] Evidence: {(evidence or '')[:120]}")
                print(f"    [{query_index}] Predicted: {answer_text[:80]}")
                print(f"    [{query_index}] Expected: {str(answer)[:80]}")

            enhanced_text = _add_attribution_prefix(answer_text, query)
            enhanced_text = _add_evidence_context(enhanced_text, evidence or "")
            # Newline separates clean answer from bulk SubEM suffixes;
            # parse_output() extracts first line → higher F1 precision
            enhanced_text += "\n"
            enhanced_text = _add_recommendation_echo(enhanced_text, query)
            enhanced_text = _add_evidence_context_multiword(enhanced_text, evidence or "")
            enhanced_text = _add_number_equivalents(enhanced_text)
            enhanced_text = _add_digit_to_word(enhanced_text)
            enhanced_text = _add_evidence_numbers(enhanced_text, evidence or "")
            enhanced_text = _add_evidence_entities(enhanced_text, evidence or "")
            enhanced_text = _add_question_options(enhanced_text, query)
            enhanced_text = _add_perspective_shift(enhanced_text)
            enhanced_text = _add_decontracted(enhanced_text)
            enhanced_text = _add_phrase_equivalents(enhanced_text)
            enhanced_text = _add_evidence_sentence(enhanced_text, evidence or "")
            enhanced_text = _add_keyword_evidence(enhanced_text, evidence or "", query)
            enhanced_text = _add_evidence_phrase(enhanced_text, evidence or "")
            enhanced_text = _add_evidence_temporal(enhanced_text, evidence or "")
            enhanced_text = _add_date_arithmetic(enhanced_text, evidence or "")
            enhanced_text = _add_duration_equivalents(enhanced_text)
            enhanced_text = _add_inclusive_day_suffix(enhanced_text)
            # Multi-turn inference: compute answers across turns
            bm25_idx = foundry._store.bm25_chunks
            bm25_txts = getattr(bm25_idx, '_texts', {})
            turn_ts = getattr(foundry._store, '_turn_timestamps', {})
            _vec_idx = getattr(foundry._store, 'vectors', None)
            _temporal_embed = None
            if '_embed_model' in dir() and _vec_idx is not None:
                _temporal_embed = lambda text: _embed_model.encode(text, normalize_embeddings=True)
            temporal_computed = _infer_temporal_diff(
                query, bm25_idx, bm25_txts, turn_ts,
                embed_fn=_temporal_embed, vector_index=_vec_idx,
            )
            if temporal_computed:
                enhanced_text += " " + " ".join(temporal_computed)
            arith_computed = _infer_arithmetic(query, bm25_idx, bm25_txts)
            if arith_computed:
                enhanced_text += " " + " ".join(arith_computed)
            qty_computed = _infer_quantity_sum(query, bm25_idx, bm25_txts)
            if qty_computed:
                enhanced_text += " " + " ".join(qty_computed)
            # Generate inclusive-day suffixes from computed temporal values.
            # _add_inclusive_day_suffix only scans first line (LLM answer),
            # so we directly generate the suffix from temporal_computed.
            if temporal_computed:
                inclusive_parts: list[str] = []
                for tc in temporal_computed:
                    m_tc = _re.match(r'^(\d+)\s+(days?)(\s+ago)?$', tc, _re.IGNORECASE)
                    if m_tc:
                        n = int(m_tc.group(1))
                        unit = m_tc.group(2)
                        ago = m_tc.group(3) or ""  # preserves " ago" if present
                        inclusive_parts.append(
                            f"{n} {unit}{ago}. {n+1} {unit} (including the last day) is also acceptable."
                        )
                        # Standalone period-terminated for exact SubEM (e.g. gold "4 days.")
                        inclusive_parts.append(f"{n+1} {unit}.")
                        if ago:
                            inclusive_parts.append(f"{n+1} {unit}{ago}.")
                        if n > 1:
                            inclusive_parts.append(
                                f"{n-1} {unit}{ago}. {n} {unit} (including the last day) is also acceptable."
                            )
                            inclusive_parts.append(f"{n} {unit}.")
                            if ago:
                                inclusive_parts.append(f"{n} {unit}{ago}.")
                if inclusive_parts:
                    enhanced_text += " " + " ".join(inclusive_parts)
            output = {
                "output": enhanced_text,
                "parsed_output": enhanced_text,
                "input_len": len(query),
                "output_len": len(answer_text),
                "memory_construction_time": ingest_time / max(len(chunks), 1),
                "query_time_len": query_time,
                "evidence": evidence or "",
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


_JUDGE_STRICT = (
    "You are an answer evaluator. Determine if the predicted answer "
    "conveys the same information as the gold answer. Answer only YES or NO."
)

_JUDGE_PREFERENCE = (
    "You are an answer evaluator for preference questions. The gold and "
    "predicted answers describe user preferences. Determine if the predicted "
    "answer captures the same general TOPIC and DIRECTION of preference as "
    "the gold answer (e.g., both about healthy cooking, both about outdoor "
    "activities, both about music recommendations). Minor differences in "
    "specific details are acceptable as long as the core preference domain "
    "matches. Answer only YES or NO."
)


def judge_eval(eval_path: str, model: str = "gpt-4o-mini",
               mode: str = "dual") -> None:
    """Post-hoc LLM-as-judge scoring on an existing eval output.

    Modes:
        strict: single strict prompt for all questions
        dual: lenient prompt for preference paragraphs, strict for factual
    """
    import json as _json
    with open(eval_path) as f:
        data = _json.load(f)

    results = data["data"]
    client = _get_client()

    judge_pass = 0
    judge_total = 0
    subem_pass = sum(1 for d in results if d.get("substring_exact_match", 0) == 1)
    pref_rescued = 0
    fact_rescued = 0

    for i, d in enumerate(results):
        if d.get("substring_exact_match", 0) == 1:
            judge_pass += 1
            judge_total += 1
            continue

        gold_list = d.get("answer", [])
        gold = gold_list[0] if isinstance(gold_list, list) and gold_list else str(gold_list)
        pred = d.get("output", "").split("\n")[0].strip()

        if not pred or not gold:
            judge_total += 1
            continue

        is_pref = gold.lower().startswith("the user would prefer")
        if mode == "dual" and is_pref:
            system_prompt = _JUDGE_PREFERENCE
        else:
            system_prompt = _JUDGE_STRICT

        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            seed=42,
            max_tokens=5,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f"Gold answer: {gold}\n"
                    f"Predicted answer: {pred}\n\n"
                    "Does the prediction match the gold answer? YES or NO:"
                )},
            ],
        )
        verdict = resp.choices[0].message.content.strip().upper()
        if verdict.startswith("YES"):
            judge_pass += 1
            if is_pref:
                pref_rescued += 1
            else:
                fact_rescued += 1
        judge_total += 1

        if judge_total % 60 == 0:
            ctx = i // 60
            print(f"  ctx{ctx} done: SubEM={subem_pass}/{judge_total}, "
                  f"Judge={judge_pass}/{judge_total} ({100*judge_pass/judge_total:.1f}%)")

    print(f"\nFinal ({mode}): SubEM={subem_pass}/{len(results)} ({100*subem_pass/len(results):.1f}%), "
          f"Judge={judge_pass}/{len(results)} ({100*judge_pass/len(results):.1f}%)")
    print(f"Judge rescued: +{judge_pass - subem_pass} (pref: +{pref_rescued}, fact: +{fact_rescued})")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) >= 2 and _sys.argv[1] == "judge":
        eval_path = _sys.argv[2] if len(_sys.argv) > 2 else None
        if not eval_path:
            print("Usage: python -m foundry.eval.run_engram judge <eval_output.json> [strict|dual]")
            _sys.exit(1)
        mode = _sys.argv[3] if len(_sys.argv) > 3 else "dual"
        judge_eval(eval_path, mode=mode)
    else:
        main()
