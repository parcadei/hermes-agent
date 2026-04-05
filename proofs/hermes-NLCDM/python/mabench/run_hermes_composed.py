#!/usr/bin/env python3
"""Run MemoryAgentBench evaluation with the composed HermesAgent.

Uses the modular agent (mabench.agent.HermesAgent) with frozen config
dataclasses instead of the monolithic HermesMemoryAgent.

Usage (from hermes-agent root):
    python3 -u -m mabench.run_hermes_composed \
        --dataset_config MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_mh_262k.yaml \
        --cooc_boost --triadic --decompose_query --conflict_resolution \
        --cr_per_hop --bm25_hops --chunk_size 512
"""

import json
import os
import pickle
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import sys
import dotenv
import numpy as np
import yaml
from tqdm import tqdm

# Add MABench and hermes paths
_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
_MABENCH = _HERMES_ROOT / "MemoryAgentBench"

sys.path.insert(0, str(_MABENCH))
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))

from conversation_creator import ConversationCreator
from dream_ops import DreamParams
from mabench.agent import HermesAgent
from mabench.configs import (
    RetrievalConfig, CRConfig, DecomposeConfig, StoreConfig,
    LLMConfig, BeliefConfig, EntityConfig, DedupConfig, KGConfig,
    GLiNER2Config, CoocExtractionConfig, RelikConfig, StructuredStoreConfig,
)
from mabench.eval_cache import cache_dir, list_caches
from utils.eval_other_utils import metrics_summarization

dotenv.load_dotenv(_NLCDM_PYTHON / ".env")


# ---------------------------------------------------------------------------
# Ingestion cache adapted for HermesAgent attribute layout
# ---------------------------------------------------------------------------

def _save_context(agent: HermesAgent, path: Path, context_idx: int) -> None:
    """Save post-ingestion state for one context."""
    ctx_dir = path / f"context_{context_idx}"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    ss = agent._supersession_state
    ei = agent._entity_index

    state = {
        "coupled_engine": agent.engine,
        "store_count": agent.store._store_count,
        "triple_index": dict(ss.triple_index),
        "superseded_texts": ss.superseded_texts,
        "fact_entities": ss.fact_entities,
        "supersession_graph": ss.supersession_graph,
        "supersedes": {k: set(v) for k, v in ss.supersedes.items()},
        "subject_index": dict(ss.subject_index),
        "gliner_entity_index": {k: set(v) for k, v in ei.index.items()},
    }

    tri = agent.store._triadic
    if tri is not None:
        state["triadic"] = tri
    bel = agent.store._belief_index
    if bel is not None:
        state["belief_index"] = bel
    ea = agent.store._entity_accumulator
    if ea is not None:
        state["entity_accumulator"] = ea
    bm25 = agent.store._bm25_index
    if bm25 is not None:
        state["bm25_index"] = bm25
    rebel = agent.rebel
    if rebel is not None:
        state["rebel_kg"] = getattr(rebel, "_kg", None) or getattr(rebel, "_rebel_kg", None)
        state["rebel_kg_reverse"] = getattr(rebel, "_kg_reverse", None) or getattr(rebel, "_rebel_kg_reverse", None)
    cc = agent.store._chain_cache
    if cc is not None:
        state["chain_cache"] = cc
    kg = agent.knowledge_graph
    if kg is not None:
        state["knowledge_graph"] = kg
    he = agent.hyperedge
    if he is not None:
        state["hyperedge"] = he
    g2 = agent.gliner2_extractor
    if g2 is not None:
        state["gliner2_extractor"] = g2
    cooc_ex = agent.cooc_extractor
    if cooc_ex is not None:
        state["cooc_extractor"] = cooc_ex
    relik_ex = agent.relik_extractor
    if relik_ex is not None:
        state["relik_extractor"] = relik_ex
    cooc_kg = agent.cooc_kg
    if cooc_kg is not None:
        state["cooc_kg"] = cooc_kg
    struct_store = getattr(agent, "_structured_store", None)
    if struct_store is not None:
        state["structured_store"] = struct_store

    state_file = ctx_dir / "state.pkl"
    with open(state_file, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_memories = len(agent.engine.memory_store)
    meta = {
        "n_memories": n_memories,
        "store_count": agent.store._store_count,
        "has_triadic": "triadic" in state,
        "has_belief": "belief_index" in state,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(ctx_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = state_file.stat().st_size / (1024 * 1024)
    print(f"Saved ingestion cache: {state_file} ({size_mb:.1f} MB, {n_memories} memories)")


def _load_context(agent: HermesAgent, path: Path, context_idx: int) -> bool:
    """Restore agent state from ingestion cache. Returns True if loaded."""
    state_file = path / f"context_{context_idx}" / "state.pkl"
    if not state_file.exists():
        return False

    t0 = time.time()
    with open(state_file, "rb") as f:
        state = pickle.load(f)

    # Core memory state
    agent.engine = state["coupled_engine"]
    agent.store._store_count = state["store_count"]

    # Supersession state
    ss = agent._supersession_state
    ss.triple_index = defaultdict(list, state.get("triple_index", {}))
    ss.superseded_texts = state.get("superseded_texts", set())
    ss.fact_entities = state.get("fact_entities", {})
    ss.supersession_graph = state.get("supersession_graph", {})
    raw = state.get("supersedes", {})
    ss.supersedes = defaultdict(set, {k: set(v) for k, v in raw.items()})
    ss.subject_index = state.get("subject_index", {})

    # Entity index
    raw_idx = state.get("gliner_entity_index", {})
    agent._entity_index.index = defaultdict(set, {k: set(v) for k, v in raw_idx.items()})

    # Optional infrastructure
    if "triadic" in state and agent.store._triadic is not None:
        agent.store._triadic = state["triadic"]
        agent.query_engine._triadic = state["triadic"]
        agent.pipeline._triadic = state["triadic"]
    if "belief_index" in state and agent.store._belief_index is not None:
        agent.store._belief_index = state["belief_index"]
        agent.pipeline._belief_index = state["belief_index"]
        agent.retriever._belief_index = state["belief_index"]
    if "entity_accumulator" in state and agent.store._entity_accumulator is not None:
        agent.store._entity_accumulator = state["entity_accumulator"]

    # BM25: always rebuild from memory_store (doc_id consistency)
    if agent.store._bm25_index is not None:
        from mabench.bm25_index import BM25Index
        fresh_bm25 = BM25Index()
        fresh_bm25.add_batch([
            (i, m.text) for i, m in enumerate(agent.engine.memory_store)
        ])
        agent.store._bm25_index = fresh_bm25
        agent.query_engine._bm25_index = fresh_bm25
        agent.pipeline._bm25_index = fresh_bm25

    # Entity inverted indexes: rebuild from memory_store (like BM25)
    from mabench.entity_index import (
        extract_entities, extract_bigrams, extract_content_unigrams,
    )
    entity_inv: dict[str, set[int]] = {}
    bigram_inv: dict[str, set[int]] = {}
    unigram_inv: dict[str, set[int]] = {}
    for i, m in enumerate(agent.engine.memory_store):
        for ent in extract_entities(m.text):
            entity_inv.setdefault(ent, set()).add(i)
        for bg in extract_bigrams(m.text):
            bigram_inv.setdefault(bg, set()).add(i)
        for ug in extract_content_unigrams(m.text):
            unigram_inv.setdefault(ug, set()).add(i)
    agent.store._entity_inv_idx = entity_inv
    agent.store._bigram_inv_idx = bigram_inv
    agent.store._unigram_inv_idx = unigram_inv
    # Thread to retriever and pipeline for query-time access
    agent.retriever._entity_inv_idx = entity_inv
    agent.retriever._bigram_inv_idx = bigram_inv
    agent.retriever._unigram_inv_idx = unigram_inv
    agent.pipeline._entity_inv_idx = entity_inv
    agent.pipeline._bigram_inv_idx = bigram_inv
    agent.pipeline._unigram_inv_idx = unigram_inv

    # Event calendar: rebuild from memory_store
    from mabench.event_calendar import (
        parse_turn_date, extract_event_keywords, extract_inline_dates,
    )
    event_calendar: list[dict] = []
    for i, m in enumerate(agent.engine.memory_store):
        turn_date = parse_turn_date(m.text)
        if turn_date is not None:
            keywords = extract_event_keywords(m.text)
            inline_dates = extract_inline_dates(m.text, turn_date)
            event_calendar.append({
                "turn_idx": i,
                "date": turn_date,
                "keywords": keywords,
                "inline_dates": inline_dates,
                "text": m.text,
            })
    agent.store._event_calendar = event_calendar
    agent.pipeline._event_calendar = event_calendar

    # Chain cache
    if agent.store._chain_cache is not None and "chain_cache" in state:
        agent.store._chain_cache = state["chain_cache"]
        agent.query_engine._chain_cache = state["chain_cache"]
        agent.pipeline._chain_cache = state["chain_cache"]

    # Knowledge graph
    if agent.store._knowledge_graph is not None and "knowledge_graph" in state:
        agent.store._knowledge_graph = state["knowledge_graph"]
        agent.query_engine._knowledge_graph = state["knowledge_graph"]
        agent.pipeline._knowledge_graph = state["knowledge_graph"]
        agent.knowledge_graph = state["knowledge_graph"]

    # Hyperedge intersection
    if agent.hyperedge is not None and "hyperedge" in state:
        agent.query_engine._hyperedge = state["hyperedge"]
        agent.pipeline._hyperedge = state["hyperedge"]
        agent.hyperedge = state["hyperedge"]

    # GLiNER2 extractor (pickle-safe: model excluded, lazy-reloaded)
    if agent.store._gliner2_extractor is not None and "gliner2_extractor" in state:
        agent.store._gliner2_extractor = state["gliner2_extractor"]
        agent.gliner2_extractor = state["gliner2_extractor"]

    # Co-occurrence extractor (pickle-safe: model excluded, lazy-reloaded)
    if agent.store._cooc_extractor is not None and "cooc_extractor" in state:
        agent.store._cooc_extractor = state["cooc_extractor"]
        agent.cooc_extractor = state["cooc_extractor"]

    # ReLiK extractor (pickle-safe: model excluded, lazy-reloaded)
    if agent.store._relik_extractor is not None and "relik_extractor" in state:
        agent.store._relik_extractor = state["relik_extractor"]
        agent.relik_extractor = state["relik_extractor"]

    # Structured fact store (pickle-safe: pure Python dicts + lists)
    if "structured_store" in state and getattr(agent, "_structured_store", None) is not None:
        agent._structured_store = state["structured_store"]
        agent.retriever._structured_store = state["structured_store"]

    # CoocKG (pickle-safe: pure Python dicts)
    if "cooc_kg" in state and agent.store._cooc_kg is not None:
        agent.store._cooc_kg = state["cooc_kg"]
        agent.query_engine._cooc_kg = state["cooc_kg"]
        agent.cooc_kg = state["cooc_kg"]

    # SupersessionState (holds _triple_index for query-time KG resolution)
    agent.query_engine._supersession_state = agent._supersession_state

    # Reload embedding disk cache
    agent._scorer._load_disk_cache()

    # Update engine references on all modules that hold one
    for mod in (agent.store, agent.query_engine, agent.pipeline,
                agent.retriever, agent.conflict_resolver, agent.entity_ops,
                agent.iterative):
        if hasattr(mod, "_engine"):
            mod._engine = agent.engine

    # Refresh structured_agent references (it holds memory_store + indexes
    # directly, not via _engine). Only if the agent was constructed.
    if getattr(agent, "structured_agent", None) is not None:
        agent.structured_agent._memory_store = agent.engine.memory_store
        agent.structured_agent._entity_inv_idx = agent.store._entity_inv_idx
        agent.structured_agent._event_calendar = agent.store._event_calendar
        agent.structured_agent._bm25 = getattr(agent.store, "_bm25_index", None)
        agent.structured_agent._engine = agent.engine
        agent.structured_agent._scorer = agent._scorer
        agent.structured_agent._date_index_len = -1  # force rebuild

    n = len(agent.engine.memory_store)
    elapsed = time.time() - t0
    print(f"Loaded ingestion cache: {n} memories in {elapsed:.1f}s from {state_file}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(description="MABench eval with composed HermesAgent")
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--retrieve_num", type=int, default=10)
    parser.add_argument("--contradiction_threshold", type=float, default=0.95)
    parser.add_argument("--dream_interval", type=int, default=0)
    parser.add_argument("--max_test_queries", type=int, default=0)
    parser.add_argument("--context_idx", type=int, default=None,
                        help="Run only this context index (0-based). For parallel eval.")
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--max_gen_tokens", type=int, default=0,
                        help="Max LLM generation tokens (0=auto: 64 for FC, 128 for LME/EventQA)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM generation temperature (0.0=deterministic, framework agent uses 0.7)")
    parser.add_argument("--recency_alpha", type=float, default=0.1)

    # Dream params
    parser.add_argument("--dream_eta", type=float, default=0.01)
    parser.add_argument("--dream_min_sep", type=float, default=0.3)
    parser.add_argument("--dream_prune_threshold", type=float, default=0.95)
    parser.add_argument("--dream_merge_threshold", type=float, default=0.90)
    parser.add_argument("--dream_explore_only", action="store_true")

    # Store
    parser.add_argument("--supersession", action="store_true",
                        help="Enable (S,P) triple-based supersession at ingestion time")

    # Retrieval
    parser.add_argument("--cooc_boost", action="store_true")
    parser.add_argument("--cooc_weight", type=float, default=1.0)
    parser.add_argument("--cooc_gate", type=float, default=0.0)
    parser.add_argument("--ppr", action="store_true")
    parser.add_argument("--ppr_weight", type=float, default=0.3)
    parser.add_argument("--ppr_damping", type=float, default=0.85)
    parser.add_argument("--triadic", action="store_true")
    parser.add_argument("--triadic_chain_hops", type=int, default=1,
                        help="Triadic SA chain-hop depth (1=single-hop, 3=recommended)")
    parser.add_argument("--triadic_decay", type=float, default=1.0,
                        help="Triadic SA decay factor per hop (0<d<=1, 1.0=no decay)")
    parser.add_argument("--bm25_hops", action="store_true")
    parser.add_argument("--chain_cache", action="store_true")
    parser.add_argument("--rebel_kg", action="store_true")

    # Query decomposition
    parser.add_argument("--decompose_query", action="store_true")
    parser.add_argument("--decompose_max_hops", type=int, default=3)

    # Conflict resolution
    parser.add_argument("--conflict_resolution", action="store_true")
    parser.add_argument("--cr_per_hop", action="store_true")
    parser.add_argument("--cr_entity_gate", action="store_true")
    parser.add_argument("--cr_overfetch", type=int, default=3)

    # Dedup
    parser.add_argument("--cosine_dedup", action="store_true")
    parser.add_argument("--cosine_dedup_threshold", type=float, default=0.95)

    # Entity
    parser.add_argument("--entity_retrieval", action="store_true")
    parser.add_argument("--entity_first", action="store_true")
    parser.add_argument("--entity_first_aggressive", action="store_true")
    parser.add_argument("--entity_rerank", action="store_true",
                        help="Entity-position residual reranking with multiplicative gating (arXiv:2603.19664)")

    # Cross-encoder reranking
    parser.add_argument("--cross_encoder", action="store_true",
                        help="Enable cross-encoder reranking (ms-marco-MiniLM-L-6-v2, 22M params)")
    parser.add_argument("--cross_encoder_top_k", type=int, default=10,
                        help="Number of top PPR nodes to retrieve")

    # BM25 primary
    parser.add_argument("--bm25_primary", action="store_true",
                        help="Use BM25 as primary retrieval channel (58.3%% vs 36.7%% cosine on LME)")
    # BM25 rerank
    parser.add_argument("--bm25_rerank", action="store_true",
                        help="Blend cosine+BM25 scores on candidates (100%% hit@10 in disc test)")
    parser.add_argument("--bm25_rerank_weight", type=float, default=0.5,
                        help="BM25 contribution in rerank blend (default 0.5)")

    # Entity primary retrieval (regex-based inverted index)
    parser.add_argument("--entity_primary", action="store_true",
                        help="Use entity/bigram/unigram inverted index as primary retrieval (72.2%% oracle match)")
    parser.add_argument("--entity_prefilter", action="store_true",
                        help="RRF fusion of BM25 + entity_scored (60%% hit@10, ties xenc 15x faster). Requires --bm25_primary.")

    # Fisher-Rao scoring
    parser.add_argument("--fisher_scoring", action="store_true",
                        help="Replace cosine base similarity with Fisher-Rao inverse-variance weighted distance (arXiv:2603.14588v1)")
    parser.add_argument("--fisher_temperature", type=float, default=1.0,
                        help="Fisher-Rao temperature (lower = more discriminative, default 1.0)")

    # Role weighting (conversation mode)
    parser.add_argument("--role_weight_assistant", type=float, default=1.0,
                        help="Retrieval score multiplier for assistant-turn facts (default 1.0)")
    parser.add_argument("--role_weight_continuation", type=float, default=1.0,
                        help="Retrieval score multiplier for continuation-turn facts (default 1.0)")

    # Temporal proximity scoring
    parser.add_argument("--temporal_alpha", type=float, default=0.0,
                        help="Temporal proximity decay rate (0=disabled, >0 enables temporal scoring + fast-lane)")

    # Hyperedge intersection
    parser.add_argument("--hyperedge", action="store_true",
                        help="Enable hyperedge intersection in decompose hops")
    parser.add_argument("--hyperedge_penalty", type=float, default=0.1,
                        help="Penalty for non-matching facts (0.0=hard filter, 1.0=no effect)")
    parser.add_argument("--hyperedge_min_candidates", type=int, default=3,
                        help="Minimum entity matches to activate penalty")

    # Knowledge graph (rustworkx PPR)
    parser.add_argument("--knowledge_graph", action="store_true",
                        help="Enable rustworkx KG + PPR retrieval")
    parser.add_argument("--kg_damping", type=float, default=0.85,
                        help="PPR damping factor (0=short hops, 1=deep exploration)")
    parser.add_argument("--kg_ppr_top_k", type=int, default=20,
                        help="Number of top PPR nodes to retrieve")

    # GLiNER2 triple extraction
    parser.add_argument("--gliner2", action="store_true",
                        help="Enable GLiNER2 dense triple extraction for KG")
    parser.add_argument("--gliner2_threshold", type=float, default=0.5,
                        help="GLiNER2 confidence threshold (0.3=broad, 0.7=precise)")
    parser.add_argument("--gliner2_model", type=str,
                        default="fastino/gliner2-base-v1",
                        help="GLiNER2 model name")

    # Co-occurrence extraction (fast path: bi-encoder NER → co-occurrence edges)
    parser.add_argument("--cooc_extraction", action="store_true",
                        help="Enable fast-path co-occurrence extraction for KG + triadic")
    parser.add_argument("--cooc_extraction_threshold", type=float, default=0.3,
                        help="NER confidence threshold for co-occurrence extractor")
    parser.add_argument("--cooc_extraction_model", type=str,
                        default="knowledgator/gliner-bi-edge-v2.0",
                        help="GLiNER bi-encoder NER model for co-occurrence extraction")

    # ReLiK v2 relation extraction (canonical Wikidata triples for supersession)
    parser.add_argument("--relik", action="store_true",
                        help="Enable ReLiK v2 triple extraction for supersession")
    parser.add_argument("--relik_model", type=str,
                        default="relik-ie/relik-relation-extraction-large",
                        help="ReLiK model name")
    parser.add_argument("--relik_min_confidence", type=float, default=0.5,
                        help="ReLiK minimum confidence threshold")

    # LLM triple extraction (vLLM server)
    parser.add_argument("--llm_extraction_url", type=str, default=None,
                        help="vLLM server URL for LLM triple extraction (e.g. http://localhost:8000/v1)")
    parser.add_argument("--llm_extraction_model", type=str, default="qwen3-triple-extractor",
                        help="Served model name for LLM triple extraction")
    parser.add_argument("--llm_extraction_mode", type=str, default="factual",
                        choices=["factual", "conversation"],
                        help="Extraction mode: factual (template sentences) or conversation (alias-aware)")
    parser.add_argument("--coref_backend", type=str, default="fastcoref",
                        choices=["fastcoref", "maverick"],
                        help="Coreference resolution backend")

    # LFM SVO extraction (local LM Studio for LME entity index enrichment)
    parser.add_argument("--lfm_extraction_url", type=str, default=None,
                        help="LM Studio URL for LFM SVO extraction (e.g. http://127.0.0.1:1234/v1)")
    parser.add_argument("--svo_cache_path", type=str, default=None,
                        help="Path to pre-extracted SVO cache JSON (skips live LFM extraction)")

    # Structured fact store (SVO triple index as RRF retrieval channel)
    parser.add_argument("--structured_store", action="store_true",
                        help="Enable structured fact store (SVO triple index as 3-way RRF retrieval channel)")

    # Dynamic prompting (per-question-type LLM system prompts)
    parser.add_argument("--dynamic_prompting", action="store_true",
                        help="Use question-type-specific LLM system prompts (aggregation/temporal/update/ordering)")

    # Raw chunk mode: skip conversation parsing
    parser.add_argument("--raw_chunks", action="store_true",
                        help="Store whole chunks without conversation parsing (matches framework agent architecture)")

    # Event calendar retrieval: keyword-based retrieval as 3rd RRF channel
    parser.add_argument("--event_calendar_retrieval", action="store_true",
                        help="Add event calendar keyword overlap as 3rd RRF channel (requires --entity_prefilter)")

    # LME event calendar: post-score additive boost for turns in date-window
    parser.add_argument("--event_calendar", action="store_true",
                        help="LME date-mention calendar: boost turns whose extracted dates "
                             "fall in ±window_days around the query's Current Date (additive, bounded).")
    parser.add_argument("--event_calendar_window_days", type=int, default=60,
                        help="Half-window (±days) for event calendar lookup (default 60)")
    parser.add_argument("--event_calendar_boost", type=float, default=0.15,
                        help="Additive score boost for in-window candidates (default 0.15)")

    # Aggregation query overfetch: bump top_k for "how many"/"total"/"count" queries
    parser.add_argument("--aggregation_overfetch", type=int, default=0,
                        help="Target top_k for aggregation queries (how many/total/count/list all). 0=disabled. Try 20-25.")

    # Structured event-DB tool-calling query (Chronos-style)
    parser.add_argument("--structured_query", action="store_true",
                        help="Use LLM tool-calling loop over entity/date/BM25 indexes (Chronos-style structured retrieval)")
    parser.add_argument("--structured_query_max_hops", type=int, default=3,
                        help="Max tool calls per question in structured query mode (default 3)")

    # Cache
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--cache_label", type=str, default=None)
    parser.add_argument("--list_caches", action="store_true")

    # Ground truth correction
    parser.add_argument("--corrected_answers", type=str, default=None,
                        help="JSON file with corrected ground truth answers (overrides dataset expected answers)")
    parser.add_argument("--patched_context", type=str, default=None,
                        help="Path to patched context file (replaces dataset context for ingestion)")

    return parser.parse_args()


def _build_agent(args):
    """Build HermesAgent from CLI args."""
    retrieval = RetrievalConfig(
        retrieve_num=args.retrieve_num,
        cooc_boost=args.cooc_boost,
        cooc_weight=args.cooc_weight,
        cooc_gate_threshold=args.cooc_gate,
        ppr=args.ppr,
        ppr_weight=args.ppr_weight,
        ppr_damping=args.ppr_damping,
        entity_rerank=args.entity_rerank,
        hyperedge_intersection=args.hyperedge,
        hyperedge_penalty=args.hyperedge_penalty,
        hyperedge_min_candidates=args.hyperedge_min_candidates,
        bm25_primary=args.bm25_primary,
        bm25_rerank=args.bm25_rerank,
        bm25_rerank_weight=args.bm25_rerank_weight,
        entity_primary=args.entity_primary,
        entity_prefilter=args.entity_prefilter,
        event_calendar_retrieval=args.event_calendar_retrieval,
        event_calendar=args.event_calendar,
        event_calendar_window_days=args.event_calendar_window_days,
        event_calendar_boost=args.event_calendar_boost,
        aggregation_overfetch=args.aggregation_overfetch,
        fisher_scoring=args.fisher_scoring,
        fisher_temperature=args.fisher_temperature,
        cross_encoder=args.cross_encoder,
        cross_encoder_top_k=args.cross_encoder_top_k,
        role_weight_assistant=args.role_weight_assistant,
        role_weight_continuation=args.role_weight_continuation,
        temporal_alpha=args.temporal_alpha,
    )
    cr = CRConfig(
        enabled=args.conflict_resolution,
        per_hop=args.cr_per_hop,
        entity_gate=args.cr_entity_gate,
        overfetch=args.cr_overfetch,
    )
    decompose = DecomposeConfig(
        enabled=args.decompose_query,
        max_hops=args.decompose_max_hops,
    )
    # Auto-enable supersession when relik is active (triples need triple_index)
    supersession = getattr(args, "supersession", False) or args.relik
    store = StoreConfig(
        supersession=supersession,
        cooc_boost=args.cooc_boost,
        dream_interval=args.dream_interval,
    )
    # Auto-detect max_gen_tokens: 64 for FC (short entity answers),
    # 128 for LME/EventQA (longer conversational answers).
    if args.max_gen_tokens > 0:
        max_gen_tokens = args.max_gen_tokens
    elif "factconsolidation" in args.dataset_config.lower():
        max_gen_tokens = 64
    else:
        max_gen_tokens = 128

    llm_cfg = LLMConfig(
        model=args.model,
        temperature=args.temperature,
        max_gen_tokens=max_gen_tokens,
    )
    entity = EntityConfig(
        entity_retrieval=args.entity_retrieval,
        entity_first=args.entity_first,
        entity_first_aggressive=args.entity_first_aggressive,
    )
    dedup = DedupConfig(
        enabled=args.cosine_dedup,
        threshold=args.cosine_dedup_threshold,
    )
    kg_cfg = KGConfig(
        enabled=args.knowledge_graph,
        damping=args.kg_damping,
        ppr_top_k=args.kg_ppr_top_k,
    )
    gliner2_cfg = GLiNER2Config(
        enabled=args.gliner2,
        model_name=args.gliner2_model,
        threshold=args.gliner2_threshold,
    )
    cooc_extraction_cfg = CoocExtractionConfig(
        enabled=args.cooc_extraction,
        model_name=args.cooc_extraction_model,
        threshold=args.cooc_extraction_threshold,
    )
    relik_cfg = RelikConfig(
        enabled=args.relik,
        model_name=args.relik_model,
        min_confidence=args.relik_min_confidence,
    )

    structured_store_cfg = StructuredStoreConfig(
        enabled=args.structured_store,
        svo_cache_path=args.svo_cache_path if args.structured_store else None,
    )

    dream_params = DreamParams(
        eta=args.dream_eta,
        min_sep=args.dream_min_sep,
        prune_threshold=args.dream_prune_threshold,
        merge_threshold=args.dream_merge_threshold,
    )
    belief = BeliefConfig(
        enabled=False,  # not exposed via CLI yet
    )

    return HermesAgent(
        dim=768,
        contradiction_threshold=args.contradiction_threshold,
        recency_alpha=args.recency_alpha,
        dream_params=dream_params,
        belief=belief,
        retrieval=retrieval,
        cr=cr,
        decompose=decompose,
        store=store,
        llm=llm_cfg,
        entity=entity,
        dedup=dedup,
        kg=kg_cfg,
        gliner2=gliner2_cfg,
        cooc_extraction=cooc_extraction_cfg,
        relik=relik_cfg,
        bm25_hops=args.bm25_hops,
        chain_cache=args.chain_cache,
        rebel_kg=args.rebel_kg,
        triadic=args.triadic,
        triadic_chain_hops=args.triadic_chain_hops,
        triadic_decay=args.triadic_decay,
        llm_extraction_base_url=args.llm_extraction_url,
        llm_extraction_model=args.llm_extraction_model,
        llm_extraction_mode=args.llm_extraction_mode,
        coref_backend=args.coref_backend,
        dynamic_prompting=args.dynamic_prompting,
        raw_chunks=args.raw_chunks,
        lfm_extraction_url=args.lfm_extraction_url,
        svo_cache_path=args.svo_cache_path,
        structured_store=structured_store_cfg,
        structured_query=args.structured_query,
        structured_query_max_hops=args.structured_query_max_hops,
    )


def main():
    args = parse_args()

    if args.list_caches:
        for c in list_caches():
            print(f"  {c['name']}:")
            for ctx in c["contexts"]:
                print(f"    {ctx['context']}: {ctx['n_memories']} memories, "
                      f"{ctx['size_mb']} MB, saved {ctx['saved_at']}")
        return

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
        "temperature": args.temperature,
        "output_dir": "output/hermes_composed",
        "retrieve_num": args.retrieve_num,
    }

    output_dir = Path(agent_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")

    # Build config tag for unique filenames
    config_parts = []
    if args.cooc_boost:
        config_parts.append("cooc")
    if args.decompose_query:
        config_parts.append("decomp")
    if args.conflict_resolution:
        config_parts.append("cr")
    if args.cr_per_hop:
        config_parts.append("perhop")
    if args.bm25_hops:
        config_parts.append("bm25h")
    if args.triadic:
        config_parts.append("tri")
        if args.triadic_chain_hops > 1:
            config_parts.append(f"sa{args.triadic_chain_hops}d{args.triadic_decay}")
    if args.entity_primary:
        config_parts.append("entpri")
    if args.entity_prefilter:
        config_parts.append("entpf")
    if args.entity_rerank:
        config_parts.append("entrank")
    if args.cross_encoder:
        config_parts.append("xenc")
    if args.fisher_scoring:
        config_parts.append("fisher")
    if args.role_weight_assistant != 1.0 or args.role_weight_continuation != 1.0:
        config_parts.append(f"role_a{args.role_weight_assistant}_c{args.role_weight_continuation}")
    if args.hyperedge:
        config_parts.append("hedge")
    if args.knowledge_graph:
        config_parts.append("kg")
    if args.gliner2:
        config_parts.append("gliner2")
    if args.cooc_extraction:
        config_parts.append("coocex")
    if args.relik:
        config_parts.append("relik")
    if args.dynamic_prompting:
        config_parts.append("dynprompt")
    if args.raw_chunks:
        config_parts.append("rawchunk")
    if args.event_calendar_retrieval:
        config_parts.append("evcal")
    if args.event_calendar:
        config_parts.append(
            f"evcalLME_w{args.event_calendar_window_days}b{args.event_calendar_boost}"
        )
    if args.aggregation_overfetch > 0:
        config_parts.append(f"aggof{args.aggregation_overfetch}")
    if args.structured_query:
        config_parts.append(f"struct_h{args.structured_query_max_hops}")
    if args.lfm_extraction_url:
        config_parts.append("lfm")
    if args.svo_cache_path:
        config_parts.append("svocache")
    if args.structured_store:
        config_parts.append("structstore")
    if args.temperature > 0:
        config_parts.append(f"t{args.temperature}")
    if args.chunk_size > 0:
        config_parts.append(f"cs{args.chunk_size}")
    config_tag = "_".join(config_parts) if config_parts else "baseline"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sub_dataset = dataset_config['sub_dataset']
    run_name = f"{timestamp}_{config_tag}_{sub_dataset}_{safe_model}"

    if args.context_idx is not None:
        output_file = output_dir / f"{run_name}_context_{args.context_idx}.json"
        trace_file = output_dir / f"{run_name}_context_{args.context_idx}_traces.jsonl"
    else:
        output_file = output_dir / f"{run_name}.json"
        trace_file = output_dir / f"{run_name}_traces.jsonl"

    # Print config summary
    print(f"Agent: HermesAgent (composed)")
    print(f"Dataset: {dataset_config['dataset']} / {dataset_config['sub_dataset']}")
    print(f"Model: {args.model}")
    print(f"Output: {output_file}")
    tri_str = f"triadic={args.triadic}"
    if args.triadic and args.triadic_chain_hops > 1:
        tri_str += f" (SA hops={args.triadic_chain_hops}, decay={args.triadic_decay})"
    evcal_str = f", event_cal={args.event_calendar_retrieval}" if args.event_calendar_retrieval else ""
    if args.event_calendar:
        evcal_str += (
            f", event_calendar_lme=True (±{args.event_calendar_window_days}d, "
            f"boost={args.event_calendar_boost})"
        )
    print(f"Retrieval: cooc_boost={args.cooc_boost}, bm25_hops={args.bm25_hops}, {tri_str}{evcal_str}")
    print(f"Query: decompose={args.decompose_query}, cr={args.conflict_resolution}, cr_per_hop={args.cr_per_hop}")
    _mgt = args.max_gen_tokens if args.max_gen_tokens > 0 else (64 if "factconsolidation" in args.dataset_config.lower() else 128)
    print(f"LLM: max_gen_tokens={_mgt}, temperature={args.temperature}, role_weight_assistant={args.role_weight_assistant}, role_weight_continuation={args.role_weight_continuation}")
    print()

    # Load data
    conversation_creator = ConversationCreator(agent_config, dataset_config)

    # Replace context with patched version if provided (fixes serial ordering)
    if args.patched_context:
        with open(args.patched_context) as f:
            patched = f.read()
        conversation_creator.contexts[0] = patched
        print(f"Loaded patched context: {args.patched_context} ({len(patched)} chars)")

    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    print(f"Contexts: {len(all_chunks)}")
    if args.context_idx is not None:
        if args.context_idx < 0 or args.context_idx >= len(all_chunks):
            print(f"ERROR: --context_idx {args.context_idx} out of range [0, {len(all_chunks)-1}]")
            sys.exit(1)
        print(f"Running ONLY context {args.context_idx} (parallel mode)")
    total_queries = sum(len(qa) for qa in all_qa_pairs)
    print(f"Total queries: {total_queries}")

    # Load corrected ground truth if provided
    corrected_map: dict[int, str] = {}
    if args.corrected_answers:
        with open(args.corrected_answers) as f:
            for entry in json.load(f):
                if entry["corrected_answer"] != entry["expected_wrong"]:
                    corrected_map[entry["qi"]] = entry["corrected_answer"]
        print(f"Corrected ground truth: {len(corrected_map)} answers overridden")

    # Ingestion cache setup
    use_cache = not args.no_cache
    cache_path = None
    effective_cs = dataset_config.get("chunk_size", 0)
    if use_cache:
        cache_path = cache_dir(dataset_config["sub_dataset"], effective_cs, args.cache_label)
        print(f"Ingestion cache: {cache_path}")
        if args.refresh_cache:
            print("  (refresh mode: will re-ingest and overwrite)")
    else:
        # --no_cache: skip loading but always save with timestamped label
        no_cache_label = f"{dataset_config['sub_dataset']}_cs{effective_cs}_{timestamp}_{config_tag}"
        cache_path = cache_dir(dataset_config["sub_dataset"], effective_cs, label=no_cache_label)
        print(f"Ingestion cache: save-only → {cache_path}")
    print()

    # Build agent
    agent = _build_agent(args)

    # Run evaluation
    metrics: dict[str, list] = defaultdict(list)
    results: list[dict] = []
    query_index = 0
    start_time = time.time()

    # Filter to single context if --context_idx is set
    if args.context_idx is not None:
        context_indices = [args.context_idx]
    else:
        context_indices = list(range(len(all_chunks)))

    for context_idx in tqdm(context_indices, desc="Contexts"):
        chunks = all_chunks[context_idx]
        qa_pairs = all_qa_pairs[context_idx]
        agent.reset()

        # Try loading from ingestion cache (skip when --no_cache)
        cached = False
        if use_cache and not args.refresh_cache:
            cached = _load_context(agent, cache_path, context_idx)

        if not cached:
            print(f"\n--- Context {context_idx}: memorizing {len(chunks)} chunks ---")
            for chunk in tqdm(chunks, desc="Memorizing", leave=False):
                agent.send_message(chunk, memorizing=True)
            # Finalize CoocKG NPMI weights after all facts are ingested
            agent.store.finalize_cooc_kg()
            agent._scorer.save_disk_cache()

            # Always save cache (--no_cache saves with timestamped label)
            if cache_path is not None:
                _save_context(agent, cache_path, context_idx)

        # Query
        print(f"    Querying {len(qa_pairs)} questions...")
        for query_data in tqdm(qa_pairs, desc="Querying", leave=False):
            if len(query_data) == 3:
                query, answer, qa_pair_id = query_data
            else:
                query, answer = query_data
                qa_pair_id = None

            if args.max_test_queries > 0 and query_index >= args.max_test_queries:
                break

            # Override broken ground truth answers
            if query_index in corrected_map:
                answer = corrected_map[query_index]

            agent_output = agent.send_message(query, memorizing=False)

            # Write per-query trace to JSONL for easy review
            trace = agent_output.pop("trace", None)
            if trace is not None:
                trace_record = {
                    "query_index": query_index,
                    "question": query,
                    "expected_answer": answer if isinstance(answer, str) else answer[0] if answer else None,
                    "predicted_answer": agent_output.get("output", ""),
                    **trace,
                }
                with open(trace_file, "a") as tf:
                    tf.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

            metrics, results = metrics_summarization(
                agent_output,
                query,
                answer,
                dataset_config,
                metrics,
                results,
                query_index,
                qa_pair_id,
            )
            query_index += 1

        # Save intermediate results
        averaged_metrics = {
            key: np.mean(values) * (1 if ("_len" in key) or ("_time" in key) else 100)
            for key, values in metrics.items()
        }
        output_data = {
            "agent_config": agent_config,
            "dataset_config": dataset_config,
            "hermes_config": {
                "architecture": "composed",
                "retrieval": {
                    "cooc_boost": args.cooc_boost,
                    "cooc_weight": args.cooc_weight if args.cooc_boost else None,
                    "bm25_hops": args.bm25_hops,
                    "triadic": args.triadic,
                    "bm25_primary": args.bm25_primary,
                    "cross_encoder": args.cross_encoder,
                    "cross_encoder_top_k": args.cross_encoder_top_k,
                    "entity_primary": getattr(args, "entity_primary", False),
                    "entity_prefilter": getattr(args, "entity_prefilter", False),
                    "event_calendar_retrieval": getattr(args, "event_calendar_retrieval", False),
                    "event_calendar": getattr(args, "event_calendar", False),
                    "event_calendar_window_days": getattr(args, "event_calendar_window_days", 60),
                    "event_calendar_boost": getattr(args, "event_calendar_boost", 0.15),
                    "aggregation_overfetch": getattr(args, "aggregation_overfetch", 0),
                    "structured_store": getattr(args, "structured_store", False),
                    "bm25_rerank": getattr(args, "bm25_rerank", False),
                    "fisher_scoring": getattr(args, "fisher_scoring", False),
                },
                "cr": {
                    "enabled": args.conflict_resolution,
                    "per_hop": args.cr_per_hop,
                },
                "decompose": {
                    "enabled": args.decompose_query,
                    "max_hops": args.decompose_max_hops,
                },
                "chunk_size": args.chunk_size,
                "max_gen_tokens": _mgt,
                "temperature": args.temperature,
                "llm_extraction": bool(getattr(args, "llm_extraction_url", None)),
                "dynamic_prompting": args.dynamic_prompting,
                "raw_chunks": args.raw_chunks,
                "structured_query": args.structured_query,
                "structured_query_max_hops": args.structured_query_max_hops,
            },
            "data": results,
            "metrics": {k: v for k, v in metrics.items()},
            "averaged_metrics": averaged_metrics,
            "time_cost": time.time() - start_time,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Evaluation complete: {query_index} queries in {elapsed:.1f}s")
    print(f"Results saved to: {output_file}")
    print(f"\nMetrics:")
    for key, values in metrics.items():
        scale = 1 if ("_len" in key) or ("_time" in key) else 100
        print(f"  {key}: {np.mean(values) * scale:.2f}")


if __name__ == "__main__":
    main()
