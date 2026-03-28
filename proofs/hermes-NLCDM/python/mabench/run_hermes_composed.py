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
    GLiNER2Config, CoocExtractionConfig, RelikConfig,
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
    parser.add_argument("--chunk_size", type=int, default=0)
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

    # Fisher-Rao scoring
    parser.add_argument("--fisher_scoring", action="store_true",
                        help="Replace cosine base similarity with Fisher-Rao inverse-variance weighted distance (arXiv:2603.14588v1)")
    parser.add_argument("--fisher_temperature", type=float, default=1.0,
                        help="Fisher-Rao temperature (lower = more discriminative, default 1.0)")

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

    # Cache
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--cache_label", type=str, default=None)
    parser.add_argument("--list_caches", action="store_true")

    return parser.parse_args()


def _build_agent(args) -> HermesAgent:
    """Map CLI flags to frozen config dataclasses and construct HermesAgent."""
    dream_params = DreamParams(
        eta=args.dream_eta,
        min_sep=args.dream_min_sep,
        prune_threshold=args.dream_prune_threshold,
        merge_threshold=args.dream_merge_threshold,
        explore_only=args.dream_explore_only,
    )

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
        fisher_scoring=args.fisher_scoring,
        fisher_temperature=args.fisher_temperature,
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
    llm_cfg = LLMConfig(
        model=args.model,
        temperature=0.0,
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

    return HermesAgent(
        dim=1024,
        contradiction_threshold=args.contradiction_threshold,
        recency_alpha=args.recency_alpha,
        dream_params=dream_params,
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
        llm_extraction_base_url=args.llm_extraction_url,
        llm_extraction_model=args.llm_extraction_model,
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
        "temperature": 0.0,
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
    if args.entity_rerank:
        config_parts.append("entrank")
    if args.fisher_scoring:
        config_parts.append("fisher")
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
    if args.chunk_size > 0:
        config_parts.append(f"cs{args.chunk_size}")
    config_tag = "_".join(config_parts) if config_parts else "baseline"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sub_dataset = dataset_config['sub_dataset']
    run_name = f"{timestamp}_{config_tag}_{sub_dataset}_{safe_model}"

    output_file = output_dir / f"{run_name}.json"
    trace_file = output_dir / f"{run_name}_traces.jsonl"

    # Print config summary
    print(f"Agent: HermesAgent (composed)")
    print(f"Dataset: {dataset_config['dataset']} / {dataset_config['sub_dataset']}")
    print(f"Model: {args.model}")
    print(f"Output: {output_file}")
    print(f"Retrieval: cooc_boost={args.cooc_boost}, bm25_hops={args.bm25_hops}, triadic={args.triadic}")
    print(f"Query: decompose={args.decompose_query}, cr={args.conflict_resolution}, cr_per_hop={args.cr_per_hop}")
    print()

    # Load data
    conversation_creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = conversation_creator.get_chunks()
    all_qa_pairs = conversation_creator.get_query_and_answers()

    print(f"Contexts: {len(all_chunks)}")
    total_queries = sum(len(qa) for qa in all_qa_pairs)
    print(f"Total queries: {total_queries}")

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

    for context_idx, (chunks, qa_pairs) in enumerate(
        tqdm(
            zip(all_chunks, all_qa_pairs),
            total=len(all_chunks),
            desc="Contexts",
        )
    ):
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
                },
                "cr": {
                    "enabled": args.conflict_resolution,
                    "per_hop": args.cr_per_hop,
                },
                "decompose": {
                    "enabled": args.decompose_query,
                    "max_hops": args.decompose_max_hops,
                },
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
