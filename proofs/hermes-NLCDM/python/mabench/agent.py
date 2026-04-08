"""Composed Hermes Memory Agent — standalone module wiring.

Constructs all memory modules with explicit dependencies and wires
cross-module callbacks.  Each module is independently importable and
testable; this file only does wiring.

    from mabench.agent import HermesAgent
    from mabench.configs import RetrievalConfig, DecomposeConfig, CRConfig

    agent = HermesAgent(
        retrieval=RetrievalConfig(cooc_boost=True),
        decompose=DecomposeConfig(enabled=True),
        cr=CRConfig(enabled=True, per_hop=True),
    )
    agent.send_message("Paris is the capital of France.", memorizing=True)
    result = agent.send_message("What is the capital of France?")
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add hermes-memory and hermes-NLCDM to path
_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))

from hermes_memory.orchestrator import MemoryOrchestrator
from hermes_memory.engine import ParameterSet
from hermes_memory.recall import RecallConfig
from hermes_memory.contradiction import ContradictionConfig
from hermes_memory.consolidation import ConsolidationConfig
from hermes_memory.encoding import EncodingConfig
from coupled_engine import CoupledEngine
from dream_ops import DreamParams

from mabench.configs import (
    RetrievalConfig, CRConfig, DecomposeConfig, StoreConfig,
    LLMConfig, BeliefConfig, EntityConfig, DedupConfig, KGConfig,
    GLiNER2Config, CoocExtractionConfig, RelikConfig, SupersessionState, EntityIndex,
    StructuredStoreConfig,
)
from mabench.ingestion import IngestionPipeline, IngestionConfig, _MutableEntityGraph
from mabench.scoring import EmbeddingRelevanceScorer
from mabench.belief import BeliefIndex
from mabench.bm25_index import BM25Index
from mabench.chain_cache import ChainCache
from mabench.triadic_memory import TriadicMemory
from mabench.rebel import RebelKG
from mabench.outgestion import Outgester
from mabench.entity import EntityMixin as EntityOps
from mabench.retrieval import Retriever
from mabench.conflict_resolution import ConflictResolver
from mabench.store import FactStore
from mabench.query import QueryEngine
from mabench.query_pipeline import QueryPipeline
from mabench.iterative_query import IterativeQuery
from mabench.entity_reranker import EntityReranker
from mabench.cross_encoder_reranker import CrossEncoderReranker
from mabench.hyperedge_intersection import HyperedgeIntersection
from mabench.knowledge_graph import KnowledgeGraph
from mabench.gliner2_extractor import GLiNER2Extractor
from mabench.cooccurrence_extractor import CooccurrenceExtractor
from mabench.relik_extractor import RelikExtractor
from mabench.conversation_parser import parse_chunk
from mabench.event_db import EventDB
from mabench.entity_graph import EntityGraph

import re as _re

_REFUSAL_PATTERNS = _re.compile(
    r"I(?:'m| am) sorry|I don'?t know|not specified|not provided|no information|"
    r"does not (?:specify|provide|mention)|not mentioned|no relevant|"
    r"cannot (?:find|determine|recall)|no record|unable to (?:find|determine)",
    _re.IGNORECASE,
)


def _is_refusal(text: str) -> bool:
    """Detect LLM refusal/hedging in generated answers."""
    return bool(_REFUSAL_PATTERNS.search(text))


# Default V1 orchestrator params (validated in test_engine.py)
_V1_PARAMS = ParameterSet(
    alpha=0.1, beta=0.1, delta_t=1.0, s_max=10.0, s0=1.0,
    temperature=5.0, novelty_start=0.5, novelty_decay=0.2,
    survival_threshold=0.05, feedback_sensitivity=0.1,
    w1=0.35, w2=0.25, w3=0.20, w4=0.20,
)


class _ScorerEmbedderAdapter:
    """Bridges EmbeddingRelevanceScorer → ChronosEventIndex embedder interface.

    ChronosEventIndex expects an embedder with .encode(texts) returning ndarray.
    The scorer lazy-loads its SentenceTransformer, so we delegate to embed_batch().
    """
    def __init__(self, scorer):
        self._scorer = scorer

    def encode(self, texts):
        import numpy as np
        embeddings = self._scorer.embed_batch(texts)
        return np.array(embeddings)


class HermesAgent:
    """Composed memory agent — modules wired with explicit dependencies.

    Each module (Retriever, ConflictResolver, FactStore, QueryEngine, …)
    is independently importable.  This class constructs them all, wires
    cross-module callbacks, and exposes the MABench-compatible public API.

    Module references are public for testing and plugin replacement:
        agent.retriever, agent.store, agent.query_engine, etc.
    """

    def __init__(
        self,
        *,
        # Engine params
        dim: int = 768,
        contradiction_threshold: float = 0.95,
        recency_alpha: float = 0.1,
        dedup_threshold: float = 0.0,
        dream_params: DreamParams | None = None,
        # Module configs (frozen dataclasses from configs.py)
        retrieval: RetrievalConfig | None = None,
        cr: CRConfig | None = None,
        decompose: DecomposeConfig | None = None,
        store: StoreConfig | None = None,
        llm: LLMConfig | None = None,
        belief: BeliefConfig | None = None,
        entity: EntityConfig | None = None,
        dedup: DedupConfig | None = None,
        kg: KGConfig | None = None,
        gliner2: GLiNER2Config | None = None,
        cooc_extraction: CoocExtractionConfig | None = None,
        relik: RelikConfig | None = None,
        # Infrastructure flags
        bm25_hops: bool = False,
        chain_cache: bool = False,
        rebel_kg: bool = False,
        triadic: bool = False,
        triadic_n: int = 1000,
        triadic_p: int = 10,
        triadic_chain_hops: int = 1,
        triadic_decay: float = 1.0,
        # Query mode
        iterative: bool = False,
        iterative_max_hops: int = 3,
        # Structured event-DB tool-calling query (Chronos-style)
        structured_query: bool = False,
        structured_query_max_hops: int = 3,
        # LLM triple extraction (vLLM server URL, None = disabled)
        llm_extraction_base_url: str | None = None,
        llm_extraction_model: str = "qwen3-triple-extractor",
        llm_extraction_mode: str = "factual",
        # Coreference resolution backend
        coref_backend: str = "fastcoref",
        # Dynamic prompting (per-question-type system prompts)
        dynamic_prompting: bool = False,
        # Raw chunk mode: skip conversation parsing, store whole chunks
        raw_chunks: bool = False,
        # Pre-computed triples cache (dict: md5_hash -> {triples, event_times})
        triples_cache: dict | None = None,
        # LFM SVO extraction (local LM Studio, LME entity index enrichment)
        lfm_extraction_url: str | None = None,
        # Path to pre-extracted SVO cache (skips live extraction)
        svo_cache_path: str | None = None,
        # Structured fact store (SVO triple index as RRF retrieval channel)
        structured_store: StructuredStoreConfig | None = None,
        # Chronos modules (reverse-engineered from arXiv 2603.16862)
        chronos_event_index: bool = False,
        chronos_dynamic_prompt: bool = False,
        chronos_context_expansion: bool = False,
        chronos_agent: bool = False,
        # Remote embedding API (replaces local SentenceTransformer)
        embed_url: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        # Store configs (frozen — safe to share across modules)
        self._retrieval_cfg = retrieval or RetrievalConfig()
        self._cr_cfg = cr or CRConfig()
        self._decompose_cfg = decompose or DecomposeConfig()
        self._store_cfg = store or StoreConfig()
        self._llm_cfg = llm or LLMConfig()
        self._belief_cfg = belief or BeliefConfig()
        self._entity_cfg = entity or EntityConfig()
        self._dedup_cfg = dedup or DedupConfig()
        self._kg_cfg = kg or KGConfig()
        self._gliner2_cfg = gliner2 or GLiNER2Config()
        self._cooc_extraction_cfg = cooc_extraction or CoocExtractionConfig()
        self._relik_cfg = relik or RelikConfig()
        self._structured_store_cfg = structured_store or StructuredStoreConfig()

        # Engine params (saved for reset)
        self._dim = dim
        self._contradiction_threshold = contradiction_threshold
        self._recency_alpha = recency_alpha
        self._dedup_threshold = dedup_threshold
        self._dream_params = dream_params

        # Infrastructure flags (saved for reset)
        self._bm25_hops = bm25_hops
        self._chain_cache_flag = chain_cache
        self._rebel_kg_flag = rebel_kg
        self._triadic_flag = triadic
        self._triadic_n = triadic_n
        self._triadic_p = triadic_p
        self._triadic_chain_hops = triadic_chain_hops
        self._triadic_decay = triadic_decay
        self._iterative_flag = iterative
        self._iterative_max_hops = iterative_max_hops
        self._structured_query_flag = structured_query
        self._structured_query_max_hops = structured_query_max_hops
        self._llm_extraction_base_url = llm_extraction_base_url
        self._llm_extraction_model = llm_extraction_model
        self._llm_extraction_mode = llm_extraction_mode
        self._dynamic_prompting = dynamic_prompting
        self._raw_chunks = raw_chunks
        self._lfm_extraction_url = lfm_extraction_url
        self._svo_cache_path = svo_cache_path
        self._triples_cache = triples_cache

        # Chronos flags
        self._chronos_event_index_flag = chronos_event_index
        self._chronos_dynamic_prompt_flag = chronos_dynamic_prompt
        self._chronos_context_expansion_flag = chronos_context_expansion
        self._chronos_agent_flag = chronos_agent

        # -- Persistent resources (survive reset) --------------------------
        self._embed_url = embed_url
        self._embed_model = embed_model
        self._scorer = EmbeddingRelevanceScorer.create(
            embed_url=embed_url, embed_model=embed_model,
        )
        try:
            import torch
            _dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _dev = "cpu"
        self._ingestion = IngestionPipeline(
            IngestionConfig(
                coref_device=_dev, coref_enabled=True,
                coref_backend=coref_backend,
                boundary_threshold=0.2, strip_numbered_prefix=False,
                llm_extraction_base_url=llm_extraction_base_url,
                llm_extraction_model=llm_extraction_model,
                llm_extraction_mode=llm_extraction_mode,
            )
        )

        # -- Conversation parser state (survives reset for continuity) ------
        self._conv_turn_counter: int = 0
        self._conv_last_timestamp: str = ""
        self._conv_mode_detected: bool = False

        # -- Build all transient resources and modules ---------------------
        self._build()

    # -----------------------------------------------------------------
    # Module construction + cross-module callback wiring
    # -----------------------------------------------------------------

    def _build(self) -> None:
        """Create engine, state containers, infrastructure, and modules.

        Called from __init__ and reset().  The IngestionPipeline and
        EmbeddingRelevanceScorer are *not* recreated — they hold loaded
        NLP models that are stateless w.r.t. memory contents.
        """

        # -- 1. Shared resources ------------------------------------------
        # CoupledEngine: vector store + cosine similarity
        self.engine = CoupledEngine(
            dim=self._dim, contradiction_aware=False,
            contradiction_threshold=self._contradiction_threshold,
            novelty_N0=0.2, novelty_gamma=0.05,
            emotional_tagging=False, reconsolidation=False,
            recency_alpha=self._recency_alpha,
            dream_params=self._dream_params,
            dedup_threshold=self._dedup_threshold,
        )
        # MemoryOrchestrator: V1 text-based contradiction detection
        self.orchestrator = MemoryOrchestrator(
            params=_V1_PARAMS,
            encoding_config=EncodingConfig(),
            contradiction_config=ContradictionConfig(),
            consolidation_config=ConsolidationConfig(),
            recall_config=RecallConfig(total_budget=4000),
            relevance_scorer=self._scorer,
        )

        # -- 2. Shared mutable state --------------------------------------
        # SupersessionState: written by FactStore, read by CR + query
        ss = SupersessionState()
        # EntityIndex: written by FactStore, read by EntityOps + query
        ei = EntityIndex()
        ea = _MutableEntityGraph()
        self._supersession_state = ss
        self._entity_index = ei

        # -- 3. Optional infrastructure -----------------------------------
        _need_bm25 = self._bm25_hops or self._retrieval_cfg.bm25_rerank or self._retrieval_cfg.bm25_primary or self._structured_query_flag or self._chronos_agent_flag
        bm25 = BM25Index() if _need_bm25 else None
        chain = ChainCache(similarity_threshold=0.95) if self._chain_cache_flag else None
        tri = TriadicMemory(n=self._triadic_n, p=self._triadic_p) if self._triadic_flag else None
        bel = (BeliefIndex(
            prior_alpha=self._belief_cfg.prior_alpha,
            prior_beta=self._belief_cfg.prior_beta,
            propagation_damping=self._belief_cfg.propagation_damping,
            hard_floor=self._belief_cfg.hard_floor,
        ) if self._belief_cfg.enabled else None)

        # -- 4. Leaf modules (no cross-module deps) -----------------------
        rebel = RebelKG() if self._rebel_kg_flag else None
        outgester = Outgester(
            self._llm_cfg,
            dynamic_prompting=self._dynamic_prompting,
            exact_words_prompt=self._retrieval_cfg.exact_words_prompt,
        )
        ent_reranker = (EntityReranker(
            self._scorer,
            nlp=getattr(self._ingestion, "_nlp", None),
        ) if self._retrieval_cfg.entity_rerank else None)

        # CrossEncoderReranker: ms-marco-MiniLM-L-6-v2 relevance reranking
        cross_enc_reranker = (CrossEncoderReranker(
            top_k=self._retrieval_cfg.cross_encoder_top_k,
        ) if self._retrieval_cfg.cross_encoder else None)

        # HyperedgeIntersection: entity-constrained reranking in decompose hops
        hyperedge = (HyperedgeIntersection(
            entity_index=ei.index,
            penalty=self._retrieval_cfg.hyperedge_penalty,
            min_candidates=self._retrieval_cfg.hyperedge_min_candidates,
        ) if self._retrieval_cfg.hyperedge_intersection else None)

        # KnowledgeGraph: rustworkx PPR graph for multi-hop retrieval
        knowledge_graph = (KnowledgeGraph(
            damping=self._kg_cfg.damping,
            ppr_top_k=self._kg_cfg.ppr_top_k,
            max_facts_per_node=self._kg_cfg.max_facts_per_node,
        ) if self._kg_cfg.enabled else None)

        # GLiNER2 triple extraction (dense triples for KG + supersession)
        gliner2_extractor = (GLiNER2Extractor(
            model_name=self._gliner2_cfg.model_name,
            threshold=self._gliner2_cfg.threshold,
            batch_size=self._gliner2_cfg.batch_size,
        ) if self._gliner2_cfg.enabled else None)

        # Co-occurrence extraction (fast path: bi-encoder NER → co-occurrence edges)
        cooc_extractor = (CooccurrenceExtractor(
            model_name=self._cooc_extraction_cfg.model_name,
            threshold=self._cooc_extraction_cfg.threshold,
            batch_size=self._cooc_extraction_cfg.batch_size,
        ) if self._cooc_extraction_cfg.enabled else None)

        # CoocKG: sentence-window co-occurrence graph for structural hop resolution
        from mabench.cooccurrence_extractor import CoocKG
        cooc_kg = CoocKG() if cooc_extractor is not None else None

        # ReLiK v2 relation extraction (canonical Wikidata triples for supersession)
        relik_extractor = (RelikExtractor(
            model_name=self._relik_cfg.model_name,
            min_confidence=self._relik_cfg.min_confidence,
        ) if self._relik_cfg.enabled else None)

        # Structured fact store (SVO triple index as RRF retrieval channel)
        if self._structured_store_cfg.enabled:
            from mabench.structured_store import StructuredFactStore
            structured_store = StructuredFactStore()
        else:
            structured_store = None
        self._structured_store = structured_store

        # EventDB: entity-keyed structured event store (PRIMARY retrieval
        # path when --structured_primary is on). Populated during ingestion
        # from LLM triple extraction output. Requires llm_extraction_url.
        event_db = EventDB() if (self._retrieval_cfg.structured_primary or self._retrieval_cfg.structured_rrf or self._chronos_agent_flag) else None
        self._event_db = event_db

        # EntityGraph: per-entity knowledge graph built from triples during ingestion.
        # Used as an additional RRF retrieval channel in entity_prefilter.
        entity_graph = EntityGraph() if self._retrieval_cfg.entity_graph else None
        self._entity_graph = entity_graph

        # -- 5. Tier 1: entity ops + retriever ----------------------------
        entity_ops = EntityOps(
            self.engine, self._scorer,
            entity_index=ei,
            retrieve_num=self._retrieval_cfg.retrieve_num,
            config=self._entity_cfg,
        )
        retriever = Retriever(
            self.engine, self._scorer,
            belief_index=bel,
            config=self._retrieval_cfg,
            cr_config=self._cr_cfg,
            dedup_config=self._dedup_cfg,
            belief_config=self._belief_cfg,
            nlp=getattr(self._ingestion, "_nlp", None),
            entity_reranker=ent_reranker,
            cross_encoder_reranker=cross_enc_reranker,
            bm25_index=bm25,
            structured_store=structured_store,
        )

        # -- 6. Tier 2: conflict resolver ---------------------------------
        conflict_resolver = ConflictResolver(
            self.engine, self._scorer, self._ingestion, ss,
            cr_entity_gate=self._cr_cfg.entity_gate,
            cosine_dedup_threshold=self._dedup_cfg.threshold,
            retrieve_fn=retriever._retrieve_v2,
            # decompose_fn wired below after pipeline exists
        )

        # -- 7. Tier 3: fact store ----------------------------------------
        self.store = FactStore(
            self.engine, self._scorer, self._ingestion, self.orchestrator,
            supersession_state=ss, entity_index=ei,
            rebel=rebel, bm25_index=bm25, triadic=tri,
            chain_cache=chain, belief_index=bel, entity_accumulator=ea,
            supersession=self._store_cfg.supersession,
            cooc_boost=self._store_cfg.cooc_boost,
            dream_interval=self._store_cfg.dream_interval,
            entity_reranker=ent_reranker,
            knowledge_graph=knowledge_graph,
            gliner2_extractor=gliner2_extractor,
            cooc_extractor=cooc_extractor,
            relik_extractor=relik_extractor,
            cooc_kg=cooc_kg,
        )

        # -- 8. Tier 4: query engine + pipeline (circular dep) ------------
        # Get entity registry reference (shared, grows during ingestion)
        _llm_ext = getattr(self._ingestion, "_llm_extractor", None)
        _entity_reg = _llm_ext.entity_registry if _llm_ext is not None else {}

        # Create QueryEngine first (pipeline callbacks wired after)
        self.query_engine = QueryEngine(
            self.engine, self._scorer,
            retriever=retriever, conflict_resolver=conflict_resolver,
            entity_ops=entity_ops, rebel=rebel, outgester=outgester,
            bm25_index=bm25, triadic=tri,
            chain_cache=chain, belief_index=bel,
            llm_config=self._llm_cfg, retrieval_config=self._retrieval_cfg,
            cr_config=self._cr_cfg, decompose_config=self._decompose_cfg,
            entity_config=self._entity_cfg, dedup_config=self._dedup_cfg,
            hyperedge=hyperedge, knowledge_graph=knowledge_graph,
            cooc_kg=cooc_kg, supersession_state=ss,
            entity_registry=_entity_reg,
        )
        # Create QueryPipeline with hop callback from QueryEngine
        self.pipeline = QueryPipeline(
            self.engine, self._scorer,
            retriever=retriever, conflict_resolver=conflict_resolver,
            outgester=outgester, rebel=rebel,
            orchestrator=self.orchestrator,
            bm25_index=bm25, triadic=tri,
            chain_cache=chain, belief_index=bel,
            retrieval_config=self._retrieval_cfg, cr_config=self._cr_cfg,
            decompose_config=self._decompose_cfg, dedup_config=self._dedup_cfg,
            llm_config=self._llm_cfg,
            hop_executor_fn=self.query_engine._query_run_hop,
            graph_walk_fn=self.query_engine._graph_walk_resolve,
            kg_hop_fn=self.query_engine._resolve_triple_index_kg_hop,
            hyperedge=hyperedge, knowledge_graph=knowledge_graph,
        )
        # Wire QueryEngine -> QueryPipeline callbacks (breaks circular dep)
        self.query_engine._final_retrieval_fn = self.pipeline._query_final_retrieval
        self.query_engine._post_process_fn = self.pipeline._query_post_process
        self.query_engine._generate_fn = self.pipeline._query_generate
        self.query_engine._extract_short_answer_fn = self.pipeline._extract_short_answer
        # Wire ConflictResolver -> QueryPipeline._decompose_query
        conflict_resolver._decompose_fn = self.pipeline._decompose_query

        # IterativeQuery (independent of the circular dep)
        self.iterative = IterativeQuery(
            self.engine, self._scorer,
            retriever=retriever, conflict_resolver=conflict_resolver,
            outgester=outgester, triadic=tri, belief_index=bel,
            llm_config=self._llm_cfg, retrieval_config=self._retrieval_cfg,
            cr_config=self._cr_cfg,
            iterative_max_hops=self._iterative_max_hops,
            supersession=self._store_cfg.supersession,
            superseded_texts=ss.superseded_texts,
        )

        # Structured event-DB query agent (Chronos-style tool-calling loop).
        # Lazy-constructed: tools reference the store's indexes by reference,
        # so they stay in sync as ingestion populates them.
        self.structured_agent = None
        if self._structured_query_flag:
            from mabench.structured_query import StructuredQueryAgent
            self.structured_agent = StructuredQueryAgent(
                memory_store=self.engine.memory_store,
                entity_inv_idx=self.store._entity_inv_idx,
                event_calendar=self.store._event_calendar,
                bm25_index=getattr(self.store, "_bm25_index", None),
                llm_config=self._llm_cfg,
                max_hops=self._structured_query_max_hops,
                engine=self.engine,
                scorer=self._scorer,
            )

        # LFM SVO extractor (local LM Studio, LME entity index enrichment)
        self._lfm_extractor = None
        if self._lfm_extraction_url:
            from mabench.lfm_extractor import LFMExtractor
            self._lfm_extractor = LFMExtractor(base_url=self._lfm_extraction_url)

        # SVO cache: load pre-extracted triples if path provided
        self._svo_cache: dict[int, list[tuple[str, str, str]]] | None = None
        if self._svo_cache_path:
            from mabench.svo_cache import load_cache
            from pathlib import Path
            self._svo_cache = load_cache(Path(self._svo_cache_path))
            if self._svo_cache is not None:
                import logging
                logging.getLogger(__name__).info(
                    f"Loaded SVO cache: {len(self._svo_cache)} turns"
                )

        # Expose remaining module references for testing / plugin swap
        self.retriever = retriever
        self.conflict_resolver = conflict_resolver
        self.entity_ops = entity_ops
        self.rebel = rebel
        self.outgester = outgester
        self.hyperedge = hyperedge
        self.knowledge_graph = knowledge_graph
        self.gliner2_extractor = gliner2_extractor
        self.cooc_extractor = cooc_extractor
        self.relik_extractor = relik_extractor
        self.cooc_kg = cooc_kg

        # -- Chronos modules (arXiv 2603.16862) -----------------------------
        if self._chronos_event_index_flag:
            from mabench.chronos_event_index import ChronosEventIndex
            self._chronos_event_idx = ChronosEventIndex(
                embedder=_ScorerEmbedderAdapter(self._scorer),
            )
        else:
            self._chronos_event_idx = None
        if self._chronos_context_expansion_flag:
            from mabench.chronos_context_expansion import ContextExpander
            self._context_expander = ContextExpander()
        else:
            self._context_expander = None
        # Chronos reranker: cross-encoder for initial retrieval + agent tool reranking
        # Paper Section 3.3: "cross-encoder reranking to rescore the 100 candidates"
        # Paper Section 3.4: "reranking is applied to the original question"
        if self._chronos_agent_flag:
            self._chronos_reranker = CrossEncoderReranker(top_k=15)
        else:
            self._chronos_reranker = None
        self._chronos_turn_positions: dict[str, int] = {}

        # Wire EventDB to retriever for structured_primary query-time access.
        retriever._event_db = event_db

        # Wire entity inverted indexes (mutable dicts, shared references).
        # Store builds them during ingestion; retriever + pipeline read at query time.
        retriever._entity_inv_idx = self.store._entity_inv_idx
        retriever._bigram_inv_idx = self.store._bigram_inv_idx
        retriever._unigram_inv_idx = self.store._unigram_inv_idx
        self.pipeline._entity_inv_idx = self.store._entity_inv_idx
        self.pipeline._bigram_inv_idx = self.store._bigram_inv_idx
        self.pipeline._unigram_inv_idx = self.store._unigram_inv_idx

        # Wire event calendar (mutable list, shared reference)
        retriever._event_calendar = self.store._event_calendar
        self.pipeline._event_calendar = self.store._event_calendar

        # Wire entity graph (shared mutable object, populated during ingestion)
        retriever._entity_graph = entity_graph
        self.pipeline._entity_graph = entity_graph

        # LME date-mention calendar — lazily constructed so the module import
        # and dateparser warmup are paid only when the flag is on. Wired to
        # the retriever under a distinct attribute (_event_calendar_lme) so
        # it never collides with the FC-style keyword calendar above.
        if getattr(self._retrieval_cfg, "event_calendar", False):
            from mabench.event_calendar_lme import EventCalendarLME
            self.store._event_calendar_lme = EventCalendarLME()
        retriever._event_calendar_lme = self.store._event_calendar_lme
        self.pipeline._event_calendar_lme = self.store._event_calendar_lme

    # -----------------------------------------------------------------
    # Public API (MABench-compatible)
    # -----------------------------------------------------------------

    def send_message(
        self,
        message: str,
        memorizing: bool = False,
        query_id: int | None = None,
        context_id: int | None = None,
    ) -> dict | str:
        """Store a fact or query memory.

        memorizing=True:  Ingest the message.  Returns ``"Memorized"``.
        memorizing=False: Query memory and generate an answer via LLM.
        """
        if memorizing:
            return self._store_with_parser(message)
        elif self._chronos_agent_flag:
            return self._chronos_query(message)
        elif self._structured_query_flag:
            # Hybrid: run baseline first, fall back to SQ on refusal
            baseline_result = self.query_engine._query(message)
            output = baseline_result.get("output", "") if isinstance(baseline_result, dict) else str(baseline_result)
            if _is_refusal(output):
                sq_result = self.structured_agent.answer(message)
                return {
                    "output": sq_result["output"],
                    "input_len": sq_result.get("prompt_tokens", 0),
                    "output_len": sq_result.get("completion_tokens", 0),
                }
            return baseline_result
        elif self._iterative_flag:
            return self.iterative._iterative_query(message)
        else:
            return self.query_engine._query(message)

    def _chronos_query(self, message: str) -> dict:
        """Chronos pipeline: dynamic prompt -> initial retrieval -> context expansion -> agent.

        Reverse-engineered from arXiv 2603.16862 Section 3.3-3.4.
        """
        from mabench.chronos_agent import ChronosAgent
        from mabench.chronos_dynamic_prompt import generate_retrieval_guidance
        from mabench.temporal import parse_query_date

        # 0. Extract question date (paper: "agent is presented with the question and its date")
        question_date = ""
        qd = parse_query_date(message)
        if qd:
            question_date = qd.strftime("%Y-%m-%d")

        # 1. Dynamic prompting (optional per-question LLM retrieval guidance)
        guidance = ""
        if self._chronos_dynamic_prompt_flag:
            try:
                client = self._ensure_openai_client()
                guidance = generate_retrieval_guidance(
                    message, client, model=self._llm_cfg.model,
                )
            except Exception:
                pass  # fall back to empty guidance

        # 2. Initial retrieval: top-100 via FAISS → rerank → top-15
        #    Paper Section 3.3: "cross-encoder reranking to rescore the 100
        #    retrieved candidates... top-15 most relevant turns are selected"
        q_emb = self._scorer.embed(message)
        hits = self.engine.query(q_emb, top_k=100)
        reranker = getattr(self, "_chronos_reranker", None)
        if reranker and len(hits) > 15:
            rerank_input = []
            for h in hits:
                if isinstance(h, dict) and "index" in h:
                    idx = int(h["index"])
                    entry = self.engine.memory_store[idx] if idx < len(self.engine.memory_store) else None
                    text = entry.text if entry and hasattr(entry, "text") else ""
                    rerank_input.append({"text": text, "score": float(h.get("score", 0)), "index": idx})
            reranked = reranker.rerank(message, rerank_input, top_k=15)
            top_indices = [r["index"] for r in reranked]
        else:
            top_indices = [
                int(h["index"]) for h in hits[:15]
                if isinstance(h, dict) and "index" in h
            ]

        # 3. Context expansion (+-1 turn from same session)
        if self._context_expander is not None and top_indices:
            top_indices = self._context_expander.expand(top_indices, k=1)

        # 4. Format pre-retrieved context with session headers
        pre_context = self._format_session_context(top_indices)

        # 5. Construct ChronosAgent and run ReAct loop
        agent = ChronosAgent(
            memory_store=self.engine.memory_store,
            event_index=self._chronos_event_idx,
            event_db=self._event_db,
            engine=self.engine,
            scorer=self._scorer,
            bm25_index=getattr(self.store, "_bm25_index", None),
            llm_config=self._llm_cfg,
            max_hops=5,
            pre_context=pre_context,
            retrieval_guidance=guidance,
            reranker=reranker,
            original_question=message,
            question_date=question_date,
        )
        result = agent.answer(message)
        return {
            "output": result["output"],
            "input_len": result.get("prompt_tokens", 0),
            "output_len": result.get("completion_tokens", 0),
        }

    def _ensure_openai_client(self):
        """Get or create OpenAI-compatible client for Chronos dynamic prompting."""
        if not hasattr(self, "_openai_client") or self._openai_client is None:
            import os
            from openai import OpenAI
            self._openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            )
        return self._openai_client

    def _format_session_context(self, indices: list[int]) -> str:
        """Format retrieved turns grouped by session date with headers.

        Paper Section 3.3: 'Retrieved turns are formatted into hierarchical
        natural language context blocks organized by conversation date.
        Turns group by session with date headers.'
        """
        if not indices:
            return ""

        session_groups: dict[str, list[str]] = {}
        ms = self.engine.memory_store

        for idx in indices:
            if idx < 0 or idx >= len(ms):
                continue
            entry = ms[idx]
            text = entry.text if hasattr(entry, "text") else str(entry)
            session_time = getattr(entry, "session_time", None)
            session_date = (
                session_time.strftime("%Y-%m-%d") if session_time else "Unknown"
            )
            session_groups.setdefault(session_date, []).append(text)

        parts: list[str] = []
        for i, (date_str, texts) in enumerate(sorted(session_groups.items()), 1):
            parts.append(f"Session {i} ({date_str})")
            for text in texts:
                parts.append(f"  {text}")
            parts.append("")  # blank line between sessions

        return "\n".join(parts).strip()

    def finalize_ingestion(self) -> None:
        """Called after all chunks are ingested. Builds deferred indexes."""
        if self._chronos_event_idx is not None:
            self._chronos_event_idx.build_index()

    @staticmethod
    def _is_conversation_chunk(text: str) -> bool:
        """Detect LongMemEval conversation format (same logic as monolithic)."""
        return ("Chat Time:" in text and "'role':" in text) or (
            text.lstrip().startswith("[") and "'role':" in text
        )

    def _store_with_parser(self, message: str) -> str:
        """Store a chunk, auto-detecting conversation format.

        Conversation chunks (LongMemEval): parsed into individual turns,
        each stored as ``"[timestamp] (role) content"`` for clean retrieval.
        Factual chunks (FactConsolidation): stored directly through the
        ingestion pipeline (spaCy segmentation + coref).

        When ``_raw_chunks`` is True, skip conversation parsing and store
        the whole chunk directly (matches framework agent architecture).

        The conversation detection is always-on — it fires regardless of
        ``_llm_extraction_mode`` because raw conversation JSON fed through
        spaCy segmentation produces garbled, unretrievable facts.
        """
        # Raw chunk mode: skip conversation parsing, store whole chunk
        if getattr(self, "_raw_chunks", False):
            return self._store_raw_chunk(message)

        # Auto-detect conversation format (LongMemEval)
        if self._is_conversation_chunk(message):
            self._conv_mode_detected = True
        if self._conv_mode_detected:
            return self._store_conversation_turns(message)

        # Non-conversation path (FactConsolidation, etc.)
        return self.store._store(message)

    def _store_raw_chunk(self, message: str) -> str:
        """Store whole chunk without conversation parsing.

        Matches the framework agent architecture: each 512-token chunk
        becomes one document in the vector store.  Skips per-turn
        splitting, preserving surrounding context for better cosine
        similarity matching at the cost of retrieval granularity.
        """
        from mabench.entity_index import (
            extract_bigrams,
            extract_content_unigrams,
            extract_entities,
        )
        from mabench.temporal import parse_chat_time

        engine = self.store._resolve_engine
        store = self.store

        # Parse session_time for temporal metadata
        session_time = parse_chat_time(message)
        if session_time is not None:
            store._last_session_time = session_time
        else:
            session_time = getattr(store, "_last_session_time", None)

        # Embed the whole chunk as one document
        emb = self._scorer.embed(message)

        # Store via V1 orchestrator (when cooc_boost inactive)
        if not store._resolve_cooc_boost:
            store._resolve_orchestrator.store(content=message, embedding=emb)

        # Store via V2 coupled engine
        ms_idx = engine.store(
            text=message,
            embedding=emb,
            recency=float(store._store_count),
            role="unknown",
            session_time=session_time,
        )
        store._resolve_text_to_store_idx[message] = ms_idx

        # BM25 index
        bm25 = getattr(store, "_bm25_index", None)
        if bm25 is not None:
            bm25.add(ms_idx, message)

        # Entity inverted indexes
        for ent in extract_entities(message):
            store._entity_inv_idx.setdefault(ent, set()).add(ms_idx)
        for bg in extract_bigrams(message):
            store._bigram_inv_idx.setdefault(bg, set()).add(ms_idx)
        for ug in extract_content_unigrams(message):
            store._unigram_inv_idx.setdefault(ug, set()).add(ms_idx)

        # Noun-chunk inverted index (when enabled)
        nc_inv = getattr(store, "_noun_chunk_inv_idx", None)
        if nc_inv is not None and nc_inv is not ...:
            from mabench.noun_chunk_index import extract_noun_chunks
            for chunk in extract_noun_chunks(message):
                nc_inv.setdefault(chunk, set()).add(ms_idx)
                if " " in chunk:
                    for word in chunk.split():
                        if len(word) > 2:
                            nc_inv.setdefault(word, set()).add(ms_idx)

        # Event calendar: extract all Chat Time timestamps from raw chunk
        from mabench.event_calendar import (
            parse_all_chat_times, extract_event_keywords, extract_inline_dates,
        )
        chat_times = parse_all_chat_times(message)
        if chat_times:
            # Use earliest timestamp as the chunk's date
            chunk_date = chat_times[0]
            keywords = extract_event_keywords(message)
            inline_dates = extract_inline_dates(message, chunk_date)
            store._event_calendar.append({
                "turn_idx": ms_idx,
                "date": chunk_date,
                "keywords": keywords,
                "inline_dates": inline_dates,
                "text": message,
            })

            # LME date-mention calendar: mirrors the conversation-turn path
            # for the raw-chunks ingestion mode. Uses the chunk's Chat Time
            # as dateparser anchor.
            cal_lme = getattr(store, "_event_calendar_lme", None)
            if cal_lme is not None:
                cal_lme.ingest_turn(ms_idx, message, chunk_date)

        # LLM triple extraction for raw chunks (populates subject_index + EventDB).
        # Checks pre-computed triples cache first; falls back to live LLM extraction.
        import hashlib as _hashlib
        triples = None
        event_times_raw = None
        cache_hit = False

        if self._triples_cache is not None:
            chunk_hash = _hashlib.md5(message.encode()).hexdigest()
            cached = self._triples_cache.get(chunk_hash)
            if cached is not None:
                triples = [tuple(t) for t in cached.get("triples", [])]
                event_times_raw = cached.get("event_times", [])
                cache_hit = True

        if triples is None:
            llm_ext = getattr(
                getattr(store, '_ingestion', None), '_llm_extractor', None,
            )
            if llm_ext is not None:
                try:
                    triples = llm_ext.extract(message, mode="conversation")
                    event_times_raw = llm_ext.last_event_times
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"LLM triple extraction (raw chunk) failed: {e}"
                    )

        if triples:
            ss = getattr(store, '_supersession_state', None)
            if ss is not None:
                for subj, pred, obj in triples:
                    subj_key = str(subj).lower().strip()
                    pred_key = str(pred).lower().strip()
                    if not subj_key or not pred_key or len(subj_key) < 3:
                        continue
                    if subj_key not in ss.subject_index:
                        ss.subject_index[subj_key] = {}
                    ss.subject_index[subj_key][pred_key] = (obj, message)
                    obj_key = str(obj).lower().strip()
                    if obj_key and len(obj_key) >= 3:
                        if obj_key not in ss.subject_index:
                            ss.subject_index[obj_key] = {}
                        ss.subject_index[obj_key][pred_key] = (subj, message)

            if self._event_db is not None:
                # Parse event_times from cache (ISO strings) or live extractor
                parsed_times = None
                if cache_hit and event_times_raw:
                    from datetime import datetime as _dt
                    parsed_times = []
                    for et_str in event_times_raw:
                        if et_str and isinstance(et_str, str):
                            try:
                                parsed_times.append(_dt.fromisoformat(et_str))
                            except (ValueError, TypeError):
                                parsed_times.append(None)
                        else:
                            parsed_times.append(None)
                elif not cache_hit and event_times_raw:
                    parsed_times = event_times_raw[0] if event_times_raw else None

                self._event_db.add_batch(
                    triples,
                    chunk_id=ms_idx,
                    event_date=session_time.date() if session_time else None,
                    event_times=parsed_times,
                )

        # EntityGraph: feed extracted triples into per-entity knowledge graph
        if self._entity_graph is not None and triples:
            for subj, pred, obj in triples:
                self._entity_graph.ingest_triple(
                    subj=str(subj), pred=str(pred), obj=str(obj),
                    turn_idx=ms_idx, chunk_idx=ms_idx,
                    timestamp=session_time,
                    source_text=message[:200],
                )

        # Chronos ContextExpander: register this chunk for +-k expansion
        if self._context_expander is not None:
            session_date_str = session_time.strftime("%Y-%m-%d") if session_time else "unknown"
            pos = self._chronos_turn_positions.get(session_date_str, 0)
            self._context_expander.register(
                chunk_id=ms_idx, session_id=session_date_str, position=pos,
            )
            self._chronos_turn_positions[session_date_str] = pos + 1

        # Chronos EventIndex: populate from extracted triples
        if self._chronos_event_idx is not None and triples:
            session_date_str_ev = session_time.strftime("%Y-%m-%d") if session_time else "unknown"
            for subj, pred, obj in triples:
                event_text = f"{subj} {pred} {obj}"
                event_date = session_time.date() if session_time else None
                self._chronos_event_idx.add_event(
                    text=event_text,
                    chunk_id=ms_idx,
                    start_date=event_date,
                    session_id=session_date_str_ev,
                )

        store._store_count += 1

        # Rebuild FAISS after chunk
        engine.rebuild_faiss_if_dirty()

        return "Memorized"

    def _store_conversation_turns(self, message: str) -> str:
        """Parse and store individual conversation turns for LongMemEval data.

        Each turn gets its own embedding for fine-grained retrieval.
        This is the LME fast-path — bypasses IngestionPipeline segmentation
        since conversation turns are already natural units.
        """
        from mabench.conversation_parser import parse_chunk
        from mabench.temporal import parse_chat_time

        turns, self._conv_turn_counter, self._conv_last_timestamp = parse_chunk(
            message,
            turn_counter=self._conv_turn_counter,
            last_timestamp=self._conv_last_timestamp,
        )

        if not turns:
            return "Memorized"

        # Build turn_entries: (turn_text, role, raw_content)
        turn_entries: list[tuple[str, str, str]] = []
        for turn in turns:
            prefix = f"[{turn.timestamp}] " if turn.timestamp else ""
            role_tag = f"({turn.role}) " if turn.role != "continuation" else ""
            turn_text = f"{prefix}{role_tag}{turn.content}"
            turn_entries.append((turn_text, turn.role, turn.content))

        if not turn_entries:
            return "Memorized"

        # Parse session_time from raw message (always-on temporal)
        session_time = parse_chat_time(message)
        if session_time is not None:
            self.store._last_session_time = session_time
        else:
            session_time = getattr(self.store, "_last_session_time", None)

        # Batch-embed all turns at once (single GPU pass)
        turn_texts = [t for t, _, _ in turn_entries]
        embeddings = self._scorer.embed_batch(turn_texts)

        engine = self.store._resolve_engine
        store = self.store

        # Import entity extraction functions (regex-based, ~1ms per turn)
        from mabench.entity_index import (
            extract_entities, extract_bigrams, extract_content_unigrams,
        )
        from mabench.event_calendar import (
            parse_turn_date, extract_event_keywords, extract_inline_dates,
        )

        for (turn_text, role, _raw), emb in zip(turn_entries, embeddings):
            # Store via V1 orchestrator (when cooc_boost inactive)
            if not store._resolve_cooc_boost:
                store._resolve_orchestrator.store(content=turn_text, embedding=emb)

            # Store via V2 coupled engine
            ms_idx = engine.store(
                text=turn_text, embedding=emb,
                recency=float(store._store_count),
                role=role, session_time=session_time,
            )
            store._resolve_text_to_store_idx[turn_text] = ms_idx

            # Triadic: store with empty triples (same as monolithic)
            triadic = getattr(store, '_triadic', None)
            if triadic is not None:
                triadic.store_fact(turn_text, [])

            # BM25: index individual turns
            bm25 = getattr(store, '_bm25_index', None)
            if bm25 is not None:
                bm25.add(ms_idx, turn_text)

            # Entity inverted indexes (always-on, ~1ms per turn)
            for ent in extract_entities(turn_text):
                store._entity_inv_idx.setdefault(ent, set()).add(ms_idx)
            for bg in extract_bigrams(turn_text):
                store._bigram_inv_idx.setdefault(bg, set()).add(ms_idx)
            for ug in extract_content_unigrams(turn_text):
                store._unigram_inv_idx.setdefault(ug, set()).add(ms_idx)

            # Noun-chunk inverted index (when enabled)
            nc_inv = getattr(store, "_noun_chunk_inv_idx", None)
            if nc_inv is not None and nc_inv is not ...:
                from mabench.noun_chunk_index import extract_noun_chunks
                for nc in extract_noun_chunks(turn_text):
                    nc_inv.setdefault(nc, set()).add(ms_idx)
                    if " " in nc:
                        for word in nc.split():
                            if len(word) > 2:
                                nc_inv.setdefault(word, set()).add(ms_idx)

            # Event calendar (temporal resolution — parse date + keywords)
            turn_date = parse_turn_date(turn_text)
            if turn_date is not None:
                keywords = extract_event_keywords(turn_text)
                inline_dates = extract_inline_dates(turn_text, turn_date)
                store._event_calendar.append({
                    "turn_idx": ms_idx,
                    "date": turn_date,
                    "keywords": keywords,
                    "inline_dates": inline_dates,
                    "text": turn_text,
                })

            # LME date-mention calendar: extracts all dates mentioned in the
            # turn (absolute and relative) using the turn's own timestamp as
            # the dateparser anchor. Only active when --event_calendar is on.
            cal_lme = getattr(store, "_event_calendar_lme", None)
            if cal_lme is not None and turn_date is not None:
                cal_lme.ingest_turn(ms_idx, turn_text, turn_date)

            # LFM SVO extraction: supplement regex entity index with LFM-extracted subjects
            global_turn_idx = store._store_count  # monotonic turn counter
            lfm_triples = None

            # Try SVO cache first (pre-extracted, fast)
            if self._svo_cache is not None:
                lfm_triples = self._svo_cache.get(global_turn_idx)

            # Live extraction fallback (slow, ~1.8s per turn)
            if lfm_triples is None and self._lfm_extractor is not None:
                try:
                    lfm_triples = self._lfm_extractor.extract(turn_text)
                except Exception:
                    pass

            # Add LFM subjects to entity inverted index
            if lfm_triples:
                for subj, verb, obj in lfm_triples:
                    subj_lower = subj.lower().strip()
                    if subj_lower and len(subj_lower) >= 2:
                        store._entity_inv_idx.setdefault(subj_lower, set()).add(ms_idx)
                        # Also add individual words from multi-word subjects
                        for word in subj_lower.split():
                            if len(word) >= 4:
                                store._unigram_inv_idx.setdefault(word, set()).add(ms_idx)
                    # Also index the object if it's a named entity (not a generic description)
                    obj_lower = obj.lower().strip()
                    if obj_lower and len(obj_lower) >= 3 and len(obj_lower) <= 40:
                        store._entity_inv_idx.setdefault(obj_lower, set()).add(ms_idx)

            # Structured fact store: index SVO triples for structured retrieval
            if self._structured_store is not None and lfm_triples:
                self._structured_store.add_batch(
                    lfm_triples, turn_idx=ms_idx,
                    turn_text=turn_text,
                    timestamp=turn_date,
                )

            store._store_count += 1

        # LLM triple extraction -> subject_index for KG hop
        llm_ext = getattr(
            getattr(store, '_ingestion', None), '_llm_extractor', None,
        )
        ss = getattr(store, '_supersession_state', None)
        if llm_ext is not None and ss is not None:
            try:
                raw_contents = [raw for _, _, raw in turn_entries]
                batch_triples = llm_ext.extract_batch(raw_contents, mode="conversation")
                for (turn_text, _, _), triples in zip(turn_entries, batch_triples):
                    for subj, pred, obj in triples:
                        subj_key = subj.lower().strip()
                        pred_key = pred.lower().strip()
                        if not subj_key or not pred_key or len(subj_key) < 3:
                            continue
                        if subj_key not in ss.subject_index:
                            ss.subject_index[subj_key] = {}
                        ss.subject_index[subj_key][pred_key] = (obj, turn_text)
                        obj_key = obj.lower().strip()
                        if obj_key and len(obj_key) >= 3:
                            if obj_key not in ss.subject_index:
                                ss.subject_index[obj_key] = {}
                            ss.subject_index[obj_key][pred_key] = (subj, turn_text)
                # Also populate structured store from LLM triples
                if self._structured_store is not None:
                    for (turn_text, _, _), triples in zip(turn_entries, batch_triples):
                        if triples:
                            ms_idx = store._resolve_text_to_store_idx.get(turn_text)
                            if ms_idx is not None:
                                self._structured_store.add_batch(
                                    triples, turn_idx=ms_idx,
                                    turn_text=turn_text,
                                )
                # Populate EventDB from LLM triples (structured_primary path)
                if self._event_db is not None:
                    event_times = llm_ext.last_event_times
                    for idx, ((turn_text, _, _), triples) in enumerate(
                        zip(turn_entries, batch_triples)
                    ):
                        if triples:
                            ms_idx = store._resolve_text_to_store_idx.get(turn_text)
                            if ms_idx is not None:
                                et = event_times[idx] if idx < len(event_times) else None
                                self._event_db.add_batch(
                                    triples,
                                    chunk_id=ms_idx,
                                    event_date=session_time.date() if session_time else None,
                                    event_times=et,
                                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"LLM triple extraction failed: {e}")

        # Rebuild FAISS after chunk (same as store._store_post_chunk)
        engine.rebuild_faiss_if_dirty()

        # Flush triadic tensor writes
        triadic = getattr(store, '_triadic', None)
        if triadic is not None:
            triadic.flush()

        return "Memorized"

    def reset(self) -> None:
        """Reset all memory state for a new context."""
        self._scorer._cache.clear()
        self._conv_turn_counter = 0
        self._conv_last_timestamp = ""
        self._conv_mode_detected = False
        self._build()

    def save_agent(self, path: str) -> None:
        """Persist agent state (placeholder — state is in-memory)."""

    def load_agent(self, path: str) -> None:
        """Load agent state (placeholder — state is in-memory)."""
