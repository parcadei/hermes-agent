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
from mabench.hyperedge_intersection import HyperedgeIntersection
from mabench.knowledge_graph import KnowledgeGraph
from mabench.gliner2_extractor import GLiNER2Extractor
from mabench.cooccurrence_extractor import CooccurrenceExtractor
from mabench.relik_extractor import RelikExtractor


# Default V1 orchestrator params (validated in test_engine.py)
_V1_PARAMS = ParameterSet(
    alpha=0.1, beta=0.1, delta_t=1.0, s_max=10.0, s0=1.0,
    temperature=5.0, novelty_start=0.5, novelty_decay=0.2,
    survival_threshold=0.05, feedback_sensitivity=0.1,
    w1=0.35, w2=0.25, w3=0.20, w4=0.20,
)


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
        dim: int = 1024,
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
        # Query mode
        iterative: bool = False,
        iterative_max_hops: int = 3,
        # LLM triple extraction (vLLM server URL, None = disabled)
        llm_extraction_base_url: str | None = None,
        llm_extraction_model: str = "qwen3-triple-extractor",
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
        self._iterative_flag = iterative
        self._iterative_max_hops = iterative_max_hops
        self._llm_extraction_base_url = llm_extraction_base_url
        self._llm_extraction_model = llm_extraction_model

        # -- Persistent resources (survive reset) --------------------------
        self._scorer = EmbeddingRelevanceScorer()
        try:
            import torch
            _dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _dev = "cpu"
        self._ingestion = IngestionPipeline(
            IngestionConfig(
                coref_device=_dev, coref_enabled=True,
                boundary_threshold=0.2, strip_numbered_prefix=False,
                llm_extraction_base_url=llm_extraction_base_url,
                llm_extraction_model=llm_extraction_model,
            )
        )

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
        bm25 = BM25Index() if self._bm25_hops else None
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
        outgester = Outgester(self._llm_cfg)
        ent_reranker = (EntityReranker(
            self._scorer,
            nlp=getattr(self._ingestion, "_nlp", None),
        ) if self._retrieval_cfg.entity_rerank else None)

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
            return self.store._store(message)
        elif self._iterative_flag:
            return self.iterative._iterative_query(message)
        else:
            return self.query_engine._query(message)

    def reset(self) -> None:
        """Reset all memory state for a new context."""
        self._scorer._cache.clear()
        self._build()

    def save_agent(self, path: str) -> None:
        """Persist agent state (placeholder — state is in-memory)."""

    def load_agent(self, path: str) -> None:
        """Load agent state (placeholder — state is in-memory)."""
