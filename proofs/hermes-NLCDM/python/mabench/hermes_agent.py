"""Hermes Memory Agent — MABench adapter.

Wires MemoryOrchestrator (V1 text-based contradiction) with CoupledEngine
(V2 cosine-threshold contradiction) into the MABench AgentWrapper interface.

Pipeline:
  store: text → V1 detect_contradictions → orchestrator.store() → embed → V2 cosine check
  query: hybrid switching per HybridBridge.lean proven criteria:
         1. V2 cosine retrieval (always primary)
         2. Estimate signal coherence across candidates
         3. If coherence ≤ 0 (signals uncorrelated): pure cosine + V1 text merge
         4. If coherence > 0 (signals correlated): multi-signal re-ranking

  For FactConsolidation: signalInversion ≈ 0 → pure_cosine_maximal_gap applies
  → cosine-only is optimal (proven in HybridBridge.lean).

Two contradiction layers:
  V1 (text):    subject extraction, polarity, value_update — high recall, no embeddings
  V2 (cosine):  cosine threshold on Qwen3-0.6B embeddings — high precision, geometric
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add hermes-memory and hermes-NLCDM to path
# mabench/ lives at proofs/hermes-NLCDM/python/mabench/
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
from mabench.ingestion import (
    IngestionPipeline,
    IngestionConfig,
    _MutableEntityGraph,
)
from mabench.belief import BeliefIndex
from mabench.bm25_index import BM25Index
from mabench.chain_cache import ChainCache
from mabench.triadic_memory import TriadicMemory
from mabench.scoring import EmbeddingRelevanceScorer
from mabench.entity import EntityMixin
from mabench.rebel import RebelKGMixin
from mabench.retrieval import RetrievalMixin
from mabench.conflict_resolution import (
    ConflictResolutionMixin,
    _extract_acl_triples,  # noqa: F401 — re-exported for external consumers
    _cr_log,  # noqa: F401 — re-exported for external consumers
)
from mabench.outgestion import OutgestionMixin
from mabench.store import StoreMixin
from mabench.query import QueryMixin
from mabench.query_pipeline import QueryPipelineMixin
from mabench.iterative_query import IterativeQueryMixin


class HermesMemoryAgent(EntityMixin, RebelKGMixin, RetrievalMixin, ConflictResolutionMixin, OutgestionMixin, StoreMixin, QueryMixin, QueryPipelineMixin, IterativeQueryMixin):
    """MABench-compatible memory agent wrapping Hermes V1+V2 systems.

    Not a subclass of AgentWrapper — instead provides the send_message
    interface that MABench's main.py calls, plus save_agent/load_agent.

    Args:
        model:                   LLM model name for answer generation
        dim:                     Embedding dimension (768 for all-mpnet-base-v2)
        contradiction_threshold: V2 cosine threshold for contradiction detection
        retrieve_num:            Number of memories to retrieve for context
        dream_interval:          Run dream cycle every N stores (0 = disabled)
        temperature:             LLM temperature for answer generation
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        dim: int = 768,
        contradiction_threshold: float = 0.95,
        retrieve_num: int = 10,
        dream_interval: int = 0,
        temperature: float = 0.0,
        max_gen_tokens: int = 64,
        recency_alpha: float = 0.1,
        dream_params: DreamParams | None = None,
        associative_retrieval: bool = False,
        sparse_retrieval: bool = False,
        hybrid_retrieval: bool = False,
        cooc_boost_retrieval: bool = False,
        cooc_weight: float = 0.3,
        cooc_gate_threshold: float = 0.0,
        ppr_retrieval: bool = False,
        ppr_weight: float = 0.5,
        ppr_damping: float = 0.85,
        coretrieval_retrieval: bool = False,
        coretrieval_bonus: float = 0.3,
        coretrieval_min_count: float = 2.0,
        transfer_retrieval: bool = False,
        transfer_k: int | None = None,
        triadic_retrieval: bool = False,
        triadic_n: int = 1000,
        triadic_p: int = 10,
        triadic_expand_k: int = 5,
        triadic_chain_hops: int = 1,
        triadic_decay: float = 1.0,
        decompose_query: bool = False,
        decompose_max_hops: int = 3,
        decompose_coverage: bool = False,
        decompose_rrf: bool = False,
        iterative_query: bool = False,
        iterative_max_hops: int = 3,
        bm25_weight: float = 0.0,
        dedup_threshold: float = 0.0,
        dream_weight: float = 0.0,
        temporal_context: bool = False,
        contradiction_context: bool = False,
        contradiction_sim_threshold: float = 0.85,
        supersession: bool = False,
        belief: bool = False,
        belief_prior_alpha: float = 2.0,
        belief_prior_beta: float = 1.0,
        belief_propagation_damping: float = 0.3,
        belief_hard_floor: float = 0.05,
        belief_overfetch: int = 3,
        conflict_resolution: bool = False,
        cr_per_hop: bool = True,
        cr_entity_gate: bool = False,
        cr_overfetch: int = 3,
        cosine_dedup: bool = False,
        cosine_dedup_threshold: float = 0.95,
        entity_retrieval: bool = False,
        entity_first: bool = False,
        entity_first_aggressive: bool = False,
        bm25_hops: bool = False,
        chain_cache: bool = False,
        rebel_kg: bool = False,
    ):
        self.model = model
        self.dedup_threshold = dedup_threshold
        self.dim = dim
        self.contradiction_threshold = contradiction_threshold
        self.associative_retrieval = associative_retrieval
        self.sparse_retrieval = sparse_retrieval
        self.hybrid_retrieval = hybrid_retrieval
        self.cooc_boost_retrieval = cooc_boost_retrieval
        self.cooc_weight = cooc_weight
        self.cooc_gate_threshold = cooc_gate_threshold
        self.bm25_weight = bm25_weight
        self.ppr_retrieval = ppr_retrieval
        self.ppr_weight = ppr_weight
        self.ppr_damping = ppr_damping
        self.coretrieval_retrieval = coretrieval_retrieval
        self.coretrieval_bonus = coretrieval_bonus
        self.coretrieval_min_count = coretrieval_min_count
        self.transfer_retrieval = transfer_retrieval
        self.transfer_k = transfer_k
        self.triadic_retrieval = triadic_retrieval
        self._triadic_expand_k = triadic_expand_k
        self._triadic_chain_hops = triadic_chain_hops
        self._triadic_decay = triadic_decay
        self.decompose_query = decompose_query
        self.decompose_max_hops = decompose_max_hops
        self.decompose_coverage = decompose_coverage
        self.decompose_rrf = decompose_rrf
        self.iterative_query = iterative_query
        self.iterative_max_hops = iterative_max_hops
        self.retrieve_num = retrieve_num
        self.dream_interval = dream_interval
        self.temperature = temperature
        self._max_gen_tokens = max_gen_tokens
        self.dream_params = dream_params
        self.dream_weight = dream_weight
        self.temporal_context = temporal_context
        self.contradiction_context = contradiction_context
        self.contradiction_sim_threshold = contradiction_sim_threshold
        self.supersession = supersession
        self.conflict_resolution = conflict_resolution
        self._cr_per_hop = cr_per_hop
        self._cr_entity_gate = cr_entity_gate
        self._cr_overfetch = cr_overfetch
        self.cosine_dedup = cosine_dedup
        self._cosine_dedup_threshold = cosine_dedup_threshold
        self.entity_retrieval = entity_retrieval
        self.entity_first = entity_first
        self.entity_first_aggressive = entity_first_aggressive
        self.bm25_hops = bm25_hops
        self.chain_cache = chain_cache
        self._bm25_index = BM25Index() if bm25_hops else None
        self._chain_cache = ChainCache(similarity_threshold=0.95) if chain_cache else None
        self.rebel_kg = rebel_kg
        self._belief_overfetch = belief_overfetch
        self._belief_hard_floor = belief_hard_floor
        self._belief_propagation_damping = belief_propagation_damping

        # Bayesian belief scoring: soft fact currency
        if belief:
            self._belief_index = BeliefIndex(
                prior_alpha=belief_prior_alpha,
                prior_beta=belief_prior_beta,
                propagation_damping=belief_propagation_damping,
                hard_floor=belief_hard_floor,
            )
        else:
            self._belief_index = None

        # Supersession: entity-gated triple-based conflict detection
        # Maps (normalized_subject, normalized_predicate) → [(normalized_object, fact_text)]
        # When a new triple has same S+P but different O AND the facts share at
        # least one named entity, the old fact is superseded.
        self._triple_index: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        self._superseded_texts: set[str] = set()
        self._fact_entities: dict[str, tuple[str, ...]] = {}  # fact_text → entities
        # Conflict resolution: directed supersession graph
        # old_text → new_text (the text that supersedes it)
        self._supersession_graph: dict[str, str] = {}
        # new_text → {old_text_1, old_text_2, ...}
        self._supersedes: dict[str, set[str]] = defaultdict(set)
        self._text_to_store_idx: dict[str, int] = {}  # fact text → memory_store index (for supersession replacement)
        self._supersession_log: list[dict] = []  # bitemporal audit trail of replaced facts

        # V1: Text-based orchestrator (validated baseline from test_engine.py)
        params = ParameterSet(
            alpha=0.1,
            beta=0.1,
            delta_t=1.0,
            s_max=10.0,
            s0=1.0,
            temperature=5.0,
            novelty_start=0.5,
            novelty_decay=0.2,
            survival_threshold=0.05,
            feedback_sensitivity=0.1,
            w1=0.35,
            w2=0.25,
            w3=0.20,
            w4=0.20,
        )
        self._scorer = EmbeddingRelevanceScorer()
        self.orchestrator = MemoryOrchestrator(
            params=params,
            encoding_config=EncodingConfig(),
            contradiction_config=ContradictionConfig(),
            consolidation_config=ConsolidationConfig(),
            recall_config=RecallConfig(total_budget=4000),
            relevance_scorer=self._scorer,
        )

        # V2: Full coupled pipeline — P@1 0.938 longitudinal config
        # Strength decay drives importances (not uniform 0.5)
        # Novelty bonus ON, emotional tagging OFF (CMA-ES proved ceiling)
        # Reconsolidation OFF (no measurable effect)
        self.coupled_engine = CoupledEngine(
            dim=dim,
            contradiction_aware=False,  # cosine > 0.95 fires on template siblings (cos=0.976) — overwrites wrong entities
            contradiction_threshold=contradiction_threshold,
            novelty_N0=0.2,
            novelty_gamma=0.05,
            emotional_tagging=False,
            reconsolidation=False,
            recency_alpha=recency_alpha,
            dream_params=dream_params,
            dedup_threshold=dedup_threshold,
        )

        # Ingestion pipeline (replaces regex _parse_facts)
        try:
            import torch

            _coref_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _coref_device = "cpu"
        self._ingestion = IngestionPipeline(
            IngestionConfig(
                coref_device=_coref_device,
                coref_enabled=True,
                boundary_threshold=0.2,
                strip_numbered_prefix=False,
            )
        )
        self._entity_accumulator = _MutableEntityGraph()

        # Triadic memory: structural recall via Overmann sparse 3D tensor
        self._triadic = TriadicMemory(n=triadic_n, p=triadic_p) if triadic_retrieval else None

        self._store_count = 0
        self._start_time = time.time()

        # GLiNER entity extractor (lazy init)
        self._gliner_model = None
        self._gliner_labels = ["person", "organization", "location", "country", "city"]

        # GLiNER entity index: entity_name.lower() → set of fact indices
        # Built during ingestion, used for entity-aware retrieval at query time.
        # Fact index = position in coupled_engine.memory_store.
        self._gliner_entity_index: dict[str, set[int]] = defaultdict(set)

        # REBEL KG: dict-of-dicts knowledge graph built at ingest time.
        # Forward: kg[subject_lower][relation_lower] = [original-case objects]
        # Reverse: kg_rev[object_lower][relation_lower] = [original-case subjects]
        # Query-time: graph hop bypasses retrieval+LLM for structured lookups.
        if rebel_kg:
            self._rebel_kg: dict[str, dict[str, list[str]]] = {}
            self._rebel_kg_reverse: dict[str, dict[str, list[str]]] = {}
            self._rebel_model = None
            self._rebel_tokenizer = None
            self._rebel_device = None
        else:
            self._rebel_kg = None
            self._rebel_kg_reverse = None

        # LLM client (lazy init)
        self._llm_client = None

    def _ensure_llm_client(self):
        if self._llm_client is not None:
            return
        from openai import OpenAI

        self._llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def send_message(
        self,
        message: str,
        memorizing: bool = False,
        query_id: int | None = None,
        context_id: int | None = None,
    ) -> dict | str:
        """MABench-compatible interface.

        memorizing=True:  Store the chunk into memory. Returns "Memorized".
        memorizing=False: Query memory, build context, generate answer via LLM.
                          Returns standard response dict.
        """
        if memorizing:
            return self._store(message)
        elif self.iterative_query:
            return self._iterative_query(message)
        else:
            return self._query(message)

    def save_agent(self, path: str) -> None:
        """Persist agent state (placeholder — hermes state is in-memory)."""
        pass

    def load_agent(self, path: str) -> None:
        """Load agent state (placeholder — hermes state is in-memory)."""
        pass

    def reset(self) -> None:
        """Reset all memory state for a new context."""
        self.orchestrator = MemoryOrchestrator(
            params=self.orchestrator.params,
            encoding_config=EncodingConfig(),
            contradiction_config=ContradictionConfig(),
            consolidation_config=ConsolidationConfig(),
            recall_config=RecallConfig(total_budget=4000),
            relevance_scorer=self._scorer,
        )
        self.coupled_engine = CoupledEngine(
            dim=self.dim,
            contradiction_aware=False,
            contradiction_threshold=self.contradiction_threshold,
            novelty_N0=0.2,
            novelty_gamma=0.05,
            emotional_tagging=False,
            reconsolidation=False,
            recency_alpha=self.coupled_engine.recency_alpha,
            dream_params=self.dream_params,
            dedup_threshold=self.dedup_threshold,
        )
        self._store_count = 0
        self._scorer._cache.clear()

        # Reset BM25 index and chain cache
        if getattr(self, "_bm25_index", None) is not None:
            self._bm25_index = BM25Index()
        if getattr(self, "_chain_cache", None) is not None:
            self._chain_cache = ChainCache(similarity_threshold=0.95)

        # Reset triadic memory (fresh tensor, codebook, mappings)
        if getattr(self, "_triadic", None) is not None:
            self._triadic = TriadicMemory(
                n=self._triadic.n,
                p=self._triadic.p,
            )

        # Reset entity accumulator
        if hasattr(self, "_entity_accumulator"):
            self._entity_accumulator = _MutableEntityGraph()

        # Reset supersession state
        self._triple_index = defaultdict(list)
        self._superseded_texts = set()
        self._fact_entities = {}
        self._supersession_graph = {}
        self._supersedes = defaultdict(set)
        self._text_to_store_idx = {}
        self._supersession_log = []

        # Reset GLiNER entity index
        self._gliner_entity_index = defaultdict(set)

        # Reset REBEL KG
        if getattr(self, "_rebel_kg", None) is not None:
            self._rebel_kg = {}
            self._rebel_kg_reverse = {}

        # Reset belief index (preserve params, clear beliefs)
        if self._belief_index is not None:
            self._belief_index = BeliefIndex(
                prior_alpha=self._belief_index._prior_alpha,
                prior_beta=self._belief_index._prior_beta,
                propagation_damping=self._belief_index._propagation_damping,
                hard_floor=self._belief_index._hard_floor,
            )
