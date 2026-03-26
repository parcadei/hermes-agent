"""Configuration and shared state for Hermes memory modules.

Each module takes a frozen config dataclass + explicit deps.
Shared mutable state lives in container classes that modules
read/write through explicit references (no self.* magic).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Module configs (frozen — safe to share, serialize, diff)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    """Controls which retrieval backend runs and with what params."""

    # retrieve_num: int = 10  # e.g. 10
    retrieve_num: int = 10
    # cooc_boost: bool — use co-occurrence boosted cosine
    cooc_boost: bool = False
    cooc_weight: float = 0.3       # e.g. 0.3 (cooc contribution)
    cooc_gate_threshold: float = 0.0
    # ppr: bool — use personalized PageRank reranking
    ppr: bool = False
    ppr_weight: float = 0.5
    ppr_damping: float = 0.85
    # bm25_weight: float — hybrid cosine+bm25 weight (0 = no bm25)
    bm25_weight: float = 0.0
    dream_weight: float = 0.0
    # backend switches (mutually exclusive groups)
    associative: bool = False
    sparse: bool = False
    hybrid: bool = False
    coretrieval: bool = False
    coretrieval_bonus: float = 0.3
    coretrieval_min_count: float = 2.0
    transfer: bool = False
    transfer_k: int | None = None


@dataclass(frozen=True)
class CRConfig:
    """Conflict resolution settings."""

    # enabled: bool — master switch for query-time CR
    enabled: bool = False
    # per_hop: bool — apply CR before each intermediate answer extraction
    per_hop: bool = True
    entity_gate: bool = False
    # overfetch: int — multiply retrieve_num by this before CR filters
    overfetch: int = 3


@dataclass(frozen=True)
class DecomposeConfig:
    """Query decomposition (multi-hop chaining) settings."""

    # enabled: bool — decompose complex questions into sub-queries
    enabled: bool = False
    max_hops: int = 3
    coverage: bool = False
    rrf: bool = False


@dataclass(frozen=True)
class StoreConfig:
    """Ingestion pipeline settings."""

    # supersession: bool — entity-gated triple-based conflict detection
    supersession: bool = False
    # cooc_boost: bool — co-occurrence matrix maintained during ingest
    cooc_boost: bool = False
    # dream_interval: int — run dream cycle every N stores (0 = disabled)
    dream_interval: int = 0


@dataclass(frozen=True)
class LLMConfig:
    """LLM client settings for answer generation."""

    # model: str — e.g. "gpt-4o-mini" (no provider prefix for OpenRouter)
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_gen_tokens: int = 64


@dataclass(frozen=True)
class BeliefConfig:
    """Bayesian belief scoring settings."""

    enabled: bool = False
    prior_alpha: float = 2.0
    prior_beta: float = 1.0
    propagation_damping: float = 0.3
    hard_floor: float = 0.05
    overfetch: int = 3


@dataclass(frozen=True)
class EntityConfig:
    """GLiNER entity extraction and retrieval settings."""

    # entity_retrieval: bool — supplement retrieval with entity index matches
    entity_retrieval: bool = False
    entity_first: bool = False
    entity_first_aggressive: bool = False
    labels: tuple[str, ...] = ("person", "organization", "location", "country", "city")


@dataclass(frozen=True)
class DedupConfig:
    """Cosine deduplication settings."""

    enabled: bool = False
    threshold: float = 0.95


# ---------------------------------------------------------------------------
# Shared mutable state containers
# ---------------------------------------------------------------------------

@dataclass
class SupersessionState:
    """Tracks which facts supersede which via (S,P) triple matching.

    Written by FactStore during ingestion.
    Read by ConflictResolver and QueryEngine at query time.
    """

    # triple_index: (norm_subj, norm_pred) -> [(norm_obj, fact_text), ...]
    # e.g. {("abbiati", "sport_played_by"): [("football", "The sport played by Abbiati is football")]}
    triple_index: dict[tuple[str, str], list[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # superseded_texts: set of fact texts that have been superseded by newer facts
    # e.g. {"The sport played by Abbiati is basketball"}
    superseded_texts: set[str] = field(default_factory=set)
    # fact_entities: fact_text -> tuple of entity strings extracted from that fact
    # e.g. {"The sport played by Abbiati is football": ("Abbiati",)}
    fact_entities: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # supersession_graph: old_text -> new_text (the fact that replaces it)
    # e.g. {"Abbiati plays basketball": "Abbiati plays football"}
    supersession_graph: dict[str, str] = field(default_factory=dict)
    # supersedes: new_text -> {old_text_1, old_text_2, ...}
    supersedes: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # text_to_store_idx: fact text -> index in coupled_engine.memory_store
    # e.g. {"The sport played by Abbiati is football": 42}
    text_to_store_idx: dict[str, int] = field(default_factory=dict)
    # supersession_log: bitemporal audit trail
    # e.g. [{"old_text": "...", "new_text": "...", "timestamp": 1234567890}]
    supersession_log: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        """Clear all supersession state for a new context."""
        self.triple_index = defaultdict(list)
        self.superseded_texts = set()
        self.fact_entities = {}
        self.supersession_graph = {}
        self.supersedes = defaultdict(set)
        self.text_to_store_idx = {}
        self.supersession_log = []


@dataclass
class EntityIndex:
    """GLiNER entity inverted index: entity_name -> set of fact indices.

    Written by FactStore during ingestion.
    Read by EntityOps at query time.
    """

    # index: entity_name.lower() -> set of memory_store indices
    # e.g. {"abbiati": {0, 5, 12}, "milan": {3, 7}}
    index: dict[str, set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def reset(self) -> None:
        self.index = defaultdict(set)
