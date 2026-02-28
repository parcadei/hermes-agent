"""Consolidation/aging pipeline for the Hermes memory system.

Implements the episodic-to-semantic memory aging pipeline: detecting when
episodic memories have decayed past usefulness but still contain valuable
semantic content, extracting that content into higher-level representations,
and archiving the originals.  This mirrors the hippocampal-to-cortical
consolidation process from cognitive science.

This module sits downstream of encoding (piece 1), recall (piece 2), and
contradiction (piece 3).  After memories are encoded and have lived through
dynamics cycles, consolidation evaluates which episodic memories should be
promoted to semantic representations.

Dependencies: stdlib only (dataclasses, enum, logging, math, re).
"""

from __future__ import annotations

import enum
import logging
import math
import re
from dataclasses import dataclass, field

from hermes_memory.core import clamp01, retention
from hermes_memory.encoding import CATEGORY_IMPORTANCE, VALID_CATEGORIES

logger = logging.getLogger(__name__)

# ============================================================
# Enums
# ============================================================


class ConsolidationLevel(enum.IntEnum):
    """Abstraction level of a memory in the consolidation hierarchy.

    Each level has a characteristic time constant (1/e-life) and detail
    granularity.  Consolidation promotes memories from lower to higher
    levels (EPISODIC_RAW -> EPISODIC_COMPRESSED -> SEMANTIC_FACT ->
    SEMANTIC_INSIGHT).

    Spec divergence (M1): the spec uses ``enum.Enum`` but this
    implementation uses ``enum.IntEnum`` to support comparison
    operators (< > <= >=).  Integer ordering is used by
    ``_next_level()`` and callers that test ``level < target``.
    """

    EPISODIC_RAW = 1
    EPISODIC_COMPRESSED = 2
    SEMANTIC_FACT = 3
    SEMANTIC_INSIGHT = 4


class ConsolidationMode(enum.Enum):
    """Trigger context for a consolidation operation."""

    INTRA_SESSION = "intra_session"
    ASYNC_BATCH = "async_batch"
    RETRIEVAL_TRIGGERED = "retrieval_triggered"


# ============================================================
# Constants
# ============================================================

LEVEL_TIME_CONSTANTS: dict[ConsolidationLevel, float] = {
    ConsolidationLevel.EPISODIC_RAW: 7.0,
    ConsolidationLevel.EPISODIC_COMPRESSED: 30.0,
    ConsolidationLevel.SEMANTIC_FACT: 180.0,
    ConsolidationLevel.SEMANTIC_INSIGHT: 365.0,
}

LEVEL_HALF_LIVES: dict[ConsolidationLevel, float] = {
    level: tc * math.log(2) for level, tc in LEVEL_TIME_CONSTANTS.items()
}

# Retention thresholds for level promotion.  SEMANTIC_FACT and SEMANTIC_INSIGHT
# are excluded: L3->L4 promotion uses cluster similarity (not retention decay),
# and SEMANTIC_INSIGHT has no further promotion path.
LEVEL_RETENTION_THRESHOLDS: dict[ConsolidationLevel, float] = {
    ConsolidationLevel.EPISODIC_RAW: 0.4,
    ConsolidationLevel.EPISODIC_COMPRESSED: 0.3,
}

# Target compression ratio (source length / extracted length) per category.
CATEGORY_COMPRESSION_RATIOS: dict[str, int] = {
    "correction": 1,
    "preference": 10,
    "fact": 5,
    "reasoning": 3,
    "instruction": 5,
    "greeting": 10,
    "transactional": 10,
    "unclassified": 5,
}

# Categories that never participate in consolidation.  Parallel to
# contradiction.py's EXCLUDED_CATEGORIES but defined independently since
# the exclusion reasons differ (greeting/transactional are structurally
# low-value for consolidation rather than semantically non-comparable).
NON_CONSOLIDATABLE_CATEGORIES: frozenset[str] = frozenset(
    {"greeting", "transactional", "correction"}
)

# Defense-in-depth markers for detecting corrections based on content
# rather than category label.
CORRECTION_CONTENT_MARKERS: frozenset[re.Pattern[str]] = frozenset(
    [
        re.compile(r"actually[,:]?\s", re.IGNORECASE),
        re.compile(r"not\s+\w+\s+but\s+", re.IGNORECASE),
        re.compile(r"i\s+was\s+wrong", re.IGNORECASE),
        re.compile(r"correction:", re.IGNORECASE),
        re.compile(
            r"(?:wrong|incorrect|mistaken)\s+(?:about|regarding)", re.IGNORECASE
        ),
        re.compile(
            r"(?:updated?|changed?)\s+(?:my\s+)?(?:mind|opinion|preference)",
            re.IGNORECASE,
        ),
    ]
)

CORRECTION_MARKER_THRESHOLD: int = 2


# ============================================================
# Extraction Patterns (per-category regex lists for extract_semantic)
# ============================================================

PREFERENCE_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:the user |i )(?:prefer|like|enjoy|love|favor)s?\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the user |i )(?:don't|do not|doesn't|does not) "
        r"(?:like|enjoy|want)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the user's |my )favorite\s+\w+\s+is\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the user |i )(?:would |'d )rather\s+(.+)",
        re.IGNORECASE,
    ),
]

FACT_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:the user |i )(?:live|reside|am based|is based|is located)s?"
        r"\s+(?:in|at)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:the user's |my )name is\s+(.+)", re.IGNORECASE),
    re.compile(
        r"(?:the user |i )(?:work|works|am employed)\s+(?:at|for|as)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the user's |my )(?:email|phone|address)\s+is\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:the user |i )(?:speak|speaks)\s+(.+)", re.IGNORECASE),
    re.compile(
        r"(?:the user |i )(?:have|has)\s+(?:a|an)\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the user |i )(?:studied|graduated from|attended)\s+(.+)",
        re.IGNORECASE,
    ),
]

INSTRUCTION_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:always|never|from now on,?\s*|remember (?:to|that)\s+)(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the user (?:instructed|requested|directed|asked)"
        r"(?: the assistant)? to\s+)(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:going forward,?\s*|in the future,?\s*)(.+)",
        re.IGNORECASE,
    ),
]

REASONING_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:because|since|due to|given that)\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(.+?)(?:therefore|thus|hence|consequently|as a result)\s+(.+)",
        re.IGNORECASE,
    ),
]

EXTRACTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "preference": PREFERENCE_EXTRACTION_PATTERNS,
    "fact": FACT_EXTRACTION_PATTERNS,
    "instruction": INSTRUCTION_EXTRACTION_PATTERNS,
    "reasoning": REASONING_EXTRACTION_PATTERNS,
}

SEMANTIC_TEMPLATES: dict[str, str] = {
    "preference": "User prefers: {extracted}",
    "fact": "User fact: {extracted}",
    "instruction": "Standing instruction: {extracted}",
    "reasoning": "User reasoning: {extracted}",
    "correction": "{original}",
    "greeting": "{extracted}",
    "transactional": "{extracted}",
}


# ============================================================
# Data Types
# ============================================================


@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration for the consolidation pipeline.

    Frozen to prevent accidental mutation.  All thresholds carry domain
    constraints validated at construction time.

    Time unit assumption: all time-based thresholds assume delta_t = 1.0
    in the dynamics engine, where 1 time step represents approximately
    1 day.  If the caller uses a different delta_t, these thresholds must
    be scaled proportionally.

    Attributes -- Trigger thresholds:
        retention_threshold:          Retention below this triggers consolidation.
                                       Domain: (0.0, 1.0).  Default: 0.3.
        semantic_value_threshold:     Semantic value must exceed this.
                                       Domain: (0.0, 1.0).  Default: 0.6.
        consolidation_window:         Minimum age (creation_time) for batch
                                       consolidation.  Domain: > 0.0.
                                       Default: 7.0 (days).
        retrieval_frequency_threshold: Access count below which a memory is
                                        consolidation-eligible (after window).
                                        Domain: >= 1.  Default: 10.
        max_pool_size:                Pool size above which storage pressure
                                       triggers.  Domain: >= 1.  Default: 5000.

    Attributes -- Inhibitor thresholds:
        recency_guard:                Minimum age before consolidation is
                                       allowed.  Domain: > 0.0.  Default: 1.0
                                       (1 day).
        high_activation_threshold:    Access count above which a memory is
                                       considered highly active (blocks
                                       consolidation if also recent).
                                       Domain: >= 1.  Default: 10.

    Attributes -- Level transition thresholds:
        l1_to_l2_retention:   Retention threshold for L1 -> L2.
                                Domain: (0.0, 1.0).  Default: 0.4.
        l2_to_l3_retention:   Retention threshold for L2 -> L3.
                                Domain: (0.0, 1.0).  Default: 0.3.
        l3_to_l4_cluster_sim: Cluster similarity threshold for L3 -> L4.
                                Domain: (0.0, 1.0].  Default: 0.6.
                                NOTE: L3->L4 promotion is via clustering,
                                not retention decay.

    Attributes -- Process parameters:
        consolidation_decay:     Decay multiplier applied to archived memory
                                  scores.  Domain: (0.0, 1.0).  Default: 0.1.
        semantic_boost:          Importance multiplier for new semantic memories.
                                  Domain: (1.0, 2.0].  Default: 1.2.
        activation_inheritance:  Fraction of source activation inherited by
                                  semantic memory.  Domain: [0.0, 1.0].
                                  Default: 0.5.

    Attributes -- Clustering (batch mode):
        semantic_weight_beta:    Weight of semantic similarity in affinity
                                  function.  Domain: [0.0, 1.0].  Default: 0.7.
                                  temporal_weight = 1.0 - semantic_weight_beta.
        temporal_decay_lambda:   Decay rate for temporal distance in affinity.
                                  Domain: > 0.0.  Default: 0.1.
        cluster_threshold:       Minimum affinity to cluster two memories.
                                  Domain: (0.0, 1.0].  Default: 0.6.
        min_content_similarity:  Minimum content similarity for clustering.
                                  If content_sim < this, affinity is forced
                                  to 0.0.  Domain: [0.0, 1.0].  Default: 0.3.

    Attributes -- Intra-session mode:
        intra_session_similarity: Similarity threshold for intra-session
                                   consolidation.  Domain: (0.0, 1.0].
                                   Default: 0.8.
        intra_session_temporal:   Maximum creation_time difference for
                                   intra-session.  Domain: > 0.0.
                                   Default: 1.0 (1 day).

    Attributes -- Storage pressure (batch mode):
        mild_pressure_retention:   Under mild storage pressure, use this
                                    relaxed retention threshold.
                                    Domain: (0.0, 1.0).  Default: 0.5.
        severe_pressure_multiplier: Pool size multiplier above which severe
                                     pressure applies.  Domain: > 1.0.
                                     Default: 2.0.

    Attributes -- General:
        max_candidates_per_batch:  Maximum candidates to process in one batch.
                                    Domain: >= 1.  Default: 100.
    """

    # Trigger thresholds
    retention_threshold: float = 0.3
    semantic_value_threshold: float = 0.6
    consolidation_window: float = 7.0
    retrieval_frequency_threshold: int = 10
    max_pool_size: int = 5000

    # Inhibitor thresholds
    recency_guard: float = 1.0
    high_activation_threshold: int = 10

    # Level transition thresholds
    l1_to_l2_retention: float = 0.4
    l2_to_l3_retention: float = 0.3
    l3_to_l4_cluster_sim: float = 0.6

    # Process parameters
    consolidation_decay: float = 0.1
    semantic_boost: float = 1.2
    activation_inheritance: float = 0.5

    # Clustering
    semantic_weight_beta: float = 0.7
    temporal_decay_lambda: float = 0.1
    cluster_threshold: float = 0.6
    min_content_similarity: float = 0.3

    # Intra-session
    intra_session_similarity: float = 0.8
    intra_session_temporal: float = 1.0

    # Storage pressure
    mild_pressure_retention: float = 0.5
    severe_pressure_multiplier: float = 2.0

    # General
    max_candidates_per_batch: int = 100

    @property
    def temporal_weight(self) -> float:
        """Complement of semantic_weight_beta for the affinity function.

        temporal_weight = 1.0 - semantic_weight_beta.  Provided for
        backward compatibility and ergonomic use in compute_affinity().
        """
        return 1.0 - self.semantic_weight_beta

    def __post_init__(self) -> None:
        """Validate all parameter constraints.

        Violations raise ValueError with a descriptive message.
        """
        # retention_threshold: (0, 1)
        if not (0.0 < self.retention_threshold < 1.0):
            raise ValueError(
                f"retention_threshold must be in (0.0, 1.0), "
                f"got {self.retention_threshold}"
            )
        # semantic_value_threshold: (0, 1)
        if not (0.0 < self.semantic_value_threshold < 1.0):
            raise ValueError(
                f"semantic_value_threshold must be in (0.0, 1.0), "
                f"got {self.semantic_value_threshold}"
            )
        # consolidation_window: > 0
        if self.consolidation_window <= 0.0:
            raise ValueError(
                f"consolidation_window must be > 0, "
                f"got {self.consolidation_window}"
            )
        # retrieval_frequency_threshold: >= 1
        if self.retrieval_frequency_threshold < 1:
            raise ValueError(
                f"retrieval_frequency_threshold must be >= 1, "
                f"got {self.retrieval_frequency_threshold}"
            )
        # max_pool_size: >= 1
        if self.max_pool_size < 1:
            raise ValueError(
                f"max_pool_size must be >= 1, got {self.max_pool_size}"
            )
        # recency_guard: > 0
        if self.recency_guard <= 0.0:
            raise ValueError(
                f"recency_guard must be > 0, got {self.recency_guard}"
            )
        # high_activation_threshold: >= 1
        if self.high_activation_threshold < 1:
            raise ValueError(
                f"high_activation_threshold must be >= 1, "
                f"got {self.high_activation_threshold}"
            )
        # l1_to_l2_retention: (0, 1)
        if not (0.0 < self.l1_to_l2_retention < 1.0):
            raise ValueError(
                f"l1_to_l2_retention must be in (0.0, 1.0), "
                f"got {self.l1_to_l2_retention}"
            )
        # l2_to_l3_retention: (0, 1)
        if not (0.0 < self.l2_to_l3_retention < 1.0):
            raise ValueError(
                f"l2_to_l3_retention must be in (0.0, 1.0), "
                f"got {self.l2_to_l3_retention}"
            )
        # l3_to_l4_cluster_sim: (0, 1]
        if not (0.0 < self.l3_to_l4_cluster_sim <= 1.0):
            raise ValueError(
                f"l3_to_l4_cluster_sim must be in (0.0, 1.0], "
                f"got {self.l3_to_l4_cluster_sim}"
            )
        # consolidation_decay: (0, 1)
        if not (0.0 < self.consolidation_decay < 1.0):
            raise ValueError(
                f"consolidation_decay must be in (0.0, 1.0), "
                f"got {self.consolidation_decay}"
            )
        # semantic_boost: (1.0, 2.0]
        if not (1.0 < self.semantic_boost <= 2.0):
            raise ValueError(
                f"semantic_boost must be in (1.0, 2.0], "
                f"got {self.semantic_boost}"
            )
        # activation_inheritance: [0.0, 1.0]
        if not (0.0 <= self.activation_inheritance <= 1.0):
            raise ValueError(
                f"activation_inheritance must be in [0.0, 1.0], "
                f"got {self.activation_inheritance}"
            )
        # semantic_weight_beta: [0, 1]
        if not (0.0 <= self.semantic_weight_beta <= 1.0):
            raise ValueError(
                f"semantic_weight_beta must be in [0.0, 1.0], "
                f"got {self.semantic_weight_beta}"
            )
        # temporal_decay_lambda: > 0
        if self.temporal_decay_lambda <= 0.0:
            raise ValueError(
                f"temporal_decay_lambda must be > 0, "
                f"got {self.temporal_decay_lambda}"
            )
        # cluster_threshold: (0, 1]
        if not (0.0 < self.cluster_threshold <= 1.0):
            raise ValueError(
                f"cluster_threshold must be in (0.0, 1.0], "
                f"got {self.cluster_threshold}"
            )
        # min_content_similarity: [0, 1]
        if not (0.0 <= self.min_content_similarity <= 1.0):
            raise ValueError(
                f"min_content_similarity must be in [0.0, 1.0], "
                f"got {self.min_content_similarity}"
            )
        # intra_session_similarity: (0, 1]
        if not (0.0 < self.intra_session_similarity <= 1.0):
            raise ValueError(
                f"intra_session_similarity must be in (0.0, 1.0], "
                f"got {self.intra_session_similarity}"
            )
        # intra_session_temporal: > 0
        if self.intra_session_temporal <= 0.0:
            raise ValueError(
                f"intra_session_temporal must be > 0, "
                f"got {self.intra_session_temporal}"
            )
        # mild_pressure_retention: (0, 1)
        if not (0.0 < self.mild_pressure_retention < 1.0):
            raise ValueError(
                f"mild_pressure_retention must be in (0.0, 1.0), "
                f"got {self.mild_pressure_retention}"
            )
        # severe_pressure_multiplier: > 1.0
        if self.severe_pressure_multiplier <= 1.0:
            raise ValueError(
                f"severe_pressure_multiplier must be > 1.0, "
                f"got {self.severe_pressure_multiplier}"
            )
        # max_candidates_per_batch: >= 1
        if self.max_candidates_per_batch < 1:
            raise ValueError(
                f"max_candidates_per_batch must be >= 1, "
                f"got {self.max_candidates_per_batch}"
            )
        # Cross-field: l2_to_l3_retention <= l1_to_l2_retention
        if self.l2_to_l3_retention > self.l1_to_l2_retention:
            raise ValueError(
                f"l2_to_l3_retention ({self.l2_to_l3_retention}) must be "
                f"<= l1_to_l2_retention ({self.l1_to_l2_retention})"
            )
        # Cross-field: mild_pressure_retention >= retention_threshold
        if self.mild_pressure_retention < self.retention_threshold:
            raise ValueError(
                f"mild_pressure_retention ({self.mild_pressure_retention}) "
                f"must be >= retention_threshold ({self.retention_threshold})"
            )


@dataclass(frozen=True)
class ConsolidationCandidate:
    """A memory being evaluated for consolidation.

    Wraps a MemoryState with additional metadata needed for the
    consolidation decision.  The caller constructs these from the memory
    store; the consolidation module does not access the store directly.

    Spec note (C4): the spec defines ``index: int`` but this codebase
    uses string identifiers throughout; ``memory_id: str`` is retained
    as an intentional divergence.

    Attributes:
        memory_id:            Unique identifier for this memory (spec: ``index``).
        content:              Full text content of the memory.
        category:             Category from encoding (one of VALID_CATEGORIES).
        level:                Current consolidation level.
        creation_time:        Time steps since memory was created.
        last_access_time:     Time steps since last access.
        access_count:         Total access count.  Domain: >= 0.
        importance:           Current importance score.  Domain: [0, 1].
        strength:             Current memory strength.  Domain: >= 0.
        relevance:            Relevance to a query.  Domain: [0, 1].
        is_contested:         True if the memory is flagged as contested.
                              Contested memories are NEVER consolidated.
        source_episodes:      Tuple of episode IDs that contributed to
                              this memory.  Empty for L1 (raw episodes are
                              their own source).
        consolidation_count:  Number of times this memory has been
                              consolidated from lower levels.  0 for L1.
        metadata:             Optional metadata dict.
    """

    memory_id: str
    content: str
    category: str
    level: ConsolidationLevel
    creation_time: float
    last_access_time: float
    access_count: int
    importance: float
    strength: float
    relevance: float
    is_contested: bool = False
    source_episodes: tuple[str, ...] = ()
    consolidation_count: int = 0
    metadata: dict | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate candidate fields."""
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(VALID_CATEGORIES)}, "
                f"got {self.category!r}"
            )
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError(
                f"importance must be in [0.0, 1.0], got {self.importance}"
            )
        if self.access_count < 0:
            raise ValueError(
                f"access_count must be >= 0, got {self.access_count}"
            )
        if self.consolidation_count < 0:
            raise ValueError(
                f"consolidation_count must be >= 0, "
                f"got {self.consolidation_count}"
            )


@dataclass(frozen=True)
class SemanticExtraction:
    """Extracted semantic knowledge from one or more episodic memories.

    Captures the core knowledge without conversational context.

    Attributes:
        content:              The extracted semantic content (text).
        category:             Category of the extracted content.
        source_episodes:      Memory IDs of source episodes (provenance).
        confidence:           Aggregate confidence from source memories.
                              Domain: [0, 1].  Computed as mean importance
                              of sources, clamped.
        first_observed:       Earliest creation_time among sources.
        last_updated:         Latest creation_time among sources.
        consolidation_count:  Number of source memories consolidated.
                              Domain: >= 1.
        compression_ratio:    Ratio of source content length to extracted
                              content length.  Domain: > 0.0.  A ratio of
                              5.0 means 5x compression.  A ratio < 1.0
                              indicates EXPANSION (template prefix made the
                              output longer than the source).
        importance:           Importance score, boosted by semantic_boost.
                              Domain: [0, 1].
        target_level:         Target consolidation level.
        extraction_mode:      The consolidation mode that produced this.
        access_count:         Inherited access count from sources, scaled by
                              activation_inheritance.  Domain: >= 0.
    """

    content: str
    category: str
    source_episodes: tuple[str, ...]
    confidence: float
    first_observed: float
    last_updated: float
    consolidation_count: int
    compression_ratio: float
    importance: float
    target_level: ConsolidationLevel
    extraction_mode: ConsolidationMode
    access_count: int = 0

    def __post_init__(self) -> None:
        """Validate extraction fields."""
        if not self.source_episodes:
            raise ValueError("source_episodes must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(VALID_CATEGORIES)}, "
                f"got {self.category!r}"
            )
        if self.first_observed > self.last_updated:
            raise ValueError(
                f"first_observed ({self.first_observed}) must be "
                f"<= last_updated ({self.last_updated})"
            )
        if self.consolidation_count < 1:
            raise ValueError(
                f"consolidation_count must be >= 1, "
                f"got {self.consolidation_count}"
            )
        if self.compression_ratio <= 0.0:
            raise ValueError(
                f"compression_ratio must be > 0.0, "
                f"got {self.compression_ratio}"
            )


@dataclass(frozen=True)
class ArchivedMemory:
    """Record of an archived episodic memory after consolidation.

    Archived memories are NOT deleted.  They remain in the memory store
    but are excluded from standard recall scoring.  Retrieval of archived
    memories requires a dedicated archive query path that bypasses
    normal scoring.

    Their scores are decayed to reduce ranking weight.  With default
    consolidation_decay=0.1, a memory with importance=0.7 and
    strength=7.0 becomes decayed_importance=0.07 and decayed_strength=0.7,
    making it effectively invisible to standard scoring.

    Spec note (C4): the spec defines ``index: int`` but this codebase
    uses string identifiers throughout; ``memory_id: str`` is retained
    as an intentional divergence.

    Attributes:
        memory_id:              Original memory identifier (spec: ``index``).
        content:                Original content (preserved in full).
        category:               Original category.
        level:                  Original consolidation level (before archival).
        archived:               Always True.  Structural marker.
        consolidated_to:        Tuple of target memory IDs that this memory
                                was consolidated into.  Must have len >= 1.
        decayed_importance:     Importance after consolidation_decay applied.
                                Domain: [0.0, 1.0].
        decayed_strength:       Strength after consolidation_decay applied.
                                Domain: >= 0.0.
        original_importance:    Importance before archival (audit trail).
        original_strength:      Strength before archival (audit trail).
        creation_time:          Original creation time (audit trail).
        archive_time:           Time of archival (audit trail).
        access_count_at_archive:  Access count at time of archival (audit trail).
    """

    memory_id: str
    content: str
    category: str
    level: ConsolidationLevel
    archived: bool
    consolidated_to: tuple[str, ...]
    decayed_importance: float
    decayed_strength: float
    original_importance: float
    original_strength: float
    creation_time: float
    archive_time: float
    access_count_at_archive: int = 0


@dataclass(frozen=True)
class ConsolidationResult:
    """Output of a consolidation operation.

    Attributes:
        extractions:             New semantic memories created.
        archived:                Episodic memories archived.
        skipped_indices:         Frozenset of candidate memory_ids that were
                                 evaluated but not consolidated (inhibitor
                                 triggered or below threshold).
        candidates_evaluated:    Total candidates considered.
        candidates_consolidated: Candidates that were consolidated.
        mode:                    The consolidation mode that produced this.
    """

    extractions: tuple[SemanticExtraction, ...]
    archived: tuple[ArchivedMemory, ...]
    skipped_indices: frozenset[str]
    candidates_evaluated: int
    candidates_consolidated: int
    mode: ConsolidationMode


# ============================================================
# Default Config
# ============================================================

DEFAULT_CONSOLIDATION_CONFIG: ConsolidationConfig = ConsolidationConfig()


# ============================================================
# Functions (Phase B)
# ============================================================


def _content_is_correction(content: str) -> bool:
    """Return True if *content* matches >= CORRECTION_MARKER_THRESHOLD patterns.

    Defense-in-depth check that detects correction-like content regardless
    of the category label.  This prevents mis-categorised corrections from
    slipping through the category-based inhibitor (I4).

    Deterministic.  Never raises.
    """
    return (
        sum(1 for p in CORRECTION_CONTENT_MARKERS if p.search(content))
        >= CORRECTION_MARKER_THRESHOLD
    )


def _passes_inhibitors(
    candidate: ConsolidationCandidate,
    config: ConsolidationConfig,
) -> bool:
    """Return True if *candidate* passes all five consolidation inhibitors.

    Inhibitors (any blocks):
        I1: Contested memory.
        I2: Creation time below recency guard.
        I3: High activation AND recent last access.
        I4: Category in NON_CONSOLIDATABLE_CATEGORIES.
        I5: Content matches correction patterns (defense-in-depth).

    Deterministic.  Never raises for valid inputs.
    """
    # I1: contested
    if candidate.is_contested:
        return False

    # I2: recency guard
    if candidate.creation_time < config.recency_guard:
        return False

    # I3: high activation + recent access
    if (
        candidate.access_count >= config.high_activation_threshold
        and candidate.last_access_time < config.recency_guard
    ):
        return False

    # I4: non-consolidatable category
    if candidate.category in NON_CONSOLIDATABLE_CATEGORIES:
        return False

    # I5: content-based correction detection
    if _content_is_correction(candidate.content):
        return False

    return True


def _compute_semantic_value(candidate: ConsolidationCandidate) -> float:
    """Compute the semantic value of a consolidation candidate.

    Formula:
        clamp01(
            CATEGORY_IMPORTANCE[category] * 0.6
            + importance * 0.3
            + min(1.0, access_count / 20.0) * 0.1
        )

    Returns a float in [0.0, 1.0].  Deterministic.
    """
    cat_weight = CATEGORY_IMPORTANCE[candidate.category] * 0.6
    imp_weight = candidate.importance * 0.3
    acc_weight = min(1.0, candidate.access_count / 20.0) * 0.1
    return clamp01(cat_weight + imp_weight + acc_weight)


def should_consolidate(
    candidate: ConsolidationCandidate,
    config: ConsolidationConfig | None = None,
) -> bool:
    """Decide whether a memory candidate should be consolidated.

    The decision has two paths:

    **Primary trigger:** The memory's retention has decayed below its level's
    threshold AND the semantic value exceeds the configured minimum.

    **Secondary trigger:** The memory is old (beyond the consolidation window),
    infrequently accessed, and still has high semantic value -- even if
    retention is above the threshold.

    Behavioral contracts:
        - Contested candidate -> always False.
        - Recent candidate (creation_time < recency_guard) -> always False.
        - Correction category or correction content -> always False.
        - SEMANTIC_INSIGHT level -> always False (no promotion path).
        - Deterministic.
    """
    config = config or DEFAULT_CONSOLIDATION_CONFIG

    if not _passes_inhibitors(candidate, config):
        return False

    safe_strength = max(candidate.strength, 1e-12)
    current_retention = retention(candidate.last_access_time, safe_strength)
    semantic_value = _compute_semantic_value(candidate)

    # Get retention threshold for candidate's level.  Only EPISODIC_RAW
    # and EPISODIC_COMPRESSED use retention-based promotion.  SEMANTIC_FACT
    # (L3->L4) is promoted via cluster similarity in batch mode, and
    # SEMANTIC_INSIGHT has no further promotion path.
    retention_threshold = LEVEL_RETENTION_THRESHOLDS.get(candidate.level)
    if retention_threshold is None:
        return False

    # Primary trigger: low retention + high semantic value
    primary = (
        current_retention < retention_threshold
        and semantic_value >= config.semantic_value_threshold
    )

    # Secondary triggers
    temporal = candidate.creation_time > config.consolidation_window
    infrequent = (
        candidate.access_count < config.retrieval_frequency_threshold
        and candidate.creation_time > config.consolidation_window
    )

    if primary:
        return True
    if temporal and infrequent and semantic_value >= config.semantic_value_threshold:
        return True

    return False


def compute_affinity(
    memory_a: ConsolidationCandidate,
    memory_b: ConsolidationCandidate,
    config: ConsolidationConfig | None = None,
) -> float:
    """Compute the affinity between two memory candidates for clustering.

    Combines token-level Jaccard content similarity with exponential
    temporal decay, applying a content-similarity floor (AP3-F2) that
    prevents temporal-only clustering.

    Behavioral contracts:
        - Symmetric: affinity(a, b) == affinity(b, a).
        - Self-affinity >= (1 - temporal_weight).
        - When content_sim < min_content_similarity -> returns 0.0.
        - Returns a float in [0.0, 1.0].
        - Deterministic.
    """
    config = config or DEFAULT_CONSOLIDATION_CONFIG

    # Token Jaccard similarity
    tokens_a = set(memory_a.content.lower().split())
    tokens_b = set(memory_b.content.lower().split())
    if not tokens_a or not tokens_b:
        content_sim = 0.0
    else:
        content_sim = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    # AP3-F2: min_content_similarity floor
    if content_sim < config.min_content_similarity:
        return 0.0

    # Temporal similarity
    time_diff = abs(memory_a.creation_time - memory_b.creation_time)
    temporal_sim = math.exp(-config.temporal_decay_lambda * time_diff)

    # Weighted combination (beta = semantic weight, alpha = temporal weight)
    affinity = (
        config.semantic_weight_beta * content_sim
        + config.temporal_weight * temporal_sim
    )
    return clamp01(affinity)


# ============================================================
# Phase C: Extraction Functions
# ============================================================


def _get_extraction_patterns(category: str) -> list[re.Pattern[str]]:
    """Return the extraction patterns for a given category.

    Returns the pattern list from EXTRACTION_PATTERNS for categories that
    have patterns (preference, fact, instruction, reasoning).  Returns an
    empty list for categories without patterns (greeting, transactional,
    correction) and for unknown categories.

    Never raises.
    """
    return EXTRACTION_PATTERNS.get(category, [])


def _truncate_to_sentences(text: str, target_length: int) -> str:
    """Truncate text to fit within target_length, respecting sentence boundaries.

    Splits on sentence-ending punctuation ``[.!?]+\\s+`` and includes as
    many complete sentences as fit.  If no sentence boundary is found,
    truncates at the raw character level.  Always returns at least one
    character (``text[:1]`` as an absolute minimum).
    """
    stripped = text.strip()
    if not stripped:
        return text[:max(1, target_length)]

    sentences = re.split(r"[.!?]+\s+", stripped)
    if not sentences:
        return text[: max(1, target_length)]

    result: list[str] = []
    total = 0
    for s in sentences:
        if total + len(s) > target_length and result:
            break
        result.append(s)
        total += len(s)

    return ". ".join(result) if result else text[: max(1, target_length)]


def _extract_by_sentence_scoring(content: str, target_length: int) -> str:
    """Fallback extraction using sentence-level information density scoring.

    Scores each sentence by ``len(sentence) * (unique_words / total_words)``
    and greedily selects sentences in descending score order until
    *target_length* is reached.  Never returns an empty string.
    Deterministic.
    """
    stripped = content.strip()
    sentences = re.split(r"[.!?]+\s+", stripped)
    if not sentences or not any(s.strip() for s in sentences):
        return content[: max(1, target_length)]

    def _score(s: str) -> float:
        words = s.split()
        unique = len(set(w.lower() for w in words))
        return len(s) * (unique / max(len(words), 1))

    scored = [(_score(s), s) for s in sentences if s.strip()]
    if not scored:
        return content[: max(1, target_length)]

    scored.sort(reverse=True, key=lambda x: x[0])
    extracted: list[str] = []
    total_len = 0
    for _, sentence in scored:
        if total_len + len(sentence) > target_length and extracted:
            break
        extracted.append(sentence)
        total_len += len(sentence)

    return " ".join(extracted) if extracted else content[: max(1, target_length)]


def extract_semantic(
    content: str,
    category: str,
    config: ConsolidationConfig | None = None,
    source_episodes: tuple[str, ...] = (),
    source_creation_times: tuple[float, ...] = (),
    source_importances: tuple[float, ...] = (),
    source_access_counts: tuple[int, ...] | None = None,
) -> SemanticExtraction:
    """Extract semantic knowledge from episodic content.

    Tries pattern-based extraction first (using EXTRACTION_PATTERNS for the
    given category), falls back to sentence-scoring extraction.  Returns a
    ``SemanticExtraction`` dataclass with provenance metadata.

    Behavioral contracts:
        - Correction category returns *content* unchanged.
        - Returned ``content`` is never empty.
        - ``confidence`` = mean of *source_importances*, clamped to [0, 1].
        - ``importance`` = clamp01(mean_importance * config.semantic_boost).
        - ``access_count`` = int(avg_access * config.activation_inheritance)
          when *source_access_counts* is provided, else 0.
        - Deterministic for identical inputs.

    Raises:
        TypeError:  If *content* is not a string.
        ValueError: If *category* is not in VALID_CATEGORIES, or
                     *source_episodes* is empty, or lengths of
                     *source_creation_times* / *source_importances* do not
                     match *source_episodes*.
    """
    config = config or DEFAULT_CONSOLIDATION_CONFIG

    # Validate inputs
    if not isinstance(content, str):
        raise TypeError(
            f"content must be a string, got {type(content).__name__}"
        )
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"category must be one of {sorted(VALID_CATEGORIES)}, "
            f"got {category!r}"
        )
    if not source_episodes:
        raise ValueError("source_episodes must be non-empty")
    if len(source_creation_times) != len(source_episodes):
        raise ValueError(
            f"source_creation_times length ({len(source_creation_times)}) "
            f"must match source_episodes length ({len(source_episodes)})"
        )
    if len(source_importances) != len(source_episodes):
        raise ValueError(
            f"source_importances length ({len(source_importances)}) "
            f"must match source_episodes length ({len(source_episodes)})"
        )

    # Pattern extraction
    patterns = _get_extraction_patterns(category)
    extracted: str | None = None
    for pattern in patterns:
        match = pattern.search(content)
        if match:
            groups = [g for g in match.groups() if g]
            if groups:
                extracted = max(groups, key=len).strip()
                break

    # Fallback: sentence scoring
    if extracted is None:
        compression_ratio = CATEGORY_COMPRESSION_RATIOS.get(category, 5)
        target_len = max(1, int(len(content) / compression_ratio))
        extracted = _extract_by_sentence_scoring(content, target_len)

    # Apply template
    template = SEMANTIC_TEMPLATES.get(category, "{extracted}")
    if category == "correction":
        formatted = content  # Corrections preserve full content
    else:
        formatted = template.format(extracted=extracted, original=content)

    # Compute provenance
    confidence = clamp01(
        sum(source_importances) / max(len(source_importances), 1)
    )
    first_observed = (
        min(source_creation_times) if source_creation_times else 0.0
    )
    last_updated = (
        max(source_creation_times) if source_creation_times else 0.0
    )
    avg_importance = sum(source_importances) / max(len(source_importances), 1)

    # M2: Apply semantic_boost to importance
    boosted_importance = clamp01(avg_importance * config.semantic_boost)

    # M3: Apply activation_inheritance to access counts
    if source_access_counts is not None and source_access_counts:
        avg_access = sum(source_access_counts) / len(source_access_counts)
        inherited_access_count = int(avg_access * config.activation_inheritance)
    else:
        inherited_access_count = 0

    # H8: Compute compression_ratio (source length / extracted length)
    source_len = max(len(content), 1)
    extracted_len = max(len(formatted), 1)
    compression_ratio = source_len / extracted_len

    # Determine target level (next level from EPISODIC_RAW by default)
    target_level = ConsolidationLevel.EPISODIC_COMPRESSED

    return SemanticExtraction(
        content=formatted,
        category=category,
        source_episodes=source_episodes,
        confidence=confidence,
        first_observed=first_observed,
        last_updated=last_updated,
        consolidation_count=len(source_episodes),
        compression_ratio=compression_ratio,
        importance=boosted_importance,
        target_level=target_level,
        extraction_mode=ConsolidationMode.ASYNC_BATCH,
        access_count=inherited_access_count,
    )


def archive_episodic(
    candidate: ConsolidationCandidate,
    consolidated_to: tuple[str, ...],
    config: ConsolidationConfig | None = None,
) -> ArchivedMemory:
    """Archive an episodic memory after consolidation.

    Creates an ``ArchivedMemory`` record with decayed importance and
    strength.

    Behavioral contracts:
        - ``decayed_importance`` = ``clamp01(candidate.importance * config.consolidation_decay)``
        - ``decayed_strength`` = ``max(0.0, candidate.strength * config.consolidation_decay)``
        - ``decayed_importance <= original_importance``
        - ``decayed_strength <= original_strength``
        - ``archived`` is always ``True``
        - Original content preserved in full.
        - Raises ``ValueError`` if *consolidated_to* is empty.
        - Deterministic.
    """
    config = config or DEFAULT_CONSOLIDATION_CONFIG
    if not consolidated_to:
        raise ValueError("consolidated_to must be non-empty")

    decayed_importance = clamp01(candidate.importance * config.consolidation_decay)
    decayed_strength = max(0.0, candidate.strength * config.consolidation_decay)

    return ArchivedMemory(
        memory_id=candidate.memory_id,
        content=candidate.content,
        category=candidate.category,
        level=candidate.level,
        archived=True,
        consolidated_to=consolidated_to,
        decayed_importance=decayed_importance,
        decayed_strength=decayed_strength,
        original_importance=candidate.importance,
        original_strength=candidate.strength,
        creation_time=candidate.creation_time,
        archive_time=candidate.last_access_time,
        access_count_at_archive=candidate.access_count,
    )


# ============================================================
# Phase D: Pipeline Functions
# ============================================================


_LEVEL_PROMOTION: dict[ConsolidationLevel, ConsolidationLevel | None] = {
    ConsolidationLevel.EPISODIC_RAW: ConsolidationLevel.EPISODIC_COMPRESSED,
    ConsolidationLevel.EPISODIC_COMPRESSED: ConsolidationLevel.SEMANTIC_FACT,
    ConsolidationLevel.SEMANTIC_FACT: ConsolidationLevel.SEMANTIC_INSIGHT,
    ConsolidationLevel.SEMANTIC_INSIGHT: None,
}


def _next_level(current: ConsolidationLevel) -> ConsolidationLevel | None:
    """Return the next consolidation level, or None if already at the top.

    Simple deterministic mapping:
        EPISODIC_RAW -> EPISODIC_COMPRESSED
        EPISODIC_COMPRESSED -> SEMANTIC_FACT
        SEMANTIC_FACT -> SEMANTIC_INSIGHT
        SEMANTIC_INSIGHT -> None

    Never raises.
    """
    return _LEVEL_PROMOTION[current]


def consolidate_memory(
    candidate: ConsolidationCandidate,
    config: ConsolidationConfig | None = None,
) -> ConsolidationResult:
    """Orchestrate single-memory consolidation.

    Calls :func:`should_consolidate`, :func:`extract_semantic`, and
    :func:`archive_episodic` in sequence to promote one episodic memory
    to its next consolidation level.

    Behavioral contracts:
        - If ``should_consolidate()`` returns False -> empty result.
        - If the candidate is already at ``SEMANTIC_INSIGHT`` -> empty result.
        - ``candidates_evaluated`` is always 1.
        - ``mode`` is always ``RETRIEVAL_TRIGGERED`` for single-memory.
        - Deterministic.
    """
    config = config or DEFAULT_CONSOLIDATION_CONFIG

    # 1. Check eligibility
    if not should_consolidate(candidate, config):
        return ConsolidationResult(
            extractions=(),
            archived=(),
            skipped_indices=frozenset({candidate.memory_id}),
            candidates_evaluated=1,
            candidates_consolidated=0,
            mode=ConsolidationMode.RETRIEVAL_TRIGGERED,
        )

    # 2. Check target level
    target_level = _next_level(candidate.level)
    if target_level is None:
        return ConsolidationResult(
            extractions=(),
            archived=(),
            skipped_indices=frozenset({candidate.memory_id}),
            candidates_evaluated=1,
            candidates_consolidated=0,
            mode=ConsolidationMode.RETRIEVAL_TRIGGERED,
        )

    # 3. Extract semantic content
    extraction = extract_semantic(
        content=candidate.content,
        category=candidate.category,
        config=config,
        source_episodes=(candidate.memory_id,),
        source_creation_times=(candidate.creation_time,),
        source_importances=(candidate.importance,),
    )

    # 4. Archive the original
    archived = archive_episodic(
        candidate=candidate,
        consolidated_to=(candidate.memory_id,),
        config=config,
    )

    # 5. Return result
    return ConsolidationResult(
        extractions=(extraction,),
        archived=(archived,),
        skipped_indices=frozenset(),
        candidates_evaluated=1,
        candidates_consolidated=1,
        mode=ConsolidationMode.RETRIEVAL_TRIGGERED,
    )


def select_consolidation_candidates(
    pool: list[ConsolidationCandidate],
    config: ConsolidationConfig | None = None,
    mode: ConsolidationMode = ConsolidationMode.ASYNC_BATCH,
    pool_size: int | None = None,
) -> list[ConsolidationCandidate]:
    """Select eligible candidates from a pool for consolidation.

    Filters the pool based on the consolidation *mode* and optional
    storage-pressure signals (via *pool_size*).

    Modes:
        - ``INTRA_SESSION``: Only checks inhibitors (not full
          ``should_consolidate``).
        - ``ASYNC_BATCH``: Uses ``should_consolidate`` plus graduated
          storage pressure when *pool_size* exceeds thresholds.
        - ``RETRIEVAL_TRIGGERED``: Uses ``should_consolidate``.

    Behavioral contracts:
        - Contested candidates are NEVER selected.
        - Correction candidates are NEVER selected.
        - Result length <= ``config.max_candidates_per_batch``.
        - Result sorted by retention ascending (lowest first).
        - Deterministic.
    """
    config = config or DEFAULT_CONSOLIDATION_CONFIG
    candidates: list[ConsolidationCandidate] = []

    for candidate in pool:
        eligible = False

        if mode == ConsolidationMode.INTRA_SESSION:
            # Just check inhibitors pass
            if _passes_inhibitors(candidate, config):
                eligible = True

        elif mode == ConsolidationMode.ASYNC_BATCH:
            if should_consolidate(candidate, config):
                eligible = True
            elif pool_size is not None and _passes_inhibitors(candidate, config):
                # Graduated storage pressure
                severe_threshold = int(
                    config.max_pool_size * config.severe_pressure_multiplier
                )
                mild = pool_size > config.max_pool_size
                severe = pool_size > severe_threshold

                if severe:
                    if candidate.creation_time > config.consolidation_window:
                        eligible = True
                elif mild:
                    safe_strength = max(candidate.strength, 1e-12)
                    current_ret = retention(candidate.last_access_time, safe_strength)
                    sv = _compute_semantic_value(candidate)
                    if (
                        current_ret < config.mild_pressure_retention
                        and sv >= config.semantic_value_threshold
                        and candidate.creation_time > config.consolidation_window
                    ):
                        eligible = True

        elif mode == ConsolidationMode.RETRIEVAL_TRIGGERED:
            eligible = should_consolidate(candidate, config)

        if eligible:
            candidates.append(candidate)

    # Sort by retention ascending (lowest first)
    def _ret_key(c: ConsolidationCandidate) -> float:
        safe_strength = max(c.strength, 1e-12)
        return retention(c.last_access_time, safe_strength)

    candidates.sort(key=_ret_key)
    return candidates[: config.max_candidates_per_batch]
