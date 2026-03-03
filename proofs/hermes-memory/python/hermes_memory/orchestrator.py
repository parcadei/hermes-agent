"""Hermes Memory System -- Proof Orchestrator.

In-memory orchestrator that wires the 5 verified modules (encoding,
contradiction, consolidation, recall, engine) into a complete memory
lifecycle.  Provides store(), query(), and consolidate() operations
with full contradiction detection, supersession, and episodic-to-semantic
promotion.

Design decisions:
  - StoredMemory uses ABSOLUTE logical timestamps internally.
  - Conversion to RELATIVE ages happens at the boundary (_stored_to_memory_state,
    _stored_to_candidate) when handing data to recall or consolidation.
  - Bridge 2 (semantic_extraction_to_memory_state) receives ALREADY-RELATIVE
    ages from consolidation output, so it passes them through directly.
  - Mutations use dataclasses.replace() on a mutable list of frozen dataclasses.
  - store() is atomic: all mutations are computed in locals, then applied at once.
"""

from __future__ import annotations

import dataclasses
import math
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from hermes_memory.consolidation import (
    ConsolidationCandidate,
    ConsolidationConfig,
    ConsolidationLevel,
    ConsolidationMode,
    SemanticExtraction,
    consolidate_memory,
    select_consolidation_candidates,
)
from hermes_memory.contradiction import (
    ContradictionConfig,
    ContradictionResult,
    SupersessionRecord,
    detect_contradictions,
    resolve_contradictions,
)
from hermes_memory.encoding import (
    EncodingConfig,
    EncodingDecision,
    EncodingPolicy,
)
from hermes_memory.engine import MemoryState, ParameterSet
from hermes_memory.recall import RecallConfig, RecallResult, recall


# ============================================================
# Relevance scoring abstraction
# ============================================================


@runtime_checkable
class RelevanceScorer(Protocol):
    """Protocol for query-content relevance scoring.

    Implementations must provide a score(query, content) method that
    returns a float in [0.0, 1.0].  This abstraction allows the
    orchestrator to be tested and used with different relevance
    strategies without coupling to a specific implementation.
    """

    def score(self, query: str, content: str) -> float:
        """Compute relevance between a query and memory content.

        Args:
            query:   Query text.
            content: Memory content text.

        Returns:
            Float in [0.0, 1.0].  0.0 means no relevance, 1.0 means identical.
        """
        ...


class JaccardRelevance:
    """Jaccard word-set similarity scorer.

    Proof-of-concept relevance metric.  Computes the Jaccard index
    (intersection over union) of lowercased word sets.  Only catches
    near-duplicate content (documented limitation from consolidation
    research Section 16.1).  Sufficient for verifying lifecycle math,
    not retrieval quality.
    """

    def score(self, query: str, content: str) -> float:
        """Compute Jaccard similarity between lowercased word sets.

        Args:
            query:   Query text.
            content: Memory content text.

        Returns:
            Float in [0.0, 1.0].  0.0 for empty inputs.
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words or not content_words:
            return 0.0

        intersection = query_words & content_words
        union = query_words | content_words

        if not union:
            return 0.0

        return len(intersection) / len(union)


# ============================================================
# Data types
# ============================================================


@dataclass(frozen=True)
class StoredMemory:
    """Internal representation combining content, encoding metadata, and dynamics state.

    Frozen to prevent accidental mutation.  Updates go through
    dataclasses.replace() which produces a new instance.

    Time model: creation_time and last_access_time store ABSOLUTE logical
    timestamps.  Conversion to relative ages happens at module boundaries
    (_stored_to_memory_state, _stored_to_candidate).

    Attributes:
        memory_id:           UUID hex string uniquely identifying this memory.
        content:             The text content of the memory.
        category:            Category from VALID_CATEGORIES (encoding output).
        importance:          Importance score in [0, 1].
        strength:            Memory strength in [0, +inf).
        creation_time:       Absolute logical time when memory was created.
        last_access_time:    Absolute logical time of last access.
        access_count:        Number of times this memory has been accessed (>= 0).
        level:               Current consolidation level.
        is_active:           Whether the memory is active (not archived/superseded).
        is_contested:        Whether the memory is flagged as contested.
        encoding_confidence: Confidence from the encoding decision.  NOTE: this
            field is write-only -- it is set by store() and consolidate() but
            never read by any downstream module (recall, consolidation, engine).
            Retained for backward compatibility; do not add logic that depends
            on it without first promoting it to a consumed field.
        source_episodes:     Tuple of source episode IDs (for semantic memories).
        consolidation_count: Number of times consolidated from lower levels.
    """

    memory_id: str
    content: str
    category: str
    importance: float
    strength: float
    creation_time: float
    last_access_time: float
    access_count: int
    level: ConsolidationLevel = ConsolidationLevel.EPISODIC_RAW
    is_active: bool = True
    is_contested: bool = False
    encoding_confidence: float = 0.5
    source_episodes: tuple[str, ...] = ()
    consolidation_count: int = 0


@dataclass(frozen=True)
class StoreResult:
    """Return type of MemoryOrchestrator.store().

    Captures the encoding decision, optional contradiction result,
    any supersession records, and IDs of deactivated memories.

    Attributes:
        memory_id:             UUID of the newly stored memory, or None if rejected.
        stored:                Whether the memory was actually stored.
        encoding_decision:     The EncodingDecision from the encoding gate.
        contradiction_result:  ContradictionResult if contradictions were checked.
        supersession_records:  Tuple of SupersessionRecords from resolution.
        deactivated_ids:       Tuple of memory IDs that were deactivated.
    """

    memory_id: str | None
    stored: bool
    encoding_decision: EncodingDecision
    contradiction_result: ContradictionResult | None = None
    supersession_records: tuple[SupersessionRecord, ...] = ()
    deactivated_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsolidationSummary:
    """Return type of MemoryOrchestrator.consolidate().

    Summarizes what happened during a consolidation pass.

    Attributes:
        candidates_evaluated:    Total candidates considered.
        candidates_consolidated: Candidates that were actually consolidated.
        new_semantic_ids:        Tuple of UUIDs for newly created semantic memories.
        archived_ids:            Tuple of memory IDs that were archived.
        skipped_ids:             Tuple of memory IDs that were skipped.
    """

    candidates_evaluated: int
    candidates_consolidated: int
    new_semantic_ids: tuple[str, ...]
    archived_ids: tuple[str, ...]
    skipped_ids: tuple[str, ...]


# ============================================================
# Bridge functions (standalone, pure, independently testable)
# ============================================================


def map_contradictions_to_candidates(
    flagged_indices: frozenset[int],
    candidates: list[ConsolidationCandidate],
    index_to_memory_id: dict[int, str],
) -> list[ConsolidationCandidate]:
    """Map contradiction flagged_indices to consolidation candidates.

    Translates flagged positions (indices into the existing_texts list used
    during contradiction detection) through the index_to_memory_id mapping
    to find matching candidates by memory_id.  Returns a new list with
    matched candidates replaced via dataclasses.replace(c, is_contested=True).

    Behavioral contracts:
      - Pure function: no side effects, deterministic.
      - Returns a NEW list; original candidates are not mutated.
      - Flagged indices not in index_to_memory_id are silently ignored.
      - Memory IDs in the mapping that don't match any candidate are ignored.
      - Already-contested candidates remain contested.

    Args:
        flagged_indices:    Frozenset of positions flagged by contradiction detection.
        candidates:         List of ConsolidationCandidate to potentially flag.
        index_to_memory_id: Mapping from existing_texts index to memory_id.

    Returns:
        New list of ConsolidationCandidate with contested flags applied.
    """
    # Build set of memory IDs that are flagged
    flagged_memory_ids: set[str] = set()
    for idx in flagged_indices:
        mid = index_to_memory_id.get(idx)
        if mid is not None:
            flagged_memory_ids.add(mid)

    # Build new list, replacing contested candidates
    result: list[ConsolidationCandidate] = []
    for candidate in candidates:
        if candidate.memory_id in flagged_memory_ids:
            result.append(dataclasses.replace(candidate, is_contested=True))
        else:
            result.append(candidate)
    return result


def semantic_extraction_to_memory_state(
    extraction: SemanticExtraction,
    relevance: float,
    initial_strength: float,
) -> MemoryState:
    """Convert a SemanticExtraction to a MemoryState for recall scoring.

    CRITICAL TIME MODEL: The extraction's first_observed and last_updated
    fields are ALREADY relative ages.  The chain is:
      1. Orchestrator converts StoredMemory (absolute) to ConsolidationCandidate
         (relative age) via _stored_to_candidate().
      2. consolidate_memory() passes candidate.creation_time (relative) to
         extract_semantic() which sets first_observed = min(source_creation_times).
      3. Therefore first_observed and last_updated are already relative ages.

    This bridge passes them through DIRECTLY -- no subtraction from current_time.

    Mapping:
      - creation_time     <- extraction.first_observed  (direct pass-through)
      - last_access_time  <- extraction.last_updated    (direct pass-through)
      - importance        <- extraction.importance
      - access_count      <- extraction.access_count
      - strength          <- initial_strength (caller passes params.s0)
      - relevance         <- passed in

    Raises:
        ValueError: If relevance is outside [0, 1].
        ValueError: If first_observed or last_updated is negative.
        ValueError: If initial_strength is negative.

    Args:
        extraction:       SemanticExtraction from consolidation.
        relevance:        Query-dependent relevance score in [0, 1].
        initial_strength: Initial strength for the semantic memory.

    Returns:
        MemoryState suitable for recall scoring.
    """
    if not (0.0 <= relevance <= 1.0):
        msg = f"relevance must be in [0, 1], got {relevance}"
        raise ValueError(msg)

    creation_time = extraction.first_observed
    last_access_time = extraction.last_updated

    if creation_time < 0.0:
        msg = f"first_observed must be >= 0, got {creation_time}"
        raise ValueError(msg)

    if last_access_time < 0.0:
        msg = f"last_updated must be >= 0, got {last_access_time}"
        raise ValueError(msg)

    if initial_strength < 0.0:
        msg = f"initial_strength must be >= 0, got {initial_strength}"
        raise ValueError(msg)

    return MemoryState(
        relevance=relevance,
        last_access_time=last_access_time,
        importance=extraction.importance,
        access_count=extraction.access_count,
        strength=initial_strength,
        creation_time=creation_time,
    )


# ============================================================
# MemoryOrchestrator
# ============================================================


class MemoryOrchestrator:
    """In-memory orchestrator wiring encoding, contradiction, consolidation, and recall.

    Manages a list of StoredMemory instances and provides three operations:
      - store(content, metadata, timestamp) -> StoreResult
      - query(message, turn_number, current_time) -> RecallResult
      - consolidate(current_time, mode) -> ConsolidationSummary

    Internal state:
      - _memories: mutable list of frozen StoredMemory instances.
      - _id_to_idx: dict mapping memory_id -> index in _memories for O(1) lookup.
      - _next_time: float logical clock, starts at 1.0, incremented by store().
    """

    def __init__(
        self,
        params: ParameterSet,
        encoding_config: EncodingConfig | None = None,
        contradiction_config: ContradictionConfig | None = None,
        consolidation_config: ConsolidationConfig | None = None,
        recall_config: RecallConfig | None = None,
        relevance_scorer: RelevanceScorer | None = None,
    ) -> None:
        """Initialize the orchestrator with parameter set and optional configs.

        Args:
            params:               ParameterSet for scoring/dynamics.
            encoding_config:      EncodingConfig for the encoding gate (default: None).
            contradiction_config: ContradictionConfig for contradiction detection.
            consolidation_config: ConsolidationConfig for consolidation pipeline.
            recall_config:        RecallConfig for the recall pipeline.
            relevance_scorer:     RelevanceScorer for query-content relevance
                                  (default: JaccardRelevance).
        """
        self.params = params
        self._encoding_policy = EncodingPolicy(encoding_config)
        self._contradiction_config = contradiction_config
        self._consolidation_config = consolidation_config
        self._recall_config = recall_config or RecallConfig()
        self._relevance_scorer: RelevanceScorer = relevance_scorer or JaccardRelevance()
        self._memories: list[StoredMemory] = []
        self._id_to_idx: dict[str, int] = {}
        # FAISS ANN pre-filter state (lazy-initialized on first embedding store)
        self._faiss_index = None          # faiss.IndexFlatIP, lazy init
        self._faiss_id_map: list[str] = []  # parallel to FAISS rows: memory_id at row i
        self._faiss_dim: int | None = None   # inferred from first embedding
        # Clock starts at 1.0 (not 0.0) because retention(age=0, ...) produces
        # exp(0)=1.0 which is the maximum, and several scoring functions use
        # age as a divisor in derived calculations.  Starting at 1.0 ensures
        # the first memory stored has creation_age >= 0 after the clock
        # increments and avoids degenerate zero-age edge cases in downstream
        # math (e.g., novelty_bonus division).
        self._next_time: float = 1.0

    # -- Properties ----------------------------------------------------------

    @property
    def memories(self) -> list[StoredMemory]:
        """Return list of active memories (is_active=True)."""
        return [m for m in self._memories if m.is_active]

    @property
    def all_memories(self) -> list[StoredMemory]:
        """Return list of ALL memories including deactivated ones."""
        return list(self._memories)

    # -- Time management -----------------------------------------------------

    def advance_time(self, delta: float) -> None:
        """Advance the internal logical clock by delta.

        Used by integration tests to push memories past recency_guard
        and consolidation_window before triggering consolidation.

        Args:
            delta: Non-negative finite time increment.

        Raises:
            ValueError: If delta is negative (clock regression).
            ValueError: If delta is NaN or infinity (would corrupt all
                age calculations downstream).
        """
        if not math.isfinite(delta):
            msg = f"advance_time delta must be finite, got {delta}"
            raise ValueError(msg)
        if delta < 0:
            msg = f"advance_time delta must be >= 0, got {delta}"
            raise ValueError(msg)
        self._next_time += delta

    # -- Internal helpers ----------------------------------------------------

    def _rebuild_index(self) -> None:
        """Update _id_to_idx for the most recently appended memory (O(1))."""
        if self._memories:
            last = self._memories[-1]
            self._id_to_idx[last.memory_id] = len(self._memories) - 1

    def _full_rebuild_index(self) -> None:
        """Full O(N) rebuild -- only called after consolidation."""
        self._id_to_idx = {m.memory_id: i for i, m in enumerate(self._memories)}

    def _simple_text_relevance(self, query: str, content: str) -> float:
        """Compute relevance between query and content using the injected scorer.

        Delegates to self._relevance_scorer.score().  This method is retained
        for backward compatibility with existing tests that call it directly.
        The default scorer (JaccardRelevance) uses Jaccard word-set similarity.

        Args:
            query:   Query text.
            content: Memory content text.

        Returns:
            Float in [0.0, 1.0].  0.0 for empty inputs.
        """
        return self._relevance_scorer.score(query, content)

    def _stored_to_memory_state(
        self,
        mem: StoredMemory,
        relevance: float,
        current_time: float,
    ) -> MemoryState:
        """Convert a StoredMemory to a MemoryState for recall scoring.

        Converts absolute timestamps to RELATIVE ages:
          creation_time    = current_time - mem.creation_time
          last_access_time = current_time - mem.last_access_time

        Validates that computed ages are non-negative, consistent with
        the validation in semantic_extraction_to_memory_state (Bridge 2).

        Args:
            mem:          StoredMemory with absolute timestamps.
            relevance:    Query-dependent relevance score.
            current_time: Current logical time for relative age computation.

        Returns:
            MemoryState with relative ages.

        Raises:
            ValueError: If creation_age < 0 (memory created in the future).
            ValueError: If last_access_age < 0 (clock inconsistency).
        """
        creation_age = current_time - mem.creation_time
        last_access_age = current_time - mem.last_access_time

        if creation_age < 0.0:
            msg = (
                f"creation_age must be >= 0, got {creation_age} "
                f"(current_time={current_time}, creation_time={mem.creation_time})"
            )
            raise ValueError(msg)

        if last_access_age < 0.0:
            msg = (
                f"last_access_age must be >= 0, got {last_access_age} "
                f"(current_time={current_time}, last_access_time={mem.last_access_time})"
            )
            raise ValueError(msg)

        return MemoryState(
            relevance=relevance,
            last_access_time=last_access_age,
            importance=mem.importance,
            access_count=mem.access_count,
            strength=mem.strength,
            creation_time=creation_age,
        )

    def _stored_to_candidate(
        self,
        mem: StoredMemory,
        current_time: float,
    ) -> ConsolidationCandidate:
        """Convert a StoredMemory to a ConsolidationCandidate.

        Converts absolute timestamps to RELATIVE ages for the consolidation
        pipeline.  CRITICAL: consolidation expects relative ages, not absolute
        timestamps.

        Args:
            mem:          StoredMemory with absolute timestamps.
            current_time: Current logical time for relative age computation.

        Returns:
            ConsolidationCandidate with relative ages.
        """
        creation_age = current_time - mem.creation_time
        last_access_age = current_time - mem.last_access_time

        return ConsolidationCandidate(
            memory_id=mem.memory_id,
            content=mem.content,
            category=mem.category,
            level=mem.level,
            creation_time=creation_age,
            last_access_time=last_access_age,
            access_count=mem.access_count,
            importance=mem.importance,
            strength=mem.strength,
            relevance=0.0,  # Consolidation uses retention(age, level), not query relevance
            is_contested=mem.is_contested,
            source_episodes=mem.source_episodes,
            consolidation_count=mem.consolidation_count,
        )

    # -- Store ---------------------------------------------------------------

    # -- ANN Pre-Filter Constants -------------------------------------------

    ANN_COSINE_THRESHOLD: float = 0.5
    ANN_K: int = 50

    @staticmethod
    def _remap_contradiction_result(
        result: ContradictionResult,
        subset_to_active: list[int],
    ) -> ContradictionResult:
        """Remap indices from subset-space to active-list-space.

        detect_contradictions() returns indices relative to the filtered
        existing_texts list (subset). This method remaps them back to
        the full active-list indices so that deactivation and flagging
        target the correct memories.

        Args:
            result:           ContradictionResult with subset-space indices.
            subset_to_active: Mapping from subset index to active-list index.

        Returns:
            New ContradictionResult with remapped indices.
        """
        from hermes_memory.contradiction import ContradictionDetection

        remapped_detections = tuple(
            ContradictionDetection(
                existing_index=subset_to_active[det.existing_index],
                contradiction_type=det.contradiction_type,
                confidence=det.confidence,
                subject_overlap=det.subject_overlap,
                candidate_subject=det.candidate_subject,
                existing_subject=det.existing_subject,
                explanation=det.explanation,
            )
            for det in result.detections
        )

        remapped_actions = tuple(
            (subset_to_active[idx], action)
            for idx, action in result.actions
        )

        remapped_superseded = frozenset(
            subset_to_active[i] for i in result.superseded_indices
        )

        remapped_flagged = frozenset(
            subset_to_active[i] for i in result.flagged_indices
        )

        return ContradictionResult(
            detections=remapped_detections,
            actions=remapped_actions,
            superseded_indices=remapped_superseded,
            flagged_indices=remapped_flagged,
            has_contradiction=result.has_contradiction,
            highest_confidence=result.highest_confidence,
        )

    def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        timestamp: float | None = None,
        embedding: np.ndarray | None = None,
    ) -> StoreResult:
        """Store a memory through the encoding and contradiction pipeline.

        Atomic: all mutations (deactivations, flagging, new memory) are computed
        in local variables, then applied to _memories at the end.  This prevents
        partial corruption if an exception occurs mid-store.

        Pipeline:
          1. Encoding gate: evaluate content and metadata.
          2. If rejected: return StoreResult(stored=False).
          3. Contradiction detection against active memories.
             - When embedding is provided and FAISS is available: ANN pre-filter
               narrows candidates to top-K nearest neighbors (O(K) instead of O(N)).
             - When embedding is None: full-scan (backwards compatible).
          4. Contradiction resolution (supersession records).
          5. Deactivate superseded memories.
          6. Flag contested memories.
          7. Create new StoredMemory and append.
          8. Add embedding to FAISS index (if provided).

        Args:
            content:   Text content to store.
            metadata:  Optional metadata dict for encoding gate.
            timestamp: Optional explicit timestamp for contradiction resolution.
            embedding: Optional pre-computed embedding for ANN pre-filter.

        Returns:
            StoreResult describing what happened.
        """
        current_time = self._next_time
        ts = timestamp if timestamp is not None else current_time

        # Step 1: Encoding gate
        decision = self._encoding_policy.evaluate(content, metadata)

        if not decision.should_store:
            self._next_time += 1.0
            return StoreResult(
                memory_id=None,
                stored=False,
                encoding_decision=decision,
            )

        # Step 2: Build existing memory lists for contradiction detection
        active = self.memories
        use_ann = (
            embedding is not None
            and faiss is not None
        )

        # Validate and prepare embedding for FAISS
        if use_ann:
            emb = np.asarray(embedding, dtype=np.float64).ravel()
            # Dimension validation
            if self._faiss_dim is not None and emb.shape[0] != self._faiss_dim:
                raise ValueError(
                    f"Embedding dimension {emb.shape[0]} != expected {self._faiss_dim}"
                )
            # L2-normalize
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            # Lazy-init FAISS index
            if self._faiss_index is None:
                self._faiss_dim = emb.shape[0]
                self._faiss_index = faiss.IndexFlatIP(self._faiss_dim)

        # Step 3: Contradiction detection
        contradiction_result: ContradictionResult | None = None
        supersession_records: tuple[SupersessionRecord, ...] = ()
        deactivation_indices: frozenset[int] = frozenset()

        if use_ann and self._faiss_index.ntotal > 0 and active:
            # ANN pre-filter path: search FAISS for top-K nearest neighbors
            K = min(self.ANN_K, self._faiss_index.ntotal)
            emb_f32 = emb.astype(np.float32).reshape(1, -1)
            distances, indices = self._faiss_index.search(emb_f32, K)

            # Filter by cosine threshold and map to active-list indices
            # Build mapping: FAISS memory_ids -> active-list positions
            active_id_to_pos = {m.memory_id: i for i, m in enumerate(active)}
            subset_to_active: list[int] = []

            for j in range(K):
                faiss_row = int(indices[0, j])
                cos_sim = float(distances[0, j])
                if faiss_row < 0 or cos_sim < self.ANN_COSINE_THRESHOLD:
                    continue
                mid = self._faiss_id_map[faiss_row]
                active_pos = active_id_to_pos.get(mid)
                if active_pos is not None:
                    subset_to_active.append(active_pos)

            if subset_to_active:
                existing_texts = [active[i].content for i in subset_to_active]
                existing_categories = [active[i].category for i in subset_to_active]

                contradiction_result = detect_contradictions(
                    candidate_text=content,
                    candidate_category=decision.category,
                    existing_texts=existing_texts,
                    existing_categories=existing_categories,
                    config=self._contradiction_config,
                )

                # Remap subset-space indices back to active-list-space
                if contradiction_result.has_contradiction:
                    contradiction_result = self._remap_contradiction_result(
                        contradiction_result, subset_to_active
                    )
                    records, deactivation_indices = resolve_contradictions(
                        result=contradiction_result,
                        candidate_text=content,
                        existing_texts=[m.content for m in active],
                        timestamp=ts,
                    )
                    supersession_records = tuple(records)
        else:
            # Full-scan path (no embedding or empty FAISS index)
            existing_texts = [m.content for m in active]
            existing_categories = [m.category for m in active]

            if existing_texts:
                contradiction_result = detect_contradictions(
                    candidate_text=content,
                    candidate_category=decision.category,
                    existing_texts=existing_texts,
                    existing_categories=existing_categories,
                    config=self._contradiction_config,
                )

                if contradiction_result.has_contradiction:
                    records, deactivation_indices = resolve_contradictions(
                        result=contradiction_result,
                        candidate_text=content,
                        existing_texts=existing_texts,
                        timestamp=ts,
                    )
                    supersession_records = tuple(records)

        # Step 4: Compute mutations in local variables (atomic path)
        # Map active list indices back to memory IDs
        deactivated_ids: list[str] = []
        mutations: list[tuple[int, StoredMemory]] = []

        for active_idx in deactivation_indices:
            mid = active[active_idx].memory_id
            deactivated_ids.append(mid)
            global_idx = self._id_to_idx[mid]
            old = self._memories[global_idx]
            mutations.append(
                (global_idx, dataclasses.replace(old, is_active=False))
            )

        # Flag contested memories
        if contradiction_result is not None:
            for active_idx in contradiction_result.flagged_indices:
                if active_idx not in deactivation_indices:
                    mid = active[active_idx].memory_id
                    global_idx = self._id_to_idx[mid]
                    old = self._memories[global_idx]
                    if not old.is_contested:
                        mutations.append(
                            (global_idx, dataclasses.replace(old, is_contested=True))
                        )

        # Create new StoredMemory
        new_id = uuid.uuid4().hex
        new_memory = StoredMemory(
            memory_id=new_id,
            content=content,
            category=decision.category,
            importance=decision.initial_importance,
            strength=self.params.s0,
            creation_time=current_time,
            last_access_time=current_time,
            access_count=0,
            encoding_confidence=decision.confidence,
        )

        # Step 5: Apply all mutations at once (atomic)
        for global_idx, replacement in mutations:
            self._memories[global_idx] = replacement

        self._memories.append(new_memory)
        self._rebuild_index()
        self._next_time += 1.0

        # Step 6: Add embedding to FAISS index (after memory is stored)
        if use_ann:
            emb_f32 = emb.astype(np.float32).reshape(1, -1)
            self._faiss_index.add(emb_f32)
            self._faiss_id_map.append(new_id)

        return StoreResult(
            memory_id=new_id,
            stored=True,
            encoding_decision=decision,
            contradiction_result=contradiction_result,
            supersession_records=supersession_records,
            deactivated_ids=tuple(deactivated_ids),
        )

    # -- Query ---------------------------------------------------------------

    def query(
        self,
        message: str,
        turn_number: int = 0,
        current_time: float | None = None,
    ) -> RecallResult:
        """Query the memory store using the recall pipeline.

        Pipeline:
          1. Collect active memories.
          2. Compute text relevance for each.
          3. Convert to MemoryState via _stored_to_memory_state().
          4. Build parallel contents list.
          5. Call recall() with memories, params, config.
          6. Increment access_count for memories in result's tier_assignments.

        Args:
            message:      User message to query against.
            turn_number:  Current conversation turn (for gating).
            current_time: Optional explicit current time (default: _next_time).

        Returns:
            RecallResult from the recall pipeline.
        """
        ct = current_time if current_time is not None else self._next_time

        active = self.memories
        if not active:
            return recall(
                memories=[],
                params=self.params,
                config=self._recall_config,
                message=message,
                turn_number=turn_number,
                contents=None,
                current_time=ct,
            )

        # Build MemoryState list and contents list
        memory_states: list[MemoryState] = []
        contents: list[str | None] = []
        for mem in active:
            relevance = self._simple_text_relevance(message, mem.content)
            ms = self._stored_to_memory_state(mem, relevance, ct)
            memory_states.append(ms)
            contents.append(mem.content)

        result = recall(
            memories=memory_states,
            params=self.params,
            config=self._recall_config,
            message=message,
            turn_number=turn_number,
            contents=contents,
            current_time=ct,
        )

        # Increment access_count for recalled memories
        if result.tier_assignments:
            recalled_indices = {ta.index for ta in result.tier_assignments}
            for recall_idx in recalled_indices:
                if 0 <= recall_idx < len(active):
                    mem = active[recall_idx]
                    global_idx = self._id_to_idx[mem.memory_id]
                    old = self._memories[global_idx]
                    self._memories[global_idx] = dataclasses.replace(
                        old,
                        access_count=old.access_count + 1,
                        last_access_time=ct,
                    )
            self._rebuild_index()

        return result

    # -- Consolidate ---------------------------------------------------------

    def consolidate(
        self,
        current_time: float | None = None,
        mode: ConsolidationMode = ConsolidationMode.ASYNC_BATCH,
    ) -> ConsolidationSummary:
        """Run the consolidation pipeline on active memories.

        Pipeline:
          1. Convert active memories to ConsolidationCandidate via _stored_to_candidate().
          2. Apply contested flag from contradiction history.
          3. select_consolidation_candidates(pool, config, mode, pool_size).
          4. For each selected: consolidate_memory(candidate, config).
          5. Create new StoredMemory from each SemanticExtraction.
          6. Deactivate source memories (is_active=False).  The downstream
             consolidation module's archive_episodic() tracks consolidated_to
             on its ArchivedMemory type; the orchestrator marks deactivation
             via is_active=False on StoredMemory.

        Args:
            current_time: Optional explicit current time (default: _next_time).
            mode:         ConsolidationMode (default: ASYNC_BATCH).

        Returns:
            ConsolidationSummary describing what happened.
        """
        ct = current_time if current_time is not None else self._next_time

        active = self.memories
        if not active:
            return ConsolidationSummary(
                candidates_evaluated=0,
                candidates_consolidated=0,
                new_semantic_ids=(),
                archived_ids=(),
                skipped_ids=(),
            )

        # Step 1: Convert to candidates (relative ages)
        pool: list[ConsolidationCandidate] = []
        for mem in active:
            candidate = self._stored_to_candidate(mem, ct)
            pool.append(candidate)

        # Step 2: Select eligible candidates
        selected = select_consolidation_candidates(
            pool=pool,
            config=self._consolidation_config,
            mode=mode,
            pool_size=len(pool),
        )

        if not selected:
            return ConsolidationSummary(
                candidates_evaluated=len(pool),
                candidates_consolidated=0,
                new_semantic_ids=(),
                archived_ids=(),
                skipped_ids=tuple(c.memory_id for c in pool),
            )

        # Step 3: Consolidate each selected candidate
        new_semantic_ids: list[str] = []
        archived_ids: list[str] = []
        # Use a set for O(1) membership checks during accumulation,
        # convert to tuple at the end for the frozen return value.
        skipped_ids_set: set[str] = set()

        for candidate in selected:
            result = consolidate_memory(
                candidate=candidate,
                config=self._consolidation_config,
            )

            if result.candidates_consolidated == 0:
                skipped_ids_set.add(candidate.memory_id)
                continue

            # Process extractions: create new semantic StoredMemory
            for extraction in result.extractions:
                new_id = uuid.uuid4().hex
                new_semantic_ids.append(new_id)

                new_memory = StoredMemory(
                    memory_id=new_id,
                    content=extraction.content,
                    category=extraction.category,
                    importance=extraction.importance,
                    strength=self.params.s0,
                    creation_time=ct,
                    last_access_time=ct,
                    access_count=extraction.access_count,
                    level=extraction.target_level,
                    is_active=True,
                    encoding_confidence=extraction.confidence,
                    source_episodes=extraction.source_episodes,
                    consolidation_count=extraction.consolidation_count,
                )
                self._memories.append(new_memory)

            # Archive originals
            for archived in result.archived:
                mid = archived.memory_id
                archived_ids.append(mid)
                if mid in self._id_to_idx:
                    global_idx = self._id_to_idx[mid]
                    old = self._memories[global_idx]
                    self._memories[global_idx] = dataclasses.replace(
                        old, is_active=False
                    )

        # Add non-selected candidates to skipped
        selected_ids = {c.memory_id for c in selected}
        for c in pool:
            if c.memory_id not in selected_ids and c.memory_id not in skipped_ids_set:
                skipped_ids_set.add(c.memory_id)

        self._full_rebuild_index()

        return ConsolidationSummary(
            candidates_evaluated=len(pool),
            candidates_consolidated=len(new_semantic_ids),
            new_semantic_ids=tuple(new_semantic_ids),
            archived_ids=tuple(archived_ids),
            skipped_ids=tuple(skipped_ids_set),
        )
