# Consolidation/Aging Spec -- hermes_memory/consolidation.py

## Overview

Consolidation is piece 4/5 of the Hermes memory system. It implements the
episodic-to-semantic memory aging pipeline: detecting when episodic memories
have decayed past usefulness but still contain valuable semantic content,
extracting that content into higher-level representations, and archiving the
originals. This mirrors the hippocampal-to-cortical consolidation process
from cognitive science.

This module sits downstream of encoding (piece 1), recall (piece 2), and
contradiction (piece 3). After memories are encoded and have lived through
dynamics cycles, consolidation evaluates which episodic memories should be
promoted to semantic representations. The consolidation system respects
contradiction state -- contested memories are never consolidated -- and
uses encoding categories to determine compression behavior.

New module: `hermes_memory/consolidation.py`
New test file: `tests/test_consolidation.py`

---

## 1. Dependencies

| Module | What consolidation.py uses |
|--------|---------------------------|
| `engine.py` | `MemoryState` (for reading relevance, importance, strength, access_count, creation_time, last_access_time), `ParameterSet` (for dynamics parameters) |
| `encoding.py` | `VALID_CATEGORIES`, `CATEGORY_IMPORTANCE` (for category-aware compression ratios and importance seeding) |
| `contradiction.py` | `EXCLUDED_CATEGORIES` (parallel exclusion concept for consolidation), `ContradictionType` (no direct use, but contested state concept) |
| `core.py` | `retention()` (for computing current retention of a candidate), `clamp01()` (for clamping derived scores), `importance_update()` (for computing new importance values), `strength_decay()` (for modeling archived memory decay) |

No new external dependencies. Only stdlib `dataclasses`, `enum`, `math`, `re`, `logging`.

---

## 2. Core Concepts

### 2.1 Consolidation Levels

Memory exists at four levels of abstraction, each with different retention
characteristics and detail granularity. Consolidation is the process of
promoting memory from one level to the next.

**Terminology note:** The values below are **time constants** (1/e-life), not
true half-lives. Since `retention(t, S) = exp(-t/S)`, setting `S = T` gives
`R(T) = exp(-1) = 0.368`, not 0.5. The true half-life (50% retention) occurs
at `t = S * ln(2) ~= 0.693 * S`. We retain the term "time constant" in code
(`LEVEL_TIME_CONSTANTS`) and provide half-life equivalents in documentation.
See Section 4.1 for the mapping.

```
Level 1 (EPISODIC_RAW)         -- Full conversation transcripts. Highest detail.
  |                                Time constant: 7.0 (half-life: ~4.85 days).
  |  consolidate when retention < 0.4
  v
Level 2 (EPISODIC_COMPRESSED)  -- Atomic facts extracted from episodes. Medium detail.
  |                                Time constant: 30.0 (half-life: ~20.79 days).
  |  consolidate when retention < 0.3
  v
Level 3 (SEMANTIC_FACT)        -- Extracted knowledge, no conversation context.
  |                                Time constant: 180.0 (half-life: ~124.77 days).
  |  consolidate when cluster similarity > 0.6
  |  NOTE: L3->L4 via clustering is effectively limited to near-duplicate
  |  detection in v1 (token Jaccard). Semantic clustering requires
  |  embedding-based affinity (planned for v2). See Section 16.
  v
Level 4 (SEMANTIC_INSIGHT)     -- Aggregated patterns, user profile entries.
                                   Time constant: 365.0 (half-life: ~252.97 days).
```

**Retention values at the time constant boundary (t = S):**

| Level | S | R(S) | R(2S) | R at threshold |
|-------|---|------|-------|----------------|
| EPISODIC_RAW | 7.0 | 0.368 | 0.135 | R < 0.4 at t > 6.41 |
| EPISODIC_COMPRESSED | 30.0 | 0.368 | 0.135 | R < 0.3 at t > 36.12 |
| SEMANTIC_FACT | 180.0 | 0.368 | 0.135 | R < 0.3 at t > 216.72 |
| SEMANTIC_INSIGHT | 365.0 | 0.368 | 0.135 | (no further promotion) |

Each level transition involves information compression: detail is lost but
the semantic core is preserved. The compression is category-aware (Section 8).

### 2.2 Consolidation Modes

Three modes of consolidation operate at different times in the memory lifecycle:

**INTRA_SESSION**: During the write phase of a single session. Detects memories
within the current session that are related (similarity > 0.8 AND temporal
proximity < 1 hour as measured by creation_time difference). This is the
cheapest consolidation mode -- it catches obvious duplicates and near-duplicates
before they accumulate.

**ASYNC_BATCH**: Periodic background process that scans the full memory pool.
Clusters memories by semantic + temporal affinity using the weighted affinity
function (Section 5.5). This is the primary consolidation path for promoting
Level 1 -> Level 2 -> Level 3.

**RETRIEVAL_TRIGGERED**: When an episodic memory is recalled, evaluate whether
it should be consolidated. This exploits the cognitive science concept of
reconsolidation: a retrieved memory enters a "labile window" where it can be
transformed. If the memory meets consolidation criteria, consolidate it during
this window.

### 2.3 Consolidation Triggers and Inhibitors

**Primary trigger**: Retention score has decayed below threshold while the
memory's semantic value remains high:
- `retention < config.retention_threshold` (default 0.3)
- `semantic_value >= config.semantic_value_threshold` (default 0.6)

Where `semantic_value` is derived from the memory's category importance and
access history (Section 5.2).

**Per-category consolidation eligibility** (see Section 5.2 for full analysis):

| Category | Min importance to reach sv >= 0.6 (0 accesses) | Practical? |
|----------|------------------------------------------------|------------|
| correction | N/A -- blocked by NON_CONSOLIDATABLE_CATEGORIES | Never |
| instruction | >= 0.30 | Easy -- starts at 0.85 |
| preference | >= 0.40 | Easy -- starts at 0.80 |
| fact | >= 0.80 | Requires dynamics boost from initial 0.6 |
| reasoning | IMPOSSIBLE (needs 1.20) | Only with access_count >= 15 AND importance >= 0.87 |
| greeting | IMPOSSIBLE (max sv = 0.40) | Never via primary path; only under storage pressure |
| transactional | IMPOSSIBLE (max sv = 0.40) | Never via primary path; only under storage pressure |

**Secondary triggers** (any one sufficient, in addition to primary):
- Temporal distance: `creation_time > config.consolidation_window` (default 7.0 time units)
- Retrieval frequency: `access_count < config.retrieval_frequency_threshold` (default 10) AND the memory has existed for at least `config.consolidation_window` time units
- Storage pressure: when the candidate pool size exceeds `config.max_pool_size` (default 5000)

**Inhibitors** (any one blocks consolidation):
1. Memory is marked as `contested` (from contradiction detection -- the caller passes this flag)
2. Recently created: `creation_time < config.recency_guard` (default 1.0 time unit, representing ~24 hours)
3. High activation: `access_count >= config.high_activation_threshold` (default 10) AND `last_access_time < config.recency_guard`
4. Category is `"correction"`: corrections preserve full context unconditionally (compression ratio 1:1)

### 2.4 Integration Points

```
 recall() retrieves episodic memory
          |
          v
 should_consolidate(candidate, config) ---- False --> no action
          |
          True
          v
 extract_semantic(content, category, config) --> SemanticExtraction
          |
          v
 consolidate_memory(candidate, config) --> ConsolidationResult
          |
          |---> archived_memory   (original, decayed scores, archived=True)
          +---> new_semantic      (extracted knowledge, boosted importance)
```

Batch consolidation pipeline:

```
 select_consolidation_candidates(pool, config) --> list[ConsolidationCandidate]
          |
          v
 For each candidate:
     compute_affinity(a, b, config) --> cluster by affinity
          |
          v
     consolidate_memory(candidate, config) --> ConsolidationResult
          |
          v
     archive_episodic(memory, consolidated_to, config) --> ArchivedMemory
```

---

## 3. Data Types

### 3.1 ConsolidationLevel Enum

```python
import enum

class ConsolidationLevel(enum.Enum):
    """Abstraction level of a memory in the consolidation hierarchy.

    Each level has a characteristic time constant (1/e-life) and detail
    granularity. Consolidation promotes memories from lower to higher
    levels (EPISODIC_RAW -> EPISODIC_COMPRESSED -> SEMANTIC_FACT ->
    SEMANTIC_INSIGHT).
    """
    EPISODIC_RAW = "episodic_raw"
    EPISODIC_COMPRESSED = "episodic_compressed"
    SEMANTIC_FACT = "semantic_fact"
    SEMANTIC_INSIGHT = "semantic_insight"
```

#### Invariants
- Exactly four members.
- Each member corresponds to a distinct consolidation behavior (Section 5).
- Ordering is defined: EPISODIC_RAW < EPISODIC_COMPRESSED < SEMANTIC_FACT < SEMANTIC_INSIGHT.

### 3.2 ConsolidationMode Enum

```python
class ConsolidationMode(enum.Enum):
    """Trigger context for a consolidation operation."""
    INTRA_SESSION = "intra_session"
    ASYNC_BATCH = "async_batch"
    RETRIEVAL_TRIGGERED = "retrieval_triggered"
```

#### Invariants
- Exactly three members.
- Each member corresponds to a distinct consolidation context (Section 2.2).

### 3.3 ConsolidationConfig

**Time unit assumption:** All time-based thresholds (consolidation_window,
recency_guard, intra_session_temporal) assume `delta_t = 1.0` in the dynamics
engine, where 1 time step represents approximately 1 day. If the caller uses
a different `delta_t`, these thresholds must be scaled proportionally. For
example, with `delta_t = 0.1`, set `consolidation_window = 70.0` to represent
7 days.

```python
@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration for the consolidation pipeline.

    Frozen to prevent accidental mutation. All thresholds carry domain
    constraints validated at construction time.

    Attributes -- Trigger thresholds:
        retention_threshold:          Retention below this triggers consolidation.
                                      Domain: (0.0, 1.0). Default: 0.3.
        semantic_value_threshold:     Semantic value must exceed this for consolidation.
                                      Domain: (0.0, 1.0). Default: 0.6.
        consolidation_window:         Minimum age (creation_time) for batch consolidation.
                                      Domain: > 0.0. Default: 7.0.
                                      Assumes delta_t = 1.0 (1 step ~= 1 day).
        retrieval_frequency_threshold: Access count below which a memory is
                                       consolidation-eligible (after window).
                                       Domain: >= 1. Default: 10.
                                       Rationale: typical episodic memories have
                                       0-10 accesses; 10+ indicates active use
                                       and the memory should stay episodic.
        max_pool_size:                Pool size above which storage pressure triggers.
                                      Domain: >= 1. Default: 5000.
                                      Set above the typical active pool (1000-5000
                                      memories) to avoid routine pressure bypass.

    Attributes -- Inhibitor thresholds:
        recency_guard:                Minimum age before consolidation is allowed.
                                      Domain: > 0.0. Default: 1.0.
                                      Assumes delta_t = 1.0 (1 step ~= 1 day).
        high_activation_threshold:    Access count above which a memory is considered
                                      highly active (blocks consolidation if also recent).
                                      Domain: >= 1. Default: 10.

    Attributes -- Level transition thresholds:
        l1_to_l2_retention:           Retention threshold for L1 -> L2 promotion.
                                      Domain: (0.0, 1.0). Default: 0.4.
        l2_to_l3_retention:           Retention threshold for L2 -> L3 promotion.
                                      Domain: (0.0, 1.0). Default: 0.3.
        l3_to_l4_cluster_sim:         Cluster similarity threshold for L3 -> L4.
                                      Domain: (0.0, 1.0]. Default: 0.6.
                                      NOTE: With token Jaccard similarity (v1),
                                      this only fires for near-duplicate content.
                                      See Section 16 for limitations.

    Attributes -- Process parameters:
        consolidation_decay:          Decay multiplier applied to archived memory scores.
                                      Domain: (0.0, 1.0). Default: 0.1.
        semantic_boost:               Importance multiplier for new semantic memories.
                                      Domain: (1.0, 2.0]. Default: 1.2.
        activation_inheritance:       Fraction of source activation inherited by semantic.
                                      Domain: [0.0, 1.0]. Default: 0.5.

    Attributes -- Clustering (batch mode):
        semantic_weight_beta:         Weight of semantic similarity in affinity function.
                                      Domain: [0.0, 1.0]. Default: 0.7.
        temporal_decay_lambda:        Decay rate for temporal distance in affinity.
                                      Domain: > 0.0. Default: 0.1.
        cluster_threshold:            Minimum affinity to cluster two memories.
                                      Domain: (0.0, 1.0]. Default: 0.6.
                                      Lowered from 0.75 to catch substring matches
                                      and near-duplicates with minor variation.
        min_content_similarity:       Minimum content similarity required for clustering.
                                      If content_sim < this value, affinity is forced to 0.0
                                      regardless of temporal proximity. Prevents clustering of
                                      semantically unrelated memories based on creation time alone.
                                      Domain: [0.0, 1.0]. Default: 0.3.

    Attributes -- Intra-session mode:
        intra_session_similarity:     Similarity threshold for intra-session consolidation.
                                      Domain: (0.0, 1.0]. Default: 0.8.
        intra_session_temporal:       Maximum creation_time difference for intra-session.
                                      Domain: > 0.0. Default: 1.0.
                                      Assumes delta_t = 1.0 (1 step ~= 1 day).

    Attributes -- Storage pressure (batch mode):
        mild_pressure_retention:      Under mild storage pressure (pool_size between
                                      max_pool_size and 2*max_pool_size), use this
                                      relaxed retention threshold instead of
                                      retention_threshold.
                                      Domain: (0.0, 1.0). Default: 0.5.
        severe_pressure_multiplier:   Pool size multiplier above which severe pressure
                                      applies (bypasses retention/semantic_value).
                                      Domain: > 1.0. Default: 2.0.

    Attributes -- General:
        max_candidates_per_batch:     Maximum candidates to process in one batch.
                                      Domain: >= 1. Default: 100.
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
```

#### Validation Rules (enforced in `__post_init__`)

```
0.0 < retention_threshold < 1.0
0.0 < semantic_value_threshold < 1.0
consolidation_window > 0.0
retrieval_frequency_threshold >= 1
max_pool_size >= 1
recency_guard > 0.0
high_activation_threshold >= 1
0.0 < l1_to_l2_retention < 1.0
0.0 < l2_to_l3_retention < 1.0
0.0 < l3_to_l4_cluster_sim <= 1.0
0.0 < consolidation_decay < 1.0
1.0 < semantic_boost <= 2.0
0.0 <= activation_inheritance <= 1.0
0.0 <= semantic_weight_beta <= 1.0
temporal_decay_lambda > 0.0
0.0 < cluster_threshold <= 1.0
0.0 <= min_content_similarity <= 1.0
0.0 < intra_session_similarity <= 1.0
intra_session_temporal > 0.0
0.0 < mild_pressure_retention < 1.0
severe_pressure_multiplier > 1.0
max_candidates_per_batch >= 1
l2_to_l3_retention <= l1_to_l2_retention   (stricter threshold for deeper consolidation)
mild_pressure_retention >= retention_threshold  (mild pressure is more lenient)
```

The last constraint ensures the hierarchy is monotonically harder to enter at
deeper levels: a memory must decay further to move from L2 to L3 than from
L1 to L2.

### 3.4 ConsolidationCandidate

```python
@dataclass(frozen=True)
class ConsolidationCandidate:
    """A memory being evaluated for consolidation.

    Wraps a MemoryState with additional metadata needed for the
    consolidation decision. The caller constructs these from the memory
    store; the consolidation module does not access the store directly.

    Attributes:
        index:           Position in the memory pool (for back-reference).
        content:         Full text content of the memory.
        category:        Category from encoding (one of VALID_CATEGORIES).
        level:           Current consolidation level.
        creation_time:   Time steps since memory was created (from MemoryState).
        last_access_time: Time steps since last access (from MemoryState).
        importance:      Current importance score (from MemoryState). Domain: [0, 1].
        access_count:    Total access count (from MemoryState). Domain: >= 0.
        strength:        Current memory strength (from MemoryState). Domain: >= 0.
        contested:       True if the memory is flagged as contested by
                         contradiction detection. Contested memories are
                         NEVER consolidated.
        source_episodes: Tuple of episode IDs that contributed to this memory.
                         Empty for L1 (raw episodes are their own source).
        consolidation_count: Number of times this memory has been consolidated
                             from lower levels. 0 for L1 memories.
    """
    index: int
    content: str
    category: str
    level: ConsolidationLevel
    creation_time: float
    last_access_time: float
    importance: float
    access_count: int
    strength: float
    contested: bool
    source_episodes: tuple[str, ...] = ()
    consolidation_count: int = 0
```

#### Validation Rules (enforced in `__post_init__`)

```
index >= 0
len(content) > 0
category in VALID_CATEGORIES
0.0 <= importance <= 1.0
access_count >= 0
strength >= 0.0
creation_time >= 0.0
last_access_time >= 0.0
consolidation_count >= 0
```

### 3.5 SemanticExtraction

```python
@dataclass(frozen=True)
class SemanticExtraction:
    """Extracted semantic knowledge from one or more episodic memories.

    This is the compressed representation produced by extract_semantic().
    It captures the core knowledge without conversational context.

    Attributes:
        content:             The extracted semantic content (text).
        category:            The category of the extracted content.
                             Inherited from source; may be different if the
                             extraction produces a more specific categorization.
        source_episodes:     Tuple of episode/memory IDs that contributed.
        confidence:          Aggregate confidence from source memories.
                             Domain: [0.0, 1.0]. Computed as mean importance
                             of sources, clamped.
        first_observed:      Earliest creation_time among sources.
        last_updated:        Latest creation_time among sources.
        consolidation_count: Number of source memories consolidated.
                             Domain: >= 1.
        compression_ratio:   Ratio of source content length to extracted
                             content length. Domain: > 0.0. A ratio of 5.0
                             means 5x compression. A ratio < 1.0 indicates
                             EXPANSION (the extracted content is longer than
                             the source, typically due to template prefixes
                             applied to short content). This is expected for
                             inputs shorter than ~50 characters.
    """
    content: str
    category: str
    source_episodes: tuple[str, ...]
    confidence: float
    first_observed: float
    last_updated: float
    consolidation_count: int
    compression_ratio: float
```

#### Validation Rules

```
len(content) > 0
category in VALID_CATEGORIES
len(source_episodes) >= 1
0.0 <= confidence <= 1.0
first_observed >= 0.0
last_updated >= first_observed
consolidation_count >= 1
compression_ratio > 0.0
```

### 3.6 ArchivedMemory

```python
@dataclass(frozen=True)
class ArchivedMemory:
    """An episodic memory that has been archived after consolidation.

    Archived memories are NOT deleted. They remain in the memory store
    but are excluded from standard recall scoring. Retrieval of archived
    memories requires a dedicated archive query path that bypasses
    normal scoring (see Section 7.5 for integration requirements).

    Their scores are decayed to reduce ranking weight. With default
    consolidation_decay=0.1, a memory with importance=0.7 and
    strength=7.0 becomes importance=0.07 and strength=0.7, making it
    effectively invisible to standard scoring. This is intentional --
    archived memories are historical records, not active knowledge.

    Attributes:
        index:             Original index in the memory pool.
        content:           Original content (preserved in full).
        category:          Original category.
        level:             Original consolidation level (before archival).
        archived:          Always True.
        consolidated_to:   Tuple of target memory IDs or indices that this
                           memory was consolidated into.
        decayed_importance: Importance after consolidation_decay applied.
                            Domain: [0.0, 1.0].
        decayed_strength:  Strength after consolidation_decay applied.
                           Domain: >= 0.0.
        original_importance: Importance before archival (for audit trail).
        original_strength:   Strength before archival (for audit trail).
    """
    index: int
    content: str
    category: str
    level: ConsolidationLevel
    archived: bool  # always True
    consolidated_to: tuple[int, ...]
    decayed_importance: float
    decayed_strength: float
    original_importance: float
    original_strength: float
```

#### Invariants

```
archived is True                           # structural invariant
0.0 <= decayed_importance <= 1.0
decayed_strength >= 0.0
decayed_importance <= original_importance   # decay never increases
decayed_strength <= original_strength       # decay never increases
len(consolidated_to) >= 1                   # must point to at least one target
```

**Provenance Chain Integrity (AP3-F7)**: Every ArchivedMemory must point to
retrievable semantic memories via `consolidated_to`. If a target semantic memory
is itself later consolidated (e.g., L2->L3), the caller MUST update the original
archived memory's `consolidated_to` to point to the new semantic target. Without
this, the promise that "archived memories remain retrievable" (Section 3.6
docstring) is broken -- following `consolidated_to` leads to another archived
memory rather than live content.

The consolidation module does NOT maintain this chain integrity automatically --
it is the caller's (memory store's) responsibility to update `consolidated_to`
references when a consolidation target is itself consolidated. A property-based
test should verify that traversing `consolidated_to` always terminates at a live
(non-archived) memory.

### 3.7 ConsolidationResult

```python
@dataclass(frozen=True)
class ConsolidationResult:
    """Output of the full consolidation pipeline for a single candidate
    or a batch of candidates.

    Attributes:
        extractions:       Tuple of SemanticExtraction objects produced.
        archived:          Tuple of ArchivedMemory objects (source memories archived).
        skipped_indices:   Frozenset of candidate indices that were evaluated
                           but not consolidated (inhibitor triggered or below threshold).
        consolidated_count: Number of memories successfully consolidated.
        mode:              The consolidation mode that produced this result.
    """
    extractions: tuple[SemanticExtraction, ...]
    archived: tuple[ArchivedMemory, ...]
    skipped_indices: frozenset[int]
    consolidated_count: int
    mode: ConsolidationMode
```

#### Invariants

```
consolidated_count == len(archived)
consolidated_count == len(extractions)   # each archived memory produces exactly one extraction
consolidated_count >= 0
```

Note on the 1:1 invariant: each archived source memory produces exactly one
SemanticExtraction. When multiple source memories are clustered together (batch
mode), the cluster is treated as a single consolidation unit: the individual
source memories are each archived, but only one SemanticExtraction is produced
for the cluster. In this case, `len(archived) >= len(extractions)` and
`consolidated_count == len(archived)`. The corrected invariant is:

```
consolidated_count == len(archived)
len(extractions) <= len(archived)        # clusters: N sources -> 1 extraction
len(extractions) >= 1 if consolidated_count >= 1
consolidated_count >= 0
```

Note on clustering: each archived source memory does NOT necessarily produce
its own SemanticExtraction. When multiple source memories are clustered
together (batch mode), the cluster is treated as a single consolidation unit:
the individual source memories are each archived, but only one
SemanticExtraction is produced for the cluster. Therefore
`len(extractions) <= len(archived)`, not equality.

---

## 4. Constants

### 4.1 Level Time Constants

```python
LEVEL_TIME_CONSTANTS: dict[ConsolidationLevel, float] = {
    ConsolidationLevel.EPISODIC_RAW: 7.0,
    ConsolidationLevel.EPISODIC_COMPRESSED: 30.0,
    ConsolidationLevel.SEMANTIC_FACT: 180.0,
    ConsolidationLevel.SEMANTIC_INSIGHT: 365.0,
}
```

These are **time constants** (1/e-life), not half-lives. The retention function
is `R(t) = exp(-t/S)` where S is the time constant. Key properties:

| Level | Time Constant S | True Half-Life (S * ln2) | R(S) | R(2S) |
|-------|-----------------|--------------------------|------|-------|
| EPISODIC_RAW | 7.0 | 4.85 | 0.368 | 0.135 |
| EPISODIC_COMPRESSED | 30.0 | 20.79 | 0.368 | 0.135 |
| SEMANTIC_FACT | 180.0 | 124.77 | 0.368 | 0.135 |
| SEMANTIC_INSIGHT | 365.0 | 252.97 | 0.368 | 0.135 |

A convenience alias is provided for documentation:

```python
import math
LEVEL_HALF_LIVES: dict[ConsolidationLevel, float] = {
    level: tc * math.log(2) for level, tc in LEVEL_TIME_CONSTANTS.items()
}
# EPISODIC_RAW: 4.85, EPISODIC_COMPRESSED: 20.79, ...
```

The actual retention is computed via `core.retention(last_access_time, strength)`
where `strength` encodes the time constant. The caller is responsible for
setting strength based on level when creating memories. The consolidation
module uses these values to determine the NEXT level's target strength when
promoting a memory: `new_strength = LEVEL_TIME_CONSTANTS[target_level]`.

### 4.2 Category Compression Ratios

```python
CATEGORY_COMPRESSION: dict[str, float] = {
    "preference": 10.0,   # "User prefers X" -- discard instance details
    "fact": 5.0,          # "User's Y is Z" -- preserve fact, discard conversation
    "reasoning": 3.0,     # Preserve reasoning chain, compress context
    "instruction": 5.0,   # "Always do X" -- preserve directive, discard context
    "correction": 1.0,    # DO NOT compress -- preserve full context
    "greeting": 10.0,     # Trivial content (should rarely reach consolidation)
    "transactional": 10.0, # Trivial content (should rarely reach consolidation)
}
```

The compression ratio defines the TARGET ratio of source content length to
extracted content length. A ratio of 10.0 means the extracted content should
be ~1/10th the length of the source. The actual ratio depends on content;
this is a target, not an exact constraint.

**Note on short content:** For inputs shorter than ~50 characters, the
template prefix ("User prefers: ", "User fact: ", etc.) can cause the
extracted content to be LONGER than the source. In this case,
`SemanticExtraction.compression_ratio` will be < 1.0, indicating expansion.
This is expected behavior -- the template provides consistent formatting
at the cost of a few extra characters on short memories.

**Critical**: `"correction"` has ratio 1.0 -- corrections are NEVER compressed.
This is enforced by the inhibitor check (Section 2.3, inhibitor 4) which
prevents corrections from entering the consolidation pipeline at all. The
ratio 1.0 is a defense-in-depth measure: even if a correction bypasses the
inhibitor check due to a bug, extraction will preserve the full content.

### 4.3 Category-Specific Extraction Patterns

Each category has patterns that guide semantic extraction. These are used by
`extract_semantic()` to identify the core knowledge in the source text.

```python
PREFERENCE_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:the user |i )(?:prefer|like|enjoy|love|favor)s?\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:the user |i )(?:don't|do not|doesn't|does not) (?:like|enjoy|want)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:the user's |my )favorite\s+\w+\s+is\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:the user |i )(?:would |'d )rather\s+(.+)", re.IGNORECASE),
]

FACT_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:the user |i )(?:live|reside|am based|is based|is located)s?\s+(?:in|at)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:the user's |my )name is\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:the user |i )(?:work|works|am employed)\s+(?:at|for|as)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:the user's |my )(?:email|phone|address)\s+is\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:the user |i )(?:speak|speaks)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:the user |i )(?:have|has)\s+(?:a|an)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:the user |i )(?:studied|graduated from|attended)\s+(.+)", re.IGNORECASE),
]

INSTRUCTION_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:always|never|from now on,?\s*|remember (?:to|that)\s+)(.+)", re.IGNORECASE),
    re.compile(r"(?:the user (?:instructed|requested|directed|asked)(?: the assistant)? to\s+)(.+)", re.IGNORECASE),
    re.compile(r"(?:going forward,?\s*|in the future,?\s*)(.+)", re.IGNORECASE),
]

REASONING_EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:because|since|due to|given that)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(.+?)(?:therefore|thus|hence|consequently|as a result)\s+(.+)", re.IGNORECASE),
]
```

### 4.4 Semantic Templates

When extraction patterns match, the extracted content is formatted using
category-specific templates to produce a consistent semantic memory format:

```python
SEMANTIC_TEMPLATES: dict[str, str] = {
    "preference": "User prefers: {extracted}",
    "fact": "User fact: {extracted}",
    "instruction": "Standing instruction: {extracted}",
    "reasoning": "User reasoning: {extracted}",
    "correction": "{original}",  # Full content preserved
    "greeting": "{extracted}",   # Rare; just pass through
    "transactional": "{extracted}",  # Rare; just pass through
}
```

### 4.5 Default Config

```python
DEFAULT_CONSOLIDATION_CONFIG: ConsolidationConfig = ConsolidationConfig()
```

Module-level instance with all defaults. Used when `config=None` is passed
to public functions.

### 4.6 Non-Consolidatable Categories

```python
NON_CONSOLIDATABLE_CATEGORIES: frozenset[str] = frozenset({"correction"})
```

Categories that are NEVER consolidated. Parallel to contradiction.py's
`EXCLUDED_CATEGORIES` (which excludes greeting/transactional from contradiction
detection). Corrections must preserve full conversational context to be useful,
so they are excluded from the compression pipeline entirely.

Note: `greeting` and `transactional` are not in this set because they should
already have been filtered by the encoding gate (piece 1). If they somehow
reach consolidation, they ARE eligible for consolidation (and will be heavily
compressed per their 10.0 compression ratio). This is defense-in-depth.
However, note that greeting/transactional can NEVER meet the semantic_value
threshold of 0.6 (their max possible semantic_value is 0.40 -- see Section
5.2). They can only enter the consolidation pipeline via storage pressure
bypass (Section 5.6), which is an intentional design choice: under pressure,
clean up trivial content; under normal conditions, let it decay naturally.

### 4.7 Correction Content Markers

Defense-in-depth markers for detecting corrections based on content rather than
category label. Used by inhibitor I5 to catch corrections that were miscategorized
by encoding.py.

```python
CORRECTION_CONTENT_MARKERS: frozenset[re.Pattern[str]] = frozenset([
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\bnot\s+\w+\s+but\s+\w+\b", re.IGNORECASE),
    re.compile(r"\bi\s+was\s+wrong\b", re.IGNORECASE),
    re.compile(r"\bcorrection:", re.IGNORECASE),
    re.compile(r"\bto\s+clarify\b", re.IGNORECASE),
    re.compile(r"\blet\s+me\s+correct\b", re.IGNORECASE),
])

CORRECTION_MARKER_THRESHOLD: int = 2
```

If a memory's content matches at least `CORRECTION_MARKER_THRESHOLD` patterns
from `CORRECTION_CONTENT_MARKERS`, it is treated as a correction regardless of
its category label. This prevents inhibitor bypass via miscategorization.

---

## 5. Functions

### 5.1 should_consolidate()

```python
def should_consolidate(
    candidate: ConsolidationCandidate,
    config: ConsolidationConfig | None = None,
) -> bool:
    """Evaluate whether a memory candidate should be consolidated.

    Checks all triggers and inhibitors from Section 2.3. Returns True
    if at least one trigger fires AND no inhibitor blocks.

    Args:
        candidate:  The memory to evaluate.
        config:     ConsolidationConfig. Uses DEFAULT_CONSOLIDATION_CONFIG if None.

    Returns:
        True if the memory should be consolidated, False otherwise.

    Raises:
        TypeError:  If candidate is not a ConsolidationCandidate.
        ValueError: If candidate.category is not in VALID_CATEGORIES.
    """
```

#### Algorithm

```
def should_consolidate(candidate, config):
    config = config or DEFAULT_CONSOLIDATION_CONFIG

    # --- Inhibitor checks (delegated to shared helper) ---
    # AP3-F5: Use shared _passes_inhibitors() to ensure consistency
    if not _passes_inhibitors(candidate, config):
        return False

    # --- Compute retention ---
    # Use strength as the decay parameter; guard against zero
    safe_strength = max(candidate.strength, 1e-12)
    current_retention = retention(candidate.last_access_time, safe_strength)

    # --- Compute semantic value ---
    semantic_value = _compute_semantic_value(candidate)

    # --- Primary trigger: retention low, semantic value high ---
    primary = (
        current_retention < config.retention_threshold
        and semantic_value >= config.semantic_value_threshold
    )

    # --- Secondary triggers (any sufficient, combined with primary) ---
    temporal = candidate.creation_time > config.consolidation_window
    infrequent = (
        candidate.access_count < config.retrieval_frequency_threshold
        and candidate.creation_time > config.consolidation_window
    )
    # Storage pressure is not checked here -- it is the caller's
    # responsibility to pass pool_size context. The select_consolidation_candidates
    # function handles storage pressure.

    # Primary trigger must be met; secondary triggers are additive context
    # that lowers the bar but does NOT independently trigger consolidation.
    # Exception: temporal + infrequent together can trigger even if retention
    # is above threshold, provided semantic_value is still high.
    if primary:
        return True

    if temporal and infrequent and semantic_value >= config.semantic_value_threshold:
        return True

    return False
```

#### Behavioral Contracts

- `should_consolidate(contested_candidate, _)` is always `False`, regardless of other fields.
- `should_consolidate(recent_candidate, _)` is always `False` when `creation_time < recency_guard`.
- `should_consolidate(correction_candidate, _)` is always `False` when `category == "correction"`.
- `should_consolidate(candidate, _)` is always `False` when content matches CORRECTION_MARKER_THRESHOLD or more correction patterns (inhibitor I5).
- `should_consolidate(candidate, config)` is deterministic: same inputs always produce same output.
- The function never raises for valid inputs. Invalid inputs (wrong types, invalid categories) raise TypeError or ValueError.

### 5.2 _compute_semantic_value() (private)

```python
def _compute_semantic_value(candidate: ConsolidationCandidate) -> float:
    """Compute the semantic value of a memory for consolidation decisions.

    Semantic value represents how much useful knowledge the memory contains,
    independent of its current retention/accessibility.

    Formula:
        semantic_value = clamp01(
            category_importance * 0.6
            + importance * 0.3
            + access_bonus * 0.1
        )

    Where:
        category_importance = CATEGORY_IMPORTANCE[candidate.category]
        importance = candidate.importance (current dynamics importance)
        access_bonus = min(1.0, candidate.access_count / 20.0)

    The weights (0.6, 0.3, 0.1) reflect that:
    - Category is the strongest signal (corrections and instructions
      have inherently high semantic value)
    - Current importance reflects dynamics feedback
    - Access count provides a weak signal of past utility

    Returns: float in [0.0, 1.0].
    """
```

#### Per-Category Effective Thresholds

The semantic_value_threshold of 0.6 interacts with CATEGORY_IMPORTANCE to
create different effective consolidation bars per category. The following
table shows the minimum importance needed to reach sv >= 0.6 at various
access levels:

| Category | cat_imp | Min importance (0 access) | Min importance (10 access) | Min importance (20+ access) | Can ever consolidate? |
|----------|---------|---------------------------|----------------------------|-----------------------------|----------------------|
| correction | 0.90 | N/A | N/A | N/A | NO (blocked by inhibitor) |
| instruction | 0.85 | 0.30 | 0.13 | auto-pass | YES -- easy |
| preference | 0.80 | 0.40 | 0.23 | 0.07 | YES -- easy |
| fact | 0.60 | 0.80 | 0.63 | 0.47 | YES -- needs dynamics boost |
| reasoning | 0.40 | IMPOSSIBLE (1.20) | IMPOSSIBLE (1.03) | 0.87 | BARELY -- needs 15+ accesses AND high importance |
| greeting | 0.00 | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE (max sv=0.40) | NO via primary path |
| transactional | 0.00 | IMPOSSIBLE | IMPOSSIBLE | IMPOSSIBLE (max sv=0.40) | NO via primary path |

**Cold-start gap for facts:** A fact encoded with default importance (0.6) and
zero accesses has sv = 0.36 + 0.18 = 0.54, which is below 0.6. If the fact
decays past the retention threshold before receiving any positive feedback
or accesses, it will fail BOTH gates and decay to zero without consolidation.
Callers should monitor for unaccessed facts approaching retention threshold
and consider boosting their importance via the dynamics feedback loop.

**Greeting/transactional exclusion:** These categories have cat_imp=0.0 and
max possible sv=0.40. They can NEVER consolidate via the primary trigger path.
They are only consolidated under storage pressure bypass (Section 5.6). This
is intentional -- trivial content should not occupy semantic memory slots
unless the system is under storage pressure and needs to clean up.

### 5.3 extract_semantic()

```python
def extract_semantic(
    content: str,
    category: str,
    config: ConsolidationConfig | None = None,
    source_episodes: tuple[str, ...] = (),
    source_creation_times: tuple[float, ...] = (),
    source_importances: tuple[float, ...] = (),
) -> SemanticExtraction:
    """Extract semantic knowledge from episodic content.

    Uses category-specific pattern matching to identify the core knowledge
    in the source text and compress it according to the category's
    compression ratio. This is a deterministic, rule-based extraction
    with NO LLM calls.

    Args:
        content:                The source text to extract from.
        category:               Category of the source memory.
        config:                 ConsolidationConfig. Uses defaults if None.
        source_episodes:        Tuple of source episode IDs. Must have len >= 1.
        source_creation_times:  Tuple of creation times for each source.
                                Must have same length as source_episodes.
        source_importances:     Tuple of importance values for each source.
                                Must have same length as source_episodes.

    Returns:
        SemanticExtraction with the compressed content and provenance.

    Raises:
        TypeError:  If content is not a str.
        ValueError: If category is not in VALID_CATEGORIES.
        ValueError: If source_episodes is empty.
        ValueError: If source_creation_times and source_episodes have different lengths.
        ValueError: If source_importances and source_episodes have different lengths.
    """
```

#### Algorithm

```
def extract_semantic(content, category, config, source_episodes,
                     source_creation_times, source_importances):
    config = config or DEFAULT_CONSOLIDATION_CONFIG

    # 1. Select extraction patterns for category
    patterns = _get_extraction_patterns(category)

    # 2. Attempt pattern-based extraction
    extracted = None
    for pattern in patterns:
        match = pattern.search(content)
        if match:
            # Use the longest captured group
            groups = [g for g in match.groups() if g]
            if groups:
                extracted = max(groups, key=len).strip()
                break

    # 3. Fallback: if no pattern matched, use sentence scoring
    if extracted is None:
        target_len = max(1, int(len(content) / CATEGORY_COMPRESSION.get(category, 5.0)))
        extracted = _extract_by_sentence_scoring(content, target_len)

    # 4. Apply template
    template = SEMANTIC_TEMPLATES.get(category, "{extracted}")
    if category == "correction":
        # Corrections: preserve full content
        formatted = content
    else:
        formatted = template.format(extracted=extracted, original=content)

    # 5. Compute provenance
    confidence = clamp01(
        sum(source_importances) / max(len(source_importances), 1)
    )
    first_observed = min(source_creation_times) if source_creation_times else 0.0
    last_updated = max(source_creation_times) if source_creation_times else 0.0
    consolidation_count = len(source_episodes)
    compression_ratio = len(content) / max(len(formatted), 1)

    # 6. Build and return
    return SemanticExtraction(
        content=formatted,
        category=category,
        source_episodes=source_episodes,
        confidence=confidence,
        first_observed=first_observed,
        last_updated=last_updated,
        consolidation_count=consolidation_count,
        compression_ratio=max(compression_ratio, 0.01),  # guard against zero
    )
```

#### Behavioral Contracts

- `extract_semantic(content, "correction", ...)` always returns `content` unchanged (compression ratio ~1.0).
- The returned `SemanticExtraction.content` is NEVER empty.
- `confidence` is the mean of `source_importances`, clamped to [0.0, 1.0].
- `first_observed <= last_updated` (guaranteed by min/max computation).
- `consolidation_count == len(source_episodes)`.
- When no extraction pattern matches, the fallback uses sentence-scoring to preserve information-dense sentences rather than blind truncation.
- The function is deterministic: same inputs always produce same output.
- For short content (< ~50 chars), compression_ratio may be < 1.0 (expansion). This is expected.

### 5.4 consolidate_memory()

```python
def consolidate_memory(
    candidate: ConsolidationCandidate,
    config: ConsolidationConfig | None = None,
) -> ConsolidationResult:
    """Execute the full consolidation pipeline for a single candidate.

    This is the main orchestrator for single-memory consolidation.
    It calls should_consolidate(), extract_semantic(), and
    archive_episodic() in sequence.

    If should_consolidate() returns False, returns an empty
    ConsolidationResult with the candidate's index in skipped_indices.

    Args:
        candidate:  The memory to consolidate.
        config:     ConsolidationConfig. Uses defaults if None.

    Returns:
        ConsolidationResult with exactly one extraction and one archived
        memory if consolidation succeeded, or an empty result with the
        candidate in skipped_indices if consolidation was inhibited.

    Raises:
        TypeError:  If candidate is not a ConsolidationCandidate.
    """
```

#### Algorithm

```
def consolidate_memory(candidate, config):
    config = config or DEFAULT_CONSOLIDATION_CONFIG

    # 1. Check eligibility
    if not should_consolidate(candidate, config):
        return ConsolidationResult(
            extractions=(),
            archived=(),
            skipped_indices=frozenset({candidate.index}),
            consolidated_count=0,
            mode=ConsolidationMode.RETRIEVAL_TRIGGERED,  # default for single
        )

    # 2. Determine target level
    target_level = _next_level(candidate.level)
    if target_level is None:
        # Already at SEMANTIC_INSIGHT -- cannot promote further
        return ConsolidationResult(
            extractions=(),
            archived=(),
            skipped_indices=frozenset({candidate.index}),
            consolidated_count=0,
            mode=ConsolidationMode.RETRIEVAL_TRIGGERED,
        )

    # 3. Extract semantic content
    source_episodes = candidate.source_episodes or (str(candidate.index),)
    extraction = extract_semantic(
        content=candidate.content,
        category=candidate.category,
        config=config,
        source_episodes=source_episodes,
        source_creation_times=(candidate.creation_time,),
        source_importances=(candidate.importance,),
    )

    # 4. Archive the original
    archived = archive_episodic(
        candidate=candidate,
        consolidated_to=(candidate.index,),  # self-reference; caller assigns real IDs
        config=config,
    )

    # 5. Return result
    return ConsolidationResult(
        extractions=(extraction,),
        archived=(archived,),
        skipped_indices=frozenset(),
        consolidated_count=1,
        mode=ConsolidationMode.RETRIEVAL_TRIGGERED,
    )
```

#### Behavioral Contracts

- If `should_consolidate()` returns False, `consolidated_count == 0` and candidate index is in `skipped_indices`.
- If consolidation succeeds, `consolidated_count == 1`, `len(extractions) == 1`, `len(archived) == 1`.
- The result's `mode` defaults to `RETRIEVAL_TRIGGERED` for single-memory consolidation. Batch callers should override this.
- A SEMANTIC_INSIGHT memory cannot be consolidated further; returns empty result.
- The function is deterministic: same inputs always produce same output.

### 5.5 compute_affinity()

```python
def compute_affinity(
    memory_a: ConsolidationCandidate,
    memory_b: ConsolidationCandidate,
    config: ConsolidationConfig | None = None,
) -> float:
    """Compute semantic + temporal clustering affinity between two memories.

    The affinity function combines content similarity and temporal proximity:

        affinity = beta * content_sim + (1 - beta) * temporal_sim

    Where:
        content_sim = token-level Jaccard similarity of content (lowercased)
        temporal_sim = exp(-lambda * |t_a - t_b|)
        beta = config.semantic_weight_beta (default 0.7)
        lambda = config.temporal_decay_lambda (default 0.1)

    IMPORTANT: Token Jaccard similarity only detects near-duplicate content.
    Semantically equivalent memories with different surface forms (e.g.,
    "works at Anthropic" vs "employed at Anthropic") will have low Jaccard
    scores (~0.25) and will NOT cluster. See Section 16 for limitations.
    Embedding-based affinity is planned for v2.

    Args:
        memory_a:  First memory.
        memory_b:  Second memory.
        config:    ConsolidationConfig. Uses defaults if None.

    Returns:
        Affinity score in [0.0, 1.0].

    Raises:
        TypeError: If either argument is not a ConsolidationCandidate.
    """
```

#### Algorithm

```
def compute_affinity(memory_a, memory_b, config):
    config = config or DEFAULT_CONSOLIDATION_CONFIG

    # 1. Content similarity: token-level Jaccard
    tokens_a = set(memory_a.content.lower().split())
    tokens_b = set(memory_b.content.lower().split())
    if not tokens_a and not tokens_b:
        content_sim = 0.0
    elif not tokens_a or not tokens_b:
        content_sim = 0.0
    else:
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        content_sim = intersection / union

    # 2. Content similarity floor check (AP3-F2: prevent purely temporal clustering)
    if content_sim < config.min_content_similarity:
        return 0.0

    # 3. Temporal similarity: exponential decay of time difference
    time_diff = abs(memory_a.creation_time - memory_b.creation_time)
    temporal_sim = math.exp(-config.temporal_decay_lambda * time_diff)

    # 4. Weighted combination
    beta = config.semantic_weight_beta
    affinity = beta * content_sim + (1.0 - beta) * temporal_sim

    return clamp01(affinity)
```

#### Behavioral Contracts

- `compute_affinity(a, b, config) == compute_affinity(b, a, config)` (symmetric).
- `compute_affinity(a, a, config) >= config.semantic_weight_beta` (self-affinity is at least the semantic weight, since Jaccard of a set with itself is 1.0 and temporal_sim is 1.0 when time_diff is 0).
- When content is identical and creation_time is identical, affinity is 1.0.
- When content has no token overlap and creation_time difference is large, affinity approaches 0.0.
- When `content_sim < min_content_similarity`, affinity is forced to 0.0 regardless of temporal proximity (AP3-F2: prevents clustering unrelated memories).
- Return value is always in [0.0, 1.0] (enforced by clamp01 and min_content_similarity floor).
- Temporal similarity alone can NEVER reach cluster_threshold due to min_content_similarity floor (default 0.3). Clustering always requires substantial content overlap.
- The function is deterministic.

### 5.6 select_consolidation_candidates()

```python
def select_consolidation_candidates(
    pool: list[ConsolidationCandidate],
    config: ConsolidationConfig | None = None,
    mode: ConsolidationMode = ConsolidationMode.ASYNC_BATCH,
    pool_size: int | None = None,
) -> list[ConsolidationCandidate]:
    """Select memories from a pool that are eligible for consolidation.

    Applies should_consolidate() to each candidate in the pool and returns
    those that pass. For ASYNC_BATCH mode, also applies storage pressure
    detection when pool_size is provided.

    Storage pressure operates at two levels:
    - Mild pressure (pool_size > max_pool_size): retention threshold is
      relaxed to mild_pressure_retention (default 0.5), but semantic_value
      gate still applies. This allows memories that have decayed somewhat
      but not past the strict threshold.
    - Severe pressure (pool_size > max_pool_size * severe_pressure_multiplier):
      semantic_value gate is bypassed. Any non-contested, non-correction
      memory past the consolidation window is eligible. This is the
      emergency cleanup path.

    The returned list is sorted by retention (ascending) -- memories with
    the lowest retention are consolidated first, as they are most at risk
    of becoming irretrievable.

    Args:
        pool:       List of ConsolidationCandidate objects.
        config:     ConsolidationConfig. Uses defaults if None.
        mode:       The consolidation mode (affects selection criteria).
        pool_size:  Total pool size (for storage pressure detection).
                    Only used in ASYNC_BATCH mode. If None, no storage
                    pressure is applied.

    Returns:
        List of ConsolidationCandidate objects, sorted by retention ascending.
        Length is at most config.max_candidates_per_batch.

    Raises:
        TypeError: If pool is not a list of ConsolidationCandidate.
    """
```

#### Algorithm

```
def select_consolidation_candidates(pool, config, mode, pool_size):
    config = config or DEFAULT_CONSOLIDATION_CONFIG

    candidates = []
    for candidate in pool:
        eligible = False

        if mode == ConsolidationMode.INTRA_SESSION:
            # Intra-session: skip the normal should_consolidate check.
            # Instead, look for duplicates/near-duplicates within session.
            # The caller is responsible for filtering to same-session memories.
            # Here we just check that inhibitors are not active.
            # AP3-F5: Use shared _passes_inhibitors() for consistency
            if _passes_inhibitors(candidate, config):
                eligible = True

        elif mode == ConsolidationMode.ASYNC_BATCH:
            # Check standard eligibility first
            if should_consolidate(candidate, config):
                eligible = True
            elif pool_size is not None and _passes_inhibitors(candidate, config):
                # Graduated storage pressure (AP3-F5: use shared inhibitor check)
                severe_threshold = int(config.max_pool_size * config.severe_pressure_multiplier)
                mild = pool_size > config.max_pool_size
                severe = pool_size > severe_threshold

                if severe:
                    # Severe pressure: consolidate anything past the window
                    # (inhibitors already checked above)
                    if candidate.creation_time > config.consolidation_window:
                        eligible = True
                elif mild:
                    # Mild pressure: relax retention threshold but keep
                    # semantic_value gate
                    safe_strength = max(candidate.strength, 1e-12)
                    current_retention = retention(candidate.last_access_time, safe_strength)
                    semantic_value = _compute_semantic_value(candidate)
                    if (current_retention < config.mild_pressure_retention
                            and semantic_value >= config.semantic_value_threshold
                            and candidate.creation_time > config.consolidation_window):
                        eligible = True

        elif mode == ConsolidationMode.RETRIEVAL_TRIGGERED:
            eligible = should_consolidate(candidate, config)

        if eligible:
            candidates.append(candidate)

    # Sort by retention ascending (lowest retention first)
    def _retention_key(c: ConsolidationCandidate) -> float:
        safe_strength = max(c.strength, 1e-12)
        return retention(c.last_access_time, safe_strength)

    candidates.sort(key=_retention_key)

    # Limit to max_candidates_per_batch
    return candidates[:config.max_candidates_per_batch]
```

#### Behavioral Contracts

- Contested candidates are NEVER selected, regardless of mode.
- Correction candidates are NEVER selected, regardless of mode.
- In INTRA_SESSION mode, `should_consolidate()` is NOT called (different eligibility criteria).
- In ASYNC_BATCH mode, storage pressure operates at two graduated levels:
  - Mild (pool_size > max_pool_size): relaxed retention threshold, semantic_value gate preserved.
  - Severe (pool_size > max_pool_size * severe_pressure_multiplier): full bypass of retention/semantic_value.
- Result length is at most `config.max_candidates_per_batch`.
- Result is sorted by retention ascending.
- The function is deterministic: same inputs always produce same output.

### 5.7 archive_episodic()

```python
def archive_episodic(
    candidate: ConsolidationCandidate,
    consolidated_to: tuple[int, ...],
    config: ConsolidationConfig | None = None,
) -> ArchivedMemory:
    """Archive an episodic memory after consolidation.

    Marks the memory as archived and applies consolidation_decay to its
    importance and strength. The original content is preserved in full.
    The original scores are recorded for audit trail.

    NOTE: With default consolidation_decay=0.1, archived memories have
    very low scores (e.g., importance 0.7 -> 0.07, strength 7.0 -> 0.7).
    At strength=0.7, retention drops to 0.24 after 1 time unit and 0.014
    after 3 time units. This makes archived memories effectively invisible
    to standard scoring. Retrieving archived memories requires a dedicated
    archive query path (see Section 7.5).

    Args:
        candidate:        The memory to archive.
        consolidated_to:  Tuple of target memory IDs/indices.
        config:           ConsolidationConfig. Uses defaults if None.

    Returns:
        ArchivedMemory with decayed scores.

    Raises:
        TypeError:  If candidate is not a ConsolidationCandidate.
        ValueError: If consolidated_to is empty.
    """
```

#### Algorithm

```
def archive_episodic(candidate, consolidated_to, config):
    config = config or DEFAULT_CONSOLIDATION_CONFIG

    if len(consolidated_to) == 0:
        raise ValueError("consolidated_to must be non-empty")

    decayed_importance = clamp01(candidate.importance * config.consolidation_decay)
    decayed_strength = candidate.strength * config.consolidation_decay

    return ArchivedMemory(
        index=candidate.index,
        content=candidate.content,
        category=candidate.category,
        level=candidate.level,
        archived=True,
        consolidated_to=consolidated_to,
        decayed_importance=decayed_importance,
        decayed_strength=decayed_strength,
        original_importance=candidate.importance,
        original_strength=candidate.strength,
    )
```

#### Behavioral Contracts

- `archived.archived` is always `True`.
- `archived.decayed_importance == clamp01(candidate.importance * config.consolidation_decay)`.
- `archived.decayed_strength == candidate.strength * config.consolidation_decay`.
- `archived.decayed_importance <= archived.original_importance`.
- `archived.decayed_strength <= archived.original_strength`.
- `archived.content == candidate.content` (content is NEVER modified).
- Raises `ValueError` if `consolidated_to` is empty.
- The function is deterministic.

### 5.8 _next_level() (private)

```python
def _next_level(current: ConsolidationLevel) -> ConsolidationLevel | None:
    """Return the next consolidation level, or None if already at the top.

    EPISODIC_RAW        -> EPISODIC_COMPRESSED
    EPISODIC_COMPRESSED -> SEMANTIC_FACT
    SEMANTIC_FACT       -> SEMANTIC_INSIGHT
    SEMANTIC_INSIGHT    -> None (cannot promote further)
    """
```

### 5.9 _truncate_to_sentences() (private)

```python
def _truncate_to_sentences(text: str, target_length: int) -> str:
    """Truncate text to approximately target_length characters, respecting
    sentence boundaries.

    Splits on sentence-ending punctuation (. ! ?) and includes as many
    complete sentences as fit within target_length. If no sentence boundary
    is found, truncates at the nearest word boundary.

    Always returns at least one sentence or the first target_length characters.
    Never returns an empty string (returns text[:1] as absolute minimum).
    """
```

### 5.10 _get_extraction_patterns() (private)

```python
def _get_extraction_patterns(category: str) -> list[re.Pattern[str]]:
    """Return the extraction patterns for a given category.

    Falls back to an empty list for categories without specific patterns
    (triggers the truncation fallback in extract_semantic).
    """
```

### 5.11 _content_is_correction() (private)

```python
def _content_is_correction(content: str) -> bool:
    """Check if content matches correction patterns (defense-in-depth).

    Returns True if content matches at least CORRECTION_MARKER_THRESHOLD
    patterns from CORRECTION_CONTENT_MARKERS. This provides a content-based
    safety check independent of category labeling.

    Args:
        content: The memory content to check.

    Returns:
        True if content is likely a correction, False otherwise.
    """
```

#### Algorithm

```
def _content_is_correction(content):
    match_count = sum(1 for pattern in CORRECTION_CONTENT_MARKERS if pattern.search(content))
    return match_count >= CORRECTION_MARKER_THRESHOLD
```

#### Behavioral Contracts

- Returns `True` if and only if at least `CORRECTION_MARKER_THRESHOLD` distinct patterns match.
- With `CORRECTION_MARKER_THRESHOLD = 2`, requires 2+ matches to trigger.
- Deterministic: same content always produces same result.
- Never raises.

### 5.12 _passes_inhibitors() (private)

```python
def _passes_inhibitors(candidate: ConsolidationCandidate, config: ConsolidationConfig) -> bool:
    """Check if a candidate passes all consolidation inhibitors.

    Shared inhibitor logic used by both should_consolidate() and
    select_consolidation_candidates(). Ensures inhibitors are applied
    consistently across all code paths (AP3-F5).

    Args:
        candidate: The memory to check.
        config: ConsolidationConfig with inhibitor thresholds.

    Returns:
        True if all inhibitors pass (memory can proceed), False if any blocks.
    """
```

#### Algorithm

```
def _passes_inhibitors(candidate, config):
    # I1: Contested memories are never consolidated
    if candidate.contested:
        return False

    # I2: Recently created memories are protected
    if candidate.creation_time < config.recency_guard:
        return False

    # I3: Highly active recent memories are protected
    if (candidate.access_count >= config.high_activation_threshold
            and candidate.last_access_time < config.recency_guard):
        return False

    # I4: Corrections are never consolidated (category-based check)
    if candidate.category in NON_CONSOLIDATABLE_CATEGORIES:
        return False

    # I5: Content-based correction detection (defense-in-depth)
    if _content_is_correction(candidate.content):
        return False

    return True
```

#### Behavioral Contracts

- Returns `False` if and only if at least one inhibitor fires.
- Applies the same 5 inhibitors used in should_consolidate().
- Used by both should_consolidate() and select_consolidation_candidates() to prevent divergent inhibitor checks.
- Deterministic: same inputs always produce same result.
- Never raises for valid inputs.

**Note**: should_consolidate() should call this helper instead of duplicating inhibitor logic. select_consolidation_candidates() should also call this for all modes rather than checking `candidate.contested` and `NON_CONSOLIDATABLE_CATEGORIES` inline.

### 5.13 _extract_by_sentence_scoring() (private)

```python
def _extract_by_sentence_scoring(content: str, target_length: int) -> str:
    """Extract most information-dense sentences to reach target_length.

    Fallback extraction method used when pattern matching fails. Scores each
    sentence by information density (heuristic: longer sentences with diverse
    vocabulary) and includes the highest-scoring sentences until target_length
    is reached.

    Args:
        content: Source text to extract from.
        target_length: Target character count for extracted content.

    Returns:
        Extracted content containing the most information-dense sentences.
        Never empty (returns at least the first sentence or first target_length chars).
    """
```

#### Algorithm

```
def _extract_by_sentence_scoring(content, target_length):
    # 1. Split into sentences
    sentences = re.split(r'[.!?]+\s+', content.strip())
    if not sentences:
        return content[:max(1, target_length)]

    # 2. Score each sentence by information density
    def score_sentence(s):
        words = s.split()
        unique_words = len(set(w.lower() for w in words))
        return len(s) * (unique_words / max(len(words), 1))

    scored = [(score_sentence(s), s) for s in sentences if s.strip()]
    if not scored:
        return content[:max(1, target_length)]

    # 3. Sort by score (descending) and take sentences until target_length reached
    scored.sort(reverse=True, key=lambda x: x[0])
    extracted = []
    total_len = 0
    for _, sentence in scored:
        if total_len + len(sentence) > target_length and extracted:
            break
        extracted.append(sentence)
        total_len += len(sentence)

    # 4. Return in original order (re-sort by position in content)
    if extracted:
        result = ' '.join(extracted)
        return result if result else content[:max(1, target_length)]
    return content[:max(1, target_length)]
```

#### Behavioral Contracts

- Never returns empty string (minimum: first sentence or first target_length chars).
- Preserves sentence boundaries (no mid-sentence truncation unless forced by minimum guarantee).
- Deterministic: same inputs always produce same output.
- Scoring heuristic: longer sentences with higher unique-word ratio score higher.
- Never raises.

---

## 6. Semantic Strengthening

When a SemanticExtraction is created from consolidation, the caller (memory
store or coordinator) should assign the new semantic memory the following
dynamics values:

```python
# New semantic memory scores (caller responsibility, not in consolidation.py)
new_importance = clamp01(max(source_importances) * config.semantic_boost)
new_activation = sum(source_access_counts) * config.activation_inheritance
new_strength = LEVEL_TIME_CONSTANTS[target_level]  # strength encodes time constant
new_relevance = weighted_average(source_relevances)  # if available
```

The consolidation module computes the SemanticExtraction but does NOT create
MemoryState objects. The caller is responsible for constructing the new
MemoryState with appropriate dynamics values. This separation keeps the
consolidation module independent of the memory store interface.

**Rationale for separation**: consolidation.py operates on text and metadata,
producing extractions and archive records. The actual memory store operations
(creating new entries, deactivating old ones) are the caller's responsibility.
This mirrors the pattern in contradiction.py, which produces SupersessionRecords
but does not modify the memory store.

---

## 7. Integration Points

### 7.1 Retrieval-Triggered Consolidation

When `recall()` retrieves an episodic memory, the caller can evaluate it
for consolidation:

```python
# Integration code (not part of consolidation.py)
for recalled_memory in recall_result.memories:
    candidate = ConsolidationCandidate(
        index=recalled_memory.index,
        content=recalled_memory.content,
        category=recalled_memory.metadata.get("encoding_category", "fact"),
        level=recalled_memory.metadata.get("consolidation_level", ConsolidationLevel.EPISODIC_RAW),
        creation_time=recalled_memory.creation_time,
        last_access_time=recalled_memory.last_access_time,
        importance=recalled_memory.importance,
        access_count=recalled_memory.access_count,
        strength=recalled_memory.strength,
        contested=recalled_memory.metadata.get("contested", False),
    )
    result = consolidate_memory(candidate, config)
    if result.consolidated_count > 0:
        # Store new semantic memory, archive old episodic memory
        apply_consolidation_result(result, memory_store)
```

### 7.2 Async Batch Consolidation

Periodic background process that scans the full memory pool:

```python
# Integration code (not part of consolidation.py)
pool = [make_candidate(m) for m in memory_store.list_active_episodic()]
candidates = select_consolidation_candidates(
    pool=pool,
    config=config,
    mode=ConsolidationMode.ASYNC_BATCH,
    pool_size=len(pool),
)

# Cluster candidates by affinity
clusters = _cluster_by_affinity(candidates, config)

for cluster in clusters:
    if len(cluster) == 1:
        result = consolidate_memory(cluster[0], config)
    else:
        result = _consolidate_cluster(cluster, config)
    apply_consolidation_result(result, memory_store)
```

### 7.3 Intra-Session Consolidation

During the write phase, detect related memories within the current session:

```python
# Integration code (not part of consolidation.py)
session_memories = [make_candidate(m) for m in current_session.memories]
candidates = select_consolidation_candidates(
    pool=session_memories,
    config=config,
    mode=ConsolidationMode.INTRA_SESSION,
)

# Find pairs with high affinity
for i, a in enumerate(candidates):
    for b in candidates[i+1:]:
        if compute_affinity(a, b, config) >= config.intra_session_similarity:
            # Merge these two into a single memory
            merged = _merge_intra_session(a, b, config)
            # ... store merged, archive originals
```

### 7.4 Relation to Contradiction Detection

Consolidation respects contradiction state through the `contested` flag on
ConsolidationCandidate. The consolidation module does NOT import from
contradiction.py at runtime. Instead:

1. The caller checks contradiction state when constructing ConsolidationCandidate.
2. If a memory has been flagged as conflicting (via ContradictionResult.flagged_indices),
   the caller sets `contested=True`.
3. Consolidation's inhibitor check blocks contested memories.

This is a clean separation: contradiction writes metadata, consolidation reads
it, neither imports the other.

### 7.5 Relation to Recall Pipeline

Recall (piece 2) does NOT import consolidation.py. The consolidation system
produces side effects (archiving memories, creating new semantic memories)
that recall observes through the memory pool:

1. When `archive_episodic()` runs, the caller sets the original memory's
   `active` flag to False.
2. When the caller creates a new semantic memory from SemanticExtraction,
   it enters the active memory pool.
3. Recall sees fewer episodic memories and more semantic memories, without
   knowing about consolidation.

**Archive retrieval requirement:** Archived memories have decayed scores
(importance ~0.07, strength ~0.7 with defaults) that make them effectively
invisible to standard scoring. For the spec's claim that archived memories
are "retrievable via explicit temporal queries" to hold, the caller MUST
implement a dedicated archive retrieval path that:
- Filters by `archived=True` flag rather than scoring
- Accepts temporal range queries (e.g., "memories from last week")
- Does NOT apply the standard scoring/ranking pipeline
- Returns results ordered by creation_time, not by score

Without this dedicated path, archived memories are de facto deleted despite
being present in the store.

### 7.6 Relation to Encoding

Encoding (piece 1) provides two critical inputs to consolidation:
- `VALID_CATEGORIES`: the category taxonomy that drives compression behavior.
- `CATEGORY_IMPORTANCE`: the importance values that feed into semantic value
  computation.

Consolidation does NOT modify encoding decisions. It consumes category
annotations that encoding assigned at write time.

---

## 8. Category-Aware Compression Rules

Each category has specific compression behavior when consolidating from
episodic to semantic:

| Category | Compression Ratio | Extraction Rule | Example |
|----------|-------------------|-----------------|---------|
| preference | 10:1 | Extract "User prefers X", discard all instance details, conversational context, hedging | "I was talking about text editors and I said I really prefer using Vim because of the modal editing" -> "User prefers: using Vim" |
| fact | 5:1 | Extract "User's Y is Z", preserve the core fact, discard conversation framing | "In our chat about work, I mentioned that I work at Anthropic as a researcher" -> "User fact: works at Anthropic as a researcher" |
| reasoning | 3:1 | Preserve the causal chain, compress surrounding narrative | "Because the database was slow and the cache was cold, therefore the API timed out" -> "User reasoning: database was slow and cache was cold, therefore API timed out" |
| instruction | 5:1 | Extract the directive, discard conversational framing | "I told the assistant that from now on it should always use TypeScript" -> "Standing instruction: always use TypeScript" |
| correction | 1:1 | DO NOT compress. Preserve full conversational context. | (full text preserved) |
| greeting | 10:1 | Heavy compression (should rarely reach consolidation) | "Hello, how are you today" -> "Hello" |
| transactional | 10:1 | Heavy compression (should rarely reach consolidation) | "Run the tests and show me the output" -> "Run tests" |

The extraction patterns (Section 4.3) implement these rules. When no pattern
matches, the fallback truncation (Section 5.9) uses the compression ratio to
determine target length.

---

## 9. Validation Rules

### 9.1 Config Validation

All ConsolidationConfig validation is performed in `__post_init__` and raises
`ValueError` with descriptive messages. The complete set of constraints is
listed in Section 3.3.

### 9.2 Input Validation

Every public function validates its inputs:

| Function | Validation |
|----------|------------|
| `should_consolidate` | candidate type check, category in VALID_CATEGORIES |
| `extract_semantic` | content type check, category in VALID_CATEGORIES, source_episodes non-empty, parallel tuple lengths |
| `consolidate_memory` | candidate type check |
| `compute_affinity` | both arguments type check |
| `select_consolidation_candidates` | pool type check |
| `archive_episodic` | candidate type check, consolidated_to non-empty |

Type checks raise `TypeError`. Domain violations raise `ValueError`.

### 9.3 ConsolidationCandidate Validation

```python
def __post_init__(self) -> None:
    if self.index < 0:
        raise ValueError(f"index must be >= 0, got {self.index}")
    if not self.content:
        raise ValueError("content must be non-empty")
    if self.category not in VALID_CATEGORIES:
        raise ValueError(f"category must be in VALID_CATEGORIES, got {self.category!r}")
    if not (0.0 <= self.importance <= 1.0):
        raise ValueError(f"importance must be in [0, 1], got {self.importance}")
    if self.access_count < 0:
        raise ValueError(f"access_count must be >= 0, got {self.access_count}")
    if self.strength < 0.0:
        raise ValueError(f"strength must be >= 0, got {self.strength}")
    if self.creation_time < 0.0:
        raise ValueError(f"creation_time must be >= 0, got {self.creation_time}")
    if self.last_access_time < 0.0:
        raise ValueError(f"last_access_time must be >= 0, got {self.last_access_time}")
    if self.consolidation_count < 0:
        raise ValueError(f"consolidation_count must be >= 0, got {self.consolidation_count}")
```

---

## 10. Error Handling

### 10.1 Error Categories

| Error | When | Handling |
|-------|------|----------|
| `TypeError` | Wrong argument type passed to any public function | Raised immediately. Caller must fix. |
| `ValueError` | Invalid config values, invalid category, empty required tuple | Raised immediately. Caller must fix. |
| Division by zero in retention | `strength == 0.0` | Guarded: `max(strength, 1e-12)` used everywhere. |
| Division by zero in compression ratio | `len(formatted) == 0` | Guarded: `max(len(formatted), 1)` used. |
| Empty content after extraction | Pattern matched but captured empty group | Guarded: `_truncate_to_sentences` fallback always returns non-empty. |

### 10.2 Logging

The module uses `logging.getLogger(__name__)`. Log levels:

| Level | Events |
|-------|--------|
| DEBUG | Each candidate evaluated, retention computed, trigger/inhibitor status |
| INFO | Each successful consolidation (candidate index, source level, target level) |
| WARNING | Fallback to truncation (no pattern matched), correction attempted to enter pipeline |
| ERROR | None. All errors are raised as exceptions. |

---

## 11. Edge Cases

### 11.1 Empty Pool

```
Input:  pool=[], mode=ASYNC_BATCH
Output: [] (empty list)
```

`select_consolidation_candidates` returns an empty list.

### 11.2 All Contested

```
Input:  pool=[contested_1, contested_2, ...], mode=ASYNC_BATCH
Output: [] (empty list)
```

All candidates are blocked by the contested inhibitor.

### 11.3 Correction Category

```
Input:  candidate.category="correction"
Output: should_consolidate() -> False, always
```

Corrections are NEVER consolidated, regardless of retention, age, or other factors.

### 11.4 Memory at SEMANTIC_INSIGHT Level

```
Input:  candidate.level=SEMANTIC_INSIGHT
Output: consolidate_memory() -> empty result (cannot promote further)
```

There is no level above SEMANTIC_INSIGHT. The memory remains as-is.

### 11.5 Self-Affinity

```
Input:  compute_affinity(a, a, config)
Output: 1.0 (identical content + identical time = perfect affinity)
```

### 11.6 Zero-Strength Memory

```
Input:  candidate.strength=0.0
Behavior: strength is clamped to 1e-12 for retention computation
          retention(t, 1e-12) approaches 0.0 for any t > 0
          This memory will always pass the retention trigger
```

### 11.7 Very Fresh Memory (creation_time=0.0)

```
Input:  candidate.creation_time=0.0
Output: should_consolidate() -> False (blocked by recency guard: 0.0 < 1.0)
```

### 11.8 Very Long Content

```
Input:  content with 100,000+ characters
Output: extract_semantic() processes it normally. Pattern matching may be
        slower but is bounded by the number of patterns (not content length
        for regex). Truncation fallback uses compression ratio to determine
        target length, so output is bounded.
```

### 11.9 Content with No Extractable Pattern

```
Input:  content="asdfghjkl random noise", category="preference"
Output: No PREFERENCE_EXTRACTION_PATTERNS match.
        Fallback: truncate to target_length = len(content) / 10.0
        Template: "User prefers: {truncated_content}"
```

### 11.10 Multiple Sources with Identical Content

```
Input:  Two ConsolidationCandidates with identical content
Output: compute_affinity() returns 1.0 (Jaccard of identical sets is 1.0,
        temporal_sim depends on creation_time difference)
        These will be clustered together in batch mode.
```

### 11.11 Single-Character Content

```
Input:  candidate.content="X"
Output: ConsolidationCandidate validates: len("X") > 0, so it's accepted.
        extract_semantic: no pattern matches, truncation fallback returns "X".
        SemanticExtraction.content = "User prefers: X" (for preference).
        compression_ratio = 1 / 17 = 0.059 (expansion, not compression).
        This is expected for short content -- see Section 4.2 note.
```

### 11.12 Storage Pressure Without Batch Mode

```
Input:  mode=RETRIEVAL_TRIGGERED, pool_size=5000 (above max_pool_size)
Output: Storage pressure is IGNORED in non-batch modes.
        Only standard should_consolidate() criteria apply.
```

### 11.13 Greeting/Transactional Under Normal Conditions

```
Input:  candidate.category="greeting", importance=1.0, access_count=100
Output: _compute_semantic_value() returns 0.0 * 0.6 + 1.0 * 0.3 + 1.0 * 0.1 = 0.40
        0.40 < 0.6 (semantic_value_threshold) -> primary trigger NEVER fires
        These categories can ONLY enter consolidation via storage pressure.
        Under normal conditions, they decay to zero retention without archival.
```

### 11.14 Reasoning Memories at Typical Importance

```
Input:  candidate.category="reasoning", importance=0.5, access_count=5
Output: _compute_semantic_value() returns 0.4 * 0.6 + 0.5 * 0.3 + 0.25 * 0.1 = 0.415
        0.415 < 0.6 (semantic_value_threshold) -> primary trigger does NOT fire
        Reasoning memories need importance >= 0.87 AND access_count >= 15
        to consolidate via primary path. Most reasoning memories decay without
        consolidation unless they receive sustained positive feedback.
```

### 11.15 Fact Memory Cold-Start

```
Input:  candidate.category="fact", importance=0.6 (initial), access_count=0
Output: _compute_semantic_value() returns 0.6 * 0.6 + 0.6 * 0.3 + 0.0 = 0.54
        0.54 < 0.6 (semantic_value_threshold) -> primary trigger does NOT fire
        With S=7.0, retention < 0.3 at t > 8.43
        The fact has decayed past usefulness but semantic_value is too low.
        Result: fact is lost without consolidation.
        Mitigation: dynamics feedback should boost importance of accessed facts.
```

---

## 12. Performance Constraints

- `should_consolidate()`: O(1). Fixed number of comparisons. No loops.
- `extract_semantic()`: O(P * T) where P = number of patterns (~20), T = content
  length. Effectively O(T). Pattern-based regex is bounded by pattern count.
- `compute_affinity()`: O(T) where T = max content length. Tokenization is the
  bottleneck (`.split()`).
- `select_consolidation_candidates()`: O(N * C) where N = pool size,
  C = cost of `should_consolidate()` (O(1)). Sorting is O(N log N).
  Overall: O(N log N).
- `consolidate_memory()`: O(T) dominated by `extract_semantic()`.
- `archive_episodic()`: O(1). Fixed number of arithmetic operations.

No external calls. No embeddings. No LLM. Pure text processing and arithmetic.

Expected wall time:
- Single consolidation: < 5ms for typical memory content (< 10K chars).
- Batch of 100 candidates: < 500ms (dominated by extraction).
- Affinity matrix for 100 candidates: O(N^2 * T) = ~4,950 comparisons.
  For 200-char average content, < 250ms.
- Affinity matrix at max_candidates_per_batch=100 is bounded at 4,950
  pairwise comparisons. This is the hard upper bound per batch.

All public functions are thread-safe. No mutable module-level state. Regex
patterns are compiled as module-level constants (thread-safe in CPython).

---

## 13. Public API Surface

### 13.1 Functions (6 public symbols)

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `should_consolidate` | `(candidate: ConsolidationCandidate, config: ConsolidationConfig \| None) -> bool` | Trigger/inhibitor evaluation |
| `extract_semantic` | `(content: str, category: str, config: ..., source_episodes: ..., source_creation_times: ..., source_importances: ...) -> SemanticExtraction` | Category-aware semantic extraction |
| `consolidate_memory` | `(candidate: ConsolidationCandidate, config: ConsolidationConfig \| None) -> ConsolidationResult` | Full single-memory pipeline |
| `compute_affinity` | `(memory_a: ConsolidationCandidate, memory_b: ConsolidationCandidate, config: ConsolidationConfig \| None) -> float` | Semantic + temporal clustering affinity |
| `select_consolidation_candidates` | `(pool: list[ConsolidationCandidate], config: ..., mode: ..., pool_size: ...) -> list[ConsolidationCandidate]` | Batch candidate selection |
| `archive_episodic` | `(candidate: ConsolidationCandidate, consolidated_to: tuple[int, ...], config: ConsolidationConfig \| None) -> ArchivedMemory` | Archive with score decay |

### 13.2 Classes/Enums (7 public symbols)

| Symbol | Type | Description |
|--------|------|-------------|
| `ConsolidationLevel` | Enum | Four abstraction levels |
| `ConsolidationMode` | Enum | Three consolidation triggers |
| `ConsolidationConfig` | frozen dataclass | Pipeline configuration |
| `ConsolidationCandidate` | frozen dataclass | Input to consolidation evaluation |
| `SemanticExtraction` | frozen dataclass | Extracted knowledge with provenance |
| `ArchivedMemory` | frozen dataclass | Archived source memory |
| `ConsolidationResult` | frozen dataclass | Pipeline output |

### 13.3 Constants (6 public symbols)

| Symbol | Type | Description |
|--------|------|-------------|
| `LEVEL_TIME_CONSTANTS` | dict | Time constant (1/e-life) per consolidation level |
| `LEVEL_HALF_LIVES` | dict | True half-life per level (convenience, computed from time constants) |
| `CATEGORY_COMPRESSION` | dict | Compression ratio per category |
| `SEMANTIC_TEMPLATES` | dict | Output templates per category |
| `NON_CONSOLIDATABLE_CATEGORIES` | frozenset | Categories excluded from consolidation |
| `DEFAULT_CONSOLIDATION_CONFIG` | ConsolidationConfig | Module-level default config |
| `PREFERENCE_EXTRACTION_PATTERNS` | list | (and FACT_, INSTRUCTION_, REASONING_) Pattern lists |

Note: The four `*_EXTRACTION_PATTERNS` constants (PREFERENCE, FACT, INSTRUCTION,
REASONING) are individually exported but counted as one logical unit. Total
individual public constants: 10.

Total public symbols: 6 functions + 7 classes/enums + 10 constants = 23.

---

## 14. Testing Strategy

### 14.1 Unit Tests (per function)

- `should_consolidate`: 25+ tests
  - Each inhibitor independently blocks (4 inhibitor tests)
  - Primary trigger fires when both conditions met
  - Primary trigger does NOT fire when only one condition met
  - Secondary triggers with semantic_value check
  - Boundary conditions: retention exactly at threshold (INCLUDE: <, not <=)
  - Boundary conditions: semantic_value exactly at threshold (INCLUDE: >=)
  - Default config vs custom config
  - Each category's behavior
  - Contested flag always blocks
  - Greeting/transactional NEVER pass primary trigger (verify sv < 0.6 always)
  - Reasoning with typical importance FAILS primary trigger (verify threshold)

- `extract_semantic`: 25+ tests
  - Each PREFERENCE_EXTRACTION_PATTERN matches its target
  - Each FACT_EXTRACTION_PATTERN matches its target
  - Each INSTRUCTION_EXTRACTION_PATTERN matches its target
  - Each REASONING_EXTRACTION_PATTERN matches its target
  - Fallback truncation when no pattern matches
  - Correction category preserves full content
  - Template application per category
  - Provenance fields (confidence, first_observed, last_updated, consolidation_count)
  - Empty group handling
  - Very long content
  - Unicode content
  - Multiple sources with aggregated provenance
  - Short content produces compression_ratio < 1.0 (expansion case)

- `consolidate_memory`: 15+ tests
  - Successful consolidation path
  - Inhibited path (should_consolidate returns False)
  - SEMANTIC_INSIGHT cannot be promoted
  - Single source episode
  - Multiple source episodes via source_episodes field
  - Result invariants (consolidated_count matches archived/extractions)

- `compute_affinity`: 15+ tests
  - Symmetry: affinity(a, b) == affinity(b, a)
  - Self-affinity >= semantic_weight_beta
  - Identical content + identical time = 1.0
  - No overlap + large time difference approaches 0.0
  - Effect of semantic_weight_beta on content vs temporal weighting
  - Edge cases: empty content, single-word content
  - Temporal decay behavior
  - Temporal alone never reaches cluster_threshold (verify max = 0.3)
  - Paraphrased content (same meaning, different words) does NOT cluster

- `select_consolidation_candidates`: 20+ tests
  - Empty pool returns empty list
  - ASYNC_BATCH mode selection
  - INTRA_SESSION mode selection (different criteria)
  - RETRIEVAL_TRIGGERED mode selection
  - Mild storage pressure relaxation (relaxed retention, keeps sv gate)
  - Severe storage pressure bypass (full bypass)
  - Storage pressure does NOT affect non-batch modes
  - Sorting by retention ascending
  - max_candidates_per_batch limit
  - Contested candidates excluded in all modes
  - Corrections excluded in all modes
  - Greeting/transactional excluded in normal mode, included in severe pressure

- `archive_episodic`: 10+ tests
  - Decay calculation correctness
  - Content preservation
  - Empty consolidated_to raises ValueError
  - Original scores preserved for audit
  - archived flag is always True
  - Decay never increases scores
  - Verify archived scores are very low (importance * 0.1)

### 14.2 Integration Tests

- Full pipeline: encode -> dynamics -> consolidate for realistic memory sequences.
- Category-specific compression: verify each category produces expected output format.
- Multi-level consolidation: L1 -> L2 -> L3 chain.
- Batch consolidation with clustering: verify affinity-based grouping.
- Retrieval-triggered consolidation: recall a memory, consolidate it.
- Interaction with contradiction: contested memory survives consolidation.
- Interaction with encoding: category importance feeds into semantic value.
- Graduated storage pressure: verify mild vs severe behavior.
- Fact cold-start scenario: fact at initial importance (0.6) with 0 accesses fails both gates.

### 14.3 Property-Based Tests (Hypothesis)

**Parameter Classification (AP3-F6)**: ConsolidationConfig has 19 parameters,
but only 7 affect provable properties (behavioral invariants). The other 12 are
calibration knobs that tune performance but don't change what is proven.

| Parameter | Classification | Why |
|-----------|---------------|-----|
| recency_guard | **Proof-relevant** | Part of safety inhibitor I2 |
| high_activation_threshold | **Proof-relevant** | Part of safety inhibitor I3 |
| consolidation_decay | **Proof-relevant** | Monotone decay property |
| semantic_boost | **Proof-relevant** | Bounded amplification property |
| activation_inheritance | **Proof-relevant** | Bounded inheritance property |
| semantic_weight_beta | **Proof-relevant** | Affinity function correctness |
| cluster_threshold | **Proof-relevant** | Clustering correctness |
| min_content_similarity | **Proof-relevant** | Affinity floor property (AP3-F2) |
| retention_threshold | Calibration | Tuning parameter, not a provable invariant |
| semantic_value_threshold | Calibration | Tuning parameter |
| consolidation_window | Calibration | Tuning parameter |
| retrieval_frequency_threshold | Calibration | Tuning parameter |
| max_pool_size | Calibration | Operational concern |
| l1_to_l2_retention | Calibration | Level-specific tuning |
| l2_to_l3_retention | Calibration | Level-specific tuning |
| l3_to_l4_cluster_sim | Calibration | Level-specific tuning |
| temporal_decay_lambda | Calibration | Affinity tuning, not behavioral |
| intra_session_similarity | Calibration | Mode-specific tuning |
| intra_session_temporal | Calibration | Mode-specific tuning |
| mild_pressure_retention | Calibration | Storage pressure tuning |
| severe_pressure_multiplier | Calibration | Storage pressure tuning |
| max_candidates_per_batch | Calibration | Operational concern |

Property-based test strategies should focus on the 8 proof-relevant parameters,
as they are the ones that participate in formal verification. The calibration
parameters can be frozen at default values for property tests without losing
verification capability.

```python
@given(...)
def test_should_consolidate_determinism(candidate):
    """Same inputs always produce same output."""
    assert should_consolidate(c, cfg) == should_consolidate(c, cfg)

@given(...)
def test_contested_never_consolidates(candidate):
    """Contested candidates are always rejected."""
    contested = replace(candidate, contested=True)
    assert should_consolidate(contested, cfg) is False

@given(...)
def test_corrections_never_consolidate(candidate):
    """Correction-category candidates are always rejected."""
    correction = replace(candidate, category="correction")
    assert should_consolidate(correction, cfg) is False

@given(...)
def test_affinity_symmetric(a, b):
    """Affinity is symmetric."""
    assert compute_affinity(a, b, cfg) == compute_affinity(b, a, cfg)

@given(...)
def test_affinity_range(a, b):
    """Affinity is always in [0, 1]."""
    val = compute_affinity(a, b, cfg)
    assert 0.0 <= val <= 1.0

@given(...)
def test_extract_semantic_non_empty(content, category):
    """Extraction always produces non-empty content."""
    result = extract_semantic(content, category, cfg, ("ep1",), (0.0,), (0.5,))
    assert len(result.content) > 0

@given(...)
def test_archive_decay_monotone(candidate):
    """Archived scores are never higher than originals."""
    archived = archive_episodic(candidate, (0,), cfg)
    assert archived.decayed_importance <= archived.original_importance
    assert archived.decayed_strength <= archived.original_strength

@given(...)
def test_consolidation_result_invariants(candidate):
    """Result invariants hold for any candidate."""
    result = consolidate_memory(candidate, cfg)
    assert result.consolidated_count == len(result.archived)
    assert len(result.extractions) <= len(result.archived)
    if result.consolidated_count > 0:
        assert len(result.extractions) >= 1

@given(...)
def test_semantic_value_range(candidate):
    """Semantic value is always in [0, 1]."""
    val = _compute_semantic_value(candidate)
    assert 0.0 <= val <= 1.0

@given(...)
def test_greeting_never_consolidates_normally(candidate):
    """Greeting-category candidates never pass should_consolidate."""
    greeting = replace(candidate, category="greeting")
    assert should_consolidate(greeting, cfg) is False

@given(...)
def test_temporal_alone_insufficient(a, b):
    """Temporal similarity alone never exceeds cluster_threshold."""
    # Set content to non-overlapping
    a_mod = replace(a, content="alpha bravo charlie")
    b_mod = replace(b, content="delta echo foxtrot")
    aff = compute_affinity(a_mod, b_mod, cfg)
    assert aff < cfg.cluster_threshold

@given(...)
def test_content_similarity_floor(a, b, cfg):
    """Affinity is zero when content_sim < min_content_similarity (AP3-F2)."""
    # Generate memories with low content similarity
    a_mod = replace(a, content="alpha bravo charlie")
    b_mod = replace(b, content="delta echo foxtrot golf hotel")
    aff = compute_affinity(a_mod, b_mod, cfg)
    # Jaccard should be 0 (no overlap), so affinity should be 0
    # regardless of temporal proximity
    assert aff == 0.0

@given(...)
def test_correction_content_detection(candidate):
    """Content-based correction detection blocks consolidation (AP3-F3)."""
    # Add multiple correction markers to content
    correction_content = "Actually, I was wrong about that. Let me correct: not X but Y."
    corrective = replace(candidate, content=correction_content, category="fact")
    # Even though category is "fact", content triggers I5 inhibitor
    assert should_consolidate(corrective, cfg) is False

@given(...)
def test_passes_inhibitors_consistency(candidate, cfg):
    """_passes_inhibitors returns same result as should_consolidate inhibitors (AP3-F5)."""
    # should_consolidate should use _passes_inhibitors
    inhibitor_result = _passes_inhibitors(candidate, cfg)
    consolidate_result = should_consolidate(candidate, cfg)
    # If inhibitors block, should_consolidate must also block
    if not inhibitor_result:
        assert consolidate_result is False

@given(...)
def test_semantic_extraction_non_empty_source_episodes(content, category):
    """extract_semantic requires non-empty source_episodes (AP3-F7)."""
    with pytest.raises(ValueError):
        extract_semantic(content, category, cfg, source_episodes=(),
                        source_creation_times=(), source_importances=())

@given(...)
def test_provenance_chain_integrity(candidate, cfg):
    """Archived memories must have valid consolidated_to targets (AP3-F7)."""
    result = consolidate_memory(candidate, cfg)
    for archived in result.archived:
        assert len(archived.consolidated_to) >= 1
        # In practice, caller must ensure these point to live memories
        # This test verifies the structural invariant
```

### 14.4 Coverage Target

80% line coverage for the consolidation module. Property-based tests with
Hypothesis will exercise edge cases that unit tests miss.

Estimated total: 130-160 tests.

---

## 15. Future Extensions (out of scope for v1)

- **Embedding-based affinity**: Replace Jaccard token similarity with
  embedding cosine similarity for higher-quality clustering. Requires the
  embedder infrastructure (Phase 1 of architecture.md). **This is critical
  for unlocking L3->L4 consolidation** -- see Section 16.
- **LLM-assisted extraction**: Use an auxiliary LLM for semantic extraction
  instead of pattern matching. Would improve extraction quality for complex
  narratives but adds external dependency.
- **Incremental consolidation**: Instead of full batch scans, maintain a
  priority queue of consolidation candidates sorted by urgency.
- **User-configurable consolidation**: Allow users to mark memories as
  "never consolidate" or "always keep full detail".
- **Cross-session temporal awareness**: Use wall-clock timestamps (not just
  logical time steps) for more accurate temporal decay.
- **Consolidation analytics**: Track consolidation rates, compression ratios,
  and semantic value preservation over time.
- **Category-adjusted semantic thresholds**: Per-category semantic_value
  thresholds (e.g., lower threshold for reasoning) to improve consolidation
  coverage for knowledge-bearing categories.

---

## 16. Known Limitations (v1)

This section documents limitations that are inherent to the v1 design choices
and will be addressed in future versions.

### 16.1 Token Jaccard Similarity

The affinity function uses token-level Jaccard similarity, which has a
fundamental quality ceiling:

| Scenario | Typical Jaccard | Clustered at 0.6? |
|----------|-----------------|-------------------|
| Exact duplicate | 1.000 | YES |
| Near-duplicate (1-2 words different) | 0.70-0.85 | YES |
| Same meaning, different surface form | 0.15-0.35 | NO |
| Same topic, different facts | 0.10-0.30 | NO |
| Unrelated content | 0.00-0.10 | NO |

Implications:
- Clustering only catches near-exact duplicates and substring matches
- Semantically equivalent memories with different wording are never merged
- The system relies on upstream deduplication (encoding gate) to prevent
  semantic duplicates rather than merging them during consolidation

### 16.2 L3->L4 Transition Effectively Gated on v2

The L3->L4 promotion requires cluster_sim > 0.6. But at L3, memories have
already been through extraction and templating (e.g., "User prefers: using Vim"
vs "User prefers: using Neovim for editing"). Even related facts from the
same template typically achieve Jaccard ~0.3-0.5.

**L3->L4 consolidation should not be expected to fire in v1.** The
SEMANTIC_INSIGHT level exists as a placeholder for the embedding-based
affinity system (v2). The time constant for SEMANTIC_INSIGHT (365 days)
ensures that L3 memories have very long lifetimes in the meantime.

**AP3-F4 Note**: The L3->L4 transition requires semantic reasoning beyond
token-level Jaccard similarity. Consolidating semantic facts into insights
("User prefers Vim-like editors" from multiple "User prefers: Vim", "User
prefers: Neovim" memories) requires understanding conceptual relationships,
which is v2 scope (embedding-based clustering or LLM-assisted consolidation).
The 4-level hierarchy models the cognitive science concept of hippocampal-to-cortical
consolidation, but the proof module's v1 implementation can only verify the
episodic-to-semantic transformation (L1/L2 -> L3). The L3->L4 pathway is a
design placeholder, not a v1 operational feature.

### 16.3 Greeting/Transactional Pathway

Greeting and transactional memories can never consolidate via the primary
trigger path (max sv = 0.40 < 0.60 threshold). They are only consolidated
under severe storage pressure. Under normal operation, they simply decay to
zero retention and become irretrievable -- neither archived nor consolidated.

This is acceptable because:
1. The encoding gate (piece 1) should filter most greeting/transactional content
2. Any that leak through carry no semantic value worth preserving
3. Under storage pressure, they are cleaned up to free pool space

### 16.4 Time Unit Coupling

All time-based thresholds (consolidation_window, recency_guard,
intra_session_temporal) assume `delta_t = 1.0` in the dynamics engine.
If the dynamics system uses a different time step, callers must scale
consolidation thresholds proportionally. There is no automatic coupling
between ParameterSet.delta_t and ConsolidationConfig thresholds.

---

## Appendix A: Adversarial Review Amendments (Pass 3)

This appendix documents amendments applied from the Pass 3 adversarial review
(Exclusion-Test + Object-Transpose operators), conducted 2026-02-27. Pass 3
identified 3 lethal design flaws and 4 structural issues that Passes 1-2 missed.

### AP3-F1: LETHAL -- extract_semantic() Information Destruction on Pattern Miss

**Finding**: The original fallback used `_truncate_to_sentences()`, which blindly
truncated content to `len(content) / compression_ratio` characters. For memories
that didn't match any extraction pattern (common for narrative-style encoding),
this destroyed semantic value -- the exact thing consolidation exists to preserve.

**Example**: A 500-character preference memory with compression ratio 10.0 would
be truncated to 50 characters, regardless of where the meaningful content lived
in the text.

**Amendment**:
- Replaced blind truncation with `_extract_by_sentence_scoring()` (Section 5.13)
- Scores sentences by information density (length × unique-word ratio)
- Preserves the most information-dense sentences to reach target length
- Updated `extract_semantic()` algorithm (Section 5.3) to call sentence-scoring fallback
- Updated behavioral contract to specify sentence-scoring fallback behavior

**Impact**: Prevents net-negative consolidation where extracted content loses
semantic value. The pattern miss rate for narrative-encoded memories was estimated
at 30-50%; sentence scoring provides a semantic-preserving fallback for these cases.

### AP3-F2: LETHAL -- Affinity Function Clusters Unrelated Memories via Temporal Proximity

**Finding**: The original affinity function allowed temporal similarity to dominate
for memories created at nearly the same time, even with zero semantic overlap.
Two completely unrelated memories (e.g., "prefers dark mode" and "dog named Charlie")
created 1ms apart could achieve affinity 0.3 from temporal_sim alone. With even
modest word overlap from common English words ("the", "user", "at"), affinity
could approach cluster_threshold.

**Example**: "User works at Google" and "User works out at gym" share 5/17 tokens
despite being unrelated facts. Created simultaneously, they could reach affinity 0.506.

**Amendment**:
- Added `min_content_similarity` parameter to ConsolidationConfig (Section 3.3), default 0.3
- Added content similarity floor check in `compute_affinity()` (Section 5.5)
- If `content_sim < min_content_similarity`, affinity is forced to 0.0 regardless of temporal proximity
- Updated behavioral contracts to document floor property
- Added validation rule: `0.0 <= min_content_similarity <= 1.0`

**Impact**: Prevents clustering of semantically unrelated memories based on
creation time alone. With default min_content_similarity=0.3, temporal proximity
can never push affinity above threshold without substantial content overlap.

### AP3-F3: LETHAL -- Inhibitor Bypass via Category Miscategorization

**Finding**: The original inhibitor I4 checked `candidate.category in NON_CONSOLIDATABLE_CATEGORIES`
to block corrections. But if encoding.py miscategorized a correction as "fact"
(possible due to heuristic pattern matching), the inhibitor would not fire. The
correction would be compressed at 5:1 ratio, losing corrective context.

**Example**: "Actually, my name is spelled Cosimo, not Cosmo" could match both
CORRECTION_PATTERNS and FACT_PATTERNS. If encoding chose "fact", the consolidation
inhibitor would fail.

**Amendment**:
- Added `CORRECTION_CONTENT_MARKERS` constant (Section 4.7): frozenset of regex patterns
- Added `CORRECTION_MARKER_THRESHOLD` constant (Section 4.7): default 2
- Added `_content_is_correction()` private helper (Section 5.11)
- Added inhibitor I5 to `should_consolidate()`: content-based correction detection
- If content matches ≥ CORRECTION_MARKER_THRESHOLD patterns, block consolidation
  regardless of category label
- Updated behavioral contracts to include I5

**Impact**: Defense-in-depth against miscategorization. Even if encoding.py labels
a correction as "fact", the content-based check will catch it and prevent compression.

### AP3-F4: Object-Transpose -- 4-Level Hierarchy Unjustified for Proof Module

**Finding**: The 4-level hierarchy (EPISODIC_RAW → EPISODIC_COMPRESSED → SEMANTIC_FACT →
SEMANTIC_INSIGHT) models cognitive science concepts, but the proof module's actual
verification task is simpler: verify the episodic-to-semantic transformation.
The L3→L4 transition requires semantic reasoning (conceptual clustering) beyond
token-level Jaccard, which is v2 scope.

**Amendment**:
- Added note to Section 16.2 clarifying that L3→L4 is a v2 placeholder
- Documented that the proof module in v1 can only verify L1/L2 → L3 transitions
- Noted that semantic reasoning (understanding "Vim" and "Neovim" are related editors)
  requires embeddings or LLM assistance, not implemented in v1
- No structural changes to enum (kept 4 levels for forward compatibility)

**Impact**: Sets correct expectations: v1 verifies episodic-to-semantic consolidation
only. SEMANTIC_INSIGHT exists as design documentation, not operational feature.

### AP3-F5: Object-Transpose -- 3-Mode System Duplicates Inhibitor Logic

**Finding**: The three consolidation modes (INTRA_SESSION, ASYNC_BATCH, RETRIEVAL_TRIGGERED)
all needed to check inhibitors, but the original spec duplicated these checks inline
in different code paths. This created risk of divergent inhibitor implementations.

**Amendment**:
- Extracted `_passes_inhibitors()` shared helper (Section 5.12)
- Applies all 5 inhibitors (I1-I5) in one place
- Updated `should_consolidate()` (Section 5.1) to delegate to `_passes_inhibitors()`
- Updated `select_consolidation_candidates()` (Section 5.6) to use `_passes_inhibitors()`
  for all modes instead of inline `candidate.contested` / `NON_CONSOLIDATABLE_CATEGORIES` checks
- Added note in behavioral contracts that this ensures consistency

**Impact**: Single source of truth for inhibitor logic. Any future inhibitor
changes (e.g., adding I6) only require updating one function. Prevents subtle
bugs where one code path checks I1-I4 and another checks I1-I5.

### AP3-F6: Object-Transpose -- ConsolidationConfig Over-Parameterized for Verification

**Finding**: ConsolidationConfig has 19 parameters, but only 7 affect provable
properties (behavioral invariants). The other 12 are calibration knobs. Property-based
tests should focus on proof-relevant parameters.

**Amendment**:
- Added parameter classification table to Section 14.3
- Documented which 8 parameters are proof-relevant (recency_guard, high_activation_threshold,
  consolidation_decay, semantic_boost, activation_inheritance, semantic_weight_beta,
  cluster_threshold, min_content_similarity)
- Noted that calibration parameters can be frozen at defaults for property tests
  without losing verification capability

**Impact**: Guides test strategy. Hypothesis generators should vary the 8 proof-relevant
parameters to exercise behavioral invariants, while leaving the 12 calibration
parameters at defaults to reduce search space.

### AP3-F7: Exclusion-Test -- Provenance Chain Breaks on Repeated Consolidation

**Finding**: If an archived memory's `consolidated_to` target is itself later
consolidated (e.g., L2 semantic memory is promoted to L3), the original archived
memory now points to another archived memory. The promise that "archived memories
remain retrievable" is broken without chain-following logic.

**Amendment**:
- Added provenance chain integrity note to Section 3.6 (ArchivedMemory invariants)
- Documented that the caller (memory store) must update `consolidated_to` references
  when a consolidation target is itself consolidated
- Added note that a property-based test should verify chain integrity (traverse
  `consolidated_to` to ensure it terminates at a live memory)
- Added test skeleton in Section 14.3 for provenance chain validation
- extract_semantic already required non-empty source_episodes (line 948 ValueError),
  so no change needed there

**Impact**: Makes explicit that chain integrity is caller's responsibility. The
consolidation module produces SemanticExtraction and ArchivedMemory records; the
memory store must maintain referential integrity when performing repeated consolidation.

### Summary Table

| Finding | Severity | Change Type | Files Affected |
|---------|----------|-------------|----------------|
| AP3-F1 | LETHAL | Algorithm | Section 5.3, 5.13 (new helper) |
| AP3-F2 | LETHAL | Config + Algorithm | Sections 3.3, 5.5 |
| AP3-F3 | LETHAL | Constants + Inhibitor | Sections 4.7, 5.1, 5.11 (new helper) |
| AP3-F4 | STRUCTURAL | Documentation | Section 16.2 |
| AP3-F5 | STRUCTURAL | Refactoring | Sections 5.1, 5.6, 5.12 (new helper) |
| AP3-F6 | STRUCTURAL | Documentation | Section 14.3 |
| AP3-F7 | LETHAL | Invariant | Sections 3.6, 14.3 |

**Review Operators Used**:
- Exclusion-Test: Identified AP3-F1 (pattern miss), AP3-F2 (zero content overlap),
  AP3-F3 (miscategorization), AP3-F7 (chain breakage)
- Object-Transpose: Identified AP3-F4 (hierarchy), AP3-F5 (mode duplication),
  AP3-F6 (parameter count)

**Pass 3 Reviewer**: architect-agent (Opus 4.6)
**Date**: 2026-02-27
**Status**: All amendments applied to spec v3.0

---

**End of spec.**
