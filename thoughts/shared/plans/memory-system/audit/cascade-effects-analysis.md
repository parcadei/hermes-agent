# Cascade Effects Analysis: Confirmed Audit Findings

Created: 2026-02-28
Author: architect-agent

## Methodology

For each confirmed finding, I traced second-order effects (what breaks or
behaves differently because of this finding) and third-order effects (what
breaks because of the second-order effect). Analysis is grounded in verified
file:line references from the actual implementation.

## Critical Context: Module Isolation

**Key architectural finding**: The consolidation module's types
(`ConsolidationCandidate`, `SemanticExtraction`, `ArchivedMemory`,
`ConsolidationResult`, `ConsolidationLevel`) are **not imported by any
other production module**. The only consumers are test files
(`test_consolidation.py`, `test_consolidation_phase_c.py`).

The import graph is:
```
consolidation.py  -->  core.py (retention, clamp01)
consolidation.py  -->  encoding.py (CATEGORY_IMPORTANCE, VALID_CATEGORIES)
optimizer.py      -->  engine.py (ParameterSet, MemoryState, rank_memories, simulate)
optimizer.py      -->  sensitivity.py (Scenario)
recall.py         -->  engine.py (MemoryState, ParameterSet) [TYPE_CHECKING only]
encoding.py       -->  (stdlib only)
contradiction.py  -->  encoding.py (VALID_CATEGORIES)
```

No module imports from `consolidation.py`. No module imports from `optimizer.py`
(except test_recall.py importing PINNED for test helpers). This means many
findings that appear to have cross-module impact are actually **isolated within
the consolidation boundary** in the current codebase.

However, this isolation is an artifact of the current proof-of-concept state.
In a production system, an orchestrator would consume `ConsolidationResult`,
`SemanticExtraction`, and `ArchivedMemory` to mutate the memory store. The
cascade analysis below considers BOTH the current isolation AND the intended
production integration described in the spec.

---

## Consolidation Findings

### M1: ConsolidationLevel IntEnum(1,2,3,4) vs Enum("episodic_raw",...)

**Implementation** (consolidation.py:35-49):
```python
class ConsolidationLevel(enum.IntEnum):
    EPISODIC_RAW = 1
    EPISODIC_COMPRESSED = 2
    SEMANTIC_FACT = 3
    SEMANTIC_INSIGHT = 4
```

**Spec** (spec.md:182-193):
```python
class ConsolidationLevel(enum.Enum):
    EPISODIC_RAW = "episodic_raw"
    EPISODIC_COMPRESSED = "episodic_compressed"
    SEMANTIC_FACT = "semantic_fact"
    SEMANTIC_INSIGHT = "semantic_insight"
```

#### Second-order effects

1. **Comparison semantics differ**: IntEnum(1) allows `level < 3` arithmetic
   comparisons, which work correctly for level ordering. String Enum would
   require explicit ordering logic. The implementation actually comments
   "IntEnum to support comparison operators (< > <= >=)" -- this is a
   *deliberate* deviation that adds functionality. However:

2. **Serialization/deserialization breaks**: If any persistence layer stores
   `ConsolidationLevel.EPISODIC_RAW`, IntEnum serializes as `1` while
   string Enum serializes as `"episodic_raw"`. A JSON store expecting
   `"episodic_raw"` would fail to deserialize `1`, and vice versa.
   Currently no serialization exists in the codebase, but the spec's
   `ArchivedMemory` dataclass implies persistence.

3. **Database schema mismatch**: If a memory store uses the level enum values
   as column values or discriminators, integer `1` vs string `"episodic_raw"`
   causes type mismatch. Any SQL query like
   `WHERE level = 'episodic_raw'` would return empty results.

4. **Truthiness edge case**: `ConsolidationLevel.EPISODIC_RAW` as IntEnum(1)
   is truthy, but IntEnum(0) would be falsy. If levels were 0-indexed, the
   base level would pass `if not level:` checks. Not currently an issue since
   values start at 1, but fragile.

#### Third-order effects

1. **Data migration required**: If the system ships with IntEnum persistence
   and later aligns to the spec's string values, all stored level values need
   migration (`1 -> "episodic_raw"`, etc.). During migration, any concurrent
   reads would fail deserialization.

2. **Cross-system interop failure**: If the Lean formalization or any other
   consumer of consolidation output expects string values, they receive
   integers, causing type errors or silent misclassification in downstream
   proofs.

#### Severity amplification
**AMPLIFIED** when persistence is added. Currently neutral (comparison operators
are useful). Becomes critical at integration time.

---

### C1: Config field naming mismatches (12+ fields renamed vs spec)

**Key mismatches** (consolidation.py:288-334 vs spec.md:314-350):

| Spec name | Implementation name | Notes |
|-----------|-------------------|-------|
| `retention_threshold` | `retention_threshold_l1` | Split into 3 per-level fields |
| `l1_to_l2_retention` | `retention_threshold_l1` | Same default, renamed |
| `l2_to_l3_retention` | `retention_threshold_l2` | Same default, renamed |
| `l3_to_l4_cluster_sim` | `retention_threshold_l3` | **Changed semantics** |
| `semantic_weight_beta` | `temporal_weight` | **Inverted semantics** |
| `temporal_decay_lambda` | `temporal_decay_rate` | Renamed only |
| `cluster_threshold` | `cluster_similarity_threshold` | Renamed only |
| `intra_session_similarity` | `intra_session_similarity_threshold` | Renamed only |
| `intra_session_temporal` | `intra_session_temporal_proximity` | Renamed only |
| `consolidation_window` | `consolidation_window` | **Time unit changed** (see C2) |
| `semantic_boost` | MISSING | Not in impl |
| `activation_inheritance` | MISSING | Not in impl |

#### Second-order effects

1. **Configuration injection fails**: Any external config loader (YAML, env vars,
   JSON) using spec field names would fail to construct `ConsolidationConfig`.
   Fields like `l1_to_l2_retention=0.4` would raise `TypeError: unexpected
   keyword argument 'l1_to_l2_retention'`.

2. **Documentation/config drift**: Users reading the spec would set
   `semantic_weight_beta=0.7` but the implementation expects `temporal_weight=0.3`
   (which is `1 - 0.7`). This is an inverted semantic, not just a rename -- see
   M4 for analysis.

3. **L3->L4 semantics change (see H6)**: The spec's `l3_to_l4_cluster_sim`
   is used for cluster-based similarity promotion. The implementation
   `retention_threshold_l3` is used as a retention decay threshold. These are
   fundamentally different criteria. The spec says L3->L4 should use cluster
   similarity; the implementation treats it as another retention check.

#### Third-order effects

1. **Incorrect L3->L4 promotion**: Memories at SEMANTIC_FACT level would be
   promoted to SEMANTIC_INSIGHT based on retention decay (wrong criterion)
   instead of cluster similarity (spec criterion). This means isolated semantic
   facts with low retention get promoted to "insight" level even when they have
   no cluster support, producing fake insights.

2. **Parameter tuning futile**: If an operator tunes `l3_to_l4_cluster_sim` per
   the spec documentation, nothing changes because the implementation uses a
   differently-named field with different semantics.

#### Severity amplification
**AMPLIFIED**. The L3->L4 semantics change (H6 + C1 combination) is the most
impactful cross-finding amplification in the consolidation module.

---

### C2: Time units hours vs spec's days

**Implementation** (consolidation.py:243-246):
```
min_temporal_distance: float = 24.0  # hours
consolidation_window: float = 168.0  # hours (7*24)
```

**Spec** (spec.md:217-222, 237-239):
```
consolidation_window: float = 7.0    # days (delta_t=1.0 = 1 day)
```

The spec explicitly states: "1 time step represents approximately 1 day"
and "consolidation_window: Default: 7.0. Assumes delta_t = 1.0 (1 step ~= 1
day)."

#### Second-order effects

1. **Integration mismatch with engine.py**: The engine uses `delta_t=1.0`
   (pinned in optimizer.py:57) where 1 time step = 1 abstract unit. If the
   engine's `MemoryState.creation_time` counts in days (as implied by
   `delta_t=1.0`), then a consolidation_window of 168.0 means a memory must
   be 168 days old before batch consolidation, not 7 days. This is a **24x
   over-retention** of episodic memories.

2. **Recency guard mismatch**: `recency_guard: float = 1.0` -- if the engine
   counts time in days, 1.0 means 1 day. The spec also says 1 day. But the
   docstring says "hours" at line 243. If someone adjusts the engine's delta_t
   or time units based on the consolidation docstring, they'll set delta_t to
   represent hours, breaking the engine's calibrated dynamics.

3. **Storage pressure delayed**: With consolidation_window = 168 vs 7, batch
   consolidation's secondary trigger (`creation_time > consolidation_window`)
   requires memories 24x older before they're eligible. This delays the
   entire aging pipeline.

#### Third-order effects

1. **Memory pool bloat**: Episodic memories that should consolidate after
   7 time steps wait 168 time steps. During this period, the pool accumulates
   ~24x more unconsolidated memories than the spec intends. The pool hits
   `max_pool_size=5000` much sooner, triggering storage pressure bypass
   (which has its own quality issues -- see below).

2. **Quality degradation under pressure**: When storage pressure triggers
   prematurely due to pool bloat, the graduated pressure system (severe
   pressure at `pool_size > max_pool_size * severe_pressure_multiplier`)
   kicks in earlier. Under severe pressure, memories are consolidated with
   only the `creation_time > consolidation_window` check (line 1181),
   bypassing semantic value filtering. This means low-value memories get
   consolidated into semantic memories that shouldn't exist.

3. **Retention curve disconnect**: The consolidation module checks
   `retention(last_access_time, strength)` where `last_access_time` uses the
   engine's time units (days). But the consolidation_window comparison
   (`candidate.creation_time > config.consolidation_window`) compares the
   engine's day-based creation_time against 168.0, not 7.0. This means the
   "old but infrequently accessed" secondary trigger never fires for the
   first 168 days, while retention decay (using correct time constants) would
   have already driven retention to near-zero well before then.

#### Severity amplification
**CRITICALLY AMPLIFIED**. This is the highest-severity cascade in the entire
analysis. The 24x over-retention cascades into pool bloat, which cascades into
premature storage pressure, which cascades into low-quality consolidation
output. Every downstream consumer of `SemanticExtraction` gets worse results.

---

### C4: ConsolidationCandidate field/type mismatches

**Spec** (spec.md:415-426):
```python
index: int                           # position in pool
contested: bool                      # spec field name
source_episodes: tuple[str, ...] = ()  # provenance tracking
consolidation_count: int = 0         # re-consolidation counter
```

**Implementation** (consolidation.py:510-521):
```python
memory_id: str                       # string identifier, not integer index
is_contested: bool = False           # renamed, has default
# source_episodes: MISSING
# consolidation_count: MISSING
relevance: float                     # ADDED (not in spec)
metadata: dict | None                # ADDED (not in spec)
```

#### Second-order effects

1. **Index vs memory_id type mismatch**: The spec uses `index: int` for
   back-referencing into the memory pool. The implementation uses
   `memory_id: str`. Any orchestrator code that expects to use the candidate's
   identifier as a list index (`pool[candidate.index]`) would get a TypeError
   when receiving a string. The `ArchivedMemory` also uses `memory_id: str`
   (line 606), so within the consolidation boundary this is consistent. But
   at the interface boundary with the memory store, the contract is broken.

2. **Missing source_episodes on candidate**: The spec tracks
   `source_episodes` on the candidate so that when L2 memories are further
   consolidated to L3, the provenance chain is preserved. Without this field,
   the orchestrator cannot know which original episodes contributed to a
   candidate that is itself a consolidated memory.

3. **Missing consolidation_count**: Without this counter, the system cannot
   distinguish between first-time consolidation and re-consolidation. The spec
   uses this to prevent over-consolidation: a memory consolidated 3+ times
   should have different treatment than a fresh L1 memory. Without it, every
   candidate is treated as a first-time consolidation.

#### Third-order effects

1. **Provenance chain breaks**: When L2 memory M2 (consolidated from L1
   episodes E1, E2, E3) is further consolidated to L3, the resulting
   `SemanticExtraction` should carry `source_episodes=(E1, E2, E3)`. Instead,
   it carries `source_episodes=(M2,)` -- only the intermediate memory ID.
   The original episodic provenance is lost.

2. **Audit trail incomplete**: If a user asks "why does the system believe X?"
   the answer should trace back to the original episodes. Without the
   provenance chain, the answer is "because semantic memory M3 says so" --
   which was consolidated from M2, which was consolidated from... lost.

3. **Re-consolidation safety compromised**: Without `consolidation_count`,
   a heavily-consolidated memory (L3, consolidated 5 times) gets the same
   compression treatment as a fresh episode. Each consolidation pass removes
   more detail. After 3-4 passes without tracking, the content may be
   compressed to near-meaninglessness.

#### Severity amplification
**AMPLIFIED in production**. Currently isolated (no orchestrator exists), but
the moment an orchestrator is built, both missing fields create architectural
debt that is expensive to backfill.

---

### C5: SemanticExtraction missing fields

**Spec** (spec.md:474-482):
```python
first_observed: float
last_updated: float
consolidation_count: int
compression_ratio: float
```

**Implementation** (consolidation.py:553-564):
```python
temporal_range: tuple[float, float]  # replaces first_observed/last_updated
# consolidation_count: MISSING
# compression_ratio: MISSING
target_level: ConsolidationLevel     # ADDED (not in spec)
importance: float                    # ADDED (not in spec)
extraction_mode: ConsolidationMode   # ADDED (not in spec)
```

#### Second-order effects

1. **temporal_range vs first_observed/last_updated**: A consumer expecting
   `extraction.first_observed` gets `AttributeError`. The information is
   available as `extraction.temporal_range[0]` and
   `extraction.temporal_range[1]`, but the interface contract is different.
   Any code written against the spec must be adapted.

2. **Missing compression_ratio**: The spec defines this as a quality metric
   (`source_length / extracted_length`). Without it, there is no way to
   validate that extraction is actually compressing content appropriately.
   The `extract_semantic` function (line 967-968) computes
   `target_len = max(1, int(len(content) / compression_ratio))` internally
   but does not record the achieved ratio on the output.

3. **Missing consolidation_count**: Same as C4 -- no re-consolidation tracking.

4. **Added fields beneficial**: `target_level`, `importance`, and
   `extraction_mode` are useful additions not in the spec. They carry
   information that the spec leaves to the orchestrator to infer.

#### Third-order effects

1. **Quality monitoring impossible**: Without `compression_ratio` on the
   extraction, there is no way to build monitoring dashboards or alerts for
   over-compression or under-compression. The operator cannot tell if the
   H4 compression ratio bug is causing reasoning memories to lose 86%
   of their content (ratio 7) instead of the intended 67% (ratio 3).

2. **Semantic memory dating ambiguous**: A consumer that needs to answer
   "when was this fact first observed?" must know to access `temporal_range[0]`
   instead of `first_observed`. If the consumer is another ML system, the
   field name mismatch causes silent feature extraction failures.

#### Severity amplification
**MODERATE**. The tuple-vs-named-fields issue is a refactoring inconvenience.
The missing `compression_ratio` has quality monitoring implications but no
runtime failure. The missing `consolidation_count` is the same issue as C4.

---

### H1: ArchivedMemory field mismatches

**Spec** (spec.md:530-539):
```python
index: int
archived: bool  # always True
consolidated_to: tuple[int, ...]
decayed_strength: float
original_strength: float
```

**Implementation** (consolidation.py:606-615):
```python
memory_id: str                          # str not int
# archived: MISSING                     # structural flag absent
successor_id: str | None = None         # single successor, not tuple
# decayed_strength: MISSING
original_level: ConsolidationLevel      # ADDED
access_count_at_archive: int = 0        # ADDED
```

#### Second-order effects

1. **No archived flag**: The spec's `archived: bool` is always True. The
   implementation omits it, meaning there is no intrinsic way to distinguish
   an `ArchivedMemory` from other dataclasses. A memory store that receives
   an `ArchivedMemory` must use isinstance checking rather than a field-based
   discriminator. This matters for stores that serialize all memories into the
   same table -- without the flag, the store cannot query "all archived
   memories" without type-based filtering.

2. **Single successor vs tuple**: The spec uses `consolidated_to: tuple[int, ...]`
   to support N:1 consolidation (multiple source memories consolidated into
   one semantic memory -- each source points to the same target). The
   implementation uses `successor_id: str | None = None`, which only allows
   one link and uses str instead of int. When a memory is consolidated into
   multiple targets (unlikely in v1, possible in v2 with clustering), only
   one link is preserved.

3. **Missing decayed_strength**: The spec decays both importance AND strength.
   The implementation only decays importance (consolidation.py:1023). Without
   decayed strength, the archived memory retains its original strength in the
   scoring function. If archived memories are not properly excluded from
   recall, they score higher than intended because only importance is decayed
   (0.7 -> 0.07) but strength remains at its original value (e.g., 7.0).

#### Third-order effects

1. **Archived memory recall leakage**: If archived memories leak into the
   recall pipeline (e.g., the memory store doesn't properly exclude them),
   their un-decayed strength means `retention(last_access_time, strength=7.0)`
   gives a much higher recency score than `retention(last_access_time,
   strength=0.7)` would. An archived memory with strength=7.0 and
   importance=0.07 could still rank above low-importance active memories,
   polluting recall results with stale content that was supposed to be
   effectively invisible.

2. **Provenance chain single-link limitation**: With `successor_id` being a
   single string, chain traversal is simple but limited. If the successor is
   itself consolidated, the chain `archived -> successor -> successor_of_successor`
   works. But if an archived memory contributed to two different semantic
   memories (via clustering), only one link is preserved, and the audit trail
   for the other target is lost.

#### Severity amplification
**AMPLIFIED**. The missing `decayed_strength` is the most dangerous effect: it
creates a scoring bug where archived memories rank higher than intended if they
leak into recall. Combined with the missing `archived` flag (harder to exclude),
this creates a pathway for stale content to appear in LLM prompts.

---

### H4: CATEGORY_COMPRESSION_RATIOS: 4/7 values wrong

**Spec** (spec.md:663-671):
```python
"reasoning": 3.0,    # Preserve reasoning chain
"instruction": 5.0,  # Preserve directive
"greeting": 10.0,    # Trivial
"transactional": 10.0 # Trivial
```

**Implementation** (consolidation.py:84-92):
```python
"reasoning": 7,      # 2.3x MORE compression than spec
"instruction": 3,    # 1.7x LESS compression than spec
"greeting": 1,       # 10x LESS compression (keep everything)
"transactional": 1,  # 10x LESS compression (keep everything)
```

#### Second-order effects

1. **Reasoning over-compression (7 vs 3)**: When `extract_semantic` falls back
   to sentence scoring (line 967-969), a reasoning memory of 700 characters
   gets target_len = 100 (spec: 233). The sentence scorer selects only the
   top-scoring 100 characters. For reasoning memories, the causal chain
   ("because X, therefore Y, which means Z") is typically spread across
   multiple sentences. At 7x compression, intermediate reasoning steps are
   dropped. At 3x, the full chain is more likely preserved.

2. **Instruction under-compression (3 vs 5)**: Instructions get 40% less
   compression than intended. A 300-character instruction produces a
   100-character extraction (spec: 60). This means instructions in the semantic
   memory are 67% larger than intended, consuming more tokens in the recall
   budget.

3. **Greeting/transactional no compression (1 vs 10)**: These categories have
   ratio 1, meaning NO compression at all. The sentence scorer gets
   `target_len = len(content)`, preserving full content. This contradicts the
   spec's intent to heavily compress trivial content. However, these categories
   almost never reach the extraction path (blocked by NON_CONSOLIDATABLE_CATEGORIES
   for correction; blocked by semantic_value < 0.6 for greeting/transactional).
   They can only reach extraction via storage pressure bypass.

#### Third-order effects

1. **Reasoning nuance loss**: With 7x compression, a reasoning chain like
   "I chose Python because it has good data science libraries, therefore I
   can use pandas and numpy for data analysis, which means I don't need to
   learn R for this project" gets compressed to approximately "I chose Python"
   or "I don't need to learn R" -- losing the causal connection. At 3x, the
   full chain or at least the key connectives would survive. This degrades
   the quality of all semantic memories derived from reasoning content.

2. **Token budget inflation from instructions**: With instructions 67% larger
   than intended, the recall pipeline's token budget (800 tokens default)
   fills faster. Under budget pressure (recall.py:579-711), instruction
   memories get demoted from HIGH to MEDIUM or LOW tier, reducing their
   rendering detail. The very memories that need high-fidelity rendering
   (standing instructions like "always respond in French") get compressed
   at the recall stage because they were under-compressed at the
   consolidation stage.

3. **Greeting/transactional: minimal real impact**: Since these categories
   almost never reach extraction (see analysis in spec section 5.2: max
   semantic_value for greeting/transactional is 0.40, below the 0.6
   threshold), the wrong compression ratio is dead code for normal operation.
   Only under storage pressure bypass does this matter, and under pressure,
   full-content greetings waste storage space but don't cause correctness
   failures.

#### Severity amplification
**AMPLIFIED for reasoning** (quality degradation in the most nuanced category).
**NEUTRAL for greeting/transactional** (dead code path in practice).
**MODERATE for instruction** (budget inflation is recoverable via recall
demotion, but reduces quality).

---

### H6: L3->L4 uses cluster sim not retention threshold

**Implementation** (consolidation.py:77-81):
```python
LEVEL_RETENTION_THRESHOLDS: dict[ConsolidationLevel, float] = {
    ConsolidationLevel.EPISODIC_RAW: 0.4,
    ConsolidationLevel.EPISODIC_COMPRESSED: 0.3,
    ConsolidationLevel.SEMANTIC_FACT: 0.3,  # <-- retention threshold
}
```

The `should_consolidate` function (line 755) looks up this dict for the
candidate's level and compares against `retention(last_access_time, strength)`.
For SEMANTIC_FACT level, this means L3->L4 promotion triggers when retention
decays below 0.3.

**Spec** (spec.md:63-67): L3->L4 should use cluster similarity > 0.6, NOT
retention decay. The spec explicitly says "consolidate when cluster similarity
> 0.6" for the SEMANTIC_FACT -> SEMANTIC_INSIGHT transition.

#### Second-order effects

1. **Wrong promotion criterion**: Semantic facts get promoted to insights
   based on how stale they are, not based on whether multiple related facts
   exist. A single isolated fact like "user lives in Tokyo" could be promoted
   to SEMANTIC_INSIGHT level purely because it hasn't been accessed recently,
   even though there are no other location-related facts to aggregate.

2. **Clustering bypass**: The spec intends that L3->L4 requires cluster
   support (multiple related SEMANTIC_FACT memories that can be aggregated
   into a higher-level insight). Using retention instead means clustering
   is irrelevant for this transition. The `compute_affinity` function (line
   781-820) exists but is never used as a gate for L3->L4 promotion.

#### Third-order effects

1. **Insight inflation**: The SEMANTIC_INSIGHT level is meant for aggregated
   patterns (e.g., "user frequently discusses Python data science" aggregated
   from multiple fact memories). With retention-based promotion, individual
   facts get promoted to "insight" level, inflating the insight count and
   diluting the semantic value of the SEMANTIC_INSIGHT tier. If a consumer
   uses level as a quality signal ("insights are higher quality than facts"),
   this signal becomes meaningless.

2. **Time constant escalation without aggregation**: A fact promoted to
   SEMANTIC_INSIGHT gets the 365-day time constant (LEVEL_TIME_CONSTANTS).
   This means a single isolated fact now persists for a year instead of 180
   days. The spec intends this longer persistence for aggregated patterns
   that represent stable user characteristics, not for individual facts that
   just happened to get stale.

#### Severity amplification
**AMPLIFIED**. This changes the fundamental semantics of the highest
consolidation tier. Every SEMANTIC_INSIGHT produced under the current
implementation represents potentially spurious "insight" from a single stale
fact rather than a validated pattern from multiple corroborating facts.

---

### M4: compute_affinity derives beta indirectly

**Spec** (spec.md:279-281): `semantic_weight_beta: float = 0.7` --
weight of semantic (content) similarity in the affinity function.

**Implementation** (consolidation.py:310, 818-819):
```python
temporal_weight: float = 0.3
# ...
beta = 1.0 - config.temporal_weight  # beta = 0.7
affinity = beta * content_sim + config.temporal_weight * temporal_sim
```

#### Second-order effects

1. **Numerically equivalent**: `beta = 1.0 - temporal_weight = 1.0 - 0.3 = 0.7`,
   which matches the spec's `semantic_weight_beta = 0.7`. The affinity
   computation produces identical results.

2. **Configuration intent differs**: A user tuning `temporal_weight = 0.5`
   thinks they're adjusting temporal influence. They're simultaneously and
   implicitly adjusting semantic weight to 0.5. The spec gives them direct
   control over both (though they're constrained to sum to 1.0, so it's the
   same degree of freedom).

#### Third-order effects

1. None significant. The mathematical equivalence means all downstream
   consumers get correct results. This is a code clarity issue, not a
   functional defect.

#### Severity amplification
**NOT AMPLIFIED**. This finding is purely cosmetic/API-ergonomic.

---

### M2: Missing semantic_boost config field and usage

**Spec** (spec.md:273-274):
```python
semantic_boost: float = 1.2  # Importance multiplier for new semantic memories
```

**Implementation**: Not present in `ConsolidationConfig`. Not used anywhere.

#### Second-order effects

1. **New semantic memories get lower importance**: Without `semantic_boost`,
   the `extract_semantic` function (line 986-996) sets importance to the mean
   of source importances. The spec intends a 1.2x boost: a source with
   importance 0.6 would produce a semantic memory with importance 0.72. Without
   the boost, it stays at 0.6.

2. **Semantic memories compete poorly in recall**: In the recall pipeline
   (recall.py), the scoring function is `w1*relevance + w2*recency +
   w3*importance + w4*sigmoid(activation)`. With importance stuck at the
   source average rather than boosted by 1.2x, semantic memories score lower
   than intended relative to episodic memories. This means the system
   produces semantic memories but they're less likely to be recalled than
   the episodic ones they were meant to replace.

#### Third-order effects

1. **Consolidation effort wasted**: The system spends computation extracting
   semantic content, archiving episodic memories, and creating semantic
   extractions -- but the resulting semantic memories rank lower in recall
   than they should. The episodic-to-semantic transition provides less
   retrieval benefit than the spec intends.

2. **Recall quality degradation**: Users expect that as the system matures,
   semantic memories (clean, compressed knowledge) should increasingly dominate
   recall over raw episodic content (noisy, conversational). Without the boost,
   this transition is slower or may not happen at all -- episodic memories
   with their original importance continue to win recall competitions.

#### Severity amplification
**AMPLIFIED**. This affects every semantic memory produced by the system.
The effect is cumulative -- each missing 0.2x boost means a semantic memory
is slightly less likely to be recalled, and over thousands of memories, the
aggregate recall quality diverges from the intended behavior.

---

### M3: Missing activation_inheritance config field and usage

**Spec** (spec.md:275-276):
```python
activation_inheritance: float = 0.5  # Fraction of source activation inherited
```

**Implementation**: Not present. `extract_semantic` does not set any
activation/access_count on the `SemanticExtraction`. The `SemanticExtraction`
dataclass has no `access_count` field at all.

#### Second-order effects

1. **Semantic memories start cold**: Without activation inheritance, a new
   semantic memory has zero access history. In the scoring function,
   `w4 * sigmoid(access_count)` -- with access_count=0, sigmoid(0)=0.5.
   With inherited activation (e.g., source had 20 accesses, inherited =
   0.5 * 20 = 10), sigmoid(10) ~= 0.99999. This is a significant scoring
   difference.

2. **Cold-start problem for semantic memories**: A semantic memory extracted
   from a frequently-accessed episodic memory loses all activation evidence.
   The novelty bonus (`novelty_bonus = N0 * exp(-gamma * t)`) compensates
   for new memories, but it decays quickly. Without inherited activation,
   the semantic memory has a narrow window to "prove itself" before the
   novelty bonus fades and it drops to baseline activation scoring.

#### Third-order effects

1. **Same as M2 third-order**: Consolidation effort wasted if semantic
   memories can't compete in recall. Combined with M2 (missing importance
   boost), semantic memories are doubly disadvantaged: lower importance AND
   lower activation signal. The recall pipeline favors episodic memories
   even more.

#### Severity amplification
**AMPLIFIED**, especially in combination with M2. Both missing boosts
compound to make semantic memories systematically disadvantaged in recall.

---

### H2: ConsolidationResult missing skipped_indices

**Spec** (spec.md:585):
```python
skipped_indices: frozenset[int]  # candidates evaluated but not consolidated
```

**Implementation** (consolidation.py:618-634): No `skipped_indices` field.
Uses `candidates_evaluated: int` and `candidates_consolidated: int` instead.

#### Second-order effects

1. **No skip audit trail**: An orchestrator that wants to log which specific
   candidates were skipped (and why) gets only aggregate counts. Debugging
   "why wasn't memory X consolidated?" requires re-running `should_consolidate`
   on the specific candidate.

2. **Batch pipeline monitoring impaired**: When `select_consolidation_candidates`
   returns 50 candidates and only 10 consolidate, the orchestrator knows 40
   were skipped but not which 40. It cannot efficiently retry or investigate.

#### Third-order effects

1. Minimal. The aggregate counts provide sufficient information for
   correctness monitoring (error rate = skipped / evaluated). The specific
   indices are needed only for debugging, which can use other means.

#### Severity amplification
**NOT AMPLIFIED**. Purely a debugging/observability convenience.

---

### H8: extract_semantic missing compression_ratio computation

**Implementation** (consolidation.py:967-969): Computes `compression_ratio`
internally for determining `target_len`, but does not record it on the
`SemanticExtraction` output.

```python
compression_ratio = CATEGORY_COMPRESSION_RATIOS.get(category, 5)
target_len = max(1, int(len(content) / compression_ratio))
```

The `SemanticExtraction` dataclass does not have a `compression_ratio` field.

#### Second-order effects

1. **Quality monitoring gap**: Same as C5 second-order effect #2. The
   achieved compression cannot be validated post-hoc.

2. **Spec validation impossible**: The spec says `compression_ratio > 0.0`
   is an invariant. Without the field, this invariant cannot be checked.

#### Third-order effects

1. Combined with H4 (wrong compression ratios), there is no way to detect
   that reasoning memories are being compressed 7x instead of 3x without
   reading the source code. The monitoring gap masks the H4 bug.

#### Severity amplification
**MODERATE**. Mostly a monitoring gap, but it masks H4 which is high-severity.

---

### C3: Missing cross-field validation constraints

**Spec** (spec.md:378-384):
```
l2_to_l3_retention <= l1_to_l2_retention  (stricter threshold for deeper consolidation)
mild_pressure_retention >= retention_threshold  (mild pressure is more lenient)
```

**Implementation**: The `__post_init__` validates each field independently
but does not enforce cross-field relationships.

#### Second-order effects

1. **Inversion possible**: A config with `retention_threshold_l1=0.2` and
   `retention_threshold_l2=0.5` would pass validation. This means L2->L3
   promotion requires HIGHER retention than L1->L2 -- memories are
   consolidated MORE easily at deeper levels, inverting the intended hierarchy.

2. **Pressure threshold inversion**: A config with
   `mild_pressure_retention=0.2` (stricter than default) combined with
   `retention_threshold_l1=0.4` means mild pressure actually requires
   LOWER retention than normal, which is backwards.

#### Third-order effects

1. With inverted thresholds, the consolidation pipeline could promote
   memories to SEMANTIC_FACT more easily than to EPISODIC_COMPRESSED.
   This creates a "promotion cliff" where episodic memories skip the
   compressed stage and jump directly to semantic facts, losing the
   intermediate compression step.

#### Severity amplification
**MODERATE**. Only manifests with non-default configs. Default values are
correct (0.4 > 0.3 satisfies the hierarchy).

---

## Optimizer Findings

### w3 >= 1.0 check unreachable (dead code)

**Implementation** (optimizer.py:179):
```python
if w3 >= 1.0:
    return None
```

Since `w3 = 1.0 - w1 - w2 - w4` and all three are clipped to minimum 0.05,
the maximum w3 is `1.0 - 0.05 - 0.05 - 0.05 = 0.85`. w3 can never reach 1.0.

#### Second-order effects
1. None functional. Dead code does not affect behavior.
2. Code confusion: a reader might wonder under what circumstances w3 >= 1.0
   is possible, wasting analysis time.

#### Third-order effects
None.

#### Severity amplification
**NOT AMPLIFIED**. Pure dead code.

---

### Hardcoded Random(42) seed

**Implementation** (optimizer.py:858):
```python
rng=Random(42),
```

Used in stability evaluation within `objective()`. All stability simulations
use the same seed.

#### Second-order effects
1. **Deterministic stability evaluation**: Every stability scenario uses the
   same random sequence. If a parameter set passes stability with this specific
   seed but fails with others, the optimization finds false-positive solutions.

2. **Overfitting to seed**: The optimizer could find parameters that exploit
   the specific Random(42) sequence to avoid monopolization, but fail under
   different seeds in production.

#### Third-order effects
1. **Production instability**: Parameters validated with seed=42 may exhibit
   monopolization under different random sequences. Since production systems
   don't pin seeds, users could see memory retrieval dominated by one memory
   (monopolization), which is exactly what stability scenarios are meant to prevent.

#### Severity amplification
**MODERATE**. The spec explicitly allows this seed (it's about reproducibility),
but the lack of multi-seed validation is a robustness concern, not a spec
violation.

---

### Silent scenario drops in _try_add

**Implementation** (optimizer.py:270):
```python
# If verification fails, skip silently
```

When `_try_add` generates a scenario but the expected_winner doesn't match
the ranking under reference params, the scenario is silently dropped.

#### Second-order effects
1. **Benchmark coverage gaps**: If many scenarios are dropped, the "120+"
   scenarios may be significantly fewer. Competency categories with more
   drops get less testing coverage, potentially leaving parameter regions
   unexplored.

2. **Hard-to-debug accuracy issues**: If the optimizer reports 100% accuracy
   but only evaluated 60 scenarios instead of 120, the accuracy metric is
   inflated. A user reviewing results sees "100% accurate" without knowing
   half the scenarios were dropped.

#### Third-order effects
1. **Overconfident parameters**: The optimizer selects parameters that score
   well on the surviving scenarios but hasn't been tested on the dropped ones.
   In production, these untested cases may reveal weaknesses.

#### Severity amplification
**MODERATE**. The silent drop is a design choice for self-healing scenarios
(don't include scenarios that are impossible under the reference), but the
lack of logging or counting makes it an observability gap.

---

### PINNED s0 < s_max validation deferred

**Implementation** (optimizer.py:57-59):
```python
PINNED: dict[str, float] = {
    "s0": 1.0,
    "s_max": 10.0,
    ...
}
```

The constraint s0 < s_max (1.0 < 10.0) is satisfied by default values, but
if someone changes PINNED["s_max"] to 0.5, the s0 < s_max check is deferred
to ParameterSet.__post_init__ (engine.py:111-113), which would raise
ValueError during decode(). The optimizer module itself doesn't validate
PINNED consistency.

#### Second-order effects
1. **Silent decode failure**: If PINNED values are inconsistent, every
   `decode()` call returns None. The optimizer treats all candidates as
   infeasible and returns a degenerate result.

2. **Error message confusion**: The ValueError from ParameterSet says
   "s0 must be in (0, s_max=0.5)" but the user changed PINNED, not
   ParameterSet. The error points to the wrong location.

#### Third-order effects
1. Minimal in practice -- PINNED values are module-level constants, not
   user-configurable at runtime.

#### Severity amplification
**NOT AMPLIFIED**. Defensive validation for a non-runtime scenario.

---

### Margin near-zero at float boundary

**Implementation** (optimizer.py:871):
```python
margin_bonus = min(margin, 0.3) / 0.3
```

If `margin` is very close to 0 (e.g., 1e-15), `margin_bonus` = 3.3e-15.
Combined with `MARGIN_WEIGHT = 0.15`, this contributes ~5e-16 to the
composite score.

#### Second-order effects
1. **Float precision noise**: At such small values, the margin component
   is indistinguishable from zero due to float precision. The margin_bonus
   provides no meaningful gradient signal to CMA-ES.

2. **Barely-feasible solutions favored equally**: Two solutions with margins
   of 1e-15 and 0.01 should have very different margin_bonus values. At
   float boundaries, they may both appear as ~0.

#### Third-order effects
1. The optimizer may accept barely-feasible parameters (margin ~= 0) that
   could violate contraction under slight parameter perturbation.

#### Severity amplification
**LOW**. CMA-ES explores widely enough that barely-feasible solutions are
usually dominated by well-feasible ones with larger margins.

---

### Magic number 100

**Implementation** (optimizer.py:856-858):
```python
sim = simulate(params, s.memories, 100, [-1] * 100, rng=Random(42))
```

100 is the number of simulation steps for stability evaluation.

#### Second-order effects
1. **No configurability**: If stability evaluation needs more or fewer
   steps (e.g., convergence requires 200 steps for certain parameter
   combinations), the hardcoded 100 cannot be changed without code modification.

2. **Undocumented assumption**: The 100-step count assumes that monopolization
   patterns are detectable within 100 steps. For some parameter combinations
   (especially with high temperature), convergence may be slow and 100 steps
   may not reveal monopolization that appears at step 200.

#### Third-order effects
1. **False stability pass**: A parameter set that monopolizes after step 150
   passes the 100-step check. In production (unlimited steps), the monopolization
   emerges, causing one memory to dominate all retrievals.

#### Severity amplification
**LOW**. Most monopolization patterns are visible within 50 steps; 100 is
generally sufficient.

---

## Summary: Cascade Severity Ranking

### Critical Cascades (must fix before production)

| Rank | Finding | Cascade Description |
|------|---------|---------------------|
| 1 | **C2** (hours vs days) | 24x over-retention -> pool bloat -> premature storage pressure -> low-quality consolidation. Touches every memory in the system. |
| 2 | **H6 + C1** (L3->L4 wrong criterion) | Retention-based promotion instead of cluster similarity -> spurious insights -> insight tier quality collapse. |
| 3 | **H1** (missing decayed_strength + archived flag) | Archived memories score higher than intended -> stale content leaks into recall -> LLM prompt pollution. |
| 4 | **M2 + M3** (missing semantic_boost + activation_inheritance) | Semantic memories systematically disadvantaged in recall -> consolidation effort wasted -> system never matures from episodic to semantic retrieval. |

### High-Severity Cascades (significant quality impact)

| Rank | Finding | Cascade Description |
|------|---------|---------------------|
| 5 | **H4** (reasoning compression 7 vs 3) | Reasoning memories lose causal chains -> degraded quality of semantic reasoning memories. |
| 6 | **C4** (missing source_episodes, consolidation_count) | Provenance chain breaks on re-consolidation -> audit trail loss -> re-consolidation safety compromised. |
| 7 | **M1** (IntEnum vs string Enum) | Serialization mismatch when persistence is added -> data migration required. |

### Moderate Cascades (quality or observability impact)

| Rank | Finding | Cascade Description |
|------|---------|---------------------|
| 8 | **C5** (SemanticExtraction field mismatches) | Interface contract deviation -> consumer adaptation required. |
| 9 | **H8 + H4** (missing compression_ratio + wrong ratios) | Monitoring gap masks compression bugs. |
| 10 | **C3** (missing cross-field validation) | Allows inverted threshold hierarchy with non-default configs. |
| 11 | **Silent scenario drops** (optimizer) | Benchmark coverage gaps -> overconfident parameter selection. |

### Isolated Findings (no downstream cascade)

| Finding | Reason |
|---------|--------|
| **w3 >= 1.0 dead code** | Pure dead code, no behavior change. |
| **Random(42) seed** | Spec-compliant; robustness concern only. |
| **PINNED s0 validation** | Constants, not runtime-configurable. |
| **Margin near-zero** | Float precision at extreme boundary only. |
| **Magic number 100** | Sufficient for most parameter combinations. |
| **M4** (beta derived indirectly) | Mathematically equivalent; API ergonomic only. |
| **H2** (missing skipped_indices) | Debugging convenience only. |
| **Minimal _try_add docstring** | Code quality, no behavior impact. |

---

## Key Insight: Compound Effects

The most dangerous cascades involve multiple findings reinforcing each other:

1. **C2 + storage pressure**: The 24x over-retention (C2) causes premature
   storage pressure, which bypasses semantic value filtering, producing
   low-quality consolidation output. Fix C2 and the storage pressure cascade
   resolves.

2. **M2 + M3**: Missing importance boost AND missing activation inheritance
   compound to make semantic memories doubly disadvantaged. Either fix alone
   would partially address the recall quality issue; both fixes together are
   needed for the spec-intended behavior.

3. **H4 + H8**: Wrong compression ratios (H4) are undetectable because
   compression_ratio is not recorded (H8). Fix H8 first (add monitoring),
   then H4 becomes visible and fixable.

4. **H6 + C1**: L3->L4 uses wrong criterion (H6) AND has a differently-named
   config field (C1), making it doubly hard to diagnose: an operator looking
   at the spec would try to tune `l3_to_l4_cluster_sim` (which doesn't exist)
   to fix cluster-based promotion (which isn't implemented).
