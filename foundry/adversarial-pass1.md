# Adversarial Pass 1: Brenner Level-Split + Paradox-Hunt

**Reviewer**: architect-agent (Opus 4.6)
**Date**: 2026-04-14
**Spec version**: 1.0 (2026-04-13)
**Operators applied**: Level-Split (14 findings), Paradox-Hunt (11 findings)

---

## Executive Summary

The spec is architecturally strong but has 25 findings across two operator classes. Of these, **7 are potentially fatal** (could cause silent data corruption or contradictory runtime behavior) and **18 are serious** (will cause confusion during implementation or create regression surface). Each finding below names the operator, states the problem, and provides the mitigation that has been added to the spec.

---

## Level-Split Findings (14)

### LS-1: Ingestion-Time vs Query-Time Entity Resolution [FATAL]

**Location**: Section 3.4 (EntityResolver) / Section 4.3.5 (Entity Resolution)

**Problem**: The spec defines `EntityResolver` with two modes — full cascade (`resolve()`) for ingestion and exact-only (`resolve_exact()`) for query time — but conflates them into a single class with no explicit contract on when each mode is invoked. The `resolve()` method includes steps 4-6 (substring, embedding, create-new) which would be catastrophic at query time (creating ghost entities from typos, merging unrelated entities via embedding similarity). But the class definition does not enforce this boundary. An implementer could call `resolve()` at query time by mistake.

Additionally, `resolve_exact()` is defined as stopping "at step 2" (determiner strip), but `resolve()` step 3 (possessive strip) is also safe for query time and arguably should be included in `resolve_exact()`.

**Mitigation**: Added explicit runtime mode contract to Section 3.4 with an enumerated safety boundary.

---

### LS-2: Temporal Fields — Event Time vs System Time vs Turn-Index Time [FATAL]

**Location**: Section 2.2 (Fact dataclass), Section 6.3 (TemporalValidity)

**Problem**: The `Fact` dataclass has THREE temporal fields with conflated semantics:
- `timestamp`: "Event time (when the fact occurred in the world)"
- `session_time`: "Conversation time (when user said it)"
- `turn_idx`: Ordinal position in conversation

The `TemporalValidity` dataclass then accepts `Optional[int | datetime]` for `valid_from`/`valid_until`, creating a union type that conflates turn-index ordering with datetime ordering. These are fundamentally different comparison operations: `turn_idx` is a monotonic ordinal (5 > 3), while `datetime` requires timezone-aware comparison. A fact with `valid_from=5` (turn index) cannot be meaningfully compared with `valid_from=datetime(2024,6,1)`.

The supersession engine uses `turn_idx` for ordering ("highest turn_idx wins"), but `TemporalValidity.is_current()` operates on the `valid_from`/`valid_until` fields which could be either type. This creates a type confusion bug.

**Mitigation**: Added explicit type separation contract to Sections 2.2 and 6.3.

---

### LS-3: QueryPlan vs Execution Results — Plan Immutability vs Result Caching [SERIOUS]

**Location**: Section 3.2 (QueryPlan)

**Problem**: The spec says QueryPlan is lazy and immutable (each method returns a NEW plan). But terminal operations like `execute()` must cache results somewhere. If results are cached on the plan object, the plan is no longer immutable in spirit. If they are NOT cached, then calling `execute()` twice on the same plan runs the full computation twice (idempotent but wasteful).

The prior-art synthesis (Section 2.1) references Django's `_result_cache` pattern, where `_fetch_all()` populates a cache on the QuerySet. But Django QuerySets are NOT truly immutable — they mutate `_result_cache` on first evaluation. The spec inherits this inconsistency without acknowledging it.

**Mitigation**: Added result-caching contract to Section 3.2c clarifying that plans are structurally immutable but may cache execution results as an internal optimization.

---

### LS-4: Config vs Runtime State in FoundryConfig [SERIOUS]

**Location**: Section 10.1 (FoundryConfig)

**Problem**: `FoundryConfig` mixes three concerns:
1. **Schema config** (chunk_size, max_entities, entity_registry_cap) — affects data layout, cannot change between sessions
2. **Runtime config** (extraction_url, embedding_model, reranker_model) — can change between sessions without data corruption
3. **Model references** (extraction_model, temporal_model, agent_model) — may change but must be compatible with existing extraction cache

Changing `chunk_size` from 512 to 1024 after ingestion silently invalidates all stored chunks and BM25 indexes. Changing `extraction_model` invalidates the extraction cache but not the stored facts. These are different failure modes but the config treats them uniformly.

**Mitigation**: Added configuration category annotations to Section 10.1.

---

### LS-5: Link Types — Schema-Level vs Instance-Level [SERIOUS]

**Location**: Section 1.5 (Link Types)

**Problem**: `LinkType` is defined as a frozen schema-level dataclass (with cardinality, source/target type constraints), but links are ALSO materialized at instance level from S-P-O triples ("the predicate becomes the link name"). This means the same predicate "visited" could appear as both a schema-level `LinkType(name="visited", inverse_name="visited_by", cardinality=ONE_TO_MANY)` and as an instance-level link from triple extraction with no cardinality constraint.

There is no reconciliation rule. Does the schema-level cardinality override instance-level? Is there even a registry of LinkTypes, or are they purely emergent from extraction?

**Mitigation**: Added schema vs instance link reconciliation rules to Section 1.5.

---

### LS-6: DerivedGraph vs EntityGraph — Different Structures, Same Name Pattern [SERIOUS]

**Location**: Section 2.4 (Persistence Model), Section 7.1 (MemorySystem)

**Problem**: The storage layer has `entity_graph: EntityGraph` (stated), `inferred_graph: EntityGraph` (inferred), and `derived: DerivedGraph` (derived). The first two share a type (`EntityGraph`) but the third is a different type (`DerivedGraph`). Yet the MemorySystem (Section 7.1) declares `self.stated: EntityGraph`, `self.inferred: EntityGraph`, `self.derived: DerivedGraph`.

The spec never defines `DerivedGraph` or its interface. It defines `DerivedLayerBuilder.build()` which returns a `DerivedGraph`, and `DerivedLayerBuilder.extract_gists()` which returns `dict[str, list[dict]]`. What IS a DerivedGraph? Is it an EntityGraph subclass? A dict? A new data structure?

**Mitigation**: Added DerivedGraph interface definition to Section 4.4.

---

### LS-7: EntityResolver.resolve() — Alias Index vs Embedding Fallback [SERIOUS]

**Location**: Section 3.4 (Resolution cascade)

**Problem**: The resolution cascade step 5 says "Embedding similarity (when embed_fn is set, threshold >= 0.85)." But embedding similarity is a fundamentally different operation from alias lookup:
- Alias lookup is deterministic: same input always gives same output
- Embedding similarity depends on the embedding model, which is configurable

If `embed_fn` changes (different model, or model update), previously unresolvable mentions may now resolve to different entities. This breaks the invariant I8 ("resolve(x) is deterministic"). The spec conflates deterministic resolution (steps 1-4) with probabilistic resolution (step 5) without acknowledging the boundary.

**Mitigation**: Added deterministic vs probabilistic resolution boundary to Section 3.4.

---

### LS-8: search_memory Tool — Conflated BM25 + Cosine + Rerank Pipeline [SERIOUS]

**Location**: Section 5.2 (Tool Definitions), Section 5.3 (Hybrid Search)

**Problem**: The `search_memory` tool bundles three retrieval channels (BM25, cosine, rerank) into a single opaque tool. But the agentic retriever also has `search_entities` and `search_temporal` as separate tools. This creates a level confusion: the agent cannot separately request BM25-only vs cosine-only retrieval, but CAN separately request entity vs chunk retrieval.

From ablation results: "BM25 baseline: 58.3% SubEM" — the BM25 channel alone is the strongest. Forcing it through RRF with cosine may dilute results (proven: "RRF fusion of entity channel: E2-E6 all regressed"). The agent has no way to request the proven-strongest channel in isolation.

**Mitigation**: Added BM25-only access note to Section 5.2.

---

### LS-9: Fact.id Generation — UUID vs Deterministic [SERIOUS]

**Location**: Section 2.2 (Fact dataclass)

**Problem**: `Fact.id` is described as "Unique identifier (uuid4)." But invariant I7 (Dedup Idempotency) says "Ingesting the same triple twice produces exactly one fact." If IDs are random UUIDs, then the same triple ingested twice produces two Facts with different IDs, and dedup must use content-based comparison (subject + predicate + object + turn_idx). But `Fact.superseded_by` references Fact IDs, and the supersession engine tracks by ID.

The spec conflates the ID (used for cross-referencing) with dedup identity (content-based). These need different mechanisms.

**Mitigation**: Added dedup identity definition to Section 2.2 and I7.

---

### LS-10: PropertyType Inference — Parsing Rules vs PropertyIndex Storage [SERIOUS]

**Location**: Section 1.3 (Property Types), Section 2.3.4 (Property Index)

**Problem**: `infer_property_type()` is called during ingestion to classify values. But the PropertyIndex stores `(entity, predicate, value, property_type)` tuples. What happens when the SAME predicate on DIFFERENT entities has different inferred types? E.g., entity "car" has `cost="$3200"` (NUMERIC) and entity "trip" has `cost="expensive"` (STRING).

`PropertyIndex.lookup_numeric("cost", "$gt", 1000)` will find the car but silently skip the trip. Is this correct? The spec does not address mixed-type predicates.

**Mitigation**: Added mixed-type predicate handling rule to Section 2.3.4.

---

### LS-11: Grounding Verification — Exact Substring vs Word-Level Fallback [SERIOUS]

**Location**: Section 4.3.9 (Grounding Verification)

**Problem**: The grounding check has two modes: exact substring and word-level fallback. The word-level fallback only checks words with `len(w) >= 3`. This means a fact object of "LA" (2 chars) passes grounding even if "LA" does not appear in source_text, because all words are filtered out by the length check. Similarly, "US" or "UK" pass trivially.

This is a level confusion between "tokens too short to be meaningful as grounding evidence" and "tokens that ARE the entire fact value." When the ENTIRE object is short, skipping it means no grounding check at all.

**Mitigation**: Added short-value grounding rule to Section 4.3.9.

---

### LS-12: FoundryStore.save/load — Atomic Claim vs Partial Write Reality [SERIOUS]

**Location**: Section 2.4 (Persistence Model)

**Problem**: The contract says "Atomic: either all indexes save/load or none do." But the implementation saves "one JSON file per index + binary files for embeddings." With 7+ indexes being written to separate files, a crash mid-save leaves partial state on disk. The spec claims atomicity without specifying the mechanism (write-to-temp + rename, journal, etc.).

**Mitigation**: Added atomicity implementation requirement to Section 2.4.

---

### LS-13: Allen Interval — Strict vs Inclusive Boundary Semantics [SERIOUS]

**Location**: Section 2.3.5 (Temporal Index)

**Problem**: Allen's original algebra defines `before(A, B)` as "A.end < B.start" (strict inequality — there is a gap). But the `meets` relation is defined as "A.end == B.start" (touching). The spec implements both but does not state whether the boundary comparison is strict or inclusive for the `events_between(start, end)` convenience method.

If `events_between(Jan_1, Jan_5)` uses inclusive comparison, then an event ending on Jan_5 is included. But `allen_before([Jan_1, Jan_5], [Jan_5, Jan_10])` should be False (they meet, not before). The convenience methods and Allen relations must use the SAME boundary semantics.

**Mitigation**: Added boundary semantics specification to Section 2.3.5.

---

### LS-14: QueryPlan .related() vs .pivot_to() vs .search_around() [SERIOUS]

**Location**: Section 3.2, 3.6

**Problem**: Three traversal methods exist with overlapping but distinct semantics:
- `.related()`: "ADDS connected entities to the set"
- `.pivot_to()`: "REPLACES the current entity set with the targets"
- `.search_around()`: "Navigate from current set through links" (unclear if ADD or REPLACE)

The spec defines `.search_around()` alongside `.related()` and `.pivot_to()` without clarifying which semantic it follows. The Palantir-enriched text (Section 3.6) says `search_around` is equivalent to `pivot_to`, but Section 3.2 lists it separately with different parameter signatures.

**Mitigation**: Added traversal semantic clarification to Section 3.2.

---

## Paradox-Hunt Findings (11)

### PH-1: "No LLM Calls for Infrastructure" vs HyDE and Query Augmentation [FATAL]

**Location**: Axiom 2 / Section 5.8 (OAG Patterns)

**Problem**: Design Axiom 2 states: "No LLM calls for infrastructure. Memory must be cheap to operate. LLM calls are reserved for extraction (ingestion-time) and answer generation (query-time)."

But Section 5.8 introduces HyDE (Hypothetical Document Embeddings): "Instead of embedding the raw query, first ask an LLM to generate a hypothetical answer chunk, then embed that." The spec even acknowledges the contradiction: "Cost: violates Axiom 2 — use only as a fallback."

There is also `augment_query()` which is deterministic (no LLM), but HyDE is explicitly LLM-powered. The spec contradicts itself. Having a "violates the axiom but use it anyway" exception undermines the axiom's authority.

**Mitigation**: Added HyDE classification as Axiom-2 exception with explicit gating conditions.

---

### PH-2: "Always-On, Never Routed" vs DeterministicRouter [FATAL]

**Location**: Axiom 4 / Section 5.6 (Deterministic Retrieval Pipelines)

**Problem**: Design Axiom 4 states: "Always-on, never routed. Every subsystem runs unconditionally. There is no intent-based routing."

But `DeterministicRouter.route()` is literally an intent router: "Classification: regex patterns, no LLM (< 1ms). Pipelines hardcoded from trace analysis of common patterns." It classifies the question type and routes to a specific pipeline (fact_lookup, temporal, aggregation, preference, comparison). This IS intent-based routing.

The distinction "no LLM routing" is not the same as "no routing." The axiom says "no routing" period, but the spec implements routing.

**Mitigation**: Added axiom refinement to the Preamble distinguishing retrieval-channel routing (banned) from query-optimization routing (permitted).

---

### PH-3: "BM25 is Primary" vs Knowledge Cascade Priority [FATAL]

**Location**: Axiom 1 / Prior-Art Synthesis Section 2.4

**Problem**: Axiom 1: "BM25 is primary. Everything else adds on top of keyword search, never replaces it."

Prior-art synthesis (adopted design): "Query -> MSL Structured Store (schema-constrained, typed entities/relations) -> if insufficient -> BM25/Cosine over extracted triples -> if insufficient -> BM25/Cosine over raw conversation chunks"

The KAG cascade puts structured store FIRST and BM25 LAST. This directly contradicts Axiom 1 which says BM25 is primary. The cascade implies BM25 is the fallback, not the primary.

**Mitigation**: Added cascade priority reconciliation to Section 5.1.

---

### PH-4: "Grounding = User's Exact Words" vs Inferred Layer Semantic Bridges [SERIOUS]

**Location**: Axiom 5 / Section 1.6 (Grounding Model)

**Problem**: Axiom 5: "Stated over inferred. The graph stores the user's exact words, not the model's interpretation."

But the Inferred layer creates "synonym/type bridges for navigation." When a query for "feline" traverses through the inferred layer to find the stated entity "cat," the system is using model-generated semantic bridges (from a synonym dictionary or WordNet) to connect user words to non-user words. The inferred layer is explicitly model interpretation.

The spec claims this is safe because "Inferred never appears in prompts to the LLM" and is "navigation-only." But navigation affects which STATED facts are retrieved. A bad synonym bridge (e.g., "jaguar" -> "cat" when the user means the car) corrupts retrieval without directly polluting the stated layer.

**Mitigation**: Added inferred-layer safety constraints to Section 1.6.

---

### PH-5: Entity Supersession "Latest Wins" vs Temporal Validity Windows [SERIOUS]

**Location**: Section 6.2 vs Section 6.3

**Problem**: The SupersessionEngine uses "latest_wins" by default: "highest turn_idx wins." But TemporalValidity tracks `valid_from`/`valid_until` windows which can overlap. Consider:

Turn 3: "I lived in SF since 2015"  -> valid_from=2015
Turn 7: "I was in Portland for a month last year" -> valid_from=2023, valid_until=2023
Turn 10: "I'm back in SF now" -> valid_from=2024

Under "latest_wins" (turn_idx=10), "back in SF" is current. But there was a period where Portland was correct (2023). The supersession model marks the turn-3 SF fact as superseded by turn-7 Portland, which is then superseded by turn-10 SF. This creates a correct timeline, BUT the "latest_wins" strategy does not consult temporal validity — it only looks at turn_idx.

The paradox: a fact with EARLIER event time but LATER turn_idx would supersede a fact with LATER event time but EARLIER turn_idx. The supersession model conflates discourse order (turn_idx) with temporal order (event time).

**Mitigation**: Added supersession ordering clarification to Section 6.2.

---

### PH-6: "Forward Compatible" Design vs No Schema Versioning [SERIOUS]

**Location**: Preamble (forward-compatible architecture) / Section 2.4 (Persistence)

**Problem**: The preamble says Foundry is a "forward-compatible architecture" and explicitly states "backward compatibility with mabench code is explicitly not a goal." But the persistence model (save/load) has no schema version field. If the Fact dataclass gains a new field (e.g., `confidence`), loading an old save file with the new code will either crash (missing field) or silently default (losing the information that this is an old-format fact).

**Mitigation**: Added schema version requirement to Section 2.4.

---

### PH-7: resolve_exact() "Steps 1-2 Only" vs Possessive Strip Safety [MINOR]

**Location**: Section 3.4 (EntityResolver)

**Problem**: `resolve_exact()` is defined as stopping at step 2 (determiner strip). But possessive stripping (step 3: "user's cat" -> "user cat") is equally safe for query time — it is a deterministic string transformation with no risk of false merges. The current definition means `resolve_exact("user's cat")` will FAIL to find entity "user_cat" even though the mapping is unambiguous.

**Mitigation**: Added possessive strip to resolve_exact scope in Section 3.4.

---

### PH-8: VerbNet Classification at Extraction vs Query Time [SERIOUS]

**Location**: Section 4.3.10 (Role Classification) / Section 8.1 (counting_aggregation)

**Problem**: VerbNet role classification happens at extraction time (Stage 3) and is stored on the Fact. But Section 8 (counting_aggregation) says counting queries filter by sub_role at QUERY time. If the VerbNet classification was wrong during extraction (e.g., "I got a parrot" classified as acquisition when user meant "I got a parrot image"), the error is permanent.

There is no query-time reclassification or correction mechanism. The spec conflates extraction-time classification (best-effort) with query-time filtering (assumed correct).

**Mitigation**: Added extraction-time classification uncertainty note to Section 4.3.10.

---

### PH-9: "Index at Ingestion, Query at Read" vs Incremental Re-indexing [SERIOUS]

**Location**: Axiom 6 / Section 2.5 (Indexing Pipeline Architecture)

**Problem**: Axiom 6: "Index at ingestion, query at read. Pre-compute everything that can be pre-computed."

But Section 2.5 describes incremental ingestion: "ingestion can skip turns with turn_idx <= max_existing_turn_idx." When new turns are added, the BM25 index must be updated (new documents), the type index must be updated (new entities may change type counts), the co-occurrence index must be updated (new pairs). These are all WRITES happening after the initial "ingestion."

The axiom implies a clean separation between write (ingestion) and read (query). But incremental ingestion means writes happen continuously. The axiom should be "write-time compute" not "ingestion-time compute."

**Mitigation**: Reworded Axiom 6 for clarity.

---

### PH-10: max_hops=3 Proven Max vs QueryPlan Accepting Arbitrary max_hops [SERIOUS]

**Location**: Section 3.2 (TraversalOp max_hops comment) / Section 3.2 (QueryPlan.related)

**Problem**: `TraversalOp` has `max_hops: int = 1` with the comment "Maximum traversal depth (proven max useful: 3)." But neither `TraversalOp.__init__` nor `QueryPlan.related()` enforces this cap. An implementer could pass `max_hops=100`, causing exponential traversal on a dense graph.

The spec says "proven diminishing returns beyond 3" but does not mandate clamping. The test suite (`test_traversal_max_hops_cap`) tests that max_hops=3 works, but does not test that max_hops=4 is rejected or clamped.

**Mitigation**: Added hard clamp requirement to Section 3.2.

---

### PH-11: "Deterministic Given Same Input" Pipeline vs Embedding Model Dependency [SERIOUS]

**Location**: Section 4.1 / Section 2.3.2 (VectorIndex)

**Problem**: Section 4.1: "The pipeline is deterministic given the same input and extraction model."

But the pipeline also populates the VectorIndex, which depends on the embedding model (configurable in FoundryConfig). Two runs with different embedding models produce different vector indexes, meaning different cosine retrieval results. The pipeline is deterministic only if ALL models (extraction + embedding + temporal) are fixed.

**Mitigation**: Added model-determinism scope to Section 4.1.

---

## Summary Table

| ID | Operator | Severity | Section | Core Issue |
|----|----------|----------|---------|------------|
| LS-1 | Level-Split | FATAL | 3.4, 4.3.5 | Ingestion vs query-time resolution not enforced |
| LS-2 | Level-Split | FATAL | 2.2, 6.3 | Turn index vs datetime types conflated in TemporalValidity |
| LS-3 | Level-Split | SERIOUS | 3.2c | Plan immutability vs result caching unaddressed |
| LS-4 | Level-Split | SERIOUS | 10.1 | Schema vs runtime config mixed |
| LS-5 | Level-Split | SERIOUS | 1.5 | Schema-level vs instance-level links unreconciled |
| LS-6 | Level-Split | SERIOUS | 2.4, 7.1 | DerivedGraph undefined |
| LS-7 | Level-Split | SERIOUS | 3.4 | Deterministic vs probabilistic resolution blended |
| LS-8 | Level-Split | SERIOUS | 5.2, 5.3 | BM25 not independently accessible to agent |
| LS-9 | Level-Split | SERIOUS | 2.2 | UUID identity vs content-based dedup conflated |
| LS-10 | Level-Split | SERIOUS | 1.3, 2.3.4 | Mixed-type predicates unaddressed |
| LS-11 | Level-Split | SERIOUS | 4.3.9 | Short-value grounding bypass |
| LS-12 | Level-Split | SERIOUS | 2.4 | Atomicity claimed without mechanism |
| LS-13 | Level-Split | SERIOUS | 2.3.5 | Allen boundary semantics unspecified |
| LS-14 | Level-Split | SERIOUS | 3.2, 3.6 | Three traversal ops with unclear semantics |
| PH-1 | Paradox-Hunt | FATAL | Axiom 2, 5.8 | HyDE contradicts no-LLM axiom |
| PH-2 | Paradox-Hunt | FATAL | Axiom 4, 5.6 | DeterministicRouter IS intent routing |
| PH-3 | Paradox-Hunt | FATAL | Axiom 1, 2.4 synthesis | KAG cascade inverts BM25 primacy |
| PH-4 | Paradox-Hunt | SERIOUS | Axiom 5, 1.6 | Inferred bridges are model interpretation |
| PH-5 | Paradox-Hunt | SERIOUS | 6.2, 6.3 | Supersession conflates discourse and temporal order |
| PH-6 | Paradox-Hunt | SERIOUS | Preamble, 2.4 | No schema version for forward compatibility |
| PH-7 | Paradox-Hunt | MINOR | 3.4 | resolve_exact too restrictive |
| PH-8 | Paradox-Hunt | SERIOUS | 4.3.10, 8.1 | Extraction-time classification assumed infallible |
| PH-9 | Paradox-Hunt | SERIOUS | Axiom 6, 2.5 | Axiom wording conflicts with incremental ingestion |
| PH-10 | Paradox-Hunt | SERIOUS | 3.2 | max_hops not enforced |
| PH-11 | Paradox-Hunt | SERIOUS | 4.1, 2.3.2 | Determinism claim ignores model dependency |

---

## Next Steps

All 25 mitigations have been added inline to the spec as `<!-- ADVERSARIAL-PASS-1: ... -->` blocks. Each block contains:
1. The finding ID (LS-N or PH-N)
2. The operator applied
3. The mitigation text

These should be reviewed by the spec author and either accepted (integrated into the main text) or rejected with rationale.
