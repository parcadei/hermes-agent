# Adversarial Pass 4: CK Architecture Review (∞→Ω→μ→∾)

**Reviewer**: architect-agent (Opus 4.6)
**Date**: 2026-04-14
**Spec version**: 1.0 (2026-04-13), with Appendix A (Pass 1), B (Pass 2), C (Pass 3) mitigations
**Operators applied**: Recursion (∞, 8 findings), Requisite Variety (Ω, 6 findings), Good Regulator (μ, 6 findings), Ultrastability (∾, 5 findings)

---

## Executive Summary

Passes 1-3 found epistemological problems: contradictions (25), quantitative infeasibility (26), decision fragility (36). This pass finds a fourth class: **structural gaps** — missing subsystems, broken feedback loops, insufficient controller variety, and absent adaptation mechanisms.

Of the 25 findings:

- **3 STRUCTURAL GAPS** Brenner missed entirely: no S3* audit subsystem, no S4 intelligence/adaptation subsystem, no model of self (the system cannot introspect on what it knows)
- **5 BROKEN RECURSION** points where the spec's structural patterns fail to compose
- **4 VARIETY DEFICITS** where controller responses are fewer than disturbance types
- **3 ABSENT REGULATORS** where controllers operate without a model of what they control
- **2 MISSING ADAPTATION LOOPS** where the system cannot learn from its own operation

The headline finding: **the system has no model of itself.** The query planner does not know what the storage layer contains. The agentic retriever cannot estimate whether an entity exists before searching. The ingestion pipeline has no model of existing entities beyond the alias index. This violates the Good Regulator theorem — every good regulator must contain a model of the system it regulates. Without self-models, every controller in Foundry operates blind.

---

## Recursion Analysis (∞)

### REC-1: Entity→Fact→Link Pattern Does NOT Repeat at Conversation Level

- **Level**: The Entity→Fact→Link pattern holds within a single EntityGraph. It does NOT extend to the conversation level.
- **Expected**: A conversation should be representable as an entity (with facts about the conversation: topic, participants, timestamps) and links to other conversations (follow-ups, topic continuation). This would allow queries like "What conversations discussed cats?" or "Show me the conversation thread about the job change."
- **Actual**: Conversations are invisible in the entity graph. They exist only as ranges of `turn_idx` values. There is no `ConversationEntity` type, no conversation-level facts, no inter-conversation links. RawChunks know their `turn_idx` and `session_time` but have no conversation_id or conversation-level metadata.
- **Impact**: Questions like "In which conversation did the user mention X?" require scanning all chunks and grouping by session_time gaps — an ad-hoc reconstruction, not a structured query. The QueryPlan cannot filter by conversation because conversations are not entities.
- **Spec location**: Section 2.2 (RawChunk has no conversation_id), Section 4.3.1 (ConversationTurn has no conversation-level aggregation)
- **Fix**: Add `ConversationNode` as a first-class entity type in the EntityGraph. During Stage 1 (Turn Parsing), create a ConversationNode for each session (delimited by "Chat Time:" headers). Link ConversationNodes to their constituent entities. Add `conversation_id: Optional[str]` to RawChunk and Fact. This allows `foundry.entities().type("conversation").where("topic", "$contains", "cats").execute()` as a structured query.

---

### REC-2: ObjectSet Pattern Does NOT Compose Recursively for Nested Queries

- **Level**: The QueryPlan supports chaining (`.type().where().related()`) but does NOT support nesting (subqueries).
- **Expected**: "Entities whose connections have property X" should be expressible as: `foundry.entities().related("visited").where("status", "$eq", "open")` — which finds entities connected to open things. But the semantic is ambiguous: does `.where()` filter the CURRENT set or the TRAVERSAL TARGETS?
- **Actual**: After `.related()`, the entity set is expanded. `.where()` filters the EXPANDED set (current + related). There is no way to express "entities that HAVE a related entity with property X" without materializing the intermediate traversal result and joining manually.
- **Impact**: Multi-hop queries with conditions on intermediate hops cannot be expressed in a single QueryPlan chain. "Find restaurants where the chef has a Michelin star" requires two separate queries and an application-level join.
- **Spec location**: Section 3.2 (QueryPlan has no subquery mechanism), Section 3.6 (Search Around)
- **Fix**: Add a `.where_related(predicate, filter_on_target)` method that applies a filter to traversal targets WITHOUT adding those targets to the result set. This is a constrained subquery: `foundry.entities().type("restaurant").where_related("chef", FilterOp("award", "$eq", "michelin"))` returns restaurants (not chefs) that have a Michelin-starred chef. Implementation: traverse, filter targets, return only source entities with at least one qualifying target.

---

### REC-3: Ingestion Pipeline Has THREE Separate Structures for Single-Turn, Multi-Turn, and Batch

- **Level**: The ingestion pipeline (Section 4.2) defines a single 11-stage pipeline. But the API surface (Section 10.2) exposes THREE entry points: `ingest(raw_text)`, `ingest_turns(turns)`, and `ingest_chunk(chunk, triples, entities)`.
- **Expected**: All three entry points should feed into the SAME pipeline stages. The only difference should be the entry point (which stage they start at).
- **Actual**: `ingest(raw_text)` runs all 11 stages. `ingest_turns(turns)` skips Stage 1 (Turn Parsing). `ingest_chunk(chunk, triples, entities)` skips Stages 1-4 (parsing, chunking, triple extraction, faceted extraction) but still runs 5-11. Each entry point is a DIFFERENT code path through the pipeline, not a parameterized single path.
- **Impact**: Bugs in one code path may not exist in another. A fix to Stage 5 (Entity Resolution) applied via `ingest()` may not propagate to the `ingest_chunk()` path if they use different function call chains. This violates structural homology — the pipeline should be one thing, entered at different points.
- **Spec location**: Section 10.2 (three ingestion methods), Section 4.2 (single pipeline definition)
- **Fix**: Define the pipeline as a single ordered list of stages. Each entry point specifies a `start_stage: int`. `ingest()` starts at stage 1. `ingest_turns()` starts at stage 2. `ingest_chunk()` starts at stage 5. All paths converge into the same stage implementations. No stage should have conditional logic based on entry point.

---

### REC-4: Supersession Does NOT Chain at Schema Level

- **Level**: Fact-level supersession is well-defined (Section 6.2). Entity-level merge is defined (Section 6.4). Schema-level supersession is absent.
- **Expected**: When the ontology evolves (e.g., a new facet is added, or a type hypernym is updated), existing entities should be re-evaluated against the new schema. Facts that were valid under the old schema may be invalid under the new one.
- **Actual**: There is no mechanism to supersede a schema change. `FoundryConfig` changes at the SCHEMA level (chunk_size, max_entities) invalidate stored data (A.5), but there is no migration pipeline. The spec says "mismatches produce a warning (or error for schema config)" but defines no migration operation.
- **Impact**: A user who adds a new facet (e.g., "MATERIAL" for tracking physical objects) cannot retroactively populate that facet on existing entities without full re-ingestion. A user who refines TYPE_HYPERNYMS cannot propagate the change to already-resolved entities.
- **Spec location**: Section 10.1 (FoundryConfig SCHEMA category), A.5 (config categories), A.13 (schema versioning)
- **Fix**: Add `Foundry.migrate(old_config, new_config)` that performs differential schema migration. For facet additions: re-run Stage 4 (Faceted Entity Extraction) on chunks containing affected entities. For hypernym changes: re-run Stage 6 (Type Inference) on affected entities. For chunk_size changes: full re-chunking required. The manifest.json (A.13) already records the config — migration compares old and new manifests to determine which stages need re-running.

---

### REC-5: Storage Layer Is NOT Uniform Across Index Types

- **Level**: Seven index types (BM25, Vector, Type, Property, Temporal, CoOccurrence, RawChunkStore) each have unique APIs. There is no common trait beyond "accept data at ingestion, return results at query."
- **Expected**: All indexes should implement a common interface: `add(key, data)`, `query(key, top_k) -> list[(id, score)]`, `remove(id)`, `save(path)`, `load(path)`, `__len__()`. This would allow generic operations (e.g., "rebuild all indexes from raw data") without index-specific code.
- **Actual**: Each index has a bespoke API. BM25 uses `add(doc_id, text)` and `query(q, top_k)`. TypeIndex uses `index(entity, type_label)` and `lookup(type_label)`. PropertyIndex uses `index(entity, predicate, value, ptype)` and `lookup(predicate)`. The signatures are all different. `BM25Index.remove()` was only added in B.16 (for entity merge). Most indexes lack `remove()`.
- **Impact**: Adding a new index type requires custom wiring in FoundryStore, custom save/load logic, and custom EntityMerger update logic (B.16 lists 13 data structures). There is no generic "add index to the system" operation. The `FoundryStore` class is a grab-bag of named sub-stores, not a uniform registry.
- **Spec location**: Section 2.3 (7 indexes with different APIs), Section 2.4 (FoundryStore with named sub-stores)
- **Fix**: Define `IndexProtocol` as a runtime-checkable protocol:

```python
class IndexProtocol(Protocol):
    def add(self, key: str, data: Any) -> None: ...
    def query(self, key: str, top_k: int) -> list[tuple[str, float]]: ...
    def remove(self, key: str) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def __len__(self) -> int: ...
```

FoundryStore becomes `indexes: dict[str, IndexProtocol]` instead of 10 named attributes. Each concrete index implements the protocol with appropriate type adapters. EntityMerger iterates `indexes.values()` instead of enumerating 13 data structures by name.

---

### REC-6: Temporal Reasoning Does NOT Compose (Temporal Query on Temporally-Filtered Results)

- **Level**: Temporal operations (`.before()`, `.after()`, `.during()`, `.overlaps()`) filter entities/facts by time. But they cannot be COMPOSED with temporal index operations.
- **Expected**: "Events that happened before event X" should be expressible as: `foundry.search_temporal("event X").before(result.start_dt)`. The output of a temporal query should be usable as input to another temporal query.
- **Actual**: `search_temporal()` returns a `dict` with matched events and an answer string. This is a terminal result, not a composable intermediate. There is no way to chain temporal operations: "Find events overlapping with the user's trip that also happened before the meeting" requires two separate temporal queries and manual intersection.
- **Impact**: Complex temporal questions (27% of the unreachable question space) often require COMPOUND temporal reasoning. The current architecture handles single temporal operations but not temporal composition.
- **Spec location**: Section 2.3.5 (TemporalIndex returns dict), Section 3.2 (QueryPlan temporal methods filter entities, not events)
- **Fix**: Unify temporal query results with the QueryPlan pattern. Add `TemporalQueryPlan` that wraps `TemporalIndex` results and supports chaining:

```python
class TemporalQueryPlan:
    def match(self, query: str) -> "TemporalQueryPlan": ...  # BM25+cosine event retrieval
    def before(self, reference: TemporalEvent | datetime) -> "TemporalQueryPlan": ...
    def after(self, reference: TemporalEvent | datetime) -> "TemporalQueryPlan": ...
    def overlapping(self, interval: TemporalEvent) -> "TemporalQueryPlan": ...
    def execute(self) -> list[TemporalEvent]: ...
    def to_entity_plan(self) -> QueryPlan: ...  # Convert event results to entity filter
```

This allows: `foundry.temporal().match("trip").overlapping(meeting_event).to_entity_plan().type("restaurant").execute()` — "restaurants visited during the trip that overlaps the meeting."

---

### REC-7: Retrieval Is Flat While Storage Is Hierarchical

- **Level**: Storage has three hierarchical layers (Stated→Inferred→Derived). Retrieval treats them as a flat pool.
- **Expected**: The retrieval layer should respect layer hierarchy. A query should be able to specify "search only STATED layer" or "include DERIVED aggregates." The agentic retriever should know which layer a result came from and weight accordingly.
- **Actual**: The `search_memory` tool searches BM25 over raw chunks (which are layer-agnostic — chunks predate extraction). The `search_entities` tool searches entity BM25 over entity documents (constructed from STATED layer facts). The `get_related` tool says it returns "properties across all three layers" but does not annotate which layer each property comes from. The agent has no way to request layer-specific retrieval or to know whether a fact is STATED vs DERIVED.
- **Impact**: The agent may present DERIVED facts (computed aggregates like type counts) as if they were STATED facts (user's actual words). This violates Axiom 5 ("Stated over inferred") at the retrieval level — the axiom is enforced at storage but not propagated to retrieval.
- **Spec location**: Section 5.2 (tool definitions — no layer parameter), Section 7.1 (traverse_layers returns stated entities but does not annotate)
- **Fix**: Add `layer: Optional[str]` parameter to `search_entities` and `get_related` tools. Default is `None` (all layers, current behavior). When specified, results are filtered to the requested layer. Each returned fact/property includes a `source_layer` annotation ("stated", "inferred", "derived"). The answer generation prompt can then distinguish "the user said X" (STATED) from "the system computed X" (DERIVED).

---

### REC-8: Grounding Verification Is Applied at Ingestion But NOT at Retrieval/Output

- **Level**: Grounding verification (Section 4.3.9) runs at ingestion time. There is no equivalent check at retrieval or answer-generation time.
- **Expected**: Grounding should be a through-line. At ingestion: fact.object must appear in source_text. At retrieval: retrieved facts should be verifiable against the chunk that produced them. At answer generation: the generated answer should be grounded in the retrieved evidence.
- **Actual**: After ingestion, the `source_text` field is stored on each Fact but never used again at query time. The answer generation step (which is outside the spec — handled by the calling LLM) has no mechanism to verify that its answer is grounded in the retrieved evidence. A hallucinated answer that contradicts the retrieved facts is indistinguishable from a correct one.
- **Impact**: The grounding invariant (I5, Section 9.1) protects storage integrity but not output integrity. The LLM can hallucinate an answer that contradicts every retrieved fact, and the system has no post-generation grounding check.
- **Spec location**: Section 4.3.9 (ingestion-time grounding only), Section 9.1 (I5 grounding invariant covers storage only)
- **Fix**: Add `verify_answer_grounding(answer: str, evidence: list[str]) -> float` as a post-generation check. This computes the fraction of answer tokens that appear in the evidence. If the grounding score is below a threshold (e.g., 0.3), the answer is flagged as potentially hallucinated. This does NOT require an LLM call — it is the same substring/token-overlap check as ingestion-time grounding, applied to the output.

---

## Requisite Variety (Ω)

### RV-1: Query Parser/Router vs Question Types

- **Controller**: DeterministicRouter (Section 5.6) — regex-based classification
- **V(R)**: 5 pipelines: fact_lookup, temporal, aggregation, preference, comparison. Plus "agentic" fallback.
- **Disturbance**: 13 question types (Section 8.1): single_hop_factual, multi_hop_bridging, temporal, counting_aggregation, comparison, preference, opinion_sentiment, spatial, causal, hypothetical, procedural, social_relational, meta_cognitive.
- **V(R)/V(D)**: 5/13 = 0.38 (or 6/13 = 0.46 counting the agentic fallback as one response)
- **Deficit**: 8 question types have no dedicated deterministic pipeline. They all collapse into the "agentic" fallback. The router has LESS variety than the disturbance space by a factor of 2.2x. The agentic fallback is a single catch-all response for 8 distinct question types.
- **Impact**: The 8 question types routed to the agent (opinion_sentiment, spatial, causal, hypothetical, procedural, social_relational, meta_cognitive, multi_hop_bridging) are handled by the SAME generic ReAct loop. But they have fundamentally different structures: a causal question needs temporal chain reasoning, a hypothetical question needs counterfactual reasoning, a social question needs relationship graph traversal. The agent must re-discover these structures on every call.
- **Spec location**: Section 5.6 (5 pipelines), Section 8.1 (13 types)
- **Fix**: Add at least 4 more deterministic pipelines to reach V(R)/V(D) >= 0.7:
  - `spatial`: entity_lookup → facts(facet='where') → format
  - `social_relational`: entity_lookup(person) → related(max_hops=2) → facts(facet='who') → format  
  - `meta_cognitive`: BM25(entity_name) → existence check → "yes/no + source turn"
  - `multi_hop`: entity_lookup → related(max_hops=3, per_hop_cr=True) → collect_evidence

  This raises V(R) to 9/13 = 0.69. The remaining 4 types (causal, hypothetical, procedural, opinion_sentiment) are genuinely hard — they require LLM reasoning, not just retrieval routing, and legitimately need the agentic fallback.

---

### RV-2: Entity Resolution vs Naming Patterns

- **Controller**: EntityResolver cascade — 6 steps (alias, determiner, possessive, substring, embedding, create)
- **V(R)**: 6 resolution strategies
- **Disturbance**: User naming patterns: exact name, nickname, "that place", "Roscioli", "the restaurant near the Vatican", "where we ate", pronoun ("it", "she"), temporal reference ("the old one"), possessive chain ("Sarah's brother's dog"), abbreviation ("Dr. → Doctor"), misspelling ("restraunt"), code-switching ("la trattoria"), implicit reference (never named, only described)
- **V(D)**: At least 13 distinct naming patterns
- **V(R)/V(D)**: 6/13 = 0.46
- **Deficit**: 7 naming patterns have no resolution strategy:
  1. **Pronoun resolution** ("it", "she") — requires coreference resolution across turns, not in the cascade
  2. **Descriptive reference** ("the restaurant near the Vatican") — requires property-based lookup, not name-based
  3. **Temporal reference** ("the old one") — requires supersession history
  4. **Possessive chains** ("Sarah's brother's dog") — requires multi-hop traversal during resolution
  5. **Code-switching** ("la trattoria" for "the restaurant") — requires multilingual alias
  6. **Implicit references** (never named entities) — requires inference from context
  7. **Misspelling** with >2 edit distance — "resteraunt" vs "restaurant" exceeds fuzzy threshold
- **Spec location**: Section 3.4 (6-step cascade), B.11 (Jaccard overlap addition)
- **Fix**: For v1, accept the variety deficit for patterns 2-6 (descriptive, temporal, possessive, code-switch, implicit) — these collapse into the agentic retriever where the LLM handles them naturally. For pronoun resolution (pattern 1), add a lightweight coreference step: track the most-recently-mentioned entity per type category. "It" resolves to the most recent entity of the inferred type. For misspelling (pattern 7), the B.11 Jaccard overlap addition partially addresses this, and `$fuzzy` filter (Section 3.2a) covers query-time spelling correction.

---

### RV-3: Conflict Resolution vs Contradiction Types

- **Controller**: SupersessionEngine — structural (S,P) matching + SUPERSESSION_GROUPS
- **V(R)**: 2 resolution strategies: same-predicate supersession (latest turn wins) and group-based cross-predicate supersession (B.12)
- **Disturbance**: Contradiction types that arise in conversational memory:
  1. **Temporal supersession**: "I live in SF" → "I moved to Portland" (same predicate, different value)
  2. **Semantic contradiction**: "I'm allergic to peanuts" → "I ate peanut butter" (different predicates, semantic conflict)
  3. **Count change**: "I have 2 cats" → "I adopted another" (implicit increment, not explicit replacement)
  4. **Preference reversal**: "I love sushi" → "I've gone off sushi" (same entity, different sentiment)
  5. **Correction**: "Actually, it was Portland, not Seattle" (explicit retraction + replacement)
  6. **Retraction without replacement**: "Never mind, forget what I said about the car" (deletion, not supersession)
  7. **Partial update**: "The car is still red but I changed the wheels" (one property changes, others preserved)
  8. **Graduated change**: "I'm getting better at piano" (progressive, not discrete supersession)
- **V(D)**: 8 contradiction types
- **V(R)/V(D)**: 2/8 = 0.25
- **Deficit**: Types 2-8 are not handled by structural supersession. Type 2 (semantic) is acknowledged as a known limitation (B.12 "structural supersession only"). Types 3 (count change), 5 (correction), and 6 (retraction) are particularly dangerous because they LOOK like new facts but are actually modifications of existing facts.
- **Spec location**: Section 6.2 (SupersessionEngine), B.12 (known limitation)
- **Fix**: This deficit is partially by design — Axiom 2 ("no LLM for infrastructure") prohibits LLM-based semantic contradiction detection. For v1, accept the deficit and document it. Add these specific mitigations:
  - **Count change (type 3)**: During extraction, classify "adopted another" as `sub_role=acquisition` with implied increment. The DerivedLayerBuilder already handles counting with ownership filters.
  - **Explicit correction (type 5)**: Add regex detection for correction markers ("actually", "I meant", "not X, Y", "correction:") during triple extraction. When detected, the extracted fact receives `supersession_hint=True` and actively searches for the fact being corrected.
  - **Retraction (type 6)**: Add a `retract` action type in Section 6.6 (MemoryAction). When the user says "forget X" or "never mind about X", mark matching facts as soft-deleted via Tombstone.

---

### RV-4: Retrieval Tools vs Information Needs

- **Controller**: 8 tools (search_memory, search_entities, get_related, search_temporal, read_turn, aggregate, execute_python, answer)
- **V(R)**: 8 tools
- **Disturbance**: Information needs from the 13 question types:
  1. Keyword retrieval (BM25) — search_memory
  2. Entity lookup — search_entities
  3. Relationship traversal — get_related
  4. Temporal reasoning — search_temporal
  5. Direct text access — read_turn
  6. Counting/aggregation — aggregate
  7. Date arithmetic — execute_python
  8. Final answer — answer
  9. **MISSING: Comparison** — "Which is more expensive, A or B?" needs structured property comparison
  10. **MISSING: List enumeration** — "List all restaurants" needs type-filtered enumeration
  11. **MISSING: Existence check** — "Has the user mentioned X?" needs boolean probe
  12. **MISSING: Timeline query** — "Show me the history of X" needs temporal fact sequence
  13. **MISSING: Conversation reference** — "What did we discuss last time?" needs session-level retrieval
- **V(D)**: 13 information needs
- **V(R)/V(D)**: 8/13 = 0.62
- **Deficit**: 5 information needs have no dedicated tool. They can all be approximated by combining existing tools (e.g., comparison = search_entities(A) + search_entities(B) + execute_python(compare)), but this requires 3+ tool calls where a single dedicated tool would suffice.
- **Spec location**: Section 5.2 (8 tool definitions)
- **Fix**: Given the tool count cap (C.10: max 12 tools before degradation), add the 2 highest-impact missing tools:
  - `compare_entities(entity_a: str, entity_b: str, property: str)` — retrieves and compares a specific property across two entities. Returns structured comparison. Eliminates a common 3-step agent pattern.
  - `list_entities(type: str, max: int = 20)` — returns entities of a given type with key facts. Eliminates the aggregate+search_entities two-step pattern for enumeration questions.

  This raises V(R) to 10/13 = 0.77. The remaining 3 (existence, timeline, conversation) can be addressed by extending existing tools with optional parameters.

---

### RV-5: Decay Model vs Memory Dynamics

- **Controller**: DecayFunction — exponential decay with Hebbian reinforcement
- **V(R)**: 2 mechanisms: time-based decay (days since access) and access-based reinforcement (retrieval resets salience)
- **Disturbance**: Ways memory should fade or strengthen:
  1. **Time decay** — old memories fade → handled by exponential decay
  2. **Access reinforcement** — retrieved memories strengthen → handled by Hebbian reset
  3. **Supersession acceleration** — replaced facts should fade faster → handled (10x lambda)
  4. **MISSING: Relevance decay** — memories irrelevant to current topics should fade faster than relevant ones
  5. **MISSING: User-initiated deletion** — "forget about X" should immediately remove, not gradually decay
  6. **MISSING: Importance-gated decay** — high-importance facts (health, allergies) should decay slower
  7. **MISSING: Emotional salience** — emotionally charged memories are more persistent
  8. **MISSING: Contextual reinforcement** — memories related to currently-active topics should be boosted
- **V(D)**: 8 memory dynamics
- **V(R)/V(D)**: 3/8 = 0.375
- **Deficit**: 5 dynamics have no controller response. Notably, user-initiated deletion (type 5) maps to the existing Tombstone/DeletionPolicy mechanism (Section 6.5) but is not wired to the decay system. Importance-gated decay (type 6) is partially addressed by ImportanceScore (Section 7.2) but the decay function does not consume the importance score.
- **Spec location**: Section 7.3 (DecayFunction), Section 7.2 (ImportanceScore — computed but unused by decay)
- **Fix**: Wire ImportanceScore into DecayFunction:

```python
def compute(self, days_since_access: float, importance: ImportanceScore,
            base_salience: float = 1.0, superseded: bool = False) -> float:
    # Importance-gated lambda: high-importance entities decay slower
    effective_lambda = self.lambda_base / (1.0 + importance.composite)
    if superseded:
        effective_lambda *= 10
    return base_salience * exp(-effective_lambda * days_since_access)
```

  This makes high-importance entities (health, allergies — high mention_frequency and retrieval_frequency) decay 2-5x slower than low-importance entities. User-initiated deletion should bypass decay entirely and trigger immediate soft-delete via Tombstone.

---

### RV-6: Ingestion Extractor vs Conversation Patterns

- **Controller**: LLM extraction pipeline (Stage 3: triples, Stage 4: faceted entities)
- **V(R)**: 1 extraction strategy: prompt-based structured output from a single LLM call per chunk
- **Disturbance**: Conversation patterns that affect extraction quality:
  1. **Direct statements** ("I have 2 cats") — handled well
  2. **Multi-turn corrections** ("Wait, I said 2 but it's actually 3") — requires cross-chunk context
  3. **Sarcasm** ("Oh sure, I just LOVE getting up at 5am") — requires sentiment detection
  4. **Lists** ("I've been to Rome, Paris, Tokyo, and...") — handled by LIST PropertyType
  5. **Narratives** ("So first we went to... then we...") — requires temporal chain extraction
  6. **Code-switching** ("I went to la trattoria") — requires multilingual entity detection
  7. **Implicit information** ("I can't eat that" [allergic]) — requires world knowledge inference
  8. **Assistant turns** (system-generated text) — different extraction rules than user turns
  9. **Hypotheticals** ("If I were to get a dog...") — must not create dog entity
  10. **Reported speech** ("Sarah said she's moving") — attribute to Sarah, not user
- **V(D)**: 10 conversation patterns
- **V(R)/V(D)**: 1/10 = 0.10
- **Deficit**: The extraction pipeline uses a single prompt template for all conversation patterns. Sarcasm (type 3), implicit information (type 7), and reported speech (type 10) are particularly problematic — they require different extraction rules that a single prompt cannot capture. The VerbNet sub_role classification (Section 4.3.10) partially addresses hypotheticals (type 9) but not the others.
- **Spec location**: Section 4.3.3 (extract_triples — single prompt), Section 4.3.4 (extract_faceted_entities — single prompt)
- **Fix**: This is partially by design — Axiom 2 limits infrastructure complexity. The extraction prompt is the single lever. For v1, enhance the extraction prompt with explicit instructions for each pattern:
  - Add "Do NOT extract entities from hypothetical or conditional statements" (pattern 9)
  - Add "When the user corrects themselves, mark the corrected fact with `correction: true`" (pattern 2)
  - Add "Attribute reported speech to the speaker, not the user" (pattern 10)
  - Add "When the user expresses sarcasm (indicated by emphasis or contradictory sentiment), tag with `sarcasm: true`" (pattern 3)
  
  This does not increase LLM call count. It increases prompt specificity for known failure patterns.

---

## Good Regulator (μ)

### GR-1: Query Planner Has NO Model of Storage Capabilities

- **Controller**: QueryPlan executor (Section 3.2d) + DeterministicRouter (Section 5.6)
- **Model status**: ABSENT
- **Abstraction**: N/A — no model exists
- **What fails**: The QueryPlan executor pushes filters to indexes (B.10) but has no way to know what TYPES exist in the TypeIndex, what PROPERTIES exist in the PropertyIndex, or how many ENTITIES exist in the graph. A query for `.type("restaurant")` executes regardless of whether any restaurant entities exist. The DeterministicRouter classifies questions by pattern but cannot check whether the target data exists before routing.
- **Impact**: Wasted computation on empty queries. The `aggregate` tool counts entities of a type — but if the type name is slightly wrong ("restaurants" vs "restaurant"), it returns 0 without knowing that a depluralized form would succeed. The router routes to `counting_aggregation` pipeline, which returns 0, which triggers agentic fallback — all because the router has no model of what types actually exist.
- **Spec location**: Section 3.2d (execution strategy — no pre-check), Section 5.6 (router — no storage awareness)
- **Fix**: Add `StorageManifest` — a lightweight model of storage contents:

```python
@dataclass
class StorageManifest:
    """Lightweight model of what the storage layer contains.
    Updated after each ingestion. Queried by the router and executor."""
    entity_types: set[str]           # All type labels in TypeIndex
    property_predicates: set[str]    # All predicates in PropertyIndex
    entity_count: int
    fact_count: int
    temporal_event_count: int
    type_counts: dict[str, int]      # type -> count
    last_updated: datetime
```

  The DeterministicRouter consults StorageManifest before routing: if `counting_aggregation` pipeline targets a type not in `entity_types`, it tries depluralization, then synonym expansion, before falling through to agentic. The QueryPlan executor uses `type_counts` to estimate selectivity for query optimization.

---

### GR-2: Agentic Retriever Has NO Model of the Knowledge Base

- **Controller**: AgenticRetriever (Section 5.5) — ReAct loop with LLM
- **Model status**: ABSENT
- **Abstraction**: N/A — no model exists
- **What fails**: The agent receives a question and a tool list. It has no knowledge of what entities exist, what types are populated, or how many chunks are stored. Each tool call is a blind probe — the agent cannot estimate whether `search_entities("parrot")` will return results or `search_temporal("trip to Japan")` will find temporal events.
- **Impact**: The agent wastes steps on blind probes. Trace analysis shows the agent frequently calls `search_entities` for entities that do not exist, then `search_memory` for the same query (redundant), then `get_related` on a zero-result entity (error). If the agent knew "parrot exists as entity, has 3 facts, mentioned in turns 45,67,89," it could skip directly to evidence collection.
- **Spec location**: Section 5.5 (AgenticRetriever — system prompt includes tool descriptions but no storage summary)
- **Fix**: Inject a `knowledge_summary` into the agent's system prompt at each invocation:

```python
KNOWLEDGE_SUMMARY_TEMPLATE = """
Available memory: {entity_count} entities, {chunk_count} chunks, {temporal_count} temporal events.
Top entity types: {top_types}
Recently mentioned entities: {recent_entities}
"""
```

  This is ~100 tokens of additional context per invocation. The knowledge summary is computed from StorageManifest (GR-1) and costs zero LLM calls. It allows the agent to make informed tool selection: "I know there are 15 restaurant entities, so `aggregate('restaurant', 'count')` will work" instead of blind probing.

---

### GR-3: Ingestion Pipeline Has a PARTIAL Model of Existing Entities (Alias Index Only)

- **Controller**: Ingestion pipeline Stage 5 (Entity Resolution)
- **Model status**: PARTIAL — the alias index exists, but the pipeline has no broader model
- **Abstraction**: TOO LOW — alias index provides name-matching but not semantic or structural awareness
- **What fails**: The entity resolution cascade can match mentions to existing entities via aliases. But the extraction stages (3 and 4) operate WITHOUT any model of existing entities. The LLM extracting triples from chunk N does not know what entities already exist from chunks 1 to N-1. It may extract "Dr. Smith" as a new entity even though "doctor_smith" already exists — the resolution happens AFTER extraction, not during.
- **Impact**: Extraction quality degrades when the LLM could benefit from knowing the existing entity inventory. If the LLM prompt included "Known entities: doctor_smith, mercy_clinic, fluffy_the_cat," the extraction would be more consistent. Without this, the same entity is extracted with different surface forms across chunks, creating unnecessary resolution work.
- **Spec location**: Section 4.3.3 (extract_triples — no context about existing entities), Section 4.3.4 (extract_faceted_entities — same)
- **Fix**: Pass a condensed entity list to the extraction prompt. This is already partially implemented as `entity_registry_cap=50` (Section 4.3.3). Extend it: instead of a cap on output entities, provide the TOP-50 existing entities (by mention frequency) as input context to the extraction prompt. This is the same pattern used in few-shot prompting — provide examples of known entities so the LLM uses consistent naming.

---

### GR-4: Conflict Resolver Has NO Model of Fact Reliability

- **Controller**: SupersessionEngine (Section 6.2)
- **Model status**: ABSENT
- **Abstraction**: N/A
- **What fails**: The supersession engine uses `latest_wins` — the most recent turn always wins. But not all sources are equally reliable. A user's direct statement ("I have 2 cats") is more reliable than an assistant's inference ("Based on our conversation, you seem to have 2 cats"). A correction ("Actually, 3 cats") is more reliable than the original statement. The engine treats all turns as equally authoritative.
- **Impact**: When an assistant turn re-states a user fact incorrectly, the assistant's version supersedes the user's original. The `user_wins` strategy (Section 6.2) exists but is not the default. More subtly, explicit corrections should have higher authority than casual mentions — "Actually, I have 3" should supersede more confidently than "I think I have 3."
- **Spec location**: Section 6.2 (strategies: latest_wins, user_wins, per_hop — no reliability model)
- **Fix**: Add a `reliability_score` to each Fact, computed deterministically at ingestion:

```python
def compute_reliability(fact: Fact) -> float:
    score = 0.5  # Base
    if fact.role == "user": score += 0.3      # User's own words
    if "actually" in fact.source_text.lower(): score += 0.1  # Correction marker
    if "I think" in fact.source_text.lower(): score -= 0.1   # Hedging
    if fact.sub_role == "hypothetical": score -= 0.3  # Hypothetical
    return min(1.0, max(0.0, score))
```

  The SupersessionEngine consults reliability: a newer fact supersedes an older one ONLY IF `new.reliability >= old.reliability * 0.8`. This prevents low-confidence assistant inferences from overwriting high-confidence user statements.

---

### GR-5: Decay Model Has NO Model of User Behavior

- **Controller**: DecayFunction (Section 7.3)
- **Model status**: ABSENT
- **Abstraction**: N/A
- **What fails**: The decay function uses fixed parameters (lambda_days=0.01, lambda_turns=0.001) for all users. But users have different conversation patterns: a daily user's "30 days without access" means something different than a monthly user's "30 days." A daily user who has not mentioned their cat in 30 days is likely losing interest. A monthly user who has not mentioned their cat in 30 days simply has not had a session yet.
- **Impact**: For infrequent users, the decay function prematurely marks entities as GC-eligible. An entity accessed 90 days ago has salience 0.41 (non-superseded). For a daily user who queries 10x/day, this is 900 sessions without mention — genuinely fading. For a monthly user, this is 3 sessions — still very current.
- **Spec location**: Section 7.3 (fixed lambda), B.17 (dual-axis decay — still fixed parameters)
- **Fix**: Scale decay parameters by user session frequency:

```python
def adjusted_lambda(base_lambda: float, avg_sessions_per_day: float) -> float:
    """Slower decay for infrequent users."""
    # If user averages 0.1 sessions/day (monthly), lambda is 10x slower
    # If user averages 10 sessions/day (power user), lambda is standard
    frequency_factor = max(0.1, min(10.0, avg_sessions_per_day))
    return base_lambda * frequency_factor
```

  Track `avg_sessions_per_day` in FoundryConfig (RUNTIME category). Update it after each session. This simple adaptation makes the decay model responsive to individual usage patterns without any LLM call.

---

### GR-6: Temporal Engine Has NO Model of Time Precision

- **Controller**: TemporalIndex (Section 2.3.5)
- **Model status**: ABSENT
- **Abstraction**: N/A
- **What fails**: All temporal events are stored as `[start_dt, end_dt]` datetime intervals. But "March 2024" has DAY precision (any day in the month), while "March 15, 2024" has DAY precision (specific day), and "3pm on March 15" has HOUR precision. The temporal index treats them all as identical-precision intervals.
- **Impact**: "March 2024" is stored as `[2024-03-01, 2024-03-31]`. "March 15, 2024" is stored as `[2024-03-15, 2024-03-15]`. An Allen `contains` check says March 2024 `contains` March 15 — correct. But a `distance` computation says "0 days between March 2024 and March 15" because it compares start_dt to start_dt (Mar 1 vs Mar 15 = 14 days) — the result depends on which interval endpoint is used. The engine cannot know that "March 2024" means "any day in March" while "March 15" means exactly that day.
- **Spec location**: Section 2.3.5 (TemporalIndex — no precision field), Section 2.2 (TemporalEvent — no precision)
- **Fix**: Add `precision: TemporalPrecision` to TemporalEvent:

```python
class TemporalPrecision(str, Enum):
    YEAR = "year"       # "2024" → [2024-01-01, 2024-12-31]
    MONTH = "month"     # "March 2024" → [2024-03-01, 2024-03-31]
    DAY = "day"         # "March 15" → [2024-03-15, 2024-03-15]
    HOUR = "hour"       # "3pm March 15" → [2024-03-15T15:00, 2024-03-15T15:59]
    EXACT = "exact"     # Precise timestamp from system clock
```

  Distance computations account for precision: `distance(month_event, day_event)` reports the range of possible distances, not a single number. "Between 0 and 30 days" is more honest than "14 days" for month-precision events.

---

## Ultrastability (∾)

### US-1: No Per-Session Adaptation Loop

- **Loop speed**: Medium (per-session)
- **Current state**: ABSENT
- **Expected**: During a conversation session, the system should improve its retrieval based on accumulated context. If the user asks about "Sarah" in turns 1, 5, and 10, the system should learn that Sarah is the focal entity and prioritize Sarah-related retrievals.
- **Actual**: Each retrieval call within a session is INDEPENDENT. The AgenticRetriever resets `DeduplicationTracker` between questions (Section 5.4: "Reset at the start of each new question"). But it does not accumulate session-level context — each question starts from scratch with the same tools, the same system prompt, and no memory of which entities were relevant to earlier questions in the session.
- **Novel condition**: A session where every question is about the same topic (e.g., planning a trip to Rome). The system re-discovers "Rome" as relevant on every question instead of maintaining it as context.
- **Fix**: Add a `SessionContext` object that persists across questions within a single session:

```python
@dataclass
class SessionContext:
    focal_entities: list[str]  # Entities mentioned in >2 questions
    topic_terms: set[str]      # Recurring query terms
    successful_tools: dict[str, int]  # Tool -> success count (for tool selection priming)
```

  The SessionContext is injected into the agent's system prompt. After each question, update the context with the entities and tools that produced useful evidence. This does not require an LLM call — it is deterministic context accumulation.

---

### US-2: No Cross-Session Learning Loop (Beyond Trace Collection)

- **Loop speed**: Slow (cross-session)
- **Current state**: PARTIAL — traces are saved (Section 5.5) but no learning mechanism exists in v1
- **Expected**: The system should adapt its behavior across sessions. If a particular retrieval pattern is effective for a user, the deterministic router should learn to apply it directly. If certain entity types are frequently queried, the derived layer should pre-compute relevant aggregates.
- **Actual**: Trace collection exists (RetrievalTrace in Section 5.5). The HebbianBridge (Section 7.4) is specified but NOT implemented in v1. DeterministicRouter pipelines are hardcoded from trace ANALYSIS (a manual process), not from automated trace-to-pipeline conversion.
- **Novel condition**: A user whose question patterns shift over time (e.g., from "what restaurants?" to "what medications?" after a health change). The deterministic pipelines, trained on old traces, become stale.
- **Fix**: Implement a minimal cross-session adaptation mechanism for v1:

```python
class PipelinePerformanceTracker:
    """Tracks which deterministic pipelines succeed and which fall through to agentic."""
    def record(self, pipeline: str, question: str, succeeded: bool) -> None: ...
    def get_accuracy(self, pipeline: str, window: int = 100) -> float: ...
```

  When a pipeline's accuracy drops below 60% over the last 100 questions, it is automatically disabled (all its questions go to agentic). When agentic traces show a new common pattern (>10 identical tool sequences), a new deterministic pipeline is automatically generated. This closes the adaptation loop without requiring model fine-tuning.

---

### US-3: System Has No Graceful Degradation Under Novel Entity Types

- **Loop speed**: Fast (per-query)
- **Current state**: ABSENT — novel conditions produce silent degradation
- **Expected**: When the system encounters a completely new entity type (e.g., user starts discussing cryptocurrency wallets, having never mentioned finance before), it should recognize the novelty and adapt its extraction/retrieval strategy.
- **Actual**: Novel entity types flow through the standard pipeline. The TypeIndex has no entry for "wallet." The extraction model creates a wallet entity with whatever facets it can populate. The DeterministicRouter has no special handling. The system functions but may miscategorize the entity (e.g., "wallet" → what="accessory" instead of "financial_tool") and the derived layer's aggregates may be wrong.
- **Impact**: Silent degradation is worse than explicit failure. A wrong type classification leads to wrong aggregation (counting wallets as accessories), wrong traversal (linking to fashion entities instead of financial entities), and wrong temporal reasoning (no relevant temporal patterns for new domains).
- **Spec location**: Section 1.4 (EntityType — no novelty detection), Section 4.3.6 (type inference — no confidence score)
- **Fix**: Add novelty detection to type inference:

```python
def infer_type_with_confidence(entity: FacetEntity, type_index: TypeIndex) -> tuple[str, float]:
    """Returns (type_label, confidence).
    
    Confidence is based on whether the type has been seen before:
    - Existing type (>5 instances): confidence = 1.0
    - Existing type (1-5 instances): confidence = 0.7
    - Novel type (0 instances): confidence = 0.3
    """
    type_label = EntityType.normalize(entity.facets.get("what", ""))
    existing_count = type_index.count(type_label)
    confidence = 1.0 if existing_count > 5 else (0.7 if existing_count > 0 else 0.3)
    return type_label, confidence
```

  Entities with confidence < 0.5 are flagged for the agentic retriever: when a query targets a low-confidence-type entity, the system should use BM25 (which does not depend on type classification) rather than entity-based retrieval.

---

### US-4: Missing Fast Loop — No Within-Query Adaptation

- **Loop speed**: Fast (per-query, within a single retrieval operation)
- **Current state**: PARTIAL — the agentic retriever adapts within a query via its ReAct loop, but deterministic pipelines do not
- **Expected**: Even deterministic pipelines should adapt within a single query execution. If the first retrieval step returns no results, the pipeline should try alternative strategies (query expansion, type relaxation, broader BM25 search) before falling through to the agent.
- **Actual**: Deterministic pipelines (Section 5.6) are rigid sequences. The `fact_lookup` pipeline runs BM25 and returns results. If BM25 returns nothing relevant, the pipeline returns empty results, self-validation fails (C.7), and the entire query falls through to the 6-10 second agentic path. There is no intermediate adaptation step.
- **Novel condition**: A query where the answer exists but is phrased differently than the question. BM25 finds no exact match. A synonym expansion or fuzzy search would find it. But the deterministic pipeline does not try these before falling through.
- **Fix**: Add a 2-level fallback within deterministic pipelines:

```python
class FactLookupPipeline:
    def _run(self, question: str) -> RetrievalResult:
        # Level 1: exact BM25
        results = self.bm25.query(question, top_k=10)
        if self._sufficient(results):
            return self._format(results)
        
        # Level 2: query expansion (augment_query from Section 5.8)
        expanded = augment_query(question, self.resolver, self.alias_index)
        results = self.bm25.query(expanded, top_k=10)
        if self._sufficient(results):
            return self._format(results)
        
        # Level 3: fall through to agentic
        return None
```

  This adds ~10ms of overhead (one alias expansion + one BM25 query) but catches the common case where a synonym or alias expansion would have found the answer. The agentic fallback is reserved for genuinely hard cases.

---

### US-5: No Degradation Signal — System Cannot Distinguish "No Answer" from "Bad Answer"

- **Loop speed**: All loops
- **Current state**: ABSENT
- **Expected**: When the system cannot find an answer, it should return a calibrated "I don't know" rather than a low-confidence guess. When it finds an answer, it should report confidence.
- **Actual**: The retrieval layer returns `evidence_text` (a string) and a `RetrievalTrace`. There is no confidence score on the evidence. The answer generation LLM receives the evidence and generates an answer — but neither the retrieval layer nor the answer layer reports whether the evidence is sufficient. The system always produces an answer, even when the evidence is empty or contradictory.
- **Impact**: The caller (LLM) has no signal to distinguish "high-confidence answer from 5 relevant chunks" from "guess based on 0 relevant chunks." The system degrades silently — wrong answers look identical to right answers at the API level.
- **Spec location**: Section 10.4 (retrieve returns `(str, RetrievalTrace)` — no confidence), Section 5.5 (RetrievalTrace has `terminated_by` but no evidence quality score)
- **Fix**: Add `evidence_confidence: float` to the retrieval result:

```python
@dataclass
class RetrievalResult:
    evidence: str
    trace: RetrievalTrace
    confidence: float  # 0.0 = no relevant evidence, 1.0 = high-quality evidence
    
    @staticmethod
    def compute_confidence(trace: RetrievalTrace) -> float:
        if trace.terminated_by == "answer":
            # Agent chose to answer — use step count as proxy
            return max(0.3, 1.0 - (len(trace.steps) / 8.0) * 0.5)
        elif trace.terminated_by == "budget":
            return 0.1  # Agent exhausted budget without answering
        else:
            return 0.0  # Error or no action
```

  When confidence < 0.3, the caller should be advised to respond with "I don't have enough information" rather than generating a potentially hallucinated answer.

---

## Always-On Guards

### Entry Check

1. **Purpose alignment**: All 25 findings serve the stated purpose (personal memory for conversational AI). No finding introduces capabilities outside this scope.
2. **Boundary integrity**: REC-5 (non-uniform storage API) and REC-3 (non-uniform ingestion paths) are boundary integrity violations. Each index and each entry point should have the same interface contract.
3. **Information flow**: REC-7 (flat retrieval vs hierarchical storage) is an information flow violation — layer hierarchy information is lost at the storage-retrieval boundary.
4. **Error propagation**: US-5 (no degradation signal) means errors in retrieval propagate silently to answer generation with no containment.

### Exit Check

1. **Purpose alignment**: MAINTAINED. All fixes are implementable within the personal memory scope.
2. **Boundary integrity**: IMPROVED by REC-5 (IndexProtocol) and REC-3 (unified pipeline entry).
3. **Information flow**: IMPROVED by REC-7 (layer annotations in retrieval) and GR-1 (StorageManifest).
4. **Error propagation**: IMPROVED by US-5 (confidence score) and US-3 (novelty detection).

---

## Viability Scorecard (VSM S1-S5)

| System | Role | Foundry Component | Score (0-2) | Gaps |
|--------|------|-------------------|-------------|------|
| **S1: Operations** | Day-to-day work | Storage + Query + Retrieval + Ingestion | **2** | Fully specified. 7 indexes, 8 tools, 11-stage pipeline. Functional if built as specified. |
| **S2: Coordination** | Resolve conflicts between S1 units | Entity Resolution + Conflict Resolution + RRF merge | **1** | Entity resolution exists but cannot handle homonyms (ET-4). Conflict resolution limited to structural supersession (RV-3). RRF coordinates retrieval channels but is static. |
| **S3: Control** | Resource allocation, audit | Memory Dynamics (decay, importance, capacity gating) | **1** | Decay exists but has no model of user behavior (GR-5). Importance is computed but not consumed by decay (RV-5). Capacity gating exists (Lean proof Phase 3) but eviction policy is underdefined (ET-5). |
| **S3*: Audit** | Sporadic deep investigation | **MISSING** | **0** | No component performs audit. There is no mechanism to detect silent corruption (e.g., entity graph inconsistency, grounding invariant violations post-ingestion, alias index desync). The `grounding_report()` method (Section 10.6) is the closest analog, but it is a manual call, not an automatic audit. |
| **S4: Intelligence** | Future-oriented, adaptation | HebbianBridge (Section 7.4) + Trace collection (Section 5.5) | **0.5** | Specified but NOT IMPLEMENTED in v1. HebbianBridge is explicitly marked "NOT implemented in Foundry v1." Trace collection is the only implemented S4 component, but traces are collected without automated analysis. The system cannot adapt to new question patterns, new entity types, or changing user behavior without manual intervention. |
| **S5: Policy** | Identity, purpose, values | Design Axioms (7) + Behavioral Invariants (8) + Formal Properties (5) | **2** | Well-specified. The 7 axioms define system identity. The invariants define behavioral boundaries. The Lean proofs formalize critical properties. |

**Viability Score: 6.5 / 10**

**Critical gaps:**
- **S3* (Audit) = 0**: The system has NO automatic integrity verification. A corrupted alias index, a grounding invariant violation, or a desync between TypeIndex and EntityGraph would be invisible until a user query returns wrong results.
- **S4 (Intelligence) = 0.5**: The system cannot adapt to its environment. It processes what it receives but does not learn from its own performance. The trace collection mechanism is the seed of S4, but without automated analysis, the traces are inert data.

---

## Summary

- **Total findings**: 25 (8 Recursion, 6 Requisite Variety, 6 Good Regulator, 5 Ultrastability)
- **STRUCTURAL gaps Brenner missed**:
  1. S3* Audit subsystem entirely absent (no automatic integrity verification)
  2. S4 Intelligence subsystem absent in v1 (traces collected but never analyzed)
  3. No self-model anywhere in the system (GR-1, GR-2 — controllers operate blind)
  4. Conversation is not an entity (REC-1 — structural gap in the entity model)
  5. No output grounding (REC-8 — grounding stops at ingestion, does not reach output)
- **Missing subsystems**: S3* Audit, S4 Intelligence (in v1), SessionContext, StorageManifest, PipelinePerformanceTracker, TemporalQueryPlan
- **Viability score**: 6.5/10 (S1 and S5 are strong; S2, S3 are partial; S3* and S4 are absent)

The system as specified is a competent S1 (operational) system with good S5 (policy/identity). It lacks the meta-level systems (audit, intelligence, adaptation) that would make it self-maintaining and self-improving. These gaps are appropriate for v1 — the system works — but they must be addressed for production use where the system must operate without continuous human oversight.
