# Prior Art Synthesis: Design Decisions for Hermes Foundry

Generated: 2026-04-13
Input: Deep dives on Graphiti, Django QuerySet, Gremlin, KAG, Hindsight

---

## 1. Executive Summary

Five systems were analyzed in depth. Each contributes a distinct architectural pattern to the Hermes Foundry design. No existing system combines all of them. The synthesis below maps each pattern to a concrete Hermes design decision with adopt/reject rationale.

| System | Primary Contribution to Hermes |
|--------|-------------------------------|
| **Django QuerySet** | Clone-on-write lazy query composition (`_chain()`, terminal ops, Q objects) |
| **Graphiti/Zep** | Bi-temporal edge model, episode lineage, edge invalidation over deletion |
| **Gremlin/TinkerPop** | Step type taxonomy, TraversalStrategy optimization, Traverser carrier object |
| **KAG/OpenSPG** | Three-layer knowledge cascade, DAG-based query plans, mutual indexing |
| **Hindsight** | Four-network epistemic separation, 4-way parallel retrieval, reflect operation |

---

## 2. Merged Architecture: What Hermes Adopts

### 2.1 Query Composition Model: Django + Gremlin Hybrid

**Decision: Clone-on-write immutable QuerySet (from Django) with typed step nodes (from Gremlin).**

Rationale:
- Django's `_chain()` pattern is the gold standard for lazy, composable, immutable query building in Python. Every method returns a new object. Original never mutated. This enables forking queries into variants (essential for set operations).
- Gremlin's five step types (Map, Filter, FlatMap, SideEffect, ReducingBarrier) provide the correct type taxonomy for query plan nodes. Each node in our plan is one of these types.
- Gremlin's mutable builder pattern is REJECTED. We use Django's immutable pattern instead.

Implementation pattern:
```python
class MemoryQuery:
    """Django-style clone-on-write with Gremlin-style step types."""

    def _chain(self, node: PlanNode) -> 'MemoryQuery':
        clone = MemoryQuery.__new__(MemoryQuery)
        clone._plan = self._plan + [node]  # Append new step
        clone._store = self._store          # Share storage reference
        clone._result_cache = None          # Reset cache
        return clone

    def where(self, **predicates) -> 'MemoryQuery':
        return self._chain(FilterNode(predicates))  # Django .filter()

    def traverse(self, relation: str) -> 'MemoryQuery':
        return self._chain(FlatMapNode(relation))   # Gremlin .out()

    def count(self) -> int:
        return self._chain(BarrierNode('count'))._execute()  # Terminal

    def resolve(self) -> List[Entity]:
        self._fetch_all()
        return self._result_cache

    def __iter__(self):
        self._fetch_all()
        return iter(self._result_cache)

    def _fetch_all(self):
        if self._result_cache is None:
            plan = QueryPlan(self._plan)
            optimized = apply_strategies(plan)  # Gremlin TraversalStrategy
            self._result_cache = optimized.execute(self._store)
```

### 2.2 Data Model: Graphiti Bi-Temporal + Hindsight Epistemic Layers

**Decision: Four timestamps per relation (from Graphiti) stored across four epistemic layers (from Hindsight).**

Rationale:
- Graphiti's bi-temporal model (t_valid, t_invalid, t_created, t_expired) is the correct way to track fact evolution. Event time (when true in the world) and system time (when the system learned it) are fundamentally different timelines.
- Hindsight's four-network model (world, experience, opinion, observation) provides the correct epistemic separation. Facts are not beliefs. Experiences are not observations.

Implementation mapping:

| Graphiti Concept | Hindsight Concept | Hermes Concept |
|-----------------|-------------------|----------------|
| Episode Subgraph | World/Experience Network | STATED layer |
| Semantic Subgraph | (edges between entities) | INFERRED layer |
| Community Subgraph | Observation Network | DERIVED layer |
| Edge invalidation | Opinion confidence update | Supersession edges with `invalid_at` |
| — | Opinion Network | Belief annotations on relations |

Each MSL relation gains:
```python
@dataclass
class TemporalMeta:
    valid_from: Optional[datetime]    # When this became true (event time)
    valid_until: Optional[datetime]   # When this stopped being true (event time)
    created_at: datetime              # When we learned this (system time)
    superseded_at: Optional[datetime] # When we learned it was wrong (system time)
    confidence: float = 1.0           # Hindsight-style confidence score
    epistemic: str = "fact"           # "fact" | "opinion" | "inference"
```

### 2.3 Query Optimization: Gremlin TraversalStrategy

**Decision: Adopt Gremlin's strategy pipeline for query plan optimization.**

Rationale:
- Gremlin's TraversalStrategy is a clean interceptor pattern for rewriting query plans before execution. Strategy categories (decoration, optimization, provider, finalization, verification) cover all optimization needs.
- The specific optimization of LazyBarrierStrategy (insert bulk-synchronization points) maps to our need for intermediate result batching.

Strategies to implement:
1. **FilterPushdownStrategy**: Move filter nodes before traversal nodes when the filter is on source entity properties.
2. **FilterMergeStrategy**: Combine adjacent filter nodes into a single compound predicate.
3. **TemporalPruningStrategy**: If a temporal filter is present, push it to the earliest possible position to prune results early.
4. **CacheLookupStrategy**: If an identical sub-plan has been executed recently, substitute a cache lookup node.

### 2.4 Knowledge Cascade: KAG Three-Layer Model

**Decision: Adopt KAG's three-layer knowledge cascade for retrieval.**

Rationale:
- KAG's insight that structured/schema-constrained knowledge should be queried first (highest precision), then schema-free knowledge, then raw text as fallback, is the correct priority ordering.
- This maps directly to our three-layer memory architecture.

Hermes cascade:
```
Query → MSL Structured Store (schema-constrained, typed entities/relations)
         └── if insufficient →
             BM25/Cosine over extracted triples (schema-free)
             └── if insufficient →
                 BM25/Cosine over raw conversation chunks (raw context)
```

### 2.5 Retrieval: Hindsight TEMPR 4-Way Parallel

**Decision: Adopt Hindsight's 4-way parallel retrieval with RRF fusion and cross-encoder reranking.**

Rationale:
- Hindsight's TEMPR achieves 91.4% on LongMemEval by running four retrieval channels in parallel: semantic, keyword, graph, temporal. No single channel is sufficient.
- We already have three of the four channels. Adding temporal filtering as an explicit channel completes the model.

Implementation:
```python
async def recall(query, store) -> List[Memory]:
    # 4-way parallel (Hindsight TEMPR pattern)
    results = await asyncio.gather(
        cosine_search(query, store),        # Channel 1: Semantic
        bm25_search(query, store),          # Channel 2: Keyword
        entity_graph_traverse(query, store), # Channel 3: Graph
        temporal_filter(query, store),       # Channel 4: Temporal
    )
    fused = reciprocal_rank_fusion(*results)
    reranked = cross_encoder_rerank(query, fused)
    return reranked
```

### 2.6 Reflect Operation: Hindsight CARA (Simplified)

**Decision: Adopt Hindsight's reflect as a first-class memory operation, but make opinion formation deterministic where possible.**

Rationale:
- The idea that memory should not just store and retrieve, but also REFLECT (synthesize new knowledge from existing memories) is correct.
- However, Hindsight's fully LLM-driven opinion formation is opaque and expensive. We want deterministic rules for common cases.

Hermes reflect:
```python
# On-demand, per-topic reflection
def reflect(topic: str, store: MemoryStore) -> DerivedFacts:
    # 1. Retrieve all facts about topic
    facts = store.recall(topic, all=True)

    # 2. Deterministic synthesis (no LLM needed)
    #    - Count mentions → confidence
    #    - Detect contradictions → supersession
    #    - Temporal ordering → progression narrative
    summary = deterministic_synthesize(facts)

    # 3. Store in DERIVED layer
    store.derived.upsert(topic, summary)

    return summary
```

### 2.7 Provenance: Graphiti Episode Lineage + KAG Mutual Indexing

**Decision: Every derived fact links bidirectionally to its source conversation turns.**

Rationale:
- Graphiti's episode subgraph ensures every semantic fact can be traced to its source episode. Bidirectional indices enable both forward and backward traversal.
- KAG's mutual indexing ensures every KG triple links to its source text chunk and vice versa.
- Both are saying the same thing: provenance is non-negotiable.

Implementation:
```python
@dataclass
class MSLRelation:
    subject: str
    predicate: str
    object: str
    temporal: TemporalMeta
    source_turn_ids: List[str]  # Provenance: which turns produced this

@dataclass
class ConversationTurn:
    turn_id: str
    text: str
    extracted_relation_ids: List[str]  # Forward index: what was extracted
```

---

## 3. What We Reject (and Why)

| Pattern | Source | Rejection Reason |
|---------|--------|-----------------|
| Mutable builder | Gremlin | Cannot fork queries for set operations. Clone-on-write (Django) is correct. |
| LLM entity resolution | Graphiti | Too expensive at scale. Use embedding + string matching + probabilistic (Splink) instead. |
| LLM query planning | KAG | Too slow/expensive for common patterns. Use deterministic planning with LLM fallback. |
| LLM opinion formation | Hindsight | Opaque, expensive. Use deterministic rules + optional LLM for synthesis text. |
| Neo4j requirement | Graphiti | Heavyweight. Use embedded (Kuzu) or in-memory for small graphs. |
| OpenSPG server | KAG | Java server dependency. Everything must be in-process Python. |
| Remote execution | Gremlin | Must work in-memory. No server required for basic operation. |
| Batch without invalidation | Graphiti | Silently creates stale data. Always check contradictions. |
| Natural language query only | Hindsight | Must support structured programmatic queries. NL queries layered on top. |
| Label propagation communities | Graphiti | Unstable with insertion order. Use deterministic community maintenance. |

---

## 4. Comparison Matrix: What Each System Provides vs. What Hermes Needs

| Capability | Graphiti | Django | Gremlin | KAG | Hindsight | **Hermes (target)** |
|-----------|---------|--------|---------|-----|-----------|-------------------|
| Lazy evaluation | No | YES | Partial | No | No | **YES** |
| Clone-on-write | No | YES | No (mutable) | No | No | **YES** |
| Boolean composition (Q objects) | No | YES | Partial | No | No | **YES** |
| Step type taxonomy | No | No | YES | Partial | No | **YES** |
| Query optimization strategies | No | SQL optimizer | YES | No | No | **YES** |
| Graph traversal | BFS only | No | YES (full) | Cypher | Graph search | **YES** |
| Bi-temporal model | YES | No | No | No | Partial | **YES** |
| Edge invalidation | YES | No | No | No | Opinion update | **YES** |
| Episode provenance | YES | No | No | Mutual index | No | **YES** |
| Knowledge cascade | No | No | No | YES (3-layer) | Priority order | **YES** |
| 4-way parallel retrieval | 3-way | No | No | Hybrid | YES (4-way) | **YES** |
| Epistemic separation | No | No | No | No | YES (4 networks) | **YES** |
| Reflect/synthesize | No | No | No | Reflect+replan | YES (CARA) | **YES** |
| Allen interval algebra | No | No | No | No | No | **YES (novel)** |
| Set operations on entities | No | SQL-level | Branch-level | No | No | **YES (novel)** |
| Aggregation on entity sets | No | YES (aggregate) | Reduce steps | Yes | No | **YES** |
| Python-native fluent API | Yes | YES | Java-ported | No | Yes | **YES** |
| Typed entity ontology | Pydantic | Model-level | No schema | SPG schema | Typed networks | **YES** |
| In-process execution | Neo4j required | In-process | Server required | Server required | In-process | **YES** |

---

## 5. Novel Contributions (What No One Has)

Based on this analysis, Hermes Foundry's novel contributions are:

1. **Allen interval algebra integrated into a composable query language.** No existing system (Graphiti, Gremlin, KAG, Hindsight) supports Allen interval reasoning. All use point-based or range-based temporal filtering. We add `.before()`, `.after()`, `.during()`, `.overlaps()`, `.meets()`, etc. as composable query operators.

2. **Set operations (union/intersect/subtract) on typed entity sets.** Django has SQL-level set ops. Gremlin has branch-level union. Neither operates on typed entity collections with lazy evaluation. Our MemoryQuery supports `query_a.union(query_b)`, `query_a.intersect(query_b)`, `query_a.subtract(query_b)` returning new MemoryQuery objects.

3. **Combined lazy composition + graph traversal + temporal reasoning in a single Python API.** Django has lazy composition but no graph. Gremlin has graph but no lazy composition. Graphiti has temporal but no composition. KAG has reasoning but no composition. We unify all four.

4. **Deterministic knowledge cascade with composable query plans.** KAG cascades through knowledge layers but via LLM-driven planning. We cascade deterministically with explicit query plan nodes, enabling inspection, caching, and optimization.

---

## 6. Implementation Priority

Based on the analysis, the build order for Hermes Foundry should be:

### Phase 1: Core Query Composition (Django patterns)
1. `MemoryQuery` with `_chain()` clone-on-write
2. `FilterNode`, `FlatMapNode`, `BarrierNode` plan nodes
3. Terminal operations (`__iter__`, `__len__`, `__bool__`, `.resolve()`, `.count()`)
4. `MemoryQ` for boolean composition (AND, OR, NOT)
5. `_fetch_all()` with result caching

### Phase 2: Graph Traversal (Gremlin patterns)
6. `.traverse(relation)` as FlatMapNode
7. `.repeat().until()` for multi-hop traversal
8. `MemoryTraverser` carrier object (entity + bulk + path)
9. `QueryStrategy` pipeline (FilterPushdown, FilterMerge)

### Phase 3: Temporal Model (Graphiti patterns)
10. `TemporalMeta` on all relations (4 timestamps)
11. Edge invalidation logic (from Graphiti's `resolve_edge_contradictions`)
12. Episode provenance (bidirectional source_turn_ids)
13. Allen interval operators (`.before()`, `.after()`, `.during()`, `.overlaps()`)

### Phase 4: Epistemic Layers (Hindsight patterns)
14. Four-layer store (stated, inferred, derived, belief)
15. Confidence scores on relations
16. `.reflect(topic)` operation
17. Knowledge cascade retrieval (structured -> triples -> chunks)

### Phase 5: Advanced Features
18. Aggregation operators (`.count()`, `.group_by()`, `.aggregate()`)
19. Set operations (`.union()`, `.intersect()`, `.subtract()`)
20. Query plan serialization (bytecode, from Gremlin)
21. Backend-agnostic compilation (in-memory vs Kuzu vs Neo4j)

---

## 7. Open Questions (Resolved by This Analysis)

| Question | Resolution |
|----------|-----------|
| Clone-on-write or mutable builder? | **Clone-on-write** (Django). Gremlin's mutable pattern prevents query forking. |
| How to handle entity resolution without LLM? | **Embedding + string matching + Splink probabilistic model.** Graphiti's LLM approach is too expensive. |
| Allen IA tractable at scale? | **Still open.** No system we studied uses Allen IA, so no empirical evidence from prior art. Need to determine which subset of Allen relations we need and test tractability. |
| Time-travel queries needed? | **Yes.** Graphiti's bi-temporal model enables "what was true as of date X?" This is needed for conversational memory (correcting past beliefs). |
| Aggregation over graphs? | **Gremlin reduce steps + Django aggregate/annotate split.** Barrier/reduce nodes consume all input before emitting. Annotate = per-entity. Aggregate = whole-set terminal. |

---

## 8. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Query plan optimization complexity | Start with simple strategies (filter pushdown, merge). Add advanced strategies only when profiling shows bottlenecks. |
| Allen IA tractability | Implement only the 7 base Allen relations. Use the ORD-Horn subalgebra (known to be polynomial). |
| Backend abstraction leaking | Design the expression tree first (from Django). Add backend compilation later. Start with in-memory execution only. |
| Over-engineering the query language | Ship Phase 1-2 first. Validate with real MABench queries. Only add Phase 3-5 features that solve real retrieval failures. |

---

## Sources

### Individual Deep Dives
1. [graphiti-deep-dive.md](prior-art/graphiti-deep-dive.md) — Graphiti/Zep bi-temporal KG
2. [django-queryset-deep-dive.md](prior-art/django-queryset-deep-dive.md) — Django QuerySet internals
3. [gremlin-deep-dive.md](prior-art/gremlin-deep-dive.md) — Gremlin/TinkerPop traversal language
4. [kag-deep-dive.md](prior-art/kag-deep-dive.md) — KAG/OpenSPG logical form solver
5. [hindsight-deep-dive.md](prior-art/hindsight-deep-dive.md) — Hindsight four memory networks

### Primary Sources
6. [Zep Paper (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
7. [Graphiti GitHub](https://github.com/getzep/graphiti)
8. [Django QuerySet Source](https://github.com/django/django/blob/main/django/db/models/query.py)
9. [Django QuerySet API](https://docs.djangoproject.com/en/5.0/ref/models/querysets/)
10. [TinkerPop Reference](https://tinkerpop.apache.org/docs/current/reference/)
11. [Gremlin Paper (arXiv:1508.03843)](https://arxiv.org/pdf/1508.03843)
12. [KAG Paper (arXiv:2409.13731)](https://arxiv.org/html/2409.13731v3)
13. [KAG GitHub](https://github.com/OpenSPG/KAG)
14. [Hindsight Paper (arXiv:2512.12818)](https://arxiv.org/abs/2512.12818)
15. [Hindsight GitHub](https://github.com/vectorize-io/hindsight)
