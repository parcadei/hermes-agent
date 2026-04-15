# Adversarial Pass 2: Scale-Check (⊞) + Materialize (⌂)

**Reviewer**: architect-agent (Opus 4.6)
**Date**: 2026-04-14
**Spec version**: 1.0 (2026-04-13), with Appendix A mitigations from Pass 1
**Operators applied**: Scale-Check (14 findings), Materialize (12 findings)

---

## Executive Summary

Pass 1 found 25 level-conflation and contradiction bugs. This pass finds a different class of problems: **quantitative infeasibility** and **under-specified observable behavior**. Of the 26 findings, **4 are FATAL** (the system as specified cannot physically work at the intended scale or produces undefined behavior on common inputs), **12 are HIGH** severity (will cause performance collapse or silent incorrect behavior), and **10 are MEDIUM** (will require design revision before the system is usable in production).

The headline finding: the spec claims `max_entities: 100_000` as a soft cap, but several O(N) and O(N^2) structures embedded in the design make anything beyond ~10K entities unworkable without architectural changes. The co-occurrence index, embedding-based resolution, and BFS graph traversal all have hidden quadratic or linear blowups that the spec's complexity analysis in Section 11.2 understates.

---

## Scale-Check Findings (⊞)

### SC-1: Co-occurrence Index Memory at Scale is O(N^2) [FATAL]

- **Severity**: FATAL
- **Operator**: ⊞ Scale-Check
- **Location**: Section 2.3.6 (CooccurrenceIndex), Section 4.3.8 (Co-occurrence Tracking)
- **Assumption**: The spec stores co-occurrence as `(entity_a, entity_b, chunk_idx)` tuples and states complexity is "O(1) per pair lookup. Memory: O(P) where P = unique entity pairs." (Section 11.2)
- **Reality**: P grows quadratically. If a conversation chunk mentions K entities, the number of co-occurrence pairs from that chunk is C(K,2) = K*(K-1)/2. Across the whole corpus, P = number of unique entity pairs that ever co-appeared.

  **Actual calculation for the proven dataset (LME):**
  - 3200 chunks, ~60 conversations, ~1280 turns per context
  - Existing mabench extraction yields ~5-15 entities per chunk (faceted extraction with Qwen2.5-7B)
  - At 10 entities per chunk: C(10,2) = 45 pairs per chunk
  - 3200 chunks x 45 = 144,000 co-occurrence entries (manageable)

  **At 10x scale (32K chunks, ~2000 entities):**
  - Unique entity pairs: up to C(2000,2) = ~2M pairs
  - Memory: 2M x (2 strings + set of chunk indices) ~ 200-400 MB
  - Still manageable but storage-heavy

  **At 100x scale (320K chunks, ~20K entities):**
  - Unique entity pairs: up to C(20000,2) = ~200M pairs
  - Memory: 200M x (2 strings + set) = **40-80 GB** just for the co-occurrence index
  - This exceeds typical machine RAM for a user-level memory system

  **At the spec's claimed max_entities=100K:**
  - Worst case: C(100000,2) = ~5 billion pairs
  - This is physically impossible to store. Even with sparse storage (only pairs that actually co-appear), the spec has no sparsity bound.

- **Numbers**: O(P) is misleading. P = O(E^2) in the worst case, where E = entity count. The spec must bound the average co-occurrence degree or use a sparse representation with a hard cap.
- **Mitigation**: 
  1. Cap co-occurrence to top-K neighbors per entity (e.g., K=100). Store as adjacency list, not pair set.
  2. Add explicit memory budget: `max_cooccurrence_pairs: int = 1_000_000` in FoundryConfig (SCHEMA category).
  3. Eviction policy: when budget exhausted, evict lowest-count pairs.
  4. Document that co-occurrence density scales with O(E * avg_K) not O(E^2).

---

### SC-2: Embedding-Based Entity Resolution is O(N) per Call [HIGH]

- **Severity**: HIGH
- **Operator**: ⊞ Scale-Check
- **Location**: Section 3.4 (EntityResolver), existing code at entity_graph.py:124-142
- **Assumption**: The spec lists embedding resolution as step 5 of the cascade, implying it is a reasonable fallback. Section 11.2 says entity resolution is "O(1) for exact match, O(N) for substring fallback."
- **Reality**: The actual implementation (`_resolve_by_embedding` in entity_graph.py L124-142) performs a **linear scan** over ALL entity embeddings:
  ```python
  for canon, emb in self._entity_embs.items():
      sim = self._cosine_sim(m_emb, emb)
  ```
  At ingestion time, this is called for EVERY unresolved mention. If 30% of mentions hit the embedding fallback:
  
  **At LME scale (1000 entities):** 1000 cosine comparisons per call x 768d vectors = ~0.5ms. Fine.
  
  **At 10K entities:** 10K comparisons per call. With 50K mentions during ingestion hitting embedding fallback (30% of ~170K triples): 50K x 10K = 500M cosine computations. At ~1M cosine/sec (numpy): **500 seconds** just for resolution. 
  
  **At 100K entities:** 100K comparisons per call. Ingestion becomes minutes per mention. The entire ingestion pipeline halts.

- **Numbers**: Embedding resolution is O(E) per call, and called O(M) times during ingestion where M = unresolved mentions. Total = O(M*E). At 100K entities this is O(10^10) dot products.
- **Mitigation**:
  1. Replace linear scan with FAISS IVF index (already in VectorIndex spec but not used for resolution).
  2. Resolution embedding index: HNSW with ef_search=50 gives O(log N) per lookup.
  3. Fallback to approximate nearest neighbor. Threshold 0.85 is high enough that ANN recall > 99%.
  4. Add to spec: "Step 5 (embedding resolution) MUST use approximate nearest neighbor search, not brute-force linear scan. Required index type: HNSW with ef_construction=200."

---

### SC-3: BM25 Query Complexity Is O(V*D) Not O(V*Q) [HIGH]

- **Severity**: HIGH
- **Operator**: ⊞ Scale-Check
- **Location**: Section 11.2 (Scale Behavior), existing code at layered_memory.py:81-107
- **Assumption**: Section 11.2 states BM25 is "O(V * Q) where V = vocabulary size, Q = query terms. Proven to 3200+ chunks."
- **Reality**: The actual BM25 implementation (layered_memory.py L81-107) performs:
  ```python
  for t in tokens:           # Q query terms
      for i in range(self._n):   # D documents
          tf_val = self._tf[i].get(t, 0)
  ```
  This is O(Q * D), not O(V * Q). The spec confuses V (vocabulary) with D (document count).
  
  **At LME scale (3200 chunks):** 5 query terms x 3200 docs = 16K ops. Sub-millisecond.
  
  **At 10K chunks:** 5 x 10K = 50K ops. Still fast (~1ms).
  
  **At 100K chunks:** 5 x 100K = 500K ops. ~5-10ms. Acceptable but approaching budget.
  
  **At 1M chunks (long-lived user):** 5 x 1M = 5M ops per query. ~50-100ms. Exceeds the 10ms budget in Section 12.2.

  The entity BM25 has the same issue. With 100K entities, each entity BM25 query scans all 100K entity documents.

- **Numbers**: True complexity is O(Q * D) per query. The 10ms budget (Section 12.2) allows ~100K chunks maximum with the current linear-scan implementation.
- **Mitigation**:
  1. Correct Section 11.2: BM25 is O(Q * D) where D = document count, Q = query term count.
  2. Add inverted index: standard BM25 with inverted posting lists is O(Q * avg_posting_length), which is O(Q * D/V) on average. This is the standard optimization.
  3. Specify transition point: "For D > 50K, BM25 implementation MUST use inverted posting lists, not linear scan."
  4. Add to performance budget: "BM25 query < 10ms at 100K chunks requires inverted index."

---

### SC-4: Agentic Retrieval Token Budget Blows Up with Context Accumulation [HIGH]

- **Severity**: HIGH
- **Operator**: ⊞ Scale-Check
- **Location**: Section 5.5 (AgenticRetriever), Section 5.7 (Budget Management), existing code at agentic_retrieval.py:588-669
- **Assumption**: The spec says max_steps=8, max_observation_chars=2000, max_tokens=512 per LLM call. Budget is "30 second timeout."
- **Reality**: The ReAct loop appends ALL previous messages to the context on each step. The actual code (agentic_retrieval.py L603-652) does:
  ```python
  messages.append({"role": "assistant", "content": llm_text})
  messages.append({"role": "user", "content": f"Observation: {obs_text}"})
  ```
  
  **Token accumulation per step:**
  - System prompt (SYSTEM_PROMPT + TOOL_DESCRIPTIONS): ~800 tokens
  - User question: ~30 tokens
  - Per step: ~100 tokens (LLM thought+action) + ~500 tokens (observation, capped at 2000 chars ~ 500 tokens)
  - Step N context: 830 + N * 600 tokens
  
  **At step 8 (max budget):**
  - Input tokens: 830 + 8 * 600 = **5,630 tokens**
  - Output tokens: 8 * 512 = 4,096 tokens
  - Total tokens per question: ~9,700 tokens
  
  **Cost at gpt-4o-mini rates ($0.15/1M input, $0.60/1M output):**
  - Per question: $0.15 * 5.63/1000 + $0.60 * 4.1/1000 = $0.0033
  - 300 questions (LME): $0.99
  - 10K questions (production daily load): **$33/day**
  
  **But the REAL problem is latency, not cost:**
  - gpt-4o-mini latency: ~200ms per call + ~100ms per 100 output tokens
  - Step 8: 200ms + (512/100 * 100ms) = ~700ms per step
  - 8 steps: 8 * 700ms = **5.6 seconds** just for LLM calls
  - Plus tool execution: ~50ms per tool call = 400ms
  - Total: ~6 seconds per question (within 10s budget)
  
  **At 10x observation size (if observation budget increases):**
  - Step 8 context: 830 + 8 * 5100 = 41,630 tokens. Exceeds gpt-4o-mini's optimal window. Quality degrades.

- **Numbers**: Token accumulation is linear: context_tokens = base + N * step_tokens. At max_steps=8 this is manageable. But the 30-second timeout and 512 max_tokens per step constrain the budget tightly. Any increase to max_steps or observation size pushes against these limits non-linearly (LLM latency scales with input length).
- **Mitigation**:
  1. Add explicit token budget: `max_context_tokens: int = 8192` to RetrievalBudget.
  2. Implement sliding window: after step 4, summarize earlier observations into a compressed context.
  3. Switch to native tool calling (spec already mentions this in Section 5.9) — parallel calls reduce step count.
  4. Document the cost model: per-question cost at each step count for budget planning.

---

### SC-5: EntityNode.facts Dict Memory at max_facts_per_entity=1000 [HIGH]

- **Severity**: HIGH
- **Operator**: ⊞ Scale-Check
- **Location**: Section 2.2 (EntityNode), Section 10.1 (FoundryConfig max_facts_per_entity)
- **Assumption**: The spec allows max_facts_per_entity=1000 and max_entities=100_000.
- **Reality**: Each Fact object contains 16 fields. Estimating Python object overhead:
  
  **Memory per Fact:**
  - id (str, uuid4): 36 bytes + 50 bytes overhead = 86 bytes
  - subject (str): ~20 bytes avg + 50 bytes = 70 bytes
  - predicate (str): ~15 bytes avg + 50 bytes = 65 bytes
  - object (str): ~50 bytes avg + 50 bytes = 100 bytes
  - source_text (str): ~200 bytes avg + 50 bytes = 250 bytes (the large one)
  - Other fields (ints, booleans, Optional): ~200 bytes
  - Total per Fact: **~770 bytes**
  
  **At LME scale (1000 entities, ~10 facts each):**
  - 10K facts x 770 bytes = ~7.7 MB. Fine.
  
  **At max config (100K entities, 1000 facts each):**
  - 100M facts x 770 bytes = **77 GB**
  - This cannot fit in memory. The spec's max_facts_per_entity=1000 combined with max_entities=100K implies a system that needs 77 GB just for the Fact objects.
  
  **Realistic projection (100K entities, 20 facts avg):**
  - 2M facts x 770 bytes = ~1.5 GB. Feasible but heavy for a user-level system.

- **Numbers**: Memory = E * F_avg * 770 bytes. At E=100K, F_avg must stay below ~13 to fit in 1 GB. The default max_facts_per_entity=1000 is unreachable at 100K entities.
- **Mitigation**:
  1. Add a TOTAL fact budget: `max_total_facts: int = 2_000_000` (caps at ~1.5 GB).
  2. Superseded facts should be evictable from memory (keep on disk, load on demand for timeline queries).
  3. source_text is the biggest field. For superseded facts, consider dropping source_text and storing only a hash for grounding verification on reload.
  4. Document memory model: "At E entities with F facts each, memory ~ E * F * 800 bytes."

---

### SC-6: VectorIndex Dimensionality x Entity Count Memory [HIGH]

- **Severity**: HIGH
- **Operator**: ⊞ Scale-Check
- **Location**: Section 2.3.2 (VectorIndex), Section 10.1 (embedding_model)
- **Assumption**: The spec says "numpy matmul for < 100K vectors, FAISS for larger" and uses all-mpnet-base-v2 (768d).
- **Reality**: Vector storage memory:
  
  **768d float32 vectors:**
  - Per vector: 768 * 4 bytes = 3,072 bytes = 3 KB
  
  **At LME scale (3200 chunks + 1000 entities = 4200 vectors):**
  - 4200 * 3 KB = 12.6 MB. Fine.
  
  **At 100K chunks + 100K entities = 200K vectors:**
  - 200K * 3 KB = 600 MB. Getting heavy.
  
  **At 1M chunks (long-lived user, 5 years of conversations):**
  - 1M * 3 KB = 3 GB just for chunk embeddings.
  - Plus entity embeddings: potentially another 300 MB.
  
  **Brute-force matmul at 100K vectors:**
  - Query: matmul of (1, 768) x (100K, 768)^T = 768 * 100K = 76.8M FLOPs
  - At ~10 GFLOPS (numpy on CPU): ~7.7ms per query. Within budget but barely.
  
  **At 1M vectors, brute-force is impossible:**
  - 768M FLOPs = ~77ms. Exceeds BM25 query budget.
  - FAISS IVF required but the spec says "for larger" without specifying the transition.

- **Numbers**: Storage = N * D * 4 bytes. Query = O(N * D) for brute-force, O(sqrt(N) * D) for IVF. The 100K transition point needs FAISS with nlist=sqrt(N)=316, nprobe=16.
- **Mitigation**:
  1. Specify the exact FAISS transition: "For N > 50K vectors, use IndexIVFFlat with nlist=int(sqrt(N)), nprobe=min(16, nlist)."
  2. Consider dimension reduction: Matryoshka embeddings at 256d reduce storage by 3x and query by 3x.
  3. Add memory budget to FoundryConfig: `max_vectors: int = 500_000` with eviction of oldest superseded chunks.
  4. Document: "VectorIndex memory = N * 3KB at 768d. Budget: 500K vectors = 1.5 GB."

---

### SC-7: Allen Interval `events_between` Is Linear Scan [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⊞ Scale-Check
- **Location**: Section 2.3.5 (TemporalIndex)
- **Assumption**: The spec defines `events_between(start, end)` and `events_on_date(date)` as convenience methods, with 13 Allen relations.
- **Reality**: Without an interval tree or segment tree, range queries require scanning ALL events:
  
  **At LME scale (~500 temporal events):** 500 comparisons. Negligible.
  
  **At 10K events:** 10K comparisons per query. ~0.1ms. Fine.
  
  **At 100K events (5 years of daily conversations with temporal content):**
  - 100K comparisons per query. ~1ms. Still fine.
  
  **Allen relation computation between all pairs (for ordering queries):**
  - "Which happened first, X or Y?" requires finding both events then comparing.
  - If match_events returns K candidates and needs to compare all pairs: O(K^2).
  - At K=100 candidates: 10K Allen comparisons. Fine.
  - At K=1000 (large candidate set): 1M comparisons. ~10ms. Approaching budget.
  
  The real bottleneck is match_events retrieval (BM25+cosine), not Allen computation.

- **Numbers**: events_between is O(N), Allen pairwise is O(K^2). Neither is a problem until N > 100K or K > 500. The spec's scale claim of 100K+ entities does not necessarily mean 100K temporal events.
- **Mitigation**:
  1. For N > 10K temporal events, use an interval tree (O(log N + K) for range queries where K = overlapping events).
  2. Document: "Allen computation is per-pair O(1). The bottleneck is candidate retrieval, not interval comparison."
  3. Add `max_temporal_events: int = 50_000` to FoundryConfig.

---

### SC-8: Graph Traversal BFS at max_hops=3 on Dense Graphs [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⊞ Scale-Check
- **Location**: Section 11.2, Section 3.2 (TraversalOp)
- **Assumption**: Section 11.2 says "O(V + E) bounded by max_hops. max_hops=3 keeps this tractable."
- **Reality**: BFS at depth 3 visits all nodes within 3 hops. The number of visited nodes depends on the graph's branching factor (average degree d):
  
  **Nodes visited at depth k: O(d^k)**
  
  **At LME scale (1000 entities, avg degree ~5):**
  - Depth 3: 5^3 = 125 nodes visited. Fast.
  
  **At 10K entities, avg degree ~10:**
  - Depth 3: 10^3 = 1000 nodes visited. Fine.
  
  **At 100K entities, avg degree ~20 (denser due to co-occurrence):**
  - Depth 3: 20^3 = 8000 nodes visited. Each node requires loading facts.
  - If each node has 20 facts: 160K fact objects loaded. ~120 MB in memory.
  - Takes ~50ms for the traversal + fact loading.
  
  **Pathological case: hub entity with degree 500 (e.g., "the user" entity):**
  - Depth 1: 500 nodes. Depth 2: 500 * 20 = 10K nodes. Depth 3: 200K nodes.
  - This EXCEEDS the entity count. The "user" entity, connected to everything, turns any 3-hop traversal into a full graph scan.
  
- **Numbers**: Worst case is O(d^3) where d = max degree. With hub entities (d=500), depth-3 BFS touches 200K+ nodes. The spec's max_hops=3 clamp is necessary but insufficient for hub entities.
- **Mitigation**:
  1. Add per-traversal node budget: `max_traversal_nodes: int = 1000`. BFS stops when budget exhausted.
  2. Hub detection: entities with degree > 100 are "hub entities." Traversal through hubs uses only top-K edges (by co-occurrence count), not all edges.
  3. The "user" entity is ALWAYS a hub. Special-case it: `.related("visited")` on the user entity should use the predicate filter to limit fan-out.
  4. Add to Section 11.2: "BFS complexity is O(min(d^k, max_traversal_nodes)) where d = branching factor."

---

### SC-9: Full-Context Mode Threshold Is Unworkable [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⊞ Scale-Check
- **Location**: Section 5.8 (OAG Pattern 1)
- **Assumption**: "If total_chunks * avg_chunk_tokens < 100K, consider full-context mode."
- **Reality**: At 512 tokens per chunk:
  - 100K / 512 = ~195 chunks. That is approximately 15 conversations (LME has ~13 chunks per conversation).
  - A user with 20+ conversations immediately exceeds this threshold.
  - gpt-4o-mini has a 128K context window, but quality degrades significantly after ~30K tokens (the "lost in the middle" effect, Liu et al. 2023).
  - At 100K input tokens, gpt-4o-mini latency is ~3-5 seconds and cost is $0.015 per question.
  
  **The threshold is only useful for brand-new users with < 15 conversations.** For any real workload, it never triggers.

- **Numbers**: 100K token threshold = ~195 chunks = ~15 conversations. Most users exceed this within the first week.
- **Mitigation**:
  1. Lower the threshold to 20K tokens (~40 chunks, ~3 conversations). This is where LLMs actually perform well on full-context.
  2. Make it configurable: `full_context_threshold_tokens: int = 20_000`.
  3. Document: "Full-context mode is a cold-start optimization for new users. After ~3 conversations, retrieval-based mode activates."

---

### SC-10: Save/Load Serialization Size at Scale [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⊞ Scale-Check
- **Location**: Section 2.4 (Persistence), Section 12.2 (Performance: "Save/load: < 5 seconds for 100K entities")
- **Assumption**: "Format: one JSON file per index + binary files for embeddings." Save/load < 5 seconds for 100K entities.
- **Reality**: JSON serialization of the EntityGraph at scale:
  
  **Per EntityNode JSON size:**
  - canonical (str): ~20 bytes
  - aliases (set, avg 3): ~60 bytes
  - types (set, avg 2): ~40 bytes
  - facts (dict, avg 20 facts): 20 * 770 bytes (fact size) = 15.4 KB
  - related (set, avg 10): ~200 bytes
  - mention_turns (set, avg 5): ~100 bytes
  - Total per entity: ~16 KB as JSON (JSON overhead ~2x over binary)
  
  **At 100K entities:**
  - Entity graph JSON: 100K * 16 KB = **1.6 GB JSON file**
  - JSON serialization at ~50 MB/s: ~32 seconds. **Exceeds 5-second budget by 6x.**
  - JSON deserialization (parsing): ~20 MB/s: ~80 seconds. Far worse.
  
  **Embedding binary files:**
  - 100K entity embeddings (768d float32): 300 MB. numpy save/load: ~2 seconds. Fine.
  - 100K chunk embeddings: another 300 MB.
  
  **BM25 index state:**
  - TF matrices: 100K docs * ~100 unique terms = 10M entries. JSON: ~200 MB.

- **Numbers**: Total save size at 100K entities: ~2.4 GB. JSON serialization: ~50-80 seconds. The 5-second budget requires binary format, not JSON.
- **Mitigation**:
  1. Use MessagePack or pickle for EntityGraph serialization (5-10x faster than JSON).
  2. Use numpy .npz for all numeric indexes (BM25 TF matrices, embeddings).
  3. Revised budget: "Save/load < 5 seconds at 10K entities. At 100K entities, < 30 seconds with binary format."
  4. Add incremental save: only serialize entities modified since last save.

---

### SC-11: EntityNode.related Set Grows Without Bound [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⊞ Scale-Check
- **Location**: Section 2.2 (EntityNode), existing code entity_graph.py L321-322
- **Assumption**: EntityNode.related is a `set[str]` of all connected entity canonicals. Every ingest_triple adds both directions: `subj_node.related.add(obj_canon)` and `obj_node.related.add(subj_canon)`.
- **Reality**: The "user" entity (or equivalent hub) accumulates a related set entry for EVERY entity it is connected to via any triple. In a 5-year conversation:
  
  **At LME scale (1000 entities):** user.related has ~300-500 entries. ~10 KB. Fine.
  
  **At 100K entities where user is connected to 50% (50K):**
  - user.related = set of 50K strings, avg 20 chars each = **1 MB** just for one set.
  - Serialization of this one set: ~1 MB JSON.
  - Set lookup (membership check): O(1) but memory pressure from 50K string hashes.
  
  This is not catastrophic but contributes to the memory bloat in SC-5. The related set duplicates information already in the facts dict and co-occurrence index.

- **Numbers**: Hub entity related set: O(E) entries. Memory: O(E * avg_name_length). At 100K entities: ~2 MB per hub.
- **Mitigation**:
  1. Cap related set: `max_related_per_entity: int = 500`. Keep only the most co-occurring neighbors.
  2. Or: make related a computed property from facts.keys() + cooccurrence, not a stored set. Eliminates duplication.
  3. Document: "EntityNode.related is a convenience index capped at 500 entries. For full neighbor enumeration, use CooccurrenceIndex."

---

### SC-12: Ingestion Pipeline LLM Calls Dominate Runtime [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⊞ Scale-Check
- **Location**: Section 4.2 (Pipeline Stages), Section 12.2 ("Ingestion: < 5 seconds per 100 turns excluding LLM extraction time")
- **Assumption**: The performance budget explicitly excludes LLM extraction time. But the spec has TWO LLM extraction stages (Stage 3: Triple Extraction, Stage 4: Faceted Entity Extraction) plus one smaller LLM stage (Stage 7: Temporal Extraction with ADELIE 3B).
- **Reality**: 
  **Triple extraction (Qwen2.5-7B on vLLM A6000):** ~4.8s per chunk.
  **Faceted entity extraction (same model, same endpoint):** ~4.8s per chunk.
  **Temporal extraction (ADELIE 3B):** ~1s per chunk.
  
  **Per chunk total LLM time: ~10.6 seconds.**
  
  **At LME scale (3200 chunks):**
  - Sequential: 3200 * 10.6s = **33,920 seconds** = **9.4 hours**
  - With batch parallelism (8 concurrent on 1x A6000): ~1.2 hours
  - With H200 and higher throughput: ~30 minutes
  
  **At 100 turns (the stated budget):**
  - ~8-10 chunks (at 512 tokens, ~140 tokens/turn)
  - LLM time: 8 * 10.6s = 85 seconds (sequential)
  - Non-LLM time: < 5 seconds (the spec's budget)
  - **Total: 90 seconds** for 100 turns, not 5 seconds
  
  The "excluding LLM extraction" caveat makes the 5-second budget misleading. The user-observable latency is dominated by LLM calls.

- **Numbers**: LLM calls are ~95% of ingestion time. The 5-second budget covers only the remaining 5%.
- **Mitigation**:
  1. Document total ingestion time honestly: "100 turns: ~90 seconds with local GPU, ~30 seconds with H200, ~5 seconds non-LLM only."
  2. Add extraction parallelism spec: "Triple and Faceted extraction are independent per chunk and SHOULD run in parallel."
  3. Pre-extraction caching: cache extraction results keyed by chunk hash. Re-ingestion of same conversation skips LLM calls.
  4. Lazy extraction: ingest raw chunks immediately (BM25 available in < 5s), run LLM extraction asynchronously in background.

---

### SC-13: QueryPlan Materialization Limit vs GroupBy Explosion [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⊞ Scale-Check
- **Location**: Section 3.2c (materialization limit 100K), Section 3.5 (GroupedQueryPlan)
- **Assumption**: "A single execute() call returns at most max_entities results (default: 100,000)."
- **Reality**: GroupBy operations can produce combinatorial explosion in group keys:
  
  `foundry.entities().type("restaurant").group_by("where").group_by("sentiment").count()`
  
  If there are 1000 restaurants in 50 cities with 5 sentiment levels: 50 * 5 = 250 groups. Fine.
  
  But: `group_by("predicate_with_high_cardinality")` where the predicate has unique values per entity (e.g., "name") produces N groups where N = entity count. The 2D GroupBy then produces N1 * N2 groups.
  
  **At 10K entities with 2 high-cardinality group keys:** 10K * 10K = 100M groups. The result dict alone would be ~10 GB.

- **Numbers**: GroupBy cardinality is unbounded. 2D GroupBy is O(G1 * G2) where G = distinct values. High-cardinality predicates cause combinatorial blowup.
- **Mitigation**:
  1. Add group key cardinality cap: `max_group_keys: int = 1000` per dimension.
  2. If group count exceeds cap, raise `GroupByCardinalityError` with a message to add filters first.
  3. Forbid GroupBy on properties known to be unique (entity canonical names, fact IDs).
  4. Document: "GroupBy is designed for categorical properties (type, location, status). Using it on high-cardinality properties (names, IDs) produces degenerate results."

---

### SC-14: Dedup Identity Tuple Comparison Cost [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⊞ Scale-Check
- **Location**: Appendix A.10 (Fact Identity), Section 2.2
- **Assumption**: Dedup checks content identity: `(subject, predicate, object, turn_idx, chunk_idx)` tuple comparison.
- **Reality**: During ingestion, every new fact must be checked against existing facts for the same (entity, predicate). The existing code (entity_graph.py L308-313) does:
  ```python
  for existing in subj_node.facts[pred_norm]:
      if (existing.object == obj and existing.turn_idx == turn_idx ...):
  ```
  This is O(F) per check where F = existing facts for that (entity, predicate).
  
  **At normal scale (5-10 facts per predicate):** Negligible.
  
  **Pathological case: hub entity "user" with predicate "mentioned":**
  - If the user mentions 10K things, "user.mentioned" has 10K facts.
  - Each new ingestion of "user mentioned X" scans 10K facts.
  - With 100K new mentions: 100K * 10K = 1 billion comparisons.
  
  The spec's max_facts_per_entity=1000 caps this per entity, but does not cap per (entity, predicate) pair.

- **Numbers**: Dedup check is O(F) per fact per predicate. With max_facts_per_entity=1000, worst case is 1000 comparisons per insert. At 1M inserts: 1B comparisons. Tolerable but slow (~100 seconds).
- **Mitigation**:
  1. Use a set for dedup: `_dedup_set: set[tuple[str, str, str, int, int]]` on EntityNode. O(1) lookup.
  2. Or: hash-based dedup index per entity, cleared on save/load.
  3. Add to spec: "Dedup MUST use hash-based lookup, not linear scan of existing facts."

---

## Materialize Findings (⌂)

### MT-1: ObjectSet Lazy Evaluation — What Actually Happens [FATAL]

- **Severity**: FATAL
- **Operator**: ⌂ Materialize
- **Location**: Section 3.2 (QueryPlan), Section 3.2c (Materialization)
- **Claimed behavior**: "Plans accumulate operations. Nothing executes until a terminal operation is called." The spec shows the chain: `.type("restaurant").where("where", "$eq", "Rome").execute()`.
- **Observable trace**: Let me trace what ACTUALLY happens step by step:

  ```
  1. foundry.entities()
     → Creates QueryPlan(store=self._store, seed=None)
     → seed=None means "all entities" — but nothing is loaded yet.
  
  2. .type("restaurant")
     → Creates NEW QueryPlan with FilterOp(field="type", operator="$eq", value="restaurant")
     → Still nothing executed. No entity data touched.
  
  3. .where("where", "$eq", "Rome")
     → Creates NEW QueryPlan with additional FilterOp
     → Still nothing executed.
  
  4. .execute()
     → NOW execution happens. But HOW?
  ```

  **The spec does not define execution order.** There are two possible strategies:
  
  **Strategy A: Materialize-then-filter (what Django does)**
  - Load ALL entities into memory
  - Apply type filter: scan all, keep restaurants
  - Apply where filter: scan remaining, keep those with where="Rome"
  - Cost: O(E) where E = all entities. At 100K entities, this loads 100K EntityNodes.
  
  **Strategy B: Index-first (what a database does)**
  - TypeIndex.lookup("restaurant") → set of canonicals
  - PropertyIndex.lookup("where") filtered to $eq "Rome" → set of canonicals
  - Intersect sets
  - Load only matching EntityNodes
  - Cost: O(T + W) where T = restaurants, W = entities with where=Rome
  
  **The spec never specifies which strategy is used.** Section 3.2c says "Materialize-then-filter" is an anti-pattern but does not mandate index-first execution. Without an execution planner, the "lazy" evaluation is just deferred eager loading.

- **Gap**: The spec defines WHAT the query language expresses but not HOW execution uses the indexes. A QueryPlan optimizer (or at least execution strategy specification) is missing.
- **Mitigation**:
  1. Add Section 3.2d: "Query Execution Strategy."
  2. Specify: TypeIndex and PropertyIndex filters are applied FIRST to produce a candidate set. Candidate set is intersected before any EntityNode loading.
  3. Define filter pushdown rules: `.type()` pushes to TypeIndex, `.where()` pushes to PropertyIndex, `.before()/.after()` pushes to TemporalIndex.
  4. Fallback: filters that cannot be pushed to an index are applied post-materialization on the candidate set.

---

### MT-2: Entity Resolution Trace — "I Visited Roscioli" [HIGH]

- **Severity**: HIGH
- **Operator**: ⌂ Materialize
- **Location**: Section 3.4 (EntityResolver), Section 4.3.5 (Entity Resolution)
- **Claimed behavior**: EntityResolver resolves mentions through a 6-step cascade.
- **Observable trace**: User says "I visited Roscioli last week." Existing entity: `roscioli_restaurant` with alias `roscioli`.

  ```
  Input mention: "Roscioli"
  
  Step 1: Exact normalized match
    normalize("Roscioli") → "roscioli"
    alias_index["roscioli"] → "roscioli_restaurant"  ✓ MATCH
    → Return "roscioli_restaurant"
  ```

  That was easy. Now a harder case: User says "I went to that new sushi place."

  ```
  Input mention: "that new sushi place"
  
  Step 1: Exact normalized match
    normalize("that new sushi place") → "that new sushi place"
    alias_index["that new sushi place"] → None
  
  Step 2: Determiner strip
    strip_determiners("that new sushi place") → "new sushi place"
    alias_index["new sushi place"] → None
  
  Step 3: Possessive strip
    (No possessive — skip)
  
  Step 4: Type-gated substring
    For each entity canonical:
      "sushi_palace" — does "new sushi place" contain "sushi_palace"? No.
      Does "sushi_palace" contain "new sushi place"? No.
      "sushi" is in "new sushi place" and in "sushi_palace" — but the check is
      canonical-to-mention containment, not token overlap.
    → No match (substring requires one to fully contain the other)
  
  Step 5: Embedding similarity
    embed("new sushi place") → vec_query
    cosine(vec_query, embed("sushi_palace")) → 0.82 (below 0.85 threshold)
    → No match
  
  Step 6: Create new entity
    → Creates "new_sushi_place" entity
  ```

  **Problem**: The user clearly means "sushi_palace" but the resolution cascade fails. The substring check requires full containment, not word overlap. The embedding threshold of 0.85 is too high for paraphrase matching. A NEW entity is created, fragmenting the graph.

- **Gap**: The resolution cascade has a dead zone between substring containment (too strict — requires full string containment) and embedding similarity (too strict — 0.85 threshold rejects valid paraphrases). Common conversational references like "that place", "the new one", "my usual spot" will consistently fail resolution.
- **Mitigation**:
  1. Add a word-overlap resolution step between steps 4 and 5: "Jaccard token overlap >= 0.5 AND type match." In the example, tokens("new sushi place") ∩ tokens("sushi_palace") / union = {"sushi"}/{"new","sushi","place","sushi_palace"} = 0.25. Still fails, but better than nothing.
  2. Lower embedding threshold to 0.80 (the ablation shows 0.85 is too conservative for conversational text).
  3. Add pronoun/deictic resolution step: "that X place" patterns resolve "X" as a type hint and search for entities of that type mentioned recently.
  4. Document the dead-zone: "Mentions that use different words than any known alias will create new entities. Entity merge (Section 6.4) is the correction mechanism."

---

### MT-3: Supersession Chain — "I Have 3 Cats" Then "I Now Have 4 Cats" [HIGH]

- **Severity**: HIGH
- **Operator**: ⌂ Materialize
- **Location**: Section 6.2 (SupersessionEngine)
- **Claimed behavior**: New facts supersede old facts for the same (entity, predicate). Latest turn wins.
- **Observable trace**:

  ```
  Turn 5: "I have 3 cats"
  → extract_triples: (user, has_count, "3", type="cat")
  → Fact A: subject="user", predicate="has_count", object="3", turn_idx=5
  → TypeIndex: user → cat (count via aggregation)
  → No existing fact for (user, has_count) → stored without supersession
  
  Turn 12: "I now have 4 cats, adopted another one"
  → extract_triples: (user, has_count, "4", type="cat")
  → Fact B: subject="user", predicate="has_count", object="4", turn_idx=12
  → SupersessionEngine.supersede("user", "has_count", Fact_B):
      existing = [Fact_A]
      Fact_A.superseded = True
      Fact_A.superseded_by = Fact_B.id
      Fact_B.superseded = False
  → TemporalValidity: Fact_A.valid_until = turn_idx 12
  
  Query: "How many cats does the user have?"
  → TypeIndex.lookup("cat") → [entity_1="fluffy", entity_2="whiskers", entity_3="mittens", entity_4="new_cat"]
  → count = 4 ✓
  ```

  **But wait.** The supersession operates on (entity, predicate) pairs. The predicate here is "has_count" on the "user" entity. But the ACTUAL cat entities (fluffy, whiskers, mittens, new_cat) are separate entities in the graph. The count of 4 comes from TypeIndex, not from the "has_count" fact.

  **The problem**: There are TWO representations of "how many cats":
  1. Explicit fact: `user.has_count = "4"` (superseded from "3")
  2. Implicit count: `TypeIndex.count("cat")` = 4 (counted from individual entities)

  If the LLM extracts "I gave away my cat Mittens" as a new acquisition (sub_role=acquisition, not divestment), the TypeIndex still shows 4 cats but the user now has 3. The supersession of the explicit count fact would need to fire, but only if the LLM extracts "(user, has_count, 3)". More likely, the LLM extracts "(user, gave_away, mittens)" which does NOT trigger supersession on has_count.

- **Gap**: The spec assumes supersession works on (entity, predicate) pairs, but real-world count changes often come from additions/removals of individual entities, not from explicit count statements. The two representations (explicit count fact vs entity set cardinality) can diverge.
- **Mitigation**:
  1. Document the dual representation: "Entity counts are authoritative from TypeIndex (derived), not from explicit count facts (stated). Explicit count facts are evidence, not source of truth."
  2. Add a reconciliation rule: when TypeIndex.count(type) != explicit_count_fact.object, flag a conflict.
  3. Divestment verbs should be added to VerbNet lemma sets: "gave_away", "sold", "lost", "died" → sub_role="divestment". Entities with divestment facts are excluded from counts.
  4. Add to Section 4.4: DerivedLayerBuilder.build() must use VerbNet sub_role filtering when computing type_counts.

---

### MT-4: Temporal Query Trace — "What Restaurants Did I Visit in March?" [HIGH]

- **Severity**: HIGH
- **Operator**: ⌂ Materialize
- **Location**: Section 2.3.5 (TemporalIndex), Section 5.2 (search_temporal tool)
- **Claimed behavior**: Temporal queries use Allen interval algebra with 5 question buckets.
- **Observable trace**:

  ```
  Query: "What restaurants did I visit in March?"
  
  Step 1: Question classification (regex, < 1ms)
    Pattern: "what .+ in March" → "contains" bucket? No, "contains" is "what happened on date".
    Actually this is more like "during" — events whose interval overlaps with March.
    Pattern match: none of the 5 buckets match "what X did I Y in Z".
    → Falls back to generic temporal retrieval.
  
  Step 2: Date extraction from query
    "March" → need a year. reference_date = now (2026-04-14).
    "March" most recently = March 2026.
    dateparser.parse("March") → 2026-03-01? Or 2026-03-15 (midpoint)?
    → AMBIGUOUS. Is "March" [March 1, March 31] or a point?
  
  Step 3: Temporal retrieval
    If "March" resolves to a range [2026-03-01, 2026-03-31]:
      events_between(2026-03-01, 2026-03-31)
      → Returns all events with intervals overlapping March
    
    But the query ALSO has an entity filter: "restaurants" AND "visit".
    The TemporalIndex has no entity-type filter. It returns ALL events in March.
    Filtering to restaurants requires post-retrieval intersection with TypeIndex.
  
  Step 4: Entity intersection
    temporal_events = events_between(March 1, March 31)
    restaurant_entities = TypeIndex.lookup("restaurant")
    
    For each temporal_event:
      if event.subject in restaurant_entities: keep
    
    → Returns events about restaurants in March.
  ```

  **Problems identified:**
  1. **"March" date resolution is ambiguous.** The spec does not specify how bare month names are resolved. dateparser may return a point (March 1) not a range.
  2. **No entity-type filter on temporal queries.** The TemporalIndex.match_events() accepts optional `entities` parameter but the question classification does not extract the entity type constraint.
  3. **No verb filter.** "visited" should filter to events with verb="visit", but temporal events store verb as an optional field that may not match.
  4. **The 5 question buckets do not cover "what X did I do in time_range" questions.** This is a listing query with temporal constraint, not any of ordering/distance/duration/contains.

- **Gap**: Temporal queries that combine entity type + temporal range + verb are common but the spec's question classification has no bucket for them. The temporal engine operates independently from the entity graph.
- **Mitigation**:
  1. Add 6th temporal question bucket: "temporal_filter" for "what [type] did I [verb] in [time_range]" queries.
  2. Specify date-range resolution: bare month names resolve to [month_start, month_end] ranges, not points.
  3. The search_temporal tool should accept optional entity_type and verb parameters for compound temporal queries.
  4. Add a join operation: `TemporalIndex.events_between(start, end, entity_type="restaurant", verb="visit")` with pushdown filters.

---

### MT-5: Conflict Resolution — Contradictory Facts From Different Turns [HIGH]

- **Severity**: HIGH
- **Operator**: ⌂ Materialize
- **Location**: Section 6.2 (SupersessionEngine)
- **Claimed behavior**: "latest_wins" strategy: highest turn_idx wins.
- **Observable trace**: 

  ```
  Turn 3 (user):     "I'm allergic to peanuts"
  → Fact A: (user, allergic_to, "peanuts", turn=3, role="state")
  
  Turn 15 (assistant): "Based on your profile, you might enjoy pad thai with peanuts"
  → Fact B: (user, might_enjoy, "pad thai with peanuts", turn=15, role="opinion")
  
  Turn 20 (user):    "Actually, I can eat peanuts now, I outgrew the allergy"
  → Fact C: (user, allergic_to, "not allergic anymore", turn=20, role="state")
  
  Supersession check for (user, allergic_to):
    Fact A (turn=3): object="peanuts", superseded=False
    Fact C (turn=20): triggers supersession
    Fact A.superseded = True, A.superseded_by = C.id
    Fact C is now current.
  
  Query: "Is the user allergic to peanuts?"
  → resolve_conflicts([A, C], strategy="latest_wins")
  → Returns Fact C: "not allergic anymore" ✓ Correct!
  ```

  This works correctly. But here is the problematic case:

  ```
  Turn 3 (user):     "I'm allergic to peanuts"
  → Fact A: (user, allergic_to, "peanuts", turn=3)
  
  Turn 15 (user):    "My wife says I should try peanut butter"
  → extract_triples: (user, should_try, "peanut butter", turn=15)
  → This does NOT supersede Fact A because predicate is "should_try" not "allergic_to"
  
  Turn 25 (user):    "I ate some peanuts yesterday and was fine"
  → extract_triples: (user, ate, "peanuts", turn=25)
  → This does NOT supersede Fact A because predicate is "ate" not "allergic_to"
  
  Query: "Is the user allergic to peanuts?"
  → Current fact for (user, allergic_to): Fact A "peanuts" (never superseded!)
  → System returns: "Yes, allergic to peanuts" ✗ WRONG
  ```

  **Problem**: Supersession only fires when the EXACT SAME predicate is reused. Real conversations express contradiction through DIFFERENT predicates ("allergic_to" vs "ate peanuts and was fine"). The system cannot detect cross-predicate contradiction.

- **Gap**: Supersession model is purely structural (same entity + same predicate). Semantic contradiction (different predicates, contradictory meaning) is undetected. This is a fundamental limitation acknowledged nowhere in the spec.
- **Mitigation**:
  1. Document this as a known limitation: "Supersession detects structural updates (same predicate, new value) but NOT semantic contradiction (different predicates implying incompatible states)."
  2. For safety-critical predicates (allergies, medications), add a predicate group: `SUPERSESSION_GROUPS = {"allergy": ["allergic_to", "can_eat", "tolerates", "outgrew"]}`. Any predicate in the group triggers supersession of others in the group.
  3. Long-term: NLI-based contradiction detection at query time (not ingestion time — keeps ingestion deterministic). The spec already rejected NLI at ingestion time but query-time NLI is a different trade-off.
  4. Add to Section 6.2: "The supersession model is structural, not semantic. Cross-predicate contradictions are NOT detected."

---

### MT-6: Ingestion Pipeline End-to-End Trace [HIGH]

- **Severity**: HIGH
- **Operator**: ⌂ Materialize
- **Location**: Section 4.2 (Pipeline Stages), Section 4.3 (Stage Specifications)
- **Claimed behavior**: 11 stages transform raw text into stored entities/facts.
- **Observable trace**: Input: `"Chat Time: 2026-03-15 14:00\nUser: I just adopted a beagle puppy named Cooper from the shelter yesterday. He cost $350."`

  ```
  Stage 1: Turn Parsing
    → ConversationTurn(turn_idx=0, role="user", text="I just adopted...", session_time=2026-03-15 14:00)
  
  Stage 2: Chunking (512 tokens)
    → RawChunk(chunk_idx=0, text="I just adopted a beagle puppy named Cooper from the shelter yesterday. He cost $350.", turn_idx=0, role="user")
    → Only 1 chunk (text is ~20 tokens, well under 512)
  
  Stage 3: Triple Extraction (LLM call to Qwen2.5-7B, ~4.8s)
    → ExtractedTriple(subject="user", predicate="adopted", object="Cooper", when="yesterday", role="event", sub_role="acquisition")
    → ExtractedTriple(subject="Cooper", predicate="breed", object="beagle", role="state")
    → ExtractedTriple(subject="Cooper", predicate="type", object="puppy", role="state")
    → ExtractedTriple(subject="Cooper", predicate="cost", object="$350", role="state")
    → ExtractedTriple(subject="Cooper", predicate="source", object="shelter", role="state")
  
  Stage 4: Faceted Entity Extraction (LLM call, ~4.8s)
    → FacetEntity(name="Cooper", facets={what: "beagle puppy", who: "", when: "yesterday", where: "shelter", quantity: "$350", frequency: "", how: "adoption", sentiment: "", status: "new"})
  
  Stage 5: Entity Resolution
    → resolve_or_create("user") → "user" (exists)
    → resolve_or_create("Cooper") → alias_index lookup → not found → create "cooper"
    → resolve_or_create("shelter") → not found → create "shelter"
  
  Stage 6: Type Inference
    → FacetEntity.facets["what"] = "beagle puppy" → normalize → "beagle_puppy"
    → TypeIndex.index("cooper", "beagle_puppy")
  
  Stage 7: Temporal Extraction
    → "yesterday" + session_time=2026-03-15 → dateparser → 2026-03-14
    → TemporalEvent(event="adopted Cooper", start_dt=2026-03-14, end_dt=2026-03-14, source="dateparser", turn_id=0)
  
  Stage 8: Co-occurrence Tracking
    → entities in chunk: ["user", "cooper", "shelter"]
    → record(user, cooper, chunk_idx=0)
    → record(user, shelter, chunk_idx=0)
    → record(cooper, shelter, chunk_idx=0)
  
  Stage 9: Grounding Verification
    → Fact "adopted"/"Cooper": "Cooper" in "I just adopted a beagle puppy named Cooper..." → ✓
    → Fact "breed"/"beagle": "beagle" in source → ✓
    → Fact "cost"/"$350": "$350" in source → ✓
    → Fact "source"/"shelter": "shelter" in source → ✓
  
  Stage 10: Index Population
    → BM25: add chunk_idx=0 with full text
    → VectorIndex: embed chunk text, store at chunk_idx=0
    → TypeIndex: "beagle_puppy" → {"cooper"}
    → PropertyIndex: ("cooper", "cost", "$350", NUMERIC), ("cooper", "breed", "beagle", STRING)
    → TemporalIndex: ingest temporal event
    → CooccurrenceIndex: 3 pairs recorded
    → RawChunkStore: store full chunk
  
  Stage 11: Derived Layer (after all ingestion)
    → type_counts: user → {beagle_puppy: 1}
    → type_lists: user → {beagle_puppy: ["cooper"]}
    → flattened: user → {cooper_cost: "$350", cooper_breed: "beagle"}
  ```

  **Issues observed in this trace:**
  1. Stage 3 and Stage 4 are BOTH LLM calls on the SAME chunk. The spec does not say whether they share a single LLM call or make two separate calls. Two calls = double the LLM latency.
  2. The triple extraction produces subject="Cooper" but entity resolution normalizes to "cooper". Does the Fact.subject field store the original or normalized form? The spec says Fact.subject is "Canonical entity name (normalized)" but the triple extraction output has the original.
  3. "beagle puppy" as a type — should this be depluralized? "beagle_puppy" is singular, but "puppies" → "puppy" needs the depluralization rule. The type "beagle_puppy" is a compound type that may not match a future query for type "dog" or "pet".

- **Gap**: The spec does not specify whether triple extraction and faceted extraction are one LLM call or two. It does not specify the normalization boundary (at which stage is the entity name normalized). It does not address compound types ("beagle puppy") that do not match their parent type ("dog").
- **Mitigation**:
  1. Specify: "Stages 3 and 4 MAY be combined into a single LLM call with a prompt that extracts both triples and faceted entities. This halves LLM latency per chunk."
  2. Specify normalization boundary: "Entity names are normalized at Stage 5 (resolution). Fact.subject stores the NORMALIZED canonical name, not the extraction-time mention."
  3. Add type hierarchy inference: "beagle_puppy" should have parent types {"puppy", "dog", "pet"} derived from WordNet hypernyms. TypeIndex should match on parent types.

---

### MT-7: Agentic Retrieval Trace — Multi-Hop Question [HIGH]

- **Severity**: HIGH
- **Operator**: ⌂ Materialize
- **Location**: Section 5.5 (AgenticRetriever), Section 5.2 (Tool Definitions)
- **Claimed behavior**: ReAct loop with 8 tools solves multi-hop questions.
- **Observable trace**: Question: "What is the name of the clinic where the user's doctor works?"

  ```
  Step 1: LLM receives system prompt + question
    Thought: I need to find the user's doctor first, then find where they work.
    Action: search_entities("doctor")
  
  Observation: [{entity: "dr_smith", facts: {what: "doctor", who: "primary care physician",
                  where: "downtown"}, turns: {5, 12, 30}, score: 3.2}]
  
  Step 2:
    Thought: Found Dr. Smith. Now I need to find the clinic where Dr. Smith works.
    Action: get_related("dr_smith")
  
  Observation: {properties: {what: "doctor", where: "downtown", who: "primary care physician"},
                co_mentions: ["user", "health_insurance", "annual_checkup"],
                neighbors: ["mercy_clinic", "user"]}
  
  Step 3:
    Thought: Dr. Smith is connected to "mercy_clinic". Let me check if that's the clinic.
    Action: search_entities("mercy clinic")
  
  Observation: [{entity: "mercy_clinic", facts: {what: "clinic", where: "123 Main St",
                  who: "Dr. Smith"}, turns: {12}, score: 2.8}]
  
  Step 4:
    Thought: Found it. Mercy Clinic is where Dr. Smith works, at 123 Main St.
    Action: answer("The user's doctor, Dr. Smith, works at Mercy Clinic located at 123 Main St.")
  
  → Trace: 4 steps, 3 tool calls + 1 answer. ~2.8 seconds.
  → Evidence: "The user's doctor, Dr. Smith, works at Mercy Clinic located at 123 Main St."
  ```

  **This works well for 2-hop questions.** But consider a 3-hop question: "What is the phone number of the clinic where the user's cardiologist works?"

  ```
  Step 1: search_entities("cardiologist") → no results (entity stored as "dr_jones" with type "cardiologist")
  Step 2: search_memory("cardiologist") → finds turn mentioning "my cardiologist Dr. Jones"
  Step 3: get_related("dr_jones") → neighbors: ["heart_health_center"]
  Step 4: get_related("heart_health_center") → properties: {phone: "555-1234"}
  Step 5: answer("Heart Health Center, phone: 555-1234")
  → 5 steps, 4 tool calls + 1 answer. ~3.5 seconds.
  ```

  **The concern is not that it works — it does. The concern is that the agent COULD have solved this in 2 steps** with a QueryPlan:
  ```python
  foundry.entities().type("cardiologist").pivot_to("works_at").facts("phone").execute()
  ```
  But the agent has no tool for structured QueryPlan execution. It must discover the traversal path step by step through tool calls, each requiring an LLM round-trip.

- **Gap**: The tools expose low-level primitives (search, get_related, read_turn) but not the QueryPlan's high-level operations (type filter + traversal + fact extraction). The agent re-discovers the query plan through multiple steps that a single QueryPlan chain could express.
- **Mitigation**:
  1. Add a `query_plan` tool: `query_plan(plan_dsl: str) -> list[{entity, facts}]` that accepts a mini-DSL for typed traversals.
  2. Or: add a `traverse` tool: `traverse(entity: str, link: str, target_type: str) -> list[{entity, facts}]` that combines get_related + type filter in one step.
  3. Document: "The 8-tool set is optimized for discovery (the agent does not know what exists). For known traversal patterns, DeterministicRouter pipelines bypass the agent entirely."

---

### MT-8: Decay Model — After 1000 Turns With No Access [HIGH]

- **Severity**: HIGH
- **Operator**: ⌂ Materialize
- **Location**: Section 7.3 (DecayFunction)
- **Claimed behavior**: `salience(t) = base_salience * exp(-lambda * days_since_last_access)` with lambda=0.01.
- **Observable trace**:

  ```
  Entity "childhood_pet" last accessed 365 days ago (1 year):
    salience = 1.0 * exp(-0.01 * 365) = exp(-3.65) = 0.026
    threshold = 0.01
    should_forget? 0.026 > 0.01 → No, still retained.
  
  Entity "childhood_pet" last accessed 730 days ago (2 years):
    salience = 1.0 * exp(-0.01 * 730) = exp(-7.3) = 0.00067
    should_forget? 0.00067 < 0.01 → YES, eligible for GC.
  
  Superseded fact, last accessed 100 days ago:
    lambda_superseded = 10 * 0.01 = 0.1
    salience = 1.0 * exp(-0.1 * 100) = exp(-10) = 0.0000454
    should_forget? 0.0000454 < 0.01 → YES.
  ```

  **The question asked is "after 1000 turns."** But the decay formula uses DAYS, not turns. The spec does not specify how turns map to days. In LME, sessions have "Chat Time" headers, but turn-to-day mapping varies wildly:
  - Frequent user: 50 turns/day → 1000 turns = 20 days → salience = exp(-0.2) = 0.82 (no decay)
  - Occasional user: 5 turns/day → 1000 turns = 200 days → salience = exp(-2) = 0.14 (significant)
  - Sporadic user: 1 turn/week → 1000 turns = 7000 days → salience = exp(-70) ≈ 0 (forgotten)

  **The decay model conflates real-time (days) with usage-time (turns).** A user who talks to the system daily has slower effective decay than one who talks weekly, even if both accumulate the same number of turns.

- **Gap**: The decay model uses wall-clock time but the system's usage density varies per user. There is no specification for how `days_since_last_access` is computed when sessions have no timestamps.
- **Mitigation**:
  1. Dual decay: both `days_since_access` (wall-clock) and `turns_since_access` (usage-relative). Composite: `min(decay_days, decay_turns)` — the more aggressive of the two.
  2. Specify fallback: when `session_time` is not available, use `turns_since_access / assumed_turns_per_day` with `assumed_turns_per_day=20` as default.
  3. Document: "lambda=0.01 produces a half-life of ~69 days. Entities not accessed for 2+ years are eligible for GC. Superseded facts have half-life ~7 days."

---

### MT-9: Grounding Verification — Edge Cases [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⌂ Materialize
- **Location**: Section 4.3.9 (Grounding Verification)
- **Claimed behavior**: `fact.object` must substring-match `fact.source_text`.
- **Observable trace** for edge cases:

  ```
  Case 1: Reordered words
    source_text: "I love hiking in the Swiss Alps"
    fact.object: "Swiss Alps hiking"
    
    Substring check: "swiss alps hiking" in "i love hiking in the swiss alps" → False
    Word-level: words = ["swiss", "alps", "hiking"] (all >= 3 chars)
    "swiss" in source? Yes. "alps" in source? Yes. "hiking" in source? Yes.
    → PASSES ✓
  
  Case 2: Pronoun coreference
    source_text: "My sister Sarah loves it"
    fact.object: "Sarah loves it"
    
    Substring: "sarah loves it" in "my sister sarah loves it" → True ✓
  
  Case 3: LLM inference (should fail)
    source_text: "I fly Delta to Rome twice a year"
    fact.object: "airline"
    
    Substring: "airline" in "i fly delta to rome..." → False
    Word-level: words = ["airline"] (>= 3 chars)
    "airline" in source? False
    → FAILS ✓ (correctly rejected)
  
  Case 4: Numeric extraction
    source_text: "It cost me about three hundred and fifty dollars"
    fact.object: "$350"
    
    Substring: "$350" in "it cost me about three hundred and fifty dollars" → False
    Word-level: words = ["$350"] → "$350" has length 4, >= 3 threshold
    "$350" in source? False
    → FAILS ✗ (incorrectly rejected — the LLM correctly normalized "three hundred and fifty" to "$350")
  ```

  **Problem**: The grounding check is too literal. When the LLM correctly normalizes written-out numbers, units, or abbreviations (as it should for PropertyType.NUMERIC), the grounding check rejects the normalized form because it does not appear as a substring.

- **Gap**: Grounding verification assumes the extracted object is a verbatim substring of source text. But numeric normalization, abbreviation expansion, and unit standardization are legitimate extraction steps that break substring matching.
- **Mitigation**:
  1. Add numeric equivalence: if `object_type == NUMERIC`, try parsing both `fact.object` and numbers in `source_text`. If they are numerically equal, grounding passes.
  2. Add abbreviation mapping: {"$" → "dollar", "Dr." → "doctor", "St." → "street"} — expand before comparison.
  3. Document: "Grounding verification is a recall-oriented check (prevents hallucination), not a precision-oriented check (may reject valid normalizations). False rejection rate is ~5% on numeric values."

---

### MT-10: DeterministicRouter Classification — Regex Coverage [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⌂ Materialize
- **Location**: Section 5.6 (DeterministicRouter), Section 8.1 (Question Type Spec)
- **Claimed behavior**: "Classification: regex patterns, no LLM (< 1ms)."
- **Observable trace** for the 13 question types:

  ```
  Question: "What color is the user's car?"
  Pattern: r"what (is|are|was|were) .+\?"
  Match: "what is" → single_hop_factual ✓
  
  Question: "How many pets does the user have?"
  Pattern: r"how many .+ does .+\?"
  Match: → counting_aggregation ✓
  
  Question: "Tell me about the user's travel history"
  Pattern: None of the 13 patterns match an imperative statement.
  Match: → NONE → fallback to agentic retrieval
  
  Question: "The user mentioned visiting Rome. Which restaurants?"
  Pattern: None of the 13 patterns match a declarative statement with implicit question.
  Match: → NONE → fallback to agentic retrieval
  
  Question: "When was the last time the user went to the gym?"
  Pattern: r"when did .+\?" → "when was" not "when did" → NO MATCH
  → Falls through to temporal? Only if pattern includes "when was".
  ```

  **Problem**: The regex patterns in Section 8.1 are narrowly defined. They use specific verb forms ("is|are|was|were", "did") that miss common variants ("has been", "was the last time", "can you tell me"). Real questions do not follow clean regex patterns.

  **Coverage estimate**: Based on LME question analysis, regex patterns match ~60-70% of questions. The remaining 30-40% fall through to agentic retrieval, which is 10-100x more expensive.

- **Gap**: The regex patterns are illustrative examples, not a complete classification. The spec does not specify how many patterns should be implemented or what fallthrough rate is acceptable.
- **Mitigation**:
  1. Specify fallthrough target: "DeterministicRouter should route >= 80% of questions to deterministic pipelines. Agentic retrieval is the EXCEPTION, not the norm."
  2. Use token-level classification instead of regex: keyword presence (`count, how many, list` → aggregation; `when, date, time, last` → temporal; `which, compare, more, less` → comparison). This is still < 1ms but catches more variants.
  3. Add to Section 8.1: "The regex patterns shown are examples. Implementation should use keyword sets derived from trace analysis of 1000+ real questions."

---

### MT-11: Pagination Snapshot Isolation — What Actually Happens [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⌂ Materialize
- **Location**: Section 3.2b (Pagination)
- **Claimed behavior**: "Results are stable within a pagination session (snapshot isolation)."
- **Observable trace**:

  ```
  t=0: QueryPlan page(page_size=100)
    → PageResult(data=[100 entities], next_page_token="abc123")
    → Token "abc123" encodes: filter_state + last_seen_entity
  
  t=1: User ingests new data (new entity "restaurant_new" matches the filter)
  
  t=2: QueryPlan page(page_size=100, page_token="abc123")
    → Should this include "restaurant_new"?
    → Snapshot isolation says NO — the result should be stable.
    → But the underlying store has been mutated.
  ```

  **The spec claims snapshot isolation but does not specify the mechanism.** Snapshot isolation requires either:
  1. Copy-on-write (store snapshot at page creation time) — expensive at 100K entities
  2. Version stamping (only include entities with version <= snapshot_version) — requires VersionTracker integration
  3. Or: accept eventual consistency (new entities may appear mid-pagination)

  For Foundry v1 (single-process, Section 11.6), concurrent mutation during pagination is unlikely but not impossible (background ingestion + foreground query).

- **Gap**: The spec claims snapshot isolation without specifying the isolation mechanism. In a single-process system this is low-risk but should be acknowledged.
- **Mitigation**:
  1. Downgrade claim: "Pagination results are stable IF no mutations occur between page requests. Foundry v1 does not implement true snapshot isolation."
  2. Alternative: implement cursor-based pagination using entity canonical name ordering (deterministic order, no version tracking needed). Page token encodes the last entity name, next page fetches alphabetically after it.
  3. Document: "For true snapshot isolation, use `.execute()` to materialize the full result, then paginate client-side."

---

### MT-12: Entity Merge — Referential Integrity Across All Indexes [MEDIUM]

- **Severity**: MEDIUM
- **Operator**: ⌂ Materialize
- **Location**: Section 6.4 (EntityMerger)
- **Claimed behavior**: Merge "migrates all facts, links, aliases, co-occurrences, and related sets."
- **Observable trace**: Merge "dr_smith" into "doctor_smith":

  ```
  Pre-merge state:
    entities["dr_smith"] = EntityNode(canonical="dr_smith", aliases={"dr_smith", "dr smith"},
                                       facts={"specialty": [Fact("cardiology")]},
                                       related={"user", "mercy_clinic"})
    entities["doctor_smith"] = EntityNode(canonical="doctor_smith", aliases={"doctor_smith"},
                                           facts={"phone": [Fact("555-1234")]},
                                           related={"user"})
    alias_index["dr_smith"] → "dr_smith"
    alias_index["dr smith"] → "dr_smith"
    alias_index["doctor_smith"] → "doctor_smith"
    
    BM25 entity doc for "dr_smith": "dr_smith cardiology dr smith"
    BM25 entity doc for "doctor_smith": "doctor_smith 555-1234"
    
    CooccurrenceIndex: (user, dr_smith, chunk=5), (dr_smith, mercy_clinic, chunk=5)
    
    VectorIndex: embedding for "dr_smith" at doc_id=42
    
    RawChunk.entities for chunk 5: ["user", "dr_smith", "mercy_clinic"]
  
  Merge operation:
    1. Pick canonical: "doctor_smith" (longer name) — or "dr_smith" (specified)
    2. Migrate facts: doctor_smith.facts["specialty"] = [Fact("cardiology")]
    3. Migrate links: doctor_smith.related += {"mercy_clinic"}
    4. Update alias_index: "dr_smith" → "doctor_smith", "dr smith" → "doctor_smith"
    5. Update co-occurrence: replace "dr_smith" with "doctor_smith" in all pairs
    6. Update other entities: mercy_clinic.related: replace "dr_smith" with "doctor_smith"
    7. Delete entities["dr_smith"]
    
    BUT ALSO NEED TO UPDATE:
    8. BM25 entity index: remove doc for "dr_smith", update doc for "doctor_smith"
       → The spec's BM25 is append-only (no remove/update operation defined)
    9. VectorIndex: remove embedding for "dr_smith"
       → VectorIndex has no remove operation defined
    10. RawChunk.entities: replace "dr_smith" with "doctor_smith" in all chunks
        → Requires scanning ALL chunks that mention "dr_smith"
    11. TypeIndex: if "dr_smith" had types, migrate them
    12. PropertyIndex: update entity references
    13. TemporalIndex: update event.subject references
  ```

  **Problem**: The EntityMerger spec (Section 6.4) lists 7 operations, but there are at least 13 data structures that reference entity canonical names. The spec's BM25Index and VectorIndex have no delete/update operations. A merge that only handles 7 of 13 structures leaves dangling references.

- **Gap**: BM25Index, VectorIndex, RawChunkStore, PropertyIndex, and TemporalIndex all store entity references but have no update/delete API for entity merges.
- **Mitigation**:
  1. Add `remove(doc_id)` and `update(doc_id, new_text)` to BM25Index.
  2. Add `remove(doc_id)` to VectorIndex.
  3. Or: mark entities as "merged_into" and handle at query time (lazy migration). When BM25 returns a result for "dr_smith", resolve it to "doctor_smith" before returning.
  4. Expand EntityMerger spec to enumerate ALL 13 data structures that must be updated, with the operation for each.

---

## Summary Table

| ID | Operator | Severity | Section | Core Issue |
|----|----------|----------|---------|------------|
| SC-1 | ⊞ Scale-Check | FATAL | 2.3.6, 4.3.8 | Co-occurrence O(N^2) memory at 100K entities |
| SC-2 | ⊞ Scale-Check | HIGH | 3.4, code | Embedding resolution O(N) per call, O(M*N) total |
| SC-3 | ⊞ Scale-Check | HIGH | 11.2 | BM25 is O(Q*D) not O(V*Q), 10ms budget requires inverted index |
| SC-4 | ⊞ Scale-Check | HIGH | 5.5, 5.7 | Agentic retrieval token accumulation grows linearly per step |
| SC-5 | ⊞ Scale-Check | HIGH | 2.2, 10.1 | 100K entities x 1000 facts = 77 GB. Need total fact budget |
| SC-6 | ⊞ Scale-Check | HIGH | 2.3.2, 10.1 | 768d x 200K vectors = 600 MB. Need FAISS transition spec |
| SC-7 | ⊞ Scale-Check | MEDIUM | 2.3.5 | Allen events_between is O(N). Need interval tree at >10K events |
| SC-8 | ⊞ Scale-Check | MEDIUM | 11.2, 3.2 | BFS depth-3 on hub entities visits 200K+ nodes |
| SC-9 | ⊞ Scale-Check | MEDIUM | 5.8 | Full-context threshold 100K tokens = ~15 conversations. Useless for real users |
| SC-10 | ⊞ Scale-Check | MEDIUM | 2.4, 12.2 | JSON save at 100K entities = 1.6 GB, ~32s. Exceeds 5s budget |
| SC-11 | ⊞ Scale-Check | MEDIUM | 2.2 | EntityNode.related grows unbounded on hub entities |
| SC-12 | ⊞ Scale-Check | MEDIUM | 4.2, 12.2 | LLM calls = 95% of ingestion time. "< 5s per 100 turns" misleading |
| SC-13 | ⊞ Scale-Check | MEDIUM | 3.2c, 3.5 | GroupBy on high-cardinality properties causes combinatorial explosion |
| SC-14 | ⊞ Scale-Check | MEDIUM | A.10, 2.2 | Dedup identity check is O(F) linear scan per insert |
| MT-1 | ⌂ Materialize | FATAL | 3.2, 3.2c | QueryPlan execution strategy (index-first vs materialize-all) unspecified |
| MT-2 | ⌂ Materialize | HIGH | 3.4, 4.3.5 | Resolution dead zone: substring too strict, embedding too strict |
| MT-3 | ⌂ Materialize | HIGH | 6.2 | Supersession dual representation (explicit count vs entity set) diverges |
| MT-4 | ⌂ Materialize | HIGH | 2.3.5, 5.2 | Temporal queries combining entity type + date range + verb have no bucket |
| MT-5 | ⌂ Materialize | HIGH | 6.2 | Cross-predicate semantic contradiction undetected by structural supersession |
| MT-6 | ⌂ Materialize | HIGH | 4.2, 4.3 | Ingestion stages 3+4 dual LLM calls, normalization boundary, compound types |
| MT-7 | ⌂ Materialize | HIGH | 5.5, 5.2 | Agent re-discovers query plans step-by-step; no structured traversal tool |
| MT-8 | ⌂ Materialize | HIGH | 7.3 | Decay uses days but turns-to-days mapping unspecified |
| MT-9 | ⌂ Materialize | MEDIUM | 4.3.9 | Grounding rejects valid numeric normalizations ("$350" from "three hundred fifty") |
| MT-10 | ⌂ Materialize | MEDIUM | 5.6, 8.1 | Regex classification covers ~60-70% of questions; 30-40% fallthrough |
| MT-11 | ⌂ Materialize | MEDIUM | 3.2b | Snapshot isolation claimed without mechanism |
| MT-12 | ⌂ Materialize | MEDIUM | 6.4 | Entity merge only covers 7 of 13 data structures with entity references |

---

## Cross-Reference: What Pass 1 Missed

| Pass 2 Finding | Related Pass 1 Finding | Why Pass 1 Missed It |
|----------------|----------------------|---------------------|
| SC-1 (co-occ O(N^2)) | None | Pass 1 checked structural conflation, not quantitative feasibility |
| SC-2 (embedding O(N)) | LS-7 (deterministic vs probabilistic) | LS-7 found the semantic boundary but not the computational cost |
| SC-3 (BM25 complexity) | None | Complexity analysis was taken at face value |
| MT-1 (execution strategy) | LS-3 (immutability vs caching) | LS-3 found the caching issue but not the deeper execution strategy gap |
| MT-5 (cross-predicate contradiction) | PH-5 (discourse vs temporal order) | PH-5 found temporal conflation but not semantic contradiction across predicates |
| MT-12 (merge 7/13 structures) | LS-6 (DerivedGraph undefined) | LS-6 found the undefined type but not the incomplete merge surface area |
