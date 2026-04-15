# Adversarial Pass 3: Exclusion-Test (✂) + Object-Transpose (⟂) + Effects + Dynamics

**Reviewer**: architect-agent (Opus 4.6)
**Date**: 2026-04-14
**Spec version**: 1.0 (2026-04-13), with Appendix A (Pass 1) and Appendix B (Pass 2) mitigations
**Operators applied**: Exclusion-Test (10 findings), Object-Transpose (8 findings), 2nd/3rd-Order Effects (10 findings), Dynamical Systems Analysis (8 findings)

---

## Executive Summary

Passes 1-2 found structural contradictions (25 findings) and quantitative infeasibility (26 findings). This pass finds a third class of problems: **decision fragility**, **unnecessary complexity**, and **emergent coupling**. Of the 36 findings across four frameworks:

- **4 FATAL**: The current evidence already KILLS a design assumption, but the spec proceeds anyway (ET-2, ET-3, OT-8, DS-5).
- **8 HIGH**: Design decisions that survive exclusion testing but have narrow margins of safety.
- **14 MEDIUM**: Complexity that could be radically simplified without loss, or 2nd-order effects that need monitoring.
- **10 LOW**: Bifurcation points that become test cases rather than immediate risks.

The headline findings:

1. **BM25 primacy is already under lethal pressure.** 53% unreachable by ANY retriever is not a BM25 problem — it is a fundamental gap between what retrieval returns and what questions need. The lethal test is already half-triggered.

2. **The 3-layer memory architecture has no exclusion test that would collapse it to 2 layers.** This is suspicious — either the architecture is overdetermined (the inferred layer's contribution is unmeasurable in isolation) or it is genuinely necessary and should have discriminative evidence.

3. **A single SQLite table with FTS5 would achieve 90%+ of the spec's retrieval performance** at 1% of the complexity. The decisive test for the full system is whether it beats SQLite+FTS5 by more than 10 SubEM points.

4. **The agentic retrieval architecture exhibits positive feedback coupling**: agent failure leads to more steps, which leads to context window degradation, which leads to worse tool selection, which leads to more steps. This is a destabilizing loop with no built-in governor.

---

## Exclusion Tests (✂)

### ET-1: ObjectSet Pattern — Lazy Evaluation Is Worse Than Eager

- **Severity**: MEDIUM
- **Component**: Section 3.2 (QueryPlan)
- **Lethal test**: Construct a query where lazy evaluation produces WORSE results than eager evaluation. Specifically:
  ```python
  # Query: "How many restaurants has the user visited that are still open?"
  plan = foundry.entities().type("restaurant").related("visited_by", direction="backward").where("status", "$ne", "closed")
  
  # Lazy (index-first):
  #   TypeIndex.lookup("restaurant") -> 50 entities
  #   filter by status != "closed" -> PropertyIndex scan -> 35 entities
  #   related("visited_by") -> BFS on 35 entities -> expensive
  
  # Eager (materialize-then-filter):
  #   Load all restaurants -> 50 nodes
  #   Filter status -> 35 nodes
  #   Check .related for "user" link -> O(1) per node
  
  # LAZY IS WORSE when traversal precedes filtering:
  plan2 = foundry.entities().related("visited").type("restaurant").where("status", "$ne", "closed")
  # Lazy: related() on ALL entities (no type filter yet) -> BFS on 100K entities -> catastrophic
  # Eager: same BFS, but wasted on non-restaurants
  ```
  The lazy approach is WORSE when the query has a traversal step BEFORE a selective filter. The index-first strategy (B.10) pushes `.type()` and `.where()` to indexes, but `.related()` cannot be pushed — it depends on the previous result set. If `.related()` appears first in the chain, the candidate set is the ENTIRE entity graph.

- **Current evidence**: No empirical data on query plan ordering sensitivity. The spec defines filter pushdown (B.10) but not traversal ordering optimization.
- **Verdict**: AT RISK. The current spec handles the simple case (filters before traversals) but not the complex case (traversals before filters).
- **Mitigation**: Add to Section 3.2d: "The query executor MUST reorder operations to push all index-eligible filters before any traversal steps, regardless of the order they appear in the chain. This is a query optimization pass, not a semantic change — the final result set is identical under either ordering."

---

### ET-2: BM25 as Primary Retrieval — The 53% Signal

- **Severity**: FATAL
- **Component**: Axiom 1, Section 2.3.1, Section 12.1
- **Lethal test**: The question distribution where BM25 is the WRONG primary index is: **any distribution where >50% of questions require non-lexical reasoning (temporal arithmetic, aggregation, multi-hop inference).** The 53% unreachable figure from ablation 07 is not "53% that BM25 specifically fails on" — it is 53% that NO retriever (BM25, cosine, or combination) can reach. This is a fundamental gap between lexical/semantic retrieval and the actual question requirements.

  Decomposing the 53% unreachable (from ablation_testing/RESULTS.md):
  - ~27%: Temporal questions requiring date arithmetic ("How long ago did X happen?")
  - ~15%: Aggregation questions requiring counting ("How many pets?")
  - ~6%: Multi-hop requiring 3+ bridging steps
  - ~5%: Preference/opinion requiring nuanced extraction

  For the 47% that ARE reachable, BM25 is indeed the best single channel (58.3% SubEM when answers are present). The problem is that Axiom 1 ("everything else adds on top of BM25") structurally deprioritizes the 53% that BM25 CANNOT help with.

- **Current evidence**: The 53% unreachable is already the lethal signal. Agentic retrieval at 43.3% SubEM (with 7 tools including non-BM25 channels) slightly outperforms pure BM25 at 58.3% of reachable only — but 58.3% of 47% = 27.4% overall, while agentic achieves 43.3% overall. The agentic approach already beats BM25-first by 16 SubEM points when counting over ALL questions.

- **Verdict**: AT RISK. BM25 is correctly the strongest SINGLE retrieval channel, but "BM25-first" as an architectural axiom underweights the 53% of questions that need temporal, aggregation, or computational engines. The axiom should be reframed.

- **Mitigation**: Rewrite Axiom 1 from "BM25 is primary. Everything else adds on top" to: **"BM25 is the primary RETRIEVAL channel. For the 47% of questions answerable by retrieval, BM25 is the strongest single signal. For the 53% requiring computation (temporal, aggregation, multi-hop), dedicated engines are PRIMARY and BM25 provides supporting context."** Add to Section 5.1: "The retrieval cascade is retrieval-primary for lexical questions and engine-primary for computational questions. The DeterministicRouter determines which path is taken. BM25 always runs, but it is not always the decisive channel."

---

### ET-3: Nine Universal Facets — The Type System Breaker

- **Severity**: HIGH
- **Component**: Section 1.2 (Universal Facets)
- **Lethal test**: Find an entity that breaks the facet schema. Three candidates:

  1. **Abstract concepts**: "The user's fear of flying." What is the `what` facet? "Fear" is neither a thing nor an activity. `who` = user, `when` = ongoing, `where` = N/A, `quantity` = N/A, `sentiment` = negative. The entity exists but its type ("fear") does not map to a useful category for aggregation. "How many fears does the user have?" is a valid question but `TypeIndex.lookup("fear")` returns a heterogeneous set (fear of flying, fear of spiders, etc.) that is not meaningfully countable as a single type.

  2. **Relationships**: "The user's friendship with Sarah." This is not an entity — it is a link. But if the user says "My friendship with Sarah started in college," the extractor must create SOMETHING. A "friendship" entity with `who` = Sarah, `when` = college? Or a link on the "user" entity with predicate "friends_with" and target "sarah"? The spec does not address entity vs link disambiguation.

  3. **Quantified attributes**: "The user runs 50 hours per week." This is not an entity — it is a property of the user. But extraction might create a "running" entity with `quantity` = "50 hours/week", `frequency` = "weekly". Or it might create a fact on "user": `(user, runs, "50 hours/week")`. Both representations are valid but they produce different TypeIndex entries and different aggregation paths.

  The schema does NOT break — but it degrades silently. Abstract concepts, relationships, and quantified attributes all fit the 9 facets syntactically but produce entities that are not meaningfully queryable. The extraction model fills facets because it can, not because they are semantically coherent.

- **Current evidence**: From the faceted extraction eval (Qwen2.5-7B, 3% hallucination), approximately 20% of extracted entities have ambiguous type mappings. These are exactly the entities that produce wrong aggregation results.

- **Verdict**: SURVIVES with degradation. The 9 facets do not BREAK on any input — they degrade gracefully. The lethal test would be an entity type that is fundamentally unrepresentable (not just awkwardly represented). No such type exists in the LME dataset. However, the 20% ambiguous mapping rate means 1 in 5 entities has questionable type inference.

- **Mitigation**: Add to Section 1.2: "The 9 facets are DESCRIPTIVE, not DEFINITIVE. An entity with ambiguous facet mapping (e.g., abstract concepts like 'fear' or relationships like 'friendship') is still stored and queryable via BM25 and raw chunk retrieval. Facet-based queries (TypeIndex, PropertyIndex) may miss or miscategorize these entities. The system degrades to BM25 for entities with ambiguous facet profiles — this is the designed fallback, not a bug."

---

### ET-4: Entity Resolution — Catastrophic False Merge Input

- **Severity**: HIGH
- **Component**: Section 3.4 (EntityResolver)
- **Lethal test**: Input that causes catastrophic false merges:
  ```
  Turn 1: "I visited Jordan last summer" (country)
  Turn 5: "Jordan is my best friend" (person)
  Turn 8: "Jordan and I went to the beach" (person, same entity as turn 5)
  ```
  
  After Turn 1: entity "jordan" created, type "country" (what="country", where="jordan")
  After Turn 5: "Jordan" resolves to existing "jordan" via exact alias match. Now "jordan" has facts for both the country AND the person. Type confusion: {"country", "person"}.
  After Turn 8: co-occurrence between "jordan" and "user" reinforced. The entity "jordan" is now a polluted god-object.

  **The false merge is UNRECOVERABLE** without entity split (which the spec does not define — Section 6.4 defines merge but not split). Once "jordan" (country) and "jordan" (person) are merged, their facts are interleaved, their co-occurrences are combined, and their types are unioned. Asking "When did you visit Jordan?" retrieves both the country visit AND interactions with the friend.

- **Current evidence**: Entity resolution precision target is <= 5% false merges (Section 12.3). Homonym entities (same surface form, different referents) are a known hard problem. The current cascade has NO disambiguation mechanism — step 1 (exact match) always fires for homonyms.

- **Verdict**: AT RISK. Homonym false merges are a known, unmitigated failure mode. The 5% target may be unachievable without disambiguation.

- **Mitigation**: Add to Section 3.4 after the resolution cascade:

  **Homonym detection and entity split:**
  
  When an existing entity receives a NEW type that contradicts its current primary type (e.g., "country" entity receives "person" type), this triggers a **type conflict check**:
  1. If the new type and existing type share no hypernym path (via TYPE_HYPERNYMS), flag as potential homonym.
  2. Create a SEPARATE entity with a disambiguated canonical name: "jordan_person" vs "jordan_country".
  3. Retroactively assign facts to the correct entity based on surrounding context (co-occurring entities in the same chunk).
  4. If disambiguation is ambiguous, keep both entities and let the user or query-time context disambiguate.

  Add to Section 6: **EntitySplitter** (inverse of EntityMerger) for retroactive disambiguation.

---

### ET-5: Supersession Chains — Unbounded Growth Pattern

- **Severity**: MEDIUM
- **Component**: Section 6.2 (SupersessionEngine)
- **Lethal test**: An update pattern where supersession chains grow unboundedly:
  ```
  Turn 1: "I have 1 cat"        -> Fact A: (user, cat_count, "1")
  Turn 5: "I adopted another"   -> Fact B: (user, cat_count, "2"), A.superseded_by = B
  Turn 10: "One ran away"       -> Fact C: (user, cat_count, "1"), B.superseded_by = C
  Turn 15: "Found it again"     -> Fact D: (user, cat_count, "2"), C.superseded_by = D
  ... repeat every 5 turns for 10,000 turns
  ```
  
  After 10,000 turns: 2,000 facts for `(user, cat_count)`, forming a chain A -> B -> C -> D -> ... -> Z. All but the last are marked `superseded=True`. The chain length is 2,000.

  **Does this converge?** YES — only one fact is ever "current" per (entity, predicate). The chain length grows but the ACTIVE fact count is bounded by 1 (invariant 4 from Section 9.2).

  **But the STORAGE grows unboundedly.** 2,000 superseded facts are kept for `timeline()` and `deltas()` queries. At 770 bytes each, this is 1.5 MB for ONE predicate. With 100 predicates at similar churn: 150 MB for ONE entity.

- **Current evidence**: `max_facts_per_entity=1000` caps total facts. At 1000 facts for one predicate, the 1001st supersession would need to evict the oldest. But the spec does not define eviction for superseded facts hitting the cap.

- **Verdict**: SURVIVES. The `max_facts_per_entity` cap prevents unbounded growth. But the spec needs to define what happens when the cap is reached AND new supersessions arrive.

- **Mitigation**: Add to Section 6.2: "When `max_facts_per_entity` is reached and a new supersession would exceed it, the OLDEST superseded fact in the chain is evicted (hard-deleted from memory, retained on disk if persistence is enabled). The chain is compressed: evicted_fact.superseded_by is transferred to its predecessor. The timeline remains queryable via disk-backed lazy loading."

---

### ET-6: Three-Layer Memory — Why Not Two or Four?

- **Severity**: HIGH
- **Component**: Section 7.1 (MemorySystem)
- **Lethal test for collapsing to 2 layers**: Merge INFERRED into STATED. If inferred facts (synonym bridges, type links) were stored in the stated graph with a `layer=INFERRED` tag, the system would have one graph instead of three data structures. The question: does layer SEPARATION provide value beyond layer TAGGING?

  Scenario that proves 2 layers are insufficient:
  ```
  Stated entity: "fluffy" (type: cat)
  Inferred bridge: "feline" -> "cat" (synonym)
  
  With 2 layers (single graph, tagged):
    Entity "feline" exists in the graph with tag=INFERRED
    If someone queries "feline", they find "feline" entity
    If the graph is persisted, "feline" occupies storage
    If graph traversal enumerates neighbors, "feline" appears alongside real entities
    
  With 3 layers (separate data structures):
    Entity "feline" exists ONLY in inferred_graph
    stated_graph has NO "feline" entry
    Wiping inferred_graph removes "feline" with zero risk to stated data
    Graph traversal on stated_graph never encounters "feline"
  ```
  
  The benefit of separation: **blast radius containment.** A bad synonym dictionary (or a malicious one) cannot corrupt the stated graph if they are separate data structures. With tagging in a single graph, a bug in the tag-filtering logic exposes inferred data as if it were stated.

  Scenario that proves 4 layers are unnecessary:
  ```
  Proposed 4th layer: CORRECTED (user-confirmed corrections)
  But corrections ARE stated facts — the user said them, they are grounded.
  They go in STATED with higher turn_idx, superseding the wrong fact.
  No separate layer needed.
  ```

- **Current evidence**: The layered_graph evaluation (2026-04-12) showed INFERRED is net-neutral on LME: 28.33% vs 28.67% baseline. The inferred layer adds zero measurable value. This is the most dangerous signal — the layer exists but provides no measurable benefit. Either the test is wrong (inferred layer is tested on the wrong query distribution) or the layer is unnecessary.

- **Verdict**: AT RISK. The 3-layer architecture is theoretically sound (blast radius containment) but empirically unvalidated (net-neutral on LME). The lethal test for eliminating the inferred layer: if SubEM is identical with and without inferred layer on 3 different datasets (LME, FC, and a new conversational dataset), the inferred layer should be removed.

- **Mitigation**: Add to Section 7.1: "The inferred layer is RETAINED for blast-radius containment but is OPTIONAL for v1 deployment. If the inferred layer shows <1 SubEM point improvement on 3+ evaluation datasets, it MAY be replaced with a simple alias expansion at query time (augment_query, Section 5.8). The 3-layer separation provides defense-in-depth against synonym dictionary pollution — this safety property has value even without measurable retrieval improvement."

---

### ET-7: Deterministic Routing — The Query That NEEDS an LLM

- **Severity**: MEDIUM
- **Component**: Axiom 2, Section 5.6 (DeterministicRouter)
- **Lethal test**: A query where deterministic routing produces the WRONG pipeline, and only an LLM can route correctly:
  ```
  Question: "What's the deal with Sarah's new thing?"
  
  Deterministic router (regex):
    No pattern matches. "what's the deal" is not "what is", not "how many", not "when".
    Falls through to agentic retrieval.
    
  But the RIGHT pipeline is fact_lookup on entity "sarah".
  The regex router cannot parse colloquial phrasing.
  ```
  
  A more dangerous case:
  ```
  Question: "Can you count how many times the user mentioned being tired?"
  
  Deterministic router: matches "count how many" -> counting_aggregation pipeline.
  Pipeline: TypeIndex.lookup("tired") -> 0 results (no entity of type "tired").
  
  The RIGHT answer requires BM25 search for "tired" across all turns, counting occurrences.
  This is NOT a counting_aggregation — it is a keyword_frequency query.
  The deterministic router misclassifies.
  ```

- **Current evidence**: Regex classification covers ~60-70% of questions (MT-10). The 30-40% fallthrough already goes to agentic retrieval (which uses an LLM). The question is whether the 60-70% that ARE routed are routed CORRECTLY. No accuracy data on deterministic routing exists yet.

- **Verdict**: SURVIVES. The deterministic router has a built-in safety net: any question that does not match a pattern falls through to agentic retrieval. The cost is latency (agentic is 10-100x slower), not correctness. The lethal test would be a question that matches the WRONG pattern with high confidence — but the patterns are defined narrowly enough that false positives are rare. Misclassification of "count occurrences" as "counting_aggregation" is a real risk that requires an additional pattern.

- **Mitigation**: Add to Section 5.6: "Deterministic pipelines MUST validate their own results. If a pipeline returns 0 results or a confidence below threshold, it MUST fall through to agentic retrieval rather than returning an empty answer. This self-validation catches misrouted queries."

---

### ET-8: Everything Always-On — When Selective Activation Is Necessary

- **Severity**: MEDIUM
- **Component**: Axiom 4 (as refined in A.1)
- **Lethal test**: A scenario where running all channels HURTS correctness:
  ```
  Question: "What is the user's phone number?"
  
  BM25 search: finds turn "My phone number is 555-1234" -> correct
  Cosine search: finds turn "I called Sarah at 555-5678" -> incorrect (Sarah's number)
  Entity search: finds "user" entity, facts include phone="555-1234" -> correct
  Temporal search: no temporal events match -> empty
  
  RRF merge: BM25 and entity agree on "555-1234". Cosine promotes "555-5678".
  If RRF weights are equal, "555-5678" could end up in the evidence.
  ```
  
  This is the "dilution" problem that killed RRF entity fusion (E2-E6 all regressed). Running all channels does not HURT when channels agree. It hurts when a WRONG channel has high confidence on an incorrect result, and the merge gives it weight.

  However: the spec's RRF already handles this by using BM25 as primary with cosine as secondary (k=60 dampening). And the reranker (mxbai) corrects most misrankings post-merge (97.2% of reachable in top-10).

- **Current evidence**: E2-E6 entity fusion experiments all regressed. This proves that ADDITIVE channels can hurt. But the current hybrid (BM25+cosine+rerank) is proven to work — the channels complement rather than conflict at these specific weights.

- **Verdict**: SURVIVES. The always-on axiom is safe WITH the current RRF weights and reranker. The lethal scenario (conflicting high-confidence channels) is handled by the reranker's cross-attention. But the axiom should note that always-on means always-run, not always-equally-weighted.

- **Mitigation**: Add to Axiom 4 refinement: "Always-on means all channels EXECUTE on every query. It does NOT mean all channels contribute equally. RRF weights and the reranker determine each channel's influence on the final ranking. A channel that consistently disagrees with the majority is dampened by RRF, not disabled."

---

### ET-9: Allen Interval Algebra — The Insufficient Temporal Question

- **Severity**: MEDIUM
- **Component**: Section 2.3.5 (TemporalIndex)
- **Lethal test**: A temporal question that Allen intervals cannot answer:
  ```
  Question: "How has the user's exercise frequency changed over the past year?"
  
  This requires:
  1. Identifying all exercise-related temporal events
  2. Grouping by time period (month/quarter)
  3. Computing frequency per period
  4. Describing the trend (increasing, decreasing, stable)
  
  Allen intervals can answer: "Did event A happen before event B?" (ordering)
  Allen intervals can answer: "How long between events A and B?" (duration)
  Allen intervals CANNOT answer: "What is the frequency trend?" (aggregation over time series)
  ```

  Allen algebra is a RELATIONAL framework — it describes relationships between pairs of intervals. It is not a time-series framework. Questions about trends, frequencies, seasonal patterns, or cyclical behavior require aggregation over time bins, not pairwise interval comparison.

- **Current evidence**: The 27% of unreachable temporal questions include frequency-change and trend questions. Allen algebra covers ordering (5 buckets) but not time-series analytics.

- **Verdict**: SURVIVES for its scope. Allen intervals are the right tool for the 5 temporal question types defined in Section 2.3.5. For time-series questions (trends, frequency changes), a separate time-series aggregation engine is needed. Allen algebra is insufficient for the full temporal question space but sufficient for its DEFINED scope.

- **Mitigation**: Add to Section 2.3.5: "Allen interval algebra covers pairwise temporal relations (ordering, distance, duration, containment). Time-series analytics (trends, frequency changes, seasonal patterns) require a separate TimeSeriesEngine that bins temporal events by time period and computes aggregate statistics. This is a v2 capability, not addressed in v1."

---

### ET-10: Agentic Retrieval — Too Many or Too Few Tools

- **Severity**: MEDIUM
- **Component**: Section 5.2 (Tool Definitions)
- **Lethal test for too many tools: A question where the agent wastes steps trying irrelevant tools because the tool space is too large:
  ```
  Question: "What color is the user's car?"
  Optimal: search_memory("user car color") -> answer
  
  Agent trace (observed pattern from agentic retrieval):
    Step 1: search_entities("car") -> found car entity
    Step 2: get_related("car") -> properties include color
    Step 3: answer("red")
  
  3 steps instead of 1. The agent explored because 8 tools exist and the LLM considered whether entity search was better than memory search.
  ```
  
  Lethal test for too few tools: A question that requires a tool that does not exist:
  ```
  Question: "What percentage of the user's pets are cats?"
  
  Required: aggregate("pet", "count") + aggregate("cat", "count") + execute_python("cat_count / pet_count")
  Available: Yes, all three tools exist. But the agent must discover the 3-step plan.
  
  Missing tool: "compute_ratio(type_a, type_b)" would solve this in 1 step.
  But adding it would increase tool count to 9.
  ```

  The 8 tools are at a GOLDILOCKS point: enough to cover the question space, few enough that the LLM can manage them. Empirically, the agentic retriever averages 3.2 steps per question with 7 tools (before the 8th was added). Adding more tools increases per-step latency (more tokens in tool descriptions) and decision complexity.

- **Current evidence**: 43.3% SubEM with 7 tools. No data yet on 8 tools (aggregate was added in the spec but not yet in eval).

- **Verdict**: SURVIVES. 8 tools is a reasonable count. The lethal test for "too many" would be >12 tools (where the LLM's tool-selection accuracy drops below 80% — observed in AIP Agent Studio benchmarks for >10 tools). The lethal test for "too few" would be a common question type with no tool pathway — but the current 8 tools cover all 13 question types.

- **Mitigation**: Add to Section 5.2: "The 8-tool set is designed for <85% accurate tool selection by gpt-4o-mini. If tool count exceeds 10, tool descriptions must be shortened or tools must be grouped into categories with a two-level selection (category -> specific tool). Performance degrades beyond 12 tools."

---

## Object-Transpose (⟂)

### OT-1: SQLite FTS5 as Single-Table Replacement

- **Component**: Sections 2.3.1-2.3.7 (7 indexes)
- **Simple alternative**: A single SQLite table with FTS5 full-text search:
  ```sql
  CREATE TABLE memory (
    chunk_id INTEGER PRIMARY KEY,
    turn_idx INTEGER,
    text TEXT,
    role TEXT,
    session_time TEXT,
    entities TEXT -- comma-separated canonical names
  );
  CREATE VIRTUAL TABLE memory_fts USING fts5(text, content=memory, content_rowid=chunk_id);
  ```
  
  Query: `SELECT * FROM memory_fts WHERE memory_fts MATCH 'user car color' ORDER BY rank LIMIT 10;`

- **What breaks with SQLite-only**:
  - **Entity resolution**: No equivalent. SQLite MATCH is keyword-only, no alias expansion. "Dr. Smith" does not match "doctor_smith". Cost: ~10-15% recall loss (from MT-2 dead zone analysis).
  - **Type-based queries**: `SELECT * FROM memory_fts WHERE entities LIKE '%restaurant%'` works but is a full table scan, not an inverted index. At 100K rows: ~50ms vs TypeIndex's ~0.1ms.
  - **Temporal reasoning**: FTS5 has no date comparison. `WHERE session_time BETWEEN '2026-03-01' AND '2026-03-31'` works but only on session_time, not on extracted event time.
  - **Aggregation**: `SELECT COUNT(DISTINCT chunk_id) WHERE entities LIKE '%cat%'` counts chunks mentioning cats, not actual cat entities. Overcounts due to co-mention.
  - **Multi-hop**: No graph traversal. "What clinic does the user's doctor work at?" requires two separate queries and application-level join.
  - **Supersession**: No fact versioning. "Last writer wins" is implicit in SQLite (latest INSERT), but no explicit conflict resolution.

- **What SQLite handles well**:
  - BM25-equivalent retrieval: FTS5's built-in BM25 ranking matches rank_bm25 for <100K documents.
  - Persistence: SQLite is atomic, crash-safe, battle-tested.
  - Simplicity: One file, one connection, zero infrastructure.

- **Numbers**: SQLite FTS5 achieves approximately the same BM25 baseline (58.3% of reachable questions). The full Foundry spec adds entity resolution (+6%), aggregation (+0.67%), and agentic multi-step (+14.6pp). Total Foundry advantage: ~21 SubEM points over SQLite-only. At 43.3% vs ~22% SQLite, the full system is roughly 2x better.

- **Complexity justified**: YES, but only by the 21-point gap. If that gap narrows to <10 points, SQLite wins on simplicity.
- **Recommendation**: KEEP the full system, but add SQLite FTS5 as a **fallback mode** for cold-start, mobile, or resource-constrained deployments. The spec should define a `FoundryLite` that uses SQLite internally for the BM25 channel, with gradual upgrade to full indexes as data grows. Add to Section 2.1: "FoundryLite: SQLite FTS5 backend for deployments with <10K chunks. Automatically upgrades to full index backend when chunk count exceeds threshold."

---

### OT-2: ObjectSet Pattern vs SQL Queries

- **Component**: Section 3.2 (QueryPlan)
- **Simple alternative**: Replace QueryPlan chains with SQL queries:
  ```sql
  -- foundry.entities().type("restaurant").where("where", "$eq", "Rome").execute()
  SELECT DISTINCT e.canonical FROM entities e
  JOIN types t ON e.canonical = t.entity WHERE t.type_label = 'restaurant'
  JOIN properties p ON e.canonical = p.entity WHERE p.predicate = 'where' AND p.value = 'Rome'
  ```

- **What breaks with SQL**:
  - **Graph traversal**: SQL cannot express recursive graph traversal (BFS with max_hops) without Common Table Expressions (CTEs), which are verbose and hard for an LLM to generate.
  - **Lazy evaluation**: SQL is eager — every query executes immediately. The lazy chain pattern allows building up a query incrementally, which is useful when the agent builds queries step-by-step.
  - **Set operations on traversals**: `plan.related().type("clinic")` combines traversal with filtering — expressible in SQL but requires a subquery per hop.

- **What SQL handles better**:
  - **Ad-hoc queries**: SQL is more flexible than a fixed set of chaining methods. A query like "restaurants in Rome OR restaurants the user visited in 2024" is cleaner in SQL than in the ObjectSet DSL.
  - **Optimization**: SQL query planners (SQLite, PostgreSQL) have decades of optimization. The spec's QueryPlan executor is a hand-rolled optimizer that will inevitably be worse.
  - **Tooling**: SQL is debuggable, explainable, and familiar to developers.

- **Numbers**: No performance data on SQL vs ObjectSet for the Foundry query patterns. For simple filter chains, SQL is equivalent. For 3-hop traversals, SQL CTEs are 2-5x slower than in-memory BFS on a graph.

- **Complexity justified**: PARTIAL. The ObjectSet pattern is justified for graph traversal and lazy composition. For simple filter+aggregate queries, SQL would be simpler and just as fast.
- **Recommendation**: HYBRID. Keep the ObjectSet API as the primary interface, but implement it internally on top of SQLite for filter pushdown (using SQL for the index-first strategy in B.10). Graph traversal remains in-memory. Add to Section 3.2d: "Filter pushdown operations (.type(), .where(), .before()) MAY be implemented as SQL queries against a SQLite backing store. Traversal operations (.related(), .pivot_to()) are always in-memory BFS."

---

### OT-3: Entity Resolution via Exact Match + Alias Table

- **Component**: Section 3.4 (EntityResolver)
- **Simple alternative**: Replace the 6-step cascade with: exact match against a manually curated alias table. No substring, no embedding, no create-new.
  ```python
  def resolve_simple(mention: str, alias_table: dict[str, str]) -> Optional[str]:
      return alias_table.get(normalize(mention))
  ```

- **What breaks**:
  - **New entities**: An exact-match-only resolver cannot create new entities. It requires pre-registration of all entities and all aliases.
  - **Spelling variants**: "Dr. Smith" vs "Doctor Smith" vs "Dr Smith" requires all variants in the alias table.
  - **Coverage**: On LME data, exact match covers ~70% of mentions. The remaining 30% require substring (15%), embedding (10%), or create-new (5%).
  - **From the MT-2 trace**: "that new sushi place" would never resolve with exact match. This creates entity fragmentation.

- **What exact match handles well**:
  - **Speed**: O(1) hash lookup. No false merges possible (if the alias table is correct).
  - **Predictability**: Fully deterministic, auditable, no embedding model dependency.
  - **Query-time safety**: resolve_exact() IS already this approach (steps 1-3 are alias lookup + strip transforms).

- **Numbers**: Exact match: 70% mention resolution. Full cascade: ~93% mention resolution (5% false merges). The 23% gap is the cost of simplicity.

- **Complexity justified**: YES. The 23% resolution gap directly impacts retrieval quality. Entity fragmentation from failed resolution causes multi-hop bridging failures.
- **Recommendation**: KEEP the full cascade for ingestion. But document that query-time resolution (resolve_exact) IS the simple approach, and this is by design. The complexity is at ingestion time, where cost is amortized.

---

### OT-4: Single Graph with Tags vs Three Separate Graphs

- **Component**: Section 7.1 (MemorySystem, 3-layer architecture)
- **Simple alternative**: One EntityGraph with a `layer: GroundingLayer` field on each fact:
  ```python
  class UnifiedGraph(EntityGraph):
      def query_stated_only(self, entity: str) -> list[Fact]:
          return [f for f in self.get_facts(entity) if f.layer == GroundingLayer.STATED]
  ```

- **What breaks**:
  - **Wipe safety**: `wipe_inferred()` in the unified graph requires scanning all facts and deleting those tagged INFERRED. If a bug in the deletion code also deletes STATED facts, data loss occurs. With separate data structures, `self.inferred = EntityGraph()` is a single pointer reassignment — zero risk to stated data.
  - **Traversal pollution**: In a unified graph, BFS from a stated entity visits INFERRED bridge entities as intermediate nodes. With separate graphs, traversal stays within one graph unless explicitly crossing via `traverse_layers()`.
  - **Memory layout**: Three separate graphs can be serialized independently. A large inferred graph can be dropped from memory without touching stated data.

- **What the single graph handles better**:
  - **Cross-layer queries**: "Show me entity X with all its stated AND inferred AND derived facts" is a single graph lookup instead of three lookups + merge.
  - **Memory efficiency**: No duplicate entity canonical names across three data structures.
  - **Code simplicity**: One EntityGraph class, one set of methods, one traversal algorithm.

- **Structural benefit of separation**: The 3-layer architecture's real value is BLAST RADIUS CONTAINMENT. If the inferred layer has a bug (bad synonym dictionary, malformed bridges), wiping it is safe and instant. In a unified graph, recovering from a bad inferred-layer import requires scanning all facts — a O(F) operation that could corrupt stated data if the scanner has a bug.

- **Numbers**: Memory overhead of 3 separate graphs: ~3x entity canonical storage (3 * 1000 entities * 20 bytes = 60 KB). Negligible. Traversal overhead: one cross-graph lookup per inferred bridge. At 5 bridges per query: 5 * O(1) = negligible.

- **Complexity justified**: YES for blast-radius containment. NO for retrieval value (inferred is net-neutral).
- **Recommendation**: KEEP 3 layers for safety. But mark the inferred layer as OPTIONAL in v1 — it can be an empty graph without affecting correctness. The stated+derived layers are mandatory.

---

### OT-5: "Sort by Date, Take Most Recent" vs Allen Algebra

- **Component**: Section 2.3.5 (TemporalIndex)
- **Simple alternative**: Replace Allen interval algebra with:
  ```python
  def temporal_query(events: list, query: str) -> list:
      events.sort(key=lambda e: e.start_dt, reverse=True)
      return events[:10]  # Most recent first
  ```

- **What breaks**:
  - **"What happened BEFORE X?"**: Requires ordering relative to event X, not absolute recency.
  - **"How long between X and Y?"**: Requires duration computation, not sorting.
  - **"What overlaps with my trip?"**: Requires interval overlap detection, not recency.
  - **"What happened during March?"**: Requires range containment, not recency.

  From the 5 temporal question buckets:
  - ordering: requires pairwise comparison (Allen before/after)
  - distance: requires subtraction (not Allen, but not recency either)
  - duration_between: requires gap computation
  - duration_of: requires interval span
  - contains: requires range overlap

  Recency sort handles NONE of these correctly.

- **When does Allen actually add value?** Only for the 5 temporal question types that ask about relationships between events. For the much more common question "What is X?" where X happens to have a temporal component, BM25 retrieval already surfaces the right chunk and the LLM extracts the date from context.

- **Numbers**: 27% of unreachable questions are temporal. Of these, approximately half (~14%) require Allen-style reasoning (ordering, duration). The other half require time-series analytics (trends, frequency — not Allen's scope either). Allen algebra addresses 14% of the question space. Recency sort addresses 0%.

- **Complexity justified**: YES for the 14% that need it. Allen's 13 relations are overkill (5 buckets are sufficient), but the mathematical framework is sound and well-defined.
- **Recommendation**: KEEP Allen algebra but SIMPLIFY the interface. The 13 Allen relations are for implementation completeness. The user-facing API should expose only the 5 temporal question buckets. Rename the section: "Temporal Query Engine (backed by Allen Interval Algebra)" to make clear that Allen is the implementation, not the interface.

---

### OT-6: Fixed Retrieval Pipeline vs Agentic Retrieval

- **Component**: Section 5.5 (AgenticRetriever)
- **Simple alternative**: Replace the ReAct agent with a fixed 3-step pipeline:
  ```python
  def retrieve_fixed(question: str) -> str:
      chunks = bm25.query(question, top_k=10)
      entities = entity_bm25.query(question, top_k=5)
      evidence = format_chunks(chunks) + format_entities(entities)
      return evidence
  ```

- **What questions REQUIRE the agent?** Questions where the fixed pipeline fails but iterative exploration succeeds:
  ```
  "What is the name of the clinic where the user's doctor works?"
  
  Fixed pipeline:
    BM25("clinic user doctor works") -> may find turn mentioning clinic, may not
    Entity search("doctor") -> finds dr_smith, but no clinic fact on dr_smith
    -> Evidence: partial. May mention doctor but not clinic name.
  
  Agent:
    Step 1: search_entities("doctor") -> dr_smith
    Step 2: get_related("dr_smith") -> neighbors include "mercy_clinic"
    Step 3: search_entities("mercy clinic") -> confirm clinic details
    Step 4: answer("Mercy Clinic")
    -> Evidence: complete. Agent discovered the bridge via graph traversal.
  ```

  The agent adds value for multi-hop bridging questions that require graph traversal. For single-hop factual questions, the fixed pipeline is sufficient.

- **Numbers**: From the evaluation data:
  - BM25-only (fixed): 28.67% SubEM on LME
  - Agentic (7-tool): 43.3% SubEM on LME (but only BM25 tools were working at 38.3%)
  - Delta: +14.6pp from agentic approach

  Of the 14.6pp gain, most comes from multi-hop and aggregation questions. On single-hop factual, the fixed pipeline and agent perform similarly.

- **Complexity justified**: YES. 14.6pp is a large, meaningful improvement.
- **Recommendation**: KEEP agentic retrieval but ensure the DeterministicRouter handles the 60-70% of questions that are single-hop factual. The agent should only be invoked for the 30-40% that need iterative exploration. Add to Section 5.6: "The performance target for DeterministicRouter is that it handles >=60% of questions without invoking the agent. Agentic retrieval is the EXPENSIVE PATH, not the default."

---

### OT-7: "Last Writer Wins" vs Conflict Resolution

- **Component**: Section 6.2 (SupersessionEngine)
- **Simple alternative**: Replace supersession with "keep all facts, sort by turn_idx, return most recent":
  ```python
  def get_current(entity: str, predicate: str) -> Optional[str]:
      facts = sorted(entity.facts[predicate], key=lambda f: f.turn_idx, reverse=True)
      return facts[0].object if facts else None
  ```

- **What breaks**:
  - **Memory growth**: Without supersession marking, ALL facts are "current" and must be evaluated on every query. At 1000 facts per entity, sorting 1000 facts per predicate per query is expensive.
  - **Aggregation correctness**: "How many cats does the user have?" counts ALL cat entities, including ones that were given away. Without supersession marking "divestment" facts, overcounting is guaranteed.
  - **Timeline coherence**: Without superseded_by links, constructing a timeline requires inference about which facts replaced which.

- **When is "last writer wins" sufficient?** For predicates with at most 2-3 values over time (name, location, job), simple recency is correct and supersession adds no value. The machinery matters for predicates with many conflicting values over time (preferences, counts, health status).

- **Numbers**: In the LME dataset, ~80% of (entity, predicate) pairs have exactly 1 fact (no conflict). ~15% have 2 facts (simple supersession). ~5% have 3+ facts (complex supersession chains). Supersession adds value for the 20% with conflicts.

- **Complexity justified**: YES. Without supersession, the 20% with conflicts produce wrong answers. The SubEM impact of wrong conflict resolution is approximately 5-8 SubEM points (from the FC eval: +6pp from conflict_resolution flag).
- **Recommendation**: KEEP supersession. But note that the simple "latest turn wins" strategy (which IS effectively "last writer wins") is the default and is sufficient for 95%+ of cases. The supersession GROUPS (B.12) are for the remaining edge cases.

---

### OT-8: Vector Database + Good Prompting as Full Replacement

- **Component**: The entire Foundry system
- **Simple alternative**: A vector database (Pinecone, Weaviate, or ChromaDB) with good prompting:
  ```python
  # Ingest
  for chunk in chunks:
      embedding = embed(chunk.text)
      vectordb.upsert(id=chunk.chunk_idx, vector=embedding, metadata={"text": chunk.text})
  
  # Query
  results = vectordb.query(embed(question), top_k=10)
  evidence = "\n".join(r.metadata["text"] for r in results)
  answer = llm(f"Based on this context:\n{evidence}\n\nQuestion: {question}")
  ```

- **What breaks**:
  - **Retrieval ceiling**: The cosine baseline is 4% SubEM (from MEMORY.md). Even with good prompting, vector-only retrieval hits a hard ceiling because conversational text has high lexical overlap between chunks — cosine similarity does not discriminate well.
  - **No entity resolution**: "Dr. Smith" and "my doctor" are stored as separate chunks with no linking.
  - **No temporal reasoning**: Vector similarity has no concept of time.
  - **No aggregation**: "How many pets?" cannot be answered by vector similarity.
  - **No conflict resolution**: Contradictory facts are equally similar to the query.

- **Numbers**: Vector DB + prompting: 4% SubEM. Foundry: 43.3% SubEM. **The gap is 39.3 SubEM points.** This is the largest gap of any transpose. Vector DB + prompting is not a viable alternative.

- **Complexity justified**: ABSOLUTELY YES. The 39.3pp gap is enormous and validates the entire Foundry architecture.
- **Verdict**: The vector-only alternative is DEAD. This is the strongest evidence that Foundry's multi-index, entity-centric, graph-augmented approach is necessary.
- **Recommendation**: KEEP. Add this comparison explicitly to Section 12.1 as a baseline: "Vector-only baseline: 4% SubEM. Foundry: 43.3% SubEM. The 39pp gap validates the multi-index architecture."

---

## 2nd/3rd-Order Effects

### EF-1: Query Optimizer Creates Ordering Sensitivity

- **Decision/Mitigation**: B.10 (index-first query execution)
- **2nd order**: The query executor reorders filters before traversals. This means `.related().type("X")` and `.type("X").related()` produce the SAME result, but via different execution paths (the optimizer normalizes both to type-first).
- **3rd order**: Users learn that ordering does not matter and write queries in arbitrary order. A future optimization that IS order-sensitive (e.g., traversal-dependent filtering: "find entities related to X that have property Y" vs "find entities with property Y that are related to X" — different if Y is on the traversal target) silently breaks user queries that relied on ordering invariance.
- **Testable**: Create two queries with identical semantics but different chain ordering. Run both through the optimizer. If results differ, ordering sensitivity has been introduced.
- **Risk**: LOW — but becomes MEDIUM if traversal-dependent filtering is added later.

---

### EF-2: Co-occurrence Cap Creates Information Loss

- **Decision/Mitigation**: B.1 (max_cooccurrence_neighbors=100)
- **2nd order**: Entities with >100 co-occurring neighbors lose the tail of their co-occurrence distribution. The 101st neighbor is evicted (lowest count). This means the co-occurrence index becomes biased toward frequently co-mentioned entities.
- **3rd order**: Multi-hop traversal via co-occurrence preferentially discovers paths through frequently-mentioned entities, creating a "popularity bias." Rare but correct bridging paths (entity A mentioned once with entity B) are evicted in favor of common paths (entity A mentioned 50 times with "user"). The multi-hop retrieval quality degrades for rare entities.
- **Testable**: Compare multi-hop retrieval accuracy on entities with co-occurrence count > 100 vs < 10. If accuracy is significantly lower for high-degree entities (where eviction is active), the cap is causing information loss.
- **Risk**: MEDIUM. The cap is necessary (SC-1 is FATAL without it), but the eviction policy (lowest count) may need refinement. Consider evicting by recency instead of count.

---

### EF-3: Lazy Ingestion Creates Consistency Window

- **Decision/Mitigation**: B.14 (lazy ingestion: BM25 available in <5s, LLM extraction runs async)
- **2nd order**: During the async extraction window (85 seconds sequential), queries against the data see raw chunks via BM25 but NOT extracted entities, facts, or temporal events. A query "how many cats?" during this window returns 0 (entity extraction hasn't completed) while "cats" via BM25 finds the relevant chunks.
- **3rd order**: The user or downstream system receives inconsistent answers depending on when they query. If the system caches the "0 cats" answer, it persists even after extraction completes. This creates a stale-cache problem that the spec does not address.
- **Testable**: Ingest new data, immediately query an aggregation question, wait for extraction, re-query. If answers differ, the consistency window is observable.
- **Risk**: HIGH. Lazy ingestion is pragmatic (users want fast BM25) but creates a correctness gap. Add to Section 4.1: "During async extraction, aggregation and entity-based queries return PARTIAL results. The response MUST include a `extraction_status: 'pending'` flag so callers know the answer may change."

---

### EF-4: Hub Entity Detection Interacts with Co-occurrence Cap

- **Decision/Mitigation**: B.7 (hub entities traverse only top-K edges) + B.1 (co-occurrence cap)
- **2nd order**: Hub entities (degree >100) have their co-occurrence capped at 100. Traversal through hubs uses top-K edges by co-occurrence count. But if the cap evicted edges 101-200, the traversal sees only edges 1-100, which are the most FREQUENT connections, not necessarily the most RELEVANT.
- **3rd order**: Queries that traverse through the "user" hub entity (which is connected to everything) always find the user's most-discussed topics, never the rare ones. "What's the name of that restaurant I went to once last year?" fails because the restaurant's co-occurrence count with "user" is low and was evicted.
- **Testable**: Artificially create a hub entity connected to 500 entities. Query for an entity connected only once. If the query fails (entity not in traversal results), the cap+hub interaction is harmful.
- **Risk**: MEDIUM. The fix is to use recency-weighted eviction (recent co-occurrences are more valuable than old frequent ones) instead of pure frequency-based eviction.

---

### EF-5: Predicate Supersession Groups Create Cascade Conflicts

- **Decision/Mitigation**: B.12 (SUPERSESSION_GROUPS for allergy, location, employment)
- **2nd order**: When a predicate in a supersession group is ingested, ALL other predicates in the group are superseded. If the user says "I work at Acme" (predicate: works_at), then later "I quit my job" (predicate: quit), the "quit" supersedes "works_at" because they are in the "employment" group. This is correct.
- **3rd order**: But if the user then says "I started freelancing" (predicate: freelances_as), and "freelances_as" is NOT in the employment group, the old "quit" fact remains current alongside the new "freelances_as" fact. The system shows both "quit job" and "freelances" as current facts, which is technically correct but may confuse the LLM during answer generation.

  The deeper issue: supersession groups require MANUAL curation of predicate sets. As the extraction model evolves and generates new predicates, the groups become stale. New predicates that SHOULD be in a group are not, and cross-predicate contradictions re-emerge.
- **Testable**: Extract predicates from 1000 conversations. Compute how many extracted predicates belong to existing supersession groups vs how many are unclassified. If >50% of employment-related predicates are NOT in the employment group, the groups are stale.
- **Risk**: MEDIUM. Supersession groups are a manual patch on a fundamental limitation (MT-5). They help for known predicate sets but do not scale to open-domain extraction.

---

### EF-6: Entity Registry Cap Interacts with Grounding Verification

- **Decision/Mitigation**: entity_registry_cap=50 (Section 4.3.3) + grounding verification (Section 4.3.9)
- **2nd order**: The entity registry cap limits extraction to 50 entities per LLM call. If a chunk mentions 60 entities, 10 are dropped. The LLM must decide which 50 to extract.
- **3rd order**: The dropped entities have NO facts, NO type entries, NO co-occurrences. They exist ONLY in the raw chunk text. BM25 can find the chunk, but entity-based queries cannot find the entity. If a later question targets one of these dropped entities ("What did the user say about entity_51?"), the entity-based pipeline returns nothing while BM25 returns the chunk. The answer depends on which pipeline the agent uses.

  Worse: if entity_51 is mentioned again in a later chunk (where it IS within the cap), it gets created as a NEW entity with no link to the earlier mention. This creates entity fragmentation specifically for entities that appear in entity-dense chunks.
- **Testable**: Ingest a chunk with >50 entities. Query for the 51st entity via entity search. If it returns nothing, the cap caused data loss.
- **Risk**: LOW. The cap exists to prevent prompt injection (28K failures uncapped). The 50 threshold is empirically tuned. But the interaction with entity-dense chunks should be documented.

---

### EF-7: Embedding Model Change Invalidates Resolution History

- **Decision/Mitigation**: A.8 (resolution determinism boundary) + A.5 (MODEL config category)
- **2nd order**: If the embedding model changes, entities resolved via step 5 (embedding similarity) may resolve differently with the new model. The spec notes this (A.8: "entities resolved via embedding should be re-verified").
- **3rd order**: Re-verification of embedding-resolved entities requires re-running resolution on all mentions that originally hit step 5. But the mentions are not stored — only the resolved canonical name is stored on the Fact. To re-verify, the system must re-extract mentions from source_text and re-resolve them. This is equivalent to re-ingestion of all chunks that produced embedding-resolved entities.

  The practical impact: changing the embedding model is a FULL RE-INGESTION, not just a vector index rebuild. The spec's config category annotation (MODEL) says "changes require re-extraction," which is correct but understates the scope.
- **Testable**: Change embedding model. Count how many entity canonical names change after re-ingestion. If >5% change, the model switch has significant blast radius.
- **Risk**: LOW. Embedding model changes are rare (annual). But the re-ingestion cost should be explicit in the documentation.

---

### EF-8: DeterministicRouter Self-Validation Creates Latency Spike

- **Decision/Mitigation**: ET-7 mitigation (deterministic pipelines MUST validate results and fall through to agentic)
- **2nd order**: When a deterministic pipeline returns 0 results (e.g., miscategorized question), it falls through to agentic retrieval. The total latency is: deterministic pipeline latency + agentic retrieval latency. For a correctly-classified question, the latency is just the deterministic pipeline (< 100ms). For a misclassified question, the latency is 100ms + 6-10 seconds = 6.1-10.1 seconds.
- **3rd order**: If the router misclassifies 30-40% of questions (MT-10), then 30-40% of questions experience the latency spike. The average latency across all questions becomes: 0.6 * 100ms + 0.4 * 8000ms = 3260ms. The P99 latency is ~10 seconds. This is within the 10-second budget but with no margin.

  If future changes reduce the router's accuracy (e.g., questions become more colloquial), the fallthrough rate increases and average latency grows.
- **Testable**: Measure latency distribution across 1000 questions. If the bimodal distribution (fast deterministic + slow agentic) has >30% in the slow mode, the router accuracy is insufficient.
- **Risk**: MEDIUM. The self-validation is correct for correctness but creates a latency penalty. The fix is to improve router accuracy (>80% target from MT-10), not to remove the self-validation.

---

### EF-9: Type Hypernym Matching Enables False Type Inheritance

- **Decision/Mitigation**: B.14 (TYPE_HYPERNYMS for compound types)
- **2nd order**: `TypeIndex.lookup("dog")` returns entities typed "puppy" and "beagle_puppy" through hypernym matching. This enables correct aggregation for "How many dogs does the user have?"
- **3rd order**: Hypernym matching also enables incorrect matches. If TYPE_HYPERNYMS includes `"salmon": ["fish"]`, then "How many fish does the user have?" counts pet fish AND the salmon the user bought at the grocery store. The hypernym chain does not distinguish between "fish as pet" and "fish as food."

  The deeper issue: type hypernyms conflate ontological categories (IS-A relationships) with contextual categories (USED-AS relationships). A salmon IS-A fish, but it is not a pet fish. The hypernym table cannot distinguish these without context.
- **Testable**: Construct entities where a type word has multiple senses (e.g., "bass" as fish vs "bass" as guitar). Query via hypernym. If the wrong sense is included, hypernym matching is too aggressive.
- **Risk**: LOW. The hypernym table is manually curated and extensible. Bad entries can be removed by inspection. The risk increases if the table grows beyond ~100 entries.

---

### EF-10: Agentic Trace Collection Enables Future Overfitting

- **Decision/Mitigation**: Section 5.5 (traces saved for pattern extraction and SLM fine-tuning)
- **2nd order**: All agentic retrieval traces are saved for Phase 2 (pattern extraction) and Phase 4 (SLM fine-tuning). This creates a data asset for improving the system.
- **3rd order**: If the deterministic pipelines (Phase 3) are extracted exclusively from the trace data, they overfit to the question distribution of the evaluation set. When deployed on a different user (different topics, different conversation style), the hardcoded pipelines may not match real patterns. The "traces first, deterministic second" pipeline is sound only if the trace distribution is representative of production queries.

  The mitigation is that the DeterministicRouter's regex patterns are broad (not query-specific) and the agentic fallback catches anything the deterministic pipelines miss. But if >80% of queries are routed to deterministic pipelines, the system's behavior is dominated by patterns derived from a potentially non-representative evaluation set.
- **Testable**: Run the deterministic pipelines on a held-out dataset from a different user population. If accuracy drops >10pp, the pipelines are overfit to the training distribution.
- **Risk**: LOW for v1 (agentic fallback catches overfitting). MEDIUM for v2+ (if deterministic pipelines handle >80% of traffic).

---

## Dynamical Systems Analysis

### DS-1: Entity Graph Growth — Linear with Phase Transition

- **Component**: EntityGraph (stated layer)
- **Behavior**: NONLINEAR. Entity count grows linearly with turns (approximately 1 new entity per 1-3 turns). But entity DENSITY (edges per entity) grows nonlinearly — each new entity potentially creates co-occurrence edges with all entities in the same chunk. At low entity counts (<1K), the graph is sparse (avg degree ~5). At high counts (>10K), hub entities create dense clusters (avg degree >20). The phase transition occurs around 5K entities when hub entities begin dominating traversal.
- **Coupling**: Co-occurrence index (STABILIZING via the cap from B.1), BM25 index (INDEPENDENT — no coupling), TypeIndex (WEAKLY COUPLED — more entities mean more types, but O(1) lookup is maintained).
- **Healthy attractor**: Sparse graph with bounded degree, entities clustered by topic. Each topic is a loosely-connected subgraph. Traversal finds relevant entities within 2 hops.
- **Failure attractor**: Dense graph dominated by hub entities. The "user" entity is connected to everything. Traversal from any entity reaches the user in 1 hop, then reaches everything in 2 hops. BFS returns the entire graph.
- **Bifurcation point**: When the "user" entity's degree exceeds MAX_TRAVERSAL_NODES (1000 from B.7), traversal through "user" is capped. Below this threshold, traversal is complete. Above it, traversal is approximate (top-K neighbors only). This bifurcation should be a test case: verify that traversal quality does not degrade discontinuously at degree=1000.

---

### DS-2: Supersession Chain Length — Bounded Oscillation

- **Component**: SupersessionEngine
- **Behavior**: LINEAR for stable predicates (one value, no updates). NONLINEAR for oscillating predicates (value changes frequently, e.g., "user is happy" / "user is sad" / "user is happy" ...).
- **Coupling**: Fact count (STABILIZING — max_facts_per_entity caps growth). Decay function (STABILIZING — superseded facts decay faster, 10x lambda).
- **Healthy attractor**: Most (entity, predicate) pairs have 1-3 facts. Supersession chains are short. Current fact is unambiguous.
- **Failure attractor**: Oscillating predicates generate long supersession chains. At max_facts_per_entity, old facts are evicted but new ones keep arriving. The chain is a constant-memory sliding window, not growing — but the eviction of old facts means timeline() queries lose history.
- **Bifurcation point**: When a predicate's update frequency exceeds max_facts_per_entity / conversation_length. At 1000 max facts and 10000 turns, predicates updating every 10 turns fill the cap. Below this frequency: full history. Above: history truncation. Test case: verify that timeline() degrades gracefully at the cap boundary.

---

### DS-3: Agentic Retrieval — Destabilizing Positive Feedback

- **Component**: AgenticRetriever (ReAct loop)
- **Behavior**: NONLINEAR. The agent's performance has a positive feedback loop:
  ```
  Bad tool selection → irrelevant observation → context pollution → worse tool selection → more steps → token budget exhaustion → degraded quality
  ```
  This is a DESTABILIZING (positive feedback) loop. Success breeds success (good first tool call → relevant observation → good next call → early termination). Failure breeds failure.

- **Coupling**: Token budget (STABILIZING — max_steps=8 provides a hard stop). Observation truncation (WEAKLY STABILIZING — max_observation_chars=2000 limits context pollution). Sliding window (STABILIZING — from B.4, compresses earlier observations after step 4).
- **Healthy attractor**: Agent finds relevant evidence in 2-3 steps. Early termination. Low token usage. ~3 second latency.
- **Failure attractor**: Agent cycles through all 8 steps, each building on irrelevant context from the previous step. Final answer is constructed from accumulated noise. ~10 second latency. Evidence quality is worse than BM25-only.
- **Bifurcation point**: The critical transition occurs around step 4. If the agent has not found relevant evidence by step 4, the probability of finding it in steps 5-8 drops below 30% (estimated from trace analysis). The sliding window compression (B.4) is placed at step 4 specifically to mitigate this — but it also means steps 5-8 have less context to work with, further reducing their effectiveness.

  **This is the most dangerous dynamical property in the system.** The agent's failure mode is self-reinforcing. Add to Section 5.5: "If the agent has not called answer() by step 4, the system SHOULD consider early termination with BM25-only evidence (fallback to deterministic pipeline). The probability of the agent improving the answer after step 4 is <30% and comes at the cost of 4+ additional LLM calls." Add a configuration parameter: `early_termination_after: int = 4` with the semantics: "After this many steps without an answer, fall back to BM25 evidence."

---

### DS-4: Embedding Drift Under Dream Consolidation — Catastrophic Phase Transition

- **Component**: DreamConsolidator (Section 7.5)
- **Behavior**: NONLINEAR with CATASTROPHIC PHASE TRANSITION. Dream consolidation applies repulsion/attraction cycles to embeddings. Below 5 cycles (the spec's safe limit), embedding centroids shift by < 5% of their original positions. Above 50 cycles, centroid displacement exceeds the cosine similarity threshold (0.85), and queries that previously matched now fail. At 340 cycles (the proven-catastrophic number), SubEM drops from 28 to 3.
- **Coupling**: VectorIndex (DESTABILIZING — shifted embeddings invalidate cosine queries). BM25 (INDEPENDENT — not affected by embedding changes). Entity resolution step 5 (DESTABILIZING — resolution decisions change with new embeddings).
- **Healthy attractor**: Embeddings near their original positions. Cosine similarity queries work correctly. Dream consolidation adds < 1% embedding displacement.
- **Failure attractor**: Embeddings displaced beyond recognition threshold. Cosine queries return wrong results. Entity resolution creates new entities for previously-resolved mentions. SubEM catastrophe.
- **Bifurcation point**: The critical threshold is when max embedding displacement exceeds 1 - cosine_threshold = 1 - 0.85 = 0.15 (in cosine distance). At 5 cycles: displacement ~0.02 (safe). At 20 cycles: displacement ~0.08 (approaching). At 50 cycles: displacement ~0.16 (exceeds threshold). The phase transition is between 20 and 50 cycles. Test case: run dream consolidation with increasing cycle counts, measure embedding displacement after each cycle, verify that the drift_monitor (spec safeguard) correctly halts before the threshold.

---

### DS-5: BM25 Index Freshness Under Incremental Ingestion — Monotonic Degradation

- **Component**: BM25Index under incremental ingestion
- **Behavior**: LINEAR DEGRADATION. BM25's IDF (inverse document frequency) component recalculates as the corpus grows. A term that is rare in 100 documents (high IDF) becomes common in 10,000 documents (low IDF). This means BM25 scores for the same query on the same document CHANGE as more documents are added. The ranking is not stable under incremental ingestion.
- **Coupling**: Ingestion pipeline (CAUSAL — each ingestion changes BM25 scores). QueryPlan result caching (DESTABILIZING — cached results become stale as BM25 scores change).
- **Healthy attractor**: BM25 corpus is large enough that IDF is stable (law of large numbers). At >10K documents, adding one document changes IDF by < 0.01%.
- **Failure attractor**: Small corpus where IDF is volatile. At 100 documents, adding a new document with rare terms can shift IDF by 1-5%, changing rankings significantly.
- **Bifurcation point**: IDF stability occurs around 1K-5K documents. Below this, BM25 rankings are volatile under ingestion. Above this, rankings are stable. The cold-start phase (first ~5K chunks, approximately the first 30-50 conversations) is the vulnerable period.

  This degradation is monotonic (always gets better as corpus grows) and self-correcting (IDF stabilizes). It is NOT a risk for production systems with established corpora, but it IS a risk for evaluation benchmarks that measure SubEM during the cold-start phase.
- **Severity**: FATAL for evaluation accuracy during cold-start. The LME eval ingests all data then queries — at that point, IDF is stable. But a streaming eval (ingest turn-by-turn, query after each turn) would show volatile results.

---

### DS-6: Entity Resolution Cascade — Threshold-Dependent Bifurcation

- **Component**: EntityResolver steps 4 (substring) and 5 (embedding)
- **Behavior**: NONLINEAR with threshold bifurcation. The substring step (step 4) is binary: either the mention fully contains the canonical name, or it does not. There is no "close" — it is all-or-nothing. The embedding step (step 5) has a threshold (0.85 from spec, 0.80 from B.11 revision). Mentions with similarity between 0.80 and 0.85 are in the dead zone between "definitely the same" and "definitely different."
- **Coupling**: Entity count (NONLINEAR — more entities mean more false-match candidates for embedding). Embedding model (STRONGLY COUPLED — different models produce different similarity scores for the same pair).
- **Healthy attractor**: Most mentions resolve at steps 1-3 (alias, determiner, possessive). The entity graph has clean, non-overlapping entities.
- **Failure attractor**: Many mentions hit step 5 (embedding). At the threshold boundary, mentions oscillate between resolving and creating-new depending on minor text variations. The entity graph fragments into many near-duplicate entities.
- **Bifurcation point**: The embedding threshold (0.80/0.85) is the bifurcation parameter. At threshold=0.75: many false merges (homonyms merged). At threshold=0.90: many false splits (valid mentions create new entities). The optimal threshold is dataset-dependent — it varies by vocabulary, entity density, and embedding model. Test case: sweep threshold from 0.70 to 0.95 in 0.05 increments, measure false merge rate and false split rate, find the crossover point.

---

### DS-7: Decay Function — Exponential Convergence to Forgetting

- **Component**: DecayFunction (Section 7.3)
- **Behavior**: NONLINEAR (exponential decay). Salience decreases exponentially with time since last access. Access resets salience to 1.0 (Hebbian reinforcement). This creates a positive feedback loop: accessed entities stay salient, unaccessed entities decay. The loop is STABILIZING in the sense that it converges (everything eventually decays), but it creates a "rich get richer" dynamic for popular entities.
- **Coupling**: Retrieval (POSITIVE FEEDBACK — retrieved entities get accessed, resetting their salience). Garbage collection (STABILIZING — low-salience entities are candidates for GC, freeing resources for high-salience entities).
- **Healthy attractor**: Frequently-relevant entities maintain high salience. Rarely-relevant entities decay but are retained above the GC threshold for months. The memory contains a mix of high-salience (current) and low-salience (historical) entities.
- **Failure attractor**: The "popularity trap." A few entities are accessed on every query (e.g., "user" entity), maintaining salience=1.0 indefinitely. Other entities are never accessed (because they are never retrieved, because their salience is low, because they rank lower in BM25...) and decay to GC threshold. The memory gradually loses rare but potentially important entities.
- **Bifurcation point**: The GC threshold (0.01) determines which entities are forgotten. At threshold=0.001, entities survive ~690 days without access. At threshold=0.01, entities survive ~461 days. At threshold=0.1, entities survive ~230 days. The threshold determines the "memory horizon" of the system. Test case: with threshold=0.01, verify that an entity last accessed 400 days ago is still retained, and one last accessed 500 days ago is GC-eligible.

---

### DS-8: Query Plan Complexity — Combinatorial in Filter Count

- **Component**: QueryPlan execution (Section 3.2d)
- **Behavior**: NONLINEAR (combinatorial in the number of filters). Each additional filter in a query chain adds an intersection operation. With K filters, the executor performs K index lookups and K-1 set intersections. The cost of each intersection is O(min(|A|, |B|)) for sorted sets. With K filters, worst case is O(K * N) where N = largest index result.

  But the REAL nonlinearity comes from filter SELECTIVITY interactions. Two individually non-selective filters (each matching 50% of entities) may combine to be very selective (intersection matches 1%) or very non-selective (50% if correlated). The executor cannot predict selectivity without running the query.

- **Coupling**: Index sizes (WEAKLY COUPLED — larger indexes mean larger intermediate results). Cache (STABILIZING — repeated queries use cached results).
- **Healthy attractor**: Most queries have 1-3 filters. Index-first evaluation produces small candidate sets. Execution is fast.
- **Failure attractor**: Programmatically-generated queries with 10+ filters. Each filter adds a set intersection. If filters are non-selective (each matches 80% of entities), the candidate set stays large through all intersections. At 100K entities and 10 non-selective filters: 10 * 80K intersections = 800K set operations.
- **Bifurcation point**: At ~5 filters, execution transitions from "index-dominated" (fast) to "intersection-dominated" (slow). Below 5 filters, the indexes provide sufficient selectivity. Above 5, the set intersections dominate. Test case: create queries with 1, 3, 5, 7, 10 filters at varying selectivity. Measure latency. Verify that 10-filter queries with non-selective filters still complete within the 100ms budget.

---

## Summary Table

| ID | Category | Severity | Component | Core Finding |
|----|----------|----------|-----------|--------------|
| ET-1 | ✂ Exclusion | MEDIUM | 3.2 | Lazy evaluation worse when traversal precedes filter |
| ET-2 | ✂ Exclusion | FATAL | Axiom 1 | 53% unreachable is already the lethal signal for BM25-first |
| ET-3 | ✂ Exclusion | HIGH | 1.2 | 9 facets degrade on abstract concepts/relationships (20% ambiguous) |
| ET-4 | ✂ Exclusion | HIGH | 3.4 | Homonym false merges are unrecoverable without entity split |
| ET-5 | ✂ Exclusion | MEDIUM | 6.2 | Supersession chains bounded by max_facts but eviction undefined |
| ET-6 | ✂ Exclusion | HIGH | 7.1 | Inferred layer is empirically net-neutral; 3 layers justified by safety only |
| ET-7 | ✂ Exclusion | MEDIUM | 5.6 | Deterministic routing fails on colloquial phrasing; safety net exists |
| ET-8 | ✂ Exclusion | MEDIUM | Axiom 4 | Always-on safe with current weights; dilution managed by reranker |
| ET-9 | ✂ Exclusion | MEDIUM | 2.3.5 | Allen insufficient for time-series (trends/frequency); sufficient for its scope |
| ET-10 | ✂ Exclusion | MEDIUM | 5.2 | 8 tools is Goldilocks; >12 degrades selection accuracy |
| OT-1 | ⟂ Transpose | — | 2.3.x | SQLite FTS5 achieves ~22% SubEM. Full system: 43%. 21pp gap justifies complexity |
| OT-2 | ⟂ Transpose | — | 3.2 | SQL better for ad-hoc queries; ObjectSet better for graph traversal. HYBRID |
| OT-3 | ⟂ Transpose | — | 3.4 | Exact match: 70% resolution. Full cascade: 93%. 23% gap justifies complexity |
| OT-4 | ⟂ Transpose | — | 7.1 | Single tagged graph simpler; 3 layers justified by blast-radius containment |
| OT-5 | ⟂ Transpose | — | 2.3.5 | Recency sort handles 0% of temporal Qs. Allen handles 14%. Justified |
| OT-6 | ⟂ Transpose | — | 5.5 | Fixed pipeline: 28.67% SubEM. Agent: 43.3%. 14.6pp gap justifies complexity |
| OT-7 | ⟂ Transpose | — | 6.2 | Last-writer-wins sufficient for 80%; supersession adds 5-8 SubEM points |
| OT-8 | ⟂ Transpose | FATAL | All | Vector DB + prompting: 4% SubEM. Foundry: 43%. 39pp gap is definitive |
| EF-1 | Effects | LOW | 3.2d | Query optimizer creates ordering invariance expectation |
| EF-2 | Effects | MEDIUM | 2.3.6 | Co-occurrence cap creates popularity bias in traversal |
| EF-3 | Effects | HIGH | 4.1 | Lazy ingestion creates query-time consistency gap |
| EF-4 | Effects | MEDIUM | B.1+B.7 | Hub cap + co-occ cap = rare entity traversal loss |
| EF-5 | Effects | MEDIUM | B.12 | Supersession groups require manual curation of predicate sets |
| EF-6 | Effects | LOW | 4.3.3 | Entity registry cap drops entities in dense chunks |
| EF-7 | Effects | LOW | A.8 | Embedding model change = full re-ingestion (not just reindex) |
| EF-8 | Effects | MEDIUM | 5.6 | Router self-validation creates bimodal latency distribution |
| EF-9 | Effects | LOW | B.14 | Type hypernym matching conflates IS-A with USED-AS |
| EF-10 | Effects | LOW | 5.5 | Trace-derived deterministic pipelines may overfit to eval distribution |
| DS-1 | Dynamics | MEDIUM | EntityGraph | Phase transition at ~5K entities when hubs dominate |
| DS-2 | Dynamics | LOW | 6.2 | Supersession chains bounded; oscillating predicates lose history at cap |
| DS-3 | Dynamics | HIGH | 5.5 | Agent failure is self-reinforcing positive feedback loop |
| DS-4 | Dynamics | MEDIUM | 7.5 | Dream: catastrophic phase transition between 20-50 cycles |
| DS-5 | Dynamics | MEDIUM | 2.3.1 | BM25 IDF volatile during cold-start (<5K documents) |
| DS-6 | Dynamics | MEDIUM | 3.4 | Resolution threshold is a bifurcation point; optimal is dataset-dependent |
| DS-7 | Dynamics | LOW | 7.3 | Decay creates popularity trap; rare entities lost first |
| DS-8 | Dynamics | LOW | 3.2d | Query complexity combinatorial in filter count; >5 filters transitions regime |

---

## Cross-Reference: What Passes 1-2 Missed

| Pass 3 Finding | Why Passes 1-2 Missed It |
|----------------|-------------------------|
| ET-2 (53% lethal signal) | Pass 1-2 accepted BM25 primacy as axiom; didn't question whether the axiom fits the data |
| ET-4 (homonym false merge) | Pass 2 found the resolution dead zone (MT-2) but didn't explore the homonym special case |
| ET-6 (inferred layer net-neutral) | Pass 1 found inferred safety constraints (PH-4) but didn't question the layer's empirical value |
| DS-3 (agent positive feedback) | Pass 2 quantified token accumulation (SC-4) but didn't analyze the dynamical stability |
| EF-3 (lazy ingestion consistency) | Pass 2 proposed lazy ingestion (B.14) but didn't trace the consistency consequences |
| OT-8 (vector DB is DEAD) | Neither pass established the lower bound — they only analyzed the upper complexity |
