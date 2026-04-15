# Hypothesis H3: Entity Resolution Sufficiency
## Brenner Kernel Phase 3.25 — Discriminative Validation

**Hypothesis**: "Entity resolution via substring containment + embedding similarity is sufficient to correctly link mentions to entities."

**Date**: 2026-04-14
**Author**: architect-agent (Brenner Kernel)

---

## 1. Level-Split Analysis

H3 as stated is a compound hypothesis. It conflates at least three independent sub-problems, each with different failure modes and evidence profiles.

### 1.1 "Entity Resolution" Conflates Two Distinct Operations

**Mention Detection** (MD): given raw text, identify spans that refer to entities.
**Entity Linking** (EL): given a detected mention, map it to the correct canonical node.

Foundry treats these as one operation inside `_resolve_or_create()`. The method does both at once: it receives a string extracted upstream (by the triple extractor or facet extractor), then tries to find a matching entity. But the failure mode of "that sushi place" is not an entity linking failure — it is a mention detection failure. The extractor produces "that sushi place" as a subject string. The linker then correctly fails to substring-match it against "sushi_han_restaurant" because the canonical name was never seen in normalized form. The problem is that "that sushi place" is a *referential expression* (an anaphor), not an entity name. Substring matching cannot resolve anaphora. Embedding similarity can, but only if the entity has been embedded under a name close enough to "sushi place" — which requires prior ingestion of the full name.

**Split verdict**: H3 must be decomposed into:
- **H3a**: "Mention spans extracted by the upstream pipeline are entity names or close synonyms." (Mention quality)
- **H3b**: "Substring containment correctly links mention strings to canonical nodes when the mention IS a name." (Linking precision)
- **H3c**: "Embedding similarity correctly links mention strings when substring fails." (Linking recall)

Evidence shows H3b has strong support (type-gated substring: "honey"/"golden honey gouramis" test suite passes, 16 resolution tests pass). H3c has weak support (embedding path exists but `_resolve_by_embedding` requires `embed_fn` to be set, and the main eval configs do not set it — it is dead code in practice). H3a is the actual bottleneck: upstream extractors produce referential expressions, not entity names.

### 1.2 "Substring Containment" Conflates Two Operations

The `resolve()` method at entity_graph.py:184-199 implements TWO different substring checks:

1. **Canonical-in-mention**: `canon in norm` — "food bowl" is in "expensive food bowl" (safe, high precision)
2. **Mention-in-canonical**: `norm in canon` — "retriever" is in "golden retriever" (dangerous, low precision)

These have fundamentally different precision profiles:

| Check | Example | Precision | Failure Mode |
|-------|---------|-----------|-------------|
| canon-in-mention | "food bowl" in "that food bowl" | High | Only fails on short canonicals ("la" in "plane") |
| mention-in-canonical | "honey" in "golden honey gouramis" | Low | False merge across types — had to add `_resolve_typed()` |

The `resolve_exact()` method (entity_graph.py:201-215) was added specifically because the bidirectional substring check was too dangerous for query time. This is empirical proof that "substring containment" is NOT one operation but two, and the dangerous direction had to be disabled for query-time paths. The code itself contradicts H3 — if substring containment were sufficient, `resolve_exact()` would not exist.

### 1.3 "Sufficient" Conflates Three Operating Regimes

**First-mention (cold start)**: Entity has never been seen. `_resolve_or_create` creates a new node. No resolution needed — but canonicalization of the first mention determines ALL future matching. If the first mention is "my doctor" instead of "Dr. Smith", the entity is forever named "doctor" and future mentions of "Dr. Smith" may or may not find it.

**Known-entity re-mention (warm)**: Entity exists. Mention is a variant of the name. Substring + alias usually works. This is the regime where H3 has strongest support (test suite covers it well).

**Cross-session re-mention (cold)**: Entity exists from a prior session but the current mention uses a different surface form. This is where substring fails hardest. "that sushi place" 6 months later has no substring overlap with "sushi_han_restaurant". The embedding path COULD help here, but it is not wired into the production eval configs.

**Split verdict**: "Sufficient" must be qualified by:
- **For warm re-mentions**: YES, substring is largely sufficient (with type gating).
- **For cold-start canonicalization**: NO, the canonical name is the first mention's surface form, which may be a referential expression.
- **For cross-session recall**: NO, substring cannot bridge the semantic gap. Embeddings are theoretically present but practically unused.

---

## 2. Cross-Domain Analysis

### 2.1 NLP Entity Linking: Where is Foundry on the Ladder?

The NLP entity linking pipeline evolved through clear stages:

| Stage | Method | When | Example System |
|-------|--------|------|----------------|
| 1 | Exact string match | Pre-2000 | Dictionary lookup |
| 2 | Edit distance + alias tables | 2000-2010 | Wikipedia redirect tables |
| 3 | Learned string similarity | 2010-2015 | Flair NER + character models |
| 4 | Dense retrieval + candidate generation | 2015-2020 | BLINK, GENRE |
| 5 | Generative linking | 2020+ | mGENRE, EntQA |

**Foundry is at Stage 1.5**. It uses exact string match (alias index) with a manual substring heuristic as a bridge to Stage 2. The embedding path exists in code but is unused in production (Stage 4 architecture, Stage 0 deployment). There is no edit distance, no learned linker, no candidate generation with re-ranking.

This is not necessarily wrong — Stages 3-5 assume a pre-existing knowledge base (Wikipedia, Wikidata) with millions of entities. Foundry's entity graph is personal and small (hundreds to low thousands of entities). The question is whether the personal scale makes substring sufficient, or whether the diversity of personal language (pet names, nicknames, referential expressions) makes it HARDER than Wikipedia linking.

**Evidence says harder**: Wikipedia entities have canonical names. Personal entities have whatever the user said first. "My cat" might be canonicalized as "cat" and never linked to "Luna" unless an explicit alias is registered. Wikipedia linking never faces this problem because the entity surface form is always close to the canonical title.

### 2.2 Database Deduplication: Blocking + Matching + Merging

In record linkage / entity resolution in databases, the standard pipeline is:

1. **Blocking**: reduce comparison space (don't compare every pair)
2. **Matching**: determine if two records refer to the same entity
3. **Merging**: combine records into a single canonical record

**Foundry's blocking**: implicit. `_resolve_or_create` iterates ALL entities for substring check (O(N) per mention). At 19K entities (as seen in the routing bottleneck from MEMORY.md), this is O(19K) string comparisons per ingested mention. No blocking key, no locality-sensitive hashing, no inverted index of name tokens.

**Foundry's matching**: substring containment only. No Jaro-Winkler, no phonetic encoding, no token overlap score, no learned matcher.

**Foundry's merging**: `register_alias()` correctly merges facts, related sets, mention_turns, and updates alias_index backpointers. This is the strongest part. The merge operation itself is sound.

**Gap analysis**: Foundry has good merging but weak blocking and matching. The matching function (substring) has known false-positive failure modes that required `_resolve_typed()` and `resolve_exact()` as patches. A proper blocking strategy (inverted index of entity name tokens) would also solve the O(N) scan problem identified in the routing bottleneck.

### 2.3 Search Engines: Exact to Semantic

Search engine query-document matching evolved:

| Era | Method | Analogy to Entity Resolution |
|-----|--------|------------------------------|
| Boolean | Exact keyword match | `alias_index` lookup |
| TF-IDF / BM25 | Weighted token overlap | NOT present in resolve() |
| Learning-to-rank | Feature combination | NOT present |
| Neural | Dense retrieval | `_resolve_by_embedding` (unused) |
| Hybrid | Sparse + dense fusion | Entity store BM25 + embedding (attempted, neutral) |

The entity STORE (entity_store.py) does use BM25 for query-time search. But the entity GRAPH resolver (entity_graph.py) does not. There is a fundamental inconsistency: at query time, entities are found via BM25 (63.4% hit@10), but at ingestion time, entities are linked via substring (unknown hit rate, no benchmark). The ingestion-time linker determines graph structure, which then determines what the query-time BM25 can find.

**Implication**: Even if query-time BM25 is excellent, the graph it searches over may be fragmented because ingestion-time substring linking failed to merge entities that should be the same node. The entity store's BM25 then searches over a fractured graph, finding entity fragments instead of consolidated entity records.

---

## 3. Third Alternative Analysis

### 3.1 Query-Time Resolution (Deferred Linking)

Instead of resolving entity identity at ingestion time, store every mention as a separate node and resolve at query time. This inverts the current architecture:

**Current**: Ingestion resolves → Graph is canonical → Query searches canonical graph
**Alternative**: Ingestion stores raw → Graph has redundant nodes → Query resolves on-the-fly

**Advantages**:
- No false merges at ingestion (undoing a merge is impossible in the current design)
- Query has MORE context (the question itself) to disambiguate
- Deferred resolution can use the query's entities as disambiguation signal

**Disadvantages**:
- Graph is larger (N mention nodes vs M canonical nodes, N >> M)
- Co-occurrence edges form between mention nodes, not entity nodes (weakens graph walks)
- Every query pays the resolution cost (latency)

**Evidence from Foundry**: `resolve_exact()` was already created to be a query-time-safe resolver. The `AliasResolver` (alias_resolver.py) IS a query-time resolver using an LLM — it resolves "What loyalty program did I enroll in?" to "Delta SkyMiles" using BM25 candidates as context. But it was tested and found NEUTRAL on SubEM (58.3% = tied with baseline). The LLM alias resolver is a query-time resolution system, and it does not help.

**Why it fails**: query-time resolution in `AliasResolver` is chicken-and-egg. It needs BM25 candidates to ground the resolution, but BM25 already found the answer. The resolution adds no NEW information. This is a fundamental limitation: if you need retrieved text to resolve the entity, and you already have the retrieved text, you do not need entity resolution at all.

**Verdict**: Pure query-time resolution is a dead end for the same reason documented in failed-approaches.md. But HYBRID deferred linking (ingest as raw, build candidate clusters, resolve lazily on first query) might work.

### 3.2 Emergent Entity Graph (Cluster-Then-Name)

Instead of naming entities at ingestion time, cluster mentions by co-occurrence and embedding similarity, then let the cluster's most frequent surface form become the canonical name.

**Current**: First mention = canonical name. Later mentions resolved to it.
**Alternative**: All mentions collected. Clusters formed. Most representative mention elected as canonical.

This is how modern entity resolution systems work in production (Dedupe.io, Zingg, Senzing). The advantage is that the canonical name is determined by the DATA, not by the accident of which mention came first.

**In Foundry terms**: Instead of `_resolve_or_create` naming the entity "doctor" because that was the first mention, the system would collect {"doctor", "Dr. Smith", "my physician", "the specialist"}, cluster them by co-occurrence (all appear in similar turn contexts) and embedding similarity, and elect "Dr. Smith" as canonical because it is the most specific surface form.

**Cost**: This requires a batch post-processing step after ingestion. It does not work incrementally. For Foundry's use case (long-running conversations with continuous ingestion), this means periodic re-clustering, which could invalidate cached query results.

**Verdict**: Theoretically superior. Practically incompatible with incremental ingestion unless implemented as an eventual-consistency model (cluster in background, serve stale canonical names until cluster completes).

### 3.3 Context Windows Instead of Entity Resolution

Maybe entity resolution is the wrong abstraction entirely. Instead of linking mentions to entities, just store overlapping context windows around co-occurring terms and retrieve windows.

This is essentially what the **raw_chunks + BM25** baseline already does. And it works: 28.67% SubEM baseline. The entity graph adds only +0-3pp on top. The agentic retriever's biggest tool by usage is `search_memory` (BM25 over raw chunks) — 142 calls vs 11 for `search_entities`.

**The provocative reading**: entity resolution in Foundry is solving a problem that BM25 over raw text already mostly solves. The cases where entity resolution helps (aggregation: "how many fish?", type queries: "what restaurants?") are precisely the cases where the entity graph's TYPE information matters, not its resolution of mentions to canonicals.

**Reframe**: The value of the entity graph is not entity RESOLUTION (linking mentions). It is entity TYPING (knowing that "golden honey gouramis" is a fish) and entity AGGREGATION (counting all fish). Resolution is a prerequisite for typing/aggregation, but the resolution itself provides minimal retrieval value.

This explains why E2-E6 (entity retrieval as RRF channel) all failed: entity resolution helps STRUCTURE queries (count, aggregate) but not RETRIEVAL queries (find the turn where X was mentioned). BM25 already handles retrieval. The agentic architecture succeeds because it uses entities for STRUCTURED operations (aggregate tool, get_related tool) while using BM25 for retrieval — this is exactly the right separation.

---

## 4. Verdict on H3

### H3 as Stated: REJECTED

"Entity resolution via substring containment + embedding similarity is sufficient to correctly link mentions to entities."

This is false because:

1. **Substring containment has known false-positive failure modes** that required two separate patches (`_resolve_typed()` for ingestion, `resolve_exact()` for query time). The codebase itself documents the insufficiency.

2. **Embedding similarity is dead code in production**. The `_resolve_by_embedding` method exists but `embed_fn` is not set in any eval config. H3 claims a two-part mechanism, but only one part runs.

3. **The hypothesis conflates mention detection with entity linking**. Referential expressions ("that sushi place") are a mention detection failure, not a linking failure. No amount of substring or embedding improvement in the linker fixes upstream extraction of anaphoric references.

4. **First-mention canonicalization is unrecoverable**. The canonical name is determined by the first surface form seen. If the first mention is "my doctor" (a referential expression, not a name), the entity is permanently named "doctor" and "Dr. Smith" may never be linked to it.

### Decomposed Sub-Hypotheses

| Sub-Hypothesis | Status | Evidence |
|---------------|--------|----------|
| H3a: Upstream extractors produce linkable entity names | WEAK | Referential expressions ("that sushi place", "my doctor") are not entity names. Extraction quality determines resolution ceiling. |
| H3b: Substring linking is precise for name-like mentions | SUPPORTED (with caveats) | 16 resolution tests pass. Type-gated substring prevents cross-type merges. But bidirectional substring remains dangerous — `resolve_exact()` exists as proof. |
| H3c: Embedding similarity covers the substring gap | UNTESTED | Code exists (`_resolve_by_embedding`). Not wired into any eval config. 0 empirical evidence of value. |

### The Structural Insight

Entity resolution matters most for structured operations (aggregation, counting, type queries) and least for retrieval (BM25 handles it). The agentic architecture already demonstrates this: `search_memory` (BM25, no entity resolution) is called 13x more than `search_entities`.

The correct framing is not "is entity resolution sufficient?" but "where does entity resolution provide value that raw text search does not?" Answer: aggregation and type-scoped queries. For those use cases, resolution PLUS typing is needed — and the typing (from faceted extraction via Qwen2.5-7B) is currently more accurate (3% hallucination) than the resolution (unknown error rate, no benchmark).

### Recommended Next Steps

1. **Benchmark ingestion-time resolution accuracy**. Run the entity graph's `_resolve_or_create` on the faceted extraction cache (3070 chunks) and measure: (a) false merges (different entities merged), (b) false splits (same entity as separate nodes), (c) canonicalization quality (is the canonical name the best surface form?). Without this benchmark, H3 cannot be quantitatively evaluated.

2. **Wire the embedding path or kill it**. `_resolve_by_embedding` is dead code. Either set `embed_fn` in the eval configs and measure its effect, or remove it to reduce cognitive overhead.

3. **Separate entity typing from entity resolution**. The entity graph's value is in typing (what IS a fish) not in resolution (which mentions are the same fish). Consider storing types on mention nodes without requiring canonical resolution, allowing aggregation queries over types without paying the false-merge cost.

4. **Do not invest further in ingestion-time resolution for retrieval**. E2-E6 proved that entity-based retrieval channels dilute BM25. The agentic architecture correctly treats entities as a structured query tool, not a retrieval channel. Invest in making the structured tools (aggregate, get_related, search_entities) more accurate rather than making resolution more complete.

---

## 5. Erotetic Closure

**X** = entity resolution sufficiency
**Q** = {Is the mechanism correct? Is it complete? Is it the right abstraction?}

| Question | Answer | Confidence |
|----------|--------|------------|
| Is substring + embedding sufficient for linking? | No — substring has documented false-positive/false-negative modes; embedding is unused | VERIFIED (code read) |
| Is entity resolution the right abstraction for retrieval? | No — BM25 over raw text dominates; entities help structured queries only | VERIFIED (E2-E6 results + agentic trace analysis) |
| Is the real bottleneck resolution or extraction? | Extraction — upstream produces referential expressions, not entity names | VERIFIED (H3a analysis) |
| What should replace current resolution? | Not "better resolution" but "better separation of concerns" — type raw mentions, aggregate over types, resolve lazily | INFERRED (cross-domain analysis) |
