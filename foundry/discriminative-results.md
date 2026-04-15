# Phase 3.25: Discriminative Validation Results

**Date:** 2026-04-14
**Method:** Brenner tribunal — 3 agents per hypothesis (generator, test designer, adversarial critic)
**Tribunal outputs:** `foundry/tribunal/` (9 documents + 4 executable test suites)

---

## Summary Table

| Hypothesis | Generator | Test Designer | Critic | Verdict |
|-----------|-----------|---------------|--------|---------|
| **H1**: BM25 sufficient as primary | TRUE but TRIVIAL | PARTIALLY VALIDATED | WEAKER | **VALIDATED WITH QUALIFICATION** |
| **H2**: Deterministic routing replaces LLM | DECOMPOSED (3 tiers) | 60% regex accuracy | 4 CRITICAL issues | **PARTIALLY VALIDATED** |
| **H3**: Substring + embedding for entity resolution | REJECTED | REJECTED (100% false merge) | FALSE at scale | **FALSIFIED** |

---

## H1: "BM25 over raw chunks is sufficient as primary retrieval"

### Verdict: VALIDATED WITH QUALIFICATION

BM25 is the correct primary retriever. It is NOT sufficient alone.

### Evidence

**Generator findings:**
- BM25 is TRUE but TRIVIAL — it's the best retriever, but retrieval is not the bottleneck
- 53% of questions are unreachable by ANY retriever — this is a computation gap, not a retrieval gap
- 79/159 failures need temporal math, aggregation, or multi-turn synthesis — no retriever can solve these
- Entity-aware retrieval adds nothing in agentic loop (search_entities called 5.6% of the time, 0 questions gained)

**Test Designer measurements (7 tests):**
- BM25 Hit@5 by question type: fact_lookup 58.7%, temporal 18.7%, preference 0.0% — FALSIFIED for non-factoid types
- Only 12.6% of failures are genuine BM25 vocabulary gaps — VALIDATES that BM25 isn't the problem
- BM25 is 72.9% of agentic retrieval calls, first tool 90% of the time — VALIDATES as primary
- 76.7% of agentic improvement comes from iterative BM25 — VALIDATES BM25 mechanism
- BM25-addressable ceiling = 67.3% of answerable questions — VALIDATES with qualification

**Critic attacks (survived):**
- 58.3% SubEM conflates metrics (SubEM vs Hit@K) and includes dynamic_prompting + entity_prefilter (not pure BM25)
- Lean BM25 (no extras) = 23.00% SubEM — the real BM25-only number
- BM25 IDF degrades non-linearly at lifetime scale (50K-180K turns) — untested regime
- Vector baseline "4% SubEM" is from FC dataset, not LME — straw man comparison. Actual BM25 vs cosine gap is only 2pp at Hit@5

### Spec Impact

The spec must:
1. **Reframe "BM25 primary"** as "BM25 is the primary RETRIEVAL channel; computation engines (temporal, aggregation, synthesis) are co-primary for their question types"
2. **Add computation engine requirements** as first-class components, not retrieval supplements
3. **Add scaling experiments** as Phase 5 acceptance criteria (test BM25 at 10K, 50K, 100K turns)
4. **Remove the vector-only 4% comparison** — it's from a different dataset and misleading

---

## H2: "Deterministic routing can replace LLM routing without quality loss"

### Verdict: PARTIALLY VALIDATED — Decomposed into 3 tiers

The hypothesis as stated is imprecise. Routing decomposes into 3 operations with different intelligence requirements.

### Evidence

**Generator decomposition:**
- **Question Classification** (type labeling): 80% deterministic, 20% needs NLU → deterministic SUFFICIENT for majority
- **Strategy Selection** (type → pipeline): 100% deterministic, unconditionally → deterministic ALWAYS sufficient
- **Adaptive Iteration** (observe → decide next step): irreducibly requires intelligence → deterministic INSUFFICIENT
- The 14.6pp agentic gain (28.67% → 43.3%) lives entirely in adaptive iteration

**Test Designer measurements (40 tests, all passing):**
- Regex classifier: 60.0% overall accuracy (temporal 61.3%, aggregation 48.0%, single-session 84.8%)
- Priority bug: 14 temporal questions misclassified because aggregation regex fires first ("how many days")
- 69% of misclassifications fixable by regex improvement; only 31% require NLU
- 58% of agentic questions follow patterns trivially replicable by deterministic pipeline
- 95.4% of agentic latency is LLM reasoning overhead, not tool execution
- Deterministic pipeline: 6.6x speedup (7088ms → 1068ms per query)

**Critic attacks (4 critical):**
1. MemoryRouter IS LLM-based (uses 8B model at 116ms/q) — the "deterministic" alternative is regex, not MemoryRouter
2. `_parse_route` has substring collision bugs ("sum" in "summary" → aggregation)
3. Aggregation route returns bare number string with no LLM generation
4. `execute_python` sandbox has `__builtins__.__import__` bypass

### Spec Impact — Adopt 3-Tier Architecture

The spec must:
1. **Replace "deterministic routing"** with 3-tier architecture:
   - **Tier 1**: All deterministic pipelines run in parallel (always-on, <100ms, $0)
   - **Tier 2**: Confidence gate selects best answer or falls through (<1ms)
   - **Tier 3**: Agentic retrieval for the ~20% that no pipeline answers confidently (6-10s, LLM cost)
2. **Axiom 2 ("no LLM for infrastructure") reframed**: Tiers 1+2 are zero-cost deterministic. Tier 3 is opt-in LLM for hard questions.
3. **Add trace→pipeline extraction** as a first-class capability: Tier 3 traces become Tier 1 pipelines over time (the user's Phase 3→4 vision)
4. **Fix sandbox**: restrict `__builtins__` to safe subset

---

## H3: "Entity resolution via substring + embedding is sufficient"

### Verdict: FALSIFIED

Current entity resolution is fundamentally broken at production scale. The correct path is EntityStore BM25 via agentic retrieval.

### Evidence

**Generator findings:**
- H3 conflates 3 sub-problems: mention detection, entity linking, and canonicalization
- Substring works for warm re-mentions, fails for cold-start and cross-session recall
- Embedding path (`_resolve_by_embedding`) is dead code in all production configs
- Entity resolution's value is in typing/aggregation, not retrieval (search_memory called 13x more than search_entities)
- Foundry is at Stage 1.5 on the NLP entity linking ladder (exact match + heuristic)

**Test Designer measurements (36 tests, all passing):**
- `resolve()` substring fallback: **100% false merge rate** on adversarial pairs
- `resolve_exact()`: **0% false merge** — the safe path
- Embedding similarity: **0/8** deictic references resolved (worse than predicted)
- EntityStore BM25: **6/8** deictic references resolved (75% — strongest mechanism)
- Type-gated resolution: **0% false merge** when types available; 18.8% without types
- Homonym merge: **100%** — no disambiguation mechanism exists
- Canonical names are **order-dependent** — first entity created wins forever

**Critic attacks (all confirmed):**
- `resolve()` O(N) scan returns first match in insertion order, not best match
- Short canonical names (2-3 chars from triple objects: "or", "it", "up") become universal attractors
- Object-as-entity contamination: every triple object creates an entity that participates in substring matching
- Phase transition at N~1,000-3,000 entities — exactly the benchmark scale
- False merge → corrupted entity doc → wrong aggregation → wrong answer stated confidently (3-hop consequence chain, no recovery mechanism)

### Spec Impact — Major Architectural Change

The spec must:
1. **Replace substring resolution** with EntityStore BM25 as primary entity resolution at query time
2. **Require `len(canon) >= 4`** for any substring matching that remains (ingestion-time only)
3. **Add homonym disambiguation** via type gating (mandatory, not optional)
4. **Separate entity RESOLUTION (linking mentions to nodes) from entity CREATION (adding new nodes)** — `_resolve_or_create` conflates both
5. **Add EntitySplitter** for false-merge recovery (inverse of EntityMerger)
6. **Restrict triple-object entities**: only proper nouns become entity nodes; adjectives/states/quantities become property values
7. **Require longest-match-wins** instead of first-match for any remaining substring paths

---

## Cross-Hypothesis Insights

### Convergent Finding: The Bottleneck is Computation, Not Retrieval

All three tribunals independently converge on the same conclusion:
- H1: "retrieval is not the bottleneck" — 53% need computation engines
- H2: "adaptive iteration is where the value lives" — the LLM gain is in deciding what to DO with retrieved data
- H3: "entity resolution's value is in typing/aggregation, not retrieval" — the entity graph serves computation, not recall

**Implication for Foundry architecture:** The spec must elevate computation engines (temporal math, aggregation, multi-turn synthesis) to co-equal status with retrieval. The current spec treats them as retrieval supplements. They should be independent subsystems that the 3-tier architecture activates based on question type.

### Convergent Finding: EntityStore BM25 > Entity Graph Traversal

Both H1 and H3 tribunals found that EntityStore BM25 (searching rich entity documents) outperforms entity graph traversal (resolve → hop → filter). The agentic retriever confirms this: `search_entities` (BM25 over entity docs) is useful; `get_related` (graph traversal) is nearly useless.

**Implication:** The Foundry query language should prioritize entity-document search over graph traversal. The ObjectSet pattern should internally use BM25 for entity lookup, not programmatic resolution.

---

## Kill/Proceed Decision (DK-guided)

### DK:I Irreversibility Gate
- Proceeding commits us to the spec architecture. Commitment level: MEDIUM (can refactor post-build, but costly).
- The spec changes required by H1/H2/H3 are ADDITIVE (add tiers, add computation engines, change entity resolution) not SUBTRACTIVE (don't need to remove major components).

### DK:H Option Value
- No value in waiting — all three hypotheses have clear verdicts with empirical evidence.
- The 3-tier architecture and computation engine additions improve the design regardless of edge cases.

### DK:O Info-Value Gate
- Additional testing (RunPod eval comparing routing strategies) would refine H2 but not change the verdict.
- H1 and H3 verdicts are strong enough that more data won't flip them.

### Decision: PROCEED to Phase 4

All critical hypotheses have clear verdicts:
- **H1 VALIDATED WITH QUALIFICATION** — spec updated to reframe "BM25 primary" and add computation engines
- **H2 PARTIALLY VALIDATED** — spec updated with 3-tier architecture
- **H3 FALSIFIED** — spec updated with EntityStore BM25 as primary entity resolution

No hypotheses were falsified in a way that kills the overall Foundry design. The falsification of H3 IMPROVES the design (EntityStore BM25 is simpler and more robust than substring + embedding).

Proceed to Phase 4: Create phased build plan.
