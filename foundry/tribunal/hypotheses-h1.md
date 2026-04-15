# H1 Tribunal: Competing Hypotheses

Created: 2026-04-14
Author: architect-agent (Phase 3.25 Discriminative Validation)

## Original Hypothesis

**H1:** "BM25 over raw conversation chunks is sufficient as PRIMARY retrieval for personal memory queries."

**Evidence FOR:**
- BM25 baseline achieved 58.3% SubEM on LME (300 questions, chunk_size=512)
- mxbai-rerank-large-v1 pushes 97.2% of reachable questions into top-10
- Vector-only (cosine) baseline: 4% SubEM
- BM25 hit@1=24.33% vs cosine hit@1=19.67% — BM25 dominates on conversational data
- RRF of BM25+cosine: marginal gain (+1.33pp hit@5), actually hurts hit@1

**Evidence AGAINST:**
- 53% of questions (159/300) are unreachable by ANY retriever (including BM25)
- Agentic retrieval with 7 tools only reached 38.3% SubEM (43.3% with full infra)
- Gap to SOTA: Chronos=95.6%, SmartSearch=88.4% — a 50+ point chasm
- Of 26 correct agentic answers, 58% used ONLY search_memory. The other 42% needed entity search, temporal search, or Python execution — tools BM25 cannot provide
- Failure taxonomy: only 3/159 failures are genuine retrieval misses. 79/159 need computation. 54/159 are eval/rubric artifacts. BM25 is not what is failing.

---

## Competing Hypotheses

### CH-1: BM25 Is Sufficient for Retrieval But Retrieval Itself Is Insufficient

- **Operator applied:** Level-Split
- **The conflation:** H1 conflates "BM25 is the best retrieval method" with "retrieval is all you need." These are orthogonal claims. BM25 can be the best retriever while retrieval can be the wrong abstraction for 53% of questions.
- **Claim:** BM25 is optimal as a RETRIEVAL primitive, but the system needs retrieval + computation + generation as co-equal primitives. The word "sufficient" in H1 smuggles the assumption that retrieval is a complete operation, when it is only one of at least three operations the system must perform.
- **Detailed evidence:**
  - The 159 unreachable questions decompose into non-retrieval failure modes:
    - 43 temporal (27%) — need date arithmetic, not retrieval
    - 36 aggregation (22.6%) — need counting across turns, not finding one turn
    - 31 rubric (19.5%) — need semantic judgment, not retrieval at all
    - 23 eval metric (14.5%) — already retrieved correctly, eval fails
    - 13 multi-turn synthesis (8.2%) — need composition, not a single lookup
    - 13 negation/yes-no (8.2%) — need absence detection, inverse of retrieval
  - In the agentic trace analysis: 23% of correct answers required execute_python (date math). Those 6 questions would be WRONG with pure BM25 retrieval no matter how good the ranking.
  - Chronos (95.6% SubEM) does not beat Hermes by having better retrieval. It beats Hermes by having a temporal reasoning module, an event calendar, and structured extraction — capabilities orthogonal to the choice of BM25 vs cosine.
- **Testable prediction:** If we add a deterministic temporal computation engine and an aggregation engine to BM25 (without changing BM25 itself), we should recover ~50% of the 79 computation-class failures (temporal + aggregation), adding ~40 questions. This would bring SubEM from ~43% to ~56% with ZERO retrieval changes.
- **What would falsify this:** If adding temporal + aggregation engines yields <10 additional correct answers (i.e., the "computation gap" is really a retrieval gap in disguise — the turns containing the date information are themselves unretrievable).

---

### CH-2: BM25 Is Sufficient for "What"-Questions But Actively Harmful for "How Many"-Questions

- **Operator applied:** Level-Split (splitting across question types)
- **The conflation:** H1 treats "personal memory queries" as a monolithic category. But the 300 LME questions contain at least 6 structurally different question types that have radically different retrieval requirements:
  1. **Factoid lookup** (What color is X?) — BM25 excels, single turn contains answer
  2. **Temporal reasoning** (How many days between X and Y?) — BM25 finds events but cannot compute
  3. **Counting/aggregation** (How many pets?) — BM25 finds ONE pet, not ALL pets
  4. **Comparison** (Which happened first?) — BM25 finds both, cannot compare
  5. **Preference/rubric** — BM25 finds opinions but cannot synthesize
  6. **Negation/absence** (Did I ever mention X?) — BM25 structurally cannot detect absence
- **Claim:** BM25 is a correct primary for types 1-2 but structurally wrong for type 3 (aggregation). For "How many X?" questions, BM25 returns the top-K most relevant turns. But the answer requires ALL turns mentioning X, and BM25's ranked list actively suppresses turns below rank K. The correct primitive for aggregation is not ranked retrieval but exhaustive entity-typed traversal.
- **Detailed evidence:**
  - In the agentic retrieval analysis, aggregation is the largest wrong-answer category at 29% of failures (10/34 wrong answers on context-0).
  - "How many fish are there in total in both of my aquariums?" — the agent correctly found the answer by making 4 separate BM25 calls with different keywords ("number of fish in aquariums", "10-gallon tank betta fish", "Bubbles betta fish", "aquarium fish"). This worked, but only because the agent retried with different queries. Single-pass BM25 would have returned only the 20-gallon tank info and missed the betta.
  - "How many sessions of the bereavement support group did I attend?" — Ground truth: 5. Agent answered 3. BM25 found 3 mentions but missed 2 others buried in different conversational contexts. No amount of BM25 reranking would fix this because the missing sessions used different vocabulary.
  - The type_index (entities_of_type) would directly answer "how many fish?" via `type_index.lookup("fish") -> [neon_tetra, golden_honey_gourami, pleco_catfish, betta_bubbles] -> count=17`. This is O(1) lookup, not ranked retrieval.
- **Testable prediction:** On the ~36 aggregation questions in the failure taxonomy, an entity-typed count/sum engine with no BM25 involvement at all should get >50% correct (>18 questions), compared to BM25-only getting <5 correct on this subset.
- **What would falsify this:** If BM25 with aggregation_overfetch=50 (instead of 25) closes the aggregation gap to <5 failures, then BM25 IS sufficient even for aggregation — it just needs more candidates.

---

### CH-3: The Real "Primary" Is Not a Retrieval Method But a Routing Decision

- **Operator applied:** Cross-Domain (from database query planning)
- **Structural analogy:** In relational databases, full-text search is "sufficient" until you need joins. No amount of improving full-text search quality replaces the need for `SELECT ... JOIN ... WHERE`. The database world solved this not by finding a better full-text search but by having a QUERY PLANNER that routes different query shapes to different execution strategies.
  - `SELECT * FROM memory WHERE text MATCH 'car color'` = BM25
  - `SELECT COUNT(*) FROM entities WHERE type='pet'` = entity type index
  - `SELECT * FROM events WHERE date BETWEEN '2023-06-01' AND '2023-06-30' ORDER BY date` = temporal index
  - `SELECT e1.name, e2.name FROM entities e1 JOIN related r ON ... JOIN entities e2 ...` = graph traversal
- **Claim:** The correct architecture is not "BM25 primary + fallbacks" but "query planner primary + multiple execution engines." BM25 is one execution engine. The type index is another. The temporal index is a third. The query planner decides which engine(s) to activate based on question classification. There is no single "primary" — the primary is the router.
- **Detailed evidence:**
  - The Foundry spec already defines 13 question types, each with a different `primary_index` field. single_hop_factual uses bm25_chunks. counting_aggregation uses type_index. temporal uses temporal_index. The spec itself contradicts H1 by assigning different primaries to different question types.
  - The agentic retriever IS a query planner — it routes questions to different tools. Its 43.3% SubEM (+14.6pp over single-pass BM25) validates the routing approach. The gains come not from better BM25 but from the agent choosing to use search_temporal, execute_python, and search_entities when appropriate.
  - In web search, BM25 was "sufficient" (BM25 Okapi dominated TREC for years) until neural retrieval arrived. But the real revolution was not replacing BM25 with BERT — it was the cascade architecture: BM25 recall stage + neural reranking stage + feature-based ranking stage. The "primary" is the cascade, not any single stage.
  - SmartSearch (88.4%) uses a "memory module selector" that routes queries to different retrieval backends. Chronos (95.6%) has specialized temporal and event modules that bypass retrieval entirely for certain question types. The 50-point gap is not a retrieval gap — it is a routing and computation gap.
- **Testable prediction:** A deterministic question classifier (regex-based, <1ms) that routes to 4 backends (BM25 for factoid, type_index for aggregation, temporal_index for temporal, BM25 for preference) should outperform both single-pass BM25 AND the agentic retriever on LME, because it eliminates the agent's wasted exploration steps while providing the right execution engine per question type. Target: >50% SubEM.
- **What would falsify this:** If the question classifier achieves <35% SubEM (below single-pass BM25's 28.67%), then the routing overhead and classification errors outweigh the benefits of specialized backends, and a single generalist retriever is actually better.

---

### CH-4: BM25 Is Sufficient Because the Vocabulary of Personal Memory Is Closed

- **Operator applied:** Cross-Domain (from recommendation systems)
- **Structural analogy:** In recommendation systems, keyword matching was "sufficient" for books on Amazon until collaborative filtering revealed latent preferences. The transition happened because user vocabulary was too impoverished to express latent needs. But in personal memory, the user IS the data source — they already used their own vocabulary to describe their life. Unlike web search (where the user must guess the document's vocabulary), in personal memory the query and the document share the same author.
- **Claim:** BM25 is fundamentally more suitable for personal memory than for general information retrieval because the vocabulary gap between query and document is structurally smaller. When the user asks "What color is my car?", the conversation history literally says "my car is red." There is no vocabulary mismatch to bridge with dense retrieval.
- **Detailed evidence:**
  - BM25 hit@1 (24.33%) > cosine hit@1 (19.67%) on LME. This is the OPPOSITE of the web search world where dense retrieval dominates BM25.
  - RRF of BM25+cosine provides only +1.33pp improvement — nearly all useful signal is already in BM25's lexical match.
  - Cosine retrieval alone achieves only 4% SubEM — catastrophically worse than BM25. Dense retrieval's ability to bridge vocabulary gaps is nearly useless when query and document share vocabulary.
  - The 53% unreachable questions are NOT unreachable because of vocabulary mismatch. They are unreachable because the answer does not EXIST in any single turn (it must be computed, aggregated, or inferred).
  - The mxbai reranker captures 97.2% of reachable answers in top-10, suggesting that BM25's recall is nearly perfect for retrievable questions. The marginal gain from better retrieval is near zero.
- **Testable prediction:** On the 141 "reachable" questions (where the answer exists verbatim in a turn), BM25 alone should achieve >95% hit@10 with proper reranking — matching or exceeding any dense retrieval approach. The gap is entirely in the 159 unreachable questions, where no retriever of any kind can help.
- **What would falsify this:** If a custom-trained dense retriever (fine-tuned on conversational memory data) achieves >55% hit@30 on the full 300 questions (beating BM25's 46.00%), then there IS a significant vocabulary-gap subset that BM25 misses and dense retrieval can capture. The closed-vocabulary assumption would be wrong.

---

### CH-5: BM25's Sufficiency Is an Artifact of the Eval Metric, Not a Property of the System

- **Operator applied:** Level-Split (splitting "sufficient" from "measured as sufficient")
- **The conflation:** H1 conflates "BM25 performs best under SubEM" with "BM25 is the best retrieval method." SubEM (substring exact match) is a lexical metric that rewards lexical retrieval. It is structurally biased toward BM25: if the gold answer is a substring, the retriever that matches substrings wins. This is circular.
- **Claim:** BM25's dominance over cosine on LME is partially an artifact of the SubEM eval metric. A semantic eval (LLM-as-judge) would narrow or reverse the gap because cosine retrieval surfaces semantically equivalent but lexically different turns that SubEM cannot score.
- **Detailed evidence:**
  - 23 questions (14.5% of failures) are already correctly retrieved but fail SubEM because gold is a reformulation. "I recommended using a Pilsner or Lager" vs gold="a pilsner or lager would work well". BM25 retrieved this at rank 1. SubEM says it is wrong. A semantic eval says it is right.
  - The BBQ event question: agent answered "the 3rd of June" but gold says "June 3rd." SubEM fails. Semantic match succeeds. BM25 found the right turn.
  - If we correct these 23 metric failures, true performance jumps from 47% to 54.7%. But this correction applies equally to ALL retrievers, not just BM25. It is possible that cosine retrieval has a DIFFERENT set of 23 "correct but metric-failed" questions that would also jump its performance.
  - In the agentic retrieval traces: "estimated true accuracy: ~43-45%" vs measured 38.3%. The ~5-point gap is entirely SubEM false negatives on format mismatches. These are questions where the system already works but the metric says it does not.
- **Testable prediction:** Under LLM-as-judge evaluation (gpt-4o scoring semantic equivalence), the gap between BM25 and cosine narrows from +4.67pp to <2pp on the reachable subset. BM25's advantage is partially a measurement artifact.
- **What would falsify this:** If under LLM-as-judge eval, BM25 still leads cosine by >4pp on the reachable subset, then the advantage is real and not metric-driven. The lexical overlap in personal memory genuinely favors keyword retrieval.

---

### CH-6: Single-Pass Retrieval of Any Kind Is the Wrong Abstraction

- **Operator applied:** Cross-Domain (from human memory recall)
- **Structural analogy:** When a human is asked "How many fish do you have?", they do not perform a single memory retrieval. They perform an iterative scan: "Let me think... I have the big tank with the tetras, that is 10... and the gouramis, 5... and the pleco... and oh right, Bubbles in the little tank." Human memory recall for aggregation is iterative, cue-dependent, and self-terminating. No single retrieval operation models this.
- **Claim:** The question is not "Is BM25 sufficient?" but "Is single-pass retrieval the right abstraction at all?" The agentic retriever's 14.6pp improvement over single-pass demonstrates that iterative retrieval is categorically different from single-pass retrieval — not an incremental improvement but a phase transition in the system's capability.
- **Detailed evidence:**
  - Single-pass BM25 baseline: 28.67% SubEM
  - Agentic BM25-only (iterative): 38.3% SubEM (+9.63pp)
  - Agentic full infra: 43.3% SubEM (+14.6pp)
  - The 9.63pp gain from BM25-only agentic over single-pass BM25 is LARGER than the 3.3pp gain from adding cosine+reranker+temporal+entities to the agent. The iteration itself provides more value than all additional infrastructure combined.
  - The fish counting example: the agent made 4 BM25 calls with progressively different keywords. Single-pass would have returned only the first set of results. Each subsequent call found NEW information because the dedup mechanism excluded previously-seen turns.
  - The Hilton points example: the agent first found "one free night", then read a nearby turn, then found "two free nights" — a 3-step iterative refinement that single-pass cannot model.
- **Testable prediction:** A BM25-only iterative agent with max_steps=12 (double the current budget) should achieve >45% SubEM — outperforming the full-infrastructure agent at max_steps=6 — because the marginal value of iteration exceeds the marginal value of tool diversity.
- **What would falsify this:** If increasing max_steps from 6 to 12 yields <2pp improvement (diminishing returns on iteration), then single-pass with better infrastructure is preferable to more iteration with BM25 alone.

---

## Third Alternatives

### TA-1: The Question Is Not "Is BM25 Sufficient?" But "Is Retrieval the Right Abstraction at All?"

- **The framing is wrong because:** H1 frames the problem as choosing between retrieval methods (BM25 vs cosine vs hybrid). But 53% of questions cannot be answered by ANY retrieval method. The real problem space is not "which retriever" but "which cognitive operations." Retrieval is one cognitive operation. Computation is another. Inference is a third. Synthesis is a fourth.
- **Better framing:** "What is the minimal set of cognitive operations needed to answer personal memory queries, and how should they be composed?"
  - Based on the failure taxonomy, the minimal set is:
    1. **Lookup** — find a turn containing X (BM25 is optimal here)
    2. **Count** — enumerate entities of type T (entity index, not retrieval)
    3. **Compute** — date arithmetic, numeric operations (Python exec, not retrieval)
    4. **Compare** — temporal ordering, numeric comparison (Allen algebra + Python)
    5. **Synthesize** — compose answer from multiple pieces (LLM generation)
    6. **Judge** — assess absence, preference, opinion (LLM reasoning)
  - Operations 1-4 are deterministic. Operations 5-6 require an LLM. BM25 is relevant ONLY for operation 1. The system needs a query planner that detects which operations are needed and composes a pipeline.
- **Testable prediction:** A system that implements all 6 operations with deterministic routing should achieve >70% SubEM on LME (matching the cumulative impact analysis: 47% current + 23 eval fix + temporal + aggregation = 81%). A system with only operation 1 (retrieval) tops out at ~55% regardless of retrieval method.

---

### TA-2: The 53% "Unreachable" Is Not a Recall Problem But a Representation Problem

- **The framing is wrong because:** "Unreachable by BM25" implies the solution is better retrieval. But "unreachable" means the answer does not EXIST as a substring in any single turn. The answer must be COMPUTED from turn data. This is not a retrieval failure — it is a representation gap. The data is stored as raw text when it should ALSO be stored as structured facts (entities, events, quantities) that can be queried computationally.
- **Better framing:** "Raw text is a sufficient storage format for lookup queries but not for computational queries. What additional representations must be maintained during ingestion to enable non-lookup operations?"
  - Required representations:
    - **Entity index**: canonical entity -> type, properties, mention turns (for counting)
    - **Event calendar**: temporal expressions -> dates, durations (for date math)
    - **Quantity store**: entity.property -> numeric values (for aggregation and comparison)
    - **Relation graph**: entity -> related entities (for multi-hop bridging)
  - These representations exist in the Foundry spec (7 specialized indexes). The question is whether they REPLACE BM25 or SUPPLEMENT it. The answer from the data: they supplement. BM25 remains primary for 47% of questions. Structured indexes handle the other 53%.
- **Testable prediction:** Building just the entity type index and event calendar (without changing retrieval at all) should convert 30+ of the 79 computation-class failures into correct answers. This validates that the problem is representation, not retrieval.

---

### TA-3: "Primary" Is a Load-Bearing Misnomer — The Architecture Should Be Ensemble, Not Primary/Secondary

- **The framing is wrong because:** "Primary retrieval" implies a hierarchy: one method dominates, others are fallbacks. But the data shows no single method dominates across question types. BM25 dominates factoid lookup. Entity index dominates aggregation. Temporal index dominates date questions. The word "primary" creates a false hierarchy that leads to architectural mistakes like using BM25 as the only retrieval path with everything else as RRF additions (which then dilute the BM25 signal, as proven by experiments E2-E6).
- **Better framing:** "Which execution engine(s) should be activated for each question class, and how should their results be composed?"
  - This is not primary/secondary. It is ensemble selection:
    - Factoid lookup: BM25 ONLY (adding anything dilutes)
    - Aggregation: entity type index ONLY (BM25 returns wrong count)
    - Temporal: event calendar + Python exec (BM25 finds events, exec computes)
    - Multi-hop: BM25 + entity graph (BM25 for first hop, graph for bridging)
    - Preference: BM25 + LLM synthesis (retrieval is easy, synthesis is hard)
  - The common pattern in E2-E6 failures: RRF of BM25 + entity always hurts. This is because RRF assumes both channels contribute signal for every query. They do not. Entity signal helps counting queries and hurts factoid queries. The fix is not better fusion but selective activation.
- **Testable prediction:** Selective activation (route to single engine per question type) outperforms both single-engine (BM25 for all) and fusion (RRF of all engines). Specifically: >48% SubEM on the 300-question LME set, compared to 28.67% BM25-only and 25.33% BM25+entity RRF.

---

## Synthesis: H1 Verdict

H1 is **TRUE but TRIVIAL**. Yes, BM25 is the best single retrieval method for personal memory queries. No other retrieval method (cosine, hybrid, reranked) outperforms it. The evidence is overwhelming.

But H1 is also **MISLEADING**. It frames the problem as a retrieval problem when the actual bottleneck is not retrieval but computation. The 53% unreachable questions cannot be solved by ANY retrieval method. They require:
- Temporal computation (43 questions, 27%)
- Aggregation/counting (36 questions, 22.6%)
- Semantic judgment (31+23 questions, 34%)
- Multi-turn synthesis (13 questions, 8.2%)

**The operative question for Hermes Foundry is not "Is BM25 sufficient?" but "What operations beyond retrieval must the system support, and how should they be composed?"**

The Foundry spec already answers this correctly with its 7 specialized indexes, 13 question types, and ObjectSet query language. H1's value is in confirming that BM25 should be the execution engine for factoid/lookup operations — not that BM25 should be the only execution engine.

---

## Recommended Discriminative Tests

These tests should run against the EXISTING mabench codebase to validate the competing hypotheses before the Foundry build:

| Test ID | Hypothesis | Test | Expected Result |
|---------|-----------|------|-----------------|
| DT-1 | CH-1 | Add deterministic date-diff engine to agentic retriever (already has execute_python). Measure on 43 temporal questions. | >20 new correct answers |
| DT-2 | CH-2 | Build entity_type_count() from faceted cache. Test on 36 aggregation questions. | >15 new correct answers |
| DT-3 | CH-3 | Regex question classifier -> 4 backends (hardcoded routes, no agent). Test on full 300q. | >48% SubEM |
| DT-4 | CH-4 | Train sentence-transformer on LME data. Compare hit@30 vs BM25 on 300q. | <2pp improvement |
| DT-5 | CH-5 | LLM-as-judge eval on current BM25 results. Count how many "wrong" answers are actually correct. | >20 recovered answers |
| DT-6 | CH-6 | Increase max_steps to 12 for BM25-only agent on 60q. | >45% SubEM |
| DT-7 | TA-2 | Build event calendar from dateparser output. Test temporal lookup on 43 temporal questions. | >25 correct with calendar + Python |

---

## File Cross-References

Evidence files that support these hypotheses (all verified by reading, not grep):

- `/Users/cosimo/.hermes/hermes-agent/ablation_testing/RESULTS.md` — full ablation results, failure taxonomy
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python/output/agentic_retrieval_full_infra_60q.json` — 60-question agentic retrieval traces
- `/Users/cosimo/.hermes/hermes-agent/foundry/spec.md` — Foundry spec with 13 question types (Section 8)
- `/Users/cosimo/.hermes/hermes-agent/foundry/adversarial-pass3.md` — adversarial review of temporal/tool-count claims
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python/mabench/agentic_retrieval.py` — agentic retriever implementation
- `/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python/mabench/configs.py` — all retrieval config flags documenting what was tried
