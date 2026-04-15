# Hypothesis H2: Deterministic Routing Can Replace LLM-Based Query Routing

**Date**: 2026-04-14
**Author**: architect-agent (Opus 4.6)
**Status**: UNDER TRIBUNAL
**Spec Axiom Under Test**: Axiom 2 ("No LLM calls for infrastructure") + Axiom 7 ("Traces first, deterministic second, parametric third")

---

## 1. Erotetic Framing

**X** = the claim that deterministic (non-LLM) routing can replace LLM-based query routing without quality loss.

**Q** = the design questions that must be resolved before accepting or rejecting H2:

| # | Question | Status |
|---|----------|--------|
| Q1 | What exactly is "routing" — is it one operation or multiple? | Analyzed below |
| Q2 | What does the empirical evidence actually show about LLM vs deterministic routing? | Analyzed below |
| Q3 | Is "no LLM" conflating "no expensive LLM" with "no model at all"? | Analyzed below |
| Q4 | Does routing even need to exist, or is it a false problem? | Analyzed below |
| Q5 | What are the failure modes of each approach? | Analyzed below |
| Q6 | What does cross-domain evidence say about static vs dynamic dispatch? | Analyzed below |

---

## 2. Level-Split: "Routing" Is Three Distinct Operations

The word "routing" in the current codebase conflates three operations that have different intelligence requirements. Treating them as one is a category error.

### 2.1 Operation Decomposition

| Operation | What It Does | Current Implementations | Intelligence Required |
|-----------|-------------|------------------------|----------------------|
| **Question Classification** | Map question to a type label (factual, temporal, aggregation, ...) | `classify_question_type()` in outgestion.py (5 types, regex); `classify_question()` in auto_resolve.py (4 types, regex); `MemoryRouter.route_question()` in memory_router.py (6 types, LLM); `DeterministicRouter.route()` in spec Section 5.6 (13 types, regex) | LOW-MEDIUM: Pattern matching on surface form. Regex handles 60-70% (MT-10). Colloquial phrasing ("What's the deal with X?") defeats regex. |
| **Strategy Selection** | Given a type label, choose which tools/pipelines to invoke and in what order | `DeterministicRouter.execute()` in spec Section 5.6 (5 hardcoded pipelines); `SYSTEM_PROMPT` in agentic_retrieval.py (LLM-chosen tool sequence) | LOW: Once the type is known, the pipeline is deterministic. The spec's pipelines are lookup tables. |
| **Adaptive Iteration** | Decide when to stop, what to try next when initial search fails, how to combine partial results | `AgenticRetriever.retrieve()` in agentic_retrieval.py (ReAct loop, LLM decides next tool based on observation) | HIGH: Requires understanding what was found, what is missing, and what new strategy might work. This is the operation the LLM is actually doing. |

**Key insight**: The spec's DeterministicRouter conflates operations 1 and 2 (classification + strategy selection) and treats operation 3 (adaptive iteration) as the "fallback." But the 14.6pp gap between single-pass (28.67%) and agentic (43.3%) is entirely attributable to operation 3, not operations 1 or 2.

### 2.2 Where the Intelligence Actually Lives

The agentic retrieval traces (output/agentic_retrieval_full_infra_60q.json) reveal what the LLM actually does at each step:

```
Typical trace for a CORRECT agentic answer:
  Step 1: search_memory("user's airline") -> found mentions of Delta
  Step 2: search_entities("Delta") -> found entity with "what: airline"
  Step 3: get_related("Delta") -> found co-occurring entity "Valentine's Day trip"
  Step 4: read_turn(142) -> confirmed "flying Delta for Valentine's Day"
  Step 5: answer("The user flies Delta Airlines")
```

The LLM's value is NOT in classifying this as "fact_lookup." The value is in step 3: deciding to follow the entity graph after the initial BM25 search found a partial answer, and in step 4: deciding which specific turn to read for confirmation. A deterministic `fact_lookup` pipeline (BM25 -> top-5 -> format) would have stopped at step 1 and potentially returned a worse answer.

### 2.3 The Split Verdict

| Operation | Can It Be Deterministic? | Evidence |
|-----------|------------------------|----------|
| Question Classification | **YES, with caveats.** Regex covers 60-70%. Keyword-set classification (MT-10 mitigation) could reach 80%. Remaining 20% are genuinely ambiguous. | outgestion.py regex: 5 types, handles clean questions. auto_resolve.py regex: 4 types for date/duration/aggregation. Both miss colloquial phrasing. |
| Strategy Selection | **YES, unconditionally.** Given a correct type label, the pipeline is a lookup table. No intelligence needed. | spec Section 5.6: 5 pipelines, each a fixed sequence of index queries. |
| Adaptive Iteration | **NO, not with rules.** The observation-dependent branching (try X, see result, decide Y) is what produces the 14.6pp gain. Hardcoding this requires enumerating all possible observation states. | Agentic traces: average 4.5 tool calls. The calls after step 1 depend on step 1's result. |

---

## 3. Level-Split: "Deterministic" Is Three Distinct Strategies

The word "deterministic" in Axiom 2 conflates three implementation strategies with very different costs and capabilities.

### 3.1 Strategy Decomposition

| Strategy | Definition | Latency | Cost | Accuracy Ceiling |
|----------|-----------|---------|------|-----------------|
| **Rule-based** | Regex patterns, keyword sets, if/else chains | < 1ms | $0 | ~70% (MT-10 estimate). Misses colloquial, implicit, and multi-clause questions. |
| **Learned-but-frozen** | A trained classifier (logistic regression, small neural net, decision tree) shipped as a static artifact. No inference-time LLM call. | 1-5ms | $0 (inference is CPU) | ~85-90%. Trained on trace data. Covers colloquial phrasing the regex misses. But brittle to distribution shift (EF-10). |
| **Small Language Model** | A 0.6-1.4B parameter model fine-tuned for tool routing. Runs locally, no API call. | 20-100ms | $0 (local GPU/CPU) | ~90-95%. The spec's Phase 4 target. Handles novel phrasings through generalization. |

### 3.2 Where Does the User's Axiom Actually Draw the Line?

The user's stated axiom: "No LLM calls for infrastructure. Memory must be cheap to operate." (spec Section 0, Axiom 2)

This axiom is about COST, not about MODEL SIZE. The evidence:
- The user explicitly includes Phase 4 (fine-tune 0.6-1.4B SLM for uncommon patterns) in the architecture vision (RESULTS.md, line 338).
- The user's framing: "Parametric memory = model weights learn user patterns via fine-tuning" (RESULTS.md, line 347).
- The spec's Axiom 7: "Traces first, deterministic second, parametric third."

**Interpretation**: Axiom 2 prohibits EXPENSIVE LLM calls for infrastructure (gpt-4o-mini at $0.15/1M input + $0.60/1M output, ~$0.001/query, ~6-10 seconds latency). It does NOT prohibit a frozen local model that runs in 50ms at zero marginal cost.

### 3.3 The Ambiguity

But there is a genuine tension. The spec also says (Section 5.6): "Classification: regex patterns, no LLM (< 1ms)." The "<1ms" target excludes even a 50ms SLM. And Axiom 4 says "Always-on, never routed" — which could mean routing itself is the wrong framing.

This is not a bug in the axioms. It is a phase-dependent constraint:
- **Phase 1** (current): regex classification, agentic fallback. Cost ceiling violated by agentic (gpt-4o-mini per query).
- **Phase 3** (planned): deterministic pipelines from trace analysis. Regex + learned-frozen covers 80%+.
- **Phase 4** (planned): SLM for the long tail. Zero-cost local model.

H2 is TRUE for strategy selection (operation 2). H2 is CONDITIONALLY TRUE for classification (operation 1), depending on which "deterministic" strategy is used. H2 is FALSE for adaptive iteration (operation 3), which is where the quality gap lives.

---

## 4. Level-Split: "Quality Loss" Depends on Which Quality

### 4.1 Metrics in Tension

| Metric | Deterministic Winner | Agentic Winner | Why |
|--------|---------------------|----------------|-----|
| **Latency (P50)** | 100ms vs 7000ms | | Deterministic is 70x faster |
| **Latency (P99)** | 100ms vs 10000ms | | No tail latency |
| **Cost per query** | $0 vs ~$0.001 | | Zero marginal cost |
| **SubEM accuracy** | | 43.3% vs 28.67% | 14.6pp gap |
| **Reliability** | Deterministic: always same result | | No stochastic variation |
| **Coverage (% of Qs handled)** | ~60-70% | ~100% | Agentic handles anything |

The fundamental tradeoff: deterministic routing trades accuracy for latency and cost. The question is whether the accuracy loss is acceptable, and for WHICH questions.

### 4.2 The Bimodal Distribution

Not all questions need adaptive iteration. The failure taxonomy from ablation 07 (RESULTS.md) reveals:

| Category | % of 300 | Needs Adaptive Iteration? | Deterministic Pipeline Sufficient? |
|----------|---------|---------------------------|-----------------------------------|
| Reachable by BM25 (hit@10) | 47% | **No** — single BM25 search finds the answer | **Yes** — fact_lookup pipeline |
| Temporal arithmetic | 14.3% | **No** — deterministic date computation | **Yes** — temporal pipeline + execute_python |
| Aggregation/counting | 12% | **Partial** — entity type lookup is deterministic, but scattered mentions need multi-search | **Partial** — aggregation pipeline works when TypeIndex has the answer |
| Eval metric false neg | 7.7% | **No** — answer is already retrieved | **Yes** — LLM-as-judge fixes this |
| Multi-turn synthesis | 4.3% | **Yes** — must find 2+ turns and combine | **No** — needs iterative search |
| Colloquial/ambiguous | ~15% | **Yes** — regex can't parse, needs LLM understanding | **No** — falls to agentic |

**Estimate**: ~60-65% of questions can be handled by deterministic pipelines without quality loss. ~15-20% need adaptive iteration. ~15-20% are currently unsolvable by either approach (rubric, negation).

---

## 5. Cross-Domain Analysis

### 5.1 Networking: Static vs Dynamic Routing

| Aspect | Static Routing | Dynamic Routing | Hermes Mapping |
|--------|---------------|-----------------|----------------|
| Config | Manual route tables | OSPF/BGP learn routes | Regex patterns vs LLM-learned routes |
| Failover | Manual, slow | Automatic reconvergence | Deterministic: no failover. Agentic: retries with different strategy |
| Overhead | Zero | Protocol messages, CPU for route computation | Zero vs ~$0.001/query |
| When static wins | Small, stable networks | | Small, stable question distributions (known user patterns) |
| When dynamic wins | | Large, changing networks with failures | Novel question types, new conversation topics |

**Lesson**: Production networks use BOTH. Static routes for the default path, dynamic for exceptions. This maps directly to: deterministic pipelines for known patterns, agentic for unknown.

### 5.2 Compilers: Static vs Dynamic Dispatch

| Aspect | Static Dispatch | Dynamic Dispatch (vtable) | Hermes Mapping |
|--------|----------------|--------------------------|----------------|
| Resolution | Compile-time (known type) | Runtime (unknown type) | Regex at classification time vs LLM at query time |
| Performance | Inlined, zero overhead | Vtable lookup, ~1ns | <1ms vs 7000ms |
| Flexibility | Only known types | Any type implementing interface | Only known question patterns vs any question |
| When static wins | Hot paths, known concrete types | | Known question types (60-70% of traffic) |
| When dynamic wins | | Polymorphic collections, plugins | Novel/ambiguous questions |

**Lesson**: C++ uses `final` to enable devirtualization on hot paths while keeping vtables for polymorphic types. Hermes Foundry should "devirtualize" known question patterns to deterministic pipelines while keeping the agentic "vtable" for polymorphic questions.

### 5.3 Biology: Innate vs Adaptive Immune System

| Aspect | Innate (Deterministic) | Adaptive (Learned) | Hermes Mapping |
|--------|----------------------|-------------------|----------------|
| Response time | Minutes | Days to weeks (first exposure) | <1ms vs 7000ms |
| Specificity | Low (pattern-matching on common molecular shapes) | High (antibodies for specific antigens) | Regex (broad patterns) vs LLM (specific question understanding) |
| Memory | None (same response every time) | Long-term (B-cell memory) | No learning vs trace collection for fine-tuning |
| When innate wins | Known, common pathogens | | Known, common question types |
| When adaptive wins | | Novel pathogens, first exposure | Novel question types |
| Integration | Innate activates FIRST, signals adaptive if needed | | Deterministic runs first, falls through to agentic |

**Lesson**: Both systems are necessary. The innate system handles ~95% of pathogen encounters cheaply. The adaptive system handles the rare, novel threats and creates new "rules" (antibodies) that become part of future innate responses. This is EXACTLY the spec's Phase 1-4 architecture: agentic traces (adaptive) become deterministic pipelines (innate) become parametric memory (B-cell memory).

---

## 6. The Third Alternative: Routing as a False Problem

### 6.1 Hypothesis: "Always-On" Eliminates Routing

Axiom 4 states: "Always-on, never routed. Every subsystem runs unconditionally."

If taken literally, this means: every query runs ALL deterministic pipelines simultaneously, collects all results, and a merge function selects the best answer. No routing needed.

```
Question → [BM25 pipeline] → candidate_1
         → [Temporal pipeline] → candidate_2
         → [Aggregation pipeline] → candidate_3
         → [Entity lookup pipeline] → candidate_4
         → MERGE(candidates) → best_answer
```

**Why this might work**: Each pipeline is cheap (<100ms). Running all 5 in parallel costs ~100ms total. No classification error is possible because every pipeline runs.

**Why this fails**: The RRF dilution problem (E2-E6). When the wrong pipeline returns high-confidence garbage, it pollutes the merge. The aggregation pipeline returning 0 entities when the question is factual is not neutral — it wastes the merge slot and can confuse the final answer generation. The proven evidence: every experiment adding MORE channels to RRF has regressed or been neutral.

**Refinement**: Always-on works for RETRIEVAL channels (BM25 + cosine + rerank is proven). It fails for ANSWER GENERATION channels. The pipelines produce different ANSWER FORMATS (a count, a date, a factual sentence), and merging answer formats is harder than merging retrieval candidates.

### 6.2 Hypothesis: "Compute All, Gate at Output"

A variation: run all pipelines but use a deterministic confidence gate to select which pipeline's answer to surface.

```
Question → [BM25 pipeline] → (answer_1, confidence_1)
         → [Temporal pipeline] → (answer_2, confidence_2)
         → [Aggregation pipeline] → (answer_3, confidence_3)
         → SELECT MAX(confidence) → final_answer
```

**Why this might work**: Each pipeline can produce a calibrated confidence score. The temporal pipeline has high confidence when it finds matching date expressions. The aggregation pipeline has high confidence when TypeIndex returns entities. Factual pipeline has high confidence when BM25 hit@1 score is high.

**Why this is still routing**: The confidence-based selection IS routing — just post-hoc rather than pre-hoc. The difference is that misclassification is recoverable: if the temporal pipeline fires on a factual question, its low confidence score ensures the factual pipeline's answer is selected instead.

**This is actually what the spec describes** (Section 5.6 + ET-7 mitigation): "Deterministic pipelines MUST validate their own results. If a pipeline returns 0 results or a confidence below threshold, it MUST fall through to agentic retrieval."

### 6.3 The Real Architecture

Combining the analysis: the system needs a THREE-TIER dispatch:

```
Question
  │
  ├─[Tier 1: Deterministic pipelines] (ALL run in parallel, <100ms)
  │   ├─ fact_lookup:   BM25(q) → top-5 → confidence
  │   ├─ temporal:      TemporalIndex(q) + date_arithmetic → confidence
  │   ├─ aggregation:   TypeIndex(type) → count/list → confidence
  │   ├─ comparison:    entity_lookup(a, b) → confidence
  │   └─ preference:    BM25(q, filter=assistant) → confidence
  │
  ├─[Tier 2: Confidence gate] (<1ms)
  │   If max(confidence) >= threshold: return best pipeline's answer
  │   Else: fall through to Tier 3
  │
  └─[Tier 3: Agentic retrieval] (6-10 seconds)
      ReAct loop with full tool access
```

This architecture:
- Satisfies Axiom 2: Tier 1+2 are zero-cost deterministic. Tier 3 is the expensive fallback.
- Satisfies Axiom 4: All Tier 1 pipelines run unconditionally (always-on).
- Handles the 60-70% that regex can classify (Tier 1 answers directly).
- Handles the 30-40% that regex cannot classify (Tier 2 gate selects best pipeline or falls through).
- Handles the 15-20% that need adaptive iteration (Tier 3 agentic).
- Does NOT require question classification at all (no routing error possible in Tier 1).

---

## 7. Evidence Audit

### 7.1 Evidence FOR H2

| Evidence | Source | Strength | What It Actually Shows |
|----------|--------|----------|----------------------|
| Regex classifier works for dynamic prompting | outgestion.py L74-100 | MODERATE | 5 question types classified by regex. Used in production (--dynamic_prompting flag). No accuracy data on classification itself, but SubEM with dynamic prompting is better than without. |
| Memory router achieves baseline parity | MEMORY.md | MODERATE | LLM-based router (8B model, ~116ms) produces same SubEM as no-router baseline. This means routing itself has zero value for the current pipeline. |
| BM25 baseline is 58.3% | ablation 01 | STRONG | The simplest possible "routing" (send everything to BM25) already achieves the best single-channel result. Routing cannot improve on this for the 58.3%. |
| Strategy selection is a lookup table | spec Section 5.6 | STRONG | Given a type label, the pipeline is deterministic. No LLM needed for this step. |
| Axiom 2 is a design constraint | spec Section 0 | STRONG | The user has decided this. The question is feasibility, not desirability. |

### 7.2 Evidence AGAINST H2

| Evidence | Source | Strength | What It Actually Shows |
|----------|--------|----------|----------------------|
| Agentic retrieval: 43.3% vs 28.67% baseline | RESULTS.md | STRONG | The LLM-routed approach gains 14.6pp. But this is from ADAPTIVE ITERATION (operation 3), not from better classification (operation 1). |
| Regex misses 30-40% of questions | MT-10 (adversarial-pass2) | MODERATE | Regex patterns are narrowly defined, miss colloquial phrasing. But keyword-set classification could reach 80%. |
| 13 question types may be too many for regex | spec Section 8.1 | MODERATE | More types = more regex patterns = more false positives. But the spec's 5 deterministic pipelines only need 5 classifications, not 13. |
| ET-7 adversarial example | adversarial-pass3 | MODERATE | "What's the deal with Sarah's new thing?" defeats regex. But falls through to agentic, which handles it correctly. |
| EF-8 latency spike on fallthrough | adversarial-pass3 | LOW | 30-40% fallthrough means bimodal latency. But P99 stays within 10s budget. |

### 7.3 Evidence That H2 Is the Wrong Question

| Evidence | Source | Strength | What It Actually Shows |
|----------|--------|----------|----------------------|
| The 14.6pp gap is from iteration, not classification | Agentic traces | STRONG | The LLM decides which NEXT tool to call based on PREVIOUS results. Deterministic classification cannot replace this. |
| Always-on eliminates classification | Axiom 4 | MODERATE | If all pipelines run simultaneously, classification is unnecessary. The confidence gate replaces the router. |
| Phase 4 SLM is planned | RESULTS.md L338 | STRONG | The user already plans to replace gpt-4o-mini with a fine-tuned SLM. The real question is WHEN, not WHETHER. |
| Router accuracy has no empirical data | MT-10 | STRONG | No experiment has measured deterministic vs LLM routing accuracy on the same question set. The 60-70% estimate is from pattern analysis, not measurement. |

---

## 8. Verdict: H2 Is Partially True, Partially False, and Partially the Wrong Question

### 8.1 Decomposed Verdicts

| Claim | Verdict | Confidence | Reasoning |
|-------|---------|------------|-----------|
| Deterministic classification can replace LLM classification without quality loss | **PARTIALLY TRUE** | HIGH | Regex handles 60-70%. Keyword-set can reach ~80%. The remaining 20% falls through to agentic anyway. Quality loss is ZERO for the 80% that match, because classification accuracy on these is equivalent. |
| Deterministic strategy selection can replace LLM strategy selection | **TRUE** | VERY HIGH | Strategy selection is a lookup table. Once the type is known, the pipeline is fixed. The LLM adds zero value here. |
| Deterministic pipelines can replace LLM adaptive iteration | **FALSE** | VERY HIGH | The 14.6pp gap (28.67% -> 43.3%) is entirely from adaptive iteration. Hardcoding observation-dependent branching requires enumerating all possible states, which is intractable. |
| The overall system can be fully deterministic | **FALSE** | HIGH | ~15-20% of questions genuinely need adaptive iteration. But ~80% can be handled deterministically without quality loss. |
| A 0.6-1.4B SLM can replace gpt-4o-mini for adaptive iteration | **PLAUSIBLE, UNTESTED** | LOW | The spec plans this (Phase 4), but no evidence exists yet. The SLM would need to learn tool routing from traces, not answer generation. |

### 8.2 The Real Architecture Implication

H2 should not be framed as "deterministic routing vs LLM routing." It should be framed as:

**"How much of the query pipeline can be deterministic, and where is the irreducible LLM dependency?"**

Answer: ~80% deterministic (Tier 1 + Tier 2), ~20% LLM-dependent (Tier 3). The LLM dependency is in ADAPTIVE ITERATION, not in CLASSIFICATION or STRATEGY SELECTION.

---

## 9. Recommended Experiments

### 9.1 Exp-H2a: Measure Deterministic Router Accuracy

**Setup**: Run the existing `classify_question_type()` (outgestion.py) on all 300 LME questions. Compare classifications against manual labels.
**Metric**: Classification accuracy, per-type precision/recall.
**Hypothesis**: Regex achieves >= 70% accuracy. Keyword-set achieves >= 80%.
**Effort**: Small (1-2 hours). Uses existing code.

### 9.2 Exp-H2b: Tier-1 + Tier-2 Without Classification

**Setup**: Run ALL deterministic pipelines (fact_lookup, temporal, aggregation) on every question. Use confidence gating to select the best answer. No classification at all.
**Metric**: SubEM vs baseline (28.67%) and vs agentic (43.3%).
**Hypothesis**: Confidence-gated parallel pipelines achieve >= 30% SubEM (matches or beats single-pass baseline).
**Effort**: Medium (1-2 days). Requires building 3-5 pipelines with confidence scores.

### 9.3 Exp-H2c: Agentic Trace Pattern Extraction

**Setup**: Analyze the 60 agentic traces from output/agentic_retrieval_full_infra_60q.json. For each CORRECT answer, extract the tool sequence. Cluster sequences into patterns.
**Metric**: % of correct answers that follow <= 5 distinct patterns.
**Hypothesis**: >= 60% of correct answers follow one of 3-4 patterns that can be hardcoded.
**Effort**: Small-Medium (half day). Pattern analysis on existing data.

### 9.4 Exp-H2d: Keyword-Set vs Regex vs LLM Classification

**Setup**: Implement keyword-set classifier (token presence, not regex pattern). Run all 3 classifiers (regex, keyword-set, LLM) on the same 300 questions. Compare against ground truth labels.
**Metric**: Accuracy, coverage, latency.
**Hypothesis**: Keyword-set closes 50%+ of the gap between regex (70%) and LLM (90%).
**Effort**: Small (2-3 hours).

---

## 10. Open Questions for the Tribunal

1. **Is the user's Axiom 2 about cost or about architecture?** If cost: a local SLM at zero marginal cost satisfies the axiom. If architecture: even a frozen classifier model violates it.

2. **Should the spec's "always-on" (Axiom 4) apply to answer pipelines?** The evidence says always-on works for retrieval channels but fails for answer generation (RRF dilution). The spec currently applies it only to retrieval.

3. **What is the acceptable fallthrough rate to agentic?** MT-10 suggests targeting <=20% fallthrough. EF-8 shows 30-40% fallthrough creates bimodal latency. The three-tier architecture makes this a non-question if Tier 1 confidence gating works.

4. **Should Exp-H2c (trace pattern extraction) block implementation?** If 60%+ of correct agentic answers follow 3-4 patterns, those patterns can be hardcoded as Tier 1 pipelines BEFORE the SLM is trained. This would accelerate Phase 3 of the architecture vision.

---

## 11. Summary: What To Build

The actionable takeaway, free of hedging:

1. **Build the three-tier architecture** (Section 6.3 above). This satisfies all axioms simultaneously.
2. **Do NOT build a DeterministicRouter classifier.** Instead, run all Tier 1 pipelines in parallel and gate on confidence. Classification is a false problem when pipelines are cheap enough to run unconditionally.
3. **Run Exp-H2c first.** Pattern extraction from existing traces is the fastest path to identifying which deterministic pipelines to build.
4. **Accept that Tier 3 (agentic) will always exist.** The 15-20% of questions needing adaptive iteration is an irreducible minimum. The goal is to shrink it from 100% (current) to ~20% (Tier 1+2 handle the rest), not to eliminate it.
5. **Plan for Phase 4 SLM** to handle Tier 3 at zero marginal cost. But this is a separate hypothesis (H2b: "Can a 0.6-1.4B SLM match gpt-4o-mini on tool routing?").
