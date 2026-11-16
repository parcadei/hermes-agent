# Proactive Context for MIMIR — A Concrete Design

**Date:** 2026-06-20
**Status:** Design proposal (not yet built)
**Author:** Synthesizer (research synthesis over 4 lanes + verification pass)

---

## What "proactive context" means here

Proactive context = surfacing the right memory at the right time **without** the user or agent issuing an explicit query. Two shapes:

1. **Always-on profile injection** — a two-tier user model (stable facts + recent activity) prepended to the prompt at conversation start, regardless of what the user asks. This is what Supermemory's `/v4/profile` and Honcho's `getContext` ship.
2. **Anticipatory mid-conversation retrieval** — firing a retrieval on a *topic shift* rather than on an explicit search call. **No surveyed production system ships this** (Supermemory, Zep, Mem0, Honcho, ChatGPT, Claude all trigger on turn-start, not mid-conversation topic change). It is the genuine frontier.

MIMIR today is **pull-only**: `recall()` in `foundry/retrieval.py` runs BM25 + cosine RRF + cross-encoder rerank, returns top-10 best-first, only when explicitly called. It never fires proactively. This document specifies the proactive layer we would add *on top of* that core, bounded by what MIMIR's own ~200 experiments proved.

---

## TL;DR

Proactive context for MIMIR is a thin layer over the existing `recall()` core, not a rewrite. At conversation start, inject a **two-tier user profile** (static stable facts + dynamic recent activity), built **offline** by the same SLM extraction pipeline (qwen2.5-3b, 1.7s/turn, no per-request LLM call) and refreshed on a background cadence — mirroring ChatGPT's "Model Set Context" and Supermemory's `/v4/profile`. Relevance-without-query is solved by using the **last 1-2 turns as an implicit query** (Zep's mechanism) feeding the existing `recall()`, plus the precomputed profile that needs no query at all. Mid-conversation, fire anticipatory retrieval **only on a topic shift**, detected by the entity-overlap `segment_events()` boundary that *already exists* in `foundry/ingestion.py` — a free signal, no new classifier. Every injected item must be a **raw source turn** (never a derived observation or summary — MIMIR proved observations-as-evidence cost −3 SubEM), capped at the proven evidence budget (top-20, ~12,000 chars, best-first), because MIMIR proved any deviation regresses. Profile correctness is a hard prerequisite: MIMIR's 11 evidence-injection experiments (D-05 through D-15) were all neutral or negative, so a wrong injected fact is strictly harmful — profile maintenance needs write-time ADD/UPDATE/DELETE/NOOP classification (Mem0/Supermemory), a static/dynamic split with differentiated decay, and version chains so a wrong fact actually dies. The whole layer is **gated** — it must be suppressible to fall back to pure pull-only, because the honest expectation is that on LongMemEval (single long sessions, exact-match SubEM) proactive injection may not beat the disciplined pull-only baseline, and it must never make it worse.

---

## Architecture

The layer has five components, each sitting beside (not inside) the existing deterministic `recall()` pipeline.

### 1. Profile Store (static + dynamic)
**What:** A per-user two-tier fact store. Static tier = stable traits (name, profession, language, long-term preferences). Dynamic tier = recent activity (current projects, recent topics), carrying an explicit `forgetAfter` tombstone.
**How:** Reuse the schema Supermemory ships (`isStatic: bool`, `forgetAfter: date|null`, `version: int`, `isLatest: bool`, `parentMemoryId`, `relation ∈ {updates,extends,derives}`) — verified in `/Users/cosimo/Desktop/supermemory/packages/validation/schemas.ts` lines 242-278. Facts are produced **offline** by MIMIR's existing SLM extraction (qwen2.5-3b, 299-char prompt, 1.7s/turn) over the conversation history, exactly as ChatGPT synthesizes "Model Set Context" out-of-band from ~40 recent conversations. Static facts are exempt from TTL; dynamic facts decay.
**Cost/latency:** Zero per-request cost — profile is precomputed. Extraction cost is amortized at ingestion (1.7s/turn, the production floor MIMIR established). Storage is a small per-user table.

### 2. Proactive Trigger
**What:** Decides *when* to fire — at session start always; mid-conversation only on topic shift.
**How:** Session-start fires unconditionally (inject profile). Mid-conversation reuses the **entity-overlap boundary already computed** by `segment_events(turns, turn_entity_sets, min_overlap=1)` in `foundry/ingestion.py` (verified at `.claude/worktrees/engram-build/foundry/ingestion.py:369`): when a new turn's entity set fails the overlap threshold against the running segment, that *is* a topic shift — and that *is* the trigger. This is MapDia's "proactive topic-shifting detection gate" (arXiv:2503.05150) but free, because MIMIR already segments on this signal. A per-turn LRU cache keyed `(session_id, segment_id, last_user_msg)` absorbs tool-call re-entrancy at zero cost (Supermemory's `MemoryCache`, max 100, verified in `packages/tools/src/shared/cache.ts`).
**Cost/latency:** Entity-overlap check is set intersection — sub-millisecond. No LLM call (respects MIMIR's "no LLM for infrastructure" rule).

### 3. Relevance-Without-Query Scorer
**What:** Produces what to surface when there is no explicit query.
**How:** Two parallel signals, both fed into the existing `recall()`. (a) **Profile mode** — inject the precomputed static+dynamic profile, no query needed (Supermemory "profile" mode, verified in `memory-client.ts`). (b) **Last-turns-as-query** — use the last 1-2 turns as the implicit semantic query into `recall()` (Zep `get_user_context`, P95 <200ms, arXiv:2501.13956). Ranking within the candidate set is **IDF** (BM25), never raw salience: MIMIR proved Fristonian salience alone scores 4/20 (gold ranks 1-111, IDF-blind), and `idf_rerank()` lifts it to 15/20 (`salience-engine-research.md`). Salience is a *recall/candidate-widening* mechanism, not a ranker.
**Cost/latency:** One `recall()` call on the last-turns query (existing latency, to be measured on 500-turn haystacks). Profile injection is a table read.

### 4. Injection / Budget Controller
**What:** Assembles and caps what reaches the prompt.
**How:** Inject profile as a markdown block (`## Static Profile` / `## Dynamic Profile`, Supermemory format) plus, on a trigger fire, raw turns from `recall()`. Hard caps from MIMIR's proven optimum: **top-20 items, ~12,000 chars, best-first order**. A deduplication pass (static > dynamic > search-results priority, Supermemory) prevents double-injecting a fact. **Only raw source turns** are injected — never derived observations or summaries (MIMIR: observations-as-evidence = −3 SubEM; "Search Observations, Serve Turns"). An AbortController-style timeout lets the LLM call proceed without memory rather than block (Supermemory `memoryRetrievalTimeoutMs`).
**Cost/latency:** Bounded by the char budget; injection competes with evidence for context, so the controller must be conservative.

### 5. Freshness / Reconciliation Loop
**What:** Keeps the profile correct over time so we never inject a wrong fact.
**How:** Every write is an **update event**, not an append: write-time LLM classification ADD/UPDATE/DELETE/NOOP against the top-10 similar existing facts (Mem0, arXiv:2504.19413; Supermemory version chains). `UPDATE` supersedes via the version chain (`parentMemoryId`, `isLatest=false` on the old). Static facts only change by explicit correction or DELETE — never by age. Dynamic facts decay with **category-differentiated half-lives** (YourMemory: fact ~24d, assumption ~19d, failure ~11d; +16pp recall@5 over flat decay on LoCoMo-10), with a **salience floor** so rare-but-critical facts (allergy, legal constraint) survive. Runs as a background "sleep-time"/"dreaming" pass (Letta, Honcho) off the critical path.
**Cost/latency:** Off critical path. Write-time classification is one SLM call per ingested batch — acceptable at ingestion, never at query time.

---

## Triggering model — when to fire

| When | How | Tradeoff |
|---|---|---|
| **Session start** | Always inject precomputed profile (no query). | Cheapest (table read, zero query latency) but context-unaware — profile may include facts irrelevant to this session. Mitigated by keeping profile short and high-precision. |
| **Every user turn** | Fire `recall()` with last 1-2 turns as implicit query (Zep/Supermemory pattern). | Most responsive, but adds a retrieval call every turn and competes with evidence budget. On LME's single long sessions this risks injecting noise on every on-topic turn. |
| **Topic shift only** | Fire `recall()` only when `segment_events()` entity-overlap boundary breaks (MapDia gate). | Most principled and **free** (signal already computed). Suppresses noise during on-topic stretches; fires exactly at topic boundaries. Coarser than embedding-cosine drift but zero added latency. **Recommended default for mid-conversation.** |
| **Generation uncertainty** | (Future) fire when responder LLM next-token logprob drops (FLARE, arXiv:2305.06983). | Powerful but requires logprob access from the serving loop; out of scope for a retrieval-only layer initially. |

**Banned by MIMIR:** keyword/intent routing on the *input question* ("Everything always-on. No keyword-based query routing") — gating must be on output/evidence patterns or structural signals (entity-overlap), never English question keywords (fails multilingual). The entity-overlap trigger is language-agnostic and compliant.

---

## Deciding relevance with no query

Three composable signals, in order of cost:

1. **Precomputed profile (no query at all).** The static+dynamic profile is relevant by construction — it is "who this user is," not "what they asked." This is the cheapest and is always available. (Supermemory profile mode, ChatGPT Model Set Context, O-Mem Persona tier.)
2. **Last-turns-as-implicit-query.** Use the last 1-2 turns as the `recall()` query. Solves "no explicit query" by treating recent conversation as the query (Zep, verified). Conv2Query (SIGIR 2025) shows raw dialogue should ideally be *normalized* to a clean query first, but MIMIR's banned-LLM-infra rule means we use the raw last turns directly into BM25+cosine rather than an LLM normalizer.
3. **Candidate-widen with salience, rank with IDF.** Use streaming entity-match (salience engine) to build the wide candidate pool (~125 turns/question, perfect recall@full per `salience-engine-research.md`), then `idf_rerank()` to order it. Never let salience pick the final rank.

**HyDE (arXiv:2212.10496)** is a future option for true cold-start (no turns yet): generate a hypothetical relevant document and retrieve by its embedding. Not in the initial build — it needs an LLM call at session start, which competes with the no-LLM-infra constraint.

---

## Profile maintenance

| Aspect | Approach |
|---|---|
| **Static** | Stable traits (name, profession, language, durable preferences), `isStatic=true`, exempt from TTL/decay. Changed only by explicit user correction or DELETE. Written when the extractor flags a durable trait. |
| **Dynamic** | Recent activity (current projects, recent topics), `forgetAfter` tombstone + `forgetReason`. Decayed by category-differentiated Ebbinghaus half-lives (fact ~24d, assumption ~19d, failure ~11d), `strength = clamp(importance · e^(−λ·active_days) · (1 + recall_count·0.2), 0, 1)`, `active_days` excludes inactivity. Pruned at strength < 0.05. |
| **Freshness** | Background "dreaming"/sleep-time consolidation off the critical path (Letta/Honcho). Profile re-synthesized on a cadence from recent history (ChatGPT model). Search-time recency re-rank (1.5× recent, 0.3× stale) as a soft layer (Mem0) rather than hard deletion. |
| **Dedup / contradiction** | Every write runs ADD/UPDATE/DELETE/NOOP vs top-10 similar (Mem0/Supermemory) — the *only* mechanism shown to keep contradiction rates low; passive aging alone causes rot (LoCoMo degrades 0.455 → 0.05 without active management). Contradiction resolved by version chain: `UPDATE` sets old `isLatest=false`, links `parentMemoryId`. Salience floor protects high-importance facts from decay — **but** a salience-floored superseded fact could resist DELETE and coexist with its replacement; resolution: DELETE/UPDATE (explicit contradiction) overrides the salience floor, decay does not. |

---

## Failure modes + mitigations

| Risk | Mitigation |
|---|---|
| **Wrong/stale fact injected → strict harm.** MIMIR's D-05–D-15 were all neutral or negative; SubEM is hypersensitive to any added text. | Profile correctness is a *prerequisite*, not optional. Write-time ADD/UPDATE/DELETE/NOOP + version chains + decay. Gate the whole layer so it can fall back to pure pull-only. |
| **Over-injection dilutes evidence.** top_k=40 = −2; 30-50 facts = SubEM 24-26 (noise). | Hard cap at proven optimum: top-20, ~12,000 chars, best-first. Dedup pass. Profile kept short. |
| **Observations/summaries injected as evidence.** −3 SubEM at ctx0; 82% of vector index displaced real turns. | Inject **raw source turns only**. Observations may bridge *search* but evidence resolves back to source turn IDs ("Search Observations, Serve Turns"). |
| **Trigger over-fires (every turn).** Mid-conversation injection on on-topic turns adds noise, like supersession over-firing (39→36, −3, on broad keys). | Gate mid-conversation injection on the entity-overlap topic-shift boundary, not every turn. High-precision, narrow trigger. |
| **Timestamps pollute the index.** exp145: timestamps in BM25 text = −4. | Keep ISO timestamps out-of-band in `_turn_timestamps` dict, looked up after retrieval, never in injected text. |
| **Cold start injects empty/hallucinated profile.** | Cold-start vs warm-prompt switching (Honcho): if no profile exists, inject nothing rather than a guess. |
| **Privacy / injection attacks.** Always-on profile can leak sensitive facts or be poisoned via indirect prompt injection (ChatGPT bio-tool spyware pattern; arXiv:2601.11000). | Project-scope the profile (Claude `/memories` model). Never inject facts above the current session's scope. Audit trail via `forgetReason`. |
| **Evaluation blindness.** 41/105 (39%) preference/long-gold failures are SubEM-unscorable; proactively surfaced behavioral context gets no SubEM credit. | Pair SubEM with a noise-aware LLM judge (dual-tier, rescues 13-15/30 preference Qs → 76%). Don't conclude "no gain" from SubEM alone. |

---

## MIMIR constraints this design must respect

- **Evidence budget is fixed:** top-20, ~12,000 chars, best-first. Any deviation (reversal, truncation, widening to 40) regresses. The injection controller cannot exceed this.
- **79% of temporal failures are retrieval misses** — BM25 top-20 insufficient on 500-turn haystacks. A proactive layer that only reranks a narrow set can't fix the dominant failure; it must *widen candidate recall* (salience-for-recall) before ranking.
- **Inject raw turns, not derived text.** Observations-as-evidence = −3 SubEM. Chunk size must be turn-granularity (cs512; cs1024 = −1.33%, cs2048 = −2.67%).
- **No LLM calls for infrastructure.** Triggering, scoring, dedup must be cheap (entity sets, BM25, regex). LLM only at ingestion (extraction, 1.7s/turn) and for the actual task answer.
- **No keyword/intent routing on input.** Gate on structural/output signals (entity-overlap, evidence patterns), never English question keywords (multilingual).
- **Salience = recall, IDF = ranking.** Never let Fristonian salience produce the final rank (4/20). `idf_rerank()` does the ranking.
- **Timestamps out-of-band**, never in BM25 text (−4).
- **No parametric/LoRA memory.** CL is a dead end (1.5B ceiling 17% high-variance, 7B diverges to 0/60). Memory must be an external retrieval store.
- **No LLM-as-retriever / ReAct agent.** −5 SubEM vs deterministic single pass. Route to the deterministic `recall()`, never ask an LLM what to retrieve.
- **SubEM hypersensitivity = correctness prerequisite.** Every evidence-injection experiment regressed or was neutral — so the bar for "proactive injection helps" is high and the design must be gated/reversible.

---

## Build plan (cheapest-highest-value first)

| Step | Value | Effort |
|---|---|---|
| **1. Gated profile injection (static+dynamic), offline-built, session-start only.** Reuse Supermemory schema + MIMIR's SLM extractor. Inject markdown profile block. Behind a flag, fall back to pull-only. | High — ships the core proactive capability competitors have; zero query latency; lowest regression risk because it's gated and session-start only. | Low — extraction pipeline exists; schema is a small table; injection is a prompt-builder change. |
| **2. Last-turns-as-implicit-query into existing `recall()`.** Per-turn proactive retrieval using last 1-2 turns, with the per-turn LRU cache for re-entrancy. | High — solves relevance-without-query using the proven `recall()` core; matches Zep/Supermemory. | Low — `recall()` already exists; wrap it with the last-turns query and cache. |
| **3. Freshness/reconciliation loop.** Write-time ADD/UPDATE/DELETE/NOOP + version chains + category decay + salience floor, as a background pass. | High — without it the profile rots and a wrong fact is strictly harmful (D-05–D-15). Prerequisite for trusting injection. | Medium — needs the classifier call at ingestion + version-chain storage + decay job. |
| **4. Topic-shift-gated mid-conversation retrieval.** Fire `recall()` on the `segment_events()` entity-overlap boundary. The genuine frontier no competitor ships. | Medium-High — the differentiator (true mid-conversation surfacing). Risk: may add noise; must be ablated against pull-only on LME. | Low-Medium — the boundary signal already exists in `foundry/ingestion.py`; wire it as the trigger + reuse step 2's machinery. |
| **5. Dual-tier evaluation harness.** SubEM + noise-aware LLM judge, to measure proactive-vs-reactive honestly (esp. on the 39% preference Qs SubEM can't score). | Medium — needed to know if any of the above actually helps; otherwise we're flying blind on a metric that penalizes added text. | Medium — judge harness partially exists; extend to score proactive runs. |
| **6. (Future) HyDE cold-start + FLARE uncertainty trigger.** Hypothetical-doc retrieval at session start; logprob-triggered mid-generation retrieval. | Medium — closes cold-start and within-generation gaps. | High — HyDE needs a session-start LLM call (tension with no-LLM-infra); FLARE needs logprob access from the serving loop. |

---

## Open questions

- Actual `recall()` round-trip latency on a 500-turn haystack is unmeasured — needed before committing to a per-turn trigger budget.
- Whether topic-shift-gated injection nets positive on LME's *single long sessions* (most proactive systems target cross-session) — must be ablated, not assumed.
- How to set `isStatic=true` at write time (Supermemory's backend logic isn't public) — LLM-classified vs entity-type rule vs user-set.
- Quantitative gap: no published controlled ablation of proactive-vs-reactive on the same benchmark exists — MIMIR would be measuring this fresh.
- The path `foundry/ingestion.py` referenced for `segment_events()` is currently only in `.claude/worktrees/engram-build/` on this branch — the trigger wiring must target wherever foundry lands in the main checkout.

---

## A note on cited numbers (verification)

Two research figures were **refuted** in verification and are *not* relied on above: PASK's IntentFlow overhead is **1.3-1.5 seconds, not milliseconds** (arXiv:2604.08000 Table 4) — so PASK's demand-detector is a full LLM call, not a free sidecar, which is *why* this design uses MIMIR's free entity-overlap signal instead. Mem0's "18ms search" is unsupported; the paper's search latency is p50≈148ms, p95≈200ms (arXiv:2504.19413). Supermemory's "50ms" is unverified marketing; its source code contains only an AbortController timeout with no hardcoded latency. Confirmed and load-bearing: Supermemory's in-process LRU cache + role-check trigger (read directly from source); Zep P95<200ms and LongMemEval +18.5% accuracy / 90% latency reduction (arXiv:2501.13956); MIMIR's `segment_events()` entity-overlap topic-shift logic (verified at `.claude/worktrees/engram-build/foundry/ingestion.py:369`).
