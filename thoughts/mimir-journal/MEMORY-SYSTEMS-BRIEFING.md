# Memory Systems — Synthesis Briefing

> A consolidated briefing distilled from (a) a full teardown of **MIMIR** — a 3-month AI long-term-memory research project (Feb 27 – May 30, 2026; 347 sessions, ~200 experiments, 94 distinct approaches, 335 sorry-free Lean theorems), and (b) a teardown of **supermemory** (the open-source SOTA product) plus a deep-research pass on **proactive context**. Written for review by another agent working on a harness project. Companion artifacts (journal webpage, structured data, proactive-context design) are indexed at the bottom.

---

## 0. TL;DR (the load-bearing conclusions)

1. **Retrieval is solved; "memory" mostly collapses into it.** MIMIR's honest best was **60.33% SubEM** on LongMemEval-S* 5-ctx (181/300) — #1 among named systems (GPT-4.1-mini 55.7%, MIRIX 51.0%, HippoRAG-v2 50.7%). That number is **plain hybrid RAG**: turn-level BM25 + dense cosine fused with RRF + a cross-encoder reranker, top-20 → top-10 best-first → LLM generates the answer. Almost every attempt to be *more* than RAG (knowledge graphs, energy/Hopfield, parametric weights, semiotics) tied it or hurt.
2. **The single lever that matters is write-time intelligence.** The gap from ~60% → 81–90% (supermemory, Honcho) is **not** a cleverer retriever — it's the quality of the model that reads each turn and decides what to write down. Honcho hits 90.4% with **frontier models** (Gemini extraction + Claude reasoning), free-form NL observations, multi-step deriver→dialectic→dreamer reasoning. MIMIR used a local Qwen2.5-7B/3B extractor (cheap), and that cost choice is most of the ceiling.
3. **"Memory must be cheap / no LLM for infrastructure" was MIMIR's self-imposed rule — and also its ceiling.** Keeping the infra model-light (BM25, regex routing, Allen temporal, a small embedder + reranker, optional SLM extractor) capped quality around 60%. There is no free lunch: quality is bounded by inference spent per stored turn.
4. **The only genuinely non-RAG bet that showed life was TTT** (test-time training, per-query weight adaptation): 30% probe vs 12% offline ceiling. Everything else parametric (LoRA/EWC/MEGa, full FT) failed — 1.5B capped ~17% high-variance, 7B *diverged* to 0/60.
5. **The real frontier is continuous learning, and it needs a new benchmark.** LongMemEval/SubEM structurally rewards finding-the-right-turn, so it will always conclude "BM25 was enough." A benchmark a frozen retriever *cannot* pass (belief updated across sessions; answer in no single turn) is a prerequisite for studying anything beyond retrieval.

---

## 1. Is it RAG? (Yes — and that's the honest field state)

MIMIR was RAG at its core for essentially the whole run. The empirical winner = retrieve (BM25 + cosine RRF + cross-encoder rerank over conversation turns) → augment → generate. The "memory-flavored" pieces stayed inside that frame:
- **Observations / dual-extraction** ("search observations, serve turns") = index enrichment, then retrieve-and-generate.
- **EventFrames / episodic** = contiguity on top of retrieval.
- **Temporal (Allen/MSL), KG/structured query** = alternate retrieval channels bolted onto the same pipeline.

The **only genuinely non-RAG branch** was parametric/continual-learning (knowledge in weights, nothing to retrieve) — which failed at every scale except TTT. NLCDM/Hopfield was *different* retrieval (content-addressable), and was strictly worse than cosine.

Strip the branding off every shipping system (supermemory, Mem0, Zep, Honcho, MIMIR) and it's the same shape: **a write-time LLM transform feeding an external store feeding RAG.** Nobody has memory in the weights; nobody has continuous learning. The competition is entirely on *how smart the writer is*.

---

## 2. The memory-mechanism landscape (taxonomy)

Sorted by **where memory lives**. `[tried]` = MIMIR ran it; `[open]` = unexplored.

- **A. External addressable (RAG-family):** vector/hybrid RAG `[tried — won]`, knowledge graphs `[tried — lost to flat text]`, recursive SLM oracle `[tried — neutral]`, NTM/DNC `[open]`, key-value Memory Networks `[open]`.
- **B. Recurrent compressed state:** RNN/LSTM → **SSMs (S4/Mamba/Mamba-2)** `[open]`; linear-attention/fast-weight RNNs (**GLA, DeltaNet, Gated DeltaNet, RWKV, RetNet, xLSTM**) `[open]`; **reservoir computing / ESN / LSM** `[open]`.
- **C. Fast-weights / test-time learning:** Fast Weight Programmers, Hebbian/differentiable plasticity `[open]`; **TTT layers** `[tried — only live signal, 30%]`; **Titans** line `[tried — failed]`; mesa-optimization `[open]`.
- **D. Associative / energy / symbolic:** **Modern Hopfield** `[tried — dead, wrong abstraction]`; Sparse Distributed Memory (Kanerva) `[open]`; **HDC/VSA/HRR/TPR** `[open]`.
- **E. Knowledge editing (write a fact to MLP weights):** ROME, MEMIT, MEND, GRACE, WISE, **AlphaEdit** `[referenced via Sleeping LLM — open]`.
- **F. Context-as-memory / compression:** gist tokens, ICAE/AutoCompressor `[open]`; KV-cache as memory (StreamingLLM/H2O/landmark) `[open]`; **Cartridges (offline KV self-study)** `[open — interesting]`.
- **G. Neuro-inspired:** CLS / wake-sleep consolidation `[tried — dream/NREM killed]`; predictive coding / **JEPA** `[tried — collapse, dead]`; cognitive maps (successor representations, TEM) `[open]`; replay `[partial]`.
- **H. Probabilistic/Bayesian:** **Kalman/particle filters** `[tried — salience engine, partial]`; Bayesian online, neural processes `[open]`.

**The fault line that predicts every result:** *bounded-state vs unbounded-store.* LongMemEval over a 500-turn haystack is an unbounded-recall task; a fixed-size state (Mamba, reservoir, Hopfield basin, compressed KV) **cannot losslessly hold it** (pigeonhole). So families B, D, F-compression lose to retrieval *by construction* — exactly why Hopfield died at N=1200. Those families are only worth running for the *continuous-learning* goal, not the benchmark.

**Highest-value open bets:** (1) **Gated DeltaNet / Titans-family as online associative memory** — generalizes the TTT signal that worked; (2) **knowledge editing (AlphaEdit/MEMIT)** — the in-weights write path, targets the interference wall that killed LoRA; (3) **Cartridges** — offline KV self-study, sidesteps both retrieval crutch and LoRA interference.

---

## 3. Multi-hop — how MIMIR tackled it

**Machinery (MABench/FactConsolidation era):** one composed config — `--cooc_boost --triadic --decompose_query --conflict_resolution --cr_per_hop --bm25_hops`:
1. **Query decomposition** — split into per-hop sub-queries.
2. **Iterative per-hop retrieval** (`bm25_hops`) — retrieve hop 1, extract its entity, re-query hop 2…
3. **Subject-index KG graph walk** — entity→fact→entity traversal; moved FC-MH **40 → 53**, "KG resolved 71% of hops"; single-hop 67 → 91 (beat GPT-4o 60 by 31, later partly eval-inflated).
4. **Entity-constrained hops (HyperedgeIntersection)** — hard filter → **SubEM=9 catastrophe**; softened to 0.1× penalty.
5. **Co-occurrence + triadic + per-hop conflict resolution.**
- Also a **rustworkx PPR (HippoRAG-style)** module built but never validated end-to-end (0.85⁴≈0.52 decay drowns 4-hop signal in dense graphs).

**Core insights:** multi-hop accuracy is **multiplicative** (0.7⁴ = 24% for 4 hops); and **70% of failures are at HOP 1** (61% entity confusion). So multi-hop ≈ a hop-1 retrieval/entity-resolution problem, not a reasoning problem.

**Verdict:** the entire FC multi-hop stack ported to LongMemEval scored **2.67% (vs cosine 16.67%)** — catastrophic (vocabulary mismatch on conversational text). The `lme_multihop` helper actively hurt (round-robin session diversification scattered top-k). **Landing position:** on conversational haystacks, multi-hop collapses back into **retrieval recall** — widen the candidate set so every hop's evidence is present and let the LLM chain in-context. Don't build a hop-reasoner.

---

## 4. Did MIMIR solve "the right information at the right time"?

**Partially.** It solved the *"too much"* half and found the sweet spot; it did **not** solve the product-grade capability.

**Solved (the "too much" curve):** over-fetch dilutes (30–50 facts → SubEM 24–26; char_budget 20K → −3; truncated/reversed evidence → −3 each); under-fetch loses recall (retrieve_num=3 → 24.67% vs 31.33%). **Sweet spot = top-10, best-first, full context.**

**Not solved (the product gaps):**
1. **Recall on big haystacks (#1 open):** ~79% of failures were *retrieval misses* — the right turn wasn't in top-20. Can't fix by widening (re-triggers dilution). This is candidate **generation** recall, not ranking.
2. **Proactive / "right time":** MIMIR was *pull-only*. No always-on profile (static + recent), the thing supermemory ships (`profile()`, ~50ms). Biggest product gap; cheap to add → see §6.
3. **Freshness / serving the current fact:** supersession/conflict detection hit 79% F1 but over-fired on broad keys; NEI/abstention partial with false positives.

---

## 5. The recommended "best memory system to set up"

**A fact-extraction store — not a clever retriever, not a learning model.** Spend the intelligence at *write* time; keep retrieval boring. Five layers:

1. **Write-time extractor (where the budget goes):** per turn/doc, extract *atomic grounded facts*, each with `source_turn_id(s)`, `timestamp`, `valid_from/forget_after`, `confidence`, `subject+predicate` key. Strongest model you can justify; or a **distilled** SLM with a **schema→deriver** guardrail (MIMIR's dual-extraction got zero-hallucination on 3B by constraining output space). This distillation is the bridge from 60% to 81%.
2. **Reconciliation at write time:** new fact contradicts old on same subject+predicate → **version it** (supersession), not append; time-bound facts get expiry. Done once at ingestion, never at query time.
3. **Dual store: facts + raw turns.** Retrieve on facts/observations as search keys, but **serve the raw source turn as evidence** (MIMIR: observations-as-evidence = −3). Distilled facts are great keys, terrible sole evidence.
4. **Retrieval = the solved thing:** hybrid BM25 + dense → RRF → cross-encoder rerank over *both* facts and turns, one fused ranked list. Plus a **static/dynamic profile** as an always-on cache.
5. **Integration loop:** retrieve-before / store-after middleware (modes `profile | query | full`). Deterministic, not agentic (ReAct = −5).

**Reference implementation:** supermemory's design is the right template (see §7); MIMIR's two upgrades are *"serve turns, not facts"* and *"distill the extractor."* The frontier worth a research name is continuous learning — and it needs the new benchmark first.

---

## 6. Proactive context — the design (full doc: `proactive-context-design.md`)

**It's a thin *gated* layer over `recall()`, not a rewrite.** Falls back to pure pull-only behind a flag so it can never regress.

**Five components:** (1) **Profile store** (static traits, no decay + dynamic recent activity with `forgetAfter`), built **offline** by the existing SLM extractor — zero per-request cost; (2) **Proactive trigger** — session-start always; mid-conversation **only on topic shift**, detected by the **free entity-overlap boundary `segment_events()` already computes** (`foundry/ingestion.py:369`) — no classifier, no LLM; (3) **Relevance-without-query** — precomputed profile (no query) + last 1–2 turns as implicit query (Zep); widen with salience, **rank with IDF** (salience-only 4/20 → idf_rerank 15/20); (4) **Injection/budget controller** — top-20, ~12K chars, best-first, **raw turns only**; (5) **Freshness loop** — write-time ADD/UPDATE/DELETE/NOOP vs top-10 similar (Mem0/supermemory) + version chains + category decay, off critical path.

**Hard constraints (MIMIR's scar tissue):** fixed evidence budget; raw turns not derived text; **no LLM in infra path**; no keyword/intent routing (gate on structural signals — multilingual); timestamps out-of-band (in-text timestamps = −4); salience=recall, IDF=rank; SubEM hypersensitivity means **profile correctness is a prerequisite** (D-05…D-15 all neutral/negative — a wrong injected fact is strictly harmful), hence the gate.

**Build plan (cheapest→highest value):** (1) gated profile injection, session-start [Low/High]; (2) last-turns-as-implicit-query [Low/High]; (3) freshness/reconciliation loop [Med/High]; (4) topic-shift-gated mid-conversation retrieval — the differentiator no competitor ships [Low-Med/Med-High]; (5) **dual-tier eval (SubEM + noise-aware judge)** [Med] — *needed*, because SubEM penalizes added text and can't score 39% of preference questions, so it will falsely report "no gain."

**Open caveat:** no controlled proactive-vs-reactive ablation exists on the same benchmark; on LME single long sessions topic-shift injection may not beat the baseline — **must be ablated (step 5 gates everything), not assumed.**

> Verification note: the research pass caught two bad numbers — IntentFlow latency is **1.3–1.5 seconds, not ms** (1000× error; flipped the trigger recommendation toward the free structural signal), and Mem0's "18ms" search figure is fabricated (real p95 = 200ms). Both corrected in the design doc.

---

## 7. supermemory teardown (the reference product)

**Crucial framing:** the cloned repo is the **client/SDK/integration/visualization layer**; the **memory engine is hosted/closed** (`api.supermemory.ai`). But its data model is fully exposed via typed API contracts.

**The model (`packages/memory-graph/src/api-types.ts`):** a `MemoryEntry` has `isStatic` (static profile vs dynamic activity); `version`/`parentMemoryId`/`rootMemoryId`/`nextVersionId`/`isLatest` (**versioned chains** = supersession); `relation: updates|extends|derives` (3-edge ontology); `isForgotten`/`forgetAfter`/`forgetReason` (**first-class forgetting**); `embedding`; `spaceId`/`containerTag` (scoping). Documents → extracted Memories → versioned + related.

**API:** `add` (queued async extraction), hybrid `search` (returns `{memory}` or `{chunk}` — memory + RAG one list), `/v4/profile` (`{static[], dynamic[]}`, fast), `/v4/conversations` (store turn → engine extracts), `forget` (exact then semantic @0.85).

**Integration = retrieve-before / store-after middleware** (TS `vercel/middleware.ts`, Python `context_provider.py`), modes `profile|query|full`. The cleverness is all server-side; the client is thin. **MCP server** = 3 tools (`memory` save/forget, `recall`, `context`/profile resource) on a Cloudflare Durable Object + OAuth.

**Stack:** Bun + Turbo monorepo; Hono; Cloudflare Workers/Durable Objects/Workflows/Hyperdrive/AI; Drizzle; Better Auth.

---

## 8. Artifact index (paths for review)

All under `/Users/cosimo/.hermes/hermes-agent/thoughts/mimir-journal/`:

| File | What it is |
|---|---|
| **`mimir-journal.html`** (326 KB) | Self-contained research-journal **webpage** — the full MIMIR record: 94 approach cards (filterable), 61-event timeline, 53-row leaderboard, 38 dead-ends, 5 era chapters, formal proofs, reflection. Open in a browser. |
| **`journal_data.json`** (220 KB) | Structured data behind the page (intro/reflection, timeline, approach_cards, leaderboard, dead_ends, formal_proofs, families, eras, stats). Machine-readable. |
| **`proactive-context-design.md`** (20 KB) | Full proactive-context architecture + build plan + failure modes + verified citations. |
| **`MEMORY-SYSTEMS-BRIEFING.md`** | *This file* — the consolidated synthesis. |
| `build.py`, `styles.css`, `app.js` | Build pipeline for the journal HTML (regenerate via `python3 build.py`). |
| `session_index.json` (242 KB) | Index of all 347 mined Claude Code sessions (path, dates, summary, first prompt). |
| `source_manifest.json`, `extract_session.py` | Source manifest + the bounded session-digest extractor used during mining. |

**External clone (for the supermemory teardown):** `/Users/cosimo/Desktop/supermemory/` — key files: `packages/memory-graph/src/api-types.ts`, `apps/mcp/src/{server,client}.ts`, `packages/tools/src/{shared/prompt-builder,vercel/middleware}.ts`, `packages/agent-framework-python/src/supermemory_agent_framework/context_provider.py`.

**MIMIR memory notes (primary record):** `/Users/cosimo/.claude/projects/-Users-cosimo--hermes-hermes-agent/memory/*.md` — esp. `MEMORY.md`, `failed-approaches.md`, `salience-engine-research.md`, `episodic-memory-research.md`, `oracle-experiments.md`, `cl-experiments.md`, `parametric-memory-landscape.md`.
