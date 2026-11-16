# Does Continuous Learning Require an Architecture Change? — A Buildable-Path Assessment (2025–2026)

**Research question.** Is the thesis *"continuous/lifelong learning requires a change to the base MODEL architecture + training framework (not just a bolt-on memory layer)"* a **real, buildable path** in 2025–2026, or is it still **research-frontier-with-no-recipe**?

**Scope note.** This project has already established (internally and against the literature) that *retrieval memory is largely solved* — the team's own Foundry/Engram stack reaches 60–81% SubEM on LongMemEval, topping published systems. The open problem is **continuous learning INTO weights from a stream**: accumulating knowledge/skill across sessions without catastrophic interference. Parametric attempts failed (LoRA-continual capped ~17%; a 7B diverged; only TTT broke the offline probe ceiling, 30% vs 12%). The hypothesis under test is that the fix is a **modular/sparse architecture** — Mixture-of-Experts where new knowledge lands in a routed expert without overwriting others, plus specialized post-pretraining (on-policy distillation / RL), sparse/multi-latent attention, and hybrid attention+SSM.

---

## VERDICT: research-only (for the strong thesis); emerging/buildable-now only for narrower pieces

The **strong version** of the thesis — *online, open-ended, modular-expert weight-learning into a live LLM at frontier scale* — is **research-frontier with a precisely-named-but-uncrossed wall**. There is **no production recipe** in 2025–2026. However, the thesis is **directionally correct**, and several **narrower architecture-change primitives are buildable now** with real quantified wins (Meta Sparse Memory Finetuning; AlphaEdit sequential editing; test-time-training memory; offline specialist-then-distill). The gap between "buildable pieces" and "the thesis as the user means it" is the crux of this report.

**One-sentence verdict:** The architecture-change thesis is right in spirit and now shares vocabulary with frontier labs, but the specific mechanism the user hypothesizes (online routed-expert continual learning) has a named wall (router/gater forgetting) that no published method crosses at scale — so it is *research-only* today, even though *narrower* architecture-change methods are shippable.

---

## TL;DR

1. The field **agrees** architecture/training-framework change is needed (not just bolt-on memory) and has **named the exact walls** the user predicted — but no published method crosses them at frontier scale on an open stream.
2. The hypothesized wall is **real and named**: *"router/gater forgetting"* + *"router drift"* + *"expert drift"* — the routing network catastrophically forgets even when expert weights are frozen (arXiv 2605.17571, 2512.10969, 2601.13020).
3. The **optimistic premise is refuted**: MoE sparsity/modularity does **not** by itself prevent forgetting (arXiv 2602.12587 — MoE Transformers "still forget substantially even with sparse, well-balanced expert utilization").
4. The user's proof-point **GLM-5.2 "~10 experts via RL" is verified as fact but is the WRONG mechanism**: it is *offline, one-shot* multi-teacher on-policy distillation of whole-model specialists into ONE dense generalist — a full consolidation run per update, not online routed-expert continual learning.
5. **Offline specialist-then-distill (OPD/MOPD)** is *shipping and buildable* across GLM/DeepSeek-V4/MiMo-V2/Kwai — but it solves *multi-domain RL interference*, not continual learning.
6. **Hybrid attention+SSM** ("mixture of architectures") is a real trend but **efficiency-motivated** (long context), with **no demonstrated link to cross-session forgetting** — including it in the CL thesis is speculative.
7. **TTT / Titans / Nested-Learning-HOPE** are the closest published instantiations of "change the architecture + training framework," but they **memorize within a sequence and reset per session**; HOPE's own authors say catastrophic forgetting is *not* solved (validated ≤1.3B).
8. **Forgetting SOTA is narrow**: best methods (Sparse Memory Finetuning 89/71→11%; AlphaEdit) work on controlled fact-streams / discrete triples at ≤1.3–8B — not open-ended skill accumulation at scale; and forgetting *worsens* with scale (1–7B).
9. **The eval gap is the biggest opportunity**: NO benchmark yet publishes the matched ablation *frozen-retriever-at-realistic-budget vs actual-weight-updated-model, latter winning*. Every "needs real learning" wall so far was closed by smarter non-parametric engineering.

---

## Section 1 — GLM / DeepSeek / Qwen reality (and what is unverifiable)

**GLM-5.2 "~10 experts via RL" — VERIFIED FACT, WRONG MECHANISM.** Z.ai's disclosure confirms GLM-5.2 trained **>10 per-domain RL specialist checkpoints** and merged them via **on-policy distillation** (the `slime` framework, ~2 days) into a **single dense generalist**. This is **offline, one-shot consolidation** — the modular boundary is *dissolved* into shared weights. It is the **opposite** of "new knowledge lands in a routed expert without disturbing others": it is a *full re-consolidation run per update*.

- GLM-5's **deployed** architecture is a **static capacity-MoE**: 256 experts, 744B total / 40B active, Multi-Latent Attention (arXiv 2602.15763, verified). This is frozen sparse FFN trained jointly — *not* a continual substrate. This distinction (capacity-MoE vs modular-experts-as-continual-substrate) is exactly the one the research brief flagged, and the public evidence lands entirely on the *capacity* side.
- The **same offline OPD pattern** appears across the frontier: DeepSeek-V4 (arXiv 2606.19348, secondary summary), MiMo-V2 (2601.02780), Kwai Keye-VL-2.0 (13 teachers, 2606.10651), formalized as **MOPD** (2606.30406). Qwen's public MoE releases are likewise capacity-MoE + offline post-training.

**UNVERIFIABLE (flagged honestly).** Whether any lab has an *internal online modular continual-learning* solution is **unknown and unpublished**. Every *disclosed* technique is offline. It would be a mistake to treat GLM-5.2 as evidence that "online routed-expert learning works." Proprietary post-training internals cannot be verified from public sources; do not assume a secret solution exists.

---

## Section 2 — Lifelong MoE + router forgetting (the core wall)

**The wall is real and now precisely named.** *"Router/gater forgetting,"* *"router drift,"* and *"expert drift"* ("misaligned co-drift") describe a studied failure mode: the routing network itself catastrophically forgets, and newly-added experts **hijack routing for old tasks** — *even when old expert weights are frozen*.

- StaR-MoE (arXiv 2605.17571, verified): router reassigns old-class samples to new experts despite frozen experts.
- MoB "gater forgetting" (2512.10969); PASs-MoE (2601.13020); theory that the **gate must stop updating** for convergence (2406.16437, ICLR 2025).

**The optimistic premise is refuted.** MoE sparsity/modularity does *not* auto-prevent forgetting. Multi-head attention creates a "pre-routing bottleneck" that collapses distinct routing needs; even the fix leaves nonzero forgetting.

- arXiv 2602.12587 (verified): MoE Transformers "still forget substantially even with sparse, well-balanced expert utilization"; the MH-MoE fix reduces backward-transfer degradation only **−BWT 11.2% → 4.5%** at ≤8B on TRACE.

**Partial fixes exist — but only at small/mid scale on bounded benchmarks.** ~A dozen 2025–2026 papers beat naive expert-append via (a) freeze + KL-align routing, (b) buffer/transient experts, (c) activation-subspace routing, (d) content-addressable rank-1 experts that *remove the learned router entirely*. **All** are on vision class-incremental learning (ViT-B/16) or ≤8B LLMs with **finite, known task boundaries** — none on unbounded open streams at frontier scale.

- StaR-MoE (2605.17571), CP-MoE (2605.20247), MoSEs on TRACE (2511.06237, low-confidence read), MoRAM (2506.21035, ICML 2026).

**Bottom line:** *online expert addition + router forgetting at frontier scale on an open stream is UNSOLVED.* This is the exact wall the user predicted.

---

## Section 3 — On-policy distillation / expert-specialized RL

**OPD/MOPD is shipping and buildable** — an industry-standard post-training primitive: *train N per-domain RL specialists → multi-teacher on-policy distill into one generalist* (GLM-4.5/5.2, DeepSeek-V4, MiMo-V2, Kwai Keye-VL-2.0; formalized MOPD 2606.30406).

- **What it buys:** resolves multi-domain RL **"see-saw" capability interference** — an *offline, per-release batch* consolidation. A small team can adopt it as a **scheduled re-consolidation** primitive (periodically distill accumulated interactions into the model).
- **What it does NOT buy:** online/continual learning. It is a full consolidation run per update and dissolves modular boundaries into shared dense weights. It is the *wrong mechanism* for the user's thesis but a *real, buildable* primitive for **periodic** (not continuous) learning.

---

## Section 4 — Hybrid attention + SSM ("mixture of architectures")

**Real trend, wrong axis.** Jamba / Bamba / Nemotron-H / Zamba / TransMamba, plus Multi-Latent Attention and sparse attention, are an active architecture direction — but they are **efficiency-motivated** (long-context compute/memory), not continual-learning-motivated. There is **essentially no demonstrated connection** between hybrid attn+SSM and mitigating cross-session catastrophic forgetting.

- The only tangential work (arXiv 2602.12204) reframes plasticity/stability *within a single training run*, not across sessions.
- **Honest call:** conflating "mixture of architectures" with the continual-learning story is **speculative extrapolation**, not established in the literature. It helps context length and throughput — a different problem than accumulation.

---

## Section 5 — Forgetting SOTA (plasticity–stability) and its limits

Stability–plasticity is **structurally unresolved** (2026 surveys, arXiv 2602.06453; MemoryBench 2510.17281 finds SOTA "far from satisfying"; forgetting *worsens* with scale in the 1–7B range, 2606.24752).

Best current methods, each **narrow**:

- **Meta Sparse Memory Finetuning** (arXiv 2510.15103, verified): parameter-localized sparse gradient updates cut forgetting from **89% (full FT) / 71% (LoRA) → 11%** on a controlled fact stream at ~1.3B. *The single most promising buildable primitive for the thesis.* Limit: controlled stream, small scale.
- **AlphaEdit** null-space-constrained editing (arXiv 2410.02355, ICLR 2025 Outstanding, +36.7% over locate-then-edit): survives many *sequential* edits without collapse. Limit: discrete fact-triples, not skills.
- Continual-MoE router-alignment fixes (Section 2): small scale, known boundaries.

**All** operate on controlled streams / fact-triples / ≤1.3–8B — none demonstrate open-ended skill accumulation at scale.

---

## Section 6 — TTT / Titans / Nested Learning as a continuous-learning substrate

These are the **clearest published instantiations of the exact "change the architecture + training framework" thesis** — and the honest news is mixed.

- **Nested Learning / HOPE** (Google, *Nested Learning: The Illusion of Deep Learning Architectures*, arXiv **2512.24695**, NeurIPS 2025, OpenReview `nbMeRvNb7A`, verified 2026-07-02): architecture-as-optimizer-levels + multi-frequency "Continuum Memory." **The authors themselves state catastrophic forgetting is NOT solved in general** (only mitigated on studied tasks; "a roadmap not a destination"), validated only to **1.3B params / 100B tokens** (15.11 WikiText PPL, 57.23% avg reasoning).
- **Titans** (arXiv 2501.00663): test-time neural memory. Real long-context wins (2M–10M tokens), but Titans-as-a-Layer **resets at dialogue boundaries** (2606.08573).
- **TTT-E2E** (2512.23675): per-document; memorizes *within* a sequence and is reset per session. Simplest TTT reduces mathematically to **linear attention** (2602.21204). Hardware-inefficient pre-optimization (<5–10% FLOP util, 2511.07343).

**Does it accumulate?** *Within-context: yes.* *Across sessions: no.* The team's own TTT-E2E result (30% probe vs 12% offline ceiling) is a **real per-query signal** but is **not persistent** — it is smarter long-context modeling, not lifelong memory. The key missing bridge is **making test-time-learned state persist and consolidate across sessions** without interference.

---

## Section 7 — The eval gap (the field's — and the team's — biggest opportunity)

**No benchmark yet publishes the decisive ablation:** *frozen retriever at a realistic budget* **vs** *actual weight-updated model*, with the **latter winning**. The 2026 wave of benchmarks (MemoryAgentBench 2507.05257; CL-Bench 2606.05661; AgentCL; RefMem-Bench; MemoryBench 2510.17281) is purpose-built to *punish* frozen retrieval — itself an admission that retrieval is largely solved — yet **every claimed "needs real learning" wall was substantially closed by smarter non-parametric engineering**:

- MemoryAgentBench multi-hop FactConsolidation (~7–30% ceiling) → **78% SH / 30.2% MH** via a deterministic BM25→extract→max(serial) pipeline (2606.01435, verified).
- CL-Bench: naive **full-context ICL beats** dedicated memory systems (Mem0, ACE) — no weight-update ablation reported.
- LongMemEval is fakeable by BM25+cosine (the team's own 60–81% SubEM).

**Implication:** there is **no clean "RAG-can't-fake-it, weight-update-can" existence proof**. Building one is the highest-leverage scientific contribution available here — and a prerequisite before betting a product on weight-learning.

---

## Buildable paths — ranked maturity table

| # | Path | Maturity | Effort | What it buys | Key refs |
|---|------|----------|--------|--------------|----------|
| 1 | **Sparse Memory Finetuning** (parameter-localized sparse gradient updates) | buildable-now / emerging | Medium | Cuts forgetting 89%FT/71%LoRA → 11% on a controlled fact stream (1.3B). Directly targets the interference that killed LoRA-continual. | 2510.15103 |
| 2 | **Corrected sequential knowledge editing** (AlphaEdit null-space) | buildable-now | Low–Med | Inject discrete fact-triples surviving many sequential edits without collapse. Narrow (facts, not skills). | 2410.02355 |
| 3 | **Test-time-training memory layer** (Titans/TTT within-context) | emerging | Med–High | Within-sequence memorization; breaks offline probe ceiling (30% vs 12%). Does NOT persist cross-session (must add consolidation). | 2501.00663, 2512.23675 |
| 4 | **Offline specialist-then-distill** (OPD/MOPD) as *scheduled* re-consolidation | buildable-now (shipping) | High | Periodic batch "learning" without multi-domain see-saw interference. NOT online — one full run per update. | 2606.30406, 2606.19348, GLM `slime` |
| 5 | **Continual-MoE**: frozen experts + router KL-alignment / content-addressable experts | emerging → research-only | High | Append experts on *known* task boundaries at small scale; mitigates router forgetting. Not open-ended/frontier. | 2605.17571, 2601.13020, 2506.21035 |
| 6 | **Nested Learning / HOPE + Continuum Memory** (multi-timescale optimizer levels) | research-only | Very High | Closest full instantiation of the thesis; but forgetting *not* solved, ≤1.3B, pretrain-from-scratch. | 2512.24695, 2501.00663 |
| 7 | **Modular ICL+IWL hybrid** (position-paper direction) | no-recipe | Research | The conceptual target the field is converging on; no benchmark or recipe yet. | 2603.01761 |

---

## Honest risks

- **The strong thesis has a named-but-uncrossed wall.** Betting a *product* on online modular-expert learning at scale is premature; the wall (router/gater forgetting) is documented but unsolved.
- **Proprietary internals are unverifiable.** GLM/DeepSeek/Qwen post-training internals are opaque; all *disclosed* techniques are offline. Do not assume a secret online-CL solution exists.
- **TTT/Titans "memory" resets per session.** Easy to mistake within-context memorization for lifelong learning; the persistence/consolidation step is the unsolved part.
- **Hybrid attn+SSM is efficiency, not CL.** Conflating the two is a category error.
- **Forgetting worsens with scale.** Small-scale wins (≤1.3B) may not transfer to frontier models.
- **Every "learning-required" benchmark wall has fallen to better retrieval.** The *need* for weight updates is asserted, not yet proven — spend effort proving it before building on it.
- **OPD/offline consolidation costs a full run per update.** It is "periodic," not "continuous," and dissolves modular boundaries.

---

## Recommendation for a small team

**Split the bet: product on the solved layer, research on the one buildable primitive that attacks the real wall.**

- **Product path (low risk, ship now):** keep the retrieval-memory stack (already SOTA at 60–81% LongMemEval) as the product's "memory." Add **scheduled offline re-consolidation** (path #4): periodically distill accumulated interactions into a LoRA/expert on a cadence. This is the honest, buildable form of "the model learns" — *periodic*, not continuous. Do **not** market it as online lifelong learning.

- **Research path (high signal, attacks the thesis directly):** the **single highest-signal next experiment** is to **reproduce Meta Sparse Memory Finetuning (2510.15103) on the team's own continual LongMemEval stream** and measure (a) accumulation of new facts across sessions and (b) forgetting of old facts, **against the frozen-retriever baseline at a matched budget**. This is the exact regime where LoRA-continual failed (capped ~17%) and where TTT showed signal (30% but non-persistent). Sparse parameter-localized updates are the most credible published mechanism for *persistent, low-interference* weight learning at the team's scale — and running it produces, as a byproduct, the **missing eval existence proof** (Section 7): *does a weight-updated model beat a well-tuned frozen retriever on a stream?* If yes, the architecture-change thesis gets its first clean empirical support. If no, that is equally decisive and saves months.

**Why this over chasing online MoE experts:** the router/gater-forgetting wall (Section 2) is real, named, and uncrossed at scale. A small team should not try to crack it first; they should (1) ship the offline consolidation product and (2) run the sparse-finetuning experiment that both tests the thesis and builds the benchmark the whole field is missing.

---

## Key sources (arXiv IDs)

- **2602.12587** — MoE Transformers still forget under sparse/balanced routing (MH-MoE fix).
- **2605.17571** — StaR-MoE: router drift / online expert-addition forgetting.
- **2512.10969** — MoB "gater forgetting"; **2601.13020** — PASs-MoE; **2406.16437** — gate must stop updating (ICLR 2025).
- **2510.15103** — Meta Sparse Memory Finetuning (89/71→11%).
- **2410.02355** — AlphaEdit (ICLR 2025 Outstanding).
- **2501.00663** — Titans; **2512.23675** — TTT-E2E; **2602.21204** — simplest TTT = linear attention; **2606.08573** — Titans-layer resets per dialogue.
- **2512.24695** — HOPE / Nested Learning: The Illusion of Deep Learning Architectures (NeurIPS 2025, OpenReview `nbMeRvNb7A`; verified 2026-07-02).
- **2507.05257** — MemoryAgentBench; **2606.01435** — deterministic pipeline closing the MH wall; **2606.05661** — CL-Bench; **2510.17281** — MemoryBench.
- **2606.30406** — MOPD formalization; **2606.19348** — DeepSeek-V4 (secondary); **2602.15763** — GLM-5 capacity-MoE architecture.
- **2603.01761** — Position: Modular Memory is the Key to Continual Learning Agents.

*Report compiled 2026-07-02. Model-internal claims about proprietary post-training (GLM/DeepSeek/Qwen) are flagged unverifiable; all quantitative claims cite the source above.*
