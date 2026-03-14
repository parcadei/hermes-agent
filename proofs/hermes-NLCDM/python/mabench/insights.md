# Hermes Memory Agent — Insights

## Multi-Hop is a Property of Knowledge State, Not the Question

**Date:** 2026-03-14
**Context:** Error classification on factconsolidation_mh_262k (100 multi-hop questions)

### The Discovery

On mh_262k, 0/100 correct answer strings appear in retrieved context — even for
the 15 questions answered correctly. Every correct answer is an LLM *inference*
from partial evidence, not fact-reading. The system does single-pass retrieval,
but multi-hop questions need iterative retrieval where each hop's result informs
the next hop's query.

### The Insight

Whether a question requires multi-hop retrieval is **not** a static property of
the question — it's a property of the agent's current knowledge state:

- If the system already has a stored derived fact ("Christian Abbiati played
  association football, which originated in England"), the question "What country
  is Abbiati's sport from?" is **single-hop**. One retrieval, done.
- If only the base facts exist separately ("Abbiati played football" and
  "Football originated in England"), the same question requires **multi-hop**:
  retrieve hop 1, extract intermediate entity, retrieve hop 2.

This is isomorphic to how humans work. When you don't know the answer, you
iteratively search — each hop narrows the gap until you reach the answer. The
termination criterion is organic: you stop when you know enough to answer.

### Production Implication: Query-Time Consolidation

The iterative retrieval loop should **store the derived answer as a new fact**
after successfully resolving a multi-hop chain. First time the question is asked:
3 hops to find "England." Then store "The country of origin of the sport played
by Christian Abbiati is England" as a derived memory. Next time: single hop.

This is what dream consolidation was *trying* to do at ingestion time (pre-compute
bridges), but it failed because you can't predict which chains will be queried.
Query-time consolidation derives exactly the bridges that are actually needed.

### Why LLM-in-the-Loop is the Only General Termination Criterion

- **Cosine similarity** can tell you "this chunk is topically related" but not
  "this chunk completes the reasoning chain."
- **Triadic memory** can check graph connectivity but only if ingestion captured
  the right triples.
- **LLM judgment** ("Can I answer with these facts? If not, what entity do I need
  next?") is the only approach that generalizes — it makes the same semantic
  judgment a human would make.

Cost: ~2-3 extra gpt-4o-mini calls per multi-hop question ($0.003 total).
Bounded by max_hops safety valve.

### Decompose Query vs Iterative Query

| Aspect | `decompose_query` (existing) | `iterative_query` (new) |
|--------|------------------------------|------------------------|
| **When decomposition happens** | Before any retrieval — templates are fixed upfront | After each retrieval — next query is dynamic |
| **Sub-query generation** | LLM generates all sub-queries at once with `{answer_N}` placeholders | LLM generates ONE next query based on what was actually found |
| **Intermediate answers** | Extracted and substituted into templates, but templates may be wrong | Each hop dynamically decides what to look up based on actual retrieved evidence |
| **Failure mode** | Bad template → wrong sub-query → wrong intermediate answer → cascade failure | Each hop is independently targeted — a bad hop doesn't corrupt the template chain |
| **Retrieval pattern** | N parallel retrievals (one per sub-query) + original question | Sequential: retrieve → judge → retrieve → judge → ... |
| **Termination** | Fixed: number of sub-queries from decomposition | Dynamic: LLM decides when it has enough |
| **Cost** | 1 LLM call (decompose) + N extract calls | N judge calls (each decides continue/stop) |

The fundamental difference: decompose_query plans the route before seeing the
territory. Iterative query explores the territory and decides the route as it goes.

### Benchmark Context

- ALL MABench agents (GraphRAG, HippoRAG, Self-RAG, MemoRAG, Zep, Mem0, Cognee,
  Letta) use single-pass retrieval through the `send_message` interface.
- No agent does tool-call simulation or iterative retrieval.
- Our SubEM=22 (cooc+triadic+decompose+temporal) is genuinely competitive against
  the field. Iterative retrieval would be a novel capability in this benchmark.

### Results Table (as of 2026-03-14)

| Config | SubEM | Notes |
|--------|-------|-------|
| cosine baseline | 4 | |
| cooc+triadic+decompose (winner) | 20 | |
| + temporal ordering | 22 | best so far |
| + hard contradiction @0.85 | 19 | false-positive detection |
| + soft contradiction @0.85 | 19 | same root cause |
| ERROR CLASSIFICATION (85 wrong) | 85/85 retrieval-bound | 0 presentation-bound |
