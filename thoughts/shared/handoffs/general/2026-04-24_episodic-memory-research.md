---
root_span_id: 8f8ef9d0-d828-4c00-a0d4-aa08028fd8ac
turn_span_id: 
session_id: 8f8ef9d0-d828-4c00-a0d4-aa08028fd8ac
---

# Handoff: Episodic Memory Research + Code Oracle

**Date**: 2026-04-24
**Branch**: `worktree-engram-build`
**Worktree**: `.claude/worktrees/engram-build/`

## What Was Done

### 1. Episodic Memory Literature + Architecture Analysis
Researched episodic memory from neuroscience (Tulving, CLS theory) through current ML systems. Identified the core gap in our Engram system: **binding**. We store entities and relations as loose rows — no event frames, no provenance pointers, no temporal contiguity.

Cloned and analyzed three systems:

**EM-LLM** (ICLR 2025, `research/EM-LLM-model/`):
- Surprise-based event segmentation using LLM perplexity
- Graph-theoretic boundary refinement (modularity/conductance)
- Temporal contiguity buffer (30% context budget for block±1 neighbors)
- LongBench avg 51.58 vs RAG 36.44. Pure KV-cache, no entity extraction.

**REMem** (ICLR 2026, `research/REMem/`):
- Hybrid memory graph: verbatim → gists → facts → entities (4 node types, 6 edge types)
- Gists = time-stamped event summaries with temporal qualifiers on facts
- Agentic retriever with 5 tools (semantic, lexical, entity, gist, answer)
- LME overall EM 55.2%. **We beat them** (66% SubEM) but their temporal architecture is cleaner.

**SEEM** (arXiv:2601.06411, no public code):
- Two-layer hierarchy: graph memory + episodic event frames
- Reverse Provenance Expansion: facts → trace to source events → expand neighbors
- Provenance pointers = the key missing piece in our system

**Triadic Memory** (`research/TriadicMemory/`):
- 3D binary tensor for SDR triple association. Dead end — content-addressable like Hopfield, no temporal binding.

### 2. Oracle-as-Code-Generator — VALIDATED

Tested bicameral oracle concept: instead of LLM arithmetic, SLM generates Python from structured facts → exec() → deterministic answer.

**Test script**: `research/REMem/test_oracle_bicameral.py`

**Results on 5 question types** (date diff, age diff, months, counting, sum):

| Model | Direct Answer | Code Generation |
|-------|--------------|-----------------|
| llama-3.2-3b-instruct | 2-3/5 | **5/5** |
| liquid/lfm2.5-1.2b | 2-3/5 | **4/5** |

Key examples:
- QI7 (March 15 - March 8 = 7 days): Both models wrong "in their head" (11 or 2). Both correct via code (7).
- QI23 (75 - 32 = 43): Code gets it right. Raw answering varies by model.

Also tested oracle scene reconstruction (narrative from event frames). Mixed: helps some questions, hurts others. **Code-gen strictly dominates for arithmetic/temporal/counting.**

### 3. Content-Addressable vs Index-Based Memory

Identified that ALL our retrieval is content-addressable (match query against content). What's missing is an **index layer** — a separate binding structure that connects fragments into episodes and enables retrieval by association (temporal contiguity, provenance tracing).

## Key Findings

1. **The fundamental problem is binding, not retrieval quality.** Our v9 extracts entities, relations, timestamps — but they float free. No event frame groups them. No provenance traces facts back to source turns.

2. **Temporal contiguity is the highest-impact addition.** EM-LLM reserves 30% of context for temporal neighbors. SEEM uses RPE. REMem uses gist graph walks. We do none of this.

3. **Oracle-as-code-generator replaces English regex post-processing.** SLM generates Python from facts → exec(). Language-independent, deterministic, 5/5 accuracy with llama-3.2-3b. Replaces ~500 lines of hardcoded English regex.

4. **We already beat REMem on SubEM** (66% vs 55.2%), so the architecture improvements should push us higher, not catch up.

## Proposed Architecture

### Event Frames (storage-time binding)
```python
EventFrame(
    id="ef_047",
    turns=[142, 143, 144],          # provenance → raw turn indices
    timestamp="2024-03-15",
    gist="User discussed Miami restaurants after trip",
    entities=["la_mar", "zuma", "miami"],
    facts=[(user, tried, la_mar), ...],
    prev="ef_046",                   # temporal contiguity
    next="ef_048",
)
```
Cheap to implement — just group existing v9 extractions.

### Temporal Contiguity (retrieval-time)
When retrieval hits frame N, also pull frames N-1, N+1. Trivial linked list traversal.

### Code Oracle (retrieval-time)
```
Question → classify → arithmetic?
    yes → SLM generates Python from facts → exec() → answer
    no  → normal retrieval → LLM answers
```

## Files Created/Modified

| File | What |
|------|------|
| `research/EM-LLM-model/` | Cloned repo (em-llm/EM-LLM-model) |
| `research/REMem/` | Cloned repo (intuit-ai-research/REMem) |
| `research/TriadicMemory/` | Cloned repo (PeterOvermann/TriadicMemory) |
| `research/REMem/test_oracle_bicameral.py` | Oracle test script |
| `memory/episodic-memory-research.md` | Detailed research notes |
| `memory/MEMORY.md` | Updated with findings and new priorities |
| `memory/cl-experiments.md` | Updated with E24 (NLCDM centroids) |

## Next Steps (Priority Order)

1. **Implement EventFrame schema** in `foundry/store.py` — group v9 entities+relations per turn cluster, add prev/next pointers
2. **Wire temporal contiguity** into `recall()` — when BM25 finds a turn, also return adjacent event frame turns
3. **Build code oracle** — classify question type, if arithmetic → SLM code-gen → exec(). Use llama-3.2-3b-instruct at localhost:1234
4. **Gist layer** for preference questions (26 failures) — event-level summaries with timestamps
5. **Eval on full 5-ctx LME** to measure impact

## Local LLM Models Available
- `liquid/lfm2.5-1.2b` at http://127.0.0.1:1234 (LM Studio)
- `llama-3.2-3b-instruct` at http://127.0.0.1:1234
- `qwen3.5-4b` at http://127.0.0.1:1234 (outputs JSON, not narrative — needs different prompting)
- `google/gemma-4-e4b` at http://127.0.0.1:1234 (outputs thinking tokens, needs parsing)
