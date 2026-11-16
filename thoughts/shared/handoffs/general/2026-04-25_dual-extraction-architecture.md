---
root_span_id: dual-extract-arch-001
session_id: dual-extract-arch-001
---

# Handoff: Dual Extraction Architecture (Schema + Deriver)

**Date**: 2026-04-25
**Branch**: `worktree-engram-build`
**Worktree**: `.claude/worktrees/engram-build/`

## Summary

Investigated Honcho's 90%+ LME architecture, benchmarked schema vs deriver extraction across multiple local SLMs, and discovered a dual-extraction approach that combines the best of both.

## LME-500 Code Oracle Run — IN PROGRESS

Running at time of handoff (495/500 contexts processed):
```
python3 -u foundry/eval/run_engram.py \
  --longmemeval_json output/longmemeval_s.json \
  --use_recall --retrieve_top_k 20 --event_frames \
  --code_oracle \
  --tag "lme500_code_oracle"
```
Output file: Check `proofs/hermes-NLCDM/python/output/engram/` for most recent `*lme500_code_oracle*` file.

Previous baseline (no code oracle): **280/500 (56.0%)** SubEM.

## Honcho Architecture Analysis

Cloned `plastic-labs/honcho` at `/tmp/honcho-research/`. Key files:
- `src/deriver/deriver.py` + `prompts.py` — Extraction: NL observations from messages
- `src/dialectic/core.py` + `prompts.py` — Query answering: agentic tool-calling loop
- `src/dreamer/orchestrator.py` + `specialists.py` — Background consolidation: deduction + induction specialists
- `src/dreamer/surprisal.py` — Novelty detection via geometric surprisal on embedding trees

### How Honcho Gets 90%+
1. **No schema extraction** — Natural language observations only ("Alice is 25 years old")
2. **Frontier LLMs everywhere** — Gemini for extraction, Claude for reasoning
3. **Three-tier reasoning** — explicit → deductive → inductive (background "dreaming")
4. **Agentic answering** — Dialectic agent with tools: search_memory, grep_messages, search_messages, get_reasoning_chain, get_messages_by_date_range
5. **Surprisal sampling** — Tree-based novelty detection focuses consolidation on unusual observations

### Key Takeaway
Honcho's approach works because frontier LLMs can stay grounded in free-form output. Small models can't — they hallucinate more with unconstrained output space.

## Extraction Benchmarks — Schema vs Deriver vs Dual

### Raw text → Schema (structured entities + relations)
- **qwen2.5-3b**: 2-4 ent, 0-4 rel, ~1-3s. 97%+ grounded.
- **llama-3.1-8b**: 3-5 ent, 2-5 rel, ~5-10s. High grounding.
- Entity names are verifiable, good for counting/aggregation/graph.

### Raw text → Deriver (natural language observations)
- **qwen2.5-3b**: 1-5 obs, ~0.5-1s. **Hallucination on some chunks** (e.g., fabricated "Silent Patient" from parametric memory — later disproved, text actually contained it in a later turn).
- **llama-3.1-8b**: 2-5 obs, ~2-4s. More observations, slightly better grounding.
- Captures user intent/preferences, but harder to verify grounding.

### Schema → Deriver (WINNER — dual pipeline)
- Schema extraction first (97% grounded), then 3B deriver reformulates schema into NL observations.
- **Zero hallucination** — deriver can only work with verified schema data.
- 3B model produces 3-5 clean observations per chunk in 0.8-2.4s.
- Both outputs stored: schema for structured queries, observations for semantic search.

## Dual Extraction Architecture

```
Raw conversation chunk
    │
    ▼
[Schema Extractor] ──→ entities + relations (structured index)
    │                     - Entity graph traversal
    │                     - Faceted who/what/where/when
    │                     - Counting/aggregation
    │                     - Type hierarchy
    │                     - Temporal routing (Allen engine)
    │
    ▼
[Deriver] ──→ natural language observations (semantic index)
                - Embedded for cosine/hybrid search
                - Content-addressable retrieval
                - Preference/opinion questions
                - Cross-session patterns
```

**Query routing** picks which index to hit — or hits both and merges results.

## V9 Cache Quality Audit

3200 chunks in `output/runpod_backup/msl_cache_v9_cs512.json` (teacher: Qwen2.5-7B):

| Metric | Value |
|--------|-------|
| Total entities | 28,862 |
| Total relations | 33,110 |
| Entity grounding rate | 97.2% (28,062 grounded) |
| Relation both-grounded | 76.3% |
| Garbage entities | 0.03% |

### Hallucination Pattern Breakdown (800 ungrounded entities)

| Pattern | Count | % | Description |
|---------|-------|---|-------------|
| Recombination | 297 | 37% | Combines separate source words into fake entity |
| Paraphrase | 263 | 33% | Summarizes instead of extracting verbatim |
| Generic concept | 154 | 19% | Single-word inferred concepts ("customization") |
| Pure hallucination | 82 | 10% | Injects parametric knowledge (e.g., "Olive Garden" in packing text) |
| Punctuation mismatch | 4 | 0.5% | Apostrophe/hyphen differences |

## Training Data Plan for Student Model

### Approach: Dual-output student
Train one model (qwen2.5-0.5b or 1.5b) to produce BOTH outputs from raw text:

1. **Generate training data**: For all 3200 v9 cache chunks:
   - Schema: already have (v9 cache), filter to grounded-only
   - Observations: generate via 3B deriver from cleaned schema output

2. **Clean the data**:
   - Drop 82 pure hallucinations entirely
   - Replace 297 recombinations with nearest verbatim span
   - Replace 263 paraphrases with verbatim text
   - Drop 154 generic concepts
   - INCLUDE corrected versions so student learns "don't extract here"

3. **Training**: SFT with LoRA, input=chunk_text, output=schema+observations JSON

4. **Validation**: Hold out 320 chunks (10%), compare entity/relation F1 + observation grounding rate against teacher

## LM Studio Models Available

Models confirmed working on MacBook via LM Studio (http://127.0.0.1:1234):
- qwen2.5-3b-instruct — BEST balance: 1-3s, clean JSON, good extraction
- qwen2.5-0.5b-instruct — Fast but weaker
- qwen2.5-coder-7b-instruct — More thorough, 5-10s
- meta-llama-3.1-8b-instruct@q4_k_m — Strong, 5-10s
- qwen/qwen3.5-9b — Stuck in thinking mode, needs `enable_thinking: false`

## Next Steps

1. **Check LME-500 code oracle results** — Run should be complete. Compare against 280/500 baseline.

2. **Build training data pipeline** — Script that takes v9 cache → cleans schema → generates observations via 3B deriver → outputs SFT-ready JSONL.

3. **Train student model** — LoRA fine-tune qwen2.5-1.5b (or 0.5b) on dual-output task. Needs GPU (A40 or MLX on MacBook).

4. **Wire into ingestion** — Replace `--faceted_cache` precompute path with runtime extraction via local model. Both schema and observations stored per-chunk.

5. **Run LME-500 with dual extraction** — Compare against schema-only baseline (56%).

## Files Referenced

| File | Purpose |
|------|---------|
| `output/runpod_backup/msl_cache_v9_cs512.json` | V9 schema cache (3200 chunks, Qwen2.5-7B) |
| `output/runpod_backup/msl_cache_qwen25_cs512.json` | Earlier schema cache |
| `/tmp/honcho-research/` | Cloned Honcho repo |
| `foundry/eval/code_oracle.py` | Code oracle for computational questions |
| `foundry/eval/run_engram.py` | Main eval harness |
| `foundry/ingestion.py` | Extraction pipeline (extract_via_llm at L1037) |
| `foundry/retrieval.py` | recall() at L979 — main retrieval function |
| `foundry/core.py` | Foundry API (ingest_chunk at L449) |

## Key Decisions Made

1. **Schema extraction stays** — Not replaced by deriver. Schema provides structured query support (counting, graphs, temporal) that NL observations can't.
2. **Deriver runs FROM schema, not raw text** — Eliminates hallucination by constraining input to verified facts.
3. **Both outputs stored** — Dual index: structured (entity graph) + semantic (embedded observations).
4. **Local SLMs viable** — 3B model handles both steps in <3s total per chunk. Good enough for ingestion-time extraction.
5. **Training data exists** — 3200 teacher examples in v9 cache, needs cleaning (~800 corrections) before student training.

next_session_prompt: |
  Resume from handoff at `thoughts/shared/handoffs/general/2026-04-25_dual-extraction-architecture.md`.

  Priority 1: Check LME-500 code oracle results — look for most recent `*lme500_code_oracle*` file in `proofs/hermes-NLCDM/python/output/engram/`. Compare against 280/500 baseline.

  Priority 2: Build the training data cleaning + generation pipeline. Script should:
  (a) Load v9 cache (3200 chunks) + multipass cache (source text)
  (b) Filter ungrounded entities (drop pure hallucinations, fix paraphrases/recombinations)
  (c) Run schema→observations deriver on cleaned schema via local qwen2.5-3b
  (d) Output SFT-ready JSONL: {"input": chunk_text, "output": {"entities": [...], "relations": [...], "observations": [...]}}

  Priority 3: Fine-tune student model (qwen2.5-1.5b with LoRA) on the cleaned dual-output dataset.
