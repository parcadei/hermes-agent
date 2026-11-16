---
root_span_id: 8f8ef9d0-d828-4c00-a0d4-aa08028fd8ac
session_id: 8f8ef9d0-d828-4c00-a0d4-aa08028fd8ac
---

# Handoff: LME-500 Results + Extraction Speed Benchmark

**Date**: 2026-04-25
**Branch**: `worktree-engram-build`
**Worktree**: `.claude/worktrees/engram-build/`

## LME-500 Results: 280/500 (56.0%) SubEM

First run on the full 500-question LongMemEval benchmark (per-question haystacks, ~475K text each).

### Per-Question-Type Breakdown

| Question Type | Score | Notes |
|---|---|---|
| single-session-user | 59/70 (84.3%) | Best category |
| knowledge-update | 63/78 (80.8%) | FAC consolidation working |
| single-session-assistant | 36/56 (64.3%) | Solid |
| multi-session | 75/133 (56.4%) | Multi-hop retrieval needed |
| temporal-reasoning | 47/133 (35.3%) | Code oracle wasn't enabled |
| single-session-preference | 0/30 (0.0%) | SubEM wrong metric |
| **TOTAL** | **280/500 (56.0%)** | **Without hard-coded computation** |

### Failure Analysis

| Category | Failures | Retrieval miss | Has evidence but wrong |
|---|---|---|---|
| Temporal | 86 | 68 (79%) | 18 (21%) |
| Multi-session | 58 | 36 (62%) | 22 (38%) |

**Key finding**: 79% of temporal failures and 62% of multi-session failures are retrieval misses ("I don't have enough information"), NOT computation failures. BM25 top-20 on ~500-turn haystacks isn't finding target turns.

### Config Used
```
python3 -u foundry/eval/run_engram.py \
  --longmemeval_json output/longmemeval_s.json \
  --use_recall --retrieve_top_k 20 --event_frames \
  --tag "lme500_v2"
```
- BM25 + cosine (all-MiniLM-L6-v2) + reranker (mxbai-rerank-base-v1): YES
- Event frames: YES
- Code oracle: **NO** (not passed)
- Faceted cache: **NO** (regex fallback for entity extraction)
- Output: `proofs/hermes-NLCDM/python/output/engram/20260425_011953_engram_cs512_lme500_v2_longmemeval_full_gpt-4o-mini.json`

### Bug Fixed
- `_load_longmemeval_json()`: Some LME-500 answers are integers (counting questions). Added `answer = [str(a) for a in answer]` to prevent `AttributeError: 'int' object has no attribute 'lower'` in `normalize_answer()`.

## Extraction Speed Benchmark — Production Constraint

### RunPod A40 (GPU) — TOO SLOW
- vLLM 0.6.6 + Qwen2.5-7B-Instruct
- **0.02 chunks/s** (40s per chunk, 27% timeout failures)
- Full v9 prompt (8065 chars) + real chunks (2500 chars) = output overflows
- LME-500 estimate: 103K chunks → 145h with 8 workers
- vLLM needed `--max-model-len 8192` (was 4096, caused 400 errors)
- SSH tunnel: `ssh -f -N -L 18000:localhost:8000 -p 22033 root@69.30.85.138`

### MacBook + LM Studio — VIABLE WITH LIGHTWEIGHT PROMPT

| Model | Latency | Entities | JSON Valid |
|---|---|---|---|
| qwen2.5-3b-instruct | 1.7s | 4 ent, 3 rel | YES |
| qwen2.5-0.5b-instruct | 1.7s | 4 ent, 2 rel | YES |
| liquid/lfm2.5-1.2b | 1.2s | 5 ent, 6 rel | YES (wrong schema) |

**Root cause of speed difference**: The v9 system prompt is 8065 chars (complex 6-step extraction). A lightweight prompt (299 chars) produces clean JSON in <2s.

#### Lightweight Prompt That Works
```
Extract entities and relations from this conversation. Return valid JSON:
{"entities": [{"name": "...", "entity_type": "..."}], "relations": [{"subject": "...", "predicate": "...", "object": "...", "when": "", "quantity": "0"}]}
Only include named entities with proper names. Keep relations concise.
```

### Production Implication
Entity extraction at ingestion time (1.7s per turn) is viable on consumer hardware. Bulk precompute of 100K+ chunks is not. The system must extract per-turn as conversations happen, not require a GPU precompute phase.

## Pod Status

**Safe to close.** All critical data already local:
- `output/runpod_backup/msl_cache_qwen25_cs512.json` (237K)
- `output/runpod_backup/msl_cache_v9_cs512.json` (9.0M)
- `proofs/.../output/multipass_cache_cs512.json` (12M)
- `output/longmemeval_s.json` (265M)

Pod-only data (not needed): Llama 3.1 cache (known-bad, 80% hallucination), sleeping-llm research results (historical), old ablation caches.

## Next Steps

1. **Build lightweight extraction into ingestion pipeline** — Replace the precomputed faceted cache path with runtime extraction via LM Studio (or any OpenAI-compatible local endpoint). The `ingest_chunk()` call should optionally hit a local LLM for entity extraction instead of requiring `--faceted_cache`.

2. **Run LME-500 with code oracle** — The temporal-reasoning category (47/133) was tested WITHOUT `--code_oracle`. The 18 "has evidence but wrong" temporal failures are the immediate target. Add `--code_oracle` to the LME-500 run.

3. **Improve retrieval coverage** — 79% of temporal and 62% of multi-session failures are retrieval misses. Options:
   - Increase `retrieve_top_k` beyond 20
   - Two-stage retrieval: broad BM25 → rerank → focused extraction
   - Entity-guided retrieval: use entity graph to find target turns

4. **Run LME-500 with faceted extraction** — Once lightweight extraction is wired in, re-run LME-500 with entity extraction active. This enables predicate routing (aggregation, temporal) which is currently falling back to regex.

## Files Modified This Session

| File | Change |
|------|--------|
| `foundry/eval/run_engram.py` | Fix: `answer = [str(a) for a in answer]` for int answers |

## Key Numbers

| Metric | Value |
|--------|-------|
| LME-500 SubEM | 280/500 (56.0%) |
| LME-S* SubEM (honest, no hard-coding) | 181/300 (60.33%) |
| LME-S* SubEM (with hard-coding) | 196/300 (65.33%) |
| GPU extraction speed | 0.02 chunks/s (A40) |
| Local extraction speed | 0.6 chunks/s (MacBook, lightweight prompt) |
| v9 prompt length | 8065 chars |
| Lightweight prompt length | 299 chars |
