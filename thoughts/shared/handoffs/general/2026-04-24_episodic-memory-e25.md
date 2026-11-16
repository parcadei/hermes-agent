---
root_span_id: 8f8ef9d0-d828-4c00-a0d4-aa08028fd8ac
session_id: 8f8ef9d0-d828-4c00-a0d4-aa08028fd8ac
---

# Handoff: E25 Experiments — V9 Cache + Code Oracle + Contiguity

**Date**: 2026-04-24
**Branch**: `worktree-engram-build`
**Worktree**: `.claude/worktrees/engram-build/`

## What Was Done

### 1. E25 Baseline Established
- ctx0 baseline: **48/60 (80.00%)** SubEM, F1=47.18
- Matches D-06b recall() baseline from oracle experiments
- Q33 rescued (+1 vs April 21) — temporal parsing improvements paid off

### 2. V9 Cache Coverage Analysis — KEY FINDING
Discovered ctx1-4 were running with **zero entity extraction**:

| Context | qwen25 cache | v9 cache | v9 non-empty |
|---------|-------------|----------|-------------|
| ctx0 | 148/739 (20%) | 739/739 (100%) | 625 (84%) |
| ctx1 | 0/737 (0%) | 737/737 (100%) | 629 (85%) |
| ctx2 | 14/793 (1%) | 793/793 (100%) | 685 (86%) |
| ctx3 | 0/732 (0%) | 732/732 (100%) | 625 (85%) |
| ctx4 | 0/752 (0%) | 248/752 (32%) | 225 (29%) |

### 3. V9 Cache 5-Context Experiment — NEUTRAL
**Hypothesis**: ctx1-4 underperform because they lack entity extraction. V9 cache (85% coverage) should close the gap.

**Result**: 65.33% (196/300) — **slightly WORSE than** qwen25 baseline 66.33% (199/300).

Per-context: ctx0=48, ctx1=38, ctx2=38, ctx3=33, ctx4=39 (vs D-18: 48,39,38,34,40)

**Conclusion**: Entity extraction quality is NOT the bottleneck for ctx1-4. The gap is in the conversations themselves (harder questions, less distinctive evidence). Regex extraction is sufficient. The v9 cache adds noise via lower-quality entities on cache-miss chunks.

### 4. Code Oracle Bug Fix
`_safe_exec()` had empty exec namespace — `datetime`/`math` not pre-imported. SLM generates `datetime.date(...)` without explicit imports. Every production call was silently returning `[]`.

**Fix**: Pre-import allowed modules into exec globals (line 148 of `code_oracle.py`).

**Verified working**: 75-32=43 ✓, March 15-March 8=7 days ✓.

### 5. conversation_parser.py Restored
Source file was deleted, only .pyc remained. Recovered from git commit `dcdad2c2`. Required for turn-level BM25 rebuild.

### 6. Failure Analysis (300 questions)

| Category | Count | % of Failures | Addressable by |
|----------|-------|---------------|---------------|
| Preference | 26 | 22% | Metric change (LLM-as-judge) — SubEM too strict |
| NEI | 9 | 8% | NEI detection (D-17 approach) |
| Computational | 37 | 31% | Code oracle (now fixed) |
| Factual | 48 | 40% | Better retrieval / evidence quality |

Preference failures are a **metric problem**: model produces semantically correct profiles but uses different wording than gold. Confirmed by examining evidence — v9 facts contain exact preference details (cherry tomatoes, basil, lemon poppyseed cake) but SubEM requires exact substring match.

### 7. REMem Deep-Dive (from prior session, documented)
- Gists = SEPARATE retrieval modality, not BM25 text augmentation
- 55.2% is GPT-4o LLM-as-judge (NOT SubEM) — our 66% SubEM is genuinely ahead
- Three-pass extraction: gists → facts (informed by gists) → temporal
- Agentic retriever with 5 tools, ReAct loop, max 3 steps

## Key Findings

1. **V9 cache does NOT help 5-ctx.** Extraction quality is not the bottleneck. This disproves the theory that ctx1-4 gaps are due to missing entity extraction.

2. **Code oracle was silently broken.** Now fixed. 37 computational failures are the target. Experiment pending.

3. **Contiguity is implemented but untested.** `contiguity_window` parameter wired into `recall()` → `_recall_retrieve()`. Neighbors get score × 0.1, reranker filters naturally.

4. **Preference questions (26/120 failures) are a metric problem.** Need LLM-as-judge or semantic similarity metric. Not addressable via retrieval or generation improvements.

## Files Modified

| File | Change |
|------|--------|
| `foundry/eval/code_oracle.py` | Fixed `_safe_exec` — pre-import datetime/math into exec namespace |
| `foundry/retrieval.py` | Added `contiguity_window` parameter (prior session, untested) |
| `foundry/eval/run_engram.py` | Added `--contiguity_window`, `--code_oracle` CLI args (prior session) |
| `proofs/.../mabench/conversation_parser.py` | Restored from git (was deleted, only .pyc remained) |

## Experiment Results

| Tag | Cache | CW | Oracle | Ctx | SubEM | Notes |
|-----|-------|-----|--------|-----|-------|-------|
| E25_baseline_cw0 | qwen25 | 0 | no | 1 | 80.00% (48/60) | Matches D-06b |
| E25_v9_ctx0 | v9 | 0 | no | 1 | 80.00% (48/60) | Safe, identical |
| E25_v9_5ctx | v9 | 0 | no | 5 | 65.33% (196/300) | NEUTRAL vs 66.33% |

## Next Steps (Priority Order)

1. **Code oracle on 5-ctx** — now that exec bug is fixed, run with `--code_oracle --faceted_cache v9`. 37 computational failures are the target. Could yield +5-10.
   ```
   python3 -u foundry/eval/run_engram.py --dataset_config .../Longmemeval_s_star.yaml \
     --use_recall --retrieve_top_k 20 --faceted_cache output/runpod_backup/msl_cache_v9_cs512.json \
     --code_oracle --tag "E25_v9_co_5ctx"
   ```

2. **Contiguity on ctx0** — test `--contiguity_window 1`. Quick ctx0 run to check safety.
   ```
   python3 -u foundry/eval/run_engram.py ... --contiguity_window 1 --max_contexts 1 --tag "E25_cw1_ctx0"
   ```

3. **Combined best config on 5-ctx** — v9 + code oracle + contiguity + NEI detection.

4. **Event Frame schema** — design doc exists, code does not. Group v9 entities+relations per turn cluster, add prev/next pointers. Only pursue if retrieval experiments plateau.

5. **LLM-as-judge metric** — needed to measure preference question quality. REMem uses GPT-4o with task-specific rubrics. 26 questions permanently stuck at SubEM=0 otherwise.

## Infrastructure Notes

- LM Studio running at localhost:1234 with qwen2.5-3b-instruct (code oracle model)
- Faceted caches: `output/runpod_backup/msl_cache_qwen25_cs512.json` (148 entries), `msl_cache_v9_cs512.json` (3200 entries)
- Dataset config: `MemoryAgentBench/configs/data_conf/Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml`
- Output dir: `proofs/hermes-NLCDM/python/output/engram/`
