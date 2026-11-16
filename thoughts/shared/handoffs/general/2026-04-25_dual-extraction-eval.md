---
root_span_id: dual-extract-eval-001
session_id: dual-extract-eval-001
---

# Handoff: Dual Extraction Eval — Schema → Deriver → Vector Index

**Date**: 2026-04-25
**Branch**: `worktree-engram-build`
**Worktree**: `.claude/worktrees/engram-build/`

## Summary

Built and tested the dual extraction architecture on ctx0 of LME. Schema extraction feeds a deriver (template or SLM) that produces NL observations, which are embedded in the vector index alongside turns. Observations are excluded from BM25 to avoid diluting counting/temporal queries.

## LME-500 Code Oracle — DEAD

- **278/500 (55.6%)** vs baseline 280/500 (56.0%). Net -2.
- Zero improvements, 2 regressions from oracle appending text to correct answers.
- Code oracle confirmed dead for LME at any scale.

## Dual Extraction ctx0 Results

| Experiment | SubEM | Notes |
|-----------|-------|-------|
| Baseline (no obs) | 48/60 (80.0%) | Best ctx0-only result |
| Dual + BM25 (template) | 45/60 (75.0%) | Observations in BM25 hurt counting (-3) |
| Dual + vec-only (template) | 46/60 (76.7%) | Better: only 1 regression |
| Dual + vec + consolidate | 46/60 (76.7%) | Clustering barely helps with templates |
| Dual + SLM deriver (vec-only) | 45/60 (75.0%) | 5913 obs, 1078s deriver. WORSE than template |

### Key Findings

1. **Observations in BM25 = BAD**: Template observations dilute IDF weights and displace real evidence for counting questions (qi=39 "how many days" regressed from 17→8).
2. **Observations in vector-only = NEUTRAL to slightly negative**: Cosine retrieval pulls observations alongside turns. They appear in evidence and most are harmless, but for counting they can confuse the LLM.
3. **Template deriver is too low quality**: "The user X Y." format is too generic. SLM deriver should produce better NL.
4. **LM Studio context overflow**: Running two pipelines concurrently against the same model causes 400 errors. Must run one at a time.

### Architecture Decision: Three-Tier Memory

| Tier | What | Stored In | Query Path |
|------|------|-----------|------------|
| Tier 1: Raw turns | Verbatim conversation | BM25 + vector | Factual, counting, temporal |
| Tier 2: Observations | NL facts from schema | Vector only | Semantic similarity, preferences |
| Tier 3: Consolidated | Clustered summaries | Vector only | Cross-session patterns |

## Files Created/Modified

| File | Purpose |
|------|---------|
| `foundry/eval/build_training_data.py` | **NEW**: Training data pipeline — clean v9 cache, generate observations, output JSONL |
| `foundry/eval/consolidate_observations.py` | **NEW**: Observation clustering via cosine similarity + representative selection |
| `foundry/eval/run_engram.py` | **MODIFIED**: Added `--dual_extraction`, `--deriver_url`, `--deriver_model`, `--local_deriver`, `--consolidate`, `--consolidate_threshold` flags |
| `foundry/retrieval.py` | **MODIFIED**: `_full_text()` in `_recall_retrieve()` now falls back to observation texts |

### New CLI Flags

```bash
# Dual extraction with SLM deriver
python3 -u foundry/eval/run_engram.py \
  --dataset_config .../Longmemeval_s_star.yaml \
  --use_recall --retrieve_top_k 20 --event_frames \
  --faceted_cache output/runpod_backup/msl_cache_v9_cs512.json \
  --dual_extraction \
  --deriver_url http://127.0.0.1:1234/v1/chat/completions \
  --deriver_model qwen2.5-3b-instruct \
  --max_contexts 1 \
  --tag "dual_slm_ctx0"

# With consolidation
python3 -u foundry/eval/run_engram.py ... --dual_extraction --local_deriver --consolidate

# Stats-only training data (no deriver)
python3 foundry/eval/build_training_data.py \
  --v9_cache output/runpod_backup/msl_cache_v9_cs512.json \
  --multipass_cache proofs/hermes-NLCDM/python/output/multipass_cache_cs512.json \
  --output output/sft_dual_extraction.jsonl \
  --stats_only
```

### Training Data Pipeline Results (stats-only)

- 3200 chunks processed in 8s
- **97.9% grounded** (28,255/28,862 entities)
- 401 paraphrases fixed to verbatim spans
- 6 recombinations fixed
- 200 pure hallucinations dropped
- 32,964 template observations generated
- Train: 2880 examples, Val: 320 examples
- Output: `output/sft_dual_extraction.{train,val}.jsonl`

## Background Task Status

- **blgyu67qo**: SLM deriver eval on ctx0 — RUNNING (started ~16:38)
  - Expected duration: ~15 min (2 min ingest + 12 min deriver for 739 chunks)
  - Output: `proofs/hermes-NLCDM/python/output/engram/20260425_163813_engram_cs512_dual_slm_ctx0_longmemeval_s*_gpt-4o-mini.json`

## What's NOT Working Yet

1. **SLM deriver quality vs template**: SLM deriver (qwen2.5-3b) produces better NL but had 400 errors under concurrent load. Fixed with truncation (max 15 entities + 15 relations per schema) and multi-array JSON parsing.
2. **Observation-based evidence for counting Qs**: Observations appearing in evidence confuse the LLM for counting queries. Need either (a) filter observations from evidence for count/sum predicates, or (b) don't include observations in cosine index at all for now.
3. **Consolidation**: Clustering at 0.85 threshold only reduces 19% for template observations. May work better with SLM observations that have more semantic diversity.

## Next Steps

1. **Check SLM deriver eval results** — Task blgyu67qo. Compare against 48/60 baseline.

2. **If SLM helps**: Run 5-ctx eval with `--dual_extraction` and compare against 181/300 baseline.

3. **If observations hurt counting**: Add predicate-aware observation filtering — exclude observations from evidence when `predicate in {count, sum, list, when}`.

4. **Rerun training data pipeline** with fixed deriver (truncation + multi-array parsing) — single process:
   ```
   python3 -u foundry/eval/build_training_data.py \
     --v9_cache output/runpod_backup/msl_cache_v9_cs512.json \
     --multipass_cache proofs/hermes-NLCDM/python/output/multipass_cache_cs512.json \
     --output output/sft_dual_extraction_slm.jsonl \
     --deriver_url http://127.0.0.1:1234/v1/chat/completions \
     --deriver_model qwen2.5-3b-instruct
   ```

5. **Consolidation layer**: After SLM observations exist, cluster and summarize. This is the "tier 3" in the architecture — requires SLM call per cluster to generate summaries.

6. **Train student model**: Once training data is clean (SLM observations + cleaned schema), LoRA fine-tune qwen2.5-1.5b on dual output.

next_session_prompt: |
  Resume from handoff at `thoughts/shared/handoffs/general/2026-04-25_dual-extraction-eval.md`.

  Priority 1: Check SLM deriver eval results — look for `*dual_slm_ctx0*` in `proofs/hermes-NLCDM/python/output/engram/`. If not found, rerun:
  ```
  python3 -u foundry/eval/run_engram.py --dataset_config /Users/cosimo/.hermes/hermes-agent/MemoryAgentBench/configs/data_conf/Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml --use_recall --retrieve_top_k 20 --event_frames --faceted_cache output/runpod_backup/msl_cache_v9_cs512.json --dual_extraction --deriver_url http://127.0.0.1:1234/v1/chat/completions --deriver_model qwen2.5-3b-instruct --max_contexts 1 --tag "dual_slm_ctx0"
  ```
  Compare against baseline 48/60 (80.0%). If neutral or positive, run 5-ctx.

  Priority 2: If observations hurt counting, add observation filtering in `_recall_retrieve()` — skip doc_ids >= 100_000 when predicate is count/sum/list/when.

  Priority 3: Rerun training data pipeline with fixed deriver (single process, no concurrency).

  Priority 4: Implement the consolidation/summarization layer (tier 3). User wants: embed observations → cluster similar → summarize clusters → store as higher-level facts.
