# Engram Autoresearch Program

Adapted from karpathy/autoresearch for memory system optimization.

## Rule 1: NO BENCHMARK HARDCODING

This is the cardinal rule. NEVER:
- Add question-specific patterns or gold-answer-informed thresholds
- Hardcode English keywords to gate logic (fails multilingual)
- Add LME-specific heuristics that wouldn't help on ANY other dataset
- Gate on question text when you can gate on prediction/evidence content

Every change must be a GENERAL improvement to the retrieval/memory system.
If it only helps LME s_star, it's benchmark overfitting. Discard it.

## Rule 2: Keep All Experiments

Never git reset. Every experiment gets committed and logged in results.tsv:
- keep: improved SubEM or simplified code
- discard: regressed or neutral with added complexity
- crash: failed to run

This preserves full experimental history for analysis.

## Rule 3: Metric

- Primary: substring_exact_match (SubEM) on 5-ctx (300 questions)
- Secondary: F1
- Quick check: ctx0 (60 questions, ~4 min) for fast iteration
- Full eval: 5-ctx (300 questions, ~20 min) to confirm

## Rule 4: Simplicity

All else equal, simpler code wins. Complexity budget:
- 0.5% SubEM gain from 50 lines of hacks -> discard
- 0% SubEM change from deleting dead code -> keep
- 2%+ SubEM gain from clean 20-line change -> keep

## Rule 5: Work Upstream, Not Downstream

Post-processing is a last-ditch crutch. The leverage hierarchy is:

1. **Extraction / Ingestion** — what gets stored determines everything downstream.
   Better entity extraction, better chunking, better temporal parsing, better
   relation extraction. This is where real improvements come from.
2. **Retrieval** — given what's stored, find the right evidence.
   BM25 tuning, embedding models, reranking, entity-aware search.
3. **Generation** — given the right evidence, produce the right answer.
   Prompt engineering, model selection, reasoning strategies.
4. **Post-processing** — format-match the answer to the gold.
   EXHAUSTED. 20+ functions, 0 additional rescues possible. Do not invest here.

When proposing experiments, always ask: "Am I improving what gets stored,
or am I papering over bad storage with downstream hacks?"

## The Loop

LOOP FOREVER:
1. Read current results.tsv and identify the frontier
2. Propose an experiment targeting: extraction, ingestion, or retrieval
3. Modify code (foundry/*.py or foundry/eval/run_engram.py)
4. git commit with descriptive message
5. Sync to RunPod: rsync foundry/ to /workspace/repo/foundry/
6. Run eval: ssh RunPod "HERMES_ROOT=/workspace/repo python3 -u foundry/eval/run_engram.py ..."
7. Read results: grep SubEM/F1 from output
8. Log to results.tsv with commit hash and status
9. If crash: tail -50 run.log, fix, re-run (max 3 attempts then move on)
10. GOTO 1

NEVER STOP. The human may be sleeping. Run experiments indefinitely.

## Current Frontier (exp151)

- ctx0: 80.00% (48/60)
- 5-ctx: 65.00% (195/300)
- Config: --use_recall --retrieve_top_k 20 --faceted_cache msl_cache_qwen25_cs512.json

## Target Gaps (from failure analysis)

- 27 temporal failures (9 have dates in evidence but system doesn't compute)
- 4 numeric computable (aggregate engine exists but extraction feeds wrong entities)
- 5 list computable (same extraction problem)
- 53 long gold inference (multi-turn synthesis, needs better evidence selection)
- 35 short factual (gold not in evidence, retrieval miss)

## Eval Command (RunPod)

```bash
cd /workspace/repo && HERMES_ROOT=/workspace/repo python3 -u foundry/eval/run_engram.py \
  --dataset_config MemoryAgentBench/configs/data_conf/Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml \
  --use_recall --retrieve_top_k 20 --chunk_size 512 \
  --faceted_cache proofs/hermes-NLCDM/python/output/msl_cache_qwen25_cs512.json \
  --agent_model gpt-4o-mini \
  --max_contexts 1 \
  --tag exp_name > run.log 2>&1
```
