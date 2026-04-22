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

## Rule 6: One GPU Job at a Time

RunPod has ONE GPU. Never run two GPU-intensive jobs concurrently:
- Do NOT start an eval while precompute/extraction is running
- Do NOT start precompute while an eval is running
- Wait for the current job to finish before starting the next

Concurrent GPU jobs cause KV cache contention, timeouts, and 10x slowdowns.
The few minutes saved by parallelism are lost to debugging hangs.

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

## Current Frontier (exp151/exp159)

- ctx0: 81.7% (49/60) — exp150_hybrid_temporal
- 5-ctx: 65.00% (195/300) — exp151_gate_widen
- Config: --use_recall --retrieve_top_k 20 --faceted_cache msl_cache_qwen25_cs512.json

## Target Gaps (from exp151 failure analysis, 105/300 failures)

### By gold length
- 53 long gold (>80 chars): preference/inference — SubEM metric failure, needs LLM-as-judge
- 43 short gold (<80 chars): retrieval miss — gold fact not in evidence
- 9 "not enough info": need information sufficiency detection

### By failure type (short gold)
- Temporal arithmetic (~10): "How many days ago", "How long" — v9 when + Allen engine
- Wrong retrieval (~15): retrieved wrong fact — better entity types from v9
- Aggregation (~8): count/sum errors — v9 quantities + type hierarchy
- Temporal ordering (~5): "Who first", "What order" — Allen engine
- Near-miss formatting (~5): prediction close but not exact SubEM match

### Key insight
104/105 failures have gold NOT in evidence. The retrieval is the bottleneck.

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
