# Saturday Report — 2026-04-18

## Composed Agent Baselines (FC)

All runs use `gpt-4o-mini`, 262k context, chunk_size=512.

### FC-SH (Single-Hop, 100 questions)

| Date | Config | SubEM |
|------|--------|-------|
| 2026-03-28 | cooc+cr+bm25h+tri | **91%** |
| 2026-04-18 | cooc+decomp+cr+bm25h+tri+xenc | 90% |

### FC-MH (Multi-Hop, 100 questions)

**Ground truth bug:** 34/100 questions have wrong expected answers. The original GT used
real-world answers instead of following the newest supersession chain (which is what the
benchmark explicitly requires — "find the newest fact with larger serial number").

| Date | Config | Original GT | Corrected GT | Notes |
|------|--------|-------------|--------------|-------|
| 2026-03-28 | cooc+decomp+cr+bm25h+tri | 76% | **55%** | 21 hits matched wrong GT |
| 2026-04-18 | cooc+decomp+cr+bm25h+tri+xenc | 39% | **47%** | Cross-encoder helps |
| 2026-04-18 | cooc+decomp+cr+bm25h+tri (repro) | — | **40%** | No cross-encoder |

**Corrected GT source:** `output/factconsolidation_mh_corrected_answers.json` (34 entries, 29 actual answer changes)

### LME (LongMemEval, 300 questions)

| Date | Config | SubEM |
|------|--------|-------|
| 2026-04-12 | entpf+dynprompt+rawchunk+aggof25+struct_h3 | 31.33% |

---

## Engram Baselines (all benchmarks)

All Engram runs use `foundry/eval/run_engram.py`, direct BM25 retrieval, `gpt-4o-mini` for answer generation.

### Conflict Resolution

| Benchmark | SubEM | Config | Notes |
|-----------|-------|--------|-------|
| FC-SH | **51%** | BM25 top_k=20, cs512, full query | Composed: 91%. Gap = entity features |
| FC-MH | **1%** | BM25 top_k=20, cs512, corrected GT | Composed: 55%. Needs chain retrieval |

### Accurate Retrieval

| Benchmark | SubEM | Config | Notes |
|-----------|-------|--------|-------|
| Ruler QA1 (197K) | **76%** | BM25 top_k=20, cs512 | BM25 works well for simple retrieval |
| Ruler QA2 (421K) | **39%** | BM25 top_k=20, cs512 | 2x haystack, BM25 diluted |
| EventQA (full) | **44.8%** | BM25 top_k=5, cs4096, full query | 500 Qs, 5 contexts |
| LME | **30.67%** | BM25 top_k=20, cs512 | Composed: 31.33%. Within 0.66pp! |

### Long Range Understanding

| Benchmark | Metric | Score | Config | Notes |
|-----------|--------|-------|--------|-------|
| Detective QA | SubEM | **49.3%** | BM25 top_k=5, cs4096 | MC, 10 contexts |
| InfBench_sum | ROUGE-Lsum F1 | **29.0%** | BM25 top_k=5, cs4096 | Summarization (SubEM N/A) |

### Test Time Learning

| Benchmark | SubEM | Config | Notes |
|-----------|-------|--------|-------|
| ICL NLU | **80%** | BM25 top_k=5, cs4096 | Best ICL result |
| ICL Banking77 | **75%** | BM25 top_k=5, cs4096 | |
| ICL Clinic150 | **72%** | BM25 top_k=5, cs4096 | |
| ICL TREC Coarse | **54%** | BM25 top_k=5, cs4096 | |
| ICL TREC Fine | **50%** | BM25 top_k=5, cs4096 | |
| Recsys | skipped | — | Missing `processed_data/Recsys_Redial/entity2id.json` |

---

## Summary Table

| Category | Benchmark | Engram BM25 | Composed Agent | Target | Gap |
|----------|-----------|-------------|----------------|--------|-----|
| CR | FC-SH | 51% | 91% | 91%+ | -40pp |
| CR | FC-MH | 1% | 55% | 55%+ | -54pp |
| AR | Ruler QA1 197K | 76% | — | — | baseline |
| AR | Ruler QA2 421K | 39% | — | — | baseline |
| AR | EventQA | 44.8% | — | — | baseline |
| AR | LME | 30.67% | 31.33% | **90%** | -59pp |
| LRU | Detective QA | 49.3% | — | — | baseline |
| LRU | InfBench_sum | 29.0% R-L | — | — | baseline |
| TTL | ICL NLU | 80% | — | — | baseline |
| TTL | ICL Banking77 | 75% | — | — | baseline |
| TTL | ICL Clinic150 | 72% | — | — | baseline |
| TTL | ICL TREC Coarse | 54% | — | — | baseline |
| TTL | ICL TREC Fine | 50% | — | — | baseline |

**Targets:**
- **LME 90%** is the primary target. Current: 30.67%. Chronos SOTA: 95.6%.
- **FC-SH/MH**: match composed agent as minimum (91%/55%), then exceed.

**Key insight:** Engram's pure BM25 matches the composed agent on LME (30.67% vs 31.33%).
The composed agent's entity_prefilter + dynamic_prompting + structured_query + aggregation_overfetch
adds only 0.66pp over Engram's simple BM25 retrieval. The real gaps are on FC benchmarks
where entity-aware features (co-occurrence, triadic, cross-encoder) matter.

---

## Key Fixes This Session

1. **Full query to LLM** (commit 5a0b0d2c): BM25 uses extracted short query, but LLM needs full benchmark prompt with options/instructions. FC-SH: 40->51%. EventQA: 0->44.8%.
2. **Char budget truncation** (earlier): Always include at least one BM25 result, truncated if needed. Detective QA: 1.4->49.3%.
3. **LME config fix**: `Longmemeval_s.yaml` filters for `longmemeval_s_-1_500` which doesn't exist in HF data. Use `Longmemeval_s_star.yaml` (filters for `longmemeval_s*`, matches 5 samples, 300 Qs).
4. **Recsys data dependency**: `metrics_summarization` calls `_process_recsys_dataset` which needs `./processed_data/Recsys_Redial/entity2id.json` — not included in MABench repo.

---

## Engram Status

**Branch:** `worktree-engram-build`
**Tests:** 943/943 passing

### Architecture

```
ontology.py -> storage.py -> [query.py, ingestion.py, writeback.py] -> retrieval.py -> core.py -> dynamics.py
```

### Performance Bottleneck

`Foundry.ingest()` calls `rebuild_derived()` + `rebuild_inferred()` after every call.
With N chunks, that's N full entity graph rebuilds. Recsys (363 chunks, cs4096) took 30+ min
to ingest. Fix: batch ingest mode or lazy rebuild.

### Key findings

1. **LME nearly matches composed agent with pure BM25**: 30.67% vs 31.33%. The composed agent's extra features (entity_prefilter, dynamic_prompting, structured_query, aggregation_overfetch) add marginal value on LME.
2. **Full query to LLM is critical**: FC-SH +11pp, EventQA +44.8pp when passing full benchmark prompt to LLM.
3. **FC-SH gap is entity-aware features**: 51% BM25 vs 91% composed. Co-occurrence + entity pre-filter + cross-encoder + triadic.
4. **FC-MH needs chain retrieval**: 1% BM25 vs 55% composed. Query decomposition + entity linking required.
5. **BM25 strong on structured tasks**: Ruler QA1 (76%), ICL NLU (80%), Detective QA (49%).
6. **Ruler QA scales inversely with haystack**: QA1 (197K) = 76%. QA2 (421K) = 39%.

---

## Eval Runner Architecture

```
foundry/eval/run_engram.py — unified runner
  ├── extract_retrieval_query(raw_query, dataset) — strips benchmark prompt wrappers
  ├── generate_answer(query, evidence, model, dataset) — LLM answer from evidence
  └── main() — ingest → BM25 retrieve → LLM answer → metrics_summarization
```

Supports: FC-SH/MH, EventQA, Ruler QA, Detective QA, LME, InfBench_sum, ICL, Recsys.
Uses ConversationCreator for data loading, metrics_summarization for scoring.
Direct BM25 retrieval (default) or AgenticRetriever (--use_agentic).
