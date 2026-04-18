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

**March "76" breakdown (verified):**
- 50 hits on 66 clean questions
- 5 hits on corrected questions where answer didn't change
- 21 hits matching wrong GT labels (our system retrieved older facts, GT also had older facts)
- Re-scored against corrected GT: 55

**Corrected GT source:** `output/factconsolidation_mh_corrected_answers.json` (34 entries, 29 actual answer changes)

**Benchmark protocol (from prompt):** "Each fact in the knowledge pool is provided with a
serial number... the newer fact has larger serial number. You need to solve conflicts by
finding the newest fact." System must answer from knowledge pool, not real-world facts.

### LME (LongMemEval, 300 questions)

| Date | Config | SubEM |
|------|--------|-------|
| 2026-04-12 | entpf+dynprompt+rawchunk+aggof25+struct_h3 | 31.33% |

Best config: `--bm25_primary --entity_prefilter --dynamic_prompting --raw_chunks --aggregation_overfetch 25 --structured_query --structured_query_max_hops 3 --chunk_size 512 --no_cache`

---

## Engram Baselines (all benchmarks)

All Engram runs use `foundry/eval/run_engram.py`, direct BM25 retrieval, `gpt-4o-mini` for answer generation.

### Conflict Resolution

| Benchmark | SubEM | Config | Notes |
|-----------|-------|--------|-------|
| FC-SH | **40%** | BM25 top_k=20, cs512 | Composed: 91% |
| FC-MH | **1%** | BM25 top_k=20, cs512, corrected GT | Composed: 55% (corrected) |

FC-SH gap: 40% vs 91% — missing co-occurrence boost, entity pre-filter, cross-encoder, triadic.
FC-MH: multi-hop needs chain retrieval (query decomposition + entity linking), BM25 alone is hopeless.

### Accurate Retrieval

| Benchmark | SubEM | Config | Notes |
|-----------|-------|--------|-------|
| Ruler QA (197K) | **76%** | BM25 top_k=20, cs512 | Simple fact retrieval, BM25 works well |
| EventQA (full) | TBD | BM25 top_k=5, cs4096 | Running (5 contexts × 100 Qs) |

### Long Range Understanding

| Benchmark | SubEM | Config | Notes |
|-----------|-------|--------|-------|
| Detective QA | **49.3%** | BM25 top_k=5, cs4096 | MC, 10 contexts × ~7 Qs |
| InfBench_sum | — | Not yet run | Summarization |

### Test Time Learning

| Benchmark | SubEM | Config | Notes |
|-----------|-------|--------|-------|
| ICL NLU | **80%** | BM25 top_k=5, cs4096 | Best ICL result |
| ICL Banking77 | **75%** | BM25 top_k=5, cs4096 | Classification via similar examples |
| ICL Clinic150 | **72%** | BM25 top_k=5, cs4096 | |
| ICL TREC Coarse | **54%** | BM25 top_k=5, cs4096 | |
| ICL TREC Fine | **50%** | BM25 top_k=5, cs4096 | |
| Recsys | — | Not yet run | |

---

## Engram Status

**Branch:** `worktree-engram-build`
**Tests:** 943/943 passing

### Architecture (13 modules)

```
ontology.py -> storage.py -> [query.py, ingestion.py, writeback.py] -> retrieval.py -> core.py -> dynamics.py
```

### Features (8 base + 5 new)

| Feature | Module | Status |
|---------|--------|--------|
| BM25 + entity graph | storage.py | Base |
| Temporal index | storage.py | Base |
| Agentic retriever (7-tool) | retrieval.py | Base |
| Triadic memory | storage.py | New, wired |
| Query decomposition | query.py | New, wired |
| Cross-encoder reranking | retrieval.py | New, wired |
| Conflict resolution (opt-in) | writeback.py | New, wired |
| LLM extraction (v9 prompt) | ingestion.py | New, wired |

### Recent changes (this session)

- Stripped DeterministicRouter (500+ lines) — all retrieval via AgenticRetriever
- Removed hardcoded regex extraction from core.py → ingestion.py handles extraction
- Net -643 lines
- Built `foundry/eval/run_engram.py` — unified MABench runner for all benchmarks
- Built `foundry/eval/run_engram_fc.py` — FC-specific runner (superseded by generic)

### Key findings

1. **BM25 alone gives strong baselines on structured tasks**: Ruler QA (76%), ICL (75%), Detective QA (49%). These don't need entity linking or multi-hop.
2. **FC-SH benefits heavily from entity-aware retrieval**: 40% BM25-only vs 91% composed agent. The gap is co-occurrence boost + entity pre-filter + cross-encoder.
3. **FC-MH is hopeless without chain retrieval**: 1% with BM25. Need query decomposition + entity linking.
4. **Char budget bug**: Large chunks (>12K chars) were silently dropped from evidence — fixed by always including at least the first result (truncated).
5. **Query extraction is critical**: FC/Ruler queries wrapped in benchmark prompts. Without extraction, SubEM=0%.

---

## Eval Runner Architecture

```
foundry/eval/run_engram.py — unified runner
  ├── extract_retrieval_query(raw_query, dataset) — strips benchmark prompt wrappers
  ├── generate_answer(query, evidence, model, dataset) — LLM answer from evidence
  └── main() — ingest → BM25 retrieve → LLM answer → metrics_summarization
```

Supports: FC-SH/MH, EventQA, Ruler QA, Detective QA, InfBench_sum, ICL, Recsys.
Uses ConversationCreator for data loading, metrics_summarization for scoring.
Direct BM25 retrieval (default) or AgenticRetriever (--use_agentic).
