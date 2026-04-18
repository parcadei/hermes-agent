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

### Other Benchmarks (not yet evaluated)

| Benchmark | Category | Description |
|-----------|----------|-------------|
| EventQA | Accurate Retrieval | Event-based QA, full context 800k |
| Detective_QA | Long Range Understanding | Detective story comprehension |
| InfBench_sum | Long Range Understanding | Summarization |
| Ruler QA | Accurate Retrieval | Ruler-based QA |
| ICL (5 configs) | Test Time Learning | In-context learning classification |
| Recsys | Test Time Learning | Recommendation system |

---

## Engram Status

**Branch:** `worktree-engram-build` at `54cbd3b6`
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

---

## Next: Isolated Engram Runners

Writing `foundry/eval/run_engram_fc.py` — isolated from composed agent, uses Engram API directly.

### MABench eval plan

1. **FC-SH / FC-MH** — fact consolidation with supersession (start here)
2. **LME** — long memory eval, multi-session conversations
3. **EventQA** — event-based temporal QA
4. **Detective_QA** — long-range comprehension
5. **InfBench_sum** — summarization
6. **Ruler** — ruler QA
7. **ICL** — in-context learning
8. **Recsys** — recommendations
