# Nemori: Architecture Analysis for Hermes Integration

**Repo**: https://github.com/nemori-ai/nemori (cloned to /tmp/nemori for analysis)
**Paper**: Nemori — two-step alignment + predict-calibrate for conversational memory
**Analysis date**: 2026-02-27

---

## What Nemori Is

A memory **formation** system. Turns conversation streams into organized memory units. Three modules, two databases, one retrieval engine.

### Pipeline

```
Messages → Buffer → Segmentation → Episode Narrative → Predict-Calibrate → Semantic Facts
                                          ↓                                      ↓
                                   Episodic DB                            Semantic DB
                                          ↓                                      ↓
                                   ←←←←←← Hybrid Search (BM25 + ChromaDB + RRF) ←←←←←←
```

### Module 1: Topic Segmentation (boundary alignment)

- Messages accumulate in per-user `MessageBuffer` (src/models/message.py:96)
- `BatchSegmenter` (src/generation/batch_segmenter.py) uses LLM to group messages into episodes
- Trigger: buffer hits `batch_threshold=20` messages
- Output: list of message index groups, possibly non-contiguous (topic coherence > temporal order)
- Fallback: if batch segmentation disabled, triggers at `buffer_size_max=25`

### Module 2: Episode Generation (representation alignment)

- `EpisodeGenerator` (src/generation/episode_generator.py) converts raw messages → third-person narrative
- Output: `Episode(title, content, original_messages, boundary_reason)` (src/models/episode.py)
- Key transforms: resolves references ("she" → "Dr. Chen"), resolves relative time ("yesterday" → date), compresses filler
- Episode merging: `EpisodeMerger` checks if new episode overlaps existing one (similarity > 0.85), merges if so

### Module 3: Semantic Memory Generation (predict-calibrate)

- `PredictionCorrectionEngine` (src/generation/prediction_correction_engine.py) — the paper's core innovation
- **Step 1 - Predict**: Given episode title + existing semantic knowledge, predict what conversation contained
- **Step 2 - Calibrate**: Compare prediction against raw conversation, extract the GAP (what system didn't know)
- **Step 3 - Integrate**: Store gap as new `SemanticMemory` objects
- Output: `SemanticMemory(content, knowledge_type, source_episodes, confidence)` (src/models/semantic.py)
- Ablation: predict-calibrate scores 0.615 vs 0.518 naive extraction (18.7% improvement)

### Retrieval

- `UnifiedSearchEngine` (src/search/unified_search.py) — BM25 + ChromaDB vector search
- Hybrid fusion via Reciprocal Rank Fusion (RRF, k=60)
- Parallel search across both databases
- No dynamics, no decay, no scoring beyond cosine similarity + BM25

---

## Key Files

| File | Purpose |
|------|---------|
| `src/core/memory_system.py` | Main orchestrator (~1000 lines). Buffer mgmt, episode creation, semantic scheduling, search |
| `src/core/message_buffer.py` | Per-user buffer management |
| `src/generation/batch_segmenter.py` | LLM-based batch segmentation |
| `src/generation/episode_generator.py` | Raw messages → narrative episodes |
| `src/generation/semantic_generator.py` | Semantic extraction (with/without predict-calibrate) |
| `src/generation/prediction_correction_engine.py` | Predict-calibrate cycle |
| `src/generation/episode_merger.py` | Dedup overlapping episodes |
| `src/generation/prompts.py` | All LLM prompt templates |
| `src/search/unified_search.py` | Hybrid BM25 + vector search with RRF |
| `src/search/bm25_search.py` | BM25 lexical search |
| `src/search/chroma_search.py` | ChromaDB vector search |
| `src/models/episode.py` | Episode dataclass |
| `src/models/semantic.py` | SemanticMemory dataclass |
| `src/models/message.py` | Message + MessageBuffer dataclasses |
| `src/config.py` | MemoryConfig with all tunables |
| `src/api/facade.py` | NemoriMemory — simplified public API |

---

## Config Defaults (src/config.py)

```
buffer_size_min: 2
buffer_size_max: 25
batch_threshold: 20
episode_min_messages: 2
episode_max_messages: 25
merge_similarity_threshold: 0.85
semantic_similarity_threshold: 1.0  (dedup effectively DISABLED)
enable_prediction_correction: True
search_top_k_episodes: 10
search_top_k_semantic: 10
llm_model: gpt-4o-mini
embedding_model: text-embedding-3-small
embedding_dimension: 1536
```

---

## What Nemori Does NOT Have (gaps for Hermes to fill)

### 1. Encoding Policy (CRITICAL — this session's target)
No gate deciding WHAT becomes a memory. Every message enters the buffer. Greetings, corrections, preferences, reasoning chains — all treated equally. The segmentation decides HOW to chunk, not WHETHER to store.

### 2. Memory Dynamics
- No R (recency), no S (strength), no decay
- No importance feedback loop
- No novelty bonus for new memories
- `SemanticMemory.confidence` hardcoded to 0.8-0.9, never used for ranking
- `SemanticMemory.revision_count` and `updated_at` exist but nothing reads them
- Memories are immutable after creation — equal standing forever

### 3. Retrieval Integration
- `NemoriMemory.search()` returns raw dicts
- No prompt injection, no context assembly, no summarization
- Caller must format results into LLM prompt
- The paper describes injecting top-k episodes + top-2k semantic into prompt — not implemented

### 4. Contradiction/Supersession
- No mechanism for "deadline moved from March to April"
- Both facts coexist with equal standing
- Predict-calibrate reduces redundancy (only stores the gap) but doesn't handle updates to existing facts
- `SemanticMemory.update_content()` exists but pipeline never calls it

### 5. Compression/Consolidation
- No episodic → semantic aging
- No offline re-scanning of old episodes
- No pattern abstraction ("user corrected me on X 3 times" → "user knows X well")
- Episode merging only handles temporal duplicates, not conceptual consolidation

---

## Integration Architecture: Nemori + Hermes Dynamics

### Connection Points

1. **Nemori's cosine similarity → replaced by dynamics-aware scoring**
   - Current: `score = cosine_sim(query, memory)`
   - Target: `score = w₁·relevance + w₂·R(t,S) + w₃·importance + w₄·σ(activation)`
   - Cosine similarity becomes one component (relevance), not the whole score

2. **Memory creation → initialize R/S state**
   - When predict-calibrate produces SemanticMemory: set R=1.0, S=S_initial
   - When episode is created: set R=1.0, S=S_initial (different params)

3. **Each retrieval → dynamics update**
   - Selected memory: S += α·(Smax - S) (strength jump)
   - All memories: R decays via exp(-t/S) between accesses

4. **Two populations with separate parameter regimes**
   - Episodic: (α_ep, β_ep) — faster decay, more volatile
   - Semantic: (α_sem, β_sem) — slower decay, more stable
   - Both satisfy contraction condition independently

### Composed Scoring Constraint (PROVEN)

**Extended contraction condition**: K < 1 AND w_rec < 0.5

- Verified in `test_contraction.py::TestComposedScoringContraction` (2 tests, 700 samples)
- With balanced weights (none exceeding 0.4), contraction always preserved
- Breaks when w_rec > 0.5 and S near zero (dR/dS diverges as S → 0)
- Documented in MEMORY_SYSTEM.md section 4.4.1

---

## Event-Driven Architecture

Nemori uses an internal `EventBus`:
- `episode_created` event triggers semantic generation
- `episode_deleted` cascades to semantic cleanup
- Semantic generation runs async in ThreadPoolExecutor (8 workers default)
- Per-user RLock prevents concurrent processing conflicts

This event-driven design maps cleanly to dynamics integration — add a `memory_accessed` event that triggers R/S updates.

---

## Paper Results (for reference)

```
LoCoMo benchmark (LLM-judge score):
  Full Context (24K tokens):  0.723
  Best baseline (Mem0):       0.613
  Nemori:                     0.744  ← beats full context

Token efficiency:
  Full Context: 23,653 tokens
  Nemori:        2,745 tokens  ← 88% reduction

LongMemEvalS (105K token conversations):
  Full Context: 55.0% avg accuracy
  Nemori:       64.2% avg accuracy  ← +9.2% with 95% fewer tokens
```

Key insight: reasoning at formation time (narration + predict-calibrate) > reasoning at query time (stuffing raw logs into context).
