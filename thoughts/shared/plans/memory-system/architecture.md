# Hermes Memory System — Architecture Plan

Created: 2026-02-26
Status: DRAFT — bird's-eye view, not yet implementation-ready
Source: session_20260226_040349_research_doc.md (30+ papers synthesized)

---

## 1. System Overview

Replace the flat-file MemoryStore (MEMORY.md / USER.md) with a layered memory
system that supports semantic retrieval, decay, and post-session consolidation.

```
                    ┌──────────────────────────────┐
                    │        AIAgent Turn           │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐   ┌───────────┐   ┌──────────────┐
        │  RECALL   │   │  WORKING  │   │   WRITE      │
        │  (read)   │   │  MEMORY   │   │   (persist)  │
        └─────┬────┘   │ (context  │   └──────┬───────┘
              │        │  window)  │          │
              ▼        └───────────┘          ▼
        ┌──────────┐                   ┌──────────────┐
        │  SCORE   │                   │ CONSOLIDATE  │
        │  & RANK  │                   │  (async LLM) │
        └─────┬────┘                   └──────┬───────┘
              │                               │
              └───────────┬───────────────────┘
                          ▼
                   ┌──────────────┐
                   │  MEMORY DB   │
                   │  (embeddings │
                   │  + metadata) │
                   └──────────────┘
                          │
                   ┌──────┴──────┐
                   │ Modal       │
                   │ Embedder    │
                   │ (Arctic S)  │
                   └─────────────┘
```

---

## 2. Open Decisions

### D1: Where does the memory DB live?

| Option | Pros | Cons |
|--------|------|------|
| A. Local SQLite + sqlite-vec (~/.hermes/memory.db) | No network for reads, simple, works offline | Can't share across machines, no gateway access from Modal sandbox |
| B. SQLite on Modal Volume | Co-located with embedder, gateway-accessible | Network latency for every read, single-writer constraint |
| C. Postgres (existing continuous_claude DB) | Already running, cross-terminal aware, real concurrency | Heavier dependency, needs pgvector extension, not portable |
| D. Hybrid: local SQLite for reads, sync to remote for cross-device | Best of both | Sync complexity |

**Leaning:** Option A (local SQLite + sqlite-vec) for v1. Simplest, fastest reads,
and the primary user is a single CLI session. Gateway can access the same file
if running on the same machine. Revisit if cross-machine sharing becomes a need.

### D2: Migration from current MemoryStore

| Option | Description |
|--------|-------------|
| A. Replace entirely | New system replaces MEMORY.md/USER.md. Migrate existing entries. |
| B. Wrap | New system wraps old MemoryStore. Old tool interface unchanged, new backend. |
| C. Run alongside | Old system for backward compat, new system as separate tool. |

**Leaning:** Option B (wrap). The `memory` tool interface (add/replace/remove) stays
the same from the agent's perspective. Under the hood, entries go to the new DB
with embeddings. The flat-file snapshot for system prompt injection transitions
to a "top-k recalled" block.

### D3: Consolidation model

Who evaluates what's worth remembering after a session?

| Option | Cost | Quality |
|--------|------|---------|
| Auxiliary client (cheap model, e.g., Haiku) | ~$0.001/session | Good enough for extraction |
| Main model | ~$0.10-1.50/session | Better judgment, expensive |
| Rule-based (no LLM) | Free | Misses nuance |

**Leaning:** Auxiliary client. The pattern already exists in the codebase
(agent/auxiliary_client.py). Consolidation is a summarization/extraction task
that a cheap model handles well.

### D4: Recall gating — when to search memory

Not every turn needs memory retrieval. Options:

| Strategy | Description |
|----------|-------------|
| Always | Embed every user message, retrieve top-k. Simple but adds latency. |
| Keyword gate | Skip recall for short/trivial messages ("yes", "ok", "thanks") |
| LLM gate | Ask cheap model "does this need context?" — adds another call |
| Heuristic | Skip for follow-up turns (no new topic), recall on first turn + topic changes |

**Leaning:** Heuristic for v1. Recall on first message of session + when
message length > N tokens or contains a question. Can iterate from usage data.

### D5: How many memories to recall per turn

- Top-k with k=3-5 seems reasonable for prompt injection
- Total budget: ~500-800 tokens of recalled context
- Scoring formula determines ranking (see Section 4)

---

## 3. Memory Entry Schema

```python
@dataclass
class MemoryEntry:
    id: str                    # UUID
    content: str               # The memory text
    embedding: list[float]     # 384-dim (Arctic Embed S)
    entry_type: str            # "episodic" | "semantic" | "procedural" | "user"
    importance: float          # 0.0-1.0, scored at write time
    created_at: datetime
    last_accessed: datetime
    access_count: int
    access_times: list[datetime]  # For ACT-R activation calc
    source_session: str        # Session ID that created it
    tags: list[str]            # Optional categorization
    strength: float            # Memory strength (affects decay rate)
    active: bool               # Soft delete
```

SQLite schema (with sqlite-vec for vector search):

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    entry_type TEXT NOT NULL DEFAULT 'episodic',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    access_times TEXT,  -- JSON array of ISO timestamps
    source_session TEXT,
    tags TEXT,           -- JSON array
    strength REAL NOT NULL DEFAULT 1.0,
    active INTEGER NOT NULL DEFAULT 1
);

-- sqlite-vec virtual table for vector search
CREATE VIRTUAL TABLE memory_embeddings USING vec0(
    id TEXT PRIMARY KEY,
    embedding float[384]
);
```

---

## 4. Retrieval Scoring

From the research doc, adapted for implementation:

```python
def score_memory(query_embedding, memory, current_time):
    # Semantic relevance (cosine similarity)
    relevance = cosine_sim(query_embedding, memory.embedding)

    # Recency decay
    hours_since_access = (current_time - memory.last_accessed).total_seconds() / 3600
    recency = math.exp(-hours_since_access / (memory.strength * 24))  # strength in days

    # ACT-R base-level activation
    if memory.access_times:
        activation = math.log(sum(
            (current_time - t).total_seconds() ** -0.5
            for t in memory.access_times
            if (current_time - t).total_seconds() > 0
        ))
        activation = sigmoid(activation)
    else:
        activation = 0.0

    # Weighted combination
    score = (
        0.35 * relevance +
        0.25 * recency +
        0.20 * memory.importance +
        0.20 * activation
    )
    return score
```

Weights are initial guesses — tune from observed recall quality.

---

## 5. Component Map

```
agent/memory/
  ├── __init__.py
  ├── modal_app.py           # Modal App: Embedder class (deploy to Modal)
  ├── embedding_client.py    # Local caller: Cls.from_name() wrapper
  ├── store.py               # MemoryDB: SQLite + sqlite-vec operations
  ├── scoring.py             # Retrieval scoring (Section 4 formula)
  ├── recall.py              # Read path: gate → embed → retrieve → rank → format
  ├── consolidation.py       # Write path: extract → score importance → embed → store
  └── migration.py           # One-time migration from MEMORY.md / USER.md
```

Integration points in existing code:

| Component | Integration Point | How |
|-----------|------------------|-----|
| Recall | `PromptAssembler.build()` | New layer: inject recalled memories between identity and tool guidance |
| Consolidation | `SessionPersister.persist()` | Trigger async consolidation after session save |
| Memory tool | `tools/memory_tool.py` | Wrap existing add/replace/remove to also write to new DB |
| Embedding | Modal deployment | Separate `modal deploy agent/memory/modal_app.py` |
| Gating | `AIAgent.run_conversation()` | Check before first PromptAssembler.build() call |

---

## 6. Build Phases

### Phase 1: Modal Embedder
- Create `agent/memory/modal_app.py` with Embedder class
- Create `agent/memory/embedding_client.py` with local caller
- Deploy and test: embed a string, get a vector back
- Acceptance: `embedder.embed.remote(["hello"]) → [[0.02, -0.11, ...]]`

### Phase 2: Memory Store
- Create `agent/memory/store.py` with SQLite + sqlite-vec
- CRUD operations: add, get, search_by_embedding, update_access, deactivate
- Schema from Section 3
- Tests: unit tests for all CRUD ops
- Acceptance: store a memory with embedding, retrieve by vector similarity

### Phase 3: Recall Path
- Create `agent/memory/scoring.py` with scoring formula
- Create `agent/memory/recall.py` with gating + retrieve + rank + format
- Wire into `PromptAssembler.build()` as new layer
- Feature flag: `HERMES_MEMORY_V2=true` to enable
- Tests: unit tests for scoring, integration test for recall in prompt
- Acceptance: user message triggers recall, top-k memories appear in system prompt

### Phase 4: Write Path (Consolidation)
- Create `agent/memory/consolidation.py`
- Post-session extraction using auxiliary client
- Importance scoring at write time
- Embed and store extracted memories
- Trigger from `SessionPersister.persist()`
- Tests: mock LLM consolidation, verify entries stored
- Acceptance: after a session, new memories appear in DB with embeddings

### Phase 5: Migration + Memory Tool Integration
- Create `agent/memory/migration.py` to import MEMORY.md entries
- Update `tools/memory_tool.py` to write to new DB (dual-write initially)
- Deprecation path for flat files
- Acceptance: existing memories searchable via new system

### Phase 6: Tuning + Observability
- Recall hit rate logging
- Latency tracking per turn
- Weight adjustment based on observed quality
- Dashboard or log output for memory system health

---

## 7. Dependencies

### New
- `sqlite-vec` — SQLite extension for vector search (pip installable)
- `sentence-transformers` — only needed on Modal side (not local)
- `modal` — already an optional dep

### Existing (reused)
- `agent/auxiliary_client.py` — for consolidation LLM calls
- `agent/prompt_assembler.py` — injection point for recall
- `agent/session_persister.py` — trigger point for consolidation
- Modal auth — `~/.modal.toml` already configured

---

## 8. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Modal cold start adds 2-5s to first recall | User perceives slow first response | Warm container via scaledown_window=300; cache embeddings locally after first call |
| sqlite-vec not available on all platforms | Install failure | Pure-Python fallback (brute-force cosine sim on small DBs) |
| Consolidation LLM hallucinates memories | False memories injected | Require consolidation to quote source messages; human-reviewable log |
| Recall injects irrelevant context | Worse responses than no memory | Gating + high relevance threshold + feature flag to disable |
| Scoring weights are wrong | Poor retrieval quality | Start conservative (high relevance weight), iterate from logs |

---

## 9. Out of Scope (v1)

- Cross-machine memory sync
- Multi-user memory isolation (gateway use case)
- Procedural memory / skill learning (Phase 2 of broader vision)
- Working memory / Ouros persistent REPL (separate project)
- Forgetting policies (active pruning) — v1 just uses decay in scoring
- Memory compression (merging similar memories) — nice-to-have, not v1

---

## 10. Open Questions

1. Should sqlite-vec be installed locally or only on Modal?
   - If local: fastest reads, but adds a native dependency
   - If Modal: all vector ops go through Modal, adds latency

2. What's the right importance scoring prompt for the consolidation LLM?
   - Need to test what auxiliary models produce useful 0-1 scores

3. Should recall be async (non-blocking) or sync (blocks prompt assembly)?
   - Sync is simpler but adds latency to every gated turn
   - Async could pre-fetch while other prompt layers build

4. How do we handle the existing `memory` tool's flat-file snapshot in system prompt?
   - Currently frozen at session start for prompt cache stability
   - New recall could be dynamic (different memories per turn)
   - This breaks the caching invariant — need to think through
