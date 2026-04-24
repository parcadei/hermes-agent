# EventFrame Implementation Plan

## Milestone: event-frames

### VAL-001: EventFrame dataclass + EventFrameIndex (storage.py)
- Add `EventFrame` dataclass after `RawChunk` in storage.py
- Fields: id (str), turns (list[int]), timestamp (Optional[datetime]), gist (str), entities (list[str]), fact_ids (list[str]), prev (Optional[str]), next (Optional[str])
- Add `EventFrameIndex` class with: add(), get(frame_id), get_by_turn(turn_idx), get_neighbors(frame_id, window=1), to_dict(), from_dict()
- Add `event_frames: EventFrameIndex` field to `FoundryStore`
- Files: foundry/storage.py

### VAL-002: Event segmentation (ingestion.py)
- Add `segment_events(turns, entity_sets)` function
- Input: list of turn indices + per-turn entity sets (from extraction)
- Logic: consecutive turns with entity overlap > threshold stay in same frame. Topic shift = new frame.
- Fallback: if no entities extracted, group by fixed window of 3 turns
- Returns: list of turn-index groups (each group = one EventFrame's turns)
- Files: foundry/ingestion.py

### VAL-003: Wire into Foundry ingestion (core.py)
- After all chunks ingested in `ingest_turns()`, collect per-turn entity sets
- Call `segment_events()` to get turn groups
- Build EventFrame for each group: id=uuid, turns=group, timestamp=first turn's session_time, gist=first turn text[:200], entities=union of turn entities, fact_ids=facts from those turns
- Wire prev/next pointers as doubly-linked list
- Store in `self._store.event_frames`
- Files: foundry/core.py

### VAL-004: Frame-aware retrieval (retrieval.py)
- In `_recall_retrieve()`, replace doc_id±1 contiguity with event frame lookup
- When BM25 returns turn N, look up EventFrame containing turn N
- Pull prev frame's turns and next frame's turns as bonus candidates
- Score bonus turns at original_score * 0.1 (same as current contiguity)
- Gate on new parameter `use_event_frames: bool = False`
- Files: foundry/retrieval.py

### VAL-005: Eval integration (run_engram.py)
- Add `--event_frames` CLI flag
- Pass `use_event_frames=True` to recall()
- Run full 5-ctx eval, verify >= 196/300
- Files: foundry/eval/run_engram.py
