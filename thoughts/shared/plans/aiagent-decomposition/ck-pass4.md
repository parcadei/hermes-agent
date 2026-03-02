# CK Architecture Review -- Pass 4: Cybernetics Kernel Operators

Created: 2026-02-26
Author: architect-agent (CK adversarial review)
Status: Completed
Operators applied: Recursion, Requisite-Variety, Good-Regulator, Ultrastability
Base: spec.md + brenner-pass1.md + brenner-pass2.md + brenner-pass3.md

---

## Context: Revised Decomposition After Brenner Passes

Three Brenner passes produced these revisions to the original 4-class spec:

| Component | Original Form | Revised Form | Rationale |
|-----------|--------------|-------------|-----------|
| PromptAssembler | Class | **Class** (retained) | Owns mutable cache (`_cached_prompt`). Genuine state lifecycle. |
| ToolExecutor | Class | **Module-level function** + frozen dataclass config | Zero mutable state. "Kingdom of Nouns" anti-pattern if kept as class. |
| SessionPersister | Class | **Class** (retained) | Owns mutable `session_id` and `_session_log_file`. File/DB lifecycle. |
| CompressionManager | Class | **DROPPED** | 34 lines, zero state, 5 callbacks threading mutations. Stays on AIAgent. |

The CK review operates on this **revised** 2+1 decomposition (2 classes + 1 function), not the original 4-class spec.

---

## Operator 1: Recursion (structural homology across nesting levels)

**Question:** Does the decomposition maintain the same structural shape at every level of nesting? Does the tool dispatch pattern (registry -> handler -> result) hold consistently from top to bottom?

### 1.1 Top-Level Recursion: AIAgent -> Components

The revised decomposition creates this structure:

```
AIAgent (orchestrator)
  |-- PromptAssembler (class, owns cache)
  |-- execute_tool_calls() (function, stateless dispatch)
  |-- SessionPersister (class, owns session lifecycle)
  |-- _compress_context() (method, stays on AIAgent)
  |-- run_conversation() (method, the 826-line state machine)
```

At the AIAgent level, the pattern is: **orchestrator delegates to specialists**. Each specialist either transforms data (PromptAssembler, execute_tool_calls) or manages a side-effect lifecycle (SessionPersister).

### 1.2 Mid-Level Recursion: execute_tool_calls -> tool dispatch

Inside `execute_tool_calls`, the pattern is:

```
for each tool_call:
  if agent_loop_tool:
    import handler -> call with agent state -> result
  else:
    handle_function_call(name, args, task_id) -> registry.dispatch() -> result
  truncate result -> append to messages -> log to DB
```

This is a **two-tier dispatch**: agent-loop tools bypass the registry and get agent-injected state; everything else goes through the registry. The structural shape at this level is `classify -> dispatch -> normalize result`.

### 1.3 Bottom-Level Recursion: registry.dispatch -> tool handler

Inside `tools/registry.py`, the dispatch is:

```
entry = self._tools.get(name)
if entry.is_async:
  run_async(entry.handler(args, **kwargs))
else:
  entry.handler(args, **kwargs)
```

This is `lookup -> call -> return`. Simple, uniform, no special cases.

### 1.4 Homology Assessment

VERIFIED: The shapes are NOT homologous across levels.

- Top level: orchestrator with 5 heterogeneous delegates (2 classes, 1 function, 2 methods)
- Mid level: two-tier dispatch (agent-loop vs. registry) with embedded UI logic
- Bottom level: uniform lookup-call-return

The top level breaks structural recursion because the 5 delegates have **different interfaces**: PromptAssembler uses build/invalidate, execute_tool_calls uses a function signature with callbacks, SessionPersister uses log/persist/flush, and _compress_context is a plain method. There is no common protocol they share.

**Finding R-1 (MEDIUM):** The decomposition produces heterogeneous components at the top level. This is acceptable for a refactoring step (the goal is cognitive chunking, not algebraic uniformity), but it means you cannot write generic "component management" logic. Each delegate requires specific wiring in AIAgent.__init__ and run_conversation.

### 1.5 The Agent-Loop / Registry Bifurcation

The two-tier dispatch inside execute_tool_calls creates a recursion break. Agent-loop tools follow:

```
name match -> import -> call(state_args) -> result
```

Registry tools follow:

```
handle_function_call(name, args, task_id) -> registry.dispatch(name, args) -> handler(args) -> result
```

These two paths have different error handling, different argument passing conventions, and different import patterns (deferred vs. top-level). The Brenner passes identified the `clarify` membership gap (B1, SC-7) but did not address the deeper structural asymmetry: **agent-loop tools get richer context** (stores, callbacks, parent_agent) while registry tools get only (args, task_id).

**Finding R-2 (LOW):** The two-tier dispatch is a pragmatic choice, not a design flaw. Agent-loop tools genuinely need agent state that registry tools do not. The asymmetry is inherent to the domain, not an artifact of the decomposition. No mitigation needed beyond the B1/SC-7 fixes already proposed.

---

## Operator 2: Requisite-Variety (Ashby's Law)

**Question:** Does the controller variety (2 classes + 1 function) match the disturbance variety (all code paths in run_conversation)?

### 2.1 Disturbance Enumeration: Code Paths in run_conversation

VERIFIED from run_agent.py L1690-2492. The distinct code paths are:

| # | Path | Lines | Exit Type | Components Involved |
|---|------|-------|-----------|-------------------|
| 1 | Invalid response after max retries | L1888-1936 | Early return | SessionPersister |
| 2 | Interrupt during retry wait | L1945-1955 | Early return | SessionPersister |
| 3 | Length finish_reason (context overflow rollback) | L1968-1979 | Early return | SessionPersister |
| 4 | First message truncated | L1982-1992 | Early return | SessionPersister |
| 5 | InterruptedError during API call | L2025-2031 | Break to end | SessionPersister |
| 6 | Interrupt during error handling | L2047-2057 | Early return | SessionPersister |
| 7 | Non-retryable client error (400/401/403/404) | L2062-2086 | Early return | SessionPersister |
| 8 | Context length exceeded -> compress -> retry | L2088-2118 | Continue loop | PromptAssembler, SessionPersister, _compress_context |
| 9 | Invalid tool names in response | L2176-2233 | Early return / continue | SessionPersister |
| 10 | Invalid JSON in tool args | L2250-2287 | Continue (retry) | (none -- stays in AIAgent) |
| 11 | Tool execution -> compression check | L2311-2321 | Continue | execute_tool_calls, PromptAssembler, SessionPersister, _compress_context |
| 12 | Final response (no tool calls) | L2326-2407 | Break to end | SessionPersister |
| 13 | Empty content after think block retries | L2331-2394 | Early return | SessionPersister |
| 14 | Max iterations reached | L2460-2461 | Post-loop | (none -- _handle_max_iterations stays on AIAgent) |
| 15 | Normal completion | L2463-2492 | Post-loop return | SessionPersister |

**Total distinct paths: 15**

### 2.2 Variety Coverage

| Component | Paths it handles | Coverage |
|-----------|-----------------|----------|
| execute_tool_calls() | Path 11 only | 1/15 (7%) |
| PromptAssembler | Paths 8, 11 (build/invalidate) | 2/15 (13%) |
| SessionPersister | Paths 1-9, 11-13, 15 (persist/log on all exit paths) | 13/15 (87%) |
| _compress_context (AIAgent method) | Paths 8, 11 | 2/15 (13%) |
| AIAgent.run_conversation (residual) | ALL 15 paths (orchestration) | 15/15 (100%) |

**Finding V-1 (MEDIUM):** SessionPersister covers the most paths but its role is always the same: call `persist()` or `log_message()`. It does not need to distinguish between the 13 paths it serves -- its interface is uniform. This is good: the variety it absorbs is flattened into a single operation.

**Finding V-2 (HIGH -- confirms Brenner Pass 3, Finding 3):** The decomposition extracts components for **3 of 15 paths** (tool execution, prompt building, compression) and leaves **12 paths** fully on AIAgent. The 12 residual paths are the error handling, retry logic, response validation, and interrupt management in `run_conversation`. This is the complexity center that remains untouched.

The Ashby assessment: **the controller (2+1 decomposition) has less variety than the disturbances (15 code paths)**. The decomposition addresses the stable, repetitive parts of the system (tool dispatch happens every iteration, prompt is built once, persistence happens on every exit) but leaves the high-variety parts (error recovery, retries, interrupt handling) monolithic.

### 2.3 Mitigation for V-2

The Brenner Pass 3 recommendation of "Phase 6: decompose run_conversation" is the correct response to this Ashby violation. However, that future phase needs scoping. Based on the path enumeration:

**Recommended sub-decomposition for Phase 6:**

| Sub-handler | Paths | Lines | Form |
|-------------|-------|-------|------|
| APICallRetrier | Paths 1, 2, 6, 10 | ~120 lines | Function with retry config |
| ResponseValidator | Paths 3, 4, 9, 10, 13 | ~150 lines | Function (pure validation) |
| ErrorRecoveryHandler | Paths 5, 6, 7, 8 | ~100 lines | Function with error classification |
| ConversationLoop | Path 11, 12, 14, 15 | ~200 lines | The residual orchestration |

This is NOT in scope for the current decomposition, but documenting the path enumeration here gives Phase 6 a concrete starting point. Each sub-handler absorbs a coherent subset of the variety.

---

## Operator 3: Good-Regulator (Conant-Ashby theorem)

**Question:** Does each controller component contain an adequate model of the system it controls?

### 3.1 PromptAssembler's Model of Its Inputs

PromptAssembler receives at build() time:

| Input | What it models | Adequate? |
|-------|---------------|-----------|
| `valid_tool_names: Set[str]` | Which tools are loaded | YES -- directly determines guidance injection |
| `system_message: Optional[str]` | Gateway/caller context | YES -- opaque string, no model needed |
| `memory_store: Optional[MemoryStore]` | Persistent memory state | PARTIAL -- calls `format_for_system_prompt()` but does not know if memory was recently written |
| `memory_enabled: bool` | Config flag | YES |
| `user_profile_enabled: bool` | Config flag | YES |

**Finding G-1 (LOW):** PromptAssembler's model of memory_store is adequate for its purpose. It calls `format_for_system_prompt()` which reads from disk. The "recently written" timing concern is handled by the caching contract: the prompt is built once and only rebuilt after compression (which flushes memory first). The Brenner Pass 1 (B2) mitigation -- making the caching contract explicit -- suffices.

### 3.2 execute_tool_calls's Model of AIAgent State

The function-based execute_tool_calls receives:

| Input | What it models | Adequate? |
|-------|---------------|-----------|
| `config: ToolExecConfig` (frozen) | Tool dispatch dependencies | YES -- all 10 fields are config-time constants |
| `is_interrupted: Callable[[], bool]` | Interrupt state | YES -- reads real-time state via closure |
| `log_msg_to_db: Callable[[dict], None]` | Persistence capability | YES -- fire-and-forget side effect |
| `on_tool_executed: Callable[[str], None]` | Nudge counter reset | YES -- simple notification |
| `parent_agent: Optional[Any]` | Delegate context | PROBLEMATIC -- see below |

**Finding G-2 (MEDIUM):** The `parent_agent` parameter is an opaque pass-through to `delegate_task`. execute_tool_calls does not model what delegate_task needs from parent_agent -- it just forwards it. This means execute_tool_calls cannot validate, mock, or type-check this dependency. The function's "model" of delegation is: "I don't know what this is, I pass it through."

This is the correct pragmatic choice (delegate_tool needs 11 AIAgent fields; abstracting them into an interface would be a separate refactoring), but it creates a **Good-Regulator gap**: execute_tool_calls cannot detect if parent_agent is malformed or missing required attributes. If delegate_task is called with a mock that lacks `_delegate_depth`, it will crash at call time, not at construction time.

**Mitigation for G-2:** Add a lightweight runtime check in the delegate_task branch of execute_tool_calls:

```python
elif function_name == "delegate_task":
    if parent_agent is None:
        function_result = json.dumps({"error": "delegate_task requires a parent agent"})
    else:
        function_result = _delegate_task(..., parent_agent=parent_agent)
```

This is a 3-line guard, not a structural change. It makes the Good-Regulator gap visible at the right moment (when delegation is attempted) rather than as an AttributeError deep in delegate_tool.

### 3.3 SessionPersister's Model of Session Lifecycle

SessionPersister owns the session lifecycle: create, log, flush, split-on-compression. Its model:

| State | What it represents | Adequate? |
|-------|-------------------|-----------|
| `_session_id: str` | Current session identity | YES -- sole owner per B6 mitigation |
| `_session_log_file: Path` | Where JSON goes | YES -- atomically updated with session_id |
| `_session_db: Optional[SessionDB]` | SQLite handle | YES -- injected, not owned |
| `_session_start: datetime` | Metadata | YES -- immutable after init |

**Finding G-3 (LOW -- positive):** SessionPersister has a GOOD model of its domain. The Brenner Pass 2 finding (M-5) that the decomposition actually fixes a pre-existing bug (session_log_file not updating after compression) demonstrates that the extracted component's model is MORE accurate than the original. The session_id setter atomically updates both `_session_id` and `_session_log_file`, which the monolithic AIAgent failed to do.

### 3.4 _compress_context's Model (Retained on AIAgent)

Since _compress_context stays on AIAgent, it has direct access to all 10 fields it reads. Its model is the AIAgent itself -- trivially adequate. The Brenner Pass 3 decision to keep this method on AIAgent is validated by the Good-Regulator theorem: any extraction would require reconstructing the model via callbacks, adding indirection without improving accuracy.

---

## Operator 4: Ultrastability (dual-loop separation)

**Question:** Are there a fast inner loop and a slow outer loop, and are they properly separated by the decomposition?

### 4.1 Loop Identification

VERIFIED from run_agent.py:

**Fast inner loop (tool execution cycle):**
```
[API call] -> [parse response] -> [execute_tool_calls] -> [append results] -> [check compression] -> [next API call]
```

Cadence: seconds per iteration. Runs 1-60 times per conversation.
Hot path: execute_tool_calls (tool dispatch + result truncation + message append)

**Slow outer loop (session management):**
```
[build system prompt] -> [run conversation] -> [persist session] -> [flush memories on exit]
```

Cadence: once per user message (or once per session for prompt building).
Cold path: PromptAssembler.build, SessionPersister.persist, _compress_context

**Ultra-slow loop (compression/session splitting):**
```
[detect threshold] -> [flush memories] -> [compress context] -> [rebuild prompt] -> [split session in DB]
```

Cadence: once per ~100K tokens of conversation. May never fire in short sessions.

### 4.2 Separation Assessment

| Loop | Component | Properly isolated? |
|------|-----------|-------------------|
| Fast (tool execution) | execute_tool_calls() | YES -- stateless function, no slow operations |
| Slow (session mgmt) | SessionPersister | MOSTLY -- persist() writes JSON + SQLite on every exit path, but also incrementally via _save_session_log after each tool iteration |
| Ultra-slow (compression) | _compress_context on AIAgent | YES -- clearly separated, triggered by threshold check |

**Finding U-1 (MEDIUM):** The fast and slow loops are NOT fully separated. After every tool execution iteration, `run_conversation` calls:

```python
# L2320-2321
self._session_messages = messages
self._save_session_log(messages)
```

This puts a **slow operation (full JSON file write)** inside the **fast loop (tool execution)**. The Brenner Pass 2 finding (SC-3) calculated this as O(N^2) writes across the session. The decomposition moves `_save_session_log` into SessionPersister but does not change the call site -- it will still be called after every tool iteration.

The fast loop should ideally contain only:
1. execute_tool_calls (tool dispatch)
2. compression check (cheap boolean)
3. message append (in-memory)

And the session JSON write should be deferred to either:
- A periodic timer (every N iterations)
- The slow loop (end of conversation / compression events)
- Both (timer + end-of-conversation flush)

**Mitigation for U-1:** Add a `save_interval` parameter to SessionPersister:

```python
class SessionPersister:
    def __init__(self, *, save_interval: int = 1, ...):
        self._save_interval = save_interval
        self._saves_since_last_write = 0

    def maybe_save_session_log(self, messages: list) -> None:
        """Save session log, but only every N calls.
        
        Reduces filesystem I/O in the fast loop while maintaining
        crash safety (at most N iterations of progress lost).
        """
        self._saves_since_last_write += 1
        if self._saves_since_last_write >= self._save_interval:
            self.save_session_log(messages)
            self._saves_since_last_write = 0
```

Default `save_interval=1` preserves current behavior (write every iteration). Can be tuned to 3-5 for sessions on slow I/O. The `persist()` method always does a full write regardless of interval (called on all exit paths).

### 4.3 Compression as a Loop Separator

The compression event is a clean boundary between the fast and slow loops:

1. Fast loop detects threshold: `context_compressor.should_compress()` returns True
2. Control transfers to slow path: `_compress_context(messages, system_message)`
3. Slow path does: memory flush (API call), compression (API call), prompt rebuild, session split
4. Fast loop resumes with compressed messages and new prompt

This is textbook ultrastability: the fast loop handles routine perturbations (tool calls, retries), and when a structural limit is hit (context window), control passes to the slow loop for reorganization.

**Finding U-2 (POSITIVE):** The compression boundary is well-designed. Keeping `_compress_context` on AIAgent (per Brenner Pass 3) is the right call because it is the interface between the fast and slow loops -- it needs access to both the fast-loop state (messages, compressor) and the slow-loop state (prompt assembler, session persister, memory store).

### 4.4 The Missing Third Loop: Inter-Session Learning

There is an implicit ultra-slow loop that the decomposition does not address:

```
[session ends] -> [memories persisted to disk] -> [next session starts] -> [memories loaded into prompt]
```

This loop operates across sessions (hours/days) and is currently mediated by the filesystem (memory .md files). The decomposition correctly treats this as out-of-scope: MemoryStore is already a standalone class, and PromptAssembler reads from it via `format_for_system_prompt()`.

**Finding U-3 (INFO):** The inter-session learning loop is properly handled by existing abstractions (MemoryStore, build_skills_system_prompt). No decomposition action needed.

---

## Viability Scorecard (VSM S1-S5)

Scoring the revised 2+1 decomposition against Beer's Viable System Model:

### S1: Operations (Do the components handle day-to-day execution?)

| Component | Operational Role | Assessment |
|-----------|-----------------|------------|
| execute_tool_calls() | Dispatches all tool calls in the fast loop | Handles 100% of tool dispatch |
| PromptAssembler | Builds system prompt once per session | Handles 100% of prompt assembly |
| SessionPersister | Logs every message, persists on every exit | Handles 100% of persistence |
| _compress_context (on AIAgent) | Compresses when threshold hit | Handles 100% of compression |

**Score: 2/2** -- All day-to-day operations have a clear owner. No operational gap.

### S2: Coordination (How do components communicate? Are interfaces clean?)

| Interface | Mechanism | Assessment |
|-----------|-----------|------------|
| AIAgent -> execute_tool_calls | Function call with ToolExecConfig + callbacks | Clean, explicit. 4 callbacks (is_interrupted, log_msg_to_db, on_tool_executed, parent_agent) |
| AIAgent -> PromptAssembler | build() with kwargs, invalidate(), cached property | Clean. Caching contract now explicit (B2 fix) |
| AIAgent -> SessionPersister | log_message(), persist(), create_compression_session() | Clean. session_id ownership clear (B6 fix) |
| execute_tool_calls -> SessionPersister | Via log_msg_to_db callback (indirect) | Clean but indirect. execute_tool_calls does not know about SessionPersister |
| _compress_context -> PromptAssembler | invalidate() + build() calls | Direct method calls on self._prompt_assembler |
| _compress_context -> SessionPersister | create_compression_session() | Direct method call on self._persister |

**Score: 2/2** -- Interfaces are explicit and documented. The callback pattern for execute_tool_calls avoids circular dependencies. Cross-component coordination goes through AIAgent (hub-and-spoke, not mesh).

### S3: Optimization (Can each component be optimized independently?)

| Component | Optimizable independently? | Constraint |
|-----------|--------------------------|------------|
| execute_tool_calls() | YES -- can add parallel dispatch, batching, caching | Must maintain messages append contract |
| PromptAssembler | YES -- can cache more aggressively, lazy-load layers | Must maintain layer ordering for prefix caching |
| SessionPersister | YES -- can add write batching, async I/O, compression | Must guarantee no data loss on any exit path |
| _compress_context | PARTIALLY -- compression strategy is in ContextCompressor | Tied to PromptAssembler and SessionPersister for rebuild/split |

**Score: 2/2** -- Each component can be optimized in isolation. The constraints are well-defined and documented.

### S4: Intelligence (Can each component be tested/monitored independently?)

| Component | Independently testable? | Monitoring surface |
|-----------|------------------------|-------------------|
| execute_tool_calls() | YES -- pass mock config, mock callbacks, mock assistant_message | Tool execution count, duration, truncation rate |
| PromptAssembler | YES -- pass known inputs, assert layer ordering | Cache hit/miss, prompt size |
| SessionPersister | YES -- pass mock session_db, temp logs_dir | Write latency, write count, DB error rate |
| _compress_context | NO -- requires AIAgent instance (direct self. accesses) | Compression count (already tracked by ContextCompressor) |

**Score: 1/2** -- Three of four components are independently testable. _compress_context requires AIAgent, reducing its testability to integration-test level only. This is the trade-off of keeping it on AIAgent (Brenner Pass 3 decision). The function-based execute_tool_calls is the biggest testability win: tests can construct a ToolExecConfig with all mocks and verify dispatch without any AIAgent.

### S5: Policy (Is AIAgent still in control of policy decisions?)

Policy decisions that remain on AIAgent:

| Policy | Location | Assessment |
|--------|----------|------------|
| Which tools to load | __init__ (enabled/disabled toolsets) | Correct -- config-time policy |
| When to compress | run_conversation L2313 | Correct -- orchestrator decides |
| When to interrupt | interrupt() method | Correct -- external signal handled by orchestrator |
| Retry limits and backoff | run_conversation retry loops | Correct -- stays on orchestrator |
| Memory nudge timing | run_conversation L1716-1736 | Correct -- conversation-level policy |
| Session ID rotation | _compress_context L1384 | Correct -- orchestrator delegates to SessionPersister.create_compression_session |
| Tool delay between calls | Passed as config to execute_tool_calls | DEBATABLE -- see below |

**Finding S5-1 (LOW):** `tool_delay` is a policy decision (rate limiting) that is passed as configuration to execute_tool_calls. The function executes the delay without understanding why. This is acceptable: the policy setter (AIAgent.__init__ reads from constructor arg) and the policy enforcer (execute_tool_calls applies time.sleep) are correctly separated. The function does not need to understand the policy rationale.

**Score: 2/2** -- AIAgent retains all policy decisions. Extracted components are policy-free executors.

### Total Viability Score: 9/10

| Subsystem | Score | Notes |
|-----------|-------|-------|
| S1 Operations | 2/2 | All operations have clear owners |
| S2 Coordination | 2/2 | Hub-and-spoke through AIAgent, explicit interfaces |
| S3 Optimization | 2/2 | Each component independently optimizable |
| S4 Intelligence | 1/2 | _compress_context not independently testable (accepted trade-off) |
| S5 Policy | 2/2 | AIAgent retains all policy decisions |

---

## Structural Gaps the Brenner Passes Missed

### Gap 1: The `_build_assistant_message` Placement

VERIFIED from run_agent.py L1221-1260. This 40-line method normalizes API response messages into the internal dict format (extracts reasoning, builds tool_calls list, handles reasoning_details). It is called from two places in run_conversation:

- L2292: after tool call validation (tool-call path)
- L2400: final response (no tool calls)

**Where does this belong?** It is currently on AIAgent and the spec does not mention it. It is a pure transformation (API response object -> dict) with no side effects and no mutable state. It could be:

1. A standalone function in `agent/message_format.py`
2. A static method on AIAgent (current, implicitly)
3. Part of execute_tool_calls (only for the tool-call path)

Since it serves BOTH the tool-call and final-response paths, it cannot be absorbed into execute_tool_calls alone. Leaving it on AIAgent is the path of least resistance.

**Mitigation:** Leave on AIAgent. Document in Phase 6 notes as a candidate for extraction when run_conversation is decomposed. No action now.

### Gap 2: The `_extract_reasoning` Method

VERIFIED: `_extract_reasoning` (42 lines, L1221 area) parses reasoning from multiple provider formats (content blocks, reasoning_content field, REASONING_SCRATCHPAD tags). It is called by `_build_assistant_message`. Pure function, no state.

**Where does it belong?** Same answer: leave on AIAgent for now, extract with _build_assistant_message in Phase 6.

### Gap 3: The `_interruptible_api_call` Thread Safety Contract

VERIFIED from run_agent.py L1136-1170. This method runs the API call in a background thread and closes the HTTP client on interrupt. The decomposition does not touch this method, but it has an interaction with execute_tool_calls:

1. `_interruptible_api_call` sets `result["error"] = InterruptedError(...)`
2. `run_conversation` catches this and sets `interrupted = True`
3. The next iteration's tool calls are skipped because `_interrupt_requested` is True

The callback pattern (`is_interrupted: Callable[[], bool]`) correctly propagates this state to execute_tool_calls. But the interrupt also causes `self.client.close()` followed by `self.client = OpenAI(**self._client_kwargs)` -- a client rebuild. If execute_tool_calls were to directly hold a reference to `self.client` (it does not), this would be a race condition. The callback pattern avoids this.

**Mitigation:** None needed. The design correctly isolates the interrupt mechanism. Document as a "why callbacks matter" example in the spec rationale.

### Gap 4: Counter Increment for `_iters_since_skill` Lives in run_conversation

VERIFIED from run_agent.py L1781-1783:

```python
if (self._skill_nudge_interval > 0
        and "skill_manage" in self.valid_tool_names):
    self._iters_since_skill += 1
```

This counter increments per API iteration, NOT per tool call. But the RESET happens inside execute_tool_calls (L1422-1423, when skill_manage is used). The `on_tool_executed` callback proposed in B8 handles the reset. But the INCREMENT stays in run_conversation, which is correct -- it counts API iterations, not tool executions.

**The gap:** The spec does not explicitly document that the increment and reset live in different components. A maintainer might assume `on_tool_executed` handles both.

**Mitigation:** Add a comment in the revised spec:

> **Nudge counter lifecycle:** `_iters_since_skill` is incremented per API iteration in `run_conversation` and reset per tool execution via the `on_tool_executed` callback from `execute_tool_calls`. These are deliberately different scopes: iteration count (how many API calls since last skill use) vs. tool event (when skill_manage was actually called).

---

## Recommended Mitigations (Applied Directly)

The following mitigations incorporate CK operator findings with the Brenner pass revisions:

### M-CK-1: Add Phase 6 Scope Definition (from V-2)

The revised spec should include after Phase 5:

> ### Phase 6: Decompose run_conversation (Future)
>
> **Not in scope for this refactoring.** Documented here as the natural follow-on.
>
> Based on the CK Requisite-Variety analysis, run_conversation contains 15 distinct code paths that are currently monolithic. Recommended sub-handlers:
>
> | Sub-handler | Paths Covered | Estimated Lines |
> |-------------|---------------|-----------------|
> | api_call_with_retry() | Invalid response, interrupt during retry, retryable errors | ~120 |
> | validate_response() | Length overflow, truncation, invalid tools, invalid JSON, empty content | ~150 |
> | handle_api_error() | InterruptedError, client errors, context length exceeded | ~100 |
> | The residual orchestration loop | Tool execution, final response, max iterations | ~200 |
>
> **Prerequisite:** Phases 1-5 must be complete. The extracted components (PromptAssembler, execute_tool_calls, SessionPersister) simplify the orchestration loop enough to make these sub-extractions tractable.

### M-CK-2: Add delegate_task Guard (from G-2)

In the execute_tool_calls function, add a None-check for parent_agent before calling _delegate_task:

```python
elif function_name == "delegate_task":
    if parent_agent is None:
        function_result = json.dumps({
            "error": "delegate_task requires a parent agent context"
        })
    else:
        # ... existing delegate_task dispatch
```

### M-CK-3: Add save_interval to SessionPersister (from U-1)

Extend the SessionPersister interface from Section 3.3:

```python
class SessionPersister:
    def __init__(self, *, save_interval: int = 1, ...):
        self._save_interval = save_interval
        self._iteration_count = 0

    def maybe_save_session_log(self, messages: list) -> None:
        """Incremental save with configurable frequency.
        
        Default save_interval=1 preserves current write-every-iteration behavior.
        Higher values reduce I/O at the cost of crash-safety window.
        """
        self._iteration_count += 1
        if self._iteration_count >= self._save_interval:
            self.save_session_log(messages)
            self._iteration_count = 0
```

Call `maybe_save_session_log` in the fast loop; call `save_session_log` directly in `persist()` (end-of-conversation).

### M-CK-4: Document Nudge Counter Split (from Gap 4)

Add to the spec in Section 6.5:

> **Nudge counter lifecycle (two components):**
> - `_iters_since_skill` INCREMENT: per API iteration in `run_conversation` (conversation-level concern)
> - `_iters_since_skill` RESET: via `on_tool_executed("skill_manage")` callback from `execute_tool_calls` (tool-level event)
> - `_turns_since_memory` INCREMENT: per user turn in `run_conversation`
> - `_turns_since_memory` RESET: via `on_tool_executed("memory")` callback from `execute_tool_calls`
>
> The increment and reset intentionally live in different components because they measure different things: iteration count (frequency) vs. tool event (last usage).

---

## Final Revised Decomposition Recommendation

Incorporating all 3 Brenner passes + CK operators:

### Components

| Component | Form | File | Mutable State | Lines (est.) |
|-----------|------|------|--------------|-------------|
| **PromptAssembler** | Class | `agent/prompt_assembler.py` | `_cached_prompt` | ~80 |
| **execute_tool_calls** | Function + frozen dataclass | `agent/tool_executor.py` | None | ~220 |
| **SessionPersister** | Class | `agent/session_persister.py` | `session_id`, `_session_log_file`, `_iteration_count` | ~250 |
| **_compress_context** | Method on AIAgent | `run_agent.py` (stays) | N/A | ~35 |
| **async_bridge.run_async** | Function | `agent/async_bridge.py` | None | ~20 |

### Interface Summary

```
AIAgent
  |
  |-- self._prompt_assembler = PromptAssembler(platform, skip_context_files)
  |     .build(valid_tool_names, system_message, memory_store, ...) -> str
  |     .cached -> Optional[str]
  |     .invalidate(memory_store) -> None
  |
  |-- execute_tool_calls(config, assistant_message, messages, task_id,
  |     is_interrupted, log_msg_to_db, on_tool_executed, parent_agent)
  |     ToolExecConfig: frozen dataclass with 10 config fields
  |     AGENT_LOOP_TOOLS: frozenset (canonical, imported by model_tools)
  |
  |-- self._persister = SessionPersister(session_id, session_db, logs_dir, ...)
  |     .log_message(msg) -> None
  |     .persist(messages, conversation_history) -> None
  |     .save_session_log(messages) -> None
  |     .maybe_save_session_log(messages) -> None  [NEW: with save_interval]
  |     .flush_to_db(messages, conversation_history) -> None
  |     .save_trajectory(messages, user_query, completed) -> None
  |     .create_compression_session(...) -> str
  |     .session_id -> str (property, sole owner)
  |
  |-- self._compress_context(messages, system_message, approx_tokens) -> tuple
  |     (stays on AIAgent, calls prompt_assembler + persister directly)
```

### Phases (Unchanged from Brenner, with additions)

1. Foundation: Create agent/async_bridge.py, agent/prompt_assembler.py, agent/tool_executor.py, agent/session_persister.py
2. Circular import fix: async_bridge replaces deferred import in registry.py
3. Eager import fix: Gut tools/__init__.py
4. Wire into AIAgent: Compose components, update run_conversation call sites
5. Testing: Unit tests for each component + import cycle tests
6. **(NEW, future)** Decompose run_conversation: Extract retry handler, response validator, error recovery

### What Changed vs. the Original Spec

| Change | Source | Rationale |
|--------|--------|-----------|
| CompressionManager DROPPED | Brenner Pass 3 (Exclusion-Test) | 34 lines, zero state, 5 callbacks worse than inline |
| ToolExecutor -> function | Brenner Pass 3 (Object-Transpose) | Zero mutable state, frozen dataclass is cleaner |
| AGENT_LOOP_TOOLS canonical set on tool_executor | Brenner Pass 1 (B1, B5) | Single source of truth, `clarify` included |
| SessionPersister.session_id as sole owner | Brenner Pass 1 (B6) | Fixes pre-existing session_log_file bug (M-5) |
| PromptAssembler caching contract explicit | Brenner Pass 1 (B2) | Returns cache on subsequent calls, documents invariant |
| on_tool_executed callback added | Brenner Pass 1 (B8) | Enables nudge counter reset from execute_tool_calls |
| save_interval on SessionPersister | CK Ultrastability (U-1) | Separates fast/slow loop I/O |
| delegate_task None guard | CK Good-Regulator (G-2) | Fail-fast instead of AttributeError |
| Phase 6 scoped | CK Requisite-Variety (V-2) + Brenner Pass 3 | Addresses the real complexity center |
| Value proposition reframed | Brenner Pass 3 (Finding 4) | Testability + cognitive chunking, NOT coupling reduction |

### Success Criteria (Updated)

1. `AIAgent.run_conversation()` produces identical results (behavioral equivalence)
2. No circular import cycles
3. `import tools` succeeds without optional dependencies
4. PromptAssembler, SessionPersister, and execute_tool_calls are independently testable in isolation
5. AIAgent method count drops from 30+ to ~18
6. No breaking changes to external callers
7. All existing tests pass
8. **(NEW)** SessionPersister.session_id setter atomically updates _session_log_file (fixes pre-existing bug)
9. **(NEW)** AGENT_LOOP_TOOLS is a single source of truth in agent/tool_executor.py
