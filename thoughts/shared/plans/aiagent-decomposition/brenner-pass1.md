# AIAgent Decomposition Spec -- Brenner Pass 1: Level-Split + Paradox-Hunt

Generated: 2026-02-26
Operator: architect-agent (Brenner adversarial review)
Base: spec.md (same directory)

---

## Findings Summary

| # | Operator | Finding | Severity | Section Affected |
|---|----------|---------|----------|------------------|
| B1 | ⊘ Level-Split | Agent-loop tool set defined at two levels with disagreeing membership | **HIGH** | 3.2 ToolExecutor, 2.6 model_tools.py |
| B2 | ⊘ Level-Split | "Stateless except for caching" conflates config-time vs runtime parameters | **MEDIUM** | 3.1 PromptAssembler |
| B3 | ◊ Paradox-Hunt | CompressionManager is "thin orchestrator" but owns the compression-enabled gate | **MEDIUM** | 3.4 CompressionManager, 6.2 |
| B4 | ⊘ Level-Split | ToolExecutor conflates dispatch routing with UI/display concerns | **MEDIUM** | 3.2 ToolExecutor, 6.6 |
| B5 | ◊ Paradox-Hunt | Spec says "no breaking changes" but `_AGENT_LOOP_TOOLS` constant stays orphaned | **HIGH** | 4, 3.2 |
| B6 | ⊘ Level-Split | session_id ownership is split across three objects without a single authority | **MEDIUM** | 6.2, 3.3 SessionPersister |
| B7 | ◊ Paradox-Hunt | `flush_memories` needs the API client but CompressionManager has no client reference | **LOW** | 3.4 CompressionManager |
| B8 | ⊘ Level-Split | "Tool delay" and "nudge counter" are conversation-level policy, not execution-level | **LOW** | 3.2 ToolExecutor, 6.5 |
| B9 | ◊ Paradox-Hunt | Spec claims tools/__init__.py is 90% covered but proposes gutting it -- test regression risk | **LOW** | 5 |
| B10 | ⊘ Level-Split | `_build_api_kwargs` and `_handle_max_iterations` duplicate provider/reasoning config logic | **LOW** | Not in spec (orphan) |

---

## B1: Agent-Loop Tool Set Mismatch (⊘ Level-Split)

### The split the spec misses

The concept "agent-loop tool" exists at **two different levels** that disagree on membership:

1. **model_tools.py L240** -- `_AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}` (4 tools)
2. **run_agent.py L1445-1521** -- `_execute_tool_calls` inline dispatch handles: `todo`, `session_search`, `memory`, `clarify`, `delegate_task` (5 tools)

`clarify` is handled inline in `_execute_tool_calls` (it needs `self.clarify_callback`, which is agent-level state) but is **not** in `_AGENT_LOOP_TOOLS`. This means:

- If `clarify` somehow reaches `handle_function_call`, it will be dispatched through the registry (which has its own `clarify_tool` handler), NOT through the agent-loop path. The registry handler would be called **without** the `callback` argument, meaning it would get `callback=None` and return an error to the LLM.
- This is currently masked because `_execute_tool_calls` checks the name before falling through to `handle_function_call`. But the defensive guard in `model_tools.py` is incomplete.

The spec's proposed `ToolExecutor` (Section 3.2) inherits this ambiguity. Its docstring says it handles "todo, memory, session_search, clarify, delegate_task" (5 tools) but the spec never proposes updating `_AGENT_LOOP_TOOLS` to match.

### Mitigation (inline in spec Section 3.2)

Add to ToolExecutor:

```python
class ToolExecutor:
    # Canonical set of tools dispatched inline (not through registry).
    # This is the SINGLE SOURCE OF TRUTH -- model_tools._AGENT_LOOP_TOOLS
    # must be updated to match (or import from here).
    AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "clarify", "delegate_task"}
```

Add to Section 4 (Circular Import Fix), as a **new sub-step**:

> **4.1 Unify agent-loop tool set:**
> Move the canonical set of agent-loop tool names into `agent/tool_executor.py` as `ToolExecutor.AGENT_LOOP_TOOLS`. Update `model_tools._AGENT_LOOP_TOOLS` to import from there:
> ```python
> from agent.tool_executor import ToolExecutor
> _AGENT_LOOP_TOOLS = ToolExecutor.AGENT_LOOP_TOOLS
> ```
> This eliminates the two-source-of-truth problem and ensures the guard in `handle_function_call` matches the actual dispatch in `ToolExecutor.execute()`.

Add to Section 3.2 tests:

> - `clarify` tool is dispatched via inline path (not through `handle_function_call`)
> - Every tool in `ToolExecutor.AGENT_LOOP_TOOLS` is handled by a specific branch in `execute()`
> - Every tool NOT in `AGENT_LOOP_TOOLS` goes through `handle_function_call`

---

## B2: PromptAssembler "Stateless Except for Caching" Is a Level Confusion (⊘ Level-Split)

### The split the spec conflates

The spec describes PromptAssembler as "stateless except for caching" (Section 3.1 docstring). But the constructor takes `platform` and `skip_context_files` -- these are **configuration-time** values that never change. Meanwhile, `build()` takes `valid_tool_names`, `memory_store`, `memory_enabled`, `user_profile_enabled`, and `system_message` -- these are **call-time** (potentially varying per invocation) parameters.

The confusion: `platform` and `skip_context_files` are not "state" in the mutable-state sense, they are **configuration**. But the caching behavior creates a hidden dependency: after the first `build()` call, the cached result depends on the *call-time* parameters that were passed. If a second `build()` is called with different `valid_tool_names` or `memory_enabled`, the cache would serve stale data -- BUT the spec says the cache is only invalidated by `invalidate()` (called after compression).

Currently this is safe because `build()` is called once per session and the call-time parameters don't change within a session. But the interface **does not enforce** this invariant. A caller could reasonably call `build()` with different parameters and get the cached (wrong) result.

### Mitigation (inline in spec Section 3.1)

Clarify the caching contract:

```python
def build(
    self,
    *,
    valid_tool_names: Set[str],
    system_message: Optional[str] = None,
    memory_store: Optional[MemoryStore] = None,
    memory_enabled: bool = False,
    user_profile_enabled: bool = False,
) -> str:
    """Assemble the full system prompt from all layers.

    CACHING CONTRACT: The result is cached after the first call.
    Subsequent calls return the cached value WITHOUT re-evaluating
    the parameters. To force a rebuild with new parameters, call
    invalidate() first.

    This is intentional: the system prompt is built once per session
    and only rebuilt after compression events. Prefix caching depends
    on prompt stability across turns.
    """
    if self._cached_prompt is not None:
        return self._cached_prompt
    # ... assembly logic ...
    self._cached_prompt = result
    return result
```

This makes the caching contract explicit rather than implicit. The method now returns the cache on subsequent calls (matching current behavior), and the docstring warns callers.

---

## B3: CompressionManager Owns the Gate but Not the Threshold (◊ Paradox-Hunt)

### The contradiction

Section 3.4 defines `CompressionManager` with a `compression_enabled: bool` flag and a `should_compress()` method. Section 6.2 shows `compression_enabled` as owned by AIAgent (set from env var in `__init__`). But `run_conversation` (L2313) currently checks:

```python
if self.compression_enabled and self.context_compressor.should_compress():
```

The spec moves `compression_enabled` into `CompressionManager` and wraps the check:

```python
def should_compress(self) -> bool:
    return self._compression_enabled and self._compressor.should_compress()
```

**The paradox:** `CompressionManager` now gates compression, but `context_compressor.update_from_response()` -- which tracks the token counts that `should_compress()` reads -- is called from `run_conversation` at L2003-2009, **outside** CompressionManager. The token tracking (input to the decision) and the compression gate (the decision) are in different objects with no enforced call ordering.

If `update_from_response` is forgotten or called after `should_compress()`, the gate will use stale token data. Currently this works because the call order is enforced by `run_conversation`'s sequential code. But the decomposition creates two objects that must be called in order by a third (AIAgent), with no compile-time or interface-level enforcement.

### Mitigation (inline in spec Section 3.4)

Add `update_from_response` as a pass-through on CompressionManager, establishing it as the single entry point for compression-related state:

```python
class CompressionManager:
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Forward API response token usage to the underlying compressor.
        
        Must be called after every API response, before should_compress().
        This ensures the compressor has current token counts for its
        threshold decision.
        """
        self._compressor.update_from_response(usage)

    def should_compress(self) -> bool:
        """Check if compression is needed. 
        
        Precondition: update_from_response() was called with the latest
        API response usage data.
        """
        return self._compression_enabled and self._compressor.should_compress()
```

Update Section 6.2 (Shared State table): Change `context_compressor` row to note that AIAgent calls `compression_manager.update_from_response()` (not `compressor.update_from_response()` directly).

Add to Section 3.4 tests:

> - `update_from_response()` forwards to underlying compressor
> - `should_compress()` returns False when `update_from_response()` has not been called (stale state)
> - Calling sequence: `update_from_response() -> should_compress() -> compress()` is the only valid order

---

## B4: ToolExecutor Conflates Dispatch Routing with UI/Display (⊘ Level-Split)

### The split the spec acknowledges but does not resolve

Section 6.6 explicitly discusses this: "_execute_tool_calls currently contains ~100 lines of spinner and display logic... Decision: Keep display logic in ToolExecutor."

The rationale is that "the spinner management IS part of tool execution UX." But this conflates two distinct levels:

1. **Dispatch level:** "Which handler do I call with what arguments, and what do I do with the result?" (~50 lines of actual logic)
2. **Presentation level:** "What emoji, spinner, and cute message do I show?" (~100 lines of UI chrome, including a 25-entry emoji map hardcoded inline)

The spec is correct that separating these fully would add callback complexity. But the current design means ToolExecutor is not independently testable for dispatch correctness without either (a) mocking all the display functions, or (b) accepting side effects on stdout. This directly undermines Success Criterion #4: "Each extracted class is independently testable."

### Mitigation (inline in spec Section 3.2)

Add a `display_callback` protocol instead of hardcoding display logic:

```python
class ToolExecutor:
    def __init__(
        self,
        *,
        # ... existing params ...
        display: Optional["ToolDisplay"] = None,  # None = no UI output
    ):
        self._display = display

class ToolDisplay(Protocol):
    """Protocol for tool execution display. Implement for CLI, gateway, or test."""
    def on_tool_start(self, tool_name: str, args: dict, index: int, total: int) -> None: ...
    def on_tool_complete(self, tool_name: str, args: dict, duration: float, result: str) -> None: ...
    def on_tool_skipped(self, tool_name: str, reason: str) -> None: ...
```

**Why this is better than the current plan:**
- Tests can pass `display=None` and get pure dispatch testing with zero stdout side effects.
- The 100 lines of spinner/emoji logic move to a `CLIToolDisplay` implementation.
- The gateway can provide its own `ToolDisplay` (or `None` for silent operation).
- ToolExecutor's `execute()` method shrinks from ~200 lines to ~80 lines of pure dispatch.

**Phase impact:** This adds a file (`agent/tool_display.py`) to Phase 1 and slightly changes the wiring in Phase 4. No impact on external callers (AIAgent constructs the display object internally).

---

## B5: `_AGENT_LOOP_TOOLS` in model_tools.py Becomes Orphaned (◊ Paradox-Hunt)

### The contradiction

The spec says (Section 6.3): "No breaking changes to external callers." It also says (Section 3.2): ToolExecutor handles agent-loop tools. But `model_tools._AGENT_LOOP_TOOLS` is a **guard** that prevents agent-loop tools from being dispatched through the registry if they accidentally reach `handle_function_call`. 

Post-decomposition, the dispatch routing lives in `ToolExecutor.execute()`. But `handle_function_call` is still called by external code -- and still needs the guard. The spec proposes no changes to `handle_function_call` itself.

**The paradox:** If we define the canonical agent-loop tool set in ToolExecutor (per B1 mitigation) and model_tools imports it, we create a new dependency: `model_tools.py -> agent/tool_executor.py`. This is the REVERSE direction of the current `run_agent.py -> model_tools.py` dependency. Is this circular?

- `run_agent.py` imports from `model_tools.py` (L66)
- `run_agent.py` would import from `agent/tool_executor.py` (new, Phase 4)
- `model_tools.py` would import from `agent/tool_executor.py` (new, per B1)

No cycle: `agent/tool_executor.py` does NOT import from `model_tools.py` or `run_agent.py`. The dependency graph is a DAG.

### Mitigation (inline in spec Section 4)

Add explicit note:

> **Dependency direction after decomposition:**
> ```
> run_agent.py ──> model_tools.py ──> agent/tool_executor.py (AGENT_LOOP_TOOLS)
>      |                                      ^
>      └──────────────────────────────────────┘ (ToolExecutor class)
> ```
> Both `model_tools.py` and `run_agent.py` depend on `agent/tool_executor.py`. No cycle.
> The `_AGENT_LOOP_TOOLS` guard in `handle_function_call` MUST be updated to import
> from `ToolExecutor.AGENT_LOOP_TOOLS` in Phase 2 (alongside the circular import fix).

---

## B6: session_id Ownership Is a Three-Body Problem (⊘ Level-Split)

### The split

The spec identifies (Section 6.2) that `session_id` changes during compression and proposes:

> "Make AIAgent.session_id a property that reads from self._persister.session_id."

But three objects need to know the current session_id:

1. **SessionPersister** -- owns it (per spec), has a setter that updates `_session_log_file`
2. **AIAgent** -- needs it for `delegate_tool.py` access to `self.session_id`, and for passing to `SessionDB.create_session()`
3. **CompressionManager** -- needs it during `split_session` callback

The spec proposes the property-on-AIAgent approach, which creates a read-through chain. But there is a subtlety: during `compress()`, the `split_session` callback updates `SessionPersister.session_id`. If AIAgent's property reads from the persister, then any code in `compress()` that reads `AIAgent.session_id` after `split_session()` will see the NEW id. This is correct for the current code (L1383: `old_session_id = self.session_id` is read BEFORE the mutation). But it depends on call ordering within the callback.

**The level confusion:** `session_id` is treated as a simple field, but it is actually an **identity token** with ownership transfer semantics during compression. The spec does not distinguish between "which session am I logging to" (persister's concern) and "what is my externally visible identity" (AIAgent's concern).

### Mitigation (inline in spec Section 6.2)

Clarify the ownership model explicitly:

> **session_id ownership protocol:**
> - `SessionPersister` is the SOLE OWNER of `session_id`.
> - `AIAgent.session_id` is a read-only property delegating to `self._persister.session_id`.
> - Compression's `split_session` callback returns the new session_id AND updates `SessionPersister.session_id` atomically (inside `create_compression_session()`).
> - External callers (delegate_tool) read `agent.session_id` which always reflects the persister's current value.
>
> **Invariant:** No code path should set `session_id` directly on AIAgent. The `session_id` setter exists only on `SessionPersister`.

Add to Section 3.3 `create_compression_session`:

```python
def create_compression_session(self, *, platform: str, model: str,
                                parent_session_id: str) -> str:
    """End current session and create a new one after compression.
    
    ATOMICITY: This method updates self.session_id (and _session_log_file)
    before returning. Callers must NOT read session_id between ending the
    old session and calling this method.
    
    Returns the new session_id.
    """
```

---

## B7: flush_memories Needs the API Client but CompressionManager Has No Reference (◊ Paradox-Hunt)

### The contradiction

The spec's `CompressionManager.compress()` takes `flush_memories: Callable[[List[Dict], int], None]` as a callback. Looking at the actual `flush_memories` method (L1262-1359), it:

1. Injects a flush message into `messages`
2. Builds `api_messages` from `messages`
3. Prepends `self._cached_system_prompt`
4. Finds the memory tool definition from `self.tools`
5. **Makes an API call** via `self.client.chat.completions.create()`
6. Executes memory tool calls from the response
7. Strips flush artifacts from `messages`

Steps 3-6 require `self.client`, `self._cached_system_prompt`, `self.tools`, and `self.model` -- all AIAgent-owned. The callback pattern handles this correctly (the callback is a bound method on AIAgent, which closes over all these references).

**But the spec's interface for the callback is `Callable[[List[Dict], int], None]`** -- suggesting it takes `(messages, min_turns)`. The actual method signature is `flush_memories(self, messages: list = None, min_turns: int = None)`. As a bound method, this becomes `(messages, min_turns)` which matches.

**The real paradox** is more subtle: `flush_memories` reads `self._cached_system_prompt` (L1308). But `CompressionManager.compress()` calls `invalidate_prompt()` as step 4 (after flush). If someone reorders the steps inside `compress()` so that `invalidate_prompt()` runs before `flush_memories()`, the flush would use `None` as the system prompt, breaking the API call.

The spec documents the ordering as "1. Pre-compression memory flush, 2. compress(), 3. todo snapshot, 4. invalidate + rebuild, 5. session split." This is correct. But nothing in the interface enforces this ordering -- a future maintainer could reorder steps 1 and 4 with no type error.

### Mitigation (inline in spec Section 3.4)

Add an explicit ordering comment in the `compress()` implementation contract:

```python
def compress(self, messages, system_message, *, ...):
    """Compress conversation context.

    ORDERING INVARIANT (do not reorder):
      1. flush_memories() -- MUST run BEFORE invalidate_prompt() because
         it reads the cached system prompt to construct its API call.
      2. compressor.compress() -- the actual summarization
      3. inject_todo_snapshot() -- append to compressed messages
      4. invalidate_prompt() + build_system_prompt() -- rebuild with fresh memory
      5. split_session() -- create new session record

    Steps 1 and 4 have a data dependency: flush reads the prompt that
    invalidate destroys. Reordering them causes a silent failure (API call
    with empty system prompt).
    """
```

Add to Section 3.4 tests:

> - Verify that `flush_memories` callback is invoked BEFORE `invalidate_prompt` callback (ordering test)
> - Verify that if `flush_memories` raises, compression still proceeds (error resilience)

---

## B8: Tool Delay and Nudge Counters Are Policy, Not Execution (⊘ Level-Split)

### The split

The spec places `tool_delay` in `ToolExecutor.__init__` and nudge counter resets in `_execute_tool_calls`. But these are **conversation policy** concerns, not **tool execution** concerns:

- `tool_delay` controls rate limiting between tool calls -- this is a conversation-level throttle, not a property of how a single tool executes.
- Nudge counter resets (`_turns_since_memory = 0` on memory tool use, `_iters_since_skill = 0` on skill_manage use) are conversation-level counters that influence the next user message -- they have nothing to do with the tool's execution.

The spec acknowledges this in Section 6.5 ("Fix: ToolExecutor needs to expose a callback... `on_tool_executed(tool_name)`") but does not add this callback to the ToolExecutor interface in Section 3.2.

### Mitigation (inline in spec Section 3.2)

Add to the `execute()` signature:

```python
def execute(
    self,
    assistant_message,
    messages: List[Dict],
    effective_task_id: str,
    *,
    is_interrupted: Callable[[], bool],
    log_msg_to_db: Callable[[Dict], None],
    on_tool_executed: Callable[[str], None],  # NEW: called with tool_name after each tool
    parent_agent: Optional[Any] = None,
) -> None:
```

Add to Section 3.2 tests:

> - `on_tool_executed` is called with the tool name after each successful tool execution
> - `on_tool_executed` is NOT called for cancelled/skipped tools

---

## B9: Gutting tools/__init__.py Breaks 90% Coverage Metric (◊ Paradox-Hunt)

### The contradiction

Coverage-gaps.md shows `tools/__init__.py` at 90% coverage. The spec proposes gutting it to an empty docstring (Section 5). This creates a paradox:

- If we gut the file, the 90% "coverage" that comes from import-time side effects disappears.
- The coverage metric was already misleading (it measured "imports succeed", not "behavior is tested").
- But removing 90%-covered code and replacing it with 0-statement code makes the aggregate coverage numbers look like a regression.

This is not a real quality risk (the coverage was hollow), but it will confuse CI dashboards and automated coverage gates if they exist.

### Mitigation (inline in spec Section 5)

Add note:

> **Coverage impact:** tools/__init__.py drops from 90% (20 stmts, 18 covered) to N/A (0 statements). This is expected -- the prior coverage was from import side effects, not behavioral tests. The actual tool registration coverage moves to `model_tools._discover_tools()` (already at 94% coverage) and individual tool module tests.
>
> If CI enforces a coverage floor, add `tools/__init__.py` to the coverage exclusion list with a comment explaining why.

---

## B10: _build_api_kwargs and _handle_max_iterations Duplicate Provider Logic (⊘ Level-Split)

### The split the spec does not address

`_build_api_kwargs` (L1173-1219) constructs the API kwargs including reasoning config and provider preferences. `_handle_max_iterations` (L1605-1665) constructs a SEPARATE set of API kwargs with duplicated reasoning/provider logic (L1628-1649). This duplication is not mentioned in the spec.

Both methods check `"openrouter" in self.base_url.lower()` and `"nousresearch" in self.base_url.lower()`, and both construct the same `reasoning` extra_body. The summary request in `_handle_max_iterations` deliberately omits `tools` (no tools for the summary call), which is the only intentional difference.

**The level confusion:** "How to construct API kwargs" is one concern. "What parameters differ for a summary call" is a separate concern. Currently these are conflated into two methods that copy-paste the provider detection logic.

This is outside the scope of the four extracted classes, but it is tech debt that the decomposition should at least document.

### Mitigation (inline in spec Section 7, Phase 4 notes)

Add:

> **Tech debt noted (not in scope for this decomposition):** `_build_api_kwargs` and `_handle_max_iterations` duplicate provider detection and reasoning config logic. A future cleanup should extract provider detection into a shared helper. This is not blocking for the current refactor because both methods remain on AIAgent.

---

## Brenner Pass 1 Verdict

The spec is **structurally sound** for its primary goal (decompose AIAgent into composed components with unchanged external API). The findings above are epistemological gaps -- places where the spec's model of the system does not match the system's actual structure, or where the decomposition introduces new ambiguities.

**Must-fix before implementation (HIGH severity):**
- B1: Unify agent-loop tool set between ToolExecutor and model_tools.py
- B5: Ensure `_AGENT_LOOP_TOOLS` guard in `handle_function_call` imports from ToolExecutor

**Should-fix (MEDIUM severity):**
- B2: Make PromptAssembler caching contract explicit
- B3: Route `update_from_response` through CompressionManager
- B4: Add ToolDisplay protocol to make ToolExecutor independently testable
- B6: Formalize session_id ownership protocol

**Nice-to-have (LOW severity):**
- B7: Document ordering invariant in CompressionManager.compress()
- B8: Add `on_tool_executed` callback to ToolExecutor interface
- B9: Document coverage impact of gutting tools/__init__.py
- B10: Note duplicated provider logic as tech debt
