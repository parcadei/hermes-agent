# Brenner Pass 3: Exclusion Tests and Object Transposition

Created: 2026-02-26
Author: architect-agent (adversarial review)
Status: Completed

---

## Operators Applied

1. **Exclusion-Test** -- What forbidden pattern would prove the design is wrong? What test would delete the entire approach?
2. **Object-Transpose** -- Is there a cheaper system/representation that makes the decisive test trivial? Are we testing in the hardest possible coordinates?

---

## 1. Exclusion-Test: What Would KILL the 4-Class Decomposition?

### 1.1 The Parameter-Explosion Exclusion Test

**Hypothesis:** "If extracting methods into classes creates more parameter surface (constructor params + method params + callbacks) than the original method had self.X field accesses, the decomposition is objectively worse -- it turns implicit coupling (self.X) into explicit coupling (function arguments) without reducing total coupling."

**Empirical result (VERIFIED via AST analysis of run_agent.py):**

| Method / Class | self.X accesses (current) | Total params in spec (init + method + callbacks) | Delta |
|----------------|--------------------------|--------------------------------------------------|-------|
| `_execute_tool_calls` / ToolExecutor | 15 fields | 10 (init) + 6 (execute) + 1 (on_tool_executed, missing from spec) = 17 | **+2 worse** |
| `_build_system_prompt` / PromptAssembler | 6 fields | 2 (init) + 5 (build) = 7 | **+1 worse** |
| `_compress_context` / CompressionManager | 10 fields | 2 (init) + 8 (compress, 5 of which are callbacks) = 10 | **0 neutral** |
| Persistence methods / SessionPersister | 9 fields (save_session_log) | 10 (init) = 10 | **+1 worse** |

**Verdict:** The decomposition does NOT reduce coupling -- it redistributes it. ToolExecutor and CompressionManager together introduce **7 callback parameters** where the current code has zero. The spec trades `self._interrupt_requested` (one field read) for `is_interrupted: Callable[[], bool]` (one callback allocation + one indirection per call). This is a lateral move, not an improvement.

**However, this test does NOT kill the approach outright.** The parameter counts are approximately equal (+1/+2 is noise). The real value proposition of the decomposition is **testability** (can instantiate ToolExecutor without a full AIAgent), not coupling reduction. The exclusion test weakens the coupling-reduction argument but does not falsify the testability argument.

**Mitigation added:** The spec should explicitly state that the primary value is testability and cognitive chunking, NOT coupling reduction. Any claim that "we reduce coupling" should be struck from the rationale.

### 1.2 The Shared-Mutable-State Exclusion Test

**Hypothesis:** "If the extracted classes need to mutate state owned by other classes (or by AIAgent) through anything other than their return values, the decomposition creates action-at-a-distance that is harder to reason about than the original god object."

**Empirical result (VERIFIED from source):**

Shared mutable state interactions in the proposed design:

| Mutator | Mutated State | Owner | Mechanism |
|---------|--------------|-------|-----------|
| ToolExecutor.execute() | `messages` list | run_conversation (local var) | In-place append (by-reference) |
| ToolExecutor.execute() | nudge counters (`_turns_since_memory`, `_iters_since_skill`) | AIAgent | Spec proposes `on_tool_executed` callback -- NOT YET IN SPEC |
| CompressionManager.compress() | `messages` list | run_conversation (local var) | Via `flush_memories` callback which does in-place mutation |
| CompressionManager.compress() | `session_id` | SessionPersister | Via `split_session` callback |
| CompressionManager.compress() | `_cached_system_prompt` | PromptAssembler | Via `invalidate_prompt` callback |

The `messages` list mutation is fine -- it matches the current contract. But the **5 callback-mediated mutations** in CompressionManager.compress() are a code smell. The method takes 5 callbacks that each mutate state on different objects. A reader of CompressionManager.compress() cannot understand what it does without reading all 5 callback implementations. This is strictly harder to understand than the current 34-line `_compress_context` method where all mutations are visible.

**Verdict:** CompressionManager fails this test. A 34-line method that reads 10 self.X fields is already clean enough. Wrapping it in a class that takes 5 callbacks to do the same thing adds indirection without adding clarity. The spec acknowledges this ("rather than giving CompressionManager references to all these objects, we use callbacks") but frames it as a feature rather than a cost.

**Mitigation added:** CompressionManager should be DROPPED from the decomposition. The `_compress_context` method should remain on AIAgent. It is small (34 lines), its dependencies are already well-documented in the spec, and it coordinates operations that genuinely belong together. If testing is desired, test via integration tests on AIAgent rather than unit tests on a callback-threaded class.

### 1.3 The delegate_tool Exclusion Test

**Hypothesis:** "If ToolExecutor's `parent_agent` parameter for delegate_task is the full AIAgent, we haven't actually decoupled anything -- the god object is still needed."

**Empirical result (VERIFIED from tools/delegate_tool.py L93-132):**

`delegate_task` accesses on parent_agent:
- `parent_agent.base_url`
- `parent_agent._client_kwargs.get("api_key")`
- `parent_agent.model`
- `parent_agent.platform`
- `parent_agent._session_db`
- `parent_agent.providers_allowed`
- `parent_agent.providers_ignored`
- `parent_agent.providers_order`
- `parent_agent.provider_sort`
- `parent_agent._delegate_depth`
- `parent_agent._active_children`

That is 11 fields. delegate_tool constructs a full new AIAgent from these fields. There is no way to pass a "thin interface" -- it needs the real AIAgent to build a child AIAgent.

**Verdict:** This does NOT kill the decomposition because ToolExecutor correctly passes `parent_agent` through without calling methods on it. The spec explicitly says "ToolExecutor itself does not call methods on it." The coupling is between delegate_tool and AIAgent, not between ToolExecutor and AIAgent. This is acceptable.

**No mitigation needed** -- the spec already handles this correctly.

---

## 2. Object-Transpose: Is There a Cheaper Representation?

### 2.1 Module-Level Functions Instead of Classes

**The decisive question:** Of the 4 proposed classes, how many have meaningful mutable state?

| Proposed Class | Mutable State | Immutable Config |
|----------------|--------------|-----------------|
| PromptAssembler | `_cached_prompt` (1 field) | `_platform`, `_skip_context_files` (2 fields) |
| ToolExecutor | **NONE** | 10 constructor params |
| SessionPersister | `session_id`, `_session_log_file` (2 fields) | 8 constructor params |
| CompressionManager | **NONE** (delegates to ContextCompressor) | `compression_enabled` (1 field) |

**Key finding:** ToolExecutor and CompressionManager have ZERO mutable state of their own. They are stateless dispatch functions wrapped in class syntax. This is the classic "Kingdom of Nouns" anti-pattern where a function is forced into a class for no reason.

**Alternative: Module-level functions with a config dataclass.**

```python
# agent/tool_executor.py -- function-based alternative

@dataclass(frozen=True)
class ToolExecConfig:
    todo_store: TodoStore
    memory_store: Optional[MemoryStore]
    session_db: Optional[Any]
    clarify_callback: Optional[Callable]
    tool_delay: float
    tool_progress_callback: Optional[Callable]
    quiet_mode: bool
    verbose_logging: bool
    log_prefix: str
    log_prefix_chars: int

def execute_tool_calls(
    config: ToolExecConfig,
    assistant_message,
    messages: list,
    effective_task_id: str,
    *,
    is_interrupted: Callable[[], bool],
    log_msg_to_db: Callable[[dict], None],
    on_tool_executed: Callable[[str], None],
    parent_agent=None,
) -> None:
    ...
```

**Comparison:**

| Dimension | Class (spec) | Function + dataclass | Winner |
|-----------|-------------|---------------------|--------|
| Testability | Instantiate class, call method | Call function with config | Tie |
| Readability | 2 files to understand (class + AIAgent glue) | 1 file (function + config) | Function |
| Parameter surface | 10 init + 6 method = 16 | 1 config + 6 method = 7 logical groups | Function |
| State confusion | "Does ToolExecutor cache anything?" (reader must check) | Frozen dataclass -- clearly no state | Function |
| IDE navigation | Class methods show up in outline | Functions show up in outline | Tie |

**Verdict:** For ToolExecutor and CompressionManager, module-level functions with a config dataclass are strictly simpler. They make the "no mutable state" property self-evident via `frozen=True`.

**Mitigation added:** The spec should use module-level functions for ToolExecutor and CompressionManager. Classes should be reserved for components with genuine mutable state: PromptAssembler (cache) and SessionPersister (session_id lifecycle).

### 2.2 Two Classes Instead of Four

Given the transpose analysis, the decomposition collapses to:

| Component | Form | Justification |
|-----------|------|---------------|
| PromptAssembler | **Class** | Owns mutable cache. Already clean (6 fields). Worth extracting. |
| execute_tool_calls | **Function** | Zero state. 207 lines of sequential dispatch logic. Extract as function. |
| SessionPersister | **Class** | Owns mutable session_id. Manages file + DB lifecycle. Worth extracting. |
| compress_context | **Keep on AIAgent** | 34 lines. Orchestrates 3 existing methods. Not worth extracting at all. |

This gives 2 classes + 1 function extraction instead of 4 classes. The total complexity budget is:

- Current: 1 god object, 30+ methods, 51 fields
- Proposed (spec): 1 orchestrator + 4 composed classes. 5 objects to understand.
- Proposed (revised): 1 orchestrator + 2 classes + 1 module function. 4 objects to understand, and compress_context stays inline where it is already readable.

### 2.3 Is `run_conversation` the Real Problem?

The spec focuses on extracting responsibilities OUT of AIAgent. But `run_conversation` (826 lines, 50 field accesses) is the actual complexity center. The proposed decomposition does NOT shrink `run_conversation` -- it still contains:

- Retry loops (80+ lines)
- Response validation (60+ lines)
- Scratchpad handling (30+ lines)
- Invalid tool call recovery (50+ lines)
- Error handling with 7 different early-return paths

These stay on AIAgent because they are "conversation orchestration." But they are also the hardest-to-test code. The 4-class decomposition extracts the EASY parts (prompt assembly, tool dispatch, persistence) and leaves the HARD part (the 826-line state machine) untouched.

**This is the real exclusion test that Passes 1-2 would miss:**

The decomposition optimizes for the wrong thing. The methods being extracted (_build_system_prompt at 64 lines, _execute_tool_calls at 207 lines, persistence at 340 lines, compression at 34 lines) are already readable. The method that NEEDS decomposition -- run_conversation at 826 lines -- is untouched by this plan.

**Mitigation added:** The plan should acknowledge that run_conversation itself needs a separate Phase 6 decomposition. The current 5-phase plan should be positioned as "extracting leaf responsibilities" that enables a future "decompose the orchestration loop" phase. Without this framing, the plan solves the wrong problem.

### 2.4 The Callback Pattern vs. Passing Self

The spec proposes callbacks to avoid ToolExecutor needing a reference to AIAgent:
```python
is_interrupted: Callable[[], bool]
log_msg_to_db: Callable[[dict], None]
on_tool_executed: Callable[[str], None]
```

**Alternative: Just pass `self` (the AIAgent).**

```python
def execute_tool_calls(agent, assistant_message, messages, effective_task_id):
    if agent._interrupt_requested:
        ...
    agent._log_msg_to_db(tool_msg)
    ...
```

**Comparison:**

| Dimension | Callbacks | Pass agent | Winner |
|-----------|-----------|-----------|--------|
| Type safety | Callable signatures are explicit | Any attribute access possible | Callbacks |
| Testability | Mock individual callbacks | Must mock entire agent or use a protocol | Callbacks |
| Simplicity | 3 extra params, lambda/closure at call site | 1 param | Pass agent |
| Discoverability | "What callbacks does it need?" (read signature) | "What does it access on agent?" (read body) | Callbacks |

**Verdict:** Callbacks are better for testability, but they add ceremony. A middle ground is a Protocol:

```python
class AgentContext(Protocol):
    _interrupt_requested: bool
    def _log_msg_to_db(self, msg: dict) -> None: ...
```

This gives type safety and testability (can create a lightweight test double) without callback ceremony. AIAgent already satisfies this protocol implicitly.

**Mitigation added:** For the function-based ToolExecutor, use a Protocol for the 3 agent interactions instead of 3 separate callbacks. This reduces the parameter count from 7 to 5 (config + assistant_message + messages + task_id + agent_context).

---

## 3. Summary of Lethal Findings and Mitigations

### Finding 1: CompressionManager Should Be Dropped
**Operator:** Exclusion-Test (shared mutable state)
**Lethal test:** 5 callbacks threading mutations through a class with zero state is objectively worse than a 34-line method.
**Mitigation:** Keep `_compress_context` on AIAgent. It is already clean and well-documented.

### Finding 2: ToolExecutor and CompressionManager Should Be Functions, Not Classes
**Operator:** Object-Transpose (cheaper representation)
**Lethal test:** Both have zero mutable state. Class syntax suggests statefulness that does not exist.
**Mitigation:** Use module-level functions with frozen dataclass configs for ToolExecutor. Drop CompressionManager entirely.

### Finding 3: The Spec Misidentifies the Complexity Center
**Operator:** Object-Transpose (testing in the wrong coordinates)
**Lethal test:** `run_conversation` (826 lines, 50 field accesses) is untouched by the decomposition. The extracted methods are already the readable parts.
**Mitigation:** Add Phase 6 to the plan: decompose `run_conversation` into sub-state-machines (retry handler, response validator, tool-call-validator, orchestration loop). Position the current plan as "enabling infrastructure" for this future phase.

### Finding 4: Coupling Is Not Reduced, Only Redistributed
**Operator:** Exclusion-Test (parameter explosion)
**Lethal test:** Total parameter surface is equal or +1/+2 compared to self.X accesses.
**Mitigation:** Reframe the spec's value proposition. The win is testability and cognitive chunking, NOT coupling reduction. Strike any coupling-reduction claims.

### Finding 5: The `on_tool_executed` Callback Is Missing From the Spec
**Operator:** Exclusion-Test (completeness)
**Lethal test:** The spec's ToolExecutor has no mechanism to reset `_turns_since_memory` and `_iters_since_skill` counters. The nudge logic in `run_conversation` depends on these resets happening inside `_execute_tool_calls` (L1420-1423). Without this callback, the decomposition silently breaks nudge behavior.
**Mitigation:** The spec mentions this in section 6.5 but only as a "fix needed" note. This must be promoted to a first-class part of the ToolExecutor interface. Recommended: `on_tool_executed: Callable[[str], None]` parameter on execute(), or include it in the AgentContext protocol.

---

## 4. Revised Recommendation

The decomposition is worth doing, but the spec should be revised:

1. **Drop CompressionManager** -- keep `_compress_context` on AIAgent
2. **Make ToolExecutor a function** -- `execute_tool_calls(config, ...)` with a frozen dataclass
3. **Keep PromptAssembler as a class** -- it owns the cache lifecycle
4. **Keep SessionPersister as a class** -- it owns the session_id lifecycle
5. **Add Phase 6** -- decompose `run_conversation` into testable sub-handlers
6. **Add on_tool_executed** -- promote from footnote to interface contract
7. **Reframe value proposition** -- testability and cognitive chunking, not coupling reduction

This gives a 2-class + 1-function decomposition that is strictly simpler than the 4-class proposal while achieving the same testability goals.
