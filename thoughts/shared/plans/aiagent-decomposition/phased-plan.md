# AIAgent Decomposition — Phased Implementation Plan

Created: 2026-02-26
Status: Phase 4 — PLAN (ready for TDD build loop)
Incorporates: spec.md + brenner-pass1/2/3 + ck-pass4 + Phase 3.25 discriminative validation

---

## Phase Order

| # | Phase | Files Created/Modified | Risk | Tests First? |
|---|-------|----------------------|------|-------------|
| 1 | Circular import fix | `agent/async_bridge.py` (new), `model_tools.py`, `tools/registry.py` | LOW | YES — structural assertion |
| 2 | Eager import fix | `tools/__init__.py`, `tools/file_tools.py` | LOW | YES — import test |
| 3 | PromptAssembler extraction | `agent/prompt_assembler.py` (new) | MEDIUM | YES — unit tests |
| 4 | SessionPersister extraction | `agent/session_persister.py` (new) | MEDIUM | YES — unit tests |
| 5 | execute_tool_calls extraction | `agent/tool_executor.py` (new) | HIGH | YES — unit tests |
| 6 | Wire into AIAgent | `run_agent.py` | HIGH | NO — validate via existing 172 tests |
| 7 | Integration validation | All | LOW | Run full suite + smoke test |

---

## Phase 1: Circular Import Fix

**Goal:** Break `model_tools.py` <-> `tools/registry.py` cycle by extracting `_run_async` to shared module.

### 1.1 Write test first

**File:** `tests/test_import_structure.py`

```python
"""Structural import tests — no deferred imports in dispatch methods."""
import ast
import inspect

def test_no_deferred_imports_in_registry_dispatch():
    """tools/registry.py dispatch() must not contain inline imports.

    The circular import between model_tools.py and tools/registry.py
    was broken by extracting _run_async to agent/async_bridge.py.
    This test ensures it stays broken.
    """
    from tools.registry import ToolRegistry
    source = inspect.getsource(ToolRegistry.dispatch)
    tree = ast.parse(source)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 0, (
        f"dispatch() has deferred imports that may create circular dependencies: "
        f"{[ast.dump(n) for n in imports]}"
    )

def test_async_bridge_importable():
    """agent/async_bridge.py is independently importable."""
    from agent.async_bridge import run_async
    assert callable(run_async)

def test_model_tools_importable():
    """model_tools.py imports without circular dependency errors."""
    import model_tools
    assert hasattr(model_tools, 'handle_function_call')

def test_registry_importable():
    """tools/registry.py imports without circular dependency errors."""
    from tools.registry import registry
    assert registry is not None
```

### 1.2 Create `agent/async_bridge.py`

```python
"""Sync-to-async bridge for tool handlers.

Single source of truth for running async coroutines from synchronous code.
Extracted from model_tools.py to break the circular import between
model_tools.py and tools/registry.py.

Dependency direction:
    model_tools.py ──> agent/async_bridge.py <── tools/registry.py
    (No cycle. Both depend on this module; this module depends on neither.)
"""

import asyncio


def run_async(coro):
    """Run an async coroutine from a sync context.

    If the current thread already has a running event loop (e.g., inside
    the gateway's async stack or Atropos's event loop), we spin up a
    disposable thread so asyncio.run() can create its own loop without
    conflicting.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=300)
    return asyncio.run(coro)
```

### 1.3 Modify `model_tools.py`

- Delete `_run_async` function (L39-62)
- Add: `from agent.async_bridge import run_async as _run_async`
- Keep `_run_async` name for backward compat within the file

### 1.4 Modify `tools/registry.py`

- In `dispatch()` method (L124), replace:
  `from model_tools import _run_async`
  with top-level import:
  `from agent.async_bridge import run_async`
- Update the call site: `run_async(entry.handler(...))` instead of `_run_async(...)`

### 1.5 Acceptance

- `test_import_structure.py` passes (all 4 tests)
- All 172 existing tests still pass
- No deferred imports in `registry.py` dispatch method

---

## Phase 2: Eager Import Fix

**Goal:** Gut `tools/__init__.py` so `import tools` works without optional dependencies.

### 2.1 Write test first

**File:** `tests/test_lazy_tools_init.py`

```python
"""Verify tools/__init__.py is clean — no eager imports of tool modules."""
import ast

def test_tools_init_has_no_tool_imports():
    """tools/__init__.py must not eagerly import tool modules.

    Tool discovery is handled by model_tools._discover_tools() which uses
    importlib.import_module with try/except for graceful degradation.
    The __init__.py eager imports are redundant and cause ImportError
    when optional dependencies (firecrawl, fal_client) aren't installed.
    """
    with open("tools/__init__.py") as f:
        source = f.read()
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(('.', 'tools.')):
                # Relative or tools.X imports — these are the eager ones
                # Allow only non-tool utility imports if any
                imports.append(f"from {node.module or '.'} import ...")
    assert len(imports) == 0, (
        f"tools/__init__.py has eager tool imports that should be removed: {imports}"
    )

def test_tools_package_importable():
    """import tools succeeds even without optional dependencies."""
    import tools
    assert tools.__doc__ is not None

def test_tool_discovery_still_works():
    """model_tools._discover_tools() still discovers tools after __init__.py cleanup."""
    from model_tools import get_all_tool_names
    names = get_all_tool_names()
    assert len(names) > 10, f"Expected 10+ tools, got {len(names)}"
    assert "terminal" in names or any("terminal" in n for n in names)
```

### 2.2 Modify `tools/__init__.py`

Replace entire file with:

```python
"""Tools Package

Individual tool implementations for the Hermes Agent.
Import from specific submodules (e.g., ``from tools.terminal_tool import ...``).

Tool discovery is handled by model_tools._discover_tools() which uses
importlib.import_module with try/except for graceful degradation.
"""
```

### 2.3 Modify `tools/file_tools.py`

Replace `_check_file_reqs()` (L195-198) — inline the function:

```python
def _check_file_reqs():
    """Check if file tool requirements (terminal backend) are available."""
    from tools.terminal_tool import check_terminal_requirements
    return check_terminal_requirements()
```

This removes the dependency on `tools.__init__.check_file_requirements`.

### 2.4 Acceptance

- `test_lazy_tools_init.py` passes (all 3 tests)
- All 172 existing tests still pass
- `python -c "import tools"` works

---

## Phase 3: PromptAssembler Extraction

**Goal:** Extract `_build_system_prompt` and `_invalidate_system_prompt` into `agent/prompt_assembler.py`.

### 3.1 Write tests first

**File:** `tests/agent/test_prompt_assembler.py`

Key test cases:
1. `test_build_returns_string` — basic assembly
2. `test_layer_ordering_identity_first` — `DEFAULT_AGENT_IDENTITY` is first segment
3. `test_layer_ordering_platform_last` — platform hint is last segment (when present)
4. `test_layer_relative_ordering` — tool guidance before system_message before memory
5. `test_tool_guidance_gated_on_tool_names` — memory guidance only when "memory" in valid_tool_names
6. `test_memory_blocks_gated_on_flags` — memory_enabled=False skips memory blocks
7. `test_context_files_skipped` — skip_context_files=True omits context files
8. `test_caching_returns_same_result` — second build() call returns cached value
9. `test_caching_ignores_new_params` — second build() with different params still returns cached
10. `test_invalidate_clears_cache` — after invalidate(), cached property is None
11. `test_invalidate_reloads_memory` — invalidate(memory_store) calls load_from_disk()
12. `test_ephemeral_prompt_not_included` — ephemeral system prompt is NOT in the output

### 3.2 Create `agent/prompt_assembler.py`

Interface (from spec, incorporating B2 caching contract):

```python
class PromptAssembler:
    def __init__(self, *, platform=None, skip_context_files=False):
        self._platform = platform
        self._skip_context_files = skip_context_files
        self._cached_prompt = None

    def build(self, *, valid_tool_names, system_message=None,
              memory_store=None, memory_enabled=False,
              user_profile_enabled=False) -> str:
        """CACHING CONTRACT: returns cached value on subsequent calls.
        Call invalidate() before build() to force a rebuild."""
        if self._cached_prompt is not None:
            return self._cached_prompt
        # ... assembly logic extracted from _build_system_prompt ...
        self._cached_prompt = result
        return result

    @property
    def cached(self):
        return self._cached_prompt

    def invalidate(self, memory_store=None):
        self._cached_prompt = None
        if memory_store is not None:
            memory_store.load_from_disk()
```

### 3.3 Implementation

Extract the body of `AIAgent._build_system_prompt` (L1068-1123) into `PromptAssembler.build()`. The imports needed:
- `from agent.prompt_builder import DEFAULT_AGENT_IDENTITY, MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE, PLATFORM_HINTS, build_skills_system_prompt, build_context_files_prompt`
- `from datetime import datetime`

### 3.4 Acceptance

- `test_prompt_assembler.py` passes (all 12 tests)
- PromptAssembler is independently importable: `from agent.prompt_assembler import PromptAssembler`
- No imports from `run_agent.py`

---

## Phase 4: SessionPersister Extraction

**Goal:** Extract session persistence methods into `agent/session_persister.py`.

### 4.1 Write tests first

**File:** `tests/agent/test_session_persister.py`

Key test cases:
1. `test_log_message_noop_without_db` — log_message is no-op when session_db is None
2. `test_log_message_calls_append` — log_message calls session_db.append_message
3. `test_persist_writes_json_and_db` — persist() writes JSON log + flushes to DB
4. `test_save_session_log_overwrites` — session log is overwritten (not appended)
5. `test_save_session_log_metadata` — JSON includes correct session_id, model, timestamps
6. `test_flush_to_db_skips_logged` — start_idx calculation skips already-logged messages
7. `test_save_trajectory_noop_when_disabled` — save_trajectories=False is no-op
8. `test_session_id_setter_updates_log_file` — setter atomically updates both fields (FIXES BUG)
9. `test_create_compression_session` — ends old session, creates new, returns new ID
10. `test_create_compression_session_updates_session_id` — session_id property reflects new ID after compression
11. `test_maybe_save_session_log_interval` — with save_interval=3, only writes every 3rd call
12. `test_content_scratchpad_to_think_conversion` — assistant content scratchpad tags converted

### 4.2 Create `agent/session_persister.py`

Interface (from spec, incorporating CK U-1 save_interval, B6 session_id ownership):

```python
class SessionPersister:
    def __init__(self, *, session_id, session_db=None, logs_dir, model, base_url,
                 platform=None, session_start, save_trajectories=False,
                 tools=None, verbose_logging=False, quiet_mode=False,
                 save_interval=1):
        ...
        self._session_log_file = logs_dir / f"session_{session_id}.json"

    @property
    def session_id(self): ...

    @session_id.setter
    def session_id(self, value):
        self._session_id = value
        self._session_log_file = self._logs_dir / f"session_{value}.json"

    def log_message(self, msg): ...
    def persist(self, messages, conversation_history=None): ...
    def save_session_log(self, messages): ...
    def maybe_save_session_log(self, messages): ...
    def flush_to_db(self, messages, conversation_history=None): ...
    def save_trajectory(self, messages, user_query, completed): ...
    def create_compression_session(self, *, platform, model, parent_session_id): ...
    def update_system_prompt(self, prompt): ...
```

### 4.3 Implementation

Extract from `run_agent.py`:
- `_persist_session` (body) -> `persist()`
- `_save_session_log` (L937-978) -> `save_session_log()`
- `_log_msg_to_db` -> `log_message()`
- `_flush_messages_to_session_db` -> `flush_to_db()`
- `_save_trajectory` + `_convert_to_trajectory_format` -> `save_trajectory()` (includes 157-line conversion, per SC-2)
- `_clean_session_content` -> internal helper
- Compression session split from `_compress_context` L1380-1393 -> `create_compression_session()`

### 4.4 Acceptance

- `test_session_persister.py` passes (all 12 tests)
- SessionPersister is independently importable
- session_id setter atomically updates both fields (bug fix verified by test 8)

---

## Phase 5: execute_tool_calls Extraction

**Goal:** Extract `_execute_tool_calls` into module-level function with frozen dataclass config.

### 5.1 Write tests first

**File:** `tests/agent/test_tool_executor.py`

Key test cases:
1. `test_agent_loop_tools_not_through_registry` — todo, memory, session_search, clarify, delegate_task bypass handle_function_call
2. `test_agent_loop_tools_canonical_set` — AGENT_LOOP_TOOLS has exactly 5 members including clarify
3. `test_other_tools_through_registry` — non-agent-loop tools go through handle_function_call
4. `test_every_tool_call_gets_response` — every tool_call.id gets exactly one tool response
5. `test_interrupt_before_first_skips_all` — all calls get cancel messages
6. `test_interrupt_after_nth_skips_remaining` — remaining calls get cancel messages
7. `test_result_truncation` — truncates at MAX_TOOL_RESULT_CHARS (100,000)
8. `test_invalid_json_args_fallback` — bad JSON gets {} fallback
9. `test_tool_delay_between_calls` — delay applied between calls, not after last
10. `test_log_msg_callback_called` — log_msg_to_db called for every appended message
11. `test_on_tool_executed_callback` — called with tool name after each successful tool
12. `test_on_tool_executed_not_called_for_skipped` — NOT called for cancelled/skipped tools
13. `test_delegate_task_none_guard` — parent_agent=None returns error JSON (CK G-2)
14. `test_config_is_frozen` — ToolExecConfig is immutable

### 5.2 Create `agent/tool_executor.py`

Interface (from spec + Brenner Pass 3 revision):

```python
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

AGENT_LOOP_TOOLS: frozenset = frozenset({
    "todo", "memory", "session_search", "clarify", "delegate_task"
})

@dataclass(frozen=True)
class ToolExecConfig:
    todo_store: Any  # TodoStore
    memory_store: Any  # Optional[MemoryStore]
    session_db: Any  # Optional[SessionDB]
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
    messages: List[Dict],
    effective_task_id: str,
    *,
    is_interrupted: Callable[[], bool],
    log_msg_to_db: Callable[[Dict], None],
    on_tool_executed: Callable[[str], None],
    parent_agent: Optional[Any] = None,
) -> None:
    """Execute all tool calls from an assistant message.

    Appends tool result messages to `messages`.
    Checks `is_interrupted()` before each tool call.
    Calls `log_msg_to_db(msg)` after each appended message.
    Calls `on_tool_executed(tool_name)` after each successful tool.
    """
```

### 5.3 Implementation

Extract the body of `AIAgent._execute_tool_calls` (L1397-1604). Key changes:
- Replace `self.X` field reads with `config.X`
- Replace `self._interrupt_requested` with `is_interrupted()` call
- Replace `self._log_msg_to_db(msg)` with `log_msg_to_db(msg)` callback
- Add `on_tool_executed(function_name)` call after each successful tool execution
- Add None guard for `parent_agent` on delegate_task branch
- Make `_delegate_spinner` a local variable (not stored on self)

### 5.4 Update `model_tools.py`

Change `_AGENT_LOOP_TOOLS` (L240) to import from tool_executor:

```python
from agent.tool_executor import AGENT_LOOP_TOOLS as _AGENT_LOOP_TOOLS
```

### 5.5 Acceptance

- `test_tool_executor.py` passes (all 14 tests)
- `AGENT_LOOP_TOOLS` is single source of truth (model_tools imports from tool_executor)
- `clarify` is in `AGENT_LOOP_TOOLS` (B1/SC-7 fix)
- No imports from `run_agent.py`

---

## Phase 6: Wire Into AIAgent

**Goal:** Compose extracted components into AIAgent, replacing the original methods.

### 6.1 Changes to `run_agent.py`

**In `__init__`:**
```python
from agent.prompt_assembler import PromptAssembler
from agent.session_persister import SessionPersister
from agent.tool_executor import ToolExecConfig, execute_tool_calls

# Create composed components
self._prompt_assembler = PromptAssembler(
    platform=self.platform,
    skip_context_files=self.skip_context_files,
)

self._persister = SessionPersister(
    session_id=self.session_id,
    session_db=self._session_db,
    logs_dir=self.logs_dir,
    model=self.model,
    base_url=self.base_url,
    platform=self.platform,
    session_start=self.session_start,
    save_trajectories=self.save_trajectories,
    tools=self.tools,
    verbose_logging=self.verbose_logging,
    quiet_mode=self.quiet_mode,
)

self._tool_exec_config = ToolExecConfig(
    todo_store=self._todo_store,
    memory_store=self._memory_store,
    session_db=self._session_db,
    clarify_callback=self.clarify_callback,
    tool_delay=self.tool_delay,
    tool_progress_callback=self.tool_progress_callback,
    quiet_mode=self.quiet_mode,
    verbose_logging=self.verbose_logging,
    log_prefix=self.log_prefix,
    log_prefix_chars=self.log_prefix_chars,
)
```

**Replace `session_id` with property:**
```python
@property
def session_id(self):
    return self._persister.session_id

@session_id.setter
def session_id(self, value):
    self._persister.session_id = value
```

**Replace method calls in `run_conversation`:**
- `self._build_system_prompt(system_message)` -> `self._prompt_assembler.build(valid_tool_names=..., system_message=..., memory_store=..., memory_enabled=..., user_profile_enabled=...)`
- `self._cached_system_prompt` reads -> `self._prompt_assembler.cached`
- `self._invalidate_system_prompt()` -> `self._prompt_assembler.invalidate(memory_store=self._memory_store)`
- `self._execute_tool_calls(msg, messages, task_id)` -> `execute_tool_calls(self._tool_exec_config, msg, messages, task_id, is_interrupted=lambda: self._interrupt_requested, log_msg_to_db=self._persister.log_message, on_tool_executed=self._on_tool_executed, parent_agent=self)`
- `self._persist_session(messages, history)` -> `self._persister.persist(messages, history)`
- `self._log_msg_to_db(msg)` -> `self._persister.log_message(msg)`
- `self._save_session_log(messages)` -> `self._persister.maybe_save_session_log(messages)`
- Session split in `_compress_context` -> `self._persister.create_compression_session(...)`

**Add nudge counter callback:**
```python
def _on_tool_executed(self, tool_name: str):
    if tool_name == "memory":
        self._turns_since_memory = 0
    elif tool_name == "skill_manage":
        self._iters_since_skill = 0
```

**Delete from `run_agent.py` (methods now on extracted components):**
- `_build_system_prompt` (body moves to PromptAssembler)
- `_invalidate_system_prompt` (replaced by PromptAssembler.invalidate)
- `_execute_tool_calls` (body moves to execute_tool_calls function)
- `_persist_session` (body moves to SessionPersister.persist)
- `_save_session_log` (body moves to SessionPersister.save_session_log)
- `_log_msg_to_db` (body moves to SessionPersister.log_message)
- `_flush_messages_to_session_db` (body moves to SessionPersister.flush_to_db)
- `_save_trajectory` (body moves to SessionPersister.save_trajectory)
- `_convert_to_trajectory_format` (body moves to SessionPersister internal)
- `_clean_session_content` (body moves to SessionPersister internal)

**Keep on AIAgent (NOT moved):**
- `_compress_context` (35 lines, orchestrates PromptAssembler + SessionPersister)
- `flush_memories` (100 lines, needs API client)
- `run_conversation` (826 lines, the orchestration state machine)
- `_build_api_kwargs`, `_handle_max_iterations`, `_interruptible_api_call`
- `_build_assistant_message`, `_extract_reasoning`
- `chat`, `interrupt`, `clear_interrupt`, `is_interrupted`
- `_hydrate_todo_store`, `_dump_api_request_debug`, `_mask_api_key_for_logs`

### 6.2 Acceptance

- All 172 existing tests pass
- External API unchanged: `AIAgent.__init__()`, `run_conversation()`, `chat()`, `interrupt()`, `flush_memories()`, `session_id`
- `delegate_tool.py` access to `self.session_id`, `self._delegate_depth`, `self._active_children` still works

---

## Phase 7: Integration Validation

**Goal:** Full regression check + smoke test.

### 7.1 Full test suite

```bash
python -m pytest tests/ -q --tb=short
```

All 172+ tests must pass (172 original + new tests from Phases 1-5).

### 7.2 Smoke test protocol (from M-1)

Run 3 conversations and verify stdout + session JSON equivalence:

1. **CLI short session** — 2-3 tool calls, no compression
2. **CLI long session** — enough iterations to trigger compression
3. **Gateway single-message** — verify gateway path still works

For each: diff the session JSON metadata structure (session_id, model, timestamps should be present; message count should be non-zero).

### 7.3 Import validation

```bash
python -c "import tools"  # No ImportError
python -c "from agent.prompt_assembler import PromptAssembler"
python -c "from agent.session_persister import SessionPersister"
python -c "from agent.tool_executor import execute_tool_calls, ToolExecConfig, AGENT_LOOP_TOOLS"
python -c "from agent.async_bridge import run_async"
```

---

## Success Criteria (Final, from CK Pass 4)

1. `AIAgent.run_conversation()` produces identical results (behavioral equivalence)
2. No circular import cycles
3. `import tools` succeeds without optional dependencies
4. PromptAssembler, SessionPersister, and execute_tool_calls are independently testable in isolation
5. AIAgent method count drops from 30+ to ~18
6. No breaking changes to external callers (cli.py, gateway/run.py, delegate_tool.py, batch_runner.py, cron/scheduler.py, rl_cli.py)
7. All existing tests pass
8. SessionPersister.session_id setter atomically updates _session_log_file (fixes pre-existing bug)
9. AGENT_LOOP_TOOLS is single source of truth in agent/tool_executor.py (includes clarify)

---

## Out of Scope (Deferred to Future Phase 6)

- Decompose `run_conversation` into sub-handlers (APICallRetrier, ResponseValidator, ErrorRecoveryHandler)
- Extract `_build_assistant_message` / `_extract_reasoning`
- Extract ToolDisplay protocol from execute_tool_calls UI logic
- De-duplicate provider detection in `_build_api_kwargs` / `_handle_max_iterations`
- Replace GIL-dependent `_interrupt_requested` bool with `threading.Event`
- Use auxiliary model for flush_memories (currently uses main model at ~$1.50/compression)
