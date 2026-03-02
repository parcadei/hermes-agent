# AIAgent Decomposition Spec

Created: 2026-02-26
Author: architect-agent
Status: Phase 1 — SPEC (drives test generation and implementation)

---

## 1. Current State Summary

`AIAgent` in `run_agent.py` is a 2500-line god object with 30+ methods, 50+ instance
fields, and five distinct responsibilities tangled into one class:

| Concern | Methods | Lines (approx) |
|---------|---------|----------------|
| System prompt assembly | `_build_system_prompt`, `_invalidate_system_prompt` | 65 |
| Tool dispatch & execution | `_execute_tool_calls` + inline agent-loop tools | 210 |
| Session persistence | `_persist_session`, `_log_msg_to_db`, `_flush_messages_to_session_db`, `_save_session_log`, `_save_trajectory`, `_convert_to_trajectory_format` | 340 |
| Context compression | `_compress_context`, `flush_memories` | 100 |
| Conversation orchestration | `run_conversation`, `_build_api_kwargs`, `_interruptible_api_call`, `_build_assistant_message`, `_handle_max_iterations`, `chat` | 800+ |

Additionally, two structural import problems exist:

1. **Circular import**: `model_tools.py` <-> `tools/registry.py`
2. **Eager imports**: `tools/__init__.py` eagerly imports every tool module

---

## 2. Current Interface Contracts

### 2.1 AIAgent._build_system_prompt

```python
def _build_system_prompt(self, system_message: str = None) -> str
```

**Behavior (VERIFIED from run_agent.py L1060-1123):**
- Assembles prompt from 7 layers in fixed order:
  1. `DEFAULT_AGENT_IDENTITY` (constant from `agent.prompt_builder`)
  2. Tool-aware guidance (MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE)
     -- only injected when the corresponding tool is in `self.valid_tool_names`
  3. Caller-provided `system_message` (gateway session context, etc.)
  4. Memory blocks from `self._memory_store.format_for_system_prompt("memory")`
     and `format_for_system_prompt("user")` -- gated on `_memory_enabled` / `_user_profile_enabled`
  5. Skills index via `build_skills_system_prompt()` -- gated on skills tools being loaded
  6. Context files via `build_context_files_prompt()` -- gated on `not self.skip_context_files`
  7. Current datetime and platform hint

**Side effects:**
- Reads filesystem (context files, skills directory) via `build_context_files_prompt()` and `build_skills_system_prompt()`
- Reads `self._memory_store` (in-memory state loaded from disk)
- Pure otherwise -- no network calls, no mutations

**Dependencies reads from AIAgent:**
- `self.valid_tool_names: set[str]`
- `self._memory_store: Optional[MemoryStore]`
- `self._memory_enabled: bool`
- `self._user_profile_enabled: bool`
- `self.skip_context_files: bool`
- `self.platform: Optional[str]`

**Caching contract:**
- Result is stored in `self._cached_system_prompt`
- Built once per session on first `run_conversation()` call
- Invalidated by `_invalidate_system_prompt()` after compression events
- `_invalidate_system_prompt()` also reloads memory from disk

**Invariants:**
- `DEFAULT_AGENT_IDENTITY` is always the first segment
- The ephemeral system prompt is NOT included (appended at API call time only)
- Layer order is stable -- prefix caching depends on it


### 2.2 AIAgent._execute_tool_calls

```python
def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str) -> None
```

**Behavior (VERIFIED from run_agent.py L1397-1603):**
- Iterates over `assistant_message.tool_calls` (OpenAI response objects with `.id`, `.function.name`, `.function.arguments`)
- For each tool call:
  1. Checks `self._interrupt_requested` -- if true, skips remaining calls with cancel messages
  2. Parses `function_args` from JSON
  3. Dispatches based on function_name:
     - **Agent-loop tools** (handled inline, NOT via `handle_function_call`):
       - `todo` -> `todo_tool(store=self._todo_store)`
       - `memory` -> `memory_tool(store=self._memory_store)`
       - `session_search` -> `session_search(db=self._session_db)`
       - `clarify` -> `clarify_tool(callback=self.clarify_callback)`
       - `delegate_task` -> `delegate_task(parent_agent=self)`
     - **All other tools** -> `handle_function_call(function_name, function_args, effective_task_id)`
  4. Truncates result to MAX_TOOL_RESULT_CHARS (100,000)
  5. Appends `{"role": "tool", "content": result, "tool_call_id": id}` to messages
  6. Logs to session DB via `self._log_msg_to_db()`
  7. Checks interrupt again after each tool, skipping remaining if triggered
  8. Sleeps `self.tool_delay` between calls

**Side effects:**
- Mutates `messages` list (appends tool results)
- Calls external tools (network, filesystem, browser, terminal)
- Logs to session DB
- Resets nudge counters (`_turns_since_memory`, `_iters_since_skill`)
- UI output: spinner management, progress callbacks

**Dependencies reads from AIAgent:**
- `self._interrupt_requested: bool`
- `self._todo_store: TodoStore`
- `self._memory_store: Optional[MemoryStore]`
- `self._session_db: Optional[SessionDB]`
- `self.clarify_callback: Optional[callable]`
- `self.tool_delay: float`
- `self.tool_progress_callback: Optional[callable]`
- `self.quiet_mode: bool`
- `self.log_prefix: str`
- `self.log_prefix_chars: int`
- `self.verbose_logging: bool`
- `self._delegate_depth: int` (used by delegate_task)
- `self._active_children: list` (used by delegate_task)

**Invariants:**
- Every `tool_call.id` in the assistant message gets exactly one tool response appended
- Interrupted tools get a `"[Tool execution cancelled]"` response (never missing)
- Agent-loop tools are NEVER routed through `model_tools.handle_function_call`


### 2.3 Session Persistence (multiple methods)

**_persist_session(messages, conversation_history)**
```python
def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None)
```
- Coordinator: calls `_save_session_log(messages)` and `_flush_messages_to_session_db(messages, conversation_history)`
- Updates `self._session_messages`

**_save_session_log(messages)**
```python
def _save_session_log(self, messages: List[Dict[str, Any]] = None)
```
- Writes JSON to `self.session_log_file` (path: `~/.hermes/sessions/session_{id}.json`)
- Cleans assistant content (scratchpad -> think tags)
- Includes session metadata (session_id, model, base_url, platform, timestamps)
- Called incrementally after each tool loop iteration AND at conversation end
- **File I/O**: writes to filesystem

**_log_msg_to_db(msg)**
```python
def _log_msg_to_db(self, msg: Dict)
```
- Appends single message to `self._session_db.append_message()` (SQLite)
- Called immediately after every `messages.append()` in the conversation loop
- No-op if `self._session_db is None`

**_flush_messages_to_session_db(messages, conversation_history)**
```python
def _flush_messages_to_session_db(self, messages: List[Dict], conversation_history: List[Dict] = None)
```
- Batch-persists unlogged messages to SQLite
- Calculates start_idx from conversation_history length
- Called from `_persist_session` as final flush

**_save_trajectory / _convert_to_trajectory_format**
```python
def _save_trajectory(self, messages, user_query, completed)
def _convert_to_trajectory_format(self, messages, user_query, completed) -> List[Dict[str, Any]]
```
- Converts internal message format to training-compatible trajectory format
- Writes to JSONL via `agent.trajectory.save_trajectory()`
- Gated on `self.save_trajectories`

**Dependencies reads from AIAgent:**
- `self.session_id: str`
- `self._session_db: Optional[SessionDB]`
- `self.session_log_file: Path`
- `self.model: str`
- `self.base_url: str`
- `self.platform: Optional[str]`
- `self.session_start: datetime`
- `self.save_trajectories: bool`
- `self.tools: list` (for trajectory format)
- `self.verbose_logging: bool`
- `self.quiet_mode: bool`

**Side effects:**
- Filesystem writes (session JSON, trajectory JSONL)
- SQLite writes (session DB)

**Invariants:**
- `_persist_session` is called on EVERY exit path from `run_conversation` (normal, error, interrupt, partial)
- Session log is overwritten (not appended) each time -- always reflects latest state
- Messages are never dropped from persistence even on errors


### 2.4 Context Compression (in AIAgent)

**_compress_context(messages, system_message, approx_tokens)**
```python
def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None) -> tuple
```
- VERIFIED from run_agent.py L1361-1395
- Steps:
  1. `self.flush_memories(messages, min_turns=0)` -- pre-compression memory save
  2. `self.context_compressor.compress(messages, current_tokens=approx_tokens)` -- actual compression
  3. Appends todo snapshot to compressed messages
  4. `self._invalidate_system_prompt()` -- forces rebuild
  5. Rebuilds system prompt via `self._build_system_prompt(system_message)`
  6. If session DB exists: ends current session with "compression" reason, creates new session with parent link
- Returns `(compressed_messages, new_system_prompt)`

**flush_memories(messages, min_turns)**
```python
def flush_memories(self, messages: list = None, min_turns: int = None)
```
- VERIFIED from run_agent.py L1262-1359
- Gives the model one API call to save memories before compression
- Injects a flush message, makes API call with only memory tool, executes memory tool calls
- Strips flush artifacts afterward
- Gated on: memory tool available, `_memory_store` exists, turn count threshold

**Dependencies reads from AIAgent:**
- `self.context_compressor: ContextCompressor`
- `self._todo_store: TodoStore`
- `self._memory_store: Optional[MemoryStore]`
- `self._session_db: Optional[SessionDB]`
- `self.session_id: str`
- `self.platform: Optional[str]`
- `self.model: str`
- `self.client: OpenAI`
- `self.tools: list`
- `self.valid_tool_names: set`
- `self._user_turn_count: int`
- `self._memory_flush_min_turns: int`
- `self.quiet_mode: bool`
- `self._cached_system_prompt: Optional[str]`

**Side effects:**
- Mutates `messages` list (compression, todo injection)
- Makes an API call (flush_memories)
- Writes to memory files on disk
- Creates new session in SQLite DB
- Rebuilds system prompt

**Invariants:**
- Memory flush always runs before compression (memories are saved before they're lost)
- Session DB always gets a new session record after compression (parent chain)
- Todo snapshot is always injected into compressed messages


### 2.5 ContextCompressor (agent/context_compressor.py)

```python
class ContextCompressor:
    def __init__(self, model, threshold_percent=0.85, protect_first_n=3,
                 protect_last_n=4, summary_target_tokens=500, quiet_mode=False)
    def update_from_response(self, usage: Dict[str, Any])
    def should_compress(self, prompt_tokens: int = None) -> bool
    def should_compress_preflight(self, messages: List[Dict]) -> bool
    def get_status(self) -> Dict[str, Any]
    def compress(self, messages: List[Dict], current_tokens: int = None) -> List[Dict]
```

- Already a clean standalone class
- Dependencies: `agent.auxiliary_client.get_text_auxiliary_client()`, `agent.model_metadata`
- Has its own OpenAI client for summarization (separate from main model)
- State: `compression_count`, `last_prompt_tokens`, `last_completion_tokens`, `last_total_tokens`
- No direct dependency on AIAgent internals


### 2.6 model_tools.py

```python
# Public API (consumed by run_agent.py, cli.py, batch_runner.py, rl environments)
def get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode) -> List[Dict]
def handle_function_call(function_name, function_args, task_id, user_task) -> str
def get_all_tool_names() -> List[str]
def get_toolset_for_tool(tool_name) -> Optional[str]
def get_available_toolsets() -> Dict[str, dict]
def check_toolset_requirements() -> Dict[str, bool]
def check_tool_availability(quiet) -> Tuple[List[str], List[dict]]

# Module-level constants (built after discovery)
TOOL_TO_TOOLSET_MAP: Dict[str, str]
TOOLSET_REQUIREMENTS: Dict[str, dict]

# Internal
_AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}
_run_async(coro)  # Sync->async bridge
_discover_tools()  # Imports all tool modules
```

**Circular import with registry.py:**
- `model_tools.py` L29: `from tools.registry import registry`
- `tools/registry.py` L124: `from model_tools import _run_async` (inside `dispatch()`)
- The import in registry.py is DEFERRED (inside the `dispatch` method body, not top-level)
- This means it works at runtime but creates a logical cycle that:
  - Confuses static analysis tools
  - Makes the dependency direction unclear
  - Would break if registry.py ever needed `_run_async` at import time


### 2.7 tools/__init__.py (Eager Import Problem)

**VERIFIED from tools/__init__.py L1-163:**
- Eagerly imports from 17 tool modules at package level:
  - `web_tools`, `terminal_tool`, `vision_tools`, `mixture_of_agents_tool`,
    `image_generation_tool`, `skills_tool`, `skill_manager_tool`, `browser_tool`,
    `cronjob_tools`, `rl_training_tool`, `file_tools`, `tts_tool`, `todo_tool`,
    `clarify_tool`, `code_execution_tool`, `delegate_tool`
- Re-exports 60+ symbols in `__all__`

**Why it breaks:**
- `web_tools.py` imports `firecrawl` which is an optional dependency
- `browser_tool.py` imports browser-specific dependencies
- `image_generation_tool.py` imports `fal_client`
- Any `from tools import X` triggers ALL of these imports
- This causes ImportError when optional dependencies aren't installed

**Who imports from `tools` directly (via `from tools import ...`):**
- `tools/file_tools.py` L9: `from tools import check_file_requirements` (SELF-IMPORT)
- Several test files
- Most real consumers import from specific submodules (`tools.registry`, `tools.terminal_tool`, etc.)

**Key insight:** `model_tools._discover_tools()` already handles graceful discovery via `importlib.import_module` with try/except. The `tools/__init__.py` eager imports are REDUNDANT with this mechanism and only serve as convenience re-exports that create import-time failures.


### 2.8 SessionDB (hermes_state.py)

```python
class SessionDB:
    def __init__(self, db_path=DEFAULT_DB_PATH)
    def create_session(self, session_id, source, model, model_config=None,
                       system_prompt=None, user_id=None, parent_session_id=None) -> str
    def end_session(self, session_id, end_reason) -> None
    def update_system_prompt(self, session_id, system_prompt) -> None
    def update_token_counts(self, session_id, input_tokens, output_tokens) -> None
    def get_session(self, session_id) -> Optional[Dict]
    def append_message(self, session_id, role, content, tool_name=None,
                       tool_calls=None, tool_call_id=None, token_count=None,
                       finish_reason=None) -> int
    def get_messages(self, session_id) -> List[Dict]
    def get_messages_as_conversation(self, session_id) -> List[Dict]
    def search_messages(self, query, source_filter=None, role_filter=None,
                        limit=10, offset=0) -> List[Dict]
    def search_sessions(self, source=None, limit=10, offset=0) -> List[Dict]
    def session_count(self, source=None) -> int
    def message_count(self, session_id=None) -> int
    def export_session(self, session_id) -> Optional[Dict]
    def delete_session(self, session_id) -> bool
    def prune_sessions(self, older_than_days=30, source=None) -> int
    def close(self)
```

- Already a clean standalone class
- Thread-safe for single writer (SQLite WAL mode, check_same_thread=False)
- Used by AIAgent, CLI, gateway, session_search_tool


---

## 3. Proposed Decomposed Interfaces

### 3.1 PromptAssembler

**Responsibility:** Build the system prompt from its component layers. Owns the prompt
assembly logic, caching, and invalidation. Does NOT own the data sources (memory store,
config, etc.) -- receives them as parameters.

**File:** `agent/prompt_assembler.py`

```python
from typing import Optional, Set
from tools.memory_tool import MemoryStore

class PromptAssembler:
    """Assembles the system prompt from component layers.

    Stateless except for caching. All data sources are injected.
    """

    def __init__(
        self,
        platform: Optional[str] = None,
        skip_context_files: bool = False,
    ):
        self._platform = platform
        self._skip_context_files = skip_context_files
        self._cached_prompt: Optional[str] = None

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

        Layer order (stable for prefix caching):
          1. DEFAULT_AGENT_IDENTITY
          2. Tool-aware guidance (memory, session_search, skills)
          3. Caller system_message (gateway context)
          4. Memory blocks (if enabled)
          5. Skills index (if skills tools loaded)
          6. Context files (if not skipped)
          7. Datetime + platform hint

        Returns the assembled prompt string.
        """

    @property
    def cached(self) -> Optional[str]:
        """Return the cached prompt, or None if not yet built."""
        return self._cached_prompt

    def invalidate(self, memory_store: Optional[MemoryStore] = None) -> None:
        """Clear the cached prompt. Optionally reload memory from disk.

        Called after compression events to force a rebuild.
        """
        self._cached_prompt = None
        if memory_store is not None:
            memory_store.load_from_disk()
```

**How it interacts with AIAgent (composition):**
```python
# In AIAgent.__init__:
self._prompt_assembler = PromptAssembler(
    platform=self.platform,
    skip_context_files=self.skip_context_files,
)

# In run_conversation (replaces self._cached_system_prompt logic):
if self._prompt_assembler.cached is None:
    self._prompt_assembler.build(
        valid_tool_names=self.valid_tool_names,
        system_message=system_message,
        memory_store=self._memory_store,
        memory_enabled=self._memory_enabled,
        user_profile_enabled=self._user_profile_enabled,
    )

active_system_prompt = self._prompt_assembler.cached
```

**Tests must verify:**
- Layer ordering is preserved (identity first, platform hint last)
- Tool guidance is only injected when tools are in `valid_tool_names`
- Memory blocks are only injected when `memory_enabled=True` / `user_profile_enabled=True`
- Context files are skipped when `skip_context_files=True`
- `invalidate()` clears cache and reloads memory
- `build()` populates the cache
- Platform hints are only added for recognized platforms
- Ephemeral system prompt is NOT included (caller adds at API call time)


### 3.2 ToolExecutor

**Responsibility:** Execute tool calls from assistant messages. Owns the dispatch logic
(including the agent-loop-tool special cases), interrupt checking, result truncation, and
progress reporting. Does NOT own the message list -- caller passes it.

**File:** `agent/tool_executor.py`

```python
from typing import Any, Callable, Dict, List, Optional
from tools.todo_tool import TodoStore
from tools.memory_tool import MemoryStore

class ToolExecutor:
    """Dispatches and executes tool calls from assistant messages.

    Handles both agent-loop tools (todo, memory, session_search, clarify,
    delegate_task) and registry-dispatched tools.
    """

    MAX_TOOL_RESULT_CHARS = 100_000

    def __init__(
        self,
        *,
        todo_store: TodoStore,
        memory_store: Optional[MemoryStore] = None,
        session_db: Optional[Any] = None,  # SessionDB
        clarify_callback: Optional[Callable] = None,
        tool_delay: float = 1.0,
        tool_progress_callback: Optional[Callable] = None,
        quiet_mode: bool = False,
        verbose_logging: bool = False,
        log_prefix: str = "",
        log_prefix_chars: int = 100,
    ):
        self._todo_store = todo_store
        self._memory_store = memory_store
        self._session_db = session_db
        self._clarify_callback = clarify_callback
        self._tool_delay = tool_delay
        self._tool_progress_callback = tool_progress_callback
        self._quiet_mode = quiet_mode
        self._verbose_logging = verbose_logging
        self._log_prefix = log_prefix
        self._log_prefix_chars = log_prefix_chars

    def execute(
        self,
        assistant_message,
        messages: List[Dict],
        effective_task_id: str,
        *,
        is_interrupted: Callable[[], bool],
        log_msg_to_db: Callable[[Dict], None],
        parent_agent: Optional[Any] = None,  # For delegate_task
    ) -> None:
        """Execute all tool calls from an assistant message.

        Appends tool result messages to `messages`.
        Checks `is_interrupted()` before each tool call.
        Calls `log_msg_to_db(msg)` after each appended message.

        Args:
            assistant_message: The API response message object with .tool_calls
            messages: The conversation messages list (mutated in place)
            effective_task_id: Task ID for VM/browser isolation
            is_interrupted: Callable returning True if agent should stop
            log_msg_to_db: Callback to persist each message to session DB
            parent_agent: The parent AIAgent (needed for delegate_task only)
        """
```

**Key design decisions:**
- `is_interrupted` and `log_msg_to_db` are passed as callbacks to avoid ToolExecutor
  needing a reference to the full AIAgent
- `parent_agent` is only passed through to `delegate_task` -- ToolExecutor itself
  does not call methods on it
- The display logic (spinner, cute messages) stays in ToolExecutor because it's
  tightly coupled to tool execution timing

**Tests must verify:**
- Agent-loop tools (todo, memory, session_search, clarify, delegate_task) are dispatched
  with their correct state arguments, NOT through `handle_function_call`
- All other tools go through `handle_function_call(name, args, task_id)`
- Every tool_call gets exactly one tool response in messages
- Interrupt before first tool -> all calls get cancel messages
- Interrupt after Nth tool -> remaining calls get cancel messages
- Result truncation at MAX_TOOL_RESULT_CHARS works
- Invalid JSON args get `{}` fallback
- Tool delay is applied between calls (not after last)
- `log_msg_to_db` is called for every appended message


### 3.3 SessionPersister

**Responsibility:** Save conversation state to JSON files and SQLite. Owns session
lifecycle (create, log, flush, end) and trajectory conversion.

**File:** `agent/session_persister.py`

```python
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

class SessionPersister:
    """Persists conversation state to JSON logs and SQLite.

    Handles both incremental logging (per-message) and bulk persistence
    (end-of-conversation flush).
    """

    def __init__(
        self,
        *,
        session_id: str,
        session_db: Optional[Any] = None,  # SessionDB
        logs_dir: Path,
        model: str,
        base_url: str,
        platform: Optional[str] = None,
        session_start: datetime,
        save_trajectories: bool = False,
        tools: Optional[list] = None,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
    ):
        self._session_id = session_id
        self._session_db = session_db
        self._logs_dir = logs_dir
        self._model = model
        self._base_url = base_url
        self._platform = platform
        self._session_start = session_start
        self._save_trajectories = save_trajectories
        self._tools = tools or []
        self._verbose_logging = verbose_logging
        self._quiet_mode = quiet_mode
        self._session_log_file = logs_dir / f"session_{session_id}.json"

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str):
        self._session_id = value
        self._session_log_file = self._logs_dir / f"session_{value}.json"

    def log_message(self, msg: Dict) -> None:
        """Log a single message to SQLite immediately."""

    def persist(self, messages: List[Dict], conversation_history: List[Dict] = None) -> None:
        """Full persistence: JSON log + SQLite flush. Called on every exit path."""

    def save_session_log(self, messages: List[Dict]) -> None:
        """Write the full session to a JSON file (overwrite mode)."""

    def flush_to_db(self, messages: List[Dict], conversation_history: List[Dict] = None) -> None:
        """Batch-persist unlogged messages to SQLite."""

    def save_trajectory(self, messages: List[Dict], user_query: str, completed: bool) -> None:
        """Save conversation trajectory to JSONL (training format)."""

    def create_compression_session(self, *, platform: str, model: str,
                                    parent_session_id: str) -> str:
        """End current session and create a new one after compression.
        Returns the new session_id."""

    def update_system_prompt(self, prompt: str) -> None:
        """Store the system prompt snapshot in SQLite."""
```

**How it interacts with AIAgent (composition):**
```python
# In AIAgent.__init__:
self._persister = SessionPersister(
    session_id=self.session_id,
    session_db=session_db,
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

# Replaces self._persist_session, self._log_msg_to_db, etc.
```

**Tests must verify:**
- `persist()` writes both JSON log and SQLite
- `log_message()` is a no-op when session_db is None
- `flush_to_db()` skips already-logged messages (start_idx calculation)
- `save_session_log()` overwrites the file (not append)
- `save_trajectory()` is a no-op when `save_trajectories=False`
- `create_compression_session()` ends old session and returns new ID
- Session log JSON includes correct metadata (session_id, model, timestamps)
- Assistant content scratchpad -> think tag conversion in session logs


### 3.4 CompressionManager

**Responsibility:** Orchestrate context compression including pre-compression memory
flush, the actual compression via ContextCompressor, todo snapshot injection, prompt
invalidation, and session DB chain splitting.

**File:** `agent/compression_manager.py`

```python
from typing import Any, Callable, Dict, List, Optional, Tuple

class CompressionManager:
    """Orchestrates context compression for long conversations.

    Coordinates between ContextCompressor (summarization), memory flush,
    todo snapshot injection, prompt invalidation, and session splitting.
    """

    def __init__(
        self,
        *,
        compressor,          # ContextCompressor
        compression_enabled: bool = True,
    ):
        self._compressor = compressor
        self._compression_enabled = compression_enabled

    @property
    def enabled(self) -> bool:
        return self._compression_enabled

    @property
    def compressor(self):
        """Access the underlying ContextCompressor for token tracking."""
        return self._compressor

    def should_compress(self) -> bool:
        """Check if compression is needed based on token usage."""
        return self._compression_enabled and self._compressor.should_compress()

    def compress(
        self,
        messages: List[Dict],
        system_message: Optional[str],
        *,
        approx_tokens: Optional[int] = None,
        flush_memories: Callable[[List[Dict], int], None],
        build_system_prompt: Callable[[Optional[str]], str],
        invalidate_prompt: Callable[[], None],
        inject_todo_snapshot: Callable[[], Optional[str]],
        split_session: Optional[Callable[[str], str]] = None,
    ) -> Tuple[List[Dict], str]:
        """Compress conversation context.

        Orchestrates:
        1. Pre-compression memory flush
        2. ContextCompressor.compress()
        3. Todo snapshot injection
        4. Prompt invalidation and rebuild
        5. Session DB chain splitting

        Args:
            messages: Conversation messages (may be mutated by flush_memories)
            system_message: The original system message for prompt rebuild
            approx_tokens: Approximate current token count
            flush_memories: Callback to flush memories before compression
            build_system_prompt: Callback to rebuild the system prompt
            invalidate_prompt: Callback to invalidate the cached prompt
            inject_todo_snapshot: Returns todo snapshot string or None
            split_session: Optional callback to split session in DB, returns new session_id

        Returns:
            (compressed_messages, new_system_prompt) tuple
        """
```

**Design rationale for callbacks:**
The CompressionManager needs to coordinate several operations that involve AIAgent state
(memory flush needs the API client, prompt rebuild needs tool names, session split needs
the persister). Rather than giving CompressionManager references to all these objects,
we use callbacks that the AIAgent provides. This keeps the dependency graph clean:
CompressionManager depends only on ContextCompressor, and the callbacks are thin wrappers
around existing AIAgent method calls.

**Tests must verify:**
- Memory flush is called before compression
- `ContextCompressor.compress()` is called with correct arguments
- Todo snapshot is injected into compressed messages when present
- Prompt is invalidated and rebuilt after compression
- Session splitting callback is invoked (if provided)
- Returns the compressed messages and new system prompt
- When `compression_enabled=False`, `should_compress()` returns False
- When compressor says no compression needed, no operations happen


---

## 4. Circular Import Fix

### The Cycle

```
model_tools.py  ─── from tools.registry import registry ───>  tools/registry.py
     ^                                                              |
     └──── from model_tools import _run_async  (inside dispatch) ───┘
```

**Details:**
- `model_tools.py` L29: `from tools.registry import registry` (top-level)
- `tools/registry.py` L124: `from model_tools import _run_async` (inside `ToolRegistry.dispatch()` method body)
- The deferred import in `dispatch()` means it works at runtime but creates a logical dependency cycle

### Proposed Fix: Extract `_run_async` to `agent/async_bridge.py`

**New file: `agent/async_bridge.py`**
```python
"""Sync-to-async bridge for tool handlers.

Single source of truth for running async coroutines from synchronous code.
Extracted from model_tools.py to break the circular import between
model_tools.py and tools/registry.py.
"""

import asyncio
import concurrent.futures


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
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=300)
    return asyncio.run(coro)
```

**Changes to `model_tools.py`:**
```python
# Remove _run_async definition (L39-62)
# Add import:
from agent.async_bridge import run_async as _run_async
```

**Changes to `tools/registry.py`:**
```python
# In dispatch() method, change:
#   from model_tools import _run_async
# To:
from agent.async_bridge import run_async
# And use run_async(entry.handler(args, **kwargs)) instead of _run_async(...)
```

This makes the import direction:
```
model_tools.py ──> tools/registry.py ──> agent/async_bridge.py
                                                ^
model_tools.py ─────────────────────────────────┘
```
No cycle. Both `model_tools.py` and `tools/registry.py` depend on `agent/async_bridge.py`,
which has no dependencies on either.

**Risk:** The `import concurrent.futures` is currently deferred inside `_run_async` body
(only imported when there's a running loop). In `agent/async_bridge.py` we can keep this
deferred pattern or import at module level -- `concurrent.futures` is stdlib and always
available, so top-level import is fine.


---

## 5. Eager Import Fix

### Current Problem

`tools/__init__.py` eagerly imports from 17 tool modules at package level. When any
optional dependency is missing (firecrawl, fal_client, browser libs), the entire
`tools` package fails to import.

### Who Actually Uses `tools/__init__.py` Exports?

From the dependency analysis:
- `tools/file_tools.py` imports from `tools/__init__` (circular self-import for `check_file_requirements`)
- `tools/code_execution_tool.py` imports through `cli.py` which touches `tools/__init__`
- Most real consumers import from specific submodules directly

### Proposed Fix: Gut `tools/__init__.py`

**Replace the entire file with:**
```python
"""Tools Package

Individual tool implementations for the Hermes Agent.
Import from specific submodules (e.g., ``from tools.terminal_tool import ...``).

Tool discovery is handled by model_tools._discover_tools() which uses
importlib.import_module with try/except for graceful degradation.
"""

# No package-level imports. Use specific submodule imports:
#   from tools.terminal_tool import terminal_tool
#   from tools.web_tools import web_search_tool
#   from tools.registry import registry
```

**Impact analysis:**

| File importing from `tools` | Current import | Fix needed |
|-----|-----|-----|
| `tools/file_tools.py` | `from tools import check_file_requirements` (L~unused?) | Move `check_file_requirements` into `file_tools.py` or `toolsets.py` |
| Test files | Various `from tools import X` | Change to `from tools.specific_module import X` |

**What about `tools/file_tools.py` self-import?**
The dependency graph shows `tools/file_tools.py -> tools/__init__.py`. Looking at the
`__init__.py`, it defines `check_file_requirements()` at line 159 which calls
`terminal_tool.check_terminal_requirements()`. This function should be moved into
`tools/file_tools.py` directly (it's a one-liner).

**Backward compatibility:**
- `model_tools._discover_tools()` already imports each tool module individually via
  `importlib.import_module`. It does NOT use `tools/__init__.py`. So gutting the
  `__init__` does not affect tool discovery.
- External consumers (if any) that do `from tools import web_search_tool` will break.
  They should use `from tools.web_tools import web_search_tool`.
- The `__all__` export list becomes unnecessary.


---

## 6. Edge Cases and Risks

### 6.1 Thread Safety

**Current situation:**
- `AIAgent` is synchronous (single-threaded conversation loop)
- Gateway (`gateway/run.py`) is async but creates a fresh `AIAgent` per message
- `_interruptible_api_call()` uses a background thread for the API call
- `interrupt()` is called from a different thread (gateway message handler)

**Risks with decomposition:**
- `ToolExecutor.execute()` mutates the `messages` list from the same thread as the
  conversation loop -- no issue
- `SessionPersister.log_message()` is called from the conversation thread -- no issue
- `self._interrupt_requested` is read by ToolExecutor and written by the interrupt
  thread -- must remain on AIAgent, passed as `is_interrupted` callback

**Mitigation:** The callback pattern (`is_interrupted: Callable[[], bool]`) ensures
thread-safety properties are the same as today. The boolean flag is inherently atomic
on CPython (GIL), so no lock is needed.


### 6.2 Shared State Between Components

| State | Owner | Consumers | Risk |
|-------|-------|-----------|------|
| `messages: list` | run_conversation (local var) | ToolExecutor, SessionPersister, CompressionManager | All receive by reference; mutations visible to all |
| `_interrupt_requested: bool` | AIAgent | ToolExecutor (via callback) | Thread-safe via GIL |
| `_todo_store: TodoStore` | AIAgent | ToolExecutor, CompressionManager (todo snapshot) | Single owner, passed to both |
| `_memory_store: MemoryStore` | AIAgent | PromptAssembler, ToolExecutor, CompressionManager | Single owner, passed to all |
| `_session_db: SessionDB` | AIAgent (injected) | SessionPersister, ToolExecutor (session_search), CompressionManager (session split) | Single writer per session (thread-safe SQLite) |
| `_cached_system_prompt: str` | PromptAssembler | AIAgent (reads .cached) | Single owner |
| `valid_tool_names: set` | AIAgent | PromptAssembler, run_conversation | Set once in __init__, never mutated |
| `_user_turn_count: int` | AIAgent | CompressionManager (flush threshold) | Single writer |
| `context_compressor` | CompressionManager | AIAgent (for update_from_response) | Owned by CompressionManager, accessed via .compressor property |
| `session_id: str` | SessionPersister | AIAgent, CompressionManager | Mutated by compression (new session), propagated via persister.session_id setter |

**Key risk:** `session_id` changes during compression. Currently AIAgent.session_id is
mutated in `_compress_context`. With decomposition, `SessionPersister` owns session_id
and `CompressionManager.compress()` calls `split_session` callback which updates the
persister's session_id. AIAgent must update its own `self.session_id` reference too
(or always read from persister).

**Mitigation:** Make AIAgent.session_id a property that reads from `self._persister.session_id`.


### 6.3 Breaking Changes to External Callers

**Callers of AIAgent (7 files):**

| File | Import | Usage |
|------|--------|-------|
| `cli.py` L337 | `from run_agent import AIAgent` | Creates agent, calls `run_conversation()`, `chat()`, `flush_memories()`, `interrupt()` |
| `gateway/run.py` L815 | `from run_agent import AIAgent` | Creates agent per message, calls `run_conversation()`, `interrupt()` |
| `tools/delegate_tool.py` L93 | `from run_agent import AIAgent` | Creates child agent, accesses `_delegate_depth`, `_active_children` |
| `cron/scheduler.py` L149 | `from run_agent import AIAgent` | Creates agent for cron jobs |
| `batch_runner.py` L37 | `from run_agent import AIAgent` | Creates agent for batch processing |
| `rl_cli.py` L64 | `from run_agent import AIAgent` | Creates agent for RL |
| `tests/tools/test_interrupt.py` L94 | `from run_agent import AIAgent` | Tests interrupt mechanism |

**Contract preserved (NO breaking changes):**
- `AIAgent.__init__()` signature: UNCHANGED
- `AIAgent.run_conversation()` signature and return type: UNCHANGED
- `AIAgent.chat()`: UNCHANGED
- `AIAgent.interrupt()` / `clear_interrupt()` / `is_interrupted`: UNCHANGED
- `AIAgent.flush_memories()`: UNCHANGED
- `AIAgent.session_id`: UNCHANGED (becomes property)
- `AIAgent._delegate_depth` / `_active_children`: UNCHANGED (delegate_tool accesses these)

**The decomposition is internal.** AIAgent becomes a thin orchestrator that delegates to
its composed components. All public methods remain on AIAgent with the same signatures.


### 6.4 Configuration Coupling

**hermes_cli.config access in AIAgent.__init__:**
- Memory config: `load_config().get("memory", {})` -> controls `_memory_enabled`, `_user_profile_enabled`, `_memory_nudge_interval`, `_memory_flush_min_turns`
- Skills config: `load_config().get("skills", {})` -> controls `_skill_nudge_interval`
- These are loaded once in `__init__` and stored as instance fields

**Risk:** The config loading stays in AIAgent.__init__ because it controls which components
are created and how they're configured. The extracted components receive already-resolved
config values, not raw config access.

**No change needed** -- config coupling is already properly bounded to `__init__`.


### 6.5 Nudge Logic Placement

**Memory nudge** (L1716-1725): Appends reminder text to user_message based on
`_turns_since_memory >= _memory_nudge_interval`. Counter resets when memory tool is used.

**Skill nudge** (L1729-1736): Appends reminder text based on
`_iters_since_skill >= _skill_nudge_interval`. Counter resets when skill_manage is used.

These nudges are in `run_conversation()` and modify the user message before it enters
the messages list. They should stay in AIAgent's `run_conversation()` method because:
1. They depend on conversation-level state (turn counts, iteration counts)
2. They modify the user message (input to the conversation loop)
3. Counter resets happen in `_execute_tool_calls` -- which moves to ToolExecutor

**Fix:** ToolExecutor needs to expose a callback or return value to notify AIAgent
when memory/skill_manage tools were used, so AIAgent can reset the counters.
Simplest: ToolExecutor calls a `on_tool_executed(tool_name)` callback that AIAgent
provides. AIAgent resets counters in its callback implementation.


### 6.6 Display/UI Coupling in ToolExecutor

`_execute_tool_calls` currently contains ~100 lines of spinner and display logic
(KawaiiSpinner, cute messages, emoji maps). This is tightly coupled to tool execution
timing (spinner starts before, stops after).

**Decision:** Keep display logic in ToolExecutor. Extracting it further would require
another level of callbacks for start/stop/result display that adds complexity without
meaningful separation. The spinner management IS part of tool execution UX.


---

## 7. Implementation Phases

### Phase 1: Foundation (No behavior change)
**Files to create:**
- `agent/async_bridge.py` -- extract `_run_async`
- `agent/prompt_assembler.py` -- PromptAssembler class
- `agent/tool_executor.py` -- ToolExecutor class
- `agent/session_persister.py` -- SessionPersister class
- `agent/compression_manager.py` -- CompressionManager class

**Acceptance:**
- All new files have type-correct signatures
- No imports from `run_agent.py` (one-way dependency)
- Each class is independently importable without circular deps

### Phase 2: Circular import fix
**Files to modify:**
- `model_tools.py` -- remove `_run_async` definition, import from `agent.async_bridge`
- `tools/registry.py` -- import `run_async` from `agent.async_bridge` instead of `model_tools`

**Acceptance:**
- `python -c "from tools.registry import registry"` works
- `python -c "from model_tools import handle_function_call"` works
- No circular import warnings from static analysis

### Phase 3: Eager import fix
**Files to modify:**
- `tools/__init__.py` -- gut to empty (docstring only)
- `tools/file_tools.py` -- inline `check_file_requirements` if needed
- Any test files that import from `tools` directly

**Acceptance:**
- `python -c "import tools"` works even without firecrawl/fal_client installed
- All existing `from tools.X import Y` imports still work
- `model_tools._discover_tools()` still discovers all tools

### Phase 4: Wire components into AIAgent
**Files to modify:**
- `run_agent.py` -- compose PromptAssembler, ToolExecutor, SessionPersister, CompressionManager

**Acceptance:**
- All existing tests pass
- `AIAgent.__init__` signature unchanged
- `AIAgent.run_conversation()` return type unchanged
- CLI, gateway, delegate_tool, batch_runner, cron all work unchanged

### Phase 5: Testing
**Files to create:**
- `tests/agent/test_prompt_assembler.py`
- `tests/agent/test_tool_executor.py`
- `tests/agent/test_session_persister.py`
- `tests/agent/test_compression_manager.py`
- `tests/agent/test_async_bridge.py`
- `tests/test_import_cycles.py` -- verifies no circular imports
- `tests/test_lazy_tools_init.py` -- verifies tools/__init__.py is clean

**Coverage target:** 80%+ on all new files


---

## 8. Success Criteria

1. AIAgent.run_conversation() produces identical results before and after decomposition
   (behavioral equivalence)
2. No circular import cycles involving model_tools.py <-> tools/registry.py
3. `import tools` succeeds without optional dependencies installed
4. Each extracted class is independently testable (can be instantiated in isolation)
5. AIAgent method count drops from 30+ to ~15 (orchestration only)
6. No breaking changes to: cli.py, gateway/run.py, tools/delegate_tool.py,
   batch_runner.py, cron/scheduler.py, rl_cli.py
7. All existing tests continue to pass
