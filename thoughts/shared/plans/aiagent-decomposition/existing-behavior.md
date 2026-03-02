# Existing Behavioral Contracts -- Phase 0 Baseline

Generated: 2026-02-26
Source: 172 passing tests across tests/ directory

This document catalogs what behaviors the EXISTING test suite verifies for each
refactor target file. Any refactoring must preserve these contracts.

---

## 1. tools/registry.py -- ToolRegistry

**Test file:** `tests/tools/test_registry.py` (10 tests)
**Coverage:** 69%

### Verified Behaviors

| Test | Contract |
|------|----------|
| `TestRegisterAndDispatch::test_register_and_dispatch` | Register a tool, then dispatch by name returns handler result |
| `TestRegisterAndDispatch::test_dispatch_passes_args` | Arguments dict is forwarded to handler unchanged |
| `TestGetDefinitions::test_returns_openai_format` | `get_definitions()` returns `[{"type": "function", "function": schema}]` format |
| `TestGetDefinitions::test_skips_unavailable_tools` | Tools whose `check_fn()` returns False are excluded from definitions |
| `TestUnknownToolDispatch::test_returns_error_json` | Dispatching unknown tool name returns `{"error": "Unknown tool: ..."}` |
| `TestToolsetAvailability::test_no_check_fn_is_available` | Toolset with no check_fn is treated as available |
| `TestToolsetAvailability::test_check_fn_controls_availability` | Toolset with check_fn returning False is unavailable |
| `TestToolsetAvailability::test_check_toolset_requirements` | `check_toolset_requirements()` returns `{toolset: bool}` mapping |
| `TestToolsetAvailability::test_get_all_tool_names` | `get_all_tool_names()` returns sorted list |
| `TestToolsetAvailability::test_handler_exception_returns_error` | Handler raising RuntimeError returns `{"error": "...RuntimeError..."}` |

### Unverified (gap) Behaviors

- Async handler dispatch (via `_run_async`)
- `get_available_toolsets()` -- UI metadata generation
- `check_tool_availability()` -- available/unavailable partitioning
- `get_toolset_for_tool()` with unknown tool name (None return)
- `get_definitions()` with `quiet=True`
- `get_definitions()` when `check_fn()` raises an exception

---

## 2. model_tools.py

**Test file:** None directly (exercised indirectly via test_code_execution.py mocks)
**Coverage:** 29%

### Verified Behaviors (indirect)

| Source | Contract |
|--------|----------|
| Module import succeeds | `_discover_tools()` runs at import time without crashing; tool modules with missing deps are skipped gracefully |
| `test_code_execution.py` patches `model_tools.handle_function_call` | Confirms handle_function_call is the expected dispatch entry point |

### Unverified Behaviors

- `get_tool_definitions()` -- toolset filtering (enabled/disabled), legacy toolset mapping, quiet mode
- `handle_function_call()` -- agent-loop tool interception, execute_code special routing, error wrapping
- `_run_async()` -- sync-to-async bridging in event-loop-running context
- All backward-compat wrapper functions (untested directly, but registry equivalents tested)

---

## 3. run_agent.py -- AIAgent

**Test files:** `tests/tools/test_interrupt.py` (partial), `tests/tools/test_delegate.py` (mocks AIAgent)
**Coverage:** 6%

### Verified Behaviors

| Test | Contract |
|------|----------|
| `TestPreToolCheck::test_all_tools_skipped_when_interrupted` | When `_interrupt_requested=True`, `_execute_tool_calls` adds "cancelled"/"interrupted" tool messages for ALL pending tool calls without executing any handler |
| `TestDelegateTask::test_depth_increments` | Child agent gets `_delegate_depth = parent._delegate_depth + 1` |
| `TestDelegateTask::test_active_children_tracking` | Child is registered/unregistered from `parent._active_children` |

### Implicitly Verified (via import-time side effects)

- AIAgent class is importable
- Constructor signature is stable (test_delegate.py creates mock parents with known fields)
- `_execute_tool_calls` is callable as an unbound method on a mock

### Unverified Behaviors (critical for refactor)

**Constructor / Initialization:**
- OpenAI client creation with base_url/api_key
- Tool definitions loading via `get_tool_definitions()`
- Session ID generation (uuid)
- Session log directory creation
- SessionDB initialization
- TodoStore / MemoryStore initialization
- Context compressor initialization
- Config loading (memory, skills, compression settings)

**Core Agent Loop (`run_conversation`):**
- Message list construction (system + user + history)
- API call loop with tool calling
- Response parsing (content extraction, tool call parsing)
- Multi-turn conversation flow
- Context compression triggering
- Trajectory saving
- Session persistence
- Max iterations enforcement
- Interrupt checking between turns
- Memory flush at conversation end
- Error handling and recovery

**System Prompt:**
- `_build_system_prompt` -- assembly of identity + platform hints + skills + context files + memory + ephemeral prompt
- `_invalidate_system_prompt` -- cache invalidation

**API Integration:**
- `_build_api_kwargs` -- model, messages, tools, max_tokens, reasoning config, provider routing, prompt caching
- `_interruptible_api_call` -- API call with interrupt checking + retry on empty response

**Tool Execution (`_execute_tool_calls`):**
- Normal tool dispatch (non-interrupted path)
- Agent-loop tool interception (todo, memory, session_search, delegate_task)
- Tool progress callback
- Tool delay between calls
- JSON argument parsing with error recovery
- Clarify tool special handling
- Send message tool special handling

**Persistence:**
- `_persist_session` -- JSON log + SQLite
- `_log_msg_to_db` -- individual message logging
- `_flush_messages_to_session_db` -- batch message logging
- `_save_session_log` -- JSON file writing

---

## 4. hermes_state.py -- SessionDB

**Test file:** None
**Coverage:** 0%

### Verified Behaviors

None. This module has zero test coverage.

### Behavioral Surface (all unverified)

- SQLite database creation with WAL mode and foreign keys
- Schema creation (sessions + messages tables, indexes)
- FTS5 virtual table creation with triggers
- Schema versioning and migration (v1 -> v2: finish_reason column)
- Session CRUD (create, get, update, end, delete)
- Message append with auto-incrementing IDs
- Message retrieval (raw and conversation format)
- FTS5 full-text search with source/role filtering
- Session search with pagination
- Token count tracking
- Session export (single and all)
- Session pruning by age
- Thread-safety via WAL mode

---

## 5. agent/context_compressor.py -- ContextCompressor

**Test file:** None
**Coverage:** 13%

### Verified Behaviors

The 13% coverage comes from class-level and field initialization lines being
hit during import -- NOT from any behavioral test.

### Unverified Behaviors

- Constructor: model context length lookup, auxiliary client creation, threshold calculation
- `update_from_response` -- token count tracking from API usage
- `should_compress` -- threshold comparison against prompt tokens
- `should_compress_preflight` -- rough estimate check before API call
- `get_status` -- status dict construction
- `_generate_summary` -- LLM-based summarization of conversation turns
- `compress` -- full compression pipeline (protect head/tail, summarize middle, inject summary)

---

## 6. agent/prompt_builder.py

**Test file:** None
**Coverage:** 12%

### Verified Behaviors

Coverage comes from constant definitions being imported (DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS, etc.) by run_agent.py's import chain. No behavioral tests exist.

### Unverified Behaviors

- `_scan_context_content` -- prompt injection detection (10 threat patterns + invisible unicode)
- `_read_skill_description` -- skill YAML/JSON metadata parsing
- `build_skills_system_prompt` -- skill index construction from filesystem
- `_truncate_content` -- content truncation with byte-size limit
- `build_context_files_prompt` -- SOUL.md / AGENTS.md / .cursorrules scanning and injection

---

## 7. agent/prompt_caching.py

**Test file:** None
**Coverage:** 12%

### Verified Behaviors

Coverage from imports only. No behavioral tests.

### Unverified Behaviors

- `_apply_cache_marker` -- cache_control injection for different content formats (str, list, tool role, None)
- `apply_anthropic_cache_control` -- system_and_3 strategy: deep copy + up to 4 breakpoints (system + last 3 non-system)

---

## 8. tools/__init__.py

**Test file:** None directly
**Coverage:** 90%

### Verified Behaviors

All imports succeed (covered by importing tools package in tests). This
verifies that all tool modules are importable and their exported symbols exist.

### Unverified Behaviors

- `check_file_requirements()` -- delegates to `check_terminal_requirements()`

---

## Summary: Must-Preserve Contracts

These are the behaviors verified by existing tests that MUST be preserved
through refactoring. Any decomposition of AIAgent must ensure:

1. **ToolRegistry contract** (10 tests):
   - register -> dispatch roundtrip
   - OpenAI-format schema output
   - check_fn-based filtering
   - Unknown tool error format
   - Handler exception wrapping
   - Sorted tool names

2. **Interrupt contract** (1 test):
   - `_execute_tool_calls` skips ALL tools when `_interrupt_requested=True`
   - Each skipped tool gets a "cancelled"/"interrupted" tool message

3. **Delegation contract** (2 tests):
   - Child agent depth = parent depth + 1
   - Active children tracking (register on start, unregister on finish)

4. **Import stability**:
   - `from run_agent import AIAgent` works
   - `from tools.registry import ToolRegistry` works
   - `from model_tools import handle_function_call` works
   - `tools` package import triggers all tool registrations

---

## Test-to-File Mapping

| Test File | Target File(s) Exercised |
|-----------|--------------------------|
| `tests/tools/test_registry.py` | `tools/registry.py` |
| `tests/tools/test_interrupt.py` | `run_agent.py` (_execute_tool_calls) |
| `tests/tools/test_delegate.py` | `run_agent.py` (AIAgent constructor fields) |
| `tests/tools/test_code_execution.py` | `model_tools.py` (mocked handle_function_call) |
| `tests/tools/test_approval.py` | `tools/approval.py` (not a refactor target) |
| `tests/tools/test_file_tools.py` | `tools/file_tools.py` (not a refactor target) |
| `tests/tools/test_fuzzy_match.py` | `tools/fuzzy_match.py` (not a refactor target) |
| `tests/tools/test_patch_parser.py` | `tools/patch_parser.py` (not a refactor target) |
| `tests/tools/test_todo_tool.py` | `tools/todo_tool.py` (not a refactor target) |
| `tests/gateway/*` | `gateway/` (not a refactor target) |
| `tests/hermes_cli/*` | `hermes_cli/` (not a refactor target) |
