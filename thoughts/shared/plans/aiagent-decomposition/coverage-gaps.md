# Coverage Gaps Report -- Phase 0 Baseline

Generated: 2026-02-26
Test suite: 172 passed, 9 deselected (integration)
Overall coverage across refactor targets: **12% (242/2034 statements)**

---

## Per-File Summary

| File | Stmts | Miss | Cover | Notes |
|------|-------|------|-------|-------|
| `run_agent.py` | 1329 | 1244 | **6%** | God object -- primary decomposition target |
| `hermes_state.py` | 173 | 173 | **0%** | SessionDB entirely untested |
| `agent/prompt_builder.py` | 171 | 150 | **12%** | Only constant imports covered |
| `agent/context_compressor.py` | 105 | 91 | **13%** | Only class definition + field defaults |
| `agent/prompt_caching.py` | 34 | 30 | **12%** | Only module-level imports |
| `model_tools.py` | 98 | 70 | **29%** | _discover_tools covered; orchestration not |
| `tools/registry.py` | 104 | 32 | **69%** | Core register/dispatch well-tested |
| `tools/__init__.py` | 20 | 2 | **90%** | Only check_file_requirements uncovered |

---

## Per-Method Coverage Detail

### run_agent.py -- AIAgent (6% total)

#### PUBLIC API (must-cover for refactor safety)

| Method | Coverage | Lines Covered | Priority |
|--------|----------|--------------|----------|
| `__init__` | 19.3% | 67/347 | **P0** -- constructor wires everything |
| `run_conversation` | 2.4% | 20/826 | **P0** -- main agent loop |
| `chat` | 83.3% | 10/12 | P2 -- thin wrapper over run_conversation |
| `interrupt` | 66.7% | 24/36 | P1 -- tested via test_interrupt.py |
| `clear_interrupt` | 40.0% | 2/5 | P1 |
| `is_interrupted` | 66.7% | 2/3 | P1 |
| `flush_memories` | 14.3% | 14/98 | P1 -- memory lifecycle |

#### INTERNAL METHODS (needed for decomposition contracts)

| Method | Coverage | Lines Covered | Priority |
|--------|----------|--------------|----------|
| `_execute_tool_calls` | 9.7% | 20/207 | **P0** -- core tool dispatch loop |
| `_build_system_prompt` | 25.0% | 16/64 | **P0** -- prompt assembly |
| `_build_api_kwargs` | 4.3% | 2/47 | **P0** -- API call construction |
| `_build_assistant_message` | 15.0% | 6/40 | P1 -- response parsing |
| `_interruptible_api_call` | 25.0% | 9/36 | P1 -- API call with interrupt |
| `_compress_context` | 20.0% | 7/35 | P1 -- context window management |
| `_handle_max_iterations` | 3.3% | 2/61 | P1 -- iteration cap logic |
| `_persist_session` | 62.5% | 5/8 | P2 -- delegation to sub-methods |
| `_log_msg_to_db` | 7.7% | 2/26 | P1 -- message persistence |
| `_flush_messages_to_session_db` | 21.2% | 7/33 | P1 -- batch persistence |
| `_has_content_after_think_block` | 61.9% | 13/21 | P2 |
| `_strip_think_blocks` | 40.0% | 2/5 | P2 |
| `_extract_reasoning` | 35.7% | 15/42 | P1 -- reasoning extraction |
| `_cleanup_task_resources` | 16.7% | 2/12 | P2 |
| `_get_messages_up_to_last_assistant` | 46.7% | 14/30 | P1 |
| `_format_tools_for_system_message` | 30.4% | 7/23 | P1 |
| `_convert_to_trajectory_format` | 7.6% | 12/157 | P2 -- trajectory serialization |
| `_save_trajectory` | 64.3% | 9/14 | P2 |
| `_mask_api_key_for_logs` | 16.7% | 1/6 | P3 |
| `_dump_api_request_debug` | 17.3% | 14/81 | P3 -- debug logging |
| `_clean_session_content` | 20.0% | 2/10 | P3 -- static method |
| `_save_session_log` | 28.6% | 12/42 | P2 |
| `_hydrate_todo_store` | 29.0% | 9/31 | P1 -- todo state restoration |
| `_invalidate_system_prompt` | 70.0% | 7/10 | P2 |

#### TOP-LEVEL FUNCTION

| Function | Coverage | Lines Covered | Priority |
|----------|----------|--------------|----------|
| `main` | 17.0% | 36/212 | P3 -- CLI entry point |

---

### hermes_state.py -- SessionDB (0% total)

**Entire class is untested.** All 19 methods at 0% coverage.

| Method | Lines | Priority | Notes |
|--------|-------|----------|-------|
| `__init__` | 14 | **P0** | SQLite connection + schema init |
| `_init_schema` | 29 | **P0** | Schema creation + migrations |
| `create_session` | 28 | **P0** | Session lifecycle |
| `end_session` | 7 | **P0** | Session lifecycle |
| `append_message` | 51 | **P0** | Core message storage |
| `get_messages` | 17 | **P0** | Message retrieval |
| `get_messages_as_conversation` | 24 | **P0** | Conversation format |
| `close` | 5 | P1 | Resource cleanup |
| `update_system_prompt` | 7 | P1 | Session update |
| `update_token_counts` | 12 | P1 | Token tracking |
| `get_session` | 7 | P1 | Session lookup |
| `search_messages` | 86 | P1 | FTS5 search |
| `search_sessions` | 18 | P1 | Session listing |
| `session_count` | 9 | P2 | Analytics |
| `message_count` | 9 | P2 | Analytics |
| `export_session` | 7 | P2 | Export |
| `export_all` | 11 | P2 | Export |
| `delete_session` | 11 | P2 | Cleanup |
| `prune_sessions` | 27 | P2 | Cleanup |

---

### model_tools.py (29% total)

| Function | Coverage | Lines Covered | Priority |
|----------|----------|--------------|----------|
| `_run_async` | 54.2% | 13/24 | P1 -- sync/async bridge |
| `_discover_tools` | 94.1% | 32/34 | P3 -- already well-covered |
| `get_tool_definitions` | 25.3% | 19/75 | **P0** -- toolset filtering logic |
| `handle_function_call` | 46.2% | 18/39 | **P0** -- main dispatcher |
| `get_all_tool_names` | 66.7% | 2/3 | P2 -- thin wrapper |
| `get_toolset_for_tool` | 66.7% | 2/3 | P2 -- thin wrapper |
| `get_available_toolsets` | 66.7% | 2/3 | P2 -- thin wrapper |
| `check_toolset_requirements` | 66.7% | 2/3 | P2 -- thin wrapper |
| `check_tool_availability` | 66.7% | 2/3 | P2 -- thin wrapper |

**Note:** The thin wrappers (get_all_tool_names, etc.) delegate to ToolRegistry methods that ARE tested in test_registry.py. The uncovered line in each is the `return` statement body -- the actual logic is covered indirectly.

---

### agent/context_compressor.py -- ContextCompressor (13% total)

| Method | Coverage | Lines Covered | Priority |
|--------|----------|--------------|----------|
| `__init__` | 36.0% | 9/25 | **P0** -- initialization |
| `update_from_response` | 40.0% | 2/5 | P1 -- token tracking |
| `should_compress` | 50.0% | 2/4 | P1 -- threshold check |
| `should_compress_preflight` | 50.0% | 2/4 | P1 -- estimate-based check |
| `get_status` | 88.9% | 8/9 | P2 -- status dict |
| `_generate_summary` | 4.0% | 2/50 | **P0** -- core summarization |
| `compress` | 7.5% | 5/67 | **P0** -- core compression |

---

### agent/prompt_builder.py (12% total)

| Function | Coverage | Lines Covered | Priority |
|----------|----------|--------------|----------|
| `_scan_context_content` | 10.5% | 2/19 | **P0** -- security scanning |
| `_read_skill_description` | 12.5% | 2/16 | P1 |
| `build_skills_system_prompt` | 24.3% | 18/74 | P1 -- skill prompt assembly |
| `_truncate_content` | 20.0% | 2/10 | P2 |
| `build_context_files_prompt` | 5.9% | 6/101 | **P0** -- context file injection |

---

### agent/prompt_caching.py (12% total)

| Function | Coverage | Lines Covered | Priority |
|----------|----------|--------------|----------|
| `_apply_cache_marker` | 9.5% | 2/21 | **P0** -- cache marker injection |
| `apply_anthropic_cache_control` | 35.5% | 11/31 | **P0** -- caching strategy |

---

### tools/registry.py -- ToolRegistry (69% total)

| Method | Coverage | Lines Covered | Priority |
|--------|----------|--------------|----------|
| `__init__` | 100% | 3/3 | -- |
| `register` | 100% | 24/24 | -- |
| `get_definitions` | 78.3% | 18/23 | P2 -- quiet mode + exception paths |
| `dispatch` | 88.9% | 16/18 | P2 -- async dispatch path |
| `get_all_tool_names` | 100% | 3/3 | -- |
| `get_toolset_for_tool` | 50.0% | 2/4 | P3 -- None return path |
| `get_tool_to_toolset_map` | 100% | 3/3 | -- |
| `is_toolset_available` | 100% | 4/4 | -- |
| `check_toolset_requirements` | 100% | 4/4 | -- |
| `get_available_toolsets` | 11.1% | 2/18 | P1 -- UI metadata |
| `get_toolset_requirements` | 100% | 19/19 | -- |
| `check_tool_availability` | 10.5% | 2/19 | P1 -- availability reporting |

---

### tools/__init__.py (90% total)

| Function | Coverage | Lines Covered | Priority |
|----------|----------|--------------|----------|
| `check_file_requirements` | 0% | 0/2 | P3 -- trivial delegation |

---

## Priority Rankings for Test Coverage

### P0 -- Must Cover Before Refactoring (behavioral contracts at risk)

1. **hermes_state.py** -- SessionDB (all methods) -- 0% coverage, standalone module, easy to test
2. **run_agent.py** -- `run_conversation` -- main agent loop, 826 lines at 2.4%
3. **run_agent.py** -- `_execute_tool_calls` -- tool dispatch, 207 lines at 9.7%
4. **run_agent.py** -- `__init__` -- constructor wiring, 347 lines at 19.3%
5. **run_agent.py** -- `_build_system_prompt` -- prompt assembly, 64 lines at 25%
6. **run_agent.py** -- `_build_api_kwargs` -- API call construction, 47 lines at 4.3%
7. **agent/prompt_caching.py** -- both functions -- pure functions, easy to test
8. **agent/context_compressor.py** -- `_generate_summary` + `compress`
9. **agent/prompt_builder.py** -- `_scan_context_content` + `build_context_files_prompt`
10. **model_tools.py** -- `get_tool_definitions` + `handle_function_call`

### P1 -- Important for Decomposition Safety

- run_agent.py: `_build_assistant_message`, `_interruptible_api_call`, `_extract_reasoning`, `flush_memories`, `_compress_context`, `_handle_max_iterations`, `_log_msg_to_db`, `_flush_messages_to_session_db`, `_hydrate_todo_store`, `_get_messages_up_to_last_assistant`, `_format_tools_for_system_message`
- agent/context_compressor.py: `update_from_response`, `should_compress`, `should_compress_preflight`
- agent/prompt_builder.py: `_read_skill_description`, `build_skills_system_prompt`
- tools/registry.py: `get_available_toolsets`, `check_tool_availability`
- model_tools.py: `_run_async`

### P2 -- Nice to Have

- run_agent.py: `chat`, `_persist_session`, `_save_trajectory`, `_save_session_log`, `_invalidate_system_prompt`, `_convert_to_trajectory_format`, `_cleanup_task_resources`, `_has_content_after_think_block`, `_strip_think_blocks`
- hermes_state.py: analytics + export + cleanup methods
- tools/registry.py: edge-case paths in `get_definitions`, `dispatch`

### P3 -- Low Priority

- run_agent.py: `_mask_api_key_for_logs`, `_dump_api_request_debug`, `_clean_session_content`, `main`
- tools/__init__.py: `check_file_requirements`
- model_tools.py: thin wrapper functions (covered indirectly via registry tests)
