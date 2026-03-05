"""Tests for parse_real_sessions and extract_memories modules.

TDD: Written before implementation to define expected behavior.
"""

import json
import os
import sys
import pytest
from datetime import datetime, timezone

# Ensure the parent directory is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parse_real_sessions import (
    ConversationTurn,
    derive_project_name,
    parse_session_file,
    parse_all_sessions,
)
from extract_memories import (
    MemoryFact,
    _is_mostly_code,
    _is_action_announcement,
    _is_acknowledgment,
    _classify_fact,
    extract_facts_from_turn,
    extract_facts,
)


# ---------------------------------------------------------------------------
# parse_real_sessions tests
# ---------------------------------------------------------------------------


class TestDeriveProjectName:
    def test_nel_project(self):
        assert derive_project_name("-Users-cosimo-Documents-Senti-NEL-") == "NEL"

    def test_hermes_agent(self):
        assert derive_project_name("-Users-cosimo--hermes-hermes-agent") == "hermes-agent"

    def test_hermes_nlcdm(self):
        assert (
            derive_project_name(
                "-Users-cosimo--hermes-hermes-agent-proofs-hermes-NLCDM-python"
            )
            == "hermes-NLCDM"
        )

    def test_co_project(self):
        assert derive_project_name("-Users-cosimo-Documents-CO") == "CO"

    def test_clinic_dashboard(self):
        assert (
            derive_project_name("-Users-cosimo-Documents-dopa-clinicDashboard")
            == "clinicDashboard"
        )

    def test_clinic_dashboard_rome(self):
        assert (
            derive_project_name(
                "-Users-cosimo-Documents-dopa-clinicDashboard--conductor-rome"
            )
            == "clinicDashboard-rome"
        )

    def test_unknown_project(self):
        # Should extract last meaningful segment
        name = derive_project_name("-Users-cosimo-Desktop-MyProject")
        assert name == "MyProject"

    def test_unknown_nested(self):
        name = derive_project_name("-Users-cosimo-Desktop--Projects-ameen")
        assert name == "ameen"


class TestParseSessionFile:
    def test_parse_real_session_file(self):
        """Use an actual session file to test parsing."""
        nel_dir = os.path.expanduser(
            "~/.claude/projects/-Users-cosimo-Documents-Senti-NEL-/"
        )
        if not os.path.isdir(nel_dir):
            pytest.skip("NEL sessions not available")
        jsonl_files = sorted(
            [f for f in os.listdir(nel_dir) if f.endswith(".jsonl")]
        )
        if not jsonl_files:
            pytest.skip("No JSONL files in NEL directory")
        path = os.path.join(nel_dir, jsonl_files[0])
        turns = parse_session_file(path)
        # Should return list of ConversationTurn
        assert isinstance(turns, list)
        for t in turns:
            assert isinstance(t, ConversationTurn)
            assert isinstance(t.timestamp, datetime)
            assert t.project == ""  # parse_session_file doesn't set project
            # Text content should be non-trivial (>20 chars filter)
            assert len(t.user_text) >= 20 or len(t.assistant_text) >= 20

    def test_filters_meta_messages(self, tmp_path):
        """Meta messages (isMeta or command-name) should be filtered."""
        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": "<command-name>/clear</command-name>"
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "This is a real substantive response from the assistant.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        # The user message should be filtered (command), so this pair may not
        # form a turn or the turn has empty user_text
        for t in turns:
            assert "<command-name>" not in t.user_text

    def test_filters_local_command_caveat(self, tmp_path):
        """User messages with <local-command tags should be filtered."""
        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": "<local-command-caveat>Caveat: stuff</local-command-caveat>"
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                    "isMeta": True,
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "This is a real substantive response from the assistant.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        for t in turns:
            assert "<local-command" not in t.user_text

    def test_filters_tool_results(self, tmp_path):
        """User messages with tool_result blocks should be filtered."""
        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": [
                            {"type": "tool_result", "content": "output"}
                        ]
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Got it, the tool result shows the build passed successfully.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        for t in turns:
            assert "tool_result" not in t.user_text

    def test_filters_internal_user_type(self, tmp_path):
        """User messages with userType=internal should be filtered."""
        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "file contents here that are long enough",
                            }
                        ]
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                    "userType": "internal",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Got it, the tool result shows the build passed successfully.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        for t in turns:
            assert "file contents" not in t.user_text

    def test_filters_short_messages(self, tmp_path):
        """Messages under 20 chars should be filtered."""
        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {"content": "ok"},
                    "uuid": "a",
                    "sessionId": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [{"type": "text", "text": "ok"}]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        assert len(turns) == 0  # Both too short

    def test_extracts_text_from_content_blocks(self, tmp_path):
        """Should extract text from content block lists, skipping tool_use."""
        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": "Tell me about the architecture of this system in detail"
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "The system uses a layered architecture with clear separation.",
                            },
                            {
                                "type": "tool_use",
                                "name": "Read",
                                "input": {"path": "/foo"},
                            },
                            {
                                "type": "text",
                                "text": " Each layer has defined interfaces.",
                            },
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        assert len(turns) == 1
        # Should contain both text blocks, not tool_use
        assert "layered architecture" in turns[0].assistant_text
        assert "defined interfaces" in turns[0].assistant_text
        # tool_use content should not appear
        assert '{"path":' not in turns[0].assistant_text

    def test_handles_empty_file(self, tmp_path):
        """Empty file should return empty list."""
        session_file = tmp_path / "empty.jsonl"
        session_file.write_text("")
        turns = parse_session_file(str(session_file))
        assert turns == []

    def test_handles_malformed_json(self, tmp_path):
        """Malformed JSON lines should be skipped gracefully."""
        session_file = tmp_path / "bad.jsonl"
        lines = [
            "this is not json",
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": "Tell me about the architecture of this system in detail"
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "The system uses a layered architecture with clear separation of concerns.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        # Should parse the valid messages despite the malformed line
        assert len(turns) >= 1

    def test_groups_user_assistant_pairs(self, tmp_path):
        """Consecutive user+assistant messages should form a single turn."""
        session_file = tmp_path / "pairs.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": "What is the purpose of this repository and its main components?"
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                    "cwd": "/home/user/project",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "This repository implements a memory system with several core components.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:01:00Z",
                    "message": {
                        "content": "How does the dream consolidation algorithm work under the hood?"
                    },
                    "uuid": "c",
                    "sessionId": "s1",
                    "cwd": "/home/user/project",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:01:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "The dream consolidation uses Hebbian learning to strengthen memory traces.",
                            }
                        ]
                    },
                    "uuid": "d",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        assert len(turns) == 2
        assert "purpose" in turns[0].user_text
        assert "memory system" in turns[0].assistant_text
        assert "dream" in turns[1].user_text
        assert "Hebbian" in turns[1].assistant_text

    def test_user_without_assistant(self, tmp_path):
        """A user message with no following assistant becomes a turn with empty assistant_text."""
        session_file = tmp_path / "orphan.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": "What is the purpose of this repository and its main components?"
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                    "cwd": "/home/user/project",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        assert len(turns) == 1
        assert turns[0].assistant_text == ""

    def test_skips_non_user_assistant_types(self, tmp_path):
        """Should skip progress, file-history-snapshot, queue-operation, system types."""
        session_file = tmp_path / "mixed.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "file-history-snapshot",
                    "messageId": "x",
                    "snapshot": {},
                }
            ),
            json.dumps(
                {
                    "type": "progress",
                    "data": {"type": "hook_progress"},
                    "uuid": "p1",
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": "Tell me about the architecture of this system in detail"
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:02Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "The system uses a layered architecture with clear separation of concerns.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "system",
                    "timestamp": "2026-01-01T00:00:03Z",
                    "message": {"content": "system message"},
                    "uuid": "c",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        assert len(turns) == 1
        assert "architecture" in turns[0].user_text

    def test_session_id_from_filename(self, tmp_path):
        """session_id should be derived from the JSONL filename."""
        session_file = tmp_path / "abc-def-123.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": "Tell me about the architecture of this system in detail"
                    },
                    "uuid": "a",
                    "sessionId": "abc-def-123",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "The system uses a layered architecture with clear separation of concerns.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "abc-def-123",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        assert len(turns) == 1
        assert turns[0].session_id == "abc-def-123"

    def test_cwd_from_message(self, tmp_path):
        """cwd should come from the top-level message object."""
        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": "Tell me about the architecture of this system in detail"
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                    "cwd": "/home/user/my-project",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "The system uses a layered architecture with clear separation of concerns.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        assert turns[0].cwd == "/home/user/my-project"

    def test_timestamp_parsing(self, tmp_path):
        """Timestamps with Z suffix should parse correctly."""
        session_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-02-06T19:34:34.397Z",
                    "message": {
                        "content": "Tell me about the architecture of this system in detail"
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-02-06T19:34:35.000Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "The system uses a layered architecture with clear separation of concerns.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        assert turns[0].timestamp.year == 2026
        assert turns[0].timestamp.month == 2
        assert turns[0].timestamp.day == 6
        assert turns[0].timestamp.tzinfo is not None

    def test_filters_thinking_blocks(self, tmp_path):
        """Assistant thinking blocks should not appear in assistant_text."""
        session_file = tmp_path / "thinking.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": "Tell me about the architecture of this system in detail"
                    },
                    "uuid": "a",
                    "sessionId": "s1",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "Let me think about this...",
                            },
                            {
                                "type": "text",
                                "text": "The system uses a layered architecture with clear separation of concerns.",
                            },
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "s1",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_session_file(str(session_file))
        assert len(turns) == 1
        assert "Let me think" not in turns[0].assistant_text
        assert "layered architecture" in turns[0].assistant_text


class TestParseAllSessions:
    def test_parse_with_project_filter(self):
        """Should filter to specific projects."""
        nel_dir = os.path.expanduser(
            "~/.claude/projects/-Users-cosimo-Documents-Senti-NEL-/"
        )
        if not os.path.isdir(nel_dir):
            pytest.skip("NEL sessions not available")
        turns = parse_all_sessions(projects=["NEL"])
        assert len(turns) > 0
        assert all(t.project == "NEL" for t in turns)

    def test_chronological_sort(self):
        """Turns should be sorted by timestamp."""
        co_dir = os.path.expanduser(
            "~/.claude/projects/-Users-cosimo-Documents-CO/"
        )
        if not os.path.isdir(co_dir):
            pytest.skip("CO sessions not available")
        turns = parse_all_sessions(projects=["CO"])
        if len(turns) > 1:
            for i in range(len(turns) - 1):
                assert turns[i].timestamp <= turns[i + 1].timestamp

    def test_custom_projects_dir(self, tmp_path):
        """Should accept a custom projects directory."""
        proj_dir = tmp_path / "-Users-cosimo-Documents-TestProj"
        proj_dir.mkdir()
        session_file = proj_dir / "sess-001.jsonl"
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {
                        "content": "Tell me about the architecture of this system in detail"
                    },
                    "uuid": "a",
                    "sessionId": "sess-001",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "The system uses a layered architecture with clear separation of concerns.",
                            }
                        ]
                    },
                    "uuid": "b",
                    "sessionId": "sess-001",
                }
            ),
        ]
        session_file.write_text("\n".join(lines))
        turns = parse_all_sessions(projects_dir=str(tmp_path))
        assert len(turns) == 1
        assert turns[0].project == "TestProj"


# ---------------------------------------------------------------------------
# extract_memories tests
# ---------------------------------------------------------------------------


class TestIsMostlyCode:
    def test_code_block(self):
        text = "```python\ndef foo():\n    pass\n```"
        assert _is_mostly_code(text) is True

    def test_mixed_text(self):
        text = "This is an explanation of the code. It does several things including parsing."
        assert _is_mostly_code(text) is False

    def test_code_with_explanation(self):
        text = "Here's the fix:\n```python\nx = 1\n```\nThis sets x to 1 which resolves the issue by initializing."
        assert _is_mostly_code(text) is False  # <80% code

    def test_empty_string(self):
        assert _is_mostly_code("") is False

    def test_multiple_code_blocks(self):
        text = "```python\ndef foo():\n    pass\n```\n```python\ndef bar():\n    return 1\n```"
        assert _is_mostly_code(text) is True


class TestIsActionAnnouncement:
    def test_let_me(self):
        assert _is_action_announcement("Let me read that file for you") is True

    def test_ill(self):
        assert _is_action_announcement("I'll check the test results now") is True

    def test_now_i(self):
        assert _is_action_announcement("Now I will run the tests") is True

    def test_im_going_to(self):
        assert _is_action_announcement("I'm going to check the logs") is True

    def test_first_let_me(self):
        assert (
            _is_action_announcement("First, let me look at the error") is True
        )

    def test_heres_what_i(self):
        assert (
            _is_action_announcement("Here's what I found in the logs") is True
        )

    def test_substantive(self):
        assert (
            _is_action_announcement(
                "The architecture uses a layered design"
            )
            is False
        )

    def test_case_insensitive(self):
        assert _is_action_announcement("let me check that for you") is True


class TestIsAcknowledgment:
    def test_ok(self):
        assert _is_acknowledgment("ok") is True

    def test_proceed(self):
        assert _is_acknowledgment("proceed") is True

    def test_thanks(self):
        assert _is_acknowledgment("thanks") is True

    def test_got_it(self):
        assert _is_acknowledgment("got it") is True

    def test_sure(self):
        assert _is_acknowledgment("sure") is True

    def test_yes(self):
        assert _is_acknowledgment("yes") is True

    def test_y(self):
        assert _is_acknowledgment("y") is True

    def test_go_ahead(self):
        assert _is_acknowledgment("go ahead") is True

    def test_substantive(self):
        assert (
            _is_acknowledgment("I think we should use PostgreSQL") is False
        )

    def test_case_insensitive(self):
        assert _is_acknowledgment("OK") is True
        assert _is_acknowledgment("Thanks") is True

    def test_with_whitespace(self):
        assert _is_acknowledgment("  ok  ") is True


class TestClassifyFact:
    def test_decision(self):
        assert (
            _classify_fact("We decided to use Redis for caching") == "decision"
        )

    def test_should_use(self):
        assert (
            _classify_fact("We should use PostgreSQL for persistence")
            == "decision"
        )

    def test_chose(self):
        assert (
            _classify_fact("We chose the microservices approach") == "decision"
        )

    def test_explanation(self):
        assert (
            _classify_fact(
                "The root cause was a race condition in the lock"
            )
            == "explanation"
        )

    def test_because(self):
        assert (
            _classify_fact("This works because the cache is invalidated")
            == "explanation"
        )

    def test_gotcha(self):
        assert (
            _classify_fact("Gotcha: the API rate limits at 100 req/min")
            == "gotcha"
        )

    def test_careful(self):
        assert (
            _classify_fact("Be careful with the thread pool size")
            == "gotcha"
        )

    def test_important(self):
        assert (
            _classify_fact("Important: always close the connection")
            == "gotcha"
        )

    def test_architecture(self):
        assert (
            _classify_fact("The architecture uses event sourcing")
            == "architecture"
        )

    def test_design_pattern(self):
        assert (
            _classify_fact("This pattern is called the saga pattern")
            == "architecture"
        )

    def test_debug(self):
        assert (
            _classify_fact("The bug was in the serialization layer") == "debug"
        )

    def test_error(self):
        assert (
            _classify_fact("The error occurs when the buffer overflows")
            == "debug"
        )

    def test_general(self):
        assert (
            _classify_fact("The system has 4 main components") == "general"
        )


class TestExtractFacts:
    def test_extracts_from_assistant(self):
        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="How does the auth system work?",
            assistant_text="The authentication system uses JWT tokens with RSA-256 signing. Tokens expire after 24 hours and are refreshed automatically.",
            project="NEL",
            session_id="s1",
            cwd="/foo",
        )
        facts = extract_facts_from_turn(turn)
        assert len(facts) > 0
        assert any("JWT" in f.text for f in facts)

    def test_skips_action_announcements(self):
        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Fix the bug in the authentication module",
            assistant_text="Let me read the file and check the error.\n\nI'll look at the test output now.",
            project="NEL",
            session_id="s1",
            cwd="/foo",
        )
        facts = extract_facts_from_turn(turn)
        assert all("Let me" not in f.text for f in facts)

    def test_skips_code_heavy_paragraphs(self):
        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Show me the code",
            assistant_text="```python\ndef foo():\n    return bar()\n```",
            project="NEL",
            session_id="s1",
            cwd="/foo",
        )
        facts = extract_facts_from_turn(turn)
        assert len(facts) == 0

    def test_keeps_user_questions(self):
        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="I think we should use PostgreSQL for the main database because it handles JSON well",
            assistant_text="That's a good choice for several architectural reasons including JSONB support.",
            project="NEL",
            session_id="s1",
            cwd="/foo",
        )
        facts = extract_facts_from_turn(turn)
        # Should have facts from both user and assistant
        sources = {f.source for f in facts}
        assert "user" in sources

    def test_extract_facts_batch(self):
        """Test extracting from multiple turns."""
        turns = [
            ConversationTurn(
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                user_text="How does caching work here?",
                assistant_text="The system uses Redis with a 5-minute TTL for all cached responses.",
                project="NEL",
                session_id="s1",
                cwd="/foo",
            ),
            ConversationTurn(
                timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
                user_text="What about the database schema?",
                assistant_text="The database has three main tables: users, sessions, and events. Each uses UUID primary keys.",
                project="CO",
                session_id="s2",
                cwd="/bar",
            ),
        ]
        facts = extract_facts(turns)
        assert len(facts) >= 2
        projects = {f.project for f in facts}
        assert "NEL" in projects
        assert "CO" in projects

    def test_skips_short_paragraphs(self):
        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="What does this do?",
            assistant_text="It works.\n\nThe system implements a comprehensive memory consolidation pipeline that processes experiences.",
            project="NEL",
            session_id="s1",
            cwd="/foo",
        )
        facts = extract_facts_from_turn(turn)
        # "It works." is <50 chars, should be skipped
        assert all("It works." not in f.text for f in facts)

    def test_skips_long_paragraphs(self):
        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Explain everything.",
            assistant_text="x " * 260,  # >500 chars
            project="NEL",
            session_id="s1",
            cwd="/foo",
        )
        facts = extract_facts_from_turn(turn)
        # All assistant paragraphs >500 chars should be skipped
        for f in facts:
            if f.source == "assistant":
                assert len(f.text) <= 500

    def test_fact_metadata(self):
        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="How does the auth system work?",
            assistant_text="The authentication system uses JWT tokens with RSA-256 signing. Tokens expire after 24 hours.",
            project="NEL",
            session_id="sess-abc",
            cwd="/foo",
        )
        facts = extract_facts_from_turn(turn)
        for f in facts:
            assert isinstance(f, MemoryFact)
            assert f.timestamp == datetime(2026, 1, 1, tzinfo=timezone.utc)
            assert f.project == "NEL"
            assert f.session_id == "sess-abc"
            assert f.fact_type in (
                "decision",
                "explanation",
                "gotcha",
                "architecture",
                "debug",
                "general",
                "self_correction",
                "finding",
                "hypothesis",
            )
            assert f.source in ("user", "assistant", "thinking")
            assert f.layer in ("user_knowledge", "agent_meta")


class TestExtractFactsOnRealData:
    def test_extract_from_real_sessions(self):
        """Integration test: parse real sessions, extract facts, verify reasonable output."""
        co_dir = os.path.expanduser(
            "~/.claude/projects/-Users-cosimo-Documents-CO/"
        )
        if not os.path.isdir(co_dir):
            pytest.skip("CO sessions not available")
        from parse_real_sessions import parse_all_sessions

        turns = parse_all_sessions(projects=["CO"])
        if len(turns) == 0:
            pytest.skip("No turns parsed from CO")
        facts = extract_facts(turns)
        # Should have a reasonable number of facts
        assert len(facts) > 10, f"Expected >10 facts from CO, got {len(facts)}"
        # All facts should have required fields
        for f in facts:
            assert isinstance(f, MemoryFact)
            assert len(f.text) > 0
            assert f.project == "CO"
            assert f.fact_type in (
                "decision",
                "explanation",
                "gotcha",
                "architecture",
                "debug",
                "general",
                # Layer 2 metacognition types
                "self_correction",
                "finding",
                "hypothesis",
                "reasoning_chain",
            )
            assert f.layer in ("user_knowledge", "agent_meta")


# ---------------------------------------------------------------------------
# build_eval_dataset tests (Phase 2)
# ---------------------------------------------------------------------------

import numpy as np


class TestGroupFactsByDate:
    def test_groups_by_date(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _group_facts_by_date

        facts = [
            MemoryFact(
                "fact1",
                datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "fact2",
                datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "fact3",
                datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc),
                "P1",
                "s2",
                "assistant",
                "general",
            ),
        ]
        groups = _group_facts_by_date(facts)
        assert "2026-01-01" in groups
        assert "2026-01-02" in groups
        assert len(groups["2026-01-01"]) == 2
        assert len(groups["2026-01-02"]) == 1

    def test_empty_facts(self):
        from build_eval_dataset import _group_facts_by_date

        groups = _group_facts_by_date([])
        assert groups == {}

    def test_chronological_within_day(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _group_facts_by_date

        facts = [
            MemoryFact(
                "later",
                datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "earlier",
                datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            ),
        ]
        groups = _group_facts_by_date(facts)
        # Should be sorted chronologically within day
        assert groups["2026-01-01"][0].text == "earlier"
        assert groups["2026-01-01"][1].text == "later"


class TestDetectDreamBoundaries:
    def test_overnight_gap(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _group_facts_by_date, _detect_dream_boundaries

        facts = [
            MemoryFact(
                "f1",
                datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "f2",
                datetime(2026, 1, 1, 23, 0, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            ),
        ]
        groups = _group_facts_by_date(facts)
        boundaries = _detect_dream_boundaries(groups)
        assert len(boundaries) == 0

    def test_multi_day_gap(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _group_facts_by_date, _detect_dream_boundaries

        facts = [
            MemoryFact(
                "f1",
                datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "f2",
                datetime(2026, 1, 2, 20, 0, tzinfo=timezone.utc),
                "P1",
                "s2",
                "assistant",
                "general",
            ),
        ]
        groups = _group_facts_by_date(facts)
        boundaries = _detect_dream_boundaries(groups)
        assert len(boundaries) >= 1

    def test_no_boundary_short_gap(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _group_facts_by_date, _detect_dream_boundaries

        facts = [
            MemoryFact(
                "f1",
                datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "f2",
                datetime(2026, 1, 2, 2, 0, tzinfo=timezone.utc),
                "P1",
                "s2",
                "assistant",
                "general",
            ),
        ]
        groups = _group_facts_by_date(facts)
        boundaries = _detect_dream_boundaries(groups)
        assert len(boundaries) == 0

    def test_custom_min_gap(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _group_facts_by_date, _detect_dream_boundaries

        facts = [
            MemoryFact(
                "f1",
                datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "f2",
                datetime(2026, 1, 2, 2, 0, tzinfo=timezone.utc),
                "P1",
                "s2",
                "assistant",
                "general",
            ),
        ]
        groups = _group_facts_by_date(facts)
        boundaries = _detect_dream_boundaries(groups, min_gap_hours=4.0)
        assert len(boundaries) >= 1


class TestGenerateSingleProjectQuestions:
    def test_generates_questions(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_single_project_questions

        facts = [
            MemoryFact(
                f"Important fact number {i} about testing and software design",
                datetime(2026, 1, 1 + i // 10, i % 24, 0, tzinfo=timezone.utc),
                "NEL",
                f"s{i}",
                "assistant",
                "general",
            )
            for i in range(50)
        ]
        questions, held_out = _generate_single_project_questions(
            facts, n_questions=10
        )
        assert len(questions) <= 10
        assert len(held_out) > 0
        for q in questions:
            assert q.category == "single_project"
            assert "NEL" in q.projects

    def test_held_out_not_empty(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_single_project_questions

        facts = [
            MemoryFact(
                f"Fact {i} about system architecture and design patterns",
                datetime(2026, 1, 1, i, 0, tzinfo=timezone.utc),
                "CO",
                f"s{i}",
                "assistant",
                "general",
            )
            for i in range(20)
        ]
        questions, held_out = _generate_single_project_questions(
            facts, n_questions=5
        )
        assert len(held_out) > 0
        for t in held_out:
            assert isinstance(t, str)

    def test_difficulty_assignment(self):
        from datetime import datetime, timezone, timedelta
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_single_project_questions

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        facts = [
            MemoryFact(
                f"Architectural decision number {i} about microservices",
                base + timedelta(days=i),
                "P1",
                f"s{i}",
                "assistant",
                "general",
            )
            for i in range(35)
        ]
        questions, _ = _generate_single_project_questions(
            facts, n_questions=10
        )
        assert len(questions) > 0
        for q in questions:
            assert q.difficulty in ("easy", "medium", "hard")


class TestGenerateCrossDomainQuestions:
    def test_finds_cross_domain(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_cross_domain_questions

        facts = [
            MemoryFact(
                "We use CMA-ES optimizer for parameter tuning",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                "hermes-agent",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "The CMA-ES optimizer converges faster than grid search",
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                "CO",
                "s2",
                "assistant",
                "general",
            ),
            MemoryFact(
                "Unrelated fact about UI design and responsiveness",
                datetime(2026, 1, 3, tzinfo=timezone.utc),
                "clinicDashboard",
                "s3",
                "assistant",
                "general",
            ),
        ]
        questions = _generate_cross_domain_questions(facts, n_questions=5)
        if len(questions) > 0:
            assert any(q.category == "cross_domain" for q in questions)
            for q in questions:
                assert len(q.projects) >= 2

    def test_all_hard_difficulty(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_cross_domain_questions

        facts = [
            MemoryFact(
                "Lean proof tactic for theorem verification",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                "hermes-agent",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "Lean mathlib provides proof automation tactics",
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                "CO",
                "s2",
                "assistant",
                "general",
            ),
        ]
        questions = _generate_cross_domain_questions(facts, n_questions=5)
        for q in questions:
            assert q.difficulty == "hard"

    def test_no_cross_domain_when_single_project(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_cross_domain_questions

        facts = [
            MemoryFact(
                "CMA-ES optimizer for parameter tuning in the system",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                "hermes-agent",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "Another CMA-ES fact about convergence in the pipeline",
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                "hermes-agent",
                "s2",
                "assistant",
                "general",
            ),
        ]
        questions = _generate_cross_domain_questions(facts, n_questions=5)
        assert len(questions) == 0


class TestGenerateTemporalQuestions:
    def test_generates_from_early_facts(self):
        from datetime import datetime, timezone, timedelta
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_temporal_questions

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        facts = [
            MemoryFact(
                f"Temporal fact number {i} about system behavior over time",
                base + timedelta(days=i),
                "P1",
                f"s{i}",
                "assistant",
                "general",
            )
            for i in range(50)
        ]
        questions = _generate_temporal_questions(facts, n_questions=10)
        assert len(questions) <= 10
        for q in questions:
            assert q.category == "temporal"
            assert q.difficulty == "hard"


class TestBuildDataset:
    def test_build_synthetic(self):
        from datetime import datetime, timezone, timedelta
        from extract_memories import MemoryFact
        from build_eval_dataset import build_dataset

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        facts = [
            MemoryFact(
                f"Synthetic fact {i} describing an architectural decision",
                base + timedelta(hours=i * 12),
                "P1" if i % 2 == 0 else "P2",
                f"s{i}",
                "assistant",
                "general",
            )
            for i in range(30)
        ]
        dataset = build_dataset(facts, max_facts_per_session=20)

        assert len(dataset.sessions) > 0
        assert dataset.total_facts > 0
        assert len(dataset.date_range) == 2
        assert len(dataset.projects) > 0

    def test_max_facts_per_session_cap(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import build_dataset

        facts = [
            MemoryFact(
                f"Fact {i} about an important design choice in the system",
                datetime(2026, 1, 1, i % 24, 0, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            )
            for i in range(100)
        ]
        dataset = build_dataset(facts, max_facts_per_session=10)
        for session in dataset.sessions:
            assert len(session.facts) <= 10

    def test_build_from_real_data(self):
        """Integration: build dataset from real CO sessions."""
        from parse_real_sessions import parse_all_sessions
        from extract_memories import extract_facts
        from build_eval_dataset import build_dataset

        turns = parse_all_sessions(projects=["CO"])
        if not turns:
            pytest.skip("CO sessions not available")
        facts = extract_facts(turns)
        if len(facts) < 10:
            pytest.skip("Insufficient CO facts")
        dataset = build_dataset(facts, max_facts_per_session=20)

        assert len(dataset.sessions) > 0
        assert dataset.total_facts > 0
        assert len(dataset.date_range) == 2

    def test_project_filter(self):
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import build_dataset

        facts = [
            MemoryFact(
                "Fact about P1 project architecture and its components",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                "P1",
                "s1",
                "assistant",
                "general",
            ),
            MemoryFact(
                "Fact about P2 project architecture and its components",
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                "P2",
                "s2",
                "assistant",
                "general",
            ),
        ]
        dataset = build_dataset(facts, projects=["P1"])
        for s in dataset.sessions:
            assert s.project == "P1"


# ---------------------------------------------------------------------------
# run_real_eval tests (Phase 2)
# ---------------------------------------------------------------------------


class TestScoreRetrieval:
    def test_full_match(self):
        from run_real_eval import score_retrieval

        retrieved = ["The system uses JWT tokens", "Redis cache expires"]
        expected = ["JWT tokens"]
        assert score_retrieval(retrieved, expected) == 1.0

    def test_no_match(self):
        from run_real_eval import score_retrieval

        retrieved = ["Completely different text here"]
        expected = ["JWT tokens"]
        assert score_retrieval(retrieved, expected) == 0.0

    def test_partial_match(self):
        from run_real_eval import score_retrieval

        retrieved = ["Uses JWT for auth", "Redis cache"]
        expected = ["JWT", "PostgreSQL"]
        assert score_retrieval(retrieved, expected) == 0.5

    def test_case_insensitive(self):
        from run_real_eval import score_retrieval

        assert score_retrieval(["uses jwt TOKENS"], ["jwt tokens"]) == 1.0

    def test_empty_expected(self):
        from run_real_eval import score_retrieval

        assert score_retrieval(["some text"], []) == 0.0

    def test_empty_retrieved(self):
        from run_real_eval import score_retrieval

        assert score_retrieval([], ["something"]) == 0.0

    def test_multiple_expected_all_found(self):
        from run_real_eval import score_retrieval

        retrieved = ["JWT token auth", "Redis cache layer", "PostgreSQL DB"]
        expected = ["JWT", "Redis", "PostgreSQL"]
        assert score_retrieval(retrieved, expected) == 1.0


class TestScoreCrossDomain:
    def test_full_coverage(self):
        from run_real_eval import score_cross_domain

        retrieved = ["fact from NEL", "fact from CO"]
        fact_to_proj = {"fact from NEL": "NEL", "fact from CO": "CO"}
        assert score_cross_domain(retrieved, ["NEL", "CO"], fact_to_proj) == 1.0

    def test_partial_coverage(self):
        from run_real_eval import score_cross_domain

        retrieved = ["fact from NEL"]
        fact_to_proj = {"fact from NEL": "NEL"}
        assert score_cross_domain(retrieved, ["NEL", "CO"], fact_to_proj) == 0.5

    def test_no_coverage(self):
        from run_real_eval import score_cross_domain

        retrieved = ["unknown fact"]
        fact_to_proj = {}
        assert score_cross_domain(retrieved, ["NEL", "CO"], fact_to_proj) == 0.0

    def test_empty_expected_projects(self):
        from run_real_eval import score_cross_domain

        retrieved = ["fact"]
        fact_to_proj = {"fact": "NEL"}
        assert score_cross_domain(retrieved, [], fact_to_proj) == 0.0


class TestEmbedText:
    def test_deterministic(self):
        from run_real_eval import _embed_text

        e1 = _embed_text("hello world", 64)
        e2 = _embed_text("hello world", 64)
        assert np.allclose(e1, e2)

    def test_normalized(self):
        from run_real_eval import _embed_text

        e = _embed_text("test text", 64)
        assert abs(np.linalg.norm(e) - 1.0) < 1e-6

    def test_different_texts(self):
        from run_real_eval import _embed_text

        e1 = _embed_text("hello world", 64)
        e2 = _embed_text("completely different", 64)
        assert not np.allclose(e1, e2)

    def test_correct_dimension(self):
        from run_real_eval import _embed_text

        e = _embed_text("some text", 128)
        assert e.shape == (128,)

    def test_various_dims(self):
        from run_real_eval import _embed_text

        for dim in [16, 32, 64, 128, 256]:
            e = _embed_text("test", dim)
            assert e.shape == (dim,)
            assert abs(np.linalg.norm(e) - 1.0) < 1e-6


class TestRunEvalSingleConfig:
    def test_baseline_config(self):
        """Test running eval with baseline config on small synthetic dataset."""
        from build_eval_dataset import RealEvalDataset, RealSession, RealEvalQuestion
        from run_real_eval import run_eval_single_config, EvalConfig

        dataset = RealEvalDataset(
            sessions=[
                RealSession(
                    date="2026-01-01",
                    day_index=0,
                    project="P1",
                    facts=[
                        "Alpha system uses Redis cache for all lookups",
                        "Beta module handles authentication via JWT tokens",
                    ],
                ),
                RealSession(
                    date="2026-01-02",
                    day_index=1,
                    project="P1",
                    facts=[
                        "Gamma service processes webhooks asynchronously",
                        "Delta layer provides database abstraction for queries",
                    ],
                ),
            ],
            questions=[
                RealEvalQuestion(
                    question="Alpha system uses Redis cache for all lookups",
                    expected_facts=["Redis cache"],
                    category="single_project",
                    projects=["P1"],
                    difficulty="easy",
                ),
            ],
            dream_boundaries=[0],
            date_range=("2026-01-01", "2026-01-02"),
            total_facts=4,
            projects=["P1"],
        )

        config = EvalConfig(name="baseline", dreams_enabled=False, dim=64)
        fact_to_project = {
            f: s.project for s in dataset.sessions for f in s.facts
        }
        result = run_eval_single_config(dataset, config, fact_to_project)

        assert result.config_name == "baseline"
        assert result.n_facts_stored == 4
        assert result.n_questions == 1
        assert 0.0 <= result.overall_recall <= 1.0

    def test_dreams_config(self):
        """Test running eval with dreams enabled."""
        from build_eval_dataset import RealEvalDataset, RealSession, RealEvalQuestion
        from run_real_eval import run_eval_single_config, EvalConfig

        dataset = RealEvalDataset(
            sessions=[
                RealSession(
                    date="2026-01-01",
                    day_index=0,
                    project="P1",
                    facts=[
                        "System architecture uses event sourcing pattern",
                        "Database layer uses PostgreSQL with JSONB columns",
                    ],
                ),
                RealSession(
                    date="2026-01-02",
                    day_index=1,
                    project="P1",
                    facts=[
                        "Cache invalidation uses pub-sub with Redis streams",
                        "API gateway handles rate limiting at edge layer",
                    ],
                ),
            ],
            questions=[
                RealEvalQuestion(
                    question="System architecture uses event sourcing pattern",
                    expected_facts=["event sourcing"],
                    category="single_project",
                    projects=["P1"],
                    difficulty="easy",
                ),
            ],
            dream_boundaries=[0],
            date_range=("2026-01-01", "2026-01-02"),
            total_facts=4,
            projects=["P1"],
        )

        config = EvalConfig(name="with_dreams", dreams_enabled=True, dim=64)
        fact_to_project = {
            f: s.project for s in dataset.sessions for f in s.facts
        }
        result = run_eval_single_config(dataset, config, fact_to_project)

        assert result.config_name == "with_dreams"
        assert result.n_facts_stored >= 0  # Dreams may prune
        assert result.n_dreams >= 1  # Should have dreamed at boundary

    def test_ppr_config(self):
        """Test running eval with PPR blend."""
        from build_eval_dataset import RealEvalDataset, RealSession, RealEvalQuestion
        from run_real_eval import run_eval_single_config, EvalConfig

        dataset = RealEvalDataset(
            sessions=[
                RealSession(
                    date="2026-01-01",
                    day_index=0,
                    project="P1",
                    facts=["PPR test fact about graph-based retrieval methods"],
                ),
            ],
            questions=[
                RealEvalQuestion(
                    question="PPR test fact about graph-based retrieval methods",
                    expected_facts=["graph-based retrieval"],
                    category="single_project",
                    projects=["P1"],
                    difficulty="easy",
                ),
            ],
            dream_boundaries=[],
            date_range=("2026-01-01", "2026-01-01"),
            total_facts=1,
            projects=["P1"],
        )

        config = EvalConfig(
            name="ppr_blend", dreams_enabled=False, ppr_blend_weight=0.3, dim=64
        )
        fact_to_project = {
            f: s.project for s in dataset.sessions for f in s.facts
        }
        result = run_eval_single_config(dataset, config, fact_to_project)

        assert result.config_name == "ppr_blend"
        assert result.n_facts_stored == 1


class TestRunRealEval:
    def test_compares_configs(self):
        """Test that run_real_eval compares multiple configs."""
        from build_eval_dataset import RealEvalDataset, RealSession, RealEvalQuestion
        from run_real_eval import run_real_eval, EvalConfig

        dataset = RealEvalDataset(
            sessions=[
                RealSession(
                    date="2026-01-01",
                    day_index=0,
                    project="P1",
                    facts=[
                        "System uses microservices architecture with gRPC",
                        "Database uses PostgreSQL with read replicas",
                    ],
                ),
                RealSession(
                    date="2026-01-02",
                    day_index=1,
                    project="P1",
                    facts=[
                        "Cache layer uses Redis with 5-min TTL expiry",
                        "Authentication handled by OAuth2 with JWT tokens",
                    ],
                ),
            ],
            questions=[
                RealEvalQuestion(
                    question="System uses microservices architecture with gRPC",
                    expected_facts=["microservices"],
                    category="single_project",
                    projects=["P1"],
                    difficulty="easy",
                ),
            ],
            dream_boundaries=[0],
            date_range=("2026-01-01", "2026-01-02"),
            total_facts=4,
            projects=["P1"],
        )

        configs = [
            EvalConfig(name="no_dreams", dreams_enabled=False, dim=64),
            EvalConfig(name="with_dreams", dreams_enabled=True, dim=64),
        ]

        results = run_real_eval(dataset, configs=configs)
        assert "configs" in results
        assert len(results["configs"]) == 2
        for cfg_result in results["configs"]:
            assert "config_name" in cfg_result
            assert "overall_recall" in cfg_result

    def test_default_configs(self):
        """Test that run_real_eval uses default configs when none provided."""
        from build_eval_dataset import RealEvalDataset, RealSession, RealEvalQuestion
        from run_real_eval import run_real_eval

        dataset = RealEvalDataset(
            sessions=[
                RealSession(
                    date="2026-01-01",
                    day_index=0,
                    project="P1",
                    facts=["Default config test fact about system behavior"],
                ),
            ],
            questions=[
                RealEvalQuestion(
                    question="Default config test fact about system behavior",
                    expected_facts=["system behavior"],
                    category="single_project",
                    projects=["P1"],
                    difficulty="easy",
                ),
            ],
            dream_boundaries=[],
            date_range=("2026-01-01", "2026-01-01"),
            total_facts=1,
            projects=["P1"],
        )

        results = run_real_eval(dataset)
        assert "configs" in results
        assert len(results["configs"]) == 4

    def test_result_structure(self):
        """Test that results have the expected structure."""
        from build_eval_dataset import RealEvalDataset, RealSession, RealEvalQuestion
        from run_real_eval import run_real_eval, EvalConfig

        dataset = RealEvalDataset(
            sessions=[
                RealSession(
                    date="2026-01-01",
                    day_index=0,
                    project="P1",
                    facts=["Test fact about system structure and design"],
                ),
            ],
            questions=[
                RealEvalQuestion(
                    question="Test fact about system structure and design",
                    expected_facts=["system structure"],
                    category="single_project",
                    projects=["P1"],
                    difficulty="easy",
                ),
            ],
            dream_boundaries=[],
            date_range=("2026-01-01", "2026-01-01"),
            total_facts=1,
            projects=["P1"],
        )

        configs = [EvalConfig(name="test", dreams_enabled=False, dim=64)]
        results = run_real_eval(dataset, configs=configs)

        assert "configs" in results
        assert "best_overall" in results
        assert "summary" in results
        cfg = results["configs"][0]
        assert "config_name" in cfg
        assert "overall_recall" in cfg
        assert "single_project_recall" in cfg
        assert "cross_domain_coverage" in cfg
        assert "temporal_recall" in cfg
        assert "n_questions" in cfg
        assert "n_facts_stored" in cfg
        assert "n_dreams" in cfg


# ---------------------------------------------------------------------------
# System artifact filtering tests (Phase: eval quality fixes)
# ---------------------------------------------------------------------------


class TestSystemArtifactFiltering:
    """Tests for filtering system artifacts from facts and questions."""

    def test_has_system_artifacts_detects_command_name(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("<command-name>/clear</command-name>") is True

    def test_has_system_artifacts_detects_task_notification(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("<task-notification>\n<task-id>abc</task-id>") is True

    def test_has_system_artifacts_detects_system_reminder(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("<system-reminder>some text</system-reminder>") is True

    def test_has_system_artifacts_detects_antml(self):
        from extract_memories import _has_system_artifacts
        # In real Claude Code logs, these appear with the antml: namespace prefix
        assert _has_system_artifacts('<invoke name="Read">') is True

    def test_has_system_artifacts_detects_command_message(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("<command-message>resume_handoff</command-message>") is True

    def test_has_system_artifacts_detects_command_args(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts('<command-args>{"foo": "bar"}</command-args>') is True

    def test_has_system_artifacts_detects_user_prompt_submit_hook(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("<user-prompt-submit-hook>data</user-prompt-submit-hook>") is True

    def test_has_system_artifacts_detects_task_status(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("<task-status>completed</task-status>") is True

    def test_has_system_artifacts_detects_output_file(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("<output-file>/path/to/file</output-file>") is True

    def test_has_system_artifacts_detects_teammate_message(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("<teammate-message>hello</teammate-message>") is True

    def test_has_system_artifacts_clean_text(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("The system uses JWT tokens for auth") is False

    def test_has_system_artifacts_html_like_not_matched(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("Use <strong>bold</strong> text") is False

    def test_has_system_artifacts_empty_string(self):
        from extract_memories import _has_system_artifacts
        assert _has_system_artifacts("") is False

    def test_extract_facts_skips_system_artifacts(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="<command-message>resume_handoff</command-message>\n<command-name>/resume_handoff</command-name>",
            assistant_text="<task-notification>\n<task-id>abc123</task-id>\n<status>completed</status>\n</task-notification>\n\nThe authentication system uses JWT tokens with RSA-256 signing for secure token validation.",
            project="NEL", session_id="s1", cwd="/foo"
        )
        facts = extract_facts_from_turn(turn)
        # Should only get the clean JWT fact, not the system artifacts
        for f in facts:
            assert "<command" not in f.text
            assert "<task-notification" not in f.text

    def test_extract_skips_json_paragraphs(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="What's the config?",
            assistant_text='{"key": "value", "nested": {"foo": "bar"}}',
            project="NEL", session_id="s1", cwd="/foo"
        )
        facts = extract_facts_from_turn(turn)
        for f in facts:
            assert not f.text.strip().startswith("{")

    def test_extract_skips_low_alpha_paragraphs(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Show status",
            assistant_text="+" + "-" * 60 + "+" + "\n| " + " " * 20 + "Property" + " " * 10 + "| Value" + " " * 20 + "|\n" + "+" + "-" * 60 + "+",
            project="NEL", session_id="s1", cwd="/foo"
        )
        facts = extract_facts_from_turn(turn)
        # Table-like content with low alpha ratio should be filtered
        for f in facts:
            if f.source == "assistant":
                alpha_chars = sum(1 for c in f.text if c.isalpha())
                assert alpha_chars / len(f.text) >= 0.4


class TestCrossDomainQuestionQuality:
    """Tests for cross-domain question quality."""

    def test_cross_domain_only_pairs_two_projects(self):
        """Cross-domain questions should list exactly 2 projects, not all."""
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_cross_domain_questions

        facts = [
            MemoryFact("We run CMA-ES optimizer for parameter search",
                      datetime(2026, 1, 1, tzinfo=timezone.utc), "hermes-agent", "s1", "assistant", "general"),
            MemoryFact("CMA-ES covariance matrix adaptation finds optimal configs",
                      datetime(2026, 1, 2, tzinfo=timezone.utc), "CO", "s2", "assistant", "general"),
            MemoryFact("Unrelated fact about UI testing and CSS layout",
                      datetime(2026, 1, 3, tzinfo=timezone.utc), "clinicDashboard", "s3", "assistant", "general"),
            MemoryFact("Another unrelated note about patient records database",
                      datetime(2026, 1, 4, tzinfo=timezone.utc), "NEL", "s4", "assistant", "general"),
        ]
        questions = _generate_cross_domain_questions(facts, n_questions=10)
        for q in questions:
            assert len(q.projects) == 2, f"Expected 2 projects, got {q.projects}"
            # Should involve hermes-agent and CO (the CMA-ES connection), not all 4
            assert "clinicDashboard" not in q.projects or "NEL" not in q.projects

    def test_no_generic_debug_matches(self):
        """Generic words like 'error' and 'debug' should not create cross-domain questions."""
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_cross_domain_questions

        facts = [
            MemoryFact("There was an error in the authentication flow when validating tokens",
                      datetime(2026, 1, 1, tzinfo=timezone.utc), "NEL", "s1", "assistant", "debug"),
            MemoryFact("Fixed an error in the database migration script for patient records",
                      datetime(2026, 1, 2, tzinfo=timezone.utc), "clinicDashboard", "s2", "assistant", "debug"),
        ]
        questions = _generate_cross_domain_questions(facts, n_questions=10)
        # "error" alone should NOT create cross-domain questions
        assert len(questions) == 0, f"Generic 'error' created {len(questions)} questions"

    def test_no_generic_eval_matches(self):
        """Generic words like 'eval' and 'metric' should not create cross-domain questions."""
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_cross_domain_questions

        facts = [
            MemoryFact("We need to evaluate the authentication performance metrics",
                      datetime(2026, 1, 1, tzinfo=timezone.utc), "NEL", "s1", "assistant", "general"),
            MemoryFact("The evaluation of the dashboard revealed several usability metrics",
                      datetime(2026, 1, 2, tzinfo=timezone.utc), "clinicDashboard", "s2", "assistant", "general"),
        ]
        questions = _generate_cross_domain_questions(facts, n_questions=10)
        assert len(questions) == 0, f"Generic 'eval/metric' created {len(questions)} questions"

    def test_specific_keyword_does_match(self):
        """Specific multi-word keywords like 'lean proof' SHOULD create cross-domain questions."""
        from datetime import datetime, timezone
        from extract_memories import MemoryFact
        from build_eval_dataset import _generate_cross_domain_questions

        facts = [
            MemoryFact("The lean proof for convergence uses Mathlib tactics",
                      datetime(2026, 1, 1, tzinfo=timezone.utc), "hermes-agent", "s1", "assistant", "general"),
            MemoryFact("Our lean proof of the energy bound requires tactic automation",
                      datetime(2026, 1, 2, tzinfo=timezone.utc), "CO", "s2", "assistant", "general"),
        ]
        questions = _generate_cross_domain_questions(facts, n_questions=10)
        assert len(questions) > 0, "Specific 'lean proof' should match"
        for q in questions:
            assert len(q.projects) == 2


class TestQuestionQualityGate:
    """Tests that questions don't contain system artifacts."""

    def test_single_project_questions_clean(self):
        """Single-project questions should not contain system artifacts."""
        from parse_real_sessions import parse_all_sessions
        from extract_memories import extract_facts, _has_system_artifacts
        from build_eval_dataset import build_dataset

        turns = parse_all_sessions(projects=["CO"])
        if not turns:
            pytest.skip("CO sessions not available")
        facts = extract_facts(turns)
        if len(facts) < 10:
            pytest.skip("Insufficient CO facts")
        dataset = build_dataset(facts, max_facts_per_session=20)

        for q in dataset.questions:
            assert not _has_system_artifacts(q.question), \
                f"Question contains artifacts: {q.question[:100]}"
            for ef in q.expected_facts:
                assert not _has_system_artifacts(ef), \
                    f"Expected fact contains artifacts: {ef[:100]}"


# ---------------------------------------------------------------------------
# Layer 2: Thinking block metacognition extraction tests
# ---------------------------------------------------------------------------


class TestThinkingBlockExtraction:
    """Tests for Layer 2 extraction from thinking blocks."""

    def test_self_correction_extracted(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Check the query results",
            assistant_text="The results look correct.",
            project="hermes-agent",
            session_id="s1",
            cwd="/foo",
            thinking_text=(
                "The query returns 10 results. Wait — that doesn't add up. "
                "The engine should only have 5 memories stored at this point. "
                "Something is wrong with the deduplication logic."
            ),
        )
        facts = extract_facts_from_turn(turn)
        thinking_facts = [f for f in facts if f.source == "thinking"]
        assert len(thinking_facts) >= 1
        assert thinking_facts[0].fact_type == "self_correction"
        assert thinking_facts[0].layer == "agent_meta"

    def test_finding_in_thinking_extracted(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Analyze the performance",
            assistant_text="Performance is within expected range.",
            project="hermes-agent",
            session_id="s1",
            cwd="/foo",
            thinking_text=(
                "This is crucial. The contradiction detection scans max_candidates=50 "
                "existing memories. But at 262K with 67 chunks, each chunk generates "
                "multiple facts. The scan is O(n) per store which makes it O(n^2) total."
            ),
        )
        facts = extract_facts_from_turn(turn)
        thinking_facts = [f for f in facts if f.source == "thinking"]
        assert len(thinking_facts) >= 1
        assert thinking_facts[0].fact_type == "finding"
        assert thinking_facts[0].layer == "agent_meta"

    def test_hypothesis_in_thinking_extracted(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Why is retrieval slow?",
            assistant_text="Looking into it.",
            project="hermes-agent",
            session_id="s1",
            cwd="/foo",
            thinking_text=(
                "I think this might be caused by the embedding dimension mismatch. "
                "If the stored embeddings are 384-dim but the query embedding is 64-dim, "
                "then the cosine similarity would be meaningless."
            ),
        )
        facts = extract_facts_from_turn(turn)
        thinking_facts = [f for f in facts if f.source == "thinking"]
        assert len(thinking_facts) >= 1
        assert thinking_facts[0].fact_type == "hypothesis"
        assert thinking_facts[0].layer == "agent_meta"

    def test_generic_thinking_discarded(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Read the file",
            assistant_text="Done.",
            project="hermes-agent",
            session_id="s1",
            cwd="/foo",
            thinking_text=(
                "OK so the user wants me to read the file. Let me look at the "
                "structure of this Python module and understand the imports."
            ),
        )
        facts = extract_facts_from_turn(turn)
        thinking_facts = [f for f in facts if f.source == "thinking"]
        assert len(thinking_facts) == 0, "Generic thinking should be discarded"

    def test_short_thinking_discarded(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Check it",
            assistant_text="Done.",
            project="hermes-agent",
            session_id="s1",
            cwd="/foo",
            thinking_text="Wait — no.",
        )
        facts = extract_facts_from_turn(turn)
        thinking_facts = [f for f in facts if f.source == "thinking"]
        assert len(thinking_facts) == 0, "Short thinking should be discarded"

    def test_no_thinking_text_no_crash(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Hello there, how are you doing today?",
            assistant_text="I'm doing well.",
            project="NEL",
            session_id="s1",
            cwd="/foo",
        )
        # thinking_text defaults to ""
        facts = extract_facts_from_turn(turn)
        thinking_facts = [f for f in facts if f.source == "thinking"]
        assert len(thinking_facts) == 0

    def test_metacognitive_text_block_routed_to_layer2(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="What happened?",
            assistant_text=(
                "The root cause of the failure was the missing index on the "
                "users table. Without it, queries took 30 seconds because the "
                "database had to do a full table scan on every authentication check."
            ),
            project="NEL",
            session_id="s1",
            cwd="/foo",
        )
        facts = extract_facts_from_turn(turn)
        assistant_facts = [f for f in facts if f.source == "assistant"]
        assert len(assistant_facts) >= 1
        # "root cause" + "because" → metacognitive text, should be Layer 2
        meta_facts = [f for f in assistant_facts if f.layer == "agent_meta"]
        assert len(meta_facts) >= 1

    def test_non_metacognitive_text_stays_layer1(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="What does the auth system do?",
            assistant_text=(
                "The authentication system uses JWT tokens with RSA-256 signing "
                "for secure token validation across all microservices in the cluster."
            ),
            project="NEL",
            session_id="s1",
            cwd="/foo",
        )
        facts = extract_facts_from_turn(turn)
        assistant_facts = [f for f in facts if f.source == "assistant"]
        assert len(assistant_facts) >= 1
        # Generic description, no metacognition markers → Layer 1
        for f in assistant_facts:
            assert f.layer == "user_knowledge"


# ---------------------------------------------------------------------------
# Layer 3: Handoff YAML extraction tests
# ---------------------------------------------------------------------------


class TestHandoffExtraction:
    """Tests for Layer 3 extraction from handoff YAML files."""

    def test_extract_worked_entries(self, tmp_path):
        from extract_memories import extract_handoff_facts

        handoff = tmp_path / "2026-03-01_10-00_test-handoff.yaml"
        handoff.write_text(
            "root_span_id: abc\n"
            "---\n"
            "session: test\n"
            "---\n"
            "goal: Build the memory extraction pipeline\n"
            "worked:\n"
            "  - 'JSONL session logs are trivially parseable using json.loads per line'\n"
            "  - 'Content-marker classification of text blocks works well for routing'\n"
            "failed: []\n"
        )
        facts = extract_handoff_facts(str(handoff), project="hermes-agent")
        worked_facts = [f for f in facts if f.fact_type == "worked"]
        assert len(worked_facts) == 2
        for f in worked_facts:
            assert f.layer == "procedural"
            assert f.source == "handoff"
            assert f.project == "hermes-agent"

    def test_extract_failed_entries(self, tmp_path):
        from extract_memories import extract_handoff_facts

        handoff = tmp_path / "2026-03-02_14-00_test-fail.yaml"
        handoff.write_text(
            "root_span_id: abc\n"
            "---\n"
            "session: test\n"
            "---\n"
            "goal: Optimize the query pipeline\n"
            "worked: []\n"
            "failed:\n"
            "  - 'Spreading activation failed because signal coherence was too low at 3 hops'\n"
        )
        facts = extract_handoff_facts(str(handoff), project="hermes-agent")
        failed_facts = [f for f in facts if f.fact_type == "failed"]
        assert len(failed_facts) == 1
        assert failed_facts[0].layer == "procedural"
        assert "signal coherence" in failed_facts[0].text

    def test_extract_findings_dict(self, tmp_path):
        from extract_memories import extract_handoff_facts

        handoff = tmp_path / "2026-03-03_08-00_findings.yaml"
        handoff.write_text(
            "root_span_id: abc\n"
            "---\n"
            "session: test\n"
            "---\n"
            "goal: Analyze data sources\n"
            "findings:\n"
            "  data_volume: >-\n"
            "    217K messages across 2700 sessions provides substantial training data\n"
            "  extraction_map: >-\n"
            "    Layer 1 maps to user messages and Layer 2 maps to thinking blocks\n"
        )
        facts = extract_handoff_facts(str(handoff), project="hermes-agent")
        finding_facts = [f for f in facts if f.fact_type == "procedural_finding"]
        assert len(finding_facts) == 2
        assert any("data_volume" in f.text for f in finding_facts)
        assert any("extraction_map" in f.text for f in finding_facts)

    def test_extract_findings_list(self, tmp_path):
        from extract_memories import extract_handoff_facts

        handoff = tmp_path / "2026-03-03_09-00_findings-list.yaml"
        handoff.write_text(
            "root_span_id: abc\n"
            "---\n"
            "session: test\n"
            "---\n"
            "goal: Review architecture\n"
            "findings:\n"
            "  - 'The graph structure enables co-retrieval across memory layers'\n"
            "  - 'Dream consolidation merges near-duplicate procedural memories'\n"
        )
        facts = extract_handoff_facts(str(handoff), project="hermes-agent")
        finding_facts = [f for f in facts if f.fact_type == "procedural_finding"]
        assert len(finding_facts) == 2

    def test_extract_decisions(self, tmp_path):
        from extract_memories import extract_handoff_facts

        handoff = tmp_path / "2026-03-04_12-00_decisions.yaml"
        handoff.write_text(
            "root_span_id: abc\n"
            "---\n"
            "session: test\n"
            "---\n"
            "goal: Design encoding policy\n"
            "decisions:\n"
            "  encoding_gate_fail_open: >-\n"
            "    V1 encoding gate is fail-open by design to avoid losing important memories\n"
            "  noise_filters_separate: >-\n"
            "    Noise filtering happens in extract_memories not in the encoding gate\n"
        )
        facts = extract_handoff_facts(str(handoff), project="hermes-agent")
        decision_facts = [f for f in facts if f.fact_type == "procedural_decision"]
        assert len(decision_facts) == 2
        for f in decision_facts:
            assert f.layer == "procedural"

    def test_short_entries_enriched_with_goal(self, tmp_path):
        from extract_memories import extract_handoff_facts

        handoff = tmp_path / "2026-03-01_10-00_short.yaml"
        handoff.write_text(
            "root_span_id: abc\n"
            "---\n"
            "session: test\n"
            "---\n"
            "goal: Fix Lean proof compilation\n"
            "worked:\n"
            "  - 'abs_add is now abs_add_le'\n"
        )
        facts = extract_handoff_facts(str(handoff))
        worked = [f for f in facts if f.fact_type == "worked"]
        assert len(worked) == 1
        # Short entry should be enriched with goal context
        assert "Fix Lean proof compilation" in worked[0].text

    def test_timestamp_from_filename(self, tmp_path):
        from extract_memories import extract_handoff_facts
        from datetime import datetime

        handoff = tmp_path / "2026-03-05_14-30_my-handoff.yaml"
        handoff.write_text(
            "root_span_id: abc\n"
            "---\n"
            "session: test\n"
            "---\n"
            "goal: Test timestamps\n"
            "worked:\n"
            "  - 'Timestamp extraction from filenames works correctly in the parser'\n"
        )
        facts = extract_handoff_facts(str(handoff))
        assert len(facts) == 1
        assert facts[0].timestamp == datetime(2026, 3, 5, 14, 30)

    def test_empty_handoff_no_crash(self, tmp_path):
        from extract_memories import extract_handoff_facts

        handoff = tmp_path / "2026-01-01_00-00_empty.yaml"
        handoff.write_text(
            "root_span_id: abc\n"
            "---\n"
            "session: test\n"
            "---\n"
            "goal: Empty handoff\n"
        )
        facts = extract_handoff_facts(str(handoff))
        assert facts == []

    def test_very_short_entries_skipped(self, tmp_path):
        from extract_memories import extract_handoff_facts

        handoff = tmp_path / "2026-01-01_00-00_short.yaml"
        handoff.write_text(
            "root_span_id: abc\n"
            "---\n"
            "session: test\n"
            "---\n"
            "goal: Test\n"
            "worked:\n"
            "  - 'ok'\n"
            "  - 'yes'\n"
        )
        facts = extract_handoff_facts(str(handoff))
        assert len(facts) == 0, "Entries < 10 chars should be skipped"

    def test_extract_all_handoff_facts(self, tmp_path):
        from extract_memories import extract_all_handoff_facts

        sub = tmp_path / "general"
        sub.mkdir()
        (sub / "2026-03-01_10-00_a.yaml").write_text(
            "root_span_id: abc\n---\nsession: a\n---\n"
            "goal: Task A\n"
            "worked:\n  - 'Approach A worked well for the optimization problem'\n"
        )
        (sub / "2026-03-02_10-00_b.yaml").write_text(
            "root_span_id: def\n---\nsession: b\n---\n"
            "goal: Task B\n"
            "failed:\n  - 'Approach B failed due to numerical instability in the solver'\n"
        )
        facts = extract_all_handoff_facts(str(tmp_path))
        assert len(facts) == 2
        # Should be sorted by timestamp
        assert facts[0].timestamp < facts[1].timestamp


# ---------------------------------------------------------------------------
# Layer field consistency tests
# ---------------------------------------------------------------------------


class TestLayerConsistency:
    """Verify layer field is set correctly across all extraction paths."""

    def test_user_facts_are_layer1(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="I prefer using TypeScript over JavaScript for all new projects",
            assistant_text="Noted, I'll use TypeScript.",
            project="NEL",
            session_id="s1",
            cwd="/foo",
        )
        facts = extract_facts_from_turn(turn)
        user_facts = [f for f in facts if f.source == "user"]
        assert len(user_facts) >= 1
        for f in user_facts:
            assert f.layer == "user_knowledge"

    def test_thinking_facts_are_layer2(self):
        from datetime import datetime, timezone
        from parse_real_sessions import ConversationTurn
        from extract_memories import extract_facts_from_turn

        turn = ConversationTurn(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            user_text="Investigate the bug",
            assistant_text="Found it.",
            project="hermes-agent",
            session_id="s1",
            cwd="/foo",
            thinking_text=(
                "I assumed the issue was in the query path but actually no, "
                "the bug is in the store path. The embedding normalization "
                "was skipped when importance was set to zero."
            ),
        )
        facts = extract_facts_from_turn(turn)
        thinking_facts = [f for f in facts if f.source == "thinking"]
        assert len(thinking_facts) >= 1
        for f in thinking_facts:
            assert f.layer == "agent_meta"

    def test_handoff_facts_are_layer3(self, tmp_path):
        from extract_memories import extract_handoff_facts

        handoff = tmp_path / "2026-03-01_10-00_test.yaml"
        handoff.write_text(
            "root_span_id: abc\n---\nsession: test\n---\n"
            "goal: Test layer tagging\n"
            "worked:\n  - 'The extraction pipeline handles all YAML formats correctly'\n"
            "findings:\n  test_finding: >-\n"
            "    The parser correctly routes findings to procedural layer\n"
        )
        facts = extract_handoff_facts(str(handoff))
        assert len(facts) >= 2
        for f in facts:
            assert f.layer == "procedural"


class TestReasoningChains:
    """Tests for reasoning chain detection and synthesis in thinking blocks.

    Chains are consecutive metacognitive paragraphs within a single thinking
    block. Individual facts get chain_id + chain_position tags, and a
    synthesized chain fact is emitted with the arc pattern prefix.
    """

    def _make_turn(self, thinking_text, session_id="test-sess"):
        from parse_real_sessions import ConversationTurn

        return ConversationTurn(
            timestamp=datetime(2026, 3, 5),
            user_text="test",
            assistant_text="ok",
            project="test",
            session_id=session_id,
            cwd="/test",
            thinking_text=thinking_text,
        )

    def test_chain_detection_two_consecutive_findings(self):
        """Two consecutive FIND paragraphs form a chain."""
        from extract_memories import _extract_thinking_facts

        turn = self._make_turn(
            "Now looking at the file structure and reading through the configuration "
            "to understand how the pipeline connects to the engine instance.\n\n"
            "I discovered that the embedding normalization was being skipped when "
            "the importance value was exactly zero, causing downstream failures "
            "in the dream consolidation pipeline.\n\n"
            "This reveals that the zero-importance edge case was never tested, "
            "and the original implementation assumed importance would always be "
            "positive, which is a flawed assumption for Layer 4 noise entries."
        )
        facts = _extract_thinking_facts(turn, turn_index=5)
        individual = [f for f in facts if f.fact_type != "reasoning_chain"]
        chains = [f for f in facts if f.fact_type == "reasoning_chain"]

        # Two individual FIND facts should be chain-tagged
        assert len(individual) == 2
        assert all(f.chain_id is not None for f in individual)
        assert individual[0].chain_id == individual[1].chain_id
        assert individual[0].chain_position == 0
        assert individual[1].chain_position == 1

        # One synthesized chain fact
        assert len(chains) == 1
        assert chains[0].text.startswith("[FIND→FIND]")
        assert chains[0].chain_id == individual[0].chain_id
        assert chains[0].chain_position is None
        assert chains[0].layer == "agent_meta"
        assert chains[0].source == "thinking"

    def test_chain_arc_pattern_corr_then_find(self):
        """CORR→FIND arc produces correct pattern prefix."""
        from extract_memories import _extract_thinking_facts

        turn = self._make_turn(
            "Wait — I was wrong about the query path being responsible for "
            "the duplicate results. The real issue is that store() doesn't "
            "check for existing entries with the same text before inserting.\n\n"
            "This reveals that the contradiction-aware mode only checks "
            "embedding similarity, not text equality, so semantically similar "
            "but textually identical facts get stored as separate entries."
        )
        facts = _extract_thinking_facts(turn, turn_index=10)
        chains = [f for f in facts if f.fact_type == "reasoning_chain"]
        assert len(chains) == 1
        assert chains[0].text.startswith("[CORR→FIND]")

    def test_chain_arc_pattern_hyp_corr_find(self):
        """HYP→CORR→FIND three-step arc produces correct pattern."""
        from extract_memories import _extract_thinking_facts

        turn = self._make_turn(
            "I think the reason the retrieval quality dropped is because the "
            "dream cycle is pruning high-importance memories when they fall "
            "below the capacity threshold during consolidation.\n\n"
            "Actually no, that doesn't add up because the capacity gating "
            "specifically preserves memories above importance 0.7. Let me "
            "reconsider what else could explain the retrieval degradation.\n\n"
            "I discovered that the issue was in the co-occurrence graph — "
            "the edge weights were being normalized globally instead of "
            "per-node, which diluted strong local connections."
        )
        facts = _extract_thinking_facts(turn, turn_index=3)
        individual = [f for f in facts if f.fact_type != "reasoning_chain"]
        chains = [f for f in facts if f.fact_type == "reasoning_chain"]

        assert len(individual) == 3
        assert individual[0].fact_type == "hypothesis"
        assert individual[1].fact_type == "self_correction"
        assert individual[2].fact_type == "finding"

        assert len(chains) == 1
        assert chains[0].text.startswith("[HYP→CORR→FIND]")
        # All 3 should share the same chain_id
        assert len(set(f.chain_id for f in individual)) == 1

    def test_no_chain_for_single_meta_paragraph(self):
        """A single metacognitive paragraph does NOT form a chain."""
        from extract_memories import _extract_thinking_facts

        turn = self._make_turn(
            "Some generic reasoning about file structure and organization.\n\n"
            "I discovered that the parser was silently dropping messages "
            "without a timestamp field, which accounted for the missing "
            "twenty percent of conversation turns in the pilot data."
        )
        facts = _extract_thinking_facts(turn, turn_index=0)
        individual = [f for f in facts if f.fact_type != "reasoning_chain"]
        chains = [f for f in facts if f.fact_type == "reasoning_chain"]

        assert len(individual) == 1
        assert individual[0].chain_id is None
        assert individual[0].chain_position is None
        assert len(chains) == 0

    def test_chain_broken_by_generic_paragraph(self):
        """A generic paragraph between two meta paragraphs breaks the chain."""
        from extract_memories import _extract_thinking_facts

        turn = self._make_turn(
            "I discovered that the embedding dimension was wrong, causing "
            "all cosine similarities to be near zero, which explains why "
            "retrieval quality was essentially random in the eval.\n\n"
            "Let me read the configuration file to check the actual dimension "
            "that was set during initialization of the engine.\n\n"
            "This reveals that the config was reading from an old YAML file "
            "that specified dimension 384 instead of 1024, explaining the "
            "dimension mismatch I discovered earlier."
        )
        facts = _extract_thinking_facts(turn, turn_index=0)
        individual = [f for f in facts if f.fact_type != "reasoning_chain"]
        chains = [f for f in facts if f.fact_type == "reasoning_chain"]

        # Two individual facts, no chain (broken by generic paragraph)
        assert len(individual) == 2
        assert all(f.chain_id is None for f in individual)
        assert len(chains) == 0

    def test_chain_id_format_includes_session_and_turn(self):
        """Chain IDs encode session_id prefix and turn index."""
        from extract_memories import _extract_thinking_facts

        turn = self._make_turn(
            "I discovered the root cause is in the normalization step "
            "where L2 norm was applied before rather than after the "
            "outer product computation in the Hebbian update.\n\n"
            "This reveals that swapping the order fixes the gradient "
            "explosion that was corrupting the coupling matrix W "
            "during long dream consolidation cycles.",
            session_id="abcd1234-5678-90ab-cdef-1234567890ab",
        )
        facts = _extract_thinking_facts(turn, turn_index=42)
        chained = [f for f in facts if f.chain_id is not None]
        assert len(chained) >= 2
        # chain_id should start with session prefix and contain turn index
        assert chained[0].chain_id.startswith("abcd1234:t42:")

    def test_chain_synthesis_text_capped_at_2000_chars(self):
        """Synthesized chain facts are truncated to ~2000 chars."""
        from extract_memories import _extract_thinking_facts

        # Create 5 long metacognitive paragraphs
        paras = []
        for i in range(5):
            paras.append(
                f"I discovered finding number {i} which is very important "
                + "and has a lot of detail " * 25
                + f"and this concludes finding {i}."
            )
        thinking = "\n\n".join(paras)
        turn = self._make_turn(thinking)
        facts = _extract_thinking_facts(turn, turn_index=0)
        chains = [f for f in facts if f.fact_type == "reasoning_chain"]
        assert len(chains) == 1
        # Should be capped around 2000 chars (prefix + truncated text)
        assert len(chains[0].text) <= 2100

    def test_multiple_chains_in_one_turn(self):
        """If generic paragraphs split metacognition, multiple chains form."""
        from extract_memories import _extract_thinking_facts

        turn = self._make_turn(
            # Chain 1: FIND→FIND
            "I discovered the issue is in the store path where embeddings "
            "are not normalized before computing the outer product.\n\n"
            "This reveals that unnormalized embeddings cause the coupling "
            "matrix W to have unbounded eigenvalues over time.\n\n"
            # Generic break — long enough to pass filter, no metacognition markers
            "Reading through the dream cycle code now to trace the full "
            "execution path from consolidation trigger to coupling matrix "
            "update and back to the pattern store reconciliation step.\n\n"
            # Chain 2: CORR→FIND
            "Wait — I was wrong about the eigenvalue issue. The real problem "
            "is that the learning rate epsilon is too large for high-dimension "
            "embeddings, causing oscillation during Hebbian update.\n\n"
            "I noticed that reducing epsilon by a factor of dim fixes the "
            "oscillation and stabilizes the coupling matrix evolution."
        )
        facts = _extract_thinking_facts(turn, turn_index=7)
        chains = [f for f in facts if f.fact_type == "reasoning_chain"]
        assert len(chains) == 2
        assert chains[0].text.startswith("[FIND→FIND]")
        assert chains[1].text.startswith("[CORR→FIND]")
        # Different chain_ids
        assert chains[0].chain_id != chains[1].chain_id

    def test_chain_facts_backward_compatible_with_extract_facts_from_turn(self):
        """extract_facts_from_turn still works and includes chain facts."""
        from extract_memories import extract_facts_from_turn

        turn = self._make_turn(
            "I discovered the parser drops messages without timestamps, "
            "causing a twenty percent loss of conversation turns.\n\n"
            "This reveals that the timestamp parsing function returns "
            "None on malformed ISO strings instead of raising an error."
        )
        facts = extract_facts_from_turn(turn, turn_index=0)
        thinking_facts = [f for f in facts if f.source == "thinking"]
        assert len(thinking_facts) >= 3  # 2 individual + 1 chain
        chain_facts = [f for f in thinking_facts if f.fact_type == "reasoning_chain"]
        assert len(chain_facts) == 1

    def test_extract_facts_passes_turn_index(self):
        """extract_facts() passes sequential turn indices for chain_id generation."""
        from extract_memories import extract_facts
        from parse_real_sessions import ConversationTurn

        turns = [
            ConversationTurn(
                timestamp=datetime(2026, 3, 5),
                user_text="test",
                assistant_text="ok",
                project="test",
                session_id="sess-001",
                cwd="/test",
                thinking_text=(
                    "I discovered the root cause is a missing null check "
                    "in the embedding normalization routine.\n\n"
                    "This reveals that the routine was never tested with "
                    "zero-length vectors, which the capacity gating can produce."
                ),
            ),
            ConversationTurn(
                timestamp=datetime(2026, 3, 5),
                user_text="test2",
                assistant_text="ok2",
                project="test",
                session_id="sess-001",
                cwd="/test",
                thinking_text=(
                    "I noticed that the dream consolidation merges patterns "
                    "that are too dissimilar, causing attractor corruption.\n\n"
                    "This shows that the similarity threshold for merge candidates "
                    "needs to be higher than the current 0.5 default value."
                ),
            ),
        ]
        facts = extract_facts(turns)
        chains = [f for f in facts if f.fact_type == "reasoning_chain"]
        assert len(chains) == 2
        # Different turn indices → different chain_ids
        assert chains[0].chain_id != chains[1].chain_id
        assert ":t0:" in chains[0].chain_id
        assert ":t1:" in chains[1].chain_id


# ---------------------------------------------------------------------------
# Layer-aware importance seeding tests (CoupledEngine integration)
# ---------------------------------------------------------------------------


class TestLayerImportanceSeeding:
    """Tests for layer-aware importance seeding in CoupledEngine.

    These tests define the expected behavior for storing memories with
    layer and fact_type metadata, which seeds initial importance values
    based on the cognitive layer the fact originated from.

    Spec: thoughts/shared/plans/layer-importance/spec.md
    """

    DIM = 16

    def _make_engine(self, **kwargs):
        from coupled_engine import CoupledEngine
        defaults = dict(dim=self.DIM)
        defaults.update(kwargs)
        return CoupledEngine(**defaults)

    def _rand_emb(self, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.DIM)

    # ------------------------------------------------------------------
    # 1. MemoryEntry field tests
    # ------------------------------------------------------------------

    def test_memory_entry_has_layer_field(self):
        """MemoryEntry can be constructed with layer='agent_meta' and it is stored."""
        from coupled_engine import MemoryEntry
        entry = MemoryEntry(
            text="test",
            embedding=np.zeros(self.DIM),
            layer="agent_meta",
        )
        assert entry.layer == "agent_meta"

    def test_memory_entry_has_fact_type_field(self):
        """MemoryEntry can be constructed with fact_type='finding'."""
        from coupled_engine import MemoryEntry
        entry = MemoryEntry(
            text="test",
            embedding=np.zeros(self.DIM),
            fact_type="finding",
        )
        assert entry.fact_type == "finding"

    def test_memory_entry_defaults(self):
        """MemoryEntry() defaults to layer='user_knowledge', fact_type='general'."""
        from coupled_engine import MemoryEntry
        entry = MemoryEntry(text="test", embedding=np.zeros(self.DIM))
        assert entry.layer == "user_knowledge"
        assert entry.fact_type == "general"

    # ------------------------------------------------------------------
    # 2. store() importance seeding by layer
    # ------------------------------------------------------------------

    def test_store_with_layer_seeds_importance(self):
        """store(text, emb, layer='agent_meta') produces importance=0.7."""
        engine = self._make_engine()
        emb = self._rand_emb(seed=1)
        idx = engine.store("agent meta fact", emb, layer="agent_meta")
        assert abs(engine.memory_store[idx].importance - 0.7) < 1e-9

    def test_store_with_procedural_layer(self):
        """store(text, emb, layer='procedural') produces importance=0.8."""
        engine = self._make_engine()
        emb = self._rand_emb(seed=2)
        idx = engine.store("procedural fact", emb, layer="procedural")
        assert abs(engine.memory_store[idx].importance - 0.8) < 1e-9

    def test_store_with_reasoning_chain_fact_type(self):
        """store(text, emb, layer='agent_meta', fact_type='reasoning_chain') produces importance=0.75."""
        engine = self._make_engine()
        emb = self._rand_emb(seed=3)
        idx = engine.store(
            "reasoning chain",
            emb,
            layer="agent_meta",
            fact_type="reasoning_chain",
        )
        assert abs(engine.memory_store[idx].importance - 0.75) < 1e-9

    def test_store_explicit_importance_overrides_layer(self):
        """store(text, emb, importance=0.3, layer='procedural') produces importance=0.3."""
        engine = self._make_engine()
        emb = self._rand_emb(seed=4)
        idx = engine.store(
            "explicit override",
            emb,
            importance=0.3,
            layer="procedural",
        )
        assert abs(engine.memory_store[idx].importance - 0.3) < 1e-9

    def test_store_user_knowledge_unchanged(self):
        """store(text, emb, layer='user_knowledge') produces importance=0.5 (backward compat)."""
        engine = self._make_engine()
        emb = self._rand_emb(seed=5)
        idx = engine.store("user knowledge fact", emb, layer="user_knowledge")
        assert abs(engine.memory_store[idx].importance - 0.5) < 1e-9

    def test_store_no_layer_unchanged(self):
        """store(text, emb) with no layer arg produces importance=0.5 (full backward compat)."""
        engine = self._make_engine()
        emb = self._rand_emb(seed=6)
        idx = engine.store("default fact", emb)
        assert abs(engine.memory_store[idx].importance - 0.5) < 1e-9

    # ------------------------------------------------------------------
    # 3. query() and query_readonly() return layer and fact_type
    # ------------------------------------------------------------------

    def test_query_returns_layer_and_fact_type(self):
        """query() results include 'layer' and 'fact_type' keys."""
        engine = self._make_engine()
        emb = self._rand_emb(seed=10)
        engine.store("test fact", emb, layer="procedural", fact_type="worked")
        results = engine.query(emb, top_k=1)
        assert len(results) == 1
        assert "layer" in results[0]
        assert "fact_type" in results[0]
        assert results[0]["layer"] == "procedural"
        assert results[0]["fact_type"] == "worked"

    def test_query_readonly_returns_layer_and_fact_type(self):
        """query_readonly() results include 'layer' and 'fact_type' keys."""
        engine = self._make_engine()
        emb = self._rand_emb(seed=11)
        engine.store("test fact", emb, layer="agent_meta", fact_type="finding")
        results = engine.query_readonly(emb, top_k=1)
        assert len(results) == 1
        assert "layer" in results[0]
        assert "fact_type" in results[0]
        assert results[0]["layer"] == "agent_meta"
        assert results[0]["fact_type"] == "finding"

    # ------------------------------------------------------------------
    # 4. Emotional tagging interaction with layer
    # ------------------------------------------------------------------

    def test_emotional_tagging_with_layer(self):
        """When emotional_tagging=True and layer='procedural', the layer base (0.8)
        replaces S_min in the emotional formula: S0 = 0.8 + (1 - 0.8) * pred_error.
        For the first memory, pred_error=1.0, so importance should be 1.0."""
        engine = self._make_engine(emotional_tagging=True)
        emb = self._rand_emb(seed=20)
        idx = engine.store("procedural fact", emb, layer="procedural")
        # First memory: pred_error = 1.0 (maximally novel)
        # importance = 0.8 + (1.0 - 0.8) * 1.0 = 1.0
        assert abs(engine.memory_store[idx].importance - 1.0) < 1e-6

    # ------------------------------------------------------------------
    # 5. Contradiction replacement preserves new layer
    # ------------------------------------------------------------------

    def test_layer_preserved_through_contradiction_replacement(self):
        """When contradiction_aware replaces a memory, the new layer is used."""
        engine = self._make_engine(
            contradiction_aware=True,
            contradiction_threshold=0.5,
        )
        emb = self._rand_emb(seed=30)
        # Store with user_knowledge layer
        idx1 = engine.store("original fact", emb, layer="user_knowledge")
        assert engine.memory_store[idx1].layer == "user_knowledge"
        # Store near-identical embedding with procedural layer (should replace)
        emb_similar = emb + np.random.default_rng(31).standard_normal(self.DIM) * 0.01
        idx2 = engine.store("updated fact", emb_similar, layer="procedural")
        # The replaced entry should now have the new layer
        assert engine.memory_store[idx2].layer == "procedural"

    # ------------------------------------------------------------------
    # 6. Unknown layer fallback
    # ------------------------------------------------------------------

    def test_unknown_layer_falls_back(self):
        """store(text, emb, layer='unknown_layer') does not raise, falls back to importance=0.5."""
        engine = self._make_engine()
        emb = self._rand_emb(seed=40)
        idx = engine.store("unknown layer fact", emb, layer="unknown_layer")
        assert abs(engine.memory_store[idx].importance - 0.5) < 1e-9
