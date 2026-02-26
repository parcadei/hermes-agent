"""Tests for agent.session_persister — SessionPersister extraction from AIAgent.

TDD Phase 4: Tests written BEFORE implementation to define expected behavior.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from agent.session_persister import SessionPersister


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logs_dir(tmp_path):
    """Temporary logs directory."""
    d = tmp_path / "sessions"
    d.mkdir()
    return d


@pytest.fixture
def mock_session_db():
    """MagicMock standing in for the SQLite session database."""
    db = MagicMock()
    db.append_message = MagicMock()
    db.create_session = MagicMock()
    db.end_session = MagicMock()
    db.update_system_prompt = MagicMock()
    return db


@pytest.fixture
def persister(logs_dir, mock_session_db):
    """SessionPersister with a mock DB and a temp logs_dir."""
    return SessionPersister(
        session_id="test_session_001",
        session_db=mock_session_db,
        logs_dir=logs_dir,
        model="anthropic/claude-opus-4.6",
        base_url="https://openrouter.ai/api/v1",
        platform="cli",
        session_start=datetime(2025, 6, 1, 12, 0, 0),
        save_trajectories=False,
        save_interval=1,
    )


@pytest.fixture
def persister_no_db(logs_dir):
    """SessionPersister without a session database."""
    return SessionPersister(
        session_id="test_session_no_db",
        session_db=None,
        logs_dir=logs_dir,
        model="anthropic/claude-opus-4.6",
        base_url="https://openrouter.ai/api/v1",
        save_interval=1,
    )


# ---------------------------------------------------------------------------
# 1. log_message is no-op when session_db is None
# ---------------------------------------------------------------------------


class TestLogMessageNoopWithoutDB:
    def test_log_message_noop_without_db(self, persister_no_db):
        """log_message does nothing when session_db is None -- no error raised."""
        msg = {"role": "user", "content": "hello"}
        # Should not raise
        persister_no_db.log_message(msg)


# ---------------------------------------------------------------------------
# 2. log_message calls append_message with correct args
# ---------------------------------------------------------------------------


class TestLogMessageCallsAppend:
    def test_log_message_calls_append(self, persister, mock_session_db):
        """log_message delegates to session_db.append_message with correct kwargs."""
        msg = {
            "role": "assistant",
            "content": "Sure, I can help.",
            "tool_calls": [{"name": "search", "arguments": "{}"}],
            "finish_reason": "stop",
        }
        persister.log_message(msg)
        mock_session_db.append_message.assert_called_once()
        kw = mock_session_db.append_message.call_args
        assert kw.kwargs["session_id"] == "test_session_001"
        assert kw.kwargs["role"] == "assistant"
        assert kw.kwargs["content"] == "Sure, I can help."
        assert kw.kwargs["finish_reason"] == "stop"

    def test_log_message_with_dict_tool_calls(self, persister, mock_session_db):
        """log_message handles dict-style tool_calls (from message dicts)."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"name": "run_code", "arguments": '{"code": "print(1)"}'}],
        }
        persister.log_message(msg)
        mock_session_db.append_message.assert_called_once()
        kw = mock_session_db.append_message.call_args
        assert kw.kwargs["tool_calls"] == msg["tool_calls"]


# ---------------------------------------------------------------------------
# 3. persist() calls save_session_log + flush_to_db
# ---------------------------------------------------------------------------


class TestPersistWritesJsonAndDB:
    def test_persist_writes_json_and_db(self, persister, mock_session_db, logs_dir):
        """persist() writes the JSON log file AND flushes to DB."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        persister.persist(messages)
        # JSON file should exist
        assert persister.session_log_file.exists()
        # DB should have been called (flush_to_db path)
        # start_idx = 0 + 1 = 1 when no conversation_history
        # So messages[1:] = [assistant msg] => 1 call
        assert mock_session_db.append_message.call_count >= 1


# ---------------------------------------------------------------------------
# 4. save_session_log overwrites the log file
# ---------------------------------------------------------------------------


class TestSaveSessionLogOverwrites:
    def test_save_session_log_overwrites(self, persister, logs_dir):
        """Session log file is created and then overwritten on second call."""
        msgs1 = [{"role": "user", "content": "first"}]
        persister.save_session_log(msgs1)
        assert persister.session_log_file.exists()
        content1 = persister.session_log_file.read_text()

        msgs2 = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        persister.save_session_log(msgs2)
        content2 = persister.session_log_file.read_text()
        # Second write should have overwritten the first
        assert content2 != content1
        # Second write should have more messages
        data2 = json.loads(content2)
        assert data2["message_count"] == 2


# ---------------------------------------------------------------------------
# 5. save_session_log metadata fields
# ---------------------------------------------------------------------------


class TestSaveSessionLogMetadata:
    def test_save_session_log_metadata(self, persister, logs_dir):
        """JSON includes session_id, model, timestamps, message_count."""
        messages = [
            {"role": "user", "content": "what time is it?"},
            {"role": "assistant", "content": "It is noon."},
        ]
        persister.save_session_log(messages)
        data = json.loads(persister.session_log_file.read_text())
        assert data["session_id"] == "test_session_001"
        assert data["model"] == "anthropic/claude-opus-4.6"
        assert data["base_url"] == "https://openrouter.ai/api/v1"
        assert data["platform"] == "cli"
        assert data["session_start"] == "2025-06-01T12:00:00"
        assert "last_updated" in data
        assert data["message_count"] == 2
        assert len(data["messages"]) == 2


# ---------------------------------------------------------------------------
# 6. flush_to_db skips already-logged messages
# ---------------------------------------------------------------------------


class TestFlushToDBSkipsLogged:
    def test_flush_to_db_skips_logged(self, persister, mock_session_db):
        """start_idx = len(conversation_history) + 1 skips pre-logged messages."""
        conversation_history = [
            {"role": "user", "content": "earlier turn 1"},
            {"role": "assistant", "content": "earlier turn 2"},
        ]
        messages = [
            {"role": "system", "content": "system prompt"},  # idx 0
            {"role": "user", "content": "earlier turn 1"},  # idx 1
            {"role": "assistant", "content": "earlier turn 2"},  # idx 2
            {"role": "user", "content": "new turn"},  # idx 3 -- should be logged
            {"role": "assistant", "content": "new reply"},  # idx 4 -- should be logged
        ]
        persister.flush_to_db(messages, conversation_history)
        # start_idx = len(conversation_history) + 1 = 2 + 1 = 3
        # messages[3:] has 2 items
        assert mock_session_db.append_message.call_count == 2
        roles = [
            c.kwargs["role"] for c in mock_session_db.append_message.call_args_list
        ]
        assert roles == ["user", "assistant"]

    def test_flush_to_db_no_conversation_history(self, persister, mock_session_db):
        """When conversation_history is None, start_idx = 0 + 1 = 1."""
        messages = [
            {"role": "system", "content": "system prompt"},  # idx 0, skipped
            {"role": "user", "content": "hello"},  # idx 1
        ]
        persister.flush_to_db(messages, conversation_history=None)
        assert mock_session_db.append_message.call_count == 1
        assert mock_session_db.append_message.call_args.kwargs["role"] == "user"


# ---------------------------------------------------------------------------
# 7. save_trajectory is no-op when disabled
# ---------------------------------------------------------------------------


class TestSaveTrajectoryNoopWhenDisabled:
    def test_save_trajectory_noop_when_disabled(self, persister):
        """save_trajectories=False means save_trajectory does nothing."""
        assert persister._save_trajectories is False
        # Should not raise, should not write anything
        persister.save_trajectory(
            messages=[{"role": "user", "content": "hi"}],
            user_query="hi",
            completed=True,
        )
        # No trajectory file should be created (we'd need to mock to verify,
        # but the key check is that it returns early without error)


# ---------------------------------------------------------------------------
# 8. session_id setter atomically updates BOTH session_id AND session_log_file
# ---------------------------------------------------------------------------


class TestSessionIdSetterUpdatesLogFile:
    def test_session_id_setter_updates_log_file(self, persister, logs_dir):
        """Setting session_id also updates session_log_file -- THE BUG FIX."""
        assert persister.session_id == "test_session_001"
        expected_original = logs_dir / "session_test_session_001.json"
        assert persister.session_log_file == expected_original

        # Update session_id
        persister.session_id = "new_session_002"
        assert persister.session_id == "new_session_002"
        expected_new = logs_dir / "session_new_session_002.json"
        assert persister.session_log_file == expected_new

    def test_session_id_setter_write_uses_new_path(self, persister, logs_dir):
        """After session_id change, save_session_log writes to the new file."""
        persister.session_id = "renamed_session"
        msgs = [{"role": "user", "content": "post-rename"}]
        persister.save_session_log(msgs)

        new_file = logs_dir / "session_renamed_session.json"
        assert new_file.exists()
        old_file = logs_dir / "session_test_session_001.json"
        assert not old_file.exists()


# ---------------------------------------------------------------------------
# 9. create_compression_session ends old and creates new
# ---------------------------------------------------------------------------


class TestCreateCompressionSession:
    def test_create_compression_session(self, persister, mock_session_db):
        """Ends old session, creates a new one, returns new session ID."""
        old_id = persister.session_id
        new_id = persister.create_compression_session()
        # Old session should be ended
        mock_session_db.end_session.assert_called_once_with(old_id, "compression")
        # New session should be created
        mock_session_db.create_session.assert_called_once()
        create_kw = mock_session_db.create_session.call_args.kwargs
        assert create_kw["session_id"] == new_id
        assert create_kw["source"] == "cli"
        assert create_kw["model"] == "anthropic/claude-opus-4.6"
        # parent_session_id must be the OLD id, not the new one (Bug 2 regression)
        assert create_kw["parent_session_id"] == old_id
        assert create_kw["parent_session_id"] != new_id
        # Return value is the new ID
        assert new_id != old_id
        assert isinstance(new_id, str)
        # Format: YYYYMMDD_HHMMSS_<6hex>
        assert re.match(r"\d{8}_\d{6}_[0-9a-f]{6}", new_id)


# ---------------------------------------------------------------------------
# 10. create_compression_session updates session_id property
# ---------------------------------------------------------------------------


class TestCreateCompressionSessionUpdatesId:
    def test_create_compression_session_updates_session_id(
        self, persister, mock_session_db, logs_dir
    ):
        """session_id and session_log_file reflect the new ID after compression."""
        old_id = persister.session_id
        new_id = persister.create_compression_session()
        assert persister.session_id == new_id
        assert persister.session_log_file == logs_dir / f"session_{new_id}.json"
        assert persister.session_id != old_id

    def test_create_compression_session_without_db(self, persister_no_db):
        """create_compression_session still updates session_id even without DB."""
        old_id = persister_no_db.session_id
        new_id = persister_no_db.create_compression_session()
        assert new_id != old_id
        assert persister_no_db.session_id == new_id


# ---------------------------------------------------------------------------
# 11. maybe_save_session_log respects save_interval
# ---------------------------------------------------------------------------


class TestMaybeSaveSessionLogInterval:
    def test_maybe_save_session_log_interval(self, logs_dir):
        """With save_interval=3, only writes every 3rd call."""
        p = SessionPersister(
            session_id="interval_test",
            session_db=None,
            logs_dir=logs_dir,
            model="test-model",
            base_url="http://localhost",
            save_interval=3,
        )
        msgs = [{"role": "user", "content": "tick"}]

        # Calls 1 and 2 should NOT write
        p.maybe_save_session_log(msgs)
        assert not p.session_log_file.exists()
        p.maybe_save_session_log(msgs)
        assert not p.session_log_file.exists()

        # Call 3 should write
        p.maybe_save_session_log(msgs)
        assert p.session_log_file.exists()
        data = json.loads(p.session_log_file.read_text())
        assert data["message_count"] == 1

        # Delete the file to test the next cycle
        p.session_log_file.unlink()

        # Calls 4 and 5 should NOT write
        p.maybe_save_session_log(msgs)
        assert not p.session_log_file.exists()
        p.maybe_save_session_log(msgs)
        assert not p.session_log_file.exists()

        # Call 6 should write again
        p.maybe_save_session_log(msgs)
        assert p.session_log_file.exists()


# ---------------------------------------------------------------------------
# 12. _clean_session_content converts scratchpad tags
# ---------------------------------------------------------------------------


class TestCleanSessionContent:
    def test_content_scratchpad_to_think_conversion(self):
        """<REASONING_SCRATCHPAD> tags are converted to <think> tags."""
        content = "Some text\n\n\n<REASONING_SCRATCHPAD>internal thoughts</REASONING_SCRATCHPAD>\n\n\nMore text"
        result = SessionPersister._clean_session_content(content)
        assert "<REASONING_SCRATCHPAD>" not in result
        assert "<think>" in result
        assert "</think>" in result
        assert "internal thoughts" in result

    def test_clean_session_content_none(self):
        """None content is returned as-is."""
        assert SessionPersister._clean_session_content(None) is None

    def test_clean_session_content_empty(self):
        """Empty string is returned as-is."""
        assert SessionPersister._clean_session_content("") == ""

    def test_clean_session_content_strips_extra_newlines(self):
        """Extra newlines around think blocks are collapsed."""
        content = "Hello\n\n\n<think>reasoning</think>\n\n\nWorld"
        result = SessionPersister._clean_session_content(content)
        # Should not have triple newlines around think blocks
        assert "\n\n\n<think>" not in result
        assert "</think>\n\n\n" not in result

    def test_save_session_log_cleans_assistant_content(self, persister, logs_dir):
        """save_session_log applies _clean_session_content to assistant messages."""
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<REASONING_SCRATCHPAD>thinking</REASONING_SCRATCHPAD>answer",
            },
        ]
        persister.save_session_log(messages)
        data = json.loads(persister.session_log_file.read_text())
        assistant_msg = data["messages"][1]
        assert "<REASONING_SCRATCHPAD>" not in assistant_msg["content"]
        assert "<think>" in assistant_msg["content"]


# ---------------------------------------------------------------------------
# 13. persist() does not duplicate incrementally-logged messages
# ---------------------------------------------------------------------------


class TestPersistDoesNotDuplicateIncrementallyLoggedMessages:
    """Bug: persist() -> flush_to_db() re-appends messages already written
    by log_message() during the conversation, causing duplicate DB rows.

    The fix must ensure flush_to_db() skips messages that were already
    appended incrementally via log_message().
    """

    def test_persist_does_not_duplicate_incrementally_logged_messages(
        self, persister, mock_session_db
    ):
        """persist() should not re-append messages already written by log_message()."""
        # Build up a messages list as the conversation progresses
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        # Simulate: user sends a message, log_message is called incrementally
        user_msg = {"role": "user", "content": "Hello"}
        messages.append(user_msg)
        persister.log_message(user_msg)

        # Simulate: assistant replies, log_message is called incrementally
        assistant_msg = {"role": "assistant", "content": "Hi there!"}
        messages.append(assistant_msg)
        persister.log_message(assistant_msg)

        # Simulate: user sends another message
        user_msg2 = {"role": "user", "content": "How are you?"}
        messages.append(user_msg2)
        persister.log_message(user_msg2)

        # Simulate: assistant replies again
        assistant_msg2 = {"role": "assistant", "content": "I'm doing well!"}
        messages.append(assistant_msg2)
        persister.log_message(assistant_msg2)

        # At this point, log_message was called 4 times (4 append_message calls)
        assert mock_session_db.append_message.call_count == 4

        # Now persist() is called at end of conversation (no conversation_history)
        # This calls flush_to_db(messages, conversation_history=None)
        # flush_to_db should NOT re-append the 4 messages already logged
        mock_session_db.append_message.reset_mock()
        persister.persist(messages)

        # The critical assertion: no duplicate calls to append_message
        # for messages that were already logged via log_message()
        assert mock_session_db.append_message.call_count == 0, (
            f"Expected 0 additional append_message calls after persist(), "
            f"but got {mock_session_db.append_message.call_count}. "
            f"This means persist() is duplicating messages already logged "
            f"incrementally via log_message()."
        )

    def test_persist_flushes_only_unlogged_trailing_messages(
        self, persister, mock_session_db
    ):
        """If some messages were logged incrementally and new ones were added
        without log_message(), persist() should only flush the new ones."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        # Log first two messages incrementally
        user_msg = {"role": "user", "content": "Hello"}
        messages.append(user_msg)
        persister.log_message(user_msg)

        assistant_msg = {"role": "assistant", "content": "Hi!"}
        messages.append(assistant_msg)
        persister.log_message(assistant_msg)

        assert mock_session_db.append_message.call_count == 2

        # Now add two more messages WITHOUT calling log_message
        # (simulating messages added during error recovery or batch append)
        messages.append({"role": "user", "content": "Bye"})
        messages.append({"role": "assistant", "content": "Goodbye!"})

        mock_session_db.append_message.reset_mock()
        persister.persist(messages)

        # Only the 2 un-logged messages should be flushed
        assert mock_session_db.append_message.call_count == 2, (
            f"Expected 2 append_message calls for un-logged messages, "
            f"but got {mock_session_db.append_message.call_count}."
        )
        roles = [
            c.kwargs["role"]
            for c in mock_session_db.append_message.call_args_list
        ]
        assert roles == ["user", "assistant"]

    def test_flush_to_db_respects_both_conversation_history_and_incremental_log(
        self, persister, mock_session_db
    ):
        """When both conversation_history and incremental logging exist,
        flush_to_db uses whichever skip count is higher."""
        conversation_history = [
            {"role": "user", "content": "old turn 1"},
            {"role": "assistant", "content": "old reply 1"},
        ]

        messages = [
            {"role": "system", "content": "system prompt"},     # idx 0
            {"role": "user", "content": "old turn 1"},          # idx 1
            {"role": "assistant", "content": "old reply 1"},    # idx 2
            {"role": "user", "content": "new turn"},            # idx 3
            {"role": "assistant", "content": "new reply"},      # idx 4
            {"role": "user", "content": "newest turn"},         # idx 5
            {"role": "assistant", "content": "newest reply"},   # idx 6
        ]

        # Incrementally log messages at idx 3, 4, 5, 6
        for msg in messages[3:]:
            persister.log_message(msg)

        assert mock_session_db.append_message.call_count == 4

        mock_session_db.append_message.reset_mock()
        persister.flush_to_db(messages, conversation_history)

        # conversation_history gives start_idx=3 (len(2)+1)
        # but incremental logging already covered idx 3,4,5,6
        # so nothing should be re-appended
        assert mock_session_db.append_message.call_count == 0, (
            f"Expected 0 calls but got {mock_session_db.append_message.call_count}. "
            f"flush_to_db did not respect incremental log count."
        )

    def test_compression_resets_flushed_msg_count(self, logs_dir):
        """After compression, _flushed_msg_count must reset so new messages flush correctly."""
        mock_session_db = MagicMock()
        persister = SessionPersister(
            session_id="old-session",
            logs_dir=str(logs_dir),
            session_db=mock_session_db,
            model="test-model",
            base_url="http://localhost",
        )

        # Simulate 3 incremental log_message() calls in old session
        for i in range(3):
            msg = {"role": "user", "content": f"msg-{i}"}
            persister.log_message(msg)
        assert persister._flushed_msg_count == 3

        # Compress — should reset counter
        mock_session_db.reset_mock()
        persister.create_compression_session()
        assert persister._flushed_msg_count == 0, (
            "Compression must reset _flushed_msg_count for the new session"
        )

        # Now log 2 new messages in the new session
        mock_session_db.reset_mock()
        for i in range(2):
            msg = {"role": "assistant", "content": f"new-msg-{i}"}
            persister.log_message(msg)
        assert persister._flushed_msg_count == 2

        # flush_to_db should NOT skip these new messages
        mock_session_db.reset_mock()
        new_msgs = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "new-msg-0"},
            {"role": "assistant", "content": "new-msg-1"},
        ]
        persister.flush_to_db(new_msgs, conversation_history=None)
        # Messages at idx 1,2 already logged; idx 0 is system (start_idx=1 + flushed=2 = 3, skip all)
        # So no additional append_message calls expected
        assert mock_session_db.append_message.call_count == 0


# ---------------------------------------------------------------------------
# 14. convert_to_trajectory_format PUBLIC API
# ---------------------------------------------------------------------------


class TestConvertToTrajectoryFormatPublicAPI:
    """SessionPersister must expose convert_to_trajectory_format as public API.

    BUG 1 (run_agent.py): calls agent._persister._convert_to_trajectory_format()
           -- reaches through two private layers.
    BUG 2 (batch_runner.py): calls agent._convert_to_trajectory_format() which
           doesn't exist on AIAgent after the refactor -- crashes with AttributeError.

    The fix is a public wrapper method on SessionPersister that delegates to
    the private _convert_to_trajectory_format.
    """

    def test_public_method_exists(self, persister):
        """SessionPersister should expose convert_to_trajectory_format (no leading underscore)."""
        assert hasattr(persister, "convert_to_trajectory_format"), (
            "SessionPersister must have a public convert_to_trajectory_format method"
        )

    def test_public_method_is_callable(self, persister):
        """The public method must be callable."""
        assert callable(getattr(persister, "convert_to_trajectory_format", None))

    def test_public_method_delegates_to_private(self, persister):
        """Public convert_to_trajectory_format returns same result as _convert_to_trajectory_format."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        user_query = "hello"
        completed = True

        private_result = persister._convert_to_trajectory_format(
            messages, user_query, completed
        )
        public_result = persister.convert_to_trajectory_format(
            messages, user_query, completed
        )
        assert public_result == private_result

    def test_public_method_basic_output_structure(self, persister):
        """Public method should produce valid ShareGPT-style trajectory."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        result = persister.convert_to_trajectory_format(
            messages, "What is 2+2?", True
        )
        assert isinstance(result, list)
        assert len(result) >= 2
        # First entry should be system
        assert result[0]["from"] == "system"
        # Second entry should be human with the user_query
        assert result[1]["from"] == "human"
        assert result[1]["value"] == "What is 2+2?"


# ---------------------------------------------------------------------------
# 15. batch_runner.py and run_agent.py call-site verification (AST-level)
# ---------------------------------------------------------------------------


class TestCallerSitesUsePublicAPI:
    """Verify that callers use the public API, not private internals.

    Uses AST inspection to check that:
    - batch_runner.py does NOT call agent._convert_to_trajectory_format (crash bug)
    - run_agent.py does NOT call _persister._convert_to_trajectory_format (private coupling)
    """

    def test_batch_runner_no_agent_convert_to_trajectory(self):
        """batch_runner.py must NOT call agent._convert_to_trajectory_format (crashes post-refactor)."""
        import ast
        from pathlib import Path

        batch_runner_path = Path(__file__).resolve().parents[2] / "batch_runner.py"
        assert batch_runner_path.exists(), f"batch_runner.py not found at {batch_runner_path}"

        source = batch_runner_path.read_text()
        tree = ast.parse(source)

        # Walk the AST looking for attribute access patterns like:
        #   agent._convert_to_trajectory_format(...)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "_convert_to_trajectory_format":
                    pytest.fail(
                        f"batch_runner.py still calls ._convert_to_trajectory_format "
                        f"(line {node.lineno}). This method does not exist on AIAgent "
                        f"and will crash with AttributeError. "
                        f"Use agent._persister.convert_to_trajectory_format() instead."
                    )

    def test_run_agent_no_private_persister_convert(self):
        """run_agent.py must NOT call _persister._convert_to_trajectory_format (double-private)."""
        import ast
        from pathlib import Path

        run_agent_path = Path(__file__).resolve().parents[2] / "run_agent.py"
        assert run_agent_path.exists(), f"run_agent.py not found at {run_agent_path}"

        source = run_agent_path.read_text()
        tree = ast.parse(source)

        # Look for chained attribute access: something._persister._convert_to_trajectory_format
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "_convert_to_trajectory_format":
                    # Check if the value is also an attribute access to _persister
                    if isinstance(node.func.value, ast.Attribute):
                        if node.func.value.attr == "_persister":
                            pytest.fail(
                                f"run_agent.py still calls _persister._convert_to_trajectory_format "
                                f"(line {node.lineno}). Use _persister.convert_to_trajectory_format() "
                                f"(public API) instead."
                            )
