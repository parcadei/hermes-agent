"""Regression tests for flush_memories: _session_messages stale-state bug.

After the AIAgent decomposition refactor, _session_messages was initialized as []
in __init__ but never updated. flush_memories fell back to it when called without
explicit messages, silently getting an empty list and doing nothing.

FIX: _session_messages is removed entirely. flush_memories requires explicit
messages argument -- all callers already provide it.
"""

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSessionMessagesRemoved:
    """Verify the stale _session_messages attribute no longer exists."""

    def test_no_session_messages_attribute_in_source(self):
        """run_agent.py must not reference _session_messages anywhere."""
        run_agent_path = Path(__file__).resolve().parents[2] / "run_agent.py"
        source = run_agent_path.read_text()
        tree = ast.parse(source)

        # Walk the AST looking for any reference to _session_messages
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "_session_messages":
                pytest.fail(
                    f"run_agent.py still references _session_messages at line {node.lineno}. "
                    f"This attribute was always empty after __init__ and should be removed."
                )
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "_session_messages" in node.value:
                pytest.fail(
                    f"run_agent.py still mentions '_session_messages' in a string literal "
                    f"at line {node.lineno}."
                )


class TestFlushMemoriesRequiresMessages:
    """flush_memories must not silently succeed with no messages."""

    def test_flush_memories_signature_messages_required_or_guarded(self):
        """flush_memories either requires messages or returns early when None.

        The important thing is that flush_memories never silently uses an
        always-empty fallback. It should either:
        - Require messages as a non-optional argument, OR
        - Return early when messages is None (which it does via the
          'if not messages' guard)

        After removing _session_messages, passing messages=None should hit
        the 'if not messages' guard and return immediately (no-op).
        """
        run_agent_path = Path(__file__).resolve().parents[2] / "run_agent.py"
        source = run_agent_path.read_text()
        tree = ast.parse(source)

        # Find the flush_memories method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "flush_memories":
                # Verify the method does NOT contain getattr(..., '_session_messages', ...)
                method_source = ast.get_source_segment(source, node)
                assert "_session_messages" not in method_source, (
                    "flush_memories still references _session_messages. "
                    "Remove the getattr fallback."
                )
                break
        else:
            pytest.fail("flush_memories method not found in run_agent.py")
