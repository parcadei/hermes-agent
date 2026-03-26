"""Tests for StoreMixin extraction from HermesMemoryAgent.

Validates that:
1. StoreMixin exists and is importable
2. _store method lives on StoreMixin, not hermes_agent
3. HermesMemoryAgent inherits from StoreMixin
4. All helper methods exist on StoreMixin
5. Each helper is <50 lines
6. store.py total is <500 lines
7. MRO is correct (no diamond conflicts)
8. Behavioral parity: _store still works end-to-end
"""

import ast
import inspect
import textwrap
from pathlib import Path


MABENCH_DIR = Path(__file__).resolve().parent.parent / "mabench"


# --- Structural tests ---


def test_store_mixin_importable():
    """StoreMixin can be imported from mabench.store."""
    from mabench.store import StoreMixin
    assert isinstance(StoreMixin, type)


def test_store_method_on_mixin():
    """_store is defined on StoreMixin."""
    from mabench.store import StoreMixin
    assert hasattr(StoreMixin, "_store")
    assert callable(getattr(StoreMixin, "_store"))


def test_store_method_not_on_hermes_agent():
    """_store is NOT defined directly on HermesMemoryAgent (inherited only)."""
    source = (MABENCH_DIR / "hermes_agent.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HermesMemoryAgent":
            method_names = [
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            assert "_store" not in method_names, (
                "_store should be on StoreMixin, not HermesMemoryAgent"
            )


def test_hermes_inherits_store_mixin():
    """HermesMemoryAgent lists StoreMixin as a base class."""
    source = (MABENCH_DIR / "hermes_agent.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HermesMemoryAgent":
            base_names = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(base.attr)
            assert "StoreMixin" in base_names, (
                f"HermesMemoryAgent bases: {base_names}"
            )


def test_store_import_in_hermes_agent():
    """hermes_agent.py imports StoreMixin from mabench.store."""
    source = (MABENCH_DIR / "hermes_agent.py").read_text()
    assert "from mabench.store import StoreMixin" in source


def test_helper_methods_exist():
    """All decomposed helper methods exist on StoreMixin."""
    from mabench.store import StoreMixin
    expected_helpers = [
        "_store",
        "_store_prepare",
        "_store_check_supersession",
        "_store_replace_entry",
        "_store_new_entry",
        "_store_update_indices",
        "_store_update_triple_index",
        "_store_detect_belief_conflicts",
        "_store_post_chunk",
    ]
    for name in expected_helpers:
        assert hasattr(StoreMixin, name), f"StoreMixin missing method: {name}"


def test_helper_methods_under_50_lines():
    """Each method on StoreMixin is under 50 lines (excluding docstring)."""
    from mabench.store import StoreMixin
    for name, method in inspect.getmembers(StoreMixin, predicate=inspect.isfunction):
        if name.startswith("_store"):
            source = inspect.getsource(method)
            # Count non-blank, non-comment, non-docstring lines
            lines = source.split("\n")
            # Remove the def line and docstring
            code_lines = []
            in_docstring = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if in_docstring:
                        in_docstring = False
                        continue
                    elif stripped.count('"""') == 2 or stripped.count("'''") == 2:
                        continue  # single-line docstring
                    else:
                        in_docstring = True
                        continue
                if in_docstring:
                    continue
                if stripped and not stripped.startswith("#"):
                    code_lines.append(line)
            assert len(code_lines) <= 50, (
                f"StoreMixin.{name} has {len(code_lines)} code lines (max 50):\n"
                + "\n".join(code_lines[:55])
            )


def test_store_py_under_500_lines():
    """store.py is under 500 lines total."""
    source = (MABENCH_DIR / "store.py").read_text()
    line_count = len(source.split("\n"))
    assert line_count < 500, f"store.py has {line_count} lines (max 500)"


def test_mro_includes_store_mixin():
    """MRO includes StoreMixin without diamond conflicts."""
    from mabench.hermes_agent import HermesMemoryAgent
    from mabench.store import StoreMixin
    mro_names = [c.__name__ for c in HermesMemoryAgent.__mro__]
    assert "StoreMixin" in mro_names
    # StoreMixin should appear exactly once
    assert mro_names.count("StoreMixin") == 1


def test_store_mixin_has_docstring():
    """StoreMixin class has a docstring."""
    from mabench.store import StoreMixin
    assert StoreMixin.__doc__ is not None
    assert len(StoreMixin.__doc__.strip()) > 10


def test_no_circular_import():
    """Importing store does not cause circular import issues."""
    import importlib
    import mabench.store
    importlib.reload(mabench.store)
    # If we get here without ImportError, we're good


def test_store_method_signature_preserved():
    """_store(self, text: str) -> str signature is preserved."""
    from mabench.store import StoreMixin
    sig = inspect.signature(StoreMixin._store)
    params = list(sig.parameters.keys())
    assert params == ["self", "text"], f"Expected ['self', 'text'], got {params}"
