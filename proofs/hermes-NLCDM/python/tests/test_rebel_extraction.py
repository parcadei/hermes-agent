"""Tests for RebelKGMixin extraction to mabench.rebel.

Verifies:
1. RebelKGMixin can be imported from mabench.rebel
2. All 5 methods exist on RebelKGMixin
3. HermesMemoryAgent inherits from RebelKGMixin
4. _parse_rebel_output is a @staticmethod
5. Methods are callable via the agent (MRO resolution)
6. mabench.rebel does NOT contain HermesMemoryAgent
7. Methods have correct signatures (same as original)
8. Class-level attributes (_KG_STOP_WORDS, _KG_STEM_MAP) are on the mixin
9. Methods NOT defined directly on HermesMemoryAgent.__dict__
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Ensure sys.path includes the necessary directories
_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))


# ===========================================================================
# Test Group 1: Import and existence
# ===========================================================================


def test_import_rebel_kg_mixin():
    """RebelKGMixin can be imported from mabench.rebel."""
    from mabench.rebel import RebelKGMixin
    assert RebelKGMixin is not None


def test_rebel_kg_mixin_has_all_methods():
    """RebelKGMixin has all 5 extracted methods."""
    from mabench.rebel import RebelKGMixin
    expected_methods = [
        "_ensure_rebel",
        "_parse_rebel_output",
        "_rebel_extract_batch",
        "_rebel_kg_add",
        "_rebel_kg_hop",
    ]
    for method_name in expected_methods:
        assert hasattr(RebelKGMixin, method_name), (
            f"Missing method on RebelKGMixin: {method_name}"
        )
        assert callable(getattr(RebelKGMixin, method_name)), (
            f"Method {method_name} is not callable"
        )


def test_hermes_agent_inherits_rebel_kg_mixin():
    """HermesMemoryAgent inherits from RebelKGMixin."""
    from mabench.rebel import RebelKGMixin
    from mabench.hermes_agent import HermesMemoryAgent
    assert issubclass(HermesMemoryAgent, RebelKGMixin), (
        "HermesMemoryAgent should inherit from RebelKGMixin"
    )


def test_rebel_kg_mixin_in_mro():
    """RebelKGMixin appears in HermesMemoryAgent's MRO."""
    from mabench.rebel import RebelKGMixin
    from mabench.hermes_agent import HermesMemoryAgent
    mro_names = [cls.__name__ for cls in HermesMemoryAgent.__mro__]
    assert "RebelKGMixin" in mro_names, (
        f"RebelKGMixin not in MRO: {mro_names}"
    )


def test_rebel_module_does_not_contain_hermes_agent():
    """mabench.rebel should NOT contain HermesMemoryAgent."""
    import mabench.rebel as rebel_mod
    assert not hasattr(rebel_mod, "HermesMemoryAgent")


def test_hermes_agent_still_exists():
    """hermes_agent.py should still contain HermesMemoryAgent."""
    from mabench.hermes_agent import HermesMemoryAgent
    assert HermesMemoryAgent is not None


# ===========================================================================
# Test Group 2: Method signatures preserved
# ===========================================================================


def test_ensure_rebel_signature():
    """_ensure_rebel takes only self."""
    from mabench.rebel import RebelKGMixin
    sig = inspect.signature(RebelKGMixin._ensure_rebel)
    params = list(sig.parameters.keys())
    assert params == ["self"], f"Unexpected params: {params}"


def test_parse_rebel_output_is_staticmethod():
    """_parse_rebel_output is a @staticmethod on RebelKGMixin."""
    from mabench.rebel import RebelKGMixin
    # Check that it's a staticmethod in the class __dict__
    assert isinstance(
        RebelKGMixin.__dict__["_parse_rebel_output"], staticmethod
    ), "_parse_rebel_output should be a @staticmethod"


def test_parse_rebel_output_signature():
    """_parse_rebel_output takes decoded: str."""
    from mabench.rebel import RebelKGMixin
    sig = inspect.signature(RebelKGMixin._parse_rebel_output)
    params = list(sig.parameters.keys())
    assert params == ["decoded"], f"Unexpected params: {params}"


def test_rebel_extract_batch_signature():
    """_rebel_extract_batch takes self, texts."""
    from mabench.rebel import RebelKGMixin
    sig = inspect.signature(RebelKGMixin._rebel_extract_batch)
    params = list(sig.parameters.keys())
    assert params == ["self", "texts"], f"Unexpected params: {params}"


def test_rebel_kg_add_signature():
    """_rebel_kg_add takes self, head, relation, tail."""
    from mabench.rebel import RebelKGMixin
    sig = inspect.signature(RebelKGMixin._rebel_kg_add)
    params = list(sig.parameters.keys())
    assert params == ["self", "head", "relation", "tail"], f"Unexpected params: {params}"


def test_rebel_kg_hop_signature():
    """_rebel_kg_hop takes self, sub_query."""
    from mabench.rebel import RebelKGMixin
    sig = inspect.signature(RebelKGMixin._rebel_kg_hop)
    params = list(sig.parameters.keys())
    assert params == ["self", "sub_query"], f"Unexpected params: {params}"


# ===========================================================================
# Test Group 3: Class-level attributes on mixin
# ===========================================================================


def test_kg_stop_words_on_mixin():
    """_KG_STOP_WORDS is defined on RebelKGMixin."""
    from mabench.rebel import RebelKGMixin
    assert hasattr(RebelKGMixin, "_KG_STOP_WORDS")
    assert isinstance(RebelKGMixin._KG_STOP_WORDS, frozenset)
    assert "what" in RebelKGMixin._KG_STOP_WORDS


def test_kg_stem_map_on_mixin():
    """_KG_STEM_MAP is defined on RebelKGMixin."""
    from mabench.rebel import RebelKGMixin
    assert hasattr(RebelKGMixin, "_KG_STEM_MAP")
    assert isinstance(RebelKGMixin._KG_STEM_MAP, dict)
    assert RebelKGMixin._KG_STEM_MAP.get("born") == "birth"


# ===========================================================================
# Test Group 4: Methods NOT defined directly on HermesMemoryAgent
# ===========================================================================


def test_methods_not_defined_directly_on_hermes_agent():
    """REBEL methods should be defined on RebelKGMixin, not HermesMemoryAgent."""
    from mabench.rebel import RebelKGMixin
    from mabench.hermes_agent import HermesMemoryAgent

    methods = [
        "_ensure_rebel",
        "_parse_rebel_output",
        "_rebel_extract_batch",
        "_rebel_kg_add",
        "_rebel_kg_hop",
    ]
    for method_name in methods:
        # Method should exist on HermesMemoryAgent (via inheritance)
        assert hasattr(HermesMemoryAgent, method_name), (
            f"{method_name} not found on HermesMemoryAgent"
        )
        # But should be defined on RebelKGMixin, not HermesMemoryAgent itself
        assert method_name in RebelKGMixin.__dict__, (
            f"{method_name} should be defined on RebelKGMixin.__dict__"
        )
        assert method_name not in HermesMemoryAgent.__dict__, (
            f"{method_name} should NOT be in HermesMemoryAgent.__dict__ "
            "(it should come from RebelKGMixin via inheritance)"
        )


def test_class_attrs_not_on_hermes_agent():
    """_KG_STOP_WORDS and _KG_STEM_MAP should be on RebelKGMixin, not HermesMemoryAgent."""
    from mabench.rebel import RebelKGMixin
    from mabench.hermes_agent import HermesMemoryAgent

    for attr in ["_KG_STOP_WORDS", "_KG_STEM_MAP"]:
        assert attr in RebelKGMixin.__dict__, (
            f"{attr} should be defined on RebelKGMixin.__dict__"
        )
        assert attr not in HermesMemoryAgent.__dict__, (
            f"{attr} should NOT be in HermesMemoryAgent.__dict__"
        )


# ===========================================================================
# Test Group 5: Return type annotations preserved
# ===========================================================================


def test_parse_rebel_output_return_annotation():
    """_parse_rebel_output has return type annotation."""
    from mabench.rebel import RebelKGMixin
    hints = RebelKGMixin._parse_rebel_output.__annotations__
    assert "return" in hints, "Missing return annotation"


def test_rebel_extract_batch_return_annotation():
    """_rebel_extract_batch has return type annotation."""
    from mabench.rebel import RebelKGMixin
    hints = RebelKGMixin._rebel_extract_batch.__annotations__
    assert "return" in hints, "Missing return annotation"


def test_rebel_kg_add_return_annotation():
    """_rebel_kg_add has return type annotation."""
    from mabench.rebel import RebelKGMixin
    hints = RebelKGMixin._rebel_kg_add.__annotations__
    assert "return" in hints, "Missing return annotation"


def test_rebel_kg_hop_return_annotation():
    """_rebel_kg_hop has return type annotation."""
    from mabench.rebel import RebelKGMixin
    hints = RebelKGMixin._rebel_kg_hop.__annotations__
    assert "return" in hints, "Missing return annotation"
