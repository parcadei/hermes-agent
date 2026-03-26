"""Tests for EntityMixin extraction to mabench.entity.

Verifies:
1. EntityMixin can be imported from mabench.entity
2. All 8 methods exist on EntityMixin
3. HermesMemoryAgent inherits from EntityMixin
4. Methods are callable via the agent (MRO resolution)
5. mabench.entity does NOT contain HermesMemoryAgent
6. Methods have correct signatures (same as original)
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


def test_import_entity_mixin():
    """EntityMixin can be imported from mabench.entity."""
    from mabench.entity import EntityMixin
    assert EntityMixin is not None


def test_entity_mixin_has_all_methods():
    """EntityMixin has all 8 extracted methods."""
    from mabench.entity import EntityMixin
    expected_methods = [
        "_ensure_gliner",
        "_extract_entities_batch",
        "_extract_primary_entity",
        "_extract_query_entities",
        "_entity_text_search",
        "_entity_first_retrieve",
        "rebuild_entity_index",
        "_entity_graph_expand",
    ]
    for method_name in expected_methods:
        assert hasattr(EntityMixin, method_name), (
            f"Missing method on EntityMixin: {method_name}"
        )
        assert callable(getattr(EntityMixin, method_name)), (
            f"Method {method_name} is not callable"
        )


def test_hermes_agent_inherits_entity_mixin():
    """HermesMemoryAgent inherits from EntityMixin."""
    from mabench.entity import EntityMixin
    from mabench.hermes_agent import HermesMemoryAgent
    assert issubclass(HermesMemoryAgent, EntityMixin), (
        "HermesMemoryAgent should inherit from EntityMixin"
    )


def test_entity_mixin_in_mro():
    """EntityMixin appears in HermesMemoryAgent's MRO."""
    from mabench.entity import EntityMixin
    from mabench.hermes_agent import HermesMemoryAgent
    mro_names = [cls.__name__ for cls in HermesMemoryAgent.__mro__]
    assert "EntityMixin" in mro_names, (
        f"EntityMixin not in MRO: {mro_names}"
    )


def test_entity_module_does_not_contain_hermes_agent():
    """mabench.entity should NOT contain HermesMemoryAgent."""
    import mabench.entity as entity_mod
    assert not hasattr(entity_mod, "HermesMemoryAgent")


def test_hermes_agent_still_exists():
    """hermes_agent.py should still contain HermesMemoryAgent."""
    from mabench.hermes_agent import HermesMemoryAgent
    assert HermesMemoryAgent is not None


# ===========================================================================
# Test Group 2: Method signatures preserved
# ===========================================================================


def test_ensure_gliner_signature():
    """_ensure_gliner takes only self."""
    from mabench.entity import EntityMixin
    sig = inspect.signature(EntityMixin._ensure_gliner)
    params = list(sig.parameters.keys())
    assert params == ["self"], f"Unexpected params: {params}"


def test_extract_entities_batch_signature():
    """_extract_entities_batch takes self, texts."""
    from mabench.entity import EntityMixin
    sig = inspect.signature(EntityMixin._extract_entities_batch)
    params = list(sig.parameters.keys())
    assert params == ["self", "texts"], f"Unexpected params: {params}"


def test_extract_primary_entity_signature():
    """_extract_primary_entity takes self, texts."""
    from mabench.entity import EntityMixin
    sig = inspect.signature(EntityMixin._extract_primary_entity)
    params = list(sig.parameters.keys())
    assert params == ["self", "texts"], f"Unexpected params: {params}"


def test_extract_query_entities_signature():
    """_extract_query_entities takes self, query."""
    from mabench.entity import EntityMixin
    sig = inspect.signature(EntityMixin._extract_query_entities)
    params = list(sig.parameters.keys())
    assert params == ["self", "query"], f"Unexpected params: {params}"


def test_entity_text_search_signature():
    """_entity_text_search takes self, query, top_k."""
    from mabench.entity import EntityMixin
    sig = inspect.signature(EntityMixin._entity_text_search)
    params = list(sig.parameters.keys())
    assert params == ["self", "query", "top_k"], f"Unexpected params: {params}"
    # top_k has default value of 10
    assert sig.parameters["top_k"].default == 10


def test_entity_first_retrieve_signature():
    """_entity_first_retrieve takes self, query."""
    from mabench.entity import EntityMixin
    sig = inspect.signature(EntityMixin._entity_first_retrieve)
    params = list(sig.parameters.keys())
    assert params == ["self", "query"], f"Unexpected params: {params}"


def test_rebuild_entity_index_signature():
    """rebuild_entity_index takes self, batch_size."""
    from mabench.entity import EntityMixin
    sig = inspect.signature(EntityMixin.rebuild_entity_index)
    params = list(sig.parameters.keys())
    assert params == ["self", "batch_size"], f"Unexpected params: {params}"
    # batch_size has default value of 64
    assert sig.parameters["batch_size"].default == 64


def test_entity_graph_expand_signature():
    """_entity_graph_expand takes self, seed_facts, top_k."""
    from mabench.entity import EntityMixin
    sig = inspect.signature(EntityMixin._entity_graph_expand)
    params = list(sig.parameters.keys())
    assert params == ["self", "seed_facts", "top_k"], f"Unexpected params: {params}"
    # top_k has default value of 5
    assert sig.parameters["top_k"].default == 5


# ===========================================================================
# Test Group 3: Methods NOT on hermes_agent.py directly
# ===========================================================================


def test_methods_not_defined_directly_on_hermes_agent():
    """Entity methods should be defined on EntityMixin, not HermesMemoryAgent."""
    from mabench.entity import EntityMixin
    from mabench.hermes_agent import HermesMemoryAgent

    methods = [
        "_ensure_gliner",
        "_extract_entities_batch",
        "_extract_primary_entity",
        "_extract_query_entities",
        "_entity_text_search",
        "_entity_first_retrieve",
        "rebuild_entity_index",
        "_entity_graph_expand",
    ]
    for method_name in methods:
        # Method should exist on HermesMemoryAgent (via inheritance)
        assert hasattr(HermesMemoryAgent, method_name), (
            f"{method_name} not found on HermesMemoryAgent"
        )
        # But should be defined on EntityMixin, not HermesMemoryAgent itself
        assert method_name in EntityMixin.__dict__, (
            f"{method_name} should be defined on EntityMixin.__dict__"
        )
        assert method_name not in HermesMemoryAgent.__dict__, (
            f"{method_name} should NOT be in HermesMemoryAgent.__dict__ "
            "(it should come from EntityMixin via inheritance)"
        )


# ===========================================================================
# Test Group 4: Return type annotations preserved
# ===========================================================================


def test_extract_entities_batch_return_annotation():
    """_extract_entities_batch has correct return type annotation."""
    from mabench.entity import EntityMixin
    hints = EntityMixin._extract_entities_batch.__annotations__
    assert "return" in hints, "Missing return annotation"


def test_extract_primary_entity_return_annotation():
    """_extract_primary_entity has correct return type annotation."""
    from mabench.entity import EntityMixin
    hints = EntityMixin._extract_primary_entity.__annotations__
    assert "return" in hints, "Missing return annotation"


def test_entity_first_retrieve_return_annotation():
    """_entity_first_retrieve has correct return type annotation."""
    from mabench.entity import EntityMixin
    hints = EntityMixin._entity_first_retrieve.__annotations__
    assert "return" in hints, "Missing return annotation"


def test_entity_graph_expand_return_annotation():
    """_entity_graph_expand has correct return type annotation."""
    from mabench.entity import EntityMixin
    hints = EntityMixin._entity_graph_expand.__annotations__
    assert "return" in hints, "Missing return annotation"
