"""Discriminative tests for ConflictResolutionMixin extraction.

Validates that:
1. conflict_resolution.py exports _get_subtree_text, _extract_acl_triples,
   and ConflictResolutionMixin
2. ConflictResolutionMixin has all 4 methods: _query_time_cr,
   _pairwise_cosine_dedup, _resolve_conflicts, _coverage_audit
3. HermesMemoryAgent inherits from ConflictResolutionMixin
4. _extract_acl_triples is re-exported from hermes_agent for backward compat
5. Module-level _cr_log logger exists in conflict_resolution.py
6. _get_subtree_text and _extract_acl_triples are module-level functions
   (not methods)

All tests run on CPU. No GPU, no external models, no LLM calls.
"""

import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestConflictResolutionModuleExists:
    """Verify conflict_resolution.py exists and exports expected symbols."""

    def test_module_importable(self):
        """conflict_resolution module must be importable."""
        import mabench.conflict_resolution  # noqa: F401

    def test_exports_get_subtree_text(self):
        """_get_subtree_text must be a module-level function."""
        from mabench.conflict_resolution import _get_subtree_text

        assert callable(_get_subtree_text)
        assert not isinstance(_get_subtree_text, type)

    def test_exports_extract_acl_triples(self):
        """_extract_acl_triples must be a module-level function."""
        from mabench.conflict_resolution import _extract_acl_triples

        assert callable(_extract_acl_triples)
        assert not isinstance(_extract_acl_triples, type)

    def test_exports_mixin_class(self):
        """ConflictResolutionMixin must be a class."""
        from mabench.conflict_resolution import ConflictResolutionMixin

        assert isinstance(ConflictResolutionMixin, type)

    def test_cr_log_exists(self):
        """Module-level _cr_log logger must exist."""
        import mabench.conflict_resolution as mod

        assert hasattr(mod, "_cr_log")


class TestConflictResolutionMixinMethods:
    """Verify ConflictResolutionMixin has all 4 required methods."""

    def test_has_query_time_cr(self):
        from mabench.conflict_resolution import ConflictResolutionMixin

        assert hasattr(ConflictResolutionMixin, "_query_time_cr")
        assert callable(ConflictResolutionMixin._query_time_cr)

    def test_has_pairwise_cosine_dedup(self):
        from mabench.conflict_resolution import ConflictResolutionMixin

        assert hasattr(ConflictResolutionMixin, "_pairwise_cosine_dedup")
        assert callable(ConflictResolutionMixin._pairwise_cosine_dedup)

    def test_has_resolve_conflicts(self):
        from mabench.conflict_resolution import ConflictResolutionMixin

        assert hasattr(ConflictResolutionMixin, "_resolve_conflicts")
        assert callable(ConflictResolutionMixin._resolve_conflicts)

    def test_has_coverage_audit(self):
        from mabench.conflict_resolution import ConflictResolutionMixin

        assert hasattr(ConflictResolutionMixin, "_coverage_audit")
        assert callable(ConflictResolutionMixin._coverage_audit)


class TestFunctionsAreModuleLevel:
    """_get_subtree_text and _extract_acl_triples must NOT be methods."""

    def test_get_subtree_text_is_plain_function(self):
        from mabench.conflict_resolution import _get_subtree_text

        sig = inspect.signature(_get_subtree_text)
        params = list(sig.parameters.keys())
        assert params == ["token"], f"Expected (token,), got {params}"

    def test_extract_acl_triples_is_plain_function(self):
        from mabench.conflict_resolution import _extract_acl_triples

        sig = inspect.signature(_extract_acl_triples)
        params = list(sig.parameters.keys())
        assert params == ["doc"], f"Expected (doc,), got {params}"


class TestHermesAgentInheritance:
    """HermesMemoryAgent must inherit from ConflictResolutionMixin."""

    def test_mro_includes_mixin(self):
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.conflict_resolution import ConflictResolutionMixin

        assert ConflictResolutionMixin in HermesMemoryAgent.__mro__

    def test_agent_has_query_time_cr(self):
        from mabench.hermes_agent import HermesMemoryAgent

        assert hasattr(HermesMemoryAgent, "_query_time_cr")

    def test_agent_has_pairwise_cosine_dedup(self):
        from mabench.hermes_agent import HermesMemoryAgent

        assert hasattr(HermesMemoryAgent, "_pairwise_cosine_dedup")

    def test_agent_has_resolve_conflicts(self):
        from mabench.hermes_agent import HermesMemoryAgent

        assert hasattr(HermesMemoryAgent, "_resolve_conflicts")

    def test_agent_has_coverage_audit(self):
        from mabench.hermes_agent import HermesMemoryAgent

        assert hasattr(HermesMemoryAgent, "_coverage_audit")


class TestBackwardCompatReExport:
    """_extract_acl_triples must still be importable from hermes_agent."""

    def test_import_from_hermes_agent(self):
        from mabench.hermes_agent import _extract_acl_triples

        assert callable(_extract_acl_triples)

    def test_same_function_object(self):
        """Re-export must be the exact same function, not a copy."""
        from mabench.hermes_agent import _extract_acl_triples as from_agent
        from mabench.conflict_resolution import (
            _extract_acl_triples as from_cr,
        )

        assert from_agent is from_cr


class TestNoMethodsLeftOnAgent:
    """Verify the 4 methods are defined in the mixin, not directly on the agent."""

    def test_query_time_cr_from_mixin(self):
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.conflict_resolution import ConflictResolutionMixin

        # Method should come from ConflictResolutionMixin, not defined locally
        assert "_query_time_cr" in ConflictResolutionMixin.__dict__
        assert "_query_time_cr" not in HermesMemoryAgent.__dict__

    def test_pairwise_cosine_dedup_from_mixin(self):
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.conflict_resolution import ConflictResolutionMixin

        assert "_pairwise_cosine_dedup" in ConflictResolutionMixin.__dict__
        assert "_pairwise_cosine_dedup" not in HermesMemoryAgent.__dict__

    def test_resolve_conflicts_from_mixin(self):
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.conflict_resolution import ConflictResolutionMixin

        assert "_resolve_conflicts" in ConflictResolutionMixin.__dict__
        assert "_resolve_conflicts" not in HermesMemoryAgent.__dict__

    def test_coverage_audit_from_mixin(self):
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.conflict_resolution import ConflictResolutionMixin

        assert "_coverage_audit" in ConflictResolutionMixin.__dict__
        assert "_coverage_audit" not in HermesMemoryAgent.__dict__


class TestNoFunctionsLeftOnAgent:
    """_get_subtree_text and _extract_acl_triples must NOT be in hermes_agent module
    (except as re-exports)."""

    def test_get_subtree_text_not_defined_in_agent(self):
        """_get_subtree_text should not be defined in hermes_agent.py source."""
        import mabench.hermes_agent as mod

        source = inspect.getsource(mod)
        # Should not have a def _get_subtree_text line (only import)
        lines = source.split("\n")
        def_lines = [l for l in lines if l.strip().startswith("def _get_subtree_text")]
        assert len(def_lines) == 0, f"_get_subtree_text still defined in hermes_agent: {def_lines}"

    def test_extract_acl_triples_not_defined_in_agent(self):
        """_extract_acl_triples should not be defined in hermes_agent.py source."""
        import mabench.hermes_agent as mod

        source = inspect.getsource(mod)
        lines = source.split("\n")
        def_lines = [l for l in lines if l.strip().startswith("def _extract_acl_triples")]
        assert len(def_lines) == 0, f"_extract_acl_triples still defined in hermes_agent: {def_lines}"
