"""Discriminative tests for ConflictResolver (formerly ConflictResolutionMixin).

Validates that:
1. conflict_resolution.py exports _get_subtree_text, _extract_acl_triples,
   ConflictResolver (primary), and ConflictResolutionMixin (backward-compat alias)
2. ConflictResolver has all 4 methods: _query_time_cr,
   _pairwise_cosine_dedup, _resolve_conflicts, _coverage_audit
3. ConflictResolver has __init__ accepting engine, scorer, ingestion, etc.
4. ConflictResolver has _resolve_* properties for attribute resolution
5. ConflictResolutionMixin is ConflictResolver (alias, same object)
6. HermesMemoryAgent inherits from ConflictResolutionMixin
7. _extract_acl_triples is re-exported from hermes_agent for backward compat
8. Module-level _cr_log logger exists in conflict_resolution.py
9. _get_subtree_text and _extract_acl_triples are module-level functions
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


# -----------------------------------------------------------------------
# Phase 2 tests: ConflictResolver standalone class
# -----------------------------------------------------------------------


class TestConflictResolverClass:
    """Verify ConflictResolver is the primary class with __init__."""

    def test_conflict_resolver_importable(self):
        """ConflictResolver must be importable from conflict_resolution."""
        from mabench.conflict_resolution import ConflictResolver

        assert isinstance(ConflictResolver, type)

    def test_mixin_alias_is_same_object(self):
        """ConflictResolutionMixin must be the same class as ConflictResolver."""
        from mabench.conflict_resolution import ConflictResolver, ConflictResolutionMixin

        assert ConflictResolutionMixin is ConflictResolver

    def test_has_init(self):
        """ConflictResolver must define __init__."""
        from mabench.conflict_resolution import ConflictResolver

        assert "__init__" in ConflictResolver.__dict__

    def test_init_signature(self):
        """__init__ must accept engine, scorer, ingestion, supersession_state,
        cr_entity_gate, cosine_dedup_threshold, decompose_fn, retrieve_fn."""
        from mabench.conflict_resolution import ConflictResolver

        sig = inspect.signature(ConflictResolver.__init__)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "engine" in params
        assert "scorer" in params
        assert "ingestion" in params
        assert "supersession_state" in params
        assert "cr_entity_gate" in params
        assert "cosine_dedup_threshold" in params
        assert "decompose_fn" in params
        assert "retrieve_fn" in params


class TestConflictResolverStandalone:
    """Verify ConflictResolver can be instantiated standalone (no mixin host)."""

    def test_instantiate_with_none_args(self):
        """ConflictResolver() with all-None args must not raise."""
        from mabench.conflict_resolution import ConflictResolver

        cr = ConflictResolver(
            engine=None,
            scorer=None,
            ingestion=None,
            supersession_state=None,
        )
        assert cr is not None

    def test_stores_engine(self):
        """_engine attribute must be set from constructor."""
        from mabench.conflict_resolution import ConflictResolver

        sentinel = object()
        cr = ConflictResolver(engine=sentinel, scorer=None, ingestion=None)
        assert cr._engine is sentinel

    def test_stores_scorer(self):
        """_scorer attribute must be set from constructor."""
        from mabench.conflict_resolution import ConflictResolver

        sentinel = object()
        cr = ConflictResolver(engine=None, scorer=sentinel, ingestion=None)
        assert cr._scorer is sentinel

    def test_stores_ingestion(self):
        """_ingestion attribute must be set from constructor."""
        from mabench.conflict_resolution import ConflictResolver

        sentinel = object()
        cr = ConflictResolver(engine=None, scorer=None, ingestion=sentinel)
        assert cr._ingestion is sentinel

    def test_stores_supersession_state(self):
        """_supersession attribute must be set from constructor."""
        from mabench.conflict_resolution import ConflictResolver

        sentinel = object()
        cr = ConflictResolver(
            engine=None, scorer=None, ingestion=None,
            supersession_state=sentinel,
        )
        assert cr._supersession is sentinel

    def test_default_cr_entity_gate(self):
        """cr_entity_gate defaults to False."""
        from mabench.conflict_resolution import ConflictResolver

        cr = ConflictResolver(engine=None, scorer=None, ingestion=None)
        assert cr._cr_entity_gate is False

    def test_default_cosine_dedup_threshold(self):
        """cosine_dedup_threshold defaults to 0.95."""
        from mabench.conflict_resolution import ConflictResolver

        cr = ConflictResolver(engine=None, scorer=None, ingestion=None)
        assert cr._cosine_dedup_threshold == 0.95

    def test_stores_decompose_fn(self):
        """decompose_fn callback must be stored."""
        from mabench.conflict_resolution import ConflictResolver

        fn = lambda q: [q]
        cr = ConflictResolver(
            engine=None, scorer=None, ingestion=None, decompose_fn=fn,
        )
        assert cr._decompose_fn is fn

    def test_stores_retrieve_fn(self):
        """retrieve_fn callback must be stored."""
        from mabench.conflict_resolution import ConflictResolver

        fn = lambda q: []
        cr = ConflictResolver(
            engine=None, scorer=None, ingestion=None, retrieve_fn=fn,
        )
        assert cr._retrieve_fn is fn


class TestResolveProperties:
    """Verify _resolve_* properties for dual-mode attribute resolution."""

    def test_resolve_engine_standalone(self):
        """_resolve_engine must return _engine in standalone mode."""
        from mabench.conflict_resolution import ConflictResolver

        sentinel = object()
        cr = ConflictResolver(engine=sentinel, scorer=None, ingestion=None)
        assert cr._resolve_engine is sentinel

    def test_resolve_supersession_graph_standalone(self):
        """_resolve_supersession_graph must return supersession_state.supersession_graph."""
        from mabench.conflict_resolution import ConflictResolver

        class FakeState:
            supersession_graph = {"old": "new"}
            supersedes = {}

        cr = ConflictResolver(
            engine=None, scorer=None, ingestion=None,
            supersession_state=FakeState(),
        )
        assert cr._resolve_supersession_graph == {"old": "new"}

    def test_resolve_supersedes_standalone(self):
        """_resolve_supersedes must return supersession_state.supersedes."""
        from mabench.conflict_resolution import ConflictResolver

        class FakeState:
            supersession_graph = {}
            supersedes = {"new": {"old1", "old2"}}

        cr = ConflictResolver(
            engine=None, scorer=None, ingestion=None,
            supersession_state=FakeState(),
        )
        assert cr._resolve_supersedes == {"new": {"old1", "old2"}}

    def test_resolve_supersession_graph_no_state(self):
        """Without supersession_state, _resolve_supersession_graph returns empty dict."""
        from mabench.conflict_resolution import ConflictResolver

        cr = ConflictResolver(engine=None, scorer=None, ingestion=None)
        assert cr._resolve_supersession_graph == {}

    def test_resolve_supersedes_no_state(self):
        """Without supersession_state, _resolve_supersedes returns empty dict."""
        from mabench.conflict_resolution import ConflictResolver

        cr = ConflictResolver(engine=None, scorer=None, ingestion=None)
        assert cr._resolve_supersedes == {}

    def test_resolve_engine_mixin_fallback(self):
        """In mixin mode (no _engine set via __init__), _resolve_engine
        falls back to self.coupled_engine."""
        from mabench.conflict_resolution import ConflictResolver

        class FakeHost(ConflictResolver):
            def __init__(self):
                # Do NOT call super().__init__ — simulates mixin mode
                self.coupled_engine = "host_engine"

        host = FakeHost()
        # _engine was never set, so _resolve_engine falls back to coupled_engine
        assert host._resolve_engine == "host_engine"


class TestMethodsOnConflictResolver:
    """All 4 methods must be defined on ConflictResolver.__dict__
    (not inherited from elsewhere)."""

    def test_query_time_cr_on_resolver(self):
        from mabench.conflict_resolution import ConflictResolver

        assert "_query_time_cr" in ConflictResolver.__dict__

    def test_pairwise_cosine_dedup_on_resolver(self):
        from mabench.conflict_resolution import ConflictResolver

        assert "_pairwise_cosine_dedup" in ConflictResolver.__dict__

    def test_resolve_conflicts_on_resolver(self):
        from mabench.conflict_resolution import ConflictResolver

        assert "_resolve_conflicts" in ConflictResolver.__dict__

    def test_coverage_audit_on_resolver(self):
        from mabench.conflict_resolution import ConflictResolver

        assert "_coverage_audit" in ConflictResolver.__dict__
