"""Discriminative tests for OutgestionMixin extraction.

Validates that:
1. outgestion.py exports OutgestionMixin
2. OutgestionMixin has both methods: _format_outgestion, _generate_answer
3. HermesMemoryAgent inherits from OutgestionMixin
4. Methods are defined on the mixin, not directly on the agent class
5. Method signatures are preserved exactly

All tests run on CPU. No GPU, no external models, no LLM calls.
"""

import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestOutgestionModuleExists:
    """Verify outgestion.py exists and exports expected symbols."""

    def test_module_importable(self):
        """outgestion module must be importable."""
        import mabench.outgestion  # noqa: F401

    def test_exports_mixin_class(self):
        """OutgestionMixin must be a class."""
        from mabench.outgestion import OutgestionMixin

        assert isinstance(OutgestionMixin, type)

    def test_module_has_docstring(self):
        """outgestion module must have a docstring."""
        import mabench.outgestion as mod

        assert mod.__doc__ is not None
        assert len(mod.__doc__.strip()) > 0


class TestOutgestionMixinMethods:
    """Verify OutgestionMixin has both required methods."""

    def test_has_format_outgestion(self):
        from mabench.outgestion import OutgestionMixin

        assert hasattr(OutgestionMixin, "_format_outgestion")
        assert callable(OutgestionMixin._format_outgestion)

    def test_has_generate_answer(self):
        from mabench.outgestion import OutgestionMixin

        assert hasattr(OutgestionMixin, "_generate_answer")
        assert callable(OutgestionMixin._generate_answer)

    def test_format_outgestion_signature(self):
        """_format_outgestion must accept (self, chunks, metadata_by_key)."""
        from mabench.outgestion import OutgestionMixin

        sig = inspect.signature(OutgestionMixin._format_outgestion)
        params = list(sig.parameters.keys())
        assert params == ["self", "chunks", "metadata_by_key"], (
            f"Expected (self, chunks, metadata_by_key), got {params}"
        )

    def test_generate_answer_signature(self):
        """_generate_answer must accept (self, question, context, extra_system_instructions)."""
        from mabench.outgestion import OutgestionMixin

        sig = inspect.signature(OutgestionMixin._generate_answer)
        params = list(sig.parameters.keys())
        assert params == [
            "self",
            "question",
            "context",
            "extra_system_instructions",
        ], f"Expected (self, question, context, extra_system_instructions), got {params}"

    def test_generate_answer_has_default(self):
        """extra_system_instructions must default to empty string."""
        from mabench.outgestion import OutgestionMixin

        sig = inspect.signature(OutgestionMixin._generate_answer)
        default = sig.parameters["extra_system_instructions"].default
        assert default == "", f"Expected default='', got {default!r}"


class TestHermesAgentInheritance:
    """HermesMemoryAgent must inherit from OutgestionMixin."""

    def test_mro_includes_mixin(self):
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.outgestion import OutgestionMixin

        assert OutgestionMixin in HermesMemoryAgent.__mro__

    def test_agent_has_format_outgestion(self):
        from mabench.hermes_agent import HermesMemoryAgent

        assert hasattr(HermesMemoryAgent, "_format_outgestion")

    def test_agent_has_generate_answer(self):
        from mabench.hermes_agent import HermesMemoryAgent

        assert hasattr(HermesMemoryAgent, "_generate_answer")


class TestNoMethodsLeftOnAgent:
    """Verify methods are defined in the mixin, not directly on the agent."""

    def test_format_outgestion_from_mixin(self):
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.outgestion import OutgestionMixin

        assert "_format_outgestion" in OutgestionMixin.__dict__
        assert "_format_outgestion" not in HermesMemoryAgent.__dict__

    def test_generate_answer_from_mixin(self):
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.outgestion import OutgestionMixin

        assert "_generate_answer" in OutgestionMixin.__dict__
        assert "_generate_answer" not in HermesMemoryAgent.__dict__


class TestFormatOutgestionBehavior:
    """Test _format_outgestion behavior without needing full agent init."""

    def test_flat_text_when_conflict_resolution(self):
        """When conflict_resolution=True, returns flat joined text."""
        from mabench.outgestion import OutgestionMixin

        class FakeAgent(OutgestionMixin):
            conflict_resolution = True
            temporal_context = False
            contradiction_context = False

        agent = FakeAgent()
        chunks = ["fact one", "fact two", "fact three"]
        ctx, instructions = agent._format_outgestion(chunks, {})
        assert ctx == "fact one\n\nfact two\n\nfact three"
        assert instructions == ""

    def test_flat_text_when_no_temporal(self):
        """When temporal_context=False and no conflict_resolution, returns flat text."""
        from mabench.outgestion import OutgestionMixin

        class FakeAgent(OutgestionMixin):
            conflict_resolution = False
            temporal_context = False
            contradiction_context = False

        agent = FakeAgent()
        chunks = ["alpha", "beta"]
        ctx, instructions = agent._format_outgestion(chunks, {})
        assert ctx == "alpha\n\nbeta"
        assert instructions == ""

    def test_temporal_ordering_with_metadata(self):
        """When temporal_context=True, chunks are sorted by recency and tagged."""
        from mabench.outgestion import OutgestionMixin

        class FakeAgent(OutgestionMixin):
            conflict_resolution = False
            temporal_context = True
            contradiction_context = False

        agent = FakeAgent()
        chunks = ["newer fact", "older fact"]
        metadata = {
            "newer fact"[:200]: {"recency": 0.9},
            "older fact"[:200]: {"recency": 0.1},
        }
        ctx, instructions = agent._format_outgestion(chunks, metadata)
        # Older should come first (ascending recency sort)
        assert ctx.index("older fact") < ctx.index("newer fact")
        assert "oldest" in instructions.lower() or "newest" in instructions.lower()


class TestMROOrder:
    """OutgestionMixin must appear in MRO between ConflictResolutionMixin and object."""

    def test_outgestion_in_mro(self):
        from mabench.hermes_agent import HermesMemoryAgent
        from mabench.outgestion import OutgestionMixin

        mro_names = [c.__name__ for c in HermesMemoryAgent.__mro__]
        assert "OutgestionMixin" in mro_names
        idx = mro_names.index("OutgestionMixin")
        # Should be after the class itself
        assert idx > 0
        # Should be before object
        assert idx < mro_names.index("object")
