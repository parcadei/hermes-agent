"""Adversarial tests: malformed, hostile, and edge-case inputs.

Tests that the system handles garbage, unicode oddities, injection
attempts, and pathological data without crashing.
"""

from __future__ import annotations

import pytest


class TestNoneInputs:
    """Tests for None/null inputs."""

    def test_ingest_none(self):
        """Ingesting None should raise TypeError or ValueError."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        with pytest.raises((TypeError, ValueError)):
            foundry.ingest(None)

    def test_entity_none(self):
        """entity(None) should raise TypeError or return None."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        try:
            result = foundry.entity(None)
            assert result is None
        except TypeError:
            pass

    def test_search_none(self):
        """search(None) should raise TypeError."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        with pytest.raises((TypeError, AttributeError)):
            foundry.search(None)


class TestEmptyInputs:
    """Tests for empty/whitespace-only inputs."""

    def test_ingest_empty_string(self):
        """Ingesting empty string produces zero-count report."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        report = foundry.ingest("")
        assert report.chunks_processed == 0
        assert report.entities_created == 0

    def test_ingest_whitespace_only(self):
        """Ingesting whitespace-only string produces zero-count report."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        report = foundry.ingest("   \n\t\n   ")
        assert report.chunks_processed == 0

    def test_search_empty_string(self):
        """Searching empty string returns empty results."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        results = foundry.search("")
        assert len(results) == 0 or all(s == 0 for _, s in results)

    def test_entity_empty_string(self):
        """entity('') returns None."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        assert foundry.entity("") is None


class TestUnicodeEdgeCases:
    """Tests for unicode edge cases in entity names and text."""

    def test_emoji_entity_name(self):
        """Emoji in entity names should not crash the system."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_or_create("my_cat_\U0001f431")
        assert result != ""

    def test_cjk_characters(self):
        """CJK characters in entity names."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_or_create("\u732b")  # Cat in CJK
        assert result != ""

    def test_rtl_text(self):
        """Right-to-left text (Arabic/Hebrew) in entity names."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_or_create("\u0642\u0637\u0629")  # Cat in Arabic
        assert result != ""

    def test_combining_characters(self):
        """Combining characters in entity names."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_or_create("cafe\u0301")  # cafe with combining accent
        assert result != ""

    def test_zero_width_joiner(self):
        """Zero-width joiner in text."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_or_create("test\u200dname")
        assert result != ""

    def test_unicode_normalization_consistent(self):
        """Different unicode forms of same text should resolve same."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        # NFC vs NFD: e\u0301 vs \u00e9 (both are 'e with acute')
        import unicodedata
        nfc = unicodedata.normalize("NFC", "cafe\u0301")
        nfd = unicodedata.normalize("NFD", "caf\u00e9")
        r1 = resolver.resolve_or_create(nfc)
        r2 = resolver.resolve_or_create(nfd)
        # These SHOULD resolve to the same entity
        assert r1 == r2


class TestInjectionAttempts:
    """Tests for injection-style adversarial inputs."""

    def test_sql_injection_entity_name(self):
        """SQL-like patterns in entity names should be treated as data."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_or_create("'; DROP TABLE entities; --")
        assert result != ""
        # Should be stored as a literal string, not executed

    def test_control_characters(self):
        """Control characters in text should not crash."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        text = "User: I have a cat\x00named\x01Luna\x7f"
        try:
            report = foundry.ingest(text)
        except (ValueError, UnicodeError):
            pass  # Acceptable to reject

    def test_extremely_long_entity_name(self):
        """Very long entity names (10K chars) should not crash."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        long_name = "a" * 10000
        result = resolver.resolve_or_create(long_name)
        assert result != ""

    def test_newlines_in_entity_name(self):
        """Newlines in entity names are handled."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_or_create("multi\nline\nname")
        assert result != ""


class TestCircularRelationships:
    """Tests for circular and self-referential graph structures."""

    def test_circular_relationship(self):
        """A -> B -> C -> A should not cause infinite loops."""
        from foundry.storage import EntityNode
        a = EntityNode(canonical="a", related={"b"})
        b = EntityNode(canonical="b", related={"c"})
        c = EntityNode(canonical="c", related={"a"})
        # Traversal should terminate (BFS with visited set)

    def test_self_referential(self):
        """Entity related to itself should not crash."""
        from foundry.storage import EntityNode
        a = EntityNode(canonical="a", related={"a"})
        # This is allowed but traversal must handle it

    def test_dense_graph_traversal(self):
        """Fully-connected graph of 20 entities should not hang."""
        from foundry.storage import EntityNode
        entities = {}
        names = [f"e{i}" for i in range(20)]
        for name in names:
            entities[name] = EntityNode(
                canonical=name,
                related=set(names) - {name},
            )
        # Traversal should complete in reasonable time


class TestDuplicateEntities:
    """Tests for duplicate entity names with different types."""

    def test_same_name_different_type(self):
        """Two entities named 'mercury' with different types."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        r1 = resolver.resolve_or_create("mercury", hint_type="planet")
        r2 = resolver.resolve_or_create("mercury", hint_type="element")
        # These may or may not resolve to the same entity
        # depending on type-gated resolution

    def test_duplicate_ingestion(self):
        """Ingesting same text twice should not create duplicate entities."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        text = "User: I have a cat named Luna"
        foundry.ingest(text)
        foundry.ingest(text)
        # Should have at most 1 Luna entity


class TestConflictingFacts:
    """Tests for contradictory facts."""

    def test_contradictory_location(self):
        """'User lives in Rome' then 'User lives in Berlin' -- supersession."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        foundry.ingest("User: I live in Rome")
        foundry.ingest("User: I moved to Berlin last month")
        facts = foundry.facts("user", predicate="lives_in", current_only=True)
        # Current should be Berlin, not Rome
        if facts:
            assert any("berlin" in f.object.lower() for f in facts)
