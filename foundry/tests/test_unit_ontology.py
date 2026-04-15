"""Unit tests for the Ontology Layer (spec section 1).

Tests all ontology types: Facet enum, PropertyType inference,
EntityType normalization/depluralization, LinkType, Cardinality,
and GroundingLayer contracts.
"""

from __future__ import annotations

import pytest
from datetime import datetime


# ---------------------------------------------------------------------------
# Facet enum
# ---------------------------------------------------------------------------

class TestFacet:
    """Tests for the Facet enum (9 universal facets)."""

    def test_all_nine_facets_exist(self):
        """Facet enum must define exactly 9 members."""
        from foundry.ontology import Facet
        assert len(Facet) == 9

    def test_facet_values(self):
        """Each facet has a lowercase string value matching its purpose."""
        from foundry.ontology import Facet
        expected = {
            "WHAT": "what",
            "WHO": "who",
            "WHEN": "when",
            "WHERE": "where",
            "QUANTITY": "quantity",
            "FREQUENCY": "frequency",
            "HOW": "how",
            "SENTIMENT": "sentiment",
            "STATUS": "status",
        }
        for name, value in expected.items():
            assert Facet[name].value == value, f"Facet.{name} should be '{value}'"

    def test_facet_is_string_enum(self):
        """Facet members are strings (for use as dict keys)."""
        from foundry.ontology import Facet
        for f in Facet:
            assert isinstance(f, str)
            assert isinstance(f.value, str)

    def test_facet_iteration_order(self):
        """Facets iterate in declaration order."""
        from foundry.ontology import Facet
        names = [f.name for f in Facet]
        assert names == [
            "WHAT", "WHO", "WHEN", "WHERE",
            "QUANTITY", "FREQUENCY", "HOW", "SENTIMENT", "STATUS",
        ]


# ---------------------------------------------------------------------------
# PropertyType inference
# ---------------------------------------------------------------------------

class TestPropertyType:
    """Tests for PropertyType enum and infer_property_type function."""

    def test_all_property_types_exist(self):
        """PropertyType must define exactly 7 members."""
        from foundry.ontology import PropertyType
        assert len(PropertyType) == 7

    def test_property_type_values(self):
        """Each PropertyType has correct string value."""
        from foundry.ontology import PropertyType
        expected = ["string", "numeric", "date", "duration", "boolean", "enum", "list"]
        actual = [pt.value for pt in PropertyType]
        assert actual == expected


class TestInferPropertyType:
    """Tests for deterministic type inference from value strings."""

    def test_numeric_integer(self):
        """Plain integers are NUMERIC."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("42") == PropertyType.NUMERIC

    def test_numeric_float(self):
        """Decimal numbers are NUMERIC."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("3.14") == PropertyType.NUMERIC

    def test_numeric_dollar_amount(self):
        """Dollar amounts ($800) are NUMERIC."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("$800") == PropertyType.NUMERIC

    def test_numeric_with_commas(self):
        """Numbers with commas (5,850) are NUMERIC."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("5,850") == PropertyType.NUMERIC

    def test_numeric_negative(self):
        """Negative numbers are NUMERIC."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("-17") == PropertyType.NUMERIC

    def test_numeric_with_unit(self):
        """Numbers with memory units (16GB) are NUMERIC."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("16GB") == PropertyType.NUMERIC

    def test_date_iso_format(self):
        """ISO date strings are DATE."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("2024-06-15") == PropertyType.DATE

    def test_date_natural_language(self):
        """Natural language dates (June 3rd, 2024) are DATE."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("June 3rd, 2024") == PropertyType.DATE

    def test_date_month_year(self):
        """Month-year strings (March 2024) are DATE."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("March 2024") == PropertyType.DATE

    def test_duration_weeks(self):
        """Duration expressions (3 weeks) are DURATION."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("3 weeks") == PropertyType.DURATION

    def test_duration_minutes(self):
        """Duration expressions (45 minutes) are DURATION."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("45 minutes") == PropertyType.DURATION

    def test_duration_hours(self):
        """Duration expressions (50 hours/week) are DURATION or STRING depending on rules."""
        from foundry.ontology import infer_property_type, PropertyType
        result = infer_property_type("50 hours/week")
        assert result in (PropertyType.DURATION, PropertyType.STRING)

    def test_duration_years(self):
        """Duration expressions (2 years) are DURATION."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("2 years") == PropertyType.DURATION

    def test_boolean_true(self):
        """Boolean literal 'true' is BOOLEAN."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("true") == PropertyType.BOOLEAN

    def test_boolean_false(self):
        """Boolean literal 'false' is BOOLEAN."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("false") == PropertyType.BOOLEAN

    def test_boolean_yes(self):
        """Boolean literal 'yes' is BOOLEAN."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("yes") == PropertyType.BOOLEAN

    def test_boolean_no(self):
        """Boolean literal 'no' is BOOLEAN."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("no") == PropertyType.BOOLEAN

    def test_list_comma_and(self):
        """Comma + 'and' separated values are LIST."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("cats, dogs, and fish") == PropertyType.LIST

    def test_list_comma_or(self):
        """Comma + 'or' separated values are LIST."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("red, blue, or green") == PropertyType.LIST

    def test_string_default(self):
        """Plain text that doesn't match any pattern is STRING."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("very friendly") == PropertyType.STRING

    def test_string_empty(self):
        """Empty string is STRING."""
        from foundry.ontology import infer_property_type, PropertyType
        assert infer_property_type("") == PropertyType.STRING

    def test_priority_numeric_over_date(self):
        """Rule: numeric is checked before date. '2024' should be NUMERIC (not DATE)."""
        from foundry.ontology import infer_property_type, PropertyType
        # A standalone year could be ambiguous; the spec says numeric regex first.
        result = infer_property_type("2024")
        assert result == PropertyType.NUMERIC


# ---------------------------------------------------------------------------
# EntityType normalization and depluralization
# ---------------------------------------------------------------------------

class TestEntityType:
    """Tests for EntityType normalization and depluralization."""

    def test_normalize_basic(self):
        """Basic normalization: lowercase, strip, spaces to underscores."""
        from foundry.ontology import EntityType
        assert EntityType.normalize("Pet Fish") == "pet_fish"

    def test_normalize_depluralize(self):
        """Normalization includes depluralization."""
        from foundry.ontology import EntityType
        assert EntityType.normalize("bikes") == "bike"

    def test_normalize_luxury_items(self):
        """Multi-word plural: 'luxury items' -> 'luxury_item'."""
        from foundry.ontology import EntityType
        assert EntityType.normalize("luxury items") == "luxury_item"

    def test_normalize_idempotent(self):
        """Invariant: normalize(normalize(t)) == normalize(t)."""
        from foundry.ontology import EntityType
        cases = ["Pet Fish", "bikes", "luxury items", "restaurant", "dog"]
        for t in cases:
            once = EntityType.normalize(t)
            twice = EntityType.normalize(once)
            assert once == twice, f"normalize is not idempotent for '{t}': '{once}' vs '{twice}'"

    def test_normalize_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        from foundry.ontology import EntityType
        assert EntityType.normalize("  cat  ") == "cat"

    def test_normalize_preserves_underscores(self):
        """Existing underscores are preserved."""
        from foundry.ontology import EntityType
        assert EntityType.normalize("pet_fish") == "pet_fish"

    def test_normalize_lowercase(self):
        """All characters are lowercased."""
        from foundry.ontology import EntityType
        assert EntityType.normalize("RESTAURANT") == "restaurant"


class TestDepluralize:
    """Tests for naive English depluralization rules."""

    def test_ies_to_y(self):
        """'hobbies' -> 'hobby' (ies -> y when len > 4)."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("hobbies") == "hobby"

    def test_ses_to_s(self):
        """'buses' -> 'bus' (ses -> drop es)."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("buses") == "bus"

    def test_xes_to_x(self):
        """'boxes' -> 'box' (xes -> drop es)."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("boxes") == "box"

    def test_zes_to_z(self):
        """'buzzes' -> 'buzz' (zes -> drop es)."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("buzzes") == "buzz"

    def test_trailing_s_basic(self):
        """'bikes' -> 'bike' (trailing s dropped)."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("bikes") == "bike"

    def test_trailing_s_cats(self):
        """'cats' -> 'cat'."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("cats") == "cat"

    def test_no_depluralize_ss(self):
        """'glass' should NOT drop s (ends in 'ss')."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("glass") == "glass"

    def test_no_depluralize_short(self):
        """Very short words (<=2 chars like 'us') should not be depluralized."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("us") == "us"

    def test_no_depluralize_mass_nouns(self):
        """'fish' stays 'fish' (no trailing 's'), 'equipment' stays."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("fish") == "fish"
        assert EntityType.depluralize("equipment") == "equipment"

    def test_depluralize_idempotent(self):
        """Invariant: depluralize(depluralize(t)) == depluralize(t)."""
        from foundry.ontology import EntityType
        cases = ["hobbies", "bikes", "cats", "fish", "glass", "buses"]
        for t in cases:
            once = EntityType.depluralize(t)
            twice = EntityType.depluralize(once)
            assert once == twice, f"depluralize not idempotent for '{t}': '{once}' vs '{twice}'"

    def test_depluralize_items(self):
        """'items' -> 'item'."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("items") == "item"

    def test_depluralize_cities(self):
        """'cities' -> 'city'."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("cities") == "city"

    def test_depluralize_babies(self):
        """'babies' -> 'baby'."""
        from foundry.ontology import EntityType
        assert EntityType.depluralize("babies") == "baby"


class TestEntityTypeObject:
    """Tests for the EntityType dataclass itself."""

    def test_primary_from_what_facet(self):
        """EntityType.primary is derived from the 'what' facet."""
        from foundry.ontology import EntityType
        et = EntityType(primary="restaurant")
        assert et.primary == "restaurant"

    def test_empty_primary_when_no_what(self):
        """Entity with no 'what' facet has primary = ''."""
        from foundry.ontology import EntityType
        et = EntityType(primary="")
        assert et.primary == ""

    def test_secondary_types(self):
        """Secondary types can be added."""
        from foundry.ontology import EntityType
        et = EntityType(primary="dog", secondary={"pet", "animal"})
        assert "pet" in et.secondary
        assert "animal" in et.secondary

    def test_count_default(self):
        """Count defaults to 0."""
        from foundry.ontology import EntityType
        et = EntityType(primary="cat")
        assert et.count == 0


# ---------------------------------------------------------------------------
# LinkType
# ---------------------------------------------------------------------------

class TestLinkType:
    """Tests for LinkType (directed relationships between entities)."""

    def test_link_type_creation(self):
        """LinkType can be created with name and inverse."""
        from foundry.ontology import LinkType
        lt = LinkType(name="visited", inverse_name="visited_by")
        assert lt.name == "visited"
        assert lt.inverse_name == "visited_by"

    def test_link_type_default_cardinality(self):
        """Default cardinality is MANY_TO_MANY."""
        from foundry.ontology import LinkType, Cardinality
        lt = LinkType(name="owns", inverse_name="owned_by")
        assert lt.cardinality == Cardinality.MANY_TO_MANY

    def test_link_type_with_type_constraints(self):
        """LinkType can have source_type and target_type."""
        from foundry.ontology import LinkType
        lt = LinkType(
            name="works_at", inverse_name="employer_of",
            source_type="person", target_type="company",
        )
        assert lt.source_type == "person"
        assert lt.target_type == "company"

    def test_link_type_frozen(self):
        """LinkType is frozen (immutable)."""
        from foundry.ontology import LinkType
        lt = LinkType(name="owns", inverse_name="owned_by")
        with pytest.raises(AttributeError):
            lt.name = "other"

    def test_link_type_hashable(self):
        """Frozen LinkType is hashable (usable in sets/dicts)."""
        from foundry.ontology import LinkType
        lt = LinkType(name="owns", inverse_name="owned_by")
        s = {lt}
        assert lt in s


# ---------------------------------------------------------------------------
# Cardinality
# ---------------------------------------------------------------------------

class TestCardinality:
    """Tests for the Cardinality enum."""

    def test_all_four_cardinalities(self):
        """Cardinality has exactly 4 members."""
        from foundry.ontology import Cardinality
        assert len(Cardinality) == 4

    def test_cardinality_values(self):
        """Each cardinality has the expected string value."""
        from foundry.ontology import Cardinality
        assert Cardinality.ONE_TO_ONE.value == "one_to_one"
        assert Cardinality.ONE_TO_MANY.value == "one_to_many"
        assert Cardinality.MANY_TO_ONE.value == "many_to_one"
        assert Cardinality.MANY_TO_MANY.value == "many_to_many"


# ---------------------------------------------------------------------------
# GroundingLayer
# ---------------------------------------------------------------------------

class TestGroundingLayer:
    """Tests for the GroundingLayer enum (3 layers)."""

    def test_three_layers(self):
        """GroundingLayer has exactly 3 members."""
        from foundry.ontology import GroundingLayer
        assert len(GroundingLayer) == 3

    def test_layer_values(self):
        """Each layer has correct string value."""
        from foundry.ontology import GroundingLayer
        assert GroundingLayer.STATED.value == "stated"
        assert GroundingLayer.INFERRED.value == "inferred"
        assert GroundingLayer.DERIVED.value == "derived"

    def test_layer_is_string(self):
        """GroundingLayer members are strings."""
        from foundry.ontology import GroundingLayer
        for layer in GroundingLayer:
            assert isinstance(layer, str)
