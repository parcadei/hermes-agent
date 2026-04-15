"""Ontology primitives for the Hermes Foundry memory system.

This module implements the core type system described in `foundry/spec.md`
section 1. It intentionally stays small and deterministic so it can be used
by storage, ingestion, and query layers without pulling in third-party
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from typing import Optional


_NUMERIC_RE = re.compile(r"^-?\$?\d[\d,]*\.?\d*(?:[A-Za-z%]+)?$")
_DURATION_RE = re.compile(
    r"^\s*\d+\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_ORDINAL_SUFFIX_RE = re.compile(r"(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)
_LIST_RE = re.compile(r"\b(?:and|or)\b", re.IGNORECASE)
_BOOLEAN_LITERALS = {"true", "false", "yes", "no"}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %Y",
    "%b %Y",
)


class Facet(str, Enum):
    """The 9 universal facets. Every entity is described by some subset."""

    WHAT = "what"
    WHO = "who"
    WHEN = "when"
    WHERE = "where"
    QUANTITY = "quantity"
    FREQUENCY = "frequency"
    HOW = "how"
    SENTIMENT = "sentiment"
    STATUS = "status"


class PropertyType(str, Enum):
    """Inferred type of a fact's object value."""

    STRING = "string"
    NUMERIC = "numeric"
    DATE = "date"
    DURATION = "duration"
    BOOLEAN = "boolean"
    ENUM = "enum"
    LIST = "list"


def _strip_ordinal_suffixes(value: str) -> str:
    return _ORDINAL_SUFFIX_RE.sub(r"\1", value)


def _looks_like_date(value: str) -> bool:
    cleaned = _strip_ordinal_suffixes(value.strip())
    if not cleaned:
        return False

    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(cleaned, fmt)
            return True
        except ValueError:
            continue
    return False


def infer_property_type(value: str) -> PropertyType:
    """Infer the property type for a string value using ordered rules."""

    text = value if isinstance(value, str) else str(value)
    stripped = text.strip()

    if not stripped:
        return PropertyType.STRING

    if _NUMERIC_RE.fullmatch(stripped):
        return PropertyType.NUMERIC

    if _looks_like_date(stripped):
        return PropertyType.DATE

    if _DURATION_RE.search(stripped):
        return PropertyType.DURATION

    if stripped.lower() in _BOOLEAN_LITERALS:
        return PropertyType.BOOLEAN

    if "," in stripped and _LIST_RE.search(stripped):
        return PropertyType.LIST

    return PropertyType.STRING


@dataclass
class EntityType:
    """Inferred type descriptor for an entity."""

    primary: str
    secondary: set[str] = field(default_factory=set)
    count: int = 0

    def __post_init__(self) -> None:
        self.primary = self.normalize(self.primary)
        self.secondary = {self.normalize(t) for t in self.secondary if self.normalize(t)}

    @staticmethod
    def normalize(type_str: str) -> str:
        """Lowercase, strip, convert whitespace to underscores, and depluralize."""

        if not isinstance(type_str, str):
            type_str = str(type_str)

        cleaned = type_str.strip().lower()
        if not cleaned:
            return ""

        normalized = "_".join(cleaned.split())
        parts = normalized.split("_")
        return "_".join(EntityType.depluralize(part) for part in parts)

    @staticmethod
    def depluralize(word: str) -> str:
        """Apply naive English depluralization rules."""

        if not isinstance(word, str):
            word = str(word)

        if not word:
            return ""

        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"

        if word.endswith(("ses", "xes", "zes")) and len(word) > 3:
            return word[:-2]

        if (
            len(word) > 2
            and word.endswith("s")
            and not word.endswith(("ss", "us"))
        ):
            return word[:-1]

        return word


class Cardinality(str, Enum):
    """Cardinality contract for typed links."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


@dataclass(frozen=True)
class LinkType:
    """Schema definition for a relationship between entity types."""

    name: str
    inverse_name: str
    cardinality: Cardinality = Cardinality.MANY_TO_MANY
    source_type: str = ""
    target_type: str = ""


class GroundingLayer(str, Enum):
    """Which grounding layer a fact belongs to."""

    STATED = "stated"
    INFERRED = "inferred"
    DERIVED = "derived"


@dataclass(frozen=True)
class Interface:
    """A common shape for entity type polymorphism."""

    name: str
    required_facets: frozenset[str]
    optional_facets: frozenset[str] = frozenset()
    description: str = ""


BUILTIN_INTERFACES = {
    "locatable": Interface("locatable", frozenset({Facet.WHERE.value})),
    "temporal": Interface("temporal", frozenset({Facet.WHEN.value})),
    "quantified": Interface("quantified", frozenset({Facet.QUANTITY.value})),
    "recurring": Interface("recurring", frozenset({Facet.FREQUENCY.value})),
    "person": Interface("person", frozenset({Facet.WHO.value})),
    "rated": Interface("rated", frozenset({Facet.SENTIMENT.value})),
}


@dataclass(frozen=True)
class ValueType:
    """Semantic wrapper around a property type with validation constraints."""

    name: str
    base_type: PropertyType
    validator: Optional[str] = None
    unit: str = ""


__all__ = [
    "BUILTIN_INTERFACES",
    "Cardinality",
    "EntityType",
    "Facet",
    "GroundingLayer",
    "Interface",
    "LinkType",
    "PropertyType",
    "ValueType",
    "infer_property_type",
]
