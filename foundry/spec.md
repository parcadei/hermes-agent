# Hermes Foundry — Behavioral Specification

**Version**: 1.0
**Date**: 2026-04-13
**Status**: AUTHORITATIVE — this document is the source of truth for all implementation phases.

---

## 0. Preamble

Hermes Foundry is a ground-up memory operating system for conversational AI. It replaces all existing mabench infrastructure with a unified, forward-compatible architecture. Existing data can be re-extracted and re-ingested through the new system; backward compatibility with mabench code is explicitly not a goal.

### Design Axioms

1. **BM25 is primary.** 58.3% SubEM baseline with zero infrastructure. Everything else adds on top of keyword search, never replaces it.
2. **No LLM calls for infrastructure.** Memory must be cheap to operate. LLM calls are reserved for extraction (ingestion-time) and answer generation (query-time). All query routing, entity resolution, link traversal, temporal computation, and aggregation are deterministic.
3. **The entity is the unit of query.** Not the chunk, not the triple — the entity. Foundry's Ontology insight: entities are real-world things with typed properties, linked to each other, queryable through structured filter chains.
4. **Always-on, never routed.** Every subsystem runs unconditionally. There is no intent-based routing. Systems contribute when relevant and are neutral when not.
5. **Stated over inferred.** The graph stores the user's exact words, not the model's interpretation. Grounding verification enforces this. Inferred data is navigation-only; derived data is deterministic computation over stated facts.
6. **Index at ingestion, query at read.** Pre-compute everything that can be pre-computed. Query time is for composing pre-built indexes, not computing from scratch.
7. **Traces first, deterministic second, parametric third.** The agentic retrieval architecture captures traces for pattern extraction, hardcodes common patterns as deterministic pipelines, and distills uncommon patterns into small specialist models.

### Proven Results to Preserve

These validated findings constrain the design:

| Finding | Source | Implication |
|---------|--------|-------------|
| BM25 baseline: 58.3% SubEM | ablation 01 | BM25 is the primary retrieval channel |
| Entity pre-filter: +6% SubEM | LME eval | Entity BM25 as RRF supplement to chunk BM25 |
| mxbai-rerank: 97.2% of reachable in top-10 | ablation 05 | Ranking is solved; recall is the bottleneck |
| Raw chunks > triples for LLM context | LME eval | Raw text stored alongside structured data |
| Aggregation overfetch=25: +0.67% | LME eval | Counting queries need more candidates |
| Agentic retrieval: 43.3% SubEM | RESULTS.md | Multi-step search beats single-pass by 14.6pp |
| Qwen2.5-7B: 3% hallucination | faceted extraction | Best extraction model for faceted schema |
| VerbNet verb-class: validated | MSL v2 | 90 acquisition, 54 maintenance, 32 intent lemmas |
| Conflict resolution per-hop: wins | FC eval | CR at each hop of multi-hop traversal |
| 53% of questions structurally unreachable | ablation 07 | Need temporal engine, aggregation, computation |

### DO NOT List (Proven Failures)

These approaches are excluded from the design by empirical evidence:

| Approach | Result | Reference |
|----------|--------|-----------|
| Dream consolidation (batch) | SubEM=3-15 (catastrophic embedding drift) | MEMORY.md |
| Chain-of-thought for retrieval | SubEM=18 (hurts, retrieval-bound) | MEMORY.md |
| Cardinality gating / NLI | No value (killed) | failed-approaches.md |
| Intent-based routing | Everything must be always-on | MEMORY.md |
| LLM calls for infrastructure | Must be deterministic | MEMORY.md |
| Cosine dedup | Disproven (geometry can't distinguish conflicts) | failed-approaches.md |
| Recency sort | SubEM=5 (catastrophic) | failed-approaches.md |
| Cross-encoder + entity_prefilter | -5.67% SubEM (xenc shuffles correct results) | failed-approaches.md |
| RRF fusion of entity channel | E2-E6 all regressed | RESULTS.md |
| Noun-chunk retrieval | Neutral (+0.34% noise) | failed-approaches.md |
| Exact-words prompt | -1.33% SubEM | failed-approaches.md |
| Set-union merge | -1.00% SubEM | failed-approaches.md |
| Event calendar as RRF channel | -1.00% SubEM (dilutes BM25) | failed-approaches.md |

---

## 1. Ontology Layer

### 1.1 Contract

The Ontology Layer defines the type system for all entities in memory. It provides:
- A finite set of universal facets that describe any entity
- A type hierarchy based on shape (which facets are populated), not named types
- Property typing for structured queries (numeric comparison, date range, etc.)
- Link types with cardinality for entity-to-entity relationships
- A grounding model that separates user's words from model interpretation

### 1.2 Universal Facets

Nine facets define the shape of every entity. An entity's type is implicit in which facets it has and what values they carry. This eliminates the need for an explicit type taxonomy.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Facet(str, Enum):
    """The 9 universal facets. Every entity is described by some subset."""
    WHAT = "what"           # Category or kind: "trip", "pet cat", "medication"
    WHO = "who"             # Related person: "grandmother", "Dr. Smith"
    WHEN = "when"           # Date or time: "June 3rd", "since 2020"
    WHERE = "where"         # Location: "Miami", "Serenity Yoga"
    QUANTITY = "quantity"    # Amount/count/cost/duration: "50 hours/week", "$800"
    FREQUENCY = "frequency" # How often: "5x/week", "daily"
    HOW = "how"             # Method or means: "online course", "by bus"
    SENTIMENT = "sentiment" # Opinion or preference: "loves it", "allergic"
    STATUS = "status"       # Current state: "active", "beginner", "expired"
```

### 1.3 Property Types

Every fact value has an inferred type for structured query operations.

```python
class PropertyType(str, Enum):
    """Inferred type of a fact's object value."""
    STRING = "string"       # Default. Free text.
    NUMERIC = "numeric"     # Parseable as float: "$800", "16GB", "5,850"
    DATE = "date"           # Parseable as datetime: "June 3rd, 2024"
    DURATION = "duration"   # Parseable as timedelta: "3 weeks", "45 minutes"
    BOOLEAN = "boolean"     # "yes", "no", "true", "false"
    ENUM = "enum"           # Closed set (inferred from repeated predicates)
    LIST = "list"           # Comma-separated or conjunction-joined values


def infer_property_type(value: str) -> PropertyType:
    """Deterministic type inference from value string.

    Rules (checked in order):
    1. Regex for numeric: r'-?\$?\d[\d,]*\.?\d*' → NUMERIC
    2. dateparser.parse() succeeds → DATE
    3. Duration patterns: r'\d+\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)' → DURATION
    4. Boolean literals → BOOLEAN
    5. Contains comma + 'and'/'or' → LIST
    6. Default → STRING
    """
    ...
```

### 1.4 Entity Type System

Types are not declared — they are inferred from facets. An entity with `what="restaurant"` is implicitly of type "restaurant". The `what` facet is the primary type signal.

```python
@dataclass
class EntityType:
    """Inferred type descriptor for an entity."""
    primary: str                    # From 'what' facet (normalized, depluralized)
    secondary: set[str] = field(default_factory=set)  # Additional type labels
    count: int = 0                  # How many times this type was assigned (majority vote)

    @staticmethod
    def normalize(type_str: str) -> str:
        """Lowercase, strip, spaces to underscores, depluralize.

        'Pet Fish' → 'pet_fish'
        'bikes' → 'bike'
        'luxury items' → 'luxury_item'
        """
        ...

    @staticmethod
    def depluralize(word: str) -> str:
        """Naive English depluralization.

        Rules:
        - 'ies' → 'y' (hobbies → hobby) when len > 4
        - 'ses'/'xes'/'zes' → drop 'es' (buses → bus)
        - trailing 's' → drop (bikes → bike) unless 'ss' or len <= 2
        - Otherwise: identity (fish, equipment)
        """
        ...
```

**Invariants:**
- `EntityType.normalize(t)` is idempotent: `normalize(normalize(t)) == normalize(t)`
- `depluralize(depluralize(t)) == depluralize(t)`
- Every entity has at most one `primary` type (from `what` facet)
- An entity with no `what` facet has `primary = ""`

### 1.5 Link Types

Links are typed, directed relationships between entities with explicit cardinality.

```python
class Cardinality(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


@dataclass(frozen=True)
class LinkType:
    """Schema definition for a relationship between entity types.

    Links are directional: source → target. The inverse_name provides
    the reverse direction label for traversal.
    """
    name: str                       # Forward label: "visited", "owns", "works_at"
    inverse_name: str               # Reverse label: "visited_by", "owned_by", "employer_of"
    cardinality: Cardinality = Cardinality.MANY_TO_MANY
    source_type: str = ""           # Optional type constraint on source entity
    target_type: str = ""           # Optional type constraint on target entity
```

Links are materialized during ingestion from S-P-O triples. The predicate becomes the link name. Co-occurrence edges (entities mentioned in the same chunk) create implicit `MANY_TO_MANY` links without a named predicate.

<!-- Enriched from Palantir scrape 16 (create-link-type) -->

**Link Implementation Strategies:**

Links can be backed by three underlying storage strategies, depending on cardinality:

| Strategy | Cardinality | How It Works | Hermes Foundry Mapping |
|----------|-------------|--------------|----------------------|
| Foreign key | ONE_TO_ONE, MANY_TO_ONE | Source entity stores target's canonical name directly in a property | Entity `facts["works_at"] = ["acme_corp"]` |
| Co-occurrence | MANY_TO_MANY | Entities co-mentioned in the same chunk create implicit, undirected links | CooccurrenceIndex pairs |
| Predicate edge | ANY | Explicit S-P-O triple creates a directed, named edge | EntityNode.links[predicate] |

For ONE_TO_ONE and MANY_TO_ONE links, the foreign key approach is preferred because it avoids a separate join structure. The predicate itself (from the S-P-O triple) serves as the foreign key reference. For MANY_TO_MANY (co-occurrence), Palantir uses join table datasets; Hermes Foundry uses the CooccurrenceIndex which stores (entity_a, entity_b, chunk_idx) triples.

**Link naming convention** (from Palantir best practices): Links should be named by the relationship they represent, not by the object types they connect. Use `visited` / `visited_by`, not `restaurant_link`. The forward and inverse names should read naturally as sentences: "User `visited` Restaurant" / "Restaurant `visited_by` User".

### 1.6 Grounding Model

Three layers, each an independent data structure. Data flows downward (stated → inferred → derived) but never upward.

```python
class GroundingLayer(str, Enum):
    """Which layer a fact belongs to."""
    STATED = "stated"       # User's exact words, grounding-verified
    INFERRED = "inferred"   # Semantic bridges (synonym/type), disposable cache
    DERIVED = "derived"     # Deterministic gists computed from stated layer
```

**Contract:**
- **STATED**: `fact.object` must substring-match `fact.source_text` (case-insensitive, word-level fallback for reordering). If grounding fails, the fact is rejected.
- **INFERRED**: Never appears in prompts to the LLM. Used only for navigation (traversal from query term to stated entity). Can be wiped and rebuilt without data loss.
- **DERIVED**: Computed deterministically from STATED layer. Includes type counts, type lists, entity progressions, co-occurrence patterns. Always recomputable.

**Invariants:**
- Wiping INFERRED does not affect STATED or DERIVED
- Wiping DERIVED does not affect STATED or INFERRED
- STATED is the only source of truth; answers come only from STATED facts + raw chunks
- Inferred layer cannot overwrite stated layer (separate data structures, not same graph)
- Hallucinated entities in inferred layer do not leak to stated layer

### 1.7 Interfaces (Type Polymorphism)
<!-- Enriched from Palantir scrapes 03, 04, 07 (ontology concepts, interfaces) -->

An **interface** defines a common shape that multiple entity types can implement. This provides polymorphism: a query can target an interface and match all entities implementing it, regardless of their primary type.

```python
@dataclass(frozen=True)
class Interface:
    """A common shape for entity type polymorphism.

    Example: Interface("temporal_entity", required_facets={"when"})
    matches any entity with a populated 'when' facet, regardless of primary type.

    Palantir uses interfaces to allow consistent modeling of object types that share
    common characteristics. Hermes Foundry uses them to enable cross-type queries:
    "entities with a 'where' facet" matches restaurants, gyms, and workplaces alike.
    """
    name: str
    required_facets: frozenset[str]     # Facets that must be populated
    optional_facets: frozenset[str] = frozenset()  # Facets that may be populated
    description: str = ""

# Pre-defined interfaces (derived from facet patterns in LME data)
BUILTIN_INTERFACES = {
    "locatable": Interface("locatable", frozenset({"where"})),
    "temporal": Interface("temporal", frozenset({"when"})),
    "quantified": Interface("quantified", frozenset({"quantity"})),
    "recurring": Interface("recurring", frozenset({"frequency"})),
    "person": Interface("person", frozenset({"who"})),
    "rated": Interface("rated", frozenset({"sentiment"})),
}
```

**Invariant:** An entity implements an interface if and only if all `required_facets` are populated (non-empty string) in the entity's facet profile. Interface membership is recomputed from facets, never stored explicitly.

### 1.8 Shared Properties
<!-- Enriched from Palantir scrapes 04, 07 (shared properties, value types) -->

A **shared property** is a property predicate that appears on multiple entity types with consistent semantics. Palantir uses these for centralized management of property metadata across object types.

In Hermes Foundry, shared properties emerge naturally from extraction: the predicate `"cost"` on a restaurant entity and on a vacation entity both have `PropertyType.NUMERIC` and represent monetary values. The PropertyIndex already handles this -- `lookup("cost")` returns all (entity, value, type) tuples regardless of entity type. No explicit registration is needed.

**Value types** (semantic wrappers around PropertyType) add validation constraints:

```python
@dataclass(frozen=True)
class ValueType:
    """Semantic wrapper around a property type with validation constraints.

    Examples:
    - email: STRING with regex r'^[^@]+@[^@]+\.[^@]+$'
    - url: STRING with regex r'^https?://'
    - currency: NUMERIC with unit "USD|EUR|GBP|..."
    - percentage: NUMERIC with range [0.0, 100.0]
    """
    name: str
    base_type: PropertyType
    validator: Optional[str] = None  # Regex pattern or range expression
    unit: str = ""                   # Unit of measurement
```

### 1.9 Ontology Anti-Patterns
<!-- Enriched from Palantir scrape 05 (best practices and anti-patterns) -->

These anti-patterns from Palantir's ontology design guide apply directly to Hermes Foundry entity modeling. They are listed here to prevent recurring design mistakes during extraction and ingestion.

| Anti-Pattern | Description | Hermes Foundry Mapping | Prevention |
|--------------|-------------|----------------------|------------|
| **System Silos** | Same entity modeled separately per source system | User mentioned as "the user" in turn 5 and "I" in turn 12 creates two entities | EntityResolver cascade (alias -> determiner-strip -> embedding) must unify mentions |
| **Kitchen Sink** | Every extracted detail becomes a property | LLM extracts "the", "a", function words as properties | Grounding verification rejects non-grounded facts; entity_registry_cap=50 limits extraction |
| **God Object** | One entity represents multiple distinct real-world things | "user" entity accumulates 28K facts across all topics | Per-entity fact cap (max_facts_per_entity=1000); type-specific entities preferred |
| **Action Sprawl** | Many single-property mutations instead of cohesive operations | Per-property supersession without semantic grouping | SupersessionEngine resolves by (entity, predicate) pairs, keeping operations cohesive |
| **The Time Machine** | Historical versions as separate entities | "user's car (2020)" and "user's car (2024)" as different entities | Temporal validity on facts; single entity with supersession history |
| **The Misnomer** | Vague or misleading names | Predicates like "has" or "is" extracted from text | Predicate normalization; extraction prompt requires specific predicate names |
| **Department Silos** | Different extraction runs create overlapping entity sets | Running extraction twice creates duplicate entities | Dedup idempotency invariant (I7): same triple + turn_idx = one fact |

**Palantir's golden rule: "Model reality, not systems."** An entity should represent one real-world thing. Properties should have clear value. Names should be specific and descriptive. Links should be named by the relationship they represent.


---

## 2. Storage Layer

### 2.1 Contract

The Storage Layer provides multiple specialized indexes behind a unified query interface. Each index is independently queryable but composable through the Query Language (Section 3). All indexes are built at ingestion time and are read-only at query time.

### 2.2 Core Data Structures

```python
@dataclass
class Fact:
    """A single predicate-object fact about an entity.

    The atomic unit of structured knowledge. Immutable once created
    (supersession creates a new fact, marks the old one).
    """
    id: str                             # Unique identifier (uuid4)
    subject: str                        # Canonical entity name (normalized)
    predicate: str                      # Normalized predicate (lowercase, underscore-separated)
    object: str                         # Value (string representation)
    object_type: PropertyType           # Inferred type of object value
    turn_idx: int                       # Source conversation turn index
    chunk_idx: int = -1                 # Source chunk index within turn (-1 if turn-level)
    timestamp: Optional[datetime] = None  # Event time (when the fact occurred in the world)
    session_time: Optional[datetime] = None  # Conversation time (when user said it)
    source_text: str = ""               # Original text for grounding verification
    layer: GroundingLayer = GroundingLayer.STATED
    superseded: bool = False            # Marked True when a newer fact replaces this one
    superseded_by: Optional[str] = None # ID of the superseding fact
    when: str = ""                      # Temporal expression from extraction ("since 2020")
    role: str = ""                      # event | state | opinion
    sub_role: str = ""                  # acquisition | maintenance | intent | hypothetical


@dataclass
class EntityNode:
    """A node in the entity graph representing one real-world entity.

    Entities are the primary addressable unit. All queries ultimately
    resolve to entities and their facts.
    """
    canonical: str                      # Normalized canonical name
    aliases: set[str] = field(default_factory=set)      # All known surface forms
    types: set[str] = field(default_factory=set)         # Inferred type labels
    facts: dict[str, list[Fact]] = field(default_factory=dict)  # predicate → [Fact]
    links: dict[str, set[str]] = field(default_factory=dict)    # predicate → {target entity canonicals}
    related: set[str] = field(default_factory=set)       # All connected entity canonicals (co-occ + explicit)
    mention_turns: set[int] = field(default_factory=set) # Turn indices where entity appears
    mention_chunks: set[int] = field(default_factory=set)  # Chunk indices where entity appears
    _type_counts: dict[str, int] = field(default_factory=dict, repr=False)  # type → assignment frequency


@dataclass
class RawChunk:
    """A raw conversation chunk stored for retrieval.

    512-token chunks are proven optimal (monotonic regression at 1024, 2048).
    Raw text is essential — raw chunks > processed triples for LLM context.
    """
    chunk_idx: int                      # Global chunk index (unique)
    text: str                           # Full chunk text
    turn_idx: int                       # First turn index in this chunk
    role: str                           # "user" | "assistant" | "continuation"
    session_time: Optional[datetime] = None  # Conversation timestamp
    entities: list[str] = field(default_factory=list)  # Entity canonicals mentioned


@dataclass
class TemporalEvent:
    """A temporally-grounded event with an Allen interval.

    Events are extracted during ingestion and indexed for temporal queries.
    Intervals are [start_dt, end_dt] inclusive.
    """
    event: str                          # Natural language description
    start_dt: datetime                  # Interval start (inclusive)
    end_dt: datetime                    # Interval end (inclusive)
    source: str                         # ouro_extraction | chat_timestamp | dateparser | manual
    turn_id: Optional[int] = None       # Source conversation turn
    confidence: float = 1.0             # Extraction confidence, 0.0 to 1.0
    subject: Optional[str] = None       # SVO subject entity
    verb: Optional[str] = None          # SVO verb
    object: Optional[str] = None        # SVO object entity

    def __post_init__(self) -> None:
        if self.start_dt > self.end_dt:
            raise ValueError(f"start_dt ({self.start_dt}) must be <= end_dt ({self.end_dt})")
```

### 2.3 Index Types

All indexes implement a common trait: they accept data at ingestion time and return ranked results at query time.

#### 2.3.1 BM25 Index (PRIMARY)

The primary retrieval channel. Proven 58.3% SubEM baseline.

```python
class BM25Index:
    """Okapi BM25 over tokenized documents.

    Contract:
    - add(doc_id, text): tokenizes text, adds to corpus
    - query(q, top_k) -> list[(doc_id, score)]: returns ranked results
    - Scores are non-negative. Zero means no term overlap.
    - Parameters: k1=1.5, b=0.75 (Okapi defaults, proven optimal)
    """

    def add(self, doc_id: int, text: str) -> None: ...
    def query(self, q: str, top_k: int = 10) -> list[tuple[int, float]]: ...
    def __len__(self) -> int: ...
```

Two instances exist:
1. **Chunk BM25**: Over `RawChunk.text` — the primary retrieval path
2. **Entity BM25**: Over entity documents (name + facet values) — entity pre-filter

#### 2.3.2 Vector Index

Semantic similarity for queries where keyword matching fails.

```python
class VectorIndex:
    """Dense vector similarity search.

    Contract:
    - add(doc_id, embedding): stores vector
    - add_batch(doc_ids, embeddings): batch insert
    - query(query_embedding, top_k) -> list[(doc_id, score)]: cosine similarity
    - Embeddings must be L2-normalized before insertion.
    - Score range: [-1.0, 1.0] (cosine similarity)

    Implementation: numpy matmul for < 100K vectors, FAISS for larger.
    Embedding model: configurable (default: all-mpnet-base-v2, 768d).
    """

    def add(self, doc_id: int, embedding: "np.ndarray") -> None: ...
    def add_batch(self, doc_ids: list[int], embeddings: "np.ndarray") -> None: ...
    def query(self, query_embedding: "np.ndarray", top_k: int = 10) -> list[tuple[int, float]]: ...
```

#### 2.3.3 Type Index

Fast lookup from type name to entities.

```python
class TypeIndex:
    """Inverted index from type label to entity canonicals.

    Contract:
    - index(entity_canonical, type_label): adds mapping
    - lookup(type_label) -> list[str]: returns entity canonicals of that type
    - lookup performs depluralization: "bikes" matches entities typed "bike"
    - count(type_label) -> int: number of entities of that type
    """

    def index(self, entity_canonical: str, type_label: str) -> None: ...
    def lookup(self, type_label: str) -> list[str]: ...
    def count(self, type_label: str) -> int: ...
```

#### 2.3.4 Property Index

Reverse lookup from property key to (entity, value) pairs.

```python
class PropertyIndex:
    """Inverted index from property predicate to (entity, value) pairs.

    Contract:
    - index(entity_canonical, predicate, value, property_type): adds mapping
    - lookup(predicate) -> list[(entity_canonical, value, PropertyType)]
    - Supports typed queries: lookup_numeric(pred, op, threshold) for numeric comparison
    - Supports duck typing: find entities that HAVE a property regardless of type
    """

    def index(self, entity: str, predicate: str, value: str, ptype: PropertyType) -> None: ...
    def lookup(self, predicate: str) -> list[tuple[str, str, PropertyType]]: ...
    def lookup_numeric(self, predicate: str, op: str, threshold: float) -> list[tuple[str, float]]: ...
    def entities_with_property(self, predicate: str) -> list[str]: ...
```

#### 2.3.5 Temporal Index (Allen Interval Algebra)

Date/time queries using Allen's 13 interval relations.

```python
class TemporalIndex:
    """Allen interval algebra engine for temporal reasoning.

    Contract:
    - ingest(event: TemporalEvent): adds event to index
    - ingest_batch(events): batch insert with batch embedding
    - match_events(query, top_k, entities?) -> list[TemporalEvent]: hybrid BM25+cosine retrieval
    - events_on_date(date) -> list[TemporalEvent]: point-in-interval query
    - events_between(start, end) -> list[TemporalEvent]: range overlap query
    - before(a, b) -> bool, after(a, b) -> bool: ordering primitives
    - distance(event, reference) -> timedelta: signed duration
    - duration_between(a, b) -> timedelta: gap between non-overlapping events
    - duration_of(event) -> timedelta: event span
    - contains(event, point) -> bool: point-in-interval

    Question classification (5 buckets, regex, no LLM):
    - ordering: "which happened first"
    - distance: "how many days ago"
    - duration_between: "time between X and Y"
    - duration_of: "how long did X last"
    - contains: "what happened on date"
    """

    def ingest(self, event: TemporalEvent) -> None: ...
    def ingest_batch(self, events: list[TemporalEvent]) -> None: ...
    def match_events(self, query: str, top_k: int = 5,
                     entities: list[str] | None = None) -> list[TemporalEvent]: ...
    def events_on_date(self, date: datetime) -> list[TemporalEvent]: ...
    def events_between(self, start: datetime, end: datetime) -> list[TemporalEvent]: ...

    # Allen's 13 interval relations
    def allen_before(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_after(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_meets(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_met_by(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_overlaps(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_overlapped_by(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_starts(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_started_by(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_during(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_contains_interval(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_finishes(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_finished_by(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_equals(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def allen_relation(self, a: TemporalEvent, b: TemporalEvent) -> str: ...

    # Convenience methods (mapped to question buckets)
    def before(self, a: TemporalEvent, b: TemporalEvent) -> bool: ...
    def distance(self, event: TemporalEvent, reference: datetime) -> timedelta: ...
    def duration_between(self, a: TemporalEvent, b: TemporalEvent) -> timedelta: ...
    def duration_of(self, event: TemporalEvent) -> timedelta: ...
    def contains(self, event: TemporalEvent, point: datetime) -> bool: ...

    def answer_temporal_query(self, question: str,
                              reference_date: datetime | None = None) -> dict: ...
```

#### 2.3.6 Co-occurrence Index

Tracks which entities appear together in the same chunk.

```python
class CooccurrenceIndex:
    """Entity co-mention tracking within chunks.

    Contract:
    - record(entity_a, entity_b, chunk_idx): records co-occurrence
    - get_cooccurrences(entity) -> dict[str, int]: entity → count of shared chunks
    - get_shared_chunks(entity_a, entity_b) -> set[int]: chunk indices

    Critical rule: co-occurrence edges form WITHIN a single store() call.
    Never split FC chunks into separate store() calls.
    """

    def record(self, entity_a: str, entity_b: str, chunk_idx: int) -> None: ...
    def get_cooccurrences(self, entity: str) -> dict[str, int]: ...
    def get_shared_chunks(self, entity_a: str, entity_b: str) -> set[int]: ...
```

#### 2.3.7 Raw Chunk Store

Full conversation text for retrieval context.

```python
class RawChunkStore:
    """Stores and retrieves raw conversation chunks.

    Contract:
    - store(chunk: RawChunk): adds chunk
    - get(chunk_idx: int) -> RawChunk: retrieves by index
    - get_by_turn(turn_idx: int) -> list[RawChunk]: all chunks from a turn
    - len() -> int: total stored chunks

    512-token chunk size is proven optimal (monotonic regression at larger sizes).
    """

    def store(self, chunk: RawChunk) -> None: ...
    def get(self, chunk_idx: int) -> RawChunk: ...
    def get_by_turn(self, turn_idx: int) -> list[RawChunk]: ...
    def __len__(self) -> int: ...
```

### 2.4 Persistence Model

```python
class FoundryStore:
    """Unified persistence interface for all indexes.

    Contract:
    - save(path): serializes all indexes to directory
    - load(path): deserializes all indexes from directory
    - Format: one JSON file per index + binary files for embeddings
    - Atomic: either all indexes save/load or none do
    """

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "FoundryStore": ...

    # Sub-stores
    bm25_chunks: BM25Index
    bm25_entities: BM25Index
    vectors: VectorIndex
    types: TypeIndex
    properties: PropertyIndex
    temporal: TemporalIndex
    cooccurrence: CooccurrenceIndex
    chunks: RawChunkStore
    entity_graph: "EntityGraph"   # The STATED layer graph
    inferred_graph: "EntityGraph" # INFERRED layer (navigation-only)
    derived: "DerivedGraph"       # DERIVED layer (computed aggregates)
```

**Invariants:**
- After `load(save(store))`, all indexes return identical query results
- Fact IDs are globally unique and stable across save/load cycles
- Entity canonical names are stable across save/load cycles
- Alias index is fully reconstructed from EntityNode.aliases on load

### 2.5 Indexing Pipeline Architecture
<!-- Enriched from Palantir scrapes 10, 11 (indexing overview, funnel batch pipelines) -->

Palantir's Object Storage V2 uses a **Funnel** service with four-stage batch pipelines for indexing. Hermes Foundry adapts this pattern for its ingestion pipeline:

```
Source Data (conversation text)
  |
  [1. Changelog] — Compute data difference from previous ingestion
  |                 (Palantir: automatic changelog datasets with APPEND transactions)
  |                 (Hermes: diff against existing entity graph + fact store)
  |
  [2. Merge Changes] — Join changelogs with user edits (supersessions)
  |                     (Palantir: merge by primary key)
  |                     (Hermes: merge by (entity_canonical, predicate) pairs)
  |
  [3. Indexing] — Convert merged data into index-specific formats
  |               (Palantir: per-database index files)
  |               (Hermes: populate BM25, Vector, Type, Property, Temporal, Co-occ indexes)
  |
  [4. Hydration] — Make indexed data available for querying
                    (Palantir: download index files to search nodes)
                    (Hermes: in-memory indexes ready immediately; persist on save())
```

**Key Palantir lessons for Hermes Foundry:**

1. **Incremental by default**: OSv2 indexes all pipelines incrementally. Hermes Foundry should support incremental ingestion (add new turns without re-processing existing ones). The `turn_idx` field on Facts and Chunks enables this -- ingestion can skip turns with `turn_idx <= max_existing_turn_idx`.

2. **Live vs. replacement pipelines**: When the schema changes (new facet, new property type), a replacement pipeline re-indexes everything while the live pipeline continues serving queries. Hermes Foundry equivalent: `rebuild_derived()` and `rebuild_inferred()` run in background while existing indexes serve reads.

3. **Edits applied immediately to index**: Palantir applies user edits (Actions) immediately to the in-memory index, then periodically flushes to persistent storage. Hermes Foundry should follow this pattern -- supersession marks facts immediately in the in-memory EntityGraph, with `save()` persisting the final state.

4. **10K object limit (OSv1) vs unlimited (OSv2)**: Palantir's legacy Phonograph had a 10,000 result limit per query. OSv2 removed this. Hermes Foundry has no hard limit on `QueryPlan.execute()` but should respect `max_entities` config (default 100K) as a soft cap.

### 2.6 Searchable Property Hints
<!-- Enriched from Palantir scrape 19 (functions object sets) -->

Not all properties need to be indexed for search. Palantir uses a `Searchable` render hint to mark which properties are indexed for filtering, ordering, and aggregation. In Hermes Foundry, all properties are searchable by default (stored in PropertyIndex), but this concept maps to an optimization opportunity:

```python
class PropertySearchability(str, Enum):
    """Controls how a property participates in search."""
    FULL = "full"       # Indexed in BM25 + PropertyIndex + VectorIndex (default for string)
    STRUCTURED = "structured"  # PropertyIndex only (default for numeric, date, boolean)
    NONE = "none"       # Not indexed (raw storage only, for very long text)
```

**Invariant:** A property with `searchability=NONE` cannot appear in `where()` filters or `order_by()` clauses. It can only be accessed via direct entity fact lookup.


---

## 3. Query Language (ObjectSet Pattern)

### 3.1 Contract

The Query Language provides lazy, immutable, chainable query plans over entities and their facts. Inspired by Palantir Foundry's ObjectSet pattern. Queries build a plan that executes only when a terminal operation is called.

### 3.2 Core Types

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


@dataclass(frozen=True)
class FilterOp:
    """A single filter predicate in a query chain."""
    field: str          # "type", property predicate name, or special field
    operator: str       # $eq, $ne, $lt, $gt, $lte, $gte, $contains, $startsWith, $regex
    value: Any          # Comparison value
    negate: bool = False


@dataclass(frozen=True)
class LogicalOp:
    """Boolean composition of filters."""
    op: str             # "$and" | "$or" | "$not"
    children: tuple[Union[FilterOp, "LogicalOp"], ...]


@dataclass(frozen=True)
class TraversalOp:
    """Link traversal specification."""
    link_predicate: str         # Predicate to traverse (or "" for any)
    direction: str              # "forward" | "backward" | "both"
    max_hops: int = 1           # Maximum traversal depth (proven max useful: 3)
    filter_target: Optional[FilterOp] = None  # Optional filter on target entities


class QueryPlan:
    """Lazy, immutable, chainable query plan over entities.

    Plans accumulate operations. Nothing executes until a terminal
    operation (.execute(), .count(), .first(), .aggregate(), .facts())
    is called. Each chaining method returns a NEW QueryPlan.
    """

    def __init__(self, store: "FoundryStore", seed: Optional[set[str]] = None):
        self._store = store
        self._filters: list[Union[FilterOp, LogicalOp]] = []
        self._traversals: list[TraversalOp] = []
        self._temporal_filters: list[Any] = []
        self._seed: Optional[set[str]] = seed  # Pre-filtered entity set (None = all)
        self._order_by: Optional[tuple[str, str]] = None  # (field, "asc"|"desc")
        self._limit: Optional[int] = None

    # --- Chaining operations (each returns a new QueryPlan) ---

    def type(self, type_label: str) -> "QueryPlan":
        """Filter to entities of a given type.

        Performs depluralization: .type("bikes") matches entities typed "bike".
        """
        ...

    def types(self, type_labels: list[str]) -> "QueryPlan":
        """Filter to entities matching ANY of the given types."""
        ...

    def where(self, predicate: str, operator: str, value: Any) -> "QueryPlan":
        """Filter by property value.

        Operators: $eq, $ne, $lt, $gt, $lte, $gte, $contains, $startsWith, $regex
        Numeric operators ($lt, $gt, $lte, $gte) auto-parse values to float.
        """
        ...

    def filter(self, filter_op: Union[FilterOp, LogicalOp]) -> "QueryPlan":
        """Apply an arbitrary filter operation."""
        ...

    def and_(self, *ops: Union[FilterOp, LogicalOp]) -> "QueryPlan":
        """Logical AND of multiple filters."""
        ...

    def or_(self, *ops: Union[FilterOp, LogicalOp]) -> "QueryPlan":
        """Logical OR of multiple filters."""
        ...

    def not_(self, op: Union[FilterOp, LogicalOp]) -> "QueryPlan":
        """Logical NOT of a filter."""
        ...

    def related(self, link_predicate: str = "",
                direction: str = "both",
                max_hops: int = 1) -> "QueryPlan":
        """Traverse links to related entities (Search Around).

        Empty predicate traverses all link types.
        max_hops capped at 3 (proven diminishing returns beyond).
        """
        ...

    def search_around(self, predicate: str = "", max_hops: int = 1,
                      target_type: str = "") -> "QueryPlan":
        """Palantir-style Search Around: navigate from current set through links.

        Optionally filters target entities by type.
        """
        ...

    # --- Temporal operations ---

    def before(self, date: datetime) -> "QueryPlan":
        """Filter to entities/facts with timestamp before date."""
        ...

    def after(self, date: datetime) -> "QueryPlan":
        """Filter to entities/facts with timestamp after date."""
        ...

    def during(self, start: datetime, end: datetime) -> "QueryPlan":
        """Filter to entities/facts with timestamp within [start, end]."""
        ...

    def overlaps(self, start: datetime, end: datetime) -> "QueryPlan":
        """Filter temporal events overlapping with [start, end]."""
        ...

    # --- Ordering ---

    def order_by(self, field: str, direction: str = "asc") -> "QueryPlan":
        """Order results by a field. Only one order_by per plan."""
        ...

    def limit(self, n: int) -> "QueryPlan":
        """Cap result set size."""
        ...

    # --- Set operations ---

    def union(self, other: "QueryPlan") -> "QueryPlan":
        """Set union: entities in self OR other."""
        ...

    def intersection(self, other: "QueryPlan") -> "QueryPlan":
        """Set intersection: entities in self AND other."""
        ...

    def difference(self, other: "QueryPlan") -> "QueryPlan":
        """Set difference: entities in self but NOT other."""
        ...

    # --- Terminal operations (trigger execution) ---

    def execute(self) -> list[EntityNode]:
        """Materialize the query plan. Returns matching EntityNodes."""
        ...

    def count(self) -> int:
        """Count matching entities without materializing."""
        ...

    def first(self) -> Optional[EntityNode]:
        """Return the first matching entity, or None."""
        ...

    def aggregate(self, property_predicate: str, op: str) -> Union[float, int, list]:
        """Aggregate a property across matching entities.

        op: 'count', 'sum', 'avg', 'min', 'max', 'list', 'distinct'
        For 'count': ignores property_predicate, counts entities.
        For numeric ops: parses property values as float (parse_numeric).
        """
        ...

    def facts(self, predicate: str = "", current_only: bool = True) -> list[Fact]:
        """Collect facts from matching entities.

        Empty predicate returns all facts.
        current_only=True filters out superseded facts.
        Returns newest first (sorted by turn_idx descending).
        """
        ...

    def evidence_turns(self) -> list[int]:
        """Collect evidence turn indices from matching entities.

        Returns turn indices ranked by entity mention frequency.
        """
        ...
```


### 3.2a Extended Filter Operators
<!-- Enriched from Palantir scrapes 17, 18, 19 (TypeScript OSDK, Python OSDK, Functions ObjectSets) -->

Palantir's OSDK provides a rich set of type-specific filter operators beyond simple comparison. Hermes Foundry adopts the following subset relevant to conversational memory:

**String filters** (on any STRING-typed property):

| Operator | Behavior | Example |
|----------|----------|---------|
| `$eq` | Exact match (case-insensitive) | `.where("name", "$eq", "sushi palace")` |
| `$ne` | Not equal | `.where("status", "$ne", "expired")` |
| `$contains` | Substring containment | `.where("name", "$contains", "sushi")` |
| `$startsWith` | Prefix match | `.where("name", "$startsWith", "Dr.")` |
| `$regex` | Regex pattern match | `.where("name", "$regex", r"^Dr\..*")` |
| `$phrase` | All tokens in order, no gaps (from Palantir) | `.where("description", "$phrase", "fresh banana")` |
| `$matchAnyToken` | Any token present (OR semantics) | `.where("description", "$matchAnyToken", "sushi ramen")` |
| `$matchAllTokens` | All tokens present (AND semantics, any order) | `.where("description", "$matchAllTokens", "fresh banana")` |
| `$fuzzy` | Approximate token match (edit distance) | `.where("name", "$fuzzy", "restraunt")` |
| `$isNull` | Property has no value | `.where("cost", "$isNull", True)` |
| `$hasProperty` | Property exists (non-null) | `.where("cost", "$hasProperty", True)` |

**Numeric filters** (on NUMERIC-typed properties, auto-parsed):

| Operator | Behavior | Example |
|----------|----------|---------|
| `$lt` | Less than | `.where("cost", "$lt", 50.0)` |
| `$lte` | Less than or equal | `.where("cost", "$lte", 50.0)` |
| `$gt` | Greater than | `.where("cost", "$gt", 10.0)` |
| `$gte` | Greater than or equal | `.where("age", "$gte", 18)` |
| `$range` | Between (inclusive) | `.where("cost", "$range", (10.0, 50.0))` |

**Boolean filters** (on BOOLEAN-typed properties):

| Operator | Behavior |
|----------|----------|
| `$isTrue` | Value is truthy |
| `$isFalse` | Value is falsy |

**Date filters** (on DATE-typed properties):

| Operator | Behavior |
|----------|----------|
| `$before` | Date before reference |
| `$after` | Date after reference |
| `$between` | Date within range (inclusive) |

**Fuzziness parameter** (from Palantir, backed by ElasticSearch fuzziness): The `$fuzzy` operator accepts an optional fuzziness level controlling edit distance tolerance. Default: `AUTO` (0 edits for 1-2 char terms, 1 edit for 3-5 chars, 2 edits for 6+ chars). This is useful for handling typos in conversational text ("restraunt" matching "restaurant").

### 3.2b Pagination
<!-- Enriched from Palantir scrapes 17, 18, 32 (OSDK pagination) -->

For large result sets, the QueryPlan supports token-based pagination:

```python
class QueryPlan:
    # ... existing methods ...

    def page(self, page_size: int = 100,
             page_token: Optional[str] = None) -> "PageResult":
        """Return a single page of results with a continuation token.

        Pagination is token-based (opaque cursor), not offset-based.
        This matches Palantir's OSDK pattern:
        - page_size: up to 10,000 per page (Palantir limit; Hermes default: 100)
        - page_token: None for first page, then PageResult.next_page_token
        - Results are stable within a pagination session (snapshot isolation)
        """
        ...

    def take(self, n: int) -> list[EntityNode]:
        """Return exactly the first n results.

        More efficient than .execute()[:n] because it short-circuits
        after finding n matches. Equivalent to Palantir's .take(N).
        """
        ...

    def iter(self) -> Iterator[EntityNode]:
        """Lazy iterator over all matching entities.

        Fetches pages internally, yielding one entity at a time.
        Matches Palantir's .asyncIter() pattern but synchronous.
        Memory-efficient for large result sets.
        """
        ...


@dataclass
class PageResult:
    """A single page of query results with continuation token."""
    data: list[EntityNode]
    next_page_token: Optional[str]  # None if this is the last page
    total_count: Optional[int] = None  # Available if count is cheap
```

### 3.2c Materialization and Lazy Evaluation
<!-- Enriched from Palantir scrape 42 (OBOE advanced object sets) -->

A `QueryPlan` is a **query plan**, not a result set. It accumulates operations lazily. The actual data is not fetched until a **terminal operation** is called. This is a core insight from Palantir's ObjectSet design:

**Terminal operations** (trigger execution): `.execute()`, `.count()`, `.first()`, `.aggregate()`, `.facts()`, `.evidence_turns()`, `.page()`, `.take()`, `.iter()`

**Non-terminal operations** (build plan): `.type()`, `.where()`, `.filter()`, `.related()`, `.search_around()`, `.before()`, `.after()`, `.order_by()`, `.limit()`, `.union()`, `.intersection()`, `.difference()`

**Anti-pattern: Materialize-then-filter** (from Palantir/OBOE):
```python
# WRONG: Fetches ALL entities, then filters in Python
all_entities = foundry.entities().execute()
restaurants = [e for e in all_entities if "restaurant" in e.types]

# RIGHT: Filter is applied in the query plan before materialization
restaurants = foundry.entities().type("restaurant").execute()
```

**Materialization limit**: A single `execute()` call returns at most `max_entities` results (default: 100,000). Use `.page()` or `.iter()` for larger result sets. This matches Palantir's 100K materialization limit per function query.

### 3.3 Entry Points

```python
class Foundry:
    """Top-level API entry point."""

    def entities(self) -> QueryPlan:
        """Start a query over all entities."""
        ...

    def entity(self, name: str) -> Optional[EntityNode]:
        """Resolve a single entity by name (exact → alias → embedding)."""
        ...

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """BM25 keyword search over raw chunks. Returns (chunk_idx, score)."""
        ...

    def search_entities(self, query: str, top_k: int = 10) -> list[tuple[str, set[int], float]]:
        """BM25 search over entity documents. Returns (canonical, turn_set, score)."""
        ...

    def search_temporal(self, query: str, top_k: int = 5,
                        reference_date: datetime | None = None) -> dict:
        """Temporal query with Allen algebra answer computation."""
        ...
```

### 3.4 Resolution

Entity resolution is a critical path. It must handle aliases, plurals, determiners, possessives, and substring containment without false merges.

```python
class EntityResolver:
    """Resolves mentions to canonical entity names.

    Resolution cascade (checked in order):
    1. Exact normalized match in alias index
    2. Determiner-stripped match ('the restaurant' → 'restaurant')
    3. Possessive-stripped match ("user's cat" → "user cat")
    4. Type-gated substring containment (min 2 chars each, type must match)
    5. Embedding similarity (when embed_fn is set, threshold >= 0.85)
    6. Create new entity (ingestion-only)

    Query-time resolution uses resolve_exact() which stops at step 2.
    This prevents false merges like 'la' matching 'plane'.
    """

    def resolve(self, mention: str) -> Optional[str]:
        """Full resolution cascade. Used during ingestion."""
        ...

    def resolve_exact(self, mention: str) -> Optional[str]:
        """Alias-index-only resolution. Used at query time."""
        ...

    def resolve_typed(self, mention: str, hint_type: str) -> Optional[str]:
        """Type-gated resolution. Substring fallback requires type match."""
        ...

    def resolve_or_create(self, mention: str, hint_type: str | None = None) -> str:
        """Resolve or create new entity. Ingestion-only."""
        ...
```

**Invariants:**
- `resolve(x)` returns `None` only if `x` matches nothing in the graph
- `resolve_exact(x)` never performs substring containment (safe for query time)
- `resolve_or_create(x)` always returns a non-empty canonical name
- For any entity `e`, `resolve(e.canonical) == e.canonical`
- `normalize(x)` is deterministic and idempotent


### 3.5 Aggregation Framework
<!-- Enriched from Palantir scrapes 32, 42 (OSDK advanced queries, OBOE advanced object sets) -->

Palantir's ObjectSet supports server-side aggregation with groupBy chains. Hermes Foundry extends the existing `.aggregate()` terminal with multi-dimensional grouping:

```python
class QueryPlan:
    # ... existing methods ...

    def group_by(self, property_predicate: str) -> "GroupedQueryPlan":
        """Group matching entities by a property value.

        Returns a GroupedQueryPlan that can be further grouped (2D)
        or terminated with an aggregation.

        Palantir pattern:
          Objects.search().employees().groupBy(e => e.office).count()
          → Map<string, number>  e.g. {"London": 50, "New York": 75}

        Hermes equivalent:
          foundry.entities().type("restaurant").group_by("where").count()
          → {"rome": 3, "paris": 5, "tokyo": 2}
        """
        ...


class GroupedQueryPlan:
    """A query plan with one or more grouping dimensions."""

    def group_by(self, property_predicate: str) -> "GroupedQueryPlan":
        """Add a second grouping dimension (2D aggregation).

        Hermes:
          foundry.entities().type("restaurant").group_by("where").group_by("sentiment").count()
          → {"rome": {"loves it": 2, "ok": 1}, "paris": {"loves it": 4, "meh": 1}}
        """
        ...

    def count(self) -> dict:
        """Count entities in each group."""
        ...

    def sum(self, property_predicate: str) -> dict:
        """Sum numeric property across groups."""
        ...

    def avg(self, property_predicate: str) -> dict:
        """Average numeric property across groups."""
        ...

    def list(self, property_predicate: str) -> dict:
        """List property values per group."""
        ...
```

### 3.6 Search Around (pivotTo)
<!-- Enriched from Palantir scrape 32 (OSDK advanced queries, pivotTo) -->

Palantir's `pivotTo` operation traverses from a set of objects through a link type to reach connected objects of a different type. This is the same operation as Hermes Foundry's `.related()` and `.search_around()`, but the Palantir pattern clarifies an important semantic:

**pivotTo changes the result type.** After `employees.pivotTo("department")`, the result set contains Department entities, not Employee entities. The original set is consumed.

```python
class QueryPlan:
    # ... existing methods ...

    def pivot_to(self, link_predicate: str) -> "QueryPlan":
        """Traverse links and REPLACE the current entity set with the targets.

        Unlike .related() which ADDS connected entities to the set,
        pivot_to() REPLACES the set with the traversal targets.

        Palantir: employee_set.pivotTo("department") → department set
        Hermes:   foundry.entities().type("person").where("who", "$eq", "Dr. Smith")
                         .pivot_to("works_at") → clinic entities
        """
        ...
```

**Combined with filters**: The power of pivotTo is chaining filters after traversal:
```python
# Find restaurants visited by the user that are in Rome
foundry.entities().type("user").search_around("visited").type("restaurant").where("where", "$eq", "Rome").execute()
```

### 3.7 Set Operations Semantics
<!-- Enriched from Palantir scrape 32 (OSDK set operations) -->

The existing `union()`, `intersection()`, and `difference()` methods are supplemented with clarified semantics from Palantir's OSDK:

**Composition ordering**: Set operations can be combined with filters. The evaluation order is:
1. Each operand ObjectSet is resolved independently
2. Set operation is applied (union/intersect/subtract)
3. Post-set-operation filters (if any) apply to the combined result

```python
# Palantir pattern: union of two filtered sets, then subtract a third
high_priority = foundry.entities().type("task").where("priority", "$eq", "high")
urgent = foundry.entities().type("task").where("status", "$eq", "urgent")
completed = foundry.entities().type("task").where("status", "$eq", "completed")

active_important = high_priority.union(urgent).difference(completed).execute()
```

**Invariant:** Set operations produce a new QueryPlan with a fresh filter chain. Filters from the operands are preserved within their respective branches but do not cross over.


---

## 4. Ingestion Pipeline

### 4.1 Contract

The Ingestion Pipeline converts raw conversation text into structured memory. It runs at store time (not query time) and populates all indexes. The pipeline is deterministic given the same input and extraction model.

### 4.2 Pipeline Stages

```
Raw Text
  │
  ├─ Stage 1: Turn Parsing
  │   └─ Split by role (user/assistant), extract session timestamps
  │
  ├─ Stage 2: Chunking
  │   └─ 512-token chunks with role inheritance
  │
  ├─ Stage 3: Triple Extraction (LLM)
  │   └─ S-P-O triples with typed properties, role/sub_role, when
  │
  ├─ Stage 4: Faceted Entity Extraction (LLM)
  │   └─ Entity name + 9-facet values from each chunk
  │
  ├─ Stage 5: Entity Resolution
  │   └─ Alias → embedding → create new
  │
  ├─ Stage 6: Type Inference
  │   └─ 'what' facet → entity type, with depluralization
  │
  ├─ Stage 7: Temporal Extraction
  │   └─ Regex + dateparser for session time; ADELIE 3B for event time
  │
  ├─ Stage 8: Co-occurrence Tracking
  │   └─ All entity pairs within a chunk
  │
  ├─ Stage 9: Grounding Verification
  │   └─ fact.object must substring-match fact.source_text
  │
  ├─ Stage 10: Index Population
  │   └─ All 7 indexes populated from extracted data
  │
  └─ Stage 11: Derived Layer Computation
      └─ Type counts, type lists, progressions, co-occurrence gists
```

### 4.3 Stage Specifications

#### 4.3.1 Turn Parsing

```python
@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_idx: int           # Sequential index
    role: str               # "user" | "assistant"
    text: str               # Full turn text
    session_time: Optional[datetime] = None  # Parsed from "Chat Time:" header


def parse_conversation(raw_text: str) -> list[ConversationTurn]:
    """Parse raw conversation text into turns.

    Rules:
    - LME format: "Chat Time:" headers delimit sessions
    - User/assistant role detected from turn markers
    - Session time inherits: turns without Chat Time get the last parsed time
    - Text is preserved exactly (no normalization at this stage)
    """
    ...
```

#### 4.3.2 Chunking

```python
def chunk_turns(turns: list[ConversationTurn],
                chunk_size: int = 512) -> list[RawChunk]:
    """Split turns into fixed-size chunks.

    Rules:
    - Target 512 tokens per chunk (proven optimal, monotonic regression at larger)
    - Chunks respect turn boundaries when possible
    - Continuation chunks inherit role from source turn
    - Each chunk records its constituent turn indices
    - Entities extracted in later stages are back-linked to chunks
    """
    ...
```

#### 4.3.3 Triple Extraction

```python
@dataclass
class ExtractedTriple:
    """A single S-P-O triple from LLM extraction."""
    subject: str
    predicate: str
    object: str
    when: str = ""                  # Temporal expression: "since 2020", "last March"
    role: str = ""                  # event | state | opinion
    sub_role: str = ""              # acquisition | maintenance | intent | hypothetical
    source_text: str = ""           # The chunk text this was extracted from
    turn_idx: int = -1
    chunk_idx: int = -1


def extract_triples(chunk: RawChunk,
                    llm_url: str,
                    model: str = "Qwen/Qwen2.5-7B-Instruct") -> list[ExtractedTriple]:
    """Extract S-P-O triples from a chunk using LLM.

    Uses response_format={"type": "json_schema", ...} for structured output.
    NOT guided_json (incompatible with some backends).

    Role classification uses VerbNet verb-class:
    - 90 acquisition lemmas (buy, get, adopt, receive, ...)
    - 54 maintenance lemmas (keep, have, own, ...)
    - 32 intent lemmas (want, plan, hope, ...)
    - Default: state

    Entity registry cap: max 50 entries per extraction call.
    Uncapped → 28K extraction failures from prompt injection.
    """
    ...
```

#### 4.3.4 Faceted Entity Extraction

```python
@dataclass
class FacetEntity:
    """An extracted entity with its 9-facet profile."""
    name: str
    facets: dict[str, str]  # Facet.value → extracted value (empty string if not applicable)
    turn_idx: int
    source_text: str


def extract_faceted_entities(chunk: RawChunk,
                             llm_url: str,
                             model: str = "Qwen/Qwen2.5-7B-Instruct") -> list[FacetEntity]:
    """Extract entities with 9-facet profiles from a chunk.

    Best model: Qwen2.5-7B-Instruct (3% hallucination, 4.8s/chunk on vLLM A6000).
    DO NOT USE: Llama 3.1 8B (80% hallucination), LFM 1.2B (hallucinated from prompt).
    Second choice: Gemma 3 4B (18% hallucination).

    Returns entities with populated facets. Empty facets are omitted.
    """
    ...
```

#### 4.3.5 Entity Resolution

Uses `EntityResolver` (Section 3.4). During ingestion, full cascade including `resolve_or_create`. At query time, `resolve_exact` only.

#### 4.3.6 Type Inference

```python
def infer_type(entity: FacetEntity) -> Optional[str]:
    """Infer entity type from the 'what' facet.

    Returns normalized, depluralized type string.
    'Pet Fish' → 'pet_fish', 'bikes' → 'bike'
    Returns None if 'what' facet is empty.
    """
    what_val = entity.facets.get("what", "").strip()
    if not what_val:
        return None
    return EntityType.normalize(what_val)
```

#### 4.3.7 Temporal Extraction

```python
def extract_temporal(chunk: RawChunk,
                     session_time: Optional[datetime] = None) -> list[TemporalEvent]:
    """Extract temporal events from a chunk.

    Two-phase extraction:
    1. Regex + dateparser for explicit dates ("June 3rd", "last Tuesday")
       - Resolves relative dates using session_time as anchor
    2. ADELIE-SFT-3B for event-date pairs (production path)
       - Prompt: v3_simple (100% accuracy on test suite)
       - Chat template: manual Tulu format for Llama 3.2 base

    Session timestamps (Chat Time headers) always create events regardless
    of content-level extraction success.
    """
    ...
```

#### 4.3.8 Co-occurrence Tracking

```python
def track_cooccurrences(entities: list[str], chunk_idx: int,
                        cooc_index: CooccurrenceIndex) -> None:
    """Record all entity pairs from a chunk into co-occurrence index.

    CRITICAL: All entities from a SINGLE store() call must be processed together.
    Splitting chunks into separate calls breaks co-occurrence edges.
    """
    for i, a in enumerate(entities):
        for b in entities[i + 1:]:
            cooc_index.record(a, b, chunk_idx)
```

#### 4.3.9 Grounding Verification

```python
def verify_grounding(fact: Fact) -> bool:
    """Check if fact.object is grounded in fact.source_text.

    Rules:
    1. If source_text is empty, skip (return True — ungated)
    2. Case-insensitive substring check: fact.object.lower() in source_text.lower()
    3. Word-level fallback: every content word (len >= 3) in object appears in source_text
    4. Fails → fact is rejected from STATED layer (may go to INFERRED)

    This catches model inference injections:
    - "Delta" → "airline" FAILS because "airline" not in "I fly Delta"
    - The graph stores user's words, not model's interpretation
    """
    if not fact.source_text:
        return True
    src = fact.source_text.lower()
    val = fact.object.lower().strip()
    if val in src:
        return True
    return all(w in src for w in val.split() if len(w) >= 3)
```

#### 4.3.10 Role Classification (VerbNet)

```python
@dataclass(frozen=True)
class VerbClassification:
    """Result of VerbNet-based role classification."""
    role: str       # "event" | "state" | "opinion"
    sub_role: str   # "acquisition" | "maintenance" | "intent" | "hypothetical"


# Lemma sets (from NLTK VerbNet corpus + WordNet lemmatizer)
ACQUISITION_LEMMAS: frozenset[str]  # 90 lemmas: buy, get, adopt, receive, purchase, ...
MAINTENANCE_LEMMAS: frozenset[str]  # 54 lemmas: keep, have, own, maintain, ...
INTENT_LEMMAS: frozenset[str]       # 32 lemmas: want, plan, hope, wish, dream, ...


def classify_verb(predicate: str) -> VerbClassification:
    """Classify a predicate's verb into role/sub_role using VerbNet lemmas.

    3-layer defense-in-depth for counting queries:
    1. role: exclude opinions entirely
    2. sub_role: exclude intent, hypothetical
    3. date_range: temporal filtering on 'when' field

    Example: jewelry count 9 → 7 (role) → 5 (sub_role) → 4 (date) → GT=3
    """
    ...
```

#### 4.3.11 Three-Layer Routing

```python
def route_to_layer(fact: Fact, grounding_passed: bool) -> GroundingLayer:
    """Determine which layer a fact belongs to.

    Rules:
    - Grounding passed + extracted from user text → STATED
    - Grounding failed (model inference, synonym bridge) → INFERRED
    - Computed aggregate (type count, progression gist) → DERIVED
    """
    if fact.layer == GroundingLayer.DERIVED:
        return GroundingLayer.DERIVED
    if grounding_passed:
        return GroundingLayer.STATED
    return GroundingLayer.INFERRED
```

### 4.4 Derived Layer Computation

Runs after all ingestion is complete. Deterministic, no LLM.

```python
class DerivedLayerBuilder:
    """Computes derived aggregate facts from the stated layer.

    Three gist types:
    1. Type counts: user → {type}_count → "17"
       - Filtered by ownership relation (VerbNet sub_role)
       - Species-based deduplication
       - Quantity extraction (explicit property > leading number > default 1)
    2. Type lists: user → {type}_list → "neon tetras, betta, pleco catfish"
    3. Flattened 2-hop properties: user → {entity}_{prop} → value
       - Only for entities with <= 20 facts (specific things)

    Gist extraction (from stated layer):
    1. Convergent: multiple entities share same facet value
    2. Progression: entity's facet value changes over time (deltas)
    3. Co-occurrence: entity pairs sharing multiple turns
    """

    def build(self, stated_graph: "EntityGraph") -> "DerivedGraph": ...
    def extract_gists(self, stated_graph: "EntityGraph") -> dict[str, list[dict]]: ...
```

---

## 5. Retrieval Layer

### 5.1 Contract

The Retrieval Layer provides tool-based access to memory for agentic use. Tools are thin wrappers over the Query Language and Storage Layer. The architecture supports both agentic (LLM-driven tool selection) and deterministic (hardcoded pipeline) modes.

### 5.2 Tool Definitions

Eight tools available to retrieval agents:

```python
TOOLS = {
    "search_memory": {
        "signature": "(query: str) -> list[{turn_id, text, score}]",
        "description": "BM25+cosine RRF over raw chunks with optional cross-encoder rerank. "
                      "Best for finding specific words, names, or phrases. Returns top-10.",
        "implementation": "BM25Index.query() + VectorIndex.query() → RRF merge → optional rerank",
    },
    "search_entities": {
        "signature": "(query: str) -> list[{entity, facts, turns, score}]",
        "description": "BM25 over entity documents (name + facets). "
                      "Use for WHO/WHAT/WHERE. Returns top-10 with known facts.",
        "implementation": "EntityBM25.query() → EntityNode fact lookup",
    },
    "get_related": {
        "signature": "(entity: str) -> {properties, co_mentions, neighbors}",
        "description": "Given an entity, retrieve its properties, co-occurring entities, "
                      "and graph neighbors across all three layers.",
        "implementation": "EntityResolver.resolve() → EntityNode properties + related traversal",
    },
    "search_temporal": {
        "signature": "(query: str) -> list[{turn_id, text, date, score}]",
        "description": "Temporal/date-based retrieval using Allen interval algebra. "
                      "Use for WHEN questions, date ranges, temporal ordering.",
        "implementation": "TemporalIndex.match_events() → answer_temporal_query()",
    },
    "read_turn": {
        "signature": "(turn_id: int) -> {text, role, turn_id}",
        "description": "Read the full text of a specific conversation turn.",
        "implementation": "RawChunkStore.get_by_turn()",
    },
    "aggregate": {
        "signature": '(entity_type: str, op: str) -> {result, entities}',
        "description": "Count/sum/list entities of a given type. "
                      "op: count, sum, list. Use for 'how many X' questions.",
        "implementation": "TypeIndex.lookup() → EntityGraph.aggregate()",
    },
    "execute_python": {
        "signature": "(code: str) -> {result: str}",
        "description": "Sandboxed Python for date arithmetic. "
                      "Only datetime and math modules allowed. 3-second timeout.",
        "implementation": "ast.parse() → import check → exec() with SIGALRM timeout",
    },
    "answer": {
        "signature": "(evidence: str) -> TERMINAL",
        "description": "Return collected evidence. Call when sufficient information gathered.",
        "implementation": "Sets trace.evidence, terminates loop",
    },
}
```

### 5.3 Hybrid Search

The `search_memory` tool implements hybrid BM25+cosine retrieval:

```python
def search_memory_hybrid(query: str,
                         bm25: BM25Index,
                         vectors: VectorIndex,
                         reranker: Optional[Any] = None,
                         seen_turns: Optional[set[int]] = None,
                         top_k: int = 10) -> list[dict]:
    """Hybrid BM25 + cosine RRF merge with optional cross-encoder rerank.

    Pipeline:
    1. BM25 top-20 from chunk BM25 index
    2. Cosine top-20 from vector index
    3. RRF merge (k=60): score = sum(1/(k + rank_i) for each channel)
    4. Filter out previously-seen turns (dedup across calls)
    5. Optional: cross-encoder rerank top-30 → top-10
       - mxbai-rerank-large-v1 (435M, DeBERTaV3) — proven winner
       - DO NOT use Qwen3-Reranker-4B (catastrophic on conversational data)
    6. Return top-10 with text[:300] preview

    Returns: [{turn_id, text, score}, ...]
    """
    ...
```

### 5.4 Deduplication

```python
class DeduplicationTracker:
    """Tracks seen turn IDs across multiple tool calls within a query.

    Critical: without dedup, agent sees same BM25 results 5x.
    Reset at the start of each new question.
    """

    def __init__(self) -> None:
        self._seen: set[int] = set()

    def filter_new(self, results: list[dict]) -> list[dict]:
        """Remove already-seen turns and mark new ones as seen."""
        ...

    def reset(self) -> None:
        """Clear for new question."""
        self._seen.clear()
```

### 5.5 Agentic Retriever

```python
class AgenticRetriever:
    """ReAct-based iterative retrieval agent.

    Loop: Thought → Action → Observation, up to max_steps.
    Agent decides which tools to call and in what order.
    All traces saved for future pattern extraction and SLM fine-tuning.

    Configuration:
    - llm_model: "gpt-4o-mini" (proven effective for tool routing)
    - max_steps: 6-8 (2-3 steps = 50-75% hit rate)
    - temperature: 0.0 (deterministic tool selection)
    - max_tokens: 512 (per LLM call)

    Termination conditions:
    - Agent calls answer() → evidence collected
    - Budget exhausted (max_steps) → all retrieved text collected as evidence
    - LLM error → terminate with error status
    - No valid action parsed → use LLM text as evidence
    """

    def retrieve(self, question: str) -> tuple[str, RetrievalTrace]:
        """Run the agentic retrieval loop.

        Returns (evidence_text, trace).
        Trace contains all tool calls, thoughts, and timing for analysis.
        """
        ...


@dataclass
class RetrievalTrace:
    """Full trace of an agentic retrieval session.

    Saved for:
    1. Pattern extraction (Phase 2 of Traces → Deterministic → SLM pipeline)
    2. Fine-tuning data preparation (Phase 4)
    3. Debugging and evaluation
    """
    question: str
    steps: list[ToolCall] = field(default_factory=list)
    evidence: str = ""
    total_ms: float = 0.0
    terminated_by: str = ""  # "answer" | "budget" | "error" | "no_action"
```

### 5.6 Deterministic Retrieval Pipelines

For questions matching known patterns, bypass the LLM entirely:

```python
class DeterministicRouter:
    """Routes questions to hardcoded retrieval pipelines.

    Classification: regex patterns, no LLM (< 1ms).
    Pipelines hardcoded from trace analysis of common patterns.

    Fallback: if no pattern matches, delegate to AgenticRetriever.
    """

    def route(self, question: str) -> str:
        """Classify question and return pipeline name.

        Returns: pipeline name or "agentic" for fallback.
        """
        ...

    def execute(self, question: str, pipeline: str) -> tuple[str, Any]:
        """Execute a deterministic pipeline.

        Pipelines (from trace pattern analysis):
        - fact_lookup: BM25(q) → top-5 → format evidence
        - temporal: BM25(q) + TemporalIndex(q) → merge → answer
        - aggregation: TypeIndex(type) → count/list → format
        - preference: BM25(q, filter=assistant) → format
        - comparison: entity_lookup(a) + entity_lookup(b) → compare
        """
        ...
```

### 5.7 Budget Management

```python
@dataclass(frozen=True)
class RetrievalBudget:
    """Resource limits for a single retrieval operation."""
    max_steps: int = 8              # Maximum tool calls
    max_observation_chars: int = 2000  # Observation text truncation
    max_total_ms: int = 30000       # 30 second timeout
    aggregation_overfetch: int = 25 # Extra candidates for counting queries
```


### 5.8 Ontology-Augmented Generation (OAG) Patterns
<!-- Enriched from Palantir scrape 24 (ontology-augmented-generation) -->

Palantir's OAG documentation outlines retrieval strategies that apply directly to Hermes Foundry's agentic retrieval. These patterns complement the existing hybrid search:

**1. Start without search if context fits**: With 128K token context windows, if the total memory fits within the context, skip retrieval entirely and pass everything to the LLM. Hermes Foundry heuristic: if `total_chunks * avg_chunk_tokens < 100K`, consider full-context mode.

**2. HyDE (Hypothetical Document Embeddings)**: Instead of embedding the raw query, first ask an LLM to generate a hypothetical answer chunk, then embed that. This bridges the query-document asymmetry.

```python
def hyde_embed(query: str, llm_fn, embed_fn) -> "np.ndarray":
    """Generate a hypothetical answer and embed it for retrieval.

    Palantir pattern: particularly useful when chunk formats differ
    significantly from query phrasing (e.g., structured medical records
    vs. natural language questions).

    WARNING: Adds one LLM call per query (~200ms). Only use when
    standard cosine retrieval underperforms significantly.
    Cost: violates Axiom 2 (no LLM for infrastructure) — use only
    as a fallback for the agentic retriever's cosine channel, not
    for deterministic pipelines.
    """
    hypothetical = llm_fn(
        f"Given this question about a user's memory, write a paragraph "
        f"that would answer it:\n\nQuestion: {query}"
    )
    return embed_fn(hypothetical)
```

**3. Query augmentation**: Expand the query with synonyms, related terms, or entity aliases before retrieval. This is already partially implemented via the inferred layer (synonym bridges), but can be made explicit:

```python
def augment_query(query: str, entity_resolver, alias_index) -> str:
    """Expand query terms with known aliases and synonyms.

    Example: "What's Sarah's cat called?" → "What's Sarah Sarah_Smith's cat pet called?"
    The alias expansion ensures BM25 matches regardless of which surface form appears.
    """
    tokens = query.split()
    expanded = []
    for token in tokens:
        canonical = entity_resolver.resolve_exact(token.lower())
        if canonical:
            aliases = alias_index.get(canonical, set())
            expanded.append(token + " " + " ".join(aliases))
        else:
            expanded.append(token)
    return " ".join(expanded)
```

**4. Hybrid search** (already implemented in search_memory_hybrid): Palantir explicitly recommends combining keyword-based and semantic search. Hermes Foundry's RRF merge of BM25+cosine already implements this.

**5. Palantir's recommendation hierarchy** (from OAG doc, adapted):
1. Full context (if it fits) — cheapest, most reliable
2. Keyword search (BM25) — fast, interpretable
3. Semantic search (cosine) — handles paraphrase
4. Hybrid (BM25 + cosine + rerank) — best recall
5. HyDE — expensive fallback for format-asymmetric corpora
6. Agentic multi-step — most powerful, most expensive

### 5.9 Agent Tool Taxonomy
<!-- Enriched from Palantir scrapes 21, 23 (AIP agent tools, AIP logic blocks) -->

Palantir's AIP Agent Studio defines six tool types. Hermes Foundry's eight tools map to this taxonomy:

| Palantir Tool Type | Palantir Behavior | Hermes Foundry Equivalent |
|-------------------|-------------------|--------------------------|
| **Object Query** | Filter, aggregate, inspect, traverse links for configured objects | `search_entities`, `get_related`, `aggregate` |
| **Action** | Execute an ontology edit (create/modify/delete) | Not in retrieval (writeback is separate) |
| **Function** | Call arbitrary code-based logic | `execute_python` (sandboxed) |
| **Request Clarification** | Pause and ask user for more info | Not needed (questions are self-contained in benchmarks; needed for production) |
| **Ontology Semantic Search** | Vector search over object properties | `search_memory` (hybrid BM25+cosine+rerank) |
| **Command** | Trigger operations in other applications | Not applicable (single-system) |

**Tool calling modes** (from Palantir):
- **Prompted tool calling**: Tools described in the prompt, LLM outputs tool call as text. Single tool per step. This is what Hermes Foundry currently uses (ReAct format).
- **Native tool calling**: Model's built-in function-calling API. Supports parallel tool calls. Faster and more token-efficient. Hermes Foundry should migrate to native tool calling when using models that support it (gpt-4o-mini does).

**Parallel tool calls** (native mode): The agent can call `search_memory` AND `search_entities` simultaneously in a single LLM turn, rather than sequentially. This halves the latency for common patterns like "search chunks AND search entities for the same query".

### 5.10 Logic Block Composition
<!-- Enriched from Palantir scrape 23 (AIP logic blocks) -->

Palantir's AIP Logic functions are composed of **blocks** (discrete operations chained together). This maps to Hermes Foundry's deterministic retrieval pipelines:

| Palantir Block | Hermes Foundry Equivalent |
|---------------|--------------------------|
| **Use LLM** | AgenticRetriever's LLM call for tool selection |
| **Apply Action** | SupersessionEngine.supersede() (during writeback) |
| **Execute Function** | execute_python tool |
| **Loops** | AgenticRetriever's ReAct loop (max_steps iterations) |
| **Conditionals** | DeterministicRouter.route() (regex classification → pipeline) |
| **Create Variable** | RetrievalTrace accumulation (evidence, seen_turns) |
| **Calculator** | execute_python with datetime/math modules |
| **Query Objects** | QueryPlan chains (type → where → execute) |

The key insight: Palantir's Logic blocks and Hermes Foundry's deterministic pipelines serve the same purpose -- encoding known-good sequences of operations so they can run without LLM involvement. The `DeterministicRouter` is Hermes Foundry's equivalent of a Logic function with Conditionals + Query Object blocks.


---

## 6. Writeback / Mutation

### 6.1 Contract

The Writeback Layer manages how memory evolves over time. Facts can be superseded, entities merged, and temporal validity tracked. All mutations maintain referential integrity across indexes.

### 6.2 Supersession Model

```python
class SupersessionEngine:
    """Manages fact supersession: new facts replace old for same (entity, predicate).

    Strategy: "latest wins" by default.
    - Structural matching: same (subject, predicate) pair
    - NO cosine-based supersession (disproven — geometry can't distinguish conflicts)
    - Per-hop resolution for multi-hop queries (proven in FC eval)

    Superseded facts are NOT deleted — they are marked superseded=True.
    The full history is preserved for timeline() and deltas() queries.
    """

    def supersede(self, entity: str, predicate: str, new_fact: Fact) -> Optional[str]:
        """Mark existing facts with same (entity, predicate) as superseded.

        Returns the ID of the superseded fact, or None if no prior fact existed.
        """
        ...

    def resolve_conflicts(self, facts: list[Fact],
                          strategy: str = "latest_wins") -> list[Fact]:
        """Given a list of facts for the same (entity, predicate), resolve conflicts.

        Strategies:
        - 'latest_wins': highest turn_idx wins (default)
        - 'user_wins': facts from user turns override assistant turns
        - 'per_hop': resolve at each hop in a multi-hop traversal

        Returns non-superseded facts only.
        """
        ...
```

### 6.3 Temporal Validity

```python
@dataclass
class TemporalValidity:
    """Temporal validity window for a fact.

    valid_from: when the fact became true (turn index or datetime)
    valid_until: when the fact ceased to be true (None = still current)

    Facts with when='since 2020' have valid_from parsed as datetime.
    Facts superseded by newer facts get valid_until = superseding fact's turn.
    """
    valid_from: Optional[int | datetime] = None   # Turn index or datetime
    valid_until: Optional[int | datetime] = None   # None = still current

    def is_current(self, as_of: Optional[int | datetime] = None) -> bool:
        """Check if fact is current as of a given point."""
        if self.valid_until is None:
            return True
        if as_of is None:
            return self.valid_until is None
        return as_of <= self.valid_until
```

### 6.4 Entity Merge

```python
class EntityMerger:
    """Merges two entities when discovered to be the same real-world thing.

    Merge operation:
    1. Pick canonical name (longer name, or explicitly specified)
    2. Migrate all facts from source to target
    3. Migrate all links (update both endpoints)
    4. Update alias index (all source aliases → target canonical)
    5. Update co-occurrence index (replace source with target)
    6. Update all other entities' .related sets
    7. Remove source entity node

    This is the most complex mutation — must maintain referential integrity.
    """

    def merge(self, source_canonical: str, target_canonical: str) -> None:
        """Merge source entity into target.

        Preconditions:
        - Both entities exist in the graph
        - source != target

        Postconditions:
        - source entity no longer exists
        - All source facts/links now belong to target
        - All alias references updated
        - All co-occurrence references updated
        """
        ...
```

### 6.5 Deletion Model

```python
class DeletionPolicy(str, Enum):
    """How facts are removed from memory."""
    SOFT_DELETE = "soft_delete"    # Mark as deleted, keep in storage (default)
    HARD_DELETE = "hard_delete"    # Remove from all indexes permanently


@dataclass
class Tombstone:
    """Record of a deleted fact."""
    fact_id: str
    deleted_at: datetime
    reason: str                   # "superseded" | "user_request" | "decay"
```


### 6.6 Action Model
<!-- Enriched from Palantir scrapes 14, 15 (action types, function-backed actions) -->

Palantir's Ontology uses **action types** as the sole mechanism for modifying objects in OSv2 (direct edits are deprecated). This is a principled constraint: all mutations go through named, validated, auditable operations.

Hermes Foundry's equivalent is the concept of **memory actions** -- named operations that bundle related edits into a single transaction:

```python
@dataclass
class MemoryAction:
    """A named, atomic mutation operation on memory.

    Palantir pattern: An action type defines a set of changes that a user
    can take at once, including side effects. In Hermes Foundry, memory actions
    are triggered during ingestion (automatic) or by explicit API calls (manual).

    All actions are logged for audit and replay.
    """
    name: str                           # "ingest_turn", "supersede_fact", "merge_entities"
    entity_edits: list[dict]            # [{entity, predicate, old_value, new_value}, ...]
    link_edits: list[dict]              # [{source, target, predicate, action}, ...]
    timestamp: datetime
    source: str                         # "ingestion" | "user_request" | "decay_gc"

    # Side effects (Palantir: notification, validation, cascading edits)
    cascade: bool = False               # If True, supersession cascades to linked facts
    validate: bool = True               # Run grounding verification before applying


# Predefined action types (matching Palantir's pattern)
ACTION_TYPES = {
    "ingest_chunk": "Add a new chunk with extracted entities and facts",
    "supersede_fact": "Replace an existing fact with a newer value",
    "merge_entities": "Merge two entities discovered to be the same",
    "add_alias": "Register a new alias for an existing entity",
    "gc_decay": "Remove facts below salience threshold",
    "rebuild_derived": "Recompute derived layer from stated",
}
```

**Function-backed actions** (from Palantir): For complex mutations that involve reading multiple entities and computing new values, Hermes Foundry uses the same pattern -- the `EntityMerger.merge()` operation reads both entities, migrates all facts/links/aliases, and updates all references. This is a "function-backed action" that cannot be expressed as a simple property edit.

### 6.7 Edit Versioning and Conflict Detection
<!-- Enriched from Palantir scrape 13 (how edits applied) -->

Palantir's OSv2 uses offset tracking for edit ordering and version control. Hermes Foundry adapts this pattern:

```python
@dataclass
class EditVersion:
    """Version tracking for optimistic concurrency control.

    Palantir pattern: When an Action is triggered, the Actions service
    sends a modification instruction with an offset. Object reads after
    that offset are guaranteed to contain the edit.

    Hermes Foundry: Each fact mutation increments a global version counter.
    The version is stored on each Fact for conflict detection.
    """
    version: int                        # Monotonically increasing counter
    entity_canonical: str
    predicate: str
    timestamp: datetime


class VersionTracker:
    """Global version counter for edit ordering.

    Invariants:
    - version is monotonically increasing (never decremented)
    - Two concurrent edits to the same (entity, predicate) are serialized
    - Read-after-write: a query after a supersession sees the new value
    """

    def __init__(self) -> None:
        self._version: int = 0

    def next_version(self) -> int:
        """Increment and return the next version number."""
        self._version += 1
        return self._version

    @property
    def current(self) -> int:
        return self._version
```

**Key Palantir lesson: "Data already containing user edits can only be updated via additional user edits. There is no mechanism to directly undo a single user edit."** Hermes Foundry follows this -- superseded facts are never deleted, only marked. To "undo" a supersession, create a new fact that restores the previous value.

**Immediate index update + periodic flush**: Palantir applies edits to the in-memory index immediately, then periodically persists to storage (every 6 hours for OSv2). Hermes Foundry should mirror this: mutations update in-memory structures instantly, with `save()` as the explicit flush point.


---

## 7. Memory Dynamics

### 7.1 Three-Layer Architecture

```python
class MemorySystem:
    """The complete three-layer memory system.

    Layer 1 - STATED: Permanent. User's exact words, grounding-verified.
    Layer 2 - INFERRED: Disposable cache. Synonym/type bridges for navigation.
    Layer 3 - DERIVED: Recomputable. Deterministic aggregates over stated.

    Each layer is an independent EntityGraph with its own facts and links.
    Layers share entity canonical names but maintain separate fact stores.
    """

    def __init__(self) -> None:
        self.stated: EntityGraph = EntityGraph()
        self.inferred: EntityGraph = EntityGraph()
        self.derived: DerivedGraph = DerivedGraph()

    def traverse_layers(self, query_term: str) -> list[str]:
        """Resolve a query through layers: stated → inferred → stated.

        1. Direct lookup in stated layer (resolve_exact)
        2. Expand through inferred layer (synonym/type bridges)
        3. Map inferred results back to stated entities

        Returns canonical entity names from STATED layer only.
        """
        ...

    def wipe_inferred(self) -> None:
        """Clear inferred layer. Stated and derived untouched."""
        self.inferred = EntityGraph()

    def rebuild_derived(self) -> None:
        """Recompute derived layer from stated."""
        self.derived = DerivedLayerBuilder().build(self.stated)

    def build_inferred(self, synonym_dict: dict[str, list[str]]) -> None:
        """Build inferred layer from stated entities + synonym dictionary.

        No LLM calls. Uses static dictionary (or WordNet) for synonym bridges.
        Creates:
        - Type links: entity → what → domain
        - Synonym links: domain → synonym → word
        """
        ...
```

### 7.2 Importance Scoring

```python
@dataclass
class ImportanceScore:
    """Multi-signal importance score for an entity or fact."""
    mention_frequency: float    # How often mentioned (normalized by total turns)
    recency: float              # How recently mentioned (exponential decay)
    centrality: float           # Graph centrality (degree / max_degree)
    retrieval_frequency: float  # How often retrieved in queries (Hebbian signal)
    composite: float = 0.0     # Weighted combination

    @staticmethod
    def compute(entity: EntityNode, total_turns: int,
                current_turn: int, retrieval_count: int,
                max_degree: int) -> "ImportanceScore":
        """Compute importance from entity statistics.

        Weights (tunable):
        - mention_frequency: 0.3
        - recency: 0.3
        - centrality: 0.2
        - retrieval_frequency: 0.2
        """
        ...
```

### 7.3 Decay Model

```python
class DecayFunction:
    """Temporal decay of fact salience.

    Formula: salience(t) = base_salience * exp(-lambda * days_since_last_access)

    lambda = 0.01 (slow decay — conversations are persistent, not ephemeral)
    base_salience = 1.0 initially

    Access refreshes salience (Hebbian: frequently accessed → stronger).
    Superseded facts decay faster (lambda_superseded = 10 * lambda).
    """

    def compute(self, days_since_access: float,
                base_salience: float = 1.0,
                superseded: bool = False) -> float: ...

    def should_forget(self, salience: float, threshold: float = 0.01) -> bool:
        """Below threshold, fact is eligible for garbage collection."""
        ...
```

### 7.4 Parametric Bridge Concept

```python
class HebbianBridge:
    """Tracks retrieval patterns for eventual parametric memory.

    The more often a retrieval path fires, the stronger it becomes:
    1. Frequent retrievals → cache hits → fast deterministic pipeline
    2. Very frequent → candidate for model fine-tuning (parametric)
    3. Stops being retrieved → decays

    This is the bridge from searchable memory (non-parametric) to
    parametric memory (model weights).

    NOT implemented in Foundry v1 — specified here for forward compatibility.
    The trace capture system (RetrievalTrace) is the data collection mechanism.
    """

    def record_retrieval(self, question: str, evidence_turns: list[int],
                         tool_sequence: list[str]) -> None:
        """Record a successful retrieval pattern."""
        ...

    def get_frequent_patterns(self, min_count: int = 5) -> list[dict]:
        """Return patterns that have fired frequently enough for hardcoding."""
        ...
```

### 7.5 Dream Consolidation (Batch, with Safeguards)

```python
class DreamConsolidator:
    """Batch memory optimization process.

    WARNING: Batch dream consolidation is PROVEN CATASTROPHIC (SubEM=3-15).
    The embedding drift from 340 cycles of nrem_repulsion + rem_unlearn
    destroys retrieval quality.

    This module is specified for LONGITUDINAL use only — small, incremental
    consolidation over many sessions, NOT batch processing of ingested data.

    Safeguards:
    - max_cycles_per_session: 5 (not 340)
    - drift_monitor: halt if embedding centroid shifts > threshold
    - validation_set: spot-check retrieval quality after each cycle
    - rollback: restore pre-dream state if quality degrades
    """

    def consolidate(self, max_cycles: int = 5) -> dict:
        """Run bounded dream consolidation.

        Returns metrics: {cycles_run, drift_detected, quality_delta}
        Halts early if drift exceeds threshold.
        """
        ...
```

---

## 8. Question Type Coverage

### 8.1 The 13 Question Types

Each question type specifies which indexes, traversals, and aggregations are needed.

```python
QUESTION_TYPE_SPEC = {
    "single_hop_factual": {
        "pattern": r"what (is|are|was|were) .+\?",
        "example": "What color is the user's car?",
        "primary_index": "bm25_chunks",
        "secondary_indexes": ["bm25_entities"],
        "traversal": "entity_lookup → fact_lookup",
        "aggregation": None,
        "temporal": None,
        "proven_subem": "80% (FC single-hop)",
    },
    "multi_hop_bridging": {
        "pattern": r"what is the name of .+'s .+\?",
        "example": "What is the name of the user's doctor's clinic?",
        "primary_index": "bm25_chunks",
        "secondary_indexes": ["bm25_entities", "cooccurrence"],
        "traversal": "entity_lookup → related(max_hops=3) → fact_lookup",
        "aggregation": None,
        "temporal": None,
        "notes": "Conflict resolution per-hop. Co-occurrence + triadic memory help.",
    },
    "temporal": {
        "pattern": r"when did .+\?|how (many|long) .+ ago",
        "example": "When did the user adopt their cat?",
        "primary_index": "temporal_index",
        "secondary_indexes": ["bm25_chunks"],
        "traversal": "temporal_match → allen_operation",
        "aggregation": None,
        "temporal": "Allen interval algebra: ordering, distance, duration_between, duration_of, contains",
        "notes": "43 unreachable questions (27% of failures). Largest genuine capability gap.",
    },
    "counting_aggregation": {
        "pattern": r"how many .+ does .+\?",
        "example": "How many pets does the user have?",
        "primary_index": "type_index",
        "secondary_indexes": ["bm25_entities", "derived"],
        "traversal": "type_lookup → count with ownership filter",
        "aggregation": "count(type, ownership_filter=True)",
        "temporal": "VerbNet sub_role filter (exclude intent/hypothetical)",
        "notes": "Aggregation overfetch=25 helps. Species-based dedup.",
    },
    "comparison": {
        "pattern": r"which .+ is (more|less|bigger|smaller|cheaper) .+\?",
        "example": "Which pet is more expensive?",
        "primary_index": "bm25_entities",
        "secondary_indexes": ["property_index"],
        "traversal": "entity_lookup(a) + entity_lookup(b) → numeric_compare",
        "aggregation": "compare(entity_a.prop, entity_b.prop)",
        "temporal": None,
    },
    "preference": {
        "pattern": r"does the user prefer .+\?|what .+ preference",
        "example": "Does the user prefer cats or dogs?",
        "primary_index": "bm25_chunks",
        "secondary_indexes": ["bm25_entities"],
        "traversal": "bm25_search(preference keywords + entities)",
        "aggregation": None,
        "temporal": None,
        "notes": "31 rubric questions (SubEM ceiling). Needs LLM-as-judge eval.",
    },
    "opinion_sentiment": {
        "pattern": r"what does .+ think about .+\?|how does .+ feel",
        "example": "What does the user think about sushi?",
        "primary_index": "bm25_chunks",
        "secondary_indexes": ["bm25_entities"],
        "traversal": "entity_lookup → facts(role='opinion')",
        "aggregation": None,
        "temporal": None,
    },
    "spatial": {
        "pattern": r"where (is|are|was|were) .+\?",
        "example": "Where is the user's favorite restaurant?",
        "primary_index": "bm25_entities",
        "secondary_indexes": ["property_index"],
        "traversal": "entity_lookup → facts(facet='where')",
        "aggregation": None,
        "temporal": None,
    },
    "causal": {
        "pattern": r"why did .+\?|what caused .+\?",
        "example": "Why did the user switch jobs?",
        "primary_index": "bm25_chunks",
        "secondary_indexes": [],
        "traversal": "bm25_search → contextual reasoning by LLM",
        "aggregation": None,
        "temporal": "May need temporal context for causal chain",
    },
    "hypothetical": {
        "pattern": r"what would happen if .+\?|if .+ then .+\?",
        "example": "What would happen if the user got a third cat?",
        "primary_index": "bm25_chunks",
        "secondary_indexes": ["bm25_entities"],
        "traversal": "bm25_search → LLM reasoning over retrieved context",
        "aggregation": None,
        "temporal": None,
        "notes": "VerbNet sub_role='hypothetical' tags these during extraction.",
    },
    "procedural": {
        "pattern": r"how does .+ do .+\?|what is .+'s process for",
        "example": "How does the user cook pasta?",
        "primary_index": "bm25_chunks",
        "secondary_indexes": ["bm25_entities"],
        "traversal": "entity_lookup → facts(facet='how')",
        "aggregation": None,
        "temporal": None,
    },
    "social_relational": {
        "pattern": r"who is .+ to .+\?|what is .+'s relationship",
        "example": "Who is Sarah to the user?",
        "primary_index": "bm25_entities",
        "secondary_indexes": ["cooccurrence"],
        "traversal": "entity_lookup(person) → related → facts(facet='who')",
        "aggregation": None,
        "temporal": None,
    },
    "meta_cognitive": {
        "pattern": r"what does .+ know about .+\?|has .+ mentioned",
        "example": "Has the user ever mentioned their allergies?",
        "primary_index": "bm25_chunks",
        "secondary_indexes": ["bm25_entities"],
        "traversal": "bm25_search → existence check",
        "aggregation": "boolean: exists(entity, predicate)",
        "temporal": None,
    },
}
```

---

## 9. Formal Properties

### 9.1 From Lean Proofs (Phases 1-14)

The following properties have been formally verified in Lean 4:

```
1. ENERGY CONVERGENCE: Graph operations (ingestion, supersession, dream)
   decrease total system energy. Energy = sum of edge weights + entity salience.

2. DREAM CONVERGENCE: Dream consolidation converges in bounded steps.
   After K cycles, embedding centroid displacement < epsilon.
   K is bounded by: K <= (E_0 - E_min) / delta_min
   where E_0 = initial energy, E_min = minimum energy, delta_min = minimum
   energy decrease per cycle.

3. CAPACITY GATING: New ingestion respects capacity bounds. If entity count
   exceeds capacity C, lowest-salience entities are candidates for forgetting.
   The system never exceeds C + buffer entities.

4. SUPERSESSION MONOTONICITY: For any (entity, predicate) pair, the number
   of current (non-superseded) facts is monotonically bounded by 1.
   Each supersession operation decreases the active fact count for that pair.

5. GROUNDING INVARIANT: Every fact in STATED layer satisfies:
   fact.object ⊂ fact.source_text (substring containment, case-insensitive).
   This is checked at ingestion time and cannot be violated post-ingestion.
```

### 9.2 Behavioral Invariants

These are system-level invariants that all implementations must preserve:

```
I1: LAYER ISOLATION
    Writing to INFERRED never modifies STATED.
    Writing to DERIVED never modifies STATED or INFERRED.
    Deleting INFERRED preserves STATED and DERIVED.

I2: ALIAS CONSISTENCY
    For every alias a → canonical c in alias_index:
      entities[c] exists AND a in entities[c].aliases

I3: LINK SYMMETRY
    If entity A has B in A.related, then B has A in B.related.

I4: FACT ORDERING
    For any (entity, predicate), facts are ordered by turn_idx.
    The latest non-superseded fact is the "current" value.

I5: TYPE CONSISTENCY
    entities_of_type(t) returns the same set as:
    {e for e in entities if t in e.types or depluralize(t) in e.types}

I6: TEMPORAL VALIDITY
    For any two events a, b: exactly one of Allen's 13 relations holds.

I7: DEDUP IDEMPOTENCY
    Ingesting the same triple twice (same subject, predicate, object, turn_idx)
    produces exactly one fact (duplicate detection at ingestion).

I8: RESOLUTION STABILITY
    resolve(e.canonical) == e.canonical for all entities e.
    resolve_exact(x) is a subset of resolve(x) results.
```

---

## 10. API Surface

### 10.1 Constructor and Configuration

```python
@dataclass
class FoundryConfig:
    """Complete configuration for a Foundry instance."""
    # Chunking
    chunk_size: int = 512

    # Extraction
    extraction_model: str = "Qwen/Qwen2.5-7B-Instruct"
    extraction_url: Optional[str] = None  # vLLM endpoint
    temporal_model: str = "THU-KEG/ADELIE-SFT-3B"

    # Retrieval
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v1"

    # Agentic
    agent_model: str = "gpt-4o-mini"
    agent_max_steps: int = 8
    agent_temperature: float = 0.0

    # Limits
    max_entities: int = 100_000
    max_facts_per_entity: int = 1000
    entity_registry_cap: int = 50  # Per extraction call

    # Synonym dictionary for inferred layer
    synonym_dict: dict[str, list[str]] = field(default_factory=dict)


class Foundry:
    """The top-level API for Hermes Foundry.

    Usage:
        config = FoundryConfig(extraction_url="http://localhost:8000/v1")
        foundry = Foundry(config)

        # Ingest conversation
        foundry.ingest("raw conversation text...")

        # Query
        result = foundry.entities().type("restaurant").where("where", "$eq", "Rome").execute()

        # Agentic retrieval
        evidence, trace = foundry.retrieve("What restaurants has the user visited in Rome?")

        # Save/load
        foundry.save("./memory_store/")
        foundry = Foundry.load("./memory_store/", config)
    """

    def __init__(self, config: FoundryConfig) -> None: ...

    @classmethod
    def load(cls, path: str, config: FoundryConfig) -> "Foundry": ...
```

### 10.2 Ingestion Methods

```python
class Foundry:
    def ingest(self, raw_text: str) -> "IngestionReport":
        """Ingest raw conversation text through the full pipeline.

        Runs all 11 stages. Returns report with counts and any grounding failures.
        """
        ...

    def ingest_turns(self, turns: list[ConversationTurn]) -> "IngestionReport":
        """Ingest pre-parsed turns."""
        ...

    def ingest_chunk(self, chunk: RawChunk,
                     triples: list[ExtractedTriple] | None = None,
                     entities: list[FacetEntity] | None = None) -> "IngestionReport":
        """Ingest a single chunk with optional pre-extracted data."""
        ...

    def rebuild_derived(self) -> None:
        """Recompute the derived layer from stated facts."""
        ...

    def rebuild_inferred(self, synonym_dict: dict[str, list[str]] | None = None) -> None:
        """Rebuild the inferred layer from stated entities + synonym dict."""
        ...


@dataclass
class IngestionReport:
    """Summary of what was ingested."""
    chunks_processed: int
    entities_created: int
    entities_resolved: int  # Matched to existing
    facts_added: int
    facts_superseded: int
    facts_rejected: int     # Failed grounding
    temporal_events: int
    cooccurrences: int
    grounding_rate: float   # 0.0-1.0
    grounding_failures: list[str]  # Human-readable failure descriptions
    elapsed_ms: float
```

### 10.3 Query Entry Points

```python
class Foundry:
    def entities(self) -> QueryPlan:
        """Start a query over all entities. Returns a lazy QueryPlan."""
        ...

    def entity(self, name: str) -> Optional[EntityNode]:
        """Resolve and return a single entity."""
        ...

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """BM25 keyword search over raw chunks."""
        ...

    def search_entities(self, query: str, top_k: int = 10) -> list[tuple[str, set[int], float]]:
        """BM25 search over entity documents."""
        ...

    def search_temporal(self, query: str,
                        reference_date: datetime | None = None) -> dict:
        """Temporal query with Allen algebra."""
        ...

    def facts(self, entity: str, predicate: str = "",
              current_only: bool = True) -> list[Fact]:
        """Query facts about an entity."""
        ...

    def timeline(self, entity: str) -> Optional[dict[str, list[Fact]]]:
        """Full temporal history of an entity, grouped by predicate."""
        ...

    def deltas(self, entity: str) -> Optional[dict[str, list[Fact]]]:
        """Change log: only facts where value differs from prior."""
        ...
```

### 10.4 Retrieval Interface

```python
class Foundry:
    def retrieve(self, question: str) -> tuple[str, RetrievalTrace]:
        """Agentic retrieval: iterative tool-based search.

        Uses ReAct loop with gpt-4o-mini (or configured agent model).
        Returns (evidence_text, trace).
        """
        ...

    def retrieve_deterministic(self, question: str) -> tuple[str, Any]:
        """Deterministic retrieval: hardcoded pipeline, no LLM.

        Routes to best pipeline based on question classification.
        Falls back to agentic retrieval if no pattern matches.
        """
        ...
```

### 10.5 Serialization

```python
class Foundry:
    def save(self, path: str) -> None:
        """Save all state to directory. Atomic: all-or-nothing."""
        ...

    @classmethod
    def load(cls, path: str, config: FoundryConfig) -> "Foundry":
        """Load from directory. Config needed for model references."""
        ...

    def export_graph(self) -> dict:
        """Export entity graph as JSON-serializable dict."""
        ...

    @classmethod
    def import_graph(cls, data: dict) -> "EntityGraph":
        """Import entity graph from dict."""
        ...
```

### 10.6 Statistics and Introspection

```python
class Foundry:
    def stats(self) -> dict:
        """Return comprehensive statistics about the memory system.

        Returns:
        {
            "entity_count": int,
            "fact_count": int,
            "current_fact_count": int,  # Non-superseded
            "alias_count": int,
            "chunk_count": int,
            "temporal_event_count": int,
            "type_distribution": dict[str, int],  # type → count
            "avg_facts_per_entity": float,
            "avg_aliases_per_entity": float,
            "grounding_rate": float,
            "layer_sizes": {"stated": int, "inferred": int, "derived": int},
        }
        """
        ...

    def grounding_report(self) -> tuple[int, int, list[str]]:
        """Grounding verification over stated layer.

        Returns (grounded_count, total_count, ungrounded_details).
        """
        ...
```

---

## 11. Invariants and Edge Cases

### 11.1 Empty State

| Scenario | Behavior |
|----------|----------|
| Empty entity graph | `entities().execute()` returns `[]`. `search()` returns `[]`. `retrieve()` returns empty evidence. |
| Query with no BM25 matches | Returns empty list. Score = 0 for all documents. |
| Entity resolution on empty graph | `resolve()` and `resolve_exact()` return `None`. `resolve_or_create()` creates new entity. |
| Temporal query with no events | `search_temporal()` returns `{"operation": ..., "matched_events": [], "answer": "", "confidence": 0.0}` |
| Aggregation on empty type | `aggregate("nonexistent", "count")` returns `0`. `.aggregate("nonexistent", "sum")` returns `0.0`. |

### 11.2 Scale Behavior (100K+ Entities)

| Component | Behavior at Scale |
|-----------|-------------------|
| BM25 index | O(V * Q) where V = vocabulary size, Q = query terms. Proven to 3200+ chunks. |
| Entity resolution (alias index) | O(1) for exact match, O(N) for substring fallback. resolve_exact() is O(1). |
| Type index lookup | O(1) hash lookup + O(K) where K = entities of that type. |
| Property index | O(1) hash lookup + O(K) where K = entities with that property. |
| Graph traversal (BFS) | O(V + E) bounded by max_hops. max_hops=3 keeps this tractable. |
| Co-occurrence index | O(1) per pair lookup. Memory: O(P) where P = unique entity pairs. |
| Vector similarity | O(N * D) for brute-force matmul. FAISS IVF for >100K vectors. |

### 11.3 Name Collisions

| Scenario | Behavior |
|----------|----------|
| Two entities with same canonical name | Cannot happen — `resolve_or_create()` always resolves to existing entity first. |
| Same mention resolves to different entities across chunks | Entity resolution is deterministic — same mention always resolves to same entity. |
| Alias collision (two canonicals claim same alias) | First registration wins. `alias_index[alias]` is overwritten only by explicit `register_alias()`. |

### 11.4 Missing Data

| Scenario | Behavior |
|----------|----------|
| Temporal extraction fails | Chunk processed without temporal events. Other indexes still populated. |
| Embedding model unavailable | Vector index skipped. BM25 becomes sole retrieval channel (still 58.3% baseline). |
| LLM extraction unavailable | No triples/facets extracted. Raw chunks still stored and BM25-indexed. |
| Grounding check with empty source_text | Fact passes grounding (ungated). This allows manual fact insertion. |

### 11.5 Text Handling

| Aspect | Behavior |
|--------|----------|
| Unicode | All normalization preserves Unicode. Lowercasing uses `str.lower()` (locale-independent). |
| Case sensitivity | All matching is case-insensitive. Canonical names are lowercased. |
| Whitespace | Collapsed to single spaces. Leading/trailing stripped. |
| Possessives | Stripped during resolution: "user's cat" → "user cat". |
| Determiners | Stripped during resolution: "the restaurant" → "restaurant". |
| Punctuation in entities | Preserved in canonical name. Stripped during resolution matching. |

### 11.6 Concurrent Access

Foundry v1 is designed for single-process use. Concurrent access is NOT supported at the storage layer. For multi-process scenarios:

- **Read-only sharing**: Multiple readers can load the same saved state safely.
- **Write exclusion**: Only one process may ingest or mutate at a time.
- **Future**: File locking or database backend for concurrent write access.


### 11.7 Foundry is Not a Knowledge Graph (Architectural Boundary)
<!-- Enriched from Palantir scrape 25 (community discussion: true KG capabilities) -->

The Palantir developer community explicitly asked for Neo4j-like knowledge graph capabilities within Foundry. Palantir's response (and the community workaround) provides a critical architectural lesson for Hermes Foundry:

**The workaround that failed:** Users created two object types (Entity and Relationship) to emulate a knowledge graph. The Relationship object had links "from" one Entity and "to" another, with the relationship text as a property. This approach suffered from:
- Visualization tools (Vertex) showing Relationship objects as nodes, not edges
- No graph traversal primitives (no shortest path, no pattern matching)
- Semantic search worked, but graph algorithms did not

**The lesson for Hermes Foundry:** The Ontology's Entity-Property-Link model IS a graph, but it is NOT a general-purpose graph database. It is optimized for:
- Entity-centric queries (lookup by name/type, filter by property)
- 1-3 hop link traversal (Search Around)
- Property aggregation (count, sum, avg by type)

It is NOT optimized for:
- Arbitrary graph pattern matching (MATCH (a)-[r]->(b) WHERE...)
- Shortest path algorithms
- Community detection
- PageRank or other centrality measures beyond degree

**Design implication:** Hermes Foundry should NOT attempt to implement Cypher-like graph queries. The QueryPlan's `.related()` with `max_hops=3` is sufficient. For complex graph analysis, export to NetworkX/igraph as a post-processing step, not as part of the query language.

### 11.8 Concept-Centric Development
<!-- Enriched from Palantir scrape 28 (arXiv 2304.14975, concept-centric software development) -->

Palantir's internal development methodology treats **concepts** as first-class citizens. Each concept (e.g., "ObjectSet", "Action", "Link") has:
- A centralized definition in a concept repository
- Clear ownership by a team
- Documented relationships to other concepts
- Versioning and evolution history

For Hermes Foundry, this means the nine facets, the three grounding layers, and the seven index types are the **concepts** of the system. Each should have:
- A clear contract (this spec)
- A single implementation (no duplicate/competing implementations)
- Documented invariants (Section 9)
- Clear evolution path (versioned schema)


---

## 12. Success Criteria

### 12.1 Functional

1. All 13 question types produce correct answers on representative test cases.
2. Single-hop factual: >= 80% SubEM on FC single-hop (matching current SOTA).
3. Multi-hop bridging: >= 70% SubEM on FC multi-hop with conflict resolution.
4. LME overall: >= 45% SubEM (current agentic: 43.3% on context-0).
5. Temporal questions: > 0% SubEM (currently 0/26 correct on temporal arithmetic).
6. Aggregation questions: >= 30% SubEM (currently ~27% on counting questions).

### 12.2 Performance

1. Ingestion: < 5 seconds per 100 turns (excluding LLM extraction time).
2. BM25 query: < 10ms for 10K chunks.
3. Entity resolution: < 1ms per resolve_exact() call.
4. Deterministic retrieval pipeline: < 100ms end-to-end.
5. Agentic retrieval: < 10 seconds per question (current: 7.1s average).
6. Save/load: < 5 seconds for 100K entities.

### 12.3 Quality

1. Grounding rate: >= 95% of stated facts pass grounding verification.
2. Entity resolution precision: <= 5% false merges on adversarial test suite.
3. Type inference accuracy: >= 90% on faceted entity test suite.
4. Temporal extraction coverage: >= 95% of date-containing turns extracted.

---

*End of specification.*


---

## Appendix A: Adversarial Pass 1 — Mitigations

*Added 2026-04-14 by architect-agent (Opus 4.6). Brenner Level-Split + Paradox-Hunt review.*
*Each mitigation MUST be incorporated into the relevant section before implementation begins.*

### A.1 Axiom Refinements

**PH-2 Mitigation (Axiom 4):** Axiom 4 ("Always-on, never routed") bans **retrieval-channel routing** — the pattern where a classifier decides whether to use BM25, cosine, or graph for a given query. All channels always run. It does NOT ban **query-optimization routing** — the pattern where a classifier selects an optimized execution plan (e.g., DeterministicRouter choosing a hardcoded pipeline that skips the LLM agent loop but still consults all relevant indexes). The key distinction: retrieval-channel routing changes WHAT data is searched; optimization routing changes HOW the search is executed. Axiom 4 bans the former, permits the latter.

Rewrite Axiom 4 to: **"Always-on retrieval, optimized execution."** Every retrieval channel (BM25, cosine, entity BM25, temporal, co-occurrence) runs unconditionally on every query. There is no intent-based gating that disables a retrieval channel. However, deterministic pipelines MAY short-circuit execution when a known-good pattern is detected — this is optimization, not routing.

**PH-9 Mitigation (Axiom 6):** Rewrite Axiom 6 from "Index at ingestion, query at read" to: **"Compute at write-time, compose at read-time."** Indexes are populated whenever data enters the system (initial ingestion OR incremental updates). Query time only composes pre-built indexes. This aligns with the incremental ingestion design in Section 2.5 without implying a single ingestion event.

**PH-1 Mitigation (Axiom 2):** HyDE (Section 5.8) is an explicit Axiom-2 exception. It is classified as a **"gated LLM infrastructure call"** — permitted ONLY when:
1. The deterministic pipeline has already been tried and returned insufficient results (confidence < threshold)
2. The agentic retriever is active (not the deterministic path)
3. It is behind a feature flag (`config.allow_hyde = False` by default)

HyDE MUST NEVER be part of a deterministic pipeline. It is an agentic-retriever-only fallback. The spec text in Section 5.8 already says this but the gating conditions must be enforced in implementation.

**PH-3 Mitigation (Axiom 1 vs KAG cascade):** The KAG knowledge cascade (prior-art synthesis Section 2.4) is REJECTED as described. The adopted cascade for Hermes Foundry is:

```
Query → BM25 over raw chunks (PRIMARY — always runs first)
  + Entity BM25 (RRF supplement — always runs in parallel)
  + Cosine over chunks (RRF supplement — always runs in parallel)
  → RRF merge → optional rerank
  → If structured query needed: QueryPlan over EntityGraph (SECONDARY — post-retrieval enrichment, not primary)
```

BM25 is NEVER a fallback. It is always the first channel evaluated. Structured query (QueryPlan traversal) is used for enrichment AFTER BM25 retrieval, not as an alternative to it. This preserves Axiom 1 while still using the entity graph for multi-hop traversal.

### A.2 Entity Resolution Safety Boundary (LS-1)

Add to Section 3.4 after the `EntityResolver` class definition:

**Resolution Mode Contract:**

```python
class ResolutionMode(str, Enum):
    """Enforces the boundary between ingestion and query resolution."""
    INGESTION = "ingestion"   # Full cascade: steps 1-6 (alias → determiner → possessive → substring → embedding → create)
    QUERY = "query"           # Safe cascade: steps 1-3 only (alias → determiner → possessive)

# In practice:
# - resolve() = INGESTION mode (used during Foundry.ingest())
# - resolve_exact() = QUERY mode (used during QueryPlan.execute(), search_entities(), etc.)

# INVARIANT: No code path in QueryPlan execution, AgenticRetriever, or DeterministicRouter
# may call resolve() or resolve_or_create(). These are ingestion-only operations.
# Query-time code paths MUST use resolve_exact() or resolve_typed().
```

**PH-7 Mitigation:** `resolve_exact()` scope is expanded to include possessive stripping (step 3). The safe cascade at query time is: (1) exact alias lookup, (2) determiner strip + re-lookup, (3) possessive strip + re-lookup. Steps 4-6 remain ingestion-only.

### A.3 Temporal Type Separation (LS-2)

Add to Section 2.2 (Fact dataclass) and Section 6.3 (TemporalValidity):

**Temporal Field Contract:**

The three temporal fields on `Fact` represent three DISTINCT time axes:
1. `turn_idx: int` — Discourse order. Monotonically increasing. Used for supersession ("latest turn wins"). ALWAYS populated.
2. `session_time: Optional[datetime]` — System time. When the conversation happened. Populated from "Chat Time:" headers. MAY be None.
3. `timestamp: Optional[datetime]` — Event time. When the fact was true in the world. Populated from temporal extraction ("since 2020"). MAY be None.

**TemporalValidity MUST use a single type axis:**

```python
@dataclass
class TemporalValidity:
    """Temporal validity window for a fact.
    
    INVARIANT: valid_from and valid_until must be the SAME type.
    Both int (turn indices) or both datetime. Never mixed.
    The comparison axis is determined at construction time.
    """
    valid_from: Optional[int | datetime] = None
    valid_until: Optional[int | datetime] = None
    axis: str = "turn"  # "turn" (int comparison) | "event" (datetime comparison) | "session" (datetime comparison)
    
    def __post_init__(self):
        if self.valid_from is not None and self.valid_until is not None:
            assert type(self.valid_from) == type(self.valid_until), \
                f"Mixed types: {type(self.valid_from)} vs {type(self.valid_until)}"
```

**PH-5 Mitigation (Supersession ordering):** The SupersessionEngine uses `turn_idx` (discourse order) for supersession, NOT event time. This is correct: if the user says "I live in SF" at turn 3 and "Actually, I moved to Portland" at turn 7, Portland supersedes SF by discourse order regardless of when the move actually happened. Event time (`timestamp`) is used only for temporal queries ("When did you move?"), not for supersession.

### A.4 QueryPlan Result Caching (LS-3)

Add to Section 3.2c:

**Result Caching Contract:**

QueryPlans are **structurally immutable**: the filter chain, traversal list, and ordering are set at construction and never modified. However, terminal operations (`execute()`, `count()`, etc.) MAY cache their results on the plan object as an internal optimization.

```python
class QueryPlan:
    # Structural fields (set at construction, never modified)
    _filters: list       # Immutable after construction
    _traversals: list    # Immutable after construction
    _seed: Optional[set] # Immutable after construction
    
    # Result cache (populated on first terminal call)
    _result_cache: Optional[list] = None  # Mutable (internal optimization only)
    
    def execute(self) -> list[EntityNode]:
        if self._result_cache is not None:
            return self._result_cache  # Cache hit
        self._result_cache = self._execute_plan()
        return self._result_cache
```

**Invariant:** Two calls to `execute()` on the same QueryPlan object return the same result (assuming no concurrent mutations to the underlying store). The result cache does not violate structural immutability — it is a memoization, not a mutation of the query specification.

### A.5 Configuration Categories (LS-4)

Add to Section 10.1:

**Configuration categories and mutability rules:**

```python
@dataclass
class FoundryConfig:
    # --- SCHEMA CONFIG (frozen after first ingest — changing invalidates stored data) ---
    chunk_size: int = 512                    # SCHEMA: changes invalidate all chunks + BM25
    max_entities: int = 100_000              # SCHEMA: soft cap, but affects capacity gating
    max_facts_per_entity: int = 1000         # SCHEMA: affects ingestion behavior
    entity_registry_cap: int = 50            # SCHEMA: affects extraction quality
    
    # --- MODEL CONFIG (frozen per extraction session — changing invalidates extraction cache) ---
    extraction_model: str = "Qwen/Qwen2.5-7B-Instruct"  # MODEL: changes require re-extraction
    temporal_model: str = "THU-KEG/ADELIE-SFT-3B"        # MODEL: changes require re-extraction
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"  # MODEL: changes invalidate vector index
    
    # --- RUNTIME CONFIG (can change freely between sessions) ---
    extraction_url: Optional[str] = None     # RUNTIME: endpoint location
    reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v1"  # RUNTIME: affects ranking, not storage
    agent_model: str = "gpt-4o-mini"         # RUNTIME: affects retrieval quality, not storage
    agent_max_steps: int = 8                 # RUNTIME: budget control
    agent_temperature: float = 0.0           # RUNTIME: generation parameter
    synonym_dict: dict[str, list[str]] = field(default_factory=dict)  # RUNTIME: can rebuild inferred
```

**Invariant:** `save()` persists the SCHEMA and MODEL config alongside the data. On `load()`, the persisted schema/model config is compared with the provided config. Mismatches produce a warning (or error for schema config).

### A.6 Link Schema vs Instance Reconciliation (LS-5)

Add to Section 1.5:

**Schema-Level vs Instance-Level Links:**

LinkType definitions are OPTIONAL schema-level constraints. They exist to document KNOWN relationships and enforce cardinality when available. Links from triple extraction are INSTANCE-level — they always have `cardinality=MANY_TO_MANY` unless a matching schema-level LinkType is registered.

Reconciliation rule:
1. If a schema-level LinkType exists with `name == predicate`, use its cardinality and type constraints.
2. If no schema-level LinkType exists, the link is implicitly `MANY_TO_MANY` with no type constraints.
3. Schema-level LinkTypes are registered explicitly via `Foundry.register_link_type(lt)`, not auto-discovered from extraction.
4. Cardinality violations at instance level are logged as warnings, not hard errors (the schema is advisory, not enforced).

### A.7 DerivedGraph Interface (LS-6)

Add to Section 4.4:

```python
@dataclass
class DerivedGraph:
    """The DERIVED layer: deterministic aggregates computed from STATED.
    
    NOT an EntityGraph. It stores computed summaries, not entity nodes.
    Structure: keyed by (entity_canonical, gist_type) -> computed values.
    """
    type_counts: dict[str, dict[str, int]] = field(default_factory=dict)
        # entity_canonical -> {type_label: count}
    type_lists: dict[str, dict[str, list[str]]] = field(default_factory=dict)
        # entity_canonical -> {type_label: [entity_names]}
    flattened_properties: dict[str, dict[str, str]] = field(default_factory=dict)
        # entity_canonical -> {"{entity}_{prop}": value}
    gists: dict[str, list[dict]] = field(default_factory=dict)
        # entity_canonical -> [{type, content, source_entities}]
    
    def lookup(self, entity: str, gist_type: str = "") -> list[dict]:
        """Query derived data for an entity. Empty gist_type returns all."""
        ...
```

### A.8 Resolution Determinism Boundary (LS-7)

Add to Section 3.4:

**Deterministic vs Probabilistic Resolution:**

Steps 1-4 of the resolution cascade are **deterministic**: given the same alias index and entity graph, the same input always produces the same output. Step 5 (embedding similarity) is **probabilistic**: the result depends on the embedding model, which is configurable.

Invariant I8 (Resolution Stability) applies ONLY to deterministic steps. The full resolution cascade (including embedding) is NOT guaranteed to be stable across model changes. Implementation must:
1. Record which resolution step succeeded on each entity creation (e.g., `resolution_method: "alias" | "determiner" | "possessive" | "substring" | "embedding" | "created"`)
2. Flag entities created via embedding resolution as potentially unstable
3. On embedding model change, entities resolved via embedding should be re-verified

### A.9 BM25-Only Tool Access (LS-8)

Add to Section 5.2:

**Tool access to individual retrieval channels:**

The `search_memory` tool's hybrid pipeline (BM25 + cosine + rerank) is the DEFAULT for the agentic retriever. However, the deterministic pipelines (Section 5.6) have direct access to `BM25Index.query()` without RRF or reranking. This is by design: the `fact_lookup` pipeline calls BM25 directly, preserving the proven 58.3% baseline without dilution.

If future ablation shows that the agentic retriever benefits from BM25-only access, add a `search_memory_bm25` tool variant. Do NOT split the existing `search_memory` tool into separate channels — the agent should not be burdened with channel selection (that is the system's job via RRF weighting).

### A.10 Fact Identity and Dedup (LS-9)

Add to Section 2.2 and Invariant I7:

**Fact Identity:**

Two Fact objects are **content-identical** (for dedup purposes) if and only if:
```python
(fact_a.subject, fact_a.predicate, fact_a.object, fact_a.turn_idx, fact_a.chunk_idx)
== (fact_b.subject, fact_b.predicate, fact_b.object, fact_b.turn_idx, fact_b.chunk_idx)
```

`Fact.id` (uuid4) is a **reference identifier** for cross-linking (e.g., `superseded_by`). It is NOT used for dedup. Two Facts with different IDs but identical content-identity tuples are duplicates — only one should be stored.

Implementation: ingestion must check content-identity before assigning a new ID. If a matching fact already exists, the existing fact is returned (no new ID generated).

### A.11 Mixed-Type Predicate Handling (LS-10)

Add to Section 2.3.4:

**Mixed-type predicates:**

When the same predicate appears with different PropertyTypes across entities (e.g., `cost="$3200"` is NUMERIC, `cost="expensive"` is STRING), the PropertyIndex stores BOTH entries with their respective types.

Typed queries (`lookup_numeric`) only return entries matching the requested type. Untyped queries (`lookup`) return all entries. This is correct behavior, not a bug — the PropertyIndex does not enforce type uniformity per predicate.

**Warning:** `aggregate("cost", "avg")` will silently skip non-numeric values. The implementation MUST report the skip count: `aggregate() -> (result, skipped_count)` or include it in the trace.

### A.12 Short-Value Grounding Rule (LS-11)

Add to Section 4.3.9 after the `verify_grounding` function:

**Short-value grounding exception:**

When `fact.object` (after stripping) has length <= 3 AND the word-level fallback would skip all tokens (because all are < 3 chars), apply an EXACT substring check instead:

```python
def verify_grounding(fact: Fact) -> bool:
    if not fact.source_text:
        return True
    src = fact.source_text.lower()
    val = fact.object.lower().strip()
    if val in src:
        return True
    # Word-level fallback
    content_words = [w for w in val.split() if len(w) >= 3]
    if content_words:
        return all(w in src for w in content_words)
    # SHORT-VALUE RULE: if ALL words are < 3 chars (e.g., "LA", "UK"),
    # require exact substring match (already checked above and failed)
    return False  # No content words and no exact match = grounding failure
```

This prevents short values like "LA" from passing grounding trivially.

### A.13 Persistence Atomicity and Schema Versioning (LS-12, PH-6)

Add to Section 2.4:

**Atomicity mechanism:**

`save(path)` MUST use write-to-temporary-directory + atomic-rename:
1. Create `path.tmp/` directory
2. Write all index files to `path.tmp/`
3. If all writes succeed, `os.rename(path.tmp, path)` (atomic on POSIX)
4. If any write fails, delete `path.tmp/` and raise

`load(path)` validates all expected files exist before reading any.

**Schema versioning:**

Every saved store includes a `manifest.json` with:
```json
{
    "schema_version": 1,
    "created_at": "2026-04-14T00:00:00Z",
    "foundry_version": "1.0.0",
    "config_snapshot": {
        "chunk_size": 512,
        "extraction_model": "Qwen/Qwen2.5-7B-Instruct",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2"
    },
    "index_files": ["bm25_chunks.json", "entities.json", ...]
}
```

On `load()`, if `schema_version` does not match the current code's expected version, raise `SchemaVersionError` with migration instructions.

### A.14 Allen Boundary Semantics (LS-13)

Add to Section 2.3.5:

**Boundary semantics:**

Allen interval relations use **strict inequalities** for gap-based relations and **equality** for touching relations, matching the original Allen 1983 definition:
- `before(A, B)`: `A.end_dt < B.start_dt` (strict — there is a gap)
- `meets(A, B)`: `A.end_dt == B.start_dt` (touching — no gap, no overlap)
- `overlaps(A, B)`: `A.start_dt < B.start_dt < A.end_dt < B.end_dt` (strict containment of B's start)

The convenience method `events_between(start, end)` uses **inclusive** comparison: `event.start_dt <= end AND event.end_dt >= start`. This is NOT an Allen relation — it is a range-overlap query. The distinction must be documented in the implementation.

### A.15 Traversal Semantic Clarification (LS-14)

Add to Section 3.2:

**Traversal method semantics:**

| Method | Semantics | Result Set | Use Case |
|--------|-----------|------------|----------|
| `.related()` | EXPAND: current set + connected entities | Union of current + neighbors | "Find cats AND anything related to cats" |
| `.pivot_to()` | REPLACE: discard current set, keep only targets | Only traversal targets | "Find the CLINICS where doctors work" (not the doctors themselves) |
| `.search_around()` | Alias for `.pivot_to()` with optional type filter | Only traversal targets of matching type | Palantir-compatible Search Around |

`.search_around(predicate, target_type)` is equivalent to `.pivot_to(predicate).type(target_type)`.

### A.16 Inferred Layer Safety (PH-4)

Add to Section 1.6:

**Inferred layer safety constraints:**

The Inferred layer is model interpretation. It IS semantic bridging, not user's words. The safety contract:
1. Inferred-layer traversal results are ALWAYS filtered against the stated layer before inclusion in query results. A query that traverses "feline" -> "cat" via the inferred layer returns only STATED entities that have type "cat" — the word "feline" never appears in any result or prompt.
2. The synonym dictionary is STATIC and AUDITABLE. It is provided in `FoundryConfig.synonym_dict`, not generated by an LLM. Bad entries (e.g., "jaguar" -> "cat") can be identified and removed by inspection.
3. Inferred-layer entries carry a `bridge_type` annotation: `"synonym"`, `"hypernym"`, `"hyponym"`, `"wordnet"`. This enables filtering by bridge quality.
4. An entity reached ONLY via the inferred layer (with no direct stated-layer presence) is returned with a `resolution_confidence` flag so the retrieval layer can deprioritize uncertain matches.

### A.17 VerbNet Classification Uncertainty (PH-8)

Add to Section 4.3.10:

**Classification uncertainty:**

VerbNet role classification is best-effort at extraction time. The `sub_role` field on Fact is an extraction-time annotation, not ground truth. Query-time aggregation (e.g., "how many pets") SHOULD filter by `sub_role != "intent"` and `sub_role != "hypothetical"`, but MUST treat the classification as potentially wrong.

Implementation guidance:
- If a counting query returns 0 results after sub_role filtering, fall back to unfiltered count with a note "includes unverified ownership"
- The VerbNet lemma sets should be versioned and auditable
- Future improvement: use the user's explicit corrections ("actually I do have a parrot") to retroactively update sub_role

### A.18 max_hops Hard Clamp (PH-10)

Add to Section 3.2:

**max_hops enforcement:**

```python
MAX_HOPS_HARD_LIMIT = 3  # Proven diminishing returns beyond this

class QueryPlan:
    def related(self, link_predicate: str = "",
                direction: str = "both",
                max_hops: int = 1) -> "QueryPlan":
        clamped = min(max_hops, MAX_HOPS_HARD_LIMIT)
        # ... create TraversalOp with clamped value
```

Values > 3 are silently clamped. This is a HARD limit, not advisory. The test suite MUST include a test that `related(max_hops=100)` behaves identically to `related(max_hops=3)`.

### A.19 Pipeline Determinism Scope (PH-11)

Add to Section 4.1:

**Determinism scope:**

The ingestion pipeline is deterministic given the same input AND the same configuration (all models fixed). Specifically, reproducing identical output requires:
1. Same raw text input
2. Same `extraction_model` (affects triple and facet extraction)
3. Same `temporal_model` (affects temporal event extraction)
4. Same `embedding_model` (affects vector index contents)
5. Same `chunk_size` (affects chunking boundaries)
6. Same `synonym_dict` (affects inferred layer)

Changing ANY of these between runs produces different stored state. The `save()` manifest (A.13) records all model identifiers for reproducibility.

---

*End of Adversarial Pass 1 mitigations.*


---

## Appendix B: Adversarial Pass 2 — Scale-Check + Materialize Mitigations

*Added 2026-04-14 by architect-agent (Opus 4.6). Brenner Scale-Check (⊞) + Materialize (⌂) review.*
*Each mitigation MUST be incorporated into the relevant section before implementation begins.*
*Full findings with calculations in `foundry/adversarial-pass2.md`.*

### B.1 Co-occurrence Index Memory Budget (SC-1, FATAL)

Add to Section 2.3.6 (CooccurrenceIndex):

**Memory-bounded co-occurrence:**

The CooccurrenceIndex stores entity pair co-mentions. At scale, the number of unique pairs grows as O(E^2) in the worst case. To prevent memory exhaustion:

```python
class CooccurrenceIndex:
    """Entity co-mention tracking within chunks.
    
    MEMORY CONTRACT: Storage is O(E * K) where E = entity count, K = max_neighbors.
    Internally stored as adjacency list, NOT pair set.
    When an entity exceeds max_neighbors, the lowest-count neighbor is evicted.
    """
    MAX_NEIGHBORS_PER_ENTITY: int = 100  # Configurable via FoundryConfig
    
    def record(self, entity_a: str, entity_b: str, chunk_idx: int) -> None:
        """Record co-occurrence. Evicts lowest-count neighbor if budget exceeded."""
        ...
```

Add to FoundryConfig (SCHEMA category):
```python
max_cooccurrence_neighbors: int = 100  # Max co-occurring entities tracked per entity
```

**Calculated memory at 100K entities:** 100K * 100 neighbors * ~40 bytes per entry = ~400 MB (feasible).
**Without cap at 100K entities:** up to 5B pairs = physically impossible.

### B.2 Approximate Nearest Neighbor for Entity Resolution (SC-2, HIGH)

Add to Section 3.4 (EntityResolver):

**Embedding resolution MUST use approximate nearest neighbor search:**

Step 5 (embedding similarity) is called O(M) times during ingestion where M = unresolved mentions. Linear scan over all entity embeddings is O(E) per call, producing O(M*E) total comparisons. At 10K+ entities, this exceeds acceptable ingestion time.

```python
class EntityResolver:
    """...
    
    Step 5 (embedding) implementation:
    - For E < 5000: brute-force cosine scan is acceptable (< 5ms per call)
    - For E >= 5000: MUST use HNSW index (e.g., FAISS IndexHNSWFlat)
      with ef_construction=200, ef_search=50
    - ANN recall > 99% at threshold=0.85 with these parameters
    """
```

### B.3 BM25 Complexity Correction and Inverted Index Threshold (SC-3, HIGH)

Correct Section 11.2:

Replace:
> BM25 index: O(V * Q) where V = vocabulary size, Q = query terms. Proven to 3200+ chunks.

With:
> BM25 index: O(Q * D) where Q = query term count, D = document count (linear scan implementation). At D > 50K, MUST use inverted posting lists reducing average complexity to O(Q * D/V). Proven to 3200+ chunks; 10ms budget (Section 12.2) allows up to 100K chunks with inverted index.

### B.4 Agentic Retrieval Token Budget and Context Window (SC-4, HIGH)

Add to Section 5.7 (RetrievalBudget):

```python
@dataclass(frozen=True)
class RetrievalBudget:
    max_steps: int = 8
    max_observation_chars: int = 2000
    max_total_ms: int = 30000
    aggregation_overfetch: int = 25
    max_context_tokens: int = 8192     # NEW: hard cap on accumulated context
    sliding_window_after: int = 4      # NEW: compress earlier observations after step 4
```

**Token accumulation model:** Base (system+question) = ~830 tokens. Per step: ~600 tokens. At step 8: ~5,630 tokens (within 8K budget). Context window management: after step `sliding_window_after`, earlier observations are replaced with a 200-token summary.

**Cost model at gpt-4o-mini rates:** ~$0.003 per question (8 steps). At 10K questions/day: ~$33/day.

### B.5 Total Fact Memory Budget (SC-5, HIGH)

Add to FoundryConfig (SCHEMA category):

```python
max_total_facts: int = 2_000_000  # SCHEMA: hard cap on total facts in memory (~1.5 GB)
```

**Memory model:** Each Fact object ≈ 770 bytes (Python object overhead + 16 fields). Memory = E * F_avg * 770 bytes.

| Scale | Entities | Avg Facts | Total Facts | Memory |
|-------|----------|-----------|-------------|--------|
| LME | 1,000 | 10 | 10K | 7.7 MB |
| Medium | 10,000 | 20 | 200K | 154 MB |
| Large | 100,000 | 20 | 2M | 1.5 GB |
| Max config (infeasible) | 100,000 | 1,000 | 100M | 77 GB |

When `max_total_facts` is reached, superseded facts older than 90 days are evicted from memory (retained on disk for timeline queries via lazy loading).

### B.6 VectorIndex FAISS Transition Specification (SC-6, HIGH)

Add to Section 2.3.2 (VectorIndex):

**Implementation transition points:**

| Vector Count | Implementation | Query Latency (768d) | Memory |
|-------------|----------------|---------------------|---------|
| < 50K | numpy matmul (brute-force) | < 8ms | N * 3 KB |
| 50K - 500K | FAISS IndexIVFFlat(nlist=sqrt(N), nprobe=16) | < 5ms | N * 3 KB + overhead |
| > 500K | FAISS IndexIVFPQ(nlist=sqrt(N), m=48, nbits=8) | < 3ms | N * 48 bytes (compressed) |

Add to FoundryConfig (SCHEMA category):
```python
max_vectors: int = 500_000  # SCHEMA: cap on stored vectors (500K * 3KB = 1.5 GB at 768d)
```

Dimension reduction option: Matryoshka embeddings at 256d reduce storage by 3x. Configure via `embedding_dimensions: int = 768`.

### B.7 Hub Entity Traversal Budget (SC-8, MEDIUM)

Add to Section 3.2 (TraversalOp):

```python
MAX_TRAVERSAL_NODES: int = 1000  # Hard cap on BFS node visits

class QueryPlan:
    def related(self, ...):
        # BFS stops when visited_count >= MAX_TRAVERSAL_NODES
        # Hub entities (degree > 100) traverse only top-K edges by co-occurrence count
        ...
```

**Branching factor calculation:** At avg degree d=20, depth 3 visits d^3 = 8000 nodes. At hub entity (d=500), depth 3 visits 125M nodes without cap. The `MAX_TRAVERSAL_NODES=1000` cap ensures bounded execution. Hub entities (the "user" entity) use predicate-filtered traversal: `.related("visited")` only follows "visited" edges, not all 500 connections.

### B.8 Full-Context Threshold Revision (SC-9, MEDIUM)

Replace Section 5.8 pattern 1:

> if `total_chunks * avg_chunk_tokens < 100K`, consider full-context mode.

With:

> if `total_chunks * avg_chunk_tokens < 20_000`, use full-context mode. At 512 tokens/chunk, this is ~40 chunks (~3 conversations). Full-context mode is a **cold-start optimization** for new users. After ~3 conversations, retrieval-based mode activates. Configurable via `full_context_threshold_tokens: int = 20_000`.

### B.9 Binary Serialization for Save/Load (SC-10, MEDIUM)

Add to Section 2.4 (Persistence Model):

**Serialization format:**

JSON serialization at 100K entities produces ~1.6 GB and takes ~32 seconds (exceeds 5-second budget by 6x). Required format:

| Index | Format | Rationale |
|-------|--------|-----------|
| EntityGraph | MessagePack | 5-10x faster than JSON, ~40% smaller |
| BM25 TF matrices | numpy .npz | Binary, O(1) per-element access |
| VectorIndex | numpy .npy (embeddings) + FAISS index file | Native binary |
| TemporalIndex | MessagePack | datetime handling |
| manifest.json | JSON | Human-readable metadata |

Revised performance budget:

| Scale | Entities | Save Time | Load Time |
|-------|----------|-----------|-----------|
| LME | 1,000 | < 1s | < 1s |
| Medium | 10,000 | < 3s | < 3s |
| Large | 100,000 | < 15s | < 20s |

### B.10 QueryPlan Execution Strategy (MT-1, FATAL)

Add Section 3.2d: **Query Execution Strategy**

The QueryPlan executor uses **index-first evaluation**: specialized indexes are consulted BEFORE materializing EntityNodes.

**Filter pushdown rules:**
1. `.type(t)` → pushes to `TypeIndex.lookup(t)` → returns `set[str]` of canonicals
2. `.where(pred, op, val)` → pushes to `PropertyIndex.lookup(pred)` filtered by op/val → returns `set[str]`
3. `.before(date)` / `.after(date)` / `.during(start, end)` → pushes to `TemporalIndex.events_between()` → extracts entity canonicals from events → returns `set[str]`
4. All pushdown results are INTERSECTED to form a candidate set
5. EntityNodes are loaded ONLY for entities in the candidate set
6. Filters that cannot be pushed (e.g., complex LogicalOp, custom predicates) are applied post-materialization on the candidate set

**Fallback:** If no index-pushable filter exists (e.g., `foundry.entities().execute()`), load all entities up to `max_entities` limit.

**Invariant:** For any query with at least one pushable filter, execution never loads more entities than the pushdown candidate set size.

### B.11 Entity Resolution Dead-Zone Mitigation (MT-2, HIGH)

Add to Section 3.4 after step 4:

**Step 4b: Jaccard token overlap (ingestion-only)**

Between substring containment (step 4) and embedding similarity (step 5), add:

```python
def _resolve_by_token_overlap(self, mention: str, hint_type: str = "") -> Optional[str]:
    """Token-level Jaccard similarity with optional type gating.
    
    Threshold: >= 0.4 Jaccard AND type match (if hint_type provided).
    Example: "new sushi place" vs "sushi_palace":
      tokens_a = {"new", "sushi", "place"}
      tokens_b = {"sushi", "palace"}
      jaccard = |{sushi}| / |{new, sushi, place, palace}| = 1/4 = 0.25
      → Below threshold, no match (correct — these could be different places)
    
    Example: "dr smith" vs "doctor_smith":
      tokens_a = {"dr", "smith"}
      tokens_b = {"doctor", "smith"}
      jaccard = |{smith}| / |{dr, smith, doctor}| = 1/3 = 0.33
      → Below threshold without synonym expansion
      → WITH alias expansion: "dr" → "doctor" → jaccard = 2/2 = 1.0 → MATCH ✓
    """
    ...
```

Also: lower embedding similarity threshold from 0.85 to 0.80 for conversational text (validated: false positive rate < 2% at 0.80 on LME entities).

### B.12 Supersession Limitation Documentation (MT-5, HIGH)

Add to Section 6.2 after the SupersessionEngine class:

**Known limitation: structural supersession only.**

The SupersessionEngine detects updates where the SAME predicate receives a new value (e.g., `allergic_to: "peanuts"` superseded by `allergic_to: "no longer allergic"`). It does NOT detect semantic contradiction across DIFFERENT predicates (e.g., `allergic_to: "peanuts"` contradicted by `ate: "peanut butter"`).

**Predicate supersession groups** (for safety-critical domains):

```python
SUPERSESSION_GROUPS: dict[str, set[str]] = {
    "allergy": {"allergic_to", "can_eat", "tolerates", "outgrew_allergy", "allergy_status"},
    "location": {"lives_in", "moved_to", "relocated_to", "based_in", "residence"},
    "employment": {"works_at", "employed_by", "job", "quit", "fired_from", "retired_from"},
}
```

When a fact with predicate P is ingested, and P belongs to a supersession group G, ALL existing non-superseded facts with predicates in G are candidates for supersession (latest turn wins within the group).

### B.13 Temporal Query Compound Bucket (MT-4, HIGH)

Add to Section 2.3.5 (TemporalIndex question classification):

**6th question bucket: temporal_filter**

```python
"temporal_filter": {
    "pattern": r"what .+ (in|during|last|this) (january|february|march|...|december|\d{4}|week|month|year)",
    "example": "What restaurants did I visit in March?",
    "operation": "events_between + entity_type_filter + verb_filter",
    "requires": ["TemporalIndex", "TypeIndex"],
}
```

The search_temporal tool signature expands to:
```python
"search_temporal": {
    "signature": "(query: str, entity_type?: str, verb?: str) -> list[{turn_id, text, date, score}]",
}
```

**Date-range resolution for bare month names:** "March" resolves to `[2026-03-01, 2026-03-31]` (most recent completed March). "Last March" resolves to the March before the most recent one if the current month is after March.

### B.14 Ingestion Pipeline LLM Call Consolidation (MT-6, SC-12, HIGH)

Add to Section 4.2:

**LLM call consolidation:**

Stages 3 (Triple Extraction) and 4 (Faceted Entity Extraction) MAY be combined into a single LLM call using a unified prompt that extracts both triples and faceted entities. This halves LLM latency per chunk.

**Normalization boundary:** Entity names extracted by the LLM are in raw form. Normalization to canonical form happens at Stage 5 (Entity Resolution). `Fact.subject` stores the NORMALIZED canonical name after resolution.

**Compound type hierarchy:** When the `what` facet produces a compound type (e.g., "beagle puppy"), the TypeIndex stores BOTH the compound form AND parent types derived from a static hypernym table:

```python
TYPE_HYPERNYMS = {
    "puppy": ["dog", "pet", "animal"],
    "kitten": ["cat", "pet", "animal"],
    "sedan": ["car", "vehicle"],
    # ... extensible via FoundryConfig
}
```

`TypeIndex.lookup("dog")` returns entities typed "puppy", "beagle_puppy", etc. through hypernym matching.

**Honest ingestion budget:** 100 turns ≈ 8-10 chunks. Non-LLM processing: < 5 seconds. LLM extraction: ~85 seconds (sequential on vLLM A6000), ~15 seconds (8x parallel), ~5 seconds (H200 batch). Lazy ingestion mode: BM25 available in < 5 seconds, LLM extraction runs asynchronously.

### B.15 Grounding Verification Numeric Equivalence (MT-9, MEDIUM)

Add to Section 4.3.9 after the verify_grounding function:

**Numeric equivalence exception:**

When `fact.object_type == NUMERIC` and the substring check fails, attempt numeric equivalence:

```python
def _check_numeric_equivalence(fact_value: str, source_text: str) -> bool:
    """Check if the fact value is numerically equal to a number in source_text.
    
    Example: fact_value="$350", source_text="three hundred and fifty dollars"
    → parse_numeric("$350") = 350.0
    → find numbers in source: word_to_num("three hundred and fifty") = 350.0
    → 350.0 == 350.0 → True
    """
    ...
```

This requires a word-to-number parser (e.g., `word2number` library). Abbreviated forms ("$" → "dollar", "k" → "thousand") are handled by the numeric parser, not the grounding check.

### B.16 Entity Merge Surface Area Expansion (MT-12, MEDIUM)

Expand Section 6.4 (EntityMerger) to enumerate ALL data structures requiring update:

```
Entity merge: source_canonical → target_canonical

Data structures requiring update (13 total):
 1. EntityGraph.entities: delete source, merge facts/links into target
 2. EntityGraph.alias_index: all source aliases → target canonical
 3. EntityNode.related: all entities referencing source update to target
 4. EntityNode.facts: all facts with subject=source change to target
 5. CooccurrenceIndex: replace source with target in all pairs
 6. BM25 (chunks): no change needed (chunks reference chunk_idx not entities)
 7. BM25 (entities): remove source doc, rebuild target doc with merged data
 8. VectorIndex (chunks): no change needed
 9. VectorIndex (entities): remove source embedding, optionally re-embed target
10. TypeIndex: remove source entries, add to target entries
11. PropertyIndex: update entity field from source to target
12. TemporalIndex: update event.subject from source to target
13. RawChunkStore: update chunk.entities lists that reference source
```

BM25Index and VectorIndex MUST support removal:
```python
class BM25Index:
    def remove(self, doc_id: int) -> None: ...
    def update(self, doc_id: int, text: str) -> None: ...

class VectorIndex:
    def remove(self, doc_id: int) -> None: ...
```

Alternative: lazy migration via alias redirect. Query-time resolution of "dr_smith" returns "doctor_smith". Index cleanup deferred to next `save()`.

### B.17 Decay Model Dual-Axis Specification (MT-8, HIGH)

Add to Section 7.3 (DecayFunction):

**Dual decay model:**

```python
class DecayFunction:
    """Temporal decay of fact salience.
    
    Two decay axes, using the MORE aggressive of the two:
    
    1. Wall-clock decay: salience_days = base * exp(-lambda_days * days_since_access)
       lambda_days = 0.01 (half-life ≈ 69 days)
    
    2. Turn-based decay: salience_turns = base * exp(-lambda_turns * turns_since_access)
       lambda_turns = 0.001 (half-life ≈ 693 turns)
    
    Composite: salience = min(salience_days, salience_turns)
    
    Fallback: when session_time is unavailable, use turns_since_access / 20 as days estimate
    (assuming 20 turns/day as default).
    """
```

**Reference decay table:**

| Days Since Access | Turns Since Access | Salience (non-superseded) | Salience (superseded, 10x lambda) |
|-------------------|--------------------|--------------------------|-----------------------------------|
| 0 | 0 | 1.000 | 1.000 |
| 7 | 140 | 0.932 | 0.497 |
| 30 | 600 | 0.741 | 0.050 |
| 69 | 1386 | 0.500 | 0.001 |
| 200 | 4000 | 0.135 | ~0 |
| 365 | 7300 | 0.026 | ~0 |
| 461 (threshold) | 9210 | 0.010 | ~0 |

Entities with salience < 0.01 are eligible for GC. Non-superseded entities reach this threshold after ~461 days without access (~9210 turns).

---

*End of Adversarial Pass 2 mitigations.*

---

## Appendix C: Adversarial Pass 3 — Exclusion-Test + Transpose Mitigations

*Added 2026-04-14 from adversarial-pass3.md (36 findings: 10 Exclusion-Test, 8 Object-Transpose, 10 2nd/3rd-Order Effects, 8 Dynamical Systems).*

### C.1 Query Plan Operation Reordering (ET-1, MEDIUM)

Add to Section 3.2d (QueryPlan execution):

The query executor MUST reorder operations to push all index-eligible filters before any traversal steps, regardless of the order they appear in the user's chain. This is a query optimization pass, not a semantic change — the final result set is identical under either ordering.

**Rationale:** Lazy evaluation is worse than eager when a traversal step precedes a selective filter. `.related().type("restaurant")` without reordering performs BFS on the entire entity graph before type-filtering. With reordering, `TypeIndex.lookup("restaurant")` runs first, producing a small candidate set for traversal.

```python
class QueryOptimizer:
    """Reorders QueryPlan steps for optimal execution.
    
    Invariant: reorder(plan).execute() == plan.execute() for all plans.
    
    Priority ordering:
    1. .type() filters  -> TypeIndex lookup (O(1))
    2. .where() filters -> PropertyIndex lookup (O(log N))
    3. .before()/.after() -> TemporalIndex range scan (O(log N))
    4. .related()/.pivot_to() -> Graph BFS (O(degree * hops))
    5. .aggregate() -> Computed last, on smallest possible set
    """
```

**Test case (from ET-1):** Construct `.related("visited").type("restaurant").where("status", "$ne", "closed")` and the reordered equivalent `.type("restaurant").where("status", "$ne", "closed").related("visited")`. Both must return identical results. The reordered version must complete in < 10ms on 100K entities; the original ordering may take > 1 second.

### C.2 Dual-Path Retrieval Axiom (ET-2, FATAL)

Rewrite Axiom 1 in Section 0.2 (Axioms):

**OLD:** "BM25 is primary. Everything else adds on top."

**NEW:** "BM25 is the primary RETRIEVAL channel. For the 47% of questions answerable by retrieval, BM25 is the strongest single signal. For the 53% requiring computation (temporal arithmetic, aggregation, multi-hop inference), dedicated engines are PRIMARY and BM25 provides supporting context."

Add to Section 5.1 (Retrieval Overview):

The retrieval cascade operates in two modes, determined by the DeterministicRouter:

1. **Retrieval-primary** (lexical/factual questions): BM25 is the decisive channel. Cosine, entity graph, and reranker augment but do not override BM25 rankings.
2. **Engine-primary** (temporal, aggregation, multi-hop questions): Dedicated engines (TemporalEngine, AggregationEngine, GraphTraversal) are the decisive channels. BM25 provides supporting context for the LLM's answer generation, but does not determine the answer.

BM25 always runs on every query. It is not always the decisive channel.

**Empirical basis:** BM25-only = 58.3% SubEM of REACHABLE questions = 27.4% SubEM overall. Agentic (multi-channel) = 43.3% SubEM overall. The 16pp gap comes entirely from the 53% that BM25 alone cannot reach.

### C.3 Facets Are Descriptive, Not Definitive (ET-3, HIGH)

Add to Section 1.2 (Universal Facets):

The 9 facets (WHAT, WHO, WHEN, WHERE, QUANTITY, FREQUENCY, HOW, SENTIMENT, STATUS) are DESCRIPTIVE, not DEFINITIVE. An entity with ambiguous facet mapping (e.g., abstract concepts like "fear," relationships like "friendship," or quantified attributes like "running 50 hours/week") is still stored and queryable via BM25 and raw chunk retrieval.

Facet-based queries (TypeIndex, PropertyIndex) may miss or miscategorize these entities. The system degrades to BM25 for entities with ambiguous facet profiles — this is the designed fallback, not a bug.

**Ambiguity rate:** Approximately 20% of extracted entities have ambiguous type mappings (from Qwen2.5-7B evaluation). These entities are valid members of the entity graph but should not be relied upon for facet-only queries.

**Entity vs Link disambiguation:** When the extracted entity is actually a relationship (e.g., "friendship with Sarah"), the extractor SHOULD create a link (fact) on the source entity rather than a standalone entity. The heuristic: if the extracted entity contains a preposition phrase referencing another entity, treat it as a link on the first entity.

### C.4 Homonym Detection and Entity Split (ET-4, HIGH)

Add to Section 3.4 after the entity resolution cascade:

**Homonym detection and entity split:**

When an existing entity receives a NEW type that contradicts its current primary type (e.g., a "country" entity receives "person" type), this triggers a **type conflict check**:

1. If the new type and existing type share no hypernym path (via TYPE_HYPERNYMS), flag as potential homonym.
2. Create a SEPARATE entity with a disambiguated canonical name: `jordan_person` vs `jordan_country`.
3. Retroactively assign facts to the correct entity based on surrounding context (co-occurring entities in the same chunk provide disambiguation signal).
4. If disambiguation is ambiguous, keep both entities and let query-time context disambiguate.

Add to Section 6: **EntitySplitter** (inverse of EntityMerger):

```python
class EntitySplitter:
    """Splits a merged entity into two or more entities.
    
    Triggered by: type conflict detection during ingestion.
    
    Process:
    1. Identify fact clusters by co-occurrence context
    2. Assign each fact to the most likely entity
    3. Create new entities with disambiguated canonical names
    4. Update all indexes (BM25, type, property, co-occurrence, temporal)
    5. Redirect alias table entries to the appropriate new entity
    
    Invariant: sum(facts across split entities) == facts of original entity.
    No facts are lost or duplicated during split.
    """
```

**False merge recovery:** Once two entities are merged, their facts are interleaved, co-occurrences combined, and types unioned. Without EntitySplitter, false merges are UNRECOVERABLE. The spec MUST define this operation for correctness.

### C.5 Supersession Chain Eviction Policy (ET-5, MEDIUM)

Add to Section 6.2 (SupersessionEngine):

When `max_facts_per_entity` is reached and a new supersession would exceed the cap, the OLDEST superseded fact in the longest chain is evicted (hard-deleted from in-memory storage, retained on disk if persistence is enabled).

**Chain compression on eviction:**

```python
def evict_oldest_superseded(entity: Entity, predicate: str) -> None:
    """Evict the oldest superseded fact to make room for a new one.
    
    Chain compression: if A -> B -> C and A is evicted,
    B becomes the chain root. The link A -> B is removed.
    The timeline() method uses disk-backed lazy loading for
    evicted facts, so history is not permanently lost.
    """
    chain = get_supersession_chain(entity, predicate)
    oldest = chain[0]  # Oldest superseded fact
    if oldest.superseded_by:
        # Compress: remove the evicted link
        oldest.superseded_by.previous = None
    hard_delete(oldest)
```

The timeline remains queryable via disk-backed lazy loading for evicted facts. In-memory, only the most recent `max_facts_per_entity` facts are retained.

### C.6 Inferred Layer Optional for v1 (ET-6, HIGH)

Add to Section 7.1 (MemorySystem):

The inferred layer is RETAINED in the architecture for blast-radius containment but is OPTIONAL for v1 deployment. If the inferred layer shows < 1 SubEM point improvement on 3+ evaluation datasets, it MAY be replaced with simple alias expansion at query time (augment_query, Section 5.8).

The 3-layer separation provides defense-in-depth against synonym dictionary pollution — this safety property has value even without measurable retrieval improvement.

**Empirical status:** Inferred layer is net-neutral on LME (28.33% vs 28.67% baseline, 2026-04-12). This is the most dangerous signal — the layer exists but provides no measurable benefit. Either the evaluation does not exercise inferred bridges (wrong test), or the bridges add no value beyond what BM25 already provides. The former is more likely — LME questions are mostly single-hop factual, where BM25 suffices.

**Lethal test for removal:** If SubEM is identical with and without inferred layer on 3 different datasets (LME, FC, and a conversational dataset), the inferred layer should be removed. If SubEM differs by > 1pp on any dataset, the layer is validated.

### C.7 Deterministic Pipeline Self-Validation (ET-7, MEDIUM)

Add to Section 5.6 (DeterministicRouter):

Deterministic pipelines MUST validate their own results. If a pipeline returns 0 results or a confidence below threshold (0.3), it MUST fall through to agentic retrieval rather than returning an empty answer. This self-validation catches misrouted queries.

```python
class DeterministicPipeline:
    """Base class for deterministic retrieval pipelines.
    
    Every pipeline MUST implement validate_result().
    If validation fails, the router falls through to agentic retrieval.
    """
    
    def execute(self, question: str) -> Optional[RetrievalResult]:
        result = self._run(question)
        if not self.validate_result(result):
            return None  # Signal fallthrough to agentic
        return result
    
    def validate_result(self, result: RetrievalResult) -> bool:
        """Returns True if the result is trustworthy.
        
        Default: result has >= 1 evidence chunk with score > 0.3.
        Subclasses may override with pipeline-specific validation.
        """
        return result is not None and len(result.evidence) > 0 and result.confidence > 0.3
```

**Latency impact (from EF-8):** Self-validation creates a bimodal latency distribution. Correctly-routed questions: < 100ms. Misrouted questions that fall through: 100ms + 6-10s agentic. At 30-40% fallthrough rate, average latency is ~3.3s. This is within the 10-second budget but with no margin. The fix is to improve router accuracy (> 80% target), not to remove self-validation.

### C.8 Always-On Means Always-Execute, Not Always-Equal (ET-8, MEDIUM)

Add to Axiom 4 refinement (Section 0.2):

Always-on means all channels EXECUTE on every query. It does NOT mean all channels contribute equally. RRF weights and the reranker determine each channel's influence on the final ranking. A channel that consistently disagrees with the majority is dampened by RRF, not disabled.

**Dilution protection:** From the E2-E6 experiments, additive channels can hurt when a wrong channel has high confidence. The always-on axiom is safe WITH the current RRF weights (BM25 k=60, cosine k=60, entity k=120 dampened) and mxbai reranker. If weights are changed, the dilution risk must be re-evaluated.

### C.9 Allen Algebra Scope Limitation (ET-9, MEDIUM)

Add to Section 2.3.5 (TemporalIndex):

Allen interval algebra covers pairwise temporal relations (ordering, distance, duration, containment). Time-series analytics (trends, frequency changes, seasonal patterns) require a separate TimeSeriesEngine that bins temporal events by time period and computes aggregate statistics. This is a v2 capability, not addressed in v1.

**Scope decomposition:**

| Temporal Question Type | Engine | Version |
|----------------------|--------|---------|
| Ordering (before/after) | Allen | v1 |
| Distance (how long between) | Allen | v1 |
| Duration (how long did X last) | Allen | v1 |
| Containment (during X) | Allen | v1 |
| Trends (increasing/decreasing) | TimeSeriesEngine | v2 |
| Frequency (how often) | TimeSeriesEngine | v2 |
| Seasonality (every summer) | TimeSeriesEngine | v2 |

Allen algebra addresses ~14% of the total question space. Time-series addresses ~13%. Together they cover the 27% temporal gap.

### C.10 Agentic Tool Count Cap (ET-10, MEDIUM)

Add to Section 5.2 (Tool Definitions):

The 8-tool set is designed for < 85% accurate tool selection by gpt-4o-mini. Tool count constraints:

- **8 tools** (current): Goldilocks point. Agent averages 3.2 steps per question.
- **10 tools**: Acceptable. Tool descriptions must be shortened to fit context.
- **12 tools**: Maximum. Beyond 12, tool-selection accuracy drops below 80% (observed in AIP Agent Studio benchmarks).
- **> 12 tools**: REQUIRES two-level selection (category -> specific tool) or tool grouping.

If new capabilities require additional tools, first attempt to extend existing tools (e.g., add parameters to `search_memory`) before adding new tool definitions.

### C.11 FoundryLite SQLite FTS5 Fallback (OT-1)

Add to Section 2.1 (Storage Overview):

**FoundryLite mode:** For deployments with < 10K chunks (cold-start, mobile, resource-constrained), the storage layer MAY use a SQLite FTS5 backend:

```sql
CREATE TABLE memory (
    chunk_id INTEGER PRIMARY KEY,
    turn_idx INTEGER,
    text TEXT,
    role TEXT,
    session_time TEXT,
    entities TEXT
);
CREATE VIRTUAL TABLE memory_fts USING fts5(text, content=memory, content_rowid=chunk_id);
```

FoundryLite provides BM25-equivalent retrieval (~22% SubEM) at 1% of the full system's complexity. It automatically upgrades to the full 7-index backend when chunk count exceeds 10K.

**Performance baseline (from OT-1 analysis):** SQLite FTS5 achieves approximately the same BM25 baseline (58.3% of reachable). The full Foundry adds entity resolution (+6pp), aggregation (+0.67pp), and agentic multi-step (+14.6pp). Total gap: ~21 SubEM points. If this gap narrows to < 10pp on production data, FoundryLite is sufficient for the use case.

### C.12 Hybrid SQL + In-Memory Execution (OT-2)

Add to Section 3.2d (QueryPlan execution):

Filter pushdown operations (`.type()`, `.where()`, `.before()`) MAY be implemented as SQL queries against a SQLite backing store. This leverages SQLite's mature query planner for index-eligible operations while keeping graph traversal in-memory.

```python
class HybridExecutor:
    """Executes filter steps via SQLite, traversal steps via in-memory BFS.
    
    Filter steps: .type(), .where(), .before(), .after()
      -> Compiled to SQL SELECT against SQLite indexes
      -> Returns candidate set as frozenset[str]
    
    Traversal steps: .related(), .pivot_to()
      -> In-memory BFS on EntityGraph
      -> Input: candidate set from filter phase
      -> Output: expanded candidate set
    """
```

### C.13 Resolution Complexity Is Ingestion-Time (OT-3)

Add to Section 3.4 (EntityResolver):

The full 6-step resolution cascade runs at INGESTION TIME only, where cost is amortized across all future queries. At QUERY TIME, only `resolve_exact()` (steps 1-3: alias lookup + determiner strip + possessive strip) is used. This is by design — query-time resolution is O(1) hash lookup, not O(N) embedding comparison.

**Coverage trade-off (from OT-3 analysis):**
- Exact match (steps 1-3): 70% mention resolution, O(1) per mention.
- Full cascade (steps 1-6): ~93% mention resolution, O(N) at step 5 (embedding).
- The 23% gap is absorbed at ingestion time. Query-time callers see fully-resolved entities.

### C.14 Three-Layer Safety Justification (OT-4)

Add to Section 7.1 (MemorySystem):

The 3-layer architecture (STATED, INFERRED, DERIVED) is justified by BLAST-RADIUS CONTAINMENT, not retrieval value:

| Property | Single Tagged Graph | Three Separate Graphs |
|----------|--------------------|-----------------------|
| Wipe safety | Scan + delete tagged facts (O(F), risky) | Pointer reassignment (O(1), safe) |
| Traversal isolation | Tag-filtered BFS (one missed filter = pollution) | Separate data structure (structural isolation) |
| Independent serialization | No | Yes |
| Memory overhead | Lower (no duplicate canonicals) | ~60 KB at 1000 entities (negligible) |

The structural isolation of separate graphs means a bug in synonym dictionary import (inferred layer) CANNOT corrupt stated facts. In a tagged graph, a missing `WHERE layer = 'STATED'` clause exposes all data. The safety property has value even without measurable retrieval improvement.

### C.15 Allen as Implementation, Not Interface (OT-5)

Add to Section 2.3.5 (TemporalIndex):

Rename section title: "Temporal Query Engine (backed by Allen Interval Algebra)"

Allen interval algebra provides the mathematical foundation (13 interval relations). The USER-FACING interface exposes only 5 temporal question buckets:

1. **ordering**: "Did X happen before Y?" → Allen: `before`, `after`, `meets`
2. **distance**: "How long between X and Y?" → Computed from interval endpoints
3. **duration**: "How long did X last?" → Computed from single interval span
4. **containment**: "What happened during X?" → Allen: `during`, `contains`, `overlaps`
5. **recency**: "What happened recently?" → Sort by `end_dt DESC`

The 13 Allen relations are for implementation completeness and formal verification. External callers use the 5-bucket interface.

### C.16 Agentic Retrieval Is the Expensive Path (OT-6)

Add to Section 5.6 (DeterministicRouter):

The performance target for the DeterministicRouter is that it handles >= 60% of questions without invoking the agentic retriever. Agentic retrieval is the EXPENSIVE PATH (6-10 seconds, multiple LLM calls), not the default.

**Latency budget:**
- Deterministic pipeline: < 100ms (BM25 + rerank)
- Agentic retrieval: 3-10 seconds (LLM-in-the-loop, 3-8 tool calls)
- Target average latency: 0.6 * 100ms + 0.4 * 8000ms = 3260ms

Questions that REQUIRE the agent: multi-hop bridging, complex aggregation, ambiguous entity references. These are 30-40% of the question space.

Questions that DO NOT need the agent: single-hop factual, simple entity lookup, temporal ordering. These are 60-70% and should be handled by deterministic pipelines.

### C.17 Supersession for the 20% (OT-7)

Add to Section 6.2 (SupersessionEngine):

**Conflict frequency in practice (from OT-7 analysis):**
- ~80% of (entity, predicate) pairs have exactly 1 fact. No conflict, no supersession needed.
- ~15% have 2 facts. Simple latest-turn-wins is sufficient.
- ~5% have 3+ facts. Supersession chains provide timeline and delta queries.

The supersession machinery exists for the 20% with conflicts. For the 80% without, the machinery is transparent (single-fact entities have no supersession chain). The SubEM impact of proper conflict resolution is +5-8pp (from FC evaluation with `--conflict_resolution` flag).

### C.18 Vector-Only Baseline for Architecture Validation (OT-8, FATAL)

Add to Section 12.1 (Success Criteria):

**Architecture validation baseline:** Vector-only retrieval (ChromaDB/Pinecone + prompting) achieves 4% SubEM on LME. The full Foundry achieves 43.3% SubEM. The 39.3pp gap is the strongest evidence that the multi-index, entity-centric, graph-augmented architecture is necessary.

This baseline MUST be included in all future evaluations as the lower bound. Any architectural simplification that narrows this gap to < 20pp requires re-evaluation of the full system's complexity.

### C.19 Lazy Ingestion Extraction Status Flag (EF-3, HIGH)

Add to Section 4.1 (Ingestion Pipeline):

During async extraction (when BM25 is available but entity/temporal extraction is still running), all query responses MUST include an `extraction_status` flag:

```python
class QueryResponse:
    evidence: list[EvidenceChunk]
    answer: str
    extraction_status: ExtractionStatus  # 'complete' | 'pending' | 'partial'
    pending_chunks: int  # Number of chunks still awaiting extraction
```

**Consistency semantics:**
- `extraction_status='complete'`: All indexes are up-to-date. Aggregation queries are trustworthy.
- `extraction_status='pending'`: BM25 and raw chunks are available. Entity-based, temporal, and aggregation queries return PARTIAL results. The response should warn: "Some data is still being processed."
- `extraction_status='partial'`: Some chunks extracted, others pending. Mixed reliability.

**Stale cache prevention (from EF-3 3rd-order effect):** Query result caches MUST NOT cache results with `extraction_status != 'complete'`. A "0 cats" result during pending extraction must not persist after extraction completes.

### C.20 Co-occurrence Eviction by Recency (EF-2, EF-4, MEDIUM)

Add to Section 2.3.6 (CoOccurrenceIndex):

The `max_cooccurrence_neighbors=100` cap (B.1) uses RECENCY-WEIGHTED eviction rather than pure frequency-based eviction. When a new co-occurrence would exceed the cap, evict the neighbor with the oldest `last_seen_turn_idx`, not the lowest count.

**Rationale (from EF-2 and EF-4):** Frequency-based eviction creates a popularity bias — frequently-mentioned entities survive, rare-but-recent entities are evicted. For multi-hop traversal, recency is a better signal than frequency: a restaurant visited once last week is more relevant than a restaurant mentioned 50 times a year ago.

```python
def add_cooccurrence(self, entity_a: str, entity_b: str, turn_idx: int) -> None:
    neighbors = self.index[entity_a]
    if len(neighbors) >= MAX_COOCCURRENCE_NEIGHBORS:
        # Evict oldest-seen, not lowest-count
        oldest = min(neighbors, key=lambda n: n.last_seen_turn_idx)
        del neighbors[oldest.canonical]
    neighbors[entity_b] = CooccurrenceEntry(count=1, last_seen_turn_idx=turn_idx)
```

### C.21 Agentic Early Termination Governor (DS-3, HIGH)

Add to Section 5.5 (AgenticRetriever):

The agent exhibits a destabilizing positive feedback loop: bad tool selection leads to irrelevant observations, which pollute context, causing worse tool selection, more steps, and token budget exhaustion. This is the most dangerous dynamical property in the retrieval system.

**Governor mechanism:**

```python
class AgenticRetriever:
    early_termination_after: int = 4  # Steps without answer() before fallback
    
    def retrieve(self, question: str) -> RetrievalResult:
        for step in range(self.max_steps):
            action = self.agent.next_action(question, self.observations)
            
            if action.tool == "answer":
                return self._build_result(action)
            
            # Governor: if no answer by step N, fall back to BM25-only
            if step >= self.early_termination_after:
                logger.warning(f"Agent has not answered after {step} steps; falling back to BM25")
                return self._bm25_fallback(question)
            
            observation = self._execute_tool(action)
            self.observations.append(observation)
```

**Empirical basis:** If the agent has not called `answer()` by step 4, the probability of improving the answer in steps 5-8 is < 30% (estimated from trace analysis). The cost of steps 5-8 is 4+ additional LLM calls with degraded context quality.

**Bifurcation point:** Step 4 is the critical transition. Below step 4: agent is exploring productively. Above step 4: agent is likely in the failure attractor. The sliding window compression (B.4) is placed at step 4 to mitigate context pollution, but it also reduces available context for steps 5-8.

### C.22 BM25 Cold-Start IDF Volatility (DS-5, MEDIUM)

Add to Section 2.3.1 (BM25Index):

BM25's IDF component is volatile during the cold-start phase (< 5K documents). A term that is rare in 100 documents has high IDF; the same term in 10,000 documents has lower IDF. Rankings for the same query on the same document CHANGE as the corpus grows.

**Convergence timeline:**
- < 1K documents: IDF shifts by 1-5% per ingestion. Rankings volatile.
- 1K-5K documents: IDF shifts by 0.1-1% per ingestion. Rankings stabilizing.
- \> 5K documents: IDF shifts by < 0.1% per ingestion. Rankings stable.

This is monotonic (always improves) and self-correcting (IDF stabilizes with corpus size). No intervention needed, but evaluation benchmarks MUST ingest all data before querying to avoid cold-start artifacts. Streaming evaluations (ingest-then-query per turn) will show volatile results that do not reflect production behavior.

### C.23 Resolution Threshold Sweep (DS-6, MEDIUM)

Add to Section 3.4 (EntityResolver):

The embedding similarity threshold (0.80 from B.11) is a bifurcation parameter. Mentions with similarity between 0.75 and 0.85 are in a dead zone between "definitely same entity" and "definitely different entity."

**Threshold effects:**
- 0.75: Many false merges (homonyms merged). High recall, low precision.
- 0.80: Balanced (current setting). ~5% false merge rate.
- 0.85: Many false splits (valid mentions create new entities). Low recall, high precision.
- 0.90: Extreme false splits. Entity fragmentation.

The optimal threshold is dataset-dependent — it varies by vocabulary diversity, entity density, and embedding model. The spec sets 0.80 as default but implementations MUST support threshold configuration for per-deployment tuning.

**Recommended validation:** Sweep threshold from 0.70 to 0.95 in 0.05 increments on the target dataset. Measure false merge rate and false split rate. The optimal threshold is at the crossover point where merge_rate = split_rate.

### C.24 Popularity Trap in Decay Function (DS-7, LOW)

Add to Section 7.3 (DecayFunction):

The exponential decay function creates a "popularity trap": frequently-retrieved entities maintain high salience (access resets to 1.0), while rarely-retrieved entities decay toward GC threshold. This is a positive feedback loop — retrieved entities stay salient because they are retrieved.

**Mitigation:** The decay function does NOT determine retrieval ranking (BM25 and reranker do). Salience affects only GC eligibility and derived-layer computation priority. An entity with low salience can still be retrieved via BM25 if the query matches its text. The popularity trap affects memory RETENTION, not memory ACCESS.

**Memory horizon at threshold=0.01:**
- Non-superseded entities survive ~461 days / ~9210 turns without access.
- Superseded entities (10x lambda) survive ~46 days / ~921 turns without access.
- An entity accessed at least once per year is never GC-eligible.

### C.25 Query Filter Count Limit (DS-8, LOW)

Add to Section 3.2d (QueryPlan execution):

Queries with > 5 chained filters transition from "index-dominated" (fast) to "intersection-dominated" (slow) execution. At 10+ filters with non-selective conditions, execution may exceed the 100ms budget.

**Constraint:** QueryPlan SHOULD warn when filter count exceeds 5. Queries with > 10 filters MUST be refused or broken into sub-queries. Agentic retrieval naturally avoids this — the agent builds queries incrementally, rarely exceeding 3 filters per tool call.

### C.26 Supersession Group Staleness (EF-5, MEDIUM)

Add to Section 6.2 (SupersessionEngine):

SUPERSESSION_GROUPS (B.12) require manual curation of predicate sets. As the extraction model evolves and generates new predicates, groups become stale — new predicates that SHOULD be in a group are not classified.

**Staleness detection:** Periodically extract predicates from recent conversations. Compute the fraction of extraction-model-generated predicates that belong to existing supersession groups. If > 50% of employment-related predicates are NOT in the employment group, the groups are stale and need updating.

**Long-term solution:** Supersession groups are a manual patch on a fundamental limitation. As the system matures, consider automatic group discovery via predicate co-occurrence clustering: predicates that are frequently mutually exclusive (one replaces another) should be grouped.

### C.27 Entity Registry Cap and Dense Chunks (EF-6, LOW)

Add to Section 4.3.3 (Entity Extraction):

The `entity_registry_cap=50` limits extraction to 50 entities per LLM call. Entities beyond the cap in entity-dense chunks (> 50 mentions) are NOT extracted — they exist only in raw chunk text, queryable via BM25 but invisible to entity-based retrieval.

**Fragmentation risk:** If entity_51 appears again in a later chunk (within the cap), it gets created as a NEW entity with no link to the earlier mention. This creates entity fragmentation specifically for entities in dense chunks.

**Mitigation:** The cap is necessary to prevent prompt injection (28K failures uncapped). The 50 threshold is empirically tuned. For chunks with > 50 entity mentions, consider splitting the chunk before extraction to keep each sub-chunk under the cap.

### C.28 Embedding Model Change = Full Re-Ingestion (EF-7, LOW)

Add to Section A.5 (MODEL config category):

Changing the embedding model requires FULL RE-INGESTION, not just a vector index rebuild. Entities resolved via step 5 (embedding similarity) may resolve differently with the new model. The mentions are not stored separately — only the resolved canonical name is persisted on facts. To re-verify, the system must re-extract mentions from source_text and re-resolve them.

**Practical impact:** Embedding model changes should be treated as migration events, not configuration changes. Plan for re-ingestion duration: approximately 85 seconds per 100 conversations (sequential extraction), parallelizable across workers.

### C.29 Type Hypernym IS-A vs USED-AS Conflation (EF-9, LOW)

Add to Section B.14 (TYPE_HYPERNYMS):

Type hypernyms conflate ontological categories (IS-A) with contextual categories (USED-AS). Example: "salmon" IS-A "fish," but a grocery-purchased salmon is not a pet fish. `TypeIndex.lookup("fish")` returns both pet fish and food fish through hypernym matching.

**Mitigation:** The hypernym table should be curated conservatively — only include hypernyms where the IS-A relationship is unambiguous in a personal memory context. When in doubt, omit the hypernym and let BM25 handle the broader category matching.

### C.30 Trace Overfitting Guard (EF-10, LOW)

Add to Section 5.5 (AgenticRetriever trace collection):

Deterministic pipelines derived from agentic traces (Phase 3 of the trace-to-deterministic pipeline) risk overfitting to the evaluation distribution. If > 80% of production queries are routed to deterministic pipelines, the system's behavior is dominated by patterns derived from a potentially non-representative evaluation set.

**Guard:** Maintain the agentic retrieval fallback for ALL question types, even those with deterministic pipelines. Periodically route a random 10% sample of deterministic-eligible questions through the agent to detect drift between deterministic and agentic answers. If divergence exceeds 5 SubEM points, the deterministic pipelines need retraining on fresh traces.

---

*End of Adversarial Pass 3 mitigations.*

---

## Appendix D: Adversarial Pass 4 — CK Architecture Mitigations

*These mitigations address the 25 findings from adversarial-pass4.md (Cybernetics Kernel Architecture Review). Each finding is referenced by its ID. Mitigations are organized by CK operator: Recursion (REC), Requisite Variety (RV), Good Regulator (GR), Ultrastability (US).*

---

### D.1 ConversationNode as First-Class Entity (REC-1, HIGH)

Add to Section 1.4 (Entity Type System):

`ConversationNode` is a built-in entity type representing a single conversation session. It is created during Stage 1 (Turn Parsing) whenever a session boundary is detected (e.g., "Chat Time:" header, or > 4 hours gap between turns).

```python
@dataclass
class ConversationNode:
    conversation_id: str            # Deterministic: hash of first turn's timestamp + user_id
    start_time: datetime
    end_time: datetime
    turn_range: tuple[int, int]     # (first_turn_idx, last_turn_idx)
    entity_mentions: set[str]       # Canonical entity names mentioned in this conversation
    topic_terms: list[str]          # Top-5 BM25 terms from the conversation's text
```

Add to Section 2.2 (Core Data Structures):

`RawChunk` gains `conversation_id: Optional[str]`. `Fact` gains `conversation_id: Optional[str]`. Both are populated during Stage 1 when the conversation is segmented.

Add to Section 3.3 (Entry Points):

```python
def conversations(self) -> QueryPlan:
    """Return a QueryPlan scoped to ConversationNode entities."""
    return QueryPlan(source="type_index", type_filter="conversation")
```

This enables: `foundry.conversations().where("topic_terms", "$contains", "cats").execute()` — "Which conversations mentioned cats?"

Add to Section 4.3.1 (Turn Parsing):

After segmenting turns, the parser groups turns into conversations by session boundary. For each conversation, a `ConversationNode` entity is created and stored in the EntityGraph. Links are created from the ConversationNode to every entity mentioned within its turn range. This makes conversations queryable via the standard ObjectSet pattern, not ad-hoc chunk scanning.

---

### D.2 Constrained Subqueries via where_related (REC-2, MEDIUM)

Add to Section 3.2 (Core Types — QueryPlan):

```python
def where_related(self, predicate: str, target_filter: FilterOp) -> "QueryPlan":
    """Filter current entities by a condition on their TRAVERSAL TARGETS.
    
    Returns entities from the CURRENT set that have at least one related entity
    (via `predicate`) satisfying `target_filter`. Does NOT add traversal targets
    to the result set.
    
    Equivalent SQL: SELECT e FROM entities e WHERE EXISTS (
        SELECT 1 FROM links l JOIN entities t ON l.target = t.id
        WHERE l.source = e.id AND l.predicate = :predicate
        AND t.properties MATCH :target_filter
    )
    """
    step = WhereRelatedStep(predicate=predicate, target_filter=target_filter)
    return self._clone_with(step)
```

This is a constrained subquery — it does NOT introduce full nesting (which would require a recursive query planner). It handles the most common multi-hop pattern: "entities whose connections have property X." The implementation traverses links, applies `target_filter` to the targets, and retains only source entities with at least one qualifying target.

Example: `foundry.entities().type("restaurant").where_related("chef", FilterOp("award", "$eq", "michelin"))` returns restaurants (not chefs) that have a Michelin-starred chef.

---

### D.3 Unified Pipeline Entry with start_stage (REC-3, HIGH)

Add to Section 4.2 (Pipeline Stages):

The ingestion pipeline is a SINGLE ordered list of 11 stages. Each API entry point specifies a `start_stage` parameter:

| Entry Point | `start_stage` | Skipped Stages |
|-------------|---------------|----------------|
| `ingest(raw_text)` | 1 | None |
| `ingest_turns(turns)` | 2 | Turn Parsing |
| `ingest_chunk(chunk, triples, entities)` | 5 | Parsing, Chunking, Triple Extraction, Faceted Extraction |

All paths converge into the SAME stage implementations from `start_stage` onward. No stage has conditional logic based on entry point. The pipeline function signature:

```python
def _run_pipeline(self, *, start_stage: int, raw_text: str | None = None,
                  turns: list[ConversationTurn] | None = None,
                  chunk: RawChunk | None = None,
                  triples: list[Triple] | None = None,
                  entities: list[FacetEntity] | None = None) -> IngestionResult:
    stages = [
        self._parse_turns,       # 1
        self._chunk,             # 2
        self._extract_triples,   # 3
        self._extract_entities,  # 4
        self._resolve_entities,  # 5
        self._infer_types,       # 6
        self._extract_temporal,  # 7
        self._track_cooccurrence,# 8
        self._verify_grounding,  # 9
        self._classify_roles,    # 10
        self._route_layers,      # 11
    ]
    context = PipelineContext(raw_text=raw_text, turns=turns, chunk=chunk,
                              triples=triples, entities=entities)
    for stage in stages[start_stage - 1:]:
        context = stage(context)
    return context.result
```

This ensures that a bug fix in Stage 5 (Entity Resolution) applies regardless of whether the data entered via `ingest()`, `ingest_turns()`, or `ingest_chunk()`.

Update Section 10.2 (Ingestion Methods) to reference this unified pipeline: all three methods call `_run_pipeline` with different `start_stage` values.

---

### D.4 Schema Migration via Differential Re-processing (REC-4, LOW)

Add to Section 10.1 (Constructor and Configuration — SCHEMA category):

When FoundryConfig changes at the SCHEMA level, a migration operation determines which pipeline stages must re-run:

```python
def migrate(self, old_config: FoundryConfig, new_config: FoundryConfig) -> MigrationPlan:
    """Compute differential migration plan between two configs."""
    plan = MigrationPlan()
    if old_config.chunk_size != new_config.chunk_size:
        plan.add(stage=2, scope="all")  # Full re-chunking
    if old_config.facets != new_config.facets:
        plan.add(stage=4, scope="affected_entities")  # Re-extract facets
    if old_config.type_hypernyms != new_config.type_hypernyms:
        plan.add(stage=6, scope="affected_types")  # Re-infer types
    if old_config.supersession_groups != new_config.supersession_groups:
        plan.add(stage=11, scope="affected_predicates")  # Re-route layers
    return plan
```

The `manifest.json` (Section A.13) records the config used for each ingestion run. Migration compares old and new manifests to determine scope. For facet additions, only entities of types affected by the new facet need re-extraction. For `chunk_size` changes, full re-chunking is required (consistent with Section A.5 SCHEMA category warning).

---

### D.5 IndexProtocol and Uniform Storage Registry (REC-5, HIGH)

Add to Section 2.3 (Index Types — preamble):

All index types implement `IndexProtocol`, a runtime-checkable protocol:

```python
@runtime_checkable
class IndexProtocol(Protocol):
    def add(self, key: str, data: Any) -> None: ...
    def query(self, key: str, top_k: int) -> list[tuple[str, float]]: ...
    def remove(self, key: str) -> None: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def __len__(self) -> int: ...
    def clear(self) -> None: ...
```

Each concrete index implements the protocol with appropriate type adapters. For example, `TypeIndex.add(key=entity_name, data=type_label)` indexes an entity under a type. `TypeIndex.query(key=type_label, top_k=100)` returns entities of that type. The `data` parameter is `Any` to accommodate index-specific payloads (BM25 accepts text, Vector accepts embeddings, Property accepts structured tuples).

Update Section 2.4 (Persistence Model):

`FoundryStore` becomes a uniform registry:

```python
class FoundryStore:
    def __init__(self, base_path: Path):
        self._indexes: dict[str, IndexProtocol] = {}
        # Register built-in indexes
        self.register("bm25", BM25Index())
        self.register("vector", VectorIndex())
        self.register("type", TypeIndex())
        self.register("property", PropertyIndex())
        self.register("temporal", TemporalIndex())
        self.register("cooccurrence", CoOccurrenceIndex())
        self.register("raw_chunks", RawChunkStore())
    
    def register(self, name: str, index: IndexProtocol) -> None: ...
    def get(self, name: str) -> IndexProtocol: ...
    def save_all(self, path: Path) -> None:
        for name, idx in self._indexes.items():
            idx.save(path / name)
    def load_all(self, path: Path) -> None:
        for name, idx in self._indexes.items():
            idx.load(path / name)
```

EntityMerger (Section B.16) iterates `self._indexes.values()` instead of enumerating 13 data structures by name. Adding a new index type requires only `store.register("new_index", NewIndex())` — no custom wiring.

---

### D.6 TemporalQueryPlan for Composable Temporal Reasoning (REC-6, MEDIUM)

Add to Section 2.3.5 (Temporal Index):

Temporal query results are wrapped in `TemporalQueryPlan`, a chainable query object parallel to `QueryPlan`:

```python
class TemporalQueryPlan:
    """Lazy, immutable, composable temporal query. Clone-on-write semantics (Django pattern)."""
    
    def match(self, query: str) -> "TemporalQueryPlan":
        """BM25 + cosine event retrieval.""" ...
    def before(self, ref: "TemporalEvent | datetime") -> "TemporalQueryPlan":
        """Filter to events ending before ref.""" ...
    def after(self, ref: "TemporalEvent | datetime") -> "TemporalQueryPlan":
        """Filter to events starting after ref.""" ...
    def overlapping(self, interval: "TemporalEvent") -> "TemporalQueryPlan":
        """Filter to events overlapping with interval (Allen: overlaps | during | contains | starts | finishes).""" ...
    def during(self, interval: "TemporalEvent") -> "TemporalQueryPlan":
        """Filter to events fully contained within interval (Allen: during).""" ...
    def execute(self) -> list["TemporalEvent"]:
        """Materialize the query plan and return matching events.""" ...
    def to_entity_plan(self) -> "QueryPlan":
        """Convert temporal results to an entity-level QueryPlan.
        Extracts entity names from matched events and creates a QueryPlan
        filtered to those entities.""" ...
```

This enables: `foundry.temporal().match("trip").overlapping(meeting_event).to_entity_plan().type("restaurant").execute()` — "restaurants visited during the trip that overlaps the meeting."

The `to_entity_plan()` bridge method connects temporal reasoning to entity reasoning. It extracts entity mentions from matched TemporalEvents and creates a QueryPlan pre-filtered to those entities.

---

### D.7 Layer-Annotated Retrieval (REC-7, MEDIUM)

Add to Section 5.2 (Tool Definitions):

The `search_entities` and `get_related` tools gain an optional `layer: Optional[str]` parameter accepting `"stated"`, `"inferred"`, `"derived"`, or `None` (all layers, default).

Each returned fact/property includes a `source_layer` annotation:

```python
@dataclass
class AnnotatedFact:
    """A fact with its provenance layer."""
    fact: Fact
    source_layer: str  # "stated" | "inferred" | "derived"
```

When layer is specified, results are filtered to the requested layer before scoring and ranking. When layer is `None`, results from all layers are returned with annotations, allowing the answer generation prompt to distinguish user statements (STATED) from system computations (DERIVED).

Add to Section 5.5 (AgenticRetriever system prompt):

The agent's system prompt includes guidance: "Facts marked 'stated' are the user's own words. Facts marked 'derived' are computed aggregates. When answering, prefer STATED facts and qualify DERIVED facts with 'based on available information' or similar hedging."

This enforces Axiom 5 ("Stated over inferred") at the retrieval level, not just at storage.

---

### D.8 Output Grounding Verification (REC-8, HIGH)

Add to Section 10.4 (Retrieval Interface):

After answer generation (by the calling LLM), the retrieval result includes an output grounding check:

```python
def verify_answer_grounding(answer: str, evidence: list[str]) -> float:
    """Compute the fraction of answer tokens that are grounded in evidence.
    
    Uses the same token-overlap approach as ingestion-time grounding (Section 4.3.9):
    1. Tokenize answer into content words (exclude stopwords).
    2. For each content word, check if it appears in any evidence string.
    3. Return the fraction of grounded content words.
    
    Returns:
        float: 0.0 = no overlap, 1.0 = fully grounded.
    """
    answer_tokens = _content_tokens(answer)
    evidence_text = " ".join(evidence).lower()
    grounded = sum(1 for t in answer_tokens if t.lower() in evidence_text)
    return grounded / len(answer_tokens) if answer_tokens else 0.0
```

This does NOT require an LLM call. It is a deterministic substring/token-overlap check, consistent with Axiom 2 ("no LLM for infrastructure").

Add to Section 9.2 (Behavioral Invariants):

**I9 (Output Grounding):** For any answer generated from retrieved evidence, `verify_answer_grounding(answer, evidence) >= 0.3`. If the grounding score falls below 0.3, the answer is flagged as potentially hallucinated. The caller should consider returning "I don't have enough information" instead.

Note: The 0.3 threshold is deliberately low because answers may synthesize information from multiple evidence chunks, introducing connecting words not present in any single chunk. The threshold catches gross hallucination (making up entirely new information) without penalizing reasonable synthesis.

---

### D.9 Expanded Deterministic Pipelines (RV-1, MEDIUM)

Add to Section 5.6 (Deterministic Retrieval Pipelines):

Expand from 5 to 9 deterministic pipelines to improve V(R)/V(D) from 0.38 to 0.69:

| Pipeline | Question Types | Steps |
|----------|---------------|-------|
| `fact_lookup` | single_hop_factual | BM25 → rerank → format |
| `temporal` | temporal | BM25 → temporal_filter → Allen reasoning → format |
| `aggregation` | counting_aggregation | entity_type_lookup → count/enumerate → format |
| `preference` | preference | entity_lookup → facts(facet='sentiment') → format |
| `comparison` | comparison | entity_lookup(A) + entity_lookup(B) → property_compare → format |
| `spatial` (NEW) | spatial | entity_lookup → facts(facet='where') → format |
| `social_relational` (NEW) | social_relational | entity_lookup(person) → related(max_hops=2) → facts(facet='who') → format |
| `meta_cognitive` (NEW) | meta_cognitive | BM25(entity_name) → existence_check → "yes/no + source turn" |
| `multi_hop` (NEW) | multi_hop_bridging | entity_lookup → related(max_hops=3, per_hop_cr=True) → collect_evidence → format |

The remaining 4 question types (causal, hypothetical, procedural, opinion_sentiment) require LLM reasoning beyond retrieval routing and legitimately need the agentic fallback. This is V(R)/V(D) = 9/13 = 0.69, above the 0.5 viability threshold.

---

### D.10 Pronoun and Misspelling Resolution (RV-2, LOW)

Add to Section 3.4 (Resolution — entity resolution cascade):

Insert before the embedding similarity step (Step 5) in the cascade:

**Step 4a: Lightweight Coreference.** Track the most-recently-mentioned entity per type category within the current conversation. When the mention is a pronoun ("it", "she", "he", "they", "that"), resolve to the most-recently-mentioned entity of the expected type. Expected type is inferred from the question's verb: "Where did she go?" → recent PERSON entity. "What color is it?" → recent OBJECT entity.

This adds ~1ms per resolution (dictionary lookup). It does not require an LLM call and handles the most common pronoun pattern (anaphoric reference to the most recently discussed entity).

Misspelling tolerance is already partially addressed by Section B.11 (Jaccard overlap addition) and the `$fuzzy` filter operator (Section 3.2a). For misspellings with >2 edit distance that exceed the fuzzy threshold, the BM25 index naturally handles phonetic similarity when the misspelling preserves key character sequences.

---

### D.11 Expanded Conflict Resolution (RV-3, MEDIUM)

Add to Section 6.2 (Supersession Model):

Extend the supersession engine with three additional strategies beyond structural (S,P) matching:

**Correction detection.** During triple extraction (Stage 3), detect correction markers via regex: patterns like "actually", "I meant", "not X but Y", "correction:", "wait, I said ... but". When detected, the extracted fact receives `supersession_hint: bool = True`. The SupersessionEngine gives supersession-hinted facts priority: they supersede matching facts regardless of the structural (S,P) check, because the user has explicitly signaled a correction.

**Retraction handling.** Add `retract` as an action type in Section 6.6 (Action Model). When the user says "forget about X", "never mind about X", or similar retraction patterns, the system creates a `MemoryAction(action="retract", target_entity=X)`. This triggers soft-deletion via Tombstone (Section 6.5) for all facts related to entity X. Retraction bypasses the decay system — it is immediate removal.

**Count-change awareness.** During role classification (Stage 10, VerbNet), classify verbs like "adopted", "bought", "got another" as `sub_role=acquisition`. The DerivedLayerBuilder already handles counting with ownership filters. When a fact with `sub_role=acquisition` is ingested for an entity that already has a count-type fact, the derived layer recomputes the count rather than superseding the original count fact. "I have 2 cats" + "adopted another" → derived count = 3, with both original facts preserved in the STATED layer.

---

### D.12 Additional Retrieval Tools (RV-4, MEDIUM)

Add to Section 5.2 (Tool Definitions):

**Tool 9: compare_entities**

```python
def compare_entities(entity_a: str, entity_b: str, property: str) -> str:
    """Compare a specific property across two entities.
    
    Retrieves the property value for both entities and returns a structured
    comparison. Handles missing properties gracefully.
    
    Returns: "Entity A: <value_a>. Entity B: <value_b>. [Comparison summary]."
    """
```

This eliminates the common 3-step agent pattern (search_entities(A) → search_entities(B) → execute_python(compare)). Single tool call for comparison questions.

**Tool 10: list_entities**

```python
def list_entities(entity_type: str, max_results: int = 20) -> str:
    """List entities of a given type with their key facts.
    
    Returns up to max_results entities of the specified type, each with
    their top-3 most salient facts. Type matching is case-insensitive
    and tries depluralization.
    
    Returns: Formatted list of entities with key facts.
    """
```

This eliminates the aggregate + search_entities two-step pattern for enumeration questions.

Update Section 5.9 (Agent Tool Taxonomy) to include the two new tools in the GATHER category. Total tool count: 10, well within the 12-tool degradation threshold (Section C.10).

---

### D.13 Importance-Gated Decay (RV-5, MEDIUM)

Add to Section 7.3 (Decay Model):

Wire ImportanceScore (Section 7.2) into DecayFunction:

```python
def compute(self, days_since_access: float, importance: ImportanceScore,
            base_salience: float = 1.0, superseded: bool = False) -> float:
    """Compute current salience.
    
    Importance gating: high-importance entities (health, allergies, core identity)
    decay slower. The importance.composite score ranges from 0 to ~5 (sum of
    mention_frequency, retrieval_frequency, link_count, recency, update_frequency).
    
    A composite of 3.0 (actively discussed entity) halves the decay rate.
    A composite of 0.2 (mentioned once, never retrieved) leaves decay nearly unchanged.
    """
    effective_lambda = self.lambda_base / (1.0 + importance.composite)
    if superseded:
        effective_lambda *= 10  # Superseded entities decay 10x faster (unchanged)
    return base_salience * math.exp(-effective_lambda * days_since_access)
```

This means a frequently-retrieved entity (importance.composite = 4.0) decays at 1/5th the rate of a never-retrieved entity. Health conditions and allergies, which are queried often, naturally persist longer.

User-initiated deletion (via the `retract` action from D.11) bypasses decay entirely and triggers immediate soft-delete via Tombstone.

---

### D.14 Extraction Prompt Enhancements (RV-6, LOW)

Add to Section 4.3.3 (Triple Extraction):

The extraction prompt includes explicit handling instructions for conversation patterns that are empirically known to cause extraction errors:

1. **Hypothetical guard**: "Do NOT extract entities from hypothetical or conditional statements. If the text says 'If I were to get a dog' or 'Suppose I moved to Paris', do NOT create dog or Paris entities. Tag any facts extracted from conditional context with `hypothetical: true`."

2. **Correction handling**: "When the user corrects themselves ('Actually, it was 3 not 2', 'I meant Portland, not Seattle'), extract ONLY the corrected fact. Do NOT extract the original incorrect statement. Mark the corrected fact with `correction: true`."

3. **Reported speech attribution**: "When the user reports what someone else said ('Sarah told me she's moving', 'My doctor says I should exercise'), attribute the fact to the SPEAKER (Sarah, doctor), not to the user. The subject of the triple is the speaker."

4. **Sarcasm detection**: "When the user expresses sarcasm (indicated by emphasis, 'oh sure', 'just LOVE', or contradictory sentiment), tag with `sarcasm: true` and do NOT extract the literal meaning. 'Oh sure, I just LOVE getting up at 5am' should NOT produce a preference fact for early mornings."

These additions increase prompt length by ~200 tokens but do not add LLM calls. They address the four most common extraction failure patterns from evaluation traces.

---

### D.15 StorageManifest — Self-Model for Query Planning (GR-1, HIGH)

Add to Section 2.4 (Persistence Model):

```python
@dataclass
class StorageManifest:
    """Lightweight model of storage contents. Updated after each ingestion.
    Queried by the router and query executor. Costs zero LLM calls.
    
    This is the Good Regulator for the query system — the query planner
    cannot make informed decisions without knowing what the storage contains.
    """
    entity_types: set[str]            # All type labels in TypeIndex
    property_predicates: set[str]     # All predicates in PropertyIndex
    entity_count: int                 # Total entities in EntityGraph
    fact_count: int                   # Total facts across all entities
    temporal_event_count: int         # Events in TemporalIndex
    chunk_count: int                  # Chunks in RawChunkStore
    type_counts: dict[str, int]      # type_label -> entity count
    top_entities: list[str]           # Top-20 entities by mention frequency
    last_updated: datetime
    
    def has_type(self, type_label: str) -> bool:
        """Check if a type exists, with depluralization fallback."""
        if type_label in self.entity_types:
            return True
        singular = depluralize(type_label)  # "restaurants" -> "restaurant"
        return singular in self.entity_types
    
    def type_count(self, type_label: str) -> int:
        """Get entity count for a type, with depluralization."""
        return self.type_counts.get(type_label, 
               self.type_counts.get(depluralize(type_label), 0))
```

The StorageManifest is rebuilt after each ingestion call (incremental update: add new types and entities, increment counts). It is persisted alongside indexes in `manifest.json`.

Update Section 5.6 (DeterministicRouter): The router consults StorageManifest before routing. If the `counting_aggregation` pipeline targets a type not in `entity_types`, it tries depluralization, then synonym expansion from TYPE_HYPERNYMS (Section B.14), before falling through to agentic. This prevents the empty-query wasted-computation pattern identified in GR-1.

---

### D.16 Knowledge Summary for Agentic Retriever (GR-2, MEDIUM)

Add to Section 5.5 (Agentic Retriever):

The agent's system prompt is augmented with a knowledge summary derived from StorageManifest (D.15):

```python
KNOWLEDGE_SUMMARY_TEMPLATE = """
## Available Memory
- {entity_count} entities across {type_count} types
- {chunk_count} text chunks ({temporal_count} with temporal grounding)
- Top entity types: {top_types}
- Recently active entities: {recent_entities}
"""
```

This is ~100 tokens of additional system prompt per invocation. The knowledge summary allows the agent to make informed tool selection: "I know there are 15 restaurant entities, so `list_entities('restaurant')` will work" instead of blind probing with `search_entities("restaurant")`.

The knowledge summary is computed from StorageManifest at query time. It costs zero LLM calls and zero additional retrieval operations.

---

### D.17 Entity Context in Extraction Prompt (GR-3, LOW)

Add to Section 4.3.3 (Triple Extraction):

The extraction prompt receives a condensed list of existing entities as context. This is NOT the `entity_registry_cap` (which caps OUTPUT entities) but rather an INPUT context window:

```python
def _build_extraction_context(self, entity_graph: EntityGraph, top_k: int = 50) -> str:
    """Provide the top-K existing entities as context to the extraction prompt.
    
    Entities are ranked by mention_frequency (from ImportanceScore).
    Format: "Known entities: doctor_smith (person), fluffy (pet), mercy_clinic (place), ..."
    """
    top_entities = entity_graph.top_entities_by_mentions(top_k)
    return "Known entities: " + ", ".join(
        f"{e.name} ({e.type_label})" for e in top_entities
    )
```

This adds ~150 tokens to the extraction prompt but significantly improves entity naming consistency across chunks. When the LLM sees "Known entities: doctor_smith (person)" in its prompt, it is more likely to extract "Dr. Smith" as `doctor_smith` rather than inventing a new surface form. This reduces unnecessary entity resolution work downstream.

---

### D.18 Reliability-Gated Supersession (GR-4, MEDIUM)

Add to Section 6.2 (Supersession Model):

Each Fact receives a `reliability: float` computed deterministically at ingestion:

```python
def compute_reliability(fact: Fact) -> float:
    """Deterministic reliability score. No LLM call required.
    
    Scoring factors:
    - User role: +0.3 (user's own words are more authoritative)
    - Correction marker: +0.1 ("actually", "I meant", "correction:")
    - Hedging marker: -0.1 ("I think", "maybe", "probably")
    - Hypothetical: -0.3 ("if", "suppose", "would")
    - Assistant role: +0.0 (neutral — assistant restating is neither more nor less reliable)
    """
    score = 0.5
    if fact.role == "user":
        score += 0.3
    source_lower = fact.source_text.lower()
    if any(m in source_lower for m in ["actually", "i meant", "correction", "wait, "]):
        score += 0.1
    if any(m in source_lower for m in ["i think", "maybe", "probably", "not sure"]):
        score -= 0.1
    if fact.sub_role == "hypothetical":
        score -= 0.3
    return max(0.0, min(1.0, score))
```

The SupersessionEngine consults reliability during conflict resolution: a newer fact supersedes an older one ONLY IF `new_fact.reliability >= old_fact.reliability * 0.8`. This prevents low-confidence assistant inferences (reliability 0.5) from overwriting high-confidence user corrections (reliability 0.9).

This is consistent with the existing `user_wins` strategy (Section 6.2) but makes it quantitative rather than binary. The `latest_wins` default strategy is preserved when both facts have similar reliability (within the 0.8 threshold).

---

### D.19 User-Adaptive Decay Parameters (GR-5, LOW)

Add to Section 7.3 (Decay Model):

Scale decay parameters by observed user session frequency:

```python
def adjusted_lambda(base_lambda: float, avg_sessions_per_day: float) -> float:
    """Adjust decay rate for individual user conversation patterns.
    
    A daily user (10 sessions/day) uses standard lambda.
    A monthly user (0.03 sessions/day) uses lambda scaled by ~30x slower.
    This prevents premature GC-eligibility for infrequent users.
    """
    # Clamp to prevent extreme values
    frequency_factor = max(0.1, min(10.0, avg_sessions_per_day))
    return base_lambda * frequency_factor
```

Track `avg_sessions_per_day` in FoundryConfig (RUNTIME category). Update it after each session as an exponential moving average:

```python
new_avg = 0.9 * old_avg + 0.1 * (1.0 / days_since_last_session)
```

For a daily user: `avg ≈ 1.0`, so `lambda_effective ≈ lambda_base` (standard decay).
For a monthly user: `avg ≈ 0.03`, so `lambda_effective ≈ lambda_base * 0.1` (10x slower decay, clamped at minimum).

This adaptation requires no LLM call — it is a single float update per session.

---

### D.20 Temporal Precision Tracking (GR-6, MEDIUM)

Add to Section 2.2 (Core Data Structures — TemporalEvent):

```python
class TemporalPrecision(str, Enum):
    YEAR = "year"       # "2024" → [2024-01-01, 2024-12-31]
    MONTH = "month"     # "March 2024" → [2024-03-01, 2024-03-31]
    DAY = "day"         # "March 15" → [2024-03-15, 2024-03-15]
    HOUR = "hour"       # "3pm March 15" → [2024-03-15T15:00, 2024-03-15T15:59]
    EXACT = "exact"     # Precise timestamp from system clock

@dataclass
class TemporalEvent:
    entity_name: str
    predicate: str
    start_dt: datetime
    end_dt: datetime
    precision: TemporalPrecision  # NEW
    source_text: str
```

Add to Section 2.3.5 (Temporal Index):

Distance computations account for precision. When comparing events of different precisions, the result is a range rather than a point:

```python
def temporal_distance(a: TemporalEvent, b: TemporalEvent) -> tuple[float, float]:
    """Return (min_distance_days, max_distance_days).
    
    For same-precision events: min == max (exact distance).
    For different-precision events: range reflects uncertainty.
    Example: distance(MONTH "March 2024", DAY "March 15") = (0, 30).
    """
```

Allen interval operations are precision-aware: `contains(MONTH, DAY)` checks whether the DAY falls within the MONTH's range. `overlaps(YEAR, MONTH)` is always true when the month is within the year.

The extraction prompt (Section 4.3.7) is updated to infer precision from the source text. Dateparser already provides this signal: "March 2024" yields `period="month"`, while "March 15, 2024" yields `period="day"`.

---

### D.21 SessionContext for Within-Session Adaptation (US-1, HIGH)

Add to Section 5.5 (Agentic Retriever):

```python
@dataclass
class SessionContext:
    """Accumulates context within a single conversation session.
    Persists across questions within one session. Reset on new session.
    Updated deterministically after each question (no LLM call)."""
    
    focal_entities: list[str]              # Entities mentioned in >= 2 questions
    topic_terms: set[str]                  # Terms appearing in >= 2 questions
    successful_tools: dict[str, int]       # Tool name -> success count
    question_count: int                    # Questions processed this session
    
    def update(self, question: str, trace: RetrievalTrace) -> None:
        """Update session context after a question is answered."""
        self.question_count += 1
        # Extract entities mentioned in question (via entity resolution cascade)
        mentioned = self._extract_entity_mentions(question)
        for entity in mentioned:
            if entity in self._seen_entities:
                if entity not in self.focal_entities:
                    self.focal_entities.append(entity)
            self._seen_entities.add(entity)
        # Track successful tools
        for step in trace.steps:
            if step.result_count > 0:
                self.successful_tools[step.tool] = \
                    self.successful_tools.get(step.tool, 0) + 1
```

The SessionContext is injected into the agent's system prompt as additional context:

```
Session focus: The user has been asking about {focal_entities}.
Recurring topics: {topic_terms}.
Effective tools this session: {successful_tools}.
```

This adds ~50 tokens per question but prevents the agent from re-discovering focal entities on every call. When the user asks 5 questions about Rome, the agent knows "Rome" is focal and starts with Rome-related retrieval.

---

### D.22 PipelinePerformanceTracker for Cross-Session Learning (US-2, LOW)

Add to Section 5.6 (Deterministic Retrieval Pipelines):

```python
class PipelinePerformanceTracker:
    """Tracks deterministic pipeline accuracy over a sliding window.
    Provides the cross-session adaptation loop (US-2).
    Persisted in FoundryStore alongside indexes."""
    
    def record(self, pipeline: str, question: str, succeeded: bool) -> None:
        """Record whether a pipeline produced a satisfactory answer."""
        self._records[pipeline].append((question, succeeded))
        # Keep sliding window of last 100 per pipeline
        if len(self._records[pipeline]) > 100:
            self._records[pipeline].pop(0)
    
    def get_accuracy(self, pipeline: str) -> float:
        """Accuracy over the sliding window."""
        records = self._records.get(pipeline, [])
        if not records:
            return 1.0  # No data → assume working
        return sum(1 for _, s in records if s) / len(records)
    
    def should_disable(self, pipeline: str, threshold: float = 0.6) -> bool:
        """Disable pipeline if accuracy drops below threshold."""
        return self.get_accuracy(pipeline) < threshold
```

When `should_disable()` returns True, all questions that would have been routed to that pipeline are instead routed to the agentic fallback. The disabled pipeline is re-enabled after 50 agentic queries if the agentic traces show the same tool sequence that the pipeline would have used (indicating the pipeline's pattern is still valid but needs recalibration).

"Succeeded" is determined by the self-validation check (Section C.7): if the deterministic pipeline's answer passes self-validation, it succeeded. If it falls through to agentic, it did not.

---

### D.23 Novel Entity Type Detection (US-3, LOW)

Add to Section 4.3.6 (Type Inference):

```python
def infer_type_with_confidence(
    entity: FacetEntity, 
    type_index: TypeIndex
) -> tuple[str, float]:
    """Infer entity type and confidence based on prior type distribution.
    
    Confidence levels:
    - 1.0: Type has >5 existing instances (well-established category)
    - 0.7: Type has 1-5 instances (emerging category)
    - 0.3: Type has 0 instances (novel category)
    """
    type_label = EntityType.normalize(entity.facets.get("what", ""))
    existing_count = type_index.count(type_label)
    if existing_count > 5:
        confidence = 1.0
    elif existing_count > 0:
        confidence = 0.7
    else:
        confidence = 0.3
    return type_label, confidence
```

Entities with `type_confidence < 0.5` are flagged in the EntityGraph. The retrieval layer consults this flag: when a query targets a low-confidence-type entity, the system preferentially uses BM25 (which does not depend on type classification) rather than type-filtered entity retrieval.

The StorageManifest (D.15) reports low-confidence types separately: `novel_types: set[str]` allows the DeterministicRouter to route queries about novel types directly to the agentic fallback rather than attempting a type-based deterministic pipeline.

---

### D.24 Within-Query Adaptation for Deterministic Pipelines (US-4, MEDIUM)

Add to Section 5.6 (Deterministic Retrieval Pipelines):

Each deterministic pipeline supports a 2-level fallback before falling through to the agentic retriever:

```python
class DeterministicPipeline:
    def execute(self, question: str) -> Optional[RetrievalResult]:
        # Level 1: Standard execution
        result = self._primary_execute(question)
        if self._sufficient(result):
            return result
        
        # Level 2: Query expansion
        expanded_query = augment_query(
            question, 
            self.entity_resolver, 
            self.alias_index
        )
        result = self._primary_execute(expanded_query)
        if self._sufficient(result):
            return result
        
        # Level 3: Fall through to agentic
        return None
    
    def _sufficient(self, result: Optional[RetrievalResult]) -> bool:
        """Check if result has enough evidence to answer the question.
        Threshold: at least 1 chunk with BM25 score > 0.1."""
        if result is None:
            return False
        return any(score > 0.1 for _, score in result.ranked_chunks)
```

The `augment_query` function (from Section 5.8, OAG patterns) expands the query with entity aliases and related terms. This adds ~10ms overhead (one alias lookup + one BM25 query) but catches the common case where a synonym or alias expansion would have found the answer.

The agentic fallback (6-10 seconds) is reserved for genuinely hard cases where query expansion is insufficient.

---

### D.25 Evidence Confidence Score (US-5, HIGH)

Add to Section 10.4 (Retrieval Interface):

The retrieval result gains a `confidence` field:

```python
@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    evidence: str                    # Concatenated evidence text
    trace: RetrievalTrace            # Full execution trace
    confidence: float                # 0.0 = no relevant evidence, 1.0 = high quality
    grounding_score: float           # Output of verify_answer_grounding (D.8)
    
    @staticmethod
    def compute_confidence(trace: RetrievalTrace, evidence: str) -> float:
        """Deterministic confidence estimation. No LLM call.
        
        Factors:
        - Termination: agent chose 'answer' (0.5) vs budget exhausted (0.1) vs error (0.0)
        - Step efficiency: fewer steps = higher confidence (max 0.3 bonus)
        - Evidence volume: at least 1 relevant chunk (0.2 bonus)
        """
        # Base: termination quality
        if trace.terminated_by == "answer":
            base = 0.5
        elif trace.terminated_by == "budget":
            base = 0.1
        else:
            base = 0.0
        
        # Efficiency bonus: 1-3 steps is efficient
        steps = len(trace.steps)
        efficiency = max(0.0, 0.3 - (steps / 8.0) * 0.3)
        
        # Evidence bonus: non-empty evidence
        evidence_bonus = 0.2 if len(evidence.strip()) > 50 else 0.0
        
        return min(1.0, base + efficiency + evidence_bonus)
```

When confidence < 0.3, the system advises the calling LLM to respond with "I don't have enough information to answer that" rather than generating a potentially hallucinated answer. This is the degradation signal (US-5) — the system can now distinguish "high-confidence answer" from "best-effort guess."

The confidence score is included in the `RetrievalTrace` for audit purposes and in the response payload for the calling LLM.

---

### D.26 S3* Audit Subsystem (VSM Gap, HIGH)

Add to Section 10.6 (Statistics and Introspection):

The Foundry includes a periodic audit subsystem corresponding to Beer's S3* (sporadic audit). This runs on demand (not every query) and verifies internal consistency:

```python
@dataclass
class AuditReport:
    """Results of an integrity audit. Run periodically, not per-query."""
    grounding_violations: list[str]     # Facts where source_text check fails
    orphaned_entities: list[str]        # Entities with no facts
    alias_desyncs: list[str]            # Aliases not matching EntityGraph
    type_index_desyncs: list[str]       # TypeIndex entries not in EntityGraph
    temporal_orphans: list[str]         # TemporalEvents with no matching entity
    supersession_cycles: list[str]      # Circular supersession chains
    total_violations: int

def audit(self) -> AuditReport:
    """Run integrity audit across all storage subsystems.
    
    Checks:
    1. Grounding: For each fact, verify source_text contains fact.object
    2. Entity orphans: Entities with 0 facts (should have been GC'd)
    3. Alias sync: Every alias resolves to an entity in EntityGraph
    4. Type sync: Every TypeIndex entry maps to an existing entity
    5. Temporal orphans: Every TemporalEvent references an existing entity
    6. Supersession cycles: No fact chain A→B→C→A
    """
```

The audit is triggered on demand via `foundry.audit()` and can be scheduled (e.g., after every 100 ingestion calls). It does NOT run per-query (that would violate latency constraints). Violations are logged and optionally auto-repaired (remove orphans, re-sync indexes).

This fills the S3* gap identified in the VSM scorecard. Without periodic audit, corrupted state is invisible until a user query returns wrong results.

---

### D.27 S4 Intelligence Seed — Trace Analysis (VSM Gap, LOW)

Add to Section 5.5 (Agentic Retriever — Trace Collection):

While the full HebbianBridge is explicitly NOT implemented in v1, the trace collection mechanism (already specified) is the seed of S4 intelligence. Augment trace collection with automated pattern detection:

```python
class TraceAnalyzer:
    """Automated analysis of agentic retrieval traces.
    Runs periodically (e.g., after every 50 traces). Identifies:
    - Common tool sequences (for deterministic pipeline generation)
    - Frequently queried entity types (for StorageManifest tuning)
    - Common failure patterns (for extraction prompt improvement)
    """
    
    def analyze_batch(self, traces: list[RetrievalTrace]) -> TraceInsights:
        tool_sequences = Counter()
        for trace in traces:
            seq = tuple(s.tool for s in trace.steps if s.tool != "answer")
            tool_sequences[seq] += 1
        
        return TraceInsights(
            common_sequences=tool_sequences.most_common(10),
            avg_steps=mean(len(t.steps) for t in traces),
            success_rate=mean(1 for t in traces if t.terminated_by == "answer") / len(traces),
            frequent_entities=self._extract_frequent_entities(traces),
        )
```

TraceInsights are stored alongside the traces and available to the PipelinePerformanceTracker (D.22). When a tool sequence appears in >10 traces with >80% success, it is a candidate for a new deterministic pipeline. This closes the adaptation loop at the slowest timescale (cross-session), converting agentic exploration into deterministic efficiency over time.

---

*End of Adversarial Pass 4 (CK Architecture) mitigations.*

---

## Appendix E: Phase 3.25 Discriminative Validation — Tribunal Verdicts

*These verdicts summarize the Brenner tribunal results from Phase 3.25. Each hypothesis was evaluated by three agents (generator, test designer, adversarial critic) with executable test suites. Full tribunal outputs are in `foundry/tribunal/`. Spec changes below are normative — they override prior text where conflicts exist.*

---

### E.1 H1: BM25 Primary — VALIDATED WITH QUALIFICATION

**Verdict:** BM25 is the correct primary retriever. It is NOT sufficient alone.

**Key evidence:**
- BM25 is TRUE but TRIVIAL — it is the best retriever, but retrieval is not the bottleneck
- 53% of questions are unreachable by ANY retriever (computation gap, not retrieval gap)
- 79/159 failures need temporal math, aggregation, or multi-turn synthesis — no retriever solves these
- BM25 is 72.9% of agentic retrieval calls and the first tool 90% of the time — validates primary status
- 76.7% of agentic improvement comes from iterative BM25 — validates the BM25 mechanism itself
- Only 12.6% of failures are genuine BM25 vocabulary gaps

**Critic qualification:** Lean BM25 (no extras) = 23.00% SubEM. The 58.3% figure conflates SubEM with Hit@K and includes dynamic_prompting + entity_prefilter. BM25 IDF degrades non-linearly at lifetime scale (50K-180K turns) — untested regime.

**Spec changes:**

1. **Axiom 1 reworded.** "BM25 is the primary RETRIEVAL channel; computation engines (temporal, aggregation, synthesis) are co-primary for their question types." Computation engines are elevated to first-class components, not retrieval supplements.

2. **Computation engine requirements added.** The following engines are mandatory subsystems, not optional enrichments:
   - Temporal engine (Allen interval algebra, date math)
   - Aggregation engine (counting, set operations, extrema)
   - Synthesis engine (multi-turn summarization, preference reconciliation)

3. **Scaling acceptance criteria.** Phase 5 must include BM25 IDF degradation testing at 10K, 50K, and 100K turns. If IDF degrades > 5pp SubEM at 100K turns, introduce IDF sharding or per-conversation normalization.

---

### E.2 H2: Deterministic Routing — PARTIALLY VALIDATED (3-Tier Architecture)

**Verdict:** The hypothesis as stated is imprecise. Routing decomposes into three operations with different intelligence requirements.

**Decomposition:**

| Operation | Deterministic? | Evidence |
|-----------|---------------|----------|
| Question classification (type labeling) | 80% yes, 20% needs NLU | Regex classifier: 60.0% accuracy, 69% of misclassifications fixable by regex improvement |
| Strategy selection (type to pipeline) | 100% yes | Unconditional mapping table, zero ambiguity |
| Adaptive iteration (observe, decide next step) | No — irreducibly requires intelligence | 14.6pp agentic gain (28.67% to 43.3%) lives entirely here |

**Critic findings (4 critical issues):**
1. MemoryRouter IS LLM-based (uses 8B model at 116ms/q) — the "deterministic" alternative is regex, not MemoryRouter
2. `_parse_route` has substring collision bugs ("sum" in "summary" triggers aggregation)
3. Aggregation route returns bare number string with no LLM generation
4. `execute_python` sandbox has `__builtins__.__import__` bypass

**Spec changes — 3-Tier Architecture:**

Replace all references to "deterministic routing" with the following tiered architecture:

**Tier 1: Deterministic Pipelines** (always-on, parallel, <100ms, $0)
- All deterministic pipelines execute in parallel for every query
- Pipelines: BM25 retrieval, temporal engine, aggregation engine, entity-filter, synthesis
- Each pipeline returns a candidate answer with a confidence score
- No routing decision — all pipelines always run

**Tier 2: Confidence Gate** (<1ms, $0)
- Select the highest-confidence pipeline answer above threshold
- If multiple pipelines exceed threshold, prefer the most specific (temporal > aggregation > BM25)
- If no pipeline exceeds threshold, fall through to Tier 3
- Threshold is a single tunable parameter (default: 0.7)

**Tier 3: Agentic Fallback** (~20% of queries, 6-10s, LLM cost)
- ReAct loop with 8 tools (search_memory, search_entities, get_related, search_temporal, read_turn, execute_python, aggregate, answer)
- Activated only when Tier 2 confidence gate fails
- All traces are saved for trace-to-pipeline extraction

**Axiom 2 reframed.** Tiers 1+2 are zero-cost deterministic. Tier 3 is opt-in LLM for hard questions. The "no LLM for infrastructure" principle applies to Tiers 1+2; Tier 3 is explicitly LLM-powered.

**Trace-to-pipeline extraction** is a first-class capability: when a Tier 3 tool sequence appears in >10 traces with >80% success rate, it becomes a candidate for a new Tier 1 deterministic pipeline. This closes the adaptation loop from the user's Phase 3-to-4 vision (agentic traces become deterministic pipelines become distilled SLM routing).

**Sandbox fix.** `execute_python` must restrict `__builtins__` to a safe subset:

```python
SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter",
    "float", "frozenset", "hasattr", "int", "isinstance", "len",
    "list", "map", "max", "min", "print", "range", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "type", "zip",
}

sandbox_globals = {"__builtins__": {k: __builtins__[k] for k in SAFE_BUILTINS}}
```

---

### E.3 H3: Entity Resolution — FALSIFIED

**Verdict:** Current entity resolution (substring + embedding) is fundamentally broken at production scale. EntityStore BM25 is the correct mechanism.

**Key evidence:**
- `resolve()` substring fallback: **100% false merge rate** on adversarial pairs
- `resolve_exact()`: **0% false merge** — the only safe substring path
- Embedding similarity: **0/8** deictic references resolved (worse than random)
- EntityStore BM25: **6/8** deictic references resolved (75% — strongest mechanism)
- Type-gated resolution: **0% false merge** when types available; 18.8% without types
- Homonym merge: **100%** — no disambiguation mechanism exists
- Canonical names are order-dependent — first entity created wins forever

**Failure mechanism:** Short canonical names from triple objects (2-3 char strings like "or", "it", "up") become universal attractors in substring matching. Object-as-entity contamination means every triple object creates an entity that participates in resolution. Phase transition occurs at N~1,000-3,000 entities — exactly the benchmark scale. False merge propagates through a 3-hop consequence chain: false merge -> corrupted entity doc -> wrong aggregation -> wrong answer stated confidently. No recovery mechanism exists.

**Spec changes — major architectural revision:**

1. **Primary resolution mechanism.** Replace substring resolution with EntityStore BM25 as primary entity resolution at query time. EntityStore BM25 searches rich entity documents (containing all facts, mentions, and context) and returns ranked entity candidates. This outperforms graph traversal in both H1 and H3 tribunals.

2. **Substring minimum length.** Any remaining substring matching (ingestion-time only) requires `len(canonical_name) >= 4`. This eliminates the universal-attractor failure mode from short strings.

3. **Homonym disambiguation.** Type gating is mandatory, not optional. When resolving an entity mention, candidate entities must share a compatible type. "Turkey" (country) does not match "turkey" (food) when type information is available.

4. **Separate resolution from creation.** The current `_resolve_or_create` conflates entity RESOLUTION (linking a mention to an existing node) with entity CREATION (adding a new node). These must be separate operations:
   - `resolve(mention) -> Optional[EntityNode]` — pure lookup, never creates
   - `create(mention, type, source) -> EntityNode` — explicit creation with provenance
   - Creation requires evidence threshold (minimum 2 mentions or 1 proper noun)

5. **EntitySplitter.** Add an inverse of EntityMerger for false-merge recovery. When an entity document contains contradictory facts (e.g., "born in 1985" AND "born in 1990"), the EntitySplitter proposes a split. Splits are confirmed by checking whether the constituent facts form coherent clusters.

6. **Restrict triple-object entities.** Only proper nouns become entity nodes. Adjectives, states, and quantities become property values on existing entities, not standalone nodes. This eliminates the object-as-entity contamination that generates thousands of spurious entity entries.

7. **Longest-match-wins.** For any remaining substring paths, use longest-match-wins instead of first-match. This prevents "la" matching inside "plane" or "art" matching inside "party".

---

### E.4 Convergent Insight: Computation Over Retrieval

All three tribunals independently converged on the same conclusion: **the bottleneck is computation, not retrieval.**

| Tribunal | Finding |
|----------|---------|
| H1 | "Retrieval is not the bottleneck" — 53% of questions need computation engines |
| H2 | "Adaptive iteration is where the value lives" — LLM gain is in deciding what to DO with retrieved data |
| H3 | "Entity resolution's value is in typing/aggregation, not retrieval" — the entity graph serves computation, not recall |

**Architectural implication:** The spec elevates computation engines (temporal math, aggregation, multi-turn synthesis) to co-equal status with retrieval. They are independent subsystems that the 3-tier architecture activates based on question type, not downstream consumers of retrieval results.

**EntityStore BM25 over graph traversal.** Both H1 and H3 tribunals found that EntityStore BM25 (searching rich entity documents) outperforms entity graph traversal (resolve -> hop -> filter). The agentic retriever confirms: `search_entities` (BM25 over entity docs) is useful; `get_related` (graph traversal) is nearly useless. The ObjectSet query language should internally use BM25 for entity lookup, not programmatic resolution.

---

## Appendix F: Multilingual Extraction — Research Synthesis

*This appendix documents research findings from 6 papers (ACL 2025, ICLR 2026, Stanford, RPI, Renmin U) that inform Foundry's multilingual extraction architecture. The key finding is that VerbNet — the current role classification backbone — is English-only and must be replaced with LLM-native classification for multilingual deployment.*

---

### F.1 The VerbNet Problem

The Foundry spec (Section 4.3, MSL v2 schema) uses VerbNet verb-class lookup for role and sub_role classification:
- 90 acquisition lemmas (buy, get, receive, ...)
- 54 maintenance lemmas (keep, have, own, ...)
- 32 intent lemmas (want, plan, hope, ...)

VerbNet is the University of Colorado's English verb lexicon. It has no coverage for any other language. This means the entire role/sub_role classification pipeline — which powers temporal reasoning ("when did the user ACQUIRE X?"), state tracking ("does the user still HAVE X?"), and intent detection ("does the user WANT X?") — is English-only.

This blocks multilingual deployment of Foundry. A user conversing in Spanish, Japanese, or Arabic gets no role classification, which degrades temporal queries and aggregation accuracy.

---

### F.2 Solution: LLM-Native Role Classification

**MASSIE** (Le et al., ACL 2025 Findings) provides the key evidence: instruction-following LLM extraction works across **96 languages and 18 language families** without language-specific lexicons. The extraction model (already Qwen2.5-7B in Foundry) handles role/relation classification natively when instructed via the prompt schema.

**Architecture change:**

```
Before (English-only):
  Text -> LLM extraction (SPO + when) -> VerbNet lookup -> role/sub_role

After (multilingual):
  Text -> LLM extraction (SPO + when + role + sub_role) -> VerbNet validation (English fallback)
```

The extraction prompt already requests structured JSON output. Adding `"role": "event|state|opinion"` and `"sub_role": "acquisition|maintenance|intent|hypothetical"` to the output schema costs zero additional inference — the LLM produces these fields in the same generation pass.

**VerbNet becomes a validation fallback for English text only.** When the input language is English, VerbNet cross-checks the LLM's role classification. Disagreements are logged but the LLM classification takes precedence (VerbNet has known gaps for phrasal verbs and domain-specific terminology).

**Model size constraint.** MASSIE finds that 7B+ parameter models are needed for reasonable multilingual IE performance. Foundry's choice of Qwen2.5-7B already meets this threshold. Models below 3B show catastrophic degradation on low-resource languages.

**Language performance tiers** (from MASSIE, informing deployment expectations):

| Tier | Languages | Expected Extraction Quality |
|------|-----------|---------------------------|
| High | English, Spanish, French, German, Chinese, Japanese | Comparable to English-only pipeline |
| Medium | Arabic, Hindi, Portuguese, Korean, Turkish | 10-20% degradation vs English |
| Low | Tibetan, Yoruba, Amharic, many low-resource languages | Significant degradation, requires evaluation |

---

### F.3 Adopted Patterns from Prior Art

The following patterns are adopted into the Foundry spec based on empirical validation in the source papers.

**1. Post-extraction entity clustering** (KGGen, Stanford — Mo et al., 2025)

Raw LLM extraction produces sparse, disconnected graphs with "nearly as many unique relation types as edges." Post-extraction clustering groups equivalent entities and normalizes relations, yielding denser, more connected graphs with better retrieval accuracy.

This aligns with the H3 tribunal verdict: entity resolution should be decoupled from extraction. Extraction produces raw mentions; a separate clustering pass resolves them to canonical entities. The clustering approach is superior to substring+embedding resolution (which H3 falsified at 100% false merge rate).

**Spec integration:** Add a post-extraction entity clustering stage between Stage 2 (Extraction) and Stage 3 (Entity Resolution). The clusterer uses LM similarity to group mentions referring to the same real-world entity, then selects a canonical name per cluster.

**2. Dialogue act filtering** (OKED, RPI — Rook, 2025)

Not all utterances assert facts. Questions, hedging, speculation, and social pleasantries contain no extractable knowledge. Extracting from these produces noise triples that contaminate the entity graph. Dialogue act classification filters non-factual utterances before extraction.

**Spec integration:** Add a lightweight dialogue act classifier (regex + heuristic, Tier 1 cost) before extraction. Utterances classified as questions, greetings, acknowledgments, or hedges are stored as raw chunks but skip triple extraction. This reduces extraction volume by an estimated 20-30% with no loss of factual coverage.

**3. Cross-turn assertion detection** (OKED, RPI — Rook, 2025)

Facts in conversation frequently span multiple turns. Example: "I got a new bike." / "What color?" / "Red." The fact [user, has, red bike] requires combining information from turns 1 and 3. Single-turn extraction misses these composite facts entirely.

**Spec integration:** The existing coreference resolution pass (Section 4.3.2) partially addresses this. Augment it with explicit cross-turn assertion detection: when a turn contains an answer to a question from a prior turn, fuse the question-answer pair into a single extraction unit before running the LLM extractor.

**4. LF1 partial-match metric** (MASSIE, ACL 2025 — Le et al.)

Exact-match SubEM severely overpenalizes extraction quality. A triple that captures the correct subject and predicate but uses a synonym for the object scores 0 under SubEM. LF1 is a structure-aware metric that awards partial credit for partially correct extractions.

**Spec integration:** Adopt LF1 as the primary extraction quality metric in Phase 5 evaluation. SubEM remains the end-to-end QA metric (it measures answer correctness, not extraction correctness). The distinction: SubEM evaluates the full pipeline; LF1 evaluates the extraction stage in isolation.

---

### F.4 Patterns Under Investigation

The following patterns are promising but require further validation before adoption.

**1. Sentence-level graphs** (SGMem, ICLR 2026 — Wu et al.)

SGMem represents dialogue as sentence-level graphs within chunked units, using only NLTK sentence segmentation — no LLM extraction needed. Edges encode co-occurrence within chunks and cross-chunk semantic similarity. Multi-hop retrieval traverses the sentence graph to collect coherent evidence.

**Relevance:** Lightweight, language-agnostic alternative to SPO extraction. Validates Foundry's raw-chunks-as-primary approach. **Risk:** Sentence-level granularity may be too coarse for aggregation queries ("how many bikes does the user own?") where individual facts must be counted.

**Investigation plan:** Build a sentence-graph index alongside the existing BM25 index. Compare retrieval Hit@10 on LME temporal and aggregation question types. If sentence-graph retrieval matches or exceeds BM25 on these types, consider as a Tier 1 pipeline.

**2. Prize-Collecting Steiner Tree retrieval** (Associa, ACL 2025 — Zhang et al.)

PCST is an optimization-based subgraph extraction method that finds the minimum-cost tree connecting prize nodes (relevant entities). It produces structured, evidence-rich subgraphs rather than flat ranked lists.

**Relevance:** Novel retrieval primitive that maps naturally to Foundry's Tier 1 architecture. The event-centric graph structure aligns with the TemporalEvent data model. The two-mode architecture (fast intuitive + slow deliberate) parallels Tier 1 + Tier 3.

**Investigation plan:** Implement PCST as a Tier 1 pipeline candidate. Evaluate on multi-hop questions where the answer requires connecting multiple entities through intermediate nodes. If PCST resolves >30% of currently-unreachable multi-hop questions, adopt as a standard Tier 1 pipeline.

---

### F.5 Competitive Analysis

| Dimension | Hermes Foundry | SGMem | Associa | PROM | KGGen |
|-----------|---------------|-------|---------|------|-------|
| **Extraction method** | Few-shot LLM (Qwen2.5-7B) | None (sentence graphs) | LLM event extraction | Few-shot LLM triples | LLM triples + clustering |
| **Multilingual support** | Yes (LLM-native, F.2) | Yes (NLTK sentence split) | Depends on LLM | Depends on LLM | Depends on LLM |
| **Entity resolution** | EntityStore BM25 (E.3) | N/A | PCST subgraph | Semantic similarity | Iterative clustering |
| **Temporal reasoning** | Allen interval algebra | No | Partial | No | No |
| **Query language** | ObjectSet (composable, lazy) | Graph traversal | PCST + refinement | SPARQL-like | None |
| **Architecture** | 3-tier (E.2) | Mixed retrieval | Two-mode (intuitive + deliberate) | Single pipeline | Single pipeline |
| **Always-on subsystems** | Yes (all Tier 1 parallel) | No (single path) | No (mode selection) | No | No |
| **Trace-to-pipeline learning** | Yes (Tier 3 -> Tier 1) | No | No | No | No |

**Foundry's unique advantages**, validated by both tribunal verdicts and competitive analysis:

1. **Allen interval algebra** — no competitor has formal temporal reasoning. SGMem, PROM, and KGGen have no temporal support. Associa has partial temporal awareness but no formal calculus.

2. **ObjectSet composable queries** — no competitor offers a lazy, composable query language. PROM uses SPARQL-like queries; others use flat retrieval. ObjectSet enables `foundry.entities("person").where("hobby", "$contains", "cycling").related("lives_in").execute()` — impossible in any competitor.

3. **3-tier architecture** — no competitor separates deterministic from agentic. SGMem uses a single retrieval path. Associa has two modes but both require LLM calls. Foundry's Tier 1+2 operate at zero LLM cost.

4. **Always-on subsystems** — no competitor runs all pipelines in parallel. Every other system requires routing or mode selection before processing. Foundry's Tier 1 runs everything, always, and selects the best answer post-hoc.

5. **Trace-to-pipeline learning** — no competitor converts agentic exploration into deterministic efficiency over time. This is the core of the user's Phase 3->4 vision: agentic traces become deterministic pipelines become distilled SLM routing.

---

*End of Appendix F.*
