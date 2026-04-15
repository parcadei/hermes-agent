# Deep Dive: Django QuerySet — Lazy Evaluation & Clone-on-Write Patterns

Generated: 2026-04-13
Source: Django source code (django/db/models/query.py), Django documentation, DeepWiki

---

## Architecture Diagram (Text)

```
User Code
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  QuerySet Layer (Lazy, Chainable API)                │
│                                                      │
│  .filter()  .exclude()  .annotate()  .order_by()     │
│  .values()  .distinct()  .union()  .intersection()   │
│                                                      │
│  Each method calls _chain() → returns NEW QuerySet   │
│  Original is NEVER mutated                           │
│  No SQL until terminal operation                     │
│                                                      │
│  Terminal ops: __iter__, __len__, __bool__, list(),   │
│                __getitem__ (with step), repr()        │
└────────────────────────┬─────────────────────────────┘
                         │
                    _chain()
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  Query Object (Accumulates State)                    │
│  django/db/models/sql/query.py                       │
│                                                      │
│  .where (WhereNode tree — AND/OR/NOT)                │
│  .annotations (dict of Expression objects)           │
│  .alias_map (JOIN tracking)                          │
│  .select (columns to return)                         │
│  .group_by (GROUP BY tracking)                       │
│  .order_by (ORDER BY tracking)                       │
│                                                      │
│  Mutated only on cloned copies                       │
└────────────────────────┬─────────────────────────────┘
                         │
                    compile()
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  Expression System (Composable Python Objects)       │
│  django/db/models/expressions.py                     │
│                                                      │
│  F("field_name")  — Column reference                 │
│  Q(field=val)     — Boolean clause (AND/OR/NOT)      │
│  Value(42)        — Literal value                    │
│  Func(...)        — Database function (LOWER, etc)   │
│  Aggregate(...)   — SUM, COUNT, AVG, etc             │
│                                                      │
│  All implement resolve_expression() → copy + validate│
└────────────────────────┬─────────────────────────────┘
                         │
                    as_sql()
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  SQL Compiler (Backend-Specific)                     │
│  django/db/models/sql/compiler.py                    │
│                                                      │
│  Translates Query object → SQL string + params       │
│  Backend-specific (PostgreSQL, MySQL, SQLite, etc)   │
│  Handles quoting, type casting, dialect differences  │
└────────────────────────┬─────────────────────────────┘
                         │
                    execute()
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  Result Processing                                   │
│                                                      │
│  _fetch_all() → list(self._iterable_class(self))     │
│  Results cached in _result_cache                     │
│  Prefetch related objects if needed                  │
│  Subsequent access uses cache, no re-query           │
└──────────────────────────────────────────────────────┘
```

---

## Data Model

Not applicable — Django QuerySets operate over arbitrary model instances. The relevant abstraction is the **Query state machine** that accumulates filter/annotation/ordering state until a terminal operation triggers compilation.

Key state tracked in Query object:
- `where`: WhereNode tree (boolean expression tree of filter conditions)
- `annotations`: dict mapping names to Expression objects
- `alias_map`: dict mapping table aliases to join information
- `select`: list of columns/expressions to include in SELECT
- `group_by`: list for GROUP BY
- `order_by`: list of ordering expressions
- `distinct`: bool or list of fields
- `limits`: (low, high) for LIMIT/OFFSET

---

## Query Patterns — The Core API We Want to Steal

### Pattern 1: Clone-on-Write via `_chain()`

```python
# From django/db/models/query.py (simplified)

def _chain(self):
    """
    Return a copy of the current QuerySet. Lightweight alternative to deepcopy().
    """
    c = self.__class__(
        model=self.model,
        query=self.query.chain(),     # Query.chain() creates a shallow copy
        using=self._db,
        hints=self._hints,
    )
    c._sticky_filter = self._sticky_filter
    c._for_write = self._for_write
    c._prefetch_related_lookups = self._prefetch_related_lookups[:]  # slice copy
    c._known_related_objects = self._known_related_objects
    c._iterable_class = self._iterable_class
    c._fields = self._fields
    return c
```

Key insight: `self.query.chain()` creates a SHALLOW copy of the Query object. This is cheap — no deep copy of the WhereNode tree or annotations. The Query object itself uses copy-on-write semantics internally.

### Pattern 2: Filter/Exclude as WhereNode Tree Building

```python
# Simplified from django/db/models/query.py

def filter(self, *args, **kwargs):
    return self._filter_or_exclude(False, args, kwargs)

def exclude(self, *args, **kwargs):
    return self._filter_or_exclude(True, args, kwargs)

def _filter_or_exclude(self, negate, args, kwargs):
    clone = self._chain()          # Step 1: Clone
    # Build Q object from kwargs
    if args:
        # args are Q objects for complex queries
        clause = reduce(operator.and_, args)
    if kwargs:
        clause &= Q(**kwargs)
    if negate:
        clause = ~clause            # NOT wrapper
    clone.query.add_q(clause)       # Step 2: Add to WhereNode tree
    return clone                    # Step 3: Return new QuerySet
```

### Pattern 3: Q Objects for Boolean Composition

```python
# Q objects support AND (&), OR (|), NOT (~)
from django.db.models import Q

# Complex filter: (name="Alice" AND age>30) OR (city="NYC")
qs = MyModel.objects.filter(
    Q(name="Alice", age__gt=30) | Q(city="NYC")
)

# Negation
qs = MyModel.objects.filter(~Q(status="inactive"))
```

Q objects compose into a tree structure stored in `Query.where` (a WhereNode). Each WhereNode has a `connector` (AND/OR) and `children` (list of Lookup or WhereNode objects). Negation wraps in a `NOT` node.

### Pattern 4: Lazy Evaluation — Terminal Operations

```python
# NOTHING happens until one of these triggers _fetch_all()

qs = MyModel.objects.filter(name="Alice").exclude(age__lt=18)
# No SQL yet

# Terminal operations that trigger evaluation:
for obj in qs:        # __iter__
    ...

len(qs)               # __len__
bool(qs)              # __bool__
list(qs)              # __iter__ via list()
qs[0]                 # __getitem__ (simple index)
qs[::2]              # __getitem__ with step (forces eval)
repr(qs)              # __repr__
```

Internally, all terminals call `_fetch_all()`:

```python
def _fetch_all(self):
    if self._result_cache is None:
        self._result_cache = list(self._iterable_class(self))
    if self._prefetch_related_lookups and not self._prefetch_done:
        self._prefetch_related_objects()
```

### Pattern 5: Annotate (Per-Object Computed Attributes)

```python
from django.db.models import Count, F, Value
from django.db.models.functions import Concat

# Add computed fields to each object
qs = Author.objects.annotate(
    book_count=Count('book'),
    full_name=Concat(F('first_name'), Value(' '), F('last_name'))
)
# Each author now has .book_count and .full_name
```

Annotations add to `Query.annotations` dict. They become part of the SELECT clause and are available as attributes on returned objects.

### Pattern 6: Aggregate (Whole-Set Summary)

```python
from django.db.models import Avg, Sum, Max

# Returns a dict, not a QuerySet
result = Book.objects.aggregate(
    avg_price=Avg('price'),
    total_pages=Sum('pages'),
    max_rating=Max('rating')
)
# result = {'avg_price': 29.99, 'total_pages': 50000, 'max_rating': 5.0}
```

`aggregate()` is a TERMINAL operation — it immediately executes and returns a dict. It adds aggregate expressions to `Query.annotations` AND triggers immediate compilation/execution.

### Pattern 7: F Expressions (Column References)

```python
from django.db.models import F

# Reference other columns in expressions
qs = Entry.objects.filter(rating__gt=F('number_of_comments'))
Entry.objects.update(rating=F('rating') + 1)
```

F objects implement `resolve_expression()` which resolves field names to actual database columns during compilation. They compose with arithmetic operators.

### Pattern 8: Expression Compilation Pipeline

Every Expression implements:
```python
class Expression:
    def resolve_expression(self, query, allow_joins, reuse, summarize, for_save):
        """
        Pre-processing before adding to query.
        Returns copy of self with resolved field references.
        """
        c = self.copy()
        # Resolve nested expressions
        for pos, child in enumerate(c.get_source_expressions()):
            c.set_source_expression(pos, child.resolve_expression(...))
        return c

    def as_sql(self, compiler, connection):
        """
        Compile to backend-specific SQL fragment.
        Returns (sql_string, params_list)
        """
        ...
```

---

## Key Code Patterns Worth Studying

### File: `django/db/models/query.py`
- `_chain()` (line ~432) — The clone-on-write core
- `_fetch_all()` (line ~1400) — Lazy evaluation trigger
- `_filter_or_exclude()` (line ~1150) — Filter/exclude unified implementation
- `filter()`, `exclude()` — Public API wrapping `_filter_or_exclude`
- `annotate()` — Adds Expression objects to Query.annotations
- `aggregate()` — Terminal: compiles + executes + returns dict
- `union()`, `intersection()`, `difference()` — Set operations (delegated to SQL UNION/INTERSECT/EXCEPT)

### File: `django/db/models/sql/query.py`
- `Query.chain()` — Shallow copy of Query state
- `Query.add_q()` — Add Q object to WhereNode tree
- `Query.setup_joins()` — Traverse field relationships to build JOIN chain
- `Query.where` — WhereNode tree (the expression AST)

### File: `django/db/models/expressions.py`
- `F` — Column reference expression
- `Q` — Boolean clause expression (actually in `django/db/models/query_utils.py`)
- `Value` — Literal value wrapper
- `Func` — Database function wrapper
- `Aggregate` — Base for SUM, COUNT, AVG, etc.
- `resolve_expression()` — The compilation hook every expression implements

---

## Limitations for Hermes Foundry

1. **SQL-only backend**: The entire pipeline assumes SQL. No graph traversal. No graph adjacency operations.

2. **No graph link traversal**: Cannot express "follow relationship R from entity set A to get entity set B." Django's JOINs traverse foreign keys but only along predefined model relationships, not arbitrary graph edges.

3. **No temporal interval reasoning**: No built-in support for Allen interval algebra or temporal validity windows. Dates are just column values to compare.

4. **Set operations are SQL-level**: `union()`, `intersection()`, `difference()` compile to SQL UNION/INTERSECT/EXCEPT. Cannot be combined freely with other QuerySet methods (Django docs warn about this).

5. **No implicit graph**: Every model and relationship must be predefined in the schema. Cannot handle dynamically-typed entities or learned ontology edges.

---

## Steal List (what to adopt)

### S1. Clone-on-Write via `_chain()`
The CORE pattern for our MemoryQuery class. Every method returns a new MemoryQuery via `_chain()`. Original never mutated. Cheap shallow copies.

```python
# Hermes adaptation
class MemoryQuery:
    def _chain(self):
        clone = MemoryQuery.__new__(MemoryQuery)
        clone._filters = self._filters[:]      # shallow copy of filter list
        clone._traversals = self._traversals[:] # shallow copy of traversal list
        clone._temporal = self._temporal.copy()  # copy temporal constraints
        clone._aggregations = self._aggregations[:]
        clone._result_cache = None              # new query, no cache
        clone._store = self._store              # shared reference to storage
        return clone
```

### S2. Terminal Operation Detection
Implement `__iter__`, `__len__`, `__bool__` as evaluation triggers. Add `.resolve()` as explicit terminal. Use `_fetch_all()` pattern with `_result_cache`.

```python
class MemoryQuery:
    def __iter__(self):
        self._fetch_all()
        return iter(self._result_cache)

    def __len__(self):
        self._fetch_all()
        return len(self._result_cache)

    def __bool__(self):
        self._fetch_all()
        return bool(self._result_cache)

    def _fetch_all(self):
        if self._result_cache is None:
            plan = self._compile()          # Build execution plan from accumulated state
            self._result_cache = plan.execute(self._store)
```

### S3. Q-Object-Like Boolean Composition
Create MemoryQ objects that support `&`, `|`, `~` for composing complex filter predicates.

```python
# Hermes adaptation
class MemoryQ:
    AND = 'AND'
    OR = 'OR'

    def __init__(self, **kwargs):
        self.children = list(kwargs.items())
        self.connector = self.AND
        self.negated = False

    def __and__(self, other):
        return self._combine(other, self.AND)

    def __or__(self, other):
        return self._combine(other, self.OR)

    def __invert__(self):
        clone = copy.deepcopy(self)
        clone.negated = not clone.negated
        return clone
```

### S4. Expression System with `resolve_expression()`
Build an expression tree where each node knows how to resolve itself against the storage backend. This enables backend-agnostic query compilation (in-memory, Kuzu, Neo4j).

### S5. Annotate/Aggregate Split
`annotate()` adds per-entity computed attributes (lazy). `aggregate()` computes whole-set summaries (terminal). This is the exact split we need.

### S6. Result Caching
Once evaluated, cache results. Subsequent access to the same QuerySet uses the cache. A new `_chain()` resets the cache.

---

## Anti-Patterns (what to avoid)

### A1. SQL Compilation Assumption
Django's entire expression system assumes SQL output. We need expression nodes that compile to different backends (in-memory Python, Cypher, SPARQL). Do not hardcode any backend.

### A2. Set Operations as SQL-Level Delegation
Django's `union()`/`intersection()`/`difference()` are limited because they delegate to SQL set operations, which cannot be freely composed with other QuerySet methods. Our set operations should be first-class plan nodes that compose with everything.

### A3. Deep Join Tracking
Django tracks JOINs via `alias_map` because SQL requires explicit JOINs. In a graph/memory context, "joins" are link traversals and should be implicit in the query plan, not tracked as SQL aliases.

---

## Sources

1. [Django QuerySet source (query.py)](https://github.com/django/django/blob/main/django/db/models/query.py) — Main QuerySet implementation
2. [Django QuerySet API Reference](https://docs.djangoproject.com/en/5.0/ref/models/querysets/) — Official docs
3. [DeepWiki: QuerySets and SQL Generation](https://deepwiki.com/django/django/2.1-models-fields-and-querysets) — Architecture analysis
4. [Django Query Expressions](https://django.readthedocs.io/en/stable/ref/models/expressions.html) — F, Q, Value, Func, Aggregate docs
5. [Django Ticket #20880](https://code.djangoproject.com/ticket/20880) — `_clone()` optimization discussion
6. [Django Ticket #16759](https://code.djangoproject.com/ticket/16759) — Expensive Query cloning discussion
