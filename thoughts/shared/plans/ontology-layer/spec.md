# Ontology Layer for Conversation Memory

## Problem

The subject_index stores ground-level entities ("10 neon tetras", "20-gallon tank")
but queries use category-level terms ("fish", "aquariums"). Substring matching can't
bridge this vocabulary gap. KG hop returned ZERO matches on the fish question
(Q1, answer=17) despite correct triple extraction.

```
Query:   "How many fish are there in total in both of my aquariums?"
Index:   "20-gallon tank", "10 neon tetras", "betta fish", "pleco catfish"
Match:   NONE — no entity name appears as substring of query
```

## Prior Art

| System | Approach | Cost |
|--------|----------|------|
| Palantir Foundry | Pre-defined ontology (Object Types, Links, Interfaces). Humans define the schema, data maps into it. Three layers: data model, logic, governance. | Human-designed |
| Zep/Graphiti | LLM classifies entity nodes at ingestion. Temporal KG nodes have `entity_type` fields. | Separate LLM call per entity |
| MemInsight (AWS, EMNLP 2025) | "Attribute Mining" — autonomous LLM pass generates semantic attributes per memory. | Heavy LLM agent loop |
| A-MEM (Zettelkasten) | Each memory node stores "tags" and "related concepts" via LLM. | LLM call at creation |
| Memoria (arXiv 2512.12686) | Weighted KG with explicit trait/preference/behavior categories at extraction. | LLM at ingestion |
| OntoType (UIUC, KDD 2024) | Zero-shot PLM typing: masked LM "X is a [MASK]" + ontology walk. No generation. | PLM forward pass only |
| LangChain EntityMemory | Named entity extraction + running summaries per entity. No hierarchy. | LLM per turn |

**Key finding:** Everyone uses LLM for entity typing. Difference is separate call vs piggybacked.
Our approach (below) is strictly cheaper than all of the above — one call after flush,
not per-entity or per-turn.

## Design: Palantir-Inspired Ontology Generation

Instead of typing entities one at a time during extraction (bottom-up, inconsistent),
we build the ontology TOP-DOWN in a single LLM pass after triple extraction completes.

### Why top-down beats bottom-up

| | Bottom-up (per-triple typing) | Top-down (ontology pass) |
|---|---|---|
| Consistency | LLM might say "fish" once, "aquatic animal" next | Sees ALL entities, normalizes globally |
| Hierarchy | One level only (entity → type) | Multi-level (neon tetras → fish → pets) |
| Cost | +2 fields per triple (~7500 extra tokens/flush) | ONE call (~3000 tokens total) |
| Implementation | Schema change + prompt change + parse change | One new function after flush |

### Architecture

```
_flush_extraction()              existing — produces ~1500 triples
     │
     ▼
_build_ontology()                NEW — one LLM call
     │
     │  Input:  unique entity names from triples
     │          ["20-gallon tank", "10 neon tetras", "Java Moss",
     │           "Bubbles", "lemon poppyseed cake", ...]
     │
     │  Prompt: "Organize these entities into a category tree.
     │           Return JSON: {category: [entity1, entity2, ...]}.
     │           Use short, common category words (fish, plant, recipe).
     │           An entity can appear in multiple categories.
     │           Only create categories with 2+ members."
     │
     │  Output: {"fish": ["neon tetras", "gouramis", "pleco", "betta"],
     │           "aquarium": ["20-gallon tank", "10-gallon tank"],
     │           "plant": ["java moss", "anacharis"],
     │           "pet": ["Bubbles"],
     │           "recipe": ["lemon poppyseed cake"], ...}
     │
     ▼
_ontology_index                  dict[category → list[stored_text]]
```

### Palantir mapping

| Palantir Concept | Hermes Equivalent |
|------------------|-------------------|
| Object Type (e.g. "Sensor", "Employee") | Category (e.g. "fish", "aquarium") |
| Object Instance | Extracted entity (e.g. "10 neon tetras") |
| Properties | Triple predicates (e.g. "contains", "name") |
| Links | Triple relationships between entities |
| Object Set (query result) | Category lookup result |

Palantir's ontology is human-designed for enterprise data. Ours is LLM-generated
from conversation content — same structural role, dynamic instead of static.

### Query resolution

```python
def _resolve_kg_hop(self, query: str) -> list[str]:
    # 1. Existing: substring entity match (specific)
    #    "20-gallon tank" in query → direct hit
    results = self._substring_entity_match(query)

    # 2. NEW: ontology category match (abstract)
    query_words = set(self._stem_words(sq_lower.split()))
    for cat, entries in self._ontology_index.items():
        cat_stemmed = self._stem(cat)
        if cat_stemmed in query_words:
            for stored_text in entries:
                results.append(stored_text)

    return deduplicate(results)
```

### Stemming (plural handling)

```python
def _stem(word: str) -> str:
    """Strip common English suffixes for category matching."""
    for suffix in ("ies", "es", "s", "ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[:-len(suffix)]
            if suffix == "ies":
                stem += "y"  # "aquaries" edge case won't happen, but "stories"→"story"
            return stem
    return word
```

"aquariums" → "aquarium", "fishes" → "fish", "recipes" → "recipe"

### Data structures

```python
# Built by _build_ontology() after flush
_ontology_index: dict[str, list[str]] = {}
# category_lower → [stored_text_1, stored_text_2, ...]

# Mapping from entity name to its stored text (built during flush)
_entity_to_stored: dict[str, list[str]] = {}
# entity_name_lower → [stored_text_1, ...]
```

### LLM prompt for ontology generation

```
Given these entity names extracted from a conversation, organize them
into a flat category map. Rules:

- Each category is a SHORT, COMMON word (1-2 words max)
- An entity can belong to multiple categories
- Only create categories with 2+ members
- Use the most natural everyday word for the category
  (e.g. "fish" not "aquatic organisms", "food" not "culinary items")

Entities:
{entity_list}

Return JSON: {"category_name": ["entity1", "entity2", ...], ...}
```

### Structured output schema

```python
ONTOLOGY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ontology_categories",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "required": ["categories"],
            "additionalProperties": False
        }
    }
}
```

Note: vLLM structured output with `additionalProperties` as a schema (not False)
may need testing. Fallback: use unstructured generation + JSON parsing.

### Cache compatibility

Add to `eval_cache.py`:

```python
# save
"ontology_index": dict(getattr(agent, "_ontology_index", {})),
"entity_to_stored": dict(getattr(agent, "_entity_to_stored", {})),

# load
agent._ontology_index = state.get("ontology_index", {})
agent._entity_to_stored = state.get("entity_to_stored", {})
```

### Token cost

- Unique entities per context: ~200-300 (from ~1500 triples, many share entities)
- Input: ~600 tokens (entity list) + ~200 tokens (prompt) = ~800 tokens
- Output: ~400 tokens (category map JSON)
- Total: ~1200 tokens, ONE call per context flush
- Wall time: ~1-2s on vLLM Qwen3-30B
- Dollar cost: $0.00 (local inference)

### Files changed

| File | Change |
|------|--------|
| `hermes_memory_agent.py` | `_build_ontology()`, `_ontology_index`, `_entity_to_stored`, updated `_resolve_kg_hop()`, `reset()` |
| `eval_cache.py` | Save/load `_ontology_index` and `_entity_to_stored` |
| `tests/test_conv_kg_hop.py` | Ontology category matching tests |

Note: `llm_triple_extractor.py` is UNCHANGED — no schema modification needed.

### Risks

1. **vLLM structured output for dynamic keys**: The `additionalProperties` schema
   allows arbitrary category names as keys. vLLM may not support this in strict mode.
   Fallback: use unstructured generation + robust JSON parsing (same as triple extractor).

2. **Overly broad categories**: LLM might create "thing" or "item". Mitigated by
   the "2+ members" and "common word" rules in the prompt, plus a stoplist filter.

3. **Inconsistent entity names**: The ontology maps entity names back to stored texts.
   Entity names from triples may not exactly match stored text substrings (e.g.,
   "neon tetras" vs "10 neon tetras"). Use substring matching for the entity→stored_text
   lookup, not exact match.

4. **Categories that conflict with real entities**: If a category name "fish" is also
   an entity name (e.g., someone named Fish), both paths trigger — category returns
   all fish entities, substring returns the specific person. The dedup layer handles this.

### Future: Local 4B Model

Once the ontology pipeline is validated, the extraction + ontology generation can be
distilled into a fine-tuned 4B model (Qwen2.5-4B or Phi-3-mini) that runs on consumer
hardware (RTX 4090, 24GB VRAM). Training data: use Qwen3-30B outputs as teacher labels.
This eliminates the need for rented GPU time entirely.

### Example walkthrough

```
Ingestion: 739 chunks → 942 turns → flush → 1500 triples

Unique entities extracted:
  "20-gallon tank", "10-gallon tank", "10 neon tetras",
  "5 golden honey gouramis", "pleco catfish", "betta fish",
  "Bubbles", "Java Moss", "Anacharis", "treasure chest",
  "lemon poppyseed cake", "Tokyo", "Kyoto", ...

_build_ontology() LLM call → returns:
  {
    "fish": ["10 neon tetras", "5 golden honey gouramis",
             "pleco catfish", "betta fish"],
    "aquarium": ["20-gallon tank", "10-gallon tank"],
    "plant": ["Java Moss", "Anacharis"],
    "city": ["Tokyo", "Kyoto"],
    "decoration": ["treasure chest"],
    "recipe": ["lemon poppyseed cake"],
    "pet": ["Bubbles", "betta fish"]
  }

_ontology_index built:
  "fish" → [stored_text for 20-gal tank convo, stored_text for betta convo, ...]
  "aquarium" → [stored_text for 20-gal setup, stored_text for 10-gal upgrade, ...]

Query: "How many fish are there in total in both of my aquariums?"
  query_words_stemmed = {"how","mani","fish","are","there","in","total","both","of","my","aquarium"}

  Category match "fish" → returns 4 stored texts (all fish mentions)
  Category match "aquarium" → returns 2 stored texts (both tank mentions)

  GPT-4o-mini sees ALL fish facts in context → answers "17" ✓
```
