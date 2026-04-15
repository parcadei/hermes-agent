# Deep Dive: Hindsight — Four Memory Networks for LLM Agents

Generated: 2026-04-13
Source: arXiv:2512.12818, GitHub vectorize-io/hindsight, VentureBeat, PRNewswire

---

## Architecture Diagram (Text)

```
                     New Information (Retain)
                              │
                              ▼
┌──────────────────────────────────────────────────────┐
│  TEMPR (Temporal Entity Memory Priming Retrieval)    │
│                                                      │
│  Retain Pipeline:                                    │
│  1. Extract narrative facts with temporal ranges     │
│  2. Entity extraction + resolution                   │
│  3. Build links (temporal, semantic, entity, causal) │
│  4. Store in appropriate network                     │
│                                                      │
│  ┌────────────┐ ┌────────────┐                       │
│  │ World      │ │ Experience │                       │
│  │ Network    │ │ Network    │                       │
│  │            │ │            │                       │
│  │ Objective  │ │ Agent's own│                       │
│  │ facts about│ │ experiences│                       │
│  │ the world  │ │ and actions│                       │
│  │            │ │            │                       │
│  │ "The stove │ │ "I touched │                       │
│  │  gets hot" │ │  the stove │                       │
│  │            │ │  and it    │                       │
│  │            │ │  hurt"     │                       │
│  └────────────┘ └────────────┘                       │
│                                                      │
│  Recall Pipeline (4-way parallel):                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │ Semantic  │ │ BM25     │ │ Graph    │ │Temporal│  │
│  │ Vector   │ │ Keyword  │ │ Entity   │ │Filter  │  │
│  │ Search   │ │ Search   │ │ Traversal│ │        │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘  │
│       └─────────────┼───────────┼────────────┘       │
│                     ▼                                │
│              ┌─────────────┐                         │
│              │ RRF Fusion  │                         │
│              └──────┬──────┘                         │
│                     ▼                                │
│              ┌─────────────┐                         │
│              │ Cross-Enc   │                         │
│              │ Reranker    │                         │
│              └─────────────┘                         │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼ Retrieved memories
┌──────────────────────────────────────────────────────┐
│  CARA (Coherent Adaptive Reasoning Agents)           │
│                                                      │
│  Reflect Pipeline:                                   │
│  1. Take retrieved facts + agent profile (Theta)     │
│  2. Apply disposition parameters:                    │
│     - Skepticism (question vs accept)                │
│     - Literalism (interpret precisely vs loosely)    │
│     - Empathy (consider others' perspectives)        │
│  3. Generate preference-conditioned response         │
│  4. Form/reinforce opinions → Opinion Network        │
│  5. Synthesize observations → Observation Network    │
│                                                      │
│  ┌────────────┐ ┌────────────┐                       │
│  │ Opinion    │ │ Observation│                       │
│  │ Network    │ │ Network    │                       │
│  │            │ │            │                       │
│  │ Subjective │ │ Preference-│                       │
│  │ beliefs    │ │ neutral    │                       │
│  │ with conf  │ │ entity     │                       │
│  │ scores     │ │ summaries  │                       │
│  │            │ │            │                       │
│  │ "I should  │ │ "Alice is  │                       │
│  │  not touch │ │  a software│                       │
│  │  the stove │ │  engineer  │                       │
│  │  again"    │ │  at Google"│                       │
│  │  conf=0.99 │ │            │                       │
│  └────────────┘ └────────────┘                       │
│                                                      │
│  Priority during reasoning:                          │
│  Mental Models → Observations → Raw Facts            │
└──────────────────────────────────────────────────────┘
```

---

## Data Model

### Memory Representation

Each memory in Hindsight is stored as a combination of:
- **Entities**: Canonical entities extracted from text, resolved across mentions
- **Relationships**: Typed links between entities
- **Time Series**: Temporal annotations (when events happened, validity ranges)
- **Vectors**: Both sparse (BM25) and dense (embedding) representations

### Four Memory Networks

| Network | Content | Epistemic Status | Update Mechanism |
|---------|---------|-----------------|------------------|
| **World** | Objective facts about external reality | Ground truth | New facts added, contradictions resolved |
| **Experience** | Agent's own actions and observations | Biographical | Append-only (experiences don't change) |
| **Opinion** | Subjective beliefs with confidence scores | Evolving beliefs | Confidence updated as new evidence arrives |
| **Observation** | Preference-neutral entity summaries | Synthesized | Regenerated during reflect operations |

### Link Types in the Memory Graph

| Link Type | Connects | Example |
|-----------|----------|---------|
| **Temporal** | Things close in time | "Meeting → Lunch" (same day) |
| **Semantic** | Things similar in meaning | "Python programmer" ↔ "Software developer" |
| **Entity** | Things referencing same entity | "Alice's job" → "Alice's hobby" (shared entity) |
| **Causal** | Cause → effect relationships | "Studied hard" → "Passed exam" |

### Memory Bank Abstraction

A Memory Bank is a dedicated memory space for a specific agent or context. Each bank maintains its own state across all four networks. This enables multi-agent or multi-user memory isolation.

---

## Query Patterns

### Three Core Operations

#### 1. Retain (Ingest)
```python
# Push new information into the memory system
hindsight.retain(
    text="Alice got promoted to Senior Engineer at Google in March 2024",
    source="conversation",
    timestamp="2024-03-15"
)
# Internally:
# 1. Extract facts with temporal ranges
# 2. Extract entities (Alice, Google, Senior Engineer)
# 3. Resolve entities against existing graph
# 4. Build links (temporal, semantic, entity, causal)
# 5. Route to World or Experience network based on perspective
```

#### 2. Recall (Retrieve)
```python
# Search memories using TEMPR's 4-way parallel retrieval
results = hindsight.recall(
    query="What is Alice's current job?",
    token_budget=1000
)
# Internally:
# 1. Semantic vector search (embedding similarity)
# 2. BM25 keyword search
# 3. Graph traversal through shared entities
# 4. Temporal filtering (time-constrained queries)
# 5. RRF fusion of all four result sets
# 6. Cross-encoder reranking for final precision
```

#### 3. Reflect (Reason + Learn)
```python
# Perform thorough analysis of existing memories
# Forms new connections, generates opinions and observations
hindsight.reflect(
    topic="Alice's career progression",
    agent_profile=AgentProfile(
        name="Assistant",
        disposition=Disposition(
            skepticism=0.3,
            literalism=0.7,
            empathy=0.5
        )
    )
)
# Internally:
# 1. Retrieve all relevant memories about the topic
# 2. Apply agent disposition parameters (skepticism, literalism, empathy)
# 3. Generate new observations (preference-neutral summaries)
# 4. Form or update opinions (subjective beliefs with confidence)
# 5. Persist new observations and opinions to their respective networks
```

### Reasoning Priority Order
During reasoning, Hindsight checks sources in priority order:
1. **Mental Models** (opinions with high confidence)
2. **Observations** (synthesized entity summaries)
3. **Raw Facts** (world and experience networks)

This means refined, high-confidence knowledge is preferred over raw, unprocessed facts.

---

## Key Code Patterns

### Source File Layout (from GitHub)

```
hindsight/
├── core/
│   ├── memory_bank.py          # MemoryBank: dedicated memory space per agent/context
│   ├── networks/
│   │   ├── world.py            # World Network: objective facts
│   │   ├── experience.py       # Experience Network: agent's actions
│   │   ├── opinion.py          # Opinion Network: beliefs + confidence
│   │   └── observation.py      # Observation Network: entity summaries
│   ├── tempr/
│   │   ├── retain.py           # Retain pipeline: extract → resolve → link → store
│   │   ├── recall.py           # Recall pipeline: 4-way search → RRF → rerank
│   │   └── entity_resolver.py  # Entity resolution
│   └── cara/
│       ├── reflect.py          # Reflect pipeline: reason → form opinions
│       └── disposition.py      # Disposition parameters (skepticism, etc.)
├── graph/
│   ├── entity.py               # Entity representation
│   ├── relationship.py         # Relationship types
│   └── temporal.py             # Temporal annotations and filtering
├── search/
│   ├── semantic.py             # Dense vector search
│   ├── keyword.py              # BM25 sparse search
│   ├── graph_search.py         # Graph traversal search
│   ├── temporal_filter.py      # Temporal constraint filtering
│   └── fusion.py               # RRF fusion + cross-encoder reranking
└── api/
    └── hindsight.py            # Top-level API: retain(), recall(), reflect()
```

### TEMPR Recall Pipeline (4-Way Parallel Search)

```python
# Pseudocode distilled from the paper and documentation

class TEMPR:
    async def recall(self, query: str, token_budget: int) -> List[Memory]:
        # 1. Run four searches in parallel
        semantic_results = await self.semantic_search(query)
        keyword_results = await self.bm25_search(query)
        graph_results = await self.graph_traversal(query)
        temporal_results = await self.temporal_filter(query)

        # 2. Fuse results via Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            graph_results,
            temporal_results
        )

        # 3. Rerank with cross-encoder for precision
        reranked = self.cross_encoder.rerank(query, fused)

        # 4. Trim to token budget
        return trim_to_budget(reranked, token_budget)
```

### Opinion Evolution

```python
# Pseudocode for opinion update mechanism

class OpinionNetwork:
    def update_opinion(self, topic: str, new_evidence: List[Memory],
                       agent_disposition: Disposition):
        existing_opinion = self.get_opinion(topic)

        if existing_opinion is None:
            # Form new opinion from evidence
            new_opinion = Opinion(
                belief=synthesize_belief(new_evidence, agent_disposition),
                confidence=compute_initial_confidence(new_evidence),
                evidence_ids=[m.id for m in new_evidence]
            )
            self.store(new_opinion)
        else:
            # Update existing opinion with new evidence
            updated_confidence = bayesian_update(
                existing_opinion.confidence,
                new_evidence,
                agent_disposition.skepticism  # How much to discount new evidence
            )
            existing_opinion.confidence = updated_confidence
            existing_opinion.evidence_ids.extend([m.id for m in new_evidence])
            self.store(existing_opinion)
```

### CARA Disposition Parameters

```python
@dataclass
class Disposition:
    skepticism: float    # 0.0 = gullible, 1.0 = highly skeptical
    literalism: float    # 0.0 = loose interpretation, 1.0 = precise/literal
    empathy: float       # 0.0 = self-focused, 1.0 = other-focused

# These parameters condition how the agent:
# - Evaluates evidence quality (skepticism)
# - Interprets language (literalism)
# - Considers perspectives (empathy)
```

---

## Limitations for Hermes Foundry

1. **Tightly coupled runtime**: Hindsight is designed as a self-contained memory service, not as a composable query library. The three operations (retain, recall, reflect) are the ONLY interface. No way to build custom queries or compose retrieval strategies.

2. **No composable query language**: Cannot express "all people Alice mentioned who work in tech AND were mentioned in the last month." Recall takes a natural language query string, not a structured query.

3. **No Allen interval algebra**: Temporal filtering is point-based (timestamps) or range-based, not interval-based. Cannot reason about temporal relations between events (overlaps, during, before, after).

4. **No lazy evaluation or query plans**: `recall()` executes immediately. No query plan inspection, optimization, or composition.

5. **No aggregation**: Cannot compute "count of times Alice mentioned cooking" or "most frequently discussed topics." Would require custom post-processing.

6. **Opaque opinion mechanism**: The opinion update logic is hidden inside CARA. No way to inspect, query, or manually adjust the opinion graph. Opinions are formed by the LLM during reflect, not by deterministic rules.

7. **LLM-dependent reflection**: The reflect operation requires LLM inference to form opinions and observations. No lightweight/local fallback for opinion formation.

8. **No set operations**: Cannot union/intersect/subtract memory result sets.

---

## Steal List (what to adopt)

### S1. Four-Network Epistemic Separation
The distinction between facts, experiences, opinions, and observations is the correct epistemic model for conversational memory.

**Adaptation for Hermes**:
- **World Network** → Our STATED layer (grounded facts from user's words)
- **Experience Network** → Agent-generated observations stored with `role=agent`
- **Opinion Network** → New: add a belief layer with confidence scores. MSL relations could carry `confidence: float` and `belief_type: fact|opinion|inference`
- **Observation Network** → Our DERIVED layer (gists, summaries, convergent facts)

This maps cleanly to our existing 3-layer architecture with the addition of an explicit belief/opinion dimension.

### S2. TEMPR 4-Way Parallel Retrieval
Four retrieval channels (semantic, keyword, graph, temporal) fused via RRF, then reranked. This is the state of the art for memory recall.

**Adaptation for Hermes**: We already have BM25 + cosine + entity graph traversal. Add explicit temporal filtering as a 4th channel. Use RRF for fusion (already implemented). Add cross-encoder reranking (already available via mxbai).

### S3. Link Types as First-Class Concepts
Temporal, semantic, entity, and causal links between memories are distinct link types with different semantics. This is richer than our current single-type edge model.

**Adaptation for Hermes**: In the EntityGraph, add typed edges:
```python
class EdgeType(Enum):
    TEMPORAL = "temporal"    # Close in time
    SEMANTIC = "semantic"    # Similar in meaning
    ENTITY = "entity"        # Share a canonical entity
    CAUSAL = "causal"        # Cause → effect
    SUPERSEDES = "supersedes"  # New fact replaces old (already have this)
```

### S4. Reflect as First-Class Operation
The idea that memory is not just store/retrieve but also REFLECT — periodically synthesizing new knowledge from existing memories — is powerful. This is what our "dream" operation was trying to achieve, but Hindsight does it more selectively (per-topic, on-demand rather than batch).

**Adaptation for Hermes**: Add a `.reflect(topic)` operation to the MemoryQuery API that:
1. Retrieves all facts about a topic
2. Synthesizes an entity summary (observation)
3. Optionally forms beliefs with confidence scores
4. Stores results in the DERIVED layer

### S5. Disposition-Parameterized Reasoning
The idea that an agent's personality (skepticism, literalism, empathy) affects how it interprets and weighs evidence is a clean way to handle agent identity.

**Adaptation for Hermes**: Add a `ReasoningProfile` to the agent configuration that parameterizes how retrieved memories are weighted and synthesized. This could control:
- How aggressively to apply supersession (skepticism)
- How literally to match entity names (literalism)
- How to weigh conflicting evidence (confidence update rate)

### S6. Memory Bank Per Agent/User
Each agent or user gets an isolated memory space. This enables multi-tenant operation without cross-contamination.

**Adaptation for Hermes**: Already partially implemented via session scoping. Formalize as a MemoryBank abstraction that wraps a complete memory store (all four layers + indices) and is instantiated per user/agent.

---

## Anti-Patterns (what to avoid)

### A1. Opaque LLM-Driven Opinion Formation
Hindsight's opinion formation is entirely LLM-driven during reflect. The opinion content and confidence scores are LLM outputs, not deterministic computations. This makes opinions unpredictable and hard to debug.

**Instead**: Use deterministic rules for opinion formation where possible (e.g., if the same fact appears N times with no contradictions, confidence = f(N)). Reserve LLM for synthesizing the opinion TEXT, not for deciding confidence scores.

### A2. Natural Language Query Interface Only
Recall only accepts natural language queries. This forces every structured retrieval need through the LLM's interpretation of the query string. For programmatic access, this is a bottleneck.

**Instead**: Expose both natural language (for agents) and structured query (for programmatic access) interfaces. The MemoryQuery fluent API serves the structured case.

### A3. No Query Composition
The three operations (retain, recall, reflect) are atomic. You cannot compose a recall with a filter, then an aggregation, then another recall. Each is a standalone call.

**Instead**: Make recall one STEP in a composable query pipeline, not the entire pipeline.

---

## Key Metrics

| Metric | Value |
|--------|-------|
| LongMemEval accuracy | 91.4% (SOTA) |
| LoCoMo accuracy | 89.61% (vs 75.78% strongest prior) |
| Multi-session improvement | 21.1% -> 79.7% (+58.6pp) |
| Temporal reasoning improvement | 31.6% -> 79.7% (+48.1pp) |
| Knowledge update improvement | 60.3% -> 84.6% (+24.3pp) |
| vs Full-context GPT-4o | 91.4% vs 60.2% (+31.2pp) |
| vs Zep + GPT-4o | 91.4% vs 71.2% (+20.2pp) |

The most impressive result: **multi-session questions improved from 21.1% to 79.7%**. This is exactly the problem space Hermes targets.

---

## Sources

1. [Hindsight Paper (arXiv:2512.12818)](https://arxiv.org/abs/2512.12818) — Full architecture paper
2. [Hindsight HTML Paper](https://arxiv.org/html/2512.12818v1) — Detailed version with figures
3. [Hindsight GitHub](https://github.com/vectorize-io/hindsight) — Source code
4. [Hindsight Documentation](https://hindsight.vectorize.io/) — API docs and concepts
5. [VentureBeat Coverage](https://venturebeat.com/data/with-91-accuracy-open-source-hindsight-agentic-memory-provides-20-20-vision) — Architecture analysis
6. [PRNewswire Announcement](https://www.prnewswire.com/news-releases/vectorize-breaks-90-on-longmemeval-with-open-source-ai-agent-memory-system-302643146.html) — Benchmark results
7. [EmergentMind Analysis](https://www.emergentmind.com/papers/2512.12818) — Paper analysis
8. [Hindsight Cookbook](https://github.com/vectorize-io/hindsight-cookbook) — Usage examples
