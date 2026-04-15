# Deep Dive: KAG / OpenSPG — Knowledge-Augmented Generation

Generated: 2026-04-13
Source: arXiv:2409.13731, GitHub OpenSPG/KAG, v0.7/v0.8 release notes

---

## Architecture Diagram (Text)

```
                    User Question
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  KAG-Solver                                          │
│                                                      │
│  ┌─────────────────────────────────────────────┐     │
│  │ LF Planner (lf_planner_abc.py)              │     │
│  │                                              │     │
│  │ Decomposes question → Logical Form DAG       │     │
│  │ Each node = function(args, constraints)       │     │
│  │ Dependencies between nodes via shared vars    │     │
│  │                                              │     │
│  │ Two modes:                                   │     │
│  │  - Static Planner: one-shot decomposition    │     │
│  │  - Iterative Planner: refine until satisfied │     │
│  └─────────────────┬───────────────────────────┘     │
│                    │                                  │
│         DAG of Logical Form Nodes                    │
│                    │                                  │
│                    ▼                                  │
│  ┌─────────────────────────────────────────────┐     │
│  │ Executor Pipeline (default_lf_executor.py)   │     │
│  │                                              │     │
│  │ For each LF node, dispatch to executor:      │     │
│  │                                              │     │
│  │ ┌──────────────┐ ┌──────────────┐            │     │
│  │ │ kag_hybrid   │ │ math         │            │     │
│  │ │ _executor    │ │ _executor    │            │     │
│  │ │              │ │              │            │     │
│  │ │ 3-layer      │ │ Numeric      │            │     │
│  │ │ knowledge    │ │ computation  │            │     │
│  │ │ retrieval    │ │              │            │     │
│  │ └──────────────┘ └──────────────┘            │     │
│  │ ┌──────────────┐ ┌──────────────┐            │     │
│  │ │ cypher       │ │ output       │            │     │
│  │ │ _executor    │ │ _executor    │            │     │
│  │ │              │ │              │            │     │
│  │ │ Graph DB     │ │ Final answer │            │     │
│  │ │ queries      │ │ generation   │            │     │
│  │ └──────────────┘ └──────────────┘            │     │
│  └─────────────────┬───────────────────────────┘     │
│                    │                                  │
│                    ▼                                  │
│  ┌─────────────────────────────────────────────┐     │
│  │ Reasoner (kag_reasoner_abc.py)               │     │
│  │                                              │     │
│  │ Evaluates executor results                   │     │
│  │ If insufficient → Reflector → re-plan        │     │
│  │ If sufficient → Generator → answer           │     │
│  └─────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  KAG-Builder (Offline Indexing)                      │
│                                                      │
│  Text → LLM Extraction → SPG Triples                 │
│                                                      │
│  Mutual Index:                                       │
│  KG nodes ←→ Original text chunks                    │
│  (every fact links back to source text)              │
│                                                      │
│  Knowledge Representation:                           │
│  - Schema-Constrained: domain-specific types         │
│  - Schema-Free: open-ended extraction                │
│  - Chunk Index: raw text retrieval                   │
└──────────────────────────────────────────────────────┘
```

---

## Data Model

### Knowledge Representation (SPG = Semantic-enhanced Programmable Graph)

KAG uses OpenSPG's graph model:
- **Entity Types**: Schema-defined (Person, Organization, Event) or schema-free
- **Concept Types**: Hierarchical categories (taxonomy)
- **Events**: Typed events with temporal annotations
- **Relations**: SPO triples (Subject-Predicate-Object) with type constraints

### Mutual Indexing Structure

The key innovation: **bidirectional links between KG structure and text chunks**.

```
KG Triple: (Alice, worksAt, TechCorp)
    │                              │
    ├── chunk_ids: [chunk_42]      │
    │                              │
    ▼                              ▼
Text Chunk #42: "Alice has been working at TechCorp since 2020..."
    │
    └── triple_ids: [(Alice, worksAt, TechCorp)]
```

This means every KG fact can be traced to its source text, and every text chunk knows which KG facts were extracted from it.

### Three Knowledge Layers

| Layer | Description | Precision | Source |
|-------|-------------|-----------|--------|
| Schema-Constrained | Domain-specific triples with typed entities and relations | Highest | OpenSPG schema + LLM extraction |
| Schema-Free | Open-ended triples without predefined types | Medium | LLM open information extraction |
| Raw Context (Chunks) | Original text chunks with embeddings | Baseline | Chunked source documents |

The solver queries these layers sequentially: schema-constrained first (most precise), then schema-free, then raw chunks as fallback.

---

## Query Patterns — The Logical Form Language

### Pattern 1: Logical Form Decomposition

Complex questions are decomposed into a DAG of logical form functions:

```
Question: "What is the total compensation for employees in the
           marketing department who joined after 2020?"

Logical Form DAG:
  Step 1: Retrieval(s=Employee, p=department, o=marketing)
             → var: $employees
  Step 2: Retrieval(s=$employees, p=joinDate, o=?, constraint=">2020")
             → var: $filtered_employees
  Step 3: Retrieval(s=$filtered_employees, p=compensation, o=?)
             → var: $compensations
  Step 4: Math(sum($compensations))
             → var: $total
  Step 5: Output($total)
```

Each step references previous step results via shared variable names. Dependencies form a DAG, enabling parallel execution of independent steps.

### Pattern 2: SPO Retrieval Functions

The core retrieval primitive operates on Subject-Predicate-Object triples:

```python
# Pseudocode of retrieval logical form
Retrieval(
    subject="Alice",           # Entity or variable
    predicate="worksAt",       # Relation type
    object=None,               # Unknown (to retrieve)
    constraints=["since > 2020"],
    knowledge_layer="schema_constrained"  # Which layer to query
)
```

Rules:
- S, P, O should not repeat in the same expression (use variables for multi-hop)
- When referencing a previous variable, names must be consistent
- For multi-hop queries, chain multiple Retrieval functions

### Pattern 3: Three Operator Types

| Operator | Purpose | Examples |
|----------|---------|----------|
| **Planning** | Decompose question into sub-problems | LF Planner, Static/Iterative |
| **Reasoning** | Evaluate evidence, draw conclusions | Deduce, Semantic reasoning |
| **Retrieval** | Fetch facts from knowledge sources | GraphRetrieval, HybridRetrieval, ChunkRetrieval |

+ Special executors: Math (numeric computation), Cypher (graph DB queries), Output (final answer).

### Pattern 4: kag_hybrid_executor — Three-Layer Knowledge Cascade

```python
# Pseudocode of the hybrid executor's knowledge layering
async def execute(self, logical_form):
    # Layer 1: Schema-constrained (highest precision)
    result = await schema_constrained_retrieval(logical_form)
    if result.is_sufficient():
        return result

    # Layer 2: Schema-free (broader coverage)
    result = await schema_free_retrieval(logical_form)
    if result.is_sufficient():
        return result

    # Layer 3: Raw context chunks (fallback)
    result = await chunk_retrieval(logical_form)
    return result
```

### Pattern 5: Iterative Planner with Reflection

```
Question → LF Planner → DAG → Executors → Results
                                              │
                                    Reflector ◄┘
                                    "Is the answer complete?"
                                              │
                                    If NO → Re-plan with new sub-questions
                                    If YES → Generator → Final Answer
```

The Reflector evaluates whether accumulated evidence answers the original question. If not, it generates new sub-questions and triggers another planning/execution cycle.

---

## Key Code Patterns

### Source File Layout (from GitHub + release notes)

```
kag/
├── solver/
│   ├── main_solver.py              # Entry point: invoke(query)
│   ├── logic/
│   │   └── solver_pipeline.py       # Orchestrates plan → execute → reflect loop
│   ├── execute/
│   │   └── default_lf_executor.py   # Routes LF nodes to executors
│   │       ├── _execute_lf()
│   │       └── _execute_chunk_answer()
│   ├── implementation/
│   │   ├── default_reasoner.py      # Reasoner implementation
│   │   ├── default_reflector.py     # Reflector for self-evaluation
│   │   └── default_generator.py     # Answer generation
│   └── retriever/
│       ├── impl/
│       │   ├── default_chunk_retrieval.py   # Text chunk retrieval
│       │   └── default_kg_retrieval.py      # KG structure retrieval
│       ├── chunk_retriever_abc.py   # Abstract base
│       └── kg_retriever_abc.py      # Abstract base
├── interface/
│   ├── solver/
│   │   ├── lf_planner_abc.py       # Logical Form Planner interface
│   │   ├── kag_reasoner_abc.py      # Reasoner interface
│   │   ├── kag_reflector_abc.py     # Reflector interface
│   │   ├── kag_generator_abc.py     # Generator interface
│   │   └── kag_memory_abc.py        # Memory interface
│   └── builder/
│       └── extractor_abc.py         # Extraction interface
└── builder/
    ├── component/                   # Extraction components
    └── implementation/              # Extractor implementations
```

### Pipeline Invocation

```python
# From solver_pipeline.py (simplified)
class SolverPipeline:
    @classmethod
    def from_config(cls, config):
        """Build pipeline from YAML config."""
        ...

    def run(self, query: str) -> Tuple[str, dict]:
        """
        Execute the full solve loop.
        Returns (answer, trace_log).
        """
        # 1. Plan: decompose query into logical form DAG
        lf_dag = self.planner.plan(query)

        # 2. Execute: run each LF node through appropriate executor
        for node in topological_sort(lf_dag):
            result = self.executor.execute(node, memory=self.memory)
            self.memory.store(node.variable, result)

        # 3. Reason: evaluate accumulated evidence
        reasoning_result = self.reasoner.reason(query, self.memory)

        # 4. Reflect: is the answer complete?
        if not self.reflector.is_complete(query, reasoning_result):
            # Re-plan with new sub-questions
            new_query = self.reflector.generate_followup(query, reasoning_result)
            return self.run(new_query)  # Recursive

        # 5. Generate: produce final answer
        return self.generator.generate(query, reasoning_result)
```

### v0.7/v0.8 Enhancements

- **v0.7**: Static/Iterative Planner split. FunctionCall-based executor matching. NaiveRAG pipeline as built-in option.
- **v0.8**: MCP protocol integration. KAG-Thinker model adaptation. 89% reduction in knowledge construction token costs ("Lightweight Build" mode). `open_benchmark` directory for RAG method comparison.

---

## Limitations for Hermes Foundry

1. **Heavyweight infrastructure**: Requires OpenSPG server (Java) as the graph backend. Not embeddable. Not suitable for an in-process Python library.

2. **LLM-driven planning**: The logical form decomposition relies entirely on LLM. No deterministic fallback. LLM planning errors cascade to incorrect retrieval and wrong answers.

3. **Professional domain focus**: Designed for domain-specific KB (E-Gov, E-Health) with predefined schemas. Not optimized for conversational memory with its informal, evolving entity landscape.

4. **No temporal interval reasoning**: No Allen algebra. No validity windows. No temporal supersession. Facts are assumed to be timelessly true.

5. **No composable query language**: The "query language" is the logical form DSL, but it is constructed by the LLM planner, not by users. There is no Python-native API for building queries programmatically.

6. **No lazy evaluation**: The solver pipeline executes immediately. No query plan inspection or optimization before execution. No clone-on-write composition.

7. **Complex setup**: Requires Java OpenSPG server, Neo4j or similar graph DB, embedding model, LLM — significant operational overhead compared to an embedded library.

---

## Steal List (what to adopt)

### S1. Three-Operator Taxonomy (Planning, Reasoning, Retrieval)
Clean separation of concerns. The idea that query execution involves PLANNING (decomposition), RETRIEVAL (fetching facts), and REASONING (evaluating evidence) is the right mental model.

**Adaptation for Hermes**: Our query plan compiler does PLANNING (building the execution DAG). The executor does RETRIEVAL (running plan nodes). The answer generator does REASONING. We already have this structure implicitly — make it explicit.

### S2. Logical Form as Query Plan Representation
The idea of expressing queries as a DAG of typed functions with variable references is powerful. Each node specifies what to retrieve and how, with dependencies making execution order explicit.

**Adaptation for Hermes**: Our query plan nodes (FilterNode, FlatMapNode, etc.) are already logical form functions. The DAG structure with shared variables maps to our query plan's intermediate result passing. Formalize variable binding between plan nodes.

### S3. Mutual Indexing (KG <-> Text Chunks)
Every KG fact links back to its source text chunk, and every chunk links to extracted facts. This enables both structured retrieval (KG queries) and unstructured retrieval (text search) with bidirectional navigation.

**Adaptation for Hermes**: Our STATED layer already stores source turns. Ensure every MSL relation has a `source_turn_ids` field, and every turn has an `extracted_relation_ids` field. This enables "show me the conversation that produced this fact."

### S4. Three-Layer Knowledge Cascade
Schema-constrained (highest precision) -> Schema-free (broader) -> Raw chunks (fallback). The cascade prioritizes precision over recall.

**Adaptation for Hermes**: Map to our three-layer memory:
- Schema-constrained = MSL structured store (typed entities + relations)
- Schema-free = BM25/cosine over extracted triples
- Raw chunks = BM25/cosine over original conversation chunks

### S5. Reflect-and-Replan Loop
After execution, evaluate whether the answer is sufficient. If not, generate follow-up sub-questions and re-execute. This is a principled form of iterative retrieval.

**Adaptation for Hermes**: Our agentic retriever already does this. Formalize the reflect step as a post-retrieval quality gate that triggers re-planning when confidence is low.

### S6. DAG-Based Execution with Variable Binding
Sub-problems form a DAG where later steps reference results from earlier steps via shared variable names. This enables parallel execution of independent sub-problems.

**Adaptation for Hermes**: Express multi-hop queries as DAGs rather than linear chains. Allow query plan nodes to reference intermediate results by name.

---

## Anti-Patterns (what to avoid)

### A1. LLM-Driven Query Planning
KAG's planner uses the LLM to decompose queries into logical forms. This is slow (LLM inference per query), expensive (token cost), and unreliable (LLM may decompose incorrectly). For common query patterns, deterministic planning is better.

**Instead**: Use deterministic planning for common patterns (entity lookup, temporal query, aggregation). Reserve LLM planning for novel/complex queries only. This matches our "traces -> deterministic -> distilled SLM" roadmap.

### A2. Java Backend Requirement
OpenSPG is a Java server that must run alongside the Python application. This is operational overhead we do not need.

**Instead**: Use embedded solutions (Kuzu, in-memory) for the graph backend. No separate server process.

### A3. No User-Facing Query Composition
The logical form DSL is an internal representation constructed by the LLM. Users cannot build queries programmatically. This limits the system to LLM-mediated access.

**Instead**: Expose a Python-native fluent API (our ObjectSet/MemoryQuery pattern) that humans can use directly AND that LLMs can generate.

---

## Key Metrics

| Metric | Value |
|--------|-------|
| HotpotQA F1 improvement | +19.6% over SOTA |
| 2WikiMultiHopQA F1 improvement | +33.5% over SOTA |
| Token cost reduction (v0.8) | -89% with Lightweight Build |
| E-Gov accuracy | "Significantly higher" than traditional RAG |

---

## Sources

1. [KAG Paper (arXiv:2409.13731)](https://arxiv.org/html/2409.13731v3) — Full paper with architecture details
2. [KAG GitHub](https://github.com/OpenSPG/KAG) — Source code
3. [KAG v0.7 Release Notes](https://openspg.github.io/v2/blog/recent_posts/release_notes/0.7) — Static/Iterative planner, FunctionCall executors
4. [KAG v0.8 Release Notes](https://github.com/OpenSPG/KAG/releases/tag/v0.8.0) — MCP protocol, KAG-Thinker, Lightweight Build
5. [KAG-Thinker Paper (arXiv:2506.17728)](https://arxiv.org/html/2506.17728) — Deep reasoning extension
6. [KAG ACM Paper](https://dl.acm.org/doi/10.1145/3701716.3715240) — WWW 2025 companion proceedings
