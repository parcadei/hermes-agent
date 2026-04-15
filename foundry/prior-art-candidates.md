# Prior Art Scan: Memory Query Systems & Structured Retrieval
Generated: 2026-04-13

## Summary

This scan covers 40+ projects, frameworks, and research papers across 7 categories relevant to building a memory operating system for conversational AI. The landscape divides into three tiers: (1) production memory frameworks (Mem0, Zep/Graphiti, Letta, Cognee, Honcho) that solve storage/retrieval but lack composable query languages; (2) graph query systems (Gremlin, Cypher/Kuzu, KAG) that solve structured querying but aren't designed for conversational memory; and (3) research papers (A-MEM, THEANINE, LiCoMemory, Hindsight) that advance temporal/structured memory but lack production implementations. No existing system combines typed entities, lazy composable query plans (ObjectSet pattern), Allen interval temporal reasoning, link traversal, and aggregation into a unified Python query language for conversational memory.

---

## 1. MEMORY SYSTEMS FOR AI AGENTS

### 1.1 Mem0
- **URL**: https://github.com/mem0ai/mem0
- **Language**: Python
- **Stars**: ~48,000
- **Key Insight**: Three-tier memory (user/session/agent) with self-editing. When facts conflict, Mem0 chooses ADD/UPDATE/DELETE/NOOP rather than appending duplicates.
- **Architecture**: Hybrid store — vectors + graph relationships + key-value. Each memory operation compares new facts against existing ones to select mutation type.
- **Limitations**: No composable query language. No temporal interval reasoning. No lazy evaluation or query plans. Retrieval is semantic similarity, not structured traversal. Graph variant exists but is bolt-on, not the primary interface.
- **Steal-worthy**: The 4-operation mutation model (ADD/UPDATE/DELETE/NOOP) for conflict resolution. Their graph variant's approach to entity-level deduplication.
- **Benchmark**: 67.13% LLM-as-Judge on LoCoMo, P95 search latency 0.200s, ~1,764 tokens/conversation.
- **Paper**: arXiv:2504.19413

### 1.2 Zep / Graphiti
- **URL**: https://github.com/getzep/graphiti
- **Language**: Python
- **Stars**: ~5,000+
- **Key Insight**: THE closest prior art. Bi-temporal knowledge graph with validity windows on every edge. Episodic → Semantic → Community subgraph hierarchy. Hybrid retrieval (BM25 + semantic + graph traversal) with no LLM calls at query time.
- **Architecture**: G = (N, E, φ) with three hierarchical tiers. Every edge has (t_valid, t_invalid, t_created, t_expired). Entity resolution via LLM + full-text search on candidate names. Episode subgraph preserves raw ground truth. Semantic entity subgraph stores resolved entities. Community subgraph provides global summaries.
- **Limitations**: Requires Neo4j (or FalkorDB/Kuzu) backend — heavyweight. No composable query language or lazy evaluation. No Allen interval algebra — only validity windows. No aggregation operators. Entity resolution is LLM-dependent (expensive). No typed entity ontology beyond prescribed Pydantic models.
- **Steal-worthy**: (1) Bi-temporal model (event time vs. ingestion time). (2) Episode subgraph as ground truth with lineage to derived facts. (3) Edge invalidation rather than deletion for temporal supersession. (4) Hybrid search combining BM25 + semantic + graph traversal at query time. (5) Prescribed vs. learned ontology via Pydantic models.
- **Benchmark**: 94.8% on DMR benchmark. P95 retrieval latency 300ms.
- **Paper**: arXiv:2501.13956

### 1.3 Letta (formerly MemGPT)
- **URL**: https://github.com/letta-ai/letta
- **Language**: Python
- **Stars**: ~13,000+
- **Key Insight**: OS-inspired memory management — core memory (in-context RAM), archival memory (disk/vector store), recall memory (conversation log). Agents control their own memory via function calls.
- **Architecture**: Three-tier with explicit paging. Agent decides what to archive, search, and promote. Memory management is itself an agentic tool-use problem.
- **Limitations**: No knowledge graph. No structured query. No temporal reasoning. The OS-paging metaphor adds latency. Simple vector store for archival. No entity resolution. No composable query API.
- **Steal-worthy**: The "agent controls its own memory" paradigm. The conceptual framing of memory as an OS resource with explicit read/write/search operations.
- **Benchmark**: 74.0% on LoCoMo with GPT-4o-mini.

### 1.4 Cognee
- **URL**: https://github.com/topoteretes/cognee
- **Language**: Python
- **Stars**: ~7,000+
- **Key Insight**: Six-stage pipeline: classify → permissions → extract chunks → LLM extract entities/relationships → generate summaries → embed + commit to graph. `memify` operation evolves the graph post-ingestion.
- **Architecture**: Graph-vector hybrid. 14 retrieval modes from classic RAG to chain-of-thought graph traversal. Extraction via Instructor-powered structured output → KnowledgeGraph (Node + Edge objects) → ontology validation → BFS traversal. Backends: SQLite + LanceDB + Kuzu (default), swappable to Neo4j/Neptune/Qdrant/pgvector.
- **Limitations**: Retrieval modes are predefined, not composable. No lazy query plans. No temporal interval algebra. No aggregation. Heavy LLM dependency for extraction. `memify` evolution is opaque — no user-facing query language to inspect or control it.
- **Steal-worthy**: (1) The `cognify` pipeline as a model for multi-stage extraction. (2) Ontology validation step during extraction. (3) `memify` as a post-ingestion graph evolution step. (4) Using Kuzu as embedded graph backend.

### 1.5 Honcho
- **URL**: https://github.com/plastic-labs/honcho
- **Language**: Python
- **Stars**: ~1,500+
- **Key Insight**: Entity-centric "peer" model where both users and agents are first-class entities. Dialectic reasoning generates "conclusions" (insights) that accumulate into "representations" (profiles). Background "dreaming" continuously optimizes peer models.
- **Architecture**: Peers → Sessions → Messages → Conclusions → Representations. Neuromancer reasoning engine processes messages in background. Hybrid search (semantic + BM25) with workspace/session/peer-level scoping. Collections for vector-embedded data.
- **Limitations**: No graph traversal. No temporal interval reasoning. No composable query language. Closed reasoning engine (Neuromancer). No explicit entity resolution — relies on session/peer scoping. No aggregation operators.
- **Steal-worthy**: (1) Peer model unifying users and agents as entities. (2) Background "dreaming" for continuous model improvement. (3) Conclusions-to-representations pipeline (fact extraction → profile synthesis). (4) Multi-level scoping (workspace/session/peer) for search.

### 1.6 MemoryBank
- **URL**: https://github.com/zhongwanjun/MemoryBank-SiliconFriend
- **Language**: Python
- **Stars**: ~300+
- **Key Insight**: Ebbinghaus forgetting curve applied to memory decay. R = e^(-t/S) where S increases on recall. Memories that are never recalled fade; frequently accessed ones persist.
- **Architecture**: Each memory piece encoded into contextual representation. Current context serves as query for most relevant memory. Memory strength S initialized at 1, incremented on recall, t reset to 0.
- **Limitations**: No structured query. No graph. No temporal reasoning beyond decay. No entity resolution. No composable operations. No aggregation.
- **Steal-worthy**: The forgetting curve model for memory strength/decay. Simple, principled, and biologically motivated.

### 1.7 Hindsight
- **URL**: https://github.com/vectorize-io/hindsight
- **Language**: Python
- **Stars**: ~2,000+
- **Key Insight**: Four logical memory networks: world facts, agent experiences, synthesized entity summaries, and evolving beliefs. Distinguishes facts from beliefs. TEMPR (Temporal Entity Memory Priming Retrieval) for temporal entity graphs.
- **Architecture**: Parallel retrieval: semantic vector search + BM25 keyword search + entity/causal/temporal graph traversal + explicit temporal filtering. CARA (Coherent Adaptive Reasoning Agents) shapes responses via agent profiles and belief evolution.
- **Limitations**: Tightly coupled to their own runtime. No composable query language exposed to users. No Allen interval algebra. No lazy evaluation. Beliefs are opaque — no query language to inspect belief state.
- **Steal-worthy**: (1) Epistemic distinction between facts and beliefs. (2) TEMPR for temporal entity graph construction. (3) Parallel multi-modal retrieval (vector + BM25 + graph + temporal). (4) Belief evolution as a first-class memory type.
- **Benchmark**: 91.4% on LongMemEval. Multi-session: 21.1% → 79.7%. Temporal reasoning: 31.6% → 79.7%.

### 1.8 Memvid
- **URL**: https://github.com/memvid/memvid
- **Language**: Python
- **Stars**: ~3,000+
- **Key Insight**: Single-file memory layer. Memories stored as "Smart Frames" in append-only immutable sequence (inspired by video encoding). No database required.
- **Architecture**: Everything — data, embeddings, search index, metadata — packaged into a single `.mv2` file. Append-only, content-hashed, timestamped.
- **Limitations**: No concurrent writes, no ACID, no fine-grained deletions. No graph. No structured query. No temporal reasoning. Read-mostly tool for static/slow-changing memory.
- **Steal-worthy**: The append-only immutable design for audit trails. Content hashing for deduplication.

---

## 2. KNOWLEDGE GRAPH QUERY LANGUAGES

### 2.1 Cypher (Neo4j / Kuzu / FalkorDB)
- **URL**: https://github.com/kuzudb/kuzu (embedded), https://github.com/FalkorDB/FalkorDB
- **Language**: Cypher query language; backends in C++ (Kuzu, FalkorDB), Java (Neo4j)
- **Key Insight**: Declarative, pattern-matching query language for property graphs. `MATCH (a)-[r]->(b) WHERE ...` syntax is the de facto standard.
- **Architecture**: Pattern matching against node/edge schemas. Supports OPTIONAL MATCH, variable-length paths, aggregation, UNION, subqueries.
- **Limitations**: Not a Python-native API — requires string-based query construction or Text2Cypher. No lazy evaluation at the API level. Temporal support is limited to property-level DATE/TIMESTAMP — no interval algebra built in.
- **Steal-worthy**: (1) Kuzu as embedded Python graph DB — `pip install kuzu`, Cypher queries, zero config. (2) FalkorDB's GraphBLAS sparse matrix representation for adjacency. (3) Cypher's MATCH pattern syntax as inspiration for a Python-native DSL. (4) Variable-length path matching `(a)-[*1..3]->(b)`.

### 2.2 Gremlin (Apache TinkerPop)
- **URL**: https://tinkerpop.apache.org/ ; https://pypi.org/project/gremlinpython/
- **Language**: Multi-language (Python via gremlinpython)
- **Key Insight**: Functional, composable, step-based graph traversal language. Every traversal is a pipeline of map/filter/sideEffect steps. Turing complete. Supports both OLTP and OLAP execution.
- **Architecture**: `Traversal<S,E>` composed of `Step<S,E>` objects. TraversalStrategy for query rewriting. Declarative mode allows runtime query planning based on statistics. Host-language function composition (fluent API).
- **Limitations**: Requires TinkerPop-enabled server (Gremlin Server). Heavier than embedded solutions. Python variant mirrors Java API — Pythonic but not idiomatic. No temporal interval built-in. No lazy evaluation at the Python level — executes against remote server.
- **Steal-worthy**: (1) Step-based composable traversal model (map/filter/sideEffect). (2) TraversalStrategy as interceptor for query optimization. (3) Declarative traversal mode with runtime query planning. (4) The `Traverser<T>` object propagating through the pipeline. (5) Bidirectional traversal support.

### 2.3 SPARQL / Temporal SPARQL Extensions
- **URL**: Various (see temporal SPARQL section)
- **Language**: Standard query language for RDF
- **Key Insight**: Multiple temporal extensions exist: τ-SPARQL (translates to standard SPARQL), T-SPARQL (TSQL2-inspired triple timestamping), tRDF (SQL:2011-backed temporal RDF).
- **Limitations**: RDF triple model is more verbose than property graphs. No Python-native composable API. Temporal extensions are academic — no production-ready unified implementation.
- **Steal-worthy**: (1) τ-SPARQL's approach of encoding temporal queries within standard syntax. (2) T-SPARQL's multi-temporal triple timestamping model. (3) The general principle of temporal elements on triples rather than separate temporal tables.

### 2.4 KAG (Knowledge Augmented Generation / OpenSPG)
- **URL**: https://github.com/OpenSPG/KAG
- **Language**: Python
- **Stars**: ~5,000+
- **Key Insight**: Logical-form-guided reasoning engine. Breaks complex queries into sub-problems using executable language with retrieval and logical operators. Three operator types: planning, reasoning, retrieval. Combines LLM reasoning + KG reasoning + mathematical logic.
- **Architecture**: KAG-Builder (offline KG construction with LLM-friendly representation + mutual indexing) → KAG-Solver (logical form hybrid reasoning with planning/reasoning/retrieval operators) → KAG-Model (capability fine-tuning). ISO GQL support via OpenSPG server.
- **Limitations**: Heavyweight infrastructure (requires OpenSPG server). Not designed for conversational memory — targets professional domain KBs. No temporal interval algebra. Python setup complex. No lazy query plan API — the solver is LLM-driven, not user-composable.
- **Steal-worthy**: (1) The logical-form DSL that decomposes questions into executable sub-problems. (2) Three-operator taxonomy (planning, reasoning, retrieval). (3) Mutual indexing between KG structure and text chunks. (4) The idea of a hybrid solver combining symbolic + neural reasoning.
- **Benchmark**: 19.6% F1 improvement on HotpotQA, 33.5% on 2WikiMultiHopQA.

---

## 3. OBJECTSET / LAZY QUERY PLAN PATTERNS

### 3.1 Palantir OSDK ObjectSet
- **URL**: https://www.palantir.com/docs/foundry/functions/api-object-sets
- **Language**: TypeScript (primary), Python (OSDK v2)
- **Key Insight**: The canonical ObjectSet pattern. `.where()` for filtering, `.union()` / `.intersect()` / `.subtract()` for set operations. Composable filters via `and()` / `or()`. Links navigable as object sets themselves. Lazy — queries only execute on terminal operations like `.all()` or `.page()`.
- **Architecture**: Typed ontology objects. Filter predicates compose. Set operations create new ObjectSets without execution. `.group_by()` for aggregation. Link traversal returns ObjectSets. REST API backs everything.
- **Limitations**: Proprietary. Python OSDK v2 is basic compared to TypeScript. No graph traversal beyond typed links. No temporal interval reasoning. Designed for enterprise ontologies, not conversational memory.
- **Steal-worthy**: THE primary inspiration. (1) `.where()` returning new ObjectSet. (2) `.union()` / `.intersect()` / `.subtract()` as set algebra. (3) Links as navigable ObjectSets. (4) Lazy evaluation — nothing executes until terminal operation. (5) Typed ontology with schema-level entity types. (6) `.group_by()` and aggregation on ObjectSets.

### 3.2 Django QuerySet
- **URL**: https://docs.djangoproject.com/en/6.0/topics/db/queries/
- **Language**: Python
- **Key Insight**: The gold standard for lazy, composable query building in Python. Clone-on-write immutability. Each `.filter()` / `.exclude()` returns a new QuerySet without executing. SQL only fires on iteration, `len()`, `list()`, slicing, or `bool()`.
- **Architecture**: Multi-stage pipeline: QuerySet layer (chainable API) → Query object (accumulates state) → Expression system (composable Python objects) → SQL compiler → Result processing. `_chain()` creates lightweight copies.
- **Limitations**: SQL-only backend. Not designed for graph queries. No graph traversal. No temporal interval reasoning.
- **Steal-worthy**: (1) Clone-on-write pattern for immutable query building. (2) Terminal operation triggers (iteration, len, list, bool, slicing). (3) Expression system as composable Python objects. (4) `annotate()` for computed attributes. (5) The `_chain()` / `_clone()` internal pattern.

### 3.3 SQLAlchemy Core Expression Language
- **URL**: https://github.com/sqlalchemy/sqlalchemy
- **Language**: Python
- **Stars**: ~10,000+
- **Key Insight**: Dual-layer: Core (low-level SQL expression builder) + ORM (high-level mapper). Core builds SQL from composable Python objects that compile to statements at execution time. Database-agnostic. Identity map + Unit of Work patterns.
- **Limitations**: SQL-oriented. No graph support. No temporal intervals.
- **Steal-worthy**: (1) Dual-layer abstraction (Core for raw power, ORM for convenience). (2) Expression trees that compile to backend-specific queries. (3) Dialect system for multi-backend support. (4) Transparent query caching (SQLAlchemy 2.0).

### 3.4 smort-query
- **URL**: https://github.com/matiuszka/smort-query
- **Language**: Python
- **Stars**: 7
- **Key Insight**: Django QuerySet API for in-memory Python objects. `.filter()` / `.exclude()` / `.order_by()` / `.annotate()` with lazy evaluation. Double-underscore syntax for nested attribute access and comparators.
- **Limitations**: Inactive/unmaintained. Very limited adoption. No graph support. No temporal reasoning.
- **Steal-worthy**: Proof that the Django QuerySet pattern can be cleanly applied to in-memory objects. The double-underscore path syntax.

### 3.5 NoGraphs
- **URL**: https://github.com/HeWeMel/nographs
- **Language**: Python (pure)
- **Stars**: ~200+
- **Key Insight**: Graph analysis via lazy evaluation on implicit graphs. Graphs are computed on-the-fly, never stored. Supports DFS, BFS, Dijkstra, A*, MST. Graph operations (union, intersection, products, pruning) are composable and lazy.
- **Limitations**: Pure analysis library — no storage, no persistence. No property graph model. No temporal reasoning. No query language per se.
- **Steal-worthy**: (1) The implicit graph concept — compute adjacency on demand. (2) Composable graph operations as lazy transformations. (3) Problem reduction via lazy composition (derived graphs from combining operations).

---

## 4. TEMPORAL KNOWLEDGE GRAPHS

### 4.1 Allen Interval Algebra Implementations

#### 4.1.1 allen-ia
- **URL**: https://github.com/psiwray/allen-ia
- **Language**: Python
- **Key Insight**: Uses SAT solver to validate boolean expressions over Allen interval constraints. Ternary constraint table for transitive reasoning.
- **Limitations**: Small, niche. Not integrated with any graph or memory system.
- **Steal-worthy**: SAT-based constraint validation for interval relationships.

#### 4.1.2 QSRlib
- **URL**: https://qsrlib.readthedocs.io/en/latest/rsts/handwritten/qsrs/allen.html
- **Language**: Python
- **Key Insight**: Full implementation of Allen's Interval Algebra with composition table. Part of a broader qualitative spatial reasoning library.
- **Limitations**: Focused on robotics/spatial reasoning. Not integrated with KG or memory systems.
- **Steal-worthy**: Clean Allen IA implementation with composition table.

#### 4.1.3 qualreas
- **Language**: Python
- **Key Insight**: Framework for qualitative reasoning over networks of relation algebras including Allen's IA, RCC-8, and integrated time point + interval algebras.
- **Steal-worthy**: Multi-algebra reasoning framework that combines Allen IA with other calculi.

### 4.2 THEANINE (NAACL 2025)
- **URL**: https://aclanthology.org/2025.naacl-long.435/
- **Language**: Python (research)
- **Key Insight**: Discards memory removal entirely. Manages large-scale memories by linking them via temporal and cause-effect relations in a directed graph. Timeline retrieval returns series of memories representing evolution/causality of past events.
- **Architecture**: Three phases: (1) Memory Graph Construction — memories linked via temporal + commonsense relations. (2) Timeline Retrieval — walks graph to extract evolution sequences. (3) Timeline Refinement — LLM refines raw timelines for current conversation context.
- **Limitations**: Research prototype. No composable query API. No entity resolution. No aggregation. LLM-dependent for refinement.
- **Steal-worthy**: (1) Keeping ALL memories and linking them rather than pruning. (2) Timeline retrieval as a first-class operation — "give me the evolution of X." (3) Relation-aware linking is more impactful than timeline retrieval or refinement (ablation result). (4) Context-aware timeline refinement at query time.
- **Benchmark**: 72% retrieval accuracy for golden memories.
- **Paper**: arXiv:2406.10996

### 4.3 Graphiti Bi-Temporal Model (see 1.2 above)
- Tracks 4 timestamps per edge: t_created, t_expired, t_valid, t_invalid.
- Invalidation rather than deletion preserves temporal history.

---

## 5. STRUCTURED RETRIEVAL FOR RAG

### 5.1 Microsoft GraphRAG
- **URL**: https://github.com/microsoft/graphrag
- **Language**: Python
- **Stars**: ~20,000+
- **Key Insight**: Hierarchical community-based search. Local search (entity matching → community reports) + Global search (high-level community summaries). 86% accuracy vs 32% for baseline RAG on enterprise benchmarks.
- **Limitations**: Extremely expensive — $33k estimated for a single 5GB document. Not designed for conversational memory. No temporal reasoning. No composable query language.
- **Steal-worthy**: Community detection for hierarchical summarization. The local/global search distinction.

### 5.2 LightRAG
- **URL**: https://github.com/HKUDS/LightRAG
- **Language**: Python
- **Stars**: ~15,000+
- **Key Insight**: Dual-level retrieval: low-level (exact key matching) + high-level (thematic). Graph-enhanced text indexing. Incremental graph updates via unioning new documents.
- **Limitations**: No temporal reasoning. No composable query. Not memory-focused.
- **Steal-worthy**: (1) Dual-level retrieval strategy. (2) Incremental graph update via union. (3) 50% faster update time than GraphRAG.

### 5.3 HippoRAG
- **URL**: https://github.com/OSU-NLP-Group/HippoRAG
- **Language**: Python
- **Stars**: ~3,000+
- **Key Insight**: Neurobiologically-inspired. Uses Personalized PageRank (PPR) for link-based retrieval traversal. 10-30x cheaper than Microsoft GraphRAG for multi-hop reasoning.
- **Limitations**: No temporal reasoning. No composable query. Narrower scope than general memory systems.
- **Steal-worthy**: PPR-guided retrieval traversal. The neurobiological framing (hippocampal indexing theory).

### 5.4 PathRAG (AAAI 2025)
- **URL**: https://arxiv.org/abs/2502.14902
- **Language**: Python
- **Key Insight**: Focuses on key relational paths between query-related nodes rather than entire communities or neighborhoods. Reduces noise and token consumption.
- **Steal-worthy**: Path-based pruning — only keep the relational paths that matter. 59.93% win rate over GraphRAG.

### 5.5 KGGen (NeurIPS 2025)
- **URL**: https://github.com/stair-lab/kg-gen
- **Language**: Python
- **Key Insight**: Uses DSPy for structured output KG generation from plain text. Supports RAG-focused graph creation and synthetic training data.
- **Steal-worthy**: DSPy-based structured extraction pipeline.

---

## 6. ENTITY RESOLUTION SYSTEMS

### 6.1 Dedupe
- **URL**: https://github.com/dedupeio/dedupe
- **Language**: Python
- **Stars**: ~4,000+
- **Key Insight**: Active learning for fuzzy matching and entity resolution on structured data. Based on Bilenko's PhD work on learnable similarity functions.
- **Limitations**: Designed for tabular data, not graph entities. Requires training data (active learning). Not real-time.
- **Steal-worthy**: Active learning loop for entity resolution training. Learnable similarity functions.

### 6.2 Splink
- **URL**: https://github.com/moj-analytical-services/splink
- **Language**: Python (DuckDB/Spark backend)
- **Stars**: ~4,500+
- **Key Insight**: Probabilistic record linkage at scale. No training data required — uses Fellegi-Sunter model. 1M records on a laptop in ~1 minute. Supports DuckDB, Spark, Athena backends.
- **Limitations**: Tabular data focused. Not designed for graph entity resolution. No real-time/streaming mode.
- **Steal-worthy**: (1) Fellegi-Sunter probabilistic model (no training data needed). (2) Multi-backend architecture. (3) Comparison templates for common matching patterns.

### 6.3 DeezyMatch
- **URL**: https://github.com/Living-with-machines/DeezyMatch
- **Language**: Python
- **Key Insight**: Deep embedding + approximate nearest-neighbor blocking for entity resolution. Useful for historical/variant name matching.
- **Steal-worthy**: ANN-based blocking for candidate generation in entity resolution.

---

## 7. RESEARCH PAPERS

### 7.1 A-RAG (arXiv:2602.03442)
- **Key Insight**: Exposes hierarchical retrieval interfaces (keyword search, semantic search, chunk read) directly to the model. Agent adaptively decides what to retrieve at what granularity. Scales with model capability.
- **Steal-worthy**: Three-tool retrieval interface. Hierarchical granularity (keyword → semantic → chunk read). Systematic study of scaling with model size and test-time compute.

### 7.2 A-MEM: Agentic Memory (arXiv:2502.12110, NeurIPS 2025)
- **URL**: https://github.com/agiresearch/A-mem
- **Key Insight**: Zettelkasten-inspired memory organization. Memories generate their own contextual descriptions, form connections to related memories, and EVOLVE (update their own attributes and links as new memories arrive). Memory is not static storage — it's a self-organizing network.
- **Steal-worthy**: (1) Zettelkasten model for memory organization. (2) Memory evolution — new memories trigger updates to old memories' representations. (3) Structured note generation (context, keywords, tags) per memory. (4) Connection discovery as a first-class operation.

### 7.3 MemWalker (arXiv:2310.05029)
- **Key Insight**: Processes long context into a tree of summary nodes. Agent navigates tree from root to find relevant segments. Two-stage: tree construction + interactive navigation.
- **Steal-worthy**: Hierarchical summary tree for navigating large memory stores. Interactive tree walking paradigm.

### 7.4 ReadAgent (ICML 2024, arXiv:2402.09727)
- **Key Insight**: Human-inspired reading: (1) decide what to group into memory episodes, (2) compress into gist memories, (3) look up original text on demand. Extends effective context window 3.5-20x.
- **Steal-worthy**: Gist memory concept — compressed episodes with on-demand expansion. The decide → compress → lookup pipeline.

### 7.5 MemOS (EMNLP 2025)
- **URL**: https://github.com/BAI-LAB/MemoryOS
- **Key Insight**: Hierarchical storage (short/mid/long-term) with four core modules: Storage, Updating, Retrieval, Generation. Inspired by OS memory management.
- **Steal-worthy**: The four-module architecture as a clean decomposition. 49.11% F1 improvement on LoCoMo.

### 7.6 MemOS (MemTensor, arXiv:2507.03724)
- **URL**: https://github.com/MemTensor/MemOS
- **Key Insight**: MemCube — unified abstraction encapsulating plaintext, activation, and parameter memories. Three-layer architecture: Interface → Operation → Infrastructure. Treats memory as a first-class operational resource.
- **Steal-worthy**: MemCube unified abstraction. The idea of memory as a schedulable resource.

### 7.7 LiCoMemory (arXiv:2511.01448)
- **Key Insight**: CogniGraph — lightweight hierarchical graph using entities and relations as semantic indexing layers. Temporal and hierarchy-aware search with reranking. End-to-end agentic memory framework for real-time updating and retrieval.
- **Steal-worthy**: (1) CogniGraph hierarchical indexing (entities → relations → facts). (2) Temporal decay function during ranking. (3) Query-time temporal range inference ("last month" → window). (4) Incremental graph construction for low-latency updates.
- **Benchmark**: 73.8% accuracy / 76.6% recall on LongMemEval with GPT-4o-mini.

### 7.8 Agentic RAG Survey (arXiv:2501.09136)
- **Key Insight**: Taxonomy of agentic RAG by agent cardinality, control structure, autonomy, and knowledge representation. Identifies long-term memory design as a key open challenge.
- **Steal-worthy**: The taxonomy framework. The identification of memory drift, bias reinforcement, and hallucination amplification as key risks.

### 7.9 Graph-Based Agent Memory Survey (arXiv:2602.05665)
- **Key Insight**: Comprehensive survey positioning graph-based memory as the frontier. Covers flat vs. hierarchical graph structures, temporal modeling, multi-agent memory sharing.
- **Steal-worthy**: The taxonomy of graph memory architectures. Analysis of what works (hierarchy, temporal) and what doesn't (flat entangled structures).

---

## 8. GRAPH DATABASE BACKENDS (Embedded)

### 8.1 Kuzu
- **URL**: https://github.com/kuzudb/kuzu
- **Language**: C++ core, Python bindings
- **Stars**: ~5,000+
- **Key Insight**: Embedded property graph DB. `pip install kuzu`. Cypher queries. Vectorized OLAP execution. Parquet/Arrow/DuckDB interop. MIT license.
- **Steal-worthy**: Zero-config embedded graph DB. Cypher support. DataFrame interop. LlamaIndex/LangChain integrations.

### 8.2 FalkorDB / FalkorDBLite
- **URL**: https://github.com/FalkorDB/FalkorDB
- **Language**: C core (GraphBLAS), Python bindings
- **Stars**: ~3,000+
- **Key Insight**: GraphBLAS sparse matrix adjacency. FalkorDBLite runs as subprocess with file-based storage — no server, no Docker. OpenCypher support.
- **Steal-worthy**: GraphBLAS for sparse matrix graph operations. Embedded subprocess architecture. Graphiti integration for temporal KG.

---

## 9. COMPARISON MATRIX

### Memory Systems
| System | Graph | Temporal | Composable Query | Entity Resolution | Lazy Eval | Aggregation | Python-Native |
|--------|-------|----------|-----------------|-------------------|-----------|-------------|---------------|
| Mem0 | Partial | No | No | Basic | No | No | Yes |
| Zep/Graphiti | Yes | Bi-temporal | No | LLM-based | No | No | Yes |
| Letta | No | No | No | No | No | No | Yes |
| Cognee | Yes | No | No | LLM-based | No | No | Yes |
| Honcho | No | No | No | Peer-scoped | No | No | Yes |
| Hindsight | Yes | TEMPR | No | Yes | No | No | Yes |
| A-MEM | Yes | No | No | Connection-based | No | No | Yes |
| LiCoMemory | Yes | Decay+range | No | Hierarchical | No | No | Yes |
| **Hermes (target)** | **Yes** | **Allen IA** | **Yes (ObjectSet)** | **Multi-strategy** | **Yes** | **Yes** | **Yes** |

### Query Pattern Systems
| System | Lazy Eval | Composable | Graph | Set Ops | Aggregation | Temporal | Python |
|--------|-----------|------------|-------|---------|-------------|----------|--------|
| Palantir OSDK | Yes | Yes | Links | Yes | Yes | No | Partial |
| Django QuerySet | Yes | Yes | No | No | Yes | No | Yes |
| SQLAlchemy Core | Yes | Yes | No | No | Yes | No | Yes |
| Gremlin | Remote | Yes (steps) | Yes | No | Yes (sideEffect) | No | Yes |
| Cypher (Kuzu) | No | Pattern-based | Yes | UNION | Yes | Property-level | Bindings |
| KAG Solver | No | LLM-driven | Yes | No | Yes | No | Yes |
| NoGraphs | Yes | Yes | Implicit | Yes | No | No | Yes |
| smort-query | Yes | Yes | No | No | No | No | Yes |
| **Hermes (target)** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Allen IA** | **Yes** |

---

## 10. RECOMMENDATIONS FOR THIS CODEBASE

### Architecture Steal List (priority order)

1. **Palantir OSDK ObjectSet API** — THE model for the query language. `.where()`, `.union()`, `.intersect()`, `.subtract()`, terminal operations. Adapt for graph entities instead of Foundry objects.

2. **Django QuerySet internals** — Clone-on-write pattern for immutable query building. Terminal operation detection (iteration, len, list, bool). `_chain()` for lightweight copies. `annotate()` for computed attributes.

3. **Gremlin step model** — Map/filter/sideEffect decomposition. TraversalStrategy for query optimization. Traverser objects propagating through pipeline. Use this for link traversal operations.

4. **Graphiti bi-temporal model** — 4-timestamp edges (t_valid, t_invalid, t_created, t_expired). Episode subgraph as ground truth. Edge invalidation not deletion. Combine with Allen IA for the temporal query operators.

5. **THEANINE timeline retrieval** — "Give me the evolution of entity X" as a first-class query. Temporal + causal linking. Context-aware refinement at query time.

6. **Cognee's extraction pipeline** — Multi-stage: classify → extract → validate ontology → embed → commit. Use as model for the ingestion side.

7. **A-MEM's Zettelkasten evolution** — Memory notes that evolve when new memories arrive. Connection discovery as explicit operation. Steal the structured note schema.

8. **LiCoMemory's CogniGraph** — Hierarchical entity → relation → fact indexing. Temporal decay in ranking. Query-time temporal range inference.

### What Does NOT Exist (Our Novel Contribution)

No existing system combines ALL of:
- Typed entities with schema-level ontology
- Lazy composable query plans (ObjectSet pattern)
- Allen interval algebra for temporal reasoning
- Link traversal returning composable entity sets
- Aggregation operators on entity sets
- Python-native fluent API (not string-based Cypher/SPARQL)
- Designed for conversational memory (not enterprise ontologies or static documents)

This is the gap we fill.

### Implementation Patterns

1. **Query plan as expression tree**: Build AST nodes for each operation (Filter, SetOp, Traverse, Aggregate, TemporalFilter). Compile to execution plan only on terminal operation.

2. **Backend-agnostic execution**: Like SQLAlchemy's dialect system. Expression tree compiles to in-memory operations, Kuzu Cypher, or Neo4j queries depending on scale.

3. **Terminal operations**: `.resolve()` (execute and return entities), `.count()`, `.first()`, `__iter__()`, `__len__()`, `__bool__()`.

4. **Temporal operators as first-class**: `.before(entity_set)`, `.after(entity_set)`, `.during(entity_set)`, `.overlaps(entity_set)` using Allen IA composition table.

5. **Entity typing**: Pydantic models for entity types (inspired by Graphiti). Schema validation at ingestion. Type-aware query filtering.

---

## 11. SOURCES

### Memory Systems
1. [Mem0 GitHub](https://github.com/mem0ai/mem0) — Production memory framework, 48K stars
2. [Graphiti/Zep GitHub](https://github.com/getzep/graphiti) — Temporal KG for agent memory
3. [Zep Paper (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956) — Temporal KG architecture paper
4. [Letta/MemGPT](https://github.com/letta-ai/letta) — OS-inspired memory management
5. [Cognee GitHub](https://github.com/topoteretes/cognee) — KG memory engine, 7K stars
6. [Honcho GitHub](https://github.com/plastic-labs/honcho) — Entity-centric memory with dreaming
7. [MemoryBank GitHub](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) — Ebbinghaus forgetting curve
8. [Hindsight GitHub](https://github.com/vectorize-io/hindsight) — Structured memory with beliefs
9. [Memvid GitHub](https://github.com/memvid/memvid) — Single-file memory layer

### Knowledge Graph Query
10. [Kuzu GitHub](https://github.com/kuzudb/kuzu) — Embedded graph DB, Cypher, MIT
11. [FalkorDB GitHub](https://github.com/FalkorDB/FalkorDB) — GraphBLAS-based graph DB
12. [KAG/OpenSPG GitHub](https://github.com/OpenSPG/KAG) — Logical-form reasoning framework
13. [Gremlin/TinkerPop](https://tinkerpop.apache.org/) — Composable graph traversal language
14. [gremlinpython PyPI](https://pypi.org/project/gremlinpython/) — Python Gremlin variant

### Query Patterns
15. [Palantir OSDK ObjectSet Docs](https://www.palantir.com/docs/foundry/functions/api-object-sets) — Composable ObjectSet pattern
16. [Palantir Foundry Python SDK](https://github.com/palantir/foundry-platform-python) — Python OSDK
17. [Django QuerySet Docs](https://docs.djangoproject.com/en/6.0/topics/db/queries/) — Lazy evaluation gold standard
18. [SQLAlchemy GitHub](https://github.com/sqlalchemy/sqlalchemy) — Expression tree + dual-layer architecture
19. [smort-query GitHub](https://github.com/matiuszka/smort-query) — Django QuerySet for in-memory objects
20. [NoGraphs GitHub](https://github.com/HeWeMel/nographs) — Lazy graph analysis

### Temporal
21. [allen-ia GitHub](https://github.com/psiwray/allen-ia) — Allen IA Python implementation
22. [QSRlib Docs](https://qsrlib.readthedocs.io/en/latest/rsts/handwritten/qsrs/allen.html) — Allen IA in QSR library
23. [THEANINE (NAACL 2025)](https://aclanthology.org/2025.naacl-long.435/) — Timeline-based memory management
24. [Temporal SPARQL Survey](https://link.springer.com/article/10.1007/s12145-021-00574-w) — RDF temporal data management

### Structured Retrieval
25. [Microsoft GraphRAG](https://github.com/microsoft/graphrag) — Hierarchical community-based GraphRAG
26. [LightRAG](https://github.com/HKUDS/LightRAG) — Dual-level graph-enhanced retrieval
27. [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) — PPR-based neurobiological retrieval
28. [PathRAG (AAAI 2025)](https://arxiv.org/abs/2502.14902) — Path-based pruning for GraphRAG
29. [KGGen (NeurIPS 2025)](https://github.com/stair-lab/kg-gen) — DSPy-based KG generation

### Entity Resolution
30. [Dedupe GitHub](https://github.com/dedupeio/dedupe) — Active learning entity resolution
31. [Splink GitHub](https://github.com/moj-analytical-services/splink) — Probabilistic record linkage
32. [Awesome Entity Resolution](https://github.com/OlivierBinette/Awesome-Entity-Resolution) — Curated list

### Research Papers
33. [A-RAG (arXiv:2602.03442)](https://arxiv.org/abs/2602.03442) — Hierarchical agentic retrieval
34. [A-MEM (arXiv:2502.12110)](https://arxiv.org/abs/2502.12110) — Zettelkasten agentic memory (NeurIPS 2025)
35. [MemWalker (arXiv:2310.05029)](https://arxiv.org/abs/2310.05029) — Tree-based memory navigation
36. [ReadAgent (arXiv:2402.09727)](https://arxiv.org/abs/2402.09727) — Gist memory (ICML 2024)
37. [MemOS EMNLP 2025](https://github.com/BAI-LAB/MemoryOS) — Hierarchical memory OS
38. [MemOS MemTensor (arXiv:2507.03724)](https://arxiv.org/abs/2507.03724) — MemCube abstraction
39. [LiCoMemory (arXiv:2511.01448)](https://arxiv.org/abs/2511.01448) — CogniGraph hierarchical memory
40. [Agentic RAG Survey (arXiv:2501.09136)](https://arxiv.org/abs/2501.09136) — Taxonomy of agentic RAG
41. [Graph-Based Agent Memory Survey (arXiv:2602.05665)](https://arxiv.org/html/2602.05665v1) — Graph memory survey

### Surveys & Lists
42. [Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG) — Curated GraphRAG list
43. [Awesome-GraphMemory](https://github.com/DEEP-PolyU/Awesome-GraphMemory) — Curated graph memory list
44. [Agent Memory Paper List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List) — Comprehensive paper list
45. [Awesome-AI-Memory](https://github.com/IAAR-Shanghai/Awesome-AI-Memory) — AI memory knowledge base
46. [LoCoMo Benchmark](https://github.com/snap-research/locomo) — Long-term conversational memory benchmark
47. [Mem0 Paper (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — Mem0 architecture paper

---

## 12. OPEN QUESTIONS

1. **Should the query plan compile to Kuzu Cypher for graph operations?** Kuzu is embedded, MIT-licensed, and fast. Could serve as the execution backend while we provide the Python-native composable API on top.

2. **How to handle entity resolution without LLM calls?** Graphiti uses LLM for entity resolution which is expensive. Splink's Fellegi-Sunter probabilistic model needs no training data — could this work for conversational entities?

3. **Is Allen IA composition tractable at scale?** Only certain subalgebras are tractable. Need to determine which Allen relations are needed for conversational memory and whether the constraint propagation stays polynomial.

4. **Should we support time-travel queries?** Graphiti's bi-temporal model allows "what was true as of date X?" This is powerful but adds complexity. Is it needed for conversational memory or is "current state + supersession history" sufficient?

5. **Aggregation semantics over graphs**: Neither Palantir's ObjectSet nor Django's QuerySet handle graph aggregation well. Gremlin's sideEffect steps and Cypher's COLLECT/UNWIND are closer models. Need to design aggregation that works naturally over entity sets connected by links.
