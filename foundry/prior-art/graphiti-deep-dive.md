# Deep Dive: Graphiti / Zep — Bi-Temporal Knowledge Graph for Agent Memory

Generated: 2026-04-13
Source: arXiv:2501.13956, GitHub getzep/graphiti, DeepWiki, Neo4j blog

---

## Architecture Diagram (Text)

```
                         ┌─────────────────────────────────┐
                         │        Graphiti Core             │
                         │   graphiti_core/graphiti.py      │
                         └────────────┬────────────────────┘
                                      │
            ┌─────────────────────────┼──────────────────────────┐
            │                         │                          │
   ┌────────▼────────┐    ┌──────────▼──────────┐    ┌─────────▼─────────┐
   │ Episode Subgraph│    │ Semantic Entity      │    │ Community         │
   │ Gₑ              │    │ Subgraph Gₛ          │    │ Subgraph Gᶜ       │
   │                  │    │                      │    │                   │
   │ Raw events/msgs  │    │ Resolved entities +  │    │ Clustered entity  │
   │ JSON, conv logs  │◄──►│ Semantic edges with  │◄──►│ summaries via     │
   │ Timestamped      │    │ 4-timestamp model    │    │ label propagation │
   │ Ground truth     │    │ t_valid, t_invalid   │    │ (not Leiden)      │
   └──────────────────┘    │ t_created, t_expired │    └───────────────────┘
                           └──────────┬───────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
             ┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼──────┐
             │ ϕcos        │  │ ϕbm25       │  │ ϕbfs         │
             │ Cosine      │  │ Okapi BM25  │  │ Breadth-First│
             │ Semantic    │  │ Full-text   │  │ Graph        │
             │ Similarity  │  │ Search      │  │ Traversal    │
             └──────┬──────┘  └──────┬──────┘  └───────┬──────┘
                    │                │                  │
                    └────────────────┼──────────────────┘
                                     │
                              ┌──────▼──────┐
                              │ RRF Reranker│
                              │ No LLM at  │
                              │ query time  │
                              └─────────────┘
```

---

## Data Model

### Node Types
- **EpisodeNode**: Raw event/message. Fields: content, timestamp, source_description, group_id.
- **EntityNode**: Resolved semantic entity. Fields: name, entity_type, summary, embeddings, created_at, expired_at.
- **CommunityNode**: Cluster summary. Fields: summary, member_entity_ids.

### Edge Types (the critical model)
Every edge carries **four timestamps** implementing bi-temporal tracking:

| Timestamp | Timeline | Meaning |
|-----------|----------|---------|
| `t_created` | Transactional (T') | When the system first learned this fact |
| `t_expired` | Transactional (T') | When the system invalidated/retired this fact |
| `t_valid` | Event (T) | When this fact became true in the real world |
| `t_invalid` | Event (T) | When this fact ceased being true in the real world |

Edge fields: `fact` (string), `fact_embedding` (vector), `episodes` (source lineage), `weight`, `created_at`, `expired_at`, `valid_at`, `invalid_at`.

### Episodic Edges
Connect episodes to their extracted entity nodes. Bidirectional indices enable both forward (episode -> entities) and backward (entity -> source episodes) traversal. This is the lineage/provenance mechanism.

---

## Query Patterns

### Search API
Three search functions composed via RRF:

1. **ϕcos(query, k)** — Cosine similarity over entity/edge embeddings. Near-constant time via vector index.
2. **ϕbm25(query, k)** — Full-text BM25 over node names and fact strings. Near-constant time via text index.
3. **ϕbfs(seed_nodes, depth)** — BFS from matched nodes up to N hops. Surfaces contextual neighbors.

No LLM calls at query time. P95 latency: 300ms. Context tokens drop from 115K to 1,600.

### Temporal Queries
- "What is true now?" — Filter edges where `t_invalid IS NULL` (still valid).
- "What was true on date X?" — Filter edges where `t_valid <= X` AND (`t_invalid IS NULL` OR `t_invalid > X`).
- "What did the system know as of date X?" — Filter on transactional timeline: `t_created <= X` AND (`t_expired IS NULL` OR `t_expired > X`).

### Community Queries
Community detection via label propagation (not Leiden). Provides global summaries. Dynamically extensible — new data triggers incremental community updates rather than full recomputation.

---

## Key Code Patterns

### Source File Layout
```
graphiti_core/
├── graphiti.py                      # Main client, orchestrates ingestion + search
├── models.py                        # EntityNode, EpisodeNode, Edge Pydantic models
├── search/
│   └── search.py                    # Hybrid search (cosine + BM25 + BFS + RRF)
├── utils/maintenance/
│   ├── edge_operations.py           # extract_edges, resolve_extracted_edges,
│   │                                # resolve_edge_contradictions (INVALIDATION LOGIC)
│   ├── node_operations.py           # extract_nodes, resolve_extracted_nodes (ENTITY RES)
│   ├── graph_data_operations.py     # build_indices_and_constraints, retrieve_episodes
│   └── community_operations.py      # build_communities, update_community
└── prompts/
    ├── extract_edges.py             # LLM prompt for relationship extraction
    ├── dedupe_edges.py              # LLM prompt for edge deduplication
    └── dedupe_nodes.py              # LLM prompt for node deduplication
```

### Edge Invalidation Algorithm (`resolve_edge_contradictions`)

```python
# Pseudocode distilled from edge_operations.py

def resolve_edge_contradictions(resolved_edge, invalidation_candidates):
    """
    For each candidate that contradicts the new edge:
    - If candidate was already invalid before new edge became valid: skip
    - If candidate.valid_at < resolved_edge.valid_at: invalidate candidate
      (set candidate.invalid_at = resolved_edge.valid_at)
    - New information always wins on the transactional timeline
    """
    edges_to_invalidate = []
    for edge in invalidation_candidates:
        if edge.invalid_at and edge.invalid_at <= resolved_edge.valid_at:
            continue  # Already invalid before new fact
        if edge.valid_at < resolved_edge.valid_at:
            edge.invalid_at = resolved_edge.valid_at
            edges_to_invalidate.append(edge)
    return edges_to_invalidate
```

Key insight: **invalidation, not deletion**. Old facts are never removed from the graph. They get `t_invalid` set, preserving full temporal history.

### Entity Resolution Pipeline

1. Extract entities from episode text via LLM (+ last N messages for context).
2. Search existing nodes by name similarity (full-text + embedding).
3. LLM-based deduplication: "Is this new entity the same as any of these candidates?"
4. If match found: merge into existing node. If not: create new node.

The three-tier strategy: exact match -> fuzzy similarity -> LLM reasoning.

### Ingestion Pipeline

```python
# Simplified flow from graphiti.py

async def add_episode(self, episode_body, source, timestamp):
    # 1. Create episode node (ground truth)
    episode = EpisodeNode(content=episode_body, created_at=timestamp)

    # 2. Extract entities from episode text
    extracted_nodes = await extract_nodes(episode, context=last_n_messages)

    # 3. Resolve entities against existing graph
    resolved_nodes = await resolve_extracted_nodes(extracted_nodes, existing_nodes)

    # 4. Extract relationships (edges)
    extracted_edges = await extract_edges(episode, resolved_nodes)

    # 5. Resolve edges (dedup + invalidation)
    resolved_edges = await resolve_extracted_edges(extracted_edges, existing_edges)

    # 6. Detect and apply contradictions
    await resolve_edge_contradictions(resolved_edges, invalidation_candidates)

    # 7. Build episodic links (episode <-> entities)
    await build_episodic_edges(episode, resolved_nodes)

    # 8. Update communities
    await update_community(affected_communities)
```

### Prescribed vs. Learned Ontology

Graphiti supports both:
- **Prescribed**: Define entity/edge types upfront via Pydantic models. Schema enforced at extraction time.
- **Learned**: Let structure emerge from data. LLM extracts entity types and relationships dynamically.

---

## Limitations for Hermes Foundry

1. **Neo4j dependency**: Requires Neo4j (or FalkorDB/Kuzu) as graph backend. Heavyweight for an embedded library. No pure in-memory mode.

2. **LLM-dependent entity resolution**: Every ingestion triggers multiple LLM calls for entity extraction, deduplication, edge extraction, and contradiction detection. Expensive at scale. No lightweight/local fallback.

3. **No composable query language**: Search is `search(query, filters, num_results)` — a single function call. No lazy evaluation. No query composition. No set operations. No aggregation. Cannot express "all people Alice worked with who later moved to New York."

4. **No Allen interval algebra**: Temporal reasoning is limited to point-in-time validity windows. Cannot express "events that overlapped with Alice's tenure at Google" or "events that started before and finished after the meeting."

5. **Batch ingestion skips invalidation**: `add_episode_bulk()` does NOT perform edge invalidation or contradiction detection. Only single-episode ingestion does. This is a production footgun.

6. **No aggregation**: Cannot compute "count of jobs Alice had" or "average duration of employment" from the graph directly.

7. **Community detection is coarse**: Label propagation gives clusters, not fine-grained entity groupings. No typed community hierarchy.

---

## Steal List (what to adopt)

### S1. Bi-Temporal 4-Timestamp Model
Adopt the (t_valid, t_invalid, t_created, t_expired) model for edges/facts. This is the correct abstraction for conversational memory where facts evolve over time. Separate event time from system time.

**Adaptation for Hermes**: Apply to MSL relation objects. Each relation gets `valid_from`, `valid_until` (event timeline) and `created_at`, `superseded_at` (system timeline). Our existing supersession model maps to `superseded_at`.

### S2. Edge Invalidation Over Deletion
Never delete facts. Set `invalid_at` when superseded. This preserves history for "what was true before?" queries and provides full audit trail.

**Adaptation for Hermes**: Already partially implemented via our supersession edges. Formalize by adding `invalid_at` timestamp to all MSL relations.

### S3. Episode Subgraph as Ground Truth
Keep raw conversational data as immutable episodes with bidirectional lineage to derived entities and facts. This enables "show me the original conversation that produced this fact" provenance queries.

**Adaptation for Hermes**: Map to our STATED layer in LayeredMemory. Ensure every derived fact has a lineage pointer to its source conversation turn.

### S4. Hybrid Search (BM25 + Cosine + Graph Traversal)
The three-way search with RRF reranking and NO LLM at query time is exactly right for low-latency retrieval.

**Adaptation for Hermes**: We already have BM25 + cosine. Add graph traversal as a third retrieval channel. Use RRF to merge.

### S5. Prescribed + Learned Ontology via Pydantic
Allow both schema-first (define entity types) and schema-free (let types emerge from data) approaches.

**Adaptation for Hermes**: Our faceted ontology (9 universal facets) is the "prescribed" part. Allow additional types to be learned from data.

---

## Anti-Patterns (what to avoid)

### A1. LLM for Entity Resolution
Graphiti's LLM-based entity resolution is its Achilles heel. At scale (10K+ entities), LLM calls per ingestion become prohibitive. Graphiti acknowledged this in the paper.

**Instead**: Use embedding similarity + string matching + Fellegi-Sunter probabilistic model (from Splink). Reserve LLM for ambiguous cases only.

### A2. Batch Ingestion Without Invalidation
`add_episode_bulk()` skipping contradiction detection is a design smell. In production, bulk-loaded data silently creates stale/contradictory edges.

**Instead**: Always run contradiction detection, even in batch mode. Use batch contradiction detection (compare new edges against each other + existing graph) rather than per-edge.

### A3. Community Detection via Label Propagation
Label propagation is fast but produces unstable communities that shift with insertion order. Leiden or Louvain are more stable for static analysis.

**Instead**: For an evolving memory graph, use incremental community maintenance that recomputes only affected clusters, not full relabeling.

---

## Key Metrics

| Metric | Value |
|--------|-------|
| DMR Benchmark | 94.8% (vs MemGPT 93.4%) |
| LongMemEval improvement | +18.5% over baseline |
| P95 retrieval latency | 300ms |
| Context tokens | 1,600 (vs 115,000 baseline) |
| Latency reduction | ~90% (2.5-3.2s vs 29-31s) |

---

## Sources

1. [Zep Paper (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956) — Full architecture paper
2. [Graphiti GitHub](https://github.com/getzep/graphiti) — Source code
3. [DeepWiki: Edge Extraction and Resolution](https://deepwiki.com/getzep/graphiti/5.3-edge-operations) — edge_operations.py analysis
4. [DeepWiki: Graphiti Core](https://deepwiki.com/getzep/graphiti/4.1-graphiti-core) — Core client analysis
5. [Neo4j Blog: Graphiti Knowledge Graph Memory](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/) — Architecture overview
6. [Zep Paper PDF](https://blog.getzep.com/content/files/2025/01/ZEP__USING_KNOWLEDGE_GRAPHS_TO_POWER_LLM_AGENT_MEMORY_2025011700.pdf) — Detailed PDF
