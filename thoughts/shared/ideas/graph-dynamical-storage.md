# Idea: Graph-Based Dynamical Storage for Hermes Memory

Date: 2026-02-28
Status: IDEA — not yet designed

---

## Core Insight

Instead of storing raw embeddings (N × 384 floats per memory), build a graph
from embedding similarity at write time, then discard the embeddings. Retrieval
happens via graph traversal + spreading activation, not cosine similarity search.

~97% storage reduction while leveraging the dynamical system we already built.

## How It Works

```
WRITE PATH:
  new memory → embed → compute similarity to existing nodes
  → add edges above threshold → store node + edges → discard embedding

READ PATH:
  query → find seed nodes (keyword/BM25/category routing)
  → spread activation through graph
  → dynamics layer (strength, importance, decay) scores candidates
  → return top-k
```

## Why This Fits Hermes

The proof system already implements:
- ACT-R spreading activation (recall.py)
- Strength decay over time (core.py)
- Importance feedback loops (optimizer.py)
- Category-based classification (encoding.py) — natural graph partitions

The dynamical system IS the retrieval engine. Vector search is redundant
if the graph topology captures semantic relationships.

## Entry Point Problem

Without stored embeddings, you can't do cosine sim against a new query.
Need seed selection for graph traversal. Options:

1. **Keyword/BM25 index** — lightweight text match finds seeds, graph-walk from there
2. **Quantized embeddings** — 1-bit/4-bit quantized (48 bytes vs 1.5 KB), still searchable
3. **Activation-based** — start from recently accessed nodes, spread (cold-start problem)
4. **Category routing** — encoding categories become graph partitions, query routes to partition first

Option 4 is compelling: encoding layer already classifies into 8 categories.
Query "what city?" → routes to fact subgraph → walk from there.

## Graph Structure

```
Nodes: memories (with dynamics state: strength, importance, activation)
Edges: semantic similarity (computed at write time, threshold-gated)
       temporal proximity (memories from same episode/session)
       co-activation (memories frequently retrieved together)

Partitions: encoding categories (fact, preference, correction, instruction, ...)
```

## Scale Considerations

- Single-user CLI: 100s-low 1000s memories → any approach works
- Graph architecture pays off at 10K+ memories
- Real value may be conceptual fit with dynamical system, not raw performance

## Open Questions

- What similarity threshold for edge creation? Too low = dense graph, too high = disconnected
- Should edges have weights that evolve with dynamics? (co-activation strengthening)
- How to handle the cold-start / first-query problem before graph has structure?
- Is quantized embeddings (option 2) a better 80/20 than full graph-only?
- Does Lean/formal verification extend naturally to graph invariants?

## Relation to V2

This could be the storage layer for the "coupled nonlinear dynamical system"
version (V2). V1 (current proof system) assumes traditional vector storage.
V2 could make the graph itself a state variable in the dynamics.
