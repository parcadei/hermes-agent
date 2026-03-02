# Dream Architecture Redesign -- Behavioral Specification

Created: 2026-03-02
Author: architect-agent
Status: SPECIFICATION (pre-implementation)

## Motivation

The previous session proved that `nrem_replay_xb` (Hopfield attractor pull on
stored patterns) is **structurally incapable** of improving P@1:

- At high beta, the attractor IS the pattern itself (softmax concentrates on
  self) -- the operation is a no-op.
- At low beta, neighbors gain softmax weight -- the attractor becomes the
  cluster centroid -- causing centroid collapse that HURTS P@1.
- This is not a tuning issue. No parameter choice avoids both failure modes
  simultaneously.

Evidence: `test_capacity_boundary.py` Phase 4 demonstrates the full tradeoff
curve. At `beta_nrem=100`, displacement < 0.001. At `beta_nrem=5`, spread
compresses >5% per cycle with monotonically decreasing P@1.

The redesign replaces `nrem_replay_xb` with neuroscience-informed operations
grounded in three complementary theories:

| Theory | Operation | What it does |
|--------|-----------|--------------|
| SHY (Tononi) | `nrem_repulsion_xb` | Global downscaling -- push similar patterns apart, anchor important ones |
| Active Systems Consolidation (Born) | `nrem_prune_xb`, `nrem_merge_xb` | Episodic-to-semantic transfer -- prune near-duplicates, merge clusters into prototypes |
| REM Creativity (Lewis) | `rem_explore_cross_domain_xb` | PGO wave simulation -- discover cross-domain associations |
| REM Reverse Learning (Crick) | `rem_unlearn_xb` (unchanged) | Destabilize mixture states via pattern separation |

---

## 1. Types

### 1.1 DreamReport

```python
@dataclass(frozen=True)
class DreamReport:
    """Immutable report of a complete dream cycle.

    All fields describe the outcome of dream_cycle_xb. The patterns array
    may have fewer rows than the input (due to pruning and merging).
    """

    patterns: np.ndarray
    # (N_out, d) array of post-dream patterns. N_out <= N_in.
    # All rows are unit vectors.

    associations: list[tuple[int, int, float]]
    # Cross-domain associations discovered by REM-explore.
    # Each tuple: (idx_i, idx_j, similarity_score).
    # Indices are in OUTPUT pattern space (post-prune/merge).
    # similarity_score in [0.0, 1.0].

    pruned_indices: list[int]
    # Indices (in INPUT pattern space) that were removed by nrem_prune_xb.
    # Sorted ascending. No duplicates.

    merge_map: dict[int, list[int]]
    # Maps OUTPUT index -> list of INPUT indices that were merged into it.
    # Only contains entries for merged patterns (not pass-through patterns).
    # Each input index appears in exactly one merge group or is absent
    # (meaning it was kept as-is and its output index is deterministic).
```

**Invariants on DreamReport:**
- `len(patterns)` = N_in - len(pruned_indices) - sum(len(v)-1 for v in merge_map.values())
- `set(pruned_indices)` and `set(idx for group in merge_map.values() for idx in group)` are disjoint
- Every input index 0..N_in-1 is either: kept as-is, in pruned_indices, or in exactly one merge group
- All rows of `patterns` have L2 norm in [1.0 - 1e-6, 1.0 + 1e-6]

### 1.2 MemoryEntry (unchanged)

```python
@dataclass
class MemoryEntry:
    text: str
    embedding: np.ndarray
    importance: float = 0.5
    creation_time: float = 0.0
    access_count: int = 0
    tagged: bool = False
```

No changes to MemoryEntry. The `importance` field (already present) is the
signal consumed by the new NREM operations.

---

## 2. Behavioral Contracts

### 2.1 nrem_repulsion_xb

**Theory:** SHY (Synaptic Homeostasis Hypothesis, Tononi 2003). During NREM
slow-wave sleep, synaptic weights undergo global downscaling. Strong
connections survive; weak ones are pruned. In pattern space, this translates
to: push apart patterns that are too close, anchoring high-importance patterns
while moving low-importance ones.

```python
def nrem_repulsion_xb(
    patterns: np.ndarray,           # (N, d) unit vectors
    importances: np.ndarray,        # (N,) in [0.0, 1.0]
    eta: float = 0.01,              # repulsion step size
    min_sep: float = 0.3,           # cosine distance threshold
) -> np.ndarray:                    # (N, d) unit vectors
```

**Algorithm:**
1. For each pair (i, j) where `i < j`:
   a. Compute cosine similarity: `sim_ij = patterns[i] @ patterns[j]`
      (patterns are unit vectors, so dot product = cosine similarity)
   b. If `sim_ij > (1 - min_sep)` (i.e., cosine distance < min_sep):
      - Compute repulsion direction: `d = normalize(patterns[i] - patterns[j])`
      - If `importances[i] < 0.7`: move `patterns[i] += eta * d`
      - If `importances[j] < 0.7`: move `patterns[j] -= eta * d`
      - High-importance patterns (>= 0.7) are anchored -- not moved
2. Re-normalize ALL output vectors to unit norm.

**Contracts:**

| ID | Contract | Verification |
|----|----------|--------------|
| R1 | delta_min(output) >= delta_min(input) | Minimum pairwise cosine distance does not decrease |
| R2 | High-importance patterns unchanged | For all i where importances[i] >= 0.7: norm(output[i] - input[i]) < 1e-10 |
| R3 | All output vectors unit norm | For all i: abs(norm(output[i]) - 1.0) < 1e-6 |
| R4 | Output shape = input shape | output.shape == input.shape |
| R5 | Input not mutated | patterns array is not modified in place |

**Edge cases:**

| Condition | Behavior |
|-----------|----------|
| N <= 1 | Return copy of input (no pairs to repel) |
| importances is None or empty | Treat all patterns as low-importance (all movable) |
| All importances >= 0.7 | Return copy of input (nothing to move) |
| eta = 0 | Return copy of input (no movement) |
| Two identical patterns | d = zero vector; skip pair (avoid division by zero) |
| All pairs already separated | Return copy of input (no pairs exceed threshold) |

**Contract R1 subtlety:** Repulsion between pair (i,j) could bring i closer to
some third pattern k. The implementation must either: (a) use sufficiently
small eta that this cannot happen, or (b) iterate until convergence, or
(c) verify the postcondition and reduce eta if violated. Option (a) is
recommended for v1 with eta=0.01 on unit vectors where the maximum single-step
displacement is 0.01.

---

### 2.2 nrem_prune_xb

**Theory:** Active Systems Consolidation (Born 2010). During sleep, redundant
episodic traces are eliminated. Near-duplicate patterns consume capacity
without adding retrieval value. Pruning removes the lower-importance member
of each near-duplicate pair.

```python
def nrem_prune_xb(
    patterns: np.ndarray,           # (N, d) unit vectors
    importances: np.ndarray,        # (N,) in [0.0, 1.0]
    threshold: float = 0.95,        # cosine similarity threshold
) -> tuple[np.ndarray, list[int]]: # (pruned_patterns, kept_indices)
```

**Algorithm:**
1. Compute pairwise cosine similarities (or iterate over pairs).
2. Build a set of indices to remove:
   - For each pair (i, j) where `sim(i, j) > threshold`:
     - Mark the one with LOWER importance for removal.
     - Tie-break: remove the higher index (preserves insertion order).
   - Process greedily: once an index is marked for removal, skip it in
     future pair comparisons (avoid cascading removals from a single
     high-similarity cluster -- that is handled by nrem_merge_xb).
3. `kept_indices` = sorted list of indices NOT removed.
4. `pruned_patterns` = `patterns[kept_indices]`.

**Contracts:**

| ID | Contract | Verification |
|----|----------|--------------|
| P1 | len(output) <= len(input) | N_out <= N_in |
| P2 | Kept patterns preserved exactly | For each k in kept_indices: norm(output[position(k)] - input[k]) == 0 |
| P3 | No close pairs in output | For all i,j in output: sim(i,j) <= threshold (or i==j) |
| P4 | kept_indices sorted ascending | kept_indices == sorted(kept_indices) |
| P5 | kept_indices subset of range(N) | all(0 <= k < N for k in kept_indices) |
| P6 | Input not mutated | patterns array is not modified in place |

**Edge cases:**

| Condition | Behavior |
|-----------|----------|
| N <= 1 | Return (patterns.copy(), [0]) or (patterns.copy(), []) for N=0 |
| No pairs exceed threshold | Return (patterns.copy(), list(range(N))) |
| All patterns identical | Keep the one with highest importance (lowest index for ties) |
| threshold >= 1.0 | No pairs can exceed threshold; return all |
| threshold <= -1.0 | All pairs exceed threshold; keep only highest-importance |

**Note on P3:** The greedy removal ensures P3 holds. If patterns A, B, C are
mutually similar (all pairs > threshold), the greedy pass removes at most 2
of 3. After removing the lowest-importance ones, the surviving patterns must
be verified to satisfy P3. If not (because the greedy order missed a pair),
a second pass is required.

---

### 2.3 nrem_merge_xb

**Theory:** Active Systems Consolidation (Born 2010). Groups of similar
episodic memories are compressed into a single semantic prototype. This is
the episodic-to-semantic transfer that creates generalized knowledge from
specific instances.

```python
def nrem_merge_xb(
    patterns: np.ndarray,           # (N, d) unit vectors
    importances: np.ndarray,        # (N,) in [0.0, 1.0]
    threshold: float = 0.90,        # pairwise cosine similarity threshold
    min_group: int = 3,             # minimum group size to trigger merge
) -> tuple[np.ndarray, dict[int, list[int]]]:
    # Returns: (merged_patterns, merge_map)
```

**Algorithm:**
1. Build an adjacency graph: edge (i,j) iff `sim(i,j) > threshold`.
2. Find connected components of size >= min_group.
   - Use union-find or BFS/DFS.
   - Components of size < min_group are left as-is (individual patterns kept).
3. For each qualifying component (group):
   a. Compute centroid: `c = mean(patterns[group])`, then normalize to unit norm.
   b. Compute merged importance: `imp = min(max(importances[group]) + 0.1, 1.0)`.
   c. Replace the group with the single centroid pattern.
4. Build output array: non-merged patterns (in original order) plus merged
   centroids (appended in order of their smallest original index).
5. Build merge_map: `{new_idx: [old_idx_1, old_idx_2, ...]}` for each merged group.

**Output ordering convention:**
- Non-merged patterns appear first, in their original relative order.
- Merged centroids appear after, in order of `min(group_indices)`.
- This means output index 0..K-1 are non-merged (K = N - sum of group sizes + num groups ... simplified: K = number of non-merged + number of merge groups).

**Contracts:**

| ID | Contract | Verification |
|----|----------|--------------|
| M1 | Merged patterns are unit norm | For each merged centroid: abs(norm - 1.0) < 1e-6 |
| M2 | Partition property | Each input index 0..N-1 appears in exactly one of: (a) kept as-is, or (b) exactly one merge group in merge_map |
| M3 | Merged importance boosted | For each merge group: importance > max(importances[group]) |
| M4 | Merged importance capped | For each merge group: importance <= 1.0 |
| M5 | merge_map keys are valid output indices | All keys in merge_map are in range(len(output)) |
| M6 | merge_map values are valid input indices | All values in merge_map lists are in range(N_in) |
| M7 | Input not mutated | patterns array is not modified in place |

**Edge cases:**

| Condition | Behavior |
|-----------|----------|
| N < min_group | No merge possible; return (patterns.copy(), {}) |
| No connected component >= min_group | Return (patterns.copy(), {}) |
| All patterns in one component | Merge all into one centroid; output has 1 row |
| min_group = 1 | Every pattern is its own group -- effectively a no-op unless threshold causes merges of pairs |
| min_group = 2 | Pairs can merge (becomes similar to prune but produces centroid instead of keeping one) |
| threshold >= 1.0 | No edges in adjacency graph; no merges |
| All importances already at 1.0 | Merged importance = min(1.0 + 0.1, 1.0) = 1.0 |

---

### 2.4 rem_explore_cross_domain_xb

**Theory:** REM Creativity (Lewis & Durrant 2011). PGO waves during REM sleep
activate patterns across unrelated memory domains, enabling creative
associations. Unlike the existing `rem_explore_xb` which discovers
within-domain co-activation at low beta, this function explicitly samples
cross-cluster pairs and measures structural similarity via perturbation
response correlation.

```python
def rem_explore_cross_domain_xb(
    patterns: np.ndarray,               # (N, d) unit vectors
    labels_or_clusters: np.ndarray,     # (N,) integer cluster assignments
    n_probes: int = 100,                # number of cross-domain probes
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int, float]]:      # [(idx_i, idx_j, sim_score), ...]
```

**Algorithm:**
1. Identify unique clusters from `labels_or_clusters`.
2. If fewer than 2 clusters, return [].
3. For each of `n_probes` iterations:
   a. Sample two distinct clusters c1, c2.
   b. Sample one pattern index from each: i from c1, j from c2.
   c. Compute structural similarity via perturbation response correlation:
      - Generate a small random perturbation vector `eps` (unit norm, scaled by 0.01).
      - Compute response_i = `patterns @ (patterns[i] + eps)` - `patterns @ patterns[i]`
      - Compute response_j = `patterns @ (patterns[j] + eps)` - `patterns @ patterns[j]`
      - similarity = Pearson correlation of (response_i, response_j).
      - Clamp to [0, 1] (negative correlations are not associations).
   d. If similarity > significance_threshold (0.3): record (i, j, similarity).
4. Deduplicate: if pair (i,j) appears multiple times, keep the maximum similarity.
5. Sort by similarity descending.

**Contracts:**

| ID | Contract | Verification |
|----|----------|--------------|
| X1 | All pairs are cross-cluster | For each (i, j, _): labels[i] != labels[j] |
| X2 | Similarity scores in [0, 1] | For each (_, _, s): 0.0 <= s <= 1.0 |
| X3 | No self-pairs | For each (i, j, _): i != j |
| X4 | Indices valid | For each (i, j, _): 0 <= i < N and 0 <= j < N |
| X5 | Input not mutated | patterns array is not modified in place |
| X6 | Sorted descending by similarity | For consecutive entries: s_k >= s_{k+1} |

**Edge cases:**

| Condition | Behavior |
|-----------|----------|
| Fewer than 2 clusters | Return [] |
| n_probes = 0 | Return [] |
| N <= 1 | Return [] |
| All patterns in same cluster | Return [] (same as <2 clusters) |
| Cluster with single pattern | That pattern can still be sampled |

**Difference from existing `rem_explore_xb`:**
The existing function probes at low beta to find patterns that co-activate
in the same basin. It does not use cluster labels and discovers within-basin
associations. The new function explicitly crosses cluster boundaries and uses
perturbation-based structural similarity instead of softmax co-activation.
Both functions return the same type signature for associations.

---

### 2.5 rem_unlearn_xb (UNCHANGED)

```python
def rem_unlearn_xb(
    patterns: np.ndarray,
    beta: float,
    beta_unlearn: float | None = None,
    n_probes: int = 200,
    separation_rate: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
```

No changes. This function implements Crick & Mitchison's reverse learning
(1983) and is working correctly: it probes for mixture fixed points at
computed T_unlearn and pushes apart the pattern pairs that co-activate in
those mixtures. The proof derivation of beta_unlearn from EnergyGap.lean
is sound.

Retained as-is in the new dream cycle pipeline.

---

### 2.6 dream_cycle_xb (UPDATED)

**Old signature:**
```python
def dream_cycle_xb(
    patterns: np.ndarray,
    beta: float,
    tagged_indices: list[int] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
```

**New signature:**
```python
def dream_cycle_xb(
    patterns: np.ndarray,               # (N, d) unit vectors
    beta: float,                        # operational inverse temperature
    tagged_indices: list[int] | None = None,
    importances: np.ndarray | None = None,  # (N,) in [0.0, 1.0] -- NEW
    labels: np.ndarray | None = None,       # (N,) cluster assignments -- NEW (optional)
    seed: int | None = None,
) -> DreamReport:                       # NEW return type
```

**Pipeline (biological order):**

```
Input X (N, d)
  |
  v
[1] nrem_repulsion_xb(X, importances, eta=0.01, min_sep=0.3)
  |  Push apart close patterns; anchor high-importance ones.
  |  Output: X1 (N, d) -- same shape, increased separation.
  v
[2] nrem_prune_xb(X1, importances, threshold=0.95)
  |  Remove near-duplicates (cosine > 0.95).
  |  Output: X2 (N2, d) where N2 <= N, plus pruned_indices.
  v
[3] nrem_merge_xb(X2, importances2, threshold=0.90, min_group=3)
  |  Merge groups of 3+ similar patterns into centroids.
  |  Output: X3 (N3, d) where N3 <= N2, plus merge_map.
  v
[4] rem_unlearn_xb(X3, beta) -- UNCHANGED
  |  Destabilize mixture states via pattern separation.
  |  Output: X4 (N3, d) -- same shape.
  v
[5] rem_explore_cross_domain_xb(X4, labels4, n_probes, rng) -- or rem_explore_xb fallback
  |  Discover cross-domain associations.
  |  Output: associations list.
  v
DreamReport(patterns=X4, associations, pruned_indices, merge_map)
```

**Parameter derivation:**
- `importances`: if None, extract from tagged_indices: importance = 0.8 for
  tagged, 0.3 for untagged.
- `labels`: if None, compute cluster assignments via cosine similarity
  thresholding or skip cross-domain explore (fall back to rem_explore_xb).
- Repulsion eta=0.01, min_sep=0.3: conservative defaults.
- Prune threshold=0.95: only near-exact duplicates.
- Merge threshold=0.90, min_group=3: moderate similarity, requires cluster.

**Contracts:**

| ID | Contract | Verification |
|----|----------|--------------|
| DC1 | Output pattern count may differ from input | N_out = N - len(pruned_indices) - sum(len(g)-1 for g in merge_map.values()) |
| DC2 | DreamReport.pruned_indices in input indexing | All indices in range(N_in) |
| DC3 | DreamReport.merge_map maps output -> input | Keys valid in output, values valid in input |
| DC4 | All output patterns unit norm | Inherited from sub-operation contracts |
| DC5 | delta_min non-decreasing | delta_min(output) >= delta_min(input) |
| DC6 | Associations reference output indices | All indices in associations are in range(N_out) |
| DC7 | Input not mutated | patterns array is not modified in place |

**Edge cases:**

| Condition | Behavior |
|-----------|----------|
| N = 0 | Return DreamReport(empty array, [], [], {}) |
| N = 1 | Repulsion/prune/merge are no-ops; unlearn is no-op; explore returns []; report reflects single pattern |
| All importances >= 0.7 | Repulsion anchors everything (no-op); prune/merge still operate on similarity |
| importances = None, tagged_indices = None | Default: all patterns importance=0.5, all tagged |

**Index mapping note:** Because prune and merge change N, the indices reported
in merge_map must be translated. The pipeline tracks an index remapping:

```
original indices: 0, 1, 2, 3, 4, 5, 6, 7
after prune (remove 2, 5): kept = [0, 1, 3, 4, 6, 7]
                           new positions: 0->0, 1->1, 3->2, 4->3, 6->4, 7->5
after merge (merge {2,3} -> centroid at position 2):
                           merge_map (in original): {new_idx: [3, 4]}
```

The DreamReport.pruned_indices always uses ORIGINAL indices.
The DreamReport.merge_map values always use ORIGINAL indices.
The DreamReport.merge_map keys use OUTPUT indices (post-prune, post-merge).

---

### 2.7 CoupledEngine.dream() (UPDATED)

**Old signature:**
```python
def dream(self, tagged_indices=None, seed=None) -> dict
```

**New signature:**
```python
def dream(self, tagged_indices=None, seed=None) -> dict
```

Signature unchanged (backward compatible), but internal behavior updated
to handle DreamReport.

**Updated algorithm:**
```python
def dream(self, tagged_indices=None, seed=None) -> dict:
    N = self.n_memories
    if N == 0:
        return {"modified": False, "associations": [], "n_tagged": 0,
                "pruned": 0, "merged": 0}

    if tagged_indices is None:
        tagged_indices = [i for i, m in enumerate(self.memory_store)
                          if m.importance >= 0.7]
    if not tagged_indices:
        tagged_indices = list(range(N))

    embeddings = self._embeddings_matrix()
    importances = np.array([m.importance for m in self.memory_store])

    report: DreamReport = dream_cycle_xb(
        embeddings, self.beta,
        tagged_indices=tagged_indices,
        importances=importances,
        seed=seed,
    )

    # --- Apply structural changes ---

    # 1. Build new memory store
    #    Need to map: which original entries survive, which are replaced
    #    by merged centroids, which are pruned.

    pruned_set = set(report.pruned_indices)
    merged_originals = set()
    for group in report.merge_map.values():
        merged_originals.update(group)

    # Sanity: pruned and merged are disjoint
    assert pruned_set.isdisjoint(merged_originals)

    new_store = []

    # Pass 1: keep non-pruned, non-merged entries with updated embeddings
    # (their embeddings were modified by repulsion + unlearn)
    output_idx = 0
    original_to_output = {}  # maps original index -> output row in report.patterns

    for orig_idx in range(N):
        if orig_idx in pruned_set or orig_idx in merged_originals:
            continue
        original_to_output[orig_idx] = output_idx
        entry = self.memory_store[orig_idx]
        entry.embedding = report.patterns[output_idx]
        new_store.append(entry)
        output_idx += 1

    # Pass 2: add merged centroid entries
    for out_idx, group in sorted(report.merge_map.items()):
        # Create a new MemoryEntry for the merged centroid
        # Text: concatenate or summarize (use highest-importance entry's text)
        best_orig = max(group, key=lambda i: self.memory_store[i].importance)
        merged_entry = MemoryEntry(
            text=self.memory_store[best_orig].text,
            embedding=report.patterns[out_idx],
            importance=min(max(importances[g] for g in group) + 0.1, 1.0),
            creation_time=min(self.memory_store[g].creation_time for g in group),
            access_count=sum(self.memory_store[g].access_count for g in group),
            tagged=True,  # merged patterns are semantically important
        )
        new_store.append(merged_entry)
        output_idx += 1

    self.memory_store = new_store
    self._invalidate_cache()

    return {
        "modified": True,
        "n_tagged": len(tagged_indices),
        "associations": report.associations,
        "pruned": len(report.pruned_indices),
        "merged": len(report.merge_map),
        "n_before": N,
        "n_after": len(new_store),
    }
```

**Contracts:**

| ID | Contract | Verification |
|----|----------|--------------|
| CE1 | After dream, len(memory_store) may be < before | N_after = N_before - pruned - (merged_patterns - merge_groups) |
| CE2 | Memory entries for pruned patterns are removed | No entry in memory_store corresponds to a pruned original |
| CE3 | Memory entries for merged groups are replaced | Each merge group becomes exactly one entry with centroid embedding |
| CE4 | Merged entry importance > max of group | Invariant inherited from M3 |
| CE5 | Merged entry preserves provenance | creation_time = min(group), access_count = sum(group) |
| CE6 | Non-pruned, non-merged entries have updated embeddings | Embeddings modified by repulsion + unlearn |
| CE7 | Cache invalidated | _embeddings_cache and _W_cache are None after dream |

---

## 3. Global Invariants

These hold across the entire dream pipeline and must be verified by tests.

### 3.1 Unit Norm Invariant

**INV-NORM:** All patterns are unit vectors at every stage of the pipeline.

```
For all i, at every intermediate state X_k:
    abs(norm(X_k[i]) - 1.0) < 1e-6
```

Verification: each function's output passes a unit-norm check. The pipeline
asserts this between stages.

### 3.2 Separation Non-Decrease Invariant

**INV-SEP:** The minimum pairwise cosine distance never decreases.

```
delta_min(output) >= delta_min(input)
```

This is the critical safety invariant. The old nrem_replay_xb violated this
(centroid collapse reduces delta_min). The new pipeline maintains it because:
- Repulsion explicitly increases separation.
- Pruning removes close pairs (increases delta_min).
- Merging replaces close groups with a single centroid (removes close pairs).
- Unlearn pushes mixture-forming pairs apart (increases separation).

### 3.3 Pattern Ordering Invariant

**INV-ORDER:** Among patterns that are neither pruned nor merged, the relative
order from the input is preserved in the output.

```
If orig_i < orig_j and both are kept (not pruned, not merged):
    output_position(orig_i) < output_position(orig_j)
```

### 3.4 Importance Anchoring Invariant

**INV-ANCHOR:** High-importance patterns (importance >= 0.7) are not displaced
by repulsion.

```
For all i where importances[i] >= 0.7:
    nrem_repulsion_xb output[i] == input[i]  (within float tolerance)
```

Note: high-importance patterns CAN be pruned or merged if they are
near-duplicates of each other. The anchoring only applies to repulsion.

### 3.5 Proof Compatibility Invariant

**INV-PROOF:** Post-dream patterns satisfy the preconditions of the Lean proofs.

The following Lean theorems apply to post-dream X:
- `EnergyGap.lean`: energy_gap = beta^(-1) * log(N) -- valid for any X,beta,N
- `Capacity.lean`: local_minima guaranteed when N < exp(beta * delta) / (4 * beta * M^2)
- `BasinVolume.lean`: basin radius >= delta / 2
- `SpuriousStates.lean`: spurious states destabilized when delta > 2*log(N)/beta

Since the dream pipeline only increases delta (INV-SEP) and may decrease N
(pruning/merging), all capacity conditions become EASIER to satisfy post-dream.
This is a key design property.

---

## 4. Implementation Phases

### Phase 1: Types and Skeleton

**Files to create/modify:**
- `dream_ops.py`: Add `DreamReport` dataclass at module level (after `DreamParams`).
- `dream_ops.py`: Add function stubs for `nrem_repulsion_xb`, `nrem_prune_xb`,
  `nrem_merge_xb`, `rem_explore_cross_domain_xb` with docstrings and
  `raise NotImplementedError`.

**Acceptance:**
- [ ] `DreamReport` is frozen dataclass with correct fields
- [ ] All function stubs have correct signatures
- [ ] Existing tests still pass (no breakage)
- [ ] Type annotations are complete

**Estimated effort:** Small

### Phase 2: nrem_repulsion_xb + nrem_prune_xb

**Files to modify:**
- `dream_ops.py`: Implement both functions.

**Test file to create:**
- `test_dream_redesign.py`: Tests for contracts R1-R5, P1-P6, edge cases.

**Key tests:**
```python
def test_repulsion_increases_separation():
    # Generate patterns with some close pairs
    # Verify delta_min(output) >= delta_min(input)

def test_repulsion_anchors_high_importance():
    # Set some importances >= 0.7
    # Verify those patterns are unchanged

def test_prune_removes_near_duplicates():
    # Create patterns with cosine > 0.95
    # Verify lower-importance one is removed

def test_prune_no_close_pairs_in_output():
    # After pruning, no pair in output exceeds threshold
```

**Dependencies:** Phase 1
**Estimated effort:** Medium

### Phase 3: nrem_merge_xb

**Files to modify:**
- `dream_ops.py`: Implement function.
- `test_dream_redesign.py`: Add tests for contracts M1-M7.

**Key tests:**
```python
def test_merge_partition_property():
    # Every input index accounted for exactly once

def test_merge_centroid_is_unit_norm():
    # All merged centroids have norm 1

def test_merge_importance_boosted():
    # Merged importance > max of group

def test_merge_min_group_respected():
    # Groups smaller than min_group are not merged
```

**Dependencies:** Phase 2
**Estimated effort:** Medium

### Phase 4: rem_explore_cross_domain_xb

**Files to modify:**
- `dream_ops.py`: Implement function.
- `test_dream_redesign.py`: Add tests for contracts X1-X6.

**Key tests:**
```python
def test_cross_domain_pairs_are_cross_cluster():
    # Every returned pair spans different clusters

def test_cross_domain_scores_bounded():
    # All scores in [0, 1]

def test_cross_domain_fewer_than_2_clusters():
    # Returns [] when only one cluster
```

**Dependencies:** Phase 1 (independent of Phases 2-3)
**Estimated effort:** Medium

### Phase 5: dream_cycle_xb Integration

**Files to modify:**
- `dream_ops.py`: Update `dream_cycle_xb` to new signature and pipeline.
- `coupled_engine.py`: Update `CoupledEngine.dream()` to handle `DreamReport`.
- `test_dream_redesign.py`: Integration tests.

**Key tests:**
```python
def test_dream_cycle_returns_dream_report():
    # Verify return type is DreamReport

def test_dream_cycle_reduces_count_when_pruning():
    # With near-duplicates, output has fewer patterns

def test_dream_cycle_delta_min_non_decreasing():
    # End-to-end: delta_min does not decrease

def test_coupled_engine_dream_removes_pruned():
    # After dream(), memory_store shrinks by pruned count

def test_coupled_engine_dream_replaces_merged():
    # After dream(), merged groups become single entries
```

**Dependencies:** Phases 2, 3, 4
**Estimated effort:** Large

### Phase 6: Backward Compatibility and test_capacity_boundary.py Update

**Files to modify:**
- `test_capacity_boundary.py`: Update to exercise new dream pipeline. The
  NREM-beta-sweep tests (Phase 4 of that file) become the primary validation
  that the redesign eliminates the centroid-collapse failure mode.
- `coupled_engine.py`: Ensure `dream()` return dict is backward compatible
  (new keys added, old keys preserved).

**Key tests:**
```python
def test_dream_no_centroid_collapse():
    # After dream, intra-cluster spread does NOT decrease
    # (This is the key improvement over old nrem_replay_xb)

def test_dream_improves_or_preserves_p1():
    # P@1 after dream >= P@1 before dream (within tolerance)
    # (This was impossible with old nrem_replay_xb)
```

**Dependencies:** Phase 5
**Estimated effort:** Medium

---

## 5. Files Changed Summary

| File | Action | Description |
|------|--------|-------------|
| `dream_ops.py` | Modify | Add DreamReport, 4 new functions, update dream_cycle_xb |
| `coupled_engine.py` | Modify | Update CoupledEngine.dream() for DreamReport handling |
| `test_dream_redesign.py` | Create | All behavioral contract tests (new file) |
| `test_capacity_boundary.py` | Modify | Update for new pipeline, add centroid-collapse-free tests |

Files NOT changed:
- `nlcdm_core.py` -- no new primitives needed
- `dream_metrics.py` -- metrics functions remain valid
- `test_dream_validation.py` -- existing tests should pass (dream still works, just better)

---

## 6. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Repulsion causes new close pairs (with third patterns) | INV-SEP violated | Medium | Small eta (0.01), postcondition check, reduce eta if violated |
| Greedy pruning misses transitive pairs | P3 violated (close pair survives) | Low | Second-pass verification; if failed, re-run prune |
| Connected component finding is O(N^2) | Slow for large N | Low | N is bounded by capacity (~100s); acceptable |
| Merge centroid not representative | Merged pattern is poor retrieval target | Medium | Unit-norm centroid of unit vectors stays in the convex cone; test retrieval |
| Index remapping bugs in DreamReport | Wrong indices in merge_map/pruned_indices | High | Extensive index-tracking tests; property-based testing |
| Backward incompatibility in dream() return | Callers break | Low | Return dict with superset of old keys |
| rem_explore_cross_domain_xb requires cluster labels | CoupledEngine doesn't store labels | Medium | Derive labels from cosine similarity clustering in dream_cycle_xb; or make labels optional with fallback to rem_explore_xb |

---

## 7. Open Questions

- [ ] **Cluster label source for rem_explore_cross_domain_xb:** CoupledEngine does
  not currently track cluster labels. Options: (a) derive clusters via
  agglomerative clustering on cosine similarity inside dream_cycle_xb,
  (b) accept labels as parameter, (c) fall back to rem_explore_xb when
  labels unavailable. Recommendation: (a) with (c) as fallback.

- [ ] **Merged entry text policy:** When merging 3+ MemoryEntry objects into
  one, what happens to the text? Options: (a) keep highest-importance entry's
  text, (b) concatenate all texts, (c) store a summary marker like
  "[MERGED: 3 patterns]". Recommendation: (a) for v1, with metadata noting
  the merge.

- [ ] **Prune vs merge ordering:** Current spec runs prune (threshold=0.95)
  before merge (threshold=0.90). This means very close pairs are pruned first,
  then remaining similar groups are merged. Alternative: merge first, then
  prune remaining close pairs. Current ordering is preferred because pruning
  is simpler (no centroid computation) and handles the highest-similarity
  cases where one pattern is genuinely redundant.

- [ ] **Repulsion eta adaptation:** Should eta decrease as patterns approach
  their target separation? Or is fixed eta=0.01 sufficient? For v1, fixed
  eta is recommended. Adaptive eta is a v2 enhancement.

---

## 8. Success Criteria

1. **No centroid collapse:** After dream, intra-cluster spread does not decrease
   (the failure mode of old nrem_replay_xb is eliminated).

2. **P@1 preserved or improved:** Dream never degrades P@1 by more than 0.02
   at any capacity level (below wall, at wall, above wall).

3. **Capacity improvement:** By pruning near-duplicates and merging clusters,
   the effective capacity (patterns that maintain P@1=1.0) increases.

4. **delta_min non-decreasing:** The minimum pairwise cosine distance in the
   pattern store never decreases after a dream cycle.

5. **All behavioral contracts pass:** Every contract (R1-R5, P1-P6, M1-M7,
   X1-X6, DC1-DC7, CE1-CE7) has a passing test.

6. **Backward compatible:** Existing tests in test_dream_validation.py pass
   without modification (the dream() method still works, returns a dict with
   at least the old keys).

---

## 9. Proof Impact Assessment

The Lean proof system (HermesNLCDM/) establishes guarantees conditioned on:
- N < exp(beta * delta) / (4 * beta * M^2)  (Capacity.lean)
- delta > 0 (minimum pairwise cosine distance)
- beta > 0 (inverse temperature)
- All patterns unit norm (M^2 = 1)

The dream redesign's effect on these conditions:

| Parameter | Direction | Proof impact |
|-----------|-----------|-------------|
| N | May decrease (prune/merge) | Capacity condition EASIER to satisfy |
| delta | Non-decreasing (repulsion, prune, merge, unlearn) | Capacity condition EASIER; basin volume increases |
| beta | Unchanged | No impact |
| M^2 | Unchanged (unit vectors) | No impact |

**Conclusion:** The redesigned dream cycle can only make the proof conditions
easier to satisfy. It never degrades the theoretical guarantees. This is by
construction (INV-SEP + N-reduction).
