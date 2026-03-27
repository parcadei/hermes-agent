"""KnowledgeGraph: rustworkx-based directed graph with Personalized PageRank retrieval.

Builds a directed knowledge graph from (subject, predicate, object) triples at ingestion time.
At query time, Personalized PageRank (PPR) traversal discovers multi-hop related entities
in a single computation, enabling graph-augmented retrieval for decompose hops.

Follows HippoRAG's node-specificity weighting: rare entities receive higher PPR seed weights
than common hub entities, improving precision on multi-hop chains.
"""

from __future__ import annotations

from collections import OrderedDict

import rustworkx as rx


class KnowledgeGraph:
    """Standalone knowledge graph with PPR-based multi-hop retrieval.

    Stores triples as bidirectional edges in a rustworkx PyDiGraph.
    Supports supersession (edge replacement) and orphan pruning.
    All entity names are normalized (lowercase, stripped).
    """

    def __init__(
        self,
        *,
        damping: float = 0.85,
        ppr_top_k: int = 20,
        max_facts_per_node: int = 50,
    ) -> None:
        self._graph: rx.PyDiGraph = rx.PyDiGraph()
        self._node_map: dict[str, int] = {}  # normalized entity -> node index
        self._node_facts: dict[int, OrderedDict[int, None]] = {}  # node -> ordered set of fact indices
        self._node_names: dict[int, str] = {}  # node index -> original entity name
        self._edge_predicates: dict[int, str] = {}  # edge index -> predicate
        self._edge_facts: dict[int, int] = {}  # edge index -> fact store index
        self._edge_recency: dict[int, float] = {}  # edge index -> recency
        self._n_triples: int = 0
        self._damping: float = damping
        self._ppr_top_k: int = ppr_top_k
        self._max_facts_per_node: int = max_facts_per_node

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalize(entity: str) -> str:
        """Normalize entity name: lowercase and strip whitespace."""
        return entity.strip().lower()

    def _get_or_create_node(self, normalized: str) -> int:
        """Return existing node index or create a new one."""
        if normalized in self._node_map:
            return self._node_map[normalized]
        node_idx = self._graph.add_node(normalized)
        self._node_map[normalized] = node_idx
        self._node_names[node_idx] = normalized
        self._node_facts[node_idx] = OrderedDict()
        return node_idx

    def _add_fact_to_node(self, node_idx: int, fact_index: int) -> None:
        """Add fact_index to node, evicting oldest if over capacity."""
        facts = self._node_facts[node_idx]
        if fact_index in facts:
            return
        facts[fact_index] = None
        while len(facts) > self._max_facts_per_node:
            # Pop oldest (first inserted)
            facts.popitem(last=False)

    def _node_fact_set(self, node_idx: int) -> set[int]:
        """Return the fact indices for a node as a plain set."""
        return set(self._node_facts.get(node_idx, {}).keys())

    def _add_edge(
        self, src: int, dst: int, predicate: str, fact_index: int, recency: float
    ) -> int:
        """Add a single directed edge with metadata."""
        edge_idx = self._graph.add_edge(src, dst, 1.0)
        self._edge_predicates[edge_idx] = predicate
        self._edge_facts[edge_idx] = fact_index
        self._edge_recency[edge_idx] = recency
        return edge_idx

    def _remove_edge_by_index(self, edge_idx: int) -> None:
        """Remove an edge and clean up its metadata."""
        self._graph.remove_edge_from_index(edge_idx)
        self._edge_predicates.pop(edge_idx, None)
        self._edge_facts.pop(edge_idx, None)
        self._edge_recency.pop(edge_idx, None)

    def _node_degree(self, node_idx: int) -> int:
        """Total degree (in + out) of a node."""
        try:
            return self._graph.in_degree(node_idx) + self._graph.out_degree(node_idx)
        except Exception:
            return 0

    def _prune_orphan(self, node_idx: int) -> None:
        """Remove node if it has degree 0 and no facts."""
        if self._node_degree(node_idx) == 0 and not self._node_facts.get(node_idx):
            # Find the normalized name for this node
            name = self._node_names.get(node_idx)
            if name and name in self._node_map and self._node_map[name] == node_idx:
                del self._node_map[name]
            self._node_names.pop(node_idx, None)
            self._node_facts.pop(node_idx, None)
            self._graph.remove_node(node_idx)

    def _find_matching_nodes(self, normalized: str) -> list[int]:
        """Find nodes matching an entity: exact first, then substring fallback."""
        # Exact match
        if normalized in self._node_map:
            return [self._node_map[normalized]]

        # Substring fallback
        matches = []
        for key, node_idx in self._node_map.items():
            if normalized in key or key in normalized:
                matches.append(node_idx)
        return matches

    def _reachable_from(self, start_nodes: set[int]) -> set[int]:
        """BFS to find all nodes reachable from start_nodes via directed edges."""
        visited: set[int] = set()
        queue = list(start_nodes)
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            try:
                for successor in self._graph.successor_indices(node):
                    if successor not in visited:
                        queue.append(successor)
            except Exception:
                pass
        return visited

    # ── Public API ───────────────────────────────────────────────────

    def add_fact(
        self,
        fact_text: str,
        fact_index: int,
        triples: list[tuple[str, str, str]],
        recency: float,
    ) -> int:
        """Add a fact's triples to the knowledge graph.

        For each triple (S, P, O):
        1. Normalize S and O (lowercase, strip)
        2. Find or create nodes for S and O
        3. Add fact_index to both nodes' fact sets (capped at max_facts_per_node)
        4. Add bidirectional edges S->O and O->S (both weight 1.0)
        5. Store predicate, fact_index, recency on each edge

        Parameters
        ----------
        fact_text : str
            Full text of the fact (for diagnostics only, not stored).
        fact_index : int
            Index of this fact in CoupledEngine.memory_store.
        triples : list of (subject, predicate, object)
            Triples extracted from this fact. May be empty.
        recency : float
            Recency value (higher = newer).

        Returns
        -------
        int
            Number of edges added (2 per triple: forward + reverse).
        """
        edges_added = 0

        for subj, pred, obj in triples:
            norm_subj = self._normalize(subj)
            norm_obj = self._normalize(obj)
            norm_pred = pred.strip().lower()

            subj_idx = self._get_or_create_node(norm_subj)
            obj_idx = self._get_or_create_node(norm_obj)

            self._add_fact_to_node(subj_idx, fact_index)
            self._add_fact_to_node(obj_idx, fact_index)

            # Forward edge: S -> O
            self._add_edge(subj_idx, obj_idx, norm_pred, fact_index, recency)
            # Reverse edge: O -> S
            self._add_edge(obj_idx, subj_idx, norm_pred, fact_index, recency)

            self._n_triples += 1
            edges_added += 2

        return edges_added

    def supersede(
        self,
        old_triple: tuple[str, str, str],
        new_triple: tuple[str, str, str],
        new_fact_index: int,
        new_recency: float,
    ) -> int:
        """Replace an old triple with a new one in the graph.

        1. Find old S and O nodes
        2. If either missing: treat as pure add for new_triple, return 0
        3. Find and remove edges between old S and O with matching predicate
        4. Remove old fact_index from node_facts
        5. Add new triple
        6. Prune orphan nodes (degree 0 AND empty facts)

        Returns
        -------
        int
            Number of edges removed.
        """
        old_subj, old_pred, old_obj = old_triple
        norm_old_subj = self._normalize(old_subj)
        norm_old_pred = old_pred.strip().lower()
        norm_old_obj = self._normalize(old_obj)

        # Check if old nodes exist
        old_subj_idx = self._node_map.get(norm_old_subj)
        old_obj_idx = self._node_map.get(norm_old_obj)

        if old_subj_idx is None or old_obj_idx is None:
            # Old triple not in graph: pure add
            self.add_fact("", new_fact_index, [new_triple], new_recency)
            return 0

        # Find edges to remove: both directions with matching predicate
        edges_to_remove: list[int] = []
        old_fact_indices: set[int] = set()

        for edge_idx in list(self._graph.edge_indices()):
            if edge_idx not in self._edge_predicates:
                continue
            if self._edge_predicates[edge_idx] != norm_old_pred:
                continue
            try:
                src, dst = self._graph.get_edge_endpoints_by_index(edge_idx)
            except Exception:
                continue
            # Match forward (old_subj -> old_obj) or reverse (old_obj -> old_subj)
            if (src == old_subj_idx and dst == old_obj_idx) or (
                src == old_obj_idx and dst == old_subj_idx
            ):
                edges_to_remove.append(edge_idx)
                old_fact_indices.add(self._edge_facts.get(edge_idx, -1))

        edges_removed = len(edges_to_remove)

        # Remove the edges
        for edge_idx in edges_to_remove:
            self._remove_edge_by_index(edge_idx)

        # Remove old fact indices from node_facts
        for fact_idx in old_fact_indices:
            if fact_idx < 0:
                continue
            for nidx in (old_subj_idx, old_obj_idx):
                facts = self._node_facts.get(nidx)
                if facts and fact_idx in facts:
                    del facts[fact_idx]

        # Add new triple
        self.add_fact("", new_fact_index, [new_triple], new_recency)

        # Prune orphan nodes (check old nodes)
        for nidx in (old_subj_idx, old_obj_idx):
            self._prune_orphan(nidx)

        return edges_removed

    def ppr_retrieve(
        self,
        seed_entities: list[str],
        *,
        top_k: int | None = None,
        damping: float | None = None,
    ) -> list[tuple[int, float]]:
        """Retrieve facts via Personalized PageRank from seed entities.

        1. Find matching nodes for each seed entity (exact then substring)
        2. Build personalization dict with node specificity weighting
        3. Run rustworkx.pagerank with personalization
        4. Rank non-seed nodes by score, take top_k
        5. Collect fact indices from top nodes
        6. Return deduplicated (fact_index, score) sorted by score descending

        Parameters
        ----------
        seed_entities : list of str
            Entity names to seed PPR from.
        top_k : int or None
            Number of top nodes to consider. Defaults to self._ppr_top_k.
        damping : float or None
            PPR damping factor. Defaults to self._damping.

        Returns
        -------
        list of (fact_index, ppr_score)
            Sorted by score descending, deduplicated by fact_index (max score wins).
        """
        if self._graph.num_nodes() == 0 or self._graph.num_edges() == 0:
            return []

        effective_top_k = top_k if top_k is not None else self._ppr_top_k
        effective_damping = damping if damping is not None else self._damping

        # Find seed nodes
        seed_node_set: set[int] = set()
        for entity in seed_entities:
            norm = self._normalize(entity)
            matches = self._find_matching_nodes(norm)
            seed_node_set.update(matches)

        if not seed_node_set:
            return []

        # Build personalization dict with node specificity weighting
        personalization: dict[int, float] = {}
        for node_idx in seed_node_set:
            n_facts = len(self._node_facts.get(node_idx, {}))
            weight = 1.0 / n_facts if n_facts > 0 else 1.0
            personalization[node_idx] = weight

        # Normalize weights to sum to 1.0
        total_weight = sum(personalization.values())
        if total_weight > 0:
            personalization = {k: v / total_weight for k, v in personalization.items()}

        # Find nodes reachable from seeds (to exclude disconnected components)
        reachable = self._reachable_from(seed_node_set)

        # Run PPR with increased max_iter for high damping values
        max_iter = 100 if effective_damping <= 0.9 else 500
        try:
            scores = rx.pagerank(
                self._graph,
                alpha=effective_damping,
                personalization=personalization,
                max_iter=max_iter,
            )
        except rx.FailedToConverge:
            # Retry with even more iterations and relaxed tolerance
            scores = rx.pagerank(
                self._graph,
                alpha=effective_damping,
                personalization=personalization,
                max_iter=2000,
                tol=1e-4,
            )

        # Rank non-seed nodes by score descending, filtered by reachability
        scored_nodes: list[tuple[int, float]] = []
        for node_idx, score in scores.items():
            if node_idx in seed_node_set:
                continue
            if node_idx not in reachable:
                continue
            if score <= 0:
                continue
            scored_nodes.append((node_idx, score))

        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scored_nodes[:effective_top_k]

        # Collect fact indices from top nodes, keeping max score per fact
        fact_scores: dict[int, float] = {}
        for node_idx, node_score in top_nodes:
            for fact_idx in self._node_facts.get(node_idx, {}):
                if fact_idx not in fact_scores or node_score > fact_scores[fact_idx]:
                    fact_scores[fact_idx] = node_score

        # Sort by score descending
        results = sorted(fact_scores.items(), key=lambda x: x[1], reverse=True)
        return results

    def ppr_retrieve_from_hop(
        self,
        intermediate_answer: str,
        sub_query: str,
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Retrieve facts via PPR seeded from an intermediate answer and sub-query.

        Extracts entities from both strings by matching tokens and n-grams
        against the node map, then runs PPR with all matched entities as seeds.

        Parameters
        ----------
        intermediate_answer : str
            Entity/answer from the previous decompose hop.
        sub_query : str
            The current sub-query being processed.
        top_k : int or None
            Number of top nodes to consider.

        Returns
        -------
        list of (fact_index, ppr_score)
            Same format as ppr_retrieve.
        """
        entities: set[str] = set()

        for text in (intermediate_answer, sub_query):
            extracted = self._extract_entities_from_text(text)
            entities.update(extracted)

        if not entities:
            return []

        return self.ppr_retrieve(list(entities), top_k=top_k)

    def _extract_entities_from_text(self, text: str) -> set[str]:
        """Extract known entities from text by matching against node_map.

        Tries: full string, individual tokens, 2-word and 3-word n-grams.
        """
        norm = self._normalize(text)
        found: set[str] = set()

        # Try full string
        if norm in self._node_map:
            found.add(norm)

        # Tokenize
        tokens = norm.split()

        # Individual tokens
        for token in tokens:
            if token in self._node_map:
                found.add(token)

        # 2-gram and 3-gram
        for n in (2, 3):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])
                if ngram in self._node_map:
                    found.add(ngram)

        # Substring fallback: check if any token is a substring of a node name
        # or a node name is a substring of a token
        if not found:
            for token in tokens:
                for key in self._node_map:
                    if token in key or key in token:
                        found.add(key)
                        break

        return found

    def find_node(self, entity: str) -> int | None:
        """Find a node by entity name. Exact match first, then substring fallback.

        Parameters
        ----------
        entity : str
            Entity name to search for.

        Returns
        -------
        int or None
            Node index if found, None otherwise.
        """
        norm = self._normalize(entity)

        # Exact match
        if norm in self._node_map:
            return self._node_map[norm]

        # Substring fallback
        for key, node_idx in self._node_map.items():
            if norm in key or key in norm:
                return node_idx

        return None

    @property
    def stats(self) -> dict[str, int]:
        """Return diagnostic counts for the knowledge graph.

        Returns
        -------
        dict with keys: nodes, edges, triples, facts_indexed
        """
        return {
            "nodes": len(self._node_map),
            "edges": self._graph.num_edges(),
            "triples": self._n_triples,
            "facts_indexed": sum(
                len(facts) for facts in self._node_facts.values()
            ),
        }

    # ── Serialization ────────────────────────────────────────────────

    def __getstate__(self) -> dict:
        """Pickle serialization: convert OrderedDicts to plain dicts for portability."""
        state = self.__dict__.copy()
        # Convert OrderedDicts to list-of-keys for deterministic serialization
        state["_node_facts_serial"] = {
            k: list(v.keys()) for k, v in self._node_facts.items()
        }
        del state["_node_facts"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Pickle deserialization: restore OrderedDicts from serialized form."""
        serial = state.pop("_node_facts_serial")
        state["_node_facts"] = {
            k: OrderedDict((fi, None) for fi in v) for k, v in serial.items()
        }
        self.__dict__.update(state)
