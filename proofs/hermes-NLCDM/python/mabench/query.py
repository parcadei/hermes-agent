"""Query orchestration: top-level _query and decompose hop execution.

Converted from QueryMixin to standalone QueryEngine class.
All self.* attribute access replaced with _resolve_* properties that
support both standalone (explicit __init__) and mixin mode (attributes
set by HermesMemoryAgent.__init__).

Provides QueryEngine with the main query entry point and hop helpers:
- _extract_question: extract bare question from MABench prompt
- _query: top-level orchestrator (calls pipeline methods via callbacks)
- _query_setup: check chain cache for cached decompose result
- _query_run_hop: execute a single decompose hop
- _query_hop_kg: handle a KG hit for a decompose hop
- _query_hop_retrieve: retrieve facts for a single decompose hop
- _query_hop_single_retrieve: single retrieval attempt
- _query_hop_judge: LLM judge for decompose hop
- _query_hop_entity_supplement: add entity-index facts
- _query_hop_filter: apply per-hop CR, dedup, or belief filtering

Post-hop processing, merging, generation, and decomposition are in
QueryPipelineMixin (query_pipeline.py).

Iterative multi-hop query is in IterativeQueryMixin (iterative_query.py).

Backward-compat alias:
  QueryMixin = QueryEngine
"""

from __future__ import annotations

import time

from mabench.conflict_resolution import _cr_log


class QueryEngine:
    """Standalone query orchestrator with decompose hop execution.

    Can be used independently or composed into HermesMemoryAgent.

    Usage::

        engine = QueryEngine(
            coupled_engine, scorer,
            retriever=retriever, conflict_resolver=resolver,
            entity_ops=entity_ops, rebel=rebel_kg,
            llm_config=LLMConfig(), retrieval_config=RetrievalConfig(),
            cr_config=CRConfig(), decompose_config=DecomposeConfig(),
            entity_config=EntityConfig(), dedup_config=DedupConfig(),
        )
        result = engine._query("What sport does Abbiati play?")
    """

    def __init__(
        self,
        engine=None,              # CoupledEngine
        scorer=None,              # EmbeddingRelevanceScorer
        retriever=None,           # Retriever instance (for _retrieve_v2)
        conflict_resolver=None,   # ConflictResolver instance
        entity_ops=None,          # EntityOps instance
        rebel=None,               # RebelKG instance
        outgester=None,           # Outgester instance (used by pipeline callbacks)
        bm25_index=None,          # BM25Index or None
        triadic=None,             # TriadicMemory or None
        chain_cache=None,         # ChainCache or None
        belief_index=None,        # BeliefIndex or None
        llm_config=None,          # LLMConfig from configs.py
        retrieval_config=None,    # RetrievalConfig
        cr_config=None,           # CRConfig
        decompose_config=None,    # DecomposeConfig
        entity_config=None,       # EntityConfig
        dedup_config=None,        # DedupConfig
        # Optional graph/hyperedge modules
        hyperedge=None,            # HyperedgeIntersection or None
        knowledge_graph=None,      # KnowledgeGraph or None
        cooc_kg=None,              # CoocKG or None
        supersession_state=None,   # SupersessionState (holds _triple_index)
        # Callbacks for cross-module deps (QueryPipeline methods)
        final_retrieval_fn=None,  # callable for _query_final_retrieval
        post_process_fn=None,     # callable for _query_post_process
        generate_fn=None,         # callable for _query_generate
        extract_short_answer_fn=None,  # callable for _extract_short_answer
    ) -> None:
        self._engine = engine
        self._scorer = scorer
        self._retriever = retriever
        self._conflict_resolver = conflict_resolver
        self._entity_ops = entity_ops
        self._rebel = rebel
        self._outgester = outgester
        self._bm25_index = bm25_index
        self._triadic = triadic
        self._chain_cache = chain_cache
        self._belief_index = belief_index
        self._llm_config = llm_config
        self._retrieval_cfg = retrieval_config
        self._cr_cfg = cr_config
        self._decompose_cfg = decompose_config
        self._entity_cfg = entity_config
        self._dedup_cfg = dedup_config
        self._hyperedge = hyperedge
        self._knowledge_graph = knowledge_graph
        self._cooc_kg = cooc_kg
        self._supersession_state = supersession_state
        # Callbacks
        self._final_retrieval_fn = final_retrieval_fn
        self._post_process_fn = post_process_fn
        self._generate_fn = generate_fn
        self._extract_short_answer_fn = extract_short_answer_fn
        # Lazy LLM client
        self._client = None
        # Query state
        self._query_q_emb = None

    # ------------------------------------------------------------------
    # Attribute resolution: standalone config or mixin self.* fallback
    # ------------------------------------------------------------------
    # Pattern: hasattr(self, '_engine') distinguishes standalone (has _engine
    # from __init__) from mixin mode (no __init__ called, uses raw self.X
    # set by HermesMemoryAgent.__init__).

    @property
    def _resolve_engine(self):
        """CoupledEngine -- from __init__ or host class self.coupled_engine."""
        if hasattr(self, "_engine") and self._engine is not None:
            return self._engine
        return self.coupled_engine  # type: ignore[attr-defined]

    @property
    def _resolve_scorer(self):
        """EmbeddingRelevanceScorer -- from __init__ or host class self._scorer."""
        if (hasattr(self, "_retrieval_cfg")
                and self._retrieval_cfg is not None
                and hasattr(self, "_scorer")
                and self._scorer is not None):
            return self._scorer
        return self._scorer  # type: ignore[has-type]

    @property
    def _resolve_decompose_query(self) -> bool:
        if hasattr(self, "_decompose_cfg") and self._decompose_cfg is not None:
            return self._decompose_cfg.enabled
        return getattr(self, "decompose_query", False)

    @property
    def _resolve_decompose_max_hops(self) -> int:
        if hasattr(self, "_decompose_cfg") and self._decompose_cfg is not None:
            return self._decompose_cfg.max_hops
        return getattr(self, "decompose_max_hops", 3)

    @property
    def _resolve_retrieve_num(self) -> int:
        if hasattr(self, "_retrieval_cfg") and self._retrieval_cfg is not None:
            return self._retrieval_cfg.retrieve_num
        return getattr(self, "retrieve_num", 10)

    @property
    def _resolve_conflict_resolution(self) -> bool:
        if hasattr(self, "_cr_cfg") and self._cr_cfg is not None:
            return self._cr_cfg.enabled
        return getattr(self, "conflict_resolution", False)

    @property
    def _resolve_cr_per_hop(self) -> bool:
        if hasattr(self, "_cr_cfg") and self._cr_cfg is not None:
            return self._cr_cfg.per_hop
        return getattr(self, "_cr_per_hop", True)

    @property
    def _resolve_cosine_dedup(self) -> bool:
        if hasattr(self, "_dedup_cfg") and self._dedup_cfg is not None:
            return self._dedup_cfg.enabled
        return getattr(self, "cosine_dedup", False)

    @property
    def _resolve_entity_retrieval(self) -> bool:
        if hasattr(self, "_entity_cfg") and self._entity_cfg is not None:
            return self._entity_cfg.entity_retrieval
        return getattr(self, "entity_retrieval", False)

    @property
    def _resolve_entity_first(self) -> bool:
        if hasattr(self, "_entity_cfg") and self._entity_cfg is not None:
            return self._entity_cfg.entity_first
        return getattr(self, "entity_first", False)

    @property
    def _resolve_entity_first_aggressive(self) -> bool:
        if hasattr(self, "_entity_cfg") and self._entity_cfg is not None:
            return self._entity_cfg.entity_first_aggressive
        return getattr(self, "entity_first_aggressive", False)

    @property
    def _resolve_bm25_hops(self) -> bool:
        # Standalone: infer from presence of BM25Index
        if hasattr(self, "_bm25_index") and self._bm25_index is not None:
            return True
        # Mixin fallback: check host class attribute
        return getattr(self, "bm25_hops", False)

    @property
    def _resolve_model(self) -> str:
        if hasattr(self, "_llm_config") and self._llm_config is not None:
            return self._llm_config.model
        return getattr(self, "model", "gpt-4o-mini")

    @property
    def _resolve_temperature(self) -> float:
        if hasattr(self, "_llm_config") and self._llm_config is not None:
            return self._llm_config.temperature
        return getattr(self, "temperature", 0.0)

    @property
    def _resolve_gliner_entity_index(self):
        """GLiNER entity index dict -- from entity_ops or host class."""
        if hasattr(self, "_entity_ops") and self._entity_ops is not None:
            return getattr(self._entity_ops, "_gliner_entity_index", {})
        return getattr(self, "_gliner_entity_index", {})

    @property
    def _resolve_triadic_expand_k(self) -> int:
        return getattr(self, "_triadic_expand_k", 5)

    # ------------------------------------------------------------------
    # Cross-module method resolution
    # ------------------------------------------------------------------

    def _resolve_retrieve_v2(self, q):
        if hasattr(self, "_retriever") and self._retriever is not None:
            return self._retriever._retrieve_v2(q)
        return self._retrieve_v2(q)  # type: ignore[attr-defined]

    def _resolve_apply_post_retrieval(self, r):
        if hasattr(self, "_retriever") and self._retriever is not None:
            return self._retriever._apply_post_retrieval(r)
        return self._apply_post_retrieval(r)  # type: ignore[attr-defined]

    def _resolve_cross_encoder_rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Apply cross-encoder reranking if available (for BM25 hops bypass)."""
        if hasattr(self, "_retriever") and self._retriever is not None:
            reranker = getattr(self._retriever, "_cross_encoder_reranker", None)
            if reranker is not None:
                return reranker.rerank(query, results)
        # Mixin fallback: check entity_reranker on self (old path)
        reranker = getattr(self, "_cross_encoder_reranker", None)
        if reranker is not None:
            return reranker.rerank(query, results)
        return results

    def _resolve_query_time_cr(self, texts):
        if hasattr(self, "_conflict_resolver") and self._conflict_resolver is not None:
            return self._conflict_resolver._query_time_cr(texts)
        return self._query_time_cr(texts)  # type: ignore[attr-defined]

    def _resolve_pairwise_cosine_dedup(self, texts):
        if hasattr(self, "_conflict_resolver") and self._conflict_resolver is not None:
            return self._conflict_resolver._pairwise_cosine_dedup(texts)
        return self._pairwise_cosine_dedup(texts)  # type: ignore[attr-defined]

    def _resolve_entity_text_search(self, q, top_k=10):
        if hasattr(self, "_entity_ops") and self._entity_ops is not None:
            return self._entity_ops._entity_text_search(q, top_k=top_k)
        return self._entity_text_search(q, top_k=top_k)  # type: ignore[attr-defined]

    def _resolve_entity_graph_expand(self, seed_texts, top_k=5):
        if hasattr(self, "_entity_ops") and self._entity_ops is not None:
            return self._entity_ops._entity_graph_expand(seed_texts, top_k=top_k)
        return self._entity_graph_expand(seed_texts, top_k=top_k)  # type: ignore[attr-defined]

    def _resolve_entity_first_retrieve(self, q):
        if hasattr(self, "_entity_ops") and self._entity_ops is not None:
            return self._entity_ops._entity_first_retrieve(q)
        return self._entity_first_retrieve(q)  # type: ignore[attr-defined]

    def _resolve_rebel_kg_hop(self, sq):
        if hasattr(self, "_rebel") and self._rebel is not None:
            return self._rebel._rebel_kg_hop(sq)
        return self._rebel_kg_hop(sq)  # type: ignore[attr-defined]

    @property
    def _resolve_triple_index(self):
        if self._supersession_state is not None:
            return self._supersession_state.triple_index
        # Mixin fallback
        return getattr(self, "_triple_index", {})

    @property
    def _resolve_subject_index(self):
        if self._supersession_state is not None:
            return self._supersession_state.subject_index
        return getattr(self, "_subject_index", {})

    def _resolve_triple_index_kg_hop(self, sub_question: str) -> str | None:
        """Resolve a decompose hop via subject-based KG lookup.

        Instead of matching both subject AND predicate as substrings,
        matches ONLY the subject entity, then returns ALL predicates
        as structured context.  For entities with few predicates (<= 1),
        returns the single fact so the decompose LLM can pick the right one.
        For multi-predicate entities, attempts predicate substring matching
        in the question text; falls through to retrieval if no predicate
        matches.

        Falls back to _resolve_triple_index_kg_hop_legacy when subject_index
        is empty but triple_index is populated.
        """
        subject_index = self._resolve_subject_index
        if not subject_index:
            # Fallback to legacy when subject_index not populated
            return self._resolve_triple_index_kg_hop_legacy(sub_question)

        sq_lower = sub_question.lower()

        # Find subject entities that appear in the question
        candidates: list[tuple[int, str, dict[str, tuple[str, str]]]] = []
        for subj, pred_dict in subject_index.items():
            if len(subj) >= 3 and subj in sq_lower:
                candidates.append((len(subj), subj, pred_dict))

        if not candidates:
            return None

        # Sort by subject length descending (prefer longest/most specific match)
        candidates.sort(key=lambda x: -x[0])
        best_subj, best_preds = candidates[0][1], candidates[0][2]

        # If only one predicate, return it directly
        if len(best_preds) == 1:
            obj, _text = next(iter(best_preds.values()))
            return obj

        # Multiple predicates: try to match predicate in question text
        # This handles the easy cases where predicate IS in the question
        pred_matches: list[tuple[int, str, str]] = []
        for pred, (obj, text) in best_preds.items():
            if len(pred) >= 3 and pred in sq_lower:
                pred_matches.append((len(pred), pred, obj))

        if pred_matches:
            pred_matches.sort(key=lambda x: -x[0])
            return pred_matches[0][2]  # best predicate match's object

        # No predicate substring match — fall back to cosine similarity.
        # Embed the sub-question, compare against each predicate's fact text.
        # This handles predicate splits where the LLM extracts different
        # predicate names for the same relation (e.g. "type of music" vs
        # "plays", "head of government" vs "head of government of").
        return self._predicate_cosine_fallback(sq_lower, best_preds)

    def _predicate_cosine_fallback(
        self,
        sq_lower: str,
        pred_dict: dict[str, tuple[str, str]],
        threshold: float = 0.35,
    ) -> str | None:
        """Pick the best predicate by cosine similarity between query and fact.

        When substring matching fails (e.g. query says "literary genre" but
        predicates are "type of music" / "plays"), embed the sub-question and
        each predicate's fact text, return the highest-cosine predicate's
        object if it exceeds the threshold.

        This is O(k) where k = number of predicates for the matched subject,
        typically 5-15.  Embeddings are cached by the scorer.
        """
        import numpy as np

        scorer = getattr(self, "_scorer", None) or getattr(self, "_resolve_scorer", None)
        if scorer is None or not hasattr(scorer, "embed"):
            return None

        q_emb = scorer.embed(sq_lower)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)

        best_score = -1.0
        best_obj = None
        for pred, (obj, fact_text) in pred_dict.items():
            f_emb = scorer.embed(fact_text)
            f_norm = f_emb / (np.linalg.norm(f_emb) + 1e-10)
            score = float(np.dot(q_norm, f_norm))
            if score > best_score:
                best_score = score
                best_obj = obj

        if best_score >= threshold:
            return best_obj
        return None

    def _graph_walk_resolve(
        self, bare_question: str, subqueries: list[str], max_hops: int = 4,
    ) -> dict | None:
        """Try to answer a multi-hop question by walking the KG subgraph.

        1. Find anchor entity from the first sub-query
        2. BFS from anchor through subject_index (up to max_hops)
        3. Build all reachable paths
        4. Ask LLM to pick the right path and extract the final answer

        Returns a hop_state dict if successful, None to fall through.
        """
        subject_index = self._resolve_subject_index
        if not subject_index or not subqueries:
            return None

        # Find anchor entity from first sub-query (longest substring match)
        sq0_lower = subqueries[0].lower()
        # Substitute any answer placeholders that exist
        for key in ["{answer_1}", "{answer_2}", "{answer_3}", "{answer_4}"]:
            sq0_lower = sq0_lower.replace(key, "")

        anchor = None
        anchor_len = 0
        for subj in subject_index:
            if len(subj) >= 3 and subj in sq0_lower and len(subj) > anchor_len:
                anchor = subj
                anchor_len = len(subj)

        if not anchor:
            return None

        # BFS to build reachable subgraph
        paths = []
        queue = [(anchor, [], 0)]
        visited = {anchor}
        while queue:
            entity, path, depth = queue.pop(0)
            if entity not in subject_index:
                continue
            for pred, (obj, _text) in subject_index[entity].items():
                new_path = path + [(entity, pred, obj)]
                paths.append(new_path)
                if depth + 1 < max_hops and obj in subject_index and obj not in visited:
                    visited.add(obj)
                    queue.append((obj, new_path, depth + 1))

        if not paths:
            return None

        # Format subgraph as structured text for the LLM
        lines = []
        for p in paths:
            chain = " → ".join(f"{s} --[{pr}]--> {o}" for s, pr, o in p)
            lines.append(chain)

        subgraph_text = "\n".join(lines)

        # Ask LLM to answer from the subgraph
        self._resolve_ensure_llm_client()
        prompt = (
            f"You are a fact lookup system. Below are all facts reachable from "
            f'the entity "{anchor}" in a knowledge graph.\n\n'
            f"Facts:\n{subgraph_text}\n\n"
            f"Question: {bare_question}\n\n"
            f"Using ONLY the facts above, trace the path that answers the "
            f"question. Give ONLY the final answer entity, nothing else. "
            f"If the facts don't contain enough information to answer, "
            f"respond with exactly: INSUFFICIENT"
        )

        try:
            response = self._resolve_llm_client.chat.completions.create(
                model=self._resolve_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            answer = response.choices[0].message.content.strip()
        except Exception:
            return None

        if not answer or "INSUFFICIENT" in answer.upper():
            return None

        # Verify the answer appears in the subgraph (prevent hallucination)
        answer_lower = answer.lower().strip().strip('"').strip("'")
        found_in_graph = any(
            answer_lower in p[-1][2].lower() or p[-1][2].lower() in answer_lower
            for p in paths
        )
        if not found_in_graph:
            return None

        # Build evidence chain from the matching path
        matching_path = None
        for p in paths:
            if answer_lower in p[-1][2].lower() or p[-1][2].lower() in answer_lower:
                matching_path = p
                break

        evidence_chain = []
        answers = {}
        for i, (s, pr, o) in enumerate(matching_path or []):
            evidence_chain.append({
                "q": f"{s} → {pr}?",
                "a": o,
                "evidence": [f"[KG-WALK: {s} --[{pr}]--> {o}]"],
            })
            answers[f"{{answer_{i + 1}}}"] = o

        return {
            "subqueries": subqueries,
            "answers": answers,
            "evidence_chain": evidence_chain,
            "final_kg_answer": answer_lower,
            "seen_texts": set(),
            "_all_v2_results": [],
        }

    def _resolve_triple_index_kg_hop_legacy(self, sub_question: str) -> str | None:
        """Legacy: resolve a decompose hop via predicated KG (_triple_index).

        Scans the triple index for (subject, predicate) keys where both
        the subject and predicate appear as substrings in the question.
        Returns the object value if found, None to fall through to retrieval.

        Kept as fallback when subject_index is not yet populated.
        """
        triple_index = self._resolve_triple_index
        if not triple_index:
            return None

        sq_lower = sub_question.lower()

        # Build candidate (subject, predicate) pairs where both appear in question
        candidates = []
        for (subj, pred), entries in triple_index.items():
            if not entries:
                continue
            if len(subj) >= 3 and len(pred) >= 3 and subj in sq_lower and pred in sq_lower:
                # Score by combined length (prefer longer/more specific matches)
                candidates.append((len(subj) + len(pred), subj, pred, entries))

        if not candidates:
            return None

        # Sort by score descending (longest combined match wins)
        candidates.sort(key=lambda x: -x[0])

        # Return the object from the best match (last entry = newest due to append order)
        best_entries = candidates[0][3]
        return best_entries[-1][0]  # newest object value

    def _resolve_cooc_kg_hop(self, sub_question: str) -> str | None:
        """Try to resolve a decompose hop via CoocKG.

        1. Find known KG entities that appear as substrings in the question.
        2. Infer target type from question keyword patterns.
        3. Try resolve_hop for each candidate entity (longest match first).
        """
        cooc_kg = self._cooc_kg
        if cooc_kg is None:
            return None

        # Extract query entity: find entity names that appear in the question
        # Use simple substring matching against known KG entities
        sq_lower = sub_question.lower()
        candidate_entities = []
        for entity_name in cooc_kg.entity_types:
            if entity_name in sq_lower:
                candidate_entities.append(entity_name)

        # Sort by length (prefer longer/more specific matches)
        candidate_entities.sort(key=len, reverse=True)

        # Infer target type from question keywords.
        # If we can't determine a target type, fall through to LLM —
        # untyped resolve_hop returns the top-1 neighbor regardless of type,
        # which produces garbage answers (e.g. "johann strauss ii" for "religion?").
        target_type = self._infer_target_type(sub_question)
        if target_type is None:
            return None

        # Try each candidate entity
        for entity in candidate_entities:
            answer = cooc_kg.resolve_hop(entity, target_type=target_type)
            if answer is not None:
                return answer

        return None

    @staticmethod
    def _infer_target_type(question: str) -> str | None:
        """Infer the expected answer type from question structure.

        Maps question keywords to GLiNER entity types (the 16 labels in
        ``DEFAULT_ENTITY_LABELS``).  Returns None if no pattern matches.

        Order matters: more specific patterns are checked first so that
        e.g. "capital city" matches "city" before "location".
        """
        q = question.lower()
        # Each entry: (keywords, entity_type)
        # Ordered from most specific to most general.  Domain-specific types
        # (animal, plant, food, disease, chemical, material) come BEFORE
        # generic geographic types (country, location) to avoid false
        # positives from substring overlaps (e.g. "national" matching
        # "nation" -> country, "region" -> location).
        TYPE_PATTERNS = [
            # Work of art (check before generic words like "book")
            (["painting", "sculpture", "novel", "poem", "song",
              "album", "film", "movie", "show", "series",
              "artwork", "composition", "symphony",
              "notable work", "wrote", "written"], "work of art"),
            # People (check before location/country to catch "who" and
            # role-specific keywords like "author", "player", etc.)
            (["who ", "player", "person", "author", "founder",
              "president", "leader", "director", "scientist",
              "artist", "musician", "actor", "writer",
              "performer", "singer", "composer"], "person"),
            # Animals (before country -- "national" substring)
            (["animal", "species", "creature", "breed", "mammal",
              "bird", "fish", "insect", "reptile"], "animal"),
            # Plants (before location -- "region" substring)
            (["plant", "flower", "tree", "herb", "crop", "grass",
              "shrub", "vine", "moss", "fern", "flora"], "plant"),
            # Diseases
            (["disease", "illness", "condition", "syndrome",
              "disorder", "infection", "virus", "pathogen",
              "ailment", "malady"], "disease"),
            # Chemicals
            (["chemical", "element", "compound", "substance",
              "molecule", "mineral", "metal", "alloy", "acid",
              "reagent"], "chemical"),
            # Food (before location -- "region" substring)
            (["food", "dish", "cuisine", "meal", "recipe",
              "ingredient", "beverage", "drink", "fruit",
              "vegetable"], "food"),
            # Materials
            (["material", "fabric", "textile", "cloth", "fiber",
              "polymer", "ceramic", "rubber", "leather",
              "concrete", "steel", "wood"], "material"),
            # Cities and capitals (check before "location")
            (["capital", "city", "town", "municipality", "village",
              "birthplace", "born in", "seat of government",
              "headquarters"], "city"),
            # Countries
            (["country", "nation ", "homeland", "sovereignty"], "country"),
            # Continents and locations
            (["continent", "region", "area", "located", "where",
              "place", "territory"], "location"),
            # Organizations and teams
            (["team", "club", "league", "company", "organization",
              "employer", "agency", "institution", "corporation",
              "university", "school", "education", "broadcaster",
              "manufacturer", "created by", "produced by"], "organization"),
            # Sports
            (["sport", "game", "athletics"], "sport"),
            # Languages
            (["language", "tongue", "dialect"], "language"),
            # Events
            (["event", "competition", "tournament", "ceremony",
              "festival", "conference", "war", "battle"], "event"),
            # Products
            (["product", "device", "gadget", "brand", "model",
              "software", "app", "tool", "book"], "product"),
        ]
        for keywords, entity_type in TYPE_PATTERNS:
            for kw in keywords:
                if kw in q:
                    return entity_type
        return None

    # Pipeline callbacks: resolve via explicit fn or mixin fallback

    def _resolve_query_final_retrieval(self, bare, question):
        if hasattr(self, "_final_retrieval_fn") and self._final_retrieval_fn is not None:
            return self._final_retrieval_fn(bare, question)
        return self._query_final_retrieval(bare, question)  # type: ignore[attr-defined]

    def _resolve_query_post_process(self, v2, sq, bare, question, hop_state=None):
        if hasattr(self, "_post_process_fn") and self._post_process_fn is not None:
            return self._post_process_fn(v2, sq, bare, question, hop_state=hop_state)
        return self._query_post_process(v2, sq, bare, question, hop_state=hop_state)  # type: ignore[attr-defined]

    def _resolve_query_generate(self, question, bare, merged, meta, hop_state, start):
        if hasattr(self, "_generate_fn") and self._generate_fn is not None:
            return self._generate_fn(question, bare, merged, meta, hop_state, start)
        return self._query_generate(  # type: ignore[attr-defined]
            question, bare, merged, meta, hop_state, start,
        )

    def _resolve_extract_short_answer(self, question, facts, evidence_chain):
        if hasattr(self, "_extract_short_answer_fn") and self._extract_short_answer_fn is not None:
            return self._extract_short_answer_fn(question, facts, evidence_chain)
        return self._extract_short_answer(question, facts, evidence_chain)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # LLM client resolution
    # ------------------------------------------------------------------

    def _resolve_ensure_llm_client(self):
        """Ensure LLM client is initialized (standalone or mixin)."""
        if hasattr(self, "_client"):
            # Standalone mode
            if self._client is not None:
                return
            import os
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            self._ensure_llm_client()  # type: ignore[attr-defined]

    @property
    def _resolve_llm_client(self):
        """OpenAI client -- from standalone _client or host class _llm_client."""
        if hasattr(self, "_client") and self._client is not None:
            return self._client
        return self._llm_client  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # _extract_question: staticmethod (no self.* access)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """Extract the bare question from a MABench prompt.

        The prompt looks like:
          "Pretend you are a knowledge management system. ... Question: ... Answer:"
        We extract just the text after the last "Question:" and before "Answer:".
        Falls back to the full prompt if the pattern isn't found.
        """
        q_start = prompt.rfind("Question:")
        if q_start >= 0:
            q_end = prompt.rfind("\nAnswer:")
            if q_end > q_start:
                return prompt[q_start + len("Question:") : q_end].strip()
        return prompt

    # ------------------------------------------------------------------
    # _query: top-level orchestrator (decomposed into 5 helpers)
    # ------------------------------------------------------------------

    def _query(self, question: str) -> dict:
        """Query memory with hybrid switching per proven criteria."""
        query_start = time.time()
        bare_question = self._extract_question(question)

        # Phase 1: setup + chain cache check
        cached_result = self._query_setup(bare_question, query_start)
        if cached_result is not None:
            return cached_result

        # Phase 2+3: decompose hops + final retrieval (or simple retrieval)
        v2_texts, sq_chunks_raw, hop_state = self._resolve_query_final_retrieval(
            bare_question, question,
        )

        # Phase 4: post-process (metadata, V1, triadic, coverage, CR, belief)
        merged_chunks, metadata_by_key = self._resolve_query_post_process(
            v2_texts, sq_chunks_raw, bare_question, question, hop_state,
        )

        # Phase 5: format, generate, cache, return
        return self._resolve_query_generate(
            question, bare_question, merged_chunks, metadata_by_key,
            hop_state, query_start,
        )

    def _query_setup(self, bare_question: str, query_start: float) -> dict | None:
        """Check chain cache for a cached decompose result.

        Returns a response dict if cache hit, None otherwise.
        Also stores _q_emb on self for later cache storage.
        """
        self._query_q_emb = None
        chain_cache = getattr(self, "_chain_cache", None)
        if chain_cache is not None and self._resolve_decompose_query:
            self._query_q_emb = self._resolve_scorer.embed(bare_question)
            cached = chain_cache.lookup(self._query_q_emb)
            if cached is not None:
                qt = time.time() - query_start
                return {
                    "output": cached.final_answer,
                    "input_len": 0,
                    "output_len": 0,
                    "memory_construction_time": qt,
                    "query_time_len": qt,
                }
        return None

    def _query_run_hop(
        self,
        sq_template: str,
        hop_idx: int,
        num_subqueries: int,
        answers: dict[str, str],
        evidence_chain: list[dict],
        sq_chunks: list[list[str]],
        sq_chunks_raw: list[list[dict]],
    ) -> str | None:
        """Execute a single decompose hop: substitute, retrieve, CR, extract.

        Returns a KG answer string if the hop was answered by REBEL KG
        and this is the last hop, otherwise returns None.
        """
        sq = sq_template
        for key, val in answers.items():
            sq = sq.replace(key, val)

        # Determine previous hop's intermediate answer (for hyperedge + KG PPR)
        prev_hop_answer = answers.get(f"{{answer_{hop_idx}}}") if hop_idx > 0 else None

        # LLM KG fast lane (predicated triple index, O(1) dict lookup)
        kg_answer = self._resolve_triple_index_kg_hop(sq)
        if kg_answer is not None:
            return self._query_hop_kg(
                sq, kg_answer, hop_idx, num_subqueries,
                answers, evidence_chain, sq_chunks, sq_chunks_raw,
            )

        # CoocKG fast lane (type-filtered graph traversal, no LLM)
        # DISABLED: 14% accuracy on real data (27/187 hops correct).
        # Co-occurrence edges are too noisy for confident hop resolution.
        # Needs: exact entity matching, NPMI threshold, top-1/top-2 gap check.
        cooc_kg = None  # getattr(self, "_cooc_kg", None)
        if cooc_kg is not None:
            kg_answer = self._resolve_cooc_kg_hop(sq)
            if kg_answer is not None:
                return self._query_hop_kg(
                    sq, kg_answer, hop_idx, num_subqueries,
                    answers, evidence_chain, sq_chunks, sq_chunks_raw,
                )

        # REBEL KG fast lane
        rebel_kg = self._resolve_rebel_kg_attr
        if rebel_kg is not None:
            kg_answer = self._resolve_rebel_kg_hop(sq)
            if kg_answer is not None:
                return self._query_hop_kg(
                    sq, kg_answer, hop_idx, num_subqueries,
                    answers, evidence_chain, sq_chunks, sq_chunks_raw,
                )

        # Retrieve + retry loop
        sq_all_texts, sq_all_results, entity_first_hit = self._query_hop_retrieve(sq)

        # KnowledgeGraph PPR retrieval: merge graph-discovered facts into results
        kg_module = getattr(self, "_knowledge_graph", None)
        if kg_module is not None and prev_hop_answer is not None:
            ppr_results = kg_module.ppr_retrieve_from_hop(
                intermediate_answer=prev_hop_answer,
                sub_query=sq,
            )
            if ppr_results:
                engine = self._resolve_engine
                store = engine.memory_store
                sq_seen = {t.strip()[:200] for t in sq_all_texts}
                for fact_idx, ppr_score in ppr_results:
                    if fact_idx < len(store):
                        text = store[fact_idx].text
                        tkey = text.strip()[:200]
                        if tkey not in sq_seen:
                            sq_seen.add(tkey)
                            sq_all_texts.append(text)
                            sq_all_results.append({
                                "text": text, "score": ppr_score,
                            })

        # Hyperedge intersection: penalize facts not mentioning the intermediate entity
        hyperedge = getattr(self, "_hyperedge", None)
        if hyperedge is not None and prev_hop_answer is not None and sq_all_results:
            import numpy as np
            scores = np.array([r.get("score", 0.0) for r in sq_all_results])
            store_size = len(self._resolve_engine.memory_store)
            adjusted, _matched = hyperedge.rerank_with_fuzzy_match(
                prev_hop_answer, scores, store_size,
            )
            # Re-sort results by adjusted scores
            paired = sorted(
                zip(adjusted.tolist(), sq_all_texts, sq_all_results),
                key=lambda x: -x[0],
            )
            sq_all_texts = [t for _, t, _ in paired]
            sq_all_results = [r for _, _, r in paired]

        pre_cr_texts = list(sq_all_texts)

        # Per-hop CR / dedup / belief
        sq_all_texts = self._query_hop_filter(sq_all_texts)

        sq_chunks.append(sq_all_texts)
        sq_chunks_raw.append(sq_all_results)

        # Extract intermediate answer + record trace
        self._query_hop_record(
            sq, hop_idx, num_subqueries, answers, evidence_chain,
            sq_all_texts, pre_cr_texts, entity_first_hit,
        )
        return None

    def _query_hop_record(
        self, sq, hop_idx, num_subqueries, answers, evidence_chain,
        sq_all_texts, pre_cr_texts, entity_first_hit,
    ) -> None:
        """Extract intermediate answer and record hop trace."""
        method = (
            "entity_first" if entity_first_hit
            else "bm25" if self._resolve_bm25_hops
            else "cosine"
        )
        if hop_idx < num_subqueries - 1 and sq_all_texts:
            short_ans = self._resolve_extract_short_answer(sq, sq_all_texts, evidence_chain)
            if short_ans:
                answers[f"{{answer_{hop_idx + 1}}}"] = short_ans
                evidence_chain.append({
                    "q": sq, "a": short_ans, "evidence": sq_all_texts[:2],
                    "retrieved_facts": pre_cr_texts,
                    "post_cr_facts": list(sq_all_texts),
                    "retrieval_method": method,
                })
        elif hop_idx == num_subqueries - 1:
            evidence_chain.append({
                "q": sq, "a": None, "evidence": sq_all_texts[:2],
                "retrieved_facts": pre_cr_texts,
                "post_cr_facts": list(sq_all_texts),
                "retrieval_method": method,
            })

    @property
    def _resolve_rebel_kg_attr(self):
        """Get the REBEL KG dict if available, else None."""
        if hasattr(self, "_rebel") and self._rebel is not None:
            return getattr(self._rebel, "_kg", None) or getattr(self._rebel, "_rebel_kg", None)
        return getattr(self, "_rebel_kg", None)

    def _query_hop_kg(
        self,
        sq: str,
        kg_answer: str,
        hop_idx: int,
        num_subqueries: int,
        answers: dict[str, str],
        evidence_chain: list[dict],
        sq_chunks: list[list[str]],
        sq_chunks_raw: list[list[dict]],
    ) -> str | None:
        """Handle a KG hit for a decompose hop.

        Returns the KG answer if this is the final hop, else None.
        """
        answers[f"{{answer_{hop_idx + 1}}}"] = kg_answer
        evidence_chain.append({
            "q": sq, "a": kg_answer,
            "evidence": [f"[KG: {kg_answer}]"],
        })
        sq_chunks.append([f"[KG] {sq} -> {kg_answer}"])
        sq_chunks_raw.append([])
        _cr_log.info(
            "KG hop %d/%d: %s -> %s",
            hop_idx + 1, num_subqueries, sq, kg_answer,
        )
        if hop_idx == num_subqueries - 1:
            return kg_answer
        return None

    def _query_hop_retrieve(self, sq: str) -> tuple[list[str], list[dict], bool]:
        """Retrieve facts for a single decompose hop with retry loop.

        Returns (all_texts, all_results, entity_first_hit).
        """
        sq_all_texts: list[str] = []
        sq_all_results: list[dict] = []
        sq_seen: set[str] = set()
        current_sq = sq
        max_sq_retries = 2
        entity_first_hit = False

        for retry in range(max_sq_retries):
            texts, results, ef_hit = self._query_hop_single_retrieve(
                current_sq, sq_seen,
            )
            sq_all_texts.extend(texts)
            sq_all_results.extend(results)
            if ef_hit:
                entity_first_hit = True

            # Triadic expansion
            triadic = getattr(self, "_triadic", None)
            if triadic is not None and texts:
                for t in triadic.expand(texts, top_k=self._resolve_triadic_expand_k):
                    tkey = t.strip()[:200]
                    if tkey not in sq_seen:
                        sq_seen.add(tkey)
                        sq_all_texts.append(t)

            # Aggressive entity-first: skip retry
            if entity_first_hit and self._resolve_entity_first_aggressive:
                break
            if retry >= max_sq_retries - 1:
                break

            # Judge: can we answer?
            should_break, new_sq = self._query_hop_judge(current_sq, sq_all_texts)
            if should_break:
                break
            current_sq = new_sq

        # Entity supplement (skip if aggressive entity-first fired)
        self._query_hop_entity_supplement(
            sq, sq_all_texts, sq_all_results, sq_seen,
            entity_first_hit,
        )
        return sq_all_texts, sq_all_results, entity_first_hit

    def _query_hop_single_retrieve(
        self, current_sq: str, sq_seen: set[str],
    ) -> tuple[list[str], list[dict], bool]:
        """Single retrieval attempt: entity-first, BM25, or cosine.

        Returns (new_texts, new_results, entity_first_hit).
        """
        entity_first_hit = False
        sq_results = None

        if self._resolve_entity_first and self._resolve_gliner_entity_index:
            ef_results = self._resolve_entity_first_retrieve(current_sq)
            if ef_results is not None:
                sq_results = ef_results
                entity_first_hit = True

        if sq_results is None:
            bm25_index = getattr(self, "_bm25_index", None)
            if self._resolve_bm25_hops and bm25_index is not None:
                engine = self._resolve_engine
                bm25_hits = bm25_index.query(current_sq, top_k=self._resolve_retrieve_num * 3)
                sq_results = [
                    {"text": engine.memory_store[doc_id].text,
                     "score": score}
                    for doc_id, score in bm25_hits
                    if doc_id < len(engine.memory_store)
                ]
                sq_results = self._resolve_apply_post_retrieval(sq_results)
                # Cross-encoder reranking (BM25 bypasses _retrieve_v2)
                sq_results = self._resolve_cross_encoder_rerank(current_sq, sq_results)
            else:
                sq_results = self._resolve_retrieve_v2(current_sq)

        new_texts: list[str] = []
        new_results: list[dict] = []
        for r in sq_results:
            key = r["text"].strip()[:200]
            if key not in sq_seen:
                sq_seen.add(key)
                new_texts.append(r["text"])
                new_results.append(r)
        return new_texts, new_results, entity_first_hit

    def _query_hop_judge(
        self, current_sq: str, sq_all_texts: list[str],
    ) -> tuple[bool, str]:
        """LLM judge: can we answer the sub-query? Returns (should_break, new_sq)."""
        self._resolve_ensure_llm_client()
        try:
            judge_resp = self._resolve_llm_client.chat.completions.create(
                model=self._resolve_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Given retrieved facts, can you answer "
                            "this specific sub-question?\n"
                            "If YES: respond ANSWER_READY\n"
                            "If NO: respond NEED: <3-6 word search "
                            "using concrete entity names>"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Sub-question: {current_sq}\n\n"
                            f"Facts:\n" + "\n".join(sq_all_texts[:10])
                            + "\n\nCan you answer?"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=64,
            )
            jt = (judge_resp.choices[0].message.content or "").strip()
        except Exception:
            return True, current_sq

        if jt.startswith("ANSWER_READY"):
            return True, current_sq
        elif "NEED:" in jt:
            need = jt[jt.index("NEED:") + 5 :].strip().strip("'\"")
            if need and need != current_sq:
                return False, need
        return True, current_sq

    def _query_hop_entity_supplement(
        self,
        sq: str,
        sq_all_texts: list[str],
        sq_all_results: list[dict],
        sq_seen: set[str],
        entity_first_hit: bool,
    ) -> None:
        """Add entity-index facts and graph expansion to hop results."""
        skip = entity_first_hit and self._resolve_entity_first_aggressive
        if not self._resolve_entity_retrieval or skip:
            return

        ent_results = self._resolve_entity_text_search(sq, top_k=10)
        for r in ent_results:
            ekey = r["text"].strip()[:200]
            if ekey not in sq_seen:
                sq_seen.add(ekey)
                sq_all_texts.append(r["text"])
                sq_all_results.append(r)

        if ent_results:
            seed_texts = [r["text"] for r in ent_results]
            graph_results = self._resolve_entity_graph_expand(seed_texts, top_k=5)
            for r in graph_results:
                gkey = r["text"].strip()[:200]
                if gkey not in sq_seen:
                    sq_seen.add(gkey)
                    sq_all_texts.append(r["text"])
                    sq_all_results.append(r)

    def _query_hop_filter(self, sq_all_texts: list[str]) -> list[str]:
        """Apply per-hop CR, dedup, or belief filtering."""
        if self._resolve_cosine_dedup:
            sq_all_texts = self._resolve_pairwise_cosine_dedup(sq_all_texts)
        if self._resolve_conflict_resolution and self._resolve_cr_per_hop:
            sq_all_texts = self._resolve_query_time_cr(sq_all_texts)
        elif not self._resolve_cosine_dedup:
            belief = getattr(self, "_belief_index", None)
            if belief is not None:
                sq_all_texts = [
                    t for t in sq_all_texts
                    if not belief.is_excluded(t)
                ]
        return sq_all_texts


# Backward-compat alias: HermesMemoryAgent inherits QueryMixin
QueryMixin = QueryEngine
