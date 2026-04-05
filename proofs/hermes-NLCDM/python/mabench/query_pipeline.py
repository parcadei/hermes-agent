"""Query pipeline: post-hop processing, merging, generation, and decomposition.

Converted from QueryPipelineMixin to standalone QueryPipeline class.
All self.* attribute access replaced with explicit config + instance fields.

Migration notes for callers:
  - QueryPipelineMixin is a backward-compat alias for QueryPipeline.
  - Mixin callers (hermes_agent.py) still work via MRO -- the alias
    keeps the import path stable.  HermesMemoryAgent.__init__ sets
    self.decompose_query / self.retrieve_num / etc. directly;
    the _resolve_* properties fall back to those when config is absent.
  - Standalone usage:
      from mabench.query_pipeline import QueryPipeline
      from mabench.configs import DecomposeConfig, RetrievalConfig, LLMConfig
      qp = QueryPipeline(
          engine=engine, scorer=scorer,
          retriever=retriever, conflict_resolver=cr,
          outgester=og, decompose_config=DecomposeConfig(enabled=True),
          retrieval_config=RetrievalConfig(), llm_config=LLMConfig(),
      )
"""

from __future__ import annotations

import time

from mabench.conflict_resolution import _cr_log


class QueryPipeline:
    """Standalone query pipeline: decompose, merge, generate.

    Can be used independently or composed into HermesMemoryAgent.

    When used as a mixin (no __init__ call), falls back to reading the
    old self.decompose_query / self.retrieve_num / etc. attributes
    set by the host class.
    """

    def __init__(
        self,
        engine=None,               # CoupledEngine
        scorer=None,               # EmbeddingRelevanceScorer
        retriever=None,            # Retriever
        conflict_resolver=None,    # ConflictResolver
        outgester=None,            # Outgester
        rebel=None,                # RebelKG
        orchestrator=None,         # MemoryOrchestrator (V1)
        bm25_index=None,           # BM25Index
        triadic=None,              # TriadicMemory
        chain_cache=None,          # ChainCache
        belief_index=None,         # BeliefIndex
        retrieval_config=None,     # RetrievalConfig
        cr_config=None,            # CRConfig
        decompose_config=None,     # DecomposeConfig
        dedup_config=None,         # DedupConfig
        llm_config=None,           # LLMConfig
        # Optional graph/hyperedge modules
        hyperedge=None,            # HyperedgeIntersection or None
        knowledge_graph=None,      # KnowledgeGraph or None
        # Callbacks
        hop_executor_fn=None,      # callable for _query_run_hop (from QueryEngine)
        graph_walk_fn=None,        # callable for _graph_walk_resolve (from QueryEngine)
        kg_hop_fn=None,            # callable for _resolve_triple_index_kg_hop (from QueryEngine)
    ) -> None:
        self._engine = engine
        self._scorer = scorer
        self._retriever = retriever
        self._conflict_resolver = conflict_resolver
        self._outgester = outgester
        self._rebel = rebel
        self._orchestrator = orchestrator
        self._bm25_index = bm25_index
        self._triadic = triadic
        self._chain_cache = chain_cache
        self._belief_index = belief_index
        self._retrieval_cfg = retrieval_config
        self._cr_cfg = cr_config
        self._decompose_cfg = decompose_config
        self._dedup_cfg = dedup_config
        self._llm_cfg = llm_config
        self._hyperedge = hyperedge
        self._knowledge_graph = knowledge_graph
        self._hop_executor_fn = hop_executor_fn
        self._graph_walk_fn = graph_walk_fn
        self._kg_hop_fn = kg_hop_fn
        # Lazy LLM client
        self._client = None
        # Query state (set per-query by QueryEngine)
        self._query_q_emb = None
        self._last_decompose_question = ""

    # ------------------------------------------------------------------
    # Config resolution: new config or old self.* fallback
    # Pattern: config object if standalone, raw self.X if mixin mode.
    # ------------------------------------------------------------------

    @property
    def _resolve_engine(self):
        if hasattr(self, "_engine") and self._engine is not None:
            return self._engine
        return self.coupled_engine  # type: ignore[attr-defined]

    @property
    def _resolve_orchestrator(self):
        if hasattr(self, "_orchestrator") and self._orchestrator is not None:
            return self._orchestrator
        return self.orchestrator  # type: ignore[attr-defined]

    @property
    def _resolve_decompose_query(self) -> bool:
        c = getattr(self, "_decompose_cfg", None)
        return c.enabled if c is not None else getattr(self, "decompose_query", False)

    @property
    def _resolve_decompose_rrf(self) -> bool:
        c = getattr(self, "_decompose_cfg", None)
        return c.rrf if c is not None else getattr(self, "decompose_rrf", False)

    @property
    def _resolve_decompose_coverage(self) -> bool:
        c = getattr(self, "_decompose_cfg", None)
        return c.coverage if c is not None else getattr(self, "decompose_coverage", False)

    @property
    def _resolve_decompose_max_hops(self) -> int:
        c = getattr(self, "_decompose_cfg", None)
        return c.max_hops if c is not None else getattr(self, "decompose_max_hops", 3)

    @property
    def _resolve_retrieve_num(self) -> int:
        c = getattr(self, "_retrieval_cfg", None)
        return c.retrieve_num if c is not None else getattr(self, "retrieve_num", 10)

    @property
    def _resolve_date_arithmetic(self) -> bool:
        """Whether to apply the deterministic date-arithmetic short-circuit."""
        c = getattr(self, "_retrieval_cfg", None)
        return c.date_arithmetic if c is not None else getattr(self, "date_arithmetic", False)

    @property
    def _resolve_date_arithmetic_confidence_threshold(self) -> str:
        """Minimum confidence required to bypass LLM ('high', 'medium', 'low')."""
        c = getattr(self, "_retrieval_cfg", None)
        if c is not None:
            return c.date_arithmetic_confidence_threshold
        return getattr(self, "date_arithmetic_confidence_threshold", "high")

    @property
    def _resolve_conflict_resolution(self) -> bool:
        c = getattr(self, "_cr_cfg", None)
        return c.enabled if c is not None else getattr(self, "conflict_resolution", False)

    @property
    def _resolve_cosine_dedup(self) -> bool:
        c = getattr(self, "_dedup_cfg", None)
        return c.enabled if c is not None else getattr(self, "cosine_dedup", False)

    @property
    def _resolve_supersession(self) -> bool:
        return getattr(self, "supersession", False)

    @property
    def _resolve_cooc_boost(self) -> bool:
        c = getattr(self, "_retrieval_cfg", None)
        return c.cooc_boost if c is not None else getattr(self, "cooc_boost_retrieval", False)

    @property
    def _resolve_bm25_hops(self) -> bool:
        if hasattr(self, "_bm25_index") and self._bm25_index is not None:
            return True
        return getattr(self, "bm25_hops", False)

    @property
    def _resolve_bm25_index(self):
        return getattr(self, "_bm25_index", None)

    @property
    def _resolve_triadic(self):
        return getattr(self, "_triadic", None)

    @property
    def _resolve_triadic_expand_k(self) -> int:
        return getattr(self, "_triadic_expand_k", 5)

    @property
    def _resolve_triadic_chain_hops(self) -> int:
        return getattr(self, "_triadic_chain_hops", 1)

    @property
    def _resolve_triadic_decay(self) -> float:
        return getattr(self, "_triadic_decay", 1.0)

    @property
    def _resolve_chain_cache(self):
        return getattr(self, "_chain_cache", None)

    @property
    def _resolve_belief_index(self):
        return getattr(self, "_belief_index", None)

    @property
    def _resolve_superseded_texts(self) -> set:
        return getattr(self, "_superseded_texts", set())

    @property
    def _resolve_model(self) -> str:
        c = getattr(self, "_llm_cfg", None)
        return c.model if c is not None else getattr(self, "model", "gpt-4o-mini")

    @property
    def _resolve_query_q_emb(self):
        return getattr(self, "_query_q_emb", None)

    @_resolve_query_q_emb.setter
    def _resolve_query_q_emb(self, value):
        self._query_q_emb = value

    # ------------------------------------------------------------------
    # Cross-module method resolvers
    # ------------------------------------------------------------------

    def _resolve_retrieve_v2(self, q):
        r = getattr(self, "_retriever", None)
        return r._retrieve_v2(q) if r is not None else self._retrieve_v2(q)  # type: ignore[attr-defined]

    def _resolve_query_run_hop(self, *args, **kwargs):
        fn = getattr(self, "_hop_executor_fn", None)
        return fn(*args, **kwargs) if fn is not None else self._query_run_hop(*args, **kwargs)  # type: ignore[attr-defined]

    def _resolve_query_time_cr(self, texts):
        cr = getattr(self, "_conflict_resolver", None)
        return cr._query_time_cr(texts) if cr is not None else self._query_time_cr(texts)  # type: ignore[attr-defined]

    def _resolve_pairwise_cosine_dedup(self, texts):
        cr = getattr(self, "_conflict_resolver", None)
        return cr._pairwise_cosine_dedup(texts) if cr is not None else self._pairwise_cosine_dedup(texts)  # type: ignore[attr-defined]

    def _resolve_coverage_audit(self, q, texts):
        cr = getattr(self, "_conflict_resolver", None)
        return cr._coverage_audit(q, texts) if cr is not None else self._coverage_audit(q, texts)  # type: ignore[attr-defined]

    def _resolve_format_outgestion(self, chunks, metadata):
        og = getattr(self, "_outgester", None)
        return og.format_outgestion(chunks, metadata) if og is not None else self._format_outgestion(chunks, metadata)  # type: ignore[attr-defined]

    def _resolve_generate_answer(self, q, ctx, **kw):
        og = getattr(self, "_outgester", None)
        return og.generate_answer(q, ctx, **kw) if og is not None else self._generate_answer(q, ctx, **kw)  # type: ignore[attr-defined]

    def _resolve_ensure_llm_client(self):
        og = getattr(self, "_outgester", None)
        if og is not None:
            og._ensure_client()
        else:
            self._ensure_llm_client()  # type: ignore[attr-defined]

    def _resolve_llm_client(self):
        og = getattr(self, "_outgester", None)
        if og is not None:
            og._ensure_client()
            return og._resolve_client
        if hasattr(self, "_client") and self._client is not None:
            return self._client
        self._ensure_llm_client()  # type: ignore[attr-defined]
        return self._llm_client  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Business logic methods
    # ------------------------------------------------------------------

    def _query_final_retrieval(
        self, bare_question: str, question: str,
    ) -> tuple[list[str], list[list[dict]], dict]:
        """Run decompose hops (if enabled) and merge with original retrieval.

        Returns (v2_texts, sq_chunks_raw, hop_state) where hop_state holds
        variables needed by _query_generate for chain caching.
        """
        from mabench.chain_cache import CachedChain

        hop_state: dict = {
            "subqueries": None,
            "answers": None,
            "evidence_chain": None,
            "final_chunks": None,
        }

        if self._resolve_decompose_query:
            v2_texts, sq_chunks_raw, hop_state = self._query_decompose_and_merge(
                bare_question,
            )
        else:
            # Single-hop: skip KG hop (too aggressive, causes false matches)
            v2_results = self._resolve_retrieve_v2(bare_question)
            v2_texts = [r["text"] for r in v2_results]
            sq_chunks_raw = []
            hop_state["_all_v2_results"] = list(v2_results)
        return v2_texts, sq_chunks_raw, hop_state

    def _query_decompose_and_merge(
        self, bare_question: str,
    ) -> tuple[list[str], list[list[dict]], dict]:
        """Decompose query into hops, run each, merge with original retrieval.

        Returns (v2_texts, sq_chunks_raw, hop_state).
        """
        subqueries = self._decompose_query(bare_question)

        # Graph walk: try to answer the full multi-hop question by traversing
        # the KG subgraph from the anchor entity. If the complete path exists,
        # this is deterministic and avoids cascade errors from independent hops.
        gw_fn = getattr(self, "_graph_walk_fn", None)
        if gw_fn is not None:
            gw_result = gw_fn(bare_question, subqueries)
            if gw_result is not None:
                gw_result["_all_v2_results"] = []
                return [], [], gw_result

        seen_texts: set[str] = set()
        answers: dict[str, str] = {}
        evidence_chain: list[dict] = []
        sq_chunks: list[list[str]] = []
        sq_chunks_raw: list[list[dict]] = []
        final_kg_answer = None

        for i, sq_template in enumerate(subqueries):
            kg_ans = self._resolve_query_run_hop(
                sq_template, i, len(subqueries),
                answers, evidence_chain, sq_chunks, sq_chunks_raw,
            )
            if kg_ans is not None:
                final_kg_answer = kg_ans

        hop_state: dict = {
            "subqueries": subqueries,
            "answers": answers,
            "evidence_chain": evidence_chain,
            "final_kg_answer": final_kg_answer,
            "seen_texts": seen_texts,
        }

        # KG short-circuit
        if final_kg_answer is not None:
            hop_state["_all_v2_results"] = []
            return [], sq_chunks_raw, hop_state

        # Original question retrieval + BM25 rerank
        orig_results = self._resolve_retrieve_v2(bare_question)
        if self._resolve_bm25_hops and self._resolve_bm25_index is not None:
            orig_results = self._bm25_rerank_orig(bare_question, orig_results)
        orig_texts = [r["text"] for r in orig_results]

        # Merge: RRF or priority
        if self._resolve_decompose_rrf:
            v2_texts = self._rrf_merge(orig_texts, sq_chunks)
        else:
            v2_texts = self._priority_merge(orig_results, sq_chunks, seen_texts)

        hop_state["_all_v2_results"] = list(orig_results)
        for sq_r in sq_chunks_raw:
            hop_state["_all_v2_results"].extend(sq_r)
        return v2_texts, sq_chunks_raw, hop_state

    def _bm25_rerank_orig(
        self, bare_question: str, orig_results: list[dict],
    ) -> list[dict]:
        """Rerank original retrieval results by BM25 for entity discrimination."""
        bm25_scores = self._resolve_bm25_index.score_texts(
            bare_question, [r["text"] for r in orig_results],
        )
        if any(s > 0 for s in bm25_scores):
            paired = sorted(
                zip(bm25_scores, orig_results), key=lambda x: -x[0],
            )
            return [r for _, r in paired]
        return orig_results

    def _rrf_merge(
        self, orig_texts: list[str], sq_chunks: list[list[str]],
    ) -> list[str]:
        """Reciprocal Rank Fusion across original + sub-query result lists."""
        rrf_k = 60
        rrf_scores: dict[str, float] = {}
        all_lists = [orig_texts] + sq_chunks
        for ranked_list in all_lists:
            for rank, text in enumerate(ranked_list):
                key = text.strip()[:200]
                if key not in rrf_scores:
                    rrf_scores[key] = 0.0
                rrf_scores[key] += 1.0 / (rrf_k + rank)

        key_to_text: dict[str, str] = {}
        for ranked_list in all_lists:
            for text in ranked_list:
                key = text.strip()[:200]
                if key not in key_to_text:
                    key_to_text[key] = text

        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True,
        )
        return [key_to_text[k] for k in sorted_keys[: self._resolve_retrieve_num]]

    def _priority_merge(
        self,
        orig_results: list[dict],
        sq_chunks: list[list[str]],
        seen_texts: set[str],
    ) -> list[str]:
        """Priority merge: orig top-5, then sub-queries top-3 each."""
        v2_texts: list[str] = []
        for r in orig_results[:5]:
            t = r["text"]
            if t not in seen_texts:
                seen_texts.add(t)
                v2_texts.append(t)
        for sq_text_list in sq_chunks:
            added = 0
            for t in sq_text_list:
                if added >= 3:
                    break
                if t not in seen_texts:
                    seen_texts.add(t)
                    v2_texts.append(t)
                    added += 1
        return v2_texts

    def _query_post_process(
        self,
        v2_texts: list[str],
        sq_chunks_raw: list[list[dict]],
        bare_question: str,
        question: str,
        hop_state: dict | None = None,
    ) -> tuple[list[str], dict[str, dict]]:
        """Post-retrieval: metadata, V1, triadic, coverage, CR, belief, supersession.

        Returns (merged_chunks, metadata_by_key).
        """
        # Build metadata lookup from all retrieval results
        all_results = hop_state.get("_all_v2_results", []) if hop_state else []
        metadata_by_key = self._build_metadata_lookup(all_results, sq_chunks_raw)

        # Merge + V1 + triadic + coverage
        merged_chunks = self._merge_and_expand(
            v2_texts, bare_question, question, metadata_by_key,
        )

        # Fact currency filtering
        merged_chunks = self._apply_currency_filter(merged_chunks)
        return merged_chunks, metadata_by_key

    def _build_metadata_lookup(
        self,
        all_results: list[dict],
        sq_chunks_raw: list[list[dict]],
    ) -> dict[str, dict]:
        """Build text-key to result-dict metadata lookup."""
        metadata_by_key: dict[str, dict] = {}
        for r in all_results:
            k = r["text"].strip()[:200]
            if k not in metadata_by_key:
                metadata_by_key[k] = r
        for sq_r in sq_chunks_raw:
            for r in sq_r:
                k = r["text"].strip()[:200]
                if k not in metadata_by_key:
                    metadata_by_key[k] = r
        return metadata_by_key

    def _merge_and_expand(
        self,
        v2_texts: list[str],
        bare_question: str,
        question: str,
        metadata_by_key: dict[str, dict],
    ) -> list[str]:
        """Merge V2 texts, add triadic expansion, V1, and coverage audit."""
        seen: set[str] = set()
        merged_chunks: list[str] = []
        for text in v2_texts:
            key = text.strip()[:200]
            if key not in seen:
                seen.add(key)
                merged_chunks.append(text)

        # Triadic expansion
        if self._resolve_triadic is not None and v2_texts:
            triadic_texts = self._resolve_triadic.expand(
                v2_texts,
                top_k=self._resolve_triadic_expand_k,
                max_hops=self._resolve_triadic_chain_hops,
                decay=self._resolve_triadic_decay,
            )
            for text in triadic_texts:
                key = text.strip()[:200]
                if key not in seen:
                    seen.add(key)
                    merged_chunks.append(text)

        # Coverage audit
        if self._resolve_decompose_coverage and not self._resolve_decompose_query:
            gap_texts = self._resolve_coverage_audit(bare_question, v2_texts)
            for text in gap_texts:
                key = text.strip()[:200]
                if key not in seen:
                    seen.add(key)
                    merged_chunks.append(text)

        # V1 text merge
        if self._resolve_cooc_boost:
            v1_context = ""
        else:
            v1_result = self._resolve_orchestrator.query(message=question)
            v1_context = v1_result.context if v1_result.context else ""

        if v1_context:
            for line in v1_context.split("\n---\n"):
                line = line.strip()
                key = line[:200]
                if line and key not in seen:
                    seen.add(key)
                    merged_chunks.append(line)

        return merged_chunks

    def _apply_currency_filter(self, merged_chunks: list[str]) -> list[str]:
        """Apply fact currency filtering: cosine dedup, CR, belief, supersession."""
        if self._resolve_cosine_dedup:
            merged_chunks = self._resolve_pairwise_cosine_dedup(merged_chunks)
        if self._resolve_conflict_resolution:
            merged_chunks = self._resolve_query_time_cr(merged_chunks)
        elif not self._resolve_cosine_dedup:
            belief = self._resolve_belief_index
            if belief is not None:
                merged_chunks = [
                    c for c in merged_chunks
                    if not belief.is_excluded(c)
                ]
            elif self._resolve_supersession and self._resolve_superseded_texts:
                superseded = self._resolve_superseded_texts
                merged_chunks = [
                    c for c in merged_chunks
                    if c not in superseded
                ]
        return merged_chunks

    def _query_generate(
        self,
        question: str,
        bare_question: str,
        merged_chunks: list[str],
        metadata_by_key: dict[str, dict],
        hop_state: dict,
        query_start: float,
    ) -> dict:
        """Format outgestion, generate LLM answer, store chain cache.

        Handles KG short-circuit and temporal short-circuit if applicable.
        """
        # KG short-circuit
        final_kg_answer = hop_state.get("final_kg_answer")
        if final_kg_answer is not None:
            qt = time.time() - query_start
            _q_emb = self._resolve_query_q_emb
            if self._resolve_chain_cache is not None and _q_emb is not None:
                self._resolve_chain_cache.store(
                    _q_emb, bare_question, final_kg_answer,
                    hop_state.get("evidence_chain", []),
                )
            return {
                "output": final_kg_answer,
                "input_len": 0,
                "output_len": 0,
                "memory_construction_time": qt,
                "query_time_len": qt,
                "trace": {
                    "subqueries": hop_state.get("subqueries"),
                    "answers": hop_state.get("answers"),
                    "evidence_chain": hop_state.get("evidence_chain"),
                    "final_kg_answer": final_kg_answer,
                    "final_context": None,
                    "final_chunks": [],
                    "outgestion_instructions": None,
                    "llm_response": final_kg_answer,
                },
            }

        # Temporal short-circuit DISABLED: 40% accuracy on full eval (10/25 correct).
        # Confidently returns wrong answers, bypassing LLM. Net negative.
        event_calendar = None  # getattr(self, "_event_calendar", None)
        if event_calendar:
            from mabench.event_calendar import (
                classify_temporal_question, resolve_temporal_question,
                parse_query_date as parse_temporal_query_date,
            )
            q_text = bare_question
            q_type = classify_temporal_question(q_text)
            if q_type != "unknown":
                q_date = parse_temporal_query_date(question)
                temporal_answer = resolve_temporal_question(
                    q_text, event_calendar, q_date,
                )
                if temporal_answer is not None:
                    qt = time.time() - query_start
                    return {
                        "output": temporal_answer,
                        "input_len": 0,
                        "output_len": 0,
                        "memory_construction_time": qt,
                        "query_time_len": qt,
                        "trace": {
                            "subqueries": hop_state.get("subqueries"),
                            "answers": hop_state.get("answers"),
                            "evidence_chain": hop_state.get("evidence_chain"),
                            "final_kg_answer": None,
                            "temporal_answer": temporal_answer,
                            "temporal_q_type": q_type,
                            "final_context": None,
                            "final_chunks": [],
                            "outgestion_instructions": None,
                            "llm_response": temporal_answer,
                        },
                    }

        final_chunks = merged_chunks[: self._resolve_retrieve_num]
        memory_context, outgestion_instructions = self._resolve_format_outgestion(
            final_chunks, metadata_by_key,
        )

        # Always-on temporal enrichment (neutral for non-temporal queries)
        # Gated on temporal_alpha > 0 so --temporal_alpha 0 disables everything temporal
        _temporal_alpha = getattr(getattr(self, '_retrieval', None), '_cfg', None)
        _temporal_alpha = getattr(_temporal_alpha, 'temporal_alpha', 0.01) if _temporal_alpha else 0.01
        from mabench.temporal import parse_query_date, enrich_with_temporal, format_temporal_context
        query_date = parse_query_date(question) if _temporal_alpha > 0 else None
        if query_date is not None:
            engine = self._resolve_engine
            text_to_idx = getattr(engine, '_text_to_store_idx', {})
            facts_for_enrichment = []
            for chunk_text in final_chunks:
                idx = text_to_idx.get(chunk_text)
                session_time = None
                if idx is not None and idx < len(engine.memory_store):
                    session_time = getattr(engine.memory_store[idx], 'session_time', None)
                facts_for_enrichment.append({"text": chunk_text, "session_time": session_time})
            enriched = enrich_with_temporal(facts_for_enrichment, query_date)
            temporal_ctx = format_temporal_context(enriched)
            memory_context = temporal_ctx
            outgestion_instructions = (
                "Temporal context for each fact is shown in brackets [date, N days before query]. "
                "Use pre-computed time differences to answer temporal questions accurately. "
                + outgestion_instructions
            )

        memory_time = time.time() - query_start

        # date_arithmetic short-circuit (feature-flagged). Runs BEFORE the LLM
        # call so we can bypass generation for deterministic date-diff
        # questions ("how many weeks ago", "on what date"). Neutral when
        # flag is off or the classifier returns "none" or confidence is low.
        if self._resolve_date_arithmetic:
            from mabench.date_arithmetic import (
                answer_date_arithmetic, classify_date_arithmetic_question,
            )
            da_classification = classify_date_arithmetic_question(question)
            if da_classification != "none":
                answered = answer_date_arithmetic(question, final_chunks)
                threshold_map = {"high": 1.0, "medium": 0.3, "low": 0.25}
                min_conf = threshold_map.get(
                    self._resolve_date_arithmetic_confidence_threshold, 1.0,
                )
                if (
                    answered.computed_answer is not None
                    and answered.confidence >= min_conf
                ):
                    qt = time.time() - query_start
                    return {
                        "output": answered.computed_answer,
                        "input_len": 0,
                        "output_len": 0,
                        "memory_construction_time": memory_time,
                        "query_time_len": qt,
                        "trace": {
                            "subqueries": hop_state.get("subqueries"),
                            "answers": hop_state.get("answers"),
                            "evidence_chain": hop_state.get("evidence_chain"),
                            "final_kg_answer": None,
                            "date_arithmetic_answer": answered.computed_answer,
                            "date_arithmetic_classification": answered.classification,
                            "date_arithmetic_event_date": (
                                answered.event_date.isoformat()
                                if answered.event_date else None
                            ),
                            "date_arithmetic_confidence": answered.confidence,
                            "final_context": memory_context,
                            "final_chunks": list(final_chunks),
                            "outgestion_instructions": outgestion_instructions,
                            "llm_response": answered.computed_answer,
                        },
                    }

        answer, input_tokens, output_tokens, gen_time = self._resolve_generate_answer(
            question, memory_context,
            extra_system_instructions=outgestion_instructions,
        )
        query_time = time.time() - query_start

        # Chain cache store
        self._query_store_chain_cache(
            bare_question, final_chunks, answer, hop_state,
        )

        result = {
            "output": answer,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
        }

        # Trace data for debugging and analysis
        result["trace"] = {
            "subqueries": hop_state.get("subqueries"),
            "answers": hop_state.get("answers"),
            "evidence_chain": hop_state.get("evidence_chain"),
            "final_kg_answer": hop_state.get("final_kg_answer"),
            "final_context": memory_context,
            "final_chunks": list(final_chunks),
            "outgestion_instructions": outgestion_instructions,
            "llm_response": answer,
        }

        return result

    def _query_store_chain_cache(
        self,
        bare_question: str,
        final_chunks: list[str],
        answer: str,
        hop_state: dict,
    ) -> None:
        """Store completed decompose chain in cache for future reuse."""
        _q_emb = self._resolve_query_q_emb
        cc = self._resolve_chain_cache
        if cc is None or not self._resolve_decompose_query or _q_emb is None:
            return

        from mabench.chain_cache import CachedChain
        chain = CachedChain(
            query_embedding=_q_emb,
            query_text=bare_question,
            sub_queries=hop_state.get("subqueries", []),
            intermediate_answers=hop_state.get("answers", {}),
            evidence_facts=list(final_chunks),
            final_answer=answer,
            chain_id=len(cc),
        )
        cc.store(chain)

    # ------------------------------------------------------------------
    # _decompose_query
    # ------------------------------------------------------------------

    def _decompose_query(self, question: str) -> list[str]:
        """Decompose a multi-hop question into chained sub-queries.

        Uses the LLM to break compositional questions like
        "country of origin of sport played by X" into:
          1. "What sport did Christian Abbiati play?"
          2. "What is the country of origin of {answer_1}?"

        Sub-queries use {answer_N} placeholders for chaining.
        Returns a list of sub-query template strings.
        """
        self._last_decompose_question = question
        self._resolve_ensure_llm_client()

        prompt = (
            "Break this question into simple sub-questions that each require "
            "looking up a single fact. Start from the innermost entity and "
            "work outward. Use {answer_1}, {answer_2} etc. as placeholders "
            "for answers from previous sub-questions. Use the actual entity "
            "names from the question, not pronouns.\n\n"
            "Example:\n"
            "Q: What is the country of origin of the sport played by Christian Abbiati?\n"
            "1. What sport did Christian Abbiati play?\n"
            "2. What is the country of origin of {answer_1}?\n\n"
            "Example:\n"
            "Q: Where was the creator of the religion Karl Lueger followed born?\n"
            "1. What religion did Karl Lueger follow?\n"
            "2. Who created {answer_1}?\n"
            "3. Where was {answer_2} born?\n\n"
            f"Q: {question}\n"
        )

        try:
            client = self._resolve_llm_client()
            response = client.chat.completions.create(
                model=self._resolve_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You decompose multi-hop questions into chained "
                            "sub-questions. Use {answer_N} placeholders. "
                            "Be concise."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            text = response.choices[0].message.content or ""
        except Exception:
            return [question]

        return self._parse_subqueries(text)

    def _parse_subqueries(self, text: str) -> list[str]:
        """Parse numbered sub-questions from LLM response text."""
        subqs: list[str] = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for prefix in ("1.", "2.", "3.", "4.", "5.",
                           "1)", "2)", "3)", "4)", "5)", "-"):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line:
                subqs.append(line)
        subqs = subqs[: self._resolve_decompose_max_hops]
        return subqs if subqs else [self._last_decompose_question]

    # ------------------------------------------------------------------
    # _extract_short_answer
    # ------------------------------------------------------------------

    def _extract_short_answer(
        self,
        question: str,
        facts: list[str],
        evidence_chain: list[dict] | None = None,
    ) -> str:
        """Extract a short factual answer from retrieved facts.

        Used in iterative query decomposition to resolve intermediate
        sub-queries before substituting into the next sub-query template.
        """
        self._resolve_ensure_llm_client()
        context = "\n".join(facts[:10])

        history = ""
        if evidence_chain:
            lines = []
            for i, hop in enumerate(evidence_chain, 1):
                lines.append(
                    f'Hop {i}: "{hop["q"]}" -> "{hop["a"]}" '
                    f'(from: "{hop["evidence"][0][:120]}...")'
                    if hop["evidence"]
                    else f'Hop {i}: "{hop["q"]}" -> "{hop["a"]}"'
                )
            history = "Previous reasoning:\n" + "\n".join(lines) + "\n\n"

        try:
            client = self._resolve_llm_client()
            response = client.chat.completions.create(
                model=self._resolve_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the question using ONLY the provided facts. "
                            "Use the previous reasoning chain for context. "
                            "Give just the entity name, nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{history}Facts:\n{context}\n\n"
                            f"Question: {question}\nAnswer:"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=32,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return ""


# Backward-compat alias: HermesMemoryAgent inherits QueryPipelineMixin
QueryPipelineMixin = QueryPipeline
