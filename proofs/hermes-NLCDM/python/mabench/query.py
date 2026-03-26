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
