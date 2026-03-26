"""Query orchestration: top-level _query and decompose hop execution.

Provides QueryMixin with the main query entry point and hop helpers:
- _extract_question: extract bare question from MABench prompt
- _query: top-level orchestrator (calls pipeline methods via self.*)
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
"""

from __future__ import annotations

import time

from mabench.conflict_resolution import _cr_log


class QueryMixin:
    """Mixin providing query orchestration and decompose hop execution.

    Requires the host class to provide:
    - self._scorer: EmbeddingRelevanceScorer with .embed(text)
    - self.coupled_engine: engine with memory_store
    - self._llm_client: OpenAI client (lazily initialised)
    - self._ensure_llm_client(): method to init LLM client
    - self.model: str, LLM model name
    - self.temperature: float
    - self._max_gen_tokens: int
    - self.decompose_query: bool
    - self.decompose_max_hops: int
    - self.retrieve_num: int
    - self.conflict_resolution: bool
    - self._cr_per_hop: bool
    - self.cosine_dedup: bool
    - self._belief_index: BeliefIndex or None
    - self.entity_retrieval: bool
    - self.cooc_boost_retrieval: bool
    - self._retrieve_v2(query): from RetrievalMixin
    - self._apply_post_retrieval(results): from RetrievalMixin
    - self._query_time_cr(texts): from ConflictResolutionMixin
    - self._pairwise_cosine_dedup(texts): from ConflictResolutionMixin
    - self._entity_text_search(query, top_k): from EntityMixin
    - self._entity_graph_expand(seed, top_k): from EntityMixin
    - self._entity_first_retrieve(query): from EntityMixin
    - self._rebel_kg_hop(query): from RebelKGMixin
    - self._query_final_retrieval(...): from QueryPipelineMixin
    - self._query_post_process(...): from QueryPipelineMixin
    - self._query_generate(...): from QueryPipelineMixin
    - self._extract_short_answer(...): from QueryPipelineMixin
    """

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
        v2_texts, sq_chunks_raw, hop_state = self._query_final_retrieval(
            bare_question, question,
        )

        # Phase 4: post-process (metadata, V1, triadic, coverage, CR, belief)
        merged_chunks, metadata_by_key = self._query_post_process(
            v2_texts, sq_chunks_raw, bare_question, question,
        )

        # Phase 5: format, generate, cache, return
        return self._query_generate(
            question, bare_question, merged_chunks, metadata_by_key,
            hop_state, query_start,
        )

    def _query_setup(self, bare_question: str, query_start: float) -> dict | None:
        """Check chain cache for a cached decompose result.

        Returns a response dict if cache hit, None otherwise.
        Also stores _q_emb on self for later cache storage.
        """
        self._query_q_emb = None
        if getattr(self, "_chain_cache", None) is not None and self.decompose_query:
            self._query_q_emb = self._scorer.embed(bare_question)
            cached = self._chain_cache.lookup(self._query_q_emb)
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
        if getattr(self, "_rebel_kg", None) is not None:
            kg_answer = self._rebel_kg_hop(sq)
            if kg_answer is not None:
                return self._query_hop_kg(
                    sq, kg_answer, hop_idx, num_subqueries,
                    answers, evidence_chain, sq_chunks, sq_chunks_raw,
                )

        # Retrieve + retry loop
        sq_all_texts, sq_all_results, entity_first_hit = self._query_hop_retrieve(sq)

        # Per-hop CR / dedup / belief
        sq_all_texts = self._query_hop_filter(sq_all_texts)

        sq_chunks.append(sq_all_texts)
        sq_chunks_raw.append(sq_all_results)

        # Extract intermediate answer for chaining
        if hop_idx < num_subqueries - 1 and sq_all_texts:
            short_ans = self._extract_short_answer(sq, sq_all_texts, evidence_chain)
            if short_ans:
                answers[f"{{answer_{hop_idx + 1}}}"] = short_ans
                evidence_chain.append({
                    "q": sq, "a": short_ans,
                    "evidence": sq_all_texts[:2],
                })
        return None

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
            if getattr(self, "_triadic", None) is not None and texts:
                for t in self._triadic.expand(texts, top_k=self._triadic_expand_k):
                    tkey = t.strip()[:200]
                    if tkey not in sq_seen:
                        sq_seen.add(tkey)
                        sq_all_texts.append(t)

            # Aggressive entity-first: skip retry
            if entity_first_hit and getattr(self, "entity_first_aggressive", False):
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

        if getattr(self, "entity_first", False) and self._gliner_entity_index:
            ef_results = self._entity_first_retrieve(current_sq)
            if ef_results is not None:
                sq_results = ef_results
                entity_first_hit = True

        if sq_results is None:
            if (getattr(self, "bm25_hops", False)
                    and getattr(self, "_bm25_index", None) is not None):
                bm25_hits = self._bm25_index.query(current_sq, top_k=self.retrieve_num * 3)
                sq_results = [
                    {"text": self.coupled_engine.memory_store[doc_id].text,
                     "score": score}
                    for doc_id, score in bm25_hits
                    if doc_id < len(self.coupled_engine.memory_store)
                ]
                sq_results = self._apply_post_retrieval(sq_results)
            else:
                sq_results = self._retrieve_v2(current_sq)

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
        self._ensure_llm_client()
        try:
            judge_resp = self._llm_client.chat.completions.create(
                model=self.model,
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
        skip = entity_first_hit and getattr(self, "entity_first_aggressive", False)
        if not self.entity_retrieval or skip:
            return

        ent_results = self._entity_text_search(sq, top_k=10)
        for r in ent_results:
            ekey = r["text"].strip()[:200]
            if ekey not in sq_seen:
                sq_seen.add(ekey)
                sq_all_texts.append(r["text"])
                sq_all_results.append(r)

        if ent_results:
            seed_texts = [r["text"] for r in ent_results]
            graph_results = self._entity_graph_expand(seed_texts, top_k=5)
            for r in graph_results:
                gkey = r["text"].strip()[:200]
                if gkey not in sq_seen:
                    sq_seen.add(gkey)
                    sq_all_texts.append(r["text"])
                    sq_all_results.append(r)

    def _query_hop_filter(self, sq_all_texts: list[str]) -> list[str]:
        """Apply per-hop CR, dedup, or belief filtering."""
        if self.cosine_dedup:
            sq_all_texts = self._pairwise_cosine_dedup(sq_all_texts)
        if self.conflict_resolution and self._cr_per_hop:
            sq_all_texts = self._query_time_cr(sq_all_texts)
        elif not self.cosine_dedup and self._belief_index is not None:
            sq_all_texts = [
                t for t in sq_all_texts
                if not self._belief_index.is_excluded(t)
            ]
        return sq_all_texts
