"""Query pipeline: post-hop processing, merging, generation, and decomposition."""

from __future__ import annotations

import time

from mabench.conflict_resolution import _cr_log


class QueryPipelineMixin:
    """Mixin: post-hop processing, merging, generation, decomposition.

    Cross-mixin dependencies (resolved via MRO at runtime):
    - RetrievalMixin: _retrieve_v2, _apply_post_retrieval
    - ConflictResolutionMixin: _query_time_cr, _pairwise_cosine_dedup, _coverage_audit
    - OutgestionMixin: _format_outgestion, _generate_answer
    - QueryMixin: _query_run_hop
    - RebelKGMixin: _rebel_kg_hop
    """

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

        if self.decompose_query:
            v2_texts, sq_chunks_raw, hop_state = self._query_decompose_and_merge(
                bare_question,
            )
        else:
            v2_results = self._retrieve_v2(bare_question)
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
        seen_texts: set[str] = set()
        answers: dict[str, str] = {}
        evidence_chain: list[dict] = []
        sq_chunks: list[list[str]] = []
        sq_chunks_raw: list[list[dict]] = []
        final_kg_answer = None

        for i, sq_template in enumerate(subqueries):
            kg_ans = self._query_run_hop(
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
        orig_results = self._retrieve_v2(bare_question)
        if (getattr(self, "bm25_hops", False)
                and getattr(self, "_bm25_index", None) is not None):
            orig_results = self._bm25_rerank_orig(bare_question, orig_results)
        orig_texts = [r["text"] for r in orig_results]

        # Merge: RRF or priority
        if self.decompose_rrf:
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
        bm25_scores = self._bm25_index.score_texts(
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
        return [key_to_text[k] for k in sorted_keys[: self.retrieve_num]]

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
    ) -> tuple[list[str], dict[str, dict]]:
        """Post-retrieval: metadata, V1, triadic, coverage, CR, belief, supersession.

        Returns (merged_chunks, metadata_by_key).
        """
        # Build metadata lookup
        metadata_by_key = self._build_metadata_lookup(v2_texts, sq_chunks_raw)

        # Merge + V1 + triadic + coverage
        merged_chunks = self._merge_and_expand(
            v2_texts, bare_question, question, metadata_by_key,
        )

        # Fact currency filtering
        merged_chunks = self._apply_currency_filter(merged_chunks)
        return merged_chunks, metadata_by_key

    def _build_metadata_lookup(
        self,
        v2_texts: list[str],
        sq_chunks_raw: list[list[dict]],
    ) -> dict[str, dict]:
        """Build text-key to result-dict metadata lookup."""
        metadata_by_key: dict[str, dict] = {}
        # _all_v2_results is set during _query_final_retrieval on hop_state
        # but we need the metadata from the actual results passed through
        # We reconstruct from what we have: the raw results contain metadata
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
        if getattr(self, "_triadic", None) is not None and v2_texts:
            triadic_texts = self._triadic.expand(
                v2_texts, top_k=self._triadic_expand_k,
            )
            for text in triadic_texts:
                key = text.strip()[:200]
                if key not in seen:
                    seen.add(key)
                    merged_chunks.append(text)

        # Coverage audit
        if self.decompose_coverage and not self.decompose_query:
            gap_texts = self._coverage_audit(bare_question, v2_texts)
            for text in gap_texts:
                key = text.strip()[:200]
                if key not in seen:
                    seen.add(key)
                    merged_chunks.append(text)

        # V1 text merge
        if self.cooc_boost_retrieval:
            v1_context = ""
        else:
            v1_result = self.orchestrator.query(message=question)
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
        if self.cosine_dedup:
            merged_chunks = self._pairwise_cosine_dedup(merged_chunks)
        if self.conflict_resolution:
            merged_chunks = self._query_time_cr(merged_chunks)
        elif not self.cosine_dedup:
            if self._belief_index is not None:
                merged_chunks = [
                    c for c in merged_chunks
                    if not self._belief_index.is_excluded(c)
                ]
            elif self.supersession and self._superseded_texts:
                merged_chunks = [
                    c for c in merged_chunks
                    if c not in self._superseded_texts
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

        Handles KG short-circuit if final_kg_answer is set.
        """
        # KG short-circuit
        final_kg_answer = hop_state.get("final_kg_answer")
        if final_kg_answer is not None:
            qt = time.time() - query_start
            _q_emb = self._query_q_emb
            if getattr(self, "_chain_cache", None) is not None and _q_emb is not None:
                self._chain_cache.store(
                    _q_emb, bare_question, final_kg_answer,
                    hop_state.get("evidence_chain", []),
                )
            return {
                "output": final_kg_answer,
                "input_len": 0,
                "output_len": 0,
                "memory_construction_time": qt,
                "query_time_len": qt,
            }

        final_chunks = merged_chunks[: self.retrieve_num]
        memory_context, outgestion_instructions = self._format_outgestion(
            final_chunks, metadata_by_key,
        )
        memory_time = time.time() - query_start

        answer, input_tokens, output_tokens, gen_time = self._generate_answer(
            question, memory_context,
            extra_system_instructions=outgestion_instructions,
        )
        query_time = time.time() - query_start

        # Chain cache store
        self._query_store_chain_cache(
            bare_question, final_chunks, answer, hop_state,
        )

        return {
            "output": answer,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
        }

    def _query_store_chain_cache(
        self,
        bare_question: str,
        final_chunks: list[str],
        answer: str,
        hop_state: dict,
    ) -> None:
        """Store completed decompose chain in cache for future reuse."""
        _q_emb = self._query_q_emb
        if (getattr(self, "_chain_cache", None) is None
                or not self.decompose_query
                or _q_emb is None):
            return

        from mabench.chain_cache import CachedChain
        chain = CachedChain(
            query_embedding=_q_emb,
            query_text=bare_question,
            sub_queries=hop_state.get("subqueries", []),
            intermediate_answers=hop_state.get("answers", {}),
            evidence_facts=list(final_chunks),
            final_answer=answer,
            chain_id=len(self._chain_cache),
        )
        self._chain_cache.store(chain)

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
        self._ensure_llm_client()

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
            response = self._llm_client.chat.completions.create(
                model=self.model,
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
        subqs = subqs[: self.decompose_max_hops]
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
        self._ensure_llm_client()
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
            response = self._llm_client.chat.completions.create(
                model=self.model,
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
