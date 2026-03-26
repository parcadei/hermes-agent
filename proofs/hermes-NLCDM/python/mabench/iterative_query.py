"""Iterative multi-hop query pipeline — standalone module.

Converted from IterativeQueryMixin to standalone IterativeQuery class.
IterativeQueryMixin is a backward-compat alias. Mixin callers still work
via MRO; the _resolve_* properties fall back to host-class self.* attrs.

Standalone usage:
    iq = IterativeQuery(engine=e, scorer=s, retriever=r, outgester=o,
                        llm_config=LLMConfig(model="gpt-4o-mini"))
    result = iq._iterative_query("Question: What sport...\\nAnswer:")
"""

from __future__ import annotations

import time

from mabench.configs import CRConfig, LLMConfig, RetrievalConfig


class IterativeQuery:
    """Standalone iterative multi-hop query with LLM-in-the-loop termination.

    Can be used independently or composed into HermesMemoryAgent.

    Usage::

        iq = IterativeQuery(
            engine=coupled_engine,
            scorer=scorer,
            retriever=retriever,
            outgester=outgester,
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )
        result = iq._iterative_query("Question: What sport...\\nAnswer:")
    """

    def __init__(
        self,
        engine=None,
        scorer=None,
        retriever=None,
        conflict_resolver=None,
        outgester=None,
        triadic=None,
        belief_index=None,
        llm_config: LLMConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
        cr_config: CRConfig | None = None,
        iterative_max_hops: int = 3,
        supersession: bool = False,
        superseded_texts=None,
    ) -> None:
        # _engine: CoupledEngine with .memory_store
        self._engine = engine
        # _scorer: EmbeddingRelevanceScorer with .embed(text)
        self._scorer = scorer
        # _retriever: Retriever with ._retrieve_v2(query)
        self._retriever = retriever
        # _conflict_resolver: ConflictResolver with ._resolve_conflicts(chunks, meta)
        self._conflict_resolver = conflict_resolver
        # _outgester: Outgester with .format_outgestion / .generate_answer
        self._outgester = outgester
        # _triadic: TriadicMemory with .expand(texts, top_k=k) or None
        self._triadic = triadic
        # _belief_index: BeliefIndex with .is_excluded(text) or None
        self._belief_index = belief_index
        # _llm_config: LLMConfig(model, temperature, max_gen_tokens)
        self._llm_config = llm_config or LLMConfig()
        # _retrieval_cfg: RetrievalConfig(retrieve_num=10, ...)
        self._retrieval_cfg = retrieval_config or RetrievalConfig()
        # _cr_cfg: CRConfig(enabled=False, ...)
        self._cr_cfg = cr_config or CRConfig()
        # _iterative_max_hops: max multi-hop iterations
        self._iterative_max_hops = iterative_max_hops
        # _supersession_flag: whether supersession filtering is enabled
        self._supersession_flag = supersession
        # _superseded_texts: set of fact texts that have been superseded
        self._superseded_texts = superseded_texts or set()
        # _client: OpenAI client (lazy-init)
        self._client = None
        # _triadic_expand_k: number of triadic expansion results
        self._triadic_expand_k = 5

    # ------------------------------------------------------------------
    # Attribute resolution: new config or old self.* fallback
    # ------------------------------------------------------------------

    @property
    def _resolve_engine(self):
        """CoupledEngine -- from __init__ or host class self.coupled_engine."""
        if hasattr(self, "_engine") and self._engine is not None:
            return self._engine
        return self.coupled_engine  # type: ignore[attr-defined]

    @property
    def _resolve_iterative_max_hops(self) -> int:
        """Max hops -- from __init__ or host class self.iterative_max_hops."""
        if hasattr(self, "_iterative_max_hops"):
            return self._iterative_max_hops
        return getattr(self, "iterative_max_hops", 3)  # type: ignore[return-value]

    @property
    def _resolve_retrieve_num(self) -> int:
        """Retrieve count -- from config or host class self.retrieve_num."""
        if hasattr(self, "_retrieval_cfg") and self._retrieval_cfg is not None:
            return self._retrieval_cfg.retrieve_num
        return getattr(self, "retrieve_num", 10)  # type: ignore[return-value]

    @property
    def _resolve_conflict_resolution(self) -> bool:
        """CR enabled -- from config or host class self.conflict_resolution."""
        if hasattr(self, "_cr_cfg") and self._cr_cfg is not None:
            return self._cr_cfg.enabled
        return getattr(self, "conflict_resolution", False)  # type: ignore[return-value]

    @property
    def _resolve_supersession(self) -> bool:
        """Supersession enabled -- from __init__ or host class self.supersession."""
        if hasattr(self, "_supersession_flag"):
            return self._supersession_flag
        return getattr(self, "supersession", False)  # type: ignore[return-value]

    @property
    def _resolve_model(self) -> str:
        """LLM model name -- from config or host class self.model."""
        if hasattr(self, "_llm_config") and self._llm_config is not None:
            return self._llm_config.model
        return self.model  # type: ignore[attr-defined]

    @property
    def _resolve_temperature(self) -> float:
        """LLM temperature -- from config or host class self.temperature."""
        if hasattr(self, "_llm_config") and self._llm_config is not None:
            return self._llm_config.temperature
        return self.temperature  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Cross-module call resolution
    # ------------------------------------------------------------------

    def _resolve_retrieve_v2(self, query):
        """Retrieve facts -- from injected Retriever or host class self._retrieve_v2."""
        if hasattr(self, "_retriever") and self._retriever is not None:
            return self._retriever._retrieve_v2(query)
        return self._retrieve_v2(query)  # type: ignore[attr-defined]

    def _resolve_resolve_conflicts(self, chunks, metadata):
        """CR filter -- from injected ConflictResolver or host class self._resolve_conflicts."""
        if hasattr(self, "_conflict_resolver") and self._conflict_resolver is not None:
            return self._conflict_resolver._resolve_conflicts(chunks, metadata)
        return self._resolve_conflicts(chunks, metadata)  # type: ignore[attr-defined]

    def _resolve_format_outgestion(self, chunks, metadata):
        """Format output -- from injected Outgester or host class self._format_outgestion."""
        if hasattr(self, "_outgester") and self._outgester is not None:
            return self._outgester.format_outgestion(chunks, metadata)
        return self._format_outgestion(chunks, metadata)  # type: ignore[attr-defined]

    def _resolve_generate_answer(self, question, context, **kwargs):
        """Generate answer -- from injected Outgester or host class self._generate_answer."""
        if hasattr(self, "_outgester") and self._outgester is not None:
            return self._outgester.generate_answer(question, context, **kwargs)
        return self._generate_answer(question, context, **kwargs)  # type: ignore[attr-defined]

    def _resolve_ensure_llm_client(self):
        """Ensure LLM client is initialized -- standalone or host class."""
        if hasattr(self, "_client"):
            if self._client is not None:
                return
            import os

            from openai import OpenAI

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            )
        else:
            self._ensure_llm_client()  # type: ignore[attr-defined]

    @property
    def _resolve_llm_client(self):
        """OpenAI client -- from standalone or host class self._llm_client."""
        if hasattr(self, "_client") and self._client is not None:
            return self._client
        # Mixin mode: host class sets self._llm_client directly
        return getattr(self, "_llm_client", self._client)

    # ------------------------------------------------------------------
    # Static helper (no deps)
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
    # Core methods (unchanged logic, self.X -> self._resolve_X)
    # ------------------------------------------------------------------

    def _iterative_query(self, question: str) -> dict:
        """Iterative multi-hop retrieval with LLM-in-the-loop termination."""
        query_start = time.time()
        bare_question = self._extract_question(question)
        self._resolve_ensure_llm_client()

        all_chunks: list[str] = []
        all_results: list[dict] = []
        seen: set[str] = set()
        evidence_chain: list[dict] = []
        current_query = bare_question
        hops_taken = 0

        for hop in range(self._resolve_iterative_max_hops):
            hop_result = self._iterative_hop(
                bare_question, current_query, hop,
                all_chunks, all_results, seen, evidence_chain,
            )
            hops_taken = hop + 1
            if hop_result is None:
                break  # done or error
            current_query = hop_result  # next entity to search

        return self._iterative_finalize(
            question, all_chunks, all_results, evidence_chain,
            hops_taken, query_start,
        )

    def _iterative_hop(
        self,
        bare_question: str,
        current_query: str,
        hop: int,
        all_chunks: list[str],
        all_results: list[dict],
        seen: set[str],
        evidence_chain: list[dict],
    ) -> str | None:
        """Single iterative hop: retrieve, triadic expand, judge.

        Returns the next entity to search, or None if done/error.
        """
        results = self._resolve_retrieve_v2(current_query)

        # Triadic expansion
        hop_texts = [r["text"] for r in results]
        triadic = getattr(self, "_triadic", None)
        if triadic is not None and hop_texts:
            triadic_texts = triadic.expand(
                hop_texts, top_k=self._triadic_expand_k,
            )
            for t in triadic_texts:
                key = t.strip()[:200]
                if key not in seen:
                    seen.add(key)
                    all_chunks.append(t)

        # Dedup new chunks
        hop_new_chunks: list[str] = []
        for r in results:
            key = r["text"].strip()[:200]
            if key not in seen:
                seen.add(key)
                all_chunks.append(r["text"])
                all_results.append(r)
                hop_new_chunks.append(r["text"])

        # Last hop: no judge
        if hop >= self._resolve_iterative_max_hops - 1:
            evidence_chain.append({
                "query": current_query,
                "found": hop_new_chunks[:3],
                "entity": None,
            })
            return None

        # LLM judge
        return self._iterative_judge(
            bare_question, current_query, hop_new_chunks,
            all_chunks, evidence_chain,
        )

    def _iterative_judge(
        self,
        bare_question: str,
        current_query: str,
        hop_new_chunks: list[str],
        all_chunks: list[str],
        evidence_chain: list[dict],
    ) -> str | None:
        """LLM judge for iterative query. Returns next entity or None."""
        chain_summary = self._build_iterative_chain_summary(evidence_chain)
        context_for_judge = "\n".join(all_chunks)

        try:
            judge_response = self._resolve_llm_client.chat.completions.create(
                model=self._resolve_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You evaluate whether retrieved facts can answer "
                            "a question, and extract intermediate answers.\n\n"
                            "If the facts contain enough to answer, respond:\n"
                            "ANSWER_READY\n\n"
                            "If not, extract what you DID learn and state "
                            "what simple fact to search next. Format:\n"
                            "LEARNED: <what you found, e.g. 'Abbiati plays "
                            "association football'>\n"
                            "NEED: <simple search, e.g. 'association football "
                            "country of origin'>\n\n"
                            "The NEED query will be used for embedding search. "
                            "Use CONCRETE entities from what you learned, "
                            "NOT abstract references like 'the sport' or "
                            "'the person'. Keep NEED to 3-8 words.\n"
                            "Do NOT repeat a previous search."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {bare_question}\n\n"
                            f"{chain_summary}"
                            f"Retrieved facts:\n{context_for_judge}\n\n"
                            f"Can you answer? If not, what did you learn "
                            f"and what concrete fact is still missing?"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=64,
            )
            judge_text = (judge_response.choices[0].message.content or "").strip()
        except Exception:
            evidence_chain.append({
                "query": current_query,
                "found": hop_new_chunks[:3],
                "entity": None,
            })
            return None

        return self._parse_iterative_judge(
            judge_text, current_query, hop_new_chunks, evidence_chain,
        )

    def _build_iterative_chain_summary(
        self, evidence_chain: list[dict],
    ) -> str:
        """Build reasoning chain summary for the iterative judge."""
        if not evidence_chain:
            return ""
        lines = []
        for i, step in enumerate(evidence_chain, 1):
            found_preview = (
                step["found"][0][:100] if step["found"] else "nothing relevant"
            )
            learned_str = (
                f" -> learned: {step['learned']}" if step.get("learned") else ""
            )
            entity_str = (
                f' -> next search: "{step["entity"]}"'
                if step.get("entity") else ""
            )
            lines.append(
                f'  Hop {i}: searched "{step["query"]}" -> '
                f'found "{found_preview}..."{learned_str}{entity_str}'
            )
        return "Reasoning chain so far:\n" + "\n".join(lines) + "\n\n"

    def _parse_iterative_judge(
        self,
        judge_text: str,
        current_query: str,
        hop_new_chunks: list[str],
        evidence_chain: list[dict],
    ) -> str | None:
        """Parse judge response for iterative query. Returns next entity or None."""
        if judge_text.startswith("ANSWER_READY"):
            evidence_chain.append({
                "query": current_query,
                "found": hop_new_chunks[:3],
                "entity": None,
            })
            return None

        learned = ""
        next_entity = ""
        for line in judge_text.split("\n"):
            line = line.strip()
            if line.startswith("LEARNED:"):
                learned = line[len("LEARNED:"):].strip()
            elif line.startswith("NEED:"):
                next_entity = line[len("NEED:"):].strip()

        # Fallback: old format
        if not next_entity and judge_text.startswith("NEED:"):
            next_entity = judge_text[len("NEED:"):].strip()

        if next_entity:
            evidence_chain.append({
                "query": current_query,
                "found": hop_new_chunks[:3],
                "entity": next_entity,
                "learned": learned,
            })
            return next_entity

        # Unparseable
        evidence_chain.append({
            "query": current_query,
            "found": hop_new_chunks[:3],
            "entity": None,
        })
        return None

    def _iterative_finalize(
        self,
        question: str,
        all_chunks: list[str],
        all_results: list[dict],
        evidence_chain: list[dict],
        hops_taken: int,
        query_start: float,
    ) -> dict:
        """Post-process and generate answer for iterative query."""
        metadata_by_key: dict[str, dict] = {}
        for r in all_results:
            k = r["text"].strip()[:200]
            if k not in metadata_by_key:
                metadata_by_key[k] = r

        # Fact currency filtering
        if self._resolve_conflict_resolution:
            all_chunks = self._resolve_resolve_conflicts(all_chunks, metadata_by_key)
        elif self._belief_index is not None:
            all_chunks = [
                c for c in all_chunks
                if not self._belief_index.is_excluded(c)
            ]
        elif self._resolve_supersession and self._superseded_texts:
            all_chunks = [
                c for c in all_chunks
                if c not in self._superseded_texts
            ]

        final_chunks = all_chunks[: self._resolve_retrieve_num]
        memory_context, outgestion_instructions = self._resolve_format_outgestion(
            final_chunks, metadata_by_key,
        )
        memory_time = time.time() - query_start

        answer, input_tokens, output_tokens, gen_time = self._resolve_generate_answer(
            question, memory_context,
            extra_system_instructions=outgestion_instructions,
        )
        query_time = time.time() - query_start

        return {
            "output": answer,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
            "hops": hops_taken,
        }


# Backward-compat alias: old imports still work
IterativeQueryMixin = IterativeQuery
