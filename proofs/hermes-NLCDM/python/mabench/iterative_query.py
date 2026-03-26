"""Iterative multi-hop query pipeline for HermesMemoryAgent.

Provides IterativeQueryMixin with _iterative_query and its decomposed helpers:
- _iterative_hop: single hop (retrieve, triadic expand, judge)
- _iterative_judge: LLM judge for iterative query
- _build_iterative_chain_summary: format reasoning chain for judge
- _parse_iterative_judge: parse ANSWER_READY / LEARNED+NEED response
- _iterative_finalize: post-process and generate final answer
"""

from __future__ import annotations

import time


class IterativeQueryMixin:
    """Mixin providing iterative multi-hop retrieval with LLM-in-the-loop termination.

    Requires the host class to provide:
    - self._scorer: EmbeddingRelevanceScorer with .embed(text)
    - self.coupled_engine: engine with memory_store
    - self._llm_client: OpenAI client (lazily initialised)
    - self._ensure_llm_client(): method to init LLM client
    - self.model: str, LLM model name
    - self.temperature: float
    - self.iterative_max_hops: int
    - self.retrieve_num: int
    - self.conflict_resolution: bool
    - self.supersession: bool
    - self._superseded_texts: set
    - self._belief_index: BeliefIndex or None
    - self._retrieve_v2(query): from RetrievalMixin
    - self._resolve_conflicts(chunks, metadata): from ConflictResolutionMixin
    - self._format_outgestion(chunks, metadata): from OutgestionMixin
    - self._generate_answer(question, context, ...): from OutgestionMixin
    - self._extract_question(prompt): from QueryMixin (staticmethod)
    """

    def _iterative_query(self, question: str) -> dict:
        """Iterative multi-hop retrieval with LLM-in-the-loop termination."""
        query_start = time.time()
        bare_question = self._extract_question(question)
        self._ensure_llm_client()

        all_chunks: list[str] = []
        all_results: list[dict] = []
        seen: set[str] = set()
        evidence_chain: list[dict] = []
        current_query = bare_question
        hops_taken = 0

        for hop in range(self.iterative_max_hops):
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
        results = self._retrieve_v2(current_query)

        # Triadic expansion
        hop_texts = [r["text"] for r in results]
        if getattr(self, "_triadic", None) is not None and hop_texts:
            triadic_texts = self._triadic.expand(
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
        if hop >= self.iterative_max_hops - 1:
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
            judge_response = self._llm_client.chat.completions.create(
                model=self.model,
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
        if self.conflict_resolution:
            all_chunks = self._resolve_conflicts(all_chunks, metadata_by_key)
        elif self._belief_index is not None:
            all_chunks = [
                c for c in all_chunks
                if not self._belief_index.is_excluded(c)
            ]
        elif self.supersession and self._superseded_texts:
            all_chunks = [
                c for c in all_chunks
                if c not in self._superseded_texts
            ]

        final_chunks = all_chunks[: self.retrieve_num]
        memory_context, outgestion_instructions = self._format_outgestion(
            final_chunks, metadata_by_key,
        )
        memory_time = time.time() - query_start

        answer, input_tokens, output_tokens, gen_time = self._generate_answer(
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
