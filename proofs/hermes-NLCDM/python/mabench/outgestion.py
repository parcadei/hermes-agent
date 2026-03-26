"""Answer formatting and LLM generation.

Extracts _format_outgestion and _generate_answer from HermesMemoryAgent
into a mixin for cleaner separation of outgestion concerns.
"""

from __future__ import annotations

import time
from collections import defaultdict


class OutgestionMixin:
    """Mixin for formatting retrieval results and generating LLM answers."""

    def _format_outgestion(
        self,
        chunks: list[str],
        metadata_by_key: dict[str, dict],
    ) -> tuple[str, str]:
        """Format retrieved chunks with outgestion signals.

        Returns (formatted_context, extra_system_instructions).
        When temporal_context is disabled, returns flat text with no
        extra instructions (baseline behavior).

        Signals applied (when enabled):
        1. Temporal ordering: sort oldest->newest, add [older]/[newer] tags
        2. Contradiction detection: flag high-similarity chunk pairs as
           [SUPERSEDED]/[CURRENT] when they likely represent conflicting
           versions of the same fact
        """
        # Conflict resolution mode: conflicts already resolved by Layer 2.
        # Layer 3 (newest-first sort) DISABLED -- eval showed it disrupts the
        # relevance ranking built by cosine+cooc+triadic, causing ~5 regressions.
        # Just return flat text preserving retrieval order.
        if getattr(self, "conflict_resolution", False):
            return "\n\n".join(chunks), ""

        if not getattr(self, "temporal_context", False):
            # Baseline: flat text, no metadata
            return "\n\n".join(chunks), ""

        # Build (recency, text, embedding) tuples for sorting + contradiction.
        # Chunks without metadata (e.g. from V1 or triadic expansion)
        # get recency=0 so they sort as oldest.
        annotated_full: list[tuple[float, str, "np.ndarray | None"]] = []
        for text in chunks:
            key = text.strip()[:200]
            meta = metadata_by_key.get(key)
            recency = meta["recency"] if meta else 0.0
            embedding = meta.get("embedding") if meta else None
            annotated_full.append((recency, text, embedding))

        # Sort oldest -> newest (ascending recency)
        annotated_full.sort(key=lambda x: x[0])

        # Soft contradiction surfacing: find pairs of chunks that are
        # semantically similar (cosine > threshold) but have different
        # recency values. Cross-reference BOTH with [CONFLICT] tags --
        # surface the conflict, never remove content.
        contradiction_threshold = getattr(self, "contradiction_sim_threshold", 0.85)
        conflict_pairs: dict[int, list[int]] = defaultdict(list)

        if getattr(self, "contradiction_context", False):
            import numpy as np

            n_ann = len(annotated_full)
            for i in range(n_ann):
                for j in range(i + 1, n_ann):
                    emb_i = annotated_full[i][2]
                    emb_j = annotated_full[j][2]
                    if emb_i is None or emb_j is None:
                        continue
                    sim = float(np.dot(emb_i, emb_j))
                    if sim >= contradiction_threshold:
                        rec_i = annotated_full[i][0]
                        rec_j = annotated_full[j][0]
                        if rec_i != rec_j:
                            conflict_pairs[i].append(j)
                            conflict_pairs[j].append(i)

        # Format with fact numbering + temporal + soft contradiction
        lines = []
        n = len(annotated_full)
        for i, (recency, text, _emb) in enumerate(annotated_full):
            fact_num = i + 1  # 1-indexed for LLM readability

            if i in conflict_pairs:
                partners = ", ".join(str(p + 1) for p in conflict_pairs[i])
                tag = f"[fact {fact_num}, CONFLICTS WITH fact {partners}] "
            elif n <= 1:
                tag = ""
            elif i < n // 3:
                tag = f"[fact {fact_num}, older] "
            elif i >= n - n // 3:
                tag = f"[fact {fact_num}, newer] "
            else:
                tag = f"[fact {fact_num}] "
            lines.append(f"{tag}{text}")

        instructions = (
            "Facts in the knowledge pool are numbered and ordered from oldest "
            "to newest. When facts conflict, prefer the most recent (newer) "
            "information."
        )
        if conflict_pairs:
            instructions += (
                " Facts marked [CONFLICTS WITH] represent potentially "
                "conflicting versions — prefer the newer one among "
                "conflicting facts."
            )
        return "\n\n".join(lines), instructions

    def _generate_answer(
        self,
        question: str,
        context: str,
        extra_system_instructions: str = "",
    ) -> tuple[str, int, int, float]:
        """Call LLM to generate an answer from retrieved context."""
        self._ensure_llm_client()

        system_msg = (
            "You are a helpful assistant that answers questions from a knowledge pool. Give a very concise answer."
        )
        if extra_system_instructions:
            system_msg = system_msg + " " + extra_system_instructions
        # The MABench query already contains full instructions about serial
        # numbers and recency. Prepend retrieved memories as the knowledge pool.
        user_msg = f"[Knowledge Pool]\n{context}\n\n{question}\nAnswer:"

        start = time.time()
        response = self._llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=max(self._max_gen_tokens, 16),
        )
        gen_time = time.time() - start

        output = response.choices[0].message.content or ""
        usage = response.usage
        return output, usage.prompt_tokens, usage.completion_tokens, gen_time
