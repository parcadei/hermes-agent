"""Answer formatting and LLM generation — standalone module.

Converted from OutgestionMixin to standalone Outgester class.
All self.* attribute access replaced with explicit config + instance fields.

Migration notes for callers:
  - OutgestionMixin is a backward-compat alias for Outgester.
  - Mixin callers (hermes_agent.py) still work via MRO — the alias
    keeps the import path stable.  HermesMemoryAgent.__init__ sets
    self.model / self.temperature / self._max_gen_tokens / etc. directly;
    the _resolve_* properties fall back to those when _cfg is absent.
  - Direct callers (interactive_query.py calling agent._generate_answer)
    continue to work because hermes_agent.py inherits OutgestionMixin
    which is now Outgester.
  - Standalone usage:
      from mabench.outgestion import Outgester
      from mabench.configs import LLMConfig
      og = Outgester(LLMConfig(model="gpt-4o-mini"))
      ctx, instr = og.format_outgestion(chunks, metadata_by_key)
      answer, in_tok, out_tok, gen_time = og.generate_answer(q, ctx, instr)
"""

from __future__ import annotations

import time
from collections import defaultdict

from mabench.configs import LLMConfig


class Outgester:
    """Formats retrieval results and generates LLM answers.

    Standalone replacement for the former OutgestionMixin.  Takes an
    explicit LLMConfig instead of reading scattered self.* attributes.

    When used as a mixin (no __init__ call), falls back to reading the
    old self.model / self.temperature / self._max_gen_tokens attributes
    set by the host class.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        conflict_resolution: bool = False,
        temporal_context: bool = False,
        contradiction_context: bool = False,
        contradiction_sim_threshold: float = 0.85,
    ) -> None:
        # _cfg: LLMConfig(model="gpt-4o-mini", temperature=0.0, max_gen_tokens=64)
        self._cfg = llm_config
        # _client: OpenAI | None — lazy-initialized on first generate_answer call
        self._client = None
        self._conflict_resolution = conflict_resolution
        self._temporal_context = temporal_context
        self._contradiction_context = contradiction_context
        self._contradiction_sim_threshold = contradiction_sim_threshold

    # ------------------------------------------------------------------
    # Attribute resolution: new _cfg-based or old self.* fallback
    # ------------------------------------------------------------------

    @property
    def _resolve_model(self) -> str:
        """LLM model name — from _cfg or host class self.model."""
        if hasattr(self, "_cfg"):
            return self._cfg.model
        return self.model  # type: ignore[attr-defined]

    @property
    def _resolve_temperature(self) -> float:
        """LLM temperature — from _cfg or host class self.temperature."""
        if hasattr(self, "_cfg"):
            return self._cfg.temperature
        return self.temperature  # type: ignore[attr-defined]

    @property
    def _resolve_max_gen_tokens(self) -> int:
        """Max generation tokens — from _cfg or host class self._max_gen_tokens."""
        if hasattr(self, "_cfg"):
            return self._cfg.max_gen_tokens
        return self._max_gen_tokens  # type: ignore[attr-defined]

    @property
    def _resolve_conflict_resolution(self) -> bool:
        """Conflict resolution flag — from _conflict_resolution or host class."""
        if hasattr(self, "_conflict_resolution"):
            return self._conflict_resolution
        return getattr(self, "conflict_resolution", False)

    @property
    def _resolve_temporal_context(self) -> bool:
        """Temporal context flag — from _temporal_context or host class."""
        if hasattr(self, "_temporal_context"):
            return self._temporal_context
        return getattr(self, "temporal_context", False)

    @property
    def _resolve_contradiction_context(self) -> bool:
        """Contradiction context flag — from _contradiction_context or host class."""
        if hasattr(self, "_contradiction_context"):
            return self._contradiction_context
        return getattr(self, "contradiction_context", False)

    @property
    def _resolve_contradiction_sim_threshold(self) -> float:
        """Contradiction similarity threshold — from instance or host class."""
        if hasattr(self, "_contradiction_sim_threshold"):
            return self._contradiction_sim_threshold
        return getattr(self, "contradiction_sim_threshold", 0.85)

    @property
    def _resolve_client(self):
        """OpenAI client — from _client or host class _llm_client."""
        if hasattr(self, "_client"):
            return self._client
        return self._llm_client  # type: ignore[attr-defined]

    @_resolve_client.setter
    def _resolve_client(self, value) -> None:
        if hasattr(self, "_client"):
            self._client = value
        else:
            self._llm_client = value  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Lazy OpenAI client
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        """Initialize OpenAI client on first use (OpenRouter endpoint)."""
        if self._resolve_client is not None:
            return
        import os

        from openai import OpenAI

        self._resolve_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    # Backward-compat alias for mixin callers that call self._ensure_llm_client()
    _ensure_llm_client = _ensure_client

    # ------------------------------------------------------------------
    # Public API (formerly _format_outgestion / _generate_answer)
    # ------------------------------------------------------------------

    def format_outgestion(
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
        if self._resolve_conflict_resolution:
            return "\n\n".join(chunks), ""

        if not self._resolve_temporal_context:
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
        contradiction_threshold = self._resolve_contradiction_sim_threshold
        conflict_pairs: dict[int, list[int]] = defaultdict(list)

        if self._resolve_contradiction_context:
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

    def generate_answer(
        self,
        question: str,
        context: str,
        extra_system_instructions: str = "",
    ) -> tuple[str, int, int, float]:
        """Call LLM to generate an answer from retrieved context."""
        self._ensure_client()

        system_msg = (
            "You are a helpful assistant that answers questions from a knowledge pool. Give a very concise answer."
        )
        if extra_system_instructions:
            system_msg = system_msg + " " + extra_system_instructions
        # The MABench query already contains full instructions about serial
        # numbers and recency. Prepend retrieved memories as the knowledge pool.
        user_msg = f"[Knowledge Pool]\n{context}\n\n{question}\nAnswer:"

        start = time.time()
        response = self._resolve_client.chat.completions.create(
            model=self._resolve_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=self._resolve_temperature,
            max_tokens=max(self._resolve_max_gen_tokens, 16),
        )
        gen_time = time.time() - start

        output = response.choices[0].message.content or ""
        usage = response.usage
        return output, usage.prompt_tokens, usage.completion_tokens, gen_time

    # ------------------------------------------------------------------
    # Backward-compat underscore aliases (used by mixin callers via MRO)
    # ------------------------------------------------------------------

    def _format_outgestion(
        self,
        chunks: list[str],
        metadata_by_key: dict[str, dict],
    ) -> tuple[str, str]:
        """Alias for format_outgestion — keeps mixin callers working."""
        return self.format_outgestion(chunks, metadata_by_key)

    def _generate_answer(
        self,
        question: str,
        context: str,
        extra_system_instructions: str = "",
    ) -> tuple[str, int, int, float]:
        """Alias for generate_answer — keeps mixin callers working."""
        return self.generate_answer(question, context, extra_system_instructions)


# Backward-compat alias so `from mabench.outgestion import OutgestionMixin` works.
OutgestionMixin = Outgester
