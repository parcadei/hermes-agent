"""LLM-based triple extraction via vLLM + structured output.

Uses Qwen3-30B-A3B MoE (3B active) served by vLLM to extract
(subject, predicate, object) triples from arbitrary text.
vLLM's response_format with json_schema enforces valid JSON
at the token-sampling level.

Architecture:
  - vLLM serves the model as an OpenAI-compatible API
  - Each extraction call is fresh/stateless (no session accumulation)
  - Thinking mode disabled (enable_thinking=False) for throughput
  - Grounding check: subject and object must be substrings of source text
  - Predicates normalized: lowercased, underscores→spaces, stripped

Feeds into _triple_index for predicated KG hop resolution.
Drop-in replacement for spaCy dep-parse template extraction.

Usage::

    extractor = LLMTripleExtractor(base_url="http://localhost:8000/v1")

    # Single text
    triples = extractor.extract("The capital of France is Paris.")
    # [("France", "capital", "Paris")]

    # Batch
    batch = extractor.extract_batch(["The capital of France is Paris.", ...])

Pickle-safe: OpenAI client excluded from state, recreated on demand.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

_log = logging.getLogger("llm_triple_extractor")

# OpenAI-compatible response_format for vLLM structured output.
RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "triple_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                        },
                        "required": ["subject", "predicate", "object"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["triples"],
            "additionalProperties": False,
        },
    },
}

EXTRACTION_PROMPT = (
    "Extract factual relationships as (subject, predicate, object) triples.\n\n"
    "Rules:\n"
    "- Subject and object must be clean entity names "
    '(e.g. "Helsinki" not "the city of Helsinki")\n'
    "- Predicate must be a short relation label "
    '(e.g. "capital" not "is the capital of")\n'
    "- Only extract facts explicitly stated in the text\n"
    "- Subject is always the entity BEING DESCRIBED, "
    "object is the attribute value\n\n"
    "Examples:\n\n"
    'Text: "The official language of United Kingdom is Finnish."\n'
    'Output: [("United Kingdom", "official language", "Finnish")]\n\n'
    'Text: "The spouse of Charles Darwin is Emma Darwin."\n'
    'Output: [("Charles Darwin", "spouse", "Emma Darwin")]\n\n'
    'Text: "Rock music is a genre that originated in Spain."\n'
    'Output: [("rock music", "country of origin", "Spain")]\n\n'
    'Text: "Association football is a sport played by Peter Schmeichel."\n'
    'Output: [("Peter Schmeichel", "sport", "association football")]\n\n'
    'Text: "The head of government of France is Emmanuel Macron."\n'
    'Output: [("France", "head of government", "Emmanuel Macron")]\n\n'
    'Text: "Chevrolet Caprice is manufactured by General Motors, '
    'which was founded in Silver Lake."\n'
    'Output: [("Chevrolet Caprice", "manufacturer", "General Motors"), '
    '("General Motors", "founded in", "Silver Lake")]\n\n'
    'Now extract triples from:\n\n'
    'Text: "{text}"'
)


def _ground_triple(
    subject: str, predicate: str, obj: str, source_text: str,
) -> bool:
    """Grounding check: subject and object must be substrings of source text.

    The predicate may be inferred (e.g. "capital" from "X is the capital of Y")
    so it is NOT required to appear literally. But subject and object must be
    grounded in the source text to prevent hallucination.
    """
    text_lower = source_text.lower()
    return (
        subject.lower() in text_lower
        and obj.lower() in text_lower
    )


def _normalize_predicate(pred: str) -> str:
    """Normalize predicate to canonical form for consistent KG lookups."""
    return pred.lower().replace("_", " ").strip()


class LLMTripleExtractor:
    """Triple extractor using a small LLM served by vLLM.

    Connects to a vLLM OpenAI-compatible server and uses response_format
    with json_schema to enforce structured output at the token-sampling level.

    Pickle-safe: the OpenAI client is excluded and recreated lazily.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "qwen3-triple-extractor",
        *,
        api_key: str = "dummy",
        temperature: float = 0.0,
        max_tokens: int = 512,
        enable_grounding: bool = True,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_grounding = enable_grounding
        self._client: Any = None
        self._stats = {"texts": 0, "triples": 0, "grounded": 0, "discarded": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[tuple[str, str, str]]:
        """Extract (subject, predicate, object) triples from a single text."""
        results = self._call_llm([text])
        return results[0] if results else []

    def extract_batch(
        self, texts: list[str],
    ) -> list[list[tuple[str, str, str]]]:
        """Extract triples from a batch of texts.

        Calls the LLM once per text (vLLM handles batching internally
        via continuous batching when requests arrive concurrently).
        """
        if not texts:
            return []
        return self._call_llm(texts)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        """Lazy-init the OpenAI client."""
        if self._client is not None:
            return
        from openai import OpenAI

        _log.info(
            "Connecting to vLLM at %s (model=%s)",
            self._base_url, self._model,
        )
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

    def _call_llm_single(self, text: str) -> dict:
        """Call vLLM for a single text. Returns parsed JSON or None on error."""
        prompt = EXTRACTION_PROMPT.format(text=text)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                seed=42,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                response_format=RESPONSE_FORMAT,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            _log.warning("LLM extraction failed for text: %s...", text[:80], exc_info=True)
            return None

    def _ground_and_collect(
        self, text: str, parsed: dict | None,
    ) -> list[tuple[str, str, str]]:
        """Ground triples from a parsed LLM response against source text."""
        if parsed is None:
            return []
        grounded: list[tuple[str, str, str]] = []
        for t in parsed.get("triples", []):
            s = t.get("subject", "").strip()
            p = _normalize_predicate(t.get("predicate", ""))
            o = t.get("object", "").strip()
            if not s or not p or not o:
                continue
            if self._enable_grounding:
                if _ground_triple(s, p, o, text):
                    grounded.append((s, p, o))
                    self._stats["grounded"] += 1
                else:
                    self._stats["discarded"] += 1
            else:
                grounded.append((s, p, o))
        self._stats["texts"] += 1
        self._stats["triples"] += len(grounded)
        return grounded

    def _call_llm(
        self, texts: list[str],
    ) -> list[list[tuple[str, str, str]]]:
        """Call vLLM concurrently for all texts, then ground triples.

        Uses ThreadPoolExecutor so vLLM's continuous batching processes
        requests in parallel on the GPU. Typically 5-10x faster than
        sequential calls for batches of 10-20 sentences.
        """
        self._ensure_client()
        if not texts:
            return []

        from concurrent.futures import ThreadPoolExecutor

        # Send all requests concurrently — vLLM batches them on the GPU
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(len(texts), 16)) as pool:
            parsed_results = list(pool.map(self._call_llm_single, texts))
        elapsed = time.perf_counter() - t0

        _log.info(
            "LLM batch: %d texts in %.1fs (%.3fs/text)",
            len(texts), elapsed, elapsed / len(texts) if texts else 0,
        )

        # Ground and collect triples (CPU-only, fast)
        return [
            self._ground_and_collect(text, parsed)
            for text, parsed in zip(texts, parsed_results)
        ]

    # ------------------------------------------------------------------
    # Pickle support (exclude client, lazy-recreate)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
