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

    # Single text — factual mode (default, for template sentences)
    triples = extractor.extract("The capital of France is Paris.")
    # [("France", "capital", "Paris")]

    # Conversation mode — for chat turns (possessions, preferences, events)
    triples = extractor.extract(
        "I have a 20-gallon tank with 10 neon tetras.",
        mode="conversation",
    )
    # [("20-gallon tank", "contains", "10 neon tetras")]

    # Batch
    batch = extractor.extract_batch(["The capital of France is Paris.", ...])

Pickle-safe: OpenAI client excluded from state, recreated on demand.
"""

from __future__ import annotations

import json
import logging
import re
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

CONVERSATION_EXTRACTION_PROMPT = (
    "Extract knowledge from a conversation turn as (subject, predicate, object) triples.\n\n"
    "Rules:\n"
    "- Extract facts about what the user has, owns, prefers, did, or plans to do\n"
    "- Extract facts from assistant responses: recommendations, information provided\n"
    "- Preserve entity names exactly as they appear in the text\n"
    "- Include quantities when mentioned "
    '(e.g. "10 neon tetras" not just "neon tetras")\n'
    "- Predicate must be a short relation label "
    '(e.g. "contains" not "currently has inside of it")\n'
    "- One turn can produce many triples — extract ALL facts\n"
    "- Only extract facts explicitly stated in the text\n"
    "- Return an empty triples list for greetings or acknowledgments "
    "with no factual content\n\n"
    "Examples:\n\n"
    'Text: "I\'m thinking of adding some live plants to my new 20-gallon tank, '
    "which currently has 10 neon tetras, 5 golden honey gouramis, "
    'and a small pleco catfish."\n'
    'Output: [("20-gallon tank", "contains", "10 neon tetras"), '
    '("20-gallon tank", "contains", "5 golden honey gouramis"), '
    '("20-gallon tank", "contains", "pleco catfish"), '
    '("20-gallon tank", "status", "new")]\n\n'
    'Text: "I really prefer Python over JavaScript for backend work, '
    "and I've been using VS Code as my primary editor for about "
    '3 years now."\n'
    'Output: [("Python", "preferred over", "JavaScript"), '
    '("VS Code", "role", "primary editor"), '
    '("VS Code", "usage duration", "3 years")]\n\n'
    'Text: "I\'m planning a trip to Miami and I want to redeem my Delta SkyMiles '
    "for a free trip. I just got back from a red-eye flight on JetBlue "
    'from San Francisco to Boston."\n'
    'Output: [("Miami", "planned activity", "trip"), '
    '("Delta SkyMiles", "intended use", "free trip"), '
    '("JetBlue", "route", "San Francisco to Boston"), '
    '("JetBlue flight", "type", "red-eye")]\n\n'
    'Text: "Based on your tank setup, I\'d recommend Java fern and Anubias nana '
    "as beginner-friendly plants. They don't need CO2 injection and "
    'your gouramis will appreciate the hiding spots."\n'
    'Output: [("Java fern", "recommended for", "tank setup"), '
    '("Anubias nana", "recommended for", "tank setup"), '
    '("Java fern", "requires", "no CO2 injection"), '
    '("Anubias nana", "requires", "no CO2 injection")]\n\n'
    'Text: "Hi! Can you help me with something?"\n'
    "Output: []\n\n"
    'Text: "So I have a 75-gallon reef tank in my living room with 2 clownfish '
    'and 3 blue-green chromis, plus a clean-up crew of 5 hermit crabs."\n'
    'Output: [("75-gallon reef tank", "location", "living room"), '
    '("75-gallon reef tank", "contains", "2 clownfish"), '
    '("75-gallon reef tank", "contains", "3 blue-green chromis"), '
    '("75-gallon reef tank", "contains", "5 hermit crabs")]\n\n'
    "Now extract triples from:\n\n"
    'Text: "{text}"'
)

_VALID_MODES = {"factual", "conversation"}

# Regex for escaped whitespace literals (the string \n, not an actual newline)
_ESCAPED_WS = re.compile(r"\\[nrt]")
# Collapse runs of whitespace to a single space
_MULTI_WS = re.compile(r"\s{2,}")


def _sanitize_text(text: str) -> str:
    """Clean input text before LLM extraction.

    Converts escaped whitespace literals (``\\n``, ``\\t``) to spaces,
    strips backslash-escaped quotes, and collapses runs of whitespace.
    This reduces JSON parse failures from noisy conversational data
    (markdown, escaped strings, decorative formatting).
    """
    # Escaped whitespace literals → space
    text = _ESCAPED_WS.sub(" ", text)
    # Backslash-escaped quotes → bare quotes
    text = text.replace("\\'", "'").replace('\\"', '"')
    # Collapse whitespace
    text = _MULTI_WS.sub(" ", text).strip()
    return text


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

    def extract(
        self, text: str, *, mode: str = "factual",
    ) -> list[tuple[str, str, str]]:
        """Extract (subject, predicate, object) triples from a single text.

        Args:
            text: The source text to extract triples from.
            mode: Extraction mode. ``"factual"`` uses the FactConsolidation
                prompt for template sentences. ``"conversation"`` uses a
                prompt tuned for chat turns (possessions, preferences,
                events, quantities).

        Raises:
            ValueError: If *mode* is not one of the valid modes.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES!r}, got {mode!r}"
            )
        results = self._call_llm([text], mode=mode)
        return results[0] if results else []

    def extract_batch(
        self, texts: list[str], *, mode: str = "factual",
    ) -> list[list[tuple[str, str, str]]]:
        """Extract triples from a batch of texts.

        Calls the LLM once per text (vLLM handles batching internally
        via continuous batching when requests arrive concurrently).

        Args:
            texts: List of source texts.
            mode: Extraction mode (see :meth:`extract`).

        Raises:
            ValueError: If *mode* is not one of the valid modes.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES!r}, got {mode!r}"
            )
        if not texts:
            return []
        return self._call_llm(texts, mode=mode)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        """Lazy-init the OpenAI client and auto-detect model if needed."""
        if self._client is not None:
            return
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

        # Auto-detect model from vLLM server when using default placeholder
        if self._model == "qwen3-triple-extractor":
            try:
                models = self._client.models.list()
                if models.data:
                    self._model = models.data[0].id
                    _log.info("Auto-detected model: %s", self._model)
            except Exception:
                pass  # keep default, will fail on first extraction

        _log.info(
            "Connecting to vLLM at %s (model=%s)",
            self._base_url, self._model,
        )

    def _call_llm_single(self, text: str, *, mode: str = "factual") -> dict:
        """Call vLLM for a single text. Returns parsed JSON or None on error.

        Args:
            text: Source text to extract from.
            mode: ``"factual"`` or ``"conversation"`` — selects the prompt.
        """
        if mode == "conversation":
            prompt = CONVERSATION_EXTRACTION_PROMPT.format(text=text)
        else:
            prompt = EXTRACTION_PROMPT.format(text=text)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=1024 if mode == "conversation" else self._max_tokens,
                seed=42,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                response_format=RESPONSE_FORMAT,
            )
            content = response.choices[0].message.content or ""
            # Strip thinking tags if model ignores enable_thinking=False
            if "<think>" in content:
                idx = content.rfind("</think>")
                if idx >= 0:
                    content = content[idx + len("</think>"):].strip()
            # Find JSON object in content (may have non-JSON prefix/suffix)
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]
            return json.loads(content)
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
        self, texts: list[str], *, mode: str = "factual",
    ) -> list[list[tuple[str, str, str]]]:
        """Call vLLM concurrently for all texts, then ground triples.

        Uses ThreadPoolExecutor so vLLM's continuous batching processes
        requests in parallel on the GPU. Typically 5-10x faster than
        sequential calls for batches of 10-20 sentences.

        Args:
            texts: Source texts to extract from.
            mode: ``"factual"`` or ``"conversation"`` — forwarded to
                :meth:`_call_llm_single`.
        """
        self._ensure_client()
        if not texts:
            return []

        # Sanitize once — LLM sees clean text, grounding matches against it
        clean_texts = [_sanitize_text(t) for t in texts]

        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        call_fn = partial(self._call_llm_single, mode=mode)

        # Send all requests concurrently — vLLM batches them on the GPU
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(len(clean_texts), 16)) as pool:
            parsed_results = list(pool.map(call_fn, clean_texts))
        elapsed = time.perf_counter() - t0

        _log.info(
            "LLM batch: %d texts in %.1fs (%.3fs/text)",
            len(texts), elapsed, elapsed / len(texts) if texts else 0,
        )

        # Ground and collect triples (CPU-only, fast)
        return [
            self._ground_and_collect(text, parsed)
            for text, parsed in zip(clean_texts, parsed_results)
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
