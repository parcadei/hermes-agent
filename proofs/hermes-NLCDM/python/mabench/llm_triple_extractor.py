"""LLM-based triple extraction via vLLM + structured output.

Uses Qwen3-30B-A3B MoE (3B active) served by vLLM to extract
(subject, predicate, object) triples from arbitrary text.
vLLM's response_format with json_schema enforces valid JSON
at the token-sampling level.

Architecture:
  - vLLM serves the model as an OpenAI-compatible API
  - Thinking mode disabled (enable_thinking=False) for throughput
  - Grounding check: subject and object must be substrings of source text
  - Predicates normalized: lowercased, underscores→spaces, stripped
  - Entity registry: in conversation mode, an incremental registry maps
    canonical entity names to aliases (pronouns, nicknames, roles).
    The LLM resolves aliases during extraction; the registry grows
    across sequential calls so later chunks benefit from earlier ones.

Feeds into _triple_index for predicated KG hop resolution.
Drop-in replacement for spaCy dep-parse template extraction.

Usage::

    extractor = LLMTripleExtractor(base_url="http://localhost:8000/v1")

    # Single text — factual mode (default, for template sentences)
    triples = extractor.extract("The capital of France is Paris.")
    # [("France", "capital", "Paris")]

    # Conversation mode with entity alias resolution
    extractor.extract("My mom Sarah called me.", mode="conversation")
    # registry grows: {"Sarah": ["Mom", "my mom"]}
    extractor.extract("Mom is visiting next week.", mode="conversation")
    # triples: [("Sarah", "visiting", "next week")] — alias resolved

    # Batch
    batch = extractor.extract_batch(["The capital of France is Paris.", ...])

Pickle-safe: OpenAI client excluded from state, recreated on demand.
Entity registry persists across pickle (included in state).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, date
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

# Extended format for conversation mode with entity alias discovery
# and temporal grounding.
RESPONSE_FORMAT_WITH_ALIASES: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "triple_extraction_with_aliases",
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
                            "when": {"type": "string"},
                        },
                        "required": ["subject", "predicate", "object", "when"],
                        "additionalProperties": False,
                    },
                },
                "aliases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "canonical": {"type": "string"},
                            "alias": {"type": "string"},
                        },
                        "required": ["canonical", "alias"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["triples", "aliases"],
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
    "Extract knowledge from a conversation turn as (subject, predicate, object, when) quads.\n\n"
    "Rules:\n"
    "- Extract ALL facts: possessions, preferences, plans, events, visits, "
    "relationships, states, locations, and actions\n"
    "- Even short statements like 'Mom is visiting next week' contain "
    "extractable facts — do not skip them\n"
    "- Extract facts from assistant responses: recommendations, information provided\n"
    "- Preserve entity names exactly as they appear in the text\n"
    "- Include quantities when mentioned "
    '(e.g. "10 neon tetras" not just "neon tetras")\n'
    "- Predicate must be a short relation label "
    '(e.g. "contains" not "currently has inside of it")\n'
    "- One turn can produce many triples — extract ALL facts\n"
    "- Only extract facts explicitly stated in the text\n"
    "- Return an empty triples list for greetings or acknowledgments "
    "with no factual content\n"
    "- Report entity aliases: when a person is referred to by a proper name "
    "AND a role/nickname/pronoun, the PROPER NAME is the canonical form. "
    'Report as canonical="Sarah", alias="Mom" (NOT the reverse). '
    "Proper names are canonical; roles (Mom, Dad), nicknames, and pronouns "
    "(he, she, the little guy) are aliases.\n"
    "- When the entity registry lists known entities, ALWAYS use the canonical "
    "name from the registry as subject/object in triples, not the alias.\n"
    "- TEMPORAL GROUNDING: if a fact has a time dimension (when something "
    "happened, will happen, started, ended, is scheduled, or changed), put the date "
    'in the "when" field as ISO 8601 (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). '
    "Resolve relative expressions ('next week', '3 days ago', 'yesterday', "
    "'last Tuesday') using the reference date: {reference_date}. "
    "In knowledge-update sentences ('started at X last month', 'joined Y last year', "
    "'raised funding two weeks ago'), ALWAYS resolve the relative date — do not leave "
    "'when' empty for facts with temporal markers.\n"
    'For timeless facts (preferences, possessions, relationships), set "when" '
    'to empty string "".\n\n'
    "Examples:\n\n"
    'Text: "I\'m thinking of adding some live plants to my new 20-gallon tank, '
    "which currently has 10 neon tetras, 5 golden honey gouramis, "
    'and a small pleco catfish."\n'
    'Output: [{{"subject": "20-gallon tank", "predicate": "contains", "object": "10 neon tetras", "when": ""}}, '
    '{{"subject": "20-gallon tank", "predicate": "contains", "object": "5 golden honey gouramis", "when": ""}}, '
    '{{"subject": "20-gallon tank", "predicate": "contains", "object": "pleco catfish", "when": ""}}, '
    '{{"subject": "20-gallon tank", "predicate": "status", "object": "new", "when": ""}}]\n\n'
    'Text: "I really prefer Python over JavaScript for backend work, '
    "and I've been using VS Code as my primary editor for about "
    '3 years now."\n'
    'Output: [{{"subject": "Python", "predicate": "preferred over", "object": "JavaScript", "when": ""}}, '
    '{{"subject": "VS Code", "predicate": "role", "object": "primary editor", "when": ""}}, '
    '{{"subject": "VS Code", "predicate": "usage duration", "object": "3 years", "when": ""}}]\n\n'
    'Text: "My mom Sarah is visiting next week and she\'s bringing her famous '
    'lasagna. My brother Jake might come too."\n'
    "(Reference date: 2026-03-15)\n"
    'Output: [{{"subject": "Sarah", "predicate": "visiting", "object": "user", "when": "2026-03-22"}}, '
    '{{"subject": "Sarah", "predicate": "bringing", "object": "famous lasagna", "when": "2026-03-22"}}, '
    '{{"subject": "Jake", "predicate": "relationship", "object": "brother", "when": ""}}, '
    '{{"subject": "Jake", "predicate": "might visit", "object": "user", "when": "2026-03-22"}}]\n'
    'Aliases: [{{"canonical": "Sarah", "alias": "mom"}}, {{"canonical": "Sarah", "alias": "she"}}, '
    '{{"canonical": "Jake", "alias": "my brother"}}]\n\n'
    'Text: "I graduated from MIT in 2019 and started working at Google last March."\n'
    "(Reference date: 2026-03-29)\n"
    'Output: [{{"subject": "user", "predicate": "graduated from", "object": "MIT", "when": "2019-06-01"}}, '
    '{{"subject": "user", "predicate": "started working at", "object": "Google", "when": "2025-03-01"}}]\n'
    "Aliases: []\n\n"
    'Text: "We adopted Bobo from the shelter about 3 weeks ago. '
    'He\'s a golden retriever mix and he loves the backyard."\n'
    "(Reference date: 2026-03-29)\n"
    'Output: [{{"subject": "Bobo", "predicate": "adopted from", "object": "shelter", "when": "2026-03-08"}}, '
    '{{"subject": "Bobo", "predicate": "breed", "object": "golden retriever mix", "when": ""}}, '
    '{{"subject": "Bobo", "predicate": "loves", "object": "backyard", "when": ""}}]\n'
    'Aliases: [{{"canonical": "Bobo", "alias": "he"}}]\n\n'
    'Text: "Mom called again. She wants to bring Bobo to the park."\n'
    "(Given known entities: Sarah = mom, Bobo = my dog)\n"
    'Output: [{{"subject": "Sarah", "predicate": "called", "object": "user", "when": ""}}, '
    '{{"subject": "Sarah", "predicate": "wants to bring", "object": "Bobo to the park", "when": ""}}]\n'
    "Aliases: []\n\n"
    'Text: "She used to work at Google but started at OpenAI last month. '
    'She also moved from Mountain View to San Francisco."\n'
    "(Reference date: 2026-03-29)\n"
    'Output: [{{"subject": "user_she", "predicate": "used to work at", "object": "Google", "when": ""}}, '
    '{{"subject": "user_she", "predicate": "started working at", "object": "OpenAI", "when": "2026-02-01"}}, '
    '{{"subject": "user_she", "predicate": "moved from", "object": "Mountain View", "when": "2026-02-01"}}, '
    '{{"subject": "user_she", "predicate": "moved to", "object": "San Francisco", "when": "2026-02-01"}}]\n'
    "Aliases: []\n\n"
    'Text: "I finished my PhD at Stanford in 2022 and joined a startup last year. '
    'The company just raised Series B funding two weeks ago."\n'
    "(Reference date: 2026-03-29)\n"
    'Output: [{{"subject": "user", "predicate": "finished PhD at", "object": "Stanford", "when": "2022-06-01"}}, '
    '{{"subject": "user", "predicate": "joined", "object": "startup", "when": "2025-01-01"}}, '
    '{{"subject": "startup", "predicate": "raised", "object": "Series B funding", "when": "2026-03-15"}}]\n'
    "Aliases: []\n\n"
    'Text: "Hi! Can you help me with something?"\n'
    "Output: []\n\n"
    "Now extract triples from:\n\n"
    "(Reference date: {reference_date})\n"
    'Text: "{text}"'
)

_VALID_MODES = {"factual", "conversation"}


def _parse_event_time(when_str: str) -> datetime | None:
    """Parse an ISO 8601 date/datetime string into a datetime object.

    Returns None for empty strings or unparseable values.
    Accepts YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS formats.
    """
    if not when_str:
        return None
    try:
        # Try full datetime first, then date-only
        if "T" in when_str:
            return datetime.fromisoformat(when_str)
        return datetime.fromisoformat(when_str + "T00:00:00")
    except (ValueError, TypeError):
        _log.debug("Unparseable temporal expression: %r", when_str)
        return None

# Regex for escaped whitespace literals (the string \n, not an actual newline)
_ESCAPED_WS = re.compile(r"\\[nrt]")
# Collapse runs of whitespace to a single space
_MULTI_WS = re.compile(r"\s{2,}")

# Type alias for the entity registry: canonical name → list of aliases.
EntityRegistry = dict[str, list[str]]


def _format_entity_registry(registry: EntityRegistry) -> str:
    """Format the entity registry as context for the extraction prompt."""
    if not registry:
        return ""
    lines = ["Known entities from prior conversation:\n"]
    for canonical, aliases in registry.items():
        if aliases:
            alias_str = ", ".join(f'"{a}"' for a in aliases)
            lines.append(f'- {canonical} (also known as: {alias_str})')
        else:
            lines.append(f"- {canonical}")
    lines.append("")
    lines.append(
        "IMPORTANT: When a mention in the text refers to a known entity, "
        "use the CANONICAL name (before the parentheses) as subject/object.\n"
        "If you discover new aliases for known entities or new entities with "
        "aliases, report them in the 'aliases' field.\n"
    )
    return "\n".join(lines)


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


def _entity_grounded(
    entity: str, source_text_lower: str, registry: EntityRegistry,
) -> bool:
    """Check if an entity is grounded in source text, accounting for aliases.

    An entity is grounded if:
    1. It appears literally in the source text, OR
    2. It is a canonical name in the registry and at least one of
       its aliases appears in the source text (alias resolution).
    """
    if entity.lower() in source_text_lower:
        return True
    # Check if entity is a canonical name with a grounded alias
    aliases = registry.get(entity, [])
    return any(a.lower() in source_text_lower for a in aliases)


def _ground_triple(
    subject: str, predicate: str, obj: str, source_text: str,
    registry: EntityRegistry | None = None,
) -> bool:
    """Grounding check: subject and object must be substrings of source text.

    The predicate may be inferred (e.g. "capital" from "X is the capital of Y")
    so it is NOT required to appear literally. But subject and object must be
    grounded in the source text to prevent hallucination.

    When an entity registry is provided, canonical names are accepted if
    one of their aliases appears in the source text.
    """
    text_lower = source_text.lower()
    reg = registry or {}
    return (
        _entity_grounded(subject, text_lower, reg)
        and _entity_grounded(obj, text_lower, reg)
    )


def _normalize_predicate(pred: str) -> str:
    """Normalize predicate to canonical form for consistent KG lookups."""
    return pred.lower().replace("_", " ").strip()


def resolve_alias(mention: str, registry: EntityRegistry) -> str | None:
    """Resolve a mention to its canonical entity name using the registry.

    Deterministic, O(total_aliases) lookup. Used at query time — no LLM.
    Tries exact match first, then case-insensitive, then substring.

    Returns:
        Canonical name if found, None otherwise.
    """
    mention_lower = mention.lower()
    # Pass 1: exact match on canonical names
    for canonical in registry:
        if canonical.lower() == mention_lower:
            return canonical
    # Pass 2: exact match on aliases
    for canonical, aliases in registry.items():
        for alias in aliases:
            if alias.lower() == mention_lower:
                return canonical
    # Pass 3: substring match (longest match wins)
    best: str | None = None
    best_len = 0
    for canonical, aliases in registry.items():
        for alias in aliases:
            if alias.lower() in mention_lower and len(alias) > best_len:
                best = canonical
                best_len = len(alias)
        if canonical.lower() in mention_lower and len(canonical) > best_len:
            best = canonical
            best_len = len(canonical)
    return best


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
        entity_registry: EntityRegistry | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_grounding = enable_grounding
        self._client: Any = None
        self._stats = {"texts": 0, "triples": 0, "grounded": 0, "discarded": 0, "aliases": 0, "temporal": 0}
        self._entity_registry: EntityRegistry = entity_registry or {}
        self._last_event_times: list[list[datetime | None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, text: str, *, mode: str = "factual",
        reference_date: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """Extract (subject, predicate, object) triples from a single text.

        In conversation mode with an entity registry, discovered aliases
        are merged into :attr:`entity_registry` automatically.

        Args:
            text: The source text to extract triples from.
            mode: Extraction mode. ``"factual"`` uses the FactConsolidation
                prompt for template sentences. ``"conversation"`` uses a
                prompt tuned for chat turns (possessions, preferences,
                events, quantities) and supports entity alias + temporal
                resolution.
            reference_date: ISO 8601 date string for resolving relative
                temporal expressions. Only used in conversation mode.
                Defaults to today's date.

        Raises:
            ValueError: If *mode* is not one of the valid modes.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES!r}, got {mode!r}"
            )
        results = self._call_llm([text], mode=mode, reference_date=reference_date)
        return results[0] if results else []

    def extract_batch(
        self, texts: list[str], *, mode: str = "factual",
        reference_date: str | None = None,
    ) -> list[list[tuple[str, str, str]]]:
        """Extract triples from a batch of texts.

        Calls the LLM once per text (vLLM handles batching internally
        via continuous batching when requests arrive concurrently).

        Note: In conversation mode, alias discovery within a batch is
        NOT incremental — each text sees the registry as it was before
        the batch started. For sequential alias accumulation, use
        :meth:`extract` in a loop.

        Args:
            texts: List of source texts.
            mode: Extraction mode (see :meth:`extract`).
            reference_date: ISO 8601 date string for resolving relative
                temporal expressions (conversation mode only).

        Raises:
            ValueError: If *mode* is not one of the valid modes.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES!r}, got {mode!r}"
            )
        if not texts:
            return []
        return self._call_llm(texts, mode=mode, reference_date=reference_date)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def entity_registry(self) -> EntityRegistry:
        """Current entity registry (canonical name → aliases). Grows during conversation extraction."""
        return self._entity_registry

    @entity_registry.setter
    def entity_registry(self, registry: EntityRegistry) -> None:
        self._entity_registry = registry

    @property
    def last_event_times(self) -> list[list[datetime | None]]:
        """Event times from the last extraction call.

        Parallel to the returned triples: ``last_event_times[text_idx][triple_idx]``
        is the parsed datetime for that triple, or None if no temporal info.
        """
        return self._last_event_times

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

    def _call_llm_single(
        self, text: str, *, mode: str = "factual",
        reference_date: str | None = None,
    ) -> dict:
        """Call vLLM for a single text. Returns parsed JSON or None on error.

        Args:
            text: Source text to extract from.
            mode: ``"factual"`` or ``"conversation"`` — selects the prompt.
            reference_date: ISO 8601 date string for resolving relative
                temporal expressions (e.g. "next week"). Only used in
                conversation mode.

        In conversation mode, the entity registry is injected as context
        and the extended response format (with aliases and temporal) is used.
        """
        use_aliases = mode == "conversation"
        if mode == "conversation":
            ref_date = reference_date or date.today().isoformat()
            registry_ctx = _format_entity_registry(self._entity_registry)
            prompt = registry_ctx + CONVERSATION_EXTRACTION_PROMPT.format(
                text=text, reference_date=ref_date,
            )
        else:
            prompt = EXTRACTION_PROMPT.format(text=text)
        resp_format = RESPONSE_FORMAT_WITH_ALIASES if use_aliases else RESPONSE_FORMAT
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
                response_format=resp_format,
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

    def _canonicalize(self, entity: str) -> str:
        """Resolve entity to canonical name via registry, or return as-is."""
        resolved = resolve_alias(entity, self._entity_registry)
        return resolved if resolved else entity

    def _ground_and_collect(
        self, text: str, parsed: dict | None,
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]], list[datetime | None]]:
        """Ground triples from a parsed LLM response against source text.

        After grounding, subjects and objects are resolved to canonical
        names using the entity registry (e.g. "Mom" → "Sarah").

        Returns:
            (grounded_triples, alias_pairs, event_times) where:
            - grounded_triples: list of (subject, predicate, object)
            - alias_pairs: list of (canonical, alias) discovered
            - event_times: parallel list of parsed datetimes (None if no temporal info)
        """
        if parsed is None:
            return [], [], []
        # Extract alias pairs FIRST so new aliases are available for
        # canonicalization of triples in the same chunk.
        alias_pairs: list[tuple[str, str]] = []
        for a in parsed.get("aliases", []):
            canonical = a.get("canonical", "").strip()
            alias = a.get("alias", "").strip()
            if canonical and alias and canonical.lower() != alias.lower():
                alias_pairs.append((canonical, alias))
        # Merge aliases before canonicalizing triples
        if alias_pairs:
            self._merge_aliases(alias_pairs)

        grounded: list[tuple[str, str, str]] = []
        event_times: list[datetime | None] = []
        for t in parsed.get("triples", []):
            s = t.get("subject", "").strip()
            p = _normalize_predicate(t.get("predicate", ""))
            o = t.get("object", "").strip()
            when_str = t.get("when", "").strip()
            if not s or not p or not o:
                continue
            if self._enable_grounding:
                if _ground_triple(s, p, o, text, self._entity_registry):
                    s = self._canonicalize(s)
                    o = self._canonicalize(o)
                    grounded.append((s, p, o))
                    event_times.append(_parse_event_time(when_str))
                    self._stats["grounded"] += 1
                else:
                    self._stats["discarded"] += 1
            else:
                s = self._canonicalize(s)
                o = self._canonicalize(o)
                grounded.append((s, p, o))
                event_times.append(_parse_event_time(when_str))
        self._stats["texts"] += 1
        self._stats["triples"] += len(grounded)
        self._stats["aliases"] += len(alias_pairs)
        n_temporal = sum(1 for et in event_times if et is not None)
        self._stats["temporal"] += n_temporal
        return grounded, alias_pairs, event_times

    def _merge_aliases(self, alias_pairs: list[tuple[str, str]]) -> None:
        """Merge discovered alias pairs into the entity registry."""
        for canonical, alias in alias_pairs:
            existing = self._entity_registry.setdefault(canonical, [])
            if alias.lower() not in {a.lower() for a in existing}:
                existing.append(alias)
                _log.debug("Registry: %s += alias %r", canonical, alias)

    def _call_llm(
        self, texts: list[str], *, mode: str = "factual",
        reference_date: str | None = None,
    ) -> list[list[tuple[str, str, str]]]:
        """Call vLLM concurrently for all texts, then ground triples.

        Uses ThreadPoolExecutor so vLLM's continuous batching processes
        requests in parallel on the GPU. Typically 5-10x faster than
        sequential calls for batches of 10-20 sentences.

        In conversation mode, discovered aliases are merged into
        :attr:`entity_registry` after grounding. Parsed event times
        are stored in :attr:`_last_event_times`.

        Args:
            texts: Source texts to extract from.
            mode: ``"factual"`` or ``"conversation"`` — forwarded to
                :meth:`_call_llm_single`.
            reference_date: ISO 8601 date for temporal resolution.
        """
        self._ensure_client()
        if not texts:
            return []

        # Sanitize once — LLM sees clean text, grounding matches against it
        clean_texts = [_sanitize_text(t) for t in texts]

        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        call_fn = partial(
            self._call_llm_single, mode=mode, reference_date=reference_date,
        )

        # Send all requests concurrently — vLLM batches them on the GPU
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(len(clean_texts), 16)) as pool:
            parsed_results = list(pool.map(call_fn, clean_texts))
        elapsed = time.perf_counter() - t0

        _log.info(
            "LLM batch: %d texts in %.1fs (%.3fs/text)",
            len(texts), elapsed, elapsed / len(texts) if texts else 0,
        )

        # Ground, canonicalize, and collect triples + aliases + event_times.
        # _ground_and_collect merges aliases into registry before canonicalizing.
        all_triples: list[list[tuple[str, str, str]]] = []
        all_event_times: list[list[datetime | None]] = []
        for text, parsed in zip(clean_texts, parsed_results):
            triples, _alias_pairs, event_times = self._ground_and_collect(text, parsed)
            all_triples.append(triples)
            all_event_times.append(event_times)

        self._last_event_times = all_event_times

        if self._entity_registry:
            _log.info(
                "Entity registry: %d entities, %d total aliases",
                len(self._entity_registry),
                sum(len(v) for v in self._entity_registry.values()),
            )

        return all_triples

    # ------------------------------------------------------------------
    # Pickle support (exclude client, lazy-recreate)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
