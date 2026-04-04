"""alias_resolver.py -- Query-time entity alias resolution via local LLM.

Given a question and a set of candidate texts (from BM25), resolves generic
references in the question to specific entity names found in the candidates.

Example:
    Q: "What loyalty program did I enroll in?"
    BM25 hit: "[2024/03/15] (user) I enrolled in Delta SkyMiles today."
    → resolved entities: ["Delta SkyMiles"]

These resolved entities are then used for a second-pass entity index lookup,
bridging the semantic gap between question phrasing and stored entity names.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)

_V4_PROMPT = (
    "You are given a question and a text. The question uses generic references. "
    "The text contains the specific names, numbers, and details.\n\n"
    "Map each generic reference in the question to the specific entity from the text.\n\n"
    "Rules:\n"
    "- 'how many X' or 'how much' → resolve to the exact number+unit (e.g. '12 fish', '$2,400')\n"
    "- Keep the COMPLETE entity name as it appears in the text — include locations, qualifiers, dosages\n"
    "  e.g. 'bereavement support group at St. Mary\\'s' NOT just 'support group'\n"
    "  e.g. 'Cyclobenzaprine 10mg' NOT just 'Cyclobenzaprine'\n"
    "- 'what plants/items/things' → list ALL specific items mentioned ('marigolds and tomatoes')\n"
    "- 'my daughter/son/friend' → resolve to their actual name\n"
    "- Include ALL resolvable references — names, amounts, people, places\n\n"
    "Example:\n"
    "Q: How much did my friend spend at the clinic?\n"
    "T: Jake paid $350 at the downtown sports clinic on Main St.\n"
    "→ my friend→Jake, how much→$350, the clinic→downtown sports clinic on Main St."
)

_ALIAS_SCHEMA = {
    "type": "object",
    "properties": {
        "aliases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reference": {"type": "string"},
                    "resolved_name": {"type": "string"},
                },
                "required": ["reference", "resolved_name"],
            },
        }
    },
    "required": ["aliases"],
}


class AliasResolver:
    """Resolves generic question references to specific entities via local LLM."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str = "llama-3.2-3b-instruct",
        api_key: str = "lm-studio",
        max_candidates: int = 5,
    ):
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._max_candidates = max_candidates
        self._client: openai.OpenAI | None = None

    def _ensure_client(self) -> openai.OpenAI:
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                base_url=self._base_url, api_key=self._api_key,
            )
        return self._client

    def resolve(
        self,
        question: str,
        candidate_texts: list[str],
    ) -> list[str]:
        """Resolve generic references in question against candidate texts.

        Args:
            question: The user's question with generic references.
            candidate_texts: Top BM25 hits to ground resolution against.

        Returns:
            List of resolved entity name strings (lowercased, deduplicated).
        """
        if not candidate_texts:
            return []

        # Combine top candidates into a single context block
        top = candidate_texts[: self._max_candidates]
        context = "\n".join(f"- {t}" for t in top)
        user_msg = f"Question: {question}\nText: {context}"

        client = self._ensure_client()
        try:
            resp = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _V4_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "aliases", "schema": _ALIAS_SCHEMA},
                },
                temperature=0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            aliases = data.get("aliases", [])
        except Exception as e:
            logger.warning("Alias resolution failed: %s", e)
            return []

        # Extract and deduplicate resolved names
        seen: set[str] = set()
        resolved: list[str] = []
        for alias in aliases:
            name = alias.get("resolved_name", "").strip()
            if not name or len(name) < 2:
                continue
            key = name.lower()
            if key not in seen:
                seen.add(key)
                resolved.append(key)
        return resolved
