"""Standalone triple extractor using GLiNER2 zero-shot relation extraction.

Follows the composition pattern: standalone class, config-gated, no mixins.
Produces (subject, predicate, object) triples consumed by:
  - KnowledgeGraph (PPR traversal edges)
  - SupersessionState.triple_index (conflict detection)
  - TriadicMemory (associative binding)

GLiNER2 is a zero-shot relation extraction model that takes arbitrary
relation type labels and extracts (head, relation, tail) spans from text.
With threshold=0.5 and 28 universal relation types, it produces ~2.1
triples per fact — dense coverage compared to spaCy dep-parse (~45 unique
S,P keys from 18K facts).

Usage::

    extractor = GLiNER2Extractor()

    # Single text
    triples = extractor.extract("Abbiati plays for AC Milan")
    # [("Abbiati", "member_of", "AC Milan")]

    # Batch (faster: 117ms/fact vs 161ms single)
    batch_triples = extractor.extract_batch(["Abbiati plays for AC Milan", ...])
    # [[("Abbiati", "member_of", "AC Milan")], ...]

    # Pickle-safe for eval_cache (model excluded, lazy-reloaded)
    import pickle
    data = pickle.dumps(extractor)
    restored = pickle.loads(data)
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger("gliner2_extractor")

# ---------------------------------------------------------------------------
# Default relation schema — 28 universal types covering the major ontological
# categories.  GLiNER2 is zero-shot so these are free-text labels, not a
# fixed enum.  Callers can override via constructor.
# ---------------------------------------------------------------------------

DEFAULT_RELATION_TYPES: list[str] = [
    # Taxonomic & Classification
    "instance_of",
    "subclass_of",
    "same_as",
    # Mereological & Compositional
    "part_of",
    "has_part",
    "made_of",
    # Spatial & Temporal
    "located_in",
    "origin",
    "time_of",
    "adjacent_to",
    # Causal & Functional
    "causes",
    "used_for",
    "requires",
    "results_in",
    # Agentive & Social
    "created_by",
    "member_of",
    "works_in",
    "owns",
    "leads",
    "related_to",
    # Descriptive & Attributive
    "has_property",
    "has_value",
    "named",
    "known_as",
    # Epistemic & Meta
    "derived_from",
    "contrasts_with",
    "similar_to",
    "supersedes",
]


class GLiNER2Extractor:
    """Standalone triple extractor using GLiNER2 zero-shot relation extraction.

    Constructed once by HermesAgent (or composed pipeline) and called at
    ingestion time to annotate facts with dense triples.  The model is
    lazy-loaded on first extraction call to avoid import-time overhead and
    allow the module to be imported in environments without gliner2.

    Pickle-safe: ``__getstate__`` / ``__setstate__`` exclude the heavy model
    object so that eval_cache can serialize the extractor.  The model is
    re-loaded transparently on next use after deserialization.
    """

    def __init__(
        self,
        model_name: str = "fastino/gliner2-base-v1",
        relation_types: list[str] | None = None,
        threshold: float = 0.5,
        batch_size: int = 8,
    ) -> None:
        self._model_name = model_name
        self._relation_types = (
            list(relation_types) if relation_types is not None
            else list(DEFAULT_RELATION_TYPES)
        )
        self._threshold = threshold
        self._batch_size = batch_size
        self._model: Any = None  # lazy-loaded GLiNER2 instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[tuple[str, str, str]]:
        """Extract (subject, predicate, object) triples from a single text.

        Returns a list of ``(subject, predicate, object)`` tuples.

        Post-processing applied:
          1. Whitespace is stripped from all triple components.
          2. Self-referential triples where ``subject == object`` (after
             lowercasing and stripping) are filtered out.
        """
        self._ensure_model()
        result = self._model.extract_relations(
            text, self._relation_types, threshold=self._threshold,
        )
        return self._parse_result(result)

    def extract_batch(
        self, texts: list[str],
    ) -> list[list[tuple[str, str, str]]]:
        """Extract triples from multiple texts using batch processing.

        Returns a list of triple-lists, one per input text.  More efficient
        than calling :meth:`extract` in a loop because GLiNER2 batches the
        forward pass internally.
        """
        if not texts:
            return []
        self._ensure_model()
        results = self._model.batch_extract_relations(
            texts,
            self._relation_types,
            threshold=self._threshold,
            batch_size=self._batch_size,
        )
        return [self._parse_result(r) for r in results]

    @property
    def stats(self) -> dict[str, Any]:
        """Return diagnostic info about the extractor."""
        return {
            "model_name": self._model_name,
            "relation_types": len(self._relation_types),
            "threshold": self._threshold,
            "model_loaded": self._model is not None,
        }

    # ------------------------------------------------------------------
    # Serialization (pickle-safe for eval_cache)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_model"] = None  # Don't pickle the heavy model
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Model will be lazy-loaded on next extract/extract_batch call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load the GLiNER2 model on first use."""
        if self._model is not None:
            return
        from gliner2 import GLiNER2

        _log.info("Loading GLiNER2 model: %s", self._model_name)
        self._model = GLiNER2.from_pretrained(self._model_name)
        _log.info("GLiNER2 model loaded")

    def _parse_result(
        self, result: dict[str, Any],
    ) -> list[tuple[str, str, str]]:
        """Parse GLiNER2 output into (subject, predicate, object) triples.

        GLiNER2 returns::

            {
                "relation_extraction": {
                    "member_of": [("Abbiati", "AC Milan"), ...],
                    "works_in": [("Abbiati", "AC Milan"), ...],
                }
            }

        Each key is a relation type, each value is a list of (head, tail)
        pairs.  We convert to (head, relation_type, tail) triples and apply
        the self-referential filter.
        """
        triples: list[tuple[str, str, str]] = []
        rel_dict = result.get("relation_extraction", {})
        for predicate, pairs in rel_dict.items():
            pred_stripped = predicate.strip()
            for pair in pairs:
                if len(pair) < 2:
                    continue
                subj = str(pair[0]).strip()
                obj = str(pair[1]).strip()
                # Filter self-referential triples
                if subj.lower() == obj.lower():
                    continue
                # Filter empty components
                if not subj or not obj or not pred_stripped:
                    continue
                triples.append((subj, pred_stripped, obj))
        return triples
