"""Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Reranks retrieval candidates by feeding [query][SEP][document] pairs through
a cross-encoder that outputs a single relevance score per pair.  This was
the biggest contributor in agentmemory's ablation (-30.7 pp).

Model is lazy-loaded: the 22M-param cross-encoder is only instantiated on the
first call to rerank(), so --cross_encoder=False costs zero startup time.

Interface matches EntityReranker: receives list[dict] with {"text", "score"},
returns the same structure re-sorted by cross_encoder_score.  The original
"score" field is preserved; "cross_encoder_score" is added as an additional
field.

Usage::

    reranker = CrossEncoderReranker()

    # At retrieval (called by Retriever._retrieve_v2):
    results = reranker.rerank(query_text, results, top_k=10)
"""

from __future__ import annotations


# Lazy import — only loaded when _ensure_model() is called
CrossEncoder = None


class CrossEncoderReranker:
    """Standalone cross-encoder reranker for retrieval candidates.

    Constructed once by HermesAgent and passed to Retriever.
    Revert = don't construct it; baseline code stays untouched.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._model = None  # lazy load

    def _ensure_model(self) -> None:
        """Load the cross-encoder model on first use."""
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder as _CrossEncoder

        global CrossEncoder
        CrossEncoder = _CrossEncoder
        self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank results by cross-encoder relevance score.

        Args:
            query: The query text.
            results: Retrieval candidates, each with at least {"text", "score"}.
            top_k: Override instance top_k. If None, uses self._top_k.

        Returns:
            Top-k results re-sorted by cross_encoder_score (descending).
            Each result dict gets an additional "cross_encoder_score" float
            field.  The original "score" field is preserved.
        """
        if not results:
            return results

        self._ensure_model()
        k = top_k if top_k is not None else self._top_k

        # Build [query, document] pairs for the cross-encoder
        pairs = [[query, r["text"]] for r in results]
        scores = self._model.predict(pairs)

        # Annotate each result with cross-encoder score
        for r, s in zip(results, scores):
            r["cross_encoder_score"] = float(s)

        # Sort by cross-encoder score descending and truncate
        results.sort(key=lambda r: r["cross_encoder_score"], reverse=True)
        return results[:k]
