"""Incremental BM25 index over memory store facts.

Pure Python implementation (math + collections only, no numpy).
Tokenization: lowercase + strip punctuation + split whitespace.

BM25 formula (Okapi BM25):
    score(q, d) = sum over t in q of:
        IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * dl / avgdl))

    IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

where:
    N     = total number of documents
    df(t) = number of documents containing term t
    tf(t,d) = frequency of term t in document d
    dl    = length of document d (in tokens)
    avgdl = average document length across corpus
    k1, b = free parameters (defaults: k1=1.5, b=0.75)
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on word characters."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Incremental BM25 index over memory store facts.

    Pure Python implementation, no external dependencies.
    Tokenization: lowercase + strip punctuation + split whitespace.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize empty index with BM25 parameters."""
        self.k1 = k1
        self.b = b

        # doc_id -> list of tokens
        self._docs: dict[int, list[str]] = {}
        # term -> number of documents containing that term
        self._df: Counter = Counter()
        # Total token count across all documents (for computing avgdl)
        self._total_tokens: int = 0

    def add(self, doc_id: int, text: str) -> None:
        """Add a single document. Called during _store() for each fact.

        If doc_id already exists, the previous document is replaced:
        its terms are removed from df counts and the new text is indexed.
        """
        # Handle duplicate doc_id: remove old document's contribution first
        if doc_id in self._docs:
            old_tokens = self._docs[doc_id]
            self._total_tokens -= len(old_tokens)
            for term in set(old_tokens):
                self._df[term] -= 1
                if self._df[term] <= 0:
                    del self._df[term]

        tokens = _tokenize(text)
        self._docs[doc_id] = tokens
        self._total_tokens += len(tokens)

        for term in set(tokens):
            self._df[term] += 1

    def add_batch(self, docs: list[tuple[int, str]]) -> None:
        """Add multiple documents. Called during rebuild from cache."""
        for doc_id, text in docs:
            self.add(doc_id, text)

    def query(self, query_text: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_id, score) pairs ranked by BM25 score descending.

        Only documents with score > 0 are returned.
        """
        if not self._docs:
            return []

        q_tokens = _tokenize(query_text)
        if not q_tokens:
            return []

        n = len(self._docs)
        avgdl = self._total_tokens / n if n > 0 else 1.0

        scores: list[tuple[int, float]] = []
        for doc_id, doc_tokens in self._docs.items():
            score = 0.0
            dl = len(doc_tokens)
            tf = Counter(doc_tokens)

            for term in q_tokens:
                df_t = self._df.get(term, 0)
                if df_t == 0:
                    continue

                idf = math.log((n - df_t + 0.5) / (df_t + 0.5) + 1.0)
                tf_val = tf.get(term, 0)
                if tf_val == 0:
                    continue

                numerator = tf_val * (self.k1 + 1.0)
                denominator = tf_val + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                score += idf * numerator / denominator

            if score > 0:
                scores.append((doc_id, score))

        # Sort by score descending, then by doc_id ascending for stability
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:top_k]

    def __len__(self) -> int:
        """Number of indexed documents."""
        return len(self._docs)
