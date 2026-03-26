"""Embedding-based relevance scoring for Hermes memory retrieval.

Provides EmbeddingRelevanceScorer, a cosine-similarity scorer using
sentence embeddings (Qwen3-0.6B by default) with disk-persistent caching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure hermes-memory is on the path for RelevanceScorer import
_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))

from hermes_memory.orchestrator import RelevanceScorer


class EmbeddingRelevanceScorer(RelevanceScorer):
    """Cosine-similarity relevance scorer using a sentence embedding model.

    Replaces JaccardRelevance with real embedding-based scoring.
    Embeddings are cached per content string to avoid redundant inference.
    """

    _DISK_CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "embed_cache"

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, np.ndarray] = {}
        self._load_disk_cache()

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel.from_pretrained(self._model_name, torch_dtype=torch.float32).to(self._device)
        self._model.eval()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string, with caching."""
        if text in self._cache:
            return self._cache[text]
        # Delegate to batch path for single text
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
        """Embed multiple texts, returning cached results where available.

        Uncached texts are embedded in GPU batches for throughput.
        """
        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            self._ensure_model()
            import torch

            # Process in batches to avoid OOM on very large chunks
            for start in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[start : start + batch_size]
                inputs = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._model(**inputs)
                last_hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embs = pooled.cpu().numpy().astype(np.float64)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / (norms + 1e-12)

                for j, emb in enumerate(embs):
                    idx = uncached_indices[start + j]
                    results[idx] = emb
                    self._cache[uncached_texts[start + j]] = emb

        return results

    def _load_disk_cache(self):
        """Load cached embeddings from disk if available."""
        cache_file = self._DISK_CACHE_DIR / "embeddings.npz"
        if cache_file.exists():
            data = np.load(str(cache_file), allow_pickle=True)
            texts = data["texts"]
            embs = data["embeddings"]
            for t, e in zip(texts, embs):
                self._cache[str(t)] = e
            print(f"Loaded {len(self._cache)} cached embeddings from disk")

    def save_disk_cache(self):
        """Persist embedding cache to disk."""
        self._DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = self._DISK_CACHE_DIR / "embeddings.npz"
        texts = list(self._cache.keys())
        embs = np.array(list(self._cache.values()))
        np.savez(str(cache_file), texts=texts, embeddings=embs)
        print(f"Saved {len(texts)} embeddings to {cache_file}")

    def score(self, query: str, content: str) -> float:
        """Cosine similarity between query and content embeddings."""
        q_emb = self.embed(query)
        c_emb = self.embed(content)
        return float(np.dot(q_emb, c_emb))
