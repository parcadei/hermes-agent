"""Embedding-based relevance scoring for Hermes memory retrieval.

Provides EmbeddingRelevanceScorer, a cosine-similarity scorer using
sentence embeddings (all-mpnet-base-v2 by default, 768-dim) with
disk-persistent caching.

CACHE INVALIDATION: The on-disk embed cache (output/embed_cache/embeddings.npz)
is keyed by text only, not by model name. When switching embedding models
(e.g. from Qwen3-0.6B 1024-dim to all-mpnet-base-v2 768-dim), the cache
MUST be invalidated (deleted) to avoid dimension mismatches and stale vectors.
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

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self._model_name = model_name
        self._model = None  # SentenceTransformer instance (lazy-loaded)
        self._tokenizer = None  # kept for entity-position methods
        self._cache: dict[str, np.ndarray] = {}
        self._load_disk_cache()

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        self._device = str(self._model.device)
        # Expose tokenizer for entity-position residual methods
        self._tokenizer = self._model.tokenizer
        # Cap _ENTITY_LAYER to model's actual layer count
        num_layers = self._model[0].auto_model.config.num_hidden_layers
        if self._ENTITY_LAYER >= num_layers:
            # Use second-to-last layer (best generic choice for entity probing)
            self._ENTITY_LAYER = num_layers - 2

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

            # SentenceTransformer.encode handles batching, pooling, and
            # normalization internally — no manual tokenize/forward/pool needed.
            embs = self._model.encode(
                uncached_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float64)

            for j, emb in enumerate(embs):
                idx = uncached_indices[j]
                results[idx] = emb
                self._cache[uncached_texts[j]] = emb

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

    # ------------------------------------------------------------------
    # Entity-position residual extraction (arXiv:2603.19664)
    # ------------------------------------------------------------------

    _ENTITY_LAYER = 14  # Capped dynamically in _ensure_model() for smaller models

    @property
    def _raw_model(self):
        """Access the underlying HuggingFace transformer inside SentenceTransformer.

        Entity-residual methods need direct forward pass with output_hidden_states=True,
        which SentenceTransformer.encode() does not expose.
        """
        self._ensure_model()
        return self._model[0].auto_model

    def embed_entity_isolated(
        self, entity_text: str, *, layer: int | None = None,
    ) -> np.ndarray:
        """Embed entity text in isolation (NOT in context).

        Tokenizes just the entity name alone, runs forward pass with
        output_hidden_states=True, extracts hidden state at the specified
        transformer layer, averaged over all entity tokens.

        In-context entity residuals suffer from template context bleed
        (0.006 gap within-template). Isolated entity residuals achieve
        0.134 gap — validated in residual_discrimination_v2.py.

        Returns a normalized vector (dimension matches the embedding model).
        """
        self._ensure_model()
        import torch

        target_layer = layer if layer is not None else self._ENTITY_LAYER
        raw = self._raw_model
        device = next(raw.parameters()).device

        inputs = self._tokenizer(
            [entity_text], return_tensors="pt", padding=True,
            truncation=True, max_length=64,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = raw(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[target_layer][0]  # (seq, d_hidden)
        mask = inputs["attention_mask"][0].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1e-9)

        vec = pooled.cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-12)

    def embed_entity_position(
        self, text: str, entity: str, *, layer: int | None = None,
    ) -> np.ndarray:
        """Extract hidden state at entity mention position.

        Uses token-ID subsequence matching to find the entity span in
        the tokenized text, then averages hidden states over that span
        at the specified transformer layer.  Falls back to first-token
        residual if the entity mention is not found.

        Returns a normalized vector (dimension matches the embedding model).
        """
        self._ensure_model()
        import torch

        target_layer = layer if layer is not None else self._ENTITY_LAYER
        raw = self._raw_model
        device = next(raw.parameters()).device

        inputs = self._tokenizer(
            [text], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = raw(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[target_layer][0]  # (seq, d_hidden)

        # Find entity token span
        span = self._find_entity_tokens(text, entity)
        if span is not None:
            # Offset for special tokens (BOS)
            bare_ids = self._tokenizer.encode(text, add_special_tokens=False)
            full_ids = self._tokenizer.encode(text, add_special_tokens=True)
            bos_offset = len(full_ids) - len(bare_ids)
            adjusted = [p + bos_offset for p in span if p + bos_offset < hidden.shape[0]]
            if adjusted:
                entity_h = hidden[adjusted].mean(dim=0)
            else:
                entity_h = hidden[bos_offset]
        else:
            # Fallback: first real token
            entity_h = hidden[1] if hidden.shape[0] > 1 else hidden[0]

        vec = entity_h.cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-12)

    def embed_entity_position_batch(
        self, texts: list[str], entities: list[str], *, layer: int | None = None,
    ) -> list[np.ndarray]:
        """Batch entity-position residual extraction.

        Processes all texts in a single forward pass.  Falls back to
        first-token for any text where the entity mention is not found.
        """
        if not texts:
            return []

        self._ensure_model()
        import torch

        target_layer = layer if layer is not None else self._ENTITY_LAYER
        raw = self._raw_model
        device = next(raw.parameters()).device

        # Pre-compute entity spans
        entity_spans = [self._find_entity_tokens(t, e) for t, e in zip(texts, entities)]

        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = raw(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[target_layer]  # (batch, seq, d_hidden)
        mask = inputs["attention_mask"]

        results = []
        for i in range(len(texts)):
            h = hidden[i]  # (seq, d_hidden)
            span = entity_spans[i]

            # Compute BOS offset per sample
            m = mask[i].cpu().numpy()
            first_real = int(np.argmax(m))
            bare_ids = self._tokenizer.encode(texts[i], add_special_tokens=False)
            full_ids = self._tokenizer.encode(texts[i], add_special_tokens=True)
            bos_offset = first_real + (len(full_ids) - len(bare_ids))

            if span is not None:
                adjusted = [p + bos_offset for p in span if p + bos_offset < h.shape[0]]
                if adjusted:
                    entity_h = h[adjusted].mean(dim=0)
                else:
                    entity_h = h[bos_offset] if bos_offset < h.shape[0] else h[0]
            else:
                entity_h = h[bos_offset] if bos_offset < h.shape[0] else h[0]

            vec = entity_h.cpu().numpy().astype(np.float64)
            norm = np.linalg.norm(vec)
            results.append(vec / (norm + 1e-12))

        return results

    def _find_entity_tokens(self, text: str, entity: str) -> list[int] | None:
        """Find token indices of entity mention via subsequence matching."""
        full_ids = self._tokenizer.encode(text, add_special_tokens=False)
        entity_ids = self._tokenizer.encode(entity, add_special_tokens=False)

        for i in range(len(full_ids) - len(entity_ids) + 1):
            if full_ids[i:i + len(entity_ids)] == entity_ids:
                return list(range(i, i + len(entity_ids)))

        # Try with leading space (common tokenizer artifact)
        entity_ids_sp = self._tokenizer.encode(" " + entity, add_special_tokens=False)
        for i in range(len(full_ids) - len(entity_ids_sp) + 1):
            if full_ids[i:i + len(entity_ids_sp)] == entity_ids_sp:
                return list(range(i, i + len(entity_ids_sp)))

        return None

    # ------------------------------------------------------------------
    # Factory: construct from --embed_url if provided
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, embed_url: str | None = None, embed_model: str | None = None):
        """Factory that returns an APIEmbeddingScorer if embed_url is set."""
        if embed_url:
            return APIEmbeddingScorer(
                api_url=embed_url,
                model_name=embed_model or "Qwen/Qwen3-Embedding-8B",
            )
        return cls(model_name=embed_model or "sentence-transformers/all-mpnet-base-v2")

    def score(self, query: str, content: str) -> float:
        """Cosine similarity between query and content embeddings.

        Returns a float in [0.0, 1.0] as required by the RelevanceScorer
        protocol. Raw cosine similarity on normalized embeddings is in
        [-1, 1]; negative values indicate anti-correlation which maps to
        zero relevance. The clamp keeps the orchestrator MemoryState
        invariant (relevance in [0, 1]) satisfied.
        """
        q_emb = self.embed(query)
        c_emb = self.embed(content)
        cos = float(np.dot(q_emb, c_emb))
        if cos < 0.0:
            return 0.0
        if cos > 1.0:
            return 1.0
        return cos


class APIEmbeddingScorer(EmbeddingRelevanceScorer):
    """Embedding scorer that calls a remote OpenAI-compatible embedding API.

    Drops the local SentenceTransformer — all embedding is done via HTTP.
    Compatible with vLLM ``--task embedding`` and OpenAI's embedding API.
    """

    def __init__(
        self,
        api_url: str,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        api_key: str = "",
    ):
        # Intentionally skip parent __init__ (no local model, no disk cache)
        self._model_name = model_name
        self._api_url = api_url
        self._api_key = api_key or "dummy"
        self._model = None  # not used — API-based
        self._tokenizer = None
        self._cache: dict[str, np.ndarray] = {}
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI
        self._client = OpenAI(base_url=self._api_url, api_key=self._api_key)

    def _ensure_model(self):
        pass  # no local model

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
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
            self._ensure_client()
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start : start + batch_size]
                resp = self._client.embeddings.create(
                    model=self._model_name, input=batch,
                )
                for j, emb_data in enumerate(resp.data):
                    vec = np.array(emb_data.embedding, dtype=np.float64)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-12:
                        vec = vec / norm
                    idx = uncached_indices[start + j]
                    results[idx] = vec
                    self._cache[uncached_texts[start + j]] = vec

        return results

    def _load_disk_cache(self):
        pass  # API scorer does not use disk cache (different model = different dims)

    def __getstate__(self):
        """Make picklable by excluding the OpenAI client (has thread locks)."""
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    @property
    def _raw_model(self):
        raise NotImplementedError("APIEmbeddingScorer has no local model")
