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

    # ------------------------------------------------------------------
    # Entity-position residual extraction (arXiv:2603.19664)
    # ------------------------------------------------------------------

    _ENTITY_LAYER = 14  # Mid-network layer — best Cohen's d in discrimination test

    def embed_entity_position(
        self, text: str, entity: str, *, layer: int | None = None,
    ) -> np.ndarray:
        """Extract hidden state at entity mention position.

        Uses token-ID subsequence matching to find the entity span in
        the tokenized text, then averages hidden states over that span
        at the specified transformer layer.  Falls back to first-token
        residual if the entity mention is not found.

        Returns a normalized 1024-dim vector.
        """
        self._ensure_model()
        import torch

        target_layer = layer if layer is not None else self._ENTITY_LAYER

        inputs = self._tokenizer(
            [text], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

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

        # Pre-compute entity spans
        entity_spans = [self._find_entity_tokens(t, e) for t, e in zip(texts, entities)]

        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

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

    def score(self, query: str, content: str) -> float:
        """Cosine similarity between query and content embeddings."""
        q_emb = self.embed(query)
        c_emb = self.embed(content)
        return float(np.dot(q_emb, c_emb))
