"""Embed all 170k HF sentences and save vectors to disk.

Usage:
    .venv/bin/python embed_170k.py

Output:
    embeddings_170k.npy  — (N, 1024) float32 array
    sentences_170k.txt   — one sentence per line (aligned by index)
"""

import gzip
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

OUT_DIR = Path(__file__).resolve().parent

def main():
    # 1. Download dataset
    print("Downloading dataset...")
    path = hf_hub_download(
        repo_id="agentlans/high-quality-english-sentences",
        filename="test.txt.gz",
        repo_type="dataset",
    )

    # 2. Read and filter
    print("Reading sentences...")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        sentences = [
            line.strip() for line in f
            if 20 <= len(line.strip()) <= 500
        ]
    print(f"  {len(sentences)} sentences after filtering")

    # 3. Save sentences
    txt_path = OUT_DIR / "sentences_170k.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"  Saved {txt_path}")

    # 4. Embed in batches
    print("Loading model...")
    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"
    print(f"  Using device: {device}")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True, device=device)

    print(f"Embedding {len(sentences)} sentences...")
    t0 = time.time()
    embeddings = model.encode(
        sentences,
        batch_size=128,
        show_progress_bar=True,
        prompt_name="query",
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/len(sentences)*1000:.1f}ms/sentence)")

    # 5. Save as float32 (half the size of float64)
    emb = np.asarray(embeddings, dtype=np.float32)
    npy_path = OUT_DIR / "embeddings_170k.npy"
    np.save(npy_path, emb)
    size_mb = npy_path.stat().st_size / 1024 / 1024
    print(f"  Saved {npy_path} — shape {emb.shape}, {size_mb:.1f}MB")

if __name__ == "__main__":
    main()
