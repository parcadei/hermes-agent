"""Precompute v9 extraction for all LME chunks.

Uses the v9 cognitive extraction schema (6-step) producing rich relations
with when, when_type, quantity, unit, confidence, sentiment, source_turn.

Usage:
    python -u -m foundry.eval.precompute_v9 \
        --input output/multipass_cache_cs512.json \
        --output output/msl_cache_v9_cs512.json \
        --base_url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Load v9 schema from JSON file
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROMPT_PATH = _SCRIPT_DIR.parent / "msl_extraction_prompt.json"

def _load_v9_schema() -> tuple[str, dict]:
    """Load v9 system prompt and JSON schema from msl_extraction_prompt.json."""
    with open(_PROMPT_PATH) as f:
        spec = json.load(f)
    return spec["system_prompt"], spec["json_schema"]

V9_SYSTEM_PROMPT, V9_JSON_SCHEMA = _load_v9_schema()


def extract_one(
    text: str,
    base_url: str,
    model: str,
    timeout: float,
) -> dict:
    """Extract v9 entities + relations from a single chunk."""
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": V9_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "response_format": {
            "type": "json_schema",
            "json_schema": V9_JSON_SCHEMA,
        },
    }

    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    raw = None
    try:
        raw = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass
    if raw is None:
        depth = 0
        start = None
        for i, ch in enumerate(content):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        raw = json.loads(content[start:i + 1])
                        break
                    except (json.JSONDecodeError, ValueError):
                        start = None
    if raw is None or not isinstance(raw, dict):
        return {"entities": [], "relations": []}

    # Clean entities
    clean_entities = []
    for ent in raw.get("entities", []):
        if not isinstance(ent, dict):
            continue
        name = str(ent.get("name", "")).strip()
        if not name:
            continue
        clean_entities.append({
            "name": name,
            "entity_type": str(ent.get("entity_type", "")).strip(),
        })

    # Clean relations — preserve all v9 fields
    clean_relations = []
    for rel in raw.get("relations", []):
        if not isinstance(rel, dict):
            continue
        subj = str(rel.get("subject", "")).strip()
        pred = str(rel.get("predicate", "")).strip()
        obj = str(rel.get("object", "")).strip()
        if not (subj and pred and obj):
            continue
        clean_relations.append({
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "role": str(rel.get("role", "")).strip(),
            "when": str(rel.get("when", "")).strip(),
            "when_type": str(rel.get("when_type", "")).strip(),
            "quantity": str(rel.get("quantity", "0")).strip(),
            "unit": str(rel.get("unit", "")).strip(),
            "confidence": str(rel.get("confidence", "")).strip(),
            "sentiment": str(rel.get("sentiment", "")).strip(),
            "source_turn": int(rel.get("source_turn", -1)),
        })

    return {"entities": clean_entities, "relations": clean_relations}


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute v9 extraction")
    parser.add_argument("--input", required=True,
                        help="Multipass triples cache JSON (has chunk texts)")
    parser.add_argument("--output", required=True,
                        help="Output v9 cache JSON")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    with open(args.input) as f:
        input_cache = json.load(f)
    logger.info("Loaded %d chunks from %s", len(input_cache), args.input)

    output_cache: dict[str, dict] = {}
    if args.resume:
        try:
            with open(args.output) as f:
                output_cache = json.load(f)
            logger.info("Resuming: %d chunks already extracted", len(output_cache))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    to_extract: list[tuple[str, str]] = []
    for chunk_hash, chunk_data in input_cache.items():
        if chunk_hash in output_cache:
            continue
        text = chunk_data.get("text", "")
        if not text.strip():
            continue
        to_extract.append((chunk_hash, text))

    logger.info(
        "Extracting %d chunks (%d already cached), %d workers",
        len(to_extract), len(output_cache), args.workers,
    )

    if not to_extract:
        logger.info("Nothing to extract. Done.")
        return

    def _do_one(item: tuple[str, str]) -> tuple[str, dict]:
        chunk_hash, text = item
        result = extract_one(text, args.base_url, args.model, args.timeout)
        return chunk_hash, result

    t0 = time.time()
    done = 0
    errors = 0
    total_entities = 0
    total_relations = 0
    when_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_do_one, item): item for item in to_extract}
        for future in as_completed(futures):
            try:
                chunk_hash, result = future.result()
                output_cache[chunk_hash] = result
                total_entities += len(result.get("entities", []))
                rels = result.get("relations", [])
                total_relations += len(rels)
                when_count += sum(1 for r in rels if r.get("when"))
                done += 1
            except Exception as exc:
                item = futures[future]
                logger.warning("Error on chunk %s: %s", item[0][:12], exc)
                errors += 1
                done += 1

            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(to_extract) - done) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.1f/s, ETA %.0fs, %d errors) | "
                    "%d entities, %d relations (%d with when)",
                    done, len(to_extract), rate, eta, errors,
                    total_entities, total_relations, when_count,
                )
                with open(args.output, "w") as f:
                    json.dump(output_cache, f)

    with open(args.output, "w") as f:
        json.dump(output_cache, f)

    elapsed = time.time() - t0
    n_with_ents = sum(1 for v in output_cache.values() if v.get("entities"))
    n_with_rels = sum(1 for v in output_cache.values() if v.get("relations"))
    n_with_when = sum(
        1 for v in output_cache.values()
        if any(r.get("when") for r in v.get("relations", []))
    )
    logger.info(
        "Done: %d chunks in %.1fs (%.1f/s)\n"
        "  %d with entities, %d with relations, %d with temporal, %d errors\n"
        "  Total: %d entities, %d relations, %d with when",
        len(output_cache), elapsed,
        len(to_extract) / elapsed if elapsed > 0 else 0,
        n_with_ents, n_with_rels, n_with_when, errors,
        total_entities, total_relations, when_count,
    )


if __name__ == "__main__":
    main()
