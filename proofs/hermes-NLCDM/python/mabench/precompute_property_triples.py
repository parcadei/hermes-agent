"""Re-extract property-style triples from LME chunks via LM Studio.

Uses the V3 property-style prompt validated at 95% grounding on lfm2.5-1.2b.
Reads chunk texts from the existing multipass cache (or raw dataset) and
produces queryable triples like ["user", "SkyMiles_balance", "10000"] instead
of narrative triples like ["user", "considers", "Chase Sapphire Preferred"].

Usage:
    python3 -u -m mabench.precompute_property_triples \
        --input output/multipass_cache_cs512.json \
        --output output/property_triples_cs512.json \
        --base_url http://127.0.0.1:1234/v1 \
        --model liquid/lfm2.5-1.2b \
        --workers 1

    # Resume after interruption:
    python3 -u -m mabench.precompute_property_triples ... --resume
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# V3 property-style extraction prompt (validated on lfm2.5-1.2b, 95% grounded)
# ---------------------------------------------------------------------------

V3_SYSTEM = """Extract facts from the message as a JSON array of ["subject", "property", "value"] triples.
CRITICAL: Every value MUST be a word or number that appears in the text. Never invent values.
Rules: "I"/"my" → subject is "user". property = short label. Skip questions/opinions.
Examples:
"I work 50 hours a week" → [["user", "work_hours_weekly", "50"]]
"My flight on JetBlue from SF to Boston" → [["user", "flight_airline", "JetBlue"], ["user", "flight_from", "SF"], ["user", "flight_to", "Boston"]]
"I scored 132 in Ticket to Ride" → [["user", "game_score", "132"], ["user", "game", "Ticket to Ride"]]
JSON array:"""


def _call_lm(base_url: str, model: str, text: str, max_tokens: int = 1024) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="unused")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": V3_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _parse_triples(raw: str) -> list[list[str]]:
    """Parse LLM output as JSON array of [s, p, v] triples."""
    raw = raw.strip()
    # Handle markdown code blocks
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON array in the output
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    result = []
    for item in data:
        if isinstance(item, list) and len(item) == 3:
            result.append([str(x) for x in item])
        elif isinstance(item, dict):
            # Handle {"subject": ..., "property": ..., "value": ...} format
            s = item.get("subject", "")
            p = item.get("property", item.get("predicate", ""))
            v = item.get("value", "")
            if s and p and v:
                result.append([str(s), str(p), str(v)])
    return result


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Re-extract property-style triples")
    parser.add_argument("--input", required=True, help="Path to multipass cache JSON")
    parser.add_argument("--output", required=True, help="Output path for property triples")
    parser.add_argument("--base_url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="liquid/lfm2.5-1.2b")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", help="Skip already-extracted entries")
    parser.add_argument("--save_every", type=int, default=100, help="Checkpoint interval")
    parser.add_argument("--limit", type=int, default=0, help="Process only N entries (0=all)")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers (use >1 for vLLM)")
    args = parser.parse_args()

    # Load input chunks
    print(f"Loading input from {args.input}...")
    with open(args.input) as f:
        input_cache = json.load(f)
    print(f"  {len(input_cache)} chunks loaded")

    # Load existing output for resume
    output_path = Path(args.output)
    output: dict[str, dict] = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            output = json.load(f)
        print(f"  Resuming: {len(output)} already done")

    # Process each chunk
    todo = [(h, v) for h, v in input_cache.items() if h not in output]
    if args.limit > 0:
        todo = todo[:args.limit]
    total = len(todo)
    print(f"  Processing {total} chunks via {args.model} (workers={args.workers})...")

    t0 = time.time()
    total_triples = 0
    errors = 0
    done_count = 0
    lock = threading.Lock()

    def _process_one(item):
        hash_key, entry = item
        text = entry.get("text", "")
        if not text.strip():
            return hash_key, text, [], None
        try:
            raw = _call_lm(args.base_url, args.model, text, args.max_tokens)
            triples = _parse_triples(raw)
            return hash_key, text, triples, None
        except Exception as e:
            return hash_key, text, [], e

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, item): item for item in todo}
        for future in concurrent.futures.as_completed(futures):
            hash_key, text, triples, err = future.result()

            with lock:
                if err:
                    print(f"  ERROR on {hash_key}: {err}", file=sys.stderr)
                    errors += 1
                output[hash_key] = {"text": text, "triples": triples}
                total_triples += len(triples)
                done_count += 1

                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / rate if rate > 0 else 0
                if done_count % 10 == 0 or done_count == 1:
                    print(
                        f"  [{done_count}/{total}] {len(triples):2d} triples | "
                        f"{rate:.1f} chunks/s | ETA {eta/60:.1f}m | "
                        f"total_triples={total_triples} errors={errors}"
                    )

                if done_count % args.save_every == 0:
                    with open(output_path, "w") as f:
                        json.dump(output, f)
                    print(f"  ** checkpoint saved ({len(output)} entries)")

    # Final save
    with open(output_path, "w") as f:
        json.dump(output, f)

    elapsed = time.time() - t0
    print(f"\nDone. {len(output)} entries, {total_triples} triples, "
          f"{errors} errors, {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
