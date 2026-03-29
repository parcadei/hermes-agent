"""Live test: temporal grounding against Qwen3-30B on RunPod.

Tests with adversarial conversation chunks to verify the LLM
actually produces valid ISO 8601 dates in the "when" field.

Usage:
    # From RunPod (vLLM running locally):
    python3 -u test_temporal_live.py

    # From laptop (SSH tunnel):
    python3 -u test_temporal_live.py --url http://localhost:8000/v1

    # Or direct RunPod:
    python3 -u test_temporal_live.py --url http://69.30.85.138:8000/v1
"""
import argparse
import json
import sys

sys.path.insert(0, ".")
from mabench.llm_triple_extractor import LLMTripleExtractor


REFERENCE_DATE = "2026-03-29"

# Each chunk: (description, text, expected_temporal_count)
# expected_temporal_count = how many triples SHOULD have a date
CHUNKS = [
    (
        "Mixed temporal + timeless (aliases + dates)",
        "My mom Sarah is visiting next week and she's bringing her famous "
        "lasagna. My brother Jake might come too. He's a software engineer.",
        3,  # visiting + bringing + might come are temporal; relationship + job are timeless
    ),
    (
        "Relative date (3 weeks ago)",
        "We adopted Bobo from the shelter about 3 weeks ago. "
        "He's a golden retriever mix and he absolutely loves the backyard.",
        1,  # adopted is temporal; breed + loves are timeless
    ),
    (
        "Multiple absolute dates",
        "I graduated from MIT in June 2019 and started working at Google "
        "last March. I'm planning to move to Austin in June.",
        3,  # all three are temporal events
    ),
    (
        "All timeless (preferences, possessions)",
        "I really prefer Python over JavaScript for backend work, and "
        "I've been using VS Code as my primary editor. I also have a "
        "75-gallon reef tank in my living room with 2 clownfish.",
        0,  # no temporal events
    ),
    (
        "Edge case: fuzzy temporal expression",
        "I visited Paris around Christmas and it was magical. "
        "We spent New Year's Eve watching fireworks from the Eiffel Tower.",
        4,  # all facts share temporal context: Christmas visit + NYE fireworks + location
    ),
    (
        "Knowledge update: job change",
        "Sarah just switched jobs — she used to work at Google but "
        "she started at OpenAI last month.",
        1,  # "started at OpenAI" is temporal; "used to work at" has no specific date
    ),
    (
        "Yesterday / today / tomorrow",
        "I had a dentist appointment yesterday and I need to pick up "
        "my prescription tomorrow. Today I'm just resting at home.",
        3,  # yesterday + tomorrow + today
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1")
    args = parser.parse_args()

    print(f"Connecting to vLLM at {args.url}")
    print(f"Reference date: {REFERENCE_DATE}")
    print("=" * 70)

    ext = LLMTripleExtractor(base_url=args.url, enable_grounding=False)

    total_triples = 0
    total_temporal = 0
    total_timeless = 0

    for desc, text, expected_temporal in CHUNKS:
        print(f"\n--- {desc} ---")
        print(f"Text: {text[:80]}...")

        triples = ext.extract(
            text, mode="conversation", reference_date=REFERENCE_DATE,
        )
        event_times = ext.last_event_times[0] if ext.last_event_times else []

        n_temporal = sum(1 for et in event_times if et is not None)
        n_timeless = len(event_times) - n_temporal

        print(f"Triples: {len(triples)}, Temporal: {n_temporal}, "
              f"Timeless: {n_timeless} (expected temporal: {expected_temporal})")

        for i, (triple, et) in enumerate(zip(triples, event_times)):
            s, p, o = triple
            et_str = et.isoformat() if et else "—"
            marker = "T" if et else " "
            print(f"  [{marker}] ({s}, {p}, {o})  when={et_str}")

        total_triples += len(triples)
        total_temporal += n_temporal
        total_timeless += n_timeless

    print("\n" + "=" * 70)
    print(f"TOTALS: {total_triples} triples, "
          f"{total_temporal} temporal, {total_timeless} timeless")
    print(f"Registry: {json.dumps(ext.entity_registry, indent=2)}")
    print(f"Stats: {ext.stats}")


if __name__ == "__main__":
    main()
