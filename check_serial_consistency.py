#!/usr/bin/env python3
"""
Check if a consistent serial assignment exists for FactConsolidation-MH.

For each question's expected answer to be correct, specific (subject, predicate)
pairs must resolve to specific objects via newest-wins. This script:

1. Parses all 18,332 facts into structured triples
2. For each of 100 questions, traces the expected-answer chain
3. Extracts constraints: (subject, predicate) MUST resolve to object X
4. Checks for conflicts: two questions needing different objects for same (S,P)

If conflicts exist, no serial reassignment can make all expected answers correct.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the parser from trace_unreachable_questions.py
from trace_unreachable_questions import (
    load_facts_and_qa,
    parse_all_facts,
    build_newest_wins_kb,
    exhaustive_bfs,
)


def trace_expected_chain(qi, question, expected_answer, context_text, sp_index):
    """Find the chain of (S,P)->O constraints needed for a question's expected answer.

    Uses exhaustive BFS (all fact versions) to find ANY path to the expected answer.
    Returns list of constraints: [(subj_l, pred, required_object), ...]
    """
    # Extract start entities from the question
    # Questions mention entities in the first phrase
    start_entities = extract_start_entities(question, sp_index)

    if not start_entities:
        return None, "no_start_entities"

    answer_l = expected_answer.lower()
    path = exhaustive_bfs(start_entities, answer_l, sp_index, max_hops=6)

    if path is None:
        return None, "no_path"

    # Each hop in the path is a constraint: (subj_l, pred) MUST resolve to obj
    constraints = []
    for subj_l, pred, obj, serial in path:
        constraints.append((subj_l, pred, obj))

    return constraints, "ok"


def extract_start_entities(question, sp_index):
    """Extract start entities from question text by matching against known subjects."""
    q_lower = question.lower()

    # Collect all known subjects
    all_subjects = set()
    for (subj_l, pred) in sp_index:
        all_subjects.add(subj_l)

    # Find subjects mentioned in the question (longest match first)
    found = []
    subjects_by_len = sorted(all_subjects, key=len, reverse=True)
    for subj in subjects_by_len:
        if len(subj) >= 3 and subj in q_lower:
            found.append(subj)
            if len(found) >= 3:
                break

    return found


def main():
    print("Loading FactConsolidation-MH dataset...")
    context_text, questions, answers = load_facts_and_qa()

    print("Parsing all facts into structured triples...")
    facts, sp_index, unparsed = parse_all_facts(context_text)
    print(f"  Parsed: {len(facts)}, Unparsed: {len(unparsed)}")

    print("Building newest-wins KB...")
    kb, entity_preds = build_newest_wins_kb(sp_index)
    print(f"  Unique (S,P) pairs: {len(kb)}")

    # For each question, find the constraint chain
    print(f"\nTracing expected-answer chains for {len(questions)} questions...")

    # constraints_by_sp: (subj_l, pred) -> {object: [list of qi that need it]}
    constraints_by_sp = defaultdict(lambda: defaultdict(list))

    traced = 0
    no_path = 0
    no_start = 0

    for qi in range(len(questions)):
        question = questions[qi]
        expected = answers[qi]
        if isinstance(expected, list):
            expected = expected[0]

        constraints, status = trace_expected_chain(qi, question, expected, context_text, sp_index)

        if status == "no_start_entities":
            no_start += 1
            continue
        if status == "no_path":
            no_path += 1
            continue

        traced += 1
        for subj_l, pred, obj in constraints:
            constraints_by_sp[(subj_l, pred)][obj].append(qi)

    print(f"  Traced: {traced}, No path: {no_path}, No start: {no_start}")

    # Find conflicts: (S,P) pairs where different questions need different objects
    print(f"\n{'='*70}")
    print("CONFLICT ANALYSIS")
    print(f"{'='*70}")

    conflicts = []
    for (subj_l, pred), obj_map in sorted(constraints_by_sp.items()):
        if len(obj_map) > 1:
            # Multiple objects needed for same (S,P) — CONFLICT
            conflict = {
                "subject": subj_l,
                "predicate": pred,
                "required_objects": {},
                "current_newest": None,
            }
            for obj, qis in obj_map.items():
                conflict["required_objects"][obj] = qis

            # What does the current newest-wins give?
            if (subj_l, pred) in kb:
                serial, obj = kb[(subj_l, pred)]
                conflict["current_newest"] = {"serial": serial, "object": obj}

            # How many versions exist?
            versions = sp_index.get((subj_l, pred), [])
            conflict["num_versions"] = len(versions)
            conflict["all_versions"] = [
                {"serial": s, "object": o} for s, o, _ in sorted(versions, key=lambda x: x[0])
            ]

            conflicts.append(conflict)

    if not conflicts:
        print("\nNO CONFLICTS FOUND!")
        print("A consistent serial assignment EXISTS.")
        print("All expected answers can be made correct simultaneously.")
    else:
        print(f"\n{len(conflicts)} CONFLICTS FOUND!")
        print("No serial assignment can make all expected answers correct.\n")

        total_questions_affected = set()
        for c in conflicts:
            print(f"  ({c['subject']}, {c['predicate']}):")
            print(f"    Current newest: {c['current_newest']}")
            print(f"    {c['num_versions']} versions exist:")
            for v in c['all_versions']:
                print(f"      serial {v['serial']}: {v['object']}")
            print(f"    CONFLICTING requirements:")
            for obj, qis in c['required_objects'].items():
                print(f"      Need '{obj}' for questions: {qis}")
                total_questions_affected.update(qis)
            print()

        print(f"Total questions involved in conflicts: {len(total_questions_affected)}")
        print(f"Conflicting (S,P) pairs: {len(conflicts)}")

        # Save detailed results
        output = {
            "num_conflicts": len(conflicts),
            "questions_affected": sorted(total_questions_affected),
            "conflicts": conflicts,
            "summary": {
                "traced": traced,
                "no_path": no_path,
                "no_start": no_start,
                "total_questions": len(questions),
            }
        }
        out_path = Path("output/factconsolidation_mh_serial_conflicts.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
