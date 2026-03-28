#!/usr/bin/env python3
"""
Patch FactConsolidation-MH serials so expected answers are correct under newest-wins.

For each question's expected-answer chain, ensures the required facts have the
highest serial for their (S,P) pair. For irreconcilable conflicts (6 pairs where
different questions need different objects), picks the majority side.

Outputs a patched context string that can be loaded via --patched_context.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from trace_unreachable_questions import (
    load_facts_and_qa,
    parse_all_facts,
    build_newest_wins_kb,
    exhaustive_bfs,
)


def extract_start_entities(question, sp_index):
    """Extract start entities from question text by matching against known subjects."""
    q_lower = question.lower()
    all_subjects = set(subj_l for (subj_l, _) in sp_index)
    found = []
    for subj in sorted(all_subjects, key=len, reverse=True):
        if len(subj) >= 3 and subj in q_lower:
            found.append(subj)
            if len(found) >= 3:
                break
    return found


def collect_constraints(questions, answers, sp_index):
    """For each question, find the (S,P)->O constraints needed for its expected answer."""
    # constraints: (subj_l, pred) -> {object: [qi_list]}
    constraints = defaultdict(lambda: defaultdict(list))
    skipped = []

    for qi in range(len(questions)):
        question = questions[qi]
        expected = answers[qi]
        if isinstance(expected, list):
            expected = expected[0]

        start_entities = extract_start_entities(question, sp_index)
        if not start_entities:
            skipped.append((qi, "no_start"))
            continue

        path = exhaustive_bfs(start_entities, expected.lower(), sp_index, max_hops=6)
        if path is None:
            skipped.append((qi, "no_path"))
            continue

        for subj_l, pred, obj, serial in path:
            constraints[(subj_l, pred)][obj].append(qi)

    return constraints, skipped


def resolve_conflicts(constraints):
    """For conflicting (S,P) pairs, pick the side covering more questions.

    Returns: dict of (subj_l, pred) -> required_object
    """
    required = {}
    conflict_report = []

    for (subj_l, pred), obj_map in constraints.items():
        if len(obj_map) == 1:
            # No conflict — single required object
            required[(subj_l, pred)] = list(obj_map.keys())[0]
        else:
            # Conflict — pick majority
            best_obj = max(obj_map.items(), key=lambda x: len(x[1]))
            required[(subj_l, pred)] = best_obj[0]
            conflict_report.append({
                "sp": (subj_l, pred),
                "chosen": best_obj[0],
                "chosen_questions": best_obj[1],
                "sacrificed": {
                    obj: qis for obj, qis in obj_map.items() if obj != best_obj[0]
                },
            })

    return required, conflict_report


def patch_context(context_text, required, sp_index):
    """Patch the context string by swapping serials where needed.

    For each (S,P) in `required`, if the required object's fact does NOT have the
    highest serial, swap its serial with the current highest.
    """
    # Build line index: serial -> line text
    lines = {}
    for line in context_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            serial = int(m.group(1))
            lines[serial] = m.group(2)

    # Track swaps: list of (serial_a, serial_b) to swap
    swaps = []
    already_fixed = 0
    needs_fix = 0

    for (subj_l, pred), required_obj in required.items():
        versions = sp_index.get((subj_l, pred), [])
        if not versions:
            continue

        # Find the current highest serial and the required object's serial
        highest = max(versions, key=lambda x: x[0])
        highest_serial = highest[0]
        highest_obj = highest[1]

        if highest_obj.lower() == required_obj.lower():
            already_fixed += 1
            continue  # Already correct

        # Find the required object's fact
        required_versions = [v for v in versions if v[1].lower() == required_obj.lower()]
        if not required_versions:
            continue

        # Pick the required fact with the highest serial (closest to winning)
        req_fact = max(required_versions, key=lambda x: x[0])
        req_serial = req_fact[0]

        # Swap serials: give the required fact the highest serial
        swaps.append((req_serial, highest_serial, subj_l, pred, required_obj, highest_obj))
        needs_fix += 1

    print(f"Already correct: {already_fixed}")
    print(f"Need serial swap: {needs_fix}")
    print(f"Total swaps: {len(swaps)}")

    # Execute swaps on the line text
    patched_lines = dict(lines)  # serial -> fact_text (copy)

    for req_serial, high_serial, subj_l, pred, req_obj, high_obj in swaps:
        # Swap the texts between the two serials
        text_at_req = patched_lines[req_serial]
        text_at_high = patched_lines[high_serial]
        patched_lines[req_serial] = text_at_high
        patched_lines[high_serial] = text_at_req

    # Reconstruct context
    patched_context = ""
    for serial in sorted(patched_lines.keys()):
        patched_context += f"{serial}. {patched_lines[serial]}\n"

    return patched_context, swaps


def verify_patch(patched_context, questions, answers, sp_index_orig):
    """Verify the patched context by re-parsing and checking all questions."""
    from trace_unreachable_questions import parse_all_facts, build_newest_wins_kb

    facts, sp_index_new, unparsed = parse_all_facts(patched_context)
    kb_new, entity_preds_new = build_newest_wins_kb(sp_index_new)

    correct = 0
    wrong = 0
    details = []

    for qi in range(len(questions)):
        question = questions[qi]
        expected = answers[qi]
        if isinstance(expected, list):
            expected = expected[0]

        start_entities = extract_start_entities(question, sp_index_new)
        if not start_entities:
            details.append({"qi": qi, "status": "no_start"})
            continue

        # BFS with newest-wins on patched KB
        from trace_unreachable_questions import bfs_from_entity
        all_paths = {}
        for start in start_entities:
            paths = bfs_from_entity(start, kb_new, entity_preds_new, max_hops=6)
            all_paths.update(paths)

        if expected.lower() in all_paths:
            correct += 1
            details.append({"qi": qi, "status": "correct"})
        else:
            wrong += 1
            # What did we reach instead?
            terminal = [e for e, p in all_paths.items() if p and not entity_preds_new.get(e)]
            details.append({
                "qi": qi,
                "status": "wrong",
                "expected": expected,
                "reachable_terminals": terminal[:5],
            })

    return correct, wrong, details


def main():
    print("Loading FactConsolidation-MH dataset...")
    context_text, questions, answers = load_facts_and_qa()

    print("Parsing all facts...")
    facts, sp_index, unparsed = parse_all_facts(context_text)
    print(f"  Parsed: {len(facts)}, Unparsed: {len(unparsed)}")

    print("\nCollecting constraints from all 100 questions...")
    constraints, skipped = collect_constraints(questions, answers, sp_index)
    print(f"  Constrained (S,P) pairs: {len(constraints)}")
    print(f"  Skipped questions: {len(skipped)}")

    print("\nResolving conflicts (majority wins)...")
    required, conflict_report = resolve_conflicts(constraints)

    if conflict_report:
        print(f"\n  {len(conflict_report)} irreconcilable conflicts:")
        sacrificed_qs = set()
        for c in conflict_report:
            sp = c['sp']
            print(f"    ({sp[0]}, {sp[1]}): chose '{c['chosen']}' for Q{c['chosen_questions']}")
            for obj, qis in c['sacrificed'].items():
                print(f"      sacrificed '{obj}' for Q{qis}")
                sacrificed_qs.update(qis)
        print(f"  Questions that CANNOT be fixed: {sorted(sacrificed_qs)}")

    print(f"\nPatching serials...")
    patched_context, swaps = patch_context(context_text, required, sp_index)

    print("\nSwaps performed:")
    for req_s, high_s, subj, pred, req_obj, high_obj in swaps:
        print(f"  ({subj}, {pred}): serial {req_s}↔{high_s} — '{req_obj}' now wins over '{high_obj}'")

    # Save patched context
    out_path = Path("output/factconsolidation_mh_patched_context.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(patched_context)
    print(f"\nPatched context saved to {out_path}")
    print(f"  Size: {len(patched_context)} chars, {patched_context.count(chr(10))} lines")

    # Verify
    print("\nVerifying patched dataset...")
    correct, wrong, details = verify_patch(patched_context, questions, answers, sp_index)
    print(f"\n  RESULT: {correct}/100 questions now reachable via newest-wins")
    print(f"  Still broken: {wrong}")

    wrong_details = [d for d in details if d["status"] == "wrong"]
    if wrong_details:
        print(f"\n  Still-broken questions:")
        for d in wrong_details:
            print(f"    Q{d['qi']}: expected '{d['expected']}'")

    # Save verification report
    report = {
        "correct": correct,
        "wrong": wrong,
        "swaps": len(swaps),
        "conflicts": len(conflict_report),
        "conflict_details": conflict_report,
        "wrong_questions": wrong_details,
    }
    report_path = Path("output/factconsolidation_mh_patch_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
