"""Test 6: Temporal misclassification deep dive.

Categorizes the 29 temporal false negatives by failure mode and measures
how many are fixable by regex improvement vs requiring genuine NLU.

OFFLINE test -- no LLM endpoint required.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proofs/hermes-NLCDM/python"))
from mabench.outgestion import classify_question_type

_LME_RE = re.compile(
    r"Now Answer the Question:\s*(.+?)(?:\s*\n\s*Answer:|$)", re.DOTALL
)

ROOT = Path(__file__).resolve().parents[2]
QUESTION_TYPES_PATH = ROOT / "output" / "lme_question_types.json"


@pytest.fixture(scope="module")
def temporal_false_negatives():
    """Temporal-reasoning questions NOT classified as temporal/ordering."""
    with open(QUESTION_TYPES_PATH) as f:
        questions = json.load(f)

    false_neg = []
    for q in questions:
        m = _LME_RE.search(q["question"])
        bare = m.group(1).strip() if m else q["question"].strip()
        regex_type = classify_question_type(bare)
        lme_type = q["question_type"]

        if lme_type == "temporal-reasoning" and regex_type not in ("temporal", "ordering"):
            false_neg.append({
                "bare": bare,
                "regex_type": regex_type,
                "question_id": q["question_id"],
            })
    return false_neg


@pytest.fixture(scope="module")
def temporal_false_positives():
    """Non-temporal questions classified as temporal/ordering."""
    with open(QUESTION_TYPES_PATH) as f:
        questions = json.load(f)

    false_pos = []
    for q in questions:
        m = _LME_RE.search(q["question"])
        bare = m.group(1).strip() if m else q["question"].strip()
        regex_type = classify_question_type(bare)
        lme_type = q["question_type"]

        if lme_type != "temporal-reasoning" and regex_type in ("temporal", "ordering"):
            false_pos.append({
                "bare": bare,
                "lme_type": lme_type,
                "regex_type": regex_type,
                "question_id": q["question_id"],
            })
    return false_pos


# ---------------------------------------------------------------------------
# Failure mode classification
# ---------------------------------------------------------------------------

# Regex pattern for "how many days/weeks/months/years" -- the priority bug pattern.
# These contain temporal unit words but get classified as aggregation because
# "how many" fires before "days/weeks" is checked.
_PRIORITY_BUG_RE = re.compile(
    r"how\s+many\s+(?:day|week|month|year)s?\b", re.IGNORECASE
)

# Pattern for ordering questions not caught by the current regex.
# "What is the order of" is a common phrasing that _ORDERING_RE misses.
_MISSING_ORDER_RE = re.compile(
    r"what\s+is\s+the\s+order\s+of"
    r"|order\s+of\s+the\s+(?:three|four|five|six)"
    r"|from\s+(?:earliest|latest|first|last)\s+to",
    re.IGNORECASE,
)


def _categorize_failure(bare: str, regex_type: str) -> str:
    """Categorize a temporal false negative by its failure mode."""
    q_lower = bare.lower()

    # Priority bug: "how many days/weeks" -> aggregation
    if regex_type == "aggregation" and _PRIORITY_BUG_RE.search(q_lower):
        return "priority_bug"

    # Missing ordering pattern
    if _MISSING_ORDER_RE.search(q_lower):
        return "missing_order_pattern"

    # Contains "first" or "last" but not caught
    if "first" in q_lower or "last" in q_lower:
        if regex_type == "factual":
            return "implicit_ordering"

    # Has a day-of-week or named day reference
    if re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", q_lower):
        return "day_of_week_reference"

    # Has a month name but classified as aggregation/factual
    if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", q_lower):
        if regex_type != "temporal":
            return "month_name_overshadowed"

    # Genuinely implicit temporal -- no obvious keyword
    return "no_temporal_keyword"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTemporalFalseNegatives:
    """Analyze the 29 temporal false negatives."""

    def test_count(self, temporal_false_negatives):
        assert len(temporal_false_negatives) == 29

    def test_failure_mode_categorization(self, temporal_false_negatives):
        """Categorize all 29 false negatives by failure mode."""
        from collections import Counter

        modes = Counter()
        mode_examples = {}
        for fn in temporal_false_negatives:
            mode = _categorize_failure(fn["bare"], fn["regex_type"])
            modes[mode] += 1
            if mode not in mode_examples:
                mode_examples[mode] = fn["bare"][:80]

        print("\n=== Temporal False Negative Failure Modes ===")
        for mode, count in modes.most_common():
            print(f"  {mode}: {count}")
            print(f"    Example: {mode_examples[mode]}")

        # The priority bug and missing pattern are fixable by regex changes.
        # Everything else requires NLU.
        fixable = modes.get("priority_bug", 0) + modes.get("missing_order_pattern", 0)
        total = len(temporal_false_negatives)
        print(f"\nFixable by regex: {fixable}/{total} ({fixable / total * 100:.0f}%)")
        print(f"Requires NLU:     {total - fixable}/{total} ({(total - fixable) / total * 100:.0f}%)")

    def test_priority_bug_is_largest_mode(self, temporal_false_negatives):
        """The priority bug (how many days -> aggregation) should be the main cause."""
        priority_bugs = [
            fn for fn in temporal_false_negatives
            if fn["regex_type"] == "aggregation"
            and _PRIORITY_BUG_RE.search(fn["bare"].lower())
        ]
        print(f"\nPriority bug count: {len(priority_bugs)}")
        for pb in priority_bugs:
            print(f"  {pb['bare'][:80]}")
        # Should be the largest single failure mode
        assert len(priority_bugs) >= 10, f"Expected >= 10 priority bugs, got {len(priority_bugs)}"

    def test_fixable_fraction(self, temporal_false_negatives):
        """At least 50% of temporal false negatives should be regex-fixable."""
        from collections import Counter

        modes = Counter(
            _categorize_failure(fn["bare"], fn["regex_type"])
            for fn in temporal_false_negatives
        )
        fixable = modes.get("priority_bug", 0) + modes.get("missing_order_pattern", 0)
        fraction = fixable / len(temporal_false_negatives)
        assert fraction >= 0.50, f"Only {fraction:.0%} fixable, expected >= 50%"


class TestTemporalFalsePositives:
    """Analyze the 15 temporal false positives."""

    def test_count(self, temporal_false_positives):
        assert len(temporal_false_positives) == 15

    def test_how_long_is_dominant_pattern(self, temporal_false_positives):
        """Most false positives should be 'how long' questions about duration/age."""
        how_long = [
            fp for fp in temporal_false_positives
            if "how long" in fp["bare"].lower() or "how old" in fp["bare"].lower()
        ]
        print(f"\n'How long/old' false positives: {len(how_long)}/{len(temporal_false_positives)}")
        for fp in how_long:
            print(f"  [{fp['lme_type']}] {fp['bare'][:80]}")
        # "How long have I been X" and "How old was I when" are the primary patterns
        assert len(how_long) >= 8

    def test_false_positives_are_harmless(self, temporal_false_positives):
        """Document whether false positives cause actual harm.

        A temporal false positive means a non-temporal question gets the temporal
        prompt. The temporal prompt says 'USE date prefixes to compute time
        differences.' If the question is actually about duration ('How long have
        I been collecting cameras?'), the temporal prompt is arguably appropriate
        -- it encourages the LLM to look at dates, which IS how you answer
        duration questions.

        So temporal false positives may actually be HELPFUL, not harmful.
        """
        # Just document -- no assertion about harm
        for fp in temporal_false_positives:
            # Check: does the question benefit from temporal awareness?
            bare_lower = fp["bare"].lower()
            benefits_from_dates = (
                "how long" in bare_lower
                or "how old" in bare_lower
                or "when" in bare_lower
            )
            fp["benefits_from_dates"] = benefits_from_dates

        beneficial = sum(1 for fp in temporal_false_positives if fp["benefits_from_dates"])
        print(f"\nFalse positives that benefit from temporal prompt: "
              f"{beneficial}/{len(temporal_false_positives)}")
