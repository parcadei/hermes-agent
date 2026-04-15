"""Test 1: Regex classifier accuracy against LME question_type labels.

Runs outgestion.classify_question_type on all 300 LME questions and computes
functional accuracy using a semantic mapping from LME types to regex categories.

This is an OFFLINE test -- no LLM endpoint required.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proofs/hermes-NLCDM/python"))
from mabench.outgestion import classify_question_type

_LME_RE = re.compile(
    r"Now Answer the Question:\s*(.+?)(?:\s*\n\s*Answer:|$)", re.DOTALL
)

# Functional mapping: LME type -> set of acceptable regex categories.
# "Acceptable" means: the regex category triggers behavior appropriate for
# the question's actual need.
FUNCTIONAL_MAP = {
    "temporal-reasoning": {"temporal", "ordering"},
    "multi-session": {"aggregation"},
    "knowledge-update": {"knowledge-update"},
    "single-session-user": {"factual"},
    "single-session-assistant": {"factual"},
    "single-session-preference": {"factual"},
}

ROOT = Path(__file__).resolve().parents[2]
QUESTION_TYPES_PATH = ROOT / "output" / "lme_question_types.json"


@pytest.fixture(scope="module")
def questions():
    with open(QUESTION_TYPES_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def classified(questions):
    """Run regex classifier on all 300 questions, return list of dicts."""
    results = []
    for q in questions:
        m = _LME_RE.search(q["question"])
        bare = m.group(1).strip() if m else q["question"].strip()
        regex_type = classify_question_type(bare)
        lme_type = q["question_type"]
        acceptable = FUNCTIONAL_MAP.get(lme_type, {"factual"})
        results.append({
            "question_id": q["question_id"],
            "bare": bare,
            "lme_type": lme_type,
            "regex_type": regex_type,
            "is_correct": regex_type in acceptable,
        })
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegexClassifierCompleteness:
    """Verify the classifier runs on all 300 questions without error."""

    def test_300_questions_classified(self, classified):
        assert len(classified) == 300

    def test_all_types_are_valid(self, classified):
        valid = {"factual", "temporal", "ordering", "aggregation", "knowledge-update"}
        for c in classified:
            assert c["regex_type"] in valid, f"Invalid type {c['regex_type']} for {c['bare'][:60]}"


class TestRegexClassifierDistribution:
    """Verify the distribution matches the pre-computed cross-tabulation."""

    def test_factual_is_majority(self, classified):
        factual_count = sum(1 for c in classified if c["regex_type"] == "factual")
        assert factual_count > 100, f"Expected factual > 100, got {factual_count}"

    def test_aggregation_count(self, classified):
        agg_count = sum(1 for c in classified if c["regex_type"] == "aggregation")
        assert 60 <= agg_count <= 80, f"Expected aggregation 60-80, got {agg_count}"

    def test_temporal_count(self, classified):
        temp_count = sum(1 for c in classified if c["regex_type"] == "temporal")
        assert 35 <= temp_count <= 55, f"Expected temporal 35-55, got {temp_count}"


class TestFunctionalAccuracy:
    """Measure functional accuracy using the LME type -> regex mapping."""

    def test_overall_accuracy(self, classified):
        correct = sum(1 for c in classified if c["is_correct"])
        accuracy = correct / len(classified) * 100
        # Record the measurement -- do not assert a minimum threshold.
        # The purpose is measurement, not gating.
        print(f"\nOverall functional accuracy: {correct}/300 ({accuracy:.1f}%)")
        # Expect approximately 60% based on pre-computation
        assert 50 <= accuracy <= 70, f"Accuracy {accuracy:.1f}% outside expected range"

    def test_temporal_recall(self, classified):
        """What fraction of temporal-reasoning questions are correctly classified?"""
        temporal_qs = [c for c in classified if c["lme_type"] == "temporal-reasoning"]
        correct = sum(1 for c in temporal_qs if c["is_correct"])
        recall = correct / len(temporal_qs) * 100
        print(f"\nTemporal recall: {correct}/{len(temporal_qs)} ({recall:.1f}%)")
        # Pre-computed: 61.3%
        assert len(temporal_qs) == 75

    def test_aggregation_recall(self, classified):
        """What fraction of multi-session questions are classified as aggregation?"""
        multi_qs = [c for c in classified if c["lme_type"] == "multi-session"]
        correct = sum(1 for c in multi_qs if c["is_correct"])
        recall = correct / len(multi_qs) * 100
        print(f"\nAggregation recall: {correct}/{len(multi_qs)} ({recall:.1f}%)")
        assert len(multi_qs) == 75

    def test_single_session_recall(self, classified):
        """Single-session questions should overwhelmingly be classified as factual."""
        ss_types = {"single-session-user", "single-session-assistant", "single-session-preference"}
        ss_qs = [c for c in classified if c["lme_type"] in ss_types]
        correct = sum(1 for c in ss_qs if c["is_correct"])
        recall = correct / len(ss_qs) * 100
        print(f"\nSingle-session recall: {correct}/{len(ss_qs)} ({recall:.1f}%)")
        assert len(ss_qs) == 105
        # These should be high -- regex defaults to factual
        assert recall >= 75


class TestMisclassificationBreakdown:
    """Detailed breakdown of misclassified questions by failure mode."""

    def test_cross_tabulation(self, classified):
        """Print cross-tabulation of LME type vs regex type."""
        cross = defaultdict(Counter)
        for c in classified:
            cross[c["lme_type"]][c["regex_type"]] += 1

        print("\n=== Cross-Tabulation ===")
        regex_cats = ["aggregation", "temporal", "ordering", "knowledge-update", "factual"]
        header = f"{'LME Type':30s}" + "".join(f"{r:>15s}" for r in regex_cats)
        print(header)
        for lme_type in sorted(cross.keys()):
            row = f"{lme_type:30s}"
            for rc in regex_cats:
                row += f"{cross[lme_type][rc]:>15d}"
            print(row)

    def test_temporal_false_negatives(self, classified):
        """Count temporal-reasoning questions missed by the regex."""
        temporal_fn = [
            c for c in classified
            if c["lme_type"] == "temporal-reasoning" and not c["is_correct"]
        ]
        print(f"\nTemporal false negatives: {len(temporal_fn)}")
        for fn in temporal_fn:
            print(f"  [{fn['regex_type']}] {fn['bare'][:80]}")
        # Pre-computed: 29
        assert len(temporal_fn) == 29

    def test_temporal_false_positives(self, classified):
        """Count non-temporal questions misclassified as temporal."""
        temporal_fp = [
            c for c in classified
            if c["lme_type"] != "temporal-reasoning"
            and c["regex_type"] in ("temporal", "ordering")
        ]
        print(f"\nTemporal false positives: {len(temporal_fp)}")
        for fp in temporal_fp:
            print(f"  [{fp['lme_type']}] {fp['bare'][:80]}")
        # Pre-computed: 15
        assert len(temporal_fp) == 15
