"""Test 3: Agentic vs deterministic head-to-head by question type.

Analyzes existing agentic retrieval traces (60 questions, context-0) to determine
WHERE the LLM-based agent adds value over deterministic retrieval.

OFFLINE test -- uses pre-computed trace files, no new eval runs needed.
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

ROOT = Path(__file__).resolve().parents[2]
TRACES_FULL = ROOT / "proofs/hermes-NLCDM/python/output/agentic_retrieval_full_infra_60q.json"
TRACES_BM25 = ROOT / "proofs/hermes-NLCDM/python/output/agentic_retrieval_bm25only_60q.json"
QUESTION_TYPES_PATH = ROOT / "output" / "lme_question_types.json"

_LME_RE = re.compile(
    r"Now Answer the Question:\s*(.+?)(?:\s*\n\s*Answer:|$)", re.DOTALL
)


@pytest.fixture(scope="module")
def full_infra_traces():
    with open(TRACES_FULL) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def bm25_traces():
    if not TRACES_BM25.exists():
        pytest.skip("BM25-only traces not available")
    with open(TRACES_BM25) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ctx0_questions():
    """First 60 questions (context-0) with LME types."""
    with open(QUESTION_TYPES_PATH) as f:
        all_q = json.load(f)
    return all_q[:60]


@pytest.fixture(scope="module")
def ctx0_regex_types(ctx0_questions):
    """Regex classification of context-0 questions."""
    types = []
    for q in ctx0_questions:
        m = _LME_RE.search(q["question"])
        bare = m.group(1).strip() if m else q["question"].strip()
        types.append(classify_question_type(bare))
    return types


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgenticSubEMByType:
    """Break down agentic retrieval SubEM by LME question type."""

    def test_subem_by_lme_type(self, full_infra_traces, ctx0_questions):
        results = full_infra_traces["results"]
        assert len(results) == 60
        assert len(ctx0_questions) == 60

        by_type = defaultdict(lambda: {"hits": 0, "total": 0})
        for i, result in enumerate(results):
            lme_type = ctx0_questions[i]["question_type"]
            by_type[lme_type]["total"] += 1
            if result.get("match", False):
                by_type[lme_type]["hits"] += 1

        print("\n=== Agentic SubEM by LME Question Type ===")
        print(f"{'Type':30s} {'SubEM':>8s} {'Hits':>6s} {'Total':>6s}")
        for qtype in sorted(by_type.keys()):
            d = by_type[qtype]
            subem = d["hits"] / d["total"] * 100 if d["total"] > 0 else 0
            print(f"{qtype:30s} {subem:7.1f}% {d['hits']:>6d} {d['total']:>6d}")

    def test_subem_by_regex_type(self, full_infra_traces, ctx0_regex_types):
        """Break down by what the regex classifier thinks the type is."""
        results = full_infra_traces["results"]
        by_type = defaultdict(lambda: {"hits": 0, "total": 0})

        for i, result in enumerate(results):
            regex_type = ctx0_regex_types[i]
            by_type[regex_type]["total"] += 1
            if result.get("match", False):
                by_type[regex_type]["hits"] += 1

        print("\n=== Agentic SubEM by Regex Classification ===")
        print(f"{'Regex Type':20s} {'SubEM':>8s} {'Hits':>6s} {'Total':>6s}")
        for rtype in sorted(by_type.keys()):
            d = by_type[rtype]
            subem = d["hits"] / d["total"] * 100 if d["total"] > 0 else 0
            print(f"{rtype:20s} {subem:7.1f}% {d['hits']:>6d} {d['total']:>6d}")


class TestToolUsagePatterns:
    """Analyze which tools the agent uses and whether tool selection is predictable."""

    def test_dominant_pattern(self, full_infra_traces):
        """The dominant pattern should be search_memory -> read_turn -> answer."""
        results = full_infra_traces["results"]
        patterns = Counter()
        for r in results:
            tools = tuple(s["tool"] for s in r["trace"]["steps"])
            patterns[tools] += 1

        print("\n=== Tool Patterns (top 10) ===")
        for pat, count in patterns.most_common(10):
            print(f"  {count}x: {' -> '.join(pat)}")

        # The top pattern should be the simple 3-step
        top_pattern = patterns.most_common(1)[0]
        assert "search_memory" in top_pattern[0], "Expected search_memory in top pattern"

    def test_tool_usage_by_question_type(self, full_infra_traces, ctx0_questions):
        """Which tools does the agent use for each question type?"""
        results = full_infra_traces["results"]
        tool_by_type = defaultdict(Counter)

        for i, result in enumerate(results):
            lme_type = ctx0_questions[i]["question_type"]
            for step in result["trace"]["steps"]:
                tool_by_type[lme_type][step["tool"]] += 1

        print("\n=== Tool Usage by Question Type ===")
        for qtype in sorted(tool_by_type.keys()):
            tools = tool_by_type[qtype]
            total = sum(tools.values())
            print(f"\n  {qtype} ({total} total tool calls):")
            for tool, count in tools.most_common():
                pct = count / total * 100
                print(f"    {tool}: {count} ({pct:.0f}%)")

    def test_execute_python_concentration(self, full_infra_traces, ctx0_questions):
        """execute_python should be concentrated on temporal questions."""
        results = full_infra_traces["results"]
        python_by_type = defaultdict(int)
        total_by_type = defaultdict(int)

        for i, result in enumerate(results):
            lme_type = ctx0_questions[i]["question_type"]
            total_by_type[lme_type] += 1
            for step in result["trace"]["steps"]:
                if step["tool"] == "execute_python":
                    python_by_type[lme_type] += 1
                    break  # Count once per question

        print("\n=== Questions Using execute_python ===")
        for qtype in sorted(total_by_type.keys()):
            used = python_by_type.get(qtype, 0)
            total = total_by_type[qtype]
            pct = used / total * 100 if total > 0 else 0
            print(f"  {qtype}: {used}/{total} ({pct:.0f}%)")


class TestDeterministicReplicability:
    """Measure what fraction of agentic behavior could be replicated deterministically."""

    def test_simple_pattern_fraction(self, full_infra_traces):
        """What fraction of questions follow the simple deterministic pattern?

        The 'simple deterministic' pattern is: search_memory -> (read_turn)* -> answer.
        These questions do NOT benefit from LLM tool selection -- they follow the
        exact sequence a deterministic pipeline would use.
        """
        results = full_infra_traces["results"]
        simple = 0
        complex_patterns = 0

        for r in results:
            tools = [s["tool"] for s in r["trace"]["steps"]]
            # Simple: only search_memory, read_turn, answer
            simple_tools = {"search_memory", "read_turn", "answer"}
            if all(t in simple_tools for t in tools):
                simple += 1
            else:
                complex_patterns += 1

        total = len(results)
        print(f"\nSimple (deterministic-replicable): {simple}/{total} ({simple / total * 100:.0f}%)")
        print(f"Complex (needs LLM reasoning):     {complex_patterns}/{total} ({complex_patterns / total * 100:.0f}%)")

        # The simple fraction represents questions where the agentic retriever
        # adds ZERO value over a deterministic pipeline.

    def test_complex_pattern_hit_rate(self, full_infra_traces):
        """What is the SubEM hit rate for complex vs simple patterns?

        If complex patterns have higher hit rate, the LLM adds value.
        If they have similar or lower hit rate, the LLM does not help.
        """
        results = full_infra_traces["results"]
        simple_hits = 0
        simple_total = 0
        complex_hits = 0
        complex_total = 0

        for r in results:
            tools = [s["tool"] for s in r["trace"]["steps"]]
            simple_tools = {"search_memory", "read_turn", "answer"}
            is_simple = all(t in simple_tools for t in tools)
            hit = r.get("match", False)

            if is_simple:
                simple_total += 1
                if hit:
                    simple_hits += 1
            else:
                complex_total += 1
                if hit:
                    complex_hits += 1

        simple_subem = simple_hits / simple_total * 100 if simple_total > 0 else 0
        complex_subem = complex_hits / complex_total * 100 if complex_total > 0 else 0

        print(f"\nSimple pattern SubEM:  {simple_hits}/{simple_total} ({simple_subem:.1f}%)")
        print(f"Complex pattern SubEM: {complex_hits}/{complex_total} ({complex_subem:.1f}%)")
        print(f"Delta: {complex_subem - simple_subem:+.1f}pp")
