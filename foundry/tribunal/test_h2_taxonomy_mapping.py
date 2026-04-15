"""Test 5: Taxonomy mapping between the two classification systems.

Establishes whether the regex classifier and LLM router are measuring the same
construct by comparing their category assignments on the overlapping taxonomy.

Partially offline: regex runs locally, LLM router would need an endpoint.
The test documents methodology and runs what is available offline.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
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
def regex_classifications():
    """Run regex classifier on all 300 questions."""
    with open(QUESTION_TYPES_PATH) as f:
        questions = json.load(f)

    results = []
    for q in questions:
        m = _LME_RE.search(q["question"])
        bare = m.group(1).strip() if m else q["question"].strip()
        regex_type = classify_question_type(bare)
        results.append({
            "question_id": q["question_id"],
            "bare": bare,
            "lme_type": q["question_type"],
            "regex_type": regex_type,
        })
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegexCoverage:
    """Measure effective classification rate of the regex classifier."""

    def test_non_default_fraction(self, regex_classifications):
        """What fraction of questions get a non-default (non-factual) classification?"""
        non_default = sum(1 for r in regex_classifications if r["regex_type"] != "factual")
        total = len(regex_classifications)
        pct = non_default / total * 100

        print(f"\nRegex non-default classifications: {non_default}/{total} ({pct:.1f}%)")
        # The effective classification rate should be meaningful (>30%)
        assert pct > 30, f"Only {pct:.1f}% non-default -- classifier too conservative"

    def test_distribution(self, regex_classifications):
        """Print full distribution."""
        dist = Counter(r["regex_type"] for r in regex_classifications)
        print("\n=== Regex Classifier Distribution ===")
        for cat, count in dist.most_common():
            print(f"  {cat}: {count} ({count / len(regex_classifications) * 100:.1f}%)")


class TestTaxonomyComparison:
    """Compare the three classification systems at the taxonomy level."""

    def test_shared_categories(self):
        """Document which categories are shared between systems."""
        regex_cats = {"factual", "temporal", "ordering", "aggregation", "knowledge-update"}
        router_cats = {"fact_lookup", "assistant_recall", "temporal", "aggregation",
                       "preference", "unanswerable"}
        lme_cats = {"temporal-reasoning", "multi-session", "knowledge-update",
                    "single-session-user", "single-session-assistant",
                    "single-session-preference"}

        shared_regex_router = regex_cats & router_cats
        print(f"\nShared (regex, router): {shared_regex_router}")
        # Only temporal and aggregation are shared
        assert shared_regex_router == {"temporal", "aggregation"}

        print(f"Regex-only:  {regex_cats - router_cats}")
        print(f"Router-only: {router_cats - regex_cats}")
        print(f"LME types:   {lme_cats}")

    def test_category_count_comparison(self):
        """Compare the number of categories in each system."""
        # More categories = finer routing but more room for misclassification
        regex_count = 5   # factual, temporal, ordering, aggregation, knowledge-update
        router_count = 6  # fact_lookup, assistant_recall, temporal, aggregation, preference, unanswerable
        lme_count = 6     # temporal-reasoning, multi-session, knowledge-update, etc.

        print(f"\nCategory counts: regex={regex_count}, router={router_count}, LME={lme_count}")
        # All three have similar granularity

    def test_default_fallback_analysis(self, regex_classifications):
        """Analyze the 'factual' fallback -- what question types land there?"""
        factual_qs = [r for r in regex_classifications if r["regex_type"] == "factual"]
        factual_lme_dist = Counter(r["lme_type"] for r in factual_qs)

        print(f"\n=== Questions Landing in 'factual' Fallback ({len(factual_qs)}) ===")
        for lme_type, count in factual_lme_dist.most_common():
            print(f"  {lme_type}: {count}")

        # Key insight: if most factual-fallback questions are truly factual
        # (single-session-*), then the fallback is appropriate.
        truly_factual = sum(
            count for lme_type, count in factual_lme_dist.items()
            if lme_type.startswith("single-session")
        )
        pct = truly_factual / len(factual_qs) * 100
        print(f"\nTruly factual (single-session-*): {truly_factual}/{len(factual_qs)} ({pct:.1f}%)")


class TestRouterVsRegexDesignDifference:
    """Document the fundamental design difference between the two systems."""

    def test_regex_is_prompt_selector(self):
        """The regex classifier selects answer-generation prompts.

        It does NOT control retrieval. All retrieval channels run regardless.
        The only effect of classification is which system prompt is used
        when generating the answer from retrieved context.
        """
        from mabench.outgestion import DYNAMIC_PROMPTS
        assert len(DYNAMIC_PROMPTS) == 5
        # All prompts produce answers from the same retrieved context
        for ptype, prompt in DYNAMIC_PROMPTS.items():
            assert "knowledge pool" in prompt.lower() or "answer" in prompt.lower(), \
                f"Prompt for {ptype} should reference knowledge pool or answering"

    def test_router_is_retrieval_controller(self):
        """The LLM router was designed to control retrieval channels.

        Its route prompts reference different context types:
        - fact_lookup: 'information from memory'
        - assistant_recall: 'conversation history' with roles
        - temporal: 'events' with dates
        - aggregation: 'structured entity data'
        - preference: 'relevant history'

        These prompts assume different retrieval outputs, implying the
        router was meant to select which retrieval channel to use.
        But in practice, the router is NOT wired into the current best config.
        """
        from mabench.memory_router import ROUTE_PROMPTS
        assert len(ROUTE_PROMPTS) == 5

        # The route prompts reference different context types
        assert "events" in ROUTE_PROMPTS["temporal"].lower()
        assert "entity" in ROUTE_PROMPTS["aggregation"].lower()
        assert "conversation history" in ROUTE_PROMPTS["assistant_recall"].lower()

    def test_agentic_bypasses_both(self):
        """The agentic retriever uses neither classifier.

        It makes its own routing decisions via LLM reasoning at each step.
        This is the most expensive approach but also the most flexible.
        """
        from mabench.agentic_retrieval import AgenticRetriever
        # AgenticRetriever has no classify_question_type or route_question call
        import inspect
        source = inspect.getsource(AgenticRetriever)
        assert "classify_question" not in source
        assert "route_question" not in source
