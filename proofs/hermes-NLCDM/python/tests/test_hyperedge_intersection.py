"""Tests for HyperedgeIntersection module.

Tests the standalone HyperedgeIntersection module that uses the GLiNER entity
index to penalize retrieval scores for facts that don't mention the
intermediate answer entity during decompose hops.

Module location: mabench/hyperedge_intersection.py
Spec: thoughts/shared/plans/knowledge-graph-retrieval/spec.md (Module 1)

These tests are written BEFORE implementation (TDD). They define the target
behavior and should FAIL until the module is implemented.
"""

from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

_TESTS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _TESTS_DIR.parent
sys.path.insert(0, str(_NLCDM_PYTHON))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entity_index(mapping: dict[str, set[int]]) -> dict[str, set[int]]:
    """Build a lowercase entity index from a mapping."""
    return {k.lower(): v for k, v in mapping.items()}


def _make_scores(n: int, value: float = 1.0) -> np.ndarray:
    """Create a uniform scores array of length n."""
    return np.full(n, value, dtype=np.float64)


# ===================================================================
# 1. Module structure
# ===================================================================

class TestModuleStructure:
    """Verify the module exists, is importable, and has the right API."""

    def test_module_importable(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        assert HyperedgeIntersection is not None

    def test_standalone_class(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        # No mixin or framework inheritance
        assert HyperedgeIntersection.__bases__ == (object,)

    def test_init_signature(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        sig = inspect.signature(HyperedgeIntersection.__init__)
        params = list(sig.parameters.keys())
        assert "entity_index" in params
        assert "penalty" in params
        assert "min_candidates" in params

    def test_has_rerank(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        assert callable(getattr(HyperedgeIntersection, "rerank", None))

    def test_has_rerank_with_fuzzy_match(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        assert callable(
            getattr(HyperedgeIntersection, "rerank_with_fuzzy_match", None)
        )

    def test_has_entity_coverage(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        assert callable(
            getattr(HyperedgeIntersection, "entity_coverage", None)
        )

    def test_rerank_signature(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        sig = inspect.signature(HyperedgeIntersection.rerank)
        params = list(sig.parameters.keys())
        assert "intermediate_entity" in params
        assert "scores" in params
        assert "store_size" in params

    def test_rerank_with_fuzzy_match_signature(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        sig = inspect.signature(HyperedgeIntersection.rerank_with_fuzzy_match)
        params = list(sig.parameters.keys())
        assert "intermediate_entity" in params
        assert "scores" in params
        assert "store_size" in params

    def test_entity_coverage_signature(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        sig = inspect.signature(HyperedgeIntersection.entity_coverage)
        params = list(sig.parameters.keys())
        assert "entity" in params

    def test_default_penalty(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        h = HyperedgeIntersection(entity_index={})
        assert h._penalty == pytest.approx(0.1)

    def test_default_min_candidates(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection
        h = HyperedgeIntersection(entity_index={})
        assert h._min_candidates == 3


# ===================================================================
# 2. Exact match reranking
# ===================================================================

class TestExactMatchReranking:
    """Entity in index, scores adjusted correctly."""

    def test_matched_scores_unchanged(self):
        """Facts mentioning the entity keep their original scores."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 2, 4}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        result = h.rerank("cornerback", scores, store_size=5)

        # Indices 0, 2, 4 should be unchanged
        assert result[0] == pytest.approx(0.9)
        assert result[2] == pytest.approx(0.7)
        assert result[4] == pytest.approx(0.5)

    def test_unmatched_scores_penalized(self):
        """Facts NOT mentioning the entity get multiplied by penalty."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 2, 4}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        result = h.rerank("cornerback", scores, store_size=5)

        # Indices 1, 3 should be penalized (multiplied by 0.1)
        assert result[1] == pytest.approx(0.08)
        assert result[3] == pytest.approx(0.06)

    def test_output_shape_matches_input(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        result = h.rerank("cornerback", scores, store_size=6)
        assert result.shape == scores.shape

    def test_output_never_exceeds_input(self):
        """Penalty only reduces, never boosts."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=0.5, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        result = h.rerank("cornerback", scores, store_size=5)
        assert np.all(result <= scores)

    def test_case_insensitive_lookup(self):
        """Entity lookup is case-insensitive."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        result = h.rerank("Cornerback", scores, store_size=5)
        # Should match despite capitalization
        assert result[0] == pytest.approx(0.9)
        assert result[3] == pytest.approx(0.06)

    def test_whitespace_stripping(self):
        """Leading/trailing whitespace is stripped from the entity."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        result = h.rerank("  cornerback  ", scores, store_size=5)
        assert result[0] == pytest.approx(0.9)
        assert result[3] == pytest.approx(0.06)

    def test_rerank_returns_copy(self):
        """rerank returns a new array, does not mutate the input."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        original = scores.copy()
        h.rerank("cornerback", scores, store_size=5)
        assert_array_equal(scores, original)

    def test_rerank_is_pure(self):
        """Calling rerank does not modify the entity_index."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        index_before = copy.deepcopy(entity_index)
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        h.rerank("cornerback", scores, store_size=5)
        assert entity_index == index_before


# ===================================================================
# 3. No match / below min_candidates
# ===================================================================

class TestNoMatchBelowMinCandidates:
    """Returns scores unchanged when matches < min_candidates."""

    def test_entity_not_in_index(self):
        """Entity has zero matches -- scores unchanged."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7])
        result = h.rerank("baseball", scores, store_size=3)
        assert_array_equal(result, scores)

    def test_too_few_matches(self):
        """Entity matches fewer than min_candidates facts."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1}})  # only 2
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        result = h.rerank("cornerback", scores, store_size=4)
        assert_array_equal(result, scores)

    def test_exactly_min_candidates_activates(self):
        """Exactly min_candidates matches should activate the penalty."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        result = h.rerank("cornerback", scores, store_size=4)
        # Index 3 should be penalized (not in matched set)
        assert result[3] == pytest.approx(0.06)

    def test_min_candidates_one(self):
        """min_candidates=1 means even a single match activates penalty."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {2}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=1)
        scores = np.array([0.9, 0.8, 0.7])
        result = h.rerank("cornerback", scores, store_size=3)
        assert result[0] == pytest.approx(0.09)
        assert result[1] == pytest.approx(0.08)
        assert result[2] == pytest.approx(0.7)  # matched, unchanged


# ===================================================================
# 4. Fuzzy matching
# ===================================================================

class TestFuzzyMatching:
    """Substring, case, and word-overlap matching all work."""

    def test_exact_match_in_fuzzy(self):
        """Fuzzy match includes exact match results."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        result_scores, matched = h.rerank_with_fuzzy_match(
            "cornerback", scores, store_size=4
        )
        assert {0, 1, 2} == matched

    def test_substring_entity_in_key(self):
        """Entity 'football' matches key 'association football'."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "association football": {0, 1, 2},
            "baseball": {3, 4, 5},
        })
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

        _, matched = h.rerank_with_fuzzy_match("football", scores, store_size=7)
        assert {0, 1, 2}.issubset(matched)

    def test_substring_key_in_entity(self):
        """Key 'cornerback' is a substring of entity 'the sport of cornerback'."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6])

        _, matched = h.rerank_with_fuzzy_match(
            "the sport of cornerback", scores, store_size=4
        )
        assert {0, 1, 2}.issubset(matched)

    def test_word_overlap_matching(self):
        """Word overlap: 'finnish language' matches 'finnish' via shared word."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"finnish": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6])

        _, matched = h.rerank_with_fuzzy_match(
            "Finnish language", scores, store_size=4
        )
        assert {0, 1, 2}.issubset(matched)

    def test_stopwords_excluded_from_word_overlap(self):
        """Stopwords ('the', 'of', 'in') should not trigger word overlap."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "the city of rome": {0, 1, 2},
            "paris": {3, 4, 5},
        })
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

        # "the land of ice" shares only stopwords with "the city of rome"
        _, matched = h.rerank_with_fuzzy_match(
            "the land of ice", scores, store_size=7
        )
        # Should NOT match "the city of rome" via stopwords alone
        assert not {0, 1, 2}.issubset(matched)

    def test_fuzzy_is_superset_of_exact(self):
        """Fuzzy matching must return at least as many indices as exact."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "cornerback": {0, 1, 2},
            "cornerback position": {3, 4},
        })
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

        # Exact match would get {0, 1, 2}.
        # Fuzzy match should get {0, 1, 2} plus {3, 4} via substring.
        _, fuzzy_matched = h.rerank_with_fuzzy_match(
            "cornerback", scores, store_size=6
        )
        assert {0, 1, 2}.issubset(fuzzy_matched)

    def test_fuzzy_returns_matched_indices(self):
        """rerank_with_fuzzy_match returns (scores, matched_indices) tuple."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6])

        result = h.rerank_with_fuzzy_match("cornerback", scores, store_size=4)
        assert isinstance(result, tuple)
        assert len(result) == 2
        result_scores, matched_indices = result
        assert isinstance(result_scores, np.ndarray)
        assert isinstance(matched_indices, set)

    def test_fuzzy_case_insensitive(self):
        """Fuzzy matching is case-insensitive."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6])

        _, matched = h.rerank_with_fuzzy_match(
            "CORNERBACK", scores, store_size=4
        )
        assert {0, 1, 2} == matched

    def test_fuzzy_escalation_order(self):
        """Exact match is tried first; substring only if exact < min_candidates."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        # Exact match for "corner" returns nothing (not a key).
        # Substring: "corner" in "cornerback" -> matches {0,1,2}.
        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6])

        _, matched = h.rerank_with_fuzzy_match("corner", scores, store_size=4)
        assert {0, 1, 2}.issubset(matched)

    def test_fuzzy_union_across_keys(self):
        """Substring match unions indices from multiple matching keys."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "finnish culture": {0, 1},
            "finnish language": {2, 3},
            "swedish culture": {4, 5},
        })
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = _make_scores(7)

        _, matched = h.rerank_with_fuzzy_match("finnish", scores, store_size=7)
        # Should match both "finnish culture" and "finnish language"
        assert {0, 1, 2, 3}.issubset(matched)
        # Should NOT match "swedish culture"
        assert 4 not in matched
        assert 5 not in matched


# ===================================================================
# 5. Empty entity index
# ===================================================================

class TestEmptyEntityIndex:
    """Empty entity index is a no-op."""

    def test_rerank_empty_index(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        h = HyperedgeIntersection(entity_index={}, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7])
        result = h.rerank("cornerback", scores, store_size=3)
        assert_array_equal(result, scores)

    def test_fuzzy_empty_index(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        h = HyperedgeIntersection(entity_index={}, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7])
        result_scores, matched = h.rerank_with_fuzzy_match(
            "cornerback", scores, store_size=3
        )
        assert_array_equal(result_scores, scores)
        assert matched == set()

    def test_entity_coverage_empty_index(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        h = HyperedgeIntersection(entity_index={})
        assert h.entity_coverage("anything") == 0


# ===================================================================
# 6. All facts match
# ===================================================================

class TestAllFactsMatch:
    """When all facts mention the entity, no penalty is applied."""

    def test_all_indices_matched(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "cornerback": {0, 1, 2, 3, 4}
        })
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        result = h.rerank("cornerback", scores, store_size=5)
        assert_array_almost_equal(result, scores)

    def test_all_indices_matched_no_modification(self):
        """No index in the output should be less than the input."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "cornerback": {0, 1, 2}
        })
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7])
        result = h.rerank("cornerback", scores, store_size=3)
        assert_array_almost_equal(result, scores)


# ===================================================================
# 7. Penalty = 0.0 (hard filter)
# ===================================================================

class TestPenaltyZero:
    """penalty=0.0 means unmatched scores become 0."""

    def test_hard_filter_zeros_unmatched(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 2, 4}})
        h = HyperedgeIntersection(entity_index, penalty=0.0, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        result = h.rerank("cornerback", scores, store_size=5)

        # Matched indices unchanged
        assert result[0] == pytest.approx(0.9)
        assert result[2] == pytest.approx(0.7)
        assert result[4] == pytest.approx(0.5)
        # Unmatched indices become exactly 0
        assert result[1] == pytest.approx(0.0)
        assert result[3] == pytest.approx(0.0)

    def test_hard_filter_matched_preserved(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {1}})
        h = HyperedgeIntersection(entity_index, penalty=0.0, min_candidates=1)
        scores = np.array([0.9, 0.8, 0.7])
        result = h.rerank("cornerback", scores, store_size=3)
        assert result[1] == pytest.approx(0.8)


# ===================================================================
# 8. Penalty = 1.0 (no effect)
# ===================================================================

class TestPenaltyOne:
    """penalty=1.0 means all scores unchanged."""

    def test_no_effect_penalty(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=1.0, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        result = h.rerank("cornerback", scores, store_size=5)
        assert_array_almost_equal(result, scores)

    def test_no_effect_penalty_with_fuzzy(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=1.0, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        result_scores, _ = h.rerank_with_fuzzy_match(
            "cornerback", scores, store_size=5
        )
        assert_array_almost_equal(result_scores, scores)


# ===================================================================
# 9. Scores array edge cases
# ===================================================================

class TestScoresEdgeCases:
    """Empty array, single element, NaN values."""

    def test_empty_scores_array(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([], dtype=np.float64)
        result = h.rerank("cornerback", scores, store_size=0)
        assert result.shape == (0,)

    def test_single_element_scores(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=1)
        scores = np.array([0.9])
        result = h.rerank("cornerback", scores, store_size=1)
        assert result[0] == pytest.approx(0.9)

    def test_nan_in_scores_passthrough(self):
        """NaN values pass through unchanged -- module does not validate scores."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = np.array([0.9, np.nan, 0.7, 0.6])
        result = h.rerank("cornerback", scores, store_size=4)
        # Index 1 is matched -> NaN stays NaN (unchanged)
        assert np.isnan(result[1])
        # Index 3 is unmatched -> NaN * 0.1 = NaN... but 0.6 * 0.1 = 0.06
        assert result[3] == pytest.approx(0.06)

    def test_inf_in_scores_passthrough(self):
        """Inf values pass through unchanged."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = np.array([np.inf, 0.8, 0.7, 0.6])
        result = h.rerank("cornerback", scores, store_size=4)
        # Index 0 is matched -> inf unchanged
        assert np.isinf(result[0])

    def test_negative_scores(self):
        """Negative scores are valid and should be penalized the same way."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, -0.5])
        result = h.rerank("cornerback", scores, store_size=4)
        assert result[3] == pytest.approx(-0.05)

    def test_scores_longer_than_store_size(self):
        """Scores may be padded beyond store_size; only 0..store_size-1 penalized."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        result = h.rerank("cornerback", scores, store_size=5)
        # Index 3 and 4 are within store_size, unmatched -> penalized
        assert result[3] == pytest.approx(0.06)
        assert result[4] == pytest.approx(0.05)
        # Index 5 and 6 are beyond store_size -> should remain unchanged
        assert result[5] == pytest.approx(0.4)
        assert result[6] == pytest.approx(0.3)

    def test_zero_scores(self):
        """Zero scores remain zero after penalty (0 * penalty = 0)."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = np.array([0.0, 0.0, 0.0, 0.0])
        result = h.rerank("cornerback", scores, store_size=4)
        assert_array_almost_equal(result, [0.0, 0.0, 0.0, 0.0])


# ===================================================================
# 10. Common entity (large match set)
# ===================================================================

class TestCommonEntity:
    """Penalty applied to remaining facts when entity matches many."""

    def test_large_match_set_penalizes_remainder(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        # "united states" matches 500 out of 1000 facts
        matched = set(range(500))
        entity_index = _make_entity_index({"united states": matched})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = _make_scores(1000, value=1.0)
        result = h.rerank("united states", scores, store_size=1000)

        # First 500 unchanged
        assert_array_almost_equal(result[:500], np.ones(500))
        # Last 500 penalized
        assert_array_almost_equal(result[500:], np.full(500, 0.1))

    def test_large_match_set_entity_coverage(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        matched = set(range(500))
        entity_index = _make_entity_index({"united states": matched})
        h = HyperedgeIntersection(entity_index)
        assert h.entity_coverage("united states") == 500

    def test_penalty_preserves_relative_order_within_unmatched(self):
        """Unmatched scores maintain their relative ordering after penalty."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=0.5, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4])
        result = h.rerank("cornerback", scores, store_size=5)
        # Unmatched: index 3 (0.30) > index 4 (0.20) -- order preserved
        assert result[3] > result[4]


# ===================================================================
# 11. entity_coverage
# ===================================================================

class TestEntityCoverage:
    """entity_coverage returns count of matching facts."""

    def test_exact_match_coverage(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index)
        assert h.entity_coverage("cornerback") == 3

    def test_fuzzy_coverage(self):
        """entity_coverage uses fuzzy matching, not just exact."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "association football": {0, 1, 2},
            "football club": {3, 4},
        })
        h = HyperedgeIntersection(entity_index)
        # "football" matches both keys via substring
        coverage = h.entity_coverage("football")
        assert coverage >= 5

    def test_no_match_coverage(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index)
        assert h.entity_coverage("baseball") == 0

    def test_case_insensitive_coverage(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index)
        assert h.entity_coverage("CORNERBACK") == 3


# ===================================================================
# 12. Integration with decompose loop
# ===================================================================

class TestDecomposeLoopIntegration:
    """Mock hop sequence, verify entity from hop N constrains hop N+1."""

    def test_hop0_no_constraint(self):
        """Hop 0 has no previous entity, so no penalty should be applied."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "abbiati": {0, 1},
            "cornerback": {2, 3, 4},
        })
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = _make_scores(10, value=1.0)

        # Hop 0: no intermediate entity (None or empty string)
        # The caller would not call rerank at all, but if called with empty
        # string, it should be a no-op (empty string not in index)
        result = h.rerank("", scores, store_size=10)
        assert_array_equal(result, scores)

    def test_hop1_constrains_by_entity(self):
        """Hop 1: intermediate answer 'cornerback' constrains retrieval."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "abbiati": {0, 1},
            "cornerback": {2, 3, 4},
            "philippines": {5, 6, 7},
        })
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = _make_scores(10, value=1.0)

        # Hop 1: LLM extracted "cornerback" from hop 0's answer
        result = h.rerank("cornerback", scores, store_size=10)

        # Facts about cornerback (2, 3, 4) unchanged
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(1.0)
        assert result[4] == pytest.approx(1.0)
        # Other facts penalized
        assert result[0] == pytest.approx(0.1)  # abbiati fact
        assert result[5] == pytest.approx(0.1)  # philippines fact

    def test_hop_sequence_narrows_progressively(self):
        """
        Simulate 3-hop chain:
          hop 0: 'What sport does Abbiati play?' -> answer: 'cornerback'
          hop 1: 'What country was cornerback created in?' -> rerank by 'cornerback' -> answer: 'Philippines'
          hop 2: 'What is the capital of Philippines?' -> rerank by 'Philippines'
        """
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "abbiati": {0, 1},
            "cornerback": {2, 3, 4},
            "philippines": {4, 5, 6},  # index 4 shared with cornerback
            "manila": {7, 8},
        })
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=2)

        # Hop 1: constrain by "cornerback"
        scores_hop1 = _make_scores(10, value=1.0)
        result_hop1 = h.rerank("cornerback", scores_hop1, store_size=10)
        # cornerback facts (2, 3, 4) preserved
        assert result_hop1[2] == pytest.approx(1.0)
        # abbiati facts penalized
        assert result_hop1[0] == pytest.approx(0.1)

        # Hop 2: constrain by "Philippines"
        scores_hop2 = _make_scores(10, value=1.0)
        result_hop2 = h.rerank("Philippines", scores_hop2, store_size=10)
        # philippines facts (4, 5, 6) preserved
        assert result_hop2[5] == pytest.approx(1.0)
        # manila facts penalized (not about Philippines in the index)
        assert result_hop2[7] == pytest.approx(0.1)

    def test_fuzzy_match_in_decompose_context(self):
        """
        In decompose loop, LLM output might not exactly match entity index.
        rerank_with_fuzzy_match handles this.
        """
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({
            "association football": {0, 1, 2, 3},
        })
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        scores = _make_scores(10, value=1.0)

        # LLM says "football" but index has "association football"
        result_scores, matched = h.rerank_with_fuzzy_match(
            "football", scores, store_size=10
        )
        assert {0, 1, 2, 3}.issubset(matched)
        # Unmatched indices penalized
        assert result_scores[5] == pytest.approx(0.1)


# ===================================================================
# 13. Mutual reference -- entity index can grow
# ===================================================================

class TestLiveEntityIndex:
    """The module holds a reference to the entity index, which may grow."""

    def test_index_growth_reflected(self):
        """New entries added to entity_index after construction are visible."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index: dict[str, set[int]] = {"cornerback": {0, 1, 2}}
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        # Before adding new entity
        result_before = h.rerank("baseball", scores, store_size=5)
        assert_array_equal(result_before, scores)  # no match -> no-op

        # Simulate ingestion adding a new entity
        entity_index["baseball"] = {1, 3, 4}

        # Now "baseball" should match
        result_after = h.rerank("baseball", scores, store_size=5)
        assert result_after[0] == pytest.approx(0.09)  # penalized
        assert result_after[1] == pytest.approx(0.8)   # matched


# ===================================================================
# 14. store_size boundary
# ===================================================================

class TestStoreSizeBoundary:
    """Only indices 0..store_size-1 are penalized."""

    def test_penalty_only_within_store_size(self):
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, penalty=0.1, min_candidates=3)
        # scores has 8 elements, but store_size is 5
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        result = h.rerank("cornerback", scores, store_size=5)

        # Indices 3, 4 within store_size, unmatched -> penalized
        assert result[3] == pytest.approx(0.06)
        assert result[4] == pytest.approx(0.05)
        # Indices 5, 6, 7 beyond store_size -> unchanged
        assert result[5] == pytest.approx(0.4)
        assert result[6] == pytest.approx(0.3)
        assert result[7] == pytest.approx(0.2)

    def test_store_size_zero(self):
        """store_size=0 means no facts to penalize, scores pass through."""
        from mabench.hyperedge_intersection import HyperedgeIntersection

        entity_index = _make_entity_index({"cornerback": {0, 1, 2}})
        h = HyperedgeIntersection(entity_index, min_candidates=3)
        scores = np.array([0.9, 0.8, 0.7])
        result = h.rerank("cornerback", scores, store_size=0)
        assert_array_equal(result, scores)
