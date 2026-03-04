"""Tests for multi-domain longitudinal evaluation dataset generator."""

from __future__ import annotations

from collections import Counter

import pytest

from multidomain_eval import (
    generate_multidomain_dataset,
    _MULTIDOMAIN_CHAINS,
    _BRIDGE_DEFS,
    _REINFORCEMENT_DEFS,
    _SINGLE_MENTIONS,
    _STABLE_BIOGRAPHICAL,
)


class TestDatasetStructure:
    """Verify the generated dataset has the correct structure."""

    @pytest.fixture(scope="class")
    def dataset(self):
        return generate_multidomain_dataset(seed=42)

    def test_session_count(self, dataset):
        assert len(dataset.sessions) == 200

    def test_question_count(self, dataset):
        assert len(dataset.questions) == 100

    def test_session_type_distribution(self, dataset):
        types = Counter(s.session_type for s in dataset.sessions)
        assert types["preference"] == 50
        assert types["update"] == 40
        assert types["repeated"] == 60
        assert types["single"] == 30
        assert types["cross_domain"] == 20

    def test_question_category_distribution(self, dataset):
        cats = Counter(q.category for q in dataset.questions)
        assert cats["current_fact"] == 30
        assert cats["graceful_forgetting"] == 20
        assert cats["reinforced_recall"] == 25
        assert cats["cross_domain"] == 25

    def test_sessions_chronologically_sorted(self, dataset):
        days = [s.day for s in dataset.sessions]
        assert days == sorted(days)

    def test_all_sessions_have_facts(self, dataset):
        for s in dataset.sessions:
            assert len(s.facts) > 0
            assert all(len(f) > 0 for f in s.facts)

    def test_all_questions_have_expected_keywords(self, dataset):
        for q in dataset.questions:
            assert len(q.expected_keywords) > 0

    def test_no_duplicate_questions(self, dataset):
        texts = [q.question for q in dataset.questions]
        assert len(texts) == len(set(texts))


class TestContentDiversity:
    """Verify the content is genuinely diverse across domains."""

    @pytest.fixture(scope="class")
    def dataset(self):
        return generate_multidomain_dataset(seed=42)

    def test_fact_length_diversity(self, dataset):
        """Multi-domain facts should be longer and more diverse than
        single-domain preference sentences."""
        lengths = [len(f) for s in dataset.sessions for f in s.facts]
        # Mean should be well above 50 (preference sentences ~30-40 chars)
        assert sum(lengths) / len(lengths) > 80
        # Max should exceed 200 (long technical descriptions)
        assert max(lengths) > 200

    def test_multiple_domains_in_chains(self):
        """Verify chains span multiple domains."""
        domains = set(d for d, _, _ in _MULTIDOMAIN_CHAINS)
        assert len(domains) == 6
        assert domains == {"code", "personal", "work", "travel", "food", "learning"}

    def test_bridge_cross_domain_pairs(self):
        """Verify bridges connect different domains."""
        for bdef in _BRIDGE_DEFS:
            assert bdef["domain_a"] != bdef["domain_b"], (
                f"Bridge {bdef['keyword_a']}-{bdef['keyword_b']} connects same domain"
            )

    def test_bridge_questions_require_both_keywords(self):
        """Cross-domain questions must require keywords from both domains."""
        for bdef in _BRIDGE_DEFS:
            for q_text, expected_kws in bdef["questions"]:
                assert len(expected_kws) >= 2, (
                    f"Bridge question '{q_text}' has <2 expected keywords"
                )


class TestGracefulForgetting:
    """Verify graceful forgetting questions reference both old and new states."""

    @pytest.fixture(scope="class")
    def dataset(self):
        return generate_multidomain_dataset(seed=42)

    def test_forgetting_questions_have_two_keywords(self, dataset):
        """Each graceful forgetting question should expect both original
        and terminal keywords."""
        gf_qs = [q for q in dataset.questions if q.category == "graceful_forgetting"]
        for q in gf_qs:
            assert len(q.expected_keywords) >= 2, (
                f"Graceful forgetting question '{q.question}' has <2 expected keywords"
            )


class TestCrossDomainBridges:
    """Verify cross-domain bridges are semantically meaningful."""

    def test_bridge_count(self):
        assert len(_BRIDGE_DEFS) == 10

    def test_bridge_seed_sessions_have_different_days(self):
        """Seed sessions should be planted at different times."""
        for bdef in _BRIDGE_DEFS:
            days = [d for d, _ in bdef["seed_facts"]]
            assert len(set(days)) == len(days), (
                f"Bridge {bdef['keyword_a']}-{bdef['keyword_b']} has duplicate seed days"
            )

    def test_bridge_covers_diverse_domain_pairs(self):
        """Bridges should cover many different domain pair combinations."""
        pairs = set()
        for bdef in _BRIDGE_DEFS:
            pair = tuple(sorted([bdef["domain_a"], bdef["domain_b"]]))
            pairs.add(pair)
        # Should have at least 6 unique domain pairs
        assert len(pairs) >= 6


class TestDeterminism:
    """Verify the dataset is deterministic."""

    def test_same_seed_same_dataset(self):
        ds1 = generate_multidomain_dataset(seed=42)
        ds2 = generate_multidomain_dataset(seed=42)

        assert len(ds1.sessions) == len(ds2.sessions)
        assert len(ds1.questions) == len(ds2.questions)

        for s1, s2 in zip(ds1.sessions, ds2.sessions):
            assert s1.day == s2.day
            assert s1.session_type == s2.session_type
            assert s1.facts == s2.facts

        for q1, q2 in zip(ds1.questions, ds2.questions):
            assert q1.question == q2.question
            assert q1.expected_keywords == q2.expected_keywords
            assert q1.category == q2.category

    def test_different_seed_different_dataset(self):
        ds1 = generate_multidomain_dataset(seed=42)
        ds2 = generate_multidomain_dataset(seed=99)

        # Same counts but different ordering
        assert len(ds1.sessions) == len(ds2.sessions)
        # Day assignments should differ
        days1 = [s.day for s in ds1.sessions]
        days2 = [s.day for s in ds2.sessions]
        assert days1 != days2


class TestReinforcementTopics:
    """Verify reinforcement topic coverage."""

    def test_reinforcement_count(self):
        assert len(_REINFORCEMENT_DEFS) == 8

    def test_all_topics_have_multiple_details(self):
        for rdef in _REINFORCEMENT_DEFS:
            assert len(rdef["details"]) >= 4, (
                f"Topic {rdef['domain']} has <4 detail sentences"
            )

    def test_details_are_diverse(self):
        """Detail sentences within a topic should be distinct."""
        for rdef in _REINFORCEMENT_DEFS:
            details = rdef["details"]
            # No duplicates
            assert len(details) == len(set(details))
            # Details should have varied first words
            first_words = set(d.split()[0] for d in details)
            assert len(first_words) >= len(details) // 2
