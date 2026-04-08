"""Tests for mabench.entity_graph -- EntityGraph, EntityNode, Fact.

TDD: written BEFORE implementation. All tests should fail (ImportError)
until entity_graph.py is implemented.

Test categories:
  1. Fact / EntityNode dataclass integrity
  2. Normalization helpers (_normalize, _strip_determiners, _strip_possessive)
  3. Basic Entity Creation (7 tests)
  4. Entity Resolution (16 tests, incl. parametrize)
  5. Cross-Turn Entity Linking (5 tests)
  6. Supersession (6 tests)
  7. Entity Mention & Register Alias (5 tests)
  8. Query (query_entity, query_facts) (8 tests)
  9. Graph Traversal (get_evidence_turns, get_related_entities) (8 tests)
 10. Retrieval Integration (retrieve_for_question) (6 tests)
 11. Serialization (to_dict / from_dict) (6 tests)
 12. Stats (3 tests)
 13. Edge Cases (8 tests)
"""
from __future__ import annotations

from datetime import datetime

import pytest

from mabench.entity_graph import EntityGraph, EntityNode, Fact


# ===========================================================================
# 1. Fact / EntityNode dataclass integrity
# ===========================================================================


class TestFactDataclass:
    """Verify Fact dataclass fields and defaults."""

    def test_fact_default_timestamp_none(self):
        f = Fact(predicate="cost", object="$15", turn_idx=0, chunk_idx=-1)
        assert f.timestamp is None

    def test_fact_default_superseded_false(self):
        f = Fact(predicate="cost", object="$15", turn_idx=0, chunk_idx=-1)
        assert f.superseded is False

    def test_fact_default_source_text_empty(self):
        f = Fact(predicate="cost", object="$15", turn_idx=0, chunk_idx=-1)
        assert f.source_text == ""

    def test_fact_all_fields(self):
        ts = datetime(2025, 1, 1)
        f = Fact(
            predicate="cost", object="$15", turn_idx=3, chunk_idx=7,
            timestamp=ts, superseded=True, source_text="The bowl cost $15",
        )
        assert f.predicate == "cost"
        assert f.object == "$15"
        assert f.turn_idx == 3
        assert f.chunk_idx == 7
        assert f.timestamp == ts
        assert f.superseded is True
        assert f.source_text == "The bowl cost $15"


class TestEntityNodeDataclass:
    """Verify EntityNode dataclass structure."""

    def test_entity_node_fields(self):
        node = EntityNode(
            canonical="luna",
            aliases={"my cat", "the cat"},
            facts={"is_a": [Fact("is_a", "cat", 0, -1)]},
            related={"cat"},
            mention_turns={0, 3},
        )
        assert node.canonical == "luna"
        assert "my cat" in node.aliases
        assert "is_a" in node.facts
        assert "cat" in node.related
        assert 0 in node.mention_turns

    def test_entity_node_empty_collections(self):
        node = EntityNode(
            canonical="luna",
            aliases=set(),
            facts={},
            related=set(),
            mention_turns=set(),
        )
        assert node.canonical == "luna"
        assert len(node.aliases) == 0
        assert len(node.facts) == 0


# ===========================================================================
# 2. Normalization helpers
# ===========================================================================


class TestNormalization:
    def test_normalize_lowercase_strip(self):
        assert EntityGraph._normalize("  Hello World  ") == "hello world"

    def test_normalize_collapse_whitespace(self):
        assert EntityGraph._normalize("foo   bar") == "foo bar"

    def test_normalize_empty(self):
        assert EntityGraph._normalize("") == ""

    def test_strip_determiners_the(self):
        assert EntityGraph._strip_determiners("the cat") == "cat"

    def test_strip_determiners_my(self):
        assert EntityGraph._strip_determiners("my dog") == "dog"

    def test_strip_determiners_case_insensitive(self):
        assert EntityGraph._strip_determiners("The Big Cat") == "Big Cat"

    def test_strip_determiners_no_match(self):
        assert EntityGraph._strip_determiners("cat") == "cat"

    def test_strip_determiners_an(self):
        assert EntityGraph._strip_determiners("an apple") == "apple"

    def test_strip_determiners_a(self):
        assert EntityGraph._strip_determiners("a car") == "car"

    @pytest.mark.parametrize("det", ["his", "her", "their", "our", "this"])
    def test_strip_determiners_all(self, det: str):
        assert EntityGraph._strip_determiners(f"{det} car") == "car"

    def test_strip_possessive_trailing(self):
        assert EntityGraph._strip_possessive("sarah's") == "sarah"

    def test_strip_possessive_mid_string(self):
        assert EntityGraph._strip_possessive("sarah's cat") == "sarah cat"

    def test_strip_possessive_no_match(self):
        assert EntityGraph._strip_possessive("dogs") == "dogs"


# ===========================================================================
# 3. Basic Entity Creation
# ===========================================================================


class TestBasicEntityCreation:
    """Ingest triples and verify entity nodes are created correctly."""

    def test_ingest_one_triple_creates_entity(self):
        """Ingesting a single triple creates the subject entity node."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        node = g.query_entity("luna")
        assert node is not None
        assert node.canonical == "luna"

    def test_ingest_one_triple_records_fact(self):
        """The fact appears on the subject entity with correct fields."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0, chunk_idx=5,
                        source_text="Luna is a cat")
        facts = g.query_facts("luna", "is_a")
        assert len(facts) == 1
        assert facts[0].predicate == "is_a"
        assert facts[0].object == "cat"
        assert facts[0].turn_idx == 0
        assert facts[0].chunk_idx == 5
        assert facts[0].source_text == "Luna is a cat"
        assert facts[0].superseded is False

    def test_ingest_same_entity_multiple_times_single_node(self):
        """Multiple triples with same subject reuse one EntityNode."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.ingest_triple("luna", "age", "3 years", turn_idx=1)
        node = g.query_entity("luna")
        assert node is not None
        assert "is_a" in node.facts
        assert "age" in node.facts

    def test_entity_mention_without_triples(self):
        """ingest_entity_mention creates entity with mention_turns but no facts."""
        g = EntityGraph()
        g.ingest_entity_mention("luna", turn_idx=7)
        node = g.query_entity("luna")
        assert node is not None
        assert 7 in node.mention_turns
        assert len(node.facts) == 0

    def test_ingest_triple_records_turn_idx_on_entity(self):
        """Ingesting a triple adds turn_idx to mention_turns."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=3)
        node = g.query_entity("luna")
        assert 3 in node.mention_turns

    def test_empty_graph_query_returns_none(self):
        """Querying an empty graph returns None / empty."""
        g = EntityGraph()
        assert g.query_entity("luna") is None
        assert g.query_facts("luna") == []
        assert g.get_evidence_turns("luna") == set()

    def test_ingest_creates_object_entity(self):
        """Ingesting a triple also creates the object as an entity node."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        node = g.query_entity("cat")
        assert node is not None
        assert 0 in node.mention_turns


# ===========================================================================
# 4. Entity Resolution
# ===========================================================================


class TestEntityResolution:
    """Tests for resolve() with normalization, determiners, substrings, aliases."""

    def test_exact_match_lowercase(self):
        g = EntityGraph()
        g.ingest_triple("alice", "likes", "cats", turn_idx=0)
        assert g.resolve("Alice") == "alice"
        assert g.resolve("ALICE") == "alice"
        assert g.resolve("alice") == "alice"

    def test_whitespace_stripping(self):
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        assert g.resolve("  luna  ") == "luna"

    def test_determiner_the(self):
        g = EntityGraph()
        g.ingest_triple("cat", "likes", "fish", turn_idx=0)
        assert g.resolve("the cat") == "cat"

    def test_determiner_my(self):
        g = EntityGraph()
        g.ingest_triple("dog", "breed", "poodle", turn_idx=0)
        assert g.resolve("my dog") == "dog"

    def test_determiner_that(self):
        g = EntityGraph()
        g.ingest_triple("food bowl", "cost", "$15", turn_idx=0)
        assert g.resolve("that food bowl") == "food bowl"

    def test_determiner_a(self):
        g = EntityGraph()
        g.ingest_triple("cat", "likes", "fish", turn_idx=0)
        assert g.resolve("a cat") == "cat"

    def test_determiner_an(self):
        g = EntityGraph()
        g.ingest_triple("apple", "color", "red", turn_idx=0)
        assert g.resolve("an apple") == "apple"

    @pytest.mark.parametrize("det", ["his", "her", "their", "our", "this"])
    def test_determiners_all(self, det: str):
        g = EntityGraph()
        g.ingest_triple("cat", "likes", "fish", turn_idx=0)
        assert g.resolve(f"{det} cat") == "cat"

    def test_substring_mention_contains_canonical(self):
        """'expensive food bowl' contains 'food bowl' -> match."""
        g = EntityGraph()
        g.ingest_triple("food bowl", "cost", "$15", turn_idx=0)
        assert g.resolve("expensive food bowl") == "food bowl"

    def test_substring_canonical_contains_mention(self):
        """'retriever' is substring of 'golden retriever' -> match."""
        g = EntityGraph()
        g.ingest_triple("golden retriever", "is_a", "dog", turn_idx=0)
        assert g.resolve("retriever") == "golden retriever"

    def test_register_explicit_alias(self):
        g = EntityGraph()
        g.ingest_triple("cat", "likes", "fish", turn_idx=0)
        g.register_alias("kitty", "cat")
        assert g.resolve("kitty") == "cat"

    def test_alias_chain_with_determiner(self):
        """register_alias('kitty', 'cat') then resolve('the kitty') -> 'cat'."""
        g = EntityGraph()
        g.ingest_triple("cat", "likes", "fish", turn_idx=0)
        g.register_alias("kitty", "cat")
        assert g.resolve("the kitty") == "cat"

    def test_unknown_entity_returns_none(self):
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        assert g.resolve("spaceship") is None

    def test_case_insensitive_alias(self):
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.register_alias("Kitty Luna", "luna")
        assert g.resolve("kitty luna") == "luna"
        assert g.resolve("KITTY LUNA") == "luna"

    def test_possessive_stripping(self):
        """\"user's car\" should match 'car' by stripping possessive."""
        g = EntityGraph()
        g.ingest_triple("car", "color", "blue", turn_idx=0)
        assert g.resolve("user's car") == "car"

    def test_resolve_or_create_existing(self):
        """_resolve_or_create returns existing canonical."""
        g = EntityGraph()
        g.ingest_triple("alice", "likes", "cats", turn_idx=0)
        canon = g._resolve_or_create("The Alice")
        assert canon == "alice"

    def test_resolve_or_create_new(self):
        """_resolve_or_create creates new entity when no match."""
        g = EntityGraph()
        canon = g._resolve_or_create("The Big Dog")
        assert canon == "big dog"
        assert "big dog" in g.entities
        assert "the big dog" in g.alias_index


# ===========================================================================
# 5. Cross-Turn Entity Linking
# ===========================================================================


class TestCrossTurnEntityLinking:
    """Triples from different turns referencing same entity merge."""

    def test_determiner_links_across_turns(self):
        """Turn 1: ('food bowl','cost','$15') -> Turn 2: ('that food bowl','cost','$20')
        -> same entity."""
        g = EntityGraph()
        g.ingest_triple("food bowl", "cost", "$15", turn_idx=1)
        g.ingest_triple("that food bowl", "cost", "$20", turn_idx=2)
        canon1 = g.resolve("food bowl")
        canon2 = g.resolve("that food bowl")
        assert canon1 == canon2
        facts = g.query_facts(canon1, "cost", current_only=False)
        assert len(facts) == 2

    def test_multiple_aliases_accumulate(self):
        """'the dog', 'my dog', and alias 'Baxter' all link to same entity."""
        g = EntityGraph()
        g.ingest_triple("dog", "breed", "golden retriever", turn_idx=0)
        g.register_alias("baxter", "dog")
        assert g.resolve("the dog") == "dog"
        assert g.resolve("my dog") == "dog"
        assert g.resolve("baxter") == "dog"

    def test_turn_indices_accumulate_across_resolutions(self):
        """mention_turns accumulates from all turns referencing the entity."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.ingest_entity_mention("luna", turn_idx=3)
        g.ingest_triple("luna", "weight", "8 lbs", turn_idx=5)
        node = g.query_entity("luna")
        assert {0, 3, 5}.issubset(node.mention_turns)

    def test_alias_links_entity_mentions(self):
        """Mentions via alias accumulate on the canonical entity."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.register_alias("my cat", "luna")
        g.ingest_entity_mention("my cat", turn_idx=4)
        node = g.query_entity("luna")
        assert 4 in node.mention_turns

    def test_related_entities_from_triples(self):
        """Ingesting ('luna','is_a','cat') makes 'cat' a related entity of 'luna'."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        node = g.query_entity("luna")
        assert "cat" in node.related


# ===========================================================================
# 6. Supersession
# ===========================================================================


class TestSupersession:
    """When same (entity, predicate) gets a new fact, old ones are superseded."""

    def test_supersedes_old_fact(self):
        g = EntityGraph()
        g.ingest_triple("cat", "cost", "$20", turn_idx=0)
        g.ingest_triple("cat", "cost", "$30", turn_idx=5)
        facts = g.entities["cat"].facts["cost"]
        assert len(facts) == 2
        assert facts[0].superseded is True
        assert facts[0].object == "$20"
        assert facts[1].superseded is False
        assert facts[1].object == "$30"

    def test_different_predicate_no_supersession(self):
        g = EntityGraph()
        g.ingest_triple("cat", "cost", "$20", turn_idx=0)
        g.ingest_triple("cat", "color", "black", turn_idx=5)
        assert g.entities["cat"].facts["cost"][0].superseded is False
        assert g.entities["cat"].facts["color"][0].superseded is False

    def test_triple_supersession_chain(self):
        """Three values for same predicate: only last is current."""
        g = EntityGraph()
        g.ingest_triple("stock", "price", "100", turn_idx=0)
        g.ingest_triple("stock", "price", "110", turn_idx=1)
        g.ingest_triple("stock", "price", "120", turn_idx=2)
        facts = g.entities["stock"].facts["price"]
        assert len(facts) == 3
        assert all(f.superseded for f in facts[:2])
        assert facts[2].superseded is False
        assert facts[2].object == "120"

    def test_query_facts_current_only_excludes_superseded(self):
        g = EntityGraph()
        g.ingest_triple("food bowl", "cost", "$15", turn_idx=1)
        g.ingest_triple("food bowl", "cost", "$18", turn_idx=2)
        g.ingest_triple("food bowl", "cost", "$20", turn_idx=3)
        current = g.query_facts("food bowl", "cost", current_only=True)
        assert len(current) == 1
        assert current[0].object == "$20"

    def test_query_facts_current_only_false_includes_all(self):
        g = EntityGraph()
        g.ingest_triple("luna", "weight", "7 lbs", turn_idx=0)
        g.ingest_triple("luna", "weight", "8 lbs", turn_idx=5)
        all_facts = g.query_facts("luna", "weight", current_only=False)
        assert len(all_facts) == 2

    def test_supersession_via_alias(self):
        """Supersession works even when second triple uses an alias."""
        g = EntityGraph()
        g.ingest_triple("food bowl", "cost", "$15", turn_idx=1)
        g.ingest_triple("the food bowl", "cost", "$20", turn_idx=2)
        current = g.query_facts("food bowl", "cost", current_only=True)
        assert len(current) == 1
        assert current[0].object == "$20"


# ===========================================================================
# 7. Entity Mention & Register Alias
# ===========================================================================


class TestIngestEntityMention:
    def test_mention_creates_entity(self):
        g = EntityGraph()
        g.ingest_entity_mention("Alice", turn_idx=7)
        assert "alice" in g.entities
        assert 7 in g.entities["alice"].mention_turns

    def test_mention_adds_to_existing(self):
        g = EntityGraph()
        g.ingest_triple("alice", "likes", "cats", turn_idx=0)
        g.ingest_entity_mention("alice", turn_idx=7)
        assert {0, 7} <= g.entities["alice"].mention_turns


class TestRegisterAlias:
    def test_alias_registered(self):
        g = EntityGraph()
        g.ingest_triple("alice", "likes", "cats", turn_idx=0)
        g.register_alias("Ali", "alice")
        assert g.resolve("Ali") == "alice"

    def test_alias_creates_entity_if_needed(self):
        g = EntityGraph()
        g.register_alias("Ali", "alice")
        assert "alice" in g.entities
        assert g.resolve("Ali") == "alice"

    def test_alias_stored_on_node(self):
        """Registered alias appears in the EntityNode.aliases set."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.register_alias("kitty", "luna")
        node = g.query_entity("luna")
        assert "kitty" in node.aliases


# ===========================================================================
# 8. Query (query_entity, query_facts)
# ===========================================================================


class TestQueryEntity:
    def test_query_existing(self):
        g = EntityGraph()
        g.ingest_triple("alice", "likes", "cats", turn_idx=0)
        node = g.query_entity("alice")
        assert node is not None
        assert node.canonical == "alice"

    def test_query_via_alias(self):
        g = EntityGraph()
        g.ingest_triple("alice", "likes", "cats", turn_idx=0)
        g.register_alias("Ali", "alice")
        node = g.query_entity("Ali")
        assert node is not None
        assert node.canonical == "alice"

    def test_query_missing(self):
        g = EntityGraph()
        assert g.query_entity("nobody") is None


class TestQueryFacts:
    def test_current_only(self):
        g = EntityGraph()
        g.ingest_triple("cat", "cost", "$20", turn_idx=0)
        g.ingest_triple("cat", "cost", "$30", turn_idx=5)
        facts = g.query_facts("cat", "cost", current_only=True)
        assert len(facts) == 1
        assert facts[0].object == "$30"

    def test_include_superseded(self):
        g = EntityGraph()
        g.ingest_triple("cat", "cost", "$20", turn_idx=0)
        g.ingest_triple("cat", "cost", "$30", turn_idx=5)
        facts = g.query_facts("cat", "cost", current_only=False)
        assert len(facts) == 2

    def test_newest_first(self):
        """query_facts returns facts with newest first."""
        g = EntityGraph()
        g.ingest_triple("cat", "cost", "$20", turn_idx=0)
        g.ingest_triple("cat", "cost", "$30", turn_idx=5)
        facts = g.query_facts("cat", "cost", current_only=False)
        assert facts[0].turn_idx == 5

    def test_filter_by_predicate(self):
        g = EntityGraph()
        g.ingest_triple("cat", "cost", "$20", turn_idx=0)
        g.ingest_triple("cat", "color", "black", turn_idx=1)
        facts = g.query_facts("cat", "color")
        assert len(facts) == 1
        assert facts[0].object == "black"

    def test_no_predicate_returns_all(self):
        g = EntityGraph()
        g.ingest_triple("cat", "cost", "$20", turn_idx=0)
        g.ingest_triple("cat", "color", "black", turn_idx=1)
        facts = g.query_facts("cat")
        assert len(facts) == 2

    def test_unresolvable_entity(self):
        g = EntityGraph()
        assert g.query_facts("nobody") == []

    def test_query_via_alias(self):
        """query_facts resolves entity via alias before lookup."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.register_alias("kitty", "luna")
        facts = g.query_facts("kitty", "is_a")
        assert len(facts) == 1
        assert facts[0].object == "cat"

    def test_query_via_determiner(self):
        """query_facts resolves via determiner stripping."""
        g = EntityGraph()
        g.ingest_triple("food bowl", "cost", "$15", turn_idx=0)
        facts = g.query_facts("the food bowl", "cost")
        assert len(facts) == 1


# ===========================================================================
# 9. Graph Traversal
# ===========================================================================


class TestGetEvidenceTurns:
    def test_basic(self):
        g = EntityGraph()
        g.ingest_triple("alice", "met", "bob", turn_idx=3)
        g.ingest_entity_mention("alice", turn_idx=7)
        turns = g.get_evidence_turns("alice")
        assert turns == {3, 7}

    def test_with_predicate_filter(self):
        g = EntityGraph()
        g.ingest_triple("alice", "met", "bob", turn_idx=3)
        g.ingest_triple("alice", "likes", "cats", turn_idx=5)
        g.ingest_entity_mention("alice", turn_idx=7)
        turns = g.get_evidence_turns("alice", predicate="met")
        assert 3 in turns
        assert 7 in turns  # mention_turns always included

    def test_excludes_superseded_fact_turns(self):
        """get_evidence_turns excludes turn indices from superseded facts."""
        g = EntityGraph()
        g.ingest_triple("food bowl", "cost", "$15", turn_idx=1)
        g.ingest_triple("food bowl", "cost", "$20", turn_idx=3)
        turns = g.get_evidence_turns("food bowl")
        assert 3 in turns
        # Turn 1 is superseded -- should NOT be in evidence
        assert 1 not in turns

    def test_unresolvable(self):
        g = EntityGraph()
        assert g.get_evidence_turns("nobody") == set()

    def test_via_alias(self):
        """get_evidence_turns resolves via alias."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.register_alias("kitty", "luna")
        turns = g.get_evidence_turns("kitty")
        assert 0 in turns


class TestGetRelatedEntities:
    def test_one_hop(self):
        g = EntityGraph()
        g.ingest_triple("alice", "met", "bob", turn_idx=0)
        g.ingest_triple("bob", "likes", "charlie", turn_idx=1)
        related = g.get_related_entities("alice", max_hops=1)
        assert "bob" in related
        assert "charlie" not in related  # 2 hops away

    def test_two_hops(self):
        g = EntityGraph()
        g.ingest_triple("alice", "met", "bob", turn_idx=0)
        g.ingest_triple("bob", "likes", "charlie", turn_idx=1)
        related = g.get_related_entities("alice", max_hops=2)
        assert "bob" in related
        assert "charlie" in related

    def test_excludes_self(self):
        g = EntityGraph()
        g.ingest_triple("alice", "met", "bob", turn_idx=0)
        related = g.get_related_entities("alice")
        assert "alice" not in related

    def test_unresolvable(self):
        g = EntityGraph()
        assert g.get_related_entities("nobody") == set()

    def test_bidirectional(self):
        """If ('luna','is_a','cat'), then 'cat' has 'luna' as related."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        cat_related = g.get_related_entities("cat", max_hops=1)
        assert "luna" in cat_related

    def test_no_self_in_multihop(self):
        """Multi-hop traversal does not include origin entity."""
        g = EntityGraph()
        g.ingest_triple("alice", "met", "bob", turn_idx=0)
        g.ingest_triple("bob", "likes", "charlie", turn_idx=1)
        g.ingest_triple("charlie", "knows", "alice", turn_idx=2)
        related = g.get_related_entities("alice", max_hops=3)
        assert "alice" not in related


# ===========================================================================
# 10. Retrieval Integration
# ===========================================================================


class TestRetrieveForQuestion:
    def test_resolves_question_entities(self):
        g = EntityGraph()
        g.ingest_triple("alice", "bought", "guitar", turn_idx=5)
        g.ingest_triple("alice", "lives_in", "paris", turn_idx=10)
        g.ingest_entity_mention("alice", turn_idx=15)
        turns = g.retrieve_for_question("What did Alice buy?")
        assert 5 in turns

    def test_multi_entity_question(self):
        g = EntityGraph()
        g.ingest_triple("alice", "met", "bob", turn_idx=3)
        g.ingest_triple("bob", "likes", "cats", turn_idx=7)
        turns = g.retrieve_for_question("When did Alice meet Bob?")
        assert 3 in turns

    def test_max_turns_limit(self):
        g = EntityGraph()
        for i in range(50):
            g.ingest_triple("alice", f"action_{i}", f"thing_{i}", turn_idx=i)
        turns = g.retrieve_for_question("What about Alice?", max_turns=10)
        assert len(turns) <= 10

    def test_empty_question(self):
        g = EntityGraph()
        g.ingest_triple("alice", "likes", "cats", turn_idx=0)
        turns = g.retrieve_for_question("")
        assert turns == []

    def test_no_match(self):
        g = EntityGraph()
        g.ingest_triple("alice", "likes", "cats", turn_idx=0)
        turns = g.retrieve_for_question("What is the weather?")
        assert isinstance(turns, list)

    def test_with_determiner_in_question(self):
        """'What breed is the dog?' resolves 'the dog' to 'dog'."""
        g = EntityGraph()
        g.ingest_triple("dog", "breed", "poodle", turn_idx=0)
        g.ingest_triple("dog", "name", "rex", turn_idx=1)
        turns = g.retrieve_for_question("What breed is the dog?")
        assert 0 in turns

    def test_returns_sorted_turns(self):
        """Returned turn indices are sorted ascending for deterministic output."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=5)
        g.ingest_triple("luna", "age", "3", turn_idx=2)
        g.ingest_entity_mention("luna", turn_idx=8)
        turns = g.retrieve_for_question("Tell me about luna")
        if len(turns) > 1:
            assert turns == sorted(turns)


# ===========================================================================
# 11. Serialization
# ===========================================================================


class TestSerialization:
    def test_round_trip(self):
        g = EntityGraph()
        ts = datetime(2024, 3, 15, 10, 30)
        g.ingest_triple("alice", "bought", "guitar", turn_idx=5,
                        chunk_idx=42, timestamp=ts,
                        source_text="Alice bought a guitar.")
        g.ingest_triple("alice", "likes", "cats", turn_idx=10)
        g.ingest_triple("bob", "met", "alice", turn_idx=3)
        g.register_alias("Ali", "alice")

        data = g.to_dict()
        g2 = EntityGraph.from_dict(data)

        # Same entities
        assert set(g2.entities.keys()) == set(g.entities.keys())
        # Same aliases
        assert g2.alias_index == g.alias_index
        # Same facts
        alice = g2.entities["alice"]
        assert "bought" in alice.facts
        f = alice.facts["bought"][0]
        assert f.object == "guitar"
        assert f.turn_idx == 5
        assert f.chunk_idx == 42
        assert f.timestamp == ts
        assert f.source_text == "Alice bought a guitar."
        # Related
        assert "bob" in alice.related
        # Mention turns
        assert 5 in alice.mention_turns
        assert 10 in alice.mention_turns

    def test_empty_graph_round_trip(self):
        g = EntityGraph()
        data = g.to_dict()
        g2 = EntityGraph.from_dict(data)
        assert len(g2.entities) == 0
        assert len(g2.alias_index) == 0

    def test_superseded_fact_survives_round_trip(self):
        g = EntityGraph()
        g.ingest_triple("cat", "cost", "$20", turn_idx=0)
        g.ingest_triple("cat", "cost", "$30", turn_idx=5)
        data = g.to_dict()
        g2 = EntityGraph.from_dict(data)
        facts = g2.entities["cat"].facts["cost"]
        assert len(facts) == 2
        assert facts[0].superseded is True
        assert facts[1].superseded is False

    def test_mention_turns_survive_round_trip(self):
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.ingest_entity_mention("luna", turn_idx=5)
        data = g.to_dict()
        g2 = EntityGraph.from_dict(data)
        node = g2.query_entity("luna")
        assert {0, 5}.issubset(node.mention_turns)

    def test_related_survives_round_trip(self):
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        data = g.to_dict()
        g2 = EntityGraph.from_dict(data)
        node = g2.query_entity("luna")
        assert "cat" in node.related

    def test_timestamps_survive_round_trip(self):
        ts = datetime(2025, 3, 15, 10, 30, 0)
        g = EntityGraph()
        g.ingest_triple("luna", "vet_visit", "annual checkup", turn_idx=0,
                        timestamp=ts)
        data = g.to_dict()
        g2 = EntityGraph.from_dict(data)
        facts = g2.query_facts("luna", "vet_visit")
        assert len(facts) == 1
        assert facts[0].timestamp == ts


# ===========================================================================
# 12. Stats
# ===========================================================================


class TestStats:
    def test_stats_basic(self):
        g = EntityGraph()
        g.ingest_triple("alice", "likes", "cats", turn_idx=0)
        g.ingest_triple("alice", "met", "bob", turn_idx=1)
        g.register_alias("Ali", "alice")
        s = g.stats()
        assert s["entity_count"] == 3  # alice, cats, bob
        assert s["fact_count"] == 2
        assert s["current_fact_count"] == 2
        assert s["alias_count"] >= 1

    def test_stats_empty(self):
        g = EntityGraph()
        s = g.stats()
        assert s["entity_count"] == 0
        assert s["fact_count"] == 0
        assert s["avg_facts_per_entity"] == 0
        assert s["avg_aliases_per_entity"] == 0

    def test_stats_with_supersession(self):
        g = EntityGraph()
        g.ingest_triple("cat", "cost", "$20", turn_idx=0)
        g.ingest_triple("cat", "cost", "$30", turn_idx=5)
        s = g.stats()
        assert s["fact_count"] == 2
        assert s["current_fact_count"] == 1


# ===========================================================================
# 13. Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_subject_ignored(self):
        g = EntityGraph()
        g.ingest_triple("", "cost", "$15", turn_idx=0)
        assert len(g.entities) == 0

    def test_empty_predicate_ignored(self):
        g = EntityGraph()
        g.ingest_triple("luna", "", "cat", turn_idx=0)
        assert len(g.entities) == 0

    def test_whitespace_only_predicate_ignored(self):
        g = EntityGraph()
        g.ingest_triple("alice", "  ", "bob", turn_idx=0)
        assert len(g.entities) == 0

    def test_empty_object_ignored(self):
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "", turn_idx=0)
        assert len(g.entities) == 0

    def test_very_long_entity_name(self):
        g = EntityGraph()
        long_name = "a" * 500
        g.ingest_triple(long_name, "is_a", "thing", turn_idx=0)
        node = g.query_entity(long_name)
        assert node is not None

    def test_unicode_entity_names(self):
        g = EntityGraph()
        g.ingest_triple("cafe", "location", "paris", turn_idx=0)
        assert g.query_entity("cafe") is not None
        g.ingest_triple("tokyo", "country", "japan", turn_idx=1)
        assert g.query_entity("tokyo") is not None

    def test_duplicate_triple_no_duplicate_facts(self):
        """Exact same triple ingested twice does not create duplicate facts."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0,
                        source_text="Luna is a cat")
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0,
                        source_text="Luna is a cat")
        facts = g.query_facts("luna", "is_a", current_only=False)
        assert len(facts) == 1

    def test_mention_turns_and_fact_turns_accumulate(self):
        """Both explicit mentions and triples contribute to mention_turns."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.ingest_entity_mention("luna", turn_idx=2)
        g.ingest_triple("luna", "weight", "8 lbs", turn_idx=5)
        g.ingest_entity_mention("luna", turn_idx=7)
        node = g.query_entity("luna")
        assert node.mention_turns == {0, 2, 5, 7}

    def test_ingest_triple_with_timestamp(self):
        ts = datetime(2025, 6, 15, 14, 30, 0)
        g = EntityGraph()
        g.ingest_triple("luna", "vet_visit", "checkup", turn_idx=0,
                        timestamp=ts)
        facts = g.query_facts("luna", "vet_visit")
        assert len(facts) == 1
        assert facts[0].timestamp == ts

    def test_ingest_triple_default_chunk_idx(self):
        """chunk_idx defaults to -1."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        facts = g.query_facts("luna", "is_a")
        assert facts[0].chunk_idx == -1

    def test_auto_resolve_subject_to_existing(self):
        """Ingesting 'the food bowl' auto-resolves to existing 'food bowl'."""
        g = EntityGraph()
        g.ingest_triple("food bowl", "cost", "$15", turn_idx=1)
        g.ingest_triple("the food bowl", "color", "blue", turn_idx=2)
        node = g.query_entity("food bowl")
        assert "cost" in node.facts
        assert "color" in node.facts

    def test_ingest_creates_new_for_unknown(self):
        """Truly new entity creates a new node."""
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.ingest_triple("spaceship", "flies_to", "mars", turn_idx=1)
        assert g.query_entity("spaceship") is not None
        assert g.query_entity("luna") is not None


class TestAnswerFromGraph:
    """Tests for answer_from_graph with predicate ranking and multi-hop."""

    @staticmethod
    def _build_pet_graph() -> EntityGraph:
        g = EntityGraph()
        g.ingest_triple("luna", "is_a", "cat", turn_idx=0)
        g.ingest_triple("luna", "age", "3 years", turn_idx=0)
        g.ingest_triple("luna", "vet", "dr. martinez", turn_idx=1)
        g.ingest_triple("dr. martinez", "works at", "happy paws clinic", turn_idx=2)
        g.ingest_triple("happy paws clinic", "located in", "downtown seattle", turn_idx=2)
        g.ingest_triple("food bowl", "purchased at", "pet world", turn_idx=5)
        g.ingest_triple("pet world", "located in", "bellevue", turn_idx=5)
        g.ingest_triple("food bowl", "brand", "kong", turn_idx=5)
        g.ingest_triple("kong", "made in", "colorado", turn_idx=6)
        return g

    def test_single_hop_direct(self):
        g = self._build_pet_graph()
        ans = g.answer_from_graph("luna", "age", "How old is Luna?")
        assert ans == "3 years"

    def test_single_hop_brand(self):
        g = self._build_pet_graph()
        ans = g.answer_from_graph("food bowl", "brand", "What brand is the food bowl?")
        assert ans == "kong"

    def test_single_hop_terminates(self):
        """'Who is Luna's vet?' should return dr. martinez, not walk further."""
        g = self._build_pet_graph()
        ans = g.answer_from_graph("luna", "vet", "Who is Luna's vet?")
        assert ans == "dr. martinez"

    def test_two_hop_works_at(self):
        """luna → vet → dr. martinez → works at → happy paws clinic."""
        g = self._build_pet_graph()
        ans = g.answer_from_graph("luna", "works at", "Where does Luna's vet work?")
        assert ans == "happy paws clinic"

    def test_two_hop_brand_made(self):
        """food bowl → brand → kong → made in → colorado."""
        g = self._build_pet_graph()
        ans = g.answer_from_graph("food bowl", "made in", "Where is the food bowl brand made?")
        assert ans == "colorado"

    def test_three_hop_located(self):
        """luna → vet → dr.martinez → works at → happy paws → located in → seattle."""
        g = self._build_pet_graph()
        ans = g.answer_from_graph("luna", "located in", "Where is Luna's vet clinic located?")
        assert ans == "downtown seattle"

    def test_unresolved_entity_returns_none(self):
        g = self._build_pet_graph()
        ans = g.answer_from_graph("weather", "temperature", "What is the weather today?")
        assert ans is None

    def test_empty_inputs_return_none(self):
        g = self._build_pet_graph()
        assert g.answer_from_graph("", "age", "How old?") is None
        assert g.answer_from_graph("luna", "", "How old?") is None
        assert g.answer_from_graph("luna", "age", "") is None

    def test_no_matching_predicate_returns_none(self):
        g = self._build_pet_graph()
        ans = g.answer_from_graph("luna", "salary", "What is Luna's salary?")
        assert ans is None

    def test_superseded_facts_ignored(self):
        """Only the current (non-superseded) fact is returned."""
        g = EntityGraph()
        g.ingest_triple("luna", "age", "2 years", turn_idx=0)
        g.ingest_triple("luna", "age", "3 years", turn_idx=1)
        ans = g.answer_from_graph("luna", "age", "How old is Luna?")
        assert ans == "3 years"

    def test_predicate_words_boost_matching(self):
        """The LLM-extracted predicate 'age' matches even though 'how old' doesn't."""
        g = EntityGraph()
        g.ingest_triple("luna", "age", "3 years", turn_idx=0)
        g.ingest_triple("luna", "color", "orange", turn_idx=0)
        ans = g.answer_from_graph("luna", "age", "How old is Luna?")
        assert ans == "3 years"

    def test_predicate_ranking_picks_best(self):
        """'brand' should beat 'purchased at' for 'What brand is the food bowl?'."""
        g = self._build_pet_graph()
        ans = g.answer_from_graph("food bowl", "brand", "What brand is the food bowl?")
        assert ans == "kong"
        # purchased_at should NOT be returned
        ans2 = g.answer_from_graph("food bowl", "purchased at", "Where was the food bowl purchased?")
        assert ans2 == "pet world"
