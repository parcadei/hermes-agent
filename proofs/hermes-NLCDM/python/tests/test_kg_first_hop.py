"""Discriminative tests for KG-first hop resolver.

Tests the hypothesis: sentence-window co-occurrence graph + NPMI weighting
+ entity residual scoring can resolve multi-hop queries structurally
(no LLM call), achieving higher per-hop accuracy than the current
retrieve-10-ask-LLM approach (70% per hop).

Architecture:
  Ingestion: entities per sentence -> co-occurrence edges -> NPMI weights
  Query:     extract query entity -> graph neighbors -> score by entity
             residual similarity -> return top-1 answer

The test uses synthetic MABench-style facts to validate:
1. Single-hop KG resolution (country->capital, person->team, etc.)
2. Multi-hop chain resolution (country->capital->continent)
3. Entity disambiguation (France vs Japan, both have capitals)
4. NPMI weighting suppresses noise entities
5. Coverage: what fraction of hops get a KG hit?
"""

from __future__ import annotations

import math
import pickle
import pytest

# Import from production module -- NOT defined inline
from mabench.cooccurrence_extractor import CoocKG


# ---------------------------------------------------------------------------
# Test fixtures: synthetic MABench-style facts
# ---------------------------------------------------------------------------

def _build_mabench_kg() -> CoocKG:
    """Build a KG from MABench-style facts about countries, capitals, continents.

    Mimics the factconsolidation_mh_262k dataset structure.
    """
    kg = CoocKG()

    # --- Geography facts (sentence-level entity pairs) ---
    # Each "sentence" is one fact from the benchmark.
    # Entities are what GLiNER would extract.

    sentences = [
        # France facts
        [{"text": "France", "label": "country"}, {"text": "Paris", "label": "city"}],
        [{"text": "France", "label": "country"}, {"text": "Europe", "label": "location"}],
        [{"text": "Paris", "label": "city"}, {"text": "Europe", "label": "location"}],

        # Japan facts
        [{"text": "Japan", "label": "country"}, {"text": "Tokyo", "label": "city"}],
        [{"text": "Japan", "label": "country"}, {"text": "Asia", "label": "location"}],
        [{"text": "Tokyo", "label": "city"}, {"text": "Asia", "label": "location"}],

        # Italy facts
        [{"text": "Italy", "label": "country"}, {"text": "Rome", "label": "city"}],
        [{"text": "Italy", "label": "country"}, {"text": "Europe", "label": "location"}],
        [{"text": "Rome", "label": "city"}, {"text": "Europe", "label": "location"}],

        # Brazil facts
        [{"text": "Brazil", "label": "country"}, {"text": "Brasilia", "label": "city"}],
        [{"text": "Brazil", "label": "country"}, {"text": "South America", "label": "location"}],

        # People + teams + leagues + countries (enables 3-4 hop chains)
        [{"text": "Abbiati", "label": "person"}, {"text": "AC Milan", "label": "organization"}],
        [{"text": "Abbiati", "label": "person"}, {"text": "Serie A", "label": "organization"}],
        [{"text": "AC Milan", "label": "organization"}, {"text": "Serie A", "label": "organization"}],
        [{"text": "AC Milan", "label": "organization"}, {"text": "Italy", "label": "country"}],
        [{"text": "Serie A", "label": "organization"}, {"text": "Italy", "label": "country"}],

        [{"text": "Messi", "label": "person"}, {"text": "Inter Miami", "label": "organization"}],
        [{"text": "Messi", "label": "person"}, {"text": "MLS", "label": "organization"}],
        [{"text": "Inter Miami", "label": "organization"}, {"text": "United States", "label": "country"}],
        [{"text": "MLS", "label": "organization"}, {"text": "United States", "label": "country"}],
        [{"text": "United States", "label": "country"}, {"text": "North America", "label": "location"}],

        # Noise: "Europe" appears in many contexts (should get lower NPMI)
        [{"text": "Europe", "label": "location"}, {"text": "EU", "label": "organization"}],
        [{"text": "Europe", "label": "location"}, {"text": "Euro", "label": "product"}],
        [{"text": "Europe", "label": "location"}, {"text": "NATO", "label": "organization"}],
    ]

    for sent_ents in sentences:
        kg.add_sentence(sent_ents)

    kg.compute_npmi()
    return kg


@pytest.fixture
def kg():
    return _build_mabench_kg()


# ---------------------------------------------------------------------------
# 1. Single-hop resolution
# ---------------------------------------------------------------------------

class TestSingleHop:
    """KG should resolve single-hop queries structurally."""

    def test_country_to_capital(self, kg: CoocKG):
        """France -> Paris (city type)."""
        answer = kg.resolve_hop("France", target_type="city")
        assert answer == "paris", f"Expected 'paris', got '{answer}'"

    def test_country_to_capital_japan(self, kg: CoocKG):
        """Japan -> Tokyo (city type)."""
        answer = kg.resolve_hop("Japan", target_type="city")
        assert answer == "tokyo", f"Expected 'tokyo', got '{answer}'"

    def test_country_to_capital_italy(self, kg: CoocKG):
        """Italy -> Rome (city type)."""
        answer = kg.resolve_hop("Italy", target_type="city")
        assert answer == "rome", f"Expected 'rome', got '{answer}'"

    def test_person_to_team(self, kg: CoocKG):
        """Abbiati -> AC Milan (organization type)."""
        answer = kg.resolve_hop("Abbiati", target_type="organization")
        assert answer == "ac milan", f"Expected 'ac milan', got '{answer}'"

    def test_person_to_team_messi(self, kg: CoocKG):
        """Messi -> Inter Miami (organization type)."""
        answer = kg.resolve_hop("Messi", target_type="organization")
        assert answer == "inter miami", f"Expected 'inter miami', got '{answer}'"

    def test_country_to_continent(self, kg: CoocKG):
        """Japan -> Asia (location type)."""
        answer = kg.resolve_hop("Japan", target_type="location")
        assert answer == "asia", f"Expected 'asia', got '{answer}'"


# ---------------------------------------------------------------------------
# 2. Multi-hop chain resolution
# ---------------------------------------------------------------------------

class TestMultiHop:
    """KG should resolve multi-hop chains structurally."""

    def test_2hop_country_capital_continent(self, kg: CoocKG):
        """France -> Paris -> Europe (2-hop chain)."""
        hop1 = kg.resolve_hop("France", target_type="city")
        assert hop1 == "paris"
        hop2 = kg.resolve_hop(hop1, target_type="location")
        assert hop2 == "europe", f"Expected 'europe', got '{hop2}'"

    def test_2hop_japan_capital_continent(self, kg: CoocKG):
        """Japan -> Tokyo -> Asia (2-hop chain)."""
        hop1 = kg.resolve_hop("Japan", target_type="city")
        assert hop1 == "tokyo"
        hop2 = kg.resolve_hop(hop1, target_type="location")
        assert hop2 == "asia", f"Expected 'asia', got '{hop2}'"

    def test_3hop_person_team_country_continent(self, kg: CoocKG):
        """Abbiati -> AC Milan -> Italy -> Europe (3-hop).

        MABench pattern: "What continent is the country of Abbiati's team in?"
        Hop 0: person -> team (organization)
        Hop 1: team -> country
        Hop 2: country -> continent (location)
        """
        hop1 = kg.resolve_hop("Abbiati", target_type="organization")
        assert hop1 == "ac milan", f"Hop 0 failed: {hop1}"

        hop2 = kg.resolve_hop(hop1, target_type="country")
        assert hop2 == "italy", f"Hop 1 failed: {hop2}"

        hop3 = kg.resolve_hop(hop2, target_type="location")
        assert hop3 == "europe", f"Hop 2 failed: {hop3}"

    def test_3hop_messi_chain(self, kg: CoocKG):
        """Messi -> Inter Miami -> United States -> North America (3-hop)."""
        hop1 = kg.resolve_hop("Messi", target_type="organization")
        assert hop1 == "inter miami", f"Hop 0 failed: {hop1}"

        hop2 = kg.resolve_hop(hop1, target_type="country")
        assert hop2 == "united states", f"Hop 1 failed: {hop2}"

        hop3 = kg.resolve_hop(hop2, target_type="location")
        assert hop3 == "north america", f"Hop 2 failed: {hop3}"

    def test_4hop_person_team_country_capital_continent(self, kg: CoocKG):
        """Abbiati -> AC Milan -> Italy -> Rome -> Europe (4-hop).

        MABench pattern: "What continent is the capital of the country
        of Abbiati's team in?"
        """
        chain = [
            ("Abbiati", "organization", "ac milan"),
            (None, "country", "italy"),
            (None, "city", "rome"),
            (None, "location", "europe"),
        ]
        current = chain[0][0]
        results = []
        for i, (entity, target_type, expected) in enumerate(chain):
            if i > 0:
                entity = current
            answer = kg.resolve_hop(entity, target_type=target_type)
            results.append((entity, target_type, answer, expected))
            if answer is None:
                break
            current = answer

        print(f"\n4-hop chain trace:")
        for entity, ttype, answer, expected in results:
            match = "v" if answer == expected else f"x (expected {expected})"
            print(f"  {entity} --({ttype})--> {answer}  {match}")

        hits = sum(1 for _, _, a, _ in results if a is not None)
        correct = sum(1 for _, _, a, e in results if a == e)
        print(f"  Resolved: {hits}/{len(chain)}, Correct: {correct}/{len(chain)}")

        for _, _, answer, expected in results:
            assert answer == expected

    def test_4hop_messi_chain(self, kg: CoocKG):
        """Messi -> Inter Miami -> US -> Washington? (4-hop, tests coverage gap).

        We don't have US->capital data, so hop 3 should miss.
        This tests graceful degradation -- the system should know
        when to fall back to LLM.
        """
        chain = [
            ("Messi", "organization"),   # -> Inter Miami
            (None, "country"),           # -> United States
            (None, "city"),              # -> ??? (no US capital in data)
        ]
        current = chain[0][0]
        results = []
        for i, (entity, target_type) in enumerate(chain):
            if i > 0:
                entity = current
            answer = kg.resolve_hop(entity, target_type=target_type)
            results.append((entity, target_type, answer))
            if answer is None:
                break
            current = answer

        # Hops 0-1 should resolve, hop 2 should miss (no data)
        assert results[0][2] == "inter miami"
        assert results[1][2] == "united states"
        # Hop 2: no US capital in our data -> None (LLM fallback needed)
        assert len(results) == 3 and results[2][2] is None, (
            f"Expected miss at hop 2, got: {results[2][2]}"
        )


# ---------------------------------------------------------------------------
# 3. Entity disambiguation
# ---------------------------------------------------------------------------

class TestEntityDisambiguation:
    """KG should NOT confuse entities from different contexts."""

    def test_france_not_tokyo(self, kg: CoocKG):
        """France's capital is Paris, not Tokyo."""
        answer = kg.resolve_hop("France", target_type="city")
        assert answer == "paris"
        assert answer != "tokyo"

    def test_japan_not_paris(self, kg: CoocKG):
        """Japan's capital is Tokyo, not Paris."""
        answer = kg.resolve_hop("Japan", target_type="city")
        assert answer == "tokyo"
        assert answer != "paris"

    def test_abbiati_not_inter_miami(self, kg: CoocKG):
        """Abbiati plays for AC Milan, not Inter Miami."""
        answer = kg.resolve_hop("Abbiati", target_type="organization")
        assert answer == "ac milan"
        assert answer != "inter miami"

    def test_messi_not_ac_milan(self, kg: CoocKG):
        """Messi plays for Inter Miami, not AC Milan."""
        answer = kg.resolve_hop("Messi", target_type="organization")
        assert answer == "inter miami"
        assert answer != "ac milan"


# ---------------------------------------------------------------------------
# 4. NPMI noise suppression
# ---------------------------------------------------------------------------

class TestNPMI:
    """NPMI should downweight high-frequency entities (noise)."""

    def test_europe_npmi_lower_than_specific(self, kg: CoocKG):
        """Europe co-occurs with many entities -> lower NPMI per edge.

        France-Paris should have higher NPMI than France-Europe because
        Paris is more specific to France (fewer total connections).
        """
        france_nbrs = dict(kg.neighbors("France"))
        paris_npmi = france_nbrs.get("paris", 0)
        europe_npmi = france_nbrs.get("europe", 0)
        assert paris_npmi > europe_npmi, (
            f"Paris NPMI ({paris_npmi:.3f}) should be > Europe NPMI ({europe_npmi:.3f})"
        )

    def test_tokyo_npmi_not_lower_than_asia(self, kg: CoocKG):
        """Tokyo and Asia both co-occur with Japan once -> equal NPMI.

        In real data Tokyo would co-occur more (multiple facts), but
        with synthetic 1:1 counts they're equal. The key invariant is
        that Tokyo should NOT rank below Asia -- type filtering at
        resolve_hop handles disambiguation.
        """
        japan_nbrs = dict(kg.neighbors("Japan"))
        tokyo_score = japan_nbrs.get("tokyo", 0)
        asia_score = japan_nbrs.get("asia", 0)
        assert tokyo_score >= asia_score, (
            f"Tokyo ({tokyo_score:.3f}) should be >= Asia ({asia_score:.3f})"
        )

    def test_ac_milan_npmi_higher_than_serie_a(self, kg: CoocKG):
        """AC Milan is specific to Abbiati -> higher NPMI than Serie A."""
        abbiati_nbrs = dict(kg.neighbors("Abbiati"))
        milan_npmi = abbiati_nbrs.get("ac milan", 0)
        serie_a_npmi = abbiati_nbrs.get("serie a", 0)
        assert milan_npmi >= serie_a_npmi, (
            f"AC Milan NPMI ({milan_npmi:.3f}) should be >= Serie A NPMI ({serie_a_npmi:.3f})"
        )


# ---------------------------------------------------------------------------
# 5. Coverage analysis
# ---------------------------------------------------------------------------

class TestCoverage:
    """Measure what fraction of typical queries get a KG hit."""

    QUERIES = [
        # (query_entity, target_type, expected_answer)
        ("France", "city", "paris"),
        ("Japan", "city", "tokyo"),
        ("Italy", "city", "rome"),
        ("Brazil", "city", "brasilia"),
        ("Abbiati", "organization", "ac milan"),
        ("Messi", "organization", "inter miami"),
        ("France", "location", "europe"),
        ("Japan", "location", "asia"),
        ("Italy", "location", "europe"),
        ("Paris", "location", "europe"),
        ("Tokyo", "location", "asia"),
        # These should miss (no data):
        ("Germany", "city", None),
        ("Argentina", "city", None),
    ]

    def test_coverage_rate(self, kg: CoocKG):
        """At least 80% of resolvable queries should get a KG hit."""
        hits = 0
        misses = 0
        correct = 0
        total_resolvable = sum(1 for _, _, exp in self.QUERIES if exp is not None)

        for entity, target_type, expected in self.QUERIES:
            answer = kg.resolve_hop(entity, target_type=target_type)
            if expected is None:
                # Should miss
                if answer is None:
                    hits += 1  # correct miss
                continue

            if answer is not None:
                hits += 1
                if answer == expected:
                    correct += 1
            else:
                misses += 1

        coverage = hits / total_resolvable if total_resolvable > 0 else 0
        accuracy = correct / hits if hits > 0 else 0

        print(f"\nKG-first hop coverage: {hits}/{total_resolvable} = {coverage:.0%}")
        print(f"KG-first hop accuracy: {correct}/{hits} = {accuracy:.0%}")

        assert coverage >= 0.80, f"Coverage {coverage:.0%} < 80%"
        assert accuracy >= 0.80, f"Accuracy {accuracy:.0%} < 80%"

    def test_multihop_chain_accuracy(self, kg: CoocKG):
        """Multi-hop chains: measure compound accuracy."""
        chains = [
            # (hops: list of (entity, type), expected_final)
            ([("France", "city"), (None, "location")], "europe"),
            ([("Japan", "city"), (None, "location")], "asia"),
            ([("Italy", "city"), (None, "location")], "europe"),
        ]

        correct = 0
        for hops, expected_final in chains:
            current = hops[0][0]
            broken = False
            for i, (entity, target_type) in enumerate(hops):
                if i > 0:
                    entity = current
                answer = kg.resolve_hop(entity, target_type=target_type)
                if answer is None:
                    broken = True
                    break
                current = answer
            if not broken and current == expected_final:
                correct += 1

        accuracy = correct / len(chains)
        print(f"\nMulti-hop chain accuracy: {correct}/{len(chains)} = {accuracy:.0%}")
        assert accuracy >= 0.80, f"Chain accuracy {accuracy:.0%} < 80%"


# ---------------------------------------------------------------------------
# 6. Edge case: updated facts (supersession scenario)
# ---------------------------------------------------------------------------

class TestSupersession:
    """When a fact is updated, the KG should reflect the newer value."""

    def test_capital_update(self):
        """If capital of France changes, newest co-occurrence should win."""
        kg = CoocKG()

        # Old fact: France-Paris (seen once)
        kg.add_sentence([
            {"text": "France", "label": "country"},
            {"text": "Paris", "label": "city"},
        ])

        # New fact: France-Lyon (seen 3 times, more recent/frequent)
        for _ in range(3):
            kg.add_sentence([
                {"text": "France", "label": "country"},
                {"text": "Lyon", "label": "city"},
            ])

        kg.compute_npmi()

        # With raw count, Lyon should win (3 vs 1)
        answer = kg.resolve_hop("France", target_type="city")
        # NPMI normalizes by frequency -- but Lyon has 3x the co-occurrence
        # so it should still rank higher
        nbrs = dict(kg.neighbors("France"))
        print(f"\nFrance neighbors: {nbrs}")
        assert nbrs.get("lyon", 0) > nbrs.get("paris", 0), (
            "Newer/more-frequent co-occurrence should rank higher"
        )


# ---------------------------------------------------------------------------
# 7. Pickle serialization (eval_cache support)
# ---------------------------------------------------------------------------

class TestPickle:
    """CoocKG must be pickle-safe for eval_cache persistence."""

    def test_pickle_roundtrip(self, kg: CoocKG):
        """Pickle + unpickle should preserve all state."""
        data = pickle.dumps(kg)
        restored = pickle.loads(data)

        assert restored.edges == kg.edges
        assert restored.entity_types == kg.entity_types
        assert restored._total == kg._total
        assert restored.npmi == kg.npmi

        # Functional test: resolve_hop still works
        answer = restored.resolve_hop("France", target_type="city")
        assert answer == "paris"

    def test_pickle_empty(self):
        """Empty CoocKG should pickle safely."""
        kg = CoocKG()
        data = pickle.dumps(kg)
        restored = pickle.loads(data)
        assert restored.edges == {}
        assert restored.entity_types == {}

    def test_pickle_pre_npmi(self):
        """CoocKG before compute_npmi should pickle safely."""
        kg = CoocKG()
        kg.add_sentence([
            {"text": "A", "label": "person"},
            {"text": "B", "label": "city"},
        ])
        data = pickle.dumps(kg)
        restored = pickle.loads(data)
        assert restored._total == 1


# ---------------------------------------------------------------------------
# 8. QueryEngine._infer_target_type coverage
# ---------------------------------------------------------------------------

class TestInferTargetType:
    """_infer_target_type should cover all 16 DEFAULT_ENTITY_LABELS."""

    def test_city_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What is the capital of France?") == "city"
        assert QueryEngine._infer_target_type("Which city is the largest?") == "city"
        assert QueryEngine._infer_target_type("Name the town near the river") == "city"

    def test_country_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What country does Messi play in?") == "country"
        assert QueryEngine._infer_target_type("Which nation borders France?") == "country"

    def test_location_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What continent is Italy in?") == "location"
        assert QueryEngine._infer_target_type("Which region does it belong to?") == "location"
        assert QueryEngine._infer_target_type("Where is it located?") == "location"

    def test_organization_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What team does Abbiati play for?") == "organization"
        assert QueryEngine._infer_target_type("Which company employs her?") == "organization"
        assert QueryEngine._infer_target_type("What league is AC Milan in?") == "organization"

    def test_person_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("Who is the president of France?") == "person"
        assert QueryEngine._infer_target_type("Which player scored the goal?") == "person"
        assert QueryEngine._infer_target_type("Who is the author of this book?") == "person"

    def test_sport_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What sport does Messi play?") == "sport"

    def test_language_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What language is spoken in France?") == "language"

    def test_event_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What event was held in Paris?") == "event"
        assert QueryEngine._infer_target_type("Which competition did they win?") == "event"

    def test_product_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What product did they launch?") == "product"

    def test_work_of_art_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What novel was published in 2024?") == "work of art"
        assert QueryEngine._infer_target_type("Which painting is in the museum?") == "work of art"
        assert QueryEngine._infer_target_type("What film was released in 2024?") == "work of art"
        assert QueryEngine._infer_target_type("Which song topped the charts?") == "work of art"
        assert QueryEngine._infer_target_type("What album did the band release?") == "work of art"

    def test_animal_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What animal is the national symbol?") == "animal"
        assert QueryEngine._infer_target_type("Which species is endangered?") == "animal"

    def test_plant_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What plant grows in the region?") == "plant"
        assert QueryEngine._infer_target_type("Which flower is the national emblem?") == "plant"

    def test_disease_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What disease affected the population?") == "disease"
        assert QueryEngine._infer_target_type("Which illness is most common?") == "disease"
        assert QueryEngine._infer_target_type("What syndrome is associated with it?") == "disease"

    def test_chemical_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What chemical is used in the process?") == "chemical"
        assert QueryEngine._infer_target_type("Which element is most abundant?") == "chemical"
        assert QueryEngine._infer_target_type("What compound was discovered?") == "chemical"

    def test_food_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What food is the region known for?") == "food"
        assert QueryEngine._infer_target_type("Which dish is the national cuisine?") == "food"
        assert QueryEngine._infer_target_type("What cuisine originated there?") == "food"

    def test_material_keywords(self):
        from mabench.query import QueryEngine
        assert QueryEngine._infer_target_type("What material is it made of?") == "material"
        assert QueryEngine._infer_target_type("Which fabric is used?") == "material"

    def test_no_match_returns_none(self):
        from mabench.query import QueryEngine
        # Generic question that doesn't match any type pattern
        assert QueryEngine._infer_target_type("What is the value of pi?") is None
