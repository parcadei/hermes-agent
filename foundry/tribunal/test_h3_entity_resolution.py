"""H3 Discriminative Tests: Entity resolution via substring containment + embedding similarity.

Hypothesis: "Entity resolution via substring containment + embedding similarity is
sufficient to correctly link mentions to entities."

7 measurements designed to falsify or validate H3. Each measurement class is
self-contained with its own fixtures. Tests produce MEASUREMENT: lines that
can be grep'd from pytest output for tabular reporting.

Run:
    /Users/cosimo/.hermes/hermes-agent/venv/bin/python3 -m pytest foundry/tribunal/test_h3_entity_resolution.py -v -s 2>&1 | grep MEASUREMENT
"""
from __future__ import annotations

import random

import pytest

from mabench.entity_graph import EntityGraph, EntityNode, Fact


# ===========================================================================
# M1: Substring False Merge Rate
# ===========================================================================


class TestSubstringFalseMergeRate:
    """M1: Of N entity pairs sharing substrings, how many does resolve() incorrectly merge?"""

    SHOULD_NOT_MERGE = [
        ("paris", "paris hilton"),
        ("java", "javascript"),
        ("ham", "hamilton"),
        ("art", "arthur"),
        ("car", "carbon"),
        ("cat", "catalog"),
        ("pan", "pandemic"),
        ("ion", "lion"),
        ("la", "plane"),
        ("mat", "material"),
        ("ace", "palace"),
        ("age", "average"),
        ("ant", "pleasant"),
        ("ash", "fashion"),
        ("bat", "batch"),
        ("bay", "ebay"),
        ("ben", "benefit"),
        ("don", "london"),
        ("ink", "thinking"),
        ("log", "blog"),
    ]

    SHOULD_MERGE = [
        ("roscioli", "roscioli bakery"),
        ("golden retriever", "retriever"),
        ("food bowl", "expensive food bowl"),
    ]

    def test_false_merge_count(self):
        """Each pair: create entity A, then resolve(B). Count false merges."""
        false_merges = 0
        for existing, query in self.SHOULD_NOT_MERGE:
            g = EntityGraph()
            g.ingest_triple(existing, "is_a", "thing", turn_idx=0)
            result = g.resolve(query)
            if result is not None and result == g._normalize(existing):
                false_merges += 1
        rate = false_merges / len(self.SHOULD_NOT_MERGE)
        print(f"\nMEASUREMENT: M1_substring_false_merge_rate = {false_merges}/{len(self.SHOULD_NOT_MERGE)} = {rate:.1%}")
        # This is a measurement -- the value is what matters
        assert false_merges >= 0  # always passes

    @pytest.mark.parametrize("existing,query", SHOULD_NOT_MERGE, ids=[f"{e}->{q}" for e, q in SHOULD_NOT_MERGE])
    def test_individual_false_merge(self, existing, query):
        """Document per-pair: does resolve() falsely merge?"""
        g = EntityGraph()
        g.ingest_triple(existing, "is_a", "thing", turn_idx=0)
        result = g.resolve(query)
        existing_norm = g._normalize(existing)
        # Record whether this pair falsely merges
        is_false_merge = (result is not None and result == existing_norm)
        print(f"\n  resolve('{query}') with entity '{existing}': {'FALSE MERGE -> ' + repr(result) if is_false_merge else 'CORRECT: ' + repr(result)}")

    def test_correct_merges_preserved(self):
        """Legitimate substring merges must still work."""
        for existing, query in self.SHOULD_MERGE:
            g = EntityGraph()
            g.ingest_triple(existing, "is_a", "thing", turn_idx=0)
            result = g.resolve(query)
            assert result == g._normalize(existing), (
                f"resolve('{query}') should merge with '{existing}' but got {result!r}"
            )

    def test_resolve_exact_avoids_all_false_merges(self):
        """resolve_exact() must return None for all SHOULD_NOT_MERGE pairs."""
        avoided = 0
        for existing, query in self.SHOULD_NOT_MERGE:
            g = EntityGraph()
            g.ingest_triple(existing, "is_a", "thing", turn_idx=0)
            result = g.resolve_exact(query)
            if result is None:
                avoided += 1
        rate = avoided / len(self.SHOULD_NOT_MERGE)
        print(f"\nMEASUREMENT: M1_resolve_exact_avoidance_rate = {avoided}/{len(self.SHOULD_NOT_MERGE)} = {rate:.1%}")
        assert avoided == len(self.SHOULD_NOT_MERGE), (
            f"resolve_exact() should avoid ALL false merges but missed {len(self.SHOULD_NOT_MERGE) - avoided}"
        )


# ===========================================================================
# M2: Deictic and Pronominal Reference Resolution
# ===========================================================================


class TestDeicticReferenceResolution:
    """M2: Can resolve() handle references sharing no substring with the canonical name?"""

    @staticmethod
    def _build_graph():
        g = EntityGraph()
        g.ingest_triple("roscioli", "is_a", "restaurant", turn_idx=0)
        g.ingest_triple("roscioli", "location", "rome", turn_idx=0)
        g.ingest_triple("user", "visited", "roscioli", turn_idx=1)
        g.ingest_triple("dr. chen", "is_a", "doctor", turn_idx=2)
        g.ingest_triple("user", "saw", "dr. chen", turn_idx=3)
        g.ingest_triple("blue bottle coffee", "is_a", "coffee shop", turn_idx=4)
        g.ingest_triple("user", "goes_to", "blue bottle coffee", turn_idx=5)
        return g

    DEICTIC_REFERENCES = [
        ("that restaurant", "roscioli"),
        ("the place we went in rome", "roscioli"),
        ("my doctor", "dr. chen"),
        ("the doctor I saw", "dr. chen"),
        ("that coffee place", "blue bottle coffee"),
        ("my usual spot", "blue bottle coffee"),
        ("the place I go to", "blue bottle coffee"),
        ("that Italian restaurant", "roscioli"),
    ]

    def test_deictic_resolve_without_embeddings(self):
        """Without embeddings, resolve() relies on substring/alias only."""
        g = self._build_graph()
        resolved = 0
        for ref, expected in self.DEICTIC_REFERENCES:
            result = g.resolve(ref)
            if result == expected:
                resolved += 1
            else:
                print(f"\n  resolve('{ref}'): got {result!r}, wanted '{expected}'")
        rate = resolved / len(self.DEICTIC_REFERENCES)
        print(f"\nMEASUREMENT: M2_deictic_no_embedding = {resolved}/{len(self.DEICTIC_REFERENCES)} = {rate:.1%}")

    def test_deictic_via_entity_store_bm25(self):
        """EntityStore BM25 can match type tokens (restaurant, doctor, coffee)."""
        from mabench.entity_store import EntityStore

        g = self._build_graph()
        store = EntityStore.from_graph(g)

        resolved = 0
        for ref, expected in self.DEICTIC_REFERENCES:
            results = store.search(ref, top_k=3)
            found = any(r[0].canonical == expected for r in results) if results else False
            if found:
                resolved += 1
            else:
                top = results[0][0].canonical if results else "EMPTY"
                print(f"\n  store.search('{ref}'): top={top}, wanted '{expected}'")
        rate = resolved / len(self.DEICTIC_REFERENCES)
        print(f"\nMEASUREMENT: M2_deictic_entity_store_bm25 = {resolved}/{len(self.DEICTIC_REFERENCES)} = {rate:.1%}")

    def test_deictic_resolve_with_embeddings(self):
        """With embeddings, can similarity bridge the gap?"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence_transformers not installed")

        model = SentenceTransformer("all-MiniLM-L6-v2")
        g = self._build_graph()
        g.set_embedder(lambda texts: model.encode(texts), sim_threshold=0.5)
        g.embed_all_entities()

        resolved = 0
        for ref, expected in self.DEICTIC_REFERENCES:
            result = g.resolve(ref)
            if result == expected:
                resolved += 1
            else:
                # Try find_similar as secondary path
                sims = g.find_similar(ref, threshold=0.3, top_k=3)
                if sims and sims[0][0] == expected:
                    resolved += 1
                else:
                    top_sim = sims[0] if sims else None
                    print(f"\n  resolve+find_similar('{ref}'): resolve={result!r}, top_sim={top_sim}, wanted '{expected}'")
        rate = resolved / len(self.DEICTIC_REFERENCES)
        print(f"\nMEASUREMENT: M2_deictic_with_embedding = {resolved}/{len(self.DEICTIC_REFERENCES)} = {rate:.1%}")


# ===========================================================================
# M3: First-Mention Accuracy (False Merge on Novel Entities)
# ===========================================================================


class TestFirstMentionAccuracy:
    """M3: For genuinely new entities, does _resolve_or_create() create a new node
    or falsely merge with an existing one?"""

    DISTINCT_ENTITIES = [
        "luna",
        "luna park",
        "blue bottle",
        "blue cross",
        "golden retriever",
        "golden gate bridge",
        "java",
        "java island",
        "paris",
        "paris hilton",
        "green tea",
        "green bay",
        "black mirror",
        "black coffee",
        "hot dog",
        "hot springs",
    ]

    def test_progressive_entity_accumulation(self):
        """Ingest entities one by one. Track false merges with prior entities."""
        g = EntityGraph()
        false_merges = 0
        merge_log = []

        for entity in self.DISTINCT_ENTITIES:
            canon = g._resolve_or_create(entity)
            norm = g._normalize(entity)
            stripped = g._normalize(g._strip_determiners(norm))
            expected = stripped if stripped else norm
            if canon != expected:
                false_merges += 1
                merge_log.append(f"  '{entity}' -> merged into '{canon}' (expected new '{expected}')")

        rate = false_merges / len(self.DISTINCT_ENTITIES)
        print(f"\nMEASUREMENT: M3_first_mention_false_merge = {false_merges}/{len(self.DISTINCT_ENTITIES)} = {rate:.1%}")
        for line in merge_log:
            print(line)
        print(f"  Total entities created: {len(g.entities)} (expected {len(self.DISTINCT_ENTITIES)})")

    def test_type_gated_first_mention(self):
        """With hint_type, does _resolve_or_create prevent some false merges?"""
        TYPED = [
            ("luna", "pet"),
            ("luna park", "place"),
            ("java", "programming_language"),
            ("java island", "place"),
            ("paris", "city"),
            ("paris hilton", "person"),
            ("green tea", "beverage"),
            ("green bay", "city"),
        ]

        g = EntityGraph()
        false_merges = 0
        merge_log = []

        for entity, etype in TYPED:
            canon = g._resolve_or_create(entity, hint_type=etype)
            g.set_entity_type(canon, etype)
            norm = g._normalize(entity)
            stripped = g._normalize(g._strip_determiners(norm))
            expected = stripped if stripped else norm
            if canon != expected:
                false_merges += 1
                merge_log.append(f"  '{entity}' (type={etype}) -> merged into '{canon}'")

        rate = false_merges / len(TYPED)
        print(f"\nMEASUREMENT: M3_type_gated_first_mention = {false_merges}/{len(TYPED)} = {rate:.1%}")
        for line in merge_log:
            print(line)


# ===========================================================================
# M4: Alias Coverage (alias index vs substring vs miss)
# ===========================================================================


class TestAliasCoverage:
    """M4: What fraction of entity mentions are resolvable via alias index alone
    (resolve_exact) vs needing substring fallback (resolve)?"""

    def _build_diverse_graph(self):
        """Build a graph with diverse entities to test coverage."""
        g = EntityGraph()
        triples = [
            ("delta airlines", "is_a", "airline", 0),
            ("user", "flies", "delta airlines", 1),
            ("luna", "is_a", "cat", 2),
            ("roscioli", "is_a", "restaurant", 3),
            ("dr. chen", "is_a", "doctor", 4),
            ("blue bottle coffee", "is_a", "coffee shop", 5),
            ("trek domane", "is_a", "bike", 6),
            ("golden gate park", "is_a", "park", 7),
            ("stanford university", "is_a", "university", 8),
            ("macbook pro", "is_a", "laptop", 9),
            ("spotify", "is_a", "music service", 10),
            ("python", "is_a", "programming language", 11),
        ]
        for s, p, o, t in triples:
            g.ingest_triple(s, p, o, turn_idx=t)
        g.register_alias("delta", "delta airlines")
        g.register_alias("my cat", "luna")
        g.register_alias("the coffee place", "blue bottle coffee")
        return g

    # Mentions drawn from realistic query patterns
    TEST_MENTIONS = [
        # Should hit alias index
        "delta airlines",
        "delta",
        "luna",
        "my cat",
        "roscioli",
        "the coffee place",
        # Should need substring (if at all)
        "the airline",
        "that restaurant in rome",
        "dr chen",
        "stanford",
        "macbook",
        # Should miss entirely
        "my dog",
        "the gym",
        "that movie we watched",
        "his favorite band",
    ]

    def test_alias_vs_substring_vs_miss(self):
        """Categorize each mention: alias hit, substring-only hit, or miss."""
        g = self._build_diverse_graph()
        alias_hits = 0
        substring_only = 0
        misses = 0

        for mention in self.TEST_MENTIONS:
            exact = g.resolve_exact(mention)
            full = g.resolve(mention)
            if exact is not None:
                alias_hits += 1
            elif full is not None:
                substring_only += 1
            else:
                misses += 1

        total = len(self.TEST_MENTIONS)
        print(f"\nMEASUREMENT: M4_alias_coverage")
        print(f"  alias_index_hits:  {alias_hits}/{total} = {alias_hits/total:.1%}")
        print(f"  substring_only:    {substring_only}/{total} = {substring_only/total:.1%}")
        print(f"  total_misses:      {misses}/{total} = {misses/total:.1%}")


# ===========================================================================
# M5: EntityStore (BM25) vs EntityGraph (Substring) by Question Type
# ===========================================================================


class TestStoreVsGraphByQuestionType:
    """M5: Per question type, does EntityStore.search() or EntityGraph.resolve()
    find the target entity more reliably?"""

    FIXTURE_TRIPLES = [
        ("delta airlines", "is_a", "airline", 0),
        ("user", "flies", "delta airlines", 1),
        ("paris trip", "date", "june 2025", 2),
        ("user", "visited", "paris", 3),
        ("trek domane", "is_a", "bike", 4),
        ("specialized roubaix", "is_a", "bike", 5),
        ("cannondale synapse", "is_a", "bike", 6),
        ("blue bottle coffee", "is_a", "coffee shop", 7),
        ("user", "prefers", "blue bottle coffee", 8),
        ("dr. chen", "works_at", "stanford medical", 9),
        ("stanford medical", "location", "palo alto", 10),
    ]

    QUERIES_BY_TYPE = {
        "factual_direct": [
            ("delta airlines", "delta airlines"),
            ("trek domane", "trek domane"),
        ],
        "factual_indirect": [
            ("the airline for valentines trip", "delta airlines"),
            ("that coffee place the user likes", "blue bottle coffee"),
        ],
        "aggregation": [
            ("bikes the user owns", "trek domane"),
            ("all the bicycles", "trek domane"),
        ],
        "temporal": [
            ("paris trip date", "paris trip"),
        ],
        "multi_hop": [
            ("doctor location palo alto", "stanford medical"),
        ],
    }

    def test_store_vs_graph_per_type(self):
        """Compare EntityStore.search() top-1 vs EntityGraph.resolve() per query type."""
        from mabench.entity_store import EntityStore

        g = EntityGraph()
        for s, p, o, t in self.FIXTURE_TRIPLES:
            g.ingest_triple(s, p, o, turn_idx=t)
        store = EntityStore.from_graph(g)

        print(f"\nMEASUREMENT: M5_store_vs_graph")
        for qtype, queries in self.QUERIES_BY_TYPE.items():
            store_hits = 0
            graph_hits = 0
            for query, target in queries:
                # EntityStore path (BM25)
                results = store.search(query, top_k=5)
                store_found = any(r[0].canonical == g._normalize(target) for r in results) if results else False
                if store_found:
                    store_hits += 1

                # EntityGraph path (resolve via extracted entities)
                cands = g._extract_question_entities(query)
                graph_found = False
                for cand in cands:
                    canon = g.resolve(cand)
                    if canon == g._normalize(target):
                        graph_found = True
                        break
                if graph_found:
                    graph_hits += 1

            total = len(queries)
            print(f"  {qtype}: store={store_hits}/{total} graph={graph_hits}/{total}")


# ===========================================================================
# M6: Homonym Disambiguation
# ===========================================================================


class TestHomonymDisambiguation:
    """M6: When multiple real-world entities share the same name, does the system
    merge them into one node or keep them separate?"""

    HOMONYM_SCENARIOS = [
        {
            "name": "java_language_vs_island",
            "first": [("java", "is_a", "programming language", 0)],
            "second": [("java", "is_a", "island", 5)],
        },
        {
            "name": "bank_financial_vs_river",
            "first": [("bank", "balance", "$5000", 0)],
            "second": [("bank", "location", "river thames", 5)],
        },
        {
            "name": "mercury_planet_vs_element",
            "first": [("mercury", "is_a", "planet", 0)],
            "second": [("mercury", "is_a", "element", 5)],
        },
        {
            "name": "apple_fruit_vs_company",
            "first": [("apple", "color", "red", 0)],
            "second": [("apple", "stock_price", "$150", 5)],
        },
        {
            "name": "bass_fish_vs_instrument",
            "first": [("bass", "is_a", "fish", 0)],
            "second": [("bass", "is_a", "instrument", 5)],
        },
        {
            "name": "spring_season_vs_water",
            "first": [("spring", "month", "march", 0)],
            "second": [("spring", "location", "yellowstone", 5)],
        },
    ]

    def test_homonym_merge_rate(self):
        """For each homonym, ingest both contexts and check if merged."""
        merged_count = 0
        for scenario in self.HOMONYM_SCENARIOS:
            g = EntityGraph()
            for s, p, o, t in scenario["first"]:
                g.ingest_triple(s, p, o, turn_idx=t)
            first_canon = g.resolve(scenario["first"][0][0])

            for s, p, o, t in scenario["second"]:
                g.ingest_triple(s, p, o, turn_idx=t)
            second_canon = g.resolve(scenario["second"][0][0])

            is_merged = (first_canon == second_canon)
            if is_merged:
                merged_count += 1
                node = g.entities[first_canon]
                fact_count = sum(len(fl) for fl in node.facts.values())
                print(f"\n  {scenario['name']}: MERGED into '{first_canon}' ({fact_count} facts, contaminated)")
            else:
                print(f"\n  {scenario['name']}: SEPARATED '{first_canon}' vs '{second_canon}'")

        total = len(self.HOMONYM_SCENARIOS)
        rate = merged_count / total
        print(f"\nMEASUREMENT: M6_homonym_merge_rate = {merged_count}/{total} = {rate:.1%}")

    def test_type_gate_on_exact_match_bypass(self):
        """Verify: _resolve_typed() type gate does NOT fire on exact alias matches.
        This is the architectural gap -- the type gate only guards the substring path."""
        g = EntityGraph()
        g.ingest_triple("java", "is_a", "programming language", turn_idx=0)
        g.set_entity_type("java", "programming_language")

        # _resolve_typed with a DIFFERENT type hint -- but "java" is an exact alias match
        result = g._resolve_typed("java", "island")
        # Expected: returns "java" because exact match in alias_index (step 1)
        # The type gate (step 3) never fires
        print(f"\nMEASUREMENT: M6_type_gate_bypass: _resolve_typed('java', 'island') = {result!r}")
        print(f"  Type gate fires: {result is None}")
        # Document the behavior: type gate does NOT prevent exact-match homonym merges
        assert result is not None, (
            "If this fails, the type gate now fires on exact matches -- behavior changed"
        )

    def test_homonym_fact_contamination(self):
        """After merging homonyms, verify contradictory facts coexist on one node."""
        g = EntityGraph()
        g.ingest_triple("mercury", "is_a", "planet", turn_idx=0)
        g.ingest_triple("mercury", "orbital_period", "88 days", turn_idx=1)
        g.ingest_triple("mercury", "is_a", "chemical element", turn_idx=5)
        g.ingest_triple("mercury", "atomic_number", "80", turn_idx=6)

        node = g.query_entity("mercury")
        assert node is not None

        # Both "planet" and "chemical element" are on the same node
        is_a_facts = [f.object for f in node.facts.get("is_a", [])]
        has_planet = "planet" in is_a_facts
        has_element = "chemical element" in is_a_facts
        has_orbit = "orbital_period" in node.facts
        has_atomic = "atomic_number" in node.facts

        print(f"\nMEASUREMENT: M6_fact_contamination")
        print(f"  is_a values: {is_a_facts}")
        print(f"  Has 'planet': {has_planet}, Has 'element': {has_element}")
        print(f"  Has orbital_period: {has_orbit}, Has atomic_number: {has_atomic}")
        print(f"  Contaminated: {has_planet and has_element}")
        assert has_planet and has_element, "Both senses should be on the same node (documenting the bug)"


# ===========================================================================
# M7: Ingestion-Order Sensitivity
# ===========================================================================


class TestIngestionOrderSensitivity:
    """M7: Does ingestion order change which entities get merged?"""

    TRIPLES = [
        ("paris", "is_a", "city", 0),
        ("paris hilton", "is_a", "celebrity", 1),
        ("java", "is_a", "language", 2),
        ("javascript", "is_a", "language", 3),
        ("golden retriever", "is_a", "dog", 4),
        ("golden gate bridge", "is_a", "landmark", 5),
        ("ham", "is_a", "food", 6),
        ("hamilton", "is_a", "city", 7),
    ]

    def test_order_changes_entity_count(self):
        """Same triples, different order -> different entity counts."""
        # Order A: as listed
        g_a = EntityGraph()
        for s, p, o, t in self.TRIPLES:
            g_a.ingest_triple(s, p, o, turn_idx=t)

        # Order B: reversed
        g_b = EntityGraph()
        for s, p, o, t in reversed(self.TRIPLES):
            g_b.ingest_triple(s, p, o, turn_idx=t)

        # Order C: shuffled deterministically
        random.seed(42)
        shuffled = list(self.TRIPLES)
        random.shuffle(shuffled)
        g_c = EntityGraph()
        for s, p, o, t in shuffled:
            g_c.ingest_triple(s, p, o, turn_idx=t)

        counts = (len(g_a.entities), len(g_b.entities), len(g_c.entities))
        is_deterministic = counts[0] == counts[1] == counts[2]

        print(f"\nMEASUREMENT: M7_ingestion_order_sensitivity")
        print(f"  Order A (sequential): {counts[0]} entities, keys={sorted(g_a.entities.keys())}")
        print(f"  Order B (reversed):   {counts[1]} entities, keys={sorted(g_b.entities.keys())}")
        print(f"  Order C (shuffled):   {counts[2]} entities, keys={sorted(g_c.entities.keys())}")
        print(f"  Deterministic: {is_deterministic}")

    def test_order_changes_canonical_name(self):
        """The canonical name for a merged pair depends on which was ingested first."""
        # Order 1: "paris" first
        g1 = EntityGraph()
        g1.ingest_triple("paris", "is_a", "city", turn_idx=0)
        g1.ingest_triple("paris hilton", "is_a", "celebrity", turn_idx=1)
        canon_1 = g1.resolve("paris hilton")

        # Order 2: "paris hilton" first
        g2 = EntityGraph()
        g2.ingest_triple("paris hilton", "is_a", "celebrity", turn_idx=0)
        g2.ingest_triple("paris", "is_a", "city", turn_idx=1)
        canon_2 = g2.resolve("paris")

        print(f"\nMEASUREMENT: M7_canonical_name_order_dependence")
        print(f"  Paris first:   resolve('paris hilton') = '{canon_1}'")
        print(f"  Hilton first:  resolve('paris')        = '{canon_2}'")
        print(f"  Same canonical: {canon_1 == canon_2}")
        # The canonical name changes based on order -- this proves non-determinism

    def test_fact_assignment_order_dependence(self):
        """Facts get assigned to different canonical entities based on order."""
        # Order 1: "java" first -> "javascript" merges into "java"
        g1 = EntityGraph()
        g1.ingest_triple("java", "type", "compiled", turn_idx=0)
        g1.ingest_triple("javascript", "type", "interpreted", turn_idx=1)
        java_facts_1 = g1.query_facts("java", current_only=False)

        # Order 2: "javascript" first -> "java" merges into "javascript"
        g2 = EntityGraph()
        g2.ingest_triple("javascript", "type", "interpreted", turn_idx=0)
        g2.ingest_triple("java", "type", "compiled", turn_idx=1)
        # In order 2, "java" may resolve to "javascript" or create separate
        java_facts_2 = g2.query_facts("java", current_only=False)
        js_facts_2 = g2.query_facts("javascript", current_only=False)

        print(f"\nMEASUREMENT: M7_fact_assignment")
        print(f"  Order 1 (java first):  query_facts('java') = {len(java_facts_1)} facts")
        print(f"  Order 2 (js first):    query_facts('java') = {len(java_facts_2)} facts, "
              f"query_facts('javascript') = {len(js_facts_2)} facts")
