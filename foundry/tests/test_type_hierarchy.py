"""Type hierarchy and mention bridge coverage for passive consolidation."""

from __future__ import annotations

import math

from foundry.query import QueryPlan
from foundry.storage import EntityNode, FoundryStore, MentionBridgeIndex, TypeIndex


def _add_entity(store: FoundryStore, canonical: str, *type_labels: str) -> None:
    entity = EntityNode(canonical=canonical, types=set(type_labels))
    store.entity_graph.add_entity(entity)
    for type_label in type_labels:
        store.types.index(canonical, type_label)


def _build_counting_store() -> FoundryStore:
    store = FoundryStore()

    _add_entity(store, "fender_strat", "item")
    _add_entity(store, "yamaha_fg800", "item")
    _add_entity(store, "pearl_export_drum_set", "item")
    _add_entity(store, "korg_b1", "item")
    _add_entity(store, "primary_care_physician", "primary_care_physician")
    _add_entity(store, "ent_specialist", "ent_specialist")
    _add_entity(store, "dermatologist", "dermatologist")
    _add_entity(store, "kitchen_faucet", "kitchen_faucet")
    _add_entity(store, "kitchen_mat", "kitchen_mat")
    _add_entity(store, "toaster", "toaster")
    _add_entity(store, "coffee_maker", "coffee_maker")
    _add_entity(store, "kitchen_shelves", "kitchen_shelves")
    _add_entity(store, "office_chair", "item")

    instrument_links = (
        ("instrument", "guitar", 1.0),
        ("instrument", "piano", 1.0),
        ("instrument", "drum_set", 1.0),
        ("guitar", "electric_guitar", 1.0),
        ("guitar", "acoustic_guitar", 1.0),
        ("piano", "digital_piano", 1.0),
    )
    for parent, child, coverage in instrument_links:
        store.types.add_subsumption(parent, child, coverage)
    store.mention_bridges.add("fender strat", {"electric guitar"})
    store.mention_bridges.add("yamaha fg800", {"acoustic guitar"})
    store.mention_bridges.add("pearl export drum set", {"drum set", "drums"})
    store.mention_bridges.add("korg b1", {"digital piano", "keyboard"})

    doctor_links = (
        ("doctor", "physician", 1.0),
        ("doctor", "specialist", 1.0),
        ("physician", "primary_care_physician", 1.0),
        ("specialist", "ent_specialist", 1.0),
        ("specialist", "dermatologist", 1.0),
    )
    for parent, child, coverage in doctor_links:
        store.types.add_subsumption(parent, child, coverage)
    store.mention_bridges.add("primary care physician", {"primary care physician", "physician"})
    store.mention_bridges.add("ent specialist", {"ent specialist", "specialist"})
    store.mention_bridges.add("dermatologist", {"dermatologist", "specialist"})

    kitchen_links = (
        ("kitchen_item", "kitchen_appliance", 1.0),
        ("kitchen_item", "kitchen_fixture", 1.0),
        ("kitchen_item", "kitchen_storage", 1.0),
        ("kitchen_item", "kitchen_accessory", 1.0),
        ("kitchen_appliance", "toaster", 1.0),
        ("kitchen_appliance", "coffee_maker", 1.0),
        ("kitchen_fixture", "kitchen_faucet", 1.0),
        ("kitchen_storage", "kitchen_shelves", 1.0),
        ("kitchen_accessory", "kitchen_mat", 1.0),
    )
    for parent, child, coverage in kitchen_links:
        store.types.add_subsumption(parent, child, coverage)
    store.mention_bridges.add("kitchen faucet", {"kitchen faucet", "faucet"})
    store.mention_bridges.add("kitchen mat", {"kitchen mat", "kitchen accessory"})
    store.mention_bridges.add("toaster", {"toaster", "kitchen appliance"})
    store.mention_bridges.add("coffee maker", {"coffee maker", "kitchen appliance"})
    store.mention_bridges.add("kitchen shelves", {"kitchen shelves", "shelves", "kitchen storage"})

    return store


class TestTypeIndexHierarchy:
    def test_add_subsumption_and_descendants(self) -> None:
        idx = TypeIndex()
        idx.add_subsumption("guitar", "fender_strat", 1.0)
        idx.add_subsumption("guitar", "yamaha", 0.9)

        assert idx.descendants("guitar") == {"guitar", "fender_strat", "yamaha"}

    def test_descendants_transitive(self) -> None:
        idx = TypeIndex()
        idx.add_subsumption("instrument", "guitar", 1.0)
        idx.add_subsumption("guitar", "fender_strat", 0.8)

        assert idx.descendants("instrument") == {"instrument", "guitar", "fender_strat"}

    def test_descendants_handles_cycles(self) -> None:
        idx = TypeIndex()
        idx.add_subsumption("A", "B", 1.0)
        idx.add_subsumption("B", "A", 1.0)

        assert idx.descendants("A") == {"a", "b"}

    def test_ancestors(self) -> None:
        idx = TypeIndex()
        idx.add_subsumption("instrument", "guitar", 1.0)
        idx.add_subsumption("guitar", "fender_strat", 1.0)

        assert idx.ancestors("fender_strat") == ["guitar", "instrument"]

    def test_specificity(self) -> None:
        idx = TypeIndex()
        idx.index("fender_strat", "instrument")
        idx.index("fender_strat", "instrument")
        idx.index("fender_strat", "guitar")
        idx.index("fender_strat", "electric_guitar")
        idx.index("yamaha_fg800", "instrument")
        idx.index("yamaha_fg800", "guitar")
        idx.index("korg_b1", "instrument")

        instrument_score = idx.specificity("fender_strat", "instrument")
        guitar_score = idx.specificity("fender_strat", "guitar")
        electric_score = idx.specificity("fender_strat", "electric_guitar")

        assert instrument_score == 0.0
        assert guitar_score > instrument_score
        assert electric_score > guitar_score
        assert math.isclose(electric_score, 0.25 * math.log(3.0), rel_tol=1e-6)

    def test_type_index_serialization(self) -> None:
        idx = TypeIndex()
        idx.index("fender_strat", "electric_guitar")
        idx.add_subsumption("instrument", "guitar", 1.0)
        idx.add_subsumption("guitar", "electric_guitar", 0.85)

        restored = TypeIndex.from_dict(idx.to_dict())

        assert restored.lookup("electric_guitar") == ["fender_strat"]
        assert restored.descendants("instrument") == {"instrument", "guitar", "electric_guitar"}
        assert restored._subsumption_coverage[("guitar", "electric_guitar")] == 0.85


class TestMentionBridgeIndex:
    def test_add_and_expand(self) -> None:
        idx = MentionBridgeIndex()
        idx.add("korg b1", {"digital piano", "keyboard"})

        expanded = idx.expand({"digital piano"})

        assert "korg b1" in expanded

    def test_bidirectional(self) -> None:
        idx = MentionBridgeIndex()
        idx.add("korg b1", {"digital piano"})

        assert idx.linked_to("korg b1") == {"digital piano"}
        assert idx.linked_to("digital piano") == {"korg b1"}

    def test_expand_chain(self) -> None:
        idx = MentionBridgeIndex()
        idx.add("korg b1", {"digital piano", "keyboard"})

        assert idx.expand({"digital piano"}) == {"digital piano", "korg b1"}
        assert idx.expand({"korg b1"}) == {"korg b1", "digital piano", "keyboard"}

    def test_skips_empty_and_question_mark(self) -> None:
        idx = MentionBridgeIndex()
        idx.add("x", {"", "?"})

        assert idx.linked_to("x") == set()
        assert idx.expand({"x"}) == {"x"}

    def test_serialization(self) -> None:
        idx = MentionBridgeIndex()
        idx.add("korg b1", {"digital piano", "keyboard"})

        restored = MentionBridgeIndex.from_dict(idx.to_dict())

        assert restored.expand({"digital piano"}) == {"digital piano", "korg b1"}
        assert restored.linked_to("korg b1") == {"digital piano", "keyboard"}


class TestQueryPlanTypeHierarchy:
    def test_type_depth_zero_unchanged(self) -> None:
        store = FoundryStore()
        _add_entity(store, "luna", "cat")
        _add_entity(store, "max", "dog")
        store.types.add_subsumption("pet", "cat", 1.0)

        default_result = QueryPlan(store).type("cat").execute()
        explicit_zero = QueryPlan(store).type("cat", depth=0).execute()

        assert [entity.canonical for entity in default_result] == [entity.canonical for entity in explicit_zero]

    def test_type_depth_one_includes_children(self) -> None:
        store = FoundryStore()
        _add_entity(store, "luna", "cat")
        _add_entity(store, "max", "dog")
        _add_entity(store, "desk", "furniture")
        store.types.add_subsumption("pet", "cat", 1.0)
        store.types.add_subsumption("pet", "dog", 1.0)

        result = QueryPlan(store).type("pet", depth=1).execute()

        assert {entity.canonical for entity in result} == {"luna", "max"}

    def test_type_depth_with_mention_bridges(self) -> None:
        store = FoundryStore()
        _add_entity(store, "korg_b1", "item")
        _add_entity(store, "yamaha_fg800", "acoustic_guitar")
        _add_entity(store, "office_chair", "item")
        store.types.add_subsumption("instrument", "keyboard", 1.0)
        store.types.add_subsumption("keyboard", "digital_piano", 1.0)
        store.types.add_subsumption("instrument", "guitar", 1.0)
        store.types.add_subsumption("guitar", "acoustic_guitar", 1.0)
        store.mention_bridges.add("korg b1", {"digital piano", "keyboard"})

        result = QueryPlan(store).type("instrument", depth=2).execute()

        assert {entity.canonical for entity in result} == {"korg_b1", "yamaha_fg800"}

    def test_counting_instruments(self) -> None:
        store = _build_counting_store()

        assert QueryPlan(store).type("instrument", depth=2).count() == 4

    def test_counting_doctors(self) -> None:
        store = _build_counting_store()

        assert QueryPlan(store).type("doctor", depth=2).count() == 3

    def test_counting_kitchen(self) -> None:
        store = _build_counting_store()

        assert QueryPlan(store).type("kitchen item", depth=2).count() == 5

    def test_foundry_store_roundtrip_preserves_mention_bridges(self, tmp_path) -> None:
        store = FoundryStore()
        _add_entity(store, "korg_b1", "item")
        store.mention_bridges.add("korg b1", {"digital piano"})

        path = tmp_path / "store"
        store.save(str(path))
        loaded = FoundryStore.load(str(path))

        assert loaded.mention_bridges.expand({"digital piano"}) == {"digital piano", "korg b1"}

    def test_bridge_only_who_and_what_facets(self) -> None:
        """Non-type facets (when, where, quantity, etc.) must not pollute the bridge.

        Simulates the filtering in core.py line 517: only who+what facets
        are passed to mention_bridges.add(), not the other 7.
        """
        facets = {
            "what": "digital piano",
            "who": "musician",
            "where": "guitar center",
            "when": "last tuesday",
            "quantity": "$500",
            "frequency": "weekly",
            "how": "online",
            "sentiment": "happy",
            "status": "new",
        }
        bridge_facets = {v for k, v in facets.items() if k in ("who", "what")}

        idx = MentionBridgeIndex()
        idx.add("korg b1", bridge_facets)

        linked = idx.linked_to("korg b1")
        assert "digital piano" in linked
        assert "musician" in linked
        assert "guitar center" not in linked
        assert "last tuesday" not in linked
        assert "$500" not in linked
        assert "weekly" not in linked
        assert len(linked) == 2
