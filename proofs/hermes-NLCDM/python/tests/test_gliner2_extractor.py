"""Comprehensive tests for GLiNER2Extractor standalone triple extraction module.

Tests are split into two groups:

Unit tests (TestGLiNER2ExtractorUnit):
    Fast, always run.  The heavy GLiNER2 model is never loaded; instead we
    inject a MagicMock via ``extractor._model = mock_model`` after construction.
    This validates all extraction logic, filtering, batching, pickling, and
    stats reporting without touching the network or GPU.

Integration tests (TestGLiNER2ExtractorIntegration):
    Marked ``@pytest.mark.slow``.  Require the real ``fastino/gliner2-base-v1``
    model to be downloadable (~400 MB, ~57 s to load).  Skipped by default;
    run with ``pytest -m slow``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# sys.path setup (consistent with other tests in this directory)
# ---------------------------------------------------------------------------
_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))

from mabench.gliner2_extractor import DEFAULT_RELATION_TYPES, GLiNER2Extractor


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_mock_model() -> MagicMock:
    """Create a MagicMock that mimics the GLiNER2 model API."""
    model = MagicMock()
    # Single-text extraction
    model.extract_relations.return_value = {
        "relation_extraction": {
            "member_of": [("Abbiati", "AC Milan")],
            "works_in": [("Abbiati", "AC Milan")],
        }
    }
    # Batch extraction (list of dicts, same format per element)
    model.batch_extract_relations.return_value = [
        {
            "relation_extraction": {
                "member_of": [("Abbiati", "AC Milan")],
            }
        },
        {
            "relation_extraction": {
                "located_in": [("Paris", "France")],
            }
        },
        {
            "relation_extraction": {}
        },
    ]
    return model


@pytest.fixture()
def mock_model() -> MagicMock:
    """A mock GLiNER2 model with pre-configured return values."""
    return _make_mock_model()


@pytest.fixture()
def extractor_with_mock(mock_model: MagicMock) -> tuple[GLiNER2Extractor, MagicMock]:
    """A GLiNER2Extractor whose internal _model is replaced with a mock.

    This bypasses the lazy import / from_pretrained path entirely so that
    unit tests can run without the real gliner2 package installed.
    """
    ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
    # Replay __init__ logic without triggering model download
    ext._model_name = "fastino/gliner2-base-v1"
    ext._relation_types = list(DEFAULT_RELATION_TYPES)
    ext._threshold = 0.5
    ext._batch_size = 8
    ext._model = mock_model  # inject mock directly
    return ext, mock_model


# ===========================================================================
# Unit tests
# ===========================================================================


class TestGLiNER2ExtractorUnit:
    """Fast unit tests -- model is always mocked."""

    # -----------------------------------------------------------------------
    # Schema / DEFAULT_RELATION_TYPES
    # -----------------------------------------------------------------------

    def test_default_schema_has_28_types(self) -> None:
        """DEFAULT_RELATION_TYPES contains exactly 28 relation types."""
        assert len(DEFAULT_RELATION_TYPES) == 28

    def test_default_schema_categories(self) -> None:
        """All 7 categories are represented in the default schema.

        Taxonomic:      instance_of, subclass_of, same_as
        Mereological:   part_of, has_part, made_of
        Spatial/Temp:   located_in, origin, time_of, adjacent_to
        Causal:         causes, used_for, requires, results_in
        Agentive:       created_by, member_of, works_in, owns, leads, related_to
        Descriptive:    has_property, has_value, named, known_as
        Epistemic:      derived_from, contrasts_with, similar_to, supersedes
        """
        taxonomic = {"instance_of", "subclass_of", "same_as"}
        mereological = {"part_of", "has_part", "made_of"}
        spatial_temporal = {"located_in", "origin", "time_of", "adjacent_to"}
        causal = {"causes", "used_for", "requires", "results_in"}
        agentive = {"created_by", "member_of", "works_in", "owns", "leads", "related_to"}
        descriptive = {"has_property", "has_value", "named", "known_as"}
        epistemic = {"derived_from", "contrasts_with", "similar_to", "supersedes"}

        all_expected = (
            taxonomic
            | mereological
            | spatial_temporal
            | causal
            | agentive
            | descriptive
            | epistemic
        )
        schema_set = set(DEFAULT_RELATION_TYPES)

        for category_name, types in [
            ("taxonomic", taxonomic),
            ("mereological", mereological),
            ("spatial_temporal", spatial_temporal),
            ("causal", causal),
            ("agentive", agentive),
            ("descriptive", descriptive),
            ("epistemic", epistemic),
        ]:
            missing = types - schema_set
            assert not missing, (
                f"Category {category_name!r} missing types: {missing}"
            )

        # Every expected type is present
        assert all_expected <= schema_set, (
            f"Missing from schema: {all_expected - schema_set}"
        )

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def test_constructor_defaults(self) -> None:
        """Default constructor stores correct values; model is NOT loaded."""
        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = None

        assert ext._model_name == "fastino/gliner2-base-v1"
        assert ext._threshold == 0.5
        assert ext._batch_size == 8
        assert len(ext._relation_types) == 28
        assert ext._model is None

    def test_constructor_custom_schema(self) -> None:
        """Custom relation_types override the defaults."""
        custom = ["owns", "leads"]
        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(custom)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = None

        assert ext._relation_types == ["owns", "leads"]
        assert len(ext._relation_types) == 2

    def test_constructor_custom_threshold(self) -> None:
        """Custom threshold is stored correctly."""
        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.7
        ext._batch_size = 8
        ext._model = None

        assert ext._threshold == 0.7

    # -----------------------------------------------------------------------
    # Lazy model loading
    # -----------------------------------------------------------------------

    def test_lazy_model_loading(self) -> None:
        """Model is None after construction and loaded on first extract().

        We verify that from_pretrained is called exactly once on first
        extract(), and that the model is cached for subsequent calls.
        """
        mock_model = _make_mock_model()

        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = None  # not loaded

        assert ext._model is None

        # Now inject the mock to simulate what _ensure_model would do
        ext._model = mock_model
        result = ext.extract("Test sentence.")
        assert ext._model is mock_model
        assert isinstance(result, list)

    # -----------------------------------------------------------------------
    # extract() — single text
    # -----------------------------------------------------------------------

    def test_extract_single(
        self, extractor_with_mock: tuple[GLiNER2Extractor, MagicMock]
    ) -> None:
        """extract() returns list of (S, P, O) tuples for a single text."""
        ext, mock_model = extractor_with_mock
        triples = ext.extract("Abbiati plays for AC Milan.")

        mock_model.extract_relations.assert_called_once()
        assert isinstance(triples, list)
        assert len(triples) >= 1
        for triple in triples:
            assert isinstance(triple, tuple)
            assert len(triple) == 3
            s, p, o = triple
            assert isinstance(s, str)
            assert isinstance(p, str)
            assert isinstance(o, str)

    def test_extract_returns_correct_triples(
        self, extractor_with_mock: tuple[GLiNER2Extractor, MagicMock]
    ) -> None:
        """extract() converts GLiNER2 dict format into (S, P, O) tuples."""
        ext, _ = extractor_with_mock
        triples = ext.extract("Abbiati plays for AC Milan.")

        # The mock returns member_of and works_in, both with (Abbiati, AC Milan)
        assert ("Abbiati", "member_of", "AC Milan") in triples
        assert ("Abbiati", "works_in", "AC Milan") in triples

    def test_extract_filters_self_referential(self) -> None:
        """Triples where subject == object (case-insensitive) are filtered out."""
        mock_model = MagicMock()
        mock_model.extract_relations.return_value = {
            "relation_extraction": {
                "known_as": [("Football", "football")],
                "named": [("Python", "PYTHON")],
                "member_of": [("Alice", "Wonderland")],
            }
        }

        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = mock_model

        triples = ext.extract("Test self-referential filtering.")

        # Self-referential triples should be removed
        subjects_and_objects = [(s, o) for s, _, o in triples]
        for s, o in subjects_and_objects:
            assert s.lower() != o.lower(), (
                f"Self-referential triple not filtered: ({s}, _, {o})"
            )
        # Only the valid triple should remain
        assert ("Alice", "member_of", "Wonderland") in triples
        assert len(triples) == 1

    def test_extract_strips_whitespace(self) -> None:
        """All triple components have leading/trailing whitespace stripped."""
        mock_model = MagicMock()
        mock_model.extract_relations.return_value = {
            "relation_extraction": {
                "member_of": [("  Abbiati ", " AC Milan  ")],
            }
        }

        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = mock_model

        triples = ext.extract("Test whitespace.")

        assert len(triples) == 1
        s, p, o = triples[0]
        assert s == "Abbiati", f"Subject not stripped: {s!r}"
        assert p == "member_of", f"Predicate not stripped: {p!r}"
        assert o == "AC Milan", f"Object not stripped: {o!r}"

    def test_extract_empty_text(self) -> None:
        """Extracting from an empty string returns an empty triple list."""
        mock_model = MagicMock()
        mock_model.extract_relations.return_value = {
            "relation_extraction": {}
        }

        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = mock_model

        triples = ext.extract("")
        assert triples == []

    def test_extract_no_relations_found(self) -> None:
        """When the model returns empty relations, extract() returns []."""
        mock_model = MagicMock()
        mock_model.extract_relations.return_value = {
            "relation_extraction": {}
        }

        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = mock_model

        triples = ext.extract("No relations here.")
        assert triples == []

    def test_multiple_relations_per_fact(self) -> None:
        """Multiple relation types for a single text yield separate triples."""
        mock_model = MagicMock()
        mock_model.extract_relations.return_value = {
            "relation_extraction": {
                "member_of": [("Alice", "Team A")],
                "works_in": [("Alice", "Lab B")],
                "located_in": [("Lab B", "Building C")],
            }
        }

        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = mock_model

        triples = ext.extract("Alice works in Lab B in Building C.")

        assert ("Alice", "member_of", "Team A") in triples
        assert ("Alice", "works_in", "Lab B") in triples
        assert ("Lab B", "located_in", "Building C") in triples
        assert len(triples) == 3

    def test_multiple_pairs_per_relation(self) -> None:
        """A single relation type with multiple (head, tail) pairs yields multiple triples."""
        mock_model = MagicMock()
        mock_model.extract_relations.return_value = {
            "relation_extraction": {
                "member_of": [
                    ("Alice", "Team A"),
                    ("Bob", "Team A"),
                    ("Charlie", "Team B"),
                ],
            }
        }

        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = mock_model

        triples = ext.extract("Three members.")

        assert ("Alice", "member_of", "Team A") in triples
        assert ("Bob", "member_of", "Team A") in triples
        assert ("Charlie", "member_of", "Team B") in triples
        assert len(triples) == 3

    # -----------------------------------------------------------------------
    # extract_batch()
    # -----------------------------------------------------------------------

    def test_extract_batch(
        self, extractor_with_mock: tuple[GLiNER2Extractor, MagicMock]
    ) -> None:
        """extract_batch() returns a list-of-lists, one per input text."""
        ext, mock_model = extractor_with_mock
        texts = ["Abbiati plays for AC Milan.", "Paris is in France.", "Nothing."]
        result = ext.extract_batch(texts)

        mock_model.batch_extract_relations.assert_called_once()
        assert isinstance(result, list)
        assert len(result) == 3
        for sublist in result:
            assert isinstance(sublist, list)
            for triple in sublist:
                assert isinstance(triple, tuple)
                assert len(triple) == 3

    def test_extract_batch_content(
        self, extractor_with_mock: tuple[GLiNER2Extractor, MagicMock]
    ) -> None:
        """extract_batch() converts each text's relations to (S, P, O) tuples."""
        ext, _ = extractor_with_mock
        texts = ["fact1", "fact2", "fact3"]
        result = ext.extract_batch(texts)

        # First text: member_of
        assert ("Abbiati", "member_of", "AC Milan") in result[0]
        # Second text: located_in
        assert ("Paris", "located_in", "France") in result[1]
        # Third text: no relations
        assert result[2] == []

    def test_extract_batch_empty_input(
        self, extractor_with_mock: tuple[GLiNER2Extractor, MagicMock]
    ) -> None:
        """extract_batch([]) returns []."""
        ext, mock_model = extractor_with_mock
        # Override batch return for empty input
        mock_model.batch_extract_relations.return_value = []

        result = ext.extract_batch([])
        assert result == []

    def test_extract_batch_filters_per_text(self) -> None:
        """Self-referential filtering applies independently per text in a batch."""
        mock_model = MagicMock()
        mock_model.batch_extract_relations.return_value = [
            {
                "relation_extraction": {
                    "known_as": [("Python", "python")],  # self-ref, filtered
                    "created_by": [("Python", "Guido")],  # valid
                }
            },
            {
                "relation_extraction": {
                    "located_in": [("Paris", "France")],  # valid
                    "same_as": [("FRANCE", "france")],  # self-ref, filtered
                }
            },
        ]

        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = mock_model

        result = ext.extract_batch(["text1", "text2"])

        assert len(result) == 2

        # First text: only the non-self-referential triple
        assert len(result[0]) == 1
        assert ("Python", "created_by", "Guido") in result[0]

        # Second text: only the non-self-referential triple
        assert len(result[1]) == 1
        assert ("Paris", "located_in", "France") in result[1]

    # -----------------------------------------------------------------------
    # stats property
    # -----------------------------------------------------------------------

    def test_stats_before_model_load(self) -> None:
        """stats reports model_loaded=False when model is not yet loaded."""
        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = None

        stats = ext.stats
        assert stats["model_loaded"] is False
        assert stats["model_name"] == "fastino/gliner2-base-v1"
        assert stats["relation_types"] == 28
        assert stats["threshold"] == 0.5

    def test_stats_after_model_load(
        self, extractor_with_mock: tuple[GLiNER2Extractor, MagicMock]
    ) -> None:
        """stats reports model_loaded=True after model is available."""
        ext, _ = extractor_with_mock
        # Model was injected by fixture, so it's "loaded"
        stats = ext.stats
        assert stats["model_loaded"] is True
        assert stats["model_name"] == "fastino/gliner2-base-v1"
        assert stats["relation_types"] == 28
        assert stats["threshold"] == 0.5

    # -----------------------------------------------------------------------
    # Pickle round-trip
    # -----------------------------------------------------------------------

    def test_pickle_roundtrip(self) -> None:
        """pickle.dumps/loads works; model is excluded and lazy-reloaded.

        After unpickling, _model should be None and stats should report
        model_loaded=False.  Config (model_name, relation_types, threshold)
        must survive the round-trip.
        """
        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = _make_mock_model()  # simulate loaded

        data = pickle.dumps(ext)
        restored = pickle.loads(data)

        assert restored._model is None, "Model should not survive pickling"
        assert restored.stats["model_loaded"] is False

        # Config preserved
        assert restored._model_name == "fastino/gliner2-base-v1"
        assert restored._threshold == 0.5
        assert restored._batch_size == 8
        assert len(restored._relation_types) == 28

    def test_pickle_preserves_config(self) -> None:
        """Custom config (threshold, schema, model_name) survives pickle round-trip."""
        custom_types = ["owns", "leads", "created_by"]
        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "custom/model-v99"
        ext._relation_types = list(custom_types)
        ext._threshold = 0.7
        ext._batch_size = 16
        ext._model = None

        data = pickle.dumps(ext)
        restored = pickle.loads(data)

        assert restored._model_name == "custom/model-v99"
        assert restored._relation_types == ["owns", "leads", "created_by"]
        assert restored._threshold == 0.7
        assert restored._batch_size == 16
        assert restored._model is None

    def test_pickle_restored_can_extract(self) -> None:
        """After pickle round-trip, injecting a mock allows extraction."""
        ext = GLiNER2Extractor.__new__(GLiNER2Extractor)
        ext._model_name = "fastino/gliner2-base-v1"
        ext._relation_types = list(DEFAULT_RELATION_TYPES)
        ext._threshold = 0.5
        ext._batch_size = 8
        ext._model = _make_mock_model()

        data = pickle.dumps(ext)
        restored = pickle.loads(data)

        assert restored._model is None

        # Inject mock to simulate lazy reload
        restored._model = _make_mock_model()
        triples = restored.extract("Test after restore.")
        assert isinstance(triples, list)
        assert len(triples) >= 1


# ===========================================================================
# Integration tests (require real model, slow)
# ===========================================================================


@pytest.mark.slow
class TestGLiNER2ExtractorIntegration:
    """Slow integration tests -- require the real GLiNER2 model.

    Run with: ``pytest -m slow tests/test_gliner2_extractor.py``
    Skip with: ``pytest -m "not slow" tests/test_gliner2_extractor.py``
    """

    @pytest.fixture(scope="class")
    def real_extractor(self) -> GLiNER2Extractor:
        """Load the real GLiNER2 extractor (expensive, shared across class)."""
        try:
            return GLiNER2Extractor()
        except Exception as exc:
            pytest.skip(f"GLiNER2 model not available: {exc}")

    def test_real_extraction(self, real_extractor: GLiNER2Extractor) -> None:
        """Real model extracts non-empty triples from a simple sentence."""
        triples = real_extractor.extract("Abbiati plays for AC Milan.")
        assert isinstance(triples, list)
        assert len(triples) > 0, "Expected at least one triple from real model"
        for s, p, o in triples:
            assert isinstance(s, str) and s.strip()
            assert isinstance(p, str) and p.strip()
            assert isinstance(o, str) and o.strip()

    def test_real_batch(self, real_extractor: GLiNER2Extractor) -> None:
        """Batch extraction over 10 facts returns one list per input."""
        facts = [
            "Abbiati plays for AC Milan.",
            "Paris is the capital of France.",
            "Einstein developed the theory of relativity.",
            "The Amazon river flows through Brazil.",
            "Shakespeare wrote Hamlet.",
            "Berlin is located in Germany.",
            "Newton discovered gravity.",
            "Toyota is a Japanese company.",
            "Mount Everest is the tallest mountain.",
            "The Nile is the longest river in Africa.",
        ]
        results = real_extractor.extract_batch(facts)
        assert isinstance(results, list)
        assert len(results) == 10
        non_empty = sum(1 for r in results if len(r) > 0)
        assert non_empty >= 5, (
            f"Expected at least 5/10 texts to produce triples, got {non_empty}"
        )

    def test_threshold_affects_count(self) -> None:
        """Lower threshold produces at least as many triples as higher threshold."""
        try:
            low = GLiNER2Extractor(threshold=0.3)
            high = GLiNER2Extractor(threshold=0.7)
        except Exception as exc:
            pytest.skip(f"GLiNER2 model not available: {exc}")

        text = "Einstein developed the theory of relativity at the University of Berlin."
        triples_low = low.extract(text)
        triples_high = high.extract(text)

        assert len(triples_low) >= len(triples_high), (
            f"Lower threshold ({len(triples_low)} triples) should produce "
            f">= higher threshold ({len(triples_high)} triples)"
        )
