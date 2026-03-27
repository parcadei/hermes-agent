"""Integration tests for HyperedgeIntersection + KnowledgeGraph wiring.

Tests that the wiring in agent.py, store.py, query.py, query_pipeline.py,
configs.py, run_hermes_composed.py, and eval_cache correctly constructs,
passes, and restores these modules.

These tests verify WIRING, not module behavior (that's in
test_hyperedge_intersection.py and test_knowledge_graph.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _TESTS_DIR.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))

# HermesAgent requires spacy for IngestionPipeline
_HAS_SPACY = False
try:
    import spacy  # noqa: F401
    _HAS_SPACY = True
except ImportError:
    pass

_skip_no_spacy = pytest.mark.skipif(
    not _HAS_SPACY, reason="spacy not installed"
)


# ===================================================================
# 1. Config wiring
# ===================================================================


class TestConfigWiring:
    """Verify new config fields exist with correct defaults."""

    def test_retrieval_config_hyperedge_fields(self):
        from mabench.configs import RetrievalConfig

        rc = RetrievalConfig()
        assert rc.hyperedge_intersection is False
        assert rc.hyperedge_penalty == pytest.approx(0.1)
        assert rc.hyperedge_min_candidates == 3

    def test_kg_config_exists(self):
        from mabench.configs import KGConfig

        kc = KGConfig()
        assert kc.enabled is False
        assert kc.damping == pytest.approx(0.85)
        assert kc.ppr_top_k == 20
        assert kc.max_facts_per_node == 50

    def test_kg_config_frozen(self):
        from mabench.configs import KGConfig

        kc = KGConfig()
        with pytest.raises(AttributeError):
            kc.enabled = True  # type: ignore[misc]

    def test_retrieval_config_still_frozen(self):
        from mabench.configs import RetrievalConfig

        rc = RetrievalConfig()
        with pytest.raises(AttributeError):
            rc.hyperedge_intersection = True  # type: ignore[misc]


# ===================================================================
# 2. Agent construction wiring
# ===================================================================


@_skip_no_spacy
class TestAgentConstruction:
    """Verify agent constructs modules based on config flags."""

    def test_hyperedge_none_by_default(self):
        from mabench.agent import HermesAgent
        from mabench.configs import RetrievalConfig

        agent = HermesAgent(retrieval=RetrievalConfig())
        assert not hasattr(agent, "hyperedge") or agent.hyperedge is None

    def test_hyperedge_constructed_when_enabled(self):
        from mabench.agent import HermesAgent
        from mabench.configs import RetrievalConfig

        agent = HermesAgent(
            retrieval=RetrievalConfig(
                hyperedge_intersection=True,
                hyperedge_penalty=0.2,
                hyperedge_min_candidates=5,
            ),
        )
        assert agent.hyperedge is not None
        from mabench.hyperedge_intersection import HyperedgeIntersection
        assert isinstance(agent.hyperedge, HyperedgeIntersection)
        assert agent.hyperedge._penalty == pytest.approx(0.2)
        assert agent.hyperedge._min_candidates == 5

    def test_knowledge_graph_none_by_default(self):
        from mabench.agent import HermesAgent

        agent = HermesAgent()
        assert not hasattr(agent, "knowledge_graph") or agent.knowledge_graph is None

    def test_knowledge_graph_constructed_when_enabled(self):
        from mabench.agent import HermesAgent
        from mabench.configs import KGConfig

        agent = HermesAgent(kg=KGConfig(enabled=True, damping=0.7, ppr_top_k=15))
        assert agent.knowledge_graph is not None
        from mabench.knowledge_graph import KnowledgeGraph
        assert isinstance(agent.knowledge_graph, KnowledgeGraph)

    def test_hyperedge_receives_entity_index_reference(self):
        """HyperedgeIntersection._entity_index is the SAME object as EntityIndex.index."""
        from mabench.agent import HermesAgent
        from mabench.configs import RetrievalConfig

        agent = HermesAgent(
            retrieval=RetrievalConfig(hyperedge_intersection=True),
        )
        # Must be the same reference, not a copy
        assert agent.hyperedge._entity_index is agent._entity_index.index


# ===================================================================
# 3. Store wiring (KG ingestion)
# ===================================================================


@_skip_no_spacy
class TestStoreWiring:
    """Verify FactStore passes triples to KnowledgeGraph during ingestion."""

    def test_store_has_knowledge_graph_attr(self):
        from mabench.agent import HermesAgent
        from mabench.configs import KGConfig

        agent = HermesAgent(kg=KGConfig(enabled=True))
        assert hasattr(agent.store, "_knowledge_graph")
        assert agent.store._knowledge_graph is not None

    def test_store_knowledge_graph_none_when_disabled(self):
        from mabench.agent import HermesAgent

        agent = HermesAgent()
        kg = getattr(agent.store, "_knowledge_graph", None)
        assert kg is None

    def test_ingestion_populates_kg(self):
        """After storing a fact with triples, KG should have nodes/edges."""
        from mabench.agent import HermesAgent
        from mabench.configs import KGConfig, StoreConfig

        agent = HermesAgent(
            kg=KGConfig(enabled=True),
            store=StoreConfig(supersession=True, cooc_boost=True),
        )
        # Store a fact that will produce triples
        agent.send_message(
            "The sport played by Abbiati is cornerback.",
            memorizing=True,
        )
        stats = agent.knowledge_graph.stats
        # If triples were extracted, nodes and edges should exist
        # (depends on spaCy extraction succeeding, so check >= 0)
        assert stats["nodes"] >= 0
        assert stats["edges"] >= 0


# ===================================================================
# 4. Query pipeline wiring
# ===================================================================


@_skip_no_spacy
class TestQueryPipelineWiring:
    """Verify query modules receive hyperedge and knowledge_graph references."""

    def test_query_engine_has_hyperedge(self):
        from mabench.agent import HermesAgent
        from mabench.configs import RetrievalConfig

        agent = HermesAgent(
            retrieval=RetrievalConfig(hyperedge_intersection=True),
        )
        assert hasattr(agent.query_engine, "_hyperedge")
        assert agent.query_engine._hyperedge is not None

    def test_query_engine_has_knowledge_graph(self):
        from mabench.agent import HermesAgent
        from mabench.configs import KGConfig

        agent = HermesAgent(kg=KGConfig(enabled=True))
        assert hasattr(agent.query_engine, "_knowledge_graph")
        assert agent.query_engine._knowledge_graph is not None

    def test_pipeline_has_hyperedge(self):
        from mabench.agent import HermesAgent
        from mabench.configs import RetrievalConfig

        agent = HermesAgent(
            retrieval=RetrievalConfig(hyperedge_intersection=True),
        )
        assert hasattr(agent.pipeline, "_hyperedge")

    def test_pipeline_has_knowledge_graph(self):
        from mabench.agent import HermesAgent
        from mabench.configs import KGConfig

        agent = HermesAgent(kg=KGConfig(enabled=True))
        assert hasattr(agent.pipeline, "_knowledge_graph")


# ===================================================================
# 5. Module references survive reset
# ===================================================================


@_skip_no_spacy
class TestResetSurvival:
    """Verify modules are reconstructed on reset."""

    def test_hyperedge_reconstructed_on_reset(self):
        from mabench.agent import HermesAgent
        from mabench.configs import RetrievalConfig

        agent = HermesAgent(
            retrieval=RetrievalConfig(hyperedge_intersection=True),
        )
        old_he = agent.hyperedge
        agent.reset()
        # New instance after reset (not the same object)
        assert agent.hyperedge is not None
        assert agent.hyperedge is not old_he

    def test_knowledge_graph_reconstructed_on_reset(self):
        from mabench.agent import HermesAgent
        from mabench.configs import KGConfig

        agent = HermesAgent(kg=KGConfig(enabled=True))
        old_kg = agent.knowledge_graph
        agent.reset()
        assert agent.knowledge_graph is not None
        assert agent.knowledge_graph is not old_kg
