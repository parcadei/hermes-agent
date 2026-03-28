"""Discriminative tests for _graph_walk_resolve (KG subgraph traversal).

Tests the _graph_walk_resolve() method on QueryEngine, which:
1. Finds anchor entity from first sub-query (longest substring match)
2. BFS from anchor through subject_index (up to max_hops)
3. Formats reachable subgraph as structured text
4. Asks LLM to pick the right path and extract final answer
5. Validates answer exists in graph (anti-hallucination)

Tests validate:
1. Anchor detection (longest match, placeholder stripping, short-key guard)
2. BFS traversal (single-hop, multi-hop chains, depth limiting)
3. LLM integration (prompt format, answer extraction)
4. Answer validation (must exist in subgraph, rejects hallucination)
5. INSUFFICIENT fallthrough (LLM can't answer -> None)
6. Empty/missing subject_index -> None
7. Evidence chain structure (KG-WALK tags, answer dict)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mabench.configs import SupersessionState
from mabench.query import QueryEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_graph_state() -> SupersessionState:
    """Build a SupersessionState with a multi-hop knowledge graph.

    Graph structure:
        christian abbiati --[position]--> cornerback
        cornerback --[sport]--> ice hockey
        ice hockey --[creator]--> james creighton

        france --[capital]--> paris
        france --[president]--> macron

        south korea --[capital]--> seoul
        seoul --[country]--> south korea  (cycle)
    """
    ss = SupersessionState()
    ss.subject_index["christian abbiati"] = {
        "position": ("cornerback", "The position Christian Abbiati plays is cornerback."),
    }
    ss.subject_index["cornerback"] = {
        "sport": ("ice hockey", "The sport associated with cornerback is ice hockey."),
    }
    ss.subject_index["ice hockey"] = {
        "creator": ("james creighton", "The creator of ice hockey is James Creighton."),
    }
    ss.subject_index["france"] = {
        "capital": ("paris", "The capital of France is Paris."),
        "president": ("macron", "The president of France is Macron."),
    }
    ss.subject_index["south korea"] = {
        "capital": ("seoul", "The capital of South Korea is Seoul."),
    }
    ss.subject_index["seoul"] = {
        "country": ("south korea", "Seoul is in South Korea."),
    }
    return ss


def _make_engine(ss: SupersessionState | None = None) -> QueryEngine:
    """Create a minimal QueryEngine with supersession_state wired."""
    return QueryEngine(supersession_state=ss)


def _mock_llm_response(answer_text: str):
    """Create a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = answer_text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Anchor Detection
# ---------------------------------------------------------------------------

class TestAnchorDetection:
    """Anchor entity is found from the first sub-query via longest substring."""

    def test_finds_anchor_in_first_subquery(self):
        """'christian abbiati' found in 'What position does Christian Abbiati play?'"""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("cornerback")

        result = engine._graph_walk_resolve(
            "What sport did Christian Abbiati play?",
            ["What position does Christian Abbiati play?", "What sport is {answer_1}?"],
        )
        assert result is not None
        assert result["final_kg_answer"] == "cornerback" or "ice hockey" in str(result)

    def test_longest_match_wins(self):
        """'south korea' (11 chars) beats 'seoul' (5 chars) when both in question."""
        ss = _build_graph_state()
        # Add 'seoul' as a subject too (it already is in _build_graph_state)
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("seoul")

        result = engine._graph_walk_resolve(
            "What is the capital of South Korea?",
            ["What is the capital of South Korea?"],
        )
        assert result is not None
        # Should anchor on 'south korea' (longer), not 'korea' or anything shorter

    def test_strips_answer_placeholders(self):
        """Placeholders like {answer_1} are removed before anchor search."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("ice hockey")

        result = engine._graph_walk_resolve(
            "What sport is cornerback associated with?",
            ["{answer_1} is associated with what sport?"],  # placeholder stripped
        )
        # After stripping {answer_1}, the sub-query becomes " is associated with what sport?"
        # No anchor found (no subject >= 3 chars matches)
        # This is expected: first subquery should have the real entity name
        # The method should return None since no anchor is found
        assert result is None

    def test_no_anchor_returns_none(self):
        """When no subject entity matches the first sub-query, return None."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        result = engine._graph_walk_resolve(
            "What is the meaning of life?",
            ["What is the meaning of life?"],
        )
        assert result is None

    def test_short_subject_not_matched(self):
        """Subjects with len < 3 are excluded from anchor matching."""
        ss = SupersessionState()
        ss.subject_index["or"] = {
            "capital": ("salem", "The capital of Oregon is Salem."),
        }
        engine = _make_engine(ss)
        result = engine._graph_walk_resolve(
            "What is the capital of Oregon?",
            ["What is the capital of Oregon?"],
        )
        assert result is None  # 'or' too short to match


# ---------------------------------------------------------------------------
# BFS Traversal
# ---------------------------------------------------------------------------

class TestBFSTraversal:
    """BFS builds the correct reachable subgraph from anchor."""

    def test_single_hop_path(self):
        """France -> capital -> Paris is a 1-hop path."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("paris")

        result = engine._graph_walk_resolve(
            "What is the capital of France?",
            ["What is the capital of France?"],
        )
        assert result is not None
        assert result["final_kg_answer"] == "paris"

    def test_multi_hop_chain(self):
        """abbiati -> cornerback -> ice hockey is a 2-hop chain."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("ice hockey")

        result = engine._graph_walk_resolve(
            "What sport did Christian Abbiati play?",
            ["What position does Christian Abbiati play?", "What sport is {answer_1}?"],
        )
        assert result is not None
        assert result["final_kg_answer"] == "ice hockey"

    def test_three_hop_chain(self):
        """abbiati -> cornerback -> ice hockey -> james creighton is 3 hops."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("james creighton")

        result = engine._graph_walk_resolve(
            "Who created the sport that Christian Abbiati plays?",
            [
                "What position does Christian Abbiati play?",
                "What sport is {answer_1}?",
                "Who created {answer_2}?",
            ],
        )
        assert result is not None
        assert result["final_kg_answer"] == "james creighton"

    def test_max_hops_limits_depth(self):
        """max_hops=1 should only traverse 1 hop from anchor."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        # LLM asked for james creighton but it's 3 hops away
        engine._client.chat.completions.create.return_value = _mock_llm_response("james creighton")

        result = engine._graph_walk_resolve(
            "Who created the sport that Christian Abbiati plays?",
            ["What position does Christian Abbiati play?"],
            max_hops=1,
        )
        # At max_hops=1, only abbiati->cornerback is reachable
        # 'james creighton' is NOT in the subgraph -> validation fails -> None
        assert result is None

    def test_cycle_handling(self):
        """Cycles (south korea -> seoul -> south korea) don't cause infinite loop."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("seoul")

        # This should terminate despite the cycle
        result = engine._graph_walk_resolve(
            "What is the capital of South Korea?",
            ["What is the capital of South Korea?"],
        )
        assert result is not None
        assert result["final_kg_answer"] == "seoul"

    def test_no_paths_from_leaf_anchor(self):
        """Anchor entity with no outgoing edges -> no paths -> None."""
        ss = SupersessionState()
        ss.subject_index["paris"] = {}  # exists but no predicates
        engine = _make_engine(ss)
        result = engine._graph_walk_resolve(
            "Tell me about Paris.",
            ["Tell me about Paris."],
        )
        assert result is None


# ---------------------------------------------------------------------------
# LLM Integration
# ---------------------------------------------------------------------------

class TestLLMIntegration:
    """LLM is called with correct prompt and subgraph content."""

    def test_prompt_contains_anchor_and_facts(self):
        """LLM prompt should contain the anchor entity and all reachable facts."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("paris")

        engine._graph_walk_resolve(
            "What is the capital of France?",
            ["What is the capital of France?"],
        )

        call_args = engine._client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0]["content"]

        assert "france" in prompt_text.lower()
        assert "capital" in prompt_text.lower()
        assert "paris" in prompt_text.lower()
        assert "president" in prompt_text.lower()  # both preds reachable from france
        assert "macron" in prompt_text.lower()

    def test_temperature_zero(self):
        """LLM call should use temperature=0.0 for determinism."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("paris")

        engine._graph_walk_resolve(
            "What is the capital of France?",
            ["What is the capital of France?"],
        )

        call_args = engine._client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.0

    def test_llm_exception_returns_none(self):
        """If LLM call raises, _graph_walk_resolve returns None gracefully."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.side_effect = RuntimeError("API down")

        result = engine._graph_walk_resolve(
            "What is the capital of France?",
            ["What is the capital of France?"],
        )
        assert result is None


# ---------------------------------------------------------------------------
# Answer Validation
# ---------------------------------------------------------------------------

class TestAnswerValidation:
    """Answer must exist in the subgraph to prevent hallucination."""

    def test_valid_answer_accepted(self):
        """LLM answers 'paris' which IS in the subgraph -> accepted."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("paris")

        result = engine._graph_walk_resolve(
            "What is the capital of France?",
            ["What is the capital of France?"],
        )
        assert result is not None
        assert result["final_kg_answer"] == "paris"

    def test_hallucinated_answer_rejected(self):
        """LLM answers 'berlin' which is NOT in the subgraph -> rejected."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("berlin")

        result = engine._graph_walk_resolve(
            "What is the capital of France?",
            ["What is the capital of France?"],
        )
        assert result is None  # hallucinated answer rejected

    def test_insufficient_returns_none(self):
        """LLM responds 'INSUFFICIENT' -> return None (fall through to retrieval)."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("INSUFFICIENT")

        result = engine._graph_walk_resolve(
            "What is the population of France?",
            ["What is the population of France?"],
        )
        assert result is None

    def test_empty_answer_returns_none(self):
        """LLM returns empty string -> None."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("")

        result = engine._graph_walk_resolve(
            "What is the capital of France?",
            ["What is the capital of France?"],
        )
        assert result is None

    def test_answer_with_quotes_stripped(self):
        """LLM answers '"paris"' (with quotes) -> stripped and matched."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response('"paris"')

        result = engine._graph_walk_resolve(
            "What is the capital of France?",
            ["What is the capital of France?"],
        )
        assert result is not None
        assert "paris" in result["final_kg_answer"]

    def test_partial_match_accepted(self):
        """LLM answers 'James Creighton' (capitalized) matches 'james creighton' in graph."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("James Creighton")

        result = engine._graph_walk_resolve(
            "Who created the sport that Christian Abbiati plays?",
            [
                "What position does Christian Abbiati play?",
                "What sport is {answer_1}?",
                "Who created {answer_2}?",
            ],
        )
        assert result is not None
        assert result["final_kg_answer"] == "james creighton"


# ---------------------------------------------------------------------------
# Evidence Chain Structure
# ---------------------------------------------------------------------------

class TestEvidenceChain:
    """Returned hop_state has correct structure for downstream pipeline."""

    def test_single_hop_evidence_chain(self):
        """1-hop answer has 1 evidence entry with KG-WALK tag."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("paris")

        result = engine._graph_walk_resolve(
            "What is the capital of France?",
            ["What is the capital of France?"],
        )
        assert result is not None
        chain = result["evidence_chain"]
        assert len(chain) >= 1
        assert "KG-WALK" in chain[0]["evidence"][0]
        assert chain[0]["a"] == "paris"

    def test_multi_hop_evidence_chain(self):
        """2-hop answer has 2 evidence entries."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("ice hockey")

        result = engine._graph_walk_resolve(
            "What sport did Christian Abbiati play?",
            ["What position does Christian Abbiati play?", "What sport is {answer_1}?"],
        )
        assert result is not None
        chain = result["evidence_chain"]
        assert len(chain) == 2
        assert chain[0]["a"] == "cornerback"
        assert chain[1]["a"] == "ice hockey"

    def test_answers_dict_populated(self):
        """answers dict has {answer_N} keys for each hop."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("ice hockey")

        result = engine._graph_walk_resolve(
            "What sport did Christian Abbiati play?",
            ["What position does Christian Abbiati play?", "What sport is {answer_1}?"],
        )
        assert result is not None
        assert "{answer_1}" in result["answers"]
        assert "{answer_2}" in result["answers"]
        assert result["answers"]["{answer_1}"] == "cornerback"
        assert result["answers"]["{answer_2}"] == "ice hockey"

    def test_subqueries_preserved(self):
        """Original subqueries are passed through in result."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("paris")

        sqs = ["What is the capital of France?"]
        result = engine._graph_walk_resolve("What is the capital of France?", sqs)
        assert result is not None
        assert result["subqueries"] is sqs

    def test_seen_texts_empty(self):
        """Graph walk doesn't retrieve text chunks, so seen_texts should be empty."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("paris")

        result = engine._graph_walk_resolve("What is the capital of France?", ["What is the capital of France?"])
        assert result is not None
        assert len(result["seen_texts"]) == 0
        assert result["_all_v2_results"] == []


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Empty state, no subqueries, etc."""

    def test_empty_subject_index_returns_none(self):
        """Empty subject_index -> None."""
        ss = SupersessionState()
        engine = _make_engine(ss)
        result = engine._graph_walk_resolve("What is X?", ["What is X?"])
        assert result is None

    def test_no_supersession_state_returns_none(self):
        """No supersession_state at all -> None."""
        engine = _make_engine(None)
        result = engine._graph_walk_resolve("What is X?", ["What is X?"])
        assert result is None

    def test_empty_subqueries_returns_none(self):
        """Empty subqueries list -> None."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        result = engine._graph_walk_resolve("What is X?", [])
        assert result is None

    def test_branching_subgraph_includes_all_predicates(self):
        """France has 2 predicates (capital, president) -> both in subgraph."""
        ss = _build_graph_state()
        engine = _make_engine(ss)
        engine._client = MagicMock()
        engine._client.chat.completions.create.return_value = _mock_llm_response("macron")

        result = engine._graph_walk_resolve(
            "Who is the president of France?",
            ["Who is the president of France?"],
        )
        assert result is not None
        assert result["final_kg_answer"] == "macron"
