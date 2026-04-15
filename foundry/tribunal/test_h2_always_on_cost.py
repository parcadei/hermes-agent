"""Test 4: Always-on cost analysis.

Computes latency breakdown from agentic traces and estimates the cost
of running all retrieval channels unconditionally (no routing).

OFFLINE test -- uses pre-computed trace files.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRACES_FULL = ROOT / "proofs/hermes-NLCDM/python/output/agentic_retrieval_full_infra_60q.json"


@pytest.fixture(scope="module")
def traces():
    with open(TRACES_FULL) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgenticLatencyDistribution:
    """Characterize the agentic retriever's latency profile."""

    def test_mean_latency(self, traces):
        latencies = [r["trace"]["total_ms"] for r in traces["results"]]
        mean = statistics.mean(latencies)
        print(f"\nAgentic mean latency: {mean:.0f}ms")
        assert mean > 5000, "Expected mean > 5s for agentic pipeline"

    def test_latency_percentiles(self, traces):
        latencies = sorted(r["trace"]["total_ms"] for r in traces["results"])
        n = len(latencies)
        print(f"\nLatency distribution (n={n}):")
        print(f"  min:    {latencies[0]:.0f}ms")
        print(f"  p25:    {latencies[n // 4]:.0f}ms")
        print(f"  median: {latencies[n // 2]:.0f}ms")
        print(f"  p75:    {latencies[3 * n // 4]:.0f}ms")
        print(f"  max:    {latencies[-1]:.0f}ms")


class TestToolLevelLatency:
    """Break down latency by tool to isolate infrastructure cost from LLM cost."""

    def test_per_tool_latency(self, traces):
        from collections import defaultdict

        tool_times = defaultdict(list)
        for r in traces["results"]:
            for s in r["trace"]["steps"]:
                tool_times[s["tool"]].append(s.get("elapsed_ms", 0))

        print("\n=== Per-Tool Latency ===")
        total_tool_ms = 0
        for tool in sorted(tool_times.keys()):
            times = tool_times[tool]
            mean_t = statistics.mean(times) if times else 0
            total = sum(times)
            total_tool_ms += total
            print(f"  {tool}: n={len(times)}, mean={mean_t:.0f}ms, total={total:.0f}ms")

        print(f"\nTotal tool execution time (all 60 questions): {total_tool_ms:.0f}ms")
        print(f"Mean tool execution per question: {total_tool_ms / 60:.0f}ms")

    def test_llm_reasoning_overhead(self, traces):
        """Compute LLM reasoning overhead = total latency - tool execution time."""
        from collections import defaultdict

        tool_total_ms = 0
        for r in traces["results"]:
            for s in r["trace"]["steps"]:
                tool_total_ms += s.get("elapsed_ms", 0)

        total_latency_ms = sum(r["trace"]["total_ms"] for r in traces["results"])
        llm_overhead_ms = total_latency_ms - tool_total_ms

        n = len(traces["results"])
        print(f"\n=== LLM Reasoning Overhead ===")
        print(f"Total latency (60q):          {total_latency_ms:.0f}ms")
        print(f"Tool execution (60q):         {tool_total_ms:.0f}ms")
        print(f"LLM reasoning overhead (60q): {llm_overhead_ms:.0f}ms")
        print(f"LLM overhead per question:    {llm_overhead_ms / n:.0f}ms")
        print(f"LLM overhead fraction:        {llm_overhead_ms / total_latency_ms * 100:.1f}%")

        # The LLM reasoning should be the dominant cost (>80%)
        assert llm_overhead_ms / total_latency_ms > 0.80, \
            "Expected LLM overhead to be > 80% of total latency"


class TestDeterministicCostEstimate:
    """Estimate deterministic always-on cost and compare to agentic."""

    def test_speedup_estimate(self, traces):
        """Estimate speedup from switching to deterministic always-on pipeline.

        Deterministic cost components:
        - BM25 retrieval: ~50ms
        - Entity BM25: ~5ms
        - Entity graph walk: ~10ms
        - Temporal search: ~3ms
        - Answer generation (gpt-4o-mini): ~1000ms
        Total estimate: ~1068ms

        This is a LOWER BOUND because it assumes answer generation latency
        is the same. In practice, the deterministic pipeline may use shorter
        prompts (no ReAct trace), which could reduce answer gen time.
        """
        latencies = [r["trace"]["total_ms"] for r in traces["results"]]
        agentic_mean = statistics.mean(latencies)

        # Component estimates from trace data
        bm25_ms = 50   # Based on search_memory median (~109ms) but without LLM overhead
        entity_ms = 5   # From search_entities median
        graph_ms = 10   # From get_related median
        temporal_ms = 3  # From search_temporal median
        ansgen_ms = 1000 # Estimated gpt-4o-mini answer generation

        deterministic_ms = bm25_ms + entity_ms + graph_ms + temporal_ms + ansgen_ms

        speedup = agentic_mean / deterministic_ms

        print(f"\n=== Cost Comparison ===")
        print(f"Agentic mean:           {agentic_mean:.0f}ms/query")
        print(f"Deterministic estimate: {deterministic_ms}ms/query")
        print(f"Speedup:                {speedup:.1f}x")
        print(f"LLM routing savings:    {agentic_mean - deterministic_ms:.0f}ms/query")
        print(f"")
        print(f"Component breakdown:")
        print(f"  BM25:              {bm25_ms}ms")
        print(f"  Entity BM25:       {entity_ms}ms")
        print(f"  Entity graph:      {graph_ms}ms")
        print(f"  Temporal search:   {temporal_ms}ms")
        print(f"  Answer generation: {ansgen_ms}ms")

        # The deterministic pipeline should be substantially faster
        assert speedup >= 3.0, f"Expected >= 3x speedup, got {speedup:.1f}x"

    def test_300q_wall_time(self, traces):
        """Estimate wall time for full 300-question eval: agentic vs deterministic."""
        latencies = [r["trace"]["total_ms"] for r in traces["results"]]
        agentic_mean = statistics.mean(latencies)
        deterministic_ms = 1068

        agentic_300q_min = agentic_mean * 300 / 1000 / 60
        deterministic_300q_min = deterministic_ms * 300 / 1000 / 60

        print(f"\n=== Full 300-Question Eval Wall Time ===")
        print(f"Agentic:       {agentic_300q_min:.1f} minutes")
        print(f"Deterministic: {deterministic_300q_min:.1f} minutes")
        print(f"Savings:       {agentic_300q_min - deterministic_300q_min:.1f} minutes")
