"""Agentic retrieval — iterative tool-based memory search.

Inspired by A-RAG (arxiv 2602.03442) and DSPy ReAct. Instead of
single-pass retrieval + RRF, the agent controls its own search
strategy through hierarchical tools:

  1. search_memory(query)    — BM25 over raw chunks
  2. search_entities(query)  — BM25 over entity documents (name + facets)
  3. get_related(entity)     — graph walk: neighbors, co-mentions, properties
  4. search_temporal(query)  — temporal/date-based retrieval
  5. read_turn(turn_id)      — read full text of a specific turn
  6. answer(text)            — terminal: return collected evidence

The agent uses a ReAct loop (Thought → Action → Observation) with a
budget of max_steps tool calls. Traces are logged for future
fine-tuning of a small specialist model (0.6-1.4B).

Usage:
    ar = AgenticRetriever(
        bm25_index=bm25, entity_bm25=lm._entity_bm25,
        entity_graph=eg, memory_store=engine.memory_store,
        llm_base_url="https://openrouter.ai/api/v1",
        llm_model="gpt-4o-mini",
    )
    evidence, trace = ar.retrieve("What airline does the user fly?")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (shown to the agent LLM)
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS = """\
You have these tools to search a user's conversation memory:

search_memory(query: str) -> list[{turn_id, text, score}]
  BM25 keyword search over conversation chunks. Best for finding specific
  words, names, or phrases the user mentioned. Returns top-10 results.

search_entities(query: str) -> list[{entity, facets, turns, score}]
  Search over extracted entities (people, places, things) by name and
  attributes. Use when looking for WHO/WHAT/WHERE. Returns top-10 entities.

get_related(entity: str) -> {properties, co_mentions, neighbors}
  Given an entity name, retrieve its known properties, entities it
  co-occurs with, and graph neighbors. Use to explore connections.

search_temporal(query: str) -> list[{turn_id, text, date, score}]
  Search for events by date/time expressions. Use for WHEN questions,
  date ranges, temporal ordering. Returns turns with extracted dates.

read_turn(turn_id: int) -> {text, role, turn_id}
  Read the full text of a specific conversation turn by its ID.
  Use when you found a relevant turn_id and need the complete content.

answer(evidence: str)
  TERMINAL. Call this when you have enough information to answer.
  Pass the collected evidence as a formatted string. The system will
  use this evidence to generate the final answer."""


SYSTEM_PROMPT = """\
You are a memory search agent. Your job is to find information in a user's
conversation history to answer their question.

{tool_descriptions}

RULES:
- Think step-by-step about what information you need.
- Start with the most specific search (entities or keywords from the question).
- If initial search doesn't find enough, try DIFFERENT search terms or tools.
  Previously-returned turns are automatically excluded, so repeating the same
  query will find NEW results, but using different keywords is more effective.
- Use search_entities for WHO/WHAT/WHERE, then get_related to explore connections.
- Use read_turn to read full text when a snippet looks promising.
- Call answer() once you have sufficient evidence. Don't over-search.
- Maximum {max_steps} tool calls allowed. Budget wisely.
- IMPORTANT: Your answer() evidence should contain the specific facts/numbers
  from the conversation, not guesses. Only state what the data shows.
- Each Thought/Action must be on its own line.

FORMAT (strict):
Thought: <your reasoning about what to search for next>
Action: <tool_name>(<arguments>)

After each Action, you'll see an Observation with the results.
When ready, use: Action: answer(<your collected evidence>)"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Single tool invocation in the trace."""
    tool: str
    args: str
    result: Any
    thought: str
    elapsed_ms: float = 0.0


@dataclass
class RetrievalTrace:
    """Full trace of an agentic retrieval session."""
    question: str
    steps: list[ToolCall] = field(default_factory=list)
    evidence: str = ""
    total_ms: float = 0.0
    terminated_by: str = ""  # "answer" | "budget" | "error"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "steps": [
                {"tool": s.tool, "args": s.args, "thought": s.thought,
                 "result_preview": str(s.result)[:200], "elapsed_ms": s.elapsed_ms}
                for s in self.steps
            ],
            "evidence": self.evidence,
            "total_ms": self.total_ms,
            "terminated_by": self.terminated_by,
            "n_steps": len(self.steps),
        }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class ToolKit:
    """Executes tool calls against the memory infrastructure."""

    def __init__(
        self,
        bm25_index=None,       # BM25Index
        entity_bm25=None,      # LayeredMemory._entity_bm25
        entity_graph=None,     # EntityGraph
        memory_store=None,     # list of MemoryEntry
        event_db=None,         # EventDB
        temporal_fn=None,      # callable(query) -> list[dict]
    ):
        self._bm25 = bm25_index
        self._entity_bm25 = entity_bm25
        self._entity_graph = entity_graph
        self._memory_store = memory_store or []
        self._event_db = event_db
        self._temporal_fn = temporal_fn
        # Track seen turn IDs to avoid returning duplicates across calls
        self._seen_turns: set[int] = set()

    def reset_seen(self):
        """Reset seen turns for a new query."""
        self._seen_turns.clear()

    def execute(self, tool: str, args: str) -> Any:
        dispatch = {
            "search_memory": self._search_memory,
            "search_entities": self._search_entities,
            "get_related": self._get_related,
            "search_temporal": self._search_temporal,
            "read_turn": self._read_turn,
            "answer": self._answer,
        }
        fn = dispatch.get(tool)
        if fn is None:
            return {"error": f"Unknown tool: {tool}"}
        return fn(args)

    def _search_memory(self, query: str) -> list[dict]:
        if self._bm25 is None:
            return [{"error": "BM25 index not available"}]
        # Overfetch to compensate for dedup filtering
        results = self._bm25.query(query, top_k=20)
        out = []
        for doc_id, score in results:
            if doc_id in self._seen_turns:
                continue
            if doc_id < len(self._memory_store):
                entry = self._memory_store[doc_id]
                text = entry.text if hasattr(entry, "text") else str(entry)
                out.append({
                    "turn_id": doc_id,
                    "text": text[:300],
                    "score": round(score, 4),
                })
                self._seen_turns.add(doc_id)
            if len(out) >= 10:
                break
        return out

    def _search_entities(self, query: str) -> list[dict]:
        if self._entity_bm25 is None or self._entity_bm25._n == 0:
            return [{"error": "Entity index not available"}]
        hits = self._entity_bm25.query(query, top_k=10)
        out = []
        for canonical, turns, score in hits:
            node = None
            if self._entity_graph:
                node = self._entity_graph.entities.get(canonical)
            facets = {}
            if node and hasattr(node, "facets"):
                facets = {k: v for k, v in node.facets.items() if v}
            out.append({
                "entity": canonical,
                "facets": facets,
                "turns": sorted(turns)[:10],
                "score": round(score, 4),
            })
        return out

    def _get_related(self, entity_name: str) -> dict:
        if self._entity_graph is None:
            return {"error": "Entity graph not available"}
        norm = entity_name.strip().lower()
        node = self._entity_graph.entities.get(norm)
        if node is None:
            # Try fuzzy: check if norm is substring of any entity
            for k, v in self._entity_graph.entities.items():
                if norm in k or k in norm:
                    node = v
                    norm = k
                    break
        if node is None:
            return {"error": f"Entity '{entity_name}' not found in graph"}

        # Properties from facets
        props = {k: v for k, v in node.facets.items() if v} if hasattr(node, "facets") else {}

        # Co-mentions: entities that share turns
        co_mentions = []
        for other_name, other_node in self._entity_graph.entities.items():
            if other_name == norm:
                continue
            shared = node.mention_turns & other_node.mention_turns
            if shared:
                co_mentions.append({
                    "entity": other_name,
                    "shared_turns": sorted(shared)[:5],
                })
        co_mentions.sort(key=lambda x: -len(x["shared_turns"]))

        return {
            "entity": norm,
            "properties": props,
            "mention_turns": sorted(node.mention_turns)[:10],
            "co_mentions": co_mentions[:10],
        }

    def _search_temporal(self, query: str) -> list[dict]:
        if self._event_db is not None:
            try:
                results = self._event_db.search(query, top_k=10)
                return [
                    {"turn_id": r.get("turn_id", -1),
                     "text": r.get("text", "")[:300],
                     "date": r.get("date", ""),
                     "score": round(r.get("score", 0), 4)}
                    for r in results
                ]
            except Exception:
                pass
        if self._temporal_fn is not None:
            return self._temporal_fn(query)
        # Fallback: search memory with temporal terms
        return self._search_memory(query)

    def _read_turn(self, args: str) -> dict:
        try:
            turn_id = int(args.strip())
        except (ValueError, TypeError):
            return {"error": f"Invalid turn_id: {args}"}
        if turn_id < 0 or turn_id >= len(self._memory_store):
            return {"error": f"Turn {turn_id} out of range (0-{len(self._memory_store)-1})"}
        entry = self._memory_store[turn_id]
        text = entry.text if hasattr(entry, "text") else str(entry)
        role = entry.role if hasattr(entry, "role") else "unknown"
        return {"turn_id": turn_id, "text": text, "role": role}

    def _answer(self, evidence: str) -> dict:
        return {"terminal": True, "evidence": evidence}


# ---------------------------------------------------------------------------
# ReAct parser
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(
    r"Action:\s*(\w+)\((.+?)\)\s*$",
    re.MULTILINE | re.DOTALL,
)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)


def parse_react_output(text: str) -> tuple[str, str, str]:
    """Parse LLM output into (thought, tool_name, tool_args).

    Returns ("", "", "") if parsing fails.
    """
    thought = ""
    tm = _THOUGHT_RE.search(text)
    if tm:
        thought = tm.group(1).strip()

    am = _ACTION_RE.search(text)
    if not am:
        return thought, "", ""

    tool_name = am.group(1).strip()
    raw_args = am.group(2).strip()
    # Strip surrounding quotes if present
    if (raw_args.startswith('"') and raw_args.endswith('"')) or \
       (raw_args.startswith("'") and raw_args.endswith("'")):
        raw_args = raw_args[1:-1]
    return thought, tool_name, raw_args


# ---------------------------------------------------------------------------
# Agentic Retriever
# ---------------------------------------------------------------------------

class AgenticRetriever:
    """ReAct-based iterative retrieval agent.

    Uses an LLM to decide which tools to call, in what order,
    building up evidence until it can answer the question.
    """

    def __init__(
        self,
        toolkit: ToolKit,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str = "gpt-4o-mini",
        max_steps: int = 6,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self._toolkit = toolkit
        self._llm_base_url = llm_base_url or "https://openrouter.ai/api/v1"
        self._llm_api_key = llm_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._llm_model = llm_model
        self._max_steps = max_steps
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._llm_base_url,
                api_key=self._llm_api_key,
            )

    def retrieve(self, question: str) -> tuple[str, RetrievalTrace]:
        """Run the agentic retrieval loop.

        Returns (evidence_text, trace).
        """
        self._ensure_client()
        self._toolkit.reset_seen()
        trace = RetrievalTrace(question=question)
        t0 = time.time()

        system = SYSTEM_PROMPT.format(
            tool_descriptions=TOOL_DESCRIPTIONS,
            max_steps=self._max_steps,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        for step_i in range(self._max_steps):
            # Get LLM decision
            try:
                resp = self._client.chat.completions.create(
                    model=self._llm_model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                llm_text = resp.choices[0].message.content or ""
            except Exception as exc:
                logger.warning("LLM call failed at step %d: %s", step_i, exc)
                trace.terminated_by = "error"
                break

            thought, tool_name, tool_args = parse_react_output(llm_text)

            if not tool_name:
                # LLM didn't produce a valid action — try to extract answer from text
                logger.info("No action parsed at step %d, using text as evidence", step_i)
                trace.evidence = llm_text
                trace.terminated_by = "no_action"
                break

            # Execute tool
            t_tool = time.time()
            result = self._toolkit.execute(tool_name, tool_args)
            elapsed_ms = (time.time() - t_tool) * 1000

            tc = ToolCall(
                tool=tool_name, args=tool_args, result=result,
                thought=thought, elapsed_ms=elapsed_ms,
            )
            trace.steps.append(tc)

            # Check for terminal answer()
            if tool_name == "answer":
                trace.evidence = tool_args
                trace.terminated_by = "answer"
                break

            # Format observation and append to messages
            obs_text = _format_observation(result)
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": f"Observation: {obs_text}"})

        else:
            # Budget exhausted — collect what we have
            trace.terminated_by = "budget"
            # Use all retrieved text as evidence
            all_texts = []
            for s in trace.steps:
                if isinstance(s.result, list):
                    for item in s.result:
                        if isinstance(item, dict) and "text" in item:
                            all_texts.append(item["text"])
                elif isinstance(s.result, dict) and "text" in s.result:
                    all_texts.append(s.result["text"])
            trace.evidence = "\n---\n".join(all_texts) if all_texts else ""

        trace.total_ms = (time.time() - t0) * 1000
        return trace.evidence, trace


def _format_observation(result: Any) -> str:
    """Format tool result for the LLM observation."""
    if isinstance(result, list):
        if not result:
            return "No results found."
        # Compact JSON for lists
        return json.dumps(result, indent=None, default=str)[:2000]
    if isinstance(result, dict):
        if "error" in result:
            return f"Error: {result['error']}"
        return json.dumps(result, indent=None, default=str)[:2000]
    return str(result)[:2000]
