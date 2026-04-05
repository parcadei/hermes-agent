"""Structured event-DB query with tool-calling agent loop.

This is Chronos's core architectural insight applied to Hermes: rather than
doing similarity search (BM25, cosine, RRF), expose the memory as a set of
structured indexes the LLM can query via function calls.  The LLM iteratively
calls these tools to gather evidence, then calls ``finalize_answer`` to emit
the final response.

Tools exposed to LLM:
  * ``find_turns_by_entity(entity)`` -- inverted index lookup
  * ``find_turns_by_date_range(start_date, end_date)`` -- date-filtered turns
  * ``search_turns_by_keyword(query, top_k)`` -- BM25 full-text
  * ``read_turns(indices)`` -- fetch turn text by indices
  * ``finalize_answer(answer)`` -- terminate loop with final answer

Cost: up to ``max_hops`` LLM calls per question (tool calls + final answer).
Bounded at ~4 calls per question in practice.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime
from typing import Any

from mabench.configs import LLMConfig


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "find_turns_by_entity",
            "description": (
                "Find conversation turns mentioning a specific entity "
                "(person name, place, product, concept). Returns turn "
                "indices sorted by recency. Case-insensitive substring match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to search for (e.g. 'John', 'Paris')",
                    },
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_turns_by_date_range",
            "description": (
                "Find conversation turns occurring within a date range. "
                "Returns turn indices for turns whose Chat Time falls between "
                "start_date and end_date (inclusive). Use YYYY-MM-DD format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_turns_by_keyword",
            "description": (
                "Full-text BM25 search across all conversation turns. "
                "Returns the top_k most relevant turn indices. Use this "
                "when you need to find turns mentioning a phrase or topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or phrase to search for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Embedding-based semantic search over conversation turns. "
                "Finds turns that are topically/semantically similar to the "
                "query text (including paraphrases). Use this when entity/"
                "keyword search miss, or to find chunks about a CONCEPT."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query text",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_date_diff",
            "description": (
                "Compute the difference between two dates in days, weeks, and "
                "months. Use this for temporal-reasoning questions like "
                "'how many days/weeks/months ago did X happen' or 'which "
                "happened first, X or Y' or 'how long between X and Y'. "
                "Both dates must be in YYYY-MM-DD format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_a": {
                        "type": "string",
                        "description": "First date in YYYY-MM-DD format",
                    },
                    "date_b": {
                        "type": "string",
                        "description": "Second date in YYYY-MM-DD format",
                    },
                },
                "required": ["date_a", "date_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_turns",
            "description": (
                "Read the full text of conversation turns by their indices. "
                "Use this after finding candidate turns with other tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Turn indices to read (max 15)",
                    },
                },
                "required": ["indices"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": (
                "Emit the final answer to the user's question. Call this when "
                "you have gathered enough evidence. Keep the answer concise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Concise final answer",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


SYSTEM_PROMPT = (
    "You are a memory retrieval agent. Answer the user's question by "
    "querying conversational memory using the provided tools. Strategy:\n"
    "1. Identify key entities (names, places, dates) in the question.\n"
    "2. Call find_turns_by_entity or find_turns_by_date_range to locate "
    "relevant turns.\n"
    "3. If those miss, call semantic_search (embedding similarity) or "
    "search_turns_by_keyword (BM25) to find topically-similar chunks.\n"
    "4. Call read_turns on the most promising indices (max 10 at a time).\n"
    "5. For temporal arithmetic (days/weeks/months ago, which came first, "
    "how long between), call compute_date_diff after finding the event dates.\n"
    "6. Once you have evidence, call finalize_answer with a concise answer.\n"
    "You have up to {max_hops} tool calls. Be efficient."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date | None:
    """Parse 'YYYY-MM-DD' into a date.  Returns None on failure."""
    if not s:
        return None
    m = re.match(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _safe_json_loads(s: str) -> dict[str, Any]:
    """Best-effort JSON parse of LLM tool arguments."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class StructuredQueryAgent:
    """Tool-calling agent that queries memory via structured indexes.

    Wraps an LLM client (OpenAI-compatible) and a set of memory indexes.
    At query time, runs an iterative tool-calling loop: the LLM chooses
    which index to query, receives results, and refines until it calls
    ``finalize_answer``.
    """

    def __init__(
        self,
        memory_store: list,
        entity_inv_idx: dict[str, set[int]],
        event_calendar: list[dict],
        bm25_index,
        llm_config: LLMConfig | None = None,
        max_hops: int = 3,
        max_turns_per_tool: int = 15,
        max_read_chars: int = 6000,
        engine=None,
        scorer=None,
    ) -> None:
        """Initialize with memory references.

        :param memory_store: list of fact objects (coupled_engine.memory_store).
            Each must expose ``.text`` attribute (or be a plain string).
        :param entity_inv_idx: entity -> set of turn/chunk indices.
        :param event_calendar: list of {"turn_idx", "date", ...} dicts.
        :param bm25_index: BM25Index with ``.query(text, top_k)`` method.
        :param llm_config: LLMConfig for the agent (model, temperature).
        :param max_hops: max tool calls before forcing termination.
        :param max_turns_per_tool: max turns returned per tool call.
        :param max_read_chars: max characters returned per read_turns call.
        """
        self._memory_store = memory_store
        self._entity_inv_idx = entity_inv_idx
        self._event_calendar = event_calendar
        self._bm25 = bm25_index
        self._engine = engine
        self._scorer = scorer
        self._llm_cfg = llm_config or LLMConfig()
        self._max_hops = max_hops
        self._max_turns_per_tool = max_turns_per_tool
        self._max_read_chars = max_read_chars
        # date -> [turn_idx] index, rebuilt lazily per query (calendar
        # is populated during ingestion, after this object is constructed).
        self._date_index: dict[date, list[int]] = {}
        self._date_index_len: int = -1  # invalidate cache
        # Lazy OpenAI client
        self._client = None

    def _ensure_date_index(self) -> None:
        """Rebuild the date index if the event_calendar has grown."""
        if len(self._event_calendar) == self._date_index_len:
            return
        self._date_index = {}
        for entry in self._event_calendar:
            d = entry.get("date")
            if isinstance(d, datetime):
                d = d.date()
            if isinstance(d, date):
                self._date_index.setdefault(d, []).append(int(entry["turn_idx"]))
        self._date_index_len = len(self._event_calendar)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_find_by_entity(self, entity: str) -> dict:
        """Regex-style substring match against entity inverted index."""
        if not entity:
            return {"indices": [], "message": "empty entity"}
        needle = entity.lower().strip()
        matching_idxs: set[int] = set()
        # Exact key match first
        if needle in self._entity_inv_idx:
            matching_idxs.update(self._entity_inv_idx[needle])
        # Substring fallback — find keys containing the needle
        if not matching_idxs:
            for key, idxs in self._entity_inv_idx.items():
                if needle in key or key in needle:
                    matching_idxs.update(idxs)
                    if len(matching_idxs) >= self._max_turns_per_tool * 3:
                        break
        idxs_sorted = sorted(matching_idxs, reverse=True)[: self._max_turns_per_tool]
        return {"indices": idxs_sorted, "count": len(matching_idxs)}

    def _tool_find_by_date_range(
        self, start_date: str, end_date: str
    ) -> dict:
        """Find turns whose Chat Time falls in [start, end]."""
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        if start is None or end is None:
            return {
                "indices": [],
                "error": "invalid date format, expected YYYY-MM-DD",
            }
        if end < start:
            start, end = end, start
        self._ensure_date_index()
        matching: list[int] = []
        for d, idxs in self._date_index.items():
            if start <= d <= end:
                matching.extend(idxs)
        matching = sorted(set(matching))[: self._max_turns_per_tool]
        return {"indices": matching, "start": start.isoformat(), "end": end.isoformat()}

    def _tool_search_by_keyword(self, query: str, top_k: int = 10) -> dict:
        """BM25 full-text search over memory store."""
        if not query or self._bm25 is None:
            return {"indices": [], "message": "empty query or no bm25"}
        top_k = min(max(1, int(top_k)), self._max_turns_per_tool)
        hits = self._bm25.query(query, top_k=top_k)
        indices = [int(idx) for idx, _ in hits]
        return {"indices": indices, "query": query}

    def _tool_semantic_search(self, query: str, top_k: int = 10) -> dict:
        """Cosine-similarity search over the embedding store."""
        if not query or self._engine is None or self._scorer is None:
            return {"indices": [], "message": "semantic search unavailable"}
        top_k = min(max(1, int(top_k)), self._max_turns_per_tool)
        q_emb = self._scorer.embed(query)
        hits = self._engine.query(q_emb, top_k=top_k)
        indices = [int(h["index"]) for h in hits if isinstance(h, dict) and "index" in h]
        return {"indices": indices, "query": query}

    def _tool_compute_date_diff(self, date_a: str, date_b: str) -> dict:
        """Return days/weeks/months between two dates (signed: a - b)."""
        d_a = _parse_date(date_a)
        d_b = _parse_date(date_b)
        if d_a is None or d_b is None:
            return {
                "error": "invalid date format, expected YYYY-MM-DD",
                "date_a": date_a,
                "date_b": date_b,
            }
        delta_days = (d_a - d_b).days
        # Months: approximate calendar months (year*12+month diff)
        delta_months = (d_a.year - d_b.year) * 12 + (d_a.month - d_b.month)
        if d_a.day < d_b.day:
            delta_months -= 1 if delta_days > 0 else 0
        elif d_a.day > d_b.day:
            delta_months += 1 if delta_days < 0 else 0
        return {
            "days_between": abs(delta_days),
            "weeks_between": abs(delta_days) // 7,
            "months_between": abs(delta_months),
            "days_signed": delta_days,
            "earlier_date": date_b if delta_days > 0 else date_a,
            "later_date": date_a if delta_days > 0 else date_b,
            "same_date": delta_days == 0,
        }

    def _get_text(self, idx: int) -> str | None:
        """Fetch fact text from memory store by index."""
        if idx < 0 or idx >= len(self._memory_store):
            return None
        entry = self._memory_store[idx]
        if hasattr(entry, "text"):
            return entry.text
        if isinstance(entry, str):
            return entry
        return None

    def _tool_read_turns(self, indices: list[int]) -> dict:
        """Fetch turn texts by indices."""
        if not indices:
            return {"turns": [], "message": "no indices"}
        # Cap at 15 and bound total chars
        indices = list(indices)[:15]
        out: list[dict] = []
        char_budget = self._max_read_chars
        for idx in indices:
            text = self._get_text(int(idx))
            if text is None:
                continue
            if char_budget <= 0:
                break
            if len(text) > char_budget:
                text = text[:char_budget] + "..."
            char_budget -= len(text)
            out.append({"idx": int(idx), "text": text})
        return {"turns": out}

    def _execute_tool(self, name: str, args: dict) -> dict:
        """Dispatch tool call to implementation."""
        if name == "find_turns_by_entity":
            return self._tool_find_by_entity(args.get("entity", ""))
        if name == "find_turns_by_date_range":
            return self._tool_find_by_date_range(
                args.get("start_date", ""), args.get("end_date", "")
            )
        if name == "search_turns_by_keyword":
            return self._tool_search_by_keyword(
                args.get("query", ""), int(args.get("top_k", 10))
            )
        if name == "semantic_search":
            return self._tool_semantic_search(
                args.get("query", ""), int(args.get("top_k", 10))
            )
        if name == "compute_date_diff":
            return self._tool_compute_date_diff(
                args.get("date_a", ""), args.get("date_b", "")
            )
        if name == "read_turns":
            return self._tool_read_turns(args.get("indices", []))
        return {"error": f"unknown tool: {name}"}

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        import os

        from openai import OpenAI

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def answer(self, question: str) -> dict:
        """Run the tool-calling loop and return the final answer.

        Returns a dict with:
          * output: final answer text (or fallback if loop ran out)
          * prompt_tokens, completion_tokens: cumulative token counts
          * gen_time: wall-clock seconds
          * hops: number of tool calls made
          * tool_log: list of (tool_name, args, result_summary) for debugging
        """
        self._ensure_client()

        system_msg = SYSTEM_PROMPT.format(max_hops=self._max_hops)
        messages: list[dict] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ]

        total_prompt = 0
        total_completion = 0
        tool_log: list[dict] = []
        start = time.time()
        final_answer: str | None = None
        last_evidence: list[str] = []

        for hop in range(self._max_hops + 1):
            # Final hop forces finalize_answer (no other tools available)
            tools = TOOL_SPECS
            tool_choice: Any = "auto"
            if hop == self._max_hops:
                tools = [TOOL_SPECS[-1]]  # only finalize_answer
                tool_choice = {
                    "type": "function",
                    "function": {"name": "finalize_answer"},
                }

            response = self._client.chat.completions.create(
                model=self._llm_cfg.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self._llm_cfg.temperature,
                max_tokens=max(self._llm_cfg.max_gen_tokens, 256),
                timeout=30.0,
            )
            usage = response.usage
            total_prompt += usage.prompt_tokens
            total_completion += usage.completion_tokens

            choice = response.choices[0]
            msg = choice.message
            # Append assistant message (with tool_calls) to history
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (msg.tool_calls or [])
                    ],
                }
            )

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                # No tool called — use content as final answer
                final_answer = msg.content or ""
                break

            for tc in tool_calls:
                name = tc.function.name
                args = _safe_json_loads(tc.function.arguments)
                if name == "finalize_answer":
                    final_answer = args.get("answer", "")
                    tool_log.append(
                        {"tool": name, "args": args, "result": "FINAL"}
                    )
                    break
                result = self._execute_tool(name, args)
                # Track evidence for fallback
                if name == "read_turns":
                    for t in result.get("turns", []):
                        last_evidence.append(t.get("text", ""))
                tool_log.append(
                    {
                        "tool": name,
                        "args": args,
                        "result_summary": _summarize_result(result),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result)[:4000],
                    }
                )

            if final_answer is not None:
                break

        gen_time = time.time() - start

        # Fallback: if loop exited without finalize, synthesize from evidence
        if final_answer is None or final_answer.strip() == "":
            final_answer = _fallback_answer(last_evidence, question)

        return {
            "output": final_answer,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "gen_time": gen_time,
            "hops": len(tool_log),
            "tool_log": tool_log,
        }


# ---------------------------------------------------------------------------
# Result formatting helpers
# ---------------------------------------------------------------------------

def _summarize_result(result: dict) -> str:
    """Compact one-line summary of a tool result (for logging)."""
    if "indices" in result:
        return f"indices={len(result['indices'])}"
    if "turns" in result:
        return f"turns={len(result['turns'])}"
    return str(result)[:80]


def _fallback_answer(evidence: list[str], question: str) -> str:
    """Emit a best-effort answer when the loop didn't finalize.

    Returns the first non-empty evidence snippet truncated to 100 chars,
    or 'I don't know' when no evidence was gathered.
    """
    for snippet in evidence:
        s = snippet.strip()
        if s:
            return s[:200]
    return "I don't know."
