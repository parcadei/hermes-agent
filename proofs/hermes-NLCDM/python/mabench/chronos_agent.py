"""Chronos-style ReAct agent for iterative memory retrieval.

Reverse-engineered from arXiv 2603.16862 Section 3.4. Six tools query two
calendars (turns + events) and iteratively build context until answering.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from typing import Any

from mabench.configs import LLMConfig

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

def _tool(name: str, desc: str, props: dict, required: list[str]) -> dict:
    """Build one OpenAI function-calling tool spec."""
    return {
        "type": "function",
        "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_QTK = {  # reusable query + top_k properties
    "query": {"type": "string", "description": "Natural-language query text"},
    "top_k": {"type": "integer", "description": "Number of results to return", "default": 10},
}

TOOL_SPECS: list[dict] = [
    _tool("search_turns",
          "Semantic search over conversation turns. Returns turns ranked by "
          "relevance with text snippets. Use for broad topic/concept search.",
          _QTK, ["query"]),
    _tool("query_events",
          "Look up structured facts about entities from the event database. "
          "Multi-hop resolution: direct entity lookup, then type-based "
          "resolution (e.g. 'museum' finds MoMA, Rijksmuseum). "
          "Use for specific entity or category queries.",
          {"entities": {"type": "array", "items": {"type": "string"},
                        "description": "Entity names to look up"},
           "predicates": {"type": "array", "items": {"type": "string"},
                          "description": "Optional predicate filter (e.g. ['visited', 'type'])"},
           "top_k": {"type": "integer", "description": "Max results", "default": 20}},
          ["entities"]),
    _tool("search_events_by_date",
          "Search events within a date range. Use for temporal queries like "
          "'What did I do last March?' or 'events between Jan and Feb 2024'. "
          "Returns events whose dates overlap the given range.",
          {"start_date": {"type": "string", "description": "Start date ISO format YYYY-MM-DD"},
           "end_date": {"type": "string", "description": "End date ISO format YYYY-MM-DD"},
           "top_k": {"type": "integer", "description": "Max results", "default": 50}},
          ["start_date", "end_date"]),
    _tool("finalize_answer",
          "Emit the final answer to the user's question. Call when you have "
          "gathered enough evidence. Keep the answer concise and factual.",
          {"answer": {"type": "string", "description": "Concise final answer"}},
          ["answer"]),
]

CHRONOS_SYSTEM_PROMPT: str = (
    "You are a memory retrieval agent with access to a user's conversation "
    "history. Answer the question using ONLY the evidence found via your tools. "
    "{question_date_line}"
    "Strategy:\n"
    "1. Start with the pre-retrieved context below (if any).\n"
    "2. Use query_events to look up specific entities or categories of interest.\n"
    "3. Use search_events_by_date for temporal queries (date ranges).\n"
    "4. Use search_turns for broad semantic/topic search when entity lookup misses.\n"
    "5. Once confident, call finalize_answer with a concise answer.\n\n"
    "You have up to {max_hops} tool calls. Be efficient.\n"
    "If the pre-retrieved context already answers the question, finalize immediately."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(s: str) -> dict[str, Any]:
    """Best-effort JSON parse of LLM tool arguments."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


def _summarize_result(result: dict) -> str:
    """Compact one-line summary of a tool result (for logging)."""
    for key in ("indices", "turns", "results"):
        if key in result:
            return f"{key}={len(result[key])}"
    return str(result)[:80]


def _fallback_answer(evidence: list[str], question: str) -> str:
    """Best-effort answer when the loop didn't finalize."""
    for snippet in evidence:
        s = snippet.strip()
        if s:
            return s[:200]
    return "I don't know."


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ChronosAgent:
    """Chronos-style ReAct agent with 4 tools for iterative memory retrieval.

    Tools: search_turns, query_events, search_events_by_date, finalize_answer.
    Graph-based entity+type resolution replaces vector search for events.
    """

    def __init__(
        self,
        memory_store: list,
        event_index: Any | None = None,
        event_db: Any | None = None,
        engine: Any | None = None,
        scorer: Any | None = None,
        bm25_index: Any | None = None,
        llm_config: LLMConfig | None = None,
        max_hops: int = 5,
        max_results: int = 15,
        max_read_chars: int = 6000,
        pre_context: str = "",
        retrieval_guidance: str = "",
        reranker: Any | None = None,
        original_question: str = "",
        question_date: str = "",
    ) -> None:
        self._memory_store = memory_store
        self._event_index = event_index
        self._event_db = event_db
        self._engine = engine
        self._scorer = scorer
        self._bm25 = bm25_index
        self._llm_cfg = llm_config or LLMConfig()
        self._max_hops = max_hops
        self._max_results = max_results
        self._max_read_chars = max_read_chars
        self._pre_context = pre_context
        self._retrieval_guidance = retrieval_guidance
        self._reranker = reranker
        self._original_question = original_question
        self._question_date = question_date
        self._client: Any | None = None

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_search_turns(self, query: str, top_k: int = 10) -> dict:
        """Vector similarity search over conversation turns.

        Paper Section 3.4: reranking is applied to the ORIGINAL question,
        not the agent's reformulated query. Overfetch 100, rerank, take top_k.
        Returns indices AND text snippets so the agent can read results.
        """
        if not query or self._scorer is None or self._engine is None:
            return {"indices": [], "turns": [], "message": "search unavailable"}
        top_k = min(max(1, int(top_k)), self._max_results)
        # Overfetch for reranking (paper: top-100 → rerank → top-k)
        fetch_k = 100 if self._reranker else top_k
        q_emb = self._scorer.embed(query)
        hits = self._engine.query(q_emb, top_k=fetch_k)
        if self._reranker and len(hits) > top_k:
            # Rerank with original question, not agent's reformulated query
            rerank_q = self._original_question or query
            rerank_input = []
            for h in hits:
                if isinstance(h, dict) and "index" in h:
                    idx = int(h["index"])
                    text = self._get_text(idx) or ""
                    rerank_input.append({"text": text, "score": float(h.get("score", 0)), "index": idx})
            reranked = self._reranker.rerank(rerank_q, rerank_input, top_k=top_k)
            indices = [r["index"] for r in reranked]
        else:
            indices = [int(h["index"]) for h in hits if isinstance(h, dict) and "index" in h]
        indices = indices[:top_k]
        # Build turns with text snippets for the agent to read
        turns = []
        for idx in indices:
            text = self._get_text(idx)
            if text:
                turns.append({"idx": idx, "text": text[:self._max_read_chars]})
        return {"indices": indices, "turns": turns, "query": query}

    def _tool_query_events(
        self, entities: list[str], predicates: list[str] | None = None, top_k: int = 20,
    ) -> dict:
        """Multi-hop graph lookup over EventDB.

        Resolution pipeline:
          1. Direct entity lookup in EventDB
          2. Type-based resolution if direct lookup misses
          3. Predicate filter (optional)
        """
        if self._event_db is None:
            return {"results": [], "message": "event database unavailable"}
        if not entities:
            return {"results": [], "message": "no entities provided"}

        top_k = min(max(1, int(top_k)), self._max_results)

        # Step 1: Direct entity lookup
        hits = self._event_db.lookup(entities, top_k=top_k)

        # Step 2: Type-based resolution (when direct lookup misses)
        if not hits:
            seen: dict[int, float] = {}
            for ent in entities:
                for cid, score in self._event_db.lookup_by_type(ent, top_k=top_k):
                    if cid not in seen or score > seen[cid]:
                        seen[cid] = score
            hits = sorted(seen.items(), key=lambda x: -x[1])[:top_k]

        # Step 3: Predicate filter
        if predicates and hits:
            pred_chunks: set[int] = set()
            for pred in predicates:
                for cid, _ in self._event_db.lookup_by_predicate(pred):
                    pred_chunks.add(cid)
            hits = [(cid, score) for cid, score in hits if cid in pred_chunks]

        # Get facts for result chunks
        chunk_ids = [cid for cid, _ in hits[:top_k]]
        facts_map = self._event_db.get_facts_for_chunks(chunk_ids)

        return {"results": [
            {"chunk_id": cid, "score": round(score, 4), "facts": facts_map.get(cid, [])}
            for cid, score in hits[:top_k]
        ]}

    def _tool_search_events_by_date(
        self, start_date: str, end_date: str, top_k: int = 50
    ) -> dict:
        """Date-range search over extracted events.

        Paper Section 3.1/4.3: temporal filtering retrieves events within
        specified time ranges. -14.7 pts ablation without date filter.
        """
        if self._event_index is None:
            return {"results": [], "message": "no event index"}
        try:
            sd = date.fromisoformat(start_date)
            ed = date.fromisoformat(end_date)
        except (ValueError, TypeError):
            return {"results": [], "message": f"invalid date format: {start_date} / {end_date}"}
        hits = self._event_index.search_by_date(sd, ed, top_k=top_k)
        return {"results": [{"chunk_id": cid, "text": txt} for cid, txt in hits], "count": len(hits)}

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

    def _execute_tool(self, name: str, args: dict) -> dict:
        """Dispatch tool call to the correct implementation."""
        if name == "search_turns":
            return self._tool_search_turns(args.get("query", ""), int(args.get("top_k", 10)))
        if name == "query_events":
            return self._tool_query_events(
                args.get("entities", []),
                args.get("predicates"),
                int(args.get("top_k", 20)),
            )
        if name == "search_events_by_date":
            return self._tool_search_events_by_date(
                args.get("start_date", ""), args.get("end_date", ""),
                int(args.get("top_k", 50)))
        if name == "finalize_answer":
            return {"answer": args.get("answer", "")}
        return {"error": f"unknown tool: {name}"}

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        """Lazy-initialize the OpenAI client."""
        if self._client is not None:
            return
        import os
        from openai import OpenAI
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    # ------------------------------------------------------------------
    # Main ReAct loop
    # ------------------------------------------------------------------

    def answer(self, question: str) -> dict:
        """Run the ReAct tool-calling loop. Returns dict with:
        output, prompt_tokens, completion_tokens, gen_time, hops, tool_log.
        """
        self._ensure_client()

        date_line = ""
        if self._question_date:
            date_line = f"The question was asked on {self._question_date}. Use this to resolve relative dates.\n"
        system = CHRONOS_SYSTEM_PROMPT.format(
            max_hops=self._max_hops, question_date_line=date_line,
        )
        if self._pre_context:
            system += "\n\n## Pre-Retrieved Context\n" + self._pre_context
        if self._retrieval_guidance:
            system += "\n\n## Retrieval Guidance\n" + self._retrieval_guidance

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        total_prompt = 0
        total_completion = 0
        tool_log: list[dict] = []
        start = time.time()
        final_answer: str | None = None
        last_evidence: list[str] = []

        for hop in range(self._max_hops + 1):
            tools = TOOL_SPECS
            tool_choice: Any = "auto"
            if hop == self._max_hops:
                tools = [TOOL_SPECS[-1]]
                tool_choice = {"type": "function", "function": {"name": "finalize_answer"}}

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

            msg = response.choices[0].message
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in (msg.tool_calls or [])
                ],
            })

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                final_answer = msg.content or ""
                break

            for tc in tool_calls:
                name = tc.function.name
                args = _safe_json_loads(tc.function.arguments)
                if name == "finalize_answer":
                    final_answer = args.get("answer", "")
                    tool_log.append({"tool": name, "args": args, "result": "FINAL"})
                    break
                result = self._execute_tool(name, args)
                if name == "query_events":
                    for r in result.get("results", []):
                        for fact in r.get("facts", []):
                            last_evidence.append(fact)
                elif name == "search_turns":
                    for t in result.get("turns", []):
                        last_evidence.append(t.get("text", ""))
                tool_log.append({"tool": name, "args": args,
                                 "result_summary": _summarize_result(result)})
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(result)[:8000],
                })

            if final_answer is not None:
                break

        gen_time = time.time() - start
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
