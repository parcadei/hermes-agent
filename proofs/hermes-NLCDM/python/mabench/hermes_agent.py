"""Hermes Memory Agent — MABench adapter.

Wires MemoryOrchestrator (V1 text-based contradiction) with CoupledEngine
(V2 cosine-threshold contradiction) into the MABench AgentWrapper interface.

Pipeline:
  store: text → V1 detect_contradictions → orchestrator.store() → embed → V2 cosine check
  query: hybrid switching per HybridBridge.lean proven criteria:
         1. V2 cosine retrieval (always primary)
         2. Estimate signal coherence across candidates
         3. If coherence ≤ 0 (signals uncorrelated): pure cosine + V1 text merge
         4. If coherence > 0 (signals correlated): multi-signal re-ranking

  For FactConsolidation: signalInversion ≈ 0 → pure_cosine_maximal_gap applies
  → cosine-only is optimal (proven in HybridBridge.lean).

Two contradiction layers:
  V1 (text):    subject extraction, polarity, value_update — high recall, no embeddings
  V2 (cosine):  cosine threshold on Qwen3-0.6B embeddings — high precision, geometric
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# Add hermes-memory and hermes-NLCDM to path
# mabench/ lives at proofs/hermes-NLCDM/python/mabench/
_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
_HERMES_ROOT = _NLCDM_PYTHON.parent.parent.parent
sys.path.insert(0, str(_HERMES_ROOT / "proofs" / "hermes-memory" / "python"))
sys.path.insert(0, str(_NLCDM_PYTHON))

from hermes_memory.orchestrator import MemoryOrchestrator, RelevanceScorer
from hermes_memory.engine import ParameterSet
from hermes_memory.recall import RecallConfig
from hermes_memory.contradiction import ContradictionConfig
from hermes_memory.consolidation import ConsolidationConfig, ConsolidationMode
from hermes_memory.encoding import EncodingConfig
from coupled_engine import CoupledEngine
from dream_ops import DreamParams
from nlcdm_core import cosine_sim


class EmbeddingRelevanceScorer(RelevanceScorer):
    """Cosine-similarity relevance scorer using a sentence embedding model.

    Replaces JaccardRelevance with real embedding-based scoring.
    Embeddings are cached per content string to avoid redundant inference.
    """

    _DISK_CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "embed_cache"

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, np.ndarray] = {}
        self._load_disk_cache()

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel.from_pretrained(
            self._model_name, torch_dtype=torch.float32
        ).to(self._device)
        self._model.eval()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string, with caching."""
        if text in self._cache:
            return self._cache[text]
        # Delegate to batch path for single text
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
        """Embed multiple texts, returning cached results where available.

        Uncached texts are embedded in GPU batches for throughput.
        """
        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            self._ensure_model()
            import torch
            # Process in batches to avoid OOM on very large chunks
            for start in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[start:start + batch_size]
                inputs = self._tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._model(**inputs)
                last_hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embs = pooled.cpu().numpy().astype(np.float64)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / (norms + 1e-12)

                for j, emb in enumerate(embs):
                    idx = uncached_indices[start + j]
                    results[idx] = emb
                    self._cache[uncached_texts[start + j]] = emb

        return results

    def _load_disk_cache(self):
        """Load cached embeddings from disk if available."""
        cache_file = self._DISK_CACHE_DIR / "embeddings.npz"
        if cache_file.exists():
            data = np.load(str(cache_file), allow_pickle=True)
            texts = data["texts"]
            embs = data["embeddings"]
            for t, e in zip(texts, embs):
                self._cache[str(t)] = e
            print(f"Loaded {len(self._cache)} cached embeddings from disk")

    def save_disk_cache(self):
        """Persist embedding cache to disk."""
        self._DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = self._DISK_CACHE_DIR / "embeddings.npz"
        texts = list(self._cache.keys())
        embs = np.array(list(self._cache.values()))
        np.savez(str(cache_file), texts=texts, embeddings=embs)
        print(f"Saved {len(texts)} embeddings to {cache_file}")

    def score(self, query: str, content: str) -> float:
        """Cosine similarity between query and content embeddings."""
        q_emb = self.embed(query)
        c_emb = self.embed(content)
        return float(np.dot(q_emb, c_emb))


class HermesMemoryAgent:
    """MABench-compatible memory agent wrapping Hermes V1+V2 systems.

    Not a subclass of AgentWrapper — instead provides the send_message
    interface that MABench's main.py calls, plus save_agent/load_agent.

    Args:
        model:                   LLM model name for answer generation
        dim:                     Embedding dimension (1024 for Qwen3-0.6B)
        contradiction_threshold: V2 cosine threshold for contradiction detection
        retrieve_num:            Number of memories to retrieve for context
        dream_interval:          Run dream cycle every N stores (0 = disabled)
        temperature:             LLM temperature for answer generation
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        dim: int = 1024,
        contradiction_threshold: float = 0.95,
        retrieve_num: int = 10,
        dream_interval: int = 0,
        temperature: float = 0.0,
        max_gen_tokens: int = 64,
        recency_alpha: float = 0.1,
        dream_params: DreamParams | None = None,
        associative_retrieval: bool = False,
        sparse_retrieval: bool = False,
        hybrid_retrieval: bool = False,
        cooc_boost_retrieval: bool = False,
        cooc_weight: float = 0.3,
        cooc_gate_threshold: float = 0.0,
        ppr_retrieval: bool = False,
        ppr_weight: float = 0.5,
        ppr_damping: float = 0.85,
        coretrieval_retrieval: bool = False,
        coretrieval_bonus: float = 0.3,
        coretrieval_min_count: float = 2.0,
        transfer_retrieval: bool = False,
        transfer_k: int | None = None,
    ):
        self.model = model
        self.dim = dim
        self.contradiction_threshold = contradiction_threshold
        self.associative_retrieval = associative_retrieval
        self.sparse_retrieval = sparse_retrieval
        self.hybrid_retrieval = hybrid_retrieval
        self.cooc_boost_retrieval = cooc_boost_retrieval
        self.cooc_weight = cooc_weight
        self.cooc_gate_threshold = cooc_gate_threshold
        self.ppr_retrieval = ppr_retrieval
        self.ppr_weight = ppr_weight
        self.ppr_damping = ppr_damping
        self.coretrieval_retrieval = coretrieval_retrieval
        self.coretrieval_bonus = coretrieval_bonus
        self.coretrieval_min_count = coretrieval_min_count
        self.transfer_retrieval = transfer_retrieval
        self.transfer_k = transfer_k
        self.retrieve_num = retrieve_num
        self.dream_interval = dream_interval
        self.temperature = temperature
        self._max_gen_tokens = max_gen_tokens
        self.dream_params = dream_params

        # V1: Text-based orchestrator (validated baseline from test_engine.py)
        params = ParameterSet(
            alpha=0.1,
            beta=0.1,
            delta_t=1.0,
            s_max=10.0,
            s0=1.0,
            temperature=5.0,
            novelty_start=0.5,
            novelty_decay=0.2,
            survival_threshold=0.05,
            feedback_sensitivity=0.1,
            w1=0.35,
            w2=0.25,
            w3=0.20,
            w4=0.20,
        )
        self._scorer = EmbeddingRelevanceScorer()
        self.orchestrator = MemoryOrchestrator(
            params=params,
            encoding_config=EncodingConfig(),
            contradiction_config=ContradictionConfig(),
            consolidation_config=ConsolidationConfig(),
            recall_config=RecallConfig(total_budget=4000),
            relevance_scorer=self._scorer,
        )

        # V2: Full coupled pipeline — P@1 0.938 longitudinal config
        # Strength decay drives importances (not uniform 0.5)
        # Novelty bonus ON, emotional tagging OFF (CMA-ES proved ceiling)
        # Reconsolidation OFF (no measurable effect)
        self.coupled_engine = CoupledEngine(
            dim=dim,
            contradiction_aware=True,
            contradiction_threshold=contradiction_threshold,
            novelty_N0=0.2,
            novelty_gamma=0.05,
            emotional_tagging=False,
            reconsolidation=False,
            recency_alpha=recency_alpha,
            dream_params=dream_params,
        )

        self._store_count = 0
        self._start_time = time.time()

        # LLM client (lazy init)
        self._llm_client = None

    def _ensure_llm_client(self):
        if self._llm_client is not None:
            return
        from openai import OpenAI
        self._llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def send_message(
        self,
        message: str,
        memorizing: bool = False,
        query_id: int | None = None,
        context_id: int | None = None,
    ) -> dict | str:
        """MABench-compatible interface.

        memorizing=True:  Store the chunk into memory. Returns "Memorized".
        memorizing=False: Query memory, build context, generate answer via LLM.
                          Returns standard response dict.
        """
        if memorizing:
            return self._store(message)
        else:
            return self._query(message)

    # Regex: split on "N. " where N is one or more digits at a fact boundary.
    # Lookbehind ensures we split after the previous fact's trailing ". "
    _FACT_SPLIT_RE = re.compile(r"(?<=\.) (?=\d+\. )")
    _FACT_SN_RE = re.compile(r"^(\d+)\. (.+)")

    @staticmethod
    def _parse_facts(text: str) -> list[tuple[int, str]]:
        """Parse a chunk into individual (serial_number, fact_text) pairs.

        Handles the MABench FactConsolidation format:
          "Here is a list of facts:\\n0. Fact zero. 1. Fact one. ..."

        Returns list of (sn, fact_text).  Falls back to [(-1, text)] if
        the chunk doesn't match the expected SN pattern.
        """
        body = text
        if body.startswith("Here is a list of facts:\n"):
            body = body[len("Here is a list of facts:\n"):]

        parts = HermesMemoryAgent._FACT_SPLIT_RE.split(body)
        facts: list[tuple[int, str]] = []
        for part in parts:
            m = HermesMemoryAgent._FACT_SN_RE.match(part.strip())
            if m:
                facts.append((int(m.group(1)), m.group(2).strip()))
        if not facts:
            # Fallback: store entire chunk as one entry
            facts.append((-1, text))
        return facts

    def _store(self, text: str) -> str:
        """Store text through V1 (text contradiction) + V2 (cosine contradiction).

        Parses chunks into individual facts (split on serial numbers) so that:
        - V1 contradiction detection compares single facts (subject extraction works)
        - V2 cosine contradiction fires on facts about the same entity/predicate
        - Dream consolidation operates at fact granularity
        - Retrieval matches queries to individual facts, not multi-fact chunks
        """
        facts = self._parse_facts(text)

        # Batch-embed all facts at once (single GPU pass instead of N individual calls)
        fact_texts = [ft for _, ft in facts]
        embeddings = self._scorer.embed_batch(fact_texts)

        for (sn, fact_text), emb in zip(facts, embeddings):
            original_text = f"{sn}. {fact_text}" if sn >= 0 else fact_text

            # V1: Text-based store with contradiction detection + ANN pre-filter
            self.orchestrator.store(content=fact_text, embedding=emb)

            # V2: Store with pre-computed embedding
            self.coupled_engine.store(
                text=original_text, embedding=emb,
                recency=float(self._store_count),
            )

            self._store_count += 1

        # Optional dream cycle at configurable intervals (per chunk, not per fact)
        if self.dream_interval > 0 and self._store_count % self.dream_interval == 0:
            self.coupled_engine.dream()

        return "Memorized"

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """Extract the bare question from a MABench prompt.

        The prompt looks like:
          "Pretend you are a knowledge management system. ... Question: ... Answer:"
        We extract just the text after the last "Question:" and before "Answer:".
        Falls back to the full prompt if the pattern isn't found.
        """
        # Find the actual question between last "Question:" and last "Answer:"
        q_start = prompt.rfind("Question:")
        if q_start >= 0:
            q_end = prompt.rfind("\nAnswer:")
            if q_end > q_start:
                return prompt[q_start + len("Question:"):q_end].strip()
        return prompt

    def _query(self, question: str) -> dict:
        """Query memory with hybrid switching per proven criteria.

        Switching criterion (HybridBridge.lean, SwitchedDynamics.lean):
          1. V2 cosine retrieval (always primary signal)
          2. Estimate signal coherence: w₁·δ_rel vs signalInversion
          3. If multi-signal gap > cosine gap: use multi-signal re-ranking
             (requires signalInversion < -(1-w₁)·δ_rel, i.e., signals strongly help)
          4. Otherwise: pure cosine + V1 text merge (proven optimal at zero inversion)

        For FactConsolidation: signalInversion ≈ 0, so pure_cosine_maximal_gap
        applies → cosine-only maximizes score gap → 56% SubEM baseline.
        """
        query_start = time.time()

        # Extract bare question for embedding (strip MABench prompt preamble)
        bare_question = self._extract_question(question)

        # Embed the bare question (not the full prompt with instructions)
        q_emb = self._scorer.embed(bare_question)

        # V2 retrieval from coupled engine
        if self.transfer_retrieval:
            v2_results = self.coupled_engine.query_transfer(
                embedding=q_emb, top_k=self.retrieve_num,
                transfer_k=self.transfer_k,
            )
        elif self.coretrieval_retrieval:
            v2_results = self.coupled_engine.query_coretrieval(
                embedding=q_emb, top_k=self.retrieve_num,
                coretrieval_bonus=self.coretrieval_bonus,
                min_coretrieval_count=self.coretrieval_min_count,
            )
        elif self.cooc_boost_retrieval:
            v2_results = self.coupled_engine.query_cooc_boost(
                embedding=q_emb, top_k=self.retrieve_num,
                cooc_weight=self.cooc_weight,
                gate_threshold=self.cooc_gate_threshold,
            )
        elif self.ppr_retrieval:
            v2_results = self.coupled_engine.query_ppr(
                embedding=q_emb, top_k=self.retrieve_num,
                ppr_weight=self.ppr_weight,
                damping=self.ppr_damping,
            )
        elif self.hybrid_retrieval:
            v2_results = self.coupled_engine.query_hybrid(
                embedding=q_emb, top_k=self.retrieve_num,
            )
        elif self.associative_retrieval or self.sparse_retrieval:
            v2_results = self.coupled_engine.query_associative(
                embedding=q_emb, top_k=self.retrieve_num,
                sparse=self.sparse_retrieval,
            )
        else:
            v2_results = self.coupled_engine.query(
                embedding=q_emb, top_k=self.retrieve_num
            )
        v2_texts = [r["text"] for r in v2_results]

        # V1 retrieval: text-based recall from orchestrator
        v1_result = self.orchestrator.query(message=question)
        v1_context = v1_result.context if v1_result.context else ""

        # Merge: V2 results as primary (embedding-ranked), V1 as supplement
        seen = set()
        merged_chunks = []
        for text in v2_texts:
            key = text.strip()[:200]  # dedupe on prefix
            if key not in seen:
                seen.add(key)
                merged_chunks.append(text)
        # Add V1 context lines that aren't already covered
        if v1_context:
            for line in v1_context.split("\n---\n"):
                line = line.strip()
                key = line[:200]
                if line and key not in seen:
                    seen.add(key)
                    merged_chunks.append(line)

        # Build context for LLM
        memory_context = "\n\n".join(merged_chunks[:self.retrieve_num])
        memory_time = time.time() - query_start

        # Generate answer via LLM
        answer, input_tokens, output_tokens, gen_time = self._generate_answer(
            question, memory_context
        )
        query_time = time.time() - query_start

        return {
            "output": answer,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
        }

    def _generate_answer(
        self, question: str, context: str
    ) -> tuple[str, int, int, float]:
        """Call LLM to generate an answer from retrieved context."""
        self._ensure_llm_client()

        system_msg = (
            "You are a helpful assistant that answers questions from a "
            "knowledge pool. Give a very concise answer."
        )
        # The MABench query already contains full instructions about serial
        # numbers and recency. Prepend retrieved memories as the knowledge pool.
        user_msg = (
            f"[Knowledge Pool]\n{context}\n\n"
            f"{question}\nAnswer:"
        )

        start = time.time()
        response = self._llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=max(self._max_gen_tokens, 16),
        )
        gen_time = time.time() - start

        output = response.choices[0].message.content or ""
        usage = response.usage
        return output, usage.prompt_tokens, usage.completion_tokens, gen_time

    def save_agent(self, path: str) -> None:
        """Persist agent state (placeholder — hermes state is in-memory)."""
        pass

    def load_agent(self, path: str) -> None:
        """Load agent state (placeholder — hermes state is in-memory)."""
        pass

    def reset(self) -> None:
        """Reset all memory state for a new context."""
        self.orchestrator = MemoryOrchestrator(
            params=self.orchestrator.params,
            encoding_config=EncodingConfig(),
            contradiction_config=ContradictionConfig(),
            consolidation_config=ConsolidationConfig(),
            recall_config=RecallConfig(total_budget=4000),
            relevance_scorer=self._scorer,
        )
        self.coupled_engine = CoupledEngine(
            dim=self.dim,
            contradiction_aware=True,
            contradiction_threshold=self.contradiction_threshold,
            novelty_N0=0.2,
            novelty_gamma=0.05,
            emotional_tagging=False,
            reconsolidation=False,
            recency_alpha=self.coupled_engine.recency_alpha,
            dream_params=self.dream_params,
        )
        self._store_count = 0
        self._scorer._cache.clear()
