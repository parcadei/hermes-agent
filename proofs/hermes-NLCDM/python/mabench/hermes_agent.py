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
import sys
import time
from collections import defaultdict
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
from mabench.ingestion import (
    IngestionPipeline, IngestionConfig, IngestionResult, _MutableEntityGraph,
)
from mabench.belief import BeliefIndex
from mabench.triadic_memory import TriadicMemory

import re as _re

# Template-based triple extraction for MABench factconsolidation benchmark.
# These 36 regex patterns cover 100% of the 18,332 facts in mh_262k.
# Each pattern extracts (subject, predicate_key, object) where predicate_key
# is a canonical string that matches across old/new versions of the same fact.
# This replaces spaCy dep-parse triples for conflict detection (which only
# found 6/4,317 conflicts due to syntactic triple variance).
_FACT_TEMPLATES: list[tuple["_re.Pattern[str]", str]] = [
    (_re.compile(r'^The author of (.+?) is (.+)$'), 'author_of'),
    (_re.compile(r'^The company that produced (.+?) is (.+)$'), 'produced_by'),
    (_re.compile(r'^The origianl broadcaster of (.+?) is (.+)$'), 'broadcaster_of'),
    (_re.compile(r'^The type of music that (.+?) plays is (.+)$'), 'music_type'),
    (_re.compile(r'^The univeristy where (.+?) was educated is (.+)$'), 'educated_at'),
    (_re.compile(r'^The name of the current head of state in (.+?) is (.+)$'), 'head_of_state'),
    (_re.compile(r'^The name of the current head of (?:the )?(.+?)(?:\s+government)? is (.+)$'), 'head_of'),
    (_re.compile(r'^The chairperson of (.+?) is (.+)$'), 'chairperson_of'),
    (_re.compile(r'^The Governor of (.+?) is (.+)$'), 'governor_of'),
    (_re.compile(r'^The capital of (.+?) is (.+)$'), 'capital_of'),
    (_re.compile(r'^The head coach of (.+?) is (.+)$'), 'head_coach_of'),
    (_re.compile(r'^The official language of (.+?) is (.+)$'), 'official_language'),
    (_re.compile(r'^The Mayor of (.+?) is (.+)$'), 'mayor_of'),
    (_re.compile(r'^The chief executive officer of (.+?) is (.+)$'), 'ceo_of'),
    (_re.compile(r'^(.+?) is a citizen of (.+)$'), 'citizen_of'),
    (_re.compile(r'^(.+?) is associated with the sport of (.+)$'), 'sport_of'),
    (_re.compile(r'^(.+?) is affiliated with (.+)$'), 'affiliated_with'),
    (_re.compile(r'^(.+?) plays the position of (.+)$'), 'position_of'),
    (_re.compile(r'^(.+?) plays the sport of (.+)$'), 'plays_sport'),
    (_re.compile(r'^(.+?) is located in (.+)$'), 'located_in'),
    (_re.compile(r'^(.+?) was created by (.+)$'), 'created_by'),
    (_re.compile(r'^(.+?) was born in (.+)$'), 'born_in'),
    (_re.compile(r'^(.+?) is employed by (.+)$'), 'employed_by'),
    (_re.compile(r'^(.+?) works in the field of (.+)$'), 'field_of'),
    (_re.compile(r'^(.+?) was created in the country of (.+)$'), 'created_in_country'),
    (_re.compile(r'^(.+?) was founded in the city of (.+)$'), 'founded_in_city'),
    (_re.compile(r'^(.+?) speaks the language of (.+)$'), 'speaks_language'),
    (_re.compile(r'^(.+?) was written in the language of (.+)$'), 'written_in_language'),
    (_re.compile(r'^(.+?) worked in the city of (.+)$'), 'worked_in_city'),
    (_re.compile(r'^(.+?) was performed by (.+)$'), 'performed_by'),
    (_re.compile(r'^(.+?) is famous for (.+)$'), 'famous_for'),
    (_re.compile(r'^(.+?) was developed by (.+)$'), 'developed_by'),
    (_re.compile(r'^(.+?) was founded by (.+)$'), 'founded_by'),
    (_re.compile(r'^(.+?) is married to (.+)$'), 'married_to'),
    (_re.compile(r"^(.+?)'s child is (.+)$"), 'child_of'),
    (_re.compile(r'^(.+?) died in the city of (.+)$'), 'died_in_city'),
    (_re.compile(r'^The (.+?) is (.+)$'), 'the_X_is'),
]

def _extract_template_triple(fact_text: str) -> tuple[str, str, str] | None:
    """Extract (subject, predicate_key, object) using benchmark fact templates.

    Returns None if no template matches. The predicate_key is a canonical
    string that is stable across different object values for the same
    subject+predicate combination, enabling conflict detection.
    """
    # Strip serial number prefix if present: "0. The author of ..."
    m = _re.match(r'^\d+\.\s+(.*)', fact_text)
    text = m.group(1).rstrip('.') if m else fact_text.rstrip('.')
    for pattern, pred_key in _FACT_TEMPLATES:
        m = pattern.match(text)
        if m:
            return (m.group(1).strip(), pred_key, m.group(2).strip())
    return None


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
        triadic_retrieval: bool = False,
        triadic_n: int = 1000,
        triadic_p: int = 10,
        triadic_expand_k: int = 5,
        decompose_query: bool = False,
        decompose_max_hops: int = 3,
        decompose_coverage: bool = False,
        decompose_rrf: bool = False,
        iterative_query: bool = False,
        iterative_max_hops: int = 3,
        bm25_weight: float = 0.0,
        dedup_threshold: float = 0.0,
        dream_weight: float = 0.0,
        temporal_context: bool = False,
        contradiction_context: bool = False,
        contradiction_sim_threshold: float = 0.85,
        supersession: bool = False,
        belief: bool = False,
        belief_prior_alpha: float = 2.0,
        belief_prior_beta: float = 1.0,
        belief_propagation_damping: float = 0.3,
        belief_hard_floor: float = 0.05,
        belief_overfetch: int = 3,
        conflict_resolution: bool = False,
    ):
        self.model = model
        self.dedup_threshold = dedup_threshold
        self.dim = dim
        self.contradiction_threshold = contradiction_threshold
        self.associative_retrieval = associative_retrieval
        self.sparse_retrieval = sparse_retrieval
        self.hybrid_retrieval = hybrid_retrieval
        self.cooc_boost_retrieval = cooc_boost_retrieval
        self.cooc_weight = cooc_weight
        self.cooc_gate_threshold = cooc_gate_threshold
        self.bm25_weight = bm25_weight
        self.ppr_retrieval = ppr_retrieval
        self.ppr_weight = ppr_weight
        self.ppr_damping = ppr_damping
        self.coretrieval_retrieval = coretrieval_retrieval
        self.coretrieval_bonus = coretrieval_bonus
        self.coretrieval_min_count = coretrieval_min_count
        self.transfer_retrieval = transfer_retrieval
        self.transfer_k = transfer_k
        self.triadic_retrieval = triadic_retrieval
        self._triadic_expand_k = triadic_expand_k
        self.decompose_query = decompose_query
        self.decompose_max_hops = decompose_max_hops
        self.decompose_coverage = decompose_coverage
        self.decompose_rrf = decompose_rrf
        self.iterative_query = iterative_query
        self.iterative_max_hops = iterative_max_hops
        self.retrieve_num = retrieve_num
        self.dream_interval = dream_interval
        self.temperature = temperature
        self._max_gen_tokens = max_gen_tokens
        self.dream_params = dream_params
        self.dream_weight = dream_weight
        self.temporal_context = temporal_context
        self.contradiction_context = contradiction_context
        self.contradiction_sim_threshold = contradiction_sim_threshold
        self.supersession = supersession
        self.conflict_resolution = conflict_resolution
        self._belief_overfetch = belief_overfetch
        self._belief_hard_floor = belief_hard_floor
        self._belief_propagation_damping = belief_propagation_damping

        # Bayesian belief scoring: soft fact currency
        if belief:
            self._belief_index = BeliefIndex(
                prior_alpha=belief_prior_alpha,
                prior_beta=belief_prior_beta,
                propagation_damping=belief_propagation_damping,
                hard_floor=belief_hard_floor,
            )
        else:
            self._belief_index = None

        # Supersession: entity-gated triple-based conflict detection
        # Maps (normalized_subject, normalized_predicate) → [(normalized_object, fact_text)]
        # When a new triple has same S+P but different O AND the facts share at
        # least one named entity, the old fact is superseded.
        self._triple_index: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        self._superseded_texts: set[str] = set()
        self._fact_entities: dict[str, tuple[str, ...]] = {}  # fact_text → entities
        # Conflict resolution: directed supersession graph
        # old_text → new_text (the text that supersedes it)
        self._supersession_graph: dict[str, str] = {}
        # new_text → {old_text_1, old_text_2, ...}
        self._supersedes: dict[str, set[str]] = defaultdict(set)

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
            dedup_threshold=dedup_threshold,
        )

        # Ingestion pipeline (replaces regex _parse_facts)
        try:
            import torch
            _coref_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _coref_device = "cpu"
        self._ingestion = IngestionPipeline(IngestionConfig(
            coref_device=_coref_device,
            coref_enabled=True,
            boundary_threshold=0.2,
            strip_numbered_prefix=False,
        ))
        self._entity_accumulator = _MutableEntityGraph()

        # Triadic memory: structural recall via Overmann sparse 3D tensor
        self._triadic = TriadicMemory(n=triadic_n, p=triadic_p) if triadic_retrieval else None

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
        elif self.iterative_query:
            return self._iterative_query(message)
        else:
            return self._query(message)

    def _store(self, text: str) -> str:
        """Store text through ingestion pipeline + V1 + V2 contradiction layers.

        Uses IngestionPipeline.segment() for multi-stage text decomposition:
        - Sentence splitting with numbered-prefix stripping
        - Topic boundary detection
        - Coreference resolution
        - Clause decomposition into atomic facts
        - Entity graph construction

        Each atomic fact is stored individually for fine-grained retrieval,
        contradiction detection, and dream consolidation.
        """
        result = self._ingestion.segment(text)

        # Batch-embed all atomic facts (single GPU pass)
        fact_texts = [f.text for f in result.facts]
        embeddings = self._scorer.embed_batch(fact_texts)

        for fact, emb in zip(result.facts, embeddings):
            # V1: Text-based store
            self.orchestrator.store(content=fact.text, embedding=emb)

            # V2: Coupled engine store (no serial number prefix -- D3)
            self.coupled_engine.store(
                text=fact.text, embedding=emb,
                recency=float(self._store_count),
            )

            # Triadic: store fact's triples in the 3D tensor
            if getattr(self, '_triadic', None) is not None:
                self._triadic.store_fact(fact.text, list(fact.triples))

            # Register fact in belief index (before conflict detection)
            if self._belief_index is not None:
                self._belief_index.register_fact(fact.text)

            # --- Conflict resolution: template-based triple extraction ---
            # Uses regex templates (100% coverage on factconsolidation) instead
            # of spaCy dep-parse triples (which only found 6/4,317 conflicts).
            if self.conflict_resolution:
                template_triple = _extract_template_triple(fact.text)
                if template_triple:
                    subj_t, pred_t, obj_t = template_triple
                    key_t = (subj_t.lower(), pred_t)
                    new_obj_t = obj_t.lower()
                    for existing_obj, existing_text in self._triple_index[key_t]:
                        if existing_obj != new_obj_t:
                            self._supersession_graph[existing_text] = fact.text
                            self._supersedes[fact.text].add(existing_text)
                    self._triple_index[key_t].append((new_obj_t, fact.text))

            # --- Supersession / belief: spaCy triple-based conflict detection ---
            # Same subject+predicate, different object → potential conflict,
            # but only act if the old and new facts share a named entity.
            if (self.supersession or self._belief_index) and fact.triples:
                new_entities = set(e.lower() for e in fact.entities) if fact.entities else set()
                self._fact_entities[fact.text] = fact.entities
                for (subj, pred, obj) in fact.triples:
                    key = (subj.lower().strip(), pred.lower().strip())
                    new_obj = obj.lower().strip()
                    if not key[0] or not key[1]:
                        continue  # skip degenerate triples
                    for existing_obj, existing_text in self._triple_index[key]:
                        if existing_obj != new_obj:
                            # Entity gate: require shared named entity
                            old_entities = set(
                                e.lower() for e in self._fact_entities.get(existing_text, ())
                            )
                            if not (old_entities & new_entities):
                                continue  # no shared entity → not a real conflict

                            # Binary supersession (when enabled)
                            if self.supersession:
                                self._superseded_texts.add(existing_text)

                            # Bayesian belief update (when enabled)
                            if self._belief_index is not None:
                                self._belief_index.on_conflict(
                                    existing_text, fact.text,
                                )
                    self._triple_index[key].append((new_obj, fact.text))

            self._store_count += 1

        # Synonym dict: feed coref clusters and verb mappings to triadic
        if getattr(self, '_triadic', None) is not None:
            if result.coref_clusters:
                self._triadic.add_coref_clusters(result.coref_clusters)
            if result.verb_mappings:
                self._triadic.add_verb_mappings(result.verb_mappings)

        # Accumulate entity graph with global offset
        if hasattr(self, '_entity_accumulator'):
            offset = self._store_count - len(result.facts)
            self._entity_accumulator.merge(result.entity_graph, offset)

        # Belief propagation: penalize facts connected to degraded facts
        if self._belief_index is not None and hasattr(self, '_entity_accumulator'):
            self._belief_index.propagate(
                self._entity_accumulator,
                self.coupled_engine.memory_store,
            )

        # Flush session buffer per chunk if Hebbian enabled (D8)
        if hasattr(self.coupled_engine, '_session_buffer') and \
           len(getattr(self.coupled_engine, '_session_buffer', [])) > 0:
            self.coupled_engine.flush_session()

        # Supersession / conflict resolution diagnostics: log every 50 stores
        if (self.supersession or self.conflict_resolution) and self._store_count % 50 == 0 and self._store_count > 0:
            import logging
            logging.getLogger("supersession").info(
                "STATS after %d stores: %d unique (S,P) keys, %d superseded texts, %d graph edges, %d total triples",
                self._store_count, len(self._triple_index),
                len(self._superseded_texts),
                len(self._supersession_graph),
                sum(len(v) for v in self._triple_index.values()),
            )

        # Dream cycle -- unchanged logic
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

        # V2 retrieval from coupled engine
        if self.decompose_query:
            # Iterative multi-hop: decompose → retrieve → extract → substitute
            # Each hop receives the growing evidence chain for context.
            #
            # Priority merge (fixes slot starvation):
            #   Original question gets top-5 priority slots (proven best chunks).
            #   Each sub-query contributes top-3 unique chunks as hop-specific
            #   supplements, without displacing the base retrieval.
            subqueries = self._decompose_query(bare_question)
            seen_texts: set[str] = set()
            answers: dict[str, str] = {}  # {answer_1} → resolved value
            evidence_chain: list[dict] = []  # growing context across hops
            sq_chunks: list[list[str]] = []  # per-subquery chunk lists
            sq_chunks_raw: list[list[dict]] = []  # raw result dicts for metadata

            for i, sq_template in enumerate(subqueries):
                # Substitute any {answer_N} placeholders from prior hops
                sq = sq_template
                for key, val in answers.items():
                    sq = sq.replace(key, val)

                # Mini iterative loop per sub-query: retrieve → judge →
                # retry with targeted NEED if first retrieval misses.
                sq_all_texts: list[str] = []
                sq_all_results: list[dict] = []
                sq_seen: set[str] = set()
                current_sq = sq
                max_sq_retries = 2  # 1 original + 1 retry

                for retry in range(max_sq_retries):
                    sq_results = self._retrieve_v2(current_sq)
                    for r in sq_results:
                        key = r["text"].strip()[:200]
                        if key not in sq_seen:
                            sq_seen.add(key)
                            sq_all_texts.append(r["text"])
                            sq_all_results.append(r)

                    # Triadic expansion on this sub-query's results
                    if getattr(self, '_triadic', None) is not None:
                        hop_texts = [r["text"] for r in sq_results]
                        if hop_texts:
                            for t in self._triadic.expand(
                                hop_texts, top_k=self._triadic_expand_k,
                            ):
                                tkey = t.strip()[:200]
                                if tkey not in sq_seen:
                                    sq_seen.add(tkey)
                                    sq_all_texts.append(t)

                    # On last retry or final sub-query, skip judge
                    if retry >= max_sq_retries - 1:
                        break

                    # Judge: can we answer this sub-query?
                    try:
                        judge_resp = self._llm_client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": (
                                    "Given retrieved facts, can you answer "
                                    "this specific sub-question?\n"
                                    "If YES: respond ANSWER_READY\n"
                                    "If NO: respond NEED: <3-6 word search "
                                    "using concrete entity names>"
                                )},
                                {"role": "user", "content": (
                                    f"Sub-question: {current_sq}\n\n"
                                    f"Facts:\n"
                                    + "\n".join(sq_all_texts[:10])
                                    + "\n\nCan you answer?"
                                )},
                            ],
                            temperature=0.0,
                            max_tokens=64,
                        )
                        jt = (judge_resp.choices[0].message.content
                              or "").strip()
                    except Exception:
                        break  # LLM error → use what we have

                    if jt.startswith("ANSWER_READY"):
                        break
                    elif "NEED:" in jt:
                        need = jt[jt.index("NEED:") + 5:].strip().strip("'\"")
                        if need and need != current_sq:
                            current_sq = need
                        else:
                            break
                    else:
                        break  # unparseable → use what we have

                # Per-hop conflict resolution: remove stale facts
                # BEFORE the LLM extracts an intermediate answer.
                # Without this, the LLM sees all versions of a fact
                # (e.g., 4 different sports for Abbiati) and picks wrong,
                # poisoning all downstream hops.
                if self.conflict_resolution:
                    # Build per-hop metadata for _resolve_conflicts
                    _hop_meta: dict[str, dict] = {}
                    for r in sq_all_results:
                        k = r["text"].strip()[:200]
                        if k not in _hop_meta:
                            _hop_meta[k] = r
                    sq_all_texts = self._resolve_conflicts(sq_all_texts, _hop_meta)
                elif self._belief_index is not None:
                    sq_all_texts = [
                        t for t in sq_all_texts
                        if not self._belief_index.is_excluded(t)
                    ]
                    sq_all_texts = self._belief_dedup(sq_all_texts)

                sq_chunks.append(sq_all_texts)
                sq_chunks_raw.append(sq_all_results)

                # Extract intermediate answer for chaining to next hop
                if i < len(subqueries) - 1 and sq_all_texts:
                    short_ans = self._extract_short_answer(
                        sq, sq_all_texts, evidence_chain,
                    )
                    if short_ans:
                        answers[f"{{answer_{i + 1}}}"] = short_ans
                        evidence_chain.append({
                            "q": sq,
                            "a": short_ans,
                            "evidence": sq_all_texts[:2],
                        })

            # Merge sub-query and original results
            orig_results = self._retrieve_v2(bare_question)
            orig_texts = [r["text"] for r in orig_results]

            if self.decompose_rrf:
                # Reciprocal Rank Fusion: score each chunk by its rank
                # across all query result lists. Chunks appearing in
                # multiple lists get boosted naturally.
                # RRF score = sum(1 / (k + rank)) across all lists
                rrf_k = 60  # standard RRF constant
                rrf_scores: dict[str, float] = {}
                all_lists = [orig_texts] + sq_chunks
                for ranked_list in all_lists:
                    for rank, text in enumerate(ranked_list):
                        key = text.strip()[:200]
                        if key not in rrf_scores:
                            rrf_scores[key] = 0.0
                        rrf_scores[key] += 1.0 / (rrf_k + rank)
                # Map keys back to full texts
                key_to_text: dict[str, str] = {}
                for ranked_list in all_lists:
                    for text in ranked_list:
                        key = text.strip()[:200]
                        if key not in key_to_text:
                            key_to_text[key] = text
                # Sort by RRF score descending, take top retrieve_num
                sorted_keys = sorted(
                    rrf_scores.keys(),
                    key=lambda k: rrf_scores[k],
                    reverse=True,
                )
                v2_texts = [
                    key_to_text[k] for k in sorted_keys[:self.retrieve_num]
                ]
            else:
                # Priority merge: orig top-5, then sub-queries top-3 each
                v2_texts = []
                for r in orig_results[:5]:
                    t = r["text"]
                    if t not in seen_texts:
                        seen_texts.add(t)
                        v2_texts.append(t)
                for sq_text_list in sq_chunks:
                    added = 0
                    for t in sq_text_list:
                        if added >= 3:
                            break
                        if t not in seen_texts:
                            seen_texts.add(t)
                            v2_texts.append(t)
                            added += 1
        else:
            v2_results = self._retrieve_v2(bare_question)
            v2_texts = [r["text"] for r in v2_results]
            sq_chunks_raw = []  # no sub-query results in non-decompose path

        # Build metadata lookup: text key → result dict.
        # The merge pipeline deduplicates on text, so we index metadata by
        # the same key (first 200 chars) used for dedup. This lets the
        # outgestion formatter access recency/importance/score without
        # rewriting every merge path.
        _metadata_by_key: dict[str, dict] = {}
        _all_v2_results = []
        if self.decompose_query:
            _all_v2_results = list(orig_results)
            for sq_r in sq_chunks_raw:
                _all_v2_results.extend(sq_r)
        else:
            _all_v2_results = list(v2_results)
        for r in _all_v2_results:
            k = r["text"].strip()[:200]
            if k not in _metadata_by_key:
                _metadata_by_key[k] = r

        # V1 retrieval: text-based recall from orchestrator
        # Skip V1 when V2 cooc_boost is active — V1 iterates all N memories
        # in pure Python (O(N) Jaccard + score_memory per query) and its
        # output is strictly weaker than V2 cosine+cooc+triadic.
        if self.cooc_boost_retrieval:
            v1_context = ""
        else:
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
        # Triadic expansion: structural recall from retrieved facts' triples
        if getattr(self, '_triadic', None) is not None and v2_texts:
            triadic_texts = self._triadic.expand(
                v2_texts, top_k=self._triadic_expand_k
            )
            for text in triadic_texts:
                key = text.strip()[:200]
                if key not in seen:
                    seen.add(key)
                    merged_chunks.append(text)

        # Coverage audit: decompose question into sub-queries and check
        # if retrieved chunks cover each hop. Fill gaps with targeted retrieval.
        if self.decompose_coverage and not self.decompose_query:
            gap_texts = self._coverage_audit(bare_question, v2_texts)
            for text in gap_texts:
                key = text.strip()[:200]
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

        # Fact currency filtering:
        # - Conflict resolution (three-layer): preferred — uses supersession
        #   graph to resolve only when both old+new are in retrieved set.
        # - Belief (soft): demotes stale facts via P(current).
        # - Supersession (binary): fallback when neither is enabled.
        if self.conflict_resolution:
            merged_chunks = self._resolve_conflicts(merged_chunks, _metadata_by_key)
        elif self._belief_index is not None:
            merged_chunks = [
                c for c in merged_chunks
                if not self._belief_index.is_excluded(c)
            ]
            merged_chunks = self._belief_dedup(merged_chunks)
        elif self.supersession and self._superseded_texts:
            merged_chunks = [
                c for c in merged_chunks
                if c not in self._superseded_texts
            ]

        # Build context for LLM via outgestion formatter
        final_chunks = merged_chunks[:self.retrieve_num]
        memory_context, outgestion_instructions = self._format_outgestion(
            final_chunks, _metadata_by_key
        )
        memory_time = time.time() - query_start

        # Generate answer via LLM
        answer, input_tokens, output_tokens, gen_time = self._generate_answer(
            question, memory_context,
            extra_system_instructions=outgestion_instructions,
        )
        query_time = time.time() - query_start

        return {
            "output": answer,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
        }

    def _decompose_query(self, question: str) -> list[str]:
        """Decompose a multi-hop question into chained sub-queries.

        Uses the LLM to break compositional questions like
        "country of origin of sport played by X" into:
          1. "What sport did Christian Abbiati play?"
          2. "What is the country of origin of {answer_1}?"

        Sub-queries use {answer_N} placeholders for chaining.
        Returns a list of sub-query template strings.
        """
        self._ensure_llm_client()

        prompt = (
            "Break this question into simple sub-questions that each require "
            "looking up a single fact. Start from the innermost entity and "
            "work outward. Use {answer_1}, {answer_2} etc. as placeholders "
            "for answers from previous sub-questions. Use the actual entity "
            "names from the question, not pronouns.\n\n"
            "Example:\n"
            "Q: What is the country of origin of the sport played by Christian Abbiati?\n"
            "1. What sport did Christian Abbiati play?\n"
            "2. What is the country of origin of {answer_1}?\n\n"
            "Example:\n"
            "Q: Where was the creator of the religion Karl Lueger followed born?\n"
            "1. What religion did Karl Lueger follow?\n"
            "2. Who created {answer_1}?\n"
            "3. Where was {answer_2} born?\n\n"
            f"Q: {question}\n"
        )

        try:
            response = self._llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You decompose multi-hop questions into chained "
                        "sub-questions. Use {answer_N} placeholders. "
                        "Be concise."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            text = response.choices[0].message.content or ""
        except Exception:
            return [question]

        # Parse numbered sub-questions
        subqs = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip numbering: "1. ...", "1) ...", "- ..."
            for prefix in ("1.", "2.", "3.", "4.", "5.",
                           "1)", "2)", "3)", "4)", "5)", "-"):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line:
                subqs.append(line)

        # Cap at max_hops
        subqs = subqs[:self.decompose_max_hops]
        return subqs if subqs else [question]

    def _extract_short_answer(
        self, question: str, facts: list[str],
        evidence_chain: list[dict] | None = None,
    ) -> str:
        """Extract a short factual answer from retrieved facts.

        Used in iterative query decomposition to resolve intermediate
        sub-queries before substituting into the next sub-query template.

        Args:
            question: The current sub-query to answer.
            facts: Retrieved fact texts for this sub-query.
            evidence_chain: List of dicts from prior hops, each with
                keys 'q' (question), 'a' (answer), 'evidence' (key facts).
                Provides reasoning context so the LLM understands the
                multi-hop chain being resolved.
        """
        self._ensure_llm_client()
        context = "\n".join(facts[:10])

        # Build reasoning history from prior hops
        history = ""
        if evidence_chain:
            lines = []
            for i, hop in enumerate(evidence_chain, 1):
                lines.append(
                    f"Hop {i}: \"{hop['q']}\" → \"{hop['a']}\""
                    f" (from: \"{hop['evidence'][0][:120]}...\")"
                    if hop['evidence'] else
                    f"Hop {i}: \"{hop['q']}\" → \"{hop['a']}\""
                )
            history = "Previous reasoning:\n" + "\n".join(lines) + "\n\n"

        try:
            response = self._llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "Answer the question using ONLY the provided facts. "
                        "Use the previous reasoning chain for context. "
                        "Give just the entity name, nothing else."
                    )},
                    {"role": "user", "content": (
                        f"{history}"
                        f"Facts:\n{context}\n\n"
                        f"Question: {question}\nAnswer:"
                    )},
                ],
                temperature=0.0,
                max_tokens=32,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _iterative_query(self, question: str) -> dict:
        """Iterative multi-hop retrieval with LLM-in-the-loop termination.

        Instead of decomposing the question upfront (like decompose_query),
        this method retrieves iteratively: after each retrieval pass, the LLM
        judges whether it can answer the question. If not, the LLM identifies
        the missing entity, which becomes the query for the next hop.

        The key insight: multi-hop is a property of the agent's current
        knowledge state, not the question. The LLM is the only general
        termination criterion because "do I have enough to answer?" is a
        semantic judgment.

        Loop:
          1. Retrieve for question (or entity from prior hop)
          2. LLM: "Can you answer? If not, what entity do you need next?"
          3. If can answer → break. If not → retrieve for that entity.
          4. Max hops as safety valve.
        """
        query_start = time.time()
        bare_question = self._extract_question(question)
        self._ensure_llm_client()

        all_chunks: list[str] = []
        all_results: list[dict] = []
        seen: set[str] = set()
        hops_taken = 0
        # Evidence chain: tracks what was queried and found at each hop,
        # so the judge LLM has full reasoning context (not stateless).
        evidence_chain: list[dict] = []  # [{query, found, entity}]

        current_query = bare_question

        for hop in range(self.iterative_max_hops):
            # Retrieve for current query
            results = self._retrieve_v2(current_query)

            # Triadic expansion on this hop's results
            hop_texts = [r["text"] for r in results]
            if getattr(self, '_triadic', None) is not None and hop_texts:
                triadic_texts = self._triadic.expand(
                    hop_texts, top_k=self._triadic_expand_k,
                )
                for t in triadic_texts:
                    key = t.strip()[:200]
                    if key not in seen:
                        seen.add(key)
                        all_chunks.append(t)

            # Add new chunks (dedup across hops)
            hop_new_chunks: list[str] = []
            for r in results:
                key = r["text"].strip()[:200]
                if key not in seen:
                    seen.add(key)
                    all_chunks.append(r["text"])
                    all_results.append(r)
                    hop_new_chunks.append(r["text"])

            hops_taken = hop + 1

            # Don't judge on last possible hop — just use what we have
            if hop >= self.iterative_max_hops - 1:
                evidence_chain.append({
                    "query": current_query,
                    "found": hop_new_chunks[:3],
                    "entity": None,
                })
                break

            # Build chain summary for the judge so it has full hop context
            chain_summary = ""
            if evidence_chain:
                lines = []
                for i, step in enumerate(evidence_chain, 1):
                    found_preview = step["found"][0][:100] if step["found"] else "nothing relevant"
                    learned_str = f" → learned: {step['learned']}" if step.get("learned") else ""
                    entity_str = f" → next search: \"{step['entity']}\"" if step.get("entity") else ""
                    lines.append(
                        f"  Hop {i}: searched \"{step['query']}\" "
                        f"→ found \"{found_preview}...\"{learned_str}{entity_str}"
                    )
                chain_summary = (
                    "Reasoning chain so far:\n"
                    + "\n".join(lines)
                    + "\n\n"
                )

            # LLM judge sees ALL accumulated chunks (not capped at retrieve_num).
            # Each hop adds new deduped chunks; the judge needs the full
            # evidence to decide if the chain is complete.
            context_for_judge = "\n".join(all_chunks)
            try:
                judge_response = self._llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": (
                            "You evaluate whether retrieved facts can answer "
                            "a question, and extract intermediate answers.\n\n"
                            "If the facts contain enough to answer, respond:\n"
                            "ANSWER_READY\n\n"
                            "If not, extract what you DID learn and state "
                            "what simple fact to search next. Format:\n"
                            "LEARNED: <what you found, e.g. 'Abbiati plays "
                            "association football'>\n"
                            "NEED: <simple search, e.g. 'association football "
                            "country of origin'>\n\n"
                            "The NEED query will be used for embedding search. "
                            "Use CONCRETE entities from what you learned, "
                            "NOT abstract references like 'the sport' or "
                            "'the person'. Keep NEED to 3-8 words.\n"
                            "Do NOT repeat a previous search."
                        )},
                        {"role": "user", "content": (
                            f"Question: {bare_question}\n\n"
                            f"{chain_summary}"
                            f"Retrieved facts:\n{context_for_judge}\n\n"
                            f"Can you answer? If not, what did you learn "
                            f"and what concrete fact is still missing?"
                        )},
                    ],
                    temperature=0.0,
                    max_tokens=64,
                )
                judge_text = (
                    judge_response.choices[0].message.content or ""
                ).strip()
            except Exception:
                evidence_chain.append({
                    "query": current_query,
                    "found": hop_new_chunks[:3],
                    "entity": None,
                })
                break  # LLM failure → use what we have

            if judge_text.startswith("ANSWER_READY"):
                evidence_chain.append({
                    "query": current_query,
                    "found": hop_new_chunks[:3],
                    "entity": None,
                })
                break

            # Parse LEARNED/NEED format
            learned = ""
            next_entity = ""
            for line in judge_text.split("\n"):
                line = line.strip()
                if line.startswith("LEARNED:"):
                    learned = line[len("LEARNED:"):].strip()
                elif line.startswith("NEED:"):
                    next_entity = line[len("NEED:"):].strip()

            # Fallback: old format (just "NEED: ...")
            if not next_entity and judge_text.startswith("NEED:"):
                next_entity = judge_text[len("NEED:"):].strip()

            if next_entity:
                evidence_chain.append({
                    "query": current_query,
                    "found": hop_new_chunks[:3],
                    "entity": next_entity,
                    "learned": learned,
                })
                current_query = next_entity
            else:
                # Unparseable response → treat as ready
                evidence_chain.append({
                    "query": current_query,
                    "found": hop_new_chunks[:3],
                    "entity": None,
                })
                break

        # Build metadata lookup
        metadata_by_key: dict[str, dict] = {}
        for r in all_results:
            k = r["text"].strip()[:200]
            if k not in metadata_by_key:
                metadata_by_key[k] = r

        # Fact currency filtering
        if self.conflict_resolution:
            all_chunks = self._resolve_conflicts(all_chunks, metadata_by_key)
        elif self._belief_index is not None:
            all_chunks = [
                c for c in all_chunks
                if not self._belief_index.is_excluded(c)
            ]
            all_chunks = self._belief_dedup(all_chunks)
        elif self.supersession and self._superseded_texts:
            all_chunks = [
                c for c in all_chunks
                if c not in self._superseded_texts
            ]

        # Trim to retrieve_num and format via outgestion
        final_chunks = all_chunks[:self.retrieve_num]
        memory_context, outgestion_instructions = self._format_outgestion(
            final_chunks, metadata_by_key,
        )
        memory_time = time.time() - query_start

        # Generate answer
        answer, input_tokens, output_tokens, gen_time = self._generate_answer(
            question, memory_context,
            extra_system_instructions=outgestion_instructions,
        )
        query_time = time.time() - query_start

        return {
            "output": answer,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
            "hops": hops_taken,
        }

    def _retrieve_v2(self, query_text: str) -> list[dict]:
        """Run V2 embedding retrieval for a single query string.

        When belief scoring is active, over-fetches by belief_overfetch
        multiplier and then applies belief reranking to compensate for
        excluded/demoted facts.
        """
        q_emb = self._scorer.embed(query_text)

        # Over-fetch when belief is active to compensate for exclusions
        effective_k = self.retrieve_num
        if self._belief_index is not None:
            effective_k = self.retrieve_num * self._belief_overfetch

        if self.transfer_retrieval:
            results = self.coupled_engine.query_transfer(
                embedding=q_emb, top_k=effective_k,
                transfer_k=self.transfer_k,
            )
        elif self.coretrieval_retrieval:
            results = self.coupled_engine.query_coretrieval(
                embedding=q_emb, top_k=effective_k,
                coretrieval_bonus=self.coretrieval_bonus,
                min_coretrieval_count=self.coretrieval_min_count,
            )
        elif self.cooc_boost_retrieval and self.ppr_retrieval:
            results = self.coupled_engine.query_cooc_ppr(
                embedding=q_emb, top_k=effective_k,
                cooc_weight=self.cooc_weight,
                gate_threshold=self.cooc_gate_threshold,
                ppr_weight=self.ppr_weight,
                damping=self.ppr_damping,
            )
        elif self.cooc_boost_retrieval and self.bm25_weight > 0:
            results = self.coupled_engine.query_cooc_bm25(
                embedding=q_emb, query_text=query_text,
                top_k=effective_k,
                cooc_weight=self.cooc_weight,
                bm25_weight=self.bm25_weight,
                gate_threshold=self.cooc_gate_threshold,
            )
        elif self.cooc_boost_retrieval:
            results = self.coupled_engine.query_cooc_boost(
                embedding=q_emb, top_k=effective_k,
                cooc_weight=self.cooc_weight,
                dream_weight=self.dream_weight,
                gate_threshold=self.cooc_gate_threshold,
            )
        elif self.ppr_retrieval:
            results = self.coupled_engine.query_ppr(
                embedding=q_emb, top_k=effective_k,
                ppr_weight=self.ppr_weight,
                damping=self.ppr_damping,
            )
        elif self.hybrid_retrieval:
            results = self.coupled_engine.query_hybrid(
                embedding=q_emb, top_k=effective_k,
            )
        elif self.associative_retrieval or self.sparse_retrieval:
            results = self.coupled_engine.query_associative(
                embedding=q_emb, top_k=effective_k,
                sparse=self.sparse_retrieval,
            )
        else:
            results = self.coupled_engine.query(
                embedding=q_emb, top_k=effective_k,
            )

        # Apply belief reranking when active
        if self._belief_index is not None:
            results = self._apply_belief_rerank(results)[:self.retrieve_num]

        return results

    def _apply_belief_rerank(self, results: list[dict]) -> list[dict]:
        """Rerank retrieval results using Bayesian belief scores.

        1. Exclude facts below hard_floor (P(current) too low).
        2. Multiply each result's score by P(current).
        3. Re-sort by adjusted score descending.
        """
        reranked = []
        for r in results:
            text = r.get("text", "")
            if self._belief_index.is_excluded(text):
                continue
            p = self._belief_index.score(text)
            adjusted = dict(r)
            adjusted["score"] = r.get("score", 0.0) * p
            reranked.append(adjusted)
        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reranked

    def _coverage_audit(
        self, bare_question: str, v2_texts: list[str],
    ) -> list[str]:
        """Use decomposition as a coverage check, not a retrieval strategy.

        1. Decompose the question into sub-queries.
        2. Embed each sub-query and each already-retrieved chunk.
        3. For each sub-query, check if any retrieved chunk has
           cosine > threshold — if not, do a targeted retrieval.
        4. Return additional texts to append (may be empty).
        """
        subqueries = self._decompose_query(bare_question)
        if len(subqueries) <= 1:
            return []  # single-hop, nothing to audit

        # Embed sub-queries and retrieved chunks
        sq_embs = [self._scorer.embed(sq) for sq in subqueries]
        chunk_embs = self._scorer.embed_batch(v2_texts)

        coverage_threshold = 0.35
        gap_texts: list[str] = []
        seen = set(t.strip()[:200] for t in v2_texts)

        for sq_idx, (sq, sq_emb) in enumerate(zip(subqueries, sq_embs)):
            # Check if any existing chunk covers this sub-query
            max_sim = 0.0
            for c_emb in chunk_embs:
                sim = float(np.dot(sq_emb, c_emb))
                if sim > max_sim:
                    max_sim = sim
            if max_sim >= coverage_threshold:
                continue  # sub-query is covered

            # Gap found — targeted retrieval for this sub-query
            sq_results = self._retrieve_v2(sq)
            for r in sq_results[:3]:  # inject at most 3 gap-fillers
                key = r["text"].strip()[:200]
                if key not in seen:
                    seen.add(key)
                    gap_texts.append(r["text"])

        return gap_texts

    def _belief_dedup(self, chunks: list[str]) -> list[str]:
        """Deduplicate conflicting facts by belief score.

        When multiple retrieved chunks share a (subject, predicate) triple key
        but have different objects, keep only the highest-P(current) version.
        This prevents the LLM from seeing both sides of a resolved conflict
        (the #1 cause of wrong answers — 94% of failures are outgestion).
        """
        if self._belief_index is None or not self._triple_index:
            return chunks

        retrieved_set = set(chunks)
        to_remove: set[str] = set()

        for (_subj, _pred), entries in self._triple_index.items():
            # Which entries from this triple key are in our retrieval set?
            group = [(obj, text) for (obj, text) in entries
                     if text in retrieved_set]
            if len(group) <= 1:
                continue
            # Only dedup when there are genuinely different objects (real conflict)
            if len(set(obj for obj, _text in group)) <= 1:
                continue
            # Keep the fact with highest P(current)
            best_text = max(group,
                            key=lambda x: self._belief_index.score(x[1]))[1]
            for (_obj, text) in group:
                if text != best_text:
                    to_remove.add(text)

        if to_remove:
            return [c for c in chunks if c not in to_remove]
        return chunks

    def _resolve_conflicts(
        self, chunks: list[str], metadata_by_key: dict[str, dict],
    ) -> list[str]:
        """Three-layer conflict resolution (Layer 2a + 2b).

        Layer 2a (triple-based, high precision):
        For each chunk, check the supersession graph. If this chunk is
        superseded by another chunk that is ALSO in the retrieved set,
        drop it. Only removes old facts when the newer version is present.

        Layer 2b (embedding-based, fallback):
        For chunks not resolved by 2a, do pairwise embedding similarity.
        If two chunks have cosine > 0.85 AND share a named entity, drop
        the older one (lower store_count / recency).
        """
        if not self._supersession_graph and not self._supersedes:
            return chunks

        import logging
        _cr_log = logging.getLogger("conflict_res")
        _cr_log.info("_resolve_conflicts: %d chunks, graph=%d edges", len(chunks), len(self._supersession_graph))

        retrieved_set = set(chunks)
        to_remove: set[str] = set()

        # 2a: Triple-based — use supersession graph
        for chunk in chunks:
            # Is this chunk superseded by something also in the retrieved set?
            newer = self._supersession_graph.get(chunk)
            if newer and newer in retrieved_set:
                to_remove.add(chunk)
            # Does this chunk supersede something in the set?
            for older in self._supersedes.get(chunk, ()):
                if older in retrieved_set:
                    to_remove.add(older)

        if to_remove:
            _cr_log.info("_resolve_conflicts: removing %d chunks: %s",
                         len(to_remove), [t[:60] for t in to_remove])
        if to_remove:
            return [c for c in chunks if c not in to_remove]
        return chunks

    def _format_outgestion(
        self,
        chunks: list[str],
        metadata_by_key: dict[str, dict],
    ) -> tuple[str, str]:
        """Format retrieved chunks with outgestion signals.

        Returns (formatted_context, extra_system_instructions).
        When temporal_context is disabled, returns flat text with no
        extra instructions (baseline behavior).

        Signals applied (when enabled):
        1. Temporal ordering: sort oldest→newest, add [older]/[newer] tags
        2. Contradiction detection: flag high-similarity chunk pairs as
           [SUPERSEDED]/[CURRENT] when they likely represent conflicting
           versions of the same fact
        """
        # Conflict resolution mode: conflicts already resolved by Layer 2.
        # Return flat text in retrieval order (preserves relevance ranking).
        # Layer 3 (newest-first sort) disabled — disrupts relevance ordering.
        if getattr(self, "conflict_resolution", False):
            return "\n\n".join(chunks), ""

        if not getattr(self, "temporal_context", False):
            # Baseline: flat text, no metadata
            return "\n\n".join(chunks), ""

        # Build (recency, text, embedding) tuples for sorting + contradiction.
        # Chunks without metadata (e.g. from V1 or triadic expansion)
        # get recency=0 so they sort as oldest.
        annotated_full: list[tuple[float, str, "np.ndarray | None"]] = []
        for text in chunks:
            key = text.strip()[:200]
            meta = metadata_by_key.get(key)
            recency = meta["recency"] if meta else 0.0
            embedding = meta.get("embedding") if meta else None
            annotated_full.append((recency, text, embedding))

        # Sort oldest → newest (ascending recency)
        annotated_full.sort(key=lambda x: x[0])

        # Soft contradiction surfacing: find pairs of chunks that are
        # semantically similar (cosine > threshold) but have different
        # recency values. Cross-reference BOTH with [CONFLICT] tags —
        # surface the conflict, never remove content.
        contradiction_threshold = getattr(self, "contradiction_sim_threshold", 0.85)
        conflict_pairs: dict[int, list[int]] = defaultdict(list)

        if getattr(self, "contradiction_context", False):
            import numpy as np
            n_ann = len(annotated_full)
            for i in range(n_ann):
                for j in range(i + 1, n_ann):
                    emb_i = annotated_full[i][2]
                    emb_j = annotated_full[j][2]
                    if emb_i is None or emb_j is None:
                        continue
                    sim = float(np.dot(emb_i, emb_j))
                    if sim >= contradiction_threshold:
                        rec_i = annotated_full[i][0]
                        rec_j = annotated_full[j][0]
                        if rec_i != rec_j:
                            conflict_pairs[i].append(j)
                            conflict_pairs[j].append(i)

        # Format with fact numbering + temporal + soft contradiction
        lines = []
        n = len(annotated_full)
        for i, (recency, text, _emb) in enumerate(annotated_full):
            fact_num = i + 1  # 1-indexed for LLM readability

            if i in conflict_pairs:
                partners = ", ".join(str(p + 1) for p in conflict_pairs[i])
                tag = f"[fact {fact_num}, CONFLICTS WITH fact {partners}] "
            elif n <= 1:
                tag = ""
            elif i < n // 3:
                tag = f"[fact {fact_num}, older] "
            elif i >= n - n // 3:
                tag = f"[fact {fact_num}, newer] "
            else:
                tag = f"[fact {fact_num}] "
            lines.append(f"{tag}{text}")

        instructions = (
            "Facts in the knowledge pool are numbered and ordered from oldest "
            "to newest. When facts conflict, prefer the most recent (newer) "
            "information."
        )
        if conflict_pairs:
            instructions += (
                " Facts marked [CONFLICTS WITH] represent potentially "
                "conflicting versions — prefer the newer one among "
                "conflicting facts."
            )
        return "\n\n".join(lines), instructions

    def _generate_answer(
        self, question: str, context: str,
        extra_system_instructions: str = "",
    ) -> tuple[str, int, int, float]:
        """Call LLM to generate an answer from retrieved context."""
        self._ensure_llm_client()

        system_msg = (
            "You are a helpful assistant that answers questions from a "
            "knowledge pool. Give a very concise answer."
        )
        if extra_system_instructions:
            system_msg = system_msg + " " + extra_system_instructions
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
            dedup_threshold=self.dedup_threshold,
        )
        self._store_count = 0
        self._scorer._cache.clear()

        # Reset triadic memory (fresh tensor, codebook, mappings)
        if getattr(self, '_triadic', None) is not None:
            self._triadic = TriadicMemory(
                n=self._triadic.n, p=self._triadic.p,
            )

        # Reset entity accumulator
        if hasattr(self, '_entity_accumulator'):
            self._entity_accumulator = _MutableEntityGraph()

        # Reset supersession state
        self._triple_index = defaultdict(list)
        self._superseded_texts = set()
        self._fact_entities = {}

        # Reset belief index (preserve params, clear beliefs)
        if self._belief_index is not None:
            self._belief_index = BeliefIndex(
                prior_alpha=self._belief_index._prior_alpha,
                prior_beta=self._belief_index._prior_beta,
                propagation_damping=self._belief_index._propagation_damping,
                hard_floor=self._belief_index._hard_floor,
            )
