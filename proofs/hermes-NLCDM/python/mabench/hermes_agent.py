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
import logging
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
    IngestionPipeline,
    IngestionConfig,
    IngestionResult,
    _MutableEntityGraph,
)
from mabench.belief import BeliefIndex
from mabench.bm25_index import BM25Index
from mabench.chain_cache import ChainCache, CachedChain
from mabench.triadic_memory import TriadicMemory


# ---------------------------------------------------------------------------
# Query-time conflict resolution helpers
# ---------------------------------------------------------------------------
_cr_log = logging.getLogger("conflict_res")

def _get_subtree_text(token) -> str:
    """Get the full text of a token's subtree (e.g., 'association football')."""
    return " ".join(t.text for t in token.subtree)


# ---------------------------------------------------------------------------
# Tier 1: Rule-based predicate cardinality classification
# ---------------------------------------------------------------------------
# Verbs/predicates where same (S,P) with different O is NOT a conflict
_ADDITIVE_LEMMAS = frozenset({
    'support', 'use', 'contain', 'include', 'feature', 'have', 'has',
    'produce', 'export', 'import', 'teach', 'study', 'offer', 'provide',
    'involve', 'receive', 'attend', 'visit', 'speak', 'know',
    'start', 'begin',
})

# Predicates where same (S,P) with different O IS a conflict
_EXCLUSIVE_LEMMAS = frozenset({
    'capital', 'ceo', 'president', 'founder', 'creator', 'spouse',
    'father', 'mother', 'birthplace', 'deathplace', 'headquarters',
    'nationality', 'gender', 'sex',
    'live', 'reside', 'born', 'die', 'locate', 'headquarter',
    'manage', 'lead', 'head',
    'field', 'sport',
    'schedule', 'deadline', 'due',
    'population', 'release', 'version', 'latest', 'default',
    'port', 'endpoint', 'limit', 'rate',
})

# For copular "be", check subject for these exclusivity signals
_EXCLUSIVE_SUBJECT_NOUNS = frozenset({
    'capital', 'ceo', 'president', 'founder', 'headquarters', 'birthplace',
    'population', 'release', 'deadline', 'port', 'endpoint', 'limit',
    'branch', 'default', 'latest', 'location', 'address', 'spouse',
    'father', 'mother', 'nationality', 'gender',
    'meeting', 'schedule', 'sport', 'field',
})


# ---------------------------------------------------------------------------
# Tier 2: Fine-tuned DeBERTa NLI predicate cardinality (lazy-loaded, cached)
# ---------------------------------------------------------------------------
_CARDINALITY_CACHE: dict[str, str] = {}
_TIER2_MODEL = None
_TIER2_TOKENIZER = None

# Path to fine-tuned model. Set via env var or defaults to model dir next to this file.
_TIER2_MODEL_PATH = os.environ.get(
    "HERMES_CARDINALITY_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "cardinality_model", "best"),
)
# Confidence threshold: below this, return 'unknown' instead of committing
_TIER2_CONFIDENCE = float(os.environ.get("HERMES_CARDINALITY_CONFIDENCE", "0.6"))


def _get_tier2_model():
    """Lazy-load fine-tuned DeBERTa for Tier 2 cardinality classification."""
    global _TIER2_MODEL, _TIER2_TOKENIZER
    if _TIER2_MODEL is None:
        model_path = _TIER2_MODEL_PATH
        if not os.path.isdir(model_path):
            _cr_log.warning("tier2: model not found at %s — Tier 2 disabled", model_path)
            return None, None
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _TIER2_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        _TIER2_MODEL = AutoModelForSequenceClassification.from_pretrained(model_path)
        _TIER2_MODEL.eval()
        _cr_log.info("tier2: loaded cardinality model from %s", model_path)
    return _TIER2_MODEL, _TIER2_TOKENIZER


def _classify_predicate_tier2(pred_str: str, subj_str: str = "") -> str:
    """Tier 2: Fine-tuned NLI classification of predicate cardinality.

    Uses DeBERTa-v3-xsmall fine-tuned on Wikidata property labels (22M params,
    CPU, ~5ms) to classify predicates as exclusive or additive.
    Results are cached permanently — predicate cardinality doesn't change.

    Returns 'exclusive', 'additive', or 'unknown' (low confidence).
    """
    cache_key = f"{pred_str}|{subj_str}"
    if cache_key in _CARDINALITY_CACHE:
        return _CARDINALITY_CACHE[cache_key]

    model, tokenizer = _get_tier2_model()
    if model is None:
        return 'unknown'

    # Convert predicate to natural language
    pred_natural = pred_str.replace('_', ' ').lower()

    # For copular "be", the real predicate is in the subject
    if pred_natural in ('be', 'is', 'was', 'are'):
        pred_natural = subj_str.lower()
        for prefix in ('the ', 'a ', 'an '):
            if pred_natural.startswith(prefix):
                pred_natural = pred_natural[len(prefix):]
                break

    # Construct premise+hypothesis (same format as training data)
    premise = f"The property '{pred_natural}' describes a relationship between entities."
    hypothesis = "This property can only have one value at a time for a given subject."

    import torch
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True,
                       max_length=128, padding=True)
    with torch.no_grad():
        scores = model(**inputs).logits.softmax(dim=-1)[0]

    # DeBERTa NLI output: 0=contradiction, 1=entailment, 2=neutral
    # entailment → exclusive (hypothesis IS true: single-valued)
    # contradiction → additive (hypothesis is FALSE: multi-valued)
    entail_score = float(scores[1])
    contr_score = float(scores[0])

    if entail_score > contr_score and entail_score >= _TIER2_CONFIDENCE:
        result = 'exclusive'
    elif contr_score > entail_score and contr_score >= _TIER2_CONFIDENCE:
        result = 'additive'
    else:
        result = 'unknown'

    _cr_log.debug(
        "tier2_finetuned: pred='%s' subj='%s' → %s (c=%.2f e=%.2f n=%.2f)",
        pred_str, subj_str, result, float(scores[0]), float(scores[1]), float(scores[2]),
    )

    _CARDINALITY_CACHE[cache_key] = result
    return result


def _classify_predicate_cardinality(pred_str: str, subj_str: str = "") -> str:
    """Classify predicate as 'exclusive', 'additive', or 'unknown'.

    Tier 1 rule-based: check tokens against known verb classes.
    For copular 'be', check subject for exclusivity signals.
    Additive check first (conservative — avoid false positive conflict detection).
    Falls through to Tier 2 (NLI zero-shot) when Tier 1 returns 'unknown'.

    Set HERMES_NO_TIER1=1 to skip Tier 1 and go straight to Tier 2.
    """
    # Skip Tier 1 if env var set (for ablation testing)
    if not os.environ.get("HERMES_NO_TIER1"):
        tokens = set(pred_str.lower().replace('_', ' ').split())

        # Tier 1: rule-based
        for token in tokens:
            if token in _ADDITIVE_LEMMAS:
                return 'additive'

        for token in tokens:
            if token in _EXCLUSIVE_LEMMAS:
                return 'exclusive'

        if tokens & {'be', 'is', 'was', 'are'}:
            subj_tokens = set(subj_str.lower().replace('_', ' ').split())
            for token in subj_tokens:
                if token in _EXCLUSIVE_SUBJECT_NOUNS:
                    return 'exclusive'
            return 'unknown'

    # Tier 2: fine-tuned DeBERTa on Wikidata property labels
    return _classify_predicate_tier2(pred_str, subj_str)


def _extract_acl_triples(doc) -> list[tuple[str, str, str]]:
    """Extract (subject, predicate, object) triples using acl-aware copular parsing.

    Handles two sentence patterns:

    1. Copular with acl modifier (most common in MABench):
       "The sport played by Abbiati is association football"
       Parse tree: ROOT=is(AUX), nsubj=sport→acl=played→agent=by→pobj=Abbiati, attr=football
       Triple: (Abbiati, sport_played_by, association football)

    2. Non-copular verbs:
       "Abbiati played basketball"
       Parse tree: ROOT=played(VERB), nsubj=Abbiati, dobj=basketball
       Triple: (Abbiati, play, basketball)

    3. Simple copular (no acl):
       "Abbiati is Italian"
       Triple: (Abbiati, be, Italian)
    """
    triples = []
    for sent in doc.sents:
        root = sent.root

        if root.pos_ == "AUX" and root.lemma_ == "be":
            # Copular sentence
            nsubj = None
            obj_token = None
            for child in root.children:
                if child.dep_ == "nsubj":
                    nsubj = child
                elif child.dep_ in ("attr", "acomp"):
                    obj_token = child

            if nsubj is None or obj_token is None:
                continue

            obj_text = _get_subtree_text(obj_token)

            # Check for acl modifier on nsubj (real predicate)
            acl_child = None
            for child in nsubj.children:
                if child.dep_ == "acl":
                    acl_child = child
                    break

            if acl_child is not None:
                # acl-aware: "The sport played by Abbiati is football"
                # Descend acl→agent/prep→pobj for real subject
                real_subject = None
                prep_text = ""
                for child in acl_child.children:
                    if child.dep_ in ("agent", "prep"):
                        prep_text = child.text.lower()
                        for gchild in child.children:
                            if gchild.dep_ == "pobj":
                                real_subject = _get_subtree_text(gchild)
                                break
                        break

                if real_subject:
                    canonical_pred = f"{nsubj.text}_{acl_child.lemma_}_{prep_text}"
                    triples.append((real_subject, canonical_pred, obj_text))
                else:
                    # acl without agent/prep → fallback to nsubj as subject
                    triples.append((_get_subtree_text(nsubj), nsubj.text, obj_text))
            else:
                # Simple copular: "Abbiati is Italian"
                triples.append((_get_subtree_text(nsubj), "be", obj_text))

        elif root.pos_ == "VERB":
            # Non-copular: nsubj/nsubjpass→VERB→dobj/attr/pobj/prep→pobj
            nsubj = None
            for child in root.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    nsubj = child
                    break
            if nsubj is None:
                continue

            pred = root.lemma_
            found_direct = False
            for child in root.children:
                if child.dep_ in ("dobj", "attr", "pobj"):
                    obj_text = _get_subtree_text(child)
                    triples.append((_get_subtree_text(nsubj), pred, obj_text))
                    found_direct = True

            # Prep-chain objects: "works in the field of Y",
            # "was born in the city of Y", "was created in the country of Y"
            # Descend prep→pobj→prep→pobj to extract the deepest pobj.
            if not found_direct:
                for child in root.children:
                    if child.dep_ in ("prep", "agent"):
                        # Build predicate from prep chain: "work_in_field_of"
                        prep_parts = [pred, child.text.lower()]
                        deepest_pobj = None
                        current = child
                        for _ in range(3):  # max depth
                            pobj = None
                            next_prep = None
                            for gc in current.children:
                                if gc.dep_ == "pobj":
                                    pobj = gc
                                elif gc.dep_ == "prep":
                                    next_prep = gc
                            if pobj is None:
                                break
                            if next_prep is not None:
                                prep_parts.append(pobj.text.lower())
                                prep_parts.append(next_prep.text.lower())
                                current = next_prep
                            else:
                                deepest_pobj = pobj
                                break
                        if deepest_pobj is not None:
                            canonical_pred = "_".join(prep_parts)
                            obj_text = _get_subtree_text(deepest_pobj)
                            triples.append((_get_subtree_text(nsubj), canonical_pred, obj_text))
                            break  # one triple per sentence

    return triples


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
        self._model = AutoModel.from_pretrained(self._model_name, torch_dtype=torch.float32).to(self._device)
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
                batch_texts = uncached_texts[start : start + batch_size]
                inputs = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
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
        cr_per_hop: bool = True,
        cr_entity_gate: bool = False,
        cr_overfetch: int = 3,
        cosine_dedup: bool = False,
        cosine_dedup_threshold: float = 0.95,
        entity_retrieval: bool = False,
        entity_first: bool = False,
        entity_first_aggressive: bool = False,
        bm25_hops: bool = False,
        chain_cache: bool = False,
        rebel_kg: bool = False,
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
        self._cr_per_hop = cr_per_hop
        self._cr_entity_gate = cr_entity_gate
        self._cr_overfetch = cr_overfetch
        self.cosine_dedup = cosine_dedup
        self._cosine_dedup_threshold = cosine_dedup_threshold
        self.entity_retrieval = entity_retrieval
        self.entity_first = entity_first
        self.entity_first_aggressive = entity_first_aggressive
        self.bm25_hops = bm25_hops
        self.chain_cache = chain_cache
        self._bm25_index = BM25Index() if bm25_hops else None
        self._chain_cache = ChainCache(similarity_threshold=0.95) if chain_cache else None
        self.rebel_kg = rebel_kg
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
        self._ingestion = IngestionPipeline(
            IngestionConfig(
                coref_device=_coref_device,
                coref_enabled=True,
                boundary_threshold=0.2,
                strip_numbered_prefix=False,
            )
        )
        self._entity_accumulator = _MutableEntityGraph()

        # Triadic memory: structural recall via Overmann sparse 3D tensor
        self._triadic = TriadicMemory(n=triadic_n, p=triadic_p) if triadic_retrieval else None

        self._store_count = 0
        self._start_time = time.time()

        # GLiNER entity extractor (lazy init)
        self._gliner_model = None
        self._gliner_labels = ["person", "organization", "location", "country", "city"]

        # GLiNER entity index: entity_name.lower() → set of fact indices
        # Built during ingestion, used for entity-aware retrieval at query time.
        # Fact index = position in coupled_engine.memory_store.
        self._gliner_entity_index: dict[str, set[int]] = defaultdict(set)

        # REBEL KG: dict-of-dicts knowledge graph built at ingest time.
        # Forward: kg[subject_lower][relation_lower] = [original-case objects]
        # Reverse: kg_rev[object_lower][relation_lower] = [original-case subjects]
        # Query-time: graph hop bypasses retrieval+LLM for structured lookups.
        if rebel_kg:
            self._rebel_kg: dict[str, dict[str, list[str]]] = {}
            self._rebel_kg_reverse: dict[str, dict[str, list[str]]] = {}
            self._rebel_model = None
            self._rebel_tokenizer = None
            self._rebel_device = None
        else:
            self._rebel_kg = None
            self._rebel_kg_reverse = None

        # LLM client (lazy init)
        self._llm_client = None

    def _ensure_gliner(self):
        """Lazy-load GLiNER entity extractor."""
        if getattr(self, "_gliner_model", None) is not None:
            return
        try:
            from gliner import GLiNER
            self._gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
            logging.getLogger(__name__).info("GLiNER loaded (gliner_multi-v2.1)")
        except ImportError:
            logging.getLogger(__name__).warning(
                "GLiNER not installed — entity extraction disabled. "
                "Install with: pip install gliner"
            )
            self._gliner_model = False  # sentinel: tried and failed

    def _extract_entities_batch(
        self, texts: list[str]
    ) -> tuple[list[str | None], list[list[dict]]]:
        """Extract entities from texts using GLiNER.

        Returns:
            primary_entities: one primary entity per text (person-preferred),
                or None if no entity found.
            all_entities: list of entity dicts per text, each dict has
                keys 'text', 'label', 'score'.
        """
        self._ensure_gliner()
        if not getattr(self, "_gliner_model", None):
            return [None] * len(texts), [[] for _ in texts]

        labels = getattr(self, "_gliner_labels",
                         ["person", "organization", "location", "country", "city"])
        results = self._gliner_model.batch_predict_entities(
            texts, labels, threshold=0.4,
        )
        primary: list[str | None] = []
        for ents in results:
            if not ents:
                primary.append(None)
                continue
            persons = [e for e in ents if e["label"] == "person"]
            if persons:
                primary.append(persons[0]["text"])
            else:
                best = max(ents, key=lambda e: e["score"])
                primary.append(best["text"])
        return primary, results

    def _extract_primary_entity(self, texts: list[str]) -> list[str | None]:
        """Extract the primary subject entity from each text using GLiNER."""
        primary, _ = self._extract_entities_batch(texts)
        return primary

    # ------------------------------------------------------------------
    # REBEL KG: knowledge graph built at ingest, queried at decompose
    # ------------------------------------------------------------------

    def _ensure_rebel(self):
        """Lazy-load REBEL model (Babelscape/rebel-large, 350M params)."""
        if getattr(self, "_rebel_model", None) is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._rebel_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._rebel_tokenizer = AutoTokenizer.from_pretrained(
                "Babelscape/rebel-large"
            )
            self._rebel_model = AutoModelForSeq2SeqLM.from_pretrained(
                "Babelscape/rebel-large",
                device_map={"": self._rebel_device},
            )
            self._rebel_model.eval()
            logging.getLogger(__name__).info(
                "REBEL loaded (rebel-large) on %s", self._rebel_device
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                "REBEL load failed: %s — KG extraction disabled", e
            )
            self._rebel_model = False  # sentinel: tried and failed

    @staticmethod
    def _parse_rebel_output(decoded: str) -> list[dict]:
        """Parse REBEL decoder output into (head, relation, tail) dicts.

        REBEL encodes: <triplet> subject <subj> relation <obj> object
        """
        triplets = []
        for part in decoded.split("<triplet>"):
            part = part.strip()
            if not part or part in ("<s>", "</s>"):
                continue
            part = part.replace("</s>", "").replace("<pad>", "").strip()
            if "<subj>" in part and "<obj>" in part:
                subj_split = part.split("<subj>")
                head = subj_split[0].strip()
                rest = subj_split[1] if len(subj_split) > 1 else ""
                obj_split = rest.split("<obj>")
                relation = obj_split[0].strip()
                tail = obj_split[1].strip() if len(obj_split) > 1 else ""
                if head and relation and tail:
                    triplets.append(
                        {"head": head, "relation": relation, "tail": tail}
                    )
        return triplets

    def _rebel_extract_batch(self, texts: list[str]) -> list[list[dict]]:
        """Extract triples from a batch of texts using REBEL.

        Returns a list of triple-lists, one per input text.
        """
        self._ensure_rebel()
        if not self._rebel_model:
            return [[] for _ in texts]

        import torch

        all_triples: list[list[dict]] = []
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._rebel_tokenizer(
                batch,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self._rebel_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._rebel_model.generate(
                    **inputs, max_length=128, num_beams=1
                )
            decoded = self._rebel_tokenizer.batch_decode(
                outputs, skip_special_tokens=False
            )
            for dec in decoded:
                all_triples.append(self._parse_rebel_output(dec))

        return all_triples

    def _rebel_kg_add(self, head: str, relation: str, tail: str) -> None:
        """Add a triple to the REBEL KG (forward + reverse index)."""
        h = head.lower().strip()
        r = relation.lower().strip()
        t_key = tail.lower().strip()
        if not h or not r or not t_key:
            return

        # Forward: subject → relation → [objects]
        if h not in self._rebel_kg:
            self._rebel_kg[h] = {}
        if r not in self._rebel_kg[h]:
            self._rebel_kg[h][r] = []
        if tail.strip() not in self._rebel_kg[h][r]:
            self._rebel_kg[h][r].append(tail.strip())

        # Reverse: object → relation → [subjects]
        if t_key not in self._rebel_kg_reverse:
            self._rebel_kg_reverse[t_key] = {}
        if r not in self._rebel_kg_reverse[t_key]:
            self._rebel_kg_reverse[t_key][r] = []
        if head.strip() not in self._rebel_kg_reverse[t_key][r]:
            self._rebel_kg_reverse[t_key][r].append(head.strip())

    # Relation keyword matching for KG query-time hop
    _KG_STOP_WORDS = frozenset({
        "what", "is", "the", "of", "a", "an", "did", "does", "was", "were",
        "where", "who", "which", "how", "do", "has", "had", "that", "this",
        "in", "on", "at", "to", "for", "by", "with", "from", "are", "be",
    })
    _KG_STEM_MAP = {
        "born": "birth", "die": "death", "died": "death",
        "married": "spouse", "marry": "spouse",
        "play": "sport", "plays": "sport", "played": "sport",
        "follow": "religion", "follows": "religion", "followed": "religion",
        "work": "field", "works": "field", "worked": "field",
        "educated": "educated", "studied": "educated",
        "created": "origin", "compose": "genre", "composed": "genre",
        "perform": "genre", "performed": "genre",
        "hold": "position", "held": "position",
        "capital": "capital",
    }

    def _rebel_kg_hop(self, sub_query: str) -> str | None:
        """Try to answer a sub-query via REBEL KG graph traversal.

        1. Find KG entities that appear in the sub-query (longest first).
        2. For each, find relations whose keywords overlap the sub-query.
        3. Return the object/subject if a clear match is found.
        4. Return None to fall through to retrieval+LLM.
        """
        if not self._rebel_kg:
            return None

        sq_lower = sub_query.lower()

        # Collect candidate entities from both forward and reverse KG
        candidates: list[tuple[str, dict, bool]] = []  # (entity_key, relations, is_reverse)
        for entity_key, rels in self._rebel_kg.items():
            if entity_key in sq_lower:
                candidates.append((entity_key, rels, False))
        for entity_key, rels in self._rebel_kg_reverse.items():
            if entity_key in sq_lower:
                candidates.append((entity_key, rels, True))

        if not candidates:
            return None

        # Prefer longest entity match (most specific)
        candidates.sort(key=lambda c: len(c[0]), reverse=True)

        # Tokenize query (minus stop words, strip punctuation)
        q_tokens = set()
        for w in sq_lower.split():
            w = w.strip(".,?!;:'\"()[]")
            if w and w not in self._KG_STOP_WORDS:
                q_tokens.add(w)
        # Add stemmed variants
        stemmed = set()
        for t in q_tokens:
            stemmed.add(t)
            if t in self._KG_STEM_MAP:
                stemmed.add(self._KG_STEM_MAP[t])

        for entity_key, relations, is_reverse in candidates:
            # Remove entity from query tokens to avoid self-matching
            entity_tokens = set(entity_key.split())
            q_clean = stemmed - entity_tokens

            best_rel = None
            best_overlap = 0
            for rel, values in relations.items():
                rel_tokens = set(rel.replace("_", " ").split()) - self._KG_STOP_WORDS
                if not rel_tokens:
                    continue
                overlap = len(rel_tokens & q_clean)
                # Also check if full relation is substring of query
                if overlap == 0 and rel in sq_lower:
                    overlap = len(rel_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_rel = rel

            if best_rel is not None and best_overlap > 0:
                values = relations[best_rel]
                # Return last added (most recent = temporal supersession)
                answer = values[-1]
                _cr_log.info(
                    "KG hop: entity=%s rel=%s → %s (reverse=%s)",
                    entity_key, best_rel, answer, is_reverse,
                )
                return answer

        return None

    def _entity_text_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Find facts by entity name match using GLiNER entity index.

        Extracts entities from the query using GLiNER, looks them up in
        the inverted entity index built during ingestion, and returns
        matching facts ranked by number of matching entities.

        Falls back to regex proper-noun extraction if GLiNER is unavailable.
        """
        # Extract query entities via GLiNER
        query_entities = self._extract_query_entities(query)

        if not query_entities:
            return []

        # Look up each query entity in the inverted index
        store = self.coupled_engine.memory_store
        fact_scores: dict[int, int] = {}  # fact_idx → match count
        for ent_text in query_entities:
            key = ent_text.lower().strip()
            matching_indices = self._gliner_entity_index.get(key, set())
            for idx in matching_indices:
                if idx < len(store):
                    fact_scores[idx] = fact_scores.get(idx, 0) + 1

        if not fact_scores:
            return []

        # Sort by match count desc, take top_k
        ranked = sorted(fact_scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, match_count in ranked:
            entry = store[idx]
            results.append({
                "text": entry.text,
                "score": float(match_count),
                "recency": getattr(entry, "recency", 0.0),
                "source": "entity_index",
            })
        return results

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entity names from a query string using GLiNER.

        Returns a list of entity text strings found in the query.
        Falls back to regex proper-noun extraction if GLiNER is unavailable.
        """
        self._ensure_gliner()
        if getattr(self, "_gliner_model", None) and self._gliner_model is not False:
            labels = self._gliner_labels
            results = self._gliner_model.predict_entities(
                query, labels, threshold=0.3,  # lower threshold for short queries
            )
            return [e["text"] for e in results if len(e["text"].strip()) > 1]

        # Fallback: regex proper nouns (less precise but works without GLiNER)
        import re as _re
        entities: list[str] = []
        _stop = {"what", "where", "which", "when", "who", "how", "that", "this",
                 "with", "from", "about", "based", "their", "does", "have", "been",
                 "associated", "knowledge", "pool", "provided", "the", "question"}
        for m in _re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query):
            word = m.group(1)
            if word.lower() not in _stop:
                entities.append(word)
        return entities

    def _entity_first_retrieve(self, query: str) -> list[dict] | None:
        """Entity-first retrieval: narrow to an entity's facts before ranking.

        Extracts entities from the query, looks them up in the GLiNER
        entity index, and returns ONLY that entity's facts ranked by
        cosine similarity to the query.

        Returns None if no entity is found (caller should fall back to
        BM25/cosine). Returns a list of result dicts if entity matched.

        This eliminates template confusion: instead of ranking 17K facts
        where 500 share the same template (mean pairwise cos=0.976),
        we rank only the 3-5 facts belonging to the target entity.
        """
        query_entities = self._extract_query_entities(query)
        if not query_entities:
            return None

        store = self.coupled_engine.memory_store
        # Union of fact indices across all extracted entities
        matched_indices: set[int] = set()
        for ent_text in query_entities:
            key = ent_text.lower().strip()
            idx_set = self._gliner_entity_index.get(key, set())
            matched_indices.update(idx_set)

        if not matched_indices:
            return None

        # Cosine-rank within the entity's fact subset
        q_emb = self._embed_query(query)
        scored: list[tuple[float, int]] = []
        for idx in matched_indices:
            if idx >= len(store):
                continue
            entry = store[idx]
            emb = getattr(entry, "embedding", None)
            if emb is not None:
                sim = float(np.dot(q_emb, emb) / (
                    np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-9
                ))
            else:
                sim = 0.0
            scored.append((sim, idx))

        # Sort by cosine similarity descending
        scored.sort(key=lambda x: -x[0])

        results = []
        for sim, idx in scored[: self.retrieve_num]:
            entry = store[idx]
            results.append({
                "text": entry.text,
                "score": sim,
                "recency": getattr(entry, "recency", 0.0),
                "source": "entity_first",
            })
        return results

    def rebuild_entity_index(self, batch_size: int = 64) -> int:
        """Rebuild GLiNER entity index from existing memory_store.

        Use after loading a cache that was built without the entity index.
        Runs GLiNER batch extraction on all facts (~24 min for 17K facts on GPU).
        Returns the number of unique entities indexed.
        """
        store = self.coupled_engine.memory_store
        n = len(store)
        if n == 0:
            return 0

        self._ensure_gliner()
        if not getattr(self, "_gliner_model", None):
            print("GLiNER not available — cannot rebuild entity index")
            return 0

        self._gliner_entity_index = defaultdict(set)
        labels = self._gliner_labels

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            texts = [store[i].text for i in range(start, end)]
            results = self._gliner_model.batch_predict_entities(
                texts, labels, threshold=0.4,
            )
            for i, ents in enumerate(results):
                fact_idx = start + i
                for ent in ents:
                    key = ent["text"].lower().strip()
                    if len(key) > 1:
                        self._gliner_entity_index[key].add(fact_idx)
            if (end % 1000) < batch_size:
                print(f"  Entity index: {end}/{n} facts processed, "
                      f"{len(self._gliner_entity_index)} unique entities")

        return len(self._gliner_entity_index)

    def _entity_graph_expand(
        self, seed_facts: list[str], top_k: int = 5
    ) -> list[dict]:
        """Expand seed facts via entity co-mention graph.

        For each seed fact, find its entities in the index, then find OTHER
        facts that share those entities. This enables multi-hop bridging:
        "X plays basketball" + "basketball created in USA" → connects X to USA.

        Returns facts NOT already in seed_facts, ranked by number of
        entity connections to seed facts.
        """
        if not self._gliner_entity_index:
            return []

        store = self.coupled_engine.memory_store
        seed_set = set(seed_facts)

        # Find all entities mentioned in seed facts
        seed_entities: set[str] = set()
        seed_indices: set[int] = set()
        for fact_text in seed_facts:
            for ent_key, fact_indices in self._gliner_entity_index.items():
                for idx in fact_indices:
                    if idx < len(store) and store[idx].text == fact_text:
                        seed_entities.add(ent_key)
                        seed_indices.add(idx)
                        break

        if not seed_entities:
            return []

        # Find facts that share entities with seed facts (but aren't seeds)
        neighbor_scores: dict[int, int] = {}
        for ent_key in seed_entities:
            for idx in self._gliner_entity_index.get(ent_key, set()):
                if idx not in seed_indices and idx < len(store):
                    neighbor_scores[idx] = neighbor_scores.get(idx, 0) + 1

        if not neighbor_scores:
            return []

        ranked = sorted(neighbor_scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in ranked:
            entry = store[idx]
            if entry.text not in seed_set:
                results.append({
                    "text": entry.text,
                    "score": float(score),
                    "recency": getattr(entry, "recency", 0.0),
                    "source": "entity_graph",
                })
        return results

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
        import time as _time
        _t0 = _time.perf_counter()
        result = self._ingestion.segment(text)
        _t_seg = _time.perf_counter()

        # --- Parallel extraction: embed, GLiNER, REBEL are independent ---
        # All three take fact_texts as input with no shared mutable state.
        # PyTorch releases the GIL during GPU kernels, so threads overlap
        # GPU work and CPU↔GPU transfers via CUDA stream interleaving.
        from concurrent.futures import ThreadPoolExecutor

        fact_texts = [f.text for f in result.facts]
        rebel_triples_batch: list[list[dict]] | None = None
        run_rebel = getattr(self, "_rebel_kg", None) is not None

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_embed = pool.submit(self._scorer.embed_batch, fact_texts)
            if run_rebel:
                fut_rebel = pool.submit(self._rebel_extract_batch, fact_texts)

            embeddings = fut_embed.result()
            if run_rebel:
                rebel_triples_batch = fut_rebel.result()

        # GLiNER entity extraction skipped — both consumers are disabled:
        #   entity_first (default False), entity_retrieval (default False),
        #   dream merge gating (dream disabled on batch ingest).
        # Pass None/empty to satisfy downstream signatures.
        primary_entities = [None] * len(fact_texts)
        all_entities: list[list] = [[] for _ in fact_texts]
        _t_gpu = _time.perf_counter()
        print(f"TIMING segment={_t_seg-_t0:.2f}s gpu_parallel={_t_gpu-_t_seg:.2f}s facts={len(fact_texts)}", flush=True)

        # Build entity index: map each entity to its fact's index in memory_store.
        # Use current memory_store length as the insertion offset (safer than
        # _store_count which tracks recency, not position).
        base_idx = len(self.coupled_engine.memory_store)
        for i, ents in enumerate(all_entities):
            fact_idx = base_idx + i
            for ent in ents:
                key = ent["text"].lower().strip()
                if len(key) > 1:  # skip single-char entities
                    self._gliner_entity_index[key].add(fact_idx)

        for fi, (fact, emb, entity) in enumerate(zip(result.facts, embeddings, primary_entities)):
            # V1: Text-based store
            self.orchestrator.store(content=fact.text, embedding=emb)

            # V2: Coupled engine store (no serial number prefix -- D3)
            # Returns the memory_store index (may replace an existing entry
            # when cosine > contradiction_threshold, so idx != _store_count).
            ms_idx = self.coupled_engine.store(
                text=fact.text,
                embedding=emb,
                recency=float(self._store_count),
                entity=entity,
            )

            # BM25: index fact for lexical sub-query hop retrieval.
            # Use ms_idx (memory_store position) as doc_id so BM25 hits
            # map directly to memory_store[doc_id].
            if getattr(self, "_bm25_index", None) is not None:
                self._bm25_index.add(ms_idx, fact.text)

            # Triadic: store fact's triples in the 3D tensor
            if getattr(self, "_triadic", None) is not None:
                self._triadic.store_fact(fact.text, list(fact.triples))

            # REBEL KG: add extracted triples to knowledge graph
            if rebel_triples_batch is not None and fi < len(rebel_triples_batch):
                for triple in rebel_triples_batch[fi]:
                    self._rebel_kg_add(
                        triple["head"], triple["relation"], triple["tail"]
                    )

            # Register fact in belief index (before conflict detection)
            if self._belief_index is not None:
                self._belief_index.register_fact(fact.text)

            # --- Conflict detection: triple-based with entity gating ---
            # Same subject+predicate, different object → potential conflict,
            # but only act if the old and new facts share a named entity.
            # When supersession=True: populate flat _superseded_texts set.
            # When belief is active: update belief scores.
            # NOTE: conflict_resolution now uses query-time CR (_query_time_cr),
            # so it no longer needs ingestion-time triple index building.
            if (self.supersession or self._belief_index) and fact.triples:
                new_entities = set(e.lower() for e in fact.entities) if fact.entities else set()
                self._fact_entities[fact.text] = fact.entities
                for subj, pred, obj in fact.triples:
                    key = (subj.lower().strip(), pred.lower().strip())
                    new_obj = obj.lower().strip()
                    if not key[0] or not key[1]:
                        continue  # skip degenerate triples
                    for existing_obj, existing_text in self._triple_index[key]:
                        if existing_obj != new_obj:
                            # Entity gate: require shared named entity via substring matching.
                            # spaCy NER set intersection misses entities when it extracts
                            # different spans from near-identical sentences.
                            old_entities = [e.lower() for e in self._fact_entities.get(existing_text, ())]
                            existing_lower = existing_text.lower()
                            fact_lower = fact.text.lower()
                            shared = any(e in existing_lower for e in new_entities if len(e) > 2) or any(
                                e in fact_lower for e in old_entities if len(e) > 2
                            )
                            if not shared:
                                continue  # no shared entity → not a real conflict

                            # Binary supersession (when enabled)
                            if self.supersession:
                                self._superseded_texts.add(existing_text)

                            # Chain cache: invalidate chains using the superseded fact
                            if getattr(self, "_chain_cache", None) is not None:
                                self._chain_cache.invalidate_fact(existing_text)

                            # Bayesian belief update (when enabled)
                            if self._belief_index is not None:
                                self._belief_index.on_conflict(
                                    existing_text,
                                    fact.text,
                                )
                    self._triple_index[key].append((new_obj, fact.text))

            self._store_count += 1

        # Synonym dict: feed coref clusters and verb mappings to triadic
        if getattr(self, "_triadic", None) is not None:
            if getattr(result, "coref_clusters", None):
                self._triadic.add_coref_clusters(result.coref_clusters)
            if getattr(result, "verb_mappings", None):
                self._triadic.add_verb_mappings(result.verb_mappings)

        # Accumulate entity graph with global offset
        if hasattr(self, "_entity_accumulator"):
            offset = self._store_count - len(result.facts)
            self._entity_accumulator.merge(result.entity_graph, offset)

        # Belief propagation: penalize facts connected to degraded facts
        if self._belief_index is not None and hasattr(self, "_entity_accumulator"):
            self._belief_index.propagate(
                self._entity_accumulator,
                self.coupled_engine.memory_store,
            )

        # Flush session buffer per chunk if Hebbian enabled (D8)
        if (
            hasattr(self.coupled_engine, "_session_buffer")
            and len(getattr(self.coupled_engine, "_session_buffer", [])) > 0
        ):
            self.coupled_engine.flush_session()

        # Supersession diagnostics: log every 50 stores
        if self.supersession and self._store_count % 50 == 0 and self._store_count > 0:
            import logging

            logging.getLogger("supersession").info(
                "STATS after %d stores: %d unique (S,P) keys, %d superseded texts, %d graph edges, %d total triples",
                self._store_count,
                len(self._triple_index),
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
                return prompt[q_start + len("Question:") : q_end].strip()
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

        # Chain cache: check for cached decompose result (zero LLM calls)
        _q_emb = None
        if getattr(self, "_chain_cache", None) is not None and self.decompose_query:
            _q_emb = self._scorer.embed(bare_question)
            cached = self._chain_cache.lookup(_q_emb)
            if cached is not None:
                qt = time.time() - query_start
                return {
                    "output": cached.final_answer,
                    "input_len": 0,
                    "output_len": 0,
                    "memory_construction_time": qt,
                    "query_time_len": qt,
                }

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

            final_kg_answer = None  # set if the last hop is KG-answered

            for i, sq_template in enumerate(subqueries):
                # Substitute any {answer_N} placeholders from prior hops
                sq = sq_template
                for key, val in answers.items():
                    sq = sq.replace(key, val)

                # --- REBEL KG fast lane: deterministic graph hop ---
                # Try graph[entity][relation] BEFORE retrieval+LLM.
                # If hit: skip the entire retrieve/judge/extract cycle.
                # If miss: fall through to existing path unchanged.
                if getattr(self, "_rebel_kg", None) is not None:
                    kg_answer = self._rebel_kg_hop(sq)
                    if kg_answer is not None:
                        answers[f"{{answer_{i + 1}}}"] = kg_answer
                        evidence_chain.append({
                            "q": sq,
                            "a": kg_answer,
                            "evidence": [f"[KG: {kg_answer}]"],
                        })
                        sq_chunks.append([f"[KG] {sq} -> {kg_answer}"])
                        sq_chunks_raw.append([])
                        if i == len(subqueries) - 1:
                            final_kg_answer = kg_answer
                        _cr_log.info(
                            "KG hop %d/%d: %s -> %s",
                            i + 1, len(subqueries), sq, kg_answer,
                        )
                        continue  # skip retrieval+LLM for this hop

                # Mini iterative loop per sub-query: retrieve → judge →
                # retry with targeted NEED if first retrieval misses.
                sq_all_texts: list[str] = []
                sq_all_results: list[dict] = []
                sq_seen: set[str] = set()
                current_sq = sq
                max_sq_retries = 2  # 1 original + 1 retry
                entity_first_hit = False

                for retry in range(max_sq_retries):
                    # Entity-first retrieval: narrow to entity's facts
                    # before cosine ranking. Eliminates template confusion
                    # (88% of failures: 500 template-similar facts with
                    # mean pairwise cos=0.976 → wrong entity picked).
                    sq_results = None
                    if getattr(self, "entity_first", False) and self._gliner_entity_index:
                        ef_results = self._entity_first_retrieve(current_sq)
                        if ef_results is not None:
                            sq_results = ef_results
                            entity_first_hit = True

                    # Fallback: BM25 or cosine when entity-first misses
                    if sq_results is None:
                        if getattr(self, "bm25_hops", False) and getattr(self, "_bm25_index", None) is not None:
                            bm25_hits = self._bm25_index.query(current_sq, top_k=self.retrieve_num * 3)
                            sq_results = [
                                {"text": self.coupled_engine.memory_store[doc_id].text,
                                 "score": score}
                                for doc_id, score in bm25_hits
                                if doc_id < len(self.coupled_engine.memory_store)
                            ]
                            sq_results = self._apply_post_retrieval(sq_results)
                        else:
                            sq_results = self._retrieve_v2(current_sq)
                    for r in sq_results:
                        key = r["text"].strip()[:200]
                        if key not in sq_seen:
                            sq_seen.add(key)
                            sq_all_texts.append(r["text"])
                            sq_all_results.append(r)

                    # Triadic expansion on this sub-query's results
                    if getattr(self, "_triadic", None) is not None:
                        hop_texts = [r["text"] for r in sq_results]
                        if hop_texts:
                            for t in self._triadic.expand(
                                hop_texts,
                                top_k=self._triadic_expand_k,
                            ):
                                tkey = t.strip()[:200]
                                if tkey not in sq_seen:
                                    sq_seen.add(tkey)
                                    sq_all_texts.append(t)

                    # AGGRESSIVE entity-first: entity matched + triadic
                    # done. Skip retry loop and judge to prevent BM25/cosine
                    # from re-polluting context with wrong-entity facts.
                    if entity_first_hit and getattr(self, "entity_first_aggressive", False):
                        break

                    # On last retry or final sub-query, skip judge
                    if retry >= max_sq_retries - 1:
                        break

                    # Judge: can we answer this sub-query?
                    try:
                        judge_resp = self._llm_client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "Given retrieved facts, can you answer "
                                        "this specific sub-question?\n"
                                        "If YES: respond ANSWER_READY\n"
                                        "If NO: respond NEED: <3-6 word search "
                                        "using concrete entity names>"
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"Sub-question: {current_sq}\n\n"
                                        f"Facts:\n" + "\n".join(sq_all_texts[:10]) + "\n\nCan you answer?"
                                    ),
                                },
                            ],
                            temperature=0.0,
                            max_tokens=64,
                        )
                        jt = (judge_resp.choices[0].message.content or "").strip()
                    except Exception:
                        break  # LLM error → use what we have

                    if jt.startswith("ANSWER_READY"):
                        break
                    elif "NEED:" in jt:
                        need = jt[jt.index("NEED:") + 5 :].strip().strip("'\"")
                        if need and need != current_sq:
                            current_sq = need
                        else:
                            break
                    else:
                        break  # unparseable → use what we have

                # Entity index search: find facts by GLiNER entity lookup.
                # Embedding retrieval groups by template similarity,
                # so "What sport did X play?" returns facts about other
                # athletes. Entity index ensures facts about X specifically
                # are included. Also expand via entity co-mention graph
                # for multi-hop bridging.
                # SKIP when aggressive entity-first already fired — entity
                # facts are already the primary results; adding more via
                # text search would re-pollute with wrong-entity facts.
                _skip_entity_supplement = (
                    entity_first_hit and getattr(self, "entity_first_aggressive", False)
                )
                if self.entity_retrieval and not _skip_entity_supplement:
                    ent_results = self._entity_text_search(sq, top_k=10)
                    for r in ent_results:
                        ekey = r["text"].strip()[:200]
                        if ekey not in sq_seen:
                            sq_seen.add(ekey)
                            sq_all_texts.append(r["text"])
                            sq_all_results.append(r)

                    # Entity graph expansion: find facts sharing entities
                    # with the entity-matched facts (multi-hop bridging).
                    if ent_results:
                        seed_texts = [r["text"] for r in ent_results]
                        graph_results = self._entity_graph_expand(
                            seed_texts, top_k=5,
                        )
                        for r in graph_results:
                            gkey = r["text"].strip()[:200]
                            if gkey not in sq_seen:
                                sq_seen.add(gkey)
                                sq_all_texts.append(r["text"])
                                sq_all_results.append(r)

                # Per-hop conflict resolution: remove stale facts
                # BEFORE the LLM extracts an intermediate answer.
                # Without this, the LLM sees all versions of a fact
                # (e.g., 4 different sports for Abbiati) and picks wrong,
                # poisoning all downstream hops.
                if self.cosine_dedup:
                    sq_all_texts = self._pairwise_cosine_dedup(sq_all_texts)
                if self.conflict_resolution and self._cr_per_hop:
                    sq_all_texts = self._query_time_cr(sq_all_texts)
                elif not self.cosine_dedup and self._belief_index is not None:
                    sq_all_texts = [t for t in sq_all_texts if not self._belief_index.is_excluded(t)]

                sq_chunks.append(sq_all_texts)
                sq_chunks_raw.append(sq_all_results)

                # Extract intermediate answer for chaining to next hop
                if i < len(subqueries) - 1 and sq_all_texts:
                    short_ans = self._extract_short_answer(
                        sq,
                        sq_all_texts,
                        evidence_chain,
                    )
                    if short_ans:
                        answers[f"{{answer_{i + 1}}}"] = short_ans
                        evidence_chain.append(
                            {
                                "q": sq,
                                "a": short_ans,
                                "evidence": sq_all_texts[:2],
                            }
                        )

            # Short-circuit: if the final hop was answered by KG, return directly.
            # No need for retrieval+LLM — we have a deterministic answer.
            if final_kg_answer is not None:
                qt = time.time() - query_start
                if getattr(self, "_chain_cache", None) is not None and _q_emb is not None:
                    self._chain_cache.store(
                        _q_emb, bare_question, final_kg_answer, evidence_chain
                    )
                return {
                    "output": final_kg_answer,
                    "input_len": 0,
                    "output_len": 0,
                    "memory_construction_time": qt,
                    "query_time_len": qt,
                }

            # Merge sub-query and original results
            orig_results = self._retrieve_v2(bare_question)
            # BM25 rerank original results for entity discrimination
            if getattr(self, "bm25_hops", False) and getattr(self, "_bm25_index", None) is not None:
                bm25_scores = self._bm25_index.score_texts(
                    bare_question, [r["text"] for r in orig_results]
                )
                if any(s > 0 for s in bm25_scores):
                    paired = sorted(
                        zip(bm25_scores, orig_results),
                        key=lambda x: -x[0],
                    )
                    orig_results = [r for _, r in paired]
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
                v2_texts = [key_to_text[k] for k in sorted_keys[: self.retrieve_num]]
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
        if getattr(self, "_triadic", None) is not None and v2_texts:
            triadic_texts = self._triadic.expand(v2_texts, top_k=self._triadic_expand_k)
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
        # - Cosine dedup: pairwise cosine > threshold → keep newest (no NLP)
        # - Conflict resolution (query-time): spaCy triples, keep newest per (S,P)
        # - Belief (soft): demotes stale facts via P(current).
        # - Supersession (binary): fallback when neither is enabled.
        # Cosine dedup can run standalone or stacked with triple-based CR.
        if self.cosine_dedup:
            merged_chunks = self._pairwise_cosine_dedup(merged_chunks)
        if self.conflict_resolution:
            merged_chunks = self._query_time_cr(merged_chunks)
        elif not self.cosine_dedup:
            # Only fall through to weaker methods if neither dedup nor CR active
            if self._belief_index is not None:
                merged_chunks = [c for c in merged_chunks if not self._belief_index.is_excluded(c)]
            elif self.supersession and self._superseded_texts:
                merged_chunks = [c for c in merged_chunks if c not in self._superseded_texts]

        # Build context for LLM via outgestion formatter
        final_chunks = merged_chunks[: self.retrieve_num]
        memory_context, outgestion_instructions = self._format_outgestion(final_chunks, _metadata_by_key)
        memory_time = time.time() - query_start

        # Generate answer via LLM
        answer, input_tokens, output_tokens, gen_time = self._generate_answer(
            question,
            memory_context,
            extra_system_instructions=outgestion_instructions,
        )
        query_time = time.time() - query_start

        # Chain cache: store completed decompose chain for future reuse
        if getattr(self, "_chain_cache", None) is not None and self.decompose_query and _q_emb is not None:
            chain = CachedChain(
                query_embedding=_q_emb,
                query_text=bare_question,
                sub_queries=subqueries,
                intermediate_answers=answers,
                evidence_facts=list(final_chunks),
                final_answer=answer,
                chain_id=len(self._chain_cache),
            )
            self._chain_cache.store(chain)

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
                    {
                        "role": "system",
                        "content": (
                            "You decompose multi-hop questions into chained "
                            "sub-questions. Use {answer_N} placeholders. "
                            "Be concise."
                        ),
                    },
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
            for prefix in ("1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "4)", "5)", "-"):
                if line.startswith(prefix):
                    line = line[len(prefix) :].strip()
                    break
            if line:
                subqs.append(line)

        # Cap at max_hops
        subqs = subqs[: self.decompose_max_hops]
        return subqs if subqs else [question]

    def _extract_short_answer(
        self,
        question: str,
        facts: list[str],
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
                    f'Hop {i}: "{hop["q"]}" → "{hop["a"]}" (from: "{hop["evidence"][0][:120]}...")'
                    if hop["evidence"]
                    else f'Hop {i}: "{hop["q"]}" → "{hop["a"]}"'
                )
            history = "Previous reasoning:\n" + "\n".join(lines) + "\n\n"

        try:
            response = self._llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the question using ONLY the provided facts. "
                            "Use the previous reasoning chain for context. "
                            "Give just the entity name, nothing else."
                        ),
                    },
                    {"role": "user", "content": (f"{history}Facts:\n{context}\n\nQuestion: {question}\nAnswer:")},
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
            if getattr(self, "_triadic", None) is not None and hop_texts:
                triadic_texts = self._triadic.expand(
                    hop_texts,
                    top_k=self._triadic_expand_k,
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
                evidence_chain.append(
                    {
                        "query": current_query,
                        "found": hop_new_chunks[:3],
                        "entity": None,
                    }
                )
                break

            # Build chain summary for the judge so it has full hop context
            chain_summary = ""
            if evidence_chain:
                lines = []
                for i, step in enumerate(evidence_chain, 1):
                    found_preview = step["found"][0][:100] if step["found"] else "nothing relevant"
                    learned_str = f" → learned: {step['learned']}" if step.get("learned") else ""
                    entity_str = f' → next search: "{step["entity"]}"' if step.get("entity") else ""
                    lines.append(
                        f'  Hop {i}: searched "{step["query"]}" → found "{found_preview}..."{learned_str}{entity_str}'
                    )
                chain_summary = "Reasoning chain so far:\n" + "\n".join(lines) + "\n\n"

            # LLM judge sees ALL accumulated chunks (not capped at retrieve_num).
            # Each hop adds new deduped chunks; the judge needs the full
            # evidence to decide if the chain is complete.
            context_for_judge = "\n".join(all_chunks)
            try:
                judge_response = self._llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
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
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {bare_question}\n\n"
                                f"{chain_summary}"
                                f"Retrieved facts:\n{context_for_judge}\n\n"
                                f"Can you answer? If not, what did you learn "
                                f"and what concrete fact is still missing?"
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=64,
                )
                judge_text = (judge_response.choices[0].message.content or "").strip()
            except Exception:
                evidence_chain.append(
                    {
                        "query": current_query,
                        "found": hop_new_chunks[:3],
                        "entity": None,
                    }
                )
                break  # LLM failure → use what we have

            if judge_text.startswith("ANSWER_READY"):
                evidence_chain.append(
                    {
                        "query": current_query,
                        "found": hop_new_chunks[:3],
                        "entity": None,
                    }
                )
                break

            # Parse LEARNED/NEED format
            learned = ""
            next_entity = ""
            for line in judge_text.split("\n"):
                line = line.strip()
                if line.startswith("LEARNED:"):
                    learned = line[len("LEARNED:") :].strip()
                elif line.startswith("NEED:"):
                    next_entity = line[len("NEED:") :].strip()

            # Fallback: old format (just "NEED: ...")
            if not next_entity and judge_text.startswith("NEED:"):
                next_entity = judge_text[len("NEED:") :].strip()

            if next_entity:
                evidence_chain.append(
                    {
                        "query": current_query,
                        "found": hop_new_chunks[:3],
                        "entity": next_entity,
                        "learned": learned,
                    }
                )
                current_query = next_entity
            else:
                # Unparseable response → treat as ready
                evidence_chain.append(
                    {
                        "query": current_query,
                        "found": hop_new_chunks[:3],
                        "entity": None,
                    }
                )
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
            all_chunks = [c for c in all_chunks if not self._belief_index.is_excluded(c)]
        elif self.supersession and self._superseded_texts:
            all_chunks = [c for c in all_chunks if c not in self._superseded_texts]

        # Trim to retrieve_num and format via outgestion
        final_chunks = all_chunks[: self.retrieve_num]
        memory_context, outgestion_instructions = self._format_outgestion(
            final_chunks,
            metadata_by_key,
        )
        memory_time = time.time() - query_start

        # Generate answer
        answer, input_tokens, output_tokens, gen_time = self._generate_answer(
            question,
            memory_context,
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

    def _embed_query(self, query_text: str):
        """Embed a query string.

        Returns:
            numpy array of shape (dim,) — the query embedding.
        """
        return self._scorer.embed(query_text)

    def _retrieve_candidates(
        self,
        q_emb,
        top_k: int,
        query_text: str | None = None,
    ) -> list[dict]:
        """Run the appropriate retrieval backend based on config flags.

        This is the big switch statement extracted from _retrieve_v2.
        Computes effective_k internally (CR overfetch, cosine dedup,
        belief overfetch).  The *top_k* parameter is the BASE
        retrieve_num before overfetch multipliers.

        Args:
            q_emb: Query embedding (numpy array).
            top_k: Base number of results to retrieve (before overfetch).
            query_text: Original query string, needed by the cooc+bm25
                backend.  May be ``None`` for backends that don't use it.

        Returns:
            List of result dicts with at minimum {"text": str, "score": float}.
        """
        # Over-fetch when CR, cosine dedup, or belief is active to give
        # filtering more candidates. Conflicting facts have cosine ~0.98-0.99,
        # so top-10 drops correct versions.
        effective_k = top_k
        if self.conflict_resolution and self._cr_overfetch > 1:
            effective_k = top_k * self._cr_overfetch
        if self.cosine_dedup:
            effective_k = max(effective_k, top_k * self._cr_overfetch)
        if self._belief_index is not None:
            effective_k = max(effective_k, top_k * self._belief_overfetch)

        if self.transfer_retrieval:
            results = self.coupled_engine.query_transfer(
                embedding=q_emb,
                top_k=effective_k,
                transfer_k=self.transfer_k,
            )
        elif self.coretrieval_retrieval:
            results = self.coupled_engine.query_coretrieval(
                embedding=q_emb,
                top_k=effective_k,
                coretrieval_bonus=self.coretrieval_bonus,
                min_coretrieval_count=self.coretrieval_min_count,
            )
        elif self.cooc_boost_retrieval and self.ppr_retrieval:
            results = self.coupled_engine.query_cooc_ppr(
                embedding=q_emb,
                top_k=effective_k,
                cooc_weight=self.cooc_weight,
                gate_threshold=self.cooc_gate_threshold,
                ppr_weight=self.ppr_weight,
                damping=self.ppr_damping,
            )
        elif self.cooc_boost_retrieval and self.bm25_weight > 0:
            results = self.coupled_engine.query_cooc_bm25(
                embedding=q_emb,
                query_text=query_text,
                top_k=effective_k,
                cooc_weight=self.cooc_weight,
                bm25_weight=self.bm25_weight,
                gate_threshold=self.cooc_gate_threshold,
            )
        elif self.cooc_boost_retrieval:
            results = self.coupled_engine.query_cooc_boost(
                embedding=q_emb,
                top_k=effective_k,
                cooc_weight=self.cooc_weight,
                dream_weight=self.dream_weight,
                gate_threshold=self.cooc_gate_threshold,
            )
        elif self.ppr_retrieval:
            results = self.coupled_engine.query_ppr(
                embedding=q_emb,
                top_k=effective_k,
                ppr_weight=self.ppr_weight,
                damping=self.ppr_damping,
            )
        elif self.hybrid_retrieval:
            results = self.coupled_engine.query_hybrid(
                embedding=q_emb,
                top_k=effective_k,
            )
        elif self.associative_retrieval or self.sparse_retrieval:
            results = self.coupled_engine.query_associative(
                embedding=q_emb,
                top_k=effective_k,
                sparse=self.sparse_retrieval,
            )
        else:
            results = self.coupled_engine.query(
                embedding=q_emb,
                top_k=effective_k,
            )

        return results

    def _apply_post_retrieval(self, results: list[dict]) -> list[dict]:
        """Apply post-retrieval reranking and filtering.

        Currently: belief reranking when active, truncation to retrieve_num.

        Args:
            results: Raw retrieval results.

        Returns:
            Reranked and truncated results.
        """
        if self._belief_index is not None:
            results = self._apply_belief_rerank(results)[: self.retrieve_num]
        return results

    def _retrieve_v2(self, query_text: str) -> list[dict]:
        """Run V2 embedding retrieval for a single query string.

        Composed from three independent stages:
          1. _embed_query      -- vectorize the query text
          2. _retrieve_candidates -- run the backend retrieval switch
          3. _apply_post_retrieval -- belief rerank + truncation
        """
        q_emb = self._embed_query(query_text)
        results = self._retrieve_candidates(
            q_emb, self.retrieve_num, query_text=query_text,
        )
        return self._apply_post_retrieval(results)

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
        self,
        bare_question: str,
        v2_texts: list[str],
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

    def _query_time_cr(self, chunks: list[str]) -> list[str]:
        """Query-time conflict resolution on retrieved facts.

        Instead of building a global supersession graph from all 18k facts
        at ingestion (unsolvable edge-count problem), extract triples from
        only the 10-20 retrieved facts. Conflicting facts share sentence
        templates → spaCy consistency is high locally → no tuning needed.

        Algorithm:
          1. Parse each chunk with spaCy (via ingestion pipeline's nlp)
          2. Extract triples (acl-aware copular + basic verb patterns)
          3. Group by (subject_lower, predicate_lower)
          4. For groups with different objects, keep only the newest chunk
        """
        if len(chunks) <= 1:
            return chunks

        nlp = self._ingestion._nlp

        # Parse all chunks in one pass
        docs = list(nlp.pipe(chunks))

        # Build text→recency lookup from memory store
        text_to_recency: dict[str, float] = {}
        for m in self.coupled_engine.memory_store:
            text_to_recency[m.text] = m.recency

        # Extract triples and NER for each chunk
        chunk_triples: list[list[tuple[str, str, str]]] = []
        chunk_ents: list[set[str]] = []
        for doc in docs:
            chunk_triples.append(_extract_acl_triples(doc))
            chunk_ents.append({e.text.lower() for e in doc.ents if len(e.text) > 2})

        # Group by (subject, predicate) → list of (chunk_idx, object, recency)
        sp_groups: dict[tuple[str, str], list[tuple[int, str, float]]] = defaultdict(list)
        for idx, triples in enumerate(chunk_triples):
            recency = text_to_recency.get(chunks[idx], 0.0)
            for subj, pred, obj in triples:
                key = (subj.lower().strip(), pred.lower().strip())
                if key[0] and key[1]:
                    sp_groups[key].append((idx, obj.lower().strip(), recency))

        # Resolve conflicts: for each SP group with different objects,
        # keep only the newest chunk — gated by predicate cardinality.
        to_remove: set[int] = set()
        for (subj, pred), members in sp_groups.items():
            objects = set(obj for _, obj, _ in members)
            if len(objects) <= 1:
                continue  # same object across all members → no conflict

            # Tier 1 cardinality gate: skip additive predicates
            cardinality = _classify_predicate_cardinality(pred, subj)
            if cardinality == 'additive':
                _cr_log.debug(
                    "query_time_cr: SKIP additive (%s, %s) objs=%s",
                    subj[:30], pred, objects,
                )
                continue

            # Sort by recency descending (newest first)
            members.sort(key=lambda x: x[2], reverse=True)
            winner_idx, winner_obj, _ = members[0]

            for idx, obj, _ in members[1:]:
                if obj != winner_obj:
                    # Entity gate: require NER overlap when enabled
                    if self._cr_entity_gate:
                        if not (chunk_ents[winner_idx] & chunk_ents[idx]):
                            continue
                    to_remove.add(idx)

        if to_remove:
            _cr_log.info(
                "query_time_cr: removing %d/%d chunks",
                len(to_remove), len(chunks),
            )
            for idx in sorted(to_remove):
                _cr_log.info(
                    "  REMOVED [rec=%.0f]: %s",
                    text_to_recency.get(chunks[idx], 0),
                    chunks[idx][:80],
                )
            return [c for i, c in enumerate(chunks) if i not in to_remove]
        return chunks

    def _pairwise_cosine_dedup(self, chunks: list[str]) -> list[str]:
        """Pairwise cosine dedup: for near-duplicate retrieved facts, keep newest.

        For every pair of retrieved facts with cosine > threshold to each other,
        remove the older one (lower recency). This catches conflicting versions
        of the same fact without requiring triple extraction or NLP parsing.

        Rationale: conflicting facts share sentence templates (same S,P different O)
        and have cosine > 0.95. Global recency sort destroys relevance ranking,
        but pairwise dedup only touches true near-duplicates.
        """
        if len(chunks) <= 1:
            return chunks

        # Build text→(embedding, recency) from memory store
        text_to_emb: dict[str, np.ndarray] = {}
        text_to_rec: dict[str, float] = {}
        for m in self.coupled_engine.memory_store:
            text_to_emb[m.text] = m.embedding
            text_to_rec[m.text] = m.recency

        # Collect embeddings for retrieved chunks
        embs = []
        recs = []
        valid_idx = []  # chunk indices that have embeddings
        for i, chunk in enumerate(chunks):
            emb = text_to_emb.get(chunk)
            if emb is not None:
                embs.append(emb)
                recs.append(text_to_rec.get(chunk, 0.0))
                valid_idx.append(i)

        if len(embs) < 2:
            return chunks

        # Pairwise cosine similarity matrix
        E = np.array(embs)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E_norm = E / norms
        sim_matrix = E_norm @ E_norm.T

        # For each pair above threshold, mark the older one for removal
        to_remove: set[int] = set()
        threshold = self._cosine_dedup_threshold
        n = len(valid_idx)
        for i in range(n):
            if valid_idx[i] in to_remove:
                continue
            for j in range(i + 1, n):
                if valid_idx[j] in to_remove:
                    continue
                if sim_matrix[i, j] >= threshold:
                    ci, cj = valid_idx[i], valid_idx[j]
                    if recs[i] >= recs[j]:
                        to_remove.add(cj)
                        _cr_log.debug(
                            "cosine_dedup: cos=%.4f, keep[rec=%.0f] drop[rec=%.0f]",
                            sim_matrix[i, j], recs[i], recs[j],
                        )
                    else:
                        to_remove.add(ci)
                        _cr_log.debug(
                            "cosine_dedup: cos=%.4f, keep[rec=%.0f] drop[rec=%.0f]",
                            sim_matrix[i, j], recs[j], recs[i],
                        )

        if to_remove:
            _cr_log.info(
                "cosine_dedup: removing %d/%d chunks (threshold=%.3f)",
                len(to_remove), len(chunks), threshold,
            )
            for idx in sorted(to_remove):
                _cr_log.info(
                    "  DEDUP [rec=%.0f, cos=max]: %s",
                    text_to_rec.get(chunks[idx], 0),
                    chunks[idx][:80],
                )
            return [c for i, c in enumerate(chunks) if i not in to_remove]
        return chunks

    def _resolve_conflicts(
        self,
        chunks: list[str],
        metadata_by_key: dict[str, dict],
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
            _cr_log.info("_resolve_conflicts 2a: removing %d chunks: %s", len(to_remove), [t[:60] for t in to_remove])

        # 2b: DISABLED — embedding similarity + entity gating produces false
        # positives. Same-template, different-entity sentences have cosine > 0.85
        # and share template words (e.g. "association football"), causing correct
        # facts to be removed. Eval showed 9 false positive removals, SubEM drop.

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
        # Layer 3 (newest-first sort) DISABLED — eval showed it disrupts the
        # relevance ranking built by cosine+cooc+triadic, causing ~5 regressions.
        # Just return flat text preserving retrieval order.
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
        self,
        question: str,
        context: str,
        extra_system_instructions: str = "",
    ) -> tuple[str, int, int, float]:
        """Call LLM to generate an answer from retrieved context."""
        self._ensure_llm_client()

        system_msg = (
            "You are a helpful assistant that answers questions from a knowledge pool. Give a very concise answer."
        )
        if extra_system_instructions:
            system_msg = system_msg + " " + extra_system_instructions
        # The MABench query already contains full instructions about serial
        # numbers and recency. Prepend retrieved memories as the knowledge pool.
        user_msg = f"[Knowledge Pool]\n{context}\n\n{question}\nAnswer:"

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

        # Reset BM25 index and chain cache
        if getattr(self, "_bm25_index", None) is not None:
            self._bm25_index = BM25Index()
        if getattr(self, "_chain_cache", None) is not None:
            self._chain_cache = ChainCache(similarity_threshold=0.95)

        # Reset triadic memory (fresh tensor, codebook, mappings)
        if getattr(self, "_triadic", None) is not None:
            self._triadic = TriadicMemory(
                n=self._triadic.n,
                p=self._triadic.p,
            )

        # Reset entity accumulator
        if hasattr(self, "_entity_accumulator"):
            self._entity_accumulator = _MutableEntityGraph()

        # Reset supersession state
        self._triple_index = defaultdict(list)
        self._superseded_texts = set()
        self._fact_entities = {}
        self._supersession_graph = {}
        self._supersedes = defaultdict(set)

        # Reset GLiNER entity index
        self._gliner_entity_index = defaultdict(set)

        # Reset REBEL KG
        if getattr(self, "_rebel_kg", None) is not None:
            self._rebel_kg = {}
            self._rebel_kg_reverse = {}

        # Reset belief index (preserve params, clear beliefs)
        if self._belief_index is not None:
            self._belief_index = BeliefIndex(
                prior_alpha=self._belief_index._prior_alpha,
                prior_beta=self._belief_index._prior_beta,
                propagation_damping=self._belief_index._propagation_damping,
                hard_floor=self._belief_index._hard_floor,
            )
