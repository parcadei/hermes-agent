"""Conflict resolution, deduplication, and coverage auditing.

Module-level functions:
  _get_subtree_text  — spaCy token subtree text helper
  _extract_acl_triples — ACL-aware triple extraction from spaCy doc

Mixin class:
  ConflictResolutionMixin — query-time CR, cosine dedup, graph-based
  conflict resolution, and coverage audit methods for HermesMemoryAgent.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from nlcdm_core import cosine_sim  # noqa: F401 — available for subclasses

# ---------------------------------------------------------------------------
# Query-time conflict resolution helpers
# ---------------------------------------------------------------------------
_cr_log = logging.getLogger("conflict_res")


def _get_subtree_text(token) -> str:
    """Get the full text of a token's subtree (e.g., 'association football')."""
    return " ".join(t.text for t in token.subtree)


def _extract_acl_triples(doc) -> list[tuple[str, str, str]]:
    """Extract (subject, predicate, object) triples using acl-aware copular parsing.

    Handles two sentence patterns:

    1. Copular with acl modifier (most common in MABench):
       "The sport played by Abbiati is association football"
       Parse tree: ROOT=is(AUX), nsubj=sport->acl=played->agent=by->pobj=Abbiati, attr=football
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
                # Descend acl->agent/prep->pobj for real subject
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
                    # acl without agent/prep -> fallback to nsubj as subject
                    triples.append((_get_subtree_text(nsubj), nsubj.text, obj_text))
            else:
                # Simple copular: "Abbiati is Italian"
                triples.append((_get_subtree_text(nsubj), "be", obj_text))

        elif root.pos_ == "VERB":
            # Non-copular: nsubj/nsubjpass->VERB->dobj/attr/pobj/prep->pobj
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
            # Descend prep->pobj->prep->pobj to extract the deepest pobj.
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


# ---------------------------------------------------------------------------
# ConflictResolutionMixin
# ---------------------------------------------------------------------------


class ConflictResolutionMixin:
    """Mixin for query-time conflict resolution and deduplication.

    Provides:
      _query_time_cr        — triple-based conflict resolution on retrieved facts
      _pairwise_cosine_dedup — cosine-based near-duplicate removal
      _resolve_conflicts    — supersession-graph-based conflict resolution
      _coverage_audit       — decomposition-based coverage gap detection

    Attributes accessed via self (defined on HermesMemoryAgent):
      coupled_engine, _ingestion, _scorer, _cr_entity_gate,
      _cosine_dedup_threshold, _supersession_graph, _supersedes,
      _decompose_query, _retrieve_v2
    """

    def _query_time_cr(self, chunks: list[str]) -> list[str]:
        """Query-time conflict resolution on retrieved facts.

        Instead of building a global supersession graph from all 18k facts
        at ingestion (unsolvable edge-count problem), extract triples from
        only the 10-20 retrieved facts. Conflicting facts share sentence
        templates -> spaCy consistency is high locally -> no tuning needed.

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

        # Build text->recency lookup from memory store
        text_to_recency: dict[str, float] = {}
        for m in self.coupled_engine.memory_store:
            text_to_recency[m.text] = m.recency

        # Extract triples and NER for each chunk
        chunk_triples: list[list[tuple[str, str, str]]] = []
        chunk_ents: list[set[str]] = []
        for doc in docs:
            chunk_triples.append(_extract_acl_triples(doc))
            chunk_ents.append({e.text.lower() for e in doc.ents if len(e.text) > 2})

        # Group by (subject, predicate) -> list of (chunk_idx, object, recency)
        sp_groups: dict[tuple[str, str], list[tuple[int, str, float]]] = defaultdict(list)
        for idx, triples in enumerate(chunk_triples):
            recency = text_to_recency.get(chunks[idx], 0.0)
            for subj, pred, obj in triples:
                key = (subj.lower().strip(), pred.lower().strip())
                if key[0] and key[1]:
                    sp_groups[key].append((idx, obj.lower().strip(), recency))

        # Resolve conflicts: for each SP group with different objects,
        # keep only the newest chunk (newest wins by recency).
        to_remove: set[int] = set()
        for (subj, pred), members in sp_groups.items():
            objects = set(obj for _, obj, _ in members)
            if len(objects) <= 1:
                continue  # same object across all members -> no conflict

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

        # Build text->(embedding, recency) from memory store
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

        # 2a: Triple-based -- use supersession graph
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

        # 2b: DISABLED -- embedding similarity + entity gating produces false
        # positives. Same-template, different-entity sentences have cosine > 0.85
        # and share template words (e.g. "association football"), causing correct
        # facts to be removed. Eval showed 9 false positive removals, SubEM drop.

        if to_remove:
            return [c for c in chunks if c not in to_remove]
        return chunks

    def _coverage_audit(
        self,
        bare_question: str,
        v2_texts: list[str],
    ) -> list[str]:
        """Use decomposition as a coverage check, not a retrieval strategy.

        1. Decompose the question into sub-queries.
        2. Embed each sub-query and each already-retrieved chunk.
        3. For each sub-query, check if any retrieved chunk has
           cosine > threshold -- if not, do a targeted retrieval.
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

            # Gap found -- targeted retrieval for this sub-query
            sq_results = self._retrieve_v2(sq)
            for r in sq_results[:3]:  # inject at most 3 gap-fillers
                key = r["text"].strip()[:200]
                if key not in seen:
                    seen.add(key)
                    gap_texts.append(r["text"])

        return gap_texts
