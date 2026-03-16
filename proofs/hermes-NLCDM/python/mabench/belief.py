"""Bayesian belief system for fact currency scoring.

Two-layer system replacing binary supersession with soft belief scoring:

  Layer 1 — Beta-Bernoulli per fact:
    Each fact carries a Beta(α, β) belief.  P(current) = α / (α + β).
    On conflict detection (same S+P, different O, shared entity),
    the old fact gets β += 1  (contradicting evidence) and
    the new fact gets α += 1  (confirming evidence).

  Layer 2 — Belief propagation via entity graph:
    When a fact drops below P(current) < 0.5, facts connected to it
    through shared entities are penalized:
      connected.β += damping × (1 − P_degraded)
    Single pass, no iteration — prevents runaway propagation.

Retrieval integration:
  - Score multiplication: result.score *= P(current)
  - Hard floor exclusion: P(current) < hard_floor → excluded entirely
  - Over-fetch: retrieve more candidates to compensate for exclusions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FactBelief:
    """Beta-Bernoulli belief state for a single fact.

    alpha: confirming evidence count (pseudo-count).
    beta:  contradicting evidence count (pseudo-count).
    """
    alpha: float
    beta: float

    @property
    def p_current(self) -> float:
        """Posterior probability that this fact is current."""
        return self.alpha / (self.alpha + self.beta)


class BeliefIndex:
    """Index of fact beliefs with two-layer scoring.

    Args:
        prior_alpha: Initial α for new facts (confirming prior).
        prior_beta:  Initial β for new facts (contradicting prior).
        propagation_damping: Controls penalty magnitude in Layer 2.
            Connected facts receive β += damping × (1 − P_degraded).
        hard_floor: Facts with P(current) below this threshold are
            excluded entirely from retrieval results.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 1.0,
        propagation_damping: float = 0.3,
        hard_floor: float = 0.05,
    ) -> None:
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._propagation_damping = propagation_damping
        self._hard_floor = hard_floor
        self._beliefs: dict[str, FactBelief] = {}

    def register_fact(self, text: str) -> None:
        """Register a new fact with the default prior Beta(α, β).

        If the fact is already registered, this is a no-op.
        """
        if text not in self._beliefs:
            self._beliefs[text] = FactBelief(
                alpha=self._prior_alpha,
                beta=self._prior_beta,
            )

    def on_conflict(self, old_text: str, new_text: str) -> None:
        """Update beliefs when a conflict is detected.

        The old fact receives contradicting evidence (β += 1).
        The new fact receives confirming evidence (α += 1).
        Both facts are registered if not already present.
        """
        if old_text not in self._beliefs:
            self.register_fact(old_text)
        if new_text not in self._beliefs:
            self.register_fact(new_text)

        self._beliefs[old_text].beta += 1.0
        self._beliefs[new_text].alpha += 1.0

    def propagate(self, entity_accumulator: Any, memory_store: list) -> None:
        """Layer 2: propagate belief penalties through the entity graph.

        For each fact with P(current) < 0.5, find facts connected through
        shared entities and penalize them:
            connected.β += damping × (1 − P_degraded)

        Single pass — no iteration, no cycles.

        Args:
            entity_accumulator: _MutableEntityGraph with entity_to_facts
                and fact_to_entities mappings (int indices → entity names).
            memory_store: List of memory entries where memory_store[idx]
                has a .text attribute giving the fact text for that index.
        """
        if not self._beliefs:
            return

        # Build reverse lookup: fact_text → set of fact indices
        text_to_indices: dict[str, set[int]] = {}
        for idx, entry in enumerate(memory_store):
            text = entry.text if hasattr(entry, 'text') else str(entry)
            text_to_indices.setdefault(text, set()).add(idx)

        # Find degraded facts (P < 0.5)
        degraded_texts: list[tuple[str, float]] = []
        for text, belief in self._beliefs.items():
            if belief.p_current < 0.5:
                degraded_texts.append((text, belief.p_current))

        if not degraded_texts:
            return

        # For each degraded fact, find connected facts via entity graph
        # and penalize them
        fact_to_entities = getattr(entity_accumulator, 'fact_to_entities', {})
        entity_to_facts = getattr(entity_accumulator, 'entity_to_facts', {})

        for degraded_text, p_degraded in degraded_texts:
            # Find indices for this degraded fact
            indices = text_to_indices.get(degraded_text, set())
            if not indices:
                continue

            # Collect all entities linked to this fact
            connected_entities: set[str] = set()
            for idx in indices:
                entities = fact_to_entities.get(idx, set())
                connected_entities.update(entities)

            # Find all connected fact indices through those entities
            connected_indices: set[int] = set()
            for entity in connected_entities:
                facts_for_entity = entity_to_facts.get(entity, set())
                connected_indices.update(facts_for_entity)

            # Remove the degraded fact's own indices
            connected_indices -= indices

            # Penalize connected facts
            penalty = self._propagation_damping * (1.0 - p_degraded)
            for conn_idx in connected_indices:
                if conn_idx < len(memory_store):
                    conn_entry = memory_store[conn_idx]
                    conn_text = (
                        conn_entry.text
                        if hasattr(conn_entry, 'text')
                        else str(conn_entry)
                    )
                    if conn_text in self._beliefs:
                        self._beliefs[conn_text].beta += penalty

    def score(self, text: str) -> float:
        """Return P(current) for a fact. Unknown facts score 1.0."""
        belief = self._beliefs.get(text)
        if belief is None:
            return 1.0
        return belief.p_current

    def is_excluded(self, text: str) -> bool:
        """Return True if the fact should be excluded (P < hard_floor)."""
        belief = self._beliefs.get(text)
        if belief is None:
            return False
        return belief.p_current < self._hard_floor

    def state_dict(self) -> dict:
        """Serialize belief state for persistence."""
        return {
            "prior_alpha": self._prior_alpha,
            "prior_beta": self._prior_beta,
            "propagation_damping": self._propagation_damping,
            "hard_floor": self._hard_floor,
            "beliefs": {
                text: {"alpha": b.alpha, "beta": b.beta}
                for text, b in self._beliefs.items()
            },
        }

    @classmethod
    def from_state_dict(cls, data: dict) -> BeliefIndex:
        """Reconstruct a BeliefIndex from serialized state."""
        index = cls(
            prior_alpha=data["prior_alpha"],
            prior_beta=data["prior_beta"],
            propagation_damping=data["propagation_damping"],
            hard_floor=data["hard_floor"],
        )
        for text, bdata in data.get("beliefs", {}).items():
            index._beliefs[text] = FactBelief(
                alpha=bdata["alpha"],
                beta=bdata["beta"],
            )
        return index
