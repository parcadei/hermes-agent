from .core import (
    retention, strength_update, strength_iter, sigmoid,
    ScoringWeights, score, clamp01, importance_update,
    strength_decay, combined_factor, steady_state_strength,
    soft_select,
    novelty_bonus, exploration_window, boosted_score,
    expected_strength_update, composed_expected_map, composed_contraction_factor,
    composed_domain_contains,
)
from .markov_chain import (
    simulate_chain, simulate_coupling, build_transition_matrix, spectral_analysis,
)
