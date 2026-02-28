"""Hermes Memory System -- Adaptive Recall Pipeline.

Implements the recall gate, adaptive-k selection, tiered rendering, and
budget-constrained assembly pipeline. This module transforms a ranked list
of memories (from engine.py) into a token-budgeted context string suitable
for injection into an LLM prompt.

Pipeline stages:
  1. should_recall   -- Gate trivial/short messages
  2. adaptive_k      -- Score-gap detection for dynamic k selection
  3. assign_tiers    -- Normalized-score tier assignment (HIGH/MEDIUM/LOW)
  4. format_memory   -- Tier-specific rendering
  5. budget_constrain -- Demotion-then-drop cascade under token budget
  6. recall          -- Main orchestrator

Dependencies: stdlib only (dataclasses, enum, math, re).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hermes_memory.engine import MemoryState, ParameterSet


# ============================================================
# 1. Constants
# ============================================================

CHARS_PER_TOKEN: int = 4
"""Default character-to-token ratio for English text."""

CHARS_PER_TOKEN_CONSERVATIVE: int = 2
"""Conservative character-to-token ratio for CJK/code-heavy content."""

MIN_DEMOTION_SAVINGS_TOKENS: int = 5
"""Minimum token savings required to justify a demotion (AP2-F4/AP3-T3)."""

MEMORY_SEPARATOR: str = "\n---\n"
"""Separator between formatted memory blocks in the assembled context."""


# ============================================================
# 2. Tier Enum
# ============================================================


class Tier(IntEnum):
    """Memory rendering tier. Lower value = higher detail."""

    HIGH = 1
    MEDIUM = 2
    LOW = 3


# ============================================================
# 3. RecallConfig -- Frozen configuration dataclass
# ============================================================


@dataclass(frozen=True)
class RecallConfig:
    """Configuration for the adaptive recall pipeline.

    All fields carry domain constraints validated at construction time.
    Frozen to prevent accidental mutation.

    Attributes:
        min_k:               Minimum memories to return. Domain: >= 1.
        max_k:               Maximum memories to return. Domain: >= min_k.
        gap_buffer:          Extra memories past the gap position. Domain: >= 0.
        epsilon:             Minimum gap to avoid fallback. Domain: > 0.
        high_threshold:      Normalized score threshold for HIGH tier. Domain: (0, 1].
        mid_threshold:       Normalized score threshold for MEDIUM tier. Domain: [0, high_threshold).
        total_budget:        Total token budget for assembled context. Domain: >= 1.
        high_budget_share:   Fraction of budget for HIGH-tier memories. Domain: (0, 1).
        medium_budget_share: Fraction of budget for MEDIUM-tier memories. Domain: (0, 1).
        low_budget_share:    Fraction of budget for LOW-tier memories. Domain: (0, 1).
        high_max_chars:      Max chars for HIGH-tier formatted memory. Domain: >= 1.
        medium_max_chars:    Max chars for MEDIUM-tier formatted memory. Domain: >= 1.
        low_max_chars:       Max chars for LOW-tier formatted memory. Domain: >= 1.
        gate_min_length:     Minimum message length to pass gate. Domain: >= 0.
        gate_trivial_patterns: Tuple of trivial patterns for gate matching.
        chars_per_token:     Character-to-token ratio for token estimation. Domain: >= 1.
    """

    min_k: int = 1
    max_k: int = 10
    gap_buffer: int = 2
    epsilon: float = 0.01
    high_threshold: float = 0.7
    mid_threshold: float = 0.4
    total_budget: int = 800
    high_budget_share: float = 0.55
    medium_budget_share: float = 0.30
    low_budget_share: float = 0.15
    high_max_chars: int = 400
    medium_max_chars: int = 200
    low_max_chars: int = 80
    gate_min_length: int = 8
    gate_trivial_patterns: tuple[str, ...] = (
        "ok",
        "yes",
        "no",
        "sure",
        "thanks",
        "thank you",
        "hi",
        "hello",
        "hey",
        "bye",
        "goodbye",
        "good",
        "great",
        "fine",
        "cool",
        "nice",
        "right",
        "yep",
        "nope",
        "yeah",
    )
    chars_per_token: int = CHARS_PER_TOKEN

    def __post_init__(self) -> None:
        """Validate all field constraints."""
        if self.min_k < 1:
            raise ValueError(f"min_k must be >= 1, got {self.min_k}")
        if self.max_k < self.min_k:
            raise ValueError(f"max_k must be >= min_k={self.min_k}, got {self.max_k}")
        if self.gap_buffer < 0:
            raise ValueError(f"gap_buffer must be >= 0, got {self.gap_buffer}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.high_threshold <= self.mid_threshold:
            raise ValueError(
                f"high_threshold must be > mid_threshold={self.mid_threshold}, "
                f"got {self.high_threshold}"
            )
        if self.mid_threshold < 0.0:
            raise ValueError(f"mid_threshold must be >= 0.0, got {self.mid_threshold}")
        if self.high_threshold > 1.0:
            raise ValueError(
                f"high_threshold must be <= 1.0, got {self.high_threshold}"
            )
        if self.total_budget < 1:
            raise ValueError(f"total_budget must be >= 1, got {self.total_budget}")
        if self.high_budget_share <= 0:
            raise ValueError(
                f"high_budget_share must be > 0, got {self.high_budget_share}"
            )
        if self.medium_budget_share <= 0:
            raise ValueError(
                f"medium_budget_share must be > 0, got {self.medium_budget_share}"
            )
        if self.low_budget_share <= 0:
            raise ValueError(
                f"low_budget_share must be > 0, got {self.low_budget_share}"
            )
        share_sum = (
            self.high_budget_share + self.medium_budget_share + self.low_budget_share
        )
        if abs(share_sum - 1.0) >= 1e-10:
            raise ValueError(f"budget shares must sum to 1.0, got {share_sum}")
        if self.high_max_chars < 1:
            raise ValueError(f"high_max_chars must be >= 1, got {self.high_max_chars}")
        if self.medium_max_chars < 1:
            raise ValueError(
                f"medium_max_chars must be >= 1, got {self.medium_max_chars}"
            )
        if self.low_max_chars < 1:
            raise ValueError(f"low_max_chars must be >= 1, got {self.low_max_chars}")
        if self.gate_min_length < 0:
            raise ValueError(
                f"gate_min_length must be >= 0, got {self.gate_min_length}"
            )
        if self.chars_per_token < 1:
            raise ValueError(
                f"chars_per_token must be >= 1, got {self.chars_per_token}"
            )


DEFAULT_RECALL_CONFIG: RecallConfig = RecallConfig()
"""Module-level default configuration instance."""


# ============================================================
# 4. TierAssignment -- Frozen tier assignment result
# ============================================================


@dataclass(frozen=True)
class TierAssignment:
    """A single memory's tier assignment after scoring and normalization.

    Attributes:
        index:            Original index of the memory in the input list.
        score:            Raw score from the engine's rank_memories.
        normalized_score: Within-k normalized score in [0.0, 1.0].
        tier:             Assigned rendering tier.
    """

    index: int
    score: float
    normalized_score: float
    tier: Tier


# ============================================================
# 5. RecallResult -- Frozen pipeline output
# ============================================================


@dataclass(frozen=True)
class RecallResult:
    """Output of the recall pipeline.

    Attributes:
        context:                Assembled context string (joined formatted memories).
        gated:                  True if message was gated (trivial/short).
        k:                      Number of memories in the output.
        tier_assignments:       Tuple of TierAssignment for each memory in output.
        total_tokens_estimate:  Estimated tokens in the context string.
        budget_exceeded:        True if budget was exceeded (demotions/drops occurred).
        memories_dropped:       Number of memories dropped during budget constraint.
        memories_demoted:       Number of memories demoted during budget constraint.
        budget_overflow:        max(0, total_tokens_estimate - config.total_budget).
    """

    context: str
    gated: bool
    k: int
    tier_assignments: tuple[TierAssignment, ...]
    total_tokens_estimate: int
    budget_exceeded: bool
    memories_dropped: int
    memories_demoted: int
    budget_overflow: int


# ============================================================
# 6. should_recall -- Gating logic
# ============================================================


def should_recall(message: str, turn_number: int, config: RecallConfig) -> bool:
    """Determine whether to invoke recall for this message.

    Gating rules (evaluated in order):
      1. Type check: message must be str, turn_number must be int >= 0.
      2. First turn (turn_number == 0): always True.
      3. Question mark in message: True (overrides length and trivial checks).
      4. Empty/whitespace after strip: False.
      5. Trivial pattern match (case-insensitive, trailing punctuation stripped): False.
      6. Length check: len(stripped) < gate_min_length -> False.
      7. Otherwise: True.

    Args:
        message:     The user's message text.
        turn_number: Current conversation turn (0-indexed).
        config:      RecallConfig with gating parameters.

    Returns:
        True if recall should proceed, False if gated.

    Raises:
        TypeError:  If message is not a str.
        ValueError: If turn_number < 0.
    """
    if not isinstance(message, str):
        raise TypeError(f"message must be str, got {type(message).__name__}")
    if turn_number < 0:
        raise ValueError(f"turn_number must be >= 0, got {turn_number}")

    # First turn always recalls
    if turn_number == 0:
        return True

    # Question mark overrides length and trivial checks
    if "?" in message:
        return True

    stripped = message.strip()

    # Empty after strip
    if not stripped:
        return False

    # Length check
    if len(stripped) < config.gate_min_length:
        return False

    # Trivial pattern matching (case-insensitive, strip trailing punctuation)
    normalized = re.sub(r"[.!,;:]+$", "", stripped).lower()
    if normalized in config.gate_trivial_patterns:
        return False

    return True


# ============================================================
# 7. adaptive_k -- Score-gap based k selection
# ============================================================


def adaptive_k(scores: list[float], config: RecallConfig) -> int:
    """Select k using score-gap detection with top-slice truncation.

    Algorithm:
      1. Empty scores -> 0.
      2. Single score -> 1.
      3. Validate: scores must be descending, no NaN.
      4. Truncate to top slice: scores[:max_k + gap_buffer + 1] (AP2-F3).
      5. Compute gaps between consecutive scores.
      6. If max_gap < epsilon (strict <): fallback to min(min_k + gap_buffer, len, max_k).
      7. Otherwise: k_raw = argmax(gaps) + 1 + gap_buffer.
      8. Clamp to [min_k, min(max_k, len(scores))].

    Args:
        scores: Descending-sorted list of scores from rank_memories.
        config: RecallConfig with k-selection parameters.

    Returns:
        Selected k value.

    Raises:
        ValueError: If scores are not descending or contain NaN.
    """
    n = len(scores)
    if n == 0:
        return 0
    if n == 1:
        return 1

    # Validate descending and no NaN
    for i, s in enumerate(scores):
        if math.isnan(s):
            raise ValueError(f"NaN found in scores at index {i}")
    for i in range(n - 1):
        if scores[i] < scores[i + 1]:
            raise ValueError(
                f"Scores must be in descending order, but "
                f"scores[{i}]={scores[i]} < scores[{i + 1}]={scores[i + 1]}"
            )

    # Top-slice truncation (AP2-F3)
    truncation_limit = config.max_k + config.gap_buffer + 1
    effective = scores[:truncation_limit]

    # Compute gaps
    gaps = [effective[i] - effective[i + 1] for i in range(len(effective) - 1)]

    if not gaps:
        # Only one score in the effective slice
        return min(max(1, config.min_k), n)

    max_gap = max(gaps)

    # Check epsilon fallback (strict less-than)
    if max_gap < config.epsilon:
        # Fallback
        k_fallback = min(config.min_k + config.gap_buffer, n, config.max_k)
        return max(k_fallback, 1)

    # Gap detection: find first occurrence of max gap
    gap_pos = gaps.index(max_gap)
    k_raw = gap_pos + 1 + config.gap_buffer

    # Clamp to valid range
    k_clamped = min(k_raw, n, config.max_k)
    k_clamped = max(k_clamped, config.min_k)
    k_clamped = min(k_clamped, n)  # Never exceed list length

    return k_clamped


# ============================================================
# 8. assign_tiers -- Normalized-score tier assignment
# ============================================================


def assign_tiers(
    scores: list[tuple[int, float]],
    k: int,
    config: RecallConfig,
) -> list[TierAssignment]:
    """Assign rendering tiers to the top-k scored memories.

    Within-k normalization maps scores to [0, 1], then applies threshold
    boundaries (strictly above for tier qualification).

    Positional fallback (AP1-F3): when all k selected scores are identical
    AND k > 1, uses positional distribution: ceil(k/3) HIGH, ceil(k/3) MEDIUM,
    rest LOW.

    Args:
        scores: List of (index, score) tuples in descending score order.
        k:      Number of memories to assign tiers to.
        config: RecallConfig with tier threshold parameters.

    Returns:
        List of TierAssignment in same order as input (descending score).

    Raises:
        ValueError: If k < 0, k > len(scores), or scores not descending.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k > len(scores):
        raise ValueError(f"k must be <= len(scores)={len(scores)}, got {k}")
    if k == 0:
        return []

    # Validate descending order
    for i in range(len(scores) - 1):
        if scores[i][1] < scores[i + 1][1]:
            raise ValueError(
                f"Scores must be in descending order, but "
                f"scores[{i}]={scores[i][1]} < scores[{i + 1}]={scores[i + 1][1]}"
            )

    selected = scores[:k]
    s_max = selected[0][1]
    s_min = selected[-1][1]
    score_range = s_max - s_min

    # Check for all-identical scores (positional fallback)
    all_identical = k > 1 and score_range == 0.0

    if all_identical:
        # Positional fallback (AP1-F3)
        n_high = math.ceil(k / 3)
        n_medium = math.ceil(k / 3)
        # Rest is LOW
        result = []
        for pos, (idx, raw_score) in enumerate(selected):
            # All identical -> normalized to 1.0
            norm = 1.0
            if pos < n_high:
                tier = Tier.HIGH
            elif pos < n_high + n_medium:
                tier = Tier.MEDIUM
            else:
                tier = Tier.LOW
            result.append(
                TierAssignment(
                    index=idx,
                    score=raw_score,
                    normalized_score=norm,
                    tier=tier,
                )
            )
        return result

    # Normal normalization
    result = []
    for idx, raw_score in selected:
        if score_range == 0.0:
            # Single element (k==1)
            norm = 1.0
        else:
            norm = (raw_score - s_min) / score_range

        # Strictly above threshold for tier qualification (AP1-F7)
        # Use tolerance to handle floating-point boundary cases:
        # a score that is approximately equal to the threshold should NOT
        # qualify for the higher tier.
        if norm > config.high_threshold + 1e-9:
            tier = Tier.HIGH
        elif norm > config.mid_threshold + 1e-9:
            tier = Tier.MEDIUM
        else:
            tier = Tier.LOW

        result.append(
            TierAssignment(
                index=idx,
                score=raw_score,
                normalized_score=norm,
                tier=tier,
            )
        )

    return result


# ============================================================
# 9. format_memory -- Tier-specific rendering
# ============================================================


def format_memory(
    memory: "MemoryState",
    tier: Tier,
    config: RecallConfig,
    content: Optional[str] = None,
) -> str:
    """Format a single memory according to its tier.

    Three-tier rendering:
      - HIGH:   Full metadata + content (if provided). Truncated to high_max_chars.
      - MEDIUM: Metadata (relevance, importance) + content (if provided). No strength.
                Truncated to medium_max_chars.
      - LOW:    Compact metadata only (content ignored). Truncated to low_max_chars.

    Content newlines are replaced with spaces. Long content is truncated at
    word boundary with '...'.

    Args:
        memory:  MemoryState to format.
        tier:    Rendering tier.
        config:  RecallConfig with max_chars parameters.
        content: Optional memory content text.

    Returns:
        Formatted string for this memory.
    """
    age = memory.creation_time

    if tier == Tier.HIGH:
        max_chars = config.high_max_chars
        lines = [
            "[Memory]",
            f"  relevance={memory.relevance:.2f}, "
            f"strength={memory.strength:.1f}, "
            f"importance={memory.importance:.1f}",
            f"  Accessed {memory.access_count} times, age={age:.0f} steps",
        ]
        if content:
            cleaned = content.replace("\n", " ")
            lines.append(f"  {cleaned}")
        text = "\n".join(lines)
        return _truncate_to_limit(text, max_chars)

    elif tier == Tier.MEDIUM:
        max_chars = config.medium_max_chars
        lines = [
            "[Memory]",
            f"  relevance={memory.relevance:.2f}, importance={memory.importance:.1f}",
        ]
        if content:
            cleaned = content.replace("\n", " ")
            lines.append(f"  {cleaned}")
        text = "\n".join(lines)
        return _truncate_to_limit(text, max_chars)

    else:  # LOW
        max_chars = config.low_max_chars
        text = (
            f"[Memory] rel={memory.relevance:.2f}, "
            f"imp={memory.importance:.1f}, "
            f"age={age:.0f}"
        )
        return _truncate_to_limit(text, max_chars)


def _truncate_to_limit(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, cutting at word boundary with '...'."""
    if len(text) <= max_chars:
        return text
    # Truncate at word boundary
    truncated = text[: max_chars - 3]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."


# ============================================================
# 10. budget_constrain -- Demotion-then-drop cascade
# ============================================================


def budget_constrain(
    assignments: list[TierAssignment],
    formatted: list[str],
    memories: list["MemoryState"],
    config: RecallConfig,
    contents: Optional[list[Optional[str]]] = None,
) -> tuple[list[TierAssignment], list[str], int, int]:
    """Constrain formatted memories to fit within token budget.

    Strategy: per-tier demotion cascade, then drop from lowest-scored.

    Algorithm:
      1. Normalize contents=None to [None]*len (AP1-F4).
      2. Check total budget. If under, return unchanged.
      3. Per-tier demotion: HIGH->MEDIUM, then MEDIUM->LOW (cascade).
         Skip demotion if savings < MIN_DEMOTION_SAVINGS_TOKENS (AP3-T3).
      4. Drop lowest-scored memories until within budget or at min_k.

    Args:
        assignments: TierAssignment list (descending score order).
        formatted:   Formatted strings (parallel to assignments).
        memories:    MemoryState list (parallel to assignments).
        config:      RecallConfig with budget parameters.
        contents:    Optional content strings (parallel to assignments).

    Returns:
        (assignments, formatted, demoted_count, dropped_count)

    Raises:
        ValueError: If parallel list lengths don't match.
    """
    n = len(assignments)
    if n == 0:
        return ([], [], 0, 0)

    # Validate parallel lists
    if len(formatted) != n:
        raise ValueError(f"formatted length {len(formatted)} != assignments length {n}")
    if len(memories) != n:
        raise ValueError(f"memories length {len(memories)} != assignments length {n}")

    # Normalize contents (AP1-F4)
    if contents is None:
        contents = [None] * n
    elif len(contents) != n:
        raise ValueError(f"contents length {len(contents)} != assignments length {n}")

    # Work with mutable copies
    work_a = list(assignments)
    work_f = list(formatted)
    work_c = list(contents)
    demoted = 0
    dropped = 0

    def _estimate_tokens_total() -> int:
        return sum(len(s) // config.chars_per_token for s in work_f)

    def _is_over_budget() -> bool:
        return _estimate_tokens_total() > config.total_budget

    def _tier_tokens(tier: Tier) -> int:
        return sum(
            len(work_f[i]) // config.chars_per_token
            for i in range(len(work_a))
            if work_a[i].tier == tier
        )

    def _tier_budget(tier: Tier) -> float:
        share = {
            Tier.HIGH: config.high_budget_share,
            Tier.MEDIUM: config.medium_budget_share,
            Tier.LOW: config.low_budget_share,
        }[tier]
        return config.total_budget * share

    def _is_over_tier_budget(tier: Tier) -> bool:
        return _tier_tokens(tier) > _tier_budget(tier)

    def _needs_reduction() -> bool:
        """Check if any budget constraint is violated."""
        if _is_over_budget():
            return True
        for tier in (Tier.HIGH, Tier.MEDIUM, Tier.LOW):
            if _is_over_tier_budget(tier):
                return True
        return False

    # Per-tier demotion cascade: HIGH -> MEDIUM, then MEDIUM -> LOW
    for from_tier, to_tier in [
        (Tier.HIGH, Tier.MEDIUM),
        (Tier.MEDIUM, Tier.LOW),
    ]:
        if not _is_over_budget() and not _is_over_tier_budget(from_tier):
            continue

        # Iterate from lowest-scored to highest-scored (reverse order)
        for i in range(len(work_a) - 1, -1, -1):
            if work_a[i].tier != from_tier:
                continue

            if not _is_over_budget() and not _is_over_tier_budget(from_tier):
                break

            # Estimate savings before demoting
            mem = memories[i] if i < len(memories) else None
            if mem is not None:
                new_formatted = format_memory(mem, to_tier, config, content=work_c[i])
                savings_chars = len(work_f[i]) - len(new_formatted)
                savings_tokens = savings_chars // config.chars_per_token

                if savings_tokens < MIN_DEMOTION_SAVINGS_TOKENS:
                    # Skip demotion -- not worth it (AP3-T3)
                    continue

                # Apply demotion
                work_a[i] = TierAssignment(
                    index=work_a[i].index,
                    score=work_a[i].score,
                    normalized_score=work_a[i].normalized_score,
                    tier=to_tier,
                )
                work_f[i] = new_formatted
                demoted += 1

    # Drop lowest-scored memories until all budgets fit or at min_k
    while _needs_reduction() and len(work_a) > config.min_k:
        # Drop last (lowest score)
        work_a.pop()
        work_f.pop()
        work_c.pop()
        dropped += 1

    return (work_a, work_f, demoted, dropped)


# ============================================================
# 11. recall -- Main orchestrator
# ============================================================


def recall(
    memories: list["MemoryState"],
    params: "ParameterSet",
    config: RecallConfig,
    message: str,
    turn_number: int,
    contents: Optional[list[Optional[str]]] = None,
    current_time: float = 0.0,
) -> RecallResult:
    """Main recall pipeline orchestrator.

    Pipeline:
      1. Empty-memories fast-path BEFORE gate (AP1-F5).
      2. Gate check via should_recall().
      3. Validate contents length if provided.
      4. Rank memories via engine.rank_memories().
      5. adaptive_k to select k.
      6. assign_tiers to assign rendering tiers.
      7. format_memory for each assignment.
      8. budget_constrain for token budget enforcement.
      9. Assemble context string with MEMORY_SEPARATOR.

    Args:
        memories:     List of MemoryState to recall from.
        params:       ParameterSet for the scoring engine.
        config:       RecallConfig for pipeline parameters.
        message:      User message (for gating).
        turn_number:  Current conversation turn (for gating).
        contents:     Optional content strings parallel to memories.
        current_time: Current time for scoring.

    Returns:
        RecallResult with assembled context and metadata.

    Raises:
        TypeError:   If message is not str (from should_recall).
        ValueError:  If turn_number < 0 or contents length mismatches.
    """
    from hermes_memory.engine import rank_memories

    # Validate turn_number early (even for empty memories)
    if turn_number < 0:
        raise ValueError(f"turn_number must be >= 0, got {turn_number}")

    # 1. Empty-memories fast-path BEFORE gate (AP1-F5)
    if not memories:
        return RecallResult(
            context="",
            gated=False,
            k=0,
            tier_assignments=(),
            total_tokens_estimate=0,
            budget_exceeded=False,
            memories_dropped=0,
            memories_demoted=0,
            budget_overflow=0,
        )

    # 2. Gate check
    if not should_recall(message, turn_number, config):
        return RecallResult(
            context="",
            gated=True,
            k=0,
            tier_assignments=(),
            total_tokens_estimate=0,
            budget_exceeded=False,
            memories_dropped=0,
            memories_demoted=0,
            budget_overflow=0,
        )

    # 3. Validate contents length
    if contents is not None and len(contents) != len(memories):
        raise ValueError(
            f"contents length {len(contents)} != memories length {len(memories)}"
        )

    # 4. Rank memories
    ranked = rank_memories(params, memories, current_time)
    scores = [s for _, s in ranked]

    # 5. Adaptive k
    k = adaptive_k(scores, config)

    # 6. Assign tiers
    tier_assignments = assign_tiers(ranked, k, config)

    # 7. Format memories
    formatted = []
    for ta in tier_assignments:
        mem = memories[ta.index]
        content = contents[ta.index] if contents is not None else None
        formatted.append(format_memory(mem, ta.tier, config, content=content))

    # 8. Budget constrain
    selected_memories = [memories[ta.index] for ta in tier_assignments]
    selected_contents: Optional[list[Optional[str]]] = None
    if contents is not None:
        selected_contents = [contents[ta.index] for ta in tier_assignments]

    final_a, final_f, n_demoted, n_dropped = budget_constrain(
        tier_assignments,
        formatted,
        selected_memories,
        config,
        contents=selected_contents,
    )

    # 9. Assemble context
    context = MEMORY_SEPARATOR.join(final_f)
    total_tokens = len(context) // config.chars_per_token
    budget_overflow = max(0, total_tokens - config.total_budget)

    # Compute budget_exceeded: True if any constraining action was taken,
    # or if the unconstrained all-HIGH rendering would have exceeded budget.
    # The tier assignment itself acts as a budget constraint when it assigns
    # MEDIUM/LOW tiers that reduce token usage.
    if n_demoted > 0 or n_dropped > 0:
        budget_exceeded = True
    else:
        # Check if all-HIGH rendering of selected memories would exceed budget
        all_high_chars = sum(
            len(
                format_memory(
                    memories[ta.index],
                    Tier.HIGH,
                    config,
                    content=(contents[ta.index] if contents is not None else None),
                )
            )
            for ta in final_a
        )
        if len(final_a) > 1:
            all_high_chars += (len(final_a) - 1) * len(MEMORY_SEPARATOR)
        all_high_tokens = all_high_chars // config.chars_per_token
        budget_exceeded = all_high_tokens > config.total_budget

    return RecallResult(
        context=context,
        gated=False,
        k=len(final_a),
        tier_assignments=tuple(final_a),
        total_tokens_estimate=total_tokens,
        budget_exceeded=budget_exceeded,
        memories_dropped=n_dropped,
        memories_demoted=n_demoted,
        budget_overflow=budget_overflow,
    )
