# Recall Spec -- hermes_memory/recall.py

## Overview

Recall is the decision layer between the scoring/ranking engine (`engine.py`) and
the consumer (prompt assembly). It takes ranked memories from `rank_memories()` and
decides **how many** to retrieve (adaptive-k), **at what detail level** to present
them (three-tier), and **whether to retrieve at all** (gating). It enforces a
configurable token budget so that recalled context does not blow up the prompt.

New module: `hermes_memory/recall.py`
New test file: `tests/test_recall.py`

---

## 1. Dependencies

| Module | What recall.py uses |
|--------|---------------------|
| `engine.py` | `ParameterSet`, `MemoryState`, `rank_memories()`, `score_memory()` |
| `optimizer.py` | `PINNED` constants (for default ParameterSet construction) |
| `encoding.py` | `EncodingDecision.category` (for detail formatting in `format_memory`) |
| `core.py` | Transitively via engine.py. No direct imports. |

No new external dependencies. Only stdlib `dataclasses`, `enum`, `math`, `re`.

---

## 2. Tier Enum

```python
import enum

class Tier(enum.Enum):
    """Detail level for a recalled memory.

    Ordered from most to least detail. The integer values encode the
    ordering: lower value = more detail.
    """
    HIGH = 1
    MEDIUM = 2
    LOW = 3
```

### Invariants
- `Tier.HIGH.value < Tier.MEDIUM.value < Tier.LOW.value`
- Exactly three members. No sentinel values.

---

## 3. RecallConfig

```python
@dataclass(frozen=True)
class RecallConfig:
    """Configuration for the recall pipeline.

    Frozen to ensure thread safety and determinism. All thresholds
    are tunable without code changes.

    Fields are organized into four concern groups:
      (A) Algorithmic: affect recall decisions (adaptive-k, tier assignment)
      (B) Budget: control token allocation across tiers
      (C) Gating: control whether recall runs at all
      (D) Presentation: affect output formatting only

    Groups A and B should be tuned alongside the optimizer. Groups C and D
    depend on the target LLM model and may change independently.
    [Adversarial Pass 1, Finding 6]
    """

    # -- (A) Adaptive-k parameters --
    min_k: int = 1
    max_k: int = 10
    gap_buffer: int = 2
    epsilon: float = 0.01  # NOTE: This threshold operates in engine-score
                           # space (raw composite scores, range ~[0, 1.1]).
                           # It must be recalibrated if the optimizer
                           # significantly changes the SCORING WEIGHTS
                           # (w1-w4), NOT temperature (which has no effect
                           # on score_memory output).
                           # [AP1-F2, AP3-E1, AP3-E6]

    # -- (A) Tier thresholds (applied to normalized scores in [0, 1]) --
    # IMPORTANT: Scores must be STRICTLY ABOVE the threshold to qualify.
    # A score of exactly high_threshold is MEDIUM, not HIGH.
    # A score of exactly mid_threshold is LOW, not MEDIUM.
    # [Adversarial Pass 1, Finding 7]
    high_threshold: float = 0.7
    mid_threshold: float = 0.4

    # -- (B) Token budget (presentation-layer budget allocation) --
    # total_budget is a best-effort constraint. When min_k memories at
    # their lowest tier already exceed total_budget, min_k takes priority.
    # See budget_constrain's priority hierarchy: min_k > budget.
    # [Adversarial Pass 1, Finding 1]
    total_budget: int = 800
    # Per-tier budget shares control the demotion pass in budget_constrain.
    # If HIGH-tier memories exceed total_budget * high_budget_share tokens,
    # lowest-scored HIGH memories are demoted even if total budget is not
    # exceeded. This prevents one tier from starving others.
    # [Adversarial Pass 1, Finding 10]
    high_budget_share: float = 0.55
    medium_budget_share: float = 0.30
    low_budget_share: float = 0.15

    # -- (C) Gating parameters --
    gate_min_length: int = 8
    gate_trivial_patterns: tuple[str, ...] = (
        "ok", "okay", "yes", "no", "yeah", "yep", "nope",
        "sure", "thanks", "thank you", "thx", "ty",
        "got it", "understood", "alright", "right",
        "cool", "nice", "great", "fine", "good",
        "bye", "goodbye", "hi", "hello", "hey",
    )

    # -- (D) Presentation parameters --
    high_max_chars: int = 400
    medium_max_chars: int = 200
    low_max_chars: int = 80
    chars_per_token: int = 4  # Token estimation ratio. Default 4 is calibrated
                              # for English text. For CJK-heavy, emoji-heavy, or
                              # code-heavy content, use 2 (conservative) to avoid
                              # underestimating actual token counts by 60%+.
                              # [Adversarial Pass 2, Finding 1]
```

### Field Contracts

| Field | Domain | Invariant |
|-------|--------|-----------|
| `min_k` | [1, +inf) integer | Always return at least this many memories (if available) |
| `max_k` | [min_k, +inf) integer | Never return more than this many memories |
| `gap_buffer` | [0, +inf) integer | Extra memories past the score gap to include |
| `epsilon` | (0, +inf) float | Max gap below this threshold means "all scores are similar." Operates in engine-score space (~[0, 1.1]); must be recalibrated if optimizer changes scoring weights (w1-w4). Temperature has no effect on scores. [AP1-F2, AP3-E1] |
| `high_threshold` | (mid_threshold, 1.0] float | Normalized score STRICTLY ABOVE this is HIGH tier. Score exactly at threshold is MEDIUM. [AP1-F7] |
| `mid_threshold` | [0.0, high_threshold) float | Normalized score STRICTLY ABOVE this (and <= high_threshold) is MEDIUM tier. Score exactly at threshold is LOW. [AP1-F7] |
| `total_budget` | [1, +inf) integer | Best-effort maximum token budget. min_k takes priority if conflict. [AP1-F1] |
| `high_budget_share` | (0.0, 1.0) float | Fraction of budget for HIGH tier. Enforced in per-tier demotion pass. [AP1-F10] |
| `medium_budget_share` | (0.0, 1.0) float | Fraction of budget for MEDIUM tier. Enforced in per-tier demotion pass. [AP1-F10] |
| `low_budget_share` | (0.0, 1.0) float | Fraction of budget for LOW tier. Enforced in per-tier demotion pass. [AP1-F10] |
| `gate_min_length` | [0, +inf) integer | Messages shorter than this (chars) skip recall |
| `gate_trivial_patterns` | tuple of lowercase strings | Messages matching any of these exactly (after strip + lower) skip recall |
| `high_max_chars` | [1, +inf) integer | Max chars per HIGH memory rendering |
| `medium_max_chars` | [1, +inf) integer | Max chars per MEDIUM memory rendering |
| `low_max_chars` | [1, +inf) integer | Max chars per LOW memory rendering |
| `chars_per_token` | [1, +inf) integer | Token estimation divisor. 4 for English, 2 for CJK/code-heavy content. [AP2-F1] |

### Validation (__post_init__)
Raises ValueError if:
- `min_k < 1`
- `max_k < min_k`
- `gap_buffer < 0`
- `epsilon <= 0`
- `high_threshold <= mid_threshold`
- `mid_threshold < 0.0`
- `high_threshold > 1.0`
- `total_budget < 1`
- `high_budget_share + medium_budget_share + low_budget_share` not within 1e-10 of 1.0
- Any budget share <= 0
- `gate_min_length < 0`
- Any `*_max_chars < 1`
- `chars_per_token < 1`

### Edge Cases
- Default config with all defaults is valid.
- min_k == max_k means fixed-k (adaptive-k is constrained to that exact value).
- gap_buffer == 0 means strictly use the gap without buffer.
- epsilon very large means the fallback (flat scores) triggers more often.
- Empty gate_trivial_patterns disables trivial-message gating (only length gate remains).

---

## 4. TierAssignment

```python
@dataclass(frozen=True)
class TierAssignment:
    """A single memory with its tier and score metadata.

    Produced by assign_tiers(), consumed by format_memory() and
    budget_constrain().
    """
    index: int           # Original index in the MemoryState list
    score: float         # Engine-space composite score from rank_memories().
                         # Range: [0, 1 + novelty_start] (NOT clamped to [0,1]).
                         # This is the raw score used for ordering and adaptive-k.
                         # [Adversarial Pass 1, Finding 2]
    normalized_score: float  # Within-k normalized presentation score in [0, 1].
                             # Produced by min-max normalization across the selected
                             # k memories. Used only for tier assignment, NOT for
                             # ordering. [Adversarial Pass 1, Finding 2]
    tier: Tier
```

### Invariants
- `index >= 0`
- `0.0 <= normalized_score <= 1.0`
- `tier` is consistent with `normalized_score` and the config thresholds
  **EXCEPT** when the positional fallback applies (all k scores identical, k > 1)
  [AP1-F3]:
  - When scores are NOT all identical:
    - `normalized_score > config.high_threshold` implies `tier == Tier.HIGH`
    - `config.mid_threshold < normalized_score <= config.high_threshold` implies `tier == Tier.MEDIUM`
    - `normalized_score <= config.mid_threshold` implies `tier == Tier.LOW`
  - When all k scores are identical and k > 1:
    - `normalized_score == 1.0` for all entries (min-max normalization)
    - Tier is assigned positionally: first ceil(k/3) HIGH, next ceil(k/3) MEDIUM, rest LOW
    - The threshold-based invariant does NOT hold in this case

---

## 5. RecallResult

```python
@dataclass(frozen=True)
class RecallResult:
    """Output of the recall pipeline.

    Contains both the formatted context string for prompt injection
    and metadata about the recall decision for logging/debugging.
    """
    context: str
    gated: bool
    k: int
    tier_assignments: tuple[TierAssignment, ...]
    total_tokens_estimate: int
    budget_exceeded: bool
    budget_overflow: int       # [Adversarial Pass 1, Finding 1]
    memories_dropped: int
    memories_demoted: int
```

### Field Contracts

| Field | Domain | Description |
|-------|--------|-------------|
| `context` | str (may be empty) | Formatted context string ready for prompt injection |
| `gated` | bool | True if recall was skipped by the gate. `gated=True, k=0` means "gate decided not to recall." |
| `k` | int >= 0 | Number of memories in the final result (after budget constraints). `gated=False, k=0` means "memory database was empty." [AP1-F5] |
| `tier_assignments` | tuple of TierAssignment | One per recalled memory, in descending score order |
| `total_tokens_estimate` | int >= 0 | Approximate token count of `context` (chars / config.chars_per_token heuristic). Calibrated for English at default ratio; may undercount 60%+ for CJK. [AP2-F1] |
| `budget_exceeded` | bool | True if budget constraint had to demote or drop memories |
| `budget_overflow` | int >= 0 | Tokens by which the result exceeds the budget (0 when within budget). Non-zero only when min_k prevents further dropping. Callers MUST handle overflow > 0 by truncating context if the prompt would exceed model limits. [AP1-F1] |
| `memories_dropped` | int >= 0 | Number of memories dropped to meet budget |
| `memories_demoted` | int >= 0 | Number of memories demoted from a higher tier to meet budget |

### Invariants
- If `gated` is True: `context == ""`, `k == 0`, `tier_assignments == ()`, `memories_dropped == 0`, `memories_demoted == 0`, `budget_overflow == 0`
- `k == len(tier_assignments)`
- `tier_assignments` is sorted by descending score (same order as rank_memories output)
- `total_tokens_estimate == len(context) // config.chars_per_token` (integer division, using the config's token estimation ratio)
- `budget_overflow == max(0, total_tokens_estimate - config.total_budget)` [AP1-F1]
- `budget_overflow > 0` implies `k == min(config.min_k, len(input_memories))` (overflow only when min_k prevents further dropping) [AP1-F1]

---

## 6. should_recall(message, turn_number, config) -> bool

### Contract
- **Input:**
  - `message: str` -- the user's current message text
  - `turn_number: int` -- 0-indexed turn within the current session
  - `config: RecallConfig` -- recall configuration
- **Output:** `bool` -- True if memory retrieval should proceed, False to skip

### Algorithm
1. If `turn_number == 0`: return True (always recall on session start)
2. Strip and lowercase the message.
3. If `len(stripped) < config.gate_min_length`: return False
4. If stripped message (with trailing punctuation removed) matches any pattern in
   `config.gate_trivial_patterns` exactly: return False
5. If `"?" in message`: return True (questions always trigger recall)
6. Return True (default: recall)

### Rationale
The gate is a cheap heuristic that avoids embedding and retrieval for messages
that are very unlikely to benefit from recalled context. It is intentionally
biased toward **recalling** (the only skip cases are trivially short messages
and exact matches to a fixed set of acknowledgment patterns). False negatives
(skipping a message that needed recall) are more costly than false positives
(unnecessary retrieval), so the gate is conservative.

### Edge Cases
- Empty string `""` on turn 0: returns True (first turn override)
- Empty string `""` on turn > 0: returns False (length < gate_min_length)
- `"ok?"` on turn > 0: returns True (question mark override fires before trivial check; but also: the
  order of checks means step 3 length check applies first -- `"ok?"` is 3 chars, which is < 8, so
  returns False. **Important**: the question-mark check must come BEFORE the length check to handle
  this correctly. Revised algorithm order below.)

### Revised Algorithm (corrected for question-mark priority)
1. If `turn_number == 0`: return True
2. Strip the message. If the stripped message is empty and turn > 0: return False
3. If `"?" in message`: return True (questions always trigger, regardless of length)
4. If `len(stripped) < config.gate_min_length`: return False
5. Lowercase the stripped message. Remove trailing punctuation (`.!,;:`).
   If the result matches any pattern in `config.gate_trivial_patterns` exactly: return False
6. Return True

### Detailed Edge Cases

| Input | Turn | Result | Reason |
|-------|------|--------|--------|
| `""` | 0 | True | First turn always recalls |
| `""` | 1 | False | Empty after strip, not turn 0 |
| `"ok"` | 1 | False | Trivial pattern match |
| `"OK"` | 1 | False | Case-insensitive trivial match |
| `"ok."` | 1 | False | Trailing punctuation stripped before trivial check |
| `"ok?"` | 1 | True | Question mark override (step 3) |
| `"yes"` | 1 | False | Trivial pattern match |
| `"Tell me about X"` | 1 | True | Not trivial, long enough |
| `"What is X?"` | 3 | True | Question mark |
| `"hi"` | 0 | True | First turn override |
| `"hi"` | 1 | False | Length < gate_min_length (8) AND trivial match |
| `"ok sure"` | 1 | True | Does not exactly match any trivial pattern |
| `"   "` | 5 | False | Empty after strip |

### Known False Negative Category [AP3-E5]
The gate_min_length=8 threshold blocks short imperative phrases (5-7 chars)
that may reference prior context: "tell me" (7 chars), "help me" (7 chars),
"why not" (7 chars), "do that" (7 chars). These are blocked by the length
check before reaching the trivial pattern check.

The question-mark override catches all interrogative forms regardless of length.
The false negative rate for conversational text is estimated at < 5%, affecting
only short imperatives without question marks. This is an acceptable tradeoff:
the alternative (gate_min_length=6) would add 2 more false positives per 100
messages from patterns like "thanks" (6 chars) that are not in the trivial list.

### Error Handling
- `turn_number < 0`: raise ValueError
- `message` is not a string: raise TypeError
- `config` is not RecallConfig: standard Python typing (no runtime check)

---

## 7. adaptive_k(scores, config) -> int

### Contract
- **Input:**
  - `scores: list[float]` -- composite scores in **descending** order (as returned by rank_memories)
  - `config: RecallConfig` -- recall configuration
- **Output:** `int` -- how many memories to retrieve

### Algorithm
1. If `len(scores) == 0`: return 0
2. If `len(scores) == 1`: return 1
3. **Truncate to top slice [AP2-F3]:** `effective = scores[:config.max_k + config.gap_buffer + 1]`.
   This restricts gap detection to the top portion of the ranking. Without
   truncation, large pools (N > 100) almost always have their largest gap
   deep in the list, causing adaptive_k to degenerate into fixed max_k.
   The `+1` ensures the gap just below the max_k boundary is visible.
4. Compute consecutive gaps on the effective slice:
   `gaps[i] = effective[i] - effective[i+1]` for i in 0..len(effective)-2
5. Find `max_gap = max(gaps)`
6. If `max_gap < config.epsilon`:
   - All scores are nearly equal within the top slice. Fall back to
     `min(config.min_k + config.gap_buffer, len(scores), config.max_k)`
7. Otherwise:
   - `gap_idx = argmax(gaps)` (first occurrence if ties)
   - `k_raw = gap_idx + 1 + config.gap_buffer`
   - The `+1` accounts for the fact that the gap at index `i` means the natural cut
     is after `i+1` items (items 0..gap_idx are above the gap)
8. Clamp: `k = max(config.min_k, min(k_raw, len(scores), config.max_k))`
9. Return `k`

### Interaction with Scoring Weights [AP3-E1]
**NOTE:** The engine's temperature parameter does NOT affect `score_memory()` or
`rank_memories()`. Temperature is used exclusively in `soft_select()` for the
dynamics pipeline (`step_dynamics`, `simulate`). The scoring pipeline that feeds
into adaptive_k is determined entirely by the scoring weights w1-w4.

With the tuned weights (w1=0.4109, w2=0.05, w3=0.30, w4=0.2391), relevance
(w1) is the dominant factor. Score spread is primarily driven by relevance
variation in the memory pool, with secondary contributions from importance (w3)
and activation (w4 * sigmoid(access_count)). The novelty bonus (up to 0.3 for
brand-new memories) adds a creation-time-dependent offset.

For the default PINNED parameters and optimized weights, typical composite scores
(base + novelty) range from approximately 0.15 to 1.1. Score differences between
relevant and irrelevant memories are typically 0.03 to 0.35. An epsilon of 0.01
catches genuinely flat score distributions without triggering on normal score
spreads.

### Recall-vs-Precision Tradeoff [AP3-E2]
Adaptive-k optimizes for **recall** (never miss a relevant memory) at the cost
of **precision** (may include irrelevant ones). Empirical testing shows adaptive-k
returns a mean of ~6.3 memories with tuned parameters, capturing all relevant
memories in 100% of trials. Fixed k=3 would miss relevant memories in ~43% of
trials. The tier system and budget constraint handle precision by demoting and
dropping the least-scored inclusions.

### Epsilon Sensitivity to Scoring Weights [AP1-F2, AP3-E1, AP3-E6]
The epsilon threshold operates in absolute engine-score space (range ~[0, 1.1]).
A gap of 0.01 is ~0.9% of this range. If the optimizer changes **scoring weights
(w1-w4)** such that all scores compress into a narrower band (e.g., increasing
w2 to make recency dominant, which compresses differences because all memories
share similar last_access_time), then 0.01 becomes a larger fraction of the
effective range -- no longer "flat." Conversely, if the score range expands,
epsilon may need to increase. Epsilon should be recalibrated whenever the
optimizer produces a new weight vector that significantly changes the score
distribution.

**Calibration fragility [AP3-E6]:** With current tuned weights, empirical testing
over 200 random memory pools shows a minimum max_gap of ~0.0102 -- only 0.0002
above the epsilon threshold. This means epsilon=0.01 has near-zero headroom.
Changes to w1-w4, novelty_start, or the memory population characteristics could
push the system into constant epsilon-fallback mode. A future weight optimization
run should include an epsilon-fallback-rate constraint (target: < 5% fallback
rate over 500+ random memory pools).

### Fallback Rationale [AP1-F9]
When all scores are within epsilon (flat distribution), adaptive-k returns
`min(min_k + gap_buffer, len(scores), max_k)` rather than just `min_k`. The
gap_buffer acts as a "benefit of the doubt" factor: when the system cannot
identify a natural cutoff, it returns more memories rather than fewer. This
biases toward recall (showing context) over silence (dropping potentially
useful memories). The rationale is that false negatives (failing to recall a
useful memory) are more costly than false positives (recalling an irrelevant
one), consistent with the gating philosophy in should_recall.

### Edge Cases

| Input | Config | Result | Reason |
|-------|--------|--------|--------|
| `[]` | default | 0 | Empty list |
| `[0.8]` | default | 1 | Single memory |
| `[0.9, 0.5, 0.4, 0.3]` | default | 3 | Gap at idx 0 (0.4), k_raw = 1 + 2 = 3 |
| `[0.9, 0.89, 0.88, 0.87]` | default(eps=0.01) | 3 | max_gap = 0.01, ties epsilon, fallback k = min(1+2, 4, 10) = 3 |
| `[0.9, 0.89, 0.88, 0.87]` | eps=0.02 | 3 | max_gap = 0.01 < 0.02, fallback |
| `[0.9, 0.5, 0.49, 0.48, 0.47]` | default | 3 | Gap at idx 0 (0.4), k_raw = 1+2 = 3 |
| `[0.9, 0.5, 0.49, 0.48, 0.47]` | gap_buffer=0 | 1 | Gap at idx 0, k_raw = 1+0 = 1 |
| `[0.5, 0.5, 0.5]` | default | 3 | All gaps = 0 < eps, fallback min(3, 3, 10) = 3 |
| `[0.9, 0.1]` | min_k=3 | 2 | k_raw = 1+2 = 3, clamped to min(3, 2) = 2 (can't exceed list) |
| scores of length 20 | max_k=5 | <= 5 | Clamped to max_k |

### Invariants
- Return value is always in [0, len(scores)]
- Return value is always in [min_k, max_k] when len(scores) >= min_k
- Return value is always <= len(scores)
- Return 0 only when scores is empty

### Error Handling
- `scores` not sorted descending: raise ValueError (check `scores[i] >= scores[i+1]`
  for all i in the full list, not just the truncated slice)
- Any score is NaN: raise ValueError (checked on full list before truncation)

---

## 8. assign_tiers(scores, k, config) -> list[TierAssignment]

### Contract
- **Input:**
  - `scores: list[tuple[int, float]]` -- output of `rank_memories()`: list of (index, score) tuples
    in descending score order
  - `k: int` -- number of memories to include (from adaptive_k)
  - `config: RecallConfig` -- recall configuration
- **Output:** `list[TierAssignment]` -- one TierAssignment per selected memory, in descending score order

### Algorithm
1. Take the first `k` entries from `scores`.
2. If k == 0: return []
3. Normalize the scores of the selected k memories to [0, 1]:
   - `s_max = scores[0][1]` (highest score among selected)
   - `s_min = scores[k-1][1]` (lowest score among selected)
   - If `s_max == s_min`: all normalized scores are 1.0 (avoid division by zero;
     single-score or identical-score case, treat all as maximally relevant within
     the selection)
   - Otherwise: `normalized = (score - s_min) / (s_max - s_min)`
4. Assign tier based on normalized score:
   - `normalized > config.high_threshold` -> Tier.HIGH
   - `config.mid_threshold < normalized <= config.high_threshold` -> Tier.MEDIUM
   - `normalized <= config.mid_threshold` -> Tier.LOW
5. Return list of TierAssignment instances.

### Normalization Rationale
Scores from `rank_memories()` include the novelty bonus and are NOT clamped to [0,1].
Absolute thresholds would break under different parameter regimes (e.g., if the
optimizer moves weights around, the raw score range changes). Normalizing within
the selected k memories makes tier assignment relative: the highest-scoring memory
in the selection is always at normalized 1.0, the lowest at 0.0.

**Amplification effect [AP2-F6]:** When k comes from the epsilon fallback
(flat scores), within-k normalization amplifies negligible score differences
into full-range [0, 1] normalized scores. For example, 3 scores
[0.550, 0.548, 0.546] (raw spread: 0.004) normalize to [1.0, 0.5, 0.0],
producing tiers [HIGH, MEDIUM, LOW]. This is acceptable because tier
assignment is a presentation decision, not a quality judgment: even among
near-identical memories, showing one at HIGH detail, one at MEDIUM, and
one at LOW provides more information diversity than showing all at the
same tier. The positional fallback (AP1-F3) handles the true all-identical
case where normalization produces no spread at all.

**Special case: k=1.** When only one memory is selected, s_max == s_min, so the
normalization defaults to 1.0 for the single memory, placing it in Tier.HIGH.
This is correct: if you selected only one memory, it should get full detail.

**Special case: all k scores identical AND k > 1.** [Adversarial Pass 1,
Finding 3] When all selected memories have the same score, min-max
normalization maps all to 1.0, which would place all in Tier.HIGH. This
wastes the budget: if 8 memories are all HIGH at 400 chars each, that is
3200 chars = 800 tokens, consuming the entire default budget on a single tier.
The tier system exists to distribute scarce budget, not to treat all memories
as maximally important.

**Positional fallback for identical scores (k > 1):** When s_max == s_min
and k > 1, assign_tiers ignores the normalized scores and uses positional
tier assignment: the first `ceil(k/3)` memories get HIGH, the next `ceil(k/3)`
get MEDIUM, the rest get LOW. This distributes budget across tiers even when
scores provide no discrimination signal. The positional order follows the
rank_memories output (descending score, then ascending index for tie-breaking),
so lower-index memories get higher tiers.

The k=1 case is unchanged: a single memory always gets HIGH.

### Boundary-Value Edge Cases [AP1-F7]

| Normalized Score | high_threshold=0.7 | mid_threshold=0.4 | Tier |
|------------------|--------------------|--------------------|------|
| 0.71 | > 0.7 | - | HIGH |
| 0.70 | NOT > 0.7 | > 0.4 | MEDIUM |
| 0.41 | NOT > 0.7 | > 0.4 | MEDIUM |
| 0.40 | NOT > 0.7 | NOT > 0.4 | LOW |
| 0.00 | NOT > 0.7 | NOT > 0.4 | LOW |

### Edge Cases

| Input | k | Result |
|-------|---|--------|
| `[]` | 0 | `[]` |
| `[(0, 0.8)]` | 1 | `[TierAssignment(0, 0.8, 1.0, HIGH)]` |
| `[(0, 0.9), (1, 0.5)]` | 2 | idx 0: norm=1.0 (HIGH), idx 1: norm=0.0 (LOW) |
| `[(0, 0.9), (1, 0.8), (2, 0.3)]` | 3 | idx 0: norm=1.0 (HIGH), idx 1: norm=0.833 (HIGH), idx 2: norm=0.0 (LOW) |
| `[(0, 0.5), (1, 0.5), (2, 0.5)]` | 3 | Positional fallback: idx 0 HIGH, idx 1 MEDIUM, idx 2 LOW [AP1-F3] |
| `[(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5), (5, 0.5)]` | 6 | Positional: idx 0-1 HIGH, idx 2-3 MEDIUM, idx 4-5 LOW [AP1-F3] |
| `[(0, 1.2), (1, 0.9)]` | 2 | norm still [0,1]; scores > 1.0 are valid (novelty bonus) |

### Invariants
- `len(output) == k`
- Output is in descending score order (same as input)
- All normalized scores in [0.0, 1.0]
- Tier assignment is monotonic: if `norm_a > norm_b`, then `tier_a.value <= tier_b.value`
  (higher normalized score never gets a lower-detail tier)

### Error Handling
- `k < 0`: raise ValueError
- `k > len(scores)`: raise ValueError
- `scores` not in descending order: raise ValueError

---

## 9. format_memory(memory, tier, config) -> str

### Contract
- **Input:**
  - `memory: MemoryState` -- the memory to format
  - `tier: Tier` -- the detail level to render at
  - `config: RecallConfig` -- recall configuration (for max_chars per tier)
- **Output:** `str` -- rendered memory text, truncated to the tier's character limit

### Rendering by Tier

#### Tier.HIGH
Full narrative rendering. Format:
```
[Memory] relevance={relevance:.2f} strength={strength:.1f} importance={importance:.1f}
  Content: {full content text, up to high_max_chars chars}
  Accessed {access_count} times, age={creation_time:.0f} steps
```
Maximum length: `config.high_max_chars` characters total (hard truncate with "..." if exceeded).

#### Tier.MEDIUM
Summary rendering. Format:
```
[Memory] relevance={relevance:.2f} importance={importance:.1f}
  Summary: {content truncated to medium_max_chars chars}
```
Maximum length: `config.medium_max_chars` characters total.

#### Tier.LOW
Fact-only rendering. Format:
```
[Memory] rel={relevance:.2f} imp={importance:.1f} age={creation_time:.0f}
```
Maximum length: `config.low_max_chars` characters total.

### Design Note: MemoryState Does Not Carry Content
The current `MemoryState` dataclass (engine.py) is a numerical snapshot: it holds
`relevance`, `last_access_time`, `importance`, `access_count`, `strength`, and
`creation_time`. It does NOT hold the memory's text content, embedding, category,
or any other metadata from the storage layer.

This is by design: engine.py is a pure-math scoring module. The text content lives
in the storage layer (architecture.md Section 3: `MemoryEntry.content`).

**Consequence for recall.py:** The `format_memory` function receives a `MemoryState`
(numerical only). To produce meaningful text at HIGH and MEDIUM tiers, it needs
access to the actual memory content. Two options:

1. **Extend the input**: Accept a parallel `content: str` parameter alongside the
   MemoryState. This keeps MemoryState pure.
2. **New wrapper type**: Create a `RichMemory` that bundles MemoryState + content +
   category. This is cleaner but adds a type.

**Decision: Option 1.** Add an optional `content: str | None = None` parameter to
`format_memory`. When content is None, the format falls back to metadata-only
rendering for all tiers. When content is provided, HIGH and MEDIUM tiers include it.

**Dual-purpose content=None semantics [AP1-F8]:** The content=None path serves
two distinct runtime scenarios:

1. **Pure-math testing:** No storage layer exists. content is absent because the
   test infrastructure does not provide text content. This is a test convenience.
2. **Production metadata-only:** The storage layer exists but a particular memory's
   content is unavailable (deleted, corrupted, or not yet loaded). content=None is
   a runtime condition requiring graceful degradation.

Both scenarios produce identical output (metadata-only rendering). This is
intentional for v1. If a future version needs to distinguish "content not
requested" from "content missing," the None sentinel must be replaced with an
explicit union type (e.g., `Content | NotRequested | Missing`).

**Demotion savings without content [AP1-F4]:** When content=None, the difference
between HIGH (~70 chars) and LOW (~35 chars) metadata-only rendering is minimal
(~35 chars = ~9 tokens per memory). budget_constrain should be aware that demotion
without content yields marginal savings. When `contents` is entirely None,
budget_constrain should prefer dropping over repeated demotion, since demoting
all memories from HIGH to LOW saves roughly 9 tokens each, which may not be
enough to bring the total under budget.

### Revised Signature
```python
def format_memory(
    memory: MemoryState,
    tier: Tier,
    config: RecallConfig,
    content: str | None = None,
) -> str:
```

### Rendering with Content

#### Tier.HIGH (content provided)
```
[Memory] relevance={relevance:.2f} strength={strength:.1f} importance={importance:.1f}
  {content, truncated to high_max_chars - header_length with "..."}
  Accessed {access_count} times, age={creation_time:.0f} steps
```

#### Tier.HIGH (content is None)
```
[Memory] relevance={relevance:.2f} strength={strength:.1f} importance={importance:.1f}
  Accessed {access_count} times, age={creation_time:.0f} steps
```

#### Tier.MEDIUM (content provided)
```
[Memory] relevance={relevance:.2f} importance={importance:.1f}
  {content, truncated to medium_max_chars - header_length with "..."}
```

#### Tier.MEDIUM (content is None)
```
[Memory] relevance={relevance:.2f} importance={importance:.1f}
```

#### Tier.LOW (always metadata-only, content ignored)
```
[Memory] rel={relevance:.2f} imp={importance:.1f} age={creation_time:.0f}
```

### Truncation
When content exceeds the remaining character budget after the header:
- Truncate at the last word boundary before the limit
- Append "..." (3 chars, included in budget)
- If even "..." doesn't fit, return just the header

### Edge Cases
- `content == ""`: treated as content-provided but empty (no content line rendered)
- `content` with newlines: replace with spaces before truncation
- Very long content (10000+ chars): truncated correctly
- `memory.relevance == 0.0` and `tier == HIGH`: still formats (low score doesn't change format)
- Negative `creation_time`: impossible per MemoryState validation, no handling needed

### Error Handling
- Invalid tier: raise ValueError (should never happen with Tier enum, but defensive)
- Invalid MemoryState: MemoryState's own __post_init__ prevents this

---

## 10. budget_constrain(assignments, formatted, config) -> tuple[list[TierAssignment], list[str]]

### Contract
- **Input:**
  - `assignments: list[TierAssignment]` -- tier assignments from assign_tiers, in descending score order
  - `formatted: list[str]` -- parallel list of formatted memory strings from format_memory
  - `config: RecallConfig` -- recall configuration
- **Output:** tuple of:
  - `list[TierAssignment]` -- possibly modified (demoted) assignments, may be shorter than input
  - `list[str]` -- corresponding formatted strings, same length as first element
  - `int` -- number of memories demoted
  - `int` -- number of memories dropped

### Priority Hierarchy [AP1-F1]
Budget constraint enforces this priority order:
1. **min_k** (highest priority): Never drop below min_k memories, even if over budget.
2. **Per-tier budget shares** (medium priority): Prevent one tier from consuming all budget.
3. **Total budget** (lowest priority): Best-effort constraint on total token count.

When min_k memories at their lowest tier (LOW) still exceed total_budget, the
budget constraint stops and returns the over-budget result. The caller receives
`budget_overflow > 0` and is responsible for truncating context if needed.

### Algorithm
1. **Normalize contents [AP1-F4]:** If `contents is None`, set
   `contents = [None] * len(assignments)`. This eliminates the double-None
   ambiguity (outer None vs inner None) before entering the demotion loop.
2. Estimate total tokens: `total = sum(len(s) // config.chars_per_token for s in formatted)`.
   (Heuristic: 1 token ~ `chars_per_token` characters. Default 4 for English.
   See AP2-F1 for non-English accuracy. For English, this overestimates by
   ~20-40% for typical memory renderings, which is conservative. [AP3-T4, AP2-F1])
3. If `total <= config.total_budget`: check per-tier budgets (step 4). If all
   per-tier budgets also satisfied: return inputs unchanged with demoted=0, dropped=0.
3a. **Content-None optimization [AP3-T3]:** If ALL entries in `contents` are
   None, skip the demotion passes (steps 4 and 5) entirely and proceed directly
   to the drop pass (step 6). Without content, demotion saves only 2-8 tokens
   per memory (metadata-only rendering differences between tiers are minimal).
   Dropping is more effective for reclaiming budget in metadata-only mode.
4. **Per-tier demotion pass [AP1-F10, AP2-F2]** (even if total budget not exceeded):
   - For each tier T in [HIGH, MEDIUM] **in order** (HIGH first, then MEDIUM):
     - Compute tier tokens: `tier_total = sum(len(s) // config.chars_per_token for s in formatted where tier == T)`
     - Compute tier budget: `tier_budget = config.total_budget * config.{T}_budget_share`
     - While `tier_total > tier_budget` and there exist memories at tier T:
       - Demote the lowest-scored T-tier memory to T+1 tier, re-format, recalculate
       - Count demotion
   - **Cascade [AP2-F2]:** The MEDIUM pass runs AFTER the HIGH pass and
     INCLUDES memories that were just demoted from HIGH. This creates
     a demotion cascade: if many HIGHs are demoted to MEDIUM, the MEDIUM
     tier may now exceed its budget, triggering further demotions to LOW.
   - **Demotion skip [AP2-F4]:** When a memory has content=None, the
     savings from MEDIUM->LOW demotion are ~1 token (5 chars). Skip this
     demotion and prefer dropping instead. Concretely: if
     `len(current_formatted) - len(demoted_formatted) < MIN_DEMOTION_SAVINGS_TOKENS * config.chars_per_token`,
     do not demote this memory; mark it as a candidate for the drop pass
     instead.
5. **Total budget demotion pass** (iterate from lowest-scored to highest-scored):
   - Recalculate total after per-tier pass (using `config.chars_per_token`).
   - For each memory, if total still exceeds budget:
     - If tier is HIGH: demote to MEDIUM, re-format, recalculate total
     - If tier is MEDIUM: demote to LOW, re-format, recalculate total
     - If tier is LOW: cannot demote further, skip for now
   - Count demotions.
6. **Drop pass** (if still over budget after demotion, iterate from lowest-scored):
   - Remove the lowest-scored memory from both lists, recalculate total
   - Repeat until within budget or only min_k memories remain
   - Count drops.
7. Return the final lists, demotion count, and drop count.

### Default Budget Utilization [AP2-F5]
With default configuration (max_k=10, high_max_chars=400, total_budget=800),
the budget constraint rarely activates. Concrete utilization for typical
scenarios:

| Scenario | Tier Distribution | Tokens | Utilization |
|---|---|---|---|
| 6 memories, mixed scores | 1H/3M/2L | 268 | 33.5% |
| 10 memories, good spread | 3H/4M/3L | 529 | 66.1% |
| 10 memories, all content-rich | 3H/4M/3L | 529 | 66.1% |
| 5 HIGH with full content | 5H/0M/0L | 505 | 63.1% |

The budget system primarily activates with non-default configurations
(tight budgets, large max_k, or large max_chars values). Tests should
include tight-budget scenarios (e.g., total_budget=100) to exercise the
full budget cascade.

### Why Demote Before Drop
Dropping a memory entirely loses information. Demoting it to a lower tier preserves
at least the existence signal (LOW tier still shows relevance, importance, age).
The pipeline always tries to keep as many memories as possible, at reduced detail,
before removing any.

### Re-formatting on Demotion
When a memory is demoted, its formatted string must be regenerated at the new tier.
This requires the original MemoryState and content. To avoid threading these through
budget_constrain, the function accepts a `reformat_fn` callback:

### Revised Signature
```python
def budget_constrain(
    assignments: list[TierAssignment],
    formatted: list[str],
    config: RecallConfig,
    reformat_fn: Callable[[MemoryState, Tier, RecallConfig], str] | None = None,
    memories: list[MemoryState] | None = None,
    contents: list[str | None] | None = None,
) -> tuple[list[TierAssignment], list[str], int, int]:
```

When `reformat_fn` is None (testing without content), demoted memories get a
shorter metadata-only string estimated by the tier's max_chars limit (a placeholder
string of `"[demoted to {tier.name}]"` padded to approximate length).

When `reformat_fn` is provided with `memories` and `contents`, demoted memories
are re-formatted properly.

**Simplified alternative:** Instead of the callback, `budget_constrain` can
directly call `format_memory`. This is simpler but couples the two functions.

**Decision: Direct coupling.** `budget_constrain` takes `memories` and `contents`
as parallel lists and calls `format_memory` directly when demoting. This avoids
callback complexity and is easier to test.

### Final Signature
```python
def budget_constrain(
    assignments: list[TierAssignment],
    formatted: list[str],
    memories: list[MemoryState],
    config: RecallConfig,
    contents: list[str | None] | None = None,
) -> tuple[list[TierAssignment], list[str], int, int]:
```

### Token Estimation [AP2-F1, AP2-F8]
The heuristic `len(s) // config.chars_per_token` approximates token count.
The default `chars_per_token=4` is calibrated for English text with typical
LLM tokenizers (GPT-4, Claude).

**Accuracy by content type:**

| Content Type | chars//4 Error | chars//2 Error |
|---|---|---|
| English prose | +/- 0% | +100% (overcount) |
| Python code | -14% undercount | +43% overcount |
| URLs | -26% undercount | +24% overcount |
| JSON metadata | -21% undercount | +28% overcount |
| CJK text | -64% undercount | -28% undercount |
| Emoji-heavy | -38% undercount | +25% overcount |

For English-only deployments, `chars_per_token=4` is accurate. For
multilingual or code-heavy content, `chars_per_token=2` is conservative
but prevents budget overflows.

**Rounding undercount [AP2-F8]:** Per-string integer division introduces
a systematic undercount of up to ~1% for k=10 (each string and separator
independently loses up to 3 chars to floor division). This is within the
tolerance of the heuristic.

The alternative (importing tiktoken or a tokenizer) adds a heavy dependency
and ~50ms latency per estimation. The heuristic is acceptable because the
budget is itself a soft target, not a hard API limit.

### Edge Cases

| Scenario | Budget | Input | Result |
|----------|--------|-------|--------|
| Under budget | 800 | 3 memories, ~200 tokens total | Unchanged |
| Over budget, demotion sufficient | 200 | 3 HIGH memories (~300 tokens) | 1-2 demoted to MEDIUM/LOW, none dropped |
| Over budget, demotion insufficient | 100 | 5 HIGH memories (~500 tokens) | All demoted, then dropped from bottom |
| Single memory over budget | 50 | 1 HIGH memory (~200 tokens) | Demoted to MEDIUM then LOW; if still over, kept anyway (min_k=1) |
| Empty input | 800 | `[]` | `([], [], 0, 0)` |

### Invariants
- `len(output[0]) == len(output[1])`
- `len(output[0]) >= min(config.min_k, len(assignments))` (never drop below min_k)
- If `len(assignments) <= config.min_k`, no memories are dropped (only demotion)
- **Priority hierarchy [AP1-F1]:** min_k > per-tier budget > total budget.
  Total estimated tokens of output formatted strings <= config.total_budget
  UNLESS min_k prevents further dropping. When min_k prevents compliance,
  the result is returned over-budget and `budget_overflow` in RecallResult
  reports the excess.
- **Per-tier demotion vs total budget [AP2-F7]:** Per-tier demotion may
  trigger even when total budget is not exceeded, because the per-tier
  shares are fixed allocations, not redistributed from underutilizing tiers.
  Example: 5H/3M/2L = 677 tokens (< 800 budget), but HIGH at 500 tokens
  exceeds its 440-token share, triggering demotion of 1 HIGH to MEDIUM.
- Output is still in descending score order
- `output[2]` (demoted count) >= 0
- `output[3]` (dropped count) == `len(assignments) - len(output[0])`

### Error Handling
- `len(assignments) != len(formatted)`: raise ValueError
- `len(memories) != len(assignments)` when memories provided: raise ValueError
- `len(contents) != len(assignments)` when contents provided: raise ValueError

---

## 11. recall(query_embedding, memories, params, config) -> RecallResult

### Contract
- **Input:**
  - `query_embedding: list[float] | None` -- not used in the current implementation
    (embedding similarity is already captured in MemoryState.relevance). Accepted for
    forward compatibility with future direct-embedding pipelines. Ignored.
  - `memories: list[MemoryState]` -- all candidate memories, already evaluated against
    the query (relevance field populated)
  - `params: ParameterSet` -- parameters for scoring (from optimizer or default)
  - `config: RecallConfig` -- recall configuration
  - `message: str` -- the user's message text (for gating)
  - `turn_number: int` -- 0-indexed session turn (for gating)
  - `contents: list[str | None] | None` -- parallel list of memory content strings
    for HIGH/MEDIUM tier rendering. None means metadata-only rendering.
  - `current_time: float` -- current time for scoring (default: 0.0)
- **Output:** `RecallResult`

### Full Signature
```python
def recall(
    memories: list[MemoryState],
    params: ParameterSet,
    config: RecallConfig,
    message: str,
    turn_number: int,
    contents: list[str | None] | None = None,
    current_time: float = 0.0,
) -> RecallResult:
```

Note: `query_embedding` removed from signature. It was in the design requirements
but is unnecessary: MemoryState.relevance already encodes the cosine similarity.
Adding a dead parameter would mislead callers. If direct-embedding pipelines are
added later, the signature can be extended.

### Algorithm
1. **Empty-memories fast-path [AP1-F5]**: `if len(memories) == 0:`
   Return `RecallResult(context="", gated=False, k=0, tier_assignments=(), total_tokens_estimate=0, budget_exceeded=False, budget_overflow=0, memories_dropped=0, memories_demoted=0)`
   Rationale: gating is about avoiding unnecessary work. If there are no
   memories, there is no work to avoid, and the gate's decision is moot.
   Placing this before the gate check also avoids running pattern matching
   on the message when the result would be k=0 regardless.

2. **Gate check**: `if not should_recall(message, turn_number, config):`
   Return `RecallResult(context="", gated=True, k=0, tier_assignments=(), total_tokens_estimate=0, budget_exceeded=False, budget_overflow=0, memories_dropped=0, memories_demoted=0)`

3. **Rank**: `ranked = rank_memories(params, memories, current_time)`
   This returns `list[tuple[int, float]]` in descending score order.

4. **Adaptive-k**: `scores = [s for _, s in ranked]`; `k = adaptive_k(scores, config)`

5. **Tier assignment**: `assignments = assign_tiers(ranked, k, config)`

6. **Format each memory**:
   ```python
   formatted = []
   for ta in assignments:
       mem = memories[ta.index]
       content = contents[ta.index] if contents is not None else None
       formatted.append(format_memory(mem, ta.tier, config, content))
   ```

7. **Budget constraint**:
   ```python
   selected_memories = [memories[ta.index] for ta in assignments]
   selected_contents = [contents[ta.index] if contents else None for ta in assignments]
   final_assignments, final_formatted, n_demoted, n_dropped = budget_constrain(
       assignments, formatted, selected_memories, config, selected_contents
   )
   ```

8. **Assemble context**: Join formatted memories with `"\n---\n"` separator.

9. **Build result**:
   ```python
   context = "\n---\n".join(final_formatted)
   tokens = len(context) // config.chars_per_token
   overflow = max(0, tokens - config.total_budget)  # [AP1-F1]
   return RecallResult(
       context=context,
       gated=False,
       k=len(final_assignments),
       tier_assignments=tuple(final_assignments),
       total_tokens_estimate=tokens,
       budget_exceeded=(n_demoted > 0 or n_dropped > 0),
       budget_overflow=overflow,
       memories_dropped=n_dropped,
       memories_demoted=n_demoted,
   )
   ```

### Edge Cases

| Scenario | Result |
|----------|--------|
| Empty memories list | `RecallResult(context="", gated=False, k=0, ...)` (fast-path before gate) [AP1-F5] |
| All memories score identically | adaptive_k returns fallback k; positional tier assignment (not all HIGH) [AP1-F3] |
| Message gated | `RecallResult(context="", gated=True, ...)` |
| Single memory | k=1, assigned HIGH, formatted at HIGH |
| Budget exceeded | Context contains fewer/shorter memories; metadata reflects demotions/drops |
| 100 memories | Only max_k (10) are considered; rest dropped by adaptive_k |
| contents shorter than memories | raise ValueError |

### Invariants
- If memories is non-empty and not gated: `k >= 1`
- `len(tier_assignments) == k`
- `total_tokens_estimate == len(context) // config.chars_per_token`
- `context` contains exactly `k - 1` separator instances (or 0 if k <= 1)

### Error Handling
- `turn_number < 0`: propagated from should_recall (ValueError)
- `contents is not None and len(contents) != len(memories)`: raise ValueError
- Invalid ParameterSet: caught at ParameterSet construction (upstream)

---

## 12. Module-Level Constants

```python
# Default RecallConfig instance for use when no custom config is provided.
DEFAULT_RECALL_CONFIG: RecallConfig = RecallConfig()

# Separator between memories in the assembled context string.
MEMORY_SEPARATOR: str = "\n---\n"

# Token estimation divisor (chars per token, approximate).
# This constant is the default for RecallConfig.chars_per_token.
# NOTE: This consistently OVERestimates token count by ~20-40% for typical
# memory renderings (metadata strings, short English text). This is
# intentional -- overestimation is conservative (stays under budget).
# The effective budget is ~60-80% of the configured total_budget.
# See AP2-F1 for accuracy analysis by content type.
# [AP3-T4, AP2-F1]
CHARS_PER_TOKEN: int = 4

# Conservative token estimation for non-English content. [AP2-F1]
# CJK text has ~1.5 chars/token; code ~3.5; emoji ~2.5. Using 2
# overestimates for English but prevents budget overflow for
# multilingual content.
CHARS_PER_TOKEN_CONSERVATIVE: int = 2

# Minimum token savings to justify demotion. [AP2-F4]
# When demoting a memory would save fewer tokens than this,
# skip the demotion and prefer dropping instead. This avoids
# the overhead of MEDIUM->LOW demotion when content=None
# (savings ~1 token per memory).
MIN_DEMOTION_SAVINGS_TOKENS: int = 5
```

---

## 13. Error Handling Strategy

| Function | Error | Behavior |
|----------|-------|----------|
| `RecallConfig.__post_init__` | Invalid field values | Raises ValueError |
| `should_recall` | turn_number < 0 | Raises ValueError |
| `should_recall` | non-string message | Raises TypeError |
| `adaptive_k` | scores not descending | Raises ValueError |
| `adaptive_k` | NaN in scores | Raises ValueError |
| `assign_tiers` | k out of range | Raises ValueError |
| `assign_tiers` | scores not descending | Raises ValueError |
| `format_memory` | content with newlines | Replaces with spaces (not an error) |
| `budget_constrain` | mismatched list lengths | Raises ValueError |
| `recall` | mismatched contents length | Raises ValueError |

The module follows the same pattern as engine.py and encoding.py: validate inputs
eagerly with ValueError/TypeError, never silently return wrong results.

---

## 14. Test Categories

### Unit Tests for should_recall (~20)
- First turn always recalls (various messages including empty)
- Short messages rejected (below gate_min_length)
- Trivial patterns rejected (each pattern in default list)
- Case insensitivity of trivial patterns
- Trailing punctuation stripped before trivial match
- Question mark overrides length check
- Question mark overrides trivial pattern check
- Non-trivial messages pass
- Custom config with different patterns
- Custom config with gate_min_length=0 (no length gating)
- Empty gate_trivial_patterns (no pattern gating)
- ValueError on negative turn_number
- TypeError on non-string message
- Known false negative: "tell me" (7 chars) blocked by length check [AP3-E5]
- Known false negative: "help me" (7 chars) blocked by length check [AP3-E5]
- Boundary: "tell me X" (9 chars) correctly passes [AP3-E5]

### Unit Tests for adaptive_k (~20)
- Empty scores returns 0
- Single score returns 1
- Clear gap selects correct k
- Multiple equal-size gaps: first occurrence wins
- All scores equal: fallback k
- All scores within epsilon: fallback k
- Gap at first position (k_raw = 1 + buffer)
- Gap at last position (k_raw = n-1 + buffer)
- min_k enforcement (k_raw < min_k, list long enough)
- max_k enforcement (k_raw > max_k)
- min_k > len(scores): clamped to len
- gap_buffer=0: exact gap position
- Large gap_buffer pushing k_raw past len: clamped
- Scores with novelty bonus (> 1.0): works correctly
- Scores not descending: ValueError
- NaN in scores: ValueError
- Hypothesis: for any descending scores, result in [0, len(scores)]
- Epsilon calibration: with tuned weights, epsilon fallback rate < 5% over 500 random pools [AP3-E6]
- Adaptive-k captures >= fixed-k relevant memories for any score distribution [AP3-E2]
- Top-slice truncation: 10000 uniformly-spaced scores produce k=3 (fallback), not k=max_k [AP2-F3]
- Top-slice truncation: 10000 scores with clear top-5 cluster produce k based on cluster gap [AP2-F3]
- Top-slice correctness: result identical for len(scores)=13 vs len(scores)=10000 when top 13 are same [AP2-F3]

### Unit Tests for assign_tiers (~15)
- k=0 returns empty
- k=1 returns single HIGH assignment
- k=2 with large gap: one HIGH, one LOW
- k=3 with even spread: HIGH, MEDIUM, LOW
- All identical scores, k=1: single HIGH (unchanged)
- All identical scores, k>1: positional fallback (HIGH/MEDIUM/LOW distribution) [AP1-F3]
- All identical scores, k=6: first 2 HIGH, next 2 MEDIUM, last 2 LOW [AP1-F3]
- Boundary value: normalized score exactly at high_threshold -> MEDIUM [AP1-F7]
- Boundary value: normalized score exactly at mid_threshold -> LOW [AP1-F7]
- Boundary value: normalized score at high_threshold + 1e-10 -> HIGH [AP1-F7]
- Custom thresholds change tier boundaries
- Normalized scores are in [0, 1]
- Output order matches input order
- k > len(scores): ValueError
- k < 0: ValueError
- Scores not descending: ValueError
- Hypothesis: for any valid input, output tiers are monotonic with score
- Normalization amplification: scores [0.550, 0.548, 0.546] produce tiers [HIGH, MEDIUM, LOW] [AP2-F6]

### Unit Tests for format_memory (~15)
- HIGH tier with content: includes content text
- HIGH tier without content: metadata only
- MEDIUM tier with content: truncated content
- MEDIUM tier without content: metadata only
- LOW tier: metadata only (content ignored)
- Content truncation at word boundary
- Content with newlines: replaced with spaces
- Very long content: correctly truncated
- Empty content string: no content line
- Formatting values match MemoryState fields
- All three tiers produce different output for same memory

### Unit Tests for budget_constrain (~20)
- Under budget: unchanged
- Over budget, demotion sufficient: memories demoted, none dropped
- Over budget, demotion and drop needed: demoted then dropped
- Over budget, all LOW and still over: dropped
- Single memory, over budget: demoted to LOW, never dropped (min_k)
- Empty input: ([], [], 0, 0)
- Budget = 1: everything demoted and dropped to min_k
- Demotion order: lowest-scored first
- Drop order: lowest-scored first
- Parallel list length mismatch: ValueError
- With contents: re-formatted strings reflect new tier
- Without contents: demoted strings use metadata-only format
- contents=None vs contents=[None]*k produce identical behavior [AP1-F4]
- Mixed contents: [None, "hello", None] demotes correctly [AP1-F8]
- min_k conflict: min_k=3, total_budget=10, result has budget_overflow > 0 [AP1-F1]
- Per-tier budget enforcement: 5 HIGH exceeding high_budget_share triggers demotion [AP1-F10]
- Per-tier budget: demotion occurs even when total budget not exceeded [AP1-F10]
- Demotion without content: verify minimal savings (metadata-only ~9 tokens) [AP1-F4]
- Hypothesis: output token estimate <= budget (or min_k prevents it)
- Hypothesis: budget_overflow == max(0, total_tokens - total_budget) [AP1-F1]
- Content-None optimization: demotion pass skipped when all contents are None [AP3-T3]
- Content-None vs content-provided: dropping-only (None) vs demote-then-drop (provided) [AP3-T3]
- Cascading demotion: 10 HIGH memories -> 4H/4M/2L after per-tier passes [AP2-F2]
- Demotion skip: MEDIUM->LOW with content=None skipped (saves ~1 token) [AP2-F4]
- MIN_DEMOTION_SAVINGS_TOKENS threshold: demotion skipped when savings < 5 tokens [AP2-F4]
- Per-tier demotion despite total budget underutilized: 5H/3M/2L at 677tok triggers HIGH demotion [AP2-F7]
- chars_per_token=2: budget constraint activates sooner for same content [AP2-F1]

### Unit Tests for recall (~20)
- Gated message: returns empty RecallResult with gated=True
- Non-gated, empty memories: returns k=0
- Non-gated, single memory: returns k=1, HIGH tier
- Non-gated, multiple memories with clear gap: correct k and tier distribution
- Non-gated, multiple memories within budget: no demotions/drops
- Non-gated, multiple memories over budget: demotions/drops reflected
- Contents provided: HIGH/MEDIUM memories include content
- Contents not provided: metadata-only rendering
- Contents length mismatch: ValueError
- Turn 0 with trivial message: recalls anyway (first turn override)
- Empty memories, non-trivial message: gated=False, k=0 (fast-path) [AP1-F5]
- Empty memories, trivial message: gated=False, k=0 (fast-path beats gate) [AP1-F5]
- budget_overflow field correct when min_k forces over-budget [AP1-F1]
- All identical scores with content: positional tier assignment distributes budget [AP1-F3]
- Context string has correct separator count
- total_tokens_estimate matches context length (using config.chars_per_token)
- Hypothesis: for any valid inputs, result invariants hold
- Hypothesis: budget_overflow == max(0, total_tokens_estimate - config.total_budget) [AP1-F1]

### Integration Tests (~10)
- Full pipeline with realistic MemoryState list (10+ memories)
- Pipeline with optimizer's tuned parameters (from PINNED + known good weights)
- Verify that higher-relevance memories get higher tiers
- Verify that budget constraint actually limits output size
- Round-trip: rank_memories -> adaptive_k -> assign_tiers -> format -> budget_constrain
- Custom RecallConfig with tight budget: forces aggressive demotion
- Custom RecallConfig with high epsilon: triggers fallback k more often
- min_k vs budget conflict: verify budget_overflow reported correctly [AP1-F1]
- All-identical-score memories: verify positional tier distribution [AP1-F3]
- Production metadata-only: one memory has content=None among content-bearing peers [AP1-F8]
- Epsilon calibration regression: known parameter set produces < 5% fallback over 500 pools [AP3-E6]
- Budget-exercising: tight budget (total_budget=100) triggers full cascade [AP2-F5]
- Large pool: 10000 memories with adaptive_k top-slice truncation [AP2-F3]
- Cascading demotion end-to-end: 10 HIGH memories -> correct final distribution [AP2-F2]
- Non-English budget: chars_per_token=2 with CJK content stays within budget [AP2-F1]

### Property Tests (~10, via Hypothesis)
- For any valid RecallConfig, RecallConfig() does not raise
- For any descending score list, adaptive_k returns valid k
- For any valid assign_tiers input, output length == k
- For any formatted memories, budget_constrain output estimate <= budget (modulo min_k)
- For any non-gated recall, tier_assignments is non-empty when memories is non-empty
- Tier assignment is monotonic with score
- RecallResult invariants hold for arbitrary valid inputs
- Budget monotonicity: for B1 > B2, indices at B2 are subset of indices at B1 [AP3-E4]
- Budget monotonicity: for B1 > B2, shared memories have equal or higher tier at B1 [AP3-E4]

### Slow Tests (~5, marked @pytest.mark.slow)
- Pipeline with 1000 memories: correct k, reasonable latency
- Pipeline with 10000 memories: top-slice truncation does not change k vs exhaustive scan for same top-13 [AP2-F3]
- Pipeline with all possible RecallConfig edge values
- Budget constraint with 100 HIGH-tier memories: demotion cascade
- Full integration with simulated conversation (10 turns, varying messages)
- Regression: known parameter set produces expected k and tier distribution

---

## 15. Open Decisions

### OD1: Normalization Scope for Tier Assignment
Current decision: normalize within the selected k memories. Alternative: normalize
across ALL memories. The within-k approach means tier assignment is independent of
the memories that were cut by adaptive-k. The across-all approach would make tier
assignment sensitive to the full score distribution.

**Decision: Within-k.** Rationale: memories below the adaptive-k cutoff are
irrelevant to presentation. Normalizing across all would dilute the tier
assignment when there are many low-scoring memories.

### OD2: Token Estimation Method
Current decision: `chars // config.chars_per_token` (default 4). Alternative:
use tiktoken or similar tokenizer library. The heuristic is sufficient for a
budget that is itself approximate (800 tokens is a soft target, not a hard
API limit). The configurable `chars_per_token` field [AP2-F1] allows
non-English deployments to use a conservative ratio (2) without importing
a tokenizer.

**Decision: configurable heuristic.** Revisit if exact token counting becomes
necessary for hard budget enforcement.

### OD3: Memory Separator in Context
Current decision: `"\n---\n"`. Alternative: XML tags (`<memory>...</memory>`) or
numbered list. The separator should be parseable by the LLM but not confused with
memory content.

**Decision: `"\n---\n"`.** Matches common markdown convention. Can revisit based
on LLM behavior.

### OD4: Minimum Recall on First Turn
Current decision: always recall on turn 0 regardless of message content. This
ensures the system injects any relevant historical context at session start.
If the memory DB is empty, this is a no-op (k=0 from empty list).

**Decision: Keep.** The cost is one scoring pass on an empty or small memory set.

### OD5: RecallConfig Splitting [AP1-F6]
RecallConfig bundles four concern groups with different change frequencies:
(A) Algorithmic parameters, (B) Budget parameters, (C) Gating parameters,
(D) Presentation parameters. Groups C and D depend on the target LLM model
and may change when switching models, while A and B should be tuned alongside
the optimizer.

**Decision: Keep single dataclass for v1.** The field documentation is organized
into labeled groups (applied in Section 3). Revisit if model-switching becomes
a frequent operation and the coupling causes friction.

### OD6: Per-Tier Budget Enforcement Complexity [AP1-F10]
The per-tier budget enforcement (high_budget_share, medium_budget_share,
low_budget_share) adds a demotion pass before the total-budget demotion pass.
This increases budget_constrain complexity. Alternative: remove the *_budget_share
fields entirely and use total_budget only.

**Decision: Keep per-tier enforcement.** Without it, a scenario with many HIGH
memories consumes the entire budget, leaving no room for MEDIUM/LOW context.
The per-tier shares prevent tier starvation. The implementation complexity is
manageable (one additional loop).

### OD7: Fixed-k Fallback [AP3-T1]
The adaptive-k + tiers + budget system is ~60% more complex than a simpler
fixed k=5 + truncate-to-budget approach. The simpler approach eliminates epsilon
calibration, threshold parameters, per-tier budget shares, and the demotion
cascade. It loses the ability to show 5 memories at mixed detail levels
(demotion), instead showing 3 memories at uniform detail when budget is tight.

**Decision: Keep adaptive-k for v1.** Demotion (showing more memories at reduced
detail) is a concrete capability that the simpler system cannot replicate. However,
if epsilon calibration (AP3-E6) becomes unmanageable or the tier system proves
too complex to maintain, the fixed-k fallback is a viable simplification path.

### OD8: Quantile-Based Tier Assignment [AP3-T2]
Instead of threshold-based tier assignment (normalized_score > high_threshold =
HIGH), use positional quantile assignment: top ceil(k/3) = HIGH, next ceil(k/3)
= MEDIUM, rest = LOW. This eliminates high_threshold and mid_threshold parameters,
removes boundary ambiguity (AP1-F7), and unifies the normal path with the
identical-score fallback (AP1-F3) which already uses positional assignment.

The tradeoff: quantile-based always forces the bottom third to LOW even when all
memories score very well. Threshold-based can assign all to HIGH when scores are
uniformly high (though AP1-F3 already limits this for identical scores).

**Decision: Keep threshold-based for v1.** Changing tier assignment affects 15+
test cases and the approach should be validated empirically post-implementation.
The quantile alternative is documented here for post-v1 evaluation.

### OD9: Token Estimation Bias [AP3-T4]
The chars//4 heuristic consistently overestimates token count by ~20-40% for
typical memory renderings. This means the effective budget is ~60-80% of the
configured total_budget. Alternative: use words * 1.3, which is more accurate
for English prose but underestimates for non-word text (JSON, paths).

**Decision: Keep chars//4.** Overestimation is the conservative direction for
a budget constraint. The bias is predictable and can be compensated by setting
a higher budget. If exact token counting is needed, add tiktoken as an optional
dependency with chars//4 as fallback.

---

## 16. Integration Points

### Upstream (recall.py consumes)
- `engine.rank_memories(params, memories, current_time)` -- scored+ranked memory list
- `engine.score_memory(params, memory, current_time)` -- individual scoring (used transitively)
- `engine.ParameterSet` -- parameter container
- `engine.MemoryState` -- memory snapshot
- `optimizer.PINNED` -- default parameter values (for constructing default ParameterSet)

### Downstream (recall.py produces for)
- `PromptAssembler.build()` -- receives `RecallResult.context` for injection into system prompt
- Logging/observability -- receives `RecallResult` metadata (k, tiers, budget_exceeded, etc.)
- `AIAgent.run_conversation()` -- calls `should_recall()` to decide whether to invoke the pipeline

### Consumer Contract for Budget Overflow [AP1-F1]
The `budget_overflow` field in RecallResult may be non-zero when min_k prevents
further dropping. Consumers (primarily PromptAssembler) MUST handle this:
- If `budget_overflow > 0`, the context is larger than the configured budget.
- The consumer should either truncate the context string or adjust other prompt
  sections to accommodate the overflow.
- Ignoring budget_overflow risks exceeding the model's context window.

### Wiring (not in this module, but documented for context)
```python
# In agent/prompt_assembler.py (future integration):
if recall_result and not recall_result.gated:
    if recall_result.budget_overflow > 0:
        # Truncate context to fit, or reduce other prompt sections
        max_chars = (recall_result.total_tokens_estimate - recall_result.budget_overflow) * config.chars_per_token
        context = recall_result.context[:max(0, max_chars)]
    else:
        context = recall_result.context
    prompt_parts.append(f"## Recalled Memories\n{context}")

# In agent/ai_agent.py (future integration):
config = RecallConfig()
if should_recall(user_message, turn_number, config):
    # ... fetch memories, compute relevances, call recall()
    result = recall(memories, params, config, user_message, turn_number, contents)
```

---

## 17. Relationship to Formal Verification

The recall module is a **presentation layer** that sits above the formally verified
scoring/dynamics engine. It does NOT modify any of the verified properties:

| Verified Property | Recall Impact |
|-------------------|---------------|
| Score boundedness (score in [0, 1+N0]) | Recall reads scores, does not compute them |
| Rank ordering determinism | Recall consumes rank_memories output unchanged |
| Contraction condition (K < 1) | Recall does not modify ParameterSet |
| Soft selection monotonicity | Recall does not use soft_select (that's for dynamics) |
| Novelty bonus decay | Recall benefits from it (new memories score higher) but doesn't modify it |

The adaptive-k algorithm, tier assignment, and budget constraint are all
**post-hoc presentation decisions** that cannot violate the mathematical
invariants of the underlying system. They are heuristics, not proven, and
are validated empirically through the test suite.

---

## 18. Adversarial Pass 1 -- Level-Split + Paradox-Hunt

Applied: 2026-02-27
Author: architect-agent (Opus 4.6)
Full findings: `thoughts/shared/plans/memory-system/recall/adversarial-pass1.md`

### Operators

- **Level-Split**: Where does the spec conflate two things that should be separated?
- **Paradox-Hunt**: Where do two parts of the spec contradict?

### Findings Summary

| # | Finding | Severity | Operator | Sections Modified |
|---|---------|----------|----------|-------------------|
| 1 | min_k vs budget invariant: priority hierarchy undefined; consumer cannot handle overflow | CRITICAL | Paradox-Hunt | 3, 5, 10, 11, 14, 16 |
| 2 | Score semantic spaces: three meanings of "score" conflated across engine/recall boundary | HIGH | Level-Split | 3, 4, 7 |
| 3 | All-identical tier assignment: all-HIGH wastes budget; no positional fallback | HIGH | Paradox-Hunt | 8, 14 |
| 4 | content=None double semantics: demotion with no content barely saves tokens; double-None ambiguity | HIGH | Paradox-Hunt | 9, 10, 14 |
| 5 | Gate vs empty-list ordering: ambiguous k=0 signal; wasted gate evaluation on empty DB | MEDIUM | Paradox-Hunt | 5, 11, 14 |
| 6 | Config vs presentation params: different change frequencies bundled in one dataclass | MEDIUM | Level-Split | 3, 15 |
| 7 | Threshold boundary ambiguity: "high_threshold" does not include the threshold value | MEDIUM | Paradox-Hunt | 3, 8, 14 |
| 8 | Test vs production content=None: two scenarios share one path with different semantics | MEDIUM | Level-Split | 9, 14 |
| 9 | Epsilon fallback includes gap_buffer: unexplained design choice in flat-score case | LOW | Paradox-Hunt | 7 |
| 10 | budget_share fields unused in budget_constrain: dead configuration with no behavioral effect | LOW | Level-Split | 3, 10, 15 |

### Mitigations Applied

**Finding 1 (CRITICAL):** Added `budget_overflow: int` field to RecallResult. Defined
priority hierarchy (min_k > per-tier budget > total budget). Updated budget_constrain
invariants to acknowledge overflow. Added consumer contract in Section 16 requiring
callers to handle overflow. Added test cases for min_k vs budget conflict.

**Finding 2 (HIGH):** Labeled TierAssignment.score as "engine-space composite score"
and normalized_score as "within-k normalized presentation score." Added epsilon
sensitivity subsection to Section 7 documenting coupling to engine score magnitude.
Added epsilon note in RecallConfig field definition.

**Finding 3 (HIGH):** Replaced "all get HIGH" with positional fallback when all k
scores are identical and k > 1: first ceil(k/3) get HIGH, next ceil(k/3) get
MEDIUM, rest get LOW. Updated edge case tables and test cases.

**Finding 4 (HIGH):** Documented double-None semantics. Added normalization step in
budget_constrain algorithm: `contents=None` is normalized to `[None] * len(assignments)`
before entering demotion loop. Added note about minimal demotion savings without
content. Added test cases for both None forms.

**Finding 5 (MEDIUM):** Added empty-memories fast-path before gate check in recall
algorithm. Documented that `gated=False, k=0` means "empty DB" while
`gated=True, k=0` means "gate decided not to recall." Updated edge case table.

**Finding 6 (MEDIUM):** Organized RecallConfig fields into labeled concern groups
(A: Algorithmic, B: Budget, C: Gating, D: Presentation). Added OD5 recording
potential future split.

**Finding 7 (MEDIUM):** Updated threshold field docs to state "STRICTLY ABOVE."
Added boundary-value edge case table to Section 8. Added boundary-value test cases.

**Finding 8 (MEDIUM):** Documented dual-purpose content=None semantics (test
convenience vs production degradation). Added warning about future sentinel type
if scenarios need distinguishing. Added production metadata-only test case.

**Finding 9 (LOW):** Added rationale note in Section 7 explaining gap_buffer in
the flat-score fallback as a "benefit of the doubt" factor biasing toward recall.

**Finding 10 (LOW):** Implemented per-tier budget enforcement using budget_share
fields. Added per-tier demotion pass to budget_constrain algorithm. Added OD6
recording the complexity tradeoff.

---

## 19. Adversarial Pass 3 -- Exclusion-Test + Object-Transpose

Applied: 2026-02-27
Author: architect-agent (Opus 4.6)
Full findings: `thoughts/shared/plans/memory-system/recall/adversarial-pass3.md`

### Operators

- **Exclusion-Test (Scissors)**: Design lethal tests that would delete the entire
  approach if they fail. Not supportive experiments.
- **Object-Transpose (Perpendicular)**: Identify cheaper representations that make
  the decisive test trivial. Treat the current design as a variable.

### Findings Summary

| # | Finding | Severity | Operator | Sections Modified |
|---|---------|----------|----------|-------------------|
| E1 | Temperature coupling claim is FALSE: temperature has zero effect on score_memory | CRITICAL | Scissors | 3, 7 |
| E2 | Adaptive-k beats fixed-k only by recall (mean k=6.3 vs 3), never precision | HIGH | Scissors | 7 |
| E3 | Highest-scored memory always gets HIGH tier (confirmed, no action) | INFO | Scissors | -- |
| E4 | Budget monotonicity holds for both drops and tiers | MEDIUM | Scissors | 14 |
| E5 | Gating false negatives: short imperatives (5-7 chars) blocked by length check | MEDIUM | Scissors | 6, 14 |
| E6 | Epsilon 0.01 has near-zero headroom (min observed max_gap = 0.0102) | MEDIUM | Scissors | 3, 7, 14 |
| T1 | Fixed k=5 + truncate is 80% as good, 60% less complex | HIGH | Perpendicular | 15 (OD7) |
| T2 | Quantile-based tiers equivalent for k<=10, eliminates 2 parameters | MEDIUM | Perpendicular | 15 (OD8) |
| T3 | Drop-from-bottom mostly sufficient; demotion marginal for content=None | MEDIUM | Perpendicular | 10, 14 |
| T4 | chars//4 better than words*1.3 (conservative overestimation) | LOW | Perpendicular | 12, 15 (OD9) |

### Mitigations Applied

**Finding E1 (CRITICAL):** Removed "Interaction with Temperature" subsection from
Section 7. Replaced with "Interaction with Scoring Weights" documenting that w1-w4
control score spread. Updated epsilon field comments in Section 3 to reference
weights, not temperature. Corrected score range from ~[0, 1.3] to ~[0, 1.1]
based on empirical measurement.

**Finding E2 (HIGH):** Added "Recall-vs-Precision Tradeoff" subsection to Section 7
documenting that adaptive-k optimizes for recall at the cost of precision. The
tier system and budget constraint handle precision. No design change needed.

**Finding E3 (INFO):** Passed. Highest-scored memory always gets HIGH by
construction (normalized to 1.0 > 0.7 threshold, or first in positional fallback).

**Finding E4 (MEDIUM):** Added budget monotonicity property tests to Section 14:
for B1 > B2, indices at B2 are a subset of B1 indices, and shared memories have
equal or higher tier at B1.

**Finding E5 (MEDIUM):** Added "Known False Negative Category" subsection to
Section 6 documenting short imperative phrases (5-7 chars) as the primary false
negative class. Added test cases for boundary (7-char blocked, 9-char passed).
gate_min_length retained at 8.

**Finding E6 (MEDIUM):** Added calibration fragility warning to Section 7 epsilon
sensitivity subsection. Added epsilon calibration regression test to Section 14.
Recommended future optimizer runs include epsilon-fallback-rate constraint.

**Finding T1 (HIGH):** Added OD7 to Section 15 documenting the fixed-k fallback
as a viable simplification path if epsilon calibration or tier complexity becomes
unmanageable. No design change for v1.

**Finding T2 (MEDIUM):** Added OD8 to Section 15 documenting quantile-based tier
assignment as a parameter-free alternative. Deferred to post-v1 evaluation.

**Finding T3 (MEDIUM):** Updated Section 10 algorithm with content-None
optimization: when ALL contents are None, demotion passes are skipped and
budget_constrain goes directly to dropping. Added corresponding test cases.

**Finding T4 (LOW):** Added overestimation bias documentation to Section 12
(CHARS_PER_TOKEN comment) and OD9 to Section 15.

### Key Empirical Results

These experiments were run against the actual engine.py scoring pipeline with
tuned parameters (w1=0.4109, w2=0.0500, w3=0.3000, w4=0.2391, temperature=4.9265):

1. **Temperature has zero effect on scores.** `score_memory()` does not use
   temperature. Verified: identical scores for temperature=4.9265 and temperature=10.0.

2. **Score distribution with tuned weights:**
   - Range: [0.145, 1.100]
   - Typical consecutive gaps: 0.03 to 0.12
   - No gap below 0.028 in the 12-memory test case

3. **Epsilon fallback rate:** 0/200 random pools (0%). Minimum max_gap: 0.0102.

4. **Adaptive-k vs fixed-3:** Adaptive captures more relevant memories in 43% of
   trials, never fewer. Mean adaptive k = 6.3.

5. **Token estimation bias:** chars//4 overestimates by 20-40% for metadata strings,
   underestimates for repeated short words. words*1.3 underestimates for non-word text.

---

## 20. Adversarial Pass 2 -- Scale-Check + Materialize

Applied: 2026-02-27
Author: architect-agent (Opus 4.6)
Full findings: `thoughts/shared/plans/memory-system/recall/adversarial-pass2.md`

### Operators

- **Scale-Check**: Are there order-of-magnitude assumptions that haven't been verified?
- **Materialize**: For each claimed behavior, what would I actually SEE if I ran this?

### Findings Summary

| # | Finding | Severity | Operator | Sections Modified |
|---|---------|----------|----------|-------------------|
| 1 | chars//4 underestimates non-English tokens by 60-167% | HIGH | Scale-Check | 3, 5, 10, 11, 12, 14 |
| 2 | Per-tier demotion cascade unspecified; MEDIUM pass must include HIGH demotees | HIGH | Materialize | 10, 14 |
| 3 | adaptive_k degenerates to max_k on large pools; full-list gap scan finds irrelevant gaps | HIGH | Scale-Check | 7, 14 |
| 4 | No-content MEDIUM->LOW demotion saves 1 token; demotion loop is pure overhead | MEDIUM | Materialize | 10, 12, 14 |
| 5 | Budget dramatically underutilized (33% typical); budget system rarely activates with defaults | MEDIUM | Materialize | 10, 14 |
| 6 | Within-k normalization amplifies negligible score differences to full [0,1] range | MEDIUM | Materialize | 8, 14 |
| 7 | Per-tier shares cause demotion even when total budget not exceeded (by design) | LOW | Materialize | 10 |
| 8 | Separator rounding compounds with per-string rounding for ~1% systematic undercount | LOW | Scale-Check | 10 |

### Mitigations Applied

**Finding 1 (HIGH):** Added `chars_per_token: int = 4` field to RecallConfig
(Group D: Presentation). Budget estimation throughout the spec now uses
`config.chars_per_token` instead of hardcoded 4. Added `CHARS_PER_TOKEN_CONSERVATIVE = 2`
constant for non-English deployments. Added accuracy table to Token Estimation
subsection documenting error rates by content type.

**Finding 2 (HIGH):** Rewrote Section 10 Step 4 to explicitly state that the
MEDIUM per-tier pass runs AFTER the HIGH pass and INCLUDES memories demoted
from HIGH. Added worked example of the 10-HIGH cascade (4H/4M/2L final).
Added cascading demotion test case.

**Finding 3 (HIGH):** Modified Section 7 adaptive_k algorithm to truncate
input to `scores[:max_k + gap_buffer + 1]` before gap detection. This
restricts gap scanning to the top portion of the ranking, preventing
degeneration to max_k on large pools where irrelevant gaps deep in the
list are larger than meaningful top-cluster gaps. Added rationale note.
Added test cases for 10,000-memory pools.

**Finding 4 (MEDIUM):** Added `MIN_DEMOTION_SAVINGS_TOKENS = 5` constant.
Section 10 Step 4 now skips demotion when estimated savings are below
this threshold (covers the MEDIUM->LOW with content=None case that saves
only 1 token). Added test case for demotion skip.

**Finding 5 (MEDIUM):** Added "Default Budget Utilization" subsection to
Section 10 with concrete utilization numbers for typical scenarios. Added
tight-budget integration test (total_budget=100) to exercise the budget
cascade with default content sizes.

**Finding 6 (MEDIUM):** Added "Amplification effect" paragraph to Section 8
Normalization Rationale explaining that near-identical raw scores normalize
to full [0, 1] range and why this is acceptable (presentation diversity
over quality judgment).

**Finding 7 (LOW):** Added invariant note to Section 10 documenting that
per-tier demotion fires even when total budget is available, with a
concrete example (5H/3M/2L at 677 tokens).

**Finding 8 (LOW):** Added rounding undercount note to the Token Estimation
subsection. Combined with Finding 1's accuracy table, implementers have
a complete picture of estimation error sources.

### Materialization Traces

Five concrete scenario traces are documented in the full findings file.
Key observations:

1. Scenario 1 (mixed scores): k=6, tiers [H,M,M,M,L,L], 268 tokens (33% budget).
2. Scenario 2 (50 flat scores): epsilon fallback k=3, normalization amplifies 0.004 to [0,1].
3. Scenario 3 (novelty memory): content budget is 313 chars (~77 tokens), adequate for 3-5 sentences.
4. Scenario 4 (gating): "???" triggers recall (question mark override), "k" does not (length < 8).
5. Scenario 5 (5 HIGH): 505 tokens total, but per-tier HIGH budget (440) triggers demotion.

---

**End of spec.**
