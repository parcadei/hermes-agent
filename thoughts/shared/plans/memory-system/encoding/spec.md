# Encoding Policy: Behavioral Specification

Created: 2026-02-27
Author: architect-agent
Status: COMPLETE (adversarial review applied 2026-02-27)

---

## 1. Overview

The `EncodingPolicy` is a deterministic gate that sits between Nemori's segmentation
stage and the episode/semantic storage stage. Nemori's pipeline currently stores
EVERYTHING --- no policy decides whether a piece of information is worth remembering.
The encoding layer introduces a classification step that evaluates each candidate
memory unit and decides:

1. Whether to store it (`should_store`)
2. What category it belongs to (one of 7 categories)
3. How important it is for the downstream dynamics system (`initial_importance`)

The policy is fail-open: when uncertain, it stores rather than discards. This
reflects the asymmetric cost of memory loss (irrecoverable) vs. memory clutter
(manageable via dynamics-driven decay).

### Primary vs. Secondary Value

> **[ADVERSARIAL FINDING 2.1 -- Scale-Check]** The gate's primary value is NOT
> filtering. Nemori's segmentation already collapses transient messages into
> multi-message episodes (episode_min_messages=2), so standalone greetings and
> short transactional commands rarely become episodes. In practice, the gate
> rejects only ~0-5% of episodes.
>
> **The gate's primary value is IMPORTANCE SEEDING and CATEGORY ANNOTATION** for
> the downstream dynamics system. A correction starting at importance 0.9 behaves
> very differently from a fact starting at 0.6 in the scoring function
> `score = w1*rel + w2*R(t,S) + w3*imp + w4*sigma(act)`. Correct classification
> drives correct importance, which drives correct retrieval ranking.
>
> **Secondary value:** Filtering is meaningful for semantic memories (which are
> fine-grained extractions that CAN be trivial) and for degenerate episodes.

### Position in Pipeline

```
Messages --> Buffer --> Segmentation --> Episode Narrative --> [ENCODING GATE] --> Predict-Calibrate --> Semantic Facts
                                              |                     |                                      |
                                              |                Episodic DB                            [ENCODING GATE]
                                              |                                                            |
                                        (gate evaluates                                              Semantic DB
                                         narrative text)
```

> **[ADVERSARIAL FINDING 1.1 -- Level-Split]** The pipeline diagram is corrected:
> the encoding gate evaluates EPISODE NARRATIVES (third-person text produced by
> EpisodeGenerator), not raw messages. It also independently evaluates SEMANTIC
> MEMORIES produced by PredictionCorrectionEngine. These are two separate gate
> invocations with potentially different pattern sets.

The gate evaluates TWO types of content:
- **Episode content**: the narrative text produced by `EpisodeGenerator.generate_episode()`
  -- these are THIRD-PERSON narratives, not raw user messages
- **Semantic facts**: the knowledge extracted by `PredictionCorrectionEngine`
  -- these are also third-person, e.g., "The user lives in San Francisco"

Both pass through the same `EncodingPolicy.evaluate()` method. The gate does not
modify content --- it only annotates and filters.

---

## 2. Data Types and Contracts

### 2.1 EncodingConfig

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class EncodingConfig:
    """Configuration for the encoding gate.

    Frozen to ensure thread safety and reproducibility.
    All thresholds are tunable without code changes.
    """

    # Reasoning category thresholds
    min_reasoning_length: int = 100
    """Minimum character count for reasoning content to pass the gate.
    Below this, reasoning is likely a trivial causal statement
    rather than a substantive chain of inference."""

    min_reasoning_connectives: int = 2
    """Minimum count of causal connectives (because, therefore, since, so,
    consequently, thus, hence, as a result) required for reasoning to pass.
    A single connective is common in transactional speech; two or more
    indicates genuine inferential structure."""

    # Greeting/transactional upgrade thresholds
    max_greeting_length: int = 50
    """Maximum character count for a message to remain classified as greeting.
    Beyond this, the message likely contains substantive content that
    happened to start with a greeting token, and should be reclassified.

    [ADVERSARIAL FINDING 2.3] For episode narratives (source_type="episode"),
    the effective threshold is max_greeting_length + episode_length_offset,
    because Nemori's narrative generation inflates text length by 150-300%
    compared to raw messages."""

    max_transactional_length: int = 80
    """Maximum character count for a message to remain classified as transactional.
    Longer imperative-style messages often contain instructions or
    preferences embedded within commands.

    Same episode_length_offset applies as for greeting threshold."""

    episode_length_offset: int = 150
    """[ADVERSARIAL FINDING 2.3] Added offset for episode narrative length.
    Nemori's EpisodeGenerator produces third-person narratives that are
    typically 2-4x longer than the underlying raw messages. When evaluating
    episode content, this offset is added to max_greeting_length and
    max_transactional_length to compensate for narrative inflation.

    Effective thresholds for episodes:
      greeting: max_greeting_length + episode_length_offset = 200
      transactional: max_transactional_length + episode_length_offset = 230
    """

    # Confidence gate
    confidence_threshold: float = 0.5
    """Below this confidence, the gate defaults to should_store=True (fail-open).
    This ensures that ambiguous content is preserved rather than lost.
    The dynamics layer downstream will handle relevance ranking."""

    # [ADVERSARIAL FINDING 2.2] Word boundary matching
    use_word_boundaries: bool = True
    """When True, pattern matching uses word boundary detection (regex \\b)
    instead of raw substring matching. This prevents false positives like
    "know" matching the correction pattern "no ", or "also" matching the
    reasoning pattern "so ".

    When False, falls back to substring matching (original behavior).
    May be set to False for performance-critical paths where regex overhead
    is unacceptable."""
```

**Invariants:**
- `min_reasoning_length > 0`
- `min_reasoning_connectives > 0`
- `max_greeting_length > 0`
- `max_transactional_length > 0`
- `0.0 <= confidence_threshold <= 1.0`
- `episode_length_offset >= 0`

### 2.2 EncodingDecision

```python
from dataclasses import dataclass

VALID_CATEGORIES = frozenset({
    "preference",
    "fact",
    "correction",
    "reasoning",
    "instruction",
    "greeting",
    "transactional",
})

@dataclass
class EncodingDecision:
    """The output of the encoding gate for a single memory candidate.

    This is NOT frozen because it is a one-shot output, not shared state.
    """

    should_store: bool
    """The gate decision. True means the memory proceeds to storage."""

    category: str
    """One of the 7 valid categories. Always set, even for rejected content."""

    confidence: float
    """Classifier confidence in [0.0, 1.0].

    [ADVERSARIAL FINDING 1.3] This measures PATTERN DENSITY (how many
    patterns in the assigned category matched), NOT classification probability.
    A value of 0.07 means "one pattern matched out of 14 in this category"
    -- the classification may still be correct with high certainty. See
    Section 3.3 for calibration details.

    After calibration (Section 4.4), single high-signal pattern matches
    receive a confidence floor, so "I prefer X" gets confidence ~0.6
    rather than raw 1/14 = 0.07."""

    reason: str
    """Human-readable explanation of the decision. Useful for debugging
    and for future LLM-based review of gate behavior."""

    initial_importance: float
    """Importance seed for the dynamics layer, in [0.0, 1.0].
    Maps directly from category to a fixed value (see Section 2.3).
    This seeds the importance component of the scoring function:
        score = w1*relevance + w2*R(t,S) + w3*importance + w4*sigmoid(activation)
    """
```

**Invariants:**
- `confidence in [0.0, 1.0]`
- `initial_importance in [0.0, 1.0]`
- `category in VALID_CATEGORIES`
- If `should_store is True` and `category in {"greeting", "transactional"}`, this can only happen via fail-open (confidence < threshold)
- If `should_store is False`, then `category in {"greeting", "transactional"}` OR (category is "reasoning" and the conditional check failed)

### 2.3 Category-to-Importance Mapping

The `initial_importance` field seeds the dynamics system's importance component.
Higher values mean the memory starts with stronger relevance weight in the
scoring function `score = w1*rel + w2*R(t,S) + w3*imp + w4*sigma(act)`.

| Category      | initial_importance | Rationale |
|---------------|-------------------|-----------|
| correction    | 0.9               | Highest. Corrections indicate the system made an error. Retaining corrections prevents repeated mistakes. |
| instruction   | 0.85              | High. Direct behavioral directives from the user. Should persist across sessions. |
| preference    | 0.8               | High. User preferences are strong personalization signals. Stable over time. |
| fact          | 0.6               | Medium. Personal facts are useful but may become stale (moved cities, changed jobs). Dynamics decay handles staleness. |
| reasoning     | 0.4               | Low. Reasoning chains are context-dependent. Valuable in the moment but their relevance decays faster than facts. |
| greeting      | 0.0               | Never stored (but set for completeness). Greetings carry no information worth persisting. |
| transactional | 0.0               | Never stored (but set for completeness). Imperative commands are session-scoped actions. |

```python
CATEGORY_IMPORTANCE: dict[str, float] = {
    "correction":    0.9,
    "instruction":   0.85,
    "preference":    0.8,
    "fact":          0.6,
    "reasoning":     0.4,
    "greeting":      0.0,
    "transactional": 0.0,
}
```

This mapping is a constant. To change importance seeds, modify the mapping, not
the classifier. This separation ensures the classification logic and the
importance semantics evolve independently.

### 2.4 Connection to Dynamics

The `initial_importance` value feeds into the Hermes dynamics system at memory
creation time:

```
Memory created --> initial_importance = CATEGORY_IMPORTANCE[category]
                --> importance_update(imp, delta, signal) adjusts over time
                --> score(w, rel, rec, imp, act) uses current importance for ranking
```

Specifically, `importance_update` from `hermes_memory.core`:
```
imp' = clamp01(imp + delta * signal)
```

The `initial_importance` is the starting value of `imp`. The dynamics system
then adjusts it based on retrieval feedback (`signal`). A correction starting
at 0.9 that is never retrieved will drift toward 0.0 via negative feedback.
A fact starting at 0.6 that is frequently retrieved will climb toward 1.0.

The contraction guarantee from `MEMORY_SYSTEM.md` Section 4.4 ensures this
feedback loop converges to a stable equilibrium rather than oscillating.

---

## 3. Classification Categories and Heuristics

### 3.1 Pattern Sets

Each category has a set of indicator patterns. The classifier checks for these
patterns using case-insensitive matching on the input text.

> **[ADVERSARIAL FINDING 2.2 -- Scale-Check]** When `config.use_word_boundaries`
> is True (default), patterns use word-boundary-aware matching (regex `\b`) rather
> than raw substring matching. This prevents false positives like "know" matching
> the correction pattern "no", or "also" matching "so".

> **[ADVERSARIAL FINDING 1.2 -- Level-Split]** The "Policy:" annotations below
> are references to the WRITE POLICY in Section 4, not part of the classification
> logic. The `classify()` method assigns categories; the `_apply_write_policy()`
> method decides storage. These are separate concerns.

#### preference

Indicators of user taste, aesthetic judgment, or personal inclination.

```python
PREFERENCE_PATTERNS = [
    "i like",
    "i prefer",
    "i want",
    "my favorite",
    "i enjoy",
    "i don't like",
    "i dont like",
    "i do not like",
    "i hate",
    "i love",
    "i dislike",
    "i'd rather",
    "id rather",
    "i would rather",
    "i tend to",
]
```

**Write policy reference:** ALWAYS store (Section 4.1).

#### fact

Indicators of personal identity, biography, or circumstantial state.

```python
FACT_PATTERNS = [
    "i am",
    "i'm",
    "im ",
    "i work at",
    "i work as",
    "i live in",
    "i live at",
    "my name is",
    "i have a",
    "i have an",
    "my birthday",
    "my age is",
    "i was born",
    "i studied",
    "i graduated",
    "my email",
    "my phone",
    "my address",
    "i speak",
    "my job",
    "my role is",
]
```

**Write policy reference:** ALWAYS store (Section 4.1).

#### correction

Indicators that the user is correcting a previous system output or clarifying
a misunderstanding.

```python
CORRECTION_PATTERNS = [
    "no,",
    "no ",
    "actually,",
    "actually ",
    "that's wrong",
    "thats wrong",
    "that is wrong",
    "i meant",
    "correction:",
    "not quite",
    "that's incorrect",
    "thats incorrect",
    "that is incorrect",
    "you're wrong",
    "youre wrong",
    "you are wrong",
    "i didn't mean",
    "i didnt mean",
    "what i meant was",
    "let me correct",
    "to clarify",
]
```

**Write policy reference:** ALWAYS store (Section 4.1).

**Note on "no" patterns:** The patterns `"no,"` and `"no "` are deliberately
specific. A bare `"no"` would match "I know", "another", etc. The comma or
space after "no" strongly indicates a correction or negation in conversational
context. False positives (e.g., "No problem") are acceptable because the
fail-open philosophy prefers over-storage to information loss.

> **[ADVERSARIAL FINDING 2.2 -- Scale-Check]** Even with the trailing space/comma,
> `"no "` still matches as a substring in words like "k**no** w" when followed by
> a space in natural text ("I know the answer" does NOT match, but "I have no idea"
> DOES match). With `use_word_boundaries=True`, the pattern becomes `\bno\b` which
> matches only standalone "no", eliminating these false positives while preserving
> matches on "No, that's wrong".

#### instruction

Indicators that the user is giving a persistent behavioral directive.

```python
INSTRUCTION_PATTERNS = [
    "always",
    "never",
    "remember to",
    "remember that",
    "from now on",
    "please always",
    "please never",
    "don't ever",
    "dont ever",
    "do not ever",
    "make sure to",
    "keep in mind",
    "going forward",
    "in the future",
]
```

**Write policy reference:** ALWAYS store (Section 4.1).

#### reasoning

Indicators of causal or inferential structure. Unlike other categories,
reasoning requires BOTH pattern matching AND length/density thresholds.

```python
REASONING_CONNECTIVES = [
    "because",
    "therefore",
    "since",
    "so ",
    "consequently",
    "thus",
    "hence",
    "as a result",
    "due to",
    "in order to",
    "which means",
    "this implies",
    "it follows",
    "given that",
]
```

**Write policy reference:** CONDITIONAL (Section 4.1) --- store only if:
1. `len(text) >= config.min_reasoning_length` (default 100)
2. `connective_count >= config.min_reasoning_connectives` (default 2)

Both conditions must hold. A short sentence with "because" is likely
transactional ("Do X because Y"), not substantive reasoning.

#### greeting

Indicators of social protocol with no informational content.

```python
GREETING_PATTERNS = [
    "hello",
    "hi ",
    "hi,",
    "hey",
    "thanks",
    "thank you",
    "goodbye",
    "bye",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "how are you",
    "nice to meet",
    "cheers",
]
```

**Write policy reference:** NEVER store (Section 4.1) --- unless the text exceeds
the effective greeting length threshold, in which case the greeting classification
is suspect and the text is reclassified by re-running the classifier with greeting
patterns excluded.

#### transactional

Indicators of imperative commands scoped to the current session.

```python
TRANSACTIONAL_PATTERNS = [
    "run ",
    "execute",
    "open ",
    "show me",
    "display",
    "list ",
    "print ",
    "delete ",
    "create ",
    "generate ",
    "compile",
    "build ",
    "test ",
    "deploy",
    "start ",
    "stop ",
    "restart",
]
```

**Write policy reference:** NEVER store (Section 4.1) --- unless the text exceeds
the effective transactional length threshold, in which case the transactional
classification is suspect and the text is reclassified with transactional patterns
excluded.

### 3.2 Classification Priority

When multiple pattern sets match, the classifier assigns the HIGHEST-PRIORITY
category. Priority order (highest first):

1. **correction** --- corrections override everything; if the user is correcting,
   that is the primary semantic content regardless of other patterns present.
2. **instruction** --- behavioral directives are second priority.
3. **preference** --- preferences are strong signals.
4. **fact** --- personal facts.
5. **reasoning** --- conditional storage, lower priority.
6. **transactional** --- session-scoped commands.
7. **greeting** --- lowest priority; only assigned if nothing else matches.

Rationale: A message like "No, I prefer dark mode --- remember that going forward"
matches correction ("No,"), preference ("I prefer"), and instruction ("going forward").
The correct classification is **correction** because the correction is the primary
communicative act; the preference and instruction are subordinate content that the
correction happens to contain.

### 3.3 Confidence Scoring

> **[ADVERSARIAL FINDING 1.3 + 2.4 -- Level-Split + Scale-Check]** The original
> confidence formula (`matched_pattern_count / total_patterns_in_category`) produces
> values of 0.05-0.15 for typical single-pattern matches, causing fail-open to
> trigger on >95% of inputs. The revised formula uses CALIBRATED CONFIDENCE with
> a floor for high-signal patterns.

Confidence is computed in two stages:

**Stage 1 -- Raw pattern density:**
```
raw_density = matched_pattern_count / total_patterns_in_category
```

**Stage 2 -- Calibrated confidence:**
```
if matched_pattern_count >= 1:
    confidence = max(SINGLE_PATTERN_CONFIDENCE_FLOOR, raw_density)
    # Each additional pattern adds diminishing bonus
    confidence = min(1.0, confidence + 0.08 * (matched_pattern_count - 1))
else:
    confidence = raw_density  # 0.0
```

Where `SINGLE_PATTERN_CONFIDENCE_FLOOR = 0.35`. This reflects that a single
clear pattern match (e.g., "I prefer" matching preference) is a moderately
strong signal, not a near-zero signal.

**Rationale for 0.35 floor:** Below the fail-open threshold of 0.5, so a single
ambiguous match still triggers fail-open. But two pattern matches yield
`0.35 + 0.08 = 0.43`, and three yield `0.35 + 0.16 = 0.51` (above threshold).
This means:
- 1 pattern match: confidence 0.35, below threshold, fail-open stores if rejected
- 2 pattern matches: confidence 0.43, below threshold, fail-open stores if rejected
- 3+ pattern matches: confidence 0.51+, above threshold, gate decision is final

For ALWAYS-store categories (preference, fact, correction, instruction), confidence
does not affect the storage decision. For NEVER-store categories (greeting,
transactional), 3+ pattern matches are needed to confidently reject.

Additional boosters:
- If `metadata` is provided and `metadata.get("message_count", 0) > 5`, add 0.1
  to confidence (capped at 1.0). Episodes with many messages are less likely to
  be greetings.
- If the text length exceeds 500 characters, add 0.05 to confidence. Longer
  texts provide more signal for classification.

The confidence value affects the fail-open gate but does NOT affect which category
is assigned. Category assignment is based on priority order among matched categories.

### 3.4 Dual Pattern Sets for Narrative Text

> **[ADVERSARIAL FINDING 1.1 -- Level-Split, CRITICAL]** Nemori's EpisodeGenerator
> produces THIRD-PERSON narratives, not first-person raw messages. The patterns in
> Section 3.1 are written for first-person text ("I like", "I live in"). They will
> NOT match third-person narratives ("The user expressed a preference for",
> "The user mentioned they reside in").

The classifier maintains TWO sets of patterns for each category: a first-person
set (Section 3.1, for raw messages or direct quotes in narratives) and a
third-person set (below, for episode narratives).

**Selection logic:**
```python
if metadata and metadata.get("source_type") == "episode":
    pattern_sets = THIRD_PERSON_PATTERNS
elif metadata and metadata.get("source_type") == "semantic":
    # Semantic memories use the shortcut (Section 5.6) when knowledge_type available
    # Fall back to third-person patterns when no knowledge_type
    pattern_sets = THIRD_PERSON_PATTERNS
else:
    # Raw messages or unknown source
    pattern_sets = FIRST_PERSON_PATTERNS  # Section 3.1
```

**Both sets are checked** (first-person then third-person) and matches are
unioned. This handles the case where a narrative contains direct quotes:
"The user said 'I prefer dark mode'."

#### Third-Person Pattern Sets

```python
THIRD_PERSON_PREFERENCE_PATTERNS = [
    "the user prefer",
    "the user expressed a preference",
    "the user indicated they prefer",
    "the user likes",
    "the user enjoys",
    "the user dislikes",
    "the user wants",
    "the user would rather",
    "the user favors",
    "the user's preference",
    "preferred",
    "preference for",
    "expressed interest in",
    "stated a preference",
]

THIRD_PERSON_FACT_PATTERNS = [
    "the user is",
    "the user works",
    "the user lives",
    "the user resides",
    "the user's name",
    "the user mentioned",
    "the user shared",
    "the user disclosed",
    "the user stated that they",
    "the user indicated that they",
    "personal detail",
    "personal information",
    "background information",
    "biographical",
]

THIRD_PERSON_CORRECTION_PATTERNS = [
    "the user corrected",
    "the user clarified",
    "the user pointed out",
    "correction",
    "corrected the",
    "clarified that",
    "disagreed with",
    "the user disagreed",
    "mistaken",
    "inaccurate",
    "the user noted the error",
    "amended",
    "revised",
]

THIRD_PERSON_INSTRUCTION_PATTERNS = [
    "the user instructed",
    "the user requested that",
    "the user asked the assistant to always",
    "the user asked the assistant to never",
    "going forward",
    "from now on",
    "the user emphasized",
    "the user directed",
    "the user specified",
    "persistent instruction",
    "behavioral directive",
    "the user wants the assistant to",
]

THIRD_PERSON_REASONING_CONNECTIVES = [
    "because",
    "therefore",
    "consequently",
    "as a result",
    "due to",
    "reasoning",
    "the user explained that",
    "the user reasoned that",
    "the rationale",
    "inference",
    "the user's logic",
    "causal connection",
]

THIRD_PERSON_GREETING_PATTERNS = [
    "exchanged greetings",
    "greeted",
    "said hello",
    "opened the conversation",
    "initial pleasantries",
    "the user thanked",
    "expressed gratitude",
    "closed the conversation",
    "said goodbye",
]

THIRD_PERSON_TRANSACTIONAL_PATTERNS = [
    "the user asked to run",
    "the user requested execution",
    "the user asked to open",
    "the user asked to show",
    "the user asked to display",
    "the user asked to create",
    "the user asked to delete",
    "the user asked to generate",
    "the user asked to compile",
    "the user asked to deploy",
    "the user asked to list",
    "task request",
    "operational request",
]
```

**Note:** These third-person patterns are calibrated for Nemori's EpisodeGenerator
output style (GPT-4o-mini default model). If the narration model changes, these
patterns should be validated against a sample of new narratives.

### 3.5 Priority Limitations

> **[ADVERSARIAL FINDING 1.5 -- Paradox-Hunt]** The priority system resolves
> multi-category matches SYNTACTICALLY (which pattern set matched first in the
> priority order), not SEMANTICALLY (what the user actually meant).

**Known misclassification patterns:**

| Input | Priority assigns | Actual intent | Impact |
|-------|-----------------|---------------|--------|
| "No, I prefer dark mode" | correction (0.9) | preference (0.8) | +0.1 importance |
| "No, always use spaces" | correction (0.9) | instruction (0.85) | +0.05 importance |
| "Actually, I live in Portland now" | correction (0.9) | fact update (0.6) | +0.3 importance |

**Impact assessment:** For ALWAYS-store categories, misclassification only affects
initial_importance (delta of 0.05-0.3). The dynamics system's feedback loop will
correct importance over time as the memory is retrieved (or not). The worst case
is "Actually, I live in Portland now" getting correction importance (0.9) instead
of fact importance (0.6), which means it starts higher but decays to its natural
level through retrieval feedback.

**The LLM classifier (Section 7.4) is the proper fix.** The heuristic priority
system is a pragmatic baseline that errs on the side of over-importance (corrections
are always high importance, and false-positive corrections are still stored).

### 3.6 Known False Positive Patterns

> **[ADVERSARIAL FINDING 2.2 -- Scale-Check]** The following patterns have known
> false positive surfaces. When `use_word_boundaries=True`, most are mitigated.

| Pattern | Category | False positive examples | Mitigated by word boundaries? |
|---------|----------|------------------------|------------------------------|
| `"no "` | correction | "I have no idea", "no problem" | Partially (still matches "no idea") |
| `"so "` | reasoning | "also went", "I'm so happy" | Yes ("also" no longer matches) |
| `"since"` | reasoning | "I've been here since Monday" | No (temporal "since" is legitimate word) |
| `"always"` | instruction | "I don't always agree" | No (legitimate word in non-instruction context) |
| `"start "` | transactional | "That's a good start for" | Yes |
| `"open "` | transactional | "I'm open to suggestions" | No (legitimate word) |
| `"test "` | transactional | "This is a test case" | No (legitimate word) |

**Residual false positive rate (estimated):** With word boundaries enabled, ~5-10%
of pattern matches are false positives. Without word boundaries, ~15-25%.

**Impact:** False positives shift category assignment. In the worst case, a fact
gets classified as a correction (over-importance by 0.3). The dynamics feedback
loop corrects this over time. False positives on NEVER-store categories (greeting,
transactional) are worse because they prevent storage, but these are overridden by
the priority system (substantive categories take priority over transactional/greeting).

### 3.7 Metadata-Assisted Classification

> **[ADVERSARIAL FINDING 3.7 -- Object-Transpose]** Message-level metadata signals
> complement content-based pattern matching and reduce false positive rates.

When metadata signals strongly indicate a category, boost confidence for that
category by 0.15:

```python
METADATA_CONFIDENCE_BOOSTS = {
    # (metadata_condition, category) -> confidence boost
    ("message_count <= 2", "greeting"): 0.15,      # Short conversations are likely greetings
    ("message_count > 10", "NOT_greeting"): 0.15,   # Long conversations are unlikely to be pure greetings
    ("source_type == 'semantic'", "fact"): 0.10,     # Semantic memories are pre-filtered for info value
}
```

These boosts apply AFTER the base confidence calculation and BEFORE the fail-open
check. They are additive and capped at 1.0.

---

## 4. Write Policy Rules

### 4.1 Decision Table

| Category      | Condition                                              | should_store |
|---------------|--------------------------------------------------------|-------------|
| preference    | (always)                                               | True        |
| fact          | (always)                                               | True        |
| correction    | (always)                                               | True        |
| instruction   | (always)                                               | True        |
| reasoning     | len >= min_reasoning_length AND connectives >= min_reasoning_connectives | True |
| reasoning     | len < min_reasoning_length OR connectives < min_reasoning_connectives | False |
| greeting      | (always, unless fail-open override)                    | False       |
| transactional | (always, unless fail-open override)                    | False       |

### 4.2 Fail-Open Override

If `confidence < config.confidence_threshold` (default 0.5):
- Override `should_store = True` regardless of category
- Set `reason` to include "fail-open: low confidence ({confidence:.2f} < {threshold:.2f})"

This means a message classified as "greeting" with confidence 0.3 WILL be stored.
The classifier was not confident enough to reject it, so we err on the side of
preservation.

### 4.3 Length-Based Reclassification

If a message is initially classified as "greeting" but exceeds the effective
greeting length threshold:
1. Remove greeting patterns from consideration
2. Re-run classification on remaining pattern sets
3. If a new category is found, use it (with potentially different should_store)
4. If no other category matches, keep "greeting" but apply fail-open if confidence is low

**Effective thresholds** (accounting for episode length offset):
```python
if metadata and metadata.get("source_type") == "episode":
    effective_greeting_threshold = config.max_greeting_length + config.episode_length_offset
    effective_transactional_threshold = config.max_transactional_length + config.episode_length_offset
else:
    effective_greeting_threshold = config.max_greeting_length
    effective_transactional_threshold = config.max_transactional_length
```

Same logic applies to "transactional" with the effective transactional threshold.

This prevents long, substantive messages that happen to start with "Hello" or
"Run this" from being incorrectly filtered out.

### 4.4 Confidence Calibration

> **[ADVERSARIAL FINDING 1.4 + 2.4 -- Paradox-Hunt + Scale-Check]** The original
> confidence formula caused fail-open to trigger on >95% of inputs, making the
> gate a no-op for filtering. The calibrated formula (Section 3.3) addresses this.

**Design rationale for calibration:**

The fail-open gate should trigger on GENUINELY AMBIGUOUS content, not on every
single-pattern match. The calibrated confidence scoring (Section 3.3) ensures:

1. **Single high-signal pattern match** (e.g., "I prefer"): confidence 0.35.
   For ALWAYS-store categories, this doesn't matter (stored regardless).
   For NEVER-store categories, this is below threshold -> fail-open stores it.
   This is the CORRECT behavior: a single greeting pattern on substantive text
   should not confidently reject.

2. **Multiple pattern matches** (e.g., "hello", "how are you", "nice to meet"):
   confidence 0.35 + 0.08 + 0.08 = 0.51. Above threshold. The gate confidently
   rejects this as a greeting.

3. **No pattern matches**: confidence 0.2-0.3 (default). Below threshold.
   Fail-open stores. Correct behavior for unclassifiable content.

**Expected fail-open rates after calibration:**
- Episodes: ~15-25% (mostly edge cases and multi-category mixes)
- Semantic memories with knowledge_type: ~0% (shortcut bypasses classification)
- Semantic memories without knowledge_type: ~25-35%
- Random/non-English text: ~100% (by design)

---

## 5. EncodingPolicy Class

### 5.1 Interface

```python
class EncodingPolicy:
    """Deterministic encoding gate for memory candidates.

    Evaluates episode content and semantic facts to decide whether
    they should be persisted to the memory database.

    Thread-safe: all methods are pure functions of their arguments
    plus the immutable config. No mutable state.

    Extensibility: the classify() method is the extension point.
    To swap heuristic classification for an LLM-based classifier,
    subclass EncodingPolicy and override classify().
    """

    def __init__(self, config: EncodingConfig | None = None) -> None:
        """Initialize with optional config. Uses defaults if None."""

    def evaluate(
        self,
        episode_content: str,
        metadata: dict | None = None,
    ) -> EncodingDecision:
        """Main API: evaluate a memory candidate.

        Args:
            episode_content: The text content to evaluate. This is either:
                - An episode narrative (from EpisodeGenerator) -- THIRD-PERSON text
                - A semantic fact (from PredictionCorrectionEngine) -- also third-person
            metadata: Optional context about the content source. Recognized keys:
                - "message_count" (int): number of messages in the source episode
                - "user_id" (str): user identifier
                - "source_type" (str): "episode" or "semantic"
                - "knowledge_type" (str): Nemori's knowledge_type for semantic memories

        Returns:
            EncodingDecision with all fields populated.

        Note on determinism: deterministic over the full input tuple
        (episode_content, metadata). Different metadata for the same content
        may produce different decisions (metadata boosts confidence).
        """

    def classify(self, text: str, metadata: dict | None = None) -> tuple[str, float]:
        """Classify text into a category with confidence.

        This is the extension point for swapping heuristic classification
        with an LLM-based classifier.

        [ADVERSARIAL FINDING 1.1] The metadata parameter was added to support
        source_type-aware pattern set selection (Section 3.4).

        Args:
            text: The text to classify.
            metadata: Optional context for pattern set selection.

        Returns:
            Tuple of (category, confidence) where:
                - category is one of the 7 valid categories
                - confidence is in [0.0, 1.0]

        Determinism: same (text, metadata) always produces same (category, confidence).
        """
```

### 5.2 evaluate() Algorithm

```
def evaluate(episode_content, metadata):
    1. Handle empty input:
       if episode_content is None or episode_content.strip() == "":
           return EncodingDecision(
               should_store=False,
               category="greeting",
               confidence=1.0,
               reason="Empty input",
               initial_importance=0.0,
           )

    2. Semantic memory shortcut (Section 5.6):
       if metadata and metadata.get("source_type") == "semantic":
           knowledge_type = metadata.get("knowledge_type")
           if knowledge_type and knowledge_type in KNOWLEDGE_TYPE_TO_CATEGORY:
               category = KNOWLEDGE_TYPE_TO_CATEGORY[knowledge_type]
               confidence = 0.85  # high confidence from LLM classification
               # Skip to step 5 (apply write policy)
               goto step_5

    3. Classify:
       text = episode_content.strip()
       category, confidence = self.classify(text, metadata)

    4. Apply metadata boost:
       if metadata and metadata.get("message_count", 0) > 5:
           confidence = min(1.0, confidence + 0.1)
       if len(text) > 500:
           confidence = min(1.0, confidence + 0.05)

    4b. Length-based reclassification:
       effective_greeting_len = config.max_greeting_length
       effective_transactional_len = config.max_transactional_length
       if metadata and metadata.get("source_type") == "episode":
           effective_greeting_len += config.episode_length_offset
           effective_transactional_len += config.episode_length_offset

       if category == "greeting" and len(text) > effective_greeting_len:
           category, confidence = self._reclassify_without(text, "greeting", metadata)
       if category == "transactional" and len(text) > effective_transactional_len:
           category, confidence = self._reclassify_without(text, "transactional", metadata)

    step_5:
    5. Apply write policy:
       should_store = self._apply_write_policy(category, text, confidence)

    6. Apply fail-open:
       reason = self._build_reason(category, confidence)
       if not should_store and confidence < config.confidence_threshold:
           should_store = True
           reason += f" [fail-open: confidence {confidence:.2f} < {config.confidence_threshold}]"

    7. Look up importance:
       initial_importance = CATEGORY_IMPORTANCE[category]

    8. Return:
       return EncodingDecision(
           should_store=should_store,
           category=category,
           confidence=confidence,
           reason=reason,
           initial_importance=initial_importance,
       )
```

### 5.3 classify() Algorithm

```
def classify(text, metadata=None):
    text_lower = text.lower()
    matches: dict[str, int] = {}  # category -> match count

    # Select pattern sets based on source type (Section 3.4)
    first_person_patterns = ALL_FIRST_PERSON_PATTERNS  # Section 3.1
    third_person_patterns = ALL_THIRD_PERSON_PATTERNS  # Section 3.4

    # Always check first-person patterns (handles direct quotes in narratives)
    For each (category, pattern_list) in first_person_patterns:
        count = _count_matches(text_lower, pattern_list)
        if count > 0:
            matches[category] = matches.get(category, 0) + count

    # Also check third-person patterns for episode/semantic content
    if metadata and metadata.get("source_type") in ("episode", "semantic"):
        For each (category, pattern_list) in third_person_patterns:
            count = _count_matches(text_lower, pattern_list)
            if count > 0:
                matches[category] = matches.get(category, 0) + count

    If no matches:
        # No patterns matched at all
        if len(text) < effective_greeting_threshold:
            return ("greeting", 0.3)  # short unclassifiable text, low confidence
        else:
            return ("fact", 0.2)  # long unclassifiable text, very low confidence -> fail-open

    # Pick highest-priority category among those with matches
    for category in PRIORITY_ORDER:
        if category in matches:
            total_patterns = len(first_person_patterns[category])
            if metadata and metadata.get("source_type") in ("episode", "semantic"):
                total_patterns += len(third_person_patterns[category])

            # Calibrated confidence (Section 3.3)
            matched_count = matches[category]
            raw_density = matched_count / total_patterns
            if matched_count >= 1:
                confidence = max(SINGLE_PATTERN_CONFIDENCE_FLOOR, raw_density)
                confidence = min(1.0, confidence + 0.08 * (matched_count - 1))
            else:
                confidence = raw_density

            return (category, confidence)

    # Unreachable if PRIORITY_ORDER covers all categories
    return ("greeting", 0.1)


def _count_matches(text_lower, pattern_list):
    """Count pattern matches using word boundaries or substring matching."""
    if config.use_word_boundaries:
        return sum(1 for p in pattern_list if re.search(r'\b' + re.escape(p) + r'\b', text_lower))
    else:
        return sum(1 for p in pattern_list if p in text_lower)
```

### 5.4 _apply_write_policy() Algorithm

```
def _apply_write_policy(category, text, confidence):
    if category in {"preference", "fact", "correction", "instruction"}:
        return True

    if category == "reasoning":
        text_len = len(text)
        connective_count = sum(
            1 for c in REASONING_CONNECTIVES
            if c in text.lower()
        )
        return (
            text_len >= config.min_reasoning_length
            and connective_count >= config.min_reasoning_connectives
        )

    # greeting, transactional
    return False
```

### 5.5 Design Properties

- **Deterministic**: Same `(episode_content, metadata)` tuple always produces same `EncodingDecision`. No randomness, no external state, no timestamps. Note: different metadata for the same content may produce different decisions, because metadata is part of the input.
- **Pure**: No side effects. Does not write to databases, files, or logs. The caller handles persistence.
- **Thread-safe**: The `EncodingConfig` is frozen. Pattern lists are module-level constants. No mutable instance state.
- **Extensible**: Override `classify()` to swap heuristics for an LLM classifier without touching the write policy logic.

### 5.6 Semantic Memory Shortcut

> **[ADVERSARIAL FINDING 2.6 -- Materialize]** Nemori's PredictionCorrectionEngine
> already classifies semantic memories via LLM into knowledge types. This is a
> HIGHER-QUALITY classification signal than heuristic pattern matching. The encoding
> gate should USE this signal rather than re-classifying with inferior heuristics.

When `metadata["source_type"] == "semantic"` and `metadata["knowledge_type"]` is
present, bypass `classify()` and map knowledge_type directly to an encoding category:

```python
KNOWLEDGE_TYPE_TO_CATEGORY: dict[str, str] = {
    # Nemori knowledge_type -> encoding category
    "personal_fact": "fact",
    "preference": "preference",
    "instruction": "instruction",
    "correction": "correction",
    "opinion": "preference",
    "skill": "fact",
    "relationship": "fact",
    "habit": "preference",
    "goal": "instruction",
    "context": "fact",
    # Unmapped types fall through to heuristic classification
}
```

**Confidence for shortcut path:** 0.85 (high confidence from LLM classification).
This is above the fail-open threshold, so the write policy decision is final.

**Rationale:** The predict-calibrate engine already invested LLM computation to
understand the content and assign a knowledge type. Re-classifying with substring
patterns discards this work and produces worse results (per Finding 2.5).

---

## 6. Edge Cases

### 6.1 Empty String

```
Input:  ""
Output: EncodingDecision(
    should_store=False,
    category="greeting",
    confidence=1.0,
    reason="Empty input",
    initial_importance=0.0,
)
```

The empty string is definitionally not worth storing. Confidence is 1.0 because
the classifier is certain this is vacuous content. This bypasses the fail-open
gate (confidence 1.0 >= threshold 0.5).

### 6.2 None Input

Treated identically to empty string. The `evaluate()` method normalizes `None`
to `""` before processing.

### 6.3 Very Long Text (>10000 chars) with No Clear Category

```
Input:  10000+ chars, no pattern matches
Output: EncodingDecision(
    should_store=True,   # fail-open
    category="fact",     # best guess for long unclassifiable content
    confidence=0.25,     # 0.2 base + 0.05 length bonus, below threshold
    reason="No category patterns matched; classified as fact by default [fail-open: confidence 0.25 < 0.50]",
    initial_importance=0.6,
)
```

Long unclassifiable text defaults to "fact" with very low confidence, triggering
fail-open. The initial_importance of 0.6 means it starts at medium importance
and the dynamics system will adjust based on retrieval feedback.

### 6.4 Mixed Categories

```
Input:  "No, actually I prefer dark mode. I live in Seattle and I always want you to remember that."
Matches: correction ("No,", "actually"), preference ("I prefer"), fact ("I live in"), instruction ("always", "remember that")
```

The priority order resolves: correction > instruction > preference > fact.
Classification: **correction** with confidence based on correction pattern matches.

```
Output: EncodingDecision(
    should_store=True,
    category="correction",
    confidence=0.43,     # calibrated: floor 0.35 + 0.08*(2-1) = 0.43 for 2 matches
    reason="Matched correction patterns: 'no,', 'actually,'. Priority: correction > instruction > preference > fact.",
    initial_importance=0.9,
)
```

Note: with calibrated confidence (0.43), this is still below the 0.5 threshold.
But the category is "correction" which has ALWAYS store policy. The fail-open gate
is irrelevant because the write policy already grants storage.

### 6.5 Non-English Text

```
Input:  "Je prefere le mode sombre. Mon nom est Pierre."
Matches: none (patterns are English-only)
```

No English patterns match. The text is moderately long (~50 chars). Classification
falls through to the default: "greeting" if short, "fact" if long. At 48 chars,
this is short, so:

```
Output: EncodingDecision(
    should_store=True,   # fail-open
    category="greeting",
    confidence=0.3,      # below threshold 0.5
    reason="No category patterns matched; short unclassifiable text [fail-open: confidence 0.30 < 0.50]",
    initial_importance=0.0,
)
```

Fail-open stores the content. The importance is 0.0 (greeting importance) which
means it starts at the bottom of the dynamics ranking. If the user later retrieves
it, importance will climb via `importance_update()`.

**Future improvement:** An LLM-based classifier (via `classify()` override) would
handle non-English text correctly.

### 6.6 Metadata-Boosted Confidence

```
Input:  "ok thanks"
Metadata: {"message_count": 12}

Without metadata:
  category="greeting", confidence=0.35 (matches "thanks", calibrated floor)
  fail-open: 0.35 < 0.50 -> should_store=True

With metadata (message_count > 5, +0.1 boost):
  category="greeting", confidence=0.45
  fail-open: 0.45 < 0.50 -> should_store=True (same result, but higher confidence)
```

The metadata boost does not change the outcome for this example, but consider:

```
Input:  "thanks for all your help with the project, cheers"
Metadata: {"message_count": 12}

Without metadata:
  category="greeting", confidence=0.43 (2 patterns: "thanks", "cheers")
  fail-open: 0.43 < 0.50 -> should_store=True

With metadata:
  confidence boosted to 0.53
  Above threshold -> should_store=False (greeting rejected)
```

The metadata boost pushes confidence above the threshold, enabling the gate to
confidently reject multi-pattern greetings in long conversations.

### 6.7 Greeting That Exceeds Length Threshold

```
Input:  "Hello! I wanted to tell you that I just moved to a new apartment in downtown Portland and I'm really excited about it."
Source: raw message (not episode)
Initial: category="greeting" (matches "hello")
Length:  119 chars > max_greeting_length (50)

Reclassification without greeting patterns:
  Matches: fact ("I'm", "I just moved")  # "i'm" matches FACT_PATTERNS
  New category: "fact", confidence = 0.43 (2 patterns, calibrated)

Output: EncodingDecision(
    should_store=True,
    category="fact",
    confidence=0.43,
    reason="Reclassified from greeting (length 119 > 50). Matched fact patterns.",
    initial_importance=0.6,
)
```

The reclassification correctly captures that this is a personal fact wrapped in
a greeting opener.

### 6.8 Reasoning Below Threshold

```
Input:  "Do X because Y."
Length: 17 chars < min_reasoning_length (100)
Connectives: 1 ("because") < min_reasoning_connectives (2)

Output: EncodingDecision(
    should_store=False,  # conditional check fails
    category="reasoning",
    confidence=0.35,     # 1 connective matched, calibrated floor
    reason="Reasoning below thresholds: length 17 < 100, connectives 1 < 2",
    initial_importance=0.4,
)
```

But wait: confidence 0.35 < threshold 0.5, so fail-open triggers:

```
Output: EncodingDecision(
    should_store=True,   # fail-open override
    category="reasoning",
    confidence=0.35,
    reason="Reasoning below thresholds: length 17 < 100, connectives 1 < 2 [fail-open: confidence 0.35 < 0.50]",
    initial_importance=0.4,
)
```

This is by design. The classifier is not confident enough to reject, so the
content is preserved. Short reasoning snippets may still contain valuable
context that the dynamics system can rank appropriately.

### 6.9 Transactional Command That Contains a Preference

```
Input:  "Run the tests but I prefer using pytest over unittest"
Initial: transactional ("run ") matched, preference ("I prefer") matched
Priority: preference > transactional

Output: category="preference" (higher priority), should_store=True
```

The priority system correctly identifies the persistent signal (preference)
over the ephemeral signal (run command).

### 6.10 Episode Narrative (Third-Person)

> **[ADVERSARIAL FINDING 2.5 -- Materialize]** New edge case demonstrating the
> episode narrative path.

```
Input:  "During the conversation, the user shared personal details about their background. They mentioned that they reside in San Francisco and work as a software engineer."
Metadata: {"source_type": "episode", "message_count": 8}

Classification with dual pattern sets:
  First-person patterns: no matches (no "I live in", "I am", etc.)
  Third-person patterns:
    fact: "the user resides" -> MATCH (via "the user resides" in third-person facts... wait, pattern is "the user lives". But "resides" != "lives")
    fact: "the user shared" -> MATCH
    fact: "the user mentioned" -> MATCH (wait, need to check: "mentioned" is in third-person fact patterns? No, "the user mentioned" IS in the list)
    fact: "personal detail" -> MATCH
    fact: "background information" -> close but "background" alone doesn't match "background information"

  Let's be precise: matched patterns = "the user shared", "the user mentioned", "personal detail"
  3 matches in fact category
  Category: "fact" (highest priority among matches)
  Confidence: min(1.0, 0.35 + 0.08*2) = 0.51

  Metadata boost: message_count=8 > 5, +0.1 -> confidence = 0.61
  Length: 163 chars, < 500, no length boost

Output: EncodingDecision(
    should_store=True,      # fact is ALWAYS-store
    category="fact",        # correct!
    confidence=0.61,
    reason="Matched third-person fact patterns. Episode source.",
    initial_importance=0.6, # correct!
)
```

With dual pattern sets, the episode narrative is correctly classified as a fact
with appropriate importance seeding.

### 6.11 Semantic Memory with knowledge_type

> **[ADVERSARIAL FINDING 2.6 -- Materialize]** New edge case demonstrating the
> semantic memory shortcut.

```
Input:  "The user lives in San Francisco"
Metadata: {"source_type": "semantic", "knowledge_type": "personal_fact"}

Semantic memory shortcut (Section 5.6):
  knowledge_type = "personal_fact"
  KNOWLEDGE_TYPE_TO_CATEGORY["personal_fact"] = "fact"
  confidence = 0.85

Output: EncodingDecision(
    should_store=True,
    category="fact",
    confidence=0.85,
    reason="Semantic memory shortcut: knowledge_type 'personal_fact' -> 'fact'",
    initial_importance=0.6,
)
```

Compare to the ORIGINAL spec (before adversarial review), where this same input
was classified as "greeting" with importance 0.0. The shortcut eliminates the
catastrophic misclassification.

---

## 7. Integration Points

### 7.1 Nemori EventBus: episode_created

Nemori's `MemorySystem` publishes an `episode_created` event via `EventBus.publish()`:

```python
# In MemorySystem (memory_system.py, line ~212)
self.event_bus = EventBus()
self.event_bus.subscribe("episode_created", self._handle_episode_created_event)
```

The payload contains:
```python
{"episode": Episode, "user_id": str}
```

**Integration:** Subscribe an encoding gate handler to `episode_created` that
runs BEFORE the semantic generation handler:

```python
# Integration code (not part of EncodingPolicy itself)
def _encoding_gate_handler(event_name: str, payload: dict) -> None:
    episode = payload.get("episode")
    if episode is None:
        return
    decision = encoding_policy.evaluate(
        episode.content,
        metadata={
            "message_count": episode.message_count,
            "user_id": episode.user_id,
            "source_type": "episode",  # triggers third-person pattern selection
        },
    )
    payload["encoding_decision"] = decision
    if not decision.should_store:
        payload["_skip_storage"] = True

event_bus.subscribe("episode_created", _encoding_gate_handler)
```

The encoding gate handler annotates the payload with the decision. Downstream
handlers (semantic generation, storage) check `payload.get("_skip_storage")`
and short-circuit if True.

### 7.2 Nemori SemanticMemory Creation

After predict-calibrate produces `SemanticMemory` objects, each one passes
through the encoding gate:

```python
# In the semantic generation pipeline
for semantic_memory in generated_memories:
    decision = encoding_policy.evaluate(
        semantic_memory.content,
        metadata={
            "knowledge_type": semantic_memory.knowledge_type,  # triggers shortcut
            "user_id": semantic_memory.user_id,
            "source_type": "semantic",
            "confidence": semantic_memory.confidence,
        },
    )
    if decision.should_store:
        # Annotate with initial_importance for dynamics
        semantic_memory.metadata["initial_importance"] = decision.initial_importance
        semantic_memory.metadata["encoding_category"] = decision.category
        repository.save(semantic_memory)
```

### 7.3 Hermes Dynamics: initial_importance

When a memory passes the encoding gate and is stored, its `initial_importance`
becomes the starting value for the dynamics system's importance state variable.

```python
# In the memory store (conceptual)
memory_entry = MemoryEntry(
    content=content,
    importance=decision.initial_importance,  # <-- seeded from encoding
    strength=S_initial,                       # <-- S_max for episodic, different for semantic
    # ... other fields
)
```

The dynamics system then evolves importance via:
```
imp' = clamp01(imp + delta * signal)
```
where `signal` comes from retrieval feedback (positive when retrieved and useful,
negative when retrieved and irrelevant).

### 7.4 Future LLM Classifier

The `classify()` method is the extension point. To add an LLM-based classifier:

```python
class LLMEncodingPolicy(EncodingPolicy):
    def __init__(self, config, llm_client):
        super().__init__(config)
        self.llm_client = llm_client

    def classify(self, text: str, metadata: dict | None = None) -> tuple[str, float]:
        # Use LLM for classification
        response = self.llm_client.classify(text, categories=VALID_CATEGORIES)
        return (response.category, response.confidence)
```

The rest of the pipeline (write policy, fail-open, importance mapping) remains
unchanged. The LLM classifier would handle:
- Non-English text
- Nuanced multi-category content
- Context-dependent classification (with metadata injection into the prompt)
- Third-person narrative classification without needing separate pattern sets

The heuristic classifier serves as the baseline and fallback if the LLM is
unavailable.

### 7.5 Alternative Architecture: Pre-Nemori Classification

> **[ADVERSARIAL FINDING 3.4 -- Object-Transpose]** An alternative architecture
> classifies at the RAW MESSAGE level BEFORE Nemori processes them.

**Concept:** Instead of gating episode narratives (third-person, post-narration),
classify individual messages (first-person, pre-narration) and propagate
annotations to episodes.

```
Messages --> [PER-MESSAGE ENCODING] --> Buffer --> Segmentation --> Episode Narrative
                  |                                                      |
             annotation per msg                              episode category =
             (category, importance)                      max-priority(msg annotations)
```

**Advantages:**
- First-person patterns (Section 3.1) work perfectly on raw messages
- No need for third-person pattern sets
- Finer-grained classification (per-message vs. per-episode)

**Disadvantages:**
- Requires modifying Nemori's MessageBuffer to carry annotations
- A message classified as "greeting" might be part of a substantive episode
- Classification happens before context is established

**Decision:** Not adopted for v1. The dual-pattern-set approach (Section 3.4)
addresses the pattern mismatch without requiring Nemori API changes. This
architecture is documented for future consideration if third-person pattern
accuracy proves insufficient.

---

## 8. Property Invariants

These invariants MUST hold for every call to `evaluate()`, regardless of input.
They are testable properties suitable for property-based testing with Hypothesis.

### 8.1 Range Invariants

```
For all (episode_content, metadata):
    decision = policy.evaluate(episode_content, metadata)
    assert 0.0 <= decision.confidence <= 1.0
    assert 0.0 <= decision.initial_importance <= 1.0
    assert decision.category in VALID_CATEGORIES
```

### 8.2 Determinism

```
For all (episode_content, metadata):
    d1 = policy.evaluate(episode_content, metadata)
    d2 = policy.evaluate(episode_content, metadata)
    assert d1.should_store == d2.should_store
    assert d1.category == d2.category
    assert d1.confidence == d2.confidence
    assert d1.initial_importance == d2.initial_importance
```

### 8.3 Write Policy Consistency

```
For all (episode_content, metadata):
    decision = policy.evaluate(episode_content, metadata)
    if decision.category in {"preference", "fact", "correction", "instruction"}:
        assert decision.should_store is True
    if decision.category in {"greeting", "transactional"} and decision.confidence >= config.confidence_threshold:
        assert decision.should_store is False
```

### 8.4 Fail-Open Guarantee

```
For all (episode_content, metadata):
    decision = policy.evaluate(episode_content, metadata)
    if decision.confidence < config.confidence_threshold:
        assert decision.should_store is True
```

### 8.5 Importance-Category Correspondence

```
For all (episode_content, metadata):
    decision = policy.evaluate(episode_content, metadata)
    assert decision.initial_importance == CATEGORY_IMPORTANCE[decision.category]
```

### 8.6 Empty Input

```
decision = policy.evaluate("")
assert decision.should_store is False
assert decision.category == "greeting"
assert decision.confidence == 1.0
assert decision.initial_importance == 0.0

decision = policy.evaluate(None)
assert decision.should_store is False
assert decision.category == "greeting"
assert decision.confidence == 1.0
assert decision.initial_importance == 0.0
```

### 8.7 ALWAYS-Store Categories Dominate

```
For all text where classify(text) returns category in {"preference", "fact", "correction", "instruction"}:
    decision = policy.evaluate(text)
    assert decision.should_store is True
    # regardless of confidence level
```

### 8.8 Priority Ordering

```
Given: text matches both correction and preference patterns
    category, _ = policy.classify(text)
    assert category == "correction"  # correction has higher priority

Given: text matches both instruction and fact patterns
    category, _ = policy.classify(text)
    assert category == "instruction"  # instruction has higher priority
```

### 8.9 Semantic Memory Shortcut Consistency

> **[ADVERSARIAL FINDING 2.6]** New invariant for the shortcut path.

```
For all (content, metadata) where metadata["source_type"] == "semantic"
    and metadata["knowledge_type"] in KNOWLEDGE_TYPE_TO_CATEGORY:
    decision = policy.evaluate(content, metadata)
    assert decision.category == KNOWLEDGE_TYPE_TO_CATEGORY[metadata["knowledge_type"]]
    assert decision.confidence == 0.85
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

Following the project's existing pattern (property-based tests with Hypothesis,
organized by theorem/invariant), the test file should be:

```
proofs/hermes-memory/python/tests/test_encoding.py
```

**Test classes:**
- `TestEncodingDecisionInvariants` --- properties 8.1 through 8.5, 8.9
- `TestClassification` --- category assignment for each pattern set
- `TestThirdPersonClassification` --- category assignment for narrative text (Section 3.4)
- `TestWritePolicy` --- decision table from Section 4.1
- `TestFailOpen` --- fail-open behavior at various confidence levels
- `TestEdgeCases` --- all cases from Section 6 including 6.10 and 6.11
- `TestReclassification` --- length-based reclassification (Sections 6.7)
- `TestPriority` --- priority ordering (Section 3.2)
- `TestMetadataInfluence` --- metadata boost effects
- `TestSemanticShortcut` --- knowledge_type bypass (Section 5.6)
- `TestConfidenceCalibration` --- calibrated confidence values (Section 3.3)

**Property-based tests with Hypothesis:**
```python
@given(text=st.text(min_size=0, max_size=10000))
def test_range_invariants(text):
    decision = policy.evaluate(text)
    assert 0.0 <= decision.confidence <= 1.0
    assert 0.0 <= decision.initial_importance <= 1.0
    assert decision.category in VALID_CATEGORIES

@given(text=st.text(min_size=0, max_size=10000))
def test_determinism(text):
    d1 = policy.evaluate(text)
    d2 = policy.evaluate(text)
    assert d1.should_store == d2.should_store
    assert d1.category == d2.category

@given(text=st.text(min_size=0, max_size=10000))
def test_fail_open_guarantee(text):
    decision = policy.evaluate(text)
    if decision.confidence < DEFAULT_CONFIG.confidence_threshold:
        assert decision.should_store is True
```

### 9.2 Integration Tests

Located in:
```
thoughts/shared/plans/memory-system/encoding/tests/
```

These test the encoding gate's interaction with Nemori's data types:
- Create an `Episode` with various content, pass through encoding gate
- Create a `SemanticMemory` with various content, pass through encoding gate
- Verify `initial_importance` is correctly set in metadata
- Verify semantic memory shortcut produces correct categories

### 9.3 Coverage Target

80% line coverage for the encoding module. Property-based tests with Hypothesis
will exercise edge cases that unit tests miss.

---

## 10. File Layout

```
proofs/hermes-memory/python/
  hermes_memory/
    encoding.py          # EncodingConfig, EncodingDecision, CATEGORY_IMPORTANCE,
                         # VALID_CATEGORIES, all pattern lists (first-person AND
                         # third-person), KNOWLEDGE_TYPE_TO_CATEGORY,
                         # EncodingPolicy class
  tests/
    test_encoding.py     # Property-based and unit tests for the encoding gate
```

The encoding module is a single file because:
1. The policy is self-contained (dependencies: stdlib `dataclasses`, `typing`, `re`)
2. Pattern lists, config, decision type, and policy class are tightly coupled
3. The existing hermes_memory package follows a flat structure (core.py, markov_chain.py)

> **Note:** The addition of `re` (for word boundary matching) is the only new
> stdlib dependency compared to the original spec.

---

## 11. Non-Goals (Explicit Exclusions)

The encoding policy does NOT:
- Modify episode or semantic content (it is a read-only gate)
- Access databases or external services (pure function)
- Handle contradiction detection (separate concern; see architecture.md Section 4)
- Implement compression or consolidation (separate concern)
- Make network calls (no LLM in the base implementation)
- Track state between calls (stateless, deterministic)
- Handle user-specific pattern customization (v1 uses universal patterns)
- Classify at the raw message level (v1 classifies post-narration; see Section 7.5 for alternative)

---

## 12. Open Questions

- **Q1:** Should the pattern lists be configurable via EncodingConfig, or
  remain as module-level constants? Current design: constants. Pattern evolution
  happens via code changes, not configuration. This keeps the config small and
  the patterns version-controlled.

- **Q2:** Should the reclassification step (Section 4.3) recurse? Currently it
  runs once --- remove the offending category and reclassify. If the second
  classification also hits a NEVER-store category, should it reclassify again?
  Current design: no recursion, single reclassification pass. The fail-open gate
  catches any remaining ambiguity.

- **Q3:** Should semantic memories from predict-calibrate receive a confidence
  boost? Nemori's predict-calibrate already filters for novelty (only stores the
  "gap" between prediction and reality). This means semantic memories are
  pre-filtered for information value. **Resolution (partial):** The semantic
  memory shortcut (Section 5.6) assigns confidence 0.85 when knowledge_type
  is available, which effectively gives a boost. When knowledge_type is absent,
  no special treatment is applied.

- **Q4 (NEW):** Should the third-person pattern sets be generated from the
  first-person patterns via a template? E.g., "I like" -> "the user likes",
  "I prefer" -> "the user preferred". This would ensure consistency and reduce
  maintenance burden. Risk: templates may not capture all narrative phrasings.

- **Q5 (NEW):** Should the `use_word_boundaries` flag affect third-person
  patterns? Third-person patterns are multi-word phrases ("the user shared")
  that are less prone to substring false positives. Word boundaries may cause
  false negatives on inflected forms ("the user's sharing" would not match
  "the user shared" with exact word boundaries).

---

## 13. Validation and Monitoring

> **[ADVERSARIAL FINDINGS 3.1, 3.2, 3.3 -- Exclusion-Test]** The heuristic
> approach has failure thresholds that must be monitored.

### 13.1 Validation Protocol

Before deploying the encoding gate, run the classifier against a corpus of
50+ real Nemori episode narratives and measure classification accuracy:

```python
# Validation script (not part of the module)
corpus = load_nemori_episodes(n=50)  # real episodes from test conversations
for episode in corpus:
    decision = policy.evaluate(episode.content, metadata={"source_type": "episode"})
    expected_category = human_labeled_category(episode)
    log(decision.category, expected_category, decision.confidence)

accuracy = correct_classifications / total
assert accuracy >= 0.70, f"Classification accuracy {accuracy} below 70% threshold"
```

**Threshold:** If accuracy < 70%, the heuristic approach provides insufficient
value and should be replaced with the LLM classifier (Section 7.4).

### 13.2 Production Monitoring

Track the following metrics in production:

| Metric | Threshold | Action if exceeded |
|--------|-----------|-------------------|
| Fail-open rate | > 40% over 7 days | Review confidence calibration |
| Correction false positive rate | > 20% | Tighten "no " pattern matching |
| Category distribution entropy | Near-uniform (H/H_max > 0.9) | Classifier is not discriminating; review patterns |
| Semantic shortcut hit rate | < 50% | Nemori's knowledge_type coverage is low; expand mapping |

### 13.3 LLM Classifier Escalation Criteria

Abandon heuristic classification and switch to LLM classifier (Section 7.4) when:

1. **Misclassification rate > 30%** on validation corpus (Section 13.1)
2. **Fail-open rate > 40%** in production over 7 days
3. **Correction false positive rate > 20%** (correction has highest importance; false positives are costly)
4. **Non-English user base > 20%** of traffic (heuristics are English-only)
5. **Nemori narration model changes** and third-person patterns haven't been re-validated

Any ONE of these conditions is sufficient to escalate.

---

## 14. Success Criteria

1. **All 9 property invariants hold** under Hypothesis fuzzing (1000+ examples)
2. **All 11 edge cases** from Section 6 produce the documented outputs
3. **Zero false negatives** on a manually curated test set of 20 preference/fact/correction/instruction examples (both first-person and third-person)
4. **Fail-open rate** on random text is > 60% (the gate should be permissive by default)
5. **Fail-open rate** on classified text is < 40% (the calibration should work)
6. **Determinism**: 100% reproducibility across 10000 random inputs
7. **No external dependencies**: encoding.py imports only stdlib (`dataclasses`, `typing`, `re`)
8. **Integration**: `initial_importance` correctly seeds the dynamics system's importance state for at least 3 categories tested end-to-end
9. **Semantic shortcut**: knowledge_type-based classification produces correct categories for all mapped types
10. **Episode narrative accuracy**: classification accuracy >= 70% on a corpus of 50 Nemori episode narratives

---

**End of specification.**
