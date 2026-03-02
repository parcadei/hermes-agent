# Contradiction/Supersession Spec -- hermes_memory/contradiction.py

## Overview

Contradiction/supersession is piece 3/5 of the Hermes memory system. It detects
when a new memory contradicts or updates an existing memory and manages the
supersession lifecycle: marking the old memory as superseded, linking it to
its replacement, and ensuring recall surfaces only the latest version.

This module sits between encoding (piece 1) and recall (piece 2). When a new
memory is encoded, the contradiction detector scans existing memories for
semantic conflicts. When contradictions are found, the system either supersedes
the old memory (replacing it) or marks both as conflicting for human resolution.

New module: `hermes_memory/contradiction.py`
New test file: `tests/test_contradiction.py`

---

## 1. Dependencies

| Module | What contradiction.py uses |
|--------|---------------------------|
| `engine.py` | `MemoryState` (for scoring interface), `ParameterSet` |
| `encoding.py` | `EncodingDecision`, `VALID_CATEGORIES`, `CATEGORY_IMPORTANCE` |
| `recall.py` | (No direct dependency. Recall reads supersession metadata.) |
| `core.py` | Transitively via engine.py. No direct imports. |

No new external dependencies. Only stdlib `dataclasses`, `enum`, `logging`, `re`.

---

## 2. Core Concepts

### 2.1 Contradiction vs Supersession

**Contradiction**: Two memories assert incompatible claims about the same subject.
- Example: "I live in New York" vs "I live in London"
- Example: "Always use tabs for indentation" vs "Never use tabs for indentation"
- Both memories are factually valid statements but cannot both be true simultaneously.

**Supersession**: A newer memory explicitly updates or replaces an older one.
- Example: "My email is old@example.com" superseded by "My email is new@example.com"
- Example: "I prefer dark mode" superseded by "I prefer light mode"
- The newer memory is the authoritative version. The older one is no longer current.

**Key difference**: Supersession is directional (new replaces old). Contradiction
detection is symmetric (either could be newer). The system resolves this by
always treating the more recent memory as authoritative when confidence is high.

### 2.2 Scope Boundaries

What contradiction detection IS:
- Pairwise semantic comparison between a candidate and existing memories
- Category-aware: corrections supersede by default; preferences update in-place
- Lightweight: no LLM calls, no embeddings (uses text features only)
- Deterministic: same inputs always produce same supersession decisions

What contradiction detection IS NOT:
- Entailment reasoning (no NLI model)
- General knowledge graph consistency checking
- Multi-hop inference ("A says X, B says Y, therefore Z is contradicted")
- Real-time conflict resolution during conversation

### 2.3 Integration Points

```
 encode(text) ─── EncodingDecision ───> detect_contradictions(candidate, pool)
                                              │
                                              ▼
                                        ContradictionResult
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                            supersede(old, new)   no_contradiction
                                    │
                                    ▼
                            SupersessionRecord
                                    │
                                    ▼
                          recall() filters superseded memories
```

---

## 3. Data Types

### 3.1 ContradictionType Enum

```python
import enum

class ContradictionType(enum.Enum):
    """Type of contradiction detected between two memories."""
    DIRECT_NEGATION = "direct_negation"       # Explicit denial
    VALUE_UPDATE = "value_update"             # Same field, different value
    PREFERENCE_REVERSAL = "preference_reversal"  # Changed preference
    INSTRUCTION_CONFLICT = "instruction_conflict"  # Conflicting instructions
```

#### Invariants
- Exactly four members.
- Each member maps to a distinct detection strategy (Section 6).

### 3.2 ContradictionConfig

```python
@dataclass(frozen=True)
class ContradictionConfig:
    """Configuration for the contradiction detection pipeline.

    Frozen to prevent accidental mutation.

    Attributes:
        similarity_threshold: Minimum subject overlap to consider contradiction.
                              Domain: (0.0, 1.0]. Default: 0.3.
        confidence_threshold: Minimum confidence to auto-supersede vs flag.
                              Domain: (0.0, 1.0]. Default: 0.7.
        max_candidates:       Maximum existing memories to scan per new memory.
                              Domain: >= 1. Default: 50.
        category_weights:     Per-category sensitivity multiplier for detection.
                              Keys must be from VALID_CATEGORIES.
                              Domain: each value in (0.0, 2.0]. Default: see below.
        enable_auto_supersede: If True, high-confidence contradictions auto-supersede.
                               If False, all contradictions are flagged only.
                               Default: True.
        value_pattern_min_tokens: Minimum token count for value extraction.
                                  Domain: >= 1. Default: 2.
        category_thresholds:  Per-category confidence thresholds for auto-supersession.
                              Keys must be from VALID_CATEGORIES.
                              A threshold of 0.0 means "always supersede regardless of
                              confidence" (used for corrections).
                              Default: {"preference": 0.6, "fact": 0.7,
                                         "instruction": 0.7, "correction": 0.0}.
                              Categories not in this dict fall back to
                              confidence_threshold.
    """
    similarity_threshold: float = 0.3
    confidence_threshold: float = 0.7
    max_candidates: int = 50
    category_weights: dict[str, float] = None  # post_init fills defaults
    enable_auto_supersede: bool = True
    value_pattern_min_tokens: int = 2
    category_thresholds: dict[str, float] = None  # post_init fills defaults
```

#### Default category_weights

```python
{
    "correction": 1.5,     # Corrections supersede aggressively
    "instruction": 1.2,    # Instructions update with high confidence
    "preference": 1.0,     # Preferences update normally
    "fact": 0.8,           # Facts require higher evidence
    "reasoning": 0.5,      # Reasoning rarely contradicts
    "greeting": 0.0,       # Greetings never contradict
    "transactional": 0.0,  # Transactional never contradicts
}
```

#### Default category_thresholds

```python
{
    "preference": 0.6,    # Preferences change easily
    "fact": 0.7,          # Facts require higher evidence
    "instruction": 0.7,   # Instructions are authoritative
    "correction": 0.0,    # Always supersede (0.0 = unconditional)
}
```

Categories not present in `category_thresholds` fall back to the global
`confidence_threshold` value.

#### Validation rules
- `0.0 < similarity_threshold <= 1.0`
- `0.0 < confidence_threshold <= 1.0`
- `max_candidates >= 1`
- All keys in category_weights are in VALID_CATEGORIES
- All values in category_weights are in [0.0, 2.0]
- All keys in category_thresholds are in VALID_CATEGORIES
- All values in category_thresholds are in [0.0, 1.0]
- `value_pattern_min_tokens >= 1`

### 3.3 SubjectExtraction

```python
@dataclass(frozen=True)
class SubjectExtraction:
    """Extracted subject and value from a memory text.

    Attributes:
        subject:     The topic/entity being discussed (normalized lowercase).
        value:       The asserted value or state (normalized lowercase). None if
                     no clear value extracted.
        field_type:  The semantic field type ("location", "name", "email",
                     "preference", "instruction", "fact", "unknown").
        raw_match:   The original text span that was matched.
    """
    subject: str
    value: str | None
    field_type: str
    raw_match: str
```

### 3.4 ContradictionDetection

```python
@dataclass(frozen=True)
class ContradictionDetection:
    """Result of comparing a candidate memory against one existing memory.

    Attributes:
        existing_index:    Index of the existing memory in the pool.
        contradiction_type: Type of contradiction detected.
        confidence:        Detection confidence in [0.0, 1.0].
        subject_overlap:   Jaccard similarity of extracted subjects.
        candidate_subject: SubjectExtraction from the candidate.
        existing_subject:  SubjectExtraction from the existing memory.
        explanation:       Human-readable explanation of why this is a contradiction.
    """
    existing_index: int
    contradiction_type: ContradictionType
    confidence: float
    subject_overlap: float
    candidate_subject: SubjectExtraction
    existing_subject: SubjectExtraction
    explanation: str
```

### 3.5 SupersessionAction Enum

```python
class SupersessionAction(enum.Enum):
    """Action to take on a detected contradiction."""
    AUTO_SUPERSEDE = "auto_supersede"    # Old memory deactivated, new replaces it
    FLAG_CONFLICT = "flag_conflict"      # Both kept, conflict flagged for review
    SKIP = "skip"                        # No action (below threshold)
```

### 3.6 ContradictionResult

```python
@dataclass(frozen=True)
class ContradictionResult:
    """Output of the full contradiction detection pipeline.

    Attributes:
        detections:          Tuple of all ContradictionDetection found.
        actions:             Tuple of (existing_index, SupersessionAction) pairs.
        superseded_indices:  Frozenset of existing memory indices to deactivate.
        flagged_indices:     Frozenset of existing memory indices flagged for review.
        has_contradiction:   True if any contradiction was detected.
        highest_confidence:  Maximum confidence across all detections (0.0 if none).
    """
    detections: tuple[ContradictionDetection, ...]
    actions: tuple[tuple[int, SupersessionAction], ...]
    superseded_indices: frozenset[int]
    flagged_indices: frozenset[int]
    has_contradiction: bool
    highest_confidence: float
```

### 3.7 SupersessionRecord

```python
@dataclass(frozen=True)
class SupersessionRecord:
    """Record of a supersession event, for audit trail.

    Attributes:
        old_index:           Index of the superseded memory.
        new_index:           Index of the new (superseding) memory. -1 for candidate.
        contradiction_type:  Type of contradiction that triggered supersession.
        confidence:          Confidence of the detection.
        timestamp:           Logical timestamp of the supersession event.
        explanation:         Human-readable explanation.
    """
    old_index: int
    new_index: int
    contradiction_type: ContradictionType
    confidence: float
    timestamp: float
    explanation: str
```

---

## 4. Subject Extraction

The subject extraction pipeline converts free-text memories into structured
subject-value pairs for comparison. This is the foundation for all contradiction
detection — if subjects cannot be compared, contradictions cannot be detected.

### 4.1 Field-Specific Extractors

Pattern-based extractors for common personal data fields:

```python
FIELD_EXTRACTORS: dict[str, list[tuple[re.Pattern, str, str]]] = {
    # (pattern, subject_name, field_type)
    "location": [
        (r"(?:i (?:live|reside|am based|am located)\s+(?:in|at)\s+)(.+)", "location", "location"),
        (r"(?:my (?:home|residence|address)\s+is\s+)(.+)", "location", "location"),
        (r"(?:i(?:'m| am) (?:from|based in)\s+)(.+)", "location", "location"),
        (r"(?:the user (?:lives|resides|is based|is located)\s+(?:in|at)\s+)(.+)", "location", "location"),
    ],
    "name": [
        (r"(?:my name is\s+)(.+)", "name", "name"),
        (r"(?:i(?:'m| am)\s+)(\w+(?:\s+\w+)?)", "name", "name"),
        (r"(?:the user(?:'s)? name is\s+)(.+)", "name", "name"),
        (r"(?:call me\s+)(.+)", "name", "name"),
    ],
    "email": [
        (r"(?:my email(?:\s+address)?\s+is\s+)(\S+@\S+)", "email", "email"),
        (r"(?:the user(?:'s)? email is\s+)(\S+@\S+)", "email", "email"),
    ],
    "job": [
        (r"(?:i (?:work|am employed)\s+(?:at|for)\s+)(.+)", "employer", "job"),
        (r"(?:i (?:work|am employed)\s+as\s+(?:a|an)\s+)(.+)", "role", "job"),
        (r"(?:my (?:job|role|position|title)\s+is\s+)(.+)", "role", "job"),
        (r"(?:the user works\s+(?:at|for|as)\s+)(.+)", "employer", "job"),
    ],
    "preference": [
        (r"(?:i (?:prefer|like|enjoy|love|favor)\s+)(.+)", "preference", "preference"),
        (r"(?:i (?:don't|dont|do not) (?:like|enjoy|want)\s+)(.+)", "dispreference", "preference"),
        (r"(?:my favorite\s+\w+\s+is\s+)(.+)", "preference", "preference"),
        (r"(?:the user (?:prefers|likes|enjoys)\s+)(.+)", "preference", "preference"),
    ],
    "instruction": [
        (r"(?:always\s+)(.+)", "instruction", "instruction"),
        (r"(?:never\s+)(.+)", "instruction", "instruction"),
        (r"(?:from now on,?\s+)(.+)", "instruction", "instruction"),
        (r"(?:remember (?:to|that)\s+)(.+)", "instruction", "instruction"),
    ],
}
```

#### Name Extractor Stop-Word Guard

The `"I'm X"` name pattern `r"(?:i(?:'m| am)\s+)(\w+(?:\s+\w+)?)"` greedily matches
non-name statements like "I'm tired", "I'm a developer", "I'm from Paris". To prevent
false positives, after extraction via this pattern, if the extracted value (lowercased)
starts with any word in `NAME_STOP_WORDS`, downgrade field_type to `"unknown"` (treat
as a non-name fallback).

```python
NAME_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "not", "from", "in", "at", "going", "tired", "working",
    "excited", "really", "very", "so", "just", "also", "still", "now", "here",
    "there", "sure", "sorry", "glad", "happy", "sad", "done", "back", "ready",
    "able", "trying", "looking", "thinking", "feeling", "getting", "doing",
    "being", "having", "making", "taking", "coming", "leaving", "staying",
    "moving", "living", "based",
})
```

The guard checks `extracted_value.split()[0].lower() in NAME_STOP_WORDS`. If True,
the extraction result is returned with `field_type="unknown"` and `subject="unknown"`
instead of `field_type="name"`.

### 4.2 Extraction Algorithm

```
def extract_subject(text: str, category: str | None = None) -> SubjectExtraction:
    1. Normalize: lowercase, strip whitespace, collapse multiple spaces.
    2. Try field-specific extractors in FIELD_EXTRACTORS order.
       - If match found: return SubjectExtraction with extracted fields.
       - Try all extractors; pick the one with the longest match.
       - If the winning extractor is a "name" pattern using the "I'm X" rule,
         apply the NAME_STOP_WORDS guard before returning.
    3. Fallback: extract first noun phrase as subject (heuristic).
       - Simple heuristic: first 1-4 words after stripping stop words.
       - value = None (no clear value extracted).
       - field_type = "unknown".
    4. Return SubjectExtraction.
```

### 4.3 Subject Normalization

Subjects are normalized for comparison:
- Lowercase
- Strip leading/trailing whitespace
- Collapse multiple whitespace to single space
- Strip trailing punctuation (.,;:!?)
- Strip leading articles ("a ", "an ", "the ")
- Strip filler words ("actually", "basically", "just")

### 4.4 Subject Overlap (Jaccard)

```python
def subject_overlap(a: SubjectExtraction, b: SubjectExtraction) -> float:
    """Compute subject similarity using token-level Jaccard index.

    Returns 0.0 when subjects have no common tokens, 1.0 for identical subjects.
    Exact field_type match gets a 0.2 bonus (clamped to 1.0).
    Same subject string gets 1.0 regardless of tokens.
    """
```

When both subjects have the same `field_type` (e.g., both "location"), add 0.2
to the Jaccard score (clamped at 1.0). This prioritizes same-field comparisons.

---

## 5. Contradiction Detection Strategies

Each `ContradictionType` has a dedicated detection strategy. The pipeline tries
all strategies and returns the highest-confidence result.

### 5.1 DIRECT_NEGATION

Detects when one memory explicitly negates or corrects another.

**Signals**:
- Encoding category is "correction" for the candidate
- Candidate text contains correction markers: "no,", "actually,", "that's wrong",
  "correction:", "not quite", etc. (reuse CORRECTION_PATTERNS from encoding.py)
- Subject overlap >= similarity_threshold

**Confidence calculation**:
```
confidence = min(1.0, subject_overlap * category_weight * correction_signal_strength)
```
where `correction_signal_strength` is the count of correction markers / 3 (capped at 1.0).

### 5.2 VALUE_UPDATE

Detects when two memories assert different values for the same field.

**Signals**:
- Both memories have SubjectExtractions with the same field_type
- Both have non-None values
- Values differ (after normalization)
- Subject overlap >= similarity_threshold

**Value containment check**: Before declaring values "different", check if one
normalized value is a substring of the other. If `a in b` or `b in a` (where
`a` and `b` are the normalized value strings), this is a partial match, not a
full contradiction. Reduce confidence by 50% (multiply by 0.5). This prevents
"New York City" vs "New York" from being treated as a full contradiction.

**Confidence calculation**:
```
confidence = min(1.0, subject_overlap * (1.0 if field_type_match else 0.5))

# Value containment penalty
a_norm = normalize(candidate_value)
b_norm = normalize(existing_value)
if a_norm in b_norm or b_norm in a_norm:
    confidence *= 0.5
```

Special case: when field_type is in {"email", "name", "location"}, and values
are clearly different strings, confidence is boosted to max(confidence, 0.8).
The value containment check applies BEFORE this boost — so "New York" vs
"New York City" gets halved first, then the boost may raise it.
Personal data fields have unambiguous values.

### 5.3 PREFERENCE_REVERSAL

Detects when a preference is reversed.

**Signals**:
- Both memories are category "preference" (or one is a correction of a preference)
- One asserts positive ("I like X") and the other negative ("I don't like X"), or
  both assert different preferences for the same subject
- Subject overlap >= similarity_threshold

**Confidence calculation**:
```
confidence = min(1.0, subject_overlap * polarity_diff)
```
where `polarity_diff` is 1.0 if polarities are opposite, 0.6 if same polarity
but different values.

**Polarity detection**: Simple keyword scan:
- Positive: "like", "prefer", "enjoy", "love", "want", "favor"
- Negative: "don't like", "hate", "dislike", "avoid", "don't want"

### 5.4 INSTRUCTION_CONFLICT

Detects when two instructions conflict.

**Signals**:
- Both memories are category "instruction"
- Both reference the same action or behavior
- One instructs to do X, the other instructs not to do X (or to do Y instead)
- Subject overlap >= similarity_threshold

**Confidence calculation**:
```
confidence = min(1.0, subject_overlap * instruction_conflict_signal)
```
where `instruction_conflict_signal` is 1.0 if one is "always" and the other
is "never" for the same action, 0.7 if both give different instructions for
the same action.

---

## 6. Detection Pipeline

### 6.1 detect_contradictions()

```python
def detect_contradictions(
    candidate_text: str,
    candidate_category: str,
    existing_texts: list[str],
    existing_categories: list[str],
    config: ContradictionConfig | None = None,
) -> ContradictionResult:
    """Main API: detect contradictions between a candidate and existing memories.

    Args:
        candidate_text:       Text of the new memory candidate.
        candidate_category:   Category from EncodingDecision.category.
        existing_texts:       Texts of existing memories to scan.
        existing_categories:  Categories of existing memories (parallel to texts).
        config:               ContradictionConfig. Uses defaults if None.

    Returns:
        ContradictionResult with all detections and recommended actions.

    Raises:
        TypeError:  If any text argument is not a str.
        ValueError: If existing_texts and existing_categories have different lengths.
        ValueError: If candidate_category is not in VALID_CATEGORIES.
        ValueError: If any existing category is not in VALID_CATEGORIES.
    """
```

### 6.2 Pipeline Steps

```
1. Validate inputs (types, lengths, categories).
2. Early exit: if candidate_category has weight 0.0 (greeting, transactional),
   return empty ContradictionResult.
3. Extract subject from candidate: extract_subject(candidate_text, candidate_category).
4. If candidate subject is empty/unknown with no value: return empty result.
5. For each existing memory (up to max_candidates):
   a. Skip if existing category has weight 0.0.
   b. Extract subject from existing memory.
   c. Compute subject_overlap.
   d. If subject_overlap < similarity_threshold: skip.
   e. Try ALL four detection strategies for every (candidate, existing) pair.
      Strategies that don't apply to the pair produce confidence 0.
      - DIRECT_NEGATION
      - VALUE_UPDATE
      - PREFERENCE_REVERSAL
      - INSTRUCTION_CONFLICT
      Take the highest-confidence result across all four strategies for that pair.
   f. Keep the strategy with highest confidence (from step 5e).
   g. If confidence > 0: create ContradictionDetection.
6. Sort detections by confidence (descending).
7. Determine actions for each detection:
   - Look up the category-aware threshold:
     `threshold = config.category_thresholds.get(candidate_category, config.confidence_threshold)`
   - If threshold == 0.0: always AUTO_SUPERSEDE (regardless of confidence),
     provided config.enable_auto_supersede is True.
   - confidence >= threshold AND config.enable_auto_supersede
     → AUTO_SUPERSEDE
   - confidence >= threshold AND NOT config.enable_auto_supersede
     → FLAG_CONFLICT
   - confidence > 0 but < threshold
     → FLAG_CONFLICT
8. Build and return ContradictionResult.
```

### 6.3 Action Resolution Rules

When multiple detections target the same existing memory:
- Take the highest-confidence detection's action.
- AUTO_SUPERSEDE wins over FLAG_CONFLICT for the same memory.
- A memory can only be superseded once per pipeline call.

---

## 7. Supersession Lifecycle

### 7.1 resolve_contradictions()

```python
def resolve_contradictions(
    result: ContradictionResult,
    candidate_text: str,
    existing_texts: list[str],
    timestamp: float = 0.0,
) -> tuple[list[SupersessionRecord], frozenset[int]]:
    """Resolve a ContradictionResult into concrete supersession actions.

    Args:
        result:          ContradictionResult from detect_contradictions().
        candidate_text:  Text of the new memory (for audit trail).
        existing_texts:  Texts of existing memories (for audit trail).
        timestamp:       Logical timestamp for the supersession event.

    Returns:
        Tuple of:
          - List of SupersessionRecord objects (one per AUTO_SUPERSEDE action).
          - Frozenset of indices that should be deactivated.

    Raises:
        TypeError: If result is not a ContradictionResult.
    """
```

### 7.2 Supersession Semantics

When a memory is superseded:
1. The old memory's `active` flag is set to False (soft delete).
2. A SupersessionRecord links old → new for audit trail.
3. Recall pipeline filters out inactive memories (already handled by existing
   MemoryState.active field convention — recall.py must check this).

**Note**: Access count inheritance is the caller's responsibility. The contradiction
module operates on text only — it receives strings, not MemoryState objects, and
therefore cannot compute or transfer access counts. The caller (memory store or
coordinator) should handle any access_count bonuses when processing SupersessionRecords.

### 7.3 Category-Specific Supersession Rules

| Category | Supersession behavior |
|----------|----------------------|
| correction | Always supersedes target. Corrections are explicit updates. |
| fact | Supersedes if VALUE_UPDATE with confidence >= 0.7. Facts are identity-stable. |
| preference | Supersedes if PREFERENCE_REVERSAL with confidence >= 0.6. Preferences change. |
| instruction | Supersedes if INSTRUCTION_CONFLICT with confidence >= 0.7. Instructions are authoritative. |
| reasoning | Flags only. Reasoning doesn't supersede — both perspectives may be valid. |
| greeting | N/A (weight 0.0, never enters pipeline). |
| transactional | N/A (weight 0.0, never enters pipeline). |

---

## 8. Polarity Detection

### 8.1 Polarity Enum

```python
class Polarity(enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
```

### 8.2 detect_polarity()

```python
def detect_polarity(text: str) -> Polarity:
    """Detect the polarity (positive/negative/neutral) of a memory text.

    Uses keyword matching — not sentiment analysis. Detects explicit
    assertion vs negation patterns.

    Positive signals: "like", "prefer", "enjoy", "love", "want", "favor",
                      "always", "do", "use"
    Negative signals: "don't", "hate", "dislike", "avoid", "never", "stop",
                      "not", "no longer"

    When both positive and negative signals present, the one with more matches
    wins. Ties resolve to NEUTRAL.
    """
```

---

## 9. Instruction Conflict Detection

### 9.1 Action Extraction

```python
def extract_action(text: str) -> str | None:
    """Extract the action from an instruction text.

    Strips instruction prefixes ("always", "never", "remember to", etc.)
    and returns the remaining action phrase, normalized.

    Returns None if no instruction pattern is found.
    """
```

### 9.2 Instruction Polarity

Instructions have inherent polarity:
- "Always do X" → positive polarity for action X
- "Never do X" → negative polarity for action X
- "Remember to X" → positive
- "Don't ever X" → negative
- "From now on, X" → positive (unless X contains negation)

Two instructions conflict when:
1. Same action (after normalization), opposite polarities
2. OR: same subject, different actions (e.g., "always use tabs" vs "always use spaces")

---

## 10. Public API Surface

### 10.1 Functions (6 public symbols)

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `extract_subject` | `(text: str, category: str \| None) -> SubjectExtraction` | Extract subject/value from memory text |
| `subject_overlap` | `(a: SubjectExtraction, b: SubjectExtraction) -> float` | Jaccard subject similarity |
| `detect_polarity` | `(text: str) -> Polarity` | Keyword-based polarity detection |
| `extract_action` | `(text: str) -> str \| None` | Extract action from instruction text |
| `detect_contradictions` | `(candidate_text, candidate_category, existing_texts, existing_categories, config) -> ContradictionResult` | Main detection pipeline |
| `resolve_contradictions` | `(result, candidate_text, existing_texts, timestamp) -> (list[SupersessionRecord], frozenset[int])` | Resolve detections to supersession actions |

### 10.2 Classes/Enums (7 public symbols)

| Symbol | Type | Description |
|--------|------|-------------|
| `ContradictionType` | Enum | Four contradiction types |
| `Polarity` | Enum | Positive/Negative/Neutral |
| `SupersessionAction` | Enum | Auto/Flag/Skip |
| `ContradictionConfig` | frozen dataclass | Pipeline configuration |
| `SubjectExtraction` | frozen dataclass | Extracted subject/value |
| `ContradictionDetection` | frozen dataclass | Single detection result |
| `ContradictionResult` | frozen dataclass | Full pipeline output |
| `SupersessionRecord` | frozen dataclass | Audit trail record |

### 10.3 Constants (3 public symbols)

| Symbol | Type | Description |
|--------|------|-------------|
| `FIELD_EXTRACTORS` | dict | Pattern-based field extractors |
| `NAME_STOP_WORDS` | frozenset[str] | Stop words that disqualify "I'm X" name extraction |
| `DEFAULT_CONTRADICTION_CONFIG` | ContradictionConfig | Module-level default config |

Total: 16 public symbols.

---

## 11. Edge Cases

### 11.1 Empty inputs
- Empty candidate_text → return empty ContradictionResult (no contradiction)
- Empty existing_texts list → return empty ContradictionResult
- Empty existing memory text → skip that memory in scan

### 11.2 Self-contradiction
- If candidate_text appears in existing_texts (exact duplicate), it is NOT
  a contradiction — it's a duplicate. Skip it. Subject overlap will be 1.0
  but value will be identical, so VALUE_UPDATE won't trigger.

### 11.3 Partial matches
- Subject overlap at exactly similarity_threshold: INCLUDE (>=, not >)
- Confidence at exactly confidence_threshold: AUTO_SUPERSEDE (>=, not >)

### 11.4 All-greeting pool
- When all existing memories are greeting/transactional: empty result.
- When candidate is greeting/transactional: early exit, empty result.

### 11.5 Max candidates
- When existing_texts has more than max_candidates entries, scan only the
  first max_candidates. Rationale: in practice, recent memories are more
  likely to be contradicted, and a DB query would return them first.

### 11.6 Unicode and case
- All comparisons are case-insensitive (lowercase normalization).
- Unicode is preserved but normalized with unicodedata.normalize("NFC").
- Non-Latin scripts: subject extraction falls back to whole-text comparison.

### 11.7 Very long texts
- Texts longer than 10,000 characters are truncated for subject extraction.
- The truncation is for performance only; the full text is preserved in the
  SubjectExtraction.raw_match field (up to the extraction point).

---

## 12. Performance Constraints

- Subject extraction: O(P * T) where P = number of patterns, T = text length.
  P is bounded (~50 patterns), T is bounded (10K truncation). Effectively O(T).
- Detection pipeline: O(N * P * T) where N = number of existing memories scanned.
  N is bounded by max_candidates (default 50). Effectively O(N * T).
- No external calls. No embeddings. No LLM. Pure text processing.
- Expected wall time: < 10ms for 50 existing memories of average length 200 chars.
- All public functions are thread-safe. No mutable module-level state. Regex
  patterns are compiled as module-level constants (re.compile is thread-safe in
  CPython; compiled pattern objects are immutable and shareable).

---

## 13. Relation to Recall Pipeline

The recall pipeline (recall.py) does NOT import contradiction.py. Instead,
the contradiction system produces side effects (deactivating memories) that
recall observes through the memory pool:

1. When `resolve_contradictions()` returns superseded_indices, the caller
   (memory store or coordinator) sets those memories' `active` flag to False.
2. When `recall()` runs, it receives only active memories in its input pool.
3. The RecallResult does not know about supersession — it just sees fewer memories.

This is a clean separation: contradiction writes to the store, recall reads
from the store. No coupling between the two modules.

---

## 14. Testing Strategy

### 14.1 Unit tests (per function)

- `extract_subject`: 20+ tests covering each FIELD_EXTRACTOR pattern, fallback,
  normalization, empty input, Unicode, truncation.
- `subject_overlap`: 10+ tests covering Jaccard computation, field_type bonus,
  empty subjects, identical subjects, no overlap.
- `detect_polarity`: 10+ tests covering positive, negative, neutral, mixed signals,
  empty text.
- `extract_action`: 10+ tests covering each instruction prefix, no match, edge cases.
- `detect_contradictions`: 30+ tests covering each ContradictionType, edge cases
  from Section 11, config variations, category interactions.
- `resolve_contradictions`: 10+ tests covering supersession records, deactivation
  indices, timestamp propagation.

### 14.2 Integration tests

- Full pipeline: encode → detect → resolve for realistic memory sequences.
- Category-specific supersession rules (Section 7.3).
- Multi-contradiction scenarios (candidate contradicts 3+ existing memories).

### 14.3 Property-based tests (Hypothesis)

- `detect_contradictions` with random texts never crashes.
- `subject_overlap` is symmetric: overlap(a, b) == overlap(b, a).
- `detect_polarity` always returns a valid Polarity member.
- `resolve_contradictions` returns disjoint superseded and flagged sets.

Estimated total: 100-130 tests.

---

## 15. Future Extensions (out of scope for v1)

- Embedding-based subject matching (currently text-only patterns)
- Multi-hop contradiction detection (A→B→C chains)
- Temporal contradiction ("I used to live in X" doesn't contradict "I live in Y")
- Confidence decay on contradictions (old contradictions lose relevance)
- LLM-assisted disambiguation for ambiguous cases

---

**End of spec.**
