"""Contradiction/supersession detection for the Hermes memory system.

Detects when a new memory contradicts or updates an existing memory and manages
the supersession lifecycle: marking the old memory as superseded, linking it to
its replacement, and ensuring recall surfaces only the latest version.

This module sits between encoding (piece 1) and recall (piece 2). When a new
memory is encoded, the contradiction detector scans existing memories for
semantic conflicts. When contradictions are found, the system either supersedes
the old memory (replacing it) or marks both as conflicting for human resolution.

Dependencies: stdlib only (dataclasses, enum, logging, re, unicodedata).
"""

from __future__ import annotations

import enum
import logging
import re
import unicodedata
from dataclasses import dataclass

from hermes_memory.encoding import VALID_CATEGORIES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CATEGORY_WEIGHTS: dict[str, float] = {
    "correction": 1.5,
    "instruction": 1.2,
    "preference": 1.0,
    "fact": 0.8,
    "reasoning": 0.5,
}

# Categories that never participate in contradiction detection.
# These are structurally excluded — not expressible via weights.
# "unclassified" is excluded because the content has not been classified
# into a semantic category, so contradictions cannot be meaningfully assessed.
EXCLUDED_CATEGORIES: frozenset[str] = frozenset({"greeting", "transactional", "unclassified"})

_DEFAULT_CATEGORY_THRESHOLDS: dict[str, float] = {
    "preference": 0.6,
    "fact": 0.7,
    "instruction": 0.7,
    "correction": 0.0,
}

NAME_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "not",
        "from",
        "in",
        "at",
        "going",
        "tired",
        "working",
        "excited",
        "really",
        "very",
        "so",
        "just",
        "also",
        "still",
        "now",
        "here",
        "there",
        "sure",
        "sorry",
        "glad",
        "happy",
        "sad",
        "done",
        "back",
        "ready",
        "able",
        "trying",
        "looking",
        "thinking",
        "feeling",
        "getting",
        "doing",
        "being",
        "having",
        "making",
        "taking",
        "coming",
        "leaving",
        "staying",
        "moving",
        "living",
        "based",
    }
)

POLARITY_KEYWORDS: dict[str, list[str]] = {
    "positive": [
        "like",
        "prefer",
        "enjoy",
        "love",
        "want",
        "favor",
        "always",
        "do",
        "use",
    ],
    "negative": [
        "don't",
        "hate",
        "dislike",
        "avoid",
        "never",
        "stop",
        "not",
        "no longer",
    ],
}

# Correction markers used for DIRECT_NEGATION detection
_CORRECTION_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\bthat'?s wrong\b", re.IGNORECASE),
    re.compile(r"\bcorrection\b", re.IGNORECASE),
    re.compile(r"\bnot quite\b", re.IGNORECASE),
    re.compile(r"\bnot really\b", re.IGNORECASE),
    re.compile(r"\bwrong\b", re.IGNORECASE),
]

# Instruction prefixes for action extraction
_INSTRUCTION_PREFIXES: list[re.Pattern[str]] = [
    re.compile(r"^always\s+", re.IGNORECASE),
    re.compile(r"^never\s+", re.IGNORECASE),
    re.compile(r"^don'?t\s+ever\s+", re.IGNORECASE),
    re.compile(r"^from\s+now\s+on,?\s*", re.IGNORECASE),
    re.compile(r"^remember\s+(?:to|that)\s+", re.IGNORECASE),
    re.compile(r"^don'?t\s+", re.IGNORECASE),
    re.compile(r"^do\s+not\s+", re.IGNORECASE),
    re.compile(r"^stop\s+", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Field Extractors (Section 4.1)
# ---------------------------------------------------------------------------

# Each entry: (compiled_pattern, subject_name, field_type)
FIELD_EXTRACTORS: dict[str, list[tuple[re.Pattern[str], str, str]]] = {
    "location": [
        (
            re.compile(
                r"(?:i (?:live|reside|am based|am located)\s+(?:in|at)\s+)(.+)",
                re.IGNORECASE,
            ),
            "location",
            "location",
        ),
        (
            re.compile(r"(?:my (?:home|residence|address)\s+is\s+)(.+)", re.IGNORECASE),
            "location",
            "location",
        ),
        (
            re.compile(r"(?:i(?:'m| am) (?:from|based in)\s+)(.+)", re.IGNORECASE),
            "location",
            "location",
        ),
        (
            re.compile(
                r"(?:the user (?:lives|resides|is based|is located)\s+(?:in|at)\s+)(.+)",
                re.IGNORECASE,
            ),
            "location",
            "location",
        ),
    ],
    "name": [
        (
            re.compile(r"(?:my name is\s+)(.+)", re.IGNORECASE),
            "name",
            "name",
        ),
        (
            re.compile(r"(?:i(?:'m| am)\s+)(\w+(?:\s+\w+)?)", re.IGNORECASE),
            "name",
            "name",
        ),
        (
            re.compile(r"(?:the user(?:'s)? name is\s+)(.+)", re.IGNORECASE),
            "name",
            "name",
        ),
        (
            re.compile(r"(?:call me\s+)(.+)", re.IGNORECASE),
            "name",
            "name",
        ),
    ],
    "email": [
        (
            re.compile(r"(?:my email(?:\s+address)?\s+is\s+)(\S+@\S+)", re.IGNORECASE),
            "email",
            "email",
        ),
        (
            re.compile(r"(?:the user(?:'s)? email is\s+)(\S+@\S+)", re.IGNORECASE),
            "email",
            "email",
        ),
    ],
    "job": [
        (
            re.compile(
                r"(?:i (?:work|am employed)\s+(?:at|for)\s+)(.+)", re.IGNORECASE
            ),
            "employer",
            "job",
        ),
        (
            re.compile(
                r"(?:i (?:work|am employed)\s+as\s+(?:a|an)\s+)(.+)",
                re.IGNORECASE,
            ),
            "role",
            "job",
        ),
        (
            re.compile(
                r"(?:my (?:job|role|position|title)\s+is\s+)(.+)", re.IGNORECASE
            ),
            "role",
            "job",
        ),
        (
            re.compile(r"(?:the user works\s+(?:at|for|as)\s+)(.+)", re.IGNORECASE),
            "employer",
            "job",
        ),
    ],
    "preference": [
        (
            re.compile(r"(?:i (?:prefer|like|enjoy|love|favor)\s+)(.+)", re.IGNORECASE),
            "preference",
            "preference",
        ),
        (
            re.compile(
                r"(?:i (?:don'?t|dont|do not) (?:like|enjoy|want)\s+)(.+)",
                re.IGNORECASE,
            ),
            "dispreference",
            "preference",
        ),
        (
            re.compile(r"(?:my favorite\s+\w+\s+is\s+)(.+)", re.IGNORECASE),
            "preference",
            "preference",
        ),
        (
            re.compile(r"(?:the user (?:prefers|likes|enjoys)\s+)(.+)", re.IGNORECASE),
            "preference",
            "preference",
        ),
    ],
    "instruction": [
        (
            re.compile(r"(?:always\s+)(.+)", re.IGNORECASE),
            "instruction",
            "instruction",
        ),
        (
            re.compile(r"(?:never\s+)(.+)", re.IGNORECASE),
            "instruction",
            "instruction",
        ),
        (
            re.compile(r"(?:from now on,?\s+)(.+)", re.IGNORECASE),
            "instruction",
            "instruction",
        ),
        (
            re.compile(r"(?:remember (?:to|that)\s+)(.+)", re.IGNORECASE),
            "instruction",
            "instruction",
        ),
    ],
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ContradictionType(enum.Enum):
    """Type of contradiction detected between two memories."""

    DIRECT_NEGATION = "direct_negation"
    VALUE_UPDATE = "value_update"
    PREFERENCE_REVERSAL = "preference_reversal"
    INSTRUCTION_CONFLICT = "instruction_conflict"


class Polarity(enum.Enum):
    """Polarity of a memory text."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SupersessionAction(enum.Enum):
    """Action to take on a detected contradiction."""

    AUTO_SUPERSEDE = "auto_supersede"
    FLAG_CONFLICT = "flag_conflict"
    SKIP = "skip"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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
                              Keys must be from VALID_CATEGORIES and NOT in
                              EXCLUDED_CATEGORIES.
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
    """

    similarity_threshold: float = 0.3
    confidence_threshold: float = 0.7
    max_candidates: int = 50
    category_weights: dict[str, float] | None = None
    enable_auto_supersede: bool = True
    value_pattern_min_tokens: int = 2
    category_thresholds: dict[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate configuration and fill defaults."""
        # Fill default category_weights
        if self.category_weights is None:
            object.__setattr__(self, "category_weights", dict(DEFAULT_CATEGORY_WEIGHTS))
        # Fill default category_thresholds
        if self.category_thresholds is None:
            object.__setattr__(
                self, "category_thresholds", dict(_DEFAULT_CATEGORY_THRESHOLDS)
            )

        # Validate scalar fields
        if not (0.0 < self.similarity_threshold <= 1.0):
            msg = f"similarity_threshold must be in (0.0, 1.0], got {self.similarity_threshold}"
            raise ValueError(msg)
        if not (0.0 < self.confidence_threshold <= 1.0):
            msg = f"confidence_threshold must be in (0.0, 1.0], got {self.confidence_threshold}"
            raise ValueError(msg)
        if self.max_candidates < 1:
            msg = f"max_candidates must be >= 1, got {self.max_candidates}"
            raise ValueError(msg)
        if self.value_pattern_min_tokens < 1:
            msg = f"value_pattern_min_tokens must be >= 1, got {self.value_pattern_min_tokens}"
            raise ValueError(msg)

        # Validate category_weights
        assert self.category_weights is not None  # for type checker
        for key, val in self.category_weights.items():
            if key not in VALID_CATEGORIES:
                msg = f"category_weights key {key!r} not in VALID_CATEGORIES"
                raise ValueError(msg)
            if key in EXCLUDED_CATEGORIES:
                msg = (
                    f"category_weights key {key!r} is in EXCLUDED_CATEGORIES; "
                    f"excluded categories cannot have weights"
                )
                raise ValueError(msg)
            if not (0.0 < val <= 2.0):
                msg = f"category_weights[{key!r}] = {val} not in (0.0, 2.0]"
                raise ValueError(msg)

        # Validate category_thresholds
        assert self.category_thresholds is not None  # for type checker
        for key, val in self.category_thresholds.items():
            if key not in VALID_CATEGORIES:
                msg = f"category_thresholds key {key!r} not in VALID_CATEGORIES"
                raise ValueError(msg)
            if not (0.0 <= val <= 1.0):
                msg = f"category_thresholds[{key!r}] = {val} not in [0.0, 1.0]"
                raise ValueError(msg)


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


# ---------------------------------------------------------------------------
# Module-level default config
# ---------------------------------------------------------------------------

DEFAULT_CONTRADICTION_CONFIG = ContradictionConfig()

# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

_FILLER_WORDS = frozenset({"actually", "basically", "just"})
_ARTICLES = frozenset({"a", "an", "the"})


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace,
    strip punctuation, strip articles and filler words."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip punctuation from text (keep alphanumeric, spaces, @, .)
    text = re.sub(r"[^\w\s@.]", " ", text)
    # Remove dots that are not part of email-like patterns (word.word)
    # Keep dots between non-space characters, remove standalone dots
    text = re.sub(r"(?<!\S)\.", " ", text)  # dot preceded by space/start
    text = re.sub(r"\.(?!\S)", " ", text)  # dot followed by space/end
    # Re-collapse whitespace after punctuation removal
    text = re.sub(r"\s+", " ", text).strip()
    # Strip leading articles
    for article in sorted(_ARTICLES, key=len, reverse=True):
        prefix = article + " "
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    # Strip filler words
    words = text.split()
    words = [w for w in words if w not in _FILLER_WORDS]
    return " ".join(words).strip()


# ---------------------------------------------------------------------------
# Subject extraction (Section 4)
# ---------------------------------------------------------------------------

_MAX_TEXT_LENGTH = 10000


def extract_subject(text: str, category: str | None = None) -> SubjectExtraction:
    """Extract subject and value from a memory text.

    Tries all FIELD_EXTRACTORS patterns, picks the longest match. Falls back
    to extracting first tokens as subject with field_type='unknown'.

    Args:
        text: The memory text to extract from.
        category: Optional category hint (currently unused but reserved).

    Returns:
        SubjectExtraction with extracted fields.
    """
    if not text or not text.strip():
        return SubjectExtraction(
            subject="", value=None, field_type="unknown", raw_match=""
        )

    # Truncate for performance (spec 11.7)
    truncated = text[:_MAX_TEXT_LENGTH]
    normalized = truncated.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)

    # Try all extractors, pick the longest match
    best_match: tuple[str, str | None, str, str, int, str] | None = (
        None  # (subject, value, field_type, raw_match, match_len, value_raw)
    )
    best_is_im_name = False  # track if best match is the "I'm X" name pattern

    for field_type_key, patterns in FIELD_EXTRACTORS.items():
        for pattern, subject_name, field_type in patterns:
            m = pattern.search(normalized)
            if m:
                raw_match = m.group(0)
                value_raw = m.group(1).strip()
                match_len = len(raw_match)

                # Track whether this is the "I'm X" name pattern
                is_im_name = (
                    field_type_key == "name"
                    and subject_name == "name"
                    and ("i'm" in raw_match or "i am" in raw_match)
                    and "my name is" not in raw_match
                    and "call me" not in raw_match
                    and "user" not in raw_match
                )

                if best_match is None or match_len > best_match[4]:
                    value_normalized = _normalize_text(value_raw)
                    best_match = (
                        subject_name,
                        value_normalized,
                        field_type,
                        raw_match,
                        match_len,
                        value_raw,  # store raw value for stop-word check
                    )
                    best_is_im_name = is_im_name

    if best_match is not None:
        subject_name, value, field_type, raw_match, _, value_raw_stored = best_match

        # Apply NAME_STOP_WORDS guard for "I'm X" pattern
        # Check against RAW extracted value (before normalization strips articles)
        if best_is_im_name and value_raw_stored:
            raw_words = value_raw_stored.strip().split()
            first_word = raw_words[0].lower() if raw_words else ""
            if first_word in NAME_STOP_WORDS:
                # Downgrade to unknown
                return SubjectExtraction(
                    subject="unknown",
                    value=None,
                    field_type="unknown",
                    raw_match=raw_match,
                )

        return SubjectExtraction(
            subject=subject_name,
            value=value,
            field_type=field_type,
            raw_match=raw_match,
        )

    # Fallback: first 1-5 meaningful tokens as subject, field_type='unknown'
    clean = _normalize_text(truncated)
    # Strip common correction/filler prefixes for better subject extraction
    _CORRECTION_PREFIXES = [
        "no ",
        "actually ",
        "that s wrong ",
        "that is wrong ",
        "correction ",
        "not quite ",
        "well ",
        "but ",
        "however ",
        "wait ",
        "sorry ",
        "wrong ",
        "not really ",
    ]
    clean_stripped = clean
    changed = True
    while changed:
        changed = False
        for prefix in _CORRECTION_PREFIXES:
            if clean_stripped.startswith(prefix):
                clean_stripped = clean_stripped[len(prefix) :].strip()
                changed = True
    tokens = clean_stripped.split()[:4] if clean_stripped else clean.split()[:4]
    fallback_subject = " ".join(tokens) if tokens else ""

    return SubjectExtraction(
        subject=fallback_subject,
        value=None,
        field_type="unknown",
        raw_match=normalized,
    )


# ---------------------------------------------------------------------------
# Subject overlap (Section 4.4)
# ---------------------------------------------------------------------------


def subject_overlap(a: SubjectExtraction, b: SubjectExtraction) -> float:
    """Compute subject similarity using token-level Jaccard index.

    Returns 0.0 when subjects have no common tokens, 1.0 for identical subjects.
    Exact field_type match gets a 0.2 bonus (clamped to 1.0).
    Same subject string gets 1.0 regardless of tokens.

    Additionally, if one subject is a substring of the other (e.g., "preference"
    inside "dispreference"), the contained subject's tokens are counted as
    matching to avoid false negatives in domain-related subjects.

    Args:
        a: First SubjectExtraction.
        b: Second SubjectExtraction.

    Returns:
        Float in [0.0, 1.0].
    """
    # Empty subjects return 0.0
    if not a.subject or not b.subject:
        return 0.0

    # Same subject string gets 1.0
    if a.subject == b.subject:
        return 1.0

    subj_a_lower = a.subject.lower()
    subj_b_lower = b.subject.lower()

    tokens_a = set(subj_a_lower.split())
    tokens_b = set(subj_b_lower.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    # Substring containment check: if one single-token subject is contained
    # in another single-token subject (e.g., "preference" in "dispreference"),
    # treat them as partially matching.
    if not intersection and len(tokens_a) == 1 and len(tokens_b) == 1:
        word_a = subj_a_lower
        word_b = subj_b_lower
        if word_a in word_b or word_b in word_a:
            # Treat as partial match: use 0.5 as base Jaccard
            jaccard = 0.5
            if a.field_type == b.field_type and a.field_type != "unknown":
                jaccard += 0.2
            return min(1.0, jaccard)

    if not union:
        return 0.0

    jaccard = len(intersection) / len(union)

    # Field type bonus
    if a.field_type == b.field_type and a.field_type != "unknown":
        jaccard += 0.2

    return min(1.0, jaccard)


# ---------------------------------------------------------------------------
# Polarity detection (Section 8)
# ---------------------------------------------------------------------------


def detect_polarity(text: str) -> Polarity:
    """Detect the polarity (positive/negative/neutral) of a memory text.

    Uses keyword matching -- not sentiment analysis. Detects explicit
    assertion vs negation patterns.

    Compound negative patterns like "don't like" count as ONE negative signal
    and the positive keyword within the compound is suppressed.

    Args:
        text: Text to analyze.

    Returns:
        Polarity enum member.
    """
    if not text:
        return Polarity.NEUTRAL

    text_lower = text.lower()

    positive_count = 0
    negative_count = 0

    # Compound negative patterns: these absorb the positive keyword
    # "don't like" = 1 negative, "like" does not also count as positive
    _COMPOUND_NEGATIVES = [
        (r"(?:don'?t|do not|don't)\s+(?:like|enjoy|want|use)", 1),
        (r"\bnever\s+(?:use|do)\b", 1),
        (r"\bno longer\s+(?:use|do|like|enjoy|want)\b", 1),
        (r"\bstop\s+(?:using|doing|liking)\b", 1),
        (r"\bavoid\s+(?:using|doing)\b", 1),
    ]

    # Track positions consumed by compound patterns so we don't double-count
    consumed_spans: list[tuple[int, int]] = []

    # Check compound negative patterns first
    for pattern_str, weight in _COMPOUND_NEGATIVES:
        for m in re.finditer(pattern_str, text_lower):
            negative_count += weight
            consumed_spans.append((m.start(), m.end()))

    def _is_consumed(match_start: int, match_end: int) -> bool:
        """Check if a match position overlaps with any consumed span."""
        for cs, ce in consumed_spans:
            if match_start >= cs and match_end <= ce:
                return True
            # Partial overlap: if the keyword starts within a consumed span
            if cs <= match_start < ce:
                return True
        return False

    # Count remaining multi-word negative patterns
    for keyword in POLARITY_KEYWORDS["negative"]:
        if " " in keyword:
            # Multi-word: substring search
            idx = 0
            while True:
                pos = text_lower.find(keyword, idx)
                if pos == -1:
                    break
                if not _is_consumed(pos, pos + len(keyword)):
                    negative_count += 1
                    consumed_spans.append((pos, pos + len(keyword)))
                idx = pos + 1
        else:
            # Single word: word boundary match
            for m in re.finditer(r"\b" + re.escape(keyword) + r"\b", text_lower):
                if not _is_consumed(m.start(), m.end()):
                    negative_count += 1
                    consumed_spans.append((m.start(), m.end()))

    # Count positive signals, skipping those consumed by negation compounds
    for keyword in POLARITY_KEYWORDS["positive"]:
        if " " in keyword:
            idx = 0
            while True:
                pos = text_lower.find(keyword, idx)
                if pos == -1:
                    break
                if not _is_consumed(pos, pos + len(keyword)):
                    positive_count += 1
                idx = pos + 1
        else:
            for m in re.finditer(r"\b" + re.escape(keyword) + r"\b", text_lower):
                if not _is_consumed(m.start(), m.end()):
                    positive_count += 1

    if positive_count > negative_count:
        return Polarity.POSITIVE
    if negative_count > positive_count:
        return Polarity.NEGATIVE
    return Polarity.NEUTRAL


# ---------------------------------------------------------------------------
# Action extraction (Section 9)
# ---------------------------------------------------------------------------


def extract_action(text: str) -> str | None:
    """Extract the action from an instruction text.

    Strips instruction prefixes ("always", "never", "remember to", etc.)
    and returns the remaining action phrase, normalized.

    Args:
        text: Instruction text.

    Returns:
        Normalized action string, or None if no instruction pattern found.
    """
    if not text or not text.strip():
        return None

    normalized = text.strip()

    for prefix_pattern in _INSTRUCTION_PREFIXES:
        m = prefix_pattern.search(normalized)
        if m:
            action = normalized[m.end() :].strip()
            action = action.lower().strip()
            # Strip trailing punctuation
            action = action.rstrip(".,;:!?")
            if not action:
                return None
            return action

    return None


# ---------------------------------------------------------------------------
# Detection strategies (Section 5)
# ---------------------------------------------------------------------------


def _detect_direct_negation(
    candidate_text: str,
    candidate_category: str,
    existing_text: str,
    _existing_category: str,
    candidate_subj: SubjectExtraction,
    _existing_subj: SubjectExtraction,
    overlap: float,
    category_weight: float,
) -> float:
    """DIRECT_NEGATION strategy (Section 5.1).

    Returns confidence, or 0.0 if not applicable.
    """
    # Candidate must be correction category or contain correction markers
    is_correction = candidate_category == "correction"

    # Count correction markers in candidate text
    text_lower = candidate_text.lower()
    marker_count = 0
    for marker in _CORRECTION_MARKERS:
        if marker.search(text_lower):
            marker_count += 1

    if not is_correction and marker_count == 0:
        return 0.0

    # correction_signal_strength: 1.0 for explicit correction category,
    # else based on marker count
    if is_correction:
        correction_signal_strength = max(1.0, marker_count / 3.0)
        correction_signal_strength = min(correction_signal_strength, 1.0)
        # If correction category AND has markers, keep at 1.0
        correction_signal_strength = 1.0
    else:
        correction_signal_strength = min(1.0, marker_count / 3.0)
        if correction_signal_strength == 0.0:
            correction_signal_strength = 0.7  # implicit negation

    confidence = min(1.0, overlap * category_weight * correction_signal_strength)
    return confidence


def _detect_value_update(
    candidate_subj: SubjectExtraction,
    existing_subj: SubjectExtraction,
    overlap: float,
    _category_weight: float,
) -> float:
    """VALUE_UPDATE strategy (Section 5.2).

    Returns confidence, or 0.0 if not applicable.
    """
    # Both must have non-None values AND same field_type AND values differ
    if candidate_subj.value is None or existing_subj.value is None:
        return 0.0

    # Instruction field_type is handled by INSTRUCTION_CONFLICT, not VALUE_UPDATE
    if (
        candidate_subj.field_type == "instruction"
        and existing_subj.field_type == "instruction"
    ):
        return 0.0

    if candidate_subj.field_type != existing_subj.field_type:
        # Different field types: lower confidence
        field_type_match = False
    else:
        field_type_match = True

    # Normalize values for comparison
    cand_val = _normalize_text(candidate_subj.value)
    exist_val = _normalize_text(existing_subj.value)

    # Same value -> not a contradiction
    if cand_val == exist_val:
        return 0.0

    confidence = min(1.0, overlap * (1.0 if field_type_match else 0.5))

    # Value containment check (applied BEFORE boost per spec 5.2)
    has_containment = False
    if cand_val and exist_val:
        if cand_val in exist_val or exist_val in cand_val:
            has_containment = True
            confidence *= 0.5

    # Special case: personal data fields get confidence boost
    # But NOT when value containment was detected (spec: containment check
    # applies BEFORE this boost, and containment means same entity)
    if candidate_subj.field_type in {"email", "name", "location"} and field_type_match:
        if not has_containment:
            confidence = max(confidence, 0.8)

    return confidence


def _detect_preference_reversal(
    candidate_text: str,
    candidate_category: str,
    existing_text: str,
    existing_category: str,
    candidate_subj: SubjectExtraction,
    existing_subj: SubjectExtraction,
    overlap: float,
) -> float:
    """PREFERENCE_REVERSAL strategy (Section 5.3).

    Returns confidence, or 0.0 if not applicable.
    """
    # At least one must be preference category (or correction of preference)
    is_pref_context = (
        candidate_category == "preference"
        or existing_category == "preference"
        or (candidate_category == "correction" and existing_category == "preference")
    )
    if not is_pref_context:
        return 0.0

    # Detect polarity of both texts
    cand_polarity = detect_polarity(candidate_text)
    exist_polarity = detect_polarity(existing_text)

    # Compute polarity_diff
    if cand_polarity != Polarity.NEUTRAL and exist_polarity != Polarity.NEUTRAL:
        if cand_polarity != exist_polarity:
            polarity_diff = 1.0  # opposite polarities
        else:
            # Same polarity but potentially different values
            if candidate_subj.value and existing_subj.value:
                cand_val = _normalize_text(candidate_subj.value)
                exist_val = _normalize_text(existing_subj.value)
                if cand_val != exist_val:
                    polarity_diff = 0.6
                else:
                    polarity_diff = 0.0
            else:
                polarity_diff = 0.0
    elif cand_polarity != exist_polarity and (
        cand_polarity != Polarity.NEUTRAL and exist_polarity != Polarity.NEUTRAL
    ):
        polarity_diff = 1.0
    else:
        # One or both neutral but different values
        if candidate_subj.value and existing_subj.value:
            cand_val = _normalize_text(candidate_subj.value)
            exist_val = _normalize_text(existing_subj.value)
            if cand_val != exist_val:
                polarity_diff = 0.6
            else:
                polarity_diff = 0.0
        else:
            polarity_diff = 0.0

    if polarity_diff == 0.0:
        return 0.0

    confidence = min(1.0, overlap * polarity_diff)
    return confidence


def _detect_instruction_conflict(
    candidate_text: str,
    candidate_category: str,
    existing_text: str,
    existing_category: str,
    candidate_subj: SubjectExtraction,
    existing_subj: SubjectExtraction,
    overlap: float,
) -> float:
    """INSTRUCTION_CONFLICT strategy (Section 5.4).

    Returns confidence, or 0.0 if not applicable.
    """
    # Both must be instruction category (or correction of instruction)
    is_inst_context = (
        (candidate_category == "instruction" and existing_category == "instruction")
        or (candidate_category == "correction" and existing_category == "instruction")
        or (candidate_category == "instruction" and existing_category == "correction")
    )
    if not is_inst_context:
        return 0.0

    # Extract actions from both
    cand_action = extract_action(candidate_text)
    exist_action = extract_action(existing_text)

    if cand_action is None and exist_action is None:
        return 0.0

    # Determine instruction polarity
    cand_polarity = _instruction_polarity(candidate_text)
    exist_polarity = _instruction_polarity(existing_text)

    # Same action + opposite polarity -> high confidence
    if cand_action and exist_action:
        cand_tokens = set(cand_action.lower().split())
        exist_tokens = set(exist_action.lower().split())

        # Check if actions overlap significantly
        if cand_tokens and exist_tokens:
            action_intersection = cand_tokens & exist_tokens
            action_union = cand_tokens | exist_tokens
            action_overlap = (
                len(action_intersection) / len(action_union) if action_union else 0.0
            )
        else:
            action_overlap = 0.0

        # Elaboration guard: if one action's tokens are a subset of the other's
        is_elaboration = False
        if cand_tokens and exist_tokens:
            if cand_tokens.issubset(exist_tokens) or exist_tokens.issubset(cand_tokens):
                is_elaboration = True

        if action_overlap > 0.3:
            if cand_polarity != exist_polarity:
                # Same action + opposite polarity
                confidence = 0.9
            else:
                # Same subject + different actions (or same polarity different instructions)
                confidence = 0.7
        else:
            # Different actions with some subject overlap
            confidence = 0.0
            return 0.0

        confidence = min(1.0, overlap * confidence)

        # Elaboration guard
        if is_elaboration and cand_polarity == exist_polarity:
            confidence = min(confidence, 0.2)

        return confidence

    return 0.0


def _instruction_polarity(text: str) -> str:
    """Determine instruction polarity: 'positive' or 'negative'.

    'Always X' -> positive, 'Never X' -> negative, etc.
    """
    text_lower = text.lower().strip()
    if text_lower.startswith("never "):
        return "negative"
    if text_lower.startswith("don't ") or text_lower.startswith("dont "):
        return "negative"
    if text_lower.startswith("do not "):
        return "negative"
    if text_lower.startswith("stop "):
        return "negative"
    # Check for negation within "from now on" instructions
    if "not " in text_lower or "don't" in text_lower or "never" in text_lower:
        return "negative"
    return "positive"


# ---------------------------------------------------------------------------
# Main detection pipeline (Section 6)
# ---------------------------------------------------------------------------


def _make_empty_result() -> ContradictionResult:
    """Build an empty ContradictionResult."""
    return ContradictionResult(
        detections=(),
        actions=(),
        superseded_indices=frozenset(),
        flagged_indices=frozenset(),
        has_contradiction=False,
        highest_confidence=0.0,
    )


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
    # --- Input validation ---
    if not isinstance(candidate_text, str):
        msg = f"candidate_text must be str, got {type(candidate_text).__name__}"
        raise TypeError(msg)

    for i, t in enumerate(existing_texts):
        if not isinstance(t, str):
            msg = f"existing_texts[{i}] must be str, got {type(t).__name__}"
            raise TypeError(msg)

    if len(existing_texts) != len(existing_categories):
        msg = (
            f"existing_texts and existing_categories must have same length, "
            f"got {len(existing_texts)} vs {len(existing_categories)}"
        )
        raise ValueError(msg)

    if candidate_category not in VALID_CATEGORIES:
        msg = f"candidate_category {candidate_category!r} not in VALID_CATEGORIES: {VALID_CATEGORIES}"
        raise ValueError(msg)

    for i, cat in enumerate(existing_categories):
        if cat not in VALID_CATEGORIES:
            msg = f"existing_categories[{i}] = {cat!r} not in VALID_CATEGORIES: {VALID_CATEGORIES}"
            raise ValueError(msg)

    if config is None:
        config = DEFAULT_CONTRADICTION_CONFIG

    # --- Early exit for empty inputs ---
    if not candidate_text.strip():
        return _make_empty_result()

    if not existing_texts:
        return _make_empty_result()

    # --- Early exit for excluded candidate category ---
    if candidate_category in EXCLUDED_CATEGORIES:
        return _make_empty_result()

    assert config.category_weights is not None
    candidate_weight = config.category_weights.get(candidate_category, 1.0)

    # --- Extract subject from candidate ---
    candidate_subj = extract_subject(candidate_text, candidate_category)

    # --- Scan existing memories ---
    detections: list[ContradictionDetection] = []

    scan_limit = min(len(existing_texts), config.max_candidates)

    for i in range(scan_limit):
        existing_text = existing_texts[i]
        existing_cat = existing_categories[i]

        # Skip empty existing memories
        if not existing_text or not existing_text.strip():
            continue

        # Skip exact duplicates (spec 11.2)
        if existing_text.strip().lower() == candidate_text.strip().lower():
            continue

        # Skip excluded existing categories
        if existing_cat in EXCLUDED_CATEGORIES:
            continue

        # Extract subject from existing
        existing_subj = extract_subject(existing_text, existing_cat)

        # Compute subject overlap
        overlap = subject_overlap(candidate_subj, existing_subj)

        # Skip if below threshold
        if overlap < config.similarity_threshold:
            continue

        # Try ALL 4 detection strategies, collect results
        strategy_results: list[tuple[ContradictionType, float, str]] = []

        # 1. DIRECT_NEGATION
        dn_conf = _detect_direct_negation(
            candidate_text,
            candidate_category,
            existing_text,
            existing_cat,
            candidate_subj,
            existing_subj,
            overlap,
            candidate_weight,
        )
        if dn_conf > 0:
            strategy_results.append(
                (
                    ContradictionType.DIRECT_NEGATION,
                    dn_conf,
                    f"Direct negation detected: candidate corrects existing memory at index {i}",
                )
            )

        # 2. VALUE_UPDATE
        vu_conf = _detect_value_update(
            candidate_subj,
            existing_subj,
            overlap,
            candidate_weight,
        )
        if vu_conf > 0:
            strategy_results.append(
                (
                    ContradictionType.VALUE_UPDATE,
                    vu_conf,
                    (
                        f"Value update: {candidate_subj.field_type} changed from "
                        f"{existing_subj.value!r} to {candidate_subj.value!r}"
                    ),
                )
            )

        # 3. PREFERENCE_REVERSAL
        pr_conf = _detect_preference_reversal(
            candidate_text,
            candidate_category,
            existing_text,
            existing_cat,
            candidate_subj,
            existing_subj,
            overlap,
        )
        if pr_conf > 0:
            strategy_results.append(
                (
                    ContradictionType.PREFERENCE_REVERSAL,
                    pr_conf,
                    f"Preference reversal: polarity changed between candidate and existing at index {i}",
                )
            )

        # 4. INSTRUCTION_CONFLICT
        ic_conf = _detect_instruction_conflict(
            candidate_text,
            candidate_category,
            existing_text,
            existing_cat,
            candidate_subj,
            existing_subj,
            overlap,
        )
        if ic_conf > 0:
            strategy_results.append(
                (
                    ContradictionType.INSTRUCTION_CONFLICT,
                    ic_conf,
                    f"Instruction conflict: candidate conflicts with existing instruction at index {i}",
                )
            )

        # Take highest confidence detection for this pair
        if strategy_results:
            strategy_results.sort(key=lambda x: x[1], reverse=True)
            best_type, best_conf, best_explanation = strategy_results[0]
            det = ContradictionDetection(
                existing_index=i,
                contradiction_type=best_type,
                confidence=best_conf,
                subject_overlap=overlap,
                candidate_subject=candidate_subj,
                existing_subject=existing_subj,
                explanation=best_explanation,
            )
            detections.append(det)

    # --- Sort detections by confidence descending ---
    detections.sort(key=lambda d: d.confidence, reverse=True)

    # --- Dedup: if multiple detections target same existing_index, keep highest ---
    seen_indices: set[int] = set()
    deduped: list[ContradictionDetection] = []
    for det in detections:
        if det.existing_index not in seen_indices:
            deduped.append(det)
            seen_indices.add(det.existing_index)
    detections = deduped

    if not detections:
        return _make_empty_result()

    # --- Determine actions ---
    assert config.category_thresholds is not None
    threshold = config.category_thresholds.get(
        candidate_category, config.confidence_threshold
    )

    actions: list[tuple[int, SupersessionAction]] = []
    superseded: set[int] = set()
    flagged: set[int] = set()

    for det in detections:
        if threshold == 0.0 and config.enable_auto_supersede:
            # Always auto-supersede (correction category)
            actions.append((det.existing_index, SupersessionAction.AUTO_SUPERSEDE))
            superseded.add(det.existing_index)
        elif det.confidence >= threshold and config.enable_auto_supersede:
            actions.append((det.existing_index, SupersessionAction.AUTO_SUPERSEDE))
            superseded.add(det.existing_index)
        elif det.confidence >= threshold and not config.enable_auto_supersede:
            actions.append((det.existing_index, SupersessionAction.FLAG_CONFLICT))
            flagged.add(det.existing_index)
        else:
            # Below threshold
            actions.append((det.existing_index, SupersessionAction.FLAG_CONFLICT))
            flagged.add(det.existing_index)

    highest_confidence = max(d.confidence for d in detections) if detections else 0.0

    return ContradictionResult(
        detections=tuple(detections),
        actions=tuple(actions),
        superseded_indices=frozenset(superseded),
        flagged_indices=frozenset(flagged),
        has_contradiction=True,
        highest_confidence=highest_confidence,
    )


# ---------------------------------------------------------------------------
# Supersession resolution (Section 7)
# ---------------------------------------------------------------------------


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
    if not isinstance(result, ContradictionResult):
        msg = f"result must be ContradictionResult, got {type(result).__name__}"
        raise TypeError(msg)

    records: list[SupersessionRecord] = []
    deactivated: set[int] = set()

    # Build a map from index -> detection for explanation lookup
    detection_map: dict[int, ContradictionDetection] = {}
    for det in result.detections:
        if det.existing_index not in detection_map:
            detection_map[det.existing_index] = det

    for idx, action in result.actions:
        if action == SupersessionAction.AUTO_SUPERSEDE:
            det = detection_map.get(idx)
            explanation = det.explanation if det else "superseded"
            contradiction_type = (
                det.contradiction_type if det else ContradictionType.VALUE_UPDATE
            )
            confidence = det.confidence if det else 0.0

            record = SupersessionRecord(
                old_index=idx,
                new_index=-1,
                contradiction_type=contradiction_type,
                confidence=confidence,
                timestamp=timestamp,
                explanation=explanation,
            )
            records.append(record)
            deactivated.add(idx)

    return records, frozenset(deactivated)
