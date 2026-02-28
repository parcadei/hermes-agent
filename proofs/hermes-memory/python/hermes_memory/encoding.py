"""Encoding gate for the Hermes memory system.

Evaluates memory candidates (episode narratives or semantic facts) and decides:
  1. Whether to store (should_store)
  2. Category classification (one of 7 categories)
  3. Initial importance for the dynamics system (initial_importance)

The policy is fail-open: when uncertain, it stores rather than discards.
Primary value is importance seeding and category annotation for the downstream
dynamics system.

Dependencies: stdlib only (dataclasses, logging, re, unicodedata).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "preference",
        "fact",
        "correction",
        "reasoning",
        "instruction",
        "greeting",
        "transactional",
        "unclassified",
    }
)

CATEGORY_IMPORTANCE: dict[str, float] = {
    "correction": 0.9,
    "instruction": 0.85,
    "preference": 0.8,
    "fact": 0.6,
    "unclassified": 0.5,
    "reasoning": 0.4,
    "greeting": 0.0,
    "transactional": 0.0,
}

# Classification disambiguation priority (spec Section 3.2). When multiple
# pattern sets match, the highest-priority category wins. This ordering is
# distinct from CATEGORY_IMPORTANCE: priority determines which category is
# ASSIGNED (e.g., correction beats instruction in classification), while
# importance determines the INITIAL WEIGHT in the dynamics scoring function.
# The two orderings happen to be similar (correction highest in both) but
# serve different purposes -- priority is about semantic primacy of the
# communicative act, importance is about retrieval ranking weight.
PRIORITY_ORDER: list[str] = [
    "correction",
    "instruction",
    "preference",
    "fact",
    "reasoning",
    "transactional",
    "greeting",
    "unclassified",
]

# Minimum confidence assigned when at least one pattern matches.
# Prevents near-zero confidence on clear single-pattern matches (e.g.,
# "I prefer X" matching 1/14 preference patterns would yield raw density
# 0.07, which is misleadingly low). Set to 0.35 based on adversarial review
# finding that raw pattern density scoring produces artificially low
# confidence. At 0.35, single matches are below the default fail-open
# threshold (0.5), while 3+ matches exceed it (0.35 + 0.16 = 0.51).
SINGLE_PATTERN_CONFIDENCE_FLOOR: float = 0.35

# ---------------------------------------------------------------------------
# Emotion keyword sets (dimensional modifier, NOT a category)
#
# Emotional valence is a float modifier on EncodingDecision, not a category.
# Per NLP literature (EMNLP 2021, NRC-VAD), emotion is always *about*
# something else -- "I'm frustrated with TypeScript" is a preference with
# negative valence. These keywords detect affect so downstream systems can
# preserve emotional signal across consolidation cycles.
# ---------------------------------------------------------------------------

POSITIVE_EMOTION_KEYWORDS: list[str] = [
    "love",
    "enjoy",
    "excited",
    "happy",
    "grateful",
    "thrilled",
    "pleased",
    "delighted",
    "appreciate",
    "amazing",
    "wonderful",
    "fantastic",
    "great",
    "awesome",
    "glad",
    "satisfied",
    "cheerful",
    "enthusiastic",
    "optimistic",
    "joyful",
    "comfortable",
    "confident",
    "proud",
    "relieved",
    "hopeful",
]

NEGATIVE_EMOTION_KEYWORDS: list[str] = [
    "hate",
    "frustrated",
    "annoyed",
    "angry",
    "disappointed",
    "upset",
    "irritated",
    "fed up",
    "sick of",
    "tired of",
    "dislike",
    "awful",
    "terrible",
    "horrible",
    "dreadful",
    "miserable",
    "unhappy",
    "worried",
    "anxious",
    "stressed",
    "uncomfortable",
    "confused",
    "overwhelmed",
    "discouraged",
    "dissatisfied",
]

STRONG_POSITIVE_EMOTION_KEYWORDS: list[str] = [
    "passionate",
    "obsessed",
    "ecstatic",
    "overjoyed",
    "elated",
    "euphoric",
    "adore",
    "cherish",
    "blissful",
]

STRONG_NEGATIVE_EMOTION_KEYWORDS: list[str] = [
    "desperate",
    "terrified",
    "furious",
    "enraged",
    "devastated",
    "disgusted",
    "loathe",
    "detest",
    "appalled",
    "horrified",
    "infuriated",
]

KNOWLEDGE_TYPE_TO_CATEGORY: dict[str, str] = {
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
    "reasoning": "reasoning",
}

# ---------------------------------------------------------------------------
# First-person pattern sets (Section 3.1)
# ---------------------------------------------------------------------------

PREFERENCE_PATTERNS: list[str] = [
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

FACT_PATTERNS: list[str] = [
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

CORRECTION_PATTERNS: list[str] = [
    # Compound "no" patterns require comma/period AND corrective context,
    # avoiding false positives on "no problem", "no thanks", "no idea", etc.
    # (Issue 2: bare "no," and "no " compiled to identical \bno\b regex.)
    "no, i ",
    "no, that",
    "no, it ",
    "no, actually",
    "no, what i",
    "no. i ",
    "no. the",
    "no. it ",
    "no. that",
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

INSTRUCTION_PATTERNS: list[str] = [
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

REASONING_CONNECTIVES: list[str] = [
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

GREETING_PATTERNS: list[str] = [
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

TRANSACTIONAL_PATTERNS: list[str] = [
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

# ---------------------------------------------------------------------------
# Third-person pattern sets (Section 3.4)
# ---------------------------------------------------------------------------

THIRD_PERSON_PREFERENCE_PATTERNS: list[str] = [
    # "the user" phrasing
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
    # "they/their" pronoun variants
    "they prefer",
    "they like",
    "they want",
    "their favorite",
    "they enjoy",
    "they dislike",
    "they hate",
    "they love",
    "they tend to",
    "their preference",
    # Generic third-person
    "preferred",
    "preference for",
    "expressed interest in",
    "stated a preference",
    "expressed a dislike",
]

THIRD_PERSON_FACT_PATTERNS: list[str] = [
    # "the user" phrasing
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
    "the user studied",
    "the user graduated",
    "the user speaks",
    # "they/their" pronoun variants
    "they work at",
    "they work as",
    "they live in",
    "they live at",
    "their name",
    "their birthday",
    "their age",
    "their email",
    "their phone",
    "their address",
    "their job",
    "their role is",
    # Generic third-person
    "personal detail",
    "personal information",
    "background information",
    "biographical",
]

THIRD_PERSON_CORRECTION_PATTERNS: list[str] = [
    # "the user" phrasing
    "the user corrected",
    "the user clarified",
    "the user pointed out",
    "the user disagreed",
    "the user noted the error",
    "the user meant",
    "the user said that was wrong",
    "the user said that is incorrect",
    "the user indicated that was wrong",
    "the user said no,",
    "the user objected",
    # "they/their" pronoun variants
    "they corrected",
    "they clarified",
    "they pointed out",
    "they disagreed",
    "they meant",
    "they noted the error",
    "they said no,",
    "they objected",
    # Generic third-person
    "correction",
    "corrected the",
    "clarified that",
    "disagreed with",
    "mistaken",
    "inaccurate",
    "amended",
    "revised",
    "not quite right",
]

THIRD_PERSON_INSTRUCTION_PATTERNS: list[str] = [
    # "the user" phrasing
    "the user instructed",
    "the user requested that",
    "the user asked the assistant to always",
    "the user asked the assistant to never",
    "the user emphasized",
    "the user directed",
    "the user specified",
    "the user wants the assistant to",
    "the user asked to remember",
    "the user asked to keep in mind",
    # "they/their" pronoun variants
    "they instructed",
    "they requested",
    "they emphasized",
    "they directed",
    "they specified",
    # Generic third-person
    "going forward",
    "from now on",
    "persistent instruction",
    "behavioral directive",
]

THIRD_PERSON_REASONING_CONNECTIVES: list[str] = [
    # Shared connectives (same as first-person)
    "because",
    "therefore",
    "consequently",
    "as a result",
    "due to",
    "in order to",
    "which means",
    # "the user" phrasing
    "the user explained that",
    "the user reasoned that",
    "the user's logic",
    # "they/their" pronoun variants
    "they reasoned that",
    "they explained that",
    "their reasoning",
    "their logic",
    # Generic third-person
    "reasoning",
    "the rationale",
    "inference",
    "causal connection",
]

THIRD_PERSON_GREETING_PATTERNS: list[str] = [
    # "the user" phrasing
    "the user thanked",
    "the user greeted",
    "the user said hello",
    "the user said goodbye",
    "the user said good morning",
    "the user said good evening",
    # "they/their" pronoun variants
    "they greeted",
    "they thanked",
    "they said hello",
    "they said goodbye",
    # Generic third-person
    "exchanged greetings",
    "greeted",
    "said hello",
    "opened the conversation",
    "initial pleasantries",
    "expressed gratitude",
    "closed the conversation",
    "said goodbye",
    "farewell",
]

THIRD_PERSON_TRANSACTIONAL_PATTERNS: list[str] = [
    # "the user" phrasing
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
    "the user asked to build",
    "the user asked to start",
    "the user asked to stop",
    "the user asked to restart",
    # "they/their" pronoun variants
    "they asked to run",
    "they asked to deploy",
    "they asked to build",
    "they requested a build",
    # Generic third-person
    "task request",
    "operational request",
]

# ---------------------------------------------------------------------------
# Aggregate pattern dicts for iteration
# ---------------------------------------------------------------------------

FIRST_PERSON_PATTERNS: dict[str, list[str]] = {
    "preference": PREFERENCE_PATTERNS,
    "fact": FACT_PATTERNS,
    "correction": CORRECTION_PATTERNS,
    "instruction": INSTRUCTION_PATTERNS,
    "reasoning": REASONING_CONNECTIVES,
    "greeting": GREETING_PATTERNS,
    "transactional": TRANSACTIONAL_PATTERNS,
}

THIRD_PERSON_PATTERNS: dict[str, list[str]] = {
    "preference": THIRD_PERSON_PREFERENCE_PATTERNS,
    "fact": THIRD_PERSON_FACT_PATTERNS,
    "correction": THIRD_PERSON_CORRECTION_PATTERNS,
    "instruction": THIRD_PERSON_INSTRUCTION_PATTERNS,
    "reasoning": THIRD_PERSON_REASONING_CONNECTIVES,
    "greeting": THIRD_PERSON_GREETING_PATTERNS,
    "transactional": THIRD_PERSON_TRANSACTIONAL_PATTERNS,
}


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _detect_non_english(text: str) -> bool:
    """Lightweight heuristic: if >30% of alpha chars are non-ASCII, likely non-English.

    Uses only character-level analysis -- no external dependencies.
    This prevents systematic linguistic bias where non-English content
    falls through to length-based heuristics that assign biased importance
    values (0.0 for short, 0.6 for long).

    Args:
        text: The text to analyze.

    Returns:
        True if the text is likely non-English based on character analysis.
    """
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    non_ascii = sum(1 for c in alpha_chars if ord(c) > 127)
    return non_ascii / len(alpha_chars) > 0.3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodingConfig:
    """Configuration for the encoding gate.

    Frozen to ensure thread safety and reproducibility.
    All thresholds are tunable without code changes.
    """

    min_reasoning_length: int = 100
    min_reasoning_connectives: int = 2
    max_greeting_length: int = 50
    max_transactional_length: int = 80
    # Added to length thresholds when classifying episode narratives
    # (typically 200-500 chars). Prevents misclassifying substantive
    # episodes as greetings or transactional based on absolute length
    # thresholds calibrated for raw messages. Nemori's EpisodeGenerator
    # inflates text length by 150-300% vs raw messages, so a raw 40-char
    # greeting becomes a 100-160 char narrative that would exceed the
    # raw max_greeting_length (50) without this offset.
    episode_length_offset: int = 150
    confidence_threshold: float = 0.5
    use_word_boundaries: bool = True
    max_input_length: int = 50000


@dataclass
class EncodingDecision:
    """The output of the encoding gate for a single memory candidate.

    Not frozen because it is a one-shot output, not shared state.
    """

    should_store: bool
    category: str
    confidence: float
    reason: str
    initial_importance: float
    # Detected language: "en" for English (patterns matched), "non-en" for
    # detected non-English content, None when uncertain (e.g., pure numbers,
    # empty input, or semantic shortcut where language is not assessed).
    detected_language: str | None = None
    # Emotional valence as a dimensional modifier (not a category).
    # None = no emotion detected; positive float (0.0 to 1.0) = positive
    # emotion; negative float (-1.0 to 0.0) = negative emotion.
    emotional_valence: float | None = None


# ---------------------------------------------------------------------------
# EncodingPolicy
# ---------------------------------------------------------------------------


class EncodingPolicy:
    """Deterministic encoding gate for memory candidates.

    Evaluates episode content and semantic facts to decide whether
    they should be persisted to the memory database.

    Thread-safe: all methods are pure functions of their arguments
    plus the immutable config. No mutable state.
    """

    def __init__(self, config: EncodingConfig | None = None) -> None:
        self.config = config or EncodingConfig()
        self._validate_config(self.config)

        # Pre-compile regex patterns for performance (Fix 1)
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}

        # Cached connective count from classify() to avoid recomputation
        # in _apply_write_policy() (Finding 11).
        self._last_connective_count: int | None = None

        # Compile first-person patterns
        for category, pattern_list in FIRST_PERSON_PATTERNS.items():
            self._compiled_patterns[f"first_{category}"] = [
                re.compile(self._make_pattern_regex(p)) for p in pattern_list
            ]

        # Compile third-person patterns
        for category, pattern_list in THIRD_PERSON_PATTERNS.items():
            self._compiled_patterns[f"third_{category}"] = [
                re.compile(self._make_pattern_regex(p)) for p in pattern_list
            ]

        # Compile emotion keyword patterns (dimensional modifier, not category)
        self._positive_emotion_patterns: list[re.Pattern] = [
            re.compile(self._make_pattern_regex(p))
            for p in POSITIVE_EMOTION_KEYWORDS
        ]
        self._negative_emotion_patterns: list[re.Pattern] = [
            re.compile(self._make_pattern_regex(p))
            for p in NEGATIVE_EMOTION_KEYWORDS
        ]
        self._strong_positive_emotion_patterns: list[re.Pattern] = [
            re.compile(self._make_pattern_regex(p))
            for p in STRONG_POSITIVE_EMOTION_KEYWORDS
        ]
        self._strong_negative_emotion_patterns: list[re.Pattern] = [
            re.compile(self._make_pattern_regex(p))
            for p in STRONG_NEGATIVE_EMOTION_KEYWORDS
        ]

    @staticmethod
    def _validate_config(config: EncodingConfig) -> None:
        """Validate EncodingConfig invariants from spec Section 2.1."""
        if not (0.0 <= config.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], "
                f"got {config.confidence_threshold}"
            )
        if config.min_reasoning_length <= 0:
            raise ValueError(
                f"min_reasoning_length must be > 0, got {config.min_reasoning_length}"
            )
        if config.min_reasoning_connectives < 0:
            raise ValueError(
                f"min_reasoning_connectives must be >= 0, "
                f"got {config.min_reasoning_connectives}"
            )
        if config.max_greeting_length <= 0:
            raise ValueError(
                f"max_greeting_length must be > 0, got {config.max_greeting_length}"
            )
        if config.max_transactional_length <= 0:
            raise ValueError(
                f"max_transactional_length must be > 0, "
                f"got {config.max_transactional_length}"
            )

    # -- public API ----------------------------------------------------------

    def evaluate(
        self,
        episode_content: str,
        metadata: dict | None = None,
    ) -> EncodingDecision:
        """Main API: evaluate a memory candidate."""

        # Step 1: Handle empty/None input
        if episode_content is None or episode_content.strip() == "":
            return EncodingDecision(
                should_store=False,
                category="greeting",
                confidence=1.0,
                reason="Empty input",
                initial_importance=0.0,
                detected_language=None,
            )

        # Fix 2: ReDoS mitigation - truncate very long inputs for classification
        if len(episode_content) > self.config.max_input_length:
            episode_content = episode_content[: self.config.max_input_length]

        # Step 2: Semantic memory shortcut (Fix 6: Safe metadata access)
        source_type = metadata.get("source_type") if metadata else None
        knowledge_type = metadata.get("knowledge_type") if metadata else None
        if source_type == "semantic" and knowledge_type in KNOWLEDGE_TYPE_TO_CATEGORY:
            category = KNOWLEDGE_TYPE_TO_CATEGORY[knowledge_type]
            confidence = 0.85
            logger.debug(
                "Semantic shortcut: knowledge_type='%s' -> category='%s'",
                knowledge_type,
                category,
            )
            should_store = self._apply_write_policy(
                category, episode_content, confidence, metadata
            )
            reason = (
                f"Semantic memory shortcut: "
                f"knowledge_type '{knowledge_type}' -> '{category}'"
            )
            if not should_store and confidence < self.config.confidence_threshold:
                should_store = True
                reason += (
                    f" [fail-open: confidence {confidence:.2f} "
                    f"< {self.config.confidence_threshold}]"
                )
            emotional_valence = self._detect_emotional_valence(episode_content)
            imp = CATEGORY_IMPORTANCE[category]
            if emotional_valence is not None:
                imp = min(1.0, imp + 0.05)
            return EncodingDecision(
                should_store=should_store,
                category=category,
                confidence=confidence,
                reason=reason,
                initial_importance=imp,
                detected_language=None,
                emotional_valence=emotional_valence,
            )

        # Step 3: Classify
        text = episode_content.strip()
        category, confidence = self.classify(text, metadata)

        # Finding 12: Validate category
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"classify() returned invalid category '{category}'. "
                f"Must be one of {VALID_CATEGORIES}"
            )

        logger.debug(
            "Classified text (len=%d) as '%s' with confidence %.2f",
            len(text),
            category,
            confidence,
        )

        # Step 4: Metadata boost (Fix 6: Safe metadata access)
        message_count = 0
        if metadata:
            message_count = metadata.get("message_count", 0)
            if not isinstance(message_count, (int, float)):
                message_count = 0
        if message_count > 5:
            confidence = min(1.0, confidence + 0.1)
        if len(text) > 500:
            confidence = min(1.0, confidence + 0.05)

        # Step 4b: Length-based reclassification (Fix 6: Safe metadata access)
        effective_greeting_len = self.config.max_greeting_length
        effective_transactional_len = self.config.max_transactional_length
        source_type = metadata.get("source_type") if metadata else None
        if source_type == "episode":
            effective_greeting_len += self.config.episode_length_offset
            effective_transactional_len += self.config.episode_length_offset

        if category == "greeting" and len(text) > effective_greeting_len:
            category, confidence = self._reclassify_without(text, "greeting", metadata)
            # Re-apply metadata boost on new confidence
            if message_count > 5:
                confidence = min(1.0, confidence + 0.1)
            if len(text) > 500:
                confidence = min(1.0, confidence + 0.05)

        if category == "transactional" and len(text) > effective_transactional_len:
            category, confidence = self._reclassify_without(
                text, "transactional", metadata
            )
            if message_count > 5:
                confidence = min(1.0, confidence + 0.1)
            if len(text) > 500:
                confidence = min(1.0, confidence + 0.05)

        # Step 5: Apply write policy
        should_store = self._apply_write_policy(category, text, confidence, metadata)
        logger.debug("Write policy for '%s': should_store=%s", category, should_store)

        # Step 6: Fail-open
        reason = self._build_reason(category, confidence, text)
        if not should_store and confidence < self.config.confidence_threshold:
            should_store = True
            reason += (
                f" [fail-open: confidence {confidence:.2f} "
                f"< {self.config.confidence_threshold}]"
            )
            logger.debug(
                "Fail-open triggered: confidence %.2f < threshold %.2f",
                confidence,
                self.config.confidence_threshold,
            )

        # Step 7: Language detection
        # "en" when patterns matched (category assigned by pattern, not fallback),
        # "non-en" when detected as non-English, None when uncertain.
        if category == "unclassified":
            detected_language = "non-en"
        elif _detect_non_english(text):
            # Non-English text, but a pattern matched (patterns take precedence)
            detected_language = "non-en"
        else:
            # Check if any alpha chars exist to distinguish "en" from "uncertain"
            has_alpha = any(c.isalpha() for c in text)
            detected_language = "en" if has_alpha else None

        # Step 8: Emotional valence (dimensional modifier, not a category)
        emotional_valence = self._detect_emotional_valence(text)

        # Step 9: Importance (with emotion boost)
        initial_importance = CATEGORY_IMPORTANCE[category]
        if emotional_valence is not None:
            initial_importance = min(1.0, initial_importance + 0.05)

        # Step 10: Return
        return EncodingDecision(
            should_store=should_store,
            category=category,
            confidence=confidence,
            reason=reason,
            initial_importance=initial_importance,
            detected_language=detected_language,
            emotional_valence=emotional_valence,
        )

    def classify(
        self,
        text: str,
        metadata: dict | None = None,
        _exclude_category: str | None = None,
    ) -> tuple[str, float]:
        """Classify text into a category with confidence.

        This is the extension point for swapping heuristic classification
        with an LLM-based classifier.

        Args:
            text: The text to classify.
            metadata: Optional context for pattern set selection.
            _exclude_category: Internal parameter used by _reclassify_without
                to skip a specific category's patterns. Not part of the public API.

        Returns:
            Tuple of (category, confidence).
        """
        # Fix 5: Unicode normalization to handle Turkish İ/ı, ligatures, etc.
        text_normalized = unicodedata.normalize("NFKC", text)
        text_lower = text_normalized.lower()
        matches: dict[str, int] = {}

        # Always check first-person patterns (using pre-compiled patterns)
        for category, pattern_list in FIRST_PERSON_PATTERNS.items():
            if category == _exclude_category:
                continue
            compiled = self._compiled_patterns.get(f"first_{category}", [])
            count = self._count_matches(text_lower, pattern_list, compiled)
            if count > 0:
                matches[category] = matches.get(category, 0) + count

        # Also check third-person patterns for episode/semantic content (Fix 6: Safe metadata access)
        source_type = metadata.get("source_type") if metadata else None
        if source_type in ("episode", "semantic"):
            for category, pattern_list in THIRD_PERSON_PATTERNS.items():
                if category == _exclude_category:
                    continue
                compiled = self._compiled_patterns.get(f"third_{category}", [])
                count = self._count_matches(text_lower, pattern_list, compiled)
                if count > 0:
                    matches[category] = matches.get(category, 0) + count

        # Cache reasoning connective count for _apply_write_policy (Finding 11)
        self._last_connective_count = matches.get("reasoning", 0)

        if not matches:
            # No patterns matched at all -- fallback based on text length
            effective_greeting_threshold = self.config.max_greeting_length
            if source_type == "episode":
                effective_greeting_threshold += self.config.episode_length_offset

            # Issue 5: Non-English text with no pattern matches gets
            # "unclassified" to avoid systematic linguistic bias where
            # English memories get higher/lower importance via the
            # length-based heuristic (short -> greeting at 0.0, long -> fact at 0.6).
            if (
                _exclude_category != "unclassified"
                and _detect_non_english(text)
            ):
                return ("unclassified", 0.4)

            # When reclassifying, avoid returning the excluded category (Fix 4)
            if _exclude_category == "greeting":
                if len(text) > 100:
                    return ("fact", 0.3)
                else:
                    return ("fact", 0.2)
            if _exclude_category == "transactional":
                if len(text) > 100:
                    return ("fact", 0.3)
                else:
                    return ("greeting", 0.3)

            if len(text) < effective_greeting_threshold:
                return ("greeting", 0.3)
            else:
                return ("fact", 0.2)

        # Pick highest-priority category among matches
        for category in PRIORITY_ORDER:
            if category == _exclude_category:
                continue
            if category in matches:
                matched_count = matches[category]
                total_patterns = len(FIRST_PERSON_PATTERNS[category])
                if source_type in ("episode", "semantic"):
                    total_patterns += len(THIRD_PERSON_PATTERNS[category])

                return (
                    category,
                    self._calibrate_confidence(matched_count, total_patterns),
                )

        # Fallback: return proper category instead of excluded one (Fix 4)
        if len(text) > 100:
            return ("fact", 0.1)
        else:
            return ("greeting", 0.1)

    # -- internal helpers ----------------------------------------------------

    def _count_matches(
        self,
        text_lower: str,
        pattern_list: list[str],
        compiled_patterns: list[re.Pattern] | None = None,
    ) -> int:
        """Count how many patterns match in the text."""
        if self.config.use_word_boundaries:
            if compiled_patterns:
                # Use pre-compiled patterns for performance
                return sum(1 for p in compiled_patterns if p.search(text_lower))
            else:
                # Fallback for cases without pre-compiled patterns
                count = 0
                for p in pattern_list:
                    regex_pattern = self._make_pattern_regex(p)
                    if re.search(regex_pattern, text_lower):
                        count += 1
                return count
        return sum(1 for p in pattern_list if p in text_lower)

    def _make_pattern_regex(self, pattern: str) -> str:
        """Convert a pattern string to a word-boundary-aware regex.

        Strips trailing spaces/punctuation before applying word boundaries
        to avoid issues with patterns like 'no ', 'hi ', 'so '.
        """
        stripped = pattern.strip().rstrip(",")
        return r"\b" + re.escape(stripped) + r"\b"

    def _calibrate_confidence(self, matched_count: int, total_patterns: int) -> float:
        """Compute calibrated confidence from match count and total patterns."""
        raw_density = matched_count / total_patterns
        if matched_count >= 1:
            confidence = max(SINGLE_PATTERN_CONFIDENCE_FLOOR, raw_density)
            confidence = min(1.0, confidence + 0.08 * (matched_count - 1))
        else:
            confidence = raw_density
        return confidence

    def _detect_emotional_valence(self, text: str) -> float | None:
        """Detect emotional valence from text using keyword matching.

        Returns:
            None if no emotion keywords detected.
            A float in [-1.0, 1.0] representing the net emotional valence:
              positive (0.0, 1.0] = positive emotion
              negative [-1.0, 0.0) = negative emotion

        Strong emotion keywords contribute a weight of 2.0 while regular
        keywords contribute 1.0. The raw score is normalized by the total
        weighted keyword count to keep the result in [-1.0, 1.0].
        """
        text_normalized = unicodedata.normalize("NFKC", text)
        text_lower = text_normalized.lower()

        # Count matches across the four emotion keyword sets
        pos_count = sum(
            1 for p in self._positive_emotion_patterns if p.search(text_lower)
        )
        neg_count = sum(
            1 for p in self._negative_emotion_patterns if p.search(text_lower)
        )
        strong_pos_count = sum(
            1 for p in self._strong_positive_emotion_patterns if p.search(text_lower)
        )
        strong_neg_count = sum(
            1 for p in self._strong_negative_emotion_patterns if p.search(text_lower)
        )

        total_matches = pos_count + neg_count + strong_pos_count + strong_neg_count
        if total_matches == 0:
            return None

        # Weighted score: strong keywords count double
        positive_score = pos_count * 1.0 + strong_pos_count * 2.0
        negative_score = neg_count * 1.0 + strong_neg_count * 2.0
        total_weight = positive_score + negative_score

        # Net valence normalized to [-1.0, 1.0]
        valence = (positive_score - negative_score) / total_weight
        return max(-1.0, min(1.0, valence))

    def _apply_write_policy(
        self, category: str, text: str, confidence: float, metadata: dict | None = None
    ) -> bool:
        """Apply the write policy decision table from spec Section 4.1."""
        if category in {"preference", "fact", "correction", "instruction", "unclassified"}:
            return True

        if category == "reasoning":
            # Fix 3: Skip reasoning length check for semantic memories (LLM already validated)
            # Fix 6: Safe metadata access
            source_type = metadata.get("source_type") if metadata else None
            knowledge_type = metadata.get("knowledge_type") if metadata else None
            if (
                source_type == "semantic"
                and knowledge_type in KNOWLEDGE_TYPE_TO_CATEGORY
            ):
                return True

            # Use cached connective count from classify() when available
            # (Finding 11: avoid recomputation)
            if self._last_connective_count is not None:
                connective_count = self._last_connective_count
            else:
                # Fallback: recompute if called outside normal evaluate() flow
                text_normalized = unicodedata.normalize("NFKC", text)
                text_lower = text_normalized.lower()
                compiled = self._compiled_patterns.get("first_reasoning", [])
                connective_count = self._count_matches(
                    text_lower, REASONING_CONNECTIVES, compiled
                )
            return (
                len(text) >= self.config.min_reasoning_length
                and connective_count >= self.config.min_reasoning_connectives
            )

        # greeting, transactional
        return False

    def _reclassify_without(
        self,
        text: str,
        exclude_category: str,
        metadata: dict | None = None,
    ) -> tuple[str, float]:
        """Re-run classification with one category's patterns excluded.

        Delegates to classify() with the _exclude_category parameter
        to avoid duplicating pattern matching logic (Finding 19).
        """
        return self.classify(text, metadata, _exclude_category=exclude_category)

    def _build_reason(self, category: str, confidence: float, text: str) -> str:
        """Build human-readable reason string.

        Includes the category, confidence, matched patterns, and write policy
        for debugging and downstream review (Finding 13).
        """
        # Identify which patterns matched in this category
        text_normalized = unicodedata.normalize("NFKC", text)
        text_lower = text_normalized.lower()

        matched = []
        if category in FIRST_PERSON_PATTERNS:
            compiled = self._compiled_patterns.get(f"first_{category}", [])
            patterns = FIRST_PERSON_PATTERNS[category]
            for pat, comp in zip(patterns, compiled):
                if comp.search(text_lower):
                    matched.append(pat)

        # Determine write policy label
        always_store = {"preference", "fact", "correction", "instruction", "unclassified"}
        never_store = {"greeting", "transactional"}
        if category in always_store:
            policy_label = "ALWAYS store"
        elif category == "reasoning":
            policy_label = "CONDITIONAL store (length + connectives)"
        elif category in never_store:
            policy_label = "NEVER store"
        else:
            policy_label = "unknown policy"

        parts = [f"Classified as '{category}' (confidence: {confidence:.2f})"]
        if matched:
            pattern_str = ", ".join(f"'{p}'" for p in matched[:5])
            parts.append(f"matched patterns: [{pattern_str}]")
        parts.append(f"Policy: {policy_label}.")
        return " ".join(parts)
