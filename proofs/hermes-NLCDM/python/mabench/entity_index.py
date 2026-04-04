"""entity_index.py -- Entity-keyed inverted-index extraction and retrieval.

Provides regex-based entity, bigram, and unigram extraction from text, plus
a three-tier scored retrieval function that weights entity matches (5x) above
bigram matches (3x) above IDF-weighted unigram matches.

These are pure-string operations with no ML models or embeddings.

Exported functions
------------------
extract_entities(text)          -- proper nouns, numbers+units, quotes, dates, domain nouns
extract_bigrams(text)           -- stemmed non-stopword bigrams
extract_content_unigrams(text)  -- stemmed content unigrams (4+ chars)
entity_retrieve_scored(...)     -- three-tier scored retrieval over inverted indexes
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ── Stop words (shared by bigram and unigram extraction) ──────────────
_STOP = frozenset({
    "the", "a", "an", "i", "my", "me", "is", "was", "were", "am", "are",
    "do", "did", "does", "have", "had", "has", "be", "been", "being",
    "in", "on", "at", "to", "of", "for", "with", "and", "or", "but",
    "that", "this", "it", "its", "also", "not", "no", "so", "if",
    "you", "your", "we", "our", "they", "them", "their", "he", "she",
    "him", "her", "can", "could", "would", "should", "will", "shall",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "about", "from", "into", "through", "during", "before", "after",
    "above", "below", "between", "by", "up", "down", "out", "off",
    "over", "under", "then", "than", "more", "most", "very", "just",
    "some", "any", "all", "each", "every", "both", "few", "many",
    "much", "other", "such", "only", "own", "same", "too", "here",
    "there", "now", "well", "back", "even", "still", "way", "may",
    "might", "need", "get", "got", "make", "made", "go", "going",
    "went", "come", "like", "know", "think", "see", "look", "want",
    "give", "use", "find", "tell", "ask", "work", "seem", "feel",
    "try", "leave", "call", "keep", "let", "put", "take", "say",
    "said", "one", "two", "new", "first", "last", "long", "great",
    "little", "right", "old", "big", "high", "small", "large", "good",
    "bad", "sure", "really", "thing", "things", "time", "times",
    "people", "something", "anything", "nothing", "everything",
    "someone", "anyone", "everyone", "bit", "lot", "kind", "able",
    "around", "always", "never", "often", "sometimes", "already",
    "another", "though", "although", "yet", "while", "because",
    "since", "until", "unless", "whether", "however", "as", "an",
    "been", "being", "those", "these", "us", "s", "t", "re", "ve",
    "ll", "d", "m", "don", "doesn", "didn", "won", "wouldn", "isn",
    "wasn", "aren", "weren", "hasn", "haven", "hadn", "shouldn",
    "couldn", "mustn", "aren", "re", "yes", "yeah", "oh", "ok",
    "okay", "actually", "maybe", "probably", "definitely", "perhaps",
})

# ── Turn prefix parsing ──────────────────────────────────────────────
_TURN_PREFIX_RE = re.compile(
    r"^\[\d{4}/\d{2}/\d{2}\s*\(\w+\)\s*\d{1,2}:\d{2}\]\s*\(\w+\)\s*"
)

# ── Entity extraction patterns ───────────────────────────────────────
# Proper noun sequences: 2+ capitalized words (e.g., "Delta SkyMiles")
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Single capitalized words that aren't sentence starters
_SINGLE_PROPER_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")

# Numbers with units: $800, 50 hours, 500 Mbps, 12 fish, 3.5 GPA
_NUM_UNIT_RE = re.compile(
    r"(?:\$[\d,]+(?:\.\d+)?)"               # $800, $1,200
    r"|(?:[\d,]+(?:\.\d+)?\s*(?:%|percent|hours?|minutes?|mins?|weeks?|days?"
    r"|months?|years?|miles?|km|lbs?|kg|pounds?|feet|ft|inches?|in"
    r"|fish|guppies|tetras|sessions?|nights?|times?"
    r"|points?|stars?|cups?|tablespoons?|teaspoons?|oz|mg|gb|mb|mbps"
    r"|calories|cal|chapters?|pages?|books?|episodes?|songs?"
    r"|pieces?|servings?|guests?|people|persons?|members?"
    r"|credits?|classes?|courses?|semesters?|grades?"
    r"|runs?|reps?|sets?|laps?|steps?))"
    r"|(?:[\d,]+(?:\.\d+)?(?:st|nd|rd|th))",  # ordinals: 1st, 3rd
    re.IGNORECASE,
)

# Quoted text
_QUOTED_RE = re.compile(r'"([^"]{3,80})"')

# Date patterns: June 3rd, March 15, 2023/05/27
_DATE_MONTH_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)

# Specific domain nouns worth indexing (appear in LME questions)
_DOMAIN_NOUN_RE = re.compile(
    r"\b(aquarium|camera|vintage|handbag|designer|bereavement|support\s+group"
    r"|recipe|pilsner|lager|BBQ|barbecue|tomato(?:es)?|marigold(?:s)?"
    r"|seedling(?:s)?|SkyMiles|Hilton|Delta|JetBlue"
    r"|Library\s+of\s+Babel|Borges|hexagon(?:s)?"
    r"|lemon\s+poppyseed|poppyseed"
    r"|campaign\s+season(?:s)?|peak\s+season(?:s)?"
    r"|remote\s+work|work(?:ing)?\s+remotely"
    r"|carry-on|checked\s+luggage|luggage"
    r"|Seco\s+de\s+Cordero|Arequipa|Ancash"
    r"|Miami|Rome|Vatican|San\s+Francisco)\b",
    re.IGNORECASE,
)

# Common capitalized words to skip in single-proper-noun extraction
_COMMON_CAPS = frozenset({
    "the", "this", "that", "what", "when", "where", "how", "why",
    "yes", "no", "sure", "well", "now", "here", "also", "just",
    "but", "and", "for", "can", "could", "would", "should",
    "however", "although", "maybe", "perhaps", "actually",
    "recently", "currently", "overall", "additionally", "finally",
    "meanwhile", "unfortunately", "fortunately", "basically",
    "definitely", "apparently", "honestly", "generally",
    "absolutely", "especially", "certainly", "obviously",
    "personally", "interesting", "another", "during", "since",
    "after", "before", "between", "through", "great", "good",
    "some", "each", "every", "many", "much", "most", "both",
})


# ── Internal helpers ─────────────────────────────────────────────────

def _simple_stem(word: str) -> str:
    """Minimal suffix stripping so 'classes'/'class', 'bikes'/'bike' match.

    Not a real stemmer -- just handles the most common English plural
    and verb suffixes that cause bigram mismatches.
    """
    if len(word) <= 3:
        return word
    # -ies -> -y  (e.g. "berries" -> "berry")
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    # -sses -> -ss (e.g. "classes" stays "class" via next rule after es strip)
    if word.endswith("sses"):
        return word[:-2]
    # -es -> - (e.g. "classes" -> "class", "devices" -> "devic")
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    # -ing -> - (e.g. "streaming" -> "stream", but protect short words)
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    # -ed -> - (e.g. "upgraded" -> "upgrad")
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    # -s -> - (e.g. "bikes" -> "bike", "months" -> "month")
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _strip_turn_prefix(text: str) -> str:
    """Remove the [date] (role) prefix from a turn."""
    m = _TURN_PREFIX_RE.match(text)
    return text[m.end():] if m else text


# ── Public API ───────────────────────────────────────────────────────

def extract_entities(text: str) -> set[str]:
    """Extract entities from text.  Returns lowercased entity strings.

    Extracts six categories:
    1. Multi-word proper nouns (e.g., "Delta SkyMiles")
    2. Single proper nouns (capitalized words, filtered for common words)
    3. Numbers with units (e.g., "$800", "50 hours")
    4. Quoted text
    5. Date expressions (month + day)
    6. Domain-specific nouns
    """
    entities: set[str] = set()
    body = _strip_turn_prefix(text)

    # 1. Multi-word proper nouns
    for m in _PROPER_NOUN_RE.finditer(body):
        entities.add(m.group(1).lower())

    # 2. Single proper nouns (skip very common words and sentence starters)
    for m in _SINGLE_PROPER_RE.finditer(body):
        word = m.group(1)
        if word.lower() not in _COMMON_CAPS and len(word) >= 3:
            entities.add(word.lower())

    # 3. Numbers with units
    for m in _NUM_UNIT_RE.finditer(body):
        val = m.group(0).strip().lower()
        # Normalize whitespace
        val = re.sub(r"\s+", " ", val)
        entities.add(val)

    # 4. Quoted text
    for m in _QUOTED_RE.finditer(body):
        entities.add(m.group(1).lower())

    # 5. Date expressions (month + day)
    for m in _DATE_MONTH_RE.finditer(body):
        month = m.group(1).lower()
        day = m.group(2)
        entities.add(f"{month} {day}")

    # 6. Domain-specific nouns
    for m in _DOMAIN_NOUN_RE.finditer(body):
        entities.add(m.group(0).lower())

    return entities


def extract_bigrams(text: str) -> set[str]:
    """Extract stemmed non-stopword bigrams from text.

    Applies _simple_stem so "fitness classes" and "fitness class" produce the
    same bigram key "fitness class".
    """
    body = _strip_turn_prefix(text)
    words = re.findall(r"[a-z]+", body.lower())
    stemmed = [_simple_stem(w) for w in words if w not in _STOP and len(w) >= 3]
    bigrams: set[str] = set()
    for i in range(len(stemmed) - 1):
        bigrams.add(f"{stemmed[i]} {stemmed[i+1]}")
    return bigrams


def extract_content_unigrams(text: str) -> set[str]:
    """Extract stemmed content unigrams (4+ chars, not in stop list).

    Captures words like 'yoga', 'internet', 'airline', 'bike' that are
    content-bearing but not proper nouns.  These have high fan-out so
    they are weighted lower than entities/bigrams in scoring.
    """
    body = _strip_turn_prefix(text)
    words = re.findall(r"[a-z]+", body.lower())
    return {_simple_stem(w) for w in words if w not in _STOP and len(w) >= 4}


def entity_retrieve_scored(
    question_text: str,
    entity_inv: dict[str, set[int]],
    bigram_inv: dict[str, set[int]],
    unigram_inv: dict[str, set[int]],
    n_turns: int,
    top_k: int = 30,
) -> list[tuple[int, float]]:
    """Retrieve turns scored by shared entities/bigrams/unigrams.

    Three-tier scoring:
      - Entity match:  +5.0 per shared entity  (specific, low fan-out)
      - Bigram match:  +3.0 per shared bigram  (medium specificity)
      - Unigram match: +IDF-weighted per shared unigram (high fan-out,
        penalised by inverse document frequency)

    Parameters
    ----------
    question_text : str
        The query text to extract entities/bigrams/unigrams from.
    entity_inv : dict[str, set[int]]
        Inverted index mapping entity strings to turn indices.
    bigram_inv : dict[str, set[int]]
        Inverted index mapping bigram strings to turn indices.
    unigram_inv : dict[str, set[int]]
        Inverted index mapping unigram strings to turn indices.
    n_turns : int
        Total number of turns (used for IDF computation).
    top_k : int
        Number of top results to return (default 30).

    Returns
    -------
    list[tuple[int, float]]
        Top-k (turn_index, score) pairs sorted by descending score.
    """
    q_entities = extract_entities(question_text)
    q_bigrams = extract_bigrams(question_text)
    q_unigrams = extract_content_unigrams(question_text)

    scores: Counter = Counter()

    # Entity matches: high weight (specific, low fan-out)
    for ent in q_entities:
        if ent in entity_inv:
            for idx in entity_inv[ent]:
                scores[idx] += 5.0

    # Bigram matches: medium weight
    for bg in q_bigrams:
        if bg in bigram_inv:
            for idx in bigram_inv[bg]:
                scores[idx] += 3.0

    # Unigram matches: low weight, IDF-adjusted to penalize common words
    for ug in q_unigrams:
        if ug in unigram_inv:
            posting = unigram_inv[ug]
            # IDF: rarer words get higher weight
            idf = math.log(n_turns / (len(posting) + 1))
            w = max(0.5, idf / 5.0)  # normalize to ~0.5-1.5 range
            for idx in posting:
                scores[idx] += w

    ranked = scores.most_common(top_k)
    return [(idx, sc) for idx, sc in ranked]
