"""query_parser.py -- Structured query parsing for LME questions.

Decomposes LME question text into structured hints (entities, predicates,
temporal references) for fact-store lookup.  Regex-only, no LLM calls.
Target latency: < 1ms per question.

Exported symbols
----------------
ParsedQuery        -- dataclass holding structured hints
parse_query(text)  -- main entry point
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mabench.event_calendar import _simple_stem, _STOPWORDS
from mabench.outgestion import classify_question_type


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    """Structured representation of a question for fact store lookup."""

    entity_hints: list[str] = field(default_factory=list)
    predicate_hints: list[str] = field(default_factory=list)
    object_hints: list[str] = field(default_factory=list)
    question_type: str = "factual"
    temporal_hint: str | None = None
    raw: str = ""
    query_date_str: str | None = None


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# LME boilerplate: "Based on our (previous) conversations, ..."
_BASED_ON_RE = re.compile(
    r"^Based\s+on\s+our\s+(?:previous\s+)?conversations?,?\s+"
    r"I\s+am\s+looking\s+for\s+very\s+specific\s+information\.\s*",
    re.IGNORECASE,
)

# "Current Date: 2023/05/27 (Sat) 22:54,"
_CURRENT_DATE_RE = re.compile(
    r"Current\s+Date:\s*"
    r"(\d{4}/\d{2}/\d{2}\s*\(\w+\)\s*\d{1,2}:\d{2})"
    r"\s*,?\s*\n*",
    re.IGNORECASE,
)

# "Now Answer the Question: " or "Answer the Question: "
_ANSWER_QUESTION_RE = re.compile(
    r"(?:Now\s+)?Answer\s+the\s+Question:\s*",
    re.IGNORECASE,
)

# Multi-word proper nouns: 2+ capitalized words (e.g., "Imagine Dragons")
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-zA-Z&]+(?:\s+(?:[A-Z][a-zA-Z&]+|de|of|the))*"
    r"(?:\s+[A-Z][a-zA-Z&]+)+)\b"
)

# Quoted phrases (single or double quotes)
_QUOTED_RE = re.compile(r"""['"\u2018\u2019\u201c\u201d]([^'"\u2018\u2019\u201c\u201d]{2,})['"\u2018\u2019\u201c\u201d]""")

# "did I/you <VERB>" pattern
_DID_I_VERB_RE = re.compile(
    r"\b(?:did|do|does)\s+(?:I|you|we)\s+(\w+)",
    re.IGNORECASE,
)

# "have I/you <VERB>ed/en" pattern
_HAVE_I_VERB_RE = re.compile(
    r"\b(?:have|has|had)\s+(?:I|you|we)\s+(?:been\s+)?(\w+)",
    re.IGNORECASE,
)

# "was I/you <VERB>ing" pattern
_WAS_I_VERB_RE = re.compile(
    r"\b(?:was|were|am|is)\s+(?:I|you|we)\s+(\w+)",
    re.IGNORECASE,
)

# "that I/you <VERB>ed" pattern (relative clause)
_THAT_I_VERB_RE = re.compile(
    r"\b(?:that|which|where|when)\s+(?:I|you|we)\s+(\w+)",
    re.IGNORECASE,
)

# "I <VERB>ed" pattern
_I_VERB_RE = re.compile(
    r"\bI\s+(\w+ed)\b",
    re.IGNORECASE,
)

# Temporal patterns
_TEMPORAL_AGO_RE = re.compile(
    r"(\d+\s+(?:day|week|month|year)s?\s+ago"
    r"|how\s+many\s+(?:day|week|month|year)s?\s+ago)",
    re.IGNORECASE,
)
_TEMPORAL_HOW_LONG_RE = re.compile(
    r"(how\s+long\b[^?]*)",
    re.IGNORECASE,
)
_TEMPORAL_LAST_RE = re.compile(
    r"(last\s+(?:week|month|year|sunday|monday|tuesday|wednesday"
    r"|thursday|friday|saturday))",
    re.IGNORECASE,
)
_TEMPORAL_IN_MONTH_RE = re.compile(
    r"\b(in\s+(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December))\b",
    re.IGNORECASE,
)
_TEMPORAL_MONTH_STANDALONE_RE = re.compile(
    r"\b(January|February|March|April|June|July|August|September"
    r"|October|November|December)\b",
    re.IGNORECASE,
)
_TEMPORAL_NAMED_DATE_RE = re.compile(
    r"(Valentine'?s?\s*(?:day)?|Christmas|Halloween|New\s+Year'?s?"
    r"|Independence\s+Day|Thanksgiving)",
    re.IGNORECASE,
)
_TEMPORAL_DAYS_PASSED_RE = re.compile(
    r"((?:how\s+many\s+)?days?\s+(?:had\s+)?passed)",
    re.IGNORECASE,
)
_TEMPORAL_RECENTLY_RE = re.compile(
    r"\b(recently|most\s+recently|currently|right\s+now)\b",
    re.IGNORECASE,
)

# Domain nouns relevant to LME questions (reuses pattern from entity_index)
_DOMAIN_NOUNS = frozenset({
    "aquarium", "aquariums", "camera", "cameras", "vintage", "handbag",
    "designer", "bereavement", "recipe", "pilsner", "lager", "bbq",
    "barbecue", "tomato", "tomatoes", "marigold", "marigolds",
    "seedling", "seedlings", "fish", "guppy", "guppies",
    "tetra", "tetras", "concert", "festival", "museum",
    "gallery", "exhibition", "marathon", "yoga", "meditation",
    "podcast", "newsletter", "subscription", "membership",
    "guitar", "piano", "violin", "pottery", "painting",
    "hiking", "camping", "kayaking", "surfing", "skiing",
    "snowboarding", "scuba", "diving", "climbing",
    "restaurant", "cafe", "bakery", "brewery", "winery",
    "hotel", "resort", "hostel", "airbnb",
    "laptop", "tablet", "smartphone", "headphones",
    "bicycle", "motorcycle", "scooter",
    "puppy", "kitten", "cat", "dog", "parrot", "hamster",
    "garden", "greenhouse", "compost",
    "book", "novel", "manga", "comic",
    "graduation", "wedding", "birthday", "anniversary",
    "internship", "scholarship", "certification",
    "therapist", "counselor", "dentist", "veterinarian",
    "luggage", "suitcase", "backpack",
})

# Single proper noun: capitalized word not at sentence start
_SINGLE_CAP_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")

# Common sentence-start words to skip
_COMMON_CAPS = frozenset({
    "the", "this", "that", "what", "when", "where", "how", "why",
    "which", "who", "whom", "yes", "no", "sure", "well", "now",
    "here", "also", "just", "but", "and", "for", "can", "could",
    "would", "should", "however", "although", "maybe", "perhaps",
    "actually", "recently", "currently", "overall", "additionally",
    "finally", "meanwhile", "unfortunately", "fortunately",
    "basically", "definitely", "apparently", "honestly", "generally",
    "absolutely", "especially", "certainly", "obviously",
    "personally", "interesting", "another", "during", "since",
    "after", "before", "between", "through", "great", "good",
    "some", "each", "every", "many", "much", "most", "both",
    "any", "answer", "question", "based", "looking", "information",
    "previous", "conversations", "not", "now",
})

# Action verbs (curated set for predicate extraction)
_ACTION_VERBS = frozenset({
    "enroll", "enrolled", "enrolling",
    "buy", "bought", "buying",
    "spend", "spent", "spending",
    "attend", "attended", "attending",
    "visit", "visited", "visiting",
    "watch", "watched", "watching",
    "start", "started", "starting",
    "finish", "finished", "finishing",
    "join", "joined", "joining",
    "travel", "traveled", "travelling", "traveling",
    "book", "booked", "booking",
    "order", "ordered", "ordering",
    "recommend", "recommended", "recommending",
    "mention", "mentioned", "mentioning",
    "discuss", "discussed", "discussing",
    "plan", "planned", "planning",
    "receive", "received", "receiving",
    "adopt", "adopted", "adopting",
    "subscribe", "subscribed", "subscribing",
    "register", "registered", "registering",
    "donate", "donated", "donating",
    "volunteer", "volunteered", "volunteering",
    "move", "moved", "moving",
    "switch", "switched", "switching",
    "cancel", "cancelled", "canceling",
    "renew", "renewed", "renewing",
    "lose", "lost",
    "find", "found",
    "read", "reading",
    "complete", "completed", "completing",
    "cook", "cooked", "cooking",
    "bake", "baked", "baking",
    "plant", "planted", "planting",
    "grow", "grew", "growing",
    "paint", "painted", "painting",
    "design", "designed", "designing",
    "build", "built", "building",
    "repair", "repaired", "repairing",
    "fix", "fixed", "fixing",
    "upgrade", "upgraded", "upgrading",
    "sign", "signed", "signing",
    "apply", "applied", "applying",
    "submit", "submitted", "submitting",
    "return", "returned", "returning",
    "exchange", "exchanged", "exchanging",
    "hire", "hired", "hiring",
    "promote", "promoted", "promoting",
    "transfer", "transferred", "transferring",
    "retire", "retired", "retiring",
    "graduate", "graduated", "graduating",
    "teach", "taught", "teaching",
    "study", "studied", "studying",
    "learn", "learned", "learning",
    "train", "trained", "training",
    "exercise", "exercised", "exercising",
    "run", "ran", "running",
    "swim", "swam", "swimming",
    "hike", "hiked", "hiking",
    "climb", "climbed", "climbing",
    "bike", "biked", "biking",
    "drive", "drove", "driving",
    "fly", "flew", "flied", "flying",
    "sail", "sailed", "sailing",
    "rent", "rented", "renting",
    "purchase", "purchased", "purchasing",
    "collect", "collected", "collecting",
})

# "<determiner/adj> <NOUN PHRASE>" — extract multi-word noun phrases
# Matches "the loyalty program", "a support group", or bare "loyalty program"
_NOUN_PHRASE_SUFFIXES = (
    r"program|group|class|club|course|team"
    r"|league|society|committee|organization|association|department"
    r"|company|firm|agency|foundation|institute|academy|school"
    r"|university|college|hospital|clinic|center|park|museum"
    r"|library|gallery|theater|theatre|restaurant|hotel|resort"
    r"|store|shop|market|mall|station|airport|port"
    r"|trail|route|path|road|highway|bridge"
    r"|festival|concert|event|show|ceremony|tournament"
    r"|recipe|book|movie|film|series|album|song"
    r"|car|truck|van|boat|plane"
    r"|plant|flower|tree|garden|greenhouse"
    r"|pet|cat|dog|fish|bird|hamster"
    r"|camera|laptop|tablet|phone"
    r"|bag|handbag|suitcase|backpack"
    r"|diet|workout|routine|schedule"
    r"|support\s+group|loyalty\s+program|reading\s+club"
    r"|book\s+club|hiking\s+club|running\s+club"
)
_THE_NOUN_RE = re.compile(
    r"\b(?:the|a|an|my|your)\s+([a-z][a-z\s]*(?:" + _NOUN_PHRASE_SUFFIXES + r"))\b",
    re.IGNORECASE,
)
# Same pattern but without requiring a determiner — for "What loyalty program..."
_BARE_NOUN_PHRASE_RE = re.compile(
    r"\b([a-z]+\s+(?:" + _NOUN_PHRASE_SUFFIXES + r"))\b",
    re.IGNORECASE,
)

# Brand / special entity patterns (H&M, etc.)
_BRAND_RE = re.compile(
    r"\b(H&M|BMW|AT&T|P&G|M&S|B&Q|H&R\s+Block"
    r"|IKEA|LEGO|NASA|FIFA|NBA|NFL|NHL|MLB)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_lme_boilerplate(text: str) -> tuple[str, str | None]:
    """Remove LME question wrappers, return (bare_question, query_date_str).

    Strips:
    - "Based on our previous conversations, ..."
    - "Current Date: YYYY/MM/DD (Day) HH:MM\\n"
    - "Now Answer the Question: "
    - "Answer the Question: "

    Returns
    -------
    tuple[str, str | None]
        (bare_question, query_date_string_or_None)
    """
    if not text:
        return "", None

    s = text.strip()
    query_date: str | None = None

    # Extract "Current Date: ..."
    m = _CURRENT_DATE_RE.search(s)
    if m:
        query_date = m.group(1).strip()
        s = s[:m.start()] + s[m.end():]

    # Strip "Based on our conversations..."
    s = _BASED_ON_RE.sub("", s).strip()

    # Strip "Now Answer the Question:" or "Answer the Question:"
    s = _ANSWER_QUESTION_RE.sub("", s).strip()

    return s, query_date


def _extract_entity_hints(text: str) -> list[str]:
    """Extract entity hints from question text.

    Returns lowercased entity strings, deduplicated, ordered by
    length descending (longer = more specific = higher priority).

    Extraction tiers:
    1. Multi-word proper nouns: "Imagine Dragons", "Universal Studios Hollywood"
    2. Quoted phrases: 'Page Turners', "Summer Nights"
    3. Domain nouns from curated set
    4. Single capitalized words not at sentence start
    5. Brand patterns (H&M, etc.)
    6. "the <noun phrase>" patterns
    """
    entities: set[str] = set()

    # Tier 1: Multi-word proper nouns
    for m in _PROPER_NOUN_RE.finditer(text):
        phrase = m.group(1).strip().lower()
        # Filter out pure stopword sequences
        words = phrase.split()
        if not all(w in _COMMON_CAPS for w in words):
            entities.add(phrase)

    # Tier 2: Quoted phrases
    for m in _QUOTED_RE.finditer(text):
        phrase = m.group(1).strip().lower()
        if phrase and len(phrase) >= 2:
            entities.add(phrase)

    # Tier 3: Brand patterns
    for m in _BRAND_RE.finditer(text):
        entities.add(m.group(1).lower())

    # Tier 4: Domain nouns
    text_lower = text.lower()
    words_in_text = set(re.findall(r"[a-z&]+", text_lower))
    for noun in _DOMAIN_NOUNS:
        if noun in words_in_text or noun in text_lower:
            # Add the stemmed form
            stemmed = _simple_stem(noun)
            entities.add(stemmed)

    # Tier 5: Single capitalized words (not sentence-start)
    # Skip words that appear right after sentence boundary (. ! ? or start)
    sentences = re.split(r"[.!?]\s+", text)
    for sent in sentences:
        # Skip first word of sentence
        words = sent.split()
        for w in words[1:]:
            # Check if it's a capitalized word
            m2 = _SINGLE_CAP_RE.match(w)
            if m2:
                wl = m2.group(1).lower()
                if wl not in _COMMON_CAPS and wl not in _STOPWORDS:
                    entities.add(wl)

    # Tier 6: "the <noun phrase>" patterns
    for m in _THE_NOUN_RE.finditer(text):
        phrase = m.group(1).strip().lower()
        if phrase and len(phrase) >= 3:
            entities.add(phrase)

    # Tier 7: Bare noun phrases ("loyalty program", "support group")
    for m in _BARE_NOUN_PHRASE_RE.finditer(text):
        phrase = m.group(1).strip().lower()
        if phrase and len(phrase) >= 3:
            words = phrase.split()
            # Strip leading question/stop words
            while words and (words[0] in _STOPWORDS or words[0] in _COMMON_CAPS):
                words = words[1:]
            if len(words) >= 2:
                cleaned = " ".join(words)
                entities.add(cleaned)

    # Sort by length descending (longer = more specific)
    return sorted(entities, key=len, reverse=True)


def _extract_predicate_hints(text: str) -> list[str]:
    """Extract predicate/verb hints from question text.

    Returns stemmed predicate strings.
    """
    predicates: set[str] = set()
    text_lower = text.lower()

    # Pattern 1: "did I/you <VERB>"
    for m in _DID_I_VERB_RE.finditer(text):
        verb = m.group(1).lower()
        if verb not in _STOPWORDS and verb not in ("the", "a", "an", "my", "your"):
            predicates.add(_simple_stem(verb))

    # Pattern 2: "have I/you (been) <VERB>"
    for m in _HAVE_I_VERB_RE.finditer(text):
        verb = m.group(1).lower()
        if verb not in _STOPWORDS and verb not in ("the", "a", "an", "my", "your", "been"):
            predicates.add(_simple_stem(verb))

    # Pattern 3: "that/which/where I <VERB>"
    for m in _THAT_I_VERB_RE.finditer(text):
        verb = m.group(1).lower()
        if verb not in _STOPWORDS and verb not in ("the", "a", "an", "my", "your"):
            predicates.add(_simple_stem(verb))

    # Pattern 4: "I <VERBed>" (past tense)
    for m in _I_VERB_RE.finditer(text):
        verb = m.group(1).lower()
        if verb not in _STOPWORDS:
            predicates.add(_simple_stem(verb))

    # Pattern 5: Scan for known action verbs
    words = re.findall(r"[a-z]+", text_lower)
    for w in words:
        if w in _ACTION_VERBS:
            predicates.add(_simple_stem(w))

    return sorted(predicates)


def _extract_object_hints(text: str) -> list[str]:
    """Extract object references from question text.

    Looks for noun phrases after determiners or bare noun phrases
    that match common object patterns.
    """
    objects: set[str] = set()

    for m in _THE_NOUN_RE.finditer(text):
        phrase = m.group(1).strip().lower()
        if phrase and len(phrase) >= 3:
            objects.add(phrase)

    for m in _BARE_NOUN_PHRASE_RE.finditer(text):
        phrase = m.group(1).strip().lower()
        if phrase and len(phrase) >= 3:
            words = phrase.split()
            while words and (words[0] in _STOPWORDS or words[0] in _COMMON_CAPS):
                words = words[1:]
            if len(words) >= 2:
                objects.add(" ".join(words))

    return sorted(objects, key=len, reverse=True)


def _extract_temporal_hint(text: str) -> str | None:
    """Extract temporal reference from question if present.

    Returns a raw temporal string (not parsed) that can be used as an
    additional filter/boost signal.
    """
    # Check patterns in priority order

    # "X weeks/days/months ago"
    m = _TEMPORAL_AGO_RE.search(text)
    if m:
        return m.group(1).strip()

    # "how long have I..."
    m = _TEMPORAL_HOW_LONG_RE.search(text)
    if m:
        return m.group(1).strip()

    # "days had passed"
    m = _TEMPORAL_DAYS_PASSED_RE.search(text)
    if m:
        return m.group(1).strip()

    # Named dates: "Valentine's day", "Christmas", etc.
    m = _TEMPORAL_NAMED_DATE_RE.search(text)
    if m:
        return m.group(1).strip()

    # "last week/month/year"
    m = _TEMPORAL_LAST_RE.search(text)
    if m:
        return m.group(1).strip()

    # "in January/June/etc"
    m = _TEMPORAL_IN_MONTH_RE.search(text)
    if m:
        return m.group(1).strip()

    # Standalone month name (not May — too ambiguous)
    m = _TEMPORAL_MONTH_STANDALONE_RE.search(text)
    if m:
        return m.group(1).strip()

    # "recently", "currently", etc.
    m = _TEMPORAL_RECENTLY_RE.search(text)
    if m:
        return m.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_query(question_text: str) -> ParsedQuery:
    """Parse an LME question into structured hints for fact store lookup.

    Regex-only, no LLM calls.  Steps:
    1. Strip LME boilerplate ("Based on our conversations..." + "Current Date:")
    2. Classify question type via classify_question_type()
    3. Extract entity hints (proper nouns, quoted phrases, domain nouns)
    4. Extract predicate hints (verbs from "did I <verb>" patterns + action verbs)
    5. Extract object hints (noun phrases after "the")
    6. Extract temporal hints (date references, "ago", named dates)

    Parameters
    ----------
    question_text : str
        Raw LME question text, may include boilerplate prefix and
        "Current Date:" header.

    Returns
    -------
    ParsedQuery
        Structured query with entity/predicate/temporal hints.
    """
    bare, query_date = _strip_lme_boilerplate(question_text)

    if not bare:
        return ParsedQuery(raw="", query_date_str=query_date)

    q_type = classify_question_type(bare)
    entity_hints = _extract_entity_hints(bare)
    predicate_hints = _extract_predicate_hints(bare)
    object_hints = _extract_object_hints(bare)
    temporal_hint = _extract_temporal_hint(bare)

    return ParsedQuery(
        entity_hints=entity_hints,
        predicate_hints=predicate_hints,
        object_hints=object_hints,
        question_type=q_type,
        temporal_hint=temporal_hint,
        raw=bare,
        query_date_str=query_date,
    )
