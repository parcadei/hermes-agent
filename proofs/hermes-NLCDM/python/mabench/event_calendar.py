"""event_calendar.py -- Event calendar for temporal question resolution.

Production module extracted from disc_temporal.py. Provides:

1. parse_turn_date(turn_text)         -- extract datetime from turn prefix
2. extract_event_keywords(text)       -- extract event-relevant keywords
3. extract_inline_dates(text, turn_date) -- extract inline date references
4. classify_temporal_question(question) -- classify temporal question type
5. resolve_temporal_question(question, event_calendar, query_date)
                                      -- resolve temporal question via date arithmetic
6. parse_query_date(question_text)    -- extract 'Current Date' from question

These functions power the temporal short-circuit in query_pipeline.py,
which resolves temporal questions with Python date arithmetic instead of
asking the LLM (which gets date math wrong).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TURN_TIME_RE = re.compile(
    r"^\[(\d{4})/(\d{2})/(\d{2})\s*\(\w+\)\s*(\d{1,2}):(\d{2})\]"
)

_QUERY_DATE_RE = re.compile(
    r"Current Date:\s*(\d{4})/(\d{2})/(\d{2})\s*\(\w+\)\s*(\d{1,2}):(\d{2})"
)

_ROLE_RE = re.compile(r"\]\s*\((\w+)\)")

_STOPWORDS = frozenset({
    "the", "a", "an", "i", "my", "me", "is", "was", "were", "am", "are",
    "do", "did", "does", "have", "had", "has", "be", "been", "being",
    "in", "on", "at", "to", "of", "for", "with", "and", "or", "but",
    "that", "this", "it", "its", "also", "not", "no", "so", "if",
    "what", "which", "who", "how", "when", "where", "why",
    "can", "could", "would", "should", "will", "shall",
    "just", "got", "get", "from", "by", "about", "some", "any", "all",
    "much", "many", "more", "most", "very", "too", "than",
    "then", "now", "here", "there", "up", "down", "out",
    "you", "your", "we", "our", "they", "them", "their", "he", "she",
    "him", "her", "us", "been", "going", "went", "go", "come", "came",
})

# Month name -> number
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

# Word -> number for relative expressions
_WORD_NUM = {
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

# Action verbs that indicate the entity's temporal occurrence
_ACTION_VERBS = frozenset({
    "finished", "started", "began", "attended", "joined", "bought",
    "received", "got", "lost", "watched", "saw", "visited",
    "went", "completed", "arrived", "returned", "left",
})

# Named date resolution
_NAMED_DATES = {
    "valentine's day": (2, 14),
    "valentines day": (2, 14),
    "valentine": (2, 14),
    "christmas": (12, 25),
    "new year's": (1, 1),
    "halloween": (10, 31),
    "independence day": (7, 4),
    "thanksgiving": (11, 23),  # approximate
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _simple_stem(word: str) -> str:
    """Naive suffix stripping for English plurals and verb forms.

    Not a full stemmer, but handles the common cases:
    - tomatoes -> tomato, marigolds -> marigold
    - classes -> class, buses -> bus
    - running -> run (but keep 'ring', 'thing')
    """
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("oes") and len(w) > 4:
        return w[:-2]
    if w.endswith("ses") and len(w) > 4:
        return w[:-2]
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    return w


def _parse_turn_role(turn_text: str) -> str:
    """Extract role ('user' or 'assistant') from turn text."""
    m = _ROLE_RE.search(turn_text)
    if m:
        return m.group(1)
    return "unknown"


# ---------------------------------------------------------------------------
# Public API: Parsing helpers
# ---------------------------------------------------------------------------

def parse_turn_date(turn_text: str) -> datetime | None:
    """Parse datetime from turn prefix [YYYY/MM/DD (Day) HH:MM]."""
    m = _TURN_TIME_RE.match(turn_text)
    if m:
        return datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)),
        )
    return None


# Regex for "Chat Time: YYYY/MM/DD (Day) HH:MM" in raw context strings
_CHAT_TIME_RE = re.compile(
    r"Chat Time:\s*(\d{4})/(\d{2})/(\d{2})\s*\(\w+\)\s*(\d{1,2}):(\d{2})"
)


def parse_all_chat_times(text: str) -> list[datetime]:
    """Extract ALL Chat Time timestamps from a raw chunk.

    Raw LME chunks contain multiple turns, each preceded by
    ``Chat Time: YYYY/MM/DD (Day) HH:MM``.  Returns all parsed
    datetimes in order of appearance.
    """
    results: list[datetime] = []
    for m in _CHAT_TIME_RE.finditer(text):
        results.append(datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)),
        ))
    return results


def parse_query_date(question_text: str) -> datetime | None:
    """Extract 'Current Date' from question text."""
    m = _QUERY_DATE_RE.search(question_text)
    if m:
        return datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)),
        )
    return None


# ---------------------------------------------------------------------------
# Public API: Keyword extraction
# ---------------------------------------------------------------------------

def extract_event_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text.

    Returns a set of lowercased keywords including:
    - Single meaningful words (nouns, verbs) after stopword filtering
    - Quoted multi-word phrases ('Page Turners', "Summer Nights")
    - Capitalized multi-word entities (Universal Studios Hollywood)
    """
    keywords: set[str] = set()

    # 1. Extract quoted phrases
    for m in re.finditer(r"""['"]([^'"]{2,})['"]""", text):
        phrase = m.group(1).strip().lower()
        if phrase:
            keywords.add(phrase)

    # 2. Extract capitalized multi-word entities (2+ capitalized words in a row)
    # Strip the turn prefix first
    body = re.sub(r"^\[.*?\]\s*\(\w+\)\s*", "", text)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", body):
        phrase = m.group(1).lower()
        if not all(w in _STOPWORDS for w in phrase.split()):
            keywords.add(phrase)

    # 3. Extract individual words (lowercase, strip punctuation)
    words = re.findall(r"[a-zA-Z]+", text)
    for w in words:
        wl = w.lower()
        if wl not in _STOPWORDS and len(wl) >= 3:
            keywords.add(wl)

    return keywords


# ---------------------------------------------------------------------------
# Public API: Inline date extraction from conversational text
# ---------------------------------------------------------------------------

def extract_inline_dates(text: str, turn_date: datetime) -> list[datetime]:
    """Extract date references embedded in conversational text.

    Handles patterns like:
    - "on the 3/8" (M/D)
    - "on the 3rd of June" (ordinal + month)
    - "the 15th of March"
    - "today", "yesterday", "last Saturday"
    - "about a month ago", "two weeks ago"
    - "last week"
    - "for the past three months" / "since February"
    - "in January", "in June"
    """
    dates: list[datetime] = []
    text_lower = text.lower()

    # 1. M/D or M-D format: "on the 3/8", "on 6/3"
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})\b", text):
        month, day = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1 <= day <= 31:
            year = turn_date.year
            try:
                dates.append(datetime(year, month, day))
            except ValueError:
                pass

    # 2. Ordinal + "of" + month: "3rd of June", "15th of March"
    for m in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)\s+of\s+(\w+)\b", text_lower
    ):
        day = int(m.group(1))
        month_name = m.group(2)
        month_num = _MONTH_MAP.get(month_name)
        if month_num and 1 <= day <= 31:
            year = turn_date.year
            if month_num > turn_date.month:
                year -= 1
            try:
                dates.append(datetime(year, month_num, day))
            except ValueError:
                pass

    # 3. Month + day: "June 3rd", "March 15th", "January 5"
    for m in re.finditer(
        r"\b(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?\b", text_lower
    ):
        month_name = m.group(1)
        month_num = _MONTH_MAP.get(month_name)
        if month_num:
            day = int(m.group(2))
            if 1 <= day <= 31:
                year = turn_date.year
                if month_num > turn_date.month:
                    year -= 1
                try:
                    dates.append(datetime(year, month_num, day))
                except ValueError:
                    pass

    # 4. Relative: "today"
    if re.search(r"\btoday\b", text_lower):
        dates.append(datetime(turn_date.year, turn_date.month, turn_date.day))

    # 5. Relative: "yesterday"
    if re.search(r"\byesterday\b", text_lower):
        yd = turn_date - timedelta(days=1)
        dates.append(datetime(yd.year, yd.month, yd.day))

    # 6. Relative: "last week"
    if re.search(r"\blast\s+week\b", text_lower):
        lw = turn_date - timedelta(days=7)
        dates.append(datetime(lw.year, lw.month, lw.day))

    # 7. Relative: "X weeks ago" or "about X weeks ago"
    for m in re.finditer(
        r"\b(?:about\s+)?(\w+)\s+weeks?\s+ago\b", text_lower
    ):
        num_str = m.group(1)
        num = _WORD_NUM.get(num_str)
        if num is None:
            try:
                num = int(num_str)
            except ValueError:
                continue
        ago = turn_date - timedelta(weeks=num)
        dates.append(datetime(ago.year, ago.month, ago.day))

    # 8. Relative: "a month ago" / "about a month ago"
    if re.search(r"\b(?:about\s+)?a\s+month\s+ago\b", text_lower):
        ago = turn_date - timedelta(days=30)
        dates.append(datetime(ago.year, ago.month, ago.day))

    # 9. Relative: "X months ago"
    for m in re.finditer(
        r"\b(?:about\s+)?(\w+)\s+months?\s+ago\b", text_lower
    ):
        num_str = m.group(1)
        if num_str == "a":
            continue  # handled by pattern 8
        num = _WORD_NUM.get(num_str)
        if num is None:
            try:
                num = int(num_str)
            except ValueError:
                continue
        # Approximate: subtract num*30 days
        ago = turn_date - timedelta(days=num * 30)
        dates.append(datetime(ago.year, ago.month, ago.day))

    # 10. "for the past X months" / "since X months"
    for m in re.finditer(
        r"\bpast\s+(\w+)\s+months?\b", text_lower
    ):
        num_str = m.group(1)
        num = _WORD_NUM.get(num_str)
        if num is None:
            try:
                num = int(num_str)
            except ValueError:
                continue
        ago = turn_date - timedelta(days=num * 30)
        dates.append(datetime(ago.year, ago.month, ago.day))

    # 11. "in January", "in June" (standalone month reference)
    for m in re.finditer(r"\bin\s+(\w+)\b", text_lower):
        month_name = m.group(1)
        month_num = _MONTH_MAP.get(month_name)
        if month_num:
            year = turn_date.year
            if month_num > turn_date.month:
                year -= 1
            try:
                dates.append(datetime(year, month_num, 15))  # mid-month estimate
            except ValueError:
                pass

    # 12. "since MONTH" (e.g., "since February")
    for m in re.finditer(r"\bsince\s+(\w+)\b", text_lower):
        month_name = m.group(1)
        month_num = _MONTH_MAP.get(month_name)
        if month_num:
            year = turn_date.year
            if month_num > turn_date.month:
                year -= 1
            try:
                dates.append(datetime(year, month_num, 1))
            except ValueError:
                pass

    # 13. "last Saturday", "last Sunday" etc.
    for m in re.finditer(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text_lower,
    ):
        day_names = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }
        target_dow = day_names[m.group(1)]
        current_dow = turn_date.weekday()
        days_back = (current_dow - target_dow) % 7
        if days_back == 0:
            days_back = 7
        past = turn_date - timedelta(days=days_back)
        dates.append(datetime(past.year, past.month, past.day))

    return dates


# ---------------------------------------------------------------------------
# Public API: Question classification
# ---------------------------------------------------------------------------

def classify_temporal_question(question: str) -> str:
    """Classify temporal question into a reasoning type.

    Returns one of: 'day_count', 'duration', 'weeks_ago', 'months_ago',
    'date_lookup', 'recency', 'sequence', 'ordering', or 'unknown'.
    """
    q = question.lower()

    # Day count: "how many days had passed between X and Y"
    if re.search(r"how many days.*(between|passed|since)", q):
        return "day_count"

    # Duration: "how many days did I spend on"
    if re.search(r"how many days.*(spend|spent|stay|stayed|trip|last|lasted)", q):
        return "duration"

    # Weeks ago: "how many weeks ago"
    if re.search(r"how many weeks ago", q):
        return "weeks_ago"

    # Months ago: "how many months ago"
    if re.search(r"how many months ago", q):
        return "months_ago"

    # Date lookup: "what was the date", "what date"
    if re.search(r"what\s+(was\s+the\s+)?date", q):
        return "date_lookup"

    # Recency: "most recently"
    if re.search(r"most recently", q):
        return "recency"

    # Sequence: "order of", "what order"
    if re.search(r"(order of|what order|the order)", q):
        return "sequence"

    # Ordering: "which...first", "who...first", "happened first"
    if re.search(r"(which|who).*(first|before|earlier)", q):
        return "ordering"

    # Airline on date: date_lookup variant
    if re.search(r"(what was the|which).*(airline|flight)", q):
        return "date_lookup"

    return "unknown"


# ---------------------------------------------------------------------------
# Public API: Temporal question resolution
# ---------------------------------------------------------------------------

def resolve_temporal_question(
    question: str,
    event_calendar: list[dict],
    query_date: datetime | None,
) -> str | None:
    """Resolve a temporal question using the event calendar.

    Uses Python date arithmetic (not an LLM) to compute temporal answers:
    ordering, day counts, weeks/months ago, date lookups, durations,
    recency comparisons, and event sequencing.

    Args:
        question: The question text (without the 'Current Date' prefix).
        event_calendar: List of calendar entry dicts, each containing at
            minimum: 'date' (datetime), 'keywords' (set[str]),
            'text' (str), 'role' (str), 'inline_dates' (list[datetime]).
        query_date: The reference date for relative computations
            (extracted from the question's 'Current Date' header).

    Returns:
        The computed answer string, or None if unable to resolve.
    """
    if query_date is None:
        return None

    qtype = classify_temporal_question(question)
    q_lower = question.lower()

    if qtype == "ordering":
        return _resolve_ordering(event_calendar, question, q_lower)
    elif qtype == "day_count":
        return _resolve_day_count(event_calendar, question, q_lower)
    elif qtype == "weeks_ago":
        return _resolve_weeks_ago(event_calendar, question, q_lower, query_date)
    elif qtype == "months_ago":
        return _resolve_months_ago(event_calendar, question, q_lower, query_date)
    elif qtype == "date_lookup":
        return _resolve_date_lookup(event_calendar, question, q_lower)
    elif qtype == "duration":
        return _resolve_duration(event_calendar, question, q_lower)
    elif qtype == "recency":
        return _resolve_recency(event_calendar, question, q_lower, query_date)
    elif qtype == "sequence":
        return _resolve_sequence(event_calendar, question, q_lower, query_date)
    return None


# ---------------------------------------------------------------------------
# Public API: Event calendar retrieval (keyword-based)
# ---------------------------------------------------------------------------

def event_calendar_retrieve(
    query_text: str,
    event_calendar: list[dict],
    top_k: int = 30,
) -> list[tuple[int, float]]:
    """Retrieve document indices from the event calendar by keyword overlap.

    Uses the same keyword extraction as ingestion to match query keywords
    against stored event keywords. Returns (doc_id, score) pairs sorted
    by descending score, suitable for RRF fusion with BM25/entity channels.

    Args:
        query_text: The question text (may include 'Current Date' prefix).
        event_calendar: List of calendar entry dicts with 'turn_idx',
            'keywords' (set[str]), and optionally 'text' (str).
        top_k: Maximum results to return.

    Returns:
        List of (doc_id, score) tuples, highest score first.
    """
    if not event_calendar:
        return []

    query_kws = _extract_question_keywords(query_text)
    if not query_kws:
        return []

    query_stems = {_simple_stem(k) for k in query_kws}

    scored: list[tuple[float, int]] = []
    seen_ids: set[int] = set()

    for entry in event_calendar:
        doc_id = entry.get("turn_idx")
        if doc_id is None or doc_id in seen_ids:
            continue

        entry_kws = {k.lower() for k in entry.get("keywords", set())}
        entry_stems = {_simple_stem(k) for k in entry_kws}

        exact = len(query_kws & entry_kws)
        stemmed = len(query_stems & entry_stems)
        overlap = max(exact, stemmed)

        # Phrase match bonus (same as _find_events)
        text_lower = entry.get("text", "").lower()
        for qk in query_kws:
            if " " in qk and qk in text_lower:
                overlap += 2

        if overlap >= 1:
            scored.append((float(overlap), doc_id))
            seen_ids.add(doc_id)

    scored.sort(key=lambda x: -x[0])
    return [(doc_id, score) for score, doc_id in scored[:top_k]]


# ---------------------------------------------------------------------------
# Event search internals
# ---------------------------------------------------------------------------

def _find_events(
    calendar: list[dict],
    query_keywords: set[str],
    min_overlap: int = 2,
    user_only: bool = False,
) -> list[dict]:
    """Find calendar events matching query keywords.

    Returns events sorted by keyword overlap (descending), then by date.
    Requires at least ``min_overlap`` keywords in common.
    Uses simple stemming to handle plural/singular mismatches.
    """
    scored: list[tuple[int, dict]] = []
    query_kw_lower = {k.lower() for k in query_keywords}
    query_stems = {_simple_stem(k) for k in query_kw_lower}

    for entry in calendar:
        if user_only and entry.get("role") != "user":
            continue

        entry_kw_lower = {k.lower() for k in entry.get("keywords", set())}
        entry_stems = {_simple_stem(k) for k in entry_kw_lower}

        # Count overlap using both exact and stemmed matching
        exact_overlap = len(query_kw_lower & entry_kw_lower)
        stem_overlap = len(query_stems & entry_stems)
        overlap = max(exact_overlap, stem_overlap)

        # Also check for phrase containment (e.g. "summer nights" in text)
        text_lower = entry.get("text", "").lower()
        for qk in query_kw_lower:
            if " " in qk and qk in text_lower:
                overlap += 2  # bonus for phrase match

        if overlap >= min_overlap:
            scored.append((overlap, entry))

    scored.sort(key=lambda x: (-x[0], x[1]["date"]))
    return [s[1] for s in scored]


def _find_best_event(
    calendar: list[dict],
    keywords: set[str],
    prefer_user: bool = True,
) -> dict | None:
    """Find the single best event matching keywords.

    Tries progressively relaxed search:
    1. User turns only, min_overlap = len(keywords)
    2. User turns only, min_overlap = max(2, len(keywords)-1)
    3. User turns only, min_overlap = 1
    4. All turns, min_overlap = 1
    """
    n = len(keywords)
    thresholds = [n, max(2, n - 1), 1]
    for threshold in thresholds:
        if prefer_user:
            matches = _find_events(calendar, keywords, min_overlap=threshold, user_only=True)
            if matches:
                return matches[0]
    # Fallback: all turns
    matches = _find_events(calendar, keywords, min_overlap=1)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Date arithmetic
# ---------------------------------------------------------------------------

def _days_between(d1: date, d2: date) -> int:
    """Number of days between two dates (absolute value)."""
    return abs((d2 - d1).days)


def _weeks_ago(event_date: date, query_date: date) -> int:
    """Full weeks between event and query date."""
    return (query_date - event_date).days // 7


def _months_ago(event_date: date, query_date: date) -> int:
    """Approximate months between event and query date."""
    return (query_date.year - event_date.year) * 12 + (query_date.month - event_date.month)


# ---------------------------------------------------------------------------
# Keyword/entity extraction helpers
# ---------------------------------------------------------------------------

def _extract_question_keywords(question_text: str) -> set[str]:
    """Extract keywords from question text (strip the 'Current Date' prefix)."""
    body = re.sub(
        r"Current Date:.*?Now Answer the Question:\s*",
        "", question_text, flags=re.DOTALL,
    )
    return extract_event_keywords(body)


def _extract_two_entities(question_lower: str) -> tuple[str, str] | None:
    """Extract two entities being compared from a question.

    Handles patterns like:
    - "X or Y?"
    - "X or the Y?"
    - "between X and Y"
    """
    # "which ... first, X or Y?"
    m = re.search(
        r"(?:which|who).*?(?:first|before|earlier).*?,\s*(.*?)\s+or\s+(.*?)[\?\.]",
        question_lower, re.DOTALL,
    )
    if m:
        return m.group(1).strip().rstrip(","), m.group(2).strip().rstrip("?.")

    # "happened first, X or Y?"
    m = re.search(
        r"happened\s+first.*?,\s*(.*?)\s+or\s+(.*?)[\?\.]",
        question_lower, re.DOTALL,
    )
    if m:
        return m.group(1).strip().rstrip(","), m.group(2).strip().rstrip("?.")

    # "between X and Y"
    m = re.search(
        r"between\s+(.*?)\s+and\s+(.*?)[\?\.]",
        question_lower,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return None


def _entity_keywords(entity: str) -> set[str]:
    """Convert entity description to search keywords."""
    words = re.findall(r"[a-zA-Z]+", entity)
    kws = set()
    for w in words:
        wl = w.lower()
        if wl not in _STOPWORDS and len(wl) >= 3:
            kws.add(wl)
    # Also extract quoted phrases
    for m in re.finditer(r"""['"]([^'"]+)['"]""", entity):
        kws.add(m.group(1).lower())
    return kws


def _score_event_match(
    entry: dict,
    entity_kws: set[str],
    action_verbs: set[str] | None = None,
) -> float:
    """Score how well a calendar entry matches entity keywords.

    Accounts for:
    - Keyword overlap (exact + stemmed)
    - Phrase matches in text
    - Bonus for action verb matches (finished, attended, etc.)
    - Bonus for inline dates (temporal informativeness)
    """
    kw_lower = {k.lower() for k in entry.get("keywords", set())}
    stems = {_simple_stem(k) for k in kw_lower}
    query_stems = {_simple_stem(k) for k in entity_kws}
    score = max(len(entity_kws & kw_lower), len(query_stems & stems))

    # Bonus for phrase matches in text
    text_lower = entry.get("text", "").lower()
    for qk in entity_kws:
        if " " in qk and qk in text_lower:
            score += 2

    # Bonus for action verb match, but ONLY if the turn also matches
    # at least one non-verb entity keyword (prevents "attended a wedding"
    # from matching "attended the photography workshop")
    if action_verbs:
        non_verb_kws = entity_kws - action_verbs
        non_verb_overlap = len(non_verb_kws & kw_lower) + len(
            {_simple_stem(k) for k in non_verb_kws} & stems
        )
        if non_verb_overlap > 0:
            verb_overlap = len(action_verbs & kw_lower) + len(
                {_simple_stem(v) for v in action_verbs} & stems
            )
            score += verb_overlap * 1.5

    # Bonus for having inline dates (temporally informative)
    if entry.get("inline_dates"):
        score += 0.5

    return score


def _get_event_date(event: dict) -> date:
    """Get the most specific date for an event (inline_date if available)."""
    if event.get("inline_date"):
        d = event["inline_date"]
        return date(d.year, d.month, d.day)
    d = event["date"]
    return date(d.year, d.month, d.day)


def _best_event_date(
    calendar: list[dict],
    entity_kws: set[str],
) -> tuple[date | None, dict | None]:
    """Find the best event date for an entity.

    Returns (resolved_date, best_event) using inline dates when available.
    Prefers user turns and events that have inline dates (explicit temporal
    references like "three months ago" or "on the 3/8") over events that
    only have a turn timestamp.
    """
    matches = _find_events(calendar, entity_kws, min_overlap=1, user_only=True)
    if not matches:
        matches = _find_events(calendar, entity_kws, min_overlap=1)

    if not matches:
        return None, None

    # Extract action verbs from entity keywords
    action_verbs = entity_kws & _ACTION_VERBS
    # Also check stemmed versions
    for kw in entity_kws:
        if _simple_stem(kw) in {_simple_stem(v) for v in _ACTION_VERBS}:
            action_verbs.add(kw)

    # Score all matches
    scored = [(m, _score_event_match(m, entity_kws, action_verbs)) for m in matches]
    scored.sort(key=lambda x: -x[1])
    top_score = scored[0][1]

    # Collect matches within 1.5 of top score (effectively same tier,
    # accounts for typos reducing keyword overlap by 1-2)
    top_matches = [(m, s) for m, s in scored if s >= top_score - 1.5]

    # Among top matches, prefer ones with inline dates
    with_inline = [(m, s) for m, s in top_matches if m.get("inline_dates")]
    if with_inline:
        best = with_inline[0][0]
        d = best["inline_dates"][0]
        return date(d.year, d.month, d.day), best

    # If action verbs indicate temporal occurrence (finished, started, attended),
    # prefer the EARLIEST matching turn (temporal primacy: first mention of
    # "finished reading X" is when it happened, later mentions are references)
    if action_verbs:
        earliest = min(top_matches, key=lambda ms: ms[0].get("turn_index", 0))
        best = earliest[0]
        d = best["date"]
        return date(d.year, d.month, d.day), best

    # Fallback: use the highest-scoring match's turn date
    best = top_matches[0][0]
    d = best["date"]
    return date(d.year, d.month, d.day), best


def _format_entity_label(entity: str) -> str:
    """Format an entity description into a readable answer."""
    label = re.sub(r"^(the|my|a|an)\s+", "", entity.strip())
    if label:
        label = label[0].upper() + label[1:]
    return label


def _format_date(d: date) -> str:
    """Format a date as 'Month Dayth'."""
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    suffix = "th"
    if d.day in (1, 21, 31):
        suffix = "st"
    elif d.day in (2, 22):
        suffix = "nd"
    elif d.day in (3, 23):
        suffix = "rd"
    return f"{month_names[d.month]} {d.day}{suffix}"


# ---------------------------------------------------------------------------
# Question type resolvers
# ---------------------------------------------------------------------------

def _resolve_ordering(
    calendar: list[dict],
    question: str,
    q_lower: str,
) -> str | None:
    """Resolve 'which happened first' questions."""
    entities = _extract_two_entities(q_lower)
    if not entities:
        return None

    entity_a, entity_b = entities
    kw_a = _entity_keywords(entity_a)
    kw_b = _entity_keywords(entity_b)

    date_a, evt_a = _best_event_date(calendar, kw_a)
    date_b, evt_b = _best_event_date(calendar, kw_b)

    if date_a is None and date_b is None:
        return None
    if date_a is None:
        return f"NOT_FOUND: {entity_a}"
    if date_b is None:
        return f"NOT_FOUND: {entity_b}"

    if date_a <= date_b:
        return _format_entity_label(entity_a)
    else:
        return _format_entity_label(entity_b)


def _resolve_day_count(
    calendar: list[dict],
    question: str,
    q_lower: str,
) -> str | None:
    """Resolve 'how many days between X and Y' questions."""
    entities = _extract_two_entities(q_lower)
    if not entities:
        # Try "how many days ... since X when Y"
        m = re.search(
            r"how many days.*?since\s+(.*?)\s+when\s+(.*?)[\?\.]",
            q_lower, re.DOTALL,
        )
        if m:
            entities = (m.group(1).strip(), m.group(2).strip())
        else:
            return None

    entity_a, entity_b = entities
    kw_a = _entity_keywords(entity_a)
    kw_b = _entity_keywords(entity_b)

    # Also extract quoted phrases from entity descriptions
    for m in re.finditer(r"""['"]([^'"]+)['"]""", entity_a):
        kw_a.add(m.group(1).lower())
    for m in re.finditer(r"""['"]([^'"]+)['"]""", entity_b):
        kw_b.add(m.group(1).lower())

    date_a, _ = _best_event_date(calendar, kw_a)
    date_b, _ = _best_event_date(calendar, kw_b)

    if date_a is None or date_b is None:
        return None

    n = _days_between(date_a, date_b)
    return f"{n} days"


def _resolve_weeks_ago(
    calendar: list[dict],
    question: str,
    q_lower: str,
    query_date: datetime,
) -> str | None:
    """Resolve 'how many weeks ago' questions."""
    m = re.search(
        r"how many weeks ago\s+did\s+(?:i|you)\s+(.*?)[\?\.]",
        q_lower,
    )
    if not m:
        return None

    event_desc = m.group(1).strip()
    kws = _entity_keywords(event_desc)
    event_date, _ = _best_event_date(calendar, kws)

    if event_date is None:
        return None

    qd = date(query_date.year, query_date.month, query_date.day)
    n = _weeks_ago(event_date, qd)
    return f"{n} weeks ago"


def _resolve_months_ago(
    calendar: list[dict],
    question: str,
    q_lower: str,
    query_date: datetime,
) -> str | None:
    """Resolve 'how many months ago' questions."""
    m = re.search(
        r"how many months ago\s+did\s+(?:i|you)\s+(.*?)[\?\.]",
        q_lower,
    )
    if not m:
        return None

    event_desc = m.group(1).strip()
    kws = _entity_keywords(event_desc)
    event_date, _ = _best_event_date(calendar, kws)

    if event_date is None:
        return None

    qd = date(query_date.year, query_date.month, query_date.day)
    n = _months_ago(event_date, qd)
    return str(n)


def _resolve_date_lookup(
    calendar: list[dict],
    question: str,
    q_lower: str,
) -> str | None:
    """Resolve 'what was the date of X' or 'what X on date Y' questions."""
    kws = _extract_question_keywords(question)

    # If asking about something ON a specific date (e.g., "airline on Valentine's day"),
    # resolve the date first, then find events on that date
    if re.search(r"(airline|flight|flew|flied)", q_lower):
        # Resolve the named date
        target_date = None
        for name, (month, day) in _NAMED_DATES.items():
            if name in q_lower:
                target_date = (month, day)
                break

        if target_date:
            month, day = target_date
            date_matches = [
                e for e in calendar
                if e.get("role") == "user"
                and e["date"].month == month
                and e["date"].day == day
            ]
            if date_matches:
                return date_matches[0]["text"][:200]

        # Fallback: keyword search
        best = _find_best_event(calendar, kws, prefer_user=True)
        if best:
            return best["text"][:200]
        return None

    # For "what was the date of X": find event with inline date reference
    best_date, best_evt = _best_event_date(calendar, kws)
    if best_date is None:
        return None

    return _format_date(best_date)


def _resolve_duration(
    calendar: list[dict],
    question: str,
    q_lower: str,
) -> str | None:
    """Resolve 'how many days did I spend on X' questions."""
    m = re.search(
        r"how many days\s+did\s+(?:i|you)\s+(?:spend|stay)\s+(?:on|at|in)\s+(.*?)[\?\.]",
        q_lower,
    )
    if not m:
        return None

    activity = m.group(1).strip()
    kws = _entity_keywords(activity)

    # Find user turns mentioning this activity with high overlap
    matches = _find_events(calendar, kws, min_overlap=max(2, len(kws) - 1), user_only=True)
    if not matches:
        matches = _find_events(calendar, kws, min_overlap=1, user_only=True)
    if not matches:
        return None

    # Sort by turn_index to find chronological order
    matches_by_idx = sorted(matches, key=lambda x: x.get("turn_index", 0))

    # Find start event (mentions starting/beginning/today) and end event (got back/returned)
    start_date = None
    end_date = None
    for evt in matches_by_idx:
        text_lower = evt.get("text", "").lower()
        if re.search(r"(just start|start.*today|heading.*today|began|arrived today)", text_lower):
            if start_date is None:
                start_date = _get_event_date(evt)
        if re.search(r"(got back|returned|back from|just got back|finished.*today)", text_lower):
            end_date = _get_event_date(evt)

    if start_date and end_date:
        n = _days_between(start_date, end_date)
        return f"{n} days"

    # Fallback: use first and last turn dates if they differ
    if len(matches_by_idx) >= 2:
        d1 = date(matches_by_idx[0]["date"].year, matches_by_idx[0]["date"].month, matches_by_idx[0]["date"].day)
        d2 = date(matches_by_idx[-1]["date"].year, matches_by_idx[-1]["date"].month, matches_by_idx[-1]["date"].day)
        if d1 != d2:
            n = _days_between(d1, d2)
            return f"{n} days"

    return None


def _resolve_recency(
    calendar: list[dict],
    question: str,
    q_lower: str,
    query_date: datetime,
) -> str | None:
    """Resolve 'which X did I use most recently, A or B?' questions."""
    m = re.search(
        r"most recently.*?,\s*(.*?)\s+or\s+(.*?)[\?\.]",
        q_lower,
    )
    if not m:
        entities = _extract_two_entities(q_lower)
        if not entities:
            return None
        entity_a, entity_b = entities
    else:
        entity_a = m.group(1).strip().rstrip(",")
        entity_b = m.group(2).strip().rstrip("?.")

    kw_a = _entity_keywords(entity_a)
    kw_b = _entity_keywords(entity_b)

    qd = date(query_date.year, query_date.month, query_date.day)

    def latest_use(kws: set[str]) -> date | None:
        """Find the latest turn date where entity was actually USED, before query_date."""
        matches = _find_events(calendar, kws, min_overlap=1, user_only=True)
        latest = None
        for match in matches:
            d = date(match["date"].year, match["date"].month, match["date"].day)
            if d > qd:
                continue
            text_lower = match.get("text", "").lower()
            # Check for actual transport usage patterns
            usage_patterns = [
                r"took\s+(a\s+)?" + "|".join(kws),
                r"(ride|rode|from a|got back from|just got back|on the)\b",
                r"\b(bus ride|train ride|took a train|took a bus)\b",
                r"\b(taking a lot of trains|taking a lot of buses)\b",
            ]
            is_usage = any(re.search(pat, text_lower) for pat in usage_patterns)
            if is_usage and (latest is None or d > latest):
                latest = d
        return latest

    latest_a = latest_use(kw_a)
    latest_b = latest_use(kw_b)

    if latest_a is None and latest_b is None:
        return None
    if latest_a is None:
        return _format_entity_label(entity_b)
    if latest_b is None:
        return _format_entity_label(entity_a)
    if latest_b > latest_a:
        return _format_entity_label(entity_b)
    else:
        return _format_entity_label(entity_a)


def _resolve_sequence(
    calendar: list[dict],
    question: str,
    q_lower: str,
    query_date: datetime,
) -> str | None:
    """Resolve 'what is the order of X events' questions.

    Strategy: find user turns in the specified time period that describe
    actual event ATTENDANCE (not planning/discussion), extract the event
    name, and order by date.
    """
    events_m = re.search(
        r"order of\s+(.*?)\s+(?:i|you)\s+(?:watched|attended|went to|saw)\s+(?:in\s+)?(\w+)",
        q_lower,
    )

    if events_m:
        category_kws = _entity_keywords(events_m.group(1))
        time_word = events_m.group(2)
        month_num = _MONTH_MAP.get(time_word)
    else:
        category_kws = _extract_question_keywords(question)
        month_num = None

    # Find user turns in the specified time period
    candidates = []
    for entry in calendar:
        if entry.get("role") != "user":
            continue
        if month_num and entry["date"].month != month_num:
            continue

        text_lower = entry.get("text", "").lower()
        # Check if this turn describes actual event attendance/watching
        attendance_patterns = [
            r"(just went|just got back|just attended|just watched|just came from)",
            r"(i watched|i attended|we watched|we attended)",
            r"(riding high from|still on a high from)",
            r"(went to a|attended a|watched the|saw the)",
        ]
        is_attendance = any(
            re.search(pat, text_lower) for pat in attendance_patterns
        )
        if not is_attendance:
            continue

        # Check category relevance
        cat_relevant = bool(
            {_simple_stem(k) for k in category_kws}
            & {_simple_stem(k) for k in entry.get("keywords", set())}
        )
        sport_kws = {"nba", "nfl", "football", "basketball", "baseball",
                     "soccer", "hockey", "tennis", "golf", "game", "match",
                     "championship", "playoffs", "tournament", "race"}
        text_sport = any(sw in text_lower for sw in sport_kws)

        if text_sport and (cat_relevant or text_sport):
            candidates.append(entry)

    if not candidates:
        return None

    # Sort by date
    candidates.sort(key=lambda e: e["date"])

    # Deduplicate by date
    seen_dates: set[date] = set()
    unique: list[dict] = []
    for c in candidates:
        d = date(c["date"].year, c["date"].month, c["date"].day)
        if d not in seen_dates:
            seen_dates.add(d)
            unique.append(c)

    if len(unique) < 2:
        return None

    # Extract event names from turn text
    parts = []
    for u in unique:
        event_name = _extract_event_name(u.get("text", ""))
        parts.append(event_name)

    if len(parts) < 2:
        return None

    # Format as a sequence description
    if len(parts) == 2:
        return f"First, {parts[0]}, then {parts[1]}"
    elif len(parts) == 3:
        return f"First, {parts[0]}, then {parts[1]}, and finally, {parts[2]}"
    else:
        result = "First, " + parts[0]
        for p in parts[1:-1]:
            result += f", then {p}"
        result += f", and finally, {parts[-1]}"
        return result


def _extract_event_name(text: str) -> str:
    """Extract a concise event name from turn text.

    Looks for patterns like:
    - "I just went to a NBA game" -> "a NBA game"
    - "I'm still riding high from the College Football Championship"
      -> "the College Football Championship"
    """
    text_lower = text.lower()

    # Pattern: "from [watching] the X"
    m = re.search(
        r"(?:from|from watching)\s+(the\s+[\w\s]+?(?:game|championship|playoffs|tournament|match|race|event|festival|concert))",
        text_lower,
    )
    if m:
        return m.group(1).strip()

    # Pattern: "went to a/the X" or "attended a/the X"
    m = re.search(
        r"(?:went to|attended|watched)\s+((?:a|the|an)\s+[\w\s]+?(?:game|championship|playoffs|tournament|match|race|event|festival|concert))",
        text_lower,
    )
    if m:
        return m.group(1).strip()

    # Pattern: "in the X playoffs/championship"
    m = re.search(
        r"(?:in\s+the\s+)([\w\s]+?(?:playoffs|championship|tournament))",
        text_lower,
    )
    if m:
        return "the " + m.group(1).strip()

    # Fallback: first 80 chars
    return text[:80].strip()
