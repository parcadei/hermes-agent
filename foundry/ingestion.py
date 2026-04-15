"""Deterministic ingestion helpers for the Engram memory system.

This module implements the public Phase 4 ingestion surface required by the
tests and later phases. The LLM-backed extraction stages described in the
spec are represented here by deterministic, stdlib-only heuristics so the
core ingestion contract remains available without runtime model dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import re
from typing import Callable, Iterable, Optional
import unicodedata

from foundry.ontology import EntityType, GroundingLayer, PropertyType, infer_property_type
from foundry.storage import CooccurrenceIndex, EntityNode, Fact, FoundryStore, RawChunk, TemporalEvent


_CHAT_TIME_RE = re.compile(r"^\s*Chat Time:\s*(.+?)\s*$", re.IGNORECASE)
_TURN_MARKER_RE = re.compile(r"^\s*(User|Assistant)\s*:\s*(.*)$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\S+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
_NON_WORD_RE = re.compile(r"[^\w\s-]+", re.UNICODE)
_DETERMINER_RE = re.compile(r"^\s*(?:a|an|the)\s+", re.IGNORECASE)
_POSSESSIVE_RE = re.compile(r"\b([\w-]+)['’]s\b", re.UNICODE)
_EXPLICIT_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTH_DAY_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*(?:k|m|b|%)?", re.IGNORECASE)
_NUMBER_WORD_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|and)+\b(?:[\s-]+\b(?:zero|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|billion|and)+\b)*",
    re.IGNORECASE,
)

_MONTH_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")
_NATURAL_DATE_FORMATS = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d",
    "%b %d",
)

_IRREGULAR_VERBS = {
    "am": "be",
    "are": "be",
    "been": "be",
    "bought": "buy",
    "buys": "buy",
    "did": "do",
    "does": "do",
    "drank": "drink",
    "flies": "fly",
    "got": "get",
    "gotten": "get",
    "had": "have",
    "has": "have",
    "is": "be",
    "kept": "keep",
    "lives": "live",
    "moved": "move",
    "owned": "own",
    "planning": "plan",
    "plans": "plan",
    "thought": "think",
    "wanted": "want",
    "wants": "want",
    "went": "go",
    "worked": "work",
}

ACQUISITION_LEMMAS = frozenset(
    {
        "acquire",
        "adopt",
        "book",
        "bring_home",
        "buy",
        "collect",
        "find",
        "get",
        "obtain",
        "pick_up",
        "purchase",
        "receive",
        "rescue",
        "win",
    }
)
MAINTENANCE_LEMMAS = frozenset(
    {
        "be",
        "have",
        "hold",
        "keep",
        "live",
        "maintain",
        "own",
        "raise",
        "stay",
        "use",
        "wear",
        "work",
    }
)
INTENT_LEMMAS = frozenset(
    {
        "consider",
        "dream",
        "hope",
        "intend",
        "plan",
        "ponder",
        "think",
        "want",
        "wish",
    }
)
HYPOTHETICAL_LEMMAS = frozenset({"could", "if", "imagine", "may", "might", "suppose", "would"})
OPINION_LEMMAS = frozenset({"adore", "dislike", "enjoy", "hate", "like", "love", "prefer"})

_PET_TYPE_HINTS = {
    "bird",
    "cat",
    "dog",
    "fish",
    "hamster",
    "lizard",
    "parrot",
    "rabbit",
    "snake",
    "turtle",
}
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    turn_idx: int
    role: str
    text: str
    session_time: Optional[datetime] = None


@dataclass
class ExtractedTriple:
    """A single extracted S-P-O triple."""

    subject: str
    predicate: str
    object: str
    when: str = ""
    role: str = ""
    sub_role: str = ""
    source_text: str = ""
    turn_idx: int = -1
    chunk_idx: int = -1


@dataclass
class FacetEntity:
    """An extracted entity paired with its facet values."""

    name: str
    facets: dict[str, str]
    turn_idx: int
    source_text: str


@dataclass(frozen=True)
class VerbClassification:
    """Result of verb-role classification."""

    role: str
    sub_role: str


@dataclass
class IngestionReport:
    """Summary metrics for one ingestion run."""

    chunks_processed: int = 0
    entities_created: int = 0
    entities_resolved: int = 0
    facts_added: int = 0
    facts_superseded: int = 0
    facts_rejected: int = 0
    temporal_events: int = 0
    cooccurrences: int = 0
    grounding_rate: float = 0.0
    grounding_failures: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


def _parse_datetime(value: str | datetime | None) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for fmt in _MONTH_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_conversation(raw_text: str) -> list[ConversationTurn]:
    """Parse raw conversation text into role-labelled turns."""

    if not raw_text:
        return []

    turns: list[ConversationTurn] = []
    active_role: Optional[str] = None
    active_lines: list[str] = []
    session_time: Optional[datetime] = None

    def flush() -> None:
        nonlocal active_role, active_lines
        if active_role is None:
            active_lines = []
            return
        turns.append(
            ConversationTurn(
                turn_idx=len(turns),
                role=active_role,
                text="\n".join(active_lines),
                session_time=session_time,
            )
        )
        active_role = None
        active_lines = []

    for line in str(raw_text).splitlines():
        chat_time = _CHAT_TIME_RE.match(line)
        if chat_time:
            flush()
            session_time = _parse_datetime(chat_time.group(1))
            continue

        marker = _TURN_MARKER_RE.match(line)
        if marker:
            flush()
            active_role = marker.group(1).lower()
            active_lines = [marker.group(2)]
            continue

        if active_role is None:
            if line.strip():
                active_role = "user"
                active_lines = [line]
            continue
        active_lines.append(line)

    flush()
    return turns


def _split_text_by_token_budget(text: str, chunk_size: int) -> list[str]:
    tokens = list(_TOKEN_RE.finditer(text))
    if not tokens:
        return [text] if text else []
    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    for start_idx in range(0, len(tokens), chunk_size):
        end_idx = min(start_idx + chunk_size, len(tokens)) - 1
        start_pos = tokens[start_idx].start()
        end_pos = tokens[end_idx].end()
        chunks.append(text[start_pos:end_pos])
    return chunks


def chunk_turns(turns: list[ConversationTurn], chunk_size: int = 512) -> list[RawChunk]:
    """Split turns into `RawChunk` objects with continuation support."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks: list[RawChunk] = []
    next_chunk_idx = 0
    for turn in turns:
        pieces = _split_text_by_token_budget(turn.text, chunk_size)
        if not pieces:
            pieces = [turn.text]
        for piece_idx, piece in enumerate(pieces):
            chunks.append(
                RawChunk(
                    chunk_idx=next_chunk_idx,
                    text=piece,
                    turn_idx=turn.turn_idx,
                    role=turn.role if piece_idx == 0 else "continuation",
                    session_time=turn.session_time,
                )
            )
            next_chunk_idx += 1
    return chunks


def infer_type(entity: FacetEntity) -> Optional[str]:
    """Infer a normalized entity type from the `what` facet."""

    what_value = entity.facets.get("what", "").strip()
    if not what_value:
        return None
    normalized = EntityType.normalize(what_value)
    return normalized or None


class EntityResolver:
    """Deterministic entity resolver with ingestion-safe creation."""

    def __init__(
        self,
        store: Optional[FoundryStore] = None,
        embed_fn: Optional[Callable[[str], Iterable[float]]] = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._similarity_threshold = similarity_threshold
        self._alias_to_canonicals: dict[str, set[str]] = {}
        self._canonical_types: dict[str, set[str]] = {}
        self._embeddings: dict[str, tuple[float, ...]] = {}
        if store is not None:
            self._seed_from_store(store)

    def _seed_from_store(self, store: FoundryStore) -> None:
        for entity in store.entity_graph.all_entities():
            aliases = set(entity.aliases)
            aliases.add(entity.canonical)
            type_hint = next(iter(entity.types), None)
            self.register(entity.canonical, aliases=aliases, type_hint=type_hint)

    def normalize(self, mention: str) -> str:
        """Normalize a mention into a stable canonical/alias key."""

        text = "" if mention is None else str(mention)
        text = unicodedata.normalize("NFKC", text)
        mapped: list[str] = []
        for char in text:
            category = unicodedata.category(char)
            if category.startswith("C"):
                mapped.append(" ")
            else:
                mapped.append(char)
        text = "".join(mapped).casefold()
        text = _NON_WORD_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub("_", text.strip())
        text = re.sub(r"_+", "_", text)
        if text:
            return text
        digest = hashlib.sha1(str(mention).encode("utf-8", "ignore")).hexdigest()[:12]
        return f"entity_{digest}"

    def register(
        self,
        canonical: str,
        aliases: Optional[set[str]] = None,
        type_hint: Optional[str] = None,
    ) -> str:
        canonical_key = self.normalize(canonical)
        all_aliases = {canonical}
        if aliases:
            all_aliases.update(aliases)
        for alias in all_aliases:
            alias_key = self.normalize(alias)
            self._alias_to_canonicals.setdefault(alias_key, set()).add(canonical_key)
        if type_hint:
            self._canonical_types.setdefault(canonical_key, set()).add(EntityType.normalize(type_hint))
        if self._store is not None:
            entity = self._store.entity_graph.ensure_entity(canonical_key)
            entity.aliases.update(self.normalize(alias) for alias in all_aliases)
            if type_hint:
                normalized_type = EntityType.normalize(type_hint)
                entity.register_type(normalized_type)
                self._store.types.index(canonical_key, normalized_type)
        if self._embed_fn is not None:
            try:
                self._embeddings[canonical_key] = tuple(float(v) for v in self._embed_fn(canonical_key))
            except Exception:
                self._embeddings.pop(canonical_key, None)
        return canonical_key

    def _candidate_forms(self, mention: str, include_articles: bool = True) -> list[str]:
        if mention is None:
            return []
        raw = unicodedata.normalize("NFKC", str(mention))
        candidates = [raw]
        if include_articles:
            stripped = _DETERMINER_RE.sub("", raw)
            if stripped != raw:
                candidates.append(stripped)
        possessive = _POSSESSIVE_RE.sub(r"\1", raw)
        if possessive != raw:
            candidates.append(possessive)
            if include_articles:
                stripped_possessive = _DETERMINER_RE.sub("", possessive)
                if stripped_possessive != possessive:
                    candidates.append(stripped_possessive)
        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = self.normalize(candidate)
            if key not in seen:
                normalized.append(key)
                seen.add(key)
        return normalized

    def _pick_typed_candidate(self, candidates: set[str], hint_type: Optional[str]) -> Optional[str]:
        if not candidates:
            return None
        if not hint_type:
            return sorted(candidates)[0]
        normalized_type = EntityType.normalize(hint_type)
        typed = sorted(
            candidate
            for candidate in candidates
            if normalized_type in self._canonical_types.get(candidate, set())
        )
        if typed:
            return typed[0]
        return sorted(candidates)[0]

    def resolve_exact(self, mention: str, hint_type: Optional[str] = None) -> Optional[str]:
        """Resolve via exact alias lookup plus safe normalization steps only."""

        for key in self._candidate_forms(mention, include_articles=True):
            match = self._pick_typed_candidate(self._alias_to_canonicals.get(key, set()), hint_type)
            if match is not None:
                return match
        return None

    def resolve_typed(self, mention: str, hint_type: str) -> Optional[str]:
        """Resolve an alias using a type hint for disambiguation."""

        return self.resolve_exact(mention, hint_type=hint_type)

    def _resolve_embedding(self, mention: str, hint_type: Optional[str]) -> Optional[str]:
        if self._embed_fn is None or not self._embeddings:
            return None
        try:
            query = tuple(float(v) for v in self._embed_fn(mention))
        except Exception:
            return None
        query_norm = sum(v * v for v in query) ** 0.5
        if query_norm == 0:
            return None
        best: tuple[float, Optional[str]] = (0.0, None)
        for canonical, vector in self._embeddings.items():
            if hint_type:
                normalized_type = EntityType.normalize(hint_type)
                if normalized_type not in self._canonical_types.get(canonical, set()):
                    continue
            if len(vector) != len(query):
                continue
            dot = sum(a * b for a, b in zip(query, vector))
            vec_norm = sum(v * v for v in vector) ** 0.5
            if vec_norm == 0:
                continue
            score = dot / (query_norm * vec_norm)
            if score > best[0]:
                best = (score, canonical)
        if best[0] >= self._similarity_threshold:
            return best[1]
        return None

    def resolve(self, mention: str, hint_type: Optional[str] = None) -> Optional[str]:
        """Resolve a mention using the ingestion-safe cascade."""

        exact = self.resolve_exact(mention, hint_type=hint_type)
        if exact is not None:
            return exact
        return self._resolve_embedding(mention, hint_type)

    def resolve_or_create(self, mention: str, hint_type: Optional[str] = None) -> str:
        """Resolve a mention or register a new canonical deterministically."""

        resolved = self.resolve(mention, hint_type=hint_type)
        if resolved is not None:
            return resolved
        canonical = self.normalize(mention)
        self.register(canonical, aliases={mention}, type_hint=hint_type)
        if self._store is not None:
            entity = self._store.entity_graph.ensure_entity(canonical)
            entity.aliases.add(self.normalize(mention))
            if hint_type:
                entity.register_type(hint_type)
                self._store.types.index(canonical, hint_type)
        return canonical


def _lemmatize_verb(predicate: str) -> str:
    text = (predicate or "").strip().casefold()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub("_", text).strip("_")
    if not text:
        return ""
    if text in _IRREGULAR_VERBS:
        return _IRREGULAR_VERBS[text]
    if text.endswith("ies") and len(text) > 3:
        return text[:-3] + "y"
    if text.endswith("ing") and len(text) > 5:
        base = text[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if text.endswith("ed") and len(text) > 4:
        base = text[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if text.endswith("es") and len(text) > 4:
        return text[:-2]
    if text.endswith("s") and len(text) > 3:
        return text[:-1]
    return text


def classify_verb(predicate: str) -> VerbClassification:
    """Classify a predicate into coarse role/sub-role buckets."""

    lemma = _lemmatize_verb(predicate)
    if lemma in HYPOTHETICAL_LEMMAS:
        return VerbClassification(role="event", sub_role="hypothetical")
    if lemma in INTENT_LEMMAS:
        return VerbClassification(role="event", sub_role="intent")
    if lemma in ACQUISITION_LEMMAS:
        return VerbClassification(role="event", sub_role="acquisition")
    if lemma in MAINTENANCE_LEMMAS:
        return VerbClassification(role="state", sub_role="maintenance")
    if lemma in OPINION_LEMMAS:
        return VerbClassification(role="opinion", sub_role="")
    return VerbClassification(role="state", sub_role="")


def _parse_numeric_token(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    multiplier = 1.0
    if text.endswith("k"):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("b"):
        multiplier = 1_000_000_000.0
        text = text[:-1]
    text = text.replace("$", "").replace(",", "").replace("%", "")
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def _parse_number_words(text: str) -> Optional[float]:
    tokens = [token for token in re.split(r"[\s-]+", text.casefold()) if token and token != "and"]
    if not tokens:
        return None
    current = 0
    total = 0
    matched = False
    for token in tokens:
        if token in _NUMBER_WORDS:
            current += _NUMBER_WORDS[token]
            matched = True
            continue
        scale = _NUMBER_SCALES.get(token)
        if scale is None:
            return None
        matched = True
        if current == 0:
            current = 1
        current *= scale
        if scale >= 1000:
            total += current
            current = 0
    if not matched:
        return None
    return float(total + current)


def _numbers_in_source(source_text: str) -> set[float]:
    found: set[float] = set()
    for match in _NUMERIC_RE.finditer(source_text):
        numeric = _parse_numeric_token(match.group(0))
        if numeric is not None:
            found.add(numeric)
    for match in _NUMBER_WORD_RE.finditer(source_text):
        numeric = _parse_number_words(match.group(0))
        if numeric is not None:
            found.add(numeric)
    return found


def _numeric_equivalent(fact: Fact) -> bool:
    fact_value = _parse_numeric_token(fact.object)
    if fact_value is None:
        return False
    return any(abs(candidate - fact_value) < 1e-9 for candidate in _numbers_in_source(fact.source_text))


def verify_grounding(fact: Fact) -> bool:
    """Check whether a fact's object is explicitly grounded in its source text."""

    if not fact.source_text:
        return True
    source = fact.source_text.casefold()
    value = (fact.object or "").casefold().strip()
    if not value:
        return True
    if value in source:
        return True
    if fact.object_type == PropertyType.NUMERIC and _numeric_equivalent(fact):
        return True
    content_words = [word for word in value.split() if len(word) >= 3]
    return all(word in source for word in content_words)


def route_to_layer(fact: Fact, grounding_passed: bool) -> GroundingLayer:
    """Route a fact into stated, inferred, or derived storage."""

    if fact.layer == GroundingLayer.DERIVED:
        return GroundingLayer.DERIVED
    if grounding_passed:
        return GroundingLayer.STATED
    return GroundingLayer.INFERRED


def track_cooccurrences(entities: list[str], chunk_idx: int, cooc_index: CooccurrenceIndex) -> None:
    """Record all unique entity pairs seen in a chunk."""

    if not entities:
        return
    seen: list[str] = []
    normalized_seen: set[str] = set()
    resolver = EntityResolver()
    for entity in entities:
        normalized = resolver.normalize(entity)
        if normalized in normalized_seen:
            continue
        normalized_seen.add(normalized)
        seen.append(normalized)
    for left_idx, left in enumerate(seen):
        for right in seen[left_idx + 1 :]:
            cooc_index.record(left, right, chunk_idx)


def _build_temporal_event(label: str, when: datetime, chunk: RawChunk) -> TemporalEvent:
    return TemporalEvent(
        event=label,
        start_dt=when,
        end_dt=when,
        source=chunk.text,
        turn_id=chunk.turn_idx,
    )


def extract_temporal(chunk: RawChunk, session_time: Optional[datetime] = None) -> list[TemporalEvent]:
    """Extract basic temporal events using deterministic parsing."""

    events: list[TemporalEvent] = []
    anchor = session_time or chunk.session_time
    if anchor is not None:
        events.append(_build_temporal_event("session_time", anchor, chunk))

    seen_datetimes: set[datetime] = set()
    for match in _EXPLICIT_DATE_RE.finditer(chunk.text):
        parsed = _parse_datetime(match.group(0))
        if parsed is None or parsed in seen_datetimes:
            continue
        events.append(_build_temporal_event(match.group(0), parsed, chunk))
        seen_datetimes.add(parsed)

    for match in _MONTH_DAY_RE.finditer(chunk.text):
        text = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", match.group(0), flags=re.IGNORECASE)
        parsed: Optional[datetime] = None
        for fmt in _NATURAL_DATE_FORMATS:
            try:
                candidate = datetime.strptime(text, fmt)
            except ValueError:
                continue
            if candidate.year == 1900 and anchor is not None:
                candidate = candidate.replace(year=anchor.year)
            parsed = candidate
            break
        if parsed is None or parsed in seen_datetimes:
            continue
        events.append(_build_temporal_event(match.group(0), parsed, chunk))
        seen_datetimes.add(parsed)

    lowered = chunk.text.casefold()
    if anchor is not None:
        relative_terms = {
            "today": anchor,
            "yesterday": anchor - timedelta(days=1),
            "tomorrow": anchor + timedelta(days=1),
            "last year": anchor.replace(year=anchor.year - 1),
        }
        for token, resolved in relative_terms.items():
            if token in lowered and resolved not in seen_datetimes:
                events.append(_build_temporal_event(token, resolved, chunk))
                seen_datetimes.add(resolved)
    return events


def _named_entities_from_text(text: str) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\bnamed\s+([A-Z][\w-]*(?:\s+(?:and\s+)?[A-Z][\w-]*)*)", text):
        block = match.group(1)
        for part in re.split(r"\s+and\s+", block):
            name = part.strip(" ,.")
            if not name or name in seen:
                continue
            seen.add(name)
            results.append(name)
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        name = match.group(1).strip()
        if name in {"User", "Assistant", "Chat", "Time"} or name in seen:
            continue
        seen.add(name)
        results.append(name)
    return results


def extract_faceted_entities(
    chunk: RawChunk,
    llm_url: str,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> list[FacetEntity]:
    """Extract faceted entities using deterministic regex heuristics."""

    del llm_url, model

    entities: list[FacetEntity] = []
    names = _named_entities_from_text(chunk.text)
    lowered = chunk.text.casefold()
    type_hint = ""
    pet_match = re.search(r"\b(?:a|an|two|three|\d+)?\s*(cat|dog|parrot|bird|fish|rabbit|hamster)\b", lowered)
    if pet_match:
        type_hint = pet_match.group(1)

    for name in names:
        facets: dict[str, str] = {}
        if type_hint:
            facets["what"] = EntityType.normalize(type_hint)
        if chunk.session_time is not None:
            facets["when"] = chunk.session_time.isoformat()
        entities.append(
            FacetEntity(
                name=name,
                facets=facets,
                turn_idx=chunk.turn_idx,
                source_text=chunk.text,
            )
        )
    return entities


def _user_subject_for_text(text: str) -> str:
    lowered = text.casefold()
    if lowered.startswith(("i ", "i'", "i’m", "i am", "my ", "we ", "we're", "we are")):
        return "user"
    return "user"


def extract_triples(
    chunk: RawChunk,
    llm_url: str,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> list[ExtractedTriple]:
    """Extract simple triples using deterministic regex heuristics."""

    del llm_url, model

    text = chunk.text.strip()
    if not text:
        return []

    triples: list[ExtractedTriple] = []
    subject = _user_subject_for_text(text)

    named_pet = re.search(
        r"\b(?:I|We)\s+(?P<verb>have|has|own|keep|adopted|got|bought)\s+(?:\w+\s+)?(?P<type>cat|dog|parrot|bird|fish|rabbit|hamster)\s+named\s+(?P<names>[A-Z][\w-]*(?:\s+and\s+[A-Z][\w-]*)*)",
        text,
        re.IGNORECASE,
    )
    if named_pet:
        classification = classify_verb(named_pet.group("verb"))
        entity_type = named_pet.group("type")
        for raw_name in re.split(r"\s+and\s+", named_pet.group("names")):
            name = raw_name.strip(" ,.")
            if not name:
                continue
            triples.append(
                ExtractedTriple(
                    subject=name,
                    predicate="type",
                    object=entity_type,
                    role=classification.role,
                    sub_role=classification.sub_role,
                    source_text=chunk.text,
                    turn_idx=chunk.turn_idx,
                    chunk_idx=chunk.chunk_idx,
                )
            )
            triples.append(
                ExtractedTriple(
                    subject=subject,
                    predicate="has",
                    object=name,
                    role=classification.role,
                    sub_role=classification.sub_role,
                    source_text=chunk.text,
                    turn_idx=chunk.turn_idx,
                    chunk_idx=chunk.chunk_idx,
                )
            )

    breed_match = re.search(
        r"\b(?P<subject>[A-Z][a-z]+)\s+is\s+a[n]?\s+(?P<breed>[\w-]+)\s+(?P<type>cat|dog|bird|fish)\b",
        text,
    )
    if breed_match:
        triples.append(
            ExtractedTriple(
                subject=breed_match.group("subject"),
                predicate="breed",
                object=breed_match.group("breed"),
                role="state",
                sub_role="maintenance",
                source_text=chunk.text,
                turn_idx=chunk.turn_idx,
                chunk_idx=chunk.chunk_idx,
            )
        )

    location_match = re.search(
        r"\b(?:I|We)\s+(?P<verb>live in|lives in|moved to|relocated to)\s+(?P<place>[A-Z][A-Za-z\s-]+)\b",
        text,
        re.IGNORECASE,
    )
    if location_match:
        verb = location_match.group("verb").split()[0]
        classification = classify_verb(verb)
        predicate = "moved_to" if "moved" in location_match.group("verb").casefold() else "lives_in"
        triples.append(
            ExtractedTriple(
                subject=subject,
                predicate=predicate,
                object=location_match.group("place").strip(),
                role=classification.role,
                sub_role=classification.sub_role,
                source_text=chunk.text,
                turn_idx=chunk.turn_idx,
                chunk_idx=chunk.chunk_idx,
            )
        )

    numeric_match = re.search(r"\b(?:salary|make|earn)\s+(?:is\s+)?(?P<value>\$?\d[\d,]*(?:k)?)", text, re.IGNORECASE)
    if numeric_match:
        triples.append(
            ExtractedTriple(
                subject=subject,
                predicate="salary",
                object=numeric_match.group("value"),
                role="state",
                sub_role="maintenance",
                source_text=chunk.text,
                turn_idx=chunk.turn_idx,
                chunk_idx=chunk.chunk_idx,
            )
        )
    return triples


class DerivedLayerBuilder:
    """Small deterministic builder for derived counts and metadata."""

    def __init__(self, store: Optional[FoundryStore] = None) -> None:
        self._store = store

    def build(self, store: Optional[FoundryStore] = None) -> dict[str, int]:
        active_store = store or self._store
        if active_store is None:
            return {}
        counts: dict[str, int] = {}
        for entity in active_store.entity_graph.all_entities():
            for type_label in entity.types:
                counts[type_label] = counts.get(type_label, 0) + 1
        active_store.derived.type_counts = dict(sorted(counts.items()))
        return active_store.derived.type_counts

    def extract_gists(self, store: Optional[FoundryStore] = None) -> dict[str, list[dict[str, str]]]:
        active_store = store or self._store
        if active_store is None:
            return {"cooccurrence": [], "progression": [], "convergent": []}
        cooccurrence_gists: list[dict[str, str]] = []
        for entity in active_store.entity_graph.all_entities():
            for neighbor, count in active_store.cooccurrence.get_cooccurrences(entity.canonical).items():
                if entity.canonical < neighbor:
                    cooccurrence_gists.append(
                        {"entity_a": entity.canonical, "entity_b": neighbor, "count": str(count)}
                    )
        return {
            "cooccurrence": cooccurrence_gists,
            "progression": [],
            "convergent": [],
        }


def fact_from_triple(
    triple: ExtractedTriple,
    canonical_subject: Optional[str] = None,
    layer: GroundingLayer = GroundingLayer.STATED,
    session_time: Optional[datetime] = None,
) -> Fact:
    """Convert an extracted triple into a `Fact`."""

    subject = canonical_subject or EntityResolver().normalize(triple.subject)
    return Fact(
        id=f"{subject}:{triple.predicate}:{hashlib.sha1((triple.object + triple.source_text).encode('utf-8', 'ignore')).hexdigest()[:12]}",
        subject=subject,
        predicate=triple.predicate,
        object=triple.object,
        object_type=infer_property_type(triple.object),
        turn_idx=triple.turn_idx,
        chunk_idx=triple.chunk_idx,
        session_time=session_time,
        source_text=triple.source_text,
        layer=layer,
        when=triple.when,
        role=triple.role,
        sub_role=triple.sub_role,
    )


__all__ = [
    "ConversationTurn",
    "EntityResolver",
    "ExtractedTriple",
    "FacetEntity",
    "IngestionReport",
    "VerbClassification",
    "chunk_turns",
    "classify_verb",
    "extract_faceted_entities",
    "extract_temporal",
    "extract_triples",
    "fact_from_triple",
    "infer_type",
    "parse_conversation",
    "route_to_layer",
    "track_cooccurrences",
    "verify_grounding",
]
