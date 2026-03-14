"""Extract memory-worthy facts from parsed conversation turns.

Takes ConversationTurn objects (from parse_real_sessions) and extracts
structured MemoryFact objects suitable for storage in a memory system.

Supports 4-layer memory architecture:
  Layer 1 (user_knowledge): User preferences, facts, corrections, instructions
  Layer 2 (agent_meta): Agent metacognition — thinking blocks, reasoning, findings
  Layer 3 (procedural): How-to knowledge from handoffs (worked/failed/findings/decisions)
  Layer 4 (noise): Filtered out, never stored
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Optional

import yaml

from parse_real_sessions import ConversationTurn


@dataclass
class MemoryFact:
    text: str  # The fact to store in memory
    timestamp: datetime
    project: str
    session_id: str
    source: str  # "user", "assistant", "thinking", "handoff"
    fact_type: str  # "decision", "explanation", "gotcha", "architecture", "debug", "general",
    #                  "self_correction", "finding", "hypothesis",
    #                  "worked", "failed", "procedural_finding", "procedural_decision",
    #                  "reasoning_chain"
    layer: str = "user_knowledge"  # "user_knowledge", "agent_meta", "procedural"
    chain_id: Optional[str] = None  # Links facts in same reasoning chain (e.g. "f0cdc7de:turn42:0")
    chain_position: Optional[int] = None  # Position within chain (0-indexed)


# ---- Acknowledgment patterns (case-insensitive, full-match) ----

_ACKNOWLEDGMENTS = frozenset(
    {
        "ok",
        "okay",
        "proceed",
        "yes",
        "y",
        "sure",
        "thanks",
        "thank you",
        "got it",
        "go ahead",
        "sounds good",
        "looks good",
        "lgtm",
        "correct",
        "right",
        "yep",
        "yeah",
        "yup",
        "ack",
        "k",
        "np",
        "no problem",
        "perfect",
        "great",
        "good",
        "nice",
        "cool",
        "fine",
        "agreed",
        "done",
        "noted",
    }
)

# ---- Action announcement patterns (case-insensitive, prefix match) ----

_ACTION_PREFIXES = [
    r"let me\b",
    r"i'll\b",
    r"i will\b",
    r"now i\b",
    r"i'm going to\b",
    r"first,?\s*let me\b",
    r"here's what i\b",
    r"here is what i\b",
]

_ACTION_RE = re.compile(
    r"^\s*(?:" + "|".join(_ACTION_PREFIXES) + r")", re.IGNORECASE
)


# Regex for system/XML artifacts that should never be in facts
_SYSTEM_ARTIFACT_RE = re.compile(
    r"<(?:"
    r"command-(?:name|message|args)|"
    r"system-reminder|local-command|task-notification|"
    r"task-id|task-status|output-file|teammate-message|"
    r"antml:|user-prompt-submit-hook|"
    r"command-name|summary|result|usage|status"
    r")[>\s/]"
    r"|"
    r"<invoke\s",
    re.IGNORECASE,
)


def _has_system_artifacts(text: str) -> bool:
    """Check if text contains XML system artifacts from Claude Code logs."""
    return bool(_SYSTEM_ARTIFACT_RE.search(text))


def _is_mostly_code(text: str) -> bool:
    """Check if >80% of text is inside triple-backtick code blocks."""
    if not text:
        return False

    # Find all code block regions
    code_chars = 0
    for match in re.finditer(r"```[^\n]*\n(.*?)```", text, re.DOTALL):
        code_chars += len(match.group(0))

    total = len(text.strip())
    if total == 0:
        return False
    return code_chars / total > 0.80


def _is_action_announcement(text: str) -> bool:
    """Check if paragraph starts with an action announcement pattern."""
    return bool(_ACTION_RE.match(text))


def _is_acknowledgment(text: str) -> bool:
    """Check if text is just an acknowledgment/confirmation."""
    normalized = text.strip().lower().rstrip(".!,")
    return normalized in _ACKNOWLEDGMENTS


# ---- Speech-act taxonomy patterns for _classify_fact ----

_SPEECH_CORRECTION_MARKERS = [
    r"\bactually\b",
    r"\bno,\s",
    r"\bwrong\b",
    r"\bcorrection:",
    r"\binstead\s+of\b",
    r"\bdon'?t\s+use\b",
    r"\bnot\s+\w+\s*,\s*(?:but|rather)\b",
    r"\bstop\s+using\b",
    r"\bthat'?s\s+(?:not|in)correct\b",
    r"\bwait\b.*\bactually\b",
    r"\bi\s+was\s+wrong\b",
]
_SPEECH_CORRECTION_RE = re.compile(
    "|".join(_SPEECH_CORRECTION_MARKERS), re.IGNORECASE
)

_SPEECH_INSTRUCTION_MARKERS = [
    r"\b(?:should|must|shall)\s",
    r"\balways\s",
    r"\bnever\s",
    r"\bmake\s+sure\b",
    r"\buse\s+\w+\s+(?:for|to|when|instead)\b",
    r"\bdecided\s+to\b",
    r"\bwe\s+(?:chose|decided|agreed)\b",
    r"^\s*(?:do|run|set|add|remove|configure|enable|disable)\s",
]
_SPEECH_INSTRUCTION_RE = re.compile(
    "|".join(_SPEECH_INSTRUCTION_MARKERS), re.IGNORECASE | re.MULTILINE
)

_SPEECH_PREFERENCE_MARKERS = [
    r"\bprefer\b",
    r"\bi\s+like\b",
    r"\bi'?d\s+rather\b",
    r"\bfavorite\b",
    r"\bpersonally\b",
    r"\bmy\s+(?:style|preference|choice)\b",
]
_SPEECH_PREFERENCE_RE = re.compile(
    "|".join(_SPEECH_PREFERENCE_MARKERS), re.IGNORECASE
)

_SPEECH_REASONING_MARKERS = [
    r"\btherefore\b",
    r"\bwhich\s+means\b",
    r"\bthis\s+implies\b",
    r"\bso\s+(?:the|we|it)\b",
    r"\b(?:because|since)\b.*\b(?:therefore|so|thus)\b",
]
_SPEECH_REASONING_RE = re.compile(
    "|".join(_SPEECH_REASONING_MARKERS), re.IGNORECASE
)


def _classify_fact(text: str, source: str = "assistant") -> str:
    """Classify fact type using speech-act taxonomy.

    Returns one of the keys in coupled_engine._CATEGORY_BOOST:
      "correction", "instruction", "preference", "fact", "reasoning_chain",
      "reasoning".

    Classification priority (highest to lowest):
      1. correction  -- speaker corrects/overrides prior information
      2. instruction -- speaker directs an action or establishes a rule
      3. preference  -- speaker expresses a non-binding preference
      4. reasoning_chain -- multi-step inference with connectives
      5. fact        -- default fallback (objective statement)

    The source parameter is available for callers but does NOT change
    classification logic. Classification is based purely on text content.
    """
    # 1. Correction markers (highest priority)
    if _SPEECH_CORRECTION_RE.search(text):
        return "correction"

    # 2. Instruction markers
    if _SPEECH_INSTRUCTION_RE.search(text):
        return "instruction"

    # 3. Preference markers
    if _SPEECH_PREFERENCE_RE.search(text):
        return "preference"

    # 4. Reasoning chain markers (text blocks only)
    if _SPEECH_REASONING_RE.search(text):
        return "reasoning_chain"

    # 5. Everything else -> "fact"
    return "fact"


# ---- Metacognition markers for thinking blocks (Layer 2) ----

_CORRECTION_MARKERS = [
    r"wait\s*[—–-]",
    r"actually\s+no",
    r"that\s+doesn't\s+add\s+up",
    r"i\s+was\s+wrong",
    r"i\s+assumed\s+.+\s+but",
    r"hmm,?\s+that's\s+not",
    r"on\s+second\s+thought",
    r"let\s+me\s+reconsider",
    r"i\s+need\s+to\s+rethink",
    r"that\s+contradicts",
    r"my\s+earlier\s+assumption",
    r"i\s+overlooked",
    r"correction:",
    r"no,?\s+that's\s+wrong",
]
_CORRECTION_RE = re.compile("|".join(_CORRECTION_MARKERS), re.IGNORECASE)

_FINDING_MARKERS = [
    r"this\s+(?:is|means|suggests|reveals|shows)",
    r"interesting(?:ly)?",
    r"key\s+(?:finding|insight|observation)",
    r"the\s+(?:root\s+cause|real\s+issue|problem\s+is)",
    r"(?:i|we)\s+(?:discovered|found|noticed|realized)",
    r"the\s+pattern\s+(?:is|here)",
    r"important(?:ly)?:\s",
]
_FINDING_RE = re.compile("|".join(_FINDING_MARKERS), re.IGNORECASE)

_HYPOTHESIS_MARKERS = [
    r"(?:my\s+)?hypothesis\s+is",
    r"i\s+think\s+(?:this|the|what)",
    r"(?:if|maybe)\s+.+\s+then\s+.+\s+would",
    r"the\s+reason\s+(?:is|might\s+be|could\s+be)",
    r"(?:because|since)\s+.+(?:therefore|so)\s",
    r"this\s+(?:implies|means\s+that|would\s+explain)",
]
_HYPOTHESIS_RE = re.compile("|".join(_HYPOTHESIS_MARKERS), re.IGNORECASE)

# ---- Text block classification markers (Layer 2 vs noise) ----

_TEXT_CORRECTION_MARKERS = [
    r"\bi\s+was\s+wrong\b",
    r"\bthat'?s\s+(?:not|in)correct\b",
    r"\bactually\s+no\b",
    r"\bcorrection:",
    r"\bi\s+(?:made\s+a\s+mistake|need\s+to\s+correct)\b",
    r"\blet\s+me\s+correct\b",
    r"\bi\s+(?:apologize|misspoke)\b",
]
_TEXT_CORRECTION_RE = re.compile("|".join(_TEXT_CORRECTION_MARKERS), re.IGNORECASE)

_TEXT_FINDING_MARKERS = [
    r"this\s+is\s+revealing",
    r"interesting",
    r"key\s+finding",
]
_TEXT_FINDING_RE = re.compile("|".join(_TEXT_FINDING_MARKERS), re.IGNORECASE)

_TEXT_REASONING_MARKERS = [
    r"\bbecause\b",
    r"root\s+cause",
    r"the\s+reason",
]
_TEXT_REASONING_RE = re.compile("|".join(_TEXT_REASONING_MARKERS), re.IGNORECASE)

_TEXT_DECISION_MARKERS = [
    r"the\s+approach",
    r"we\s+need\s+to",
    r"we\s+should",
    r"decided\s+to",
    r"the\s+fix\s+is",
]
_TEXT_DECISION_RE = re.compile("|".join(_TEXT_DECISION_MARKERS), re.IGNORECASE)


def _classify_thinking_paragraph(text: str) -> Optional[str]:
    """Classify a thinking-block paragraph into a metacognition type.

    Returns the fact_type if the paragraph contains metacognition markers,
    or None if it should be skipped (generic reasoning without markers).
    """
    if _CORRECTION_RE.search(text):
        return "self_correction"
    if _FINDING_RE.search(text):
        return "finding"
    if _HYPOTHESIS_RE.search(text):
        return "hypothesis"
    return None


def _is_metacognitive_text(text: str) -> Optional[str]:
    """Check if an assistant text block contains metacognition (Layer 2).

    Returns fact_type if metacognitive, None if not.
    Text blocks that are corrections, findings, reasoning, or decisions
    get routed to Layer 2 (agent_meta) instead of Layer 1 (user_knowledge).

    Return values are mapped to _CATEGORY_BOOST keys:
      correction -> "self_correction" (agent correcting itself)
      finding    -> "fact"            (agent observation, factual)
      reasoning  -> "fact"            (explanation is a factual statement)
      decision   -> "instruction"     (a decision establishes a directive)

    Priority order matches _classify_fact: correction > finding > reasoning > decision.
    """
    if _TEXT_CORRECTION_RE.search(text):
        return "self_correction"
    if _TEXT_FINDING_RE.search(text):
        return "fact"
    if _TEXT_REASONING_RE.search(text):
        return "fact"
    if _TEXT_DECISION_RE.search(text):
        return "instruction"
    return None


def _extract_thinking_facts(
    turn: ConversationTurn,
    turn_index: int = 0,
) -> list[MemoryFact]:
    """Extract Layer 2 metacognition facts from thinking blocks.

    Splits thinking text on double newlines (paragraph level), then keeps
    only paragraphs with metacognition markers (self-correction, findings,
    hypotheses). Unmarked paragraphs are discarded as generic reasoning.

    Consecutive metacognitive paragraphs are linked into reasoning chains
    via chain_id and chain_position fields. When 2+ consecutive paragraphs
    are metacognitive, a synthesized chain fact is also produced with the
    arc pattern (e.g. "FIND→CORR→FIND") as a prefix.
    """
    facts: list[MemoryFact] = []
    if not turn.thinking_text:
        return facts

    paragraphs = re.split(r"\n\n+", turn.thinking_text)

    # Phase 1: classify all valid paragraphs
    classified: list[tuple[str, str, Optional[str]]] = []  # (para, fact_type_or_none, arc_label)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) < 60 or len(para) > 1500:
            continue
        fact_type = _classify_thinking_paragraph(para)
        if fact_type is not None:
            arc = _ARC_LABELS.get(fact_type, "GEN")
            classified.append((para, fact_type, arc))
        else:
            classified.append((para, None, None))

    # Phase 2: identify chains (consecutive metacognitive paragraphs)
    chains: list[list[int]] = []  # list of runs of consecutive meta indices
    current_run: list[int] = []
    for i, (_, fact_type, _) in enumerate(classified):
        if fact_type is not None:
            current_run.append(i)
        else:
            if len(current_run) >= 2:
                chains.append(current_run[:])
            current_run = []
    if len(current_run) >= 2:
        chains.append(current_run[:])

    # Build chain_id map: index -> (chain_id, chain_position)
    chain_map: dict[int, tuple[str, int]] = {}
    chain_id_base = f"{turn.session_id[:8]}:t{turn_index}"
    for ci, chain_indices in enumerate(chains):
        cid = f"{chain_id_base}:c{ci}"
        for pos, idx in enumerate(chain_indices):
            chain_map[idx] = (cid, pos)

    # Phase 3: emit individual facts with chain tags
    for i, (para, fact_type, arc) in enumerate(classified):
        if fact_type is None:
            continue
        chain_info = chain_map.get(i)
        # Map internal thinking labels to _CATEGORY_BOOST keys
        boost_key = _THINKING_TO_BOOST_KEY.get(fact_type, fact_type)
        facts.append(
            MemoryFact(
                text=para,
                timestamp=turn.timestamp,
                project=turn.project,
                session_id=turn.session_id,
                source="thinking",
                fact_type=boost_key,
                layer="agent_meta",
                chain_id=chain_info[0] if chain_info else None,
                chain_position=chain_info[1] if chain_info else None,
            )
        )

    # Phase 4: emit synthesized chain facts
    for ci, chain_indices in enumerate(chains):
        cid = f"{chain_id_base}:c{ci}"
        arc_parts = [classified[idx][2] for idx in chain_indices]
        arc_pattern = "→".join(arc_parts)

        # Concatenate paragraphs with newline separation, capped at 2000 chars
        chain_paras = [classified[idx][0] for idx in chain_indices]
        chain_text = "\n\n".join(chain_paras)
        if len(chain_text) > 2000:
            # Truncate to fit, preserving whole paragraphs
            truncated = []
            total = 0
            for p in chain_paras:
                if total + len(p) > 1900:
                    break
                truncated.append(p)
                total += len(p) + 2
            chain_text = "\n\n".join(truncated)

        facts.append(
            MemoryFact(
                text=f"[{arc_pattern}] {chain_text}",
                timestamp=turn.timestamp,
                project=turn.project,
                session_id=turn.session_id,
                source="thinking",
                fact_type="reasoning_chain",
                layer="agent_meta",
                chain_id=cid,
                chain_position=None,  # Synthesis fact, not a position
            )
        )

    return facts


# Arc label mapping for chain patterns
_ARC_LABELS = {
    "self_correction": "CORR",
    "finding": "FIND",
    "hypothesis": "HYP",
}

# Mapping from internal _classify_thinking_paragraph labels to _CATEGORY_BOOST
# keys. Internal labels are preserved for arc-pattern generation (above), but
# fact_type stored in MemoryFact must be a _CATEGORY_BOOST key.
_THINKING_TO_BOOST_KEY = {
    "self_correction": "self_correction",
    "finding": "finding",
    "hypothesis": "hypothesis",
}


def _extract_assistant_facts(
    turn: ConversationTurn,
) -> list[MemoryFact]:
    """Extract facts from assistant text in a turn."""
    facts: list[MemoryFact] = []
    if not turn.assistant_text:
        return facts

    # Split on double newlines (paragraph level)
    paragraphs = re.split(r"\n\n+", turn.assistant_text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Skip system artifacts
        if _has_system_artifacts(para):
            continue

        # Skip paragraphs that look like raw JSON
        if para.strip().startswith("{") or para.strip().startswith("["):
            continue

        # Skip paragraphs that are mostly special characters (tables, separators, etc.)
        alpha_chars = sum(1 for c in para if c.isalpha())
        if len(para) > 0 and alpha_chars / len(para) < 0.4:
            continue

        # Length filters
        if len(para) < 50 or len(para) > 500:
            continue

        # Skip mostly-code paragraphs
        if _is_mostly_code(para):
            continue

        # Skip action announcements
        if _is_action_announcement(para):
            continue

        # Skip pure file path lists (lines that are all paths or bullet items
        # starting with -)
        lines = para.split("\n")
        path_like = sum(
            1
            for ln in lines
            if re.match(r"^\s*[-*]\s*/", ln) or re.match(r"^\s*/[\w/]+", ln)
        )
        if path_like > 0 and path_like >= len(lines) * 0.8:
            continue

        # Classify: metacognitive text blocks → Layer 2, others → Layer 1
        meta_type = _is_metacognitive_text(para)
        if meta_type is not None:
            facts.append(
                MemoryFact(
                    text=para,
                    timestamp=turn.timestamp,
                    project=turn.project,
                    session_id=turn.session_id,
                    source="assistant",
                    fact_type=meta_type,
                    layer="agent_meta",
                )
            )
        else:
            facts.append(
                MemoryFact(
                    text=para,
                    timestamp=turn.timestamp,
                    project=turn.project,
                    session_id=turn.session_id,
                    source="assistant",
                    fact_type=_classify_fact(para),
                    layer="user_knowledge",
                )
            )

    return facts


def _extract_user_facts(
    turn: ConversationTurn,
) -> list[MemoryFact]:
    """Extract facts from user text in a turn."""
    facts: list[MemoryFact] = []
    if not turn.user_text:
        return facts

    text = turn.user_text.strip()

    # Skip system artifacts
    if _has_system_artifacts(text):
        return facts

    # Length filter for user messages
    if len(text) < 30:
        return facts

    # Skip acknowledgments
    if _is_acknowledgment(text):
        return facts

    facts.append(
        MemoryFact(
            text=text,
            timestamp=turn.timestamp,
            project=turn.project,
            session_id=turn.session_id,
            source="user",
            fact_type=_classify_fact(text),
            layer="user_knowledge",
        )
    )

    return facts


def extract_facts_from_turn(
    turn: ConversationTurn, turn_index: int = 0
) -> list[MemoryFact]:
    """Extract facts from a single conversation turn (Layers 1 + 2)."""
    facts: list[MemoryFact] = []
    facts.extend(_extract_assistant_facts(turn))
    facts.extend(_extract_user_facts(turn))
    facts.extend(_extract_thinking_facts(turn, turn_index=turn_index))
    return facts


def extract_facts(turns: list[ConversationTurn]) -> list[MemoryFact]:
    """Extract memory-worthy facts from all conversation turns."""
    all_facts: list[MemoryFact] = []
    for i, turn in enumerate(turns):
        all_facts.extend(extract_facts_from_turn(turn, turn_index=i))
    return all_facts


# ---------------------------------------------------------------------------
# Layer 3: Procedural memory from handoff YAML files
# ---------------------------------------------------------------------------


def _parse_handoff_yaml(path: str) -> dict:
    """Parse a handoff YAML file, handling the multi-document format.

    Handoff files have two '---' delimiters: metadata header, then main content.
    """
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    sections = content.split("---")
    if len(sections) < 3:
        try:
            return yaml.safe_load(content) or {}
        except yaml.YAMLError:
            return {}

    try:
        return yaml.safe_load(sections[2]) or {}
    except yaml.YAMLError:
        return {}


def extract_handoff_facts(
    path: str,
    project: str = "",
) -> list[MemoryFact]:
    """Extract Layer 3 (procedural) facts from a handoff YAML file.

    Extracts from:
      - worked: What approaches succeeded (fact_type="worked")
      - failed: What approaches failed (fact_type="failed")
      - findings: Discoveries and insights (fact_type="procedural_finding")
      - decisions: Architectural/design decisions (fact_type="procedural_decision")

    Short entries (< 50 chars) are enriched with the handoff's goal context.
    """
    data = _parse_handoff_yaml(path)
    if not data:
        return []

    facts: list[MemoryFact] = []
    goal = data.get("goal", "")
    session_id = Path(path).stem

    # Derive a timestamp from filename if possible (format: YYYY-MM-DD_HH-MM_*)
    timestamp = datetime(2026, 1, 1)
    fname = Path(path).stem
    ts_match = re.match(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})", fname)
    if ts_match:
        try:
            timestamp = datetime(
                int(ts_match.group(1)),
                int(ts_match.group(2)),
                int(ts_match.group(3)),
                int(ts_match.group(4)),
                int(ts_match.group(5)),
            )
        except ValueError:
            pass

    def _enrich(text: str) -> str:
        """Prepend goal context to short entries for retrievability."""
        if len(text) < 50 and goal:
            return f"[{goal}] {text}"
        return text

    # worked entries
    for item in data.get("worked", []) or []:
        if isinstance(item, str) and len(item.strip()) > 10:
            facts.append(
                MemoryFact(
                    text=_enrich(item.strip()),
                    timestamp=timestamp,
                    project=project,
                    session_id=session_id,
                    source="handoff",
                    fact_type="worked",
                    layer="procedural",
                )
            )

    # failed entries
    for item in data.get("failed", []) or []:
        if isinstance(item, str) and len(item.strip()) > 10:
            facts.append(
                MemoryFact(
                    text=_enrich(item.strip()),
                    timestamp=timestamp,
                    project=project,
                    session_id=session_id,
                    source="handoff",
                    fact_type="failed",
                    layer="procedural",
                )
            )

    # findings — can be dict or list
    findings_raw = data.get("findings") or {}
    if isinstance(findings_raw, dict):
        for key, val in findings_raw.items():
            if isinstance(val, str) and len(val.strip()) > 10:
                facts.append(
                    MemoryFact(
                        text=f"{key}: {val.strip()}",
                        timestamp=timestamp,
                        project=project,
                        session_id=session_id,
                        source="handoff",
                        fact_type="procedural_finding",
                        layer="procedural",
                    )
                )
    elif isinstance(findings_raw, list):
        for item in findings_raw:
            if isinstance(item, str) and len(item.strip()) > 10:
                facts.append(
                    MemoryFact(
                        text=_enrich(item.strip()),
                        timestamp=timestamp,
                        project=project,
                        session_id=session_id,
                        source="handoff",
                        fact_type="procedural_finding",
                        layer="procedural",
                    )
                )
            elif isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(v, str) and len(v.strip()) > 10:
                        facts.append(
                            MemoryFact(
                                text=f"{k}: {v.strip()}",
                                timestamp=timestamp,
                                project=project,
                                session_id=session_id,
                                source="handoff",
                                fact_type="procedural_finding",
                                layer="procedural",
                            )
                        )

    # decisions — can be dict or list
    decisions_raw = data.get("decisions") or {}
    if isinstance(decisions_raw, dict):
        for key, val in decisions_raw.items():
            if isinstance(val, str) and len(val.strip()) > 10:
                facts.append(
                    MemoryFact(
                        text=f"{key}: {val.strip()}",
                        timestamp=timestamp,
                        project=project,
                        session_id=session_id,
                        source="handoff",
                        fact_type="procedural_decision",
                        layer="procedural",
                    )
                )
    elif isinstance(decisions_raw, list):
        for item in decisions_raw:
            if isinstance(item, str) and len(item.strip()) > 10:
                facts.append(
                    MemoryFact(
                        text=_enrich(item.strip()),
                        timestamp=timestamp,
                        project=project,
                        session_id=session_id,
                        source="handoff",
                        fact_type="procedural_decision",
                        layer="procedural",
                    )
                )
            elif isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(v, str) and len(v.strip()) > 10:
                        facts.append(
                            MemoryFact(
                                text=f"{k}: {v.strip()}",
                                timestamp=timestamp,
                                project=project,
                                session_id=session_id,
                                source="handoff",
                                fact_type="procedural_decision",
                                layer="procedural",
                            )
                        )

    return facts


def extract_all_handoff_facts(
    handoffs_dir: str,
    project: str = "",
) -> list[MemoryFact]:
    """Extract Layer 3 facts from all handoff YAML files in a directory tree."""
    all_facts: list[MemoryFact] = []
    handoffs_path = Path(handoffs_dir)

    for yaml_path in sorted(handoffs_path.rglob("*.yaml")):
        facts = extract_handoff_facts(str(yaml_path), project=project)
        all_facts.extend(facts)

    all_facts.sort(key=lambda f: f.timestamp)
    return all_facts
