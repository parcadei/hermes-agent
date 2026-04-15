# Engram Memory Architecture: Four-Layer Model

## Overview

An agent's cognitive architecture requires four distinct memory systems working in concert. Each serves a fundamentally different purpose, has different update dynamics, and occupies a different position relative to the context window.

```
┌─────────────────────────────────────────────────────┐
│                   CONTEXT WINDOW                     │
│                   (200K tokens)                      │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Persona  │  │   ATD    │  │ Retrieved │          │
│  │  (~1-2K)  │  │ (~3-5K)  │  │  Engram   │          │
│  │  static   │  │ compiled │  │  chunks   │          │
│  │  identity │  │ behavior │  │ (dynamic) │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │            Working Memory                     │   │
│  │         (conversation + reasoning)            │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
├─────────────────────────────────────────────────────┤
│              PARAMETRIC KNOWLEDGE                    │
│          (frozen in model weights)                   │
│   Python, git, React, HTTP, math, language...        │
│   Everything the base model learned at training.     │
│   Never memorized by Engram. Always available.       │
└─────────────────────────────────────────────────────┘
```

---

## Layer 1: Parametric Knowledge

**What:** Everything the base model learned during training. Languages, libraries, APIs, protocols, common patterns, world knowledge up to the training cutoff.

**Where it lives:** Model weights. Outside the context window entirely.

**Update dynamics:** Frozen. Only changes with model retraining or fine-tuning.

**Key principle:** *Never memorize what you can look up.* Engram should not store "Python uses `def` for functions" or "git push uploads commits." The model already knows this. Storing it wastes both ingestion compute and retrieval bandwidth.

**What belongs here:**
- Language syntax and semantics
- Library APIs and common patterns
- General world knowledge
- Reasoning capabilities

**What does NOT belong here:**
- User-specific facts (→ Engram)
- Project-specific conventions (→ ATD)
- Agent behavioral patterns (→ ATD)
- Identity and personality (→ Persona)

---

## Layer 2: Engram (Searchable World Knowledge)

**What:** Facts about the world as observed through conversations. User preferences, entity relationships, temporal events, conversation history. The agent's long-term declarative memory.

**Where it lives:** External store (entity graph, BM25 index, temporal index, triadic tensor). Retrieved into context window on demand.

**Update dynamics:** Continuous. Every ingested conversation adds facts, supersedes old ones, builds co-occurrence edges.

**Key principle:** *Retrieved, not loaded.* Engram content enters the context window only when a query activates it. A user with 10,000 stored facts only sees the 5-20 relevant ones per query.

**Architecture (current):**
```
ontology.py → storage.py → [query.py, ingestion.py, writeback.py]
    → retrieval.py → core.py → dynamics.py
```

**What belongs here:**
- "User has a cat named Luna"
- "User moved from Seattle to Portland in March 2024"
- "User's favorite restaurant is Bamboo Sushi"
- Entity relationships, temporal events, supersession chains
- Conversation chunks for BM25 retrieval

**What does NOT belong here:**
- How the agent should behave (→ ATD)
- The agent's personality (→ Persona)
- General knowledge (→ Parametric)

---

## Layer 3: ATD (Agentic Trace Distillation)

**What:** Compiled behavioral patterns distilled from agent decision traces across multiple sessions. Not what the agent knows, but *how the agent decides*. The agent's procedural memory.

**Where it lives:** Compiled module (~3-5K tokens) loaded into context at session start. Backed by a trace store for recompilation.

**Update dynamics:** Periodic recompilation. Accumulates raw traces continuously, recompiles the distilled module when sufficient new evidence exists.

**Key principle:** *Compiled, not searched.* Unlike Engram (retrieved per-query), ATD is loaded once at session boot and stays in context for the entire session. It shapes every decision, not just relevant ones.

### 3.1 Trace Collection

Raw traces are the input. Every agent session produces decision traces:

```python
@dataclass
class AgentTrace:
    session_id: str
    timestamp: datetime

    # What happened
    task_type: str              # "debug", "implement", "refactor", "review"
    tool_calls: list[ToolCall]  # sequence of tools used
    decision_points: list[DecisionPoint]  # where the agent chose between alternatives

    # Outcome signal
    user_corrections: int       # times the user corrected the agent
    task_completed: bool        # did the agent finish the task
    context_tokens_used: int    # efficiency metric

    # Self-assessment (optional, from agent reflection)
    what_worked: list[str]
    what_failed: list[str]

@dataclass
class DecisionPoint:
    context: str                # what the agent was deciding
    chosen: str                 # what it chose
    alternatives: list[str]    # what it considered
    rationale: str              # why it chose this
    outcome: str                # how it turned out
```

### 3.2 Distillation

Distillation converts N raw traces into a compact behavioral module. This is the core algorithm:

**Input:** N agent traces (minimum threshold: 10 sessions)

**Process:**
1. **Pattern extraction** — Cluster decision points by task type. For each cluster, identify:
   - Dominant tool sequences (what tools are used in what order)
   - Common decision rationales (why the agent chooses X over Y)
   - Failure modes (what goes wrong and how it's recovered)

2. **Consistency scoring** — For each extracted pattern, compute:
   - Frequency: how often does this pattern appear?
   - Stability: does it persist across sessions or is it noise?
   - Outcome correlation: does following this pattern correlate with success?

3. **Compilation** — Emit patterns that pass all three thresholds as natural language directives:
   ```
   When debugging test failures:
   - Read the failing test first, then the implementation (12/15 sessions)
   - Check git diff before assuming the code is wrong (8/10 sessions)
   - Avoid: running the full test suite before isolating the failure (3 failures from this)
   ```

4. **Budget enforcement** — The compiled ATD must fit in 3-5K tokens. Rank patterns by (frequency × outcome_correlation) and take the top N that fit.

**Output:** A natural language document — the ATD payload.

### 3.3 Recompilation Triggers

The ATD is NOT recompiled on a schedule. It recompiles when evidence demands it:

```python
def should_recompile(current_atd: ATDPayload, trace_buffer: list[AgentTrace]) -> bool:
    # Minimum evidence threshold
    if len(trace_buffer) < 10:
        return False

    # Prediction accuracy: does the current ATD predict recent decisions?
    accuracy = evaluate_atd_predictions(current_atd, trace_buffer[-20:])
    if accuracy < 0.7:  # drift detected
        return True

    # Volume threshold: enough new data to potentially improve
    if len(trace_buffer) >= 30:
        return True

    return False
```

### 3.4 Falsifiability

Every ATD recompilation must be validated:

1. **Held-out evaluation** — Reserve 20% of traces. The new ATD must predict decisions on held-out traces at least as well as the previous ATD.

2. **A/B measurement** — Run the same task with and without ATD loaded. Measure:
   - Tool call count (fewer = more efficient)
   - User corrections (fewer = more aligned)
   - Task completion rate (higher = more capable)
   - Context tokens consumed (fewer = more focused)

3. **Regression gate** — If the new ATD performs worse on ANY metric by more than 5%, roll back to the previous version. The ATD must be monotonically useful.

4. **Context budget ROI** — A 4K token ATD must demonstrably save more than 4K tokens of unnecessary exploration, correction, or re-work per session. If the ROI is negative, the ATD is dropped entirely until more traces accumulate.

### 3.5 Scoped ATD Modules

ATD modules are scoped, not monolithic:

| Scope | What it captures | Loaded when |
|-------|-----------------|-------------|
| **Per-codebase** | "In this repo, tests are in `__tests__/`, use vitest, prefer functional components" | Working in that repo |
| **Per-task-type** | "For debugging: check logs → reproduce → isolate → fix → test" | Task type detected |
| **Per-user** | "This user prefers direct answers, dislikes over-engineering, wants tests first" | User identified |
| **Global** | "When uncertain, ask. When blocked, try alternatives before retrying." | Always |

At session boot, the relevant modules are concatenated (respecting the 3-5K budget):
```
[global ATD] + [per-user ATD] + [per-codebase ATD] + [per-task ATD if known]
```

### 3.6 Architecture in Engram

```python
# New module: foundry/atd.py

@dataclass
class ATDConfig:
    min_traces_for_compilation: int = 10
    recompile_trace_threshold: int = 30
    drift_accuracy_threshold: float = 0.7
    max_payload_tokens: int = 4000
    regression_tolerance: float = 0.05

class TraceStore:
    """Persistent store for raw agent traces."""
    def ingest(self, trace: AgentTrace) -> None: ...
    def traces_since(self, atd_version: int) -> list[AgentTrace]: ...
    def held_out_set(self, fraction: float = 0.2) -> list[AgentTrace]: ...

class ATDCompiler:
    """Distills raw traces into a compiled behavioral module."""
    def compile(self, traces: list[AgentTrace], budget_tokens: int) -> ATDPayload: ...
    def evaluate(self, payload: ATDPayload, traces: list[AgentTrace]) -> ATDMetrics: ...
    def should_recompile(self, current: ATDPayload, new_traces: list[AgentTrace]) -> bool: ...

class ATDPayload:
    """The compiled behavioral module loaded into context."""
    version: int
    scope: str                    # "global", "user:cosimo", "repo:hermes-agent", "task:debug"
    compiled_at: datetime
    trace_count: int              # how many traces contributed
    content: str                  # the natural language directives
    metrics: ATDMetrics           # performance metrics at compilation time

    def bootstrap(self) -> str:
        """Return the content to inject into context at session start."""
        return self.content

@dataclass
class ATDMetrics:
    prediction_accuracy: float    # how well ATD predicts held-out decisions
    avg_tool_calls_with: float    # average tool calls per task with ATD
    avg_tool_calls_without: float # average tool calls per task without ATD
    user_correction_rate: float   # corrections per session
    context_efficiency: float     # tokens saved vs tokens consumed by ATD
```

---

## Layer 4: Persona

**What:** The agent's static identity. Who it is, how it communicates, what its values are. The agent's character — distinct from its learned behaviors (ATD) and its knowledge (Engram).

**Where it lives:** Loaded into context at session start (~1-2K tokens). Authored by the user, the agent, or both.

**Update dynamics:** Rarely. Changes only when the user explicitly edits it or when the agent proposes an update that the user approves.

**Key principle:** *Authored, not learned.* Persona is a creative act, not a statistical distillation. The user decides whether their agent is formal or casual, verbose or terse, cautious or bold. ATD captures behavioral patterns that emerge; Persona captures intentional identity choices.

### 4.1 Persona vs ATD

| Dimension | Persona | ATD |
|-----------|---------|-----|
| **Source** | Authored (user/agent) | Distilled (from traces) |
| **Content** | Identity, values, style | Decision patterns, procedures |
| **Update** | Explicit edits | Automatic recompilation |
| **Example** | "Be direct, no fluff" | "When debugging, check logs first" |
| **Failure mode** | Wrong personality | Inefficient procedures |

### 4.2 Structure

```python
@dataclass
class Persona:
    name: str                     # agent's name (optional)
    voice: str                    # communication style description
    values: list[str]             # core principles ("simplicity over cleverness")
    boundaries: list[str]         # what the agent won't do
    relationship_notes: str       # calibrated rapport with the user

    def bootstrap(self) -> str:
        """Return persona block for context injection."""
        ...
```

### 4.3 Evolution

Persona changes are **proposed, not imposed**:

1. After N sessions (configurable), the agent may reflect: "I've noticed you prefer X over Y. Should I update my persona to reflect this?"
2. The user approves, edits, or rejects.
3. Approved changes are written to the Persona store.
4. Rejected proposals are recorded (the agent should not re-propose the same change).

This keeps the user in control of identity while allowing the agent to surface patterns it notices.

---

## Interaction Between Layers

### Session Boot Sequence

```
1. Load Persona           (~1-2K tokens)  — "who am I"
2. Load ATD modules       (~3-5K tokens)  — "how I operate"
3. Load task context       (variable)      — handoff docs, CLAUDE.md, etc.
4. Begin conversation      (variable)      — user messages
5. Retrieve from Engram    (on demand)     — "what I know about this"
6. Parametric fallback     (always)        — "what the model knows"
```

### Information Flow

```
User says something
    │
    ├─→ Persona shapes HOW the agent responds (tone, style, boundaries)
    │
    ├─→ ATD shapes WHAT PROCESS the agent follows (tool order, decision heuristics)
    │
    ├─→ Engram provides FACTS the agent retrieves (user history, entity data)
    │
    └─→ Parametric provides CAPABILITIES (language, APIs, reasoning)
```

### What Goes Where (Decision Tree)

```
Is this about the agent's identity or communication style?
  YES → Persona

Is this a behavioral pattern that emerged from multiple sessions?
  YES → ATD candidate (verify with falsifiability checks)

Is this a fact about the user, their world, or their conversations?
  YES → Engram

Is this general knowledge any model would know?
  YES → Parametric (don't store it)
```

---

## Design Principles

1. **Don't memorize what you can look up.** Parametric knowledge is free. Engram storage has a cost. Only store what the model can't know.

2. **Don't search what should be loaded.** ATD and Persona are always-on context, not retrieved per-query. The agent's identity and procedures should inform every response, not just relevant ones.

3. **Don't load what should be searched.** Engram facts are demand-driven. Loading all 10,000 user facts at boot wastes context. Retrieve the 10 that matter.

4. **Don't compile what should be authored.** Persona is a creative choice. ATD is a statistical distillation. Don't auto-generate personality. Don't manually write decision procedures.

5. **Everything is falsifiable.** ATD must prove its value through measurable improvement. Persona must be user-approved. Engram must ground facts in source text. No layer gets to consume context "because it looks cool."

6. **Context is finite and precious.** 200K tokens is the working memory budget. Every token spent on Persona/ATD/Engram retrieval is a token unavailable for reasoning. Budget ruthlessly.

---

## Implementation Roadmap

### Phase 1: Engram (Current)
- World knowledge store: entity graph, BM25, temporal, triadic ✓
- Ingestion, retrieval, query, writeback ✓
- 5 feature additions in progress: triadic, decompose, cross-encoder, CR, LLM extraction

### Phase 2: ATD Foundation
- Trace collection: instrument agent sessions to emit AgentTrace objects
- Trace store: persist traces with session metadata
- Basic distillation: frequency-based pattern extraction (no ML)
- Payload format: natural language directives with scope tags

### Phase 3: ATD Validation
- A/B framework: run with/without ATD, measure metrics
- Regression gate: automated rollback on performance drop
- Context ROI tracking: prove the ATD earns its token budget
- Scoped modules: per-codebase and per-task-type compilation

### Phase 4: Persona
- Persona schema and storage
- Boot sequence integration (load Persona + ATD before conversation)
- User-facing editing interface
- Agent-proposed persona evolution with approval gate

### Phase 5: ATD Compression (Parametric Bridge)
- Learned encoding: train compressor to produce short token sequences from natural language ATD
- Behavioral hash validation: A/B test compressed vs full ATD on behavioral metrics
- Context savings measurement: prove compressed form preserves steering with fewer tokens

### Phase 6: Parametric Lookup Keys
- Fine-tune model to associate short keys with behavioral profiles stored in weights
- ATD content migrates from context → parameters via the Hebbian bridge
- Key format: `ATD:<scope>:<hash>` (e.g., `ATD:repo:a3f7b2c1`, `ATD:user:e9d1c4a0`)
- At boot, load keys (10-50 tokens) instead of full ATD (3-5K tokens)

### Phase 7: Integration
- Unified boot sequence across all four layers
- Cross-layer coherence (ATD doesn't contradict Persona)
- Decay/refresh: stale ATD modules expire, stale Persona notes surface for review

---

## Appendix: Parametric Lookup Keys (ATD Phase 6)

### The Core Idea

Natural language ATD directives consume 3-5K tokens of context. But the information-theoretic content — the actual behavioral signal — is far smaller. Most of those tokens are English syntax, not decision-relevant bits.

The parametric lookup key is the endgame of ATD compression: a short opaque token sequence that the model has been fine-tuned to associate with a specific behavioral profile. The behavioral patterns are stored in the model's weights, not in the context. The key is an address into parametric memory.

```
Evolution of ATD representation:

Phase 2:  Natural language    "When debugging, check logs first..."     ~4000 tokens
Phase 5:  Compressed encoding  [learned token sequence]                  ~500 tokens
Phase 6:  Parametric key       ATD:repo:a3f7b2c1                        ~10 tokens
```

### How It Works

**Training (offline, periodic):**

1. Accumulate validated ATD payloads — natural language behavioral directives that have passed the falsifiability gate (proven to improve agent performance).

2. Generate a unique key for each ATD payload. The key is an opaque identifier, not a compression of the content.

3. Fine-tune the model on (key, behavior) pairs:
   - Input: system prompt containing `ATD:repo:a3f7b2c1` + task prompt
   - Target: agent responses that follow the behavioral profile associated with that key
   - The model learns: "when I see this key, I behave this way"

4. Validate: the fine-tuned model with just the key must match the base model with the full natural language ATD on behavioral metrics. If it doesn't, the key hasn't been learned — keep using natural language.

**Inference (runtime):**

```
Session boot:
1. Identify relevant ATD scopes (user, repo, task type)
2. For each scope, check: does a validated parametric key exist?
   YES → load key (10 tokens)
   NO  → load full natural language ATD (3-5K tokens)
3. Total ATD context cost: 10-50 tokens instead of 3-5K
```

### Why This Is Different From Fine-Tuning

Standard fine-tuning bakes behavior into the model globally — every user, every task gets the same behavioral shift. Parametric keys are **addressable**: the behavior only activates when the key is present. Different keys activate different profiles.

```
No key in context     → base model behavior
ATD:user:cosimo       → behavioral profile tuned to Cosimo's preferences
ATD:repo:hermes       → behavioral profile tuned to hermes-agent codebase
ATD:task:debug        → behavioral profile tuned to debugging workflows
```

This is controllable fine-tuning. The weights hold multiple behavioral profiles; the key selects which one is active. Remove the key, behavior reverts to base.

### The Encryption Analogy

The parametric key is "encrypted" behavioral information:

| Concept | Analogy |
|---------|---------|
| Plaintext | Natural language ATD ("When debugging, check logs first...") |
| Ciphertext | The key token (`ATD:repo:a3f7b2c1`) |
| Encryption | Fine-tuning process (maps key → behavioral activation) |
| Decryption | Forward pass (model sees key, activates learned behavior) |
| Key | Model weights (only this model can "decrypt" this behavioral key) |

The key is "unencryptable" in the sense that:
- A human reading `ATD:repo:a3f7b2c1` cannot infer the behavioral profile
- A different model seeing the same key would not activate the same behavior
- The mapping from key to behavior exists only in this model's weight space
- You cannot reverse-engineer the ATD content from the key without access to the model

### Falsifiability

Same framework as natural language ATD, plus compression-specific checks:

1. **Behavioral equivalence** — Fine-tuned model with key must match base model with full ATD on held-out task set. Metric: behavioral similarity ≥ 0.95.

2. **Key isolation** — Removing the key must revert behavior to base. If behavior persists without the key, the fine-tuning leaked globally (failure).

3. **Key specificity** — Different keys must activate different behaviors. If `ATD:task:debug` and `ATD:task:refactor` produce the same behavior, the model hasn't learned the distinction (failure).

4. **Catastrophic forgetting check** — Adding parametric keys must not degrade the model's base capabilities. Standard eval benchmarks must remain within tolerance.

### Open Questions

1. **Key capacity** — How many distinct behavioral profiles can a single model hold before interference? This is an empirical question related to the model's capacity for conditional behavior.

2. **Composability** — Can multiple keys be loaded simultaneously? Does `ATD:user:cosimo` + `ATD:repo:hermes` compose correctly, or do they interfere? If they interfere, what's the resolution strategy?

3. **Versioning** — When an ATD payload is updated (new traces, recompilation), the key must be re-trained. How to handle the transition? Blue/green deployment of keys?

4. **Cross-model transfer** — When the base model is upgraded (Claude 4 → Claude 5), all parametric keys are invalidated. The natural language ATD remains valid. This argues for maintaining both representations — natural language as source of truth, parametric key as optimization.
