# Adversarial Review: Encoding Layer Spec
# Three Passes with Brenner Operators

Created: 2026-02-27
Reviewer: architect-agent (adversarial mode)
Spec under review: `encoding/spec.md`

---

## Pass 1 -- Level-Split + Paradox-Hunt

### Operator: Level-Split ("Where does the spec conflate two things that should be separated?")

#### Finding 1.1: Episode Narratives vs. Raw Messages (CRITICAL)

**The conflation:** The spec's pattern lists (Section 3.1) are written for FIRST-PERSON raw user messages: "I like", "I prefer", "I live in", "No, actually". But the encoding gate evaluates EPISODE NARRATIVES produced by Nemori's `EpisodeGenerator`, which converts raw messages into THIRD-PERSON narratives.

**Evidence from Nemori reference (nemori.md, lines 35-36):**
> `EpisodeGenerator` converts raw messages into third-person narrative
> Key transforms: resolves references ("she" -> "Dr. Chen"), resolves relative time ("yesterday" -> date), compresses filler

**What this means concretely:**

| Raw message (first-person) | Episode narrative (third-person) |
|---|---|
| "I like dark themes for my IDE" | "The user expressed a preference for dark themes in their IDE." |
| "I live in San Francisco" | "The user mentioned that they reside in San Francisco." |
| "No, actually the deadline is March 15th" | "The user corrected the previously stated deadline, clarifying it is March 15th rather than March 10th." |
| "Always use type hints in Python" | "The user instructed the assistant to always include type hints when writing Python code." |
| "Hello!" | Unlikely to become an episode (too few messages; Nemori requires episode_min_messages=2) |

**Impact:** The pattern "I like" will NOT match "The user expressed a preference for". The pattern "I live in" will NOT match "The user mentioned that they reside in". Nearly ALL preference, fact, correction, and instruction patterns will fail on episode narratives. The classifier will fall through to the no-match default, classifying most episodes as "fact" (long) or "greeting" (short) with low confidence, triggering fail-open on everything. **The gate becomes a no-op.**

**Mitigation added to spec:** Section 3.4 "Dual Pattern Sets" -- the gate must have separate pattern sets for first-person raw text and third-person narrative text, selected based on `metadata["source_type"]`.

#### Finding 1.2: Classification vs. Policy (MODERATE)

**The conflation:** The `classify()` method returns a category, and the `_apply_write_policy()` method decides storage. But the spec mixes these concerns in its narrative: "Policy: ALWAYS store" appears under classification pattern definitions (Section 3.1). Classification is "what IS this?"; policy is "what do we DO with it?"

**Impact:** When someone overrides `classify()` for LLM-based classification (Section 7.4), they might assume their classifier also needs to return policy decisions, because the spec describes policy alongside patterns.

**Mitigation added to spec:** Section 3.1 now explicitly marks the "Policy:" lines as references to Section 4, not as part of classification logic. Added clarifying comment.

#### Finding 1.3: Heuristic Confidence vs. Classification Accuracy (MODERATE)

**The conflation:** `confidence = matched_pattern_count / total_patterns_in_category` (Section 3.3) measures pattern coverage, NOT classification correctness. A text matching 3/14 preference patterns gets confidence 0.21, but it IS a preference with near-certainty. A text matching 10/14 preference patterns gets confidence 0.71, but could be a synthetic test string. "Confidence" in the spec means "pattern density" but the fail-open gate treats it as "how sure are we about the classification."

**Impact:** The fail-open gate triggers when confidence < 0.5. With 14-21 patterns per category, matching 1-2 patterns gives confidence 0.05-0.14. Even clear, unambiguous single-pattern matches (like "I prefer X") trigger fail-open. The fail-open mechanism fires on nearly every real input, not just ambiguous ones.

**Mitigation added to spec:** Renamed to `pattern_density` in the algorithmic description and added a minimum confidence floor for single high-signal patterns. Also added a "confidence calibration" note explaining that the confidence score measures pattern density, not classification probability.

---

### Operator: Paradox-Hunt ("Where do two parts of the spec contradict?")

#### Finding 1.4: Fail-Open Negates the Gate (CRITICAL)

**The contradiction:** Section 1 says the gate's purpose is to decide "whether to store." Section 4.2 says fail-open stores everything below confidence 0.5. Section 3.3's confidence formula means nearly every single-pattern match produces confidence < 0.5 (e.g., 1/14 = 0.07). Therefore:

- Categories with ALWAYS-store policy: stored regardless of confidence
- Categories with NEVER-store policy (greeting, transactional): stored via fail-open whenever confidence < 0.5
- Reasoning with conditional policy: stored via fail-open whenever the condition fails AND confidence < 0.5

**The paradox:** The gate says it FILTERS, but the confidence formula ensures that the fail-open override fires on almost every rejection attempt, storing everything anyway. The gate's filtering effect is limited to:
1. High-confidence greetings/transactional (matching 8+ patterns out of 15-17), which is extremely rare
2. Empty strings

**Quantitative estimate:** With typical 1-2 pattern matches per input, confidence will be 0.05-0.15. The threshold is 0.5. Fail-open triggers on >95% of greeting/transactional classifications. The gate is effectively a no-op for filtering.

**Mitigation added to spec:** Section 4.4 "Confidence Calibration" -- introduced a minimum confidence floor for high-signal single-pattern matches (like "hello" matching greeting gets a floor of 0.6, not 1/15=0.067). Added `SINGLE_PATTERN_CONFIDENCE_FLOOR` mapping for patterns that are unambiguous indicators.

#### Finding 1.5: Priority Overlap Creates Ambiguous Boundaries (MODERATE)

**The contradiction:** Section 3.2 says "No, always use spaces" should be classified as correction (highest priority) because it starts with "No,". But the communicative intent is an instruction ("always use spaces"). The priority system resolves syntactically (which pattern matched first) rather than semantically (what the user meant).

**Impact:** "No, I prefer dark mode" is classified as "correction" (importance 0.9) rather than "preference" (importance 0.8). The 0.1 importance difference is small, but the category label is wrong -- the user is not correcting a mistake, they are stating a preference while disagreeing. This is a distinction the heuristic approach cannot resolve.

**Mitigation added to spec:** Section 3.5 "Priority Limitations" -- explicitly documenting that the priority system is a syntactic heuristic that may misclassify communicative intent. For always-store categories, the misclassification only affects `initial_importance` (0.1 delta), not storage. Added note that the LLM classifier (Section 7.4) is the real fix.

#### Finding 1.6: "Deterministic" Claim vs. Metadata Influence (MINOR)

**The contradiction:** Section 5.5 claims "Deterministic: Same (episode_content, metadata) always produces same EncodingDecision." Section 3.3 says metadata boosts confidence by +0.1. If metadata is optional and absent, the confidence differs. So `evaluate("hello")` and `evaluate("hello", metadata={"message_count": 12})` produce different confidences. This is technically deterministic (same inputs = same outputs) but the "no external state" framing is misleading -- metadata IS external state that changes the result.

**Mitigation added to spec:** Clarified that determinism is over the full input tuple (content, metadata), not over content alone. No code change needed.

---

## Pass 2 -- Scale-Check + Materialize

### Operator: Scale-Check ("Are there order-of-magnitude assumptions that haven't been verified?")

#### Finding 2.1: Episode Volume and Gate Rejection Rate (CRITICAL)

**The assumption:** The gate should filter out low-value content to reduce clutter.

**The reality (from Nemori config):**
- `batch_threshold=20` messages triggers segmentation
- `episode_min_messages=2`, `episode_max_messages=25`
- One segmentation produces 1-5 episodes typically
- In a 20-message conversation, maybe 3-4 episodes

**What percentage are filterable?**
- Pure greetings rarely survive Nemori's segmentation (episode_min_messages=2 means a standalone "hello" never becomes an episode)
- Transactional commands ("run tests") accumulate into task-oriented episodes with other messages
- Most episodes mix multiple message types

**Estimate:** In a typical 20-message conversation producing 3-4 episodes:
- 0 pure greeting episodes (greetings bundled with substantive messages)
- 0-1 pure transactional episodes (commands bundled with context)
- 2-4 substantive episodes (preferences, facts, instructions, mixed)

**Conclusion:** The gate will reject ~0-5% of episodes in normal operation. Its primary value is NOT filtering but CLASSIFICATION and IMPORTANCE SEEDING for the dynamics system. The spec should reframe the gate's purpose.

**Mitigation added to spec:** Section 1 reframed -- the gate's primary value is importance seeding and category annotation for dynamics, not filtering. Filtering is a secondary benefit that mainly applies to semantic memories (which can be fine-grained and include trivial extractions).

#### Finding 2.2: Pattern Count vs. False Positive Rate (MODERATE)

**The numbers:**
- 7 categories with 14-21 patterns each (~120 total patterns)
- Case-insensitive substring matching
- Common English words in pattern lists: "so " (reasoning), "since" (reasoning), "always" (instruction), "no " (correction), "start " (transactional)

**False positive analysis for high-frequency patterns:**
- `"so "` matches "I also went to the store" -> NO (no space before "so"... wait, "also " contains "so " at position 2: "al**so **went"). YES, this is a false positive.
- `"since"` matches "I've been here since Monday" (temporal, not causal). False positive for reasoning.
- `"no "` matches "I have no idea what to do" -> correction false positive. Also "I know the answer" -> "k**no **w" contains "no ". False positive.
- `"always"` in "I don't always understand the instructions" -> instruction false positive.
- `"start "` in "That's a good start for the project" -> transactional false positive.

**Scale:** With 120 patterns doing substring matching on 200-500 char episode narratives, expect 3-8 false positive pattern matches per episode. The priority system means the HIGHEST-priority false positive wins.

**Critical concern:** `"no "` is a correction pattern (priority 1). The substring "no " appears in: "know", "another", "cannot", "announce", "innovation", etc. While the spec says `"no "` (with trailing space) is "deliberately specific," the substring "no " appears in many common words when they are followed by spaces in natural text. In third-person narratives: "The user had no objections" matches "no " as a correction.

**Mitigation added to spec:** Section 3.6 "Known False Positive Patterns" -- documented the high-risk patterns and their false positive surface. Added recommendation to use word-boundary-aware matching (`\bno\b` regex) instead of substring matching in implementation. Added a `use_word_boundaries` flag to EncodingConfig (default True).

#### Finding 2.3: Greeting Length Threshold vs. Episode Length (MODERATE)

**The assumption:** `max_greeting_length=50` chars triggers reclassification for greetings.

**The reality:** Nemori's episode narratives are typically 200-500 characters (third-person narratives with resolved references, temporal markers, and compressed filler). A narrative that starts with "hello" will ALWAYS exceed 50 chars. The reclassification step fires on EVERY greeting-classified episode narrative.

**Similarly:** `max_transactional_length=80` chars. Episode narratives almost always exceed this.

**Conclusion:** The length-based reclassification thresholds are calibrated for raw messages, not episode narratives. For episodes, reclassification fires 100% of the time, making the initial greeting/transactional classification and the length check meaningless. Every episode goes through reclassification.

**Mitigation added to spec:** Section 2.1 EncodingConfig now includes `episode_length_offset=150` -- when source_type is "episode", add this offset to the greeting/transactional length thresholds to account for narrative inflation. Alternative: separate thresholds for episode vs. semantic content.

#### Finding 2.4: Fail-Open Trigger Rate (CRITICAL)

**Quantitative analysis:**
- Confidence = matched_patterns / total_patterns_in_category
- Typical single-pattern match: 1/14 to 1/21 = 0.048 to 0.071
- Two-pattern match: 2/14 to 2/21 = 0.095 to 0.143
- Three-pattern match: 3/14 to 3/21 = 0.143 to 0.214
- Threshold: 0.5

**To reach confidence >= 0.5, need:** 7/14 = 0.5 (half the patterns in a category must match)

**Probability of matching 7+ patterns in a single category:** Extremely low for real text. Even a text dense with preference signals ("I like, prefer, enjoy, love, want, tend to, would rather using Python") matches 7 patterns -- but real text rarely contains 7 preference indicators.

**Conclusion:** Fail-open triggers on >95% of all classified content. The gate stores virtually everything. This confirms Finding 1.4 at the quantitative level.

**Mitigation added to spec:** Section 3.3 revised confidence formula to use a calibrated scoring approach: `confidence = min(1.0, base_confidence + pattern_strength_bonus)` where `base_confidence` starts at 0.3 for any single high-signal pattern match, with diminishing returns for additional matches. This ensures a clear "I prefer X" gets confidence ~0.6 (above threshold) rather than 0.07.

---

### Operator: Materialize ("For each claimed behavior, what would I actually SEE?")

#### Finding 2.5: End-to-End Trace Through Nemori Pipeline (CRITICAL)

**Scenario:** User says "I live in San Francisco" in a conversation.

**Step 1: Message enters buffer**
```
Message(role="user", content="I live in San Francisco", timestamp=...)
```

**Step 2: After 20 messages, BatchSegmenter groups messages into episodes**
The segmenter uses an LLM to group by topic. "I live in San Francisco" may be grouped with nearby messages about the user's background.

**Step 3: EpisodeGenerator produces third-person narrative**
```
Episode(
    title="User's Background and Location",
    content="During the conversation, the user shared personal details about their background. They mentioned that they reside in San Francisco and work as a software engineer. The assistant acknowledged this information.",
    original_messages=[...5 messages...],
    boundary_reason="topic_shift"
)
```

**Step 4: Episode enters the encoding gate**
Input to `evaluate()`: `"During the conversation, the user shared personal details about their background. They mentioned that they reside in San Francisco and work as a software engineer. The assistant acknowledged this information."`

**Step 5: Classification with CURRENT spec patterns**
- Check all pattern lists against lowered text
- Preference patterns: "i like", "i prefer", etc. -- NONE match (third-person narrative)
- Fact patterns: "i am", "i'm", "i live in", etc. -- NONE match (text says "they reside in", not "i live in")
- Correction patterns: "no,", "actually," -- NONE match
- Instruction patterns: "always", "never", etc. -- NONE match
- Reasoning connectives: "because", "therefore" -- NONE match
- Greeting patterns: "hello", "thanks" -- NONE match
- Transactional patterns: "run ", "execute" -- NONE match

**Result:** No patterns matched at all. Text is 240 chars (long), so default to "fact" with confidence 0.2. Length > 500? No (240 chars). Metadata boost? Depends. Base confidence: 0.2.

**Gate decision:**
```
EncodingDecision(
    should_store=True,     # fail-open (0.2 < 0.5)
    category="fact",       # default for long unclassifiable
    confidence=0.2,
    reason="No category patterns matched; classified as fact by default [fail-open]",
    initial_importance=0.6  # fact importance
)
```

**Assessment:** The gate CORRECTLY stores this (because it contains a fact), but for the WRONG reason (fail-open default, not because it recognized a fact). The classification is accidentally correct (it IS a fact), but the confidence is garbage (0.2 means "I have no idea").

**Step 6: PredictionCorrectionEngine produces semantic memories**
```
SemanticMemory(
    content="The user lives in San Francisco",
    knowledge_type="personal_fact",
    confidence=0.85
)
```

**Step 7: Semantic memory enters encoding gate**
Input: `"The user lives in San Francisco"`

**Classification with CURRENT patterns:**
- Fact patterns: "i live in" -- DOES NOT match ("The user lives in" != "i live in")
- No other patterns match

**Result:** Short text (33 chars), no matches -> "greeting" with confidence 0.3.
Length 33 < max_greeting_length 50, so no reclassification.
Greeting policy: should_store=False.
Fail-open: 0.3 < 0.5 -> override to should_store=True.

```
EncodingDecision(
    should_store=True,     # fail-open again
    category="greeting",   # WRONG
    confidence=0.3,
    reason="[fail-open]",
    initial_importance=0.0  # WRONG -- this is a fact, not a greeting
)
```

**CATASTROPHIC:** The semantic memory "The user lives in San Francisco" is stored with importance 0.0 (greeting importance) instead of 0.6 (fact importance). The dynamics system will treat this as disposable content. It will decay to zero and never be retrieved.

**Root cause:** The patterns are written for first-person raw text. Both episode narratives AND semantic memory content use third-person phrasing.

**Mitigation added to spec:** This finding drives the dual-pattern-set solution from Finding 1.1. Additionally, for semantic memories, the spec now recommends using `metadata["knowledge_type"]` (provided by Nemori's PredictionCorrectionEngine) as a direct category signal, bypassing pattern matching entirely.

#### Finding 2.6: Semantic Memory knowledge_type Mapping (NEW FINDING)

**Insight from materialization:** Nemori's `SemanticMemory` already includes a `knowledge_type` field (personal_fact, preference, instruction, etc.) assigned by the LLM during predict-calibrate. This is a HIGHER-QUALITY classification signal than heuristic pattern matching, because it was produced by an LLM that understood the full conversation context.

**The spec currently ignores this.** The metadata key `"knowledge_type"` is listed in the evaluate() docstring but never used in classification.

**Mitigation added to spec:** Section 5.6 "Semantic Memory Shortcut" -- when `metadata["source_type"] == "semantic"` and `metadata["knowledge_type"]` is present, map knowledge_type directly to an encoding category, bypassing `classify()`. This uses Nemori's existing LLM classification rather than trying to re-classify with inferior heuristics.

---

## Pass 3 -- Exclusion-Test + Object-Transpose

### Operator: Exclusion-Test ("What forbidden pattern would prove the design is wrong?")

#### Finding 3.1: Misclassification Rate Threshold

**Forbidden pattern:** If >30% of episode narratives are misclassified (wrong category), the heuristic approach provides negative value -- it would be better to store everything with a default importance of 0.5.

**Assessment based on Pass 2 materialization:** With current first-person patterns on third-person narratives, the misclassification rate is approximately 90-95% (essentially nothing matches). After the dual-pattern-set mitigation, the rate depends on how well third-person patterns are designed. Estimated: 15-25% misclassification with third-person patterns, which is below the 30% threshold but marginal.

**Recommendation:** The spec must include a validation gate: run the classifier against a corpus of 50+ real Nemori episode narratives and measure classification accuracy. If accuracy < 70%, escalate to LLM classifier. Added as Section 13.1 "Validation Protocol."

#### Finding 3.2: Fail-Open Trigger Rate Threshold

**Forbidden pattern:** If fail-open triggers >40% of the time, the gate provides no filtering value.

**Assessment with CURRENT confidence formula:** Fail-open triggers on >95% of inputs. The gate is a no-op for filtering. After the confidence calibration mitigation (Finding 2.4), estimated fail-open rate: 15-25% for episodes, 30-40% for semantic memories. This is at the boundary of the 40% threshold.

**Recommendation:** Added monitoring requirement (Section 13.2) -- track fail-open rate in production. If it exceeds 40% over any 7-day window, trigger a review.

#### Finding 3.3: LLM Classifier Escalation Criteria

**When to abandon heuristics entirely:**
1. Misclassification rate >30% on a validation corpus
2. Fail-open rate >40% in production over 7 days
3. False positive rate for correction category >20% (correction has highest importance; false positives are costly)
4. Non-English user base exceeds 20% of traffic

**Added to spec as Section 13.3 "Escalation Criteria."**

---

### Operator: Object-Transpose ("Is there a cheaper system that makes the decisive test trivial?")

#### Finding 3.4: Classify at Raw Message Level BEFORE Nemori (SIGNIFICANT)

**The transpose:** Instead of classifying AFTER Nemori produces episode narratives (third-person, compressed, merged), classify at the RAW MESSAGE level BEFORE Nemori processes them.

**Advantages:**
- First-person patterns ("I like", "I live in") work correctly on raw messages
- No need for third-person pattern sets
- Classification happens per-message, not per-episode (finer granularity)
- Cheaper (no LLM narration step needed before gate decision)

**Disadvantages:**
- Nemori's buffer accumulates 20 messages before segmenting; pre-classification means annotating individual messages, then aggregating annotations per episode
- A message classified as "greeting" might be part of a substantive episode (the "Hello! I just moved to Portland" problem)
- Classification before context is established (predict-calibrate hasn't run yet)

**Assessment:** This is a viable alternative architecture but requires different integration. Instead of gating episodes, we gate MESSAGE ANNOTATIONS that propagate to episodes. An episode's category becomes the highest-priority category among its constituent messages' annotations. An episode is stored if ANY message in it is store-worthy.

**Mitigation added to spec:** Section 7.5 "Alternative Architecture: Pre-Nemori Classification" documents this option as a future consideration. Not adopted for v1 because it requires modifying Nemori's message buffer to carry annotations, which is a larger integration change.

#### Finding 3.5: Use Nemori's knowledge_type Instead of Re-Classifying (SIGNIFICANT)

**The transpose:** For semantic memories, Nemori's `PredictionCorrectionEngine` already classifies content via LLM into knowledge types. Don't re-classify with heuristics; just MAP knowledge_type to encoding category.

**This is strictly cheaper:** Zero computation (just a dictionary lookup) vs. pattern matching against 120+ patterns.

**This is strictly more accurate:** LLM classification on full conversation context vs. substring matching on extracted text.

**Assessment:** This is a clear win. Already captured in Finding 2.6 and added to spec. For episodes, this option is not available (episodes don't have a knowledge_type), so heuristic classification is still needed there.

#### Finding 3.6: Simplified Two-Category System (CONSIDERED, REJECTED)

**The transpose:** Instead of 7 categories, use just 2: "store" (importance 0.7) and "skip" (importance 0.0). Captures 90% of the value with 10% of the complexity.

**Assessment:** This loses the importance seeding, which is the gate's primary value (per Finding 2.1). A correction at importance 0.9 and a greeting at importance 0.0 behave very differently in the dynamics system. The 7-category system provides granular importance seeding that a 2-category system cannot.

**Verdict:** Rejected. The complexity of 7 categories is justified by the importance gradient they provide to the dynamics system.

#### Finding 3.7: Message-Level Metadata as Classifier Input (MODERATE)

**The transpose:** Instead of content analysis, use metadata signals:
- Message position in conversation (first message = likely greeting)
- Message length (short = likely transactional or greeting)
- Message role (system messages = never store)
- Conversation turn count (early turns = more likely greetings)
- Presence of code blocks (= likely transactional)

**Assessment:** These signals are complementary to content patterns, not replacements. A short first message is likely a greeting, but "I'm a Python developer" is also short and first. Metadata signals reduce false positive rates when combined with content patterns.

**Mitigation added to spec:** Section 3.7 "Metadata-Assisted Classification" -- when metadata signals strongly indicate a category, boost confidence for that category by 0.15. Specifically:
- `message_position == 0` + greeting pattern match -> confidence += 0.15
- `len(text) < 20` + transactional/greeting pattern match -> confidence += 0.15
- `has_code_block == True` + transactional pattern match -> confidence += 0.15

---

## Summary of All Mitigations Added to Spec

| # | Finding | Operator | Severity | Mitigation |
|---|---------|----------|----------|------------|
| 1.1 | Episode narratives are third-person; patterns are first-person | Level-Split | CRITICAL | Dual pattern sets (Section 3.4) |
| 1.2 | Classification mixed with policy | Level-Split | MODERATE | Clarifying annotations in Section 3.1 |
| 1.3 | Confidence measures pattern density, not accuracy | Level-Split | MODERATE | Renamed + confidence floor (Section 3.3) |
| 1.4 | Fail-open negates the gate | Paradox-Hunt | CRITICAL | Confidence calibration (Section 4.4) |
| 1.5 | Priority is syntactic, not semantic | Paradox-Hunt | MODERATE | Documented limitation (Section 3.5) |
| 1.6 | Determinism over full input tuple | Paradox-Hunt | MINOR | Clarified in Section 5.5 |
| 2.1 | Gate rejects ~0-5% of episodes | Scale-Check | CRITICAL | Reframed gate purpose to importance seeding (Section 1) |
| 2.2 | High false positive rate on substring patterns | Scale-Check | MODERATE | Word-boundary matching + documentation (Section 3.6) |
| 2.3 | Length thresholds calibrated for raw messages, not episodes | Scale-Check | MODERATE | Episode length offset (Section 2.1) |
| 2.4 | Fail-open triggers >95% of the time | Scale-Check | CRITICAL | Calibrated confidence scoring (Section 3.3) |
| 2.5 | End-to-end trace shows gate is a no-op | Materialize | CRITICAL | Drives all mitigations above |
| 2.6 | Nemori's knowledge_type is a better classifier | Materialize | SIGNIFICANT | Semantic memory shortcut (Section 5.6) |
| 3.1 | Misclassification rate threshold | Exclusion-Test | HIGH | Validation protocol (Section 13.1) |
| 3.2 | Fail-open rate threshold | Exclusion-Test | HIGH | Monitoring requirement (Section 13.2) |
| 3.3 | LLM escalation criteria | Exclusion-Test | HIGH | Documented criteria (Section 13.3) |
| 3.4 | Pre-Nemori classification alternative | Object-Transpose | SIGNIFICANT | Documented as future architecture (Section 7.5) |
| 3.5 | knowledge_type bypass for semantics | Object-Transpose | SIGNIFICANT | Already in Section 5.6 |
| 3.6 | Two-category simplification | Object-Transpose | LOW | Rejected (importance gradient needed) |
| 3.7 | Metadata-assisted classification | Object-Transpose | MODERATE | Metadata confidence boosts (Section 3.7) |

---

## Remaining Risks Not Mitigable at Spec Level

1. **Third-person pattern quality:** The dual pattern set (Finding 1.1) mitigates the mismatch, but the quality of third-person patterns depends on implementation. Nemori's narration style may vary by LLM model. Patterns written for GPT-4o-mini narrations may not match Claude narrations.

2. **Episode merging obscures categories:** Nemori's `EpisodeMerger` merges overlapping episodes (similarity > 0.85). A merged episode may combine a fact and a greeting. The gate sees one merged narrative and must pick one category. The highest-priority pattern wins, which may not represent the episode's overall character.

3. **Predict-calibrate gap dependency:** The semantic memory shortcut (Section 5.6) depends on Nemori's knowledge_type being accurate. If the predict-calibrate LLM produces wrong knowledge_types, the encoding gate inherits those errors. There is no independent validation.

4. **Language coverage:** Patterns are English-only. Non-English content triggers fail-open with importance 0.0 (greeting) or 0.6 (fact). The dynamics system will treat non-English content as low-value, which is wrong for multilingual users.

5. **Adversarial inputs:** A user who knows the pattern lists could craft messages to game classification (e.g., always starting with "No, actually" to get correction importance 0.9). The dynamics system's feedback loop partially mitigates this (irrelevant corrections will decay), but the initial importance boost is free.

---

**End of adversarial review.**
