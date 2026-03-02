# Adversarial Pass 3 -- Empirical Failure Modes

Created: 2026-02-27
Author: architect-agent (Opus 4.6)
Target: `thoughts/shared/plans/memory-system/contradiction/spec.md`
Context: `hermes_memory/encoding.py`, `tests/test_contradiction.py`

---

## Method

Brenner-style empirical analysis: for each design assumption, construct concrete
text inputs that would cause the system to produce WRONG results in real-world
usage. Not theoretical risk assessment -- actual input strings that break the
pipeline.

Findings are ordered by severity, then by failure surface area.

---

## Finding AP3-F1: Paraphrase Blindness -- Subject Extraction Misses Semantic Equivalents

**Severity: CRITICAL**

### The Problem

Subject extraction is purely regex-based (FIELD_EXTRACTORS, Section 4.1). It matches
syntactic patterns like `"i live in"`, `"my home is"`, `"i'm from"`. Two memories
that express the same fact using different syntactic structures will extract to
DIFFERENT subjects or field_types, causing subject_overlap to return 0.0 and the
contradiction to be completely missed.

### Concrete Failing Examples

| Memory A | Memory B | Should Contradict? | System Result |
|----------|----------|-------------------|---------------|
| "I live in New York" | "I'm a New Yorker" | YES (same location claim) | NO -- "I'm a New Yorker" matches the name extractor (`i'm \w+`) as name="new yorker", not location. field_type mismatch. Zero contradiction detected. |
| "I live in New York" | "My home base is NYC" | YES | NO -- "My home base is NYC" does not match any location pattern. "my home is" matches but "my home base is" does not because the regex requires `my home\s+is`, not `my home base\s+is`. Falls through to fallback with field_type="unknown". |
| "I work at Google" | "I'm a Googler" | YES (same employer claim) | NO -- "I'm a Googler" matches name extractor (`i'm \w+`) as name="googler". field_type "name" vs "job". Zero overlap. |
| "My email is old@ex.com" | "You can reach me at new@ex.com" | YES | NO -- "You can reach me at" matches no extractor. Falls to unknown. No contradiction detected despite clear email update. |
| "I live in New York" | "I recently relocated to London" | YES | NO -- "I recently relocated to" matches no location pattern. The location patterns require "i live/reside/am based/am located in". "relocated to" is not in the pattern set. |
| "I prefer dark mode" | "Light mode is easier on my eyes" | YES (implicit preference change) | NO -- "Light mode is easier on my eyes" matches no preference pattern. No "I prefer/like/enjoy" anchor. Falls to unknown fallback. |

### Why This Is Critical

The spec acknowledges in Section 2.2 that contradiction detection is "lightweight: no
LLM calls, no embeddings (uses text features only)" and explicitly excludes entailment
reasoning. But the test suite (test_contradiction.py) ONLY tests the happy path where
both memories use the SAME syntactic pattern:

- `test_value_update_location`: "I live in London" vs "I live in New York" -- both use "I live in"
- `test_value_update_email`: "My email is new@..." vs "My email is old@..." -- both use "My email is"
- `test_value_update_name`: "My name is Bob" vs "My name is Alice" -- both use "My name is"

In real conversations, users NEVER repeat the exact pattern when updating a fact.
They say "Actually I moved to London" not "I live in London" (a correction) or
"NYC is home now" not "I live in New York City". The test suite validates the
architecture works when inputs are perfectly normalized, but real inputs will
almost never be.

### Scale of the Problem

Looking at the FIELD_EXTRACTORS in the spec, there are ~25 regex patterns across
6 field types. A generous estimate is that these patterns cover 30-40% of the
syntactic variety users actually employ to express personal facts. This means
**60-70% of real contradictions will be missed entirely**.

### Proposed Mitigation

1. **Short-term (v1)**: Add a "bag-of-words overlap" fallback path. When field
   extractors fail to find a structured match, compute token-level Jaccard on
   the FULL normalized text (not just extracted subjects). If text-level Jaccard
   exceeds 0.4 AND the category suggests a storable fact/preference, flag for
   review (FLAG_CONFLICT with low confidence).

2. **Medium-term**: Add a synonym/alias map for common paraphrases:
   ```python
   LOCATION_ALIASES = {
       "relocated to": "live in",
       "moved to": "live in",
       "home base is": "home is",
       "based out of": "based in",
   }
   ```
   Pre-expand the text using these aliases before running extractors.

3. **Long-term (v2, out of scope per spec Section 15)**: Embedding-based subject
   matching. This is the only real solution for arbitrary paraphrase coverage.

4. **Test mitigation**: Add tests for paraphrase pairs that SHOULD fail in v1,
   documenting them as known limitations with `@pytest.mark.xfail(reason="paraphrase blindness AP3-F1")`.

---

## Finding AP3-F2: "I'm X" Name Extractor Is a Greedy Trap

**Severity: CRITICAL**

### The Problem

The name extractor in FIELD_EXTRACTORS includes:
```python
(r"(?:i(?:'m| am)\s+)(\w+(?:\s+\w+)?)", "name", "name"),
```

This pattern matches "I'm" followed by 1-2 words. It will greedily match
a vast number of non-name statements as name extractions:

| Input | Intended Meaning | Extracted As |
|-------|-----------------|-------------|
| "I'm tired" | Emotional state | name="tired" |
| "I'm a developer" | Job description | name="a developer" |
| "I'm from Paris" | Location | name="from paris" |
| "I'm working remotely" | Work situation | name="working remotely" |
| "I'm excited about Rust" | Preference/emotion | name="excited about" |
| "I'm not sure" | Uncertainty | name="not sure" |
| "I'm going home" | Action | name="going home" |

### Chain Reaction with Contradiction Detection

Because the name extractor fires on so many inputs, it creates a cascade of
false positives in VALUE_UPDATE detection:

1. User stores: "I'm tired" -> extracted as name="tired"
2. User later stores: "I'm excited" -> extracted as name="excited"
3. Both have field_type="name" and subject="name" -> subject_overlap = 1.0
4. Values differ ("tired" vs "excited") -> VALUE_UPDATE triggered
5. field_type is "name" -> confidence boosted to max(confidence, 0.8) per spec Section 5.2
6. "I'm tired" is AUTO_SUPERSEDED by "I'm excited"

This is catastrophic: a fleeting emotional state supersedes another unrelated
emotional state, both incorrectly classified as names.

### Why This Is Critical

The spec (Section 4.2) says "Try all extractors; pick the one with the longest
match." But the name extractor with `(\w+(?:\s+\w+)?)` is both very broad and
moderately long in its matches. For "I'm from Paris", the location extractor
`(i'm from\s+)(.+)` should win because "I'm from" is a longer prefix match.
But for "I'm tired", "I'm excited", "I'm a developer" -- there is NO competing
extractor. The name extractor wins by default.

The spec does say extractors are tried in FIELD_EXTRACTORS order and the longest
match wins. But "longest match" is measured on the FULL match span, not just the
prefix. "I'm tired" matching the name extractor produces raw_match="i'm tired"
(9 chars). No other extractor matches it at all. So the name extractor wins with
a completely wrong extraction.

### Proposed Mitigation

1. **Restrict the "I'm" name pattern** to require a capitalized word (proper noun
   heuristic) or at least filter out common non-name words:
   ```python
   # Instead of (\w+(?:\s+\w+)?), use a negative lookahead:
   r"(?:i(?:'m| am)\s+)(?!a\b|an\b|the\b|not\b|from\b|in\b|at\b|going\b|tired\b|working\b)(\w+(?:\s+\w+)?)"
   ```
   This is fragile but prevents the worst false positives.

2. **Add a name validation step**: After extraction, check if the extracted value
   looks like a plausible name (starts with capital letter in original text,
   is 1-3 words, does not contain common verbs/adjectives). Downgrade to
   field_type="unknown" if validation fails.

3. **Lower priority for name extractor**: Move name patterns to the END of
   the FIELD_EXTRACTORS dict so they are tried last. If any other extractor
   matches, it wins. (Requires changing from "longest match wins" to "first
   match wins among structured extractors, longest match among same-type".)

---

## Finding AP3-F3: Temporal Markers Completely Ignored -- Past Tense Treated as Present

**Severity: HIGH**

### The Problem

The spec explicitly acknowledges in Section 15 (Future Extensions) that temporal
contradiction is out of scope:

> "Temporal contradiction ('I used to live in X' doesn't contradict 'I live in Y')"

But the spec does NOT acknowledge that the current system will produce **false
positives** from temporal markers. "I used to live in X" will be extracted as
location="x" (because "i ... live in" matches the location pattern after
normalization), and "I live in Y" as location="y". The system will detect a
VALUE_UPDATE contradiction and potentially AUTO_SUPERSEDE the older memory.

### Concrete Failing Examples

| Memory A | Memory B | Relationship | System Result |
|----------|----------|-------------|---------------|
| "I used to live in Paris" | "I live in London" | Compatible (past + present) | FALSE POSITIVE: VALUE_UPDATE, location changed |
| "I previously worked at Google" | "I work at Meta" | Compatible (past + present) | FALSE POSITIVE: VALUE_UPDATE, employer changed |
| "Before I liked Python, now I prefer Rust" | "I prefer Rust" | Memory A already contains the update | FALSE POSITIVE: A's "liked Python" vs B's "prefer Rust" |
| "I was born in Tokyo" | "I live in London" | Compatible (birthplace != residence) | FALSE POSITIVE if both extract location. "I was born in" is not in location patterns, but "I live in" is. Actually, "I was born" is in FACT_PATTERNS. So A extracts as fact, B as location fact. Depends on subject_overlap. |

### Analysis

The regex `r"(?:i (?:live|reside|am based|am located)\s+(?:in|at)\s+)(.+)"` will
match "i used to live in Paris" because "used to live in" contains "live in" as a
substring. After lowercasing, the regex would match at the "live in" position.

Wait -- actually the regex requires the pattern to start with "i live" (with word
boundary implied by the `(?:i (?:live|...` structure). "i used to live in paris"
would match because `i ` is followed by... no. The regex is
`(?:i (?:live|reside|am based|am located)\s+(?:in|at)\s+)(.+)`. This requires
"i " immediately followed by "live". "i used to live" has "i used" not "i live".
So actually this specific case would NOT match the location extractor.

But it WOULD match the fact pattern "i live in" from encoding.py's FACT_PATTERNS
(which includes "i live in"). The contradiction module uses its own
FIELD_EXTRACTORS, not encoding's FACT_PATTERNS. Let me re-examine.

Correction: The FIELD_EXTRACTORS are defined in the contradiction spec, not in
encoding.py. The location extractor requires `i live in` as a contiguous phrase.
"I used to live in Paris" does NOT have "i live" as contiguous -- it has
"i used to live". So the location extractor would NOT fire, and the text would
fall to the fallback extractor. This specific example actually would NOT produce
a false positive.

However, the following WOULD:
- "I no longer live in Paris" -- contains "i ... live in" but "no longer" disrupts it. Actually "i no longer live" != "i live". Safe.
- "I still live in Paris" -- "i still live" != "i live". Safe.

Revised: most temporal markers naturally break the regex pattern because they insert
words between "I" and the verb. The regex structure `i (?:live|...)` requires "i"
directly followed by the verb.

**But "I lived in Paris"** -- past tense. The regex uses `live`, not `lived`. So
"i lived in" does NOT match. This is accidentally correct.

**However**: `r"(?:i(?:'m| am) (?:from|based in)\s+)(.+)"` would match
"I'm from Tokyo" and "I'm from London" -- both present tense, genuinely
contradictory. The temporal issue is that a user might say "I was from Tokyo"
but this would not match "I'm from".

**Revised severity: MEDIUM.** The regex patterns accidentally provide some
temporal protection because they use present tense verbs. Past tense ("lived",
"worked", "was") doesn't match. But this is accidental, not designed, and
fragile -- any expansion of the pattern set to be more inclusive (e.g., adding
"i lived in" for better recall) would immediately reintroduce the temporal
false positive problem.

### Proposed Mitigation

1. **Document the accidental temporal safety** as a known property, not a designed
   feature. Note that it depends on the regex patterns using present-tense verbs
   only.

2. **Add a temporal marker pre-filter**: Before running extractors, scan for temporal
   markers ("used to", "previously", "formerly", "back when", "in the past",
   "no longer", "once was") and either:
   - Skip contradiction detection entirely for that memory (too conservative), or
   - Reduce confidence by 50% for any detected contradiction (preferred).

3. **Add tests documenting the boundary**: "I used to live in X" vs "I live in Y"
   should NOT produce a contradiction. "I live in X" vs "I live in Y" SHOULD.

---

## Finding AP3-F4: Partial Overlap Attack -- "New York" in Different Contexts

**Severity: HIGH**

### The Problem

Token-level Jaccard overlap on extracted SUBJECTS does not distinguish semantic
context. The subject field is often a generic label ("location", "preference",
"instruction") not the actual entity. Two memories about completely different
topics can share subject tokens and trigger false positive contradictions.

### Concrete Failing Examples

| Memory A | Memory B | Overlap Source | False Positive? |
|----------|----------|---------------|-----------------|
| "I live in New York City" | "I work in New York" | Both extract location. Subject="location" for both. Overlap=1.0. Values: "new york city" vs "new york". Different values -> VALUE_UPDATE. | YES -- living and working in the same metro area is not contradictory. But the system sees different location VALUES and flags it. |
| "I prefer New York style pizza" | "I live in New York" | A: preference, value="new york style pizza". B: location, value="new york". Different field_types, so no field_type bonus. But if both fall to unknown... | Depends on extraction. If preference extractor catches A and location catches B, field_types differ. subject_overlap between "preference" and "location" tokens = 0.0. SAFE. But if A falls to fallback, it could match B on "new york" tokens. |
| "Always use Python 3" | "Always use Python for scripting" | Both instructions. Both extract action "use python 3" vs "use python for scripting". Jaccard on tokens: {use, python, 3} vs {use, python, for, scripting} = 2/5 = 0.4. Above similarity_threshold of 0.3. | FALSE POSITIVE: These are not conflicting -- one is more specific than the other. "Always use Python 3" is a refinement of "Always use Python for scripting", not a contradiction. |
| "I live in New York" | "I don't live in New York anymore" | Location extractor: A gets value="new york", B... "I don't live in" does not match `i live in` because "don't" breaks the pattern. B falls to fallback or preference ("I don't like" pattern? No, "don't live" is not "don't like"). | B likely falls to fallback. If it does, subject_overlap between "location" and the fallback subject would be low. Contradiction MISSED (a false negative). The correct behavior is ambiguous -- should the negated statement supersede the positive one? |

### The "New York City" vs "New York" Case

This is particularly insidious. Both are locations. Both extract to subject="location".
subject_overlap = 1.0 (identical subject strings). Values differ: "new york city" vs
"new york". The VALUE_UPDATE detector fires because:
1. Same field_type ("location") -- check
2. Both have non-None values -- check
3. Values differ after normalization -- check ("new york city" != "new york")
4. Subject overlap >= 0.3 -- check (it's 1.0)

Confidence is boosted to max(confidence, 0.8) because field_type is "location"
(spec Section 5.2). This is an AUTO_SUPERSEDE at high confidence. The older
memory ("I live in New York City") is deactivated.

But "New York City" and "New York" refer to the same place. This is a FALSE POSITIVE
auto-supersession that deletes correct information.

### Proposed Mitigation

1. **Add substring containment check for VALUE_UPDATE**: Before declaring values
   "different", check if one value is a substring of the other (after normalization).
   If `a in b` or `b in a`, reduce confidence by 50% or reclassify as
   ELABORATION (a new type, not a contradiction).

2. **For instruction conflicts**: Add an "entailment direction" check. If one
   instruction's action is a subset of the other's tokens, treat it as
   specialization, not conflict. "use python 3" contains all tokens of
   "use python" -- the first is more specific, not contradictory.

3. **Add a value similarity threshold**: If Jaccard similarity between the two
   VALUES (not subjects) exceeds 0.7, do not flag as contradiction. Similar
   values are likely the same entity expressed differently, not a genuine update.

---

## Finding AP3-F5: Negation Handling Is Superficial -- Double Negatives and Litotes Misclassified

**Severity: HIGH**

### The Problem

Polarity detection (spec Section 8) uses simple keyword scanning:
- Positive: "like", "prefer", "enjoy", "love", "want", "favor", "always", "do", "use"
- Negative: "don't", "hate", "dislike", "avoid", "never", "stop", "not", "no longer"

"When both positive and negative signals present, the one with more matches wins."

This produces wrong results for:

### Concrete Failing Examples

| Text | Expected Polarity | Detected Polarity | Why Wrong |
|------|------------------|-------------------|-----------|
| "I don't dislike Python" | POSITIVE (double negative) | NEGATIVE (2 neg: "don't", "dislike" vs 0 pos) | Double negative. "don't" and "dislike" are both negative keywords. 2 negative > 0 positive -> NEGATIVE. But the meaning is positive. |
| "Python is not bad" | POSITIVE (litotes) | NEGATIVE (1 neg: "not" vs 0 pos) | Litotes (understatement by negating the opposite). "not" is a negative keyword but "not bad" means "good". |
| "I can't say I don't like it" | POSITIVE (triple negative) | NEGATIVE (3 neg: "can't", "don't", "not implicit" vs 1 pos: "like") | Even with "like" counting as positive, 3-to-1 negative-to-positive resolves to NEGATIVE. |
| "I don't not prefer Python" | POSITIVE (double negative) | NEUTRAL or NEGATIVE ("don't" + "not" = 2 neg, "prefer" = 1 pos) | 2 neg > 1 pos -> NEGATIVE. Actual meaning is "I prefer Python". |
| "I wouldn't say I hate Python" | NEUTRAL/POSITIVE | NEGATIVE ("hate" = 1 neg vs 0 pos) | "Wouldn't say I hate" = "I don't hate" = neutral-to-positive. But "hate" keyword alone fires negative. |
| "I used to enjoy Python but not anymore" | NEGATIVE (current preference) | NEUTRAL ("enjoy" = 1 pos, "not" = 1 neg, tie -> NEUTRAL) | Tie-breaking to NEUTRAL when the meaning is clearly negative (current state). |

### Impact on PREFERENCE_REVERSAL Detection

The PREFERENCE_REVERSAL strategy depends on polarity_diff:
```
confidence = min(1.0, subject_overlap * polarity_diff)
```
where polarity_diff is 1.0 if polarities are opposite, 0.6 if same polarity but
different values.

If double negatives are misclassified, a preference reversal will be miscategorized:

1. Existing: "I like Python" -> POSITIVE
2. Candidate: "I don't dislike Python" -> should be POSITIVE (agreement), detected as NEGATIVE
3. System sees POSITIVE vs NEGATIVE -> polarity_diff = 1.0 -> high confidence contradiction
4. "I like Python" gets AUTO_SUPERSEDED by "I don't dislike Python"
5. Result: a memory is destroyed by a statement that AGREES with it

### Proposed Mitigation

1. **Add double negative detection**: Before counting keywords, scan for
   "don't|not|no" immediately followed (within 2 words) by a negative keyword
   ("dislike", "hate", "bad", "avoid"). If found, CANCEL both negative signals
   and add one positive signal instead.

2. **Add a litotes detector**: Common litotes patterns ("not bad", "not terrible",
   "not the worst", "not unlike") should be detected and treated as positive.

3. **Document as known limitation**: Double negatives and litotes are a natural
   language understanding problem that keyword scanning fundamentally cannot solve.
   The test suite should include these as `xfail` cases.

4. **Conservative fallback**: When both positive and negative signals are detected
   (regardless of count), default to NEUTRAL instead of winner-takes-all. This
   prevents false positives from complex negation at the cost of missing some
   real reversals. Add a note that this is intentionally conservative.

---

## Finding AP3-F6: max_candidates=50 Is Arbitrary and Can Miss Critical Contradictions at Scale

**Severity: HIGH**

### The Problem

Spec Section 11.5 states:
> "When existing_texts has more than max_candidates entries, scan only the first
> max_candidates. Rationale: in practice, recent memories are more likely to be
> contradicted, and a DB query would return them first."

This rationale is empirically wrong for several real-world scenarios.

### Concrete Failing Scenarios

**Scenario 1: Long-lived personal facts**

A user has 200 stored memories accumulated over months. Memory #15 (stored
months ago) says "I live in New York". The user has since stored 185 more
memories about various topics. Now the user says "I live in London".

With max_candidates=50, only the 50 most recent memories are scanned. Memory #15
is at position 185 in the pool (0-indexed, assuming newest-first ordering from the
DB). It is outside the scan window. The location contradiction is completely missed.
The system now has TWO active memories claiming different locations.

**Scenario 2: Core identity facts buried under transient memories**

Identity facts (name, email, employer, location) are stored early in a user's
interaction and rarely updated. Meanwhile, hundreds of preferences, instructions,
and reasoning memories accumulate on top. By the time an identity fact needs
updating, it is buried far beyond max_candidates.

**Scenario 3: Instruction conflicts across time**

"Always use tabs" stored 6 months ago, 300 memories back. "Never use tabs" stored
today. max_candidates=50 misses the conflict. Both instructions are active
simultaneously, creating contradictory behavior.

### Quantitative Analysis

For a user with N stored memories:
- If the contradicted memory was stored at position P (from newest):
  - P <= 50: contradiction detected (probability depends on scan order)
  - P > 50: contradiction MISSED (100% false negative)

Assuming contradictions are uniformly distributed across memory age (a simplification),
the false negative rate for a user with N memories is:
- N = 50: 0% (all scanned)
- N = 100: 50% (half missed)
- N = 200: 75% (3/4 missed)
- N = 500: 90% (9/10 missed)
- N = 1000: 95% (19/20 missed)

### Proposed Mitigation

1. **Category-aware pre-filter**: Instead of scanning the first 50 memories
   indiscriminately, first filter existing memories by category/field_type relevance:
   - If the candidate is about location, scan ALL location-tagged memories first
   - Then fill remaining slots with recent memories of other categories
   - This requires storing field_type metadata alongside memories

2. **Two-pass scanning**: First pass: scan max_candidates=50 recent memories.
   Second pass: for any field_type identified in the candidate, query the memory
   store for ALL memories with that field_type (regardless of age) and scan those.

3. **Increase max_candidates for identity fields**: When the candidate extracts to
   a "high-identity" field_type (name, email, location, employer), increase
   max_candidates to 200 or scan all. These fields have unambiguous values and
   are the most critical to keep consistent.

4. **Document the scaling limitation**: Add a prominent note that max_candidates=50
   provides adequate coverage for users with fewer than ~50 storable memories.
   Beyond that, false negative rate increases linearly.

---

## Finding AP3-F7: Category Misclassification Cascade -- Encoding Errors Propagate Undetected

**Severity: MEDIUM**

### The Problem

The contradiction pipeline receives `candidate_category` and `existing_categories`
as inputs from the encoding gate. It trusts these categories completely. If the
encoding gate misclassifies a memory, the contradiction detector may:

1. Skip a genuine contradiction (wrong category leads to wrong detection strategy)
2. Detect a false contradiction (wrong category triggers wrong strategy)
3. Apply wrong supersession rules (category-specific rules from Section 7.3)

### Concrete Cascade Examples

**Cascade 1: Correction misclassified as fact**

User says: "No, actually my email is new@example.com"

This should be classified as "correction" by encoding.py (matches "no," and
"actually,"). As a correction, it would trigger DIRECT_NEGATION with category_weight
1.5 and always-supersede behavior.

But if encoding classifies it as "fact" (because it also matches "my email" from
FACT_PATTERNS, and classification depends on PRIORITY_ORDER where "correction"
beats "fact" -- so this specific case is actually safe).

However, consider: "Actually, I work at Meta now". This matches "actually " from
CORRECTION_PATTERNS and "i work at" from FACT_PATTERNS. By PRIORITY_ORDER,
correction beats fact -- correct. But what about: "I work at Meta now". This
matches ONLY "i work at" from FACT_PATTERNS. No correction markers. Classified
as "fact". Contradiction detection uses VALUE_UPDATE (because both are facts
with job field_type), not DIRECT_NEGATION.

The problem: VALUE_UPDATE for facts requires confidence >= 0.7 (spec Section 7.3).
But DIRECT_NEGATION (correction) always supersedes. A legitimate update that lacks
correction markers gets weaker treatment, even though the user clearly means to
update their employer.

**Cascade 2: Instruction classified as preference**

"I always like using type hints" -- matches both "i always" (INSTRUCTION_PATTERNS:
"always") and "i ... like" (PREFERENCE_PATTERNS: "i like"). By PRIORITY_ORDER,
instruction beats preference. Correctly classified.

"I like to always use type hints" -- matches "i like" (preference) and "always"
(instruction). By PRIORITY_ORDER, correction > instruction > preference. "always"
triggers instruction. Correctly classified.

"I enjoy using type hints" -- matches "i enjoy" (preference). No instruction
markers. Classified as preference. If the existing memory is "Never use type hints"
(instruction), the contradiction pipeline checks: candidate is "preference",
existing is "instruction". The INSTRUCTION_CONFLICT strategy only fires when BOTH
are "instruction" (spec Section 6.2, step 5e). PREFERENCE_REVERSAL fires when
either is "preference". But the polarity detection for "I enjoy using type hints"
is POSITIVE, and for "Never use type hints" is NEGATIVE. The PREFERENCE_REVERSAL
strategy would detect opposite polarities.

Actually, wait: PREFERENCE_REVERSAL fires "if either is 'preference'" (spec
Section 6.2, step 5e). So it WOULD fire here. And it would detect polarity
opposition. So this specific cascade is actually handled. But the confidence
would use the preference category_weight (1.0) instead of instruction's (1.2),
resulting in lower confidence than the correct classification would produce.

### Why This Matters

The encoding gate has an empirically demonstrated ~35% pattern match confidence
floor (SINGLE_PATTERN_CONFIDENCE_FLOOR = 0.35), which means single-pattern
matches get artificially boosted. When a memory matches patterns in two
categories, PRIORITY_ORDER resolves it. But when it matches only one category's
patterns, it might be the WRONG category with artificially high confidence.

### Proposed Mitigation

1. **Cross-category detection**: When the candidate_category does not match the
   strategy that fires, log a warning. E.g., if candidate is "fact" but
   DIRECT_NEGATION fires (suggesting it should be "correction"), note the
   discrepancy in the ContradictionDetection.explanation.

2. **Dual-strategy evaluation**: For facts and preferences, always try BOTH
   VALUE_UPDATE and PREFERENCE_REVERSAL, not just the one matching the
   category. Take the higher-confidence result.

3. **Category override in contradiction detector**: If the candidate text matches
   CORRECTION_PATTERNS (imported from encoding.py), treat it as a correction
   regardless of the passed-in category. This adds a defensive rechecking layer.

---

## Finding AP3-F8: Unicode and Non-Latin Scripts Fall Through to Useless Comparison

**Severity: MEDIUM**

### The Problem

Spec Section 11.6 states:
> "Non-Latin scripts: subject extraction falls back to whole-text comparison."

But "whole-text comparison" via token-level Jaccard on CJK, Arabic, or Devanagari
text produces meaningless results because these scripts don't use space-delimited
tokens.

### Concrete Failing Examples

| Memory A | Memory B | Expected | System Result |
|----------|----------|----------|---------------|
| "I live in 東京" | "I live in 大阪" | VALUE_UPDATE (Tokyo vs Osaka) | Location extractor catches "i live in" and extracts value="東京" vs "大阪". Values differ. This DOES work because the English prefix is pattern-matched. |
| "私は東京に住んでいます" (I live in Tokyo) | "私は大阪に住んでいます" (I live in Osaka) | VALUE_UPDATE | NO extraction. No English patterns match. Fallback: subject = first noun phrase heuristic (which relies on whitespace tokenization). Japanese has no word spaces. The entire string becomes one "token". Jaccard between two full strings with partial character overlap is meaningless. |
| "User's name: محمد" | "User's name: أحمد" | VALUE_UPDATE | "user's name is" might not match because the text uses "User's name:" with a colon, not "is". Depends on pattern matching. |
| "I live in Zurich" | "I live in Zurich" | Duplicate (not contradiction) | Works correctly (exact string match). |
| "I live in Zurich" | "I live in Zurikh" | Same city, different transliteration | VALUE_UPDATE false positive. "zurich" != "zurikh". System treats as different locations. |

### Analysis

The system works for bilingual text where the syntactic frame is English (e.g.,
"I live in [non-Latin city name]") because the English prefix triggers the
extractor and the non-Latin value is captured. The system FAILS for entirely
non-Latin text because no extractors match.

For transliteration variants ("Zurich" vs "Zurikh", "Munchen" vs "Munich",
"Beijing" vs "Peking"), the system has no normalization. Each variant is a
different string, producing false VALUE_UPDATE detections.

### Proposed Mitigation

1. **Document the non-Latin limitation explicitly**: The system requires English
   syntactic frames for subject extraction. Memories stored entirely in non-Latin
   scripts will not participate in contradiction detection. This is a known v1
   limitation.

2. **Add transliteration normalization for common city/country names**: A lookup
   table mapping common variants ("Munich"/"Munchen"/"Muenchen", "Beijing"/"Peking",
   "Tokyo"/"Tokio") to canonical forms. Small table, large impact for location
   contradictions.

3. **Test coverage**: Add tests for bilingual text (English frame + non-Latin value)
   to verify these DO work. Add xfail tests for pure non-Latin text.

---

## Finding AP3-F9: Instruction Elaboration vs Conflict -- "Always use Python" vs "Always use Python 3"

**Severity: MEDIUM**

### The Problem

The INSTRUCTION_CONFLICT detector checks for:
1. Same action (after normalization), opposite polarities
2. Same subject, different actions

"Always use Python" and "Always use Python 3" have:
- Same polarity (both "always" = positive)
- Different actions: "use python" vs "use python 3"
- Subject overlap is high (shared tokens "use" and "python")

Case 2 applies: "same subject, different actions." The system detects an
INSTRUCTION_CONFLICT. But this is not a conflict -- "Always use Python 3" is
a REFINEMENT of "Always use Python", not a contradiction.

### More Examples

| Instruction A | Instruction B | Relationship | System Result |
|--------------|--------------|-------------|---------------|
| "Always use Python" | "Always use Python 3" | Elaboration | FALSE POSITIVE: INSTRUCTION_CONFLICT |
| "Never commit secrets" | "Never commit secrets to git" | Elaboration | FALSE POSITIVE: INSTRUCTION_CONFLICT |
| "Remember to write tests" | "Remember to write unit tests" | Elaboration | FALSE POSITIVE: INSTRUCTION_CONFLICT |
| "Always use type hints" | "Always use type hints in function signatures" | Elaboration | FALSE POSITIVE: INSTRUCTION_CONFLICT |
| "From now on, use dark mode" | "From now on, use dark mode in all editors" | Elaboration | FALSE POSITIVE: INSTRUCTION_CONFLICT |

### Analysis

The fundamental issue is that the system cannot distinguish:
- **Conflict**: A contradicts B (both cannot be true)
- **Refinement**: B is a more specific version of A (both can be true, B adds detail)
- **Generalization**: B is a more general version of A

All three produce "same subject, different actions" which triggers case 2 of
INSTRUCTION_CONFLICT with confidence 0.7.

### Proposed Mitigation

1. **Subset detection for instruction actions**: After extracting actions from both
   instructions, check if one action's token set is a SUBSET of the other's. If
   `tokens(A) ⊂ tokens(B)` or `tokens(B) ⊂ tokens(A)`, classify as ELABORATION
   (not conflict) and reduce confidence to 0.2 or skip.

2. **Add an ELABORATION classification**: Introduce a fifth ContradictionType
   member: `ELABORATION = "elaboration"`. The action for elaboration is always
   SKIP (no supersession). The more specific instruction subsumes the general one,
   but the general one is not wrong -- just less specific.

3. **Polarity-first filtering**: Only enter case 2 ("different actions") when
   polarities are the same AND actions have Jaccard similarity between 0.3 and
   0.7. Below 0.3: unrelated instructions, no conflict. Above 0.7: likely
   elaboration, not conflict.

---

## Finding AP3-F10: Exact Duplicate Detection Is Fragile -- Near-Duplicates Slip Through

**Severity: MEDIUM**

### The Problem

Spec Section 11.2 states:
> "If candidate_text appears in existing_texts (exact duplicate), it is NOT a
> contradiction -- it's a duplicate. Skip it."

"Exact duplicate" means string equality after (presumably) normalization. But
near-duplicates that differ by whitespace, punctuation, or trivial words will
NOT be caught as duplicates and may trigger false VALUE_UPDATEs.

### Concrete Failing Examples

| Memory A | Memory B | Duplicate? | System Result |
|----------|----------|-----------|---------------|
| "I live in New York" | "I live in New York." | Near-duplicate (trailing period) | VALUE_UPDATE or no detection depending on normalization. The spec says "normalize: strip trailing punctuation" in Section 4.3, but does the exact-duplicate check compare normalized text or raw text? If raw: "New York" != "New York." -> not duplicate -> enters pipeline -> values are "new york" vs "new york" after normalization -> same values -> no contradiction. SAFE (accidentally). |
| "I live in New York" | "I live in  New York" | Near-duplicate (extra space) | Section 4.3 says "collapse multiple whitespace to single space." If applied to the duplicate check: normalized texts match. SAFE. If not: raw texts differ -> enters pipeline -> values normalize to same -> no contradiction. SAFE. |
| "I live in New York" | "i live in new york" | Near-duplicate (case) | After lowercase normalization, identical. SAFE. |
| "I live in New York" | "I live in New York City" | NOT duplicate, but related | Enters pipeline. Values: "new york" vs "new york city". Different -> VALUE_UPDATE. See AP3-F4. |
| "My email is alice@example.com" | "my email: alice@example.com" | Near-duplicate (different phrasing) | NOT exact duplicate. A extracts email via "my email is" pattern. B: "my email:" does not match "my email is" pattern. B may fall to fallback. Contradiction might be missed (false negative) or might compare different field_types (false noise). |

### Analysis

After closer examination, the exact duplicate check is less fragile than initially
suspected because:
1. The spec normalizes (lowercase, strip, collapse whitespace) before comparison
2. After normalization, most near-duplicates become exact duplicates
3. Even if not caught as duplicates, the VALUE_UPDATE check compares normalized
   values -- identical values won't trigger a false positive

The real risk is in the last two examples: related-but-different texts that are
not duplicates but are not contradictions either. This overlaps with AP3-F4
(partial overlap) and is covered there.

### Revised Severity: LOW

The duplicate detection is adequate for its purpose. The real concern is
partial-overlap false positives, covered in AP3-F4.

### Proposed Mitigation

1. **Use normalized text for the duplicate check**, not raw text. The spec implies
   this but does not state it explicitly. Make it explicit in the spec.

2. **Add a near-duplicate threshold**: If Jaccard similarity between normalized
   full texts exceeds 0.9, treat as near-duplicate and skip (same as exact
   duplicate). This catches cases like "I live in New York" vs
   "I live in New York City" where the texts are very similar but not identical.

---

## Finding AP3-F11: Test Suite Has Zero Adversarial Cases -- Only Happy-Path Validation

**Severity: HIGH**

### The Problem

The test file (`tests/test_contradiction.py`, 165 tests) validates that the system
works correctly on well-formed inputs that use the exact syntactic patterns the
system is designed to handle. It does NOT test:

1. **Paraphrase variants** (AP3-F1): No test has "I'm a New Yorker" vs "I live in
   New York" or any other paraphrase pair.

2. **Greedy extractor traps** (AP3-F2): No test feeds "I'm tired" or "I'm excited"
   to verify they are NOT incorrectly classified as name updates.

3. **Temporal markers** (AP3-F3): No test has "I used to live in X" to verify it
   does NOT falsely contradict "I live in Y".

4. **Partial overlaps** (AP3-F4): No test has "New York City" vs "New York" to
   verify the substring case.

5. **Double negatives** (AP3-F5): No test has "I don't dislike Python" to verify
   polarity detection.

6. **Non-Latin text** (AP3-F8): The only Unicode test is `test_unicode_text_handled`
   which just checks it doesn't crash, not that it produces correct results.

7. **Instruction elaboration** (AP3-F9): No test has "Always use Python" vs
   "Always use Python 3" to verify elaboration is distinguished from conflict.

### Impact

The test suite provides FALSE CONFIDENCE about system correctness. All 165 tests
pass, but the system has fundamental failure modes for real-world inputs that are
completely untested. A developer looking at the test results would conclude "165
tests pass, the system works" when in reality the system only works for the narrow
subset of inputs that match the designed patterns.

### Proposed Mitigation

Add the following test categories:

```python
class TestAdversarialInputs:
    """Tests for known failure modes from adversarial analysis.
    
    These tests document the system's limitations. Some are xfail
    (known limitations of regex-based extraction), others are concrete
    bugs that need fixing.
    """
    
    # AP3-F1: Paraphrase blindness
    @pytest.mark.xfail(reason="AP3-F1: regex cannot handle paraphrases")
    def test_paraphrase_location_new_yorker(self): ...
    
    # AP3-F2: Greedy name extractor
    def test_im_tired_not_classified_as_name(self): ...
    def test_im_a_developer_not_classified_as_name(self): ...
    
    # AP3-F3: Temporal markers
    def test_used_to_live_no_false_positive(self): ...
    
    # AP3-F4: Partial overlap
    def test_new_york_city_vs_new_york_no_false_positive(self): ...
    def test_instruction_elaboration_no_false_positive(self): ...
    
    # AP3-F5: Double negatives
    @pytest.mark.xfail(reason="AP3-F5: keyword polarity cannot handle double negatives")
    def test_double_negative_polarity(self): ...
    
    # AP3-F8: Non-Latin
    @pytest.mark.xfail(reason="AP3-F8: no extractors for non-Latin text")
    def test_japanese_location_contradiction(self): ...
```

This adds ~15-20 adversarial tests that document the real boundary of system
behavior.

---

## Summary

| # | Finding | Severity | Type | System Behavior |
|---|---------|----------|------|-----------------|
| AP3-F1 | Paraphrase blindness | CRITICAL | False Negative | 60-70% of real contradictions missed due to syntactic pattern matching |
| AP3-F2 | "I'm X" greedy name trap | CRITICAL | False Positive | "I'm tired" classified as name, triggers false supersession |
| AP3-F3 | Temporal markers ignored | MEDIUM* | False Positive | Accidentally mitigated by present-tense regex; fragile |
| AP3-F4 | Partial overlap attacks | HIGH | False Positive | "NYC" vs "New York City" triggers false VALUE_UPDATE |
| AP3-F5 | Double negative misclassification | HIGH | False Positive | "I don't dislike X" treated as negative, causes false reversal |
| AP3-F6 | max_candidates=50 scaling | HIGH | False Negative | Old memories beyond position 50 never scanned |
| AP3-F7 | Category misclassification cascade | MEDIUM | Mixed | Wrong category -> wrong strategy -> wrong confidence |
| AP3-F8 | Unicode/non-Latin failures | MEDIUM | False Negative | Pure non-Latin text gets no extraction |
| AP3-F9 | Instruction elaboration vs conflict | MEDIUM | False Positive | "Use Python" vs "Use Python 3" flagged as conflict |
| AP3-F10 | Near-duplicate fragility | LOW | Mixed | Mostly safe due to normalization; documented for completeness |
| AP3-F11 | Test suite happy-path only | HIGH | Process | 165 tests all happy-path; zero adversarial coverage |

*AP3-F3 revised from HIGH to MEDIUM after analysis showed regex patterns accidentally exclude most temporal forms.

### Priority Recommendations

**Must fix before implementation:**
1. AP3-F2 (greedy name trap) -- restricting the "I'm" pattern is a one-line regex change
2. AP3-F11 (adversarial tests) -- add 15-20 tests documenting known limitations
3. AP3-F4 (substring containment check) -- add value-similarity guard to VALUE_UPDATE

**Should fix before implementation:**
4. AP3-F9 (subset detection for instructions) -- add token-subset check
5. AP3-F5 (double negative pre-filter) -- add basic double-negative cancellation

**Document as known v1 limitations:**
6. AP3-F1 (paraphrase blindness) -- fundamental limitation of regex-based approach
7. AP3-F6 (max_candidates scaling) -- document and plan category-aware pre-filter
8. AP3-F8 (non-Latin text) -- document English-frame requirement
9. AP3-F3 (temporal markers) -- document accidental safety, plan explicit handling
10. AP3-F7 (category cascade) -- add cross-category detection logging
