# Encoding Layer Audit Findings

## Critical Issues (6)

### Finding 1: Regex Compilation Performance
- **Severity**: CRITICAL
- **File**: encoding.py (pattern matching functions)
- **Issue**: Regex patterns compiled on every match call
- **Risk**: ~240 regex compilations per classification call
- **Fix**: Pre-compile all regexes in `__init__` and store in instance variable
- **Operator**: ⊞ Scale-Check

### Finding 2: ReDoS Vulnerability
- **Severity**: CRITICAL
- **File**: encoding.py (re.search calls)
- **Issue**: User-controlled text in `re.search` can cause catastrophic backtracking
- **Risk**: `"no " * 10000 + "idea"` causes exponential backtracking
- **Fix**: Use `re.search` with timeout or limit input length
- **Operator**: ΔE Exception-Quarantine

### Finding 3: Semantic Memory Shortcut Bypasses Write Policy
- **Severity**: CRITICAL
- **File**: encoding.py (semantic shortcut section)
- **Issue**: Short reasoning semantic memories rejected despite LLM validation
- **Risk**: knowledge_type="reasoning" with 28-char text fails length check
- **Fix**: Bypass reasoning conditional when source_type="semantic"
- **Operator**: † Theory-Kill

### Finding 4: Reclassification Returns Excluded Category
- **Severity**: CRITICAL
- **File**: encoding.py (reclassification logic)
- **Issue**: Fallback returns the same category being excluded (dead code)
- **Risk**: Reclassification becomes a no-op
- **Fix**: Return "fact" for long text, "greeting" for short when no alternate found
- **Operator**: † Theory-Kill

### Finding 5: Unicode Text Not Normalized
- **Severity**: CRITICAL
- **File**: encoding.py (text normalization)
- **Issue**: Turkish 'İ' lowercases to 'i̇' not 'i', breaking pattern matching
- **Fix**: Add Unicode normalization (NFKC) before `.lower()`
- **Operator**: ΔE Exception-Quarantine

### Finding 6: No Metadata Type Validation
- **Severity**: CRITICAL
- **File**: encoding.py (metadata access)
- **Issue**: Assumes metadata["message_count"] is int, crashes if None
- **Fix**: Validate metadata types or use safe accessors
- **Operator**: ΔE Exception-Quarantine

## High Priority Issues (8)

### Finding 7-14: Pattern matching and confidence issues
- Pattern "hello" matches "othello" without word boundaries
- Pattern "so " matches "also" substring
- Third-person pattern sets double confidence denominator
- Connective count recomputed unnecessarily in write policy
- No validation that category is in VALID_CATEGORIES
- _build_reason minimally informative
- No logging/telemetry for monitoring

## Moderate Issues (5)

### Finding 15-19: Code quality
- PRIORITY_ORDER hardcoded, not derived from CATEGORY_IMPORTANCE
- EncodingConfig lacks __post_init__ validation
- Missing docstring for SINGLE_PATTERN_CONFIDENCE_FLOOR
- Episode length offset not documented in code
- _reclassify_without duplicates logic from classify
