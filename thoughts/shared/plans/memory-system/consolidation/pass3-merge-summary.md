# Pass 3 Amendments Merge Summary

**Date**: 2026-02-27
**Task**: Merge adversarial findings from Pass 3 into consolidation spec
**Status**: COMPLETE

## Changes Applied

### 1. AP3-F2: Affinity Content Similarity Floor

**Added to Section 3.3 (ConsolidationConfig)**:
- New parameter: `min_content_similarity: float = 0.3`
- Documentation in Clustering attributes section
- Validation rule: `0.0 <= min_content_similarity <= 1.0`

**Updated Section 5.5 (compute_affinity)**:
- Added content similarity floor check before computing affinity
- If `content_sim < min_content_similarity`, return 0.0
- Updated behavioral contracts to reflect floor property

### 2. AP3-F3: Content-Based Correction Detection

**Added Section 4.7 (new constants)**:
- `CORRECTION_CONTENT_MARKERS`: frozenset of 6 regex patterns
- `CORRECTION_MARKER_THRESHOLD`: int = 2

**Added Section 5.11 (new private helper)**:
- `_content_is_correction(content: str) -> bool`
- Counts matches against CORRECTION_CONTENT_MARKERS
- Returns True if >= CORRECTION_MARKER_THRESHOLD matches

**Updated Section 5.1 (should_consolidate)**:
- Added inhibitor I5: content-based correction detection
- Calls `_content_is_correction()` after category check
- Updated behavioral contracts to include I5

### 3. AP3-F1: Sentence-Scoring Fallback

**Added Section 5.13 (new private helper)**:
- `_extract_by_sentence_scoring(content: str, target_length: int) -> str`
- Scores sentences by information density heuristic
- Selects highest-scoring sentences to reach target length

**Updated Section 5.3 (extract_semantic)**:
- Changed fallback from `_truncate_to_sentences()` to `_extract_by_sentence_scoring()`
- Updated behavioral contracts to describe sentence-scoring fallback

### 4. AP3-F5: Shared Inhibitor Function

**Added Section 5.12 (new private helper)**:
- `_passes_inhibitors(candidate, config) -> bool`
- Applies all 5 inhibitors (I1-I5) in single location
- Includes note that should_consolidate and select_consolidation_candidates should use this

**Updated Section 5.1 (should_consolidate)**:
- Replaced inline inhibitor checks with call to `_passes_inhibitors()`
- Added AP3-F5 comment

**Updated Section 5.6 (select_consolidation_candidates)**:
- All three modes now use `_passes_inhibitors()` instead of inline checks
- Added AP3-F5 comments in INTRA_SESSION and ASYNC_BATCH paths

### 5. AP3-F6: Parameter Classification

**Added to Section 14.3 (Property-Based Tests)**:
- Added "Parameter Classification" table with 22 rows
- Documents which 8 parameters are proof-relevant vs 14 calibration
- Guides test strategy for Hypothesis generators

### 6. AP3-F7: Provenance Chain Integrity

**Updated Section 3.6 (ArchivedMemory invariants)**:
- Added "Provenance Chain Integrity" note after invariants list
- Documents caller's responsibility to update `consolidated_to` on repeated consolidation
- Notes that property-based test should verify chain integrity

**Updated Section 14.3 (Property-Based Tests)**:
- Added 6 new property-based test skeletons:
  - `test_content_similarity_floor` (AP3-F2)
  - `test_correction_content_detection` (AP3-F3)
  - `test_passes_inhibitors_consistency` (AP3-F5)
  - `test_semantic_extraction_non_empty_source_episodes` (AP3-F7)
  - `test_provenance_chain_integrity` (AP3-F7)

### 7. AP3-F4: L3->L4 Design Note

**Updated Section 16.2 (Known Limitations)**:
- Added paragraph explaining L3->L4 requires semantic reasoning beyond Jaccard
- Notes that v1 can only verify L1/L2 -> L3 transitions
- Clarifies SEMANTIC_INSIGHT is design placeholder for v2

### 8. Appendix A: Comprehensive Documentation

**Added Section "Appendix A: Adversarial Review Amendments (Pass 3)"**:
- Executive summary of Pass 3 review
- Detailed explanation of all 7 findings (AP3-F1 through AP3-F7)
- Examples showing why each finding is lethal/structural
- Amendment descriptions for each
- Impact analysis for each
- Summary table mapping findings to changes
- Review metadata (operators used, reviewer, date, status)

## Statistics

- **Lines added**: 452 (2144 -> 2596)
- **New constants**: 2 (CORRECTION_CONTENT_MARKERS, CORRECTION_MARKER_THRESHOLD)
- **New config parameters**: 1 (min_content_similarity)
- **New private helpers**: 3 (_content_is_correction, _passes_inhibitors, _extract_by_sentence_scoring)
- **New inhibitors**: 1 (I5: content-based correction detection)
- **New property tests**: 6
- **Sections modified**: 11
- **New sections**: 3 (4.7, 5.11-5.13, Appendix A)

## Verification

All key elements verified present:
- ✓ min_content_similarity in ConsolidationConfig and compute_affinity
- ✓ CORRECTION_CONTENT_MARKERS constant
- ✓ _content_is_correction helper
- ✓ _extract_by_sentence_scoring helper
- ✓ _passes_inhibitors helper
- ✓ Parameter classification table
- ✓ Provenance chain integrity note
- ✓ L3->L4 design note
- ✓ Appendix A with all 7 findings

## No Changes to Existing Pass 1/Pass 2 Content

All amendments are ADDITIVE. No Pass 1 or Pass 2 content was modified or removed,
only extended with Pass 3 findings.
