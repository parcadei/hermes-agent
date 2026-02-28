#!/usr/bin/env bash
# Hermes Memory System — Test Baseline Runner
# Runs each test module individually, captures counts + durations,
# then runs the full suite. Saves results to baseline.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${SCRIPT_DIR}/.venv/bin/python"
OUTFILE="${SCRIPT_DIR}/baseline.txt"

# Override pyproject.toml addopts so we get the "N passed" summary line
PYTEST_OPTS=(-o "addopts=" --tb=short --no-header -q)

# All test files — add new ones here
TEST_FILES=(
    tests/test_encoding.py
    tests/test_recall.py
    tests/test_contradiction.py
    tests/test_consolidation.py
    tests/test_consolidation_phase_c.py
    tests/test_optimizer.py
    tests/test_contraction.py
    tests/test_coupling.py
    tests/test_composed_system.py
    tests/test_engine.py
    tests/test_markov_chain.py
    tests/test_memory_dynamics.py
    tests/test_monte_carlo.py
    tests/test_novelty_bonus.py
    tests/test_sensitivity.py
    tests/test_soft_selection.py
    tests/test_spectral.py
    tests/test_strength_decay.py
)

TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_XFAILED=0
TOTAL_ERRORS=0
MODULE_RESULTS=()

header="Hermes Memory — Test Baseline"
timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
separator="$(printf '=%.0s' {1..60})"
dash_sep="$(printf -- '-%.0s' {1..60})"

echo "$separator"
echo "$header"
echo "$timestamp"
echo "$separator"
echo ""

for tf in "${TEST_FILES[@]}"; do
    module_name="$(basename "$tf" .py | sed 's/^test_//')"

    start_ts="$(python3 -c 'import time; print(time.time())')"
    output="$("$PYTHON" -m pytest "$tf" "${PYTEST_OPTS[@]}" 2>&1)" || true
    end_ts="$(python3 -c 'import time; print(time.time())')"
    duration="$(python3 -c "print(f'{$end_ts - $start_ts:.1f}s')")"

    # Parse the summary line (e.g. "100 passed in 0.75s")
    summary="$(echo "$output" | grep -E '[0-9]+ passed' | tail -1)" || summary=""

    passed=0; failed=0; xfailed=0; errors=0
    if [[ "$summary" =~ ([0-9]+)\ passed ]]; then passed="${BASH_REMATCH[1]}"; fi
    if [[ "$summary" =~ ([0-9]+)\ failed ]]; then failed="${BASH_REMATCH[1]}"; fi
    if [[ "$summary" =~ ([0-9]+)\ xfailed ]]; then xfailed="${BASH_REMATCH[1]}"; fi
    if [[ "$summary" =~ ([0-9]+)\ error ]]; then errors="${BASH_REMATCH[1]}"; fi

    TOTAL_PASSED=$((TOTAL_PASSED + passed))
    TOTAL_FAILED=$((TOTAL_FAILED + failed))
    TOTAL_XFAILED=$((TOTAL_XFAILED + xfailed))
    TOTAL_ERRORS=$((TOTAL_ERRORS + errors))

    # Status indicator
    status="OK"
    if [[ $failed -gt 0 || $errors -gt 0 ]]; then status="FAIL"; fi

    line="$(printf "  %-30s %4d passed  %s  [%s]" "$module_name" "$passed" "$status" "$duration")"
    if [[ $failed -gt 0 ]]; then line="$line  ($failed failed)"; fi
    if [[ $xfailed -gt 0 ]]; then line="$line  ($xfailed xfailed)"; fi
    if [[ $errors -gt 0 ]]; then line="$line  ($errors errors)"; fi

    echo "$line"
    MODULE_RESULTS+=("$line")
done

echo ""
echo "$dash_sep"

# Full suite run
echo "  Running full suite..."
full_start="$(python3 -c 'import time; print(time.time())')"
full_output="$("$PYTHON" -m pytest tests/ "${PYTEST_OPTS[@]}" 2>&1)" || true
full_end="$(python3 -c 'import time; print(time.time())')"
full_duration="$(python3 -c "print(f'{$full_end - $full_start:.1f}s')")"
full_summary="$(echo "$full_output" | grep -E '[0-9]+ passed' | tail -1)" || full_summary="(no summary)"

echo "  Full suite: $full_summary  [$full_duration]"
echo ""
echo "$separator"
echo "  TOTALS (per-module):  ${TOTAL_PASSED} passed, ${TOTAL_FAILED} failed, ${TOTAL_XFAILED} xfailed, ${TOTAL_ERRORS} errors"
echo "$separator"

# Write to file
{
    echo "$separator"
    echo "$header"
    echo "$timestamp"
    echo "$separator"
    echo ""
    for r in "${MODULE_RESULTS[@]}"; do echo "$r"; done
    echo ""
    echo "$dash_sep"
    echo "  Full suite: $full_summary  [$full_duration]"
    echo ""
    echo "$separator"
    echo "  TOTALS (per-module):  ${TOTAL_PASSED} passed, ${TOTAL_FAILED} failed, ${TOTAL_XFAILED} xfailed, ${TOTAL_ERRORS} errors"
    echo "$separator"
} > "$OUTFILE"

echo ""
echo "Saved to: $OUTFILE"
