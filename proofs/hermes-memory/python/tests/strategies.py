"""Shared Hypothesis strategies for Hermes memory property tests.

Each strategy maps to a domain constraint from the Lean formalization.
"""

from hypothesis import strategies as st


def pos_real():
    """Positive reals: (0, 100]."""
    return st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)


def unit_interval():
    """[0, 1] with no subnormals."""
    return st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


def open_unit():
    """(0, 1) -- strictly interior."""
    return st.floats(
        min_value=0.01,
        max_value=0.99,
        allow_nan=False,
        allow_infinity=False,
    )


def alpha_st():
    """Learning rate alpha in (0, 1)."""
    return st.floats(
        min_value=0.01,
        max_value=0.99,
        allow_nan=False,
        allow_infinity=False,
    )


def pos_time():
    """Non-negative time: [0, 100]."""
    return st.floats(
        min_value=0.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    )


def strength_st():
    """Strength values: [0.1, 10]."""
    return st.floats(
        min_value=0.1,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    )


def smax_st():
    """Maximum strength: [1, 100]."""
    return st.floats(
        min_value=1.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
    )


def temperature_st():
    """Temperature: (0, 10]."""
    return st.floats(
        min_value=0.01,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    )
