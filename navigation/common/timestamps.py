"""All timestamps use nonnegative simulation seconds, never wall-clock time."""

import math


def validate_timestamp(t):
    t = float(t)
    if not math.isfinite(t) or t < 0:
        raise ValueError("timestamp must be finite and nonnegative")
    return t


def require_monotonic(previous, current, allow_equal=False):
    current = validate_timestamp(current)
    if previous is not None and (
        current < previous or (current == previous and not allow_equal)
    ):
        raise ValueError("timestamps must increase")
    return current
