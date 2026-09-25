"""The shared exception root of the congen tree.

Every exception any congen component raises on purpose derives from
`CongenError`, so a caller that wants to catch "anything this system raised
deliberately" has one name to catch, and anything else escaping is a bug rather
than a condition.

The root lives in p7 because p7 is the common dependency: wirt, gamma and bench
all import it, and aufbau is Rust. There is nowhere else a name shared by all
of them can live.

Only the root lives here. The leaves live where they are raised -- wirt's
`ConfigError` in `wirt.config`, gamma's `EvaluationError` in its evaluator,
p7's `CompositionError` in `scheme.compose` -- because an error class belongs
next to the code that can raise it, not in a catalogue far from it.

Each of those keeps the builtin base it already had *as well as* this one:

    class ConfigError(CongenError, ValueError): ...

so `except ValueError` keeps working for every caller that already relied on it
while `except CongenError` starts working for callers that want the whole tree.
Widening the bases is backward compatible; narrowing them would not be.

For expected failure that flows on as a value rather than aborting, see
`proposition7.types.Result`.
"""

from __future__ import annotations

__all__ = ["CongenError"]


class CongenError(Exception):
    """Root of every deliberate exception in the congen tree."""
