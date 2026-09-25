"""The shared value types of the congen tree.

These live here, in p7, because p7 is the one package every other Python
component depends on: wirt and gamma both import it, bench measures through it,
and aufbau is Rust. A type that has to mean the same thing in all of them has
nowhere else to go.

`Result` is the expected-failure channel (ARCHITECTURE I1a). A fallible
primitive returns it; it does not raise. That is what lets a failing call still
bind its name at the type the decode promised, holding `Err` instead of `Ok` --
the model cannot reach the payload without matching, and the mask enforces the
match at decode time. Failure being a value is therefore structural here, not a
coding convention.

Not to be confused with an exception: `Result` is a value that flows on, an
exception aborts. `proposition7.errors` holds the things that abort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Ok", "Err", "Result", "is_ok", "is_err"]


@dataclass(frozen=True)
class Ok:
    """A successful outcome carrying its value."""

    value: Any

    def __str__(self) -> str:
        return f"Ok({self.value!r})"


@dataclass(frozen=True)
class Err:
    """An expected failure carrying its error. Not an exception."""

    error: Any

    def __str__(self) -> str:
        return f"Err({self.error!r})"


#: The sum. `raises` on a primitive declaration forces a return type of
#: `Result[T, E]` in both projections -- the grammar and the dispatch -- so the
#: two cannot drift.
Result = Ok | Err


def is_ok(result: Result) -> bool:
    return isinstance(result, Ok)


def is_err(result: Result) -> bool:
    return isinstance(result, Err)
