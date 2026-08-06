"""`Result` — the expected-failure channel.

ARCHITECTURE I1a: the agent language has no exceptional control flow.  A
primitive that can fail returns `Result[T, E]`; it does not raise.  That is
what makes `updated_aufbau_context` authoritative — a failing `glob` still
leaves `x` bound, at exactly the type the decode promised, holding an `Err`
instead of an `Ok`.

The payoff is not only soundness.  Because the variants are distinct in the
type system, the model cannot use `x` as a `PathSet` without matching on it
first, and the mask enforces that at decode time.  Error handling stops being
something the harness reminds the model to do and becomes something the
grammar will not let it skip.

Do not confuse this with `ir.Todo`, which is the *model*-failure channel:
`Result` is a value that flows onward, `Todo` aborts the turn before anything
runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T

    @property
    def is_ok(self) -> bool:
        return True

    def unwrap(self) -> T:
        return self.value

    def map(self, fn: Callable[[T], Any]) -> "Result":
        return Ok(fn(self.value))

    def __str__(self) -> str:
        return f"Ok({self.value!r})"


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E

    @property
    def is_ok(self) -> bool:
        return False

    def unwrap(self) -> Any:
        raise ValueError(f"unwrapped an Err: {self.error!r}")

    def map(self, fn: Callable[[Any], Any]) -> "Result":
        return self

    def __str__(self) -> str:
        return f"Err({self.error!r})"


Result = Union[Ok, Err]


class PrimitiveFailure(Exception):
    """Raised by a host implementation to signal expected failure.

    Host functions are ordinary Python and Python raises; the evaluator catches
    this and converts it to `Err` at the language boundary.  This keeps the
    no-throw invariant a property of the *language* without forcing every host
    implementation to be written in a Result-returning style.

    Anything *other* than this escaping a host function is a bug in the host,
    not an expected failure, and the evaluator lets it propagate.
    """

    def __init__(self, error: Any):
        super().__init__(str(error))
        self.error = error
