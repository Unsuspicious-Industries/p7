"""The expected-failure channel (ARCHITECTURE I1a).

A fallible primitive returns Result; it does not raise. That is what makes Γ′
authoritative — a failing call still binds its name at the type the decode
promised, holding Err instead of Ok. The model cannot reach the payload without
matching, and the mask enforces that at decode time.

Not to be confused with `todo`, the model-failure channel: Result is a value
that flows on, `todo` aborts the turn before anything runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Ok:
    value: Any

    def __str__(self) -> str:
        return f"Ok({self.value!r})"


@dataclass(frozen=True)
class Err:
    error: Any

    def __str__(self) -> str:
        return f"Err({self.error!r})"


Result = Ok | Err


class PrimitiveFailure(Exception):
    """Raised by a host to signal expected failure; the evaluator converts it to
    Err at the language boundary. Anything else escaping a host is a host bug
    and propagates."""

    def __init__(self, error: Any):
        super().__init__(str(error))
        self.error = error
