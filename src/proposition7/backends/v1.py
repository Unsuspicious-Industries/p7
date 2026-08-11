"""`proposition7.backend/v1` — the cross-repository constrained-decode contract.

This is the whole of p7's public surface (see `PLAN.md` §4 and
`../../../ARCHITECTURE.md` §3). Two operations over **raw `.auf` source**:

    generate(grammar_source, prompt, ...) -> GenerationResult
    verify(grammar_source, text, ...)     -> aufbau.Verification

What this contract deliberately excludes, because p7 does not own it: tools,
registries, type-name tables, sessions, HTTP, policy, execution. A grammar
arrives as source text and leaves as a compiled artifact p7 caches internally
by content hash; no grammar handle ever crosses the boundary.

The context mapping is name -> *type source in the grammar's own `Type*`
language* (ARCHITECTURE I4). p7 treats every one of those strings as opaque:
it does not parse them, does not branch on constructors, and maintains no table
of known type names. Meaning is the grammar's business.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

#: Names the whole contract. A backend advertising this identifier guarantees
#: semantic mask/feed, atomic typed context, strict goal verification, and
#: request-local state.
BACKEND_API = "proposition7.backend/v1"

if TYPE_CHECKING:  # pragma: no cover
    from aufbau import Verification

    from ..llm import GenerationResult


@runtime_checkable
class ConstraintBackend(Protocol):
    """What a consumer needs from a constrained-decoding backend.

    `ConstrainedModel` implements this directly; there is no adapter class.
    The Protocol exists for injection, fakes, and static checks.
    """

    def generate(
        self,
        grammar_source: str,
        prompt: str,
        *,
        initial: str = "",
        context: Mapping[str, str] | None = None,
        max_tokens: int = 64,
        temperature: float = 0.0,
        seed: int | None = None,
        expected_type: str | None = None,
    ) -> "GenerationResult":
        """Decode under `grammar_source`, masked against `context`.

        `context` is applied atomically before the first mask, so the very
        first token is already constrained by every binding. Every candidate
        the mask accepts is fed transactionally: a failed feed must never
        corrupt subsequent generation.

        `is_complete` on the result is true only when a fresh final
        verification is typed, unambiguous, and satisfies `expected_type` when
        one was supplied — never merely because decoding stopped.
        """
        ...

    def verify(
        self,
        grammar_source: str,
        text: str,
        *,
        context: Mapping[str, str] | None = None,
        expected_type: str | None = None,
    ) -> "Verification":
        """Check `text` against `grammar_source`, independently of generation.

        Returns aufbau's own `Verification` — p7 does not define a parallel
        result type.
        """
        ...


__all__ = ["BACKEND_API", "ConstraintBackend"]
