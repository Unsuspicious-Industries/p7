"""Γ — the typing context, and the thing gamma is named after.

ARCHITECTURE I1: the client owns Γ; the server borrows its *type projection*
for the duration of one request and never holds it between requests.

Γ therefore exists at two layers, and conflating them is the failure mode this
module exists to prevent:

    client-side (here)   name -> (type source, runtime value)
    wire  (`.types()`)   name -> type source

`types()` is the only thing that crosses to provider7.  Values never leave the
client, because the client is what executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from .scheme import TypeSource


@dataclass(frozen=True)
class Binding:
    """One name in Γ: its type, its value, and where it came from."""

    name: str
    type: TypeSource
    value: Any
    #: Turn index that created this binding, for display and debugging.
    turn: int = 0
    #: Compact human-readable summary. Large values (file contents, search
    #: results) are summarised for the prompt while the full value stays here.
    summary: str = ""

    def render(self) -> str:
        detail = f"  // {self.summary}" if self.summary else ""
        return f"{self.name} : {self.type}{detail}"


@dataclass
class Gamma:
    """The client-side typing context.

    Append-mostly: bindings accumulate across turns and a rebind shadows the
    previous entry under the same name, exactly as a REPL namespace does.
    """

    bindings: dict[str, Binding] = field(default_factory=dict)

    # ── Client-side view (types + values) ───────────────────────────────

    def bind(
        self,
        name: str,
        type: TypeSource,
        value: Any,
        *,
        turn: int = 0,
        summary: str = "",
    ) -> None:
        self.bindings[name] = Binding(
            name=name, type=type, value=value, turn=turn, summary=summary
        )

    def value(self, name: str) -> Any:
        binding = self.bindings.get(name)
        if binding is None:
            raise KeyError(f"unbound name {name!r}")
        return binding.value

    def type_of(self, name: str) -> TypeSource:
        binding = self.bindings.get(name)
        if binding is None:
            raise KeyError(f"unbound name {name!r}")
        return binding.type

    def __contains__(self, name: object) -> bool:
        return name in self.bindings

    def __len__(self) -> int:
        return len(self.bindings)

    def __iter__(self) -> Iterator[Binding]:
        return iter(self.bindings.values())

    # ── Wire view (types only) ──────────────────────────────────────────

    def types(self) -> dict[str, TypeSource]:
        """The `aufbau_context` sent to provider7. Types only, never values."""
        return {name: b.type for name, b in sorted(self.bindings.items())}

    def render(self) -> str:
        """Compact display of Γ for the prompt.

        Goes *last* in the prompt, immediately before generation. Γ changes
        every turn, so rendering it earlier invalidates the whole prefix cache
        on every request — see ARCHITECTURE §3.1.
        """
        if not self.bindings:
            return "(empty)"
        return "\n".join(b.render() for b in sorted(self.bindings.values(), key=lambda b: b.name))

    # ── Commit ──────────────────────────────────────────────────────────

    def commit(self, updates: Mapping[str, tuple[TypeSource, Any]], turn: int) -> None:
        """Apply a turn's new bindings.

        Unconditional by design. Because the language has no exceptional
        control flow (ARCHITECTURE I1a), a primitive that fails still returns a
        value — an `Err` variant — so every name the decode promised exists
        with exactly the type the decode said it had. There is no partial
        commit and no rollback path.
        """
        for name, (type_source, value) in updates.items():
            self.bind(name, type_source, value, turn=turn)

    def clone(self) -> "Gamma":
        return Gamma(bindings=dict(self.bindings))
