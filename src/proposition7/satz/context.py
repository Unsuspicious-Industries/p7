"""Γ. The client owns it; the server borrows its types for one request.

    client (here)      name -> (type source, value)
    wire `.types()`    name -> type source

Values never cross: the client is what executes (ARCHITECTURE I1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .scheme import TypeSource


@dataclass
class Gamma:
    bindings: dict[str, tuple[TypeSource, Any]] = field(default_factory=dict)

    def bind(self, name: str, type: TypeSource, value: Any) -> None:
        self.bindings[name] = (type, value)

    def value(self, name: str) -> Any:
        return self.bindings[name][1]

    def type_of(self, name: str) -> TypeSource:
        return self.bindings[name][0]

    def __contains__(self, name: object) -> bool:
        return name in self.bindings

    def __len__(self) -> int:
        return len(self.bindings)

    def types(self) -> dict[str, TypeSource]:
        """The `aufbau_context` sent to provider7."""
        return {n: t for n, (t, _) in sorted(self.bindings.items())}

    def render(self) -> str:
        """For the prompt. Goes last: Γ changes every turn, and anything before
        it stays in the cacheable prefix (ARCHITECTURE §3.1)."""
        if not self.bindings:
            return "(empty)"
        return "\n".join(f"{n} : {t}" for n, (t, _) in sorted(self.bindings.items()))

    def commit(self, updates: Mapping[str, tuple[TypeSource, Any]]) -> None:
        """Unconditional: with no exceptional control flow every promised name
        exists at its promised type (ARCHITECTURE I1a)."""
        self.bindings.update(updates)
