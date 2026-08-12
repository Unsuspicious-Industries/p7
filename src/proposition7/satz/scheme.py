"""Primitive declarations. The single source of truth (ARCHITECTURE I2).

Two projections read this and nothing else:
    grammar.compose()   -> .auf the mask enforces
    evaluator.Dispatch  -> host calls that run

Types are source text in the active grammar's own `Type*` language. Nothing
here parses them or branches on their constructors (ARCHITECTURE I4).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

TypeSource = str


@dataclass(frozen=True)
class Param:
    name: str
    type: TypeSource


@dataclass(frozen=True)
class Effect:
    """`kind` is read/write/execute/network/delete. `scope` narrows it for
    approval. Both opaque here; gamma's policy interprets them."""

    kind: str
    scope: str = "*"

    def __str__(self) -> str:
        return f"{self.kind}({self.scope})"


@dataclass(frozen=True)
class Primitive:
    """A typed foreign function callable from the agent language.

    `raises` set makes the return type a Result wrapper in both projections, so
    a fallible primitive cannot return a bare value and no exceptional control
    flow enters the language through this table (ARCHITECTURE I1a).
    """

    name: str
    params: tuple[Param, ...] = ()
    returns: TypeSource = ""
    raises: TypeSource | None = None
    effects: tuple[Effect, ...] = ()

    @property
    def fallible(self) -> bool:
        return self.raises is not None

    def signature(self) -> str:
        args = ", ".join(f"{p.name}: {p.type}" for p in self.params)
        ret = f"{self.returns}!{self.raises}" if self.fallible else self.returns
        eff = "".join(f"  [{e}]" for e in self.effects)
        return f"{self.name}({args}) -> {ret}{eff}"


@dataclass(frozen=True)
class Scheme:
    """The primitives granted for one request.

    The scheme is the policy (ARCHITECTURE I7): a primitive absent here is
    absent from the grammar, so the mask cannot emit a call to it. Grant and
    constraint are the same object.
    """

    primitives: tuple[Primitive, ...] = ()

    def __post_init__(self) -> None:
        names = [p.name for p in self.primitives]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate primitives: {sorted(dupes)}")

    def ordered(self) -> tuple[Primitive, ...]:
        """Name-sorted. Every projection iterates through this: the SPG cache is
        keyed by content hash, so an unstable order recompiles the grammar on
        every request (ARCHITECTURE §3.1)."""
        return tuple(sorted(self.primitives, key=lambda p: p.name))

    def get(self, name: str) -> Primitive | None:
        return next((p for p in self.primitives if p.name == name), None)

    def restrict(self, names: Iterable[str]) -> "Scheme":
        keep = set(names)
        return replace(self, primitives=tuple(p for p in self.primitives if p.name in keep))

    def without_effects(self, kinds: Iterable[str]) -> "Scheme":
        """Drop every primitive carrying any of `kinds`. A read-only agent is
        `without_effects({"write", "delete", "execute"})`."""
        banned = set(kinds)
        return replace(
            self,
            primitives=tuple(
                p for p in self.primitives if not any(e.kind in banned for e in p.effects)
            ),
        )

    def effects(self) -> tuple[Effect, ...]:
        seen: dict[tuple[str, str], Effect] = {}
        for p in self.ordered():
            for e in p.effects:
                seen.setdefault((e.kind, e.scope), e)
        return tuple(seen.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitives": [
                {
                    "name": p.name,
                    "params": [{"name": q.name, "type": q.type} for q in p.params],
                    "returns": p.returns,
                    "raises": p.raises,
                    "effects": [{"kind": e.kind, "scope": e.scope} for e in p.effects],
                }
                for p in self.ordered()
            ],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Scheme":
        return Scheme(
            primitives=tuple(
                Primitive(
                    name=p["name"],
                    params=tuple(Param(q["name"], q["type"]) for q in p.get("params", ())),
                    returns=p["returns"],
                    raises=p.get("raises"),
                    effects=tuple(
                        Effect(e["kind"], e.get("scope", "*")) for e in p.get("effects", ())
                    ),
                )
                for p in data.get("primitives", ())
            ),
        )
