"""Generic primitive declarations for grammar composition."""

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
    kind: str
    scope: str = "*"

    def __str__(self) -> str:
        return f"{self.kind}({self.scope})"


@dataclass(frozen=True)
class Primitive:
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
    """The primitive declarations included in one grammar."""

    primitives: tuple[Primitive, ...] = ()

    def __post_init__(self) -> None:
        names = [p.name for p in self.primitives]
        dupes = {name for name in names if names.count(name) > 1}
        if dupes:
            raise ValueError(f"duplicate primitives: {sorted(dupes)}")

    def ordered(self) -> tuple[Primitive, ...]:
        return tuple(sorted(self.primitives, key=lambda p: p.name))

    def get(self, name: str) -> Primitive | None:
        return next((p for p in self.primitives if p.name == name), None)

    def restrict(self, names: Iterable[str]) -> "Scheme":
        keep = set(names)
        return replace(self, primitives=tuple(p for p in self.primitives if p.name in keep))

    def without_effects(self, kinds: Iterable[str]) -> "Scheme":
        banned = set(kinds)
        return replace(
            self,
            primitives=tuple(
                p for p in self.primitives if not any(e.kind in banned for e in p.effects)
            ),
        )

    def effects(self) -> tuple[Effect, ...]:
        seen: dict[tuple[str, str], Effect] = {}
        for primitive in self.ordered():
            for effect in primitive.effects:
                seen.setdefault((effect.kind, effect.scope), effect)
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
                    effects=tuple(Effect(e["kind"], e.get("scope", "*")) for e in p.get("effects", ())),
                )
                for p in data.get("primitives", ())
            ),
        )
