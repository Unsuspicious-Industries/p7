"""The primitive scheme — `provider7.constraints/v1`.

This module is the single source of truth for what a primitive *is*
(ARCHITECTURE I2).  Everything downstream is a projection of a `Scheme`:

    Scheme ──> .auf grammar fragment   (proposition7.satz.grammar)   what the mask enforces
           └─> evaluator dispatch      (proposition7.satz.evaluator) what actually runs
           └─> effect audit            (proposition7.satz.evaluator)   what the user approves

There is deliberately no second place to declare a primitive.  A signature
that appears in the grammar but not the dispatch table is a build error, not
a runtime surprise.

The scheme is *data*.  It serialises to JSON and crosses the wire to
provider7 as part of a request, which is what keeps provider7 ignorant of the
agent language (ARCHITECTURE §5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

SCHEME_VERSION = "provider7.constraints/v1"


# ── Types ────────────────────────────────────────────────────────────────
#
# A type is *source text in the active grammar's own `Type*` language*
# (ARCHITECTURE I4).  it never parses it, never branches on its
# constructors, and never maintains a table of known type names.  Aufbau
# parses it; the grammar's rules give it meaning.
#
# So `TypeSource` is a `str`, and that is not laziness — it is the invariant.

TypeSource = str


@dataclass(frozen=True)
class Param:
    """One typed parameter of a primitive."""

    name: str
    type: TypeSource
    description: str = ""


@dataclass(frozen=True)
class Effect:
    """A side effect a primitive may perform.

    `kind` is a coarse class (`read`, `write`, `execute`, `network`, `delete`).
    `scope` narrows it for display and approval — a path glob, a domain, a
    command class.  Both are opaque here; gamma's policy interprets them.
    """

    kind: str
    scope: str = "*"

    def __str__(self) -> str:
        return f"{self.kind}({self.scope})"


@dataclass(frozen=True)
class Primitive:
    """A typed foreign function callable from the agent language.

    Not a "tool" in the tool-call sense: there is no separate call protocol
    and no JSON envelope.  A primitive is an ordinary function in the
    language, and the mask enforces its signature at decode time.

    Fallibility is structural.  A primitive that can fail declares `raises`,
    and its return type is then wrapped as a `Result` by the language binding
    (ARCHITECTURE I1a).  There is no way to declare a fallible primitive that
    returns a bare value, which is the point: no exceptional control flow can
    enter the language through the primitive table.
    """

    name: str
    params: tuple[Param, ...]
    returns: TypeSource
    raises: TypeSource | None = None
    effects: tuple[Effect, ...] = ()
    description: str = ""

    @property
    def fallible(self) -> bool:
        return self.raises is not None

    @property
    def pure(self) -> bool:
        return not self.effects

    def signature(self) -> str:
        """Human-readable signature, for prompts and approval UI."""
        args = ", ".join(f"{p.name}: {p.type}" for p in self.params)
        ret = self.returns if not self.fallible else f"{self.returns}!{self.raises}"
        suffix = f"  [{', '.join(str(e) for e in self.effects)}]" if self.effects else ""
        return f"{self.name}({args}) -> {ret}{suffix}"


@dataclass(frozen=True)
class Scheme:
    """The full set of primitives granted for one request.

    The scheme *is* the policy (ARCHITECTURE I7).  Granting a capability means
    including the primitive that carries it; there is no separate permission
    list, and provider7 enforces nothing.  A primitive absent from the scheme
    is absent from the grammar, so the model cannot emit a call to it —
    the grant and the constraint are the same object.
    """

    primitives: tuple[Primitive, ...] = ()
    version: str = SCHEME_VERSION

    def __post_init__(self) -> None:
        names = [p.name for p in self.primitives]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate primitive names: {sorted(duplicates)}")

    # ── Ordering ────────────────────────────────────────────────────────
    #
    # Byte-determinism of the composed grammar is a caching requirement, not
    # a style preference: the SPG compile cache is keyed by content hash, so
    # an order-unstable scheme recompiles the grammar on every single request
    # (ARCHITECTURE §3.1).  Every projection iterates via `ordered()`.

    def ordered(self) -> tuple[Primitive, ...]:
        """Primitives in a stable, name-sorted order."""
        return tuple(sorted(self.primitives, key=lambda p: p.name))

    # ── Set operations ──────────────────────────────────────────────────

    def get(self, name: str) -> Primitive | None:
        return next((p for p in self.primitives if p.name == name), None)

    def restrict(self, names: Iterable[str]) -> "Scheme":
        """Narrow the scheme to `names`. This is how a capability is revoked."""
        keep = set(names)
        return replace(
            self, primitives=tuple(p for p in self.primitives if p.name in keep)
        )

    def without_effects(self, kinds: Iterable[str]) -> "Scheme":
        """Drop every primitive carrying any of `kinds`.

        The read-only agent is `scheme.without_effects({"write", "delete", "execute"})`
        — enforced by the mask, not by asking the model nicely.
        """
        banned = set(kinds)
        return replace(
            self,
            primitives=tuple(
                p
                for p in self.primitives
                if not any(e.kind in banned for e in p.effects)
            ),
        )

    def effects(self) -> tuple[Effect, ...]:
        """Every effect reachable through this scheme, deduplicated."""
        seen: dict[tuple[str, str], Effect] = {}
        for prim in self.ordered():
            for eff in prim.effects:
                seen.setdefault((eff.kind, eff.scope), eff)
        return tuple(seen.values())

    # ── Wire format ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "primitives": [
                {
                    "name": p.name,
                    "params": [
                        {"name": q.name, "type": q.type, "description": q.description}
                        for q in p.params
                    ],
                    "returns": p.returns,
                    "raises": p.raises,
                    "effects": [{"kind": e.kind, "scope": e.scope} for e in p.effects],
                    "description": p.description,
                }
                for p in self.ordered()
            ],
        }

    def to_json(self) -> str:
        # sort_keys for byte-determinism; see `ordered()`.
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Scheme":
        version = data.get("version", SCHEME_VERSION)
        if version != SCHEME_VERSION:
            raise ValueError(
                f"unsupported scheme version {version!r}, expected {SCHEME_VERSION!r}"
            )
        return Scheme(
            version=version,
            primitives=tuple(
                Primitive(
                    name=p["name"],
                    params=tuple(
                        Param(
                            name=q["name"],
                            type=q["type"],
                            description=q.get("description", ""),
                        )
                        for q in p.get("params", ())
                    ),
                    returns=p["returns"],
                    raises=p.get("raises"),
                    effects=tuple(
                        Effect(kind=e["kind"], scope=e.get("scope", "*"))
                        for e in p.get("effects", ())
                    ),
                    description=p.get("description", ""),
                )
                for p in data.get("primitives", ())
            ),
        )

    @staticmethod
    def from_json(text: str) -> "Scheme":
        return Scheme.from_dict(json.loads(text))
