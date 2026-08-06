"""The execution enclave — the only place in p7 that runs anything.

Everything else in `proposition7` and `aufbau` is pure checking: aufbau decides
typed/live/dead, p7 masks logits. Neither opens a file, spawns a process, or
touches a network. This module executes. "p7 runs code" is false everywhere
except here.

Isolation:
  - ships no capabilities: `Dispatch.hosts` is caller-supplied, and there is no
    open/subprocess/socket anywhere in `proposition7.satz`
  - cannot acquire one: a primitive absent from the scheme is absent from both
    the grammar and the dispatch table (ARCHITECTURE I7)
  - blast radius is enumerable before execution, via `Evaluator.audit`
  - provider7 must never import it; that is what keeps the inference host
    non-executing (ARCHITECTURE I3)

Structure comes from the FFI, never from a second model: `node.nt_name()`,
`node.children`, `ast.type_of(evidence)`, `ast.input[start:end]`. No type is
re-derived here. Dispatch keys close the loop — `grammar.nonterminal()`
generates a primitive's node name and `Dispatch.by_node()` inverts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .context import Gamma
from .grammar import LanguageBinding, nonterminal
from .result import Err, Ok, PrimitiveFailure
from .scheme import Effect, Primitive, Scheme, TypeSource

Host = Callable[..., Any]


class DispatchError(ValueError):
    pass


class EvaluationError(RuntimeError):
    """A host bug or a gap between the language binding and the grammar. The
    mask should have made every other failure mode unreachable."""


@dataclass(frozen=True)
class Dispatch:
    """Scheme plus host implementations, checked to correspond exactly."""

    scheme: Scheme
    hosts: dict[str, Host]

    def __post_init__(self) -> None:
        declared = {p.name for p in self.scheme.primitives}
        missing = declared - set(self.hosts)
        if missing:
            raise DispatchError(f"no implementation for {sorted(missing)}")
        extra = set(self.hosts) - declared
        if extra:
            raise DispatchError(f"no scheme entry for {sorted(extra)}; unreachable")

    def by_node(self) -> dict[str, Primitive]:
        return {nonterminal(p.name): p for p in self.scheme.ordered()}


@dataclass(frozen=True)
class Audit:
    """What a turn could do, known before it does any of it (ARCHITECTURE §2.1).

    With branches this over-approximates: a call in an untaken arm still
    appears. That is the right bias for approval."""

    effects: tuple[Effect, ...] = ()
    primitives: tuple[str, ...] = ()

    def kinds(self) -> frozenset[str]:
        return frozenset(e.kind for e in self.effects)

    def render(self) -> str:
        return "\n".join(f"  {e}" for e in self.effects) or "no effects"


@dataclass
class Outcome:
    updates: dict[str, tuple[TypeSource, Any]] = field(default_factory=dict)
    aborted: bool = False
    #: "todo", "denied", or "incomplete".
    reason: str = ""
    audit: Audit | None = None

    @property
    def ok(self) -> bool:
        return not self.aborted


class Evaluator:
    """Runs a decoded turn against client-side Γ. Atomic: all or nothing.

    `spg` must be the grammar the decode was masked under. Re-parsing the
    completion with it is what makes every type in the tree the engine's answer.
    """

    def __init__(
        self,
        dispatch: Dispatch,
        binding: LanguageBinding,
        spg: Any,
        *,
        approve: Callable[[Audit], bool] | None = None,
    ):
        self.dispatch = dispatch
        self.binding = binding
        self.spg = spg
        self.approve = approve or (lambda _: True)
        self.by_node = dispatch.by_node()

    # tree access

    @staticmethod
    def _kids(node: Any) -> list[Any]:
        return [c.node for c in node.children if c.node is not None]

    def _text(self, node: Any) -> str:
        """Source text of a node. `node.start`/`end` index the *token* stream,
        not characters, and `node.text` is not populated — so slice the tokens
        the engine produced."""
        return "".join(seg.text for seg in self._tokens[node.start : node.end]).strip()

    def _type(self, ast: Any, node: Any) -> TypeSource:
        term = ast.type_of(node.evidence)
        if term is None:
            return ""
        # type_of hands back rendered text or a Term depending on the evidence.
        # This picks a spelling; it never derives a type.
        return term if isinstance(term, str) else self.spg.show(term)

    def _walk(self, node: Any):
        yield node
        for kid in self._kids(node):
            yield from self._walk(kid)

    def _nodes(self, ast: Any):
        for root in ast.roots:
            yield from self._walk(root)

    def parse(self, completion: str) -> Any:
        import aufbau

        self._tokens = list(self.spg.tokenize(completion))
        return aufbau.Synthesizer.from_grammar(self.spg, completion).ast()

    def audit(self, ast: Any) -> Audit:
        effects: dict[tuple[str, str], Effect] = {}
        names: list[str] = []
        for node in self._nodes(ast):
            prim = self.by_node.get(node.nt_name())
            if prim is None:
                continue
            if prim.name not in names:
                names.append(prim.name)
            for e in prim.effects:
                effects.setdefault((e.kind, e.scope), e)
        return Audit(tuple(effects.values()), tuple(names))

    def run(self, completion: str, gamma: Gamma) -> Outcome:
        ast = self.parse(completion)
        if not ast.is_complete():
            return Outcome(aborted=True, reason="incomplete")

        audit = self.audit(ast)

        # Model-failure channel: nothing runs, Γ is untouched, the model retries.
        if any(n.nt_name() == self.binding.todo_nt for n in self._nodes(ast)):
            return Outcome(aborted=True, reason="todo", audit=audit)

        if not self.approve(audit):
            return Outcome(aborted=True, reason="denied", audit=audit)

        local: dict[str, tuple[TypeSource, Any]] = {}
        b = self.binding

        def value_of(name: str) -> Any:
            if name in local:
                return local[name][1]
            if name in gamma:
                return gamma.value(name)
            raise EvaluationError(f"unbound name {name!r}; the mask should have caught this")

        def evaluate(node: Any) -> Any:
            name = node.nt_name()

            prim = self.by_node.get(name)
            if prim is not None:
                return self._call(prim, [evaluate(k) for k in self._kids(node)])

            # A single-alternative production is transparent in aufbau's tree,
            # so Expression -> Variable -> Identifier collapses and a bare
            # Identifier is what a variable reference looks like.
            if name in (b.variable_nt, b.identifier_nt):
                return value_of(self._text(node))

            if name in b.literal_nts:
                return self._text(node)

            kids = self._kids(node)
            if len(kids) == 1:
                return evaluate(kids[0])
            raise EvaluationError(f"no rule for {name!r} with {len(kids)} children")

        for stmt in self._statements(ast):
            kids = self._kids(stmt)
            if len(kids) >= 2 and kids[0].nt_name() == b.identifier_nt:
                local[self._text(kids[0])] = (
                    self._type(ast, kids[-1]),
                    evaluate(kids[-1]),
                )
            elif kids:
                evaluate(kids[-1])

        return Outcome(updates=local, audit=audit)

    def _statements(self, ast: Any) -> list[Any]:
        """Flatten the right-recursive StatementList in source order."""
        # An in-progress parse leaves a trailing incomplete statement node;
        # it has no value and must not be evaluated.
        return [
            n
            for n in self._nodes(ast)
            if n.nt_name() == self.binding.statement_nt and n.is_complete()
        ]

    def _call(self, prim: Primitive, args: list[Any]) -> Any:
        if len(args) != len(prim.params):
            raise EvaluationError(
                f"{prim.name} takes {len(prim.params)}, got {len(args)}; "
                "the grammar fixes arity, so this is a binding mismatch"
            )
        host = self.dispatch.hosts[prim.name]
        if not prim.fallible:
            return host(*args)
        # Convert the host's exception to a value at the language boundary, so
        # no exceptional control flow enters the language (ARCHITECTURE I1a).
        try:
            return Ok(host(*args))
        except PrimitiveFailure as failure:
            return Err(failure.error)
