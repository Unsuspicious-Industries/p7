"""The second projection of a `Scheme`: what actually runs.

`proposition7.satz.grammar` turns a scheme into the constraint; this module turns the same
scheme into dispatch.  Both read one table, so a primitive cannot exist in the
grammar without an implementation or vice versa — ARCHITECTURE I2.  `Dispatch`
checks that correspondence at construction time, so the failure is a startup
error rather than a mid-turn surprise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .context import Gamma
from .ir import Bind, Call, Do, Expr, Lit, Program, Todo, Var
from .result import Err, Ok, PrimitiveFailure
from .scheme import Effect, Primitive, Scheme, TypeSource

#: A host implementation. Ordinary Python, called positionally. It may raise
#: `PrimitiveFailure` to signal expected failure; anything else escaping is a
#: host bug and propagates.
Host = Callable[..., Any]


class DispatchError(ValueError):
    """The dispatch table and the scheme disagree."""


class EvaluationError(RuntimeError):
    """A program could not be evaluated. Indicates a lowering or host bug —
    the mask should have made every other failure mode unreachable."""


@dataclass(frozen=True)
class Dispatch:
    """Scheme + host implementations, checked to correspond exactly."""

    scheme: Scheme
    hosts: dict[str, Host]

    #: How this language spells a wrapped fallible return. Must be the same
    #: template as the `LanguageBinding` that generated the grammar, or the
    #: type the evaluator reports for a call will not be the type the mask
    #: constrained it to. `session.build()` wires both from one binding;
    #: construct `Dispatch` directly only in tests.
    result_type: str = "Result[{ok}, {err}]"

    def __post_init__(self) -> None:
        declared = {p.name for p in self.scheme.primitives}
        implemented = set(self.hosts)

        missing = declared - implemented
        if missing:
            raise DispatchError(
                f"scheme declares primitives with no implementation: "
                f"{sorted(missing)}. The grammar would admit calls that cannot run."
            )

        extra = implemented - declared
        if extra:
            raise DispatchError(
                f"implementations with no scheme entry: {sorted(extra)}. "
                "These are unreachable — the mask cannot emit a call to a "
                "primitive absent from the grammar."
            )

    def primitive(self, name: str) -> Primitive:
        prim = self.scheme.get(name)
        if prim is None:
            raise EvaluationError(f"unknown primitive {name!r}")
        return prim


# ── Effect audit ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EffectAudit:
    """Everything a program will do, computed before it does any of it.

    This is what atomic turns buy (ARCHITECTURE §2.1): the whole effect set is
    knowable up front, so approval is granted once against a complete picture
    rather than discovered one call at a time.
    """

    effects: tuple[Effect, ...]
    primitives: tuple[str, ...]

    @property
    def is_pure(self) -> bool:
        return not self.effects

    def kinds(self) -> frozenset[str]:
        return frozenset(e.kind for e in self.effects)

    def render(self) -> str:
        if self.is_pure:
            return "no effects"
        return "\n".join(f"  {e}" for e in self.effects)


def audit(program: Program, dispatch: Dispatch) -> EffectAudit:
    """Static effect analysis of a whole turn."""
    effects: dict[tuple[str, str], Effect] = {}
    names: list[str] = []
    for call in program.calls():
        prim = dispatch.primitive(call.primitive)
        if prim.name not in names:
            names.append(prim.name)
        for eff in prim.effects:
            effects.setdefault((eff.kind, eff.scope), eff)
    return EffectAudit(effects=tuple(effects.values()), primitives=tuple(names))


# ── Turn outcome ─────────────────────────────────────────────────────────


@dataclass
class TurnOutcome:
    """The result of evaluating one turn."""

    #: New bindings, name -> (type source, value). Empty when aborted.
    updates: dict[str, tuple[TypeSource, Any]] = field(default_factory=dict)
    #: Set when the turn did not run at all.
    aborted: bool = False
    #: Why it aborted: "todo" (model failure) or "denied" (approval refused).
    reason: str = ""
    #: Model's note, when it emitted `todo`.
    note: str = ""
    #: What the turn was going to do, whether or not it did it.
    effects: EffectAudit | None = None

    @property
    def ok(self) -> bool:
        return not self.aborted


# ── Evaluation ───────────────────────────────────────────────────────────


class Evaluator:
    """Executes a lowered turn against client-side Γ.

    Atomic: the whole program runs or none of it does (ARCHITECTURE §2.1).
    """

    def __init__(
        self,
        dispatch: Dispatch,
        *,
        approve: Callable[[EffectAudit], bool] | None = None,
    ):
        self.dispatch = dispatch
        #: Called once per turn with the complete effect set, before anything
        #: runs. Default approves everything — gamma supplies the real policy.
        self.approve = approve or (lambda _audit: True)

    def run(self, program: Program, gamma: Gamma, *, turn: int = 0) -> TurnOutcome:
        """Evaluate one turn. Does not mutate `gamma`; the caller commits."""
        # Model-failure channel: `todo` means the model could not write a
        # program. Abort before any effect, leave Γ untouched, let it retry.
        if program.has_todo():
            return TurnOutcome(
                aborted=True,
                reason="todo",
                note=_first_todo_note(program),
                effects=audit(program, self.dispatch),
            )

        effects = audit(program, self.dispatch)
        if not self.approve(effects):
            return TurnOutcome(aborted=True, reason="denied", effects=effects)

        # Local scope for the duration of the turn: statements see bindings
        # made earlier in the same sequence, which is what the grammar's
        # right-recursive effect propagation already guarantees statically.
        local: dict[str, tuple[TypeSource, Any]] = {}

        def resolve(name: str) -> Any:
            if name in local:
                return local[name][1]
            if name in gamma:
                return gamma.value(name)
            raise EvaluationError(
                f"unbound name {name!r} reached the evaluator; the mask should "
                "have made this unemittable, so this is a lowering bug"
            )

        def evaluate(expr: Expr) -> tuple[Any, TypeSource]:
            if isinstance(expr, Lit):
                return expr.value, expr.type
            if isinstance(expr, Var):
                value = resolve(expr.name)
                type_source = (
                    local[expr.name][0] if expr.name in local else gamma.type_of(expr.name)
                )
                return value, type_source
            if isinstance(expr, Call):
                return self._call(expr, evaluate)
            if isinstance(expr, Todo):
                raise EvaluationError("todo reached evaluation; run() should have aborted")
            raise EvaluationError(f"unknown expression node: {type(expr).__name__}")

        for stmt in program.statements:
            value, type_source = evaluate(stmt.expr)
            if isinstance(stmt, Bind):
                local[stmt.name] = (type_source, value)
            elif not isinstance(stmt, Do):
                raise EvaluationError(f"unknown statement node: {type(stmt).__name__}")

        return TurnOutcome(updates=local, effects=effects)

    def _call(
        self, call: Call, evaluate: Callable[[Expr], tuple[Any, TypeSource]]
    ) -> tuple[Any, TypeSource]:
        prim = self.dispatch.primitive(call.primitive)

        if len(call.args) != len(prim.params):
            raise EvaluationError(
                f"{prim.name} takes {len(prim.params)} argument(s), "
                f"got {len(call.args)}; the grammar fixes arity, so this is a "
                "lowering bug"
            )

        args = [evaluate(arg)[0] for arg in call.args]
        host = self.dispatch.hosts[prim.name]

        if not prim.fallible:
            # Total primitive: a raise here is a host bug and propagates.
            return host(*args), prim.returns

        # Fallible primitive: convert the host's exception into a value at the
        # language boundary, so no exceptional control flow enters the language
        # (ARCHITECTURE I1a). The static return type is the Result wrapper —
        # the caller must match on it before using the payload.
        wrapped = self._result_type(prim)
        try:
            return Ok(host(*args)), wrapped
        except PrimitiveFailure as failure:
            return Err(failure.error), wrapped

    def _result_type(self, prim: Primitive) -> TypeSource:
        """The wrapped return type a call to `prim` has.

        Must agree with `LanguageBinding.wrap_return`, which produced the
        conclusion of this primitive's typing rule. Both read `returns` and
        `raises` off the same `Primitive` and use the same template, so they
        agree by construction as long as `Dispatch.result_type` was wired from
        the binding — which `session.build()` does and `test_evaluator.py`
        asserts.
        """
        return self.dispatch.result_type.format(ok=prim.returns, err=prim.raises)


def _first_todo_note(program: Program) -> str:
    for stmt in program.statements:
        expr = stmt.expr
        if isinstance(expr, Todo):
            return expr.note
    return ""
