"""The execution enclave — the one place in p7 that runs anything.

## Read this before using or extending this module

Everything else under `proposition7` and `aufbau` is **pure checking**. aufbau
decides whether a prefix is typed, live, or dead. p7 masks logits against that
verdict. Neither opens a file, spawns a process, or touches a network. Importing
them cannot cause a side effect, and that is a property people rely on when they
put p7 on an inference host.

This module is the exception. It *executes*. It is an enclave inside an
otherwise effect-free library, and it must be read as such — "p7 runs code" is
false everywhere except here.

## What keeps the enclave safe

**It ships no capabilities.** `Dispatch(hosts=...)` is entirely caller-supplied.
There is no `open`, no `subprocess`, no socket anywhere in `proposition7.satz` —
the functions that touch the world are gamma's, handed in at construction. This
module is a dispatcher with an empty hand.

**It cannot acquire one.** A primitive absent from the scheme is absent from the
generated grammar *and* absent from the dispatch table, so it is neither
emittable by the model nor callable by the evaluator. `Dispatch.__post_init__`
rejects a host with no scheme entry precisely so a capability cannot be smuggled
in past the constraint (ARCHITECTURE I7).

**Its blast radius is enumerable before it runs.** `audit()` returns the whole
effect set of a turn from the parse tree alone, which is what atomic turns are
for (ARCHITECTURE §2.1).

**provider7 must never import it.** That is what keeps the inference host
non-executing, and it is the invariant most likely to be broken by someone
reaching for a convenient import. provider7 receives composed `.auf` and returns
completions; the client executes (ARCHITECTURE I3).

## How it reads the program

There is no intermediate representation and no lowering function. aufbau
already produces a typed tree, so building a second one in Python would mean a
second encoding of the grammar's productions and a second derivation of its
types — the drift I2 forbids for primitives, applied to the language itself.

Everything structural comes from the FFI:

    node.nt_name()             which production matched
    node.children              sub-nodes
    node.text                  source text
    ast.type_of(evidence)      the engine's type Term
    spg.show(term)             that Term rendered back to type source

Nothing here re-derives a type. When the evaluator reports that a binding has
type `Result[PathSet, IoError]`, that string came from the engine that masked
the decode, not from a Python reconstruction of the same rule.

The dispatch keys close the loop: `grammar._nonterminal()` generates the
nonterminal name for each primitive, and `Dispatch.by_node()` inverts exactly
that function. Scheme → grammar → node name → dispatch → scheme, with no
hand-maintained table anywhere along the path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .context import Gamma
from .grammar import LanguageBinding, _nonterminal
from .result import Err, Ok, PrimitiveFailure
from .scheme import Effect, Primitive, Scheme, TypeSource

#: A host implementation. Ordinary Python, called positionally. It may raise
#: `PrimitiveFailure` to signal expected failure; anything else escaping is a
#: host bug and propagates.
Host = Callable[..., Any]


class DispatchError(ValueError):
    """The dispatch table and the scheme disagree."""


class EvaluationError(RuntimeError):
    """A program could not be evaluated. Indicates a host bug or a gap between
    the language binding and the grammar — the mask should have made every
    other failure mode unreachable."""


@dataclass(frozen=True)
class Dispatch:
    """Scheme + host implementations, checked to correspond exactly."""

    scheme: Scheme
    hosts: dict[str, Host]

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

    def by_node(self) -> dict[str, Primitive]:
        """Nonterminal name -> primitive, inverting `grammar._nonterminal`."""
        return {_nonterminal(p.name): p for p in self.scheme.ordered()}


# ── Effect audit ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EffectAudit:
    """Everything a program could do, computed before it does any of it.

    This is what atomic turns buy (ARCHITECTURE §2.1). With branches it is a
    conservative over-approximation — a call in an arm that will not be taken
    still appears — which is the right bias for approval: the user sees
    everything the turn *could* do.
    """

    effects: tuple[Effect, ...] = ()
    primitives: tuple[str, ...] = ()

    @property
    def is_pure(self) -> bool:
        return not self.effects

    def kinds(self) -> frozenset[str]:
        return frozenset(e.kind for e in self.effects)

    def render(self) -> str:
        return "no effects" if self.is_pure else "\n".join(f"  {e}" for e in self.effects)


@dataclass
class TurnOutcome:
    """The result of evaluating one turn."""

    updates: dict[str, tuple[TypeSource, Any]] = field(default_factory=dict)
    aborted: bool = False
    #: "todo" (model failure), "denied" (approval refused), or "incomplete:…".
    reason: str = ""
    note: str = ""
    effects: EffectAudit | None = None

    @property
    def ok(self) -> bool:
        return not self.aborted


# ── Evaluation ───────────────────────────────────────────────────────────


class Evaluator:
    """Executes a decoded turn against client-side Γ.

    Atomic: the whole program runs or none of it does (ARCHITECTURE §2.1).

    `spg` must be the *same* compiled grammar the decode was masked under.
    Re-parsing the completion under it is what makes the tree — and therefore
    every type in it — the engine's answer rather than this module's opinion.
    """

    def __init__(
        self,
        dispatch: Dispatch,
        binding: LanguageBinding,
        spg: Any,
        *,
        approve: Callable[[EffectAudit], bool] | None = None,
    ):
        self.dispatch = dispatch
        self.binding = binding
        self.spg = spg
        self.approve = approve or (lambda _audit: True)
        self._by_node = dispatch.by_node()

    # ── tree access, all via the FFI ────────────────────────────────────

    def _nodes(self, node: Any) -> list[Any]:
        """Child nodes, skipping literal-only children."""
        out = []
        for child in node.children:
            inner = getattr(child, "node", None)
            if inner is not None:
                out.append(inner)
        return out

    def _type(self, ast: Any, node: Any) -> TypeSource:
        """The engine's type for a node, rendered back to type source."""
        term = ast.type_of(node.evidence)
        if term is None:
            return ""
        # `type_of` may hand back an already-rendered type or a Term, depending
        # on the evidence. Either way the *engine* produced it — this branch
        # chooses a spelling, it never derives a type.
        return term if isinstance(term, str) else self.spg.show(term)

    def _walk(self, node: Any):
        yield node
        for child in self._nodes(node):
            yield from self._walk(child)

    # ── audit ───────────────────────────────────────────────────────────

    def audit(self, ast: Any) -> EffectAudit:
        effects: dict[tuple[str, str], Effect] = {}
        names: list[str] = []
        for root in ast.roots:
            for node in self._walk(root):
                prim = self._by_node.get(node.nt_name())
                if prim is None:
                    continue
                if prim.name not in names:
                    names.append(prim.name)
                for eff in prim.effects:
                    effects.setdefault((eff.kind, eff.scope), eff)
        return EffectAudit(effects=tuple(effects.values()), primitives=tuple(names))

    def _has_todo(self, ast: Any) -> bool:
        return any(
            node.nt_name() == self.binding.todo_nt
            for root in ast.roots
            for node in self._walk(root)
        )

    # ── run ─────────────────────────────────────────────────────────────

    def parse(self, completion: str) -> Any:
        """Re-parse a completion under the grammar it was decoded against."""
        import aufbau

        synth = aufbau.Synthesizer.from_grammar(self.spg, completion)
        return synth.ast()

    def run(self, completion: str, gamma: Gamma, *, turn: int = 0) -> TurnOutcome:
        ast = self.parse(completion)
        if not ast.is_complete():
            return TurnOutcome(aborted=True, reason="incomplete:parse")

        effects = self.audit(ast)

        # Model-failure channel: `todo` means the model could not write a
        # program. Abort before any effect; Γ is untouched and it can retry.
        if self._has_todo(ast):
            return TurnOutcome(aborted=True, reason="todo", effects=effects)

        if not self.approve(effects):
            return TurnOutcome(aborted=True, reason="denied", effects=effects)

        local: dict[str, tuple[TypeSource, Any]] = {}

        def value_of(name: str, env: dict) -> Any:
            if name in env:
                return env[name][1]
            if name in local:
                return local[name][1]
            if name in gamma:
                return gamma.value(name)
            raise EvaluationError(
                f"unbound name {name!r} reached the evaluator; the mask should "
                "have made this unemittable"
            )

        def evaluate(node: Any, env: dict) -> Any:
            name = node.nt_name()

            prim = self._by_node.get(name)
            if prim is not None:
                args = [evaluate(child, env) for child in self._nodes(node)]
                return self._call(prim, args)

            # A single-alternative production is transparent in aufbau's tree,
            # so `Expression -> Variable -> Identifier` can collapse and the
            # bare identifier is what a variable reference looks like here.
            if name in (self.binding.variable_nt, self.binding.identifier_nt):
                return value_of(node.text.strip(), env)

            if name in self.binding.literal_nts:
                return node.text.strip()

            children = self._nodes(node)
            if len(children) == 1:
                # Transparent wrapper (Expression -> AtomicExpr -> …).
                return evaluate(children[0], env)
            raise EvaluationError(
                f"no evaluation rule for {name!r} with {len(children)} children; "
                "the language binding needs to name this production"
            )

        for root in ast.roots:
            for stmt in self._statements(root):
                bound, expr = self._binding_of(stmt)
                value = evaluate(expr, {})
                if bound is not None:
                    local[bound] = (self._type(ast, expr), value)

        return TurnOutcome(updates=local, effects=effects)

    # ── statement shape: the one place D3's productions are named ───────

    def _statements(self, root: Any) -> list[Any]:
        """Flatten the right-recursive StatementList into a sequence."""
        out: list[Any] = []
        stack = [root]
        while stack:
            node = stack.pop(0)
            if node.nt_name() == self.binding.statement_nt:
                out.append(node)
                continue
            stack = self._nodes(node) + stack
        return out

    def _binding_of(self, stmt: Any) -> tuple[str | None, Any]:
        """`(name, value_node)` for a binding statement; `(None, node)` else."""
        children = self._nodes(stmt)
        if len(children) >= 2 and children[0].nt_name() == self.binding.identifier_nt:
            return children[0].text.strip(), children[-1]
        return None, children[-1] if children else stmt

    def _call(self, prim: Primitive, args: list[Any]) -> Any:
        if len(args) != len(prim.params):
            raise EvaluationError(
                f"{prim.name} takes {len(prim.params)} argument(s), got {len(args)}; "
                "the grammar fixes arity, so this is a binding mismatch"
            )
        host = self.dispatch.hosts[prim.name]

        if not prim.fallible:
            return host(*args)

        # Convert the host's exception into a value at the language boundary,
        # so no exceptional control flow enters the language (I1a). The static
        # type is the Result wrapper; the model must match before using it.
        try:
            return Ok(host(*args))
        except PrimitiveFailure as failure:
            return Err(failure.error)
