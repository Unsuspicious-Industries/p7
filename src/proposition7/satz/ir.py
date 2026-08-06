"""The intermediate representation a turn's program lowers to.

## Why an IR exists

The evaluator has to walk a program, but the program's concrete syntax is D3
— the core language, not yet designed.  Rather than block the whole runtime on
that decision, the evaluator is written against this IR and D3 supplies one
function:

    aufbau Ast  ──lower()──>  Program   (D3 writes this)
    Program     ──evaluate()─> Gamma'   (proposition7.satz.evaluator, already written)

So the surface syntax can change freely without touching the runtime, and the
runtime is testable today by constructing IR directly.

## Shape

The IR mirrors ARCHITECTURE §2: a turn is a `StatementList`, statements bind
names into a flat same-scope Γ, and the sequence is what carries bindings
forward.  It is deliberately minimal — this is the *runtime's* view, not the
language's.  Pattern matching, arithmetic, and control flow are D3's concern
and lower to `Call`s or to whatever `Expr` variants D3 adds here later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from .scheme import TypeSource


@dataclass(frozen=True)
class Var:
    """Reference to a name in Γ.

    Cannot be unbound: the mask rejects any prefix naming a variable absent
    from Γ, so by the time a program reaches the evaluator every `Var` resolves.
    The evaluator still checks, because a lowering bug should surface as an
    error rather than a wrong answer.
    """

    name: str


@dataclass(frozen=True)
class Lit:
    """A literal, carrying the type the grammar assigned it."""

    value: Any
    type: TypeSource


@dataclass(frozen=True)
class Call:
    """Application of a primitive to arguments."""

    primitive: str
    args: tuple["Expr", ...] = ()


@dataclass(frozen=True)
class Todo:
    """The universal inhabitant (ARCHITECTURE §2.2).

    Present so the mask can never trap a small model in a `live`-but-
    uncompletable prefix. A program containing `Todo` is well-typed and
    *unexecutable by construction*: the evaluator aborts the turn before
    running anything, so no effect occurs and Γ is unchanged.

    This is the model-failure channel. It is not an exception, and it must not
    be confused with the expected-failure channel, which is `Result` — see
    `proposition7.satz.result`.
    """

    #: Optional note the model emitted about why it could not proceed.
    note: str = ""


Expr = Union[Var, Lit, Call, Todo]


@dataclass(frozen=True)
class Bind:
    """`name = expr;` — binds into Γ for the rest of the sequence and for
    every subsequent turn."""

    name: str
    expr: Expr


@dataclass(frozen=True)
class Do:
    """An expression evaluated for effect, binding nothing."""

    expr: Expr


Stmt = Union[Bind, Do]


@dataclass(frozen=True)
class Program:
    """One turn: a statement sequence."""

    statements: tuple[Stmt, ...] = ()

    def calls(self) -> tuple[Call, ...]:
        """Every primitive call in the program, in evaluation order.

        Used for the pre-execution effect audit (ARCHITECTURE §2.1), which is
        only possible because a turn is a complete program rather than a cell
        that is discovered one step at a time.
        """
        found: list[Call] = []

        def walk(expr: Expr) -> None:
            if isinstance(expr, Call):
                for arg in expr.args:
                    walk(arg)
                found.append(expr)

        for stmt in self.statements:
            walk(stmt.expr)
        return tuple(found)

    def has_todo(self) -> bool:
        def walk(expr: Expr) -> bool:
            if isinstance(expr, Todo):
                return True
            if isinstance(expr, Call):
                return any(walk(a) for a in expr.args)
            return False

        return any(walk(s.expr) for s in self.statements)
