"""Scheme -> .auf, spliced into a core language.

Core language contract (D3 supplies the file):
  1. declares a `Type*` fragment
  2. contains `@PRIMITIVES@` exactly once, at an alternative position where a
     primitive call is a legal expression
  3. uses `expression_nt` for argument positions
  4. defines what `result_type` names, and a `todo` term inhabiting every sort

`compose()` is a pure function of (core, scheme) and byte-stable; the SPG cache
is keyed by content hash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .scheme import Primitive, Scheme

PRIMITIVE_MARKER = "@PRIMITIVES@"
PRIMITIVE_NT = "PrimitiveCall"


@dataclass(frozen=True)
class LanguageBinding:
    """How a core language spells what satz generates and walks.

    Temporary public API while D3, the core language, remains unresolved.
    """

    core_source: str

    #: Argument positions in a generated primitive call.
    expression_nt: str = "Expression"
    #: Wrapper for a fallible return. Must name a sum type the core defines.
    result_type: str = "Result[{ok}, {err}]"
    #: Prefix for generated rule labels. Must not collide with the core's.
    rule_prefix: str = "prim_"

    # Node names the evaluator dispatches on. Primitive calls need no entry:
    # their nonterminals are generated here and inverted by Dispatch.by_node().
    statement_nt: str = "Stmt"
    identifier_nt: str = "Identifier"
    variable_nt: str = "Variable"
    literal_nts: tuple[str, ...] = ("StringLit", "IntLit")
    todo_nt: str = "Todo"

    def wrap_return(self, prim: Primitive) -> str:
        if not prim.fallible:
            return prim.returns
        return self.result_type.format(ok=prim.returns, err=prim.raises)


class CompositionError(ValueError):
    pass


def nonterminal(name: str) -> str:
    """`read_file` -> `PrimReadFile`. Inverted by `Dispatch.by_node()`."""
    return "Prim" + "".join(part.capitalize() for part in name.split("_"))


def fragment(scheme: Scheme, binding: LanguageBinding) -> str:
    prims = scheme.ordered()
    if not prims:
        return ""

    def production(p: Primitive) -> str:
        nt, label = nonterminal(p.name), binding.rule_prefix + p.name
        if not p.params:
            return f"{nt}({label}) ::= '{p.name}' '(' ')'"
        # The separator must be a quoted literal; a bare comma silently makes
        # every multi-argument call dead.
        slots = " ',' ".join(
            f"{binding.expression_nt}[arg{i}]" for i in range(len(p.params))
        )
        return f"{nt}({label}) ::= '{p.name}' '(' {slots} ')'"

    def rule(p: Primitive) -> str:
        label, out = binding.rule_prefix + p.name, binding.wrap_return(p)
        if not p.params:
            return f"------------------------- ({label})\n{out}"
        # Types emitted as raw source, not quoted atoms: a parameter type may be
        # any term the `Type*` fragment derives.
        premises = ", ".join(f"Γ ⊢ arg{i} : {q.type}" for i, q in enumerate(p.params))
        return f"{premises}\n------------------------- ({label})\n{out}"

    return (
        "// Primitives (generated from a Scheme; edit the scheme, not this).\n"
        + "\n".join(production(p) for p in prims)
        + f"\n\n{PRIMITIVE_NT} ::= "
        + " | ".join(nonterminal(p.name) for p in prims)
        + "\n\n"
        + "\n\n".join(rule(p) for p in prims)
        + "\n"
    )


def compose(scheme: Scheme, binding: LanguageBinding) -> str:
    """Complete .auf source, ready to send as `constraint_scheme_auf`."""
    core = binding.core_source
    count = core.count(PRIMITIVE_MARKER)
    if count != 1:
        raise CompositionError(f"expected 1 {PRIMITIVE_MARKER}, found {count}")

    if not scheme.primitives:
        spliced = re.sub(rf"\s*\|\s*{re.escape(PRIMITIVE_MARKER)}", "", core)
        if PRIMITIVE_MARKER in spliced:
            raise CompositionError(
                f"{PRIMITIVE_MARKER} is the only alternative in its production; "
                "an empty scheme leaves it underivable"
            )
        return spliced

    # Fragment first: aufbau takes the last-declared nonterminal as the start
    # symbol, so appending would make PrimitiveCall the root and the grammar
    # would accept a bare primitive call as a whole program.
    return fragment(scheme, binding) + "\n" + core.replace(PRIMITIVE_MARKER, PRIMITIVE_NT)


def validate_binding(binding: LanguageBinding) -> list[str]:
    """Problems with a core language; empty list means the contract holds."""
    core, problems = binding.core_source, []
    if core.count(PRIMITIVE_MARKER) != 1:
        problems.append(f"expected exactly 1 {PRIMITIVE_MARKER}")
    if not re.search(rf"^\s*{re.escape(binding.expression_nt)}\b", core, re.M):
        problems.append(f"{binding.expression_nt!r} is never declared")
    if "Type*" not in core:
        problems.append("no `Type*` fragment; aufbau cannot elaborate rule types")
    ctor = binding.result_type.split("[", 1)[0].strip()
    if ctor and ctor not in core:
        problems.append(f"result constructor {ctor!r} is not declared")
    if re.search(rf"\({re.escape(binding.rule_prefix)}", core):
        problems.append(f"core already uses {binding.rule_prefix!r} rule labels")
    return problems
