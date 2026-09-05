"""Compose generic primitive declarations into `.auf` source."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .declaration import Primitive, Scheme

PRIMITIVE_MARKER = "@PRIMITIVES@"
PRIMITIVE_NT = "PrimitiveCall"


@dataclass(frozen=True)
class LanguageBinding:
    """The composition contract supplied by a concrete language."""

    core_source: str
    expression_nt: str = "Expression"
    result_type: str = "Result[{ok}, {err}]"
    rule_prefix: str = "prim_"

    def wrap_return(self, primitive: Primitive) -> str:
        if not primitive.fallible:
            return primitive.returns
        return self.result_type.format(ok=primitive.returns, err=primitive.raises)


class CompositionError(ValueError):
    pass


def nonterminal(name: str) -> str:
    return "Prim" + "".join(part.capitalize() for part in name.split("_"))


def fragment(scheme: Scheme, binding: LanguageBinding) -> str:
    primitives = scheme.ordered()
    if not primitives:
        return ""

    def production(primitive: Primitive) -> str:
        node, label = nonterminal(primitive.name), binding.rule_prefix + primitive.name
        if not primitive.params:
            return f"{node}({label}) ::= '{primitive.name}' '(' ')'"
        slots = " ',' ".join(
            f"{binding.expression_nt}[arg{i}]" for i in range(len(primitive.params))
        )
        return f"{node}({label}) ::= '{primitive.name}' '(' {slots} ')'"

    def rule(primitive: Primitive) -> str:
        label, output = binding.rule_prefix + primitive.name, binding.wrap_return(primitive)
        if not primitive.params:
            return f"------------------------- ({label})\n{output}"
        premises = ", ".join(
            f"Γ ⊢ arg{i} : {param.type}" for i, param in enumerate(primitive.params)
        )
        return f"{premises}\n------------------------- ({label})\n{output}"

    return (
        "// Primitives (generated from a Scheme; edit the scheme, not this).\n"
        + "\n".join(production(primitive) for primitive in primitives)
        + f"\n\n{PRIMITIVE_NT} ::= "
        + " | ".join(nonterminal(primitive.name) for primitive in primitives)
        + "\n\n"
        + "\n\n".join(rule(primitive) for primitive in primitives)
        + "\n"
    )


def compose(scheme: Scheme, binding: LanguageBinding) -> str:
    """Return byte-stable grammar source for a scheme and language binding."""
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
    return fragment(scheme, binding) + "\n" + core.replace(PRIMITIVE_MARKER, PRIMITIVE_NT)


def validate_binding(binding: LanguageBinding) -> list[str]:
    """Return every violated composition-contract condition."""
    core, problems = binding.core_source, []
    if core.count(PRIMITIVE_MARKER) != 1:
        problems.append(f"expected exactly 1 {PRIMITIVE_MARKER}")
    if not re.search(rf"^\s*{re.escape(binding.expression_nt)}\b", core, re.M):
        problems.append(f"{binding.expression_nt!r} is never declared")
    if "Type*" not in core:
        problems.append("no `Type*` fragment; aufbau cannot elaborate rule types")
    constructor = binding.result_type.split("[", 1)[0].strip()
    if constructor and constructor not in core:
        problems.append(f"result constructor {constructor!r} is not declared")
    if re.search(rf"\({re.escape(binding.rule_prefix)}", core):
        problems.append(f"core already uses {binding.rule_prefix!r} rule labels")
    return problems
