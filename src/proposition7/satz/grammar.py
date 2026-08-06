"""Scheme → `.auf` grammar fragment, and composition with the core language.

This is one of the two projections of a `Scheme` (the other is
`proposition7.satz.evaluator`).  Both are generated from the same table so they cannot
drift — ARCHITECTURE I2.

## The seam with the core language

this package does **not** define the agent language.  That is D3, and it lives in a
`.auf` file authored against aufbau.  What it defines is the *contract*
that file must satisfy so primitives can be spliced into it:

1.  It declares a `Type*` fragment, as any aufbau grammar must.
2.  It contains the marker `@PRIMITIVES@` in exactly one production, at the
    alternative position where a primitive call is a legal expression:

        Expression ::= AtomicExpr | ArithOp | @PRIMITIVES@

3.  It uses the binding's `expression_nt` (default `Expression`) as the
    nonterminal for argument positions.
4.  It defines whatever `result_type` names — a sum type with two variants —
    and a `todo` term inhabiting every sort (ARCHITECTURE §2.2).

Composition is textual substitution of the marker.  That is deliberate: the
composed artifact is a plain `.auf` file, readable and diffable, and there is
no grammar-combination framework to debug when something goes wrong.

## Determinism

`compose()` is a pure function of (core source, scheme) and is byte-stable.
The SPG compile cache is keyed by content hash, so instability here would
recompile the grammar on every request (ARCHITECTURE §3.1).  `test_grammar.py`
asserts it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .scheme import Primitive, Scheme

#: Substituted by `compose()` with the primitive-call nonterminal.
PRIMITIVE_MARKER = "@PRIMITIVES@"

#: Nonterminal the generated fragment defines.
PRIMITIVE_NT = "PrimitiveCall"


@dataclass(frozen=True)
class LanguageBinding:
    """How a core language spells the things this package needs to generate.

    Keeping these as parameters rather than hardcoding them means D3 can
    settle the surface syntax without this package changing at all.
    """

    #: `.auf` source of the core agent language (D3's artifact).
    core_source: str

    #: Nonterminal for argument positions in a primitive call.
    expression_nt: str = "Expression"

    #: Template wrapping a fallible primitive's return type. Must name a sum
    #: type the core language defines. `{ok}` and `{err}` are substituted with
    #: raw type source, so it composes with arbitrary type terms.
    result_type: str = "Result[{ok}, {err}]"

    #: Prefix for generated typing-rule labels. Must not collide with any rule
    #: label in `core_source`.
    rule_prefix: str = "prim_"

    def wrap_return(self, primitive: Primitive) -> str:
        """The type a call to `primitive` has.

        Fallibility is applied here and only here, which is what makes I1a
        structural: there is no path from a `raises` declaration to a bare
        return type.
        """
        if not primitive.fallible:
            return primitive.returns
        return self.result_type.format(ok=primitive.returns, err=primitive.raises)


class CompositionError(ValueError):
    """The core language does not satisfy the contract above."""


# ── Fragment generation ──────────────────────────────────────────────────


def _nonterminal(name: str) -> str:
    """`read_file` -> `PrimReadFile`. Stable and collision-resistant."""
    return "Prim" + "".join(part.capitalize() for part in name.split("_"))


def _rule_label(binding: LanguageBinding, name: str) -> str:
    return f"{binding.rule_prefix}{name}"


def _production(binding: LanguageBinding, prim: Primitive) -> str:
    nt = _nonterminal(prim.name)
    label = _rule_label(binding, prim.name)
    if not prim.params:
        return f"{nt}({label}) ::= '{prim.name}' '(' ')'"
    # The separator must be a *quoted* grammar literal. A bare comma is
    # tokenized as part of the production's structure and silently makes every
    # multi-argument call dead — a bug this codebase has already hit once.
    slots = " ',' ".join(
        f"{binding.expression_nt}[arg{i}]" for i in range(len(prim.params))
    )
    return f"{nt}({label}) ::= '{prim.name}' '(' {slots} ')'"


def _typing_rule(binding: LanguageBinding, prim: Primitive) -> str:
    label = _rule_label(binding, prim.name)
    conclusion = binding.wrap_return(prim)
    if not prim.params:
        return f"------------------------- ({label})\n{conclusion}"
    # Types are emitted as raw source, never quoted atoms: a parameter type may
    # be any term the grammar's `Type*` fragment derives, including constructors
    # like `Code[python, module]`. Quoting would flatten those to literals.
    premises = ", ".join(
        f"Γ ⊢ arg{i} : {p.type}" for i, p in enumerate(prim.params)
    )
    return f"{premises}\n------------------------- ({label})\n{conclusion}"


def fragment(scheme: Scheme, binding: LanguageBinding) -> str:
    """The `.auf` source declaring every primitive in `scheme`.

    Returns empty string for an empty scheme — a legitimate configuration
    (pure computation, no capabilities granted at all).
    """
    prims = scheme.ordered()
    if not prims:
        return ""

    productions = "\n".join(_production(binding, p) for p in prims)
    alternatives = " | ".join(_nonterminal(p.name) for p in prims)
    rules = "\n\n".join(_typing_rule(binding, p) for p in prims)

    return f"""// ===================== Primitives (generated) =====================
// Generated from a Scheme. Do not edit: edit the scheme.
// The set of primitives here IS the set of granted capabilities.

{productions}

{PRIMITIVE_NT} ::= {alternatives}

// ---------------------- Primitive typing rules --------------------

{rules}
"""


# ── Composition ──────────────────────────────────────────────────────────


def compose(scheme: Scheme, binding: LanguageBinding) -> str:
    """Splice `scheme`'s primitives into the core language.

    The result is a complete `.auf` source, ready to hand to p7 as
    `constraint_scheme_auf`.
    """
    core = binding.core_source
    marker_count = core.count(PRIMITIVE_MARKER)
    if marker_count == 0:
        raise CompositionError(
            f"core language declares no {PRIMITIVE_MARKER} splice point; "
            "primitives would be unreachable from any production"
        )
    if marker_count > 1:
        raise CompositionError(
            f"core language declares {marker_count} {PRIMITIVE_MARKER} markers, "
            "expected exactly 1"
        )

    prims = scheme.ordered()
    if not prims:
        # Remove the marker and the alternation bar that introduced it, so an
        # empty scheme yields a valid grammar rather than a dangling `|`.
        spliced = re.sub(
            rf"\s*\|\s*{re.escape(PRIMITIVE_MARKER)}", "", core
        )
        if PRIMITIVE_MARKER in spliced:
            # The marker was the sole alternative; nothing sensible remains.
            raise CompositionError(
                f"{PRIMITIVE_MARKER} is the only alternative in its production, "
                "so an empty scheme leaves it underivable; give the production "
                "at least one non-primitive alternative"
            )
        return spliced

    spliced = core.replace(PRIMITIVE_MARKER, PRIMITIVE_NT)

    # The fragment goes BEFORE the core, not after. aufbau takes the
    # last-declared nonterminal as the grammar's start symbol, so appending the
    # fragment silently makes `PrimitiveCall` the root and the grammar then
    # accepts a bare primitive call as a whole program. Emitting the fragment
    # first leaves whatever the core declares last as the start symbol, so
    # composition cannot change the language's root.
    return f"{fragment(scheme, binding)}\n{spliced}"


def validate_binding(binding: LanguageBinding) -> list[str]:
    """Static checks on a core language, as a list of problems.

    Empty list means the contract above is satisfied. This catches the
    failures that would otherwise surface as an opaque aufbau parse error
    or, worse, as a grammar that compiles and silently masks nothing.
    """
    problems: list[str] = []
    core = binding.core_source

    count = core.count(PRIMITIVE_MARKER)
    if count != 1:
        problems.append(
            f"expected exactly 1 {PRIMITIVE_MARKER} marker, found {count}"
        )

    if not re.search(rf"^\s*{re.escape(binding.expression_nt)}\b", core, re.M):
        problems.append(
            f"argument nonterminal {binding.expression_nt!r} is never declared"
        )

    if "Type*" not in core and "Type *" not in core:
        problems.append(
            "no `Type*` fragment declared; aufbau needs one to elaborate "
            "the typing rules' type expressions"
        )

    # The result constructor must exist, or every fallible primitive's return
    # type elaborates against a nonexistent sort.
    ctor = binding.result_type.split("[", 1)[0].strip()
    if ctor and ctor not in core:
        problems.append(
            f"result constructor {ctor!r} is not declared by the core language, "
            "so fallible primitives would have an underivable return type"
        )

    if binding.rule_prefix and re.search(
        rf"\({re.escape(binding.rule_prefix)}", core
    ):
        problems.append(
            f"core language already uses rule labels prefixed {binding.rule_prefix!r}; "
            "generated labels would collide"
        )

    return problems
