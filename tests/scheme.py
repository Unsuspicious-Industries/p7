"""Generic scheme composition against the real aufbau engine."""

from __future__ import annotations

import subprocess
import sys

import aufbau
import pytest

from proposition7.scheme import (
    LanguageBinding,
    Param,
    Primitive,
    Scheme,
    compose,
    nonterminal,
    validate_binding,
)

CORE = '''Identifier ::= /[a-z_][a-z0-9_]*/
Type* ::= TAtom
TAtom ::= 'Text' | 'PathSet' | 'IoError' | Result
Result ::= 'Result' '[' Type ',' Type ']'
Variable(var) ::= Identifier[x]
StringLit(str_lit) ::= /"[a-zA-Z0-9_]*"/
Todo(todo) ::= 'todo'
Expression ::= Variable | StringLit | Todo | @PRIMITIVES@
Stmt(decl) ::= Identifier[name] '=' Expression[value] ';'
StatementList ::= Stmt StatementList | Stmt
Program ::= StatementList

x ∈ Γ
----------- (var)
Γ(x)

----------- (str_lit)
'Text'

----------- (todo)
?A

Γ ⊢ value : ?t
----------------------- (decl)
Γ → Γ[name:?t] ⊢ 'void'
'''
BINDING = LanguageBinding(CORE)


@pytest.mark.parametrize("arity", range(5))
def test_generated_arities_typecheck_in_the_engine(arity: int) -> None:
    primitive = Primitive(
        f"call_{arity}",
        tuple(Param(f"arg{i}", "Text") for i in range(arity)),
        returns="Text",
    )
    source = compose(Scheme((primitive,)), BINDING)
    arguments = ",".join('"x"' for _ in range(arity))
    call = f"answer = {primitive.name}({arguments});"
    assert aufbau.Synthesizer.from_grammar(aufbau.SPG(source), call).status() == "typed"


def test_fallible_return_is_result_type_in_a_real_rule() -> None:
    primitive = Primitive("fallible", returns="Text", raises="IoError")
    source = compose(Scheme((primitive,)), BINDING)
    assert "Result[Text, IoError]" in source
    assert aufbau.Synthesizer.from_grammar(
        aufbau.SPG(source), "answer = fallible();"
    ).status() == "typed"


def test_composition_is_deterministic_and_labels_have_rules() -> None:
    primitives = (
        Primitive("zeta", returns="Text"),
        Primitive("alpha", (Param("x", "Text"),), returns="Text"),
    )
    source = compose(Scheme(primitives), BINDING)
    assert source == compose(Scheme(tuple(reversed(primitives))), BINDING)
    for primitive in primitives:
        label = BINDING.rule_prefix + primitive.name
        assert f"{nonterminal(primitive.name)}({label})" in source
        assert f"({label})\n" in source


def test_binding_reports_all_contract_errors() -> None:
    problems = validate_binding(LanguageBinding("Expression ::= @PRIMITIVES@ @PRIMITIVES@"))
    assert len(problems) == 3


def test_scheme_is_a_clean_pure_import_surface() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import proposition7.scheme, sys; print(' '.join(sys.modules))"],
        check=True,
        capture_output=True,
        text=True,
    )
    modules = set(result.stdout.split())
    assert not {"torch", "numpy", "proposition7.api", "proposition7.llm", "proposition7.models"} & modules
