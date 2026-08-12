"""Temporary agent-language binding for benchmark episodes.

This is a D3 placeholder, expected to disappear when the public core language
lands. `tests/satz_wiring.py` deliberately carries the other small stand-in.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable

from proposition7.satz import Effect, LanguageBinding, Param, Primitive, Scheme


CORE = r'''Identifier ::= /[a-z_][a-z0-9_]*/
Type* ::= TAtom
TAtom ::= 'PathSet' | 'Text' | 'IoError' | 'ProcError' | Result
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


BINDING = LanguageBinding(core_source=CORE)
CAPABILITIES = Scheme((
    Primitive("read_file", (Param("path", "Text"),), "Text", "IoError", (Effect("read", "<path>"),)),
    Primitive("write_file", (Param("path", "Text"), Param("content", "Text")), "Text", "IoError", (Effect("write", "<path>"),)),
    Primitive("list_dir", (Param("path", "Text"),), "PathSet", "IoError", (Effect("read", "<path>"),)),
    Primitive("search", (Param("pattern", "Text"), Param("root", "Text")), "PathSet", "IoError", (Effect("read", "<root>"),)),
    Primitive("run", (Param("command", "Text"),), "Text", "ProcError", (Effect("execute"),)),
))


def scheme_without(*names: str) -> Scheme:
    return CAPABILITIES.restrict({p.name for p in CAPABILITIES.primitives} - set(names))


# Single-completion value oracles use this deliberately separate fixed world.
@dataclass(frozen=True)
class _Tool:
    name: str
    call: Callable[..., Any]


class _Registry:
    def __init__(self):
        self.tools = {
            "search": _Tool("search", lambda arg: f"docs_about_{arg}"),
            "summarize": _Tool("summarize", lambda arg: f"summary_of_{arg}"),
            "count": _Tool("count", lambda arg: 3),
            "format": _Tool("format", lambda arg: f"formatted_{arg}"),
        }

    def execute(self, name: str, args: list[Any]) -> Any:
        return self.tools[name].call(*args)


MOCK_REGISTRY = _Registry()
_LET_RE = re.compile(r"^let\s+(\w+)\s*=\s*(\w+)\s*\(\s*(.*?)\s*\)\s*;$")
_RETURN_RE = re.compile(r"^return\s+(.*?)\s*;$")
_LET_SEXPR_RE = re.compile(r"^\(let\s+(\w+)\s*\(\s*(\w+)(?:\s+(.*?))?\s*\)\s*\)$")
_RETURN_SEXPR_RE = re.compile(r"^\(return\s+(.*?)\s*\)$")


class InterpretError(Exception):
    pass


def _value(token: str, env: dict[str, Any]) -> Any:
    if re.fullmatch(r'"[a-zA-Z0-9_]*"', token):
        return token[1:-1]
    if token.isdigit():
        return int(token)
    if token in env:
        return env[token]
    raise InterpretError(f"unbound variable {token!r}")


def run_program(program: str, syntax: str = "tool") -> tuple[Any, dict[str, Any]]:
    if syntax == "tool":
        statements = [part.strip() + ";" for part in program.split(";") if part.strip()]
        let_re, return_re, split = _LET_RE, _RETURN_RE, lambda value: [x.strip() for x in value.split(",") if x.strip()]
    elif syntax == "tool_sexpr":
        statements, depth, start = [], 0, None
        for index, char in enumerate(program):
            if char == "(":
                start = index if depth == 0 else start; depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and start is not None:
                    statements.append(program[start:index + 1]); start = None
        let_re, return_re, split = _LET_SEXPR_RE, _RETURN_SEXPR_RE, lambda value: value.split()
    else:
        raise ValueError(f"unknown tool DSL syntax: {syntax}")
    if not statements:
        raise InterpretError("empty program")
    env: dict[str, Any] = {}
    for statement in statements[:-1]:
        match = let_re.match(statement)
        if match is None or match.group(2) not in MOCK_REGISTRY.tools:
            raise InterpretError(f"invalid tool statement {statement!r}")
        args = [] if not (match.group(3) or "").strip() else [_value(x, env) for x in split(match.group(3) or "")]
        env[match.group(1)] = MOCK_REGISTRY.execute(match.group(2), args)
    match = return_re.match(statements[-1])
    if match is None:
        raise InterpretError(f"invalid return {statements[-1]!r}")
    return _value(match.group(1), env), env
