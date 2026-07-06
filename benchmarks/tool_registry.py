"""Whole-program interpreter for the `tool`/`tool_sexpr` single-shot DSLs,
built on the same canonical registry `benchmarks/agent.py`'s multi-turn
episodes use (`proposition7.agents.Tool`/`ToolRegistry`) -- one place either
could go stale, not two (lmpl-plan.md section 5.4).

Used by benchmarks/oracles.py's `mode == "value"` grading: a single-shot
`tool`/`tool_sexpr` program is a whole `let`-chain generated at once (unlike
the agent's one-step-per-turn loop), so grading it needs to execute the
*entire* program, not one step.
"""

from __future__ import annotations

import re
from typing import Any

from proposition7.agents import Tool, ToolRegistry

MOCK_REGISTRY = ToolRegistry(
    [
        Tool.unary("search", "string", "docs", lambda arg: f"docs_about_{arg}"),
        Tool.unary("summarize", "docs", "string", lambda arg: f"summary_of_{arg}"),
        Tool.unary("count", "docs", "int", lambda arg: 3),
        Tool.unary("format", "int", "string", lambda arg: f"formatted_{arg}"),
    ]
)

_LET_RE = re.compile(r"^let\s+(\w+)\s*=\s*(\w+)\s*\(\s*(.*?)\s*\)\s*;$")
_RETURN_RE = re.compile(r"^return\s+(.*?)\s*;$")

_LET_SEXPR_RE = re.compile(r"^\(let\s+(\w+)\s*\(\s*(\w+)(?:\s+(.*?))?\s*\)\s*\)$")
_RETURN_SEXPR_RE = re.compile(r"^\(return\s+(.*?)\s*\)$")

_STRING_RE = re.compile(r'^"([a-zA-Z0-9_]*)"$')
_INT_RE = re.compile(r"^[0-9]+$")


class InterpretError(Exception):
    """The program didn't parse/execute against the fixed registry -- should
    be unreachable for a genuinely well-typed program, but the interpreter
    never assumes that and reports clearly rather than crashing silently."""


def eval_value(token: str, env: dict[str, Any]) -> Any:
    if m := _STRING_RE.match(token):
        return m.group(1)
    if _INT_RE.match(token):
        return int(token)
    if token in env:
        return env[token]
    raise InterpretError(f"unbound variable {token!r}")


def _split_args(arg_list: str, syntax: str) -> list[str]:
    """Split a call's argument-list interior into individual Value tokens.
    Safe as a naive split: no Value alternative (Variable/StringLit/IntLit)
    can itself contain a comma or embedded whitespace."""
    if not arg_list.strip():
        return []
    if syntax == "tool":
        return [part.strip() for part in arg_list.split(",")]
    return arg_list.split()


def _split_statements(program: str, syntax: str) -> list[str]:
    if syntax == "tool":
        return [s.strip() + ";" for s in program.split(";") if s.strip()]
    if syntax == "tool_sexpr":
        statements = []
        depth = 0
        start = None
        for i, ch in enumerate(program):
            if ch == "(":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and start is not None:
                    statements.append(program[start : i + 1])
                    start = None
        return statements
    raise ValueError(f"unknown tool DSL syntax: {syntax}")


def run_program(program: str, syntax: str = "tool") -> tuple[Any, dict[str, Any]]:
    """Execute a complete `tool`/`tool_sexpr` program against the mock
    registry. Returns (executed_return_value, final_bindings). Raises
    InterpretError if a statement doesn't match the grammar's own shape or
    calls a tool outside the fixed registry -- this is the grading-side
    twin of what the type checker already guarantees for well-typed input,
    kept independent so a checker bug can't also corrupt the value grade."""
    let_re, return_re = (_LET_RE, _RETURN_RE) if syntax == "tool" else (_LET_SEXPR_RE, _RETURN_SEXPR_RE)

    env: dict[str, Any] = {}
    statements = _split_statements(program, syntax)
    if not statements:
        raise InterpretError("empty program")

    for statement in statements[:-1]:
        m = let_re.match(statement)
        if not m:
            raise InterpretError(f"expected a let statement, got {statement!r}")
        name, tool_name = m.group(1), m.group(2)
        arg_list = m.group(3) or ""
        if tool_name not in MOCK_REGISTRY.tools:
            raise InterpretError(f"unknown tool {tool_name!r}")
        args = [eval_value(token, env) for token in _split_args(arg_list, syntax)]
        env[name] = MOCK_REGISTRY.execute(tool_name, args)

    m = return_re.match(statements[-1])
    if not m:
        raise InterpretError(f"expected a return statement, got {statements[-1]!r}")
    return eval_value(m.group(1), env), env
