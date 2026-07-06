"""A typed tool registry, and the `.auf` grammar it generates.

This is the reusable core the rest of `proposition7.agents` builds on: given
a small set of tools -- each with an ordered list of named, typed
parameters and a return type -- generate a grammar (in either of two
concrete syntaxes) that constrains an LLM to only ever emit well-typed
calls against exactly that registry. No `.auf` text to hand-write, no drift
between "what the mock executes" and "what the checker accepts" (they come
from the same `Tool` objects).

Three independent axes:
  - Each tool's arity is fixed but arbitrary (0, 1, 2, ... parameters) --
    unlike a user-definable function (c.auf's genuinely unbounded arity,
    which needs a cons-list encoding), a *registered* tool's signature is
    known at grammar-generation time, so each parameter gets its own
    grammar slot: no cons-list machinery needed.
  - `root`: "program" (a whole `let`-chain ending in `return`, for one-shot
    generation) or "step" (a single `LetStmt | ReturnStmt`, for driving one
    turn of a multi-turn AgentSession).
  - `syntax`: "let_c" (`let x = f(a, b);`) or "sexpr" (`(let x (f a b))`).

The generated grammar's shape mirrors (and, for arity 1, is verified
against) the hand-written tool.auf/tool_sexpr.auf/tool_agent.auf this
replaces -- see tests/agents_registry.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Param:
    name: str
    type: str


@dataclass(frozen=True)
class Tool:
    """One entry in a typed tool registry: a name, an ordered list of typed
    parameters, a return type, and the callable that actually runs it (a
    mock in a benchmark, a real action in a CLI -- the registry doesn't
    care). `execute` is called positionally: `execute(*args)`."""

    name: str
    params: tuple[Param, ...]
    return_type: str
    execute: Callable[..., Any]
    description: str = ""

    @staticmethod
    def unary(
        name: str,
        arg_type: str,
        return_type: str,
        execute: Callable[[Any], Any],
        description: str = "",
    ) -> "Tool":
        """Convenience constructor for the common single-argument case."""
        return Tool(name, (Param("arg", arg_type),), return_type, execute, description)

    @staticmethod
    def nullary(
        name: str,
        return_type: str,
        execute: Callable[[], Any],
        description: str = "",
    ) -> "Tool":
        """Convenience constructor for a zero-argument tool (e.g. a clock)."""
        return Tool(name, (), return_type, execute, description)


class ToolRegistry:
    """A fixed set of typed tools plus the grammar(s) they generate.
    Building the grammar from the registry (not the other way around) is
    what keeps the mock/real executor and the type checker from ever
    describing two different tool sets."""

    def __init__(self, tools: dict[str, Tool] | list[Tool]):
        if isinstance(tools, dict):
            self.tools = dict(tools)
        else:
            self.tools = {tool.name: tool for tool in tools}
        for name, tool in self.tools.items():
            if name != tool.name:
                raise ValueError(f"registry key {name!r} does not match Tool.name {tool.name!r}")

    def execute(self, name: str, args: list[Any]) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool {name!r}")
        if len(args) != len(tool.params):
            raise ValueError(f"{name} takes {len(tool.params)} argument(s), got {len(args)}")
        return tool.execute(*args)

    def spec(self, root: str = "program", syntax: str = "let_c") -> str:
        """Generate the complete `.auf` grammar source for this registry."""
        if syntax == "let_c":
            return _spec_let_c(self.tools, root)
        if syntax == "sexpr":
            return _spec_sexpr(self.tools, root)
        raise ValueError(f"unknown tool DSL syntax: {syntax!r}")


def _tool_label(name: str) -> str:
    return f"{name}_call"


def _call_alternatives(tools: dict[str, Tool]) -> list[str]:
    return [f"{name.capitalize()}Call" for name in tools]


def _typing_rule(name: str, tool: Tool) -> str:
    if not tool.params:
        return f"----------- ({_tool_label(name)})\n'{tool.return_type}'"
    premises = ", ".join(f"Γ ⊢ arg{i} : '{p.type}'" for i, p in enumerate(tool.params))
    return f"{premises}\n-------------------- ({_tool_label(name)})\n'{tool.return_type}'"


def _root_production(root: str) -> str:
    return (
        "Program ::= LetStmt Program | ReturnStmt"
        if root == "program"
        else "AgentStep ::= LetStmt | ReturnStmt"
    )


def _spec_let_c(tools: dict[str, Tool], root: str) -> str:
    def call_production(name: str, tool: Tool) -> str:
        if not tool.params:
            return f"{name.capitalize()}Call({_tool_label(name)}) ::= '{name}' '(' ')'"
        slots = " ',' ".join(f"Value[arg{i}]" for i in range(len(tool.params)))
        return f"{name.capitalize()}Call({_tool_label(name)}) ::= '{name}' '(' {slots} ')'"

    call_productions = "\n".join(call_production(name, tool) for name, tool in tools.items())
    call_alt = " | ".join(_call_alternatives(tools))
    typing_rules = "\n\n".join(_typing_rule(name, tool) for name, tool in tools.items())
    return f"""// Generated tool-calling grammar (let_c syntax) -- see proposition7.agents.registry.
// `let name = tool(arg0, arg1, ...);` bindings, ending in `return value;`.

Identifier ::= /[a-z_][a-z0-9_]*/
StringLit(str_lit) ::= /"[a-zA-Z0-9_]*"/
IntLit(int_lit) ::= /[0-9]+/

Variable(var) ::= Identifier[x]
Value ::= Variable | StringLit | IntLit

{call_productions}

Call ::= {call_alt}

LetStmt(let) ::= 'let' Identifier[name] '=' Call[value] ';'
ReturnStmt(return) ::= 'return' Value[e] ';'

{_root_production(root)}

// ===================== Typing rules =====================

x ∈ Γ
----------- (var)
Γ(x)

----------- (str_lit)
'string'

----------- (int_lit)
'int'

{typing_rules}

Γ ⊢ value : ?τ
------------------------- (let)
Γ → Γ[name:?τ] ⊢ 'void'

Γ ⊢ e : ?A
----------- (return)
'void'
"""


def _spec_sexpr(tools: dict[str, Tool], root: str) -> str:
    def call_production(name: str, tool: Tool) -> str:
        if not tool.params:
            return f"{name.capitalize()}Call({_tool_label(name)}) ::= '(' '{name}' ')'"
        slots = " ".join(f"Value[arg{i}]" for i in range(len(tool.params)))
        return f"{name.capitalize()}Call({_tool_label(name)}) ::= '(' '{name}' {slots} ')'"

    call_productions = "\n".join(call_production(name, tool) for name, tool in tools.items())
    call_alt = " | ".join(_call_alternatives(tools))
    typing_rules = "\n\n".join(_typing_rule(name, tool) for name, tool in tools.items())
    return f"""// Generated tool-calling grammar (S-expression syntax) -- see proposition7.agents.registry.
// `(let name (tool arg0 arg1 ...))` bindings, ending in `(return value)`.

Identifier ::= /[a-z_][a-z0-9_]*/
StringLit(str_lit) ::= /"[a-zA-Z0-9_]*"/
IntLit(int_lit) ::= /[0-9]+/

Variable(var) ::= Identifier[x]
Value ::= Variable | StringLit | IntLit

{call_productions}

Call ::= {call_alt}

LetStmt(let) ::= '(' 'let' Identifier[name] Call[value] ')'
ReturnStmt(return) ::= '(' 'return' Value[e] ')'

{_root_production(root)}

// ===================== Typing rules =====================

x ∈ Γ
----------- (var)
Γ(x)

----------- (str_lit)
'string'

----------- (int_lit)
'int'

{typing_rules}

Γ ⊢ value : ?τ
------------------------- (let)
Γ → Γ[name:?τ] ⊢ 'void'

Γ ⊢ e : ?A
----------- (return)
'void'
"""
