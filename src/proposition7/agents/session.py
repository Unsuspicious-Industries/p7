"""A real, reusable multi-turn agentic session driver.

Each turn: a short unconstrained "think" block (why this tool, what
argument -- the same think-then-formal protocol `ReasoningEnvironment` uses
once per program, here applied once per turn), then one grammar-constrained
step against the registry's single-step grammar (`ToolRegistry.spec(root=
"step", ...)`), executed against the registry's real tool implementations.
The growing type context (Γ) is threaded turn to turn via
`aufbau.Synthesizer.add_to_ctx` (through `ConstrainedModel.generate_constrained
(..., context=...)`); a parallel value environment is threaded here so a
completed episode can be graded on the *value* it actually returns, not
merely on having reached a well-typed `return` (see benchmarks/agent.py's
episode grading, and lmpl-plan.md section 5.2).

This module has no benchmark-specific code in it: it is meant to back a
benchmark harness, an interactive CLI, or (via the same step-by-step
`AgentSession.step()` call) a TUI that wants to show/pause between turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .registry import ToolRegistry

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_LET_RE = re.compile(r"^\s*let\s+(\w+)\s*=\s*(\w+)\s*\(\s*(.*?)\s*\)\s*;\s*$")
_RETURN_RE = re.compile(r"^\s*return\s+(.*?)\s*;\s*$")
_LET_SEXPR_RE = re.compile(r"^\s*\(\s*let\s+(\w+)\s*\(\s*(\w+)(?:\s+(.*?))?\s*\)\s*\)\s*$")
_RETURN_SEXPR_RE = re.compile(r"^\s*\(\s*return\s+(.*?)\s*\)\s*$")

_STRING_RE = re.compile(r'^"([a-zA-Z0-9_]*)"$')
_INT_RE = re.compile(r"^[0-9]+$")


def _eval_value(token: str, env: dict[str, Any]) -> Any:
    if m := _STRING_RE.match(token):
        return m.group(1)
    if _INT_RE.match(token):
        return int(token)
    if token in env:
        return env[token]
    raise ValueError(f"unbound variable {token!r}")


def _split_args(arg_list: str, syntax: str) -> list[str]:
    """Split a call's argument-list interior into individual Value tokens.
    Safe as a naive split: no Value alternative (Variable/StringLit/IntLit)
    can itself contain a comma or embedded whitespace."""
    if not arg_list.strip():
        return []
    if syntax == "let_c":
        return [part.strip() for part in arg_list.split(",")]
    return arg_list.split()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [item for item in items if item and not (item in seen or seen.add(item))]


@dataclass
class AgentTurn:
    turn: int
    think_text: str = ""
    think_tokens: int = 0
    text: str = ""
    is_complete: bool = False
    stopped_reason: str = ""
    kind: str = ""  # "let" | "return" | "invalid"
    tool: str = ""
    bound_name: str = ""
    bound_type: str = ""
    bound_value: Any = None


@dataclass
class AgentResult:
    success: bool
    reason: str
    turns: list[AgentTurn] = field(default_factory=list)
    final_context: dict[str, str] = field(default_factory=dict)
    final_values: dict[str, Any] = field(default_factory=dict)
    return_expr: Optional[str] = None
    return_value: Any = None


class AgentSession:
    """Drives one grammar-constrained multi-turn tool-calling episode, one
    turn at a time. Call `.step()` repeatedly -- an interactive CLI or TUI
    can inspect state and pause between turns -- or `.run_to_completion()`
    for a benchmark that just wants the final result."""

    def __init__(
        self,
        model: Any,
        registry: ToolRegistry,
        task: str,
        *,
        syntax: str = "let_c",
        max_turns: int = 6,
        think_budget: int = 64,
        step_budget: int = 32,
        prompt_builder: Optional[Callable[[str, dict[str, str], int, int], str]] = None,
    ):
        self.model = model
        self.registry = registry
        self.task = task
        self.syntax = syntax
        self.max_turns = max_turns
        self.think_budget = think_budget
        self.step_budget = step_budget
        self.prompt_builder = prompt_builder or self._default_prompt

        self.spec = registry.spec(root="step", syntax=syntax)
        self.type_context: dict[str, str] = {}
        self.value_context: dict[str, Any] = {}
        self.turns: list[AgentTurn] = []
        self.done = False
        self.result: Optional[AgentResult] = None

    def _default_prompt(self, task: str, context: dict[str, str], turn: int, max_turns: int) -> str:
        def signature(tool: Any) -> str:
            params = ", ".join(f"{p.name}: {p.type}" for p in tool.params)
            return f"  {tool.name}({params}) -> {tool.return_type}"

        tools = "\n".join(signature(tool) for tool in self.registry.tools.values())
        bindings = ", ".join(f"{name}: {ty}" for name, ty in context.items()) or "(none yet)"
        call_shape = (
            "`let <name> = <tool>(<args>);` calling one tool, or `return <value>;` to finish"
            if self.syntax == "let_c"
            else "`(let <name> (<tool> <args>))` calling one tool, or `(return <value>)` to finish"
        )
        return (
            f"Task: {task}\n"
            f"Turn {turn + 1} of {max_turns} max.\n"
            f"Available tools:\n{tools}\n"
            f"Variables already bound: {bindings}\n"
            f"Think briefly, then produce exactly one step: {call_shape}"
        )

    def _think(self, prompt: str, think: Optional[Callable[..., Any]]) -> tuple[str, int]:
        if think is not None:
            result = think(prompt=prompt, max_tokens=self.think_budget)
        else:
            stop_tokens = _dedupe(self.model.stop_tokens_unconstrained(self.spec) + [_THINK_CLOSE])
            result = self.model.generate_unconstrained(
                prompt=f"{prompt}\n{_THINK_OPEN}",
                initial="",
                max_tokens=self.think_budget,
                temperature=0.0,
                stop_tokens=stop_tokens,
                grammar_name=self.spec,
            )
        content = str(getattr(result, "text", ""))
        if content.startswith(_THINK_OPEN):
            content = content[len(_THINK_OPEN) :]
        if _THINK_CLOSE in content:
            content = content[: content.find(_THINK_CLOSE)]
        return content, int(getattr(result, "tokens_generated", 0))

    def step(
        self,
        *,
        think: Optional[Callable[..., Any]] = None,
        generate: Optional[Callable[..., Any]] = None,
        seed: Optional[int] = None,
    ) -> AgentTurn:
        """Advance exactly one turn. Sets `self.done`/`self.result` when the
        episode ends (a `return`, a failed step, or the turn budget)."""
        if self.done:
            raise RuntimeError("session already finished; check .done before calling .step() again")

        turn_index = len(self.turns)
        prompt = self.prompt_builder(self.task, self.type_context, turn_index, self.max_turns)
        think_text, think_tokens = self._think(prompt, think)

        call = generate or (
            lambda **kwargs: self.model.generate_constrained(grammar_name=self.spec, **kwargs)
        )
        result = call(
            prompt=f"{prompt}\n{_THINK_OPEN}{think_text}{_THINK_CLOSE}",
            initial="",
            max_tokens=self.step_budget,
            seed=seed,
            context=dict(self.type_context),
        )
        text = str(getattr(result, "text", ""))
        is_complete = bool(getattr(result, "is_complete", False))
        stopped_reason = str(getattr(result, "stopped_reason", ""))
        turn = AgentTurn(
            turn=turn_index,
            think_text=think_text,
            think_tokens=think_tokens,
            text=text,
            is_complete=is_complete,
            stopped_reason=stopped_reason,
        )
        self.turns.append(turn)

        if not is_complete:
            self._finish(False, f"turn_{turn_index}_incomplete_step:{stopped_reason}")
            return turn

        let_re, return_re = (
            (_LET_RE, _RETURN_RE) if self.syntax == "let_c" else (_LET_SEXPR_RE, _RETURN_SEXPR_RE)
        )

        if m := return_re.match(text):
            turn.kind = "return"
            expr = m.group(1)
            try:
                value = _eval_value(expr, self.value_context)
            except ValueError as error:
                self._finish(False, f"turn_{turn_index}_unevaluable_return:{error}")
                return turn
            self._finish(True, "", return_expr=expr, return_value=value)
            return turn

        m = let_re.match(text)
        if not m:
            self._finish(False, f"turn_{turn_index}_unparseable_step:{text!r}")
            return turn

        name, tool_name = m.group(1), m.group(2)
        arg_list = m.group(3) or ""
        if tool_name not in self.registry.tools:
            # aufbau already rejected any tool name outside the registry (the
            # grammar's Call alternatives are the fixed tool set); guarded
            # here only so a hand-crafted test step can't crash the loop.
            self._finish(False, f"turn_{turn_index}_unknown_tool:{tool_name}")
            return turn

        try:
            args = [
                _eval_value(token, self.value_context)
                for token in _split_args(arg_list, self.syntax)
            ]
        except ValueError as error:
            self._finish(False, f"turn_{turn_index}_unevaluable_arg:{error}")
            return turn

        tool = self.registry.tools[tool_name]
        value = self.registry.execute(tool_name, args)
        turn.kind = "let"
        turn.tool = tool_name
        turn.bound_name = name
        turn.bound_type = tool.return_type
        turn.bound_value = value
        self.type_context[name] = tool.return_type
        self.value_context[name] = value

        if turn_index + 1 >= self.max_turns:
            self._finish(False, f"max_turns_exceeded:{self.max_turns}")
        return turn

    def _finish(
        self,
        success: bool,
        reason: str,
        *,
        return_expr: Optional[str] = None,
        return_value: Any = None,
    ) -> None:
        self.done = True
        self.result = AgentResult(
            success=success,
            reason=reason,
            turns=list(self.turns),
            final_context=dict(self.type_context),
            final_values=dict(self.value_context),
            return_expr=return_expr,
            return_value=return_value,
        )

    def run_to_completion(self, **kwargs: Any) -> AgentResult:
        while not self.done:
            self.step(**kwargs)
        assert self.result is not None
        return self.result
