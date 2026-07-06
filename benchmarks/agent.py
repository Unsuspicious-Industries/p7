"""Multi-turn tool-calling agent loop, constrained one step at a time.

Unlike the single-shot `c`/`ml`/`stlc`/`tool` grammars — where a whole
program is generated and graded once — this drives a real turn-by-turn
loop: at each turn the model proposes exactly one `let` or `return` step
(constrained against `tool_agent.auf`, whose root is a single AgentStep, not
a whole Program), a mock tool executor "runs" any tool call the step makes,
and the resulting binding is fed into the *next* turn's context via
`generate_constrained(..., context=...)` (`aufbau.Synthesizer.add_to_ctx`
under the hood). So Γ grows across turns exactly like a real agent's
scratchpad, with aufbau checking every individual step against it — a
step can never reference a variable that doesn't exist yet, or hand a tool
an argument of the wrong type, no matter how many turns already happened.

There is no external oracle here (the tool registry is invented, and the
tools are mocked): success means the episode reaches a well-typed `return`
within the turn budget using only well-typed intermediate steps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_AGENT_SPEC_PATH = Path(__file__).resolve().parent.parent / "src" / "grammars" / "tool_agent.auf"

# The same fixed tool registry as tool.auf/tool_sexpr.auf/tool_agent.auf's
# typing rules: (argument type, return type, mock implementation). The mock
# never fails on its own — an episode only fails when a *step* isn't
# well-typed, not because a tool "errored".
MOCK_TOOLS: dict[str, dict[str, Any]] = {
    "search": {
        "arg_type": "string",
        "return_type": "docs",
        "execute": lambda arg: f"docs_about_{arg}",
    },
    "summarize": {
        "arg_type": "docs",
        "return_type": "string",
        "execute": lambda arg: f"summary_of_{arg}",
    },
    "count": {
        "arg_type": "docs",
        "return_type": "int",
        "execute": lambda arg: 3,
    },
    "format": {
        "arg_type": "int",
        "return_type": "string",
        "execute": lambda arg: f"formatted_{arg}",
    },
}

_LET_RE = re.compile(r"^\s*let\s+(\w+)\s*=\s*(\w+)\s*\(\s*(.*?)\s*\)\s*;\s*$")
_RETURN_RE = re.compile(r"^\s*return\s+(.*?)\s*;\s*$")


def _agent_spec() -> str:
    return _AGENT_SPEC_PATH.read_text(encoding="utf-8")


def _build_step_prompt(task: str, context: dict[str, str], turn: int, max_turns: int) -> str:
    tools = "\n".join(
        f"  {name}({info['arg_type']}) -> {info['return_type']}"
        for name, info in MOCK_TOOLS.items()
    )
    bindings = (
        ", ".join(f"{name}: {ty}" for name, ty in context.items()) or "(none yet)"
    )
    return (
        f"Task: {task}\n"
        f"Turn {turn + 1} of {max_turns} max.\n"
        f"Available tools:\n{tools}\n"
        f"Variables already bound: {bindings}\n"
        "Produce exactly one step: either `let <name> = <tool>(<arg>);` calling one "
        "tool with a value or an already-bound variable of the right type, or "
        "`return <value>;` to finish."
    )


@dataclass
class AgentStep:
    turn: int
    text: str
    is_complete: bool
    stopped_reason: str
    kind: str = ""  # "let" | "return" | "invalid"
    tool: str = ""
    bound_name: str = ""
    bound_type: str = ""


@dataclass
class AgentResult:
    success: bool
    reason: str
    steps: list[AgentStep] = field(default_factory=list)
    final_context: dict[str, str] = field(default_factory=dict)
    return_value: Optional[str] = None


def run_agent_episode(
    model: Any,
    task: str,
    max_turns: int = 6,
    max_tokens_per_step: int = 32,
    seed: Optional[int] = None,
    generate: Optional[Callable[..., Any]] = None,
) -> AgentResult:
    """Drive one multi-turn episode. `model` must expose
    `generate_constrained(prompt, initial, max_tokens, grammar_name, seed,
    context)` returning an object with `.text`/`.is_complete`/
    `.stopped_reason` (duck-typed: `proposition7.GenerationResult` or a test
    double both work). `generate` overrides the call entirely, for tests
    that want to inject canned steps without a real model."""
    spec = _agent_spec()
    context: dict[str, str] = {}
    steps: list[AgentStep] = []
    call = generate or (
        lambda **kwargs: model.generate_constrained(grammar_name=spec, **kwargs)
    )

    for turn in range(max_turns):
        prompt = _build_step_prompt(task, context, turn, max_turns)
        result = call(
            prompt=prompt,
            initial="",
            max_tokens=max_tokens_per_step,
            seed=seed,
            context=dict(context),
        )
        text = str(getattr(result, "text", ""))
        is_complete = bool(getattr(result, "is_complete", False))
        stopped_reason = str(getattr(result, "stopped_reason", ""))
        step = AgentStep(turn=turn, text=text, is_complete=is_complete, stopped_reason=stopped_reason)
        steps.append(step)

        if not is_complete:
            return AgentResult(
                success=False,
                reason=f"turn_{turn}_incomplete_step:{stopped_reason}",
                steps=steps,
                final_context=context,
            )

        if m := _RETURN_RE.match(text):
            step.kind = "return"
            return AgentResult(
                success=True,
                reason="",
                steps=steps,
                final_context=context,
                return_value=m.group(1),
            )

        m = _LET_RE.match(text)
        if not m:
            return AgentResult(
                success=False,
                reason=f"turn_{turn}_unparseable_step:{text!r}",
                steps=steps,
                final_context=context,
            )

        name, tool_name, _arg = m.group(1), m.group(2), m.group(3)
        tool = MOCK_TOOLS.get(tool_name)
        if tool is None:
            # aufbau already rejected any tool name outside the registry
            # (the grammar's Call alternatives are the fixed tool set), so
            # this is unreachable for a genuinely "typed" step — guarded
            # here only so a mocked/hand-crafted step can't crash the loop.
            return AgentResult(
                success=False,
                reason=f"turn_{turn}_unknown_tool:{tool_name}",
                steps=steps,
                final_context=context,
            )
        step.kind = "let"
        step.tool = tool_name
        step.bound_name = name
        step.bound_type = tool["return_type"]
        context[name] = tool["return_type"]

    return AgentResult(
        success=False,
        reason=f"max_turns_exceeded:{max_turns}",
        steps=steps,
        final_context=context,
    )
