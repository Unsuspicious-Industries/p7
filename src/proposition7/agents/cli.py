"""Interactive CLI for proposition7.agents.

A real multi-turn, grammar-constrained tool-calling session driven by an
actual HF model, using real (not mocked) tools -- this is the proof that
`ToolRegistry`/`AgentSession` are a genuine, reusable library and not
benchmark-only glue: the exact same two classes back
`benchmarks/agent.py`'s episode grading.

A TUI (or any other frontend) can reuse `AgentSession` the same way: build a
registry, construct a session, call `.step()` in whatever event loop the
frontend runs, and render `AgentTurn`/`AgentResult` however it likes -- this
CLI's `while not session.done: session.step()` loop is the whole interface.

Usage:
    python -m proposition7.agents.cli --model <hf-model-name>
"""

from __future__ import annotations

import argparse
import time

from ..llm import ConstrainedModel
from .registry import Param, Tool, ToolRegistry
from .session import AgentSession


def default_tools() -> ToolRegistry:
    """A small set of genuinely real (not mocked) tools: real arithmetic,
    real string transforms, real wall-clock time. Deliberately simple and
    safe -- proving reusability doesn't require a dangerous tool surface."""
    return ToolRegistry(
        [
            Tool(
                "add",
                (Param("a", "int"), Param("b", "int")),
                "int",
                lambda a, b: a + b,
                "add two integers",
            ),
            Tool.unary("upper", "string", "string", lambda s: s.upper(), "uppercase a string"),
            Tool.unary("length", "string", "int", lambda s: len(s), "length of a string"),
            Tool.nullary("now", "int", lambda: int(time.time()), "current unix timestamp"),
        ]
    )


def _print_turn(turn) -> None:
    if turn.think_text:
        print(f"  [think] {turn.think_text}")
    if turn.kind == "let":
        print(f"  [step]  {turn.text}  =>  {turn.tool}(...) = {turn.bound_value!r}")
    elif turn.kind == "return":
        print(f"  [step]  {turn.text}")
    else:
        print(f"  [step]  {turn.text!r}  ({turn.stopped_reason})")


def run_repl(model, registry: ToolRegistry, *, syntax: str = "let_c", max_turns: int = 6) -> None:
    print("Type a task; Ctrl-D or 'quit' to exit.")
    while True:
        try:
            task = input("\ntask> ").strip()
        except EOFError:
            print()
            break
        if not task or task.lower() in {"quit", "exit"}:
            break

        session = AgentSession(model, registry, task, syntax=syntax, max_turns=max_turns)
        while not session.done:
            _print_turn(session.step())

        result = session.result
        if result.success:
            print(f"=> {result.return_value!r}")
        else:
            print(f"episode failed: {result.reason}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive grammar-constrained tool-calling agent (proposition7.agents)"
    )
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--syntax", default="let_c", choices=["let_c", "sexpr"])
    parser.add_argument("--max-turns", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    registry = default_tools()
    model = ConstrainedModel.from_pretrained(args.model, grammar="", device=args.device)
    run_repl(model, registry, syntax=args.syntax, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
