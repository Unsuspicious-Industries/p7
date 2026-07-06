"""Benchmark glue for the multi-turn tool-calling agent.

The actual mechanism -- the typed tool registry, grammar generation, and the
turn-by-turn constrained session driver -- lives in `proposition7.agents`
(a real, reusable library, not benchmark-only code: the same `ToolRegistry`/
`AgentSession` back an interactive CLI too, see `proposition7.agents.cli`).
This module is just the benchmark's use of it: a fixed mock registry graded
by executed value (not typedness -- see lmpl-plan.md section 5.2), and a
thin `run_agent_episode` wrapper matching the shape `benchmarks/run.py`'s
job dispatch expects.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Optional

from proposition7.agents import AgentResult, AgentSession

from benchmarks.tool_registry import MOCK_REGISTRY


def _unconstrained_step_generate(model: Any, spg: Any) -> Callable[..., Any]:
    """The unconstrained arm's step call: the model proposes a step with no
    grammar masking at all, and the *same* checker the constrained arm is
    decoded under judges it afterward -- one step is `is_complete` exactly
    when it would have survived masking, so the two arms are graded by the
    identical criterion and differ only in whether the mask was applied
    during decoding (lmpl-plan.md section 4.2)."""
    import aufbau

    def generate(prompt: str, initial: str, max_tokens: int, seed, context: dict[str, str]):
        result = model.generate_unconstrained(
            prompt=prompt, initial=initial, max_tokens=max_tokens, seed=seed
        )
        text = result.text
        synth = aufbau.Synthesizer.from_grammar(spg, "")
        for name, ty in context.items():
            synth.add_to_ctx(name, ty)
        try:
            synth.set_input(text)
            status = synth.status()
        except Exception:
            status = "dead"
        is_complete = status == "typed"
        return SimpleNamespace(
            text=text,
            is_complete=is_complete,
            stopped_reason="complete" if is_complete else status,
            tokens_generated=getattr(result, "tokens_generated", 0),
        )

    return generate


def run_agent_episode(
    model: Any,
    task: str,
    max_turns: int = 6,
    max_tokens_per_step: int = 32,
    think_budget: int = 64,
    seed: Optional[int] = None,
    generate: Optional[Callable[..., Any]] = None,
    think: Optional[Callable[..., Any]] = None,
    syntax: str = "let_c",
    mode: str = "constrained_direct",
) -> AgentResult:
    """Drive one multi-turn episode against the fixed mock registry.
    `generate`/`think` override the constrained-step/unconstrained-think
    calls entirely, for tests that want to inject canned turns without a
    real model (both duck-type `proposition7.GenerationResult`: `.text` /
    `.is_complete` / `.stopped_reason` / `.tokens_generated`). `mode`
    selects the built-in step call when `generate` is not given:
    "constrained_direct" (masked per turn) or "unconstrained" (no masking,
    judged after the fact by the same checker -- see
    `_unconstrained_step_generate`)."""
    session = AgentSession(
        model,
        MOCK_REGISTRY,
        task,
        syntax=syntax,
        max_turns=max_turns,
        think_budget=think_budget,
        step_budget=max_tokens_per_step,
    )
    if generate is None and mode == "unconstrained":
        import aufbau

        generate = _unconstrained_step_generate(model, aufbau.SPG(session.spec))
    return session.run_to_completion(generate=generate, think=think, seed=seed)
