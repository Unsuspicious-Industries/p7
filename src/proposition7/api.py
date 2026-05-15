"""High-level API — one function to rule them all."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .grammars import get_grammar, GRAMMARS
from .llm import ConstrainedModel
from .models import get_model_class
from .environment import ReasoningEnvironment


def _resolve_grammar(grammar: str) -> str:
    return get_grammar(grammar) if grammar in GRAMMARS else grammar


@dataclass
class Result:
    text: str
    complete: bool
    tokens: int
    reason: str
    thoughts: str = ""


class Session:
    """Reusable constrained generation session."""

    def __init__(self, model_name: str = "gpt2", grammar: str = "stlc", **hf_kwargs):
        cls = get_model_class(model_name)
        if "device" not in hf_kwargs and "device_map" not in hf_kwargs:
            try:
                import torch

                hf_kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
            except Exception:
                hf_kwargs["device_map"] = "cpu"
        self.model = cls.from_pretrained(
            model_name, grammar=_resolve_grammar(grammar), **hf_kwargs
        )
        self.grammar = grammar

    def generate(
        self,
        prompt: str,
        *,
        initial: str = "",
        max_tokens: int = 50,
        reason: bool = False,
        think_budget: int = 200,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> Result:
        if reason:
            env = ReasoningEnvironment(
                self.model,
                self.grammar,
                think_budget=think_budget,
                formal_budget=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            r = env.generate(prompt, initial=initial)
            out = r.final_output
            return Result(
                text=out.content if out else "",
                complete=r.is_complete,
                tokens=r.total_tokens,
                reason=r.stopped_reason,
                thoughts=r.all_thoughts,
            )
        r = self.model.generate_constrained(
            prompt=prompt,
            initial=initial,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        return Result(
            text=r.text,
            complete=r.is_complete,
            tokens=r.tokens_generated,
            reason=r.stopped_reason,
        )


def generate(
    prompt: str,
    *,
    model: str = "gpt2",
    grammar: str = "stlc",
    initial: str = "",
    max_tokens: int = 50,
    reason: bool = False,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    **kwargs,
) -> Result:
    """Generate typed, constrained output from a prompt.

    Args:
        prompt: What to generate.
        model: HuggingFace model id.
        grammar: Built-in grammar name or raw spec string.
        initial: Partial output to continue.
        max_tokens: Max tokens to generate.
        reason: Enable chain-of-thought before constrained output.
        remote: Deprecated remote path.
        gpu: Reserved for remote container launchers.
    """
    think_budget = kwargs.pop("think_budget", 200)
    if "device" not in kwargs and "device_map" not in kwargs:
        try:
            import torch

            kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
        except Exception:
            kwargs["device_map"] = "cpu"
    mdl = get_model_class(model).from_pretrained(
        model, grammar=_resolve_grammar(grammar), **kwargs
    )
    if reason:
        env = ReasoningEnvironment(
            mdl,
            grammar,
            think_budget=think_budget,
            formal_budget=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        r = env.generate(prompt, initial=initial)
        out = r.final_output
        return Result(
            text=out.content if out else "",
            complete=r.is_complete,
            tokens=r.total_tokens,
            reason=r.stopped_reason,
            thoughts=r.all_thoughts,
        )
    r = mdl.generate_constrained(
        prompt=prompt,
        initial=initial,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    return Result(
        text=r.text,
        complete=r.is_complete,
        tokens=r.tokens_generated,
        reason=r.stopped_reason,
    )
