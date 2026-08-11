"""High-level API — one function to rule them all."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Optional

from grammars import get_grammar, list_grammars, GRAMMARS
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
    exported_context: Mapping[str, str] = field(default_factory=dict)


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
    ) -> Result:
        if reason:
            env = ReasoningEnvironment(
                self.model,
                self.grammar,
                think_budget=think_budget,
                formal_budget=max_tokens,
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
            grammar_name=self.grammar,
        )
        return Result(
            text=r.text,
            complete=r.is_complete,
            tokens=r.tokens_generated,
            reason=r.stopped_reason,
        )


def generate(
    model_context: tuple[tuple[str, str], ...],
    *,
    model: str,
    grammar: str,
    aufbau_context: Mapping[str, str],
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
    **kwargs,
) -> Result:
    """Generate against a raw grammar and typing context using a local model."""
    if "device" not in kwargs and "device_map" not in kwargs:
        try:
            import torch

            kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
        except Exception:
            kwargs["device_map"] = "cpu"
    mdl = get_model_class(model).from_pretrained(
        model, grammar=_resolve_grammar(grammar), **kwargs
    )
    template = getattr(mdl.tokenizer, "apply_chat_template", None)
    if template is None:
        raise ValueError(
            f"model {model!r} has no chat template for structured model_context"
        )
    prompt = template(
        [{"role": role, "content": content} for role, content in model_context],
        tokenize=False,
        add_generation_prompt=True,
    )
    r = mdl.generate_constrained(
        prompt=prompt,
        max_tokens=max_tokens,
        grammar_name=grammar,
        context=dict(aufbau_context),
        temperature=temperature,
        seed=seed,
    )
    return Result(
        text=r.text,
        complete=r.is_complete,
        tokens=r.tokens_generated,
        reason=r.stopped_reason,
        exported_context=r.exported_context,
    )
