"""High-level API — one function to rule them all."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Optional

from grammars import get_grammar, list_grammars, GRAMMARS

# torch is an optional extra. `llm` and `models` are the transformers backend
# and importing either drags it in, so they are imported where they are used —
# a caller that supplies its own runtime (wirt does) must be able to decode
# without torch installed at all.


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
    step_token_ids: list[int] = field(default_factory=list)
    step_pre_entropies: list[float] = field(default_factory=list)
    step_entropies: list[float] = field(default_factory=list)
    step_retries: list[int] = field(default_factory=list)

def _result(r) -> "Result":
    return Result(
        text=r.text,
        complete=r.is_complete,
        tokens=r.tokens_generated,
        reason=r.stopped_reason,
        exported_context=r.exported_context,
        step_token_ids=r.step_token_ids,
        step_pre_entropies=r.step_pre_entropies,
        step_entropies=r.step_entropies,
        step_retries=r.step_retries,
    )


def _via_runtime(
    runtime,
    model_context,
    *,
    grammar: str,
    aufbau_context: Mapping[str, str],
    max_tokens: int,
    temperature: float,
    seed: int | None,
    constrained: bool = True,
    deadline_seconds: float | None = None,
) -> "Result":
    """Decode against a runtime the caller already has.

    This is how wirt serves: it owns the model and hands it in, so nothing here
    loads weights, chooses a device, or imports torch. `model` is then only a
    label for the result.
    """
    from . import decode

    prompt_ids = runtime.encode_prompt(tuple(model_context))
    if constrained:
        return _result(
            decode.generate(
                runtime,
                grammar=_resolve_grammar(grammar),
                prompt_ids=prompt_ids,
                aufbau_context=aufbau_context,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                deadline_seconds=deadline_seconds,
            )
        )
    return _result(
        decode.generate_unconstrained(
            runtime,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            deadline_seconds=deadline_seconds,
        )
    )



class Session:
    """Reusable constrained generation session."""

    def __init__(self, model_name: str = "gpt2", grammar: str = "stlc", **hf_kwargs):
        from .models import get_model_class

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
            from .environment import ReasoningEnvironment

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
    runtime=None,
    deadline_seconds: float | None = None,
    **kwargs,
) -> Result:
    """Generate against a raw grammar and typing context using a local model."""
    if runtime is not None:
        return _via_runtime(
            runtime, model_context, grammar=grammar, aufbau_context=aufbau_context,
            max_tokens=max_tokens, temperature=temperature, seed=seed, constrained=True,
            deadline_seconds=deadline_seconds,
        )
    if "device" not in kwargs and "device_map" not in kwargs:
        try:
            import torch

            kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
        except Exception:
            kwargs["device_map"] = "cpu"
    from .models import get_model_class

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
        step_token_ids=r.step_token_ids,
        step_pre_entropies=r.step_pre_entropies,
        step_entropies=r.step_entropies,
        step_retries=r.step_retries,
    )


def generate_unconstrained(
    model_context: tuple[tuple[str, str], ...],
    *,
    model: str,
    grammar: str,
    aufbau_context: Mapping[str, str],
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
    runtime=None,
    deadline_seconds: float | None = None,
    **kwargs,
) -> Result:
    """Generate without a grammar mask using the constrained run's prompt setup."""
    if runtime is not None:
        return _via_runtime(
            runtime, model_context, grammar=grammar, aufbau_context=aufbau_context,
            max_tokens=max_tokens, temperature=temperature, seed=seed, constrained=False,
            deadline_seconds=deadline_seconds,
        )
    if "device" not in kwargs and "device_map" not in kwargs:
        try:
            import torch

            kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
        except Exception:
            kwargs["device_map"] = "cpu"
    from .models import get_model_class

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
    r = mdl.generate_unconstrained(
        prompt=prompt,
        max_tokens=max_tokens,
        grammar_name=grammar,
        temperature=temperature,
        seed=seed,
    )
    return Result(
        text=r.text,
        complete=r.is_complete,
        tokens=r.tokens_generated,
        reason=r.stopped_reason,
        exported_context=r.exported_context,
        step_token_ids=r.step_token_ids,
        step_pre_entropies=r.step_pre_entropies,
        step_entropies=r.step_entropies,
        step_retries=r.step_retries,
    )


def generate_pair(
    model_context: tuple[tuple[str, str], ...],
    *,
    model: str,
    grammar: str,
    aufbau_context: Mapping[str, str],
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
    runtime=None,
    **kwargs,
) -> tuple[Result, Result]:
    """Generate masked and unmasked completions from one resident model."""
    if runtime is not None:
        shared = dict(
            grammar=grammar, aufbau_context=aufbau_context,
            max_tokens=max_tokens, temperature=temperature, seed=seed,
        )
        # Same runtime, same prompt, same seed - the only difference between
        # the arms is whether the mask is applied. That is the comparison the
        # playground exists to show, so it must not also differ by model,
        # device or sampling.
        return (
            _via_runtime(runtime, model_context, constrained=True, **shared),
            _via_runtime(runtime, model_context, constrained=False, **shared),
        )
    if "device" not in kwargs and "device_map" not in kwargs:
        try:
            import torch

            kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
        except Exception:
            kwargs["device_map"] = "cpu"
    from .models import get_model_class

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
    constrained = mdl.generate_constrained(
        prompt=prompt, max_tokens=max_tokens, grammar_name=grammar,
        context=dict(aufbau_context), temperature=temperature, seed=seed,
    )
    unconstrained = mdl.generate_unconstrained(
        prompt=prompt, max_tokens=max_tokens, grammar_name=grammar,
        temperature=temperature, seed=seed,
    )
    return (
        Result(
            text=constrained.text, complete=constrained.is_complete,
            tokens=constrained.tokens_generated, reason=constrained.stopped_reason,
            exported_context=constrained.exported_context,
            step_token_ids=constrained.step_token_ids,
            step_pre_entropies=constrained.step_pre_entropies,
            step_entropies=constrained.step_entropies,
            step_retries=constrained.step_retries,
        ),
        Result(
            text=unconstrained.text, complete=unconstrained.is_complete,
            tokens=unconstrained.tokens_generated, reason=unconstrained.stopped_reason,
            exported_context=unconstrained.exported_context,
            step_token_ids=unconstrained.step_token_ids,
            step_pre_entropies=unconstrained.step_pre_entropies,
            step_entropies=unconstrained.step_entropies,
            step_retries=unconstrained.step_retries,
        ),
    )


def verify(
    grammar_source: str,
    text: str,
    *,
    context: Mapping[str, str] | None = None,
):
    """Return aufbau's verification for raw grammar source and candidate text."""
    import aufbau

    synth = aufbau.Synthesizer.from_grammar(aufbau.SPG(grammar_source), "")
    for name, type_source in (context or {}).items():
        synth.add_to_ctx(name, type_source)
    synth.set_input(text)
    return synth.verify()
