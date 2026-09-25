"""High-level API, one function to rule them all."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional

if TYPE_CHECKING:
    from .decode import SettlePolicy

from grammars import get_grammar, list_grammars, GRAMMARS

# torch is an optional extra. `llm` and `models` are the transformers backend
# and importing either drags it in, so they are imported where they are used,
# a caller that supplies its own runtime (wirt does) must be able to decode
# without torch installed at all.


def _resolve_grammar(grammar: str) -> str:
    return get_grammar(grammar) if grammar in GRAMMARS else grammar


@dataclass
class Generation:
    """What one generation produced: the text, why it stopped, and the telemetry.

    Named `Generation` and not `Generation` because `Generation` in this tree is the
    Ok/Err sum of `proposition7.types` -- the expected-failure channel. This is
    not that: a generation that stopped early is still a `Generation`, and it
    reports the fact in `complete` and `reason` rather than by being an `Err`.
    """

    text: str
    complete: bool
    tokens: int
    reason: str
    thoughts: str = ""
    """The mixed arm's reasoning, which is not part of the answer.

    Empty for every other arm. Kept beside the answer rather than inside it so
    the same grader can face all four arms without knowing which produced its
    input, while the reasoning stays available to analyse."""
    think_tokens: int = 0
    think_closed: bool = False
    think_forced_close: bool = False
    """Whether the model closed its own reasoning channel before the grammar
    switched on. False means the budget ran out mid-thought, which makes the
    program that follows conditioned on an unfinished sentence -- so it has to
    be visible in the data rather than inferred from a duration."""
    exported_context: Mapping[str, str] = field(default_factory=dict)
    step_token_ids: list[int] = field(default_factory=list)
    step_pre_entropies: list[float] = field(default_factory=list)
    step_entropies: list[float] = field(default_factory=list)
    step_retries: list[int] = field(default_factory=list)
    step_trace: list[dict] = field(default_factory=list)
    """Per-step reconstruction of the decode, empty unless tracing was asked.

    `step_retries` says a position cost 136 attempts; this says which 136 and
    why each was refused -- whitespace, a stop token at an unfinished program,
    or the grammar turning down every spelling of it."""

def _result(r) -> "Generation":
    diagnostics = getattr(r, "diagnostics", None) or {}
    return Generation(
        text=r.text,
        complete=r.is_complete,
        tokens=r.tokens_generated,
        reason=r.stopped_reason,
        thoughts=str(diagnostics.get("think_text", "")),
        think_tokens=int(diagnostics.get("think_tokens", 0)),
        think_closed=bool(diagnostics.get("think_closed", False)),
        think_forced_close=bool(diagnostics.get("think_forced_close", False)),
        exported_context=r.exported_context,
        step_token_ids=r.step_token_ids,
        step_pre_entropies=r.step_pre_entropies,
        step_entropies=r.step_entropies,
        step_retries=r.step_retries,
        step_trace=list(getattr(r, "step_trace", []) or []),
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
    mode: str = "constrained",
    deadline_seconds: float | None = None,
    think_budget: int | None = None,
    trace: bool = False,
    assistant_prefix: str | None = None,
    stop_at_complete: bool = False,
    settle: "SettlePolicy | None" = None,
) -> "Generation":
    """Decode against a runtime the caller already has.

    This is how wirt serves: it owns the model and hands it in, so nothing here
    loads weights, chooses a device, or imports torch. `model` is then only a
    label for the result.
    """
    from . import decode

    # Passed only when asked for. `encode_prompt` is the Runtime protocol's
    # oldest method and implementations predate this argument; sending it
    # unconditionally would break every one of them over a value that means
    # "leave the framing alone". A runtime that cannot honour a prefix the
    # caller did ask for still fails loudly here, which is the right outcome --
    # silently dropping it would change the prompt without changing the result.
    framing = {} if assistant_prefix is None else {"assistant_prefix": assistant_prefix}
    prompt_ids = runtime.encode_prompt(tuple(model_context), **framing)
    # The policy crossed as thresholds; the closure is built here, next to the
    # logits it reads, so the same block boundary happens whether the caller is
    # in this process or on the other end of a request.
    hook = {} if settle is None else {"stop": settle.hook()}
    if mode == "unconstrained":
        return _result(
            decode.generate_unconstrained(
                runtime,
                prompt_ids=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                deadline_seconds=deadline_seconds,
                trace=trace,
                **hook,
            )
        )
    if mode not in ("constrained", "mixed"):
        raise ValueError(f"unknown decode mode {mode!r}")
    extra = {}
    if mode == "mixed" and think_budget is not None:
        extra["think_budget"] = think_budget
    decoder = decode.generate if mode == "constrained" else decode.generate_mixed
    return _result(
        decoder(
            runtime,
            grammar=_resolve_grammar(grammar),
            prompt_ids=prompt_ids,
            aufbau_context=aufbau_context,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            deadline_seconds=deadline_seconds,
            trace=trace,
            stop_at_complete=stop_at_complete,
            **hook,
            **extra,
        )
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
    trace: bool = False,
    assistant_prefix: str | None = None,
    stop_at_complete: bool = False,
    settle: "SettlePolicy | None" = None,
    **kwargs,
) -> Generation:
    """Generate against a raw grammar and typing context using a local model."""
    if runtime is not None:
        return _via_runtime(
            runtime, model_context, grammar=grammar, aufbau_context=aufbau_context,
            max_tokens=max_tokens, temperature=temperature, seed=seed, mode="constrained",
            deadline_seconds=deadline_seconds, trace=trace,
            assistant_prefix=assistant_prefix, stop_at_complete=stop_at_complete,
            settle=settle,
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
    return Generation(
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
    trace: bool = False,
    assistant_prefix: str | None = None,
    settle: "SettlePolicy | None" = None,
    **kwargs,
) -> Generation:
    """Generate without a grammar mask using the constrained run's prompt setup."""
    if runtime is not None:
        return _via_runtime(
            runtime, model_context, grammar=grammar, aufbau_context=aufbau_context,
            max_tokens=max_tokens, temperature=temperature, seed=seed, mode="unconstrained",
            deadline_seconds=deadline_seconds, trace=trace,
            assistant_prefix=assistant_prefix, settle=settle,
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
    return Generation(
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
    deadline_seconds: float | None = None,
    stop_at_complete: bool = False,
    **kwargs,
) -> tuple[Generation, Generation]:
    """Generate masked and unmasked completions from one resident model.

    `deadline_seconds` bounds each arm, not the pair. A caller serialised behind
    a shared slot should size it accordingly: two arms with no bound is how one
    request holds a GPU indefinitely.
    """
    if runtime is not None:
        shared = dict(
            grammar=grammar, aufbau_context=aufbau_context,
            max_tokens=max_tokens, temperature=temperature, seed=seed,
            deadline_seconds=deadline_seconds,
        )
        # Same runtime, same prompt, same seed - the only difference between
        # the arms is whether the mask is applied. That is the comparison the
        # playground exists to show, so it must not also differ by model,
        # device or sampling.
        # `mode=`, not `constrained=`. This call site was missed when the
        # boolean became a three-valued mode ("constrained" / "unconstrained" /
        # "mixed"), and because generate_pair is only reachable through the
        # runtime path that the playground uses, nothing caught it: every live
        # /playground/generate raised TypeError inside a blanket except and
        # came back as a generic 500.
        # `stop_at_complete` reaches only the constrained arm. It asks the
        # grammar whether the program is finished, and the unmasked arm has
        # no grammar to ask.
        return (
            _via_runtime(
                runtime, model_context, mode="constrained",
                stop_at_complete=stop_at_complete, **shared,
            ),
            _via_runtime(runtime, model_context, mode="unconstrained", **shared),
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
        Generation(
            text=constrained.text, complete=constrained.is_complete,
            tokens=constrained.tokens_generated, reason=constrained.stopped_reason,
            exported_context=constrained.exported_context,
            step_token_ids=constrained.step_token_ids,
            step_pre_entropies=constrained.step_pre_entropies,
            step_entropies=constrained.step_entropies,
            step_retries=constrained.step_retries,
        ),
        Generation(
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


def generate_mixed(
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
    think_budget: int | None = None,
    trace: bool = False,
    assistant_prefix: str | None = None,
    stop_at_complete: bool = False,
    **kwargs,
) -> Generation:
    """Reason freely, then emit a program the grammar vouches for.

    Server-side only. The two phases have to share one runtime for the program
    to be written in the context of the reasoning, and the `from_pretrained`
    path below rebuilds its own generation loop per call, so there is nothing
    there to hand a half-finished sequence to. A caller without a runtime is
    asking for something this arm cannot mean, so it is refused rather than
    quietly served as an ordinary constrained decode.
    """
    if runtime is None:
        raise ValueError("mixed decoding needs a runtime; the local model path cannot resume a sequence")
    return _via_runtime(
        runtime, model_context, grammar=grammar, aufbau_context=aufbau_context,
        max_tokens=max_tokens, temperature=temperature, seed=seed, mode="mixed",
        deadline_seconds=deadline_seconds, think_budget=think_budget, trace=trace,
        assistant_prefix=assistant_prefix, stop_at_complete=stop_at_complete,
    )
