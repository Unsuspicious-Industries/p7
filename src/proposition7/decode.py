"""The masked decode loop: logits in, a well-typed program out.

This is proposition7's whole contribution, and it is deliberately small. The
model is behind `Runtime` (see `runtime.py`) and the grammar is behind aufbau's
`Synthesizer`; what lives here is only the part that belongs to neither — which
token to try next, what to do when the grammar refuses it, and what to record
about the choice.

No torch, and no tensors of any framework: logits arrive as a numpy array, so
the same loop drives transformers locally and tinygrad on a server. That is not
a portability nicety. It is what lets `wirt` own inference without owning any of
the reasoning below.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .inference import GenerationResult
from .mask_cache import DEFAULT_MASK_CACHE_SIZE, mask_candidates
from .runtime import Runtime

MAX_RETRIES = 2048


def _entropy_bits(logits: np.ndarray) -> float:
    """Shannon entropy of the distribution, in bits.

    H = -∑ p_i · log₂(p_i)   where p = softmax(logits)

    `-inf` entries (grammar-invalid, or excluded by an earlier retry) get p=0
    and are dropped before the log, which would otherwise be log₂(0). A fully
    masked or single-token distribution is 0.0.
    """
    finite = np.isfinite(logits)
    if not finite.any():
        return 0.0
    values = logits[finite].astype(np.float64)
    values -= values.max()                       # stabilise before exp
    weights = np.exp(values)
    probabilities = weights / weights.sum()
    # Clamp guards log₂(0) from underflow, exactly as the torch version did.
    return float(-(probabilities * np.log2(np.maximum(probabilities, 1e-40))).sum())


def _draw(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """One token from the masked distribution. Greedy at temperature 0."""
    if temperature == 0.0:
        return int(np.argmax(logits))
    finite = np.isfinite(logits)
    values = logits.astype(np.float64) / max(temperature, 1e-6)
    values[~finite] = -np.inf
    values -= values[finite].max()
    weights = np.exp(values)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return -1
    return int(rng.choice(len(logits), p=weights / total))


def _spellings(token: str, prefix: str) -> list[str]:
    """The candidate spellings of one token, in priority order.

    LM tokenizers prefix most tokens with a space (' x', ' 3'). That space is
    meaningful at a keyword or identifier boundary ('let' + ' x' → 'let x') and
    fatal inside a regex terminal ('4' + ' 3' must mean '43', not '4 3'). So a
    token is offered several ways and the first the grammar admits wins:

        raw          preserves the model's own spacing
        lstripped    digit/operator continuation
        space-joined bare tokenizers with no leading space, where a grammar
                     token boundary is still needed ('x' then 'y' → 'x y')

    Order matters: the space-join is last so it can never override a spelling
    the model actually emitted.
    """
    candidates: list[str] = []
    for candidate in (token, token.lstrip()):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    stripped = token.lstrip()
    if stripped and prefix and not prefix[-1].isspace() and not token[:1].isspace():
        joined = " " + stripped
        if joined not in candidates:
            candidates.append(joined)
    return candidates


@dataclass
class Step:
    """One decoded position. `content` is the spelling the grammar accepted."""

    token_id: int | None = None
    token: str | None = None
    content: str | None = None
    is_stop: bool = False
    pre_entropy: float = 0.0
    post_entropy: float = 0.0
    retries: int = 0


def sample(
    runtime: Runtime,
    synth: Any,
    logits: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float = 0.0,
    grammar_hash: bytes = b"",
    mask_cache_size: int = DEFAULT_MASK_CACHE_SIZE,
) -> Step:
    """Draw a token the grammar will accept, retrying past the ones it will not.

    `synth` holds the accepted prefix and is only *read* here — `mask` is
    state-free — so a rejected candidate costs nothing to undo. That is also
    why the runtime is never told about a rejection: nothing was committed to
    it either.

    `pre_entropy` is computed once, before the retry loop, so it measures the
    model's own uncertainty rather than how many attempts this position took.
    `post_entropy` is measured at acceptance, over grammar-valid tokens minus
    those already excluded — a lower bound on the true grammar-valid entropy
    whenever there were retries.
    """
    valid = np.isfinite(logits)
    step = Step(pre_entropy=_entropy_bits(logits))

    for _ in range(MAX_RETRIES):
        if not valid.any():
            return step
        masked = np.where(valid, logits, -np.inf)
        token_id = _draw(masked, temperature, rng)
        if token_id < 0:
            return step

        token = runtime.token_text(token_id)
        if not token or not token.strip():
            valid[token_id] = False
            step.retries += 1
            continue

        # Stop tokens are checked BEFORE the grammar, because they are not
        # grammar tokens and masking would forbid the model from ever ending
        # its turn. But ending a turn is only a legal move once the program is
        # actually finished: at an incomplete prefix, stopping produces exactly
        # the ill-formed output this whole loop exists to make unemittable, so
        # it is refused like any other inadmissible token.
        #
        # This is not hypothetical. A reasoning model wants to open with
        # `<think>`, the grammar refuses it, and the next thing it reaches for
        # is end-of-turn — so honouring that unconditionally returns an empty
        # program on the first step.
        if runtime.is_stop(token_id):
            if synth.status() == "typed":
                step.token_id, step.token, step.is_stop = token_id, token, True
                step.post_entropy = _entropy_bits(masked)
                return step
            valid[token_id] = False
            step.retries += 1
            continue

        candidates = _spellings(token, synth.input())
        admitted = mask_candidates(synth, grammar_hash, candidates, mask_cache_size)
        content = next((c for c, ok in zip(candidates, admitted) if ok), None)
        if content is None:
            valid[token_id] = False
            step.retries += 1
            continue

        step.token_id, step.token, step.content = token_id, token, content
        step.post_entropy = _entropy_bits(masked)
        return step

    return step


def generate(
    runtime: Runtime,
    *,
    grammar: str,
    prompt_ids: Sequence[int],
    aufbau_context: Mapping[str, str] | None = None,
    initial: str = "",
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
    mask_cache_size: int = DEFAULT_MASK_CACHE_SIZE,
) -> GenerationResult:
    """Decode under `grammar`, returning the text aufbau itself validated."""
    import aufbau

    rng = np.random.default_rng(seed)
    grammar_hash = hashlib.blake2b(grammar.encode(), digest_size=16).digest()
    synth = aufbau.Synthesizer(grammar, "")

    # A multi-turn caller pre-populates Γ with bindings from prior turns, so a
    # step can reference an earlier result and be typechecked against it
    # without re-parsing all the preceding text.
    for name, type_source in (aufbau_context or {}).items():
        synth.add_to_ctx(name, type_source)

    if initial:
        synth.set_input(initial)
        if synth.status() == "dead":
            return GenerationResult(
                initial, False, 0, "type_error: initial text is not a live prefix"
            )

    runtime.reset(list(prompt_ids))

    generated = 0
    reason = "max_tokens"
    token_ids: list[int] = []
    pre_entropies: list[float] = []
    entropies: list[float] = []
    retries: list[int] = []

    for _ in range(max_tokens):
        try:
            step = sample(
                runtime,
                synth,
                runtime.logits(),
                rng,
                temperature=temperature,
                grammar_hash=grammar_hash,
                mask_cache_size=mask_cache_size,
            )
        except Exception as error:  # noqa: BLE001 - becomes the stop reason
            reason = f"type_error: {error}"
            break

        if step.token is None:
            # Nothing admissible. At a complete program that is success; mid
            # prefix it means the model's lattice and the grammar diverged.
            reason = "complete" if synth.status() == "typed" else "no_valid"
            break
        if step.is_stop:
            reason = (
                "complete" if synth.status() == "typed" else f"stop_token:{step.token}"
            )
            break

        try:
            synth.feed(step.content)
        except RuntimeError as error:  # mask() admitted it; defensive only
            reason = f"type_error: {error}"
            break

        token_ids.append(step.token_id)
        pre_entropies.append(step.pre_entropy)
        entropies.append(step.post_entropy)
        retries.append(step.retries)
        generated += 1
        runtime.extend(step.token_id)

    return GenerationResult(
        text=synth.input(),
        is_complete=synth.status() == "typed",
        tokens_generated=generated,
        stopped_reason=reason,
        exported_context=dict(synth.context()),
        step_token_ids=token_ids,
        step_pre_entropies=pre_entropies,
        step_entropies=entropies,
        step_retries=retries,
    )


def generate_unconstrained(
    runtime: Runtime,
    *,
    prompt_ids: Sequence[int],
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
) -> GenerationResult:
    """The comparison arm: the same model and prompt with no grammar at all.

    Telemetry stays empty rather than being filled with the unmasked
    distribution twice over — pre and post entropy are equal by definition when
    there is no mask, and a chart of that would imply a measurement nobody made.
    """
    rng = np.random.default_rng(seed)
    runtime.reset(list(prompt_ids))
    pieces: list[str] = []
    reason = "max_tokens"
    generated = 0

    for _ in range(max_tokens):
        token_id = _draw(runtime.logits(), temperature, rng)
        if token_id < 0:
            reason = "no_valid"
            break
        if runtime.is_stop(token_id):
            reason = "stop_token"
            break
        pieces.append(runtime.token_text(token_id))
        generated += 1
        runtime.extend(token_id)

    return GenerationResult(
        text="".join(pieces),
        is_complete=reason == "stop_token",
        tokens_generated=generated,
        stopped_reason=reason,
    )
