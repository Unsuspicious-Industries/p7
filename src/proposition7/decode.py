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
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .inference import GenerationResult
from .mask_cache import DEFAULT_MASK_CACHE_SIZE, mask_candidates
from .runtime import Runtime

log = logging.getLogger(__name__)

DEADLINE_OVERSHOOT_NOTE = """A deadline is checked between engine calls, never inside one.

Checking only between *steps* was not enough, and the way that failed is worth
keeping. One step may call the engine up to `MAX_RETRIES` times and then `_scan`
a ~152k-token vocabulary a block at a time; on a real run a single constrained
step ran 1200s against a 600s deadline, the client gave up first, and the
generation slot it still held turned the whole server into a 429 -- the exact
outage the deadline exists to prevent, reached by overshooting the deadline
rather than by ignoring it.

So `sample` and `_scan` take the deadline too and give up between engine calls.
The bound is `deadline_seconds` plus one call, not plus one step. Interrupting
*inside* a call is still not available to us and still would not be wanted: the
alternative is abandoning a thread that keeps running, holds the GIL and never
releases the caller's slot."""

THINK_MARKER = "</think>"
"""Where a reasoning model stops thinking and starts answering.

The marker is the model's own, not something this loop imposes: these models
open a reasoning channel unprompted and close it with this token before the
answer proper. `generate_mixed` switches the grammar on at that boundary, and
graders strip everything before it, so the two agree on what counts as the
answer. A model that never closes the channel has not produced one."""

MAX_RETRIES = 2048
"""Sampled draws before `_scan` takes over.

Not a correctness bound -- `_scan` is what makes `no_valid` mean what it says.
This is how long the *sampled* distribution gets to find an admissible token
before the search stops respecting temperature and simply takes the best one."""


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
    timed_out: bool = False
    """The deadline passed while this step was still looking for a token.

    Distinct from an empty step, which claims the grammar admits nothing here.
    A search that was cut short has not established that, and recording it as
    `no_valid` would blame the grammar for the clock."""
    scanned: int = 0
    """Tokens examined by the exhaustive fallback, 0 if it was not needed.

    Non-zero means the sampled retry loop exhausted `MAX_RETRIES` without
    finding an admissible token and `_scan` had to be asked. That is a fallback,
    so it is recorded rather than hidden: a run where it fires often is a run
    whose grammar and model disagree about almost every token.
    """


def sample(
    runtime: Runtime,
    synth: Any,
    logits: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float = 0.0,
    grammar_hash: bytes = b"",
    mask_cache_size: int = DEFAULT_MASK_CACHE_SIZE,
    deadline: float | None = None,
) -> Step:
    """Draw a token the grammar will accept, retrying past the ones it will not.

    `synth` holds the accepted prefix and is only *read* here — `mask` is
    state-free — so a rejected candidate costs nothing to undo. That is also
    why the runtime is never told about a rejection: nothing was committed to
    it either.

    `pre_entropy` is computed once, before the retry loop, so it measures the
    model's own uncertainty rather than how many attempts this position took.
    `post_entropy` is measured at acceptance over whatever the retry loop had
    not yet excluded — an upper bound on true post-mask entropy, never the
    thing itself, since knowing the valid set means asking the engine about the
    whole vocabulary. See `inference.GenerationResult.step_entropies`.
    """
    valid = np.isfinite(logits)
    step = Step(pre_entropy=_entropy_bits(logits))

    for _ in range(MAX_RETRIES):
        if deadline is not None and time.monotonic() > deadline:
            step.timed_out = True
            return step
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

    # The retry loop ran out of budget, which is NOT the grammar admitting
    # nothing. `valid` still holds every token the loop never happened to draw:
    # `MAX_RETRIES` is 2048 against a ~152k vocabulary, so at most 1.3% of it
    # was examined. Returning an empty step here reports "no_valid" and blames
    # the model for the search giving up -- and it fires exactly where this loop
    # is most needed, at step 0 of a reasoning model, whose distribution is
    # concentrated on the `<think>` the grammar refuses.
    #
    # A productive grammar at a live prefix always admits some token, so the
    # only correct way to conclude otherwise is to have looked.
    return _scan(
        runtime, synth, logits, valid, step, grammar_hash, mask_cache_size,
        deadline=deadline,
    )


def _scan(
    runtime: Runtime,
    synth: Any,
    logits: np.ndarray,
    valid: np.ndarray,
    step: Step,
    grammar_hash: bytes,
    mask_cache_size: int,
    *,
    block: int = 256,
    deadline: float | None = None,
) -> Step:
    """The most probable admissible token, found by looking at all of them.

    Reached only when the sampled retry loop exhausted its budget. Walks the
    tokens that loop never examined in descending-logit order, so the token
    returned is still the highest-probability one the grammar admits, and stops
    at the first hit. `synth.mask` answers a whole list in one engine call, so
    this costs one call per `block` tokens rather than one per token.

    Deterministic: no draw from `rng`, so two identical `generate()` calls still
    return identical results. It is a fallback rather than the main path because
    it abandons the temperature -- at this point the alternative is emitting
    nothing at all.

    An empty step from *here* means the grammar genuinely admits no token at
    this position. For a productive grammar and a live prefix that is an
    invariant violation rather than a model outcome, so it is logged as one.
    """
    order = [int(t) for t in np.argsort(logits, kind="stable")[::-1] if valid[t]]
    for start in range(0, len(order), block):
        # One check per block is one check per engine call: the whole scan is
        # ~600 calls, and it is the scan that overran the deadline in practice.
        if deadline is not None and time.monotonic() > deadline:
            step.timed_out = True
            step.scanned = start
            return step
        chunk = order[start : start + block]
        offered: list[str] = []
        spans: list[tuple[int, int, int, str]] = []
        for token_id in chunk:
            token = runtime.token_text(token_id)
            # Same admissibility rules as the retry loop: whitespace-only tokens
            # carry no content, and a stop token only ends a *finished* program.
            if not token or not token.strip():
                continue
            if runtime.is_stop(token_id):
                if synth.status() == "typed":
                    step.token_id, step.token, step.is_stop = token_id, token, True
                    step.post_entropy = _entropy_bits(np.where(valid, logits, -np.inf))
                    step.scanned = start + len(chunk)
                    return step
                continue
            candidates = _spellings(token, synth.input())
            spans.append((len(offered), len(candidates), token_id, token))
            offered.extend(candidates)
        if not offered:
            continue
        admitted = mask_candidates(synth, grammar_hash, offered, mask_cache_size)
        for begin, count, token_id, token in spans:
            verdicts = admitted[begin : begin + count]
            content = next(
                (c for c, ok in zip(offered[begin : begin + count], verdicts) if ok),
                None,
            )
            if content is None:
                continue
            step.token_id, step.token, step.content = token_id, token, content
            step.post_entropy = _entropy_bits(np.where(valid, logits, -np.inf))
            step.scanned = start + len(chunk)
            log.warning(
                "retry budget exhausted; exhaustive scan admitted %r after "
                "examining %d token(s). The grammar and the model disagree "
                "about nearly every token at this position.",
                content, step.scanned,
            )
            return step
    log.error(
        "no token in the vocabulary is admissible at prefix %r (status %s). "
        "A productive grammar at a live prefix always admits something, so "
        "this is a grammar or engine fault, not a model outcome.",
        synth.input()[-80:], synth.status(),
    )
    step.scanned = len(order)
    return step


@dataclass
class _Steps:
    """What one run of the masked loop produced, before it becomes a result."""

    reason: str
    generated: int = 0
    token_ids: list[int] = field(default_factory=list)
    pre_entropies: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    retries: list[float] = field(default_factory=list)


def _constrained_loop(
    runtime: Runtime,
    synth: Any,
    rng: np.random.Generator,
    *,
    grammar_hash: bytes,
    max_tokens: int,
    temperature: float,
    mask_cache_size: int,
    deadline_seconds: float | None,
    started: float | None = None,
) -> _Steps:
    """The masked loop itself, without deciding where the model starts.

    `generate` calls this with a runtime holding only the prompt; `generate_mixed`
    calls it with one that already holds the model's own reasoning, so the
    grammar governs the program while the thinking before it stays in context.
    Sharing the loop is the point rather than a tidiness: an arm decoding through
    a second copy of it would be measuring a different decoder, and the
    comparison between arms is the entire experiment.
    """
    steps = _Steps("max_tokens")
    if started is None:
        started = time.monotonic()
    deadline = None if deadline_seconds is None else started + deadline_seconds

    for _ in range(max_tokens):
        if deadline is not None and time.monotonic() > deadline:
            steps.reason = "deadline"
            break
        try:
            step = sample(
                runtime,
                synth,
                runtime.logits(),
                rng,
                temperature=temperature,
                grammar_hash=grammar_hash,
                mask_cache_size=mask_cache_size,
                deadline=deadline,
            )
        except Exception as error:  # noqa: BLE001 - becomes the stop reason
            steps.reason = f"type_error: {error}"
            break

        if step.timed_out:
            # The search was cut short, so it never established anything about
            # the grammar. Reported as the clock, not as the grammar.
            steps.reason = "deadline"
            break
        if step.token is None:
            # Nothing admissible. At a complete program that is success; mid
            # prefix it means the model's lattice and the grammar diverged.
            steps.reason = "complete" if synth.status() == "typed" else "no_valid"
            break
        if step.is_stop:
            steps.reason = (
                "complete" if synth.status() == "typed" else f"stop_token:{step.token}"
            )
            break

        try:
            synth.feed(step.content)
        except RuntimeError as error:  # mask() admitted it; defensive only
            steps.reason = f"type_error: {error}"
            break

        steps.token_ids.append(step.token_id)
        steps.pre_entropies.append(step.pre_entropy)
        steps.entropies.append(step.post_entropy)
        steps.retries.append(step.retries)
        steps.generated += 1
        runtime.extend(step.token_id)

    return steps


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
    deadline_seconds: float | None = None,
) -> GenerationResult:
    """Decode under `grammar`, returning the text aufbau itself validated.

    `deadline_seconds` bounds the wall time this loop may spend. It exists
    because a constrained step is not bounded by `max_tokens` alone: each one
    may retry up to `MAX_RETRIES` times and then scan the vocabulary, so a
    model far from its grammar can decode for hours. Exceeding it stops the
    loop and reports `deadline`, which is a measured outcome; the alternative
    is a caller that waits forever. See `DEADLINE_OVERSHOOT_NOTE`."""
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
    steps = _constrained_loop(
        runtime,
        synth,
        rng,
        grammar_hash=grammar_hash,
        max_tokens=max_tokens,
        temperature=temperature,
        mask_cache_size=mask_cache_size,
        deadline_seconds=deadline_seconds,
    )

    return GenerationResult(
        text=synth.input(),
        is_complete=synth.status() == "typed",
        tokens_generated=steps.generated,
        stopped_reason=steps.reason,
        exported_context=dict(synth.context()),
        step_token_ids=steps.token_ids,
        step_pre_entropies=steps.pre_entropies,
        step_entropies=steps.entropies,
        step_retries=steps.retries,
    )


def generate_unconstrained(
    runtime: Runtime,
    *,
    prompt_ids: Sequence[int],
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int | None = None,
    deadline_seconds: float | None = None,
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
    started = time.monotonic()

    for _ in range(max_tokens):
        # Bounded for the same reason as the constrained arm, so that the two
        # arms of one experiment fail the same way and stay comparable.
        if deadline_seconds is not None and time.monotonic() - started > deadline_seconds:
            reason = "deadline"
            break
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


def generate_mixed(
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
    deadline_seconds: float | None = None,
    think_marker: str = THINK_MARKER,
    think_budget: int | None = None,
) -> GenerationResult:
    """Free reasoning, then a grammar-constrained program.

    The model decodes unconstrained until it closes its own reasoning channel
    with `think_marker`; every token after that is masked by `grammar`. The two
    phases share one runtime, so the program is written in the context of the
    thinking that produced it -- that sharing is the arm's whole claim, and it is
    why this cannot be two calls stitched together by a caller.

    They do not share a budget, and that is deliberate. `max_tokens` in this
    corpus is sized for a program -- 256 to 600 tokens -- and a reasoning model
    does not finish a thought in that. Splitting it would spend half on a
    truncated thought and then mask a program onto the wreckage: measured on a
    0.8b model, a 128-token thinking half never once reached `</think>`, and the
    program that followed was conditioned on a sentence stopped mid-word. So
    `think_budget` defaults to a full `max_tokens` of its own, and the
    constrained phase gets its own `max_tokens` after it. `deadline_seconds`,
    not the token count, is what bounds the pair.

    `text` is the program alone. The reasoning is not part of the answer, so it
    is reported in `diagnostics` instead -- which is also what lets this arm face
    the same compiler as every other arm without a grader that knows about it.
    """
    import aufbau

    rng = np.random.default_rng(seed)
    grammar_hash = hashlib.blake2b(grammar.encode(), digest_size=16).digest()
    synth = aufbau.Synthesizer(grammar, "")

    for name, type_source in (aufbau_context or {}).items():
        synth.add_to_ctx(name, type_source)

    if initial:
        synth.set_input(initial)
        if synth.status() == "dead":
            return GenerationResult(
                initial, False, 0, "type_error: initial text is not a live prefix"
            )

    if think_budget is None:
        think_budget = max_tokens
    started = time.monotonic()
    runtime.reset(list(prompt_ids))

    thought: list[str] = []
    think_tokens = 0
    closed = False
    stopped_thinking = ""
    # Only the tail can complete the marker, and keeping just the tail is what
    # makes this check O(1) per step instead of rescanning the whole thought.
    tail = ""

    # Not `min(think_budget, max_tokens)`: the two phases have separate budgets
    # (see above), and taking the smaller one hands the thinking phase the
    # program's budget -- 256 where the caller asked for 1280 -- which is the
    # truncated-thought failure this arm exists to avoid.
    for _ in range(max(0, think_budget)):
        if deadline_seconds is not None and time.monotonic() - started > deadline_seconds:
            stopped_thinking = "deadline"
            break
        token_id = _draw(runtime.logits(), temperature, rng)
        if token_id < 0:
            stopped_thinking = "no_valid"
            break
        if runtime.is_stop(token_id):
            # The model ended its turn without ever opening a program. The
            # grammar still gets its phase: this arm's output is constrained by
            # construction, so an early stop shortens the thinking rather than
            # cancelling the answer.
            stopped_thinking = "stop_token"
            break
        text = runtime.token_text(token_id)
        thought.append(text)
        think_tokens += 1
        runtime.extend(token_id)
        tail = (tail + text)[-2 * len(think_marker) :]
        if think_marker in tail:
            closed = True
            stopped_thinking = "closed"
            break
    else:
        stopped_thinking = stopped_thinking or "think_budget"

    steps = _constrained_loop(
        runtime,
        synth,
        rng,
        grammar_hash=grammar_hash,
        max_tokens=max_tokens,
        temperature=temperature,
        mask_cache_size=mask_cache_size,
        deadline_seconds=deadline_seconds,
        started=started,
    )

    return GenerationResult(
        text=synth.input(),
        is_complete=synth.status() == "typed",
        tokens_generated=steps.generated,
        stopped_reason=steps.reason,
        exported_context=dict(synth.context()),
        diagnostics={
            "think_tokens": think_tokens,
            "think_closed": closed,
            "think_stopped_reason": stopped_thinking,
            "think_text": "".join(thought),
        },
        step_token_ids=steps.token_ids,
        step_pre_entropies=steps.pre_entropies,
        step_entropies=steps.entropies,
        step_retries=steps.retries,
    )
