"""Tracing: what the decoder refused at each position, and why.

`step_retries` says a position cost 136 attempts. It does not say which 136, so
a run that produces a surprising program leaves no way to tell a grammar that
refused the right token from a model that never proposed it. That distinction
decided a real question: `c_absolute` came back as `return x < x;` under the
constrained arm, which reads like the grammar forbidding integer literals --
and the trace is what shows the literal being admitted and simply out-ranked.

Tracing must therefore be an observation and nothing more. Every test here runs
at temperature 1.0 as well as 0.0, because greedy decoding walks the ranked list
deterministically and barely retries; sampling is what actually exercises the
rejection path, and it is also the frozen setting the benchmark now runs under.
"""

import numpy as np
import pytest

from proposition7 import decode

# `a` is the only spelling the grammar admits, so every other token in the
# vocabulary is refused -- one for being whitespace, two by the grammar.
GRAMMAR = 'Start ::= Digits\nDigits ::= "a" Digits | "a"\n'
VOCAB = ["a", "<think>", "  ", "</think>"]

TEMPERATURES = [0.0, 1.0]


class NoisyRuntime:
    """Ranks the inadmissible tokens above the admissible one at every step.

    That is the shape that matters: the model wants something the grammar will
    not take, so the retry loop has to turn it down before it can get to `a`.
    """

    vocab_size = len(VOCAB)

    def __init__(self):
        self.emitted = []

    def encode_prompt(self, model_context, *, initial: str = ""):
        return [0]

    def reset(self, prompt_ids) -> None:
        self.emitted = []

    def extend(self, token_id: int) -> None:
        self.emitted.append(token_id)

    def logits(self):
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        scores[VOCAB.index("<think>")] = 10.0
        scores[VOCAB.index("  ")] = 8.0
        scores[VOCAB.index("</think>")] = 6.0
        scores[VOCAB.index("a")] = 4.0
        return scores

    def encode(self, text: str) -> list[int]:
        return [VOCAB.index("a")]

    def token_text(self, token_id: int) -> str:
        return VOCAB[token_id]

    def is_stop(self, token_id: int) -> bool:
        return False


def _run(*, trace, temperature, max_tokens=4, seed=7):
    return decode.generate(
        NoisyRuntime(),
        grammar=GRAMMAR,
        prompt_ids=[0],
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        trace=trace,
    )


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_an_untraced_run_carries_no_trace(temperature):
    """Off by default: the trace costs time, and timings are a reported number."""
    assert _run(trace=False, temperature=temperature).step_trace == []


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_a_traced_step_names_every_candidate_it_refused(temperature):
    result = _run(trace=True, temperature=temperature)

    assert len(result.step_trace) == result.tokens_generated
    assert result.step_trace, "a run that generated nothing proves nothing here"

    for entry, retries in zip(result.step_trace, result.step_retries):
        # The count and the list are two views of one fact and must agree; a
        # trace that under-reports is worse than no trace, because it reads as
        # evidence that the missing rejections never happened.
        assert len(entry["rejected"]) == retries
        # `content` is the spelling the grammar took, not the token text: after
        # the first `a` the grammar needs a token boundary, so what it accepts
        # is ` a`. Recording the token would lose that distinction, which is the
        # one that makes a trace replayable.
        assert entry["content"].strip() == "a"
        assert entry["token"] == "a"
        for refusal in entry["rejected"]:
            assert refusal["why"] in {"whitespace", "grammar", "stop_before_typed"}
            assert refusal["token"] == VOCAB[refusal["token_id"]]


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_the_trace_distinguishes_a_whitespace_refusal_from_a_grammar_one(temperature):
    """Two different refusals that `step_retries` reports as the same number.

    `  ` is turned down before the grammar is ever consulted; `<think>` is turned
    down by it. Collapsing them would hide exactly the case this tracing was
    added to settle -- whether a token was inadmissible or merely out-ranked.
    """
    result = _run(trace=True, temperature=temperature, max_tokens=8)
    refusals = [r for entry in result.step_trace for r in entry["rejected"]]
    by_token = {r["token"]: r["why"] for r in refusals}

    assert by_token.get("  ") == "whitespace"
    assert by_token.get("<think>") == "grammar"
    # A grammar refusal says which spellings were offered; a whitespace one
    # cannot, because none were ever built.
    for refusal in refusals:
        if refusal["why"] == "grammar":
            assert refusal["spellings"], "a grammar refusal must name what it refused"


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_tracing_observes_the_decode_without_steering_it(temperature):
    """The load-bearing property: trace on and trace off decode identically.

    If recording changed the draw, every traced investigation would be of a run
    that never happened, and `trace` would have to enter the request fingerprint.
    It does not, and this is why.
    """
    traced = _run(trace=True, temperature=temperature)
    plain = _run(trace=False, temperature=temperature)

    assert traced.text == plain.text
    assert traced.step_token_ids == plain.step_token_ids
    assert traced.step_retries == plain.step_retries
    assert traced.tokens_generated == plain.tokens_generated
    assert traced.stopped_reason == plain.stopped_reason


def test_sampling_reaches_the_admissible_token_by_a_different_road():
    """Guards the parametrisation above from being two copies of one test.

    Greedy is the *worst* case for the retry loop, not the mildest: it walks the
    ranked list in order, so it refuses every token above `a` at every single
    position and its retry count is constant. Sampling draws from the surviving
    distribution and can reach `a` early, so its counts vary. Both settle on the
    same program -- which is the point, since the grammar decides what is
    admissible and the temperature only decides how fast it is found.

    Seeded, so this is deterministic rather than usually-true.
    """
    greedy = _run(trace=True, temperature=0.0, max_tokens=8)
    sampled = _run(trace=True, temperature=1.0, max_tokens=8)

    assert set(greedy.step_retries) == {len(VOCAB) - 1}, (
        "greedy should refuse exactly the tokens ranked above the admissible one"
    )
    assert set(sampled.step_retries) != set(greedy.step_retries), (
        "sampling that reproduced greedy's counts exactly would mean the "
        "temperature-1.0 parametrisation exercises nothing greedy does not"
    )
    assert greedy.text == sampled.text
