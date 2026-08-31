"""The decode loop's wall-clock bound.

These exist because an unbounded decode is not merely slow. wirt serialises
generation behind one slot on one thread, so a loop that does not return makes
the whole server answer 429 until it is restarted -- which is exactly how a
benchmark run was lost. The bound is what turns that outage into a recorded
outcome, so it is worth a test that watches the clock rather than the tokens.
"""

import time

import numpy as np
import pytest

from proposition7 import decode


class SlowRuntime:
    """A model that always wants to keep going, and is slow about it.

    It never offers a stop token and never runs out of vocabulary, so nothing
    but the deadline can end a loop driven by it.
    """

    vocab_size = 8

    def __init__(self, step_seconds: float = 0.01) -> None:
        self.step_seconds = step_seconds
        self.steps = 0

    def encode_prompt(self, model_context, *, initial: str = ""):
        return [0]

    def reset(self, prompt_ids) -> None:
        self.steps = 0

    def extend(self, token_id: int) -> None:
        pass

    def logits(self):
        self.steps += 1
        time.sleep(self.step_seconds)
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        scores[1] = 1.0
        return scores

    def token_text(self, token_id: int) -> str:
        return "a"

    def is_stop(self, token_id: int) -> bool:
        return False


def test_unconstrained_decode_stops_at_the_deadline():
    runtime = SlowRuntime(step_seconds=0.01)
    started = time.monotonic()
    result = decode.generate_unconstrained(
        runtime, prompt_ids=[0], max_tokens=1_000_000, deadline_seconds=0.2
    )
    elapsed = time.monotonic() - started

    assert result.stopped_reason == "deadline"
    assert not result.is_complete
    # The bound is the deadline plus at most one step, not the token budget.
    assert elapsed < 5.0, f"loop ran {elapsed:.1f}s past a 0.2s deadline"
    # It did real work before stopping, so this is a bound and not a no-op.
    assert result.tokens_generated > 0


def test_without_a_deadline_the_token_budget_is_the_only_bound():
    runtime = SlowRuntime(step_seconds=0.0)
    result = decode.generate_unconstrained(
        runtime, prompt_ids=[0], max_tokens=17, deadline_seconds=None
    )
    assert result.stopped_reason == "max_tokens"
    assert result.tokens_generated == 17


def test_a_deadline_that_cannot_be_hit_does_not_change_the_outcome():
    """The bound must be inert when it is not reached, or it is not a bound."""
    bounded = decode.generate_unconstrained(
        SlowRuntime(step_seconds=0.0), prompt_ids=[0], max_tokens=5, deadline_seconds=3600
    )
    unbounded = decode.generate_unconstrained(
        SlowRuntime(step_seconds=0.0), prompt_ids=[0], max_tokens=5, deadline_seconds=None
    )
    assert bounded.text == unbounded.text
    assert bounded.stopped_reason == unbounded.stopped_reason == "max_tokens"
    assert bounded.tokens_generated == unbounded.tokens_generated == 5


def test_constrained_decode_stops_at_the_deadline():
    """The arm that actually wedged: a grammar the model keeps feeding."""
    pytest.importorskip("aufbau")
    grammar = 'Start ::= Digits\nDigits ::= "a" Digits | "a"\n'

    runtime = SlowRuntime(step_seconds=0.01)
    started = time.monotonic()
    result = decode.generate(
        runtime,
        grammar=grammar,
        prompt_ids=[0],
        max_tokens=1_000_000,
        deadline_seconds=0.2,
    )
    elapsed = time.monotonic() - started

    assert result.stopped_reason == "deadline"
    assert elapsed < 10.0, f"loop ran {elapsed:.1f}s past a 0.2s deadline"


class SlowVocabRuntime:
    """A model whose every candidate is expensive to even look at.

    `token_text` is called once per candidate in the retry loop and once per
    token in `_scan`, so making it slow makes a *single step* slow -- which is
    the shape of the failure this guards. No aufbau call is slowed: the point is
    that a step can outlive a deadline without any one call doing so.
    """

    def __init__(self, vocab_size: int = 4096, token_seconds: float = 0.002) -> None:
        self.vocab_size = vocab_size
        self.token_seconds = token_seconds

    def encode_prompt(self, model_context, *, initial: str = ""):
        return [0]

    def reset(self, prompt_ids) -> None:
        pass

    def extend(self, token_id: int) -> None:
        pass

    def logits(self):
        scores = np.linspace(1.0, 0.0, self.vocab_size).astype(np.float32)
        return scores

    def token_text(self, token_id: int) -> str:
        time.sleep(self.token_seconds)
        # Nothing the grammar below will take, so the retry loop never succeeds
        # and the exhaustive scan is reached.
        return "z"

    def is_stop(self, token_id: int) -> bool:
        return False


def test_one_step_cannot_outlive_the_deadline():
    """The regression: a step, not a loop, is what overran in production.

    A constrained step ran 1200s against a 600s deadline because the bound was
    only checked between steps. The client timed out first, the server kept the
    generation slot it was still holding, and every later request became a 429 --
    so the cell died in the same way the between-steps bound was added to
    prevent. `MAX_RETRIES` alone is 2048 candidates before `_scan` even starts.
    """
    runtime = SlowVocabRuntime()
    started = time.monotonic()
    result = decode.generate(
        runtime,
        grammar='Start ::= Digits\nDigits ::= "a" Digits | "a"\n',
        prompt_ids=[0],
        max_tokens=4,
        deadline_seconds=0.3,
    )
    elapsed = time.monotonic() - started

    assert result.stopped_reason == "deadline"
    # Unbounded, the first step alone is 2048 x 2ms of retries plus a 4096-token
    # scan. Finishing near the deadline is the whole claim.
    assert elapsed < 5.0, f"one step ran {elapsed:.1f}s past a 0.3s deadline"


def test_a_step_cut_short_is_not_reported_as_an_empty_grammar():
    """`no_valid` is a claim about the grammar; a stopped clock cannot make it.

    The search never finished, so nothing was established about what the grammar
    admits. Recording that as `no_valid` would put "we ran out of time" and "the
    grammar admits nothing here" in one column, and the second is an invariant
    violation worth noticing on its own.
    """
    runtime = SlowVocabRuntime()
    result = decode.generate(
        runtime,
        grammar='Start ::= Digits\nDigits ::= "a" Digits | "a"\n',
        prompt_ids=[0],
        max_tokens=4,
        deadline_seconds=0.3,
    )
    assert result.stopped_reason == "deadline"
    assert result.stopped_reason != "no_valid"
