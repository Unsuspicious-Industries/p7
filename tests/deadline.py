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
