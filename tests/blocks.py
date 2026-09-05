"""The per-step stop hook: the mechanism a block boundary is built from.

The policy that uses this lives in the client (gamma's BLOCKS.md). What is
tested here is the half that has to sit next to the logits: that the loop asks
after every accepted token, that a returned reason ends the decode as that
reason, and that the model's own signals still win.
"""

import numpy as np
import pytest

from proposition7 import decode
from proposition7.decode import SettlePolicy, StepInfo


class FlatRuntime:
    """A model with a fixed distribution and no stop token.

    `peak` sets how concentrated the distribution is, which is the only thing
    an entropy policy reacts to: a large value makes every step confident, a
    small one leaves it broad.
    """

    vocab_size = 8

    def __init__(self, peak: float = 1.0) -> None:
        self.peak = peak

    def encode_prompt(self, model_context, *, initial: str = ""):
        return [0]

    def reset(self, prompt_ids) -> None:
        pass

    def extend(self, token_id: int) -> None:
        pass

    def logits(self):
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        scores[1] = self.peak
        return scores

    def token_text(self, token_id: int) -> str:
        return "a"

    def is_stop(self, token_id: int) -> bool:
        return False


def test_the_hook_sees_every_accepted_token_in_order():
    seen = []
    result = decode.generate_unconstrained(
        FlatRuntime(), prompt_ids=[0], max_tokens=5,
        stop=lambda step: seen.append(step) or None,
    )
    assert result.stopped_reason == "max_tokens"
    assert [s.index for s in seen] == [0, 1, 2, 3, 4]
    assert all(s.text == "a" for s in seen)


def test_a_returned_reason_ends_the_decode_as_that_reason():
    result = decode.generate_unconstrained(
        FlatRuntime(), prompt_ids=[0], max_tokens=100,
        stop=lambda step: "settled" if step.index == 2 else None,
    )
    assert result.stopped_reason == "settled"
    # The token that closed the block is counted, so the text holds it.
    assert result.tokens_generated == 3
    assert result.text == "aaa"


def test_no_hook_leaves_the_loop_exactly_as_it_was():
    """The mechanism must be inert when nobody uses it."""
    with_hook = decode.generate_unconstrained(
        FlatRuntime(), prompt_ids=[0], max_tokens=6, stop=lambda step: None
    )
    without = decode.generate_unconstrained(
        FlatRuntime(), prompt_ids=[0], max_tokens=6
    )
    assert with_hook.text == without.text
    assert with_hook.stopped_reason == without.stopped_reason == "max_tokens"
    assert with_hook.tokens_generated == without.tokens_generated == 6


def test_entropy_is_measured_only_when_a_hook_asks_for_it():
    """An unconstrained arm reports no entropy, because pre and post are equal
    without a mask. A hook still needs the number, so it is computed for the
    hook alone and the recorded telemetry stays empty either way."""
    seen = []
    result = decode.generate_unconstrained(
        FlatRuntime(peak=20.0), prompt_ids=[0], max_tokens=3,
        stop=lambda step: seen.append(step) or None,
    )
    assert result.step_entropies == []
    assert all(s.pre_entropy == s.post_entropy for s in seen)
    assert all(s.post_entropy < 0.1 for s in seen), "a peaked distribution"


def test_a_broad_distribution_does_not_settle():
    seen = []
    decode.generate_unconstrained(
        FlatRuntime(peak=0.0), prompt_ids=[0], max_tokens=3,
        stop=lambda step: seen.append(step) or None,
    )
    # Eight equiprobable tokens is exactly three bits.
    assert all(abs(s.post_entropy - 3.0) < 1e-6 for s in seen)


def test_the_policy_closes_a_block_once_the_run_is_long_enough():
    policy = SettlePolicy(entropy_floor=0.5, settle=4)
    result = decode.generate_unconstrained(
        FlatRuntime(peak=20.0), prompt_ids=[0], max_tokens=100, stop=policy.hook()
    )
    assert result.stopped_reason == "settled"
    assert result.tokens_generated == 4


def test_a_settled_policy_never_fires_on_an_open_distribution():
    policy = SettlePolicy(entropy_floor=0.5, settle=4)
    result = decode.generate_unconstrained(
        FlatRuntime(peak=0.0), prompt_ids=[0], max_tokens=9, stop=policy.hook()
    )
    assert result.stopped_reason == "max_tokens"
    assert result.tokens_generated == 9


def test_the_hook_cannot_outrank_the_deadline():
    """A policy that never fires must not remove the wall-clock bound."""
    import time

    class Slow(FlatRuntime):
        def logits(self):
            time.sleep(0.01)
            return super().logits()

    result = decode.generate_unconstrained(
        Slow(), prompt_ids=[0], max_tokens=1_000_000,
        deadline_seconds=0.2, stop=lambda step: None,
    )
    assert result.stopped_reason == "deadline"


def test_step_info_is_a_stable_shape():
    """It crosses a repository boundary, so keep it additive."""
    step = StepInfo(index=3, token_id=7, text="x",
                    pre_entropy=1.5, post_entropy=0.5, retries=2)
    assert (step.index, step.token_id, step.text) == (3, 7, "x")
    assert (step.pre_entropy, step.post_entropy, step.retries) == (1.5, 0.5, 2)
    with pytest.raises(Exception):
        step.index = 4  # frozen


# ── the masked loop ──────────────────────────────────────────────────────

GRAMMAR = 'Start ::= Digits\nDigits ::= "a" Digits | "a"\n'


def test_the_hook_closes_a_masked_block_too():
    from mixed import ScriptedRuntime

    result = decode.generate(
        ScriptedRuntime([]), grammar=GRAMMAR, prompt_ids=[0], max_tokens=32,
        stop=lambda step: "settled" if step.index == 3 else None,
    )
    assert result.stopped_reason == "settled"
    assert result.tokens_generated == 4


def test_the_grammar_s_verdict_outranks_the_policy():
    """Ordering: a block that finished a term reports `complete`, not whatever
    the policy would have said about the same token."""
    from mixed import ScriptedRuntime

    result = decode.generate(
        ScriptedRuntime([]), grammar=GRAMMAR, prompt_ids=[0], max_tokens=32,
        stop_at_complete=True, stop=lambda step: "settled",
    )
    assert result.stopped_reason == "complete"
    assert result.tokens_generated == 1
