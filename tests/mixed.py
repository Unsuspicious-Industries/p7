"""The mixed arm: free reasoning, then a grammar-constrained answer.

The arm exists because the other two each give something up. The unconstrained
arm reasons but cannot be held to a form -- in the run these tests were written
against it spent 44-62% of its characters thinking and hit its token budget
76-100% of the time. The constrained arm holds the form from the first token,
which is also the first token of the model's reasoning, so it cannot think at
all. Mixed decodes one sequence in two phases, and what these tests pin down is
the boundary: that the switch happens where the model closes its own reasoning
channel, that the reasoning never reaches the answer, and that a model which
never stops thinking still produces a program rather than nothing.
"""

import numpy as np

from proposition7 import decode

GRAMMAR = 'Start ::= Digits\nDigits ::= "a" Digits | "a"\n'
VOCAB = ["<think>", "a", "</think>", " hmm"]


class ScriptedRuntime:
    """Emits a fixed token script, then leaves the choice to whoever is asking.

    The script is how a test says "the model thought this"; once it runs out the
    logits go flat, so the constrained phase is genuinely choosing under the
    grammar rather than replaying more script.
    """

    vocab_size = len(VOCAB)

    def __init__(self, script):
        self.script = list(script)
        self.emitted = []

    def encode_prompt(self, model_context, *, initial: str = ""):
        return [0]

    def reset(self, prompt_ids) -> None:
        self.emitted = []

    def extend(self, token_id: int) -> None:
        self.emitted.append(token_id)

    def logits(self):
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        step = len(self.emitted)
        if step < len(self.script):
            scores[self.script[step]] = 10.0
        else:
            scores[VOCAB.index("a")] = 1.0
        return scores

    def encode(self, text: str) -> list[int]:
        """Token ids for injected text, greedily over the toy vocabulary."""
        ids, rest = [], text
        while rest:
            for index, piece in enumerate(VOCAB):
                if piece and rest.startswith(piece):
                    ids.append(index)
                    rest = rest[len(piece) :]
                    break
            else:
                rest = rest[1:]  # separators the toy vocabulary has no token for
        return ids

    def token_text(self, token_id: int) -> str:
        return VOCAB[token_id]

    def is_stop(self, token_id: int) -> bool:
        return False


def _think_then_answer():
    return ScriptedRuntime([VOCAB.index("<think>"), VOCAB.index(" hmm"), VOCAB.index("</think>")])


def test_the_grammar_takes_over_where_the_model_closes_its_reasoning():
    result = decode.generate_mixed(
        _think_then_answer(), grammar=GRAMMAR, prompt_ids=[0], max_tokens=32
    )
    assert result.diagnostics["think_closed"] is True
    assert result.diagnostics["think_stopped_reason"] == "closed"
    # Three tokens of reasoning, and the marker is the last of them.
    assert result.diagnostics["think_tokens"] == 3
    assert result.diagnostics["think_text"].endswith("</think>")


def test_the_reasoning_is_not_part_of_the_answer():
    result = decode.generate_mixed(
        _think_then_answer(), grammar=GRAMMAR, prompt_ids=[0], max_tokens=32
    )
    # `text` is what a grader compiles. Reasoning in it would be a syntax error
    # in every language the corpus uses, which is exactly how the unconstrained
    # arm scored 0% before anyone noticed the wrapper was being compiled.
    assert "<think>" not in result.text
    assert "</think>" not in result.text
    # aufbau renders its own separators, so the answer is "a"s and whitespace.
    assert result.text.split() and set(result.text.split()) == {"a"}


def test_reasoning_tokens_are_not_counted_as_answer_tokens():
    result = decode.generate_mixed(
        _think_then_answer(), grammar=GRAMMAR, prompt_ids=[0], max_tokens=32
    )
    # tokens_generated measures the constrained phase, so it stays comparable
    # with the constrained arm's own count rather than being inflated by however
    # long the model chose to think.
    assert result.tokens_generated == result.text.count("a")
    assert result.diagnostics["think_tokens"] == 3


def test_a_model_that_never_stops_thinking_still_answers():
    # No marker anywhere in the script: the reasoning budget is what ends phase one.
    runtime = ScriptedRuntime([VOCAB.index(" hmm")] * 100)
    result = decode.generate_mixed(
        runtime, grammar=GRAMMAR, prompt_ids=[0], max_tokens=20, think_budget=4
    )
    assert result.diagnostics["think_closed"] is False
    assert result.diagnostics["think_stopped_reason"] == "think_budget"
    assert result.diagnostics["think_tokens"] == 4
    # The answer is still grammar-shaped: an arm defined by its constrained
    # output must not degrade into the unconstrained one when thinking overruns.
    assert result.text.split() and set(result.text.split()) == {"a"}


def test_the_answer_gets_its_own_budget_not_what_thinking_left_over():
    runtime = ScriptedRuntime([VOCAB.index(" hmm")] * 100)
    result = decode.generate_mixed(
        runtime, grammar=GRAMMAR, prompt_ids=[0], max_tokens=10, think_budget=4
    )
    # `max_tokens` here is a program budget. Charging the thinking to it would
    # leave six tokens for a program that needed ten, and the arm would be
    # measuring the split rather than the constraint.
    assert result.diagnostics["think_tokens"] == 4
    assert result.tokens_generated == 10


def test_the_deadline_still_bounds_both_phases():
    import time

    class Slow(ScriptedRuntime):
        def logits(self):
            time.sleep(0.01)
            return super().logits()

    runtime = Slow([VOCAB.index(" hmm")] * 10_000)
    started = time.monotonic()
    result = decode.generate_mixed(
        runtime,
        grammar=GRAMMAR,
        prompt_ids=[0],
        max_tokens=1_000_000,
        think_budget=1_000_000,
        deadline_seconds=0.2,
    )
    elapsed = time.monotonic() - started
    # One deadline covers the whole call: a per-phase bound would let a mixed
    # request take twice as long as the constrained request it is compared with.
    assert elapsed < 5.0, f"mixed ran {elapsed:.1f}s past a 0.2s deadline"
    assert result.diagnostics["think_stopped_reason"] == "deadline"


def test_the_program_budget_does_not_cap_the_thinking_budget():
    """A long think_budget survives a short max_tokens.

    The two phases are documented as having separate budgets. Clamping phase one
    to `min(think_budget, max_tokens)` quietly gave the thinker the program's
    budget instead of its own, reintroducing the mid-word truncation that the
    separate budget exists to remove -- and it did so silently, because a
    truncated thought still yields a well-formed program.
    """
    script = [VOCAB.index(" hmm")] * 400 + [VOCAB.index("</think>")]
    result = decode.generate_mixed(
        ScriptedRuntime(script), grammar=GRAMMAR, prompt_ids=[0],
        max_tokens=8, think_budget=600,
    )
    # 8 tokens of program budget must not stop a 401-token thought.
    assert result.diagnostics["think_closed"] is True
    assert result.diagnostics["think_stopped_reason"] == "closed"
    assert result.diagnostics["think_tokens"] == 401
    # And the answer still respects its own, separate budget.
    assert result.text and "hmm" not in result.text


def test_a_thought_that_never_closes_is_closed_for_the_model():
    """Phase two must not decode from inside the reasoning block.

    When the thinking budget runs out mid-sentence, the model is still in its
    reasoning distribution. Handing that context to the grammar produces
    programs that are well-formed and meaningless, because the mask only
    constrains shape. Closing the channel first is what makes the second phase
    an answer instead of more reasoning in a program's clothing.
    """
    runtime = ScriptedRuntime([VOCAB.index(" hmm")] * 50)
    result = decode.generate_mixed(
        runtime, grammar=GRAMMAR, prompt_ids=[0], max_tokens=8, think_budget=5
    )
    assert result.diagnostics["think_closed"] is False
    assert result.diagnostics["think_stopped_reason"] == "think_budget"
    assert result.diagnostics["think_forced_close"] is True
    # The marker really reached the context the constrained phase decodes from.
    assert VOCAB.index("</think>") in runtime.emitted
    marker_at = runtime.emitted.index(VOCAB.index("</think>"))
    assert marker_at == 5, "the marker must land right after the thinking, not later"


def test_a_thought_the_model_closes_itself_is_left_alone():
    runtime = _think_then_answer()
    result = decode.generate_mixed(
        runtime, grammar=GRAMMAR, prompt_ids=[0], max_tokens=8
    )
    assert result.diagnostics["think_closed"] is True
    assert result.diagnostics["think_forced_close"] is False
    # Exactly one marker: the model's own, with nothing appended after it.
    assert runtime.emitted.count(VOCAB.index("</think>")) == 1


def test_a_model_that_ends_its_turn_early_still_gets_its_block_closed():
    """An early stop leaves the block open just as an exhausted budget does."""
    class Stopping(ScriptedRuntime):
        def is_stop(self, token_id: int) -> bool:
            return token_id == VOCAB.index(" hmm")

    runtime = Stopping([VOCAB.index(" hmm")])
    result = decode.generate_mixed(
        runtime, grammar=GRAMMAR, prompt_ids=[0], max_tokens=8, think_budget=20
    )
    assert result.diagnostics["think_stopped_reason"] == "stop_token"
    assert result.diagnostics["think_forced_close"] is True
    assert VOCAB.index("</think>") in runtime.emitted


def test_a_masked_arm_can_stop_where_the_grammar_says_the_program_is_finished():
    """"Complete" and "finished" are different questions.

    This grammar accepts one "a" and also accepts more, so every position after
    the first is both a complete program and a prefix of a longer one. Without
    the flag the decoder runs to its budget and returns a complete term with
    another growing out of it. Off by default, since a caller who wants the
    whole translation unit would see silent truncation.
    """
    ran_on = decode.generate_mixed(
        _think_then_answer(), grammar=GRAMMAR, prompt_ids=[0], max_tokens=32
    )
    stopped = decode.generate_mixed(
        _think_then_answer(), grammar=GRAMMAR, prompt_ids=[0], max_tokens=32,
        stop_at_complete=True,
    )

    assert ran_on.stopped_reason == "max_tokens"
    assert ran_on.tokens_generated == 32

    assert stopped.stopped_reason == "complete"
    assert stopped.tokens_generated == 1
    assert stopped.text.split() == ["a"]
    # Both are complete; only one of them is *only* the program asked for.
    assert ran_on.is_complete and stopped.is_complete
    # The reasoning still happened, and still is not part of the answer.
    assert stopped.diagnostics["think_closed"] is True
    assert "</think>" not in stopped.text


def test_stopping_at_complete_applies_to_the_plain_constrained_arm_too():
    """The two masked arms share `_constrained_loop`, and must share this.

    Otherwise they would differ by more than where the mask applies, which is
    the one thing the comparison holds fixed.
    """
    runtime = ScriptedRuntime([])
    result = decode.generate(
        runtime, grammar=GRAMMAR, prompt_ids=[0], max_tokens=32,
        stop_at_complete=True,
    )
    assert result.stopped_reason == "complete"
    assert result.tokens_generated == 1
    assert result.text.split() == ["a"]
