"""The candidate-spelling policy, and the decode it used to corrupt.

A benchmark run on 2026-08-31 produced 'longlongrev' for `long long rev` and
'size_ti' for `size_t i`, and the mixed arm failed 8 of 11 cases by emitting
'chingingingingching...' until it hit max_tokens. All three are one bug: the
lstripped fallback was offered unconditionally, so a token boundary the model
had asked for could be silently removed, welding two words into one identifier
that the grammar would then extend forever.
"""

import pytest

from proposition7.spelling import candidate_spellings

aufbau = pytest.importorskip("aufbau", reason="requires the aufbau engine")


def _offers(token, prefix):
    return candidate_spellings(token, prefix)


def test_a_space_the_model_asked_for_is_not_removed_between_words():
    """' long' after 'long' must not be offerable as 'long'."""
    assert _offers(" long", "int solve(int x) { long") == [" long"]
    assert _offers(" i", "int solve(int x) { size_t") == [" i"]
    assert _offers(" rev", "int solve(int x) { longlong") == [" rev"]


def test_digit_continuation_still_fuses():
    """The case the fallback exists for: '4' + ' 3' must be able to mean '43'."""
    assert _offers(" 3", "int solve(int x) { return 4") == [" 3", "3"]


def test_fusion_is_allowed_where_no_word_boundary_is_at_stake():
    """After punctuation there is no word to weld to, so the fallback stands."""
    assert "long" in _offers(" long", "int solve(int x) {")
    assert "x" in _offers(" x", "int solve(")


def test_the_models_own_spelling_always_comes_first():
    for token, prefix in ((" x", "let"), ("x", "let "), (" 3", "4"), (" long", "{")):
        assert _offers(token, prefix)[0] == token


def _decode(spg, pieces):
    """Drive the real policy against the real engine, as the sampler does."""
    synth = aufbau.Synthesizer.from_grammar(spg, "")
    rejected = 0
    for piece in pieces:
        candidates = candidate_spellings(piece, synth.input())
        verdicts = list(synth.mask(candidates))
        winner = next((c for c, ok in zip(candidates, verdicts) if ok), None)
        if winner is None:
            rejected += 1          # the sampler would draw another token here
            continue
        synth.feed(winner)
    return synth.input(), rejected


@pytest.fixture(scope="module")
def c_grammar():
    from grammars import get_grammar

    return aufbau.SPG(get_grammar("c"))


HEAD = ["int", " solve", "(", "int", " x", ")", " {"]


RUNAWAY = HEAD + [" char", " line"] + [" ment"] * 12
"""A declaration puts the decoder inside an identifier, which is the absorbing
state: every alphabetic suffix extends it, so nothing can ever reject."""


def test_the_identifier_runaway_is_refused(c_grammar):
    """Before the fix this welded twelve ' ment' tokens into one identifier and
    would have gone on emitting until max_tokens."""
    text, rejected = _decode(c_grammar, RUNAWAY)
    assert text == "int solve(int x) { char line"
    assert rejected == 12


def test_the_runaway_is_exactly_what_the_old_policy_did(c_grammar):
    """The contrast the fix exists for, pinned so it cannot quietly return."""
    synth = aufbau.Synthesizer.from_grammar(c_grammar, "")
    for piece in RUNAWAY:
        old_style = [piece]
        if piece.lstrip() and piece.lstrip() not in old_style:
            old_style.append(piece.lstrip())
        winner = next(
            (c for c, ok in zip(old_style, synth.mask(old_style)) if ok), None
        )
        if winner is not None:
            synth.feed(winner)
    assert synth.input() == "int solve(int x) { char line" + "ment" * 12
