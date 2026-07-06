"""Tests for public API pieces that do not require an HF model."""

import proposition7
from proposition7.api import Result, _resolve_grammar


def test_result_dataclass():
    result = Result(text="hello", complete=True, tokens=3, reason="complete")
    assert result.text == "hello" and result.complete and result.thoughts == ""


def test_resolve_grammar_name():
    assert len(_resolve_grammar("stlc")) > 50


def test_resolve_grammar_raw_spec():
    spec = "A ::= 'x'"
    assert _resolve_grammar(spec) == spec


def test_all_grammars_parseable():
    for name in proposition7.list_grammars():
        synthesizer = proposition7.Synthesizer(proposition7.get_grammar(name), "")
        synthesizer.parse()


def test_proposition7_exports_public_api():
    assert proposition7.ConstrainedModel is not None
    assert proposition7.generate is not None


def test_synthesizer_set_input_and_feed_round_trip():
    synthesizer = proposition7.Synthesizer("start ::= 'x' 'y'", "")

    # feed() appends raw characters; the caller supplies token separators
    # (here the space before "y"), matching how decoded LM tokens carry their
    # own leading whitespace.
    assert not synthesizer.is_complete()
    synthesizer.feed("x")
    assert synthesizer.input() == "x"
    assert not synthesizer.is_complete()

    synthesizer.feed(" y")
    assert synthesizer.input() == "x y"
    assert synthesizer.is_complete()

    synthesizer.set_input("x")
    assert synthesizer.input() == "x"
    assert not synthesizer.is_complete()


def test_latest_aufbau_requires_parse_for_dead_prefixes():
    synthesizer = proposition7.Synthesizer("start ::= 'x' 'y'", "")
    synthesizer.set_input("x z")

    try:
        synthesizer.parse()
    except RuntimeError as error:
        assert "no parse found" in str(error)
    else:
        raise AssertionError("dead prefix unexpectedly parsed")
