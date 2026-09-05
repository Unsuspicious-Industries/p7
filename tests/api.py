"""Tests for public API pieces that do not require an HF model."""

import pytest

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
    pytest.importorskip("torch", reason="ConstrainedModel requires the transformers extra")
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


def test_generate_pair_dispatches_both_arms_by_mode(monkeypatch):
    """generate_pair must speak the `mode=` API, not the boolean it replaced.

    A regression guard with a real incident behind it. `_via_runtime` grew a
    three-valued `mode` ("constrained" / "unconstrained" / "mixed") and every
    call site was updated except this one, which kept `constrained=True/False`.
    Nothing caught it: the runtime path is only reachable with a resident
    model, so it never ran in CI, and in production the resulting TypeError
    was swallowed by a blanket `except Exception` and served as a generic
    HTTP 500 with nothing in the logs.

    The assertion is deliberately about the call, not the output: it needs no
    model, no GPU and no weights, which is exactly why it can run everywhere
    the real path cannot.
    """
    from proposition7 import api

    calls = []

    def fake_via_runtime(runtime, model_context, **kwargs):
        calls.append(kwargs)
        return api.Result(text="", complete=False, tokens=0, reason="stub")

    monkeypatch.setattr(api, "_via_runtime", fake_via_runtime)

    api.generate_pair(
        (("user", "hi"),),
        model="stub",
        grammar="A ::= 'x'",
        aufbau_context={},
        max_tokens=4,
        temperature=0.0,
        seed=1,
        runtime=object(),
    )

    assert [call["mode"] for call in calls] == ["constrained", "unconstrained"]
    # The removed spelling must not come back under any value.
    assert all("constrained" not in call for call in calls)
    # Beyond the mode and the stop rule the arms must be identical: same
    # model, device and sampling. `stop_at_complete` asks the grammar whether
    # the program is finished, and only the masked arm has one to ask.
    assert calls[0]["stop_at_complete"] is False
    assert "stop_at_complete" not in calls[1]
    ignore = {"mode", "stop_at_complete"}
    first, second = ({k: v for k, v in c.items() if k not in ignore} for c in calls)
    assert first == second
