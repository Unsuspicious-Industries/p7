import pytest

import p7.sampler as sampler_module


class FakeGrammar:
    def __init__(self, spec):
        self.spec = spec


class FakeSynthesizer:
    def __init__(self, grammar, input_text):
        self.grammar = grammar
        self.input_text = input_text
        self.set_input_calls = []
        self.extend_calls = []

    def set_input(self, text):
        self.set_input_calls.append(text)
        self.input_text = text

    def extend(self, token):
        self.extend_calls.append(token)
        if token == "bad":
            raise RuntimeError("Parse failed for input='seed' token='bad'")
        self.input_text += token
        return True

    def get_completions(self):
        return []

    def current_text(self):
        return self.input_text

    def is_complete(self):
        return False


def test_feed_keeps_existing_state_when_parse_fails(monkeypatch):
    monkeypatch.setattr(sampler_module, "Grammar", FakeGrammar)
    monkeypatch.setattr(sampler_module, "Synthesizer", FakeSynthesizer)

    sampler = sampler_module.TypedSampler(
        grammar="fake",
        vocab=["bad", "good"],
        logit_fn=lambda: [1.0, 0.0],
    )

    sampler.set_input("seed")
    assert sampler.synthesizer.set_input_calls == ["seed"]

    sampler.feed("good")
    assert sampler.synthesizer.extend_calls == ["good"]
    assert sampler.current_text() == "seedgood"

    with pytest.raises(RuntimeError, match="Parse failed"):
        sampler.feed("bad")

    assert sampler.current_text() == "seedgood"


def test_dbg_reads_environment_after_import(monkeypatch, capsys):
    monkeypatch.delenv("P7_CONSTRAINED_DEBUG", raising=False)
    monkeypatch.delenv("P7_SAMPLER_DEBUG", raising=False)
    monkeypatch.setattr(sampler_module, "_DEBUG_OVERRIDE", None)
    monkeypatch.setattr(sampler_module, "DEBUG_SAMPLER", False)

    sampler_module._dbg("before")
    assert capsys.readouterr().err == ""

    monkeypatch.setenv("P7_CONSTRAINED_DEBUG", "1")
    sampler_module._dbg("after {}", "env")
    captured = capsys.readouterr().err

    assert "[p7-debug] after env" in captured


def test_dbg_survives_bad_format_string(monkeypatch, capsys):
    monkeypatch.setattr(sampler_module, "_DEBUG_OVERRIDE", True)
    monkeypatch.setattr(sampler_module, "DEBUG_SAMPLER", True)

    sampler_module._dbg("literal braces {oops}")
    captured = capsys.readouterr().err

    assert "literal braces {oops}" in captured
    assert "debug-format-error" in captured