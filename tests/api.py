"""Tests for public API pieces that do not require an HF model."""

from pathlib import Path

import proposition7
import proposition7.grammars as bundled_grammars
from proposition7.api import Result, _resolve_grammar
from proposition7.inference import GenerationResult


def test_result_dataclass():
    result = Result(text="hello", complete=True, tokens=3, reason="complete")
    assert result.text == "hello" and result.complete and result.thoughts == ""


def test_generation_result_dataclass():
    result = GenerationResult(
        text="λx:Int.x",
        is_complete=True,
        tokens_generated=3,
        stopped_reason="complete",
        step_token_ids=[1, 2, 3],
        step_pre_entropies=[1.5, 1.2, 0.8],
        step_entropies=[0.5, 0.3, 0.1],
        step_retries=[0, 1, 0],
    )
    assert result.text == "λx:Int.x"
    assert result.is_complete is True
    assert result.tokens_generated == 3
    assert result.stopped_reason == "complete"
    assert len(result.step_token_ids) == 3


def test_resolve_grammar_name():
    assert len(_resolve_grammar("stlc")) > 50


def test_resolve_grammar_raw_spec():
    spec = "A ::= 'x'"
    assert _resolve_grammar(spec) == spec


def test_all_grammars_parseable():
    for name in proposition7.list_grammars():
        synthesizer = proposition7.Synthesizer(proposition7.get_grammar(name), "")
        synthesizer.parse()


def test_builtin_grammar_list_matches_readme():
    assert set(proposition7.list_grammars()) == {"stlc", "fun", "imp", "toy"}


def test_grammars_are_bundled_inside_public_package():
    grammar_dir = Path(bundled_grammars.__file__).resolve().parent

    assert grammar_dir.name == "grammars"
    assert grammar_dir.parent.name == "proposition7"
    assert (grammar_dir / "fun.auf").exists()


def test_proposition7_exports_public_api():
    assert proposition7.ConstrainedModel is not None
    assert proposition7.generate is not None
    assert proposition7.get_grammar is not None
    assert proposition7.list_grammars is not None
    assert proposition7.Synthesizer is not None
    assert proposition7.GenerationResult is not None
    assert proposition7.ReasoningEnvironment is not None
    assert proposition7.Session is not None
    assert proposition7.Result is not None
    assert proposition7.BENCHMARK_MODES is not None


def test_synthesizer_set_input_and_feed_round_trip():
    synthesizer = proposition7.Synthesizer("start ::= 'x' 'y'", "")

    assert not synthesizer.is_complete()
    synthesizer.feed("x")
    assert synthesizer.input() == "x"
    assert not synthesizer.is_complete()

    synthesizer.feed("y")
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


def test_stlc_grammar_accepts_identity():
    spec = proposition7.get_grammar("stlc")
    s = proposition7.Synthesizer(spec, "")
    s.set_input("λx:Int.x")
    s.parse()
    assert s.is_complete()


def test_stlc_grammar_accepts_application():
    spec = proposition7.get_grammar("stlc")
    s = proposition7.Synthesizer(spec, "")
    s.set_input("λf:(Int->Bool).λx:Int.(f x)")
    s.parse()
    assert s.is_complete()


def test_stlc_grammar_rejects_untyped_lambda():
    spec = proposition7.get_grammar("stlc")
    s = proposition7.Synthesizer(spec, "")
    s.set_input("λx.x")
    try:
        s.parse()
    except RuntimeError:
        pass
    else:
        raise AssertionError("untyped lambda unexpectedly parsed")


def test_fun_grammar_accepts_let_binding():
    spec = proposition7.get_grammar("fun")
    s = proposition7.Synthesizer(spec, "")
    s.set_input("let x: Int = 1; x + 2")
    s.parse()
    assert s.is_complete()


def test_fun_grammar_accepts_lambda():
    spec = proposition7.get_grammar("fun")
    s = proposition7.Synthesizer(spec, "")
    s.set_input("(x: Int) => x + 1")
    s.parse()
    assert s.is_complete()


def test_imp_grammar_accepts_block():
    spec = proposition7.get_grammar("imp")
    s = proposition7.Synthesizer(spec, "")
    s.set_input("{ let x: Int = 5; }")
    s.parse()
    assert s.is_complete()


def test_toy_grammar_accepts_value():
    spec = proposition7.get_grammar("toy")
    s = proposition7.Synthesizer(spec, "")
    s.set_input("beep:Fizz")
    s.parse()
    assert s.is_complete()


def test_toy_grammar_accepts_concatenation():
    spec = proposition7.get_grammar("toy")
    s = proposition7.Synthesizer(spec, "")
    s.set_input("beep:Fizz + boop:Fizz")
    s.parse()
    assert s.is_complete()


def test_grammar_info_returns_required_fields():
    for name in proposition7.list_grammars():
        info = proposition7.get_grammar_info(name)
        assert "spec" in info
        assert "name" in info
        assert "short" in info
        assert "description" in info
        assert "summary" in info
        assert "syntax_hints" in info
        assert "examples" in info


def test_grammar_summary_is_nonempty():
    for name in proposition7.list_grammars():
        summary = proposition7.get_grammar_summary(name)
        assert len(summary) > 10


def test_get_grammar_raises_on_unknown():
    try:
        proposition7.get_grammar("nonexistent")
    except ValueError as e:
        assert "Unknown grammar" in str(e)
        assert "nonexistent" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown grammar")


def test_constrained_model_has_required_methods():
    model_cls = proposition7.ConstrainedModel
    assert hasattr(model_cls, "from_pretrained")
    assert hasattr(model_cls, "generate_constrained")
    assert hasattr(model_cls, "generate_unconstrained")
    assert hasattr(model_cls, "stop_tokens")
    assert hasattr(model_cls, "stop_tokens_constrained")
    assert hasattr(model_cls, "stop_tokens_unconstrained")
    assert hasattr(model_cls, "start_tokens_constrained")
    assert hasattr(model_cls, "start_tokens_unconstrained")
    assert hasattr(model_cls, "format_prompt")
    assert hasattr(model_cls, "think_open")
    assert hasattr(model_cls, "think_close")


def test_environment_build_system_prompt():
    prompt = proposition7.build_system_prompt("stlc")
    assert "typed" in prompt.lower()
    assert "<think>" in prompt
    assert "</think>" in prompt
    assert "<formal>" in prompt


def test_environment_build_task_prompt_constrained_direct():
    prompt = proposition7.build_task_prompt("stlc", "Write identity", mode="constrained_direct")
    assert "Write identity" in prompt
    assert "Write the completed program directly" in prompt


def test_environment_build_task_prompt_constrained_mixed():
    prompt = proposition7.build_task_prompt("stlc", "Write identity", mode="constrained_mixed")
    assert "think briefly" in prompt.lower()
    assert "write the final program" in prompt.lower()


def test_benchmark_modes_contains_expected_values():
    assert "constrained_direct" in proposition7.BENCHMARK_MODES
    assert "constrained_mixed" in proposition7.BENCHMARK_MODES
    assert "unconstrained" in proposition7.BENCHMARK_MODES
    assert "outlines" in proposition7.BENCHMARK_MODES
    assert "outlines_mixed" in proposition7.BENCHMARK_MODES
