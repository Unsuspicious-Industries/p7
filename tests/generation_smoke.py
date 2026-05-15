"""Smoke tests that run real constrained generation with gpt2 on CPU.

These tests verify the full generation stack (model loading, grammar
constraining, token sampling, and output validation) without needing a GPU.
gpt2 is small enough (~500 MB) to load on a GitHub Actions runner.
"""

import proposition7
import pytest


GRAMMARS = ["stlc", "fun", "imp", "toy"]

TASKS = {
    "stlc": {
        "prompt": "Define the identity function.",
        "initial": "λx:Int.",
        "max_tokens": 32,
    },
    "fun": {
        "prompt": "Define inc:Int->Int and call it on 1.",
        "initial": "let inc: Int -> Int = (n: Int) =>",
        "max_tokens": 32,
    },
    "imp": {
        "prompt": "Declare a variable x with value 5.",
        "initial": "{ let x: Int =",
        "max_tokens": 32,
    },
    "toy": {
        "prompt": "Produce a typed value.",
        "initial": "",
        "max_tokens": 16,
    },
}


@pytest.fixture(scope="module")
def models():
    """Load gpt2 once per grammar for the entire test module."""
    loaded = {}
    for name in GRAMMARS:
        grammar = proposition7.get_grammar(name)
        model = proposition7.ConstrainedModel.from_pretrained(
            "gpt2",
            grammar=grammar,
            device="cpu",
        )
        loaded[name] = model
    return loaded


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_constrained_generation_parses(grammar, models):
    """Constrained generation must produce output that parses under the grammar."""
    model = models[grammar]
    task = TASKS[grammar]

    result = model.generate_constrained(
        prompt=task["prompt"],
        initial=task["initial"],
        max_tokens=task["max_tokens"],
        seed=42,
    )

    assert result.text, f"{grammar}: empty output"
    assert result.text.startswith(task["initial"]) or not task["initial"], (
        f"{grammar}: output {result.text!r} does not start with initial {task['initial']!r}"
    )
    assert result.tokens_generated > 0, f"{grammar}: no tokens generated"
    assert result.stopped_reason in (
        "complete",
        "max_tokens",
        "no_valid",
    ) or result.stopped_reason.startswith("stop_token:"), (
        f"{grammar}: unexpected stop reason: {result.stopped_reason}"
    )

    spec = proposition7.get_grammar(grammar)
    s = proposition7.Synthesizer(spec, "")
    s.set_input(result.text)
    try:
        s.parse()
    except RuntimeError as e:
        pytest.fail(f"{grammar}: generated text does not parse: {e}\nText: {result.text!r}")


def test_unconstrained_generation_produces_text(models):
    """Unconstrained generation must produce non-empty text."""
    model = models["toy"]

    result = model.generate_unconstrained(
        prompt="Produce a typed value.",
        max_tokens=16,
        seed=42,
    )

    assert result.text, "unconstrained: empty output"
    assert result.tokens_generated > 0, "unconstrained: no tokens generated"


def test_high_level_generate_helper():
    """The convenience proposition7.generate() helper must work end-to-end."""
    result = proposition7.generate(
        "Define the identity function.",
        model="gpt2",
        grammar="stlc",
        initial="λx:Int.",
        max_tokens=16,
        device="cpu",
        seed=42,
    )

    assert result.text, "generate helper: empty output"
    assert result.complete or result.text.startswith("λx:Int.")
    assert result.tokens > 0
