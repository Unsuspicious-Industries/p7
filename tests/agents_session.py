from types import SimpleNamespace

import aufbau

from proposition7.agentic import Tool, ToolRegistry, AgentSession

_REGISTRY = ToolRegistry(
    [
        Tool.unary("search", "string", "docs", lambda arg: f"docs_about_{arg}"),
        Tool.unary("summarize", "docs", "string", lambda arg: f"summary_of_{arg}"),
        Tool.unary("count", "docs", "int", lambda arg: 3),
        Tool.unary("format", "int", "string", lambda arg: f"formatted_{arg}"),
    ]
)


def _fake_think(text="thinking"):
    def think(**kwargs):
        return SimpleNamespace(text=text, tokens_generated=3)

    return think


def _scripted_generate_against_real_aufbau(spg, steps):
    """A test double that validates each canned step against a real aufbau
    Synthesizer (not a fake status string) -- so a type-mismatched step is
    genuinely rejected by the engine, exactly as it would be mid-decode."""

    def generate(prompt, initial, max_tokens, seed, context):
        synth = aufbau.Synthesizer.from_grammar(spg, "")
        for name, ty in context.items():
            synth.add_to_ctx(name, ty)
        text = steps[generate.i]
        generate.i += 1
        try:
            synth.set_input(text)
            status = synth.status()
        except Exception:
            status = "dead"
        return SimpleNamespace(text=text, is_complete=(status == "typed"), stopped_reason=status)

    generate.i = 0
    return generate


def test_episode_succeeds_and_grades_the_executed_return_value():
    spg = aufbau.SPG(_REGISTRY.spec(root="step", syntax="let_c"))
    steps = [
        'let r = search("agents");',
        "let s = summarize(r);",
        "return s;",
    ]
    session = AgentSession(model=None, registry=_REGISTRY, task="t", syntax="let_c", max_turns=6)
    result = session.run_to_completion(
        think=_fake_think(), generate=_scripted_generate_against_real_aufbau(spg, steps)
    )

    assert result.success is True
    assert result.return_value == "summary_of_docs_about_agents"
    assert result.final_context.types == {"r": "docs", "s": "string"}
    assert result.final_context.values == {
        "r": "docs_about_agents",
        "s": "summary_of_docs_about_agents",
    }
    assert all(turn.think_text == "thinking" for turn in result.turns)


def test_episode_rejects_a_type_mismatched_step():
    spg = aufbau.SPG(_REGISTRY.spec(root="step", syntax="let_c"))
    steps = [
        'let r = search("agents");',
        "let n = format(r);",  # format wants int, r is docs
    ]
    session = AgentSession(model=None, registry=_REGISTRY, task="t", syntax="let_c", max_turns=6)
    result = session.run_to_completion(
        think=_fake_think(), generate=_scripted_generate_against_real_aufbau(spg, steps)
    )

    assert result.success is False
    assert "incomplete_step" in result.reason


def test_episode_fails_on_turn_budget_exhaustion():
    spg = aufbau.SPG(_REGISTRY.spec(root="step", syntax="let_c"))
    steps = ['let r = search("agents");'] * 3
    session = AgentSession(model=None, registry=_REGISTRY, task="t", syntax="let_c", max_turns=3)
    result = session.run_to_completion(
        think=_fake_think(), generate=_scripted_generate_against_real_aufbau(spg, steps)
    )

    assert result.success is False
    assert result.reason == "max_turns_exceeded:3"
    assert len(result.turns) == 3


def test_degenerate_immediate_return_is_not_conflated_with_task_success():
    # A one-turn "return <literal>;" episode is mechanically successful
    # (session.success reflects reaching a well-typed return), but its
    # executed value is exactly the literal -- callers grade task success
    # by comparing return_value to an expected value, not by `success`
    # alone (lmpl-plan.md section 5.2).
    spg = aufbau.SPG(_REGISTRY.spec(root="step", syntax="let_c"))
    session = AgentSession(model=None, registry=_REGISTRY, task="t", syntax="let_c", max_turns=6)
    result = session.run_to_completion(
        think=_fake_think(),
        generate=_scripted_generate_against_real_aufbau(spg, ['return "hello";']),
    )

    assert result.success is True
    assert result.return_value == "hello"


def test_session_uses_the_models_own_think_tags():
    # A model family with reasoning tags other than Qwen's <think> must see
    # its OWN tags in both the think-phase prompt and the step transcript --
    # hardcoded <think> scaffolding killed every non-Qwen agent episode at
    # turn 0 in the first collection run.
    class TaggedModel:
        def __init__(self):
            self.think_prompts = []
            self.think_stop_tokens = []
            self.step_prompts = []

        def think_open(self):
            return "<reason>"

        def think_close(self):
            return "</reason>"

        def stop_tokens_unconstrained(self, grammar_name=None):
            return ["<|end|>"]

        def generate_unconstrained(
            self, prompt, initial, max_tokens, temperature, stop_tokens, grammar_name
        ):
            self.think_prompts.append(prompt)
            self.think_stop_tokens.append(list(stop_tokens))
            return SimpleNamespace(text="pondering</reason> junk", tokens_generated=2)

        def generate_constrained(self, grammar_name, prompt, initial, max_tokens, seed, context):
            self.step_prompts.append(prompt)
            return SimpleNamespace(
                text='return "done";', is_complete=True, stopped_reason="complete"
            )

    model = TaggedModel()
    session = AgentSession(model=model, registry=_REGISTRY, task="t", syntax="let_c", max_turns=3)
    result = session.run_to_completion()

    assert result.success is True
    assert model.think_prompts[0].endswith("<reason>")
    assert "</reason>" in model.think_stop_tokens[0]
    assert result.turns[0].think_text == "pondering"
    assert "<reason>pondering</reason>" in model.step_prompts[0]
    assert "<think>" not in model.step_prompts[0]


def test_step_raises_once_session_is_done():
    import pytest

    spg = aufbau.SPG(_REGISTRY.spec(root="step", syntax="let_c"))
    session = AgentSession(model=None, registry=_REGISTRY, task="t", syntax="let_c", max_turns=6)
    session.run_to_completion(
        think=_fake_think(),
        generate=_scripted_generate_against_real_aufbau(spg, ['return "x";']),
    )
    assert session.done is True
    with pytest.raises(RuntimeError):
        session.step(think=_fake_think(), generate=lambda **kw: None)
