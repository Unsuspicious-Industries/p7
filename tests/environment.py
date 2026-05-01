import p7


class FakeReasoningModel:
    def __init__(self):
        self.think_prompts = []
        self.formal_calls = []

    def think_open(self):
        return "<think>"

    def think_close(self):
        return "</think>"

    def allow_system_prompt(self):
        return False

    def stop_tokens_unconstrained(self, grammar_name=None):
        return ["</think>"]

    def generate_unconstrained(self, prompt, **kwargs):
        self.think_prompts.append((prompt, kwargs))
        return p7.GenerationResult(
            text="I will solve it",
            is_complete=False,
            tokens_generated=3,
            stopped_reason="max_tokens",
        )

    def generate_constrained(self, prompt, initial="", **kwargs):
        self.formal_calls.append((prompt, initial, kwargs))
        return p7.GenerationResult(
            text=initial + "x",
            is_complete=bool(kwargs.get("stop_on_complete", False)),
            tokens_generated=1,
            stopped_reason=(
                "complete" if kwargs.get("stop_on_complete", False) else "max_tokens"
            ),
        )


def test_reasoning_environment_uses_one_think_and_one_formal_block():
    model = FakeReasoningModel()
    env = p7.ReasoningEnvironment(
        model,
        grammar_name="stlc",
        think_budget=8,
        formal_budget=8,
        stop_on_complete=True,
    )

    result = env.generate("task", initial="λx:Int.")

    assert result.stopped_reason == "complete"
    assert result.total_tokens == 4
    assert result.think_tokens == 3
    assert result.formal_tokens == 1
    assert result.all_thoughts == "I will solve it"
    assert result.final_output.content == "λx:Int.x"
    assert model.think_prompts[0][0] == "task\n<think>"
    assert "λx:Int." not in model.think_prompts[0][0]
    assert model.formal_calls[0][0].endswith("\n<formal>")
    assert "<think>I will solve it</think>" in model.formal_calls[0][0]
    assert model.formal_calls[0][1] == "λx:Int."
    assert [str(block) for block in result.blocks] == [
        "<think>I will solve it</think>",
        "<formal>λx:Int.x</formal>",
    ]


def test_reasoning_environment_uses_model_think_start_token_once():
    model = FakeReasoningModel()
    model.start_tokens = ["<think>"]

    def start_tokens_unconstrained(grammar_name=None):
        return model.start_tokens

    model.start_tokens_unconstrained = start_tokens_unconstrained
    env = p7.ReasoningEnvironment(
        model,
        grammar_name="stlc",
        formal_budget=8,
        stop_on_complete=True,
    )

    result = env.generate("task", initial="λx:Int.")

    assert result.stopped_reason == "complete"
    assert model.think_prompts[0][0] == "task"
    assert result.final_output.content == "λx:Int.x"
