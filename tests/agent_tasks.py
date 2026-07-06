from types import SimpleNamespace

import aufbau
import pytest

from benchmarks.api import _validate_task_resolution, load_tasks, run_agent_interaction

# Canonical well-typed steps for each builtin agent task, mirroring exactly
# what the task's prompt asks for -- used to verify the whole pipeline
# (task loading -> AgentSession -> real aufbau validation -> episode
# grading) end to end, not just that imports succeed.
_CANNED_STEPS = {
    "agent_search_summarize": ['let r = search("agents");', "let s = summarize(r);", "return s;"],
    "agent_search_count_format": [
        'let r = search("agents");',
        "let n = count(r);",
        "let s = format(n);",
        "return s;",
    ],
    "agent_direct_return": ['return "done";'],
    "agent_search_only": ['let r = search("docs");', "return r;"],
    "agent_reports_count_format": [
        'let r = search("reports");',
        "let n = count(r);",
        "let s = format(n);",
        "return s;",
    ],
    "agent_count_as_int": ['let r = search("weather");', "let n = count(r);", "return n;"],
    "agent_papers_summarize": ['let r = search("papers");', "let s = summarize(r);", "return s;"],
    "agent_format_literal": ["let s = format(42);", "return s;"],
}


class _ScriptedModel:
    """Validates each canned step against a real aufbau Synthesizer -- not
    a fake status string -- exactly like the constrained decode loop would."""

    def __init__(self, steps):
        self.steps = steps
        self.i = 0

    def generate_constrained(self, prompt, initial, max_tokens, grammar_name, seed, context):
        spg = aufbau.SPG(grammar_name)
        synth = aufbau.Synthesizer.from_grammar(spg, "")
        for name, ty in context.items():
            synth.add_to_ctx(name, ty)
        text = self.steps[self.i]
        self.i += 1
        try:
            synth.set_input(text)
            status = synth.status()
        except Exception:
            status = "dead"
        return SimpleNamespace(
            text=text, is_complete=(status == "typed"), stopped_reason=status, tokens_generated=5
        )

    def generate_unconstrained(self, prompt, initial, max_tokens, temperature=0.0, stop_tokens=None, grammar_name=None, seed=None):
        return SimpleNamespace(text="", tokens_generated=0, stopped_reason="max_tokens")

    def stop_tokens_unconstrained(self, grammar_name=None):
        return []


def _agent_tasks():
    return {t.task_id: t for t in load_tasks(["all"]) if t.kind == "agent"}


def test_eight_builtin_agent_tasks_are_loaded():
    tasks = _agent_tasks()
    assert len(tasks) >= 8
    assert all(t.resolution.get("mode") == "episode" for t in tasks.values())
    assert all("expected_value" in t.resolution for t in tasks.values())


@pytest.mark.parametrize("task_id", list(_CANNED_STEPS.keys()))
def test_canned_correct_episode_passes_for_every_builtin_task(task_id):
    task = _agent_tasks()[task_id]
    model = _ScriptedModel(_CANNED_STEPS[task_id])
    record = run_agent_interaction(model, task, "constrained_direct", seed=0)

    assert record["passed"] is True
    assert record["return_value"] == task.resolution["expected_value"]


def test_wrong_value_episode_is_correctly_rejected():
    task = _agent_tasks()["agent_direct_return"]
    model = _ScriptedModel(['return "wrong";'])
    record = run_agent_interaction(model, task, "constrained_direct", seed=0)

    assert record["passed"] is False
    assert record["episode_success"] is True  # mechanically reached a well-typed return
    assert record["return_value"] == "wrong"  # but the wrong one -- caught by value grading


def test_type_mismatched_step_fails_the_episode():
    task = _agent_tasks()["agent_search_summarize"]
    model = _ScriptedModel(['let r = search("agents");', "let n = format(r);"])  # format wants int
    record = run_agent_interaction(model, task, "constrained_direct", seed=0)

    assert record["passed"] is False
    assert record["episode_success"] is False


def test_unconstrained_arm_rejects_gibberish_via_the_same_checker():
    class GibberishModel:
        def generate_unconstrained(self, prompt, initial, max_tokens, temperature=0.0, stop_tokens=None, grammar_name=None, seed=None):
            return SimpleNamespace(text="I will now search the web!!!", tokens_generated=10, stopped_reason="max_tokens")

        def stop_tokens_unconstrained(self, grammar_name=None):
            return []

    task = _agent_tasks()["agent_search_summarize"]
    record = run_agent_interaction(GibberishModel(), task, "unconstrained", seed=0)

    assert record["passed"] is False
    assert record["episode_success"] is False


def test_agent_task_validation_requires_episode_mode_and_expected_value():
    with pytest.raises(ValueError, match="episode"):
        _validate_task_resolution(
            "some/path.toml", "bad_task", "tool", {"mode": "type"}, kind="agent"
        )
    with pytest.raises(ValueError, match="expected_value"):
        _validate_task_resolution(
            "some/path.toml", "bad_task", "tool", {"mode": "episode"}, kind="agent"
        )
