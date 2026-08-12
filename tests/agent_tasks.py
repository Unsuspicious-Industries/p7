from pathlib import Path

from benchmarks.agent import GenerationResult, TASK_BY_NAME, run_agent_episode


class ScriptedGeneration:
    def __init__(self, turns):
        self.turns = iter(turns)

    def __call__(self, prompt, **kwargs):
        del prompt, kwargs
        return GenerationResult(next(self.turns), True, 1, "complete", {})


def test_episode_world_oracle_accepts_accomplished_task():
    result = run_agent_episode(
        TASK_BY_NAME["count_python_files"],
        ScriptedGeneration(['x = run("count");']),
        mode="constrained",
    )
    assert result == {"task": "count_python_files", "success": True, "turns": 1, "todo": False, "approval_denied": False, "error": "", "unconstrained_failure": ""}


def test_episode_todo_leaves_unachievable_world_unchanged():
    result = run_agent_episode(TASK_BY_NAME["ungranted_write"], ScriptedGeneration(["x = todo;"]), mode="constrained")
    assert result == {"task": "ungranted_write", "success": True, "turns": 1, "todo": True, "approval_denied": False, "error": "", "unconstrained_failure": ""}


def test_episode_world_oracle_rejects_wrong_action():
    result = run_agent_episode(
        TASK_BY_NAME["count_python_files"],
        ScriptedGeneration(['x = run("wrong");']),
        mode="constrained",
    )
    assert result["success"] is False
