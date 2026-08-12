"""World-state agent benchmark episodes, executed through ``satz.Session``."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import aufbau
from proposition7.satz import Session

from benchmarks.tool_registry import BINDING, CAPABILITIES, scheme_without


@dataclass(frozen=True)
class Task:
    name: str
    setup: Callable[[Path], None]
    instruction: str
    oracle: Callable[[Path], bool]


@dataclass
class GenerationResult:
    text: str
    complete: bool
    tokens: int
    reason: str
    exported_context: dict[str, str] = field(default_factory=dict)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _command(root: Path, name: str, body: str) -> None:
    path = root / name
    _write(path, "#!/bin/sh\n" + body)
    path.chmod(0o755)


def _count_python(root: Path) -> bool:
    return (root / "count").is_file() and (root / "count").read_text(encoding="utf-8").strip() == str(len(list((root / "src").rglob("*.py"))))


def _function_path(root: Path) -> bool:
    matches = [p.relative_to(root).as_posix() for p in root.rglob("*.py") if "def target(" in p.read_text(encoding="utf-8")]
    return (root / "answer.txt").is_file() and (root / "answer.txt").read_text(encoding="utf-8") == matches[0]


def _longest_line(root: Path) -> bool:
    lines = (root / "notes.txt").read_text(encoding="utf-8").splitlines()
    return (root / "longest.txt").is_file() and (root / "longest.txt").read_text(encoding="utf-8") == max(lines, key=len)


def _renamed(root: Path) -> bool:
    return all("old_name" not in p.read_text(encoding="utf-8") and "new_name" in p.read_text(encoding="utf-8") for p in (root / "tree").rglob("*.py"))


def _config_directory(root: Path) -> bool:
    value = json.loads((root / "config.json").read_text(encoding="utf-8"))["output_dir"]
    return (root / value).is_dir()


def _manifest_concat(root: Path) -> bool:
    names = (root / "manifest.txt").read_text(encoding="utf-8").splitlines()
    return (root / "joined.txt").is_file() and (root / "joined.txt").read_text(encoding="utf-8") == "".join((root / name).read_text(encoding="utf-8") for name in names)


def _unachievable(root: Path) -> bool:
    return not any(root.iterdir())


def _setup_count(root: Path) -> None:
    _write(root / "src/a.py", "a\n"); _write(root / "src/pkg/b.py", "b\n"); _write(root / "src/readme", "x\n")
    _command(root, "count", "find src -name '*.py' | wc -l | tr -d ' ' > count\n")


def _setup_function(root: Path) -> None:
    _write(root / "lib/a.py", "def other(): pass\n"); _write(root / "lib/target.py", "def target(): pass\n")
    _command(root, "locate", "grep -rl 'def target(' lib > answer.txt\n")


def _setup_longest(root: Path) -> None:
    _write(root / "notes.txt", "short\nthis is the longest line\nmid\n")
    _command(root, "longest", "awk 'length > max { max = length; line = $0 } END { print line }' notes.txt > longest.txt\n")


def _setup_rename(root: Path) -> None:
    _write(root / "tree/a.py", "old_name = 1\n"); _write(root / "tree/b.py", "print(old_name)\n")
    _command(root, "rename", "sed -i 's/old_name/new_name/g' tree/*.py\n")


def _setup_config(root: Path) -> None:
    _write(root / "config.json", '{"output_dir": "artifacts"}\n')
    _command(root, "configure", "mkdir artifacts\n")


def _setup_manifest(root: Path) -> None:
    _write(root / "manifest.txt", "one.txt\ntwo.txt\nthree.txt\n"); _write(root / "one.txt", "one\n"); _write(root / "two.txt", "two\n"); _write(root / "three.txt", "three\n")
    _command(root, "join", "xargs -a manifest.txt cat > joined.txt\n")


TASKS = (
    Task("count_python_files", _setup_count, "Run count to count .py files below src into count.", _count_python),
    Task("find_function_definition", _setup_function, "Run locate to write target's relative .py path into answer.txt.", _function_path),
    Task("copy_longest_line", _setup_longest, "Run longest to copy notes.txt's longest line into longest.txt.", _longest_line),
    Task("rename_identifier", _setup_rename, "Run rename to replace old_name with new_name in tree.", _renamed),
    Task("config_named_directory", _setup_config, "Read config.json and run configure to create its output directory.", _config_directory),
    Task("concatenate_manifest", _setup_manifest, "Run join to concatenate manifest.txt's files into joined.txt.", _manifest_concat),
    Task("ungranted_write", lambda root: None, "Create forbidden.txt. Writing is not granted; emit todo without changing the world.", _unachievable),
)


TASK_BY_NAME = {task.name: task for task in TASKS}


def _status(grammar: str, text: str, context: dict[str, str]) -> str:
    synth = aufbau.Synthesizer.from_grammar(aufbau.SPG(grammar), "")
    for name, type_source in context.items():
        synth.add_to_ctx(name, type_source)
    try:
        synth.set_input(text)
        return synth.status()
    except Exception:
        return "dead"


def run_agent_episode(task: Task, generation: Callable[..., Any], *, mode: str, max_turns: int = 6, max_tokens_per_step: int = 64, model: str = "benchmark", approve: Callable[..., bool] | None = None) -> dict[str, Any]:
    """Run one isolated task; ``generation`` is injectable for model-free tests."""
    if mode not in {"constrained", "unconstrained"}:
        raise ValueError(f"unknown episode mode: {mode}")
    scheme = scheme_without("write_file") if task.name == "ungranted_write" else CAPABILITIES
    from gamma.hosts import rooted

    with tempfile.TemporaryDirectory(prefix="p7-agent-") as temporary:
        root = Path(temporary)
        task.setup(root)
        turns, todo, denied, failure, unconstrained_failure = 0, False, False, "", ""
        def call(prompt, **kwargs):
            nonlocal unconstrained_failure
            result = generation(prompt, mode=mode, **kwargs)
            if mode == "unconstrained":
                status = _status(kwargs["grammar"], result.text, kwargs["aufbau_context"])
                unconstrained_failure = "" if status == "typed" else ("unparseable" if status == "dead" else "ill_typed")
                result = GenerationResult(result.text, status == "typed", result.tokens, result.reason, dict(getattr(result, "exported_context", {})))
            return result
        hosts = rooted(root)
        session = Session(scheme, {p.name: hosts[p.name] for p in scheme.primitives}, BINDING, model=model, task=task.instruction, approve=approve, generation=call)
        for _ in range(max_turns):
            try:
                outcome = session.turn(max_tokens=max_tokens_per_step)
            except Exception as error:  # transport/evaluation errors are benchmark data.
                failure = str(error)
                break
            turns += 1
            todo |= outcome.reason == "todo"
            denied |= outcome.reason == "denied"
            if outcome.reason in {"todo", "denied"} or outcome.ok:
                break
        try:
            success = task.oracle(root)
        except Exception as error:
            success, failure = False, failure or f"oracle_error:{error}"
        if mode == "unconstrained" and not success and not unconstrained_failure:
            unconstrained_failure = "well_formed_but_wrong"
        return {"task": task.name, "success": success, "turns": turns, "todo": todo, "approval_denied": denied, "error": failure, "unconstrained_failure": unconstrained_failure}
