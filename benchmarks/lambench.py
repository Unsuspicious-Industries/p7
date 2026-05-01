#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import p7


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "lambench"


@dataclass(frozen=True)
class LambenchTest:
    expr: str
    want: str


@dataclass(frozen=True)
class LambenchTask:
    task_id: str
    description: str
    tests: list[LambenchTest]


def parse_task(path: Path) -> LambenchTask:
    text = path.read_text(encoding="utf-8")
    sections = text.split("\n---\n")
    if len(sections) != 2:
        raise ValueError(f"{path.name}: expected description and tests")

    lines = [line.strip() for line in sections[1].splitlines() if line.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"{path.name}: tests must be expression/expected pairs")

    tests: list[LambenchTest] = []
    for index in range(0, len(lines), 2):
        want = lines[index + 1]
        if not want.startswith("= "):
            raise ValueError(f"{path.name}: expected '= ...' after test expression")
        tests.append(LambenchTest(expr=lines[index], want=want[2:]))

    return LambenchTask(
        task_id=path.stem,
        description=sections[0].strip(),
        tests=tests,
    )


def load_tasks(tasks_dir: Path, *, filter_text: str = "") -> list[LambenchTask]:
    paths = sorted(tasks_dir.glob("*.tsk"))
    if filter_text:
        paths = [path for path in paths if filter_text in path.stem]
    return [parse_task(path) for path in paths]


def build_prompt(task: LambenchTask) -> str:
    tests = "\n\n".join(f"{test.expr}\n= {test.want}" for test in task.tests)
    return f"""You are solving one Lambench task using Lamb, a pure lambda-calculus language.

Return exactly one .lam program and nothing else.
The program must define @main as the final top-level definition.
You may define helper functions with @name = term before @main.

Lamb syntax:
- top-level definition: @name = term
- lambda: λname.term
- reference: @name
- application: f(x,y,z), meaning (((f x) y) z)
- grouping: (term)
- names use ASCII letters, digits, and underscore only

Task id: {task.task_id}

Description:
{task.description}

Tests:
{tests}
""".strip()


def extract_submission(text: str) -> str:
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1]
            if "\n" in stripped:
                first, rest = stripped.split("\n", 1)
                stripped = rest if first.strip().isalnum() else stripped
    lines = stripped.splitlines()
    first_definition = next(
        (index for index, line in enumerate(lines) if line.strip().startswith("@")),
        0,
    )
    return "\n".join(lines[first_definition:]).strip()


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Lambench submissions with p7.")
    parser.add_argument("--tasks-dir", required=True, help="Path to Lambench tsk/ directory")
    parser.add_argument("--models", default="gpt2", help="Comma-separated HF model ids")
    parser.add_argument("--mode", choices=["constrained", "unconstrained"], default="constrained")
    parser.add_argument("--filter", default="", help="Substring filter over task ids, e.g. cnat_")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--dry", action="store_true")
    return parser.parse_args()


def make_generator(args: argparse.Namespace, model_name: str):
    return p7.get_model_class(model_name).from_pretrained(
        model_name,
        grammar=p7.get_grammar("lamb"),
        device=args.device,
    )


def generate_task(generator, task: LambenchTask, args: argparse.Namespace):
    prompt = build_prompt(task)
    if args.mode == "constrained":
        return generator.generate_constrained(
            prompt=prompt,
            max_tokens=args.max_tokens,
            grammar_name="lamb",
            stop_on_complete=False,
        )
    return generator.generate_unconstrained(
        prompt=prompt,
        max_tokens=args.max_tokens,
        top_k=50,
        temperature=0.8,
        grammar_name="lamb",
    )


def main() -> None:
    args = parse_args()
    tasks = load_tasks(Path(args.tasks_dir), filter_text=args.filter)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    models = [model.strip() for model in args.models.split(",") if model.strip()]

    if args.dry:
        print(f"tasks={len(tasks)} models={len(models)} backend=local mode={args.mode}")
        print(f"first_task={tasks[0].task_id if tasks else 'none'}")
        return

    out_dir = Path(args.out_dir)
    rows: list[dict[str, object]] = []
    for model_name in models:
        generator = make_generator(args, model_name)
        model_dir = out_dir / model_name.replace("/", "__") / args.mode
        model_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            result = generate_task(generator, task, args)
            submission = extract_submission(result.text)
            output_path = model_dir / f"{task.task_id}.lam"
            output_path.write_text(submission + "\n", encoding="utf-8")
            row = {
                "model": model_name,
                "backend": "local",
                "mode": args.mode,
                "task_id": task.task_id,
                "output_path": str(output_path),
                "tokens": result.tokens_generated,
                "stop_reason": result.stopped_reason,
                "parse_complete": result.is_complete,
            }
            rows.append(row)
            print(model_name, task.task_id, result.stopped_reason, output_path)

    write_jsonl(out_dir / "raw.jsonl", rows)


if __name__ == "__main__":
    main()
