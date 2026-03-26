#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT.parent))

import p7

DEFAULT_MODELS = [
    "gpt2",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "Qwen/Qwen3.5-0.5B",
    "Qwen/Qwen3.5-1.5B",
]


@dataclass
class Task:
    task_id: str
    language: str
    grammar: str
    category: str
    instruction: str
    initial: str
    expected: str
    max_tokens: int
    feed: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run p7 benchmark suite.")
    p.add_argument("--tasks", default="stlc,fun,imp,spec", help="Comma list from {stlc,fun,imp,spec}")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma list of HF model ids")
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--max-tasks", type=int, default=0, help="0 means all")
    p.add_argument("--out", default=str(OUT / "raw.jsonl"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--dry", action="store_true")
    return p.parse_args()


def load_tasks(names: list[str]) -> list[Task]:
    out: list[Task] = []
    for name in names:
        path = DATA / f"{name}.csv"
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                out.append(
                    Task(
                        task_id=row["task_id"],
                        language=row["language"],
                        grammar=row["grammar"],
                        category=row["category"],
                        instruction=row["instruction"],
                        initial=row["initial"],
                        expected=row["expected"],
                        max_tokens=int(row["max_tokens"]),
                        feed=row["feed"],
                    )
                )
    return out


def grammar_name(task_grammar: str) -> str:
    if task_grammar in p7.list_grammars():
        return task_grammar
    if task_grammar == "stlc_union" and "stlc" in p7.list_grammars():
        return "stlc"
    return task_grammar


def grammar_prompt(gname: str, feed: str) -> str:
    if feed == "full":
        spec = p7.get_grammar(gname)
        return f"Grammar specification:\n{spec}\n\nGenerate a valid program."
    info = p7.get_grammar_info(gname)
    hints = "\n".join(f"- {h}" for h in info.get("syntax_hints", [])[:5])
    return f"Language: {gname}\nSyntax hints:\n{hints}"


def normalize(s: str) -> str:
    return " ".join(s.split())


def check_parse(spec: str, text: str) -> tuple[bool, bool, str]:
    try:
        g = p7.Grammar(spec)
        s = p7.Synthesizer(g, "")
        s.set_input(text)
        return True, bool(s.is_complete()), ""
    except Exception as e:  # parser/type may raise runtime errors
        return False, False, str(e)


def classify(exact: bool, parse_ok: bool, complete: bool) -> str:
    if exact:
        return "ok"
    if not parse_ok:
        return "parse_error"
    if not complete:
        return "incomplete"
    return "semantic_mismatch"


def run_one(model: p7.ConstrainedModel, task: Task, mode: str) -> dict[str, Any]:
    gname = grammar_name(task.grammar)
    prompt = f"{grammar_prompt(gname, task.feed)}\n\nTask: {task.instruction}\nOutput only program text.\n"
    t0 = time.time()
    if mode == "constrained":
        result = model.until_complete(
            prompt=prompt,
            initial=task.initial,
            max_tokens=task.max_tokens,
            grammar_name=gname,
            greedy_k=1,
            pre_top_k=100,
        )
    else:
        result = model.generate_unconstrained(
            prompt=prompt,
            initial=task.initial,
            max_tokens=task.max_tokens,
            top_k=50,
            temperature=0.8,
            grammar_name=gname,
        )
    sec = time.time() - t0
    spec = p7.get_grammar(gname)
    parse_ok, complete, parse_err = check_parse(spec, result.text)
    exact = normalize(result.text) == normalize(task.expected)
    error = classify(exact, parse_ok, complete)
    return {
        "task_id": task.task_id,
        "language": task.language,
        "grammar": task.grammar,
        "category": task.category,
        "feed": task.feed,
        "mode": mode,
        "expected": task.expected,
        "output": result.text,
        "exact": exact,
        "parse_ok": parse_ok,
        "parse_complete": complete,
        "error": error,
        "parse_error": parse_err,
        "stop_reason": result.stopped_reason,
        "tokens": result.tokens_generated,
        "seconds": sec,
    }


def main() -> None:
    args = parse_args()
    names = [x.strip() for x in args.tasks.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    tasks = load_tasks(names)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry:
        print(f"tasks={len(tasks)} models={len(models)} tries={args.tries}")
        print(f"first_task={tasks[0].task_id if tasks else 'none'}")
        return

    with out_path.open("w", encoding="utf-8") as out:
        for model_name in models:
            for gname in sorted({grammar_name(t.grammar) for t in tasks}):
                model = p7.ConstrainedModel.from_pretrained(
                    model_name,
                    grammar=p7.get_grammar(gname),
                    device=args.device,
                )
                for t in tasks:
                    if grammar_name(t.grammar) != gname:
                        continue
                    for mode in ("constrained", "unconstrained"):
                        for tr in range(args.tries):
                            rec = run_one(model, t, mode)
                            rec["model"] = model_name
                            rec["try"] = tr
                            out.write(json.dumps(rec, ensure_ascii=True) + "\n")
                            out.flush()
                            print(model_name, t.task_id, mode, tr, rec["error"])


if __name__ == "__main__":
    main()
