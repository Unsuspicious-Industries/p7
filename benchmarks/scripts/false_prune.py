#!/usr/bin/env python3
"""False-prune rate (paper section 5.1): for every character-level prefix of
every valid, differentially-certified corpus program (ml against ocamlc, c
against cc), aufbau must never answer "dead". A single violation would mean
the mask can reject a token sequence that leads to a program a real compiler
accepts -- unconditional soundness broken. Pure CPU, no model.

One synthesizer is built per grammar and reused via feed() across every
program's every prefix (never rebuilt): rebuilding the grammar per case is
the known cost trap here, not the prefix-walk itself.

Usage: uv run python benchmarks/scripts/false_prune.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import aufbau
import proposition7
from benchmarks.api import BenchmarkTask, load_all_tasks

# Grammars with a real, external, differential oracle -- see lmpl-plan.md
# section 5.1/5.3. Invented DSLs (tool/tool_sexpr) have no such oracle and
# are out of scope for this specific soundness claim.
CERTIFIED_GRAMMARS = ("ml", "c")


def prefix_violations(spg, program: str) -> list[int]:
    """1-indexed character offsets where status() == "dead" partway through
    a program known to be valid in full."""
    synth = aufbau.Synthesizer.from_grammar(spg, "")
    violations = []
    for i, ch in enumerate(program, start=1):
        synth.feed(ch)
        if synth.status() == "dead":
            violations.append(i)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None, help="write per-program results as JSON")
    args = parser.parse_args()

    tasks = load_all_tasks()
    by_grammar: dict[str, list[BenchmarkTask]] = {}
    for task in tasks:
        if task.grammar in CERTIFIED_GRAMMARS:
            by_grammar.setdefault(task.grammar, []).append(task)

    total_programs = 0
    total_violations = 0
    records: list[dict] = []
    for grammar in CERTIFIED_GRAMMARS:
        grammar_tasks = by_grammar.get(grammar, [])
        spg = aufbau.SPG(proposition7.get_grammar(grammar))
        print(f"\n=== {grammar} ({len(grammar_tasks)} programs) ===")
        for task in grammar_tasks:
            total_programs += 1
            violations = prefix_violations(spg, task.expected)
            total_violations += len(violations)
            records.append(
                {"grammar": grammar, "task_id": task.task_id, "violations": violations}
            )
            if violations:
                print(f"  FALSE-PRUNE  {task.task_id}: dead at char(s) {violations}")
            else:
                print(f"  ok           {task.task_id}")

    print(f"\n{total_programs} programs, {total_violations} false-prune violation(s).")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "total_programs": total_programs,
                    "total_violations": total_violations,
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )
    if total_violations:
        print("false_prune_rate > 0 -- SOUNDNESS VIOLATION, see above")
        return 1
    print("false_prune_rate = 0.0 (empirical soundness holds on this corpus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
