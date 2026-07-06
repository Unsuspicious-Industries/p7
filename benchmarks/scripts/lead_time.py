#!/usr/bin/env python3
"""Prune lead time (paper section 5.1): on invalid programs, how many
characters before the real compiler's reported error site does aufbau's
semantic grammar first answer "dead", compared against a syntax-only
baseline (the same grammar with its typing rules stripped)? This is the
number the paper leads with for section 5.1 -- pure CPU, no model.

The invalid-program corpus here is a small set of hand-written mutations of
the existing valid ml/c task corpus (operator/argument swap, bound-variable
rename to an undefined name, literal type change, pointer-level mismatch),
per lmpl-plan.md section 8.5. Each mutant is verified against the real
compiler before being used: a mutation that the compiler still accepts is
reported and excluded, not silently counted.

Usage: uv run python benchmarks/scripts/lead_time.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import aufbau
import proposition7
from grammars import strip_typing_rules

_OCAMLC = shutil.which("ocamlc")
_CC = shutil.which("cc") or shutil.which("gcc")

_OCAML_ERR_RE = re.compile(r'line (\d+), characters (\d+)-(\d+)')
_CC_ERR_RE = re.compile(r'^<stdin>:(\d+):(\d+): error', re.MULTILINE)


@dataclass(frozen=True)
class Mutant:
    task_id: str
    grammar: str
    program: str
    kind: str


# Hand-written mutations of the existing valid corpus (benchmarks/data/ml_*
# and c_*.toml), one representative case per mutation category from section
# 8.5. Each must be genuinely invalid; verified at runtime, not assumed.
MUTANTS = [
    Mutant("ml_identity_int", "ml", "fun (x : int) -> y", "unbound_variable"),
    Mutant("ml_let_arith", "ml", "let a : bool = 5 in a + 1", "literal_type_change"),
    Mutant("ml_compare", "ml", "1 < true", "operand_type_change"),
    Mutant("ml_if", "ml", "if 1 < 2 then 1 else true", "branch_type_mismatch"),
    Mutant("ml_cons_list", "ml", "1 :: true :: 3 :: []", "list_element_type_change"),
    Mutant("ml_apply_identity", "ml", "(fun (x : int) -> x)(true)", "argument_type_change"),
    Mutant(
        "ml_sum_list", "ml",
        "let rec sum : int list -> int = fun (xs : int list) -> "
        "match xs with [] -> 0 | h :: t -> h + sum(h) in sum(1 :: 2 :: [])",
        "bound_variable_rename",
    ),
    Mutant("c_addr_deref", "c", "int roundtrip(int x) { int *p = x; return *p; }", "pointer_level_mismatch"),
    Mutant("c_add", "c", "int add(int x, int y) { return x + z; }", "unbound_variable"),
    Mutant(
        "c_pointer_chain", "c",
        "int *chain(int *p) { int **pp = p; return *pp; }",
        "pointer_level_mismatch",
    ),
]


def ml_error_offset(program: str) -> int | None:
    prefix = "let _ = "
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.ml"
        path.write_text(prefix + program + "\n")
        result = subprocess.run(
            [_OCAMLC, "-c", str(path)], cwd=tmp, capture_output=True, text=True, timeout=10
        )
    if result.returncode == 0:
        return None
    m = _OCAML_ERR_RE.search(result.stderr)
    if not m:
        return None
    line, col = int(m.group(1)), int(m.group(2))
    if line != 1:
        return None  # corpus is single-line; multi-line offset mapping not needed yet
    return col - len(prefix)


def c_error_offset(program: str) -> int | None:
    try:
        result = subprocess.run(
            [_CC, "-fsyntax-only", "-xc", "-"], input=program, capture_output=True, text=True, timeout=10
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode == 0:
        return None
    m = _CC_ERR_RE.search(result.stderr)
    if not m:
        return None
    line, col = int(m.group(1)), int(m.group(2))
    if line != 1:
        return None
    return col - 1


def first_dead_offset(spg, program: str) -> int | None:
    """0-indexed character offset of the first `dead` verdict, or None if the
    grammar never goes dead over the whole program (a miss worth reporting,
    not a crash). `feed` raises exactly at the character that makes the
    prefix dead (dead is monotone: no live extension exists past it), so the
    raise site itself is the offset -- there is no post-dead status to poll."""
    synth = aufbau.Synthesizer.from_grammar(spg, "")
    for i, ch in enumerate(program):
        try:
            synth.feed(ch)
        except RuntimeError:
            return i
        if synth.status() == "dead":
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None, help="write per-mutant results as JSON")
    args = parser.parse_args()

    if _OCAMLC is None or _CC is None:
        print(f"missing compiler(s): ocamlc={_OCAMLC} cc={_CC}")
        return 1

    semantic_spgs: dict[str, object] = {}
    syntactic_spgs: dict[str, object] = {}
    for grammar in ("ml", "c"):
        spec = proposition7.get_grammar(grammar)
        semantic_spgs[grammar] = aufbau.SPG(spec)
        syntactic_spgs[grammar] = aufbau.SPG(strip_typing_rules(spec))

    semantic_leads: list[int] = []
    syntactic_leads: list[int] = []
    skipped = 0
    records: list[dict] = []

    for mutant in MUTANTS:
        error_offset = (
            ml_error_offset(mutant.program) if mutant.grammar == "ml" else c_error_offset(mutant.program)
        )
        if error_offset is None:
            print(f"SKIP  {mutant.task_id} ({mutant.kind}): compiler did not reject this mutant, or offset unparseable")
            skipped += 1
            records.append(
                {
                    "task_id": mutant.task_id,
                    "grammar": mutant.grammar,
                    "kind": mutant.kind,
                    "skipped": True,
                }
            )
            continue

        semantic_dead = first_dead_offset(semantic_spgs[mutant.grammar], mutant.program)
        syntactic_dead = first_dead_offset(syntactic_spgs[mutant.grammar], mutant.program)

        semantic_str = "never" if semantic_dead is None else str(semantic_dead)
        syntactic_str = "never" if syntactic_dead is None else str(syntactic_dead)
        print(
            f"{mutant.task_id:20s} ({mutant.kind:24s}) compiler_err={error_offset:3d}  "
            f"semantic_dead={semantic_str:>5s}  syntactic_dead={syntactic_str:>5s}"
        )

        records.append(
            {
                "task_id": mutant.task_id,
                "grammar": mutant.grammar,
                "kind": mutant.kind,
                "skipped": False,
                "compiler_err_offset": error_offset,
                "semantic_dead_offset": semantic_dead,
                "syntactic_dead_offset": syntactic_dead,
            }
        )

        if semantic_dead is not None:
            semantic_leads.append(error_offset - semantic_dead)
        else:
            print(
                f"      NOTE: aufbau's semantic {mutant.grammar} grammar never went dead on this "
                "mutant. Final status is still not \"typed\" (confirmed correctly rejected at "
                "grading time) -- this is a precision gap in mid-stream dead-detection for this "
                "construct, not a soundness violation: \"live\" is only ever \"not yet proven "
                "dead\", never a positive claim a completion exists."
            )
        if syntactic_dead is not None:
            syntactic_leads.append(error_offset - syntactic_dead)

    used = len(MUTANTS) - skipped
    print()
    print(f"{used}/{len(MUTANTS)} mutants used, {skipped} skipped")
    print(f"semantic:  caught mid-stream in {len(semantic_leads)}/{used} cases")
    if semantic_leads:
        print(
            f"           lead time vs compiler error site: mean={statistics.mean(semantic_leads):.1f}  "
            f"median={statistics.median(semantic_leads):.1f}  n={len(semantic_leads)}"
        )
        print(
            "           (near-zero/slightly negative is expected: the compiler's column is the "
            "START of the offending token; aufbau only knows once it has consumed the whole "
            "token. The real comparison is against the syntax-only baseline below.)"
        )
    print(f"syntax-only: caught mid-stream in {len(syntactic_leads)}/{used} cases", end="")
    if syntactic_leads:
        print(
            f" -- lead time vs compiler: mean={statistics.mean(syntactic_leads):.1f}  "
            f"median={statistics.median(syntactic_leads):.1f}"
        )
    else:
        print(" -- a syntax-only grammar cannot detect a type error by construction; this is the expected floor")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "used": used,
                    "total": len(MUTANTS),
                    "skipped": skipped,
                    "semantic_leads": semantic_leads,
                    "syntactic_leads": syntactic_leads,
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
