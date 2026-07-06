#!/usr/bin/env python3
"""Differential certification of ml.auf against ocamlc (paper section 5.1),
with emphasis on letrec self-recursion soundness (section 5.7).

aufbau's c.auf had a real, confirmed bug: binding a function's own name to
a signature containing an *unresolved metavariable*, before checking its
body, let a wrong-typed recursive call prematurely resolve that
metavariable (a wrong-arity self-recursive call was wrongly accepted).
ml.auf's letrec instead binds the name to the concrete syntactic
annotation `tau` -- already ground before value/body are checked -- so
there is no shared open metavariable for a bad recursive call to corrupt.
This script is the empirical check of that claim, not just the reasoning.

This mirrors aufbau/ocaml/cert.ml's corpus (including the three adversarial
letrec probes added alongside this script) but talks to plain `ocamlc` on
PATH instead of linking compiler-libs through dune, so it needs no OCaml
FFI build: only `aufbau-rs` (already a p7 dependency) and an `ocamlc`
binary. This is what makes it fit the artifact's CPU-only reproduction
tier (lmpl-plan.md section 9.5) -- a reviewer with no GPU and no OCaml
build toolchain beyond the compiler itself can still run this.

Usage: uv run python benchmarks/scripts/letrec_probe.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "src"))

import aufbau
import proposition7

_OCAMLC = shutil.which("ocamlc")

# Mirrors aufbau/ocaml/cert.ml's `valid` list.
VALID = [
    "fun (x : int) -> x",
    "fun (x : int) -> fun (y : bool) -> x",
    "(fun (x : int) -> x)(5)",
    "let a : int = 5 in a",
    "let a : int = 5 in a + 1",
    "1 < 2",
    "if true then 1 else 2",
    "if 1 < 2 then 1 else 0",
    "(1, true)",
    "fst (1, true)",
    "snd (1, true)",
    "1 :: []",
    "1 :: 2 :: 3 :: []",
    "(1, true) :: []",
    "let xs : int list = 1 :: [] in xs",
    "let rec f : int -> int = fun (n : int) -> f(n) in f(0)",
    "assert false",
    "let a : int = assert false in a",
    "if true then 1 else assert false",
    "let f : int -> bool = assert false in f(0)",
    "match [] with [] -> 0 | h :: t -> h + 1",
    "let rec length : int list -> int = fun (xs : int list) -> match xs with [] -> 0 | h :: t -> 1 + length(t) in length(1 :: 2 :: 3 :: [])",
    "let rec sum : int list -> int = fun (xs : int list) -> match xs with [] -> 0 | h :: t -> h + sum(t) in sum(1 :: 2 :: [])",
    "let rec member : int list -> bool = fun (xs : int list) -> match xs with [] -> false | h :: t -> if h = 0 then true else member(t) in member(0 :: 1 :: [])",
    "let rec copy : int list -> int list = fun (xs : int list) -> match xs with [] -> [] | h :: t -> h :: copy(t) in copy(1 :: 2 :: [])",
]

# Mirrors aufbau/ocaml/cert.ml's `invalid` list, plus the three
# self-recursion soundness probes (the section-5.7 stress test).
INVALID = [
    "fun (x : int) -> y",
    "1 + true",
    "if 1 then 2 else 3",
    "if true then 1 else false",
    "fst 5",
    "let a : bool = 5 in a",
    "true < 2",
    "1 :: true :: []",
    "1 :: 2",
    "let xs : bool list = 1 :: [] in xs",
    "match 1 :: [] with [] -> 0 | h :: t -> true",
    "match 5 with [] -> 0 | h :: t -> 1",
    "let rec f : int list -> int = fun (xs : int list) -> match xs with [] -> 0 | h :: t -> f(h) in f(1 :: [])",
    # Self-recursion soundness probes: a wrong-typed recursive argument, and
    # a wrong-typed sibling branch beside one that recurses correctly (both
    # branch positions). All three: aufbau must reject, OCaml must reject.
    "let rec f : int -> int = fun (n : int) -> f(true) in f(0)",
    "let rec f : int -> bool = fun (n : int) -> if n = 0 then 1 else f(n) in f(1)",
    "let rec f : int -> int = fun (n : int) -> if n = 0 then f(n) else true in f(1)",
]

# Rejected by monomorphic ml.auf, accepted by OCaml's generalization -- the
# documented fragment boundary, not a failure.
BEYOND = [
    "match [] with [] -> 0 | h :: t -> if h then 1 else h + 1",
]


def aufbau_accepts(spg, program: str) -> bool:
    return aufbau.Synthesizer.from_grammar(spg, program).status() == "typed"


def ocamlc_accepts(program: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.ml"
        path.write_text(f"let _ = {program}\n")
        result = subprocess.run(
            [_OCAMLC, "-c", str(path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None, help="write per-case results as JSON")
    args = parser.parse_args()

    if _OCAMLC is None:
        print("ocamlc not found on PATH -- cannot run the differential probe")
        return 1

    spg = aufbau.SPG(proposition7.get_grammar("ml"))
    failed = 0
    records: list[dict] = []

    def case(category: str, program: str, ok: bool, aufbau_r: bool, ocaml_r: bool) -> None:
        nonlocal failed
        label = {"valid": "agree+", "invalid": "agree-", "beyond": "bound "}[category]
        print(("ok   " if ok else "FAIL ") + f"{label} {program}")
        records.append(
            {
                "category": category,
                "program": program,
                "aufbau_accepts": aufbau_r,
                "ocamlc_accepts": ocaml_r,
                "agree": ok,
            }
        )
        if not ok:
            failed += 1

    for program in VALID:
        a, o = aufbau_accepts(spg, program), ocamlc_accepts(program)
        case("valid", program, a and o, a, o)
    for program in INVALID:
        a, o = aufbau_accepts(spg, program), ocamlc_accepts(program)
        case("invalid", program, (not a) and (not o), a, o)
    for program in BEYOND:
        a, o = aufbau_accepts(spg, program), ocamlc_accepts(program)
        case("beyond", program, (not a) and o, a, o)

    print()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"total_cases": len(records), "disagreements": failed, "records": records}, indent=2)
            + "\n"
        )
    if failed:
        print(f"{failed} disagreement(s) -- see FAIL lines above")
        return 1
    print("aufbau == ocamlc on this corpus; letrec self-recursion probes confirm no")
    print("c.auf-style soundness bug in ml.auf's letrec (section 5.7 resolved).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
