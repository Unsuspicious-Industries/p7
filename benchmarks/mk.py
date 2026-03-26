#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


@dataclass
class Row:
    task_id: str
    language: str
    grammar: str
    category: str
    instruction: str
    initial: str
    expected: str
    max_tokens: int
    feed: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "task_id": self.task_id,
            "language": self.language,
            "grammar": self.grammar,
            "category": self.category,
            "instruction": self.instruction,
            "initial": self.initial,
            "expected": self.expected,
            "max_tokens": self.max_tokens,
            "feed": self.feed,
        }


def stlc() -> list[Row]:
    rows: list[Row] = []
    variants = [
        ("Int", "Bool"),
        ("Bool", "Int"),
        ("Int", "Int"),
        ("Bool", "Bool"),
        ("Int", "Bool"),
        ("Bool", "Int"),
        ("Int", "Int"),
        ("Bool", "Bool"),
        ("Int", "Bool"),
        ("Bool", "Int"),
    ]
    for i, (ta, tb) in enumerate(variants, start=1):
        rows.extend(
            [
                Row(f"stlc_{i:02d}_id", "stlc", "stlc", "smoke_identity", f"Use the provided complete identity over {ta}.", f"λx:{ta}.x", f"λx:{ta}.x", 0, "hints"),
                Row(f"stlc_{i:02d}_k", "stlc", "stlc", "const", f"Return first arg for {ta} then {tb}.", f"λx:{ta}.", f"λx:{ta}.λy:{tb}.x", 40, "hints"),
                Row(f"stlc_{i:02d}_k2", "stlc", "stlc", "const", f"Return second arg for {ta} then {tb}.", f"λx:{ta}.", f"λx:{ta}.λy:{tb}.y", 40, "hints"),
                Row(f"stlc_{i:02d}_app", "stlc", "stlc", "application", f"Apply f:({ta}->{tb}) to x:{ta}.", f"λf:({ta}->{tb}).", f"λf:({ta}->{tb}).λx:{ta}.(f x)", 48, "hints"),
                Row(f"stlc_{i:02d}_twice", "stlc", "stlc", "repetition", f"Apply f:({ta}->{ta}) twice.", f"λf:({ta}->{ta}).", f"λf:({ta}->{ta}).λx:{ta}.(f (f x))", 56, "hints"),
                Row(f"stlc_{i:02d}_comp", "stlc", "stlc", "composition", f"Compose ({tb}->{ta}) with ({ta}->{tb}).", f"λf:({tb}->{ta}).", f"λf:({tb}->{ta}).λg:({ta}->{tb}).λx:{ta}.(f (g x))", 72, "hints"),
                Row(f"stlc_{i:02d}_eta", "stlc", "stlc", "eta", f"Eta-expand ({ta}->{tb}).", f"λf:({ta}->{tb}).", f"λf:({ta}->{tb}).λx:{ta}.(f x)", 56, "hints"),
                Row(f"stlc_{i:02d}_arg", "stlc", "stlc", "argument_order", f"Take x:{ta}, then f:({ta}->{tb}), return f x.", f"λx:{ta}.", f"λx:{ta}.λf:({ta}->{tb}).(f x)", 56, "hints"),
                Row(f"stlc_{i:02d}_hid", "stlc", "stlc", "higher_order", f"Identity over ({ta}->{tb}).", f"λh:({ta}->{tb}).", f"λh:({ta}->{tb}).h", 40, "hints"),
                Row(f"stlc_{i:02d}_triple", "stlc", "stlc", "repetition", f"Apply f:({ta}->{ta}) three times.", f"λf:({ta}->{ta}).", f"λf:({ta}->{ta}).λx:{ta}.(f (f (f x)))", 80, "hints"),
            ]
        )
    return rows


def fun() -> list[Row]:
    rows: list[Row] = []
    floats = ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "5.5"]
    for i in range(10):
        idx = i + 1
        k = idx
        f = floats[i]
        rows.extend(
            [
                Row(f"fun_{idx:02d}_sq", "fun", "fun", "smoke_square", "Use the provided complete square program.", f"let square: Int -> Int = (x: Int) => x * x; square({k})", f"let square: Int -> Int = (x: Int) => x * x; square({k})", 0, "hints"),
                Row(f"fun_{idx:02d}_inc", "fun", "fun", "int_ops", f"Define inc:Int->Int and call on {k}.", "let inc: Int -> Int = (n: Int) =>", f"let inc: Int -> Int = (n: Int) => n + 1; inc({k})", 72, "hints"),
                Row(f"fun_{idx:02d}_dbl", "fun", "fun", "application", f"Define dbl:Int->Int and call twice on {k}.", "let dbl: Int -> Int = (x: Int) =>", f"let dbl: Int -> Int = (x: Int) => x + x; dbl(dbl({k}))", 84, "hints"),
                Row(f"fun_{idx:02d}_fa", "fun", "fun", "float_ops", "Bind Float and add with +.", "let f: Float =", f"let f: Float = {f}; f +. 2.0", 56, "hints"),
                Row(f"fun_{idx:02d}_fm", "fun", "fun", "float_ops", "Bind Float and multiply with *.", "let f: Float =", f"let f: Float = {f}; f *. 2.0", 56, "hints"),
                Row(f"fun_{idx:02d}_la", "fun", "fun", "lambda", f"Apply lambda adding {k} to 10.", "((x: Int) =>", f"((x: Int) => x + {k})(10)", 56, "hints"),
                Row(f"fun_{idx:02d}_let", "fun", "fun", "let", "Use two Int lets then multiply.", "let a: Int =", f"let a: Int = {k}; let b: Int = {k + 1}; a * b", 72, "hints"),
                Row(f"fun_{idx:02d}_sub", "fun", "fun", "int_ops", "Subtract one from bound Int.", "let n: Int =", f"let n: Int = {k}; n - 1", 48, "hints"),
                Row(f"fun_{idx:02d}_fd", "fun", "fun", "float_ops", "Bind Float and divide with /.", "let f: Float =", f"let f: Float = {f}; f /. 2.0", 56, "hints"),
                Row(f"fun_{idx:02d}_id", "fun", "fun", "identity", f"Define id:Int->Int and call on {k}.", "let id: Int -> Int = (x: Int) =>", f"let id: Int -> Int = (x: Int) => x; id({k})", 64, "hints"),
            ]
        )
    return rows


def imp() -> list[Row]:
    rows: list[Row] = []
    for i in range(10):
        idx = i + 1
        start = i + 1
        stop = i + 4
        inc = i + 2
        rows.extend(
            [
                Row(f"imp_{idx:02d}_dec", "imp", "imp", "smoke_declaration", "Use the provided complete declaration program.", f"{{ let x: Int = {start}; }}", f"{{ let x: Int = {start}; }}", 0, "hints"),
                Row(f"imp_{idx:02d}_seq", "imp", "imp", "sequence", "Declare x then y = x + c.", "{ let x: Int =", f"{{ let x: Int = {start}; let y: Int = x + {inc}; }}", 64, "hints"),
                Row(f"imp_{idx:02d}_if", "imp", "imp", "if_else", "If-else on x < limit defines y.", "{ let x: Int =", f"{{ let x: Int = {start}; if (x < {stop}) {{ let y: Int = x + 1; }} else {{ let y: Int = {inc}; }} }}", 96, "hints"),
                Row(f"imp_{idx:02d}_wh", "imp", "imp", "while", "Count up with while.", "{ let counter: Int =", f"{{ let counter: Int = {start}; while (counter < {stop}) {{ counter = counter + 1; }} }}", 96, "hints"),
                Row(f"imp_{idx:02d}_bf", "imp", "imp", "bool", "Branch on Bool flag, define Int z both branches.", "{ let flag: Bool =", f"{{ let flag: Bool = true; if (flag) {{ let z: Int = {start}; }} else {{ let z: Int = {inc}; }} }}", 96, "hints"),
                Row(f"imp_{idx:02d}_ar", "imp", "imp", "arithmetic", "Chain arithmetic declarations.", "{ let a: Int =", f"{{ let a: Int = {start}; let b: Int = a * 2; let c: Int = b - 1; }}", 88, "hints"),
                Row(f"imp_{idx:02d}_as", "imp", "imp", "assignment", "Declare x then update x by assignment.", "{ let x: Int =", f"{{ let x: Int = {start}; x = x + {inc}; }}", 72, "hints"),
                Row(f"imp_{idx:02d}_ni", "imp", "imp", "nested_if", "Nested if in then branch.", "{ let x: Int =", f"{{ let x: Int = {start}; if (x < {stop}) {{ if (x < {stop + 1}) {{ let y: Int = x + 1; }} else {{ let y: Int = x; }} }} else {{ let y: Int = {inc}; }} }}", 120, "hints"),
                Row(f"imp_{idx:02d}_w2", "imp", "imp", "while", "While loop with +2 increment.", "{ let n: Int =", f"{{ let n: Int = {start}; while (n < {stop + 3}) {{ n = n + 2; }} }}", 96, "hints"),
                Row(f"imp_{idx:02d}_mix", "imp", "imp", "mixed", "Bool guard and Int output.", "{ let ready: Bool =", f"{{ let ready: Bool = true; let x: Int = {start}; if (ready) {{ let out: Int = x + 1; }} else {{ let out: Int = x; }} }}", 112, "hints"),
            ]
        )
    return rows


def spec() -> list[Row]:
    """Special test: core grammars with full grammar fed into prompt."""
    rows: list[Row] = []
    for i in range(34):
        idx = i + 1
        k = (i % 10) + 1
        rows.extend(
            [
                Row(f"spec_{idx:03d}_s", "special", "stlc", "grammar_feed_full", "Using full STLC grammar text, complete identity for Int.", "λx:Int.", "λx:Int.x", 40, "full"),
                Row(f"spec_{idx:03d}_f", "special", "fun", "grammar_feed_full", "Using full FUN grammar text, define square then apply.", "let square: Int -> Int = (x: Int) =>", f"let square: Int -> Int = (x: Int) => x * x; square({k})", 80, "full"),
                Row(f"spec_{idx:03d}_i", "special", "imp", "grammar_feed_full", "Using full IMP grammar text, make while-counter program.", "{ let counter: Int =", f"{{ let counter: Int = {k}; while (counter < {k + 2}) {{ counter = counter + 1; }} }}", 104, "full"),
            ]
        )
    return rows[:100]


def write(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["task_id", "language", "grammar", "category", "instruction", "initial", "expected", "max_tokens", "feed"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r.as_dict())


def main() -> None:
    stlc_rows = stlc()
    fun_rows = fun()
    imp_rows = imp()
    spec_rows = spec()
    if len(stlc_rows) != 100 or len(fun_rows) != 100 or len(imp_rows) != 100:
        raise RuntimeError("stlc/fun/imp must each have exactly 100 tasks")
    write(DATA / "stlc.csv", stlc_rows)
    write(DATA / "fun.csv", fun_rows)
    write(DATA / "imp.csv", imp_rows)
    write(DATA / "spec.csv", spec_rows)
    print("wrote", len(stlc_rows), "stlc")
    print("wrote", len(fun_rows), "fun")
    print("wrote", len(imp_rows), "imp")
    print("wrote", len(spec_rows), "spec")


if __name__ == "__main__":
    main()
