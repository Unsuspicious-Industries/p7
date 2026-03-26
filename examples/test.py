#!/usr/bin/env python3
"""CompletionEngine smoke checks for built-in fun/imp constrained decoding."""

import p7 as p7


CASES = {
    "fun": [
        "(x: Int) => x +",
        "let x: Int = 1; x +",
        "let f: Float = 1.0; f +.",
    ],
    "imp": [
        "{ let x: Int = 1; if (x < 5) { let y: Int = x + 1; } else { let y: Int =",
        "{ let counter: Int = 0; while (counter < 3) { counter = counter + 1; } }",
        "{ let flag: Bool = true; if (flag) { let z: Int = 1; } else { let z: Int =",
    ],
}


def run_case(grammar_name: str, seed: str) -> None:
    engine = p7.CompletionEngine(p7.get_grammar(grammar_name))
    print(f"[{grammar_name}] {seed}")
    try:
        engine.feed(seed)

        raw = engine.get_completions()
        print(f"  completions: {raw[:10]}")

    except Exception as err:
        print(f"  error: {err}")
    print()


def main() -> None:
    for grammar_name, seeds in CASES.items():
        print(f"=== {grammar_name.upper()} ===")
        for seed in seeds:
            run_case(grammar_name, seed)


if __name__ == "__main__":
    main()
