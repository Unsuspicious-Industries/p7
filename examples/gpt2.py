#!/usr/bin/env python3
"""GPT-2 constrained decoding demo for built-in grammars."""

import argparse
import time
import p7 as p7

import os
os.environ["P7_CONSTRAINED_DEBUG"] = "1"


PRESETS = {
    "stlc": {
        "prompt": "Complete this simply typed lambda calculus expression:\n",
        "initial": "λf:(Int->Bool).λx:Int.",
    },
    "fun": {
        "prompt": "Build me a function that squares an input with:\n",
        "initial": "let square: Int -> Int = fn x: Int =>",
    },
    "imp": {
        "prompt": "Complete this typed imperative program:\n",
        "initial": "{ let x: Int = 1; if (x < 5) { let y: Int = x + 1; } else { let y: Int =",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grammar",
        default="fun",
        choices=p7.list_grammars(),
        help="Built-in grammar to use for constrained decoding",
    )
    parser.add_argument("--prompt", default=None, help="Override prompt text")
    parser.add_argument("--initial", default=None, help="Override initial constrained seed")
    parser.add_argument("--max-tokens", type=int, default=20, help="Maximum generated tokens")
    parser.add_argument("--pre-top-k", type=int, default=100, help="Candidate tokens before filtering")
    parser.add_argument("--greedy-k", type=int, default=1, help="Top-k among valid tokens")
    parser.add_argument("--compare", action="store_true", help="Compare with unconstrained generation")
    parser.add_argument("--profile", action="store_true", help="Profile generation")
    return parser.parse_args()


def main():
    args = parse_args()
    preset = PRESETS.get(args.grammar, PRESETS["fun"])
    prompt = args.prompt if args.prompt is not None else preset["prompt"]
    initial_code = args.initial if args.initial is not None else preset["initial"]

    print("=" * 60)
    print("GPT-2: Constrained Generation Demo")
    print("=" * 60)

    print("\nLoading GPT-2...")
    model = p7.ConstrainedModel.from_pretrained(
        "gpt2",
        grammar=p7.get_grammar(args.grammar),
    )
    model.model.eval()
    print(f"Vocab size: {len(model.vocab)}")
    print(f"Grammar: {args.grammar}")

    print(f"\nPrompt: '{prompt}'")
    print(f"Initial: '{initial_code}'")

    constrained_text = initial_code
    print("\n" + "-" * 30)
    print("--- Constrained Generation ---")
    print("-" * 30)

    if args.profile:
        profile_times = {
            "forward": 0.0,
            "get_completions": 0.0,
            "token_matching": 0.0,
            "sampling": 0.0,
        }
    else:
        profile_times = None

    gen = model.iter_constrained(
        prompt=prompt,
        initial=initial_code,
        max_tokens=args.max_tokens,
        greedy_k=args.greedy_k,
        pre_top_k=args.pre_top_k,
        stop_on_complete=True,
        grammar_name=args.grammar,
    )

    step = 0
    total_start = time.perf_counter()
    while True:
        step_start = time.perf_counter() if args.profile else None

        try:
            token = next(gen)
        except StopIteration as e:
            result = e.value
            print("\n--- Result ---")
            print(f"Generated: '{result.text}'")
            print(f"Complete: {result.is_complete}")
            print(f"Reason: {result.stopped_reason}")
            break

        if args.profile and step_start is not None:
            profile_times["sampling"] += time.perf_counter() - step_start

        constrained_text += token
        print(f"  Step {step:2d}: '{repr(token)[1:-1]}' => '{constrained_text}'")
        step += 1

    total_elapsed = time.perf_counter() - total_start

    if args.profile:
        print("\n" + "-" * 30)
        print("--- Profile ---")
        print("-" * 30)
        print(f"Total time: {total_elapsed:.3f}s")
        print(f"Steps: {step}")
        print(f"Time per token: {total_elapsed/step*1000:.1f}ms")


if __name__ == "__main__":
    main()
