#!/usr/bin/env python3
"""Basic constrained decoding demo for built-in grammars."""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import proposition7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grammar",
        default="stlc",
        choices=proposition7.list_grammars(),
        help="Built-in grammar to use for constrained decoding",
    )
    parser.add_argument("--prompt", default=None, help="Override prompt text")
    parser.add_argument(
        "--initial", default=None, help="Override initial constrained seed"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20, help="Maximum generated tokens"
    )
    parser.add_argument(
        "--model", default="gpt2", help="HuggingFace model name for generation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prompt = args.prompt if args.prompt is not None else "Find me a closed lambda term of type A -> A."
    initial_code = args.initial if args.initial is not None else "λx"

    print("=" * 60)
    print("Basic: Constrained Generation Demo")
    print("=" * 60)

    print(f"\nLoading {args.model}...")
    model = proposition7.ConstrainedModel.from_pretrained(
        args.model,
        grammar=proposition7.get_grammar(args.grammar),
    )
    print(f"Grammar: {args.grammar}")

    print(f"\nPrompt: '{prompt}'")
    print(f"Initial: '{initial_code}'")

    print("\n" + "-" * 30)
    print("--- Constrained Generation ---")
    print("-" * 30)

    result = model.generate_constrained(
        prompt=prompt,
        initial=initial_code,
        max_tokens=args.max_tokens,
    )

    print("\n--- Result ---")
    print(f"Generated: '{result.text}'")
    print(f"Complete: {result.is_complete}")
    print(f"Tokens: {result.tokens_generated}")
    print(f"Stopped: {result.stopped_reason}")


if __name__ == "__main__":
    main()
