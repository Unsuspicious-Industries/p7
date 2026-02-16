#!/usr/bin/env python3
"""GPT-2 constrained decoding demo for built-in grammars."""

import argparse

import p7 as p7


PRESETS = {
    "stlc": {
        "prompt": "Complete this simply typed lambda calculus expression:\n",
        "initial": "λf:(Int->Bool).λx:Int.",
    },
    "fun": {
        "prompt": "Complete this typed functional expression:\n",
        "initial": "let x: Int = 1; x +",
    },
    "imp": {
        "prompt": "Complete this typed imperative program:\n",
        "initial": "x: Int = 1; if x < 3 { y: Int = x +",
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
    return parser.parse_args()


def main():
    args = parse_args()
    preset = PRESETS.get(args.grammar, PRESETS["fun"])
    prompt = args.prompt if args.prompt is not None else preset["prompt"]
    initial_code = args.initial if args.initial is not None else preset["initial"]

    print("=" * 60)
    print("GPT-2 + Typed Constrained Generation")
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
    print("\n--- Constrained Generation ---")
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
    while True:
        try:
            token = next(gen)
        except StopIteration as e:
            result = e.value
            print("\n--- Result ---")
            print(f"Generated: '{result.text}'")
            print(f"Complete: {result.is_complete}")
            break

        constrained_text += token
        print(f"  Step {step:2d}: '{repr(token)[1:-1]}' => '{constrained_text}'")
        step += 1


if __name__ == "__main__":
    main()
