#!/usr/bin/env python3
"""Simple modal generation test for Qwen 3.5 0.8B using lambda calculus."""

import argparse
import os
import sys

# Ensure the repository src directory is importable when running this example.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src as p7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-0.8B",
        help="Modal model id to use for generation",
    )
    parser.add_argument(
        "--grammar",
        default="stlc",
        choices=p7.list_grammars(),
        help="Grammar used for constrained generation",
    )
    parser.add_argument(
        "--initial",
        default="λ",
        help="Initial lambda calculus seed text",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--modal-env",
        default=".env",
        help="Path to .env containing MODAL_TOKEN_ID and MODAL_TOKEN_SECRET",
    )
    parser.add_argument(
        "--app-name",
        default="proposition7-generation",
        help="Modal app name to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Modal STLC Generation Test ===")
    print(f"Model: {args.model}")
    print(f"Grammar: {args.grammar}")
    print(f"Initial prefix: {args.initial}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Modal env: {args.modal_env}")

    deployment = p7.ModalDeployment(
        args.model,
        grammar=args.grammar,
        app_name=args.app_name,
        env_path=args.modal_env,
        gpu="H100",
    )

    prompt = (
        "make a function from type A to B that returns the input applied to itself in simply type lambda calculus. "
    )

    print("\n--- Generating ---")
    result = deployment.generate_constrained(
        prompt=prompt,
        initial=args.initial,
        max_tokens=args.max_tokens,
    )

    print("\n--- Result ---")
    print(f"Text: {result.text}")
    print(f"Complete: {result.is_complete}")
    print(f"Stopped reason: {result.stopped_reason}")
    if hasattr(result, "diagnostics"):
        print(f"Diagnostics: {result.diagnostics}")


if __name__ == "__main__":
    main()
