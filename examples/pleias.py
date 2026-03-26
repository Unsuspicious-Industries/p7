#!/usr/bin/env python3
"""
PleIAs constrained generation demo.

Demonstrates grammar-constrained decoding and the ReasoningEnvironment
(CoT + constrained output) with PleIAs SYNTH-series models:

  PleIAs/Monad         — 56.7M params, English-only, vocab=8192
  PleIAs/Baguettotron  — 321M params, multilingual,  vocab=65536

Both models use native <think> / </think> reasoning tokens and the
ChatML prompt format.  No trust_remote_code required.

Usage examples
--------------
# Constrained generation with Monad (smallest, good for quick tests)
python examples/pleias.py --model PleIAs/Monad --grammar fun --mode constrained

# Constrained generation with Baguettotron
python examples/pleias.py --model PleIAs/Baguettotron --grammar stlc --mode constrained

# ReasoningEnvironment: CoT thinking + grammar-constrained output
python examples/pleias.py --model PleIAs/Monad --grammar stlc --mode reasoning

# Run all grammars in constrained mode
python examples/pleias.py --model PleIAs/Monad --mode all
"""

import argparse
import sys
import time

import p7
from p7.models import PleiasConstrainedModel


# ---------------------------------------------------------------------------
# Task presets: one entry per grammar
# ---------------------------------------------------------------------------
TASKS = {
    "stlc": {
        "prompt": "Write a simply-typed lambda calculus term that applies a function f of type (Int->Bool) to an integer x.",
        "initial": "λf:(Int->Bool).",
        "description": "Should produce λf:(Int->Bool).λx:Int.(f x)",
    },
    "fun": {
        "prompt": "Write a typed functional expression that doubles an integer.",
        "initial": "let double: Int -> Int = (x: Int) =>",
        "description": "Should produce let double: Int -> Int = (x: Int) => x + x",
    },
    "imp": {
        "prompt": "Write a typed imperative program that stores 1 in x and conditionally assigns y.",
        "initial": "{ let x: Int = 1; if (x < 5) { let y: Int =",
        "description": "Should produce a well-typed if/else block",
    },
}

# ReasoningEnvironment tasks (STLC only — richer CoT payoff)
REASONING_TASKS = [
    {
        "name": "identity",
        "prompt": "Create the identity function for Int in simply typed lambda calculus.",
        "initial": "λx:",
        "description": "Should produce λx:Int.x",
    },
    {
        "name": "const_K",
        "prompt": "Create the K combinator: a function that takes an Int and returns a function that ignores a Bool and returns the Int.",
        "initial": "λx:Int.",
        "description": "Should produce λx:Int.λy:Bool.x",
    },
    {
        "name": "apply",
        "prompt": "Create a function that applies f:(Int->Bool) to x:Int.",
        "initial": "λf:(Int->Bool).",
        "description": "Should produce λf:(Int->Bool).λx:Int.(f x)",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(text: str, char: str = "=", width: int = 70) -> None:
    print(char * width)
    print(text)
    print(char * width)


def load_model(model_name: str, grammar_name: str) -> "PleiasConstrainedModel":
    print(f"Loading {model_name} …")
    t0 = time.perf_counter()
    model = PleiasConstrainedModel.from_pretrained(  # type: ignore[return-value]
        model_name,
        grammar=p7.get_grammar(grammar_name),
    )
    elapsed = time.perf_counter() - t0
    print(f"Loaded in {elapsed:.1f}s  |  vocab size: {len(model.vocab)}")
    model.model.eval()
    return model  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Constrained generation demo
# ---------------------------------------------------------------------------

def run_constrained(model: PleiasConstrainedModel, grammar_name: str, task: dict) -> None:
    print_header(f"Constrained generation  [{grammar_name}]", "-")
    print(f"  Prompt:  {task['prompt']}")
    print(f"  Initial: {task['initial']}")
    print(f"  Expect:  {task['description']}")
    print()

    tokens: list[str] = []

    def on_token(tok: str, step: int) -> None:
        tokens.append(tok)
        print(tok, end="", flush=True)

    print(f"  Output:  {task['initial']}", end="", flush=True)
    t0 = time.perf_counter()

    result = model.until_complete(
        prompt=task["prompt"],
        initial=task["initial"],
        max_tokens=60,
        greedy_k=3,
        pre_top_k=100,
        grammar_name=grammar_name,
        on_token=on_token,
    )

    elapsed = time.perf_counter() - t0
    print()
    print()
    print(f"  Complete:  {result.is_complete}")
    print(f"  Tokens:    {result.tokens_generated}  ({elapsed*1000/max(result.tokens_generated,1):.0f}ms/tok)")
    print(f"  Reason:    {result.stopped_reason}")
    print(f"  Full text: {result.text!r}")
    print()


# ---------------------------------------------------------------------------
# ReasoningEnvironment demo
# ---------------------------------------------------------------------------

def run_reasoning(model: PleiasConstrainedModel, grammar_name: str) -> None:
    print_header(f"ReasoningEnvironment  [{grammar_name}]  (CoT + grammar)", "=")

    # Show the auto-generated system prompt once, using the model's actual think tokens
    system_prompt = p7.build_system_prompt(
        grammar_name,
        think_open=model.think_open(),
        think_close=model.think_close(),
    )
    print_header("System prompt", "-", 50)
    print(system_prompt)
    print()

    env = p7.ReasoningEnvironment(
        model=model,
        grammar_name=grammar_name,
        think_budget=100,
        formal_budget=50,
    )

    results = []
    for task in REASONING_TASKS:
        print_header(f"Task: {task['name']}", "-", 50)
        print(f"  {task['description']}")
        print(f"  Prompt:  {task['prompt']}")
        print(f"  Initial: {task['initial']}")
        print()

        def on_mode_switch(mode: p7.Mode, tag: str) -> None:
            print(f"\n[{mode.value.upper()}] {tag}", flush=True)

        def on_think_token(tok: str, step: int) -> None:
            print(tok, end="", flush=True)

        def on_formal_token(tok: str, step: int) -> None:
            # highlight constrained output in bold
            print(f"\033[1m{tok}\033[0m", end="", flush=True)

        try:
            result = env.generate(
                prompt=task["prompt"],
                initial=task["initial"],
                max_blocks=4,
                start_thinking=True,
                on_mode_switch=on_mode_switch,
                on_think_token=on_think_token,
                on_formal_token=on_formal_token,
            )
            print("\n")
            print(f"  Complete:      {result.is_complete}")
            print(f"  Total tokens:  {result.total_tokens}")
            print(f"  Think blocks:  {len(result.think_blocks)}")
            print(f"  Grammar blocks:{len(result.grammar_blocks)}")
            if result.final_output:
                print(f"  Final output:  {result.final_output.content!r}")

            results.append({
                "name": task["name"],
                "complete": result.is_complete,
                "output": result.final_output.content if result.final_output else None,
            })

        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            results.append({"name": task["name"], "complete": False, "output": None})

        print()

    # Summary
    print_header("Summary", "=")
    print(f"{'Task':<15} {'Complete':<10} {'Output'}")
    print("-" * 60)
    for r in results:
        out = (r["output"] or "N/A")[:40]
        print(f"{r['name']:<15} {str(r['complete']):<10} {out}")
    complete = sum(1 for r in results if r["complete"])
    print(f"\nComplete: {complete}/{len(results)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="PleIAs/Monad",
        choices=["PleIAs/Monad", "PleIAs/Baguettotron"],
        help="PleIAs model to use (default: PleIAs/Monad)",
    )
    parser.add_argument(
        "--grammar",
        default="stlc",
        choices=list(p7.list_grammars()),
        help="Grammar for constrained generation (default: stlc)",
    )
    parser.add_argument(
        "--mode",
        default="constrained",
        choices=["constrained", "reasoning", "all"],
        help=(
            "constrained: single grammar-constrained run; "
            "reasoning: ReasoningEnvironment CoT demo; "
            "all: constrained run for every grammar"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Max tokens for constrained generation (default: 60)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_header(f"P7 × PleIAs  —  {args.model}", "#", 70)
    print(f"Mode: {args.mode}  |  Grammar: {args.grammar}")
    print()

    if args.mode == "all":
        for grammar_name, task in TASKS.items():
            model = load_model(args.model, grammar_name)
            run_constrained(model, grammar_name, task)
            del model  # free memory before loading next
        return

    model = load_model(args.model, args.grammar)
    print()

    if args.mode == "constrained":
        task = TASKS.get(args.grammar, TASKS["stlc"])
        run_constrained(model, args.grammar, task)

    elif args.mode == "reasoning":
        if args.grammar != "stlc":
            print(f"Note: reasoning demo uses stlc tasks regardless of --grammar flag", file=sys.stderr)
        run_reasoning(model, "stlc")


if __name__ == "__main__":
    main()
