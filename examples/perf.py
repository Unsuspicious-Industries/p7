#!/usr/bin/env python3
"""Performance test for typed completions."""

import argparse
import time
import statistics

import p7 as p7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", default="fun", help="Grammar to test")
    parser.add_argument("--initial", default="let x: Int = 1; x +", help="Initial text")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grammar_str = p7.get_grammar(args.grammar)
    grammar = p7.Grammar(grammar_str)
    print(f"Grammar: {args.grammar}")
    print(f"Initial: {args.initial!r}")
    print(f"Iterations: {args.iterations}")
    print()

    # Create synthesizer
    synth = p7.Synthesizer(grammar, args.initial)

    # Measure get_completions
    times = []
    for i in range(args.iterations):
        start = time.perf_counter()
        completions = synth.get_completions()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Iter {i+1}: {elapsed*1000:.1f}ms ({len(completions)} completions)")

    print()
    print(f"Mean: {statistics.mean(times)*1000:.1f}ms")
    print(f"Median: {statistics.median(times)*1000:.1f}ms")
    print(f"Min: {min(times)*1000:.1f}ms")
    print(f"Max: {max(times)*1000:.1f}ms")
    if len(times) > 1:
        print(f"Stdev: {statistics.stdev(times)*1000:.1f}ms")

    print()
    print(f"Example completions: {completions[:10]}...")


if __name__ == "__main__":
    main()
