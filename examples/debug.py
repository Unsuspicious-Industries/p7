#!/usr/bin/env python3
"""
Debug and smoke-test script for proposition7.

Covers everything needed to verify a clean benchmark run without actually loading
a model (sections 1–4) and optionally with a small model (section 5).

Run without a model:
    python examples/debug.py

Run with a model (slow, requires GPU or patience):
    python examples/debug.py --model Qwen/Qwen3.5-0.5B

Exit code 0 means all assertions passed.
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import aufbau
import proposition7
from proposition7 import build_task_prompt, build_system_prompt, BENCHMARK_MODES
from benchmarks.api import (
    build_prompt,
    build_interaction,
    check_parse,
    load_tasks,
    grammar_name,
)
from benchmarks.run import (
    load_benchmark_config,
    selected_tasks,
    normalize_modes,
)
from benchmarks.utils import build_jobs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        msg = f"{label}" + (f": {detail}" if detail else "")
        print(f"  {FAIL}  {msg}")
        _failures.append(msg)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Section 1: aufbau prefix validation
# ---------------------------------------------------------------------------

def test_aufbau_prefix_validation() -> None:
    section("1. aufbau prefix validation (parse)")

    spec = proposition7.get_grammar("stlc")

    # Valid complete terms
    for term in ["λx:Int.x", "λf:(Int->Bool).λx:Int.(f x)", "(λx:Int.x)"]:
        ok, complete, _ = check_parse(spec, term)
        check(ok and complete, f"stlc complete: {term!r}")

    # Valid prefixes (ok=True, complete=False)
    for prefix in ["λx", "λx:Int", "λx:Int.", "λf:(Int->"]:
        ok, complete, _ = check_parse(spec, prefix)
        check(ok and not complete, f"stlc valid prefix: {prefix!r}")

    # Invalid (not even a prefix)
    for bad in ["xyz blah", "NOTVALID"]:
        ok, complete, _ = check_parse(spec, bad)
        check(not ok, f"stlc invalid: {bad!r}")

    spec_fun = proposition7.get_grammar("fun")

    for term in ["(x: Int) => x", "let x: Int = 1; x + 2", "((x: Int) => x + 1)(41)"]:
        ok, complete, _ = check_parse(spec_fun, term)
        check(ok and complete, f"fun complete: {term!r}")

    spec_imp = proposition7.get_grammar("imp")

    for prog in ["{ let x: Int = 5; }", "{ let x: Int = 1; let y: Int = x + 2; }"]:
        ok, complete, _ = check_parse(spec_imp, prog)
        check(ok and complete, f"imp complete: {prog!r}")


# ---------------------------------------------------------------------------
# Section 2: aufbau spacing / tokenization jank
# ---------------------------------------------------------------------------

def test_aufbau_spacing_jank() -> None:
    section("2. aufbau spacing / tokenization (the jank)")

    spec = proposition7.get_grammar("fun")

    # JANK: aufbau's parse() tokenizes the full character string.
    # A space between digits creates two grammar tokens, so '4 3' ≠ '43'.
    for text, expected_ok, desc in [
        ("43", True, "integer 43: single token"),
        ("4 3", False, "4 3: two tokens — fails as integer"),
        ("let x: Int = 1; x", True, "let with spaces: ok"),
    ]:
        ok, complete, _ = check_parse(spec, text)
        check(ok == expected_ok, desc, f"parse({text!r}) ok={ok} expected={expected_ok}")

    # The spacing workaround in llm.py: try token as-is first, then lstripped.
    # Simulate what happens when an LM produces ' 3' (space-prefixed) after '4'.
    acc = "4"
    for candidate in (" 3", "3"):          # as-is first, then stripped
        ok, _, _ = check_parse(spec, acc + candidate)
        if ok:
            check(True, f"digit continuation: '{acc}' + '{candidate}' → '{acc + candidate}'")
            break
    else:
        check(False, "digit continuation: neither variant valid")

    # Keyword boundary: ' x' after 'let'.  We want 'let x' not 'letx'.
    # Note: 'letx: Int = 1; x' DOES parse in the fun grammar (letx is a valid
    # identifier), which is the subtle jank — lstrip produces a different-but-valid
    # expression rather than an obvious error.  The workaround (try as-is first)
    # picks 'let x' since that also parses, and it is the correct interpretation.
    acc2 = "let"
    for candidate in (" x", "x"):
        ok, _, _ = check_parse(spec, acc2 + candidate)
        if ok:
            check(
                candidate == " x",
                f"keyword boundary: 'let' + '{candidate}' → '{acc2 + candidate}' (want space-preserved)",
                f"got candidate={candidate!r}",
            )
            break
    else:
        check(False, "keyword boundary: neither variant valid")

    # Document the silent-wrong-output jank:
    ok_letx, complete_letx, _ = check_parse(spec, "letx: Int = 1; x")
    check(
        ok_letx and complete_letx,
        "JANK: 'letx: Int = 1; x' parses as valid fun (letx is a valid identifier)"
        " — lstrip error is silent, not a parse failure",
    )


# ---------------------------------------------------------------------------
# Section 3: Prompt construction
# ---------------------------------------------------------------------------

def test_prompt_construction() -> None:
    section("3. Prompt construction (build_task_prompt)")

    tasks = load_tasks(["stlc"])
    task = tasks[0]
    gname = grammar_name(task.grammar)

    for mode in ["constrained_direct", "constrained_mixed", "unconstrained",
                 "unconstrained_thinking", "unconstrained_cleaned"]:
        prompt = build_task_prompt(gname, task.prompt, mode=mode, initial=task.initial)
        check(
            f"Language: {gname}" in prompt,
            f"{mode}: contains grammar name",
        )
        check(
            "Task:" in prompt,
            f"{mode}: contains task instruction",
        )
        if mode in {"constrained_mixed", "unconstrained_thinking"}:
            check(
                "think" in prompt.lower() or "workflow" in prompt.lower(),
                f"{mode}: contains think workflow hint",
            )
        if mode == "constrained_direct" and task.initial:
            check(
                task.initial in prompt,
                f"{mode}: prefix present in prompt when initial is set",
            )

    # Verify build_prompt (alias in benchmarks.api) gives the same result
    for mode in ["constrained_direct", "constrained_mixed"]:
        p1 = build_task_prompt(gname, task.prompt, mode=mode, initial=task.initial)
        p2 = build_prompt(gname, task.prompt, mode=mode, initial=task.initial)
        check(p1 == p2, f"build_prompt == build_task_prompt for mode={mode}")

    # Verify build_interaction matches build_task_prompt
    interaction = build_interaction(task, "constrained_direct")
    expected_prompt = build_task_prompt(
        gname, task.prompt, mode="constrained_direct", initial=task.initial
    )
    check(
        interaction.prompt == expected_prompt,
        "build_interaction.prompt matches build_task_prompt",
    )

    # build_system_prompt still works (used by ReasoningEnvironment)
    sys_prompt = build_system_prompt(gname)
    check("formal" in sys_prompt, "build_system_prompt: contains 'formal' block instruction")
    check("think" in sys_prompt.lower(), "build_system_prompt: contains think block instruction")


# ---------------------------------------------------------------------------
# Section 4: Benchmark config and dry-run
# ---------------------------------------------------------------------------

def test_benchmark_dry_run() -> None:
    section("4. Benchmark config parsing and dry-run")

    from pathlib import Path
    config_path = Path(__file__).resolve().parent.parent / "benchmarks" / "configs" / "minimum.toml"
    check(config_path.exists(), f"minimum.toml exists at {config_path}")
    if not config_path.exists():
        return

    config = load_benchmark_config(config_path)
    check(config.schema_version == 1, "schema_version == 1")
    check(len(config.matrices) > 0, f"at least one matrix ({len(config.matrices)} found)")

    tasks = selected_tasks(config)
    check(len(tasks) > 0, f"tasks loaded ({len(tasks)} tasks)")

    for matrix in config.matrices:
        modes = normalize_modes(matrix.modes, matrix.backend)
        jobs = build_jobs(tasks[:3], matrix.models, modes, 1, grammar_name)
        check(
            len(jobs) == 3 * len(matrix.models) * len(modes),
            f"matrix={matrix.name}: job count = tasks({3}) × models({len(matrix.models)}) × modes({len(modes)})",
        )
        for job in jobs:
            check(
                job.grammar_name in proposition7.list_grammars(),
                f"job grammar valid: {job.grammar_name}",
            )
            check(
                job.mode in BENCHMARK_MODES,
                f"job mode valid: {job.mode}",
            )

    # Verify all benchmark modes are accepted by the mode normalizer
    all_modes = list(BENCHMARK_MODES - {"outlines", "outlines_mixed"})
    try:
        normalized = normalize_modes(all_modes, "local")
        check(set(normalized) == set(all_modes), "all modes normalize cleanly")
    except SystemExit as e:
        check(False, f"normalize_modes raised SystemExit: {e}")


# ---------------------------------------------------------------------------
# Section 5: End-to-end with a real model (optional)
# ---------------------------------------------------------------------------

def test_with_model(model_name: str) -> None:
    section(f"5. End-to-end generation: {model_name}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    grammar = "stlc"
    spec = proposition7.get_grammar(grammar)

    model = proposition7.get_model_class(model_name).from_pretrained(
        model_name,
        grammar=spec,
        device=device,
        torch_dtype="auto" if device == "cuda" else "float32",
        device_map="auto" if device == "cuda" else None,
    )

    # --- constrained_direct ---
    prompt = build_task_prompt(grammar, "Write the identity function for Int",
                               mode="constrained_direct")
    result = model.generate_constrained(
        prompt=prompt, initial="λx:Int.", max_tokens=10, grammar_name=grammar,
        seed=7, temperature=0.0,
    )
    ok_parse, complete, err = check_parse(spec, result.text)
    check(result.text.startswith("λx:Int."), "constrained_direct: output starts with injected prefix")
    check(ok_parse, f"constrained_direct: output parses ({result.text!r})", err)
    check(complete, f"constrained_direct: grammar complete ({result.stopped_reason})")
    print(f"    output: {result.text!r}  stopped={result.stopped_reason}  tokens={result.tokens_generated}")

    # --- constrained generation never produces non-completable output ---
    check(
        result.stopped_reason != "no_valid" or complete,
        "constrained_direct: no_valid only when complete",
    )

    # --- unconstrained generation still parses (sanity check on prompts) ---
    result_unc = model.generate_unconstrained(
        prompt=prompt, initial="", max_tokens=20, grammar_name=grammar, seed=7, temperature=0.0,
    )
    print(f"    unconstrained output: {result_unc.text!r}")

    # --- constrained_mixed (ReasoningEnvironment) ---
    env = proposition7.ReasoningEnvironment(
        model=model,
        grammar_name=grammar,
        think_budget=64,
        formal_budget=20,
    )
    env_result = env.generate(
        prompt=build_task_prompt(grammar, "Write the identity function for Int",
                                 mode="constrained_mixed"),
        initial="",
    )
    check(len(env_result.think_blocks) == 1, "constrained_mixed: exactly one think block")
    check(len(env_result.formal_blocks) == 1, "constrained_mixed: exactly one formal block")
    if env_result.final_output:
        ok2, complete2, err2 = check_parse(spec, env_result.final_output.content)
        check(ok2, f"constrained_mixed: formal output parses", err2)
        print(f"    thoughts: {env_result.all_thoughts[:80]!r}")
        print(f"    formal: {env_result.final_output.content!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default=None,
        help="Optional HuggingFace model name for end-to-end tests (e.g. Qwen/Qwen3.5-0.5B)"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    test_aufbau_prefix_validation()
    test_aufbau_spacing_jank()
    test_prompt_construction()
    test_benchmark_dry_run()

    if args.model:
        test_with_model(args.model)
    else:
        print("\n[skip] Pass --model <name> to run end-to-end generation tests.")

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} assertion(s) failed:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"\033[32mAll assertions passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
