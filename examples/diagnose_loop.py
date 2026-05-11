#!/usr/bin/env python3
"""
Logit-level diagnostic for constrained generation looping.

Two modes:
  --grammar-only   (default when no GPU/model available)
      Checks grammar state at each step of a known looping output.
      Answers: "what tokens COULD close the expression at the looping point?"
      No model required.

  --model MODEL    (needs GPU)
      Full logit inspection: shows raw model probabilities alongside grammar
      validity for every top-K token at each decode step.
      Answers: "does the model assign high prob to looping tokens vs. closers?"

Usage:
    # Grammar-only (always works):
    python examples/diagnose_loop.py
    python examples/diagnose_loop.py --grammar stlc --initial "λx:Int."

    # With model (needs GPU):
    python examples/diagnose_loop.py --model Qwen/Qwen3.5-0.8B
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import aufbau
import proposition7
from proposition7 import build_task_prompt

_G = "\033[32m"   # green  — grammar-valid
_C = "\033[36m"   # cyan   — grammar-complete
_S = "\033[33m"   # yellow — stop token
_R = "\033[31m"   # red    — grammar-invalid / masked
_B = "\033[1m"
_N = "\033[0m"


# ---------------------------------------------------------------------------
# Shared grammar helpers
# ---------------------------------------------------------------------------

def check_grammar(grammar_spec: str, text: str) -> tuple[bool, bool]:
    """Return (is_valid_prefix, is_complete)."""
    synth = aufbau.Synthesizer(grammar_spec, "")
    synth.set_input(text)
    try:
        synth.parse()
        return True, synth.is_complete()
    except RuntimeError:
        return False, False


def grammar_analysis_step(
    grammar_spec: str,
    accumulated: str,
    candidates: list[tuple[str, str]],  # (label, token_text)
) -> None:
    """Print grammar validity of each candidate at the current position."""
    valid_prefix, is_complete = check_grammar(grammar_spec, accumulated)

    print(f"\n  prefix_valid={valid_prefix}  already_complete={is_complete}")
    print(f"  {'token':<28}  status")
    print(f"  {'─'*28}  {'─'*40}")
    for label, token in candidates:
        v, c = check_grammar(grammar_spec, accumulated + token)
        if c:
            status = f"{_C}VALID + COMPLETE{_N}"
        elif v:
            status = f"{_G}VALID prefix{_N}"
        else:
            status = f"{_R}INVALID (would be masked){_N}"
        print(f"  {label:<28}  {status}")


# ---------------------------------------------------------------------------
# Mode 1: Grammar-only — trace the observed looping output
# ---------------------------------------------------------------------------

# Candidates we want to try at every looping position
CLOSING_CANDIDATES: list[tuple[str, str]] = [
    ("';' (stmt sep)",              ";"),
    ("'; times2' (complete)",       "; times2"),
    ("'*' (multiply — the loop)",   " *"),
    ("' x' (ident)",                " x"),
    (r"'\n' (newline)",             "\n"),
    ("' 2' (integer)",              " 2"),
    ("'(2)' (parens int)",          "(2)"),
    ("' (2)' (parens int sp)",      " (2)"),
    ("' + 1' (add)",                " + 1"),
    ("')' (close paren)",           ")"),
    ("') ; times2' (close+return)", ") ; times2"),
]


def run_grammar_only(grammar_name: str, initial: str) -> None:
    grammar_spec = proposition7.get_grammar(grammar_name)

    print(f"\n{'═'*70}")
    print(f"  Grammar-only loop diagnostic")
    print(f"  grammar: {grammar_name}   initial: {initial!r}")
    print(f"{'═'*70}")

    # ── Step 0: initial state ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"[initial]  {initial!r}")
    grammar_analysis_step(grammar_spec, initial, CLOSING_CANDIDATES)

    # ── Simulate first few tokens that the model emits in constrained mode ──
    # These come from the observed benchmark output for fun/constrained_direct.
    loop_suffix_tokens = [
        " (x",        # model writes "let x: Int = (x..."
        ": Int)",     # "..."
        " =>",        # lambda arrow
        " x",         # body starts
        " *",         # multiplication — the loop begins
        " x",
        " *",
        " x",
        " *",
    ]

    acc = initial
    for i, tok in enumerate(loop_suffix_tokens):
        v, _ = check_grammar(grammar_spec, acc + tok)
        if not v:
            print(f"\n  Token {i+1} {tok!r} rejected by grammar — stopping trace")
            break
        acc = acc + tok
        step_label = f"after token {i+1}: ...{acc[-30:]!r}"
        print(f"\n{'─'*70}")
        print(f"[step {i+1:02d}]  accumulated tail: ...{acc[-50:]!r}")
        grammar_analysis_step(grammar_spec, acc, CLOSING_CANDIDATES)

    # ── Deep analysis: at the looping point, find ALL single-char completions ──
    print(f"\n{'═'*70}")
    print(f"  Exhaustive single-char scan at looping position")
    print(f"  accumulated tail: ...{acc[-40:]!r}")
    print(f"{'═'*70}")

    completing: list[tuple[str, str]] = []
    valid_non_completing: list[str] = []

    for c in " \t\n;()[].,+-><=!|&*/%^~@#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:\"'`\\":
        v, complete = check_grammar(grammar_spec, acc + c)
        if complete:
            completing.append((repr(c), acc + c))
        elif v:
            valid_non_completing.append(repr(c))

    if completing:
        print(f"\n  {_C}Completing continuations exist — NOT a grammar dead-end:{_N}")
        for label, full in completing[:10]:
            print(f"    {label}  →  {full!r}")
    else:
        print(f"\n  {_R}No single-char completing continuation found at this position.{_N}")
        print(f"  This means the grammar is NOT completable from here in one character.")
        print(f"  Checking longer completions...")
        # Try common multi-char closers
        for closer in ["; times2", "; nope", "; applytwiceoffset", "2; times2", " 2; times2"]:
            v, c = check_grammar(grammar_spec, acc + closer)
            if c:
                print(f"    {_C}VALID+COMPLETE:{_N} + {closer!r}")
            elif v:
                print(f"    {_G}VALID prefix:{_N}   + {closer!r}")

    print(f"\n  Valid non-completing single chars ({len(valid_non_completing)}):")
    print(f"    {' '.join(valid_non_completing[:40])}")

    # ── Conclusion ──────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  CONCLUSION")
    print(f"{'═'*70}")
    if completing:
        print(f"""
  The grammar CAN be completed from the looping position.
  Completing tokens exist but the model never chooses them.
  → This is a MODEL QUALITY issue (A), not a masking bug.
  → The model assigns higher probability to continuation tokens
    (like ' *', ' x') than to closing tokens (like ';').
  → Constrained generation faithfully reflects the model's
    distribution; it doesn't get stuck by masking.
""")
    else:
        print(f"""
  No completing continuation found from the looping position.
  → The model has generated a dead-end prefix from which the
    grammar can never be completed.
  → This is a GRAMMAR / PREFIX issue.
  → The model should be steered away from this prefix via prompting
    or the initial should be refined.
""")


# ---------------------------------------------------------------------------
# Mode 2: Full logit inspection (needs a model on GPU)
# ---------------------------------------------------------------------------

def classify_token(
    token_id: int,
    token_decoded: str,
    accumulated: str,
    grammar_spec: str,
    stop_token_ids: set,
    stop_tokens: set,
    synth: aufbau.Synthesizer,
) -> tuple[bool, bool, bool]:
    is_stop = token_id in stop_token_ids or token_decoded in stop_tokens
    if not token_decoded or token_decoded.strip() == "":
        return is_stop, False, False
    for candidate in (token_decoded, token_decoded.lstrip()):
        if not candidate:
            continue
        synth.set_input(accumulated + candidate)
        try:
            synth.parse()
            return is_stop, True, synth.is_complete()
        except RuntimeError:
            pass
    return is_stop, False, False


def decode_step_report(step, model, accumulated, grammar_spec,
                       stop_token_ids, stop_tokens, top_k=20):
    import torch
    logits = model._get_logits_tensor().float()
    probs = torch.softmax(logits, dim=-1)
    vocab_size = logits.shape[0]

    # Sample vocab to estimate valid/completing counts
    sample_size = min(8192, vocab_size)
    sample_ids = torch.randperm(vocab_size)[:sample_size].tolist()
    synth = aufbau.Synthesizer(grammar_spec, "")
    n_valid = n_completing = 0
    for tid in sample_ids:
        _, gv, gc = classify_token(tid, model._decode_token_id(tid),
                                   accumulated, grammar_spec,
                                   stop_token_ids, stop_tokens, synth)
        n_valid += gv
        n_completing += gc
    est_valid = int(n_valid / sample_size * vocab_size)
    est_completing = int(n_completing / sample_size * vocab_size)

    top_ids = torch.topk(probs, min(top_k, vocab_size)).indices.tolist()
    rows = []
    for tid in top_ids:
        tok = model._decode_token_id(tid)
        p = probs[tid].item()
        is_stop, is_valid, is_complete = classify_token(
            tid, tok, accumulated, grammar_spec,
            stop_token_ids, stop_tokens, synth)
        rows.append((tid, tok, p, is_stop, is_valid, is_complete))

    print(f"\n{'─'*70}")
    print(f"Step {step:3d}  ...{accumulated[-40:]!r}")
    print(f"  ~{est_valid:,}/{vocab_size:,} grammar-valid  ~{est_completing} completing  [sample={sample_size}]")
    print(f"\n  {'prob':>8}  {'id':>7}  {'fl':4}  token")

    chosen_id = chosen_tok = None
    for tid, tok, p, is_stop, is_valid, is_complete in rows:
        flag = ("S" if is_stop else "") + ("V" if is_valid else "") + ("C" if is_complete else "")
        colour = _S if is_stop else (_C if is_complete else (_G if is_valid else _R))
        marker = " "
        if chosen_id is None and (is_stop or is_valid):
            marker = "►"
            chosen_id, chosen_tok = tid, tok
        print(f"{marker} {p:8.4f}  {tid:7d}  {colour}{flag:<4}{_N}  {tok!r}")

    # Show EOS probability even if outside top-K
    for tid in stop_token_ids:
        if tid < vocab_size and not any(r[0] == tid for r in rows):
            p = probs[tid].item()
            rank = (probs > probs[tid]).sum().item() + 1
            tok = model._decode_token_id(tid)
            print(f"  {_S}[EOS id={tid} rank={rank:,} p={p:.6f} {tok!r}  ← not in top-{top_k}]{_N}")

    if chosen_id is None:
        print("  (no valid token in top-K — scanning...)")
        sorted_ids = torch.argsort(probs, descending=True).tolist()
        synth2 = aufbau.Synthesizer(grammar_spec, "")
        for tid in sorted_ids:
            tok = model._decode_token_id(tid)
            is_stop, is_valid, _ = classify_token(
                tid, tok, accumulated, grammar_spec,
                stop_token_ids, stop_tokens, synth2)
            if is_stop or is_valid:
                rank = sorted_ids.index(tid) + 1
                print(f"  First valid rank={rank} p={probs[tid]:.4f} {tok!r}")
                chosen_id, chosen_tok = tid, tok
                break

    # Compute new accumulated
    new_accumulated = accumulated
    if chosen_id is not None:
        is_stop_c, is_valid_c, _ = classify_token(
            chosen_id, chosen_tok, accumulated, grammar_spec,
            stop_token_ids, stop_tokens, synth)
        if is_valid_c and not is_stop_c:
            for candidate in (chosen_tok, chosen_tok.lstrip()):
                if candidate:
                    synth.set_input(accumulated + candidate)
                    try:
                        synth.parse()
                        new_accumulated = accumulated + candidate
                        break
                    except RuntimeError:
                        pass

    return chosen_id, chosen_tok, new_accumulated


def run_with_model(model_name, grammar_name, task_prompt, initial, steps, top_k):
    import torch
    print(f"\n{'═'*70}")
    print(f"  Logit diagnostic  model={model_name}  grammar={grammar_name}")
    print(f"{'═'*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("WARNING: CPU inference will be very slow.")

    # Avoid device_map — it puts the embedding table on CPU and confuses the
    # device detection in from_pretrained, causing all input tensors to end up
    # on CPU even when the model weights are on GPU.
    model = proposition7.get_model_class(model_name).from_pretrained(
        model_name,
        grammar=proposition7.get_grammar(grammar_name),
        device=device,
        torch_dtype="auto",
    )
    grammar_spec = proposition7.get_grammar(grammar_name)
    stop_tokens = set(model.stop_tokens_constrained(grammar_name))
    stop_token_ids = set(model._stop_token_ids(list(stop_tokens)))

    prompt = build_task_prompt(grammar_name, task_prompt,
                               mode="constrained_direct", initial=initial)
    model._set_prompt(prompt, initial, model.start_tokens_constrained(grammar_name))

    accumulated = initial
    for step in range(steps):
        chosen_id, chosen_tok, accumulated = decode_step_report(
            step, model, accumulated, grammar_spec,
            stop_token_ids, stop_tokens, top_k=top_k)
        if chosen_id is None:
            break
        is_stop = chosen_id in stop_token_ids or chosen_tok in stop_tokens
        if is_stop:
            print(f"\n  Stop token emitted: {chosen_tok!r}")
            break
        model._append_token_id(chosen_id)

    print(f"\n{'═'*70}")
    print(f"Final: {accumulated!r}")
    v, c = check_grammar(proposition7.get_grammar(grammar_name), accumulated)
    print(f"Grammar: valid={v}  complete={c}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None,
                   help="HuggingFace model name for full logit mode (needs GPU)")
    p.add_argument("--grammar", default="fun")
    p.add_argument(
        "--task",
        default="Define a function that doubles its input, then return that function as the final expression.",
    )
    p.add_argument("--initial", default="let times2: Int -> Int =")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    if args.model:
        run_with_model(args.model, args.grammar, args.task, args.initial,
                       args.steps, args.top_k)
    else:
        run_grammar_only(args.grammar, args.initial)


if __name__ == "__main__":
    main()
