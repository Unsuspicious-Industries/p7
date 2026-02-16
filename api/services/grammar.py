from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import p7 as p7
from p7.environment import build_system_prompt


def get_system_prompt_for_spec(spec: str) -> Optional[str]:
    """Return a system prompt with examples for built-in grammars."""
    for name in p7.list_grammars():
        try:
            if p7.get_grammar(name) == spec:
                return build_system_prompt(name, include_examples=True)
        except Exception:
            continue
    return None


def get_grammar_name_for_spec(spec: str) -> Optional[str]:
    """Return built-in grammar name if spec matches exactly."""
    spec_norm = spec.strip()
    for name in p7.list_grammars():
        try:
            if p7.get_grammar(name).strip() == spec_norm:
                return name
        except Exception:
            continue
    return None


def check_partial_completable(spec: str, text: str) -> tuple[bool, str]:
    """Check whether a partial string is completable under the grammar."""
    gen = p7.ConstrainedGenerator(p7.Grammar(spec))
    if text:
        if not gen.feed_raw(text):
            return False, "invalid_prefix"
        if not gen.is_complete() and len(gen.get_completions()) == 0:
            return False, "not_completable"
    return True, ""


def extract_syntax_hints(spec: str) -> List[str]:
    heads: List[str] = []
    seen = set()
    for line in spec.splitlines():
        if "::=" not in line:
            continue
        head = line.split("::=", 1)[0].strip()
        if head and head not in seen:
            heads.append(head)
            seen.add(head)

    hints = []
    if heads:
        hints.append(f"Nonterminals: {', '.join(heads[:8])}")
        if len(heads) > 8:
            hints.append(f"...and {len(heads) - 8} more")

    try:
        start_symbol = p7.Grammar(spec).start_nonterminal()
        if start_symbol:
            hints.append(f"Start symbol: {start_symbol}")
    except Exception:
        pass

    return hints


def build_fallback_system_prompt(spec: str) -> str:
    hints = extract_syntax_hints(spec)
    lines = [
        "You are a reasoning assistant that produces well-typed output.",
        "",
        "Use the grammar spec below to guide syntax.",
    ]
    if hints:
        lines.extend(["", "Syntax:"])
        for hint in hints:
            lines.append(f"  - {hint}")
    return "\n".join(lines)


def append_stop_instructions(system_prompt: str, adapter, grammar_name: Optional[str]) -> str:
    if not system_prompt or adapter is None:
        return system_prompt
    stop_tokens = adapter.stop_tokens(grammar_name or "grammar")
    stop_hint = ", ".join(stop_tokens)
    return (
        f"{system_prompt}\n\n"
        f"Stop tokens: {stop_hint}.\n"
        "Stop when you emit a stop token."
    )


def build_prompt_context(spec: str, prompt: str, model_name: str):
    grammar_name = get_grammar_name_for_spec(spec)
    base_prompt = get_system_prompt_for_spec(spec) or build_fallback_system_prompt(spec)
    system_prompt = f"{base_prompt}\n\nLanguage spec:\n{spec}"
    full_prompt = f"{system_prompt}\n\n{prompt}"

    return full_prompt, system_prompt, grammar_name, None

# Grammar validation logic

@dataclass
class GrammarValidationResult:
    valid: bool
    errors: List[str]
    start_nonterminal: Optional[str] = None


def validate_grammar(spec: str) -> GrammarValidationResult:
    """Validate a grammar spec and return detailed error information."""
    errors = []
    start_nt = None
    
    try:
        grammar = p7.Grammar(spec)
        start_nt = grammar.start_nonterminal()
        return GrammarValidationResult(
            valid=True,
            errors=[],
            start_nonterminal=start_nt
        )
    except BaseException as e:
        error_msg = str(e)
        errors.append(error_msg)
        
        # Try to provide more helpful error messages
        if "Grammar parse error" in error_msg:
            # Extract line number if available
            if "line" in error_msg.lower():
                errors.append("Check syntax around the indicated line number")
            else:
                errors.append("Common issues:")
                errors.append("  - Missing '::=' in production rules")
                errors.append("  - Unmatched parentheses or quotes")
                errors.append("  - Invalid regex patterns")
                errors.append("  - Typing rules must be separated from grammar by blank lines")
        
        return GrammarValidationResult(
            valid=False,
            errors=errors,
            start_nonterminal=None
        )
