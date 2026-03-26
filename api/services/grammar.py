from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import p7 as p7


def check_partial_completable(spec: str, text: str) -> tuple[bool, str]:
    """Check whether a partial string is completable under the grammar."""
    try:
        grammar = p7.Grammar(spec)
        synth = p7.Synthesizer(grammar, text if text else "")
        if not synth.is_complete() and len(synth.get_completions()) == 0:
            return False, "not_completable"
    except RuntimeError:
        return False, "invalid_prefix"
    return True, ""


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
