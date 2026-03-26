"""Typed sampler for constrained LLM generation."""

from __future__ import annotations

from typing import Callable, List, Optional
import math
import random
import os
import sys

# Enable runtime debugging via environment variable `P7_CONSTRAINED_DEBUG` or
# `P7_SAMPLER_DEBUG` (accepted values: 1/true/yes/on).
def _env_bool(name: str) -> bool:
    v = os.getenv(name, "")
    return str(v).lower() in ("1", "true", "yes", "on")

DEBUG_SAMPLER = _env_bool("P7_CONSTRAINED_DEBUG") or _env_bool("P7_SAMPLER_DEBUG")
_DEBUG_OVERRIDE: Optional[bool] = None
PROFILE_SAMPLER = _env_bool("P7_PROFILE")

if PROFILE_SAMPLER:
    import time
    _prof_timers = {}
    _prof_counts = {}
    
    def _prof_start(name):
        if PROFILE_SAMPLER:
            _prof_timers[name] = time.perf_counter()
    
    def _prof_end(name):
        if PROFILE_SAMPLER:
            elapsed = time.perf_counter() - _prof_timers.get(name, 0)
            _prof_counts[name] = _prof_counts.get(name, 0) + 1
            return elapsed
        return 0
    
    def _prof_report():
        if PROFILE_SAMPLER:
            print("\n=== SAMPLER PROFILING REPORT ===", file=sys.stderr)
            for name, count in sorted(_prof_counts.items(), key=lambda x: -x[1]):
                total = _prof_counts.get(name, 0)
                print(f"  {name}: count={count}", file=sys.stderr)
            print("==============================\n", file=sys.stderr)
else:
    _prof_timers = {}
    _prof_counts = {}
    
    def _prof_start(name):
        pass
    
    def _prof_end(name):
        return 0
    
    def _prof_report():
        pass

def _debug_enabled() -> bool:
    if _DEBUG_OVERRIDE is not None:
        return _DEBUG_OVERRIDE
    return DEBUG_SAMPLER or _env_bool("P7_CONSTRAINED_DEBUG") or _env_bool("P7_SAMPLER_DEBUG")


def _dbg(msg: str, /, *args, **kwargs) -> None:
    if not _debug_enabled():
        return

    try:
        rendered = msg.format(*args, **kwargs)
    except Exception as exc:
        rendered = (
            f"{msg} | <debug-format-error {exc!r}> "
            f"args={args!r} kwargs={kwargs!r}"
        )

    print("[p7-debug] " + rendered, file=sys.stderr, flush=True)

from .p7 import Grammar, Synthesizer, regex_matches
import re


class CompletionEngine:
    """Low-level completion engine wrapper."""
    
    def __init__(self, grammar_str: str):
        self.grammar = Grammar(grammar_str)
        self.synthesizer = Synthesizer(self.grammar, "")
    
    def feed(self, prompt: str) -> None:
        """Seed the synthesizer with an initial string."""
        if prompt:
            self.synthesizer.set_input(prompt)
    
    def reset(self) -> None:
        self.synthesizer = Synthesizer(self.grammar, "")

    def get_completions(self) -> List[str]:
        return self.synthesizer.get_completions()
    
    def current_text(self) -> str:
        return self.synthesizer.current_text()


class TypedSampler:
    """
    Typed sampler for constrained LLM generation.
    
    Filters LLM outputs to only well-typed completions using stateful Synthesizer.
    """
    
    def _get_patterns(self) -> tuple[set, list]:
        """Get literal patterns (set) and regex patterns (list)."""
        patterns = self.synthesizer.get_completions()
        
        literals = set()
        regexes = []
        for p in patterns:
            if not p:
                continue
            if p.startswith('^'):
                try:
                    regexes.append((p, re.compile(p)))
                except re.error:
                    pass
            else:
                literals.add(p)
        
        return literals, regexes
    
    def _token_matches_any(self, token: str, literals: set, regexes: list) -> bool:
        """Check if token matches any literal or regex pattern."""
        stripped = token.lstrip()
        
        if stripped in literals:
            return True
        if token in literals:
            return True
        
        for _, regex in regexes:
            if regex.match(token):
                return True
            if stripped != token and regex.match(stripped):
                return True
        
        return False
    
    def __init__(
        self,
        grammar: str,
        vocab: List[str],
        logit_fn: Callable[[], List[float]],
        input_text: str = "",
    ):
        self.grammar = Grammar(grammar)
        self.synthesizer = Synthesizer(self.grammar, input_text)
        self.vocab = vocab
        self.logit_fn = logit_fn
        self._neg_inf = -float('inf')
    
    def reset(self) -> None:
        self.set_input("")

    def set_input(self, text: str) -> None:
        """Seed the synthesizer with an initial string.

        Uses `set_input()` to seed arbitrary initial text (full strings),
        resetting synthesizer state to parse the given text from scratch.

        When `P7_CONSTRAINED_DEBUG=1` (or `P7_SAMPLER_DEBUG=1`) this will
        print the `set_input()` value plus `current_text()` and the set of
        valid completions after the reset so you can inspect synthesizer state.
        """
        self.synthesizer.set_input(text)

        # Debug: show set_input and completions/state
        _dbg("set_input(text={!r}) -> current='{}'", text, self.current_text())
        try:
            comps = self.synthesizer.get_completions()
            _dbg("  completions (n={}): {}", len(comps), comps)
        except Exception as e:
            _dbg("  completions: <error calling get_completions(): {}>", e)

    def feed(self, token: str) -> None:
        """Extend the synthesizer with one token during generation.

        Unlike `set_input()`, this accumulates state incrementally — it calls
        `synthesizer.extend(token)` which appends the token to the current
        text.  Raises `RuntimeError` if the token is not a valid extension.
        """
        self.synthesizer.extend(token)
        _dbg("feed(token={!r}) -> current='{}'", token, self.current_text())
        try:
            comps = self.synthesizer.get_completions()
            _dbg("  completions (n={}): {}", len(comps), comps)
        except Exception as e:
            _dbg("  completions: <error calling get_completions(): {}>", e)
    
    def current_text(self) -> str:
        return self.synthesizer.current_text()
    
    def infer(self, pre_top_k: Optional[int] = None) -> List[float]:
        """Get masked logits (invalid tokens set to -inf)."""
        _prof_start("infer_total")
        
        logits = list(self.logit_fn())
        _prof_start("infer_get_patterns")
        literals, regexes = self._get_patterns()
        _prof_end("infer_get_patterns")
        
        _prof_start("infer_validation")
        valid_set = set()
        for i, token in enumerate(self.vocab):
            if self._token_matches_any(token, literals, regexes):
                valid_set.add(i)
        _prof_end("infer_validation")
        
        _prof_start("infer_mask")
        for i in range(len(logits)):
            if i not in valid_set:
                logits[i] = self._neg_inf
        _prof_end("infer_mask")
        
        # Debug
        if _debug_enabled():
            valid_tokens = [self.vocab[i] for i in list(valid_set)[:10]]
            _dbg("infer(): valid_count={} samples={}", len(valid_tokens), valid_tokens)

        _prof_end("infer_total")
        return logits

    
    def infer_text(self, k: int = 10, pre_top_k: Optional[int] = None) -> List[str]:
        """Get top-k valid token strings sorted by logit."""
        _prof_start("infer_text_total")
        
        logits = self.logit_fn()
        
        _prof_start("infer_text_get_patterns")
        literals, regexes = self._get_patterns()
        _prof_end("infer_text_get_patterns")
        
        _prof_start("infer_text_validation")
        valid_pairs = []
        
        if pre_top_k is not None:
            indexed = [(i, l) for i, l in enumerate(logits)]
            indexed.sort(key=lambda x: x[1], reverse=True)
            candidates = indexed[:pre_top_k]
            
            for i, l in candidates:
                token = self.vocab[i]
                if self._token_matches_any(token, literals, regexes):
                    valid_pairs.append((i, l))
        else:
            for i, l in enumerate(logits):
                token = self.vocab[i]
                if self._token_matches_any(token, literals, regexes):
                    valid_pairs.append((i, l))
            valid_pairs.sort(key=lambda x: x[1], reverse=True)
        _prof_end("infer_text_validation")
        
        top_k = valid_pairs[:k]

        # Debug: print a sample of valid candidates for this turn.
        if _debug_enabled():
            full_valid = [(self.vocab[i], float(l)) for i, l in valid_pairs[:20]]
            _dbg("infer_text(k={}, pre_top_k={}): valid_pairs_count={} => {}", k, pre_top_k, len(valid_pairs), full_valid)

        _prof_end("infer_text_total")
        return [self.vocab[i] for i, _ in top_k]
    
    def infer_greedy(
        self, 
        k: int = 1, 
        pre_top_k: Optional[int] = None
    ) -> Optional[str]:
        """Pick one token from top-k valid. k=1 is pure greedy."""
        _prof_start("infer_greedy_total")
        
        logits = self.logit_fn()
        _prof_start("infer_greedy_get_patterns")
        literals, regexes = self._get_patterns()
        _prof_end("infer_greedy_get_patterns")
        
        _prof_start("infer_greedy_validation")
        valid_pairs = []
        
        if pre_top_k:
            indexed = [(i, l) for i, l in enumerate(logits)]
            indexed.sort(key=lambda x: x[1], reverse=True)
            candidates = indexed[:pre_top_k]

            for i, l in candidates:
                token = self.vocab[i]
                if self._token_matches_any(token, literals, regexes):
                    valid_pairs.append((i, l))
        else:
            for i, l in enumerate(logits):
                token = self.vocab[i]
                if self._token_matches_any(token, literals, regexes):
                    valid_pairs.append((i, l))
            valid_pairs.sort(key=lambda x: x[1], reverse=True)
        _prof_end("infer_greedy_validation")

        if not valid_pairs:
            _dbg("infer_greedy(k={}, pre_top_k={}): no valid tokens", k, pre_top_k)
            _prof_end("infer_greedy_total")
            return None

        if _debug_enabled():
            full_valid = [(self.vocab[i], float(l)) for i, l in valid_pairs[:10]]
            _dbg("infer_greedy(k={}, pre_top_k={}): valid_pairs_count={} => {}", k, pre_top_k, len(valid_pairs), full_valid)

        if k == 1 or len(valid_pairs) == 1:
            token = self.vocab[valid_pairs[0][0]]
            _dbg("infer_greedy: selected (greedy) {}", token)
            _prof_end("infer_greedy_total")
            return token


        top_k = valid_pairs[:k]
        max_logit = max(l for _, l in top_k)
        weights = [math.exp(l - max_logit) for _, l in top_k]
        total = sum(weights)
        if total <= 0:
            token = self.vocab[top_k[0][0]]
            _dbg("infer_greedy: selected (fallback) {}", token)
            _prof_end("infer_greedy_total")
            return token

        r = random.random() * total
        upto = 0.0
        for (idx, _), weight in zip(top_k, weights):
            upto += weight
            if upto >= r:
                token = self.vocab[idx]
                _dbg("infer_greedy: selected (weighted) {}", token)
                return token

        token = self.vocab[top_k[-1][0]]
        _dbg("infer_greedy: selected (tail fallback) {}", token)
        return token
    
    def infer_unconstrained(self, k: int = 1) -> Optional[str]:
        """Pick from top-k ignoring grammar. For comparison."""
        logits = self.logit_fn()
        
        indexed = [(i, l) for i, l in enumerate(logits)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_k = indexed[:k]
        
        if not top_k:
            return None
        
        if _debug_enabled():
            display = [(self.vocab[i], float(l)) for i, l in top_k]
            _dbg("infer_unconstrained(k={}): top_k={}", k, display)
        
        if k == 1:
            return self.vocab[top_k[0][0]]
        
        idx, _ = random.choice(top_k)
        return self.vocab[idx]
    
    def is_complete(self) -> bool:
        return self.synthesizer.is_complete()
    
    def check_completion(self, token: str) -> bool:
        """Check if a token is a valid extension."""
        valid_patterns = self.synthesizer.get_completions()
        stripped_token = token.lstrip()
        for p in valid_patterns:
            if p == stripped_token:
                return True
            try:
                if regex_matches(p, token) or (stripped_token != token and regex_matches(p, stripped_token)):
                    return True
            except Exception:
                if p == token:
                    return True
        return False


# Runtime helper to toggle debug (useful in REPL/tests)
def set_debug(enabled: bool) -> None:
    """Toggle sampler debug printing at runtime.

    Environment variable `P7_CONSTRAINED_DEBUG` (or `P7_SAMPLER_DEBUG`) still
    controls the initial default; use `set_debug(True)` to enable or
    `set_debug(False)` to silence debug output for the current process.
    """
    global DEBUG_SAMPLER, _DEBUG_OVERRIDE
    DEBUG_SAMPLER = bool(enabled)
    _DEBUG_OVERRIDE = bool(enabled)
    _dbg("set_debug({})", _DEBUG_OVERRIDE)
