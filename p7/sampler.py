"""Typed sampler for constrainted LLM generation."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union
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

def _dbg(msg: str, /, *args, **kwargs) -> None:
    if DEBUG_SAMPLER:
        print("[p7-sampler-debug] " + msg.format(*args, **kwargs), file=sys.stderr)

from p7.p7 import Grammar, ConstrainedGenerator


class CompletionEngine:
    """Low-level completoin engine wrapper."""
    
    def __init__(self, grammar_str: str):
        self.grammar = Grammar(grammar_str)
        self.generator = ConstrainedGenerator(self.grammar)
    
    def feed(self, prompt: str) -> None:
        self.generator.feed_raw(prompt)
    
    def reset(self) -> None:
        self.generator.reset()

    def get_completions(self) -> List[str]:
        return self.generator.get_completions()
    
    def debug_completions(self) -> Any:
        return self.generator.debug_completions()


class TypedSampler:
    """
    Typed sampler for constrainted LLM generation.
    
    Filters LLM outputs to only well-typed completions.
    """
    
    def __init__(
        self,
        grammar: str,
        vocab: List[str],
        logit_fn: Callable[[], List[float]],
    ):
        self.grammar = Grammar(grammar)
        self.generator = ConstrainedGenerator(self.grammar)
        self.vocab = vocab
        self.logit_fn = logit_fn
        self._neg_inf = -float('inf')
    
    def reset(self) -> None:
        self.generator.reset()
    
    def feed(self, text: str) -> None:
        """Feed text. Raises TypeError if ill-typed.

        When `P7_CONSTRAINED_DEBUG=1` (or `P7_SAMPLER_DEBUG=1`) this will
        print the `feed()` input plus `current_text()` and the set of valid
        completions after the feed so you can inspect parser/sampler state.
        """
        self.generator.feed_raw(text)

        # Debug: show feed and completions/state
        _dbg("feed(text={!r}) -> current='{}'", text, self.current_text())
        try:
            comps = self.generator.get_completions()
            _dbg("  completions (n={}): {}", len(comps), comps)
        except Exception as e:
            _dbg("  completions: <error calling get_completions(): {}>", e)

        try:
            dbg = self.generator.debug_completions()
            _dbg("  debug_completions: {}", dbg)
        except Exception:
            # debug_completions is optional / may raise for some backends
            pass
    
    def current_text(self) -> str:
        return self.generator.current_text()
    
    def infer(self, pre_top_k: Optional[int] = None) -> List[float]:
        """Get masked logits (invalid tokens set to -inf)."""
        logits = list(self.logit_fn())
        
        if pre_top_k is not None and pre_top_k < len(logits):
            indexed = [(i, l) for i, l in enumerate(logits)]
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = [i for i, _ in indexed[:pre_top_k]]
            
            top_k_tokens = [self.vocab[i] for i in top_k_indices]
            valid_tokens = self.generator.filter_completions(top_k_tokens)
            valid_set = set(valid_tokens)
            
            valid_indices = {i for i in top_k_indices if self.vocab[i] in valid_set}
        else:
            valid_indices = set(self.generator.filter_completion_indices(self.vocab))
        
        for i in range(len(logits)):
            if i not in valid_indices:
                logits[i] = self._neg_inf
        
        # Debug: summarize valid completions for this turn
        try:
            valid_tokens = [self.vocab[i] for i in sorted(valid_indices)]
            _dbg("infer(): valid_count={} tokens={}", len(valid_tokens), valid_tokens)
        except Exception:
            _dbg("infer(): valid_count={} (unable to list tokens)", len(valid_indices))

        return logits
    
    def infer_text(self, k: int = 10, pre_top_k: Optional[int] = None) -> List[str]:
        """Get top-k valid token strings sorted by logit."""
        logits = self.logit_fn()
        
        if pre_top_k is not None:
            indexed = [(i, l) for i, l in enumerate(logits)]
            indexed.sort(key=lambda x: x[1], reverse=True)
            candidates = indexed[:pre_top_k]
            
            candidate_tokens = [self.vocab[i] for i, _ in candidates]
            valid_tokens = set(self.generator.filter_completions(candidate_tokens))
            
            valid_pairs = [(i, l) for i, l in candidates if self.vocab[i] in valid_tokens]
        else:
            valid_indices = self.generator.filter_completion_indices(self.vocab)
            if not valid_indices:
                return []
            valid_pairs = [(i, logits[i]) for i in valid_indices]
            valid_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_k = valid_pairs[:k]

        # Debug: print full set of valid candidates (token, logit) for this turn
        if DEBUG_SAMPLER:
            full_valid = [(self.vocab[i], float(l)) for i, l in valid_pairs]
            _dbg("infer_text(k={}, pre_top_k={}): valid_pairs_count={} => {}", k, pre_top_k, len(full_valid), full_valid)

        return [self.vocab[i] for i, _ in top_k]
    
    def infer_greedy(
        self, 
        k: int = 1, 
        pre_top_k: Optional[int] = None
    ) -> Optional[str]:
        """Pick one token from top-k valid. k=1 is pure greedy."""
        logits = self.logit_fn()

        if pre_top_k:
            indexed = [(i, l) for i, l in enumerate(logits)]
            indexed.sort(key=lambda x: x[1], reverse=True)
            candidates = indexed[:pre_top_k]

            candidate_tokens = [self.vocab[i] for i, _ in candidates]
            valid_tokens = set(self.generator.filter_completions(candidate_tokens))
            valid_pairs = [(i, l) for i, l in candidates if self.vocab[i] in valid_tokens]
        else:
            valid_indices = self.generator.filter_completion_indices(self.vocab)
            if not valid_indices:
                _dbg("infer_greedy(k={}, pre_top_k={}): no valid tokens", k, pre_top_k)
                return None
            valid_pairs = [(i, logits[i]) for i in valid_indices]
            valid_pairs.sort(key=lambda x: x[1], reverse=True)

        if not valid_pairs:
            _dbg("infer_greedy(k={}, pre_top_k={}): no valid tokens", k, pre_top_k)
            return None

        if DEBUG_SAMPLER:
            full_valid = [(self.vocab[i], float(l)) for i, l in valid_pairs]
            _dbg("infer_greedy(k={}, pre_top_k={}): valid_pairs_count={} => {}", k, pre_top_k, len(full_valid), full_valid)

        if k == 1 or len(valid_pairs) == 1:
            token = self.vocab[valid_pairs[0][0]]
            _dbg("infer_greedy: selected (greedy) {}", token)
            return token

        top_k = valid_pairs[:k]
        max_logit = max(l for _, l in top_k)
        weights = [math.exp(l - max_logit) for _, l in top_k]
        total = sum(weights)
        if total <= 0:
            token = self.vocab[top_k[0][0]]
            _dbg("infer_greedy: selected (fallback) {}", token)
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
        """Pick from top-k ignoring grammar. For comparision."""
        logits = self.logit_fn()
        
        indexed = [(i, l) for i, l in enumerate(logits)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_k = indexed[:k]
        
        if not top_k:
            return None
        
        if DEBUG_SAMPLER:
            display = [(self.vocab[i], float(l)) for i, l in top_k]
            _dbg("infer_unconstrained(k={}): top_k={}", k, display)
        
        if k == 1:
            return self.vocab[top_k[0][0]]
        
        idx, _ = random.choice(top_k)
        return self.vocab[idx]
    
    def is_complete(self) -> bool:
        return self.generator.is_complete()
    
    def check_completion(self, token: str) -> bool:
        return self.generator.check_completion(token)


# Runtime helper to toggle debug (useful in REPL/tests)
def set_debug(enabled: bool) -> None:
    """Toggle sampler debug printing at runtime.

    Environment variable `P7_CONSTRAINED_DEBUG` (or `P7_SAMPLER_DEBUG`) still
    controls the initial default; use `set_debug(True)` to enable or
    `set_debug(False)` to silence debug output for the current process.
    """
    global DEBUG_SAMPLER
    DEBUG_SAMPLER = bool(enabled)
    _dbg("set_debug({})", DEBUG_SAMPLER)
