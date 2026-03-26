"""High-level inference for typed constrained generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .sampler import TypedSampler, _dbg, _debug_enabled


@dataclass
class GenerationResult:
    text: str
    is_complete: bool
    tokens_generated: int
    stopped_reason: str  # max_tokens | complete | no_valid | type_error


def _preview_tokens(tokens: List[str], limit: int = 10) -> List[str]:
    if len(tokens) <= limit:
        return list(tokens)
    return list(tokens[:limit]) + [f"...(+{len(tokens) - limit} more)"]


def _log_result(label: str, result: GenerationResult) -> None:
    _dbg(
        "{} result: text={!r} is_complete={} tokens_generated={} stopped_reason={!r}",
        label,
        result.text,
        result.is_complete,
        result.tokens_generated,
        result.stopped_reason,
    )


def _is_parse_failed_error(error: BaseException) -> bool:
    return isinstance(error, RuntimeError) and "parse failed" in str(error).lower()


def _select_and_feed_token(
    sampler: TypedSampler,
    greedy_k: int,
    pre_top_k: Optional[int],
    use_feed_only: bool = False,
) -> Optional[str]:
    """Pick next token then validate via feed().

    Default path uses completion-based candidate filtering. Feed-only mode is
    opt-in and tries tokens by raw logit rank.
    """
    _dbg(
        "_select_and_feed_token: current_text={!r} greedy_k={} pre_top_k={}",
        sampler.current_text(),
        greedy_k,
        pre_top_k,
    )
    if use_feed_only:
        logits = list(sampler.logit_fn())
        indexed = [(i, l) for i, l in enumerate(logits)]
        indexed.sort(key=lambda x: x[1], reverse=True)

        candidate_batches: List[List[str]] = []
        if pre_top_k is not None:
            candidate_batches.append([sampler.vocab[i] for i, _ in indexed[:pre_top_k]])
            candidate_batches.append([sampler.vocab[i] for i, _ in indexed])
        else:
            candidate_batches.append([sampler.vocab[i] for i, _ in indexed])

        if _debug_enabled():
            _dbg(
                "_select_and_feed_token: feed-only candidate_batches={}",
                [len(batch) for batch in candidate_batches],
            )

        attempted = set()
        for batch_idx, batch in enumerate(candidate_batches):
            _dbg(
                "_select_and_feed_token: candidate batch {} sample={}",
                batch_idx,
                _preview_tokens(batch, limit=20),
            )
            for token in batch:
                if token in attempted:
                    continue
                attempted.add(token)
                try:
                    sampler.feed(token)
                    _dbg(
                        "_select_and_feed_token: accepted batch={} token={!r}",
                        batch_idx,
                        token,
                    )
                    return token
                except RuntimeError as error:
                    if _is_parse_failed_error(error):
                        continue
                    _dbg(
                        "_select_and_feed_token: token={!r} raised non-parse error={}",
                        token,
                        error,
                    )
                    raise
    else:
        next_token = sampler.infer_greedy(k=greedy_k, pre_top_k=pre_top_k)
        _dbg("_select_and_feed_token: primary_candidate={!r}", next_token)
        if next_token is None:
            _dbg("_select_and_feed_token: no primary candidate")
            return None

        try:
            sampler.feed(next_token)
            _dbg("_select_and_feed_token: accepted primary candidate={!r}", next_token)
            return next_token
        except RuntimeError as error:
            if not _is_parse_failed_error(error):
                _dbg(
                    "_select_and_feed_token: primary candidate={!r} raised non-parse error={}",
                    next_token,
                    error,
                )
                raise
            _dbg(
                "_select_and_feed_token: primary candidate={!r} rejected by parse error={}",
                next_token,
                error,
            )

        candidate_limit = len(getattr(sampler, "vocab", [])) or max(greedy_k, 1)
        attempted = {next_token}
        candidate_batches = [sampler.infer_text(k=candidate_limit, pre_top_k=pre_top_k)]

        if pre_top_k is not None and candidate_limit > pre_top_k:
            candidate_batches.append(sampler.infer_text(k=candidate_limit, pre_top_k=None))

        if _debug_enabled():
            batch_sizes = [len(batch) for batch in candidate_batches]
            _dbg(
                "_select_and_feed_token: fallback candidate_limit={} batch_sizes={}",
                candidate_limit,
                batch_sizes,
            )

        for batch_idx, batch in enumerate(candidate_batches):
            _dbg(
                "_select_and_feed_token: fallback batch {} candidates={}",
                batch_idx,
                _preview_tokens(batch, limit=20),
            )
            for token in batch:
                if token in attempted:
                    continue
                attempted.add(token)

                try:
                    sampler.feed(token)
                    _dbg(
                        "_select_and_feed_token: accepted fallback candidate batch={} token={!r}",
                        batch_idx,
                        token,
                    )
                    return token
                except RuntimeError as error:
                    if _is_parse_failed_error(error):
                        _dbg(
                            "_select_and_feed_token: rejected fallback candidate batch={} token={!r} error={}",
                            batch_idx,
                            token,
                            error,
                        )
                        continue
                    _dbg(
                        "_select_and_feed_token: fallback candidate batch={} token={!r} raised non-parse error={}",
                        batch_idx,
                        token,
                        error,
                    )
                    raise

    _dbg("_select_and_feed_token: exhausted all candidates without success")
    return None


def generate(
    grammar: str,
    vocab: List[str],
    logit_fn: Callable[[], List[float]],
    initial: str = "",
    max_tokens: int = 50,
    greedy_k: int = 1,
    pre_top_k: Optional[int] = 100,
    use_feed_only: bool = False,
    on_token: Optional[Callable[[str, int], None]] = None,
) -> tuple[GenerationResult, TypedSampler]:
    """Generate tokens using typed constrained decoding."""
    _dbg(
        "inference.generate start: initial={!r} max_tokens={} greedy_k={} pre_top_k={} vocab_size={}",
        initial,
        max_tokens,
        greedy_k,
        pre_top_k,
        len(vocab),
    )
    sampler = TypedSampler(grammar=grammar, vocab=vocab, logit_fn=logit_fn)
    
    if initial:
        try:
            sampler.set_input(initial)
        except TypeError as e:
            result = GenerationResult(
                text=initial,
                is_complete=False,
                tokens_generated=0,
                stopped_reason=f"type_error: {e}"
            )
            _log_result("inference.generate(initial type error)", result)
            return result, sampler
    
    tokens_generated = 0
    stopped_reason = "max_tokens"
    
    for step in range(max_tokens):
        try:
            next_token = _select_and_feed_token(
                sampler=sampler,
                greedy_k=greedy_k,
                pre_top_k=pre_top_k,
                use_feed_only=use_feed_only,
            )
        except Exception as e:
            stopped_reason = f"type_error: {e}"
            _dbg("inference.generate step {}: selection error={}", step, e)
            break

        if next_token is None:
            stopped_reason = "no_valid"
            _dbg("inference.generate step {}: no valid token", step)
            break

        tokens_generated += 1
        _dbg(
            "inference.generate step {}: token={!r} text={!r}",
            step,
            next_token,
            sampler.current_text(),
        )

        if on_token:
            on_token(next_token, step)
    
    result = GenerationResult(
        text=sampler.current_text(),
        is_complete=sampler.is_complete(),
        tokens_generated=tokens_generated,
        stopped_reason=stopped_reason
    )
    _log_result("inference.generate", result)
    return result, sampler


def until_complete(
    grammar: str,
    vocab: List[str],
    logit_fn: Callable[[], List[float]],
    initial: str = "",
    max_tokens: int = 100,
    greedy_k: int = 1,
    pre_top_k: Optional[int] = 100,
    use_feed_only: bool = False,
    on_token: Optional[Callable[[str, int], None]] = None,
) -> tuple[GenerationResult, TypedSampler]:
    """Generate until the parse is complete or max_tokens reached."""
    _dbg(
        "inference.until_complete start: initial={!r} max_tokens={} greedy_k={} pre_top_k={} vocab_size={}",
        initial,
        max_tokens,
        greedy_k,
        pre_top_k,
        len(vocab),
    )
    sampler = TypedSampler(grammar=grammar, vocab=vocab, logit_fn=logit_fn)
    
    if initial:
        try:
            sampler.set_input(initial)
        except TypeError as e:
            result = GenerationResult(
                text=initial,
                is_complete=False,
                tokens_generated=0,
                stopped_reason=f"type_error: {e}"
            )
            _log_result("inference.until_complete(initial type error)", result)
            return result, sampler
    
    if sampler.is_complete():
        result = GenerationResult(
            text=sampler.current_text(),
            is_complete=True,
            tokens_generated=0,
            stopped_reason="complete"
        )
        _log_result("inference.until_complete(initially complete)", result)
        return result, sampler
    
    tokens_generated = 0
    stopped_reason = "max_tokens"
    
    for step in range(max_tokens):
        try:
            next_token = _select_and_feed_token(
                sampler=sampler,
                greedy_k=greedy_k,
                pre_top_k=pre_top_k,
                use_feed_only=use_feed_only,
            )
        except Exception as e:
            stopped_reason = f"type_error: {e}"
            _dbg("inference.until_complete step {}: selection error={}", step, e)
            break

        if next_token is None:
            stopped_reason = "no_valid"
            _dbg("inference.until_complete step {}: no valid token", step)
            break

        tokens_generated += 1
        _dbg(
            "inference.until_complete step {}: token={!r} text={!r}",
            step,
            next_token,
            sampler.current_text(),
        )

        if on_token:
            on_token(next_token, step)

        if sampler.is_complete():
            stopped_reason = "complete"
            _dbg("inference.until_complete step {}: sampler is complete", step)
            break
    
    result = GenerationResult(
        text=sampler.current_text(),
        is_complete=sampler.is_complete(),
        tokens_generated=tokens_generated,
        stopped_reason=stopped_reason
    )
    _log_result("inference.until_complete", result)
    return result, sampler
