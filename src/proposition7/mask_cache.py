from collections import OrderedDict
from typing import Any, Optional


DEFAULT_MASK_CACHE_SIZE = 4096
_cache: OrderedDict[tuple[bytes, tuple[tuple[str, str], ...], str, str], bool] = (
    OrderedDict()
)
_hits = 0
_misses = 0


def clear() -> None:
    """Clear the process-wide semantic-mask verdict cache and its counters."""
    global _hits, _misses
    _cache.clear()
    _hits = 0
    _misses = 0


def stats() -> dict[str, int]:
    """Return process-wide semantic-mask cache counters."""
    return {"hits": _hits, "misses": _misses, "size": len(_cache)}


def mask_candidates(
    synth: Any, grammar_hash: bytes, candidates: list[str], cache_size: int
) -> list[bool]:
    """Mask candidates, filling missing verdicts in one engine call."""
    global _hits, _misses
    if cache_size <= 0:
        return list(synth.mask(candidates))

    context = tuple(sorted((str(name), str(ty)) for name, ty in synth.context()))
    prefix = synth.input()
    keys = [(grammar_hash, context, prefix, candidate) for candidate in candidates]
    verdicts: list[Optional[bool]] = []
    missing: list[str] = []
    for key, candidate in zip(keys, candidates):
        verdict = _cache.get(key)
        if verdict is None:
            _misses += 1
            verdicts.append(None)
            missing.append(candidate)
        else:
            _hits += 1
            _cache.move_to_end(key)
            verdicts.append(verdict)

    if missing:
        fresh = iter(synth.mask(missing))
        for index, verdict in enumerate(verdicts):
            if verdict is not None:
                continue
            verdict = bool(next(fresh))
            verdicts[index] = verdict
            _cache[keys[index]] = verdict
            _cache.move_to_end(keys[index])
            while len(_cache) > cache_size:
                _cache.popitem(last=False)
    return [bool(verdict) for verdict in verdicts]
