import hashlib

import aufbau
import grammars

from proposition7.mask_cache import clear, mask_candidates, stats


def _synth(context=None):
    # The shared base library, not a copy beside the test: `aufbau.examples`
    # holds the one `stlc.auf`, so this cannot drift from what p7 decodes with.
    spec = grammars.get_grammar("stlc")
    synth = aufbau.Synthesizer.from_grammar(aufbau.SPG(spec), "")
    for name, ty in (context or {}).items():
        synth.add_to_ctx(name, ty)
    return synth, hashlib.blake2b(spec.encode(), digest_size=16).digest()


def test_context_is_part_of_the_mask_cache_key():
    clear()
    empty, grammar_hash = _synth()
    scoped, _ = _synth({"x": "Int"})

    assert mask_candidates(empty, grammar_hash, ["x"], 4) == [False]
    assert mask_candidates(scoped, grammar_hash, ["x"], 4) == [True]


def test_mask_cache_is_bounded_lru_and_caches_rejections():
    clear()
    synth, grammar_hash = _synth()
    assert mask_candidates(synth, grammar_hash, ["x"], 1) == [False]
    assert mask_candidates(synth, grammar_hash, ["x"], 1) == [False]
    assert stats() == {"hits": 1, "misses": 1, "size": 1}
    assert mask_candidates(synth, grammar_hash, ["y"], 1) == [False]
    assert stats()["size"] == 1
