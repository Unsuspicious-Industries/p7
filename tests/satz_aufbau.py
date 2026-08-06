"""Composition checked against the real engine, not just as string assembly.

Skipped when aufbau is not importable, so the suite still runs in a bare
environment. Run it with the engine on the path before trusting any change to
`lang7.grammar`:

    PYTHONPATH=src:<venv>/lib/python3.12/site-packages pytest tests/
"""

from __future__ import annotations

import pytest

from proposition7.satz import Scheme, compose
from tests.satz_wiring import BINDING, CORE, SCHEME

aufbau = pytest.importorskip("aufbau")


def core_only() -> str:
    return CORE.replace(" | @PRIMITIVES@", "")


def test_composed_grammar_loads():
    spg = aufbau.SPG(compose(SCHEME, BINDING))
    assert spg.start


def test_composition_does_not_change_the_start_symbol():
    """A regression test for a bug this code actually had.

    aufbau takes the last-declared nonterminal as the start symbol. Appending
    the primitive fragment therefore made `PrimitiveCall` the root, so the
    grammar accepted a bare primitive call as an entire program — the mask
    would have permitted `glob("x")` where a full `StatementList` was required.
    Composition must leave the core's root intact.
    """
    assert aufbau.SPG(compose(SCHEME, BINDING)).start == aufbau.SPG(core_only()).start


def test_empty_scheme_still_loads():
    assert aufbau.SPG(compose(Scheme(), BINDING)).start


def test_primitive_calls_are_masked_by_the_scheme():
    """ARCHITECTURE I7: a primitive absent from the scheme is unemittable."""
    spg = aufbau.SPG(compose(SCHEME, BINDING))
    synth = aufbau.Synthesizer.from_grammar(spg, "")

    for token in ["files", "=", "list_dir"]:
        assert synth.mask([token])[0], f"{token!r} should be admissible"
        synth.feed(token)

    # A primitive that is not in the scheme cannot even begin.
    fresh = aufbau.Synthesizer.from_grammar(spg, "")
    for token in ["x", "="]:
        fresh.feed(token)
    assert not fresh.mask(["rm_rf"])[0], "ungranted primitive must be masked out"


def test_completeness_reports_uninhabited_sorts():
    """ARCHITECTURE I5 / §2.2 — the CI assertion D3 has to satisfy.

    The toy core language is *not* expected to pass: it has no universal
    inhabitant, which is exactly the gap D3 must close. This test pins the
    engine's answer so the check is wired up and ready, and documents what a
    failing language looks like.
    """
    kind, sorts = aufbau.SPG(compose(SCHEME, BINDING)).completeness()
    assert kind in {"syntactic", "inhabited", "sound"}
    if kind == "sound" and sorts:
        pytest.xfail(
            f"toy core language has possibly-uninhabited sorts {sorts}; "
            "D3 must add a universal inhabitant (ARCHITECTURE §2.2)"
        )
