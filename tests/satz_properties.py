"""Property checks for generic scheme composition and policy."""

from __future__ import annotations

from dataclasses import dataclass

import aufbau
from hypothesis import given, settings, strategies as st

from proposition7.scheme import Effect, Param, Primitive, Scheme, compose

from tests.scheme import BINDING


TYPE_SOURCES = st.sampled_from(("Text", "PathSet", "IoError"))
EFFECT_KINDS = st.sampled_from(("read", "write", "execute", "network", "delete"))
IDENTIFIER = st.from_regex(r"[a-z][a-z0-9]{0,5}", fullmatch=True)


@st.composite
def schemes(draw: st.DrawFn) -> Scheme:
    """Generate declarations whose source types the toy grammar accepts."""
    names = draw(st.lists(IDENTIFIER, min_size=0, max_size=5, unique=True))
    primitives = []
    for name in names:
        param_names = draw(st.lists(IDENTIFIER, min_size=0, max_size=2, unique=True))
        params = tuple(
            Param(param_name, draw(TYPE_SOURCES)) for param_name in param_names
        )
        effects = tuple(
            Effect(kind, "**")
            for kind in draw(st.lists(EFFECT_KINDS, min_size=0, max_size=2, unique=True))
        )
        primitives.append(
            Primitive(
                name,
                params,
                returns=draw(TYPE_SOURCES),
                raises=draw(st.one_of(st.none(), st.just("IoError"))),
                effects=effects,
            )
        )
    return Scheme(tuple(primitives))


@settings(max_examples=50, deadline=None)
@given(scheme=schemes())
def test_composition_is_byte_deterministic(scheme: Scheme) -> None:
    first = compose(scheme, BINDING)
    assert first == compose(scheme, BINDING)
    assert first == compose(Scheme(tuple(reversed(scheme.primitives))), BINDING)


@settings(max_examples=50, deadline=None)
@given(scheme=schemes())
def test_every_grantable_scheme_yields_an_inhabited_grammar(scheme: Scheme) -> None:
    spg = aufbau.SPG(compose(scheme, BINDING))
    assert spg.completeness() == ("inhabited", [])


@settings(max_examples=50, deadline=None)
@given(data=st.data())
def test_the_scheme_is_the_policy(data: st.DataObject) -> None:
    scheme = data.draw(schemes().filter(lambda value: bool(value.primitives)))
    primitive = data.draw(st.sampled_from(scheme.primitives))
    restricted = scheme.restrict(
        p.name for p in scheme.primitives if p.name != primitive.name
    )
    spg = aufbau.SPG(compose(restricted, BINDING))

    # Ask aufbau about the actual call prefix, rather than reimplementing a
    # source-level search for the generated nonterminal.
    prefix = f'x = {primitive.name}('
    assert aufbau.Synthesizer.from_grammar(spg, prefix).status() == "dead"


@dataclass
class Value:
    payload: object


PYTHON_VALUES = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.text(), st.builds(Value, st.integers())),
    lambda children: st.one_of(st.lists(children, max_size=3), st.dictionaries(st.text(), children, max_size=3)),
    max_leaves=8,
)
