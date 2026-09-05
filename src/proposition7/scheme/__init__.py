"""Generic primitive declaration and grammar composition."""

from .compose import CompositionError, LanguageBinding, compose, nonterminal, validate_binding
from .declaration import Effect, Param, Primitive, Scheme, TypeSource

__all__ = [
    "Primitive", "Param", "Effect", "Scheme", "TypeSource", "LanguageBinding", "compose",
    "validate_binding", "nonterminal", "CompositionError",
]
