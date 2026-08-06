"""The agent language: primitive scheme, grammar composition, typed evaluator.

`[LANG]` in ../../../../ARCHITECTURE.md. One primitive table, two projections
that cannot drift (I2):

    Scheme -> grammar.compose()  -> .auf source  -> provider7 masks it
           -> evaluator.Dispatch -> host calls   -> the client runs it

Here rather than in gamma because `benchmarks/agent.py` grades episodes on
executed value: the paper's artifact needs the evaluator as well as the
constraint, and an open artifact cannot depend on a closed product. provider7
never imports this — it receives composed `.auf` — so `backends.v1` stays
minimal. It ships no capabilities: `Dispatch` takes hosts as an argument.

`evaluator` is an execution enclave; read its module docstring before using it.

The core language is not here. That is D3: a `.auf` file meeting the contract in
`grammar`, supplied as `LanguageBinding.core_source`.
"""

from .context import Gamma
from .evaluator import Audit, Dispatch, DispatchError, EvaluationError, Evaluator, Outcome
from .grammar import (
    PRIMITIVE_MARKER,
    PRIMITIVE_NT,
    CompositionError,
    LanguageBinding,
    compose,
    fragment,
    nonterminal,
    validate_binding,
)
from .result import Err, Ok, PrimitiveFailure, Result
from .scheme import SCHEME_VERSION, Effect, Param, Primitive, Scheme, TypeSource
from .turn import Client, Request, Response, Session, prompt

__all__ = [
    "SCHEME_VERSION", "PRIMITIVE_MARKER", "PRIMITIVE_NT",
    "Scheme", "Primitive", "Param", "Effect", "TypeSource",
    "LanguageBinding", "compose", "fragment", "nonterminal", "validate_binding",
    "CompositionError",
    "Gamma",
    "Result", "Ok", "Err", "PrimitiveFailure",
    "Dispatch", "Evaluator", "Audit", "Outcome", "DispatchError", "EvaluationError",
    "Session", "Request", "Response", "Client", "prompt",
]
