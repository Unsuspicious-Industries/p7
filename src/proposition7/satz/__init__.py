"""The agent language: primitive scheme, grammar composition, typed evaluator.

`[LANG]` in ../../../../ARCHITECTURE.md. One primitive table, two projections
that cannot drift (I2):

    Scheme -> grammar.compose()  -> .auf source  -> the mask
           -> evaluator.Dispatch -> host calls   -> execution, here

Here rather than in gamma because `benchmarks/agent.py` grades episodes on
executed value: the paper's artifact needs the evaluator as well as the
constraint, and an open artifact cannot depend on a closed product. It ships no
capabilities: `Dispatch` takes hosts as an argument.

`evaluator` is an execution enclave; read its module docstring before using it.

The core language is not here. That is D3: a `.auf` file meeting the contract in
`grammar`, supplied as `LanguageBinding.core_source`.
"""

from .evaluator import Audit, Outcome
from .grammar import LanguageBinding
from .result import PrimitiveFailure
from .scheme import Effect, Param, Primitive, Scheme
from .turn import Session

__all__ = [
    "Session", "Scheme", "Primitive", "Param", "Effect", "Audit", "Outcome",
    "PrimitiveFailure", "LanguageBinding",
]
