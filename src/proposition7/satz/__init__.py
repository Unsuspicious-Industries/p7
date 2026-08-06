"""The agent language: primitive scheme, grammar composition, typed evaluator.

The client-side half of congen — what `../../../../ARCHITECTURE.md` calls
`[LANG]`. It lives here rather than in gamma because `benchmarks/agent.py`
needs it: agent episodes are graded on *executed* value, so the paper's
artifact needs both the constraint and the evaluator, and an open reproducible
artifact cannot depend on a closed product.

This does not violate p7's charter. That charter constrains
`proposition7.backends.v1` — the minimal cross-repository surface provider7
imports — and provider7 never imports this package: it receives already-composed
`.auf` source and only compiles and masks (ARCHITECTURE §5).

Nor does it put execution on the inference host. `Dispatch` takes its host
implementations as an argument; this package ships a dispatcher with no
capabilities of its own. The functions that actually touch a filesystem or a
shell are gamma's, injected at construction.

## The current API

    Scheme ──> grammar.compose()   ──> .auf source  ──> provider7 masks it
           └─> evaluator.Dispatch  ──> host calls   ──> the client runs it

One primitive table, two projections, so they cannot drift (ARCHITECTURE I2).

- `scheme`     `Primitive`/`Param`/`Effect`/`Scheme` + the JSON wire format
- `grammar`    scheme → `.auf` fragment, composed with a core language
- `gamma`      `Gamma`: client-side (name → value), wire view (name → type)
- `ir`         the IR the evaluator runs, so D3 supplies only a lowering
- `result`     `Result`/`Ok`/`Err` — failure is a value, never an exit
- `evaluator`  `Dispatch`, effect audit, atomic turns
- `turn`       the turn loop and `PromptLayout`

The core language itself is not here: that is D3, supplied as `.auf` source to
`LanguageBinding.core_source`."""


# ── Current API ──────────────────────────────────────────────────────────

from .evaluator import (
    Dispatch,
    DispatchError,
    EffectAudit,
    EvaluationError,
    Evaluator,
    TurnOutcome,
    audit,
)
from .context import Binding, Gamma
from .grammar import (
    PRIMITIVE_MARKER,
    PRIMITIVE_NT,
    CompositionError,
    LanguageBinding,
    compose,
    fragment,
    validate_binding,
)
from .ir import Bind, Call, Do, Lit, Program, Todo, Var
from .result import Err, Ok, PrimitiveFailure, Result
from .scheme import SCHEME_VERSION, Effect, Param, Primitive, Scheme, TypeSource
from .session import (
    ConstraintClient,
    CoreRequest,
    CoreResponse,
    PromptLayout,
    Session,
    TurnRecord,
    build,
)

__all__ = [
    "SCHEME_VERSION",
    "PRIMITIVE_MARKER",
    "PRIMITIVE_NT",
    "Scheme",
    "Primitive",
    "Param",
    "Effect",
    "TypeSource",
    "LanguageBinding",
    "compose",
    "fragment",
    "validate_binding",
    "CompositionError",
    "Gamma",
    "Binding",
    "Program",
    "Bind",
    "Do",
    "Call",
    "Var",
    "Lit",
    "Todo",
    "Result",
    "Ok",
    "Err",
    "PrimitiveFailure",
    "Dispatch",
    "Evaluator",
    "EffectAudit",
    "TurnOutcome",
    "audit",
    "DispatchError",
    "EvaluationError",
    "Session",
    "TurnRecord",
    "CoreRequest",
    "CoreResponse",
    "ConstraintClient",
    "PromptLayout",
    "build",
]
