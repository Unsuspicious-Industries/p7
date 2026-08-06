"""The turn loop — where all four components meet.

One turn, end to end:

    1. gamma   composes  core language + scheme  ->  constraint_scheme_auf
    2. gamma   renders   prompt (mutable-last)   ->  model_context
    3. gamma   projects  Γ                       ->  aufbau_context  (types only)
    4. provider7  decodes under the mask         ->  completion, Γ′-types
    5. gamma   lowers    completion              ->  ir.Program      (D3 supplies)
    6. gamma   audits    Program                 ->  EffectAudit, approval
    7. gamma   evaluates Program                 ->  values
    8. gamma   commits   Γ′                      ->  Γ for the next turn

Steps 1-3 and 5-8 are the client's. Step 4 is the only thing that crosses the
wire, and it carries no values and no session identity — provider7 holds
nothing between requests (ARCHITECTURE §3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .context import Gamma
from .evaluator import Dispatch, EffectAudit, Evaluator, Host, TurnOutcome
from .grammar import LanguageBinding, compose
from .ir import Program
from .scheme import Scheme, TypeSource


# ── The provider7 core API, from the client side ─────────────────────────


@dataclass(frozen=True)
class CoreRequest:
    """A constrained-generation request. Carries everything; assumes nothing."""

    model: str
    model_context: str
    aufbau_context: Mapping[str, TypeSource]
    constraint_scheme_auf: str
    max_tokens: int = 512
    temperature: float = 0.0
    seed: int | None = None


@dataclass(frozen=True)
class CoreResponse:
    completion: str
    updated_aufbau_context: Mapping[str, TypeSource]
    #: True when the decode reached a complete, well-typed program.
    is_complete: bool = True
    stopped_reason: str = ""


class ConstraintClient(Protocol):
    """provider7's core surface. The only thing that crosses the wire."""

    def generate(self, request: CoreRequest) -> CoreResponse: ...


# ── Prompt assembly ──────────────────────────────────────────────────────


@dataclass
class PromptLayout:
    """Assembles `model_context` with the cache-stable parts first.

    Order is load-bearing, not cosmetic (ARCHITECTURE §3.1). Server-side prefix
    caching hits on the longest stable prefix, and Γ changes every single turn,
    so Γ must render *last*. Putting it near the top — the natural place, since
    it reads like a declaration block — invalidates the entire KV cache on
    every request.
    """

    system: str = ""
    task: str = ""
    history: list[str] = field(default_factory=list)

    def render(self, scheme: Scheme, gamma: Gamma, instruction: str = "") -> str:
        parts: list[str] = []

        # ── stable across the whole session ──
        if self.system:
            parts.append(self.system)
        if scheme.primitives:
            signatures = "\n".join(
                f"  {p.signature()}" for p in scheme.ordered()
            )
            parts.append(f"Primitives:\n{signatures}")
        if self.task:
            parts.append(f"Task:\n{self.task}")

        # ── append-only ──
        parts.extend(self.history)

        # ── mutable: everything below changes every turn ──
        parts.append(f"Context:\n{gamma.render()}")
        if instruction:
            parts.append(instruction)

        return "\n\n".join(parts)


# ── Session ──────────────────────────────────────────────────────────────

#: D3 supplies this: completion text -> IR. See `proposition7.satz.ir`.
Lowering = Callable[[str], Program]


class Session:
    """A coding session. Owns Γ; the server owns nothing."""

    def __init__(
        self,
        client: ConstraintClient,
        binding: LanguageBinding,
        dispatch: Dispatch,
        lower: Lowering,
        *,
        model: str,
        layout: PromptLayout | None = None,
        approve: Callable[[EffectAudit], bool] | None = None,
    ):
        self.client = client
        self.binding = binding
        self.dispatch = dispatch
        self.lower = lower
        self.model = model
        self.layout = layout or PromptLayout()
        self.evaluator = Evaluator(dispatch, approve=approve)
        self.gamma = Gamma()
        self.turn_index = 0

        # Composed once per scheme, not per turn: the SPG compile cache is
        # keyed by content hash, so a stable string here means the grammar
        # compiles once for the whole session (ARCHITECTURE §3.1).
        self.grammar_source = compose(dispatch.scheme, binding)

    @property
    def scheme(self) -> Scheme:
        return self.dispatch.scheme

    def turn(self, instruction: str = "", **generation: Any) -> "TurnRecord":
        """Run one turn: generate a program, evaluate it, commit Γ′."""
        model_context = self.layout.render(self.scheme, self.gamma, instruction)

        response = self.client.generate(
            CoreRequest(
                model=self.model,
                model_context=model_context,
                aufbau_context=self.gamma.types(),
                constraint_scheme_auf=self.grammar_source,
                **generation,
            )
        )

        record = TurnRecord(
            turn=self.turn_index,
            completion=response.completion,
            is_complete=response.is_complete,
        )

        if not response.is_complete:
            # The decode did not reach a well-typed program. Nothing ran, Γ is
            # untouched — the same shape as a `todo` abort.
            record.outcome = TurnOutcome(
                aborted=True, reason=f"incomplete:{response.stopped_reason}"
            )
            self.turn_index += 1
            return record

        program = self.lower(response.completion)
        record.program = program

        outcome = self.evaluator.run(program, self.gamma, turn=self.turn_index)
        record.outcome = outcome

        if outcome.ok:
            self.gamma.commit(outcome.updates, turn=self.turn_index)

        self.layout.history.append(f"> {response.completion}")
        self.turn_index += 1
        return record


@dataclass
class TurnRecord:
    """What happened in one turn. The unit of display and of audit."""

    turn: int
    completion: str = ""
    is_complete: bool = True
    program: Program | None = None
    outcome: TurnOutcome | None = None

    @property
    def ok(self) -> bool:
        return self.outcome is not None and self.outcome.ok


# ── Construction ─────────────────────────────────────────────────────────


def build(
    client: ConstraintClient,
    binding: LanguageBinding,
    scheme: Scheme,
    hosts: dict[str, Host],
    lower: Lowering,
    *,
    model: str,
    layout: PromptLayout | None = None,
    approve: Callable[[EffectAudit], bool] | None = None,
) -> Session:
    """Build a session, wiring the grammar and the evaluator from one binding.

    Prefer this over constructing `Dispatch` and `Session` separately: it is
    what guarantees `Dispatch.result_type` matches the template the grammar was
    generated with, so the type the evaluator reports for a fallible call is
    the type the mask constrained it to.
    """
    dispatch = Dispatch(scheme=scheme, hosts=hosts, result_type=binding.result_type)
    return Session(
        client,
        binding,
        dispatch,
        lower,
        model=model,
        layout=layout,
        approve=approve,
    )
