"""One turn, end to end. The client half of the loop.

    compose scheme+core -> constraint_scheme_auf ─┐
    Γ.types()           -> aufbau_context         ├─> provider7 (masks, decodes)
    prompt pairs        -> model_context          ─┘
                                                   │
    evaluate the parse tree, commit Γ′  <──────────┘

provider7 holds nothing between calls; session identity, history, and approval
are all here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .context import Gamma
from .evaluator import Audit, Dispatch, Evaluator, Outcome
from .grammar import LanguageBinding, compose
from .scheme import Scheme, TypeSource


@dataclass(frozen=True)
class Request:
    model: str
    #: Ordered (kind, content) pairs. The server owns chat templating; the
    #: client owns ordering. Γ renders last — it changes every turn, and
    #: anything before it stays in the cacheable prefix (ARCHITECTURE §3.1).
    model_context: tuple[tuple[str, str], ...] = ()
    aufbau_context: Mapping[str, TypeSource] = field(default_factory=dict)
    constraint_scheme_auf: str = ""
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class Response:
    completion: str
    updated_aufbau_context: Mapping[str, TypeSource] = field(default_factory=dict)
    is_complete: bool = True
    stopped_reason: str = ""
    retries: int = 0


class Client(Protocol):
    """provider7's core surface: the only thing that crosses the wire."""

    def generate(self, request: Request) -> Response: ...


def prompt(
    scheme: Scheme, gamma: Gamma, *, system: str = "", task: str = "",
    history: tuple[str, ...] = (), instruction: str = "",
) -> tuple[tuple[str, str], ...]:
    """Ordered (kind, content) pairs, stable prefix first, Γ last."""
    parts: list[tuple[str, str]] = []
    if system:
        parts.append(("system", system))
    if scheme.primitives:
        parts.append(
            ("system", "Primitives:\n" + "\n".join(f"  {p.signature()}" for p in scheme.ordered()))
        )
    if task:
        parts.append(("user", f"Task:\n{task}"))
    parts.extend(("assistant", h) for h in history)
    parts.append(("user", f"Context:\n{gamma.render()}"))
    if instruction:
        parts.append(("user", instruction))
    return tuple(parts)


class Session:
    """Owns Γ and the composed grammar for a fixed scheme."""

    def __init__(
        self,
        client: Client,
        binding: LanguageBinding,
        dispatch: Dispatch,
        spg: Any,
        *,
        model: str,
        system: str = "",
        task: str = "",
        approve: Callable[[Audit], bool] | None = None,
    ):
        self.client = client
        self.scheme = dispatch.scheme
        self.model = model
        self.system = system
        self.task = task
        self.history: list[str] = []
        self.gamma = Gamma()
        self.evaluator = Evaluator(dispatch, binding, spg, approve=approve)
        # Composed once: the SPG cache is keyed by content hash, so a stable
        # string compiles the grammar once for the whole session.
        self.grammar_source = compose(dispatch.scheme, binding)

    def turn(self, instruction: str = "", **generation: Any) -> Outcome:
        response = self.client.generate(
            Request(
                model=self.model,
                model_context=prompt(
                    self.scheme,
                    self.gamma,
                    system=self.system,
                    task=self.task,
                    history=tuple(self.history),
                    instruction=instruction,
                ),
                aufbau_context=self.gamma.types(),
                constraint_scheme_auf=self.grammar_source,
                **generation,
            )
        )
        if not response.is_complete:
            return Outcome(aborted=True, reason=f"incomplete:{response.stopped_reason}")

        outcome = self.evaluator.run(response.completion, self.gamma)
        if outcome.ok:
            self.gamma.commit(outcome.updates)
        self.history.append(response.completion)
        return outcome
