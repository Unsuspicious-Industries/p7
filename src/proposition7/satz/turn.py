"""One turn, end to end. The session half of the loop.

    compose scheme+core -> constraint_scheme_auf ─┐
     Γ.types()           -> aufbau_context         ├─> generation callable
    prompt pairs        -> model_context          ─┘
                                                   │
    evaluate the parse tree, commit Γ′  <──────────┘

Session identity, history, Γ and approval all live here. The generation callable
holds nothing between calls, which is what lets a remote one be substituted for
the local default without this file knowing the difference.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .context import Gamma
from .evaluator import Audit, Dispatch, EvaluationError, Evaluator, Outcome
from .grammar import LanguageBinding, compose
from .scheme import Scheme, TypeSource


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
        scheme: Scheme,
        hosts: Mapping[str, Callable[..., Any]],
        binding: LanguageBinding,
        *,
        model: str,
        system: str = "",
        task: str = "",
        approve: Callable[[Audit], bool] | None = None,
        generation: Callable[..., Any] | None = None,
    ):
        if generation is None:
            from proposition7.api import generate

            generation = generate
        self.generation = generation
        self.scheme = scheme
        self.model = model
        self.system = system
        self.task = task
        self.history: list[str] = []
        self.gamma = Gamma()
        # Composed once: the SPG cache is keyed by content hash, so a stable
        # string compiles the grammar once for the whole session.
        self.grammar_source = compose(scheme, binding)
        import aufbau

        self.evaluator = Evaluator(
            Dispatch(scheme, dict(hosts)), binding, aufbau.SPG(self.grammar_source), approve=approve
        )

    def turn(self, instruction: str = "", **generation: Any) -> Outcome:
        result = self.generation(
            prompt(
                self.scheme,
                self.gamma,
                system=self.system,
                task=self.task,
                history=tuple(self.history),
                instruction=instruction,
            ),
            model=self.model,
            grammar=self.grammar_source,
            aufbau_context=self.gamma.types(),
            **generation,
        )
        if not result.complete:
            return Outcome(aborted=True, reason=f"incomplete:{result.reason}")

        outcome = self.evaluator.run(result.text, self.gamma)
        if outcome.ok:
            self.gamma.commit(self._reconcile(outcome.updates, result.exported_context))
        self.history.append(result.text)
        return outcome

    def _reconcile(
        self,
        updates: Mapping[str, tuple[TypeSource, Any]],
        exported: Mapping[str, TypeSource],
    ) -> dict[str, tuple[TypeSource, Any]]:
        """Types come from the decode, values from here (ARCHITECTURE I1a).

        The evaluator re-parses the completion and derives the same types
        independently, which makes it a cross-check rather than a second
        opinion: where both sides speak they must agree, and a clash means the
        typing rules and the evaluator have drifted. Agreement is checked by
        unification, not string equality — the engine renders types in a
        normal form (`Result[A, B]` comes back as `Result [ A , B ]`), so the
        spellings differ while the types do not.

        Names the decode did not export keep their locally derived type. That
        is the degraded path, not the design: it covers a grammar whose rules
        export nothing, and it is why this merges rather than replaces.
        """
        merged: dict[str, tuple[TypeSource, Any]] = {}
        for name, (local, value) in updates.items():
            served = exported.get(name)
            if served is None:
                merged[name] = (local, value)
                continue
            if self.evaluator.spg.unify(local, served) is None:
                raise EvaluationError(
                    f"{name}: evaluator typed it {local!r}, the decode exported "
                    f"{served!r}; the typing rules and the evaluator disagree"
                )
            merged[name] = (served, value)
        return merged
