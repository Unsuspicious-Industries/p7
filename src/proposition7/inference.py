"""Shared generation result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class GenerationResult:
    text: str
    is_complete: bool
    tokens_generated: int
    stopped_reason: str
    exported_context: Mapping[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Per-step arrays (parallel, length == tokens_generated).
    # Populated by constrained generation; empty for unconstrained.
    step_token_ids: list[int] = field(default_factory=list)
    # Pre-mask entropy in bits: model's raw uncertainty before grammar filtering.
    # H = -∑ p_i·log₂(p_i) over all finite-logit tokens (no grammar mask applied).
    step_pre_entropies: list[float] = field(default_factory=list)
    # Residual entropy in bits: the distribution left once the candidates the
    # grammar actually refused at this step have been removed.
    #
    # Read the name carefully, because the obvious reading is wrong. This is
    # NOT entropy over the grammar-valid tokens: establishing that set means
    # asking the engine about every token in the vocabulary, ~150k calls at
    # milliseconds each, which is minutes per token. Only the candidates the
    # retry loop actually tried are known to be invalid, so those are what is
    # excluded. The support is therefore an over-approximation of the valid
    # set, and this number an upper bound on true post-mask entropy.
    #
    # It is still the interesting quantity, and it is frequently HIGHER than
    # `step_pre_entropies` rather than lower, which surprises people until
    # they see why. A peaked model with its top choice vetoed goes from near
    # certainty to a flat field of alternatives: measured here, 0.01 bits
    # before, 10.14 bits after. That is the grammar refusing the thing the
    # model was sure about, which is exactly the story worth telling; it is
    # just not "the mask reduced uncertainty".
    step_entropies: list[float] = field(default_factory=list)
    # Grammar-rejected candidates before the accepted token at each step.
    step_retries: list[int] = field(default_factory=list)

    step_trace: list[dict[str, Any]] = field(default_factory=list)
    """Per-step reconstruction of the decode, empty unless tracing was asked.

    One entry per accepted position, each carrying the token and spelling that
    was accepted and the full list of candidates this position refused with the
    reason for each. `step_retries` says a position cost 136 attempts; this says
    which 136 and why every one of them was turned down.
    """
