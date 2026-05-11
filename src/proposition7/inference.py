"""Shared generation result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    text: str
    is_complete: bool
    tokens_generated: int
    stopped_reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Per-step arrays (parallel, length == tokens_generated).
    # Populated by constrained generation; empty for unconstrained.
    step_token_ids: list[int] = field(default_factory=list)
    # Pre-mask entropy in bits: model's raw uncertainty before grammar filtering.
    # H = -∑ p_i·log₂(p_i) over all finite-logit tokens (no grammar mask applied).
    step_pre_entropies: list[float] = field(default_factory=list)
    # Post-mask entropy in bits: uncertainty over grammar-valid tokens only.
    # Same formula but restricted to tokens that pass grammar validation.
    # High value → many competing valid continuations (good future branch point).
    # Difference (pre − post) measures how much the grammar mask shapes the distribution.
    step_entropies: list[float] = field(default_factory=list)
    # Grammar-rejected candidates before the accepted token at each step.
    step_retries: list[int] = field(default_factory=list)
