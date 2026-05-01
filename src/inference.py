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
