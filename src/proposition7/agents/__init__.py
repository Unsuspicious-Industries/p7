"""Reusable multi-turn, grammar-constrained tool-calling agent library.

Not benchmark-only: `ToolRegistry` + `AgentSession` are the mechanism, usable
from a benchmark harness (`benchmarks/agent.py`), an interactive CLI
(`proposition7.agents.cli`), or a TUI (same `AgentSession.step()` interface,
called from whatever event loop the frontend uses).
"""

from __future__ import annotations

from .registry import Tool, ToolRegistry
from .session import AgentResult, AgentSession, AgentTurn

__all__ = [
    "Tool",
    "ToolRegistry",
    "AgentSession",
    "AgentTurn",
    "AgentResult",
]
