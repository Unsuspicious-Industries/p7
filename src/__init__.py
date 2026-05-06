from __future__ import annotations

from aufbau import Synthesizer

from .grammars import (
    GRAMMARS,
    get_grammar,
    get_grammar_info,
    get_grammar_summary,
    list_grammars,
)
from .inference import GenerationResult
from .llm import ConstrainedModel
from .models import (
    get_model_class,
    PleiasConstrainedModel,
    DeepseekConstrainedModel,
    LlamaConstrainedModel,
    MistralConstrainedModel,
    GlmConstrainedModel,
    ChatConstrainedModel,
)
from .environment import (
    ReasoningEnvironment,
    EnvironmentResult,
    ThinkBlock,
    FormalBlock,
    GrammarBlock,
    Mode,
    build_system_prompt,
)

# High-level API
from .api import Session, generate, Result

__all__ = [
    "generate",
    "Session",
    "Result",
    "Synthesizer",
    "GenerationResult",
    "ConstrainedModel",
    "get_model_class",
    "ChatConstrainedModel",
    "DeepseekConstrainedModel",
    "LlamaConstrainedModel",
    "MistralConstrainedModel",
    "GlmConstrainedModel",
    "PleiasConstrainedModel",
    "ReasoningEnvironment",
    "EnvironmentResult",
    "ThinkBlock",
    "FormalBlock",
    "GrammarBlock",
    "Mode",
    "build_system_prompt",
    "GRAMMARS",
    "list_grammars",
    "get_grammar",
    "get_grammar_info",
    "get_grammar_summary",
]

__version__: str = "0.1.0"
