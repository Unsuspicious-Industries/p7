# p7: Type-aware constrained decoding for LLMs
#
# Unlike CFG-only approaches, P7 supports context-dependent grammars
# with typing rules - enabling generation of well-typed code.

from __future__ import annotations

from p7.p7 import (
    Grammar,
    Synthesizer,
    regex_matches,
    regex_prefix_valid,
)

from .sampler import CompletionEngine, TypedSampler, set_debug
from .grammars import GRAMMARS, list_grammars, get_grammar, get_grammar_info
from .inference import generate, until_complete, GenerationResult
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
    GrammarBlock,
    Mode,
    build_system_prompt,
)

__all__ = [
    # rust bindings
    "Grammar",
    "Synthesizer",
    "regex_matches",
    "regex_prefix_valid",
    # python stuff
    "CompletionEngine",
    "TypedSampler",
    "set_debug",
    "generate",
    "until_complete",
    "GenerationResult",
    "ConstrainedModel",
    # model adapters
    "get_model_class",
    "ChatConstrainedModel",
    "DeepseekConstrainedModel",
    "LlamaConstrainedModel",
    "MistralConstrainedModel",
    "GlmConstrainedModel",
    "PleiasConstrainedModel",
    # environment
    "ReasoningEnvironment",
    "EnvironmentResult",
    "ThinkBlock",
    "GrammarBlock",
    "Mode",
    "build_system_prompt",
    "get_grammar_info",
    # grammars
    "GRAMMARS",
    "list_grammars",
    "get_grammar",
]

__version__: str = "0.1.0"
