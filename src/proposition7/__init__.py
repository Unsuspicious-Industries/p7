from __future__ import annotations

from importlib import import_module

from aufbau import Synthesizer

from grammars import (
    GRAMMARS,
    base_grammar,
    get_grammar,
    get_grammar_info,
    get_grammar_summary,
    list_grammars,
    strip_typing_rules,
)
from .inference import GenerationResult
from .environment import (
    ReasoningEnvironment,
    EnvironmentResult,
    ThinkBlock,
    FormalBlock,
    GrammarBlock,
    Mode,
    BENCHMARK_MODES,
    build_system_prompt,
    build_task_prompt,
)


_LAZY_EXPORTS = {
    "ConstrainedModel": (".llm", "ConstrainedModel"),
    "get_model_class": (".models", "get_model_class"),
    "ChatConstrainedModel": (".models", "ChatConstrainedModel"),
    "DeepseekConstrainedModel": (".models", "DeepseekConstrainedModel"),
    "LlamaConstrainedModel": (".models", "LlamaConstrainedModel"),
    "MistralConstrainedModel": (".models", "MistralConstrainedModel"),
    "GlmConstrainedModel": (".models", "GlmConstrainedModel"),
    "PleiasConstrainedModel": (".models", "PleiasConstrainedModel"),
    "Session": (".api", "Session"),
    "generate": (".api", "generate"),
    "Result": (".api", "Result"),
}


def __getattr__(name: str):
    """Load model-backed APIs only when a caller actually requests them."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

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
    "BENCHMARK_MODES",
    "build_system_prompt",
    "build_task_prompt",
    "GRAMMARS",
    "list_grammars",
    "get_grammar",
    "get_grammar_info",
    "get_grammar_summary",
    "base_grammar",
    "strip_typing_rules",
]

__version__: str = "0.1.0"
