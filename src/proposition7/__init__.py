from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "Synthesizer": ("aufbau", "Synthesizer"),
    "ConstrainedModel": (".llm", "ConstrainedModel"),
    "get_model_class": (".models", "get_model_class"),
    "ChatConstrainedModel": (".models", "ChatConstrainedModel"),
    "DeepseekConstrainedModel": (".models", "DeepseekConstrainedModel"),
    "LlamaConstrainedModel": (".models", "LlamaConstrainedModel"),
    "MistralConstrainedModel": (".models", "MistralConstrainedModel"),
    "GlmConstrainedModel": (".models", "GlmConstrainedModel"),
    "PleiasConstrainedModel": (".models", "PleiasConstrainedModel"),
    "generate": (".api", "generate"),
    "generate_unconstrained": (".api", "generate_unconstrained"),
    "generate_mixed": (".api", "generate_mixed"),
    "generate_pair": (".api", "generate_pair"),
    "verify": (".api", "verify"),
    "Result": (".api", "Result"),
    "GenerationResult": (".inference", "GenerationResult"),
    "GRAMMARS": ("grammars", "GRAMMARS"),
    "list_grammars": ("grammars", "list_grammars"),
    "get_grammar": ("grammars", "get_grammar"),
    "get_grammar_info": ("grammars", "get_grammar_info"),
    "get_grammar_summary": ("grammars", "get_grammar_summary"),
    "base_grammar": ("grammars", "base_grammar"),
    "strip_typing_rules": ("grammars", "strip_typing_rules"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = import_module(module_name, __name__) if module_name.startswith(".") else import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "generate", "generate_unconstrained", "generate_mixed", "generate_pair", "verify", "Result",
    "Synthesizer", "GenerationResult", "ConstrainedModel", "get_model_class",
    "ChatConstrainedModel", "DeepseekConstrainedModel", "LlamaConstrainedModel",
    "MistralConstrainedModel", "GlmConstrainedModel", "PleiasConstrainedModel",
    "GRAMMARS", "list_grammars", "get_grammar", "get_grammar_info",
    "get_grammar_summary", "base_grammar", "strip_typing_rules",
]

__version__: str = "0.1.0"
