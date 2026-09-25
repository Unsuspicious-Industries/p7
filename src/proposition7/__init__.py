"""proposition7: type-aware constrained decoding.

The bridge between logits and the aufbau engine, plus `scheme`, the generic
utility that turns a primitive table into `.auf` source. It ships no agent
language, no evaluator, no session and no capabilities -- it is a library for
building constrained-generation systems, not one of them.

Import style, which is the congen convention: every name below is reachable
from the package root and resolved lazily, because the transformers backend
drags in torch and a caller that supplies its own runtime (wirt does) must be
able to decode without torch installed at all. Nothing here imports a heavy
dependency at module scope.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version

_LAZY_EXPORTS = {
    # The shared vocabulary of the tree. See `types` and `errors`.
    "Ok": (".types", "Ok"),
    "Err": (".types", "Err"),
    "Result": (".types", "Result"),
    "is_ok": (".types", "is_ok"),
    "is_err": (".types", "is_err"),
    "CongenError": (".errors", "CongenError"),
    "GrammarError": (".errors", "GrammarError"),
    "DecodeError": (".errors", "DecodeError"),
    "RuntimeAdapterError": (".errors", "RuntimeAdapterError"),
    # The public generation surface.
    "generate": (".api", "generate"),
    "generate_unconstrained": (".api", "generate_unconstrained"),
    "generate_mixed": (".api", "generate_mixed"),
    "generate_pair": (".api", "generate_pair"),
    "verify": (".api", "verify"),
    "Generation": (".api", "Generation"),
    "GenerationResult": (".inference", "GenerationResult"),
    # The decode loop, for a caller serving its own batches. These were reached
    # for through the submodule (and `_draw` through its private name) before
    # they were published here.
    "draw": (".decode", "draw"),
    "sample": (".decode", "sample"),
    "SettlePolicy": (".decode", "SettlePolicy"),
    "Step": (".decode", "Step"),
    "StepInfo": (".decode", "StepInfo"),
    "DEFAULT_MASK_CACHE_SIZE": (".decode", "DEFAULT_MASK_CACHE_SIZE"),
    "Runtime": (".runtime", "Runtime"),
    # The engine, re-exported so a caller needs one import.
    "Synthesizer": ("aufbau", "Synthesizer"),
    # The transformers backend. Importing any of these pulls in torch.
    "ConstrainedModel": (".llm", "ConstrainedModel"),
    "get_model_class": (".models", "get_model_class"),
    "ChatConstrainedModel": (".models", "ChatConstrainedModel"),
    "DeepseekConstrainedModel": (".models", "DeepseekConstrainedModel"),
    "LlamaConstrainedModel": (".models", "LlamaConstrainedModel"),
    "MistralConstrainedModel": (".models", "MistralConstrainedModel"),
    "GlmConstrainedModel": (".models", "GlmConstrainedModel"),
    "PleiasConstrainedModel": (".models", "PleiasConstrainedModel"),
    # The bundled grammars.
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
    # Deliberately NOT cached into globals(). Caching would freeze the first
    # value resolved, so `monkeypatch.setattr(proposition7.api, "generate", ...)`
    # would stop being visible through `proposition7.generate` -- the package
    # root and the submodule would disagree about what the name means. Resolving
    # every time keeps them the same name. `import_module` is a sys.modules hit
    # and no caller holds this in a hot loop; they bind it once at their import.
    return getattr(module, attribute)


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = sorted(_LAZY_EXPORTS)

try:
    __version__: str = _pkg_version("proposition7")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"
