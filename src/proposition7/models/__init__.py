from __future__ import annotations

from importlib import import_module
from typing import Any


MODEL_CATALOG = [
    {"name": "gpt2", "display_name": "GPT-2 (124M)"},
    {"name": "gpt2-medium", "display_name": "GPT-2 Medium (355M)"},
    {"name": "EleutherAI/pythia-160m", "display_name": "Pythia-160M"},
    {"name": "EleutherAI/pythia-410m", "display_name": "Pythia-410M"},
    {"name": "EleutherAI/pythia-1.4b", "display_name": "Pythia-1.4B"},
    {"name": "EleutherAI/pythia-2.8b", "display_name": "Pythia-2.8B"},
    {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "display_name": "DeepSeek R1 Distill 1.5B"},
    {"name": "Qwen/Qwen3.5-9B", "display_name": "Qwen 3.5 9B"},
    {"name": "EleutherAI/pythia-6.9b", "display_name": "Pythia-6.9B"},
    {"name": "mistralai/Mistral-7B-v0.1", "display_name": "Mistral-7B"},
    {"name": "meta-llama/Meta-Llama-3.1-8B-Instruct", "display_name": "Llama-3.1-8B-Instruct"},
    {"name": "THUDM/glm-4-9b", "display_name": "GLM-4-9B"},
    {"name": "PleIAs/Monad", "display_name": "PleIAs Monad (56.7M)"},
    {"name": "PleIAs/Baguettotron", "display_name": "PleIAs Baguettotron (321M)"},
]


_LAZY_EXPORTS = {
    "ConstrainedModel": ("..llm", "ConstrainedModel"),
    "ChatConstrainedModel": (".chat", "ChatConstrainedModel"),
    "DeepseekConstrainedModel": (".deepseek", "DeepseekConstrainedModel"),
    "GlmConstrainedModel": (".glm", "GlmConstrainedModel"),
    "GptOssConstrainedModel": (".gptoss", "GptOssConstrainedModel"),
    "LlamaConstrainedModel": (".llama", "LlamaConstrainedModel"),
    "MistralConstrainedModel": (".mistral", "MistralConstrainedModel"),
    "PleiasConstrainedModel": (".pleias", "PleiasConstrainedModel"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def list_models() -> list[dict[str, str]]:
    return list(MODEL_CATALOG)


def get_model_class(model_name: str) -> type[Any]:
    lower_name = model_name.lower()
    if "pleias" in lower_name or "baguettotron" in lower_name or "monad" in lower_name:
        return PleiasConstrainedModel
    if "deepseek" in lower_name:
        return DeepseekConstrainedModel
    if "gpt-oss" in lower_name or "gpt_oss" in lower_name:
        return GptOssConstrainedModel
    if "qwen" in lower_name and "base" not in lower_name:
        return ChatConstrainedModel
    if "glm" in lower_name:
        return GlmConstrainedModel
    if "llama" in lower_name:
        return LlamaConstrainedModel
    if "mistral" in lower_name:
        return MistralConstrainedModel
    return ConstrainedModel


__all__ = [
    "ConstrainedModel", "ChatConstrainedModel", "DeepseekConstrainedModel",
    "GlmConstrainedModel", "GptOssConstrainedModel", "LlamaConstrainedModel",
    "MistralConstrainedModel", "PleiasConstrainedModel", "get_model_class",
    "list_models",
]
