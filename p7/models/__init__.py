from __future__ import annotations

from typing import Dict, List, Type

from ..llm import ConstrainedModel
from .chat import ChatConstrainedModel
from .deepseek import DeepseekConstrainedModel
from .glm import GlmConstrainedModel
from .llama import LlamaConstrainedModel
from .mistral import MistralConstrainedModel


MODEL_CATALOG: List[Dict[str, str]] = [
    {"name": "gpt2", "display_name": "GPT-2 (124M)"},
    {"name": "gpt2-medium", "display_name": "GPT-2 Medium (355M)"},
    {"name": "EleutherAI/pythia-160m", "display_name": "Pythia-160M"},
    {"name": "EleutherAI/pythia-410m", "display_name": "Pythia-410M"},
    {"name": "EleutherAI/pythia-1.4b", "display_name": "Pythia-1.4B"},
    {"name": "EleutherAI/pythia-2.8b", "display_name": "Pythia-2.8B"},
    {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "display_name": "DeepSeek R1 Distill 1.5B"},
    {"name": "EleutherAI/pythia-6.9b", "display_name": "Pythia-6.9B"},
    {"name": "mistralai/Mistral-7B-v0.1", "display_name": "Mistral-7B"},
    {"name": "meta-llama/Meta-Llama-3.1-8B-Instruct", "display_name": "Llama-3.1-8B-Instruct"},
    {"name": "THUDM/glm-4-9b", "display_name": "GLM-4-9B"},
]


def list_models() -> List[Dict[str, str]]:
    return list(MODEL_CATALOG)


def get_model_class(model_name: str) -> Type[ConstrainedModel]:
    lower_name = model_name.lower()
    if "deepseek" in lower_name:
        return DeepseekConstrainedModel
    if "glm" in lower_name:
        return GlmConstrainedModel
    if "llama" in lower_name:
        return LlamaConstrainedModel
    if "mistral" in lower_name:
        return MistralConstrainedModel
    return ConstrainedModel


__all__ = [
    "ConstrainedModel",
    "ChatConstrainedModel",
    "DeepseekConstrainedModel",
    "GlmConstrainedModel",
    "LlamaConstrainedModel",
    "MistralConstrainedModel",
    "get_model_class",
    "list_models",
]
