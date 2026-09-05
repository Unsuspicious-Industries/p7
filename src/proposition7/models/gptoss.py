"""OpenAI gpt-oss adapter (harmony response format)."""

from __future__ import annotations

import dataclasses
from typing import List, Optional

from ..inference import GenerationResult
from .chat import ChatConstrainedModel
from .harmony import extract_harmony_final


class GptOssConstrainedModel(ChatConstrainedModel):
    """gpt-oss speaks the harmony chat format: prompts go through the
    tokenizer's chat template (inherited from ChatConstrainedModel) and
    unconstrained completions carry channel markup, so their final channel
    is post-extracted before any parsing. Constrained generation stays as
    the base class does it -- the grammar mask already pins the output."""

    def stop_tokens_unconstrained(
        self, grammar_name: Optional[str] = None
    ) -> List[str]:
        extra = ["<|return|>", "<|call|>"]
        return self._dedupe_tokens(
            super().stop_tokens_unconstrained(grammar_name) + extra
        )

    def stop_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        extra = ["<|return|>", "<|call|>"]
        return self._dedupe_tokens(
            super().stop_tokens_constrained(grammar_name) + extra
        )

    def generate_unconstrained(self, *args, **kwargs) -> GenerationResult:
        result = super().generate_unconstrained(*args, **kwargs)
        extracted = extract_harmony_final(result.text)
        if extracted != result.text:
            result = dataclasses.replace(result, text=extracted)
        return result
