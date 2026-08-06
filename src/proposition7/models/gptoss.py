"""OpenAI gpt-oss adapter (harmony response format)."""

from __future__ import annotations

import dataclasses
from typing import List, Optional

from ..inference import GenerationResult
from .chat import ChatConstrainedModel

_FINAL_MARKER = "<|channel|>final<|message|>"
_END_MARKERS = ("<|return|>", "<|end|>", "<|call|>")


def extract_harmony_final(text: str) -> str:
    """The final-channel content of a harmony-format completion.

    gpt-oss models emit channelled output even when decoding is
    unconstrained: ``<|channel|>analysis<|message|>...<|channel|>final
    <|message|>...<|return|>``. The benchmark grades the answer itself, so
    keep only what follows the LAST final-channel marker and cut it at the
    first end-of-message control token. Text without the marker is returned
    unchanged."""
    marker_at = text.rfind(_FINAL_MARKER)
    if marker_at == -1:
        return text
    content = text[marker_at + len(_FINAL_MARKER) :]
    cut = min(
        (at for at in (content.find(marker) for marker in _END_MARKERS) if at != -1),
        default=len(content),
    )
    return content[:cut]


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
