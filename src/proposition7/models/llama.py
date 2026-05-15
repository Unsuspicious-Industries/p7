from __future__ import annotations

from typing import List

from .chat import ChatConstrainedModel


class LlamaConstrainedModel(ChatConstrainedModel):
    def stop_tokens_unconstrained(self) -> List[str]:
        extra = [
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
        ]
        return self._dedupe_tokens(super().stop_tokens_unconstrained() + extra)

    def stop_tokens_constrained(self) -> List[str]:
        extra = [
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
        ]
        return self._dedupe_tokens(super().stop_tokens_constrained() + extra)
