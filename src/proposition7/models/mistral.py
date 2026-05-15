from __future__ import annotations

from typing import List

from ..llm import ConstrainedModel


class MistralConstrainedModel(ConstrainedModel):
    def stop_tokens_unconstrained(self) -> List[str]:
        extra = ["</s>", "<s>"]
        return self._dedupe_tokens(super().stop_tokens_unconstrained() + extra)

    def stop_tokens_constrained(self) -> List[str]:
        extra = ["</s>", "<s>"]
        return self._dedupe_tokens(super().stop_tokens_constrained() + extra)
