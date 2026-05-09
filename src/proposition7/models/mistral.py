from __future__ import annotations

from typing import List, Optional

from ..llm import ConstrainedModel


class MistralConstrainedModel(ConstrainedModel):
    def stop_tokens_unconstrained(self, grammar_name: Optional[str] = None) -> List[str]:
        extra = ["</s>", "<s>"]
        return self._dedupe_tokens(super().stop_tokens_unconstrained(grammar_name) + extra)

    def stop_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        extra = ["</s>", "<s>"]
        return self._dedupe_tokens(super().stop_tokens_constrained(grammar_name) + extra)
