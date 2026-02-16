from __future__ import annotations

from typing import Any, Dict, Optional, List

from .chat import ChatConstrainedModel


class DeepseekConstrainedModel(ChatConstrainedModel):
    @classmethod
    def tokenizer_kwargs(cls) -> Dict[str, Any]:
        return {"trust_remote_code": True, "use_fast": False}

    @classmethod
    def model_kwargs(cls) -> Dict[str, Any]:
        return {"trust_remote_code": True}

    def allow_system_prompt(self) -> bool:
        return False

    def think_open(self) -> str:
        return "<think>\n"

    def think_close(self) -> str:
        return "</think>"

    def start_tokens_unconstrained(self, grammar_name: Optional[str] = None) -> List[str]:
        return [self.think_open()]

    def start_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        return []
