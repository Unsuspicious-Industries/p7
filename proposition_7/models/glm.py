from __future__ import annotations

from typing import Any, Dict, Optional, List

from .chat import ChatConstrainedModel


class GlmConstrainedModel(ChatConstrainedModel):
    @classmethod
    def tokenizer_kwargs(cls) -> Dict[str, Any]:
        return {"trust_remote_code": True, "use_fast": False}

    @classmethod
    def model_kwargs(cls) -> Dict[str, Any]:
        return {"trust_remote_code": True}

    def start_tokens_unconstrained(self, grammar_name: Optional[str] = None) -> List[str]:
        return ["<sop>"]

    def start_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        return ["<sop>"]

    def format_prompt(self, prompt_text: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return super().format_prompt(prompt_text)
        return f"<|user|>\n{prompt_text}\n<|assistant|>\n"
