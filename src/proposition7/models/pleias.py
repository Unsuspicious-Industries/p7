"""PleIAs adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .chat import ChatConstrainedModel


class PleiasConstrainedModel(ChatConstrainedModel):
    @classmethod
    def tokenizer_kwargs(cls) -> Dict[str, Any]:
        return {"use_fast": False}

    @classmethod
    def model_kwargs(cls) -> Dict[str, Any]:
        return {}

    def think_open(self) -> str:
        return "<think>\n"

    def think_close(self) -> str:
        return "</think>"

    def start_tokens_unconstrained(
        self, grammar_name: Optional[str] = None
    ) -> List[str]:
        return [self.think_open()]

    def start_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        return []

    def stop_tokens_unconstrained(
        self, grammar_name: Optional[str] = None
    ) -> List[str]:
        extra = [
            "<|end_of_text|>",
            "<|begin_of_text|>",
            "<|im_end>",
            "<|im_end|>",
            "<|im_start|>",
        ]
        return self._dedupe_tokens(
            super().stop_tokens_unconstrained(grammar_name) + extra
        )

    def stop_tokens_constrained(self, grammar_name: Optional[str] = None) -> List[str]:
        extra = [
            "<|end_of_text|>",
            "<|begin_of_text|>",
            "<|im_end>",
            "<|im_end|>",
            "<|im_start|>",
        ]
        return self._dedupe_tokens(
            super().stop_tokens_constrained(grammar_name) + extra
        )

    def format_prompt(self, prompt_text: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        return f"<|im_start|>user\n{prompt_text}<|im_end>\n<|im_start|>assistant\n"
