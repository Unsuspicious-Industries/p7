from __future__ import annotations

from typing import Optional

from ..llm import ConstrainedModel


class ChatConstrainedModel(ConstrainedModel):
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
                return prompt_text
        return prompt_text
