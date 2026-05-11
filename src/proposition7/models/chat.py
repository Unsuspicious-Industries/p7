from __future__ import annotations

import re
from typing import List, Optional

from ..llm import ConstrainedModel


class ChatConstrainedModel(ConstrainedModel):
    def format_prompt(self, prompt_text: str) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt_text
        messages = [{"role": "user", "content": prompt_text}]
        # enable_thinking=False produces a pre-closed empty think block
        # (<think>\n\n</think>\n\n) so non-thinking generations start in the
        # correct context — past the think block, not inside it.  Fall back to
        # the default template for tokenizers that don't support the kwarg.
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs
                )
            except Exception:
                continue
        return prompt_text

    def _set_prompt(self, prompt: str, initial: str, start_tokens: List[str]) -> None:
        import torch
        formatted = self.format_prompt(prompt + "".join(start_tokens))
        # When initial opens a think block (think phase), strip the pre-closed
        # empty block that enable_thinking=False injects (<think>\n\n</think>\n\n).
        # Without stripping, the model would see two consecutive think tags and
        # immediately close the second one (think_tokens=1).
        think_open = self.think_open()
        think_close = self.think_close()
        if initial.lstrip().startswith(think_open):
            pattern = re.compile(
                re.escape(think_open) + r"\s*" + re.escape(think_close) + r"\s*$"
            )
            formatted = pattern.sub("", formatted)
        encoded = self.tokenizer(formatted + initial, return_tensors="pt")
        self._input_ids = encoded.input_ids.to(self.device)
        self._past_key_values = None
        self._pending_input_ids = self._input_ids
